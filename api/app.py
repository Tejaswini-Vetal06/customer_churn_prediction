"""
FastAPI Production Microservice for Customer Churn Prediction.
Provides endpoints for single and batch predictions, healthchecks, and model diagnostics.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
import json
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.config import MODEL_PATH, METRICS_PATH, PLOTS_DIR, ARTIFACTS_DIR
from src.predict import CustomerChurnPredictor
from api.schemas import (
    CustomerFeatures,
    SinglePredictionRequest,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchCustomerResult,
    ModelMetadataResponse,
)

predictor: CustomerChurnPredictor = None
metadata_cache: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model pipeline and metadata on startup."""
    global predictor, metadata_cache
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Trained model artifact not found at {MODEL_PATH}")

    predictor = CustomerChurnPredictor(model_path=MODEL_PATH)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metadata_cache = json.load(f)

    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready machine learning API for predicting telecom customer churn, risk tiers, and interventions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Healthcheck endpoint verifying model loading status."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Diagnostics"])
async def get_model_metadata():
    """Retrieve test set generalization metrics and model identification."""
    if not metadata_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics metadata not found.",
        )
    test_metrics = metadata_cache.get("test_metrics", {})
    return ModelMetadataResponse(
        model_name=metadata_cache.get("best_model_name", "HistGradientBoostingClassifier"),
        status="active",
        test_roc_auc=test_metrics.get("test_roc_auc", 0.0),
        test_pr_auc=test_metrics.get("test_pr_auc", 0.0),
        test_recall=test_metrics.get("test_recall", 0.0),
        test_precision=test_metrics.get("test_precision", 0.0),
    )


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    tags=["Inference"],
    summary="Predict churn probability for a single customer",
)
async def predict_single_customer(request: SinglePredictionRequest):
    """
    Receives customer demographic, contractual, and usage attributes.
    Returns churn prediction, calibrated probability, risk tier, and retention action.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized.",
        )

    customer_dict = request.customer.model_dump()
    result = predictor.predict_single(
        customer_features=customer_dict,
        custom_threshold=request.custom_threshold,
    )
    return SinglePredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Predict churn for a batch of customers",
)
async def predict_batch_customers(request: BatchPredictionRequest):
    """
    Receives an array of customer records.
    Returns individual risk evaluations plus aggregate churn rate.
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not initialized.",
        )

    records = [c.model_dump() for c in request.customers]
    import pandas as pd

    df = pd.DataFrame(records)
    batch_results_df = predictor.predict_batch(df, custom_threshold=request.custom_threshold)

    results_list = []
    for idx, row in batch_results_df.iterrows():
        results_list.append(
            BatchCustomerResult(
                customer_index=int(idx),
                churn_prediction=row["churn_prediction"],
                churn_probability=float(row["churn_probability"]),
                risk_tier=row["risk_tier"],
            )
        )

    total_count = len(results_list)
    churn_count = sum(1 for r in results_list if r.churn_prediction == "Yes")
    churn_rate = round(churn_count / total_count, 4) if total_count > 0 else 0.0

    return BatchPredictionResponse(
        total_customers=total_count,
        predicted_churn_count=churn_count,
        predicted_churn_rate=churn_rate,
        results=results_list,
    )


@app.get("/model/plots/{plot_name}", tags=["Diagnostics"])
async def get_diagnostic_plot(plot_name: str):
    """Fetch diagnostic plot image (e.g. roc_curve.png, confusion_matrix.png)."""
    valid_plots = [
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "calibration_curve.png",
        "cost_threshold_analysis.png",
        "feature_importance.png",
    ]
    if plot_name not in valid_plots:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot not found. Must be one of: {valid_plots}",
        )

    plot_file = PLOTS_DIR / plot_name
    if not plot_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plot file {plot_name} does not exist yet. Run src.evaluate first.",
        )
    return FileResponse(plot_file, media_type="image/png")

