"""
Unit and Integration Tests for Preprocessing and Inference Pipelines.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.config import MODEL_PATH, PROCESSED_DATA_DIR
from src.preprocessor import ChurnFeatureEngineer, build_preprocessor
from src.predict import CustomerChurnPredictor


@pytest.fixture
def sample_customer_df():
    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.50,
                "TotalCharges": 1026.0,
            }
        ]
    )


def test_churn_feature_engineer(sample_customer_df):
    """Test engineered features generation."""
    fe = ChurnFeatureEngineer()
    transformed = fe.transform(sample_customer_df)

    assert "tenure_group" in transformed.columns
    assert transformed.loc[0, "tenure_group"] == "0-12m"
    assert "monthly_to_total_ratio" in transformed.columns
    assert "num_services" in transformed.columns
    assert transformed.loc[0, "num_services"] == 3  # Backup, TV, Movies
    assert "is_month_to_month" in transformed.columns
    assert transformed.loc[0, "is_month_to_month"] == 1


def test_pipeline_transformation_shape(sample_customer_df):
    """Verify ColumnTransformer outputs numeric matrix without NaNs."""
    pipe = build_preprocessor()
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    pipe.fit(X_train)

    X_transformed = pipe.transform(sample_customer_df)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 1
    assert not np.isnan(X_transformed).any()


def test_predictor_single_and_batch(sample_customer_df):
    """Verify CustomerChurnPredictor returns valid probabilities and tiers."""
    predictor = CustomerChurnPredictor(model_path=MODEL_PATH)

    # Test Single
    single_res = predictor.predict_single(sample_customer_df.iloc[0].to_dict())
    assert "churn_prediction" in single_res
    assert single_res["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= single_res["churn_probability"] <= 1.0
    assert single_res["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(single_res["recommendation"]) > 0

    # Test Batch
    batch_res = predictor.predict_batch(sample_customer_df)
    assert len(batch_res) == 1
    assert "churn_probability" in batch_res.columns
    assert "risk_tier" in batch_res.columns

