"""
Inference Module for Single and Batch Customer Churn Prediction.
Loads serialized pipeline and outputs risk tiers, probabilities, and recommendations.
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import joblib

from src.config import MODEL_PATH, TARGET_COLUMN


class CustomerChurnPredictor:
    """Production Inference Engine for Customer Churn."""

    def __init__(self, model_path: Path = MODEL_PATH, decision_threshold: float = 0.50):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run src.train first."
            )
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.decision_threshold = decision_threshold

    @staticmethod
    def _assign_risk_tier(probability: float) -> str:
        if probability < 0.30:
            return "Low Risk"
        elif probability < 0.65:
            return "Medium Risk"
        else:
            return "High Risk"

    @staticmethod
    def _generate_recommendation(risk_tier: str, probability: float, input_data: Dict[str, Any]) -> str:
        if risk_tier == "High Risk":
            contract = input_data.get("Contract", "")
            if contract == "Month-to-month":
                return "Urgent: Offer 1-year discounted contract incentive with dedicated technical onboarding support."
            return "High churn probability: Trigger account manager outreach and satisfaction review."
        elif risk_tier == "Medium Risk":
            return "Monitor usage patterns; offer loyalty bundle on digital security or backup add-ons."
        else:
            return "Healthy customer relationship; suitable candidate for service upselling."

    def predict_single(
        self, customer_features: Dict[str, Any], custom_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Predict churn probability and classification for a single customer payload.
        """
        df = pd.DataFrame([customer_features])
        proba = float(self.pipeline.predict_proba(df)[0, 1])

        threshold = custom_threshold if custom_threshold is not None else self.decision_threshold
        churn_pred = int(proba >= threshold)
        churn_label = "Yes" if churn_pred == 1 else "No"
        risk_tier = self._assign_risk_tier(proba)
        recommendation = self._generate_recommendation(risk_tier, proba, customer_features)

        return {
            "churn_prediction": churn_label,
            "churn_probability": round(proba, 4),
            "risk_tier": risk_tier,
            "threshold_used": threshold,
            "recommendation": recommendation,
        }

    def predict_batch(
        self, df: pd.DataFrame, custom_threshold: float = None
    ) -> pd.DataFrame:
        """
        Vectorized batch prediction for a DataFrame of customers.
        """
        df_in = df.copy()
        if "customerID" in df_in.columns:
            customer_ids = df_in["customerID"]
        else:
            customer_ids = None

        probas = self.pipeline.predict_proba(df_in)[:, 1]
        threshold = custom_threshold if custom_threshold is not None else self.decision_threshold
        preds = (probas >= threshold).astype(int)

        results = pd.DataFrame(
            {
                "churn_probability": np.round(probas, 4),
                "churn_prediction": np.where(preds == 1, "Yes", "No"),
                "risk_tier": [self._assign_risk_tier(p) for p in probas],
            }
        )

        if customer_ids is not None:
            results.insert(0, "customerID", customer_ids.values)

        return results

