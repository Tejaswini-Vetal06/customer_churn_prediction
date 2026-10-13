"""
Feature Engineering & Preprocessing Pipeline Module.
Implements leak-free feature transformations and Scikit-Learn ColumnTransformer.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMERICAL_FEATURES, ALL_CATEGORICAL_FEATURES


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer for Telco customer churn.
    Extracts high-signal behavioural and contractual features:
    - tenure_group: categorical binning of tenure
    - monthly_to_total_ratio: ratio of monthly fee to cumulative spend
    - num_services: count of active digital add-on services
    - is_month_to_month: binary indicator of flexible/uncommitted contract
    - has_fiber_optic: flag for high-churn fiber tier
    - has_tech_support: flag for tech support presence
    - auto_payment: flag for automatic billing methods
    """

    SERVICE_COLUMNS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. Tenure Grouping
        if "tenure" in X_out.columns:
            tenure_numeric = pd.to_numeric(X_out["tenure"], errors="coerce").fillna(0)
            bins = [-1, 12, 24, 48, 72, np.inf]
            labels = ["0-12m", "13-24m", "25-48m", "49-72m", "72m+"]
            X_out["tenure_group"] = pd.cut(
                tenure_numeric,
                bins=bins,
                labels=labels,
            ).astype(str)

        # 2. Monthly to Total Charges Ratio
        if "MonthlyCharges" in X_out.columns and "TotalCharges" in X_out.columns:
            monthly = pd.to_numeric(X_out["MonthlyCharges"], errors="coerce").fillna(0)
            total = pd.to_numeric(X_out["TotalCharges"], errors="coerce").fillna(0)
            X_out["monthly_to_total_ratio"] = monthly / (total + 1.0)

        # 3. Number of Subscribed Services (Switching Cost proxy)
        existing_service_cols = [c for c in self.SERVICE_COLUMNS if c in X_out.columns]
        if existing_service_cols:
            X_out["num_services"] = (X_out[existing_service_cols] == "Yes").sum(axis=1)

        # 4. Month-to-Month Contract indicator
        if "Contract" in X_out.columns:
            X_out["is_month_to_month"] = (
                X_out["Contract"] == "Month-to-month"
            ).astype(int)

        # 5. Fiber Optic indicator
        if "InternetService" in X_out.columns:
            X_out["has_fiber_optic"] = (
                X_out["InternetService"] == "Fiber optic"
            ).astype(int)

        # 6. Tech Support indicator
        if "TechSupport" in X_out.columns:
            X_out["has_tech_support"] = (X_out["TechSupport"] == "Yes").astype(int)

        # 7. Auto Payment Method (Direct Debit / Credit Card vs Paper Check)
        if "PaymentMethod" in X_out.columns:
            X_out["auto_payment"] = (
                X_out["PaymentMethod"].isin(
                    ["Bank transfer (automatic)", "Credit card (automatic)"]
                )
            ).astype(int)

        return X_out


def build_preprocessor() -> Pipeline:
    """
    Constructs an end-to-end ColumnTransformer preprocessor:
    - Numerical features: Median Imputation -> Standard Scaling
    - Categorical features: Most Frequent Imputation -> One-Hot Encoding
    Wrapped with custom ChurnFeatureEngineer.
    """
    # Numerical features (original + engineered numeric)
    num_cols = NUMERICAL_FEATURES + [
        "monthly_to_total_ratio",
        "num_services",
        "is_month_to_month",
        "has_fiber_optic",
        "has_tech_support",
        "auto_payment",
    ]

    # Categorical features (original + tenure_group)
    cat_cols = ALL_CATEGORICAL_FEATURES + ["tenure_group"]

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", ChurnFeatureEngineer()),
            ("column_preprocessor", column_preprocessor),
        ]
    )

    return pipeline

