"""
Configuration module for the Customer Churn Prediction project.
Contains file paths, feature groups, hyperparameters, and business cost matrices.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODEL_PATH = ARTIFACTS_DIR / "churn_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Ensure runtime directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Target Variable
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

# Feature Column Definitions
NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

MULTI_CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

ALL_CATEGORICAL_FEATURES = BINARY_CATEGORICAL_FEATURES + MULTI_CATEGORICAL_FEATURES

# Cross-Validation Configuration
CV_SPLITS = 5
TEST_SIZE = 0.20

# Business Cost Matrix (for cost-sensitive threshold tuning)
# Assumption:
# - False Negative (missed churner): lost customer lifetime revenue
# - False Positive (unnecessary intervention): customer retention incentive cost
COST_FALSE_NEGATIVE = 500.0  # Estimated churn cost in dollars
COST_FALSE_POSITIVE = 100.0  # Cost of promotional incentive

