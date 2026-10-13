"""
Data Loader & Cleaning Module.
Responsible for ingestion, data sanity checks, and strict leakage-free splitting.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    ID_COLUMN,
    TARGET_COLUMN,
    POSITIVE_CLASS,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERICAL_FEATURES,
    ALL_CATEGORICAL_FEATURES,
)


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from disk."""
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and sanitize raw dataset:
    1. Parse 'TotalCharges' into numeric float; empty strings (' ') become NaN.
    2. Impute TotalCharges for 0-tenure customers as 0.0 (they haven't completed a month).
    3. Ensure SeniorCitizen is treated as an object/string category.
    4. Validate target column existence.
    """
    df = df.copy()

    # TotalCharges contains blank strings ' ' for customers with tenure == 0
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Customers with tenure == 0 haven't had a billing cycle, set total to 0.0
        df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0.0)

    # Cast SeniorCitizen to string so it's treated as categorical consistently
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


def get_feature_and_target_matrices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y), dropping ID column."""
    cleaned = clean_churn_data(df)

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataframe.")

    # Encode target to binary (1 for Churned, 0 for Retained)
    y = (cleaned[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)

    # Drop ID and Target from feature matrix
    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in cleaned.columns]
    X = cleaned.drop(columns=drop_cols)

    return X, y


def split_and_save_data(
    data_path: Path = RAW_DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw data, performs stratified train/test split BEFORE any transformation,
    and saves training/testing sets to data/processed for reproducibility.
    """
    raw_df = load_raw_data(data_path)
    X, y = get_feature_and_target_matrices(raw_df)

    # Stratified split to maintain class ratio (73.5% Retained / 26.5% Churned)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Save processed splits
    X_train.to_parquet(PROCESSED_DATA_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(PROCESSED_DATA_DIR / "X_test.parquet", index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_parquet(PROCESSED_DATA_DIR / "y_train.parquet", index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_parquet(PROCESSED_DATA_DIR / "y_test.parquet", index=False)

    return X_train, X_test, y_train, y_test


def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes a schema and quality health check of the dataset."""
    cleaned = clean_churn_data(df)
    total_rows = len(cleaned)
    missing_counts = cleaned.isnull().sum()
    missing_summary = {col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0}

    churn_rate = None
    if TARGET_COLUMN in cleaned.columns:
        churn_rate = float((cleaned[TARGET_COLUMN] == POSITIVE_CLASS).mean())

    return {
        "total_records": total_rows,
        "total_features": len(cleaned.columns),
        "missing_values": missing_summary,
        "churn_rate": churn_rate,
        "numerical_columns": NUMERICAL_FEATURES,
        "categorical_columns": ALL_CATEGORICAL_FEATURES,
    }


if __name__ == "__main__":
    print("Testing data ingestion and stratified split...")
    X_train, X_test, y_train, y_test = split_and_save_data()
    print(f"X_train shape: {X_train.shape}, y_train mean (churn rate): {y_train.mean():.4f}")
    print(f"X_test shape:  {X_test.shape},  y_test mean (churn rate):  {y_test.mean():.4f}")
    print("Data splits saved successfully to:", PROCESSED_DATA_DIR)

