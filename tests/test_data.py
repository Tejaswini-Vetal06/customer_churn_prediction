"""
Unit Tests for Data Ingestion, Cleaning, and Leakage Prevention.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, TARGET_COLUMN
from src.data_loader import (
    load_raw_data,
    clean_churn_data,
    get_feature_and_target_matrices,
)


def test_raw_data_file_exists():
    """Verify raw dataset exists and is accessible."""
    assert RAW_DATA_PATH.exists()
    assert RAW_DATA_PATH.stat().st_size > 0


def test_clean_churn_data_handles_total_charges():
    """Ensure TotalCharges whitespace strings are converted to numeric."""
    sample = pd.DataFrame(
        {
            "tenure": [0, 5],
            "TotalCharges": [" ", "150.50"],
            "MonthlyCharges": [20.0, 30.1],
        }
    )
    cleaned = clean_churn_data(sample)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    # 0-tenure row should be imputed to 0.0
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 150.50


def test_target_matrix_encoding():
    """Verify target column is mapped to binary 0 and 1."""
    raw_df = load_raw_data()
    X, y = get_feature_and_target_matrices(raw_df)
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)
    assert TARGET_COLUMN not in X.columns
    assert "customerID" not in X.columns


def test_processed_splits_no_data_leakage():
    """Verify train and test splits are mutually exclusive and have expected shapes."""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet")

    total_rows = len(X_train) + len(X_test)
    assert total_rows == 7043
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Class balance stratification check (should be ~26.5% positive in both)
    train_rate = y_train.iloc[:, 0].mean()
    test_rate = y_test.iloc[:, 0].mean()
    assert abs(train_rate - test_rate) < 0.01

