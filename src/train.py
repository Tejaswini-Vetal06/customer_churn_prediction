"""
Model Training, Cross-Validation, Comparison, and Hyperparameter Tuning Module.
Follows rigorous ML best practices:
- Majority-class and linear baselines
- 5-Fold Stratified Cross Validation
- Multi-metric evaluation (ROC-AUC, PR-AUC, F1, Recall, Precision, Brier score)
- Hyperparameter tuning via RandomizedSearchCV
- Test set generalization evaluation
- Artifact persistence
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    brier_score_loss,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    CV_SPLITS,
)
from src.preprocessor import build_preprocessor


def load_processed_splits() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load train/test parquet files."""
    X_train = pd.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet").iloc[:, 0].to_numpy()
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet").iloc[:, 0].to_numpy()
    return X_train, X_test, y_train, y_test


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns baseline and candidate model estimators.
    Uses class_weight='balanced' where available to mitigate churn class imbalance.
    """
    return {
        "Dummy (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_models_cross_validation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_splits: int = CV_SPLITS,
) -> pd.DataFrame:
    """
    Executes 5-fold Stratified Cross-Validation across candidate models.
    Collects mean and standard deviation for key performance metrics.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }

    results = []
    models = get_candidate_models()

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", model),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        res_dict = {
            "Model": name,
            "ROC-AUC (Mean)": np.mean(scores["test_roc_auc"]),
            "ROC-AUC (Std)": np.std(scores["test_roc_auc"]),
            "PR-AUC (Mean)": np.mean(scores["test_pr_auc"]),
            "PR-AUC (Std)": np.std(scores["test_pr_auc"]),
            "F1 (Mean)": np.mean(scores["test_f1"]),
            "Recall (Mean)": np.mean(scores["test_recall"]),
            "Precision (Mean)": np.mean(scores["test_precision"]),
            "Accuracy (Mean)": np.mean(scores["test_accuracy"]),
        }
        results.append(res_dict)

    df_results = pd.DataFrame(results).sort_values(by="ROC-AUC (Mean)", ascending=False)
    return df_results


def tune_champion_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Performs RandomizedSearchCV on HistGradientBoostingClassifier
    (the top performing ensemble gradient booster).
    Optimizes for PR-AUC (Average Precision) given class imbalance.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_distributions = {
        "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "classifier__max_iter": [100, 150, 200, 250],
        "classifier__max_leaf_nodes": [15, 25, 31, 45, 63],
        "classifier__min_samples_leaf": [15, 20, 30, 50],
        "classifier__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        cv=cv,
        scoring="average_precision",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    print("\n=== Hyperparameter Tuning Complete ===")
    print(f"Best CV PR-AUC Score: {search.best_score_:.4f}")
    print("Best Hyperparameters:")
    for param, val in search.best_params_.items():
        print(f"  {param}: {val}")

    return search.best_estimator_


def evaluate_on_test_set(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute final evaluation metrics on held-out test data."""
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "test_pr_auc": float(average_precision_score(y_test, y_pred_proba)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_brier_score": float(brier_score_loss(y_test, y_pred_proba)),
    }
    return metrics


def run_training_pipeline() -> Tuple[Pipeline, pd.DataFrame, Dict[str, Any]]:
    """Complete training orchestrator."""
    print("Loading data splits...")
    X_train, X_test, y_train, y_test = load_processed_splits()

    print("\nRunning 5-Fold Stratified Cross-Validation across candidate models...")
    comparison_df = evaluate_models_cross_validation(X_train, y_train)
    print("\n=== Cross-Validation Model Comparison ===")
    print(
        comparison_df[
            ["Model", "ROC-AUC (Mean)", "PR-AUC (Mean)", "F1 (Mean)", "Recall (Mean)", "Accuracy (Mean)"]
        ].to_string(index=False)
    )

    print("\nTuning champion model hyperparameters...")
    best_pipeline = tune_champion_model(X_train, y_train)

    print("\nFitting champion pipeline on entire training set...")
    best_pipeline.fit(X_train, y_train)

    print("\nEvaluating champion pipeline on held-out test set...")
    test_metrics = evaluate_on_test_set(best_pipeline, X_test, y_test)
    print("\n=== Final Test Set Metrics ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save artifacts
    print(f"\nPersisting best pipeline to: {MODEL_PATH}")
    joblib.dump(best_pipeline, MODEL_PATH)

    # Save metrics report
    metrics_payload = {
        "test_metrics": test_metrics,
        "cv_comparison": comparison_df.to_dict(orient="records"),
        "best_model_name": "HistGradientBoostingClassifier (Tuned)",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    return best_pipeline, comparison_df, metrics_payload


if __name__ == "__main__":
    run_training_pipeline()

