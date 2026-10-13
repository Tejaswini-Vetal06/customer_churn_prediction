"""
Model Evaluation, Diagnostic Plotting, and Cost-Sensitive Threshold Analysis Module.
Generates:
- Confusion matrices (counts & normalized)
- ROC and Precision-Recall curves
- Calibration curve (reliability diagram)
- Business cost vs threshold curve (cost-sensitive analysis)
- Permutation feature importance
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    f1_score,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    PLOTS_DIR,
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    RANDOM_STATE,
)


def load_eval_data_and_model():
    """Loads held-out test data and trained pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")

    X_test = pd.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet").iloc[:, 0].to_numpy()
    pipeline = joblib.load(MODEL_PATH)
    return X_test, y_test, pipeline


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path):
    """Plot and save annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = np.array(
        [
            [f"{cm[0,0]}\n({cm_norm[0,0]:.1%})", f"{cm[0,1]}\n({cm_norm[0,1]:.1%})"],
            [f"{cm[1,0]}\n({cm_norm[1,0]:.1%})", f"{cm[1,1]}\n({cm_norm[1,1]:.1%})"],
        ]
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
    )
    plt.title("Confusion Matrix (Counts & % by True Class)", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_and_pr_curves(
    y_true: np.ndarray, y_proba: np.ndarray, roc_path: Path, pr_path: Path
) -> Tuple[float, float]:
    """Plot ROC Curve and Precision-Recall Curve."""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline_churn_rate = float(np.mean(y_true))

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#2ca02c", lw=2, label=f"PR Curve (AP = {pr_auc:.3f})")
    plt.axhline(
        y=baseline_churn_rate,
        color="grey",
        linestyle="--",
        label=f"Baseline Churn Rate ({baseline_churn_rate:.1%})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

    return roc_auc, pr_auc


def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: Path) -> float:
    """Plot calibration reliability diagram and calculate Brier Score."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "s-", color="#9467bd", label=f"Model (Brier = {brier:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Positives", fontsize=11)
    plt.title("Probability Calibration Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return brier


def perform_cost_sensitive_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    save_path: Path = None,
) -> Dict[str, Any]:
    """
    Evaluates business cost and F1-score across all possible classification thresholds.
    Finds:
    1. Optimal cost-minimizing threshold (tau_cost)
    2. Optimal F1-maximizing threshold (tau_f1)
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    total_costs = []
    f1_scores = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = (fn * cost_fn) + (fp * cost_fp)
        total_costs.append(cost)
        f1_scores.append(f1_score(y_true, preds))

    best_cost_idx = np.argmin(total_costs)
    opt_threshold_cost = float(thresholds[best_cost_idx])
    min_cost = float(total_costs[best_cost_idx])

    best_f1_idx = np.argmax(f1_scores)
    opt_threshold_f1 = float(thresholds[best_f1_idx])
    max_f1 = float(f1_scores[best_f1_idx])

    default_preds = (y_proba >= 0.50).astype(int)
    _, def_fp, def_fn, _ = confusion_matrix(y_true, default_preds).ravel()
    default_cost = float((def_fn * cost_fn) + (def_fp * cost_fp))

    if save_path:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        color = "#d62728"
        ax1.set_xlabel("Decision Threshold", fontsize=11)
        ax1.set_ylabel("Total Business Cost ($)", color=color, fontsize=11)
        ax1.plot(thresholds, total_costs, color=color, lw=2, label="Total Cost ($)")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.axvline(
            opt_threshold_cost,
            color=color,
            linestyle="--",
            label=f"Min Cost Threshold ({opt_threshold_cost:.2f})",
        )

        ax2 = ax1.twinx()
        color = "#1f77b4"
        ax2.set_ylabel("F1 Score", color=color, fontsize=11)
        ax2.plot(thresholds, f1_scores, color=color, lw=2, linestyle="-.", label="F1 Score")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.axvline(
            opt_threshold_f1,
            color=color,
            linestyle=":",
            label=f"Max F1 Threshold ({opt_threshold_f1:.2f})",
        )

        plt.title("Cost-Sensitive Threshold Analysis", fontsize=12, fontweight="bold")
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    return {
        "optimal_cost_threshold": opt_threshold_cost,
        "minimized_total_cost": min_cost,
        "default_threshold_cost": default_cost,
        "cost_savings": default_cost - min_cost,
        "optimal_f1_threshold": opt_threshold_f1,
        "max_f1_score": max_f1,
    }


def compute_permutation_importance(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, save_path: Path
) -> pd.DataFrame:
    """Calculate and plot permutation feature importance on held-out test data."""
    print("Computing permutation feature importance...")
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance_Mean": result.importances_mean,
            "Importance_Std": result.importances_std,
        }
    ).sort_values(by="Importance_Mean", ascending=False)

    top_features = perm_df.head(10).iloc[::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(
        y=top_features["Feature"],
        width=top_features["Importance_Mean"],
        xerr=top_features["Importance_Std"],
        color="#1f77b4",
        capsize=3,
        alpha=0.85,
    )
    plt.title("Top 10 Features (Permutation Importance - ROC-AUC)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean ROC-AUC Drop When Permuted", fontsize=11)
    plt.ylabel("Raw Feature", fontsize=11)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return perm_df


def run_full_evaluation():
    """Orchestrates comprehensive model evaluation and diagnostics."""
    print("Running diagnostic evaluation suite...")
    X_test, y_test, pipeline = load_eval_data_and_model()

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # 1. Plots
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    roc_path = PLOTS_DIR / "roc_curve.png"
    pr_path = PLOTS_DIR / "precision_recall_curve.png"
    calib_path = PLOTS_DIR / "calibration_curve.png"
    cost_path = PLOTS_DIR / "cost_threshold_analysis.png"
    feat_path = PLOTS_DIR / "feature_importance.png"

    plot_confusion_matrix(y_test, y_pred, cm_path)
    roc_auc, pr_auc = plot_roc_and_pr_curves(y_test, y_proba, roc_path, pr_path)
    brier = plot_calibration_curve(y_test, y_proba, calib_path)
    cost_analysis = perform_cost_sensitive_threshold_analysis(
        y_test, y_proba, save_path=cost_path
    )
    perm_df = compute_permutation_importance(pipeline, X_test, y_test, feat_path)

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    summary = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "classification_report": report_dict,
        "cost_analysis": cost_analysis,
        "top_features": perm_df.head(10).to_dict(orient="records"),
    }

    summary_file = ARTIFACTS_DIR / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Diagnostics Complete ===")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    print(f"Default Cost: ${cost_analysis['default_threshold_cost']:,.2f}")
    print(f"Optimal Cost-Minimizing Threshold: {cost_analysis['optimal_cost_threshold']:.2f}")
    print(f"Optimized Cost: ${cost_analysis['minimized_total_cost']:,.2f}")
    print(f"Business Cost Savings: ${cost_analysis['cost_savings']:,.2f}")
    print(f"Plots saved to: {PLOTS_DIR}")

    return summary


if __name__ == "__main__":
    run_full_evaluation()
