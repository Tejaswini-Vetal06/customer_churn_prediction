"""
Streamlit Web Dashboard for Customer Churn Prediction & Retention Analytics.
Interactive UI for single predictions, batch CSV scoring, and model diagnostic visualization.
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    MODEL_PATH,
    METRICS_PATH,
    PLOTS_DIR,
    ARTIFACTS_DIR,
    RAW_DATA_PATH,
)
from src.predict import CustomerChurnPredictor

st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_predictor():
    """Load customer churn predictor once and cache."""
    return CustomerChurnPredictor(model_path=MODEL_PATH)


@st.cache_data
def load_metrics_and_eval():
    """Load model metrics and evaluation summary."""
    metrics_data = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

    eval_summary = {}
    eval_file = ARTIFACTS_DIR / "evaluation_summary.json"
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_summary = json.load(f)

    return metrics_data, eval_summary


try:
    predictor = load_predictor()
    metrics_data, eval_summary = load_metrics_and_eval()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure src.train has been executed.")
    st.stop()

# Sidebar
st.sidebar.title("Churn Prediction Hub")
st.sidebar.markdown("Advanced ML for Customer Retention")

cost_analysis = eval_summary.get("cost_analysis", {})
optimal_threshold = cost_analysis.get("optimal_cost_threshold", 0.35)

decision_threshold = st.sidebar.slider(
    "Decision Threshold (Tau)",
    min_value=0.10,
    max_value=0.90,
    value=float(optimal_threshold),
    step=0.05,
    help="Default is cost-optimal threshold minimizing business churn loss.",
)
st.sidebar.caption(f"Cost-Optimal Threshold: **{optimal_threshold:.2f}**")

page = st.sidebar.radio(
    "Navigation",
    [
        "Single Customer Predictor",
        "Batch CSV Scoring",
        "Model Comparison & Diagnostics",
        "Business Cost & ROI Analysis",
    ],
)

# Header
st.title("Enterprise Customer Churn Prediction System")
st.markdown(
    "Production Machine Learning workflow powered by **HistGradientBoosting**, 5-fold Stratified Cross-Validation, and Cost-Sensitive Threshold Optimization."
)
st.divider()

if page == "Single Customer Predictor":
    st.subheader("Single Customer Risk Profiling")
    st.markdown("Enter customer details to evaluate churn probability and retention interventions.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen (65+)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

        st.markdown("### Account Tenancy")
        tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=4)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col2:
        st.markdown("### Phone & Internet Services")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])

        st.markdown("### Digital Security Add-ons")
        online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("### Support & Entertainment")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("### Billing & Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=200.0, value=75.50)
        # Sensible total charges estimate
        default_total = max(monthly_charges * max(tenure, 1), 0.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(default_total))

    predict_btn = st.button("Evaluate Churn Risk", type="primary", use_container_width=True)

    if predict_btn:
        customer_payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        result = predictor.predict_single(customer_payload, custom_threshold=decision_threshold)
        prob = result["churn_probability"]
        risk = result["risk_tier"]
        pred = result["churn_prediction"]

        st.markdown("---")
        st.subheader("Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.progress(prob)

        with res_col2:
            color = "🔴" if risk == "High Risk" else "🟡" if risk == "Medium Risk" else "🟢"
            st.metric("Risk Classification", f"{color} {risk}")

        with res_col3:
            st.metric("Binary Decision", "CHURN" if pred == "Yes" else "RETAIN", delta=f"Threshold: {decision_threshold:.2f}")

        # Actionable Business Recommendation
        st.info(f"**Recommended Retention Action:** {result['recommendation']}")

elif page == "Batch CSV Scoring":
    st.subheader("Batch Customer Churn Scoring")
    st.markdown("Upload a customer CSV or load sample dataset records for high-throughput vectorized scoring.")

    sample_col, upload_col = st.columns([1, 2])

    with sample_col:
        load_sample_btn = st.button("Load 50 Sample Raw Customers", use_container_width=True)

    with upload_col:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    df_to_score = None
    if load_sample_btn and RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(RAW_DATA_PATH)
        df_to_score = raw_df.sample(50, random_state=42).copy()
        st.success("Loaded 50 random records from raw dataset.")
    elif uploaded_file is not None:
        df_to_score = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_to_score)} customer rows.")

    if df_to_score is not None:
        batch_results = predictor.predict_batch(df_to_score, custom_threshold=decision_threshold)
        enriched = pd.concat([df_to_score.reset_index(drop=True), batch_results.reset_index(drop=True)], axis=1)

        total_cust = len(enriched)
        churn_count = (enriched["churn_prediction"] == "Yes").sum()
        churn_rate = churn_count / total_cust

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Scored Customers", f"{total_cust:,}")
        stat_col2.metric("Flagged Churners", f"{churn_count:,}")
        stat_col3.metric("Predicted Churn Rate", f"{churn_rate:.1%}")

        st.dataframe(
            enriched[
                [
                    "customerID" if "customerID" in enriched.columns else enriched.columns[0],
                    "tenure",
                    "Contract",
                    "MonthlyCharges",
                    "churn_probability",
                    "churn_prediction",
                    "risk_tier",
                ]
            ].sort_values(by="churn_probability", ascending=False),
            use_container_width=True,
        )

        csv_download = enriched.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Scored Predictions CSV",
            data=csv_download,
            file_name="churn_predictions_scored.csv",
            mime="text/csv",
        )

elif page == "Model Comparison & Diagnostics":
    st.subheader("Cross-Validation Comparison & Model Diagnostics")

    cv_records = metrics_data.get("cv_comparison", [])
    if cv_records:
        st.markdown("### 5-Fold Stratified Cross-Validation Results")
        cv_df = pd.DataFrame(cv_records)
        st.dataframe(
            cv_df.style.highlight_max(
                subset=["ROC-AUC (Mean)", "PR-AUC (Mean)", "F1 (Mean)", "Recall (Mean)"],
                color="#c6e2ff",
            ),
            use_container_width=True,
        )

    test_metrics = metrics_data.get("test_metrics", {})
    if test_metrics:
        st.markdown("### Held-Out Test Set Performance")
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        mcol1.metric("ROC-AUC", f"{test_metrics.get('test_roc_auc', 0):.3f}")
        mcol2.metric("PR-AUC", f"{test_metrics.get('test_pr_auc', 0):.3f}")
        mcol3.metric("Recall", f"{test_metrics.get('test_recall', 0):.3f}")
        mcol4.metric("Precision", f"{test_metrics.get('test_precision', 0):.3f}")
        mcol5.metric("Brier Score", f"{test_metrics.get('test_brier_score', 0):.3f}")

    st.markdown("### Diagnostic Performance Curves")
    plot_tabs = st.tabs(
        [
            "ROC & PR Curves",
            "Confusion Matrix",
            "Cost Analysis Curve",
            "Feature Importance",
            "Calibration Curve",
        ]
    )

    with plot_tabs[0]:
        pcol1, pcol2 = st.columns(2)
        roc_f = PLOTS_DIR / "roc_curve.png"
        pr_f = PLOTS_DIR / "precision_recall_curve.png"
        if roc_f.exists():
            pcol1.image(str(roc_f), caption="Receiver Operating Characteristic")
        if pr_f.exists():
            pcol2.image(str(pr_f), caption="Precision-Recall Curve")

    with plot_tabs[1]:
        cm_f = PLOTS_DIR / "confusion_matrix.png"
        if cm_f.exists():
            st.image(str(cm_f), caption="Confusion Matrix on Held-Out Test Set", width=550)

    with plot_tabs[2]:
        cost_f = PLOTS_DIR / "cost_threshold_analysis.png"
        if cost_f.exists():
            st.image(str(cost_f), caption="Business Cost vs Decision Threshold", width=700)

    with plot_tabs[3]:
        feat_f = PLOTS_DIR / "feature_importance.png"
        if feat_f.exists():
            st.image(str(feat_f), caption="Permutation Feature Importance (ROC-AUC drop)", width=750)

    with plot_tabs[4]:
        calib_f = PLOTS_DIR / "calibration_curve.png"
        if calib_f.exists():
            st.image(str(calib_f), caption="Reliability Diagram (Calibration Curve)", width=550)

elif page == "Business Cost & ROI Analysis":
    st.subheader("Business Impact & Threshold Optimization")
    st.markdown(
        """
        Standard machine learning models use a fixed decision threshold of **0.50**.
        In customer churn, misclassifying an actual churner (**False Negative**) costs a lost customer (\$500 LTV),
        while misclassifying a loyal customer (**False Positive**) only costs a promotional discount (\$100).
        """
    )

    if cost_analysis:
        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Default Threshold Cost (0.50)", f"${cost_analysis.get('default_threshold_cost', 0):,.2f}")
        bcol2.metric("Optimized Threshold Cost", f"${cost_analysis.get('minimized_total_cost', 0):,.2f}")
        bcol3.metric(
            "Net Cost Savings",
            f"${cost_analysis.get('cost_savings', 0):,.2f}",
            delta=f"Threshold: {cost_analysis.get('optimal_cost_threshold', 0.35):.2f}",
        )

        st.success(
            f"By tuning the threshold from 0.50 to **{cost_analysis.get('optimal_cost_threshold', 0.35):.2f}**, the retention team recovers an estimated **${cost_analysis.get('cost_savings', 0):,.2f}** per test cohort!"
        )

