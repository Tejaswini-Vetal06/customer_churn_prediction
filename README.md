# 🚀 Enterprise Customer Churn Prediction Platform

An end-to-end Machine Learning system engineered for customer retention analytics in telecommunications. This project demonstrates advanced ML engineering practices: **leak-free Scikit-Learn pipelines**, **5-fold Stratified Cross-Validation**, **multi-model comparison**, **hyperparameter tuning**, **cost-sensitive threshold optimization**, and **dual deployment** (FastAPI microservice + Streamlit interactive dashboard).

---

## 📌 Project Architecture

```
customer_churn_prediction/
├── .vscode/
│   ├── settings.json              # Auto-activates .venv and configures pytest in VS Code
│   └── launch.json                # 1-Click debug launch configurations (Train, API, UI, Tests)
├── data/
│   ├── raw/                       # Original IBM Telco Customer Churn dataset (7,043 records)
│   └── processed/                 # Leakage-free Stratified Train/Test splits (.parquet)
├── src/
│   ├── config.py                  # Global configurations, paths, random seed, business costs
│   ├── data_loader.py             # Ingestion, TotalCharges sanitization, and stratified split
│   ├── preprocessor.py            # ChurnFeatureEngineer & Scikit-Learn ColumnTransformer
│   ├── train.py                   # 5-Fold Stratified CV, Model Comparison, RandomizedSearchCV
│   ├── evaluate.py                # ROC/PR curves, confusion matrix, calibration, cost tuning
│   └── predict.py                 # Vectorized batch & single customer inference engine
├── api/
│   ├── schemas.py                 # Pydantic v2 request/response schemas with field validations
│   └── app.py                     # FastAPI production service (/health, /predict, /predict/batch)
├── app/
│   └── streamlit_app.py           # Web UI for business stakeholders, risk scoring & diagnostics
├── tests/
│   ├── test_data.py               # Ingestion, imputation & split stratification tests
│   ├── test_pipeline.py           # Preprocessing & predictor inference tests
│   └── test_api.py                # FastAPI endpoints, status codes & validation error tests
├── artifacts/
│   ├── churn_pipeline.joblib      # Serialized end-to-end champion pipeline
│   ├── metrics.json               # CV comparison & test evaluation metrics
│   ├── evaluation_summary.json    # Cost optimization & classification reports
│   └── plots/                     # High-res diagnostic visualizations
├── pytest.ini                     # Pytest root configuration
├── requirements.txt               # Locked project dependencies
└── README.md                      # Complete guide and documentation
```

---

## 📊 Model Comparison & Results

Across **5-Fold Stratified Cross-Validation**, candidate models were benchmarked on the training set (5,634 samples):

| Model | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | F1-Score | Recall (Churners) | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Tuned)** | **0.845 ± 0.008** | **0.665 ± 0.017** | **0.634** | **78.3%** | **75.9%** |
| **Logistic Regression (Baseline)** | 0.846 ± 0.011 | 0.664 ± 0.013 | 0.628 | 79.8% | 75.0% |
| **Gradient Boosting (GBDT)** | 0.845 ± 0.010 | 0.662 ± 0.014 | 0.574 | 51.2% | 79.8% |
| **Random Forest** | 0.843 ± 0.009 | 0.653 ± 0.019 | 0.629 | 75.3% | 76.4% |
| **AdaBoost** | 0.841 ± 0.009 | 0.640 ± 0.010 | 0.498 | 39.0% | 79.2% |
| **Dummy (Majority Class)** | 0.500 ± 0.000 | 0.265 ± 0.000 | 0.000 | 0.0% | 73.5% |

### Held-Out Test Generalization (1,409 unseen records)
- **ROC-AUC**: `0.8449`
- **PR-AUC (Average Precision)**: `0.6585`
- **Recall (Churner Sensitivity)**: `78.34%` (catches ~4 out of every 5 churners)
- **Brier Score Loss (Calibration)**: `0.1594`

---

## 💰 Cost-Sensitive Threshold Optimization

Standard classifiers use a default threshold of $\tau = 0.50$. In customer churn, the business cost of a **False Negative** (losing a customer, \$500 LTV) is **5× greater** than a **False Positive** (giving an existing customer an unnecessary \$100 retention discount).

- **Default Threshold ($\tau = 0.50$) Expected Loss**: \$66,300.00
- **Cost-Optimal Threshold ($\tau = 0.29$) Expected Loss**: \$59,400.00
- **Net Cost Savings**: **\$6,900.00** per 1,409 customer cohort.

---

## 🛠️ Step-by-Step Setup & Execution

### 1. Environment Setup
The project uses a local virtual environment:
```powershell
# Virtual environment is located at .venv
# Activate in PowerShell:
.\.venv\Scripts\Activate.ps1

# (Optional) Install dependencies:
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### 2. VS Code Integration
This repository includes pre-configured `.vscode/` settings:
- **Interpreter Selection**: Automatically points to `.venv\Scripts\python.exe`.
- **Test Explorer**: Open the **Testing** tab in VS Code sidebar to run and debug individual tests interactively.
- **Run & Debug (F5)**: Open `Run & Debug` panel and choose:
  - `Python: Train & Evaluate Pipeline`
  - `Python: Run FastAPI Server`
  - `Python: Run Streamlit Dashboard`
  - `Python: Run Pytest Test Suite`

### 3. Re-running the Training Pipeline
```powershell
.\.venv\Scripts\python.exe -m src.train
```

### 4. Generating Diagnostic Curves & Plots
```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```
Generated visual charts are saved directly to `artifacts/plots/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `calibration_curve.png`
- `cost_threshold_analysis.png`
- `feature_importance.png`

### 5. Running the FastAPI Service
```powershell
.\.venv\Scripts\uvicorn.exe api.app:app --reload --port 8000
```
- Interactive OpenAPI Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

#### Sample Prediction Request (curl / PowerShell):
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{
  "customer": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65
  },
  "custom_threshold": 0.29
}'
```

### 6. Launching the Interactive Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
Explore:
- **Single Customer Predictor**: Sliders and dropdowns with instant churn risk gauging and actionable interventions.
- **Batch CSV Scoring**: Score 50 sample customers or upload your own CSV with one-click export.
- **Model Diagnostics**: Interactive viewing of cross-validation tables and high-resolution curves.
- **Business Cost & ROI**: Cost reduction calculations under customizable decision thresholds.

### 7. Running Pytest Test Suite
```powershell
.\.venv\Scripts\pytest.exe -v
```
All 12 unit and integration tests validate data integrity, leak-free preprocessing, predictor outputs, and FastAPI schemas.

###### DEPLOYED WEB LINK  
https://customerchurnprediction-qg2c8mpy3tyawgtun4itnq.streamlit.app/

