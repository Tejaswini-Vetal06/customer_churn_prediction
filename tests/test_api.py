"""
FastAPI Microservice Integration Tests.
Uses TestClient to test endpoints, schemas, and input validation.
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    """Create test client with lifespan context entered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify healthcheck returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_metadata_endpoint(client):
    """Verify metadata returns valid scores."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["test_roc_auc"] > 0.70
    assert data["test_pr_auc"] > 0.50


def test_predict_single_endpoint(client):
    """Verify single customer prediction."""
    payload = {
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
            "TotalCharges": 151.65,
        },
        "custom_threshold": 0.35,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in ["Yes", "No"]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert data["threshold_used"] == 0.35
    assert len(data["recommendation"]) > 0


def test_predict_batch_endpoint(client):
    """Verify batch customer prediction endpoint."""
    customer_template = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.0,
        "TotalCharges": 3120.0,
    }

    payload = {
        "customers": [customer_template, customer_template],
        "custom_threshold": 0.50,
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["results"]) == 2
    assert 0.0 <= data["predicted_churn_rate"] <= 1.0


def test_validation_error_handling(client):
    """Verify pydantic raises 422 for invalid payloads."""
    # Negative tenure violates ge=0 constraint
    payload = {
        "customer": {
            "gender": "Female",
            "tenure": -10,
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

