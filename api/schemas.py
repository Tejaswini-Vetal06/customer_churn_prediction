"""
Pydantic Data Schemas for the Churn Prediction API.
Ensures strong typing, input validation, and clear documentation.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Features required to predict customer churn probability."""

    gender: Literal["Male", "Female"] = Field(
        default="Female", description="Customer gender"
    )
    SeniorCitizen: Literal["0", "1", 0, 1] = Field(
        default=0, description="Whether customer is 65 or older (0 or 1)"
    )
    Partner: Literal["Yes", "No"] = Field(
        default="No", description="Whether customer has a partner"
    )
    Dependents: Literal["Yes", "No"] = Field(
        default="No", description="Whether customer has dependents"
    )
    tenure: int = Field(
        default=1, ge=0, le=120, description="Months customer has stayed with company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        default="Yes", description="Whether customer has phone service"
    )
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(
        default="No", description="Whether customer has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        default="Fiber optic", description="Internet service provider"
    )
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(
        default="No", description="Online security add-on"
    )
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(
        default="No", description="Online backup add-on"
    )
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(
        default="No", description="Device protection add-on"
    )
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(
        default="No", description="Tech support add-on"
    )
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(
        default="No", description="Streaming TV add-on"
    )
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(
        default="No", description="Streaming movies add-on"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month", description="Contract term"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        default="Yes", description="Paperless billing status"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(default="Electronic check", description="Payment method")
    MonthlyCharges: float = Field(
        default=70.0, ge=0.0, le=500.0, description="Monthly recurring charge"
    )
    TotalCharges: float = Field(
        default=70.0, ge=0.0, le=10000.0, description="Cumulative charges to date"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }


class SinglePredictionRequest(BaseModel):
    customer: CustomerFeatures
    custom_threshold: Optional[float] = Field(
        default=None, ge=0.01, le=0.99, description="Optional custom decision threshold"
    )


class SinglePredictionResponse(BaseModel):
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"]
    threshold_used: float
    recommendation: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]
    custom_threshold: Optional[float] = Field(
        default=None, ge=0.01, le=0.99, description="Optional custom decision threshold"
    )


class BatchCustomerResult(BaseModel):
    customer_index: int
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float
    risk_tier: Literal["Low Risk", "Medium Risk", "High Risk"]


class BatchPredictionResponse(BaseModel):
    total_customers: int
    predicted_churn_count: int
    predicted_churn_rate: float
    results: List[BatchCustomerResult]


class ModelMetadataResponse(BaseModel):
    model_name: str
    status: str
    test_roc_auc: float
    test_pr_auc: float
    test_recall: float
    test_precision: float

