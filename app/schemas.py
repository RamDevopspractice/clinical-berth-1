"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """Single sentence prediction request"""

    sentence: str = Field(..., min_length=1, description="Clinical sentence to classify")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple sentences"""

    sentences: List[str] = Field(..., min_items=1, description="List of clinical sentences to classify")


class PredictionResponse(BaseModel):
    """Prediction response with label and confidence score"""

    label: str = Field(..., description="Predicted assertion status (PRESENT, ABSENT, CONDITIONAL, etc.)")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_name: str
