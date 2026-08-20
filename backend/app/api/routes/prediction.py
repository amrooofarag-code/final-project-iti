"""
FastAPI Routes for Health Check and Price Prediction endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction import PredictionRequest, PredictionResponse, HealthResponse
from app.services.inference import model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend service status."""
    if model_service.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(request: PredictionRequest):
    """
    Predict house price based on property features (location, area, floors, bathrooms, balconies, etc.).
    """
    try:
        response = model_service.predict(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
