"""
Pydantic Request & Response Schemas for House Price Prediction.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    location: str = Field(..., example="bangalore", description="Property city or location name")
    carpet_area_sqft: float = Field(..., gt=0, example=1200.0, description="Carpet or total area in square feet")
    floor_num: int = Field(..., ge=-1, example=3, description="Floor number (-1 for Basement, 0 for Ground, >0 for upper floors)")
    bathroom: int = Field(..., ge=1, example=2, description="Number of bathrooms")
    balcony: int = Field(..., ge=0, example=1, description="Number of balconies")
    furnishing: str = Field(default="Semi-Furnished", example="Semi-Furnished", description="Furnishing status: Furnished, Semi-Furnished, Unfurnished")
    transaction: str = Field(default="Resale", example="Resale", description="Transaction type: New Property, Resale")
    ownership: str = Field(default="Freehold", example="Freehold", description="Ownership type: Freehold, Leasehold, Co-operative Society, Power of Attorney")
    facing: str = Field(default="East", example="East", description="Property facing direction (e.g. East, North-East, West)")


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., example=7500000.0, description="Predicted price in Indian Rupees (INR)")
    formatted_price: str = Field(..., example="₹ 75.0 Lacs", description="Human-readable formatted price in Lacs or Crores")
    currency: str = Field(default="INR", example="INR")
    location_used: str = Field(..., example="bangalore", description="Location name used by model after grouping")


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
