"""
Inference service responsible for loading model pickle and running prediction pipelines.
"""

import os
import json
import joblib
from typing import Optional, Set
from app.core.config import settings
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.preprocessing import prepare_features, format_inr


class ModelService:
    def __init__(self):
        self.model = None
        self.allowed_locations: Set[str] = set()

    def load_model(self):
        """Loads fitted Sklearn Pipeline and allowed locations list from disk."""
        model_path = os.path.abspath(settings.MODEL_PATH)
        locations_path = os.path.abspath(settings.LOCATIONS_PATH)

        # Fallback path resolution if running from backend root
        if not os.path.exists(model_path):
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "house_price.pkl"))
        if not os.path.exists(locations_path):
            locations_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "locations.json"))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(locations_path):
            raise FileNotFoundError(f"Locations file not found at: {locations_path}")

        print(f"Loading model from: {model_path}")
        self.model = joblib.load(model_path)

        with open(locations_path, 'r', encoding='utf-8') as f:
            locations_list = json.load(f)
            self.allowed_locations = set(locations_list)
        print(f"Loaded {len(self.allowed_locations)} allowed locations.")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Executes model inference pipeline on input request."""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Ensure load_model() was called on startup.")

        df_input, location_used = prepare_features(request, self.allowed_locations)
        raw_pred = self.model.predict(df_input)[0]
        
        # Enforce non-negative price floor
        predicted_price = float(max(100000.0, raw_pred))
        formatted_price = format_inr(predicted_price)

        return PredictionResponse(
            predicted_price=predicted_price,
            formatted_price=formatted_price,
            currency="INR",
            location_used=location_used
        )


model_service = ModelService()
