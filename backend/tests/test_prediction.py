"""
Pytest Integration & Unit Tests for FastAPI House Price Prediction API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test GET /health returns 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"


def test_predict_happy_path():
    """Test POST /predict with valid payload returns 200 OK with formatted price."""
    payload = {
        "location": "bangalore",
        "carpet_area_sqft": 1200.0,
        "floor_num": 3,
        "bathroom": 2,
        "balcony": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_price" in json_data
    assert json_data["predicted_price"] > 0
    assert "formatted_price" in json_data
    assert "₹" in json_data["formatted_price"]
    assert json_data["location_used"] == "bangalore"


def test_predict_unknown_location_fallback():
    """Test POST /predict with unknown location falls back to 'other'."""
    payload = {
        "location": "unknown_custom_city_999",
        "carpet_area_sqft": 1500.0,
        "floor_num": 5,
        "bathroom": 3,
        "balcony": 2,
        "furnishing": "Furnished",
        "transaction": "New Property",
        "ownership": "Freehold",
        "facing": "North-East"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["location_used"] == "other"


def test_predict_validation_error_negative_area():
    """Test POST /predict with invalid carpet area (<= 0) returns 422 Unprocessable Entity."""
    payload = {
        "location": "bangalore",
        "carpet_area_sqft": -500.0,  # Invalid negative area
        "floor_num": 2,
        "bathroom": 2,
        "balcony": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_error_missing_required_fields():
    """Test POST /predict with missing required fields returns 422."""
    payload = {
        "location": "bangalore"
        # missing carpet_area_sqft, floor_num, bathroom, balcony
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
