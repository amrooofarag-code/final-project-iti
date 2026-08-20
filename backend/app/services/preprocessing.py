"""
Preprocessing service to convert PredictionRequest into 1-row Pandas DataFrame.
"""

from typing import List, Set
import pandas as pd
from app.schemas.prediction import PredictionRequest


def prepare_features(request: PredictionRequest, allowed_locations: Set[str]) -> tuple[pd.DataFrame, str]:
    """
    Transforms a single Pydantic PredictionRequest into a 1-row DataFrame 
    with exact column names and feature mapping expected by the trained pipeline.
    """
    raw_loc = request.location.strip().lower()
    location_grouped = raw_loc if raw_loc in allowed_locations else "other"

    data = {
        "carpet_area_sqft": [float(request.carpet_area_sqft)],
        "floor_num": [int(request.floor_num)],
        "bathroom": [int(request.bathroom)],
        "balcony": [int(request.balcony)],
        "location_grouped": [location_grouped],
        "Furnishing": [request.furnishing],
        "Transaction": [request.transaction],
        "Ownership": [request.ownership],
        "facing": [request.facing]
    }

    df = pd.DataFrame(data)
    return df, location_grouped


def format_inr(amount: float) -> str:
    """
    Formats numeric rupees into human-readable Indian currency representation 
    (e.g., ₹ 42.5 Lacs or ₹ 1.25 Cr).
    """
    if amount >= 1e7:
        cr = amount / 1e7
        return f"₹ {cr:.2f} Cr"
    elif amount >= 1e5:
        lacs = amount / 1e5
        return f"₹ {lacs:.2f} Lacs"
    else:
        return f"₹ {amount:,.0f}"
