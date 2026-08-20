"""
Application Configuration Settings using Pydantic BaseSettings.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "House Price Prediction API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""
    
    # Paths to model & locations json
    MODEL_PATH: str = "app/models/house_price.pkl"
    LOCATIONS_PATH: str = "app/models/locations.json"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
