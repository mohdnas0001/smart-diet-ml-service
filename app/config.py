from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # API Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DEBUG: bool = True
    
    # Model & Data Configuration
    MODEL_DIR: str = "./models"
    DATA_DIR: str = "./data"
    
    # Service Configuration
    APP_VERSION: str = "1.0.0"
    DEMO_MODE: bool = True
    
    # External API Keys
    USDA_API_KEY: str = ""
    NUTRITIONIX_APP_ID: str = ""
    NUTRITIONIX_APP_KEY: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
