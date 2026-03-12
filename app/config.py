from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    USDA_API_KEY: str = ""
    NUTRITIONIX_APP_ID: str = ""
    NUTRITIONIX_APP_KEY: str = ""
    MODEL_DIR: str = "./models"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    DEMO_MODE: bool = True
    APP_VERSION: str = "1.0.0"
    DATA_DIR: str = "./data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
