from fastapi import APIRouter, HTTPException, Query
from app.schemas.response import NutrientProfile
from app.utils.logger import logger

router = APIRouter()


@router.get("/api/nutrients/{food_name}", response_model=NutrientProfile)
async def get_nutrients(food_name: str, portion_grams: float = Query(default=100.0, ge=1.0, le=5000.0)):
    """
    Look up nutrient profile for a given food name (per portion_grams).
    """
    from app.main import nutrient_service
    if nutrient_service is None:
        raise HTTPException(status_code=503, detail="Nutrient service not initialised")

    profile = await nutrient_service.get_nutrients(food_name, portion_grams)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Food '{food_name}' not found")
    return profile
