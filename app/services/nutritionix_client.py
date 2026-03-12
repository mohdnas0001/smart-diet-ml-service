import httpx
from typing import Optional
from app.schemas.response import NutrientProfile
from app.utils.logger import logger


class NutritionixClient:
    """Async client for Nutritionix API."""

    BASE_URL = "https://trackapi.nutritionix.com/v2"

    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key

    async def query(self, food_name: str) -> Optional[NutrientProfile]:
        if not self.app_id or not self.app_key:
            logger.debug("Nutritionix credentials not set — skipping lookup")
            return None
        try:
            headers = {
                "x-app-id": self.app_id,
                "x-app-key": self.app_key,
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{self.BASE_URL}/natural/nutrients",
                    headers=headers,
                    json={"query": f"100g {food_name}"},
                )
                resp.raise_for_status()
                data = resp.json()
                foods = data.get("foods", [])
                if not foods:
                    return None
                return self._parse_nutrients(foods[0])
        except Exception as exc:
            logger.warning("Nutritionix query failed for '%s': %s", food_name, exc)
            return None

    def _parse_nutrients(self, food_data: dict) -> NutrientProfile:
        return NutrientProfile(
            calories=food_data.get("nf_calories", 0.0),
            total_fat=food_data.get("nf_total_fat", 0.0),
            saturated_fat=food_data.get("nf_saturated_fat", 0.0),
            cholesterol=food_data.get("nf_cholesterol", 0.0),
            sodium=food_data.get("nf_sodium", 0.0),
            carbohydrates=food_data.get("nf_total_carbohydrate", 0.0),
            dietary_fiber=food_data.get("nf_dietary_fiber", 0.0),
            total_sugars=food_data.get("nf_sugars", 0.0),
            protein=food_data.get("nf_protein", 0.0),
            potassium=food_data.get("nf_potassium", 0.0),
        )
