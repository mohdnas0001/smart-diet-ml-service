import json
from pathlib import Path
from typing import Optional
from app.schemas.response import NutrientProfile
from app.services.usda_client import USDAClient
from app.services.nutritionix_client import NutritionixClient
from app.utils.logger import logger
from app.utils.atwater import validate_calories


class NutrientService:
    """
    Hierarchical nutrient lookup:
      1. Nigerian Food DB (local JSON)
      2. USDA FoodData Central API
      3. Nutritionix API
      4. Default empty profile with warning
    """

    def __init__(
        self,
        nigerian_foods_path: str,
        usda_client: USDAClient,
        nutritionix_client: NutritionixClient,
    ):
        self.usda_client = usda_client
        self.nutritionix_client = nutritionix_client
        self.nigerian_db: dict = {}
        self._load_nigerian_db(nigerian_foods_path)

    def _load_nigerian_db(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                self.nigerian_db = json.load(f)
            logger.info("NutrientService: loaded %d Nigerian foods", len(self.nigerian_db))
        except Exception as exc:
            logger.warning("Could not load Nigerian food DB: %s", exc)

    def _scale_nutrients(self, profile: NutrientProfile, portion_grams: float) -> NutrientProfile:
        """Scale all nutrient values from per-100g to actual portion."""
        factor = portion_grams / 100.0
        data = {k: round(v * factor, 4) for k, v in profile.model_dump().items()}
        return NutrientProfile(**data)

    def get_nutrients_from_nigerian_db(self, food_name: str) -> Optional[NutrientProfile]:
        entry = self.nigerian_db.get(food_name)
        if entry is None:
            return None
        return NutrientProfile(**entry)

    async def get_nutrients(self, food_name: str, portion_grams: float = 100.0) -> NutrientProfile:
        """Get nutrients for food_name scaled to portion_grams."""
        # 1. Nigerian DB
        profile = self.get_nutrients_from_nigerian_db(food_name)
        if profile:
            logger.debug("NutrientService: '%s' found in Nigerian DB", food_name)
            return self._scale_nutrients(profile, portion_grams)

        # 2. USDA
        profile = await self.usda_client.query(food_name)
        if profile:
            logger.debug("NutrientService: '%s' found via USDA", food_name)
            return self._scale_nutrients(profile, portion_grams)

        # 3. Nutritionix
        profile = await self.nutritionix_client.query(food_name)
        if profile:
            logger.debug("NutrientService: '%s' found via Nutritionix", food_name)
            return self._scale_nutrients(profile, portion_grams)

        # 4. Default
        logger.warning("NutrientService: no data found for '%s' — using defaults", food_name)
        return NutrientProfile()

    def atwater_validate(self, nutrients: NutrientProfile):
        """Validate calorie consistency using Atwater factors."""
        return validate_calories(nutrients)
