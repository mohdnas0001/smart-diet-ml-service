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
        self._nutrient_cache: dict[tuple[str, float], NutrientProfile] = {}
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
        normalized_name = food_name.strip().lower()
        lookup_key = (normalized_name, round(float(portion_grams), 1))
        cached_profile = self._nutrient_cache.get(lookup_key)
        if cached_profile is not None:
            logger.debug("NutrientService: cache hit for '%s' (%sg)", food_name, portion_grams)
            return cached_profile

        # 1. Nigerian DB
        profile = self.get_nutrients_from_nigerian_db(food_name)
        if profile:
            logger.debug("NutrientService: '%s' found in Nigerian DB", food_name)
            scaled_profile = self._scale_nutrients(profile, portion_grams)
            self._nutrient_cache[lookup_key] = scaled_profile
            return scaled_profile

        # 2. USDA
        profile = await self.usda_client.query(food_name)
        if profile:
            logger.debug("NutrientService: '%s' found via USDA", food_name)
            scaled_profile = self._scale_nutrients(profile, portion_grams)
            self._nutrient_cache[lookup_key] = scaled_profile
            return scaled_profile

        # 3. Nutritionix
        profile = await self.nutritionix_client.query(food_name)
        if profile:
            logger.debug("NutrientService: '%s' found via Nutritionix", food_name)
            scaled_profile = self._scale_nutrients(profile, portion_grams)
            self._nutrient_cache[lookup_key] = scaled_profile
            return scaled_profile

        # 4. Default
        logger.warning("NutrientService: no data found for '%s' — using defaults", food_name)
        default_profile = NutrientProfile()
        self._nutrient_cache[lookup_key] = default_profile
        return default_profile

    def atwater_validate(self, nutrients: NutrientProfile):
        """Validate calorie consistency using Atwater factors."""
        return validate_calories(nutrients)
