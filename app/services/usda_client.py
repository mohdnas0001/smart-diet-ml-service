import httpx
from typing import Optional
from app.schemas.response import NutrientProfile
from app.utils.logger import logger


class USDAClient:
    """Async client for USDA FoodData Central API."""

    BASE_URL = "https://api.nal.usda.gov/fdc/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def query(self, food_name: str) -> Optional[NutrientProfile]:
        if not self.api_key:
            logger.debug("USDA API key not set — skipping USDA lookup")
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.BASE_URL}/foods/search",
                    params={"query": food_name, "api_key": self.api_key, "pageSize": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                foods = data.get("foods", [])
                if not foods:
                    return None
                return self._parse_nutrients(foods[0])
        except Exception as exc:
            logger.warning("USDA query failed for '%s': %s", food_name, exc)
            return None

    def _parse_nutrients(self, food_data: dict) -> NutrientProfile:
        nutrient_map = {
            1008: "calories",
            1005: "carbohydrates",
            1003: "protein",
            1004: "total_fat",
            1079: "dietary_fiber",
            2000: "total_sugars",
            1093: "sodium",
            1092: "potassium",
            1087: "calcium",
            1089: "iron",
            1090: "magnesium",
            1091: "phosphorus",
            1095: "zinc",
            1098: "copper",
            1101: "manganese",
            1103: "selenium",
            1106: "vitamin_a",
            1162: "vitamin_c",
            1114: "vitamin_d",
            1109: "vitamin_e",
            1185: "vitamin_k",
            1165: "thiamin_b1",
            1166: "riboflavin_b2",
            1167: "niacin_b3",
            1170: "pantothenic_acid_b5",
            1175: "vitamin_b6",
            1177: "folate_b9",
            1178: "vitamin_b12",
            1210: "tryptophan",
            1211: "threonine",
            1212: "isoleucine",
            1213: "leucine",
            1214: "lysine",
            1215: "methionine",
            1217: "phenylalanine",
            1219: "valine",
            1221: "histidine",
        }
        profile_data = {}
        for n in food_data.get("foodNutrients", []):
            nid = n.get("nutrientId")
            if nid in nutrient_map:
                profile_data[nutrient_map[nid]] = n.get("value", 0.0)
        return NutrientProfile(**profile_data)
