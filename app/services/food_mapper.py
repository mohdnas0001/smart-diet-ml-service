import json
from pathlib import Path
from typing import Optional
from thefuzz import process
from app.utils.logger import logger


# Explicit aliases: maps classifier output / scrape folder names → canonical DB names.
# Add entries here whenever a model label doesn't fuzzy-match the DB correctly.
ALIASES: dict = {
    # scrape folder name variants → canonical DB key
    "boli_nigerian":           "boli",
    "chin_chin_nigerian_snack": "chin_chin",
    "puff_puff_nigerian":       "puff_puff",
    "moi_moi":                  "moin_moin",
    "fried_rice_nigerian":      "nigerian_fried_rice",
    "zobo_drink":               "zobo",
    "salad":                    "nigerian_salad",
    "rice_and_chicken":         "fried_rice",
    # International foods NOT in Nigerian DB — map to closest equivalent
    "pasta":                    "jollof_spaghetti",
    "pizza":                    "nigerian_shawarma",   # closest baked/dough item
    "sandwich":                 "meat_pie",            # closest hand-held snack
    "sushi":                    "fried_fish",          # closest fish-based item
    "soup_bowl":                "pepper_soup",
    "burger":                   "meat_pie",
    "omelette":                 "scotch_egg",
}


class FoodMapper:
    """Maps arbitrary food label strings to canonical names in the Nigerian food DB."""

    def __init__(self, nigerian_foods_path: str):
        self.food_names = []
        try:
            with open(nigerian_foods_path, "r") as f:
                db = json.load(f)
            self.food_names = list(db.keys())
            logger.info("FoodMapper loaded %d food names", len(self.food_names))
        except Exception as exc:
            logger.warning("FoodMapper could not load food DB: %s", exc)

    def map_food_label(self, label: str, score_threshold: int = 60) -> Optional[str]:
        """
        Map label to canonical food name.
        Priority: 1) exact match  2) explicit alias  3) fuzzy match
        Returns canonical name if found, else None.
        """
        if not self.food_names:
            return None
        label_norm = label.lower().replace(" ", "_").replace("-", "_")
        # 1. Exact match
        if label_norm in self.food_names:
            return label_norm
        # 2. Explicit alias override (highest priority — avoids bad fuzzy matches)
        if label_norm in ALIASES:
            alias = ALIASES[label_norm]
            if alias in self.food_names:
                return alias
        # 3. Fuzzy match
        result = process.extractOne(label_norm, self.food_names)
        if result and result[1] >= score_threshold:
            return result[0]
        return None
