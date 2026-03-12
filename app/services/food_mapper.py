import json
from pathlib import Path
from typing import Optional
from thefuzz import process
from app.utils.logger import logger


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
        Fuzzy-match label to canonical food name.
        Returns canonical name if score >= threshold, else None.
        """
        if not self.food_names:
            return None
        label_norm = label.lower().replace(" ", "_").replace("-", "_")
        if label_norm in self.food_names:
            return label_norm
        result = process.extractOne(label_norm, self.food_names)
        if result and result[1] >= score_threshold:
            return result[0]
        return None
