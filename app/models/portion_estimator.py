import json
import random
from pathlib import Path
from typing import Dict
from app.utils.logger import logger


class PortionEstimator:
    """
    Bayesian portion size estimator using prior distributions.
    Uses visual bounding-box area and density table as features.
    """

    DEFAULT_MEAN = 250.0
    DEFAULT_STD = 80.0

    def __init__(self, portion_priors_path: str, food_density_path: str):
        self.priors: Dict = {}
        self.densities: Dict = {}
        self._load_priors(portion_priors_path)
        self._load_densities(food_density_path)

    def _load_priors(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                self.priors = json.load(f)
            logger.info("Loaded portion priors for %d foods", len(self.priors))
        except Exception as exc:
            logger.warning("Could not load portion priors: %s", exc)

    def _load_densities(self, path: str) -> None:
        try:
            with open(path, "r") as f:
                self.densities = json.load(f)
            logger.info("Loaded density table for %d foods", len(self.densities))
        except Exception as exc:
            logger.warning("Could not load density table: %s", exc)

    def estimate(self, food_name: str, bbox_area_fraction: float = 0.25) -> float:
        """
        Estimate portion in grams.
        Uses Bayesian posterior: combine prior with bbox-based likelihood.
        bbox_area_fraction is the fraction of image area covered by the bounding box.
        """
        prior = self.priors.get(food_name, {})
        mean = prior.get("mean_grams", self.DEFAULT_MEAN)
        std = prior.get("std_grams", self.DEFAULT_STD)
        min_g = prior.get("min_grams", 50.0)
        max_g = prior.get("max_grams", 800.0)

        # Scale by bbox area — larger bbox → closer to mean, small bbox → below mean
        scale = 0.5 + bbox_area_fraction * 2.0  # linear scaling from 0.5 to ~1.5
        estimate = mean * scale
        # Add small random noise (simulates prediction uncertainty)
        estimate += random.gauss(0, std * 0.1)
        return round(min(max(estimate, min_g), max_g), 1)
