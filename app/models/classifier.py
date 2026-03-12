import json
import random
from pathlib import Path
from typing import Dict, Any, List
from app.utils.logger import logger


class FoodClassifier:
    """
    EfficientNet-B4 food classifier.
    Falls back to demo mode when model weights are not available.
    """

    def __init__(self, model_path: str, food_categories_path: str):
        self.demo_mode = True
        self.model = None
        self._load_food_categories(food_categories_path)
        path = Path(model_path)
        if path.exists():
            try:
                import torch
                import timm
                self.model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(self.food_categories))
                self.model.load_state_dict(torch.load(str(path), map_location="cpu"))
                self.model.eval()
                self.demo_mode = False
                logger.info("EfficientNet-B4 model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load classifier model: %s — running in demo mode", exc)
        else:
            logger.info("Classifier weights not found at %s — running in demo mode", model_path)

    def _load_food_categories(self, categories_path: str) -> None:
        try:
            with open(categories_path, "r") as f:
                self.food_categories: List[Dict] = json.load(f)
        except Exception:
            self.food_categories = [
                {"id": 1, "name": "jollof_rice", "region": "nigerian", "typical_portion_grams": 350},
                {"id": 2, "name": "egusi_soup", "region": "nigerian", "typical_portion_grams": 250},
                {"id": 3, "name": "fried_plantain", "region": "nigerian", "typical_portion_grams": 150},
            ]

    def classify(self, image_array, detected_label: str = "") -> Dict[str, Any]:
        """
        Classify food image crop.
        Returns dict with keys: name, confidence, region.
        """
        if self.demo_mode:
            return self._demo_classify(detected_label)
        import torch
        tensor = torch.tensor(image_array).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
        cat = self.food_categories[idx.item()]
        return {"name": cat["name"], "confidence": float(conf.item()), "region": cat.get("region", "international")}

    def _demo_classify(self, detected_label: str) -> Dict[str, Any]:
        if detected_label:
            matches = [c for c in self.food_categories if c["name"] == detected_label]
            if matches:
                cat = matches[0]
                return {"name": cat["name"], "confidence": round(random.uniform(0.80, 0.97), 3), "region": cat.get("region", "nigerian")}
        cat = random.choice(self.food_categories)
        return {"name": cat["name"], "confidence": round(random.uniform(0.70, 0.95), 3), "region": cat.get("region", "nigerian")}
