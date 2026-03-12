import json
import random
from pathlib import Path
from typing import List, Dict, Any
from app.utils.logger import logger


class FoodDetector:
    """
    YOLOv8-based food detector.
    Falls back to demo mode when model weights are not available.
    """

    def __init__(self, model_path: str, food_categories_path: str):
        self.demo_mode = True
        self.model = None
        self._load_food_categories(food_categories_path)
        path = Path(model_path)
        if path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(path))
                self.demo_mode = False
                logger.info("YOLOv8 model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Could not load YOLOv8 model: %s — running in demo mode", exc)
        else:
            logger.info("YOLOv8 weights not found at %s — running in demo mode", model_path)

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

    def detect(self, image_array) -> List[Dict[str, Any]]:
        """
        Detect food items in an image.
        Returns list of dicts with keys: label, confidence, bbox (x, y, w, h normalised 0-1).
        """
        if self.demo_mode:
            return self._demo_detect()
        results = self.model(image_array)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                detections.append({
                    "label": self.model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    },
                })
        return detections

    def _demo_detect(self) -> List[Dict[str, Any]]:
        n = random.randint(1, 3)
        selected = random.sample(self.food_categories, min(n, len(self.food_categories)))
        detections = []
        x_cursor = 0.05
        for item in selected:
            w = round(random.uniform(0.2, 0.4), 3)
            h = round(random.uniform(0.3, 0.5), 3)
            detections.append({
                "label": item["name"],
                "confidence": round(random.uniform(0.72, 0.97), 3),
                "bbox": {
                    "x": round(x_cursor, 3),
                    "y": round(random.uniform(0.1, 0.3), 3),
                    "width": w,
                    "height": h,
                },
            })
            x_cursor += w + 0.05
        return detections
