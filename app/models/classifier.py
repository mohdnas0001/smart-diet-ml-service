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
        self.trained_classes: List[str] = []  # class names in training order
        self._load_food_categories(food_categories_path)
        path = Path(model_path)
        if path.exists():
            try:
                import torch
                import timm
                # Infer num_classes from the checkpoint to avoid shape mismatches
                ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
                num_classes = ckpt["classifier.weight"].shape[0]
                self.model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
                self.model.load_state_dict(ckpt)
                self.model.eval()
                self.demo_mode = False
                # Load class names saved during training
                classes_path = Path(model_path).parent / "classifier_classes.json"
                if classes_path.exists():
                    with open(classes_path) as f:
                        self.trained_classes = json.load(f)
                    logger.info("EfficientNet-B4 model loaded from %s (%d classes)", model_path, num_classes)
                else:
                    logger.warning("classifier_classes.json not found — predictions may use wrong labels")
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

    def classify(self, image, detected_label: str = "") -> Dict[str, Any]:
        """
        Classify food image crop.
        Accepts PIL Image or numpy array.
        Returns dict with keys: name, confidence, region.
        """
        if self.demo_mode:
            return self._demo_classify(detected_label)
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        
        # Ensure we have a PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to tensor using standard ImageNet transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)
        
        # Use trained class names if available, else fall back to food_categories
        if self.trained_classes:
            class_name = self.trained_classes[idx.item()]
            # Try to find region from food_categories
            region = "international"
            for cat in self.food_categories:
                if cat["name"] == class_name:
                    region = cat.get("region", "international")
                    break
            return {"name": class_name, "confidence": float(conf.item()), "region": region}
        # Fallback: use food_categories index (only correct if sizes match)
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
