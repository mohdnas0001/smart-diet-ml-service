import time
import uuid
from typing import List, Tuple
from PIL import Image

from app.models.detector import FoodDetector
from app.models.classifier import FoodClassifier
from app.models.portion_estimator import PortionEstimator
from app.services.preprocessing import preprocess_for_detection, preprocess_for_classification
from app.services.nutrient_service import NutrientService
from app.services.food_mapper import FoodMapper
from app.schemas.response import (
    AnalysisResponse,
    FoodItem,
    BoundingBox,
    MacroSummary,
)
from app.schemas.common import FoodRegion
from app.utils.logger import logger


class AnalysisPipeline:
    """
    Orchestrates the full food analysis pipeline:
    Preprocess → Detect → Classify → Estimate Portion → Compute Nutrients
    """

    def __init__(
        self,
        detector: FoodDetector,
        classifier: FoodClassifier,
        portion_estimator: PortionEstimator,
        nutrient_service: NutrientService,
        food_mapper: FoodMapper,
        demo_mode: bool = True,
    ):
        self.detector = detector
        self.classifier = classifier
        self.portion_estimator = portion_estimator
        self.nutrient_service = nutrient_service
        self.food_mapper = food_mapper
        self.demo_mode = demo_mode

    async def run(self, image: Image.Image) -> AnalysisResponse:
        start_ts = time.time()
        warnings: List[str] = []

        if self.demo_mode:
            warnings.append("Running in DEMO mode — model weights not loaded")

        img_w, img_h = image.size

        # 1. Preprocess for detection
        det_array = preprocess_for_detection(image)

        # 2. Detect
        detections = self.detector.detect(det_array)

        food_items: List[FoodItem] = []
        for det in detections:
            label = det["label"]
            confidence = det["confidence"]
            bbox_dict = det["bbox"]

            # 3. Classify (refine label)
            cls_array = preprocess_for_classification(image)
            cls_result = self.classifier.classify(cls_array, detected_label=label)
            refined_label = cls_result["name"]
            refined_confidence = cls_result["confidence"]
            region_str = cls_result.get("region", "nigerian")

            # 4. Map to canonical name
            canonical = self.food_mapper.map_food_label(refined_label)
            if canonical is None:
                canonical = refined_label
                warnings.append(f"Could not map '{refined_label}' to Nigerian food DB")

            # 5. Estimate portion
            bbox_area = bbox_dict["width"] * bbox_dict["height"]
            portion_grams = self.portion_estimator.estimate(canonical, bbox_area_fraction=bbox_area)

            # 6. Get nutrients
            nutrients = await self.nutrient_service.get_nutrients(canonical, portion_grams)

            food_items.append(FoodItem(
                name=canonical,
                confidence=refined_confidence,
                bounding_box=BoundingBox(
                    x=bbox_dict["x"],
                    y=bbox_dict["y"],
                    width=bbox_dict["width"],
                    height=bbox_dict["height"],
                ),
                portion_grams=portion_grams,
                nutrients=nutrients,
                food_region=FoodRegion(region_str) if region_str in ("nigerian", "international") else FoodRegion.nigerian,
            ))

        total_calories = sum(fi.nutrients.calories for fi in food_items)
        total_macros = MacroSummary(
            total_calories=total_calories,
            total_protein=sum(fi.nutrients.protein for fi in food_items),
            total_carbs=sum(fi.nutrients.carbohydrates for fi in food_items),
            total_fat=sum(fi.nutrients.total_fat for fi in food_items),
            total_fiber=sum(fi.nutrients.dietary_fiber for fi in food_items),
        )

        elapsed_ms = round((time.time() - start_ts) * 1000, 2)

        return AnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            image_width=img_w,
            image_height=img_h,
            food_items=food_items,
            total_calories=total_calories,
            total_macronutrients=total_macros,
            processing_time_ms=elapsed_ms,
            model_versions={
                "detector": "demo" if self.detector.demo_mode else "yolov8",
                "classifier": "demo" if self.classifier.demo_mode else "efficientnet_b4",
                "portion_estimator": "bayesian_prior",
            },
            warnings=warnings,
        )
