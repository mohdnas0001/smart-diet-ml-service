import io
import pytest
from PIL import Image
from app.services.analysis_pipeline import AnalysisPipeline
from app.models.detector import FoodDetector
from app.models.classifier import FoodClassifier
from app.models.portion_estimator import PortionEstimator
from app.services.nutrient_service import NutrientService
from app.services.food_mapper import FoodMapper
from app.services.usda_client import USDAClient
from app.services.nutritionix_client import NutritionixClient


@pytest.fixture(scope="module")
def pipeline():
    detector = FoodDetector(
        model_path="./models/nonexistent.pt",
        food_categories_path="./data/food_categories.json",
    )
    classifier = FoodClassifier(
        model_path="./models/nonexistent.pt",
        food_categories_path="./data/food_categories.json",
    )
    portion_estimator = PortionEstimator(
        portion_priors_path="./data/portion_priors.json",
        food_density_path="./data/food_density_table.json",
    )
    usda = USDAClient(api_key="")
    nutritionix = NutritionixClient(app_id="", app_key="")
    nutrient_svc = NutrientService(
        nigerian_foods_path="./data/nigerian_foods.json",
        usda_client=usda,
        nutritionix_client=nutritionix,
    )
    food_mapper = FoodMapper("./data/nigerian_foods.json")
    return AnalysisPipeline(
        detector=detector,
        classifier=classifier,
        portion_estimator=portion_estimator,
        nutrient_service=nutrient_svc,
        food_mapper=food_mapper,
        demo_mode=True,
    )


def make_test_image():
    return Image.new("RGB", (640, 480), color=(180, 120, 80))


@pytest.mark.asyncio
async def test_pipeline_returns_response(pipeline):
    img = make_test_image()
    result = await pipeline.run(img)
    assert result is not None


@pytest.mark.asyncio
async def test_pipeline_returns_food_items(pipeline):
    img = make_test_image()
    result = await pipeline.run(img)
    assert len(result.food_items) >= 1


@pytest.mark.asyncio
async def test_pipeline_sets_demo_warning(pipeline):
    img = make_test_image()
    result = await pipeline.run(img)
    assert any("DEMO" in w.upper() for w in result.warnings)


@pytest.mark.asyncio
async def test_pipeline_image_dimensions(pipeline):
    img = make_test_image()
    result = await pipeline.run(img)
    assert result.image_width == 640
    assert result.image_height == 480
