import pytest
from app.services.nutrient_service import NutrientService
from app.services.usda_client import USDAClient
from app.services.nutritionix_client import NutritionixClient


@pytest.fixture(scope="module")
def nutrient_svc():
    usda = USDAClient(api_key="")
    nutritionix = NutritionixClient(app_id="", app_key="")
    return NutrientService(
        nigerian_foods_path="./data/nigerian_foods.json",
        usda_client=usda,
        nutritionix_client=nutritionix,
    )


@pytest.mark.asyncio
async def test_get_nutrients_jollof_rice(nutrient_svc):
    profile = await nutrient_svc.get_nutrients("jollof_rice", 100.0)
    assert profile.calories > 0


@pytest.mark.asyncio
async def test_get_nutrients_scales_correctly(nutrient_svc):
    p100 = await nutrient_svc.get_nutrients("jollof_rice", 100.0)
    p200 = await nutrient_svc.get_nutrients("jollof_rice", 200.0)
    assert abs(p200.calories - p100.calories * 2) < 1.0


@pytest.mark.asyncio
async def test_get_nutrients_unknown_food_returns_default(nutrient_svc):
    profile = await nutrient_svc.get_nutrients("totally_unknown_food_xyz", 100.0)
    assert profile is not None


@pytest.mark.asyncio
async def test_get_nutrients_from_nigerian_db(nutrient_svc):
    profile = nutrient_svc.get_nutrients_from_nigerian_db("egusi_soup")
    assert profile is not None
    assert profile.protein > 0
