from app.utils.atwater import validate_calories
from app.schemas.response import NutrientProfile


def test_valid_calories():
    # 4*10 + 4*20 + 9*5 + 2*3 = 40+80+45+6 = 171, set actual to 171
    nutrients = NutrientProfile(protein=10, carbohydrates=20, total_fat=5, dietary_fiber=3, calories=171)
    is_valid, expected, actual = validate_calories(nutrients)
    assert is_valid is True
    assert abs(expected - 171) < 1.0


def test_invalid_calories_too_high():
    nutrients = NutrientProfile(protein=10, carbohydrates=20, total_fat=5, dietary_fiber=3, calories=250)
    is_valid, expected, actual = validate_calories(nutrients)
    assert is_valid is False


def test_invalid_calories_too_low():
    nutrients = NutrientProfile(protein=10, carbohydrates=20, total_fat=5, dietary_fiber=3, calories=50)
    is_valid, expected, actual = validate_calories(nutrients)
    assert is_valid is False


def test_zero_nutrients_valid():
    nutrients = NutrientProfile()
    is_valid, expected, actual = validate_calories(nutrients)
    assert is_valid is True


def test_formula_accuracy():
    # 4*20 + 4*50 + 9*10 + 2*5 = 80+200+90+10 = 380
    nutrients = NutrientProfile(protein=20, carbohydrates=50, total_fat=10, dietary_fiber=5, calories=380)
    is_valid, expected, actual = validate_calories(nutrients)
    assert is_valid is True
    assert abs(expected - 380) < 1.0
