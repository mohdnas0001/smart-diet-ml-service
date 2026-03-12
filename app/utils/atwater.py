from app.schemas.response import NutrientProfile


def validate_calories(nutrients: NutrientProfile):
    """
    Validate calories using Atwater factors.
    Returns (is_valid, expected_calories, actual_calories).
    Formula: 4*protein + 4*carbs + 9*fat + 2*fiber
    Tolerance: 15%
    """
    expected = (
        4 * nutrients.protein
        + 4 * nutrients.carbohydrates
        + 9 * nutrients.total_fat
        + 2 * nutrients.dietary_fiber
    )
    actual = nutrients.calories
    if expected == 0:
        return True, expected, actual
    diff_pct = abs(expected - actual) / expected
    is_valid = diff_pct <= 0.15
    return is_valid, round(expected, 2), round(actual, 2)
