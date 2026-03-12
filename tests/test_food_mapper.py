from app.services.food_mapper import FoodMapper


def test_exact_match():
    mapper = FoodMapper("./data/nigerian_foods.json")
    result = mapper.map_food_label("jollof_rice")
    assert result == "jollof_rice"


def test_fuzzy_match_similar():
    mapper = FoodMapper("./data/nigerian_foods.json")
    result = mapper.map_food_label("jollof rice")
    assert result == "jollof_rice"


def test_fuzzy_match_typo():
    mapper = FoodMapper("./data/nigerian_foods.json")
    result = mapper.map_food_label("egusi suop")
    assert result is not None


def test_no_match_returns_none():
    mapper = FoodMapper("./data/nigerian_foods.json")
    result = mapper.map_food_label("xyznonexistentfood999", score_threshold=99)
    assert result is None
