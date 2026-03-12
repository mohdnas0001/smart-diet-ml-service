def test_nutrient_endpoint_returns_200(test_client):
    response = test_client.get("/api/nutrients/jollof_rice")
    assert response.status_code == 200


def test_nutrient_endpoint_returns_calories(test_client):
    data = test_client.get("/api/nutrients/jollof_rice").json()
    assert "calories" in data
    assert data["calories"] > 0


def test_nutrient_endpoint_with_portion(test_client):
    data = test_client.get("/api/nutrients/jollof_rice?portion_grams=200").json()
    base = test_client.get("/api/nutrients/jollof_rice?portion_grams=100").json()
    assert abs(data["calories"] - base["calories"] * 2) < 1.0


def test_nutrient_endpoint_unknown_food_returns_default(test_client):
    response = test_client.get("/api/nutrients/unknown_xyz_food_999")
    assert response.status_code == 200
    data = response.json()
    assert "calories" in data
