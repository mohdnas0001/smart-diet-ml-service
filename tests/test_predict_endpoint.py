def test_predict_returns_200(test_client, sample_image_bytes):
    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200


def test_predict_returns_food_items(test_client, sample_image_bytes):
    data = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    assert "food_items" in data
    assert isinstance(data["food_items"], list)
    assert len(data["food_items"]) >= 1


def test_predict_returns_analysis_id(test_client, sample_image_bytes):
    data = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    assert "analysis_id" in data


def test_predict_returns_processing_time(test_client, sample_image_bytes):
    data = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    assert "processing_time_ms" in data
    assert data["processing_time_ms"] >= 0


def test_predict_rejects_non_image(test_client):
    response = test_client.post(
        "/api/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
