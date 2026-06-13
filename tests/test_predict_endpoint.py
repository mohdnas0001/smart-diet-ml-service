def test_predict_returns_200(test_client, sample_image_bytes):
    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200


def test_predict_returns_food_items(test_client, sample_image_bytes):
    """Test that prediction returns food_items (may be empty in demo with real model, non-empty in demo mode)."""
    data = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    ).json()
    assert "food_items" in data
    assert isinstance(data["food_items"], list)
    # In demo mode or with a real model on a synthetic image, food_items may be empty
    # Just verify the structure is correct
    if len(data["food_items"]) > 0:
        item = data["food_items"][0]
        assert "name" in item
        assert "confidence" in item
        assert "portion_grams" in item
        assert "nutrients" in item


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


def test_predict_error_payload_contains_status_and_code(test_client):
    response = test_client.post(
        "/api/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    data = response.json()
    assert data["status"] == "error"
    assert data["error_code"] == "HTTPException"
    assert data["message"] == "Unsupported media type 'text/plain'. Use JPEG or PNG."
