def test_health_returns_200(test_client):
    response = test_client.get("/api/health")
    assert response.status_code == 200


def test_health_has_status_field(test_client):
    data = test_client.get("/api/health").json()
    assert "status" in data
    assert data["status"] == "ok"


def test_health_has_version(test_client):
    data = test_client.get("/api/health").json()
    assert "version" in data


def test_health_has_demo_mode(test_client):
    data = test_client.get("/api/health").json()
    assert "demo_mode" in data
    # demo_mode is a boolean that reflects the actual state:
    # True if models not loaded, False if models successfully loaded
    assert isinstance(data["demo_mode"], bool)
