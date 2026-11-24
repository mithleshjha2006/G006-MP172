import pytest

from fastapi.testclient import TestClient
from mp172 import api


class FakeModel:

    feature_names_in_ = [
        "distance_to_campus_km",
        "room_area_sqft",
        "sharing",
        "meals_per_day",
        "building_age_yr",
        "wifi_mbps",
        "furnishing",
        "locality",
        "date_ordinal",
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
    ]

    def predict(self, features):

        assert len(features) == 1
        assert "date_ordinal" in features.columns
        assert "month_sin" in features.columns
        assert "month_cos" in features.columns
        assert "weekday_sin" in features.columns
        assert "weekday_cos" in features.columns

        return [25000.0]


@pytest.fixture
def client(monkeypatch):

    monkeypatch.setattr(
        api,
        "load_model",
        lambda: FakeModel()
    )

    return TestClient(api.app)


def valid_payload():

    return {
        "distance_to_campus_km": 3.5,
        "room_area_sqft": 365,
        "sharing": 3,
        "meals_per_day": 3,
        "building_age_yr": 15,
        "wifi_mbps": 280,
        "furnishing": "full",
        "locality": "Market Side"
    }


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction(client):

    response = client.post(
        "/predict",
        json=valid_payload()
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 25000.0


def test_negative_distance_rejected(client):

    payload = valid_payload()
    payload["distance_to_campus_km"] = -1

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_wifi_above_550_rejected(client):

    payload = valid_payload()
    payload["wifi_mbps"] = 600

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_fields_rejected(client):

    payload = valid_payload()
    payload["date"] = "2026-11-24"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_model_unavailable(monkeypatch):

    def failed_model():
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(api, "load_model", failed_model)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json=valid_payload()
        )

    assert response.status_code == 503

