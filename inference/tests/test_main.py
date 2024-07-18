import pytest
from app.main import app
from app.model import select_model
from fastapi.testclient import TestClient

client = TestClient(app)


def test_predict_naive_bayes():
    response = client.post(
        "/predict",
        json={
            "cap-diameter": 1372,
            "cap-shape": 2,
            "gill-attachment": 2,
            "gill-color": 10,
        },
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert response.json()["prediction"] in ["edible", "poisonous"]
    assert 0 <= response.json()["probability"] <= 1


def test_predict_logistic_regression():
    response = client.post(
        "/predict",
        json={
            "stem_height": 3.8074667544799388,
            "stem_width": 1545,
            "stem_color": 11,
            "season": 1.8042727086281731,
        },
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert response.json()["prediction"] in ["edible", "poisonous"]
    assert 0 <= response.json()["probability"] <= 1


def test_predict_invalid_features():
    response = client.post("/predict", json={"invalid-feature": 5})
    assert response.status_code == 422

@pytest.mark.parametrize("input_data, expected_model", [
    ({"cap-diameter": 1200, "cap-shape": 2, "gill-attachment": 1, "gill-color": 3}, "naive_bayes_mushroom_classifier"),
    ({"stem-height": 10.5, "stem-width": 1300, "stem-color": 1.0, "season": 2.0}, "logistic_regression_mushroom_classifier")
])
def test_select_model(input_data, expected_model):
    assert select_model(input_data.keys()) == expected_model
