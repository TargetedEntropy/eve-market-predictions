"""API contract tests (issues 1.3, 1.4, 7.2, 7.5, 9.1).

The app is exercised without a database: startup records the DB failure and
/health reports it, which is exactly the degraded path we want covered.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from src.api.main import app
from src.models.artifacts import ScalerParams, save_artifacts
from src.models.lstm_model import MarketLSTM
from src.models.pipeline import FEATURE_COLUMNS


@pytest.fixture
def client(isolated_model_dir):
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_root_reports_version(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["version"] == "0.2.0"


def test_health_returns_503_when_no_model_is_servable(client):
    """Issue 9.1: /health used to return 200 with status='degraded'."""
    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["models_loaded"] == 0


def test_health_body_shape(client):
    body = client.get("/health").json()
    assert set(body) >= {"status", "models_loaded", "timestamp", "device", "version", "errors"}


def test_predict_returns_503_when_the_item_has_no_model(client):
    response = client.post("/predict", json={"type_id": 44992, "horizon_days": 1})

    assert response.status_code == 503
    assert "train" in response.json()["detail"].lower()


def test_predict_for_an_untrained_item_does_not_serve_another_items_model(
    client, isolated_model_dir, price_series, make_metadata
):
    """Issue 1.5: one shared best_model.pth was served for every type_id."""
    scaler = MinMaxScaler().fit(price_series.reshape(-1, 1))
    metadata = make_metadata(type_id=44992)
    metadata.scaler = ScalerParams.from_scaler(scaler)
    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)
    save_artifacts(model.state_dict(), metadata)

    # 34 (Tritanium) has no model of its own -> 503, not PLEX's model.
    assert client.post("/predict", json={"type_id": 34}).status_code == 503


def test_predict_validates_input(client):
    assert client.post("/predict", json={"type_id": 0}).status_code == 422
    assert client.post("/predict", json={"type_id": 44992,
                                         "horizon_days": 0}).status_code == 422
    assert client.post("/predict", json={"type_id": 44992,
                                         "horizon_days": 500}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_predict_request_uses_days_not_hours(client):
    """The model is daily; horizon_hours was validated, echoed and ignored."""
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    properties = schema["PredictionRequest"]["properties"]
    assert "horizon_days" in properties
    assert "horizon_hours" not in properties


def test_models_endpoint_lists_trained_artifacts(
    client, isolated_model_dir, price_series, make_metadata
):
    assert client.get("/models").json() == []

    metadata = make_metadata(type_id=44992, version="plex-42")
    metadata.scaler = ScalerParams.from_scaler(
        MinMaxScaler().fit(price_series.reshape(-1, 1))
    )
    save_artifacts(
        MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0).state_dict(),
        metadata,
    )

    listed = client.get("/models").json()
    assert len(listed) == 1
    assert listed[0]["type_id"] == 44992
    assert listed[0]["item_name"] == "PLEX"
    assert listed[0]["model_version"] == "plex-42"
    assert listed[0]["feature_columns"] == list(FEATURE_COLUMNS)


def test_cors_is_not_a_wildcard_with_credentials():
    """Issue 7.2 -- assert on the middleware the app actually installed."""
    cors = [m for m in app.user_middleware if "CORSMiddleware" in str(m.cls)]
    assert cors, "CORS middleware should be installed"
    options = cors[0].kwargs
    assert not (options.get("allow_credentials") and "*" in options.get("allow_origins", []))


def test_error_responses_do_not_leak_internals(client, monkeypatch):
    """Issue 7.5: detail=str(e) exposed internals; 500s must be generic."""
    async def boom(*args, **kwargs):
        raise RuntimeError("secret connection string postgresql://user:pw@host/db")

    monkeypatch.setattr("src.api.main.generate_predictions", boom)

    response = client.post("/predict", json={"type_id": 44992})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "postgresql" not in response.text


def test_prediction_response_schema_is_honest(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    properties = schema["PredictionResponse"]["properties"]
    for field in ("model_version", "confidence", "daily", "item_name", "horizon_days"):
        assert field in properties


def test_untrained_model_does_not_leak_scaled_values(
    isolated_model_dir, price_series, make_metadata
):
    """A raw 0-1 model output must never be reported as an ISK price."""
    from src.models.pipeline import forecast_from_prices

    metadata = make_metadata()
    metadata.scaler = ScalerParams.from_scaler(MinMaxScaler().fit(price_series.reshape(-1, 1)))
    model = MarketLSTM(input_size=1, hidden_size=16, num_layers=1, dropout=0.0)

    forecast = forecast_from_prices(model, metadata, price_series, 3)
    assert np.all(forecast > 1000), "output should be ISK-scale, not 0-1"
