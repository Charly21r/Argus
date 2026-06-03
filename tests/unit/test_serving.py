"""Tests for the serving API — schema validation and batch endpoint logic."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from pydantic import ValidationError
from src.serving.schemas import BatchTextRequest, LabelResult

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_batch_request_rejects_empty_list():
    with pytest.raises(ValidationError):
        BatchTextRequest(items=[])


def test_batch_request_rejects_oversized_batch():
    with pytest.raises(ValidationError):
        BatchTextRequest(items=[{"content": "x"} for _ in range(101)])


def test_batch_request_accepts_valid_payload():
    req = BatchTextRequest(items=[{"content": "hello"}, {"content": "world"}])
    assert len(req.items) == 2


def test_batch_request_item_inherits_text_input_limits():
    with pytest.raises(ValidationError):
        BatchTextRequest(items=[{"content": ""}])


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

FAKE_RESULT = (
    LabelResult(prob=0.1, flagged=False),
    LabelResult(prob=0.05, flagged=False),
)


@pytest.fixture()
def client():
    with patch("src.serving.model_manager.load_model"), patch("src.serving.tracing.setup_tracing"):
        from src.serving.app import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def client_no_raise():
    with patch("src.serving.model_manager.load_model"), patch("src.serving.tracing.setup_tracing"):
        from src.serving.app import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _counter_delta(endpoint: str, status_code: str, before: float) -> float:
    after = (
        REGISTRY.get_sample_value("moderation_requests_total", {"endpoint": endpoint, "status_code": status_code})
        or 0.0
    )
    return after - before


def test_batch_endpoint_happy_path(client):
    mock_results = [FAKE_RESULT, FAKE_RESULT]
    with patch("src.serving.app.predict_batch", return_value=mock_results) as mock_pb:
        resp = client.post(
            "/v1/moderate/text/batch",
            json={
                "items": [
                    {"content": "hello world"},
                    {"content": "how are you"},
                ]
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert "total_processing_time_ms" in body
    mock_pb.assert_called_once_with(["hello world", "how are you"])


def test_batch_endpoint_returns_correct_ids(client):
    mock_results = [FAKE_RESULT]
    with patch("src.serving.app.predict_batch", return_value=mock_results):
        resp = client.post("/v1/moderate/text/batch", json={"items": [{"id": "abc-123", "content": "test text"}]})

    assert resp.json()["items"][0]["id"] == "abc-123"


def test_batch_endpoint_flags_toxic_item(client):
    toxic_result = (
        LabelResult(prob=0.95, flagged=True),
        LabelResult(prob=0.05, flagged=False),
    )
    with patch("src.serving.app.predict_batch", return_value=[toxic_result]):
        resp = client.post("/v1/moderate/text/batch", json={"items": [{"content": "you are disgusting trash"}]})

    item = resp.json()["items"][0]
    assert item["safe"] is False
    assert item["toxicity"]["flagged"] is True
    assert item["hate"]["flagged"] is False


def test_batch_endpoint_rejects_empty_items(client):
    resp = client.post("/v1/moderate/text/batch", json={"items": []})
    assert resp.status_code == 422


def test_batch_endpoint_rejects_oversized_batch(client):
    resp = client.post("/v1/moderate/text/batch", json={"items": [{"content": "x"} for _ in range(101)]})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# REQUEST_COUNT metric correctness
# ---------------------------------------------------------------------------


def test_single_endpoint_increments_200_counter(client):
    endpoint = "/v1/moderate/text"
    before = REGISTRY.get_sample_value("moderation_requests_total", {"endpoint": endpoint, "status_code": "200"}) or 0.0
    with patch("src.serving.app.predict", return_value=FAKE_RESULT):
        resp = client.post("/v1/moderate/text", json={"content": "hello world"})
    assert resp.status_code == 200
    assert _counter_delta(endpoint, "200", before) == 1.0


def test_single_endpoint_increments_500_counter_on_predict_error(client_no_raise):
    endpoint = "/v1/moderate/text"
    before = REGISTRY.get_sample_value("moderation_requests_total", {"endpoint": endpoint, "status_code": "500"}) or 0.0
    with patch("src.serving.app.predict", side_effect=RuntimeError("model crash")):
        resp = client_no_raise.post("/v1/moderate/text", json={"content": "hello world"})
    assert resp.status_code == 500
    assert _counter_delta(endpoint, "500", before) == 1.0


def test_single_endpoint_does_not_increment_200_on_error(client_no_raise):
    endpoint = "/v1/moderate/text"
    before = REGISTRY.get_sample_value("moderation_requests_total", {"endpoint": endpoint, "status_code": "200"}) or 0.0
    with patch("src.serving.app.predict", side_effect=RuntimeError("model crash")):
        client_no_raise.post("/v1/moderate/text", json={"content": "hello world"})
    assert _counter_delta(endpoint, "200", before) == 0.0


def test_batch_endpoint_increments_200_counter(client):
    endpoint = "/v1/moderate/text/batch"
    before = REGISTRY.get_sample_value("moderation_requests_total", {"endpoint": endpoint, "status_code": "200"}) or 0.0
    with patch("src.serving.app.predict_batch", return_value=[FAKE_RESULT]):
        resp = client.post("/v1/moderate/text/batch", json={"items": [{"content": "hello world"}]})
    assert resp.status_code == 200
    assert _counter_delta(endpoint, "200", before) == 1.0


def test_batch_endpoint_increments_500_counter_on_predict_error(client_no_raise):
    endpoint = "/v1/moderate/text/batch"
    before = REGISTRY.get_sample_value("moderation_requests_total", {"endpoint": endpoint, "status_code": "500"}) or 0.0
    with patch("src.serving.app.predict_batch", side_effect=RuntimeError("model crash")):
        resp = client_no_raise.post("/v1/moderate/text/batch", json={"items": [{"content": "hello world"}]})
    assert resp.status_code == 500
    assert _counter_delta(endpoint, "500", before) == 1.0
