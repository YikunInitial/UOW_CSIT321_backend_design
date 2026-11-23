import json
import urllib.error

import pytest

from Agents.backend_client import BackendAPIError, SteelLoadingAPIClient


class FakeResponse:
    status = 200
    headers = {"x-request-id": "server-request", "x-response-time-ms": "12.5"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    @staticmethod
    def read() -> bytes:
        return b'{"status":"ok"}'


def test_client_preserves_real_request_and_transport_evidence(monkeypatch) -> None:
    captured = {}

    def open_request(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    client = SteelLoadingAPIClient(
        "http://127.0.0.1:8080",
        api_key="secret",
        user_id="operator-1",
    )

    response = client.submit_plan(
        {"products": [{"material_no": "A"}]},
        idempotency_key="planning-12345678",
    )

    assert captured["url"].endswith("/v1/planning-runs")
    assert captured["headers"]["X-api-key"] == "secret"
    assert captured["headers"]["Idempotency-key"] == "planning-12345678"
    assert captured["body"]["products"][0]["material_no"] == "A"
    assert response.request_id == "server-request"
    assert response.response_time_ms == 12.5


def test_client_exposes_structured_backend_error_without_html(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1:8080/v1/planning-runs",
        503,
        "unavailable",
        {"x-request-id": "request-503", "retry-after": "30"},
        None,
    )
    error.read = lambda: (  # type: ignore[method-assign]
        b'{"error":{"code":"QUEUE_CAPACITY_REACHED","message":"Queue full"}}'
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    client = SteelLoadingAPIClient("http://127.0.0.1:8080")

    with pytest.raises(BackendAPIError) as raised:
        client.submit_plan({"products": [{}]}, idempotency_key="planning-12345678")

    assert raised.value.status_code == 503
    assert raised.value.code == "QUEUE_CAPACITY_REACHED"
    assert raised.value.request_id == "request-503"
    assert raised.value.retry_after == "30"


def test_client_confirms_only_selected_enrichment_suggestions(monkeypatch) -> None:
    captured = {}

    def open_request(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    client = SteelLoadingAPIClient("http://127.0.0.1:8080", user_id="operator-1")

    client.confirm_enrichment(
        "plan-1",
        2,
        ["enr_1"],
        confirmed_by="operator-1",
    )

    assert captured["url"].endswith(
        "/v1/plans/plan-1/enrichment-confirmations"
    )
    assert captured["body"] == {
        "expected_version": 2,
        "suggestion_ids": ["enr_1"],
        "confirmed_by": "operator-1",
    }
