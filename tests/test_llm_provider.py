import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

from Agents.llm_provider import (
    HTTPModelProvider,
    LLMProviderConfig,
    LLMProviderError,
    reset_circuit_breakers,
)
from Agents.input_service import infer_product_field_mapping
from Agents.provider_semantic_planner import ProviderLLMAgent


class _ModelHandler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(
            {
                "path": self.path,
                "payload": payload,
                "authorization": self.headers.get("Authorization"),
                "client_token": self.headers.get("X-Client-Token"),
            }
        )
        if self.path == "/api/chat":
            response = {
                "message": {"content": '{"source":"ollama"}'},
                "eval_count": 5,
            }
        elif self.path == "/v1/chat/completions":
            response = {
                "choices": [{"message": {"content": '{"source":"compatible"}'}}],
                "usage": {"total_tokens": 12},
            }
        elif self.path == "/mapping":
            response = {
                "data": {
                    "answer": {
                        "field_mapping": {
                            "material_no": "code",
                            "thickness": "t",
                            "width": "w",
                            "outer_diameter": "d",
                            "gross_weight_tonne": "gross",
                            "net_weight_tonne": "net",
                            "destination": "dest",
                        },
                        "confidence": "high",
                        "reason": "mock mapping",
                    }
                }
            }
        elif self.path == "/strategy":
            response = {
                "data": {
                    "answer": {
                        "selected_strategy": "balance_aware_greedy",
                        "optimisation_config": {
                            "group_by_destination": False,
                            "sort_by_weight_desc": True,
                            "use_balance_aware_scoring": True,
                            "use_simulated_annealing": False,
                            "use_rule_based_constraints": True,
                        },
                        "required_constraints": [
                            "weight_limit",
                            "boundary_check",
                            "overlap_check",
                        ],
                        "next_agent": "optimisation_agent",
                    }
                }
            }
        else:
            response = {"data": {"answer": {"source": "custom"}}}
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _model_server():
    _ModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_ollama_protocol_and_health() -> None:
    with _model_server() as base_url:
        provider = HTTPModelProvider(
            LLMProviderConfig(
                provider="ollama",
                base_url=base_url,
                model="local-model",
            )
        )
        assert provider.check_health()
        response = provider.invoke([{"role": "user", "content": "test"}])

    assert json.loads(response.content) == {"source": "ollama"}
    assert response.response_metadata["provider"] == "ollama"
    assert _ModelHandler.requests[0]["path"] == "/api/chat"
    assert _ModelHandler.requests[0]["payload"]["model"] == "local-model"


def test_openai_compatible_protocol_is_vendor_neutral() -> None:
    with _model_server() as base_url:
        provider = HTTPModelProvider(
            LLMProviderConfig(
                provider="openai_compatible",
                base_url=base_url,
                model="enterprise-deployment",
                api_key="secret-value",
            )
        )
        response = provider.invoke([{"role": "user", "content": "test"}])

    assert json.loads(response.content) == {"source": "compatible"}
    request = _ModelHandler.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer secret-value"


def test_generic_rest_supports_custom_request_and_response_shapes() -> None:
    with _model_server() as base_url:
        provider = HTTPModelProvider(
            LLMProviderConfig(
                provider="generic_rest",
                base_url=base_url,
                endpoint_path="private/inference",
                model="steel-llm",
                api_key="internal-token",
                api_key_header="X-Client-Token",
                api_key_prefix="",
                request_template={
                    "deployment": "{model}",
                    "input": "{prompt}",
                    "conversation": "{messages}",
                    "limit": "{max_tokens}",
                },
                response_path="data.answer",
            )
        )
        response = provider.invoke(
            [
                {"role": "system", "content": "JSON only"},
                {"role": "user", "content": "test"},
            ],
            max_tokens=321,
        )

    assert json.loads(response.content) == {"source": "custom"}
    request = _ModelHandler.requests[0]
    assert request["path"] == "/private/inference"
    assert request["client_token"] == "internal-token"
    assert request["payload"]["deployment"] == "steel-llm"
    assert request["payload"]["limit"] == 321
    assert request["payload"]["conversation"][1]["content"] == "test"


def test_business_field_mapping_uses_generic_provider(monkeypatch) -> None:
    with _model_server() as base_url:
        monkeypatch.setenv("CLIENT_LLM_ENDPOINT_PATH", "mapping")
        monkeypatch.setenv("CLIENT_LLM_RESPONSE_PATH", "data.answer")
        dataframe = pd.DataFrame(
            [
                {
                    "code": "P1",
                    "t": 1.2,
                    "w": 1000,
                    "d": 1600,
                    "gross": 20,
                    "net": 19.8,
                    "dest": "A",
                }
            ]
        )
        result = infer_product_field_mapping(
            dataframe,
            model_name="client-model",
            base_url=base_url,
            provider_name="generic_rest",
        )

    assert result["field_mapping"]["material_no"] == "code"
    assert result["_provider_metadata"]["provider"] == "generic_rest"


def test_frozen_llm_agent_can_use_deployment_provider_extension(
    monkeypatch,
    tmp_path,
) -> None:
    with _model_server() as base_url:
        monkeypatch.setenv("LLM_ENDPOINT_PATH", "strategy")
        monkeypatch.setenv("LLM_RESPONSE_PATH", "data.answer")
        agent = ProviderLLMAgent(
            products_json_path=tmp_path / "products.json",
            retrieved_context_path=tmp_path / "context.json",
            output_path=tmp_path / "strategy.json",
            provider_prefix="LLM",
            provider_name="generic_rest",
            model_name="client-model",
            base_url=base_url,
        )
        raw, metadata = agent.call_llm("Select a valid strategy")

    assert json.loads(raw)["selected_strategy"] == "balance_aware_greedy"
    assert metadata["provider"] == "generic_rest"


def test_circuit_breaker_stops_repeated_calls_to_unavailable_model() -> None:
    reset_circuit_breakers()
    provider = HTTPModelProvider(
        LLMProviderConfig(
            provider="ollama",
            base_url="http://127.0.0.1:1",
            model="unavailable-model",
            timeout_seconds=0.1,
            max_retries=0,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown_seconds=60,
        )
    )
    codes = []
    for _ in range(3):
        try:
            provider.invoke([{"role": "user", "content": "test"}])
        except LLMProviderError as error:
            codes.append(error.code)
        else:
            raise AssertionError("unavailable provider unexpectedly succeeded")

    assert codes[:2] == ["connection_error", "connection_error"]
    assert codes[2] == "circuit_open"
    reset_circuit_breakers()
