"""Provider-neutral HTTP connector for local and enterprise LLM services."""

from __future__ import annotations

import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from Agents.runtime_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)

SUPPORTED_PROVIDERS = {
    "disabled",
    "ollama",
    "openai_compatible",
    "generic_rest",
}


class LLMProviderError(RuntimeError):
    """Structured connector error that never contains credentials."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class LLMProviderResponse:
    content: str
    response_metadata: Dict[str, Any]


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str = DEFAULT_LLM_PROVIDER
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    api_key: str = ""
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    timeout_seconds: float = 60.0
    max_retries: int = 1
    endpoint_path: str = ""
    health_path: str = ""
    response_path: str = "result"
    request_template: Optional[Dict[str, Any]] = None
    extra_headers: Optional[Dict[str, str]] = None
    ca_bundle: str = ""
    client_cert: str = ""
    client_key: str = ""
    circuit_breaker_threshold: int = 3
    circuit_breaker_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"Expected one of {sorted(SUPPORTED_PROVIDERS)}."
            )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "timeout_seconds", max(0.1, self.timeout_seconds))
        object.__setattr__(self, "max_retries", max(0, self.max_retries))
        object.__setattr__(
            self,
            "circuit_breaker_threshold",
            max(1, int(self.circuit_breaker_threshold)),
        )
        object.__setattr__(
            self,
            "circuit_breaker_cooldown_seconds",
            max(0.1, float(self.circuit_breaker_cooldown_seconds)),
        )


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None
    half_open_in_flight: bool = False


_CIRCUIT_LOCK = threading.Lock()
_CIRCUITS: Dict[str, _CircuitState] = {}


def reset_circuit_breakers() -> None:
    with _CIRCUIT_LOCK:
        _CIRCUITS.clear()


def _env(prefix: str, name: str, default: str = "") -> str:
    return os.getenv(f"{prefix}_{name}", os.getenv(f"LLM_{name}", default))


def _json_object_from_env(value: str, field_name: str) -> Dict[str, Any]:
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{field_name} must contain valid JSON.") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must contain a JSON object.")
    return parsed


def provider_config_from_env(
    *,
    prefix: str = "CLIENT_LLM",
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> LLMProviderConfig:
    request_template = _json_object_from_env(
        _env(prefix, "REQUEST_TEMPLATE_JSON"),
        f"{prefix}_REQUEST_TEMPLATE_JSON",
    )
    extra_headers = _json_object_from_env(
        _env(prefix, "EXTRA_HEADERS_JSON"),
        f"{prefix}_EXTRA_HEADERS_JSON",
    )
    return LLMProviderConfig(
        provider=provider or _env(prefix, "PROVIDER", DEFAULT_LLM_PROVIDER),
        base_url=base_url or _env(prefix, "BASE_URL", DEFAULT_LLM_BASE_URL),
        model=model or _env(prefix, "MODEL", DEFAULT_LLM_MODEL),
        api_key=_env(prefix, "API_KEY"),
        api_key_header=_env(prefix, "API_KEY_HEADER", "Authorization"),
        api_key_prefix=_env(prefix, "API_KEY_PREFIX", "Bearer"),
        timeout_seconds=(
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(_env(prefix, "TIMEOUT_SECONDS", "60"))
        ),
        max_retries=int(_env(prefix, "MAX_RETRIES", "1")),
        endpoint_path=_env(prefix, "ENDPOINT_PATH"),
        health_path=_env(prefix, "HEALTH_PATH"),
        response_path=_env(prefix, "RESPONSE_PATH", "result"),
        request_template=request_template or None,
        extra_headers={str(key): str(value) for key, value in extra_headers.items()}
        or None,
        ca_bundle=_env(prefix, "CA_BUNDLE"),
        client_cert=_env(prefix, "CLIENT_CERT"),
        client_key=_env(prefix, "CLIENT_KEY"),
        circuit_breaker_threshold=int(_env(prefix, "CIRCUIT_BREAKER_THRESHOLD", "3")),
        circuit_breaker_cooldown_seconds=float(
            _env(prefix, "CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30")
        ),
    )


def _normalise_messages(messages: Iterable[Any]) -> List[Dict[str, str]]:
    normalised: List[Dict[str, str]] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = str(message.get("role", "user"))
            content = message.get("content", "")
        else:
            class_name = message.__class__.__name__.lower()
            role = "system" if "system" in class_name else "user"
            if "assistant" in class_name or "ai" == class_name:
                role = "assistant"
            content = getattr(message, "content", str(message))
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        normalised.append({"role": role, "content": content})
    return normalised


def _replace_template_tokens(
    value: Any,
    replacements: Mapping[str, Any],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_template_tokens(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_template_tokens(item, replacements) for item in value]
    if isinstance(value, str):
        if value in replacements:
            return deepcopy(replacements[value])
        rendered = value
        for token, replacement in replacements.items():
            if isinstance(replacement, (str, int, float, bool)):
                rendered = rendered.replace(token, str(replacement))
        return rendered
    return value


def _extract_path(payload: Any, path: str) -> Any:
    current = payload
    if not path:
        return current
    for component in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(component)]
            except (ValueError, IndexError) as error:
                raise LLMProviderError(
                    f"Response path '{path}' was not found.",
                    code="invalid_response_path",
                ) from error
        elif isinstance(current, dict) and component in current:
            current = current[component]
        else:
            raise LLMProviderError(
                f"Response path '{path}' was not found.",
                code="invalid_response_path",
            )
    return current


class HTTPModelProvider:
    """Calls an LLM endpoint without depending on a vendor SDK."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    @property
    def provider_name(self) -> str:
        return self.config.provider

    def _circuit_key(self) -> str:
        return f"{self.config.provider}|{self.config.base_url}|{self.config.model}"

    def _before_request(self) -> None:
        key = self._circuit_key()
        now = time.monotonic()
        with _CIRCUIT_LOCK:
            state = _CIRCUITS.setdefault(key, _CircuitState())
            if state.opened_at is None:
                return
            elapsed = now - state.opened_at
            if elapsed < self.config.circuit_breaker_cooldown_seconds:
                raise LLMProviderError(
                    "LLM circuit breaker is open; deterministic fallback should be used.",
                    code="circuit_open",
                    retryable=True,
                )
            if state.half_open_in_flight:
                raise LLMProviderError(
                    "LLM circuit breaker probe is already in progress.",
                    code="circuit_half_open",
                    retryable=True,
                )
            state.half_open_in_flight = True

    def _record_success(self) -> None:
        with _CIRCUIT_LOCK:
            _CIRCUITS[self._circuit_key()] = _CircuitState()

    def _record_transient_failure(self) -> None:
        with _CIRCUIT_LOCK:
            state = _CIRCUITS.setdefault(self._circuit_key(), _CircuitState())
            state.half_open_in_flight = False
            state.consecutive_failures += 1
            if state.consecutive_failures >= self.config.circuit_breaker_threshold:
                state.opened_at = time.monotonic()

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if self.config.endpoint_path:
            return base + "/" + self.config.endpoint_path.lstrip("/")
        if self.config.provider == "ollama":
            return base if base.endswith("/api/chat") else base + "/api/chat"
        if self.config.provider == "openai_compatible":
            if base.endswith("/chat/completions"):
                return base
            if base.endswith("/v1"):
                return base + "/chat/completions"
            return base + "/v1/chat/completions"
        return base

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        }
        headers.update(self.config.extra_headers or {})
        if self.config.api_key:
            credential = self.config.api_key
            if self.config.api_key_prefix:
                credential = f"{self.config.api_key_prefix} {credential}"
            headers[self.config.api_key_header] = credential
        return headers

    def _ssl_context(self) -> ssl.SSLContext:
        if self.config.ca_bundle:
            context = ssl.create_default_context(
                cafile=str(Path(self.config.ca_bundle))
            )
        else:
            context = ssl.create_default_context()
        if self.config.client_cert:
            context.load_cert_chain(
                certfile=str(Path(self.config.client_cert)),
                keyfile=(
                    str(Path(self.config.client_key))
                    if self.config.client_key
                    else None
                ),
            )
        return context

    def _request_payload(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        num_ctx: int,
        seed: int,
        json_mode: bool,
    ) -> Dict[str, Any]:
        if self.config.provider == "ollama":
            return {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "format": "json" if json_mode else "",
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": max_tokens,
                    "seed": seed,
                },
            }
        if self.config.provider == "openai_compatible":
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            return payload

        prompt = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        )
        replacements = {
            "{model}": self.config.model,
            "{prompt}": prompt,
            "{messages}": messages,
            "{temperature}": temperature,
            "{max_tokens}": max_tokens,
            "{num_ctx}": num_ctx,
            "{seed}": seed,
        }
        if self.config.request_template:
            return _replace_template_tokens(
                self.config.request_template,
                replacements,
            )
        return {
            "model": self.config.model,
            "messages": messages,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

    def invoke(
        self,
        messages: Iterable[Any],
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
        num_ctx: int = 8192,
        seed: int = 42,
        json_mode: bool = True,
    ) -> LLMProviderResponse:
        if self.config.provider == "disabled":
            raise LLMProviderError(
                "LLM provider is disabled.",
                code="provider_disabled",
            )
        self._before_request()
        normalised = _normalise_messages(messages)
        payload = self._request_payload(
            normalised,
            temperature=temperature,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            seed=seed,
            json_mode=json_mode,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        endpoint = self._endpoint()
        attempts = self.config.max_retries + 1
        started = time.perf_counter()
        last_error: Optional[LLMProviderError] = None

        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.timeout_seconds,
                    context=self._ssl_context(),
                ) as response:
                    raw_body = response.read().decode("utf-8")
                    status_code = response.status
                parsed = json.loads(raw_body)
                if self.config.provider == "ollama":
                    content = _extract_path(parsed, "message.content")
                elif self.config.provider == "openai_compatible":
                    content = _extract_path(parsed, "choices.0.message.content")
                else:
                    content = _extract_path(parsed, self.config.response_path)
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                elapsed = time.perf_counter() - started
                metadata = {
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "status_code": status_code,
                    "attempt_count": attempt,
                    "elapsed_seconds": round(elapsed, 4),
                }
                if isinstance(parsed, dict):
                    for key in (
                        "usage",
                        "prompt_eval_count",
                        "eval_count",
                        "load_duration",
                        "prompt_eval_duration",
                        "eval_duration",
                        "total_duration",
                    ):
                        if key in parsed:
                            metadata[key] = parsed[key]
                self._record_success()
                return LLMProviderResponse(content=content, response_metadata=metadata)
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                last_error = LLMProviderError(
                    f"LLM service returned HTTP {error.code}.",
                    code="http_error",
                    retryable=retryable,
                    status_code=error.code,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = LLMProviderError(
                    f"LLM service is unavailable: {type(error).__name__}.",
                    code="connection_error",
                    retryable=True,
                )
            except json.JSONDecodeError as error:
                self._record_success()
                raise LLMProviderError(
                    "LLM service returned non-JSON data.",
                    code="invalid_json_response",
                ) from error
            except LLMProviderError:
                self._record_success()
                raise

            if not last_error.retryable or attempt >= attempts:
                if last_error.retryable:
                    self._record_transient_failure()
                else:
                    self._record_success()
                raise last_error
            time.sleep(min(0.25 * (2 ** (attempt - 1)), 1.0))

        raise last_error or LLMProviderError(
            "LLM request failed.",
            code="unknown_error",
        )

    def check_health(self, timeout_seconds: float = 1.0) -> bool:
        if self.config.provider == "disabled":
            return False
        base = self.config.base_url.rstrip("/")
        path = self.config.health_path
        if not path:
            if self.config.provider == "ollama":
                path = "/api/tags"
            elif self.config.provider == "openai_compatible":
                path = "/models" if base.endswith("/v1") else "/v1/models"
            else:
                return False
        endpoint = base + "/" + path.lstrip("/")
        request = urllib.request.Request(
            endpoint, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=max(0.1, timeout_seconds),
                context=self._ssl_context(),
            ) as response:
                return 200 <= response.status < 400
        except Exception:
            return False


def create_llm_provider(
    *,
    prefix: str = "CLIENT_LLM",
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
) -> HTTPModelProvider:
    return HTTPModelProvider(
        provider_config_from_env(
            prefix=prefix,
            provider=provider,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    )
