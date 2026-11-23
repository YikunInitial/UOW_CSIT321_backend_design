# Local and Enterprise LLM Integration

## Scope and safety boundary

The system can call an LLM hosted entirely inside the client's network. No
OpenAI account or public Internet connection is required. The connector is an
outbound adapter: the planning system calls the client's model endpoint.

Supported protocols:

| Provider value | Intended endpoint |
|---|---|
| `ollama` | Native Ollama `/api/chat` |
| `openai_compatible` | Any local server implementing `/v1/chat/completions` |
| `generic_rest` | Client-specific HTTP POST request and response structure |
| `disabled` | No model calls; deterministic processing only |

LLM output is recommendation data. It cannot release a loading plan, modify
hard constraints, or bypass the independent Validator.

The business UI can also request evidence-driven optional metadata repair. The
original file is immutable. Whitespace and formatting defects can be repaired
from the original value, while a missing factual field remains blank unless an
authoritative source is available. Identifiers, dimensions, weights,
destinations, and other loading facts are never guessed. The model may create a
non-operational product summary or propose a normalisation only when the
original field is present as direct evidence. A whitelist and grounding checks
reject all other output. Operators confirm selected suggestions through a
separate API call;
the accepted metadata is stored in `enrichment_confirmation.json`, while
`products.json` and `loading_solution.json` remain unchanged. Provider failure
produces a `safe_fallback` report and never fails the planning job.

```powershell
$env:CLIENT_ENABLE_OPTIONAL_ENRICHMENT = "true"
$env:CLIENT_OPTIONAL_ENRICHMENT_MAX_PRODUCTS = "5"
```

## Common configuration

Business data-recovery calls use the `CLIENT_LLM_` prefix. RAG semantic-planner
calls use the `LLM_` prefix. Both prefixes support the same variables:

| Suffix | Example | Description |
|---|---|---|
| `PROVIDER` | `generic_rest` | Protocol adapter |
| `BASE_URL` | `https://llm.internal:9443` | Internal service base URL |
| `MODEL` | `steel-qwen` | Model or deployment name |
| `API_KEY` | secret | Credential; never place in source control |
| `API_KEY_HEADER` | `Authorization` | Credential header |
| `API_KEY_PREFIX` | `Bearer` | Prefix before the credential; may be empty |
| `TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `MAX_RETRIES` | `1` | Retry count for network, HTTP 429, and HTTP 5xx errors |
| `ENDPOINT_PATH` | `api/inference` | Optional path appended to `BASE_URL` |
| `HEALTH_PATH` | `health` | Optional GET endpoint used by UI health status |
| `CA_BUNDLE` | `C:\certs\client-ca.pem` | Optional private certificate-authority bundle |
| `CLIENT_CERT` | `C:\certs\app.pem` | Optional mTLS client certificate |
| `CLIENT_KEY` | `C:\certs\app-key.pem` | Optional mTLS private key |
| `EXTRA_HEADERS_JSON` | `{"X-Site":"A"}` | Additional non-secret request headers |

Credentials must be injected by the operating-system service, container secret,
or enterprise secret manager. They are not displayed in the UI or written to
run manifests.

## Native Ollama example

```powershell
$env:CLIENT_LLM_PROVIDER = "ollama"
$env:CLIENT_LLM_BASE_URL = "http://10.20.30.15:11434"
$env:CLIENT_LLM_MODEL = "qwen2.5:7b-internal"
```

The connector sends `POST /api/chat` and checks `GET /api/tags`.

## Local OpenAI-compatible example

This protocol name describes the HTTP schema only. The model can be Qwen,
DeepSeek, Llama, or another model running on vLLM, LocalAI, LM Studio, or a
client platform inside the private network.

```powershell
$env:CLIENT_LLM_PROVIDER = "openai_compatible"
$env:CLIENT_LLM_BASE_URL = "https://model-gateway.internal/v1"
$env:CLIENT_LLM_MODEL = "steel-assistant"
$env:CLIENT_LLM_API_KEY = "<injected-secret>"
```

The connector sends `POST /v1/chat/completions` and expects content at
`choices[0].message.content`.

## Client-specific REST example

Suppose the client's endpoint requires:

```json
{
  "deployment": "steel-llm",
  "input": "combined prompt text",
  "conversation": [{"role": "user", "content": "..."}],
  "limit": 500
}
```

and returns:

```json
{
  "data": {
    "answer": {
      "field_mapping": {},
      "confidence": "high"
    }
  }
}
```

Configure:

```powershell
$env:CLIENT_LLM_PROVIDER = "generic_rest"
$env:CLIENT_LLM_BASE_URL = "https://model-gateway.internal"
$env:CLIENT_LLM_ENDPOINT_PATH = "api/inference"
$env:CLIENT_LLM_MODEL = "steel-llm"
$env:CLIENT_LLM_API_KEY_HEADER = "X-Client-Token"
$env:CLIENT_LLM_API_KEY_PREFIX = ""
$env:CLIENT_LLM_API_KEY = "<injected-secret>"
$env:CLIENT_LLM_RESPONSE_PATH = "data.answer"
$env:CLIENT_LLM_HEALTH_PATH = "health"
$env:CLIENT_LLM_REQUEST_TEMPLATE_JSON = '{"deployment":"{model}","input":"{prompt}","conversation":"{messages}","limit":"{max_tokens}"}'
```

Template tokens retain their JSON type when the entire value is one token:

- `{model}`
- `{prompt}`: system and user messages combined into text
- `{messages}`: structured message array
- `{temperature}`
- `{max_tokens}`
- `{num_ctx}`
- `{seed}`

`RESPONSE_PATH` is a dot-separated path. Numeric components address list
indexes, for example `choices.0.message.content`.

## RAG semantic-planner configuration

Use the same settings without the `CLIENT_` prefix:

```powershell
$env:LLM_PROVIDER = "generic_rest"
$env:LLM_BASE_URL = "https://model-gateway.internal"
$env:LLM_ENDPOINT_PATH = "api/inference"
$env:LLM_MODEL = "steel-llm"
$env:LLM_RESPONSE_PATH = "data.answer"
```

The RAG pipeline retrieves enterprise evidence locally, sends only bounded
evidence excerpts and product summaries, validates the returned strategy JSON
against the strategy whitelist, and falls back to deterministic policy if the
model is unavailable or returns invalid output.
The deployment planner forces offline embedding loading and copies the frozen
Chroma store into each run directory before opening it, preventing inference
sessions from mutating the master index.

For the business UI, enable this deployment-only path with:

```powershell
$env:CLIENT_ENABLE_SEMANTIC_LLM = "true"
$env:CLIENT_RAG_TOP_K = "6"
```

The extension is implemented in `Agents/provider_semantic_planner.py` and uses
the existing `semantic_strategy_provider` boundary. Frozen experimental
`LLMAgent` and `SemanticPlanner` source behavior remains unchanged.

## Information required from the client

Before connection testing, obtain:

1. network address, port, TLS and firewall route;
2. request method, headers and JSON body example;
3. successful response JSON example and response-content path;
4. authentication method and credential-rotation process;
5. health endpoint;
6. timeout, rate and concurrency limits;
7. maximum context and output sizes;
8. confirmation of which business fields may be sent to the model.

## Acceptance tests

1. Health endpoint succeeds from the deployed planning host.
2. A structured JSON inference succeeds without Internet access.
3. Invalid JSON triggers controlled deterministic fallback.
4. Timeout and HTTP 5xx errors do not expose credentials to the UI or logs.
5. Run audit records provider, model, status, latency and fallback reason.
6. The final plan still passes the independent Validator.
