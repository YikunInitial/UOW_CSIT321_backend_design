"""Central runtime defaults shared by research and client entry points.

Environment-specific settings may override these values at the application
boundary.  Domain and orchestration modules import constants from here so a
run cannot silently report a different model from the one configured for the
deployment.
"""

from __future__ import annotations

DEFAULT_LLM_MODEL = "qwen2.5:0.5b"
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_RAG_TOP_K = 6
DEFAULT_RANDOM_SEED = 42
