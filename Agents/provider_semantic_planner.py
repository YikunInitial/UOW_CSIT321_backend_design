"""Deployment-only RAG planner using the provider-neutral LLM connector."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from Agents.llm_agent import LLMAgent
from Agents.llm_provider import HTTPModelProvider, create_llm_provider
from Agents.rag_agent import RAGAgent
from Agents.runtime_config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
from Agents.strategy_policy import validate_strategy_plan


class _ConfiguredProviderClient:
    """Adapts HTTPModelProvider to the frozen LLMAgent invoke boundary."""

    def __init__(
        self,
        provider: HTTPModelProvider,
        *,
        temperature: float,
        num_ctx: int,
        num_predict: int,
        seed: int,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed

    def invoke(self, messages: Iterable[Any]):
        return self.provider.invoke(
            messages,
            temperature=self.temperature,
            max_tokens=self.num_predict,
            num_ctx=self.num_ctx,
            seed=self.seed,
            json_mode=True,
        )


class ProviderLLMAgent(LLMAgent):
    """LLMAgent extension that leaves the frozen source file unchanged."""

    def __init__(
        self,
        *args: Any,
        provider_prefix: str = "LLM",
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.provider_prefix = provider_prefix
        self.requested_provider_name = provider_name
        super().__init__(*args, **kwargs)
        print(
            f"[ProviderLLMAgent] Active provider protocol: {self.provider_name}"
        )

    def _get_or_create_llm(self):
        provider = create_llm_provider(
            prefix=self.provider_prefix,
            provider=self.requested_provider_name,
            model=self.model_name,
            base_url=self.base_url,
            timeout_seconds=self.request_timeout_seconds,
        )
        self.provider_name = provider.provider_name
        return (
            _ConfiguredProviderClient(
                provider,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                num_predict=self.num_predict,
                seed=self.seed,
            ),
            False,
        )

    def run(self) -> Dict[str, Any]:
        strategy = super().run()
        metadata = strategy.setdefault("_metadata", {})
        legacy_metadata = metadata.get("ollama_response_metadata", {})
        metadata["llm_provider"] = self.provider_name
        metadata["provider_response_metadata"] = legacy_metadata
        self.export_json(strategy, self.output_path)
        return strategy


class ProviderSemanticPlanner:
    """RAG-to-strategy provider for client deployment configurations."""

    def __init__(
        self,
        knowledge_base_dir: str | Path = "KnowledgeBase",
        vector_store_dir: str | Path = "VectorDB/chroma_db",
        provider_prefix: str = "LLM",
        provider_name: Optional[str] = None,
        model_name: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_LLM_BASE_URL,
        embedding_model_name: str = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        top_k: int = 6,
        request_timeout_seconds: float = 60.0,
        seed: int = 42,
    ) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.provider_prefix = provider_prefix
        self.provider_name = provider_name
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_model_name = embedding_model_name
        self.top_k = max(1, int(top_k))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.seed = int(seed)

    def recommend(self, products_path: Path, run_dir: Path) -> Dict[str, Any]:
        retrieved_context_path = run_dir / "retrieved_context.json"
        strategy_path = run_dir / "semantic_strategy_plan.json"
        isolated_vector_store = (
            run_dir / f"provider_vector_store_{uuid4().hex[:8]}"
        )
        shutil.copytree(
            self.vector_store_dir,
            isolated_vector_store,
            dirs_exist_ok=False,
        )
        rag_result = RAGAgent(
            products_json_path=products_path,
            knowledge_base_dir=self.knowledge_base_dir,
            persist_directory=isolated_vector_store,
            embedding_model_name=self.embedding_model_name,
        ).run(
            output_path=retrieved_context_path,
            rebuild_vector_db=False,
            top_k=self.top_k,
        )
        strategy = ProviderLLMAgent(
            products_json_path=products_path,
            retrieved_context_path=retrieved_context_path,
            output_path=strategy_path,
            provider_prefix=self.provider_prefix,
            provider_name=self.provider_name,
            model_name=self.model_name,
            base_url=self.base_url,
            request_timeout_seconds=self.request_timeout_seconds,
            seed=self.seed,
        ).run()
        strategy = validate_strategy_plan(strategy)
        strategy["decision_source"] = "rag_grounded_provider_llm"
        strategy["semantic_planning"] = {
            "rag_output_path": str(retrieved_context_path),
            "strategy_output_path": str(strategy_path),
            "retrieved_document_count": len(
                rag_result.get("retrieved_documents", [])
            ),
            "top_k": self.top_k,
            "model_name": self.model_name,
            "provider": strategy.get("_metadata", {}).get("llm_provider"),
            "isolated_vector_store": str(isolated_vector_store),
        }
        return strategy
