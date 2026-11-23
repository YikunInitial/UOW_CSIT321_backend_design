"""Concrete RAG-to-LLM bounded semantic strategy provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from Agents.llm_agent import LLMAgent
from Agents.rag_agent import RAGAgent
from Agents.strategy_policy import validate_strategy_plan


class SemanticPlanner:
    def __init__(
        self,
        knowledge_base_dir: str | Path = "KnowledgeBase",
        vector_store_dir: str | Path = "VectorDB/chroma_db",
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        embedding_model_name: str = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        top_k: int = 6,
        request_timeout_seconds: float = 60.0,
        seed: int = 42,
    ) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_model_name = embedding_model_name
        self.top_k = max(1, int(top_k))
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.seed = int(seed)

    def recommend(
        self,
        products_path: Path,
        run_dir: Path,
    ) -> Dict[str, Any]:
        retrieved_context_path = run_dir / "retrieved_context.json"
        strategy_path = run_dir / "semantic_strategy_plan.json"
        rag_result = RAGAgent(
            products_json_path=products_path,
            knowledge_base_dir=self.knowledge_base_dir,
            persist_directory=self.vector_store_dir,
            embedding_model_name=self.embedding_model_name,
        ).run(
            output_path=retrieved_context_path,
            rebuild_vector_db=False,
            top_k=self.top_k,
        )
        strategy = LLMAgent(
            products_json_path=products_path,
            retrieved_context_path=retrieved_context_path,
            output_path=strategy_path,
            model_name=self.model_name,
            base_url=self.base_url,
            request_timeout_seconds=self.request_timeout_seconds,
            seed=self.seed,
        ).run()
        strategy = validate_strategy_plan(strategy)
        strategy["decision_source"] = "rag_grounded_llm"
        strategy["semantic_planning"] = {
            "rag_output_path": str(retrieved_context_path),
            "strategy_output_path": str(strategy_path),
            "retrieved_document_count": len(
                rag_result.get("retrieved_documents", [])
            ),
            "top_k": self.top_k,
            "model_name": self.model_name,
        }
        return strategy
