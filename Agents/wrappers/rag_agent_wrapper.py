from pathlib import Path
from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.task import AgentTask
from Agents.rag_agent import RAGAgent


class RAGAgentWrapper(BaseAgent):
    """
    Managed wrapper for the existing Agents/rag_agent.py.

    This wrapper does not replace the original RAGAgent.
    It only adds:
    - agent_id
    - status tracking
    - task tracking
    - BDI fields
    - standard act(task) interface
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="rag",
            name="RAG Agent Wrapper",
            config=config,
        )

    def act(self, task: AgentTask) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(task.task_id)

        try:
            products_json_path = self._resolve_products_json_path(task)
            output_path = self._resolve_output_path(task)
            knowledge_base_dir = self._resolve_knowledge_base_dir(task)
            persist_directory = self._resolve_persist_directory(task)

            rebuild_vector_db = self._resolve_rebuild_vector_db(task)
            top_k = self._resolve_top_k(task)

            self.update_belief("task_id", task.task_id)
            self.update_belief("products_json_path", str(products_json_path))
            self.update_belief("retrieved_context_output_path", str(output_path))
            self.update_belief("knowledge_base_dir", str(knowledge_base_dir))
            self.update_belief("persist_directory", str(persist_directory))
            self.update_belief("rebuild_vector_db", rebuild_vector_db)
            self.update_belief("top_k", top_k)
            self.update_belief("objective", task.objective)
            self.update_belief("constraints", task.constraints)

            self.add_desire("retrieve_relevant_domain_context")
            self.add_intention("call_existing_RAGAgent_run")

            rag_agent = RAGAgent(
                products_json_path=products_json_path,
                knowledge_base_dir=knowledge_base_dir,
                persist_directory=persist_directory,
                collection_name=self.config.get(
                    "collection_name",
                    task.metadata.get("collection_name", "baosteel_loading_rag"),
                ),
                embedding_model_name=self.config.get(
                    "embedding_model_name",
                    task.metadata.get(
                        "embedding_model_name",
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    ),
                ),
                device=self.config.get(
                    "device",
                    task.metadata.get("device", "cpu"),
                ),
                chunk_size=self.config.get(
                    "chunk_size",
                    task.metadata.get("chunk_size", 800),
                ),
                chunk_overlap=self.config.get(
                    "chunk_overlap",
                    task.metadata.get("chunk_overlap", 120),
                ),
            )

            result_data = rag_agent.run(
                output_path=output_path,
                rebuild_vector_db=rebuild_vector_db,
                top_k=top_k,
            )

            retrieved_documents = result_data.get("retrieved_documents", [])

            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "completed",
                "message": "RAGAgentWrapper completed successfully.",
                "products_json_path": str(products_json_path),
                "retrieved_context_output_path": str(output_path),
                "knowledge_base_dir": str(knowledge_base_dir),
                "persist_directory": str(persist_directory),
                "retrieved_document_count": len(retrieved_documents),
                "recommended_context_preview": result_data.get(
                    "recommended_context", ""
                )[:1000],
            }

            self.update_belief("retrieved_document_count", len(retrieved_documents))
            self.update_status("completed")
            self.clear_current_task()

            return result

        except Exception as e:
            self.update_belief("error", str(e))
            self.update_status("failed")
            self.clear_current_task()

            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "failed",
                "message": "RAGAgentWrapper failed.",
                "error": str(e),
            }

    def _resolve_products_json_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["products_json_path"]
        2. task.metadata["json_output_path"]
        3. self.config["products_json_path"]
        4. Agents/products.json
        """

        if task.metadata.get("products_json_path"):
            return Path(task.metadata["products_json_path"])

        if task.metadata.get("json_output_path"):
            return Path(task.metadata["json_output_path"])

        if self.config.get("products_json_path"):
            return Path(self.config["products_json_path"])

        return Path("Agents/products.json")

    def _resolve_output_path(self, task: AgentTask) -> Path:
        """
        Priority:
        1. task.metadata["retrieved_context_output_path"]
        2. self.config["retrieved_context_output_path"]
        3. Agents/retrieved_context.json
        """

        if task.metadata.get("retrieved_context_output_path"):
            return Path(task.metadata["retrieved_context_output_path"])

        if self.config.get("retrieved_context_output_path"):
            return Path(self.config["retrieved_context_output_path"])

        return Path("Agents/retrieved_context.json")

    def _resolve_knowledge_base_dir(self, task: AgentTask) -> Path:
        if task.metadata.get("knowledge_base_dir"):
            return Path(task.metadata["knowledge_base_dir"])

        if self.config.get("knowledge_base_dir"):
            return Path(self.config["knowledge_base_dir"])

        return Path("KnowledgeBase")

    def _resolve_persist_directory(self, task: AgentTask) -> Path:
        if task.metadata.get("persist_directory"):
            return Path(task.metadata["persist_directory"])

        if self.config.get("persist_directory"):
            return Path(self.config["persist_directory"])

        return Path("VectorDB/chroma_db")

    def _resolve_rebuild_vector_db(self, task: AgentTask) -> bool:
        if "rebuild_vector_db" in task.metadata:
            return bool(task.metadata["rebuild_vector_db"])

        if "rebuild_vector_db" in self.config:
            return bool(self.config["rebuild_vector_db"])

        return False

    def _resolve_top_k(self, task: AgentTask) -> int:
        if task.metadata.get("top_k"):
            return int(task.metadata["top_k"])

        if self.config.get("top_k"):
            return int(self.config["top_k"])

        return 6