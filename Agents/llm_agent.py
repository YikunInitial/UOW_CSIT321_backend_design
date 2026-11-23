"""
LLM Agent
=========

Responsibility:
- Input:
    1. products.json
    2. retrieved_context.json

- Process:
    1. Summarise current loading task
    2. Select compact evidence from RAG retrieval
    3. Call local Ollama LLM
    4. Generate structured loading-strategy JSON

- Output:
    strategy_plan.json

Performance design:
- Keeps the real RAG -> LLM reasoning step.
- Uses Ollama JSON mode.
- Caps planning-context size and generated output length.
- Reuses the ChatOllama client within one Python process.
- Records Ollama timing/token metadata for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from Agents.strategy_policy import validate_strategy_plan


class LLMAgent:
    """
    LLMAgent uses a real local Ollama model to generate a loading strategy.

    The model remains a real planning stage in the demo. This class only
    controls prompt size, output size, JSON reliability, and diagnostics.
    """

    _LLM_CACHE: ClassVar[
        Dict[Tuple[str, float, int, int, str, float, int], ChatOllama]
    ] = {}

    _LLM_CACHE_LOCK = RLock()

    def __init__(
        self,
        products_json_path: str | Path = "Agents/products.json",
        retrieved_context_path: str | Path = "Agents/retrieved_context.json",
        output_path: str | Path = "Agents/strategy_plan.json",
        model_name: str = "qwen2.5:7b",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        num_predict: int = 600,
        max_context_chars: int = 3200,
        max_context_chars_per_document: int = 550,
        base_url: str = "http://127.0.0.1:11434",
        request_timeout_seconds: float = 60.0,
        seed: int = 42,
    ) -> None:
        self.products_json_path = Path(products_json_path)
        self.retrieved_context_path = Path(retrieved_context_path)
        self.output_path = Path(output_path)

        self.model_name = model_name
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

        self.max_context_chars = max_context_chars
        self.max_context_chars_per_document = (
            max_context_chars_per_document
        )
        self.base_url = base_url
        self.request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self.seed = int(seed)

        self.llm, self.llm_cache_hit = self._get_or_create_llm()

        if self.llm_cache_hit:
            print(
                f"[LLMAgent] Reusing cached ChatOllama client: "
                f"{self.model_name}"
            )
        else:
            print(
                f"[LLMAgent] Initialised ChatOllama client: "
                f"{self.model_name}"
            )

    # ============================================================
    # Shared Ollama client cache
    # ============================================================

    def _get_or_create_llm(self) -> Tuple[ChatOllama, bool]:
        cache_key = (
            self.model_name,
            float(self.temperature),
            int(self.num_ctx),
            int(self.num_predict),
            self.base_url,
            self.request_timeout_seconds,
            self.seed,
        )

        with self._LLM_CACHE_LOCK:
            cached_llm = self._LLM_CACHE.get(cache_key)

            if cached_llm is not None:
                return cached_llm, True

            llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                num_predict=self.num_predict,
                format="json",
                base_url=self.base_url,
                seed=self.seed,
                client_kwargs={"timeout": self.request_timeout_seconds},
            )

            self._LLM_CACHE[cache_key] = llm

            return llm, False

    @classmethod
    def clear_llm_cache(cls) -> None:
        with cls._LLM_CACHE_LOCK:
            cls._LLM_CACHE.clear()

        print("[LLMAgent] In-memory ChatOllama cache cleared.")

    # ============================================================
    # Main pipeline
    # ============================================================

    def run(self) -> Dict[str, Any]:
        pipeline_started = time.perf_counter()

        input_started = time.perf_counter()
        products = self.load_products()
        retrieved_context = self.load_retrieved_context()
        input_seconds = time.perf_counter() - input_started

        summary_started = time.perf_counter()
        product_summary = self.summarise_products(products)

        (
            compact_rag_context,
            compact_sources,
            compact_context_document_count,
        ) = self.build_compact_rag_context(retrieved_context)

        prompt = self.build_prompt(
            product_summary=product_summary,
            compact_sources=compact_sources,
            compact_rag_context=compact_rag_context,
        )

        prompt_build_seconds = time.perf_counter() - summary_started

        invoke_started = time.perf_counter()
        raw_response, ollama_metadata = self.call_llm(prompt)
        invoke_seconds = time.perf_counter() - invoke_started

        parse_started = time.perf_counter()
        strategy_plan = validate_strategy_plan(
            self.parse_llm_json(raw_response)
        )
        parse_seconds = time.perf_counter() - parse_started

        export_started = time.perf_counter()

        strategy_plan["_metadata"] = {
            "llm_model": self.model_name,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "output_format": "json",
            "request_timeout_seconds": self.request_timeout_seconds,
            "seed": self.seed,
            "products_json_path": str(self.products_json_path),
            "retrieved_context_path": str(self.retrieved_context_path),
            "llm_client_cache_hit": self.llm_cache_hit,
            "prompt_char_count": len(prompt),
            "rag_context_char_count": len(compact_rag_context),
            "rag_context_document_count": compact_context_document_count,
            "timing_seconds": {
                "input_loading": round(input_seconds, 4),
                "prompt_build": round(prompt_build_seconds, 4),
                "ollama_invoke": round(invoke_seconds, 4),
                "json_parse": round(parse_seconds, 4),
            },
            "ollama_response_metadata": ollama_metadata,
        }

        self.export_json(strategy_plan, self.output_path)

        export_seconds = time.perf_counter() - export_started
        total_seconds = time.perf_counter() - pipeline_started

        strategy_plan["_metadata"]["timing_seconds"]["export"] = round(
            export_seconds,
            4,
        )
        strategy_plan["_metadata"]["timing_seconds"]["total"] = round(
            total_seconds,
            4,
        )

        self.export_json(strategy_plan, self.output_path)

        print(
            f"[LLMAgent] Strategy plan generated by real LLM: "
            f"{self.model_name}"
        )
        print(f"[LLMAgent] Output saved to: {self.output_path}")
        print(
            "[LLMAgent] Timing | "
            f"prompt_chars={len(prompt)} | "
            f"rag_context_chars={len(compact_rag_context)} | "
            f"invoke={invoke_seconds:.2f}s | "
            f"parse={parse_seconds:.2f}s | "
            f"total={total_seconds:.2f}s"
        )

        self._print_ollama_runtime_metrics(ollama_metadata)

        return strategy_plan

    # ============================================================
    # Loading inputs
    # ============================================================

    def load_products(self) -> List[Dict[str, Any]]:
        if not self.products_json_path.exists():
            raise FileNotFoundError(
                f"products.json not found: {self.products_json_path}. "
                "Please run DataAgent first."
            )

        with self.products_json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "products.json should contain a list of product records."
            )

        return data

    def load_retrieved_context(self) -> Dict[str, Any]:
        if not self.retrieved_context_path.exists():
            raise FileNotFoundError(
                f"retrieved_context.json not found: "
                f"{self.retrieved_context_path}. "
                "Please run RAGAgent first."
            )

        with self.retrieved_context_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "retrieved_context.json should contain a dictionary."
            )

        return data

    # ============================================================
    # Product summary
    # ============================================================

    def summarise_products(
        self,
        products: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        destinations = [
            str(product.get("destination", "")).strip()
            for product in products
        ]
        destinations = [
            destination
            for destination in destinations
            if destination
        ]

        weights = [
            self.product_weight_kg(product)
            for product in products
        ]

        widths = [
            self.safe_float(product.get("width", 0))
            for product in products
        ]

        outer_diameters = [
            self.safe_float(
                product.get(
                    "outer_diameter",
                    product.get("length", 0),
                )
            )
            for product in products
        ]

        total_weight_kg = sum(weights)

        return {
            "num_products": len(products),
            "destinations": sorted(list(set(destinations))),
            "destination_counts": dict(Counter(destinations)),
            "total_weight_kg": total_weight_kg,
            "average_weight_kg": (
                total_weight_kg / len(products)
                if products
                else 0
            ),
            "max_weight_kg": max(weights) if weights else 0,
            "min_weight_kg": min(weights) if weights else 0,
            "average_width": (
                sum(widths) / len(widths)
                if widths
                else 0
            ),
            "average_outer_diameter": (
                sum(outer_diameters) / len(outer_diameters)
                if outer_diameters
                else 0
            ),
            "has_multiple_destinations": len(set(destinations)) > 1,
        }

    # ============================================================
    # Compact RAG context
    # ============================================================

    def build_compact_rag_context(
        self,
        retrieved_context: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]], int]:
        """
        Keep RAG retrieval intact while bounding the text sent to the LLM.

        All retrieved documents remain stored in retrieved_context.json.
        The LLM receives compact excerpts across ranked evidence documents.
        """

        retrieved_documents = retrieved_context.get(
            "retrieved_documents",
            [],
        )

        if not isinstance(retrieved_documents, list):
            retrieved_documents = []

        compact_sources: List[Dict[str, Any]] = []
        sections: List[str] = []
        remaining_chars = self.max_context_chars

        for document in retrieved_documents:
            if remaining_chars <= 0:
                break

            if not isinstance(document, dict):
                continue

            rank = document.get("rank")
            source = document.get("source")
            doc_type = document.get("doc_type")
            score = document.get("score")

            content = str(document.get("content", "")).strip()

            if not content:
                continue

            content_limit = min(
                self.max_context_chars_per_document,
                remaining_chars,
            )

            excerpt = content[:content_limit].strip()

            if len(content) > content_limit:
                excerpt += "\n[Excerpt truncated for planning prompt.]"

            section = (
                f"[Rank {rank}] Source: {source}\n"
                f"Document type: {doc_type}\n"
                f"Retrieval score: {score}\n"
                f"Evidence excerpt:\n{excerpt}"
            )

            if len(section) > remaining_chars:
                section = section[:remaining_chars].strip()

            sections.append(section)

            compact_sources.append(
                {
                    "rank": rank,
                    "source": source,
                    "doc_type": doc_type,
                    "score": score,
                }
            )

            remaining_chars -= len(section)

        if not sections:
            fallback_context = str(
                retrieved_context.get("recommended_context", "")
            ).strip()

            if fallback_context:
                fallback_context = fallback_context[
                    : self.max_context_chars
                ]

                sections.append(
                    "Fallback RAG context:\n"
                    f"{fallback_context}"
                )

        return (
            "\n\n---\n\n".join(sections),
            compact_sources,
            len(compact_sources),
        )

    # ============================================================
    # Prompt construction
    # ============================================================

    def build_prompt(
        self,
        product_summary: Dict[str, Any],
        compact_sources: List[Dict[str, Any]],
        compact_rag_context: str,
    ) -> str:
        """
        Build a bounded planning prompt for the real LLM.

        It retains RAG evidence, but requires concise structured output.
        """

        return f"""
You are the LLM planning agent in a multi-agent steel coil loading optimisation system.

Your task:
Use the product summary and RAG evidence below to select a strategy for the downstream OptimisationAgent.

Strict boundaries:
- Do not generate product coordinates.
- Do not claim final loading validation has been completed.
- Do not invent unsupported operational constraints.
- Use only evidence provided in the RAG excerpts.
- Return one valid JSON object only. No Markdown and no code fence.
- Keep strategy_reason to at most 80 words.
- Include 1 to 3 rag_evidence items.
- Include at most 3 risk_notes, each at most 25 words.

Current product summary:
{json.dumps(product_summary, ensure_ascii=False, indent=2)}

Retrieved evidence sources:
{json.dumps(compact_sources, ensure_ascii=False, indent=2)}

RAG evidence excerpts:
{compact_rag_context}

Return exactly this JSON structure:

{{
  "scenario_type": "string",
  "selected_strategy": "string",
  "strategy_reason": "string",
  "rag_evidence": [
    {{
      "source": "string",
      "doc_type": "string",
      "how_it_supports_the_decision": "string"
    }}
  ],
  "optimisation_config": {{
    "group_by_destination": true,
    "sort_by_weight_desc": true,
    "use_balance_aware_scoring": true,
    "use_simulated_annealing": false,
    "use_rule_based_constraints": true
  }},
  "required_constraints": [
    "weight_limit",
    "boundary_check",
    "overlap_check",
    "centerline_balance",
    "axle_balance",
    "diagonal_balance"
  ],
  "risk_notes": [
    "string"
  ],
  "next_agent": "optimisation_agent"
}}

Available selected_strategy values:
- simple_greedy
- destination_grouping_greedy
- balance_aware_greedy
- destination_grouping_plus_balance_aware_greedy
- greedy_plus_simulated_annealing
- rule_based_loading

Decision guidance:
- Multiple destinations support destination grouping.
- Heavy products or balance evidence support balance-aware scoring.
- Greedy-plus-simulated-annealing evidence may justify simulated annealing.
- Failed imbalance cases should produce a concise risk note.
""".strip()

    # ============================================================
    # LLM call and JSON parsing
    # ============================================================

    def call_llm(
        self,
        prompt: str,
    ) -> Tuple[str, Dict[str, Any]]:
        messages = [
            SystemMessage(
                content=(
                    "You are a careful steel coil loading planning agent. "
                    "Return valid JSON only."
                )
            ),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        response_metadata = getattr(
            response,
            "response_metadata",
            {},
        )

        if not isinstance(response_metadata, dict):
            response_metadata = {
                "raw_response_metadata": str(response_metadata)
            }

        return (
            str(response.content),
            self._make_json_safe(response_metadata),
        )

    def parse_llm_json(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse JSON from local model output.

        JSON mode should make the strict parse succeed. Existing repair logic
        remains as a fallback for robustness.
        """

        raw_response = raw_response.strip()

        try:
            parsed = json.loads(raw_response)

            if not isinstance(parsed, dict):
                raise ValueError(
                    "LLM JSON response must be an object."
                )

            return parsed

        except json.JSONDecodeError:
            pass

        json_text = self.extract_json_object(raw_response)

        try:
            parsed = json.loads(json_text)

            if not isinstance(parsed, dict):
                raise ValueError(
                    "LLM JSON response must be an object."
                )

            return parsed

        except json.JSONDecodeError:
            repaired_json_text = self.remove_trailing_commas(json_text)

            try:
                parsed = json.loads(repaired_json_text)

                if not isinstance(parsed, dict):
                    raise ValueError(
                        "LLM JSON response must be an object."
                    )

                print(
                    "[LLMAgent] Repaired trailing commas in "
                    "LLM JSON response."
                )

                return parsed

            except json.JSONDecodeError as exc:
                error_output_path = self.output_path.with_suffix(
                    ".raw_response.txt"
                )

                error_output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                error_output_path.write_text(
                    raw_response,
                    encoding="utf-8",
                )

                raise ValueError(
                    "LLM did not return valid JSON. "
                    f"Raw response saved to: {error_output_path}"
                ) from exc

    @staticmethod
    def remove_trailing_commas(text: str) -> str:
        repaired: List[str] = []
        in_string = False
        escaped = False
        index = 0

        while index < len(text):
            char = text[index]

            if in_string:
                repaired.append(char)

                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False

                index += 1
                continue

            if char == '"':
                in_string = True
                repaired.append(char)
                index += 1
                continue

            if char == ",":
                next_index = index + 1

                while (
                    next_index < len(text)
                    and text[next_index].isspace()
                ):
                    next_index += 1

                if (
                    next_index < len(text)
                    and text[next_index] in "}]"
                ):
                    index += 1
                    continue

            repaired.append(char)
            index += 1

        return "".join(repaired)

    @staticmethod
    def extract_json_object(text: str) -> str:
        match = re.search(r"\{[\s\S]*\}", text)

        if not match:
            raise ValueError(
                "No JSON object found in LLM response."
            )

        return match.group(0)

    # ============================================================
    # Output and diagnostics
    # ============================================================

    def export_json(
        self,
        data: Dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        return output_path

    def _print_ollama_runtime_metrics(
        self,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Print timing if Ollama/LangChain exposes it in response metadata.
        """

        source = metadata.get("message", metadata)

        if not isinstance(source, dict):
            source = metadata

        prompt_eval_count = source.get("prompt_eval_count")
        eval_count = source.get("eval_count")

        load_seconds = self._nanoseconds_to_seconds(
            source.get("load_duration")
        )
        prompt_eval_seconds = self._nanoseconds_to_seconds(
            source.get("prompt_eval_duration")
        )
        eval_seconds = self._nanoseconds_to_seconds(
            source.get("eval_duration")
        )
        total_seconds = self._nanoseconds_to_seconds(
            source.get("total_duration")
        )

        if any(
            value is not None
            for value in [
                prompt_eval_count,
                eval_count,
                load_seconds,
                prompt_eval_seconds,
                eval_seconds,
                total_seconds,
            ]
        ):
            print(
                "[LLMAgent] Ollama runtime | "
                f"load={self._format_seconds(load_seconds)} | "
                f"prompt_eval={self._format_seconds(prompt_eval_seconds)} "
                f"({prompt_eval_count} tokens) | "
                f"generation={self._format_seconds(eval_seconds)} "
                f"({eval_count} tokens) | "
                f"server_total={self._format_seconds(total_seconds)}"
            )

    @staticmethod
    def _nanoseconds_to_seconds(value: Any) -> float | None:
        try:
            if value is None:
                return None

            return float(value) / 1_000_000_000

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_seconds(value: float | None) -> str:
        if value is None:
            return "n/a"

        return f"{value:.2f}s"

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): LLMAgent._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                LLMAgent._make_json_safe(item)
                for item in value
            ]

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    @staticmethod
    def safe_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0

            return float(value)

        except Exception:
            return 0.0

    @classmethod
    def product_weight_kg(cls, product: Dict[str, Any]) -> float:
        """Return a canonical product weight in kilograms."""

        for field in ("gross_weight_kg", "weight"):
            value = product.get(field)
            if value not in (None, ""):
                return cls.safe_float(value)

        return cls.safe_float(product.get("gross_weight_tonne")) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LLM Agent using a real local Ollama model "
            "through LangChain."
        )
    )

    parser.add_argument(
        "--products",
        default="Agents/products.json",
        help="Path to products.json generated by DataAgent.",
    )
    parser.add_argument(
        "--context",
        default="Agents/retrieved_context.json",
        help="Path to retrieved_context.json generated by RAGAgent.",
    )
    parser.add_argument(
        "--output",
        default="Agents/strategy_plan.json",
        help="Path to output strategy_plan.json.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Lower values make output more stable.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Ollama context window size.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=600,
        help="Maximum output tokens for the strategy JSON.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=3200,
        help="Maximum RAG evidence characters sent to the LLM.",
    )
    parser.add_argument(
        "--max-context-chars-per-document",
        type=int,
        default=550,
        help="Maximum evidence excerpt characters per retrieved document.",
    )

    args = parser.parse_args()

    agent = LLMAgent(
        products_json_path=args.products,
        retrieved_context_path=args.context,
        output_path=args.output,
        model_name=args.model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        max_context_chars=args.max_context_chars,
        max_context_chars_per_document=(
            args.max_context_chars_per_document
        ),
    )

    agent.run()


if __name__ == "__main__":
    main()
