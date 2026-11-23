"""Evidence-bound LLM enrichment for non-operational product metadata."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from Agents.llm_provider import HTTPModelProvider, create_llm_provider
from Agents.planning_orchestrator import PlanningOrchestrator
from Agents.run_manifest import sha256_file

CORE_LOADING_FIELDS = frozenset(
    {
        "material_no",
        "product_id",
        "width",
        "outer_diameter",
        "length",
        "gross_weight_tonne",
        "weight",
        "destination",
        "thickness",
        "net_weight_tonne",
    }
)
FACTUAL_OPTIONAL_FIELDS = frozenset(
    {"product_name", "receiver", "contract_no", "storage_location", "remarks"}
)
GENERATED_OPTIONAL_FIELDS = frozenset({"product_summary"})
ALLOWED_ENRICHMENT_FIELDS = FACTUAL_OPTIONAL_FIELDS | GENERATED_OPTIONAL_FIELDS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("Enrichment model did not return a JSON object.")


def _clean_text(value: Any, *, maximum: int = 300) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text[:maximum] if text else None


def _identifier(product: Mapping[str, Any]) -> str:
    return str(product.get("material_no") or product.get("product_id") or "").strip()


def _suggestion_id(material_no: str, field: str, value: str) -> str:
    payload = f"{material_no}|{field}|{value}".encode("utf-8")
    return "enr_" + hashlib.sha256(payload).hexdigest()[:16]


class EnrichmentAgent:
    """Generate auditable suggestions without mutating optimisation input."""

    def __init__(
        self,
        *,
        provider: Optional[HTTPModelProvider] = None,
        provider_prefix: str = "CLIENT_LLM",
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 60.0,
        seed: int = 42,
        max_products: int = 25,
    ) -> None:
        self.provider = provider or create_llm_provider(
            prefix=provider_prefix,
            provider=provider_name,
            model=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self.seed = int(seed)
        self.max_products = max(1, min(int(max_products), 100))

    def run(
        self,
        products: list[Dict[str, Any]],
        output_path: str | Path,
        *,
        requested_fields: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_path)
        requested = set(requested_fields or ALLOWED_ENRICHMENT_FIELDS)
        forbidden_requested = sorted(requested - ALLOWED_ENRICHMENT_FIELDS)
        requested &= ALLOWED_ENRICHMENT_FIELDS
        selected_products = products[: self.max_products]
        report: Dict[str, Any] = {
            "schema_version": "1.0",
            "generated_at": _utc_now(),
            "status": "completed",
            "policy": {
                "mode": "evidence_driven_repair_suggestions",
                "loading_input_mutated": False,
                "operator_confirmation_required": True,
                "missing_fact_without_evidence": "leave_blank",
                "core_loading_fields": sorted(CORE_LOADING_FIELDS),
                "allowed_fields": sorted(ALLOWED_ENRICHMENT_FIELDS),
                "requested_fields": sorted(requested),
                "forbidden_requested_fields": forbidden_requested,
            },
            "provider": getattr(self.provider, "provider_name", "injected"),
            "model": getattr(getattr(self.provider, "config", None), "model", None),
            "input_product_count": len(products),
            "processed_product_count": len(selected_products),
            "truncated": len(products) > len(selected_products),
            "suggestions": [],
            "rejected_suggestions": [],
            "unresolved_missing_fields": [],
            "errors": [],
        }
        if not requested or not selected_products:
            report["status"] = "not_required"
            PlanningOrchestrator.write_json(output_path, report)
            return report

        report["response_metadata"] = []
        report["response_adapters"] = []
        for index, product in enumerate(selected_products):
            material_no = _identifier(product)
            destination = _clean_text(product.get("destination"), maximum=160)
            optional_context = {
                field: _clean_text(product.get(field), maximum=180)
                for field in FACTUAL_OPTIONAL_FIELDS
                if _clean_text(product.get(field), maximum=180)
            }
            for field in sorted(requested & FACTUAL_OPTIONAL_FIELDS):
                raw_value = product.get(field)
                cleaned_value = _clean_text(raw_value, maximum=240)
                if cleaned_value is None:
                    report["unresolved_missing_fields"].append(
                        {
                            "material_no": material_no,
                            "field": field,
                            "reason": "no_authoritative_source_evidence",
                        }
                    )
                    continue
                raw_text = str(raw_value)
                if raw_text != cleaned_value:
                    report["suggestions"].append(
                        {
                            "suggestion_id": _suggestion_id(
                                material_no, field, cleaned_value
                            ),
                            "material_no": material_no,
                            "field": field,
                            "value": cleaned_value,
                            "kind": "deterministic_format_repair",
                            "evidence": f"input:{field}={raw_text}",
                            "confidence": "certain",
                            "status": "proposed",
                            "repair_reason": "trimmed_or_collapsed_whitespace",
                        }
                    )
            facts = {
                "material_no": material_no,
                "weight_tonne": product.get("gross_weight_tonne")
                or product.get("weight"),
                "width_mm": product.get("width"),
                "loading_length_mm": product.get("outer_diameter")
                or product.get("length"),
                "destination": destination,
            }
            try:
                combined = {"material_no": material_no, "proposals": {}}
                if "product_summary" in requested:
                    summary_response = self.provider.invoke(
                        [
                            {"role": "system", "content": "Return valid JSON only."},
                            {
                                "role": "user",
                                "content": (
                                    "For this product, write one concise Chinese business "
                                    "summary using only the exact facts. material_no is a "
                                    "material number: label it 材料号, never 产品名称. Copy the "
                                    "destination exactly without translating it. Return "
                                    f'{{"material_no":"{material_no}",'
                                    '"product_summary":"..."}}. Facts: '
                                    f"material_no={material_no}; "
                                    f"weight={facts['weight_tonne']} tonne; "
                                    f"size={facts['width_mm']} x "
                                    f"{facts['loading_length_mm']} mm; "
                                    f"destination={destination}."
                                ),
                            },
                        ],
                        temperature=0,
                        max_tokens=250,
                        num_ctx=2048,
                        seed=self.seed + index,
                        json_mode=True,
                    )
                    combined.update(_parse_json_object(summary_response.content))
                    report["response_metadata"].append(
                        dict(summary_response.response_metadata)
                    )
                factual_requested = sorted(requested & FACTUAL_OPTIONAL_FIELDS)
                if factual_requested and optional_context:
                    source_subset = {
                        field: optional_context[field]
                        for field in factual_requested
                        if field in optional_context
                    }
                    if source_subset:
                        proposal_shape = {
                            field: {
                                "value": value,
                                "evidence_field": field,
                                "confidence": "high",
                            }
                            for field, value in source_subset.items()
                        }
                        optional_response = self.provider.invoke(
                            [
                                {
                                    "role": "system",
                                    "content": "Return valid JSON only.",
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "Normalise only these supplied optional fields. Do not "
                                        "add information. Return "
                                        f"{json.dumps({'proposals': proposal_shape}, ensure_ascii=False)}. "
                                        "Source: "
                                        f"{json.dumps(source_subset, ensure_ascii=False)}"
                                    ),
                                },
                            ],
                            temperature=0,
                            max_tokens=260,
                            num_ctx=2048,
                            seed=self.seed + index + 1000,
                            json_mode=True,
                        )
                        optional_parsed = _parse_json_object(
                            optional_response.content
                        )
                        combined["proposals"] = optional_parsed.get(
                            "proposals", optional_parsed
                        )
                        report["response_metadata"].append(
                            dict(optional_response.response_metadata)
                        )
                self._validate_items([combined], [product], requested, report)
            except Exception as error:
                report["errors"].append(
                    {
                        "material_no": material_no,
                        "type": type(error).__name__,
                        "message": _clean_text(error, maximum=300),
                    }
                )
        if report["errors"] and not report["suggestions"]:
            report["status"] = "safe_fallback"
        elif report["errors"]:
            report["status"] = "partial"
        report["suggestion_count"] = len(report["suggestions"])
        report["rejected_suggestion_count"] = len(report["rejected_suggestions"])
        PlanningOrchestrator.write_json(output_path, report)
        return report

    @staticmethod
    def _validate_items(
        items: Any,
        products: list[Dict[str, Any]],
        requested: set[str],
        report: Dict[str, Any],
    ) -> None:
        if not isinstance(items, list):
            raise ValueError("Enrichment response is missing items.")
        source_by_id = {_identifier(product): product for product in products}
        for item in items:
            if not isinstance(item, dict):
                continue
            material_no = _clean_text(item.get("material_no"), maximum=160)
            if not material_no or material_no not in source_by_id:
                report["rejected_suggestions"].append(
                    {"reason": "unknown_product_identifier", "value": material_no}
                )
                continue
            product = source_by_id[material_no]
            raw_summary = _clean_text(item.get("product_summary"), maximum=300)
            summary = (
                re.sub(r"^产品名称\s*[：:]\s*", "", raw_summary).strip()
                if raw_summary
                else None
            )
            destination = _clean_text(product.get("destination"), maximum=160)
            numeric_facts = [
                product.get("gross_weight_tonne") or product.get("weight"),
                product.get("width"),
                product.get("outer_diameter") or product.get("length"),
            ]
            numeric_tokens = {
                str(int(float(value)))
                if float(value).is_integer()
                else str(float(value))
                for value in numeric_facts
                if value is not None
            }
            summary_is_grounded = bool(
                summary
                and material_no in summary
                and destination
                and destination.casefold() in summary.casefold()
                and sum(token in summary for token in numeric_tokens) >= 2
                and "产品名称" not in summary
                and "product name" not in summary.casefold()
            )
            if summary and "product_summary" in requested and summary_is_grounded:
                report["suggestions"].append(
                    {
                        "suggestion_id": _suggestion_id(
                            material_no, "product_summary", summary
                        ),
                        "material_no": material_no,
                        "field": "product_summary",
                        "value": summary,
                        "kind": "llm_generated_non_operational",
                        "evidence": "Supplied core product context",
                        "confidence": "assistive",
                        "status": "proposed",
                        "normalisation": (
                            "removed_misleading_product_name_prefix"
                            if summary != raw_summary
                            else None
                        ),
                    }
                )
            elif summary and "product_summary" in requested:
                report["rejected_suggestions"].append(
                    {
                        "material_no": material_no,
                        "field": "product_summary",
                        "reason": "summary_not_grounded_in_identifier_and_destination",
                        "candidate_value": raw_summary,
                    }
                )
            proposals = item.get("proposals")
            if not isinstance(proposals, dict):
                continue
            for field, proposal in proposals.items():
                if field in CORE_LOADING_FIELDS:
                    report["rejected_suggestions"].append(
                        {
                            "material_no": material_no,
                            "field": field,
                            "reason": "core_loading_field_forbidden",
                        }
                    )
                    continue
                if field not in FACTUAL_OPTIONAL_FIELDS or field not in requested:
                    report["rejected_suggestions"].append(
                        {
                            "material_no": material_no,
                            "field": field,
                            "reason": "field_not_whitelisted",
                        }
                    )
                    continue
                if not isinstance(proposal, dict):
                    continue
                value = _clean_text(proposal.get("value"), maximum=240)
                evidence_field = _clean_text(
                    proposal.get("evidence_field"), maximum=80
                )
                evidence_value = (
                    _clean_text(product.get(str(evidence_field)), maximum=240)
                    if evidence_field in FACTUAL_OPTIONAL_FIELDS
                    else None
                )
                if (
                    not value
                    or not evidence_field
                    or evidence_field != field
                    or not evidence_value
                ):
                    report["rejected_suggestions"].append(
                        {
                            "material_no": material_no,
                            "field": field,
                            "reason": "missing_direct_source_evidence",
                        }
                    )
                    continue
                if value == evidence_value:
                    continue
                report["suggestions"].append(
                    {
                        "suggestion_id": _suggestion_id(material_no, field, value),
                        "material_no": material_no,
                        "field": field,
                        "value": value,
                        "kind": "evidence_bound_normalisation",
                        "evidence": f"input:{evidence_field}={evidence_value}",
                        "confidence": _clean_text(
                            proposal.get("confidence"), maximum=20
                        )
                        or "unknown",
                        "status": "proposed",
                    }
                )


class EnrichmentConfirmationService:
    """Persist operator decisions without touching products or loading solution."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def confirm(
        self,
        *,
        plan_id: str,
        expected_version: int,
        suggestion_ids: list[str],
        confirmed_by: str,
    ) -> Dict[str, Any]:
        plan = self.repository.get_plan(plan_id)
        if int(plan["version"]) != int(expected_version):
            raise ValueError("Plan version changed; reload before confirming enrichment.")
        run_dir = Path(plan["run_dir"])
        report_path = run_dir / "enrichment_report.json"
        if not report_path.exists():
            raise FileNotFoundError("Plan has no enrichment report.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        available = {
            str(item["suggestion_id"]): item
            for item in report.get("suggestions", [])
            if item.get("suggestion_id")
        }
        unknown = sorted(set(suggestion_ids) - set(available))
        if unknown:
            raise ValueError(f"Unknown enrichment suggestions: {unknown}")
        accepted = [available[item] for item in dict.fromkeys(suggestion_ids)]
        accepted_metadata: Dict[str, Dict[str, Any]] = {}
        for item in accepted:
            accepted_metadata.setdefault(str(item["material_no"]), {})[
                str(item["field"])
            ] = item["value"]
        confirmation = {
            "schema_version": "1.0",
            "plan_id": plan_id,
            "plan_version": int(expected_version),
            "confirmed_at": _utc_now(),
            "confirmed_by": confirmed_by,
            "loading_input_mutated": False,
            "accepted_suggestions": accepted,
            "accepted_metadata": accepted_metadata,
        }
        confirmation_path = run_dir / "enrichment_confirmation.json"
        PlanningOrchestrator.write_json(confirmation_path, confirmation)
        manifest_path = Path(plan["payload"]["manifest_path"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = [
                item
                for item in manifest.get("artifacts", [])
                if item.get("name") != "enrichment_confirmation"
            ]
            artifacts.append(
                {
                    "name": "enrichment_confirmation",
                    "path": str(confirmation_path),
                    "sha256": sha256_file(confirmation_path),
                }
            )
            manifest["artifacts"] = artifacts
            manifest.setdefault("governance", {})["optional_enrichment"] = {
                "status": "operator_confirmed",
                "confirmed_at": confirmation["confirmed_at"],
                "confirmed_by": confirmed_by,
                "accepted_count": len(accepted),
                "loading_input_mutated": False,
            }
            PlanningOrchestrator.write_json(manifest_path, manifest)
        return {**confirmation, "confirmation_path": str(confirmation_path)}
