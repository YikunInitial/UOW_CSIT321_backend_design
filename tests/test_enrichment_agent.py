import json
from copy import deepcopy

import pytest

from Agents.enrichment_agent import EnrichmentAgent, EnrichmentConfirmationService
from Agents.llm_provider import LLMProviderResponse


class FakeProvider:
    provider_name = "fake-local"

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        if self.error:
            raise self.error
        payload = self.payload
        if isinstance(payload, list):
            payload = payload[self.calls]
        self.calls += 1
        return LLMProviderResponse(
            content=json.dumps(payload, ensure_ascii=False),
            response_metadata={"model": "test-0.5b"},
        )


def product():
    return {
        "material_no": "M-001",
        "thickness": 1.2,
        "width": 1200.0,
        "outer_diameter": 1500.0,
        "gross_weight_tonne": 20.0,
        "net_weight_tonne": 19.8,
        "destination": "Sydney",
        "receiver": "BAJC Sydney DC",
    }


def test_enrichment_is_suggestion_only_and_evidence_bound(tmp_path) -> None:
    products = [product()]
    products[0]["receiver"] = "  BAJC   Sydney DC  "
    original = deepcopy(products)
    provider = FakeProvider(
        [
            {
                "material_no": "M-001",
                "product_summary": (
                    "M-001: 20 tonne product, 1200 x 1500 mm, for Sydney."
                ),
            },
            {
                "proposals": {
                    "receiver": {
                        "value": "BAJC Sydney DC",
                        "evidence_field": "receiver",
                        "confidence": "high",
                    }
                }
            },
        ]
    )

    report = EnrichmentAgent(provider=provider).run(
        products,
        tmp_path / "enrichment_report.json",
        requested_fields=["receiver", "product_summary"],
    )

    assert products == original
    assert report["status"] == "completed"
    assert {item["field"] for item in report["suggestions"]} == {
        "receiver",
        "product_summary",
    }
    assert report["policy"]["loading_input_mutated"] is False


def test_enrichment_rejects_core_fields_and_unsupported_cross_field_evidence(
    tmp_path,
) -> None:
    provider = FakeProvider(
        [
            {"material_no": "M-001", "product_summary": None},
            {
                "proposals": {
                    "width": {
                        "value": "999",
                        "evidence_field": "receiver",
                    },
                    "receiver": {
                        "value": "invented",
                        "evidence_field": "remarks",
                    },
                }
            },
        ]
    )

    report = EnrichmentAgent(provider=provider).run(
        [product()],
        tmp_path / "enrichment_report.json",
        requested_fields=["receiver", "product_summary"],
    )

    assert report["suggestions"] == []
    assert {item["reason"] for item in report["rejected_suggestions"]} == {
        "core_loading_field_forbidden",
        "missing_direct_source_evidence",
    }


def test_enrichment_failure_does_not_fail_planning_path(tmp_path) -> None:
    report = EnrichmentAgent(provider=FakeProvider(error=TimeoutError("offline"))).run(
        [product()],
        tmp_path / "enrichment_report.json",
    )

    assert report["status"] == "safe_fallback"
    assert report["suggestions"] == []
    assert report["errors"][0]["type"] == "TimeoutError"


def test_missing_optional_fact_stays_blank_without_authoritative_evidence(
    tmp_path,
) -> None:
    provider = FakeProvider(
        {
            "material_no": "M-001",
            "product_summary": (
                "M-001: 20 tonne product, 1200 x 1500 mm, for Sydney."
            ),
        }
    )

    report = EnrichmentAgent(provider=provider).run(
        [product()],
        tmp_path / "enrichment_report.json",
        requested_fields=["product_name", "product_summary"],
    )

    assert not any(
        item["field"] == "product_name" for item in report["suggestions"]
    )
    assert report["unresolved_missing_fields"] == [
        {
            "material_no": "M-001",
            "field": "product_name",
            "reason": "no_authoritative_source_evidence",
        }
    ]


def test_confirmation_creates_separate_audited_metadata_record(tmp_path) -> None:
    suggestion = {
        "suggestion_id": "enr_test",
        "material_no": "M-001",
        "field": "product_summary",
        "value": "Coil for Sydney.",
    }
    (tmp_path / "enrichment_report.json").write_text(
        json.dumps({"suggestions": [suggestion]}), encoding="utf-8"
    )
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"artifacts": []}), encoding="utf-8")

    class Repository:
        @staticmethod
        def get_plan(_plan_id):
            return {
                "plan_id": "plan-1",
                "version": 1,
                "run_dir": str(tmp_path),
                "payload": {"manifest_path": str(manifest_path)},
            }

    result = EnrichmentConfirmationService(Repository()).confirm(
        plan_id="plan-1",
        expected_version=1,
        suggestion_ids=["enr_test"],
        confirmed_by="operator-1",
    )

    assert result["loading_input_mutated"] is False
    assert result["accepted_metadata"] == {
        "M-001": {"product_summary": "Coil for Sydney."}
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["governance"]["optional_enrichment"]["accepted_count"] == 1

    with pytest.raises(ValueError, match="Unknown enrichment suggestions"):
        EnrichmentConfirmationService(Repository()).confirm(
            plan_id="plan-1",
            expected_version=1,
            suggestion_ids=["unknown"],
            confirmed_by="operator-1",
        )
