"""Independent, versioned validation boundary for loading solutions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from Agents.core.research_contracts import (
    CONSTRAINT_SET_VERSION,
    OBJECTIVE_SET_VERSION,
    VALIDATOR_VERSION,
    VersionInfo,
)
from Agents.optimisation_agent import validate_loading_solution


HARD_CONSTRAINTS: Dict[str, str] = {
    "weight_limit": "Total wagon weight must not exceed its configured limit.",
    "boundary": "Every product footprint must remain inside the wagon.",
    "non_overlap": "Product footprints must not overlap.",
    "valid_dimensions": "Every product must have positive finite dimensions.",
    "product_coverage": "Expected products must appear exactly once.",
    "resource_availability": "Unavailable wagon identifiers must not be used.",
}

SOFT_OBJECTIVES: Dict[str, str] = {
    "wagon_count": "Prefer fewer wagons after hard feasibility is satisfied.",
    "unused_area": "Prefer higher loading-area utilisation.",
    "front_rear_balance": "Reduce regional front/rear weight imbalance.",
    "left_right_balance": "Reduce regional left/right weight imbalance.",
    "diagonal_balance": "Reduce regional diagonal weight imbalance.",
    "mixed_destination": "Prefer operational destination clarity.",
    "operational_complexity": "Prefer simpler loading operations.",
}

REPORTING_METRICS: Dict[str, str] = {
    "runtime_seconds": "Measured execution time; not a feasibility constraint.",
    "area_utilisation": "Area-based utilisation indicator.",
    "plan_stability": "Fraction of prior decisions preserved after replanning.",
    "scope_amplification": "Replanned scope divided by directly affected scope.",
}


class ValidationAgent:
    """Revalidates persisted candidate data at the safety boundary."""

    def __init__(self, tolerance_mm: float = 1e-6) -> None:
        self.tolerance_mm = tolerance_mm

    def validate(
        self,
        solution: Dict[str, Any],
        expected_material_numbers: Optional[Iterable[str]] = None,
        unavailable_wagon_ids: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        expected = (
            [str(item) for item in expected_material_numbers]
            if expected_material_numbers is not None
            else self.material_numbers_from_solution(solution)
        )
        validation = validate_loading_solution(
            wagons=solution.get("wagons", []),
            expected_material_numbers=expected,
            tolerance_mm=self.tolerance_mm,
        )
        unavailable = {str(item) for item in unavailable_wagon_ids or [] if str(item)}
        reused = sorted(
            str(wagon.get("wagon_id"))
            for wagon in solution.get("wagons", [])
            if str(wagon.get("wagon_id")) in unavailable
        )
        validation["unavailable_wagon_ids"] = sorted(unavailable)
        validation["reused_unavailable_wagon_ids"] = reused
        validation["unavailable_wagon_violation_count"] = len(reused)
        if reused:
            validation.setdefault("violations", []).extend(
                {"type": "unavailable_wagon_reuse", "wagon_id": wagon_id}
                for wagon_id in reused
            )
            validation["violation_count"] = int(validation.get("violation_count", 0)) + len(reused)
            validation["is_valid"] = False
        stored = solution.get("validation_summary")
        stored_is_valid = (
            stored.get("is_valid") if isinstance(stored, dict) else None
        )
        validation["validator"] = {
            "version": VALIDATOR_VERSION,
            "constraint_set_version": CONSTRAINT_SET_VERSION,
            "objective_set_version": OBJECTIVE_SET_VERSION,
            "independent_revalidation": True,
            "stored_validation_present": isinstance(stored, dict),
            "stored_validation_agrees": (
                stored_is_valid == validation["is_valid"]
                if stored_is_valid is not None
                else None
            ),
        }
        validation["constraint_classification"] = self.classification()
        validation["veto_reasons"] = self.veto_reasons(validation)
        return validation

    @staticmethod
    def material_numbers_from_solution(
        solution: Dict[str, Any],
    ) -> List[str]:
        return [
            str(
                product.get("material_no")
                or product.get("product_id")
                or product.get("id")
                or ""
            )
            for wagon in solution.get("wagons", [])
            for product in wagon.get("products", [])
            if (
                product.get("material_no")
                or product.get("product_id")
                or product.get("id")
            )
        ]

    @staticmethod
    def classification() -> Dict[str, Any]:
        return {
            "versions": VersionInfo().to_dict(),
            "hard_constraints": dict(HARD_CONSTRAINTS),
            "soft_objectives": dict(SOFT_OBJECTIVES),
            "reporting_metrics": dict(REPORTING_METRICS),
            "notes": [
                "Balance measures are regional soft objectives, not hard limits.",
                "The current front/rear metric is not a physical axle-reaction model.",
                "Destination grouping is an operational objective unless a future "
                "constraint version explicitly promotes it to a hard rule.",
            ],
        }

    @staticmethod
    def veto_reasons(validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        reasons: List[Dict[str, Any]] = []
        count_fields = {
            "overweight": "overweight_violation_count",
            "boundary": "boundary_violation_count",
            "overlap": "overlap_violation_count",
            "invalid_dimension": "invalid_dimension_count",
            "coverage": "coverage_violation_count",
            "unavailable_wagon_reuse": "unavailable_wagon_violation_count",
        }
        for reason_type, field in count_fields.items():
            count = int(validation.get(field) or 0)
            if count:
                reasons.append({"type": reason_type, "count": count})
        for field, reason_type in (
            ("missing_product_ids", "missing_products"),
            ("duplicate_product_ids", "duplicate_products"),
            ("unexpected_product_ids", "unexpected_products"),
        ):
            product_ids = list(validation.get(field) or [])
            if product_ids:
                reasons.append(
                    {
                        "type": reason_type,
                        "count": len(product_ids),
                        "product_ids": product_ids,
                    }
                )
        return reasons
