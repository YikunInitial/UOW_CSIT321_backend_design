import unittest

from Agents.core.research_contracts import (
    AgenticSystemError,
    ErrorCode,
    EventStatus,
    EventVersion,
    PlanStatus,
    PlanVersion,
    StrategyDecisionTrace,
    StructuredError,
    VersionInfo,
)


class ResearchContractsTest(unittest.TestCase):
    def test_strategy_trace_distinguishes_authority_stages(self) -> None:
        trace = StrategyDecisionTrace(
            requested_strategy="greedy",
            recommended_strategy="greedy_plus_sa",
            authorised_strategy="greedy_plus_sa",
            executed_strategy="greedy_plus_sa",
            recommendation_source="llm_rag",
            authorisation_source="deterministic_policy",
        )

        payload = trace.to_dict()

        self.assertEqual(payload["requested_strategy"], "greedy")
        self.assertEqual(payload["authorised_strategy"], "greedy_plus_sa")
        self.assertEqual(payload["executed_strategy"], "greedy_plus_sa")

    def test_plan_state_machine_accepts_valid_lifecycle(self) -> None:
        plan = PlanVersion(plan_id="PLAN-1")

        for status in (
            PlanStatus.GENERATED,
            PlanStatus.VALIDATED,
            PlanStatus.PENDING_APPROVAL,
            PlanStatus.APPROVED,
            PlanStatus.ACTIVE,
            PlanStatus.SUPERSEDED,
        ):
            plan.transition(status)

        self.assertEqual(plan.status, PlanStatus.SUPERSEDED)

    def test_plan_state_machine_rejects_unsafe_activation(self) -> None:
        plan = PlanVersion(plan_id="PLAN-1")

        with self.assertRaises(ValueError):
            plan.transition(PlanStatus.ACTIVE)

    def test_structured_error_is_machine_readable(self) -> None:
        error = StructuredError(
            code=ErrorCode.VALIDATION_FAILED,
            message="Candidate failed hard validation.",
            requires_manual_review=True,
            details={"violation_count": 2},
        )
        exception = AgenticSystemError(error)

        self.assertEqual(exception.to_dict()["code"], "VALIDATION_FAILED")
        self.assertEqual(exception.to_dict()["details"]["violation_count"], 2)

    def test_versions_and_events_have_stable_serialisation(self) -> None:
        versions = VersionInfo().to_dict()
        event = EventVersion(
            event_id="EVENT-1",
            expected_plan_id="PLAN-1",
            expected_plan_version=3,
            status=EventStatus.VALIDATED,
        ).to_dict()

        self.assertEqual(versions["system"], "research-system-v2.0.0")
        self.assertEqual(event["expected_plan_version"], 3)
        self.assertEqual(event["status"], "validated")


if __name__ == "__main__":
    unittest.main()
