import unittest

from Agents.core.research_contracts import ErrorCode
from Agents.strategy_policy import (
    deterministic_strategy_plan,
    fallback_decision,
    validate_strategy_plan,
)


class StrategyPolicyTest(unittest.TestCase):
    def products(self, count: int = 2) -> list[dict]:
        return [
            {
                "material_no": str(index),
                "destination": "A" if index % 2 == 0 else "B",
            }
            for index in range(count)
        ]

    def test_rejects_unauthorised_llm_strategy(self) -> None:
        plan = deterministic_strategy_plan(self.products())
        plan["selected_strategy"] = "invent_coordinates"

        with self.assertRaises(ValueError):
            validate_strategy_plan(plan)

    def test_rejects_plan_that_omits_mandatory_constraints(self) -> None:
        plan = deterministic_strategy_plan(self.products())
        plan["required_constraints"] = ["weight_limit"]

        with self.assertRaises(ValueError):
            validate_strategy_plan(plan)

    def test_deterministic_policy_respects_tight_budget(self) -> None:
        plan = deterministic_strategy_plan(
            self.products(20),
            time_budget_seconds=3.0,
        )

        self.assertFalse(plan["optimisation_config"]["use_simulated_annealing"])
        self.assertNotEqual(plan["selected_strategy"], "greedy_plus_simulated_annealing")

    def test_deterministic_policy_can_authorise_sa_with_time_redundancy(self) -> None:
        plan = deterministic_strategy_plan(
            self.products(20),
            time_budget_seconds=10.0,
        )

        self.assertTrue(plan["optimisation_config"]["use_simulated_annealing"])

    def test_fallback_preserves_machine_readable_failure(self) -> None:
        decision = fallback_decision(
            self.products(),
            ErrorCode.RAG_UNAVAILABLE,
            "Vector store unavailable.",
        )

        self.assertEqual(decision["semantic_failure"]["code"], "RAG_UNAVAILABLE")
        self.assertIsNotNone(
            decision["strategy_decision_trace"]["fallback_strategy"]
        )


if __name__ == "__main__":
    unittest.main()
