import json
import tempfile
import time
import unittest
from pathlib import Path

from Agents.multi_agent_council import MultiAgentCouncil


def product(product_id: str, x: float, destination: str = "A") -> dict:
    return {
        "material_no": product_id,
        "product_id": product_id,
        "destination": destination,
        "x": x,
        "y": 0.0,
        "placed_length": 1000.0,
        "placed_width": 1000.0,
        "weight": 10.0,
    }


def solution(products: list[dict], stored_valid: bool = True) -> dict:
    return {
        "validation_summary": {"is_valid": stored_valid},
        "wagon_summary": {
            "wagon_count": 1,
            "average_area_utilisation": 0.5,
        },
        "final_solution_score": {
            "front_rear_balance_penalty": 0.1,
            "left_right_balance_penalty": 0.1,
            "diagonal_balance_penalty": 0.1,
        },
        "input_summary": {"destinations": ["A"]},
        "strategy_execution": {"applied_strategy": "balance_aware_greedy"},
        "wagons": [
            {
                "wagon_id": "W001",
                "length": 5000.0,
                "width": 3000.0,
                "max_weight": 60.0,
                "current_weight": sum(item["weight"] for item in products),
                "products": products,
            }
        ],
    }


class MultiAgentCouncilTest(unittest.TestCase):
    def test_safety_agent_revalidates_and_vetoes_false_stored_pass(self) -> None:
        council = MultiAgentCouncil()
        candidate = {
            "agent_name": "UnsafeAgent",
            "status": "completed",
            "runtime_seconds": 0.1,
            "solution": solution(
                [product("A", 0.0), product("B", 500.0)],
                stored_valid=True,
            ),
        }

        evaluated = council.evaluate_candidate(candidate, ["A", "B"])

        self.assertTrue(evaluated["evaluation"]["SafetyAgent"]["veto"])
        self.assertIn(
            "overlap",
            {
                item["type"]
                for item in evaluated["evaluation"]["SafetyAgent"]["veto_reasons"]
            },
        )

    def test_coordinator_never_selects_vetoed_candidate(self) -> None:
        council = MultiAgentCouncil()
        unsafe = council.evaluate_candidate(
            {
                "agent_name": "UnsafeAgent",
                "selected_strategy": "simple_greedy",
                "status": "completed",
                "runtime_seconds": 0.1,
                "solution": solution([product("A", 0.0), product("B", 500.0)]),
            },
            ["A", "B"],
        )
        safe = council.evaluate_candidate(
            {
                "agent_name": "SafeAgent",
                "selected_strategy": "balance_aware_greedy",
                "status": "completed",
                "runtime_seconds": 0.1,
                "solution": solution([product("A", 0.0), product("B", 1100.0)]),
            },
            ["A", "B"],
        )

        decision = council.select_candidate([unsafe, safe])

        self.assertEqual(decision["selected_candidate"], "SafeAgent")
        self.assertIn("UnsafeAgent", decision["vetoed_candidates"])

    def test_isolated_safety_ablation_observes_but_does_not_veto(self) -> None:
        council = MultiAgentCouncil(safety_veto_enabled=False)
        evaluated = council.evaluate_candidate(
            {
                "agent_name": "UnsafeAgent",
                "selected_strategy": "simple_greedy",
                "status": "completed",
                "runtime_seconds": 0.1,
                "solution": solution(
                    [product("A", 0.0), product("B", 500.0)],
                    stored_valid=True,
                ),
            },
            ["A", "B"],
        )

        safety = evaluated["evaluation"]["SafetyAgent"]
        self.assertFalse(safety["enabled"])
        self.assertFalse(safety["pass"])
        self.assertFalse(safety["veto"])
        self.assertFalse(
            evaluated["evaluation"]["CoordinatorAgent"]["vetoed"]
        )

    def test_operations_score_uses_actual_destinations_not_strategy_label(self) -> None:
        council = MultiAgentCouncil()
        grouped = solution([product("A", 0.0), product("B", 1100.0)])
        mixed = solution(
            [product("A", 0.0, "A"), product("B", 1100.0, "B")]
        )
        grouped["strategy_execution"]["applied_strategy"] = "simple_greedy"
        mixed["strategy_execution"]["applied_strategy"] = (
            "destination_grouping_plus_balance_aware_greedy"
        )

        self.assertGreater(
            council.operations_score(grouped),
            council.operations_score(mixed),
        )

    def test_balance_score_normalises_fleet_penalties_per_wagon(self) -> None:
        council = MultiAgentCouncil()
        balanced = {
            "wagon_count": 26,
            "front_rear_balance_penalty": 13.0,
            "left_right_balance_penalty": 0.13,
            "diagonal_balance_penalty": 0.052,
        }
        front_loaded = {
            "wagon_count": 26,
            "front_rear_balance_penalty": 26.0,
            "left_right_balance_penalty": 0.13,
            "diagonal_balance_penalty": 0.13,
        }

        self.assertGreater(
            council.balance_score(balanced),
            council.balance_score(front_loaded),
        )
        self.assertAlmostEqual(council.balance_score(balanced), 0.8310, places=4)

    def test_deadline_returns_first_safe_anytime_candidate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("runtime") / "test_tmp") as tmp:
            root = Path(tmp)
            products_path = root / "products.json"
            products_path.write_text(
                json.dumps(
                    [
                        {"material_no": "A"},
                        {"material_no": "B"},
                    ]
                ),
                encoding="utf-8",
            )

            def callback(_products: Path, _plan: dict, candidate_dir: Path):
                candidate_dir.mkdir(parents=True, exist_ok=True)
                time.sleep(0.01)
                output = candidate_dir / "loading_solution.json"
                output.write_text("{}", encoding="utf-8")
                return solution(
                    [product("A", 0.0), product("B", 1100.0)]
                ), output

            result = MultiAgentCouncil(time_budget_seconds=0.001).run(
                products_path=products_path,
                base_strategy_plan={
                    "selected_strategy": "balance_aware_greedy",
                    "optimisation_config": {},
                },
                run_dir=root,
                run_candidate=callback,
            )

            self.assertTrue(result["deadline"]["anytime_solution_returned"])
            self.assertTrue(result["deadline"]["deadline_exhausted"])
            self.assertTrue(
                any(
                    item["status"] == "skipped_insufficient_budget"
                    for item in result["candidate_agents"]
                )
            )
            self.assertEqual(result["deadline"]["deadline_type"], "cooperative")
            self.assertEqual(
                result["deadline"]["coordination_reserve_ratio"],
                0.10,
            )
            self.assertFalse(result["deadline"]["deadline_met"])
            self.assertGreater(result["deadline"]["overrun_seconds"], 0.0)
            for artifact_path in result["artifacts"].values():
                self.assertTrue(Path(artifact_path).exists())

    def test_all_failed_candidates_stop_without_unsafe_selection(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("runtime") / "test_tmp") as tmp:
            root = Path(tmp)
            products_path = root / "products.json"
            products_path.write_text(
                json.dumps([{"material_no": "A"}]),
                encoding="utf-8",
            )

            def callback(_products: Path, _plan: dict, _candidate_dir: Path):
                raise RuntimeError("injected candidate failure")

            with self.assertRaisesRegex(RuntimeError, "ALL_CANDIDATES_VETOED"):
                MultiAgentCouncil(time_budget_seconds=1.0).run(
                    products_path=products_path,
                    base_strategy_plan={"optimisation_config": {}},
                    run_dir=root,
                    run_candidate=callback,
                )

    def test_council_artifacts_preserve_utf8_destinations(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path("runtime") / "test_tmp") as tmp:
            root = Path(tmp)
            result = {
                "enabled": True,
                "evaluation_matrix": [
                    {
                        "candidate": "A",
                        "destination": "昆明王家营站",
                        "safety_veto_reasons": [],
                    }
                ],
                "selected_solution": {},
                "selected_strategy_plan": {},
            }

            artifacts = MultiAgentCouncil.write_council_artifacts(root, result)

            decision = json.loads(
                Path(artifacts["decision_json"]).read_text(encoding="utf-8")
            )
            matrix = json.loads(
                Path(artifacts["evaluation_matrix_json"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(decision["enabled"])
            self.assertEqual(matrix[0]["destination"], "昆明王家营站")
            self.assertIn(
                "昆明王家营站",
                Path(artifacts["evaluation_matrix_csv"]).read_text(
                    encoding="utf-8-sig"
                ),
            )

    def test_candidate_budget_weights_are_fair_and_complete(self) -> None:
        specs = MultiAgentCouncil(time_budget_seconds=15.0).candidate_specs()

        self.assertAlmostEqual(sum(item.budget_weight for item in specs), 1.0)
        self.assertEqual(len({item.agent_name for item in specs}), len(specs))
        self.assertTrue(
            all(
                item.optimisation_config["allow_policy_strategy_override"] is False
                for item in specs
            )
        )

    def test_operator_required_destination_grouping_is_enforced_for_every_candidate(self) -> None:
        council = MultiAgentCouncil(time_budget_seconds=12.0)
        base_plan = {
            "scenario_type": "multi_destination_batch",
            "selected_strategy": "destination_grouping_plus_balance_aware_greedy",
            "strategy_reason": "Operator-confirmed business requirement.",
            "optimisation_config": {"group_by_destination": True},
            "required_constraints": [
                "weight_limit",
                "boundary_check",
                "overlap_check",
            ],
            "operational_requirements": {
                "destination_grouping_required": True,
            },
        }

        candidates = [
            council.build_candidate_strategy_plan(base_plan, spec)
            for spec in council.candidate_specs()
        ]

        self.assertTrue(
            all(
                item["optimisation_config"]["group_by_destination"] is True
                for item in candidates
            )
        )
        self.assertTrue(
            all(
                item["candidate_agent"]["optimisation_config"][
                    "group_by_destination"
                ]
                is True
                for item in candidates
            )
        )
        self.assertEqual(
            candidates[0]["selected_strategy"],
            "destination_grouping_plus_balance_aware_greedy",
        )

    def test_tie_break_prefers_lower_runtime_for_equal_score_and_objective(self) -> None:
        council = MultiAgentCouncil()
        candidates = []
        for name, runtime in (("SlowAgent", 2.0), ("FastAgent", 0.5)):
            candidate = council.evaluate_candidate(
                {
                    "agent_name": name,
                    "selected_strategy": "balance_aware_greedy",
                    "status": "completed",
                    "runtime_seconds": runtime,
                    "solution": solution(
                        [product("A", 0.0), product("B", 1100.0)]
                    ),
                },
                ["A", "B"],
            )
            candidate["evaluation"]["CoordinatorAgent"]["final_score"] = 0.8
            candidate["solution"]["final_solution_score"]["objective_score"] = 10.0
            candidates.append(candidate)

        decision = council.select_candidate(candidates)

        self.assertEqual(decision["selected_candidate"], "FastAgent")


if __name__ == "__main__":
    unittest.main()
