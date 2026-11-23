import json
import tempfile
import unittest
from pathlib import Path

from Agents.planning_orchestrator import PlanningOrchestrator, PlanningRequest


class PlanningOrchestratorTest(unittest.TestCase):
    def products(self) -> list[dict]:
        return [
            {
                "material_no": "A",
                "destination": "Sydney",
                "gross_weight_tonne": 10.0,
                "width": 1000.0,
                "outer_diameter": 1200.0,
            },
            {
                "material_no": "B",
                "destination": "Sydney",
                "gross_weight_tonne": 9.0,
                "width": 1000.0,
                "outer_diameter": 1100.0,
            },
        ]

    def run_request(self, strategy_plan=None) -> tuple[dict, Path]:
        root = Path(tempfile.mkdtemp(dir=Path("runtime") / "test_tmp"))
        products_path = root / "products.json"
        products_path.write_text(
            json.dumps(self.products()),
            encoding="utf-8",
        )
        result = PlanningOrchestrator().run(
            PlanningRequest(
                products_path=products_path,
                run_dir=root / "run",
                strategy_plan=strategy_plan,
                random_seed=42,
            )
        )
        return result, root

    def test_single_entry_point_requires_approval_after_validation(self) -> None:
        result, _ = self.run_request()

        self.assertTrue(result["solution"]["validation_summary"]["is_valid"])
        self.assertEqual(result["plan"]["status"], "pending_approval")
        self.assertTrue(result["manifest_path"].exists())
        self.assertEqual(
            result["solution"]["research_system"]["versions"]["system"],
            "research-system-v2.0.0",
        )
        manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))
        trace = manifest["agent_decisions"][0]
        self.assertEqual(
            trace["executed_strategy"],
            result["solution"]["strategy_execution"]["applied_strategy"],
        )
        self.assertEqual(manifest["configuration"]["llm_model_name"], "qwen2.5:0.5b")

    def test_invalid_requested_strategy_falls_back_deterministically(self) -> None:
        invalid = {
            "selected_strategy": "free_form_coordinates",
            "optimisation_config": {},
            "required_constraints": [],
        }

        result, _ = self.run_request(invalid)
        trace = result["solution"]["research_system"]["strategy_decision_trace"]

        self.assertEqual(trace["requested_strategy"], "free_form_coordinates")
        self.assertIsNotNone(trace["fallback_strategy"])
        self.assertTrue(trace["fallback_reasons"])

    def test_semantic_planner_failure_uses_safe_deterministic_fallback(self) -> None:
        root = Path(tempfile.mkdtemp(dir=Path("runtime") / "test_tmp"))
        products_path = root / "products.json"
        products_path.write_text(json.dumps(self.products()), encoding="utf-8")

        def unavailable(_products_path: Path, _run_dir: Path) -> dict:
            raise ConnectionError("Ollama unavailable")

        result = PlanningOrchestrator().run(
            PlanningRequest(
                products_path=products_path,
                run_dir=root / "run",
                semantic_planning_enabled=True,
                semantic_strategy_provider=unavailable,
            )
        )
        trace = result["solution"]["research_system"]["strategy_decision_trace"]

        self.assertEqual(trace["recommendation_source"], "semantic_planner_failed")
        self.assertIsNotNone(trace["fallback_strategy"])
        self.assertEqual(
            result["strategy_plan"]["semantic_failure"]["code"],
            "LLM_UNAVAILABLE",
        )

    def test_council_run_exports_decision_matrix_and_manifest_evidence(self) -> None:
        root = Path(tempfile.mkdtemp(dir=Path("runtime") / "test_tmp"))
        products_path = root / "products.json"
        products_path.write_text(json.dumps(self.products()), encoding="utf-8")

        result = PlanningOrchestrator().run(
            PlanningRequest(
                products_path=products_path,
                run_dir=root / "run",
                execution_mode="multi_agent_council",
                system_variant="multi_agent_council",
            )
        )
        council = result["solution"]["multi_agent_council"]
        manifest = json.loads(
            result["manifest_path"].read_text(encoding="utf-8")
        )

        self.assertTrue(council["coordinator_decision"]["selected_candidate"])
        self.assertGreaterEqual(
            council["diversity_summary"]["completed_candidate_count"],
            2,
        )
        self.assertTrue(
            Path(council["artifacts"]["evaluation_matrix_csv"]).exists()
        )
        artifact_names = {item["name"] for item in manifest["artifacts"]}
        self.assertIn("evaluation_matrix_csv", artifact_names)
        self.assertTrue(
            any(
                item.get("decision_type") == "multi_agent_council"
                for item in manifest["agent_decisions"]
            )
        )


if __name__ == "__main__":
    unittest.main()
