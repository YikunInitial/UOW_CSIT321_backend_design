import unittest

from Agents.validation_agent import ValidationAgent


def product(product_id: str, x: float) -> dict:
    return {
        "material_no": product_id,
        "product_id": product_id,
        "x": x,
        "y": 0.0,
        "placed_length": 1000.0,
        "placed_width": 1000.0,
        "weight": 10.0,
    }


class ValidationAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = ValidationAgent()

    def solution(self, products: list[dict]) -> dict:
        return {
            "validation_summary": {"is_valid": True},
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

    def test_revalidates_valid_solution_with_versioned_contract(self) -> None:
        result = self.agent.validate(
            self.solution([product("A", 0.0), product("B", 1100.0)]),
            expected_material_numbers=["A", "B"],
        )

        self.assertTrue(result["is_valid"])
        self.assertTrue(result["validator"]["independent_revalidation"])
        self.assertEqual(result["veto_reasons"], [])
        self.assertIn("left_right_balance", result["constraint_classification"]["soft_objectives"])

    def test_independent_validation_detects_overlap_and_disagreement(self) -> None:
        result = self.agent.validate(
            self.solution([product("A", 0.0), product("B", 500.0)]),
            expected_material_numbers=["A", "B"],
        )

        self.assertFalse(result["is_valid"])
        self.assertFalse(result["validator"]["stored_validation_agrees"])
        self.assertIn("overlap", {item["type"] for item in result["veto_reasons"]})

    def test_expected_input_detects_missing_product(self) -> None:
        result = self.agent.validate(
            self.solution([product("A", 0.0)]),
            expected_material_numbers=["A", "B"],
        )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["missing_product_ids"], ["B"])


if __name__ == "__main__":
    unittest.main()
