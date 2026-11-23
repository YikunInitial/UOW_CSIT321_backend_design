import unittest

from Agents.llm_agent import LLMAgent
from Agents.rag_agent import RAGAgent


class ProductSummaryWeightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.products = [
            {
                "material_no": "A",
                "destination": "Sydney",
                "gross_weight_tonne": 12.5,
                "width": 1000.0,
                "outer_diameter": 1200.0,
            },
            {
                "material_no": "B",
                "destination": "Melbourne",
                "gross_weight_kg": 8700.0,
                "gross_weight_tonne": 8.7,
                "width": 900.0,
                "outer_diameter": 1100.0,
            },
        ]

    def test_llm_summary_normalises_tonnes_to_kilograms(self) -> None:
        agent = object.__new__(LLMAgent)

        summary = agent.summarise_products(self.products)

        self.assertEqual(summary["total_weight_kg"], 21200.0)
        self.assertEqual(summary["average_weight_kg"], 10600.0)

    def test_rag_summary_normalises_tonnes_to_kilograms(self) -> None:
        agent = object.__new__(RAGAgent)

        summary = agent.summarise_products(self.products)

        self.assertEqual(summary["total_weight_kg"], 21200.0)
        self.assertEqual(summary["average_weight_kg"], 10600.0)

    def test_explicit_kilograms_take_precedence_over_tonnes(self) -> None:
        product = {
            "gross_weight_kg": 5000.0,
            "weight": 6000.0,
            "gross_weight_tonne": 7.0,
        }

        self.assertEqual(LLMAgent.product_weight_kg(product), 5000.0)
        self.assertEqual(RAGAgent._product_weight_kg(product), 5000.0)


if __name__ == "__main__":
    unittest.main()
