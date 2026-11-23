import unittest

from Agents.optimisation_agent import (
    FreightCarSpec,
    ObjectiveWeights,
    OptimisationAgent,
    Product,
    build_initial_feasible_solution,
    evaluate_loading_solution,
    generate_loading_blocks,
    improve_by_wagon_reassignment,
    merge_low_utilisation_cars,
    pack_destination_into_cars,
    positioned_to_visual_product,
    resolve_strategy_execution_plan,
    validate_loading_solution,
    validate_wagon_layout,
)


def product(
    material_no: str,
    weight: float,
    width: float = 1000.0,
    diameter: float = 1200.0,
    destination: str = "Shanghai",
) -> Product:
    return Product(
        material_no=material_no,
        destination=destination,
        gross_weight=weight,
        width=width,
        outer_diameter=diameter,
    )


def pack(products, *, spec=None, balance_aware=True):
    return pack_destination_into_cars(
        products=products,
        spec=spec or FreightCarSpec(),
        parallel_gap=50.0,
        bogie_tol_tons=2.0,
        max_weight_diff_for_pair=1.0,
        force_same_outer_diameter_for_pair=True,
        sort_products_by_weight=True,
        allow_parallel_pairs=True,
        use_balance_aware_scoring=balance_aware,
    )


class OptimisationLoadingCoreTest(unittest.TestCase):
    def test_generate_loading_blocks_pairs_near_diameter_products(self):
        blocks = generate_loading_blocks(
            products=[
                product("P001", weight=24.41, diameter=2015.0),
                product("P002", weight=24.45, diameter=2006.0),
            ],
            spec=FreightCarSpec(),
            max_weight_diff_for_pair=1.0,
            max_outer_diameter_diff_for_pair=100.0,
            force_same_outer_diameter_for_pair=True,
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "parallel_pair")
        self.assertEqual(blocks[0].length, 2015.0)
        self.assertEqual(blocks[0].width, 2000.0)
        self.assertAlmostEqual(blocks[0].weight, 48.86)

    def test_build_initial_feasible_solution_uses_loading_blocks(self):
        spec = FreightCarSpec()
        blocks = generate_loading_blocks(
            products=[
                product("P001", weight=24.41, diameter=2015.0),
                product("P002", weight=24.45, diameter=2006.0),
                product("P003", weight=10.0, diameter=1400.0),
            ],
            spec=spec,
            max_weight_diff_for_pair=1.0,
            max_outer_diameter_diff_for_pair=100.0,
            force_same_outer_diameter_for_pair=True,
        )

        cars = build_initial_feasible_solution(
            loading_blocks=blocks,
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
        )

        wagon = {
            "wagon_id": "W001",
            "length": spec.length,
            "width": spec.width,
            "max_weight": spec.max_weight * 1000.0,
            "current_weight": sum(
                placed.product.gross_weight * 1000.0
                for placed in cars[0]
            ),
            "products": [
                positioned_to_visual_product(
                    placed,
                    car_width=spec.width,
                )
                for placed in cars[0]
            ],
        }

        report = validate_wagon_layout(wagon)

        self.assertEqual(len(cars), 1)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["overlap_violation_count"], 0)

    def test_merge_low_utilisation_cars_combines_valid_candidates(self):
        spec = FreightCarSpec()
        first_car = build_initial_feasible_solution(
            loading_blocks=generate_loading_blocks(
                products=[
                    product("P001", weight=8.0, diameter=1200.0),
                ],
                spec=spec,
                allow_parallel_pairs=True,
            ),
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
        )[0]
        second_car = build_initial_feasible_solution(
            loading_blocks=generate_loading_blocks(
                products=[
                    product("P002", weight=8.2, diameter=1210.0),
                ],
                spec=spec,
                allow_parallel_pairs=True,
            ),
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
        )[0]

        merged = merge_low_utilisation_cars(
            cars=[first_car, second_car],
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
            max_weight_diff_for_pair=1.0,
            max_outer_diameter_diff_for_pair=100.0,
            force_same_outer_diameter_for_pair=True,
            sort_products_by_weight=True,
            allow_parallel_pairs=True,
            utilisation_threshold=0.35,
        )

        wagon = {
            "wagon_id": "W001",
            "length": spec.length,
            "width": spec.width,
            "max_weight": spec.max_weight * 1000.0,
            "current_weight": sum(
                placed.product.gross_weight * 1000.0
                for placed in merged[0]
            ),
            "products": [
                positioned_to_visual_product(
                    placed,
                    car_width=spec.width,
                )
                for placed in merged[0]
            ],
        }

        report = validate_wagon_layout(wagon)

        self.assertEqual(len(merged), 1)
        self.assertTrue(report["is_valid"])
        self.assertEqual(report["overlap_violation_count"], 0)

    def test_wagon_reassignment_keeps_different_destinations_separate(self):
        spec = FreightCarSpec()
        first_car = build_initial_feasible_solution(
            loading_blocks=generate_loading_blocks(
                products=[
                    product(
                        "P001",
                        weight=8.0,
                        diameter=1200.0,
                        destination="Shanghai",
                    ),
                ],
                spec=spec,
                allow_parallel_pairs=True,
            ),
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
        )[0]
        second_car = build_initial_feasible_solution(
            loading_blocks=generate_loading_blocks(
                products=[
                    product(
                        "P002",
                        weight=8.1,
                        diameter=1210.0,
                        destination="Beijing",
                    ),
                ],
                spec=spec,
                allow_parallel_pairs=True,
            ),
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
        )[0]

        improved = improve_by_wagon_reassignment(
            cars=[first_car, second_car],
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
            max_weight_diff_for_pair=1.0,
            max_outer_diameter_diff_for_pair=100.0,
            force_same_outer_diameter_for_pair=True,
            sort_products_by_weight=True,
            allow_parallel_pairs=True,
            max_passes=2,
        )

        destinations_by_car = [
            {
                positioned.product.destination
                for positioned in car
            }
            for car in improved
        ]

        self.assertEqual(len(improved), 2)
        self.assertIn({"Shanghai"}, destinations_by_car)
        self.assertIn({"Beijing"}, destinations_by_car)

    def test_wagon_reassignment_returns_valid_same_destination_solution(self):
        spec = FreightCarSpec()
        cars = pack(
            [
                product("P001", weight=8.0, diameter=1200.0),
                product("P002", weight=8.2, diameter=1210.0),
                product("P003", weight=8.4, diameter=1220.0),
                product("P004", weight=8.6, diameter=1230.0),
            ],
            spec=spec,
        )

        improved = improve_by_wagon_reassignment(
            cars=cars,
            spec=spec,
            parallel_gap=50.0,
            bogie_tol_tons=2.0,
            use_balance_aware_scoring=True,
            max_weight_diff_for_pair=1.0,
            max_outer_diameter_diff_for_pair=100.0,
            force_same_outer_diameter_for_pair=True,
            sort_products_by_weight=True,
            allow_parallel_pairs=True,
            max_passes=2,
        )

        wagons = []

        for index, car in enumerate(improved, start=1):
            wagons.append(
                {
                    "wagon_id": f"W{index:03d}",
                    "length": spec.length,
                    "width": spec.width,
                    "max_weight": spec.max_weight * 1000.0,
                    "current_weight": sum(
                        positioned.product.gross_weight * 1000.0
                        for positioned in car
                    ),
                    "products": [
                        positioned_to_visual_product(
                            positioned,
                            car_width=spec.width,
                        )
                        for positioned in car
                    ],
                }
            )

        summary = validate_loading_solution(
            wagons=wagons,
            expected_material_numbers=[
                "P001",
                "P002",
                "P003",
                "P004",
            ],
        )

        self.assertTrue(summary["is_valid"])

    def test_simulated_annealing_perturbation_preserves_products(self):
        agent = OptimisationAgent(seed=42)
        products = [
            product("P001", weight=8.0),
            product("P002", weight=8.1),
            product("P003", weight=8.2),
            product("P004", weight=8.3),
        ]
        before_ids = sorted(item.material_no for item in products)

        move_type = agent.perturb_products_for_simulated_annealing(
            products
        )
        after_ids = sorted(item.material_no for item in products)

        self.assertIn(
            move_type,
            {"swap_products", "move_product", "reverse_segment"},
        )
        self.assertEqual(after_ids, before_ids)

    def test_simulated_annealing_logs_move_statistics(self):
        agent = OptimisationAgent(seed=42)
        products = [
            product("P001", weight=8.0, diameter=1200.0),
            product("P002", weight=8.1, diameter=1210.0),
            product("P003", weight=8.2, diameter=1220.0),
            product("P004", weight=8.3, diameter=1230.0),
        ]
        execution_plan = resolve_strategy_execution_plan(
            strategy_plan={
                "selected_strategy": "greedy_plus_simulated_annealing",
                "optimisation_config": {
                    "use_simulated_annealing": True,
                    "sa_iterations": 5,
                },
            },
            products=products,
        )

        packed_groups, log = agent.run_simulated_annealing(
            grouped_products={"Shanghai": products},
            execution_plan=execution_plan,
        )
        wagons = agent.convert_packed_groups_to_wagons(packed_groups)
        summary = validate_loading_solution(
            wagons=wagons,
            expected_material_numbers=[
                item.material_no for item in products
            ],
        )

        self.assertTrue(log["executed"])
        self.assertEqual(log["iterations_requested"], 5)
        self.assertGreaterEqual(
            sum(log["move_attempt_counts"].values()),
            1,
        )
        self.assertTrue(summary["is_valid"])

    def test_algorithm_policy_skips_sa_when_time_budget_is_tight(self):
        agent = OptimisationAgent(seed=42)
        products = [
            product(f"P{index:03d}", weight=8.0 + index * 0.1)
            for index in range(1, 18)
        ]
        strategy_plan = {
            "selected_strategy": "greedy_plus_simulated_annealing",
            "optimisation_config": {
                "use_simulated_annealing": True,
                "sa_iterations": 40,
                "time_budget_seconds": 5,
            },
        }
        execution_plan = resolve_strategy_execution_plan(
            strategy_plan=strategy_plan,
            products=products,
        )

        policy = agent.resolve_algorithm_policy(
            strategy_plan=strategy_plan,
            products=products,
            execution_plan=execution_plan,
        )
        agent.apply_algorithm_policy(execution_plan, policy)

        self.assertFalse(policy.use_simulated_annealing)
        self.assertFalse(execution_plan.use_simulated_annealing)
        self.assertEqual(policy.sa_iterations, 0)
        self.assertIn(
            "simulated_annealing",
            [item["step"] for item in policy.skipped_steps],
        )

    def test_algorithm_policy_can_enable_sa_for_larger_batch(self):
        agent = OptimisationAgent(seed=42)
        products = [
            product(f"P{index:03d}", weight=7.0 + index * 0.05)
            for index in range(1, 31)
        ]
        strategy_plan = {
            "selected_strategy": "balance_aware_greedy",
            "optimisation_config": {
                "use_simulated_annealing": False,
                "time_budget_seconds": 20,
            },
        }
        execution_plan = resolve_strategy_execution_plan(
            strategy_plan=strategy_plan,
            products=products,
        )

        policy = agent.resolve_algorithm_policy(
            strategy_plan=strategy_plan,
            products=products,
            execution_plan=execution_plan,
        )
        agent.apply_algorithm_policy(execution_plan, policy)

        self.assertTrue(policy.use_simulated_annealing)
        self.assertTrue(execution_plan.use_simulated_annealing)
        self.assertGreaterEqual(policy.sa_iterations, 30)
        self.assertIn(
            "simulated_annealing",
            policy.selected_pipeline,
        )

    def test_council_candidate_can_disable_policy_strategy_override(self):
        agent = OptimisationAgent(seed=42)
        products = [
            product(f"P{index:03d}", weight=7.0 + index * 0.05)
            for index in range(1, 31)
        ]
        strategy_plan = {
            "selected_strategy": "destination_grouping_greedy",
            "optimisation_config": {
                "use_simulated_annealing": False,
                "time_budget_seconds": 20,
                "allow_policy_strategy_override": False,
            },
        }
        execution_plan = resolve_strategy_execution_plan(
            strategy_plan=strategy_plan,
            products=products,
        )

        policy = agent.resolve_algorithm_policy(
            strategy_plan=strategy_plan,
            products=products,
            execution_plan=execution_plan,
        )
        agent.apply_algorithm_policy(execution_plan, policy)

        self.assertFalse(policy.use_simulated_annealing)
        self.assertFalse(execution_plan.use_simulated_annealing)

    def test_balance_aware_loading_creates_parallel_pair_inside_wagon(self):
        cars = pack(
            [
                product("P001", weight=12.0),
                product("P002", weight=12.5),
            ]
        )

        self.assertEqual(len(cars), 1)
        self.assertEqual(
            [placed.product.material_no for placed in cars[0]],
            ["P002", "P001"],
        )
        self.assertEqual({placed.lane for placed in cars[0]}, {0, 1})
        self.assertAlmostEqual(cars[0][0].x_mm, cars[0][1].x_mm)

        visual_products = [
            positioned_to_visual_product(placed, car_width=3000.0)
            for placed in cars[0]
        ]

        for visual_product in visual_products:
            self.assertGreaterEqual(visual_product["x"], 0.0)
            self.assertLessEqual(
                visual_product["x"] + visual_product["length"],
                13000.0,
            )
            self.assertGreaterEqual(visual_product["y"], 0.0)
            self.assertLessEqual(
                visual_product["y"] + visual_product["width"],
                3000.0,
            )

    def test_loading_splits_wagons_when_weight_capacity_is_exceeded(self):
        cars = pack(
            [
                product("P001", weight=20.0),
                product("P002", weight=20.0),
                product("P003", weight=25.0),
            ],
            balance_aware=False,
        )

        self.assertEqual(len(cars), 2)
        self.assertLessEqual(
            sum(placed.product.gross_weight for placed in cars[0]),
            60.0,
        )
        self.assertLessEqual(
            sum(placed.product.gross_weight for placed in cars[1]),
            60.0,
        )

    def test_loading_rejects_single_product_that_cannot_fit_spec(self):
        too_long = product(
            "P_TOO_LONG",
            weight=10.0,
            diameter=14000.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Cannot place block safely",
        ):
            pack([too_long], spec=FreightCarSpec(length=13000.0))

    def test_validator_accepts_non_overlapping_generated_layout(self):
        cars = pack(
            [
                product("P001", weight=12.0),
                product("P002", weight=12.5),
                product("P003", weight=8.0),
            ]
        )

        wagon = {
            "wagon_id": "W001",
            "length": 13000.0,
            "width": 3000.0,
            "max_weight": 60000.0,
            "current_weight": sum(
                placed.product.gross_weight * 1000.0
                for placed in cars[0]
            ),
            "products": [
                positioned_to_visual_product(placed, car_width=3000.0)
                for placed in cars[0]
            ],
        }

        report = validate_wagon_layout(wagon)

        self.assertTrue(report["is_valid"])
        self.assertEqual(report["overlap_violation_count"], 0)
        self.assertEqual(report["boundary_violation_count"], 0)

    def test_validator_rejects_overlapping_products(self):
        wagon = {
            "wagon_id": "W001",
            "length": 13000.0,
            "width": 3000.0,
            "max_weight": 60000.0,
            "current_weight": 20000.0,
            "products": [
                {
                    "material_no": "P001",
                    "x": 1000.0,
                    "y": 500.0,
                    "placed_length": 1200.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
                {
                    "material_no": "P002",
                    "x": 1500.0,
                    "y": 800.0,
                    "placed_length": 1200.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
            ],
        }

        report = validate_wagon_layout(wagon)

        self.assertFalse(report["is_valid"])
        self.assertEqual(report["overlap_violation_count"], 1)
        self.assertEqual(report["violations"][0]["type"], "overlap")

    def test_evaluate_loading_solution_reports_score_components(self):
        wagon = {
            "wagon_id": "W001",
            "length": 13000.0,
            "width": 3000.0,
            "max_weight": 60000.0,
            "current_weight": 10000.0,
            "area_utilisation": 0.1,
            "balance_metrics": {
                "front_rear_penalty": 0.1,
                "left_right_penalty": 0.0,
                "diagonal_penalty": 0.0,
            },
            "products": [
                {
                    "material_no": "P001",
                    "destination": "Shanghai",
                    "x": 0.0,
                    "y": 0.0,
                    "placed_length": 1000.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
            ],
        }

        score = evaluate_loading_solution(
            wagons=[wagon],
            weights=ObjectiveWeights(),
            expected_material_numbers=["P001"],
            enforce_hard_constraints=True,
        )

        self.assertTrue(score["validation_summary"]["is_valid"])
        self.assertIn("components", score)
        self.assertGreater(score["components"]["wagon_count"], 0.0)
        self.assertEqual(score["overlap_violation_count"], 0)

    def test_evaluate_loading_solution_enforces_hard_constraints(self):
        wagon = {
            "wagon_id": "W001",
            "length": 13000.0,
            "width": 3000.0,
            "max_weight": 60000.0,
            "current_weight": 20000.0,
            "products": [
                {
                    "material_no": "P001",
                    "destination": "Shanghai",
                    "x": 0.0,
                    "y": 0.0,
                    "placed_length": 1000.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
                {
                    "material_no": "P002",
                    "destination": "Shanghai",
                    "x": 500.0,
                    "y": 500.0,
                    "placed_length": 1000.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
            ],
        }

        with self.assertRaisesRegex(
            ValueError,
            "violates hard constraints",
        ):
            evaluate_loading_solution(
                wagons=[wagon],
                weights=ObjectiveWeights(),
                expected_material_numbers=["P001", "P002"],
                enforce_hard_constraints=True,
            )

    def test_solution_validator_detects_missing_and_duplicate_products(self):
        wagon = {
            "wagon_id": "W001",
            "length": 13000.0,
            "width": 3000.0,
            "max_weight": 60000.0,
            "current_weight": 20000.0,
            "products": [
                {
                    "material_no": "P001",
                    "x": 0.0,
                    "y": 0.0,
                    "placed_length": 1000.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
                {
                    "material_no": "P001",
                    "x": 1000.0,
                    "y": 0.0,
                    "placed_length": 1000.0,
                    "placed_width": 1000.0,
                    "weight": 10000.0,
                },
            ],
        }

        summary = validate_loading_solution(
            wagons=[wagon],
            expected_material_numbers=["P001", "P002"],
        )

        self.assertFalse(summary["is_valid"])
        self.assertEqual(summary["duplicate_product_ids"], ["P001"])
        self.assertEqual(summary["missing_product_ids"], ["P002"])
        self.assertEqual(summary["coverage_violation_count"], 2)


if __name__ == "__main__":
    unittest.main()
