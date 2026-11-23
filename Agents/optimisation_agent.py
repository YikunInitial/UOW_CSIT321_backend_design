# Agents/optimisation_agent.py

"""
Optimisation Agent - Strategy-Driven Steel Coil Loading
=======================================================

Input:
    products.json
    strategy_plan.json

Output:
    loading_solution.json

Pipeline:
    LLM strategy plan
        -> strategy validation and normalisation
        -> executable optimisation controls
        -> loading solution
        -> strategy execution audit trail

Supported strategies:
- simple_greedy
- destination_grouping_greedy
- balance_aware_greedy
- destination_grouping_plus_balance_aware_greedy
- greedy_plus_simulated_annealing
- rule_based_loading

Important:
- Mandatory safety constraints remain enforced regardless of LLM output:
  weight limit, boundary fit, overlap-safe placement.
- The optimiser records requested_strategy and applied_strategy separately.
- Single-destination batches do not pretend to perform destination grouping.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Domain objects
# =============================================================================


@dataclass(frozen=True)
class Product:
    material_no: str
    destination: str
    gross_weight: float
    width: float
    outer_diameter: float
    thickness: Optional[float] = None
    net_weight: Optional[float] = None


@dataclass
class FreightCarSpec:
    length: float = 13000.0
    width: float = 3000.0
    max_weight: float = 60.0


@dataclass(frozen=True)
class ObjectiveWeights:
    wagon_count: float = 100000.0
    validation_violation: float = 1000000.0
    unused_area: float = 1000.0
    front_rear_balance: float = 15000.0
    left_right_balance: float = 10000.0
    diagonal_balance: float = 8000.0
    mixed_destination: float = 5000.0
    operational_complexity: float = 500.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "wagon_count": self.wagon_count,
            "validation_violation": self.validation_violation,
            "unused_area": self.unused_area,
            "front_rear_balance": self.front_rear_balance,
            "left_right_balance": self.left_right_balance,
            "diagonal_balance": self.diagonal_balance,
            "mixed_destination": self.mixed_destination,
            "operational_complexity": self.operational_complexity,
        }


@dataclass
class Positioned:
    product: Product
    x_mm: float
    lane: Optional[int]


@dataclass
class LoadingBlock:
    products: List[Product]
    block_type: str
    length: float
    width: float
    weight: float
    destination: str
    score: float
    reason: str


@dataclass
class StrategyExecutionPlan:
    requested_strategy: str
    requested_scenario_type: str
    applied_strategy: str
    actual_batch_scenario: str

    group_by_destination: bool
    sort_by_weight_desc: bool
    use_balance_aware_scoring: bool
    use_simulated_annealing: bool
    use_rule_based_constraints: bool
    allow_parallel_pairs: bool

    strict_pairing_rules: bool
    sa_iterations: int

    resolution_notes: List[str] = field(default_factory=list)
    fallback_reasons: List[str] = field(default_factory=list)


@dataclass
class AlgorithmPolicy:
    mode: str
    time_budget_seconds: Optional[float]
    product_count: int
    destination_count: int
    complexity_class: str
    selected_pipeline: List[str]
    skipped_steps: List[Dict[str, str]]
    decision_notes: List[str]
    enable_low_utilisation_merge: bool
    enable_wagon_reassignment: bool
    use_simulated_annealing: bool
    sa_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "time_budget_seconds": self.time_budget_seconds,
            "product_count": self.product_count,
            "destination_count": self.destination_count,
            "complexity_class": self.complexity_class,
            "selected_pipeline": self.selected_pipeline,
            "skipped_steps": self.skipped_steps,
            "decision_notes": self.decision_notes,
            "enable_low_utilisation_merge": (
                self.enable_low_utilisation_merge
            ),
            "enable_wagon_reassignment": (
                self.enable_wagon_reassignment
            ),
            "use_simulated_annealing": self.use_simulated_annealing,
            "sa_iterations": self.sa_iterations,
        }


# =============================================================================
# Utility functions
# =============================================================================


SUPPORTED_STRATEGIES = {
    "simple_greedy",
    "destination_grouping_greedy",
    "balance_aware_greedy",
    "destination_grouping_plus_balance_aware_greedy",
    "greedy_plus_simulated_annealing",
    "rule_based_loading",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()

        if not text:
            return default

        return float(text)

    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        return int(value)

    except Exception:
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """
    Strict bool parsing.

    Avoids the common Python bug:
        bool("false") == True
    """

    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "on"}:
        return True

    if text in {"false", "0", "no", "n", "off", ""}:
        return False

    return default


def first_present(data: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return None


def normalise_destination(raw: Any) -> str:
    if raw is None:
        return "UNKNOWN_DEST"

    destination = str(raw).strip()

    return destination if destination else "UNKNOWN_DEST"


def normalise_strategy_name(raw: Any) -> str:
    if raw is None:
        return "balance_aware_greedy"

    strategy = str(raw).strip().lower()

    if strategy in SUPPORTED_STRATEGIES:
        return strategy

    return "balance_aware_greedy"


def dedupe_by_material_no(
    products: List[Product],
) -> Tuple[List[Product], List[str]]:
    """
    Keep one record per material_no.

    This remains as a defensive safeguard for production data, but the
    event-driven workflow should already guarantee no duplicate products.
    """

    best: Dict[str, Product] = {}
    dropped: List[str] = []

    def score(product: Product) -> Tuple[float, float, float]:
        return (
            product.gross_weight,
            product.outer_diameter,
            product.width,
        )

    for product in products:
        existing = best.get(product.material_no)

        if existing is None:
            best[product.material_no] = product
            continue

        if score(product) > score(existing):
            dropped.append(existing.material_no)
            best[product.material_no] = product
        else:
            dropped.append(product.material_no)

    seen = set()
    output: List[Product] = []

    for product in products:
        chosen = best.get(product.material_no)

        if chosen == product and product.material_no not in seen:
            output.append(product)
            seen.add(product.material_no)

    if len(output) != len(best):
        output = list(best.values())

    return output, dropped


def group_by_destination(
    products: List[Product],
) -> Dict[str, List[Product]]:
    grouped: Dict[str, List[Product]] = defaultdict(list)

    for product in products:
        grouped[product.destination].append(product)

    return dict(grouped)


def sort_by_weight_desc(products: List[Product]) -> List[Product]:
    return sorted(
        products,
        key=lambda product: product.gross_weight,
        reverse=True,
    )


def validate_products_fit_spec(
    products: List[Product],
    spec: FreightCarSpec,
) -> None:
    invalid: List[str] = []

    for product in products:
        if product.outer_diameter > spec.length:
            invalid.append(
                f"{product.material_no}: outer_diameter="
                f"{product.outer_diameter} exceeds car_length={spec.length}"
            )

        if product.width > spec.width:
            invalid.append(
                f"{product.material_no}: width={product.width} "
                f"exceeds car_width={spec.width}"
            )

        if product.gross_weight > spec.max_weight:
            invalid.append(
                f"{product.material_no}: gross_weight="
                f"{product.gross_weight} exceeds max_weight="
                f"{spec.max_weight}"
            )

    if invalid:
        details = "\n".join(invalid[:10])

        raise ValueError(
            "Some products cannot fit safely in an empty freight car:\n"
            f"{details}"
        )


# =============================================================================
# Strategy interpretation
# =============================================================================


def strategy_profile(strategy_name: str) -> Dict[str, Any]:
    """
    Base executable controls for each strategy.

    The LLM may also supply optimisation_config. Its valid values are merged
    later, then the final applied strategy is derived from actual controls.
    """

    profiles: Dict[str, Dict[str, Any]] = {
        "simple_greedy": {
            "group_by_destination": False,
            "sort_by_weight_desc": False,
            "use_balance_aware_scoring": False,
            "use_simulated_annealing": False,
            "use_rule_based_constraints": True,
            "allow_parallel_pairs": False,
            "strict_pairing_rules": False,
            "sa_iterations": 0,
        },
        "destination_grouping_greedy": {
            "group_by_destination": True,
            "sort_by_weight_desc": True,
            "use_balance_aware_scoring": False,
            "use_simulated_annealing": False,
            "use_rule_based_constraints": True,
            "allow_parallel_pairs": True,
            "strict_pairing_rules": True,
            "sa_iterations": 0,
        },
        "balance_aware_greedy": {
            "group_by_destination": False,
            "sort_by_weight_desc": True,
            "use_balance_aware_scoring": True,
            "use_simulated_annealing": False,
            "use_rule_based_constraints": True,
            "allow_parallel_pairs": True,
            "strict_pairing_rules": True,
            "sa_iterations": 0,
        },
        "destination_grouping_plus_balance_aware_greedy": {
            "group_by_destination": True,
            "sort_by_weight_desc": True,
            "use_balance_aware_scoring": True,
            "use_simulated_annealing": False,
            "use_rule_based_constraints": True,
            "allow_parallel_pairs": True,
            "strict_pairing_rules": True,
            "sa_iterations": 0,
        },
        "greedy_plus_simulated_annealing": {
            "group_by_destination": True,
            "sort_by_weight_desc": True,
            "use_balance_aware_scoring": True,
            "use_simulated_annealing": True,
            "use_rule_based_constraints": True,
            "allow_parallel_pairs": True,
            "strict_pairing_rules": True,
            "sa_iterations": 40,
        },
        "rule_based_loading": {
            "group_by_destination": True,
            "sort_by_weight_desc": True,
            "use_balance_aware_scoring": True,
            "use_simulated_annealing": False,
            "use_rule_based_constraints": True,
            "allow_parallel_pairs": True,
            "strict_pairing_rules": True,
            "sa_iterations": 0,
        },
    }

    return dict(profiles[strategy_name])


def infer_applied_strategy(
    group_by_destination: bool,
    use_balance_aware_scoring: bool,
    use_simulated_annealing: bool,
    requested_strategy: str,
    actual_destination_count: int,
) -> str:
    """
    Return a truthful strategy label that matches actual execution.
    """

    if use_simulated_annealing:
        return "greedy_plus_simulated_annealing"

    if requested_strategy == "rule_based_loading":
        return "rule_based_loading"

    if actual_destination_count <= 1:
        if use_balance_aware_scoring:
            return "balance_aware_greedy"

        return "simple_greedy"

    if group_by_destination and use_balance_aware_scoring:
        return "destination_grouping_plus_balance_aware_greedy"

    if group_by_destination:
        return "destination_grouping_greedy"

    if use_balance_aware_scoring:
        return "balance_aware_greedy"

    return "simple_greedy"


def resolve_strategy_execution_plan(
    strategy_plan: Dict[str, Any],
    products: List[Product],
) -> StrategyExecutionPlan:
    """
    Translate LLM output into a safe, executable plan.

    A batch with one destination is always described as single_destination_batch.
    The optimiser does not falsely claim destination grouping in that case.
    """

    requested_strategy = normalise_strategy_name(
        strategy_plan.get("selected_strategy")
    )

    requested_scenario_type = str(
        strategy_plan.get("scenario_type", "")
    ).strip()

    destination_count = len({product.destination for product in products})

    actual_batch_scenario = (
        "single_destination_batch"
        if destination_count <= 1
        else "multi_destination_batch"
    )

    profile = strategy_profile(requested_strategy)

    raw_config = strategy_plan.get("optimisation_config", {})

    if not isinstance(raw_config, dict):
        raw_config = {}

    resolution_notes: List[str] = []
    fallback_reasons: List[str] = []

    controls = dict(profile)

    configurable_booleans = [
        "group_by_destination",
        "sort_by_weight_desc",
        "use_balance_aware_scoring",
        "use_simulated_annealing",
        "use_rule_based_constraints",
    ]

    for key in configurable_booleans:
        if key in raw_config:
            controls[key] = safe_bool(
                raw_config.get(key),
                default=bool(profile[key]),
            )

    if "sa_iterations" in raw_config:
        controls["sa_iterations"] = max(
            0,
            min(
                safe_int(
                    raw_config.get("sa_iterations"),
                    default=int(profile["sa_iterations"]),
                ),
                200,
            ),
        )

    if "allow_parallel_pairs" in raw_config:
        controls["allow_parallel_pairs"] = safe_bool(
            raw_config.get("allow_parallel_pairs"),
            default=bool(profile["allow_parallel_pairs"]),
        )

    if requested_strategy == "simple_greedy":
        controls["allow_parallel_pairs"] = False

    if controls["use_simulated_annealing"]:
        controls["use_balance_aware_scoring"] = True

        if controls["sa_iterations"] <= 0:
            controls["sa_iterations"] = 40

    if destination_count <= 1 and controls["group_by_destination"]:
        resolution_notes.append(
            "group_by_destination was requested, but the current batch "
            "contains one destination only; grouping has no operational "
            "effect for this batch."
        )

    if destination_count <= 1:
        controls["group_by_destination"] = False

    if not controls["use_rule_based_constraints"]:
        resolution_notes.append(
            "LLM requested use_rule_based_constraints=False, but mandatory "
            "safety constraints remain enforced: weight limit, boundaries, "
            "and overlap-safe placement."
        )

    applied_strategy = infer_applied_strategy(
        group_by_destination=controls["group_by_destination"],
        use_balance_aware_scoring=controls[
            "use_balance_aware_scoring"
        ],
        use_simulated_annealing=controls["use_simulated_annealing"],
        requested_strategy=requested_strategy,
        actual_destination_count=destination_count,
    )

    if requested_strategy != applied_strategy:
        resolution_notes.append(
            f"Requested strategy '{requested_strategy}' was normalised to "
            f"applied strategy '{applied_strategy}' based on actual batch "
            "destinations and executable optimisation controls."
        )

    if (
        requested_scenario_type
        and requested_scenario_type != actual_batch_scenario
    ):
        resolution_notes.append(
            f"LLM scenario_type '{requested_scenario_type}' differs from "
            f"actual batch scenario '{actual_batch_scenario}'."
        )

    return StrategyExecutionPlan(
        requested_strategy=requested_strategy,
        requested_scenario_type=requested_scenario_type,
        applied_strategy=applied_strategy,
        actual_batch_scenario=actual_batch_scenario,
        group_by_destination=controls["group_by_destination"],
        sort_by_weight_desc=controls["sort_by_weight_desc"],
        use_balance_aware_scoring=controls[
            "use_balance_aware_scoring"
        ],
        use_simulated_annealing=controls["use_simulated_annealing"],
        use_rule_based_constraints=controls[
            "use_rule_based_constraints"
        ],
        allow_parallel_pairs=controls["allow_parallel_pairs"],
        strict_pairing_rules=controls["strict_pairing_rules"],
        sa_iterations=controls["sa_iterations"],
        resolution_notes=resolution_notes,
        fallback_reasons=fallback_reasons,
    )


# =============================================================================
# Pairing and core loading logic
# =============================================================================


def build_single_block(product: Product) -> LoadingBlock:
    return LoadingBlock(
        products=[product],
        block_type="single",
        length=product.outer_diameter,
        width=product.width,
        weight=product.gross_weight,
        destination=product.destination,
        score=100.0,
        reason="single product block",
    )


def build_parallel_pair_block(
    product_one: Product,
    product_two: Product,
    score: float,
    reason: str,
) -> LoadingBlock:
    destination = (
        product_one.destination
        if product_one.destination == product_two.destination
        else "MIXED_DESTINATION"
    )

    return LoadingBlock(
        products=[product_one, product_two],
        block_type="parallel_pair",
        length=max(
            product_one.outer_diameter,
            product_two.outer_diameter,
        ),
        width=product_one.width + product_two.width,
        weight=product_one.gross_weight + product_two.gross_weight,
        destination=destination,
        score=score,
        reason=reason,
    )


def score_parallel_pair_candidate(
    product_one: Product,
    product_two: Product,
    max_weight_diff: float,
    max_outer_diameter_diff: float,
) -> Tuple[float, str]:
    weight_diff = abs(
        product_one.gross_weight - product_two.gross_weight
    )
    diameter_diff = abs(
        product_one.outer_diameter - product_two.outer_diameter
    )

    same_destination_bonus = (
        -5.0
        if product_one.destination == product_two.destination
        else 5.0
    )

    weight_ratio = (
        weight_diff / max(max_weight_diff, 1e-9)
        if max_weight_diff > 0
        else weight_diff
    )
    diameter_ratio = (
        diameter_diff / max(max_outer_diameter_diff, 1e-9)
        if max_outer_diameter_diff > 0
        else diameter_diff
    )

    score = weight_ratio * 10.0 + diameter_ratio + same_destination_bonus

    reason = (
        "parallel pair candidate: "
        f"weight_diff={round(weight_diff, 4)}, "
        f"outer_diameter_diff={round(diameter_diff, 4)}, "
        f"same_destination={product_one.destination == product_two.destination}"
    )

    return score, reason


def generate_loading_blocks(
    products: List[Product],
    spec: FreightCarSpec,
    max_weight_diff_for_pair: float = 1.0,
    max_outer_diameter_diff_for_pair: float = 100.0,
    force_same_outer_diameter_for_pair: bool = True,
    sort_products_by_weight: bool = True,
    allow_parallel_pairs: bool = True,
) -> List[LoadingBlock]:
    """
    Build loadable blocks before placement.

    This is the first explicit optimisation layer: it identifies feasible
    arrangement candidates independently from the wagon-placement algorithm.
    Later phases can extend this with sequence blocks, merge candidates, or
    richer operational constraints without rewriting the placement loop.
    """

    ordered_products = (
        sort_by_weight_desc(products)
        if sort_products_by_weight
        else list(products)
    )

    if not allow_parallel_pairs:
        return [
            build_single_block(product)
            for product in ordered_products
        ]

    used = set()
    blocks: List[LoadingBlock] = []

    for index, product_one in enumerate(ordered_products):
        if index in used:
            continue

        best_candidate_index: Optional[int] = None
        best_candidate_score: Optional[float] = None
        best_candidate_reason = ""

        for other_index in range(index + 1, len(ordered_products)):
            if other_index in used:
                continue

            product_two = ordered_products[other_index]

            width_ok = (
                product_one.width + product_two.width
            ) <= spec.width

            if not width_ok:
                continue

            weight_diff = abs(
                product_one.gross_weight
                - product_two.gross_weight
            )

            if weight_diff > max_weight_diff_for_pair:
                continue

            diameter_diff = abs(
                product_one.outer_diameter
                - product_two.outer_diameter
            )

            if force_same_outer_diameter_for_pair:
                if diameter_diff > max_outer_diameter_diff_for_pair:
                    continue
            elif (
                max_outer_diameter_diff_for_pair > 0
                and diameter_diff > max_outer_diameter_diff_for_pair
            ):
                continue

            score, reason = score_parallel_pair_candidate(
                product_one=product_one,
                product_two=product_two,
                max_weight_diff=max_weight_diff_for_pair,
                max_outer_diameter_diff=(
                    max_outer_diameter_diff_for_pair
                ),
            )

            if (
                best_candidate_score is None
                or score < best_candidate_score
            ):
                best_candidate_index = other_index
                best_candidate_score = score
                best_candidate_reason = reason

        if best_candidate_index is None:
            used.add(index)
            blocks.append(build_single_block(product_one))
            continue

        product_two = ordered_products[best_candidate_index]

        used.add(index)
        used.add(best_candidate_index)

        blocks.append(
            build_parallel_pair_block(
                product_one=product_one,
                product_two=product_two,
                score=best_candidate_score or 0.0,
                reason=best_candidate_reason,
            )
        )

    return blocks


def find_parallel_pairs(
    products: List[Product],
    car_width: float,
    max_weight_diff: float = 1.0,
    max_outer_diameter_diff: float = 0.0,
    force_same_outer_diameter: bool = True,
    sort_products_by_weight: bool = True,
    allow_parallel_pairs: bool = True,
) -> List[List[Product]]:
    """
    Backward-compatible wrapper for the older pairing API.

    New code should use generate_loading_blocks() so candidate metadata remains
    available for later optimisation stages.
    """

    spec = FreightCarSpec(width=car_width)

    blocks = generate_loading_blocks(
        products=products,
        spec=spec,
        max_weight_diff_for_pair=max_weight_diff,
        max_outer_diameter_diff_for_pair=max_outer_diameter_diff,
        force_same_outer_diameter_for_pair=(
            force_same_outer_diameter
        ),
        sort_products_by_weight=sort_products_by_weight,
        allow_parallel_pairs=allow_parallel_pairs,
    )

    return [block.products for block in blocks]


def compute_bogie_balance(
    placed: List[Positioned],
    car_length: float,
) -> Tuple[float, float]:
    """
    Compute front/rear bogie loads using moment balance.
    """

    total_weight = sum(
        positioned.product.gross_weight
        for positioned in placed
    )

    if total_weight <= 0:
        return 0.0, 0.0

    moment = sum(
        positioned.product.gross_weight * positioned.x_mm
        for positioned in placed
    )

    rear_bogie = moment / car_length
    front_bogie = total_weight - rear_bogie

    return round(front_bogie, 2), round(rear_bogie, 2)


def is_within_bogie_balance(
    placed: List[Positioned],
    car_length: float,
    tol_tons: float = 2.0,
) -> bool:
    front_bogie, rear_bogie = compute_bogie_balance(
        placed,
        car_length,
    )

    return abs(front_bogie - rear_bogie) <= tol_tons


def smart_shift_to_center_of_mass(
    placed: List[Positioned],
    spec: FreightCarSpec,
) -> List[Positioned]:
    """
    Shift a whole layout toward the vehicle centre of mass where possible.
    """

    total_weight = sum(
        positioned.product.gross_weight
        for positioned in placed
    )

    if total_weight <= 0:
        return placed

    center_of_mass = sum(
        positioned.product.gross_weight * positioned.x_mm
        for positioned in placed
    ) / total_weight

    target_center = spec.length / 2.0
    delta = target_center - center_of_mass

    if abs(delta) < 1e-6:
        return placed

    shifted: List[Positioned] = []

    for positioned in placed:
        diameter = (
            positioned.product.outer_diameter
            if positioned.product.outer_diameter > 0
            else 1000.0
        )

        new_x = positioned.x_mm + delta

        if new_x - diameter / 2.0 < 0:
            return placed

        if new_x + diameter / 2.0 > spec.length:
            return placed

        shifted.append(
            Positioned(
                product=positioned.product,
                x_mm=new_x,
                lane=positioned.lane,
            )
        )

    return shifted


def block_length_and_weight(
    block: List[Product],
) -> Tuple[float, float]:
    if len(block) == 2:
        product_one, product_two = block

        return (
            max(
                product_one.outer_diameter,
                product_two.outer_diameter,
            ),
            product_one.gross_weight + product_two.gross_weight,
        )

    product = block[0]

    return product.outer_diameter, product.gross_weight


def append_block_to_car(
    current_car: List[Positioned],
    block: List[Product],
    x_mm: float,
) -> None:
    if len(block) == 2:
        current_car.append(
            Positioned(
                product=block[0],
                x_mm=x_mm,
                lane=0,
            )
        )
        current_car.append(
            Positioned(
                product=block[1],
                x_mm=x_mm,
                lane=1,
            )
        )
        return

    current_car.append(
        Positioned(
            product=block[0],
            x_mm=x_mm,
            lane=None,
        )
    )


def build_initial_feasible_solution(
    loading_blocks: List[LoadingBlock],
    spec: FreightCarSpec,
    parallel_gap: float,
    bogie_tol_tons: float,
    use_balance_aware_scoring: bool,
) -> List[List[Positioned]]:
    """
    Place pre-built loading blocks into cars and return an initial feasible
    solution.

    Balance-aware mode:
    - center-first placement
    - alternating front/rear placement
    - bogie balance check
    - centre-of-mass shift

    Simple mode:
    - front-to-rear sequential placement
    - still enforces weight and boundary safety
    """

    cars: List[List[Positioned]] = []
    current_car: List[Positioned] = []
    used_weight = 0.0

    center = spec.length / 2.0
    center_offset = 0.0
    direction = -1

    sequential_cursor = 0.0

    def flush_current_car() -> None:
        nonlocal current_car
        nonlocal used_weight
        nonlocal center_offset
        nonlocal direction
        nonlocal sequential_cursor

        if not current_car:
            return

        final_car = current_car

        if use_balance_aware_scoring:
            if not is_within_bogie_balance(
                placed=final_car,
                car_length=spec.length,
                tol_tons=bogie_tol_tons,
            ):
                final_car = smart_shift_to_center_of_mass(
                    placed=final_car,
                    spec=spec,
                )

        cars.append(final_car)

        current_car = []
        used_weight = 0.0
        center_offset = 0.0
        direction = -1
        sequential_cursor = 0.0

    for loading_block in loading_blocks:
        block = loading_block.products
        block_length = loading_block.length
        block_weight = loading_block.weight

        if block_length <= 0:
            block_length = 1000.0

        if block_weight > spec.max_weight:
            material_numbers = [
                product.material_no
                for product in block
            ]

            raise ValueError(
                "A loading block exceeds the wagon maximum weight: "
                f"products={material_numbers}, weight={block_weight}, "
                f"max_weight={spec.max_weight}"
            )

        if use_balance_aware_scoring:
            tentative_x = center + direction * (
                center_offset + block_length / 2.0
            )

            exceeds_boundary = (
                tentative_x - block_length / 2.0 < 0
                or tentative_x + block_length / 2.0 > spec.length
            )
        else:
            tentative_x = sequential_cursor + block_length / 2.0

            exceeds_boundary = (
                sequential_cursor + block_length > spec.length
            )

        overweight = (
            used_weight + block_weight > spec.max_weight + 1e-9
        )

        if current_car and (exceeds_boundary or overweight):
            flush_current_car()

            if use_balance_aware_scoring:
                tentative_x = center + direction * (
                    center_offset + block_length / 2.0
                )
            else:
                tentative_x = sequential_cursor + block_length / 2.0

        if use_balance_aware_scoring:
            if tentative_x - block_length / 2.0 < 0:
                raise ValueError(
                    "Cannot place block safely after starting a new car."
                )

            if tentative_x + block_length / 2.0 > spec.length:
                raise ValueError(
                    "Cannot place block safely after starting a new car."
                )
        else:
            if sequential_cursor + block_length > spec.length:
                raise ValueError(
                    "Cannot place block safely after starting a new car."
                )

        append_block_to_car(
            current_car=current_car,
            block=block,
            x_mm=tentative_x,
        )

        used_weight += block_weight

        if use_balance_aware_scoring:
            center_offset += block_length + parallel_gap
            direction *= -1
        else:
            sequential_cursor += block_length + parallel_gap

    if current_car:
        flush_current_car()

    return cars


def positioned_car_weight(placed: List[Positioned]) -> float:
    return sum(
        positioned.product.gross_weight
        for positioned in placed
    )


def positioned_car_area_utilisation(
    placed: List[Positioned],
    spec: FreightCarSpec,
) -> float:
    area_total = spec.length * spec.width

    if area_total <= 0:
        return 0.0

    area_used = sum(
        positioned.product.outer_diameter * positioned.product.width
        for positioned in placed
    )

    return area_used / area_total


def positioned_car_to_validation_wagon(
    placed: List[Positioned],
    spec: FreightCarSpec,
    wagon_id: str = "MERGE_CANDIDATE",
) -> Dict[str, Any]:
    visual_products = [
        positioned_to_visual_product(
            positioned,
            spec.width,
        )
        for positioned in placed
    ]

    return {
        "wagon_id": wagon_id,
        "length": spec.length,
        "width": spec.width,
        "max_weight": spec.max_weight * 1000.0,
        "current_weight": positioned_car_weight(placed) * 1000.0,
        "products": visual_products,
    }


def is_valid_positioned_car(
    placed: List[Positioned],
    spec: FreightCarSpec,
) -> bool:
    report = validate_wagon_layout(
        positioned_car_to_validation_wagon(
            placed=placed,
            spec=spec,
        )
    )

    return report["is_valid"]


def try_merge_positioned_cars(
    first_car: List[Positioned],
    second_car: List[Positioned],
    spec: FreightCarSpec,
    parallel_gap: float,
    bogie_tol_tons: float,
    use_balance_aware_scoring: bool,
    max_weight_diff_for_pair: float,
    max_outer_diameter_diff_for_pair: float,
    force_same_outer_diameter_for_pair: bool,
    sort_products_by_weight: bool,
    allow_parallel_pairs: bool,
) -> Optional[List[Positioned]]:
    products = [
        positioned.product
        for positioned in first_car + second_car
    ]

    if sum(product.gross_weight for product in products) > spec.max_weight:
        return None

    loading_blocks = generate_loading_blocks(
        products=products,
        spec=spec,
        max_weight_diff_for_pair=max_weight_diff_for_pair,
        max_outer_diameter_diff_for_pair=(
            max_outer_diameter_diff_for_pair
        ),
        force_same_outer_diameter_for_pair=(
            force_same_outer_diameter_for_pair
        ),
        sort_products_by_weight=sort_products_by_weight,
        allow_parallel_pairs=allow_parallel_pairs,
    )

    try:
        candidate_cars = build_initial_feasible_solution(
            loading_blocks=loading_blocks,
            spec=spec,
            parallel_gap=parallel_gap,
            bogie_tol_tons=bogie_tol_tons,
            use_balance_aware_scoring=use_balance_aware_scoring,
        )
    except ValueError:
        return None

    if len(candidate_cars) != 1:
        return None

    candidate_car = candidate_cars[0]

    if not is_valid_positioned_car(
        placed=candidate_car,
        spec=spec,
    ):
        return None

    return candidate_car


def merge_low_utilisation_cars(
    cars: List[List[Positioned]],
    spec: FreightCarSpec,
    parallel_gap: float,
    bogie_tol_tons: float,
    use_balance_aware_scoring: bool,
    max_weight_diff_for_pair: float,
    max_outer_diameter_diff_for_pair: float,
    force_same_outer_diameter_for_pair: bool,
    sort_products_by_weight: bool,
    allow_parallel_pairs: bool,
    utilisation_threshold: float = 0.35,
) -> List[List[Positioned]]:
    """
    Conservatively merge low-utilisation cars within one destination group.

    A merge is accepted only when the combined products can be repacked into
    one valid car. This keeps the phase independent from the exact heuristic
    used to generate the initial solution.
    """

    merged_cars = list(cars)
    changed = True

    while changed:
        changed = False

        low_utilisation_indices = [
            index
            for index, placed in enumerate(merged_cars)
            if positioned_car_area_utilisation(
                placed=placed,
                spec=spec,
            )
            < utilisation_threshold
        ]

        best_merge: Optional[
            Tuple[int, int, List[Positioned], float]
        ] = None

        for first_pos, first_index in enumerate(
            low_utilisation_indices
        ):
            for second_index in low_utilisation_indices[
                first_pos + 1:
            ]:
                candidate_car = try_merge_positioned_cars(
                    first_car=merged_cars[first_index],
                    second_car=merged_cars[second_index],
                    spec=spec,
                    parallel_gap=parallel_gap,
                    bogie_tol_tons=bogie_tol_tons,
                    use_balance_aware_scoring=(
                        use_balance_aware_scoring
                    ),
                    max_weight_diff_for_pair=(
                        max_weight_diff_for_pair
                    ),
                    max_outer_diameter_diff_for_pair=(
                        max_outer_diameter_diff_for_pair
                    ),
                    force_same_outer_diameter_for_pair=(
                        force_same_outer_diameter_for_pair
                    ),
                    sort_products_by_weight=sort_products_by_weight,
                    allow_parallel_pairs=allow_parallel_pairs,
                )

                if candidate_car is None:
                    continue

                candidate_utilisation = positioned_car_area_utilisation(
                    placed=candidate_car,
                    spec=spec,
                )

                if (
                    best_merge is None
                    or candidate_utilisation > best_merge[3]
                ):
                    best_merge = (
                        first_index,
                        second_index,
                        candidate_car,
                        candidate_utilisation,
                    )

        if best_merge is None:
            continue

        first_index, second_index, candidate_car, _ = best_merge

        for index in sorted(
            [first_index, second_index],
            reverse=True,
        ):
            del merged_cars[index]

        merged_cars.append(candidate_car)
        changed = True

    return merged_cars


def positioned_car_to_scored_wagon(
    placed: List[Positioned],
    spec: FreightCarSpec,
    wagon_id: str,
) -> Dict[str, Any]:
    visual_products = [
        positioned_to_visual_product(
            positioned,
            spec.width,
        )
        for positioned in placed
    ]

    current_weight_kg = positioned_car_weight(placed) * 1000.0
    area_utilisation = positioned_car_area_utilisation(
        placed=placed,
        spec=spec,
    )
    balance_metrics = compute_visual_balance_metrics(
        products=visual_products,
        car_length=spec.length,
        car_width=spec.width,
    )
    balance_metrics.update(
        compute_bogie_metrics_for_car(
            placed=placed,
            spec=spec,
        )
    )

    return {
        "wagon_id": wagon_id,
        "length": spec.length,
        "width": spec.width,
        "max_weight": spec.max_weight * 1000.0,
        "max_weight_tonne": spec.max_weight,
        "current_weight": current_weight_kg,
        "current_weight_tonne": current_weight_kg / 1000.0,
        "area_utilisation": area_utilisation,
        "balance_metrics": balance_metrics,
        "products": visual_products,
    }


def positioned_cars_to_scored_wagons(
    cars: List[List[Positioned]],
    spec: FreightCarSpec,
) -> List[Dict[str, Any]]:
    return [
        positioned_car_to_scored_wagon(
            placed=placed,
            spec=spec,
            wagon_id=f"CANDIDATE_{index:03d}",
        )
        for index, placed in enumerate(cars, start=1)
    ]


def evaluate_positioned_cars(
    cars: List[List[Positioned]],
    spec: FreightCarSpec,
) -> float:
    score = evaluate_loading_solution(
        wagons=positioned_cars_to_scored_wagons(cars, spec),
        weights=ObjectiveWeights(
            wagon_count=1000.0,
            validation_violation=1000000.0,
            unused_area=10.0,
            front_rear_balance=150.0,
            left_right_balance=100.0,
            diagonal_balance=80.0,
            mixed_destination=50.0,
            operational_complexity=5.0,
        ),
    )

    return safe_float(score.get("objective_score"))


def summarize_positioned_cars_stage(
    stage: str,
    destination: str,
    cars: List[List[Positioned]],
    spec: FreightCarSpec,
) -> Dict[str, Any]:
    wagons = positioned_cars_to_scored_wagons(cars, spec)
    score = evaluate_loading_solution(wagons=wagons)

    return {
        "stage": stage,
        "destination": destination,
        "wagon_count": len(cars),
        "product_count": sum(len(car) for car in cars),
        "objective_score": score["objective_score"],
        "average_area_utilisation": score[
            "average_area_utilisation"
        ],
        "is_valid": score["validation_summary"]["is_valid"],
        "violation_count": score["validation_summary"][
            "violation_count"
        ],
        "overlap_violation_count": score[
            "overlap_violation_count"
        ],
        "boundary_violation_count": score[
            "boundary_violation_count"
        ],
        "overweight_violation_count": score[
            "validation_summary"
        ]["overweight_violation_count"],
    }


def repack_products_for_reassignment(
    products: List[Product],
    spec: FreightCarSpec,
    parallel_gap: float,
    bogie_tol_tons: float,
    use_balance_aware_scoring: bool,
    max_weight_diff_for_pair: float,
    max_outer_diameter_diff_for_pair: float,
    force_same_outer_diameter_for_pair: bool,
    sort_products_by_weight: bool,
    allow_parallel_pairs: bool,
) -> Optional[List[List[Positioned]]]:
    if not products:
        return []

    if sum(product.gross_weight for product in products) > spec.max_weight:
        return None

    loading_blocks = generate_loading_blocks(
        products=products,
        spec=spec,
        max_weight_diff_for_pair=max_weight_diff_for_pair,
        max_outer_diameter_diff_for_pair=(
            max_outer_diameter_diff_for_pair
        ),
        force_same_outer_diameter_for_pair=(
            force_same_outer_diameter_for_pair
        ),
        sort_products_by_weight=sort_products_by_weight,
        allow_parallel_pairs=allow_parallel_pairs,
    )

    try:
        candidate_cars = build_initial_feasible_solution(
            loading_blocks=loading_blocks,
            spec=spec,
            parallel_gap=parallel_gap,
            bogie_tol_tons=bogie_tol_tons,
            use_balance_aware_scoring=use_balance_aware_scoring,
        )
    except ValueError:
        return None

    if all(
        is_valid_positioned_car(
            placed=placed,
            spec=spec,
        )
        for placed in candidate_cars
    ):
        return candidate_cars

    return None


def can_move_product_to_car(
    product: Product,
    target_car: List[Positioned],
) -> bool:
    return all(
        positioned.product.destination == product.destination
        for positioned in target_car
    )


def improve_by_wagon_reassignment(
    cars: List[List[Positioned]],
    spec: FreightCarSpec,
    parallel_gap: float,
    bogie_tol_tons: float,
    use_balance_aware_scoring: bool,
    max_weight_diff_for_pair: float,
    max_outer_diameter_diff_for_pair: float,
    force_same_outer_diameter_for_pair: bool,
    sort_products_by_weight: bool,
    allow_parallel_pairs: bool,
    max_passes: int = 2,
) -> List[List[Positioned]]:
    """
    Improve a same-destination group by moving/swapping products between cars.

    The hard business rule is enforced locally: a product may move only into a
    car whose products share its destination, and swaps are only between
    products with the same destination.
    """

    improved_cars = list(cars)

    for _ in range(max(0, max_passes)):
        current_score = evaluate_positioned_cars(
            cars=improved_cars,
            spec=spec,
        )
        best_candidate: Optional[List[List[Positioned]]] = None
        best_score = current_score

        for first_index in range(len(improved_cars)):
            for second_index in range(len(improved_cars)):
                if first_index == second_index:
                    continue

                first_products = [
                    positioned.product
                    for positioned in improved_cars[first_index]
                ]
                second_products = [
                    positioned.product
                    for positioned in improved_cars[second_index]
                ]

                for product_index, product in enumerate(first_products):
                    if not can_move_product_to_car(
                        product=product,
                        target_car=improved_cars[second_index],
                    ):
                        continue

                    candidate_product_lists = [
                        [
                            positioned.product
                            for positioned in car
                        ]
                        for car in improved_cars
                    ]
                    moved_product = candidate_product_lists[
                        first_index
                    ].pop(product_index)
                    candidate_product_lists[second_index].append(
                        moved_product
                    )

                    repacked_first = repack_products_for_reassignment(
                        products=candidate_product_lists[first_index],
                        spec=spec,
                        parallel_gap=parallel_gap,
                        bogie_tol_tons=bogie_tol_tons,
                        use_balance_aware_scoring=(
                            use_balance_aware_scoring
                        ),
                        max_weight_diff_for_pair=(
                            max_weight_diff_for_pair
                        ),
                        max_outer_diameter_diff_for_pair=(
                            max_outer_diameter_diff_for_pair
                        ),
                        force_same_outer_diameter_for_pair=(
                            force_same_outer_diameter_for_pair
                        ),
                        sort_products_by_weight=sort_products_by_weight,
                        allow_parallel_pairs=allow_parallel_pairs,
                    )
                    repacked_second = repack_products_for_reassignment(
                        products=candidate_product_lists[second_index],
                        spec=spec,
                        parallel_gap=parallel_gap,
                        bogie_tol_tons=bogie_tol_tons,
                        use_balance_aware_scoring=(
                            use_balance_aware_scoring
                        ),
                        max_weight_diff_for_pair=(
                            max_weight_diff_for_pair
                        ),
                        max_outer_diameter_diff_for_pair=(
                            max_outer_diameter_diff_for_pair
                        ),
                        force_same_outer_diameter_for_pair=(
                            force_same_outer_diameter_for_pair
                        ),
                        sort_products_by_weight=sort_products_by_weight,
                        allow_parallel_pairs=allow_parallel_pairs,
                    )

                    if (
                        repacked_first is None
                        or repacked_second is None
                    ):
                        continue

                    candidate_cars = [
                        car
                        for index, car in enumerate(improved_cars)
                        if index not in {first_index, second_index}
                    ]
                    candidate_cars.extend(repacked_first)
                    candidate_cars.extend(repacked_second)

                    candidate_score = evaluate_positioned_cars(
                        cars=candidate_cars,
                        spec=spec,
                    )

                    if candidate_score < best_score - 1e-9:
                        best_candidate = candidate_cars
                        best_score = candidate_score

        for first_index in range(len(improved_cars)):
            for second_index in range(first_index + 1, len(improved_cars)):
                first_products = [
                    positioned.product
                    for positioned in improved_cars[first_index]
                ]
                second_products = [
                    positioned.product
                    for positioned in improved_cars[second_index]
                ]

                for first_product_index, first_product in enumerate(
                    first_products
                ):
                    for second_product_index, second_product in enumerate(
                        second_products
                    ):
                        if (
                            first_product.destination
                            != second_product.destination
                        ):
                            continue

                        candidate_product_lists = [
                            [
                                positioned.product
                                for positioned in car
                            ]
                            for car in improved_cars
                        ]
                        (
                            candidate_product_lists[first_index][
                                first_product_index
                            ],
                            candidate_product_lists[second_index][
                                second_product_index
                            ],
                        ) = (
                            candidate_product_lists[second_index][
                                second_product_index
                            ],
                            candidate_product_lists[first_index][
                                first_product_index
                            ],
                        )

                        repacked_first = repack_products_for_reassignment(
                            products=candidate_product_lists[
                                first_index
                            ],
                            spec=spec,
                            parallel_gap=parallel_gap,
                            bogie_tol_tons=bogie_tol_tons,
                            use_balance_aware_scoring=(
                                use_balance_aware_scoring
                            ),
                            max_weight_diff_for_pair=(
                                max_weight_diff_for_pair
                            ),
                            max_outer_diameter_diff_for_pair=(
                                max_outer_diameter_diff_for_pair
                            ),
                            force_same_outer_diameter_for_pair=(
                                force_same_outer_diameter_for_pair
                            ),
                            sort_products_by_weight=(
                                sort_products_by_weight
                            ),
                            allow_parallel_pairs=allow_parallel_pairs,
                        )
                        repacked_second = repack_products_for_reassignment(
                            products=candidate_product_lists[
                                second_index
                            ],
                            spec=spec,
                            parallel_gap=parallel_gap,
                            bogie_tol_tons=bogie_tol_tons,
                            use_balance_aware_scoring=(
                                use_balance_aware_scoring
                            ),
                            max_weight_diff_for_pair=(
                                max_weight_diff_for_pair
                            ),
                            max_outer_diameter_diff_for_pair=(
                                max_outer_diameter_diff_for_pair
                            ),
                            force_same_outer_diameter_for_pair=(
                                force_same_outer_diameter_for_pair
                            ),
                            sort_products_by_weight=(
                                sort_products_by_weight
                            ),
                            allow_parallel_pairs=allow_parallel_pairs,
                        )

                        if (
                            repacked_first is None
                            or repacked_second is None
                        ):
                            continue

                        candidate_cars = [
                            car
                            for index, car in enumerate(improved_cars)
                            if index
                            not in {first_index, second_index}
                        ]
                        candidate_cars.extend(repacked_first)
                        candidate_cars.extend(repacked_second)

                        candidate_score = evaluate_positioned_cars(
                            cars=candidate_cars,
                            spec=spec,
                        )

                        if candidate_score < best_score - 1e-9:
                            best_candidate = candidate_cars
                            best_score = candidate_score

        if best_candidate is None:
            break

        improved_cars = best_candidate

    return improved_cars


def pack_destination_into_cars(
    products: List[Product],
    spec: FreightCarSpec,
    parallel_gap: float,
    bogie_tol_tons: float,
    max_weight_diff_for_pair: float,
    force_same_outer_diameter_for_pair: bool,
    sort_products_by_weight: bool,
    allow_parallel_pairs: bool,
    use_balance_aware_scoring: bool,
    max_outer_diameter_diff_for_pair: float = 100.0,
    enable_low_utilisation_merge: bool = True,
    low_utilisation_threshold: float = 0.35,
    enable_wagon_reassignment: bool = True,
    max_wagon_reassignment_passes: int = 2,
    trace_destination: str = "UNKNOWN_DEST",
    return_trace: bool = False,
) -> Any:
    """
    Pack a product list into freight cars.

    This is now a thin orchestration wrapper:
    1. dynamically identify loading blocks;
    2. build the initial feasible solution from those blocks.
    """

    loading_blocks = generate_loading_blocks(
        products=products,
        spec=spec,
        max_weight_diff_for_pair=max_weight_diff_for_pair,
        max_outer_diameter_diff_for_pair=(
            max_outer_diameter_diff_for_pair
        ),
        force_same_outer_diameter_for_pair=(
            force_same_outer_diameter_for_pair
        ),
        sort_products_by_weight=sort_products_by_weight,
        allow_parallel_pairs=allow_parallel_pairs,
    )

    cars = build_initial_feasible_solution(
        loading_blocks=loading_blocks,
        spec=spec,
        parallel_gap=parallel_gap,
        bogie_tol_tons=bogie_tol_tons,
        use_balance_aware_scoring=use_balance_aware_scoring,
    )

    trace: List[Dict[str, Any]] = [
        summarize_positioned_cars_stage(
            stage="initial_feasible_solution",
            destination=trace_destination,
            cars=cars,
            spec=spec,
        )
    ]

    if enable_low_utilisation_merge:
        cars = merge_low_utilisation_cars(
            cars=cars,
            spec=spec,
            parallel_gap=parallel_gap,
            bogie_tol_tons=bogie_tol_tons,
            use_balance_aware_scoring=use_balance_aware_scoring,
            max_weight_diff_for_pair=max_weight_diff_for_pair,
            max_outer_diameter_diff_for_pair=(
                max_outer_diameter_diff_for_pair
            ),
            force_same_outer_diameter_for_pair=(
                force_same_outer_diameter_for_pair
            ),
            sort_products_by_weight=sort_products_by_weight,
            allow_parallel_pairs=allow_parallel_pairs,
            utilisation_threshold=low_utilisation_threshold,
        )
        trace.append(
            summarize_positioned_cars_stage(
                stage="low_utilisation_merge",
                destination=trace_destination,
                cars=cars,
                spec=spec,
            )
        )

    if enable_wagon_reassignment:
        cars = improve_by_wagon_reassignment(
            cars=cars,
            spec=spec,
            parallel_gap=parallel_gap,
            bogie_tol_tons=bogie_tol_tons,
            use_balance_aware_scoring=use_balance_aware_scoring,
            max_weight_diff_for_pair=max_weight_diff_for_pair,
            max_outer_diameter_diff_for_pair=(
                max_outer_diameter_diff_for_pair
            ),
            force_same_outer_diameter_for_pair=(
                force_same_outer_diameter_for_pair
            ),
            sort_products_by_weight=sort_products_by_weight,
            allow_parallel_pairs=allow_parallel_pairs,
            max_passes=max_wagon_reassignment_passes,
        )
        trace.append(
            summarize_positioned_cars_stage(
                stage="wagon_reassignment",
                destination=trace_destination,
                cars=cars,
                spec=spec,
            )
        )

    if return_trace:
        return cars, trace

    return cars


# =============================================================================
# Conversion for VisualizationAgent
# =============================================================================


def positioned_to_visual_product(
    positioned: Positioned,
    car_width: float,
) -> Dict[str, Any]:
    product = positioned.product

    placed_length = (
        product.outer_diameter
        if product.outer_diameter > 0
        else 1000.0
    )

    placed_width = (
        product.width
        if product.width > 0
        else 1000.0
    )

    x = positioned.x_mm - placed_length / 2.0

    if positioned.lane is None:
        y = car_width / 2.0 - placed_width / 2.0
    elif positioned.lane == 0:
        y = car_width / 2.0 - placed_width
    elif positioned.lane == 1:
        y = car_width / 2.0
    else:
        y = car_width / 2.0 - placed_width / 2.0

    y = max(0.0, min(y, car_width - placed_width))

    return {
        "product_id": product.material_no,
        "material_no": product.material_no,
        "destination": product.destination,
        "x": x,
        "y": y,
        "length": placed_length,
        "width": placed_width,
        "placed_length": placed_length,
        "placed_width": placed_width,
        "weight": product.gross_weight * 1000.0,
        "gross_weight_tonne": product.gross_weight,
        "net_weight_tonne": product.net_weight,
        "thickness": product.thickness,
        "outer_diameter": product.outer_diameter,
        "rotated": False,
        "lane": positioned.lane,
        "x_mm_center": positioned.x_mm,
    }


def rectangle_overlap_area(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rx1: float,
    ry1: float,
    rx2: float,
    ry2: float,
) -> float:
    overlap_x = max(0.0, min(x2, rx2) - max(x1, rx1))
    overlap_y = max(0.0, min(y2, ry2) - max(y1, ry1))

    return overlap_x * overlap_y


def product_rectangle(
    product: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    x = safe_float(product.get("x"))
    y = safe_float(product.get("y"))
    placed_length = safe_float(
        first_present(product, ["placed_length", "length"])
    )
    placed_width = safe_float(
        first_present(product, ["placed_width", "width"])
    )

    return (
        x,
        y,
        x + placed_length,
        y + placed_width,
    )


def visual_product_id(product: Dict[str, Any]) -> str:
    raw_id = first_present(
        product,
        ["material_no", "product_id", "id"],
    )

    return str(raw_id).strip() if raw_id is not None else ""


def validate_wagon_layout(
    wagon: Dict[str, Any],
    tolerance_mm: float = 1e-6,
) -> Dict[str, Any]:
    """
    Validate hard layout constraints for one wagon.

    This validator is intentionally independent from the packing algorithm so
    future heuristics, merge steps, or simulated annealing moves can all be
    checked through the same final safety gate.
    """

    products = wagon.get("products", [])

    if not isinstance(products, list):
        products = []

    car_length = safe_float(wagon.get("length"))
    car_width = safe_float(wagon.get("width"))
    max_weight = safe_float(wagon.get("max_weight"))

    current_weight = safe_float(wagon.get("current_weight"))

    if current_weight <= 0:
        current_weight = sum(
            safe_float(product.get("weight"))
            for product in products
            if isinstance(product, dict)
        )

    violations: List[Dict[str, Any]] = []

    boundary_count = 0
    overlap_count = 0
    invalid_dimension_count = 0

    if max_weight > 0 and current_weight > max_weight + tolerance_mm:
        violations.append(
            {
                "type": "overweight",
                "wagon_id": wagon.get("wagon_id"),
                "current_weight": current_weight,
                "max_weight": max_weight,
                "excess_weight": current_weight - max_weight,
            }
        )

    for product in products:
        if not isinstance(product, dict):
            continue

        product_id = visual_product_id(product)
        x = safe_float(product.get("x"))
        y = safe_float(product.get("y"))
        placed_length = safe_float(
            first_present(product, ["placed_length", "length"])
        )
        placed_width = safe_float(
            first_present(product, ["placed_width", "width"])
        )

        if placed_length <= 0 or placed_width <= 0:
            invalid_dimension_count += 1
            violations.append(
                {
                    "type": "invalid_dimension",
                    "wagon_id": wagon.get("wagon_id"),
                    "product_id": product_id,
                    "placed_length": placed_length,
                    "placed_width": placed_width,
                }
            )
            continue

        exceeds_boundary = (
            x < -tolerance_mm
            or y < -tolerance_mm
            or x + placed_length > car_length + tolerance_mm
            or y + placed_width > car_width + tolerance_mm
        )

        if exceeds_boundary:
            boundary_count += 1
            violations.append(
                {
                    "type": "boundary",
                    "wagon_id": wagon.get("wagon_id"),
                    "product_id": product_id,
                    "x": x,
                    "y": y,
                    "placed_length": placed_length,
                    "placed_width": placed_width,
                    "car_length": car_length,
                    "car_width": car_width,
                }
            )

    for first_index in range(len(products)):
        first_product = products[first_index]

        if not isinstance(first_product, dict):
            continue

        first_length = safe_float(
            first_present(first_product, ["placed_length", "length"])
        )
        first_width = safe_float(
            first_present(first_product, ["placed_width", "width"])
        )

        if first_length <= 0 or first_width <= 0:
            continue

        x1, y1, x2, y2 = product_rectangle(first_product)

        for second_index in range(first_index + 1, len(products)):
            second_product = products[second_index]

            if not isinstance(second_product, dict):
                continue

            second_length = safe_float(
                first_present(
                    second_product,
                    ["placed_length", "length"],
                )
            )
            second_width = safe_float(
                first_present(
                    second_product,
                    ["placed_width", "width"],
                )
            )

            if second_length <= 0 or second_width <= 0:
                continue

            rx1, ry1, rx2, ry2 = product_rectangle(second_product)

            overlap_area = rectangle_overlap_area(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                rx1=rx1,
                ry1=ry1,
                rx2=rx2,
                ry2=ry2,
            )

            if overlap_area > tolerance_mm:
                overlap_count += 1
                violations.append(
                    {
                        "type": "overlap",
                        "wagon_id": wagon.get("wagon_id"),
                        "product_id_1": visual_product_id(
                            first_product
                        ),
                        "product_id_2": visual_product_id(
                            second_product
                        ),
                        "overlap_area": overlap_area,
                    }
                )

    overweight_count = sum(
        1 for violation in violations if violation["type"] == "overweight"
    )

    return {
        "wagon_id": wagon.get("wagon_id"),
        "is_valid": not violations,
        "violation_count": len(violations),
        "overweight_violation_count": overweight_count,
        "boundary_violation_count": boundary_count,
        "overlap_violation_count": overlap_count,
        "invalid_dimension_count": invalid_dimension_count,
        "violations": violations,
    }


def validate_loading_solution(
    wagons: List[Dict[str, Any]],
    expected_material_numbers: Optional[List[str]] = None,
    tolerance_mm: float = 1e-6,
) -> Dict[str, Any]:
    wagon_reports = [
        validate_wagon_layout(
            wagon=wagon,
            tolerance_mm=tolerance_mm,
        )
        for wagon in wagons
    ]

    actual_ids: List[str] = []

    for wagon in wagons:
        products = wagon.get("products", [])

        if not isinstance(products, list):
            continue

        for product in products:
            if isinstance(product, dict):
                product_id = visual_product_id(product)

                if product_id:
                    actual_ids.append(product_id)

    duplicate_ids = sorted(
        {
            product_id
            for product_id in actual_ids
            if actual_ids.count(product_id) > 1
        }
    )

    missing_ids: List[str] = []
    unexpected_ids: List[str] = []

    if expected_material_numbers is not None:
        expected_set = set(expected_material_numbers)
        actual_set = set(actual_ids)
        missing_ids = sorted(expected_set - actual_set)
        unexpected_ids = sorted(actual_set - expected_set)

    coverage_violation_count = (
        len(duplicate_ids)
        + len(missing_ids)
        + len(unexpected_ids)
    )

    boundary_count = sum(
        report["boundary_violation_count"] for report in wagon_reports
    )
    overlap_count = sum(
        report["overlap_violation_count"] for report in wagon_reports
    )
    overweight_count = sum(
        report["overweight_violation_count"] for report in wagon_reports
    )
    invalid_dimension_count = sum(
        report["invalid_dimension_count"] for report in wagon_reports
    )

    violation_count = (
        sum(report["violation_count"] for report in wagon_reports)
        + coverage_violation_count
    )

    return {
        "is_valid": violation_count == 0,
        "violation_count": violation_count,
        "overweight_violation_count": overweight_count,
        "boundary_violation_count": boundary_count,
        "overlap_violation_count": overlap_count,
        "invalid_dimension_count": invalid_dimension_count,
        "coverage_violation_count": coverage_violation_count,
        "duplicate_product_ids": duplicate_ids,
        "missing_product_ids": missing_ids,
        "unexpected_product_ids": unexpected_ids,
        "wagon_reports": wagon_reports,
    }


def evaluate_loading_solution(
    wagons: List[Dict[str, Any]],
    weights: Optional[ObjectiveWeights] = None,
    expected_material_numbers: Optional[List[str]] = None,
    enforce_hard_constraints: bool = False,
) -> Dict[str, Any]:
    weights = weights or ObjectiveWeights()

    validation_summary = validate_loading_solution(
        wagons=wagons,
        expected_material_numbers=expected_material_numbers,
    )

    if enforce_hard_constraints and not validation_summary["is_valid"]:
        raise ValueError(
            "Final loading solution violates hard constraints: "
            f"{validation_summary}"
        )

    wagon_count = len(wagons)

    average_area_utilisation = (
        sum(
            safe_float(wagon.get("area_utilisation"))
            for wagon in wagons
        )
        / wagon_count
        if wagon_count
        else 0.0
    )

    front_rear_penalty = sum(
        safe_float(
            wagon.get("balance_metrics", {}).get(
                "front_rear_penalty"
            )
        )
        for wagon in wagons
    )
    left_right_penalty = sum(
        safe_float(
            wagon.get("balance_metrics", {}).get(
                "left_right_penalty"
            )
        )
        for wagon in wagons
    )
    diagonal_penalty = sum(
        safe_float(
            wagon.get("balance_metrics", {}).get(
                "diagonal_penalty"
            )
        )
        for wagon in wagons
    )

    mixed_destination_count = 0

    for wagon in wagons:
        destinations = {
            str(product.get("destination", "")).strip()
            for product in wagon.get("products", [])
            if isinstance(product, dict)
        }
        destinations.discard("")

        if len(destinations) > 1:
            mixed_destination_count += 1

    total_products = sum(
        len(wagon.get("products", []))
        for wagon in wagons
    )
    average_products_per_wagon = (
        total_products / wagon_count
        if wagon_count
        else 0.0
    )

    components = {
        "wagon_count": wagon_count * weights.wagon_count,
        "validation": (
            validation_summary["violation_count"]
            * weights.validation_violation
        ),
        "unused_area": (
            (1.0 - average_area_utilisation)
            * weights.unused_area
        ),
        "front_rear_balance": (
            front_rear_penalty * weights.front_rear_balance
        ),
        "left_right_balance": (
            left_right_penalty * weights.left_right_balance
        ),
        "diagonal_balance": (
            diagonal_penalty * weights.diagonal_balance
        ),
        "mixed_destination": (
            mixed_destination_count * weights.mixed_destination
        ),
        "operational_complexity": (
            average_products_per_wagon
            * weights.operational_complexity
        ),
    }

    objective_score = sum(components.values())

    return {
        "objective_score": objective_score,
        "components": components,
        "weights": weights.to_dict(),
        "wagon_count": wagon_count,
        "average_area_utilisation": average_area_utilisation,
        "front_rear_balance_penalty": front_rear_penalty,
        "left_right_balance_penalty": left_right_penalty,
        "diagonal_balance_penalty": diagonal_penalty,
        "mixed_destination_count": mixed_destination_count,
        "average_products_per_wagon": average_products_per_wagon,
        "overweight_penalty": (
            validation_summary["overweight_violation_count"]
            * weights.validation_violation
        ),
        "boundary_violation_count": validation_summary[
            "boundary_violation_count"
        ],
        "overlap_violation_count": validation_summary[
            "overlap_violation_count"
        ],
        "coverage_violation_count": validation_summary[
            "coverage_violation_count"
        ],
        "invalid_dimension_count": validation_summary[
            "invalid_dimension_count"
        ],
        "validation_summary": validation_summary,
    }


def distribute_product_weight_to_region(
    product: Dict[str, Any],
    region: Tuple[float, float, float, float],
) -> float:
    x = safe_float(product.get("x"))
    y = safe_float(product.get("y"))
    placed_length = safe_float(product.get("placed_length"))
    placed_width = safe_float(product.get("placed_width"))
    weight = safe_float(product.get("weight"))

    if placed_length <= 0 or placed_width <= 0 or weight <= 0:
        return 0.0

    x1 = x
    y1 = y
    x2 = x + placed_length
    y2 = y + placed_width

    rx1, ry1, rx2, ry2 = region

    overlap_area = rectangle_overlap_area(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        rx1=rx1,
        ry1=ry1,
        rx2=rx2,
        ry2=ry2,
    )

    product_area = placed_length * placed_width

    return weight * (overlap_area / product_area)


def compute_visual_balance_metrics(
    products: List[Dict[str, Any]],
    car_length: float,
    car_width: float,
) -> Dict[str, float]:
    center_x = car_length / 2.0
    center_y = car_width / 2.0

    left_region = (0.0, 0.0, car_length, center_y)
    right_region = (0.0, center_y, car_length, car_width)

    front_region = (0.0, 0.0, center_x, car_width)
    rear_region = (center_x, 0.0, car_length, car_width)

    front_left = (0.0, 0.0, center_x, center_y)
    front_right = (0.0, center_y, center_x, car_width)
    rear_left = (center_x, 0.0, car_length, center_y)
    rear_right = (center_x, center_y, car_length, car_width)

    left_weight = sum(
        distribute_product_weight_to_region(product, left_region)
        for product in products
    )

    right_weight = sum(
        distribute_product_weight_to_region(product, right_region)
        for product in products
    )

    front_weight = sum(
        distribute_product_weight_to_region(product, front_region)
        for product in products
    )

    rear_weight = sum(
        distribute_product_weight_to_region(product, rear_region)
        for product in products
    )

    diagonal_one_weight = sum(
        distribute_product_weight_to_region(product, front_left)
        for product in products
    ) + sum(
        distribute_product_weight_to_region(product, rear_right)
        for product in products
    )

    diagonal_two_weight = sum(
        distribute_product_weight_to_region(product, front_right)
        for product in products
    ) + sum(
        distribute_product_weight_to_region(product, rear_left)
        for product in products
    )

    def penalty(first_value: float, second_value: float) -> float:
        total = first_value + second_value

        if total <= 0:
            return 0.0

        return abs(first_value - second_value) / total

    return {
        "left_weight": left_weight,
        "right_weight": right_weight,
        "front_weight": front_weight,
        "rear_weight": rear_weight,
        "diagonal_1_weight": diagonal_one_weight,
        "diagonal_2_weight": diagonal_two_weight,
        "left_right_penalty": penalty(left_weight, right_weight),
        "front_rear_penalty": penalty(front_weight, rear_weight),
        "diagonal_penalty": penalty(
            diagonal_one_weight,
            diagonal_two_weight,
        ),
    }


def compute_bogie_metrics_for_car(
    placed: List[Positioned],
    spec: FreightCarSpec,
) -> Dict[str, float]:
    front_bogie, rear_bogie = compute_bogie_balance(
        placed,
        spec.length,
    )

    total = front_bogie + rear_bogie
    difference = abs(front_bogie - rear_bogie)

    return {
        "front_bogie_tonne": front_bogie,
        "rear_bogie_tonne": rear_bogie,
        "bogie_diff_tonne": difference,
        "bogie_diff_ratio": (
            difference / total
            if total > 0
            else 0.0
        ),
    }


# =============================================================================
# Optimisation Agent
# =============================================================================


class OptimisationAgent:
    def __init__(
        self,
        products_json_path: str | Path = "Agents/products.json",
        strategy_plan_path: str | Path = "Agents/strategy_plan.json",
        output_path: str | Path = "Agents/loading_solution.json",
        car_length: float = 13000.0,
        car_width: float = 3000.0,
        max_weight_tonne: float = 60.0,
        parallel_gap: float = 50.0,
        bogie_tol_tons: float = 2.0,
        max_weight_diff_for_pair: float = 1.0,
        max_outer_diameter_diff_for_pair: float = 100.0,
        force_same_outer_diameter_for_pair: bool = True,
        enable_low_utilisation_merge: bool = True,
        low_utilisation_threshold: float = 0.35,
        enable_wagon_reassignment: bool = True,
        max_wagon_reassignment_passes: int = 2,
        seed: int = 42,
    ) -> None:
        self.products_json_path = Path(products_json_path)
        self.strategy_plan_path = Path(strategy_plan_path)
        self.output_path = Path(output_path)
        self.disabled_wagon_ids: set[str] = set()

        self.spec = FreightCarSpec(
            length=car_length,
            width=car_width,
            max_weight=max_weight_tonne,
        )

        self.parallel_gap = parallel_gap
        self.bogie_tol_tons = bogie_tol_tons
        self.max_weight_diff_for_pair = max_weight_diff_for_pair
        self.max_outer_diameter_diff_for_pair = (
            max_outer_diameter_diff_for_pair
        )
        self.force_same_outer_diameter_for_pair = (
            force_same_outer_diameter_for_pair
        )
        self.enable_low_utilisation_merge = (
            enable_low_utilisation_merge
        )
        self.low_utilisation_threshold = low_utilisation_threshold
        self.enable_wagon_reassignment = enable_wagon_reassignment
        self.max_wagon_reassignment_passes = (
            max_wagon_reassignment_passes
        )

        self.seed = seed
        self.random = random.Random(seed)

    @staticmethod
    def _safe_log(message: str) -> None:
        try:
            print(message)
        except (OSError, UnicodeEncodeError):
            pass

    # -------------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        started_at = time.perf_counter()

        products = self.load_products()
        strategy_plan = self.load_strategy_plan()
        raw_disabled = strategy_plan.get("optimisation_config", {}).get(
            "disabled_wagon_ids", []
        )
        self.disabled_wagon_ids = {
            str(item) for item in raw_disabled if str(item).strip()
        } if isinstance(raw_disabled, list) else set()

        products, dropped_duplicates = dedupe_by_material_no(products)

        validate_products_fit_spec(
            products=products,
            spec=self.spec,
        )

        execution_plan = resolve_strategy_execution_plan(
            strategy_plan=strategy_plan,
            products=products,
        )
        algorithm_policy = self.resolve_algorithm_policy(
            strategy_plan=strategy_plan,
            products=products,
            execution_plan=execution_plan,
        )
        self.apply_algorithm_policy(
            execution_plan=execution_plan,
            algorithm_policy=algorithm_policy,
        )

        self._safe_log(f"[OptimisationAgent] Loaded products: {len(products)}")
        self._safe_log(
            "[OptimisationAgent] Destinations: "
            f"{sorted({product.destination for product in products})}"
        )
        self._safe_log(
            "[OptimisationAgent] Strategy | "
            f"requested={execution_plan.requested_strategy} | "
            f"applied={execution_plan.applied_strategy} | "
            f"scenario={execution_plan.actual_batch_scenario}"
        )
        self._safe_log(
            "[OptimisationAgent] Algorithm policy | "
            f"mode={algorithm_policy.mode} | "
            f"budget={algorithm_policy.time_budget_seconds} | "
            f"pipeline={algorithm_policy.selected_pipeline}"
        )

        if execution_plan.resolution_notes:
            for note in execution_plan.resolution_notes:
                self._safe_log(f"[OptimisationAgent] Strategy note: {note}")

        if algorithm_policy.decision_notes:
            for note in algorithm_policy.decision_notes:
                self._safe_log(f"[OptimisationAgent] Policy note: {note}")

        if dropped_duplicates:
            self._safe_log(
                "[OptimisationAgent] Dropped duplicate material_no count: "
                f"{len(dropped_duplicates)}"
            )

        grouped_products = self.prepare_product_groups(
            products=products,
            execution_plan=execution_plan,
        )

        packed_groups, optimisation_trace = self.pack_all_groups(
            grouped_products=grouped_products,
            execution_plan=execution_plan,
            algorithm_policy=algorithm_policy,
            collect_trace=True,
        )
        baseline_objective = self.evaluate_packed_groups(packed_groups)
        optimisation_trace.append(
            self.build_anytime_trace_entry(
                stage="anytime_baseline",
                packed_groups=packed_groups,
                expected_material_numbers=[
                    product.material_no for product in products
                ],
                search_objective_score=baseline_objective,
                elapsed_seconds=time.perf_counter() - started_at,
                status="accepted",
                reason=(
                    "Initial feasible solution accepted as the "
                    "anytime baseline."
                ),
            )
        )

        deadline_monotonic = (
            started_at + algorithm_policy.time_budget_seconds
            if algorithm_policy.time_budget_seconds is not None
            else None
        )
        deadline_available_for_sa = (
            deadline_monotonic is None
            or time.perf_counter() < deadline_monotonic
        )

        if execution_plan.use_simulated_annealing and deadline_available_for_sa:
            candidate_packed_groups, sa_log = self.run_simulated_annealing(
                grouped_products=grouped_products,
                execution_plan=execution_plan,
                initial_packed_groups=packed_groups,
                deadline_monotonic=deadline_monotonic,
            )
            candidate_objective = self.evaluate_packed_groups(
                candidate_packed_groups
            )
            candidate_wagons = self.convert_packed_groups_to_wagons(
                candidate_packed_groups
            )
            candidate_validation = validate_loading_solution(
                wagons=candidate_wagons,
                expected_material_numbers=[
                    product.material_no for product in products
                ],
            )
            candidate_is_better = (
                candidate_objective <= baseline_objective
            )
            candidate_is_valid = candidate_validation["is_valid"]

            if candidate_is_valid and candidate_is_better:
                packed_groups = candidate_packed_groups
                accepted_status = "accepted"
                accepted_reason = (
                    "Simulated annealing candidate satisfied all hard "
                    "constraints and improved or matched the baseline "
                    "objective."
                )
            else:
                accepted_status = "rejected"
                accepted_reason = (
                    "Simulated annealing candidate was rejected because "
                    "it did not improve the baseline objective or failed "
                    "hard-constraint validation."
                )

            optimisation_trace.append(
                self.build_anytime_trace_entry(
                    stage="anytime_simulated_annealing",
                    packed_groups=candidate_packed_groups,
                    expected_material_numbers=[
                    product.material_no for product in products
                ],
                    search_objective_score=candidate_objective,
                    elapsed_seconds=time.perf_counter() - started_at,
                    status=accepted_status,
                    reason=accepted_reason,
                    candidate_validation=candidate_validation,
                    baseline_search_objective=baseline_objective,
                    search_improvement=(
                        baseline_objective - candidate_objective
                    ),
                )
            )
            sa_log["candidate_status"] = accepted_status
            sa_log["candidate_reason"] = accepted_reason
        else:
            sa_log = {
                "enabled": execution_plan.use_simulated_annealing,
                "executed": False,
                "iterations_requested": 0,
                "iterations_completed": 0,
                "accepted_moves": 0,
                "initial_objective": baseline_objective,
                "final_objective": baseline_objective,
                "termination_reason": (
                    "deadline_exhausted_before_start"
                    if execution_plan.use_simulated_annealing
                    and not deadline_available_for_sa
                    else "not_selected"
                ),
                "note": (
                    "Simulated annealing was skipped because the safe "
                    "baseline consumed the available deadline."
                    if execution_plan.use_simulated_annealing
                    and not deadline_available_for_sa
                    else "Simulated annealing was not selected by the "
                    "algorithm policy."
                ),
            }

        wagons = self.convert_packed_groups_to_wagons(
            packed_groups=packed_groups,
        )

        runtime_seconds = time.perf_counter() - started_at

        solution = self.build_solution_json(
            strategy_plan=strategy_plan,
            execution_plan=execution_plan,
            wagons=wagons,
            products=products,
            dropped_duplicates=dropped_duplicates,
            runtime_seconds=runtime_seconds,
            simulated_annealing_log=sa_log,
            optimisation_trace=optimisation_trace,
            algorithm_policy=algorithm_policy,
        )

        self.export_json(solution, self.output_path)

        self._safe_log(f"[OptimisationAgent] Generated wagons: {len(wagons)}")
        self._safe_log(f"[OptimisationAgent] Output saved to: {self.output_path}")

        return solution

    # -------------------------------------------------------------------------
    # Strategy-driven execution
    # -------------------------------------------------------------------------

    def resolve_algorithm_policy(
        self,
        strategy_plan: Dict[str, Any],
        products: List[Product],
        execution_plan: StrategyExecutionPlan,
    ) -> AlgorithmPolicy:
        raw_config = strategy_plan.get("optimisation_config", {})

        if not isinstance(raw_config, dict):
            raw_config = {}

        product_count = len(products)
        destination_count = len(
            {product.destination for product in products}
        )
        time_budget_seconds = None

        if "time_budget_seconds" in raw_config:
            time_budget_seconds = max(
                0.0,
                safe_float(raw_config.get("time_budget_seconds")),
            )
        elif "time_budget_seconds" in strategy_plan:
            time_budget_seconds = max(
                0.0,
                safe_float(strategy_plan.get("time_budget_seconds")),
            )

        mode = str(
            raw_config.get(
                "algorithm_policy_mode",
                raw_config.get("optimisation_mode", "time_budgeted"),
            )
        ).strip() or "time_budgeted"

        if product_count <= 12:
            complexity_class = "small"
        elif product_count <= 40:
            complexity_class = "medium"
        else:
            complexity_class = "large"

        selected_pipeline = ["initial_feasible_solution"]
        skipped_steps: List[Dict[str, str]] = []
        decision_notes: List[str] = []

        enable_merge = self.enable_low_utilisation_merge

        if enable_merge:
            selected_pipeline.append("low_utilisation_merge")
        else:
            skipped_steps.append(
                {
                    "step": "low_utilisation_merge",
                    "reason": "Disabled by agent configuration.",
                }
            )

        enable_reassignment = self.enable_wagon_reassignment

        if product_count < 4 and enable_reassignment:
            enable_reassignment = False
            skipped_steps.append(
                {
                    "step": "same_destination_wagon_reassignment",
                    "reason": (
                        "Batch is too small for cross-wagon migration to "
                        "provide meaningful improvement."
                    ),
                }
            )

        if (
            time_budget_seconds is not None
            and time_budget_seconds < 3.0
            and enable_reassignment
        ):
            enable_reassignment = False
            skipped_steps.append(
                {
                    "step": "same_destination_wagon_reassignment",
                    "reason": (
                        "Skipped to keep optimisation within a tight "
                        "time budget."
                    ),
                }
            )

        if enable_reassignment:
            selected_pipeline.append("same_destination_wagon_reassignment")
        elif not self.enable_wagon_reassignment:
            skipped_steps.append(
                {
                    "step": "same_destination_wagon_reassignment",
                    "reason": "Disabled by agent configuration.",
                }
            )

        requested_sa = execution_plan.use_simulated_annealing
        use_sa = requested_sa
        sa_iterations = execution_plan.sa_iterations

        if product_count <= 12:
            if use_sa:
                decision_notes.append(
                    "Simulated annealing was requested, but the batch is "
                    "small enough for deterministic heuristics."
                )
            use_sa = False
            sa_iterations = 0
            skipped_steps.append(
                {
                    "step": "simulated_annealing",
                    "reason": (
                        "Small batch; heuristic placement is preferred "
                        "for faster and more predictable execution."
                    ),
                }
            )
        elif (
            time_budget_seconds is not None
            and time_budget_seconds < 8.0
        ):
            if use_sa:
                decision_notes.append(
                    "Simulated annealing was requested, but the available "
                    "time budget is below the policy threshold."
                )
            use_sa = False
            sa_iterations = 0
            skipped_steps.append(
                {
                    "step": "simulated_annealing",
                    "reason": (
                        "Estimated runtime exceeds the available "
                        "time budget."
                    ),
                }
            )
        else:
            allow_policy_strategy_override = safe_bool(
                raw_config.get("allow_policy_strategy_override"),
                default=True,
            )
            if (
                not use_sa
                and product_count >= 25
                and allow_policy_strategy_override
            ):
                use_sa = True
                decision_notes.append(
                    "Simulated annealing was enabled by policy because the "
                    "batch is medium/large and no tight time budget was set."
                )

            if use_sa:
                if time_budget_seconds is None:
                    sa_iterations = max(sa_iterations, 20)
                elif time_budget_seconds < 15.0:
                    sa_iterations = min(max(sa_iterations, 10), 20)
                else:
                    sa_iterations = min(max(sa_iterations, 30), 80)

                selected_pipeline.append("simulated_annealing")

        if not use_sa and not any(
            item.get("step") == "simulated_annealing"
            for item in skipped_steps
        ):
            skipped_steps.append(
                {
                    "step": "simulated_annealing",
                    "reason": (
                        "Not selected by LLM strategy or local policy."
                    ),
                }
            )

        if time_budget_seconds is not None:
            decision_notes.append(
                f"Algorithm policy used a {time_budget_seconds:.2f}s "
                "time budget."
            )

        return AlgorithmPolicy(
            mode=mode,
            time_budget_seconds=time_budget_seconds,
            product_count=product_count,
            destination_count=destination_count,
            complexity_class=complexity_class,
            selected_pipeline=selected_pipeline,
            skipped_steps=skipped_steps,
            decision_notes=decision_notes,
            enable_low_utilisation_merge=enable_merge,
            enable_wagon_reassignment=enable_reassignment,
            use_simulated_annealing=use_sa,
            sa_iterations=sa_iterations,
        )

    def apply_algorithm_policy(
        self,
        execution_plan: StrategyExecutionPlan,
        algorithm_policy: AlgorithmPolicy,
    ) -> None:
        previous_applied_strategy = execution_plan.applied_strategy

        execution_plan.use_simulated_annealing = (
            algorithm_policy.use_simulated_annealing
        )
        execution_plan.sa_iterations = algorithm_policy.sa_iterations

        if execution_plan.use_simulated_annealing:
            execution_plan.use_balance_aware_scoring = True

        execution_plan.applied_strategy = infer_applied_strategy(
            group_by_destination=execution_plan.group_by_destination,
            use_balance_aware_scoring=(
                execution_plan.use_balance_aware_scoring
            ),
            use_simulated_annealing=(
                execution_plan.use_simulated_annealing
            ),
            requested_strategy=execution_plan.requested_strategy,
            actual_destination_count=(
                algorithm_policy.destination_count
            ),
        )

        if execution_plan.applied_strategy != previous_applied_strategy:
            execution_plan.resolution_notes.append(
                "Algorithm policy adjusted applied strategy from "
                f"'{previous_applied_strategy}' to "
                f"'{execution_plan.applied_strategy}' based on batch "
                "complexity and time-budget controls."
            )

        self.enable_low_utilisation_merge = (
            algorithm_policy.enable_low_utilisation_merge
        )
        self.enable_wagon_reassignment = (
            algorithm_policy.enable_wagon_reassignment
        )

    def prepare_product_groups(
        self,
        products: List[Product],
        execution_plan: StrategyExecutionPlan,
    ) -> Dict[str, List[Product]]:
        """
        Return executable product groups.

        When group_by_destination=False and multiple destinations exist,
        products are deliberately packed together under MIXED_DESTINATION.
        """

        if execution_plan.group_by_destination:
            return group_by_destination(products)

        if len({product.destination for product in products}) <= 1:
            only_destination = (
                products[0].destination
                if products
                else "UNKNOWN_DEST"
            )

            return {only_destination: list(products)}

        return {"MIXED_DESTINATION": list(products)}

    def pack_all_groups(
        self,
        grouped_products: Dict[str, List[Product]],
        execution_plan: StrategyExecutionPlan,
        algorithm_policy: Optional[AlgorithmPolicy] = None,
        force_unsorted_for_sa_candidate: bool = False,
        collect_trace: bool = False,
    ) -> Any:
        packed_groups: Dict[str, List[List[Positioned]]] = {}
        optimisation_trace: List[Dict[str, Any]] = []

        for destination, products in grouped_products.items():
            result = pack_destination_into_cars(
                products=products,
                spec=self.spec,
                parallel_gap=self.parallel_gap,
                bogie_tol_tons=self.bogie_tol_tons,
                max_weight_diff_for_pair=(
                    self.max_weight_diff_for_pair
                ),
                max_outer_diameter_diff_for_pair=(
                    self.max_outer_diameter_diff_for_pair
                ),
                force_same_outer_diameter_for_pair=(
                    self.force_same_outer_diameter_for_pair
                    if execution_plan.strict_pairing_rules
                    else False
                ),
                sort_products_by_weight=(
                    False
                    if force_unsorted_for_sa_candidate
                    else execution_plan.sort_by_weight_desc
                ),
                allow_parallel_pairs=(
                    execution_plan.allow_parallel_pairs
                ),
                use_balance_aware_scoring=(
                    execution_plan.use_balance_aware_scoring
                ),
                enable_low_utilisation_merge=(
                    self.enable_low_utilisation_merge
                ),
                low_utilisation_threshold=(
                    self.low_utilisation_threshold
                ),
                enable_wagon_reassignment=(
                    self.enable_wagon_reassignment
                ),
                max_wagon_reassignment_passes=(
                    self.max_wagon_reassignment_passes
                ),
                trace_destination=destination,
                return_trace=collect_trace,
            )

            if collect_trace:
                cars, trace = result
                packed_groups[destination] = cars
                optimisation_trace.extend(trace)
            else:
                packed_groups[destination] = result

        if collect_trace:
            return packed_groups, optimisation_trace

        return packed_groups

    def run_simulated_annealing(
        self,
        grouped_products: Dict[str, List[Product]],
        execution_plan: StrategyExecutionPlan,
        initial_packed_groups: Optional[
            Dict[str, List[List[Positioned]]]
        ] = None,
        deadline_monotonic: Optional[float] = None,
    ) -> Tuple[Dict[str, List[List[Positioned]]], Dict[str, Any]]:
        """
        A lightweight real SA stage.

        It perturbs product sequence within each destination/mixed group and
        repacks candidates. Constraints remain enforced by the packing logic.
        """

        if initial_packed_groups is None:
            initial_packed_groups = self.pack_all_groups(
                grouped_products=grouped_products,
                execution_plan=execution_plan,
            )

        current_groups = {
            destination: list(products)
            for destination, products in grouped_products.items()
        }

        current_packed_groups = initial_packed_groups

        current_objective = self.evaluate_packed_groups(
            current_packed_groups
        )

        best_groups = {
            destination: list(products)
            for destination, products in current_groups.items()
        }

        best_packed_groups = current_packed_groups
        best_objective = current_objective

        accepted_moves = 0
        iterations_completed = 0
        iteration_limit = max(1, execution_plan.sa_iterations)
        rejected_invalid_candidates = 0
        move_attempt_counts: Dict[str, int] = defaultdict(int)
        move_accept_counts: Dict[str, int] = defaultdict(int)
        termination_reason = "iteration_limit_reached"

        for iteration in range(iteration_limit):
            if (
                deadline_monotonic is not None
                and time.perf_counter() >= deadline_monotonic
            ):
                termination_reason = "deadline_exhausted"
                break
            candidate_groups = {
                destination: list(products)
                for destination, products in current_groups.items()
            }

            mutable_destinations = [
                destination
                for destination, products in candidate_groups.items()
                if len(products) >= 2
            ]

            if not mutable_destinations:
                break

            selected_destination = self.random.choice(
                mutable_destinations
            )

            candidate_products = candidate_groups[selected_destination]

            move_type = self.perturb_products_for_simulated_annealing(
                candidate_products
            )
            move_attempt_counts[move_type] += 1

            try:
                candidate_packed_groups = self.pack_all_groups(
                    grouped_products=candidate_groups,
                    execution_plan=execution_plan,
                    force_unsorted_for_sa_candidate=True,
                )
            except ValueError:
                rejected_invalid_candidates += 1
                iterations_completed += 1
                continue

            candidate_objective = self.evaluate_packed_groups(
                candidate_packed_groups
            )

            temperature = max(
                0.05,
                35.0 * (
                    1.0 - iteration / max(iteration_limit, 1)
                ),
            )

            delta = candidate_objective - current_objective

            accept_candidate = delta <= 0

            if not accept_candidate:
                acceptance_probability = math.exp(
                    -delta / temperature
                )

                accept_candidate = (
                    self.random.random() < acceptance_probability
                )

            if accept_candidate:
                current_groups = candidate_groups
                current_packed_groups = candidate_packed_groups
                current_objective = candidate_objective
                accepted_moves += 1
                move_accept_counts[move_type] += 1

            if candidate_objective < best_objective:
                best_groups = {
                    destination: list(products)
                    for destination, products in candidate_groups.items()
                }
                best_packed_groups = candidate_packed_groups
                best_objective = candidate_objective

            iterations_completed += 1

        return (
            best_packed_groups,
            {
                "enabled": True,
                "executed": True,
                "iterations_requested": iteration_limit,
                "iterations_completed": iterations_completed,
                "termination_reason": termination_reason,
                "accepted_moves": accepted_moves,
                "rejected_invalid_candidates": (
                    rejected_invalid_candidates
                ),
                "move_attempt_counts": dict(move_attempt_counts),
                "move_accept_counts": dict(move_accept_counts),
                "initial_objective": round(
                    self.evaluate_packed_groups(
                        initial_packed_groups
                    ),
                    6,
                ),
                "final_objective": round(best_objective, 6),
                "note": (
                    "A simulated annealing stage perturbed product order "
                    "with swap, move, and segment-reversal moves. "
                    "Candidates were repacked through the same feasible "
                    "loading pipeline before objective comparison."
                ),
                "best_group_sizes": {
                    destination: len(products)
                    for destination, products in best_groups.items()
                },
            },
        )

    def build_anytime_trace_entry(
        self,
        stage: str,
        packed_groups: Dict[str, List[List[Positioned]]],
        expected_material_numbers: List[str],
        search_objective_score: float,
        elapsed_seconds: float,
        status: str,
        reason: str,
        candidate_validation: Optional[Dict[str, Any]] = None,
        baseline_search_objective: Optional[float] = None,
        search_improvement: Optional[float] = None,
    ) -> Dict[str, Any]:
        wagons = self.convert_packed_groups_to_wagons(packed_groups)
        validation = candidate_validation or validate_loading_solution(
            wagons=wagons,
            expected_material_numbers=expected_material_numbers,
        )
        solution_score = evaluate_loading_solution(
            wagons=wagons,
            weights=ObjectiveWeights(),
            expected_material_numbers=expected_material_numbers,
        )

        return {
            "stage": stage,
            "destination": "ALL",
            "status": status,
            "reason": reason,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "wagon_count": len(wagons),
            "product_count": sum(
                len(wagon.get("products", []))
                for wagon in wagons
            ),
            "objective_score": round(
                safe_float(solution_score.get("objective_score")),
                6,
            ),
            "search_objective_score": round(
                search_objective_score,
                6,
            ),
            "baseline_search_objective": (
                round(baseline_search_objective, 6)
                if baseline_search_objective is not None
                else None
            ),
            "search_objective_improvement": (
                round(search_improvement, 6)
                if search_improvement is not None
                else None
            ),
            "is_valid": validation["is_valid"],
            "violation_count": validation["violation_count"],
            "overlap_violation_count": validation[
                "overlap_violation_count"
            ],
            "boundary_violation_count": validation[
                "boundary_violation_count"
            ],
            "overweight_violation_count": validation[
                "overweight_violation_count"
            ],
            "coverage_violation_count": validation[
                "coverage_violation_count"
            ],
        }

    def perturb_products_for_simulated_annealing(
        self,
        products: List[Product],
    ) -> str:
        """
        Apply one in-place sequence perturbation for simulated annealing.

        Product order drives candidate block generation when SA repacks with
        force_unsorted_for_sa_candidate=True, so these moves explore pairings
        and loading order without bypassing hard layout validation.
        """

        if len(products) < 2:
            return "noop"

        if len(products) == 2:
            move_type = "swap_products"
        else:
            move_type = self.random.choice(
                [
                    "swap_products",
                    "move_product",
                    "reverse_segment",
                ]
            )

        if move_type == "swap_products":
            index_one, index_two = self.random.sample(
                range(len(products)),
                2,
            )
            products[index_one], products[index_two] = (
                products[index_two],
                products[index_one],
            )
            return move_type

        if move_type == "move_product":
            from_index, to_index = self.random.sample(
                range(len(products)),
                2,
            )
            product = products.pop(from_index)
            products.insert(to_index, product)
            return move_type

        start_index, end_index = sorted(
            self.random.sample(
                range(len(products)),
                2,
            )
        )

        products[start_index:end_index + 1] = reversed(
            products[start_index:end_index + 1]
        )

        return "reverse_segment"

    def evaluate_packed_groups(
        self,
        packed_groups: Dict[str, List[List[Positioned]]],
    ) -> float:
        """
        Lower objective is better. Uses the same central objective function
        as final solution scoring so optimisation stages share one standard.
        """

        score = evaluate_loading_solution(
            wagons=self.convert_packed_groups_to_wagons(packed_groups),
            weights=ObjectiveWeights(
                wagon_count=1000.0,
                validation_violation=1000000.0,
                unused_area=10.0,
                front_rear_balance=150.0,
                left_right_balance=100.0,
                diagonal_balance=80.0,
                mixed_destination=50.0,
                operational_complexity=5.0,
            ),
        )

        return safe_float(score.get("objective_score"))

    def convert_packed_groups_to_wagons(
        self,
        packed_groups: Dict[str, List[List[Positioned]]],
    ) -> List[Dict[str, Any]]:
        wagons: List[Dict[str, Any]] = []
        global_car_index = 1

        for destination, cars in packed_groups.items():
            for local_car_index, placed in enumerate(cars, start=1):
                while f"W{global_car_index:03d}" in self.disabled_wagon_ids:
                    global_car_index += 1
                wagon_id = f"W{global_car_index:03d}"
                global_car_index += 1

                wagon = self.build_wagon_json(
                    wagon_id=wagon_id,
                    destination=destination,
                    local_car_index=local_car_index,
                    placed=placed,
                )

                wagons.append(wagon)

        return wagons

    # -------------------------------------------------------------------------
    # Input loading
    # -------------------------------------------------------------------------

    def load_products(self) -> List[Product]:
        if not self.products_json_path.exists():
            raise FileNotFoundError(
                f"products.json not found: {self.products_json_path}. "
                "Please run DataAgent first."
            )

        with self.products_json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)

        if not isinstance(raw, list):
            raise ValueError(
                "products.json should contain a list of product records."
            )

        products: List[Product] = []

        for index, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                continue

            material = first_present(
                item,
                [
                    "material_no",
                    "product_id",
                    "materialNo",
                    "material_number",
                    "materialNumber",
                    "材料号",
                    "物料号",
                    "钢卷号",
                    "编号",
                ],
            )

            material_no = (
                str(material).strip()
                if material is not None
                else f"UNKNOWN-{index:04d}"
            )

            if not material_no:
                material_no = f"UNKNOWN-{index:04d}"

            destination = normalise_destination(
                first_present(
                    item,
                    [
                        "destination",
                        "destination_station",
                        "dest",
                        "to",
                        "目的地",
                        "到站",
                    ],
                )
            )

            gross_weight = safe_float(
                first_present(
                    item,
                    [
                        "gross_weight_tonne",
                        "gross_weight",
                        "grossWeight",
                        "毛重",
                        "重量(t)",
                        "重量(T)",
                    ],
                )
            )

            if gross_weight <= 0:
                weight_kg = safe_float(
                    first_present(
                        item,
                        [
                            "weight",
                            "gross_weight_kg",
                            "重量",
                        ],
                    )
                )

                gross_weight = (
                    weight_kg / 1000.0
                    if weight_kg > 1000
                    else weight_kg
                )

            net_weight = safe_float(
                first_present(
                    item,
                    [
                        "net_weight_tonne",
                        "net_weight",
                        "净重",
                    ],
                ),
                default=0.0,
            )

            width = safe_float(
                first_present(
                    item,
                    [
                        "width",
                        "demo_width",
                        "coil_width",
                        "宽度",
                    ],
                )
            )

            outer_diameter = safe_float(
                first_present(
                    item,
                    [
                        "outer_diameter",
                        "length",
                        "outerDiameter",
                        "外径",
                        "diameter",
                        "直径",
                    ],
                )
            )

            thickness = safe_float(
                first_present(
                    item,
                    [
                        "thickness",
                        "厚度",
                    ],
                ),
                default=0.0,
            )

            if (
                gross_weight <= 0
                or width <= 0
                or outer_diameter <= 0
            ):
                continue

            products.append(
                Product(
                    material_no=material_no,
                    destination=destination,
                    gross_weight=gross_weight,
                    width=width,
                    outer_diameter=outer_diameter,
                    thickness=thickness,
                    net_weight=net_weight,
                )
            )

        if not products:
            raise ValueError(
                "No valid products found in products.json."
            )

        return products

    def load_strategy_plan(self) -> Dict[str, Any]:
        if not self.strategy_plan_path.exists():
            raise FileNotFoundError(
                f"strategy_plan.json not found: "
                f"{self.strategy_plan_path}. "
                "Please run LLMAgent first."
            )

        with self.strategy_plan_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "strategy_plan.json should contain a JSON object."
            )

        return data

    # -------------------------------------------------------------------------
    # Output conversion
    # -------------------------------------------------------------------------

    def build_wagon_json(
        self,
        wagon_id: str,
        destination: str,
        local_car_index: int,
        placed: List[Positioned],
    ) -> Dict[str, Any]:
        visual_products = [
            positioned_to_visual_product(
                positioned,
                self.spec.width,
            )
            for positioned in placed
        ]

        for product in visual_products:
            product["wagon_id"] = wagon_id

        current_weight_kg = sum(
            product["weight"]
            for product in visual_products
        )

        current_weight_tonne = current_weight_kg / 1000.0

        area_used = sum(
            safe_float(product.get("placed_length"))
            * safe_float(product.get("placed_width"))
            for product in visual_products
        )

        area_total = self.spec.length * self.spec.width

        area_utilisation = (
            area_used / area_total
            if area_total > 0
            else 0.0
        )

        visual_balance = compute_visual_balance_metrics(
            products=visual_products,
            car_length=self.spec.length,
            car_width=self.spec.width,
        )

        bogie_metrics = compute_bogie_metrics_for_car(
            placed=placed,
            spec=self.spec,
        )

        visual_balance.update(bogie_metrics)

        return {
            "wagon_id": wagon_id,
            "destination": destination,
            "local_car_index": local_car_index,
            "length": self.spec.length,
            "width": self.spec.width,
            "max_weight": self.spec.max_weight * 1000.0,
            "max_weight_tonne": self.spec.max_weight,
            "current_weight": current_weight_kg,
            "current_weight_tonne": current_weight_tonne,
            "area_utilisation": area_utilisation,
            "balance_metrics": visual_balance,
            "products": visual_products,
        }

    def build_solution_json(
        self,
        strategy_plan: Dict[str, Any],
        execution_plan: StrategyExecutionPlan,
        wagons: List[Dict[str, Any]],
        products: List[Product],
        dropped_duplicates: List[str],
        runtime_seconds: float,
        simulated_annealing_log: Dict[str, Any],
        optimisation_trace: Optional[List[Dict[str, Any]]] = None,
        algorithm_policy: Optional[AlgorithmPolicy] = None,
    ) -> Dict[str, Any]:
        total_products = sum(
            len(wagon.get("products", []))
            for wagon in wagons
        )

        total_weight = sum(
            safe_float(wagon.get("current_weight"))
            for wagon in wagons
        )

        score = evaluate_loading_solution(
            wagons=wagons,
            weights=ObjectiveWeights(),
            expected_material_numbers=[
                product.material_no for product in products
            ],
            enforce_hard_constraints=True,
        )

        validation_summary = score["validation_summary"]
        optimisation_trace = list(optimisation_trace or [])
        optimisation_trace.append(
            {
                "stage": "final_solution",
                "destination": "ALL",
                "wagon_count": len(wagons),
                "product_count": total_products,
                "objective_score": score["objective_score"],
                "average_area_utilisation": score[
                    "average_area_utilisation"
                ],
                "is_valid": validation_summary["is_valid"],
                "violation_count": validation_summary[
                    "violation_count"
                ],
                "overlap_violation_count": validation_summary[
                    "overlap_violation_count"
                ],
                "boundary_violation_count": validation_summary[
                    "boundary_violation_count"
                ],
                "overweight_violation_count": validation_summary[
                    "overweight_violation_count"
                ],
                "coverage_violation_count": validation_summary[
                    "coverage_violation_count"
                ],
            }
        )
        anytime_entries = [
            entry
            for entry in optimisation_trace
            if str(entry.get("stage", "")).startswith("anytime_")
        ]
        accepted_anytime_entries = [
            entry
            for entry in anytime_entries
            if entry.get("status") == "accepted"
        ]
        baseline_entry = next(
            (
                entry
                for entry in anytime_entries
                if entry.get("stage") == "anytime_baseline"
            ),
            None,
        )
        final_accepted_entry = (
            accepted_anytime_entries[-1]
            if accepted_anytime_entries
            else None
        )
        candidate_entries = [
            entry
            for entry in anytime_entries
            if entry.get("stage") != "anytime_baseline"
        ]
        accepted_candidate_entries = [
            entry
            for entry in candidate_entries
            if entry.get("status") == "accepted"
        ]
        baseline_objective = (
            safe_float(baseline_entry.get("objective_score"))
            if baseline_entry
            else safe_float(score.get("objective_score"))
        )
        final_objective = safe_float(score.get("objective_score"))
        objective_improvement = baseline_objective - final_objective
        improvement_percent = (
            objective_improvement / baseline_objective * 100.0
            if baseline_objective > 0
            else 0.0
        )
        baseline_elapsed = (
            safe_float(baseline_entry.get("elapsed_seconds"))
            if baseline_entry
            else 0.0
        )
        final_accepted_elapsed = (
            safe_float(final_accepted_entry.get("elapsed_seconds"))
            if final_accepted_entry
            else runtime_seconds
        )
        candidate_seconds = max(
            0.0,
            final_accepted_elapsed - baseline_elapsed,
        )
        performance_summary = {
            "baseline_seconds": round(baseline_elapsed, 4),
            "candidate_seconds": round(candidate_seconds, 4),
            "total_seconds": round(runtime_seconds, 4),
            "baseline_objective": round(baseline_objective, 6),
            "final_objective": round(final_objective, 6),
            "accepted_improvement": round(objective_improvement, 6),
            "accepted_improvement_percent": round(
                improvement_percent,
                6,
            ),
            "baseline_wagon_count": (
                baseline_entry.get("wagon_count")
                if baseline_entry
                else len(wagons)
            ),
            "final_wagon_count": len(wagons),
            "accepted_candidate_count": len(accepted_candidate_entries),
            "rejected_candidate_count": len(
                [
                    entry
                    for entry in candidate_entries
                    if entry.get("status") == "rejected"
                ]
            ),
        }
        algorithm_policy_dict = (
            algorithm_policy.to_dict()
            if algorithm_policy is not None
            else None
        )
        algorithm_decision_trace = [
            {
                "step": "llm_strategy_request",
                "decision": strategy_plan.get("selected_strategy"),
                "reason": strategy_plan.get("strategy_reason"),
                "details": {
                    "requested_scenario_type": strategy_plan.get(
                        "scenario_type"
                    ),
                    "optimisation_config": strategy_plan.get(
                        "optimisation_config"
                    ),
                },
            },
            {
                "step": "batch_reality_check",
                "decision": execution_plan.actual_batch_scenario,
                "reason": (
                    "The optimiser derived the actual batch scenario "
                    "from loaded products before executing any strategy."
                ),
                "details": {
                    "product_count": len(products),
                    "destination_count": len(
                        {product.destination for product in products}
                    ),
                    "destinations": sorted(
                        {product.destination for product in products}
                    ),
                },
            },
            {
                "step": "strategy_normalisation",
                "decision": execution_plan.applied_strategy,
                "reason": (
                    "The LLM recommendation was converted into safe, "
                    "executable optimisation controls."
                ),
                "details": {
                    "requested_strategy": (
                        execution_plan.requested_strategy
                    ),
                    "resolution_notes": execution_plan.resolution_notes,
                    "fallback_reasons": execution_plan.fallback_reasons,
                },
            },
            {
                "step": "algorithm_policy_selection",
                "decision": (
                    algorithm_policy.selected_pipeline
                    if algorithm_policy is not None
                    else []
                ),
                "reason": (
                    "Local policy selected optimisation stages using "
                    "batch complexity and time-budget controls."
                ),
                "details": algorithm_policy_dict or {},
            },
            {
                "step": "anytime_baseline",
                "decision": (
                    baseline_entry.get("status")
                    if baseline_entry
                    else None
                ),
                "reason": (
                    baseline_entry.get("reason")
                    if baseline_entry
                    else "Baseline trace entry was not available."
                ),
                "details": baseline_entry or {},
            },
        ]

        for entry in candidate_entries:
            algorithm_decision_trace.append(
                {
                    "step": entry.get("stage"),
                    "decision": entry.get("status"),
                    "reason": entry.get("reason"),
                    "details": entry,
                }
            )

        algorithm_decision_trace.append(
            {
                "step": "final_solution",
                "decision": "accepted",
                "reason": (
                    "The final solution passed all hard constraints and "
                    "was exported for visualisation."
                ),
                "details": {
                    "objective_score": final_objective,
                    "is_valid": validation_summary["is_valid"],
                    "violation_count": validation_summary[
                        "violation_count"
                    ],
                    "wagon_count": len(wagons),
                    "total_products": total_products,
                    "performance_summary": performance_summary,
                },
            }
        )

        return {
            "solution_metadata": {
                "generated_by": "OptimisationAgent",
                "version": "6.0_strategy_driven_center_first",
                "runtime_seconds": runtime_seconds,
                "random_seed": self.seed,
                "note": (
                    "Strategy-driven centre-first steel coil loading. "
                    "Mandatory safety constraints remain enforced "
                    "independently of LLM preferences."
                ),
            },
            "strategy_execution": {
                "requested_strategy": execution_plan.requested_strategy,
                "requested_scenario_type": (
                    execution_plan.requested_scenario_type
                ),
                "applied_strategy": execution_plan.applied_strategy,
                "actual_batch_scenario": (
                    execution_plan.actual_batch_scenario
                ),
                "execution_config": {
                    "group_by_destination": (
                        execution_plan.group_by_destination
                    ),
                    "sort_by_weight_desc": (
                        execution_plan.sort_by_weight_desc
                    ),
                    "use_balance_aware_scoring": (
                        execution_plan.use_balance_aware_scoring
                    ),
                    "use_simulated_annealing": (
                        execution_plan.use_simulated_annealing
                    ),
                    "use_rule_based_constraints": (
                        execution_plan.use_rule_based_constraints
                    ),
                    "allow_parallel_pairs": (
                        execution_plan.allow_parallel_pairs
                    ),
                    "strict_pairing_rules": (
                        execution_plan.strict_pairing_rules
                    ),
                    "sa_iterations": execution_plan.sa_iterations,
                    "max_outer_diameter_diff_for_pair": (
                        self.max_outer_diameter_diff_for_pair
                    ),
                    "enable_low_utilisation_merge": (
                        self.enable_low_utilisation_merge
                    ),
                    "low_utilisation_threshold": (
                        self.low_utilisation_threshold
                    ),
                    "enable_wagon_reassignment": (
                        self.enable_wagon_reassignment
                    ),
                    "max_wagon_reassignment_passes": (
                        self.max_wagon_reassignment_passes
                    ),
                },
                "resolution_notes": execution_plan.resolution_notes,
                "fallback_reasons": execution_plan.fallback_reasons,
                "algorithm_policy": algorithm_policy_dict,
            },
            "llm_strategy_plan_summary": {
                "scenario_type": strategy_plan.get("scenario_type"),
                "selected_strategy": strategy_plan.get(
                    "selected_strategy"
                ),
                "strategy_reason": strategy_plan.get(
                    "strategy_reason"
                ),
                "optimisation_config": strategy_plan.get(
                    "optimisation_config"
                ),
                "required_constraints": strategy_plan.get(
                    "required_constraints"
                ),
                "risk_notes": strategy_plan.get("risk_notes"),
            },
            "input_summary": {
                "product_count": len(products),
                "dropped_duplicate_count": len(dropped_duplicates),
                "dropped_duplicates": dropped_duplicates,
                "destinations": sorted(
                    {
                        product.destination
                        for product in products
                    }
                ),
            },
            "initial_solution_score": score,
            "final_solution_score": score,
            "validation_summary": validation_summary,
            "anytime_summary": {
                "enabled": True,
                "stage_count": len(anytime_entries),
                "accepted_stage_count": len(accepted_anytime_entries),
                "candidate_stage_count": max(
                    0,
                    len(anytime_entries) - 1,
                ),
                "final_accepted_stage": (
                    final_accepted_entry.get("stage")
                    if final_accepted_entry
                    else None
                ),
                "guarantee": (
                    "The optimiser first creates a hard-constraint-valid "
                    "baseline and only accepts later candidates when they "
                    "remain valid and improve or match the objective."
                ),
            },
            "performance_summary": performance_summary,
            "algorithm_decision_trace": algorithm_decision_trace,
            "optimisation_trace": optimisation_trace,
            "simulated_annealing_log": simulated_annealing_log,
            "wagon_summary": {
                "wagon_count": len(wagons),
                "total_products": total_products,
                "total_weight": total_weight,
                "average_area_utilisation": (
                    score["average_area_utilisation"]
                ),
                "wagon_ids": [
                    wagon.get("wagon_id")
                    for wagon in wagons
                ],
            },
            "wagons": wagons,
        }

    def export_json(
        self,
        data: Dict[str, Any],
        path: str | Path,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimisation Agent using strategy-driven "
            "steel coil loading."
        )
    )

    parser.add_argument(
        "--products",
        default="Agents/products.json",
        help="Path to products.json generated by DataAgent.",
    )

    parser.add_argument(
        "--strategy",
        default="Agents/strategy_plan.json",
        help="Path to strategy_plan.json generated by LLMAgent.",
    )

    parser.add_argument(
        "--output",
        default="Agents/loading_solution.json",
        help="Path to output loading_solution.json.",
    )

    parser.add_argument(
        "--car-length",
        type=float,
        default=13000.0,
        help="Freight car length in mm.",
    )

    parser.add_argument(
        "--car-width",
        type=float,
        default=3000.0,
        help="Freight car width in mm.",
    )

    parser.add_argument(
        "--max-weight-tonne",
        type=float,
        default=60.0,
        help="Maximum car loading weight in tonnes.",
    )

    parser.add_argument(
        "--parallel-gap",
        type=float,
        default=50.0,
        help="Gap between loading blocks in mm.",
    )

    parser.add_argument(
        "--bogie-tol-tons",
        type=float,
        default=2.0,
        help="Allowed front/rear bogie difference in tonnes.",
    )

    parser.add_argument(
        "--max-weight-diff-for-pair",
        type=float,
        default=1.0,
        help="Maximum parallel-pair weight difference in tonnes.",
    )

    parser.add_argument(
        "--max-outer-diameter-diff-for-pair",
        type=float,
        default=100.0,
        help="Maximum parallel-pair outer diameter difference in mm.",
    )

    parser.add_argument(
        "--allow-different-outer-diameter-pair",
        action="store_true",
        help="Allow parallel pairs with different outer diameters.",
    )

    parser.add_argument(
        "--disable-low-utilisation-merge",
        action="store_true",
        help="Disable conservative low-utilisation wagon merging.",
    )

    parser.add_argument(
        "--low-utilisation-threshold",
        type=float,
        default=0.35,
        help="Area utilisation threshold for merge candidates.",
    )

    parser.add_argument(
        "--disable-wagon-reassignment",
        action="store_true",
        help="Disable same-destination cross-wagon reassignment.",
    )

    parser.add_argument(
        "--max-wagon-reassignment-passes",
        type=int,
        default=2,
        help="Maximum same-destination wagon reassignment passes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    agent = OptimisationAgent(
        products_json_path=args.products,
        strategy_plan_path=args.strategy,
        output_path=args.output,
        car_length=args.car_length,
        car_width=args.car_width,
        max_weight_tonne=args.max_weight_tonne,
        parallel_gap=args.parallel_gap,
        bogie_tol_tons=args.bogie_tol_tons,
        max_weight_diff_for_pair=(
            args.max_weight_diff_for_pair
        ),
        max_outer_diameter_diff_for_pair=(
            args.max_outer_diameter_diff_for_pair
        ),
        force_same_outer_diameter_for_pair=(
            not args.allow_different_outer_diameter_pair
        ),
        enable_low_utilisation_merge=(
            not args.disable_low_utilisation_merge
        ),
        low_utilisation_threshold=args.low_utilisation_threshold,
        enable_wagon_reassignment=(
            not args.disable_wagon_reassignment
        ),
        max_wagon_reassignment_passes=(
            args.max_wagon_reassignment_passes
        ),
        seed=args.seed,
    )

    agent.run()


if __name__ == "__main__":
    main()
