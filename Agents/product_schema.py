"""Canonical product schema helpers shared outside presentation code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from Agents.data_agent import DataAgent

STANDARD_PRODUCT_FIELDS = [
    "material_no",
    "thickness",
    "width",
    "outer_diameter",
    "gross_weight_tonne",
    "net_weight_tonne",
    "destination",
]
DOWNSTREAM_PRODUCT_FIELDS = [
    "product_id",
    "length",
    "width",
    "weight",
    "destination",
]

# The deployed wagon specification has a 60-tonne capacity.  A single product
# above that limit cannot be loaded by any available wagon and usually means
# that kilogram-scale values were supplied in a tonne-labelled field.  Keep
# this guard deterministic: an LLM must never reinterpret or rewrite weight.
MAX_SINGLE_PRODUCT_WEIGHT_TONNE = 60.0
MAX_SINGLE_PRODUCT_WEIGHT_KG = MAX_SINGLE_PRODUCT_WEIGHT_TONNE * 1000.0


def validate_product_records(records: Any) -> List[str]:
    if not isinstance(records, list):
        return ["Products JSON must be a list."]
    if not records:
        return ["Products JSON is empty."]
    issues: List[str] = []
    first = records[0]
    if not isinstance(first, dict):
        return ["Products JSON rows must be objects."]
    if not any(
        all(field in first for field in schema)
        for schema in (STANDARD_PRODUCT_FIELDS, DOWNSTREAM_PRODUCT_FIELDS)
    ):
        issues.append(
            "Product records do not contain a supported business or loading schema."
        )
    identifiers = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            issues.append(f"Row {index} is not an object.")
            continue
        identifier = str(
            record.get("material_no") or record.get("product_id") or ""
        ).strip()
        if not identifier:
            issues.append(f"Row {index} is missing a product identifier.")
        elif identifier in identifiers:
            issues.append(f"Row {index} duplicates product identifier {identifier}.")
        identifiers.add(identifier)
        if not str(record.get("destination") or "").strip():
            issues.append(f"Row {index} is missing destination.")
        for alternatives in (
            ("width",),
            ("outer_diameter", "length"),
            ("gross_weight_tonne", "weight"),
        ):
            raw = next(
                (
                    record.get(name)
                    for name in alternatives
                    if record.get(name) is not None
                ),
                None,
            )
            try:
                if raw is None or float(raw) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(f"Row {index} has invalid {alternatives[0]}.")
        gross_tonne = record.get("gross_weight_tonne")
        downstream_weight = record.get("weight")
        try:
            if (
                gross_tonne is not None
                and float(gross_tonne) > MAX_SINGLE_PRODUCT_WEIGHT_TONNE
            ):
                issues.append(
                    f"Row {index} gross_weight_tonne exceeds the deployed "
                    f"single-product limit of {MAX_SINGLE_PRODUCT_WEIGHT_TONNE:g} tonnes."
                )
        except (TypeError, ValueError):
            pass  # The generic numeric validation above already reports this value.
        try:
            if (
                gross_tonne is None
                and downstream_weight is not None
                and float(downstream_weight) > MAX_SINGLE_PRODUCT_WEIGHT_KG
            ):
                issues.append(
                    f"Row {index} weight exceeds the deployed single-product "
                    f"limit of {MAX_SINGLE_PRODUCT_WEIGHT_KG:g} kilograms."
                )
        except (TypeError, ValueError):
            pass
    return issues


def standardise_dataframe_with_mapping(
    dataframe: pd.DataFrame,
    field_mapping: Dict[str, str],
    output_path: Path,
) -> List[Dict[str, Any]]:
    missing_targets = set(STANDARD_PRODUCT_FIELDS) - set(field_mapping)
    missing_sources = {
        source for source in field_mapping.values() if source not in dataframe.columns
    }
    if missing_targets or missing_sources:
        raise ValueError(
            f"Invalid field mapping; missing targets={sorted(missing_targets)}, "
            f"missing sources={sorted(missing_sources)}"
        )
    mapped = pd.DataFrame(
        {target: dataframe[source] for target, source in field_mapping.items()}
    )
    agent = DataAgent()
    records = agent.convert_dataframe_to_records(mapped)
    issues = validate_product_records(records)
    if issues:
        raise ValueError("; ".join(issues[:10]))
    agent.export_json(records, output_path)
    return records
