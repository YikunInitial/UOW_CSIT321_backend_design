"""Evidence-driven pre-planning data repair without silent source mutation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

CORE_PLANNING_FIELDS = frozenset(
    {
        "material_no",
        "thickness",
        "width",
        "outer_diameter",
        "gross_weight_tonne",
        "net_weight_tonne",
        "destination",
    }
)
OPTIONAL_FACT_FIELDS = frozenset(
    {"product_name", "receiver", "contract_no", "storage_location", "remarks"}
)
REPAIRABLE_FIELDS = CORE_PLANNING_FIELDS | OPTIONAL_FACT_FIELDS

MASTER_FIELD_ALIASES: Dict[str, tuple[str, ...]] = {
    "material_no": ("material_no", "material number", "材料号", "物料号"),
    "thickness": ("thickness", "厚度"),
    "width": ("width", "宽度", "带宽"),
    "outer_diameter": ("outer_diameter", "outer diameter", "外径", "卷径"),
    "gross_weight_tonne": (
        "gross_weight_tonne",
        "gross weight",
        "毛重",
        "毛重吨",
    ),
    "net_weight_tonne": ("net_weight_tonne", "net weight", "净重", "净重吨"),
    "destination": ("destination", "目的地", "到站", "收货地"),
    "product_name": ("product_name", "product name", "产品名称", "品名"),
    "receiver": ("receiver", "consignee", "收货人", "收货单位", "收货用户"),
    "contract_no": ("contract_no", "contract number", "合同号", "合同编号"),
    "storage_location": ("storage_location", "storage location", "库位", "库区"),
    "remarks": ("remarks", "remark", "note", "备注", "说明"),
}


def _normalise_column(value: Any) -> str:
    return "".join(character for character in str(value).strip().casefold() if character.isalnum())


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _normalised_value(value: Any) -> str | None:
    text = _clean_text(value)
    if text is None:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text.casefold()
    return format(number, ".12g")


def _suggestion_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return "repair_" + hashlib.sha256(encoded).hexdigest()[:16]


def infer_master_mapping(dataframe: pd.DataFrame) -> Dict[str, str]:
    """Map any recognised master-data columns; material_no is mandatory."""
    normalised = {
        _normalise_column(column): str(column)
        for column in dataframe.columns
        if not str(column).strip().casefold().startswith("unnamed")
    }
    mapping: Dict[str, str] = {}
    for target, aliases in MASTER_FIELD_ALIASES.items():
        source = next(
            (
                normalised[_normalise_column(alias)]
                for alias in aliases
                if _normalise_column(alias) in normalised
            ),
            None,
        )
        if source is not None:
            mapping[target] = source
    if "material_no" not in mapping:
        raise ValueError("权威主数据必须包含材料号/material_no列。")
    if len(mapping) == 1:
        raise ValueError("权威主数据除材料号外至少需要一个可用于核对的字段。")
    return mapping


@dataclass(frozen=True)
class RepairAnalysis:
    suggestions: list[Dict[str, Any]]
    unresolved: list[Dict[str, Any]]
    master_mapping: Dict[str, str]
    duplicate_master_materials: list[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "policy": {
                "mode": "evidence_driven_preplanning_repair",
                "original_input_mutated": False,
                "operator_confirmation_required": True,
                "missing_without_evidence": "leave_blank",
            },
            "suggestions": self.suggestions,
            "unresolved": self.unresolved,
            "master_mapping": self.master_mapping,
            "duplicate_master_materials": self.duplicate_master_materials,
        }


def analyse_repairs(
    dataframe: pd.DataFrame,
    field_mapping: Mapping[str, str],
    *,
    master_dataframe: pd.DataFrame | None = None,
    master_source_name: str | None = None,
) -> RepairAnalysis:
    """Create format and exact-material master-data suggestions."""
    master_mapping: Dict[str, str] = {}
    master_by_material: Dict[str, Mapping[str, Any]] = {}
    duplicate_master_materials: list[str] = []
    if master_dataframe is not None:
        master_mapping = infer_master_mapping(master_dataframe)
        for _, master_row in master_dataframe.iterrows():
            material = _clean_text(master_row.get(master_mapping["material_no"]))
            if not material:
                continue
            key = material.casefold()
            if key in master_by_material:
                duplicate_master_materials.append(material)
                master_by_material.pop(key, None)
                continue
            if material not in duplicate_master_materials:
                master_by_material[key] = master_row

    suggestions: list[Dict[str, Any]] = []
    unresolved: list[Dict[str, Any]] = []
    for row_position, (_, row) in enumerate(dataframe.iterrows()):
        source_row = row_position + 2
        material_source = field_mapping.get("material_no")
        material = _clean_text(row.get(material_source)) if material_source else None

        for field, source_column in field_mapping.items():
            if field not in REPAIRABLE_FIELDS:
                continue
            raw_value = row.get(source_column)
            cleaned = _clean_text(raw_value)
            if cleaned is None or not isinstance(raw_value, str) or raw_value == cleaned:
                continue
            payload = {
                "row_position": row_position,
                "field": field,
                "original_value": raw_value,
                "suggested_value": cleaned,
                "evidence_type": "original_cell_format",
            }
            suggestions.append(
                {
                    "repair_id": _suggestion_id(payload),
                    "source_row": source_row,
                    "row_position": row_position,
                    "material_no": material,
                    "field": field,
                    "original_value": raw_value,
                    "suggested_value": cleaned,
                    "repair_type": "deterministic_format_repair",
                    "evidence": f"原始文件第{source_row}行/{source_column}",
                    "confidence": "certain",
                    "changes_loading_input": field in CORE_PLANNING_FIELDS,
                }
            )

        master_row = master_by_material.get(material.casefold()) if material else None
        for field in sorted(REPAIRABLE_FIELDS - {"material_no"}):
            source_column = field_mapping.get(field)
            source_value = row.get(source_column) if source_column else None
            authoritative_value = (
                master_row.get(master_mapping[field])
                if master_row is not None and field in master_mapping
                else None
            )
            source_normalised = _normalised_value(source_value)
            authoritative_normalised = _normalised_value(authoritative_value)
            if authoritative_normalised is not None and (
                source_normalised is None
                or source_normalised != authoritative_normalised
            ):
                suggested_value = _clean_text(authoritative_value)
                payload = {
                    "row_position": row_position,
                    "field": field,
                    "original_value": source_normalised,
                    "suggested_value": suggested_value,
                    "evidence_type": "exact_material_master_data",
                    "material_no": material,
                }
                suggestions.append(
                    {
                        "repair_id": _suggestion_id(payload),
                        "source_row": source_row,
                        "row_position": row_position,
                        "material_no": material,
                        "field": field,
                        "original_value": _clean_text(source_value),
                        "suggested_value": suggested_value,
                        "repair_type": (
                            "authoritative_fill"
                            if source_normalised is None
                            else "authoritative_correction"
                        ),
                        "evidence": (
                            f"{master_source_name or '权威主数据'}：材料号 {material} 精确匹配"
                        ),
                        "confidence": "authoritative_exact_match",
                        "changes_loading_input": field in CORE_PLANNING_FIELDS,
                    }
                )
            elif source_normalised is None:
                unresolved.append(
                    {
                        "source_row": source_row,
                        "material_no": material,
                        "field": field,
                        "reason": "no_authoritative_source_evidence",
                    }
                )

    return RepairAnalysis(
        suggestions=suggestions,
        unresolved=unresolved,
        master_mapping=master_mapping,
        duplicate_master_materials=sorted(set(duplicate_master_materials)),
    )


def apply_confirmed_repairs(
    dataframe: pd.DataFrame,
    field_mapping: Mapping[str, str],
    suggestions: Iterable[Mapping[str, Any]],
    accepted_repair_ids: Iterable[str],
) -> tuple[pd.DataFrame, Dict[str, str], list[Dict[str, Any]]]:
    """Apply selected suggestions to a copy and return a complete change log."""
    repaired = dataframe.copy(deep=True)
    mapping = dict(field_mapping)
    accepted = set(accepted_repair_ids)
    change_log: list[Dict[str, Any]] = []
    for suggestion in suggestions:
        if str(suggestion.get("repair_id")) not in accepted:
            continue
        field = str(suggestion["field"])
        row_position = int(suggestion["row_position"])
        if row_position < 0 or row_position >= len(repaired):
            raise ValueError("修复建议对应的数据行已发生变化，请重新预检。")
        source_column = mapping.get(field)
        if source_column is None:
            source_column = field
            repaired[source_column] = None
            mapping[field] = source_column
        current_value = repaired.iloc[row_position][source_column]
        expected = suggestion.get("original_value")
        if _normalised_value(current_value) != _normalised_value(expected):
            raise ValueError("原始数据与修复预览不一致，请重新预检后确认。")
        suggested_value = suggestion.get("suggested_value")
        if isinstance(suggested_value, str) and not pd.api.types.is_object_dtype(
            repaired[source_column].dtype
        ):
            repaired[source_column] = repaired[source_column].astype(object)
        column_index = repaired.columns.get_loc(source_column)
        repaired.iat[row_position, column_index] = suggested_value
        change_log.append(dict(suggestion))
    return repaired, mapping, change_log
