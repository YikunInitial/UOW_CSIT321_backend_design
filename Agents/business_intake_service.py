"""Browser-safe business file intake with evidence-backed repair review."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from Agents.data_agent import DataAgent
from Agents.data_repair_service import analyse_repairs, apply_confirmed_repairs
from Agents.input_service import save_uploaded_products
from Agents.product_schema import (
    MAX_SINGLE_PRODUCT_WEIGHT_TONNE,
    standardise_dataframe_with_mapping,
    validate_product_records,
)
from Agents.run_manifest import sha256_file
from Agents.runtime_config import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)

CORE_ALIASES: Dict[str, tuple[str, ...]] = {
    "material_no": ("材料号", "物料号", "material_no", "material number", "product_id"),
    "thickness": ("厚度", "thickness"),
    "width": ("宽度", "带宽", "width"),
    "outer_diameter": ("外径", "卷径", "直径", "outer_diameter", "outer diameter"),
    "gross_weight_tonne": ("毛重", "毛重吨", "gross_weight_tonne", "gross weight"),
    "net_weight_tonne": ("净重", "净重吨", "net_weight_tonne", "net weight"),
    "destination": ("到站", "目的地", "收货地", "destination"),
}
OPTIONAL_ALIASES: Dict[str, tuple[str, ...]] = {
    "product_name": ("产品名称", "品名", "product_name", "product name", "description"),
    "receiver": ("收货人", "收货单位", "receiver", "consignee"),
    "contract_no": ("合同号", "合同编号", "contract_no", "contract number"),
    "storage_location": ("库位", "库区", "storage_location", "storage location"),
    "remarks": ("备注", "说明", "remarks", "remark", "note"),
}
ALLOWED_SUFFIXES = frozenset({".xlsx", ".xls", ".csv", ".json"})
LLM_ASSISTANCE_FILE = "llm_assistance.json"
NUMERIC_CORE_FIELDS = (
    "thickness",
    "width",
    "outer_diameter",
    "gross_weight_tonne",
    "net_weight_tonne",
)
AUXILIARY_HEADER_MARKERS = frozenset(
    {
        "车皮号",
        "车号",
        "wagonid",
        "wagonno",
        "wagonnumber",
        "railcarid",
        "railcarno",
        "railcarnumber",
    }
)


class _StoredUpload:
    def __init__(self, path: Path, original_name: str) -> None:
        self.path = path
        self.name = original_name

    def getbuffer(self) -> bytes:
        return self.path.read_bytes()


def _normalise_column(value: Any) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(value).strip().casefold())


def safe_filename(value: str, *, default: str = "upload.xlsx") -> str:
    name = Path(value or default).name
    stem = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", Path(name).stem)[:80]
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("仅支持 Excel、CSV 或 JSON 文件。")
    return f"{stem or 'upload'}{suffix}"


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = _read_excel_business_frame(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON 顶层必须是产品记录列表。")
        frame = pd.DataFrame(value)
    else:
        raise ValueError("仅支持 Excel、CSV 或 JSON 文件。")
    # Preserve empty business columns: an empty but recognised core field must
    # remain visible so authoritative master data can fill it during review.
    attrs = dict(frame.attrs)
    cleaned = frame.dropna(how="all")
    cleaned.attrs.update(attrs)
    return cleaned


def _header_alias_matches(values: list[Any]) -> set[str]:
    normalised = {_normalise_column(value) for value in values if _has_value(value)}
    matches = {
        field
        for field, aliases in {**CORE_ALIASES, **OPTIONAL_ALIASES}.items()
        if any(_normalise_column(alias) in normalised for alias in aliases)
    }
    return matches


def _read_excel_business_frame(path: Path) -> pd.DataFrame:
    """Find a likely business header across sheets before falling back to sheet 1."""
    with pd.ExcelFile(path) as workbook:
        sheet_names = list(workbook.sheet_names)
        structural_candidates: list[tuple[float, str, int]] = []
        for sheet_name in sheet_names:
            raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)
            for header_index in range(min(len(raw), 30)):
                matches = _header_alias_matches(raw.iloc[header_index].tolist())
                if "material_no" in matches and len(matches) >= 2:
                    frame = pd.read_excel(
                        workbook,
                        sheet_name=sheet_name,
                        header=header_index,
                    )
                    frame.attrs.update(
                        {
                            "source_sheet_name": str(sheet_name),
                            "source_header_row": header_index + 1,
                            "sheet_names": sheet_names,
                            "structure_detection": "known_business_header",
                        }
                    )
                    return frame
                header_values = [
                    str(value).strip()
                    for value in raw.iloc[header_index].tolist()
                    if _has_value(value)
                ]
                if len(header_values) < 7 or len(set(header_values)) < 7:
                    continue
                sample = raw.iloc[header_index + 1 : header_index + 21]
                if sample.empty:
                    continue
                populated_columns = sample.notna().sum().ge(1)
                numeric_columns = sum(
                    pd.to_numeric(sample[column], errors="coerce").gt(0).sum() >= 1
                    for column in sample.columns[populated_columns]
                )
                populated_rows = int(sample.notna().any(axis=1).sum())
                if numeric_columns >= 4 and populated_rows:
                    structural_candidates.append(
                        (
                            numeric_columns * 10 + min(populated_rows, 20) + len(header_values) / 100,
                            str(sheet_name),
                            header_index,
                        )
                    )
        if structural_candidates:
            _, sheet_name, header_index = max(structural_candidates, key=lambda item: item[0])
            frame = pd.read_excel(workbook, sheet_name=sheet_name, header=header_index)
            frame.attrs.update(
                {
                    "source_sheet_name": sheet_name,
                    "source_header_row": header_index + 1,
                    "sheet_names": sheet_names,
                    "structure_detection": "candidate_business_table",
                }
            )
            return frame
        frame = pd.read_excel(workbook, sheet_name=sheet_names[0])
        frame.attrs.update(
            {
                "source_sheet_name": str(sheet_names[0]),
                "source_header_row": 1,
                "sheet_names": sheet_names,
                "structure_detection": "unresolved_first_sheet_fallback",
            }
        )
        return frame


def infer_business_mapping(dataframe: pd.DataFrame) -> Dict[str, str] | None:
    columns = {
        _normalise_column(column): str(column)
        for column in dataframe.columns
        if not str(column).strip().casefold().startswith("unnamed")
    }
    mapping: Dict[str, str] = {}
    for target, aliases in CORE_ALIASES.items():
        source = next(
            (columns[_normalise_column(alias)] for alias in aliases if _normalise_column(alias) in columns),
            None,
        )
        if source is None:
            return None
        mapping[target] = source
    for target, aliases in OPTIONAL_ALIASES.items():
        source = next(
            (columns[_normalise_column(alias)] for alias in aliases if _normalise_column(alias) in columns),
            None,
        )
        if source is not None:
            mapping[target] = source
    return mapping


def _has_value(value: Any) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _is_positive_number(value: Any) -> bool:
    if not _has_value(value):
        return False
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return not pd.isna(converted) and float(converted) > 0


def _is_repeated_product_header(row: pd.Series, mapping: Dict[str, str]) -> bool:
    matches = 0
    for field, aliases in CORE_ALIASES.items():
        source = mapping[field]
        value = _normalise_column(row.get(source))
        if value and value in {_normalise_column(alias) for alias in aliases}:
            matches += 1
    return matches >= 3


def _is_auxiliary_header(row: pd.Series, mapping: Dict[str, str]) -> bool:
    marker = _normalise_column(row.get(mapping["material_no"]))
    return marker in AUXILIARY_HEADER_MARKERS


def _is_auxiliary_detail_after_header(
    row: pd.Series,
    mapping: Dict[str, str],
) -> bool:
    """Recognise a summary record only when an explicit auxiliary header precedes it."""
    if not _has_value(row.get(mapping["material_no"])):
        return False
    if _has_value(row.get(mapping["destination"])):
        return False
    thickness = row.get(mapping["thickness"])
    if not _has_value(thickness) or _is_positive_number(thickness):
        return False
    numeric_evidence = sum(
        _is_positive_number(row.get(mapping[field]))
        for field in NUMERIC_CORE_FIELDS
        if field != "thickness"
    )
    return numeric_evidence >= 2


def classify_business_rows(
    dataframe: pd.DataFrame,
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Classify mixed business-report rows without discarding uncertain products."""
    material = dataframe[mapping["material_no"]]
    has_material = material.notna() & material.astype(str).str.strip().ne("")
    structural_mask = pd.Series(False, index=dataframe.index, dtype=bool)
    excluded_rows: list[Dict[str, Any]] = []
    auxiliary_header_pending = False

    for position, (index, row) in enumerate(dataframe.iterrows()):
        source_row = _source_row(dataframe, index, position)
        if _is_repeated_product_header(row, mapping):
            structural_mask.loc[index] = True
            excluded_rows.append(
                {"source_row": source_row, "reason": "repeated_product_header"}
            )
            auxiliary_header_pending = False
            continue
        if _is_auxiliary_header(row, mapping):
            structural_mask.loc[index] = True
            excluded_rows.append(
                {"source_row": source_row, "reason": "auxiliary_section_header"}
            )
            auxiliary_header_pending = True
            continue
        if auxiliary_header_pending:
            if _is_auxiliary_detail_after_header(row, mapping):
                structural_mask.loc[index] = True
                excluded_rows.append(
                    {"source_row": source_row, "reason": "auxiliary_section_record"}
                )
            auxiliary_header_pending = False

    candidate_mask = has_material & ~structural_mask
    valid_mask = candidate_mask & dataframe[mapping["destination"]].notna()
    valid_mask &= dataframe[mapping["destination"]].astype(str).str.strip().ne("")
    for field in NUMERIC_CORE_FIELDS:
        valid_mask &= pd.to_numeric(
            dataframe[mapping[field]], errors="coerce"
        ).gt(0)

    reason_counts: Dict[str, int] = {}
    for item in excluded_rows:
        reason = str(item["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "candidate_mask": candidate_mask,
        "valid_mask": valid_mask,
        "structural_mask": structural_mask,
        "excluded_rows": excluded_rows,
        "reason_counts": reason_counts,
    }


def duplicate_material_issues(
    dataframe: pd.DataFrame,
    mapping: Dict[str, str],
) -> list[Dict[str, Any]]:
    classification = classify_business_rows(dataframe, mapping)
    candidates = dataframe.loc[classification["candidate_mask"]]
    material_column = mapping["material_no"]
    normalised = candidates[material_column].map(
        lambda value: DataAgent._as_string(value).casefold()
    )
    duplicated = normalised.duplicated(keep=False)
    issues: list[Dict[str, Any]] = []
    for identifier in normalised.loc[duplicated].drop_duplicates():
        matching_indexes = normalised.index[normalised.eq(identifier)]
        source_rows = [_source_row(dataframe, index, position) for position, index in enumerate(matching_indexes)]
        displayed = DataAgent._as_string(
            candidates.loc[matching_indexes[0], material_column]
        )
        issues.append(
            {
                "field": "material_no",
                "count": len(source_rows),
                "source_rows": source_rows,
                "material_no": displayed,
                "issue_type": "duplicate_material_no",
            }
        )
    return issues


def _input_structure(dataframe: pd.DataFrame) -> Dict[str, Any]:
    return {
        "selected_sheet": dataframe.attrs.get("source_sheet_name"),
        "header_row": dataframe.attrs.get("source_header_row"),
        "available_sheets": list(dataframe.attrs.get("sheet_names") or []),
        "detection": dataframe.attrs.get("structure_detection"),
    }


def _source_row(dataframe: pd.DataFrame, index: Any, fallback_position: int) -> int:
    """Return the one-based source row, including a detected Excel title/header offset."""
    header_row = int(dataframe.attrs.get("source_header_row") or 1)
    try:
        position = int(dataframe.index.get_loc(index))
    except (KeyError, TypeError, ValueError):
        position = fallback_position
    return header_row + position + 1


def _assistance_path(source_path: Path) -> Path:
    return source_path.parent / LLM_ASSISTANCE_FILE


def _load_llm_assistance(source_path: Path) -> Dict[str, Any] | None:
    cache_path = _assistance_path(source_path)
    if not cache_path.is_file():
        return None
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached.get("source_sha256") != sha256_file(source_path):
        return None
    products = cached.get("products")
    if validate_product_records(products):
        return None
    return cached


def _assisted_intake_report(cached: Dict[str, Any]) -> Dict[str, Any]:
    products = list(cached["products"])
    return {
        "planning_ready": True,
        "requires_assistance": False,
        "assistance_status": "completed",
        "message": "智能字段识别已完成，规范产品数据可以进入人工核对。",
        "record_count": len(products),
        "candidate_count": len(products),
        "invalid_product_count": 0,
        "duplicate_product_count": 0,
        "excluded_non_product_row_count": 0,
        "destination_count": len(
            {str(item.get("destination", "")).strip() for item in products}
        ),
        "preview": products[:30],
        "field_mapping": cached.get("data_intake_audit", {}).get("field_mapping", {}),
        "input_structure": cached.get("input_structure", {}),
        "repair_analysis": {
            "schema_version": "1.0",
            "suggestions": [],
            "unresolved": [],
            "policy": {
                "mode": "llm_field_mapping_with_canonical_validation",
                "original_input_mutated": False,
            },
        },
        "blocking_issues": [],
    }


def assist_intake(
    intake_dir: Path,
    *,
    max_products: int = 2000,
) -> Dict[str, Any]:
    metadata = json.loads((intake_dir / "intake.json").read_text(encoding="utf-8"))
    source_path = intake_dir / metadata["source_stored_name"]
    cached = _load_llm_assistance(source_path)
    if cached is not None:
        return _assisted_intake_report(cached)
    dataframe = read_dataframe(source_path)
    if infer_business_mapping(dataframe) is not None:
        return inspect_intake(source_path, max_products=max_products)
    if source_path.suffix.lower() not in {".xlsx", ".xls", ".json"}:
        raise ValueError("该文件无法使用智能字段识别，请使用标准 Excel 或 JSON 模板。")
    assist_dir = intake_dir / "llm_assist"
    assist_dir.mkdir(parents=True, exist_ok=True)
    assist_source = source_path
    assist_name = metadata["source_original_name"]
    if source_path.suffix.lower() in {".xlsx", ".xls"}:
        # The bounded LLM receives only the table selected during structure
        # discovery. It cannot silently choose a different Sheet or header.
        assist_source = assist_dir / "detected_business_table.xlsx"
        dataframe.to_excel(assist_source, index=False)
        assist_name = "detected_business_table.xlsx"
    products_path, audit = save_uploaded_products(
        _StoredUpload(assist_source, assist_name),
        assist_dir,
        llm_data_recovery_mode="Agent decides",
        llm_model_name=DEFAULT_LLM_MODEL,
        llm_base_url=DEFAULT_LLM_BASE_URL,
        llm_provider_name=DEFAULT_LLM_PROVIDER,
    )
    products = json.loads(products_path.read_text(encoding="utf-8"))
    issues = validate_product_records(products)
    if issues:
        raise ValueError("智能字段识别后的数据验证未通过：" + "；".join(issues[:5]))
    if len(products) > max_products:
        raise ValueError(f"产品数量超过当前部署上限 {max_products}。")
    audit.update(
        {
            "source_input_path": str(source_path),
            "source_input_sha256": sha256_file(source_path),
            "record_count": len(products),
            "input_structure": _input_structure(dataframe),
        }
    )
    cached = {
        "schema_version": "1.0",
        "source_sha256": sha256_file(source_path),
        "products": products,
        "data_intake_audit": audit,
        "input_structure": _input_structure(dataframe),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _assistance_path(source_path).write_text(
        json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return _assisted_intake_report(cached)


def select_candidate_rows(dataframe: pd.DataFrame, mapping: Dict[str, str]) -> tuple[pd.DataFrame, int, int]:
    classification = classify_business_rows(dataframe, mapping)
    candidate_mask = classification["candidate_mask"]
    valid_mask = classification["valid_mask"]
    candidates = int(candidate_mask.sum())
    return dataframe.loc[valid_mask].copy(), candidates, candidates - int(valid_mask.sum())


def blocking_core_issues(dataframe: pd.DataFrame, mapping: Dict[str, str]) -> list[Dict[str, Any]]:
    """Explain exactly which candidate rows prevent a complete planning request."""
    candidates = classify_business_rows(dataframe, mapping)["candidate_mask"]
    issues: list[Dict[str, Any]] = []
    for field in NUMERIC_CORE_FIELDS:
        invalid = candidates & ~pd.to_numeric(dataframe[mapping[field]], errors="coerce").gt(0)
        if invalid.any():
            issues.append({
                "field": field,
                "count": int(invalid.sum()),
                "source_rows": [
                    _source_row(dataframe, index, position)
                    for position, index in enumerate(dataframe.index[invalid][:20])
                ],
            })
    destination = dataframe[mapping["destination"]]
    invalid_destination = candidates & (
        destination.isna() | destination.astype(str).str.strip().eq("")
    )
    if invalid_destination.any():
        issues.append({
            "field": "destination",
            "count": int(invalid_destination.sum()),
            "source_rows": [
                _source_row(dataframe, index, position)
                for position, index in enumerate(dataframe.index[invalid_destination][:20])
            ],
        })
    gross_weight = pd.to_numeric(
        dataframe[mapping["gross_weight_tonne"]], errors="coerce"
    )
    excessive_weight = candidates & gross_weight.gt(MAX_SINGLE_PRODUCT_WEIGHT_TONNE)
    if excessive_weight.any():
        issues.append({
            "field": "gross_weight_tonne",
            "count": int(excessive_weight.sum()),
            "source_rows": [
                _source_row(dataframe, index, position)
                for position, index in enumerate(dataframe.index[excessive_weight][:20])
            ],
            "issue_type": "gross_weight_exceeds_deployment_limit",
            "maximum_tonne": MAX_SINGLE_PRODUCT_WEIGHT_TONNE,
        })
    return issues


def preview_records(dataframe: pd.DataFrame, mapping: Dict[str, str], limit: int = 30) -> list[Dict[str, Any]]:
    labels = {
        "material_no": "material_no",
        "thickness": "thickness",
        "width": "width",
        "outer_diameter": "outer_diameter",
        "gross_weight_tonne": "gross_weight_tonne",
        "net_weight_tonne": "net_weight_tonne",
        "destination": "destination",
        "product_name": "product_name",
        "receiver": "receiver",
        "contract_no": "contract_no",
        "storage_location": "storage_location",
        "remarks": "remarks",
    }
    result = pd.DataFrame(
        {label: dataframe[source] for field, source in mapping.items() if (label := labels.get(field))}
    ).head(limit)
    return json.loads(result.to_json(orient="records", force_ascii=False))


def inspect_intake(
    source_path: Path,
    *,
    master_path: Path | None = None,
    accepted_repair_ids: list[str] | None = None,
    max_products: int = 2000,
) -> Dict[str, Any]:
    dataframe = read_dataframe(source_path)
    mapping = infer_business_mapping(dataframe)
    if mapping is None:
        cached_assistance = _load_llm_assistance(source_path)
        if cached_assistance is not None:
            return _assisted_intake_report(cached_assistance)
        if source_path.suffix.lower() == ".json":
            records = json.loads(source_path.read_text(encoding="utf-8"))
            issues = validate_product_records(records)
            if not issues:
                if len(records) > max_products:
                    raise ValueError(f"产品数量 {len(records)} 超过当前部署上限 {max_products}。")
                return {
                    "planning_ready": True,
                    "requires_assistance": False,
                    "canonical_json": True,
                    "message": "规范 JSON 数据检查通过，可以确认后生成方案。",
                    "record_count": len(records),
                    "destination_count": len({str(item.get("destination")) for item in records}),
                    "invalid_product_count": 0,
                    "preview": records[:30],
                    "repair_analysis": {
                        "schema_version": "1.0", "suggestions": [], "unresolved": [],
                        "policy": {"mode": "validated_canonical_json", "original_input_mutated": False},
                    },
                }
        llm_supported = source_path.suffix.lower() in {".xlsx", ".xls", ".json"}
        return {
            "planning_ready": False,
            "requires_assistance": llm_supported,
            "assistance_status": "pending" if llm_supported else "unavailable",
            "message": (
                "尚未确认产品字段，必须先完成智能字段识别并核对产品预览。"
                if llm_supported
                else "核心字段名称无法确定；CSV 请使用标准字段模板。"
            ),
            "columns": [str(item) for item in dataframe.columns],
            "record_count": None,
            "candidate_count": None,
            "invalid_product_count": None,
            "duplicate_product_count": None,
            "input_structure": _input_structure(dataframe),
            "preview": [],
            "repair_analysis": {
                "schema_version": "1.0",
                "suggestions": [],
                "unresolved": [],
                "policy": {
                    "mode": "pending_llm_field_mapping",
                    "original_input_mutated": False,
                },
            },
        }
    master = read_dataframe(master_path) if master_path else None
    analysis = analyse_repairs(
        dataframe,
        mapping,
        master_dataframe=master,
        master_source_name=master_path.name if master_path else None,
    )
    valid_repair_ids = {str(item["repair_id"]) for item in analysis.suggestions}
    selected_ids = [
        str(item) for item in (accepted_repair_ids or []) if str(item) in valid_repair_ids
    ]
    inspected_dataframe, inspected_mapping, confirmed_repairs = apply_confirmed_repairs(
        dataframe, mapping, analysis.suggestions, selected_ids
    )
    valid, candidate_count, invalid_count = select_candidate_rows(
        inspected_dataframe, inspected_mapping
    )
    row_classification = classify_business_rows(
        inspected_dataframe, inspected_mapping
    )
    duplicate_issues = duplicate_material_issues(
        inspected_dataframe, inspected_mapping
    )
    duplicate_count = sum(max(0, int(item["count"]) - 1) for item in duplicate_issues)
    core_issues = blocking_core_issues(inspected_dataframe, inspected_mapping)
    implausible_weight_rows = {
        int(row)
        for item in core_issues
        if item.get("issue_type") == "gross_weight_exceeds_deployment_limit"
        for row in item.get("source_rows", [])
    }
    total_invalid_count = invalid_count + len(implausible_weight_rows)
    if candidate_count > max_products:
        raise ValueError(f"产品数量 {candidate_count} 超过当前部署上限 {max_products}。")
    return {
        "planning_ready": bool(len(valid)) and total_invalid_count == 0 and duplicate_count == 0,
        "requires_assistance": False,
        "message": (
            "数据检查通过，可以确认后生成方案。"
            if total_invalid_count == 0 and duplicate_count == 0
            else (
                f"发现 {duplicate_count} 条重复材料号记录，请修正后重新上传。"
                if duplicate_count
                else (
                    f"发现 {len(implausible_weight_rows)} 条产品重量超过当前单件上限 "
                    f"{MAX_SINGLE_PRODUCT_WEIGHT_TONNE:g} 吨，请核对重量单位和源数据。"
                    if implausible_weight_rows
                    else f"仍有 {invalid_count} 条产品缺少有效核心字段。"
                )
            )
        ),
        "record_count": int(len(valid)),
        "candidate_count": candidate_count,
        "invalid_product_count": total_invalid_count,
        "duplicate_product_count": duplicate_count,
        "excluded_non_product_row_count": int(len(dataframe) - candidate_count),
        "destination_count": int(valid[inspected_mapping["destination"]].nunique()) if not valid.empty else 0,
        "field_mapping": inspected_mapping,
        "preview": preview_records(valid, inspected_mapping),
        "repair_analysis": analysis.as_dict(),
        "selected_repair_ids": selected_ids,
        "confirmed_repairs": confirmed_repairs,
        "blocking_issues": core_issues + duplicate_issues,
        "input_structure": _input_structure(inspected_dataframe),
        "row_classification": {
            "policy": "evidence_backed_mixed_report_v1",
            "structural_row_count": int(row_classification["structural_mask"].sum()),
            "reason_counts": row_classification["reason_counts"],
            "excluded_rows": row_classification["excluded_rows"],
        },
    }


def finalise_intake(
    intake_dir: Path,
    *,
    accepted_repair_ids: list[str],
    reviewed_by: str,
    max_products: int = 2000,
) -> Dict[str, Any]:
    metadata = json.loads((intake_dir / "intake.json").read_text(encoding="utf-8"))
    source_path = intake_dir / metadata["source_stored_name"]
    master_path = intake_dir / metadata["master_stored_name"] if metadata.get("master_stored_name") else None
    dataframe = read_dataframe(source_path)
    mapping = infer_business_mapping(dataframe)
    if mapping is None:
        if source_path.suffix.lower() == ".json":
            products = json.loads(source_path.read_text(encoding="utf-8"))
            issues = validate_product_records(products)
            if issues:
                raise ValueError("业务数据完整性检查未通过：" + "；".join(issues[:5]))
            if len(products) > max_products:
                raise ValueError(f"产品数量超过当前部署上限 {max_products}。")
            products_path = intake_dir / "products.json"
            products_path.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
            now = datetime.now(timezone.utc).isoformat()
            audit = {
                "schema_version": "1.0", "input_file": metadata["source_original_name"],
                "input_suffix": ".json", "source_input_path": str(source_path),
                "source_input_sha256": sha256_file(source_path), "field_mapping": {},
                "deterministic_parser": "validated_canonical_json",
                "deterministic_parser_succeeded": True, "llm_called": False,
                "llm_call_reason": "The JSON already matched a supported product schema.",
                "record_count": len(products),
                "data_repair": {"policy": {"mode": "validated_canonical_json"}, "suggestion_count": 0,
                    "confirmed_repair_count": 0, "unresolved_count": 0, "reviewed_by": reviewed_by,
                    "reviewed_at": now, "original_input_mutated": False},
            }
            audit_path = intake_dir / "data_repair_audit.json"
            audit["repair_audit_path"] = str(audit_path)
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"products": products, "data_intake_audit": audit}
        cached_assistance = _load_llm_assistance(source_path)
        if cached_assistance is None:
            raise ValueError("必须先完成智能字段识别并核对产品预览，才能进入规划。")
        products = list(cached_assistance["products"])
        products_path = intake_dir / "products.json"
        products_path.write_text(
            json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        intake_audit = dict(cached_assistance["data_intake_audit"])
        intake_audit.update(
            {
                "source_input_path": str(source_path),
                "source_input_sha256": sha256_file(source_path),
                "record_count": len(products),
                "operator_reviewed_by": reviewed_by,
                "operator_reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        audit_path = intake_dir / "data_repair_audit.json"
        intake_audit["repair_audit_path"] = str(audit_path)
        audit_path.write_text(
            json.dumps(intake_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"products": products, "data_intake_audit": intake_audit}
    master = read_dataframe(master_path) if master_path else None
    analysis = analyse_repairs(
        dataframe,
        mapping,
        master_dataframe=master,
        master_source_name=metadata.get("master_original_name"),
    )
    valid_ids = {str(item["repair_id"]) for item in analysis.suggestions}
    accepted = list(dict.fromkeys(str(item) for item in accepted_repair_ids))
    unknown = set(accepted) - valid_ids
    if unknown:
        raise ValueError("修复确认与当前源文件不一致，请重新预检。")
    repaired, mapping, changes = apply_confirmed_repairs(
        dataframe, mapping, analysis.suggestions, accepted
    )
    repaired_path = intake_dir / "repaired_source.xlsx"
    repaired.to_excel(repaired_path, index=False)
    products_frame, candidate_count, invalid_count = select_candidate_rows(repaired, mapping)
    row_classification = classify_business_rows(repaired, mapping)
    if invalid_count:
        raise ValueError(f"仍有 {invalid_count} 条产品缺少有效核心字段，不能进入规划。")
    if products_frame.empty:
        raise ValueError("没有找到可用于规划的有效产品记录。")
    duplicate_issues = duplicate_material_issues(repaired, mapping)
    if duplicate_issues:
        details = "；".join(
            f"材料号 {item['material_no']}（源文件行 {', '.join(str(row) for row in item['source_rows'])}）"
            for item in duplicate_issues[:5]
        )
        raise ValueError("存在重复材料号：" + details)
    if len(products_frame) > max_products:
        raise ValueError(f"产品数量超过当前部署上限 {max_products}。")
    products_path = intake_dir / "products.json"
    products = standardise_dataframe_with_mapping(products_frame, mapping, products_path)
    issues = validate_product_records(products)
    if issues:
        raise ValueError("业务数据完整性检查未通过：" + "；".join(issues[:5]))
    now = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": "1.0",
        "input_file": metadata["source_original_name"],
        "input_suffix": source_path.suffix.lower(),
        "source_input_path": str(source_path),
        "source_input_sha256": sha256_file(source_path),
        "repaired_source_path": str(repaired_path),
        "repaired_source_sha256": sha256_file(repaired_path),
        "authoritative_master_path": str(master_path) if master_path else None,
        "authoritative_master_sha256": sha256_file(master_path) if master_path else None,
        "field_mapping": mapping,
        "deterministic_parser": "business_intake_service",
        "deterministic_parser_succeeded": True,
        "llm_called": False,
        "llm_call_reason": "Known Chinese or English business columns were mapped deterministically.",
        "record_count": len(products),
        "candidate_count": candidate_count,
        "duplicate_product_count": 0,
        "excluded_non_product_row_count": int(len(repaired) - candidate_count),
        "row_classification": {
            "policy": "evidence_backed_mixed_report_v1",
            "structural_row_count": int(row_classification["structural_mask"].sum()),
            "reason_counts": row_classification["reason_counts"],
            "excluded_rows": row_classification["excluded_rows"],
        },
        "input_structure": _input_structure(repaired),
        "data_repair": {
            "policy": analysis.as_dict()["policy"],
            "suggestion_count": len(analysis.suggestions),
            "confirmed_repair_count": len(changes),
            "unresolved_count": len(analysis.unresolved),
            "confirmed_repair_ids": accepted,
            "confirmed_repairs": changes,
            "reviewed_by": reviewed_by,
            "reviewed_at": now,
            "original_input_mutated": False,
        },
    }
    audit_path = intake_dir / "data_repair_audit.json"
    audit["repair_audit_path"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"products": products, "data_intake_audit": audit}
