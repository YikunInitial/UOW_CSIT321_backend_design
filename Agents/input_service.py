"""Deterministic-first product ingestion with bounded LLM mapping fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from Agents.data_agent import DataAgent
from Agents.llm_provider import create_llm_provider
from Agents.planning_orchestrator import PlanningOrchestrator
from Agents.product_schema import (
    STANDARD_PRODUCT_FIELDS,
    standardise_dataframe_with_mapping,
    validate_product_records,
)
from Agents.runtime_config import DEFAULT_LLM_PROVIDER


def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(raw_text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM did not return a JSON object.")


def infer_product_field_mapping(
    dataframe: pd.DataFrame,
    *,
    model_name: str,
    base_url: str,
    provider_name: Optional[str],
) -> Dict[str, Any]:
    columns = [str(column) for column in dataframe.columns]
    samples = (
        dataframe.head(5)
        .where(pd.notna(dataframe.head(5)), None)
        .to_dict(orient="records")
    )
    field_definitions = {
        "material_no": "唯一材料号/钢卷编号/物料编号 (unique material or coil identifier)",
        "thickness": "钢材厚度 (thickness)",
        "width": "钢材宽度 (width)",
        "outer_diameter": "钢卷外径/卷径 (outer coil diameter)",
        "gross_weight_tonne": "毛重，单位吨 (gross weight in tonnes)",
        "net_weight_tonne": "净重，单位吨 (net weight in tonnes)",
        "destination": "到站/卸货城市/目的地 (delivery destination)",
    }
    semantic_tokens = {
        "material_no": ("材料", "物料", "钢卷", "编号", "material", "product", "coil", "id", "no"),
        "thickness": ("厚", "thickness", "gauge"),
        "width": ("宽", "width"),
        "outer_diameter": ("外径", "卷径", "直径", "diameter", "outer"),
        "gross_weight_tonne": ("毛重", "gross"),
        "net_weight_tonne": ("净重", "net"),
        "destination": ("到站", "目的", "卸货", "城市", "地点", "destination", "delivery", "city"),
    }
    normalised_columns = {column: column.casefold().replace(" ", "") for column in columns}
    candidate_columns = {
        field: [
            column for column, normalised in normalised_columns.items()
            if any(token.casefold() in normalised for token in tokens)
        ]
        for field, tokens in semantic_tokens.items()
    }
    proposed_mapping = {
        field: values[0] if len(values) == 1 else None
        for field, values in candidate_columns.items()
    }
    prompt = {
        "task": "从真实输入列中选择每个钢材产品字段。Map every required field to one exact available column.",
        "field_definitions": field_definitions,
        "available_columns": columns,
        "candidate_columns_from_business_terms": candidate_columns,
        "sample_rows": samples,
        "rules": [
            "检查 proposed_mapping；正确则原样保留，错误则改为 available_columns 中的正确列名。",
            "field_mapping 的每个值必须逐字复制 available_columns 中的一个真实列名。",
            "每个 required field 都必须填写；不确定时填写 null，禁止编造列名。",
            "候选列只是提示，必须结合字段含义和样例值判断。",
            "不要转换数据，不要批准装载方案，只返回 JSON。",
        ],
        "output_json_template": {
            "field_mapping": proposed_mapping,
            "confidence": "high",
            "reason": "简短说明",
        },
    }
    response = create_llm_provider(
        prefix="CLIENT_LLM",
        provider=provider_name,
        model=model_name,
        base_url=base_url,
    ).invoke(
        [
            {
                "role": "system",
                "content": (
                    "你是钢材业务字段识别器。只从提供的 available_columns 中选择列名，"
                    "填写 output_json_template 中的 null；禁止照抄说明文字，禁止编造业务事实。"
                    "Return one valid JSON object only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=500,
        num_ctx=4096,
        seed=42,
        json_mode=True,
    )
    parsed = _parse_json_object(response.content)
    mapping = parsed.get("field_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("LLM response is missing field_mapping.")
    missing = [field for field in STANDARD_PRODUCT_FIELDS if not mapping.get(field)]
    if missing:
        raise ValueError(f"LLM could not map required fields: {missing}")
    unknown = [source for source in mapping.values() if source not in columns]
    if unknown:
        raise ValueError(f"LLM selected unknown columns: {unknown}")
    parsed["_provider_metadata"] = response.response_metadata
    return parsed


def save_uploaded_products(
    uploaded_file: Any,
    run_dir: Path,
    llm_data_recovery_mode: str,
    llm_model_name: str,
    llm_base_url: str,
    llm_provider_name: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".xlsx", ".xls", ".json"}:
        raise ValueError("Only Excel and JSON product files are supported.")
    raw_path = run_dir / f"uploaded{suffix}"
    raw_path.write_bytes(uploaded_file.getbuffer())
    products_path = run_dir / "products.json"
    audit: Dict[str, Any] = {
        "input_file": uploaded_file.name,
        "input_suffix": suffix,
        "llm_data_recovery_mode": llm_data_recovery_mode,
        "llm_provider": llm_provider_name or DEFAULT_LLM_PROVIDER,
        "deterministic_parser": None,
        "deterministic_parser_succeeded": False,
        "deterministic_quality_issues": [],
        "llm_called": False,
        "llm_call_reason": None,
    }
    force_llm = llm_data_recovery_mode == "Force on input"
    llm_allowed = llm_data_recovery_mode != "Disabled"
    deterministic_error: Optional[Exception] = None
    if not force_llm:
        try:
            if suffix in {".xlsx", ".xls"}:
                audit["deterministic_parser"] = "DataAgent"
                records = DataAgent().run(raw_path, products_path)
            else:
                audit["deterministic_parser"] = "JSON loader"
                records = json.loads(raw_path.read_text(encoding="utf-8"))
            audit["deterministic_quality_issues"] = validate_product_records(records)
            if not audit["deterministic_quality_issues"]:
                PlanningOrchestrator.write_json(products_path, records)
                audit["deterministic_parser_succeeded"] = True
                return products_path, audit
        except Exception as error:
            deterministic_error = error
            audit["deterministic_error"] = str(error)
    critical_fact_issues = [
        issue
        for issue in audit["deterministic_quality_issues"]
        if "exceeds the deployed single-product limit" in issue
    ]
    if critical_fact_issues:
        raise ValueError(
            "Critical loading weight validation failed; LLM assistance is not "
            "permitted to reinterpret or overwrite weight: "
            + "; ".join(critical_fact_issues[:5])
        )
    if not llm_allowed:
        if deterministic_error:
            raise deterministic_error
        raise ValueError(
            "Input checks failed and LLM data recovery is disabled: "
            + "; ".join(audit["deterministic_quality_issues"][:5])
        )
    dataframe = (
        pd.read_excel(raw_path)
        if suffix in {".xlsx", ".xls"}
        else pd.DataFrame(json.loads(raw_path.read_text(encoding="utf-8")))
    )
    if dataframe.empty:
        raise ValueError("Input file is empty.")
    inferred = infer_product_field_mapping(
        dataframe,
        model_name=llm_model_name,
        base_url=llm_base_url,
        provider_name=llm_provider_name,
    )
    records = standardise_dataframe_with_mapping(
        dataframe, inferred["field_mapping"], products_path
    )
    audit.update(
        {
            "llm_called": True,
            "llm_call_reason": (
                "User forced LLM data recovery."
                if force_llm
                else "Deterministic parsing or validation did not succeed."
            ),
            "field_mapping": inferred["field_mapping"],
            "confidence": inferred.get("confidence"),
            "reason": inferred.get("reason"),
            "llm_request_metadata": inferred.get("_provider_metadata", {}),
            "recovered_product_count": len(records),
        }
    )
    return products_path, audit
