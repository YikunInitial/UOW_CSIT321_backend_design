"""Bounded natural-language analytics over verified planning evidence."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from Agents.llm_provider import LLMProviderError, create_llm_provider
from Agents.china_geography_service import (
    CATALOG_VERSION,
    REGION_LABELS,
    classify_row,
    geography_region_from_text,
)

ALLOWED_GROUPS = {"province", "destination", "plan_status", "date"}
ALLOWED_METRICS = {
    "dispatch_count",
    "product_count",
    "weight_tonne",
    "average_weight_utilisation",
    "average_area_utilisation",
}
ALLOWED_INTENTS = {"analytics", "plan_explanation", "plan_comparison"}
ALLOWED_TOPICS = {"overview", "wagon_count", "balance", "constraints", "council", "strategy"}
ALLOWED_CHARTS = {"bar", "line", "table"}
GROUP_LABELS = {
    "province": "省份",
    "destination": "目的地",
    "plan_status": "方案状态",
    "date": "方案日期",
}
METRIC_LABELS = {
    "dispatch_count": "计划车皮数",
    "product_count": "产品数量",
    "weight_tonne": "装载重量（吨）",
    "average_weight_utilisation": "平均重量利用率",
    "average_area_utilisation": "平均面积利用率",
}


def _json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.S)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object")
        candidate = candidate[start : end + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("LLM query plan must be an object")
    return value


def _bounded_top_n(text: str) -> int | None:
    match = re.search(r"(?:前|top)([0-9一二三四五六七八九十两]{1,3})", text, re.I)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return min(max(int(token), 1), 50)
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if token == "十":
        value = 10
    elif "十" in token:
        left, right = token.split("十", 1)
        value = digits.get(left, 1) * 10 + digits.get(right, 0)
    else:
        value = digits.get(token)
    return min(max(value, 1), 50) if value else None


def deterministic_query_plan(
    question: str,
    rows: Iterable[Mapping[str, Any]],
    previous_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    text = "".join(question.split())
    previous = previous_plan or {}
    previous_group = str(previous.get("group_by") or "province")
    if previous_group not in ALLOWED_GROUPS:
        previous_group = "province"
    previous_metric = str(previous.get("metric") or "dispatch_count")
    if previous_metric not in ALLOWED_METRICS:
        previous_metric = "dispatch_count"
    try:
        previous_limit = min(max(int(previous.get("limit") or 10), 1), 50)
    except (TypeError, ValueError):
        previous_limit = 10
    comparison = any(word in text for word in ("对比方案", "比较方案", "两个方案", "版本变化", "前后方案", "最近方案"))
    explanation = any(word in text for word in ("为什么", "当前方案", "解释方案", "约束", "平衡", "Council", "council", "策略"))
    intent = "plan_comparison" if comparison else ("plan_explanation" if explanation else "analytics")
    if any(word in text for word in ("Council", "council", "候选")):
        topic = "council"
    elif "平衡" in text:
        topic = "balance"
    elif "约束" in text or "验证" in text:
        topic = "constraints"
    elif "策略" in text or "算法" in text:
        topic = "strategy"
    elif "车皮" in text:
        topic = "wagon_count"
    else:
        topic = "overview"
    if any(word in text for word in ("目的地", "到站", "收货地", "城市")):
        group_by = "destination"
    elif any(word in text for word in ("日期", "每天", "每日", "趋势")):
        group_by = "date"
    elif any(word in text for word in ("状态", "生效", "待批准", "拒绝")):
        group_by = "plan_status"
    else:
        group_by = previous_group

    if "面积" in text:
        metric = "average_area_utilisation"
    elif any(word in text for word in ("重量利用率", "利用率", "载重率", "载重利用率")):
        metric = "average_weight_utilisation"
    elif any(word in text for word in ("总重量", "装载重量", "吨位")):
        metric = "weight_tonne"
    elif any(word in text for word in ("产品", "材料", "货物")):
        metric = "product_count"
    else:
        metric = previous_metric

    requested_limit = _bounded_top_n(text)
    limit = requested_limit if requested_limit is not None else previous_limit
    previous_sort = str(previous.get("sort") or "desc")
    direction = "asc" if any(word in text for word in ("最低", "最少", "最差", "低到高")) else (previous_sort if previous_sort in {"asc", "desc"} else "desc")
    if any(word in text for word in ("最高", "最多", "高到低")):
        direction = "desc"
    chart_type = "line" if any(word in text for word in ("折线", "趋势", "变化曲线")) else ("table" if "表格" in text else "bar")
    known_provinces = sorted({str(row.get("province") or "") for row in rows if row.get("province")}, key=len, reverse=True)
    known_destinations = sorted({str(row.get("destination") or "") for row in rows if row.get("destination")}, key=len, reverse=True)
    filters: dict[str, str] = {}
    province = next((value for value in known_provinces if value in text or value.removesuffix("省") in text), "")
    destination = next((value for value in known_destinations if value in text), "")
    if province:
        filters["province"] = province
    if destination:
        filters["destination"] = destination
    geographic_region = geography_region_from_text(text)
    if geographic_region:
        filters["geographic_region"] = geographic_region
    return {
        "intent": intent,
        "topic": topic,
        "group_by": group_by,
        "metric": metric,
        "sort": direction,
        "limit": limit,
        "filters": filters,
        "chart_type": chart_type,
    }


def _validated_plan(value: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    intent = str(value.get("intent") or fallback.get("intent") or "analytics")
    topic = str(value.get("topic") or fallback.get("topic") or "overview")
    chart_type = str(value.get("chart_type") or fallback.get("chart_type") or "bar")
    group_by = str(value.get("group_by") or fallback["group_by"])
    metric = str(value.get("metric") or fallback["metric"])
    direction = str(value.get("sort") or fallback["sort"]).lower()
    if group_by not in ALLOWED_GROUPS or metric not in ALLOWED_METRICS:
        raise ValueError("LLM selected an unsupported analytics field")
    if intent not in ALLOWED_INTENTS or topic not in ALLOWED_TOPICS or chart_type not in ALLOWED_CHARTS:
        raise ValueError("LLM selected an unsupported assistant operation")
    if direction not in {"asc", "desc"}:
        direction = "desc"
    try:
        limit = min(max(int(value.get("limit", fallback["limit"])), 1), 50)
    except (TypeError, ValueError):
        limit = int(fallback["limit"])
    filters = dict(fallback.get("filters", {}))
    if isinstance(value.get("filters"), dict):
        filters.update(value["filters"])
    clean_filters = {
        key: str(item).strip()[:120]
        for key, item in filters.items()
        if key in {"province", "destination", "plan_status", "date", "geographic_region"} and str(item).strip()
    }
    if clean_filters.get("geographic_region") not in {None, *REGION_LABELS}:
        raise ValueError("LLM selected an unknown geographic region")
    return {"intent": intent, "topic": topic, "group_by": group_by, "metric": metric, "sort": direction, "limit": limit, "filters": clean_filters, "chart_type": chart_type}


def llm_query_plan(question: str, rows: list[Mapping[str, Any]], fallback: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    provider = create_llm_provider(prefix="ANALYTICS_LLM", timeout_seconds=20.0)
    dimensions = {
        "provinces": sorted({str(row.get("province")) for row in rows if row.get("province")})[:40],
        "destinations": sorted({str(row.get("destination")) for row in rows if row.get("destination")})[:80],
        "statuses": sorted({str(row.get("plan_status")) for row in rows if row.get("plan_status")}),
    }
    prompt = f"""You convert a Chinese business analytics question into a safe JSON query plan.
Return JSON only. Never write SQL. Use exactly this schema:
{{"intent":"analytics|plan_explanation|plan_comparison","topic":"overview|wagon_count|balance|constraints|council|strategy","group_by":"province|destination|plan_status|date","metric":"dispatch_count|product_count|weight_tonne|average_weight_utilisation|average_area_utilisation","sort":"asc|desc","limit":1-50,"filters":{{"province":"optional exact value","destination":"optional exact value","plan_status":"optional exact value","date":"optional YYYY-MM-DD","geographic_region":"optional north_of_yangtze|south_of_yangtze|yangtze_boundary|north_china|northeast|east_china|central_china|south_china|southwest|northwest|coastal"}},"chart_type":"bar|line|table"}}
Known values: {json.dumps(dimensions, ensure_ascii=False)}
Previous validated plan / deterministic context: {json.dumps(dict(fallback), ensure_ascii=False)}
Question: {question}
"""
    response = provider.invoke([
        {"role": "system", "content": "You are a bounded analytics intent parser. Output one JSON object only."},
        {"role": "user", "content": prompt},
    ])
    return _validated_plan(_json_object(response.content), fallback), response.response_metadata


def execute_query_plan(rows: Iterable[Mapping[str, Any]], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    filtered: list[Mapping[str, Any]] = []
    for row in rows:
        ordinary_filters = {key: value for key, value in plan["filters"].items() if key != "geographic_region"}
        region = plan["filters"].get("geographic_region")
        ordinary_match = all(str(row.get(key) or "") == expected for key, expected in ordinary_filters.items())
        region_match = classify_row(row, region) if region else True
        if ordinary_match and region_match is True:
            filtered.append(row)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in filtered:
        key = str(row.get(plan["group_by"]) or "未映射")
        groups[key].append(row)
    metric = str(plan["metric"])
    result: list[dict[str, Any]] = []
    for group, items in groups.items():
        if metric == "dispatch_count":
            value = len(items)
        elif metric in {"product_count", "weight_tonne"}:
            value = sum(float(item.get(metric) or 0) for item in items)
        else:
            source_field = {
                "average_weight_utilisation": "weight_utilisation",
                "average_area_utilisation": "area_utilisation",
            }[metric]
            value = sum(float(item.get(source_field) or 0) for item in items) / max(1, len(items))
        result.append({"group": group, "value": round(value, 6), "record_count": len(items)})
    if plan.get("chart_type") == "line" and plan["group_by"] == "date":
        result.sort(key=lambda item: item["group"])
    else:
        result.sort(key=lambda item: (item["value"], item["group"]), reverse=plan["sort"] == "desc")
    return result[: int(plan["limit"])]


def _value(value: Any, digits: int = 3) -> float:
    try:
        return round(float(value or 0), digits)
    except (TypeError, ValueError):
        return 0.0


def _plan_details(plan: Mapping[str, Any], topic: str) -> tuple[str, list[dict[str, Any]]]:
    summary = plan.get("summary") or {}
    score = plan.get("score") or {}
    validation = plan.get("validation") or {}
    strategy = plan.get("strategy") or {}
    council = plan.get("council") or {}
    common = [
        {"label": "方案版本", "value": f"{plan.get('plan_id', '—')} · v{plan.get('version', 0)}"},
        {"label": "产品数量", "value": int(summary.get("total_products") or 0)},
        {"label": "使用车皮", "value": int(summary.get("wagon_count") or score.get("wagon_count") or 0)},
        {"label": "平均面积利用率", "value": _value(score.get("average_area_utilisation"))},
    ]
    if topic == "balance":
        count = max(1, int(summary.get("wagon_count") or score.get("wagon_count") or 1))
        details = common[:3] + [
            {"label": "单车平均前后平衡惩罚", "value": _value(_value(score.get("front_rear_balance_penalty"), 6) / count, 6)},
            {"label": "单车平均左右平衡惩罚", "value": _value(_value(score.get("left_right_balance_penalty"), 6) / count, 6)},
            {"label": "单车平均对角平衡惩罚", "value": _value(_value(score.get("diagonal_balance_penalty"), 6) / count, 6)},
        ]
        answer = "当前方案在装载坐标计算中同时评价前后、左右和对角平衡；数值来自最终方案评分，不由 LLM 推测。"
    elif topic == "constraints":
        details = common[:3] + [
            {"label": "独立验证", "value": "PASS" if validation.get("is_valid") else "CHECK"},
            {"label": "约束违规", "value": int(validation.get("violation_count") or 0)},
            {"label": "超载 / 越界 / 重叠", "value": f"{int(validation.get('overweight_violation_count') or 0)} / {int(validation.get('boundary_violation_count') or 0)} / {int(validation.get('overlap_violation_count') or 0)}"},
        ]
        answer = "方案只有在产品完整性、重量、边界、重叠和尺寸检查全部通过后才能进入批准流程。"
    elif topic == "council":
        details = common[:3] + [
            {"label": "Council 选择", "value": council.get("selected_candidate") or "未启用"},
            {"label": "候选策略", "value": council.get("selected_strategy") or strategy.get("applied_strategy") or "—"},
            {"label": "次优候选", "value": council.get("runner_up") or "—"},
        ]
        answer = "Council 对通过 SafetyAgent 筛查的候选进行效率、平衡、运营和时间评价；这里展示的是实际选择记录。"
    elif topic == "strategy":
        details = common + [
            {"label": "请求策略", "value": strategy.get("requested_strategy") or "—"},
            {"label": "实际执行策略", "value": strategy.get("applied_strategy") or "—"},
            {"label": "目标函数值", "value": _value(score.get("objective_score"), 2)},
        ]
        answer = "LLM或规则可以提出策略建议，但实际执行策略经过有界策略检查，并由优化模块和独立验证器完成计算与把关。"
    else:
        details = common + [
            {"label": "重量利用率", "value": _value(score.get("average_weight_utilisation"))},
            {"label": "独立验证", "value": "PASS" if validation.get("is_valid") else "CHECK"},
            {"label": "实际策略", "value": strategy.get("applied_strategy") or "—"},
        ]
        answer = "这是当前数据范围内最新、且通过产物完整性检查的方案摘要。所有数值均直接读取持久化方案证据。"
    return answer, details


def _comparison_details(plans: list[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if len(plans) < 2:
        return "当前只有一个可验证的方案，尚不足以进行历史方案比较。", []
    current, previous = plans[0], plans[1]
    metrics = [
        ("产品数量", "total_products", "summary"),
        ("车皮数量", "wagon_count", "summary"),
        ("平均面积利用率", "average_area_utilisation", "score"),
        ("目标函数值", "objective_score", "score"),
        ("约束违规", "violation_count", "validation"),
    ]
    details: list[dict[str, Any]] = []
    for label, key, section in metrics:
        before = _value((previous.get(section) or {}).get(key))
        after = _value((current.get(section) or {}).get(key))
        details.append({"label": label, "before": before, "after": after, "delta": round(after - before, 6)})
    def short_id(value: Any) -> str:
        text = str(value or "—")
        return f"{text[:13]}…{text[-4:]}" if len(text) > 20 else text
    answer = f"已比较最新两个可验证方案：{short_id(previous.get('plan_id'))} v{previous.get('version')} → {short_id(current.get('plan_id'))} v{current.get('version')}。差异由后端直接计算。"
    return answer, details


def answer_analytics_question(
    question: str,
    dashboard: Mapping[str, Any],
    *,
    plan_evidence: list[Mapping[str, Any]] | None = None,
    previous_query_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(dashboard.get("_query_dispatches") or dashboard.get("recent_dispatches") or [])
    fallback = _validated_plan(
        deterministic_query_plan(question, rows, previous_query_plan),
        {
            "intent": "analytics", "topic": "overview", "group_by": "province",
            "metric": "dispatch_count", "sort": "desc", "limit": 10,
            "filters": {}, "chart_type": "bar",
        },
    )
    source = "deterministic_fallback"
    provider_metadata: dict[str, Any] = {}
    fallback_reason = ""
    try:
        plan, provider_metadata = llm_query_plan(question, rows, fallback)
        source = "llm_validated"
    except (LLMProviderError, ValueError, json.JSONDecodeError) as error:
        plan = fallback
        fallback_reason = getattr(error, "code", type(error).__name__)
    results = execute_query_plan(rows, plan) if plan["intent"] == "analytics" else []
    metric_label = METRIC_LABELS[plan["metric"]]
    group_label = GROUP_LABELS[plan["group_by"]]
    details: list[dict[str, Any]] = []
    evidence_plans = list(plan_evidence or [])
    if plan["intent"] == "plan_explanation":
        if evidence_plans:
            summary, details = _plan_details(evidence_plans[0], plan["topic"])
        else:
            summary = "当前没有通过完整性检查的方案可以解释。"
    elif plan["intent"] == "plan_comparison":
        summary, details = _comparison_details(evidence_plans)
    elif results:
        leader = results[0]
        unit = "%" if "utilisation" in plan["metric"] else {
            "weight_tonne": " 吨", "dispatch_count": " 节", "product_count": " 件"
        }.get(plan["metric"], "")
        display_value = f"{leader['value'] * 100:.1f}" if "utilisation" in plan["metric"] else f"{leader['value']:g}"
        geographic_region = plan["filters"].get("geographic_region")
        if geographic_region:
            total = sum(item["value"] for item in results) if plan["metric"] in {"dispatch_count", "product_count", "weight_tonne"} else None
            total_unit = {"weight_tonne": " 吨", "dispatch_count": " 节计划车皮", "product_count": " 件产品"}.get(plan["metric"], "")
            total_text = f"，合计 {total:g}{total_unit}" if total is not None else ""
            group_noun = "目的地城市" if plan["group_by"] == "destination" else "分组"
            summary = f"按版本化地理目录识别出 {len(results)} 个{REGION_LABELS[geographic_region]}{group_noun}{total_text}；其中{leader['group']}的{metric_label}为 {display_value}{unit}。"
        else:
            summary = f"在当前数据范围内，{leader['group']}的{metric_label}为 {display_value}{unit}。共返回 {len(results)} 个分组。"
    else:
        summary = "当前筛选条件下没有可用于计算的已验证计划车皮。"
    geographic_region = plan["filters"].get("geographic_region")
    geography_resolution = None
    if geographic_region:
        states = [classify_row(row, geographic_region) for row in rows]
        geography_resolution = {
            "catalog_version": CATALOG_VERSION,
            "region": geographic_region,
            "region_label": REGION_LABELS[geographic_region],
            "matched_record_count": sum(value is True for value in states),
            "excluded_record_count": sum(value is False for value in states),
            "unresolved_record_count": sum(value is None for value in states),
            "policy": "城市级目录优先；跨江或证据不足的目的地不进行推测。",
        }
    return {
        "answer": summary,
        "response_kind": plan["intent"],
        "query_plan": plan,
        "interpretation_source": source,
        "fallback_reason": fallback_reason or None,
        "provider": provider_metadata.get("provider"),
        "model": provider_metadata.get("model"),
        "columns": [
            {"key": "group", "label": group_label, "format": "text"},
            {"key": "value", "label": metric_label, "format": "percent" if "utilisation" in plan["metric"] else "number"},
            {"key": "record_count", "label": "参与计算的车皮记录", "format": "integer"},
        ],
        "rows": results,
        "chart": {"type": plan["chart_type"], "category_key": "group", "value_key": "value"},
        "details": details,
        "geography_resolution": geography_resolution,
        "evidence": {
            "data_source": dashboard.get("data_source"),
            "plan_count": dashboard.get("summary", {}).get("plan_count", 0),
            "dispatch_record_count": len(rows),
            "generated_at": dashboard.get("generated_at"),
            "scope_note": "仅统计通过完整性检查和独立验证的计划车皮；不代表现场已经完成发车。",
        },
    }
