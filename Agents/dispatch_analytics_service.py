"""Read-only operational analytics derived from persisted, validated plans."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROVINCES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("110000", "北京市", ("北京", "Beijing")),
    ("120000", "天津市", ("天津", "Tianjin")),
    ("130000", "河北省", ("河北", "石家庄", "唐山", "秦皇岛", "保定", "邯郸", "Hebei")),
    ("140000", "山西省", ("山西", "太原", "大同", "长治", "临汾", "Shanxi")),
    ("150000", "内蒙古自治区", ("内蒙古", "呼和浩特", "包头", "鄂尔多斯", "Inner Mongolia")),
    ("210000", "辽宁省", ("辽宁", "沈阳", "大连", "鞍山", "营口", "Liaoning")),
    ("220000", "吉林省", ("吉林", "长春", "四平", "Jilin")),
    ("230000", "黑龙江省", ("黑龙江", "哈尔滨", "大庆", "齐齐哈尔", "Heilongjiang")),
    ("310000", "上海市", ("上海", "Shanghai")),
    ("320000", "江苏省", ("江苏", "南京", "苏州", "无锡", "常州", "徐州", "南通", "Jiangsu")),
    ("330000", "浙江省", ("浙江", "杭州", "宁波", "温州", "嘉兴", "绍兴", "金华", "北仑", "Zhejiang")),
    ("340000", "安徽省", ("安徽", "合肥", "芜湖", "马鞍山", "蚌埠", "Anhui")),
    ("350000", "福建省", ("福建", "福州", "厦门", "泉州", "漳州", "Fujian")),
    ("360000", "江西省", ("江西", "南昌", "九江", "赣州", "Jiangxi")),
    ("370000", "山东省", ("山东", "济南", "青岛", "烟台", "潍坊", "临沂", "日照", "Shandong")),
    ("410000", "河南省", ("河南", "郑州", "洛阳", "安阳", "新乡", "Henan")),
    ("420000", "湖北省", ("湖北", "武汉", "黄石", "宜昌", "襄阳", "荆州", "Hubei")),
    ("430000", "湖南省", ("湖南", "长沙", "株洲", "湘潭", "衡阳", "Hunan")),
    ("440000", "广东省", ("广东", "广州", "深圳", "东莞", "常平", "佛山", "惠州", "珠海", "Guangdong")),
    ("450000", "广西壮族自治区", ("广西", "南宁", "柳州", "桂林", "防城港", "Guangxi")),
    ("460000", "海南省", ("海南", "海口", "三亚", "Hainan")),
    ("500000", "重庆市", ("重庆", "唐家沱", "Chongqing")),
    ("510000", "四川省", ("四川", "成都", "绵阳", "攀枝花", "宜宾", "Sichuan")),
    ("520000", "贵州省", ("贵州", "贵阳", "遵义", "六盘水", "Guizhou")),
    ("530000", "云南省", ("云南", "昆明", "王家营", "曲靖", "玉溪", "Yunnan")),
    ("540000", "西藏自治区", ("西藏", "拉萨", "Tibet", "Xizang")),
    ("610000", "陕西省", ("陕西", "西安", "宝鸡", "榆林", "Shaanxi")),
    ("620000", "甘肃省", ("甘肃", "兰州", "嘉峪关", "Gansu")),
    ("630000", "青海省", ("青海", "西宁", "Qinghai")),
    ("640000", "宁夏回族自治区", ("宁夏", "银川", "Ningxia")),
    ("650000", "新疆维吾尔自治区", ("新疆", "乌鲁木齐", "库尔勒", "Xinjiang")),
    ("710000", "台湾省", ("台湾", "台北", "高雄", "Taiwan")),
    ("810000", "香港特别行政区", ("香港", "Hong Kong")),
    ("820000", "澳门特别行政区", ("澳门", "Macao", "Macau")),
)


def resolve_china_province(destination: str) -> tuple[str, str] | None:
    """Return deterministic province code/name without inventing an assignment."""
    normalised = "".join(str(destination or "").split()).casefold()
    if not normalised:
        return None
    matches: list[tuple[int, str, str]] = []
    for code, name, aliases in PROVINCES:
        for alias in (name, *aliases):
            candidate = "".join(alias.split()).casefold()
            if candidate and candidate in normalised:
                matches.append((len(candidate), code, name))
    if not matches:
        return None
    _, code, name = max(matches)
    return code, name


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _empty_bucket(code: str, name: str) -> dict[str, Any]:
    return {
        "adcode": code,
        "province": name,
        "dispatch_count": 0,
        "product_count": 0,
        "weight_tonne": 0.0,
        "weight_utilisation_sum": 0.0,
        "area_utilisation_sum": 0.0,
        "plan_ids": set(),
        "destinations": set(),
    }


def _solution_path(plan: Mapping[str, Any]) -> Path:
    payload = plan.get("payload") or {}
    candidate = payload.get("solution_path") or Path(str(plan.get("run_dir", ""))) / "loading_solution.json"
    path = Path(str(candidate)).resolve()
    if path.name != "loading_solution.json":
        raise ValueError("Unexpected solution artifact name")
    return path


def _read_solution(plan: Mapping[str, Any]) -> dict[str, Any]:
    path = _solution_path(plan)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("wagons"), list):
        raise ValueError("Invalid loading solution structure")
    return value


def _finalise_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
    dispatches = int(bucket["dispatch_count"])
    return {
        "adcode": bucket.get("adcode"),
        "province": bucket.get("province"),
        "destination": bucket.get("destination"),
        "dispatch_count": dispatches,
        "product_count": int(bucket["product_count"]),
        "weight_tonne": round(_number(bucket["weight_tonne"]), 3),
        "average_weight_utilisation": round(_number(bucket["weight_utilisation_sum"]) / dispatches, 6) if dispatches else 0.0,
        "average_area_utilisation": round(_number(bucket["area_utilisation_sum"]) / dispatches, 6) if dispatches else 0.0,
        "plan_count": len(bucket["plan_ids"]),
        "destinations": sorted(bucket["destinations"]),
    }


def build_dispatch_dashboard(plans: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate real wagon assignments while retaining unmapped destinations."""
    province_buckets = {code: _empty_bucket(code, name) for code, name, _ in PROVINCES}
    unmapped: dict[str, dict[str, Any]] = {}
    trend: dict[str, dict[str, Any]] = defaultdict(lambda: {"dispatch_count": 0, "product_count": 0, "weight_tonne": 0.0})
    recent_dispatches: list[dict[str, Any]] = []
    included_plan_ids: set[str] = set()
    skipped_plans: list[dict[str, str]] = []

    for plan in plans:
        plan_id = str(plan.get("plan_id") or "")
        try:
            solution = _read_solution(plan)
            if not bool((solution.get("validation_summary") or {}).get("is_valid")):
                raise ValueError("Plan is not independently validated")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            skipped_plans.append({"plan_id": plan_id, "reason": str(error)})
            continue

        included_plan_ids.add(plan_id)
        timestamp = str(plan.get("updated_at") or plan.get("created_at") or "")
        day = timestamp[:10] if len(timestamp) >= 10 else "unknown"
        for wagon in solution.get("wagons") or []:
            products = wagon.get("products") or []
            destination = str(wagon.get("destination") or "").strip()
            if not destination:
                destinations = {str(item.get("destination") or "").strip() for item in products}
                destination = next(iter(destinations)) if len(destinations) == 1 else "未指定"
            weight = _number(wagon.get("current_weight_tonne"))
            if not weight and wagon.get("current_weight") is not None:
                weight = _number(wagon.get("current_weight")) / 1000.0
            capacity = _number(wagon.get("max_weight_tonne"))
            if not capacity and wagon.get("max_weight") is not None:
                capacity = _number(wagon.get("max_weight")) / 1000.0
            weight_utilisation = min(max(weight / capacity, 0.0), 1.0) if capacity else 0.0
            area_utilisation = min(max(_number(wagon.get("area_utilisation")), 0.0), 1.0)
            province = resolve_china_province(destination)
            if province:
                code, name = province
                bucket = province_buckets[code]
            else:
                bucket = unmapped.setdefault(
                    destination or "未指定",
                    {
                        **_empty_bucket("", ""),
                        "destination": destination or "未指定",
                    },
                )
                code, name = "", ""

            bucket["dispatch_count"] += 1
            bucket["product_count"] += len(products)
            bucket["weight_tonne"] += weight
            bucket["weight_utilisation_sum"] += weight_utilisation
            bucket["area_utilisation_sum"] += area_utilisation
            bucket["plan_ids"].add(plan_id)
            bucket["destinations"].add(destination or "未指定")

            trend[day]["dispatch_count"] += 1
            trend[day]["product_count"] += len(products)
            trend[day]["weight_tonne"] += weight
            recent_dispatches.append(
                {
                    "plan_id": plan_id,
                    "plan_version": int(plan.get("version") or 0),
                    "plan_status": str(plan.get("status") or "unknown"),
                    "updated_at": timestamp,
                    "date": day,
                    "wagon_id": str(wagon.get("wagon_id") or ""),
                    "destination": destination or "未指定",
                    "province_adcode": code or None,
                    "province": name or None,
                    "product_count": len(products),
                    "weight_tonne": round(weight, 3),
                    "weight_utilisation": round(weight_utilisation, 6),
                    "area_utilisation": round(area_utilisation, 6),
                }
            )

    provinces = [_finalise_bucket(province_buckets[code]) for code, _, _ in PROVINCES]
    unmapped_values = sorted(
        (_finalise_bucket(item) for item in unmapped.values()),
        key=lambda item: (-item["dispatch_count"], item["destination"]),
    )
    dispatch_count = sum(item["dispatch_count"] for item in provinces) + sum(item["dispatch_count"] for item in unmapped_values)
    product_count = sum(item["product_count"] for item in provinces) + sum(item["product_count"] for item in unmapped_values)
    weight_tonne = sum(item["weight_tonne"] for item in provinces) + sum(item["weight_tonne"] for item in unmapped_values)
    weight_utilisation_sum = sum(item["average_weight_utilisation"] * item["dispatch_count"] for item in provinces + unmapped_values)
    area_utilisation_sum = sum(item["average_area_utilisation"] * item["dispatch_count"] for item in provinces + unmapped_values)
    mapped_dispatch_count = sum(item["dispatch_count"] for item in provinces)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "persisted_plan_database_and_loading_solution_artifacts",
        "semantics": {
            "dispatch_count": "planned_wagon_assignments",
            "time_axis": "plan_updated_at",
            "execution_confirmation_included": False,
        },
        "summary": {
            "plan_count": len(included_plan_ids),
            "dispatch_count": dispatch_count,
            "product_count": product_count,
            "weight_tonne": round(weight_tonne, 3),
            "average_weight_utilisation": round(weight_utilisation_sum / dispatch_count, 6) if dispatch_count else 0.0,
            "average_area_utilisation": round(area_utilisation_sum / dispatch_count, 6) if dispatch_count else 0.0,
            "mapped_dispatch_count": mapped_dispatch_count,
            "unmapped_dispatch_count": dispatch_count - mapped_dispatch_count,
            "mapped_rate": round(mapped_dispatch_count / dispatch_count, 6) if dispatch_count else 0.0,
            "skipped_plan_count": len(skipped_plans),
        },
        "provinces": provinces,
        "unmapped_destinations": unmapped_values,
        "trend": [
            {"date": day, **{key: round(value, 3) if key == "weight_tonne" else value for key, value in values.items()}}
            for day, values in sorted(trend.items())
        ],
        "recent_dispatches": sorted(recent_dispatches, key=lambda item: item["updated_at"], reverse=True)[:50],
        # Internal full record set for the bounded analytics assistant.  The
        # public dashboard endpoint removes this field before serialisation.
        "_query_dispatches": sorted(recent_dispatches, key=lambda item: item["updated_at"], reverse=True),
        "skipped_plans": skipped_plans,
    }
