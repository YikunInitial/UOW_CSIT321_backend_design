"""Versioned, deterministic geography catalogue for operational analytics.

The catalogue deliberately returns ``None`` where province-level evidence is
not precise enough (for example provinces crossed by the Yangtze).  This keeps
the analytics assistant from inventing a city-side classification.
"""

from __future__ import annotations

from typing import Any, Mapping

CATALOG_VERSION = "china-operational-geography-v1"

REGION_LABELS = {
    "north_of_yangtze": "长江以北",
    "south_of_yangtze": "长江以南",
    "yangtze_boundary": "长江沿线或跨江城市",
    "north_china": "华北",
    "northeast": "东北",
    "east_china": "华东",
    "central_china": "华中",
    "south_china": "华南",
    "southwest": "西南",
    "northwest": "西北",
    "coastal": "沿海地区",
}

REGION_PROVINCES = {
    "north_china": {"北京市", "天津市", "河北省", "山西省", "内蒙古自治区"},
    "northeast": {"辽宁省", "吉林省", "黑龙江省"},
    "east_china": {"上海市", "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省"},
    "central_china": {"河南省", "湖北省", "湖南省"},
    "south_china": {"广东省", "广西壮族自治区", "海南省", "香港特别行政区", "澳门特别行政区"},
    "southwest": {"重庆市", "四川省", "贵州省", "云南省", "西藏自治区"},
    "northwest": {"陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"},
    "coastal": {"辽宁省", "天津市", "河北省", "山东省", "江苏省", "上海市", "浙江省", "福建省", "广东省", "广西壮族自治区", "海南省", "香港特别行政区", "澳门特别行政区", "台湾省"},
}

NORTH_PROVINCES = {
    "北京市", "天津市", "河北省", "山西省", "内蒙古自治区", "辽宁省", "吉林省",
    "黑龙江省", "山东省", "河南省", "陕西省", "甘肃省", "宁夏回族自治区", "新疆维吾尔自治区",
}
SOUTH_PROVINCES = {
    "浙江省", "福建省", "江西省", "湖南省", "广东省", "广西壮族自治区", "海南省",
    "贵州省", "云南省", "香港特别行政区", "澳门特别行政区", "台湾省",
}

# City-level entries resolve destinations in provinces crossed by, bordering,
# or sourced by the Yangtze.  Values describe the city centre / named station,
# not every administrative-area point.
YANGTZE_CITY_SIDE = {
    "南京": "south_of_yangtze", "苏州": "south_of_yangtze", "无锡": "south_of_yangtze",
    "常州": "south_of_yangtze", "南通": "north_of_yangtze", "徐州": "north_of_yangtze",
    "合肥": "north_of_yangtze", "蚌埠": "north_of_yangtze", "芜湖": "south_of_yangtze",
    "马鞍山": "south_of_yangtze", "成都": "north_of_yangtze", "绵阳": "north_of_yangtze",
    "攀枝花": "south_of_yangtze", "宜宾": "yangtze_boundary", "武汉": "yangtze_boundary",
    "重庆": "yangtze_boundary", "唐家沱": "yangtze_boundary", "上海": "yangtze_boundary",
    "西宁": "north_of_yangtze", "拉萨": "south_of_yangtze",
}


def geography_region_from_text(text: str) -> str | None:
    compact = "".join(str(text or "").split()).casefold()
    aliases = (
        ("north_of_yangtze", ("长江以北", "长江北部", "江北城市", "江北地区")),
        ("south_of_yangtze", ("长江以南", "长江南部", "江南城市", "江南地区")),
        ("yangtze_boundary", ("长江沿线", "跨江城市", "长江两岸")),
        ("north_china", ("华北",)), ("northeast", ("东北",)),
        ("east_china", ("华东",)), ("central_china", ("华中",)),
        ("south_china", ("华南",)), ("southwest", ("西南",)),
        ("northwest", ("西北",)), ("coastal", ("沿海", "海岸省份")),
    )
    return next((key for key, values in aliases if any(value in compact for value in values)), None)


def classify_row(row: Mapping[str, Any], region: str) -> bool | None:
    province = str(row.get("province") or "").strip()
    destination = "".join(str(row.get("destination") or "").split())
    if region in REGION_PROVINCES:
        return province in REGION_PROVINCES[region] if province else None
    city_side = next((side for alias, side in sorted(YANGTZE_CITY_SIDE.items(), key=lambda item: len(item[0]), reverse=True) if alias in destination), None)
    if region == "yangtze_boundary":
        return city_side == "yangtze_boundary" if city_side else None
    if region not in {"north_of_yangtze", "south_of_yangtze"}:
        return None
    if city_side:
        return city_side == region
    if province in NORTH_PROVINCES:
        return region == "north_of_yangtze"
    if province in SOUTH_PROVINCES:
        return region == "south_of_yangtze"
    return None
