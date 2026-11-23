"""Stable presentation contract for the simplified web planning workflow."""

from __future__ import annotations

from typing import Any, Dict


def planning_request_view(
    payload: Dict[str, Any],
    *,
    analysis_error: str | None = None,
) -> Dict[str, Any]:
    """Expose one authoritative state and next action to thin web clients."""
    suggestions = list(payload.get("repair_analysis", {}).get("suggestions") or [])
    if analysis_error:
        status, next_action, progress = "BLOCKED", "REPLACE_FILE", 100
        message = analysis_error
    elif payload.get("requires_assistance"):
        status, next_action, progress = "ANALYSING", "WAIT", 60
        message = str(payload.get("message") or "正在识别业务数据。")
    elif payload.get("planning_ready"):
        status, next_action, progress = "READY_TO_PLAN", "CONFIRM_AND_PLAN", 100
        message = str(payload.get("message") or "数据可以进入正式规划。")
    elif suggestions:
        status, next_action, progress = "NEEDS_REVIEW", "REVIEW", 100
        message = str(payload.get("message") or "需要核对数据修复建议。")
    else:
        status, next_action, progress = "BLOCKED", "REPLACE_FILE", 100
        message = str(payload.get("message") or "当前文件无法安全进入规划。")
    return {
        **payload,
        "request_id": str(payload.get("intake_id") or ""),
        "status": status,
        "next_action": next_action,
        "progress": progress,
        "stage": "DATA_REVIEW" if status in {"READY_TO_PLAN", "NEEDS_REVIEW", "BLOCKED"} else "DATA_ANALYSIS",
        "message": message,
    }
