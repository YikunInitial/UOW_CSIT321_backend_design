"""Stable domain contracts for the research-system-v2.0.0 workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


SYSTEM_VERSION = "research-system-v2.0.0"
CONTRACT_VERSION = "1.0"
CONSTRAINT_SET_VERSION = "steel-loading-hard-constraints-v1"
OBJECTIVE_SET_VERSION = "steel-loading-objective-v1"
VALIDATOR_VERSION = "steel-loading-validator-v1"
STRATEGY_POLICY_VERSION = "steel-loading-strategy-policy-v1"
BDI_POLICY_VERSION = "steel-loading-bdi-shadow-v2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlanStatus(str, Enum):
    DRAFT = "draft"
    GENERATED = "generated"
    VALIDATED = "validated"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class EventStatus(str, Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    STALE = "stale"
    MANUAL_REVIEW = "manual_review"


class ErrorCode(str, Enum):
    DATA_INVALID = "DATA_INVALID"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    OPTIMISER_TIMEOUT = "OPTIMISER_TIMEOUT"
    CANDIDATE_TIMEOUT = "CANDIDATE_TIMEOUT"
    COUNCIL_DEADLINE_EXCEEDED = "COUNCIL_DEADLINE_EXCEEDED"
    ALL_CANDIDATES_VETOED = "ALL_CANDIDATES_VETOED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_SCHEMA_INVALID = "LLM_SCHEMA_INVALID"
    RAG_UNAVAILABLE = "RAG_UNAVAILABLE"
    STALE_PLAN_VERSION = "STALE_PLAN_VERSION"
    DUPLICATE_EVENT = "DUPLICATE_EVENT"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class VersionInfo:
    system: str = SYSTEM_VERSION
    contract: str = CONTRACT_VERSION
    constraint_set: str = CONSTRAINT_SET_VERSION
    objective_set: str = OBJECTIVE_SET_VERSION
    validator: str = VALIDATOR_VERSION
    strategy_policy: str = STRATEGY_POLICY_VERSION
    bdi_policy: str = BDI_POLICY_VERSION

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class StrategyDecisionTrace:
    """Records every authority boundary between a request and execution."""

    requested_strategy: Optional[str] = None
    recommended_strategy: Optional[str] = None
    authorised_strategy: Optional[str] = None
    executed_strategy: Optional[str] = None
    fallback_strategy: Optional[str] = None
    recommendation_source: Optional[str] = None
    authorisation_source: Optional[str] = None
    execution_source: Optional[str] = None
    override_reasons: List[str] = field(default_factory=list)
    fallback_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredError:
    code: ErrorCode
    message: str
    retryable: bool = False
    requires_manual_review: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["code"] = self.code.value
        return payload


class AgenticSystemError(RuntimeError):
    """Exception carrying a stable machine-readable failure contract."""

    def __init__(self, error: StructuredError) -> None:
        super().__init__(error.message)
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return self.error.to_dict()


@dataclass
class PlanVersion:
    plan_id: str
    version: int = 1
    status: PlanStatus = PlanStatus.DRAFT
    parent_plan_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)

    def transition(self, new_status: PlanStatus) -> None:
        allowed = {
            PlanStatus.DRAFT: {PlanStatus.GENERATED, PlanStatus.REJECTED},
            PlanStatus.GENERATED: {
                PlanStatus.VALIDATED,
                PlanStatus.REJECTED,
                PlanStatus.MANUAL_REVIEW,
            },
            PlanStatus.VALIDATED: {
                PlanStatus.PENDING_APPROVAL,
                PlanStatus.APPROVED,
                PlanStatus.REJECTED,
            },
            PlanStatus.PENDING_APPROVAL: {
                PlanStatus.APPROVED,
                PlanStatus.REJECTED,
            },
            PlanStatus.APPROVED: {PlanStatus.ACTIVE, PlanStatus.REJECTED},
            PlanStatus.ACTIVE: {PlanStatus.SUPERSEDED},
            PlanStatus.MANUAL_REVIEW: {
                PlanStatus.APPROVED,
                PlanStatus.REJECTED,
            },
            PlanStatus.REJECTED: set(),
            PlanStatus.SUPERSEDED: set(),
        }
        if new_status not in allowed[self.status]:
            raise ValueError(
                f"Invalid plan transition: {self.status.value} -> "
                f"{new_status.value}"
            )
        self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class EventVersion:
    event_id: str
    expected_plan_id: Optional[str] = None
    expected_plan_version: Optional[int] = None
    status: EventStatus = EventStatus.RECEIVED
    received_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
