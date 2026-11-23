"""Validated external API request contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanningRunRequest(StrictRequest):
    products: list[dict[str, Any]] = Field(min_length=1)
    execution_mode: str = "multi_agent_council"
    scenario_id: str = Field(default="api_request", max_length=120)
    system_variant: str = Field(default="full_hybrid", max_length=120)
    time_budget_seconds: float = Field(default=12.0, gt=0, le=600)
    random_seed: int = 42
    generate_visuals: bool = False
    strategy_plan: dict[str, Any] | None = None
    data_intake_audit: dict[str, Any] | None = None
    semantic_planning_enabled: bool = False
    rag_top_k: int = Field(default=6, ge=1, le=20)
    llm_provider_name: str | None = Field(default=None, max_length=80)
    llm_model_name: str | None = Field(default=None, max_length=160)
    llm_base_url: str | None = Field(default=None, max_length=500)
    optional_enrichment_enabled: bool = False
    optional_enrichment_fields: list[
        Literal[
            "product_name",
            "receiver",
            "contract_no",
            "storage_location",
            "remarks",
            "product_summary",
        ]
    ] = Field(default_factory=lambda: ["product_summary"], max_length=6)
    optional_enrichment_max_products: int = Field(default=25, ge=1, le=100)
    source_input_path: str | None = Field(default=None, max_length=1000)
    source_input_name: str | None = Field(default=None, max_length=255)
    source_input_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    repaired_source_path: str | None = Field(default=None, max_length=1000)
    repair_audit_path: str | None = Field(default=None, max_length=1000)
    authoritative_master_path: str | None = Field(default=None, max_length=1000)


class PlanningRequestReviewRequest(StrictRequest):
    accepted_repair_ids: list[str] = Field(default_factory=list, max_length=2000)


class PlanningRequestSubmitRequest(StrictRequest):
    time_budget_seconds: float = Field(default=12.0, gt=0, le=600)
    destination_grouping_required: bool = False
    semantic_planning_enabled: bool = False
    rag_top_k: int = Field(default=6, ge=1, le=20)
    optional_enrichment_enabled: bool = False
    optional_enrichment_fields: list[
        Literal[
            "product_name",
            "receiver",
            "contract_no",
            "storage_location",
            "remarks",
            "product_summary",
        ]
    ] = Field(default_factory=lambda: ["product_summary"], max_length=6)
    optional_enrichment_max_products: int = Field(default=5, ge=1, le=100)
    generate_visuals: bool = True


class ApprovalRequest(StrictRequest):
    expected_version: int = Field(ge=1)
    approved_by: str | None = Field(default=None, min_length=1, max_length=160)


class PlanDispositionRequest(StrictRequest):
    expected_version: int = Field(ge=1)
    reason: str = Field(
        default="Operator rejected pending successor", min_length=3, max_length=500
    )
    rejected_by: str | None = Field(default=None, min_length=1, max_length=160)


class DynamicEventRequest(StrictRequest):
    expected_version: int = Field(ge=1)
    event_type: Literal[
        "Add products", "Cancel products", "Change destination", "Disable wagon"
    ]
    product_count: int = Field(default=1, ge=1, le=500)
    source_destination: str | None = Field(default=None, max_length=160)
    target_destination: str | None = Field(default=None, max_length=160)
    affected_product_ids: list[str] = Field(default_factory=list, max_length=500)
    new_products: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    new_products_request_id: str | None = Field(default=None, max_length=80)
    add_product_mode: Literal["provided", "template_simulation"] = "template_simulation"
    disabled_wagon_id: str | None = Field(default=None, max_length=160)
    event_id: str | None = Field(default=None, max_length=160)
    time_budget_seconds: float = Field(default=5.0, gt=0, le=600)
    generate_visuals: bool = False

    @model_validator(mode="after")
    def validate_event_parameters(self) -> "DynamicEventRequest":
        requires_target = self.event_type == "Change destination" or (
            self.event_type == "Add products"
            and self.add_product_mode == "template_simulation"
        )
        if requires_target:
            target_destination = (self.target_destination or "").strip()
            if not target_destination:
                raise ValueError("新增产品或变更目的地时，必须选择目标目的地。")
            self.target_destination = target_destination
        if self.event_type == "Change destination":
            source_destination = (self.source_destination or "").strip()
            if not source_destination:
                raise ValueError("变更目的地时，必须选择当前目的地。")
            self.source_destination = source_destination
            if self.source_destination == self.target_destination:
                raise ValueError("当前目的地和新目的地不能相同。")
            self.affected_product_ids = [
                str(item).strip()
                for item in self.affected_product_ids
                if str(item).strip()
            ]
            if (
                self.affected_product_ids
                and len(self.affected_product_ids) != self.product_count
            ):
                raise ValueError("所选材料数量必须与变更产品数量一致。")
        if self.event_type == "Add products":
            if self.add_product_mode == "provided":
                if not self.new_products and not self.new_products_request_id:
                    raise ValueError("真实新增产品模式必须提供已确认的数据请求。")
                if self.new_products and self.new_products_request_id:
                    raise ValueError("真实新增产品不能同时提交浏览器数据和后端快照。")
                if self.new_products:
                    self.product_count = len(self.new_products)
            elif self.new_products:
                raise ValueError("模拟新增模式不能同时提交真实产品数据。")
        if self.event_type == "Disable wagon":
            disabled_wagon_id = (self.disabled_wagon_id or "").strip()
            if not disabled_wagon_id:
                raise ValueError("车皮不可用时，必须选择车皮编号。")
            self.disabled_wagon_id = disabled_wagon_id
        if self.event_type == "Cancel products":
            self.affected_product_ids = [
                str(item).strip()
                for item in self.affected_product_ids
                if str(item).strip()
            ]
            if not self.affected_product_ids:
                raise ValueError("取消订单时必须明确选择需要取消的材料。")
            if len(self.affected_product_ids) != self.product_count:
                raise ValueError("所选取消材料数量必须与取消产品数量一致。")
        return self


class ScheduledDynamicEventRequest(DynamicEventRequest):
    """A durable dynamic event collected into a priority time window."""

    priority: Literal["auto", "P0", "P1", "P2", "P3"] = "auto"
    window_seconds: int | None = Field(default=None, ge=0, le=86_400)
    business_instruction: str = Field(default="", max_length=1000)
    enable_llm_priority_advice: bool = True


class ContinuousStreamStartRequest(StrictRequest):
    expected_version: int = Field(ge=1)
    min_interval_seconds: int = Field(default=35, ge=1, le=86_400)
    max_interval_seconds: int = Field(default=65, ge=1, le=86_400)
    max_products: int = Field(default=500, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_intervals(self) -> "ContinuousStreamStartRequest":
        if self.max_interval_seconds < self.min_interval_seconds:
            raise ValueError("最长接单间隔不能小于最短接单间隔。")
        return self


class EnrichmentConfirmationRequest(StrictRequest):
    expected_version: int = Field(ge=1)
    suggestion_ids: list[str] = Field(min_length=1, max_length=100)
    confirmed_by: str | None = Field(default=None, min_length=1, max_length=160)


class AnalyticsAssistantRequest(StrictRequest):
    question: str = Field(min_length=2, max_length=500)
    status: Literal["active", "pending_approval", "superseded", "rejected", "all"] = (
        "active"
    )
    previous_query_plan: dict[str, Any] | None = None


def openapi_document() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Steel Loading Backend API", "version": "1.0.0"},
        "paths": {
            "/v1/data-intakes": {
                "post": {
                    "summary": "Upload and inspect a business dataset",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {"schema": {"type": "object"}}
                        },
                    },
                    "responses": {"201": {"description": "Intake inspected"}},
                }
            },
            "/v2/planning-requests": {
                "post": {
                    "summary": "Upload and automatically analyse a planning dataset",
                    "description": "Simplified web entry point. Sheet discovery, deterministic mapping and bounded LLM fallback are orchestrated server-side.",
                    "responses": {"201": {"description": "Planning request analysed"}},
                }
            },
            "/v2/planning-requests/sample": {
                "post": {
                    "summary": "Create a planning request from labelled training data",
                    "responses": {"201": {"description": "Training request analysed"}},
                }
            },
            "/v2/planning-requests/{request_id}": {
                "get": {
                    "summary": "Read the authoritative planning request state",
                    "responses": {
                        "200": {"description": "Current state and next action"}
                    },
                }
            },
            "/v2/planning-requests/{request_id}/review": {
                "post": {
                    "summary": "Apply selected evidence-backed repairs and revalidate",
                    "responses": {
                        "200": {"description": "Updated state and next action"}
                    },
                }
            },
            "/v2/planning-requests/{request_id}/confirm": {
                "post": {
                    "summary": "Confirm the reviewed snapshot for formal planning",
                    "responses": {
                        "200": {"description": "Canonical products and audit evidence"}
                    },
                }
            },
            "/v2/planning-requests/{request_id}/submit": {
                "post": {
                    "summary": "Queue planning from a confirmed server-side snapshot",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": PlanningRequestSubmitRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {
                        "202": {"description": "Planning job queued or replayed"}
                    },
                }
            },
            "/v1/data-intakes/{intake_id}/confirm": {
                "post": {
                    "summary": "Confirm reviewed data repairs and create canonical products",
                    "responses": {"200": {"description": "Intake finalised"}},
                }
            },
            "/v1/data-intakes/{intake_id}/assist": {
                "post": {
                    "summary": "Run bounded LLM field mapping before operator review",
                    "responses": {
                        "200": {"description": "Canonical preview available"}
                    },
                }
            },
            "/v1/data-intakes/{intake_id}/preview": {
                "post": {
                    "summary": "Reinspect the working copy after repair selection",
                    "responses": {"200": {"description": "Updated intake preview"}},
                }
            },
            "/v1/data-intakes/sample": {
                "post": {
                    "summary": "Stage clearly labelled training data when enabled",
                    "responses": {"201": {"description": "Training intake staged"}},
                }
            },
            "/v1/planning-runs": {
                "post": {
                    "summary": "Queue a planning run",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": PlanningRunRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {"202": {"description": "Job queued"}},
                }
            },
            "/v1/jobs/{job_id}": {
                "get": {
                    "summary": "Read durable job state",
                    "responses": {"200": {"description": "Current job state"}},
                }
            },
            "/v1/plans": {
                "get": {
                    "summary": "List recent persisted plans",
                    "responses": {"200": {"description": "Compact plan history"}},
                }
            },
            "/v1/session": {
                "get": {
                    "summary": "Read the trusted gateway identity for this session",
                    "responses": {
                        "200": {"description": "Authenticated user and roles"},
                        "403": {"description": "Trusted production identity required"},
                    },
                }
            },
            "/v1/analytics/dispatch-dashboard": {
                "get": {
                    "summary": "Read province-level planned-dispatch analytics",
                    "description": "Aggregates persisted plan records and validated loading-solution artifacts. Counts represent planned wagon assignments, not confirmed physical departures. Unmapped destinations are reported explicitly and are never assigned by inference.",
                    "responses": {
                        "200": {
                            "description": "Planned-wagon KPIs, province metrics, plan-update trend and recent records"
                        }
                    },
                }
            },
            "/v1/analytics/assistant/query": {
                "post": {
                    "summary": "Run a bounded natural-language analytics query",
                    "description": "The LLM may map the question to an allow-listed aggregation plan. The server validates and executes the plan over verified planning evidence; arbitrary SQL is never accepted.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": AnalyticsAssistantRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {
                        "200": {"description": "Auditable table and chart data"}
                    },
                }
            },
            "/v1/plans/{plan_id}/result": {
                "get": {
                    "summary": "Read a verified browser-safe plan result",
                    "responses": {
                        "200": {"description": "Plan result and artifact links"}
                    },
                }
            },
            "/v1/plans/{plan_id}": {
                "get": {
                    "summary": "Read the persisted plan record",
                    "responses": {"200": {"description": "Plan record"}},
                }
            },
            "/v1/plans/{plan_id}/artifacts/{artifact_name}": {
                "get": {
                    "summary": "Download an allow-listed plan artifact",
                    "responses": {"200": {"description": "Artifact file"}},
                }
            },
            "/v1/plans/{plan_id}/approve": {
                "post": {
                    "summary": "Approve a validated plan",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": ApprovalRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {"200": {"description": "Plan activated"}},
                }
            },
            "/v1/plans/{plan_id}/reject": {
                "post": {
                    "summary": "Reject an unapproved successor while retaining evidence",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": PlanDispositionRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {"200": {"description": "Plan marked as rejected"}},
                }
            },
            "/v1/plans/{plan_id}/events": {
                "post": {
                    "summary": "Queue a dynamic event",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": DynamicEventRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {"202": {"description": "Event queued"}},
                }
            },
            "/v1/plans/{plan_id}/scheduled-events": {
                "post": {
                    "summary": "Persist a dynamic event in a priority time window",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": ScheduledDynamicEventRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {
                        "202": {"description": "Event persisted and scheduled"}
                    },
                }
            },
            "/v1/plans/{plan_id}/continuous-stream": {
                "post": {
                    "summary": "Start one durable development/UAT continuous order stream",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": ContinuousStreamStartRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {"202": {"description": "Stream started or replayed"}},
                }
            },
            "/v1/continuous-streams/active": {
                "get": {
                    "summary": "Read the active durable continuous stream",
                    "responses": {"200": {"description": "Active stream or null"}},
                }
            },
            "/v1/continuous-streams/{stream_id}": {
                "get": {
                    "summary": "Read one durable continuous stream",
                    "responses": {"200": {"description": "Stream state"}},
                }
            },
            "/v1/continuous-streams/{stream_id}/stop": {
                "post": {
                    "summary": "Request a safe stop after any in-flight window settles",
                    "responses": {"202": {"description": "Stop accepted"}},
                }
            },
            "/v1/scheduling/windows": {
                "get": {
                    "summary": "List durable scheduling windows",
                    "responses": {"200": {"description": "Scheduling windows"}},
                }
            },
            "/v1/scheduling/windows/{window_id}": {
                "get": {
                    "summary": "Get one durable scheduling window",
                    "responses": {"200": {"description": "Scheduling window state"}},
                }
            },
            "/v1/plans/{plan_id}/enrichment-confirmations": {
                "post": {
                    "summary": "Confirm non-operational enrichment suggestions",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": EnrichmentConfirmationRequest.model_json_schema()
                            }
                        },
                    },
                    "responses": {"200": {"description": "Suggestions confirmed"}},
                }
            },
        },
    }
