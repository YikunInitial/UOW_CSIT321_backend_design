# Steel Loading Backend API v1

## Purpose

`backend_api.py` is the presentation-neutral integration boundary for client
ERP, MES, dashboard, or workflow systems. It does not expose Streamlit and it
does not allow an LLM to approve or execute a loading plan.

The API is asynchronous: planning and replanning requests return a job id.
Callers poll the job resource until it is `completed`, `failed`, or
`dead_letter`. Production write requests require an `Idempotency-Key`;
replaying the same key and body returns the original job, while reusing the key
for a different body returns HTTP 409.

Both the legacy Streamlit client and the deployable React web client use this
boundary. Neither client performs optimisation in the presentation layer.
Consequently, the same idempotency, queue limits, Worker isolation, approval
authority, validation, and audit trail apply to UI, ERP, and MES integrations.

## Start

```powershell
$env:STEEL_API_KEY = "replace-with-a-secret-managed-by-the-client"
$env:STEEL_API_EMBEDDED_WORKER = "false"
.\start_backend_api.ps1 -Port 8080 -DataRoot "D:\SteelLoadingData"
.\start_backend_worker.ps1 -DataRoot "D:\SteelLoadingData"
```

Run the API and at least one worker as separate supervised processes in a
production installation. For a developer workstation only, setting
`STEEL_API_EMBEDDED_WORKER=true` starts one worker thread inside the API
process so the UI remains easy to run.

Keep the process behind the client's authenticated reverse proxy or VPN. The
built-in `X-API-Key` check is a service credential, not a replacement for TLS,
user identity, role-based access control, or network controls.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process and repository health |
| `GET` | `/health/live` | Lightweight process liveness probe |
| `GET` | `/health/ready` | Repository, queue, and worker readiness |
| `GET` | `/metrics` | Queue, worker, plan, event, and recovery counters |
| `POST` | `/v1/data-intakes` | Upload and inspect Excel, CSV, or JSON business data |
| `POST` | `/v1/data-intakes/sample` | Stage clearly labelled training data when enabled |
| `POST` | `/v1/data-intakes/{intake_id}/preview` | Reinspect an intake after repair selections change |
| `POST` | `/v1/data-intakes/{intake_id}/confirm` | Apply operator-confirmed evidence repairs |
| `POST` | `/v2/planning-requests` | Upload and analyse a Web planning request |
| `GET` | `/v2/planning-requests/{request_id}` | Read its durable authoritative state |
| `POST` | `/v2/planning-requests/{request_id}/review` | Apply selected repairs and revalidate |
| `POST` | `/v2/planning-requests/{request_id}/confirm` | Freeze the reviewed server-side snapshot |
| `POST` | `/v2/planning-requests/{request_id}/submit` | Queue planning by request ID |
| `POST` | `/v1/planning-runs` | Submit products for planning |
| `GET` | `/v1/jobs/{job_id}` | Read asynchronous job status/result |
| `GET` | `/v1/plans` | List compact recent plan history |
| `GET` | `/v1/analytics/dispatch-dashboard` | Aggregate verified plan history into planned-wagon, utilisation, province, plan-update trend, and data-quality metrics |
| `GET` | `/v1/plans/{plan_id}` | Read durable plan metadata |
| `GET` | `/v1/plans/{plan_id}/result` | Read verified solution, audit, strategy, and artifact links |
| `GET` | `/v1/plans/{plan_id}/artifacts/{name}` | Download an allow-listed result artifact |
| `POST` | `/v1/plans/{plan_id}/approve` | Approve and activate a validated plan |
| `POST` | `/v1/plans/{plan_id}/reject` | Reject an unapproved successor without deleting evidence |
| `POST` | `/v1/plans/{plan_id}/events` | Submit a version-bound dynamic event |
| `POST` | `/v1/plans/{plan_id}/scheduled-events` | Persist an event in a priority time window |
| `GET` | `/v1/scheduling/windows` | List durable scheduling windows |
| `GET` | `/v1/scheduling/windows/{window_id}` | Read window, event, job, and result state |

Every non-health request must include `X-API-Key` when `STEEL_API_KEY` is set.
In production, the authenticated reverse proxy must also supply `X-User-Id`
and `X-User-Roles`. Confirmation, submission, rejection, enrichment confirmation
and dynamic events require an `operator` or `approver`; activation requires the
`approver` role. The trusted identity
is written to the manifest and audit record. The proxy must remove any
client-supplied copies of these headers before adding its authenticated values.
The gateway should preserve `Idempotency-Key` across client retries. When the
configured active-job ceiling is reached, the API returns HTTP 503 with a
`Retry-After` header instead of accepting an unbounded backlog.

### Analytics semantics

`GET /v1/analytics/dispatch-dashboard` accepts `status=active`,
`pending_approval`, `superseded`, `rejected`, or `all`. The default is
`active`. Its `dispatch_count` is the number of wagon assignments in validated
loading plans. It is not a physical departure count because this release does
not ingest ERP/TMS departure confirmations. Likewise, the trend time axis is
the persisted plan `updated_at` value. The response carries these definitions
in `semantics` so downstream Web or BI clients cannot silently reinterpret the
metric. The `all` scope can double-count historical versions and is intended
for audit analysis rather than current operational totals.

## Planning example

```json
{
  "products": [
    {
      "material_no": "P001",
      "destination": "Sydney",
      "thickness": 1.2,
      "width": 1000,
      "outer_diameter": 1600,
      "gross_weight_tonne": 20,
      "net_weight_tonne": 19.8
    }
  ],
  "execution_mode": "multi_agent_council",
  "time_budget_seconds": 12,
  "random_seed": 42,
  "generate_visuals": false
}
```

Successful planning produces a `pending_approval` plan. Technical validation
does not activate it automatically.

## Approval example

```json
{"expected_version": 1, "approved_by": "authenticated.user"}
```

Approval checks the optimistic version, changes the plan to `active`, updates
the solution and audit artifacts atomically, and refreshes the manifest hash.

## Dynamic event example

```json
{
  "expected_version": 1,
  "event_type": "Add products",
  "product_count": 2,
  "add_product_mode": "provided",
  "new_products": [
    {
      "material_no": "P002",
      "destination": "Melbourne",
      "thickness": 1.2,
      "width": 1000,
      "outer_diameter": 1500,
      "gross_weight_tonne": 18,
      "net_weight_tonne": 17.8
    },
    {
      "material_no": "P003",
      "destination": "Brisbane",
      "thickness": 1.1,
      "width": 1050,
      "outer_diameter": 1450,
      "gross_weight_tonne": 17,
      "net_weight_tonne": 16.8
    }
  ],
  "time_budget_seconds": 5
}
```

An event can target only an active plan and its expected version must match.
Event ids are unique in SQLite, so retrying the same event cannot execute it
twice. The API recovery path creates a validated successor plan using safe full
replanning. The successor again requires explicit approval; activating it
supersedes the parent plan.

For a destination change, submit both `source_destination` and
`target_destination`; `affected_product_ids` can identify the exact materials.
For an order cancellation, use `event_type: "Cancel products"` and provide the
exact `affected_product_ids`; cancellation never guesses which business records
to remove and cannot remove every active product in one event.
For an unavailable resource, submit `disabled_wagon_id`. For a real addition,
the confirmed product records are validated again by the Worker before
replanning. A pending successor can be declined without deleting evidence:

```json
{
  "expected_version": 2,
  "rejected_by": "authenticated.user",
  "reason": "Operational review retained the current plan"
}
```

### Priority time-window event

Use the scheduled endpoint when events should be collected before replanning:

```json
{
  "expected_version": 1,
  "event_id": "erp-event-2026-00031",
  "event_type": "Add products",
  "product_count": 2,
  "target_destination": "Melbourne",
  "add_product_mode": "template_simulation",
  "priority": "auto",
  "window_seconds": 30,
  "business_instruction": "Customer requests dispatch in the current shift"
}
```

The event is committed before the API responds. Safety floors are P0 for wagon
unavailability, P1 for destination changes, and P2 for normal additions. A
caller or configured LLM may escalate but cannot lower that floor. P0 always
closes immediately. One open window is maintained per active plan version; a
more urgent arrival promotes and shortens that window instead of creating a
competing successor.

At expiry, a Worker creates one priority job. The impact assessor chooses a
local replan for bounded changes, locks unaffected wagons, merges the affected
solution, and independently validates the complete result. Unsafe or
un-isolatable local results automatically escalate to the normal full
Multi-Agent Council path. Both paths produce a validated `pending_approval`
successor and retain the scope decision in the manifest.

### Durable continuous-order sandbox

Development/UAT may demonstrate long-running arrivals through the backend-owned
stream controller:

```text
POST /v1/plans/{plan_id}/continuous-stream
GET  /v1/continuous-streams/active
GET  /v1/continuous-streams/{stream_id}
POST /v1/continuous-streams/{stream_id}/stop
```

The start body supplies `expected_version`, `min_interval_seconds`,
`max_interval_seconds`, and a total `max_products` horizon. Only one stream can
be active. Its clock, in-flight window and plan version are persisted in
SQLite and processed under a worker lease; the browser only controls and polls
it. Intermediate cycles disable visual generation and use deterministic event
IDs, so a lost response or worker restart cannot duplicate an order. Stop first
prevents another arrival, then lets an in-flight window settle.

This interface is rejected when `STEEL_ALLOW_SIMULATION_EVENTS=false`, as it
must be in production. See `docs/CONTINUOUS_OPERATIONS_HANDOFF.md` for the full
state and recovery contract.

## Persistence

SQLite stores plan state, event identity/status, dynamic environments,
background jobs, worker heartbeats, and cross-storage operation journals. JSON
solutions, manifests, reports, and optional layouts stay in the configured
data root. Versioned forward migrations preserve existing records. Workers
claim jobs with expiring leases and heartbeats: an interrupted job is requeued
until its retry limit is exhausted, after which it enters the explicit
`dead_letter` state with durable attempt history. Only known transient failures
are retried; permanent failures stop immediately. SQLite uses transactions and
WAL mode for safe single-site concurrency. A future multi-site deployment can
replace this repository with PostgreSQL without changing the application
service contracts.
