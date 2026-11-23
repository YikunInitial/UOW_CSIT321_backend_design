# Continuous Operations — implementation and handoff

## Scope and safety boundary

Continuous Operations is a development/UAT sandbox that demonstrates durable
order arrival and dynamic replanning against the real backend. It generates
small synthetic additions only when `STEEL_ALLOW_SIMULATION_EVENTS=true`.
Production configuration rejects stream creation and the worker will not
advance a previously persisted simulation stream.

This feature does not replace the ordinary production path for uploaded and
operator-confirmed orders. Production release remains human-authorised.

## Authoritative flow

1. The browser sends one start command for an active plan.
2. The API creates or returns the single active `continuous_streams` record.
3. A worker claims the stream with a short SQLite lease.
4. When `next_event_at` is due, it creates a deterministic event ID from the
   stream ID and cycle number and schedules a nonvisual event window.
5. The normal durable window dispatcher queues `window_replanning`.
6. BDI impact assessment selects bounded local recovery or safe global
   Multi-Agent Council replanning.
7. Safety and independent validation remain part of the normal planning path.
8. In UAT only, the backend controller approves the validated successor and
   advances the stream baseline/version.
9. The browser polls persisted state. Closing or refreshing it has no effect on
   backend execution.

## State model

- `running`: may receive the next event when due;
- `stopping`: no new event is accepted; an existing window is allowed to
  settle;
- `stopped`: terminal operator stop or configured product horizon reached;
- `failed`: a permanent replanning/controller failure requires review.

Only one `running` or `stopping` stream is allowed per deployment. Start and
stop are idempotent. The current window ID, plan ID/version, cycle count, next
arrival, last error and latest 40 compact events are persisted.

## Workload and storage bounds

- Browser history: 40 records.
- New products per simulated event: 1–3.
- Default UI horizon: current product count plus 250, never above
  `STEEL_API_MAX_PRODUCTS`.
- Intermediate continuous cycles: JSON/audit evidence only; layout PNG/PDF
  generation is disabled.
- Disk admission floor: 2 GB by default through
  `STEEL_MINIMUM_FREE_DISK_BYTES`.

The stream stops at its product horizon. It never silently drops real business
records to make room.

## Recovery semantics

- Lost start response: start returns the existing active stream.
- Lost scheduling response: the same deterministic event ID replays the same
  window.
- Worker restart: the lease expires and another worker continues from the
  persisted stream/window state.
- Browser/network loss: backend execution continues; the page resynchronises
  from the stream endpoint.
- Stop during replanning: state becomes `stopping`; the current cycle settles,
  then no further event is created.
- Permanent window failure: state becomes `failed`; no infinite retry loop.
- Transient job errors: handled by the existing bounded job retry/dead-letter
  mechanism before the stream sees a terminal window failure.

## Interfaces

- `POST /v1/plans/{plan_id}/continuous-stream`
- `GET /v1/continuous-streams/active`
- `GET /v1/continuous-streams/{stream_id}`
- `POST /v1/continuous-streams/{stream_id}/stop`

The web client is deliberately thin: it controls and observes these endpoints.
It does not schedule events, approve plans or own a long-running timer.

## Remaining assigned improvements

Before treating continuous receiving as a production service rather than a UAT
demonstrator, complete LR-07 through LR-14 in the readiness audit. The most
important are automated audited retention, rotating logs, separation of sandbox
analytics, and business-defined batching/rolling-horizon rules.

