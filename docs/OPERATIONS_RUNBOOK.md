# Steel Loading Service Operations Runbook

## Deployment topology

Production uses three independently supervised processes:

1. an authorised API client, if required;
2. the HTTP API (`backend_api.py`);
3. one or more durable job workers (`worker.py`).

The API and workers must share the same `STEEL_API_RUN_ROOT` and
`STEEL_API_STATE_DATABASE`. Set `STEEL_API_EMBEDDED_WORKER=false` in
production. Put the API behind the client's TLS reverse proxy or private API
gateway. The gateway owns user authentication and must replace, rather than
forward, the `X-User-Id` and `X-User-Roles` headers.

## Before first start

1. Create a dedicated service account and writable data directory.
2. Copy `.env.example` into the deployment secret/configuration system; do not
   commit the populated file.
3. Generate an API key of at least 32 random characters.
4. Install `requirements.lock` into a clean Python 3.11-3.13 environment.
5. Run `python -m compileall -q Agents backend_api.py worker.py` and the test
   suite before promoting the build.
6. Start the API and worker under the site's process supervisor and configure
   automatic restart.

Database schema migrations run forward automatically at repository startup.
Always take and verify a backup before installing a build whose schema version
is newer than the deployed version.

## Health and monitoring

- `/health/live`: API process is serving requests.
- `/health/ready`: state store is available and queued work has an active
  worker. HTTP 503 with `degraded` means intervention is required.
- `/metrics`: authenticated operational counters for jobs, workers, plans,
  events, and incomplete operations.
- `python operations_cli.py --data-root <path> status`: equivalent local
  status for an operator.

Alert on repeated failed/dead-letter jobs, a non-zero incomplete-operation
count, queued jobs with zero active workers, stale workers, degraded storage,
or sustained queue depth near `STEEL_MAX_ACTIVE_JOBS`. Logs are JSON by default
and contain job/request identifiers, not uploaded product payloads.

## Backup

Create a consistent online backup while the service is running:

```powershell
python operations_cli.py --data-root D:\SteelLoadingData backup `
  --output E:\SteelLoadingBackups\steel-loading-20260818.zip
python operations_cli.py verify-backup `
  --archive E:\SteelLoadingBackups\steel-loading-20260818.zip
```

The archive contains an online SQLite snapshot, run artifacts, SHA-256 hashes,
and a schema version. Schedule backups according to the client's recovery
point objective and copy them to storage outside the application host.

## Restore drill

Restore is deliberately allowed only into an empty directory:

```powershell
python operations_cli.py restore `
  --archive E:\SteelLoadingBackups\steel-loading-20260818.zip `
  --target D:\SteelLoadingRestoreTest `
  --confirm RESTORE
```

After restoration, run `status`, start one worker against the restored path,
and inspect at least one historical plan, manifest, audit report, and pending
job. Perform this drill periodically; a backup is not accepted merely because
the archive exists.

## Retention

Retention only targets artifact directories belonging to superseded plans,
older than the configured threshold, and strictly below the configured data
root. The database audit record is retained. Preview is the default:

```powershell
python operations_cli.py --data-root D:\SteelLoadingData prune `
  --retention-days 90
```

Review every candidate, take a verified backup, then add `--apply`. Active and
pending-approval plans are never candidates. Applied operations are appended
to `operations_audit.jsonl`.

## Incident recovery

- **API unavailable:** restart the API. Queued jobs remain in SQLite.
- **Worker stopped:** restart a worker. Expired or legacy null leases are
  requeued; jobs that exhaust their configured attempts enter `dead_letter`.
- **Dead-letter job:** inspect its durable attempt history and sanitised failure
  class, correct the underlying dependency/data fault, then submit a new
  request with a new idempotency key. Do not edit queue rows manually.
- **Queue saturated:** stop non-essential submission, confirm active workers and
  storage health, then retry according to `Retry-After`. Do not raise the queue
  ceiling until disk and recovery-time impact have been measured.
- **Approval interrupted:** worker startup replays any `preparing` approval
  journal entries and reconciles files with database state.
- **Database integrity failure:** stop API and workers, preserve the damaged
  directory for investigation, verify the latest backup, and restore it into a
  new empty data root.
- **LLM unavailable:** planning follows the configured deterministic fallback;
  review the manifest and audit report for the executed strategy and provider
  error before approval. Repeated transient failures open the circuit breaker
  temporarily so later requests fail fast into fallback rather than waiting on
  the same unavailable model.
- **Low disk/unwritable data root:** the API rejects new work and readiness is
  degraded while liveness remains available. Restore capacity or permissions,
  run the local `status`/storage checks, then resume submission.

Never edit SQLite, manifests, or audit JSON manually in production. Preserve
the request ID, job ID, plan ID, logs, and relevant backup when escalating an
incident.
