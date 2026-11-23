"""Independent durable worker for steel-loading optimisation jobs."""

from __future__ import annotations

import argparse
import hashlib
import logging
import multiprocessing
import os
import platform
import signal
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Optional
from uuid import uuid4

from Agents.job_service import (
    JobExecutor,
    PermanentJobError,
    RetryableJobError,
    is_retryable_job_error,
    job_error_payload,
)
from Agents.continuous_stream_service import ContinuousStreamController
from Agents.observability import configure_logging
from Agents.state_repository import SQLiteStateRepository
from Agents.storage_health import check_storage_health

LOGGER = logging.getLogger("steel.worker")


def _execute_job_subprocess(
    database_path: str,
    max_products: int,
    request: dict,
    result_queue: multiprocessing.Queue,
) -> None:
    """Process boundary that prevents native/optimizer faults killing a worker."""
    try:
        repository = SQLiteStateRepository(database_path)
        result = JobExecutor(repository, max_products=max_products).execute(request)
    except BaseException as error:
        result_queue.put(
            {
                "ok": False,
                "retryable": is_retryable_job_error(error),
                "error": job_error_payload(error),
            }
        )
    else:
        result_queue.put({"ok": True, "result": result})


class JobWorker:
    def __init__(
        self,
        repository: SQLiteStateRepository,
        *,
        worker_id: Optional[str] = None,
        lease_seconds: float = 90.0,
        poll_seconds: float = 1.0,
        max_products: int = 2000,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 60.0,
        isolate_jobs: bool = False,
        execution_timeout_seconds: float = 600.0,
        process_target: Callable[[str, int, dict, Any], None] = _execute_job_subprocess,
        minimum_free_disk_bytes: int = 0,
        storage_check_interval_seconds: float = 10.0,
        enable_continuous_stream: Optional[bool] = None,
    ) -> None:
        self.repository = repository
        self.worker_id = worker_id or (
            f"{platform.node()}-{os.getpid()}-{uuid4().hex[:8]}"
        )
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds
        self.max_products = max_products
        self.executor = JobExecutor(repository, max_products=max_products)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.retry_max_seconds = max(self.retry_base_seconds, retry_max_seconds)
        self.isolate_jobs = isolate_jobs
        self.execution_timeout_seconds = max(1.0, execution_timeout_seconds)
        self.process_target = process_target
        self.minimum_free_disk_bytes = max(0, minimum_free_disk_bytes)
        self.storage_check_interval_seconds = max(1.0, storage_check_interval_seconds)
        self.stop_event = threading.Event()
        if enable_continuous_stream is None:
            environment = os.getenv("STEEL_ENV", "development").strip().lower()
            configured = os.getenv("STEEL_ALLOW_SIMULATION_EVENTS")
            simulation_enabled = (
                configured.strip().lower() in {"1", "true", "yes", "on"}
                if configured is not None
                else environment != "production"
            )
            enable_continuous_stream = environment != "production" and simulation_enabled
        self.stream_controller = (
            ContinuousStreamController(
                repository,
                worker_id=self.worker_id,
                lease_seconds=min(self.lease_seconds, 30.0),
            )
            if enable_continuous_stream
            else None
        )

    def _execute(self, request: dict) -> dict:
        # Reject malformed queue records before paying the cost of a spawned
        # optimiser process. This is deterministic, keeps permanent failures
        # out of the retry queue and avoids classifying process-start latency as
        # a transient planning timeout.
        job_type = request.get("job_type")
        if (
            self.isolate_jobs
            and self.process_target is _execute_job_subprocess
            and job_type not in {"planning", "replanning", "window_replanning"}
        ):
            raise PermanentJobError(f"Unsupported job_type: {job_type!r}")
        if not self.isolate_jobs:
            return self.executor.execute(request)
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=self.process_target,
            args=(
                str(self.repository.database_path),
                self.max_products,
                request,
                result_queue,
            ),
            name=f"steel-job-{request.get('job_type', 'unknown')}",
            daemon=False,
        )
        process.start()
        process.join(timeout=self.execution_timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5)
            raise RetryableJobError(
                f"Job exceeded the {self.execution_timeout_seconds:.1f}s execution limit."
            )
        try:
            envelope = result_queue.get(timeout=1.0)
        except Empty as error:
            raise RetryableJobError(
                f"Job process exited without a result (exit code {process.exitcode})."
            ) from error
        finally:
            result_queue.close()
            result_queue.join_thread()
        if envelope["ok"]:
            return envelope["result"]
        message = str(envelope["error"].get("message", "Job process failed."))
        if envelope["retryable"]:
            raise RetryableJobError(message)
        raise PermanentJobError(message)

    def _retry_delay(self, job_id: str, attempt: int) -> float:
        raw = min(
            self.retry_base_seconds * (2 ** max(0, attempt - 1)),
            self.retry_max_seconds,
        )
        digest = hashlib.sha256(f"{job_id}:{attempt}".encode()).digest()[0]
        jitter = 0.8 + (digest / 255.0) * 0.4
        return raw * jitter

    def _safe_worker_heartbeat(
        self, *, status: str, current_job_id: Optional[str] = None
    ) -> None:
        try:
            self.repository.heartbeat_worker(
                self.worker_id, status=status, current_job_id=current_job_id
            )
        except Exception:
            LOGGER.exception(
                "worker_heartbeat_failed",
                extra={"event": "worker_heartbeat_failed", "worker_id": self.worker_id},
            )

    def _heartbeat_loop(self, job_id: str, job_done: threading.Event) -> None:
        interval = max(1.0, self.lease_seconds / 3.0)
        while not job_done.wait(interval):
            try:
                if not self.repository.heartbeat_job(
                    job_id, self.worker_id, lease_seconds=self.lease_seconds
                ):
                    return
                self._safe_worker_heartbeat(status="busy", current_job_id=job_id)
            except Exception:
                LOGGER.exception(
                    "job_heartbeat_failed",
                    extra={
                        "event": "job_heartbeat_failed",
                        "job_id": job_id,
                        "worker_id": self.worker_id,
                    },
                )
                return

    def run_once(self) -> bool:
        stream_worked = self.stream_controller.tick() if self.stream_controller else False
        # Due windows are persisted before they become jobs. Any healthy worker
        # may perform this idempotent dispatch, so restarts do not lose timers.
        self.repository.dispatch_due_windows()
        job = self.repository.claim_next_job(
            self.worker_id, lease_seconds=self.lease_seconds
        )
        if job is None:
            self._safe_worker_heartbeat(status="idle")
            return stream_worked
        job_id = str(job["job_id"])
        LOGGER.info(
            "job_started",
            extra={
                "event": "job_started",
                "job_id": job_id,
                "worker_id": self.worker_id,
                "attempt": job["attempt_count"],
                "job_type": job["request"].get("job_type"),
            },
        )
        self._safe_worker_heartbeat(status="busy", current_job_id=job_id)
        job_done = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, job_done),
            name=f"heartbeat-{job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            result = self._execute(job["request"])
        except Exception as error:
            error_payload = job_error_payload(error)
            if is_retryable_job_error(error):
                event_id = job["request"].get("event_id")
                if job["request"].get("job_type") == "replanning" and event_id:
                    try:
                        self.repository.fail_event(str(event_id), error_payload)
                    except Exception:
                        LOGGER.exception(
                            "event_failure_record_failed",
                            extra={
                                "event": "event_failure_record_failed",
                                "event_id": event_id,
                                "job_id": job_id,
                            },
                        )
                delay = self._retry_delay(job_id, int(job["attempt_count"]))
                status = self.repository.retry_job(
                    job_id,
                    self.worker_id,
                    error_payload,
                    delay_seconds=delay,
                )
                LOGGER.warning(
                    "job_retry_scheduled"
                    if status == "queued"
                    else "job_dead_lettered",
                    extra={
                        "event": status,
                        "job_id": job_id,
                        "worker_id": self.worker_id,
                        "retry_delay_seconds": delay,
                    },
                    exc_info=True,
                )
            else:
                self.repository.fail_job(job_id, self.worker_id, error_payload)
                LOGGER.exception(
                    "job_failed",
                    extra={
                        "event": "job_failed",
                        "job_id": job_id,
                        "worker_id": self.worker_id,
                    },
                )
        else:
            self.repository.complete_job(job_id, self.worker_id, result)
            LOGGER.info(
                "job_completed",
                extra={
                    "event": "job_completed",
                    "job_id": job_id,
                    "worker_id": self.worker_id,
                },
            )
        finally:
            job_done.set()
            heartbeat.join(timeout=1.0)
            self._safe_worker_heartbeat(status="idle")
        return True

    def run_forever(self) -> None:
        self.repository.recover_expired_jobs()
        self.repository.recover_incomplete_operations()
        self.repository.recover_orphaned_events()
        self._safe_worker_heartbeat(status="idle")
        consecutive_failures = 0
        next_storage_check = 0.0
        next_recovery_check = 0.0
        storage_healthy = True
        while not self.stop_event.is_set():
            if time.monotonic() >= next_storage_check:
                storage = check_storage_health(
                    self.repository.database_path.parent,
                    self.repository.database_path,
                    minimum_free_bytes=self.minimum_free_disk_bytes,
                )
                storage_healthy = bool(storage["healthy"])
                next_storage_check = (
                    time.monotonic() + self.storage_check_interval_seconds
                )
                if not storage_healthy:
                    LOGGER.error(
                        "worker_storage_degraded",
                        extra={
                            "event": "worker_storage_degraded",
                            "worker_id": self.worker_id,
                            "storage": storage,
                        },
                    )
            if not storage_healthy:
                self.stop_event.wait(min(self.poll_seconds, 5.0))
                continue
            if time.monotonic() >= next_recovery_check:
                try:
                    lease_recovery = self.repository.recover_expired_jobs()
                    event_recovery = self.repository.recover_orphaned_events()
                    if any(lease_recovery.values()) or any(event_recovery.values()):
                        LOGGER.warning(
                            "abandoned_work_recovered",
                            extra={
                                "event": "abandoned_work_recovered",
                                "worker_id": self.worker_id,
                                "jobs": lease_recovery,
                                "events": event_recovery,
                            },
                        )
                except Exception:
                    LOGGER.exception(
                        "periodic_recovery_failed",
                        extra={
                            "event": "periodic_recovery_failed",
                            "worker_id": self.worker_id,
                        },
                    )
                next_recovery_check = time.monotonic() + max(
                    1.0, min(self.lease_seconds / 2.0, 10.0)
                )
            try:
                worked = self.run_once()
            except Exception:
                consecutive_failures += 1
                LOGGER.exception(
                    "worker_loop_error",
                    extra={
                        "event": "worker_loop_error",
                        "worker_id": self.worker_id,
                        "consecutive_failures": consecutive_failures,
                    },
                )
                self.stop_event.wait(
                    min(self.poll_seconds * (2**consecutive_failures), 30)
                )
            else:
                consecutive_failures = 0
                if not worked:
                    self.stop_event.wait(self.poll_seconds)
        self._safe_worker_heartbeat(status="stopped")

    def stop(self) -> None:
        self.stop_event.set()


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=os.getenv("STEEL_API_STATE_DATABASE", "runtime/api/state.db"),
    )
    parser.add_argument("--lease-seconds", type=float, default=90.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    worker = JobWorker(
        SQLiteStateRepository(Path(args.database)),
        lease_seconds=args.lease_seconds,
        poll_seconds=args.poll_seconds,
        max_products=int(os.getenv("STEEL_API_MAX_PRODUCTS", "2000")),
        retry_base_seconds=float(os.getenv("STEEL_JOB_RETRY_BASE_SECONDS", "2")),
        retry_max_seconds=float(os.getenv("STEEL_JOB_RETRY_MAX_SECONDS", "60")),
        isolate_jobs=os.getenv("STEEL_JOB_ISOLATION", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        execution_timeout_seconds=float(
            os.getenv("STEEL_JOB_EXECUTION_TIMEOUT_SECONDS", "600")
        ),
        minimum_free_disk_bytes=int(
            os.getenv("STEEL_MINIMUM_FREE_DISK_BYTES", "2147483648")
        ),
        storage_check_interval_seconds=float(
            os.getenv("STEEL_STORAGE_CHECK_INTERVAL_SECONDS", "10")
        ),
    )

    def request_stop(*_: object) -> None:
        worker.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker.run_forever()


if __name__ == "__main__":
    main()
