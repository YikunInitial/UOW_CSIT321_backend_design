import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from Agents.state_repository import (
    IdempotencyConflictError,
    QueueCapacityError,
    SQLiteStateRepository,
    StateConflictError,
)
from worker import JobWorker


def _sleeping_subprocess(_: str, __: int, ___: dict, ____: object) -> None:
    time.sleep(30)


def _request_hash(request: dict) -> str:
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_concurrent_idempotent_enqueue_creates_exactly_one_job(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    request = {"job_type": "planning", "products": [{"id": "A"}]}
    digest = _request_hash(request)

    def enqueue(index: int) -> tuple[str, bool]:
        return SQLiteStateRepository(database_path).enqueue_job(
            f"job-{index}",
            request,
            idempotency_key="same-client-request",
            endpoint="/v1/planning-runs",
            request_hash=digest,
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(enqueue, range(32)))

    assert len({job_id for job_id, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    assert SQLiteStateRepository(database_path).job_counts()["queued"] == 1


def test_idempotency_key_reuse_with_different_body_is_rejected(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.enqueue_job(
        "job-1",
        {"value": 1},
        idempotency_key="key-1",
        endpoint="/planning",
        request_hash="hash-1",
    )

    try:
        repository.enqueue_job(
            "job-2",
            {"value": 2},
            idempotency_key="key-1",
            endpoint="/planning",
            request_hash="hash-2",
        )
    except IdempotencyConflictError:
        pass
    else:
        raise AssertionError("idempotency key was accepted for a different body")


def test_queue_capacity_is_checked_atomically(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.enqueue_job("job-1", {"job_type": "test"}, max_active_jobs=1)

    try:
        repository.enqueue_job("job-2", {"job_type": "test"}, max_active_jobs=1)
    except QueueCapacityError:
        pass
    else:
        raise AssertionError("queue accepted work beyond its capacity")


def test_transient_failures_retry_then_complete_with_attempt_history(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "test"}, max_attempts=3)
    worker = JobWorker(
        repository,
        worker_id="worker-1",
        lease_seconds=30,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    class FlakyExecutor:
        calls = 0

        @classmethod
        def execute(cls, _: dict) -> dict:
            cls.calls += 1
            if cls.calls < 3:
                raise TimeoutError("temporary dependency timeout")
            return {"recovered": True}

    worker.executor = FlakyExecutor()

    assert worker.run_once()
    assert repository.get_job("job-1")["status"] == "queued"
    assert worker.run_once()
    assert repository.get_job("job-1")["status"] == "queued"
    assert worker.run_once()

    job = repository.get_job("job-1")
    assert job["status"] == "completed"
    assert job["result"] == {"recovered": True}
    attempts = repository.list_job_attempts("job-1")
    assert [attempt["outcome"] for attempt in attempts] == [
        "retry_scheduled",
        "retry_scheduled",
        "completed",
    ]


def test_poison_job_is_dead_lettered_without_blocking_following_work(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-poison", {"job_type": "poison"}, max_attempts=2)
    repository.create_job("job-healthy", {"job_type": "healthy"}, max_attempts=2)
    worker = JobWorker(
        repository,
        worker_id="worker-1",
        lease_seconds=30,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )

    class SelectiveExecutor:
        @staticmethod
        def execute(request: dict) -> dict:
            if request["job_type"] == "poison":
                raise TimeoutError("dependency remains unavailable")
            return {"healthy": True}

    worker.executor = SelectiveExecutor()

    assert worker.run_once()
    assert worker.run_once()
    assert worker.run_once()

    assert repository.get_job("job-poison")["status"] == "dead_letter"
    assert repository.get_job("job-healthy")["status"] == "completed"
    assert repository.job_counts()["dead_letter"] == 1


def test_job_completion_requires_current_lease_owner(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "test"})
    assert repository.claim_next_job("worker-a") is not None

    try:
        repository.complete_job("job-1", "worker-b", {"invalid": True})
    except StateConflictError:
        pass
    else:
        raise AssertionError("non-owner completed a leased job")

    assert repository.get_job("job-1")["status"] == "running"
    repository.complete_job("job-1", "worker-a", {"valid": True})
    assert repository.get_job("job-1")["status"] == "completed"


def test_short_database_write_lock_is_waited_out_without_losing_job(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    repository = SQLiteStateRepository(database_path)
    blocker = sqlite3.connect(database_path, timeout=30)
    blocker.execute("BEGIN IMMEDIATE")
    completed = threading.Event()
    errors: list[Exception] = []

    def enqueue() -> None:
        try:
            repository.create_job("job-1", {"job_type": "test"})
        except Exception as error:
            errors.append(error)
        finally:
            completed.set()

    thread = threading.Thread(target=enqueue)
    thread.start()
    time.sleep(0.2)
    assert not completed.is_set()
    blocker.rollback()
    blocker.close()
    thread.join(timeout=5)

    assert completed.is_set()
    assert not errors
    assert repository.get_job("job-1")["status"] == "queued"


def test_isolated_job_process_failure_does_not_terminate_worker(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "unsupported"})
    worker = JobWorker(
        repository,
        worker_id="isolated-worker",
        isolate_jobs=True,
        execution_timeout_seconds=10,
    )

    assert worker.run_once()

    job = repository.get_job("job-1")
    assert job["status"] == "failed"
    assert job["failure_class"] == "permanent"
    assert repository.worker_status()["active"] == 1


def test_isolated_job_timeout_is_terminated_and_dead_lettered(tmp_path: Path) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "hung"}, max_attempts=1)
    worker = JobWorker(
        repository,
        worker_id="timeout-worker",
        isolate_jobs=True,
        execution_timeout_seconds=1,
        retry_base_seconds=0,
        retry_max_seconds=0,
        process_target=_sleeping_subprocess,
    )

    started = time.perf_counter()
    assert worker.run_once()
    elapsed = time.perf_counter() - started

    assert elapsed < 8
    job = repository.get_job("job-1")
    assert job["status"] == "dead_letter"
    assert job["failure_class"] == "transient"


def test_shutdown_request_keeps_current_job_lease_alive_until_drain(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "slow"})
    worker = JobWorker(repository, worker_id="worker-1", lease_seconds=1.2)
    entered = threading.Event()
    release = threading.Event()

    class BlockingExecutor:
        @staticmethod
        def execute(_: dict) -> dict:
            entered.set()
            assert release.wait(timeout=10)
            return {"drained": True}

    worker.executor = BlockingExecutor()
    thread = threading.Thread(target=worker.run_once)
    thread.start()
    assert entered.wait(timeout=5)
    worker.stop()
    time.sleep(1.5)

    assert repository.get_job("job-1")["status"] == "running"
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert repository.get_job("job-1")["status"] == "completed"


def test_live_peer_periodically_recovers_crashed_workers_expired_lease(
    tmp_path: Path,
) -> None:
    repository = SQLiteStateRepository(tmp_path / "state.db")
    repository.create_job("job-1", {"job_type": "recoverable"}, max_attempts=3)
    assert repository.claim_next_job("crashed-worker", lease_seconds=0.2) is not None

    survivor = JobWorker(
        SQLiteStateRepository(tmp_path / "state.db"),
        worker_id="surviving-worker",
        lease_seconds=0.5,
        poll_seconds=0.02,
    )

    class RecoveryExecutor:
        @staticmethod
        def execute(_: dict) -> dict:
            return {"recovered_by_live_peer": True}

    survivor.executor = RecoveryExecutor()
    thread = threading.Thread(target=survivor.run_forever)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if repository.get_job("job-1")["status"] == "completed":
            break
        time.sleep(0.05)
    survivor.stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    job = repository.get_job("job-1")
    assert job["status"] == "completed"
    assert job["result"] == {"recovered_by_live_peer": True}
    assert job["attempt_count"] == 2
    assert [item["outcome"] for item in repository.list_job_attempts("job-1")] == [
        "lease_expired",
        "completed",
    ]


def test_multi_worker_continuous_run_has_no_lost_or_duplicate_completions(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state.db"
    producer = SQLiteStateRepository(database_path)
    total_jobs = 200
    for index in range(total_jobs):
        producer.create_job(f"job-{index:03d}", {"job_type": "test", "index": index})

    class EchoExecutor:
        @staticmethod
        def execute(request: dict) -> dict:
            return {"index": request["index"]}

    workers = []
    for index in range(4):
        worker = JobWorker(
            SQLiteStateRepository(database_path),
            worker_id=f"worker-{index}",
            lease_seconds=30,
        )
        worker.executor = EchoExecutor()
        workers.append(worker)

    def drain(worker: JobWorker) -> None:
        while worker.run_once():
            pass

    threads = [threading.Thread(target=drain, args=(worker,)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    counts = producer.job_counts()
    assert counts["completed"] == total_jobs
    assert counts["queued"] == 0
    assert counts["running"] == 0
    observed = {
        producer.get_job(f"job-{index:03d}")["result"]["index"]
        for index in range(total_jobs)
    }
    assert observed == set(range(total_jobs))
    assert all(
        len(producer.list_job_attempts(f"job-{index:03d}")) == 1
        for index in range(total_jobs)
    )
