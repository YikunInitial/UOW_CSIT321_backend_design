from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from Agents.core.batch_task import BatchTask, ProcessedTask


class ProcessedTaskStore:
    """
    Stores processed individual customer tasks.
    """

    def __init__(self):
        self.tasks: Dict[str, ProcessedTask] = {}

    def add(self, task: ProcessedTask) -> None:
        self.tasks[task.task_id] = task

    def get(self, task_id: str) -> Optional[ProcessedTask]:
        return self.tasks.get(task_id)

    def get_all(self) -> List[ProcessedTask]:
        return list(self.tasks.values())

    def get_by_destination(self, destination: str) -> List[ProcessedTask]:
        return [
            task
            for task in self.tasks.values()
            if task.destination == destination
        ]

    def remove(self, task_id: str) -> Optional[ProcessedTask]:
        return self.tasks.pop(task_id, None)

    def count(self) -> int:
        return len(self.tasks)

    def summary(self) -> Dict:
        return {
            task_id: task.to_dict()
            for task_id, task in self.tasks.items()
        }


class BatchBuffer:
    """
    Flexible destination-aware batch buffer.

    New behaviour:
    - min_batch_size does NOT immediately close a batch.
    - tasks with the same destination stay in an open buffer.
    - a batch is closed only when one of the close conditions is met.

    Close conditions:
    1. max_batch_size reached
    2. max_total_weight reached
    3. waiting time >= max_wait_seconds AND task_count >= min_batch_size
    4. manual flush_all()
    """

    def __init__(
        self,
        min_batch_size: int = 2,
        max_batch_size: int = 10,
        max_wait_seconds: float = 30.0,
        max_total_weight: Optional[float] = None,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self.max_total_weight = max_total_weight

        self.buffer: Dict[str, List[ProcessedTask]] = defaultdict(list)
        self.group_created_at: Dict[str, datetime] = {}
        self.created_batches: Dict[str, BatchTask] = {}

    # --------------------------------------------------
    # Add task
    # --------------------------------------------------

    def add_processed_task(self, task: ProcessedTask) -> Optional[BatchTask]:
        """
        Add one processed task to the destination buffer.

        Important:
        - This no longer creates a batch just because min_batch_size is reached.
        - It only creates a batch immediately if hard limits are reached:
          max_batch_size or max_total_weight.
        """

        destination = task.destination

        if destination not in self.group_created_at:
            self.group_created_at[destination] = datetime.now()

        self.buffer[destination].append(task)

        return self._check_immediate_close(destination)

    def _check_immediate_close(self, destination: str) -> Optional[BatchTask]:
        tasks = self.buffer.get(destination, [])

        if not tasks:
            return None

        if self.max_batch_size and len(tasks) >= self.max_batch_size:
            return self.create_batch(
                destination=destination,
                close_reason="max_batch_size_reached",
            )

        if self.max_total_weight is not None:
            total_weight = self._calculate_total_weight(tasks)
            if total_weight >= self.max_total_weight:
                return self.create_batch(
                    destination=destination,
                    close_reason="max_total_weight_reached",
                )

        return None

    # --------------------------------------------------
    # Time/window based close
    # --------------------------------------------------

    def check_ready_batches(self) -> List[BatchTask]:
        """
        Check all open destination buffers.

        If a group has waited long enough and has at least min_batch_size tasks,
        close it and create a BatchTask.

        This should be called:
        - periodically by a scheduler
        - manually in tests
        - after new data.processed events if needed
        """

        ready_batches = []
        now = datetime.now()

        for destination in list(self.buffer.keys()):
            tasks = self.buffer.get(destination, [])

            if not tasks:
                continue

            created_at = self.group_created_at.get(destination, now)
            waiting_seconds = (now - created_at).total_seconds()

            if (
                len(tasks) >= self.min_batch_size
                and waiting_seconds >= self.max_wait_seconds
            ):
                batch = self.create_batch(
                    destination=destination,
                    close_reason="max_wait_seconds_reached",
                )
                if batch:
                    ready_batches.append(batch)

        return ready_batches

    # --------------------------------------------------
    # Create batch
    # --------------------------------------------------

    def create_batch(
        self,
        destination: str,
        close_reason: str = "manual_create_batch",
    ) -> Optional[BatchTask]:
        tasks = self.buffer.get(destination, [])

        if not tasks:
            return None

        products = []
        source_task_ids = []
        customer_ids = []
        all_constraints = set()

        for task in tasks:
            products.extend(task.products)
            source_task_ids.append(task.task_id)
            customer_ids.append(task.customer_id)
            all_constraints.update(task.constraints)

        batch = BatchTask(
            destination=destination,
            source_task_ids=source_task_ids,
            customer_ids=list(sorted(set(customer_ids))),
            products=products,
            objective=tasks[0].objective if tasks else "minimise_vehicle_number",
            constraints=list(sorted(all_constraints)),
            metadata={
                "batching_rule": "destination_window",
                "close_reason": close_reason,
                "source_task_count": len(tasks),
                "min_batch_size": self.min_batch_size,
                "max_batch_size": self.max_batch_size,
                "max_wait_seconds": self.max_wait_seconds,
                "max_total_weight": self.max_total_weight,
                "total_weight": self._calculate_total_weight(tasks),
            },
        )

        self.created_batches[batch.batch_id] = batch

        # Clear destination buffer after batch creation
        self.buffer[destination] = []
        self.group_created_at.pop(destination, None)

        return batch

    # --------------------------------------------------
    # Manual flush
    # --------------------------------------------------

    def flush_all(self, close_reason: str = "manual_flush") -> List[BatchTask]:
        """
        Create batches for all remaining destination groups.

        This ignores min_batch_size and is useful when:
        - testing ends
        - waiting window expires externally
        - operator manually triggers processing
        - system is shutting down
        """

        batches = []

        for destination in list(self.buffer.keys()):
            tasks = self.buffer.get(destination, [])

            if tasks:
                batch = self.create_batch(
                    destination=destination,
                    close_reason=close_reason,
                )
                if batch:
                    batches.append(batch)

        return batches

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _calculate_total_weight(self, tasks: List[ProcessedTask]) -> float:
        total = 0.0

        for task in tasks:
            total += task.total_weight()

        return total

    def _get_waiting_seconds(self, destination: str) -> float:
        created_at = self.group_created_at.get(destination)

        if not created_at:
            return 0.0

        return (datetime.now() - created_at).total_seconds()

    def get_buffer_summary(self) -> Dict:
        return {
            destination: {
                "task_count": len(tasks),
                "task_ids": [task.task_id for task in tasks],
                "product_count": sum(task.product_count() for task in tasks),
                "total_weight": sum(task.total_weight() for task in tasks),
                "waiting_seconds": self._get_waiting_seconds(destination),
                "min_batch_size": self.min_batch_size,
                "max_batch_size": self.max_batch_size,
                "max_wait_seconds": self.max_wait_seconds,
                "max_total_weight": self.max_total_weight,
                "ready_by_min_size": len(tasks) >= self.min_batch_size,
            }
            for destination, tasks in self.buffer.items()
            if tasks
        }

    def get_created_batches(self) -> List[BatchTask]:
        return list(self.created_batches.values())