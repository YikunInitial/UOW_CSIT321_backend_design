from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.batch_buffer import BatchBuffer, ProcessedTaskStore
from Agents.core.batch_task import ProcessedTask, BatchTask
from Agents.core.event import Event
from Agents.core.event_bus import EventBus


class BatchingAgentWrapper(BaseAgent):
    """
    BatchingAgentWrapper listens to data.processed events and groups tasks
    into BatchTask objects based on destination.

    Current batching rule:
    - same destination
    - when destination group reaches min_batch_size, publish batch.ready

    Later we can extend batching rule to:
    - destination
    - departure time window
    - constraint profile
    - priority
    - wagon type
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: EventBus,
        processed_task_store: ProcessedTaskStore,
        batch_buffer: BatchBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="batching",
            name="Batching Agent Wrapper",
            config=config,
        )

        self.event_bus = event_bus
        self.processed_task_store = processed_task_store
        self.batch_buffer = batch_buffer

    def start(self) -> None:
        """
        Subscribe this agent to data.processed events.
        """

        self.event_bus.subscribe(
            topic="data.processed",
            handler=self.handle_data_processed,
        )

        self.update_status("idle")
        self.update_belief("subscribed_topic", "data.processed")
        self.add_desire("group_processed_tasks_into_batches")
        self.add_intention("listen_to_data_processed_events")

        print(f"[{self.agent_id}] Subscribed to data.processed")

    def handle_data_processed(self, event: Event) -> None:
        """
        Event handler for data.processed.

        Expected event.payload:
        {
            "task_id": "...",
            "customer_id": "...",
            "destination": "...",
            "products": [...],
            "objective": "...",
            "constraints": [...],
            "priority": "...",
            "due_time": "...",
            "metadata": {...}
        }
        """

        self.update_status("running")

        try:
            processed_task = self._event_to_processed_task(event)

            self.processed_task_store.add(processed_task)
            batch = self.batch_buffer.add_processed_task(processed_task)

            self.update_belief("last_processed_task_id", processed_task.task_id)
            self.update_belief("last_destination", processed_task.destination)
            self.update_belief(
                "processed_task_store_count",
                self.processed_task_store.count(),
            )
            self.update_belief(
                "buffer_summary",
                self.batch_buffer.get_buffer_summary(),
            )

            print(
                f"[{self.agent_id}] Added processed task: "
                f"{processed_task.task_id} | destination={processed_task.destination}"
            )

            if batch:
                self.update_belief("last_created_batch_id", batch.batch_id)
                self.publish_batch_ready(batch)

            # Check window-based ready batches.
            # For example, batches that have reached max_wait_seconds.
            ready_batches = self.batch_buffer.check_ready_batches()

            for ready_batch in ready_batches:
                self.update_belief("last_created_batch_id", ready_batch.batch_id)
                self.publish_batch_ready(ready_batch)

            self.update_status("idle")

        except Exception as e:
            self.update_belief("error", str(e))
            self.update_status("failed")
            print(f"[{self.agent_id}] Failed to handle data.processed: {e}")

    def publish_batch_ready(self, batch: BatchTask) -> None:
        """
        Publish batch.ready event when a BatchTask is created.
        """

        event = Event(
            topic="batch.ready",
            source_agent_id=self.agent_id,
            correlation_id=batch.batch_id,
            payload={
                "batch": batch.to_dict(),
                "batch_id": batch.batch_id,
                "destination": batch.destination,
                "source_task_ids": batch.source_task_ids,
                "customer_ids": batch.customer_ids,
                "product_count": batch.product_count(),
                "total_weight": batch.total_weight(),
                "objective": batch.objective,
                "constraints": batch.constraints,
                "metadata": batch.metadata,
            },
        )

        print(
            f"[{self.agent_id}] Publishing batch.ready: "
            f"{batch.batch_id} | destination={batch.destination} | "
            f"tasks={batch.source_task_ids}"
        )

        self.event_bus.publish(event)

    def flush_remaining_batches(self) -> Dict[str, Any]:
        """
        Force all remaining buffered tasks to become batches.

        This is useful when:
        - waiting time expires
        - testing ends
        - operator wants to process all pending tasks
        """

        self.update_status("running")

        batches = self.batch_buffer.flush_all()

        for batch in batches:
            self.publish_batch_ready(batch)

        self.update_belief("flushed_batch_count", len(batches))
        self.update_status("idle")

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "message": "Remaining batches flushed.",
            "flushed_batch_count": len(batches),
            "batch_ids": [batch.batch_id for batch in batches],
        }

    def check_ready_batches(self) -> Dict[str, Any]:
        """
        Manually check whether any open destination groups should be closed
        based on window-based batching rules.
        """

        self.update_status("running")

        batches = self.batch_buffer.check_ready_batches()

        for batch in batches:
            self.publish_batch_ready(batch)

        self.update_belief("checked_ready_batch_count", len(batches))
        self.update_status("idle")

        return {
            "agent_id": self.agent_id,
            "status": "completed",
            "message": "Ready batches checked.",
            "ready_batch_count": len(batches),
            "batch_ids": [batch.batch_id for batch in batches],
        }

    def act(self, task: Any) -> Any:
        """
        Standard BaseAgent interface.

        For BatchingAgentWrapper, act() is not the main execution method.
        It is event-driven through handle_data_processed().
        """

        raise NotImplementedError(
            "BatchingAgentWrapper is event-driven. "
            "Use start() and handle_data_processed(event)."
        )

    def _event_to_processed_task(self, event: Event) -> ProcessedTask:
        payload = event.payload

        task_id = payload.get("task_id") or payload.get("request_id")
        customer_id = payload.get("customer_id")
        destination = payload.get("destination")
        products = payload.get("products", [])

        if not task_id:
            raise ValueError("data.processed event missing task_id/request_id.")

        if not customer_id:
            raise ValueError("data.processed event missing customer_id.")

        if not destination:
            raise ValueError("data.processed event missing destination.")

        if not products:
            raise ValueError("data.processed event missing products.")

        return ProcessedTask(
            task_id=task_id,
            customer_id=customer_id,
            destination=destination,
            products=products,
            objective=payload.get("objective", "minimise_vehicle_number"),
            constraints=payload.get("constraints", []),
            priority=payload.get("priority", "normal"),
            due_time=payload.get("due_time"),
            metadata=payload.get("metadata", {}),
        )