from collections import defaultdict
from typing import Callable, Dict, List

from Agents.core.event import Event


class EventBus:
    """
    Simple in-memory publish-subscribe event bus.

    Current version:
    - synchronous
    - in-memory
    - good for local testing and demo

    Later it can be replaced by:
    - asyncio event bus
    - Redis Pub/Sub
    - Redis Stream
    - Kafka
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self.event_history: List[Event] = []

    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """
        Subscribe a handler to a topic.

        Example:
            event_bus.subscribe("data.processed", batching_agent.handle_event)
        """

        if handler not in self.subscribers[topic]:
            self.subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Event], None]) -> None:
        """
        Remove a handler from a topic.
        """

        if handler in self.subscribers[topic]:
            self.subscribers[topic].remove(handler)

    def publish(self, event: Event) -> None:
        """
        Publish an event to all handlers subscribed to the event topic.
        """

        self.event_history.append(event)

        handlers = self.subscribers.get(event.topic, [])

        print(
            f"[EventBus] Publishing event: topic={event.topic}, "
            f"event_id={event.event_id}, subscribers={len(handlers)}"
        )

        for handler in handlers:
            handler(event)

    def get_history(self) -> List[Event]:
        return self.event_history

    def get_history_by_topic(self, topic: str) -> List[Event]:
        return [
            event
            for event in self.event_history
            if event.topic == topic
        ]

    def summary(self) -> Dict[str, int]:
        return {
            topic: len(handlers)
            for topic, handlers in self.subscribers.items()
        }

    def print_summary(self) -> None:
        print("EventBus Summary")
        print("----------------")

        if not self.subscribers:
            print("No subscribers.")
            return

        for topic, handlers in self.subscribers.items():
            print(f"{topic}: {len(handlers)} subscriber(s)")