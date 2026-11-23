from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid


@dataclass
class Event:
    """
    Event represents one message published inside the agent system.

    Example topics:
    - customer.request.submitted
    - data.processed
    - batch.ready
    - optimisation.completed
    - report.generated
    """

    topic: str
    payload: Dict[str, Any]

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "source_agent_id": self.source_agent_id,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"Event("
            f"event_id='{self.event_id}', "
            f"topic='{self.topic}', "
            f"source_agent_id='{self.source_agent_id}', "
            f"correlation_id='{self.correlation_id}'"
            f")"
        )