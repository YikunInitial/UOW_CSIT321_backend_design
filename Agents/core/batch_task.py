from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class ProcessedTask:
    """
    ProcessedTask represents one customer request after data preprocessing.

    It is the unit that can be grouped into a BatchTask.
    """

    task_id: str
    customer_id: str
    destination: str
    products: List[Dict[str, Any]]

    objective: str = "minimise_vehicle_number"
    constraints: List[str] = field(default_factory=list)
    priority: str = "normal"
    due_time: Optional[str] = None

    status: str = "processed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def product_count(self) -> int:
        return len(self.products)

    def total_weight(self) -> float:
        total = 0.0
        for product in self.products:
            try:
                total += float(product.get("weight", 0.0))
            except Exception:
                pass
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "customer_id": self.customer_id,
            "destination": self.destination,
            "products": self.products,
            "objective": self.objective,
            "constraints": self.constraints,
            "priority": self.priority,
            "due_time": self.due_time,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "product_count": self.product_count(),
            "total_weight": self.total_weight(),
        }


@dataclass
class BatchTask:
    """
    BatchTask represents a group of compatible processed tasks.

    For now, compatibility is based on destination.
    Later, we can extend it to:
    - destination
    - departure time window
    - constraint profile
    - priority
    - wagon type
    """

    destination: str
    source_task_ids: List[str]
    customer_ids: List[str]
    products: List[Dict[str, Any]]

    batch_id: str = field(default_factory=lambda: f"BATCH_{uuid.uuid4().hex[:8]}")
    objective: str = "minimise_vehicle_number"
    constraints: List[str] = field(default_factory=list)
    status: str = "ready"

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def product_count(self) -> int:
        return len(self.products)

    def total_weight(self) -> float:
        total = 0.0
        for product in self.products:
            try:
                total += float(product.get("weight", 0.0))
            except Exception:
                pass
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "destination": self.destination,
            "source_task_ids": self.source_task_ids,
            "customer_ids": self.customer_ids,
            "products": self.products,
            "objective": self.objective,
            "constraints": self.constraints,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "product_count": self.product_count(),
            "total_weight": self.total_weight(),
        }