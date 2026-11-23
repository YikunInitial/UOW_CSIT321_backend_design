from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Customer:
    """
    Customer represents a customer/user who submits loading requests.
    """

    customer_id: str
    name: Optional[str] = None
    priority: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "priority": self.priority,
            "metadata": self.metadata,
        }


@dataclass
class CustomerRequest:
    """
    CustomerRequest represents one request submitted by a customer.

    In this testing stage, products can be provided directly as dictionaries.
    Later, products can come from uploaded Excel files or frontend forms.
    """

    customer_id: str
    destination: str
    products: List[Dict[str, Any]]

    request_id: str = field(default_factory=lambda: f"REQ_{uuid.uuid4().hex[:8]}")
    objective: str = "minimise_vehicle_number"
    constraints: List[str] = field(default_factory=list)
    priority: str = "normal"
    due_time: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "customer_id": self.customer_id,
            "destination": self.destination,
            "products": self.products,
            "objective": self.objective,
            "constraints": self.constraints,
            "priority": self.priority,
            "due_time": self.due_time,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }