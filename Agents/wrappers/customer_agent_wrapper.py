from typing import Any, Dict, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.customer import Customer, CustomerRequest
from Agents.core.event import Event
from Agents.core.event_bus import EventBus


class CustomerAgentWrapper(BaseAgent):
    """
    CustomerAgentWrapper represents a customer-side agent.

    Its current role:
    - receive a CustomerRequest
    - add source_task_id/customer_id mapping into each product
    - publish customer.request.submitted event

    It does not directly call DataAgent, RAGAgent, or OptimisationAgent.
    """

    def __init__(
        self,
        agent_id: str,
        customer: Customer,
        event_bus: EventBus,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="customer",
            name="Customer Agent Wrapper",
            config=config,
        )

        self.customer = customer
        self.event_bus = event_bus

    def submit_request(self, request: CustomerRequest) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(request.request_id)

        try:
            self.update_belief("customer_id", self.customer.customer_id)
            self.update_belief("request_id", request.request_id)
            self.update_belief("destination", request.destination)
            self.update_belief("product_count", len(request.products))
            self.update_belief("objective", request.objective)
            self.update_belief("constraints", request.constraints)

            self.add_desire("submit_customer_loading_request")
            self.add_intention("publish_customer_request_submitted_event")

            enriched_products = self._attach_customer_mapping(request)

            event = Event(
                topic="customer.request.submitted",
                source_agent_id=self.agent_id,
                correlation_id=request.request_id,
                payload={
                    "request_id": request.request_id,
                    "customer_id": request.customer_id,
                    "customer": self.customer.to_dict(),
                    "destination": request.destination,
                    "products": enriched_products,
                    "objective": request.objective,
                    "constraints": request.constraints,
                    "priority": request.priority,
                    "due_time": request.due_time,
                    "metadata": request.metadata,
                },
            )

            self.event_bus.publish(event)

            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "completed",
                "message": "Customer request submitted successfully.",
                "request_id": request.request_id,
                "customer_id": request.customer_id,
                "destination": request.destination,
                "product_count": len(enriched_products),
                "published_topic": "customer.request.submitted",
            }

            self.update_status("completed")
            self.clear_current_task()

            return result

        except Exception as e:
            self.update_belief("error", str(e))
            self.update_status("failed")
            self.clear_current_task()

            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "failed",
                "message": "CustomerAgentWrapper failed to submit request.",
                "error": str(e),
            }

    def act(self, task: Any) -> Dict[str, Any]:
        """
        Standard BaseAgent interface.

        For CustomerAgentWrapper, task should be a CustomerRequest.
        """

        if not isinstance(task, CustomerRequest):
            raise TypeError(
                "CustomerAgentWrapper.act(task) expects a CustomerRequest."
            )

        return self.submit_request(task)

    def _attach_customer_mapping(
        self,
        request: CustomerRequest,
    ):
        """
        Add request/customer mapping to each product.

        This is important for batching:
        after multiple customer requests are merged,
        we still know which product belongs to which customer/request.
        """

        enriched_products = []

        for index, product in enumerate(request.products):
            enriched_product = dict(product)

            enriched_product["source_request_id"] = request.request_id
            enriched_product["source_task_id"] = request.request_id
            enriched_product["customer_id"] = request.customer_id

            if "product_id" not in enriched_product:
                enriched_product["product_id"] = f"{request.request_id}_P{index + 1:03d}"

            if "destination" not in enriched_product:
                enriched_product["destination"] = request.destination

            enriched_products.append(enriched_product)

        return enriched_products