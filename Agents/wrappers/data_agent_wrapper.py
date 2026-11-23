from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from Agents.core.base_agent import BaseAgent
from Agents.core.event import Event
from Agents.core.event_bus import EventBus
from Agents.core.task import AgentTask
from Agents.data_agent import DataAgent


class DataAgentWrapper(BaseAgent):
    """
    Managed wrapper for the existing Agents/data_agent.py.

    Supports two modes:

    1. Synchronous workflow mode:
       CoordinatorAgent -> DataAgentWrapper.act(task)
       This mode reads a cleaned Excel file through DataAgent.

    2. Event-driven mode:
       CustomerAgentWrapper publishes customer.request.submitted.
       The event already contains the request-specific products.
       DataAgentWrapper validates, saves, groups, and publishes data.processed.

    Important:
    Event-driven mode must NOT reload the full Excel file for every request.
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="data",
            name="Data Agent Wrapper",
            config=config,
        )

        self.data_agent = DataAgent()
        self.event_bus = event_bus

    # ==================================================
    # Mode 1: Synchronous call for CoordinatorAgent
    # ==================================================

    def act(self, task: AgentTask) -> Dict[str, Any]:
        self.update_status("running")
        self.set_current_task(task.task_id)

        try:
            if not task.input_file:
                raise ValueError("task.input_file is required for DataAgentWrapper.")

            input_path = Path(task.input_file)
            output_path = self._resolve_output_path(task)

            self.update_belief("task_id", task.task_id)
            self.update_belief("input_file", str(input_path))
            self.update_belief("json_output_path", str(output_path))
            self.update_belief("objective", task.objective)
            self.update_belief("constraints", task.constraints)

            self.add_desire("convert_cleaned_excel_to_standard_json")
            self.add_intention("call_existing_DataAgent_run")

            records = self.data_agent.run(
                cleaned_excel_path=input_path,
                json_output_path=output_path,
            )

            result = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "completed",
                "message": "DataAgentWrapper completed successfully.",
                "input_file": str(input_path),
                "json_output_path": str(output_path),
                "record_count": len(records),
                "records_preview": records[:3],
            }

            self.update_belief("record_count", len(records))
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
                "message": "DataAgentWrapper failed.",
                "error": str(e),
            }

    def _resolve_output_path(self, task: AgentTask) -> Path:
        """
        Decide where synchronous-mode JSON output should be saved.

        Priority:
        1. task.metadata["json_output_path"]
        2. self.config["json_output_path"]
        3. Agents/products.json
        """

        if task.metadata.get("json_output_path"):
            output_path = Path(task.metadata["json_output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

        if self.config.get("json_output_path"):
            output_path = Path(self.config["json_output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

        output_path = Path("Agents/products.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    # ==================================================
    # Mode 2: Event-driven mode
    # ==================================================

    def attach_event_bus(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.update_belief("event_bus_attached", True)

    def start_event_listener(self) -> None:
        """
        Subscribe this DataAgentWrapper to customer.request.submitted.
        """

        if self.event_bus is None:
            raise ValueError(
                "DataAgentWrapper requires an EventBus before starting "
                "event listener."
            )

        self.event_bus.subscribe(
            topic="customer.request.submitted",
            handler=self.handle_customer_request_submitted,
        )

        self.update_status("idle")
        self.update_belief("subscribed_topic", "customer.request.submitted")
        self.add_desire("process_customer_request_products")
        self.add_intention("listen_to_customer_request_submitted_events")

        print(f"[{self.agent_id}] Subscribed to customer.request.submitted")

    def handle_customer_request_submitted(self, event: Event) -> None:
        """
        Process request-specific products already included in the event.

        Expected event.payload:
        {
            "request_id": "...",
            "customer_id": "...",
            "destination": "...",
            "products": [{...}, {...}],
            "objective": "...",
            "constraints": [...],
            "priority": "...",
            "due_time": "...",
            "metadata": {
                "json_output_path": "...",
                "source_dataset_path": "..."
            }
        }

        Event-driven mode intentionally does not call DataAgent.run().
        Excel is loaded once before requests are published.
        """

        if self.event_bus is None:
            raise ValueError("EventBus is not attached to DataAgentWrapper.")

        self.update_status("running")

        try:
            payload = event.payload
            request_id = payload.get("request_id")
            customer_id = payload.get("customer_id")
            metadata = payload.get("metadata", {})

            if not request_id:
                raise ValueError(
                    "customer.request.submitted event missing request_id."
                )

            if not customer_id:
                raise ValueError(
                    "customer.request.submitted event missing customer_id."
                )

            products = self._extract_event_products(payload)

            output_path = self._resolve_event_output_path(
                request_id=request_id,
                metadata=metadata,
            )

            self.set_current_task(request_id)

            self.update_belief("last_request_id", request_id)
            self.update_belief("last_customer_id", customer_id)
            self.update_belief("event_product_count", len(products))
            self.update_belief("event_json_output_path", str(output_path))
            self.update_belief(
                "event_product_source",
                "customer.request.submitted.payload.products",
            )

            self.add_desire("validate_group_and_publish_processed_data")
            self.add_intention("process_request_specific_products")

            print(
                f"[{self.agent_id}] Processing request products | "
                f"request={request_id} | customer={customer_id} | "
                f"products={len(products)} | output={output_path}"
            )

            enriched_products = self._attach_customer_mapping(
                products=products,
                request_id=request_id,
                customer_id=customer_id,
            )

            # Each request gets its own audit/debug JSON file.
            self._export_event_products(
                products=enriched_products,
                output_path=output_path,
            )

            destination_groups = self._group_products_by_destination(
                products=enriched_products,
                fallback_destination=payload.get("destination"),
            )

            self.update_belief("event_destination_count", len(destination_groups))
            self.update_belief(
                "event_destinations",
                list(destination_groups.keys()),
            )

            for destination, destination_products in destination_groups.items():
                self._publish_data_processed(
                    request_id=request_id,
                    customer_id=customer_id,
                    destination=destination,
                    products=destination_products,
                    products_json_path=output_path,
                    source_event=event,
                    payload=payload,
                )

            self.update_status("idle")
            self.clear_current_task()

        except Exception as e:
            self.update_belief("error", str(e))
            self.update_status("failed")
            self.clear_current_task()

            print(
                f"[{self.agent_id}] Failed to handle "
                f"customer.request.submitted: {e}"
            )

    def _publish_data_processed(
        self,
        request_id: str,
        customer_id: str,
        destination: str,
        products: List[Dict[str, Any]],
        products_json_path: Path,
        source_event: Event,
        payload: Dict[str, Any],
    ) -> None:
        metadata = payload.get("metadata", {})

        processed_event = Event(
            topic="data.processed",
            source_agent_id=self.agent_id,
            correlation_id=request_id,
            payload={
                "task_id": f"{request_id}_{destination}",
                "request_id": request_id,
                "customer_id": customer_id,
                "destination": destination,
                "products": products,
                "products_json_path": str(products_json_path),
                "objective": payload.get(
                    "objective",
                    "minimise_vehicle_number",
                ),
                "constraints": payload.get("constraints", []),
                "priority": payload.get("priority", "normal"),
                "due_time": payload.get("due_time"),
                "metadata": {
                    **metadata,
                    "processed_by": self.agent_id,
                    "source_event_id": source_event.event_id,
                    "source_products_json_path": str(products_json_path),
                    "destination": destination,
                    "destination_product_count": len(products),
                },
            },
        )

        print(
            f"[{self.agent_id}] Publishing data.processed | "
            f"request={request_id} | destination={destination} | "
            f"products={len(products)}"
        )

        self.event_bus.publish(processed_event)

    # ==================================================
    # Event helper methods
    # ==================================================

    def _extract_event_products(
        self,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Validate the request-specific products supplied by the event.

        Products must already be assigned before publishing the request.
        """

        products = payload.get("products")

        if not isinstance(products, list) or not products:
            raise ValueError(
                "customer.request.submitted must include a non-empty "
                "'products' list."
            )

        validated_products: List[Dict[str, Any]] = []
        seen_product_ids = set()

        for index, product in enumerate(products):
            if not isinstance(product, dict):
                raise ValueError(
                    f"products[{index}] must be a dictionary, "
                    f"got {type(product).__name__}."
                )

            product_copy = dict(product)
            product_id = str(product_copy.get("product_id", "")).strip()

            if not product_id:
                raise ValueError(
                    f"products[{index}] is missing a valid product_id."
                )

            if product_id in seen_product_ids:
                raise ValueError(
                    f"Duplicate product_id '{product_id}' found inside "
                    f"request {payload.get('request_id')}."
                )

            seen_product_ids.add(product_id)
            validated_products.append(product_copy)

        return validated_products

    def _export_event_products(
        self,
        products: List[Dict[str, Any]],
        output_path: Path,
    ) -> Path:
        """
        Save only the current request's product subset.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

        return output_path

    def _resolve_event_output_path(
        self,
        request_id: str,
        metadata: Dict[str, Any],
    ) -> Path:
        """
        Each request has its own output JSON file.
        """

        if metadata.get("json_output_path"):
            output_path = Path(metadata["json_output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return output_path

        output_dir = Path("runtime") / "tasks" / request_id
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir / "products.json"

    def _attach_customer_mapping(
        self,
        products: List[Dict[str, Any]],
        request_id: str,
        customer_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Ensure each product is linked to its source request and customer.

        CustomerAgentWrapper normally already adds these fields.
        This method verifies that existing values do not conflict.
        """

        enriched_products: List[Dict[str, Any]] = []

        for index, product in enumerate(products):
            enriched_product = dict(product)

            existing_request_id = enriched_product.get("source_request_id")
            if (
                existing_request_id
                and existing_request_id != request_id
            ):
                raise ValueError(
                    f"Product '{enriched_product.get('product_id')}' belongs "
                    f"to source_request_id='{existing_request_id}', not "
                    f"'{request_id}'."
                )

            existing_customer_id = enriched_product.get("customer_id")
            if (
                existing_customer_id
                and existing_customer_id != customer_id
            ):
                raise ValueError(
                    f"Product '{enriched_product.get('product_id')}' belongs "
                    f"to customer_id='{existing_customer_id}', not "
                    f"'{customer_id}'."
                )

            enriched_product["source_request_id"] = request_id
            enriched_product["source_task_id"] = request_id
            enriched_product["customer_id"] = customer_id

            if "product_id" not in enriched_product:
                enriched_product["product_id"] = (
                    f"{request_id}_P{index + 1:03d}"
                )

            enriched_products.append(enriched_product)

        return enriched_products

    def _group_products_by_destination(
        self,
        products: List[Dict[str, Any]],
        fallback_destination: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        groups = defaultdict(list)

        for product in products:
            destination = self._extract_destination(product)

            if not destination:
                destination = fallback_destination

            if not destination or destination == "AUTO_FROM_EXCEL":
                destination = "UNKNOWN_DESTINATION"

            groups[str(destination)].append(product)

        return dict(groups)

    def _extract_destination(
        self,
        product: Dict[str, Any],
    ) -> Optional[str]:
        """
        Support possible destination field names from DataAgent output.
        """

        destination_keys = [
            "destination",
            "Destination",
            "DESTINATION",
            "destination_position",
            "DestinationPosition",
            "target_position",
            "TargetPosition",
            "departure_position",
            "DeparturePosition",
            "arrival_station",
            "ArrivalStation",
            "station",
            "Station",
            "到站",
            "到站名",
            "目的地",
            "收货地",
            "到达站",
        ]

        for key in destination_keys:
            value = product.get(key)

            if value is not None and str(value).strip():
                return str(value).strip()

        return None