from Agents.core.event import Event
from Agents.core.event_bus import EventBus


def handle_customer_request(event: Event):
    print("\n[Handler] Received customer request event:")
    print(event.to_dict())


def handle_data_processed(event: Event):
    print("\n[Handler] Received data processed event:")
    print(event.to_dict())


def main():
    event_bus = EventBus()

    # Subscribe handlers
    event_bus.subscribe(
        topic="customer.request.submitted",
        handler=handle_customer_request,
    )

    event_bus.subscribe(
        topic="data.processed",
        handler=handle_data_processed,
    )

    print("After subscribing handlers:")
    event_bus.print_summary()

    # Publish customer request event
    customer_event = Event(
        topic="customer.request.submitted",
        source_agent_id="CustomerAgent_001",
        correlation_id="REQ_001",
        payload={
            "task_id": "REQ_001",
            "customer_id": "Customer_001",
            "destination": "Shanghai",
            "products": [
                {
                    "product_id": "P001",
                    "weight": 12.5,
                    "destination": "Shanghai",
                }
            ],
        },
    )

    event_bus.publish(customer_event)

    # Publish data processed event
    data_event = Event(
        topic="data.processed",
        source_agent_id="DataAgent_001",
        correlation_id="REQ_001",
        payload={
            "task_id": "REQ_001",
            "customer_id": "Customer_001",
            "destination": "Shanghai",
            "products_json_path": "runtime/tasks/REQ_001/products.json",
            "product_count": 1,
        },
    )

    event_bus.publish(data_event)

    print("\nEvent history:")
    for event in event_bus.get_history():
        print(event)

    print("\nData processed history:")
    for event in event_bus.get_history_by_topic("data.processed"):
        print(event.to_dict())


if __name__ == "__main__":
    main()