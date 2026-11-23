import unittest

from Agents.core.research_contracts import AgenticSystemError
from Agents.event_state import bind_and_validate_event, complete_event, new_event_request


class EventStateTest(unittest.TestCase):
    def state(self) -> dict:
        return {
            "active_plan_id": "PLAN-1",
            "active_plan_version": 3,
            "processed_event_ids": [],
        }

    def test_unbound_queued_event_binds_at_processing_time(self) -> None:
        event = new_event_request({"event_type": "add_products"})
        bound = bind_and_validate_event(event, self.state())
        self.assertEqual(bound["expected_plan_id"], "PLAN-1")
        self.assertEqual(bound["expected_plan_version"], 3)
        self.assertEqual(bound["event_status"], "processing")

    def test_duplicate_event_is_rejected(self) -> None:
        state = self.state()
        state["processed_event_ids"] = ["EVENT-1"]
        with self.assertRaises(AgenticSystemError) as raised:
            bind_and_validate_event({"event_id": "EVENT-1"}, state)
        self.assertEqual(raised.exception.to_dict()["code"], "DUPLICATE_EVENT")

    def test_stale_plan_version_is_rejected(self) -> None:
        event = new_event_request(
            {
                "event_type": "disable_wagon",
                "expected_plan_id": "PLAN-1",
                "expected_plan_version": 2,
            }
        )
        with self.assertRaises(AgenticSystemError) as raised:
            bind_and_validate_event(event, self.state())
        self.assertEqual(raised.exception.to_dict()["code"], "STALE_PLAN_VERSION")

    def test_completion_advances_active_plan_once(self) -> None:
        state = self.state()
        event = new_event_request({"event_id": "EVENT-1"})
        complete_event(state, event, "PLAN-2")
        self.assertEqual(state["active_plan_id"], "PLAN-2")
        self.assertEqual(state["active_plan_version"], 4)
        self.assertEqual(state["processed_event_ids"], ["EVENT-1"])


if __name__ == "__main__":
    unittest.main()
