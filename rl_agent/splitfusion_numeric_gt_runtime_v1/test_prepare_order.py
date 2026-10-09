"""Sent-frame admission must outlive adversarial worker scheduling."""
from __future__ import annotations

import threading
import unittest

from . import ue_live as U
from .test_ue_live import BOOT, DIGEST, handoff


def second_ticket(bridge):
    ticket = U._Ticket(2, 13, {"frame_id": 13}, {},
        U.event("collector", "owned_tick_snapshot", 101, 101, BOOT),
        source_owned=U.event("ue", "source_owned", 111, 101, BOOT))
    ticket.plan_bytes = b"second full plan"
    ticket.action_open = U.event("ue", "action_open", 121, 101, BOOT)
    bridge._tickets[13] = ticket
    return ticket


class OrderedPublisher:
    targets = []
    first_target = threading.Event()

    @classmethod
    def publish_target(cls, path, **kwargs):
        cls.targets.append(kwargs["packet"])
        if kwargs["packet"] == b"target1":
            cls.first_target.set()

    @classmethod
    def publish_terminal(cls, path, **kwargs):
        return None

    @classmethod
    def await_completion(cls, path, **kwargs):
        return {"status": "SCORED_DURABLE", "scorer_durable_proof": {"status": "DURABLE"}}


class OrderedClient:
    pending = None

    def __init__(self):
        self.calls = []

    def call(self, operation, value):
        self.calls.append((operation, value["job"]))
        if operation == "PUBLISH_READY":
            return f"target{value['job']}".encode()
        if operation == "EXPORT_LIVE_SOURCE":
            return {"identity_bytes": value["identity_bytes"],
                    "semantic_sha256": value["semantic_sha256"]}
        return {"status": "OK"}


class PrepareOrderTests(unittest.TestCase):
    def setUp(self):
        OrderedPublisher.targets.clear()
        OrderedPublisher.first_target.clear()

    def _bridge(self):
        bridge, first = handoff()
        second = second_ticket(bridge)
        bridge.publisher = OrderedPublisher
        bridge.client = OrderedClient()
        prepared = []
        def prepare(ticket):
            prepared.append(ticket.job)
            ticket.stage_event = U.event("producer", "source_stage_received",
                                           140 + ticket.job, 102, BOOT)
            return {"job": ticket.job, "identity_bytes": f"identity{ticket.job}".encode(),
                    "semantic_sha256": DIGEST}
        bridge._prepare_sent = prepare
        return bridge, first, second, prepared

    def test_later_worker_cannot_prepare_first_and_first_feedback_wait_does_not_block_second(self):
        bridge, first, second, prepared = self._bridge()
        older_delayed = threading.Event()
        release_older = threading.Event()
        later_waiting = threading.Event()
        original = bridge._await_prepare_turn
        def adversarial_schedule(ticket):
            if ticket is first:
                older_delayed.set()
                if not release_older.wait(2):
                    raise AssertionError("older scheduling gate not released")
            else:
                later_waiting.set()
            return original(ticket)
        bridge._await_prepare_turn = adversarial_schedule
        second.terminal = U.event("ue", "feedback_timeout", 201, 101, BOOT)
        bridge.after_submit(type("Runtime", (), {"last_datagram": {12: 130, 13: 131}})(),
                            12, {"sent": True})
        self.assertTrue(older_delayed.wait(1))
        bridge.after_submit(type("Runtime", (), {"last_datagram": {12: 130, 13: 131}})(),
                            13, {"sent": True})
        self.assertTrue(later_waiting.wait(1))
        self.assertEqual(list(bridge._producer_order), [1, 2])
        self.assertEqual(prepared, [])
        release_older.set()
        self.assertTrue(OrderedPublisher.first_target.wait(1))
        # The first thread is now awaiting actual UE terminal, yet the second
        # can prepare and publish because the producer history turn was freed.
        second.thread.join(2)
        self.assertFalse(second.thread.is_alive())
        self.assertEqual(prepared, [1, 2])
        self.assertEqual(OrderedPublisher.targets, [b"target1", b"target2"])
        self.assertIsNone(first.terminal)
        first.terminal = U.event("ue", "feedback_timeout", 200, 101, BOOT)
        with first.condition:
            first.condition.notify_all()
        first.thread.join(2)
        self.assertFalse(first.thread.is_alive())
        self.assertEqual(bridge.errors, [])
        self.assertTrue(first.retired and second.retired)
        self.assertEqual(list(bridge._producer_order), [])

    def test_earlier_fault_wakes_later_order_waiter_without_prepare(self):
        bridge, first, second, prepared = self._bridge()
        older_delayed = threading.Event()
        release_older = threading.Event()
        later_waiting = threading.Event()
        original = bridge._await_prepare_turn
        def adversarial_schedule(ticket):
            if ticket is first:
                older_delayed.set()
                release_older.wait(2)
            else:
                later_waiting.set()
            return original(ticket)
        bridge._await_prepare_turn = adversarial_schedule
        bridge.after_submit(type("Runtime", (), {"last_datagram": {12: 130, 13: 131}})(),
                            12, {"sent": True})
        self.assertTrue(older_delayed.wait(1))
        bridge.after_submit(type("Runtime", (), {"last_datagram": {12: 130, 13: 131}})(),
                            13, {"sent": True})
        self.assertTrue(later_waiting.wait(1))
        bridge._fault(first, RuntimeError("older producer failed"))
        release_older.set()
        first.thread.join(2)
        second.thread.join(2)
        self.assertFalse(first.thread.is_alive() or second.thread.is_alive())
        self.assertEqual(prepared, [])
        self.assertEqual(OrderedPublisher.targets, [])
        self.assertIn("older producer failed", " ".join(bridge.errors))


if __name__ == "__main__":
    unittest.main()
