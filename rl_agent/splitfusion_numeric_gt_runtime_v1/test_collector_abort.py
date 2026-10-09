"""Retained collector failure must not abandon its explicitly owned child."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from . import ue_live as U
from .test_worker_owner import _sources
from .worker_owner import OwnedClient


class RetainedFinishError(RuntimeError):
    pass


def failing_collector(handoff):
    world = object()
    handoff.catalogue.world = world

    class RetainedCollector:
        def __init__(self, **kwargs):
            self.world = world

        def finish(self):
            raise RetainedFinishError("retained collector finish failed")

    return LG.build_live_gt_collector(RetainedCollector)()


def pending_ticket(handoff):
    ticket = U._Ticket(1, 12, {}, {}, U.event("collector", "owned_tick_snapshot",
                        1, os.getpid(), handoff.boot_id))
    handoff._tickets[12] = ticket
    return ticket


class CollectorAbortTests(unittest.TestCase):
    def test_retained_finish_exception_preserved_and_real_owned_child_reaped(self):
        with tempfile.TemporaryDirectory(prefix="numeric-collector-abort-", dir="/tmp") as root:
            attempt = Path(root)
            with U.install_numeric_ue(attempt_dir=attempt, registration={}) as handoff:
                collector = failing_collector(handoff)
                ticket = pending_ticket(handoff)
                client = OwnedClient("echo_test_only", {"offline_test": True},
                    directory=handoff.directory / "producer_process",
                    writable_attempt_root=attempt, source_sha256=_sources(),
                    cpu=min(os.sched_getaffinity(0)), timeout_s=15)
                handoff.client = client
                self.assertEqual(client.call("ECHO", {"owned": True}), {"owned": True})
                with self.assertRaisesRegex(RetainedFinishError, "retained collector finish failed"):
                    collector.finish()
                self.assertTrue(client.closed)
                self.assertFalse(client.owner_thread_alive)
                self.assertIsNotNone(client.process.poll())
                self.assertFalse((handoff.directory / "PROCESS_CLOSE.json").exists())
                evidence = C._parse((handoff.directory / "UE_ABORT_FAULT.json").read_bytes())
                self.assertEqual(evidence["cause"],
                                 "RetainedFinishError: retained collector finish failed")
                self.assertIn("RetainedFinishError", evidence["tickets"][0]["fault"])
                self.assertEqual(evidence["producer_events"][-1]["operation"], "ECHO")
                self.assertIsNone(ticket.owned_source)

    def test_secondary_abort_failures_do_not_mask_retained_finish_exception(self):
        class FailingAbortClient:
            closed = False
            ready = {"status": "READY"}
            pending = None
            events = []

            def __init__(self):
                self.abort_calls = 0

            def abort(self):
                self.abort_calls += 1
                self.closed = True
                raise RuntimeError("owned child abort failed")

        with tempfile.TemporaryDirectory(prefix="numeric-collector-abort-", dir="/tmp") as root:
            with U.install_numeric_ue(attempt_dir=Path(root), registration={}) as handoff:
                collector = failing_collector(handoff)
                pending_ticket(handoff)
                client = FailingAbortClient()
                handoff.client = client
                with mock.patch.object(handoff, "_write_abort_fault",
                                       side_effect=OSError("abort evidence fsync failed")):
                    with self.assertRaisesRegex(RetainedFinishError,
                                                "retained collector finish failed"):
                        collector.finish()
                self.assertEqual(client.abort_calls, 1)
                secondaries = " ".join(handoff.errors)
                self.assertIn("abort evidence fsync failed", secondaries)
                self.assertIn("owned child abort failed", secondaries)


if __name__ == "__main__":
    unittest.main()
