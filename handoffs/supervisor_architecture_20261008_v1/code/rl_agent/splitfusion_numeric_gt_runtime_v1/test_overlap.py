"""Offline checks for two owned frames and independent worker failure.

These tests exercise retention and shutdown boundaries. They make no claim about
live route throughput or the radio feedback deadline.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_agent.splitfusion_numeric_gt_integration_v2.test_ownership import advance, plan_for

from .process import Client, ROOT
from .test_process import sources
from .test_producer import fixture, prepare, score, receipt


class OverlapTests(unittest.TestCase):
    def test_next_frame_can_commit_while_previous_score_and_archive_are_pending(self):
        # The source history advances once per accepted frame, even when the
        # previous frame is still waiting for score and durable evidence.
        with fixture(max_frames=2, max_retained_jobs=2) as (engine, kwargs, _):
            first = prepare(engine, kwargs)
            first_plan = plan_for(kwargs)[0]
            committed_first = engine.confirm_send(first.token, first_plan)
            engine.publish_ready(first.token)

            advance(kwargs)
            second = prepare(engine, kwargs)
            second_plan = plan_for(kwargs)[0]
            committed_second = engine.confirm_send(second.token, second_plan)
            engine.publish_ready(second.token)

            self.assertEqual(engine.snapshot()["retained_jobs"], 2)
            self.assertEqual(engine.snapshot()["owner"]["commit_calls"], 2)
            self.assertEqual(committed_first["history_after_sha256"],
                             committed_second["history_before_sha256"])
            self.assertNotEqual(first.identity_bytes, second.identity_bytes)

            # The later frame may finish first. Ownership of the older frame
            # must persist until both its scorer and durable archive complete.
            score(engine, second)
            second_archive = engine.build_archive(second.token)
            engine.archive_durable(second.token,
                receipt(second, second_archive, "frame_101/producer"),
                relative_prefix="frame_101/producer")
            self.assertEqual(engine.snapshot()["retained_jobs"], 2)
            engine.scorer_archive_completed(second.token, second.identity_bytes,
                                             second.semantic_sha256)
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)

            score(engine, first)
            first_archive = engine.build_archive(first.token)
            engine.scorer_archive_completed(first.token, first.identity_bytes,
                                             first.semantic_sha256)
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            engine.archive_durable(first.token,
                receipt(first, first_archive, "frame_100/producer"),
                relative_prefix="frame_100/producer")
            finished = engine.snapshot()
            self.assertEqual(finished["retained_jobs"], 0)
            self.assertEqual(finished["owner"]["reserved_bytes"], 0)
            self.assertEqual([finished["events"][str(n)]["outcome"] for n in (1, 2)],
                             ["DURABLE", "DURABLE"])
            self.assertTrue(engine.close()["valid"])

    def test_full_two_frame_retention_refuses_third_before_source_compute(self):
        with fixture(max_frames=3, max_retained_jobs=2) as (engine, kwargs, _):
            for _ in range(2):
                pending = prepare(engine, kwargs)
                engine.confirm_send(pending.token, plan_for(kwargs)[0])
                engine.publish_ready(pending.token)
                advance(kwargs)
            before = engine._owner._producer.history_sha256()
            with patch.object(engine._owner._producer, "compute",
                              wraps=engine._owner._producer.compute) as compute:
                with self.assertRaises(RuntimeError):
                    prepare(engine, kwargs)
                self.assertEqual(compute.call_count, 0)
            snapshot = engine.snapshot()
            self.assertEqual(snapshot["retained_jobs"], 2)
            self.assertEqual(snapshot["owner"]["commit_calls"], 2)
            self.assertEqual(engine._owner._producer.history_sha256(), before)

    def test_failed_worker_does_not_revive_or_interrupt_another_pending_worker(self):
        affinity = os.sched_getaffinity(0)
        cpu = min(affinity)
        with tempfile.TemporaryDirectory(prefix="numeric-overlap-test-", dir=ROOT / "experiments") as tmp:
            first = second = None
            try:
                first = Client("echo_test_only", {"offline_test": True},
                    directory=Path(tmp) / "first", source_sha256=sources(), cpu=cpu, timeout_s=15)
                second = Client("echo_test_only", {"offline_test": True},
                    directory=Path(tmp) / "second", source_sha256=sources(), cpu=cpu, timeout_s=15)
                value = {"frame": 2, "payload": b"x" * 100_000}
                second.begin("ECHO", value)
                self.assertIsNotNone(second.pending)

                first.process.kill()
                first.process.wait(timeout=2)
                with self.assertRaises(Exception):
                    first.call("ECHO", {"frame": 1})
                self.assertTrue(first.failed)
                with self.assertRaises(ValueError):
                    first.begin("ECHO", {"frame": 1})

                self.assertEqual(second.finish(), value)
                self.assertEqual(second.close(), {})
                self.assertEqual(second.process.returncode, 0)
                self.assertEqual(os.sched_getaffinity(0), affinity)
            finally:
                for client in (first, second):
                    if client is not None and not client.closed:
                        client.abort()
                self.assertTrue(first is None or first.process.poll() is not None)
                self.assertTrue(second is None or second.process.poll() is not None)


if __name__ == "__main__":
    unittest.main()
