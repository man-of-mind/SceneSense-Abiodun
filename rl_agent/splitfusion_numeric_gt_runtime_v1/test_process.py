import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .process import Client, Engine, ROOT, LIVE_PROCESS_CONTRACT
from .producer import LiveSourceExport


def sources():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py") if p.name in ("ipc.py", "process.py", "__init__.py")}


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="numeric-ipc-test-", dir=ROOT / "experiments")
        self.cpu = min(os.sched_getaffinity(0))
        self.client = None

    def tearDown(self):
        if self.client is not None and not self.client.closed:
            self.client.abort()
        self.temp.cleanup()

    def launch(self):
        self.client = Client("echo_test_only", {"offline_test": True}, directory=Path(self.temp.name) / "child",
                             source_sha256=sources(), cpu=self.cpu, timeout_s=15)
        return self.client

    def test_real_child_roundtrip_restore_and_bounded_single_inflight(self):
        affinity = os.sched_getaffinity(0)
        c = self.launch()
        self.assertNotEqual(c.ready["pid"], os.getpid())
        self.assertEqual(c.ready["affinity"], [self.cpu])
        self.assertFalse(c.ready["cuda_initialized"])
        value = dict(identity={"frame": 12}, payload=b"x" * 700000)
        c.begin("ECHO", value)
        with self.assertRaises(ValueError):
            c.begin("ECHO", value)
        self.assertEqual(c.finish(), value)
        self.assertEqual(c.close(), {})
        self.assertEqual(c.process.returncode, 0)
        self.assertEqual(os.sched_getaffinity(0), affinity)

    def test_child_death_invalidates_no_retry_and_reaped(self):
        c = self.launch()
        c.process.kill(); c.process.wait(timeout=2)
        with self.assertRaises(Exception):
            c.call("ECHO", {"x": 1})
        self.assertTrue(c.failed)
        with self.assertRaises(ValueError):
            c.begin("ECHO", {})
        c.abort()
        self.assertIsNotNone(c.process.returncode)

    def test_engine_error_no_successful_close(self):
        c = self.launch()
        with self.assertRaises(ValueError):
            c.call("UNKNOWN", {})
        self.assertTrue(c.failed)
        with self.assertRaises(ValueError):
            c.close()
        self.assertTrue(c.closed)

    def test_explicit_writable_attempt_root_supports_readonly_code_mount(self):
        with tempfile.TemporaryDirectory(prefix="numeric-edge-state-", dir="/tmp") as owned:
            destination = Path(owned) / "scorer_worker"
            client = Client("echo_test_only", {"offline_test": True}, directory=destination,
                writable_attempt_root=owned, source_sha256=sources(), cpu=self.cpu, timeout_s=15)
            try:
                self.assertEqual(client.call("ECHO", {"mounted_code": "read_only"}),
                                 {"mounted_code": "read_only"})
                self.assertEqual(client.close(), {})
                self.assertTrue((destination / "worker.log").is_file())
            finally:
                if not client.closed:
                    client.abort()

    def test_foreign_or_symbolic_writable_root_refused(self):
        with tempfile.TemporaryDirectory(prefix="numeric-edge-state-", dir="/tmp") as owned:
            link = Path(owned) / "alias"
            link.symlink_to(owned, target_is_directory=True)
            with self.assertRaises(ValueError):
                Client("echo_test_only", {"offline_test": True}, directory=link / "child",
                    writable_attempt_root=link, source_sha256=sources(), cpu=self.cpu)

    def test_live_source_dispatch_keeps_exact_identity_and_primitive_bytes(self):
        exported = LiveSourceExport("live_source.v1", b"identity", "a" * 64,
            b"{}", b"{}", b"{}", b"{}", 16)
        observed = []
        engine = Engine.__new__(Engine)
        engine.role, engine.tokens = "producer", {1: object()}
        engine.engine = SimpleNamespace(export_live_source=lambda token, identity, semantic:
            (observed.append((token, identity, semantic)) or exported))
        result = engine.call("EXPORT_LIVE_SOURCE", dict(job=1, identity_bytes=b"identity",
                                                        semantic_sha256="a" * 64))
        self.assertEqual(observed, [(engine.tokens[1], b"identity", "a" * 64)])
        self.assertEqual(result["identity_bytes"], exported.identity_bytes)
        self.assertEqual(result["source_synchronization_bytes"], exported.source_synchronization_bytes)
        self.assertEqual(result["producer_events_bytes"], exported.producer_events_bytes)

    def test_scored_live_archive_dispatch_keeps_terminal_and_receipt_separate(self):
        calls = []
        engine = Engine.__new__(Engine)
        engine.role = "scorer"
        engine.engine = SimpleNamespace(
            attach_live_archive=lambda key, **kwargs:
                (calls.append(("attach", key, kwargs)) or {"terminal": "FEEDBACK_TIMEOUT"}),
            confirm_live_archive=lambda key, **kwargs:
                (calls.append(("receipt", key, kwargs)) or
                    {"status": "COMPLETION_SIDECAR_PENDING_DURABILITY"}),
            confirm_live_completion_durable=lambda key, proof:
                (calls.append(("durable", key, proof)) or {"status": "DURABLE"}))
        attached = engine.call("ATTACH_LIVE_ARCHIVE",
            dict(key=b"full-identity", arguments={"clock_events": []}))
        received = engine.call("LIVE_ARCHIVE_DURABLE",
            dict(key=b"full-identity", receipt={"status": "COMPLETE"},
                 relative_prefix="frame_1/live/", archive_submitted_event={"raw_ns": 10},
                 durable_completion_event={"raw_ns": 20}))
        completed = engine.call("LIVE_COMPLETION_DURABLE",
            dict(key=b"full-identity", proof={"sha256": "a" * 64}))
        self.assertEqual(attached["terminal"], "FEEDBACK_TIMEOUT")
        self.assertEqual(received["status"], "COMPLETION_SIDECAR_PENDING_DURABILITY")
        self.assertEqual(completed["status"], "DURABLE")
        self.assertEqual(calls[0], ("attach", b"full-identity", {"clock_events": []}))
        self.assertEqual(calls[1][2]["relative_prefix"], "frame_1/live/")
        self.assertEqual(calls[1][2]["durable_completion_event"], {"raw_ns": 20})
        self.assertEqual(calls[2], ("durable", b"full-identity", {"sha256": "a" * 64}))

    def test_live_engine_requires_explicit_clock_and_release_contract(self):
        with patch("rl_agent.splitfusion_numeric_gt_runtime_v1.producer.ProducerEngine") as constructor:
            for config in ({}, {"offline_replay": False},
                           {"offline_replay": False, "live_process_contract": {"deadline_ns": 200000000}}):
                with self.assertRaises(ValueError):
                    Engine("producer", config)
            constructor.assert_not_called()
            Engine("producer", dict(offline_replay=False, live_process_contract=LIVE_PROCESS_CONTRACT))
            constructor.assert_called_once_with(offline_replay=False)

    def test_live_retirement_dispatch_removes_only_completed_token(self):
        calls = []
        engine = Engine.__new__(Engine)
        engine.role, engine.tokens = "producer", {1: object(), 2: object()}
        first, second = engine.tokens[1], engine.tokens[2]
        engine.engine = SimpleNamespace(complete_live_scored=lambda *args:
            (calls.append(args) or {"status": "RETIRED"}),
            prepare_live_unscored_terminal=lambda *args, **kwargs: SimpleNamespace(
                identity_bytes=b"identity2", semantic_sha256="b" * 64,
                files={"unscored_terminal.json": b"{}"}, terminal_sha256="c" * 64),
            complete_live_unscored=lambda *args, **kwargs:
                (calls.append((args, kwargs)) or {"status": "RETIRED"}))
        engine.call("COMPLETE_LIVE_SCORED", dict(job=1, identity_bytes=b"identity1",
                    semantic_sha256="a" * 64, scorer_proof={"status": "DURABLE"}))
        self.assertEqual(set(engine.tokens), {2})
        result = engine.call("PREPARE_LIVE_UNSCORED_TERMINAL", dict(job=2,
            identity_bytes=b"identity2", semantic_sha256="b" * 64,
            clock_descriptor_bytes=b"clock", reason="EDGE_PENDING_REPLACED"))
        self.assertEqual(set(engine.tokens), {2})
        self.assertEqual(result["terminal_sha256"], "c" * 64)
        engine.call("COMPLETE_LIVE_UNSCORED", dict(job=2, identity_bytes=b"identity2",
            semantic_sha256="b" * 64, terminal_sha256="c" * 64,
            receipt={"status": "COMPLETE"}, relative_prefix="frame_2"))
        self.assertEqual(engine.tokens, {})
        self.assertIs(calls[0][0], first)
        self.assertIs(calls[1][0][0], second)


if __name__ == "__main__":
    unittest.main()
