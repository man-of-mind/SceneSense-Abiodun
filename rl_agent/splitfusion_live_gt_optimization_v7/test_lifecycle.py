"""Actual retained shutdown code over real CPU packet/scorer/archive scopes.

The real nested shutdown and write_report code objects run with fixture-owned
closures. Model setup, GPU work, radio/map sockets and process-wide signal
registration never run. Two local waiting threads stand in for service loops;
the evaluator, broker, quality/feedback, prediction and pose archives are real.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
from types import CodeType, FunctionType, SimpleNamespace as NS
import unittest
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_live_gt_handoff_v1 import edge as H, broker as B, scoring_packet as P
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_live_parity_repair_v2.runtime import install_prediction_contract
from rl_agent.splitfusion_live_parity_runtime_v1 import pose
from rl_agent.splitfusion_live_parity_runtime_v7.container_check import _ticket_and_target
from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from . import lifecycle as L

ROOT = Path(__file__).resolve().parents[2]
BINDING = "c" * 64


def _cell(value):
    return (lambda: value).__closure__[0]


def _retained_nested(name, bindings):
    codes = [item for item in E.run_run4_edge_service.__code__.co_consts
             if isinstance(item, CodeType) and item.co_name == name]
    if len(codes) != 1:
        raise AssertionError("retained nested service function changed")
    code = codes[0]
    return FunctionType(code, E.__dict__, name,
        closure=tuple(_cell(bindings[key]) for key in code.co_freevars))


class LifecycleTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        original_close = E.Run4EvaluatorV2.close
        with tempfile.TemporaryDirectory(prefix="handoff_shutdown_") as tmp, ExitStack() as guards:
            root = Path(tmp)
            directory, profile = root / "original", root / "profile"
            ticket, expected = _ticket_and_target(directory)
            target = T.read_bundle(directory, expected_identity=ticket.gt_identity, timeout_s=0.)
            metadata = json.loads((target.directory / "inputs.json").read_bytes())
            avo = json.loads(LG.sidecar_path(directory, ticket.gt_identity).read_bytes())
            avo.update(record_kind="AVO_SCORED_FRAME", native_shape=[720, 1280])
            for row in avo["avo"]["qualified"]:
                row["qualified"] = True
            packet = P.encode_packet(identity=ticket.gt_identity, prepared=target.prepared,
                input_metadata=metadata, avo_document=avo, binding=BINDING,
                ready_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), ready_wall_ns=time.time_ns())
            stop, received = threading.Event(), threading.Event()
            threads = [threading.Thread(target=stop.wait, name="offline-service-join-fixture") for _ in range(2)]
            for thread in threads:
                thread.start()
            sent = []
            state = NS(root=root, directory=directory, profile=profile, socket=root / "packet.sock",
                ticket=ticket, expected=expected, packet=packet, stop=stop, threads=threads,
                sent=sent, received=received, worker=None, report_path=root / "edge_report.json")
            guards.enter_context(mock.patch.object(torch.cuda, "init", side_effect=AssertionError("GPU forbidden")))
            guards.enter_context(mock.patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden")))
            try:
                yield state
            finally:
                stop.set()
                for thread in threads:
                    thread.join(timeout=2.)
                    self.assertFalse(thread.is_alive())
                if state.worker is not None and state.worker._thread.is_alive():
                    state.worker.close(timeout_s=2.)
                self.assertIs(E.Run4EvaluatorV2.close, original_close)
                self.assertFalse(torch.cuda.is_initialized())

    def scope(self, f):
        return H.install_edge_handoff(socket_path=f.socket, binding=BINDING, max_frames=2,
            archive_directory=f.profile / "prediction_archive", pose_archive_directory=f.profile / "pose_archive",
            publisher_uid=os.geteuid(), publisher_gid=os.getegid())

    def start_evaluator(self, f, *, score=True):
        def send(payload):
            f.sent.append(payload)
            f.received.set()
        f.worker = E.Run4EvaluatorV2(spec=W.load_run4_quality_spec(ROOT), send=send,
            read_ground_truth=lambda **kw: G.read_ground_truth(f.directory, **kw),
            match_distance_m=3., gt_timeout_s=.3, probe_timeout_s=.002,
            poll_interval_s=.001, evidence_dir=f.directory)
        f.worker.start()
        B.publish(f.socket, f.packet, binding=BINDING)
        if score:
            f.worker.submit(f.ticket)
            self.assertTrue(f.received.wait(2.), "real evaluator did not emit feedback")

    def shutdown(self, f, *, close_owner=None, report_path=None):
        report = _retained_nested("write_report", dict(args=NS(run_id="offline", cell_id="shutdown"),
            counters={"evaluations_submitted": len(f.worker.records)}, evaluator=f.worker,
            failures=[], publisher=NS(published=[]), report_path=report_path or f.report_path))
        result = _retained_nested("shutdown", dict(stop=f.stop, threads=f.threads,
            evaluator=f.worker if close_owner is None else close_owner, write_report=report))
        self.assertIs(result.__code__, L._retained_shutdown_code())
        return result

    def assert_preserved(self, f, state):
        self.assertTrue(state["complete"])
        self.assertEqual(state["errors"], [])
        self.assertFalse(f.socket.exists())
        self.assertTrue(f.stop.is_set())
        self.assertTrue(all(not thread.is_alive() for thread in f.threads))
        self.assertFalse(f.worker._thread.is_alive())
        self.assertEqual(state["broker"]["admitted"], 1)
        self.assertEqual(state["broker"]["consumed"], 1)
        self.assertFalse(state["broker"]["worker_alive"])
        self.assertTrue(state["evaluator_close"]["parity_archive_complete"])
        self.assertTrue(state["pose_state"]["flushed"])
        self.assertTrue(state["prediction_state"]["flushed"])
        self.assertEqual(f.worker.records[0]["measurement"], f.expected.measurement)
        self.assertEqual(len(f.sent), 1)
        report = json.loads(f.report_path.read_bytes())
        self.assertTrue(report["final"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["evaluations"], f.worker.records)
        self.assertEqual(report["evaluator"]["submitted"], report["evaluator"]["emitted"])
        for name in ("prediction_archive", "pose_archive"):
            directory = f.profile / name
            manifest = json.loads((directory / "MANIFEST.json").read_bytes())
            for filename, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((directory / filename).read_bytes()).hexdigest(), digest)
        events = pose.read_archive(f.profile / "pose_archive")["events"]
        self.assertEqual(sum(row["kind"] == "EDGE_TARGET_RECONCILED" for row in events), 1)

    def test_actual_retained_shutdown_normalized_only_after_real_scoring_close_and_report(self):
        with self.fixture() as f:
            with install_prediction_contract(), self.scope(f) as state, ExitStack() as guards:
                for owner, name in ((T, "read_bundle"), (T, "_read"), (LG, "read_avo_bundle")):
                    guards.enter_context(mock.patch.object(owner, name,
                        side_effect=AssertionError("durable scoring fallback forbidden")))
                self.start_evaluator(f)
                shutdown = self.shutdown(f)
                self.assertEqual(L.dispatch_with_retained_shutdown(shutdown, signal.SIGTERM, None,
                    handoff_state=state), 0)
                self.assertFalse(state["complete"], "helper must not fabricate completed broker drain")
                self.assertNotIn("broker", state)
                self.assertEqual(state["retained_shutdown"]["schema"], L.SCHEMA)
            self.assert_preserved(f, state)

    def test_original_unadapted_shutdown_reproduces_v8_false_failure(self):
        with self.fixture() as f:
            with self.assertRaises(SystemExit) as caught:
                with install_prediction_contract(), self.scope(f) as state:
                    self.start_evaluator(f)
                    self.shutdown(f)(signal.SIGTERM, None)
            self.assertEqual(caught.exception.code, 0)
            self.assertFalse(state["complete"])
            self.assertEqual(state["errors"], ["SystemExit: 0"])
            self.assertTrue(state["broker"]["all_consumed"])
            self.assertTrue(state["prediction_state"]["flushed"])
            self.assertTrue(state["pose_state"]["flushed"])

    def test_zero_exit_from_retained_handler_without_close_is_not_normalized(self):
        with self.fixture() as f:
            with self.assertRaises(SystemExit) as caught:
                with install_prediction_contract(), self.scope(f) as state:
                    self.start_evaluator(f)
                    shutdown = self.shutdown(f, close_owner=NS(close=lambda: None))
                    L.dispatch_with_retained_shutdown(shutdown, signal.SIGTERM, None, handoff_state=state)
            self.assertEqual(caught.exception.code, 0)
            self.assertNotIn("retained_shutdown", state)
            self.assertFalse(state["complete"])

    def test_normalized_exit_does_not_hide_unconsumed_broker_packet(self):
        with self.fixture() as f:
            with self.assertRaisesRegex(B.BrokerError, "close/drain/preserve"):
                with install_prediction_contract(), self.scope(f) as state:
                    self.start_evaluator(f, score=False)
                    self.assertEqual(L.dispatch_with_retained_shutdown(self.shutdown(f), signal.SIGTERM, None,
                        handoff_state=state), 0)
            self.assertFalse(state["complete"])
            self.assertEqual(state["broker"]["pending_jobs"], 1)
            self.assertEqual(state["broker"]["consumed"], 0)
            self.assertFalse(state["broker"]["worker_alive"])

    def test_real_final_report_io_failure_is_preserved(self):
        with self.fixture() as f:
            with self.assertRaises(FileNotFoundError):
                with install_prediction_contract(), self.scope(f) as state:
                    self.start_evaluator(f)
                    shutdown = self.shutdown(f, report_path=f.root / "missing" / "report.json")
                    L.dispatch_with_retained_shutdown(shutdown, signal.SIGTERM, None, handoff_state=state)
            self.assertNotIn("retained_shutdown", state)
            self.assertFalse(state["complete"])
            self.assertTrue(any("FileNotFoundError" in error for error in state["errors"]))

    def test_nonzero_exit_after_genuine_close_is_unchanged(self):
        with self.fixture() as f:
            failure = SystemExit(7)
            with self.assertRaises(SystemExit) as caught:
                with install_prediction_contract(), self.scope(f) as state:
                    self.start_evaluator(f)
                    shutdown = self.shutdown(f)
                    def dispatch():
                        try:
                            shutdown(signal.SIGTERM, None)
                        except SystemExit:
                            raise failure
                    L.dispatch_with_retained_shutdown(dispatch, handoff_state=state)
            self.assertIs(caught.exception, failure)
            self.assertFalse(state["complete"])
            self.assertNotIn("retained_shutdown", state)

    def test_foreign_zero_and_other_exceptions_are_unchanged(self):
        for failure in (SystemExit(0), SystemExit(None), SystemExit(False), SystemExit(0.),
                SystemExit("0"), SystemExit(1), RuntimeError("primary error"), KeyboardInterrupt()):
            with self.subTest(failure=repr(failure)):
                def dispatch():
                    raise failure
                with self.assertRaises(type(failure)) as caught:
                    L.dispatch_with_retained_shutdown(dispatch, handoff_state={})
                self.assertIs(caught.exception, failure)

    def test_normal_return_and_arguments_unchanged(self):
        token = object()
        self.assertIs(L.dispatch_with_retained_shutdown(lambda value, *, check: value if check else None,
            token, handoff_state={}, check=True), token)


if __name__ == "__main__":
    unittest.main()
