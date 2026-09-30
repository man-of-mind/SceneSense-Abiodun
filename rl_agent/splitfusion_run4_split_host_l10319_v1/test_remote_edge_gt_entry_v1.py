"""CPU/offline tests for the remote edge GT listener and evaluator seam."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import uuid

from . import gt_transport as GT
from . import remote_edge_gt_entry_v1 as RGT
from . import remote_edge_lifecycle_v1 as E
from .test_remote_edge_lifecycle_contract_v1 import Fixture, measured_binding


def ticket() -> tuple[SimpleNamespace, GT.GtTransportIdentityV1]:
    expected = GT.GtTransportIdentityV1(
        run_id="run4-live", cell_id="a71__favorable_stable",
        stream_id="ue288_a71__favorable_stable",
        session_uuid=str(uuid.UUID("12345678-1234-5678-1234-567812345678")),
        controller_lineage_sha256="a" * 64,
        decision_seq=4, ticket_seq=2, frame_id=1312, tensor_seq=9,
        capture_timestamp_ns=123_456_789_000,
        action_id=71, profile_id="split_ae32_uint4_q9800")
    envelope = SimpleNamespace(
        session_uuid=expected.session_uuid,
        controller_lineage_sha256=expected.controller_lineage_sha256,
        decision_seq=expected.decision_seq, ticket_seq=expected.ticket_seq,
        frame_id=expected.frame_id, tensor_seq=expected.tensor_seq,
        capture_timestamp_ns=expected.capture_timestamp_ns,
        anchor_action_id=expected.action_id)
    context = SimpleNamespace(
        stream_id=expected.stream_id, frame_id=expected.frame_id,
        capture_timestamp_ns=expected.capture_timestamp_ns)
    return SimpleNamespace(envelope=envelope, context=context,
                           gt_identity=expected.gt_identity()), expected


class _ControlledServer:
    """Socket-shaped acceptor with no real network side effects."""

    def __init__(self) -> None:
        self.bound = None
        self.closed = threading.Event()
        self.fail = threading.Event()

    def setsockopt(self, *_args) -> None:
        pass

    def bind(self, endpoint) -> None:
        self.bound = endpoint

    def listen(self, _depth: int) -> None:
        pass

    def settimeout(self, _seconds: float) -> None:
        pass

    def accept(self):
        while not self.fail.wait(0.005):
            if self.closed.is_set():
                raise OSError("closed")
        if self.closed.is_set():
            raise OSError("closed")
        raise OSError("injected accept failure")

    def shutdown(self, _how: int) -> None:
        self.closed.set()
        self.fail.set()

    def close(self) -> None:
        self.closed.set()
        self.fail.set()


def _listener(root: Path, server: _ControlledServer) -> RGT.PersistentGtIngressListenerV1:
    evidence = root / "evidence"
    evidence.mkdir()
    return RGT.PersistentGtIngressListenerV1(
        run_id="run4-live", cell_id="a71__favorable_stable",
        evidence_dir=evidence, bind_host="192.168.70.140",
        advertised_host="192.168.70.140", port=RGT.REGISTERED_GT_PORT,
        socket_timeout_s=0.1, expectation_timeout_s=0.1,
        ready_evidence=root / "ready.json", final_evidence=root / "final.json",
        socket_factory=lambda *_args: server)


class ListenerTests(unittest.TestCase):
    def test_listener_authorizes_exact_ticket_and_stops_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = _ControlledServer()
            listener = _listener(root, server)
            listener.start()
            actual_ticket, expected = ticket()
            self.assertEqual(listener.authorize(actual_ticket), expected)
            listener.registry.await_authorized(expected, 0)
            listener.mark_edge_ready_checked()
            final = listener.stop()
            self.assertEqual(final["status"], "STOPPED")
            self.assertFalse(final["thread_alive"])
            self.assertEqual(final["counters"]["authorized"], 1)
            self.assertTrue(final["edge_ready_health_checked"])
            self.assertEqual(listener.stop(), final)
            ready = json.loads((root / "ready.json").read_text())
            self.assertFalse(ready["cross_host_clock_subtraction"])
            self.assertEqual(ready["policy_deadline_clock_owner"], "W10275")

    def test_listener_thread_failure_is_fail_closed_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = _ControlledServer()
            listener = _listener(root, server)
            listener.start()
            server.fail.set()
            deadline = time.monotonic() + 1.0
            while listener._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.005)
            with self.assertRaises(RGT.RemoteEdgeGtError):
                listener.require_healthy()
            final = listener.stop()
            self.assertEqual(final["status"], "FAILED")
            self.assertIn("injected accept failure", final["failure"])
            self.assertFalse(final["thread_alive"])

    def test_no_cross_host_clock_subtraction_exists_in_edge_seam(self) -> None:
        tree = ast.parse(Path(RGT.__file__).read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(node, ast.BinOp)
                             and isinstance(node.op, ast.Sub)
                             for node in ast.walk(tree)))
        self.assertNotIn("import time", Path(RGT.__file__).read_text(encoding="utf-8"))


class _RecordingListener:
    def __init__(self, events: list[str], *, healthy: bool = True) -> None:
        self.events = events
        self.healthy = healthy
        self.authorized = []
        self.stopped = 0

    def start(self) -> None:
        self.events.append("listener.start")

    def require_healthy(self) -> None:
        self.events.append("listener.health")
        if not self.healthy:
            raise RGT.RemoteEdgeGtError("injected listener failure")

    def authorize(self, value) -> None:
        self.require_healthy()
        actual = GT.identity_from_phase6(
            run_id="run4-live", cell_id="a71__favorable_stable",
            envelope=value.envelope, context=value.context,
            gt_identity=value.gt_identity)
        self.authorized.append(actual)
        self.events.append("listener.authorize")

    def mark_edge_ready_checked(self) -> None:
        self.require_healthy()
        self.events.append("listener.ready-check")

    def stop(self):
        self.events.append("listener.stop")
        self.stopped += 1
        return {"status": "STOPPED", "thread_alive": False}


class _FakeEvaluator:
    def __init__(self, *, evidence_dir: Path, events: list[str]) -> None:
        self.events = events
        self.counters = {"submitted": 0}
        self.records = []
        self.events.append("base.init")

    def start(self) -> None:
        self.events.append("base.start")

    def submit(self, _ticket) -> None:
        self.events.append("base.submit")
        self.counters["submitted"] += 1

    def close(self, timeout_s=None):
        self.events.append("base.close")
        return {**self.counters, "worker_alive": False, "records": len(self.records)}


def _publish(events: list[str]):
    def implementation(warm, write_ready, *args, **kwargs):
        events.append("prewarm.begin")
        result = warm()
        events.append("prewarm.done")
        write_ready()
        return result
    return implementation


class EvaluatorHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[str] = []
        self.listener = _RecordingListener(self.events)
        self.frozen = SimpleNamespace(Run4EvaluatorV2=_FakeEvaluator)
        self.prewarm = SimpleNamespace(
            publish_ready_after_warmup=_publish(self.events))
        self.hooks = RGT.InstalledGtEvaluatorHooksV1(
            frozen=self.frozen, prewarm=self.prewarm,
            listener_factory=lambda _path: self.listener)
        self.hooks.install()

    def tearDown(self) -> None:
        self.hooks.close_latest()
        self.hooks.restore()

    def test_exact_authorization_precedes_unchanged_evaluator_submit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluator = self.frozen.Run4EvaluatorV2(
                evidence_dir=Path(directory), events=self.events)
            evaluator.start()
            value, expected = ticket()
            evaluator.submit(value)
            self.assertEqual(self.listener.authorized, [expected])
            self.assertLess(self.events.index("listener.authorize"),
                            self.events.index("base.submit"))
            result = evaluator.close()
            self.assertEqual(result["submitted"], 1)
            self.assertLess(self.events.index("base.close"),
                            self.events.index("listener.stop"))
            evaluator.close()
            self.assertEqual(self.events.count("base.close"), 1)
            self.assertEqual(self.events.count("listener.stop"), 1)

    def test_listener_health_is_required_immediately_before_edge_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.frozen.Run4EvaluatorV2(evidence_dir=Path(directory),
                                        events=self.events).start()
            self.prewarm.publish_ready_after_warmup(
                lambda: self.events.append("warm") or {"completed": True},
                lambda: self.events.append("edge.ready"))
            self.assertLess(self.events.index("listener.ready-check"),
                            self.events.index("edge.ready"))
            self.listener.healthy = False
            with self.assertRaises(RGT.RemoteEdgeGtError):
                self.prewarm.publish_ready_after_warmup(
                    lambda: {"completed": True},
                    lambda: self.events.append("forbidden.ready"))
            self.assertNotIn("forbidden.ready", self.events)


class LifecycleBindingTests(unittest.TestCase):
    def test_compose_cli_and_evidence_bind_registered_gt_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            plan = E.build_plan(binding=measured_binding(), paths=fixture.paths,
                                invocation=fixture.invocation)
            command = plan.compose_document["services"][E.SERVICE]["command"]
            self.assertEqual(command[:4], ["python3", "-u", "-m", E.EDGE_MODULE])
            self.assertEqual(command.count("--remote-gt-ready-evidence"), 1)
            parsed, remaining = RGT.parse_remote_gt_args(command[4:])
            self.assertEqual(parsed.remote_gt_ready_evidence,
                             Path(E.GT_READY_DESTINATION))
            self.assertEqual(parsed.remote_gt_final_evidence,
                             Path(E.GT_FINAL_DESTINATION))
            self.assertIn("--edge", remaining)
            self.assertNotIn("--remote-gt-ready-evidence", remaining)
            self.assertEqual(command[command.index("--remote-gt-bind-host") + 1],
                             "192.168.70.140")
            self.assertEqual(command[command.index("--remote-gt-port") + 1], "51015")
            self.assertEqual(plan.as_evidence()["gt_ingress"]["endpoint"],
                             "192.168.70.140:51015")
            ready = {
                "schema": RGT.SCHEMA, "status": "LISTENING",
                "run_id": fixture.invocation.run_id,
                "cell_id": fixture.invocation.cell_id,
                "bind_host": "192.168.70.140",
                "advertised_endpoint": "192.168.70.140:51015",
                "max_tickets": RGT.MAX_GT_TICKETS,
                "socket_timeout_s": E.GT_SOCKET_TIMEOUT_S,
                "expectation_timeout_s": E.GT_EXPECTATION_TIMEOUT_S,
                "cross_host_clock_subtraction": False,
                "policy_deadline_clock_owner": "W10275",
            }
            E.validate_gt_ready_record(ready, plan=plan)
            ready["policy_deadline_clock_owner"] = "L10319"
            with self.assertRaises(E.RemoteEdgeLifecycleError):
                E.validate_gt_ready_record(ready, plan=plan)


if __name__ == "__main__":
    unittest.main()
