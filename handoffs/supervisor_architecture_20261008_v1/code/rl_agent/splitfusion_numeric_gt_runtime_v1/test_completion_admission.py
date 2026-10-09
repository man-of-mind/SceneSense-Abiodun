"""CPU-only adversarial ordering/ownership tests; no live performance claim."""
import copy
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import NumericalScorer
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from . import edge_live as L, process as P
from .test_edge_live import ticket


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = fixture(NumericalScorer())

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="numeric-admission-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.inbox = L.NumericEdgeInbox(Path(self.directory.name)/"numeric_gt.sock", max_frames=2)
        self.target = self.case["target"]
        self.key = self.target.identity_bytes
        self.capture = C.thaw(self.target.metadata["capture_identity"])
        self.action = C.thaw(self.target.metadata["action_identity"])
        self.limit = self.capture["capture_timestamp_ns"]/1e9 + .5

    def admit(self):
        return self.inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=self.key,
            packet=R.encode_target(self.target), semantic_sha256=self.target.semantic_sha256))

    def bind(self):
        self.inbox.bind_processing_horizon(run_id=self.capture["run_id"],
            cell_id=self.capture["cell_id"], ack_timeout_s=.5)

    def timeout(self):
        # Owner-envelope validation and real-clock parity are covered by the
        # separate actual-child rehearsal. Here isolate admission state races.
        self.inbox._rows[self.key]["terminal"] = dict(feedback=None, ue_events=[
            dict(role="ue", name="feedback_timeout")])

    def terminal(self):
        return MP.build_run4_edge_terminal(run_id=self.capture["run_id"],
            cell_id=self.capture["cell_id"], stream_id=self.capture["stream_id"],
            identity=self.action, capture_timestamp_ns=self.capture["capture_timestamp_ns"],
            service_deadline_at=1., ack_timeout_at=2., outcome="SUPERSEDED_PENDING",
            stage="EDGE_PENDING_REPLACED", age_ms=10., emit_at=1.5,
            superseded_by_frame_id=self.capture["frame_id"]+1)

    def test_early_terminal_is_owned_and_joined_without_score(self):
        document = self.terminal()
        expected = copy.deepcopy(document)
        self.assertTrue(self.inbox.observe_edge_terminal(document))
        document["stage"] = "MUTATED"
        self.admit()
        self.assertEqual(self.inbox._rows[self.key]["edge_terminal"], expected)
        with self.assertRaisesRegex(L.NumericEdgeError, "no-score"):
            self.inbox.matching_target(ticket(self.case), claim=True)
        summary = self.inbox.close()
        self.assertEqual(summary["edge_terminals_before_target"], 1)
        self.assertEqual(summary["early_edge_terminals_joined"], 1)
        self.assertEqual(summary["edge_terminals_without_target"], 0)

    def test_holds_never_consume_terminal_capacity(self):
        document = self.terminal()
        document["run4_identity"].update(reward_requested=False, frame_kind="POLICY_HOLD")
        for _ in range(10):
            self.assertFalse(self.inbox.observe_edge_terminal(document))
        self.assertEqual(self.inbox._bytes, 0)
        self.assertEqual(self.inbox._edge_terminals, {})

    def test_early_duplicate_and_conflict_fail_closed(self):
        document = self.terminal()
        self.inbox.observe_edge_terminal(document)
        for changed in (document, dict(document, stage="DIFFERENT")):
            with self.assertRaisesRegex(L.NumericEdgeError, "duplicate or conflicting"):
                self.inbox.observe_edge_terminal(changed)

    def test_terminal_capacity_and_bytes_are_bounded(self):
        document = self.terminal()
        self.inbox.max_bytes = len(C.canonical(document))-1
        with self.assertRaisesRegex(L.NumericEdgeError, "capacity"):
            self.inbox.observe_edge_terminal(document)
        self.assertEqual(self.inbox._edge_terminals, {})
        self.inbox.max_bytes = L.MAX_INBOX_BYTES
        for index in range(2):
            current = copy.deepcopy(document)
            current["frame_id"] += index
            current["run4_identity"]["frame_id"] += index
            self.inbox.observe_edge_terminal(current)
        extra = copy.deepcopy(document)
        extra["frame_id"] += 2
        extra["run4_identity"]["frame_id"] += 2
        with self.assertRaisesRegex(L.NumericEdgeError, "capacity"):
            self.inbox.observe_edge_terminal(extra)

    def test_foreign_terminal_capture_cannot_bind_either_order(self):
        for name in ("run_id", "cell_id", "stream_id", "capture_timestamp_ns"):
            for early in (False, True):
                with self.subTest(name=name, early=early):
                    self.inbox = L.NumericEdgeInbox(Path(self.directory.name)/"numeric_gt.sock", max_frames=2)
                    document = self.terminal()
                    document[name] = document[name]+1 if name.endswith("_ns") else "FOREIGN"
                    if early:
                        self.inbox.observe_edge_terminal(document)
                        with self.assertRaisesRegex(L.NumericEdgeError, "capture differs"):
                            self.admit()
                    else:
                        self.admit()
                        with self.assertRaisesRegex(L.NumericEdgeError, "capture differs"):
                            self.inbox.observe_edge_terminal(document)

    def test_foreign_action_terminal_does_not_close_another_target(self):
        document = self.terminal()
        document["run4_identity"]["tensor_seq"] += 1
        self.inbox.observe_edge_terminal(document)
        self.admit()
        self.assertIsNone(self.inbox._rows[self.key]["edge_terminal"])
        self.assertEqual(self.inbox.close()["edge_terminals_without_target"], 1)

    def test_unreceived_target_requires_actual_timeout_and_bound_horizon(self):
        self.admit()
        self.timeout()
        self.assertIsNone(self.inbox.close_unadmitted_after_horizon(self.key))
        self.bind()
        with patch.object(L.time, "time", return_value=self.limit):
            self.assertIsNone(self.inbox.close_unadmitted_after_horizon(self.key))
        self.inbox._rows[self.key]["terminal"] = None
        with patch.object(L.time, "time", return_value=self.limit+1):
            self.assertIsNone(self.inbox.close_unadmitted_after_horizon(self.key))
            self.timeout()
            proof = self.inbox.close_unadmitted_after_horizon(self.key)
        self.assertFalse(proof["score_exists"])
        self.assertFalse(proof["verified_processing_admitted"])
        self.assertEqual(proof["ready_identity_sha256"], C.sha256(self.key))
        self.assertIs(proof, self.inbox.close_unadmitted_after_horizon(self.key))

    def test_inflight_before_target_is_not_retired_as_lost(self):
        self.bind()
        with patch.object(L.time, "time", return_value=self.limit):
            self.inbox.reserve_processing(self.capture, self.action)
        self.admit()
        self.timeout()
        with patch.object(L.time, "time", return_value=self.limit+100):
            self.assertIsNone(self.inbox.close_unadmitted_after_horizon(self.key))
        self.inbox.observe_evaluation_submission(ticket(self.case))
        self.assertIsNotNone(self.inbox.matching_target(ticket(self.case), claim=True))
        with self.assertRaisesRegex(L.NumericEdgeError, "reserved"):
            self.inbox.observe_edge_terminal(self.terminal())

    def test_duplicate_terminal_cannot_steal_pre_target_processing_owner(self):
        self.bind()
        with patch.object(L.time, "time", return_value=self.limit):
            self.inbox.reserve_processing(self.capture, self.action)
        with self.assertRaisesRegex(L.NumericEdgeError, "reserved"):
            self.inbox.observe_edge_terminal(self.terminal())
        self.admit()
        self.inbox.observe_evaluation_submission(ticket(self.case))
        with self.assertRaisesRegex(L.NumericEdgeError, "reserved"):
            self.inbox.observe_edge_terminal(self.terminal())
        self.assertIsNotNone(self.inbox.matching_target(ticket(self.case), claim=True))

    def test_genuine_processor_expiry_allows_explicit_no_score_terminal(self):
        self.bind()
        with patch.object(L.time, "time", return_value=self.limit):
            self.inbox.reserve_processing(self.capture, self.action)
        self.inbox.processing_expired_before_evaluation(self.capture, self.action)
        self.assertTrue(self.inbox.observe_edge_terminal(self.terminal()))
        self.admit()
        with self.assertRaises(L.NumericEdgeError):
            self.inbox.observe_evaluation_submission(ticket(self.case))
        self.assertEqual(self.inbox._rows[self.key]["edge_terminal"], self.terminal())

    def test_reserved_expiry_terminal_with_foreign_capture_refused_before_target(self):
        self.bind()
        with patch.object(L.time, "time", return_value=self.limit):
            self.inbox.reserve_processing(self.capture, self.action)
        self.inbox.processing_expired_before_evaluation(self.capture, self.action)
        with self.assertRaisesRegex(L.NumericEdgeError, "capture differs"):
            self.inbox.observe_edge_terminal(dict(self.terminal(), stream_id="OTHER"))

    def test_tombstone_refuses_late_compute_even_after_wall_clock_rewinds(self):
        self.bind(); self.admit(); self.timeout()
        with patch.object(L.time, "time", return_value=self.limit+1):
            self.inbox.close_unadmitted_after_horizon(self.key)
        with patch.object(L.time, "time", return_value=self.limit-1):
            with self.assertRaises(B.DeadlineExpired):
                self.inbox.reserve_processing(self.capture, self.action)
        with self.assertRaisesRegex(L.NumericEdgeError, "closed/no-score"):
            self.inbox.observe_evaluation_submission(ticket(self.case))
        # Real delayed stale-frame control is retained without reopening it.
        self.inbox.result(self.key, dict(status="UNSCORED_REQUIRED"))
        self.assertTrue(self.inbox.observe_edge_terminal(self.terminal()))

    def test_outside_deadline_guard_race_rechecked_before_reservation(self):
        self.bind(); self.admit(); self.timeout()
        with patch.object(L.time, "time", return_value=self.limit+1):
            with self.assertRaises(B.DeadlineExpired):
                self.inbox.reserve_processing(self.capture, self.action)
            self.assertEqual(self.inbox._processing, {})
            self.assertIsNotNone(self.inbox.close_unadmitted_after_horizon(self.key))

    def test_live_submit_without_processor_and_rebinding_refused(self):
        self.bind(); self.admit()
        with self.assertRaisesRegex(L.NumericEdgeError, "reservation"):
            self.inbox.observe_evaluation_submission(ticket(self.case))
        with self.assertRaisesRegex(L.NumericEdgeError, "changed"):
            self.inbox.bind_processing_horizon(run_id=self.capture["run_id"],
                cell_id=self.capture["cell_id"], ack_timeout_s=.6)

    def test_same_action_cannot_be_rebound_to_another_plan(self):
        self.admit()
        expected = dict(self.case["expected"])
        expected["plan_sha256"] = "7" * 64
        other = R.admit_payload(self.case["payload"], **expected)
        with self.assertRaisesRegex(L.NumericEdgeError, "one action"):
            self.inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=other.identity_bytes,
                packet=R.encode_target(other), semantic_sha256=other.semantic_sha256))

    def test_waiting_expiry_thread_sees_reserved_owner_after_lock_release(self):
        self.bind(); self.admit(); self.timeout()
        waiting = threading.Event()
        answer = []
        def expire():
            waiting.set()
            answer.append(self.inbox.close_unadmitted_after_horizon(self.key))
        with self.inbox._lock:
            thread = threading.Thread(target=expire)
            thread.start()
            self.assertTrue(waiting.wait(1))
            with patch.object(L.time, "time", return_value=self.limit):
                self.inbox.reserve_processing(self.capture, self.action)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(answer, [None])


class ProcessorCompositionTests(unittest.TestCase):
    def setUp(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import (
            Pipeline, processor, CONTRACT)
        self.fixture = Pipeline().step()
        self.old = processor()
        self.directory = tempfile.TemporaryDirectory(prefix="numeric-process-composition-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.inbox = L.NumericEdgeInbox(Path(self.directory.name)/"numeric_gt.sock", max_frames=2)
        self.arguments = dict(contract=CONTRACT, run_id="run", cell_id="cell",
            service_deadline_s=.1, ack_timeout_s=.5, compute=self.old._compute)
        self.limit = self.fixture.envelope.capture_timestamp_ns/1e9+.5
        class Bound(L.NumericProcessorV1):
            _inbox = self.inbox
        self.bound_class = Bound

    def test_real_sfd4_success_preserves_map_and_evaluation_inputs(self):
        retained = self.old.process(self.fixture.wire, edge_timing={})
        new = self.bound_class(**self.arguments)
        with patch.object(L.time, "time", return_value=self.limit):
            actual = new.process(self.fixture.wire, edge_timing={})
        self.assertEqual(C.canonical(retained.update), C.canonical(actual.update))
        for field in ("envelope", "context", "records", "gt_identity"):
            self.assertEqual(getattr(retained.evaluation, field), getattr(actual.evaluation, field))
        self.assertTrue(np.array_equal(retained.evaluation.predicted_mask,
                                       actual.evaluation.predicted_mask))
        measurement = {"frame_id": self.fixture.envelope.frame_id,
                       "loc_vehicle_eligible_gt": 0, "loc_person_eligible_gt": 0}
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import SPEC
        self.assertEqual(L.W.quality_feedback(SPEC, measurement, retained.envelope),
                         L.W.quality_feedback(SPEC, measurement, actual.envelope))
        self.inbox.observe_evaluation_submission(actual.evaluation)

    def test_real_process_deadline_failure_permits_only_its_exact_terminal(self):
        def expired(**kwargs):
            raise B.DeadlineExpired("fixture_tail", capture_timestamp_ns=
                self.fixture.envelope.capture_timestamp_ns, age_ms=501.)
        new = self.bound_class(**dict(self.arguments, compute=expired))
        with patch.object(L.time, "time", return_value=self.limit):
            with self.assertRaises(B.DeadlineExpired):
                new.process(self.fixture.wire, edge_timing={})
        document = new.terminal(self.fixture.wire, outcome=L.E.DP.OUTCOME_STALE_BEFORE_MAP,
                                stage="fixture_tail")
        self.assertEqual(document["run4_identity"], MP.run4_identity(self.fixture.envelope))
        self.assertEqual(len(self.inbox._edge_terminals), 1)

    def test_real_process_error_marks_infrastructure_failure_not_expiry(self):
        def fail(**kwargs):
            raise RuntimeError("injected decode failure")
        new = self.bound_class(**dict(self.arguments, compute=fail))
        with patch.object(L.time, "time", return_value=self.limit):
            with self.assertRaisesRegex(RuntimeError, "decode failure"):
                new.process(self.fixture.wire, edge_timing={})
        self.assertIn("injected decode failure", self.inbox._failure)
        self.assertEqual(self.inbox._admission_expiries, {})
        self.assertEqual(self.inbox._edge_terminals, {})


class WorkerIdleTests(unittest.TestCase):
    def test_idle_over_120s_does_not_start_partial_message_timeout(self):
        sock = object()
        # Virtual passage exceeds both old60s and coordinator120s. Only the
        # read AFTER readability gets a bounded deadline; never wait live here.
        with patch.object(P.os, "getppid", return_value=123), \
             patch.object(P.select, "select", side_effect=[([], [], [])]*130 + [([sock], [], [])]) as polling, \
             patch.object(P.time, "monotonic", return_value=130), \
             patch.object(P.I, "receive", return_value={"command": "CLOSE"}) as receive:
            self.assertEqual(P.receive_parent_command(sock, 123), {"command": "CLOSE"})
        self.assertEqual(polling.call_count, 131)
        receive.assert_called_once_with(sock, deadline=190)

    def test_parent_loss_or_readable_eof_still_fail(self):
        with patch.object(P.os, "getppid", return_value=999):
            with self.assertRaisesRegex(ValueError, "parent changed"):
                P.receive_parent_command(object(), 123)
        sock = object()
        with patch.object(P.os, "getppid", return_value=123), \
             patch.object(P.select, "select", return_value=([sock], [], [])), \
             patch.object(P.I, "receive", side_effect=EOFError("closed")):
            with self.assertRaises(EOFError):
                P.receive_parent_command(sock, 123)
