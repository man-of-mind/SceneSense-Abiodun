"""CPU-only contracts for the prospective numerical edge adapter.

No live radio, CARLA, container or named socket is launched here.  A fake
scorer client tests edge orchestration; scorer semantics have separate tests.
"""
import copy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import stat
from types import SimpleNamespace
import tempfile
import time
import unittest
import uuid

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from rl_agent.splitfusion_minimal_quality_scorer_v1.scorer import NumericalScorer
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP

from . import edge_live as L, live_entries as ENTRY


def ticket(case, target=None):
    target = case["target"] if target is None else target
    capture = C.thaw(target.metadata["capture_identity"])
    action = C.thaw(target.metadata["action_identity"])
    envelope = SimpleNamespace(**action, capture_timestamp_ns=capture["capture_timestamp_ns"])
    return E.EvaluationTicketV2(envelope=envelope,
        context=case["expected"]["frame_context"],
        records=tuple(case["prediction_kwargs"]["records"]),
        predicted_mask=case["prediction_kwargs"]["labels"],
        gt_identity=capture, enqueued_wall_ns=time.time_ns())


class _FakeScorer:
    def __init__(self, response):
        self.response, self.calls = response, []

    def call(self, op, value):
        self.calls.append((op, value))
        if op == "SCORE":
            return self.response
        return True


class EdgeLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = NumericalScorer()
        cls.case = fixture(cls.scorer)
        cls.spec, _ = quality_bridge()

    def inbox(self):
        temporary = tempfile.TemporaryDirectory(prefix="numeric-edge-unit-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        return L.NumericEdgeInbox(Path(temporary.name) / "numeric_gt.sock",
                                  max_frames=2, max_bytes=16 * 1024 * 1024)

    def test_private_numeric_evidence_handoff_changes_only_owned_subtree(self):
        with tempfile.TemporaryDirectory(prefix="numeric-handoff-", dir="/tmp") as root:
            profile = Path(root) / "profiler"
            numeric = profile / "numeric_runtime"
            nested = numeric / "archive" / "frame_0001"
            nested.mkdir(parents=True)
            file = nested / "live_clock.json"
            file.write_bytes(b"durable clock bytes")
            scope = profile / "numeric_edge_scope.json"
            scope.write_bytes(b"scope bytes")
            unrelated = profile / "unrelated.txt"
            unrelated.write_bytes(b"unchanged")
            before = {path: (path.read_bytes(), path.stat().st_ino)
                      for path in (file, scope, unrelated)}
            original_unrelated_mode = stat.S_IMODE(unrelated.stat().st_mode)
            report = ENTRY.handoff_numeric_evidence(numeric, scope,
                publisher_uid=os.geteuid(), publisher_gid=os.getegid())
            self.assertEqual(report, {"files": 2, "directories": 3})
            for path in (file, scope):
                self.assertEqual((path.read_bytes(), path.stat().st_ino), before[path])
                self.assertEqual((path.stat().st_uid, path.stat().st_gid),
                                 (os.geteuid(), os.getegid()))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            for path in (numeric, numeric / "archive", nested):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
            self.assertEqual((unrelated.read_bytes(), unrelated.stat().st_ino),
                             before[unrelated])
            self.assertEqual(stat.S_IMODE(unrelated.stat().st_mode), original_unrelated_mode)

    def test_handoff_rejects_symlink_and_never_touches_external_target(self):
        with tempfile.TemporaryDirectory(prefix="numeric-handoff-link-", dir="/tmp") as root:
            profile = Path(root) / "profiler"
            numeric = profile / "numeric_runtime"
            numeric.mkdir(parents=True)
            outside = Path(root) / "outside.txt"
            outside.write_bytes(b"keep")
            (numeric / "escape").symlink_to(outside)
            scope = profile / "numeric_edge_scope.json"
            scope.write_bytes(b"scope")
            with self.assertRaisesRegex(ValueError, "symlink or special"):
                ENTRY.handoff_numeric_evidence(numeric, scope,
                    publisher_uid=os.geteuid(), publisher_gid=os.getegid())
            self.assertEqual(outside.read_bytes(), b"keep")

    def test_only_ordered_edge_clean_system_exit_is_normalized(self):
        self.assertEqual(ENTRY._run_edge_ordered(lambda args: 7, []), 7)
        self.assertEqual(ENTRY._run_edge_ordered(lambda args: (_ for _ in ()).throw(
            SystemExit(0)), []), 0)
        with self.assertRaises(SystemExit) as caught:
            ENTRY._run_edge_ordered(lambda args: (_ for _ in ()).throw(SystemExit(2)), [])
        self.assertEqual(caught.exception.code, 2)

    def test_actual_ready_packet_and_native_prediction_match_full_ticket(self):
        case = self.case
        inbox = self.inbox()
        target = case["target"]
        reply = inbox._admit(dict(schema=L.SCHEMA, op="TARGET",
            identity_bytes=target.identity_bytes, packet=R.encode_target(target),
            semantic_sha256=target.semantic_sha256))
        self.assertEqual(reply["identity_sha256"], C.sha256(target.identity_bytes))
        frame = ticket(case)
        key, row = inbox.matching_target(frame)
        self.assertEqual(key, target.identity_bytes)
        self.assertEqual(row["semantic_sha256"], target.semantic_sha256)
        prediction = L.prediction_for_ticket(frame)
        self.assertEqual(prediction["identity"], case["prediction_kwargs"]["identity"])
        self.assertEqual(prediction["labels"], case["prediction_kwargs"]["labels"].tobytes())
        with self.assertRaises(L.NumericEdgeError):
            inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=key,
                packet=R.encode_target(target), semantic_sha256=target.semantic_sha256))

    def test_changed_frame_or_action_cannot_pick_another_target(self):
        inbox = self.inbox()
        target = self.case["target"]
        inbox._admit(dict(schema=L.SCHEMA, op="TARGET",
            identity_bytes=target.identity_bytes, packet=R.encode_target(target),
            semantic_sha256=target.semantic_sha256))
        frame = ticket(self.case)
        other = replace(frame, gt_identity=dict(frame.gt_identity,
            frame_id=frame.gt_identity["frame_id"] + 1))
        self.assertIsNone(inbox.matching_target(other))
        wrong = dict(schema=L.SCHEMA, op="TARGET", identity_bytes=target.identity_bytes,
            packet=R.encode_target(target), semantic_sha256="0" * 64)
        another = self.inbox()
        with self.assertRaises(L.NumericEdgeError):
            another._admit(wrong)

    def test_get_result_returns_exact_durable_proof_without_duplicate_identity(self):
        inbox = self.inbox()
        target = self.case["target"]
        key = target.identity_bytes
        inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=key,
            packet=R.encode_target(target), semantic_sha256=target.semantic_sha256))
        pending = inbox._admit(dict(schema=L.SCHEMA, op="GET_RESULT", identity_bytes=key))
        self.assertEqual(pending["status"], "PENDING")
        final = dict(status="SCORED_DURABLE", identity_sha256=C.sha256(key),
                     scorer_durable_proof={"status": "DURABLE", "sidecar_sha256": "a"*64})
        inbox.result(key, final)
        reply = inbox._admit(dict(schema=L.SCHEMA, op="GET_RESULT", identity_bytes=key))
        self.assertEqual(reply["identity_sha256"], C.sha256(key))
        self.assertEqual(reply["scorer_durable_proof"], final["scorer_durable_proof"])

    def test_second_unscored_poll_waits_until_archive_worker_completes(self):
        inbox = self.inbox()
        target = self.case["target"]
        key = target.identity_bytes
        inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=key,
            packet=R.encode_target(target), semantic_sha256=target.semantic_sha256))
        clock = C.canonical({"schema": "clock-test"})
        inbox.result(key, dict(status="UNSCORED_REQUIRED",
                               clock_descriptor_bytes=clock, reason="FEEDBACK_TIMEOUT"))
        self.assertEqual(inbox._admit(dict(schema=L.SCHEMA, op="GET_RESULT",
            identity_bytes=key))["status"], "UNSCORED_REQUIRED")
        inbox._admit(dict(schema=L.SCHEMA, op="UNSCORED_FILES", identity_bytes=key,
            files={"live_clock.json": clock, "unscored_terminal.json": b"{}"}))
        self.assertEqual(inbox._admit(dict(schema=L.SCHEMA, op="GET_RESULT",
            identity_bytes=key))["status"], "PENDING")
        inbox.replace_unscored_result(key, dict(status="UNSCORED_DURABLE",
            receipt={"status": "COMPLETE"}, relative_prefix="numeric_gt/frame_10/"))
        self.assertEqual(inbox._admit(dict(schema=L.SCHEMA, op="GET_RESULT",
            identity_bytes=key))["status"], "UNSCORED_DURABLE")

    def test_actual_edge_terminal_admits_unscored_retirement_but_not_duplicate(self):
        inbox = self.inbox()
        target = self.case["target"]
        inbox._admit(dict(schema=L.SCHEMA, op="TARGET",
            identity_bytes=target.identity_bytes, packet=R.encode_target(target),
            semantic_sha256=target.semantic_sha256))
        capture = C.thaw(target.metadata["capture_identity"])
        action = C.thaw(target.metadata["action_identity"])
        document = MP.build_run4_edge_terminal(run_id=capture["run_id"],
            cell_id=capture["cell_id"], stream_id=capture["stream_id"],
            identity=action, capture_timestamp_ns=capture["capture_timestamp_ns"],
            service_deadline_at=1., ack_timeout_at=2.,
            outcome="SUPERSEDED_PENDING", stage="EDGE_PENDING_REPLACED",
            age_ms=10., emit_at=1.5, superseded_by_frame_id=11)
        self.assertTrue(inbox.observe_edge_terminal(document))
        self.assertEqual(inbox.rows_snapshot()[0][1]["edge_terminal"], document)
        with self.assertRaisesRegex(L.NumericEdgeError, "duplicate"):
            inbox.observe_edge_terminal(document)

    def test_actual_timeout_without_score_produces_distinct_unscored_clock(self):
        inbox = self.inbox()
        target = self.case["target"]
        key, semantic = target.identity_bytes, target.semantic_sha256
        inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=key,
            packet=R.encode_target(target), semantic_sha256=semantic))
        boot = "11111111-1111-4111-8111-111111111111"
        base = 1_000_000_000_000
        def event(role, name, offset, pid):
            return dict(role=role, name=name, raw_ns=base+offset,
                        pid=pid, boot_id=boot)
        observed = [event("collector", "owned_tick_snapshot", 10, 101),
            event("producer", "source_stage_received", 50, 102),
            event("ue", "action_open", 100, 104),
            event("ue", "last_datagram_sent", 130, 104),
            event("ue", "feedback_timeout", 170_000_101, 104)]
        producer = dict(schema="splitfusion.numeric_gt.producer_live_events.v1",
            role="producer", pid=102, boot_id=boot,
            ready_identity=C._parse(key), ready_identity_sha256=C.sha256(key),
            semantic_sha256=semantic, clock_domain="CLOCK_MONOTONIC_RAW",
            events=[dict(name="prepare_start", raw_ns=base+110),
                dict(name="source_alignment_observed", raw_ns=base+115),
                dict(name="numeric_ready", raw_ns=base+120),
                dict(name="ready_published", raw_ns=base+135)])
        source = dict(producer_events_bytes=C.canonical(producer),
            numerical_avo_bytes=C.canonical({"avo": True}),
            source_synchronization_bytes=C.canonical({"sync": True}))
        terminal = dict(ue_events=observed, feedback=None, source_export=source)
        result = L.NumericEvaluatorV1._unscored_descriptor(
            None, key, inbox.rows_snapshot()[0][1], terminal, None)
        self.assertEqual(result["status"], "UNSCORED_REQUIRED")
        clock = C._parse(result["clock_descriptor_bytes"])
        self.assertEqual(clock["scoring_state"], "NOT_SCORED_AT_TIMEOUT")
        self.assertEqual(clock["deadline"]["terminal"], "FEEDBACK_TIMEOUT")
        self.assertIsNone(result["fault_feedback"])

    def test_async_scorer_uses_cpu_client_and_unchanged_quality_feedback(self):
        case = self.case
        action = C.thaw(case["target"].metadata["action_identity"])
        action["session_uuid"] = str(uuid.UUID(int=42))
        target = R.admit_payload(case["payload"], **dict(case["expected"],
            expected_action_identity=action))
        inbox = self.inbox()
        inbox._admit(dict(schema=L.SCHEMA, op="TARGET", identity_bytes=target.identity_bytes,
            packet=R.encode_target(target), semantic_sha256=target.semantic_sha256))
        frame = Q.join_admitted_inputs(self.scorer, target=case["target"],
                                       prediction=case["prediction"])
        result = self.scorer.score(frame)
        from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
        quality_value = quality.evaluate_exact_quality(self.spec, result.measurement)
        score = dict(measurement=result.measurement,
            q_perc=None if quality_value is None else float(quality_value.q_perc),
            ready_semantic_sha256=target.semantic_sha256,
            identity_sha256=C.sha256(target.identity_bytes))
        client = _FakeScorer(score)
        payloads = []
        spans = []

        class BoundEvaluator(L.NumericEvaluatorV1):
            _binding = dict(inbox=inbox, scorer_client=client,
                archive_client=None, archive_root_index=0,
                sidecar_directory=Path("/tmp"), terminal_timeout_s=1.)

            def _evaluate_with(self, ticket, gt, timing):
                spans.append("numeric_evaluate")
                return super()._evaluate_with(ticket, gt, timing)

            def _emit(self, ticket, feedback, reason, **kwargs):
                spans.append("feedback_emit")
                return super()._emit(ticket, feedback, reason, **kwargs)

        evaluator = BoundEvaluator(spec=self.spec, read_ground_truth=None,
            send=payloads.append, match_distance_m=2., gt_timeout_s=.25)
        evaluator.start()
        evaluator.submit(ticket(case, target))
        deadline = time.monotonic() + 2.
        while not payloads and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(len(payloads), 1, evaluator._failure)
        emitted, reason = W.decode_feedback(payloads[0])
        expected, expected_reason = W.quality_feedback(self.spec, result.measurement,
                                                        ticket(case, target).envelope)
        self.assertEqual((emitted, reason), (expected, expected_reason))
        self.assertEqual([op for op, _ in client.calls],
                         ["REGISTER", "TARGET", "PREDICTION", "SCORE"])
        self.assertEqual(evaluator.counters["scored"], 1)
        self.assertEqual(spans, ["numeric_evaluate", "feedback_emit"])
        with self.assertRaisesRegex(L.NumericEdgeError, "unarchived observed terminals"):
            evaluator.close()
        self.assertFalse(evaluator._thread.is_alive())
        self.assertFalse(evaluator._archive_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
