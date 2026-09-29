"""Phase-6 offline acceptance tests (CPU fakes; no CARLA, OAI, Docker or CUDA).

Covers: exact continuous identity end to end, 72 anchors, legacy protocol
invariance, exact FrameContextV1 round trip into the contextual tail,
fail-closed corruption, authoritative Q_perc agreement, the 170-ms boundary,
late/duplicate feedback, k_min and one reward per decision, fallback
semantics, post-assignment infrastructure faults, clock-domain separation,
the non-vacuous coverage gate, and bounded lifecycle/teardown wiring.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rl_agent.splitfusion_direct_edge_map_v1 import map_ingest as MI
from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

from . import continuous_execution_v2 as X
from . import phase6_decision_engine_v2 as E
from . import phase6_edge_runtime_v2 as ER
from . import phase6_live_child_v2 as CH
from . import phase6_live_runner_v2 as RUN
from . import phase6_map_server_v2 as MS2
from . import phase6_ue_runtime_v2 as U
from . import reward_hold_controller_v2 as R
from . import run4_live_wire_v2 as W
from . import run4_map_protocol_v2 as MP
from . import run4_ue_ledger_v2 as UL
from .test_phase3_continuous_execution_v2 import CONTRACT, FakeCodec, runtimes
from .test_ue_telemetry_provider_v2 import Harness, _valid_tick

ROOT = Path(__file__).resolve().parents[2]
MS = 1_000_000
LINEAGE = hashlib.sha256(b"phase6-test-lineage").hexdigest()
SPEC = W.load_run4_quality_spec(ROOT)


class FakeActor:
    def __init__(self, mode_id=9, q_e4=4321) -> None:
        self.calls = 0
        self.mode_id, self.q_e4 = mode_id, q_e4

    def act(self, features):
        features.as_tuple()               # must be an attested vector
        self.calls += 1
        return SimpleNamespace(mode_id=self.mode_id, q_e4=self.q_e4)


def _scene(h: Harness, *, camera=112.0, radar=0.4) -> U.SceneDescriptorsV2:
    return U.SceneDescriptorsV2(camera_si=camera, radar_p40=radar, camera_status="VALID",
                                radar_status="VALID", source_raw_ns=h.host.raw() - 20 * MS,
                                available_raw_ns=h.host.raw() - 10 * MS)


def _fresh_radio(h: Harness, mcs=24) -> None:
    h.dci(mcs=mcs)
    _valid_tick(h, 3, 1, values=(0, 0, 0))
    _valid_tick(h, 3, 2, values=(0, 0, 0))


class Pipeline:
    def __init__(self, actor=None) -> None:
        self.h = Harness()
        self.actor = actor or FakeActor()
        self.engine = E.Run4DecisionEngineV2(
            contract_=CONTRACT, actor=self.actor, ue_id=self.h.provider.ue_label,
            controller_lineage_sha256=LINEAGE)
        ue, self.edge_rt, _ = runtimes()
        self.pipe = U.Run4FramePipelineV2(
            engine=self.engine, continuous_ue=ue, provider=self.h.provider,
            stream_id="ue288_phase6_test", run_id="run", cell_id="cell",
            clock=self.h._ticking_clock())
        self.frame = 1000

    def step(self, *, fresh=True, scene=None, input_fn=None):
        self.frame += 1
        self.h.host.advance(100 * MS)
        if fresh:
            _fresh_radio(self.h)
        return self.pipe.process(
            frame_id=self.frame, capture_wall_ns=1_790_000_000_000_000_000 + self.frame * 100 * MS,
            ego_pose=(1.0, 2.0, 0.1, 0.0, 90.0, 0.0),
            scene=scene or _scene(self.h), input_7ch=input_fn or (lambda: object()))


def processor(compute=None):
    def fake_compute(*, envelope, context, profile, identity):
        return [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}], np.zeros((4, 4), np.uint8)
    return ER.Run4EdgeProcessorV2(contract=CONTRACT, run_id="run", cell_id="cell",
                                  service_deadline_s=0.1, ack_timeout_s=0.5,
                                  compute=compute or fake_compute)


class IdentityTest(unittest.TestCase):
    def test_every_legal_action_identity_validates_exactly(self) -> None:
        for mode_id in range(12):
            for q_e4 in range(0, 9801):
                profile = CONTRACT.resolve_q_e4(mode_id, q_e4)
                identity = {"session_uuid": str(uuid.UUID(int=1)),
                            "controller_lineage_sha256": LINEAGE, "decision_seq": 0,
                            "ticket_seq": 0, "frame_id": 1, "tensor_seq": 0,
                            "mode_id": mode_id, "q_e4": q_e4,
                            "execution_bundle_sha256": profile.execution_bundle_sha256,
                            "anchor_action_id": profile.action_id, "reward_requested": True,
                            "frame_kind": "POLICY_DECISION"}
                MP.validate_run4_identity(identity, contract=CONTRACT)
        tampered = dict(identity, anchor_action_id=5)
        with self.assertRaises(DP.DirectMapProtocolError):
            MP.validate_run4_identity(tampered, contract=CONTRACT)

    def test_all_72_anchor_identities_unchanged_downstream(self) -> None:
        for mode_id in range(12):
            for anchor in CONTRACT.action_contract.anchors_for_mode(mode_id):
                profile = CONTRACT.resolve_q_e4(mode_id, anchor.q_e4)
                self.assertEqual((profile.action_id, profile.profile_id),
                                 (anchor.action_id, anchor.profile_id))
                doc = {"schema": MP.RUN4_UPDATE_SCHEMA,
                       "run4_identity": {"anchor_action_id": profile.action_id}}
                self.assertEqual(MP.run4_install_document(doc)["action_id"],
                                 str(anchor.action_id))


class ContextAndCorruptionTest(unittest.TestCase):
    def _frame(self):
        p = Pipeline()
        prepared = p.step()
        return p, prepared

    def test_frame_context_round_trip_and_contextual_tail_receipt(self) -> None:
        _p, prepared = self._frame()
        envelope, context = W.unpack_sfd4(prepared.wire)
        self.assertEqual(envelope, prepared.envelope)
        self.assertEqual(context.sequence_id, envelope.tensor_seq)
        received = {}

        class Tail:
            def compute_product(self, c2, metadata):
                received["metadata"] = metadata
                return "WORK"

        counters = SimpleNamespace(frames_attempted=0, frames_completed=0,
                                   ae_decoder_dispatches=0, tail_dispatches=0)
        accepted = []
        runtime = SimpleNamespace(
            _counters=counters,
            _camera_registry=SimpleNamespace(resolve=lambda m, n: (m, n)),
            _context_session=SimpleNamespace(accept=accepted.append),
            _codec=FakeCodec(), _ae_decoders={f: object() for f in X.AE_FAMILIES},
            _tail_device=torch.device("cpu"), _detached_tail=Tail())
        profile = CONTRACT.resolve_q_e4(envelope.mode_id, envelope.q_e4)
        result = ER.run4_compute_on_detached_runtime(
            runtime, envelope=envelope, context=context, profile=profile,
            identity=MP.run4_identity(envelope))
        metadata = received["metadata"]
        self.assertEqual(metadata.frame_context, context)
        self.assertEqual((metadata.sequence_id, metadata.capture_timestamp_ns),
                         (context.sequence_id, context.capture_timestamp_ns))
        self.assertEqual(accepted, [context])
        self.assertEqual(result.work, "WORK")
        self.assertFalse(hasattr(metadata, "action_id"))

    def test_corrupted_context_action_ticket_bundle_fail_closed(self) -> None:
        _p, prepared = self._frame()
        wire = bytearray(prepared.wire)
        for index in (14, 40, 120, len(wire) - 40, len(wire) - 1):
            corrupt = bytearray(wire)
            corrupt[index] ^= 0x01
            with self.assertRaises(W.WireError):
                W.unpack_sfd4(bytes(corrupt))
        env, ctx = W.unpack_sfd4(prepared.wire)
        for change in ({"execution_bundle_sha256": "ab" * 32}, {"q_e4": env.q_e4 + 1},
                       {"anchor_action_id": 3}):
            forged = W.pack_sfd4(dataclasses.replace(env, **change), ctx)
            with self.assertRaises(Exception):
                processor().verify(forged)
        # A resealed forged ticket is structurally valid at the edge, which owns
        # no ticket authority; the UE controller can never attach its feedback.
        forged = W.pack_sfd4(dataclasses.replace(env, ticket_seq=env.ticket_seq + 1), ctx)
        processed = processor().process(forged, edge_timing={})
        feedback, _ = W.quality_feedback(SPEC, {"frame_id": env.frame_id, **{
            f"loc_{c}_eligible_gt": 0 for c in ("vehicle", "person")}}, processed.envelope)
        current = _p.engine.controller.current
        self.assertEqual(_p.engine.on_feedback(
            feedback, receipt_raw_ns=current.action_open_ns + 10 * MS),
            R.FeedbackClass.UNKNOWN_ORPHAN)
        self.assertIsNone(current.resolution)
        with self.assertRaises(W.WireError):
            W.pack_sfd4(env, dataclasses.replace(ctx, frame_id=ctx.frame_id + 1))


class QualityTest(unittest.TestCase):
    def test_authoritative_qperc_agreement_on_grid_rows(self) -> None:
        database = ROOT / ("experiments/splitfusion_hybrid_sac_quality_grid_v1/"
                           "20260918_exact_continuous_q_grid_a1b_full/quality_rows.sqlite3")
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        _p, prepared = ContextAndCorruptionTest()._frame()
        checked = 0
        for (blob,) in connection.execute("SELECT row_json FROM quality_rows LIMIT 400"):
            row = json.loads(blob)
            envelope = dataclasses.replace(prepared.envelope, frame_id=int(row["frame_id"]))
            feedback, reason = W.quality_feedback(SPEC, row, envelope)
            if row["q_perc"] is None:
                self.assertEqual(feedback.kind, "EVALUATOR_FAULT")
                self.assertIs(reason, W.EvaluatorReason.QUALITY_UNDEFINED_NO_ELIGIBLE_GT)
            else:
                self.assertEqual(feedback.q_perc, row["q_perc"])
                checked += 1
        connection.close()
        self.assertGreater(checked, 50)

    def test_live_measurement_is_consistent_with_live_scorer(self) -> None:
        from rl_agent.splitfusion_quality_feedback_probe_v1 import scoring

        pred = np.zeros((72, 128), np.uint8); pred[10:20, 10:40] = 1; pred[40:50, 60:70] = 2
        gt = np.zeros((72, 128), np.uint8); gt[12:22, 10:40] = 1; gt[40:52, 60:70] = 2
        preds = [{"class_name": "vehicle", "world_x": 1.0, "world_y": 1.0}]
        truth = [{"class_name": "vehicle", "world_x": 1.2, "world_y": 1.0},
                 {"class_name": "person", "world_x": 5.0, "world_y": 5.0}]
        m = W.live_measurement(frame_id=3, predicted_mask=pred, ground_truth_mask=gt,
                               predictions=preds, ground_truth_objects=truth,
                               match_distance_m=3.0)
        seg = scoring.score_segmentation(pred, gt)
        self.assertAlmostEqual(m["seg_vehicle_intersection_pixels"]
                               / m["seg_vehicle_union_pixels"], seg["miou_vehicle_iou"])
        self.assertEqual((m["loc_vehicle_tp"], m["loc_person_fn"]), (1, 1))
        self.assertEqual(Q.evaluate_exact_quality(SPEC, m).q_perc,
                         W.quality_feedback(SPEC, m, dataclasses.replace(
                             ContextAndCorruptionTest()._frame()[1].envelope,
                             frame_id=3))[0].q_perc)

    def test_evaluator_only_reward_frames_and_fault_classes(self) -> None:
        sent = []
        evaluator = ER.Run4EvaluatorV2(
            spec=SPEC, send=sent.append, match_distance_m=3.0, gt_timeout_s=0.1,
            read_ground_truth=lambda **kw: (_ for _ in ()).throw(RuntimeError("no gt")))
        _p, prepared = ContextAndCorruptionTest()._frame()
        env, ctx = W.unpack_sfd4(prepared.wire)
        ticket = ER.EvaluationTicketV2(envelope=env, context=ctx, records=(),
                                       predicted_mask=np.zeros((4, 4), np.uint8),
                                       gt_identity={}, enqueued_wall_ns=1)
        evaluator.evaluate(ticket)
        feedback, reason = W.decode_feedback(sent[-1])
        self.assertEqual((feedback.kind, reason),
                         ("EVALUATOR_FAULT", W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE))
        with self.assertRaises(ER.EdgeError):
            evaluator.submit(dataclasses.replace(
                ticket, envelope=dataclasses.replace(env, reward_requested=False)))


class EndToEndOffAnchorTest(unittest.TestCase):
    def test_off_anchor_through_execution_map_evaluator_feedback(self) -> None:
        p = Pipeline(FakeActor(mode_id=9, q_e4=4321))
        prepared = p.step()
        self.assertIs(prepared.plan.kind, E.FrameKind.POLICY_DECISION)
        self.assertIsNone(prepared.identity["anchor_action_id"])
        processed = processor().process(prepared.wire, edge_timing={})
        update = processed.update
        self.assertNotIn("action_id", update)
        self.assertEqual(update["run4_identity"], prepared.identity)
        # Map process: proxy-validated install and Run-4 ACK.
        legacy = MI.protocol
        MI.protocol = MP.MapProtocolProxyV2(DP, contract=CONTRACT)
        installed = []
        try:
            service = MS2.Run4DirectMapIngestServiceV2(
                bind_host="127.0.0.1", bind_port=0, feedback_host="127.0.0.1",
                feedback_port=9, install=lambda d, at: installed.append(
                    MP.run4_install_document(d)) or {"install_timestamp": time.time()},
                expected_run_id="run", expected_cell_id="cell", allowed_action_ids=(71,))
            outcome = service.ingest(update, ingest_at=update["capture_timestamp"] + 0.01,
                                     emit=False)
            service.close()
        finally:
            MI.protocol = legacy
        self.assertEqual(outcome["outcome"], DP.OUTCOME_RESULT_INSTALLED)
        self.assertEqual(outcome["feedback"]["run4_identity"], prepared.identity)
        self.assertEqual(installed[0]["action_id"], "")          # nothing fabricated
        # UE ledger reconciles the ACK to the exact transmitted identity.
        with tempfile.TemporaryDirectory() as tmp:
            ledger = UL.Run4TerminalLedgerV2(output_csv=Path(tmp) / "f.csv",
                                             experiment_id="run", cell_id="cell")
            compat = UL.Run4CompatLedgerV2(ledger)
            capture_id = f"ue288_phase6_test:{p.frame}"
            compat.stage(capture_id, identity=prepared.identity, anchor_profile_id=None)
            compat.register_capture(stream_id="ue288_phase6_test", capture_id=capture_id,
                                    frame_id=p.frame, capture_at=1.0, action_id="71",
                                    service_deadline_at=update["service_deadline_at"],
                                    ack_timeout_at=update["ack_timeout_at"])
            forged = dict(outcome["feedback"], run4_identity=dict(
                prepared.identity, execution_bundle_sha256="cd" * 32))
            with self.assertRaises(Exception):
                ledger.record_message(forged, time.time())
            compat.enqueue(outcome["feedback"], time.time())
            self.assertEqual(compat.receive_once()["status"], "ACK_INSTALLED")
            ledger.close()
        # Evaluator -> R4FB -> controller closes the ticket exactly.
        gt = {"semantic": np.zeros((4, 4), np.uint8),
              "objects": [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}]}
        sent = []
        evaluator = ER.Run4EvaluatorV2(spec=SPEC, send=sent.append, match_distance_m=3.0,
                                       gt_timeout_s=0.1, read_ground_truth=lambda **kw: gt)
        evaluator.evaluate(processed.evaluation)
        feedback, _reason = W.decode_feedback(sent[-1])
        self.assertIsNone(feedback.anchor_action_id)
        ticket = p.engine.controller.current
        self.assertEqual(p.engine.on_feedback(
            feedback, receipt_raw_ns=ticket.action_open_ns + 60 * MS),
            R.FeedbackClass.ACCEPTED)


class TimingAndTicketTest(unittest.TestCase):
    def _decided(self):
        p = Pipeline()
        prepared = p.step()
        processed = processor().process(prepared.wire, edge_timing={})
        feedback, _ = W.quality_feedback(SPEC, {
            "frame_id": processed.envelope.frame_id,
            **{f"seg_{c}_{k}": 10 for c in ("vehicle", "person")
               for k in ("gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")},
            **{f"loc_{c}_{k}": v for c in ("vehicle", "person")
               for k, v in (("eligible_gt", 1), ("tp", 1), ("fn", 0), ("fp", 0))},
            "loc_vehicle_matched_xy_errors_m": [0.1], "loc_person_matched_xy_errors_m": [0.1],
        }, processed.envelope)
        return p, feedback

    def test_exactly_170ms_succeeds_and_one_ns_later_times_out(self) -> None:
        p, feedback = self._decided()
        opened = p.engine.controller.current.action_open_ns
        self.assertEqual(p.engine.on_feedback(feedback, receipt_raw_ns=opened + 170 * MS),
                         R.FeedbackClass.ACCEPTED)
        self.assertIs(p.engine.controller.current.resolution.terminal,
                      contract.RewardTerminal.SUCCESS)
        p2, feedback2 = self._decided()
        opened = p2.engine.controller.current.action_open_ns
        self.assertEqual(p2.engine.on_feedback(feedback2,
                                               receipt_raw_ns=opened + 170 * MS + 1),
                         R.FeedbackClass.LATE_ORPHAN)
        self.assertIs(p2.engine.controller.current.resolution.terminal,
                      contract.RewardTerminal.TIMEOUT)

    def test_duplicate_is_idempotent_and_late_never_attaches(self) -> None:
        p, feedback = self._decided()
        opened = p.engine.controller.current.action_open_ns
        self.assertEqual(p.engine.on_feedback(feedback, receipt_raw_ns=opened + 50 * MS),
                         R.FeedbackClass.ACCEPTED)
        self.assertEqual(p.engine.on_feedback(feedback, receipt_raw_ns=opened + 60 * MS),
                         R.FeedbackClass.DUPLICATE_IGNORED)
        p.step()                       # held tensor (k_min)
        p.step()                       # next decision
        self.assertEqual(p.engine.counters.policy_decisions, 2)
        new_open = p.engine.controller.current.action_open_ns
        self.assertEqual(p.engine.on_feedback(feedback, receipt_raw_ns=new_open + 10 * MS),
                         R.FeedbackClass.DUPLICATE_IGNORED)
        self.assertIsNone(p.engine.controller.current.resolution)   # never attached

    def test_kmin_and_one_reward_request_per_policy_decision(self) -> None:
        p = Pipeline()
        kinds = [p.step().plan for _ in range(7)]
        per_ticket: dict[int, list] = {}
        for plan in kinds:
            per_ticket.setdefault(plan.frame_identity.ticket_seq, []).append(plan)
        closed = sorted(per_ticket)[:-1]          # the final ticket is still open
        self.assertGreaterEqual(len(closed), 2)
        for ticket in closed:
            plans = per_ticket[ticket]
            self.assertGreaterEqual(len(plans), 2)
            self.assertEqual(sum(p_.frame_identity.reward_requested for p_ in plans), 1)
        self.assertEqual(sum(p_.frame_identity.reward_requested
                             for p_ in per_ticket[sorted(per_ticket)[-1]]), 1)
        seqs = [plan.tensor_seq for plan in kinds]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(set(seqs)), len(seqs))


class FallbackAndFaultTest(unittest.TestCase):
    def test_fallback_skips_actor_ticket_and_previous_state_then_retries(self) -> None:
        p = Pipeline()
        first = p.step()                                   # policy decision
        p.step()                                           # held tensor
        opened = p.engine.controller.current.action_open_ns
        p.engine.controller.poll(opened + 200 * MS)        # timeout resolves
        previous = p.engine.controller.previous_for_next_decision()
        calls = p.actor.calls
        refused = p.step(scene=U.SceneDescriptorsV2(None, 0.4, "Invalid", "VALID", 0, 0))
        self.assertIs(refused.plan.kind, E.FrameKind.FALLBACK)
        self.assertEqual(p.actor.calls, calls)
        self.assertEqual((refused.envelope.mode_id, refused.envelope.q_e4,
                          refused.envelope.anchor_action_id), (11, 9800, 71))
        self.assertFalse(refused.envelope.reward_requested)
        self.assertEqual(refused.identity["frame_kind"], "FALLBACK")
        self.assertEqual(p.engine.controller.previous_for_next_decision(), previous)
        self.assertTrue(any("CAMERA_SI" in r for r in refused.plan.fallback_reasons))
        # Fallback still reaches edge verification and the map path.
        processed = processor().process(refused.wire, edge_timing={})
        self.assertIsNone(processed.evaluation)
        MP.validate_run4_map_update(processed.update, contract=CONTRACT)
        retried = p.step()                                 # next frame retries
        self.assertIs(retried.plan.kind, E.FrameKind.POLICY_DECISION)
        self.assertEqual(p.actor.calls, calls + 1)
        self.assertEqual(first.plan.decision_session_uuid, retried.plan.decision_session_uuid)

    def test_hold_is_not_interrupted_by_guard_refusal(self) -> None:
        p = Pipeline()
        decided = p.step()
        held = p.step(fresh=False, scene=U.SceneDescriptorsV2(None, None, "X", "X", 0, 0))
        self.assertIs(held.plan.kind, E.FrameKind.POLICY_HOLD)
        self.assertEqual((held.envelope.mode_id, held.envelope.q_e4),
                         (decided.envelope.mode_id, decided.envelope.q_e4))

    def test_post_assignment_failure_is_excluded_infrastructure_fault(self) -> None:
        p = Pipeline()

        def broken():
            raise OSError("front failed after assignment")
        with self.assertRaises(E.InfrastructureFault):
            p.step(input_fn=broken)
        self.assertIsNotNone(p.engine.faulted)
        self.assertIsNone(p.engine.controller.current.resolution)   # not a timeout
        with self.assertRaises(E.EngineError):
            p.step()

    def test_excluded_evaluator_outcome_rolls_over_to_genesis_session(self) -> None:
        p = Pipeline()
        first = p.step()
        processed = processor().process(first.wire, edge_timing={})
        feedback, _ = W.quality_feedback(SPEC, {"frame_id": processed.envelope.frame_id,
                                                **{f"loc_{c}_eligible_gt": 0
                                                   for c in ("vehicle", "person")}},
                                         processed.envelope)
        self.assertEqual(feedback.kind, "EVALUATOR_FAULT")
        opened = p.engine.controller.current.action_open_ns
        p.engine.on_feedback(feedback, receipt_raw_ns=opened + 50 * MS)
        p.step()                                           # held tensor
        nxt = p.step()
        self.assertIs(nxt.plan.kind, E.FrameKind.POLICY_DECISION)
        self.assertNotEqual(nxt.plan.decision_session_uuid, first.plan.decision_session_uuid)
        self.assertEqual(nxt.plan.frame_identity.decision_seq, 0)
        self.assertEqual(p.engine.counters.session_rollovers, 1)


class ClockAndCoverageTest(unittest.TestCase):
    def test_mixed_clock_subtraction_is_impossible(self) -> None:
        with self.assertRaises(W.WireError):
            W.wall(10) - W.raw(5)
        with self.assertRaises(W.WireError):
            W.raw(10) < W.wall(5)
        self.assertEqual(W.raw(10) - W.raw(4), 6)

    def test_policy_coverage_gate_is_non_vacuous(self) -> None:
        rows = [{"opportunity_index": i, "admitted": False} for i in range(300)]
        self.assertEqual(E.policy_coverage(rows)["verdict"], "INCONCLUSIVE_OR_FAILED")
        rows = [{"opportunity_index": i, "admitted": not (10 <= i < 25)} for i in range(300)]
        self.assertEqual(E.policy_coverage(rows)["verdict"], "INCONCLUSIVE_OR_FAILED")
        rows = [{"opportunity_index": i, "admitted": not (10 <= i < 24)} for i in range(300)]
        self.assertEqual(E.policy_coverage(rows)["verdict"], "PASS")
        self.assertEqual(E.policy_coverage([])["verdict"], "INCONCLUSIVE_OR_FAILED")
        plan = json.loads((Path(__file__).with_name("live_qualification_300_v2.json")).read_text())
        self.assertEqual(plan["gates"]["P0_POLICY_COVERAGE"]["min_admitted_fraction"], 0.95)
        verdict = RUN.evaluate_phase6(ue={"coverage": E.policy_coverage([])}, edge={},
                                      map_identity_rows=[], feedback_packets_on_ue_tunnel=[],
                                      cleanup_ok=True)
        self.assertEqual(verdict["verdict"], "INCONCLUSIVE_OR_FAILED")


class DownlinkEvidenceTest(unittest.TestCase):
    def test_r4fb_three_way_reconciliation_from_pcap(self) -> None:
        import struct

        _p, prepared = ContextAndCorruptionTest()._frame()
        env, _ctx = W.unpack_sfd4(prepared.wire)
        feedback, _ = W.quality_feedback(SPEC, {"frame_id": env.frame_id, **{
            f"loc_{c}_eligible_gt": 0 for c in ("vehicle", "person")}}, env)
        payload = W.encode_feedback(feedback, W.EvaluatorReason.QUALITY_UNDEFINED_NO_ELIGIBLE_GT)
        udp = struct.pack("!HHHH", 40000, 51014, 8 + len(payload), 0) + payload
        ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0,
                         bytes([192, 168, 70, 140]), bytes([10, 0, 0, 2])) + udp
        pcap = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 101)
        pcap += struct.pack("<IIII", 1, 0, len(ip), len(ip)) + ip
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.pcap"
            path.write_bytes(pcap)
            tunnel = RUN.r4fb_digests_in_pcap(path)
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(tunnel, [digest])
        self.assertTrue(RUN._three_way([digest], tunnel, [digest]))
        self.assertFalse(RUN._three_way([digest], [], [digest]))       # host-local
        self.assertFalse(RUN._three_way([], [], []))                   # vacuous


class LegacyAndLifecycleTest(unittest.TestCase):
    LEGACY = (
        "rl_agent/splitfusion_direct_edge_map_v1/protocol.py",
        "rl_agent/splitfusion_direct_edge_map_v1/map_ingest.py",
        "rl_agent/splitfusion_direct_edge_map_v1/spatial_map_direct_server_v1.py",
        "rl_agent/splitfusion_direct_edge_map_v1/adapter_direct_v1.py",
        "rl_agent/splitfusion_direct_edge_map_v1/live_pilot_runtime_direct_v1.py",
        "rl_agent/splitfusion_direct_edge_map_v1/ue_ledger.py",
        "rl_agent/splitfusion_quality_feedback_probe_v1/protocol.py",
        "rl_agent/splitfusion_quality_feedback_probe_v1/adapter_quality_v1.py",
        "rl_agent/splitfusion_quality_feedback_probe_v1/live_cell_child.py",
        "rl_agent/splitfusion_live_dispatch_v1/envelope.py",
        "rl_agent/ue_route_b_split_cell_adapter_v1.py",
    )

    def test_legacy_files_are_byte_identical_to_head(self) -> None:
        for relpath in self.LEGACY:
            head = subprocess.run(["git", "show", f"HEAD:{relpath}"], cwd=ROOT,
                                  capture_output=True, check=True).stdout
            self.assertEqual((ROOT / relpath).read_bytes(), head, relpath)

    def test_proxy_forwards_legacy_messages_unchanged(self) -> None:
        proxy = MP.MapProtocolProxyV2(DP, contract=CONTRACT)
        legacy = DP.build_object_map_update(
            run_id="r", cell_id="c", stream_id="s", frame_id=3, sequence_id=3,
            action_id=50, profile_id="p", decoder_identity="d", capture_timestamp_ns=10**18,
            carla_timestamp=0.0, records=[], service_deadline_at=1.0, ack_timeout_at=2.0,
            edge_timing={}, segmentation={})
        proxy.validate_object_map_update(legacy)
        kwargs = dict(outcome=DP.OUTCOME_MAP_REJECTED, terminal=True, install_timestamp=None,
                      map_ingest_at=1.0, feedback_emit_at=1.0, map_age_at_install_ms=None)
        self.assertEqual(DP.encode(proxy.build_map_feedback(update=legacy, **kwargs)),
                         DP.encode(DP.build_map_feedback(update=legacy, **kwargs)))

    def test_feedback_path_requires_distinct_endpoints_and_upf_route(self) -> None:
        ok = lambda *a, **k: SimpleNamespace(returncode=0, stdout="10.0.0.2 via 192.168.70.134 dev eth0")
        local = lambda *a, **k: SimpleNamespace(returncode=0, stdout="10.0.0.2 via 192.168.70.129 dev eth0")
        self.assertTrue(CH.verify_feedback_path(map_host="192.168.70.129", map_port=39320,
                                                ue_host="10.0.0.2", ue_port=51014, run=ok)["via_upf"])
        with self.assertRaises(CH.ChildError):
            CH.verify_feedback_path(map_host="192.168.70.129", map_port=39320,
                                    ue_host="10.0.0.2", ue_port=51014, run=local)
        with self.assertRaises(CH.ChildError):
            CH.verify_feedback_path(map_host="10.0.0.2", map_port=51014,
                                    ue_host="10.0.0.2", ue_port=51014, run=ok)

    def test_edge_import_closure_has_no_host_only_dependency(self) -> None:
        code = (
            "import sys, importlib.abc\n"
            "class B(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, n, p, t=None):\n"
            "        if n.split('.')[0] in ('yaml','pandas'): raise ImportError(n)\n"
            "sys.meta_path.insert(0, B())\n"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2\n"
            "import rl_agent.splitfusion_quality_feedback_probe_v1.gt_evidence\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                              text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin"})
        self.assertEqual(done.returncode, 0, done.stderr[-800:])

    def test_teardown_is_attempted_for_every_resource_on_child_failure(self) -> None:
        calls = []
        cell = SimpleNamespace(cell_id="a71_fav", action_index=71, action_id=71,
                               profile_id="split_ae32_uint4_q9800", model_family="AE32",
                               network_profile_id="FAVORABLE_STABLE", trace_id="t", seed=1)

        class Lifecycle:
            def start_carla(self, port, log):
                calls.append("start_carla")
                return SimpleNamespace(), 99999999

            def wait_for_rpc(self, port, timeout):
                return "0.10"

            def child_env(self):
                return {"PATH": "/usr/bin"}

            def stop_carla(self, server, pgid, port):
                calls.append("stop_carla")
                return {"shutdown_verified": True}

        supervisor = SimpleNamespace(
            Cell=lambda **kw: SimpleNamespace(**kw),
            cell_to_dict=lambda c: dict(vars(c)),
            import_lifecycle_helper=lambda cfg: Lifecycle(),
            _require_phase15_application_cold=lambda c: calls.append("cold") or {"cold": True},
            _start_live_radio=lambda *a: calls.append("radio_up") or ("ns", "state", {}),
            _stop_live_radio=lambda *a: calls.append("radio_down") or {"ok": True},
            _stop_phase15_application=lambda c: calls.append("app_down") or {"ok": True})

        class Capture:
            def __init__(self, *a, **k): pass
            def start(self): calls.append("capture_up")
            def stop(self): calls.append("capture_down")

        original_child, original_tracer = RUN.CHILD_MODULE, RUN.start_ue_tracer
        RUN.CHILD_MODULE = "json.tool"          # exits non-zero, writes no result
        RUN.start_ue_tracer = lambda *a, **k: calls.append("tracer_up") or []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report = RUN.run_one_cell(
                    base_config={"runtime": {"ue_bind_host": "10.0.0.2",
                                             "ue_control_port": 51014}},
                    registered=cell, output_root=Path(tmp), run_id="t",
                    transmitted_budget=3, safety_timeout_s=1.0, carla_port=1,
                    child_timeout_s=20.0, supervisor=supervisor, capture_class=Capture)
                self.assertTrue((Path(tmp) / "cells" / "a71_fav" / "CELL_RESULT.json").is_file())
        finally:
            RUN.CHILD_MODULE, RUN.start_ue_tracer = original_child, original_tracer
        self.assertEqual(report["status"], "FAILED")
        for step in ("capture_down", "app_down", "stop_carla", "radio_down"):
            self.assertIn(step, calls)
        self.assertLess(calls.index("capture_down"), calls.index("radio_down"))
        self.assertTrue(report["cleanup"]["tracer_stopped"])


if __name__ == "__main__":
    unittest.main()
