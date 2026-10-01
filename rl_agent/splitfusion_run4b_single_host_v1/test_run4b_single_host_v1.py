"""Focused offline tests for the single-host Run-4B seams (CPU, no I/O).

The real Phase-6 hold controller base, decision engine, Run-4 observation
guard, Run-4B feature builder, ACK wire, edge processor seam and gate
evaluation are exercised; only scene/radio values are synthetic fixtures.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    phase6_decision_engine_v2 as E,
    phase6_edge_runtime_v2 as E6,
    reward_hold_controller_v2 as R,
)
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase4_reward_hold_v2 import (
    identity_for,
)
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import (
    CONTRACT,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import state_adapter as SA
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4B

from . import ack_wire_v1 as AW
from . import edge_runtime_v1 as EDGE
from . import postrun_v1 as POST
from . import runner_v1 as RUN
from . import ue_v1 as UE

ROOT = Path(__file__).resolve().parents[2]
CLOCK = "CLOCK_MONOTONIC_RAW"
SESSION = str(uuid.UUID(int=4343))
UE_ID = "oai-ue0-rnti55228"
MS = 1_000_000
LINEAGE = UE.controller_lineage_sha256()


def _inputs(identity, *, commit, mcs=24, backlog=0, camera=110.0, radar=0.4,
            camera_age_ms=30):
    boundary = contract.DecisionBoundaryV1(
        identity=identity, state_commit_timestamp_ns=commit,
        action_open_timestamp_ns=commit + 1, clock_domain=CLOCK)
    camera_obs, radar_obs = LS.scene_observations(
        identity, sample_seq=identity.decision_seq, camera_si=camera, radar_p40=radar,
        source="SYNTHETIC_SCENE_FIXTURE", source_timestamp_ns=commit - camera_age_ms * MS,
        available_timestamp_ns=commit - 20 * MS, clock_domain=CLOCK)
    grant = SA.RawUeUlDciGrantCandidateV1(
        identity=contract.SampleIdentityV1(identity.session_uuid, UE_ID,
                                           10 + identity.decision_seq),
        grant_identity=f"synthetic-{identity.decision_seq}",
        link_direction=contract.LinkDirection.UPLINK, mcs_table=0, mcs_index=mcs,
        harq_round=0, new_data_indicator=identity.decision_seq % 2,
        scheduler_policy_id=contract.UL_MCS_POLICY_ID, source="SYNTHETIC_RADIO_FIXTURE",
        source_timestamp_ns=commit - 40 * MS, available_timestamp_ns=commit - 39 * MS,
        clock_domain=CLOCK)
    tick = SA.RawUeRlcBacklogSampleV1(
        identity=contract.SampleIdentityV1(identity.session_uuid, UE_ID,
                                           1000 + identity.decision_seq),
        backlog_bytes=backlog, link_direction=contract.LinkDirection.UPLINK,
        source="SYNTHETIC_RADIO_FIXTURE", source_timestamp_ns=commit - 3 * MS,
        available_timestamp_ns=commit - 2 * MS, clock_domain=CLOCK)
    return dict(identity=identity, boundary=boundary, camera_si=camera_obs,
                radar_p40=radar_obs,
                prior_ul_mcs=SA.select_prior_new_data_ul_mcs((grant,), boundary),
                pre_action_rlc_backlog=SA.select_pre_action_rlc_backlog(
                    (tick,), boundary, payload_enqueue_timestamp_ns=commit + 5 * MS))


def _controller():
    return UE.OperationalHoldControllerV1(session_uuid=SESSION, ue_id=UE_ID,
                                          controller_lineage_sha256=LINEAGE,
                                          clock_domain=CLOCK)


def _ack_for(frame: R.FrameAssignmentV2, **changes) -> tuple[AW.OperationalAckV1, bytes]:
    ack = AW.OperationalAckV1(
        session_uuid=SESSION, controller_lineage_sha256=LINEAGE,
        decision_seq=frame.decision_seq, ticket_seq=frame.ticket_seq,
        frame_id=frame.frame_id, tensor_seq=frame.tensor_seq,
        capture_timestamp_ns=frame.capture_timestamp_ns, mode_id=frame.mode_id,
        q_e4=frame.q_e4, keep_count=1, execution_bundle_sha256=frame.execution_bundle_sha256,
        anchor_action_id=frame.anchor_action_id, reward_requested=True,
        tail_output_sha256="a" * 64, record_count=0, reassembly_complete_wall_s=1.0,
        compute_start_wall_s=1.0, tail_ready_wall_ns=1, ack_send_wall_ns=2)
    ack = dataclasses.replace(ack, **changes) if changes else ack
    return ack, AW.encode(ack)


def _open(c, *, open_ns=1_000 * MS, mode_id=6, q_e4=7777):
    profile, action = identity_for(mode_id, q_e4)
    c.open_decision(identity=c.next_identity(), action=action,
                    execution_bundle_sha256=profile.execution_bundle_sha256,
                    action_open_ns=open_ns)
    return c.next_frame(frame_id=10 + c._tensor_seq, capture_timestamp_ns=open_ns - 5 * MS,
                        now_ns=open_ns + 1)


class AckWireTest(unittest.TestCase):
    def test_round_trip_and_tamper(self) -> None:
        c = _controller()
        frame = _open(c)
        ack, payload = _ack_for(frame)
        self.assertEqual(AW.decode(payload), ack)
        tampered = payload[:-40] + bytes([payload[-40] ^ 1]) + payload[-39:]
        with self.assertRaises(AW.AckWireError):
            AW.decode(tampered)
        with self.assertRaises(AW.AckWireError):
            dataclasses.replace(ack, reward_requested=False)

    def test_ack_schema_has_no_quality_reward_or_gt_field(self) -> None:
        names = {f.name for f in dataclasses.fields(AW.OperationalAckV1)}
        for name in names:
            self.assertFalse(any(t in name for t in ("q_perc", "quality", "gt_",
                                                     "ground_truth")), name)
        self.assertEqual({n for n in names if "reward" in n}, {"reward_requested"})


class StateTest(unittest.TestCase):
    def test_genesis_state_is_the_registered_20_features(self) -> None:
        identity = contract.DecisionIdentityV1(SESSION, UE_ID, 0)
        guarded, features = UE.build_run4b_live_state(
            **_inputs(identity, commit=1_000 * MS, mcs=24, backlog=4096), previous=None)
        expected = C4B.build_features(
            C4B.ObservationV1(camera_si=110.0, radar_p40=0.4, prior_ul_mcs=24,
                              pre_action_rlc_backlog_bytes=4096),
            C4B.OperationalPriorV1.genesis(), UE.SCALING)
        self.assertEqual(features.as_tuple(), tuple(expected))
        self.assertEqual(len(features.as_tuple()), 20)
        self.assertEqual(guarded.boundary.identity, identity)
        self.assertNotIn("prev_quality_qperc", C4B.FEATURE_ORDER)

    def test_prior_enters_as_operational_outcome(self) -> None:
        identity = contract.DecisionIdentityV1(SESSION, UE_ID, 1)
        prior = UE.LivePriorV1(
            identity=contract.DecisionIdentityV1(SESSION, UE_ID, 0),
            prior=C4B.OperationalPriorV1.from_outcome(
                mode_id=8, q_e4=5000, timely=True, operational_latency_ns=85 * MS),
            available_timestamp_ns=900 * MS, clock_domain=CLOCK)
        _, features = UE.build_run4b_live_state(
            **_inputs(identity, commit=1_000 * MS), previous=prior)
        values = features.as_dict()
        self.assertEqual(values["prev_joint_mode_8_one_hot"], 1.0)
        self.assertAlmostEqual(values["prev_operational_latency_normalized"], 85 / 170)
        self.assertEqual(values["prev_operational_success"], 1.0)

    def test_causal_and_support_refusals(self) -> None:
        identity = contract.DecisionIdentityV1(SESSION, UE_ID, 1)
        prior = UE.LivePriorV1(
            identity=contract.DecisionIdentityV1(SESSION, UE_ID, 0),
            prior=C4B.OperationalPriorV1.from_outcome(
                mode_id=8, q_e4=5000, timely=False, operational_latency_ns=None),
            available_timestamp_ns=1_001 * MS, clock_domain=CLOCK)
        with self.assertRaises(contract.ExternalFallbackRequired):  # not yet available
            UE.build_run4b_live_state(**_inputs(identity, commit=1_000 * MS),
                                      previous=prior)
        with self.assertRaises(UE.Run4BUeError):                     # missing prior
            UE.build_run4b_live_state(**_inputs(identity, commit=1_000 * MS),
                                      previous=None)
        genesis = contract.DecisionIdentityV1(SESSION, UE_ID, 0)
        with self.assertRaises(contract.ExternalFallbackRequired):  # MCS support
            UE.build_run4b_live_state(**_inputs(genesis, commit=1_000 * MS, mcs=5),
                                      previous=None)
        with self.assertRaises(contract.ExternalFallbackRequired):  # stale camera
            UE.build_run4b_live_state(**_inputs(genesis, commit=20_000 * MS,
                                                camera_age_ms=10_000), previous=None)


class ControllerTest(unittest.TestCase):
    def test_timely_ack_success_hold_then_prior(self) -> None:
        c = _controller()
        frame = _open(c)
        ack, payload = _ack_for(frame)
        self.assertFalse(c.can_decide(1_050 * MS))
        self.assertIs(c.on_operational_ack(ack, payload=payload, receipt_ns=1_120 * MS),
                      R.FeedbackClass.ACCEPTED)
        self.assertEqual(c.current.resolution.terminal, "SUCCESS")
        self.assertEqual(c.current.resolution.operational_latency_ns, 120 * MS)
        self.assertFalse(c.can_decide(1_130 * MS), "k_min=2 hold not satisfied")
        held = c.next_frame(frame_id=99, capture_timestamp_ns=1_090 * MS, now_ns=1_135 * MS)
        self.assertFalse(held.reward_requested)
        self.assertEqual((held.mode_id, held.q_e4, held.decision_seq),
                         (frame.mode_id, frame.q_e4, frame.decision_seq))
        self.assertTrue(c.can_decide(1_140 * MS))
        prior = c.previous_for_next_decision()
        self.assertIs(prior.prior.kind, C4B.PriorKind.TIMELY_ACK)
        self.assertEqual(prior.prior.operational_latency_ns, 120 * MS)
        self.assertIs(c.on_operational_ack(ack, payload=payload, receipt_ns=1_141 * MS),
                      R.FeedbackClass.DUPLICATE_IGNORED)

    def test_late_ack_is_timeout_then_late_orphan(self) -> None:
        c = _controller()
        frame = _open(c)
        ack, payload = _ack_for(frame)
        self.assertIs(c.on_operational_ack(ack, payload=payload, receipt_ns=1_171 * MS),
                      R.FeedbackClass.LATE_ORPHAN)
        self.assertEqual(c.current.resolution.terminal, "TIMEOUT")
        self.assertIsNone(c.current.resolution.operational_latency_ns)
        self.assertIs(c.previous_for_next_decision().prior.kind, C4B.PriorKind.FAILURE)
        self.assertEqual(c.ledger.feedback[-1]["observed_latency_ns"], 171 * MS)

    def test_exact_170ms_is_success(self) -> None:
        c = _controller()
        frame = _open(c)
        ack, payload = _ack_for(frame)
        c.on_operational_ack(ack, payload=payload, receipt_ns=1_170 * MS)
        self.assertEqual(c.current.resolution.terminal, "SUCCESS")

    def test_identity_mismatch_and_quality_feedback_fail_closed(self) -> None:
        c = _controller()
        frame = _open(c)
        ack, payload = _ack_for(frame, q_e4=frame.q_e4 + 1)
        with self.assertRaises(R.ConflictingFeedbackError):
            c.on_operational_ack(ack, payload=payload, receipt_ns=1_050 * MS)
        c2 = _controller()
        _open(c2)
        with self.assertRaises(R.ConflictingFeedbackError):
            c2.on_feedback(object(), receipt_ns=1)

    def test_resolution_never_carries_qperc(self) -> None:
        c = _controller()
        frame = _open(c)
        with self.assertRaises(UE.Run4BUeError):
            c._resolve(c.current, kind=contract.RewardEventKind.DELIVERED_SUCCESS,
                       resolution_ns=1_050 * MS, q_perc=0.9, by="X")
        c.poll(1_200 * MS)
        record = c.current.resolution.to_canonical_dict()
        self.assertFalse(record["live_qperc_computed"])
        self.assertNotIn("reward", record)


class EngineTest(unittest.TestCase):
    def test_decision_hold_alternation_with_operational_prior(self) -> None:
        priors = []

        class Actor:
            calls = 0

            def act(self, features):
                Actor.calls += 1
                return SimpleNamespace(mode_id=6, q_e4=7777)

        engine = UE.Run4BDecisionEngineV1(contract_=CONTRACT, actor=Actor(), ue_id=UE_ID,
                                          controller_lineage_sha256=LINEAGE)
        now = [1_000 * MS]

        def build_state(identity, previous):
            priors.append(previous)
            return UE.build_run4b_live_state(
                **_inputs(identity, commit=now[0] - 2), previous=previous)

        kinds = []
        for index in range(8):
            now[0] += 100 * MS
            plan = engine.plan_frame(frame_id=100 + index, capture_wall_ns=now[0],
                                     now_raw_ns=now[0], build_state=build_state)
            kinds.append(plan.kind.value)
            if plan.kind is E.FrameKind.POLICY_DECISION:
                ack, payload = _ack_for(plan.assignment)
                ack = dataclasses.replace(ack, session_uuid=engine.controller.session_uuid)
                payload = AW.encode(ack)
                engine.on_operational_ack(ack, payload=payload,
                                          receipt_raw_ns=now[0] + 60 * MS)
        self.assertEqual(kinds, ["POLICY_DECISION", "POLICY_HOLD"] * 4)
        self.assertIsNone(priors[0])
        self.assertTrue(all(p.prior.kind is C4B.PriorKind.TIMELY_ACK for p in priors[1:]))
        self.assertEqual(Actor.calls, 4)
        self.assertEqual(engine.counters.policy_decisions, 4)
        self.assertEqual(engine.counters.policy_holds, 4)

    def test_refused_state_is_the_registered_fallback(self) -> None:
        class Actor:
            def act(self, features):
                raise AssertionError("actor called after refusal")

        engine = UE.Run4BDecisionEngineV1(contract_=CONTRACT, actor=Actor(), ue_id=UE_ID,
                                          controller_lineage_sha256=LINEAGE)

        def refuse(identity, previous):
            raise contract.ExternalFallbackRequired("RADIO:STALE")

        plan = engine.plan_frame(frame_id=1, capture_wall_ns=1, now_raw_ns=10,
                                 build_state=refuse)
        self.assertIs(plan.kind, E.FrameKind.FALLBACK)
        self.assertEqual((plan.profile.mode_id, plan.profile.q_e4), (11, 9800))
        self.assertFalse(plan.frame_identity.reward_requested)


class EdgeSeamTest(unittest.TestCase):
    def test_ack_is_sent_when_tail_output_is_ready_and_only_for_decisions(self) -> None:
        link = EDGE._AckLink()
        sent = []
        link.send = sent.append
        events = []

        def compute(**_):
            events.append("tail_ready")
            return [{"class_name": "vehicle", "x": 1.0}], np.zeros((4, 4), np.uint8)

        processor = EDGE.OperationalAckEdgeProcessorV1.__new__(
            EDGE.OperationalAckEdgeProcessorV1)
        processor._inner_compute, processor._link = compute, link
        processor._timing = {"reassembly_complete_wall_s": 1.0, "compute_start_wall_s": 1.1}
        envelope = SimpleNamespace(
            session_uuid=SESSION, controller_lineage_sha256=LINEAGE, decision_seq=0,
            ticket_seq=0, frame_id=7, tensor_seq=0, capture_timestamp_ns=5, mode_id=6,
            q_e4=7777, keep_count=3, execution_bundle_sha256="b" * 64,
            anchor_action_id=None, reward_requested=True)
        records, labels = processor._compute_then_ack(
            envelope=envelope, context=None, profile=None, identity=None)
        self.assertEqual(events, ["tail_ready"])
        self.assertEqual(len(sent), 1)
        ack = AW.decode(sent[0])
        self.assertEqual(ack.tail_output_sha256, AW.tail_output_sha256(records, labels))
        self.assertEqual(link.records[0]["ack_sha256"], hashlib.sha256(sent[0]).hexdigest())
        processor._compute_then_ack(envelope=dataclasses.replace(envelope)
                                    if False else SimpleNamespace(
                                        **{**vars(envelope), "reward_requested": False}),
                                    context=None, profile=None, identity=None)
        self.assertEqual(len(sent), 1, "a held/fallback tensor must never be ACKed")

    def test_ack_precedes_map_publication_in_the_unchanged_edge_loop(self) -> None:
        source = Path(E6.__file__).read_text()
        loop = source[source.index("def process_loop"):source.index("ready = Path(")]
        self.assertLess(loop.index("processor.process("), loop.index("publisher.publish("))
        self.assertLess(loop.index("publisher.publish("), loop.index("evaluator.submit("))

    def test_prediction_sink_retains_without_reading_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def no_gt(**_):
                raise AssertionError("the GT-free edge read ground truth")

            sink = EDGE.PredictionSinkV1(
                spec=None, send=lambda payload: None, read_ground_truth=no_gt,
                match_distance_m=3.0, gt_timeout_s=2.0, queue_depth=4,
                evidence_dir=Path(directory), link=EDGE._AckLink())
            sink.start()
            ticket = SimpleNamespace(
                envelope=SimpleNamespace(frame_id=7, tensor_seq=0, decision_seq=0,
                                         session_uuid=SESSION, mode_id=6, q_e4=7777),
                context=SimpleNamespace(stream_id="ue288_x", frame_id=7),
                records=({"class_name": "person"},), predicted_mask=np.ones((2, 3), np.uint8),
                gt_identity={"frame_id": 7, "stream_id": "ue288_x"})
            sink.submit(ticket)
            sink.close()
            files = sorted(p.name for p in (Path(directory) / EDGE.PREDICTION_LEAF).iterdir())
            self.assertEqual(files, ["ue288_x_0000000007.mask.npy",
                                     "ue288_x_0000000007.records.json"])
            self.assertEqual(sink.counters["live_quality_evaluations"], 0)
            self.assertEqual(sink.counters["predictions_written"], 1)


class GateTest(unittest.TestCase):
    def evidence(self):
        identity = {"frame_id": 1}
        frames = [{"ticket_seq": k // 2, "session_uuid": SESSION,
                   "reward_requested": k % 2 == 0} for k in range(300)]
        ue = {"frames": frames,
              "transmitted_identities": [{"run4_identity": identity}] * 300,
              "counters": {"policy_decisions": 150, "actor_calls": 150},
              "resolutions": [{"terminal": "SUCCESS"}] * 150,
              "unresolved_tickets_at_close": 0, "faulted": None,
              "feedback_rows": [{"kind": "OPERATIONAL_ACK", "sha256": "d1",
                                 "class": "ACCEPTED", "q_perc": None}]}
        edge = {"counters": {}, "evaluator": {"live_quality_evaluations": 0},
                "evaluations": [{"ack_sha256": "d1"}]}
        rows = [{"run4_identity_json": json.dumps(identity)}]
        return ue, edge, rows, {"ack": ["d1"], "r4fb": []}

    def test_complete_and_each_failure(self) -> None:
        ue, edge, rows, tunnel = self.evidence()
        ok = RUN.evaluate_run4b(ue=ue, edge=edge, map_identity_rows=rows, tunnel=tunnel,
                                cleanup_ok=True)
        self.assertEqual(ok["verdict"], "INTEGRATION_COMPLETE", ok["gates"])
        cases = {
            "G6_GT_FREE_LIVE_PATH": dict(tunnel={"ack": ["d1"], "r4fb": ["x"]}),
            "G5_ACK_OVER_DOWNLINK": dict(tunnel={"ack": [], "r4fb": []}),
            "G8_RESTORE_COLD": dict(cleanup_ok=False),
        }
        for gate, change in cases.items():
            args = dict(ue=ue, edge=edge, map_identity_rows=rows, tunnel=tunnel,
                        cleanup_ok=True)
            args.update(change)
            result = RUN.evaluate_run4b(**args)
            self.assertFalse(result["gates"][gate], gate)
            self.assertEqual(result["verdict"], "FAILED")
        short = dict(ue, transmitted_identities=ue["transmitted_identities"][:299])
        self.assertFalse(RUN.evaluate_run4b(ue=short, edge=edge, map_identity_rows=rows,
                                            tunnel=tunnel, cleanup_ok=True)
                         ["gates"]["G0_TRANSMITTED_BUDGET"])
        unresolved = dict(ue, unresolved_tickets_at_close=1)
        self.assertFalse(RUN.evaluate_run4b(ue=unresolved, edge=edge,
                                            map_identity_rows=rows, tunnel=tunnel,
                                            cleanup_ok=True)["gates"]["G3_EXACT_TERMINALS"])


class IsolationTest(unittest.TestCase):
    def test_phase6_package_is_byte_identical_to_6199d49(self) -> None:
        diff = subprocess.run(
            ["git", "-C", str(ROOT), "diff", "--quiet", "6199d49", "--",
             "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"], check=False)
        self.assertEqual(diff.returncode, 0)

    def test_seams_install_only_inside_a_child_process(self) -> None:
        code = (
            "from rl_agent.splitfusion_run4b_single_host_v1 import child_v1, ue_v1\n"
            "from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import ("
            "frozen_actor_v2 as FA, live_state_v2 as LS, phase6_decision_engine_v2 as E,"
            " phase6_live_child_v2 as C, phase6_ue_runtime_v2 as U)\n"
            "child_v1.install()\n"
            "assert LS.build_live_state is ue_v1.build_run4b_live_state\n"
            "assert E.Run4DecisionEngineV2 is ue_v1.Run4BDecisionEngineV1\n"
            "assert C.RUN4_EDGE_MODULE == child_v1.EDGE_MODULE\n"
            "assert FA.load_registered_actor.__name__ == '<lambda>'\n"
            "print('SEAMS_OK')\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                capture_output=True, text=True, timeout=300)
        self.assertIn("SEAMS_OK", result.stdout, result.stderr[-2000:])
        self.assertIsNot(LS.build_live_state, UE.build_run4b_live_state,
                         "seams leaked into the test process")

    def test_postrun_module_never_imports_live_runtime(self) -> None:
        source = Path(POST.__file__).read_text()
        for name in ("phase6_ue_runtime_v2", "phase6_edge_runtime_v2", "child_v1",
                     "runner_v1", "reward_hold_controller_v2"):
            self.assertNotIn(name, source)


class ActorTest(unittest.TestCase):
    def test_exact_run4b_actor_loads_and_is_deterministic(self) -> None:
        actor = UE.load_run4b_actor()
        self.assertEqual(actor.boundary_sha256, UE.ACTOR_TREE_SHA256)
        features = UE.Run4BFeatureVectorV1(C4B.build_features(
            C4B.ObservationV1(camera_si=117.0, radar_p40=0.5, prior_ul_mcs=20,
                              pre_action_rlc_backlog_bytes=0),
            C4B.OperationalPriorV1.genesis(), UE.SCALING))
        first, second = actor.act(features), actor.act(features)
        self.assertEqual((first.mode_id, first.q_e4), (second.mode_id, second.q_e4))
        with self.assertRaises(UE.Run4BUeError):
            actor.act(tuple(features.as_tuple()))


if __name__ == "__main__":
    unittest.main()
