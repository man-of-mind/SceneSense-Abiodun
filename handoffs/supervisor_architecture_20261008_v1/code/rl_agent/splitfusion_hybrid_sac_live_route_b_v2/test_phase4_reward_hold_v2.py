"""Phase-4 CPU tests: action hold, 170-ms reward ticket, composed pipeline.

The composed test runs: typed state -> frozen seed-43 actor (CPU) -> dynamic
execution identity -> SFD3 UE/edge (fake codec) -> independent map and
evaluation worker threads -> ACK -> ticket closure.  Every scene/radio value
is a labelled synthetic fixture; no telemetry parsing, CUDA, OAI or network
is involved.
"""

from __future__ import annotations

import dataclasses
import hashlib
import queue
import threading
import unittest
import uuid

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import state_adapter as SA
from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec

from . import continuous_execution_v2 as X
from . import frozen_actor_v2 as FA
from . import live_state_v2 as LS
from . import reward_hold_controller_v2 as R
from .test_phase3_continuous_execution_v2 import CONTRACT, runtimes

CLOCK = "CLOCK_MONOTONIC_RAW"
SESSION = str(uuid.UUID(int=4343))
LINEAGE = hashlib.sha256(b"run4-live-v2-phase4").hexdigest()
UE = "oai-ue0-rnti55228"
MS = 1_000_000


def controller():
    return R.RewardHoldControllerV2(session_uuid=SESSION, ue_id=UE,
                                    controller_lineage_sha256=LINEAGE,
                                    clock_domain=CLOCK)


def identity_for(mode_id=6, q_e4=7777):
    profile = CONTRACT.resolve_q_e4(mode_id, q_e4)
    return profile, X.executed_identity_from_profile(profile, CONTRACT)


def feedback(frame: R.FrameAssignmentV2, *, kind="DELIVERED_SUCCESS",
             q_perc=0.8, **changes) -> R.RewardFeedbackV2:
    fb = R.RewardFeedbackV2(
        session_uuid=SESSION, controller_lineage_sha256=LINEAGE,
        decision_seq=frame.decision_seq, ticket_seq=frame.ticket_seq,
        frame_id=frame.frame_id, tensor_seq=frame.tensor_seq,
        capture_timestamp_ns=frame.capture_timestamp_ns, mode_id=frame.mode_id,
        q_e4=frame.q_e4, execution_bundle_sha256=frame.execution_bundle_sha256,
        anchor_action_id=frame.anchor_action_id, reward_requested=True,
        kind=kind, q_perc=q_perc if kind == "DELIVERED_SUCCESS" else None)
    return dataclasses.replace(fb, **changes) if changes else fb


def opened(c, *, open_ns=1_000 * MS, mode_id=6, q_e4=7777):
    profile, action = identity_for(mode_id, q_e4)
    c.open_decision(identity=c.next_identity(), action=action,
                    execution_bundle_sha256=profile.execution_bundle_sha256,
                    action_open_ns=open_ns)
    first = c.next_frame(frame_id=10, capture_timestamp_ns=open_ns - 5 * MS,
                         now_ns=open_ns + 1)
    return profile, action, first


class DeadlineTest(unittest.TestCase):
    def test_ack_exactly_at_170_ms_is_success(self) -> None:
        c = controller()
        _, _, first = opened(c)
        self.assertEqual(c.on_feedback(feedback(first),
                                       receipt_ns=1_000 * MS + 170 * MS),
                         R.FeedbackClass.ACCEPTED)
        resolution = c.current.resolution
        self.assertIs(resolution.terminal, contract.RewardTerminal.SUCCESS)
        self.assertEqual(resolution.latency_ms, 170.0)
        self.assertAlmostEqual(resolution.reward, 0.8 - 0.25)

    def test_one_tick_late_is_timeout_and_later_ack_is_orphan(self) -> None:
        c = controller()
        _, _, first = opened(c)
        late = 1_000 * MS + 170 * MS + 1
        self.assertEqual(c.on_feedback(feedback(first), receipt_ns=late),
                         R.FeedbackClass.LATE_ORPHAN)
        resolution = c.current.resolution
        self.assertIs(resolution.terminal, contract.RewardTerminal.TIMEOUT)
        self.assertEqual(resolution.reward, -1.0)
        self.assertEqual(resolution.resolution_timestamp_ns,
                         1_000 * MS + R.TIMEOUT_RESOLUTION_ELAPSED_NS)

    def test_lost_decision_tensor_or_lost_ack_times_out(self) -> None:
        c = controller()
        opened(c)
        c.next_frame(frame_id=11, capture_timestamp_ns=1_095 * MS, now_ns=1_100 * MS)
        c.poll(1_171 * MS)
        self.assertIs(c.current.resolution.terminal, contract.RewardTerminal.TIMEOUT)
        self.assertTrue(c.can_decide(1_200 * MS))   # timeout is not termination
        previous = c.previous_for_next_decision()
        self.assertFalse(previous.success)


class HoldTest(unittest.TestCase):
    def test_minimum_two_tensors_and_held_frames(self) -> None:
        c = controller()
        _, action, first = opened(c)
        self.assertTrue(first.reward_requested)
        c.on_feedback(feedback(first), receipt_ns=1_050 * MS)
        self.assertFalse(c.can_decide(1_060 * MS))          # only one tensor sent
        held = c.next_frame(frame_id=11, capture_timestamp_ns=1_095 * MS,
                            now_ns=1_100 * MS)
        self.assertFalse(held.reward_requested)
        self.assertEqual((held.action, held.mode_id, held.q_e4,
                          held.execution_bundle_sha256),
                         (first.action, first.mode_id, first.q_e4,
                          first.execution_bundle_sha256))
        self.assertTrue(c.can_decide(1_101 * MS))

    def test_unresolved_ticket_keeps_holding(self) -> None:
        c = controller()
        _, _, first = opened(c)
        frames = [c.next_frame(frame_id=11 + i, capture_timestamp_ns=(1_095 + 100 * i) * MS,
                               now_ns=(1_000 + 60 * (i + 1)) * MS) for i in range(2)]
        self.assertTrue(all(not f.reward_requested for f in frames))
        self.assertFalse(c.can_decide(1_150 * MS))
        with self.assertRaises(R.ControllerError):
            _, action = identity_for()
            c.open_decision(identity=c.next_identity(), action=action,
                            execution_bundle_sha256="0" * 64, action_open_ns=1_150 * MS)

    def test_prior_outcome_enters_next_state(self) -> None:
        c = controller()
        _, action, first = opened(c)
        c.on_feedback(feedback(first, q_perc=0.6), receipt_ns=1_085 * MS)
        c.next_frame(frame_id=11, capture_timestamp_ns=1_095 * MS, now_ns=1_100 * MS)
        previous = c.previous_for_next_decision()
        self.assertEqual(previous.canonical_sha256(),
                         contract.PreviousOutcomeV1.from_resolution(
                             c.current.resolution).canonical_sha256())
        _, features = _state(c.next_identity(), previous, commit=1_190 * MS)
        named = features.as_dict()
        self.assertEqual(named[f"prev_joint_mode_{action.mode_id}_one_hot"], 1.0)
        self.assertEqual(named["prev_q_normalized"], action.q_e4 / 9800.0)
        self.assertEqual(named["prev_quality_qperc"], 0.6)
        self.assertEqual(named["prev_latency_normalized"], 85.0 / 170.0)
        self.assertEqual((named["prev_present"], named["prev_success"]), (1.0, 1.0))


class FeedbackIdentityTest(unittest.TestCase):
    def test_duplicate_ignored_conflict_fails_closed(self) -> None:
        c = controller()
        _, _, first = opened(c)
        fb = feedback(first)
        self.assertEqual(c.on_feedback(fb, receipt_ns=1_050 * MS), R.FeedbackClass.ACCEPTED)
        self.assertEqual(c.on_feedback(fb, receipt_ns=1_060 * MS),
                         R.FeedbackClass.DUPLICATE_IGNORED)
        with self.assertRaises(R.ConflictingFeedbackError):
            c.on_feedback(feedback(first, q_perc=0.1), receipt_ns=1_061 * MS)
        with self.assertRaises(R.ControllerError):
            c.can_decide(1_200 * MS)

    def test_identity_mismatch_fails_closed(self) -> None:
        for change in ({"q_e4": 7778}, {"mode_id": 5}, {"frame_id": 99},
                       {"tensor_seq": 1}, {"execution_bundle_sha256": "1" * 64},
                       {"anchor_action_id": 3}, {"reward_requested": False}):
            c = controller()
            _, _, first = opened(c)
            with self.assertRaises(R.ConflictingFeedbackError, msg=str(change)):
                c.on_feedback(feedback(first, **change), receipt_ns=1_050 * MS)

    def test_unknown_and_late_orphans_never_close_newer_ticket(self) -> None:
        c = controller()
        _, _, first = opened(c)
        c.next_frame(frame_id=11, capture_timestamp_ns=1_095 * MS, now_ns=1_100 * MS)
        c.poll(1_200 * MS)                              # ticket 0 timed out
        profile, action = identity_for(3, 8500)
        c.open_decision(identity=c.next_identity(), action=action,
                        execution_bundle_sha256=profile.execution_bundle_sha256,
                        action_open_ns=1_200 * MS)
        c.next_frame(frame_id=12, capture_timestamp_ns=1_195 * MS, now_ns=1_201 * MS)
        self.assertEqual(c.on_feedback(feedback(first), receipt_ns=1_210 * MS),
                         R.FeedbackClass.LATE_ORPHAN)
        self.assertIsNone(c.current.resolution)
        other = feedback(first, session_uuid=str(uuid.uuid4()))
        self.assertEqual(c.on_feedback(other, receipt_ns=1_211 * MS),
                         R.FeedbackClass.UNKNOWN_ORPHAN)

    def test_excluded_fault_breaks_session_not_policy(self) -> None:
        c = controller()
        _, _, first = opened(c)
        c.on_feedback(feedback(first, kind="EVALUATOR_FAULT"), receipt_ns=1_050 * MS)
        resolution = c.current.resolution
        self.assertFalse(resolution.learning_included)
        self.assertIsNone(resolution.reward)
        c.next_frame(frame_id=11, capture_timestamp_ns=1_095 * MS, now_ns=1_100 * MS)
        with self.assertRaises(R.SessionBreakRequired):
            c.previous_for_next_decision()

    def test_map_acks_are_independent_of_reward_state(self) -> None:
        c = controller()
        _, _, first = opened(c)
        c.on_map_ack(frame_id=first.frame_id, receipt_ns=1_020 * MS)
        self.assertIsNone(c.current.resolution)
        c.poll(1_171 * MS)
        c.on_map_ack(frame_id=first.frame_id, receipt_ns=1_300 * MS)
        self.assertIs(c.current.resolution.terminal, contract.RewardTerminal.TIMEOUT)
        self.assertEqual(len(c.ledger.map_acks), 2)


# ---------------------------------------------------------------------------
# Composed synthetic pipeline
# ---------------------------------------------------------------------------


def _state(identity, previous, *, commit, mcs=24, backlog=0, camera=110.0, radar=0.4):
    boundary = contract.DecisionBoundaryV1(
        identity=identity, state_commit_timestamp_ns=commit,
        action_open_timestamp_ns=commit + 1, clock_domain=CLOCK)
    camera_obs, radar_obs = LS.scene_observations(
        identity, sample_seq=identity.decision_seq, camera_si=camera, radar_p40=radar,
        source="SYNTHETIC_SCENE_FIXTURE", source_timestamp_ns=commit - 30 * MS,
        available_timestamp_ns=commit - 20 * MS, clock_domain=CLOCK)
    grant = SA.RawUeUlDciGrantCandidateV1(
        identity=contract.SampleIdentityV1(SESSION, UE, 10 + identity.decision_seq),
        grant_identity=f"synthetic-{identity.decision_seq}",
        link_direction=contract.LinkDirection.UPLINK, mcs_table=0, mcs_index=mcs,
        harq_round=0, new_data_indicator=identity.decision_seq % 2,
        scheduler_policy_id=contract.UL_MCS_POLICY_ID, source="SYNTHETIC_RADIO_FIXTURE",
        source_timestamp_ns=commit - 40 * MS, available_timestamp_ns=commit - 39 * MS,
        clock_domain=CLOCK)
    tick = SA.RawUeRlcBacklogSampleV1(
        identity=contract.SampleIdentityV1(SESSION, UE, 1000 + identity.decision_seq),
        backlog_bytes=backlog, link_direction=contract.LinkDirection.UPLINK,
        source="SYNTHETIC_RADIO_FIXTURE", source_timestamp_ns=commit - 3 * MS,
        available_timestamp_ns=commit - 2 * MS, clock_domain=CLOCK)
    return LS.build_live_state(
        identity=identity, boundary=boundary, camera_si=camera_obs, radar_p40=radar_obs,
        prior_ul_mcs=SA.select_prior_new_data_ul_mcs((grant,), boundary),
        pre_action_rlc_backlog=SA.select_pre_action_rlc_backlog(
            (tick,), boundary, payload_enqueue_timestamp_ns=commit + 5 * MS),
        previous=previous)


class LiveStateTest(unittest.TestCase):
    def test_support_refusals_before_actor(self) -> None:
        identity = contract.DecisionIdentityV1(SESSION, UE, 0)
        for kwargs in ({"mcs": 7}, {"backlog": 36_002_537}):
            with self.assertRaises(contract.ExternalFallbackRequired):
                _state(identity, None, commit=1_000 * MS, **kwargs)
        guarded, features = _state(identity, None, commit=1_000 * MS, backlog=40_000)
        self.assertEqual(len(features.as_tuple()), 21)
        self.assertEqual(features.feature_names, contract.POLICY_FEATURE_ORDER)

    def test_training_bindings_exact(self) -> None:
        self.assertEqual(LS.TRAINING_SCALING.canonical_sha256(), LS.TRAINING_SCALING_SHA256)


@unittest.skipUnless((FA.REPOSITORY_ROOT / FA.ACTOR_EXPORT_RELPATH).is_dir(),
                     "actor export not present")
class ComposedPipelineTest(unittest.TestCase):
    # Scripted feedback latency (ms after action open) per decision; None = lost.
    SCRIPT = (60.0, 90.0, 170.0, None, 171.0, 40.0, "LOST_TENSOR", 120.0)

    def test_state_actor_execution_map_eval_ack_closure(self) -> None:
        actor = FA.load_registered_actor()
        ue, edge, _ = runtimes()
        c = controller()
        map_in: "queue.Queue" = queue.Queue()
        eval_in: "queue.Queue" = queue.Queue()
        results: "queue.Queue" = queue.Queue()
        installed: list[int] = []
        evaluated: list[int] = []

        def map_worker():
            while (item := map_in.get()) is not None:
                installed.append(item.envelope.frame_id)
                results.put(("map_ack", item.envelope.frame_id))

        def eval_worker():
            while (item := eval_in.get()) is not None:
                env = item.envelope
                self.assertTrue(env.reward_requested)
                evaluated.append(env.frame_id)
                results.put(("eval", env))

        threads = [threading.Thread(target=map_worker, daemon=True),
                   threading.Thread(target=eval_worker, daemon=True)]
        for t in threads:
            t.start()
        decisions = []
        actor_calls = 0
        slot = 0
        now = 10_000 * MS
        frames_sent = 0
        while len(decisions) < len(self.SCRIPT) or (c.current and (
                c.current.resolution is None or c.current.transmitted < R.K_MIN)):
            now = 10_000 * MS + slot * 100 * MS
            if len(decisions) < len(self.SCRIPT) and c.can_decide(now):
                identity = c.next_identity()
                previous = c.previous_for_next_decision()
                if previous is not None:
                    self.assertEqual(previous.identity.decision_seq,
                                     identity.decision_seq - 1)
                guarded, features = _state(identity, previous, commit=now - 1,
                                           mcs=20 + slot % 8, backlog=slot * 1_000)
                decision = actor.act(features)
                actor_calls += 1
                profile = CONTRACT.resolve_q_e4(decision.mode_id, decision.q_e4)
                action = X.executed_identity_from_profile(profile, CONTRACT)
                c.open_decision(identity=identity, action=action,
                                execution_bundle_sha256=profile.execution_bundle_sha256,
                                action_open_ns=guarded.boundary.action_open_timestamp_ns)
                decisions.append((identity.decision_seq, profile))
            assignment = c.next_frame(frame_id=500 + slot,
                                      capture_timestamp_ns=now - 5 * MS, now_ns=now)
            frames_sent += 1
            script = self.SCRIPT[assignment.decision_seq]
            lost_tensor = assignment.reward_requested and script == "LOST_TENSOR"
            prepared = ue.prepare(
                CONTRACT.resolve_q_e4(assignment.mode_id, assignment.q_e4), object(),
                X.FrameIdentityV2(
                    session_uuid=SESSION, controller_lineage_sha256=LINEAGE,
                    decision_seq=assignment.decision_seq,
                    ticket_seq=assignment.ticket_seq, frame_id=assignment.frame_id,
                    tensor_seq=assignment.tensor_seq,
                    capture_timestamp_ns=assignment.capture_timestamp_ns,
                    reward_requested=assignment.reward_requested))
            self.assertEqual(prepared.action, assignment.action)
            if not lost_tensor:
                result = edge.process(prepared.wire_bytes)
                map_in.put(result)                  # every frame -> map
                if result.envelope.reward_requested:
                    eval_in.put(result)             # only reward frames -> eval
            expected = 0 if lost_tensor else (2 if assignment.reward_requested else 1)
            for _ in range(expected):
                kind, payload = results.get(timeout=10)   # no deadlock
                if kind == "map_ack":
                    c.post("map_ack", payload, now + 20 * MS)
                elif script is not None and script != "LOST_TENSOR":
                    env = payload
                    fb = R.RewardFeedbackV2(
                        session_uuid=env.session_uuid,
                        controller_lineage_sha256=env.controller_lineage_sha256,
                        decision_seq=env.decision_seq, ticket_seq=env.ticket_seq,
                        frame_id=env.frame_id, tensor_seq=env.tensor_seq,
                        capture_timestamp_ns=env.capture_timestamp_ns,
                        mode_id=env.mode_id, q_e4=env.q_e4,
                        execution_bundle_sha256=env.execution_bundle_sha256,
                        anchor_action_id=env.anchor_action_id,
                        reward_requested=env.reward_requested,
                        kind="DELIVERED_SUCCESS", q_perc=0.7)
                    receipt = c.current.action_open_ns + int(script * MS)
                    c.post("feedback", fb, receipt)
                    c.post("feedback", fb, receipt + 1)          # duplicate
            c.run_pending()
            slot += 1
            self.assertLess(slot, 200)
        c.poll(now + 500 * MS)
        for q in (map_in, eval_in):
            q.put(None)
        for t in threads:
            t.join(timeout=5)
            self.assertFalse(t.is_alive())

        terminals = [r.terminal for r in c.resolutions()]
        expected = [contract.RewardTerminal.SUCCESS if isinstance(s, float) and s <= 170.0
                    else contract.RewardTerminal.TIMEOUT for s in self.SCRIPT]
        self.assertEqual(terminals, expected)
        self.assertEqual(actor_calls, len(self.SCRIPT))
        self.assertEqual(len(c.resolutions()), len(self.SCRIPT))
        feedback_classes = [row["class"] for row in c.ledger.feedback]
        self.assertEqual(feedback_classes.count("ACCEPTED"),
                         sum(1 for t in terminals if t is contract.RewardTerminal.SUCCESS))
        self.assertEqual(feedback_classes.count("LATE_ORPHAN"), 2)   # 171 ms + dup
        # No cross-frame attachment: every reward frame is its decision's first tensor.
        reward_frames = [f for f in c.ledger.frames if f.reward_requested]
        self.assertEqual([f.decision_seq for f in reward_frames], list(range(len(self.SCRIPT))))
        held = [f for f in c.ledger.frames if not f.reward_requested]
        self.assertTrue(held)
        for f in c.ledger.frames:
            ticket_first = next(r for r in reward_frames if r.decision_seq == f.decision_seq)
            self.assertEqual((f.mode_id, f.q_e4, f.execution_bundle_sha256),
                             (ticket_first.mode_id, ticket_first.q_e4,
                              ticket_first.execution_bundle_sha256))
        # Every delivered frame (held ones included) reached the map; one lost tensor.
        self.assertEqual(len(installed), frames_sent - 1)
        self.assertEqual(len(c.ledger.map_acks), frames_sent - 1)
        self.assertTrue(set(f.frame_id for f in held) <= set(installed))
        self.assertEqual(len(evaluated), len(self.SCRIPT) - 1)
        per_decision = {}
        for f in c.ledger.frames:
            per_decision[f.decision_seq] = per_decision.get(f.decision_seq, 0) + 1
        self.assertTrue(all(count >= R.K_MIN for count in per_decision.values()), per_decision)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
