"""Adversarial causality/replay checks added by the Phase-4a.2 repair."""

from __future__ import annotations

import dataclasses

from . import reward_ticket_controller as rtc
from . import scene_descriptors as sd
from . import state_reward_transition_contract as src
from . import transaction_identity as ti
from .test_state_reward_transition_contract import (
    BaseContractTest,
    HEX_A,
    MS,
    SESSION,
    T0,
)


class CausalReplayPhase4a2Test(BaseContractTest):
    def _action_failure_transition_inputs(self):
        normalization = self._norm()
        freshness = self._freshness()
        spec = self._reward_spec()
        controller = rtc.RewardTicketController(
            SESSION, controller_lineage_uuid=self.test_lineage_uuid
        )
        action = self._action()
        genesis = controller.authorize_episode_start(
            first_decision_seq=1,
            first_tensor_seq=10,
            first_carla_frame_id=500,
            state_observed_ns=T0,
        )
        # This state and actor trace are constructible before the controller
        # admits or completes the first decision.
        state = self._state(
            episode_start=self._episode_start(
                controller_genesis=genesis
            )
        )
        trace = self._trace(
            action,
            state=state,
            normalization=normalization,
            freshness=freshness,
            decision_seq=1,
        )
        self.assertEqual(controller.completed_count, 0)
        self.assertIs(controller.state, rtc.ControllerState.READY)
        ticket = self._ticket(
            terminal=rtc.TerminalClass.ACTION_PATH_FAILURE,
            controller=controller,
            action=action,
            policy_trace=trace,
        )
        outcome = src.evaluate_completed_decision(ticket, spec)
        return state, normalization, freshness, spec, ticket, outcome, trace

    def test_previous_snapshot_cannot_arrive_from_the_future(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket(terminal=rtc.TerminalClass.ACTION_PATH_FAILURE)
        outcome = src.evaluate_completed_decision(ticket, spec)
        previous = src.PreviousOutcomeV1.from_completed(ticket, outcome, spec)
        self.assertEqual(previous.available_ns, ticket.closed_ns)
        with self.assertRaises(src.CausalStateError) as caught:
            self._state(
                observed_ns=ticket.closed_ns - 1,
                previous=previous,
                episode_start=None,
            )
        self.assertIn("became available", str(caught.exception))

    def test_controller_precommit_rejects_retrospective_state_substitution(
        self,
    ) -> None:
        (
            state,
            normalization,
            freshness,
            spec,
            ticket,
            outcome,
            _trace,
        ) = self._action_failure_transition_inputs()
        altered = dataclasses.replace(
            state,
            scene=self._scene(
                sample=sd.SceneDescriptorSample(
                    camera_si=99.0, radar_p40=0.9
                ),
                carla_frame_id=state.carla_frame_id,
                measured_ns=state.scene.measured_ns,
            ),
        )
        altered_trace = self._trace(
            ticket.action,
            state=altered,
            normalization=normalization,
            freshness=freshness,
            decision_seq=ticket.decision_seq,
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=altered,
                next_state=self._next_state(ticket, outcome, spec),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=altered_trace,
                reward_spec=spec,
                normalization=normalization,
                freshness=freshness,
            )
        self.assertIn("exact trace committed", str(caught.exception))

    def test_timeout_history_is_closure_pending_and_never_rewritten(self) -> None:
        spec = self._reward_spec()
        ticket = self._ticket(terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT)
        pending = src.evaluate_completed_decision(ticket, spec)
        snapshot = src.PreviousOutcomeV1.from_completed(ticket, pending, spec)
        self.assertEqual(
            snapshot.eligibility,
            src.LearningEligibility.CENSORED_PENDING_ADJUDICATION.value,
        )
        self.assertIsNone(snapshot.quality_normalized)

        # An arbitrary later verdict is not a verified reconciliation carrier
        # and cannot turn the censored timeout into training reward.
        arbitrary = self._adjudication(
            ticket, src.Adjudication.AUTHORITATIVE_SERVICE_FAILURE
        )
        with self.assertRaises(src.AdjudicationError):
            src.evaluate_completed_decision(
                ticket, spec, adjudication=arbitrary
            )

    def test_missing_predecessor_requires_episode_start_proof(self) -> None:
        self.assertFalse(hasattr(src.EpisodeStartProofV1, "issue"))
        self.assertFalse(
            hasattr(src.EpisodeStartProofV1, "from_first_completed_ticket")
        )
        ordinary = self._state()
        with self.assertRaises(src.CausalStateError) as caught:
            src.CausalStateV1(
                scene=ordinary.scene,
                radio=ordinary.radio,
                session_uuid=ordinary.session_uuid,
                observed_ns=ordinary.observed_ns,
                tensor_seq=ordinary.tensor_seq,
                carla_frame_id=ordinary.carla_frame_id,
                previous=None,
                episode_start=None,
            )
        self.assertIn("exactly one predecessor proof", str(caught.exception))

        (
            state,
            normalization,
            freshness,
            spec,
            ticket,
            outcome,
            _,
        ) = self._action_failure_transition_inputs()
        wrong_controller = rtc.RewardTicketController(
            SESSION,
            controller_lineage_uuid="7aef680e-9bc0-4a92-b4ad-7e64d1a6112f",
        )
        wrong_genesis = wrong_controller.authorize_episode_start(
            first_decision_seq=ticket.decision_seq,
            first_tensor_seq=state.tensor_seq,
            first_carla_frame_id=state.carla_frame_id,
            state_observed_ns=state.observed_ns,
        )
        wrong_start = src.EpisodeStartProofV1.from_controller_genesis(
            wrong_genesis,
            source_id="wrong_episode_start",
            source_sha256=HEX_A,
        )
        wrong_state = dataclasses.replace(state, episode_start=wrong_start)
        wrong_trace = self._trace(
            ticket.action,
            state=wrong_state,
            normalization=normalization,
            freshness=freshness,
            decision_seq=ticket.decision_seq,
        )
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=wrong_state,
                next_state=self._next_state(ticket, outcome, spec),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=wrong_trace,
                reward_spec=spec,
                normalization=normalization,
                freshness=freshness,
            )
        self.assertIn("different concrete controller episode", str(caught.exception))

    def test_transition_rejects_stale_bootstrap_state(self) -> None:
        (
            state,
            normalization,
            freshness,
            spec,
            ticket,
            outcome,
            trace,
        ) = self._action_failure_transition_inputs()
        observed = ticket.closed_ns + 50 * MS
        stale_next = self._next_state(
            ticket,
            outcome,
            spec,
            observed_ns=observed,
            scene=self._scene(
                carla_frame_id=600,
                measured_ns=observed - freshness.max_scene_age_ns - 1,
            ),
        )
        with self.assertRaises(src.StaleTelemetryError):
            src.build_replay_transition(
                state=state,
                next_state=stale_next,
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=trace,
                reward_spec=spec,
                normalization=normalization,
                freshness=freshness,
            )

    def test_features_and_policy_trace_are_bound_to_decision_sources(self) -> None:
        (
            state,
            normalization,
            freshness,
            _,
            ticket,
            _,
            trace,
        ) = self._action_failure_transition_inputs()
        features = src.build_policy_features(state, normalization, freshness)
        self.assertTrue(features.is_attested)
        self.assertEqual(features.source_state_sha256, state.canonical_sha256())
        self.assertEqual(
            trace.policy_feature_sha256, features.canonical_sha256()
        )
        self.assertEqual(trace.session_uuid, ticket.session_uuid)
        self.assertEqual(trace.decision_seq, ticket.decision_seq)

        other = self._state(observed_ns=T0 - 1)
        with self.assertRaises(src.TransitionIdentityError):
            trace.assert_binds(
                state=other,
                features=src.build_policy_features(
                    other, normalization, freshness
                ),
                decision_seq=ticket.decision_seq,
                executed_action=ticket.action,
            )

    def test_behavior_density_is_not_replay_input_and_q_may_not_clip(self) -> None:
        state = self._state()
        normalization, freshness = self._norm(), self._freshness()
        action = self._action(mode_id=3, q_e4=9800)
        trace = self._trace(
            action,
            state=state,
            normalization=normalization,
            freshness=freshness,
        )
        self.assertNotIn("log_prob_discrete", trace.to_canonical_dict())
        self.assertNotIn("log_prob_continuous", trace.to_canonical_dict())
        for legacy_field in ("log_prob_discrete", "log_prob_continuous"):
            with self.subTest(legacy_field=legacy_field):
                with self.assertRaises(TypeError):
                    self._trace(
                        action,
                        state=state,
                        normalization=normalization,
                        freshness=freshness,
                        **{legacy_field: -0.25},
                    )
        for invalid in (-0.0001, 0.9801, 1.2):
            with self.subTest(sampled_q=invalid):
                with self.assertRaises(src.StateRewardContractError):
                    self._trace(
                        action,
                        state=state,
                        normalization=normalization,
                        freshness=freshness,
                        sampled_q=invalid,
                    )

    def test_gap_tolerant_controller_lineage_proves_exact_predecessor(self) -> None:
        """A legitimate decision gap passes; a merely lower ID cannot."""
        action = self._action(mode_id=3, q_e4=5000)
        controller = rtc.RewardTicketController(
            SESSION, controller_lineage_uuid=self.test_lineage_uuid
        )

        def close(
            decision_seq, tensor_seq, frame_id, opened_ns, policy_trace=None
        ):
            controller.open_decision(
                decision_seq=decision_seq,
                tensor_seq=tensor_seq,
                carla_frame_id=frame_id,
                action=action,
                now_ns=opened_ns,
                policy_decision_trace_sha256=(
                    None
                    if policy_trace is None
                    else policy_trace.canonical_sha256()
                ),
            )
            controller.reuse_held_action(
                tensor_seq=tensor_seq + 1,
                carla_frame_id=frame_id + 1,
                now_ns=opened_ns + 10 * MS,
            )
            completed = controller.submit_feedback(
                rtc.RewardFeedbackMessage(
                    identity=ti.RewardFeedbackIdentity(
                        session_uuid=SESSION,
                        decision_seq=decision_seq,
                        reward_tensor_seq=tensor_seq,
                        carla_frame_id=frame_id,
                        action=action,
                    ),
                    terminal_status=(
                        rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE
                    ),
                ),
                opened_ns + 50 * MS,
            ).completed_ticket
            assert completed is not None
            return completed

        first = close(5, 10, 500, T0)
        spec = self._reward_spec()
        first_outcome = src.evaluate_completed_decision(first, spec)
        previous = src.PreviousOutcomeV1.from_completed(
            first, first_outcome, spec
        )
        state = self._state(
            observed_ns=T0 + 100 * MS,
            tensor_seq=20,
            carla_frame_id=600,
            previous=previous,
            episode_start=None,
        )
        norm, fresh = self._norm(), self._freshness()
        trace = self._trace(
            action,
            state=state,
            normalization=norm,
            freshness=fresh,
            decision_seq=19,
        )
        second = close(19, 20, 600, T0 + 100 * MS, trace)
        second_outcome = src.evaluate_completed_decision(
            second, spec, previous_action=first.action
        )
        transition = src.build_replay_transition(
            state=state,
            next_state=self._next_state(second, second_outcome, spec),
            completed_ticket=second,
            outcome=second_outcome,
            policy_trace=trace,
            reward_spec=spec,
            normalization=norm,
            freshness=fresh,
        )
        self.assertEqual(transition.decision_seq, 19)
        self.assertEqual(previous.decision_seq, 5)
        self.assertEqual(second.lineage_ordinal, 1)
        self.assertEqual(
            second.predecessor_completed_ticket_sha256,
            first.canonical_sha256(),
        )

        # A lower decision ID from another controller is not a predecessor.
        unrelated = self._ticket(
            terminal=rtc.TerminalClass.ACTION_PATH_FAILURE,
            decision_seq=7,
            first_tensor_seq=30,
            first_frame_id=700,
        )
        unrelated_outcome = src.evaluate_completed_decision(unrelated, spec)
        unrelated_previous = src.PreviousOutcomeV1.from_completed(
            unrelated, unrelated_outcome, spec
        )
        bad_state = self._state(
            observed_ns=second.opened_ns,
            tensor_seq=second.reward_tensor_seq,
            carla_frame_id=second.reward_carla_frame_id,
            previous=unrelated_previous,
            episode_start=None,
        )
        with self.assertRaises(src.TransitionIdentityError):
            src.build_replay_transition(
                state=bad_state,
                next_state=self._next_state(second, second_outcome, spec),
                completed_ticket=second,
                outcome=second_outcome,
                policy_trace=self._trace(
                    action,
                    state=bad_state,
                    normalization=norm,
                    freshness=fresh,
                    decision_seq=second.decision_seq,
                ),
                reward_spec=spec,
                normalization=norm,
                freshness=fresh,
            )


if __name__ == "__main__":  # pragma: no cover
    import unittest

    unittest.main()
