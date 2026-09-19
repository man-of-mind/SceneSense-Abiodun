"""Focused tests for the Hybrid-SAC production replay boundary (Phase 4b.2 B).

``SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY``.  CPU-only and deterministic.

Every accepted transition in these tests is a genuine ``ReplayTransitionV1``
assembled through the public contracts -- a real ``RewardTicketController``
episode, a real pre-committed policy trace and
``src.build_replay_transition`` -- never a hand-forged record.  The only
terminal class that is currently learning-eligible is
``ACTION_PATH_FAILURE``; ``REWARD_FINAL_EXACT`` remains fail-closed behind the
unauthenticated quality producer, and these tests do not weaken that.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from typing import Any, Optional

import torch

from . import action_contract as ac
from . import replay_buffer as rbuf
from . import reward_ticket_controller as rtc
from . import state_reward_transition_contract as src
from . import synthetic_contract_environment as sce
from . import transaction_identity as ti
from .test_state_reward_transition_contract import (
    BaseContractTest,
    HEX_A,
    SESSION,
    T0,
)

#: A second session uuid, to prove cross-session (off-policy) acceptance.
OTHER_SESSION = "7c1e9a02-3b44-4f18-9a55-2d0c7e8b1f63"


class ReplayBufferTestBase(BaseContractTest):
    """Builds genuine eligible transitions with controllable identity."""

    def _transition(
        self,
        *,
        mode_id: int = 3,
        q_e4: int = 9800,
        terminal: rtc.TerminalClass = rtc.TerminalClass.ACTION_PATH_FAILURE,
        terminated: bool = False,
        truncated: bool = False,
        with_next: bool = True,
        spec: Optional[src.RewardSpecV1] = None,
        norm: Optional[src.StateNormalizationSpecV1] = None,
        freshness: Optional[src.StateFreshnessPolicyV1] = None,
        extra_reuses: int = 0,
        session_uuid: str = SESSION,
        actor_version: Optional[str] = None,
        tag: str = "",
    ) -> src.ReplayTransitionV1:
        """Assemble one genuine transition entirely through public contracts."""
        spec = spec if spec is not None else self._reward_spec()
        norm = norm if norm is not None else self._norm()
        freshness = freshness if freshness is not None else self._freshness()

        self._transition_index += 1
        lineage = str(
            uuid.uuid5(
                uuid.UUID(self.test_lineage_uuid),
                f"replay-{tag}-{self._transition_index}",
            )
        )
        controller = rtc.RewardTicketController(
            session_uuid, controller_lineage_uuid=lineage
        )
        action = self._action(mode_id=mode_id, q_e4=q_e4)
        offset = self._transition_index * 100
        first_tensor_seq = 10 + offset
        first_frame_id = 500 + offset
        genesis = controller.authorize_episode_start(
            first_decision_seq=1,
            first_tensor_seq=first_tensor_seq,
            first_carla_frame_id=first_frame_id,
            state_observed_ns=T0,
        )
        state = self._state(
            session_uuid=session_uuid,
            tensor_seq=first_tensor_seq,
            carla_frame_id=first_frame_id,
            episode_start=self._episode_start(controller_genesis=genesis),
        )
        trace_overrides: dict = {}
        if actor_version is not None:
            trace_overrides["actor_version_sha256"] = actor_version
        trace = self._trace(
            action,
            state=state,
            normalization=norm,
            freshness=freshness,
            decision_seq=1,
            **trace_overrides,
        )
        ticket = self._ticket(
            terminal=terminal,
            extra_reuses=extra_reuses,
            first_tensor_seq=first_tensor_seq,
            first_frame_id=first_frame_id,
            action=action,
            controller=controller,
            policy_trace=trace,
            session_uuid=session_uuid,
        )
        outcome = src.evaluate_completed_decision(ticket, spec)
        next_state = (
            self._next_state(
                ticket,
                outcome,
                spec,
                carla_frame_id=ticket.reward_carla_frame_id + 1000,
            )
            if with_next
            else None
        )
        return src.build_replay_transition(
            state=state,
            next_state=next_state,
            completed_ticket=ticket,
            outcome=outcome,
            policy_trace=trace,
            reward_spec=spec,
            normalization=norm,
            freshness=freshness,
            terminated=terminated,
            truncated=truncated,
            episode_end_reason=(
                "unit-test-episode-end" if (terminated or truncated) else None
            ),
        )

    def _filled(self, count: int, capacity: Optional[int] = None, **kwargs):
        """A buffer holding ``count`` freshly built eligible transitions."""
        buffer = rbuf.ReplayBufferV1(capacity=capacity or max(count, 1))
        stored = []
        for index in range(count):
            transition = self._transition(tag=f"fill{index}", **kwargs)
            buffer.insert(transition)
            stored.append(transition)
        return buffer, stored


class AcceptanceTest(ReplayBufferTestBase):
    """Genuine eligible transitions are accepted and bound correctly."""

    def test_accepts_genuine_eligible_failure_transitions(self) -> None:
        buffer, stored = self._filled(4)
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.accepted_count, 4)
        self.assertEqual(buffer.seen_digest_count, 4)
        self.assertEqual(buffer.seen_identity_count, 4)
        self.assertEqual(buffer.evicted_count, 0)
        self.assertEqual(buffer.stored_transitions(), tuple(stored))
        for transition in stored:
            self.assertIs(transition.eligibility, src.LearningEligibility.ELIGIBLE)
            self.assertIs(
                transition.terminal_class, rtc.TerminalClass.ACTION_PATH_FAILURE
            )

    def test_eligible_failure_reward_follows_the_registered_formula(self) -> None:
        # The value is r_registered_failure - switch_penalty.total, not a
        # constant.  Vary the spec parameter and the reward must follow.
        for failure_value in (-1.0, -2.5, -0.25):
            spec = self._reward_spec(r_registered_failure=failure_value)
            transition = self._transition(spec=spec, tag=f"rf{failure_value}")
            penalty = transition.outcome.switch_penalty
            self.assertEqual(
                transition.scalar_reward,
                failure_value - penalty.total,
                "reward is not r_registered_failure - switch_penalty.total",
            )
            buffer = rbuf.ReplayBufferV1(capacity=2)
            buffer.insert(transition)
            batch = buffer.sample(1, torch.Generator().manual_seed(0))
            self.assertAlmostEqual(
                float(batch.reward[0]), failure_value - penalty.total, places=6
            )

    def test_binding_is_frozen_from_the_first_record(self) -> None:
        buffer, stored = self._filled(2)
        binding = buffer.binding
        self.assertIsNotNone(binding)
        self.assertEqual(binding.reward_spec_sha256, stored[0].reward_spec_sha256)
        self.assertEqual(
            binding.state_normalization_spec_sha256,
            stored[0].state_normalization_spec_sha256,
        )
        self.assertEqual(
            binding.freshness_policy_sha256, stored[0].freshness_policy_sha256
        )
        self.assertEqual(binding.gamma_per_tensor, stored[0].gamma_per_tensor)
        self.assertEqual(binding.schema_id, src.SCHEMA_ID)
        self.assertEqual(binding.schema_sha256, src.SCHEMA_SHA256)
        self.assertEqual(binding.catalog_sha256, ac.CATALOG_SHA256)
        self.assertEqual(
            binding.policy_feature_order, tuple(src.POLICY_FEATURE_ORDER)
        )
        self.assertEqual(binding.policy_feature_count, 31)
        self.assertIsNone(rbuf.ReplayBufferV1(capacity=1).binding)

    def test_capacity_must_be_a_positive_integer(self) -> None:
        for capacity in (0, -1, True, 2.5, None, "8"):
            with self.assertRaises(rbuf.ReplayBufferError):
                rbuf.ReplayBufferV1(capacity=capacity)
        with self.assertRaises(rbuf.ReplayBufferError):
            rbuf.ReplayBufferV1(capacity=4, float_dtype=torch.int64)


class RejectionTest(ReplayBufferTestBase):
    """Nothing but an exact, revalidating, eligible transition may enter."""

    def test_rejects_measured_aggregates_and_synthetic_fixtures(self) -> None:
        from . import anchor_store as ast

        store = ast.load_anchor_store()
        record = store.by_action_id(0)
        candidates = [
            record,
            record.quality,
            record.outcome("FAVORABLE_STABLE"),
            sce.SyntheticPolicyObservation(values=tuple([0.0] * 31)),
            sce.SyntheticFixtureValueRecord(
                decision_seq=1,
                state_sha256=HEX_A,
                action_sha256=HEX_A,
                mode_id=3,
                q_e4=9800,
                fixture_quality=0.5,
                preferred_mode_id=3,
                preferred_q=0.98,
                terminal_class="ACTION_PATH_FAILURE",
                realized_duration_d=2,
                feedback_latency_ns=None,
                completed_ticket_sha256=HEX_A,
            ),
            {"state": [0.0] * 31},
            ([0.0] * 31,),
            None,
            42,
        ]
        for candidate in candidates:
            buffer = rbuf.ReplayBufferV1(capacity=4)
            with self.assertRaises(
                rbuf.TransitionRejectedError,
                msg=f"{type(candidate).__name__} was not rejected",
            ) as caught:
                buffer.insert(candidate)
            self.assertIn("ReplayTransitionV1", str(caught.exception))
            self.assertEqual(len(buffer), 0)
            self.assertEqual(buffer.seen_digest_count, 0)

    def test_rejects_a_synthetic_run_report(self) -> None:
        report = sce.SyntheticContractEnvironment(
            state_trace=sce.AnalyticStateTrace("replay-reject-trace"),
            outcome_provider=sce.AnalyticOutcomeProvider("replay-reject-outcome"),
            fixture_seed="replay-reject-environment",
        ).run(frame_count=8, policy=sce.CyclingFixturePolicy("replay-reject-policy"))
        self.assertIsInstance(report, sce.SyntheticRunReport)
        self.assertGreater(len(report.decisions), 0, "vacuous rejection loop")
        buffer = rbuf.ReplayBufferV1(capacity=4)
        with self.assertRaises(rbuf.TransitionRejectedError):
            buffer.insert(report)
        for decision in report.decisions:
            with self.assertRaises(rbuf.TransitionRejectedError):
                buffer.insert(decision)
        self.assertEqual(len(buffer), 0)

    def test_rejects_a_subclass_of_the_transition_type(self) -> None:
        # Exact type, not isinstance: a subclass could override revalidate.
        genuine = self._transition(tag="subclass")

        class Sneaky(src.ReplayTransitionV1):
            pass

        self.assertTrue(issubclass(Sneaky, src.ReplayTransitionV1))
        buffer = rbuf.ReplayBufferV1(capacity=2)
        with self.assertRaises(rbuf.TransitionRejectedError):
            buffer.insert(
                Sneaky(
                    state=genuine.state,
                    executed_action=genuine.executed_action,
                    completed_ticket=genuine.completed_ticket,
                    outcome=genuine.outcome,
                    policy_trace=genuine.policy_trace,
                    state_features=genuine.state_features,
                    next_state_features=genuine.next_state_features,
                    reward_spec=genuine.reward_spec,
                    state_normalization_spec_sha256=(
                        genuine.state_normalization_spec_sha256
                    ),
                    freshness_policy_sha256=genuine.freshness_policy_sha256,
                    terminated=genuine.terminated,
                    truncated=genuine.truncated,
                    next_state=genuine.next_state,
                )
            )
        self.assertEqual(len(buffer), 0)

    def test_rejects_censored_and_excluded_transitions(self) -> None:
        for terminal in (
            rtc.TerminalClass.FEEDBACK_TIMEOUT,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        ):
            transition = self._transition(terminal=terminal, tag=terminal.value)
            self.assertFalse(transition.learning_eligible)
            self.assertIsNone(transition.scalar_reward)
            buffer = rbuf.ReplayBufferV1(capacity=2)
            with self.assertRaises(rbuf.IneligibleTransitionError):
                buffer.insert(transition)
            self.assertEqual(len(buffer), 0)

    def test_exact_positive_reward_remains_fail_closed(self) -> None:
        # Not a replay-store behaviour: the contract itself refuses to produce
        # a REWARD_FINAL_EXACT outcome.  Asserted here, at both layers, so a
        # future weakening is noticed by this phase's tests.
        #
        # Without quality components the outcome cannot be scored at all.
        with self.assertRaises(src.QualityContractError):
            self._transition(
                terminal=rtc.TerminalClass.REWARD_FINAL_EXACT, tag="exact"
            )
        # With components supplied, the deeper gate fires: the quality
        # producer is not source-authenticated, so no exact-positive reward
        # exists until the protocol-v2 carrier does.
        with self.assertRaises(src.QualityProducerUnavailableError):
            self._full_transition(
                terminal=rtc.TerminalClass.REWARD_FINAL_EXACT
            )

    def test_rejects_a_mutated_transition(self) -> None:
        genuine = self._transition(tag="mutate")
        buffer = rbuf.ReplayBufferV1(capacity=4)
        buffer.insert(genuine)
        # The contract detects mutation at construction of the replaced copy.
        with self.assertRaises(src.UnattestedRecordError):
            dataclasses.replace(
                genuine, terminated=True, episode_end_reason="forged"
            )
        self.assertEqual(len(buffer), 1)

    def test_rejects_an_unattested_directly_constructed_transition(self) -> None:
        genuine = self._transition(tag="unattested")
        direct = src.ReplayTransitionV1(
            state=genuine.state,
            executed_action=genuine.executed_action,
            completed_ticket=genuine.completed_ticket,
            outcome=genuine.outcome,
            policy_trace=genuine.policy_trace,
            state_features=genuine.state_features,
            next_state_features=genuine.next_state_features,
            reward_spec=genuine.reward_spec,
            state_normalization_spec_sha256=(
                genuine.state_normalization_spec_sha256
            ),
            freshness_policy_sha256=genuine.freshness_policy_sha256,
            terminated=genuine.terminated,
            truncated=genuine.truncated,
            next_state=genuine.next_state,
        )
        self.assertFalse(direct.is_attested)
        buffer = rbuf.ReplayBufferV1(capacity=2)
        with self.assertRaises(src.UnattestedRecordError):
            buffer.insert(direct)
        self.assertEqual(len(buffer), 0)

    def test_rejected_insertion_leaves_the_buffer_bit_identical(self) -> None:
        buffer, stored = self._filled(3, capacity=5)
        before = (
            len(buffer),
            buffer.accepted_count,
            buffer.seen_digest_count,
            buffer.seen_identity_count,
            buffer.evicted_count,
            buffer.binding,
            buffer.stored_transitions(),
        )
        for bad in (
            None,
            {"x": 1},
            self._transition(
                terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT, tag="bad"
            ),
            stored[0],  # duplicate
        ):
            with self.assertRaises(rbuf.TransitionRejectedError):
                buffer.insert(bad)
        after = (
            len(buffer),
            buffer.accepted_count,
            buffer.seen_digest_count,
            buffer.seen_identity_count,
            buffer.evicted_count,
            buffer.binding,
            buffer.stored_transitions(),
        )
        self.assertEqual(before, after)


class BindingHomogeneityTest(ReplayBufferTestBase):
    """One buffer is one learning problem; off-policy variation still allowed."""

    def test_rejects_a_mixed_reward_spec(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4)
        buffer.insert(self._transition(tag="spec-a"))
        other = self._reward_spec(r_registered_failure=-2.0)
        with self.assertRaises(rbuf.BindingMismatchError) as caught:
            buffer.insert(self._transition(spec=other, tag="spec-b"))
        self.assertIn("reward_spec_sha256", str(caught.exception))
        self.assertEqual(len(buffer), 1)

    def test_rejects_a_mixed_gamma(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4)
        buffer.insert(self._transition(tag="gamma-a"))
        other = self._reward_spec(gamma_per_tensor=0.90)
        with self.assertRaises(rbuf.BindingMismatchError):
            buffer.insert(self._transition(spec=other, tag="gamma-b"))
        self.assertEqual(len(buffer), 1)

    def test_rejects_a_mixed_normalization_or_freshness(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4)
        buffer.insert(self._transition(tag="norm-a"))
        with self.assertRaises(rbuf.BindingMismatchError) as caught:
            buffer.insert(
                self._transition(norm=self._norm(spec_version=99), tag="norm-b")
            )
        self.assertIn("state_normalization_spec_sha256", str(caught.exception))

        buffer2 = rbuf.ReplayBufferV1(capacity=4)
        buffer2.insert(self._transition(tag="fresh-a"))
        with self.assertRaises(rbuf.BindingMismatchError) as caught:
            buffer2.insert(
                self._transition(
                    freshness=self._freshness(policy_id="other_policy"),
                    tag="fresh-b",
                )
            )
        self.assertIn("freshness_policy_sha256", str(caught.exception))

    def test_accepts_mixed_sessions_actor_versions_modes_and_durations(
        self,
    ) -> None:
        # SAC is off-policy: none of these may partition a buffer.
        buffer = rbuf.ReplayBufferV1(capacity=16)
        buffer.insert(self._transition(tag="base"))
        buffer.insert(self._transition(session_uuid=OTHER_SESSION, tag="sess"))
        buffer.insert(self._transition(actor_version=HEX_A, tag="actor"))
        buffer.insert(self._transition(mode_id=0, q_e4=0, tag="mode0"))
        buffer.insert(self._transition(mode_id=11, q_e4=5000, tag="mode11"))
        buffer.insert(self._transition(extra_reuses=2, tag="dur"))
        self.assertEqual(len(buffer), 6)

        sessions = {row["session_uuid"] for row in self._audit(buffer)}
        self.assertEqual(sessions, {SESSION, OTHER_SESSION})
        actors = {row["actor_version_sha256"] for row in self._audit(buffer)}
        self.assertGreaterEqual(len(actors), 2)
        batch = buffer.sample(6, torch.Generator().manual_seed(1))
        self.assertEqual(set(batch.mode_id.tolist()), {0, 3, 11})
        self.assertEqual(set(batch.q_e4.tolist()), {0, 5000, 9800})
        self.assertEqual(set(batch.duration.tolist()), {2, 4})

    def _audit(self, buffer):
        return buffer.sample(
            len(buffer), torch.Generator().manual_seed(0)
        ).audit


class DuplicateAndIdentityTest(ReplayBufferTestBase):
    """Digest duplicates and logical-identity conflicts are both refused."""

    def test_duplicate_is_rejected(self) -> None:
        buffer, stored = self._filled(2, capacity=4)
        with self.assertRaises(rbuf.DuplicateTransitionError):
            buffer.insert(stored[0])
        self.assertEqual(len(buffer), 2)

    def test_duplicate_is_still_rejected_after_eviction(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=2)
        first = self._transition(tag="evict-0")
        buffer.insert(first)
        buffer.insert(self._transition(tag="evict-1"))
        buffer.insert(self._transition(tag="evict-2"))
        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.evicted_count, 1)
        self.assertNotIn(first, buffer.stored_transitions())
        # The lifetime index outlives eviction.
        self.assertEqual(buffer.seen_digest_count, 3)
        with self.assertRaises(rbuf.DuplicateTransitionError):
            buffer.insert(first)
        self.assertEqual(len(buffer), 2)

    def test_identity_conflict_is_rejected_and_distinguished(self) -> None:
        # Same (session, lineage, decision_seq) but a different record: two
        # outcomes for one decision.  Built by driving two controllers that
        # share a lineage uuid.
        lineage = str(uuid.uuid5(uuid.UUID(self.test_lineage_uuid), "conflict"))

        def build(resolution_offset_ns: int) -> src.ReplayTransitionV1:
            spec, norm, fresh = self._reward_spec(), self._norm(), self._freshness()
            controller = rtc.RewardTicketController(
                SESSION, controller_lineage_uuid=lineage
            )
            action = self._action()
            genesis = controller.authorize_episode_start(
                first_decision_seq=1,
                first_tensor_seq=10,
                first_carla_frame_id=500,
                state_observed_ns=T0,
            )
            state = self._state(
                tensor_seq=10,
                carla_frame_id=500,
                episode_start=self._episode_start(controller_genesis=genesis),
            )
            trace = self._trace(
                action,
                state=state,
                normalization=norm,
                freshness=fresh,
                decision_seq=1,
            )
            ticket = self._ticket(
                terminal=rtc.TerminalClass.ACTION_PATH_FAILURE,
                action=action,
                controller=controller,
                policy_trace=trace,
                resolution_offset_ns=resolution_offset_ns,
            )
            outcome = src.evaluate_completed_decision(ticket, spec)
            return src.build_replay_transition(
                state=state,
                next_state=self._next_state(
                    ticket, outcome, spec, carla_frame_id=1500
                ),
                completed_ticket=ticket,
                outcome=outcome,
                policy_trace=trace,
                reward_spec=spec,
                normalization=norm,
                freshness=fresh,
            )

        first = build(50 * 1_000_000)
        second = build(60 * 1_000_000)
        self.assertNotEqual(first.canonical_sha256(), second.canonical_sha256())
        buffer = rbuf.ReplayBufferV1(capacity=4)
        buffer.insert(first)
        with self.assertRaises(rbuf.IdentityConflictError) as caught:
            buffer.insert(second)
        self.assertIn("two outcomes", str(caught.exception))
        self.assertEqual(len(buffer), 1)

        # And an identity conflict survives eviction too.
        buffer2 = rbuf.ReplayBufferV1(capacity=1)
        buffer2.insert(first)
        buffer2.insert(self._transition(tag="push"))
        self.assertEqual(buffer2.evicted_count, 1)
        with self.assertRaises(rbuf.IdentityConflictError):
            buffer2.insert(second)


class EvictionAndSamplingTest(ReplayBufferTestBase):
    """Eviction is oldest-first; sampling is uniform, distinct and local."""

    def test_eviction_is_deterministic_oldest_first(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=3)
        stored = [self._transition(tag=f"fifo{i}") for i in range(5)]
        for transition in stored:
            buffer.insert(transition)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.evicted_count, 2)
        self.assertEqual(buffer.stored_transitions(), tuple(stored[2:]))
        self.assertEqual(buffer.accepted_count, 5)

    def test_sampling_requires_an_explicit_generator(self) -> None:
        buffer, _ = self._filled(3)
        for bad in (None, 1234, "seed"):
            with self.assertRaises(rbuf.ReplaySamplingError):
                buffer.sample(2, bad)

    def test_sampling_is_without_replacement_and_bounded(self) -> None:
        buffer, _ = self._filled(4)
        batch = buffer.sample(4, torch.Generator().manual_seed(3))
        digests = [row["transition_sha256"] for row in batch.audit]
        self.assertEqual(len(set(digests)), 4)
        for size in (0, -1, 5, True, 2.0):
            with self.assertRaises(rbuf.ReplaySamplingError):
                buffer.sample(size, torch.Generator().manual_seed(0))
        empty = rbuf.ReplayBufferV1(capacity=2)
        with self.assertRaises(rbuf.ReplaySamplingError):
            empty.sample(1, torch.Generator().manual_seed(0))

    def test_sampling_is_deterministic_for_a_seed_and_varies_across_seeds(
        self,
    ) -> None:
        buffer, _ = self._filled(8)

        def draw(seed: int):
            return [
                row["transition_sha256"]
                for row in buffer.sample(
                    4, torch.Generator().manual_seed(seed)
                ).audit
            ]

        self.assertEqual(draw(11), draw(11))
        self.assertNotEqual(draw(11), draw(12))

    def test_sampling_touches_no_other_rng_stream(self) -> None:
        buffer, _ = self._filled(6)
        torch.manual_seed(777)
        global_before = torch.get_rng_state()
        other = torch.Generator().manual_seed(5)
        other_before = other.get_state()

        replay_generator = torch.Generator().manual_seed(9)
        buffer.sample(3, replay_generator)

        self.assertTrue(torch.equal(torch.get_rng_state(), global_before))
        self.assertTrue(torch.equal(other.get_state(), other_before))
        # The replay stream itself must have advanced.
        self.assertFalse(
            torch.equal(
                replay_generator.get_state(),
                torch.Generator().manual_seed(9).get_state(),
            )
        )


class TensorizationTest(ReplayBufferTestBase):
    """Every learning tensor equals the exact value the contract recorded."""

    def test_dtypes_and_shapes_are_exact(self) -> None:
        buffer, _ = self._filled(5)
        batch = buffer.sample(5, torch.Generator().manual_seed(2))
        self.assertEqual(batch.batch_size, 5)
        self.assertEqual(tuple(batch.state.shape), (5, 31))
        self.assertEqual(tuple(batch.next_state.shape), (5, 31))
        self.assertEqual(batch.state.dtype, torch.float32)
        self.assertEqual(batch.next_state.dtype, torch.float32)
        self.assertEqual(batch.reward.dtype, torch.float32)
        self.assertEqual(batch.q_normalized_executed.dtype, torch.float32)
        self.assertEqual(batch.mode_id.dtype, torch.int64)
        self.assertEqual(batch.q_e4.dtype, torch.int64)
        self.assertEqual(batch.duration.dtype, torch.int64)
        for mask in (
            batch.has_next_state,
            batch.bootstrap,
            batch.terminated,
            batch.truncated,
        ):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(tuple(mask.shape), (5,))

    def test_values_match_the_source_transitions(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=8)
        specs = [(0, 0), (3, 9800), (7, 3000), (11, 7000)]
        for index, (mode_id, q_e4) in enumerate(specs):
            buffer.insert(
                self._transition(
                    mode_id=mode_id, q_e4=q_e4, extra_reuses=index, tag=f"v{index}"
                )
            )
        batch = buffer.sample(4, torch.Generator().manual_seed(4))
        by_digest = {
            transition.canonical_sha256(): transition
            for transition in buffer.stored_transitions()
        }
        for row in range(4):
            source = by_digest[batch.audit[row]["transition_sha256"]]
            self.assertEqual(
                int(batch.mode_id[row]), source.executed_action.mode_id
            )
            self.assertEqual(int(batch.q_e4[row]), source.executed_action.q_e4)
            self.assertAlmostEqual(
                float(batch.q_normalized_executed[row]),
                source.executed_action.q_e4 / 9800.0,
                places=6,
            )
            self.assertAlmostEqual(
                float(batch.reward[row]), source.scalar_reward, places=6
            )
            self.assertEqual(
                int(batch.duration[row]), source.hold_duration_tensors
            )
            self.assertGreaterEqual(int(batch.duration[row]), 2)
            expected_state = torch.tensor(
                source.state_features.values, dtype=torch.float32
            )
            self.assertTrue(torch.equal(batch.state[row], expected_state))
            expected_next = torch.tensor(
                source.next_state_features.values, dtype=torch.float32
            )
            self.assertTrue(torch.equal(batch.next_state[row], expected_next))

    def test_executed_q_is_used_never_the_sampled_request(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=2)
        transition = self._transition(mode_id=5, q_e4=3000, tag="exec-q")
        buffer.insert(transition)
        batch = buffer.sample(1, torch.Generator().manual_seed(0))
        self.assertEqual(int(batch.q_e4[0]), transition.executed_action.q_e4)
        self.assertAlmostEqual(
            float(batch.q_normalized_executed[0]), 3000 / 9800.0, places=6
        )
        # Derived from the retained integer, so it is 0 and 1 at the bounds.
        for q_e4, expected in ((0, 0.0), (9800, 1.0)):
            other = rbuf.ReplayBufferV1(capacity=2)
            other.insert(self._transition(q_e4=q_e4, tag=f"bound{q_e4}"))
            got = other.sample(1, torch.Generator().manual_seed(0))
            self.assertAlmostEqual(
                float(got.q_normalized_executed[0]), expected, places=7
            )

    def test_discount_matches_gamma_to_the_duration(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4)
        for index in range(3):
            buffer.insert(
                self._transition(extra_reuses=index, tag=f"disc{index}")
            )
        batch = buffer.sample(3, torch.Generator().manual_seed(6))
        by_digest = {
            transition.canonical_sha256(): transition
            for transition in buffer.stored_transitions()
        }
        for row in range(3):
            source = by_digest[batch.audit[row]["transition_sha256"]]
            # The emitted discount is the contract's own derived value
            # converted once -- not gamma ** duration recomputed in float32.
            self.assertEqual(
                float(batch.discount()[row]),
                float(
                    torch.tensor(
                        source.discount_multiplier, dtype=batch.float_dtype
                    )
                ),
            )
        for transition in buffer.stored_transitions():
            self.assertEqual(
                transition.discount_multiplier,
                transition.gamma_per_tensor ** transition.hold_duration_tensors,
            )

    def test_batch_accessors_return_isolated_clones(self) -> None:
        buffer, _ = self._filled(3)
        batch = buffer.sample(3, torch.Generator().manual_seed(7))
        for name in (
            "state",
            "next_state",
            "mode_id",
            "q_e4",
            "reward",
            "duration",
            "has_next_state",
            "bootstrap",
            "terminated",
            "truncated",
        ):
            first = getattr(batch, name)
            second = getattr(batch, name)
            self.assertFalse(
                first.data_ptr() == second.data_ptr(),
                f"{name} accessor aliases its own storage",
            )
            if first.dtype == torch.bool:
                first.logical_not_()
            else:
                first.add_(99)
            self.assertTrue(
                torch.equal(getattr(batch, name), second),
                f"mutating the {name} accessor reached the batch storage",
            )
        # The frozen dataclass also refuses attribute rebinding.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            batch.binding = None

    def test_policy_features_carry_no_identifiers(self) -> None:
        src.assert_policy_features_exclude_forbidden_fields()
        buffer, _ = self._filled(2)
        batch = buffer.sample(2, torch.Generator().manual_seed(0))
        self.assertEqual(
            batch.binding.policy_feature_order, tuple(src.POLICY_FEATURE_ORDER)
        )
        for name in src.POLICY_FEATURE_ORDER:
            for forbidden in ("uuid", "seq", "frame_id", "sha256", "ns"):
                self.assertNotIn(forbidden, name)
        # Identity lives only in the audit block, never in the tensors.
        audit_keys = set(batch.audit[0])
        self.assertIn("session_uuid", audit_keys)
        self.assertIn("decision_seq", audit_keys)
        self.assertEqual(batch.state.shape[1], len(src.POLICY_FEATURE_ORDER))

    def test_metadata_is_serializable_and_labelled(self) -> None:
        buffer, _ = self._filled(2)
        batch = buffer.sample(2, torch.Generator().manual_seed(0))
        metadata = batch.to_canonical_metadata()
        self.assertEqual(metadata["schema"], rbuf.REPLAY_BUFFER_SCHEMA_ID)
        self.assertEqual(
            metadata["evidence_class"], "SYNTHETIC_HYBRID_SAC_SMOKE_TEST_ONLY"
        )
        self.assertEqual(metadata["batch_size"], 2)
        json.dumps(metadata, sort_keys=True)


class BootstrapTruthTableTest(ReplayBufferTestBase):
    """bootstrap == has_next_state AND NOT terminated, over every legal row."""

    def test_every_reachable_terminal_combination(self) -> None:
        cases = [
            # (terminated, truncated, with_next, has_next, bootstrap)
            (False, False, True, True, True),
            (True, False, False, False, False),
            (True, False, True, True, False),   # terminal with a real successor
            (False, True, False, False, False),
            (False, True, True, True, True),    # truncation may bootstrap
        ]
        buffer = rbuf.ReplayBufferV1(capacity=16)
        expected = {}
        for index, (term, trunc, with_next, _, _) in enumerate(cases):
            transition = self._transition(
                terminated=term,
                truncated=trunc,
                with_next=with_next,
                tag=f"bt{index}",
            )
            buffer.insert(transition)
            expected[transition.canonical_sha256()] = cases[index]
        batch = buffer.sample(len(cases), torch.Generator().manual_seed(8))
        seen = 0
        for row in range(batch.batch_size):
            term, trunc, _, has_next, boot = expected[
                batch.audit[row]["transition_sha256"]
            ]
            self.assertEqual(bool(batch.terminated[row]), term)
            self.assertEqual(bool(batch.truncated[row]), trunc)
            self.assertEqual(bool(batch.has_next_state[row]), has_next)
            self.assertEqual(
                bool(batch.bootstrap[row]),
                boot,
                f"row {row}: terminated={term} truncated={trunc} "
                f"has_next={has_next}",
            )
            self.assertEqual(
                bool(batch.bootstrap[row]),
                bool(batch.has_next_state[row]) and not bool(batch.terminated[row]),
            )
            seen += 1
        self.assertEqual(seen, len(cases))

    def test_non_terminal_without_a_next_state_is_unreachable(self) -> None:
        # The contract itself refuses it, so the buffer never has to.
        with self.assertRaises(src.TransitionIdentityError):
            self._transition(
                terminated=False, truncated=False, with_next=False, tag="nt"
            )
        with self.assertRaises(src.TransitionIdentityError):
            self._transition(terminated=True, truncated=True, tag="both")

    def test_terminated_row_preserves_its_real_successor_vector(self) -> None:
        # Correction 2: a real next state is kept even when it must not be
        # bootstrapped.  Only a genuinely absent successor is zero-filled.
        buffer = rbuf.ReplayBufferV1(capacity=4)
        with_successor = self._transition(
            terminated=True, with_next=True, tag="term-next"
        )
        without = self._transition(
            terminated=True, with_next=False, tag="term-none"
        )
        buffer.insert(with_successor)
        buffer.insert(without)
        batch = buffer.sample(2, torch.Generator().manual_seed(0))
        by_digest = {
            row["transition_sha256"]: index
            for index, row in enumerate(batch.audit)
        }
        kept = by_digest[with_successor.canonical_sha256()]
        zeroed = by_digest[without.canonical_sha256()]

        expected = torch.tensor(
            with_successor.next_state_features.values, dtype=torch.float32
        )
        self.assertTrue(torch.equal(batch.next_state[kept], expected))
        self.assertFalse(torch.all(batch.next_state[kept] == 0.0))
        self.assertTrue(bool(batch.has_next_state[kept]))
        self.assertFalse(bool(batch.bootstrap[kept]))

        self.assertTrue(torch.all(batch.next_state[zeroed] == 0.0))
        self.assertFalse(bool(batch.has_next_state[zeroed]))
        self.assertFalse(bool(batch.bootstrap[zeroed]))


class ReplayDtypeOverflowTest(ReplayBufferTestBase):
    """A contract-finite value must stay finite in the configured dtype."""

    def test_float32_reward_overflow_is_rejected_before_any_mutation(self) -> None:
        # -1e300 is a perfectly legal finite negative service reward, and is
        # fully reachable through the reward spec.  In float32 it is -inf.
        spec = self._reward_spec(r_registered_failure=-1e300)
        transition = self._transition(spec=spec, tag="overflow")
        self.assertEqual(transition.scalar_reward, -1e300)
        self.assertTrue(math.isfinite(transition.scalar_reward))
        self.assertEqual(
            float(torch.tensor(transition.scalar_reward, dtype=torch.float32)),
            float("-inf"),
        )

        buffer = rbuf.ReplayBufferV1(capacity=4, float_dtype=torch.float32)
        with self.assertRaises(rbuf.NonFiniteInReplayDtypeError) as caught:
            buffer.insert(transition)
        self.assertIn("scalar_reward", str(caught.exception))
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.seen_digest_count, 0)
        self.assertEqual(buffer.seen_identity_count, 0)
        self.assertEqual(buffer.accepted_count, 0)
        self.assertIsNone(buffer.binding)

        # The same record is representable in float64 and is accepted there,
        # which proves the rejection is about the dtype and not the contract.
        wide = rbuf.ReplayBufferV1(capacity=4, float_dtype=torch.float64)
        wide.insert(transition)
        self.assertEqual(len(wide), 1)
        batch = wide.sample(1, torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(batch.reward).all())
        self.assertEqual(float(batch.reward[0]), -1e300)

    def test_no_stored_batch_ever_contains_a_non_finite_value(self) -> None:
        buffer, _ = self._filled(4)
        batch = buffer.sample(4, torch.Generator().manual_seed(1))
        for name in (
            "state",
            "next_state",
            "reward",
            "q_normalized_executed",
        ):
            self.assertTrue(
                torch.isfinite(getattr(batch, name)).all(), f"{name} not finite"
            )
        self.assertTrue(torch.isfinite(batch.discount()).all())

    def test_finiteness_helper_rejects_overflowing_feature_vectors(self) -> None:
        # A state-feature overflow is not reachable through the normalization
        # contract today: every feature is clipped into a bounded range.  The
        # guard is still exercised directly so it cannot silently rot.
        overflowing = [0.0] * 30 + [1e300]
        with self.assertRaises(rbuf.NonFiniteInReplayDtypeError) as caught:
            rbuf._assert_finite_in_dtype(
                overflowing, "state_features", torch.float32
            )
        self.assertIn("state_features[30]", str(caught.exception))
        # float64 keeps it finite, so the same values pass there.
        rbuf._assert_finite_in_dtype(overflowing, "state_features", torch.float64)
        rbuf._assert_finite_in_dtype(
            [0.0] * 31, "next_state_features", torch.float32
        )

    def test_unsupported_dtypes_are_rejected(self) -> None:
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.int64,
            torch.bool,
            torch.complex64,
        ):
            with self.assertRaises(rbuf.ReplayBufferError) as caught:
                rbuf.ReplayBufferV1(capacity=4, float_dtype=dtype)
            self.assertIn("float_dtype", str(caught.exception))
        self.assertEqual(
            rbuf.SUPPORTED_FLOAT_DTYPES, (torch.float32, torch.float64)
        )
        for dtype in rbuf.SUPPORTED_FLOAT_DTYPES:
            self.assertEqual(
                rbuf.ReplayBufferV1(capacity=1, float_dtype=dtype).float_dtype,
                dtype,
            )


class DiscountFidelityTest(ReplayBufferTestBase):
    """The emitted discount is the contract's value, never a recomputation.

    Rounding ``gamma`` into the replay dtype and *then* exponentiating is a
    different number from rounding the contract's own float64 power.  These
    cases pin the difference so the recomputation cannot come back.
    """

    def _single(self, gamma: float, extra_reuses: int, **kwargs):
        spec = self._reward_spec(gamma_per_tensor=gamma)
        transition = self._transition(
            spec=spec, extra_reuses=extra_reuses, tag=f"disc{gamma}", **kwargs
        )
        buffer = rbuf.ReplayBufferV1(capacity=2, float_dtype=torch.float32)
        buffer.insert(transition)
        batch = buffer.sample(1, torch.Generator().manual_seed(0))
        return transition, batch

    def test_case_a_subnormal_discount_is_not_flushed_to_zero(self) -> None:
        gamma = 0.500000021051642
        transition, batch = self._single(gamma, extra_reuses=148)
        self.assertEqual(transition.hold_duration_tensors, 150)
        self.assertTrue(bool(batch.bootstrap[0]))

        direct = torch.tensor(
            transition.discount_multiplier, dtype=torch.float32
        )
        self.assertNotEqual(float(direct), 0.0)
        self.assertEqual(float(batch.discount()[0]), float(direct))
        self.assertNotEqual(
            float(batch.discount()[0]),
            0.0,
            "a nonzero direct conversion was flushed to zero",
        )
        # What the removed recomputation would have produced.
        recomputed = float(
            torch.pow(
                torch.tensor(gamma, dtype=torch.float32),
                torch.tensor(150.0, dtype=torch.float32),
            )
        )
        self.assertEqual(recomputed, 0.0)
        self.assertNotEqual(float(batch.discount()[0]), recomputed)

    def test_case_b_near_one_gamma_does_not_collapse_to_one(self) -> None:
        gamma = 0.99999999
        transition, batch = self._single(gamma, extra_reuses=148)
        self.assertEqual(transition.hold_duration_tensors, 150)

        emitted = float(batch.discount()[0])
        direct = float(
            torch.tensor(transition.discount_multiplier, dtype=torch.float32)
        )
        self.assertEqual(emitted, direct)
        self.assertAlmostEqual(emitted, 0.9999985098838806, places=12)
        self.assertNotEqual(emitted, 1.0)
        # float32 rounds this gamma to exactly 1.0, so the old recomputation
        # produced 1.0 and erased the discount entirely.
        self.assertEqual(float(torch.tensor(gamma, dtype=torch.float32)), 1.0)
        self.assertEqual(
            float(
                torch.pow(
                    torch.tensor(gamma, dtype=torch.float32),
                    torch.tensor(150.0, dtype=torch.float32),
                )
            ),
            1.0,
        )

    def test_ordinary_discount_also_comes_from_the_contract(self) -> None:
        # Even the default gamma drifts: 0.9800999760627747 (correct) versus
        # 0.9801000356674194 (recomputed).
        transition, batch = self._single(0.99, extra_reuses=0)
        self.assertEqual(
            float(batch.discount()[0]),
            float(
                torch.tensor(transition.discount_multiplier, dtype=torch.float32)
            ),
        )
        self.assertNotEqual(
            float(batch.discount()[0]),
            float(
                torch.pow(
                    torch.tensor(0.99, dtype=torch.float32),
                    torch.tensor(2.0, dtype=torch.float32),
                )
            ),
        )

    def test_discount_accessor_returns_an_isolated_clone(self) -> None:
        buffer, _ = self._filled(3)
        batch = buffer.sample(3, torch.Generator().manual_seed(2))
        first = batch.discount()
        second = batch.discount()
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        first.add_(5.0)
        self.assertTrue(torch.equal(batch.discount(), second))
        self.assertEqual(first.dtype, batch.float_dtype)
        self.assertEqual(tuple(first.shape), (3,))

    def test_bootstrap_eligible_underflow_is_rejected_before_mutation(
        self,
    ) -> None:
        # gamma ** d that underflows to exactly zero in float32 on a row that
        # will actually bootstrap.
        spec = self._reward_spec(gamma_per_tensor=0.5)
        transition = self._transition(
            spec=spec, extra_reuses=148, tag="under-boot"
        )
        self.assertFalse(transition.terminated)
        self.assertIsNotNone(transition.next_state)
        self.assertGreater(transition.discount_multiplier, 0.0)
        self.assertEqual(
            float(
                torch.tensor(transition.discount_multiplier, dtype=torch.float32)
            ),
            0.0,
        )
        buffer = rbuf.ReplayBufferV1(capacity=4, float_dtype=torch.float32)
        with self.assertRaises(rbuf.NonFiniteInReplayDtypeError) as caught:
            buffer.insert(transition)
        self.assertIn("bootstrap-eligible", str(caught.exception))
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.seen_digest_count, 0)
        self.assertEqual(buffer.seen_identity_count, 0)
        self.assertIsNone(buffer.binding)

    def test_terminal_and_no_next_state_underflow_are_accepted(self) -> None:
        # The same underflowing discount is harmless when the row never
        # bootstraps, so it must not be rejected.
        spec = self._reward_spec(gamma_per_tensor=0.5)
        buffer = rbuf.ReplayBufferV1(capacity=8, float_dtype=torch.float32)
        cases = (
            {"terminated": True, "with_next": True},
            {"terminated": True, "with_next": False},
            {"truncated": True, "with_next": False},
        )
        for index, case in enumerate(cases):
            buffer.insert(
                self._transition(
                    spec=spec,
                    extra_reuses=148,
                    tag=f"under-term{index}",
                    **case,
                )
            )
        self.assertEqual(len(buffer), 3)
        batch = buffer.sample(3, torch.Generator().manual_seed(0))
        self.assertFalse(bool(batch.bootstrap.any()))
        # Their discounts are emitted as the contract's value converted once,
        # and stay finite even where that value is zero.
        self.assertTrue(torch.isfinite(batch.discount()).all())
        self.assertTrue(torch.all(batch.discount() == 0.0))

    def test_emitted_discounts_are_always_finite(self) -> None:
        for gamma in (0.5, 0.99, 0.99999999, 1.0, 0.500000021051642):
            spec = self._reward_spec(gamma_per_tensor=gamma)
            buffer = rbuf.ReplayBufferV1(capacity=4, float_dtype=torch.float32)
            buffer.insert(
                self._transition(
                    spec=spec,
                    terminated=True,
                    with_next=False,
                    extra_reuses=20,
                    tag=f"fin{gamma}",
                )
            )
            batch = buffer.sample(1, torch.Generator().manual_seed(0))
            self.assertTrue(
                torch.isfinite(batch.discount()).all(), f"gamma={gamma}"
            )


class GeneratorRestrictionTest(ReplayBufferTestBase):
    """Replay sampling owns a private CPU stream and nothing else."""

    def test_default_generator_is_rejected_without_disturbing_global_rng(
        self,
    ) -> None:
        buffer, _ = self._filled(5)
        torch.manual_seed(31337)
        before_state = torch.get_rng_state()
        before_default = torch.default_generator.get_state()

        with self.assertRaises(rbuf.ReplaySamplingError) as caught:
            buffer.sample(2, torch.default_generator)
        self.assertIn("default_generator", str(caught.exception))

        # Rejected before sampling, so the global stream is untouched.
        self.assertTrue(torch.equal(torch.get_rng_state(), before_state))
        self.assertTrue(
            torch.equal(torch.default_generator.get_state(), before_default)
        )
        # And the buffer still samples normally from a private stream.
        batch = buffer.sample(2, torch.Generator().manual_seed(4))
        self.assertEqual(batch.batch_size, 2)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_state))

    def test_torch_random_default_generator_is_the_same_rejected_object(
        self,
    ) -> None:
        self.assertIs(torch.default_generator, torch.random.default_generator)
        buffer, _ = self._filled(2)
        with self.assertRaises(rbuf.ReplaySamplingError):
            buffer.sample(1, torch.random.default_generator)

    def test_a_fresh_cpu_generator_is_accepted(self) -> None:
        buffer, _ = self._filled(3)
        explicit = torch.Generator(device="cpu").manual_seed(8)
        self.assertEqual(explicit.device.type, "cpu")
        self.assertIsNot(explicit, torch.default_generator)
        self.assertEqual(buffer.sample(2, explicit).batch_size, 2)

    @unittest.skipUnless(
        os.environ.get("SPLITFUSION_ALLOW_CUDA_TEST") == "1"
        and torch.cuda.is_available(),
        "CUDA generators are not constructed in this phase: building one "
        "initializes a CUDA context, which this phase forbids.  Set "
        "SPLITFUSION_ALLOW_CUDA_TEST=1 to run the real-device check.",
    )
    def test_non_cpu_generator_is_rejected(self) -> None:  # pragma: no cover
        buffer, _ = self._filled(2)
        cuda_generator = torch.Generator(device="cuda")
        self.assertNotEqual(cuda_generator.device.type, "cpu")
        with self.assertRaises(rbuf.ReplaySamplingError) as caught:
            buffer.sample(1, cuda_generator)
        self.assertIn("CPU generator", str(caught.exception))

    def test_device_guard_is_present_in_the_sampling_path(self) -> None:
        # The CUDA path above cannot run here, so at minimum prove the guard
        # exists and that the accepted generator satisfies it.
        source = Path(rbuf.__file__).read_text(encoding="utf-8")
        self.assertIn('generator.device.type != "cpu"', source)
        self.assertIn("generator is torch.default_generator", source)


class PostAttestationTamperTest(ReplayBufferTestBase):
    """A record mutated after attestation is refused and poisons nothing."""

    def test_object_setattr_tamper_is_rejected_and_leaves_no_trace(self) -> None:
        buffer, stored = self._filled(2, capacity=8)
        victim = self._transition(tag="tamper")
        self.assertTrue(victim.is_attested)
        self.assertFalse(victim.terminated)
        original_digest = victim.canonical_sha256()

        before = (
            len(buffer),
            buffer.accepted_count,
            buffer.seen_digest_count,
            buffer.seen_identity_count,
            buffer.evicted_count,
            buffer.stored_transitions(),
        )

        # Tamper in place, bypassing the frozen dataclass entirely.  This is
        # the same object; no copy was made, so the attestation travels with
        # it and the only defence is recomputing the binding.
        object.__setattr__(victim, "terminated", True)
        object.__setattr__(victim, "episode_end_reason", "forged-terminal")
        self.assertFalse(victim.is_attested)

        with self.assertRaises(src.UnattestedRecordError):
            buffer.insert(victim)

        after = (
            len(buffer),
            buffer.accepted_count,
            buffer.seen_digest_count,
            buffer.seen_identity_count,
            buffer.evicted_count,
            buffer.stored_transitions(),
        )
        self.assertEqual(before, after, "a rejected tamper changed the buffer")
        self.assertNotIn(victim, buffer.stored_transitions())

        # Restore the object to exactly what was attested.  If the rejected
        # attempt had poisoned either lifetime index, this insert would now
        # fail as a duplicate or an identity conflict.
        object.__setattr__(victim, "terminated", False)
        object.__setattr__(victim, "episode_end_reason", None)
        self.assertTrue(victim.is_attested)
        self.assertEqual(victim.canonical_sha256(), original_digest)

        buffer.insert(victim)
        self.assertEqual(len(buffer), before[0] + 1)
        self.assertEqual(buffer.seen_digest_count, before[2] + 1)
        self.assertEqual(buffer.seen_identity_count, before[3] + 1)
        self.assertIn(victim, buffer.stored_transitions())
        self.assertEqual(buffer.stored_transitions()[:2], tuple(stored))

    def test_tampering_a_reward_bearing_field_is_rejected(self) -> None:
        buffer = rbuf.ReplayBufferV1(capacity=4)
        victim = self._transition(tag="tamper-reward")
        object.__setattr__(victim, "state_normalization_spec_sha256", HEX_A)
        with self.assertRaises(src.UnattestedRecordError):
            buffer.insert(victim)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.seen_digest_count, 0)


class ConstructorAliasingTest(ReplayBufferTestBase):
    """The batch never shares storage with its inputs or with the buffer."""

    def test_batch_clones_its_constructor_inputs(self) -> None:
        state = torch.zeros(2, 31, dtype=torch.float32)
        next_state = torch.zeros(2, 31, dtype=torch.float32)
        mode_id = torch.zeros(2, dtype=torch.int64)
        q_e4 = torch.zeros(2, dtype=torch.int64)
        reward = torch.zeros(2, dtype=torch.float32)
        duration = torch.full((2,), 2, dtype=torch.int64)
        discount = torch.zeros(2, dtype=torch.float32)
        mask = torch.zeros(2, dtype=torch.bool)
        binding = rbuf.ReplayBindingV1(
            reward_spec_sha256=HEX_A,
            state_normalization_spec_sha256=HEX_A,
            freshness_policy_sha256=HEX_A,
            gamma_per_tensor=0.99,
            schema_id=src.SCHEMA_ID,
            schema_version=src.SCHEMA_VERSION,
            schema_sha256=src.SCHEMA_SHA256,
            catalog_sha256=ac.CATALOG_SHA256,
            policy_feature_order=tuple(src.POLICY_FEATURE_ORDER),
            policy_feature_count=31,
        )
        batch = rbuf.ReplayTensorBatchV1(
            _state=state,
            _next_state=next_state,
            _mode_id=mode_id,
            _q_e4=q_e4,
            _reward=reward,
            _duration=duration,
            _discount=discount,
            _has_next_state=mask,
            _bootstrap=mask,
            _terminated=mask,
            _truncated=mask,
            binding=binding,
            audit=(),
            float_dtype=torch.float32,
        )
        # Mutating the originals must not reach the batch.
        state.fill_(7.0)
        next_state.fill_(9.0)
        mode_id.fill_(5)
        q_e4.fill_(9800)
        reward.fill_(3.0)
        duration.fill_(11)
        discount.fill_(0.5)
        mask.fill_(True)

        self.assertTrue(torch.all(batch.state == 0.0))
        self.assertTrue(torch.all(batch.next_state == 0.0))
        self.assertTrue(torch.all(batch.mode_id == 0))
        self.assertTrue(torch.all(batch.q_e4 == 0))
        self.assertTrue(torch.all(batch.reward == 0.0))
        self.assertTrue(torch.all(batch.duration == 2))
        self.assertTrue(torch.all(batch.discount() == 0.0))
        for name in ("has_next_state", "bootstrap", "terminated", "truncated"):
            self.assertFalse(bool(getattr(batch, name).any()), name)
        # The four masks were one shared tensor; the batch must hold four
        # independent copies rather than four references to it.
        pointers = {
            getattr(batch, f"_{name}").data_ptr()
            for name in (
                "has_next_state",
                "bootstrap",
                "terminated",
                "truncated",
            )
        }
        self.assertEqual(len(pointers), 4, "batch aliased one shared mask")

    def test_batch_rejects_non_tensor_constructor_inputs(self) -> None:
        buffer, _ = self._filled(1)
        batch = buffer.sample(1, torch.Generator().manual_seed(0))
        with self.assertRaises(rbuf.ReplayBufferError):
            dataclasses.replace(batch, _state=[[0.0] * 31])

    def test_batch_does_not_alias_buffer_row_storage(self) -> None:
        buffer, _ = self._filled(2)
        first = buffer.sample(2, torch.Generator().manual_seed(0))
        second = buffer.sample(2, torch.Generator().manual_seed(0))
        self.assertNotEqual(first._state.data_ptr(), second._state.data_ptr())
        first._state.add_(100.0)
        third = buffer.sample(2, torch.Generator().manual_seed(0))
        self.assertTrue(torch.equal(second.state, third.state))


class ImportPurityTest(unittest.TestCase):
    """Importing the module reads no evidence and launches no runtime."""

    def test_import_has_no_side_effects(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        probe = """
import json, sys
violations = []

def hook(event, args):
    try:
        if event == "open":
            path = str(args[0])
            low = path.lower()
            if "site-packages" in low or "dist-packages" in low:
                return
            if (
                low.endswith(".csv")
                or "/experiments/" in low
                or "action_catalog" in low
            ):
                violations.append([event, path])
        elif event in (
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "socket.socket",
            "socket.connect",
        ):
            violations.append([event, str(args)[:120]])
    except Exception:
        pass

sys.addaudithook(hook)
import rl_agent.splitfusion_hybrid_sac_v1.replay_buffer as m
assert m.DEFAULT_FLOAT_DTYPE.__str__() == "torch.float32"
assert m.Q_CRITIC_NORMALIZER == 9800
print("VIOLATIONS:" + json.dumps(violations))
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(
            completed.returncode, 0, f"probe failed: {completed.stderr[-2000:]}"
        )
        marker = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("VIOLATIONS:")
        ]
        self.assertEqual(len(marker), 1, completed.stdout[-2000:])
        self.assertEqual(json.loads(marker[0][len("VIOLATIONS:") :]), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
