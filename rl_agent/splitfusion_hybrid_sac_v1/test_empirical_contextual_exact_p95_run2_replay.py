"""Focused Phase-1 tests for the isolated exact-P95 Run-2 replay."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

import torch

from . import empirical_contextual_terminal_replay as d1_replay
from . import empirical_contextual_terminal_trainer as d1_trainer
from .empirical_contextual_contract import fixed_stage_latency_ms
from .empirical_contextual_exact_p95_run2_replay import (
    EXACT_P95_RUN2_REWARD_SPEC_SHA256,
    ExactP95Run2BatchV1,
    ExactP95Run2BindingError,
    ExactP95Run2ReplayError,
    ExactP95Run2ReplayV1,
    ExactP95Run2RewardBindingV1,
    ExactP95Run2ShapedTransitionV1,
    RUN2_DEADLINE_MS,
    RUN2_DEADLINE_PENALTY,
    RUN2_DEADLINE_PENALTY_HEX,
    RUN2_TERMINAL_DISCOUNT,
    RUN2_PREREGISTRATION_FILE_SHA256,
    exact_p95_run2_reward_spec_document,
)
from .test_empirical_contextual_terminal_replay import make_binding, make_transition
from .test_empirical_contextual_terminal_trainer import build_trainer, make_batch
from .transaction_identity import canonical_sha256
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT


def with_p95_proxy(transition, latency_p95_ms: float):
    fixed = fixed_stage_latency_ms()
    conditional_p95 = latency_p95_ms - fixed
    policy = transition.result.policy
    conditional_p99 = max(float(policy.conditional_feature_uplink_p99_ms), conditional_p95)
    changed = replace(
        policy,
        conditional_feature_uplink_p95_ms=conditional_p95,
        conditional_feature_uplink_p99_ms=conditional_p99,
        latency_proxy_p95_ms=latency_p95_ms,
        latency_proxy_p99_ms=fixed + conditional_p99,
        modeled_budget_miss_p95=latency_p95_ms > policy.deadline_ms,
        modeled_budget_miss_p99=(fixed + conditional_p99) > policy.deadline_ms,
    )
    return d1_replay.EmpiricalTerminalTransitionV1.from_d1(
        collection_session_uuid=transition.collection_session_uuid,
        collection_seq=transition.collection_seq,
        observation=transition.observation,
        action=transition.action,
        result=replace(transition.result, policy=changed),
        d1_binding=transition.d1_binding,
    )


class Run2RewardProvenanceTest(unittest.TestCase):
    def test_reward_spec_and_float_are_pinned_and_distinct_from_d1(self) -> None:
        self.assertEqual(RUN2_DEADLINE_MS, 200.0)
        self.assertEqual(RUN2_DEADLINE_PENALTY, 0.5742957604173842)
        self.assertEqual(RUN2_DEADLINE_PENALTY.hex(), RUN2_DEADLINE_PENALTY_HEX)
        self.assertEqual(
            canonical_sha256(exact_p95_run2_reward_spec_document()),
            EXACT_P95_RUN2_REWARD_SPEC_SHA256,
        )
        source = make_transition()
        binding = ExactP95Run2RewardBindingV1.from_validated_d1(source)
        self.assertNotEqual(
            binding.shaped_reward_spec_sha256,
            binding.source_d1_utility_spec_sha256,
        )
        self.assertEqual(binding.deadline_penalty_float_hex, RUN2_DEADLINE_PENALTY_HEX)
        self.assertEqual(
            binding.run2_preregistration_file_sha256,
            RUN2_PREREGISTRATION_FILE_SHA256,
        )

    def test_missing_registered_evidence_fails_before_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ExactP95Run2BindingError):
                ExactP95Run2RewardBindingV1.from_validated_d1(
                    make_transition(), project_root=directory
                )

    def test_adapter_preserves_original_transition_and_both_rewards(self) -> None:
        source = with_p95_proxy(make_transition(), 220.0)
        original_reward = source.reward
        shaped = ExactP95Run2ShapedTransitionV1.from_validated_d1(source)
        self.assertIs(shaped.source_d1_transition, source)
        self.assertEqual(shaped.source_d1_reward, original_reward)
        self.assertEqual(source.reward, original_reward)
        self.assertEqual(
            shaped.shaped_reward,
            shaped.p95_base_reward
            - float(source.result.policy.p_edge_admission_given_sent)
            * RUN2_DEADLINE_PENALTY,
        )
        self.assertNotEqual(shaped.source_d1_reward, shaped.shaped_reward)
        shaped.revalidate()

    def test_p_zero_stays_minus_one_even_above_deadline(self) -> None:
        source = with_p95_proxy(make_transition(p_reassembly=0.0), 10_000.0)
        shaped = ExactP95Run2ShapedTransitionV1.from_validated_d1(source)
        self.assertEqual(shaped.source_d1_reward, -1.0)
        self.assertEqual(shaped.p95_base_reward, -1.0)
        self.assertEqual(shaped.shaped_reward, -1.0)

    def test_exact_deadline_has_no_binary_penalty(self) -> None:
        source = with_p95_proxy(make_transition(), RUN2_DEADLINE_MS)
        shaped = ExactP95Run2ShapedTransitionV1.from_validated_d1(source)
        self.assertEqual(shaped.shaped_reward, shaped.p95_base_reward)

    def test_exact_types_and_attestations_fail_closed(self) -> None:
        source = make_transition()

        class D1Subclass(d1_replay.EmpiricalTerminalTransitionV1):
            pass

        subclass = D1Subclass(
            source.collection_session_uuid,
            source.collection_seq,
            source.observation,
            source.action,
            source.result,
            source.d1_binding,
            source._attestation_sha256,
        )
        with self.assertRaises(ExactP95Run2ReplayError):
            ExactP95Run2ShapedTransitionV1.from_validated_d1(subclass)

        shaped = ExactP95Run2ShapedTransitionV1.from_validated_d1(source)
        original = shaped.shaped_reward
        object.__setattr__(shaped, "shaped_reward", original + 0.01)
        with self.assertRaises(ExactP95Run2ReplayError):
            shaped.revalidate()
        object.__setattr__(shaped, "shaped_reward", original)
        shaped.revalidate()

        original_deadline = shaped.reward_binding.deadline_ms
        object.__setattr__(shaped.reward_binding, "deadline_ms", 201.0)
        with self.assertRaises(ExactP95Run2BindingError):
            shaped.revalidate()
        object.__setattr__(shaped.reward_binding, "deadline_ms", original_deadline)
        shaped.revalidate()

    def test_reward_binding_can_be_reused_only_for_the_same_d1_problem(self) -> None:
        first = make_transition(collection_seq=0)
        binding = ExactP95Run2RewardBindingV1.from_validated_d1(first)
        second = make_transition(collection_seq=1)
        shaped = ExactP95Run2ShapedTransitionV1.from_validated_d1(
            second, reward_binding=binding
        )
        self.assertIs(shaped.reward_binding, binding)
        foreign = make_transition(collection_seq=2, binding=make_binding(network_sha="a" * 64))
        with self.assertRaises(ExactP95Run2BindingError):
            ExactP95Run2ShapedTransitionV1.from_validated_d1(
                foreign, reward_binding=binding
            )


class Run2ReplayTest(unittest.TestCase):
    def _row(self, index: int, *, binding=None, p95: float = 220.0):
        source = make_transition(
            collection_seq=index,
            q_e4=5000 + index,
            binding=binding,
            state_offset=float(index),
        )
        return ExactP95Run2ShapedTransitionV1.from_validated_d1(
            with_p95_proxy(source, p95)
        )

    def _filled(self) -> ExactP95Run2ReplayV1:
        replay = ExactP95Run2ReplayV1(4)
        for index in range(4):
            replay.insert(self._row(index, p95=190.0 + 10.0 * index))
        return replay

    @staticmethod
    def _rebuild_batch(batch, **changes) -> ExactP95Run2BatchV1:
        values = {
            "_state": batch.state,
            "_mode_id": batch.mode_id,
            "_q_e4": batch.q_e4,
            "_source_d1_reward": batch.source_d1_reward,
            "_p95_base_reward": batch.p95_base_reward,
            "_reward": batch.reward,
            "_discount": batch.discount(),
            "binding": batch.binding,
            "audit": batch.audit,
        }
        values.update(changes)
        return ExactP95Run2BatchV1(**values)

    def test_replay_preserves_sources_and_rejects_foreign_records(self) -> None:
        replay = ExactP95Run2ReplayV1(2)
        row = self._row(0)
        replay.insert(row)
        self.assertIs(replay.resident_sources()[0], row.source_d1_transition)
        before = (len(replay), replay.accepted_count, replay.binding)
        with self.assertRaises(ExactP95Run2ReplayError):
            replay.insert(row.source_d1_transition)
        with self.assertRaises(ExactP95Run2ReplayError):
            replay.insert(row)
        self.assertEqual((len(replay), replay.accepted_count, replay.binding), before)

    def test_binding_mismatch_is_atomic(self) -> None:
        replay = ExactP95Run2ReplayV1(2)
        replay.insert(self._row(0))
        before = (len(replay), replay.accepted_count, replay.binding)
        other = make_binding(network_sha="a" * 64)
        with self.assertRaises(ExactP95Run2BindingError):
            replay.insert(self._row(1, binding=other))
        self.assertEqual((len(replay), replay.accepted_count, replay.binding), before)

    def test_batch_emits_terminal_discount_and_unchanged_target_reward(self) -> None:
        replay = self._filled()
        torch.manual_seed(77)
        global_before = torch.get_rng_state().clone()
        batch = replay.sample(3, torch.Generator().manual_seed(9))
        self.assertIs(type(batch), ExactP95Run2BatchV1)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertTrue(
            torch.equal(
                batch.discount(),
                torch.full_like(batch.reward, RUN2_TERMINAL_DISCOUNT),
            )
        )
        self.assertTrue(torch.equal(batch.terminal_target(), batch.reward))
        self.assertTrue(
            all(
                row["reward_binding_sha256"]
                == batch.binding.reward_binding.canonical_sha256()
                for row in batch.audit
            )
        )
        self.assertTrue(
            all(
                row["row_attestation_sha256"]
                == canonical_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in (
                            "row_attestation_sha256",
                            "_authenticated_run2_transition",
                        )
                    }
                )
                for row in batch.audit
            )
        )

    def test_batch_rejects_tensor_substitution_and_audit_reordering(self) -> None:
        batch = self._filled().sample(4, torch.Generator().manual_seed(8))

        state = batch.state
        state[0, 0] += 0.125

        mode = batch.mode_id
        q_for_mode = batch.q_e4
        mode[0] = (int(mode[0]) + 1) % len(
            MODELED_SMOKE_SUPPORT.mode_q_e4_bounds
        )
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[int(mode[0])]
        q_for_mode[0] = (lower + upper) // 2

        q_only = batch.q_e4
        original_mode = int(batch.mode_id[0])
        lower, upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[original_mode]
        q_only[0] = lower if int(q_only[0]) != lower else lower + 1

        p95_base = batch.p95_base_reward
        p95_base[0] += 0.125

        discount = batch.discount()
        discount[0] = 0.5

        cases = (
            {"_reward": batch.source_d1_reward},
            {"_state": state},
            {"_mode_id": mode, "_q_e4": q_for_mode},
            {"_q_e4": q_only},
            {"_p95_base_reward": p95_base},
            {"_discount": discount},
            {"audit": tuple(reversed(batch.audit))},
        )
        for changes in cases:
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(ExactP95Run2ReplayError):
                    self._rebuild_batch(batch, **changes)

    def test_batch_revalidation_rejects_postconstruction_raw_d1_target(self) -> None:
        replay = ExactP95Run2ReplayV1(1)
        replay.insert(self._row(0, p95=220.0))
        batch = replay.sample(1, torch.Generator().manual_seed(3))
        original = batch.reward
        forged = batch.source_d1_reward
        self.assertFalse(torch.equal(original, forged))

        object.__setattr__(batch, "_reward", forged)
        with self.assertRaises(ExactP95Run2ReplayError):
            batch.revalidate()

        object.__setattr__(batch, "_reward", original)
        batch.revalidate()
        self.assertTrue(torch.equal(batch.reward, original))

    def test_existing_d1_batch_and_trainer_cannot_mislabel_run2_reward(self) -> None:
        run2_batch = self._filled().sample(2, torch.Generator().manual_seed(4))
        with self.assertRaises(d1_replay.TerminalReplayError):
            d1_replay.EmpiricalTerminalBatchV1(
                _state=run2_batch.state,
                _mode_id=run2_batch.mode_id,
                _q_e4=run2_batch.q_e4,
                _reward=run2_batch.reward,
                binding=run2_batch.binding,
                audit=run2_batch.audit,
            )
        d1_batch = make_batch(2)
        existing_trainer = build_trainer(d1_batch)
        with self.assertRaises(d1_trainer.TrainerPreflightError):
            existing_trainer.update_once(run2_batch)
        self.assertEqual(existing_trainer.update_count, 0)

    def test_sampling_is_deterministic_clone_isolated_and_bound(self) -> None:
        replay = self._filled()
        first = replay.sample(3, torch.Generator().manual_seed(123))
        second = replay.sample(3, torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(first.state, second.state))
        self.assertTrue(torch.equal(first.reward, second.reward))
        exposed = first.reward
        exposed.zero_()
        self.assertFalse(torch.equal(exposed, first.reward))
        self.assertEqual(first.binding, second.binding)
        first.binding.require_valid()

    def test_capacity_and_generator_guards(self) -> None:
        for value in (0, -1, True, 1.5, None):
            with self.assertRaises(ExactP95Run2ReplayError):
                ExactP95Run2ReplayV1(value)
        replay = self._filled()
        for generator in (None, torch.default_generator, "rng"):
            with self.assertRaises(ExactP95Run2ReplayError):
                replay.sample(1, generator)
        with self.assertRaises(ExactP95Run2ReplayError):
            replay.sample(5, torch.Generator().manual_seed(1))

    def test_batch_binding_and_discount_tamper_fail_closed(self) -> None:
        batch = self._filled().sample(2, torch.Generator().manual_seed(3))
        kwargs = {
            "_state": batch.state,
            "_mode_id": batch.mode_id,
            "_q_e4": batch.q_e4,
            "_source_d1_reward": batch.source_d1_reward,
            "_p95_base_reward": batch.p95_base_reward,
            "_reward": batch.reward,
            "_discount": torch.ones_like(batch.discount()),
            "binding": batch.binding,
            "audit": batch.audit,
        }
        with self.assertRaises(ExactP95Run2ReplayError):
            ExactP95Run2BatchV1(**kwargs)
        with self.assertRaises(ExactP95Run2BindingError):
            ExactP95Run2BatchV1(**{**kwargs, "_discount": batch.discount(), "binding": object()})
        with self.assertRaises(ExactP95Run2ReplayError):
            ExactP95Run2BatchV1(
                **{
                    **kwargs,
                    "_discount": batch.discount(),
                    "_q_e4": torch.zeros_like(batch.q_e4),
                }
            )


if __name__ == "__main__":
    unittest.main()
