"""Focused adversarial tests for the isolated exact-P95 Run-2-v2 replay."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace

import torch

from . import empirical_contextual_terminal_replay as d1_replay
from .empirical_contextual_contract import fixed_stage_latency_ms
from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    base_p95_expected_utility64_v2,
    shaped_p95_expected_utility64_v2,
)
from .empirical_contextual_exact_p95_run2_replay import (
    ExactP95Run2ShapedTransitionV1,
)
from .empirical_contextual_exact_p95_run2_replay_v2 import (
    EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
    ExactP95Run2BatchV2,
    ExactP95Run2BindingV2Error,
    ExactP95Run2ReplayBindingV2,
    ExactP95Run2ReplayV2,
    ExactP95Run2ReplayV2Error,
    ExactP95Run2RewardBindingV2,
    ExactP95Run2ShapedTransitionV2,
    RUN2_V2_DEADLINE_MS,
    RUN2_V2_DEADLINE_PENALTY,
    RUN2_V2_DEADLINE_PENALTY_HEX,
    RUN2_V2_DEADLINE_PENALTY_UINT64_HEX,
    RUN2_V2_TERMINAL_DISCOUNT,
    _BATCH_ATTESTATION_KEY,
    _BATCH_SOURCE_KEY,
    _emit_cpu_float32,
    _float32_bits_hex,
    _float64_bits_hex,
    exact_p95_run2_reward_spec_document_v2,
)
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT
from .test_empirical_contextual_terminal_replay import (
    make_binding,
    make_transition,
)
from .transaction_identity import canonical_sha256


def with_p95_proxy(transition, latency_p95_ms: float):
    fixed = fixed_stage_latency_ms()
    conditional_p95 = latency_p95_ms - fixed
    policy = transition.result.policy
    conditional_p99 = max(
        float(policy.conditional_feature_uplink_p99_ms), conditional_p95
    )
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


class Run2V2RewardContractTest(unittest.TestCase):
    def test_reward_contract_is_pinned_and_distinct(self) -> None:
        self.assertEqual(RUN2_V2_DEADLINE_MS, 200.0)
        self.assertEqual(RUN2_V2_DEADLINE_PENALTY, 0.5742957622788527)
        self.assertEqual(
            RUN2_V2_DEADLINE_PENALTY.hex(), RUN2_V2_DEADLINE_PENALTY_HEX
        )
        self.assertEqual(
            _float64_bits_hex(RUN2_V2_DEADLINE_PENALTY),
            RUN2_V2_DEADLINE_PENALTY_UINT64_HEX,
        )
        self.assertEqual(
            canonical_sha256(exact_p95_run2_reward_spec_document_v2()),
            EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
        )
        self.assertNotEqual(
            EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
            make_transition().d1_binding.utility_spec_sha256,
        )

    def test_selected_lambda_repairs_the_exact_float32_collision(self) -> None:
        target64 = -0.13886236214769926
        base64 = 0.4307913313758727
        p_admit = 0.9919169403402207
        predecessor = math.nextafter(RUN2_V2_DEADLINE_PENALTY, -math.inf)
        predecessor_emitted = _emit_cpu_float32(base64 - p_admit * predecessor)
        selected_emitted = _emit_cpu_float32(
            base64 - p_admit * RUN2_V2_DEADLINE_PENALTY
        )
        target = _emit_cpu_float32(target64)
        self.assertEqual(_float32_bits_hex(target), "0xbe0e31ef")
        self.assertEqual(_float32_bits_hex(predecessor_emitted), "0xbe0e31ef")
        self.assertEqual(_float32_bits_hex(selected_emitted), "0xbe0e31f0")
        self.assertLess(selected_emitted, target)

    def test_emission_is_explicitly_cpu_even_under_meta_default(self) -> None:
        with torch.device("meta"):
            self.assertEqual(_emit_cpu_float32(1.25), 1.25)
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_registered_evidence_fails_before_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ExactP95Run2BindingV2Error):
                ExactP95Run2RewardBindingV2.from_validated_d1(
                    make_transition(), project_root=directory
                )

    def test_real_evidence_binds_train_prereg_and_gate(self) -> None:
        binding = ExactP95Run2RewardBindingV2.from_validated_d1(
            make_transition()
        )
        binding.require_valid()
        self.assertEqual(
            binding.shaped_reward_spec_sha256,
            EXACT_P95_RUN2_REWARD_SPEC_V2_SHA256,
        )
        self.assertEqual(
            binding.deadline_penalty_uint64_hex,
            RUN2_V2_DEADLINE_PENALTY_UINT64_HEX,
        )
        self.assertEqual(len(binding.validation_summary_file_sha256), 64)
        self.assertNotEqual(
            binding.validation_summary_file_sha256,
            binding.train_summary_file_sha256,
        )

    def test_exact_types_and_post_attestation_tamper_fail(self) -> None:
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
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            ExactP95Run2ShapedTransitionV2.from_validated_d1(subclass)

        shaped = ExactP95Run2ShapedTransitionV2.from_validated_d1(source)
        original = shaped.shaped_reward64
        object.__setattr__(shaped, "shaped_reward64", original + 0.01)
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            shaped.revalidate()
        object.__setattr__(shaped, "shaped_reward64", original)
        shaped.revalidate()

        old_deadline = shaped.reward_binding.deadline_ms
        object.__setattr__(shaped.reward_binding, "deadline_ms", 201.0)
        with self.assertRaises(ExactP95Run2BindingV2Error):
            shaped.revalidate()
        object.__setattr__(shaped.reward_binding, "deadline_ms", old_deadline)
        shaped.revalidate()

        policy = shaped.source_d1_transition.result.policy
        old_quality = policy.q_perc
        object.__setattr__(policy, "q_perc", old_quality + 0.01)
        with self.assertRaises(d1_replay.TransitionRejectedError):
            shaped.revalidate()
        object.__setattr__(policy, "q_perc", old_quality)
        shaped.revalidate()

    def test_p_zero_exact_deadline_and_infeasible_semantics(self) -> None:
        zero = ExactP95Run2ShapedTransitionV2.from_validated_d1(
            with_p95_proxy(make_transition(p_reassembly=0.0), 10_000.0)
        )
        self.assertEqual(zero.source_d1_reward64, -1.0)
        self.assertEqual(zero.p95_base_reward64, -1.0)
        self.assertEqual(zero.shaped_reward64, -1.0)
        self.assertEqual(zero.emitted_reward_float32, -1.0)

        exact = ExactP95Run2ShapedTransitionV2.from_validated_d1(
            with_p95_proxy(make_transition(collection_seq=1), 200.0)
        )
        self.assertEqual(exact.p95_base_reward64, exact.shaped_reward64)

        late = ExactP95Run2ShapedTransitionV2.from_validated_d1(
            with_p95_proxy(make_transition(collection_seq=2), 220.0)
        )
        p = float(late.source_d1_transition.result.policy.p_edge_admission_given_sent)
        self.assertEqual(
            late.shaped_reward64,
            late.p95_base_reward64 - p * RUN2_V2_DEADLINE_PENALTY,
        )

    def test_binary64_operation_order_and_bits_are_preserved(self) -> None:
        source = with_p95_proxy(make_transition(), 220.0)
        shaped = ExactP95Run2ShapedTransitionV2.from_validated_d1(source)
        policy = source.result.policy
        base = base_p95_expected_utility64_v2(
            p_admit=policy.p_edge_admission_given_sent,
            q_perc=policy.q_perc,
            latency_p95_ms=policy.latency_proxy_p95_ms,
        )
        expected = shaped_p95_expected_utility64_v2(
            p_admit=policy.p_edge_admission_given_sent,
            q_perc=policy.q_perc,
            latency_p95_ms=policy.latency_proxy_p95_ms,
            deadline_penalty=RUN2_V2_DEADLINE_PENALTY,
        )
        self.assertEqual(shaped.p95_base_reward64, base)
        self.assertEqual(shaped.shaped_reward64, expected)
        self.assertEqual(shaped.emitted_reward_float32, _emit_cpu_float32(expected))
        document = shaped._document()
        self.assertEqual(
            document["source_d1_reward64_bits_hex"],
            _float64_bits_hex(source.reward),
        )
        self.assertEqual(
            document["p95_base_reward64_bits_hex"], _float64_bits_hex(base)
        )
        self.assertEqual(
            document["shaped_reward64_bits_hex"], _float64_bits_hex(expected)
        )
        self.assertEqual(
            document["emitted_reward_float32_bits_hex"],
            _float32_bits_hex(shaped.emitted_reward_float32),
        )


class Run2V2ReplayTest(unittest.TestCase):
    def _row(self, index: int, *, binding=None, p95: float = 220.0, q_e4=None):
        source = make_transition(
            collection_seq=index,
            q_e4=5000 + index if q_e4 is None else q_e4,
            binding=binding,
            state_offset=float(index),
        )
        return ExactP95Run2ShapedTransitionV2.from_validated_d1(
            with_p95_proxy(source, p95)
        )

    def _filled(self, capacity: int = 4) -> ExactP95Run2ReplayV2:
        replay = ExactP95Run2ReplayV2(capacity)
        for index in range(4):
            replay.insert(self._row(index, p95=190.0 + 10.0 * index))
        return replay

    @staticmethod
    def _rebuild(batch, **changes) -> ExactP95Run2BatchV2:
        values = {
            "_state": batch.state,
            "_mode_id": batch.mode_id,
            "_q_e4": batch.q_e4,
            "_source_d1_reward64": batch.source_d1_reward64,
            "_p95_base_reward64": batch.p95_base_reward64,
            "_shaped_reward64": batch.shaped_reward64,
            "_reward": batch.reward,
            "_discount": batch.discount(),
            "binding": batch.binding,
            "audit": batch.audit,
        }
        values.update(changes)
        return ExactP95Run2BatchV2(**values)

    def test_batch_dtypes_bits_target_and_source_identity(self) -> None:
        batch = self._filled().sample(3, torch.Generator().manual_seed(7))
        self.assertIs(type(batch), ExactP95Run2BatchV2)
        self.assertEqual(batch.state.dtype, torch.float32)
        self.assertEqual(batch.source_d1_reward64.dtype, torch.float64)
        self.assertEqual(batch.p95_base_reward64.dtype, torch.float64)
        self.assertEqual(batch.shaped_reward64.dtype, torch.float64)
        self.assertEqual(batch.reward.dtype, torch.float32)
        self.assertTrue(torch.equal(batch.terminal_target(), batch.reward))
        self.assertTrue(torch.equal(batch.discount(), torch.zeros_like(batch.reward)))
        for index, row in enumerate(batch.audit):
            self.assertEqual(
                _float64_bits_hex(float(batch.source_d1_reward64[index])),
                row["source_d1_reward64_bits_hex"],
            )
            self.assertEqual(
                _float64_bits_hex(float(batch.p95_base_reward64[index])),
                row["p95_base_reward64_bits_hex"],
            )
            self.assertEqual(
                _float64_bits_hex(float(batch.shaped_reward64[index])),
                row["shaped_reward64_bits_hex"],
            )
            self.assertEqual(
                _float32_bits_hex(float(batch.reward[index])),
                row["emitted_reward_float32_bits_hex"],
            )
            self.assertEqual(
                row[_BATCH_ATTESTATION_KEY],
                canonical_sha256(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in (_BATCH_ATTESTATION_KEY, _BATCH_SOURCE_KEY)
                    }
                ),
            )

    def test_batch_rejects_every_tensor_substitution_and_reordering(self) -> None:
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
        old_mode = int(batch.mode_id[0])
        lower, _ = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[old_mode]
        q_only[0] = lower if int(q_only[0]) != lower else lower + 1
        raw = batch.source_d1_reward64
        raw[0] += 0.125
        base = batch.p95_base_reward64
        base[0] += 0.125
        shaped = batch.shaped_reward64
        shaped[0] += 0.125
        reward = batch.reward
        reward[0] = batch.source_d1_reward64[0].to(torch.float32)
        discount = batch.discount()
        discount[0] = 1.0
        cases = (
            {"_state": state},
            {"_mode_id": mode, "_q_e4": q_for_mode},
            {"_q_e4": q_only},
            {"_source_d1_reward64": raw},
            {"_p95_base_reward64": base},
            {"_shaped_reward64": shaped},
            {"_reward": reward},
            {"_discount": discount},
            {"audit": tuple(reversed(batch.audit))},
        )
        for changes in cases:
            with self.subTest(fields=tuple(changes)):
                with self.assertRaises(ExactP95Run2ReplayV2Error):
                    self._rebuild(batch, **changes)

    def test_recomputed_attestation_cannot_authorize_forged_bits(self) -> None:
        batch = self._filled().sample(1, torch.Generator().manual_seed(4))
        row = dict(batch.audit[0])
        source = row.pop(_BATCH_SOURCE_KEY)
        row.pop(_BATCH_ATTESTATION_KEY)
        row["shaped_reward64_bits_hex"] = "0x0000000000000000"
        forged = {
            **row,
            _BATCH_ATTESTATION_KEY: canonical_sha256(row),
            _BATCH_SOURCE_KEY: source,
        }
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            self._rebuild(batch, audit=(forged,))

    def test_public_revalidation_rejects_postconstruction_tamper(self) -> None:
        batch = self._filled().sample(1, torch.Generator().manual_seed(2))
        original = batch.reward
        object.__setattr__(batch, "_reward", batch.source_d1_reward64.to(torch.float32))
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            batch.revalidate()
        object.__setattr__(batch, "_reward", original)
        batch.revalidate()
        self.assertTrue(torch.equal(batch.reward, original))

        source = batch.audit[0][_BATCH_SOURCE_KEY]
        old = source.p95_base_reward64
        object.__setattr__(source, "p95_base_reward64", old + 1.0)
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            batch.revalidate()
        object.__setattr__(source, "p95_base_reward64", old)
        batch.revalidate()

    def test_v1_and_d1_records_are_rejected(self) -> None:
        source = make_transition()
        v1 = ExactP95Run2ShapedTransitionV1.from_validated_d1(source)
        replay = ExactP95Run2ReplayV2(2)
        for foreign in (source, v1):
            with self.assertRaises(ExactP95Run2ReplayV2Error):
                replay.insert(foreign)
        self.assertEqual(len(replay), 0)

    def test_duplicate_and_conflict_indexes_survive_eviction(self) -> None:
        replay = ExactP95Run2ReplayV2(1)
        first = self._row(0, q_e4=5000)
        replay.insert(first)
        replay.insert(self._row(1))
        self.assertEqual(len(replay), 1)
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            replay.insert(first)
        conflict = self._row(0, q_e4=5001)
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            replay.insert(conflict)
        self.assertEqual((len(replay), replay.accepted_count), (1, 2))

    def test_binding_mismatch_is_atomic(self) -> None:
        replay = ExactP95Run2ReplayV2(2)
        replay.insert(self._row(0))
        before = (len(replay), replay.accepted_count, replay.binding)
        foreign = self._row(1, binding=make_binding(network_sha="a" * 64))
        with self.assertRaises(ExactP95Run2BindingV2Error):
            replay.insert(foreign)
        self.assertEqual((len(replay), replay.accepted_count, replay.binding), before)

    def test_float32_state_overflow_is_refused_before_any_mutation(self) -> None:
        source = make_transition(state_offset=1e300)
        shaped = ExactP95Run2ShapedTransitionV2.from_validated_d1(source)
        replay = ExactP95Run2ReplayV2(2)
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            replay.insert(shaped)
        self.assertEqual(len(replay), 0)
        self.assertEqual(replay.accepted_count, 0)
        self.assertEqual(replay.evicted_count, 0)
        self.assertIsNone(replay.binding)

    def test_sampling_is_deterministic_clone_isolated_and_rng_local(self) -> None:
        replay = self._filled()
        torch.manual_seed(77)
        global_before = torch.get_rng_state().clone()
        first_generator = torch.Generator().manual_seed(123)
        unrelated = torch.Generator().manual_seed(456)
        unrelated_before = unrelated.get_state().clone()
        first = replay.sample(3, first_generator)
        second = replay.sample(3, torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(first.state, second.state))
        self.assertTrue(torch.equal(first.reward, second.reward))
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        self.assertTrue(torch.equal(unrelated_before, unrelated.get_state()))
        exposed = first.reward
        exposed.zero_()
        self.assertFalse(torch.equal(exposed, first.reward))

    def test_capacity_generator_and_exact_batch_type_guards(self) -> None:
        for value in (0, -1, True, 1.5, None):
            with self.assertRaises(ExactP95Run2ReplayV2Error):
                ExactP95Run2ReplayV2(value)
        replay = self._filled()
        for generator in (None, torch.default_generator, "rng"):
            with self.assertRaises(ExactP95Run2ReplayV2Error):
                replay.sample(1, generator)
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            replay.sample(5, torch.Generator().manual_seed(1))

        batch = replay.sample(1, torch.Generator().manual_seed(1))

        class BatchSubclass(ExactP95Run2BatchV2):
            pass

        subclass = BatchSubclass(
            batch.state,
            batch.mode_id,
            batch.q_e4,
            batch.source_d1_reward64,
            batch.p95_base_reward64,
            batch.shaped_reward64,
            batch.reward,
            batch.discount(),
            batch.binding,
            batch.audit,
        )
        with self.assertRaises(ExactP95Run2ReplayV2Error):
            subclass.revalidate()

    def test_replay_binding_is_v2_and_carries_both_dtypes(self) -> None:
        row = self._row(0)
        binding = ExactP95Run2ReplayBindingV2.from_transition(row)
        self.assertEqual(binding.float_dtype, str(torch.float32))
        self.assertEqual(binding.diagnostic_float_dtype, str(torch.float64))
        self.assertEqual(binding.terminal_discount, RUN2_V2_TERMINAL_DISCOUNT)
        binding.require_valid()


if __name__ == "__main__":
    unittest.main()
