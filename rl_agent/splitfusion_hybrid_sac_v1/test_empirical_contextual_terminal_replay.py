"""CPU-only adversarial tests for the private D1 terminal replay."""

from __future__ import annotations

import dataclasses
import unittest
import uuid
from dataclasses import replace

import torch

from . import empirical_contextual_contract as contract
from . import empirical_contextual_environment as environment
from . import empirical_contextual_terminal_replay as replay
from .action_contract import CATALOG_SHA256
from .state_reward_transition_contract import (
    POLICY_FEATURE_ORDER,
    POLICY_OBSERVATION_DEPLOYABILITY,
)


COLLECTION_UUID = "801de129-3b96-44f9-854f-3fd792e28d9d"


def make_binding(*, network_sha: str = "3" * 64):
    return contract.EmpiricalPilotBindingV1(
        corrected_p40_binding_sha256="1" * 64,
        quality_surface_binding_sha256="2" * 64,
        network_surrogate_sha256=network_sha,
        radio_calibration_sha256="4" * 64,
        modeled_smoke_support_sha256=contract.MODELED_SMOKE_SUPPORT_SHA256,
        utility_spec_sha256=contract.PILOT_UTILITY_SPEC_SHA256,
        normalization_spec_sha256="5" * 64,
        freshness_policy_sha256="6" * 64,
        eligible_fit_context_index_sha256="7" * 64,
        action_catalog_sha256=CATALOG_SHA256,
        surface_qualification_report_sha256=(
            contract.SURFACE_QUALIFICATION_REPORT_SHA256
        ),
        profile_order_rng_contract_sha256=(
            contract.PROFILE_ORDER_RNG_CONTRACT_SHA256
        ),
        implementation_bundle_sha256="8" * 64,
    )


def make_transition(
    *,
    collection_seq: int = 0,
    mode_id: int = 8,
    q_e4: int = 5000,
    binding=None,
    state_offset: float = 0.0,
    p_reassembly: float = 0.9,
    p_admit: float = 0.8,
    q_perc: float = 0.7,
):
    binding = make_binding() if binding is None else binding
    state = tuple(float(state_offset + index / 100.0) for index in range(31))
    observation = environment.EmpiricalPolicyObservationV1(
        values=state,
        policy_feature_order=tuple(POLICY_FEATURE_ORDER),
        environment_binding_sha256=binding.canonical_sha256(),
        normalization_spec_sha256=binding.normalization_spec_sha256,
        freshness_policy_sha256=binding.freshness_policy_sha256,
        deployability=POLICY_OBSERVATION_DEPLOYABILITY,
    )
    action = contract.require_supported_action(mode_id, q_e4)
    p_service = p_reassembly * p_admit
    p50, p95, p99 = 50.0, 75.0, 100.0
    fixed = contract.fixed_stage_latency_ms()
    proxy = fixed + p50
    proxy95 = fixed + p95
    proxy99 = fixed + p99
    reward = contract.PILOT_UTILITY_SPEC.expected_utility(
        p_edge_admission_given_sent=p_service,
        q_perc=q_perc,
        latency_proxy_ms=proxy,
    )
    policy = environment.EmpiricalOutcomeV1(
        reward=reward,
        status="MODELED_EXPECTED_UTILITY_DEFINED",
        terminated=True,
        truncated=False,
        q_perc=q_perc,
        p_complete_reassembly_given_sent=p_reassembly,
        p_edge_admission_given_reassembled=p_admit,
        p_edge_admission_given_sent=p_service,
        conditional_feature_uplink_p50_ms=p50,
        conditional_feature_uplink_p95_ms=p95,
        conditional_feature_uplink_p99_ms=p99,
        fixed_stage_latency_ms=fixed,
        fixed_latency_stages_ms=contract.FIXED_END_TO_FEEDBACK_STAGES_MS,
        latency_proxy_ms=proxy,
        latency_proxy_p95_ms=proxy95,
        latency_proxy_p99_ms=proxy99,
        deadline_ms=contract.PILOT_UTILITY_SPEC.deadline_ms,
        modeled_budget_miss=proxy > contract.PILOT_UTILITY_SPEC.deadline_ms,
        modeled_budget_miss_p95=(
            proxy95 > contract.PILOT_UTILITY_SPEC.deadline_ms
        ),
        modeled_budget_miss_p99=(
            proxy99 > contract.PILOT_UTILITY_SPEC.deadline_ms
        ),
        estimator=contract.PILOT_UTILITY_SPEC.estimator,
        service_non_admission_semantics="unit-test-non-admission-semantics",
        timeout_probability_status="unit-test-no-timeout-probability",
    )
    audit = environment.EmpiricalStepAuditV1(
        sample_id=f"sample-{collection_seq}",
        episode_id="episode-unit-test",
        frame_id=1000 + collection_seq,
        hidden_network_profile="FAVORABLE_STABLE",
        hidden_radio_csv_row_number=11,
        hidden_trace_id="trace-unit-test",
        hidden_trace_step_index=collection_seq,
        hidden_target_snr_db=20.0,
        hidden_radio_row_sha256="9" * 64,
        surface_evidence_status="EMPIRICAL_FIT_SURFACE_INTERPOLATION",
        total_transmitted_bytes=12345.0,
        datagram_count=2,
        network_evidence_class="EMPIRICAL_288_CELL_SURROGATE",
        utility_spec_sha256=binding.utility_spec_sha256,
        executed_mode_id=mode_id,
        executed_q_e4=q_e4,
    )
    result = environment.EmpiricalStepResultV1(policy=policy, audit=audit)
    return replay.EmpiricalTerminalTransitionV1.from_d1(
        collection_session_uuid=COLLECTION_UUID,
        collection_seq=collection_seq,
        observation=observation,
        action=action,
        result=result,
        d1_binding=binding,
    )


class TransitionContractTest(unittest.TestCase):
    def test_valid_transition_revalidates_and_binds_full_d1_record(self) -> None:
        transition = make_transition()
        transition.revalidate()
        self.assertEqual(transition.reward, transition.result.policy.reward)
        self.assertEqual(len(transition.canonical_sha256()), 64)
        self.assertEqual(
            transition.observation.environment_binding_sha256,
            transition.d1_binding.canonical_sha256(),
        )

    def test_unrewarded_unsupported_and_nonterminal_records_are_refused(self) -> None:
        valid = make_transition()
        unavailable = replace(
            valid.result.policy,
            reward=None,
            status="NETWORK_OR_CONDITIONAL_LATENCY_UNSUPPORTED_NO_REWARD",
        )
        for changed in (
            replace(valid.result, policy=unavailable),
            replace(valid.result, policy=replace(valid.result.policy, terminated=False)),
            replace(valid.result, policy=replace(valid.result.policy, truncated=True)),
        ):
            with self.assertRaises(replay.TransitionRejectedError):
                replay.EmpiricalTerminalTransitionV1.from_d1(
                    collection_session_uuid=COLLECTION_UUID,
                    collection_seq=90,
                    observation=valid.observation,
                    action=valid.action,
                    result=changed,
                    d1_binding=valid.d1_binding,
                )
        with self.assertRaises(contract.ActionSupportError):
            contract.require_supported_action(8, 100)

    def test_reward_probability_latency_and_action_are_rederived(self) -> None:
        valid = make_transition()
        mutations = (
            replace(valid.result.policy, reward=valid.reward + 0.01),
            replace(valid.result.policy, p_edge_admission_given_sent=0.1),
            replace(valid.result.policy, latency_proxy_ms=999.0),
            replace(valid.result.policy, modeled_budget_miss=True),
        )
        for index, policy in enumerate(mutations):
            with self.assertRaises(replay.TransitionRejectedError):
                replay.EmpiricalTerminalTransitionV1.from_d1(
                    collection_session_uuid=COLLECTION_UUID,
                    collection_seq=100 + index,
                    observation=valid.observation,
                    action=valid.action,
                    result=replace(valid.result, policy=policy),
                    d1_binding=valid.d1_binding,
                )
        mismatched_audit = replace(valid.result.audit, executed_q_e4=6000)
        with self.assertRaises(replay.TransitionRejectedError):
            replay.EmpiricalTerminalTransitionV1.from_d1(
                collection_session_uuid=COLLECTION_UUID,
                collection_seq=110,
                observation=valid.observation,
                action=valid.action,
                result=replace(valid.result, audit=mismatched_audit),
                d1_binding=valid.d1_binding,
            )

    def test_post_attestation_tamper_fails_and_can_be_restored(self) -> None:
        transition = make_transition(collection_seq=4)
        original = transition.collection_seq
        object.__setattr__(transition, "collection_seq", 5)
        with self.assertRaises(replay.TransitionRejectedError):
            transition.revalidate()
        object.__setattr__(transition, "collection_seq", original)
        transition.revalidate()


class ReplayBoundaryTest(unittest.TestCase):
    def test_fifo_binding_and_audit(self) -> None:
        store = replay.EmpiricalTerminalReplayV1(capacity=2)
        rows = [make_transition(collection_seq=index) for index in range(3)]
        for row in rows:
            store.insert(row)
        self.assertEqual(len(store), 2)
        self.assertEqual(store.accepted_count, 3)
        self.assertEqual(store.evicted_count, 1)
        self.assertEqual(store.seen_digest_count, 3)
        self.assertEqual(store.seen_identity_count, 3)
        self.assertEqual(
            tuple(record["collection_seq"] for record in store.resident_audit()),
            (1, 2),
        )
        self.assertEqual(
            store.binding.d1_binding, rows[0].d1_binding
        )

    def test_duplicate_and_identity_conflict_survive_eviction(self) -> None:
        store = replay.EmpiricalTerminalReplayV1(capacity=1)
        first = make_transition(collection_seq=1)
        store.insert(first)
        store.insert(make_transition(collection_seq=2))
        with self.assertRaises(replay.DuplicateTransitionError):
            store.insert(first)
        conflict = make_transition(collection_seq=1, q_e4=6000)
        with self.assertRaises(replay.IdentityConflictError):
            store.insert(conflict)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.accepted_count, 2)

    def test_same_d1_context_is_allowed_under_new_collection_identity(self) -> None:
        first = make_transition(collection_seq=0)
        second = replay.EmpiricalTerminalTransitionV1.from_d1(
            collection_session_uuid=COLLECTION_UUID,
            collection_seq=1,
            observation=first.observation,
            action=first.action,
            result=first.result,
            d1_binding=first.d1_binding,
        )
        store = replay.EmpiricalTerminalReplayV1(capacity=2)
        store.insert(first)
        store.insert(second)
        self.assertEqual(len(store), 2)

    def test_foreign_types_subclasses_and_production_shapes_are_refused(self) -> None:
        class Subclass(replay.EmpiricalTerminalTransitionV1):
            pass

        valid = make_transition()
        subclass = Subclass(
            valid.collection_session_uuid,
            valid.collection_seq,
            valid.observation,
            valid.action,
            valid.result,
            valid.d1_binding,
            valid._attestation_sha256,
        )
        for candidate in (
            valid.result,
            valid.observation,
            {"state": valid.observation.values},
            subclass,
            None,
        ):
            store = replay.EmpiricalTerminalReplayV1(capacity=2)
            with self.assertRaises(replay.TransitionRejectedError):
                store.insert(candidate)
            self.assertEqual(len(store), 0)
            self.assertEqual(store.seen_digest_count, 0)

    def test_binding_mismatch_and_float32_overflow_are_zero_mutation(self) -> None:
        store = replay.EmpiricalTerminalReplayV1(capacity=3)
        first = make_transition(collection_seq=0)
        store.insert(first)
        before = (
            len(store),
            store.accepted_count,
            store.seen_digest_count,
            store.seen_identity_count,
            store.binding,
        )
        other_binding = make_binding(network_sha="a" * 64)
        with self.assertRaises(replay.BindingMismatchError):
            store.insert(
                make_transition(collection_seq=1, binding=other_binding)
            )
        self.assertEqual(
            (
                len(store),
                store.accepted_count,
                store.seen_digest_count,
                store.seen_identity_count,
                store.binding,
            ),
            before,
        )

        overflow = make_transition(collection_seq=2, state_offset=1e300)
        with self.assertRaises(replay.NonFiniteInReplayDtypeError):
            store.insert(overflow)
        self.assertEqual(len(store), before[0])

    def test_rejected_tamper_does_not_poison_lifetime_indexes(self) -> None:
        store = replay.EmpiricalTerminalReplayV1(capacity=2)
        transition = make_transition(collection_seq=7)
        original = transition.collection_seq
        object.__setattr__(transition, "collection_seq", 8)
        with self.assertRaises(replay.TransitionRejectedError):
            store.insert(transition)
        self.assertEqual(store.seen_digest_count, 0)
        self.assertEqual(store.seen_identity_count, 0)
        object.__setattr__(transition, "collection_seq", original)
        store.insert(transition)
        self.assertEqual(len(store), 1)

    def test_capacity_contract(self) -> None:
        for invalid in (0, -1, True, 1.5, None):
            with self.assertRaises(replay.TerminalReplayError):
                replay.EmpiricalTerminalReplayV1(invalid)


class SamplingTest(unittest.TestCase):
    def _filled(self, count: int = 8):
        store = replay.EmpiricalTerminalReplayV1(capacity=count)
        for index in range(count):
            store.insert(
                make_transition(
                    collection_seq=index,
                    state_offset=float(index),
                    q_e4=5000 + index,
                )
            )
        return store

    def test_local_generator_is_deterministic_and_global_rng_unchanged(self) -> None:
        store = self._filled()
        torch.manual_seed(771)
        global_before = torch.get_rng_state().clone()
        first = store.sample(4, torch.Generator().manual_seed(123))
        global_after = torch.get_rng_state().clone()
        second = store.sample(4, torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(global_before, global_after))
        self.assertTrue(torch.equal(first.state, second.state))
        self.assertTrue(torch.equal(first.q_e4, second.q_e4))

    def test_default_and_invalid_generators_are_refused_without_advancing_global(self) -> None:
        store = self._filled(2)
        torch.manual_seed(31)
        before = torch.get_rng_state().clone()
        for generator in (None, torch.default_generator, "rng"):
            with self.assertRaises(replay.ReplaySamplingError):
                store.sample(1, generator)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaises(replay.ReplaySamplingError):
            store.sample(3, torch.Generator().manual_seed(1))

    def test_batch_is_clone_isolated_and_has_no_trajectory_fields(self) -> None:
        batch = self._filled(4).sample(3, torch.Generator().manual_seed(4))
        originals = {
            "state": batch.state,
            "mode_id": batch.mode_id,
            "q_e4": batch.q_e4,
            "reward": batch.reward,
        }
        for name, original in originals.items():
            exposed = getattr(batch, name)
            exposed.fill_(0)
            self.assertTrue(torch.equal(getattr(batch, name), original))
            self.assertNotEqual(exposed.data_ptr(), getattr(batch, name).data_ptr())
        fields = {item.name for item in dataclasses.fields(batch)}
        forbidden = {
            "next_state",
            "_next_state",
            "duration",
            "_duration",
            "discount",
            "_discount",
            "bootstrap",
            "_bootstrap",
        }
        self.assertTrue(fields.isdisjoint(forbidden))
        self.assertEqual(batch.state.dtype, torch.float32)
        self.assertEqual(batch.mode_id.dtype, torch.int64)
        self.assertEqual(batch.q_e4.dtype, torch.int64)
        self.assertTrue(
            torch.equal(
                batch.q_normalized_executed,
                batch.q_e4.to(torch.float32) / 9800.0,
            )
        )

    def test_constructor_clones_aliased_inputs(self) -> None:
        sampled = self._filled(2).sample(2, torch.Generator().manual_seed(0))
        # Mode 11 with q_e4=11 is valid, letting one aliased constructor
        # tensor exercise both semantically different fields.
        shared = torch.tensor([11, 11], dtype=torch.int64)
        batch = replay.EmpiricalTerminalBatchV1(
            _state=sampled.state,
            _mode_id=shared,
            _q_e4=shared,
            _reward=sampled.reward,
            binding=sampled.binding,
            audit=sampled.audit,
        )
        self.assertNotEqual(batch._mode_id.data_ptr(), batch._q_e4.data_ptr())

    def test_batch_constructor_independently_rejects_malformed_tensors(self) -> None:
        sampled = self._filled(3).sample(3, torch.Generator().manual_seed(0))
        base = {
            "_state": sampled.state,
            "_mode_id": sampled.mode_id,
            "_q_e4": sampled.q_e4,
            "_reward": sampled.reward,
            "binding": sampled.binding,
            "audit": sampled.audit,
        }
        candidates = []
        wrong_shape = dict(base)
        wrong_shape["_state"] = sampled.state[:, :-1]
        candidates.append(wrong_shape)
        wrong_dtype = dict(base)
        wrong_dtype["_state"] = sampled.state.to(torch.float64)
        candidates.append(wrong_dtype)
        nonfinite = dict(base)
        broken_state = sampled.state
        broken_state[0, 0] = float("inf")
        nonfinite["_state"] = broken_state
        candidates.append(nonfinite)
        unsupported = dict(base)
        broken_mode = sampled.mode_id
        broken_mode[0] = 0
        broken_q = sampled.q_e4
        broken_q[0] = 0
        unsupported["_mode_id"] = broken_mode
        unsupported["_q_e4"] = broken_q
        candidates.append(unsupported)
        wrong_audit = dict(base)
        wrong_audit["audit"] = sampled.audit[:-1]
        candidates.append(wrong_audit)
        for candidate in candidates:
            with self.assertRaises(replay.TerminalReplayError):
                replay.EmpiricalTerminalBatchV1(**candidate)


if __name__ == "__main__":
    unittest.main()
