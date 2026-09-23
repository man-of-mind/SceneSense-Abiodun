"""Adversarial integration tests for the bounded Run-3 training path."""

from __future__ import annotations

import copy
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import torch

from .action_contract import round_half_up_q_e4
from .empirical_contextual_contract import require_supported_action
from .empirical_contextual_fit_partition import FIT_VALIDATION_SPLIT
from .empirical_contextual_partitioned_environment import (
    PartitionedEmpiricalOneStepEnvironmentV1,
)
from .empirical_contextual_run3_reward import (
    QuantileLatencyProxyV1,
    Run3CounterRngV1,
    Run3TerminalOutcome,
    evaluate_run3_reward,
    sample_run3_simulator_outcome,
)
from .empirical_contextual_run3_runner import (
    RUN3_PREFLIGHT_RESULT_SHA256,
    RUN3_REGISTERED_CONFIG,
    _REGISTERED_DECISION_LINEAGE_SEED17_BINDING_SHA256,
    _REGISTERED_DECISION_LINEAGE_SEED17_SESSION_UUID,
    _decision_lineage_binding_sha256,
    Run3RunnerConfigV1,
    Run3TrainingRunnerV1,
)
from .empirical_contextual_run3_terminal_replay import (
    Run3DuplicateTransition,
    Run3IdentityConflict,
    Run3ReplayError,
    Run3ReplayBindingV1,
    Run3TerminalBatchV1,
    Run3TerminalReplayV1,
    Run3TerminalTransitionV1,
    Run3TransitionRejected,
)
from .empirical_contextual_run3_terminal_trainer import (
    Run3TerminalHybridSacTrainerV1,
    Run3TerminalTrainerConfigV1,
    Run3TrainerError,
)
from .empirical_contextual_terminal_replay import EmpiricalTerminalTransitionV1
from .hybrid_sac_models import HybridSacModelConfig, build_actor, build_twin_critics
from .modeled_smoke_support import MODELED_SMOKE_SUPPORT
from .run_empirical_contextual_run3 import (
    PROJECTED_BYTES_PER_SEED,
    Run3ArtifactError,
    _bootstrap_checkpoint_pointer,
    _disk_gate,
    _discard_non_authoritative_tail,
    _json_bytes,
    _remaining_campaign_projection,
    _resume_checkpoint,
    _sha256_file,
    _torch_bytes,
    _verified_completed_seed_report,
    run_campaign,
)
from .transaction_identity import canonical_sha256


TINY_CONFIG = Run3RunnerConfigV1(
    seeds=(17,),
    warmup_transitions=4,
    batch_size=4,
    collect_per_update=2,
    update_count=2,
    matched_horizon_update=1,
    replay_capacity=16,
    model_snapshot_interval=1,
)


def _direct_batch(row, *, binding, partition) -> Run3TerminalBatchV1:
    return Run3TerminalBatchV1(
        binding=binding,
        partition=partition,
        rows=(row,),
        _state=torch.tensor([row.observation.values], dtype=torch.float32),
        _mode_id=torch.tensor([row.action.mode_id], dtype=torch.int64),
        _q_e4=torch.tensor([row.action.q_e4], dtype=torch.int64),
        _reward=torch.tensor([row.reward], dtype=torch.float32),
    )


def _make_environment_transition(environment, *, seq: int, master_seed: int):
    observation = environment.reset()
    lower, _upper = MODELED_SMOKE_SUPPORT.mode_q_e4_bounds[0]
    action = require_supported_action(0, lower)
    requested_q = lower / 10_000.0
    source = environment.step(action)
    query = environment._surface.query_fit_q_e4(
        source.audit.sample_id, action.mode_id, action.q_e4
    )
    components = tuple(
        (item.name, item.value, item.valid, item.status)
        for item in query.policy.quality
    )
    values = {name: value for name, value, _valid, _status in components}
    policy = source.policy
    proxy = QuantileLatencyProxyV1(
        p50_ms=float(policy.latency_proxy_ms),
        p95_ms=float(policy.latency_proxy_p95_ms),
        p99_ms=float(policy.latency_proxy_p99_ms),
    )
    session = str(uuid.UUID("cd52010d-2d3b-4bc9-8270-77767160e091"))
    key = f"run3:{session}:{seq}"
    realized = sample_run3_simulator_outcome(
        q_perc=float(values["q_perc"]),
        p_complete_reassembly_given_sent=float(
            policy.p_complete_reassembly_given_sent
        ),
        p_edge_admission_given_reassembled=float(
            policy.p_edge_admission_given_reassembled
        ),
        latency_proxy=proxy,
        random_draws=Run3CounterRngV1(master_seed).draws(key),
    )
    authenticated = EmpiricalTerminalTransitionV1.from_d1(
        collection_session_uuid=session,
        collection_seq=seq,
        observation=observation,
        action=action,
        result=source,
        d1_binding=environment.binding,
    )
    return Run3TerminalTransitionV1.issue(
        collection_session_uuid=session,
        collection_seq=seq,
        decision_key=key,
        observation=observation,
        action=action,
        requested_q=requested_q,
        source_result=source,
        source_d1_transition=authenticated,
        quality_query=query,
        quality_components=components,
        q_loc=float(values["q_loc"]),
        q_seg=float(values["q_seg"]),
        q_perc=float(values["q_perc"]),
        realized=realized,
        d1_environment_binding_sha256=environment.binding.canonical_sha256(),
    )


def _completed_seed_artifacts(directory: Path, seed: int) -> dict:
    directory.mkdir(parents=True)
    summary = {
        "seed": seed,
        "configured_updates": RUN3_REGISTERED_CONFIG.update_count,
        "completed_updates": RUN3_REGISTERED_CONFIG.update_count,
        "transition_count": RUN3_REGISTERED_CONFIG.total_transitions,
        "replay_resident_count": RUN3_REGISTERED_CONFIG.total_transitions,
        "replay_eviction_count": 0,
        "excluded_fault_count": 0,
        "completed_training_hard_gates_passed": True,
    }
    report = {
        "artifact_hashes": {},
        "fixed_endpoint_update": RUN3_REGISTERED_CONFIG.update_count,
        "matched_horizon_update": RUN3_REGISTERED_CONFIG.matched_horizon_update,
        "peak_checkpoint_selection": False,
        "record": "run3_training_report_v1",
        "summary": summary,
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = directory / "report.json"
    report_path.write_bytes(_json_bytes(report))
    terminal = {
        "report_file_sha256": _sha256_file(report_path),
        "report_sha256": report["report_sha256"],
        "schema": "splitfusion.run3_training_terminal.v1",
        "seed": seed,
        "status": "RUN3_FIXED_ENDPOINT_COMPLETE",
        "update": RUN3_REGISTERED_CONFIG.update_count,
    }
    terminal["terminal_sha256"] = canonical_sha256(terminal)
    (directory / "RUN3_TRAINING_COMPLETE.json").write_bytes(
        _json_bytes(terminal)
    )
    return report


class _CheckpointOnlyRunner:
    def __init__(self, checkpoint) -> None:
        self.completed_updates = 0
        self._checkpoint = checkpoint

    def checkpoint(self):
        return self._checkpoint


class Run3TrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        runner = Run3TrainingRunnerV1(seed=17, config=TINY_CONFIG)
        try:
            cls.partition = runner.environment._fit_partition
            cls.checkpoint_0 = runner.checkpoint()
            runner.run_until_updates(1)
            cls.checkpoint_1 = runner.checkpoint()
            runner.run_until_updates(2)
            cls.checkpoint_2 = runner.checkpoint()
            cls.rows = runner.transitions
            cls.binding = runner.replay.binding
            cls.replay = runner.replay
            generator = torch.Generator(device="cpu")
            generator.manual_seed(991)
            cls.batch = runner.replay.sample(4, generator=generator)
            cls.summary = runner.summary()
        finally:
            runner.close()

        resumed = Run3TrainingRunnerV1(seed=17, config=TINY_CONFIG)
        try:
            resumed.load_checkpoint(cls.checkpoint_1)
            resumed.run_until_updates(2)
            cls.resumed_checkpoint_2 = resumed.checkpoint()
        finally:
            resumed.close()

        # Reference execution repeats the former deep row/binding audit after
        # each sample.  It must produce the exact same learning/checkpoint
        # state as the optimized authenticated compact path.
        reference = Run3TrainingRunnerV1(seed=17, config=TINY_CONFIG)
        try:
            optimized_sample = reference.replay.sample

            def deep_reference_sample(batch_size, *, generator):
                batch = optimized_sample(batch_size, generator=generator)
                for row in batch.rows:
                    row.revalidate()
                    self_binding = Run3ReplayBindingV1.from_transition(
                        row, reference.environment._fit_partition
                    )
                    if self_binding != batch.binding:
                        raise AssertionError("reference replay binding drift")
                return batch

            reference.replay.sample = deep_reference_sample
            reference.run_until_updates(2)
            cls.deep_reference_checkpoint_2 = reference.checkpoint()
        finally:
            reference.close()

        validation = PartitionedEmpiricalOneStepEnvironmentV1.load_registered(
            seed=773, split=FIT_VALIDATION_SPLIT
        )
        try:
            cls.validation_row = _make_environment_transition(
                validation, seq=0, master_seed=881
            )
        finally:
            validation.close()

    def _fresh_trainer(self):
        config = HybridSacModelConfig(
            dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT
        )
        actor = build_actor(config, seed=101)
        critics = build_twin_critics(config, seed=202)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(303)
        trainer = Run3TerminalHybridSacTrainerV1(
            actor,
            critics,
            Run3TerminalTrainerConfigV1(batch_size=4),
            expected_binding=self.binding,
            expected_batch_issuer_capability=(
                self.replay.trainer_issuer_capability
            ),
            actor_generator=generator,
        )
        return trainer

    def test_registered_schedule_and_binding_are_exact(self) -> None:
        self.assertEqual(RUN3_REGISTERED_CONFIG.seeds, (17, 29, 43))
        self.assertEqual(RUN3_REGISTERED_CONFIG.total_transitions, 41_024)
        self.assertEqual(RUN3_REGISTERED_CONFIG.replay_capacity, 65_536)
        self.assertEqual(
            RUN3_REGISTERED_CONFIG.full_checkpoint_updates, (0, 5_000, 10_000)
        )
        self.assertEqual(len(range(500, 10_001, 500)), 20)
        self.assertEqual(
            self.checkpoint_2.runner_binding_document["preflight_result_sha256"],
            RUN3_PREFLIGHT_RESULT_SHA256,
        )

    def test_registered_seed17_decision_lineage_matches_prefixed_run(self) -> None:
        document = copy.deepcopy(self.checkpoint_0.runner_binding_document)
        document["config"] = RUN3_REGISTERED_CONFIG.to_canonical_dict()
        document["config_sha256"] = RUN3_REGISTERED_CONFIG.canonical_sha256()
        document["trainer_config_sha256"] = Run3TerminalTrainerConfigV1(
            batch_size=RUN3_REGISTERED_CONFIG.batch_size
        ).canonical_sha256()
        lineage = _decision_lineage_binding_sha256(document)
        self.assertEqual(
            lineage, _REGISTERED_DECISION_LINEAGE_SEED17_BINDING_SHA256
        )
        session = str(uuid.uuid5(
            uuid.UUID("5b8c1332-6d6c-59a4-8ec1-df34f8c2f6db"),
            f"{lineage}:17",
        ))
        self.assertEqual(session, _REGISTERED_DECISION_LINEAGE_SEED17_SESSION_UUID)

    def test_resume_from_matched_boundary_is_byte_state_equivalent(self) -> None:
        self.assertEqual(
            self.resumed_checkpoint_2.checkpoint_sha256,
            self.checkpoint_2.checkpoint_sha256,
        )
        self.assertEqual(
            self.resumed_checkpoint_2.metrics, self.checkpoint_2.metrics
        )

    def test_optimized_and_deep_reference_checkpoints_are_identical(self) -> None:
        self.assertEqual(
            self.deep_reference_checkpoint_2.checkpoint_sha256,
            self.checkpoint_2.checkpoint_sha256,
        )

    def test_decision_identity_is_unique_and_action_independent(self) -> None:
        self.assertEqual(len({row.decision_key for row in self.rows}), len(self.rows))
        for index, row in enumerate(self.rows):
            self.assertEqual(
                row.decision_key,
                f"run3:{row.collection_session_uuid}:{index}",
            )
            self.assertEqual(
                row.decision_key.split(":"),
                ["run3", row.collection_session_uuid, str(index)],
            )

    def test_collection_canonically_quantizes_raw_q(self) -> None:
        for row in self.rows:
            self.assertEqual(round_half_up_q_e4(row.requested_q), row.action.q_e4)

    def test_terminal_learner_tensors_exclude_probabilities(self) -> None:
        tensors = self.batch.learner_tensors()
        self.assertEqual(
            set(tensors), {"state", "mode_id", "q_e4", "q_normalized", "reward"}
        )
        self.assertIsNone(self.batch.next_state)
        self.assertTrue(bool(self.batch.terminated.all()))
        self.assertFalse(bool(self.batch.bootstrap.any()))
        self.assertTrue(bool((self.batch.duration == 1).all()))
        self.assertTrue(bool((self.batch.discount == 0).all()))
        self.assertEqual(self.batch.mode_id.dtype, torch.int64)
        self.assertEqual(self.batch.q_e4.dtype, torch.int64)
        self.assertEqual(self.batch.reward.dtype, torch.float32)

    def test_realized_reward_not_d1_expected_utility_is_target(self) -> None:
        self.assertTrue(
            any(row.source_result.policy.reward != row.reward for row in self.rows)
        )
        expected = torch.tensor([row.reward for row in self.batch.rows], dtype=torch.float32)
        self.assertTrue(torch.equal(self.batch.terminal_target(), expected))
        source = Path(
            sys.modules[
                "rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_run3_terminal_trainer"
            ].__file__
        ).read_text(encoding="utf-8")
        self.assertNotIn("expected_run3_reward", source)
        self.assertNotIn("soft_state_value", source)
        self.assertNotIn("critic_target", source)

    def test_terminal_reward_and_failure_semantics_are_exact(self) -> None:
        success_rows = [
            row for row in self.rows
            if row.terminal_outcome is Run3TerminalOutcome.SUCCESS_WITHIN_DEADLINE
        ]
        self.assertTrue(success_rows)
        for row in success_rows:
            expected = row.q_perc - 0.25 * float(row.latency_ms) / 200.0
            self.assertEqual(row.reward64, expected)
            self.assertEqual(
                row.emitted_reward_float32_bits_hex,
                struct.pack(">f", row.reward).hex(),
            )
        failure = evaluate_run3_reward(Run3TerminalOutcome.REASSEMBLY_FAILURE)
        self.assertTrue(failure.learning_eligible)
        self.assertEqual(failure.scalar_reward, -1.0)
        excluded = evaluate_run3_reward(
            Run3TerminalOutcome.INFRASTRUCTURE_FAULT_EXCLUDED
        )
        self.assertFalse(excluded.learning_eligible)
        self.assertIsNone(excluded.scalar_reward)

    def test_batch_rejects_dtype_replacement_that_torch_equal_would_accept(self) -> None:
        bad = replace(self.batch, _state=self.batch.state.to(torch.float64))
        self.assertTrue(torch.equal(bad._state, self.batch.state))
        with self.assertRaises(Run3ReplayError):
            bad.revalidate(
                expected_issuer_capability=self.replay.trainer_issuer_capability
            )
        bad_mode = replace(self.batch, _mode_id=self.batch.mode_id.to(torch.float32))
        self.assertTrue(torch.equal(bad_mode._mode_id, self.batch.mode_id))
        with self.assertRaises(Run3ReplayError):
            bad_mode.revalidate(
                expected_issuer_capability=self.replay.trainer_issuer_capability
            )

    def test_sampling_and_update_hot_path_never_repeat_deep_authentication(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(817)
        with mock.patch.object(
            Run3TerminalTransitionV1,
            "revalidate",
            side_effect=AssertionError("deep row validation entered hot path"),
        ), mock.patch.object(
            Run3ReplayBindingV1,
            "from_transition",
            side_effect=AssertionError("binding reconstruction entered hot path"),
        ), mock.patch.object(
            type(self.partition),
            "canonical_sha256",
            side_effect=AssertionError("partition hashing entered hot path"),
        ):
            batch = self.replay.sample(4, generator=generator)
            metric = self._fresh_trainer().update_once(batch)
        self.assertEqual(metric.batch_size, 4)

    def test_resident_rich_row_tamper_is_detected_at_sampling(self) -> None:
        replay = Run3TerminalReplayV1(16, partition=self.partition)
        row = self.rows[0]
        replay.insert(row)
        original = row.q_loc
        object.__setattr__(row, "q_loc", original + 0.001)
        try:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(19)
            with self.assertRaises(Run3ReplayError):
                replay.sample(1, generator=generator)
        finally:
            object.__setattr__(row, "q_loc", original)
        row.revalidate()

    def test_resident_compact_tamper_is_rejected_without_rng_poisoning(self) -> None:
        replay = Run3TerminalReplayV1(16, partition=self.partition)
        replay.insert(self.rows[0])
        record = replay._records[0]
        original = record.reward
        object.__setattr__(record, "reward", original + 0.001)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(119)
        rng_before = generator.get_state().clone()
        indexes_before = (
            len(replay), replay.accepted_count, replay.evicted_count,
            set(replay._lifetime_digests), dict(replay._identity_digests), replay.binding,
        )
        try:
            with self.assertRaises(Run3ReplayError):
                replay.sample(1, generator=generator)
        finally:
            object.__setattr__(record, "reward", original)
        self.assertTrue(torch.equal(generator.get_state(), rng_before))
        self.assertEqual(
            indexes_before,
            (
                len(replay), replay.accepted_count, replay.evicted_count,
                set(replay._lifetime_digests), dict(replay._identity_digests), replay.binding,
            ),
        )

    def test_resident_compact_signed_zero_tamper_is_rejected(self) -> None:
        replay = Run3TerminalReplayV1(16, partition=self.partition)
        replay.insert(self.rows[0])
        record = replay._records[0]
        values = list(record.observation_values)
        zero_index = next(
            (index for index, value in enumerate(values) if value == 0.0), None
        )
        self.assertIsNotNone(zero_index)
        original = values[zero_index]
        values[zero_index] = -0.0 if struct.pack(">d", original) == struct.pack(">d", 0.0) else 0.0
        object.__setattr__(record, "observation_values", tuple(values))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(120)
        rng_before = generator.get_state().clone()
        try:
            with self.assertRaises(Run3ReplayError):
                replay.sample(1, generator=generator)
        finally:
            values[zero_index] = original
            object.__setattr__(record, "observation_values", tuple(values))
        self.assertTrue(torch.equal(generator.get_state(), rng_before))

    def test_post_sample_rich_row_and_compact_record_tamper_are_rejected(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(23)
        batch = self.replay.sample(4, generator=generator)
        row = batch.rows[0]
        original_row = row.q_seg
        object.__setattr__(row, "q_seg", original_row + 0.001)
        try:
            with self.assertRaises(Run3ReplayError):
                batch.revalidate(
                    expected_issuer_capability=(
                        self.replay.trainer_issuer_capability
                    )
                )
        finally:
            object.__setattr__(row, "q_seg", original_row)
        record = batch.compact_records[0]
        original_record = record.q_seg
        object.__setattr__(record, "q_seg", original_record + 0.001)
        try:
            with self.assertRaises(Run3ReplayError):
                batch.revalidate(
                    expected_issuer_capability=(
                        self.replay.trainer_issuer_capability
                    )
                )
        finally:
            object.__setattr__(record, "q_seg", original_record)

    def test_cross_replay_batch_is_rejected_before_optimizer_mutation(self) -> None:
        foreign = Run3TerminalReplayV1(16, partition=self.partition)
        for row in self.rows[:4]:
            foreign.insert(row)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(29)
        foreign_batch = foreign.sample(4, generator=generator)
        trainer = self._fresh_trainer()
        actor_before = copy.deepcopy(trainer.actor.state_dict())
        with self.assertRaises(Run3TransitionRejected):
            trainer.update_once(foreign_batch)
        self.assertEqual(trainer.update_count, 0)
        self.assertFalse(trainer.actor_optimizer.state)
        self.assertFalse(trainer.critic_optimizer.state)
        for name, value in trainer.actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor_before[name]))

    def test_sampled_indices_and_learner_tensor_bytes_match_reference(self) -> None:
        observed_generator = torch.Generator(device="cpu")
        reference_generator = torch.Generator(device="cpu")
        observed_generator.manual_seed(31337)
        reference_generator.manual_seed(31337)
        expected_indices = torch.randperm(
            len(self.replay), generator=reference_generator
        )[:4].tolist()
        batch = self.replay.sample(4, generator=observed_generator)
        expected_rows = tuple(self.replay.rows[index] for index in expected_indices)
        self.assertEqual(batch.rows, expected_rows)
        self.assertTrue(torch.equal(
            batch.state,
            torch.tensor([row.observation.values for row in expected_rows], dtype=torch.float32),
        ))
        self.assertTrue(torch.equal(
            batch.mode_id,
            torch.tensor([row.action.mode_id for row in expected_rows], dtype=torch.int64),
        ))
        self.assertTrue(torch.equal(
            batch.q_e4,
            torch.tensor([row.action.q_e4 for row in expected_rows], dtype=torch.int64),
        ))
        self.assertTrue(torch.equal(
            batch.reward,
            torch.tensor([row.reward for row in expected_rows], dtype=torch.float32),
        ))

    def test_duplicate_and_identity_conflict_survive_replay_gates(self) -> None:
        replay = Run3TerminalReplayV1(1, partition=self.partition)
        first = self.rows[0]
        replay.insert(first)
        with self.assertRaises(Run3DuplicateTransition):
            replay.insert(first)
        shifted_q = min(0.98, first.requested_q + 1e-8)
        if round_half_up_q_e4(shifted_q) != first.action.q_e4:
            shifted_q = max(0.0, first.requested_q - 1e-8)
        conflict = Run3TerminalTransitionV1.issue(
            collection_session_uuid=first.collection_session_uuid,
            collection_seq=first.collection_seq,
            decision_key=first.decision_key,
            observation=first.observation,
            action=first.action,
            requested_q=shifted_q,
            source_result=first.source_result,
            source_d1_transition=first.source_d1_transition,
            quality_query=first.quality_query,
            quality_components=first.quality_components,
            q_loc=first.q_loc,
            q_seg=first.q_seg,
            q_perc=first.q_perc,
            realized=first.realized,
            d1_environment_binding_sha256=first.d1_environment_binding_sha256,
        )
        self.assertNotEqual(conflict.canonical_sha256(), first.canonical_sha256())
        with self.assertRaises(Run3IdentityConflict):
            replay.insert(conflict)

    def test_zero_eviction_gate_holds_for_tiny_and_registered_schedules(self) -> None:
        self.assertEqual(self.summary.replay_eviction_count, 0)
        self.assertEqual(self.summary.transition_count, TINY_CONFIG.total_transitions)
        self.assertLessEqual(
            RUN3_REGISTERED_CONFIG.total_transitions,
            RUN3_REGISTERED_CONFIG.replay_capacity,
        )

    def test_trainer_uses_y_equal_reward_and_reports_diagnostics(self) -> None:
        trainer = self._fresh_trainer()
        metric = trainer.update_once(self.batch)
        self.assertEqual(metric.target_reward_bit_mismatch_count, 0)
        self.assertAlmostEqual(metric.reward_mean, float(self.batch.reward.mean()))
        self.assertEqual(sum(metric.terminal_outcome_counts), 4)
        self.assertEqual(sum(metric.executed_mode_counts), 4)
        self.assertTrue(0.0 <= metric.q_loc_mean <= 1.0)
        self.assertTrue(0.0 <= metric.q_seg_mean <= 1.0)
        self.assertTrue(0.0 <= metric.q_perc_mean <= 1.0)

    def test_trainer_rechecks_optimizer_before_any_mutation(self) -> None:
        trainer = self._fresh_trainer()
        before = copy.deepcopy(trainer.actor.state_dict())
        target_parameter = next(trainer.critics.target_1.parameters())
        trainer.actor_optimizer.param_groups[0]["params"].append(target_parameter)
        with self.assertRaises(Run3TrainerError):
            trainer.update_once(self.batch)
        self.assertEqual(trainer.update_count, 0)
        for name, value in trainer.actor.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertFalse(trainer.actor_optimizer.state)

    def test_trainer_rechecks_target_trainability_before_mutation(self) -> None:
        trainer = self._fresh_trainer()
        before = copy.deepcopy(trainer.critics.state_dict())
        next(trainer.critics.target_1.parameters()).requires_grad_(True)
        with self.assertRaises(Run3TrainerError):
            trainer.update_once(self.batch)
        self.assertEqual(trainer.update_count, 0)
        for name, value in trainer.critics.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_direct_batch_rejects_fit_validation_scene(self) -> None:
        replay = Run3TerminalReplayV1(16, partition=self.partition)
        before = (
            len(replay), replay.accepted_count, replay.evicted_count,
            set(replay._lifetime_digests), dict(replay._identity_digests), replay.binding,
        )
        with self.assertRaises(Run3TransitionRejected):
            replay.insert(self.validation_row)
        self.assertEqual(
            before,
            (
                len(replay), replay.accepted_count, replay.evicted_count,
                set(replay._lifetime_digests), dict(replay._identity_digests), replay.binding,
            )
        )

    def test_direct_batch_rejects_nontraining_radio_row(self) -> None:
        base = self.rows[0]
        foreign = self.validation_row.source_result.audit
        changed_audit = replace(
            base.source_result.audit,
            hidden_network_profile=foreign.hidden_network_profile,
            hidden_radio_csv_row_number=foreign.hidden_radio_csv_row_number,
            hidden_trace_id=foreign.hidden_trace_id,
            hidden_trace_step_index=foreign.hidden_trace_step_index,
            hidden_target_snr_db=foreign.hidden_target_snr_db,
            hidden_radio_row_sha256=foreign.hidden_radio_row_sha256,
        )
        changed_source = replace(base.source_result, audit=changed_audit)
        authenticated = EmpiricalTerminalTransitionV1.from_d1(
            collection_session_uuid=base.collection_session_uuid,
            collection_seq=base.collection_seq,
            observation=base.observation,
            action=base.action,
            result=changed_source,
            d1_binding=base.source_d1_transition.d1_binding,
        )
        changed = Run3TerminalTransitionV1.issue(
            collection_session_uuid=base.collection_session_uuid,
            collection_seq=base.collection_seq,
            decision_key=base.decision_key,
            observation=base.observation,
            action=base.action,
            requested_q=base.requested_q,
            source_result=changed_source,
            source_d1_transition=authenticated,
            quality_query=base.quality_query,
            quality_components=base.quality_components,
            q_loc=base.q_loc,
            q_seg=base.q_seg,
            q_perc=base.q_perc,
            realized=base.realized,
            d1_environment_binding_sha256=base.d1_environment_binding_sha256,
        )
        replay = Run3TerminalReplayV1(16, partition=self.partition)
        before = (
            len(replay), replay.accepted_count, replay.evicted_count,
            set(replay._lifetime_digests), dict(replay._identity_digests), replay.binding,
        )
        with self.assertRaises(Run3TransitionRejected):
            replay.insert(changed)
        self.assertEqual(
            before,
            (
                len(replay), replay.accepted_count, replay.evicted_count,
                set(replay._lifetime_digests), dict(replay._identity_digests), replay.binding,
            )
        )

    def test_direct_forged_batch_is_rejected_independently_of_split(self) -> None:
        forged = _direct_batch(
            self.rows[0], binding=self.binding, partition=self.partition
        )
        with self.assertRaises(Run3TransitionRejected):
            forged.revalidate(
                expected_issuer_capability=self.replay.trainer_issuer_capability
            )

    def test_batch_partition_replacement_is_rejected_without_partition_hashing(self) -> None:
        changed = replace(self.batch, partition=replace(self.partition))
        with self.assertRaises(Run3ReplayError):
            changed.revalidate(
                expected_issuer_capability=self.replay.trainer_issuer_capability
            )

    def test_transition_tamper_fails_before_replay_mutation(self) -> None:
        replay = Run3TerminalReplayV1(16, partition=self.partition)
        row = self.rows[0]
        original = row.reward64
        object.__setattr__(row, "reward64", original + 0.1)
        try:
            with self.assertRaises(Run3TransitionRejected):
                replay.insert(row)
            self.assertEqual(len(replay), 0)
            self.assertEqual(replay.accepted_count, 0)
        finally:
            object.__setattr__(row, "reward64", original)
        row.revalidate()

    def test_snapshot_and_full_checkpoint_cadence_are_disjoint_roles(self) -> None:
        self.assertEqual(TINY_CONFIG.full_checkpoint_updates, (0, 1, 2))
        self.assertEqual(TINY_CONFIG.model_snapshot_interval, 1)
        snapshot = {
            "actor_state",
            "critic_1_state",
            "critic_2_state",
            "binding_sha256",
            "schema",
            "seed",
            "update",
            "snapshot_sha256",
        }
        self.assertEqual(set(self.checkpoint_2.config.to_canonical_dict()) >= {"full_checkpoint_updates"}, True)
        runner = Run3TrainingRunnerV1(seed=17, config=TINY_CONFIG)
        try:
            runner.load_checkpoint(self.checkpoint_2)
            self.assertEqual(set(runner.model_only_snapshot()), snapshot)
        finally:
            runner.close()

    def test_bootstrap_recovery_reuses_valid_full_zero_without_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "checkpoints" / "full_000000.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(_torch_bytes(self.checkpoint_0))
            original_hash = _sha256_file(path)
            _bootstrap_checkpoint_pointer(
                directory, _CheckpointOnlyRunner(self.checkpoint_0)
            )
            self.assertEqual(_sha256_file(path), original_hash)
            self.assertEqual(
                _resume_checkpoint(directory).checkpoint_sha256,
                self.checkpoint_0.checkpoint_sha256,
            )
            self.assertEqual(len(list(directory.glob("latest_checkpoint*.pt"))), 0)

    def test_bootstrap_recovery_quarantines_conflicting_full_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "checkpoints" / "full_000000.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"conflicting checkpoint")
            _bootstrap_checkpoint_pointer(
                directory, _CheckpointOnlyRunner(self.checkpoint_0)
            )
            self.assertEqual(
                _resume_checkpoint(directory).checkpoint_sha256,
                self.checkpoint_0.checkpoint_sha256,
            )
            quarantined = list(
                (directory / "orphaned_non_authoritative").glob(
                    "checkpoints__full_000000.pt__*"
                )
            )
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_bytes(), b"conflicting checkpoint")

    def test_checkpoint_tail_is_quarantined_without_snapshot_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "checkpoints" / "full_000001.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"orphan")
            _discard_non_authoritative_tail(directory, 0)
            self.assertFalse(path.exists())
            self.assertEqual(
                len(list((directory / "orphaned_non_authoritative").iterdir())), 1
            )

    def test_completed_seed_verification_catches_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "seed_17"
            _completed_seed_artifacts(directory, 17)
            report = _verified_completed_seed_report(
                directory, seed=17, config=RUN3_REGISTERED_CONFIG
            )
            self.assertIsNotNone(report)
            (directory / "unexpected.bin").write_bytes(b"drift")
            with self.assertRaises(Run3ArtifactError):
                _verified_completed_seed_report(
                    directory, seed=17, config=RUN3_REGISTERED_CONFIG
                )

    def test_late_resume_disk_projection_excludes_only_complete_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _completed_seed_artifacts(root / "seed_17", 17)
            partial = root / "seed_29"
            partial.mkdir()
            (partial / "partial.bin").write_bytes(b"x" * 123)
            projected = _remaining_campaign_projection(root, (17, 29, 43))
            self.assertEqual(projected, 2 * PROJECTED_BYTES_PER_SEED)

    def test_disk_gate_accepts_nested_not_yet_created_campaign_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "new" / "nested" / "campaign"
            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.run_empirical_contextual_run3."
                "shutil.disk_usage",
                return_value=shutil._ntuple_diskusage(
                    total=100 * 1024**3,
                    used=10 * 1024**3,
                    free=90 * 1024**3,
                ),
            ) as disk_usage:
                result = _disk_gate(target, seed_count=3)
            disk_usage.assert_called_once_with(Path(temporary))
            self.assertEqual(result["projected_bytes"], 3 * PROJECTED_BYTES_PER_SEED)

    def test_official_campaign_refuses_tiny_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(Run3ArtifactError):
                run_campaign(Path(temporary) / "campaign", config=TINY_CONFIG)

    def test_campaign_resume_skips_verified_completed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "campaign"
            root.mkdir()
            manifest = {
                "config": RUN3_REGISTERED_CONFIG.to_canonical_dict(),
                "config_sha256": RUN3_REGISTERED_CONFIG.canonical_sha256(),
                "disk_preflight": {
                    "free_bytes_at_preflight": 1,
                    "projected_bytes": 1,
                    "required_reserve_bytes": 1,
                },
                "record": "run3_three_seed_campaign_manifest_v1",
                "seeds": [17, 29, 43],
            }
            manifest["campaign_manifest_sha256"] = canonical_sha256(manifest)
            (root / "campaign_manifest.json").write_bytes(_json_bytes(manifest))
            report_17 = _completed_seed_artifacts(root / "seed_17", 17)
            calls = []

            def fake_seed(directory, *, seed, config, project_root, resume):
                calls.append((seed, resume))
                return {"report_sha256": f"{seed:064x}"[-64:]}

            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.run_empirical_contextual_run3._disk_gate",
                return_value={},
            ), mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.run_empirical_contextual_run3.run_seed_to_directory",
                side_effect=fake_seed,
            ):
                result = run_campaign(root, resume=True)
            self.assertEqual(calls, [(29, False), (43, False)])
            self.assertEqual(
                result["seed_reports"]["17"], report_17["report_sha256"]
            )

    def test_import_has_no_validation_evaluator_evidence_io_or_cuda(self) -> None:
        package = "rl_agent.splitfusion_hybrid_sac_v1"
        code = f"""
import sys
opened = []
def audit(event, args):
    if event == 'open' and args and isinstance(args[0], str):
        opened.append(args[0])
sys.addaudithook(audit)
import torch
before = torch.cuda.is_initialized()
import {package}.empirical_contextual_run3_terminal_replay
import {package}.empirical_contextual_run3_terminal_trainer
import {package}.empirical_contextual_run3_runner
import {package}.run_empirical_contextual_run3
forbidden = [name for name in sys.modules if 'fit_validation_evaluator' in name or 'fit_validation_panel' in name]
evidence = [path for path in opened if '/experiments/' in path]
if before or torch.cuda.is_initialized() or forbidden or evidence:
    raise SystemExit(repr((before, torch.cuda.is_initialized(), forbidden, evidence)))
"""
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
