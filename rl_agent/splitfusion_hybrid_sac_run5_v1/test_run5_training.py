"""CPU-only tests: Run-5 collector causality, SNR invariance, training and resume."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as T
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_collector as RC
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as V1
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT

EVIDENCE_ROOT = Path(os.environ.get(
    "RUN5_EVIDENCE_ROOT", Path(__file__).resolve().parents[3] / "abiodun"))
WORKTREE = Path(__file__).resolve().parents[2]
ARTIFACT = WORKTREE / RC.CV.C2.PACKAGE_RELPATH.replace(
    "rl_agent/ue_production_transport_model_v2", "rl_agent/experiments/"
    "ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json")


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.shared = RC.build_shared_sources(ARTIFACT, EVIDENCE_ROOT)

    def collector(self) -> RC.Run5ModeledCollectorV1:
        return RC.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=17,
                                         evidence_root=EVIDENCE_ROOT,
                                         shared_sources=self.shared)

    def warmup(self, collector, count, start=0):
        schedule = orch.build_frozen_warmup_schedule(17)
        records = []
        for ordinal in range(start, start + count):
            selected = schedule.action_at(ordinal)
            records.append(collector.collect(orch.ModeledActionRequestV1(
                decision_ordinal=ordinal, mode_id=selected.mode_id, q_e4=selected.q_e4,
                source="STRATIFIED_WARMUP", warmup_q_bin_index=selected.q_bin_index)))
        return records


class CollectorTest(_Base):
    def test_22d_state_with_run4_prefix_and_registered_snr_scaling(self) -> None:
        collector = self.collector()
        records = self.warmup(collector, 30)
        for record, diag in zip(records, collector.diagnostics()):
            self.assertEqual(len(record.state), 22)
            self.assertEqual(record.state[21], (diag["snr_db"] - 5.5) / 19.0)
            self.assertEqual(record.state[2], diag["prior_ul_mcs"] / 28.0)
        for previous, record in zip(records, records[1:]):
            self.assertEqual(previous.next_state, record.state)

    def test_prefix_drift_is_refused(self) -> None:
        collector = self.collector()
        original = SNR.build_run5_features_v2

        def drifted(guarded, scaling):
            vector = original(guarded, scaling)
            values = list(vector.as_tuple())
            values[3] += 1e-9
            forged = SNR.Run5FeatureVectorV2(tuple(values), vector.run4_prefix_sha256,
                                             vector.guarded_state_sha256)
            from dataclasses import replace
            return replace(forged, _attestation=SNR._issue_features(forged._binding()))

        collector._feature_cache.clear()
        with mock.patch.object(SNR, "build_run5_features_v2", drifted):
            with self.assertRaises(RC.Run5CollectorError):
                collector.current_state_features()

    def test_no_future_snr_leakage_and_causal_call_order(self) -> None:
        collector = self.collector()
        events = []
        kernel_execute = collector._kernel.execute_cycle
        channel_advance = collector._channel.advance
        features = collector.current_state_features

        def logged_kernel(request):
            events.append("kernel")
            return kernel_execute(request)

        def logged_advance(duration):
            events.append("advance")
            return channel_advance(duration)

        def logged_features():
            events.append("features")
            return features()

        collector._kernel.execute_cycle = logged_kernel
        collector._channel.advance = logged_advance
        collector.current_state_features = logged_features
        states_before = []
        for record in range(20):
            states_before.append(collector.current_state_features())
            self.warmup(collector, 1, start=record)
        # Per decision: current features are fixed before the kernel, and the
        # SNR future is generated only after the kernel resolved the outcome.
        order = [e for e in events if e in ("kernel", "advance")]
        self.assertEqual(order, ["kernel", "advance"] * 20)
        for before, record, diag in zip(states_before, collector.history(),
                                        collector.diagnostics()):
            self.assertEqual(before, record.state)
            self.assertTrue(diag["generated_ticks_after_observed"])
            self.assertNotEqual(record.state[21], (diag["successor_snr_db"] - 5.5) / 19.0)
        self.assertEqual(collector._channel.future_sample_violations, 0)

    def test_no_profile_or_trace_identity_in_policy_features(self) -> None:
        self.assertEqual(V1.RUN5_POLICY_FEATURE_ORDER[21], "effective_external_ul_snr_proxy_scaled")
        collector = self.collector()
        self.warmup(collector, 5)
        before = collector.current_state_features()
        collector._channel._profile = 1 - collector._channel._profile
        collector._channel._state = (collector._channel._state + 1) % 3
        collector._feature_cache.clear()
        self.assertEqual(collector.current_state_features(), before)
        record_fields = set(RC.Run5CollectedTransitionV1.__dataclass_fields__)
        for forbidden in ("profile", "trace", "markov", "hidden", "tick", "noise", "gnb"):
            self.assertFalse(any(forbidden in name for name in record_fields), forbidden)
        text = json.dumps([r.ledger_dict() for r in collector.history()])
        for forbidden in ("MID_VARIABLE", "FADE_RECOVERY", "hidden_state"):
            self.assertNotIn(forbidden, text)

    def test_immediate_outcome_is_bit_identical_under_changed_snr(self) -> None:
        for at in (0, 7):
            a, b = self.collector(), self.collector()
            if at:
                self.warmup(a, at)
                self.warmup(b, at)
            other = 23.0 if a.context.snr_db < 15 else 6.0
            b._channel._snr = other
            b._snr_current = other
            b.context.snr_db = other
            b._feature_cache.clear()
            ra, = self.warmup(a, 1, start=at)
            rb, = self.warmup(b, 1, start=at)
            da, db = a.diagnostics()[-1], b.diagnostics()[-1]
            self.assertNotEqual(ra.state[21], rb.state[21])
            self.assertEqual(ra.state[:21], rb.state[:21])
            for field in ("reward", "terminal", "q_perc", "latency_ms", "discount", "duration"):
                self.assertEqual(getattr(ra, field), getattr(rb, field), field)
            for field in ("on_time_probability", "on_time", "ingress_bytes",
                          "reward_frame_wire_bytes", "next_backlog_bytes"):
                self.assertEqual(da[field], db[field], field)

    def test_changed_snr_affects_only_the_successor_channel(self) -> None:
        differ = 0
        for seed_offset in range(12):
            a, b = self.collector(), self.collector()
            self.warmup(a, seed_offset)
            self.warmup(b, seed_offset)
            b._channel._snr = 24.4
            b._snr_current = 24.4
            b.context.snr_db = 24.4
            a._channel._snr = 5.6
            a._snr_current = 5.6
            a.context.snr_db = 5.6
            a._feature_cache.clear()
            b._feature_cache.clear()
            self.warmup(a, 1, start=seed_offset)
            self.warmup(b, 1, start=seed_offset)
            differ += int(a.diagnostics()[-1]["successor_mcs"]
                          != b.diagnostics()[-1]["successor_mcs"])
            self.assertLessEqual(a.diagnostics()[-1]["successor_mcs"],
                                 b.diagnostics()[-1]["successor_mcs"])
        self.assertGreater(differ, 0)

    def test_previous_action_and_outcome_propagate(self) -> None:
        collector = self.collector()
        records = self.warmup(collector, 60)
        report = RT.preflight_report(records, collector.diagnostics())
        self.assertEqual(report["previous_outcome_mismatches"], 0)
        self.assertEqual(report["future_tick_violations"], 0)
        self.assertFalse(records[0].state[19])
        self.assertTrue(all(r.state[19] == 1.0 for r in records[1:]))

    def test_collector_restore_rebuilds_joint_channel(self) -> None:
        a = self.collector()
        self.warmup(a, 25)
        b = self.collector()
        b.restore(a.checkpoint())
        self.assertEqual(b.channel_checkpoint(), a.channel_checkpoint())
        self.assertEqual(b.checkpoint(), a.checkpoint())


class TrainingTest(_Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp(prefix="run5_training_test_"))
        cls.orchestrator = RT.Run5OrchestratorV1(collector_factory=cls.factory, seed=17,
                                                 checkpoint_updates=(0, 100))
        cls.checkpoints = {}

        def callback(checkpoint):
            cls.checkpoints[checkpoint.update_count] = checkpoint
            RT.write_event_checkpoint(cls.tmp / f"u{checkpoint.update_count}.json", checkpoint)
            RT.write_sidecar(cls.tmp / f"u{checkpoint.update_count}.sidecar",
                             cls.orchestrator, checkpoint)

        cls.orchestrator.run_to(100, checkpoint_callback=callback, emit_current=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    @classmethod
    def factory(cls):
        return RC.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=17,
                                         evidence_root=EVIDENCE_ROOT,
                                         shared_sources=cls.shared)

    def test_update_math_and_hyperparameters_are_run4(self) -> None:
        self.assertIs(RT.Run5HybridSacTrainerV1.update_once, T._Run4TrainerCore.update_once)
        config = smoke_preregistration.FROZEN_CONFIG
        trainer = self.orchestrator.trainer
        self.assertEqual((trainer.config.alpha_d, trainer.config.alpha_c, trainer.config.tau,
                          trainer.config.actor_lr, trainer.config.critic_lr,
                          trainer.config.nominal_batch_size),
                         (config.alpha_d, config.alpha_c, config.polyak_tau,
                          config.actor_learning_rate, config.critic_learning_rate,
                          config.batch_size))
        self.assertEqual(self.orchestrator.replay.capacity, config.replay_capacity)
        self.assertEqual(self.orchestrator.decision_count, 288 + 4 * 100)
        self.assertEqual({r.discount for r in self.orchestrator.history}, {0.99 ** 2})
        run4_config = R4M.run4_model_config()
        self.assertEqual(self.orchestrator.actor.config.hidden_width, run4_config.hidden_width)
        self.assertEqual(self.orchestrator.actor.config.state_dim, 22)

    def test_21d_batch_is_refused_by_the_trainer(self) -> None:
        batch = self.orchestrator.replay.sample(8, torch.Generator().manual_seed(3))
        narrow = RT.Run5ReplayBatchV1(**{**{f: getattr(batch, f) for f in
                                            RT.Run5ReplayBatchV1.__dataclass_fields__},
                                         "_state": batch.state[:, :21]})
        with self.assertRaises(T.TrainerPreflightError):
            self.orchestrator.trainer.update_once(narrow)

    def test_event_replay_restore_and_continuation_are_bit_identical(self) -> None:
        restored = RT.Run5OrchestratorV1.restore_by_replay(
            RT.read_event_checkpoint(self.tmp / "u0.json"), collector_factory=self.factory,
            checkpoint_updates=(0, 100))
        seen = []
        restored.run_to(100, checkpoint_callback=seen.append)
        self.assertEqual(seen[-1].canonical_sha256, self.checkpoints[100].canonical_sha256)

    def test_sidecar_resume_restores_every_component(self) -> None:
        for update in (0, 100):
            resumed = RT.Run5OrchestratorV1.restore_from_sidecar(
                self.checkpoints[update], self.tmp / f"u{update}.sidecar",
                collector_factory=self.factory, checkpoint_updates=(0, 100))
            self.assertEqual(resumed.boundary(), dict(self.checkpoints[update].boundary))
        resumed = RT.Run5OrchestratorV1.restore_from_sidecar(
            self.checkpoints[0], self.tmp / "u0.sidecar", collector_factory=self.factory,
            checkpoint_updates=(0, 100))
        seen = []
        resumed.run_to(100, checkpoint_callback=seen.append)
        self.assertEqual(seen[-1].canonical_sha256, self.checkpoints[100].canonical_sha256)

    def test_cold_load_actor_from_sidecar(self) -> None:
        actor = RT.cold_load_actor(self.tmp / "u100.sidecar", self.checkpoints[100])
        for key, value in self.orchestrator.actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor.state_dict()[key]), key)

    def test_21d_or_foreign_checkpoints_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run4_like = Path(tmp) / "run4.json"
            run4_like.write_text(json.dumps({"schema": orch.CHECKPOINT_SCHEMA_ID}))
            with self.assertRaises(RT.Run5TrainingError):
                RT.read_event_checkpoint(run4_like)
            forged = Path(tmp) / "forged.sidecar"
            shutil.copytree(self.tmp / "u100.sidecar", forged)
            run4 = R4M.build_run4_models(actor_seed=1, critic_seed=2)
            (forged / "actor.pt").unlink()
            torch.save(run4.actor.state_dict(), forged / "actor.pt")
            with self.assertRaises(RT.Run5TrainingError):
                RT.read_sidecar(forged, self.checkpoints[100])
            manifest = json.loads((forged / "SIDECAR_MANIFEST.json").read_text())
            manifest["schema_id"] = "splitfusion.hybrid_sac.materialized_checkpoint_sidecar.v1"
            (forged / "SIDECAR_MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaises(RT.Run5TrainingError):
                RT.read_sidecar(forged, self.checkpoints[100])
        with self.assertRaises(RM.Run5CheckpointRefused):
            RM.load_run5_model_state(binding=RM.RUN5_V2_MODEL_BINDING,
                                     actor_state=run4.actor.state_dict(),
                                     critic_state=run4.critics.state_dict(),
                                     expected_binding_sha256=RM.RUN5_V2_MODEL_BINDING_SHA256)

    def test_cuda_never_initialized(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
