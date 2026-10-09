"""CPU-only tests: Run-5 collector causality, reward, previous outcomes, bundles, refusals."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration as R4PREREG
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as T
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_bundle as B
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_collector as RC
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_preregistration as PR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as V1
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
from rl_agent.splitfusion_hybrid_sac_run5_v1 import test_run5_snr_v2 as TS

EVIDENCE_ROOT = Path(os.environ.get(
    "RUN5_EVIDENCE_ROOT", Path(__file__).resolve().parents[3] / "abiodun"))
WORKTREE = Path(__file__).resolve().parents[2]
ARTIFACT = WORKTREE / ("rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/"
                       "transport_model_v2.json")
PREREG = "0" * 64


def force_next_snr(collector: RC.Run5ModeledCollectorV1, value: float) -> None:
    """Make the sample at the next decision's tick equal ``value`` (RNG stream kept)."""
    channel = collector._channel
    target_tick = channel._tick + 2
    original = channel._emit

    def emit():
        sample = original()
        return value if channel._tick == target_tick else sample

    channel._emit = emit


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.shared = RC.build_shared_sources(ARTIFACT, EVIDENCE_ROOT)

    @classmethod
    def factory(cls) -> RC.Run5ModeledCollectorV1:
        return RC.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=17,
                                         evidence_root=EVIDENCE_ROOT, shared_sources=cls.shared)

    def warmup(self, collector, count, start=0):
        schedule = RT.warmup_schedule(17)
        out = []
        for ordinal in range(start, start + count):
            selected = schedule.action_at(ordinal)
            out.append(collector.collect(orch.ModeledActionRequestV1(
                decision_ordinal=ordinal, mode_id=selected.mode_id, q_e4=selected.q_e4,
                source="STRATIFIED_WARMUP", warmup_q_bin_index=selected.q_bin_index)))
        return out


class CollectorTest(_Base):
    def test_22d_state_with_run4_prefix_and_registered_snr_scaling(self) -> None:
        collector = self.factory()
        records = self.warmup(collector, 30)
        for record, diag in zip(records, collector.diagnostics()):
            self.assertEqual(len(record.state), 22)
            self.assertEqual(record.state[21], (diag["snr_db"] - 5.5) / 19.0)
            self.assertEqual(record.state[2], diag["prior_ul_mcs"] / 28.0)
        for previous, record in zip(records, records[1:]):
            self.assertEqual(previous.next_state, record.state)

    def test_modeled_snr_uses_the_live_selection_rule(self) -> None:
        collector = self.factory()
        self.assertIsInstance(collector._snr_adapter, SNR.RfsimLeaseSnrAdapterV1)
        self.assertIs(type(collector._snr_adapter).observe, SNR.RfsimLeaseSnrAdapterV1.observe)
        with self.assertRaises(R4.MetadataError):
            collector._snr_adapter.record_heartbeat(active_command_id="x")  # no wall clock
        collector.current_state_features()
        ack = collector._snr_adapter._acks[-1]
        beat = collector._snr_adapter._heartbeats[-1]
        commit = collector._env.current_state.state.boundary.state_commit_timestamp_ns
        self.assertEqual((ack.ack_ns, beat[0]), (commit - 20_000_000, commit - 5_000_000))

    def test_prefix_drift_is_refused(self) -> None:
        collector = self.factory()
        original = SNR.build_run5_features_v2

        def drifted(guarded, scaling):
            vector = original(guarded, scaling)
            values = list(vector.as_tuple())
            values[3] += 1e-9
            forged = SNR.Run5FeatureVectorV2(tuple(values), vector.run4_prefix_sha256,
                                             vector.guarded_state_sha256)
            return replace(forged, _attestation=SNR._issue_features(forged._binding()))

        with mock.patch.object(SNR, "build_run5_features_v2", drifted):
            with self.assertRaises(RC.Run5CollectorError):
                collector.current_state_features()

    def test_no_future_snr_leakage_and_causal_call_order(self) -> None:
        collector = self.factory()
        events = []
        kernel_execute, channel_advance = collector._kernel.execute_cycle, collector._channel.advance
        collector._kernel.execute_cycle = lambda r: (events.append("kernel"), kernel_execute(r))[1]
        collector._channel.advance = lambda d: (events.append("advance"), channel_advance(d))[1]
        before = []
        for ordinal in range(20):
            before.append(collector.current_state_features())
            self.warmup(collector, 1, start=ordinal)
        self.assertEqual(events, ["kernel", "advance"] * 20)
        for state, record, diag in zip(before, collector.history(), collector.diagnostics()):
            self.assertEqual(state, record.state)
            self.assertTrue(diag["generated_ticks_after_observed"])
            self.assertNotEqual(record.state[21], (diag["successor_snr_db"] - 5.5) / 19.0)
        self.assertEqual(collector._channel.future_sample_violations, 0)

    def test_no_profile_or_trace_identity_in_policy_features(self) -> None:
        a, b = self.factory(), self.factory()
        b._channel._profile = (a._channel._profile + 1) % 4
        b._channel._state = (a._channel._state + 1) % 3
        self.assertEqual(a.current_state_features(), b.current_state_features())
        fields = set(RC.Run5CollectedTransitionV1.__dataclass_fields__)
        for forbidden in ("profile", "trace", "markov", "hidden", "tick", "noise", "gnb"):
            self.assertFalse(any(forbidden in name for name in fields), forbidden)
        self.warmup(a, 5)
        text = json.dumps([r.ledger_dict() for r in a.history()])
        for forbidden in ("FAVORABLE", "ADVERSE", "MID_VARIABLE", "FADE", "hidden_state"):
            self.assertNotIn(forbidden, text)

    def test_immediate_outcome_is_bit_identical_under_changed_snr(self) -> None:
        for at in (0, 7):
            a, b = self.factory(), self.factory()
            if at:
                self.warmup(a, at - 1)
                self.warmup(b, at - 1)
                force_next_snr(b, 23.9 if a._channel._snr < 15 else 6.1)
                self.warmup(a, 1, start=at - 1)
                self.warmup(b, 1, start=at - 1)
            else:
                b._channel._snr = b._snr_current = b.context.snr_db = 23.9
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

    def test_changed_snr_affects_only_the_successor_mcs(self) -> None:
        differ = 0
        for trial in range(10):
            a, b = self.factory(), self.factory()
            if trial:
                self.warmup(a, trial - 1)
                self.warmup(b, trial - 1)
                force_next_snr(a, 5.6)
                force_next_snr(b, 24.4)
                self.warmup(a, 1, start=trial - 1)
                self.warmup(b, 1, start=trial - 1)
            else:
                a._channel._snr = a._snr_current = a.context.snr_db = 5.6
                b._channel._snr = b._snr_current = b.context.snr_db = 24.4
            self.warmup(a, 1, start=trial)
            self.warmup(b, 1, start=trial)
            self.assertEqual(a.history()[-1].reward, b.history()[-1].reward)
            low, high = a.diagnostics()[-1]["successor_mcs"], b.diagnostics()[-1]["successor_mcs"]
            self.assertLessEqual(low, high)
            differ += int(low != high)
        self.assertGreater(differ, 0)

    def test_collector_restore_rebuilds_joint_channel(self) -> None:
        a = self.factory()
        self.warmup(a, 25)
        b = self.factory()
        b.restore(a.checkpoint())
        self.assertEqual(b.channel_checkpoint(), a.channel_checkpoint())
        self.assertEqual(b.checkpoint(), a.checkpoint())


class RewardAndPreviousOutcomeTest(_Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        collector = cls.factory()
        cls.records = cls.warmup(cls, collector, 288)
        cls.diagnostics = collector.diagnostics()

    def test_reward_is_exactly_the_registered_formula(self) -> None:
        self.assertTrue(all(RT.reward_matches(r) for r in self.records))
        success = [r for r in self.records if r.terminal == "SUCCESS"]
        timeout = [r for r in self.records if r.terminal == "TIMEOUT"]
        self.assertTrue(success and timeout)
        for r in success:
            self.assertEqual(r.reward, r.q_perc - 0.25 * (r.latency_ms / 170.0))
        self.assertEqual({r.reward for r in timeout}, {-1.0})
        self.assertEqual(R4.REWARD_SCHEMA_DESCRIPTOR["absent_terms"],
                         ("p_admit", "switch_penalty", "payload_penalty", "aggression_penalty"))

    def test_previous_action_and_outcome_are_nonzero_and_propagate(self) -> None:
        seen = set()
        for previous, record in zip(self.records, self.records[1:]):
            block = record.state[4:21]
            self.assertEqual(tuple(block), RT.expected_previous_features(previous))
            self.assertEqual(sum(block[:12]), 1.0)
            # previous q (the registered mode-11 support legitimately includes q_e4 = 0)
            self.assertEqual(block[12] > 0.0, previous.q_e4 > 0)
            self.assertEqual(block[15], 1.0)           # present
            if previous.terminal == "SUCCESS":
                self.assertEqual(block[13], previous.q_perc)
                self.assertGreater(block[14], 0.0)
                self.assertEqual(block[16], 1.0)
            else:
                self.assertEqual((block[13], block[14], block[16]), (0.0, 0.0, 0.0))
            seen.add(previous.terminal)
        self.assertEqual(seen, {"SUCCESS", "TIMEOUT"})

    def test_registered_delivery_failure_is_encoded_like_the_contract(self) -> None:
        fx = TS.SnrV2Test("setUp")
        TS.SnrV2Test.setUpClass()
        fx.fx = TS.SnrV2Test.fx
        vectors = {}
        for kind in (R4.RewardEventKind.DELIVERED_SUCCESS,
                     R4.RewardEventKind.REGISTERED_DELIVERY_FAILURE,
                     R4.RewardEventKind.TIMEOUT):
            latency = 171_000_000 if kind is R4.RewardEventKind.TIMEOUT else 85_000_000
            event = fx.fx.event(kind=kind, latency_ns=latency)
            event = replace(event, clock_domain=SNR.LIVE_CLOCK_DOMAIN)
            previous = R4.PreviousOutcomeV1.from_resolution(R4.resolve_reward(event))
            commit = 1_250_000_000
            state = replace(fx.state(commit), identity=R4.DecisionIdentityV1(TS.SESSION,
                                                                            TS.UE_ID, 1),
                            previous=previous)
            boundary = R4.DecisionBoundaryV1(
                identity=R4.DecisionIdentityV1(TS.SESSION, TS.UE_ID, 1),
                state_commit_timestamp_ns=commit, action_open_timestamp_ns=commit + 10_000_000,
                clock_domain=SNR.LIVE_CLOCK_DOMAIN)
            adapter = fx.adapter(start=commit - 20_000_000)
            adapter.record_command_ack(command_id="c", status="ACK", clamped=False,
                                       target_snr_db=12.0)
            adapter.record_heartbeat(active_command_id="c")
            observation = adapter.observe(boundary)
            guarded = SNR.guard_run5_state_v2(state, observation, boundary,
                                              fx.fx.freshness(), fx.lease())
            vectors[kind] = SNR.build_run5_features_v2(guarded, fx.fx.scaling()).as_tuple()
        failure = vectors[R4.RewardEventKind.REGISTERED_DELIVERY_FAILURE]
        self.assertEqual(failure[19:21], (1.0, 0.0))
        self.assertGreater(failure[16], 0.0)
        self.assertEqual(failure[17:19], (0.0, 0.0))
        self.assertNotEqual(failure[4:21], vectors[R4.RewardEventKind.DELIVERED_SUCCESS][4:21])
        self.assertEqual(failure[:21], vectors[R4.RewardEventKind.TIMEOUT][:21])


class TrainingTest(_Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp(prefix="run5_training_test_"))
        cls.orchestrator = RT.Run5OrchestratorV1(collector_factory=cls.factory, seed=17,
                                                 checkpoint_updates=(0, 100),
                                                 preregistration_sha256=PREREG)
        cls.bundles = {}

        def boundary(kind):
            payloads, identity = cls.orchestrator.bundle_payloads()
            identity.update(kind=kind.upper(), selection_candidate=True, metrics_prefix={},
                            decisions_prefix={}, actor_tree_sha256="")
            name = B.bundle_name(kind, cls.orchestrator.update_count)
            B.publish_bundle(cls.tmp, name, payloads, identity)
            cls.bundles[cls.orchestrator.update_count] = cls.tmp / name

        cls.orchestrator.run_to(100, on_decision=lambda *_: None, on_update=lambda *_: None,
                                on_boundary=boundary)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    def restore(self, path):
        return RT.Run5OrchestratorV1.restore_from_bundle(
            B.verify_bundle(path), collector_factory=self.factory, preregistration_sha256=PREREG)

    def test_config_equals_run4_hyperparameters_without_binding_run4_preregistration(self) -> None:
        run4 = R4PREREG.FROZEN_CONFIG
        run5 = PR.CONFIG
        for name in ("gamma_per_tensor", "alpha_d", "alpha_c", "actor_learning_rate",
                     "critic_learning_rate", "polyak_tau", "batch_size", "replay_capacity",
                     "warmup_decision_count", "environment_transitions_per_update",
                     "torch_intraop_threads", "seed_order"):
            self.assertEqual(getattr(run5, name), getattr(run4, name), name)
        config = R4M.run4_model_config()
        self.assertEqual((run5.hidden_width, run5.hidden_depth, run5.log_std_min, run5.log_std_max),
                         (config.hidden_width, config.hidden_depth, config.log_std_min,
                          config.log_std_max))
        text = json.dumps(self.orchestrator.binding_document) + json.dumps(
            RT._plain(RM.RUN5_TRAINING_MODEL_BINDING))
        for forbidden in (R4PREREG.PREREGISTRATION_SHA256, R4M.RUN4_MODEL_BINDING_SHA256,
                          R4M.RUN4_MODEL_SCHEMA, "modeled_smoke_orchestrator"):
            self.assertNotIn(forbidden, text)
        self.assertIs(RT.Run5HybridSacTrainerV1.update_once, T._Run4TrainerCore.update_once)
        self.assertEqual({r.discount for r in self.orchestrator.history}, {0.99 ** 2})
        self.assertEqual(self.orchestrator.actor.config.state_dim, 22)
        self.assertEqual(PR.CONFIG.deep_checkpoints[-1], 10_000)
        gaps = [b - a for a, b in zip(PR.CONFIG.deep_checkpoints, PR.CONFIG.deep_checkpoints[1:])]
        self.assertLessEqual(max(gaps), 500)

    def test_bundle_restore_is_exact_without_gradient_replay(self) -> None:
        with mock.patch.object(RT.Run5HybridSacTrainerV1, "update_once",
                               side_effect=AssertionError("gradient replay")):
            for update, path in self.bundles.items():
                restored = self.restore(path)
                self.assertEqual(restored.update_count, update)
                self.assertEqual(restored.boundary(),
                                 json.loads(B.verify_bundle(path).payload("event.json"))["boundary"])
        restored = self.restore(self.bundles[0])
        restored.run_to(100, on_decision=lambda *_: None, on_update=lambda *_: None,
                        on_boundary=lambda *_: None)
        self.assertEqual(restored.boundary(), self.orchestrator.boundary())

    def test_21d_batch_is_refused_by_the_trainer(self) -> None:
        batch = self.orchestrator.replay.sample(8, torch.Generator().manual_seed(3))
        narrow = RT.Run5ReplayBatchV1(**{**{f: getattr(batch, f) for f in
                                            RT.Run5ReplayBatchV1.__dataclass_fields__},
                                         "_state": batch.state[:, :21]})
        with self.assertRaises(T.TrainerPreflightError):
            self.orchestrator.trainer.update_once(narrow)

    def _forge(self, *, manifest_edit=None, actor_state=None):
        source = B.verify_bundle(self.bundles[100])
        payloads = {name: source.payload(name) for name in source.manifest["files"]}
        if actor_state is not None:
            payloads["actor_state_dict.pt"] = B.torch_bytes(actor_state)
        identity = {k: v for k, v in source.manifest.items()
                    if k not in ("bundle_schema", "name", "files")}
        if manifest_edit:
            manifest_edit(identity)
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        B.publish_bundle(root, "checkpoint_000100", payloads, identity)
        return root / "checkpoint_000100"

    def test_run4_or_migrated_checkpoints_and_identity_mismatches_are_refused(self) -> None:
        run4 = R4M.build_run4_models(actor_seed=1, critic_seed=2)
        padded = {k: v.clone() for k, v in run4.actor.state_dict().items()}
        padded["encoder.0.weight"] = torch.cat(
            [padded["encoder.0.weight"], torch.zeros(128, 1)], dim=1)
        order = list(V1.RUN5_POLICY_FEATURE_ORDER)
        cases = {
            "run4_21d_actor": dict(actor_state=run4.actor.state_dict()),
            "zero_padded_run4_actor": dict(actor_state=padded),
            "relabelled_run4_binding": dict(manifest_edit=lambda m: m.update(
                model_binding_sha256=R4M.RUN4_MODEL_BINDING_SHA256)),
            "feature_order": dict(manifest_edit=lambda m: m.update(
                feature_order=[order[1], order[0], *order[2:]])),
            "feature_count": dict(manifest_edit=lambda m: m.update(feature_order=order[:21])),
            "scaling_identity": dict(manifest_edit=lambda m: m.update(
                feature_schema_sha256=V1.FEATURE_SCHEMA_SHA256)),
            "preregistration": dict(manifest_edit=lambda m: m.update(
                preregistration_sha256="1" * 64)),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises((RT.Run5TrainingError, RM.Run5CheckpointRefused)):
                    self.restore(self._forge(**kwargs))
        with self.assertRaises(B.BundleCorrupt):   # an ordinary Run-4 smoke checkpoint file
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp) / "checkpoint_000500"
                fake.mkdir()
                (fake / "update_000500.checkpoint.json").write_text(
                    json.dumps({"schema": orch.CHECKPOINT_SCHEMA_ID}))
                B.verify_bundle(fake)
        with self.assertRaises(RM.Run5CheckpointRefused):
            RM.load_run5_model_state(binding=R4M.RUN4_MODEL_BINDING,
                                     actor_state=self.orchestrator.actor.state_dict(),
                                     critic_state=self.orchestrator.critics.state_dict(),
                                     expected_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256)

    def test_cuda_never_initialized(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
