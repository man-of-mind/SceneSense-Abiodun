"""CPU-only tests: Run-5B collector causality, reward, transport prior, bundles, refusals."""

from __future__ import annotations

import dataclasses
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
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_collector as R5C
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as R5M
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_preregistration as R5PR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_bundle as B
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_collector as RC
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_models as RM
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_preregistration as PR
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as C
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_training as RT

WORKTREE = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(os.environ.get("RUN5_EVIDENCE_ROOT", WORKTREE.parent / "abiodun")).resolve()
ARTIFACT = WORKTREE / ("rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/"
                       "transport_model_v2.json")
PREREG = "0" * 64
SNR_I = C.SNR_FEATURE_INDEX


def force_next_snr(collector, value: float) -> None:
    """Make the sample at the next decision's tick equal ``value`` (RNG stream kept)."""
    channel = collector._channel
    target_tick = channel._tick + 2
    original = channel._emit

    def emit():
        sample = original()
        return value if channel._tick == target_tick else sample

    channel._emit = emit


class QualityPerturbedCatalog:
    """Same scenes and payloads; every Q_perc replaced by ``f(q_perc)``."""

    def __init__(self, catalog, transform) -> None:
        self._catalog, self._transform = catalog, transform

    def __getattr__(self, name):
        return getattr(self._catalog, name)

    def draw(self, key, **kwargs):
        draw = self._catalog.draw(key, **kwargs)
        return dataclasses.replace(draw, q_perc=self._transform(draw.q_perc))


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.shared = RC.build_shared_sources(ARTIFACT, EVIDENCE_ROOT)

    @classmethod
    def factory(cls) -> RC.Run5BModeledCollectorV1:
        return RC.Run5BModeledCollectorV1(artifact_path=ARTIFACT, seed=17,
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
    def test_21d_state_with_run4b_order_and_registered_snr_scaling(self) -> None:
        collector = self.factory()
        records = self.warmup(collector, 30)
        for record, diag in zip(records, collector.diagnostics()):
            self.assertEqual(len(record.state), 21)
            self.assertEqual(record.state[SNR_I], (diag["snr_db"] - 5.5) / 19.0)
            self.assertEqual(record.state[2], diag["prior_ul_mcs"] / 28.0)
        for previous, record in zip(records, records[1:]):
            self.assertEqual(previous.next_state, record.state)

    def test_outcomes_identical_to_frozen_run5_and_state_is_run5_minus_qperc(self) -> None:
        # Same seed, same actions: the environment, channel and reward are Run 5's.
        run5 = R5C.Run5ModeledCollectorV1(artifact_path=ARTIFACT, seed=17,
                                          evidence_root=EVIDENCE_ROOT, shared_sources=self.shared)
        run5b = self.factory()
        a, b = self.warmup(run5, 60), self.warmup(run5b, 60)
        for x, y in zip(a, b):
            self.assertEqual((x.reward, x.terminal, x.q_perc, x.latency_ms, x.discount),
                             (y.reward, y.terminal, y.q_perc, y.latency_ms, y.discount))
            self.assertEqual(x.state[:17] + x.state[18:], y.state)
        for dx, dy in zip(run5.diagnostics(), run5b.diagnostics()):
            for key in ("snr_db", "prior_ul_mcs", "successor_mcs", "pre_enqueue_backlog_bytes",
                        "reward_scene_key", "held_scene_key", "on_time", "reward"):
                self.assertEqual(dx[key], dy[key], key)
        self.assertEqual(run5.channel_checkpoint(), run5b.channel_checkpoint())

    def test_qperc_changes_reward_but_never_the_actor_state(self) -> None:
        base = self.factory()
        perturbed = self.factory()
        perturbed.catalog = QualityPerturbedCatalog(perturbed.catalog, lambda q: q * 0.5)
        a, b = self.warmup(base, 60), self.warmup(perturbed, 60)
        self.assertEqual([r.state for r in a], [r.state for r in b])
        self.assertEqual([r.next_state for r in a], [r.next_state for r in b])
        self.assertEqual([r.prior_sha256 for r in a], [r.prior_sha256 for r in b])
        success = [(x, y) for x, y in zip(a, b) if x.terminal == "SUCCESS"]
        self.assertEqual([x.terminal for x in a], [y.terminal for y in b])
        for x, y in success:
            self.assertEqual(y.q_perc, x.q_perc * 0.5)
        changed = [(x, y) for x, y in success if x.q_perc > 0.0]
        self.assertGreater(len(changed), 20)
        for x, y in changed:
            self.assertNotEqual(x.reward, y.reward)

    def test_modeled_snr_uses_the_frozen_live_selection_rule(self) -> None:
        collector = self.factory()
        self.assertIsInstance(collector._snr_adapter, SNR.RfsimLeaseSnrAdapterV1)
        self.assertIs(type(collector._snr_adapter).observe, SNR.RfsimLeaseSnrAdapterV1.observe)
        self.assertEqual(collector._lease, R5C.lease_policy())
        with self.assertRaises(R4.MetadataError):
            collector._snr_adapter.record_heartbeat(active_command_id="x")  # no wall clock
        collector.current_state_features()
        ack = collector._snr_adapter._acks[-1]
        beat = collector._snr_adapter._heartbeats[-1]
        commit = collector._env.current_state.state.boundary.state_commit_timestamp_ns
        self.assertEqual((ack.ack_ns, beat[0]), (commit - 20_000_000, commit - 5_000_000))

    def test_builder_drift_is_caught_by_the_environment_audit(self) -> None:
        collector = self.factory()
        original = C.build_run5b_policy_features

        def drifted(guarded, scaling):
            vector = original(guarded, scaling)
            values = list(vector.as_tuple())
            values[3] += 1e-9
            forged = C.Run5BPolicyFeatureVectorV1(tuple(values), vector.guarded_state_sha256,
                                                  vector.empirical_scaling_sha256)
            return replace(forged, _attestation=C._issue_features(forged._binding()))

        with mock.patch.object(C, "build_run5b_policy_features", drifted):
            with self.assertRaises(RC.Run5BCollectorError):
                collector.current_state_features()

    def test_builder_is_native_and_never_calls_run4_or_run5_builders(self) -> None:
        collector = self.factory()
        with mock.patch.object(R4, "guard_state_for_action",
                               side_effect=AssertionError("Run-4 guard called")), \
                mock.patch.object(R4, "build_policy_features",
                                  side_effect=AssertionError("Run-4 builder called")), \
                mock.patch.object(SNR, "build_run5_features_v2",
                                  side_effect=AssertionError("Run-5 builder called")):
            # The environment already built its internal state at reset; the
            # Run-5B actor state for this decision is built with both patched.
            values = collector.current_state_features()
        self.assertEqual(len(values), 21)

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
            self.assertNotEqual(record.state[SNR_I], (diag["successor_snr_db"] - 5.5) / 19.0)
        self.assertEqual(collector._channel.future_sample_violations, 0)

    def test_no_profile_trace_or_quality_identity_in_policy_records(self) -> None:
        a, b = self.factory(), self.factory()
        b._channel._profile = (a._channel._profile + 1) % 4
        b._channel._state = (a._channel._state + 1) % 3
        self.assertEqual(a.current_state_features(), b.current_state_features())
        fields = set(RC.Run5BCollectedTransitionV1.__dataclass_fields__)
        for forbidden in ("profile", "trace", "markov", "hidden", "tick", "noise", "gnb"):
            self.assertFalse(any(forbidden in name for name in fields), forbidden)
        self.warmup(a, 5)
        text = json.dumps([r.ledger_dict() for r in a.history()])
        for forbidden in ("FAVORABLE", "ADVERSE", "MID_VARIABLE", "FADE", "hidden_state"):
            self.assertNotIn(forbidden, text)
        prior_text = json.dumps([a.transport_prior(k).to_canonical_dict() for k in range(1, 5)])
        self.assertNotIn("q_perc", prior_text)

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
            self.assertNotEqual(ra.state[SNR_I], rb.state[SNR_I])
            self.assertEqual(ra.state[:SNR_I], rb.state[:SNR_I])
            for field in ("reward", "terminal", "q_perc", "latency_ms", "discount", "duration"):
                self.assertEqual(getattr(ra, field), getattr(rb, field), field)
            for field in ("on_time_probability", "on_time", "ingress_bytes",
                          "reward_frame_wire_bytes", "next_backlog_bytes"):
                self.assertEqual(da[field], db[field], field)

    def test_collector_restore_rebuilds_joint_channel_and_priors(self) -> None:
        a = self.factory()
        self.warmup(a, 25)
        b = self.factory()
        b.restore(a.checkpoint())
        self.assertEqual(b.channel_checkpoint(), a.channel_checkpoint())
        self.assertEqual(b.checkpoint(), a.checkpoint())
        self.assertEqual([r.prior_sha256 for r in a.history()],
                         [r.prior_sha256 for r in b.history()])
        run5_checkpoint = R5C.Run5ModeledCollectorV1(
            artifact_path=ARTIFACT, seed=17, evidence_root=EVIDENCE_ROOT,
            shared_sources=self.shared).checkpoint()
        with self.assertRaises(RC.Run5BCollectorError):
            b.restore(run5_checkpoint)


class RewardAndPreviousOutcomeTest(_Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        collector = cls.factory()
        cls.records = cls.warmup(cls, collector, 288)
        cls.collector = collector

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

    def test_transport_prior_encoding_propagates_and_carries_no_qperc(self) -> None:
        seen = set()
        for k, (previous, record) in enumerate(zip(self.records, self.records[1:]), start=1):
            block = record.state[C.PREVIOUS_SLICE]
            self.assertEqual(tuple(block), RT.expected_previous_features(previous))
            prior = self.collector.transport_prior(k)
            self.assertEqual(record.prior_sha256, prior.canonical_sha256())
            self.assertEqual(prior.terminal.value, previous.terminal)
            self.assertEqual(prior.action.mode_id, previous.mode_id)
            self.assertEqual(prior.operational_latency_ms, previous.latency_ms)
            self.assertEqual(sum(block[:12]), 1.0)
            self.assertEqual(block[12] > 0.0, previous.q_e4 > 0)
            self.assertEqual(block[14], 1.0)            # present
            if previous.terminal == "SUCCESS":
                self.assertGreater(block[13], 0.0)
                self.assertEqual(block[15], 1.0)
                if 0.0 < previous.q_perc < 1.0:
                    self.assertNotIn(previous.q_perc, record.state)
            else:
                self.assertEqual((block[13], block[15]), (0.0, 0.0))
            seen.add(previous.terminal)
        self.assertEqual(seen, {"SUCCESS", "TIMEOUT"})
        self.assertEqual(RT.qperc_leaks(self.records), 0)
        self.assertIsNone(self.records[0].prior_sha256)
        self.assertEqual(self.records[0].state[C.PREVIOUS_SLICE], (0.0,) * 16)


class TrainingTest(_Base):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp(prefix="run5b_training_test_"))
        cls.orchestrator = RT.Run5BOrchestratorV1(collector_factory=cls.factory, seed=17,
                                                  checkpoint_updates=(0, 100),
                                                  preregistration_sha256=PREREG)
        cls.bundles = {}

        def boundary(kind):
            payloads, identity = cls.orchestrator.bundle_payloads()
            identity.update(kind=kind.upper(), selection_candidate=True, metrics_prefix={},
                            decisions_prefix={}, actor_tree_sha256=RT._tree_sha256(
                                cls.orchestrator.actor.state_dict()))
            name = B.bundle_name(kind, cls.orchestrator.update_count)
            B.publish_bundle(cls.tmp, name, payloads, identity)
            cls.bundles[cls.orchestrator.update_count] = cls.tmp / name

        cls.orchestrator.run_to(100, on_decision=lambda *_: None, on_update=lambda *_: None,
                                on_boundary=boundary)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp)

    def restore(self, path):
        return RT.Run5BOrchestratorV1.restore_from_bundle(
            B.verify_bundle(path), collector_factory=self.factory, preregistration_sha256=PREREG)

    def test_config_equals_run4_and_run5_without_binding_their_preregistrations(self) -> None:
        run4, run5, run5b = R4PREREG.FROZEN_CONFIG, R5PR.CONFIG, PR.CONFIG
        for name in ("gamma_per_tensor", "alpha_d", "alpha_c", "actor_learning_rate",
                     "critic_learning_rate", "polyak_tau", "batch_size", "replay_capacity",
                     "warmup_decision_count", "environment_transitions_per_update",
                     "torch_intraop_threads", "seed_order"):
            self.assertEqual(getattr(run5b, name), getattr(run4, name), name)
        for name in R5PR.Run5TrainingConfigV1.__dataclass_fields__:
            self.assertEqual(getattr(run5b, name), getattr(run5, name), name)
        self.assertEqual((run5b.live_actor_seed, run5b.live_actor_update), (43, 10_000))
        config = R4M.run4_model_config()
        self.assertEqual((run5b.hidden_width, run5b.hidden_depth, run5b.log_std_min,
                          run5b.log_std_max),
                         (config.hidden_width, config.hidden_depth, config.log_std_min,
                          config.log_std_max))
        text = json.dumps(self.orchestrator.binding_document) + json.dumps(
            RT._plain(RM.RUN5B_TRAINING_MODEL_BINDING))
        for forbidden in (R4PREREG.PREREGISTRATION_SHA256, R4M.RUN4_MODEL_BINDING_SHA256,
                          R4M.RUN4_MODEL_SCHEMA, R5M.RUN5_TRAINING_MODEL_BINDING_SHA256,
                          RM.RUN5_PREREGISTRATION_SHA256, "modeled_smoke_orchestrator"):
            self.assertNotIn(forbidden, text)
        self.assertIs(RT.Run5BHybridSacTrainerV1.update_once, T._Run4TrainerCore.update_once)
        self.assertEqual({r.discount for r in self.orchestrator.history}, {0.99 ** 2})
        self.assertEqual(self.orchestrator.actor.config.state_dim, 21)
        gaps = [b - a for a, b in zip(PR.CONFIG.deep_checkpoints, PR.CONFIG.deep_checkpoints[1:])]
        self.assertLessEqual(max(gaps), 500)
        self.assertEqual(PR.CONFIG.deep_checkpoints[-1], 10_000)

    def test_critic_sees_34_inputs_and_actor_21(self) -> None:
        batch = self.orchestrator.replay.sample(8, torch.Generator().manual_seed(3))
        self.assertEqual(tuple(batch.state.shape), (8, 21))
        q1, q2 = self.orchestrator.critics.q_values(
            batch.state, torch.nn.functional.one_hot(batch.mode_id, 12).float(),
            batch.q_normalized_executed)
        self.assertEqual(tuple(q1.shape), (8,))

    def test_bundle_restore_is_exact_without_gradient_replay(self) -> None:
        with mock.patch.object(RT.Run5BHybridSacTrainerV1, "update_once",
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

    def test_22d_batch_is_refused_by_the_trainer(self) -> None:
        batch = self.orchestrator.replay.sample(8, torch.Generator().manual_seed(3))
        wide = RT.Run5BReplayBatchV1(**{**{f: getattr(batch, f) for f in
                                           RT.Run5BReplayBatchV1.__dataclass_fields__},
                                        "_state": torch.cat([batch.state,
                                                             torch.zeros(8, 1)], dim=1)})
        with self.assertRaises(T.TrainerPreflightError):
            self.orchestrator.trainer.update_once(wide)

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

    def test_run4_run5_or_migrated_checkpoints_and_identity_mismatches_are_refused(self) -> None:
        run4 = R4M.build_run4_models(actor_seed=1, critic_seed=2)
        run5_actor, _ = R5M.build_run5_models(actor_seed=1, critic_seed=2)
        sliced = {k: v.clone() for k, v in run5_actor.state_dict().items()}
        weight = sliced["encoder.0.weight"]
        sliced["encoder.0.weight"] = torch.cat([weight[:, :17], weight[:, 18:]], dim=1)
        order = list(C.RUN5B_POLICY_FEATURE_ORDER)
        cases = {
            "run4_21d_actor": dict(actor_state=run4.actor.state_dict()),
            "run5_22d_actor": dict(actor_state=run5_actor.state_dict()),
            "run5_actor_sliced_to_21": dict(actor_state=sliced),
            "relabelled_run4_binding": dict(manifest_edit=lambda m: m.update(
                model_binding_sha256=R4M.RUN4_MODEL_BINDING_SHA256)),
            "relabelled_run5_binding": dict(manifest_edit=lambda m: m.update(
                model_binding_sha256=R5M.RUN5_TRAINING_MODEL_BINDING_SHA256)),
            "run4_feature_order": dict(manifest_edit=lambda m: m.update(
                feature_order=list(R4.POLICY_FEATURE_ORDER))),
            "feature_order_swap": dict(manifest_edit=lambda m: m.update(
                feature_order=[order[1], order[0], *order[2:]])),
            "feature_count": dict(manifest_edit=lambda m: m.update(feature_order=order[:20])),
            "run5_feature_schema": dict(manifest_edit=lambda m: m.update(
                feature_schema_sha256=SNR.FEATURE_SCHEMA_SHA256)),
            "schema_id": dict(manifest_edit=lambda m: m.update(feature_schema_id="other")),
            "preregistration": dict(manifest_edit=lambda m: m.update(
                preregistration_sha256="1" * 64)),
            "run5_preregistration": dict(manifest_edit=lambda m: m.update(
                preregistration_sha256=RM.RUN5_PREREGISTRATION_SHA256)),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                with self.assertRaises((RT.Run5BTrainingError, RM.Run5BCheckpointRefused)):
                    self.restore(self._forge(**kwargs))
        with self.assertRaises(B.BundleCorrupt):   # an ordinary Run-4 smoke checkpoint file
            with tempfile.TemporaryDirectory() as tmp:
                fake = Path(tmp) / "checkpoint_000500"
                fake.mkdir()
                (fake / "update_000500.checkpoint.json").write_text(
                    json.dumps({"schema": orch.CHECKPOINT_SCHEMA_ID}))
                B.verify_bundle(fake)

    def test_cuda_never_initialized(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
