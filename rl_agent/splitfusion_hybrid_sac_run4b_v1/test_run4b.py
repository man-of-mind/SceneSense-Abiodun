"""Adversarial Run-4B contract, environment, learner and checkpoint tests.

CPU-only; run with ``env -u PYTHONPATH CUDA_VISIBLE_DEVICES=``.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import torch

torch.set_num_threads(4)

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as run4_models
from rl_agent.splitfusion_hybrid_sac_run4_v1 import trainer as run4_trainer
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import environment as E
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import learner as L
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import models as M
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import runner as R
from rl_agent.splitfusion_operational_latency_v1 import provider as OPL
from rl_agent.splitfusion_operational_latency_v1 import test_provider as TP

ROOT = R.ROOT
RUN4_EVENT_CHECKPOINT = ROOT / (
    "rl_agent/experiments/splitfusion_hybrid_sac_run4_v2_campaign/"
    "20260928_2a92201_three_seed_10000_v1/seed_43/checkpoints/"
    "update_010000.checkpoint.json")
RUN4_ACTOR_EXPORT = ROOT / (
    "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
    "20260929_seed43_update10000_actor_export")
SCALING = C.ScalingV1(110.0, 12.0, 17.7275)


def _bits(values) -> bytes:
    return struct.pack(f"<{len(values)}d", *values)


def _obs() -> C.ObservationV1:
    return C.ObservationV1(115.0, 0.4, 20, 1234)


class ContractTest(unittest.TestCase):
    def test_feature_order_exact(self) -> None:
        expected = ("camera_si_scaled", "radar_p40", "prior_ul_mcs_normalized",
                    "pre_action_rlc_backlog_log1p_scaled",
                    *(f"prev_joint_mode_{m}_one_hot" for m in range(12)),
                    "prev_q_normalized", "prev_operational_latency_normalized",
                    "prev_present", "prev_operational_success")
        self.assertEqual(C.FEATURE_ORDER, expected)
        self.assertEqual((C.FEATURE_COUNT, C.CRITIC_INPUT_WIDTH), (20, 33))

    def test_no_forbidden_information_in_feature_names(self) -> None:
        for name in C.FEATURE_ORDER:
            for token in C.FORBIDDEN_FEATURE_TOKENS:
                self.assertNotIn(token, name.lower(), (name, token))

    def test_prior_has_no_quality_or_reward_field(self) -> None:
        names = {f.name for f in dataclasses.fields(C.OperationalPriorV1)}
        self.assertEqual(names, {"kind", "mode_id", "q_e4",
                                 "operational_latency_ns"})
        self.assertNotIn("q_perc", inspect.signature(
            C.OperationalPriorV1.from_outcome).parameters)
        self.assertNotIn("q_perc", inspect.signature(C.build_features).parameters)

    def test_successful_state_builds_without_qperc(self) -> None:
        prior = C.OperationalPriorV1.from_outcome(
            mode_id=7, q_e4=4321, timely=True,
            operational_latency_ns=123_456_789)
        values = C.build_features(_obs(), prior, SCALING)
        self.assertEqual(len(values), 20)
        self.assertEqual(values[4 + 7], 1.0)
        self.assertEqual(values[16], 4321 / 9800)
        self.assertEqual(values[17], 123.456789 / 170.0)
        self.assertEqual(values[18:], (1.0, 1.0))

    def test_genesis_and_failure_semantics(self) -> None:
        genesis = C.build_features(_obs(), C.OperationalPriorV1.genesis(),
                                   SCALING)
        self.assertEqual(genesis[4:], (0.0,) * 16)
        failure = C.build_features(_obs(), C.OperationalPriorV1.from_outcome(
            mode_id=2, q_e4=100, timely=False, operational_latency_ns=None),
            SCALING)
        self.assertEqual(failure[4 + 2], 1.0)
        self.assertEqual(failure[16], 100 / 9800)
        self.assertEqual(failure[17:], (0.0, 1.0, 0.0))
        with self.assertRaises(C.ContractError):
            C.OperationalPriorV1(C.PriorKind.FAILURE, 2, 100, 170_000_001)
        with self.assertRaises(C.ContractError):  # censoring value
            C.OperationalPriorV1(C.PriorKind.TIMELY_ACK, 2, 100, 170_000_001)
        with self.assertRaises(C.ContractError):
            C.OperationalPriorV1(C.PriorKind.GENESIS, 0, None, None)

    def test_previous_latency_and_success_change_intended_fields_only(self):
        def features(timely, latency):
            return C.build_features(_obs(), C.OperationalPriorV1.from_outcome(
                mode_id=9, q_e4=5000, timely=timely,
                operational_latency_ns=latency), SCALING)
        fast, slow = features(True, 40_000_000), features(True, 150_000_000)
        failed = features(False, None)
        self.assertEqual([i for i in range(20) if fast[i] != slow[i]], [17])
        self.assertEqual([i for i in range(20) if fast[i] != failed[i]],
                         [17, 19])

    def test_reward_boundary(self) -> None:
        exact = C.resolve_reward(kind=C.RewardKind.TIMELY_SUCCESS, q_perc=0.8,
                                 operational_latency_ns=170_000_000)
        self.assertEqual(exact.reward, 0.8 - 0.25)
        with self.assertRaises(C.ContractError):
            C.resolve_reward(kind=C.RewardKind.TIMELY_SUCCESS, q_perc=0.8,
                             operational_latency_ns=170_000_001)
        self.assertEqual(C.resolve_reward(
            kind=C.RewardKind.TIMEOUT_OR_FAILURE).reward, -1.0)
        excluded = C.resolve_reward(kind=C.RewardKind.EXCLUDED_FAULT)
        self.assertFalse(excluded.learning_included)
        self.assertIsNone(excluded.reward)
        a = C.resolve_reward(kind=C.RewardKind.TIMELY_SUCCESS, q_perc=0.3,
                             operational_latency_ns=85_000_000)
        b = C.resolve_reward(kind=C.RewardKind.TIMELY_SUCCESS, q_perc=0.9,
                             operational_latency_ns=85_000_000)
        self.assertAlmostEqual(b.reward - a.reward, 0.6, places=12)


class _SourcesMixin:
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = E.SharedSourcesV1.load()
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
            modeled_smoke_orchestrator as R4O)
        cls.schedule = R4O.build_frozen_warmup_schedule(17)

    def _actions(self, n):
        return [(self.schedule.action_at(i % 288).mode_id,
                 self.schedule.action_at(i % 288).q_e4) for i in range(n)]


class EnvironmentTest(_SourcesMixin, unittest.TestCase):
    def test_qperc_changes_reward_but_never_actor_features(self) -> None:
        reference = E.Run4BEnvironmentV1(self.sources, seed=17)
        scaled_sources = dataclasses.replace(self.sources)
        original_draw = self.sources.catalog.draw

        class ScaledCatalog:
            def __getattr__(self, name):
                return getattr(self_catalog, name)

            def draw(self, key, *, mode_id, q_e4):
                draw = original_draw(key, mode_id=mode_id, q_e4=q_e4)
                return dataclasses.replace(draw, q_perc=draw.q_perc * 0.5 + 0.1)

        self_catalog = self.sources.catalog
        scaled_sources.catalog = ScaledCatalog()
        changed = E.Run4BEnvironmentV1(scaled_sources, seed=17)
        reward_differs = 0
        for mode_id, q_e4 in self._actions(288):
            a, b = reference.step(mode_id, q_e4), changed.step(mode_id, q_e4)
            self.assertEqual(_bits(a.state), _bits(b.state))
            self.assertEqual(_bits(a.next_state), _bits(b.next_state))
            if a.diagnostics["terminal"] == "TIMELY_SUCCESS":
                self.assertNotEqual(a.reward, b.reward)
                reward_differs += 1
            else:
                self.assertEqual(a.reward, b.reward)
        self.assertGreater(reward_differs, 200)
        self.assertEqual(reference.state_dict(), changed.state_dict())

    def test_gt_and_map_timestamps_change_no_bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = TP._copy_sources(Path(tmp))
            TP._mutate_gt_and_map(paths)
            mutated = OPL.OperationalLatencyProviderV1.unpinned_for_tests(paths)
            sources_m = dataclasses.replace(self.sources, provider=mutated)
            # The production preflight refuses the unregistered provider.
            with self.assertRaisesRegex(R.RunnerError,
                                        "provider_binding_registered.: False"):
                R.Run4BRunnerV1(sources_m, 17).run_preflight(lambda row: None)
            outputs = []
            for sources in (self.sources, sources_m):
                runner = R.Run4BRunnerV1(sources, 17)
                digests = [runner.collect_one()[0]["transition_sha256"]
                           for _ in range(R.WARMUP + 8)]
                runner.train_once()
                runner.train_once()
                outputs.append((digests, runner.decision_chain,
                                runner.fingerprint(), runner.binding_sha256))
            self.assertEqual(outputs[0][:3], outputs[1][:3])
            # The source pins differ, so the binding detects the drift.
            self.assertNotEqual(outputs[0][3], outputs[1][3])

    def test_components_once_and_family_independent_draws(self) -> None:
        env = E.Run4BEnvironmentV1(self.sources, seed=29)
        other = E.Run4BEnvironmentV1(self.sources, seed=29)
        actions = self._actions(288)
        for (mode_id, q_e4), (mode_b, q_b) in zip(actions,
                                                  reversed(actions)):
            a, b = env.step(mode_id, q_e4), other.step(mode_b, q_b)
            parts = a.diagnostics["components_ns"]
            self.assertEqual(sum(parts.values()),
                             a.diagnostics["composed_total_ns"])
            self.assertEqual(a.diagnostics["composed_total_ns"] - parts["E"],
                             parts["A"] + parts["S"] + parts["T"] + parts["D"])
            if a.diagnostics["terminal"] == "TIMELY_SUCCESS":
                self.assertEqual(a.diagnostics["operational_latency_ns"],
                                 a.diagnostics["composed_total_ns"])
                self.assertLessEqual(a.diagnostics["operational_latency_ns"],
                                     170_000_000)
            for key in ("A", "E", "D"):  # drawn before the action is known
                self.assertEqual(parts[key],
                                 b.diagnostics["components_ns"][key])

    def test_restore_reproduces_every_stream(self) -> None:
        env = E.Run4BEnvironmentV1(self.sources, seed=43)
        actions = self._actions(288)
        for mode_id, q_e4 in actions[:100]:
            env.step(mode_id, q_e4)
        snapshot = json.loads(json.dumps(env.state_dict()))
        expected = [env.step(m, q) for m, q in actions[100:200]]
        restored = E.Run4BEnvironmentV1(self.sources, seed=43)
        restored.load_state_dict(snapshot)
        actual = [restored.step(m, q) for m, q in actions[100:200]]
        self.assertEqual([t.digest() for t in actual],
                         [t.digest() for t in expected])
        self.assertEqual([t.diagnostics for t in actual],
                         [t.diagnostics for t in expected])
        self.assertEqual(env.state_dict(), restored.state_dict())

    def test_current_state_carries_no_future_information(self) -> None:
        env = E.Run4BEnvironmentV1(self.sources, seed=17)
        env.step(*self._actions(1)[0])
        before = env.current_features()
        env.context["success_uniform"] = 0.999999
        env.context["latency_draw"] = dict(env.context["latency_draw"],
                                           a_ns=1, e_ns=1, d_ns=1)
        env.context["held_scene_key"] = env.context["reward_scene_key"]
        self.assertEqual(_bits(before), _bits(env.current_features()))

    def test_exact_deadline_in_environment(self) -> None:
        outcomes = {}
        for total in (170_000_000, 170_000_001):
            env = E.Run4BEnvironmentV1(self.sources, seed=17)
            provider = self.sources.provider

            class Fixed:
                def __getattr__(self, name):
                    return getattr(provider, name)

                def resolve(self, **kwargs):
                    return OPL.OperationalOutcomeV1(
                        a_ns=total - 4, s_ns=1, t_ns=1, e_ns=1, d_ns=1,
                        total_ns=total, on_time_probability=1.0,
                        transport_success=True,
                        timely=OPL.is_timely(transport_success=True,
                                             total_ns=total))
            env.sources = dataclasses.replace(self.sources, provider=Fixed())
            transition = env.step(*self._actions(1)[0])
            outcomes[total] = transition
        ok, late = outcomes[170_000_000], outcomes[170_000_001]
        self.assertEqual(ok.diagnostics["terminal"], "TIMELY_SUCCESS")
        self.assertEqual(ok.reward, ok.diagnostics["q_perc_training_only"] - 0.25)
        self.assertEqual(ok.next_state[17:], (1.0, 1.0, 1.0))
        self.assertEqual(late.diagnostics["terminal"], "TIMEOUT_OR_FAILURE")
        self.assertEqual(late.reward, -1.0)
        self.assertEqual(late.next_state[17:], (0.0, 1.0, 0.0))

    def test_shared_provider_module_and_binding(self) -> None:
        self.assertIs(E.OPL, OPL)
        self.assertIs(type(self.sources.provider),
                      OPL.OperationalLatencyProviderV1)
        committed = json.loads((Path(OPL.__file__).parent /
                                "PROVIDER_BINDING.json").read_text())
        self.assertEqual(self.sources.provider.binding_sha256,
                         committed["binding_sha256"])
        self.assertEqual(R.REGISTERED_PROVIDER_BINDING_SHA256,
                         committed["binding_sha256"])
        self.assertEqual(self.sources.binding_document()[
            "operational_latency_provider_sha256"], committed["binding_sha256"])


class LearnerTest(unittest.TestCase):
    def test_widths_and_inherited_update(self) -> None:
        actor, critics = M.build_models(actor_seed=1, critic_seed=2)
        self.assertEqual(M._first_linear(actor.encoder).in_features, 20)
        for name in ("critic_1", "critic_2", "target_1", "target_2"):
            self.assertEqual(M._first_linear(
                getattr(critics, name).trunk).in_features, 33)
        self.assertIs(L.Run4BTrainerV1.update_once,
                      run4_trainer._Run4TrainerCore.update_once)
        self.assertNotIn("update_once", L.Run4BTrainerV1.__dict__)

    def test_trainer_refuses_foreign_width(self) -> None:
        actor, critics = M.build_models(actor_seed=1, critic_seed=2)
        trainer = L.Run4BTrainerV1(
            actor=actor, critics=critics, config=R.TRAINER_CONFIG,
            target_generator=torch.Generator().manual_seed(3),
            actor_generator=torch.Generator().manual_seed(4))
        batch = L.Run4BBatchV1(
            state=torch.zeros((4, 21)), next_state=torch.zeros((4, 21)),
            mode_id=torch.zeros(4, dtype=torch.int64),
            q_e4=torch.zeros(4, dtype=torch.int64), reward=torch.zeros(4),
            _discount=torch.full((4,), 0.9801),
            bootstrap=torch.ones(4, dtype=torch.bool))
        with self.assertRaises(run4_trainer.TrainerPreflightError):
            trainer.update_once(batch)
        with self.assertRaises(M.ModelError):
            M.validate_models(*[run4_models.build_run4_models(
                actor_seed=1, critic_seed=2).__getattribute__(n)
                for n in ("actor", "critics")])


class CheckpointTest(_SourcesMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tmp = Path(tempfile.mkdtemp(prefix="run4b_ckpt_"))
        runner = R.Run4BRunnerV1(cls.sources, 17)
        runner.run_preflight(lambda row: None)
        for _ in range(3):
            for _ in range(4):
                runner.collect_one()
            runner.train_once()
        cls.bundle = R.write_bundle(runner, cls.tmp / "checkpoints")
        cls.continued = []
        for _ in range(3):
            for _ in range(4):
                runner.collect_one()
            runner.train_once()
        cls.continued = runner.fingerprint()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _forged(self, mutate) -> Path:
        target = Path(tempfile.mkdtemp(dir=self.tmp)) / "forged"
        shutil.copytree(self.bundle, target)
        manifest = json.loads((target / "manifest.json").read_text())
        mutate(manifest, target)
        data = R._canonical(manifest)
        (target / "manifest.json").write_bytes(data)
        (target / "COMMITTED").write_text(R._sha_bytes(data) + "\n")
        return target

    def test_restore_continues_bit_identically(self) -> None:
        restored = R.restore_runner(self.bundle, self.sources, 17)
        self.assertEqual(restored.update_count, 3)
        for _ in range(3):
            for _ in range(4):
                restored.collect_one()
            restored.train_once()
        self.assertEqual(restored.fingerprint(), self.continued)
        latest = json.loads((self.tmp / "checkpoints" / "LATEST").read_text())
        self.assertEqual(latest["bundle"], self.bundle.name)

    def test_create_only(self) -> None:
        runner = R.restore_runner(self.bundle, self.sources, 17)
        with self.assertRaises(R.RunnerError):
            R.write_bundle(runner, self.tmp / "checkpoints")

    def _refused(self, path: Path, pattern: str) -> None:
        with self.assertRaisesRegex(R.CheckpointRefused, pattern):
            R.restore_runner(path, self.sources, 17)

    def test_run4_and_run5_artifacts_refused(self) -> None:
        self._refused(RUN4_EVENT_CHECKPOINT, "not a Run-4B bundle")
        self._refused(RUN4_ACTOR_EXPORT, "not COMMITTED")
        with self.assertRaisesRegex(R.CheckpointRefused, "ACTOR_EXPORT"):
            R.load_actor_export(RUN4_ACTOR_EXPORT)
        sidecar = self._forged(lambda m, t: m.update(
            schema="splitfusion.hybrid_sac.materialized_checkpoint_sidecar.v1"))
        self._refused(sidecar, "foreign checkpoint schema")

    def test_refusal_by_order_hash_and_binding_not_width(self) -> None:
        permuted = list(C.FEATURE_ORDER)
        permuted[17], permuted[19] = permuted[19], permuted[17]
        self._refused(self._forged(lambda m, t: m.update(
            feature_order=permuted)), "feature order")
        self._refused(self._forged(lambda m, t: m.update(
            feature_schema_sha256="0" * 64)), "feature schema hash")
        self._refused(self._forged(lambda m, t: m.update(
            model_binding_sha256=run4_models.RUN4_MODEL_BINDING_SHA256)),
            "model binding hash")
        self._refused(self._forged(lambda m, t: m.update(
            operational_latency_provider_sha256="1" * 64)),
            "provider binding")
        self._refused(self._forged(lambda m, t: m["runner_binding"].update(
            seed=29)), "runner binding")

        def corrupt(m, t):
            data = bytearray((t / "training_state.pt").read_bytes())
            data[-100] ^= 1
            (t / "training_state.pt").write_bytes(bytes(data))
        self._refused(self._forged(corrupt), "training_state.pt hash")

    def test_actor_export_round_trip(self) -> None:
        runner = R.restore_runner(self.bundle, self.sources, 17)
        out = self.tmp / "export"
        manifest = R.export_actor(runner, out)
        actor = R.load_actor_export(out)
        self.assertEqual(R.R4O._tree_sha256(actor.state_dict()),
                         manifest["actor_tree_sha256"])
        self.assertFalse(manifest["preregistered_live_actor"])


if __name__ == "__main__":
    unittest.main()
