"""CPU-only Run-5B tests: provider/channel identity, parity, state, widths, refusals."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as R4O
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import environment as E4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import models as M4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import runner as R4B
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as R5M
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import derive_from_run4b as D
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_checks as K
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_learner as L
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_models as M
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_registration as REG
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_runner as R
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as C
from rl_agent.splitfusion_joint_channel_v1 import joint_channel as JC
from rl_agent.splitfusion_operational_latency_v1 import provider as OPL

WORKTREE = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(os.environ.get("RUN5B_EVIDENCE_ROOT", WORKTREE.parent / "abiodun")).resolve()
PROVIDER_COMMIT_SHA256 = "579e2f64a922159716530beeefb364d22d789cfd51afd3fa186584a30bacc0c4"
FAKE_REGISTRATION = "7" * 64


def scaling() -> C4.ScalingV1:
    return C4.ScalingV1(10.0, 5.0, 17.7275)


class SchemaAndModelTest(unittest.TestCase):
    """Items 4 and 5: exact Run-4B prefix + SNR; widths; refusals."""

    def test_state_is_exact_run4b_prefix_plus_snr(self) -> None:
        self.assertEqual(C.FEATURE_ORDER, (*C4.FEATURE_ORDER, R5V1.SNR_FEATURE_NAME))
        self.assertEqual((C.FEATURE_COUNT, C.SNR_FEATURE_INDEX), (21, 20))
        rng = random.Random(5)
        for trial in range(300):
            observation = C4.ObservationV1(rng.uniform(0, 40), rng.random(), rng.randint(0, 28),
                                           rng.randint(0, 50_000_000))
            kind = trial % 3
            prior = (C4.OperationalPriorV1.genesis() if kind == 0 else
                     C4.OperationalPriorV1.from_outcome(
                         mode_id=rng.randint(0, 11), q_e4=rng.randint(0, 9800),
                         timely=kind == 1,
                         operational_latency_ns=rng.randint(1, 170_000_000) if kind == 1 else None))
            snr = rng.random()
            values = C.build_features(observation, prior, scaling(), snr)
            self.assertEqual(values[:20], C4.build_features(observation, prior, scaling()))
            self.assertEqual(values[20], snr)
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.build_features(observation, prior, scaling(), 1.5)

    def test_operational_prior_cannot_carry_qperc(self) -> None:
        fields = {f.name for f in dataclasses.fields(C4.OperationalPriorV1)}
        self.assertEqual(fields, {"kind", "mode_id", "q_e4", "operational_latency_ns"})
        with self.assertRaises(TypeError):
            C4.OperationalPriorV1(C4.PriorKind.TIMELY_ACK, 1, 100, 5, q_perc=0.5)  # noqa

    def test_widths_21_and_34_and_config_equals_run4b(self) -> None:
        actor, critics = M.build_models(actor_seed=1, critic_seed=2)
        self.assertEqual(tuple(actor.state_dict()[M.ACTOR_INPUT_KEY].shape), (128, 21))
        for key in M.CRITIC_INPUT_KEYS:
            self.assertEqual(tuple(critics.state_dict()[key].shape), (128, 34))
        a, b = dataclasses.asdict(M.model_config()), dataclasses.asdict(M4.model_config())
        self.assertEqual((a.pop("state_dim"), b.pop("state_dim")), (21, 20))
        self.assertEqual(a, b)

    def test_identity_classification_and_refusal(self) -> None:
        run4b_export = json.loads((WORKTREE / "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
                                   "evidence/seed_43_ACTOR_EXPORT.json").read_text())
        run4b_registration = json.loads((WORKTREE / "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
                                         "RUN4B_REGISTRATION.json").read_text())
        good = {"model_binding_sha256": M.MODEL_BINDING_SHA256,
                "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
                "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],
                "feature_order": list(C.FEATURE_ORDER),
                "feature_order_sha256": C.FEATURE_ORDER_SHA256,
                "registration_sha256": FAKE_REGISTRATION}
        cases = {
            M.RUN4_IDENTITY: [R4M.RUN4_MODEL_BINDING, {**good, "feature_order": list(
                R4.POLICY_FEATURE_ORDER)}, {**good, "feature_schema_sha256": R4.FEATURE_SCHEMA_SHA256}],
            M.RUN5_IDENTITY: [R5M.RUN5_TRAINING_MODEL_BINDING, R5M.RUN5_MODEL_BINDING,
                              {**good, "feature_schema_sha256": SNR.FEATURE_SCHEMA_SHA256},
                              {**good, "preregistration_sha256": M.RUN5_PREREGISTRATION_SHA256}],
            M.RUN4B_IDENTITY: [M4.MODEL_BINDING, run4b_export, run4b_registration["model_binding"],
                               {**good, "model_binding_sha256": M4.MODEL_BINDING_SHA256},
                               {**good, "feature_order": list(C4.FEATURE_ORDER)}],
        }
        for expected, documents in cases.items():
            for document in documents:
                with self.subTest(expected=expected, keys=sorted(document)[:3]):
                    self.assertEqual(M.classify_identity(document), expected)
                    with self.assertRaises(M.CheckpointRefused):
                        M.require_run5b_identity(document, registration_sha256=FAKE_REGISTRATION)
        self.assertEqual(M.classify_identity(good), M.RUN5B_IDENTITY)
        self.assertEqual(M.classify_identity(M.MODEL_BINDING), M.RUN5B_IDENTITY)
        M.require_run5b_identity(good, registration_sha256=FAKE_REGISTRATION)
        for key, value in (("feature_schema_id", "x"), ("feature_order_sha256", "0" * 64),
                           ("registration_sha256", "1" * 64)):
            with self.assertRaises(M.CheckpointRefused):
                M.require_run5b_identity({**good, key: value},
                                         registration_sha256=FAKE_REGISTRATION)

    def test_actor_tensor_refusals(self) -> None:
        run4b_actor, _ = M4.build_models(actor_seed=3, critic_seed=4)
        run5_actor, _ = R5M.build_run5_models(actor_seed=3, critic_seed=4)
        run5b_actor, _ = M.build_models(actor_seed=3, critic_seed=4)
        for state, pattern in ((run4b_actor.state_dict(), "20 is not 21"),
                               (run5_actor.state_dict(), "22 is not 21")):
            with self.assertRaisesRegex(M.CheckpointRefused, pattern):
                M.require_run5b_actor_state(state, expected_tree_sha256=None)
        tree = R4O._tree_sha256(run5b_actor.state_dict())
        self.assertEqual(M.require_run5b_actor_state(run5b_actor.state_dict(),
                                                     expected_tree_sha256=tree), tree)
        run4_actor = R4M.build_run4_models(actor_seed=5, critic_seed=6).actor
        with self.assertRaisesRegex(M.CheckpointRefused, "tensor-tree"):
            M.require_run5b_actor_state(run4_actor.state_dict(), expected_tree_sha256=tree)
        frozen = EVIDENCE_ROOT / ("rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
                                  "20260929_seed43_update10000_actor_export/actor_state_dict.pt")
        state = torch.load(frozen, map_location="cpu", weights_only=True)
        with self.assertRaisesRegex(M.CheckpointRefused, "frozen Run-4"):
            M.require_run5b_actor_state(state, expected_tree_sha256=None)

    def test_snr_admission_falls_back_and_never_zero_fills(self) -> None:
        observer = C.ModeledSnrObserverV1(17)
        self.assertEqual(observer.observe(0, 12.0), (12.0 - 5.5) / 19.0)
        self.assertEqual(observer.observe(0, 12.0), (12.0 - 5.5) / 19.0)   # idempotent
        with self.assertRaises(C.Run5BContractError):
            observer.observe(0, 13.0)
        for value in (5.49, 24.51):
            with self.assertRaises(R4.ExternalFallbackRequired):
                C.ModeledSnrObserverV1(17).observe(1, value)
        boundary = observer.boundary(3)
        adapter = SNR.ModeledLeaseSnrAdapterV1(provider_id="t", session_uuid=observer.session_uuid,
                                               ue_id=C.UE_ID)
        commit = boundary.state_commit_timestamp_ns
        adapter.record_command_ack_at(at_ns=commit - 300_000_000, command_id="c", status="ACK",
                                      clamped=False, target_snr_db=12.0)
        adapter.record_heartbeat_at(at_ns=commit - 250_000_000, active_command_id="c")
        with self.assertRaises(R4.ExternalFallbackRequired):          # stale lease
            C.admit_snr(adapter.observe(boundary), boundary, C.lease_policy())
        with self.assertRaises(R4.ExternalFallbackRequired):
            C.admit_snr(None, boundary, C.lease_policy())


class DerivationAndProviderIdentityTest(unittest.TestCase):
    """Item 1: the exact same provider implementation and binding, no copied logic."""

    def test_learner_and_runner_are_mechanical_derivations_of_run4b(self) -> None:
        for target, text in D.derived_files().items():
            self.assertEqual((D.PACKAGE / target).read_text(), text, target)
        self.assertIs(L.Run4BTrainerV1.update_once, R4B.L.Run4BTrainerV1.update_once)
        for name in ("SEEDS", "WARMUP", "TRANSITIONS_PER_UPDATE", "BATCH", "THREADS",
                     "SMOKE_UPDATE", "FINAL_UPDATE", "RESUME_STOP_UPDATE", "CHECKPOINT_UPDATES",
                     "LIVE_ACTOR", "TRAINER_CONFIG", "GENERATOR_NAMES",
                     "REGISTERED_PROVIDER_BINDING_SHA256"):
            self.assertEqual(getattr(R, name), getattr(R4B, name), name)
        self.assertEqual(L.REPLAY_CAPACITY, R4B.L.REPLAY_CAPACITY)

    def test_one_provider_module_and_one_binding(self) -> None:
        self.assertIs(R.OPL, R4B.OPL)
        self.assertIs(JC.OPL, E4.OPL)
        self.assertIs(JC.OPL, OPL)
        source = (WORKTREE / "rl_agent/splitfusion_operational_latency_v1/provider.py").read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), PROVIDER_COMMIT_SHA256)
        committed = subprocess.run(
            ["git", "show", "056f0fd:rl_agent/splitfusion_operational_latency_v1/provider.py"],
            cwd=WORKTREE, capture_output=True).stdout
        self.assertEqual(committed, source)
        binding = json.loads((WORKTREE / "rl_agent/splitfusion_operational_latency_v1/"
                              "PROVIDER_BINDING.json").read_text())
        self.assertEqual(binding["binding_sha256"], JC.REGISTERED_PROVIDER_BINDING_SHA256)
        self.assertEqual(JC.REGISTERED_PROVIDER_BINDING_SHA256,
                         R4B.REGISTERED_PROVIDER_BINDING_SHA256)
        doc = binding["binding"]
        self.assertEqual(doc["composition"], "L_op = A + S + T + E + D")
        self.assertEqual(doc["label"], "EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION")
        self.assertEqual(doc["deadline_ns_inclusive"], 170_000_000)
        self.assertEqual(set(doc["excluded_components"]), {
            "GT_WAIT", "GT_SCORING", "QPERC_COMPUTATION", "MAP_INSTALLATION",
            "PREDICTION_READY_TO_EVALUATION_ENQUEUE", "OLD_ACTOR_RESERVE"})
        self.assertTrue(OPL.is_timely(transport_success=True, total_ns=170_000_000))
        self.assertFalse(OPL.is_timely(transport_success=True, total_ns=170_000_001))
        self.assertEqual(OPL.compose_total_ns(a_ns=1, s_ns=2, t_ns=3, e_ns=4, d_ns=5), 15)


class _Evidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)
        cls.sources = JC.load_sources(EVIDENCE_ROOT)
        sealed = {"document": {"joint_channel_binding_sha256": cls.sources.binding_sha256},
                  "sha256": FAKE_REGISTRATION}
        cls.registration = mock.patch.object(REG, "load_sealed", return_value=sealed)
        cls.registration.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.registration.stop()

    def runner(self, seed=17):
        return R.Run5BRunnerV1(self.sources, seed)

    def warmup(self, seed=17):
        return R4O.build_frozen_warmup_schedule(seed)


class JointChannelParityTest(_Evidence):
    """Items 2 and 3: bit-identical outcomes; SNR is the only observation difference."""

    def test_sources_are_run4b_registered_sources(self) -> None:
        registered = json.loads((WORKTREE / "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
                                 "RUN4B_REGISTRATION.json").read_text())
        self.assertEqual(self.sources.run4b.binding_document(), registered["environment_binding"])
        self.assertEqual(self.sources.run4b.binding_sha256,
                         registered["environment_binding_sha256"])
        self.assertEqual(self.sources.snr_kernel.base.binding_sha256,
                         self.sources.run4b.mcs_model.binding_sha256)

    def test_joint_step_equals_run4b_step_for_an_identical_context(self) -> None:
        run4b = E4.Run4BEnvironmentV1(self.sources.run4b, seed=29)
        joint = JC.JointChannelEnvironmentV1(self.sources, seed=29)
        schedule = self.warmup(29)
        for k in range(120):
            action = schedule.action_at(k)
            # Give the joint environment Run-4B's exact pre-decision context.
            joint.context = {**copy.deepcopy(run4b.context), "snr_db": joint.context["snr_db"]}
            joint._backlog, joint.prior = run4b._backlog, run4b.prior
            joint._mcs_current = run4b._mcs_current
            joint.decision_seq = run4b.decision_seq
            joint._channel._mcs = run4b._mcs_current
            a = run4b.step(action.mode_id, action.q_e4).diagnostics
            b = joint.step(action.mode_id, action.q_e4).diagnostics
            for key in ("components_ns", "composed_total_ns", "terminal", "reward",
                        "operational_latency_ns", "transport_success", "q_perc_training_only",
                        "prior_ul_mcs", "pre_action_backlog_bytes", "reward_wire_bytes"):
                self.assertEqual(a[key], b[key], (k, key))
            self.assertEqual(run4b._backlog, joint._backlog)
            self.assertEqual(run4b.prior, joint.prior)

    def test_run5b_and_run4b_joint_paths_are_bit_identical(self) -> None:
        """Run-4B-Joint = same environment, 20-D state, SNR never observed."""
        runner = self.runner(17)
        bare = JC.JointChannelEnvironmentV1(self.sources, seed=17)
        schedule = self.warmup(17)
        for k in range(200):
            row, transition = runner.collect_one()
            action = schedule.action_at(k)
            run4b_joint_state = bare.current_features()          # 20-D, no SNR
            other = bare.step(action.mode_id, action.q_e4)
            self.assertEqual((row["mode_id"], row["q_e4"]), (action.mode_id, action.q_e4))
            self.assertEqual(transition.state[:20], run4b_joint_state)
            self.assertEqual(transition.state[:20], other.state)
            self.assertEqual(transition.diagnostics, other.diagnostics)   # MCS, A/S/T/E/D, ...
            self.assertEqual(runner.env.state_dict(), bare.state_dict())  # streams, backlog
        self.assertEqual(runner.env.future_sample_violations, 0)

    def test_snr_absent_from_reward_and_outcome(self) -> None:
        schedule = self.warmup(43)
        for at in (0, 5, 17, 40):
            a = JC.JointChannelEnvironmentV1(self.sources, seed=43)
            b = JC.JointChannelEnvironmentV1(self.sources, seed=43)
            for k in range(at):
                action = schedule.action_at(k)
                a.step(action.mode_id, action.q_e4)
                b.step(action.mode_id, action.q_e4)
            forced = 24.4 if a.context["snr_db"] < 15 else 5.6
            b.context["snr_db"] = b._channel._snr = forced     # same decision, other SNR
            action = schedule.action_at(at)
            da = a.step(action.mode_id, action.q_e4).diagnostics
            db = b.step(action.mode_id, action.q_e4).diagnostics
            self.assertNotEqual(da["snr_db"], db["snr_db"])
            for key in ("components_ns", "composed_total_ns", "terminal", "reward",
                        "operational_latency_ns", "transport_success", "prior_ul_mcs",
                        "pre_action_backlog_bytes"):
                self.assertEqual(da[key], db[key], (at, key))
            self.assertEqual(a._backlog, b._backlog)

    def test_no_future_snr_and_feature_is_the_causal_sample(self) -> None:
        runner = self.runner(29)
        for _ in range(100):
            row, transition = runner.collect_one()
            d = transition.diagnostics
            self.assertEqual(transition.state[20], (d["snr_db"] - 5.5) / 19.0)
            self.assertTrue(d["generated_ticks_after_observed"])
            self.assertEqual(transition.next_state[20], (d["successor_snr_db"] - 5.5) / 19.0)
        names = set(runner.env.context) - {"snr_db"}
        for forbidden in ("profile", "hidden", "trace", "tick", "markov"):
            self.assertFalse(any(forbidden in n for n in names), forbidden)

    def test_qperc_changes_reward_but_never_the_state(self) -> None:
        class Halved:
            def __init__(self, catalog):
                self._c = catalog

            def __getattr__(self, name):
                return getattr(self._c, name)

            def draw(self, key, **kw):
                d = self._c.draw(key, **kw)
                return dataclasses.replace(d, q_perc=d.q_perc * 0.5)

        base, perturbed = self.runner(17), self.runner(17)
        perturbed.env.sources = dataclasses.replace(self.sources.run4b,
                                                    catalog=Halved(self.sources.run4b.catalog))
        changed = 0
        for _ in range(120):
            _, x = base.collect_one()
            _, y = perturbed.collect_one()
            self.assertEqual((x.state, x.next_state), (y.state, y.next_state))
            if x.diagnostics["terminal"] == "TIMELY_SUCCESS" and \
                    x.diagnostics["q_perc_training_only"] > 0:
                self.assertNotEqual(x.reward, y.reward)
                changed += 1
        self.assertGreater(changed, 30)


class RunnerTest(_Evidence):
    def test_preflight_bundle_roundtrip_and_refusals(self) -> None:
        runner = self.runner(17)
        report = runner.run_preflight(on_decision=lambda row: None)
        self.assertTrue(report["passed"], report["checks"])
        for key in ("snr_feature_is_causal_sample", "no_future_snr_tick",
                    "reward_formula_exact", "no_qperc_in_successor_state",
                    "exact_operational_prior_propagation"):
            self.assertIs(report["checks"][key], True, key)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = R.write_bundle(runner, Path(tmp) / "checkpoints")
            restored = R.restore_runner(bundle, self.sources, 17)
            self.assertEqual(restored.fingerprint(), runner.fingerprint())
            _, a = runner.collect_one()
            _, b = restored.collect_one()
            self.assertEqual(a, b)
            manifest = json.loads((bundle / "manifest.json").read_text())
            for edit in ({"feature_order": list(C4.FEATURE_ORDER)},
                         {"model_binding_sha256": M4.MODEL_BINDING_SHA256},
                         {"model_binding_sha256": R5M.RUN5_TRAINING_MODEL_BINDING_SHA256},
                         {"registration_sha256": "1" * 64}):
                forged = Path(tmp) / f"forged_{len(os.listdir(tmp))}"
                forged.mkdir()
                data = json.dumps({**manifest, **edit}, sort_keys=True,
                                  separators=(",", ":")).encode()
                for name in ("training_state.pt", "environment_state.json"):
                    (forged / name).write_bytes((bundle / name).read_bytes())
                (forged / "manifest.json").write_bytes(data)
                (forged / "COMMITTED").write_text(hashlib.sha256(data).hexdigest() + "\n")
                with self.assertRaises((M.CheckpointRefused, R.CheckpointRefused)):
                    R.read_bundle(forged, runner)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
