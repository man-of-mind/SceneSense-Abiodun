"""Phase-1 CPU-only tests: restored seed-43/update-10,000 actor, frozen export.

These tests read the pinned Run-4 sources and the untracked export written by
``export_frozen_actor_v2``.  They launch no CARLA, OAI, Docker or network
process and assert that CUDA is never initialized.  Run with
``CUDA_VISIBLE_DEVICES=`` and without ``PYTHONPATH``.
"""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_SCALE,
    round_half_up_q_e4,
)

from . import frozen_actor_v2 as FA


def _export_dir() -> Path:
    return FA.REPOSITORY_ROOT / FA.ACTOR_EXPORT_RELPATH


def _manifest() -> dict:
    return FA.load_json(FA.TRACKED_BINDING_PATH)


def _weights() -> Path:
    return _export_dir() / _manifest()["actor"]["weights_file_name"]


@unittest.skipUnless(
    FA.TRACKED_BINDING_PATH.is_file()
    and (FA.REPOSITORY_ROOT / FA.ACTOR_EXPORT_RELPATH).is_dir(),
    "actor export not present on this host",
)
class FrozenActorExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.actor = FA.load_registered_actor()

    # -- identity -----------------------------------------------------------
    def test_pinned_sources_reverify(self) -> None:
        self.assertEqual(len(FA.verify_pinned_sources()), 5)

    def test_restored_actor_equals_checkpoint_boundary(self) -> None:
        self.assertEqual(
            FA.actor_boundary_sha256(self.actor.module),
            "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3",
        )
        manifest = _manifest()
        self.assertEqual(manifest["selected"]["seed"], 43)
        self.assertEqual(manifest["selected"]["update"], 10_000)
        self.assertEqual(manifest["actor"]["restored_update_count"], 10_000)
        self.assertEqual(manifest["actor"]["restored_decision_count"], 40_288)

    def test_restore_result_records_bit_identical_restore(self) -> None:
        result = FA.load_json(_export_dir() / "RESTORE_RESULT.json")
        self.assertEqual(result["verdict"], "RESTORED_BIT_IDENTICAL_AND_EXPORTED")
        self.assertEqual(result["training_steps_after_restore"], 0)
        self.assertIs(result["cuda_initialized"], False)
        self.assertIs(result["reload_exact_tensor_equality"], True)
        self.assertEqual(
            result["restored_boundary"]["actor_sha256"],
            FA.SELECTED.actor_boundary_sha256,
        )
        self.assertEqual(result["manifest_sha256"], _manifest()["manifest_sha256"])

    def test_tracked_binding_is_byte_copy_of_export(self) -> None:
        self.assertEqual(
            FA.TRACKED_BINDING_PATH.read_bytes(),
            (_export_dir() / "ACTOR_EXPORT_MANIFEST.json").read_bytes(),
        )

    # -- exact reload -------------------------------------------------------
    def test_weights_only_reload_is_tensor_exact(self) -> None:
        state = torch.load(_weights(), map_location="cpu", weights_only=True)
        live = self.actor.module.state_dict()
        self.assertEqual(set(state), set(live))
        for name in state:
            self.assertEqual(state[name].dtype, live[name].dtype, name)
            self.assertTrue(torch.equal(state[name], live[name]), name)
        self.assertEqual(FA.tensor_inventory(state),
                         _manifest()["actor"]["tensor_inventory"])

    def test_fixture_outputs_equal_restored_actor(self) -> None:
        self.assertEqual(FA.fixture_outputs(self.actor.module),
                         _manifest()["fixtures"])

    def test_batch1_output_is_repeatable(self) -> None:
        for values in FA.registered_fixture_states():
            first = self.actor.act_on_vector(values)
            second = self.actor.act_on_vector(values)
            self.assertEqual(first, second)

    def test_deployment_rule_matches_independent_recomputation(self) -> None:
        module = self.actor.module
        lower, upper = module.active_q_e4_bounds()
        for values in FA.registered_fixture_states():
            decision = self.actor.act_on_vector(values)
            with torch.inference_mode():
                heads = module(torch.tensor((values,), dtype=torch.float32))
            mode = int(torch.argmax(heads.logits[0]))
            mean = heads.mean[0, mode]
            z = 0.5 * (torch.tanh(mean) + 1.0)
            low = lower[mode].to(torch.float32) / float(Q_E4_SCALE)
            width = (upper[mode] - lower[mode]).to(torch.float32) / float(Q_E4_SCALE)
            q = float(low + width * z)
            self.assertEqual(decision.mode_id, mode)
            self.assertEqual(decision.q_e4, round_half_up_q_e4(q))
            self.assertGreaterEqual(decision.q_e4, int(lower[mode]))
            self.assertLessEqual(decision.q_e4, int(upper[mode]))

    # -- frozen / CPU / RNG ---------------------------------------------------
    def test_parameters_frozen_eval_cpu_float32(self) -> None:
        module = self.actor.module
        self.assertFalse(module.training)
        for name, value in module.named_parameters():
            self.assertFalse(value.requires_grad, name)
            self.assertEqual(value.device.type, "cpu", name)
            self.assertIs(value.dtype, torch.float32, name)
        self.assertFalse(torch.cuda.is_initialized())

    def test_train_mode_is_refused(self) -> None:
        module = self.actor.module
        module.train()
        try:
            with self.assertRaises(FA.FrozenActorError):
                self.actor.act_on_vector(FA.registered_fixture_states()[0])
        finally:
            module.eval()

    def test_global_rng_neutral(self) -> None:
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        before = FA.rng_fingerprint()
        FA.load_registered_actor()
        for values in FA.registered_fixture_states():
            self.actor.act_on_vector(values)
        self.assertEqual(FA.rng_fingerprint(), before)
        self.assertFalse(torch.cuda.is_initialized())

    # -- input validation -----------------------------------------------------
    def test_rejects_malformed_vectors(self) -> None:
        good = FA.registered_fixture_states()[0]
        for bad in (
            list(good),
            good[:-1],
            good + (0.0,),
            good[:3] + (math.nan,) + good[4:],
            good[:3] + (math.inf,) + good[4:],
            (True,) + good[1:],
        ):
            with self.assertRaises(FA.FrozenActorError):
                self.actor.act_on_vector(bad)

    def test_act_requires_attested_feature_vector(self) -> None:
        with self.assertRaises(FA.FrozenActorError):
            self.actor.act(FA.registered_fixture_states()[0])
        unattested = contract.PolicyFeatureVectorV2(
            values=FA.registered_fixture_states()[0],
            guarded_state_sha256="0" * 64,
            empirical_scaling_sha256="0" * 64,
        )
        with self.assertRaises(contract.ScalingError):
            self.actor.act(unattested)

    # -- tamper matrix --------------------------------------------------------
    def _reject(self, manifest: dict, weights: Path | None = None) -> None:
        with self.assertRaises(FA.FrozenActorError):
            FA.load_frozen_actor(weights or _weights(), manifest)

    def test_unsealed_tamper_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["fixtures"][0]["q_e4"] += 1
        self._reject(manifest)

    def test_resealed_fixture_tamper_is_rejected(self) -> None:
        manifest = _manifest()
        manifest.pop("manifest_sha256")
        manifest["fixtures"][0]["q_e4"] = (manifest["fixtures"][0]["q_e4"] + 1) % 9801
        self._reject(FA.seal_manifest(manifest))

    def test_wrong_seed_or_update_is_rejected(self) -> None:
        for field, value in (("seed", 17), ("seed", 29), ("update", 1500)):
            manifest = _manifest()
            manifest.pop("manifest_sha256")
            manifest["selected"][field] = value
            self._reject(FA.seal_manifest(manifest))

    def test_wrong_feature_order_is_rejected(self) -> None:
        manifest = _manifest()
        manifest.pop("manifest_sha256")
        order = manifest["binding"]["policy_feature_order"]
        order[2], order[3] = order[3], order[2]
        self._reject(FA.seal_manifest(manifest))

    def test_wrong_actor_digest_is_rejected(self) -> None:
        manifest = _manifest()
        manifest.pop("manifest_sha256")
        manifest["actor"]["boundary_sha256"] = "0" * 64
        self._reject(FA.seal_manifest(manifest))

    def test_altered_tensor_is_rejected(self) -> None:
        state = torch.load(_weights(), map_location="cpu", weights_only=True)
        name = "encoder.0.weight"
        state[name] = state[name].clone()
        state[name][0, 0] = torch.nextafter(
            state[name][0, 0], torch.tensor(math.inf))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.pt"
            torch.save(state, path)
            # (a) file digest no longer matches.
            self._reject(_manifest(), path)
            # (b) digest and inventory re-pinned: the boundary digest refuses.
            manifest = _manifest()
            manifest.pop("manifest_sha256")
            manifest["actor"]["weights_file_sha256"] = FA.sha256_file(path)
            manifest["actor"]["tensor_inventory"] = FA.tensor_inventory(state)
            self._reject(FA.seal_manifest(manifest), path)

    def test_feature_contract_is_unchanged(self) -> None:
        self.assertEqual(len(contract.POLICY_FEATURE_ORDER), 21)
        self.assertEqual(
            _manifest()["binding"]["policy_feature_order"],
            list(contract.POLICY_FEATURE_ORDER),
        )
        self.assertEqual(_manifest()["binding"]["mode_count"], EXPECTED_MODE_COUNT)


class FixtureGridTest(unittest.TestCase):
    def test_fixtures_are_legal_contract_encodings(self) -> None:
        fixtures = FA.registered_fixture_states()
        self.assertEqual(len(fixtures), 20)
        self.assertEqual(fixtures, FA.registered_fixture_states())
        for values in fixtures:
            named = dict(zip(contract.POLICY_FEATURE_ORDER, values))
            one_hot = [named[f"prev_joint_mode_{m}_one_hot"] for m in range(12)]
            if named["prev_present"] == 0.0:
                self.assertEqual(sum(one_hot), 0.0)
                self.assertEqual(named["prev_success"], 0.0)
            else:
                self.assertEqual(sum(one_hot), 1.0)
                if named["prev_success"] == 0.0:
                    self.assertEqual(named["prev_quality_qperc"], 0.0)
                    self.assertEqual(named["prev_latency_normalized"], 0.0)
            self.assertGreaterEqual(named["pre_action_rlc_backlog_log1p_scaled"], 0.0)
            self.assertLessEqual(named["pre_action_rlc_backlog_log1p_scaled"], 1.0)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    unittest.main()
