from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import (
    frozen_actor_loader_v1 as F,
)


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
ARTIFACT_RELPATH = Path(
    "rl_agent/experiments/splitfusion_hybrid_sac_run4b_v1/"
    "20261001_a589250/campaign/seed_43/final_actor/actor_state_dict.pt"
)


def actual_weights() -> Path:
    candidates = (REPO / ARTIFACT_RELPATH,
                  REPO.parent / "abiodun" / ARTIFACT_RELPATH)
    for path in candidates:
        if path.is_file():
            return path
    raise unittest.SkipTest("the gitignored Run-4B actor artifact is absent")


def rng_snapshot() -> tuple[object, tuple, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


class FrozenActorLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest_path = PACKAGE / "RUN4B_ACTOR_ARTIFACT_V1.json"
        self.manifest = F.load_manifest(self.manifest_path)

    def test_manifest_is_the_selected_run4b_actor(self) -> None:
        m = self.manifest
        self.assertIs(m.variant, F.ActorVariant.RUN4B)
        self.assertEqual(m.feature_order, F.RUN4B_FEATURE_ORDER)
        self.assertEqual(len(m.feature_order), 20)
        self.assertEqual(m.selected_seed, 43)
        self.assertEqual(m.selected_update, 10_000)
        self.assertEqual(
            m.weights_file_sha256,
            "b4337f000be9acbd16113d0891ed40fdac28c4f0bcb31c5b970a52d901c2ebc2",
        )
        self.assertEqual(
            m.operational_latency_provider_sha256,
            "3cc1e6e36ae6e4f43dc8110077c62efbf5c719aaa179935b0f6de4c5bb1f6c29",
        )
        self.assertEqual(m.actor_boundary_sha256, m.actor_tree_sha256)
        self.assertEqual(F.load_manifest(self.manifest_path).as_dict(),
                         json.loads(self.manifest_path.read_text()))

    def test_loads_actual_export_on_cpu_and_is_frozen(self) -> None:
        actor = F.load_frozen_actor(
            actual_weights(), self.manifest,
            expected_variant=F.ActorVariant.RUN4B,
        )
        self.assertFalse(actor.module.training)
        self.assertTrue(all(not p.requires_grad for p in actor.module.parameters()))
        self.assertTrue(all(p.device.type == "cpu"
                            for p in actor.module.parameters()))
        self.assertEqual(actor.module.encoder[0].in_features, 20)

    def test_registered_batch1_decisions(self) -> None:
        actor = F.load_frozen_actor(
            actual_weights(), self.manifest,
            expected_variant=F.ActorVariant.RUN4B,
        )
        probes = (
            ((0.0,) * 20, (11, 5106)),
            (tuple(i / 20 for i in range(20)), (11, 6153)),
            (tuple(-0.5 + i / 19 for i in range(20)), (9, 7352)),
        )
        for state, expected in probes:
            decision = actor.act_on_vector(state)
            self.assertEqual((decision.mode_id, decision.q_e4), expected)
            self.assertEqual(decision.actor_boundary_sha256,
                             self.manifest.actor_boundary_sha256)

    def test_load_and_inference_preserve_global_rngs(self) -> None:
        random.seed(111)
        np.random.seed(222)
        torch.manual_seed(333)
        before = rng_snapshot()
        actor = F.load_frozen_actor(
            actual_weights(), self.manifest,
            expected_variant=F.ActorVariant.RUN4B,
        )
        actor.act_on_vector((0.0,) * 20)
        after = rng_snapshot()
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[1][0], after[1][0])
        np.testing.assert_array_equal(before[1][1], after[1][1])
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_refuses_run5b_expectation(self) -> None:
        with self.assertRaisesRegex(F.FrozenActorLoadError, "variant differs"):
            F.load_frozen_actor(
                actual_weights(), self.manifest,
                expected_variant=F.ActorVariant.RUN5B,
            )

    def test_refuses_wrong_feature_name_and_width(self) -> None:
        raw = json.loads(self.manifest_path.read_text())
        raw["feature_order"][-1] = "prev_success"
        with self.assertRaisesRegex(F.FrozenActorLoadError, "feature order"):
            F.FrozenActorArtifactManifestV1.from_mapping(raw)
        raw = json.loads(self.manifest_path.read_text())
        raw["feature_order"].append("effective_external_ul_snr_proxy_scaled")
        with self.assertRaisesRegex(F.FrozenActorLoadError, "feature order"):
            F.FrozenActorArtifactManifestV1.from_mapping(raw)

    def test_refuses_foreign_manifest_fields(self) -> None:
        raw = json.loads(self.manifest_path.read_text())
        raw["unexpected"] = True
        with self.assertRaisesRegex(F.FrozenActorLoadError, "foreign"):
            F.FrozenActorArtifactManifestV1.from_mapping(raw)

    def test_refuses_tampered_weights_before_model_load(self) -> None:
        state = torch.load(actual_weights(), map_location="cpu", weights_only=True)
        state = {name: value.clone() for name, value in state.items()}
        state["logit_head.bias"][0] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor_state_dict.pt"
            torch.save(state, path)
            with self.assertRaisesRegex(F.FrozenActorLoadError, "file hash differs"):
                F.load_frozen_actor(
                    path, self.manifest,
                    expected_variant=F.ActorVariant.RUN4B,
                )

    def test_refuses_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor_state_dict.pt"
            path.symlink_to(actual_weights())
            with self.assertRaisesRegex(F.FrozenActorLoadError, "non-symlink"):
                F.load_frozen_actor(
                    path, self.manifest,
                    expected_variant=F.ActorVariant.RUN4B,
                )

    def test_input_requires_exact_finite_tuple(self) -> None:
        actor = F.load_frozen_actor(
            actual_weights(), self.manifest,
            expected_variant=F.ActorVariant.RUN4B,
        )
        for bad in ([0.0] * 20, (0.0,) * 19, (0.0,) * 19 + (float("nan"),)):
            with self.assertRaises(F.FrozenActorLoadError):
                actor.act_on_vector(bad)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
