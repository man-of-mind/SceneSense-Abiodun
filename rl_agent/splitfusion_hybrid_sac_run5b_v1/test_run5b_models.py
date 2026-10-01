"""CPU-only tests: Run-5B widths and exact identity refusal of Run-4 (21-D) and Run-5 (22-D)."""

from __future__ import annotations

import dataclasses
import json
import os
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import _tree_sha256
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as R5M
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_models as M
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import run5b_state_contract as C

WORKTREE = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(os.environ.get("RUN5_EVIDENCE_ROOT", WORKTREE.parent / "abiodun")).resolve()
PREREG = "7" * 64
RUN4_ACTOR = (EVIDENCE_ROOT / "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
              "20260929_seed43_update10000_actor_export/actor_state_dict.pt")


def plain(value):
    if isinstance(value, (dict, type(M.RUN5B_TRAINING_MODEL_BINDING))):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


class Run5BModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.actor, cls.critics = M.build_run5b_models(actor_seed=3, critic_seed=4)
        # A different init seed: with the same seed an untrained Run-4 actor is
        # bit-identical to an untrained Run-5B actor (same architecture, same width),
        # which is exactly why tensors alone cannot carry the identity.
        run4 = R4M.build_run4_models(actor_seed=5, critic_seed=6)
        cls.run4_actor, cls.run4_critics = run4.actor, run4.critics
        cls.run5_actor, cls.run5_critics = R5M.build_run5_models(actor_seed=3, critic_seed=4)

    def manifest(self, actor=None, **edits):
        actor = self.actor if actor is None else actor
        document = {"model_binding": plain(M.RUN5B_TRAINING_MODEL_BINDING),
                    "model_binding_sha256": M.RUN5B_TRAINING_MODEL_BINDING_SHA256,
                    "feature_schema_id": C.FEATURE_SCHEMA_ID,
                    "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
                    "feature_order": list(C.RUN5B_POLICY_FEATURE_ORDER),
                    "feature_order_sha256": C.FEATURE_ORDER_SHA256,
                    "preregistration_sha256": PREREG,
                    "actor_tree_sha256": _tree_sha256(actor.state_dict())}
        document.update(edits)
        return document

    def load(self, manifest, actor_state, critic_state=None, expected=None):
        return M.require_run5b_checkpoint(manifest=manifest, actor_state=actor_state,
                                          critic_state=critic_state, preregistration_sha256=PREREG,
                                          expected_tree_sha256=expected)

    # -- shape --------------------------------------------------------------
    def test_only_state_dim_differs_from_run4_and_run5(self) -> None:
        run4 = dataclasses.asdict(R4M.run4_model_config())
        run5 = dataclasses.asdict(R5M.run5_model_config())
        run5b = dataclasses.asdict(M.run5b_model_config())
        self.assertEqual((run4.pop("state_dim"), run5.pop("state_dim"), run5b.pop("state_dim")),
                         (21, 22, 21))
        self.assertEqual(run5b, run4)
        self.assertEqual(run5b, run5)

    def test_widths_are_21_and_34(self) -> None:
        self.assertEqual(tuple(self.actor.state_dict()[M.ACTOR_INPUT_KEY].shape), (128, 21))
        for key in M.CRITIC_INPUT_KEYS:
            self.assertEqual(tuple(self.critics.state_dict()[key].shape), (128, 34))

    # -- identity classification ---------------------------------------------
    def test_classifies_the_three_actor_families_exactly(self) -> None:
        self.assertEqual(M.classify_identity(R4M.RUN4_MODEL_BINDING), M.RUN4_IDENTITY)
        self.assertEqual(M.classify_identity(R5M.RUN5_TRAINING_MODEL_BINDING), M.RUN5_IDENTITY)
        self.assertEqual(M.classify_identity(R5M.RUN5_MODEL_BINDING), M.RUN5_IDENTITY)
        self.assertEqual(M.classify_identity(M.RUN5B_TRAINING_MODEL_BINDING), M.RUN5B_IDENTITY)
        self.assertEqual(M.classify_identity(self.manifest()), M.RUN5B_IDENTITY)
        # Equal width, different identity: Run-4 and Run-5B are both 21-D.
        self.assertEqual(R4M.RUN4_MODEL_BINDING["policy_feature_count"],
                         M.RUN5B_TRAINING_MODEL_BINDING["policy_feature_count"])
        self.assertNotEqual(R4M.RUN4_MODEL_BINDING_SHA256, M.RUN5B_TRAINING_MODEL_BINDING_SHA256)

    def test_run4_identity_refused_in_every_form(self) -> None:
        cases = {
            "run4_binding": dict(model_binding_sha256=R4M.RUN4_MODEL_BINDING_SHA256),
            "run4_feature_schema": dict(feature_schema_sha256=R4.FEATURE_SCHEMA_SHA256),
            "run4_order": dict(feature_order=list(R4.POLICY_FEATURE_ORDER)),
            "order_with_qperc": dict(feature_order=[*C.RUN5B_POLICY_FEATURE_ORDER[:17],
                                                    "prev_quality_qperc",
                                                    *C.RUN5B_POLICY_FEATURE_ORDER[17:20]]),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(M.Run5BCheckpointRefused, "Run-4"):
                    self.load(self.manifest(**edit), self.actor.state_dict())

    def test_run5_identity_refused_in_every_form(self) -> None:
        cases = {
            "run5_training_binding": dict(
                model_binding_sha256=R5M.RUN5_TRAINING_MODEL_BINDING_SHA256),
            "run5_contract_binding": dict(model_binding_sha256=R5M.RUN5_MODEL_BINDING_SHA256),
            "run5_v2_schema": dict(feature_schema_sha256=R5SNR.FEATURE_SCHEMA_SHA256),
            "run5_v1_schema": dict(feature_schema_sha256=R5V1.FEATURE_SCHEMA_SHA256),
            "run5_order": dict(feature_order=list(R5V1.RUN5_POLICY_FEATURE_ORDER)),
            "run5_preregistration": dict(preregistration_sha256=M.RUN5_PREREGISTRATION_SHA256),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(M.Run5BCheckpointRefused, "Run-5"):
                    self.load(self.manifest(**edit), self.actor.state_dict())
        with self.assertRaisesRegex(M.Run5BCheckpointRefused, "22-D"):
            self.load(self.manifest(), self.run5_actor.state_dict())

    def test_exact_run5b_fields_are_required(self) -> None:
        order = list(C.RUN5B_POLICY_FEATURE_ORDER)
        cases = {
            "schema_id": dict(feature_schema_id="splitfusion_run5b_policy_features_v0"),
            "order_hash": dict(feature_order_sha256="0" * 64),
            "order_swap": dict(feature_order=[order[1], order[0], *order[2:]]),
            "order_count": dict(feature_order=order[:20]),
            "binding": dict(model_binding_sha256="0" * 64),
            "feature_schema": dict(feature_schema_sha256="0" * 64),
            "preregistration": dict(preregistration_sha256="1" * 64),
            "declared_tree": dict(actor_tree_sha256="0" * 64),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                with self.assertRaises(M.Run5BCheckpointRefused):
                    self.load(self.manifest(**edit), self.actor.state_dict())

    # -- tensors ------------------------------------------------------------
    def test_run5b_state_round_trips_strictly(self) -> None:
        tree = _tree_sha256(self.actor.state_dict())
        actor, critics = self.load(self.manifest(), self.actor.state_dict(),
                                   self.critics.state_dict(), expected=tree)
        for key, value in self.actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor.state_dict()[key]), key)
        for key, value in self.critics.state_dict().items():
            self.assertTrue(torch.equal(value, critics.state_dict()[key]), key)

    def test_21d_run4_tensors_under_a_forged_run5b_manifest_are_refused(self) -> None:
        # Same width; only the tensor-tree identity can tell them apart.
        same_seed = R4M.build_run4_models(actor_seed=3, critic_seed=4).actor
        self.assertEqual(_tree_sha256(same_seed.state_dict()),
                         _tree_sha256(self.actor.state_dict()))
        forged = self.manifest(actor=self.run4_actor)
        with self.assertRaisesRegex(M.Run5BCheckpointRefused, "tensor-tree"):
            self.load(forged, self.run4_actor.state_dict(),
                      expected=_tree_sha256(self.actor.state_dict()))
        with self.assertRaisesRegex(M.Run5BCheckpointRefused, "critic input width 35"):
            self.load(self.manifest(), self.actor.state_dict(), self.run5_critics.state_dict())

    def test_frozen_run4_live_actor_tensors_are_refused(self) -> None:
        binding = json.loads((WORKTREE / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/"
                              "ACTOR_BINDING_V2.json").read_text())["actor"]
        self.assertEqual(binding["boundary_sha256"], M.RUN4_FROZEN_ACTOR_TREE_SHA256)
        self.assertEqual(binding["weights_file_sha256"], M.RUN4_FROZEN_ACTOR_WEIGHTS_SHA256)
        encoder = next(t for t in binding["tensor_inventory"] if t["name"] == M.ACTOR_INPUT_KEY)
        self.assertEqual(encoder["tensor_sha256"], M.RUN4_FROZEN_ACTOR_ENCODER_TENSOR_SHA256)
        if not RUN4_ACTOR.is_file():
            self.skipTest("frozen Run-4 actor export is not under the evidence root")
        state = torch.load(RUN4_ACTOR, map_location="cpu", weights_only=True)
        forged = self.manifest(actor_tree_sha256=_tree_sha256(state))
        with self.assertRaisesRegex(M.Run5BCheckpointRefused, "frozen Run-4"):
            self.load(forged, state)

    def test_sliced_or_padded_run5_actor_is_refused(self) -> None:
        state = {k: v.clone() for k, v in self.run5_actor.state_dict().items()}
        weight = state[M.ACTOR_INPUT_KEY]
        state[M.ACTOR_INPUT_KEY] = torch.cat([weight[:, :17], weight[:, 18:]], dim=1)
        self.assertEqual(tuple(state[M.ACTOR_INPUT_KEY].shape), (128, 21))
        with self.assertRaisesRegex(M.Run5BCheckpointRefused, "tensor-tree"):
            self.load(self.manifest(actor_tree_sha256=None), state,
                      expected=_tree_sha256(self.actor.state_dict()))
        padded = {k: v.clone() for k, v in self.actor.state_dict().items()}
        padded[M.ACTOR_INPUT_KEY] = torch.cat([padded[M.ACTOR_INPUT_KEY],
                                               torch.zeros(128, 1)], dim=1)
        with self.assertRaisesRegex(M.Run5BCheckpointRefused, "22-D"):
            self.load(self.manifest(actor_tree_sha256=None), padded)

    def test_cuda_is_never_initialized(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        for value in self.actor.state_dict().values():
            self.assertEqual(value.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
