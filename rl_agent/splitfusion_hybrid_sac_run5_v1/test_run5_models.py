"""CPU-only tests: Run-5 model width and refusal of every 21-D Run-4 checkpoint."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io as R4IO
from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as M


def _pad(state: dict, keys, width: int) -> dict:
    out = {k: v.clone() for k, v in state.items()}
    for key in keys:
        weight = out[key]
        extra = torch.zeros(weight.shape[0], width - weight.shape[1], dtype=weight.dtype)
        out[key] = torch.cat([weight[:, :21], extra, weight[:, 21:]], dim=1)
    return out


class Run5ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.actor, cls.critics = M.build_run5_models(actor_seed=3, critic_seed=4)
        run4 = R4M.build_run4_models(actor_seed=3, critic_seed=4)
        cls.run4_actor, cls.run4_critics = run4.actor, run4.critics

    def test_only_state_dim_differs_from_run4(self) -> None:
        run4 = dataclasses.asdict(R4M.run4_model_config())
        run5 = dataclasses.asdict(M.run5_model_config())
        self.assertEqual(run4.pop("state_dim"), 21)
        self.assertEqual(run5.pop("state_dim"), 22)
        self.assertEqual(run4, run5)

    def test_widths_are_22_and_35(self) -> None:
        self.assertEqual(tuple(self.actor.state_dict()[M.ACTOR_INPUT_KEY].shape), (128, 22))
        for key in M.CRITIC_INPUT_KEYS:
            self.assertEqual(tuple(self.critics.state_dict()[key].shape), (128, 35))

    def test_run5_state_round_trips_strictly(self) -> None:
        actor, critics = M.load_run5_model_state(
            binding=M.RUN5_MODEL_BINDING, actor_state=self.actor.state_dict(),
            critic_state=self.critics.state_dict())
        for key, value in self.actor.state_dict().items():
            self.assertTrue(torch.equal(value, actor.state_dict()[key]), key)
        for key, value in self.critics.state_dict().items():
            self.assertTrue(torch.equal(value, critics.state_dict()[key]), key)

    def test_21d_run4_state_is_refused_under_either_binding(self) -> None:
        for binding in (M.RUN5_MODEL_BINDING, R4M.RUN4_MODEL_BINDING):
            with self.assertRaisesRegex(M.Run5CheckpointRefused, "21-D"):
                M.load_run5_model_state(
                    binding=binding, actor_state=self.run4_actor.state_dict(),
                    critic_state=self.run4_critics.state_dict())

    def test_run4_binding_is_refused_even_with_22d_tensors(self) -> None:
        with self.assertRaisesRegex(M.Run5CheckpointRefused, "Run-4"):
            M.load_run5_model_state(
                binding=R4M.RUN4_MODEL_BINDING, actor_state=self.actor.state_dict(),
                critic_state=self.critics.state_dict())
        relabelled = dict(R4M.RUN4_MODEL_BINDING)
        relabelled["policy_feature_count"] = 22
        with self.assertRaises(M.Run5CheckpointRefused):
            M.refuse_run4_binding(relabelled)
        tampered = dict(M.RUN5_MODEL_BINDING)
        tampered["mode_count"] = 11
        with self.assertRaises(M.Run5CheckpointRefused):
            M.refuse_run4_binding(tampered)
        M.refuse_run4_binding(M.RUN5_MODEL_BINDING)

    def test_zero_padded_run4_weights_are_refused_under_forged_run5_binding(self) -> None:
        actor = _pad(self.run4_actor.state_dict(), [M.ACTOR_INPUT_KEY], 22)
        critics = _pad(self.run4_critics.state_dict(), M.CRITIC_INPUT_KEYS, 35)
        with self.assertRaisesRegex(M.Run5CheckpointRefused, "padded"):
            M.load_run5_model_state(binding=M.RUN5_MODEL_BINDING,
                                    actor_state=actor, critic_state=critics)

    def test_foreign_widths_are_refused(self) -> None:
        actor = _pad(self.run4_actor.state_dict(), [M.ACTOR_INPUT_KEY], 23)
        with self.assertRaisesRegex(M.Run5CheckpointRefused, "not 22"):
            M.load_run5_model_state(binding=M.RUN5_MODEL_BINDING, actor_state=actor,
                                    critic_state=self.critics.state_dict())
        with self.assertRaisesRegex(M.Run5CheckpointRefused, "critic input width 34"):
            M.load_run5_model_state(binding=M.RUN5_MODEL_BINDING,
                                    actor_state=self.actor.state_dict(),
                                    critic_state=self.run4_critics.state_dict())
        with self.assertRaises(M.Run5CheckpointRefused):
            M.load_run5_model_state(binding=M.RUN5_MODEL_BINDING, actor_state={},
                                    critic_state=self.critics.state_dict())

    def test_durable_run4_checkpoint_directory_is_refused_without_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / R4IO.MANIFEST_FILENAME).write_text(json.dumps(
                {"schema_id": R4IO.SCHEMA_ID, "schema_version": R4IO.SCHEMA_VERSION}))
            (root / R4IO.PAYLOAD_FILENAME).write_bytes(b"not a torch file")
            with self.assertRaisesRegex(M.Run5CheckpointRefused, "Run-4"):
                M.refuse_checkpoint_directory(root)
            (root / R4IO.MANIFEST_FILENAME).write_text(json.dumps({"schema_id": "other"}))
            with self.assertRaisesRegex(M.Run5CheckpointRefused, "no durable"):
                M.refuse_checkpoint_directory(root)
            with self.assertRaises(M.Run5CheckpointRefused):
                M.refuse_checkpoint_directory(root / "missing")

    def test_cuda_is_never_initialized(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        for value in self.actor.state_dict().values():
            self.assertEqual(value.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
