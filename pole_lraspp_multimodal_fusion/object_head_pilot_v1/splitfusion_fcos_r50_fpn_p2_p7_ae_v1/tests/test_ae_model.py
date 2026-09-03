"""One CPU-only synthetic check of the AE families themselves.

Covers, for B in {128, 64, 32}: deterministic and RNG-neutral initialization,
exact tensor shapes, keep-mask input behaviour, finite forward/backward, and
optimizer ownership restricted to the AE. No checkpoint, CUDA, dataset, cache,
inference, validation or CARLA is touched.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.ranker import build_ranker
from .. import ae_composition, ae_contract, ae_loss
from ..ae_model import SplitFeatureAE, build_split_feature_ae


def synthetic_c2(frames: int = 1, *, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (frames, contract.SPLIT_CHANNELS, contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def synthetic_importance(frames: int, *, seed: int = 11) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    maps = {}
    for offset, group in enumerate(contract.TEACHER_GROUPS[:3]):  # D, G, S valid
        raw = torch.rand(
            frames,
            contract.SPLIT_HEIGHT,
            contract.SPLIT_WIDTH,
            generator=generator,
            dtype=torch.float32,
        ) + float(offset)
        maps[group] = raw / raw.reshape(frames, -1).sum(dim=1).reshape(frames, 1, 1)
    maps["A"] = None  # an unavailable task map is ignored, not fatal
    return maps


class AeModelChecks(unittest.TestCase):
    def test_families_initialize_shape_mask_and_train_ownership(self) -> None:
        c2 = synthetic_c2(frames=2)
        frame = c2[0]
        seeds = set()

        for bottleneck in ae_contract.AE_BOTTLENECKS:
            with self.subTest(bottleneck=bottleneck):
                # --- deterministic, RNG-neutral construction -----------------
                before = torch.random.get_rng_state()
                autoencoder = build_split_feature_ae(bottleneck)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
                twin = build_split_feature_ae(bottleneck)
                for (name, first), (twin_name, second) in zip(
                    autoencoder.state_dict().items(), twin.state_dict().items()
                ):
                    self.assertEqual(name, twin_name)
                    self.assertTrue(torch.equal(first, second), name)

                self.assertEqual(autoencoder.init_seed, ae_contract.AE_INIT_BASE_SEED + bottleneck)
                self.assertNotIn(autoencoder.init_seed, seeds)  # families separated
                seeds.add(autoencoder.init_seed)
                self.assertEqual(
                    autoencoder.family_id, ae_contract.family_for_bottleneck(bottleneck)
                )

                # --- registered initial values --------------------------------
                projection = autoencoder.project.weight.reshape(
                    bottleneck, contract.SPLIT_CHANNELS
                )
                gram = projection @ projection.t()
                self.assertTrue(
                    torch.allclose(gram, torch.eye(bottleneck), atol=1e-5),
                    "encoder projection rows are not orthonormal",
                )
                self.assertTrue(
                    torch.equal(
                        autoencoder.expand.weight[:, :bottleneck, 0, 0], projection.t()
                    ),
                    "decoder latent weights are not the encoder transpose",
                )
                mask_column = autoencoder.expand.weight[
                    :, autoencoder.mask_channel_index, 0, 0
                ]
                self.assertTrue(torch.equal(mask_column, torch.zeros_like(mask_column)))
                for zero_name in (
                    "latent_context.weight",
                    "latent_context.bias",
                    "spatial_context.weight",
                    "spatial_context.bias",
                    "project.bias",
                    "expand.bias",
                ):
                    value = autoencoder.state_dict()[zero_name]
                    self.assertTrue(torch.equal(value, torch.zeros_like(value)), zero_name)

                # --- exact shapes, unbatched and batched ---------------------
                latent = autoencoder.encode(frame)
                self.assertEqual(
                    tuple(latent.shape),
                    (bottleneck, contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH),
                )
                self.assertEqual(latent.dtype, torch.float32)
                self.assertEqual(tuple(autoencoder.encode(c2).shape), (2, bottleneck, 112, 192))
                reconstructed = autoencoder.decode(latent)
                self.assertEqual(tuple(reconstructed.shape), contract.SPLIT_SHAPE)
                self.assertEqual(
                    tuple(autoencoder.decode(autoencoder.encode(c2)).shape),
                    (2, *contract.SPLIT_SHAPE),
                )

                # At init the residual branches are zero, so the AE is exactly
                # the rank-B orthogonal channel projection -- and therefore not
                # an identity, which is what "q=0 is not identity" means here.
                expected = torch.einsum(
                    "dc,chw->dhw", projection.t() @ projection, frame
                )
                self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-4))
                self.assertFalse(torch.allclose(reconstructed, frame, atol=1e-3))

                # --- keep-mask input behaviour --------------------------------
                sparse_mask = ae_composition.all_keep_mask()
                sparse_mask[:, 96:] = False
                # Mask weights start at zero, so the mask cannot change the
                # output yet; once they are non-zero, it must.
                self.assertTrue(
                    torch.equal(autoencoder.decode(latent, sparse_mask), reconstructed)
                )
                self.assertTrue(
                    torch.equal(
                        autoencoder.decode(latent, ae_composition.all_keep_mask()),
                        reconstructed,
                    )
                )
                with torch.no_grad():
                    autoencoder.expand.weight[:, autoencoder.mask_channel_index, 0, 0].fill_(0.5)
                self.assertFalse(
                    torch.equal(autoencoder.decode(latent, sparse_mask), reconstructed)
                )
                self.assertTrue(
                    torch.allclose(
                        autoencoder.decode(latent, None),
                        autoencoder.decode(latent, ae_composition.all_keep_mask()),
                    ),
                    "a None mask must behave exactly like the q=0 all-one mask",
                )
                for bad in (
                    torch.ones(contract.SPLIT_SPATIAL_SHAPE),  # not boolean
                    torch.ones((64, 64), dtype=torch.bool),  # wrong spatial shape
                ):
                    with self.assertRaises(guards.HybridQPayloadError):
                        autoencoder.decode(latent, bad)
                with self.assertRaises(guards.HybridQPayloadError):
                    autoencoder.encode(frame[:8])  # wrong channel count

                # --- static accounting ---------------------------------------
                complexity = autoencoder.complexity()
                cells = contract.SPLIT_CELLS
                self.assertEqual(complexity.encoder_parameters, 267 * bottleneck)
                self.assertEqual(complexity.decoder_parameters, 256 * (bottleneck + 1) + 256 + 2560)
                self.assertEqual(
                    complexity.total_parameters, autoencoder.parameter_count()
                )
                self.assertEqual(complexity.encoder_macs, cells * bottleneck * 265)
                self.assertEqual(complexity.decoder_macs, cells * 256 * (bottleneck + 10))

                # --- finite forward/backward, AE-only optimizer ---------------
                ranker = build_ranker()
                ranker.eval()
                for parameter in ranker.parameters():
                    parameter.requires_grad_(False)

                optimizer = torch.optim.AdamW(
                    autoencoder.parameters(), lr=contract.LEARNING_RATE
                )
                ae_loss.require_ae_only_optimizer(optimizer, autoencoder)
                ae_loss.require_frozen_companions([ranker])

                selection, keep_mask = ae_composition.detached_hard_mask(
                    ranker.score_cells(frame), 0.50
                )
                self.assertIsNotNone(selection)
                fresh_latent = autoencoder.encode(c2)
                masked = fresh_latent * keep_mask.to(fresh_latent.dtype)
                loss = ae_loss.task_aware_reconstruction_loss(
                    c2, autoencoder.decode(masked, keep_mask), synthetic_importance(2)
                )
                self.assertEqual(loss.valid_groups, ("D", "G", "S"))
                self.assertEqual(loss.excluded_groups["A"], "absent")
                self.assertTrue(torch.isfinite(loss.total))
                optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                for name, parameter in autoencoder.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for name, parameter in ranker.named_parameters():
                    self.assertIsNone(parameter.grad, f"gradient reached the ranker: {name}")
                optimizer.step()
                guards.require_module_parameters_finite(autoencoder, "AE after step")

                # An optimizer that reaches outside the AE must be refused.
                intruder = torch.optim.AdamW(
                    list(autoencoder.parameters()) + [nn.Parameter(torch.zeros(1))],
                    lr=contract.LEARNING_RATE,
                )
                with self.assertRaises(guards.HybridQOwnershipError):
                    ae_loss.require_ae_only_optimizer(intruder, autoencoder)

        # Unregistered families are not constructible at all.
        for bad in (256, 16, 0, -64):
            with self.assertRaises(guards.HybridQConfigError):
                SplitFeatureAE(bad)

        # Fewer than three valid task maps is a hard failure, not a silent mean.
        thin = build_split_feature_ae(32)
        with self.assertRaises(guards.HybridQConfigError):
            ae_loss.task_aware_reconstruction_loss(
                frame,
                thin(frame),
                {"D": synthetic_importance(1)["D"][0], "G": None, "S": None},
            )
        # The training objective never accepts quantized inputs.
        with self.assertRaises(guards.HybridQPayloadError):
            ae_loss.task_aware_reconstruction_loss(
                frame.to(torch.uint8), thin(frame), synthetic_importance(1)
            )
        self.assertEqual(
            [ae_loss.stage_b_q_for_update(index) for index in range(5)],
            [0.00, 0.30, 0.50, 0.70, 0.00],
        )
        for excluded in (0.90, 0.98):
            with self.assertRaises(guards.HybridQConfigError):
                ae_loss.require_optimization_q(excluded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
