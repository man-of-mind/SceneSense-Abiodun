"""One CPU-only synthetic check of the AE families themselves.

Covers, for B in {128, 64, 32}: deterministic and RNG-neutral initialization,
exact tensor shapes, keep-mask input behaviour, batched training composition
with independent per-frame selection, finite forward/backward, the Phase-4
teacher-cache loss interface, and optimizer ownership restricted to the AE. No
checkpoint, CUDA, dataset, cache, inference, validation or CARLA is touched, and
no teacher cache is built or rebuilt.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.ranker import build_ranker
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.teacher_cache import SHARD_SCHEMA
from .. import ae_composition, ae_contract, ae_loss
from ..ae_model import SplitFeatureAE, build_split_feature_ae


def synthetic_c2(frames: int = 1, *, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (frames, contract.SPLIT_CHANNELS, contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def synthetic_shard(frames: int, *, valid=("D", "G", "S"), seed: int = 11) -> dict:
    """A shard payload with exactly the fields the Phase-4 cache actually stores.

    One combined FP32 importance map per frame plus `valid_groups` /
    `excluded_groups`; there are deliberately no separate D/G/S/A maps, because
    the real cache has none. Nothing is written to disk.
    """
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(
        frames,
        contract.SPLIT_HEIGHT,
        contract.SPLIT_WIDTH,
        generator=generator,
        dtype=torch.float32,
    )
    combined = raw / raw.reshape(frames, -1).sum(dim=1).reshape(frames, 1, 1)
    excluded = {
        group: "zero_gradient" for group in contract.TEACHER_GROUPS if group not in valid
    }
    return {
        "schema": SHARD_SCHEMA,
        "shard_index": 0,
        "frames": frames,
        "split": "fit",
        "sample_ids": [f"synthetic_{index:04d}" for index in range(frames)],
        "splits": ["fit"] * frames,
        "importance": combined,
        "valid_groups": [list(valid) for _ in range(frames)],
        "excluded_groups": [dict(excluded) for _ in range(frames)],
    }


class SeparatedScoreRanker:
    """Batched stub whose frames occupy disjoint, ordered score ranges.

    Frame 1 outscores every cell of frame 0, so a single global top-K over the
    flattened batch would starve frame 0 entirely. Per-frame selection must
    still give both frames the same keep count.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, c2: torch.Tensor) -> torch.Tensor:
        if c2.dim() != 4:
            raise AssertionError("batched ranker did not receive a batched C2")
        self.calls += 1
        frames = int(c2.shape[0])
        base = torch.arange(contract.SPLIT_CELLS, dtype=torch.float32)
        offsets = torch.arange(frames, dtype=torch.float32) * 1.0e6
        return (base.reshape(1, -1) + offsets.reshape(-1, 1)).reshape(
            frames, *contract.SPLIT_SPATIAL_SHAPE
        )


class AeModelChecks(unittest.TestCase):
    def test_families_initialize_shape_mask_batch_and_train_ownership(self) -> None:
        c2 = synthetic_c2(frames=2)
        frame = c2[0]
        teacher = ae_loss.CachedTeacherBatch.from_shard(synthetic_shard(2))
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
                # A fresh AE is unbound and stays unusable on deployable paths.
                self.assertFalse(autoencoder.is_bound)
                with self.assertRaises(guards.HybridQConfigError):
                    autoencoder.bind_routing_tag(ae_contract.AE_UNBOUND_ROUTING_TAG)

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

                # --- batched training composition -----------------------------
                ranker = build_ranker()
                ranker.eval()
                for parameter in ranker.parameters():
                    parameter.requires_grad_(False)

                bypass = ae_composition.compose_batch(c2, autoencoder, ranker, 0.00)
                self.assertIsNone(bypass.selections)
                self.assertEqual(
                    tuple(bypass.keep_mask.shape), (2, *contract.SPLIT_SPATIAL_SHAPE)
                )
                self.assertTrue(bool(bypass.keep_mask.all()))
                self.assertTrue(torch.equal(bypass.masked_latent, bypass.latent))

                stub = SeparatedScoreRanker()
                batched = ae_composition.compose_batch(c2, autoencoder, stub, 0.50)
                self.assertEqual(stub.calls, 1)  # one batched ranker pass
                self.assertEqual(len(batched.selections), 2)
                keep = contract.keep_count(0.50)
                # Independent per-frame top-K: a single global top-K over the
                # flattened batch would have left frame 0 with zero cells.
                counts = batched.keep_mask.reshape(2, -1).sum(dim=1)
                self.assertEqual([int(value) for value in counts], [keep, keep])
                self.assertEqual(
                    tuple(batched.masked_latent.shape), (2, bottleneck, 112, 192)
                )
                for index in range(2):
                    dropped = ~batched.keep_mask[index]
                    self.assertTrue(
                        bool((batched.masked_latent[index][:, dropped] == 0.0).all())
                    )
                    self.assertTrue(
                        torch.equal(
                            batched.masked_latent[index][:, batched.keep_mask[index]],
                            batched.latent[index][:, batched.keep_mask[index]],
                        )
                    )
                self.assertFalse(batched.keep_mask.requires_grad)
                with self.assertRaises(guards.HybridQConfigError):
                    ae_composition.compose_batch(c2, autoencoder, stub, 0.90)
                with self.assertRaises(guards.HybridQPayloadError):
                    ae_composition.compose_batch(frame, autoencoder, stub, 0.50)

                # --- finite forward/backward, AE-only optimizer ---------------
                optimizer = torch.optim.AdamW(
                    autoencoder.parameters(), lr=contract.LEARNING_RATE
                )
                ae_loss.require_ae_only_optimizer(optimizer, autoencoder)
                ae_loss.require_frozen_companions([ranker])

                trained = ae_composition.compose_batch(
                    c2, autoencoder, ranker, ae_loss.stage_b_q_for_update(2)
                )
                self.assertEqual(trained.plan.wire_q, 0.50)
                loss = ae_loss.task_aware_reconstruction_loss(
                    c2,
                    autoencoder.decode(trained.masked_latent, trained.keep_mask),
                    teacher,
                )
                self.assertEqual(loss.frames, 2)
                self.assertEqual(loss.min_valid_groups_observed, 3)
                self.assertEqual(
                    loss.group_availability, {"D": 2, "G": 2, "S": 2, "A": 0}
                )
                self.assertEqual(loss.excluded_groups, {"A": {"zero_gradient": 2}})
                self.assertNotIn("group_terms", loss.report())
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

        # --- Phase-4 teacher-cache interface --------------------------------
        thin = build_split_feature_ae(32)
        shard = synthetic_shard(4)
        sliced = ae_loss.CachedTeacherBatch.from_shard(shard, offsets=[1, 3])
        self.assertEqual(sliced.frames, 2)
        self.assertTrue(
            torch.equal(sliced.importance, shard["importance"].index_select(
                0, torch.tensor([1, 3])
            ))
        )
        # A shard without the stored fields, or with fabricated per-group maps,
        # is not a substitute for the real representation.
        for missing in ("importance", "valid_groups", "excluded_groups"):
            broken = {key: value for key, value in shard.items() if key != missing}
            with self.assertRaises(guards.HybridQPayloadError):
                ae_loss.CachedTeacherBatch.from_shard(broken)
        with self.assertRaises(guards.HybridQConfigError):
            ae_loss.CachedTeacherBatch.from_shard(
                {**shard, "valid_groups": [["D", "G", "X"]] * 4}
            )

        # Fewer than three valid groups on any frame is a hard failure.
        with self.assertRaises(guards.HybridQConfigError):
            ae_loss.task_aware_reconstruction_loss(
                c2,
                thin(c2),
                ae_loss.CachedTeacherBatch.from_shard(
                    synthetic_shard(2, valid=("D", "G"))
                ),
            )
        # Raw maps are not accepted: the cache stores one combined map.
        with self.assertRaises(guards.HybridQConfigError):
            ae_loss.task_aware_reconstruction_loss(
                c2, thin(c2), {"D": shard["importance"]}
            )
        # The training objective never accepts quantized inputs.
        with self.assertRaises(guards.HybridQPayloadError):
            ae_loss.task_aware_reconstruction_loss(
                c2.to(torch.uint8), thin(c2), teacher
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
