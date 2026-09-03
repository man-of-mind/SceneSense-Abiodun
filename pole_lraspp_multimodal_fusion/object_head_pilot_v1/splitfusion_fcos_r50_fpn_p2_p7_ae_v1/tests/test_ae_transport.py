"""One CPU-only synthetic end-to-end check of the AE latent transport.

Covers, at q=0, one registered q and one arbitrary q: the ranker sees the
original FP32 C2, q=0 bypasses the ranker but not the AE, exact keep
count/mask, byte-exact zstd round trip, bounded UINT8 retained error, exact
zero-scatter, correct decoder output shape, per-frame family provenance, and
malformed-payload rejection. No checkpoint, CUDA, dataset, cache, inference,
validation or CARLA is touched.
"""

from __future__ import annotations

import struct
import unittest

import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.continuous_q import quantize_q
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import ZstdWireCodec
from .. import ae_contract, ae_family_dispatch, ae_uint8_transport
from ..ae_model import build_split_feature_ae


BYPASS_Q = 0.00
REGISTERED_Q = 0.50
ARBITRARY_Q = 0.2345
Q_VALUES = (BYPASS_Q, REGISTERED_Q, ARBITRARY_Q)


def synthetic_c2() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260829)
    return torch.randn(
        contract.SPLIT_SHAPE, generator=generator, dtype=torch.float32
    )


def synthetic_scores() -> torch.Tensor:
    # Unique, deterministic ordering; higher row-major cells rank first.
    return torch.arange(contract.SPLIT_CELLS, dtype=torch.float32).reshape(
        contract.SPLIT_SPATIAL_SHAPE
    )


class CountingRanker:
    """Asserts it is handed the original FP32 C2 object, never a latent."""

    def __init__(self, expected_c2: torch.Tensor) -> None:
        self.expected_c2 = expected_c2
        self.calls = 0

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        if c2 is not self.expected_c2:
            raise AssertionError("ranker did not receive the original FP32 C2 object")
        if c2.dtype is not torch.float32 or tuple(c2.shape) != contract.SPLIT_SHAPE:
            raise AssertionError("ranker did not receive the frozen FP32 C2 tensor")
        self.calls += 1
        return synthetic_scores()


def _rebuild_header(data: bytes, index: int, value: int | bytes) -> bytes:
    fields = list(
        struct.unpack(
            ae_uint8_transport.HEADER_FORMAT, data[: ae_uint8_transport.HEADER_BYTES]
        )
    )
    fields[index] = value
    return (
        struct.pack(ae_uint8_transport.HEADER_FORMAT, *fields)
        + data[ae_uint8_transport.HEADER_BYTES :]
    )


class AeTransportChecks(unittest.TestCase):
    def test_end_to_end_transport_provenance_and_fail_closed(self) -> None:
        c2 = synthetic_c2()
        ranker = CountingRanker(c2)
        wire = ZstdWireCodec()
        autoencoders = {
            bottleneck: build_split_feature_ae(bottleneck).bind_checkpoint(
                0x11110000 + bottleneck
            )
            for bottleneck in ae_contract.AE_BOTTLENECKS
        }
        keep_indices_by_q: dict[float, torch.Tensor] = {}

        with torch.no_grad():
            for bottleneck, autoencoder in autoencoders.items():
                latent = autoencoder.encode(c2).detach()
                prepared = ae_uint8_transport.prepare(latent)
                full = latent.reshape(bottleneck, contract.SPLIT_CELLS)
                expected_ranges = torch.stack((full.amin(dim=1), full.amax(dim=1)), dim=1)
                # Ranges come from the complete latent, before any q.
                self.assertTrue(torch.equal(prepared.channel_ranges, expected_ranges))

                for q in Q_VALUES:
                    with self.subTest(bottleneck=bottleneck, q=q):
                        plan = quantize_q(q)
                        calls_before = ranker.calls
                        result = ae_uint8_transport.encode_frame(
                            c2, autoencoder, ranker, q, wire_codec=wire
                        )

                        # q=0 bypasses the ranker; the AE still runs.
                        if q == BYPASS_Q:
                            self.assertEqual(ranker.calls, calls_before)
                            self.assertIsNone(result.selection)
                            self.assertTrue(bool(result.keep_mask.all()))
                        else:
                            self.assertEqual(ranker.calls, calls_before + 1)
                            self.assertIsNotNone(result.selection)

                        # Exact keep count and mask.
                        self.assertEqual(result.plan.keep_count, plan.keep_count)
                        self.assertEqual(int(result.keep_mask.sum()), plan.keep_count)

                        # Byte-exact zstd round trip against independently
                        # rebuilt sparse bytes.
                        expected_sparse = ae_uint8_transport.encode_sparse(
                            prepared,
                            q,
                            result.selection,
                            checkpoint_binding=autoencoder.checkpoint_binding,
                        )
                        restored = ae_uint8_transport.decompress_payload(
                            result.packet, wire_codec=wire
                        )
                        self.assertEqual(restored, expected_sparse.data)
                        self.assertTrue(
                            wire.round_trip_is_exact(expected_sparse.data, result.packet.data)
                        )
                        analytical = ae_uint8_transport.analytical_size(q, bottleneck)
                        self.assertEqual(
                            result.packet.uncompressed_bytes, analytical.total_bytes
                        )
                        self.assertEqual(analytical.value_bytes, plan.keep_count * bottleneck)
                        self.assertEqual(
                            analytical.mask_bytes,
                            0 if q == BYPASS_Q else contract.mask_byte_count(),
                        )
                        self.assertEqual(analytical.range_bytes, bottleneck * 8)

                        parsed = ae_uint8_transport.inspect(restored)
                        self.assertEqual(parsed.header.q_e4, plan.q_e4)
                        self.assertEqual(parsed.family_id, autoencoder.family_id)
                        self.assertEqual(parsed.bottleneck, bottleneck)
                        self.assertEqual(
                            parsed.checkpoint_binding, autoencoder.checkpoint_binding
                        )
                        self.assertEqual(tuple(parsed.values.shape), (plan.keep_count, bottleneck))
                        self.assertTrue(torch.equal(parsed.keep_mask, result.keep_mask))
                        self.assertTrue(torch.equal(parsed.channel_ranges, expected_ranges))

                        # One q-independent ordering serves every AE family.
                        stored = keep_indices_by_q.setdefault(q, parsed.keep_indices)
                        self.assertTrue(torch.equal(stored, parsed.keep_indices))

                        decoded, keep_mask, decoded_q, _ = ae_uint8_transport.decode(
                            result.packet, wire_codec=wire
                        )
                        self.assertEqual(decoded_q, plan.wire_q)
                        self.assertEqual(
                            tuple(decoded.shape), (bottleneck, 112, 192)
                        )

                        # Exact zero-scatter at dropped cells.
                        self.assertTrue(bool((decoded[:, ~keep_mask] == 0.0).all()))

                        # Bounded UINT8 error at retained cells.
                        retained = parsed.keep_indices
                        original = full.index_select(1, retained)
                        recovered = decoded.reshape(
                            bottleneck, contract.SPLIT_CELLS
                        ).index_select(1, retained)
                        error = (recovered - original).abs().amax(dim=1)
                        spans = expected_ranges[:, 1] - expected_ranges[:, 0]
                        magnitude = torch.maximum(
                            expected_ranges[:, 0].abs(), expected_ranges[:, 1].abs()
                        ).clamp_min(1.0)
                        bound = spans / (2.0 * 255.0) + (
                            8.0 * torch.finfo(torch.float32).eps * magnitude
                        )
                        self.assertTrue(bool((error <= bound).all()))

                        # Correct decoder output shape at the tail boundary.
                        reconstructed, reconstructed_q = (
                            ae_uint8_transport.reconstruct_c2(
                                result.packet, autoencoder, wire_codec=wire
                            )
                        )
                        self.assertEqual(tuple(reconstructed.shape), contract.SPLIT_SHAPE)
                        self.assertEqual(reconstructed.dtype, torch.float32)
                        self.assertEqual(reconstructed_q, plan.wire_q)
                        # Channel compression is lossy: q=0 is not an identity.
                        if q == BYPASS_Q:
                            self.assertFalse(torch.allclose(reconstructed, c2, atol=1e-3))

            # --- per-frame family provenance ------------------------------
            ae64 = autoencoders[64]
            packet = ae_uint8_transport.encode_frame(
                c2, ae64, ranker, REGISTERED_Q, wire_codec=wire
            ).packet
            preloaded = ae_family_dispatch.PreloadedAeDecoders(autoencoders.values())
            self.assertIs(preloaded.select_for_packet(packet), ae64)
            identity = ae_family_dispatch.identify_sparse_frame(
                ae_uint8_transport.decompress_payload(packet, wire_codec=wire)
            )
            self.assertEqual(identity.family_name, "AE64")
            self.assertEqual(identity.transported_channels, 64)

            # A packet delayed across a profile switch must not be decoded by
            # whichever AE happens to be selected now.
            for wrong in (autoencoders[128], autoencoders[32]):
                with self.assertRaises(guards.HybridQPayloadError):
                    ae_uint8_transport.reconstruct_c2(packet, wrong, wire_codec=wire)
            rebound = build_split_feature_ae(64).bind_checkpoint(0xDEADBEEF)
            with self.assertRaises(guards.HybridQPayloadError):
                ae_uint8_transport.reconstruct_c2(packet, rebound, wire_codec=wire)
            with self.assertRaises(guards.HybridQPayloadError):
                ae_family_dispatch.PreloadedAeDecoders(
                    [autoencoders[128]]
                ).select_for_packet(packet)

            # --- malformed payload rejection -------------------------------
            good = ae_uint8_transport.decompress_payload(packet, wire_codec=wire)
            plan = quantize_q(REGISTERED_Q)
            range_start = ae_uint8_transport.HEADER_BYTES + contract.mask_byte_count()

            nonfinite_range = bytearray(good)
            struct.pack_into("<f", nonfinite_range, range_start, float("nan"))
            reversed_range = bytearray(good)
            struct.pack_into("<ff", reversed_range, range_start, 1.0, -1.0)
            malformed_mask = bytearray(good)
            malformed_mask[ae_uint8_transport.HEADER_BYTES] ^= 0x80

            malformed = {
                "magic": _rebuild_header(good, 0, b"BAD!"),
                "version": _rebuild_header(good, 1, 2),
                "codec identity": _rebuild_header(good, 2, 99),
                "noAE family id": _rebuild_header(good, 3, ae_contract.AE_FAMILY_NOAE),
                "family/channel disagreement": _rebuild_header(
                    good, 3, ae_contract.AE_FAMILY_AE128
                ),
                "unregistered bottleneck": _rebuild_header(good, 5, 96),
                "spatial dimensions": _rebuild_header(good, 6, 111),
                "q": _rebuild_header(good, 8, 9801),
                "keep count": _rebuild_header(good, 9, plan.keep_count + 1),
                "mask length": _rebuild_header(good, 10, contract.mask_byte_count() + 1),
                "range length": _rebuild_header(good, 11, 508),
                "value length": _rebuild_header(good, 12, plan.keep_count * 64 + 1),
                "nonfinite range": bytes(nonfinite_range),
                "min greater than max": bytes(reversed_range),
                "malformed mask": bytes(malformed_mask),
                "trailing bytes": good + b"\x00",
                "truncated payload": good[:-1],
            }
            for name, bad_sparse in malformed.items():
                with self.subTest(malformed=name):
                    compressed = wire.compress(bad_sparse)
                    bad_packet = ae_uint8_transport.AeZstdPacket(
                        data=compressed.data,
                        uncompressed_bytes=compressed.uncompressed_bytes,
                        family_id=packet.family_id,
                        checkpoint_binding=packet.checkpoint_binding,
                        bottleneck=packet.bottleneck,
                    )
                    with self.assertRaises(guards.HybridQError):
                        ae_uint8_transport.reconstruct_c2(
                            bad_packet, ae64, wire_codec=wire
                        )

            # A header binding that disagrees with the envelope is refused even
            # though both are individually well formed.
            rebound_header = _rebuild_header(good, 4, packet.checkpoint_binding + 1)
            compressed = wire.compress(rebound_header)
            with self.assertRaises(guards.HybridQPayloadError):
                ae_uint8_transport.decode(
                    ae_uint8_transport.AeZstdPacket(
                        data=compressed.data,
                        uncompressed_bytes=compressed.uncompressed_bytes,
                        family_id=packet.family_id,
                        checkpoint_binding=packet.checkpoint_binding,
                        bottleneck=packet.bottleneck,
                    ),
                    wire_codec=wire,
                )

            # The deployment decoder accepts only the mandatory zstd packet.
            with self.assertRaises(guards.HybridQPayloadError):
                ae_uint8_transport.decode(good)  # type: ignore[arg-type]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
