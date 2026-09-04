"""Two focused CPU checks for the shared UINT6/UINT4 transport."""

from __future__ import annotations

import struct
import unittest

import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import ZstdWireCodec
from .. import ae_contract, lowbit_dispatch, lowbit_transport
from ..ae_model import build_split_feature_ae


class CountingWireCodec(ZstdWireCodec):
    def __init__(self) -> None:
        super().__init__()
        self.decompressions = 0

    def decompress_bytes(self, frame: bytes) -> bytes:
        self.decompressions += 1
        return super().decompress_bytes(frame)


class Ranker:
    def __init__(self, expected: torch.Tensor) -> None:
        self.expected = expected
        self.calls = 0

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        if c2 is not self.expected:
            raise AssertionError("ranker did not receive original C2")
        self.calls += 1
        return torch.arange(contract.SPLIT_CELLS, dtype=torch.float32).reshape(
            contract.SPLIT_SPATIAL_SHAPE
        )


def feature() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260903)
    return torch.randn(contract.SPLIT_SHAPE, generator=generator)


class LowBitTransportChecks(unittest.TestCase):
    def test_fixed_order_bit_packing_and_padding(self) -> None:
        cases = {
            4: (torch.tensor([0, 1, 14, 15, 7], dtype=torch.uint8), bytes([0x01, 0xEF, 0x70])),
            6: (
                torch.tensor([0, 1, 62, 63, 42], dtype=torch.uint8),
                bytes([0x00, 0x1F, 0xBF, 0xA8]),
            ),
        }
        for bits, (codes, expected) in cases.items():
            with self.subTest(bits=bits):
                packed = lowbit_transport._pack_codes(codes, bits)
                self.assertEqual(packed, expected)
                self.assertTrue(
                    torch.equal(
                        lowbit_transport._unpack_codes(packed, codes.numel(), bits),
                        codes,
                    )
                )
                corrupted = packed[:-1] + bytes([packed[-1] | 1])
                with self.assertRaises(guards.HybridQPayloadError):
                    lowbit_transport._unpack_codes(corrupted, codes.numel(), bits)

    def test_noae_and_ae_use_one_wire_and_dispatch_from_bytes(self) -> None:
        c2 = feature()
        ranker = Ranker(c2)
        ae32 = build_split_feature_ae(32).bind_routing_tag(0x1234ABCD)
        decoders = lowbit_dispatch.PreloadedLowBitDecoders([ae32])
        wire = CountingWireCodec()

        # Every family uses the same accounting rule; q=0 carries no mask.
        for family_id in ae_contract.AE_FAMILY_IDS:
            for bits in lowbit_transport.SUPPORTED_BIT_WIDTHS:
                dense = lowbit_transport.analytical_size(0.0, family_id, bits)
                sparse = lowbit_transport.analytical_size(0.5, family_id, bits)
                channels = (
                    contract.SPLIT_CHANNELS
                    if family_id == ae_contract.AE_FAMILY_NOAE
                    else ae_contract.bottleneck_for_family(family_id)
                )
                self.assertEqual(dense.mask_bytes, 0)
                self.assertEqual(sparse.mask_bytes, contract.mask_byte_count())
                self.assertEqual(
                    dense.value_bytes,
                    lowbit_transport.packed_value_byte_count(
                        contract.SPLIT_CELLS * channels, bits
                    ),
                )

        with torch.no_grad():
            for bits in lowbit_transport.SUPPORTED_BIT_WIDTHS:
                for family_name, q in (("noAE", 0.0), ("noAE", 0.5), ("AE32", 0.5)):
                    with self.subTest(bits=bits, family=family_name, q=q):
                        calls = ranker.calls
                        if family_name == "noAE":
                            transport = lowbit_transport.encode_noae_frame(
                                c2, ranker, q, bits, wire_codec=wire
                            )
                        else:
                            transport = lowbit_transport.encode_ae_frame(
                                c2, ae32, ranker, q, bits, wire_codec=wire
                            )
                        self.assertEqual(ranker.calls, calls + (q != 0.0))

                        wire.decompressions = 0
                        received = decoders.receive(
                            transport.packet.data,
                            wire_codec=wire,
                            expected_packet=transport.packet,
                            diagnostics=True,
                        )
                        self.assertEqual(wire.decompressions, 1)
                        self.assertEqual(received.family.family_name, family_name)
                        self.assertEqual(received.family.bit_width, bits)
                        self.assertEqual(received.keep_count, transport.plan.keep_count)
                        self.assertEqual(tuple(received.c2.shape), contract.SPLIT_SHAPE)
                        self.assertTrue(bool(torch.isfinite(received.c2).all()))
                        self.assertIsNotNone(received.diagnostics)
                        self.assertTrue(
                            bool(
                                (
                                    received.diagnostics.decoded_feature[
                                        :, ~received.diagnostics.keep_mask
                                    ]
                                    == 0.0
                                ).all()
                            )
                        )
                        if family_name == "noAE" and q == 0.0:
                            recovered = received.diagnostics.decoded_feature
                            flat = c2.reshape(contract.SPLIT_CHANNELS, -1)
                            spans = flat.amax(dim=1) - flat.amin(dim=1)
                            magnitude = torch.maximum(
                                flat.amin(dim=1).abs(), flat.amax(dim=1).abs()
                            ).clamp_min(1.0)
                            bound = spans / (2.0 * ((1 << bits) - 1)) + (
                                8.0 * torch.finfo(torch.float32).eps * magnitude
                            )
                            error = (recovered - c2).abs().reshape(
                                contract.SPLIT_CHANNELS, -1
                            ).amax(dim=1)
                            self.assertTrue(bool((error <= bound).all()))

                # Header corruption is rejected before decoder selection.
                packet = lowbit_transport.encode_ae_frame(
                    c2, ae32, ranker, 0.5, bits, wire_codec=wire
                ).packet
                sparse = wire.decompress_bytes(packet.data)
                fields = list(
                    struct.unpack(
                        lowbit_transport.HEADER_FORMAT,
                        sparse[: lowbit_transport.HEADER_BYTES],
                    )
                )
                fields[3] = 5
                malformed = struct.pack(lowbit_transport.HEADER_FORMAT, *fields) + sparse[
                    lowbit_transport.HEADER_BYTES :
                ]
                with self.assertRaises(guards.HybridQPayloadError):
                    lowbit_transport.inspect(malformed)


if __name__ == "__main__":
    unittest.main()
