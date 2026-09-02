"""Two CPU-only checks for per-channel UINT8 plus mandatory zstd transport.

No checkpoint, model, CUDA, dataset, cache, inference, validation or CARLA is
loaded or executed here.
"""

from __future__ import annotations

import functools
import struct
import unittest

import torch

from .. import contract, guards, uint8_codec, uint8_zstd_transport
from ..continuous_q import quantize_q, select_cells
from ..zstd_transport import ZstdWireCodec


@functools.lru_cache(maxsize=1)
def synthetic_c2() -> torch.Tensor:
    cells = contract.SPLIT_CELLS
    base = torch.linspace(-5.0, 7.0, cells, dtype=torch.float32).reshape(
        1, contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH
    )
    scales = torch.linspace(0.25, 2.25, contract.SPLIT_CHANNELS).reshape(-1, 1, 1)
    offsets = torch.linspace(-3.0, 3.0, contract.SPLIT_CHANNELS).reshape(-1, 1, 1)
    c2 = (base * scales + offsets).contiguous()
    c2[0] = base[0]  # explicitly crosses zero
    c2[1].fill_(2.5)  # positive constant
    c2[2].fill_(-4.75)  # negative constant
    return c2


def synthetic_scores() -> torch.Tensor:
    # Unique, deterministic ordering; higher row-major cells rank first.
    return torch.arange(contract.SPLIT_CELLS, dtype=torch.float32).reshape(
        contract.SPLIT_SPATIAL_SHAPE
    )


class CountingRanker:
    def __init__(self, expected_c2: torch.Tensor) -> None:
        self.expected_c2 = expected_c2
        self.calls = 0

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        if c2 is not self.expected_c2:
            raise AssertionError("ranker did not receive the original FP32 C2 object")
        if c2.dtype is not torch.float32:
            raise AssertionError("ranker did not receive FP32 C2")
        self.calls += 1
        return synthetic_scores()


def _rebuild_header(data: bytes, index: int, value: int | bytes) -> bytes:
    fields = list(
        struct.unpack(
            uint8_codec.HEADER_FORMAT, data[: uint8_codec.HEADER_BYTES]
        )
    )
    fields[index] = value
    return (
        struct.pack(uint8_codec.HEADER_FORMAT, *fields)
        + data[uint8_codec.HEADER_BYTES :]
    )


class Uint8TransportChecks(unittest.TestCase):
    def test_quantization_correctness_and_zero_scatter(self) -> None:
        c2 = synthetic_c2()
        prepared = uint8_codec.prepare(c2)
        selection = select_cells(synthetic_scores(), 0.50)
        payload = uint8_codec.encode(prepared, 0.50, selection)
        parsed = uint8_codec.inspect(payload)
        decoded, decoded_q = uint8_codec.decode(payload)

        self.assertEqual(decoded_q, 0.50)
        full_flat = c2.reshape(contract.SPLIT_CHANNELS, contract.SPLIT_CELLS)
        expected_ranges = torch.stack(
            (full_flat.amin(dim=1), full_flat.amax(dim=1)), dim=1
        )
        self.assertTrue(torch.equal(parsed.channel_ranges, expected_ranges))
        self.assertTrue(torch.equal(prepared.channel_ranges.cpu(), expected_ranges))

        keep = selection.keep_indices
        retained_original = full_flat.index_select(1, keep)
        retained_decoded = decoded.reshape(
            contract.SPLIT_CHANNELS, contract.SPLIT_CELLS
        ).index_select(1, keep)
        error = (retained_decoded - retained_original).abs().amax(dim=1)
        spans = expected_ranges[:, 1] - expected_ranges[:, 0]
        constant = spans <= uint8_codec.CONSTANT_SPAN_EPSILON
        magnitude = torch.maximum(
            expected_ranges[:, 0].abs(), expected_ranges[:, 1].abs()
        ).clamp_min(1.0)
        fp32_tolerance = 8.0 * torch.finfo(torch.float32).eps * magnitude
        bound = spans / (2.0 * 255.0) + fp32_tolerance
        self.assertTrue(bool((error[~constant] <= bound[~constant]).all()))

        # Exact constant-channel code/decode behavior, including negative data.
        self.assertTrue(bool((parsed.values[:, constant] == 0).all()))
        self.assertTrue(
            torch.equal(
                retained_decoded[constant],
                expected_ranges[constant, 0].unsqueeze(1).expand_as(
                    retained_decoded[constant]
                ),
            )
        )

        dropped = ~selection.keep_mask
        self.assertTrue(bool((decoded[:, dropped] == 0.0).all()))
        self.assertEqual(
            payload.total_bytes, uint8_codec.analytical_size(0.50).total_bytes
        )

    def test_transport_integrity_continuous_q_zstd_and_fail_closed(self) -> None:
        c2 = synthetic_c2()
        prepared = uint8_zstd_transport.prepare_frame(c2)
        ranges_before = prepared.channel_ranges.clone()
        ranker = CountingRanker(c2)
        wire = ZstdWireCodec()
        q_values = (0.00, 0.2345, 0.30, 0.50, 0.70, 0.90, 0.98)
        inspected: list[uint8_codec.InspectedUint8Payload] = []

        with torch.no_grad():
            for q in q_values:
                with self.subTest(q=q):
                    calls_before = ranker.calls
                    result = uint8_zstd_transport.encode(
                        prepared, ranker, q, wire_codec=wire
                    )
                    if q == 0.0:
                        self.assertEqual(ranker.calls, calls_before)
                        self.assertIsNone(result.selection)
                    else:
                        self.assertEqual(ranker.calls, calls_before + 1)
                        self.assertIsNotNone(result.selection)

                    # Independently rebuild the sparse bytes, then require the
                    # zstd decode to restore them byte-for-byte.
                    expected_sparse = uint8_codec.encode(
                        prepared, q, result.selection
                    )
                    restored = uint8_zstd_transport.decompress_payload(
                        result.packet, wire_codec=wire
                    )
                    self.assertEqual(restored, expected_sparse.data)
                    self.assertTrue(wire.round_trip_is_exact(expected_sparse.data, result.packet.data))
                    self.assertEqual(
                        result.packet.uncompressed_bytes,
                        uint8_codec.analytical_size(q).total_bytes,
                    )

                    parsed = uint8_codec.inspect(restored)
                    plan = quantize_q(q)
                    self.assertEqual(parsed.header.q_e4, plan.q_e4)
                    self.assertEqual(parsed.header.keep_count, plan.keep_count)
                    self.assertEqual(parsed.header.range_bytes, 2048)
                    self.assertEqual(parsed.values.dtype, torch.uint8)
                    self.assertEqual(parsed.header.mask_bytes, 0 if q == 0.0 else 2688)
                    self.assertTrue(
                        torch.equal(parsed.channel_ranges, ranges_before.cpu())
                    )
                    decoded, decoded_q = uint8_zstd_transport.decode(
                        result.packet, wire_codec=wire
                    )
                    self.assertEqual(decoded.dtype, torch.float32)
                    self.assertEqual(tuple(decoded.shape), contract.SPLIT_SHAPE)
                    self.assertEqual(decoded_q, plan.wire_q)
                    inspected.append(parsed)

        # One prepared frame supplied the unchanged full-C2 ranges to every q.
        self.assertTrue(torch.equal(prepared.channel_ranges, ranges_before))

        # Increasing q produces nested sets, and shared cells carry identical
        # channel-code vectors regardless of the chosen q.
        for less_sparse, more_sparse in zip(inspected, inspected[1:]):
            positions = torch.searchsorted(
                less_sparse.keep_indices, more_sparse.keep_indices
            )
            self.assertTrue(
                torch.equal(
                    less_sparse.keep_indices.index_select(0, positions),
                    more_sparse.keep_indices,
                )
            )
            self.assertTrue(
                torch.equal(
                    less_sparse.values.index_select(0, positions),
                    more_sparse.values,
                )
            )

        # The deployment decoder rejects raw sparse bytes: its only accepted
        # input is the mandatory-zstd packet type.
        with self.assertRaises(guards.HybridQPayloadError):
            uint8_zstd_transport.decode(inspected[-1])  # type: ignore[arg-type]

        good_result = uint8_zstd_transport.encode(
            prepared, ranker, 0.90, wire_codec=wire
        )
        good = uint8_zstd_transport.decompress_payload(
            good_result.packet, wire_codec=wire
        )
        range_start = uint8_codec.HEADER_BYTES + contract.mask_byte_count()

        nonfinite_range = bytearray(good)
        struct.pack_into("<f", nonfinite_range, range_start, float("nan"))
        reversed_range = bytearray(good)
        struct.pack_into("<ff", reversed_range, range_start, 1.0, -1.0)
        malformed_mask = bytearray(good)
        malformed_mask[uint8_codec.HEADER_BYTES] ^= 0x80

        malformed = {
            "magic": _rebuild_header(good, 0, b"BAD!"),
            "version": _rebuild_header(good, 1, 2),
            "codec identity": _rebuild_header(good, 2, 99),
            "dimensions": _rebuild_header(good, 3, 255),
            "q": _rebuild_header(good, 6, 9801),
            "keep count": _rebuild_header(
                good, 7, quantize_q(0.90).keep_count + 1
            ),
            "mask length": _rebuild_header(
                good, 8, contract.mask_byte_count() + 1
            ),
            "range length": _rebuild_header(good, 9, 2044),
            "value length": _rebuild_header(
                good, 10, quantize_q(0.90).keep_count * contract.SPLIT_CHANNELS + 1
            ),
            "nonfinite range": bytes(nonfinite_range),
            "min greater than max": bytes(reversed_range),
            "malformed mask": bytes(malformed_mask),
            "trailing bytes": good + b"\x00",
            "truncated payload": good[:-1],
        }
        for name, bad_sparse in malformed.items():
            with self.subTest(malformed=name):
                compressed = wire.compress(bad_sparse)
                packet = uint8_zstd_transport.Uint8ZstdPacket(
                    data=compressed.data,
                    uncompressed_bytes=compressed.uncompressed_bytes,
                )
                with self.assertRaises(guards.HybridQError):
                    uint8_zstd_transport.decode(packet, wire_codec=wire)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
