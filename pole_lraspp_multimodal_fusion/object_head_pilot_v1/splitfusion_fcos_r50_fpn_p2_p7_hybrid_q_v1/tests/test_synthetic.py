"""CPU-only synthetic checks. No real data, no checkpoint, no CUDA, no CARLA.

Public transport/runtime entry points are exercised at the frozen [256,112,192]
boundary. Small-tensor wire-layout checks use the private generic helpers
(`_select_cells`, `_encode`, `_decode`, `_apply_selection`, `_score_any`), which
exist so the public API can stay strictly frozen-shaped.
"""

from __future__ import annotations

import functools
import struct
import unittest

import torch
from torch import nn

from .. import codec, contract, guards, training
from ..ranker import SpatialRanker, build_ranker
from ..selection import (
    CellSelection,
    _apply_selection,
    _select_cells,
    apply_selection,
    select_and_apply,
    select_cells,
)


@functools.lru_cache(maxsize=1)
def frozen_c2() -> torch.Tensor:
    generator = torch.Generator().manual_seed(contract.RANKER_INIT_SEED)
    return torch.randn(*contract.SPLIT_SHAPE, generator=generator, dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def frozen_scores() -> torch.Tensor:
    generator = torch.Generator().manual_seed(4242)
    return torch.randn(*contract.SPLIT_SPATIAL_SHAPE, generator=generator)


class ExplodingRanker(nn.Module):
    """Fails loudly if the q=0 path ever invokes ranking."""

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise AssertionError("ranker must be bypassed at q=0")


class FrozenSyntheticPerception(nn.Module):
    """Stand-in for the frozen trunk: non-trainable parameters plus a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(7, 4, kernel_size=1)
        self.head = nn.Conv2d(4, 2, kernel_size=1)
        self.register_buffer("calibration", torch.arange(4, dtype=torch.float32))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()


def _generic_c2(channels: int = 8, height: int = 4, width: int = 6) -> torch.Tensor:
    return torch.randn(channels, height, width, dtype=torch.float32)


def _selection_from_indices(indices, cells, shape, q=0.50) -> CellSelection:
    """Hand-built selection; the caller must supply a q whose keep count matches."""
    keep_indices = torch.tensor(sorted(indices), dtype=torch.int64)
    keep = int(keep_indices.numel())
    assert keep == contract.keep_count(q, cells), (keep, contract.keep_count(q, cells))
    mask = torch.zeros(cells, dtype=torch.bool)
    mask[keep_indices] = True
    return CellSelection(
        q=q,
        cells=cells,
        keep_count=keep,
        drop_count=cells - keep,
        keep_indices=keep_indices,
        keep_mask=mask.reshape(shape),
    )


# ===========================================================================
# q semantics
# ===========================================================================


class QSemanticsCheck(unittest.TestCase):
    def test_registered_keep_counts_are_exact(self) -> None:
        expected = {
            0.00: 21504,
            0.30: 15053,
            0.50: 10752,
            0.70: 6451,
            0.90: 2150,
            0.98: 430,
        }
        self.assertEqual(contract.SPLIT_CELLS, 21504)
        self.assertEqual(contract.registered_keep_table(), expected)
        scores = frozen_scores()
        for q, keep in expected.items():
            self.assertEqual(contract.keep_count(q), keep)
            self.assertEqual(contract.drop_count(q), 21504 - keep)
            selection = select_cells(scores, q)
            self.assertEqual(selection.keep_count, keep)
            self.assertEqual(int(selection.keep_indices.numel()), keep)
            self.assertEqual(int(selection.keep_mask.sum()), keep)

    def test_invalid_q_fails_closed(self) -> None:
        for bad in (-0.1, 1.0, 1.5, float("nan"), 0.25, "0.5", None):
            with self.assertRaises(guards.HybridQConfigError):
                guards.require_valid_q(bad)
        with self.assertRaises(guards.HybridQConfigError):
            select_cells(frozen_scores(), 0.25)
        self.assertEqual(guards.require_valid_q(0.25, registered_only=False), 0.25)


class TieBreakCheck(unittest.TestCase):
    def test_ties_prefer_lower_row_major_index(self) -> None:
        selection = _select_cells(torch.zeros(4, 6), 0.50, registered_only=False)
        self.assertEqual(selection.keep_count, 12)
        self.assertEqual(selection.keep_indices.tolist(), list(range(12)))

    def test_partial_ties_are_broken_by_index(self) -> None:
        scores = torch.tensor([[5.0, 1.0, 1.0], [1.0, 1.0, 9.0], [1.0, 1.0, 1.0]])
        selection = _select_cells(scores, 0.70, registered_only=False)
        # keep = 9 - floor(0.7*9+0.5) = 3 -> {5, 0} by score, then index 1.
        self.assertEqual(selection.keep_count, 3)
        self.assertEqual(selection.keep_indices.tolist(), [0, 1, 5])

    def test_selection_is_repeatable_at_the_frozen_shape(self) -> None:
        first = select_cells(frozen_scores(), 0.70)
        second = select_cells(frozen_scores(), 0.70)
        self.assertTrue(torch.equal(first.keep_indices, second.keep_indices))
        self.assertTrue(torch.equal(first.keep_mask, second.keep_mask))


# ===========================================================================
# 1. Frozen C2 boundary
# ===========================================================================


class FrozenBoundaryCheck(unittest.TestCase):
    def _bad_tensors(self):
        return [
            torch.randn(255, 112, 192),
            torch.randn(256, 112, 191),
            torch.randn(256, 111, 192),
            torch.randn(256, 4, 6),
            torch.randn(*contract.SPLIT_SHAPE, dtype=torch.float64),
            torch.randn(112, 192),
            torch.randn(1, 256, 112, 192),
        ]

    def test_encode_rejects_non_frozen_tensors(self) -> None:
        for tensor in self._bad_tensors():
            with self.assertRaises(guards.HybridQPayloadError):
                codec.encode(tensor.to(tensor.dtype), 0.00)

    def test_masking_and_selection_entry_points_reject_non_frozen_tensors(self) -> None:
        ranker = build_ranker()
        for tensor in self._bad_tensors():
            with self.assertRaises(guards.HybridQPayloadError):
                select_and_apply(tensor, ranker, 0.50)
            with self.assertRaises(guards.HybridQPayloadError):
                ranker.score_cells(tensor)
        for scores in (torch.randn(4, 6), torch.randn(112, 191), torch.randn(112, 192, 1)):
            with self.assertRaises(guards.HybridQPayloadError):
                select_cells(scores, 0.50)
            with self.assertRaises(guards.HybridQPayloadError):
                training.straight_through_mask(scores, 0.50)

    def test_ranker_forward_accepts_only_frozen_and_batched_frozen_shapes(self) -> None:
        ranker = build_ranker()
        self.assertEqual(tuple(ranker(frozen_c2()).shape), contract.SPLIT_SPATIAL_SHAPE)
        batched = torch.zeros(2, *contract.SPLIT_SHAPE)
        self.assertEqual(tuple(ranker(batched).shape), (2, *contract.SPLIT_SPATIAL_SHAPE))
        for bad in (torch.randn(256, 4, 6), torch.randn(2, 255, 112, 192), torch.randn(112, 192)):
            with self.assertRaises(guards.HybridQPayloadError):
                ranker(bad)
        with self.assertRaises(guards.HybridQPayloadError):
            ranker(torch.zeros(*contract.SPLIT_SHAPE, dtype=torch.float64))

    def test_non_finite_input_fails_closed_at_every_entry_point(self) -> None:
        bad = frozen_c2().clone()
        bad[0, 0, 0] = float("nan")
        ranker = build_ranker()
        for call in (
            lambda: codec.encode(bad, 0.00),
            lambda: select_and_apply(bad, ranker, 0.50),
            lambda: ranker.score_cells(bad),
            lambda: ranker(bad),
        ):
            with self.assertRaises(guards.HybridQNumericalError):
                call()

    def test_q0_bypass_validates_before_returning(self) -> None:
        bad = frozen_c2().clone()
        bad[5, 5, 5] = float("inf")
        with self.assertRaises(guards.HybridQNumericalError):
            select_and_apply(bad, ExplodingRanker(), 0.00)
        with self.assertRaises(guards.HybridQPayloadError):
            select_and_apply(torch.randn(256, 4, 6), ExplodingRanker(), 0.00)

    def test_decode_requires_the_frozen_header(self) -> None:
        generic = codec._encode(_generic_c2(channels=2, height=3, width=4), 0.00)
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(generic)

        payload = codec.encode(frozen_c2(), 0.98, select_cells(frozen_scores(), 0.98))
        fields = list(struct.unpack(codec.HEADER_FORMAT, payload.data[: codec.HEADER_BYTES]))
        body = payload.data[codec.HEADER_BYTES :]

        def rebuilt(index: int, value) -> bytes:
            mutated = list(fields)
            mutated[index] = value
            return struct.pack(codec.HEADER_FORMAT, *mutated) + body

        for index, value in ((5, 255), (6, 111), (7, 191)):  # channels, height, width
            with self.assertRaises(guards.HybridQPayloadError):
                codec.decode(rebuilt(index, value))
        with self.assertRaises(guards.HybridQPayloadError):  # dtype code not FP32
            codec.decode(rebuilt(2, 2))
        with self.assertRaises(guards.HybridQConfigError):  # unregistered q
            codec.decode(rebuilt(8, 2500))


# ===========================================================================
# 2. Selection integrity cross-check
# ===========================================================================


class SelectionIntegrityCheck(unittest.TestCase):
    def _good(self) -> CellSelection:
        return select_cells(frozen_scores(), 0.90)

    def _corrupt(self, **overrides) -> CellSelection:
        base = self._good()
        fields = {
            "q": base.q,
            "cells": base.cells,
            "keep_count": base.keep_count,
            "drop_count": base.drop_count,
            "keep_indices": base.keep_indices,
            "keep_mask": base.keep_mask,
        }
        fields.update(overrides)
        return CellSelection(**fields)

    def test_valid_selection_passes_and_encodes(self) -> None:
        payload = codec.encode(frozen_c2(), 0.90, self._good())
        self.assertEqual(payload.keep_count, 2150)

    def test_mismatched_q_fails_closed(self) -> None:
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(frozen_c2(), 0.70, self._good())
        with self.assertRaises(guards.HybridQPayloadError):
            guards.require_selection_integrity(
                self._corrupt(q=0.70),
                0.90,
                cells=contract.SPLIT_CELLS,
                spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
            )

    def test_wrong_cell_count_fails_closed(self) -> None:
        with self.assertRaises(guards.HybridQPayloadError):
            guards.require_selection_integrity(
                self._corrupt(cells=21503),
                0.90,
                cells=contract.SPLIT_CELLS,
                spatial_shape=contract.SPLIT_SPATIAL_SHAPE,
            )

    def test_wrong_keep_or_drop_count_fails_closed(self) -> None:
        for override in ({"keep_count": 2149}, {"drop_count": 19353}):
            with self.assertRaises(guards.HybridQPayloadError):
                codec.encode(frozen_c2(), 0.90, self._corrupt(**override))

    def test_wrong_mask_shape_fails_closed(self) -> None:
        base = self._good()
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(
                frozen_c2(), 0.90, self._corrupt(keep_mask=base.keep_mask.reshape(192, 112))
            )

    def test_mask_popcount_mismatch_fails_closed(self) -> None:
        base = self._good()
        flipped = base.keep_mask.clone().reshape(-1)
        flipped[int(base.keep_indices[0])] = False
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(
                frozen_c2(),
                0.90,
                self._corrupt(keep_mask=flipped.reshape(contract.SPLIT_SPATIAL_SHAPE)),
            )

    def test_unordered_or_duplicate_indices_fail_closed(self) -> None:
        base = self._good()
        shuffled = base.keep_indices.flip(0)
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(frozen_c2(), 0.90, self._corrupt(keep_indices=shuffled))
        duplicated = base.keep_indices.clone()
        duplicated[1] = duplicated[0]
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(frozen_c2(), 0.90, self._corrupt(keep_indices=duplicated))

    def test_mask_and_index_set_must_describe_the_same_cells(self) -> None:
        base = self._good()
        # Same cardinality, same ordering, different cells: only the cross-check catches it.
        moved = base.keep_indices.clone()
        absent = int((~base.keep_mask.reshape(-1)).nonzero()[-1])
        moved[-1] = absent
        moved = torch.sort(moved).values
        self.assertEqual(int(moved.numel()), base.keep_count)
        guards.require_sorted_unique_indices(moved, contract.SPLIT_CELLS)
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(frozen_c2(), 0.90, self._corrupt(keep_indices=moved))

    def test_non_boolean_mask_fails_closed(self) -> None:
        base = self._good()
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(
                frozen_c2(), 0.90, self._corrupt(keep_mask=base.keep_mask.to(torch.uint8))
            )

    def test_apply_selection_also_cross_checks(self) -> None:
        with self.assertRaises(guards.HybridQPayloadError):
            apply_selection(frozen_c2(), self._corrupt(keep_count=2149))


# ===========================================================================
# 3. Payload accounting
# ===========================================================================


class PayloadAccountingCheck(unittest.TestCase):
    def test_q0_tensor_identity_and_framed_length(self) -> None:
        c2 = frozen_c2()
        out, selection = select_and_apply(c2, ExplodingRanker(), 0.00)
        self.assertIsNone(selection)
        self.assertIs(out, c2)

        payload = codec.encode(c2, 0.00)
        self.assertEqual(payload.mask_bytes, 0)
        self.assertEqual(payload.total_bytes, contract.FRAMED_Q0_PAYLOAD_BYTES)
        self.assertEqual(payload.total_bytes, 22020140)
        decoded, q = codec.decode(payload)
        self.assertEqual(q, 0.0)
        self.assertTrue(torch.equal(decoded, c2))

    def test_framed_q0_is_not_byte_identical_to_the_raw_tensor(self) -> None:
        c2 = frozen_c2()
        self.assertEqual(codec.raw_fp32_reference_bytes(c2), 22020096)
        self.assertEqual(contract.RAW_FP32_REFERENCE_BYTES, 22020096)
        self.assertEqual(
            contract.FRAMED_Q0_PAYLOAD_BYTES - contract.RAW_FP32_REFERENCE_BYTES,
            codec.HEADER_BYTES,
        )
        self.assertEqual(contract.HEADER_OVERHEAD_BYTES, 44)

    def test_primary_ratios_use_the_framed_q0_denominator(self) -> None:
        c2 = frozen_c2()
        scores = frozen_scores()
        expected_bytes = {
            0.00: 22020140,
            0.30: 15417004,
            0.50: 11012780,
            0.70: 6608556,
            0.90: 2204332,
            0.98: 443052,
        }
        rows = []
        q0_bytes = None
        for q in contract.REGISTERED_Q_VALUES:
            payload = (
                codec.encode(c2, q)
                if q == 0.00
                else codec.encode(c2, q, select_cells(scores, q))
            )
            self.assertEqual(payload.total_bytes, len(payload.data))
            self.assertEqual(
                payload.total_bytes,
                codec.HEADER_BYTES + payload.mask_bytes + payload.value_bytes,
            )
            self.assertEqual(payload.total_bytes, expected_bytes[q])
            if q == 0.00:
                q0_bytes = payload.total_bytes
                self.assertEqual(payload.framed_ratio, 1.0)
            self.assertAlmostEqual(
                payload.framed_ratio, payload.total_bytes / q0_bytes, places=12
            )
            rows.append((q, payload.keep_count, payload.total_bytes, payload.framed_ratio))

        print(
            "\nactual serialized hybrid-q payloads "
            f"(framed q=0 denominator = {q0_bytes}; raw FP32 reference = "
            f"{contract.RAW_FP32_REFERENCE_BYTES}, reported separately):"
        )
        for q, keep, total, ratio in rows:
            print(f"  q={q:.2f} keep={keep:5d} bytes={total:9d} framed_ratio={ratio:.6f}")


# ===========================================================================
# 4. Locked configuration
# ===========================================================================


class LockedConfigCheck(unittest.TestCase):
    def test_locked_config_agrees_with_the_module_constants(self) -> None:
        config = contract.load_locked_config()
        self.assertEqual(config["seed"], 20260829)
        self.assertEqual(config["seed"], contract.RANKER_INIT_SEED)
        binding = config["perception_binding"]
        self.assertEqual(
            binding["perception_forward_lock_path"], contract.PERCEPTION_LOCK_RELPATH
        )
        self.assertEqual(
            binding["perception_forward_lock_sha256"], contract.PERCEPTION_LOCK_SHA256
        )
        self.assertEqual(binding["checkpoint_sha256"], contract.FROZEN_CHECKPOINT_SHA256)
        # Phase 3 loads the frozen checkpoint, but only for the bounded qualification.
        self.assertTrue(binding["checkpoint_loaded_in_this_phase"])
        self.assertEqual(binding["frozen_model_runtime_mode"], "eval")
        self.assertFalse(config["training"]["executed_in_this_phase"])
        self.assertFalse(config["phase3_qualification"]["teacher_cache_written"])
        self.assertEqual(config["phase3_qualification"]["epochs_trained"], 0)
        self.assertFalse(config["phase3_qualification"]["validation_or_test_accessed"])
        self.assertEqual(tuple(config["c2_contract"]["shape"]), contract.SPLIT_SHAPE)
        self.assertEqual(config["wire_format"]["header_bytes"], codec.HEADER_BYTES)
        self.assertEqual(
            config["wire_format"]["framed_q0_payload_bytes"],
            contract.FRAMED_Q0_PAYLOAD_BYTES,
        )
        self.assertEqual(config["ranker"]["parameter_count"], contract.RANKER_PARAMETER_COUNT)
        self.assertFalse(config["ranker"]["final_layer_bias"])
        self.assertEqual(config["ranker"]["mac_count_112x192"], 45760512)

    def test_locked_config_hash_matches_the_perception_lock_on_disk(self) -> None:
        import hashlib

        digest = hashlib.sha256(contract.perception_lock_path().read_bytes()).hexdigest()
        self.assertEqual(digest, contract.PERCEPTION_LOCK_SHA256)

    def test_locked_training_semantics(self) -> None:
        config = contract.load_locked_config()["training"]
        self.assertFalse(config["executed_in_this_phase"])
        self.assertEqual(tuple(config["teacher"]["groups"]), ("D", "G", "S", "A"))
        self.assertEqual(config["teacher"]["normalization"], "l1")
        self.assertEqual(config["teacher"]["combination"], "equal_weight")
        self.assertEqual(config["distillation"]["temperature"], 1.0)
        self.assertEqual(config["distillation"]["epochs"], 4)
        self.assertEqual(config["q_aware"]["epochs"], 8)
        self.assertEqual(tuple(config["q_aware"]["cycle"]), (0.30, 0.50, 0.70))
        self.assertEqual(config["q_aware"]["q_per_optimizer_update"], 1)
        self.assertEqual(config["q_aware"]["distillation_weight"], 0.1)
        self.assertEqual(config["q_aware"]["reference_median_source"], "fit_train")
        self.assertEqual(tuple(config["q_aware"]["evaluation_stress_q"]), (0.90, 0.98))
        optimization = config["optimization"]
        self.assertEqual(optimization["optimizer"], "AdamW")
        self.assertEqual(optimization["learning_rate"], 1e-3)
        self.assertEqual(optimization["weight_decay"], 1e-4)
        self.assertEqual(optimization["lr_schedule"], "constant")
        self.assertEqual(optimization["grad_clip_global_norm"], 5.0)
        self.assertEqual(tuple(optimization["checkpoint_epochs"]), (4, 8, 12))
        self.assertFalse(optimization["augmentation"])
        self.assertEqual(config["straight_through"]["temperature"], 1.0)
        self.assertEqual(
            config["straight_through"]["boundary"],
            "midpoint_lowest_retained_highest_dropped",
        )


# ===========================================================================
# Masking and codec layout
# ===========================================================================


class MaskingCheck(unittest.TestCase):
    def test_retained_cells_keep_all_channels_and_dropped_cells_are_zero(self) -> None:
        c2 = frozen_c2()
        masked, selection = select_and_apply(c2, build_ranker(), 0.98)
        self.assertIsNotNone(selection)
        self.assertEqual(selection.keep_count, 430)
        flat_in = c2.reshape(contract.SPLIT_CHANNELS, contract.SPLIT_CELLS)
        flat_out = masked.reshape(contract.SPLIT_CHANNELS, contract.SPLIT_CELLS)
        kept = selection.keep_indices
        self.assertTrue(
            torch.equal(flat_out.index_select(1, kept), flat_in.index_select(1, kept))
        )
        dropped = (~selection.keep_mask.reshape(-1)).nonzero().reshape(-1)
        self.assertEqual(int(dropped.numel()), 21074)
        self.assertEqual(float(flat_out.index_select(1, dropped).abs().sum()), 0.0)


class CodecRoundTripCheck(unittest.TestCase):
    def test_frozen_round_trip_matches_the_masked_tensor(self) -> None:
        c2 = frozen_c2()
        scores = frozen_scores()
        for q in (0.30, 0.90, 0.98):
            selection = select_cells(scores, q)
            masked = apply_selection(c2, selection)
            payload = codec.encode(c2, q, selection)
            decoded, decoded_q = codec.decode(payload)
            self.assertEqual(decoded_q, q)
            self.assertEqual(tuple(decoded.shape), contract.SPLIT_SHAPE)
            self.assertEqual(decoded.dtype, torch.float32)
            self.assertTrue(torch.equal(decoded, masked))

    def test_generic_round_trip_across_every_registered_q(self) -> None:
        c2 = _generic_c2(channels=256, height=4, width=6)
        scores = torch.randn(4, 6)
        for q in (0.30, 0.50, 0.70, 0.90, 0.98):
            selection = _select_cells(scores, q, registered_only=False)
            masked = _apply_selection(c2, selection)
            payload = codec._encode(c2, q, selection, registered_only=False)
            decoded, decoded_q = codec._decode(payload, require_frozen=False)
            self.assertEqual(decoded_q, q)
            self.assertTrue(torch.equal(decoded, masked))

    def test_bitmask_byte_and_bit_order_is_msb_first(self) -> None:
        # 10 cells at q=0.50 -> keep 5; 2 mask bytes with 6 padding bits.
        selection = _selection_from_indices([0, 3, 7, 8, 9], 10, (2, 5))
        payload = codec._encode(torch.zeros(2, 2, 5), 0.50, selection)
        mask = payload.data[codec.HEADER_BYTES : codec.HEADER_BYTES + payload.mask_bytes]
        self.assertEqual(payload.mask_bytes, 2)
        # cell i -> byte i//8, bit 7-(i%8).
        # byte 0 holds cells 0,3,7 -> 0b10010001; byte 1 holds cells 8,9 -> 0b11000000.
        self.assertEqual(tuple(mask), (0x91, 0xC0))
        self.assertEqual(
            tuple(codec._decode(payload, require_frozen=False)[0].shape), (2, 2, 5)
        )

    def test_values_are_stored_in_ascending_row_major_cell_order(self) -> None:
        c2 = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        kept = [1, 2, 5, 6, 9, 10]  # 12 cells at q=0.50 -> keep 6
        payload = codec._encode(c2, 0.50, _selection_from_indices(kept, 12, (3, 4)))
        values = payload.data[codec.HEADER_BYTES + payload.mask_bytes :]
        got = torch.frombuffer(bytearray(values), dtype=torch.float32).reshape(6, 2)
        flat = c2.reshape(2, 12)
        for row, cell in enumerate(kept):
            self.assertTrue(torch.equal(got[row], flat[:, cell]))


class MalformedPayloadCheck(unittest.TestCase):
    def _payload(self) -> codec.SparsePayload:
        selection = _selection_from_indices(
            [0, 2, 3, 5, 7, 11, 15, 19, 20, 21, 22, 23], 24, (4, 6)
        )
        return codec._encode(torch.randn(4, 4, 6), 0.50, selection)

    def _decode(self, data):
        return codec._decode(data, require_frozen=False)

    def test_truncated_and_corrupt_payloads_fail_closed(self) -> None:
        data = self._payload().data
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(data[: codec.HEADER_BYTES - 1])
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(data[:-4])
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(data + b"\x00\x00\x00\x00")
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(b"XXXX" + data[4:])
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(b"")
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode("not bytes")

    def test_header_field_corruption_fails_closed(self) -> None:
        payload = self._payload()
        fields = list(struct.unpack(codec.HEADER_FORMAT, payload.data[: codec.HEADER_BYTES]))
        body = payload.data[codec.HEADER_BYTES :]

        def rebuilt(index: int, value) -> bytes:
            mutated = list(fields)
            mutated[index] = value
            return struct.pack(codec.HEADER_FORMAT, *mutated) + body

        with self.assertRaises(guards.HybridQPayloadError):  # format version
            self._decode(rebuilt(1, 99))
        with self.assertRaises(guards.HybridQPayloadError):  # dtype code
            self._decode(rebuilt(2, 7))
        with self.assertRaises(guards.HybridQPayloadError):  # unknown flag bits
            self._decode(rebuilt(3, 0b110))
        with self.assertRaises(guards.HybridQPayloadError):  # reserved must be zero
            self._decode(rebuilt(4, 1))
        with self.assertRaises(guards.HybridQPayloadError):  # channel count
            self._decode(rebuilt(5, 5))
        with self.assertRaises(guards.HybridQPayloadError):  # width -> shape mismatch
            self._decode(rebuilt(7, 0))
        with self.assertRaises(guards.HybridQPayloadError):  # keep cardinality
            self._decode(rebuilt(9, 11))
        with self.assertRaises(guards.HybridQPayloadError):  # mask length
            self._decode(rebuilt(10, 4))
        with self.assertRaises(guards.HybridQPayloadError):  # value block length
            self._decode(rebuilt(11, 16))

    def test_bitmask_corruption_fails_closed(self) -> None:
        data = bytearray(self._payload().data)
        data[codec.HEADER_BYTES] ^= 0xFF  # popcount no longer matches the header
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(bytes(data))

    def test_padding_bits_past_the_last_cell_fail_closed(self) -> None:
        selection = _selection_from_indices([0, 1, 2, 3, 4, 5], 12, (3, 4))
        payload = codec._encode(torch.randn(2, 3, 4), 0.50, selection)
        data = bytearray(payload.data)
        data[codec.HEADER_BYTES + 1] |= 0x01  # set a padding bit
        with self.assertRaises(guards.HybridQPayloadError):
            self._decode(bytes(data))

    def test_duplicate_or_unordered_retained_indices_fail_closed(self) -> None:
        for bad in ([0, 2, 2, 5], [5, 2, 9], [0, 30]):
            with self.assertRaises(guards.HybridQPayloadError):
                guards.require_sorted_unique_indices(torch.tensor(bad), 24)

    def test_encode_rejects_selection_mismatch_and_bad_dtype(self) -> None:
        c2 = frozen_c2()
        with self.assertRaises(guards.HybridQConfigError):  # q>0 without a selection
            codec.encode(c2, 0.50)
        with self.assertRaises(guards.HybridQPayloadError):  # q=0 with a selection
            codec.encode(c2, 0.00, select_cells(frozen_scores(), 0.90))
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(c2.to(torch.float64), 0.00)

    def test_non_finite_ranker_scores_fail_closed(self) -> None:
        scores = frozen_scores().clone()
        scores[1, 1] = float("inf")
        with self.assertRaises(guards.HybridQNumericalError):
            select_cells(scores, 0.50)


# ===========================================================================
# Ranker shape and deterministic initialization
# ===========================================================================


class RankerShapeCheck(unittest.TestCase):
    def test_parameter_and_mac_counts_match_the_contract(self) -> None:
        ranker = build_ranker()
        self.assertEqual(ranker.parameter_count(), 2144)
        self.assertEqual(contract.RANKER_PARAMETER_COUNT, 2144)
        cells = contract.SPLIT_CELLS
        macs = sum(
            conv.weight.numel() * cells
            for conv in (ranker.reduce, ranker.depthwise, ranker.score)
        )
        self.assertEqual(macs, 45760512)
        self.assertEqual(ranker.mac_count(contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH), macs)
        self.assertEqual(contract.ranker_mac_count(), macs)
        self.assertAlmostEqual(macs / 1e6, 45.76, places=2)

    def test_layer_stack_is_exactly_the_specified_five_operations(self) -> None:
        ranker = build_ranker()
        self.assertEqual(ranker.reduce.in_channels, 256)
        self.assertEqual(ranker.reduce.out_channels, 8)
        self.assertEqual(ranker.reduce.kernel_size, (1, 1))
        self.assertEqual(ranker.depthwise.groups, 8)
        self.assertEqual(ranker.depthwise.kernel_size, (3, 3))
        self.assertEqual(ranker.depthwise.padding, (1, 1))
        self.assertEqual(ranker.score.out_channels, 1)
        self.assertEqual(ranker.score.kernel_size, (1, 1))
        # A global scalar score offset cannot change cell ranking, so the final
        # layer carries no unidentifiable bias.
        self.assertIsNone(ranker.score.bias)
        kinds = {type(module) for module in ranker.modules()}
        self.assertNotIn(nn.BatchNorm2d, kinds)
        self.assertEqual(kinds - {SpatialRanker, nn.Conv2d, nn.ReLU}, set())

    def test_scores_are_spatial_and_the_input_is_detached(self) -> None:
        ranker = build_ranker()
        c2 = frozen_c2().clone().requires_grad_(True)
        scores = ranker.score_cells(c2)
        self.assertEqual(tuple(scores.shape), contract.SPLIT_SPATIAL_SHAPE)
        scores.sum().backward()
        self.assertIsNone(c2.grad)
        self.assertIsNotNone(ranker.reduce.weight.grad)


class DeterministicInitCheck(unittest.TestCase):
    def test_registered_seed_reproduces_identical_parameters(self) -> None:
        first = build_ranker()
        second = build_ranker(seed=contract.RANKER_INIT_SEED)
        self.assertEqual(contract.RANKER_INIT_SEED, 20260829)
        for (name, a), (other, b) in zip(
            first.named_parameters(), second.named_parameters()
        ):
            self.assertEqual(name, other)
            self.assertTrue(torch.equal(a.detach(), b.detach()), name)
        third = build_ranker(seed=1)
        self.assertFalse(
            torch.equal(first.reduce.weight.detach(), third.reduce.weight.detach())
        )

    def test_construction_does_not_advance_the_caller_rng(self) -> None:
        torch.manual_seed(7)
        expected = torch.randn(5)
        torch.manual_seed(7)
        state_before = torch.get_rng_state()
        build_ranker()
        self.assertTrue(torch.equal(torch.get_rng_state(), state_before))
        self.assertTrue(torch.equal(torch.randn(5), expected))

    def test_parameter_count_is_still_2144(self) -> None:
        self.assertEqual(build_ranker().parameter_count(), 2144)


# ===========================================================================
# Optimizer ownership and frozen state
# ===========================================================================


class OptimizerOwnershipCheck(unittest.TestCase):
    def test_optimizer_is_locked_adamw_over_ranker_parameters_only(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        optimizer = training.build_ranker_optimizer(ranker, frozen_modules=[frozen])
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(owned, {id(p) for p in ranker.parameters()})
        # reduce and depthwise carry weight + bias; the final score conv has no bias.
        self.assertEqual(len(owned), 5)
        self.assertEqual(sum(p.numel() for p in ranker.parameters()), 2144)
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], contract.LEARNING_RATE)
            self.assertEqual(group["weight_decay"], contract.WEIGHT_DECAY)
        for parameter in frozen.parameters():
            self.assertNotIn(id(parameter), owned)

    def test_trainable_or_training_mode_frozen_stack_fails_closed(self) -> None:
        ranker = build_ranker()
        trainable = FrozenSyntheticPerception()
        trainable.stem.weight.requires_grad_(True)
        with self.assertRaises(guards.HybridQOwnershipError):
            training.build_ranker_optimizer(ranker, frozen_modules=[trainable])
        in_training_mode = FrozenSyntheticPerception()
        in_training_mode.train()
        with self.assertRaises(guards.HybridQOwnershipError):
            training.build_ranker_optimizer(ranker, frozen_modules=[in_training_mode])

    def test_ownership_guard_rejects_foreign_parameters(self) -> None:
        ranker = build_ranker()
        smuggled = nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.AdamW([*ranker.parameters(), smuggled], lr=1e-3)
        with self.assertRaises(guards.HybridQOwnershipError):
            guards.require_optimizer_owns_only(optimizer, list(ranker.parameters()))


class FrozenStateCheck(unittest.TestCase):
    def test_frozen_parameters_and_buffers_are_unchanged_after_an_optimizer_step(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        state = guards.snapshot_module_state(frozen)
        self.assertIn("buffer:calibration", state)
        ranker_before = guards.snapshot_parameters(ranker)
        optimizer = training.build_ranker_optimizer(ranker, frozen_modules=[frozen])

        c2 = frozen_c2()
        teacher = training.build_teacher_maps(c2, {"D": torch.randn_like(c2)})
        self.assertTrue(teacher.is_supervisable)

        optimizer.zero_grad(set_to_none=True)
        loss = training.ranker_distillation_loss(
            ranker.score_cells(c2), teacher.importance
        )
        loss.backward()
        qualification = training.GradientQualification.for_module(ranker, window=1)
        self.assertTrue(qualification.observe(ranker, loss=loss))
        training.clip_ranker_gradients(ranker)
        optimizer.step()
        training.require_post_step_health(ranker, optimizer)

        guards.require_module_state_unchanged(frozen, state)
        guards.require_frozen_perception([frozen])
        guards.require_eval_mode([frozen])
        qualification.require_qualified()
        self.assertTrue(
            any(
                not torch.equal(current.detach(), before)
                for current, before in zip(ranker.parameters(), ranker_before.values())
            )
        )

    def test_changed_frozen_parameter_or_buffer_fails_closed(self) -> None:
        frozen = FrozenSyntheticPerception()
        state = guards.snapshot_module_state(frozen)
        with torch.no_grad():
            frozen.calibration[0] += 1.0
        with self.assertRaises(guards.HybridQOwnershipError):
            guards.require_module_state_unchanged(frozen, state)

        other = FrozenSyntheticPerception()
        other_state = guards.snapshot_module_state(other)
        with torch.no_grad():
            other.stem.weight[0, 0, 0, 0] += 1.0
        with self.assertRaises(guards.HybridQOwnershipError):
            guards.require_module_state_unchanged(other, other_state)


# ===========================================================================
# Teacher maps over the registered loss groups
# ===========================================================================


class TeacherMapCheck(unittest.TestCase):
    def test_importance_definition_and_independent_l1_normalization(self) -> None:
        c2 = torch.tensor([[[1.0, -2.0]], [[3.0, 4.0]]])  # [2,1,2]
        grad = torch.tensor([[[2.0, 1.0]], [[-1.0, 0.5]]])
        raw = training.teacher_importance_map(c2, grad)
        self.assertTrue(torch.equal(raw, torch.tensor([[5.0, 4.0]])))

        # A group scaled by 1000x must not dominate after independent L1 normalization.
        result = training.build_teacher_maps(c2, {"D": grad, "S": grad * 1000.0})
        self.assertEqual(result.valid_groups, ("D", "S"))
        self.assertTrue(torch.allclose(result.group_maps["D"], result.group_maps["S"]))
        self.assertAlmostEqual(float(result.importance.sum()), 1.0, places=6)
        self.assertAlmostEqual(
            result.gradient_mass["S"] / result.gradient_mass["D"], 1000.0, places=3
        )
        self.assertEqual(result.normalization, "l1")
        self.assertEqual(result.combination, "equal_weight")

    def test_only_the_registered_groups_are_accepted(self) -> None:
        c2 = torch.randn(4, 3, 5)
        with self.assertRaises(guards.HybridQConfigError):
            training.build_teacher_maps(c2, {"detect": torch.randn(4, 3, 5)})
        with self.assertRaises(guards.HybridQConfigError):
            training.build_teacher_maps(
                c2, {"D": torch.randn(4, 3, 5)}, task_losses={"person_only": 1.0}
            )
        result = training.build_teacher_maps(
            c2, {group: torch.randn(4, 3, 5) for group in ("D", "G", "S", "A")}
        )
        self.assertEqual(result.valid_groups, ("D", "G", "S", "A"))
        self.assertEqual(result.excluded_groups, {})

    def test_absent_zero_and_non_finite_groups_are_recorded_and_excluded(self) -> None:
        c2 = torch.randn(4, 3, 5)
        result = training.build_teacher_maps(
            c2,
            {
                "D": torch.randn(4, 3, 5),
                "G": None,
                "S": torch.zeros(4, 3, 5),
                "A": torch.full((4, 3, 5), float("nan")),
            },
        )
        self.assertEqual(result.valid_groups, ("D",))
        self.assertEqual(
            result.excluded_groups,
            {"G": "absent", "S": "zero_gradient", "A": "non_finite"},
        )
        self.assertTrue(result.is_supervisable)
        self.assertEqual(set(result.gradient_mass), {"D"})

    def test_missing_group_key_counts_as_absent(self) -> None:
        c2 = torch.randn(4, 3, 5)
        result = training.build_teacher_maps(c2, {"D": torch.randn(4, 3, 5)})
        self.assertEqual(result.valid_groups, ("D",))
        self.assertEqual(
            result.excluded_groups, {"G": "absent", "S": "absent", "A": "absent"}
        )

    def test_frame_with_no_valid_group_is_not_supervisable(self) -> None:
        c2 = torch.randn(4, 3, 5)
        result = training.build_teacher_maps(
            c2, {"D": None, "S": torch.zeros(4, 3, 5)}
        )
        self.assertIsNone(result.importance)
        self.assertFalse(result.is_supervisable)
        self.assertEqual(result.valid_groups, ())

    def test_gradient_mass_is_diagnostic_and_task_losses_are_separate(self) -> None:
        c2 = torch.randn(4, 3, 5)
        result = training.build_teacher_maps(
            c2,
            {"D": torch.randn(4, 3, 5), "G": torch.randn(4, 3, 5)},
            task_losses={"D": 2.5, "G": 0.25},
        )
        self.assertEqual(result.task_losses, {"D": 2.5, "G": 0.25})
        self.assertEqual(set(result.gradient_mass), {"D", "G"})
        # Equal weight: the combined map is the mean of the normalized group maps.
        expected = (result.group_maps["D"] + result.group_maps["G"]) / 2.0
        self.assertTrue(torch.allclose(result.importance, expected, atol=1e-7))
        self.assertNotIn("loss_scales", result.__dataclass_fields__)

    def test_cache_record_excludes_c2_and_carries_the_locked_fields(self) -> None:
        record = training.TeacherCacheRecord(
            frame_id="synthetic_0001",
            sequence_id="synthetic",
            importance=torch.rand(3, 5),
            valid_groups=("D",),
            excluded_groups={"S": "absent"},
            gradient_mass={"D": 1.5},
            task_losses={"D": 0.75},
        )
        self.assertEqual(
            record.perception_checkpoint_sha256, contract.FROZEN_CHECKPOINT_SHA256
        )
        self.assertEqual(record.normalization, "l1")
        for forbidden in ("c2", "features", "rgb", "radar", "detections"):
            self.assertNotIn(forbidden, record.__dataclass_fields__)


# ===========================================================================
# Locked distillation and q-aware contract
# ===========================================================================


class QAwareContractCheck(unittest.TestCase):
    def test_locked_distillation_temperature_is_one(self) -> None:
        self.assertEqual(contract.DISTILLATION_TEMPERATURE, 1.0)
        self.assertEqual(contract.DISTILLATION_EPOCHS, 4)
        scores = frozen_scores()
        teacher = torch.rand(*contract.SPLIT_SPATIAL_SHAPE)
        teacher = teacher / teacher.sum()
        default = training.ranker_distillation_loss(scores, teacher)
        explicit = training.ranker_distillation_loss(scores, teacher, temperature=1.0)
        self.assertTrue(torch.equal(default, explicit))
        for bad in (None, 0.0, -1.0, float("nan")):
            with self.assertRaises(guards.HybridQConfigError):
                training.ranker_distillation_loss(scores, teacher, temperature=bad)

    def test_q_schedule_is_a_deterministic_repeated_cycle(self) -> None:
        self.assertEqual(contract.Q_AWARE_EPOCHS, 8)
        self.assertEqual(contract.Q_AWARE_TRAINING_CYCLE, (0.30, 0.50, 0.70))
        self.assertEqual(
            training.q_aware_schedule(7),
            (0.30, 0.50, 0.70, 0.30, 0.50, 0.70, 0.30),
        )
        self.assertEqual(training.q_for_update(0), 0.30)
        self.assertEqual(training.q_for_update(101), training.q_for_update(101 % 3))
        self.assertNotIn(0.00, training.q_aware_schedule(30))
        self.assertNotIn(0.90, training.q_aware_schedule(30))
        self.assertNotIn(0.98, training.q_aware_schedule(30))
        self.assertEqual(contract.EVALUATION_STRESS_Q_VALUES, (0.90, 0.98))
        with self.assertRaises(guards.HybridQConfigError):
            training.q_for_update(-1)

    def test_reference_medians_must_come_from_fit_training_data(self) -> None:
        good = training.ReferenceMedians({"D": 1.0, "G": 2.0})
        self.assertEqual(good.require("D"), 1.0)
        with self.assertRaises(guards.HybridQConfigError):
            training.ReferenceMedians({"D": 1.0}, source="validation")
        with self.assertRaises(guards.HybridQConfigError):
            training.ReferenceMedians({"D": 1.0}, source="test")
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(guards.HybridQConfigError):
                training.ReferenceMedians({"D": bad})
        with self.assertRaises(guards.HybridQConfigError):
            training.ReferenceMedians({"person": 1.0})
        with self.assertRaises(guards.HybridQConfigError):
            good.require("S")

    def test_q_aware_objective_shape(self) -> None:
        references = training.ReferenceMedians({"D": 2.0, "S": 4.0})
        losses = {"D": torch.tensor(1.0), "G": None, "S": torch.tensor(2.0)}
        distillation = torch.tensor(3.0)
        value = training.q_aware_objective(losses, distillation, references)
        # mean(1/2, 2/4) + 0.1 * 3 = 0.5 + 0.3
        self.assertAlmostEqual(float(value), 0.8, places=6)
        self.assertEqual(contract.Q_AWARE_DISTILLATION_WEIGHT, 0.1)
        with self.assertRaises(guards.HybridQConfigError):
            training.q_aware_objective({"D": None}, distillation, references)
        with self.assertRaises(guards.HybridQConfigError):
            training.q_aware_objective({"person": torch.tensor(1.0)}, distillation, references)
        with self.assertRaises(guards.HybridQNumericalError):
            training.q_aware_objective(
                {"D": torch.tensor(float("nan"))}, distillation, references
            )


class StraightThroughCheck(unittest.TestCase):
    def test_forward_output_equals_the_hard_mask(self) -> None:
        scores = frozen_scores()
        for q in (0.30, 0.50, 0.70):
            selection = select_cells(scores, q)
            mask = training.straight_through_mask(scores, q)
            self.assertTrue(torch.equal(mask.detach(), selection.keep_mask.to(scores.dtype)))
            self.assertEqual(set(mask.detach().unique().tolist()), {0.0, 1.0})
            self.assertEqual(int(mask.detach().sum()), selection.keep_count)

    def test_locked_temperature_and_boundary(self) -> None:
        self.assertEqual(contract.STRAIGHT_THROUGH_TEMPERATURE, 1.0)
        self.assertEqual(
            contract.STRAIGHT_THROUGH_BOUNDARY,
            "midpoint_lowest_retained_highest_dropped",
        )
        # No production temperature override is exposed on the public entry point.
        import inspect

        signature = inspect.signature(training.straight_through_mask)
        self.assertEqual(list(signature.parameters), ["scores", "q"])

    def test_q0_is_an_identity_bypass_with_no_ranker_gradient(self) -> None:
        ranker = build_ranker()
        c2 = frozen_c2()
        scores = ranker.score_cells(c2)
        mask = training.straight_through_mask(scores, 0.00)
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
        self.assertIsNone(mask.grad_fn)
        self.assertFalse(mask.requires_grad)
        ranker.zero_grad(set_to_none=True)
        masked = training.masked_c2_forward(c2, mask)
        self.assertFalse(masked.requires_grad)
        self.assertIsNone(masked.grad_fn)
        with self.assertRaises(RuntimeError):
            masked.sum().backward()  # no ranker-training gradient path exists at q=0
        self.assertTrue(all(p.grad is None for p in ranker.parameters()))

    def test_gradient_flows_through_the_surrogate_for_q_above_zero(self) -> None:
        ranker = build_ranker()
        c2 = frozen_c2()
        scores = ranker.score_cells(c2)
        mask = training.straight_through_mask(scores, 0.50)
        training.masked_c2_forward(c2, mask).sum().backward()
        self.assertIsNotNone(ranker.score.weight.grad)
        self.assertTrue(torch.isfinite(ranker.score.weight.grad).all())


# ===========================================================================
# 5. Gradient qualification
# ===========================================================================


class GradientQualificationCheck(unittest.TestCase):
    def _ranker_with_grads(self, value: float = 1.0):
        ranker = build_ranker()
        for parameter in ranker.parameters():
            parameter.grad = torch.full_like(parameter, value)
        return ranker

    def test_tracks_every_named_trainable_tensor(self) -> None:
        ranker = build_ranker()
        qualification = training.GradientQualification.for_module(ranker, window=2)
        self.assertEqual(
            qualification.parameter_names,
            ("reduce.weight", "reduce.bias", "depthwise.weight", "depthwise.bias",
             "score.weight"),
        )
        for _ in range(2):
            for parameter in ranker.parameters():
                parameter.grad = torch.ones_like(parameter)
            self.assertTrue(qualification.observe(ranker, loss=torch.tensor(0.5)))
        qualification.require_qualified()
        self.assertTrue(qualification.qualified())
        self.assertEqual(qualification.zero_gradient_batches, [])
        self.assertEqual(qualification.missing_gradient_batches, [])

    def test_isolated_zero_gradient_batch_is_logged_not_fatal(self) -> None:
        ranker = self._ranker_with_grads(0.0)
        qualification = training.GradientQualification.for_module(ranker, window=2)
        self.assertFalse(qualification.observe(ranker, loss=torch.tensor(0.1)))
        self.assertEqual(len(qualification.zero_gradient_batches), 1)
        self.assertEqual(qualification.zero_gradient_batches[0][0], 1)
        self.assertEqual(
            set(qualification.zero_gradient_batches[0][1]),
            set(qualification.parameter_names),
        )
        with self.assertRaises(guards.HybridQQualificationError):
            qualification.require_qualified()  # window incomplete
        for parameter in ranker.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.assertTrue(qualification.observe(ranker, loss=torch.tensor(0.1)))
        qualification.require_qualified()

    def test_parameter_that_never_goes_nonzero_fails_at_window_end(self) -> None:
        ranker = self._ranker_with_grads(1.0)
        ranker.score.weight.grad = torch.zeros_like(ranker.score.weight)
        qualification = training.GradientQualification.for_module(ranker, window=1)
        qualification.observe(ranker, loss=torch.tensor(0.1))
        self.assertEqual(qualification.never_nonzero(), ("score.weight",))
        self.assertFalse(qualification.qualified())
        with self.assertRaises(guards.HybridQQualificationError) as caught:
            qualification.require_qualified()
        self.assertIn("score.weight", str(caught.exception))

    def test_disconnected_parameter_fails_at_window_end(self) -> None:
        ranker = self._ranker_with_grads(1.0)
        ranker.depthwise.bias.grad = None
        qualification = training.GradientQualification.for_module(ranker, window=1)
        qualification.observe(ranker, loss=torch.tensor(0.1))
        self.assertEqual(qualification.missing_gradient_batches[0][1], ("depthwise.bias",))
        self.assertEqual(qualification.disconnected(), ("depthwise.bias",))
        with self.assertRaises(guards.HybridQQualificationError) as caught:
            qualification.require_qualified()
        self.assertIn("never received a gradient", str(caught.exception))

    def test_incomplete_window_fails(self) -> None:
        ranker = self._ranker_with_grads(1.0)
        qualification = training.GradientQualification.for_module(ranker, window=3)
        qualification.observe(ranker, loss=torch.tensor(0.1))
        with self.assertRaises(guards.HybridQQualificationError):
            qualification.require_qualified()

    def test_non_finite_loss_or_gradient_fails_closed(self) -> None:
        ranker = self._ranker_with_grads(1.0)
        qualification = training.GradientQualification.for_module(ranker, window=1)
        with self.assertRaises(guards.HybridQNumericalError):
            qualification.observe(ranker, loss=torch.tensor(float("nan")))
        ranker.reduce.weight.grad = torch.full_like(ranker.reduce.weight, float("inf"))
        with self.assertRaises(guards.HybridQNumericalError):
            qualification.observe(ranker, loss=torch.tensor(0.1))

    def test_trainable_tensor_set_drift_fails_closed(self) -> None:
        ranker = self._ranker_with_grads(1.0)
        qualification = training.GradientQualification.for_module(ranker, window=2)
        qualification.observe(ranker, loss=torch.tensor(0.1))
        ranker.score.weight.requires_grad_(False)
        with self.assertRaises(guards.HybridQQualificationError):
            qualification.observe(ranker, loss=torch.tensor(0.1))

    def test_post_step_health_checks_parameters_and_optimizer_state(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        optimizer = training.build_ranker_optimizer(ranker, frozen_modules=[frozen])
        for parameter in ranker.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        training.require_post_step_health(ranker, optimizer)

        state = optimizer.state[ranker.reduce.weight]
        state["exp_avg"] = torch.full_like(state["exp_avg"], float("inf"))
        with self.assertRaises(guards.HybridQNumericalError):
            guards.require_finite_optimizer_state(optimizer)

        with torch.no_grad():
            ranker.score.weight.fill_(float("nan"))
        with self.assertRaises(guards.HybridQNumericalError):
            guards.require_module_parameters_finite(ranker, "ranker")

    def test_gradient_clipping_uses_the_locked_global_norm(self) -> None:
        self.assertEqual(contract.GRAD_CLIP_GLOBAL_NORM, 5.0)
        ranker = self._ranker_with_grads(100.0)
        training.clip_ranker_gradients(ranker)
        total = torch.sqrt(
            sum((p.grad.detach() ** 2).sum() for p in ranker.parameters())
        )
        self.assertAlmostEqual(float(total), 5.0, places=4)


# ===========================================================================
# Contract binding
# ===========================================================================


class ContractBindingCheck(unittest.TestCase):
    def test_perception_lock_is_readable_and_agrees(self) -> None:
        lock = contract.load_perception_lock()
        self.assertEqual(tuple(lock["architecture"]["split_shape"]), (256, 112, 192))
        self.assertEqual(
            lock["base_checkpoint"]["sha256"], contract.FROZEN_CHECKPOINT_SHA256
        )
        self.assertIn(
            "hybrid-q or ROI transport at frozen C2 Z", lock["permitted_next_changes"]
        )
        self.assertEqual(
            contract.SPLIT_CELLS * 256 * 4, contract.RAW_FP32_REFERENCE_BYTES
        )
        self.assertEqual(contract.mask_byte_count(), 2688)


if __name__ == "__main__":
    unittest.main()
