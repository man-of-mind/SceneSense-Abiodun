"""CPU-only synthetic checks. No real data, no checkpoint, no CUDA, no CARLA."""

from __future__ import annotations

import struct
import unittest

import torch
from torch import nn

from .. import codec, contract, guards, training
from ..ranker import SpatialRanker, build_ranker
from ..selection import CellSelection, apply_selection, select_and_apply, select_cells


torch.manual_seed(0)


class ExplodingRanker(nn.Module):
    """Fails loudly if the q=0 path ever invokes ranking."""

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise AssertionError("ranker must be bypassed at q=0")


class FrozenSyntheticPerception(nn.Module):
    """Stand-in for the frozen trunk: small, and entirely non-trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(7, 4, kernel_size=1)
        self.head = nn.Conv2d(4, 2, kernel_size=1)
        for parameter in self.parameters():
            parameter.requires_grad_(False)


def _c2(channels: int = 8, height: int = 4, width: int = 6) -> torch.Tensor:
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
        for q, keep in expected.items():
            self.assertEqual(contract.keep_count(q), keep)
            self.assertEqual(contract.drop_count(q), 21504 - keep)
            scores = torch.randn(contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH)
            selection = select_cells(scores, q)
            self.assertEqual(selection.keep_count, keep)
            self.assertEqual(int(selection.keep_indices.numel()), keep)
            self.assertEqual(int(selection.keep_mask.sum()), keep)

    def test_invalid_q_fails_closed(self) -> None:
        for bad in (-0.1, 1.0, 1.5, float("nan"), 0.25, "0.5", None):
            with self.assertRaises(guards.HybridQConfigError):
                guards.require_valid_q(bad)
        # Unregistered but numerically sane q is allowed only when opted into.
        self.assertEqual(guards.require_valid_q(0.25, registered_only=False), 0.25)


class TieBreakCheck(unittest.TestCase):
    def test_ties_prefer_lower_row_major_index(self) -> None:
        scores = torch.zeros(4, 6)
        selection = select_cells(scores, 0.5, registered_only=False)
        self.assertEqual(selection.keep_count, 12)
        self.assertEqual(selection.keep_indices.tolist(), list(range(12)))

    def test_partial_ties_are_broken_by_index(self) -> None:
        scores = torch.tensor([[5.0, 1.0, 1.0], [1.0, 1.0, 9.0], [1.0, 1.0, 1.0]])
        selection = select_cells(scores, 0.7, registered_only=False)
        # keep = 9 - floor(0.7*9+0.5) = 9 - 6 = 3 -> {5, 0} by score, then index 1.
        self.assertEqual(selection.keep_count, 3)
        self.assertEqual(selection.keep_indices.tolist(), [0, 1, 5])

    def test_selection_is_repeatable(self) -> None:
        scores = torch.randn(8, 8)
        first = select_cells(scores, 0.5, registered_only=False)
        second = select_cells(scores, 0.5, registered_only=False)
        self.assertTrue(torch.equal(first.keep_indices, second.keep_indices))


class DenseIdentityCheck(unittest.TestCase):
    def test_q_zero_bypasses_ranking_and_is_exactly_dense(self) -> None:
        c2 = _c2()
        out, selection = select_and_apply(c2, ExplodingRanker(), 0.00)
        self.assertIsNone(selection)
        self.assertIs(out, c2)
        self.assertTrue(torch.equal(out, c2))

    def test_q_zero_round_trip_is_bit_exact(self) -> None:
        c2 = _c2()
        payload = codec.encode(c2, 0.00)
        self.assertEqual(payload.mask_bytes, 0)
        decoded, q = codec.decode(payload)
        self.assertEqual(q, 0.0)
        self.assertTrue(torch.equal(decoded, c2))


class MaskingCheck(unittest.TestCase):
    def test_retained_cells_keep_all_channels_and_dropped_cells_are_zero(self) -> None:
        c2 = _c2(channels=256, height=4, width=6)
        ranker = build_ranker()
        masked, selection = select_and_apply(c2, ranker, 0.50)
        self.assertIsNotNone(selection)
        cells = 4 * 6
        self.assertEqual(selection.keep_count, contract.keep_count(0.50, cells))
        flat_in = c2.reshape(256, cells)
        flat_out = masked.reshape(256, cells)
        kept = set(selection.keep_indices.tolist())
        for cell in range(cells):
            if cell in kept:
                self.assertTrue(torch.equal(flat_out[:, cell], flat_in[:, cell]))
            else:
                self.assertTrue(torch.equal(flat_out[:, cell], torch.zeros(256)))
                self.assertEqual(float(flat_out[:, cell].abs().sum()), 0.0)


class CodecRoundTripCheck(unittest.TestCase):
    def test_sparse_round_trip_matches_masked_tensor(self) -> None:
        c2 = _c2(channels=256, height=4, width=6)
        scores = torch.randn(4, 6)
        for q in (0.30, 0.50, 0.70, 0.90, 0.98):
            selection = select_cells(scores, q, registered_only=False)
            masked = apply_selection(c2, selection)
            payload = codec.encode(c2, q, selection)
            decoded, decoded_q = codec.decode(payload)
            self.assertEqual(decoded_q, q)
            self.assertEqual(decoded.shape, c2.shape)
            self.assertEqual(decoded.dtype, torch.float32)
            self.assertTrue(torch.equal(decoded, masked))

    def test_bitmask_byte_and_bit_order_is_msb_first(self) -> None:
        # 10 cells at q=0.50 -> keep 5; 2 mask bytes with 6 padding bits.
        cells, shape = 10, (2, 5)
        selection = _selection_from_indices([0, 3, 7, 8, 9], cells, shape)
        payload = codec.encode(torch.zeros(2, 2, 5), 0.50, selection)
        mask = payload.data[codec.HEADER_BYTES : codec.HEADER_BYTES + payload.mask_bytes]
        self.assertEqual(payload.mask_bytes, 2)
        # cell i -> byte i//8, bit 7-(i%8).
        # byte 0 holds cells 0,3,7 -> 0b10010001; byte 1 holds cells 8,9 -> 0b11000000.
        self.assertEqual(tuple(mask), (0x91, 0xC0))
        self.assertEqual(codec.decode(payload)[0].shape, (2, 2, 5))

    def test_values_are_stored_in_ascending_row_major_cell_order(self) -> None:
        c2 = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        kept = [1, 2, 5, 6, 9, 10]  # 12 cells at q=0.50 -> keep 6
        selection = _selection_from_indices(kept, 12, (3, 4))
        payload = codec.encode(c2, 0.50, selection)
        values = payload.data[codec.HEADER_BYTES + payload.mask_bytes :]
        got = torch.frombuffer(bytearray(values), dtype=torch.float32).reshape(6, 2)
        flat = c2.reshape(2, 12)
        for row, cell in enumerate(kept):
            self.assertTrue(torch.equal(got[row], flat[:, cell]))

    def test_serialized_byte_counts_are_measured_and_reported(self) -> None:
        c2 = torch.randn(*contract.SPLIT_SHAPE, dtype=torch.float32)
        scores = torch.randn(contract.SPLIT_HEIGHT, contract.SPLIT_WIDTH)
        expected = {
            0.00: 22020140,
            0.30: 15417004,
            0.50: 11012780,
            0.70: 6608556,
            0.90: 2204332,
            0.98: 443052,
        }
        report = {}
        for q in contract.REGISTERED_Q_VALUES:
            if q == 0.00:
                payload = codec.encode(c2, q)
            else:
                payload = codec.encode(c2, q, select_cells(scores, q))
            self.assertEqual(payload.total_bytes, len(payload.data))
            self.assertEqual(
                payload.total_bytes,
                codec.HEADER_BYTES + payload.mask_bytes + payload.value_bytes,
            )
            self.assertEqual(payload.total_bytes, expected[q])
            report[q] = payload.total_bytes
        raw = codec.encode_dense_diagnostic_size(c2)
        self.assertEqual(raw, contract.SPLIT_PAYLOAD_FP32_BYTES)
        print("\nactual serialized hybrid-q payload bytes (dense FP32 raw = %d):" % raw)
        for q, total in report.items():
            print(f"  q={q:.2f} keep={contract.keep_count(q):5d} bytes={total:9d}")


class MalformedPayloadCheck(unittest.TestCase):
    def _payload(self) -> codec.SparsePayload:
        c2 = torch.randn(4, 4, 6)
        selection = _selection_from_indices(
            [0, 2, 3, 5, 7, 11, 15, 19, 20, 21, 22, 23], 24, (4, 6)
        )
        return codec.encode(c2, 0.50, selection)

    def test_truncated_and_corrupt_payloads_fail_closed(self) -> None:
        payload = self._payload()
        data = payload.data
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(data[: codec.HEADER_BYTES - 1])
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(data[:-4])
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(data + b"\x00\x00\x00\x00")
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(b"XXXX" + data[4:])
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(bytearray(data)[:0])
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode("not bytes")

    def test_header_field_corruption_fails_closed(self) -> None:
        payload = self._payload()
        fields = list(struct.unpack(codec.HEADER_FORMAT, payload.data[: codec.HEADER_BYTES]))
        body = payload.data[codec.HEADER_BYTES :]

        def rebuilt(index: int, value) -> bytes:
            mutated = list(fields)
            mutated[index] = value
            return struct.pack(codec.HEADER_FORMAT, *mutated) + body

        with self.assertRaises(guards.HybridQPayloadError):  # format version
            codec.decode(rebuilt(1, 99))
        with self.assertRaises(guards.HybridQPayloadError):  # dtype code
            codec.decode(rebuilt(2, 7))
        with self.assertRaises(guards.HybridQPayloadError):  # unknown flag bits
            codec.decode(rebuilt(3, 0b110))
        with self.assertRaises(guards.HybridQPayloadError):  # reserved must be zero
            codec.decode(rebuilt(4, 1))
        with self.assertRaises(guards.HybridQPayloadError):  # channel count
            codec.decode(rebuilt(5, 5))
        with self.assertRaises(guards.HybridQPayloadError):  # width -> shape mismatch
            codec.decode(rebuilt(7, 0))
        with self.assertRaises(guards.HybridQConfigError):  # unregistered q
            codec.decode(rebuilt(8, 2500))
        with self.assertRaises(guards.HybridQPayloadError):  # keep cardinality
            codec.decode(rebuilt(9, 11))
        with self.assertRaises(guards.HybridQPayloadError):  # mask length
            codec.decode(rebuilt(10, 4))
        with self.assertRaises(guards.HybridQPayloadError):  # value block length
            codec.decode(rebuilt(11, 16))

    def test_bitmask_corruption_fails_closed(self) -> None:
        payload = self._payload()
        data = bytearray(payload.data)
        mask_start = codec.HEADER_BYTES
        flipped = bytearray(data)
        flipped[mask_start] ^= 0xFF  # popcount no longer matches the header
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(bytes(flipped))

    def test_padding_bits_past_the_last_cell_fail_closed(self) -> None:
        c2 = torch.randn(2, 3, 4)  # 12 cells -> 2 mask bytes, 4 padding bits
        selection = _selection_from_indices([0, 1, 2, 3, 4, 5], 12, (3, 4))
        payload = codec.encode(c2, 0.50, selection)
        data = bytearray(payload.data)
        data[codec.HEADER_BYTES + 1] |= 0x01  # set a padding bit
        with self.assertRaises(guards.HybridQPayloadError):
            codec.decode(bytes(data))

    def test_duplicate_or_unordered_retained_indices_fail_closed(self) -> None:
        with self.assertRaises(guards.HybridQPayloadError):
            guards.require_sorted_unique_indices(torch.tensor([0, 2, 2, 5]), 24)
        with self.assertRaises(guards.HybridQPayloadError):
            guards.require_sorted_unique_indices(torch.tensor([5, 2, 9]), 24)
        with self.assertRaises(guards.HybridQPayloadError):
            guards.require_sorted_unique_indices(torch.tensor([0, 30]), 24)

    def test_encode_rejects_selection_mismatch_and_bad_dtype(self) -> None:
        c2 = torch.randn(4, 4, 6)
        with self.assertRaises(guards.HybridQConfigError):  # q>0 without a selection
            codec.encode(c2, 0.50)
        with self.assertRaises(guards.HybridQConfigError):  # q=0 with a selection
            codec.encode(c2, 0.00, _selection_from_indices(list(range(12)), 24, (4, 6)))
        with self.assertRaises(guards.HybridQPayloadError):
            codec.encode(c2.to(torch.float64), 0.00)
        with self.assertRaises(guards.HybridQNumericalError):
            bad = c2.clone()
            bad[0, 0, 0] = float("nan")
            codec.encode(bad, 0.00)

    def test_non_finite_ranker_scores_fail_closed(self) -> None:
        scores = torch.zeros(4, 6)
        scores[1, 1] = float("inf")
        with self.assertRaises(guards.HybridQNumericalError):
            select_cells(scores, 0.50, registered_only=False)


class RankerShapeCheck(unittest.TestCase):
    def test_parameter_and_mac_counts_match_the_contract(self) -> None:
        ranker = build_ranker()
        self.assertEqual(ranker.parameter_count(), 2145)
        self.assertEqual(contract.RANKER_PARAMETER_COUNT, 2145)

        # Independent recomputation from the realized module weights.
        cells = contract.SPLIT_HEIGHT * contract.SPLIT_WIDTH
        macs = 0
        for conv in (ranker.reduce, ranker.depthwise, ranker.score):
            macs += conv.weight.numel() * cells
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
        kinds = {type(module) for module in ranker.modules()}
        self.assertNotIn(nn.BatchNorm2d, kinds)
        self.assertEqual(kinds - {SpatialRanker, nn.Conv2d, nn.ReLU}, set())

    def test_scores_are_spatial_and_input_is_detached(self) -> None:
        ranker = build_ranker()
        c2 = torch.randn(256, 4, 6, requires_grad=True)
        scores = ranker.score_cells(c2)
        self.assertEqual(tuple(scores.shape), (4, 6))
        scores.sum().backward()
        self.assertIsNone(c2.grad)
        self.assertIsNotNone(ranker.reduce.weight.grad)
        batched = ranker(torch.randn(2, 256, 4, 6))
        self.assertEqual(tuple(batched.shape), (2, 4, 6))


class OptimizerOwnershipCheck(unittest.TestCase):
    def test_optimizer_contains_only_ranker_parameters(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        optimizer = training.build_ranker_optimizer(ranker, lr=1e-3, frozen_modules=[frozen])
        owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(owned, {id(p) for p in ranker.parameters()})
        self.assertEqual(len(owned), 6)  # three convs, weight + bias each
        for parameter in frozen.parameters():
            self.assertNotIn(id(parameter), owned)

    def test_trainable_frozen_stack_fails_closed(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        frozen.stem.weight.requires_grad_(True)
        with self.assertRaises(guards.HybridQOwnershipError):
            training.build_ranker_optimizer(ranker, lr=1e-3, frozen_modules=[frozen])

    def test_optimizer_ownership_guard_rejects_foreign_parameters(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        smuggled = nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.Adam([*ranker.parameters(), smuggled], lr=1e-3)
        with self.assertRaises(guards.HybridQOwnershipError):
            guards.require_optimizer_owns_only(optimizer, list(ranker.parameters()))
        guards.require_frozen_perception([frozen])


class TeacherMapCheck(unittest.TestCase):
    def test_importance_definition_and_independent_normalization(self) -> None:
        c2 = torch.tensor([[[1.0, -2.0]], [[3.0, 4.0]]])  # [2,1,2]
        grad = torch.tensor([[[2.0, 1.0]], [[-1.0, 0.5]]])
        raw = training.teacher_importance_map(c2, grad)
        self.assertTrue(torch.equal(raw, torch.tensor([[5.0, 4.0]])))

        # A task scaled by 1000x must not dominate after independent normalization.
        result = training.build_teacher_maps(
            c2, {"detect": grad, "segment": grad * 1000.0}
        )
        self.assertEqual(result.valid_tasks, ("detect", "segment"))
        self.assertEqual(result.excluded_tasks, {})
        self.assertTrue(
            torch.allclose(result.task_maps["detect"], result.task_maps["segment"])
        )
        self.assertAlmostEqual(float(result.importance.sum()), 1.0, places=6)
        self.assertAlmostEqual(result.loss_scales["segment"] / result.loss_scales["detect"], 1000.0, places=3)

    def test_absent_and_zero_tasks_are_recorded_and_excluded(self) -> None:
        c2 = torch.randn(4, 3, 5)
        grad = torch.randn(4, 3, 5)
        result = training.build_teacher_maps(
            c2,
            {
                "detect": grad,
                "segment": None,
                "geometry": torch.zeros(4, 3, 5),
                "depth": torch.full((4, 3, 5), float("nan")),
            },
        )
        self.assertEqual(result.valid_tasks, ("detect",))
        self.assertEqual(
            result.excluded_tasks,
            {"segment": "absent", "geometry": "zero_gradient", "depth": "non_finite"},
        )
        self.assertTrue(result.is_supervisable)
        self.assertNotIn("segment", result.loss_scales)

    def test_frame_with_no_valid_task_is_not_supervisable(self) -> None:
        c2 = torch.randn(4, 3, 5)
        result = training.build_teacher_maps(
            c2, {"detect": None, "segment": torch.zeros(4, 3, 5)}
        )
        self.assertIsNone(result.importance)
        self.assertFalse(result.is_supervisable)
        self.assertEqual(result.valid_tasks, ())
        self.assertEqual(
            result.excluded_tasks, {"detect": "absent", "segment": "zero_gradient"}
        )

    def test_max_normalization_and_bad_scheme(self) -> None:
        importance = torch.tensor([[1.0, 3.0]])
        self.assertTrue(
            torch.equal(
                training.normalize_importance(importance, "max"),
                torch.tensor([[1.0 / 3.0, 1.0]]),
            )
        )
        with self.assertRaises(guards.HybridQConfigError):
            training.normalize_importance(importance, "l2")

    def test_cache_record_excludes_c2(self) -> None:
        record = training.TeacherCacheRecord(
            frame_id="synthetic_0001",
            sequence_id="synthetic",
            importance=torch.rand(3, 5),
            valid_tasks=("detect",),
            excluded_tasks={"segment": "absent"},
            loss_scales={"detect": 1.5},
            normalization="l1",
        )
        self.assertEqual(
            record.perception_checkpoint_sha256, contract.FROZEN_CHECKPOINT_SHA256
        )
        self.assertNotIn("c2", record.__dataclass_fields__)
        self.assertNotIn("features", record.__dataclass_fields__)


class StraightThroughCheck(unittest.TestCase):
    def test_forward_output_equals_the_hard_mask(self) -> None:
        scores = torch.randn(4, 6, requires_grad=True)
        selection = select_cells(scores, 0.50, registered_only=False)
        mask = training.straight_through_mask(
            scores, 0.50, temperature=0.1, registered_only=False
        )
        hard = selection.keep_mask.to(scores.dtype)
        self.assertTrue(torch.equal(mask.detach(), hard))
        self.assertEqual(set(mask.detach().unique().tolist()), {0.0, 1.0})
        self.assertEqual(int(mask.detach().sum()), selection.keep_count)

    def test_gradient_flows_through_the_surrogate(self) -> None:
        ranker = build_ranker()
        c2 = torch.randn(256, 4, 6)
        scores = ranker.score_cells(c2)
        mask = training.straight_through_mask(
            scores, 0.50, temperature=0.5, registered_only=False
        )
        training.masked_c2_forward(c2, mask).sum().backward()
        self.assertIsNotNone(ranker.score.weight.grad)
        self.assertTrue(torch.isfinite(ranker.score.weight.grad).all())

    def test_temperature_must_come_from_configuration(self) -> None:
        scores = torch.randn(4, 6)
        with self.assertRaises(guards.HybridQConfigError):
            training.straight_through_mask(scores, 0.50, temperature=None, registered_only=False)
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(guards.HybridQConfigError):
                training.straight_through_mask(
                    scores, 0.50, temperature=bad, registered_only=False
                )
        with self.assertRaises(guards.HybridQConfigError):
            training.ranker_distillation_loss(scores, torch.rand(4, 6), temperature=None)


class FrozenWeightCheck(unittest.TestCase):
    def test_frozen_parameters_are_unchanged_after_an_optimizer_step(self) -> None:
        ranker = build_ranker()
        frozen = FrozenSyntheticPerception()
        snapshot = guards.snapshot_parameters(frozen)
        ranker_before = guards.snapshot_parameters(ranker)
        optimizer = training.build_ranker_optimizer(ranker, lr=1e-2, frozen_modules=[frozen])

        seven_channel = torch.randn(1, 7, 4, 6)
        c2 = torch.randn(256, 4, 6)
        teacher_grad = torch.randn(256, 4, 6)
        teacher = training.build_teacher_maps(c2, {"detect": teacher_grad})
        self.assertTrue(teacher.is_supervisable)

        optimizer.zero_grad(set_to_none=True)
        scores = ranker.score_cells(c2)
        loss = training.ranker_distillation_loss(
            scores, teacher.importance, temperature=1.0
        )
        loss.backward()
        qualification = training.GradientQualification(window=1)
        self.assertTrue(qualification.observe(list(ranker.parameters())))
        optimizer.step()

        guards.require_parameters_unchanged(frozen, snapshot)
        guards.require_frozen_perception([frozen])
        # The step must have moved the ranker, so the frozen check is not vacuous.
        self.assertTrue(
            any(
                not torch.equal(current.detach(), before)
                for current, before in zip(ranker.parameters(), ranker_before.values())
            )
        )
        # The frozen stack still produces identical outputs.
        with torch.no_grad():
            self.assertTrue(
                torch.equal(
                    frozen.head(frozen.stem(seven_channel)),
                    frozen.head(frozen.stem(seven_channel)),
                )
            )
        self.assertTrue(qualification.qualified())

    def test_isolated_zero_gradient_batch_is_logged_not_fatal(self) -> None:
        ranker = build_ranker()
        qualification = training.GradientQualification(window=2)
        for parameter in ranker.parameters():
            parameter.grad = torch.zeros_like(parameter)
        self.assertFalse(qualification.observe(list(ranker.parameters())))
        self.assertEqual(qualification.zero_gradient_batches, [1])
        self.assertFalse(qualification.qualified())
        for parameter in ranker.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.assertTrue(qualification.observe(list(ranker.parameters())))
        self.assertTrue(qualification.window_complete())
        self.assertTrue(qualification.qualified())

    def test_non_finite_gradient_fails_closed(self) -> None:
        ranker = build_ranker()
        qualification = training.GradientQualification(window=1)
        for parameter in ranker.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaises(guards.HybridQNumericalError):
            qualification.observe(list(ranker.parameters()))


class ContractBindingCheck(unittest.TestCase):
    def test_perception_lock_is_readable_and_agrees(self) -> None:
        lock = contract.load_perception_lock()
        self.assertEqual(tuple(lock["architecture"]["split_shape"]), (256, 112, 192))
        self.assertEqual(
            lock["base_checkpoint"]["sha256"], contract.FROZEN_CHECKPOINT_SHA256
        )
        self.assertIn("hybrid-q or ROI transport at frozen C2 Z", lock["permitted_next_changes"])
        self.assertEqual(contract.SPLIT_CELLS * 256 * 4, contract.SPLIT_PAYLOAD_FP32_BYTES)
        self.assertEqual(contract.mask_byte_count(), 2688)


if __name__ == "__main__":
    unittest.main()
