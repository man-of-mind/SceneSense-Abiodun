"""CPU-only continuous-q interface checks. No real data, no checkpoint, no CUDA.

Two focused cases only: byte-for-byte parity with the discrete path at every
registered q, and one arbitrary q (0.2345) executed end to end. Nothing here
trains, loads a checkpoint or reads validation/test data; the ranker is the
deterministic contract-seed construction already used by the synthetic suite.
"""

from __future__ import annotations

import functools
import struct
import unittest

import torch

from .. import codec, continuous_q, contract, guards
from ..ranker import build_ranker
from ..selection import select_and_apply, select_cells


ARBITRARY_Q = 0.2345
ARBITRARY_KEEP = 16461


@functools.lru_cache(maxsize=1)
def frozen_c2() -> torch.Tensor:
    generator = torch.Generator().manual_seed(contract.RANKER_INIT_SEED)
    return torch.randn(*contract.SPLIT_SHAPE, generator=generator, dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def cpu_ranker():
    return build_ranker().eval()


def _header_q_e4(payload: codec.SparsePayload) -> int:
    return struct.unpack(codec.HEADER_FORMAT, payload.data[: codec.HEADER_BYTES])[8]


class ContinuousQRegisteredParityTest(unittest.TestCase):
    """Every registered q must be bit-identical through the new interface."""

    def test_registered_q_is_byte_identical_to_the_discrete_path(self) -> None:
        c2 = frozen_c2()
        ranker = cpu_ranker()
        with torch.no_grad():
            for value in contract.REGISTERED_Q_VALUES:
                with self.subTest(q=value):
                    # Discrete path, exactly as the registered runners call it.
                    discrete_masked, discrete_selection = select_and_apply(
                        c2, ranker, value
                    )
                    if discrete_selection is None:
                        discrete_payload = codec.encode(discrete_masked, value)
                    else:
                        discrete_payload = codec.encode(
                            discrete_masked, value, discrete_selection
                        )
                    discrete_decoded, discrete_q = codec.decode(discrete_payload)

                    result = continuous_q.transport(c2, ranker, value)
                    continuous_decoded, continuous_q_out = continuous_q.decode(
                        result.payload
                    )

                    self.assertTrue(result.plan.is_registered)
                    self.assertEqual(result.plan.wire_q, value)
                    self.assertEqual(
                        result.plan.keep_count, contract.keep_count(value)
                    )
                    self.assertEqual(
                        result.plan.is_bypass, discrete_selection is None
                    )

                    if discrete_selection is None:
                        self.assertIsNone(result.selection)
                        # q=0 bypass returns the input object itself.
                        self.assertIs(result.masked, c2)
                    else:
                        self.assertEqual(
                            int(result.selection.keep_count),
                            int(discrete_selection.keep_count),
                        )
                        self.assertTrue(
                            torch.equal(
                                result.selection.keep_indices,
                                discrete_selection.keep_indices,
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                result.selection.keep_mask,
                                discrete_selection.keep_mask,
                            )
                        )

                    self.assertEqual(
                        result.masked.numpy().tobytes(),
                        discrete_masked.numpy().tobytes(),
                    )
                    self.assertEqual(result.payload.data, discrete_payload.data)
                    self.assertEqual(
                        result.payload.total_bytes, discrete_payload.total_bytes
                    )
                    self.assertEqual(continuous_q_out, discrete_q)
                    self.assertTrue(
                        torch.equal(continuous_decoded, discrete_decoded)
                    )


class ContinuousQArbitraryValueTest(unittest.TestCase):
    """q=0.2345 must execute as itself, not as a registered anchor."""

    def test_arbitrary_q_round_trips_without_snapping(self) -> None:
        c2 = frozen_c2()
        ranker = cpu_ranker()

        plan = continuous_q.quantize_q(ARBITRARY_Q)
        self.assertEqual(plan.requested_q, ARBITRARY_Q)
        self.assertEqual(plan.wire_q, ARBITRARY_Q)
        self.assertEqual(plan.q_e4, 2345)
        self.assertEqual(plan.keep_count, ARBITRARY_KEEP)
        self.assertEqual(plan.drop_count, contract.SPLIT_CELLS - ARBITRARY_KEEP)
        self.assertFalse(plan.is_registered)
        self.assertFalse(plan.is_bypass)
        self.assertFalse(plan.snapped)
        self.assertNotEqual(plan.wire_q, contract.snap_continuous_q(ARBITRARY_Q))

        with torch.no_grad():
            result = continuous_q.transport(c2, ranker, ARBITRARY_Q)
            decoded, decoded_q = continuous_q.decode(result.payload)
            anchor = select_cells(ranker.score_cells(c2), 0.30)

        self.assertEqual(decoded_q, ARBITRARY_Q)
        self.assertEqual(result.payload.q, ARBITRARY_Q)
        self.assertEqual(_header_q_e4(result.payload), 2345)
        self.assertEqual(int(result.payload.keep_count), ARBITRARY_KEEP)
        self.assertEqual(int(result.selection.keep_count), ARBITRARY_KEEP)

        # Neither q=0 (dense) nor the q=0.30 anchor: strictly between them, and
        # a superset of the anchor's cells because the ordering is q-independent.
        self.assertNotEqual(ARBITRARY_KEEP, contract.SPLIT_CELLS)
        self.assertNotEqual(ARBITRARY_KEEP, contract.keep_count(0.30))
        self.assertLess(int(anchor.keep_mask.sum()), ARBITRARY_KEEP)
        self.assertEqual(
            int((anchor.keep_mask & ~result.selection.keep_mask).sum()), 0
        )

        # Exact wire semantics: retained values bit-exact, dropped cells zero.
        mask = result.selection.keep_mask.unsqueeze(0).expand_as(c2)
        self.assertTrue(torch.equal(decoded[mask], c2[mask]))
        self.assertTrue(bool((decoded[~mask] == 0).all()))
        self.assertEqual(
            result.payload.total_bytes,
            codec.HEADER_BYTES
            + contract.mask_byte_count()
            + ARBITRARY_KEEP * contract.SPLIT_CHANNELS * 4,
        )

        # Fails closed on an out-of-range q and on a truncated payload.
        for bad in (-1e-4, 0.9801, 1.0, float("nan"), float("inf")):
            with self.assertRaises(guards.HybridQConfigError):
                continuous_q.quantize_q(bad)
        with self.assertRaises(guards.HybridQPayloadError):
            continuous_q.decode(result.payload.data[:-4])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
