"""Exact acceptance/rejection equivalence of the lookup-table ``_mask`` against the original ``np.unique`` check.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m unittest -v rl_agent.splitfusion_gt_mask_lookup_v1.test_equivalence
"""
import os
from pathlib import Path
import random
import unittest

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C

LIVE_SETS = ({0, 1, 2, 255}, {0, 255})
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(os.environ.get("MASK_LOOKUP_ARCHIVE", ROOT.parent / (
    "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
    "20261006_run5_pilot_live_v1/valid_b1000_02/capture/cells/a71__favorable_stable/numeric_gt/service/archive")))


def original_mask(raw, labels):            # byte-exact body of the original C._mask
    C.require(type(raw) is bytes and len(raw) == C.MASK_BYTES, "exact immutable uint8 mask bytes required")
    C.require(set(np.unique(np.frombuffer(raw, dtype=np.uint8))).issubset(labels), "illegal target labels")


def outcome(function, raw, labels):
    try:
        function(raw, labels)
        return ("ACCEPT", None)
    except Exception as error:               # compare exception type AND message
        return (type(error).__name__, str(error))


class EquivalenceTest(unittest.TestCase):
    def same(self, raw, labels):
        self.assertEqual(outcome(C._mask, raw, labels), outcome(original_mask, raw, labels))
        return outcome(C._mask, raw, labels)[0]

    def test_every_byte_value_single_pixel_and_uniform(self):
        rng = random.Random(1)
        label_sets = list(LIVE_SETS) + [set(), {0}, set(range(256)), {255}, {1, 2}, set(range(0, 256, 3))]
        counts = {"ACCEPT": 0, "ContractError": 0}
        for labels in label_sets:
            for value in range(256):
                base = bytearray(C.MASK_BYTES)
                base[rng.randrange(C.MASK_BYTES)] = value
                for raw in (bytes(base), bytes([value]) * C.MASK_BYTES):
                    result = self.same(raw, labels)
                    counts[result] = counts.get(result, 0) + 1
        self.assertGreater(counts["ACCEPT"], 0); self.assertGreater(counts.get("ContractError", 0), 0)

    def test_random_mixtures_including_invalid_values(self):
        rng = np.random.default_rng(7)
        for i in range(400):
            labels = LIVE_SETS[i % 2]
            pool = np.array(sorted(labels) + ([int(rng.integers(0, 256))] if i % 3 == 0 else []), dtype=np.uint8)
            raw = rng.choice(pool, size=C.MASK_BYTES).astype(np.uint8).tobytes()
            self.same(raw, labels)

    def test_type_and_length_checks_preserved(self):
        good = bytes(C.MASK_BYTES)
        for raw in (bytearray(good), memoryview(good), np.zeros(C.MASK_BYTES, np.uint8), good[:-1], good + b"\0", b"",
                    "x" * C.MASK_BYTES, None):
            self.assertEqual(self.same(raw, {0, 255}), "ContractError")
        bad_and_short = b"\x07" * (C.MASK_BYTES - 1)     # length failure must win over the label failure
        self.assertEqual(outcome(C._mask, bad_and_short, {0}), outcome(original_mask, bad_and_short, {0}))
        self.assertIn("exact immutable", outcome(C._mask, bad_and_short, {0})[1])

    def test_label_container_forms_and_table_is_read_only(self):
        raw = bytes([0, 1, 2, 255]) * (C.MASK_BYTES // 4)
        for labels in ({0, 1, 2, 255}, frozenset({0, 1, 2, 255}), [0, 1, 2, 255], (255, 2, 1, 0), {0, 1, 2}, [0, 255]):
            self.same(raw, labels)
        table = C._label_table({0, 255})
        self.assertFalse(table.flags.writeable)
        with self.assertRaises(ValueError):  # numpy read-only
            table[3] = True
        self.assertEqual(int(table.sum()), 2)

    @unittest.skipUnless(ARCHIVE.is_dir(), "retained valid_02 archive not present")
    def test_all_retained_live_masks_accept_identically(self):
        segs = sorted(ARCHIVE.glob("frame_*/scorer/segmentation.u8"))
        ignores = sorted(ARCHIVE.glob("frame_*/scorer/object_ignore.u8"))
        self.assertEqual((len(segs), len(ignores)), (493, 493))
        for path in segs:
            self.assertEqual(self.same(path.read_bytes(), {0, 1, 2, 255}), "ACCEPT")
        for path in ignores:
            self.assertEqual(self.same(path.read_bytes(), {0, 255}), "ACCEPT")
            corrupted = bytearray(path.read_bytes()); corrupted[len(corrupted) // 2] = 1   # one illegal ignore label
            self.assertEqual(self.same(bytes(corrupted), {0, 255}), "ContractError")


if __name__ == "__main__":
    unittest.main()
