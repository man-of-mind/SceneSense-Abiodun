"""One focused CPU test for all reversible candidate layouts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import guards
from .. import run_layout_feasibility as runner
from ..layout_transforms import Layout, ValueBlockPlan, inverse, transform


class LayoutRoundTripTest(unittest.TestCase):
    def test_channel_major_and_modular_delta_are_reversible_at_all_widths(self) -> None:
        # Per-channel sequences deliberately wrap at every registered modulus.
        for bits in (8, 6, 4):
            modulus = 1 << bits
            plan = ValueBlockPlan(value_offset=7, keep_count=5, channels=3, bit_width=bits)
            symbols = np.asarray(
                [[modulus - 2, 1, modulus - 1], [1, modulus - 1, 0], [modulus - 1, 0, 2], [0, 2, modulus - 2], [2, modulus - 2, 1]],
                dtype=np.uint8,
            )
            from ..layout_transforms import _pack  # private only to construct synthetic wire bytes
            inner = b"prefix!" + _pack(symbols, plan)
            for layout in (Layout.CHANNEL_MAJOR, Layout.CHANNEL_MAJOR_MODULAR_DELTA):
                transformed = transform(inner, plan, layout)
                self.assertEqual(transformed[: plan.value_offset], inner[: plan.value_offset])
                self.assertEqual(inverse(transformed, plan, layout), inner)


class OutputPathCreationTest(unittest.TestCase):
    def test_create_only_leaf_refuses_reuse_and_inputs_remain_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            relative = "experiments/new_namespace/previously_nonexistent_leaf"
            created = runner._create_output_directory(relative, repository_root=root)
            self.assertEqual(
                created,
                (root / "experiments/new_namespace/previously_nonexistent_leaf").resolve(strict=True),
            )
            with self.assertRaises(guards.HybridQConfigError):
                runner._create_output_directory(relative, repository_root=root)
            with self.assertRaises(FileNotFoundError):
                runner._root_path("experiments/missing_input.json", repository_root=root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
