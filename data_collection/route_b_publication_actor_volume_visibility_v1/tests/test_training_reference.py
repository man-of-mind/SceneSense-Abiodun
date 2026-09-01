"""Synthetic checks for the training-only expected-clear-support reference.

Run with:  CUDA_VISIBLE_DEVICES="" python3 -m unittest discover \
             -s data_collection/route_b_publication_actor_volume_visibility_v1/tests -t .
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import (
    training_reference as tref,
)


def _records(spec):
    """spec: iterable of (actor_type, angle_bin, height_bin, [densities])."""
    out = []
    for actor_type, angle, height, densities in spec:
        for density in densities:
            out.append(
                {
                    "actor_type": actor_type,
                    "angle_bin": angle,
                    "height_bin": height,
                    "support_density": density,
                }
            )
    return out


class FoldedViewAngle(unittest.TestCase):
    def test_head_on_and_directly_away_both_fold_to_zero(self) -> None:
        camera = (0.0, 0.0, 1.5)
        centre = (10.0, 0.0, 1.0)
        # Facing +x is directly away from a camera at the origin looking down +x.
        self.assertAlmostEqual(tref.folded_view_angle_deg(centre, 0.0, camera), 0.0, places=9)
        # Facing -x is head-on; the fold makes it identical.
        self.assertAlmostEqual(tref.folded_view_angle_deg(centre, 180.0, camera), 0.0, places=9)

    def test_profile_is_ninety_degrees(self) -> None:
        camera = (0.0, 0.0, 1.5)
        centre = (10.0, 0.0, 1.0)
        self.assertAlmostEqual(tref.folded_view_angle_deg(centre, 90.0, camera), 90.0, places=9)
        self.assertAlmostEqual(tref.folded_view_angle_deg(centre, -90.0, camera), 90.0, places=9)

    def test_angle_is_always_inside_the_folded_range(self) -> None:
        rng = np.random.default_rng(20260901)
        for _ in range(500):
            centre = (rng.uniform(-40, 40), rng.uniform(-40, 40), 1.0)
            camera = (rng.uniform(-5, 5), rng.uniform(-5, 5), 1.5)
            if math.hypot(centre[0] - camera[0], centre[1] - camera[1]) < 1e-6:
                continue
            angle = tref.folded_view_angle_deg(centre, rng.uniform(-360, 360), camera)
            self.assertGreaterEqual(angle, 0.0)
            self.assertLessEqual(angle, 90.0)
            tref.angle_bin(angle)  # must be bindable

    def test_coincident_actor_and_camera_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tref.folded_view_angle_deg((1.0, 2.0, 1.0), 0.0, (1.0, 2.0, 1.5))


class BinEdges(unittest.TestCase):
    def test_angle_bins_are_half_open_with_closed_top(self) -> None:
        self.assertEqual(tref.angle_bin(0.0), "a00_30")
        self.assertEqual(tref.angle_bin(29.999), "a00_30")
        self.assertEqual(tref.angle_bin(30.0), "a30_60")
        self.assertEqual(tref.angle_bin(59.999), "a30_60")
        self.assertEqual(tref.angle_bin(60.0), "a60_90")
        self.assertEqual(tref.angle_bin(90.0), "a60_90")
        with self.assertRaises(ValueError):
            tref.angle_bin(90.001)

    def test_height_bins_match_the_locked_edges(self) -> None:
        self.assertEqual(tref.height_bin(0.0), "h_lt24")
        self.assertEqual(tref.height_bin(23.999), "h_lt24")
        self.assertEqual(tref.height_bin(24.0), "h24_48")
        self.assertEqual(tref.height_bin(47.999), "h24_48")
        self.assertEqual(tref.height_bin(48.0), "h48_96")
        self.assertEqual(tref.height_bin(95.999), "h48_96")
        self.assertEqual(tref.height_bin(96.0), "h_ge96")
        self.assertEqual(tref.height_bin(10000.0), "h_ge96")


class SupportDensity(unittest.TestCase):
    def test_density_is_pixels_over_clipped_projected_area(self) -> None:
        self.assertAlmostEqual(tref.support_density(50, 200.0), 0.25)

    def test_non_positive_area_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tref.support_density(50, 0.0)


class ReferenceEstimator(unittest.TestCase):
    def test_percentile_uses_method_higher(self) -> None:
        values = [float(v) for v in range(1, 21)]  # 1..20
        reference = tref.build_reference(
            _records([("t", "a00_30", "h48_96", values)])
        )
        expected = float(np.percentile(np.asarray(values), 95.0, method="higher"))
        got = reference["tables"]["type_angle_height"]["t|a00_30|h48_96"][
            "expected_clear_support_density"
        ]
        self.assertEqual(got, expected)
        # `higher` must land on an actual observation, not an interpolation.
        self.assertIn(got, values)

    def test_fallback_hierarchy_respects_the_locked_counts(self) -> None:
        # 40 samples for type A (below 50) but 120 in the angle+height cell.
        spec = [
            ("A", "a00_30", "h48_96", [0.30] * 40),
            ("B", "a00_30", "h48_96", [0.50] * 80),
            ("C", "a30_60", "h_lt24", [0.10] * 30),
        ]
        reference = tref.build_reference(_records(spec))
        # Type A: its own cell has n=40 < 50, so it falls back to angle+height
        # (n = 120 >= 100).
        value, tier, n, key = tref.lookup(reference, "A", "a00_30", "h48_96")
        self.assertEqual(tier, tref.TIER_ANGLE_HEIGHT)
        self.assertEqual(n, 120)
        self.assertEqual(key, "a00_30|h48_96")
        # Type B: its own cell has n=80 >= 50, so it resolves at the finest tier.
        value, tier, n, key = tref.lookup(reference, "B", "a00_30", "h48_96")
        self.assertEqual(tier, tref.TIER_TYPE_ANGLE_HEIGHT)
        self.assertEqual(n, 80)
        self.assertEqual(value, 0.50)
        # Type C: 30 in its own cell, 30 in angle+height, 30 in height -> global.
        value, tier, n, key = tref.lookup(reference, "C", "a30_60", "h_lt24")
        self.assertEqual(tier, tref.TIER_GLOBAL)
        self.assertEqual(n, 150)

    def test_unseen_actor_type_falls_back_rather_than_failing(self) -> None:
        reference = tref.build_reference(
            _records([("A", "a00_30", "h48_96", [0.4] * 150)])
        )
        value, tier, n, _key = tref.lookup(reference, "never_seen", "a00_30", "h48_96")
        self.assertEqual(tier, tref.TIER_ANGLE_HEIGHT)
        self.assertEqual(n, 150)
        self.assertGreater(value, 0.0)

    def test_every_tier_table_is_populated(self) -> None:
        reference = tref.build_reference(
            _records([("A", "a00_30", "h48_96", [0.4] * 10), ("B", "a60_90", "h_lt24", [0.1] * 10)])
        )
        for tier in tref.TIER_ORDER:
            self.assertIn(tier, reference["tables"])
        self.assertEqual(reference["tables"]["global"]["global"]["n"], 20)
        self.assertEqual(reference["total_records"], 20)

    def test_zero_support_records_are_counted_not_dropped(self) -> None:
        reference = tref.build_reference(
            _records([("A", "a00_30", "h48_96", [0.0] * 5 + [0.4] * 5)])
        )
        cell = reference["tables"]["type_angle_height"]["A|a00_30|h48_96"]
        self.assertEqual(cell["n"], 10)
        self.assertEqual(cell["zero_support_count"], 5)

    def test_empty_record_set_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tref.build_reference([])


class NormalizedScore(unittest.TestCase):
    def test_clamps_above_one_and_preserves_the_ratio_below(self) -> None:
        self.assertEqual(tref.normalized_visibility(0.6, 0.4), 1.0)
        self.assertAlmostEqual(tref.normalized_visibility(0.2, 0.4), 0.5)
        self.assertEqual(tref.normalized_visibility(0.0, 0.4), 0.0)

    def test_non_positive_or_non_finite_expected_is_rejected(self) -> None:
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tref.normalized_visibility(0.2, bad)


if __name__ == "__main__":
    unittest.main()
