from __future__ import annotations

import unittest
from types import SimpleNamespace

import carla

from data_collection.phase2_naturalistic_pair_scenario import (
    NATURALISTIC_PAIR_CONTRACT_ID,
    ROUTE_FAMILIES,
    anchor_spec,
    load_source_route,
    point_to_polyline_distance_m,
    resolve_pair,
)


class Phase2NaturalisticPairTests(unittest.TestCase):
    def test_source_loops_are_hash_checked_and_bounded(self) -> None:
        for family_id, family in ROUTE_FAMILIES.items():
            route = load_source_route(family_id)
            self.assertEqual(len(route), family.source_row_count)
            self.assertLess(route[-1].distance(route[0]), 8.0)

    def test_each_pending_family_has_six_unique_geometry_only_anchors(self) -> None:
        for family in ROUTE_FAMILIES.values():
            self.assertEqual(len(family.anchors), 6)
            self.assertEqual(
                {item.anchor_id for item in family.anchors},
                {f"a{index}" for index in range(6)},
            )
            self.assertEqual(
                len({item.recipient_start_index for item in family.anchors}), 6
            )
            for item in family.anchors:
                self.assertLess(item.recipient_start_index, item.helper_start_index)

    def test_schedule_reaches_beyond_the_shared_prefix(self) -> None:
        for family in ROUTE_FAMILIES.values():
            self.assertTrue(
                any(item.recipient_start_index >= 44 for item in family.anchors)
            )

    def test_anchor_lookup_is_fail_closed(self) -> None:
        self.assertEqual(
            anchor_spec("signalized_demo_region", "a3").recipient_start_index,
            59,
        )
        with self.assertRaises(ValueError):
            anchor_spec("signalized_demo_region", "missing")
        with self.assertRaises(ValueError):
            load_source_route("missing")

    def test_contract_is_final_after_both_route_families_pass_review(self) -> None:
        self.assertEqual(
            NATURALISTIC_PAIR_CONTRACT_ID,
            "town10hd_opt_same_lane_helper_ahead_v1",
        )

    def test_cross_track_uses_route_segments_not_sparse_vertices(self) -> None:
        route = [carla.Location(x=0.0, y=0.0), carla.Location(x=4.0, y=0.0)]
        self.assertAlmostEqual(
            point_to_polyline_distance_m(2.0, 1.0, route),
            1.0,
        )
        with self.assertRaises(ValueError):
            point_to_polyline_distance_m(0.0, 0.0, [])

    def test_pair_resolution_rotates_each_role_from_its_frozen_index(self) -> None:
        class FakeMap:
            def get_waypoint(self, location, **_kwargs):
                transform = carla.Transform(
                    carla.Location(
                        x=float(location.x), y=float(location.y), z=float(location.z)
                    ),
                    carla.Rotation(yaw=0.15919755399227142),
                )
                return SimpleNamespace(
                    road_id=20,
                    section_id=0,
                    lane_id=-2,
                    is_junction=False,
                    transform=transform,
                )

        transforms, routes, contract = resolve_pair(
            FakeMap(), "signalized_demo_region", "a0"
        )
        self.assertTrue(contract["pass"])
        self.assertFalse(contract["collection_authorized"])
        self.assertEqual(contract["recipient_start_index"], 0)
        self.assertEqual(contract["helper_start_index"], 4)
        self.assertAlmostEqual(contract["along_route_separation_m"], 16.0, places=3)
        self.assertAlmostEqual(
            routes["recipient"][0].x, transforms["recipient"].location.x, places=3
        )
        self.assertAlmostEqual(
            routes["helper"][0].x, transforms["helper"].location.x, places=3
        )


if __name__ == "__main__":
    unittest.main()
