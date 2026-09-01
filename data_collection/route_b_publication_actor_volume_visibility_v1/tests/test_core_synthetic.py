"""Synthetic checks for the actor-volume visibility geometry.

Run with:  python3 -m unittest discover -s data_collection/route_b_publication_actor_volume_visibility_v1/tests -t .
These use hand-built arrays only; no dataset, model, CARLA or CUDA involvement.
"""

from __future__ import annotations

import math

import unittest

import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import core, scoring


WIDTH, HEIGHT = 200, 200
FX = FY = 100.0
CX, CY = 100.0, 100.0
INTRINSICS = np.asarray([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
# Camera at the world origin, sensor axes aligned with world axes.
CAMERA_MATRIX = np.eye(4)
CAMERA_INVERSE = np.eye(4)


def _render_depth(planes):
    """Paint a depth image from (row0, row1, col0, col1, depth) rectangles."""
    depth = np.full((HEIGHT, WIDTH), 900.0, dtype=np.float64)
    for row0, row1, col0, col1, value in planes:
        depth[row0:row1, col0:col1] = value
    return depth


def _pedestrian(key, centre, extent=(0.25, 0.25, 0.9), yaw=0.0):
    return {"key": key, "centre": centre, "extent": extent, "yaw_deg": yaw}


class ActorVolumeVisibilityGeometry(unittest.TestCase):
    def test_band_edges_are_half_open_with_closed_top(self) -> None:
        assert core.band_for_score(0.0) == core.BAND_NOT_OBSERVABLE
        assert core.band_for_score(0.199999) == core.BAND_NOT_OBSERVABLE
        assert core.band_for_score(0.20) == core.BAND_HEAVY
        assert core.band_for_score(0.649999) == core.BAND_HEAVY
        assert core.band_for_score(0.65) == core.BAND_PARTIAL
        assert core.band_for_score(0.899999) == core.BAND_PARTIAL
        assert core.band_for_score(0.90) == core.BAND_BARE
        assert core.band_for_score(1.0) == core.BAND_BARE


    def test_clamp_unit_accepts_roundoff_and_rejects_real_excursions(self) -> None:
        assert core.clamp_unit(1.0 + 1e-12) == 1.0
        assert core.clamp_unit(-1e-12) == 0.0
        with self.assertRaises(ValueError):
            core.clamp_unit(1.01)
        with self.assertRaises(ValueError):
            core.clamp_unit(float("nan"))


    def test_projection_back_projection_round_trip_is_exact(self) -> None:
        rng = np.random.default_rng(20260901)
        depth = rng.uniform(2.0, 40.0, size=(HEIGHT, WIDTH))
        bounds = (40, 60, 30, 70)
        roi = core.back_project_roi(depth, bounds, CAMERA_MATRIX, INTRINSICS)
        u, v, _ = core.project_points(roi["world"], CAMERA_INVERSE, INTRINSICS)
        assert np.max(np.abs(u - roi["u"])) < 1e-9
        assert np.max(np.abs(v - roi["v"])) < 1e-9


    def test_oriented_box_rotation_matches_manual_yaw(self) -> None:
        corners = core.oriented_box_corners((10.0, 0.0, 1.0), (2.0, 0.5, 0.9), 90.0)
        # A 90 degree yaw swaps the world x and y footprint of the box.
        assert math.isclose(float(np.ptp(corners[:, 0])), 1.0, abs_tol=1e-9)
        assert math.isclose(float(np.ptp(corners[:, 1])), 4.0, abs_tol=1e-9)
        assert math.isclose(float(np.ptp(corners[:, 2])), 1.8, abs_tol=1e-9)


    def test_ground_plane_pixels_are_rejected_but_body_pixels_are_kept(self) -> None:
        centre = (10.0, 0.0, 1.0)
        extent = (0.25, 0.25, 0.9)
        local = np.asarray(
            [
                [0.0, 0.0, 0.0],       # body centre
                [0.0, 0.0, -0.9],      # exactly the box bottom plane -> ground
                [0.0, 0.0, -0.88],     # 0.02 m above the bottom -> still ground
                [0.0, 0.0, -0.86],     # 0.04 m above the bottom -> body
                [0.0, 0.0, 0.94],      # inside the 0.05 m tolerance above the top
                [0.0, 0.0, 0.96],      # beyond the tolerance
            ]
        )
        retained, in_volume = core.inside_actor_volume(local, extent)
        assert in_volume.tolist() == [True, True, True, True, True, False]
        assert retained.tolist() == [True, False, False, True, True, False]


    def test_road_pixels_inside_the_depth_interval_do_not_become_actor_support(self) -> None:
        """The failure mode the new metric exists to remove."""
        centre = (10.0, 0.0, 0.0)
        extent = (0.25, 0.25, 0.9)
        # Camera at world origin looking down +x; the actor spans x in [9.75, 10.25].
        # Paint a wide band of road at 10.0 m depth, i.e. inside the actor's global
        # near/far depth interval, but well below and beside the actor volume.
        depth = _render_depth([(100, 160, 60, 140, 10.0)])
        boxes = core.projected_boxes(
            core.oriented_box_corners(centre, extent, 0.0),
            CAMERA_INVERSE,
            INTRINSICS,
            width=WIDTH,
            height=HEIGHT,
        )
        bounds = core.roi_pixel_bounds(boxes, width=WIDTH, height=HEIGHT)
        roi = core.back_project_roi(depth, bounds, CAMERA_MATRIX, INTRINSICS)
        local = core.actor_local_points(roi["world"], centre, 0.0)
        retained, in_volume = core.inside_actor_volume(local, extent)
        # The old-style depth-interval test would accept every one of these pixels.
        interval = (roi["depth_m"] >= 9.75 - 0.25) & (roi["depth_m"] <= 10.25 + 0.25)
        assert int(np.count_nonzero(interval)) > int(np.count_nonzero(retained))
        # Nothing retained may sit at or below the box bottom plus the margin.
        assert np.all(local[retained, 2] > -extent[2] + core.GROUND_REJECT_MARGIN_M)
        assert np.all(np.abs(local[retained]) <= np.asarray(extent) + core.ACTOR_VOLUME_TOLERANCE_M)


    def test_competing_pedestrians_split_shared_points_deterministically(self) -> None:
        near = _pedestrian("a", (10.0, 0.0, 0.0))
        far = _pedestrian("b", (10.4, 0.0, 0.0))
        # A point 0.1 m in front of `a`'s centre is inside both volumes (they
        # overlap within the 0.05 m tolerance band) but closer, in normalised
        # actor-local terms, to `a`.
        shared = np.asarray([[10.15, 0.0, 0.0]])
        to_a = core.assign_competing_pedestrians(shared, "a", [near, far])
        to_b = core.assign_competing_pedestrians(shared, "b", [near, far])
        assert to_a["target_contains"].tolist() == [True]
        assert to_b["target_contains"].tolist() == [True]
        assert to_a["owned"].tolist() == [True]
        assert to_b["owned"].tolist() == [False]
        assert to_a["competing_actor_boxes"] == 1
        # The rule is independent of candidate ordering.
        reversed_order = core.assign_competing_pedestrians(shared, "a", [far, near])
        assert reversed_order["owned"].tolist() == to_a["owned"].tolist()


    def test_fully_visible_actor_scores_near_one_and_half_occluded_actor_drops(self) -> None:
        centre = (10.0, 0.0, 0.0)
        extent = (0.25, 0.25, 0.9)
        boxes = core.projected_boxes(
            core.oriented_box_corners(centre, extent, 0.0),
            CAMERA_INVERSE,
            INTRINSICS,
            width=WIDTH,
            height=HEIGHT,
        )
        row0, row1, col0, col1 = core.roi_pixel_bounds(boxes, width=WIDTH, height=HEIGHT)
        # Whole projected box at the actor's own surface depth.
        full = _render_depth([(row0, row1, col0, col1, 9.8)])
        pedestrians = [_pedestrian("target", centre, extent)]
        clear = scoring.score_actor_frame(
            depth_m=full, camera_matrix=CAMERA_MATRIX, camera_inverse=CAMERA_INVERSE,
            intrinsics=INTRINSICS, width=WIDTH, height=HEIGHT, target_key="target",
            target_centre=centre, target_extent=extent, target_yaw_deg=0.0,
            pedestrian_boxes=pedestrians,
        )
        assert clear["visibility"] > 0.95
        assert clear["visibility_band"] == core.BAND_BARE
        assert clear["no_support"] is False

        # Occlude the lower half with a closer surface: the visible box shrinks.
        half_row = (row0 + row1) // 2
        occluded = _render_depth(
            [(row0, half_row, col0, col1, 9.8), (half_row, row1, col0, col1, 4.0)]
        )
        partial = scoring.score_actor_frame(
            depth_m=occluded, camera_matrix=CAMERA_MATRIX, camera_inverse=CAMERA_INVERSE,
            intrinsics=INTRINSICS, width=WIDTH, height=HEIGHT, target_key="target",
            target_centre=centre, target_extent=extent, target_yaw_deg=0.0,
            pedestrian_boxes=pedestrians,
        )
        assert partial["visibility"] < clear["visibility"]
        assert partial["visible_box_height_ratio"] < clear["visible_box_height_ratio"]


    def test_no_support_case_reports_zero_and_flags_itself(self) -> None:
        centre = (10.0, 0.0, 0.0)
        extent = (0.25, 0.25, 0.9)
        blocked = _render_depth([(0, HEIGHT, 0, WIDTH, 3.0)])
        result = scoring.score_actor_frame(
            depth_m=blocked, camera_matrix=CAMERA_MATRIX, camera_inverse=CAMERA_INVERSE,
            intrinsics=INTRINSICS, width=WIDTH, height=HEIGHT, target_key="target",
            target_centre=centre, target_extent=extent, target_yaw_deg=0.0,
            pedestrian_boxes=[_pedestrian("target", centre, extent)],
        )
        assert result["retained_actor_point_count"] == 0
        assert result["no_support"] is True
        assert result["visibility"] == 0.0
        assert result["visibility_band"] == core.BAND_NOT_OBSERVABLE


    def test_visible_box_is_a_sub_box_of_the_full_clipped_box(self) -> None:
        """Outward ROI rasterisation must not push the ratio above one."""
        centre = (10.0, 0.0, 0.0)
        extent = (0.25, 0.25, 0.9)
        boxes = core.projected_boxes(
            core.oriented_box_corners(centre, extent, 0.0),
            CAMERA_INVERSE,
            INTRINSICS,
            width=WIDTH,
            height=HEIGHT,
        )
        row0, row1, col0, col1 = core.roi_pixel_bounds(boxes, width=WIDTH, height=HEIGHT)
        # Every ROI pixel is actor surface, so the tight pixel-extent box is the
        # whole ROI, which is strictly larger than the continuous clipped box.
        full = _render_depth([(row0, row1, col0, col1, 9.8)])
        result = scoring.score_actor_frame(
            depth_m=full, camera_matrix=CAMERA_MATRIX, camera_inverse=CAMERA_INVERSE,
            intrinsics=INTRINSICS, width=WIDTH, height=HEIGHT, target_key="target",
            target_centre=centre, target_extent=extent, target_yaw_deg=0.0,
            pedestrian_boxes=[_pedestrian("target", centre, extent)],
        )
        assert result["visible_box_raster_ratio"] > 1.0
        assert result["visibility"] <= 1.0
        assert result["degenerate_visible_box"] is False
        assert result["visible_bbox_x"] >= result["clipped_bbox_x"] - 1e-12
        assert (
            result["visible_bbox_x"] + result["visible_bbox_w"]
            <= result["clipped_bbox_x"] + result["clipped_bbox_w"] + 1e-12
        )

    def test_box_intersection_handles_disjoint_and_nested_cases(self) -> None:
        assert core.intersect_boxes((0, 0, 10, 10), (20, 20, 5, 5))[2:] == (0.0, 0.0)
        assert core.intersect_boxes((0, 0, 10, 10), (2, 3, 4, 5)) == (2.0, 3.0, 4.0, 5.0)
        assert core.intersect_boxes((0.0, 0.0, 4.0, 4.0), (-1.0, -1.0, 6.0, 6.0)) == (
            0.0, 0.0, 4.0, 4.0,
        )

    def test_truncation_is_independent_of_occlusion(self) -> None:
        assert core.truncation_from_boxes(100.0, 100.0) == 0.0
        assert math.isclose(core.truncation_from_boxes(40.0, 100.0), 0.6)


if __name__ == "__main__":
    unittest.main()
