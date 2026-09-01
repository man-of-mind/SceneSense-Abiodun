"""Per-actor-frame actor-volume visibility score and its diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import core


def score_actor_frame(
    *,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    camera_inverse: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    target_key: str,
    target_centre: Sequence[float],
    target_extent: Sequence[float],
    target_yaw_deg: float,
    pedestrian_boxes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the locked visibility ratio for one target pedestrian actor-frame.

        visibility = area(B_visible) / area(B_full_clipped)

    ``B_visible`` is the tight 2D box enclosing the depth pixels whose
    back-projected 3D point lies inside the target's oriented bounding volume
    (0.05 m tolerance), sits more than 0.03 m above the box bottom plane, and is
    awarded to the target rather than to a competing pedestrian volume.

    The retained pixels are counted on the integer image grid while
    ``B_full_clipped`` is a continuous projected rectangle, and the sampled ROI
    is the outward (floor/ceil) rasterisation of that rectangle.  A tight
    pixel-extent box can therefore stick out past the projected box by up to one
    pixel per side, which for a small distant pedestrian is a large fraction of
    the box.  ``B_visible`` is consequently defined as a sub-box of
    ``B_full_clipped``: the tight pixel-extent box intersected with the clipped
    projected rectangle.  This is a fixed area convention applied identically to
    every sample, not a tuned constant, and it makes the ratio a genuine
    sub-area fraction rather than something that needs clamping.  The
    un-intersected ratio is still reported as
    ``visible_box_raster_ratio`` so the size of the effect stays visible.
    """
    corners = core.oriented_box_corners(target_centre, target_extent, target_yaw_deg)
    boxes = core.projected_boxes(
        corners, camera_inverse, intrinsics, width=width, height=height
    )
    bounds = core.roi_pixel_bounds(boxes, width=width, height=height)
    roi = core.back_project_roi(depth_m, bounds, camera_matrix, intrinsics)

    assignment = core.assign_competing_pedestrians(
        roi["world"], target_key, pedestrian_boxes
    )
    owned = assignment["owned"]
    retained_count = int(np.count_nonzero(owned))

    clipped_area = float(boxes["clipped_projected_area_px"])
    result: dict[str, Any] = {
        "algorithm_version": core.ALGORITHM_VERSION,
        "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
        "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
        **boxes,
        "sampled_roi_px": roi["roi_px"],
        "valid_depth_px": roi["valid_px"],
        "retained_actor_point_count": retained_count,
        "competing_actor_boxes": int(assignment["competing_actor_boxes"]),
        "truncation": core.truncation_from_boxes(
            clipped_area, float(boxes["unclipped_projected_area_px"])
        ),
    }

    if retained_count == 0:
        result.update(
            {
                "visibility": 0.0,
                "visibility_band": core.band_for_score(0.0),
                "no_support": True,
                "visible_bbox_x": float("nan"),
                "visible_bbox_y": float("nan"),
                "visible_bbox_w": 0.0,
                "visible_bbox_h": 0.0,
                "visible_box_area_px": 0.0,
                "visible_box_raster_ratio": 0.0,
                "degenerate_visible_box": False,
                "visible_box_height_ratio": 0.0,
                "visible_box_width_ratio": 0.0,
                "actor_point_surface_occupancy": 0.0,
                "visible_box_fill_ratio": 0.0,
                "retained_local_z_min_above_floor_m": float("nan"),
                "retained_max_abs_local_excess_m": float("nan"),
            }
        )
        return result

    rows = roi["row"][owned]
    cols = roi["col"][owned]
    raster = core.visible_box_from_pixels(rows, cols)
    clipped_box = (
        float(boxes["clipped_bbox_x"]),
        float(boxes["clipped_bbox_y"]),
        float(boxes["clipped_bbox_w"]),
        float(boxes["clipped_bbox_h"]),
    )
    vx, vy, vw, vh = core.intersect_boxes(raster, clipped_box)
    visibility = core.clamp_unit(vw * vh / clipped_area)

    local = core.actor_local_points(
        roi["world"][owned], target_centre, target_yaw_deg
    )
    half = np.asarray([float(v) for v in target_extent], dtype=np.float64)
    result.update(
        {
            "visibility": visibility,
            "visibility_band": core.band_for_score(visibility),
            "no_support": False,
            "visible_bbox_x": vx,
            "visible_bbox_y": vy,
            "visible_bbox_w": vw,
            "visible_bbox_h": vh,
            "visible_box_area_px": vw * vh,
            "visible_box_raster_ratio": raster[2] * raster[3] / clipped_area,
            "degenerate_visible_box": bool(vw * vh <= 0.0),
            "visible_box_height_ratio": core.clamp_unit(
                vh / float(boxes["clipped_bbox_h"])
            ),
            "visible_box_width_ratio": core.clamp_unit(
                vw / float(boxes["clipped_bbox_w"])
            ),
            # Retained actor points per sampled ROI pixel: the direct analogue of
            # the old depth-interval `visible_fraction`, reported as a diagnostic
            # only and never folded into `visibility`.
            "actor_point_surface_occupancy": core.clamp_unit(
                retained_count / float(roi["roi_px"])
            ),
            # Retained actor points per derived visible-box pixel: how solidly
            # the visible box is filled.  Diagnostic only.
            "visible_box_fill_ratio": core.clamp_unit(
                min(1.0, retained_count / (vw * vh)) if vw * vh > 0.0 else 0.0
            ),
            "retained_local_z_min_above_floor_m": float(
                np.min(local[:, 2]) + half[2]
            ),
            "retained_max_abs_local_excess_m": float(
                np.max(np.abs(local) - half[None, :])
            ),
        }
    )
    return result
