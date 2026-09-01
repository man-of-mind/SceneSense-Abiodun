"""Training-only expected-clear-support reference for normalised visibility.

The actor-volume pilot (`AUDIT_REPORT_20260901_191239.md`) showed that the
extraction is sound but the *scale* is biased, because the denominator was the
projected 3D cuboid, which carries head/foot clearance and, once yawed, projects
wider than the rendered silhouette.  An unoccluded pedestrian therefore scored a
median 0.763 rather than >= 0.90.

This module replaces only the denominator.  Instead of the cuboid area it uses
the *expected unoccluded surface support* for comparable pedestrians, estimated
from training data alone with no human labels:

    support_density = retained_actor_volume_pixels / clipped_projected_box_area
    expected_clear_support_density = 95th percentile of support_density
                                     over a comparable training group
    normalized_visibility = clamp(support_density / expected, 0, 1)

The 95th percentile stands in for "unoccluded": within a group of comparable
pedestrians, the best-supported few per cent are the ones nothing was blocking.
No occlusion, visibility or eligibility flag is consulted to build it.

Conditioning is on actor type, folded relative view angle, and projected box
height, with a fixed count-based fallback hierarchy.  Every constant here is
locked; see the module constants.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

REFERENCE_VERSION = "route_b_train_normalized_actor_volume_reference_v1"

# --- locked conditioning scheme ---------------------------------------------
ANGLE_BIN_EDGES: tuple[tuple[float, float, str], ...] = (
    (0.0, 30.0, "a00_30"),
    (30.0, 60.0, "a30_60"),
    (60.0, 90.0, "a60_90"),
)
HEIGHT_BIN_EDGES: tuple[tuple[float, float, str], ...] = (
    (0.0, 24.0, "h_lt24"),
    (24.0, 48.0, "h24_48"),
    (48.0, 96.0, "h48_96"),
    (96.0, math.inf, "h_ge96"),
)

# --- locked estimator --------------------------------------------------------
REFERENCE_PERCENTILE = 95.0
PERCENTILE_METHOD = "higher"

# --- locked fallback hierarchy ----------------------------------------------
MIN_N_TYPE_ANGLE_HEIGHT = 50
MIN_N_ANGLE_HEIGHT = 100
MIN_N_HEIGHT = 100

TIER_TYPE_ANGLE_HEIGHT = "type_angle_height"
TIER_ANGLE_HEIGHT = "angle_height"
TIER_HEIGHT = "height"
TIER_GLOBAL = "global"
TIER_ORDER: tuple[str, ...] = (
    TIER_TYPE_ANGLE_HEIGHT,
    TIER_ANGLE_HEIGHT,
    TIER_HEIGHT,
    TIER_GLOBAL,
)

# --- locked training-population filter --------------------------------------
MAX_DISTANCE_M = 40.0
MIN_IN_FRAME_FRACTION = 0.98


def folded_view_angle_deg(
    centre_world: Sequence[float], yaw_deg: float, camera_position: Sequence[float]
) -> float:
    """Angle between the actor's facing axis and the line of sight, folded to [0, 90].

    Both vectors are taken in the world XY plane.  0 degrees means the actor is
    seen head-on or directly from behind; 90 degrees means full profile.  The
    fold is deliberate: the projected silhouette of a pedestrian is the same
    width whether they face the camera or face away from it.
    """
    centre = np.asarray(centre_world, dtype=np.float64)
    camera = np.asarray(camera_position, dtype=np.float64)
    line_of_sight = centre[:2] - camera[:2]
    norm = float(np.linalg.norm(line_of_sight))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("actor is coincident with the camera in the XY plane")
    line_of_sight = line_of_sight / norm
    yaw = math.radians(float(yaw_deg))
    facing = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    cosine = float(np.clip(np.dot(facing, line_of_sight), -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def angle_bin(angle_deg: float) -> str:
    value = float(angle_deg)
    if not math.isfinite(value) or value < 0.0 or value > 90.0:
        raise ValueError(f"folded view angle {value!r} outside [0, 90]")
    for lower, upper, name in ANGLE_BIN_EDGES:
        if value >= lower and (value < upper or upper == 90.0):
            return name
    raise ValueError(f"unbinned folded view angle {value!r}")


def height_bin(height_px: float) -> str:
    value = float(height_px)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"projected box height {value!r} is not a finite non-negative")
    for lower, upper, name in HEIGHT_BIN_EDGES:
        if value >= lower and value < upper:
            return name
    raise ValueError(f"unbinned projected box height {value!r}")


def support_density(retained_pixels: int, clipped_projected_area_px: float) -> float:
    """Retained actor-volume pixels per unit of clipped projected box area."""
    area = float(clipped_projected_area_px)
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"clipped projected area {area!r} must be finite and positive")
    return float(int(retained_pixels)) / area


def _percentile(values: Sequence[float]) -> float:
    return float(
        np.percentile(
            np.asarray(values, dtype=np.float64),
            REFERENCE_PERCENTILE,
            method=PERCENTILE_METHOD,
        )
    )


def build_reference(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Group the training records and take the locked 95th percentile per group.

    ``records`` need ``actor_type``, ``angle_bin``, ``height_bin`` and
    ``support_density``.  Groups are emitted for all four tiers so the fallback
    hierarchy can resolve deterministically at lookup time.
    """
    rows = list(records)
    if not rows:
        raise ValueError("cannot build a reference from zero training records")

    buckets: dict[str, dict[str, list[float]]] = {tier: {} for tier in TIER_ORDER}
    for row in rows:
        density = float(row["support_density"])
        keys = {
            TIER_TYPE_ANGLE_HEIGHT: "|".join(
                (str(row["actor_type"]), str(row["angle_bin"]), str(row["height_bin"]))
            ),
            TIER_ANGLE_HEIGHT: "|".join((str(row["angle_bin"]), str(row["height_bin"]))),
            TIER_HEIGHT: str(row["height_bin"]),
            TIER_GLOBAL: TIER_GLOBAL,
        }
        for tier, key in keys.items():
            buckets[tier].setdefault(key, []).append(density)

    tables: dict[str, dict[str, dict[str, float]]] = {}
    for tier, groups in buckets.items():
        tables[tier] = {
            key: {
                "n": len(values),
                "expected_clear_support_density": _percentile(values),
                "median_support_density": float(np.median(values)),
                "zero_support_count": int(sum(1 for v in values if v <= 0.0)),
            }
            for key, values in sorted(groups.items())
        }

    return {
        "reference_version": REFERENCE_VERSION,
        "percentile": REFERENCE_PERCENTILE,
        "percentile_method": PERCENTILE_METHOD,
        "min_n": {
            TIER_TYPE_ANGLE_HEIGHT: MIN_N_TYPE_ANGLE_HEIGHT,
            TIER_ANGLE_HEIGHT: MIN_N_ANGLE_HEIGHT,
            TIER_HEIGHT: MIN_N_HEIGHT,
        },
        "angle_bin_edges": [list(edge) for edge in ANGLE_BIN_EDGES],
        "height_bin_edges": [
            [lower, None if math.isinf(upper) else upper, name]
            for lower, upper, name in HEIGHT_BIN_EDGES
        ],
        "total_records": len(rows),
        "tables": tables,
    }


def lookup(
    reference: dict[str, Any], actor_type: str, angle: str, height: str
) -> tuple[float, str, int, str]:
    """Resolve the expected clear support density through the locked hierarchy.

    Returns ``(expected, tier, group_n, group_key)``.
    """
    tables = reference["tables"]
    minimums = reference["min_n"]
    candidates = (
        (TIER_TYPE_ANGLE_HEIGHT, "|".join((str(actor_type), angle, height)),
         int(minimums[TIER_TYPE_ANGLE_HEIGHT])),
        (TIER_ANGLE_HEIGHT, "|".join((angle, height)), int(minimums[TIER_ANGLE_HEIGHT])),
        (TIER_HEIGHT, height, int(minimums[TIER_HEIGHT])),
    )
    for tier, key, minimum in candidates:
        group = tables[tier].get(key)
        if group is not None and int(group["n"]) >= minimum:
            return (
                float(group["expected_clear_support_density"]),
                tier,
                int(group["n"]),
                key,
            )
    group = tables[TIER_GLOBAL][TIER_GLOBAL]
    return (
        float(group["expected_clear_support_density"]),
        TIER_GLOBAL,
        int(group["n"]),
        TIER_GLOBAL,
    )


def normalized_visibility(density: float, expected: float) -> float:
    """clamp(support_density / expected_clear_support_density, 0, 1)."""
    reference_value = float(expected)
    if not math.isfinite(reference_value) or reference_value <= 0.0:
        raise ValueError(f"expected clear support {reference_value!r} must be finite and positive")
    value = float(density) / reference_value
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"non-finite or negative normalised visibility {value!r}")
    return min(1.0, value)
