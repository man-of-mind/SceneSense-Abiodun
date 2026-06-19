from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from .schemas import Point2D, SensorPose2D, SpatialObject


EPS = 1e-9


def rotate2d(x: float, y: float, yaw_deg: float) -> Point2D:
    theta = math.radians(float(yaw_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    return (float(c * x - s * y), float(s * x + c * y))


def translate_point(point: Point2D, dx: float, dy: float) -> Point2D:
    return (float(point[0] + dx), float(point[1] + dy))


def polygon_area(poly: Sequence[Point2D]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def ensure_ccw(poly: Sequence[Point2D]) -> List[Point2D]:
    signed = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        signed += (x2 - x1) * (y2 + y1)
    if signed > 0:
        return list(reversed(poly))
    return list(poly)


def sensor_fov_polygon(
    pose: SensorPose2D,
    fov_deg: float,
    range_m: float,
    near_m: float = 0.0,
) -> List[Point2D]:
    """Build a top-down ground footprint for a camera/radar FoV.

    The output is a convex wedge/trapezoid in world coordinates. It is a
    first-pass approximation, not a true 3D frustum.
    """

    half = math.radians(float(fov_deg) * 0.5)
    far = max(0.0, float(range_m))
    near = max(0.0, min(float(near_m), far))

    far_left = (far * math.cos(half), far * math.sin(half))
    far_right = (far * math.cos(-half), far * math.sin(-half))

    if near <= EPS:
        local_points = [(0.0, 0.0), far_right, far_left]
    else:
        near_left = (near * math.cos(half), near * math.sin(half))
        near_right = (near * math.cos(-half), near * math.sin(-half))
        local_points = [near_left, near_right, far_right, far_left]

    world_points = [
        translate_point(rotate2d(x, y, pose.yaw_deg), pose.x, pose.y)
        for x, y in local_points
    ]
    return ensure_ccw(world_points)


def point_in_polygon(point: Point2D, poly: Sequence[Point2D]) -> bool:
    if len(poly) < 3:
        return False
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / max(EPS, yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _inside_half_plane(point: Point2D, edge_start: Point2D, edge_end: Point2D) -> bool:
    px, py = point
    ax, ay = edge_start
    bx, by = edge_end
    return ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) >= -EPS


def _line_intersection(
    p1: Point2D,
    p2: Point2D,
    q1: Point2D,
    q2: Point2D,
) -> Point2D:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < EPS:
        return p2
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom
    return (float(px), float(py))


def convex_polygon_intersection(
    subject_polygon: Sequence[Point2D],
    clip_polygon: Sequence[Point2D],
) -> List[Point2D]:
    """Sutherland-Hodgman clipping for convex polygons."""

    output = ensure_ccw(subject_polygon)
    clip = ensure_ccw(clip_polygon)
    if len(output) < 3 or len(clip) < 3:
        return []

    for i, edge_start in enumerate(clip):
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        previous = input_list[-1]
        for current in input_list:
            current_inside = _inside_half_plane(current, edge_start, edge_end)
            previous_inside = _inside_half_plane(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def overlap_area(poly_a: Sequence[Point2D], poly_b: Sequence[Point2D]) -> float:
    return polygon_area(convex_polygon_intersection(poly_a, poly_b))


def overlap_ratio(poly_a: Sequence[Point2D], poly_b: Sequence[Point2D]) -> float:
    intersection = overlap_area(poly_a, poly_b)
    denom = max(EPS, min(polygon_area(poly_a), polygon_area(poly_b)))
    return float(intersection / denom)


def object_footprint_polygon(obj: SpatialObject) -> List[Point2D]:
    half_l = max(0.05, float(obj.length)) * 0.5
    half_w = max(0.05, float(obj.width)) * 0.5
    corners = [
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, -half_w),
        (-half_l, half_w),
    ]
    return [
        translate_point(rotate2d(x, y, obj.yaw_deg), obj.x, obj.y)
        for x, y in corners
    ]


def bounds(points: Iterable[Point2D]) -> Tuple[float, float, float, float]:
    pts = list(points)
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))
