"""Registered route geometry for M3.1: ordered progress coverage and predeclared route sections.

Inputs are the pinned route JSON and its 4-m progress CSV (``contract.ROUTE_JSON`` / ``PROGRESS_CSV``) and the
measured UE ego poses (pose archive OWNED_SOURCE ``ego_world``; every route tick). Nothing here reads latency,
quality, action or outcome.

Ordered coverage: progress points are visited in order; point i is matched at the first pose sample at or after the
previous match within ``PROGRESS_MATCH_M``; coverage = matched / points.
Sections (predeclared): LEG_k (k = 1..19; leg k ends at ordered waypoint k, leg 19 returns to the start), the region
group of each leg (B1/B2/B3 from ``route_b_regions``; leg 19 = B1), and DECILE_d (10 equal progress-index bins).
A sample's progress index is its ordered match against the progress points (monotone, nearest within the match radius).
"""
import csv
import json
import math
from pathlib import Path

from rl_agent.splitfusion_m3_route_b_v2 import contract as C


def load_route(root):
    root = Path(root)
    route = json.loads((root / C.ROUTE_JSON).read_bytes())
    with (root / C.PROGRESS_CSV).open(newline="") as handle:
        progress = [(float(r["ego_x"]), float(r["ego_y"])) for r in csv.DictReader(handle)]
    waypoints = [(float(w["x"]), float(w["y"])) for w in route["intermediate_waypoints"]]
    waypoints.append((float(route["start"]["location"]["x"]), float(route["start"]["location"]["y"])))
    regions = list(route["route_b_regions"]) + ["B1"]
    return dict(progress=progress, waypoints=waypoints, regions=regions, name=route["name"])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def waypoint_progress_index(route):
    """Progress index closest to each ordered waypoint, searched forward (monotone)."""
    out, start = [], 0
    progress = route["progress"]
    for k, w in enumerate(route["waypoints"]):
        last = k == len(route["waypoints"]) - 1
        lo = start if not last else max(start, len(progress) // 2)
        best = min(range(lo, len(progress)), key=lambda i: _dist(progress[i], w))
        out.append(best)
        start = best
    return out


def ordered_coverage(poses, route, radius=C.PROGRESS_MATCH_M):
    """poses: time-ordered (x, y). Returns coverage and the per-point matched pose index (None if unmatched)."""
    progress = route["progress"]
    matched, t = [], 0
    for point in progress:
        hit = None
        for j in range(t, len(poses)):
            if _dist(poses[j], point) <= radius:
                hit = j
                break
        matched.append(hit)
        if hit is not None:
            t = hit
    count = sum(1 for m in matched if m is not None)
    return dict(points=len(progress), matched=count, coverage=count / len(progress) if progress else 0.0,
                first_unmatched=[i for i, m in enumerate(matched) if m is None][:20], matched_index=matched,
                radius_m=radius)


def progress_index_series(poses, route, radius=C.PROGRESS_MATCH_M, window=40):
    """Monotone progress index per pose sample (nearest point within a forward window; carries forward)."""
    progress = route["progress"]
    current, series = 0, []
    for p in poses:
        hi = min(len(progress), current + window)
        best = min(range(current, hi), key=lambda i: _dist(progress[i], p)) if current < hi else current
        if _dist(progress[best], p) <= radius:
            current = best
        series.append(current)
    return series


def sections(progress_index, route):
    """Predeclared section labels for one progress index."""
    marks = waypoint_progress_index(route)
    leg = next((k + 1 for k, m in enumerate(marks) if progress_index <= m), len(marks))
    decile = min(9, int(10 * progress_index / max(1, len(route["progress"]))))
    return dict(leg=f"LEG_{leg:02d}", region=route["regions"][leg - 1], decile=f"DECILE_{decile}")


def poses_from_archive(pose_jsonl):
    """Time-ordered measured ego (x, y) and frame ids from the UE pose archive OWNED_SOURCE records."""
    rows = []
    with open(pose_jsonl) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") != "OWNED_SOURCE":
                continue
            source = record["source"]
            rows.append((int(source["frame_id"]), float(source["ego_world"][0]), float(source["ego_world"][1]),
                         float(source.get("timestamp", "nan"))))
    rows.sort()
    return rows
