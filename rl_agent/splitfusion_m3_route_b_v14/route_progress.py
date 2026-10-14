"""Causal Route-B progress publication for the M3.4 BAD_TO_GOOD transition.

The retained route loop remains the sole CARLA tick owner.  This module wraps
that owner's world object and observes the ego pose only *after* each completed
tick.  Progress is a monotone projection onto the ordered ``planned_path`` in
the hash-bound Town10 route JSON.  It is never exposed to the actor.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

from . import contract as C


SCHEMA = "splitfusion.m3_route_b_v14.route_progress.v1"
SOURCE = "PINNED_ROUTE_JSON_PLANNED_PATH_ORDERED_PROJECTION"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_geometry(root: Path) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    route_path = root / C.ROUTE_JSON
    if _sha(route_path) != C.ROUTE_JSON_SHA256:
        raise ValueError("pinned Route-B JSON drift")
    document = json.loads(route_path.read_bytes())
    points = [(float(row["x"]), float(row["y"])) for row in document["planned_path"]]
    if len(points) < 3:
        raise ValueError("planned route needs at least three points")
    lengths = [math.hypot(points[(i + 1) % len(points)][0] - points[i][0],
                          points[(i + 1) % len(points)][1] - points[i][1])
               for i in range(len(points))]
    if any(not math.isfinite(value) or value <= 0.0 for value in lengths):
        raise ValueError("planned route contains a non-positive segment")
    cumulative = [0.0]
    for value in lengths[:-1]:
        cumulative.append(cumulative[-1] + value)
    total = float(sum(lengths))
    return dict(points=points, lengths=lengths, cumulative=cumulative,
                route_length_m=total, threshold_distance_m=total * C.TRANSITION_FRACTION,
                route_json=str(C.ROUTE_JSON), route_json_sha256=C.ROUTE_JSON_SHA256,
                progress_csv=str(C.PROGRESS_CSV), progress_csv_sha256=C.PROGRESS_CSV_SHA256,
                source=SOURCE)


class OrderedProgress:
    """Monotone local projection; backtracking carries the prior progress.

    The forward search is bounded and a sample cannot advance farther than the
    observed ego displacement plus the registered margin.  This prevents an
    intersection, loop closure, or teleported/jittered pose from jumping to a
    later route section.
    """

    def __init__(self, geometry: dict[str, Any]):
        self.geometry = geometry
        self.segment_index = 0
        self.cumulative_distance_m = 0.0
        self.previous_xy: tuple[float, float] | None = None
        self.samples = 0
        self.rejected = 0

    def _project(self, x: float, y: float, indices: Iterable[int]):
        points = self.geometry["points"]
        lengths = self.geometry["lengths"]
        cumulative = self.geometry["cumulative"]
        candidates = []
        for index in indices:
            start, end = points[index], points[(index + 1) % len(points)]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = lengths[index]
            fraction = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / (length * length)))
            px, py = start[0] + fraction * dx, start[1] + fraction * dy
            along = cumulative[index] + fraction * length
            candidates.append((math.hypot(x - px, y - py), along, index, fraction))
        return min(candidates)

    def observe(self, x: float, y: float) -> dict[str, Any]:
        x, y = float(x), float(y)
        count = len(self.geometry["points"])
        if self.samples == 0:
            indices = range(0, min(count, C.PROGRESS_INITIAL_SEGMENTS))
        else:
            lo = max(0, self.segment_index - C.PROGRESS_BACKTRACK_SEGMENTS)
            hi = min(count, self.segment_index + C.PROGRESS_FORWARD_SEGMENTS + 1)
            indices = range(lo, hi)
        cross_track, along, segment, fraction = self._project(x, y, indices)
        displacement = (0.0 if self.previous_xy is None else
                        math.hypot(x - self.previous_xy[0], y - self.previous_xy[1]))
        maximum_advance = max(C.PROGRESS_MAX_FORWARD_MARGIN_M,
                              displacement * C.PROGRESS_DISPLACEMENT_MULTIPLIER +
                              C.PROGRESS_FORWARD_ALLOWANCE_M)
        accepted = (cross_track <= C.PROGRESS_MAX_CROSS_TRACK_M and
                    along <= self.cumulative_distance_m + maximum_advance)
        if self.samples == 0:
            accepted = cross_track <= C.PROGRESS_MAX_CROSS_TRACK_M
        previous = self.cumulative_distance_m
        if accepted and along >= previous:
            self.cumulative_distance_m = min(float(along), self.geometry["route_length_m"])
            self.segment_index = int(segment)
        elif not accepted:
            self.rejected += 1
        self.previous_xy = (x, y)
        self.samples += 1
        return dict(cumulative_distance_m=self.cumulative_distance_m,
                    previous_cumulative_distance_m=previous,
                    cross_track_m=float(cross_track), projected_distance_m=float(along),
                    segment_index=int(segment), segment_fraction=float(fraction),
                    accepted=bool(accepted), carried_for_backtracking=bool(accepted and along < previous),
                    displacement_m=displacement, maximum_forward_advance_m=maximum_advance)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.is_symlink():
        raise ValueError("route-progress snapshot path must not be a symlink")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".route_progress_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class Publisher:
    def __init__(self, attempt: Path, root: Path):
        self.attempt = Path(attempt).resolve(strict=True)
        self.directory = self.attempt / "m3_route"
        self.directory.mkdir(exist_ok=True)
        self.snapshot_path = self.attempt / C.PROGRESS_SNAPSHOT_RELATIVE
        self.trace_path = self.directory / "ROUTE_PROGRESS_TRACE.jsonl"
        self.summary_path = self.directory / "ROUTE_PROGRESS_SUMMARY.json"
        if any(path.exists() or path.is_symlink() for path in
               (self.snapshot_path, self.trace_path, self.summary_path)):
            raise FileExistsError("route-progress evidence is create-only")
        self.geometry = load_geometry(root)
        expected = C.route_transition_registration(root)
        if any(abs(self.geometry[key] - expected[key]) > 1e-9
               for key in ("route_length_m", "threshold_distance_m")):
            raise ValueError("route transition geometry differs from registration")
        self.tracker = OrderedProgress(self.geometry)
        self.trace = self.trace_path.open("x", encoding="utf-8")
        self.crossing: dict[str, Any] | None = None
        self.last: dict[str, Any] | None = None
        self.failed = False

    def publish(self, frame_id: int, x: float, y: float) -> dict[str, Any]:
        projection = self.tracker.observe(x, y)
        mono = time.monotonic_ns()
        raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        row = dict(schema=SCHEMA, status="ACTIVE", source=SOURCE,
                   source_frame_id=int(frame_id), source_monotonic_ns=mono,
                   source_monotonic_raw_ns=raw, source_wall_ns=time.time_ns(),
                   ego_x=float(x), ego_y=float(y),
                   route_length_m=self.geometry["route_length_m"],
                   threshold_distance_m=self.geometry["threshold_distance_m"],
                   transition_fraction=C.TRANSITION_FRACTION,
                   route_json_sha256=C.ROUTE_JSON_SHA256,
                   progress_csv_sha256=C.PROGRESS_CSV_SHA256, **projection)
        crossed = (self.crossing is None and
                   row["cumulative_distance_m"] >= row["threshold_distance_m"])
        row["threshold_crossed_now"] = crossed
        row["latched_phase"] = "GOOD" if crossed or self.crossing is not None else "BAD"
        if crossed:
            self.crossing = {key: row[key] for key in
                             ("source_frame_id", "source_monotonic_ns", "source_monotonic_raw_ns",
                              "source_wall_ns", "cumulative_distance_m", "threshold_distance_m")}
        self.trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.trace.flush()
        _atomic_json(self.snapshot_path, row)
        self.last = row
        return row

    def close(self, status: str, error: str | None = None) -> None:
        if self.trace.closed:
            return
        self.failed = bool(error)
        final = dict(schema=SCHEMA, status=status, source=SOURCE,
                     source_monotonic_ns=time.monotonic_ns(),
                     source_monotonic_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                     source_wall_ns=time.time_ns(), route_length_m=self.geometry["route_length_m"],
                     threshold_distance_m=self.geometry["threshold_distance_m"],
                     cumulative_distance_m=(None if self.last is None else self.last["cumulative_distance_m"]),
                     threshold_crossed=self.crossing is not None, error=error)
        self.trace.write(json.dumps(final, sort_keys=True, allow_nan=False) + "\n")
        self.trace.flush()
        self.trace.close()
        _atomic_json(self.snapshot_path, final)
        summary = dict(schema=SCHEMA, source=SOURCE, status=status, error=error,
                       samples=self.tracker.samples, rejected_samples=self.tracker.rejected,
                       route_length_m=self.geometry["route_length_m"],
                       threshold_distance_m=self.geometry["threshold_distance_m"],
                       transition_fraction=C.TRANSITION_FRACTION,
                       threshold_crossed=final["threshold_crossed"], crossing=self.crossing,
                       final_cumulative_distance_m=final["cumulative_distance_m"],
                       route_json=C.ROUTE_JSON, route_json_sha256=C.ROUTE_JSON_SHA256,
                       progress_csv=C.PROGRESS_CSV, progress_csv_sha256=C.PROGRESS_CSV_SHA256)
        _atomic_json(self.summary_path, summary)


class ProgressWorld:
    def __init__(self, world: Any, vehicle: Any, publisher: Publisher):
        self._world, self._vehicle, self._publisher = world, vehicle, publisher

    def tick(self, *args: Any, **kwargs: Any) -> int:
        frame_id = int(self._world.tick(*args, **kwargs))
        transform = self._vehicle.get_transform()
        self._publisher.publish(frame_id, transform.location.x, transform.location.y)
        return frame_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)


@contextmanager
def install_route_progress(attempt: Path):
    """Patch the already-retained route drive at its tick-owner call boundary."""
    import data_collection.run_route_b_density_loop as density
    original = density.drive_one_loop_with_traffic
    if getattr(original, "__m3_v14_route_progress__", False):
        raise RuntimeError("V14 route-progress seam already installed")
    root = Path(__file__).resolve().parents[2]

    def drive(world, vehicle, agent, route, collisions, args, loop_index, maintain, janitor):
        publisher = Publisher(attempt, root)
        error = None
        try:
            snapshot = world.get_snapshot()
            transform = vehicle.get_transform()
            publisher.publish(int(snapshot.frame), transform.location.x, transform.location.y)
            result = original(ProgressWorld(world, vehicle, publisher), vehicle, agent, route,
                              collisions, args, loop_index, maintain, janitor)
            publisher.close("ROUTE_COMPLETED" if result.get("completed") else "ROUTE_INCOMPLETE")
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"[:500]
            publisher.close("ABORTED", error)
            raise

    drive.__m3_v14_route_progress__ = True
    density.drive_one_loop_with_traffic = drive
    try:
        yield
    finally:
        density.drive_one_loop_with_traffic = original


def registration(root: Path) -> dict[str, Any]:
    geometry = load_geometry(root)
    return dict(schema=SCHEMA, source=SOURCE, route_json=geometry["route_json"],
                route_json_sha256=geometry["route_json_sha256"],
                progress_csv=geometry["progress_csv"],
                progress_csv_sha256=geometry["progress_csv_sha256"],
                route_length_m=geometry["route_length_m"],
                threshold_distance_m=geometry["threshold_distance_m"],
                transition_fraction=C.TRANSITION_FRACTION,
                projection=dict(max_cross_track_m=C.PROGRESS_MAX_CROSS_TRACK_M,
                                initial_segments=C.PROGRESS_INITIAL_SEGMENTS,
                                backtrack_segments=C.PROGRESS_BACKTRACK_SEGMENTS,
                                forward_segments=C.PROGRESS_FORWARD_SEGMENTS,
                                max_forward_margin_m=C.PROGRESS_MAX_FORWARD_MARGIN_M,
                                displacement_multiplier=C.PROGRESS_DISPLACEMENT_MULTIPLIER,
                                forward_allowance_m=C.PROGRESS_FORWARD_ALLOWANCE_M,
                                monotone=True, backtracking="carry forward",
                                loop_closure="never wrap to zero"),
                snapshot=C.PROGRESS_SNAPSHOT_RELATIVE,
                trace="m3_route/ROUTE_PROGRESS_TRACE.jsonl",
                summary="m3_route/ROUTE_PROGRESS_SUMMARY.json")
