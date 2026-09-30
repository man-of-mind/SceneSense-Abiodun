"""Addenda 9/10: instrumented, range-limited object-GT construction with LOW
deferral, admitted to the HIGH queue only after the last datagram is sent.

The v8 handshake showed the reward ACK missing 170 ms because simulator
object-GT generation for the reward frame took 372 ms. The pinned builder
(``carla_collect_parked_ego_fusion_training_data.build_object_rows``) runs at
140 m. For every actor it builds bbox corners, projects them, updates the
stationary tracker and counts radar support against the whole radar window.
The pinned ``_ground_truth`` then discards everything beyond
``max_gt_distance_m`` (40 m).

:func:`build_object_rows_v2` reproduces the pinned builder exactly. It calls
the same pinned helper functions in the same actor order, with the same
exception handling and the same stationary-tracker side effects. It adds the
following:

* The identical actor-origin distance check
  (``actor.get_location().distance(camera_location)``) runs before corner
  generation. It has no side effects, so moving it earlier changes nothing.
* ``eligibility_distance_m`` (the authoritative 40 m) skips only the work
  whose output the pinned ``valid_localization_objects`` discards under the
  same ``> max_distance_m`` comparison: radar support and row construction.
  Projection and ``stationary_tracker.update`` still run for every actor up to
  the 140-m builder limit that projects. Tracker state therefore evolves
  exactly as before, and every surviving row is bit-identical.
* Per-stage wall time, thread CPU time, context switches and actor counts
  (:class:`ObjectGtProfileV2`).
* Addendum 10 removed the addendum-9 decision-open prefetch thread (it
  contended with the reward frame's 7-channel preparation in v9). Reward GT
  is admitted to the existing HIGH queue only after the frame's exact
  last-datagram-sent mark (:class:`LastDatagramMarksV2`) exists; the mark is
  record-only and never enqueues or computes anything.
* Cooperative LOW preemption. A non-reward ticket stops as soon as a reward
  ticket becomes pending (:class:`LowObjectGtPreempted`), so LOW never runs
  while HIGH is pending or running.

It starts no thread or process and makes no approximation. GT definitions are
unchanged. Importing this module performs no I/O.
"""

from __future__ import annotations

import collections
import math
import resource
import threading
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np

from . import phase6_gt_priority_v2 as GP

PATTERNS = (("vehicle", "vehicle.*"), ("person", "walker.pedestrian.*"))
STAGES = ("actor_list", "filter", "distance", "bbox_corners", "projection", "tracker",
          "radar_support", "row_build")
LOW_SKIPPED_STATUS = "OBJECT_GT_LOW_SKIPPED_REWARD_PRIORITY"


def raw_now_ns() -> int:
    """UE CLOCK_MONOTONIC_RAW, the domain of the Run-4 stage clock."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class LowObjectGtPreempted(RuntimeError):
    """A non-reward object-GT ticket yielded to a pending reward ticket."""


def _rusage_thread() -> tuple[int, int]:
    usage = resource.getrusage(resource.RUSAGE_THREAD)
    return int(usage.ru_nvcsw), int(usage.ru_nivcsw)


class ObjectGtProfileV2:
    """Per-ticket builder timing (wall/thread-CPU/context switches, per stage)."""

    def __init__(self, *, frame_id: int, queue_class: Optional[str] = None,
                 raw_clock: Callable[[], int] = raw_now_ns) -> None:
        self.frame_id = int(frame_id)
        self.queue_class = queue_class
        self._raw_clock = raw_clock
        self.stage_ns = collections.Counter()
        self.counts = collections.Counter()
        self.data: dict[str, Any] = {}

    def begin(self) -> None:
        vol, invol = _rusage_thread()
        self.data.update(start_wall_ns=time.time_ns(), start_raw_ns=self._raw_clock(),
                         _perf=time.perf_counter_ns(), _cpu=time.thread_time_ns(),
                         _vol=vol, _invol=invol)

    def end(self, *, outcome: str = "COMPLETED") -> dict[str, Any]:
        vol, invol = _rusage_thread()
        perf, cpu = time.perf_counter_ns(), time.thread_time_ns()
        self.data.update(
            end_wall_ns=time.time_ns(), end_raw_ns=self._raw_clock(),
            wall_ms=(perf - self.data.pop("_perf")) / 1e6,
            thread_cpu_ms=(cpu - self.data.pop("_cpu")) / 1e6,
            voluntary_ctx_switches=vol - self.data.pop("_vol"),
            involuntary_ctx_switches=invol - self.data.pop("_invol"),
            outcome=outcome)
        return self.snapshot()

    def add(self, stage: str, ns: int) -> None:
        self.stage_ns[stage] += int(ns)

    def snapshot(self) -> dict[str, Any]:
        out = {k: v for k, v in self.data.items() if not k.startswith("_")}
        out.update(frame_id=self.frame_id, queue_class=self.queue_class,
                   stage_ms={s: self.stage_ns[s] / 1e6 for s in STAGES},
                   actor_counts=dict(self.counts))
        return out


def _person_support(label: str, mode: str) -> str:
    return "person_radius" if label == "person" and mode == "radius" else "bbox"


def build_object_rows_v2(parked: Any, *, world: Any, ego_vehicle: Any,
                         sample_base: Mapping[str, Any], camera_location: Any,
                         camera_matrix: np.ndarray, camera_inverse_matrix: np.ndarray,
                         intrinsics: np.ndarray, width: int, height: int,
                         max_distance_m: float, radar_world_xyz: np.ndarray,
                         stationary_tracker: Any, include_pedestrians: bool,
                         radar_support_margin_m: float,
                         radar_person_support_mode: str = "radius",
                         radar_person_support_radius_m: float = 1.5,
                         radar_person_support_z_down_m: float = 0.5,
                         radar_person_support_z_up_m: float = 2.0,
                         eligibility_distance_m: Optional[float] = None,
                         profile: Optional[ObjectGtProfileV2] = None,
                         preempt: Optional[Callable[[], bool]] = None) -> list[dict[str, Any]]:
    """The pinned ``build_object_rows`` with range-limited post-projection work.

    With ``eligibility_distance_m=None`` this returns exactly the
    pinned rows. With it, it returns exactly the pinned rows whose
    ``gt_distance_m`` is not greater than it, and it leaves the tracker in the
    identical state.
    """
    del camera_matrix  # unused by the pinned per-actor function as well
    perf = time.perf_counter_ns
    prof = profile
    counts = prof.counts if prof is not None else collections.Counter()
    rows: list[dict[str, Any]] = []
    patterns = [PATTERNS[0]] + ([PATTERNS[1]] if include_pedestrians else [])
    t0 = perf()
    actors = world.get_actors()
    if prof is not None:
        prof.add("actor_list", perf() - t0)
        try:
            counts["actors_total"] = len(actors)
        except TypeError:
            pass
    limit, eligible_limit = float(max_distance_m), eligibility_distance_m
    for label, pattern in patterns:
        t0 = perf()
        selected = actors.filter(pattern)
        if prof is not None:
            prof.add("filter", perf() - t0)
        for actor in selected:
            if preempt is not None and preempt():
                counts["preempted_at_actor"] = counts["actors_visited"]
                raise LowObjectGtPreempted(
                    f"frame {sample_base.get('frame_id')} yielded to a pending reward ticket")
            counts["actors_visited"] += 1
            counts[f"filtered_{label}"] += 1
            if int(actor.id) == int(ego_vehicle.id):
                counts["ego_skipped"] += 1
                continue
            # Same try-scope semantics as the pinned function: any RuntimeError
            # from transform/bbox/corners/distance drops the actor.
            t0 = perf()
            try:
                transform = actor.get_transform()
                bbox = actor.bounding_box
                distance_m = float(actor.get_location().distance(camera_location))
            except RuntimeError:
                counts["runtime_error_dropped"] += 1
                continue
            t1 = perf()
            if prof is not None:
                prof.add("distance", t1 - t0)
            if limit > 0.0 and distance_m > limit:
                counts["beyond_builder_limit"] += 1
                continue
            counts["within_builder_limit"] += 1
            try:
                center_world, corners_world = parked.actor_bbox_world_points(actor)
            except RuntimeError:
                counts["runtime_error_dropped"] += 1
                continue
            t2 = perf()
            if prof is not None:
                prof.add("bbox_corners", t2 - t1)
            projection = parked.project_world_points_to_bbox(
                corners_world, camera_inverse_matrix, intrinsics, int(width), int(height))
            if prof is not None:
                prof.add("projection", perf() - t2)
            if projection is None:
                counts["not_projected"] += 1
                continue
            counts["projected"] += 1
            t3 = perf()
            velocity = actor.get_velocity()
            speed = math.sqrt(float(velocity.x) ** 2 + float(velocity.y) ** 2
                              + float(velocity.z) ** 2)
            stationary_age_s, stationary_label, parked_label = stationary_tracker.update(
                actor, float(sample_base["timestamp"]))
            t4 = perf()
            if prof is not None:
                prof.add("tracker", t4 - t3)
            if eligible_limit is not None and distance_m > float(eligible_limit):
                counts["beyond_eligibility_limit"] += 1
                continue
            counts["within_eligibility_limit"] += 1
            support = parked.radar_support_count(
                actor=actor, label=label, radar_world_xyz=radar_world_xyz,
                margin_m=float(radar_support_margin_m),
                person_support_mode=str(radar_person_support_mode),
                person_radius_m=float(radar_person_support_radius_m),
                person_z_down_m=float(radar_person_support_z_down_m),
                person_z_up_m=float(radar_person_support_z_up_m))
            t5 = perf()
            if prof is not None:
                prof.add("radar_support", t5 - t4)
            rows.append(_row(sample_base, label, actor, transform, bbox, projection,
                             distance_m, center_world, camera_inverse_matrix, velocity, speed,
                             stationary_age_s, stationary_label, parked_label, support,
                             radar_person_support_mode, radar_person_support_radius_m,
                             radar_person_support_z_down_m, radar_person_support_z_up_m))
            counts["rows_built"] += 1
            if prof is not None:
                prof.add("row_build", perf() - t5)
    return rows


def _row(sample_base, label, actor, transform, bbox, projection, distance_m, center_world,
         camera_inverse_matrix, velocity, speed, stationary_age_s, stationary_label,
         parked_label, radar_support_points, person_mode, person_radius_m, person_z_down_m,
         person_z_up_m) -> dict[str, Any]:
    """Field-for-field the dict ``project_actor_to_object_row`` returns."""
    sensor_center = (camera_inverse_matrix
                     @ np.asarray([*center_world, 1.0], dtype=np.float64).T).T[:3]
    mode = _person_support(label, person_mode)
    return {
        **sample_base,
        "label": label,
        "gt_actor_id": str(actor.id),
        "gt_source": "actor",
        "gt_actor_type_id": str(getattr(actor, "type_id", "")),
        **projection,
        "gt_distance_m": distance_m,
        "gt_extent_x_m": float(bbox.extent.x),
        "gt_extent_y_m": float(bbox.extent.y),
        "gt_extent_z_m": float(bbox.extent.z),
        "gt_size_x_m": float(bbox.extent.x) * 2.0,
        "gt_size_y_m": float(bbox.extent.y) * 2.0,
        "gt_size_z_m": float(bbox.extent.z) * 2.0,
        "object_world_x": float(center_world[0]),
        "object_world_y": float(center_world[1]),
        "object_world_z": float(center_world[2]),
        "object_sensor_x": float(sensor_center[0]),
        "object_sensor_y": float(sensor_center[1]),
        "object_sensor_z": float(sensor_center[2]),
        "object_yaw_deg": float(transform.rotation.yaw),
        "object_velocity_x_mps": float(velocity.x),
        "object_velocity_y_mps": float(velocity.y),
        "object_velocity_z_mps": float(velocity.z),
        "object_speed_mps": float(speed),
        "stationary_age_s": float(stationary_age_s),
        "stationary_label": int(stationary_label),
        "parked_label": int(parked_label),
        "radar_support_points": radar_support_points,
        "radar_support_mode": mode,
        "radar_support_radius_m": float(person_radius_m) if mode == "person_radius" else "",
        "radar_support_z_down_m": float(person_z_down_m) if mode == "person_radius" else "",
        "radar_support_z_up_m": float(person_z_up_m) if mode == "person_radius" else "",
    }


def support_parameters(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {"margin_m": float(kwargs["radar_support_margin_m"]),
            "person_mode": str(kwargs.get("radar_person_support_mode", "radius")),
            "person_radius_m": float(kwargs.get("radar_person_support_radius_m", 1.5)),
            "person_z_down_m": float(kwargs.get("radar_person_support_z_down_m", 0.5)),
            "person_z_up_m": float(kwargs.get("radar_person_support_z_up_m", 2.0))}


class RewardPendingGateV2:
    """Reward tickets that are pending or running (opened at decision open)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: set[int] = set()
        self.events: list[dict[str, Any]] = []

    def open(self, frame_id: int) -> None:
        with self._lock:
            self._pending.add(int(frame_id))
            self.events.append({"frame_id": int(frame_id), "event": "open",
                                "raw_ns": raw_now_ns()})

    def close(self, frame_id: int, reason: str) -> None:
        with self._lock:
            if int(frame_id) in self._pending:
                self._pending.discard(int(frame_id))
                self.events.append({"frame_id": int(frame_id), "event": "close",
                                    "reason": reason, "raw_ns": raw_now_ns()})

    def blocked(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def pending(self) -> list[int]:
        with self._lock:
            return sorted(self._pending)


class LastDatagramMarksV2:
    """Record-only, first-wins last-datagram-sent instant per frame (raw clock).

    Marking never enqueues or computes object GT; a repeated completion for
    the same frame keeps the first instant and is only counted.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._marks: dict[int, int] = {}
        self.duplicates: dict[int, int] = {}

    def mark(self, frame_id: int, raw_ns: int) -> bool:
        with self._lock:
            if int(frame_id) in self._marks:
                self.duplicates[int(frame_id)] = self.duplicates.get(int(frame_id), 0) + 1
                return False
            self._marks[int(frame_id)] = int(raw_ns)
            return True

    def get(self, frame_id: int) -> Optional[int]:
        with self._lock:
            return self._marks.get(int(frame_id))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"marks": {str(k): v for k, v in sorted(self._marks.items())},
                    "duplicates": {str(k): v for k, v in sorted(self.duplicates.items())}}


class DeferringRewardPriorityGtQueueV2(GP.RewardPriorityGtQueueV2):
    """Addendum-7 queue plus: LOW never starts while a reward ticket is pending.

    A LOW ticket that arrives, or is still queued, while the reward gate is
    blocked is skipped explicitly (``on_skip``), never silently dropped and
    never allowed to build a backlog that could overflow. HIGH handling is
    unchanged.
    """

    def __init__(self, *, low_blocked: Callable[[], bool],
                 on_skip: Callable[[int, str], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._low_blocked = low_blocked
        self._on_skip = on_skip

    def put_nowait(self, item: Optional[Mapping[str, Any]]) -> None:
        if item is not None and self._low_blocked():
            frame_id = int(item["frame_id"])
            klass = self._classify(item)
            if klass == GP.LOW:
                with self._cond:
                    if frame_id in self._seen:
                        self.counters["duplicate_refused"] += 1
                        raise GP.GtQueueError(f"duplicate object-GT ticket for frame {frame_id}")
                    self._seen.add(frame_id)
                    self.counters["low_skipped_on_enqueue"] += 1
                self._on_skip(frame_id, "REWARD_PENDING_AT_ENQUEUE")
                return
        super().put_nowait(item)

    put = put_nowait

    def get(self, block: bool = True, timeout: Optional[float] = None):
        skipped: list[int] = []
        with self._cond:
            if not self._queues[GP.HIGH] and self._queues[GP.LOW] and self._low_blocked():
                while self._queues[GP.LOW]:
                    item, _klass, _at = self._queues[GP.LOW].popleft()
                    skipped.append(int(item["frame_id"]))
                    self.unfinished_tasks -= 1
                    self.counters["low_skipped_queued"] += 1
        for frame_id in skipped:
            self._on_skip(frame_id, "REWARD_PENDING_WHILE_QUEUED")
        return super().get(block=block, timeout=timeout)


class GtTicketLogV3(GP.GtTicketLogV2):
    """Addendum-7 ticket timeline plus builder profiles and LOW skips."""

    def object_profile(self, frame_id: int, profile: Mapping[str, Any]) -> None:
        with self._lock:
            self._row(frame_id)["object_builder"] = dict(profile)

    def note(self, frame_id: int, **fields: Any) -> None:
        with self._lock:
            self._row(frame_id).update(fields)

    def low_skipped(self, frame_id: int, reason: str, at: int) -> None:
        with self._lock:
            self._row(frame_id).update(low_skipped=reason, low_skipped_wall_ns=int(at))


def _interval_overlap_ms(a0: int, a1: int, b0: Optional[int], b1: Optional[int]) -> float:
    if b0 is None or b1 is None:
        return 0.0
    return max(0, min(a1, int(b1)) - max(a0, int(b0))) / 1e6


def overlap_report(profiles: Mapping[int, Mapping[str, Any]],
                   decisions: Any) -> list[dict[str, Any]]:
    """Per object-GT build: overlap (ms, same raw clock) with UE front/codec/send.

    ``front`` is front_start -> front_end (front, ranker, AE and codec in
    ``ContinuousUERuntimeV2.prepare``); ``send`` is first -> last datagram.
    """
    out = []
    for frame_id, prof in sorted(profiles.items()):
        s, e = prof.get("start_raw_ns"), prof.get("end_raw_ns")
        if s is None or e is None:
            continue
        front = send = input7 = 0.0
        overlapping: list[int] = []
        for record in decisions or ():
            st = record.get("stages") or {}
            f = _interval_overlap_ms(s, e, st.get("front_start_raw_ns"), st.get("front_end_raw_ns"))
            n = _interval_overlap_ms(s, e, st.get("first_packet_send_raw_ns"),
                                     st.get("last_packet_send_raw_ns"))
            i = _interval_overlap_ms(s, e, st.get("input_7ch_start_raw_ns"),
                                     st.get("front_start_raw_ns"))
            if f or n or i:
                overlapping.append(int(record.get("frame_id", -1)))
            front, send, input7 = front + f, send + n, input7 + i
        out.append({"frame_id": int(frame_id), "queue_class": prof.get("queue_class"),
                    "builder_wall_ms": prof.get("wall_ms"),
                    "overlap_front_codec_ms": front, "overlap_send_ms": send,
                    "overlap_input_7ch_ms": input7, "overlapping_frames": overlapping})
    return out


__all__ = ["PATTERNS", "STAGES", "LowObjectGtPreempted", "ObjectGtProfileV2",
           "build_object_rows_v2", "support_parameters", "LastDatagramMarksV2",
           "RewardPendingGateV2", "DeferringRewardPriorityGtQueueV2",
           "GtTicketLogV3", "overlap_report", "raw_now_ns", "LOW_SKIPPED_STATUS"]
