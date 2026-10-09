"""Addendum-9 offline parity and scheduling tests for the object-GT repair.

Both arms drive the unchanged pinned ``PassiveSplitCollector._ground_truth``
(which calls ``valid_localization_objects``) on identical frozen snapshots
built from the pinned ``FrozenActor``/``FrozenWorld`` types. The reference
arm uses the pinned builder; the repaired arm routes ``build_object_rows``
through the collector hook. No CARLA server, OAI, Docker, CUDA or network.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
import types
import unittest
from pathlib import Path

import numpy as np

from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned

from . import phase6_gt_priority_v2 as GP
from . import phase6_object_gt_v2 as OG
from . import phase6_ue_runtime_v2 as U
from . import run4_live_wire_v2 as W

ROOT = Path(__file__).resolve().parents[2]

# Exactly what the pinned run_cell does before the collector imports the
# parked builder: expose the legacy fusion package on the namespace path.
import pole_lraspp_multimodal_fusion as _fusion_namespace  # noqa: E402

_LEGACY = (ROOT / "pole_lraspp_multimodal_fusion" / "pole_lraspp_multimodal_fusion").resolve()
if str(_LEGACY) not in {str(Path(p).resolve()) for p in _fusion_namespace.__path__}:
    _fusion_namespace.__path__.append(str(_LEGACY))
import carla_collect_parked_ego_fusion_training_data as parked  # noqa: E402
from pole_lraspp_multimodal_fusion.pole_lraspp_multimodal_fusion.object_targets import (  # noqa: E402,F401
    valid_localization_objects,
)

WIDTH, HEIGHT = 768, 448


# ---------------------------------------------------------------------------
# carla-shaped value types (float32 storage, like carla.Location)
# ---------------------------------------------------------------------------


def _f32(value: float) -> float:
    return float(np.float32(value))


class Loc:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = _f32(x), _f32(y), _f32(z)

    def distance(self, other: "Loc") -> float:
        d = np.float32(np.sqrt(np.float32((self.x - other.x) ** 2 + (self.y - other.y) ** 2
                                          + (self.z - other.z) ** 2)))
        return float(d)


class Rot:
    def __init__(self, yaw: float) -> None:
        self.pitch, self.yaw, self.roll = 0.0, _f32(yaw), 0.0


class Tf:
    def __init__(self, loc: Loc, yaw: float) -> None:
        self.location, self.rotation = loc, Rot(yaw)

    def _m(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.rotation.yaw)), math.sin(math.radians(self.rotation.yaw))
        m = np.eye(4, dtype=np.float32)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        m[:3, 3] = (self.location.x, self.location.y, self.location.z)
        return m

    def get_matrix(self) -> list:
        return self._m().astype(np.float64).tolist()

    def get_inverse_matrix(self) -> list:
        return np.linalg.inv(self._m().astype(np.float64)).astype(np.float32).astype(
            np.float64).tolist()


class BBox:
    def __init__(self, ex: float, ey: float, ez: float) -> None:
        self.location, self.extent = Loc(0.0, 0.0, ez), Loc(ex, ey, ez)


class Vel:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y, self.z = _f32(x), _f32(y), 0.0


class BrokenActor(pinned.FrozenActor):
    __slots__ = ()

    def get_transform(self):
        raise RuntimeError("actor destroyed")


CAMERA = Tf(Loc(0.0, 0.0, 1.8), 0.0)
CAMERA_INVERSE = np.asarray(CAMERA.get_inverse_matrix(), dtype=np.float64)
CAMERA_MATRIX = np.asarray(CAMERA.get_matrix(), dtype=np.float64)
FOCAL = WIDTH / (2.0 * math.tan(math.radians(90.0) / 2.0))
INTRINSICS = np.asarray([[FOCAL, 0.0, WIDTH / 2.0], [0.0, FOCAL, HEIGHT / 2.0],
                         [0.0, 0.0, 1.0]])
EGO = types.SimpleNamespace(id=1)


def scene_sequence(seed: int, frames: int = 5):
    """Frozen worlds for consecutive frames (moving and parked actors)."""
    rng = np.random.default_rng(seed)
    specs = []
    for index in range(90):
        person = index >= 60
        # camera looks along +x, so the visible +x half-plane is sampled densely
        distance = float(rng.uniform(2.0, 170.0))
        bearing = float(rng.uniform(-100.0, 100.0))
        specs.append({
            "id": 100 + index,
            "type_id": "walker.pedestrian.0001" if person else "vehicle.tesla.model3",
            "bbox": BBox(0.3, 0.3, 0.9) if person else BBox(2.3, 1.0, 0.8),
            "x": distance * math.cos(math.radians(bearing)),
            "y": distance * math.sin(math.radians(bearing)),
            "yaw": float(rng.uniform(-180.0, 180.0)),
            "v": (0.0, 0.0) if rng.random() < 0.4 else tuple(rng.uniform(-6.0, 6.0, 2)),
        })
    worlds, radar = [], []
    for frame in range(frames):
        actors = [pinned.FrozenActor(actor_id=1, type_id="vehicle.ego",
                                     bounding_box=BBox(2.3, 1.0, 0.8),
                                     transform=Tf(Loc(0.0, 0.0, 0.0), 0.0),
                                     velocity=Vel(0.0, 0.0))]
        points = [rng.normal(0.0, 60.0, (1500, 3))]
        for spec in specs:
            dt = 0.1 * frame
            x, y = spec["x"] + spec["v"][0] * dt, spec["y"] + spec["v"][1] * dt
            actors.append(pinned.FrozenActor(
                actor_id=spec["id"], type_id=spec["type_id"], bounding_box=spec["bbox"],
                transform=Tf(Loc(x, y, 0.0), spec["yaw"]), velocity=Vel(*spec["v"])))
            points.append(np.asarray([x, y, 0.8]) + rng.normal(0.0, 0.7, (12, 3)))
        actors.append(BrokenActor(actor_id=999, type_id="vehicle.broken",
                                  bounding_box=BBox(1, 1, 1), transform=None, velocity=None))
        worlds.append(pinned.FrozenWorld(pinned.FrozenActorList(actors)))
        radar.append({"world_xyz": np.concatenate(points).astype(np.float64)})
    return worlds, radar


def gt_host(parked_like, tracker):
    return types.SimpleNamespace(
        parked=parked_like, world=object(), ego=EGO, camera=types.SimpleNamespace(
            get_transform=lambda: CAMERA), intrinsics=INTRINSICS,
        model_size=(WIDTH, HEIGHT), actor_tracker=tracker, min_gt_area_px=16.0,
        max_gt_distance_m=40.0)


COLLECTOR = U.build_run4_collector_class(type("Base", (), {}))


class RepairedHost:
    """Exactly the attributes ``_run4_object_rows`` touches."""

    def __init__(self, tracker, *, reward_frames=()) -> None:
        self.world = object()
        self.max_gt_distance_m = 40.0
        self.live = types.SimpleNamespace(
            _run4_identity={f: {"reward_requested": f in reward_frames} for f in range(64)},
            gt_log=OG.GtTicketLogV3())
        self._run4_gate = OG.RewardPendingGateV2()
        self.parked = U._PreparationTimer(parked, lambda real, **kw: real(**kw),
                                          self._object_rows)
        self.gt = gt_host(self.parked, tracker)

    def _object_rows(self, real_build, **kwargs):
        return COLLECTOR._run4_object_rows(self, real_build, **kwargs)

    def ground_truth(self, frame, world, radar):
        return pinned.PassiveSplitCollector._ground_truth(
            self.gt, frame_id=frame, timestamp=frame / 10.0, camera_matrix=CAMERA_MATRIX,
            camera_inverse=CAMERA_INVERSE, radar_points=radar, world=world,
            camera_location=CAMERA.location)

def reference_ground_truth(tracker, frame, world, radar):
    return pinned.PassiveSplitCollector._ground_truth(
        gt_host(parked, tracker), frame_id=frame, timestamp=frame / 10.0,
        camera_matrix=CAMERA_MATRIX, camera_inverse=CAMERA_INVERSE, radar_points=radar,
        world=world, camera_location=CAMERA.location)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=repr, allow_nan=True,
                      separators=(",", ":"), ensure_ascii=True)


def exact(rows) -> str:
    """Bit-exact serialization (float.hex for every float)."""
    def walk(v):
        if isinstance(v, float):
            return v.hex()
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        return v
    return canonical(walk(rows))


def tracker_state(tracker) -> str:
    return exact({"ages": dict(sorted(tracker._ages.items())),
                  "last": dict(sorted(tracker._last_time.items()))})


# ---------------------------------------------------------------------------
# Parity gates
# ---------------------------------------------------------------------------


class BuilderParityTest(unittest.TestCase):
    def test_unlimited_v2_equals_pinned_rows_and_tracker(self) -> None:
        worlds, radar = scene_sequence(1)
        t_ref = parked.ActorStationaryTracker(0.35, 5.0)
        t_new = copy.deepcopy(t_ref)
        for frame, (world, points) in enumerate(zip(worlds, radar)):
            kwargs = dict(world=world, ego_vehicle=EGO,
                          sample_base={"timestamp": frame / 10.0, "frame_id": frame},
                          camera_location=CAMERA.location, camera_matrix=CAMERA_MATRIX,
                          camera_inverse_matrix=CAMERA_INVERSE, intrinsics=INTRINSICS,
                          width=WIDTH, height=HEIGHT, max_distance_m=140.0,
                          radar_world_xyz=points["world_xyz"], include_pedestrians=True,
                          **COLLECTOR._run4_support_kwargs())
            ref = parked.build_object_rows(stationary_tracker=t_ref, **kwargs)
            new = OG.build_object_rows_v2(parked, stationary_tracker=t_new, **kwargs)
            self.assertEqual(exact(ref), exact(new))
            self.assertGreater(len(ref), 20)
        self.assertEqual(tracker_state(t_ref), tracker_state(t_new))

    def test_limited_build_gives_identical_eligible_targets_over_a_sequence(self) -> None:
        for seed in (2, 3, 4):
            worlds, radar = scene_sequence(seed)
            t_ref = parked.ActorStationaryTracker(0.35, 5.0)
            host = RepairedHost(copy.deepcopy(t_ref))
            for frame, (world, points) in enumerate(zip(worlds, radar)):
                ref = reference_ground_truth(t_ref, frame, world, points)
                new = host.ground_truth(frame, world, points)
                self.assertEqual(exact(ref), exact(new))          # full target rows
                self.assertEqual([(o["class_name"], o["world_x"], o["world_y"]) for o in ref],
                                 [(o["class_name"], o["world_x"], o["world_y"]) for o in new])
                self.assertTrue(ref, "scene must contain eligible targets")
                prof = host.live.gt_log.snapshot()["tickets"][-1]["object_builder"]
                counts = prof["actor_counts"]
                self.assertGreater(counts["beyond_eligibility_limit"], 0)
                self.assertEqual(counts["within_eligibility_limit"], counts["rows_built"])
                self.assertLessEqual(counts["within_eligibility_limit"], counts["projected"])
                self.assertLessEqual(counts["projected"], counts["within_builder_limit"])
                self.assertEqual(counts["runtime_error_dropped"], 1)
            # tracker updated for every projected actor up to 140 m, exactly as before
            self.assertEqual(tracker_state(t_ref), tracker_state(host.gt.actor_tracker))
    
    def test_limit_is_the_identical_actor_origin_distance_comparison(self) -> None:
        actor = pinned.FrozenActor(actor_id=5, type_id="vehicle.x", bounding_box=BBox(2, 1, 1),
                                   transform=Tf(Loc(40.0, 0.0, 1.8), 0.0),
                                   velocity=Vel(0, 0))
        edge = pinned.FrozenActor(actor_id=6, type_id="vehicle.x", bounding_box=BBox(2, 1, 1),
                                  transform=Tf(Loc(np.nextafter(np.float32(40.0),
                                                                np.float32(41.0)), 3.0, 1.8),
                                               0.0), velocity=Vel(0, 0))
        world = pinned.FrozenWorld(pinned.FrozenActorList([actor, edge]))
        radar = {"world_xyz": np.zeros((0, 3))}
        t_ref = parked.ActorStationaryTracker(0.35, 5.0)
        host = RepairedHost(copy.deepcopy(t_ref))
        self.assertEqual(exact(reference_ground_truth(t_ref, 0, world, radar)),
                         exact(host.ground_truth(0, world, radar)))

    def test_identical_prediction_inputs_give_bit_identical_q_perc(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q

        spec = W.load_run4_quality_spec(ROOT)
        worlds, radar = scene_sequence(8, frames=3)
        rng = np.random.default_rng(8)
        t_ref = parked.ActorStationaryTracker(0.35, 5.0)
        host = RepairedHost(copy.deepcopy(t_ref))
        defined = 0
        for frame, (world, points) in enumerate(zip(worlds, radar)):
            ref = reference_ground_truth(t_ref, frame, world, points)
            new = host.ground_truth(frame, world, points)
            predictions = [{"class_name": o["class_name"],
                            "world_x": o["world_x"] + float(rng.normal(0, 0.6)),
                            "world_y": o["world_y"] + float(rng.normal(0, 0.6))}
                           for o in ref[::2]]
            mask_p = rng.integers(0, 3, (HEIGHT, WIDTH), dtype=np.uint8)
            mask_t = rng.integers(0, 3, (HEIGHT, WIDTH), dtype=np.uint8)
            m_ref = W.live_measurement(frame_id=frame, predicted_mask=mask_p,
                                       ground_truth_mask=mask_t, predictions=predictions,
                                       ground_truth_objects=ref, match_distance_m=2.0)
            m_new = W.live_measurement(frame_id=frame, predicted_mask=mask_p,
                                       ground_truth_mask=mask_t, predictions=predictions,
                                       ground_truth_objects=new, match_distance_m=2.0)
            self.assertEqual(exact(m_ref), exact(m_new))
            q_ref, q_new = (Q.evaluate_exact_quality(spec, m_ref),
                            Q.evaluate_exact_quality(spec, m_new))
            self.assertEqual(exact(q_ref), exact(q_new))
            defined += q_ref is not None
        self.assertGreater(defined, 0)

    def test_live_q_perc_consumes_only_class_and_world_xy(self) -> None:
        """Velocity, stationary tracking and radar support are carried, not used."""
        worlds, radar = scene_sequence(9, frames=1)
        ref = reference_ground_truth(parked.ActorStationaryTracker(0.35, 5.0), 0,
                                     worlds[0], radar[0])
        stripped = [{"class_name": o["class_name"], "world_x": o["world_x"],
                     "world_y": o["world_y"]} for o in ref]
        preds = [dict(o) for o in stripped[::3]]
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        full = W.live_measurement(frame_id=0, predicted_mask=mask, ground_truth_mask=mask,
                                  predictions=preds, ground_truth_objects=ref,
                                  match_distance_m=2.0)
        minimal = W.live_measurement(frame_id=0, predicted_mask=mask, ground_truth_mask=mask,
                                     predictions=preds, ground_truth_objects=stripped,
                                     match_distance_m=2.0)
        self.assertEqual(exact(full), exact(minimal))


class SemanticAndIdentityTest(unittest.TestCase):
    def test_semantic_path_and_identities_are_untouched(self) -> None:
        for name in ("_semantic_for", "_segmentation_worker", "_records_for",
                     "identity_for_frame"):
            self.assertNotIn(name, COLLECTOR.__dict__)
        proxy = U._PreparationTimer(parked, lambda real, **kw: "radar", lambda real, **kw: "obj")
        self.assertEqual(proxy.build_radar_sample(), "radar")
        self.assertEqual(proxy.build_object_rows(), "obj")
        self.assertIs(proxy.semantic_tags_from_image, parked.semantic_tags_from_image)
        self.assertIs(proxy.ActorStationaryTracker, parked.ActorStationaryTracker)
        src = Path(OG.__file__).read_text(encoding="utf-8")
        for forbidden in ("write_semantic_ground_truth", "identity_for_frame",
                          "import multiprocessing", "concurrent.futures",
                          "session_uuid", "ticket_seq", "decision_seq"):
            self.assertNotIn(forbidden, src)

    def test_live_world_diagnostic_path_calls_pinned_builder_unchanged(self) -> None:
        host = RepairedHost(parked.ActorStationaryTracker(0.35, 5.0))
        calls = []
        result = COLLECTOR._run4_object_rows(host, lambda **kw: calls.append(kw) or ["x"],
                                             world=host.world, sample_base={"frame_id": 1})
        self.assertEqual(result, ["x"])
        self.assertEqual(len(calls), 1)


# ---------------------------------------------------------------------------
# LOW never runs while a reward ticket is pending or running
# ---------------------------------------------------------------------------


def _classifier(high: set):
    return lambda item: GP.HIGH if int(item["frame_id"]) in high else GP.LOW


def _ticket(frame: int) -> dict:
    return {"frame_id": frame}


class LowDeferralTest(unittest.TestCase):
    def queue(self, high=()):
        gate, skipped = OG.RewardPendingGateV2(), []
        q = OG.DeferringRewardPriorityGtQueueV2(
            classify=_classifier(set(high)), low_blocked=gate.blocked,
            on_skip=lambda frame, reason: skipped.append((frame, reason)))
        return q, gate, skipped

    def test_low_arriving_while_reward_pending_is_skipped_explicitly(self) -> None:
        q, gate, skipped = self.queue(high={5})
        gate.open(5)
        q.put_nowait(_ticket(4))
        self.assertEqual(skipped, [(4, "REWARD_PENDING_AT_ENQUEUE")])
        self.assertEqual(q.unfinished_tasks, 0)
        with self.assertRaises(GP.GtQueueError):
            q.put_nowait(_ticket(4))                      # identity still single-use
        q.put_nowait(_ticket(5))
        self.assertEqual(q.get(timeout=0.1)["frame_id"], 5)

    def test_queued_low_is_skipped_once_the_gate_opens(self) -> None:
        q, gate, skipped = self.queue()
        q.put_nowait(_ticket(1))
        q.put_nowait(_ticket(2))
        gate.open(9)
        import queue as _queue
        with self.assertRaises(_queue.Empty):
            q.get(timeout=0.01)
        self.assertEqual(skipped, [(1, "REWARD_PENDING_WHILE_QUEUED"),
                                   (2, "REWARD_PENDING_WHILE_QUEUED")])
        self.assertEqual(q.unfinished_tasks, 0)
        gate.close(9, "OBJECT_GT_DONE")
        q.put_nowait(_ticket(3))
        self.assertEqual(q.get(timeout=0.1)["frame_id"], 3)

    def test_running_low_yields_when_a_reward_ticket_opens(self) -> None:
        worlds, radar = scene_sequence(10, frames=1)
        host = RepairedHost(parked.ActorStationaryTracker(0.35, 5.0))
        calls = {"n": 0}

        def blocked() -> bool:
            calls["n"] += 1
            return calls["n"] > 10                        # reward opens mid-build
        host._run4_gate.blocked = blocked
        with self.assertRaises(OG.LowObjectGtPreempted):
            host.ground_truth(3, worlds[0], radar[0])
        prof = host.live.gt_log.snapshot()["tickets"][-1]["object_builder"]
        self.assertEqual(prof["outcome"], "LOW_PREEMPTED")
        self.assertEqual(prof["actor_counts"]["preempted_at_actor"], 10)

    def test_high_is_never_preempted(self) -> None:
        worlds, radar = scene_sequence(11, frames=1)
        host = RepairedHost(parked.ActorStationaryTracker(0.35, 5.0), reward_frames={3})
        host._run4_gate.open(3)
        host._run4_gate.open(4)
        self.assertTrue(host.ground_truth(3, worlds[0], radar[0]))

    def test_pinned_worker_accounts_every_ticket_with_skips(self) -> None:
        from .test_phase6_prewarm_gt_priority_v2 import FakeHost, _drain

        gate = OG.RewardPendingGateV2()
        q = OG.DeferringRewardPriorityGtQueueV2(
            classify=_classifier({4}), low_blocked=gate.blocked,
            on_skip=lambda frame, reason: host.evaluation_errors.__setitem__(
                frame, f"{OG.LOW_SKIPPED_STATUS}:{reason}"))
        host = FakeHost(q)
        thread = host.run()
        q.put_nowait({**_full_ticket(1)})
        time.sleep(0.05)
        gate.open(4)
        q.put_nowait({**_full_ticket(2)})                 # hold during reward pending
        q.put_nowait({**_full_ticket(4)})
        deadline = time.monotonic() + 2.0
        while 4 not in host.source_gt and time.monotonic() < deadline:
            time.sleep(0.005)
        gate.close(4, "OBJECT_GT_DONE")
        q.put_nowait({**_full_ticket(5)})
        self.assertTrue(_drain(host, thread))
        self.assertEqual(sorted(host.source_gt), [1, 4, 5])
        self.assertTrue(host.evaluation_errors[2].startswith(OG.LOW_SKIPPED_STATUS))
        self.assertEqual(q.unfinished_tasks, 0)


def _full_ticket(frame: int) -> dict:
    return {"frame_id": frame, "timestamp": frame / 10.0, "camera_matrix": None,
            "camera_inverse": None, "radar_points": {}, "scene": object(),
            "camera_location": None}


class InstrumentationTest(unittest.TestCase):
    def test_profile_records_wall_cpu_ctx_and_every_stage(self) -> None:
        worlds, radar = scene_sequence(12, frames=1)
        host = RepairedHost(parked.ActorStationaryTracker(0.35, 5.0))
        host.ground_truth(0, worlds[0], radar[0])
        prof = host.live.gt_log.snapshot()["tickets"][-1]["object_builder"]
        for key in ("wall_ms", "thread_cpu_ms", "voluntary_ctx_switches",
                    "involuntary_ctx_switches", "start_raw_ns", "end_raw_ns",
                    "start_wall_ns", "end_wall_ns"):
            self.assertIn(key, prof)
        self.assertEqual(set(prof["stage_ms"]), set(OG.STAGES))
        self.assertGreater(prof["actor_counts"]["actors_total"], 90)
        self.assertEqual(prof["queue_class"], "LOW")
        self.assertEqual(prof["outcome"], "COMPLETED")

    def test_overlap_with_front_codec_and_send_uses_one_raw_clock(self) -> None:
        profiles = {7: {"start_raw_ns": 1_000, "end_raw_ns": 5_000_000,
                        "queue_class": "HIGH", "wall_ms": 5.0}}
        decisions = [{"frame_id": 9, "stages": {
            "input_7ch_start_raw_ns": 0, "front_start_raw_ns": 1_000_000,
            "front_end_raw_ns": 3_000_000, "first_packet_send_raw_ns": 4_000_000,
            "last_packet_send_raw_ns": 9_000_000}}]
        (row,) = OG.overlap_report(profiles, decisions)
        self.assertAlmostEqual(row["overlap_front_codec_ms"], 2.0)
        self.assertAlmostEqual(row["overlap_send_ms"], 1.0)
        self.assertAlmostEqual(row["overlap_input_7ch_ms"], 0.999)
        self.assertEqual(row["overlapping_frames"], [9])

    def test_module_import_starts_nothing(self) -> None:
        import subprocess

        code = ("import threading, rl_agent.splitfusion_hybrid_sac_live_route_b_v2."
                "phase6_object_gt_v2; print(threading.active_count())")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                             text=True, timeout=120, check=True)
        self.assertEqual(out.stdout.strip(), "1")


# ---------------------------------------------------------------------------
# Addendum 10: reward GT is admitted only after the last datagram is sent
# ---------------------------------------------------------------------------

SEMANTIC_FILES_AT_00DC504 = {
    "reward_hold_controller_v2.py": "ea90b149d84a29a3a008472891246384cec4431f52a4c159ee8a0ea4a05d3f3c",
    "phase6_decision_engine_v2.py": "bea2e8467c4d1dd1a7d1b26d59faa3c48af127bd6b83a4f42f6757c5b96d0a16",
    "run4_live_wire_v2.py": "2337edbfa2613a418a76a76235c190638cc088576903a61b5e408aabba669924",
    "run4_map_protocol_v2.py": "ef6489f266ef3ca523b596edcece445f55b78eb24367a2789b4697089d02a6b5",
    "phase6_edge_runtime_v2.py": "8446eaa50ec881028ff66e0718060410edf1c066b1f1589fc985457eb1111cd6",
    "continuous_execution_v2.py": "3dd0ec16cf70e52585b5d20e23d1fe47a443eb6a1e01c4304ac7ce07728f9184",
    "live_state_v2.py": "c74b2d1c43ef878651c5394802bd80213d41ea4baf3e67c59b23822e80c0a21a",
    "frozen_actor_v2.py": "8851e3a0e10008ecd36b2be842a599c51d1de234584baa3581787e490cd29c8d",
}


class SendOrderHost:
    """The attributes ``_run4_gt_class`` and the pinned GT worker touch."""

    def __init__(self, *, reward_frames=(), compute_s=0.0) -> None:
        from .test_phase6_prewarm_gt_priority_v2 import FakeHost

        identities = {f: {"reward_requested": f in reward_frames, "frame_id": f,
                          "session_uuid": "s-1", "decision_seq": f, "ticket_seq": f}
                      for f in range(32)}
        self.live = types.SimpleNamespace(
            _run4_identity=identities,
            _gt_identity={f: {"frame_id": f, "run_id": "r"} for f in range(32)},
            last_datagram=OG.LastDatagramMarksV2(), gt_log=OG.GtTicketLogV3())
        self.gate = OG.RewardPendingGateV2()
        self.skipped: list = []
        self.queue = OG.DeferringRewardPriorityGtQueueV2(
            classify=lambda item: COLLECTOR._run4_gt_class(self, item),
            low_blocked=self.gate.blocked,
            on_skip=lambda frame, reason: self.skipped.append((frame, reason)),
            on_enqueue=self.live.gt_log.enqueued, on_dequeue=self.live.gt_log.dequeued)
        self.worker = FakeHost(self.queue, compute_s=compute_s)


def _gt_ticket(frame: int) -> dict:
    return {"frame_id": frame, "timestamp": frame / 10.0, "camera_matrix": None,
            "camera_inverse": None, "radar_points": {}, "scene": object(),
            "camera_location": None}


class SendOrderedRewardGtTest(unittest.TestCase):
    def test_reward_gt_cannot_start_before_last_datagram_sent(self) -> None:
        from .test_phase6_prewarm_gt_priority_v2 import _drain

        host = SendOrderHost(reward_frames={4})
        thread = host.worker.run()
        with self.assertRaisesRegex(GP.GtQueueError, "before last-datagram-sent"):
            host.queue.put_nowait(_gt_ticket(4))
        time.sleep(0.05)
        self.assertEqual(host.worker.order, [])            # never started
        self.assertEqual(host.queue.unfinished_tasks, 0)
        self.assertTrue(_drain(host.worker, thread))

    def test_reward_gt_starts_exactly_once_after_the_mark(self) -> None:
        from .test_phase6_prewarm_gt_priority_v2 import _drain

        host = SendOrderHost(reward_frames={4})
        thread = host.worker.run()
        self.assertTrue(host.live.last_datagram.mark(4, 1_000))
        host.queue.put_nowait(_gt_ticket(4))
        with self.assertRaises(GP.GtQueueError):            # a second ticket is refused
            host.queue.put_nowait(_gt_ticket(4))
        self.assertTrue(_drain(host.worker, thread))
        self.assertEqual(host.worker.order, [4])
        self.assertEqual(sorted(host.worker.source_gt), [4])
        row = {r["frame_id"]: r for r in host.live.gt_log.snapshot()["tickets"]}[4]
        self.assertEqual(row["last_datagram_raw_ns"], 1_000)
        self.assertEqual(row["queue_class"], "HIGH")
        self.assertGreaterEqual(row["gt_enqueue_raw_ns"], 1_000)

    def test_duplicate_send_completion_cannot_duplicate_gt(self) -> None:
        from .test_phase6_prewarm_gt_priority_v2 import _drain

        host = SendOrderHost(reward_frames={6})
        thread = host.worker.run()
        self.assertTrue(host.live.last_datagram.mark(6, 500))
        self.assertFalse(host.live.last_datagram.mark(6, 900))    # duplicate completion
        self.assertEqual(host.live.last_datagram.get(6), 500)       # first instant kept
        self.assertEqual(host.live.last_datagram.snapshot()["duplicates"], {"6": 1})
        self.assertEqual(host.queue.qsize(), 0)                     # marking enqueues nothing
        host.queue.put_nowait(_gt_ticket(6))
        self.assertTrue(_drain(host.worker, thread))
        self.assertEqual(host.worker.order, [6])

    def test_marking_never_enqueues_computes_or_touches_identity(self) -> None:
        host = SendOrderHost(reward_frames={2})
        before = copy.deepcopy((host.live._run4_identity, host.live._gt_identity))
        host.live.last_datagram.mark(2, 42)
        self.assertEqual(host.queue.qsize(), 0)
        self.assertEqual(host.worker.order, [])
        self.assertEqual(before, (host.live._run4_identity, host.live._gt_identity))
        self.assertEqual(COLLECTOR._run4_gt_class(host, {"frame_id": 2}), GP.HIGH)
        self.assertEqual(host.live._run4_identity[2]["ticket_seq"], 2)

    def test_low_work_cannot_delay_reward_gt(self) -> None:
        from .test_phase6_prewarm_gt_priority_v2 import _drain

        host = SendOrderHost(reward_frames={5}, compute_s=0.002)
        host.queue.put_nowait(_gt_ticket(1))            # LOW queued before decision open
        host.gate.open(5)                               # decision open (reward)
        host.queue.put_nowait(_gt_ticket(3))            # LOW during the reward window
        host.live.last_datagram.mark(5, 10)
        host.queue.put_nowait(_gt_ticket(5))
        thread = host.worker.run()
        deadline = time.monotonic() + 2.0
        while 5 not in host.worker.source_gt and time.monotonic() < deadline:
            time.sleep(0.002)
        host.gate.close(5, "OBJECT_GT_DONE")
        self.assertTrue(_drain(host.worker, thread))
        self.assertEqual(host.worker.order, [5])        # no LOW ran before or during it
        self.assertEqual(sorted(host.skipped), [(1, "REWARD_PENDING_WHILE_QUEUED"),
                                                (3, "REWARD_PENDING_AT_ENQUEUE")])
        row = {r["frame_id"]: r for r in host.live.gt_log.snapshot()["tickets"]}[5]
        self.assertLess(row["queue_wait_ms"], 50.0)

    def test_low_tickets_need_no_send_mark(self) -> None:
        host = SendOrderHost()
        self.assertEqual(COLLECTOR._run4_gt_class(host, {"frame_id": 3}), GP.LOW)

    def test_submit_marks_after_the_last_sendto_and_before_return(self) -> None:
        import inspect

        src = inspect.getsource(U.build_run4_runtime_class)
        submit = src[src.index("        def submit("):src.index("        def _result_loop(")]
        last = submit.index('stages["last_packet_send_raw_ns"] = T.raw_now_ns()')
        mark = submit.index("self.last_datagram.mark(int(frame_id)")
        loop = submit.index("self.sender.sendto(chunk, self.remote)")
        ret = submit.index('return {"sent": True')
        self.assertLess(loop, last)
        self.assertLess(last, mark)
        self.assertLess(mark, ret)
        self.assertNotIn("evaluation_queue", submit)
        self.assertNotIn("gt_queue", submit)
        self.assertNotIn("_ground_truth", submit)

    def test_no_helper_thread_pool_or_new_queue_remains(self) -> None:
        import inspect

        og = Path(OG.__file__).read_text(encoding="utf-8")
        self.assertNotIn("threading.Thread(", og)
        self.assertNotIn("Prefetch", og)
        self.assertNotIn("Precompute", og)
        for forbidden in ("concurrent.futures", "import multiprocessing", "ProcessPool",
                          "ThreadPool"):
            self.assertNotIn(forbidden, og)
        collector_src = inspect.getsource(U.build_run4_collector_class)
        self.assertNotIn("prefetch", collector_src.lower())
        self.assertNotIn("threading.Thread(", collector_src)
        self.assertEqual(collector_src.count("DeferringRewardPriorityGtQueueV2("), 1)
        planned = inspect.getsource(COLLECTOR._run4_on_planned)
        self.assertNotIn("_quality_scenes", planned)          # no GT input at decision open

    def test_reward_deadline_and_terminal_semantics_unchanged(self) -> None:
        import hashlib

        from . import reward_hold_controller_v2 as R

        for name, digest in SEMANTIC_FILES_AT_00DC504.items():
            data = (Path(__file__).parent / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)
        self.assertEqual(R.REWARD_DEADLINE_NS, 170_000_000)
        self.assertEqual(R.K_MIN, 2)

    def test_addendum_10_binds_v8_v9_evidence_and_is_prospective(self) -> None:
        import hashlib

        doc = json.loads((Path(__file__).parent / "phase6_gt_send_order_addendum_10.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE")
        self.assertEqual(doc["base_commit_prefix"], "00dc504")
        self.assertEqual(doc["handshake"]["acceptance"]["feedback_after_action_open_ms_max"],
                         170.0)
        self.assertTrue(doc["handshake"]["no_300_frame_run"])
        self.assertEqual(len(doc["evidence_sha256_v8_v9"]), 116)
        for rel, digest in doc["evidence_sha256_v8_v9"].items():
            path = ROOT / rel
            if path.is_file():             # evidence is kept off Git; verify when present
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, rel)


class CarlaProbeOfflineTest(unittest.TestCase):
    """The Phase-C probe's summary, gates, shadow and stub runtime, offline."""

    def _evidence(self, *, high_ready_ms: float, low_overlap: bool, queue_wait: float):
        from . import phase6_object_gt_carla_probe_v2 as P  # noqa: F401

        base_wall, base_raw = 1_000_000_000_000, 5_000_000_000
        records, tickets, events = [], [], []
        for index, frame in enumerate((10, 12)):
            reward = index == 0
            ao_wall, ao_raw = base_wall + index * 100_000_000, base_raw + index * 100_000_000
            records.append({"frame_id": frame, "reward_requested": reward,
                            "action_open_wall_ns": ao_wall, "action_open_raw_ns": ao_raw})
            ready = ao_wall + int(high_ready_ms * 1e6)
            tickets.append({
                "frame_id": frame, "queue_class": "HIGH" if reward else "LOW",
                "queue_wait_ms": queue_wait if reward else 0.1,
                "enqueue_wall_ns": ao_wall + 27_000_000,
                "objects_write_start_wall_ns": ready - 1_000_000,
                "objects_write_end_wall_ns": ready,
                "object_builder": {"wall_ms": 20.0, "thread_cpu_ms": 18.0,
                                   "start_raw_ns": ao_raw + 28_000_000,
                                   "end_raw_ns": ao_raw + 48_000_000,
                                   "end_wall_ns": ready - 2_000_000,
                                   "outcome": "COMPLETED"}})
            if reward:
                events += [{"frame_id": frame, "event": "open", "raw_ns": ao_raw},
                           {"frame_id": frame, "event": "close", "raw_ns": ao_raw + 60_000_000}]
        if low_overlap:          # the LOW builder runs inside the reward window
            tickets[1]["object_builder"].update(start_raw_ns=base_raw + 10_000_000,
                                                end_raw_ns=base_raw + 40_000_000)
        return {"records": records, "gt_objects": {"tickets": tickets},
                "gt_reward_gate": {"events": events}}

    def test_timing_gates_pass_and_fail_on_the_registered_criteria(self) -> None:
        from . import phase6_object_gt_carla_probe_v2 as P

        good = P.timing_summary(self._evidence(high_ready_ms=60.0, low_overlap=False,
                                               queue_wait=0.2))
        self.assertTrue(good["gates"]["passed"], good["gates"])
        (high,) = good["high"]
        self.assertAlmostEqual(high["objects_ready_after_action_open_ms"], 60.0)
        # GT ready before the edge: the v8 edge path bounds the prediction
        self.assertAlmostEqual(high["predicted_feedback_after_action_open_ms"],
                               123.5 + P.V8_GT_DETECT_TO_EMIT_MS + P.V8_EMIT_TO_UE_RECEIPT_MS)
        late = P.timing_summary(self._evidence(high_ready_ms=155.0, low_overlap=False,
                                               queue_wait=0.2))
        self.assertFalse(late["gates"]["objects_ready_max_le_150ms"])
        self.assertFalse(late["gates"]["passed"])
        intruded = P.timing_summary(self._evidence(high_ready_ms=60.0, low_overlap=True,
                                                   queue_wait=0.2))
        self.assertFalse(intruded["gates"]["no_low_delaying_high"])
        waited = P.timing_summary(self._evidence(high_ready_ms=60.0, low_overlap=False,
                                                 queue_wait=9.0))
        self.assertFalse(waited["gates"]["no_low_delaying_high"])

    def test_shadow_parity_detects_equal_and_divergent_rows(self) -> None:
        from . import phase6_object_gt_carla_probe_v2 as P

        worlds, radar = scene_sequence(13, frames=1)
        tracker = parked.ActorStationaryTracker(0.35, 5.0)
        kwargs = dict(world=worlds[0], ego_vehicle=EGO,
                      sample_base={"timestamp": 0.0, "frame_id": 0},
                      camera_location=CAMERA.location, camera_matrix=CAMERA_MATRIX,
                      camera_inverse_matrix=CAMERA_INVERSE, intrinsics=INTRINSICS,
                      width=WIDTH, height=HEIGHT, max_distance_m=140.0,
                      radar_world_xyz=radar[0]["world_xyz"], include_pedestrians=True,
                      stationary_tracker=tracker, **COLLECTOR._run4_support_kwargs())
        before = (dict(tracker._ages), dict(tracker._last_time))
        rows = OG.build_object_rows_v2(parked, eligibility_distance_m=40.0, **kwargs)
        after = (dict(tracker._ages), dict(tracker._last_time))
        item = {"kwargs": kwargs, "tracker_before": before, "rows": rows,
                "tracker_after": after}
        ok = P.shadow_parity(parked, [item], max_gt_distance_m=40.0, min_gt_area_px=16.0,
                             valid_localization_objects=valid_localization_objects)
        self.assertTrue(ok["passed"], ok)
        broken = dict(item, rows=[dict(r, object_world_x=r["object_world_x"] + 1e-9)
                                  for r in rows])
        bad = P.shadow_parity(parked, [broken], max_gt_distance_m=40.0, min_gt_area_px=16.0,
                              valid_localization_objects=valid_localization_objects)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["failures"], [0])

    def test_stub_runtime_opens_the_decision_before_its_emulated_send(self) -> None:
        import tempfile as _tempfile

        from . import phase6_object_gt_carla_probe_v2 as P

        with _tempfile.TemporaryDirectory() as tmp:
            rt = P.ProbeRuntimeV2(campaign={"campaign_id": "c"}, cell={"cell_id": "k"},
                                  attempt_dir=Path(tmp), evidence_out=Path(tmp) / "e.json",
                                  sleep_ms=1.0)
            opened = []
            rt.scene_hooks = lambda frame, ts: ({}, 7)
            rt.reward_planned_hook = lambda frame, ts, reward: opened.append(
                (frame, reward, dict(rt._run4_identity)))
            commits = []
            for frame in (3, 5, 7):
                out = rt.submit(frame_bgr=None, radar_tensor=None, frame_id=frame,
                                capture_timestamp_ns=frame, ego_pose=(0,) * 6,
                                stream_id="s", carla_timestamp=frame / 10.0, capture_id="x",
                                on_commit=lambda: commits.append(1))
                self.assertTrue(out["sent"])
            self.assertEqual([(f, r) for f, r, _ in opened], [(3, True), (5, False), (7, True)])
            self.assertEqual(opened[0][2], {})      # identity is set after decision open
            self.assertEqual(len(commits), 3)
            self.assertEqual(rt.identity_for_frame(5)["frame_id"], 5)
            rt.close()
            evidence = json.loads((Path(tmp) / "e.json").read_text())
            self.assertEqual([r["reward_requested"] for r in evidence["records"]],
                             [True, False, True])


class AddendumNineTest(unittest.TestCase):
    def test_addendum_binds_v8_evidence_and_registers_before_evidence(self) -> None:
        import hashlib

        doc = json.loads((Path(__file__).parent / "phase6_object_gt_repair_addendum_9.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE")
        self.assertEqual(doc["base_commit_prefix"], "43b82b6")
        self.assertEqual(doc["claim_scope"], "SYSTEMS_INTEGRATION_QUALIFICATION_ONLY")
        self.assertEqual(doc["v8_result_unchanged"]["classification"], "FAIL")
        files = doc["v8_result_unchanged"]["files_sha256"]
        self.assertEqual(len(files), 51)
        for rel, digest in files.items():
            path = ROOT / rel
            if path.is_file():             # evidence is kept off Git; verify when present
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, rel)
        self.assertTrue(doc["phase_d"]["no_300_frame_run"])
        self.assertEqual(doc["phase_d"]["retry"], "none")
        self.assertEqual(doc["phase_c"]["criteria"]["objects_ready_after_action_open_max_ms"],
                         150.0)
        from . import phase6_object_gt_carla_probe_v2 as P

        self.assertEqual(P.TARGET_GT_READY_MS, 150.0)
        self.assertEqual(P.DEADLINE_MS, 170.0)
        self.assertIn(P.EXECUTE_TOKEN, doc["phase_c"]["command"])


try:  # real carla value types (the CARLA wheel, no server) when importable
    import carla as _carla
except ImportError:  # pragma: no cover - system interpreter
    _carla = None


@unittest.skipUnless(_carla is not None, "carla wheel not importable in this interpreter")
class RealCarlaTypesParityTest(unittest.TestCase):
    def test_limited_build_parity_with_carla_transforms_and_locations(self) -> None:
        rng = np.random.default_rng(21)
        cam = _carla.Transform(_carla.Location(3.0, -2.0, 1.8), _carla.Rotation(yaw=12.0))
        inverse = np.asarray(cam.get_inverse_matrix(), dtype=np.float64)
        matrix = np.asarray(cam.get_matrix(), dtype=np.float64)
        t_ref = parked.ActorStationaryTracker(0.35, 5.0)
        host = RepairedHost(copy.deepcopy(t_ref), reward_frames={1, 3})
        specs = [(100 + i, "walker.pedestrian.0002" if i % 3 == 0 else "vehicle.audi.a2",
                  float(rng.uniform(0, 160)), float(rng.uniform(-80, 80)),
                  float(rng.uniform(-180, 180)), float(rng.uniform(-4, 4))) for i in range(80)]
        for frame in range(5):
            actors = []
            for aid, type_id, dist, bearing, yaw, v in specs:
                x = 3.0 + dist * math.cos(math.radians(bearing + 12.0)) + v * 0.1 * frame
                y = -2.0 + dist * math.sin(math.radians(bearing + 12.0))
                extent = (_carla.Vector3D(0.3, 0.3, 0.9) if type_id.startswith("walker")
                          else _carla.Vector3D(2.2, 1.0, 0.75))
                actors.append(pinned.FrozenActor(
                    actor_id=aid, type_id=type_id,
                    bounding_box=_carla.BoundingBox(_carla.Location(0, 0, extent.z), extent),
                    transform=_carla.Transform(_carla.Location(x, y, 0.0),
                                               _carla.Rotation(yaw=yaw)),
                    velocity=_carla.Vector3D(v, 0.0, 0.0)))
            world = pinned.FrozenWorld(pinned.FrozenActorList(actors))
            radar = {"world_xyz": rng.normal(0.0, 40.0, (3000, 3))}
            kwargs = dict(frame_id=frame, timestamp=frame / 10.0, camera_matrix=matrix,
                          camera_inverse=inverse, radar_points=radar, world=world,
                          camera_location=cam.location)
            ref = pinned.PassiveSplitCollector._ground_truth(gt_host(parked, t_ref), **kwargs)
            new = pinned.PassiveSplitCollector._ground_truth(host.gt, **kwargs)
            self.assertTrue(ref)
            self.assertEqual(exact(ref), exact(new))
        self.assertEqual(tracker_state(t_ref), tracker_state(host.gt.actor_tracker))



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
