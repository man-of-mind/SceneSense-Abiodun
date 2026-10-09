"""Bounded CARLA-only sensor-header diagnostic with optional retained launcher.

The same RGB/semantic/radar spawn method and live-AVO depth/callback mixin are
used without any model, radio, policy, GT evaluation or transport runtime.
Control advances owned ticks without waiting for camera callbacks. Treatment
waits for all four callbacks before the next tick; no timestamp is rewritten.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

SCHEMA = "splitfusion.run5.carla_sensor_sync_probe.v1"
TOKEN = "RUN5_CARLA_SENSOR_SYNC_PROBE_V1_EXECUTE"
ROLES = ("rgb", "semantic", "depth", "radar")


def _raw_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def sensor_header(value: Any) -> dict:
    timestamp = float(value.timestamp)
    return {"frame_id": int(value.frame), "timestamp": timestamp,
            "timestamp_hex": timestamp.hex(), "receipt_raw_ns": _raw_ns(),
            "matrix": np.asarray(value.transform.get_matrix(), dtype=np.float64).tolist(),
            "width": getattr(value, "width", None), "height": getattr(value, "height", None)}


def compare_headers(snapshot: dict, sources: dict[str, dict]) -> dict:
    missing = [role for role in ROLES if role not in sources]
    result = {"frame_id": snapshot["frame_id"], "snapshot": snapshot,
              "sources": sources, "missing_roles": missing, "valid": False}
    if missing:
        result["reason"] = "MISSING_CALLBACK"
        return result
    frame_errors = [role for role in ROLES if sources[role]["frame_id"] != snapshot["frame_id"]]
    deltas = {role: sources[role]["timestamp"] - snapshot["timestamp"] for role in ROLES}
    result["timestamp_delta_to_world_s"] = deltas
    result["camera_pose_max_abs_delta"] = {
        role: float(np.max(np.abs(np.asarray(sources[role]["matrix"]) - np.asarray(sources["rgb"]["matrix"]))))
        for role in ("semantic", "depth")}
    result["camera_pose_to_owned_tick_max_abs_delta"] = {
        role: float(np.max(np.abs(np.asarray(sources[role]["matrix"]) - np.asarray(snapshot["camera_matrix"]))))
        for role in ("rgb", "semantic", "depth")}
    # Existing pose tolerance is preserved; timestamps remain EXACT, including
    # comparison with owned world snapshot and CPU radar, not cameras alone.
    reasons = []
    if frame_errors:
        reasons.append("FRAME_MISMATCH")
    if any(delta != 0.0 for delta in deltas.values()):
        reasons.append("EXACT_TIMESTAMP_MISMATCH")
    if any(delta > 1e-5 for delta in result["camera_pose_max_abs_delta"].values()):
        reasons.append("CAMERA_POSE_MISMATCH")
    if any(delta > 1e-5 for delta in result["camera_pose_to_owned_tick_max_abs_delta"].values()):
        reasons.append("OWNED_TICK_POSE_MISMATCH")
    result["reason"] = "|".join(reasons) or "EXACTLY_ALIGNED"
    result["valid"] = not reasons
    return result


class HeaderStore:
    def __init__(self):
        self.condition = threading.Condition()
        self.rows: dict[int, dict[str, dict]] = {}
        self.duplicates: list[dict] = []

    def put(self, role: str, image: Any):
        value = sensor_header(image)
        with self.condition:
            bucket = self.rows.setdefault(value["frame_id"], {})
            if role in bucket:
                self.duplicates.append({"role": role, "frame_id": value["frame_id"]})
            bucket[role] = value
            self.condition.notify_all()

    def wait(self, frame_id: int, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        with self.condition:
            while any(role not in self.rows.get(frame_id, {}) for role in ROLES):
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self.condition.wait(left)
            return True

    def get(self, frame_id):
        with self.condition:
            return dict(self.rows.get(frame_id, {}))


def make_collector(world, ego, store, owned_sensors):
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from . import live_gt as G

    class OwnedWorld:
        def spawn_actor(self, *args, **kwargs):
            actor = world.spawn_actor(*args, **kwargs)
            owned_sensors.append(actor)
            return actor

        def __getattr__(self, name):
            return getattr(world, name)

    class Base:
        _spawn_sensors = P.PassiveSplitCollector._spawn_sensors

        def __init__(self, **kwargs):
            self.world, self.ego = kwargs["world"], kwargs["ego"]
            self.sensors = []
            self._spawn_sensors()

        def _on_rgb(self, image):
            store.put("rgb", image)

        def _on_semantic(self, image):
            store.put("semantic", image)

        def _on_radar(self, image):
            store.put("radar", image)

    Live = G.build_live_gt_collector(Base)

    class Probe(Live):
        def _on_depth(self, image):
            store.put("depth", image)
            return super()._on_depth(image)

    return Probe(world=OwnedWorld(), ego=ego)


def _settings(settings):
    return {name: getattr(settings, name) for name in (
        "synchronous_mode", "fixed_delta_seconds", "no_rendering_mode", "substepping",
        "max_substep_delta_time", "max_substeps")}


def run(args):
    if args.execute != TOKEN:
        raise ValueError("Explicit CARLA-only probe authorization token required")
    if not (40 <= args.ticks <= 400 and 0 <= args.warmup_ticks < args.ticks):
        raise ValueError("Probe ticks must be40..400 with bounded warmup")
    if not (0 < args.callback_timeout_s <= 3 and args.wall_bound_s <= 600):
        raise ValueError("Probe bounds invalid")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # Import only after authorization; this module and unit tests need no CARLA.
    import carla
    from . import live_gt as G
    from rl_agent import ue_route_b_split_cell_adapter_v1 as P
    from . import gt_source_sync as GS
    source_files = [Path(__file__), Path(G.__file__), Path(GS.__file__), Path(P.__file__)]
    source_root = Path(__file__).resolve().parents[2]
    source_hashes = {str(p.resolve().relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result = {"schema": SCHEMA, "source_sha256": source_hashes,
              "mode": args.mode, "ticks_per_arm": args.ticks,
              "warmup_ticks_per_arm": args.warmup_ticks, "world_delta_s": .05,
              "sensor_tick_s": 0.0, "new_policy_decisions": 0,
              "radio_or_models_started": False,
              "hypothesis": "Capture frame retained while async camera header timestamp/pose use later world state",
              "binary_source_equivalence_proven": False, "arms": [], "cleanup": {}}
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.)
    world = client.get_world()
    original = world.get_settings()
    result["settings_before"] = _settings(original)
    result["client_version"] = carla.__version__ if hasattr(carla, "__version__") else "unknown"
    result["server_version"] = client.get_server_version()
    result["map"] = world.get_map().name
    ego = collector = None
    owned_sensors = []
    began = time.monotonic()
    failed = None
    try:
        if "Town10HD_Opt" not in result["map"]:
            raise RuntimeError("Root must start Town10HD_Opt before the bounded probe")
        existing = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("sensor.*"))
        if existing:
            raise RuntimeError("Probe refuses pre-existing vehicles or sensors")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = .05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        vehicles = sorted(library.filter("vehicle.*"), key=lambda bp: bp.id)
        bp = next((bp for bp in vehicles if bp.id == "vehicle.lincoln.mkz_2020"), vehicles[0])
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "run5_sensor_sync_probe")
        start = world.get_map().get_spawn_points()[0]
        ego = world.spawn_actor(bp, start)
        ego.set_simulate_physics(False)
        store = HeaderStore()
        collector = make_collector(world, ego, store, owned_sensors)
        result["sensor_ids"] = {name: getattr(collector, name).id for name in ("camera", "semantic_camera", "depth_camera", "radar")}
        result["sensor_attributes"] = {str(s.id): dict(s.attributes) for s in collector.sensors}
        modes = ("control", "barrier") if args.mode == "both" else (args.mode,)
        for mode in modes:
            snapshots, waits, timeouts, barrier_errors = [], [], [], []
            waypoint = world.get_map().get_waypoint(start.location)
            last_frame = None
            next_tick = time.monotonic()
            for index in range(args.ticks):
                if time.monotonic() - began > args.wall_bound_s:
                    raise TimeoutError("Probe whole-run wall bound reached")
                nexts = waypoint.next(.25)
                if nexts:
                    waypoint = sorted(nexts, key=lambda wp: (wp.road_id, wp.lane_id, wp.s))[0]
                transform = waypoint.transform
                transform.location.z += .35
                ego.set_transform(transform)
                tick_start = _raw_ns()
                frame = int(world.tick(10.))
                tick_end = _raw_ns()
                snapshot = world.get_snapshot()
                if int(snapshot.frame) != frame or (last_frame is not None and frame != last_frame + 1):
                    raise RuntimeError("World tick ownership drift")
                last_frame = frame
                camera_actor = snapshot.find(int(collector.camera.id))
                if camera_actor is None:
                    raise RuntimeError("Owned snapshot lacks RGB camera")
                snap = {"frame_id": frame, "timestamp": float(snapshot.timestamp.elapsed_seconds),
                        "timestamp_hex": float(snapshot.timestamp.elapsed_seconds).hex(),
                        "camera_matrix": np.asarray(camera_actor.get_transform().get_matrix()).tolist(),
                        "tick_start_raw_ns": tick_start, "tick_end_raw_ns": tick_end,
                        "arm": mode, "arm_tick": index + 1, "warmup": index < args.warmup_ticks,
                        "selected_10hz": index % 2 == 0}
                snapshots.append(snap)
                if mode == "barrier":
                    before = _raw_ns()
                    try:
                        proof = collector.wait_for_sensor_frame(
                            frame, snapshot=snapshot, timeout_s=args.callback_timeout_s)
                        ok = True
                        snap["shared_barrier_proof"] = proof
                    except Exception as exc:
                        ok = False
                        barrier_errors.append({"frame_id": frame,
                            "status": getattr(exc, "status", type(exc).__name__),
                            "detail": str(exc), "source_record": getattr(exc, "record", None),
                            "warmup": index < args.warmup_ticks})
                    waits.append((_raw_ns() - before) / 1e6)
                    if not ok:
                        timeouts.append(frame)
                        if index >= args.warmup_ticks:
                            break  # Diagnostic failure, never silently advance.
                next_tick += .05
                # No catch-up burst after a long tick or callback barrier.
                next_tick = max(next_tick, time.monotonic())
                time.sleep(max(0., next_tick - time.monotonic()))
            # Allow already-enqueued callbacks to finish without advancing time.
            if snapshots:
                store.wait(snapshots[-1]["frame_id"], args.callback_timeout_s)
            aligned = [compare_headers(snap, store.get(snap["frame_id"])) for snap in snapshots]
            scored = [row for row in aligned if not row["snapshot"]["warmup"]]
            arm = {"mode": mode, "frames": len(aligned), "scored_frames": len(scored),
                   "valid_scored_frames": sum(row["valid"] for row in scored),
                   "invalid_scored_frames": sum(not row["valid"] for row in scored),
                   "barrier_timeout_frames": timeouts,
                   "barrier_errors": barrier_errors,
                   "barrier_wait_ms": waits, "rows": aligned}
            result["arms"].append(arm)
            (output / f"{mode}_headers.json").write_text(json.dumps(arm, indent=2, allow_nan=False) + "\n")
        result["duplicate_callbacks"] = store.duplicates
    except BaseException as exc:
        failed = exc
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        errors = []
        owned = owned_sensors
        for sensor in owned:
            try:
                sensor.stop()
                if not sensor.destroy():
                    errors.append(f"sensor_destroy_false:{sensor.id}")
            except Exception as exc:
                errors.append(f"sensor:{sensor.id}:{type(exc).__name__}:{exc}")
        if ego is not None:
            try:
                if not ego.destroy():
                    errors.append(f"ego_destroy_false:{ego.id}")
            except Exception as exc:
                errors.append(f"ego:{type(exc).__name__}:{exc}")
        try:
            world.apply_settings(original)
            result["settings_after"] = _settings(world.get_settings())
            if result["settings_after"] != result["settings_before"]:
                errors.append("world_settings_not_restored")
        except Exception as exc:
            errors.append(f"settings:{type(exc).__name__}:{exc}")
        result["cleanup"] = {"owned_sensors": [s.id for s in owned], "errors": errors, "ok": not errors}
        result["settings_restored"] = result.get("settings_after") == result["settings_before"]
        try:
            alive_ids = {int(actor.id) for actor in world.get_actors()}
            result["owned_actors_destroyed"] = not any(int(s.id) in alive_ids for s in owned) and (ego is None or int(ego.id) not in alive_ids)
        except Exception as exc:
            result["owned_actors_destroyed"] = False
            errors.append(f"owned_actor_verify:{type(exc).__name__}:{exc}")
        if not result["owned_actors_destroyed"]:
            errors.append("owned_actor_still_present")
            result["cleanup"]["ok"] = False
        result["elapsed_wall_s"] = time.monotonic() - began
        result["verdict"] = "PROBE_ERROR" if failed or errors else "DIAGNOSTIC_COMPLETE_NOT_LIVE_QUALIFICATION"
        result["interpretation"] = "Barrier changes wall tick pacing; preserves simulated0.05s steps and alternate-tick10Hzselection. Light CARLA-only load cannot qualify loaded policy behavior."
        barrier = next((arm for arm in result["arms"] if arm["mode"] == "barrier"), None)
        result["complete_ticks"] = sum(arm["frames"] for arm in result["arms"])
        result["checks"] = {
            "barrier_arm_complete": barrier is not None and barrier["frames"] == args.ticks,
            "barrier_scored_exact": barrier is not None and barrier["scored_frames"] > 0 and barrier["invalid_scored_frames"] == 0,
            "no_scored_barrier_timeout": barrier is not None and not any(row["snapshot"]["frame_id"] in barrier["barrier_timeout_frames"] for row in barrier["rows"] if not row["snapshot"]["warmup"]),
            "settings_restored": result["settings_restored"],
            "owned_actors_destroyed": result["owned_actors_destroyed"],
            "no_probe_error": failed is None and not errors,
            "no_duplicate_callbacks": not result.get("duplicate_callbacks", [])}
        result["passed"] = all(result["checks"].values())
        result["gate_scope"] = "CARLA_ONLY_SOURCE_ALIGNMENT_NOT_LOADED_RUNTIME_QUALIFICATION"
        if getattr(args, "parent_server_launcher", False):
            result["parent_launcher"] = "../LAUNCH_RESULT.json"
        assert {str(p.resolve().relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files} == source_hashes
        (output / "SYNC_PROBE_RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
        (output / "OUTPUT_SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    if failed:
        raise RuntimeError(result["error"]) from failed
    return 0 if result["cleanup"]["ok"] else 2


def verify_probe(path, *, source_root=None):
    """Recompute strict barrier checks from retained per-frame headers, read-only."""
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    if document["schema"] != SCHEMA or document.get("passed") is not True:
        raise ValueError("CARLA source-alignment diagnostic did not pass")
    required_checks = {"barrier_arm_complete", "barrier_scored_exact", "no_scored_barrier_timeout",
                       "settings_restored", "owned_actors_destroyed", "no_probe_error", "no_duplicate_callbacks"}
    if set(document.get("checks", {})) != required_checks or not all(value is True for value in document["checks"].values()):
        raise ValueError("Registered probe checks do not all pass")
    expected_sources = {"rl_agent/splitfusion_run5_live_avo_v1/sync_probe.py",
        "rl_agent/splitfusion_run5_live_avo_v1/live_gt.py",
        "rl_agent/splitfusion_run5_live_avo_v1/gt_source_sync.py",
        "rl_agent/ue_route_b_split_cell_adapter_v1.py"}
    if set(document.get("source_sha256", {})) != expected_sources:
        raise ValueError("Probe source binding is incomplete")
    source_root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    for relative, digest in document["source_sha256"].items():
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root) or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source hash drift: {relative}")
    manifest = json.loads((path.parent / "OUTPUT_SHA256.json").read_text())
    if path.name not in manifest:
        raise ValueError("Probe result not hash-bound")
    for name, digest in manifest.items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent) or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Probe evidence hash drift: {name}")
    barrier = next(arm for arm in document["arms"] if arm["mode"] == "barrier")
    if json.loads((path.parent / "barrier_headers.json").read_text()) != barrier:
        raise ValueError("Barrier raw evidence differs from report")
    rows = barrier["rows"]
    scored = [row for row in rows if not row["snapshot"]["warmup"]]
    if len(rows) != document["ticks_per_arm"] or not scored:
        raise ValueError("Incomplete diagnostic barrier arm")
    frame_ids = [row["snapshot"]["frame_id"] for row in rows]
    if any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("Probe rows repeat or skip owned world frames")
    for index, row in enumerate(rows):
        snapshot = row["snapshot"]
        if (snapshot["arm"] != "barrier" or snapshot["arm_tick"] != index + 1
                or snapshot["warmup"] is not (index < document["warmup_ticks_per_arm"])):
            raise ValueError("Probe warmup/scored accounting drift")
    if len(scored) != document["ticks_per_arm"] - document["warmup_ticks_per_arm"]:
        raise ValueError("Probe scored-row count drift")
    if document.get("duplicate_callbacks") or document.get("error"):
        raise ValueError("Probe callback or process fault")
    if any(row["snapshot"]["frame_id"] in barrier.get("barrier_timeout_frames", []) for row in scored):
        raise ValueError("A scored barrier timed out")
    for row in rows:
        replay = compare_headers(row["snapshot"], row["sources"])
        if replay != row or (not row["snapshot"]["warmup"] and not replay["valid"]):
            raise ValueError("Header replay failed strict source alignment")
    if not document["settings_restored"] or not document["owned_actors_destroyed"] or not document["cleanup"]["ok"]:
        raise ValueError("Probe did not restore owned resources")
    parent_sha = None
    if document.get("parent_launcher"):
        if document["parent_launcher"] != "../LAUNCH_RESULT.json":
            raise ValueError("Unexpected parent launcher path")
        parent_path = path.parent / document["parent_launcher"]
        launch = json.loads(parent_path.read_text())
        parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
        if launch.get("carla_stopped") is not True or launch.get("rpc_closed_after") is not True or launch.get("error"):
            raise ValueError("Probe server lifecycle did not restore cold state")
    return {"passed": True, "report": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "complete_ticks": len(rows), "scored_exact_frames": len(scored), "source_sha256": document["source_sha256"],
            "parent_launcher_sha256": parent_sha}


def launch(args):
    """Use retained server lifecycle; durable log and unconditional stop."""
    if args.execute != TOKEN:
        raise ValueError("Explicit CARLA-only probe authorization token required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Launch with CUDA_VISIBLE_DEVICES=0, never empty")
    import socket
    from rl_agent import ue_288_campaign_supervisor as S
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for port in (args.port, args.port + 1, args.port + 2):
        with socket.socket() as sock:
            sock.settimeout(.2)
            if sock.connect_ex((args.host, port)) == 0:
                raise RuntimeError(f"Refuse existing CARLA endpoint {port}")
    config = json.loads(R.DEFAULT_CONFIG.read_text())
    lifecycle = S.import_lifecycle_helper(config)
    record = {"schema": SCHEMA, "cuda_visible_devices": "0", "carla_stopped": False,
              "rpc_closed_after": False, "error": None}
    process = pgid = None
    rc = 2
    try:
        process, pgid = lifecycle.start_carla(args.port, output / "carla_server.log")
        if lifecycle.wait_for_rpc(args.port, 180.) is None:
            raise RuntimeError("CARLA RPC startup failed")
        nested = SimpleNamespace(**vars(args))
        nested.output_dir = str(output / "sensor_probe")
        nested.parent_server_launcher = True
        nested.launch_server = False
        rc = run(nested)
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and pgid is not None:
            try:
                stopped = lifecycle.stop_carla(process, pgid, args.port)
                record["carla_teardown"] = stopped
                record["carla_stopped"] = isinstance(stopped, dict) and stopped.get("shutdown_verified") is True
            except BaseException as exc:
                record["stop_error"] = f"{type(exc).__name__}: {exc}"
        with socket.socket() as sock:
            sock.settimeout(.2)
            record["rpc_closed_after"] = sock.connect_ex((args.host, args.port)) != 0
        record["probe_returncode"] = rc
        (output / "LAUNCH_RESULT.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return rc if record["carla_stopped"] and record["rpc_closed_after"] and not record["error"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=200)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--mode", choices=("control", "barrier", "both"), default="both")
    parser.add_argument("--callback-timeout-s", type=float, default=2.)
    parser.add_argument("--wall-bound-s", type=float, default=180.)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--launch-server", action="store_true")
    args = parser.parse_args(argv)
    return launch(args) if args.launch_server else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
