#!/usr/bin/env python3
"""Phase-1 radar smoke for the Route B perception dataset v2 timing contract.

Spawns the exact v2 sensor rig (1280x720 RGB / semantic / depth at
``sensor_tick = 0.1`` plus a free-running 200,000 PPS radar) on a moving ego and
drives the shared :mod:`radar_sweep_aggregator_v1` cadence for a short window.
Nothing is persisted to the dataset layout; the script only measures and gates
the timing contract:

  20 Hz world tick -> timestamp-binned 100 Hz.. 100 ms logical sweeps ->
  10 Hz prepared model inputs over a contiguous 200 ms radar window ->
  5 Hz persisted frames.

It deliberately does **not** raise ``points_per_second``: the doubled support
comes from accumulating the measured 200,000 PPS stream over two logical
sweeps.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from data_collection.radar_sweep_aggregator_v1 import (  # noqa: E402
    PREPARE_EVERY_N_TICKS,
    RADAR_POINTS_PER_SECOND,
    SAVE_EVERY_N_PREPARED,
    SWEEP_PERIOD_S,
    WINDOW_SWEEPS,
    WORLD_DELTA_S,
    CadencePlan,
    RadarSweepAggregator,
    RadarSweepError,
    cadence_stats,
    summarize,
)
from data_collection.render_provenance_v1 import (  # noqa: E402
    RenderProvenanceError,
    assert_epic_rendering,
    check_frames,
    render_provenance,
)

SMOKE_VERSION = "route_b_radar_cadence_smoke_v1"
CAMERA_NAMES = ("rgb", "semantic", "depth")
GATE_HZ_TOLERANCE = 0.02


class SmokeError(RuntimeError):
    """A smoke precondition failed before any gate could be evaluated."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--seconds", type=float, default=45.0,
                        help="simulated seconds of steady-state measurement (30-60 s)")
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--target-speed-kph", type=float, default=25.0)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--reload-world", action="store_true",
                        help="reload the map instead of reusing the running world")
    parser.add_argument("--rasterizer", choices=("fast", "legacy"), default="legacy",
                        help="radar rasterizer under test in this smoke")
    parser.add_argument("--report-json", type=Path,
                        default=HERE / "route_b_perception_v2" / "radar_cadence_smoke_v1.json")
    return parser


def peek_frames(sensor_queue: "queue.Queue[Any]") -> list[int]:
    with sensor_queue.mutex:
        return [int(getattr(item, "frame", -1)) for item in sensor_queue.queue]


def drain_until(sensor_queue: "queue.Queue[Any]", frame: int, timeout_s: float) -> Any:
    """Return the record with ``frame`` exactly, discarding strictly older ones."""
    deadline = time.time() + float(timeout_s)
    while True:
        remaining = deadline - time.time()
        if remaining <= 0.0:
            return None
        try:
            item = sensor_queue.get(timeout=remaining)
        except queue.Empty:
            return None
        observed = int(getattr(item, "frame", -1))
        if observed < frame:
            continue
        return item if observed == frame else None


class SmokeRig:
    def __init__(self, args: argparse.Namespace) -> None:
        import carla

        import carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai as fusion_runtime

        self.carla = carla
        self.fusion_runtime = fusion_runtime
        self.args = args
        self.actors: list[Any] = []
        self.sensors: dict[str, Any] = {}
        self.queues: dict[str, "queue.Queue[Any]"] = {}
        self.original_settings = None
        self.original_tm_sync: bool | None = None

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        world = self.client.get_world()
        current_map = str(world.get_map().name)
        if args.reload_world or args.map not in current_map:
            world = self.client.load_world(args.map)
        self.world = world
        self.map_name = str(self.world.get_map().name)

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = WORLD_DELTA_S
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.traffic_manager = self.client.get_trafficmanager(args.tm_port)
        self.original_tm_sync = True
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.seed))

        self.sensor_config = SimpleNamespace(
            camera_width=1280,
            camera_height=720,
            camera_fov=120.0,
            model_input_width=768,
            model_input_height=432,
            ego_camera_x=fusion_runtime.DEFAULT_EGO_CAMERA_X,
            ego_camera_y=fusion_runtime.DEFAULT_EGO_CAMERA_Y,
            ego_camera_z=fusion_runtime.DEFAULT_EGO_CAMERA_Z,
            ego_camera_pitch=fusion_runtime.DEFAULT_EGO_CAMERA_PITCH,
            ego_camera_yaw=fusion_runtime.DEFAULT_EGO_CAMERA_YAW,
            ego_camera_roll=fusion_runtime.DEFAULT_EGO_CAMERA_ROLL,
            ego_radar_x=fusion_runtime.DEFAULT_EGO_RADAR_X,
            ego_radar_y=fusion_runtime.DEFAULT_EGO_RADAR_Y,
            ego_radar_z=fusion_runtime.DEFAULT_EGO_RADAR_Z,
            ego_radar_pitch=fusion_runtime.DEFAULT_EGO_RADAR_PITCH,
            ego_radar_yaw=fusion_runtime.DEFAULT_EGO_RADAR_YAW,
            ego_radar_roll=fusion_runtime.DEFAULT_EGO_RADAR_ROLL,
            radar_range=120.0,
            radar_hfov=120.0,
            radar_vfov=30.0,
            radar_points_per_second=RADAR_POINTS_PER_SECOND,
        )
        self.configured_radar_attributes: dict[str, str] = {}
        # Renderer preflight gate: refuse to measure anything unless the running
        # server was launched at explicit Epic quality with rendering enabled.
        self.render_provenance = render_provenance(
            self.world,
            self.client,
            port=int(args.port),
            camera_width=self.sensor_config.camera_width,
            camera_height=self.sensor_config.camera_height,
            camera_fov=self.sensor_config.camera_fov,
        )
        assert_epic_rendering(self.render_provenance)

        self.ego = self._spawn_ego()
        self._spawn_sensors()

    # -- setup -----------------------------------------------------------
    def _spawn_ego(self) -> Any:
        import random

        blueprints = self.world.get_blueprint_library()
        # Same ego blueprint as the accepted Route B ego loop.
        ego_bp = blueprints.find("vehicle.lincoln.mkz")
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")
        spawn_points = self.world.get_map().get_spawn_points()
        rng = random.Random(int(self.args.seed))
        rng.shuffle(spawn_points)
        for spawn_point in spawn_points:
            ego = self.world.try_spawn_actor(ego_bp, spawn_point)
            if ego is not None:
                self.actors.append(ego)
                ego.set_autopilot(True, int(self.args.tm_port))
                free_flow_kph = 30.0
                percentage = 100.0 * (1.0 - float(self.args.target_speed_kph) / free_flow_kph)
                self.traffic_manager.vehicle_percentage_speed_difference(ego, percentage)
                return ego
        raise SmokeError("could not spawn the smoke ego vehicle")

    def _spawn_sensors(self) -> None:
        blueprints = self.world.get_blueprint_library()
        camera_transform = self.fusion_runtime._ego_camera_transform(self.sensor_config)
        for name, blueprint_id in (
            ("rgb", "sensor.camera.rgb"),
            ("semantic", "sensor.camera.semantic_segmentation"),
            ("depth", "sensor.camera.depth"),
        ):
            bp = blueprints.find(blueprint_id)
            bp.set_attribute("image_size_x", str(self.sensor_config.camera_width))
            bp.set_attribute("image_size_y", str(self.sensor_config.camera_height))
            bp.set_attribute("fov", str(self.sensor_config.camera_fov))
            # Free-running like the radar. A commanded sensor_tick of 0.1 s was
            # measured to skip roughly one capture per 200 world ticks, which
            # permanently shifts the 10 Hz phase; the cadence is derived from
            # world ticks instead of trusted from sensor_tick.
            bp.set_attribute("sensor_tick", "0.0")
            self.queues[name] = queue.Queue()
            sensor = self.world.spawn_actor(bp, camera_transform, attach_to=self.ego)
            sensor.listen(lambda item, key=name: self.queues[key].put(item))
            self.sensors[name] = sensor
            self.actors.append(sensor)

        radar_bp = blueprints.find("sensor.other.radar")
        radar_bp.set_attribute("range", str(self.sensor_config.radar_range))
        radar_bp.set_attribute("horizontal_fov", str(self.sensor_config.radar_hfov))
        radar_bp.set_attribute("vertical_fov", str(self.sensor_config.radar_vfov))
        radar_bp.set_attribute("points_per_second", str(self.sensor_config.radar_points_per_second))
        # Free-running: one raw callback per world tick. The logical 100 ms sweep
        # is rebuilt from timestamps, not commanded through sensor_tick.
        radar_bp.set_attribute("sensor_tick", "0.0")
        self.queues["radar"] = queue.Queue()
        radar = self.world.spawn_actor(
            radar_bp, self.fusion_runtime._ego_radar_transform(self.sensor_config), attach_to=self.ego
        )
        radar.listen(lambda item: self.queues["radar"].put(item))
        self.sensors["radar"] = radar
        self.actors.append(radar)
        self.configured_radar_attributes = {
            key: str(radar.attributes.get(key))
            for key in ("points_per_second", "sensor_tick", "range", "horizontal_fov", "vertical_fov")
        }

    # -- teardown --------------------------------------------------------
    def close(self) -> dict[str, Any]:
        report: dict[str, Any] = {"destroyed": [], "surviving": [], "settings_restored": False}
        for name, sensor in self.sensors.items():
            try:
                sensor.stop()
            except RuntimeError:
                pass
        for actor in reversed(self.actors):
            actor_id = int(actor.id)
            try:
                actor.destroy()
            except RuntimeError:
                pass
            report["destroyed"].append(actor_id)
        try:
            self.world.tick()
        except RuntimeError:
            pass
        for actor_id in report["destroyed"]:
            try:
                actor = self.world.get_actor(actor_id)
            except RuntimeError:
                continue
            if actor is not None and actor.is_alive:
                report["surviving"].append(actor_id)
        try:
            if self.original_tm_sync:
                self.traffic_manager.set_synchronous_mode(False)
            if self.original_settings is not None:
                self.world.apply_settings(self.original_settings)
            report["settings_restored"] = True
        except RuntimeError as exc:
            report["settings_restore_error"] = str(exc)
        return report


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    rig = SmokeRig(args)
    aggregator = RadarSweepAggregator()
    plan = CadencePlan()
    prepared_records: list[dict[str, Any]] = []
    camera_frame_gaps: list[int] = []
    camera_misses = 0
    tensor_failures: list[str] = []
    prep_wall_s: list[float] = []
    roundtrip_max_error_m = 0.0
    last_camera_frame: int | None = None
    warmup_camera_frames: list[int] = []
    frame_content_failures: list[dict[str, Any]] = []
    frame_checks_run = 0

    try:
        import carla_collect_parked_ego_fusion_training_data as parked

        intrinsics_input = rig.fusion_runtime.intrinsics_at(
            rig.sensor_config.model_input_width,
            rig.sensor_config.model_input_height,
            rig.sensor_config.camera_fov,
        )
        tracker = parked.StationaryTrackAccumulator(
            stationary_velocity_mps=0.35,
            parked_threshold_s=5.0,
            association_grid_m=1.5,
            max_stale_s=2.0,
        )

        total_ticks = int(args.warmup_ticks) + int(round(float(args.seconds) / WORLD_DELTA_S))
        settings_snapshot = rig.world.get_settings()
        for tick_number in range(total_ticks):
            frame = int(rig.world.tick())
            snapshot_time = float(rig.world.get_snapshot().timestamp.elapsed_seconds)
            measurement = drain_until(rig.queues["radar"], frame, 10.0)
            if measurement is None:
                raise SmokeError(f"no radar callback for world frame {frame}")
            aggregator.ingest(measurement)
            plan.note_tick(float(measurement.timestamp))

            if tick_number < int(args.warmup_ticks):
                # Observe the camera cadence phase and discard warmup records.
                for name in CAMERA_NAMES:
                    while True:
                        try:
                            item = rig.queues[name].get_nowait()
                        except queue.Empty:
                            break
                        if name == "rgb":
                            warmup_camera_frames.append(int(item.frame))
                continue

            if plan.camera_frame_parity is None:
                if len(warmup_camera_frames) < 2:
                    raise SmokeError(
                        "camera cadence phase was not observed during warmup "
                        f"(rgb frames seen: {warmup_camera_frames})"
                    )
                gaps = {
                    warmup_camera_frames[i] - warmup_camera_frames[i - 1]
                    for i in range(1, len(warmup_camera_frames))
                }
                if gaps != {1}:
                    raise SmokeError(
                        f"free-running camera stream is not one capture per world tick: gaps={sorted(gaps)}"
                    )
                plan.lock_camera_parity(warmup_camera_frames[-1])
            if not plan.is_prepare_frame(frame):
                continue

            images: dict[str, Any] = {}
            for name in CAMERA_NAMES:
                item = drain_until(rig.queues[name], frame, 5.0)
                if item is None:
                    camera_misses += 1
                    raise SmokeError(f"missing {name} camera record at prepared frame {frame}")
                images[name] = item
            if last_camera_frame is not None:
                camera_frame_gaps.append(frame - last_camera_frame)
            last_camera_frame = frame

            frame_check = check_frames(
                images["rgb"], images["semantic"],
                width=rig.sensor_config.camera_width,
                height=rig.sensor_config.camera_height,
            )
            if not frame_check["ok"]:
                frame_content_failures.append({"world_frame": frame, "problems": frame_check["problems"]})
            frame_checks_run += 1

            if aggregator.anchor_s is None:
                # Bin boundaries close exactly on prepared-input ticks.
                aggregator.set_anchor(float(measurement.timestamp))
            current_index = aggregator.sweep_index_for(float(measurement.timestamp))
            if not aggregator.has_window(current_index):
                plan.warmup_prepared_skipped += 1
                continue

            started = time.perf_counter()
            radar_inverse = rig.fusion_runtime.actor_world_inverse_matrix(rig.sensors["radar"])
            radar_matrix = rig.fusion_runtime.actor_world_matrix(rig.sensors["radar"])
            camera_inverse = rig.fusion_runtime.actor_world_inverse_matrix(rig.sensors["rgb"])
            try:
                detections, window_meta = aggregator.window_detections(
                    current_index, sensor_inverse_matrix=radar_inverse
                )
            except RadarSweepError as exc:
                tensor_failures.append(str(exc))
                continue

            tensor, points, radar_summary = parked.build_radar_sample(
                detections=detections,
                sensor_matrix=radar_matrix,
                camera_inverse_matrix=camera_inverse,
                camera_intrinsics=intrinsics_input,
                width=rig.sensor_config.model_input_width,
                height=rig.sensor_config.model_input_height,
                frame_time_s=float(measurement.timestamp),
                tracker=tracker,
                max_range_m=rig.sensor_config.radar_range,
                max_abs_velocity_mps=20.0,
                parked_threshold_s=5.0,
                point_radius_px=4,
                rasterizer=str(args.rasterizer),
            )
            prep_wall_s.append(time.perf_counter() - started)

            expected_shape = (4, rig.sensor_config.model_input_height, rig.sensor_config.model_input_width)
            if tuple(tensor.shape) != expected_shape:
                tensor_failures.append(f"frame {frame}: radar tensor shape {tensor.shape} != {expected_shape}")
            if not np.all(np.isfinite(tensor)):
                tensor_failures.append(f"frame {frame}: non-finite radar tensor values")

            # Motion-compensation round trip: the current sweep's own returns must
            # come back to their originally measured world positions.
            current_sweep = aggregator.sweep_report(current_index)
            if points["world_xyz"].size:
                tail = int(sum(
                    cb.returns for cb in aggregator._sweeps[current_index].callbacks
                ))
                if tail:
                    reference = np.concatenate(
                        [cb.world_velocity[:, :3] for cb in aggregator._sweeps[current_index].callbacks],
                        axis=0,
                    ).astype(np.float64)
                    produced = points["world_xyz"][-tail:].astype(np.float64)
                    if produced.shape == reference.shape and reference.size:
                        roundtrip_max_error_m = max(
                            roundtrip_max_error_m,
                            float(np.max(np.linalg.norm(produced - reference, axis=1))),
                        )

            saved = plan.note_prepared(float(measurement.timestamp))
            prepared_records.append({
                "world_frame": frame,
                "world_timestamp_s": float(measurement.timestamp),
                "snapshot_timestamp_s": snapshot_time,
                "camera_frames": {name: int(images[name].frame) for name in CAMERA_NAMES},
                "camera_timestamp_delta_s": max(
                    abs(float(images[name].timestamp) - float(measurement.timestamp))
                    for name in CAMERA_NAMES
                ),
                "sweep_index": current_index,
                "sweep_callbacks": current_sweep["callbacks"],
                "sweep_returns": current_sweep["returns"],
                "window_returns": window_meta["returns"],
                "window_span_s": window_meta["window_span_s"],
                "window_sweeps": window_meta["sweep_indices"],
                "window_contiguous": window_meta["sweeps_contiguous"],
                "window_callback_frames": window_meta["callback_frames"],
                "radar_points_in_tensor": float(radar_summary["radar_points"]),
                "persisted": bool(saved),
            })
    finally:
        cleanup = rig.close()

    sweeps = [row for row in aggregator.all_sweep_reports() if row.get("present")]
    sweep_by_index = {row["sweep_index"]: row for row in sweeps}
    # Only sweeps a prepared input actually consumed are contractual. The run ends
    # mid-bin, so the trailing partial sweep is reported but never gated.
    consumed_indices = sorted({
        index for row in prepared_records for index in row["window_sweeps"]
    })
    steady_sweeps = [sweep_by_index[index] for index in consumed_indices if index in sweep_by_index]
    unconsumed_sweeps = [row["sweep_index"] for row in sweeps if row["sweep_index"] not in set(consumed_indices)]
    sweep_starts = [row["bin_end_s"] for row in steady_sweeps]
    window_spans = [row["window_span_s"] for row in prepared_records if row["window_span_s"] is not None]
    contiguity_failures = [
        row["world_frame"] for row in prepared_records
        if not row["window_contiguous"] or row["window_sweeps"] != [row["sweep_index"] - 1, row["sweep_index"]]
    ]

    report: dict[str, Any] = {
        "schema": f"{SMOKE_VERSION}.report",
        "map": rig.map_name,
        "render_provenance": rig.render_provenance,
        "frame_content": {
            "checks_run": frame_checks_run,
            "failures": frame_content_failures,
        },
        "configured": {
            "radar_attributes": rig.configured_radar_attributes,
            "configured_points_per_second": int(rig.sensor_config.radar_points_per_second),
            "fixed_delta_seconds": float(settings_snapshot.fixed_delta_seconds),
            "synchronous_mode": bool(settings_snapshot.synchronous_mode),
            "camera_sensor_tick_s": 0.0,
            "sweep_period_s": SWEEP_PERIOD_S,
            "window_sweeps": WINDOW_SWEEPS,
            "prepare_every_n_ticks": PREPARE_EVERY_N_TICKS,
            "save_every_n_prepared": SAVE_EVERY_N_PREPARED,
            "target_speed_kph": float(args.target_speed_kph),
            "rasterizer": str(args.rasterizer),
        },
        "world_tick": cadence_stats(plan.tick_timestamps_s),
        "raw_callbacks": {
            "count": aggregator.raw_callbacks,
            "cadence": {
                "hz": (1.0 / (sum(aggregator.callback_intervals_s) / len(aggregator.callback_intervals_s)))
                if aggregator.callback_intervals_s else None,
                "interval_s": summarize(aggregator.callback_intervals_s),
            },
            "returns_per_callback": summarize(aggregator.returns_per_callback),
            "total_returns": aggregator.total_returns,
            "duplicates": aggregator.duplicate_callbacks,
            "out_of_order": aggregator.out_of_order_callbacks,
            "dropped_frames": aggregator.dropped_callback_frames,
            "timestamp_reversals": aggregator.timestamp_reversals,
        },
        "logical_sweeps": {
            "count": len(steady_sweeps),
            "cadence": cadence_stats(sweep_starts),
            "expected_callbacks_per_sweep": aggregator.expected_callbacks_per_sweep,
            "consumed_sweep_index_range": [consumed_indices[0], consumed_indices[-1]] if consumed_indices else None,
            "unconsumed_partial_sweeps": unconsumed_sweeps,
            "callbacks_per_sweep": summarize([row["callbacks"] for row in steady_sweeps]),
            "returns_per_sweep": summarize([row["returns"] for row in steady_sweeps]),
        },
        "temporal_window": {
            "span_s": summarize(window_spans),
            "returns": summarize([row["window_returns"] for row in prepared_records]),
            "expected_support_s": SWEEP_PERIOD_S * WINDOW_SWEEPS,
            "expected_window_callbacks": aggregator.expected_window_callbacks,
            "callbacks_per_window": summarize([len(row["window_callback_frames"]) for row in prepared_records]),
            "non_consecutive_prepared_frames": contiguity_failures,
            "motion_compensation_roundtrip_max_error_m": roundtrip_max_error_m,
        },
        "prepared_inputs": cadence_stats(plan.prepared_timestamps_s),
        "saved_frames": cadence_stats(plan.saved_timestamps_s),
        "warmup_prepared_skipped": plan.warmup_prepared_skipped,
        "alignment": {
            "camera_frame_parity": plan.camera_frame_parity,
            "warmup_rgb_frames": warmup_camera_frames,
            "camera_frame_gap": summarize(camera_frame_gaps),
            "camera_misses": camera_misses,
            "max_camera_radar_timestamp_delta_s": max(
                (row["camera_timestamp_delta_s"] for row in prepared_records), default=None
            ),
            "max_snapshot_radar_timestamp_delta_s": max(
                (abs(row["snapshot_timestamp_s"] - row["world_timestamp_s"]) for row in prepared_records),
                default=None,
            ),
        },
        "prepare_wall_clock_s": summarize(prep_wall_s),
        "tensor_failures": tensor_failures,
        "cleanup": cleanup,
    }

    def hz_ok(observed: float | None, target: float) -> bool:
        return observed is not None and abs(observed - target) <= GATE_HZ_TOLERANCE * target

    gates = {
        "world_tick_20hz": hz_ok(report["world_tick"]["hz"], 20.0),
        "logical_sweeps_10hz": hz_ok(report["logical_sweeps"]["cadence"]["hz"], 10.0),
        "prepared_inputs_10hz": hz_ok(report["prepared_inputs"]["hz"], 10.0),
        "saved_frames_5hz": hz_ok(report["saved_frames"]["hz"], 5.0),
        "consecutive_logical_sweeps": not contiguity_failures,
        "window_span_within_200ms": bool(window_spans) and max(window_spans) <= SWEEP_PERIOD_S * WINDOW_SWEEPS + 1e-6,
        "no_dropped_callbacks": aggregator.dropped_callback_frames == 0,
        "no_duplicate_callbacks": aggregator.duplicate_callbacks == 0,
        "no_out_of_order_callbacks": aggregator.out_of_order_callbacks == 0,
        "no_timestamp_reversals": (
            aggregator.timestamp_reversals == 0
            and report["world_tick"]["reversals"] == 0
            and report["prepared_inputs"]["reversals"] == 0
            and report["saved_frames"]["reversals"] == 0
        ),
        "no_tensor_failures": not tensor_failures,
        "no_camera_misses": camera_misses == 0,
        "callbacks_per_sweep_exact": bool(steady_sweeps) and all(
            row["callbacks"] == aggregator.expected_callbacks_per_sweep for row in steady_sweeps
        ),
        "window_callbacks_exact": bool(prepared_records) and all(
            len(row["window_callback_frames"]) == aggregator.expected_window_callbacks
            for row in prepared_records
        ),
        "rgb_and_segmentation_frames_valid": frame_checks_run > 0 and not frame_content_failures,
        "epic_quality_confirmed": (
            str(rig.render_provenance["launch"].get("quality_level", "")).lower() == "epic"
            and not rig.render_provenance["no_rendering_mode"]
            and not rig.render_provenance["launch"]["render_disabling_flags"]
        ),
        "sensor_frames_aligned": all(
            set(row["camera_frames"].values()) == {row["world_frame"]} for row in prepared_records
        ),
        "pps_not_inflated": int(float(rig.configured_radar_attributes["points_per_second"])) == RADAR_POINTS_PER_SECOND,
        "cleanup_clean": not cleanup["surviving"] and cleanup["settings_restored"],
    }
    report["gates"] = gates
    report["status"] = "RADAR_SMOKE_PASSED" if all(gates.values()) else "RADAR_SMOKE_FAILED"
    report["prepared_records_head"] = prepared_records[:8]
    report["prepared_records_tail"] = prepared_records[-4:]
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 20.0 <= float(args.seconds) <= 120.0:
        print("--seconds must stay inside the short smoke range", file=sys.stderr)
        return 2
    try:
        report = run_smoke(args)
    except (SmokeError, RadarSweepError, RenderProvenanceError, RuntimeError) as exc:
        print(json.dumps({"status": "RADAR_SMOKE_FAILED", "error": str(exc)}, indent=2), flush=True)
        return 2
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    printable = {key: value for key, value in report.items()
                 if key not in ("prepared_records_head", "prepared_records_tail")}
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    print(f"report written to {report_path}", flush=True)
    return 0 if report["status"] == "RADAR_SMOKE_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
