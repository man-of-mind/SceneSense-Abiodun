#!/usr/bin/env python3
"""Collect one qualified Route B episode in the historical fusion-data layout.

The accepted density runner remains byte-for-byte unchanged. This adapter
injects a synchronous sensor sampler around its world ticks, supplies a
separate Traffic Manager seed, and writes the existing manifest/object-box
format. It is intentionally bounded to one Route B loop and one density.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_ROUTE = HERE / "routes" / "town10hd_opt_route_b_full_map_loop_v1.json"
DEFAULT_PROGRESS = HERE / "routes" / "town10hd_opt_route_b_full_map_loop_v1.progress.csv"
ACCEPTED_DENSITY_RUNNER = HERE / "run_route_b_density_loop.py"
EXPECTED_ROUTE_SHA256 = "fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5"
EXPECTED_PROGRESS_SHA256 = "974593859368f24ee2bc4ac31b82118bf2e932d0de1c96858b8771e2dd4d90c0"
EXPECTED_RUNNER_SHA256 = "59592ee83184a227f324ff872d1cc7f5601d5a1efb0300dc08dec7b7f26749a4"
DENSITIES = {
    "low": (5, 5),
    "medium": (15, 15),
    "dense": (25, 25),
}


class PilotError(RuntimeError):
    """The bounded collection contract or an episode invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    expected = (
        (args.route_config, EXPECTED_ROUTE_SHA256, "Route B JSON"),
        (args.route_progress_csv, EXPECTED_PROGRESS_SHA256, "Route B progress CSV"),
        (ACCEPTED_DENSITY_RUNNER, EXPECTED_RUNNER_SHA256, "qualified density runner"),
    )
    observed: dict[str, str] = {}
    for path, wanted, label in expected:
        path = Path(path).resolve()
        if not path.is_file():
            raise PilotError(f"{label} is missing: {path}")
        actual = sha256_file(path)
        if actual != wanted:
            raise PilotError(f"{label} hash drift: expected {wanted}, observed {actual}")
        observed[label] = actual

    route = json.loads(Path(args.route_config).read_text(encoding="utf-8"))
    if route.get("schema_version") != 1 or route.get("type") != "carla_ego_route":
        raise PilotError("Route B JSON must be accepted carla_ego_route schema version 1")
    if route.get("name") != "Town10HD_Opt Route B full-map loop v1":
        raise PilotError(f"unexpected route name: {route.get('name')!r}")
    if route.get("map") != "Carla/Maps/Town10HD_Opt" or route.get("loop") is not True:
        raise PilotError("Route B must be a closed Town10HD_Opt loop")
    if len(route.get("intermediate_waypoints", [])) != 18:
        raise PilotError("accepted Route B must contain exactly 18 intermediate waypoints")

    with Path(args.route_progress_csv).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ego_x", "ego_y", "ego_z"]:
            raise PilotError("Route B progress CSV header drift")
        progress_rows = sum(1 for _ in reader)
    if progress_rows != 301:
        raise PilotError(f"Route B progress CSV must contain 301 points, found {progress_rows}")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise PilotError(f"create-only output directory already exists: {output_dir}")
    vehicles, pedestrians = DENSITIES[args.density]
    return {
        "density": args.density,
        "vehicles": vehicles,
        "pedestrians": pedestrians,
        "scenario_seed": int(args.scenario_seed),
        "traffic_manager_seed": int(args.tm_seed),
        "output_dir": str(output_dir),
        "route_name": route["name"],
        "route_progress_points": progress_rows,
        "hashes": observed,
    }


def weather_payload(weather: Any) -> dict[str, float]:
    fields = (
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )
    return {name: float(getattr(weather, name)) for name in fields if hasattr(weather, name)}


class TrafficManagerSeedProxy:
    def __init__(self, target: Any, seed: int) -> None:
        self._target = target
        self._seed = int(seed)

    def set_random_device_seed(self, _ignored: int) -> None:
        self._target.set_random_device_seed(self._seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class ClientProxy:
    def __init__(self, real_client_class: Any, tm_seed: int, *args: Any, **kwargs: Any) -> None:
        self._client = real_client_class(*args, **kwargs)
        self._tm_seed = int(tm_seed)

    def get_trafficmanager(self, port: int) -> TrafficManagerSeedProxy:
        return TrafficManagerSeedProxy(self._client.get_trafficmanager(port), self._tm_seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class SamplingWorld:
    def __init__(self, world: Any, collector: "LegacyPerceptionCollector") -> None:
        self._world = world
        self._collector = collector
        self._route_ticks = 0

    def tick(self, *args: Any, **kwargs: Any) -> int:
        frame_id = int(self._world.tick(*args, **kwargs))
        self._route_ticks += 1
        if self._route_ticks % 10 == 0:
            self._collector.save_frame(frame_id, self._route_ticks)
        return frame_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)


class LegacyPerceptionCollector:
    """Synchronous 2 Hz sampler using the historical dataset helper functions."""

    def __init__(
        self,
        *,
        parked: Any,
        world: Any,
        ego: Any,
        output_dir: Path,
        density: str,
        vehicles: int,
        pedestrians: int,
        scenario_seed: int,
        tm_seed: int,
        route_path: Path,
        progress_path: Path,
    ) -> None:
        import numpy as np

        self.np = np
        self.parked = parked
        self.world = world
        self.ego = ego
        self.output_dir = output_dir.resolve()
        self.density = density
        self.vehicles = int(vehicles)
        self.pedestrians = int(pedestrians)
        self.scenario_seed = int(scenario_seed)
        self.tm_seed = int(tm_seed)
        self.route_path = route_path.resolve()
        self.progress_path = progress_path.resolve()
        self.experiment_id = self.output_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.dirs = parked.prepare_dataset_dirs(self.output_dir)
        self.manifest_path = self.output_dir / "manifest.csv"
        self.object_boxes_path = self.output_dir / "object_boxes.csv"
        self.queues = {name: queue.Queue(maxsize=4) for name in ("rgb", "semantic", "depth", "radar")}
        self.sensors: list[Any] = []
        self.saved = 0
        self.sample_stats: list[dict[str, Any]] = []
        self.max_timestamp_delta_s = 0.0
        self.max_camera_transform_delta_m = 0.0
        self.max_radar_transform_delta_m = 0.0
        self.cleanup_succeeded = False
        self.failure = ""

        fr = parked.fusion_runtime
        self.args = SimpleNamespace(
            camera_width=1280,
            camera_height=720,
            camera_fov=120.0,
            model_input_width=768,
            model_input_height=432,
            ego_camera_x=fr.DEFAULT_EGO_CAMERA_X,
            ego_camera_y=fr.DEFAULT_EGO_CAMERA_Y,
            ego_camera_z=fr.DEFAULT_EGO_CAMERA_Z,
            ego_camera_pitch=fr.DEFAULT_EGO_CAMERA_PITCH,
            ego_camera_yaw=fr.DEFAULT_EGO_CAMERA_YAW,
            ego_camera_roll=fr.DEFAULT_EGO_CAMERA_ROLL,
            ego_radar_x=fr.DEFAULT_EGO_RADAR_X,
            ego_radar_y=fr.DEFAULT_EGO_RADAR_Y,
            ego_radar_z=fr.DEFAULT_EGO_RADAR_Z,
            ego_radar_pitch=fr.DEFAULT_EGO_RADAR_PITCH,
            ego_radar_yaw=fr.DEFAULT_EGO_RADAR_YAW,
            ego_radar_roll=fr.DEFAULT_EGO_RADAR_ROLL,
            radar_range=120.0,
            radar_hfov=120.0,
            radar_vfov=30.0,
            radar_points_per_second=200000,
            radar_max_velocity=20.0,
            radar_raster_radius_px=4,
            radar_temporal_window_frames=1,
            stationary_velocity_mps=0.35,
            parked_threshold_s=5.0,
            association_grid_m=1.5,
            max_stale_s=2.0,
            radar_support_margin_m=1.0,
            radar_person_support_mode="radius",
            radar_person_support_radius_m=1.5,
            radar_person_support_z_down_m=0.5,
            radar_person_support_z_up_m=2.0,
            gt_max_distance_m=140.0,
            include_pedestrians=True,
            jpeg_quality=92,
            npc_vehicles=self.vehicles,
            npc_pedestrians=self.pedestrians,
            ego_spawn_index=0,
        )
        self.tracker = parked.StationaryTrackAccumulator(
            stationary_velocity_mps=self.args.stationary_velocity_mps,
            parked_threshold_s=self.args.parked_threshold_s,
            association_grid_m=self.args.association_grid_m,
            max_stale_s=self.args.max_stale_s,
        )
        self.actor_tracker = parked.ActorStationaryTracker(
            stationary_velocity_mps=self.args.stationary_velocity_mps,
            parked_threshold_s=self.args.parked_threshold_s,
        )
        self.intrinsics_full = fr.intrinsics_at(
            self.args.camera_width, self.args.camera_height, self.args.camera_fov
        )
        self.intrinsics_input = fr.intrinsics_at(
            self.args.model_input_width, self.args.model_input_height, self.args.camera_fov
        )
        self._spawn_sensors()
        self._write_metadata()

    def _spawn_sensors(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        camera_transform = self.parked.fusion_runtime._ego_camera_transform(self.args)
        for queue_name, blueprint_id in (
            ("rgb", "sensor.camera.rgb"),
            ("semantic", "sensor.camera.semantic_segmentation"),
            ("depth", "sensor.camera.depth"),
        ):
            bp = bp_lib.find(blueprint_id)
            bp.set_attribute("image_size_x", str(self.args.camera_width))
            bp.set_attribute("image_size_y", str(self.args.camera_height))
            bp.set_attribute("fov", str(self.args.camera_fov))
            bp.set_attribute("sensor_tick", "0.1")
            sensor = self.world.spawn_actor(bp, camera_transform, attach_to=self.ego)
            sensor.listen(lambda item, name=queue_name: self.parked.od_demo.put_latest(self.queues[name], item))
            self.sensors.append(sensor)
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("range", str(self.args.radar_range))
        radar_bp.set_attribute("horizontal_fov", str(self.args.radar_hfov))
        radar_bp.set_attribute("vertical_fov", str(self.args.radar_vfov))
        radar_bp.set_attribute("points_per_second", str(self.args.radar_points_per_second))
        radar_bp.set_attribute("sensor_tick", "0.1")
        radar = self.world.spawn_actor(
            radar_bp, self.parked.fusion_runtime._ego_radar_transform(self.args), attach_to=self.ego
        )
        radar.listen(lambda item: self.parked.od_demo.put_latest(self.queues["radar"], item))
        self.sensors.append(radar)
        self.camera, self.semantic_camera, self.depth_camera, self.radar = self.sensors

    def _write_metadata(self) -> None:
        metadata = {
            "schema": "scenesense_moving_ego_fusion_training_data.v1",
            "experiment_id": self.experiment_id,
            "description": "Qualified Route B moving-ego perception pilot in the historical fusion layout.",
            "world": str(self.world.get_map().name),
            "scenario_id": f"route_b_{self.density}_seed101_tm1101",
            "view_id": "qualified_route_b_controller",
            "split": "train",
            "density": self.density,
            "requested_vehicles": self.vehicles,
            "requested_pedestrians": self.pedestrians,
            "scenario_seed": self.scenario_seed,
            "traffic_manager_seed": self.tm_seed,
            "simulator_hz": 20,
            "save_every_nth_tick": 10,
            "saved_hz": 2,
            "route_file": str(self.route_path),
            "route_file_sha256": EXPECTED_ROUTE_SHA256,
            "route_progress_csv": str(self.progress_path),
            "route_progress_csv_sha256": EXPECTED_PROGRESS_SHA256,
            "qualified_density_runner_sha256": EXPECTED_RUNNER_SHA256,
            "controller": {
                "lane_offset_m": -0.5,
                "walker_detection_distance_m": 10.0,
                "npc_hardening": True,
                "safe_vehicle_filter": True,
                "interventions": False,
            },
            "weather": weather_payload(self.world.get_weather()),
            "camera_resolution": [self.args.camera_width, self.args.camera_height],
            "model_input_size": [self.args.model_input_width, self.args.model_input_height],
            "sensor_tick_s": 0.1,
            "radar": {
                "points_per_second": self.args.radar_points_per_second,
                "range_m": self.args.radar_range,
                "horizontal_fov": self.args.radar_hfov,
                "vertical_fov": self.args.radar_vfov,
                "temporal_window_frames": self.args.radar_temporal_window_frames,
            },
        }
        self.parked.save_json(self.output_dir / "metadata.json", metadata)

    @staticmethod
    def _transform_delta_m(a: Any, b: Any) -> float:
        return math.sqrt(
            (float(a.location.x) - float(b.location.x)) ** 2
            + (float(a.location.y) - float(b.location.y)) ** 2
            + (float(a.location.z) - float(b.location.z)) ** 2
        )

    def _wait_exact(self, name: str, frame_id: int) -> Any:
        if name == "radar":
            item = self.parked.wait_for_measurement(self.queues[name], frame_id, 5.0)
        else:
            item = self.parked.od_demo.wait_for_camera_frame(self.queues[name], frame_id, 5.0)
        if item is None:
            raise PilotError(f"missing {name} sensor record at frame {frame_id}")
        observed = int(getattr(item, "frame", -1))
        if observed != frame_id:
            raise PilotError(f"{name} frame misalignment: world={frame_id}, sensor={observed}")
        return item

    def _density_counts(self, object_rows: list[dict[str, Any]]) -> dict[str, int]:
        origin = self.ego.get_location()
        local = {"vehicle": 0, "person": 0}
        for label, pattern in (("vehicle", "vehicle.*"), ("person", "walker.pedestrian.*")):
            for actor in self.world.get_actors().filter(pattern):
                if int(actor.id) == int(self.ego.id):
                    continue
                try:
                    if actor.get_location().distance(origin) <= 50.0:
                        local[label] += 1
                except RuntimeError:
                    continue
        in_view = {
            label: sum(1 for row in object_rows if row.get("label") == label)
            for label in ("vehicle", "person")
        }
        eligible = {
            label: sum(
                1 for row in object_rows
                if row.get("label") == label
                and float(row.get("gt_bbox_area_px", 0.0)) >= 12.0
                and float(row.get("gt_distance_m", float("inf"))) <= 40.0
            )
            for label in ("vehicle", "person")
        }
        return {
            "local_vehicle_count": local["vehicle"],
            "local_person_count": local["person"],
            "in_view_vehicle_count": in_view["vehicle"],
            "in_view_person_count": in_view["person"],
            "training_eligible_vehicle_count": eligible["vehicle"],
            "training_eligible_person_count": eligible["person"],
        }

    def save_frame(self, frame_id: int, route_tick: int) -> None:
        image = self._wait_exact("rgb", frame_id)
        semantic_image = self._wait_exact("semantic", frame_id)
        depth_image = self._wait_exact("depth", frame_id)
        radar_measurement = self._wait_exact("radar", frame_id)
        timestamps = [
            float(image.timestamp), float(semantic_image.timestamp), float(depth_image.timestamp),
            float(radar_measurement.timestamp), float(self.world.get_snapshot().timestamp.elapsed_seconds),
        ]
        timestamp_delta = max(timestamps) - min(timestamps)
        self.max_timestamp_delta_s = max(self.max_timestamp_delta_s, timestamp_delta)
        if timestamp_delta > 1e-4:
            raise PilotError(f"timestamp misalignment at frame {frame_id}: delta={timestamp_delta:.9f}s")

        camera_matrix = self.parked.fusion_runtime.actor_world_matrix(self.camera)
        camera_inverse = self.parked.fusion_runtime.actor_world_inverse_matrix(self.camera)
        radar_matrix = self.parked.fusion_runtime.actor_world_matrix(self.radar)
        radar_inverse = self.parked.fusion_runtime.actor_world_inverse_matrix(self.radar)
        self.max_camera_transform_delta_m = max(
            self.max_camera_transform_delta_m,
            self._transform_delta_m(image.transform, self.camera.get_transform()),
        )
        self.max_radar_transform_delta_m = max(
            self.max_radar_transform_delta_m,
            self._transform_delta_m(radar_measurement.transform, self.radar.get_transform()),
        )

        detections = self.parked.radar_raw_to_alt_az_depth_velocity(bytes(radar_measurement.raw_data))
        radar_tensor, radar_points, radar_summary = self.parked.build_radar_sample(
            detections=detections,
            sensor_matrix=radar_matrix,
            camera_inverse_matrix=camera_inverse,
            camera_intrinsics=self.intrinsics_input,
            width=self.args.model_input_width,
            height=self.args.model_input_height,
            frame_time_s=float(radar_measurement.timestamp),
            tracker=self.tracker,
            max_range_m=self.args.radar_range,
            max_abs_velocity_mps=self.args.radar_max_velocity,
            parked_threshold_s=self.args.parked_threshold_s,
            point_radius_px=self.args.radar_raster_radius_px,
        )
        sample_id = f"{self.experiment_id}_{self.saved:06d}_frame{frame_id}"
        file_paths, mask = self.parked.save_sample_files(
            dataset_dir=self.output_dir,
            dirs=self.dirs,
            sample_id=sample_id,
            image=image,
            semantic_image=semantic_image,
            radar_tensor=radar_tensor,
            radar_points=radar_points,
            jpeg_quality=self.args.jpeg_quality,
        )
        manifest_row = self.parked.build_manifest_row(
            args=self.args,
            dataset_dir=self.output_dir,
            experiment_id=self.experiment_id,
            sample_id=sample_id,
            split="train",
            file_paths=file_paths,
            image=image,
            semantic_image=semantic_image,
            radar_measurement=radar_measurement,
            mask=mask,
            world=self.world,
            camera=self.camera,
            radar=self.radar,
            ego_vehicle=self.ego,
            camera_matrix=camera_matrix,
            camera_inverse_matrix=camera_inverse,
            radar_matrix=radar_matrix,
            radar_inverse_matrix=radar_inverse,
            intrinsics_full=self.intrinsics_full,
            radar_summary=radar_summary,
        )
        scenario_id = f"route_b_{self.density}_seed101_tm1101"
        manifest_row["scenario_id"] = scenario_id
        manifest_row["view_id"] = "qualified_route_b_controller"
        sample_base = {
            "experiment_id": self.experiment_id,
            "sample_id": sample_id,
            "frame_id": frame_id,
            "timestamp": float(image.timestamp),
            "traffic_light_id": "",
            "scenario_id": scenario_id,
            "view_id": "qualified_route_b_controller",
        }
        object_rows = self.parked.build_object_rows(
            world=self.world,
            ego_vehicle=self.ego,
            sample_base=sample_base,
            camera_location=self.camera.get_transform().location,
            camera_matrix=camera_matrix,
            camera_inverse_matrix=camera_inverse,
            intrinsics=self.intrinsics_full,
            width=self.args.camera_width,
            height=self.args.camera_height,
            max_distance_m=self.args.gt_max_distance_m,
            radar_world_xyz=self.np.asarray(radar_points["world_xyz"], dtype=self.np.float32),
            stationary_tracker=self.actor_tracker,
            include_pedestrians=True,
            radar_support_margin_m=self.args.radar_support_margin_m,
            radar_person_support_mode=self.args.radar_person_support_mode,
            radar_person_support_radius_m=self.args.radar_person_support_radius_m,
            radar_person_support_z_down_m=self.args.radar_person_support_z_down_m,
            radar_person_support_z_up_m=self.args.radar_person_support_z_up_m,
        )
        person_rows = [
            row for row in object_rows
            if row.get("label") == "person"
            and float(row.get("gt_bbox_w", 0.0)) > 0.0
            and float(row.get("gt_bbox_h", 0.0)) > 0.0
        ]
        if person_rows:
            boxes = [
                (
                    float(row["gt_bbox_x"]), float(row["gt_bbox_y"]),
                    float(row["gt_bbox_x"]) + float(row["gt_bbox_w"]),
                    float(row["gt_bbox_y"]) + float(row["gt_bbox_h"]),
                )
                for row in person_rows
            ]
            self.parked.rasterize_person_regions(mask, boxes, shape="box")
            if not self.parked.cv2.imwrite(str(file_paths["mask_path"]), mask):
                raise PilotError(f"failed to write person mask at frame {frame_id}")
        manifest_row["vehicle_pixels"] = int(self.np.count_nonzero(mask == 1))
        manifest_row["person_pixels"] = int(self.np.count_nonzero(mask == 2))
        counts = self._density_counts(object_rows)
        self.parked.append_manifest_rows(self.manifest_path, [manifest_row])
        self.parked.append_object_box_rows(self.object_boxes_path, object_rows)
        self.saved += 1
        self.sample_stats.append({
            "frame_id": frame_id,
            "timestamp_s": float(image.timestamp),
            "route_tick": route_tick,
            "raw_vehicle_count": sum(1 for row in object_rows if row.get("label") == "vehicle"),
            "raw_person_count": sum(1 for row in object_rows if row.get("label") == "person"),
            **counts,
        })
        if self.saved == 1 or self.saved % 50 == 0:
            print(
                f"perception saved={self.saved} frame={frame_id} "
                f"objects={len(object_rows)} timestamp_delta_s={timestamp_delta:.9f}",
                flush=True,
            )

    def stop_sensors(self) -> bool:
        sensor_ids = [int(sensor.id) for sensor in self.sensors]
        ok = True
        for sensor in self.sensors:
            try:
                sensor.stop()
            except RuntimeError:
                ok = False
        for sensor in reversed(self.sensors):
            try:
                if not sensor.destroy():
                    ok = False
            except RuntimeError:
                ok = False
        for actor_id in sensor_ids:
            try:
                actor = self.world.get_actor(actor_id)
                if actor is not None and actor.is_alive:
                    ok = False
            except RuntimeError:
                ok = False
        self.cleanup_succeeded = ok
        return ok

    @staticmethod
    def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        if not values:
            return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
        return {
            "minimum": min(values),
            "mean": sum(values) / len(values),
            "maximum": max(values),
        }

    def write_summary(self, route_result: dict[str, Any] | None, error: str = "") -> None:
        intervals = [
            self.sample_stats[index]["timestamp_s"] - self.sample_stats[index - 1]["timestamp_s"]
            for index in range(1, len(self.sample_stats))
        ]
        summary = {
            "schema": "scenesense_moving_ego_fusion_training_data.v1.route_summary",
            "density": self.density,
            "scenario_seed": self.scenario_seed,
            "traffic_manager_seed": self.tm_seed,
            "saved_samples": self.saved,
            "sampling": {
                "simulator_hz": 20,
                "save_every_nth_tick": 10,
                "target_interval_s": 0.5,
                "observed_interval_s_min": min(intervals) if intervals else None,
                "observed_interval_s_max": max(intervals) if intervals else None,
            },
            "sensor_alignment": {
                "max_timestamp_delta_s": self.max_timestamp_delta_s,
                "max_camera_transform_delta_m": self.max_camera_transform_delta_m,
                "max_radar_transform_delta_m": self.max_radar_transform_delta_m,
            },
            "counts": {
                key: self._aggregate(self.sample_stats, key)
                for key in (
                    "raw_vehicle_count", "raw_person_count",
                    "training_eligible_vehicle_count", "training_eligible_person_count",
                    "local_vehicle_count", "local_person_count",
                    "in_view_vehicle_count", "in_view_person_count",
                )
            },
            "per_frame_density_counts": self.sample_stats,
            "sensor_cleanup_succeeded": self.cleanup_succeeded,
            "error": error,
            "route_result": route_result,
        }
        self.parked.save_json(self.output_dir / "route_summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", required=True, choices=tuple(DENSITIES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--route-progress-csv", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--scenario-seed", type=int, default=101)
    parser.add_argument("--tm-seed", type=int, default=1101)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        preflight = verify_inputs(args)
    except (PilotError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Route B perception preflight failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({"preflight": "PASS", **preflight}, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0

    if (int(args.scenario_seed), int(args.tm_seed)) != (101, 1101):
        print("pilot requires the exact seed bundle 101/1101", file=sys.stderr)
        return 2

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import data_collection.run_route_b_density_loop as density
    import carla_collect_parked_ego_fusion_training_data as parked

    vehicles, pedestrians = DENSITIES[args.density]
    output_dir = Path(args.output_dir).resolve()
    density_argv = [
        "--density", args.density,
        "--vehicles", str(vehicles),
        "--pedestrians", str(pedestrians),
        "--loops", "1",
        "--seed", str(args.scenario_seed),
        "--host", str(args.host),
        "--port", str(args.port),
        "--tm-port", str(args.tm_port),
        "--route-config", str(Path(args.route_config).resolve()),
        "--lane-offset-m", "-0.5",
        "--walker-brake-distance-m", "10.0",
        "--fixed-delta-seconds", "0.05",
        "--real-time-tick-period-s", "0.05",
        "--no-spectator",
        "--out-csv", str(output_dir / "route_metrics.csv"),
        "--summary-json", str(output_dir / "route_metrics_summary.json"),
    ]
    density_args = density.build_parser().parse_args(density_argv)

    real_client_class = density.carla.Client
    density.carla.Client = lambda *a, **kw: ClientProxy(real_client_class, args.tm_seed, *a, **kw)
    original_drive = density.drive_one_loop_with_traffic
    collector_holder: dict[str, LegacyPerceptionCollector] = {}

    def collecting_drive(
        world: Any, vehicle: Any, agent: Any, route: dict[str, Any], collisions: Any,
        run_args: argparse.Namespace, loop_index: int, maintain: Any, janitor: Any,
    ) -> dict[str, Any]:
        collector = LegacyPerceptionCollector(
            parked=parked,
            world=world,
            ego=vehicle,
            output_dir=output_dir,
            density=args.density,
            vehicles=vehicles,
            pedestrians=pedestrians,
            scenario_seed=args.scenario_seed,
            tm_seed=args.tm_seed,
            route_path=Path(args.route_config),
            progress_path=Path(args.route_progress_csv),
        )
        collector_holder["collector"] = collector
        result: dict[str, Any] | None = None
        failure = ""
        try:
            result = original_drive(
                SamplingWorld(world, collector), vehicle, agent, route, collisions,
                run_args, loop_index, maintain, janitor,
            )
            return result
        except Exception as exc:
            failure = str(exc)
            collector.failure = failure
            raise
        finally:
            cleanup_ok = collector.stop_sensors()
            if result is not None and not cleanup_ok:
                result["completed"] = False
                result["abort_reason"] = "perception sensor cleanup failure"
            collector.write_summary(result, failure)

    density.drive_one_loop_with_traffic = collecting_drive
    try:
        return density.run(density_args)
    except (PilotError, density.RouteBError, RuntimeError) as exc:
        print(f"Route B perception episode failed: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        density.drive_one_loop_with_traffic = original_drive
        density.carla.Client = real_client_class


if __name__ == "__main__":
    raise SystemExit(main())
