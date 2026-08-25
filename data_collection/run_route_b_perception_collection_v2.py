#!/usr/bin/env python3
"""Collect one canonical Route B perception episode (v2 timing contract).

v2 differences from ``run_route_b_perception_collection.py``:

* **Cadence.** The world still ticks at 20 Hz, but the perception pipeline now
  runs at 10 Hz and persistence at 5 Hz (v1 saved 2 Hz). Radar input is the
  current plus the immediately previous logical 100 ms sweep - contiguous
  200 ms of support at the measured 200,000 PPS, never an inflated PPS.
* **Sensors are free-running.** ``sensor_tick`` is not trusted: a commanded
  0.1 s camera tick was measured to skip roughly one capture per 200 world
  ticks, which permanently shifts the 10 Hz phase. Cameras and radar now emit
  once per world tick and the cadence is derived from world ticks and
  timestamps (see :mod:`data_collection.radar_sweep_aggregator_v1`).
* **Epic rendering provenance.** Collection aborts unless the running server was
  launched with an explicit ``-quality-level=Epic`` and rendering enabled; the
  full launch command, quality level, rendering mode, resolution, weather and
  server version are written into the episode manifest.
* **Population and incident accounting.** Live NPC counts are recorded at every
  saved frame, deficit spans are measured against the replenish interval, and
  collision incident windows are emitted as a separate per-sample artifact so
  no frame is ever silently deleted.

The accepted density runner remains byte-for-byte unchanged. This adapter
injects a synchronous sensor sampler around its world ticks, supplies a separate
Traffic Manager seed, and writes the existing manifest/object-box format.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from data_collection.radar_sweep_aggregator_v1 import (  # noqa: E402
    PREPARE_EVERY_N_TICKS,
    RADAR_POINTS_PER_SECOND,
    SAVE_EVERY_N_PREPARED,
    SWEEP_PERIOD_S,
    WINDOW_SWEEPS,
    WORLD_DELTA_S,
    WORLD_TICK_HZ,
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
    inspect_launch,
    render_provenance,
)
DEFAULT_ROUTE = HERE / "routes" / "town10hd_opt_route_b_full_map_loop_v1.json"
DEFAULT_PROGRESS = HERE / "routes" / "town10hd_opt_route_b_full_map_loop_v1.progress.csv"
ACCEPTED_DENSITY_RUNNER = HERE / "run_route_b_density_loop.py"
EXPECTED_ROUTE_SHA256 = "fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5"
EXPECTED_PROGRESS_SHA256 = "974593859368f24ee2bc4ac31b82118bf2e932d0de1c96858b8771e2dd4d90c0"
# Re-verified 2026-08-24 after the reviewed population-ledger + traffic-profile change
# to the density runner (ROUTE_B_TRAFFIC_30_50_CONFIGURATION.md §3). Previous accepted
# value: 59592ee83184a227f324ff872d1cc7f5601d5a1efb0300dc08dec7b7f26749a4.
EXPECTED_RUNNER_SHA256 = "0f3147fad30cf9e03657c98373c8ef2783af04853ccefc97dc1acbdfa33b336b"
DENSITIES = {
    # Canonical v2 profiles. The name states the requested actor counts only.
    "traffic_30_30": (30, 30),
    "traffic_50_50": (50, 50),
}
DEFAULT_TARGET_SPEED_KPH = 25.0
# Registered pilot corpus bundles: one bundle per split, per density. Splits are
# by complete episode, never by frame.
SEED_BUNDLES = {
    1: {"scenario_seed": 101, "tm_seed": 1101, "split": "train"},
    2: {"scenario_seed": 202, "tm_seed": 1202, "split": "val"},
    3: {"scenario_seed": 303, "tm_seed": 1303, "split": "test"},
    # Collected only if the architecture pilot passes; a second training episode.
    4: {"scenario_seed": 404, "tm_seed": 1404, "split": "train"},
}
# Smoke bundles reuse the already-qualified Route B seeds and are never a split.
SMOKE_SEED_BUNDLES = {(101, 1101), (31, 31)}
ALLOWED_SEED_BUNDLES = SMOKE_SEED_BUNDLES | {
    (bundle["scenario_seed"], bundle["tm_seed"]) for bundle in SEED_BUNDLES.values()
}
SENSOR_NAMES = ("rgb", "semantic", "depth", "radar")
CAMERA_NAMES = ("rgb", "semantic", "depth")
QUALIFIED_WALKER_BRAKE_DISTANCE_M = 10.0
POPULATION_ALIVE_FRACTION_GATE = 0.95
# Same fraction and the same replenish+2 s simulated bound as the body gate.
CONTROLLER_READY_FRACTION_GATE = 0.95
COLLISION_INCIDENT_WINDOW_S = 2.0
CADENCE_WARMUP_TICKS = 20
# Measured, not assumed - see data_collection/route_b_perception_v2/
# radar_velocity_sign_diagnostic_v1.json (83/83 boresight samples negative while
# closing, 9/9 positive while receding, |ratio to ego speed| = 0.989).
RASTERIZER_CHOICES = ("fast", "legacy")
DEFAULT_RASTERIZER = "fast"
# Accepted numerical-equivalence tolerance for rasterize_radar_channels_fast,
# measured on 40 real saved 30/30 frames (mean 37,105 returns/frame):
# occupancy / radial_velocity / stationary_age bit-identical, inverse_range
# differing on 0.84% of elements by at most 5.96e-08 (~half a float32 ULP at 1.0),
# no equal-magnitude velocity tie observed, 23.3x faster.
FAST_RASTERIZER_TOLERANCE = {
    "bit_identical_channels": ["occupancy", "radial_velocity", "stationary_age"],
    "tolerant_channels": ["inverse_range"],
    "max_abs_difference": 5.960464477539063e-08,
    "max_abs_difference_note": "approximately half a float32 ULP at 1.0",
    "evidence": "data_collection/route_b_perception_v2/rasterizer_comparison_v1.json",
}
RADAR_VELOCITY_SIGN_CONVENTION = (
    "CARLA radar velocity is the range rate: negative = closing (range decreasing), "
    "positive = receding; magnitude equals the closing speed"
)


class TickOwnershipError(RuntimeError):
    """A CARLA world frame advanced outside the route's tick owner."""


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
        "target_speed_kph": float(args.target_speed_kph),
        "hybrid_physics": bool(args.hybrid_physics),
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
    """Drive the 20 Hz -> 10 Hz -> 5 Hz cadence around the route runner's ticks.

    Every world tick ingests one radar callback into the logical-sweep
    aggregator. Every second tick prepares a model input from the current plus
    previous 100 ms sweep, and every second prepared input is persisted.

    This object is also the route's sole tick owner. Any frame that advances
    without passing through ``tick`` is a correctness failure, not reduced
    effective Hz, so the frame IDs are required to be strictly contiguous.
    """

    def __init__(
        self,
        world: Any,
        collector: "PerceptionCollectorV2",
        population: Any = None,
    ) -> None:
        self._world = world
        self._collector = collector
        self._route_ticks = 0
        self._last_frame_id: int | None = None
        self._population = population

    def _last_population_event(self) -> str:
        events = getattr(self._population, "lifecycle_events", None)
        if not events:
            return "none recorded"
        return json.dumps(events[-1], sort_keys=True)

    def tick(self, *args: Any, **kwargs: Any) -> int:
        previous = self._last_frame_id
        frame_id = int(self._world.tick(*args, **kwargs))
        if previous is not None and frame_id != previous + 1:
            raise TickOwnershipError(
                f"unobserved CARLA world frame(s) between {previous} and {frame_id} "
                f"(gap={frame_id - previous - 1}) at route tick {self._route_ticks + 1}: "
                "only SamplingWorld.tick() may advance the world during the route; "
                f"most recent population event: {self._last_population_event()}"
            )
        self._last_frame_id = frame_id
        self._route_ticks += 1
        self._collector.on_world_tick(frame_id, self._route_ticks)
        return frame_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._world, name)


class PerceptionCollectorV2:
    """Synchronous 10 Hz prepare / 5 Hz persist sampler on the historical layout."""

    def __init__(
        self,
        *,
        parked: Any,
        world: Any,
        client: Any,
        ego: Any,
        collisions: Any,
        rpc_port: int,
        split: str,
        seed_bundle: int | None,
        rasterizer: str,
        output_dir: Path,
        density: str,
        vehicles: int,
        pedestrians: int,
        scenario_seed: int,
        tm_seed: int,
        target_speed_kph: float,
        hybrid_physics: bool,
        route_path: Path,
        progress_path: Path,
        population: Any = None,
    ) -> None:
        import numpy as np

        self.np = np
        self.parked = parked
        # Read-only handle used for controller-health telemetry. Never a model input.
        self.population = population
        self.world = world
        self.client = client
        self.ego = ego
        self.collisions = collisions
        self.rpc_port = int(rpc_port)
        self.split = str(split)
        self.seed_bundle = seed_bundle
        self.output_dir = output_dir.resolve()
        self.density = density
        self.vehicles = int(vehicles)
        self.pedestrians = int(pedestrians)
        self.scenario_seed = int(scenario_seed)
        self.tm_seed = int(tm_seed)
        self.target_speed_kph = float(target_speed_kph)
        self.hybrid_physics = bool(hybrid_physics)
        self.scenario_id = f"route_b_{density}_seed{int(scenario_seed)}_tm{int(tm_seed)}"
        self.route_path = route_path.resolve()
        self.progress_path = progress_path.resolve()
        self.experiment_id = self.output_dir.name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.dirs = parked.prepare_dataset_dirs(self.output_dir)
        self.manifest_path = self.output_dir / "manifest.csv"
        self.object_boxes_path = self.output_dir / "object_boxes.csv"
        # Unbounded: dropping a callback would silently break the logical sweep.
        # Bounded by construction because every queue is drained each tick.
        self.queues = {name: queue.Queue() for name in SENSOR_NAMES}
        self.sensors: list[Any] = []
        self.saved = 0
        self.sample_stats: list[dict[str, Any]] = []
        self.max_timestamp_delta_s = 0.0
        self.aggregator = RadarSweepAggregator()
        self.plan = CadencePlan()
        self.warmup_camera_frames: list[int] = []
        self.route_ticks = 0
        self.prepared_records: list[dict[str, Any]] = []
        self.frame_content_failures: list[dict[str, Any]] = []
        self.frame_content_checks = 0
        self.prepare_wall_s: list[float] = []
        self.population_samples: list[dict[str, Any]] = []
        self.controller_health_samples: list[dict[str, Any]] = []
        self.initial_population: dict[str, int] = {}
        self.saved_bytes = 0
        self.cadence_failures: list[str] = []
        self.runtime_provenance_rows: list[dict[str, Any]] = []
        self.max_camera_transform_delta_m = 0.0
        self.max_radar_transform_delta_m = 0.0
        self.cleanup_succeeded = False
        self.cleanup_tick = "not_run"
        self.cleanup_records: list[dict[str, Any]] = []
        self.cleanup_warnings: list[str] = []
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
            radar_rasterizer=str(rasterizer),
            # Two logical 100 ms sweeps = contiguous 200 ms of radar support.
            radar_temporal_window_frames=WINDOW_SWEEPS,
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
        # Renderer preflight: refuse to collect anything unless the running
        # server proves explicit Epic quality with rendering enabled.
        self.render_provenance = render_provenance(
            self.world,
            self.client,
            port=self.rpc_port,
            camera_width=self.args.camera_width,
            camera_height=self.args.camera_height,
            camera_fov=self.args.camera_fov,
        )
        assert_epic_rendering(self.render_provenance)

        self.initial_population = self._live_population()
        self._spawn_sensors()
        self._lock_cadence_phase()
        self._write_metadata()

    # -- population truth -------------------------------------------------
    def _live_population(self) -> dict[str, int]:
        """Live NPC counts read from the world, not from the ownership set."""
        actors = self.world.get_actors()
        ego_id = int(self.ego.id)
        vehicles = sum(
            1 for actor in actors.filter("vehicle.*") if int(actor.id) != ego_id
        )
        walkers = sum(1 for _ in actors.filter("walker.pedestrian.*"))
        return {"npc_vehicles_alive": vehicles, "npc_pedestrians_alive": walkers}

    def _controller_health(self) -> dict[str, int]:
        """Walker-controller health for one saved frame.

        Liveness comes from a world snapshot, phase and readiness from the
        manager's registry. Purely observational: no controller id and no
        readiness flag ever reaches the manifest, the mask, or a model tensor.
        """
        blank = {
            "managed_walker_bodies_alive": -1,
            "live_attached_walker_controllers": -1,
            "live_ready_walker_controllers": -1,
            "controllers_marked_ready": -1,
            "pending_body_phase": -1,
            "pending_controller_phase": -1,
            "orphan_controllers": -1,
        }
        population = self.population
        if population is None:
            return blank
        try:
            census = population.phase_summary()
            snapshot = self.world.get_actors()
        except (AttributeError, RuntimeError):
            return blank
        alive: dict[int, Any] = {}
        for actor in snapshot:
            try:
                if actor.is_alive:
                    alive[int(actor.id)] = actor
            except (AttributeError, RuntimeError):
                continue
        bodies_alive = 0
        attached = 0
        live_ready = 0
        for record in list(population.walkers):
            body_id = record.get("id")
            body = alive.get(int(body_id)) if body_id is not None else None
            if body is None or not str(body.type_id).startswith("walker.pedestrian."):
                continue
            bodies_alive += 1
            controller_id = record.get("con")
            controller = alive.get(int(controller_id)) if controller_id is not None else None
            if controller is None or not str(controller.type_id).startswith("controller.ai.walker"):
                continue
            try:
                parent = controller.parent
            except (AttributeError, RuntimeError):
                parent = None
            if parent is None or int(parent.id) != int(body_id):
                continue
            attached += 1
            # Terminal health: live in this snapshot, parented to this live body,
            # and started. The registry flag alone can outlive the actor it
            # describes, so it is kept for diagnosis but never gates on its own.
            if record.get("controller_ready", False):
                live_ready += 1
        return {
            "managed_walker_bodies_alive": bodies_alive,
            "live_attached_walker_controllers": attached,
            "live_ready_walker_controllers": live_ready,
            "controllers_marked_ready": int(census["controllers_marked_ready"]),
            "pending_body_phase": int(census["pending_body_phase"]),
            "pending_controller_phase": int(census["pending_controller_phase"]),
            "orphan_controllers": int(census["orphan_controllers"]),
        }

    # -- cadence phase ----------------------------------------------------
    def _lock_cadence_phase(self) -> None:
        """Observe the free-running camera stream and lock the 10 Hz phase.

        Runs before the route starts. The cameras must deliver one capture per
        world tick; any other pattern means ``sensor_tick`` is being applied and
        the derived cadence would be unsound.
        """
        for _ in range(CADENCE_WARMUP_TICKS):
            frame_id = int(self.world.tick())
            measurement = self._drain_exact("radar", frame_id, 10.0)
            if measurement is None:
                raise PilotError(f"no radar callback during cadence warmup at frame {frame_id}")
            self.aggregator.ingest(measurement)
            for name in CAMERA_NAMES:
                while True:
                    try:
                        item = self.queues[name].get_nowait()
                    except queue.Empty:
                        break
                    if name == "rgb":
                        self.warmup_camera_frames.append(int(item.frame))
        if len(self.warmup_camera_frames) < 2:
            raise PilotError(
                f"camera cadence not observed during warmup: frames={self.warmup_camera_frames}"
            )
        gaps = {
            self.warmup_camera_frames[index] - self.warmup_camera_frames[index - 1]
            for index in range(1, len(self.warmup_camera_frames))
        }
        if gaps != {1}:
            raise PilotError(
                f"free-running cameras are not one capture per world tick: gaps={sorted(gaps)}"
            )
        self.plan.lock_camera_parity(self.warmup_camera_frames[-1])

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
            # Free-running. A commanded 0.1 s tick was measured to skip about one
            # capture per 200 world ticks, which permanently shifts the 10 Hz phase.
            bp.set_attribute("sensor_tick", "0.0")
            sensor = self.world.spawn_actor(bp, camera_transform, attach_to=self.ego)
            sensor.listen(lambda item, name=queue_name: self.queues[name].put(item))
            self.sensors.append(sensor)
        radar_bp = bp_lib.find("sensor.other.radar")
        radar_bp.set_attribute("range", str(self.args.radar_range))
        radar_bp.set_attribute("horizontal_fov", str(self.args.radar_hfov))
        radar_bp.set_attribute("vertical_fov", str(self.args.radar_vfov))
        radar_bp.set_attribute("points_per_second", str(self.args.radar_points_per_second))
        # Free-running: one raw callback per world tick. The logical 100 ms sweep
        # is rebuilt from timestamps, never commanded through sensor_tick and never
        # faked by inflating points_per_second.
        radar_bp.set_attribute("sensor_tick", "0.0")
        radar = self.world.spawn_actor(
            radar_bp, self.parked.fusion_runtime._ego_radar_transform(self.args), attach_to=self.ego
        )
        radar.listen(lambda item: self.queues["radar"].put(item))
        self.sensors.append(radar)
        self.camera, self.semantic_camera, self.depth_camera, self.radar = self.sensors
        self.radar_attributes = {
            key: str(radar.attributes.get(key))
            for key in ("points_per_second", "sensor_tick", "range", "horizontal_fov", "vertical_fov")
        }

    def _write_metadata(self) -> None:
        metadata = {
            "schema": "scenesense_moving_ego_fusion_training_data.v2",
            "experiment_id": self.experiment_id,
            "description": "Canonical Route B moving-ego perception episode, v2 timing contract.",
            "world": str(self.world.get_map().name),
            "scenario_id": self.scenario_id,
            "view_id": "qualified_route_b_controller",
            "split": self.split,
            "seed_bundle": self.seed_bundle,
            "density": self.density,
            "requested_vehicles": self.vehicles,
            "requested_pedestrians": self.pedestrians,
            "scenario_seed": self.scenario_seed,
            "traffic_manager_seed": self.tm_seed,
            "simulator_hz": WORLD_TICK_HZ,
            "prepare_every_nth_tick": PREPARE_EVERY_N_TICKS,
            "save_every_nth_prepared_input": SAVE_EVERY_N_PREPARED,
            "prepared_hz": WORLD_TICK_HZ / PREPARE_EVERY_N_TICKS,
            "saved_hz": WORLD_TICK_HZ / (PREPARE_EVERY_N_TICKS * SAVE_EVERY_N_PREPARED),
            "route_file": str(self.route_path),
            "route_file_sha256": EXPECTED_ROUTE_SHA256,
            "route_progress_csv": str(self.progress_path),
            "route_progress_csv_sha256": EXPECTED_PROGRESS_SHA256,
            "qualified_density_runner_sha256": EXPECTED_RUNNER_SHA256,
            "controller": {
                "lane_offset_m": -0.5,
                "walker_detection_distance_m": QUALIFIED_WALKER_BRAKE_DISTANCE_M,
                "npc_hardening": True,
                "safe_vehicle_filter": True,
                "interventions": False,
                "target_speed_kph": self.target_speed_kph,
                "hybrid_physics": self.hybrid_physics,
            },
            "weather": weather_payload(self.world.get_weather()),
            "camera_resolution": [self.args.camera_width, self.args.camera_height],
            "model_input_size": [self.args.model_input_width, self.args.model_input_height],
            "sensor_tick_s": 0.0,
            "sensor_tick_policy": "free-running; cadence derived from world ticks and timestamps",
            "radar": {
                "points_per_second": self.args.radar_points_per_second,
                "range_m": self.args.radar_range,
                "horizontal_fov": self.args.radar_hfov,
                "vertical_fov": self.args.radar_vfov,
                "sweep_period_s": SWEEP_PERIOD_S,
                "temporal_window_sweeps": WINDOW_SWEEPS,
                "temporal_support_s": SWEEP_PERIOD_S * WINDOW_SWEEPS,
                "configured_attributes": self.radar_attributes,
                "radial_velocity_sign_convention": RADAR_VELOCITY_SIGN_CONVENTION,
                "raw_layout": "CARLA raw float32 order is (velocity, azimuth, altitude, depth)",
                "rasterizer": self.args.radar_rasterizer,
                "rasterizer_tolerance": (
                    FAST_RASTERIZER_TOLERANCE if self.args.radar_rasterizer == "fast" else None
                ),
                "raw_range_note": (
                    "CARLA occasionally reports original_range_m beyond the configured range "
                    "(measured 0.043% of returns, up to 172.7 m at a 120 m setting). Raw "
                    "provenance is saved unmodified and unclamped; a downstream reducer should "
                    "treat only finite returns with 0 < original_range_m <= range_m as range-valid."
                ),
            },
            "ground_truth": {
                "convention": "actor origin",
                "gt_max_distance_m": self.args.gt_max_distance_m,
                "raw_gt_preserved": True,
                "evaluation_eligibility": {
                    "camera_frustum": "projected bounding-box centre inside the full-resolution frame",
                    "min_projected_area_px": 12.0,
                    "max_distance_m": 40.0,
                    "note": "eligibility is an evaluation filter only; collection is unrestricted",
                },
                "person_mask_provenance": (
                    "semantic-tag training mask, then person regions overpainted as filled "
                    "axis-aligned boxes from projected actor bounding boxes "
                    "(parked.rasterize_person_regions shape='box')"
                ),
                "visibility_fields": "recorded where the historical object-box schema provides them",
            },
            "render_provenance": self.render_provenance,
        }
        self.parked.save_json(self.output_dir / "metadata.json", metadata)

    @staticmethod
    def _transform_delta_m(a: Any, b: Any) -> float:
        return math.sqrt(
            (float(a.location.x) - float(b.location.x)) ** 2
            + (float(a.location.y) - float(b.location.y)) ** 2
            + (float(a.location.z) - float(b.location.z)) ** 2
        )

    def _drain_exact(self, name: str, frame_id: int, timeout_s: float) -> Any:
        """Return the record for ``frame_id`` exactly, discarding strictly older ones."""
        sensor_queue = self.queues[name]
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
            if observed < frame_id:
                continue
            return item if observed == frame_id else None

    def _wait_exact(self, name: str, frame_id: int) -> Any:
        item = self._drain_exact(name, frame_id, 5.0)
        if item is None:
            raise PilotError(f"missing or misaligned {name} sensor record at frame {frame_id}")
        return item

    # -- cadence driver ---------------------------------------------------
    def on_world_tick(self, frame_id: int, route_tick: int) -> None:
        """One 20 Hz world tick: always ingest radar, prepare at 10 Hz, save at 5 Hz."""
        self.route_ticks = route_tick
        measurement = self._drain_exact("radar", frame_id, 10.0)
        if measurement is None:
            raise PilotError(f"missing radar callback at world frame {frame_id}")
        self.aggregator.ingest(measurement)
        self.plan.note_tick(float(measurement.timestamp))
        if not self.plan.is_prepare_frame(frame_id):
            return

        if self.aggregator.anchor_s is None:
            # Bin boundaries close exactly on prepared-input ticks.
            self.aggregator.set_anchor(float(measurement.timestamp))
        sweep_index = self.aggregator.sweep_index_for(float(measurement.timestamp))
        if not self.aggregator.has_window(sweep_index):
            self.plan.warmup_prepared_skipped += 1
            return
        self.prepare_input(frame_id, route_tick, measurement, sweep_index)

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

    def prepare_input(
        self,
        frame_id: int,
        route_tick: int,
        radar_measurement: Any,
        sweep_index: int,
    ) -> None:
        """Build one 10 Hz model input; persist every second one at 5 Hz.

        The full input - including ``build_radar_sample`` and therefore the
        stationary-track accumulator update - is built at the deployed 10 Hz
        rate so a persisted frame is bit-identical to what the 10 Hz pipeline
        would produce at that instant. Persistence is the only thing decimated.
        """
        started = time.perf_counter()
        image = self._wait_exact("rgb", frame_id)
        semantic_image = self._wait_exact("semantic", frame_id)
        depth_image = self._wait_exact("depth", frame_id)
        timestamps = [
            float(image.timestamp), float(semantic_image.timestamp), float(depth_image.timestamp),
            float(radar_measurement.timestamp), float(self.world.get_snapshot().timestamp.elapsed_seconds),
        ]
        timestamp_delta = max(timestamps) - min(timestamps)
        self.max_timestamp_delta_s = max(self.max_timestamp_delta_s, timestamp_delta)
        if timestamp_delta > 1e-4:
            raise PilotError(f"timestamp misalignment at frame {frame_id}: delta={timestamp_delta:.9f}s")

        frame_check = check_frames(
            image, semantic_image,
            width=self.args.camera_width, height=self.args.camera_height,
        )
        self.frame_content_checks += 1
        if not frame_check["ok"]:
            self.frame_content_failures.append(
                {"frame_id": frame_id, "problems": frame_check["problems"]}
            )

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

        # Current plus immediately previous logical sweep, motion compensated into
        # the radar pose at this prepared tick.
        detections, window_meta = self.aggregator.window_detections(
            sweep_index,
            sensor_inverse_matrix=radar_inverse,
            reference_timestamp_s=float(radar_measurement.timestamp),
        )
        if window_meta["sweep_indices"] != [sweep_index - 1, sweep_index]:
            self.cadence_failures.append(
                f"frame {frame_id}: non-consecutive temporal window {window_meta['sweep_indices']}"
            )
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
            rasterizer=self.args.radar_rasterizer,
        )
        self.prepare_wall_s.append(time.perf_counter() - started)

        ego_velocity = self.ego.get_velocity()
        ego_speed_mps = math.sqrt(
            float(ego_velocity.x) ** 2 + float(ego_velocity.y) ** 2 + float(ego_velocity.z) ** 2
        )

        expected_shape = (4, self.args.model_input_height, self.args.model_input_width)
        if tuple(radar_tensor.shape) != expected_shape:
            raise PilotError(
                f"radar tensor shape drift at frame {frame_id}: "
                f"{tuple(radar_tensor.shape)} != {expected_shape}"
            )
        if not bool(self.np.all(self.np.isfinite(radar_tensor))):
            raise PilotError(f"non-finite radar tensor at frame {frame_id}")

        must_save = self.plan.note_prepared(float(radar_measurement.timestamp))
        self.prepared_records.append({
            "frame_id": frame_id,
            "route_tick": route_tick,
            "timestamp_s": float(radar_measurement.timestamp),
            "sweep_index": sweep_index,
            "window_sweeps": window_meta["sweep_indices"],
            "window_callbacks": window_meta["callbacks"],
            "window_returns": window_meta["returns"],
            "window_span_s": window_meta["window_span_s"],
            "persisted": bool(must_save),
        })
        if not must_save:
            return

        self._persist(
            frame_id=frame_id,
            route_tick=route_tick,
            image=image,
            semantic_image=semantic_image,
            radar_measurement=radar_measurement,
            radar_tensor=radar_tensor,
            radar_points=radar_points,
            radar_summary=radar_summary,
            camera_matrix=camera_matrix,
            camera_inverse=camera_inverse,
            radar_matrix=radar_matrix,
            radar_inverse=radar_inverse,
            window_meta=window_meta,
            timestamp_delta=timestamp_delta,
            prepared_timestamp_s=float(radar_measurement.timestamp),
            ego_speed_mps=ego_speed_mps,
            ego_velocity_mps=(float(ego_velocity.x), float(ego_velocity.y), float(ego_velocity.z)),
        )

    def _persist(
        self,
        *,
        frame_id: int,
        route_tick: int,
        image: Any,
        semantic_image: Any,
        radar_measurement: Any,
        radar_tensor: Any,
        radar_points: dict[str, Any],
        radar_summary: dict[str, float],
        camera_matrix: Any,
        camera_inverse: Any,
        radar_matrix: Any,
        radar_inverse: Any,
        window_meta: dict[str, Any],
        timestamp_delta: float,
        prepared_timestamp_s: float,
        ego_speed_mps: float,
        ego_velocity_mps: tuple[float, float, float],
    ) -> None:
        sample_id = f"{self.experiment_id}_{self.saved:06d}_frame{frame_id}"
        # Raw runtime-observable provenance for later radar-activity analysis.
        # Per-return rows are aligned with the existing world/camera/projection
        # arrays; nothing here uses ground truth or actor identity.
        radar_points = dict(radar_points)
        radar_points.update(window_meta["raw_provenance"])
        radar_points["prepared_timestamp_s"] = self.np.asarray(
            [prepared_timestamp_s], dtype=self.np.float64
        )
        radar_points["ego_speed_mps"] = self.np.asarray([ego_speed_mps], dtype=self.np.float32)
        radar_points["ego_velocity_mps"] = self.np.asarray(ego_velocity_mps, dtype=self.np.float32)
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
            split=self.split,
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
        scenario_id = self.scenario_id
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

        missing = [str(path) for path in file_paths.values() if not Path(path).is_file()]
        if missing:
            raise PilotError(f"saved record missing at frame {frame_id}: {missing}")
        sample_bytes = sum(Path(path).stat().st_size for path in file_paths.values())
        self.saved_bytes += sample_bytes

        self.parked.append_manifest_rows(self.manifest_path, [manifest_row])
        self.parked.append_object_box_rows(self.object_boxes_path, object_rows)

        self.runtime_provenance_rows.append({
            "sample_id": sample_id,
            "frame_id": frame_id,
            "prepared_timestamp_s": round(float(prepared_timestamp_s), 6),
            "ego_speed_mps": round(float(ego_speed_mps), 6),
            "ego_velocity_x_mps": round(float(ego_velocity_mps[0]), 6),
            "ego_velocity_y_mps": round(float(ego_velocity_mps[1]), 6),
            "ego_velocity_z_mps": round(float(ego_velocity_mps[2]), 6),
            "radar_window_returns": int(window_meta["returns"]),
            "sweep_index": int(window_meta["sweep_indices"][-1]),
        })

        population = self._live_population()
        controller_health = self._controller_health()
        self.population_samples.append({
            "sample_id": sample_id,
            "frame_id": frame_id,
            "timestamp_s": float(image.timestamp),
            **population,
        })
        self.controller_health_samples.append({
            "sample_id": sample_id,
            "frame_id": frame_id,
            "timestamp_s": float(image.timestamp),
            **controller_health,
        })

        self.saved += 1
        self.sample_stats.append({
            "sample_id": sample_id,
            "frame_id": frame_id,
            "timestamp_s": float(image.timestamp),
            "route_tick": route_tick,
            "sample_bytes": int(sample_bytes),
            "timestamp_delta_s": float(timestamp_delta),
            "radar_window_returns": int(window_meta["returns"]),
            "raw_vehicle_count": sum(1 for row in object_rows if row.get("label") == "vehicle"),
            "raw_person_count": sum(1 for row in object_rows if row.get("label") == "person"),
            **population,
            **counts,
            **controller_health,
        })
        if self.saved == 1 or self.saved % 100 == 0:
            print(
                f"perception saved={self.saved} prepared={self.plan.prepared} frame={frame_id} "
                f"radar_window_returns={window_meta['returns']} objects={len(object_rows)} "
                f"timestamp_delta_s={timestamp_delta:.9f}",
                flush=True,
            )

    def stop_sensors(self) -> bool:
        """Stop and destroy the perception sensors, then verify final actor absence.

        In synchronous mode a destroy request is only applied by the server on the next
        tick, so liveness is verified after one dedicated cleanup tick. Success is decided
        by final absence: a False destroy return on a confirmed-absent actor is a warning,
        while a surviving actor or unavailable verification is a failure.
        """
        records: list[dict[str, Any]] = []
        for name, sensor in zip(SENSOR_NAMES, self.sensors):
            record: dict[str, Any] = {
                "sensor": name,
                "type_id": str(sensor.type_id),
                "actor_id": int(sensor.id),
                "stop_result": "ok",
                "destroy_result": None,
                "final_state": "unverified",
            }
            try:
                sensor.stop()
            except RuntimeError as exc:
                record["stop_result"] = f"error: {exc}"
            records.append(record)
        for index in range(len(self.sensors) - 1, -1, -1):
            try:
                records[index]["destroy_result"] = bool(self.sensors[index].destroy())
            except RuntimeError as exc:
                records[index]["destroy_result"] = f"error: {exc}"
        cleanup_tick = "ok"
        try:
            self.world.tick()
        except RuntimeError as exc:
            cleanup_tick = f"error: {exc}"
        for record in records:
            try:
                actor = self.world.get_actor(record["actor_id"])
                record["final_state"] = "absent" if actor is None or not actor.is_alive else "alive"
            except RuntimeError as exc:
                record["final_state"] = f"unverified: {exc}"

        ok = cleanup_tick == "ok"
        warnings: list[str] = []
        if cleanup_tick != "ok":
            warnings.append(f"cleanup tick failed, final verification unavailable ({cleanup_tick})")
        for record in records:
            if record["final_state"] != "absent":
                ok = False
                continue
            if record["destroy_result"] is not True:
                warnings.append(
                    f"{record['sensor']} destroy returned {record['destroy_result']!r} "
                    "but the actor is confirmed absent"
                )
            if record["stop_result"] != "ok":
                warnings.append(
                    f"{record['sensor']} stop reported {record['stop_result']!r} "
                    "but the actor is confirmed absent"
                )
        self.cleanup_tick = cleanup_tick
        self.cleanup_records = records
        self.cleanup_warnings = warnings
        self.cleanup_succeeded = ok
        for message in warnings:
            print(f"perception sensor cleanup warning: {message}", flush=True)
        print(
            "perception sensor cleanup "
            f"{'OK' if ok else 'FAILED'}: "
            + ", ".join(f"{r['sensor']}#{r['actor_id']}={r['final_state']}" for r in records),
            flush=True,
        )
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

    # -- episode-level accounting ----------------------------------------
    def _population_report(self) -> dict[str, Any]:
        """Live-population gates evaluated on saved frames only."""
        requested = {"npc_vehicles_alive": self.vehicles, "npc_pedestrians_alive": self.pedestrians}
        report: dict[str, Any] = {
            "requested": {"vehicles": self.vehicles, "pedestrians": self.pedestrians},
            "at_episode_start": self.initial_population,
            "exact_at_start": all(
                self.initial_population.get(key) == value for key, value in requested.items()
            ),
            "alive_fraction_gate": POPULATION_ALIVE_FRACTION_GATE,
        }
        for key, wanted in requested.items():
            values = [int(row[key]) for row in self.population_samples]
            floor = math.ceil(POPULATION_ALIVE_FRACTION_GATE * float(wanted))
            below = [row for row in self.population_samples if int(row[key]) < floor]
            report[key] = {
                "requested": wanted,
                "floor": floor,
                **summarize(values),
                "frames_below_floor": len(below),
                "first_frame_below_floor": below[0]["frame_id"] if below else None,
            }

        # Deficit spans measured in simulated seconds across saved frames. A gap in
        # saved frames longer than one save period ends the span conservatively.
        spans: list[dict[str, Any]] = []
        for key, wanted in requested.items():
            open_span: dict[str, Any] | None = None
            for row in self.population_samples:
                deficit = int(row[key]) < int(wanted)
                if deficit and open_span is None:
                    open_span = {"actor_kind": key, "start_s": row["timestamp_s"], "end_s": row["timestamp_s"]}
                elif deficit and open_span is not None:
                    open_span["end_s"] = row["timestamp_s"]
                elif not deficit and open_span is not None:
                    spans.append(open_span)
                    open_span = None
            if open_span is not None:
                spans.append(open_span)
        for span in spans:
            span["duration_s"] = round(float(span["end_s"]) - float(span["start_s"]), 3)
        report["deficit_spans"] = spans
        report["max_deficit_span_s"] = max((span["duration_s"] for span in spans), default=0.0)
        return report

    def _controller_health_report(self, deficit_limit_s: float) -> dict[str, Any]:
        """Controller-health gates evaluated on saved frames only.

        A frame is "ready" when at least 95% of the managed bodies alive in that
        frame have a controller that is *simultaneously* live in the world
        snapshot, parented to that live body, and marked ready - the
        ``live_ready_walker_controllers`` field. ``controllers_marked_ready`` is
        a registry flag that can outlive the actor it describes, so it is
        retained for diagnosis and never gates on its own.

        The deficit bound is the same ``replenish_interval + 2 s`` of simulated
        time used for body population: a replacement is allowed to be mid-phase
        for one reconcile plus slack, not indefinitely.
        """
        rows = self.controller_health_samples
        report: dict[str, Any] = {
            "ready_fraction_gate": CONTROLLER_READY_FRACTION_GATE,
            "terminal_gate_field": "live_ready_walker_controllers",
            "diagnostic_only_fields": [
                "live_attached_walker_controllers", "controllers_marked_ready",
                "controllers_ready_95pct_every_saved_frame_diagnostic"],
            "deficit_limit_s": round(float(deficit_limit_s), 3),
            "frames": len(rows),
            "population_manager_observed": bool(
                rows and rows[0]["managed_walker_bodies_alive"] >= 0),
        }
        if rows and "live_ready_walker_controllers" not in rows[0]:
            report["population_manager_observed"] = False
        if not rows or not report["population_manager_observed"]:
            report.update({
                "ready_controllers_min": None,
                "controllers_marked_ready_min": None,
                "live_attached_controllers_min": None,
                "frames_below_ready_floor": None,
                "first_frame_below_ready_floor": None,
                "deficit_spans": [],
                "max_controller_deficit_span_s": 0.0,
                "controllers_ready_95pct_every_saved_frame_diagnostic": None,
            })
            return report

        for key in (
            "managed_walker_bodies_alive", "live_attached_walker_controllers",
            "live_ready_walker_controllers", "controllers_marked_ready",
            "pending_body_phase", "pending_controller_phase", "orphan_controllers",
        ):
            report[key] = summarize([int(row.get(key, -1)) for row in rows])

        def ready_ok(row: dict[str, Any]) -> bool:
            bodies = int(row["managed_walker_bodies_alive"])
            if bodies <= 0:
                return True
            floor = math.ceil(CONTROLLER_READY_FRACTION_GATE * float(bodies))
            return int(row["live_ready_walker_controllers"]) >= floor

        below = [row for row in rows if not ready_ok(row)]
        # Terminal count: verified live + attached + ready.
        report["ready_controllers_min"] = min(
            int(row["live_ready_walker_controllers"]) for row in rows)
        # Diagnosis only, never a gate.
        report["controllers_marked_ready_min"] = min(
            int(row["controllers_marked_ready"]) for row in rows)
        report["live_attached_controllers_min"] = min(
            int(row["live_attached_walker_controllers"]) for row in rows)
        report["frames_below_ready_floor"] = len(below)
        report["first_frame_below_ready_floor"] = below[0]["frame_id"] if below else None

        spans: list[dict[str, Any]] = []
        open_span: dict[str, Any] | None = None
        for row in rows:
            if not ready_ok(row):
                if open_span is None:
                    open_span = {"start_s": row["timestamp_s"], "end_s": row["timestamp_s"],
                                 "start_frame_id": row["frame_id"], "end_frame_id": row["frame_id"]}
                else:
                    open_span["end_s"] = row["timestamp_s"]
                    open_span["end_frame_id"] = row["frame_id"]
            elif open_span is not None:
                spans.append(open_span)
                open_span = None
        if open_span is not None:
            spans.append(open_span)
        for span in spans:
            span["duration_s"] = round(float(span["end_s"]) - float(span["start_s"]), 3)
        report["deficit_spans"] = spans
        report["max_controller_deficit_span_s"] = max(
            (span["duration_s"] for span in spans), default=0.0)
        # Diagnostic, not a gate: a replaced walker needs separate ticks for body
        # spawn, controller attach and controller start, so instantaneous
        # readiness contradicts the crash-safe lifecycle. Prolonged loss is
        # caught by no_controller_deficit_beyond_replenish_plus_2s.
        report["controllers_ready_95pct_every_saved_frame_diagnostic"] = (
            report["frames_below_ready_floor"] == 0)
        return report

    def _incident_report(self, replenish_interval_s: float) -> dict[str, Any]:
        """Collision incident windows, written per sample. No frame is deleted."""
        rows = list(getattr(self.collisions, "rows", []) or [])
        events: list[dict[str, Any]] = []
        for row in rows:
            timestamp = row.get("simulation_time_s")
            if timestamp is None:
                continue
            events.append({
                "simulation_time_s": float(timestamp),
                "other_actor_id": row.get("other_actor_id"),
                "other_actor": row.get("other_actor") or row.get("other_actor_type"),
            })
        incident_path = self.output_dir / "collision_incident_windows.csv"
        with incident_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["sample_id", "frame_id", "timestamp_s",
                            "incident_window", "nearest_collision_dt_s"],
            )
            writer.writeheader()
            flagged = 0
            for row in self.sample_stats:
                deltas = [abs(float(row["timestamp_s"]) - event["simulation_time_s"]) for event in events]
                nearest = min(deltas) if deltas else None
                in_window = nearest is not None and nearest <= COLLISION_INCIDENT_WINDOW_S
                flagged += int(in_window)
                writer.writerow({
                    "sample_id": row["sample_id"],
                    "frame_id": row["frame_id"],
                    "timestamp_s": round(float(row["timestamp_s"]), 6),
                    "incident_window": int(in_window),
                    "nearest_collision_dt_s": "" if nearest is None else round(float(nearest), 4),
                })
        return {
            "window_half_width_s": COLLISION_INCIDENT_WINDOW_S,
            "collision_events": events,
            "collision_count": len(events),
            "frames_in_incident_window": flagged,
            "frames_total": len(self.sample_stats),
            "artifact": str(incident_path),
            "policy": (
                "primary metrics are reported on all frames; the flag supports a "
                "sensitivity result that excludes +/-2 s around a collision"
            ),
            "replenish_interval_s": float(replenish_interval_s),
        }

    def write_summary(
        self,
        route_result: dict[str, Any] | None,
        error: str = "",
        *,
        replenish_interval_s: float = 20.0,
    ) -> None:
        saved_timestamps = [float(row["timestamp_s"]) for row in self.sample_stats]
        prepared_timestamps = [float(row["timestamp_s"]) for row in self.prepared_records]
        sweeps = [row for row in self.aggregator.all_sweep_reports() if row.get("present")]
        sweep_by_index = {row["sweep_index"]: row for row in sweeps}
        consumed = sorted({index for row in self.prepared_records for index in row["window_sweeps"]})
        consumed_sweeps = [sweep_by_index[index] for index in consumed if index in sweep_by_index]

        population = self._population_report()
        incidents = self._incident_report(replenish_interval_s)
        provenance_path = self.output_dir / "frame_runtime_provenance.csv"
        provenance_fields = [
            "sample_id", "frame_id", "prepared_timestamp_s", "ego_speed_mps",
            "ego_velocity_x_mps", "ego_velocity_y_mps", "ego_velocity_z_mps",
            "radar_window_returns", "sweep_index",
        ]
        with provenance_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=provenance_fields)
            writer.writeheader()
            writer.writerows(self.runtime_provenance_rows)
        sample_bytes = [int(row["sample_bytes"]) for row in self.sample_stats]
        route_completed = bool((route_result or {}).get("completed"))
        deficit_limit_s = float(replenish_interval_s) + 2.0
        controller_health = self._controller_health_report(deficit_limit_s)

        cadence = {
            "world_tick": cadence_stats(self.plan.tick_timestamps_s),
            "prepared_inputs": cadence_stats(prepared_timestamps),
            "saved_frames": cadence_stats(saved_timestamps),
            "logical_sweeps": cadence_stats([row["bin_end_s"] for row in consumed_sweeps]),
            "expected_callbacks_per_sweep": self.aggregator.expected_callbacks_per_sweep,
            "callbacks_per_sweep": summarize([row["callbacks"] for row in consumed_sweeps]),
            "returns_per_sweep": summarize([row["returns"] for row in consumed_sweeps]),
            "returns_per_raw_callback": summarize(self.aggregator.returns_per_callback),
            "raw_callbacks": self.aggregator.raw_callbacks,
            "duplicate_callbacks": self.aggregator.duplicate_callbacks,
            "out_of_order_callbacks": self.aggregator.out_of_order_callbacks,
            "dropped_callback_frames": self.aggregator.dropped_callback_frames,
            "timestamp_reversals": self.aggregator.timestamp_reversals,
            "window_callbacks": summarize([row["window_callbacks"] for row in self.prepared_records]),
            "window_returns": summarize([row["window_returns"] for row in self.prepared_records]),
            "window_span_s": summarize([
                row["window_span_s"] for row in self.prepared_records
                if row["window_span_s"] is not None
            ]),
            "warmup_prepared_skipped": self.plan.warmup_prepared_skipped,
            "prepare_wall_clock_s": summarize(self.prepare_wall_s),
        }

        def hz_ok(observed: Any, target: float) -> bool:
            return observed is not None and abs(float(observed) - target) <= 0.02 * target

        gates = {
            "route_completed": route_completed,
            "no_watchdog_abort": not bool((route_result or {}).get("watchdog_aborted")),
            "exact_population_at_start": bool(population["exact_at_start"]),
            "population_alive_95pct_every_saved_frame": all(
                population[key]["frames_below_floor"] == 0
                for key in ("npc_vehicles_alive", "npc_pedestrians_alive")
            ),
            "no_population_deficit_beyond_replenish_plus_2s":
                population["max_deficit_span_s"] <= deficit_limit_s,
            "no_controller_deficit_beyond_replenish_plus_2s":
                controller_health["max_controller_deficit_span_s"] <= deficit_limit_s,
            "zero_missing_or_corrupt_records": (
                self.saved > 0
                and len(self.sample_stats) == self.saved
                and not self.frame_content_failures
            ),
            "rgb_and_segmentation_frames_valid": (
                self.frame_content_checks > 0 and not self.frame_content_failures
            ),
            "sensor_frames_exactly_aligned": self.max_timestamp_delta_s <= 1e-4,
            "world_tick_20hz": hz_ok(cadence["world_tick"]["hz"], WORLD_TICK_HZ),
            "logical_sweeps_10hz": hz_ok(cadence["logical_sweeps"]["hz"], 10.0),
            "prepared_inputs_10hz": hz_ok(cadence["prepared_inputs"]["hz"], 10.0),
            "saved_frames_5hz": hz_ok(cadence["saved_frames"]["hz"], 5.0),
            "callbacks_per_sweep_exact": bool(consumed_sweeps) and all(
                row["callbacks"] == self.aggregator.expected_callbacks_per_sweep
                for row in consumed_sweeps
            ),
            "window_callbacks_exact": bool(self.prepared_records) and all(
                row["window_callbacks"] == self.aggregator.expected_window_callbacks
                for row in self.prepared_records
            ),
            "consecutive_logical_sweeps": not self.cadence_failures,
            "no_dropped_duplicate_or_reordered_callbacks": (
                self.aggregator.dropped_callback_frames == 0
                and self.aggregator.duplicate_callbacks == 0
                and self.aggregator.out_of_order_callbacks == 0
                and self.aggregator.timestamp_reversals == 0
            ),
            "pps_not_inflated":
                int(float(self.radar_attributes["points_per_second"])) == RADAR_POINTS_PER_SECOND,
            "epic_quality_confirmed": (
                str(self.render_provenance["launch"].get("quality_level", "")).lower() == "epic"
                and not self.render_provenance["no_rendering_mode"]
                and not self.render_provenance["launch"]["render_disabling_flags"]
            ),
            "sensor_cleanup_succeeded": bool(self.cleanup_succeeded),
            "no_collector_error": not error,
        }

        summary = {
            "schema": "scenesense_moving_ego_fusion_training_data.v2.route_summary",
            "density": self.density,
            "split": self.split,
            "seed_bundle": self.seed_bundle,
            "scenario_seed": self.scenario_seed,
            "traffic_manager_seed": self.tm_seed,
            "target_speed_kph": self.target_speed_kph,
            "hybrid_physics": self.hybrid_physics,
            "saved_samples": self.saved,
            "prepared_inputs": self.plan.prepared,
            "render_provenance": self.render_provenance,
            "rasterizer": {
                "name": self.args.radar_rasterizer,
                "frozen_for_campaign": True,
                "tolerance": (
                    FAST_RASTERIZER_TOLERANCE if self.args.radar_rasterizer == "fast" else None
                ),
            },
            "cadence": cadence,
            "population": population,
            "controller_health": controller_health,
            "per_frame_controller_health": self.controller_health_samples,
            "incidents": incidents,
            "runtime_provenance": {
                "artifact": str(provenance_path),
                "rows": len(self.runtime_provenance_rows),
                "per_frame_fields": provenance_fields,
                "per_return_fields_in_radar_points_npz": [
                    "original_range_m", "original_azimuth_rad", "original_altitude_rad",
                    "radial_velocity_mps", "observation_age_s", "sweep_offset",
                ],
                "retained_existing_radar_points_fields": [
                    "world_xyz", "camera_xyz", "velocity_mps", "u", "v",
                    "camera_depth_m", "stationary_age_s", "valid_projection",
                ],
                "sweep_offset_semantics": "0 = current logical sweep, 1 = immediately previous",
                "observation_age_semantics":
                    "prepared-frame radar timestamp minus this return's callback timestamp, seconds",
                "original_spherical_semantics":
                    "as measured by CARLA in the callback's own sensor frame, before motion "
                    "compensation; world_xyz/camera_xyz/u/v remain motion compensated to the "
                    "prepared-frame radar pose",
                "radial_velocity_sign_convention": RADAR_VELOCITY_SIGN_CONVENTION,
                "not_included_by_design": [
                    "no agent reducer", "no forward-angle thresholds",
                    "no radar-to-GT association", "no actor IDs",
                ],
            },
            "storage": {
                "saved_bytes": int(self.saved_bytes),
                "saved_gib": round(self.saved_bytes / (1024 ** 3), 3),
                "bytes_per_sample": summarize(sample_bytes),
                "per_frame_bytes_note": "rgb jpg + mask png + semantic png + radar tensor npy + radar points npz",
            },
            "sensor_alignment": {
                "max_timestamp_delta_s": self.max_timestamp_delta_s,
                "max_camera_transform_delta_m": self.max_camera_transform_delta_m,
                "max_radar_transform_delta_m": self.max_radar_transform_delta_m,
                "camera_frame_parity": self.plan.camera_frame_parity,
                "frame_content_checks": self.frame_content_checks,
                "frame_content_failures": self.frame_content_failures,
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
            "walker_brake_distance_m": QUALIFIED_WALKER_BRAKE_DISTANCE_M,
            "sensor_cleanup_succeeded": self.cleanup_succeeded,
            "sensor_cleanup": {
                "succeeded": self.cleanup_succeeded,
                "cleanup_tick": self.cleanup_tick,
                "warnings": self.cleanup_warnings,
                "sensors": self.cleanup_records,
            },
            "gates": gates,
            "status": "COLLECTION_EPISODE_PASSED" if all(gates.values()) else "COLLECTION_EPISODE_FAILED",
            "error": error,
            "route_result": route_result,
        }
        self.parked.save_json(self.output_dir / "route_summary.json", summary)
        failed = sorted(name for name, ok in gates.items() if not ok)
        print(
            f"episode status={summary['status']} saved={self.saved} prepared={self.plan.prepared} "
            f"bytes={self.saved_bytes} failed_gates={failed}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", required=True, choices=tuple(DENSITIES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--route-progress-csv", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--seed-bundle", type=int, choices=sorted(SEED_BUNDLES),
                        help="registered pilot bundle: 1=train, 2=val, 3=locked test, 4=extra train")
    parser.add_argument("--scenario-seed", type=int, default=101)
    parser.add_argument("--tm-seed", type=int, default=1101)
    parser.add_argument("--split", default=None,
                        choices=("train", "val", "test", "smoke"),
                        help="episode split label; implied by --seed-bundle")
    parser.add_argument("--target-speed-kph", type=float, default=DEFAULT_TARGET_SPEED_KPH)
    parser.add_argument("--hybrid-physics", dest="hybrid_physics", action="store_true",
                        help="restore the density runner's stock hybrid physics")
    parser.add_argument("--no-hybrid-physics", dest="hybrid_physics", action="store_false",
                        help="keep full physics for every NPC (pilot default)")
    parser.set_defaults(hybrid_physics=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    # Forwarded verbatim to run_route_b_density_loop.py, whose schedule reads it
    # as CARLA simulated seconds. Default preserved for backward compatibility.
    parser.add_argument("--replenish-interval-s", type=float, default=5.0,
                        help="population reconciliation interval in CARLA simulated "
                             "seconds (default: %(default)s)")
    parser.add_argument("--spectator", dest="spectator", action="store_true",
                        help="chase the ego with the CARLA spectator camera; for watching a "
                             "manual validation run, not for headless collection")
    parser.add_argument("--no-spectator", dest="spectator", action="store_false",
                        help="default: leave the spectator alone")
    parser.set_defaults(spectator=False)
    parser.add_argument("--rasterizer", choices=RASTERIZER_CHOICES, default=DEFAULT_RASTERIZER,
                        help="radar rasterizer; the canonical campaign is frozen on 'fast'")
    parser.add_argument("--maximum-loop-sim-s", type=float, default=900.0,
                        help="density-runner simulated budget passthrough; a small value runs a "
                             "plumbing-only integration check that cannot pass the episode gates")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def resolve_bundle(args: argparse.Namespace) -> None:
    """Apply a registered seed bundle, or label a bare-seed run as a smoke."""
    if args.seed_bundle is not None:
        bundle = SEED_BUNDLES[int(args.seed_bundle)]
        args.scenario_seed = bundle["scenario_seed"]
        args.tm_seed = bundle["tm_seed"]
        if args.split is None:
            args.split = bundle["split"]
        elif args.split != bundle["split"]:
            raise PilotError(
                f"seed bundle {args.seed_bundle} is the {bundle['split']!r} split, "
                f"not {args.split!r}"
            )
    elif args.split is None:
        args.split = "smoke"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        resolve_bundle(args)
        # Fail fast on the renderer before loading a world or spawning anything.
        launch = inspect_launch(int(args.port))
        assert_epic_rendering({"launch": launch, "no_rendering_mode": False})
        preflight = verify_inputs(args)
        preflight["launch"] = launch
        preflight["split"] = args.split
        preflight["seed_bundle"] = args.seed_bundle
        preflight["rasterizer"] = args.rasterizer
        preflight["spectator"] = bool(args.spectator)
        preflight["replenish_interval_s"] = float(args.replenish_interval_s)
    except (PilotError, RenderProvenanceError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Route B perception preflight failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps({"preflight": "PASS", **preflight}, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0

    if (int(args.scenario_seed), int(args.tm_seed)) not in ALLOWED_SEED_BUNDLES:
        allowed = ", ".join(f"{a}/{b}" for a, b in sorted(ALLOWED_SEED_BUNDLES))
        print(f"pilot requires one of the registered seed bundles: {allowed}", file=sys.stderr)
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
        "--target-speed-kph", str(float(args.target_speed_kph)),
        "--walker-brake-distance-m", str(QUALIFIED_WALKER_BRAKE_DISTANCE_M),
        "--fixed-delta-seconds", "0.05",
        "--maximum-loop-sim-s", str(float(args.maximum_loop_sim_s)),
        "--replenish-interval-s", str(float(args.replenish_interval_s)),
        "--real-time-tick-period-s", "0.05",
        "--spectator" if args.spectator else "--no-spectator",
        "--out-csv", str(output_dir / "route_metrics.csv"),
        "--summary-json", str(output_dir / "route_metrics_summary.json"),
    ]
    if not args.hybrid_physics:
        density_argv.append("--no-hybrid-physics")
    density_args = density.build_parser().parse_args(density_argv)

    real_client_class = density.carla.Client
    # Read-only companion connection used for renderer/version provenance only.
    provenance_client = real_client_class(args.host, args.port)
    provenance_client.set_timeout(30.0)
    density.carla.Client = lambda *a, **kw: ClientProxy(real_client_class, args.tm_seed, *a, **kw)
    original_drive = density.drive_one_loop_with_traffic
    collector_holder: dict[str, PerceptionCollectorV2] = {}

    def collecting_drive(
        world: Any, vehicle: Any, agent: Any, route: dict[str, Any], collisions: Any,
        run_args: argparse.Namespace, loop_index: int, maintain: Any, janitor: Any,
    ) -> dict[str, Any]:
        collector = PerceptionCollectorV2(
            parked=parked,
            world=world,
            client=provenance_client,
            ego=vehicle,
            collisions=collisions,
            rpc_port=int(args.port),
            split=str(args.split),
            seed_bundle=args.seed_bundle,
            rasterizer=str(args.rasterizer),
            output_dir=output_dir,
            density=args.density,
            vehicles=vehicles,
            pedestrians=pedestrians,
            scenario_seed=args.scenario_seed,
            tm_seed=args.tm_seed,
            target_speed_kph=args.target_speed_kph,
            hybrid_physics=args.hybrid_physics,
            route_path=Path(args.route_config),
            progress_path=Path(args.route_progress_csv),
            population=getattr(maintain, "population", None),
        )
        collector_holder["collector"] = collector
        result: dict[str, Any] | None = None
        failure = ""
        try:
            result = original_drive(
                SamplingWorld(world, collector, getattr(maintain, "population", None)),
                vehicle, agent, route, collisions,
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
            collector.write_summary(
                result, failure,
                replenish_interval_s=float(density_args.replenish_interval_s),
            )

    density.drive_one_loop_with_traffic = collecting_drive
    try:
        route_status = density.run(density_args)
    except (PilotError, RadarSweepError, RenderProvenanceError, density.RouteBError, RuntimeError) as exc:
        print(
            f"Route B perception episode failed [{type(exc).__name__}]: {exc}",
            file=sys.stderr, flush=True,
        )
        return 2
    else:
        collector = collector_holder.get("collector")
        summary_path = output_dir / "route_summary.json"
        episode_status = "COLLECTION_EPISODE_FAILED"
        if collector is not None and summary_path.is_file():
            episode_status = json.loads(summary_path.read_text(encoding="utf-8")).get("status", episode_status)
        print(
            json.dumps({
                "route_runner_exit": route_status,
                "episode_status": episode_status,
                "summary": str(summary_path),
            }, indent=2),
            flush=True,
        )
        return 0 if (route_status == 0 and episode_status == "COLLECTION_EPISODE_PASSED") else 1
    finally:
        density.drive_one_loop_with_traffic = original_drive
        density.carla.Client = real_client_class


if __name__ == "__main__":
    raise SystemExit(main())
