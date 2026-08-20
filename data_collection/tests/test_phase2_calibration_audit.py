from __future__ import annotations

import copy
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import carla
import pandas as pd

from data_collection.phase2_causal_runtime import (
    Phase2CaptureRuntime,
    Phase2RuntimeConfig,
)
from data_collection.phase2_paired_causal_collector import (
    _capture_checkpoint_identity,
    _load_retention_config,
)
from data_collection.phase2_calibration_scenario import (
    CalibrationScenarioRuntime,
    ResolvedScenario,
)
from data_collection.run_phase2_calibration_audit import (
    DEFAULT_CONFIG,
    _ambient_counts,
    _audit_record,
    _compare_ambient_initial_signatures,
    _compare_ambient_trajectories,
    _load_config,
    _load_world_with_retry,
    _population_command,
    _require_population_process_alive,
    _role_metrics_csv,
    _scenario_owned_nontreatment_signature,
    _select_trajectory_ids,
    _stage_heavy_bytes,
    _traffic_monitor_integration,
    _validate_replay_grid,
    _validate_population_ready_manifest,
    build_plan,
)
from data_collection.run_advisor_generate_traffic import (
    READY_SCHEMA,
    _RetryingClientProxy,
    _nearest_unique_spawn_assignments,
    _pairwise_spaced_spawn_transforms,
    _registered_spawn_pose_errors,
    _require_registered_spawn_pose,
    _route_distance_and_heading_error_deg,
    _route_derived_spawn_transforms,
    _settled_spawn_transform,
    _wait_for_external_tick_with_retry,
)
from data_collection.run_advisor_policy_corpus import (
    TrafficSanityMonitor,
    _traffic_sanity_summary,
)
from data_collection.run_phase2_traffic_preflight import _lane_motion_audit, _wait_ready
from phase2_map_sharing.causal_contract import (
    CausalAuditWriter,
    CausalDecisionAudit,
    CausalField,
    DecisionRecord,
)


class _Location:
    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def distance(self, other: object) -> float:
        return ((self.x - float(other.x)) ** 2 + (self.y - float(other.y)) ** 2) ** 0.5


class _DeferredTransformActor:
    """Minimal actor reproducing CARLA's transform-at-next-tick behavior."""

    def __init__(self, actor_id: int, transform: carla.Transform) -> None:
        self.id = int(actor_id)
        self._transform = transform
        self._pending_transform = None
        self.autopilot = None
        self.physics = None

    def set_autopilot(self, enabled: bool, tm_port: int) -> None:
        self.autopilot = (bool(enabled), int(tm_port))

    def set_simulate_physics(self, enabled: bool) -> None:
        self.physics = bool(enabled)

    def set_transform(self, transform: carla.Transform) -> None:
        self._pending_transform = transform

    def set_target_velocity(self, _velocity: carla.Vector3D) -> None:
        return None

    def set_target_angular_velocity(self, _velocity: carla.Vector3D) -> None:
        return None

    def get_transform(self) -> carla.Transform:
        return self._transform

    def commit(self) -> None:
        if self._pending_transform is not None:
            self._transform = self._pending_transform
            self._pending_transform = None


class _PlacementWorld:
    def __init__(self, actors: list[_DeferredTransformActor]) -> None:
        self.actors = actors
        self.tick_count = 0

    def tick(self, _timeout: float) -> int:
        self.tick_count += 1
        for actor in self.actors:
            actor.commit()
        return 1000 + self.tick_count


class _RouteProjectionMap:
    def get_waypoint(self, location: object, **_kwargs: object) -> object:
        return SimpleNamespace(
            transform=carla.Transform(
                carla.Location(x=float(location.x), y=float(location.y), z=0.0),
                carla.Rotation(yaw=0.0),
            )
        )


class _RouteProjectionWorld:
    def get_map(self) -> _RouteProjectionMap:
        return _RouteProjectionMap()


class _TraceActor:
    def __init__(self, actor_id: int, x: float) -> None:
        self.id = int(actor_id)
        self._transform = carla.Transform(
            carla.Location(x=float(x), y=0.0, z=0.4),
            carla.Rotation(yaw=0.0),
        )

    def get_transform(self) -> carla.Transform:
        return self._transform

    def get_velocity(self) -> carla.Vector3D:
        return carla.Vector3D(x=2.0, y=0.0, z=0.0)


class _DiagnosticController:
    def __init__(self, last_yield: object) -> None:
        self.last_yield = last_yield
        self.tick_count = 0
        self.finished = False

    def tick(self) -> None:
        self.tick_count += 1


class Phase2CalibrationAuditTests(unittest.TestCase):
    def test_runtime_records_per_frame_direct_route_yield_and_first_event(self) -> None:
        actors = {
            "helper": _TraceActor(1, 0.0),
            "recipient": _TraceActor(2, -8.0),
        }
        scenario = SimpleNamespace(
            geometry_or_route_id="test_geometry",
            layout="curbside_opposite",
            scenario_role="controlled_positive_occlusion",
            hazard_present=True,
            lane_contract={"pass": True},
        )
        runtime = CalibrationScenarioRuntime(
            SimpleNamespace(), scenario, actors, tm_port=8010
        )
        helper_yield = {
            "actor_id": 91,
            "type_id": "walker.pedestrian.0001",
            "forward_m": 8.5,
            "lateral_m": 1.1,
            "predicted_lateral_m": 0.4,
            "prediction_horizon_s": 1.2,
            "stopping_m": 12.0,
            "lateral_limit_m": 1.8,
        }
        runtime.controllers = {
            "helper": _DiagnosticController(helper_yield),
            "recipient": _DiagnosticController(None),
        }

        runtime.before_tick(0.0)
        runtime.after_tick(frame_id=100, elapsed_s=0.1)

        first_row = runtime.trace[0]
        self.assertEqual(1, first_row["helper_direct_route_yield_active"])
        self.assertEqual(91, first_row["helper_direct_route_yield_actor_id"])
        self.assertEqual(
            "walker.pedestrian.0001",
            first_row["helper_direct_route_yield_actor_type"],
        )
        self.assertAlmostEqual(
            8.5, first_row["helper_direct_route_yield_forward_m"]
        )
        self.assertAlmostEqual(
            1.1, first_row["helper_direct_route_yield_lateral_m"]
        )
        self.assertAlmostEqual(
            12.0, first_row["helper_direct_route_yield_stopping_m"]
        )
        self.assertEqual(0, first_row["recipient_direct_route_yield_active"])
        self.assertEqual("", first_row["recipient_direct_route_yield_actor_id"])

        runtime.controllers["helper"].last_yield = {
            **helper_yield,
            "actor_id": 92,
        }
        runtime.before_tick(0.1)
        runtime.after_tick(frame_id=101, elapsed_s=0.2)
        summary = runtime.summary()

        self.assertEqual(
            {
                "frame_id": 100,
                "elapsed_s": 0.1,
                "actor_id": 91,
                "actor_type": "walker.pedestrian.0001",
            },
            summary["first_direct_route_yield_by_role"]["helper"],
        )
        self.assertIsNone(
            summary["first_direct_route_yield_by_role"]["recipient"]
        )
        self.assertEqual(
            {"helper": True, "recipient": False},
            summary["direct_route_yield_ever_by_role"],
        )

    def test_runtime_does_not_report_stale_yield_when_controller_did_not_tick(
        self,
    ) -> None:
        actors = {
            "helper": _TraceActor(1, 0.0),
            "recipient": _TraceActor(2, -8.0),
        }
        scenario = SimpleNamespace(
            geometry_or_route_id="test_geometry",
            layout="curbside_opposite",
            scenario_role="matched_benign_control",
            hazard_present=False,
            lane_contract={"pass": True},
        )
        runtime = CalibrationScenarioRuntime(
            SimpleNamespace(), scenario, actors, tm_port=8010
        )
        runtime.controllers = {
            "helper": _DiagnosticController(
                {
                    "actor_id": 91,
                    "type_id": "vehicle.example",
                    "forward_m": 5.0,
                    "lateral_m": 0.0,
                    "predicted_lateral_m": 0.0,
                    "prediction_horizon_s": 0.0,
                    "stopping_m": 7.0,
                    "lateral_limit_m": 2.0,
                }
            )
        }

        runtime.after_tick(frame_id=100, elapsed_s=0.1)

        self.assertEqual(
            0, runtime.trace[0]["helper_direct_route_yield_active"]
        )
        runtime.controllers["helper"].finished = True
        runtime.before_tick(0.1)
        runtime.after_tick(frame_id=101, elapsed_s=0.2)
        self.assertEqual(
            0, runtime.trace[1]["helper_direct_route_yield_active"]
        )
        self.assertIsNone(
            runtime.summary()["first_direct_route_yield_by_role"]["helper"]
        )

    def test_owned_signature_includes_legacy_and_audit_occluder_roles(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)

        def actor(role_name: str, actor_id: int) -> object:
            return SimpleNamespace(
                id=actor_id,
                type_id="vehicle.example",
                attributes={"role_name": role_name},
                get_transform=lambda: carla.Transform(
                    carla.Location(x=float(actor_id), y=0.0, z=0.4),
                    carla.Rotation(yaw=0.0),
                ),
            )

        actors = [
            actor("scenesense_phase2_helper", 1),
            actor("scenesense_phase2_recipient", 2),
            actor("phase2_curbside_occluder", 3),
            actor("phase2_audit_midblock_van_occluder", 4),
            actor("phase2_audit_target_vehicle", 5),
        ]

        class Actors(list):
            def filter(self, _pattern: str) -> list[object]:
                return list(self)

        signature = _scenario_owned_nontreatment_signature(
            SimpleNamespace(get_actors=lambda: Actors(actors)), config
        )
        self.assertEqual(
            {
                "scenesense_phase2_helper",
                "scenesense_phase2_recipient",
                "phase2_curbside_occluder",
                "phase2_audit_midblock_van_occluder",
            },
            {row["role_name"] for row in signature},
        )

    def test_traffic_preflight_lane_audit_rejects_offroad_frame(self) -> None:
        class Map:
            def get_waypoint(self, location, project_to_road, lane_type):
                del lane_type
                if not project_to_road and float(location.x) > 1.0:
                    return None
                return SimpleNamespace(
                    road_id=10,
                    lane_id=2,
                    transform=SimpleNamespace(
                        location=carla.Location(x=float(location.x), y=0.0, z=0.0)
                    ),
                )

        rows = [
            {"actor_id": 7, "world_x": 0.0, "world_y": 0.0, "world_z": 0.0},
            {"actor_id": 7, "world_x": 2.0, "world_y": 0.0, "world_z": 0.0},
        ]
        result = _lane_motion_audit(SimpleNamespace(get_map=Map), rows)
        self.assertFalse(result["pass"])
        self.assertIn("actor_7_left_native_driving_lane", result["failures"])

    def test_world_load_retries_only_bare_carla_transient(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls = 0

            def get_world(self):
                return SimpleNamespace(
                    get_map=lambda: SimpleNamespace(name="Carla/Maps/Town10HD_Opt")
                )

            def reload_world(self, reset):
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("Operation aborted.")
                return ("Town10HD_Opt", reset)

        client = Client()
        self.assertEqual(("Town10HD_Opt", True), _load_world_with_retry(
            client, "Town10HD_Opt", True
        ))
        self.assertEqual(3, client.calls)

        with self.assertRaisesRegex(RuntimeError, "descriptive failure"):
            _load_world_with_retry(
                SimpleNamespace(
                    get_world=lambda: SimpleNamespace(
                        get_map=lambda: SimpleNamespace(name="Carla/Maps/Town10HD_Opt")
                    ),
                    reload_world=lambda _reset: (_ for _ in ()).throw(
                        RuntimeError("descriptive failure")
                    ),
                ),
                "Town10HD_Opt",
                True,
            )
    def test_ego_relocation_waits_for_one_shared_carla_tick(self) -> None:
        staged = carla.Transform(carla.Location(x=4.5, y=-60.9, z=0.4))
        expected = {
            "helper": carla.Transform(
                carla.Location(x=105.0, y=48.0, z=0.4),
                carla.Rotation(yaw=90.0),
            ),
            "recipient": carla.Transform(
                carla.Location(x=98.0, y=44.0, z=0.4),
                carla.Rotation(yaw=90.0),
            ),
        }
        actors = {
            "helper": _DeferredTransformActor(1, staged),
            "recipient": _DeferredTransformActor(2, staged),
        }
        world = _PlacementWorld(list(actors.values()))
        runtime = CalibrationScenarioRuntime(
            world,
            SimpleNamespace(transforms=expected, lane_contract={}),
            actors,
            tm_port=8010,
        )

        realized = runtime.place_egos()

        self.assertEqual(1, world.tick_count)
        self.assertEqual({1001}, {
            value["placement_barrier_frame_id"] for value in realized.values()
        })
        for role, actor in actors.items():
            self.assertLess(actor.get_transform().location.distance(expected[role].location), 1e-9)
            self.assertEqual((False, 8010), actor.autopilot)
            self.assertFalse(actor.physics)
            self.assertTrue(math.isclose(realized[role]["pose_error_m"], 0.0))
            self.assertEqual(1, realized[role]["placement_tick_count"])

    def test_frozen_selector_is_nine_groups_fifteen_trajectories(self) -> None:
        config, _source, selected = _load_config(DEFAULT_CONFIG)
        self.assertEqual(15, len(selected))
        self.assertEqual(9, selected.group_id.nunique())
        self.assertEqual(
            {"inputs_plus_logits_window"}, set(selected.raw_retention_tier)
        )
        self.assertEqual({"calibration"}, set(selected.split))
        self.assertFalse(config["authorization"]["oai_launch"])
        self.assertFalse(config["authorization"]["remaining_calibration"])

    def test_replay_grid_is_frozen_to_72_map_engine_settings(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        replay = config["verification"]["replay_grid"]
        self.assertEqual(72, _validate_replay_grid(replay))
        self.assertEqual(
            0.05, replay["fixed_source_contract"]["detector_confidence_floor"]
        )
        self.assertEqual(
            "fixed_capture_contract_not_replayed_or_tuned",
            replay["fixed_source_contract"]["source_local_tracker"]["status"],
        )
        self.assertEqual(
            [0.05, 0.10, 0.15, 0.20],
            replay["map_engine_axes"]["warning_emission_confidence_floors"],
        )
        self.assertEqual(
            [2.0, 3.0, 4.0],
            replay["map_engine_axes"]["association_base_gates_m"],
        )
        self.assertEqual(
            [0.5, 1.0], replay["map_engine_axes"]["track_ttls_s"]
        )
        self.assertEqual(
            [0.0, 1.0, 2.0],
            replay["map_engine_axes"]["warning_uncertainty_multipliers"],
        )

    def test_replay_grid_rejects_obsolete_source_tracker_sweep(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        replay = copy.deepcopy(config["verification"]["replay_grid"])
        replay["map_engine_axes"]["association_base_gates_m"] = [3.0, 5.0, 7.5]
        replay["expected_combinations"] = 72
        with self.assertRaisesRegex(ValueError, "association_base_gates_m drifted"):
            _validate_replay_grid(replay)

        replay = copy.deepcopy(config["verification"]["replay_grid"])
        replay["fixed_source_contract"]["detector_confidence_floor"] = 0.01
        with self.assertRaisesRegex(ValueError, "must remain fixed at 0.05"):
            _validate_replay_grid(replay)

    def test_replay_grid_rejects_ambiguous_legacy_confidence_key(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        replay = copy.deepcopy(config["verification"]["replay_grid"])
        axes = replay["map_engine_axes"]
        axes["confidence_floors"] = axes.pop(
            "warning_emission_confidence_floors"
        )
        with self.assertRaisesRegex(ValueError, "map_engine_axes keys differ"):
            _validate_replay_grid(replay)

    def test_bounded_regression_selector_is_ordered_and_fail_closed(self) -> None:
        _config, _source, selected = _load_config(DEFAULT_CONFIG)
        requested = [
            "sa_signalized_corner_occluded_pedestrian_low_short_r00_ben",
            "sa_signalized_corner_occluded_pedestrian_low_short_r00_pos",
        ]
        subset = _select_trajectory_ids(selected, requested)
        self.assertEqual(requested, subset["trajectory_id"].astype(str).tolist())
        with self.assertRaisesRegex(ValueError, "unknown calibration trajectory"):
            _select_trajectory_ids(selected, ["missing_trajectory"])

    def test_plan_commands_pin_exact_sensor_and_window_contract(self) -> None:
        config, source, selected = _load_config(DEFAULT_CONFIG)
        plan = build_plan(config, source, selected, Path("/tmp/audit-test"))
        self.assertEqual(15, plan["trajectory_count"])
        self.assertEqual(9, plan["group_count"])
        self.assertFalse(plan["next_stage_chained"])
        for trajectory in plan["trajectories"]:
            if trajectory["scenario_role"] == "naturalistic_operation":
                self.assertEqual(
                    "resolved_after_frozen_geometry_is_loaded",
                    trajectory["population_command_status"],
                )
            else:
                self.assertEqual(
                    "not_launched_scenario_owned_only",
                    trajectory["population_command_status"],
                )
                self.assertEqual(
                    {"vehicles": 0, "walkers": 0, "minimum_walkers_ready": 0},
                    trajectory["ambient_counts"],
                )
            for command in trajectory["collector_commands"].values():
                options = {
                    token: command[index + 1]
                    for index, token in enumerate(command[:-1])
                    if token.startswith("--") and not command[index + 1].startswith("--")
                }
                self.assertEqual("10.0", options["--fps"])
                self.assertEqual("10.0", options["--world-tick-hz"])
                self.assertEqual("1280", options["--camera-width"])
                self.assertEqual("720", options["--camera-height"])
                self.assertEqual("120.0", options["--camera-fov"])
                self.assertEqual("200000", options["--radar-points-per-second"])
                self.assertEqual("4", options["--radar-raster-radius-px"])
                self.assertEqual("2", options["--radar-temporal-window-frames"])
                self.assertEqual("40", options["--phase2-retention-frame-count"])
                self.assertEqual("0.05", options["--object-score-threshold"])
                self.assertEqual(
                    "5.0", options["--phase2-tracker-association-gate-m"]
                )
                self.assertEqual(
                    "3", options["--phase2-tracker-maximum-missed-frames"]
                )

    def test_retention_override_is_hard_and_non_destructive(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        retention = _load_retention_config(Path(config["retention_config"]).resolve())
        self.assertEqual(3_000_000_000, retention["maximum_raw_bytes_per_trajectory"])
        self.assertFalse(retention["allow_automatic_dataset_deletion"])

    def test_ambient_npcs_are_assigned_to_nearest_reviewed_route(self) -> None:
        routes = [
            [_Location(0.0, 0.0), _Location(10.0, 0.0)],
            [_Location(0.0, 20.0), _Location(10.0, 20.0)],
        ]
        selected = TrafficSanityMonitor._nearest_loop_route(
            _Location(5.0, 19.0), routes
        )
        self.assertIs(routes[1], selected)

    def test_npc_route_topology_never_treats_open_polyline_as_loop(self) -> None:
        open_route = [_Location(0.0, 0.0), _Location(50.0, 0.0)]
        closed_route = [
            _Location(0.0, 0.0),
            _Location(50.0, 0.0),
            _Location(4.0, 0.0),
        ]
        self.assertFalse(TrafficSanityMonitor._route_is_closed(open_route))
        self.assertTrue(TrafficSanityMonitor._route_is_closed(closed_route))

    def test_trace_replay_interpolates_distance_and_wrapped_yaw(self) -> None:
        transforms = [
            carla.Transform(
                carla.Location(x=0.0, y=0.0, z=0.0),
                carla.Rotation(yaw=179.0),
            ),
            carla.Transform(
                carla.Location(x=10.0, y=0.0, z=2.0),
                carla.Rotation(yaw=-179.0),
            ),
        ]
        sampled = TrafficSanityMonitor._sample_replay_transform(
            transforms, [0.0, 10.0], 5.0
        )
        self.assertAlmostEqual(5.0, float(sampled.location.x))
        self.assertAlmostEqual(1.0, float(sampled.location.z))
        self.assertAlmostEqual(180.0, float(sampled.rotation.yaw))
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            TrafficSanityMonitor._sample_replay_transform(
                transforms, [0.0, 10.0], 10.1
            )

    def test_full_trajectory_gate_is_id_free_and_detects_drift(self) -> None:
        rows = [
            {
                "replay_identity": "vehicle|autopilot|1|2|0",
                "replay_plan_sha256": "a" * 64,
                "replay_tick_index": tick,
                "world_x": float(tick),
                "world_y": 2.0,
                "world_z": 0.0,
                "speed_mps": 5.0,
            }
            for tick in (1, 2)
        ]
        gate = {
            "required_frames_per_actor": 2,
            "maximum_horizontal_error_m": 0.02,
            "maximum_vertical_error_m": 0.02,
            "maximum_speed_error_mps": 0.001,
            "require_identical_replay_plan_sha256": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            left = Path(temporary) / "left.csv"
            right = Path(temporary) / "right.csv"
            pd.DataFrame(rows).assign(actor_id=7).to_csv(left, index=False)
            pd.DataFrame(rows).assign(actor_id=99).to_csv(right, index=False)
            result = _compare_ambient_trajectories(left, right, gate)
            self.assertTrue(result["pass"])
            self.assertEqual(2, result["paired_rows"])

            drifted = [dict(row) for row in rows]
            drifted[-1]["world_x"] += 0.5
            pd.DataFrame(drifted).assign(actor_id=99).to_csv(right, index=False)
            result = _compare_ambient_trajectories(left, right, gate)
            self.assertFalse(result["pass"])
            self.assertIn("horizontal_trajectory_drift", result["failures"])

    def test_empty_ambient_pair_requires_explicit_scenario_owned_contract(self) -> None:
        columns = [
            "replay_identity",
            "replay_plan_sha256",
            "replay_tick_index",
            "world_x",
            "world_y",
            "world_z",
            "speed_mps",
        ]
        gate = {
            "required_frames_per_actor": 120,
            "maximum_horizontal_error_m": 0.02,
            "maximum_vertical_error_m": 0.25,
            "maximum_speed_error_mps": 0.001,
            "require_identical_replay_plan_sha256": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            left = Path(temporary) / "left.csv"
            right = Path(temporary) / "right.csv"
            pd.DataFrame(columns=columns).to_csv(left, index=False)
            pd.DataFrame(columns=columns).to_csv(right, index=False)
            rejected = _compare_ambient_trajectories(left, right, gate)
            self.assertFalse(rejected["pass"])
            self.assertIn(
                "unexpected_empty_ambient_trajectory", rejected["failures"]
            )
            admitted = _compare_ambient_trajectories(
                left, right, gate, allow_declared_both_empty=True
            )
            self.assertTrue(admitted["pass"])
            self.assertEqual(
                "declared_scenario_owned_only_no_generic_ambient_actors",
                admitted["basis"],
            )

            pd.DataFrame(
                [
                    {
                        "replay_identity": "unexpected",
                        "replay_plan_sha256": "a" * 64,
                        "replay_tick_index": 1,
                        "world_x": 0.0,
                        "world_y": 0.0,
                        "world_z": 0.0,
                        "speed_mps": 0.0,
                    }
                ]
            ).to_csv(right, index=False)
            one_sided = _compare_ambient_trajectories(
                left, right, gate, allow_declared_both_empty=True
            )
            self.assertFalse(one_sided["pass"])
            self.assertIn(
                "one_sided_empty_ambient_trajectory", one_sided["failures"]
            )

    def test_spawn_heading_at_open_route_endpoint_uses_preceding_segment(self) -> None:
        transform = SimpleNamespace(
            location=SimpleNamespace(x=50.0, y=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )
        distance_m, heading_error_deg = _route_distance_and_heading_error_deg(
            transform, [(0.0, 0.0), (50.0, 0.0)]
        )
        self.assertEqual(0.0, distance_m)
        self.assertEqual(0.0, heading_error_deg)

    def test_tm_motion_activates_only_after_explicit_barrier_call(self) -> None:
        actor = SimpleNamespace(
            id=7,
            calls=[],
            set_autopilot=lambda enabled, port: actor.calls.append((enabled, port)),
        )
        monitor = TrafficSanityMonitor.__new__(TrafficSanityMonitor)
        monitor._route_mode = "tm_autonomous"
        monitor.actor_ids = [7]
        monitor.integration = {"tm_port": 8010}
        monitor.initial_geometry = {}
        monitor._live_vehicle_map = lambda: {7: actor}
        monitor.activate_vehicle_motion()
        self.assertEqual([(True, 8010)], actor.calls)

    def test_tm_motion_uses_acknowledged_batch_in_live_orchestrator(self) -> None:
        actor = SimpleNamespace(id=7)
        client = SimpleNamespace(
            calls=[],
            apply_batch_sync=lambda commands, do_tick: (
                client.calls.append((list(commands), do_tick))
                or [SimpleNamespace(error="") for _command in commands]
            ),
        )
        monitor = TrafficSanityMonitor.__new__(TrafficSanityMonitor)
        monitor._route_mode = "tm_autonomous"
        monitor.actor_ids = [7]
        monitor.integration = {"tm_port": 8010}
        monitor.initial_geometry = {}
        monitor._live_vehicle_map = lambda: {7: actor}

        monitor.activate_vehicle_motion(client)

        self.assertEqual(1, len(client.calls))
        self.assertFalse(client.calls[0][1])
        self.assertEqual(
            "acknowledged_set_autopilot_batch",
            monitor.initial_geometry["tm_activation"]["basis"],
        )

    def test_stationary_context_skips_only_gridlock_gate(self) -> None:
        trajectories = pd.DataFrame(
            [
                {
                    "frame_id": frame_id,
                    "carla_timestamp": frame_id * 0.1,
                    "actor_id": actor_id,
                    "speed_mps": 0.0,
                }
                for frame_id in range(1, 121)
                for actor_id in (1, 2)
            ]
        )
        gate = {
            "maximum_collision_incidents": 0,
            "minimum_static_collision_horizontal_impulse": 50.0,
            "minimum_actor_observation_fraction": 0.95,
            "minimum_per_actor_frame_observation_fraction": 0.95,
            "maximum_stationary_context_path_distance_m": 0.05,
            "stopped_speed_max_mps": 0.5,
            "gridlock_minimum_npc_count": 2,
            "gridlock_stopped_fraction": 0.75,
            "persistent_gridlock_min_s": 5.0,
        }
        summary = _traffic_sanity_summary(
            trajectories,
            pd.DataFrame(),
            [1, 2],
            gate,
            expected_frame_count=120,
            stationary_context_expected=True,
        )
        self.assertTrue(summary["pass"])
        self.assertFalse(summary["gridlock_gate_applicable"])
        self.assertGreaterEqual(summary["persistent_gridlock_dwell_s"], 5.0)
        self.assertNotIn("persistent_network_gridlock", summary["failures"])

        moved = trajectories.copy()
        moved["world_x"] = [
            frame_id * 0.001
            for frame_id in range(1, 121)
            for _actor_id in (1, 2)
        ]
        moved["world_y"] = 0.0
        moved_summary = _traffic_sanity_summary(
            moved,
            pd.DataFrame(),
            [1, 2],
            gate,
            expected_frame_count=120,
            stationary_context_expected=True,
        )
        self.assertFalse(moved_summary["pass"])
        self.assertIn("stationary_context_moved", moved_summary["failures"])

        collision = pd.DataFrame(
            [
                {
                    "frame_id": 2,
                    "npc_actor_id": 1,
                    "other_actor_id": 99,
                    "other_type_id": "vehicle.example",
                }
            ]
        )
        collision_summary = _traffic_sanity_summary(
            trajectories,
            collision,
            [1, 2],
            gate,
            expected_frame_count=120,
            stationary_context_expected=True,
        )
        self.assertFalse(collision_summary["pass"])
        self.assertIn(
            "owned_actor_collision_incidents_above_gate",
            collision_summary["failures"],
        )

    def test_walker_only_static_context_keeps_collision_and_motion_gates(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        gate = config["ambient_traffic"]["traffic_sanity_gate"]
        walkers = pd.DataFrame(
            [
                {
                    "frame_id": frame_id,
                    "actor_id": actor_id,
                    "world_x": float(actor_id),
                    "world_y": 0.0,
                }
                for frame_id in range(1, 121)
                for actor_id in (21, 22)
            ]
        )
        summary = _traffic_sanity_summary(
            pd.DataFrame(),
            pd.DataFrame(),
            [],
            gate,
            expected_frame_count=120,
            stationary_context_expected=True,
            stationary_context_trajectories=walkers,
        )
        self.assertTrue(summary["pass"])
        self.assertTrue(summary["applicable"])
        self.assertEqual(0, summary["monitored_npc_vehicles"])
        self.assertEqual(
            {"21": 0.0, "22": 0.0},
            summary["stationary_context_path_distance_by_actor_m"],
        )

        collision = pd.DataFrame(
            [
                {
                    "frame_id": 3,
                    "npc_actor_id": 101,
                    "other_actor_id": 21,
                    "other_type_id": "walker.pedestrian.0001",
                    "contact_owner_scope": "owned_scenario_actor",
                }
            ]
        )
        collision_summary = _traffic_sanity_summary(
            pd.DataFrame(),
            collision,
            [],
            gate,
            expected_frame_count=120,
            stationary_context_expected=True,
            stationary_context_trajectories=walkers,
        )
        self.assertFalse(collision_summary["pass"])
        self.assertIn(
            "owned_actor_collision_incidents_above_gate",
            collision_summary["failures"],
        )

    def test_audit_passes_every_reviewed_route_to_npc_controller(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        scenario = SimpleNamespace(
            ambient_route_paths=(Path("/tmp/route_a.csv"), Path("/tmp/route_b.csv"))
        )
        integration = _traffic_monitor_integration(config, scenario)
        self.assertEqual(
            ["/tmp/route_a.csv", "/tmp/route_b.csv"],
            integration["npc_loop_route_progress_csvs"],
        )
        self.assertEqual(6.0, integration["npc_direct_route_speed_mps"])
        self.assertEqual("stationary_context", integration["npc_route_mode"])
        self.assertEqual("designed_frozen", integration["ambient_evidence_layer"])
        self.assertTrue(integration["expected_stationary_context"])
        self.assertEqual(4.0, integration["npc_trace_replay_speed_mps"])
        self.assertEqual(0.1, integration["npc_trace_replay_fixed_delta_seconds"])
        self.assertEqual(80.0, integration["npc_trace_replay_horizon_m"])
        self.assertTrue(integration["external_sync_tick_owner"])
        self.assertEqual(120, integration["traffic_expected_frame_count"])

        midblock = ResolvedScenario(
            geometry_or_route_id="test",
            layout="midblock_van",
            scenario_role="controlled_positive_occlusion",
            hazard_present=True,
            transforms={},
            routes={},
            lane_contract={},
        )
        self.assertEqual(2, len(midblock.ambient_motion_route_paths))
        for path in midblock.ambient_motion_route_paths:
            self.assertIn("_ambient_v1.progress.csv", path.name)
            self.assertEqual(114, len(path.read_text(encoding="utf-8").splitlines()))
        midblock_integration = _traffic_monitor_integration(config, midblock)
        self.assertEqual(
            [str(path) for path in midblock.ambient_motion_route_paths],
            midblock_integration["npc_loop_route_progress_csvs"],
        )

    def test_matched_pair_gate_uses_immutable_xy_yaw_not_physics_settling(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        gate = config["verification"]["matched_pair_initial_realization_gate"]
        left = [
            {
                "type_id": "vehicle.example",
                "role_name": "autopilot",
                "x": 1.0,
                "y": 2.0,
                "z": 0.5461,
                "yaw_deg": 179.98,
            }
        ]
        right = [
            {
                **left[0],
                "z": 0.3944,
                "yaw_deg": -179.98,
            }
        ]
        result = _compare_ambient_initial_signatures(left, right, gate)
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(0.04, result["maximum_observed_yaw_error_deg"], places=6)
        self.assertNotIn("maximum_observed_vertical_settle_error_m", result)

    def test_matched_pair_gate_rejects_horizontal_or_identity_drift(self) -> None:
        config, _source, _selected = _load_config(DEFAULT_CONFIG)
        gate = config["verification"]["matched_pair_initial_realization_gate"]
        reference = [
            {
                "type_id": "walker.pedestrian.0001",
                "role_name": "pedestrian",
                "x": 1.0,
                "y": 2.0,
                "z": 2.1,
                "yaw_deg": 0.0,
            }
        ]
        drifted = [{**reference[0], "x": 1.5}]
        result = _compare_ambient_initial_signatures(reference, drifted, gate)
        self.assertFalse(result["pass"])
        self.assertIn("horizontal_pose_drift", result["failures"])

        different_identity = [{**reference[0], "type_id": "walker.pedestrian.0002"}]
        result = _compare_ambient_initial_signatures(reference, different_identity, gate)
        self.assertFalse(result["pass"])
        self.assertIn("type_role_multiset_mismatch", result["failures"])

        reference[0].update(
            {
                "motion_mode": "walker_ai_destination",
                "motion_speed_mps": 1.4,
                "motion_target_x": 10.0,
                "motion_target_y": 11.0,
                "motion_target_z": 0.2,
            }
        )
        different_motion = [{**reference[0], "motion_target_x": 12.0}]
        result = _compare_ambient_initial_signatures(reference, different_motion, gate)
        self.assertFalse(result["pass"])
        self.assertIn("motion_contract_drift", result["failures"])

    def test_population_command_requires_three_way_held_release_handshake(self) -> None:
        config, _source, selected = _load_config(DEFAULT_CONFIG)
        scenario = SimpleNamespace(
            protected_locations=(),
            ambient_route_paths=(Path("/tmp/ambient_route.csv"),),
        )
        coordination = Path("/tmp/audit-handshake")
        designed_row = next(
            item
            for item in selected.to_dict("records")
            if str(item["scenario_role"]) != "naturalistic_operation"
        )
        with self.assertRaisesRegex(ValueError, "scenario-owned-only"):
            _population_command(config, designed_row, scenario, coordination)
        naturalistic_row = next(
            item
            for item in selected.to_dict("records")
            if str(item["scenario_role"]) == "naturalistic_operation"
        )
        command = _population_command(
            config, naturalistic_row, scenario, coordination
        )
        self.assertEqual(
            str(coordination / "population.ready.json"),
            command[command.index("--population-ready-manifest") + 1],
        )
        self.assertEqual(
            str(coordination / "population.release.json"),
            command[command.index("--population-release-sentinel") + 1],
        )
        self.assertEqual(
            str(coordination / "population.released.json"),
            command[command.index("--population-released-manifest") + 1],
        )
        self.assertIn("--defer-vehicle-control-to-runner", command)
        self.assertEqual(
            "runner_owned_tm_autonomous",
            command[command.index("--released-vehicle-motion-mode") + 1],
        )
        self.assertEqual(
            "walker_ai_destination",
            command[command.index("--released-walker-motion-mode") + 1],
        )
        pair_gate = config["verification"]["matched_pair_initial_realization_gate"]
        self.assertEqual(
            str(pair_gate["maximum_horizontal_error_m"]),
            command[
                command.index("--registered-spawn-maximum-horizontal-error-m") + 1
            ],
        )
        self.assertEqual(
            str(pair_gate["maximum_yaw_error_deg"]),
            command[command.index("--registered-spawn-maximum-yaw-error-deg") + 1],
        )
        naturalistic_command = _population_command(
            config, naturalistic_row, scenario, coordination / "naturalistic"
        )
        self.assertEqual(
            "runner_owned_tm_autonomous",
            naturalistic_command[
                naturalistic_command.index("--released-vehicle-motion-mode") + 1
            ],
        )
        self.assertEqual(
            "walker_ai_destination",
            naturalistic_command[
                naturalistic_command.index("--released-walker-motion-mode") + 1
            ],
        )
        self.assertIn("--route-derived-spawn-fallback", command)
        self.assertIn("--route-derived-spawn-fallback", naturalistic_command)
        self.assertEqual(
            "0.0", command[command.index("--minimum-route-offset-m") + 1]
        )
        self.assertEqual(
            "0.0",
            naturalistic_command[
                naturalistic_command.index("--minimum-route-offset-m") + 1
            ],
        )
        for density, same_spacing, cross_clearance in (
            ("sparse", "12.0", "8.0"),
            ("typical", "9.0", "7.0"),
            ("dense", "7.0", "6.0"),
        ):
            row = {**naturalistic_row, "traffic_density": density}
            density_command = _population_command(
                config, row, scenario, coordination / density
            )
            _layer_id, _layer, counts = _ambient_counts(config, row)
            requested = str(counts["vehicles"])
            self.assertEqual(
                requested,
                density_command[
                    density_command.index("--minimum-filtered-spawn-points") + 1
                ],
            )
            self.assertEqual(
                same_spacing,
                density_command[
                    density_command.index("--route-derived-spawn-spacing-m") + 1
                ],
            )
            self.assertEqual(
                cross_clearance,
                density_command[
                    density_command.index(
                        "--route-derived-cross-route-clearance-m"
                    )
                    + 1
                ],
            )

        for density, expected_vehicles, expected_walkers in (
            ("sparse", "6", "4"),
            ("typical", "10", "8"),
            ("dense", "15", "12"),
        ):
            row = {**naturalistic_row, "traffic_density": density}
            naturalistic_density_command = _population_command(
                config,
                row,
                scenario,
                coordination / f"naturalistic-{density}",
            )
            self.assertEqual(
                expected_vehicles,
                naturalistic_density_command[
                    naturalistic_density_command.index("--number-of-vehicles") + 1
                ],
            )
            self.assertEqual(
                expected_walkers,
                naturalistic_density_command[
                    naturalistic_density_command.index("--number-of-walkers") + 1
                ],
            )

    def test_population_ready_manifest_is_id_free_and_complete(self) -> None:
        signature = [
            {
                "type_id": "vehicle.example",
                "role_name": "autopilot",
                "x": 1.0,
                "y": 2.0,
                "yaw_deg": 3.0,
                "motion_mode": "runner_owned_direct_loop",
                "motion_speed_mps": None,
                "motion_target_x": None,
                "motion_target_y": None,
                "motion_target_z": None,
            },
            {
                "type_id": "walker.pedestrian.0001",
                "role_name": "pedestrian",
                "x": 4.0,
                "y": 5.0,
                "yaw_deg": 0.0,
                "motion_mode": "walker_ai_destination",
                "motion_speed_mps": 1.4,
                "motion_target_x": 6.0,
                "motion_target_y": 7.0,
                "motion_target_z": 0.2,
            },
        ]
        payload = {
            "schema": READY_SCHEMA,
            "status": "held_ready",
            "vehicle_count": 1,
            "walker_count": 1,
            "walker_controller_count": 1,
            "vehicle_spawn_contract": {
                "all_outside_junctions": True,
                "verified_vehicle_count": 1,
            },
            "spawn_signature_basis": (
                "id_free_held_type_role_pose_and_motion_before_any_ambient_motion"
            ),
            "spawn_signature": signature,
        }
        self.assertEqual(
            signature,
            _validate_population_ready_manifest(payload, vehicles=1, walkers=1),
        )
        with self.assertRaisesRegex(RuntimeError, "walker_count mismatch"):
            _validate_population_ready_manifest(payload, vehicles=1, walkers=2)

    def test_vehicle_stabilization_recovers_unique_commanded_transforms(self) -> None:
        candidates = [
            carla.Transform(carla.Location(x=0.0, y=0.0)),
            carla.Transform(carla.Location(x=10.0, y=0.0)),
        ]
        actors = [
            SimpleNamespace(id=20, get_location=lambda: carla.Location(x=9.8, y=0.0)),
            SimpleNamespace(id=10, get_location=lambda: carla.Location(x=0.2, y=0.0)),
        ]
        assignments = _nearest_unique_spawn_assignments(actors, candidates)
        self.assertEqual([10, 20], [actor.id for actor, _transform in assignments])
        self.assertEqual(
            [0.0, 10.0],
            [float(transform.location.x) for _actor, transform in assignments],
        )

    def test_route_derived_prefix_is_balanced_across_parallel_routes(self) -> None:
        routes = [
            [(float(x), 0.0) for x in range(0, 33, 2)],
            [(float(x), 4.0) for x in range(0, 33, 2)],
        ]
        selected = _route_derived_spawn_transforms(
            _RouteProjectionWorld(),
            routes,
            (),
            protected_clearance_m=5.0,
            pairwise_clearance_m=8.0,
            cross_route_clearance_m=5.0,
        )
        first_six_route_rows = [round(float(item.location.y)) for item in selected[:6]]
        self.assertEqual(3, first_six_route_rows.count(0))
        self.assertEqual(3, first_six_route_rows.count(4))

    def test_route_derived_spawns_exclude_junction_and_keep_open_route_tail(self) -> None:
        class ProjectionMap:
            def get_waypoint(self, location: object, **_kwargs: object) -> object:
                x = float(location.x)

                def next_waypoints(distance: float) -> list[object]:
                    return [
                        SimpleNamespace(
                            is_junction=False,
                            transform=carla.Transform(
                                carla.Location(x=x + float(distance), y=0.0),
                                carla.Rotation(yaw=0.0),
                            ),
                        )
                    ]

                return SimpleNamespace(
                    is_junction=math.isclose(x, 4.0),
                    transform=carla.Transform(
                        carla.Location(x=x, y=0.0), carla.Rotation(yaw=0.0)
                    ),
                    next=next_waypoints,
                )

        world = SimpleNamespace(get_map=lambda: ProjectionMap())
        selected = _route_derived_spawn_transforms(
            world,
            [[(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)]],
            (),
            protected_clearance_m=5.0,
            pairwise_clearance_m=4.0,
            cross_route_clearance_m=4.0,
        )
        selected_x = [round(float(item.location.x)) for item in selected]
        self.assertNotIn(4, selected_x)
        self.assertIn(8, selected_x)
        self.assertIn(12, selected_x)

    def test_native_catalog_capacity_applies_advisor_pairwise_clearance(self) -> None:
        candidates = [
            carla.Transform(carla.Location(x=0.0, y=0.0)),
            carla.Transform(carla.Location(x=4.0, y=0.0)),
            carla.Transform(carla.Location(x=16.0, y=0.0)),
        ]
        selected = _pairwise_spaced_spawn_transforms(candidates, clearance_m=12.0)
        self.assertEqual(
            [0.0, 16.0], [float(transform.location.x) for transform in selected]
        )

    def test_preflight_surfaces_population_failure_manifest_before_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "population.ready.json"
            ready.with_name("population.failed.json").write_text(
                json.dumps(
                    {
                        "error": "RuntimeError: vehicles=1/2 candidates=2",
                    }
                ),
                encoding="utf-8",
            )
            process = SimpleNamespace(poll=lambda: None, returncode=None)
            world = SimpleNamespace(
                tick=lambda _timeout: self.fail("failure must surface before ticking")
            )
            with self.assertRaisesRegex(RuntimeError, "vehicles=1/2 candidates=2"):
                _wait_ready(world, process, ready, 60.0)

    def test_settled_spawn_keeps_registered_xy_yaw_and_physical_z(self) -> None:
        commanded = carla.Transform(
            carla.Location(x=1.0, y=2.0, z=0.6),
            carla.Rotation(pitch=0.1, yaw=30.0, roll=0.2),
        )
        settled = carla.Transform(
            carla.Location(x=1.04, y=1.98, z=-0.05),
            carla.Rotation(pitch=1.0, yaw=31.0, roll=2.0),
        )
        frozen = _settled_spawn_transform(commanded, settled)
        self.assertEqual((frozen.location.x, frozen.location.y), (1.0, 2.0))
        self.assertAlmostEqual(float(frozen.location.z), -0.05)
        self.assertAlmostEqual(float(frozen.rotation.yaw), 30.0)

    def test_registered_spawn_pose_error_uses_scientific_pair_tolerance(self) -> None:
        commanded = carla.Transform(
            carla.Location(x=10.0, y=20.0, z=0.6),
            carla.Rotation(yaw=15.0),
        )
        realized = carla.Transform(
            carla.Location(x=10.007335, y=20.0, z=0.1),
            carla.Rotation(yaw=15.024307),
        )
        horizontal_error, yaw_error = _registered_spawn_pose_errors(
            commanded, realized
        )
        self.assertAlmostEqual(0.007335, horizontal_error, places=6)
        self.assertAlmostEqual(0.024307, yaw_error, places=6)
        self.assertLessEqual(horizontal_error, 0.10)
        self.assertLessEqual(yaw_error, 0.10)
        self.assertGreater(yaw_error, 0.02)
        self.assertEqual(
            (horizontal_error, yaw_error),
            _require_registered_spawn_pose(
                actor_id=1964,
                commanded=commanded,
                realized=realized,
                maximum_horizontal_error_m=0.10,
                maximum_yaw_error_deg=0.10,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "yaw_limit=0.020000"):
            _require_registered_spawn_pose(
                actor_id=1964,
                commanded=commanded,
                realized=realized,
                maximum_horizontal_error_m=0.02,
                maximum_yaw_error_deg=0.02,
            )

    def test_external_tick_wait_retries_only_bare_carla_transient(self) -> None:
        class TransientWorld:
            def __init__(self) -> None:
                self.calls = 0

            def wait_for_tick(self) -> None:
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError("std::exception")

        transient = TransientWorld()
        _wait_for_external_tick_with_retry(transient)
        self.assertEqual(3, transient.calls)

        class FatalWorld:
            def wait_for_tick(self) -> None:
                raise RuntimeError("time-out while waiting for the simulator")

        with self.assertRaisesRegex(RuntimeError, "time-out"):
            _wait_for_external_tick_with_retry(FatalWorld())

    def test_retrying_client_covers_advisor_main_loop_world_wait(self) -> None:
        class World:
            marker = "delegated"

            def __init__(self) -> None:
                self.calls = 0

            def wait_for_tick(self, timeout: float) -> str:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("std::exception")
                return f"tick-{timeout}"

        raw_world = World()
        raw_client = SimpleNamespace(get_world=lambda: raw_world, port=2000)
        client = _RetryingClientProxy(raw_client)
        world = client.get_world()
        self.assertEqual("tick-2.0", world.wait_for_tick(2.0))
        self.assertEqual(2, raw_world.calls)
        self.assertEqual("delegated", world.marker)
        self.assertEqual(2000, client.port)

    def test_population_process_exit_is_fatal_during_capture(self) -> None:
        _require_population_process_alive(
            SimpleNamespace(poll=lambda: None), phase="capture frame 1 pre-tick"
        )
        with self.assertRaisesRegex(RuntimeError, "returncode=1"):
            _require_population_process_alive(
                SimpleNamespace(poll=lambda: 1),
                phase="capture frame 1 post-tick",
            )

    def test_spawn_protection_covers_hazard_endpoint_and_conflict_point(self) -> None:
        pedestrian = ResolvedScenario(
            geometry_or_route_id="test",
            layout="signalized_corner",
            scenario_role="controlled_positive_occlusion",
            hazard_present=True,
            transforms={},
            routes={},
            lane_contract={},
        )
        pedestrian_xy = {
            (round(float(item.x), 3), round(float(item.y), 3))
            for item in pedestrian.protected_locations
        }
        self.assertIn((85.7, 32.0), pedestrian_xy)
        self.assertIn((85.7, 22.5), pedestrian_xy)
        self.assertIn((85.7, 27.25), pedestrian_xy)

        vehicle = ResolvedScenario(
            geometry_or_route_id="test",
            layout="parked_vehicle_pullout",
            scenario_role="controlled_positive_occlusion",
            hazard_present=True,
            transforms={},
            routes={},
            lane_contract={
                "registered_conflict_point": {"x": 1.0, "y": 2.0, "z": 3.0}
            },
        )
        vehicle_xyz = {
            (
                round(float(item.x), 3),
                round(float(item.y), 3),
                round(float(item.z), 3),
            )
            for item in vehicle.protected_locations
        }
        self.assertIn((1.0, 2.0, 3.0), vehicle_xyz)

        queue_path = [
            carla.Location(x=-10.0, y=69.73, z=0.0),
            carla.Location(x=-4.0, y=72.5, z=0.0),
            carla.Location(x=2.0, y=72.5, z=0.0),
        ]
        queue = ResolvedScenario(
            geometry_or_route_id="test",
            layout="queue_reveal_vehicle",
            scenario_role="controlled_positive_occlusion",
            hazard_present=True,
            transforms={},
            routes={"occluder": queue_path},
            lane_contract={},
        )
        queue_xy = {
            (round(float(item.x), 3), round(float(item.y), 3))
            for item in queue.protected_locations
        }
        self.assertTrue(
            {
                (-10.0, 69.73),
                (-4.0, 72.5),
                (2.0, 72.5),
            }.issubset(queue_xy)
        )

    def test_npc_tick_callback_error_is_promoted_to_fatal_gate(self) -> None:
        monitor = TrafficSanityMonitor.__new__(TrafficSanityMonitor)
        monitor._lock = threading.Lock()
        monitor._tick_failure = None
        monitor._direct_route_state = {1: {}}
        monitor.actor_ids = []

        def fail_control() -> None:
            raise KeyError("missing_contract_field")

        monitor._apply_direct_route_controls = fail_control
        snapshot = SimpleNamespace(
            frame=1,
            timestamp=SimpleNamespace(elapsed_seconds=0.1),
        )
        monitor._on_tick(snapshot)
        with self.assertRaisesRegex(RuntimeError, "missing_contract_field"):
            monitor.raise_if_failed()

    def test_external_tick_owner_records_every_explicit_snapshot(self) -> None:
        monitor = TrafficSanityMonitor.__new__(TrafficSanityMonitor)
        monitor._lock = threading.Lock()
        monitor._tick_failure = None
        monitor._direct_route_state = {}
        monitor.actor_ids = [7]
        monitor.actor_metadata = {
            7: {"role_name": "autopilot", "type_id": "vehicle.example"}
        }
        monitor.trajectory_rows = []
        live_actor = SimpleNamespace(id=7)
        monitor._live_vehicle_map = lambda: {7: live_actor}
        monitor._infer_registered_hazard_yield_actor_ids = lambda _live: set()
        monitor._registered_hazard_yield_actor_ids = set()
        actor_snapshot = SimpleNamespace(
            get_transform=lambda: SimpleNamespace(
                location=SimpleNamespace(x=1.0, y=2.0, z=0.3)
            ),
            get_velocity=lambda: SimpleNamespace(x=3.0, y=4.0, z=0.0),
        )
        for frame in range(120):
            snapshot = SimpleNamespace(
                frame=frame,
                timestamp=SimpleNamespace(elapsed_seconds=frame / 10.0),
                find=lambda _actor_id, value=actor_snapshot: value,
            )
            monitor.before_world_tick()
            monitor.observe_snapshot(snapshot)
        monitor.raise_if_failed()
        self.assertEqual(120, len(monitor.trajectory_rows))
        self.assertEqual(5.0, monitor.trajectory_rows[-1]["speed_mps"])
        self.assertEqual("world_snapshot", monitor.trajectory_rows[-1]["sample_source"])

    def test_external_tick_owner_falls_back_when_snapshot_omits_live_npc(self) -> None:
        monitor = TrafficSanityMonitor.__new__(TrafficSanityMonitor)
        monitor._lock = threading.Lock()
        monitor._tick_failure = None
        monitor.actor_ids = [9]
        monitor.actor_metadata = {
            9: {"role_name": "autopilot", "type_id": "vehicle.example"}
        }
        monitor.trajectory_rows = []
        live_actor = SimpleNamespace(
            is_alive=True,
            get_transform=lambda: SimpleNamespace(
                location=SimpleNamespace(x=4.0, y=5.0, z=0.2)
            ),
            get_velocity=lambda: SimpleNamespace(x=1.0, y=0.0, z=0.0),
        )
        monitor._live_vehicle_map = lambda: {9: live_actor}
        monitor._infer_registered_hazard_yield_actor_ids = lambda _live: set()
        monitor._registered_hazard_yield_actor_ids = set()
        snapshot = SimpleNamespace(
            frame=77,
            timestamp=SimpleNamespace(elapsed_seconds=7.7),
            find=lambda _actor_id: None,
        )
        monitor.observe_snapshot(snapshot)
        monitor.raise_if_failed()
        self.assertEqual(1, len(monitor.trajectory_rows))
        self.assertEqual("live_actor_fallback", monitor.trajectory_rows[0]["sample_source"])

    def test_external_tick_owner_rejects_destroyed_actor_tombstone(self) -> None:
        monitor = TrafficSanityMonitor.__new__(TrafficSanityMonitor)
        monitor._lock = threading.Lock()
        monitor._tick_failure = None
        monitor.actor_ids = [9]
        monitor.actor_metadata = {
            9: {"role_name": "autopilot", "type_id": "vehicle.example"}
        }
        monitor.trajectory_rows = []
        tombstone = SimpleNamespace(
            is_alive=True,
            get_transform=lambda: SimpleNamespace(
                location=SimpleNamespace(x=0.0, y=0.0, z=0.0)
            ),
            get_velocity=lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0),
        )
        # A cached get_actor proxy can still claim is_alive=True. The actor is
        # absent from the authoritative full world inventory and must be fatal.
        monitor.world = SimpleNamespace(get_actor=lambda _actor_id: tombstone)
        monitor._live_vehicle_map = lambda: {}
        monitor._registered_hazard_yield_actor_ids = set()
        snapshot = SimpleNamespace(
            frame=78,
            timestamp=SimpleNamespace(elapsed_seconds=7.8),
            find=lambda _actor_id: None,
        )
        monitor.observe_snapshot(snapshot)
        with self.assertRaisesRegex(RuntimeError, "membership disappeared"):
            monitor.raise_if_failed()
        self.assertEqual([], monitor.trajectory_rows)

    def test_runtime_selects_offset_then_exact_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = Phase2CaptureRuntime(
                Phase2RuntimeConfig(
                    role="helper",
                    trajectory_id="audit_t0",
                    scenario_role="naturalistic_operation",
                    run_dir=root / "role",
                    ready_sentinel=root / "ready.json",
                    capture_start_sentinel=root / "start.json",
                    tick_ready_path=root / "tick.json",
                    heartbeat_path=root / "heartbeat.json",
                    contract_config_path=root / "contract.yaml",
                    retention_start_offset_s=0.3,
                    retention_frame_count=2,
                ),
                {
                    "maximum_window_seconds_per_trajectory": 4.0,
                    "maximum_raw_bytes_per_trajectory": 2_000_000,
                    "maximum_raw_bytes_pilot_total": 2_000_000,
                    "minimum_free_bytes_after_reservation": 1,
                },
            )
            try:
                runtime._capture_start_clock_s = 10.0
                self.assertFalse(runtime._retain_frame(1, 10.2))
                self.assertTrue(runtime._retain_frame(2, 10.3))
                runtime._retained_input_frames.add(2)
                self.assertTrue(runtime._retain_frame(3, 10.4))
                runtime._retained_input_frames.add(3)
                self.assertFalse(runtime._retain_frame(4, 10.5))
            finally:
                runtime.close(status="complete")

    def test_live_verifier_uses_production_streams_metrics_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            role_dir = Path(temporary) / "helper"
            metrics = role_dir / "streams/fusion_ego_7_metrics.csv"
            metrics.parent.mkdir(parents=True)
            metrics.write_text("frame_id\n1\n", encoding="utf-8")
            self.assertEqual(metrics, _role_metrics_csv(role_dir))

    def test_capture_manifest_authenticates_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "best.pt"
            checkpoint.write_bytes(b"frozen-model-bytes")
            identity = _capture_checkpoint_identity(
                {"checkpoint_path": str(checkpoint)}
            )
            self.assertEqual(str(checkpoint.resolve()), identity["checkpoint_path_at_capture"])
            self.assertEqual(
                "55d15cd099ef5a181b4634f2dc808491963a428a9c0e7119024ae21aea15c663",
                identity["checkpoint_sha256"],
            )
            self.assertEqual(
                "capture_time_file_bytes", identity["checkpoint_identity_basis"]
            )

    def test_capture_manifest_rejects_missing_checkpoint(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "checkpoint is not a file"):
            _capture_checkpoint_identity({"checkpoint_path": "/missing/factor.pt"})

    def test_stage_heavy_bytes_counts_retained_inputs_and_logits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "trajectory/helper/retained_inputs/frame_1_inputs.npz"
            logits = root / "trajectory/helper/retained_logits/frame_1_logits.npz"
            unrelated = root / "trajectory/helper/runtime/metrics.npz"
            for path, payload in (
                (inputs, b"input"),
                (logits, b"logits"),
                (unrelated, b"not-heavy-retention"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            self.assertEqual(len(b"input") + len(b"logits"), _stage_heavy_bytes(root))

    def test_live_verifier_uses_causal_writer_hash_contract(self) -> None:
        decision = DecisionRecord(
            trajectory_id="t0",
            arm_id="selected_runtime",
            decision_id="t0:placement:1",
            decision_stage="placement",
            decision_at_s=1.0,
            clock_id="host_perf_counter",
            action="SPLIT_FEATURE",
        )
        field = CausalField(
            field_name="helper_state",
            value={"frame_id": 1},
            source_stage="helper_localization",
            observed_at_s=1.0,
            available_at_s=1.0,
            consuming_decision_id=decision.decision_id,
            consuming_decision_stage=decision.decision_stage,
            clock_id=decision.clock_id,
            arm_id=decision.arm_id,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            with CausalAuditWriter(path) as writer:
                writer.write(CausalDecisionAudit(decision, (field,)))
            envelope = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(decision, _audit_record(envelope).decision)
            envelope["record_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _audit_record(envelope)


if __name__ == "__main__":
    unittest.main()
