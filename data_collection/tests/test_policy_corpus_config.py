import copy
import csv
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from data_collection import run_advisor_spawn_blocker as blocker_wrapper
from data_collection.run_policy_corpus import (
    _effective_options,
    _load_config,
    _resolved_run_args,
    _validate_collection_contract,
)
from data_collection.rescore_policy_corpus_freshness import (
    _load_yaml as _load_freshness_config,
    _validate_reference,
)
from data_collection.verify_policy_corpus import _longest_true_dwell
from data_collection.run_advisor_policy_corpus import (
    _controlled_pedestrian_gate_rows,
    _exact_fast_scenario_summary,
    _in_forward_corridor,
    _vehicle_envelope_requires_yield,
    _vehicle_requires_yield,
    _walker_requires_yield,
    _load_config as _load_advisor_config,
    _population_commands,
    _pedestrian_motion_summary,
    _radar_density_summary,
    _tick_until_empty,
    _traffic_sanity_summary,
    _validate_advisor_contract,
)
from data_collection.run_advisor_spawn_blocker import _poll_for_tick
from data_collection.replay_on_contract_pedestrian_diagnostic import (
    target_radar_hit_count,
)
from data_collection.author_advisor_demo_route import _PlanningWorld
from data_collection.review_phase2_pair_geometry import (
    CURBSIDE_ROUTE_PROGRESS,
    _settle_parked_occluder,
    load_route_progress,
    opposite_lane_route,
    offset_transform,
)
from data_collection.phase2_curbside_scenario import (
    CURBSIDE_GEOMETRY_ID,
    DirectRouteController,
)
from data_collection.phase2_signalized_corner_scenario import (
    SIGNALIZED_EXPECTED_END_LANES,
    SIGNALIZED_EXPECTED_START_LANES,
    SIGNALIZED_GEOMETRY_ID,
    SIGNALIZED_JUNCTION_ID,
    SIGNALIZED_ROUTE_SHA256,
    _standard_agents_root,
    frozen_routes,
    line_of_sight_bearings_deg,
)
from data_collection.phase2_midblock_van_scenario import (
    MIDBLOCK_EXPECTED_LANES,
    MIDBLOCK_GEOMETRY_ID,
    MIDBLOCK_HELPER_TRANSFORM,
    MIDBLOCK_OCCLUDER_MAX_CENTER_OFFSET_M,
    MIDBLOCK_OCCLUDER_MIN_CENTER_OFFSET_M,
    MIDBLOCK_OCCLUDER_TRANSFORM,
    MIDBLOCK_RECIPIENT_TRANSFORM,
    MIDBLOCK_ROAD_ID,
    MIDBLOCK_ROUTE_SHA256,
    frozen_routes as midblock_frozen_routes,
    line_of_sight_bearings_deg as midblock_line_of_sight_bearings_deg,
)
from data_collection.phase2_cross_traffic_vehicle_scenario import (
    CROSS_TRAFFIC_GEOMETRY_ID,
    CROSS_TRAFFIC_HELPER_TRANSFORM,
    CROSS_TRAFFIC_OCCLUDER_TRANSFORM,
    CROSS_TRAFFIC_RECIPIENT_TRANSFORM,
    CROSS_TRAFFIC_TARGET_TRANSFORM,
    CROSS_TRAFFIC_TARGET_ROUTE_SHA256,
    frozen_routes as cross_traffic_frozen_routes,
    visibility_state as cross_traffic_visibility_state,
)
from data_collection.phase2_parked_vehicle_pullout_scenario import (
    PULLOUT_GEOMETRY_ID,
    PULLOUT_HELPER_TRANSFORM,
    PULLOUT_MERGE_POINT,
    PULLOUT_OCCLUDER_TRANSFORM,
    PULLOUT_RECIPIENT_TRANSFORM,
    PULLOUT_TARGET_ROUTE_VALUES,
    PULLOUT_TARGET_ROUTE_SHA256,
    PULLOUT_TARGET_TRANSFORM,
    frozen_routes as pullout_frozen_routes,
)
from data_collection.phase2_queue_reveal_vehicle_scenario import (
    QUEUE_REVEAL_GEOMETRY_ID,
    QUEUE_REVEAL_HELPER_TRANSFORM,
    QUEUE_REVEAL_OCCLUDER_ROUTE_VALUES,
    QUEUE_REVEAL_OCCLUDER_TRANSFORM,
    QUEUE_REVEAL_RECIPIENT_TRANSFORM,
    QUEUE_REVEAL_TARGET_TRANSFORM,
    frozen_routes as queue_reveal_frozen_routes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "data_collection" / "configs" / "policy_corpus_vehicle_v2.yaml"
ADVISOR_CONFIG_PATH = (
    REPO_ROOT / "data_collection" / "configs" / "policy_corpus_advisor_rich_v3.yaml"
)
ADVISOR_V4_CONFIG_PATH = (
    REPO_ROOT / "data_collection" / "configs" / "policy_corpus_advisor_rich_v4.yaml"
)
ADVISOR_V5_CONFIG_PATH = (
    REPO_ROOT / "data_collection" / "configs" / "policy_corpus_advisor_rich_v5.yaml"
)
ON_CONTRACT_CONFIG_PATH = (
    REPO_ROOT
    / "data_collection"
    / "configs"
    / "pedestrian_on_contract_diagnostic_v1.yaml"
)


class VehicleCorpusConfigTests(unittest.TestCase):
    def test_route_planning_world_does_not_register_unused_tick_callback(self):
        class FakeWorld:
            marker = object()

            def on_tick(self, _callback):
                raise AssertionError("planning-only route author must not subscribe")

        wrapped = FakeWorld()
        planning_world = _PlanningWorld(wrapped)
        self.assertIsNone(planning_world.on_tick(lambda _snapshot: None))
        self.assertIs(planning_world.marker, wrapped.marker)

    def test_phase2_geometry_offset_stays_on_base_heading(self):
        base = __import__("carla").Transform(
            __import__("carla").Location(x=-3.974, y=28.104, z=0.6),
            __import__("carla").Rotation(yaw=0.16),
        )
        helper = offset_transform(base, forward_m=10.0)
        self.assertAlmostEqual(
            math.hypot(
                float(helper.location.x - base.location.x),
                float(helper.location.y - base.location.y),
            ),
            10.0,
            places=5,
        )
        self.assertAlmostEqual(helper.location.x, 6.02596, places=4)
        self.assertAlmostEqual(helper.location.y, 28.13193, places=4)
        self.assertAlmostEqual(helper.location.z, 0.75, places=6)

    def test_phase2_geometry_viewer_uses_accepted_v3_route(self):
        route = load_route_progress(
            REPO_ROOT
            / "data_collection/routes/town10hd_opt_advisor_safe_perimeter_loop_v3.progress.csv"
        )
        self.assertGreater(len(route), 80)
        self.assertAlmostEqual(float(route[0].x), 3.273855, places=5)
        self.assertAlmostEqual(float(route[0].y), 28.193132, places=5)

    def test_phase2_curbside_routes_cross_centerline_and_are_opposite(self):
        recipient = load_route_progress(CURBSIDE_ROUTE_PROGRESS)
        helper = opposite_lane_route(recipient)
        frozen_helper = load_route_progress(
            REPO_ROOT
            / "data_collection/routes/town10hd_opt_curbside_helper_v1.progress.csv"
        )
        self.assertEqual(len(recipient), 28)
        self.assertGreater(len(helper), 10)
        self.assertEqual(len(frozen_helper), len(helper))
        self.assertEqual(CURBSIDE_GEOMETRY_ID, "town10hd_opt_curbside_legal_opposing_v1")
        self.assertAlmostEqual(float(recipient[0].x), 57.500881195, places=6)
        self.assertAlmostEqual(float(helper[0].x), 6.516562939, places=6)
        self.assertAlmostEqual(
            float(helper[-1].y - recipient[0].y), 7.0, places=5
        )
        recipient_dx = float(recipient[1].x - recipient[0].x)
        helper_dx = float(helper[1].x - helper[0].x)
        self.assertLess(recipient_dx, 0.0)
        self.assertGreater(helper_dx, 0.0)
        for generated, frozen in zip(helper, frozen_helper):
            self.assertAlmostEqual(float(generated.x), float(frozen.x), places=6)
            self.assertAlmostEqual(float(generated.y), float(frozen.y), places=6)

    def test_phase2_signalized_frozen_geometry_has_lane_and_camera_contract(self):
        self.assertEqual(
            SIGNALIZED_GEOMETRY_ID,
            "town10hd_opt_signalized_corner_van_crosswalk_v1",
        )
        self.assertEqual(SIGNALIZED_JUNCTION_ID, 532)
        self.assertEqual(SIGNALIZED_EXPECTED_START_LANES["recipient"], (21, -1))
        self.assertEqual(SIGNALIZED_EXPECTED_START_LANES["occluder"], (21, -2))
        self.assertNotEqual(
            SIGNALIZED_EXPECTED_END_LANES["recipient"],
            SIGNALIZED_EXPECTED_END_LANES["helper"],
        )
        self.assertTrue(_standard_agents_root().is_dir())
        routes = frozen_routes()
        self.assertEqual(len(routes["recipient"]), 39)
        self.assertEqual(len(routes["helper"]), 33)
        self.assertEqual(len(SIGNALIZED_ROUTE_SHA256["recipient"]), 64)
        bearings = line_of_sight_bearings_deg()
        self.assertEqual(set(bearings), {"helper", "recipient"})
        self.assertTrue(all(abs(value) < 55.0 for value in bearings.values()))

    def test_phase2_midblock_candidate_has_opposing_lane_and_camera_contract(self):
        self.assertEqual(
            MIDBLOCK_GEOMETRY_ID,
            "town10hd_opt_midblock_curbside_van_v1",
        )
        self.assertEqual(MIDBLOCK_ROAD_ID, 12)
        self.assertEqual(MIDBLOCK_EXPECTED_LANES["recipient"], (12, 1))
        self.assertEqual(MIDBLOCK_EXPECTED_LANES["helper"], (12, -1))
        start_separation_m = math.hypot(
            MIDBLOCK_RECIPIENT_TRANSFORM[0] - MIDBLOCK_HELPER_TRANSFORM[0],
            MIDBLOCK_RECIPIENT_TRANSFORM[1] - MIDBLOCK_HELPER_TRANSFORM[1],
        )
        self.assertGreater(start_separation_m, 40.0)
        self.assertLess(
            MIDBLOCK_RECIPIENT_TRANSFORM[0], MIDBLOCK_OCCLUDER_TRANSFORM[0]
        )
        self.assertLess(
            MIDBLOCK_OCCLUDER_TRANSFORM[0], MIDBLOCK_HELPER_TRANSFORM[0]
        )
        self.assertLess(abs(MIDBLOCK_RECIPIENT_TRANSFORM[3]), 5.0)
        self.assertLess(abs(abs(MIDBLOCK_HELPER_TRANSFORM[3]) - 180.0), 5.0)
        curb_center_offset_m = abs(
            MIDBLOCK_OCCLUDER_TRANSFORM[1] - MIDBLOCK_RECIPIENT_TRANSFORM[1]
        )
        self.assertGreaterEqual(
            curb_center_offset_m, MIDBLOCK_OCCLUDER_MIN_CENTER_OFFSET_M
        )
        self.assertLessEqual(
            curb_center_offset_m, MIDBLOCK_OCCLUDER_MAX_CENTER_OFFSET_M
        )
        bearings = midblock_line_of_sight_bearings_deg()
        self.assertEqual(set(bearings), {"helper", "recipient"})
        self.assertTrue(all(abs(value) < 55.0 for value in bearings.values()))
        routes = midblock_frozen_routes()
        self.assertEqual(len(routes["recipient"]), 33)
        self.assertEqual(len(routes["helper"]), 33)
        self.assertTrue(
            all(len(value) == 64 for value in MIDBLOCK_ROUTE_SHA256.values())
        )

    def test_phase2_cross_traffic_frozen_geometry_has_differential_initial_visibility(self):
        carla = __import__("carla")

        def transform(values):
            return carla.Transform(
                carla.Location(x=values[0], y=values[1], z=values[2]),
                carla.Rotation(yaw=values[3]),
            )

        self.assertEqual(
            CROSS_TRAFFIC_GEOMETRY_ID,
            "town10hd_opt_occluded_cross_traffic_vehicle_v1",
        )
        self.assertEqual(len(CROSS_TRAFFIC_TARGET_ROUTE_SHA256), 64)
        routes = cross_traffic_frozen_routes()
        self.assertEqual(len(routes["recipient"]), 39)
        self.assertEqual(len(routes["helper"]), 33)
        self.assertEqual(len(routes["target"]), 107)
        target = transform(CROSS_TRAFFIC_TARGET_TRANSFORM)
        occluder = transform(CROSS_TRAFFIC_OCCLUDER_TRANSFORM)
        helper = cross_traffic_visibility_state(
            transform(CROSS_TRAFFIC_HELPER_TRANSFORM), target, occluder
        )
        recipient = cross_traffic_visibility_state(
            transform(CROSS_TRAFFIC_RECIPIENT_TRANSFORM), target, occluder
        )
        self.assertTrue(helper["in_fov"])
        self.assertTrue(helper["geometrically_visible"])
        self.assertFalse(helper["occluded_by_controlled_truck"])
        self.assertTrue(recipient["in_fov"])
        self.assertFalse(recipient["geometrically_visible"])
        self.assertTrue(recipient["occluded_by_controlled_truck"])

    def test_phase2_pullout_frozen_geometry_has_explicit_conflict_route(self):
        carla = __import__("carla")

        def transform(values):
            return carla.Transform(
                carla.Location(x=values[0], y=values[1], z=values[2]),
                carla.Rotation(yaw=values[3]),
            )

        self.assertEqual(
            PULLOUT_GEOMETRY_ID,
            "town10hd_opt_parked_vehicle_pullout_v1",
        )
        self.assertEqual(len(PULLOUT_TARGET_ROUTE_SHA256), 64)
        routes = pullout_frozen_routes()
        self.assertEqual(set(routes), {"recipient", "helper", "target"})
        self.assertEqual(len(routes["target"]), len(PULLOUT_TARGET_ROUTE_VALUES))
        self.assertAlmostEqual(
            float(routes["target"][0].x), PULLOUT_TARGET_TRANSFORM[0], places=5
        )
        self.assertAlmostEqual(
            float(routes["target"][0].y), PULLOUT_TARGET_TRANSFORM[1], places=5
        )
        self.assertLessEqual(
            min(
                math.hypot(
                    float(point.x) - PULLOUT_MERGE_POINT[0],
                    float(point.y) - PULLOUT_MERGE_POINT[1],
                )
                for point in routes["target"]
            ),
            0.05,
        )
        target = transform(PULLOUT_TARGET_TRANSFORM)
        occluder = transform(PULLOUT_OCCLUDER_TRANSFORM)
        helper = cross_traffic_visibility_state(
            transform(PULLOUT_HELPER_TRANSFORM), target, occluder
        )
        recipient = cross_traffic_visibility_state(
            transform(PULLOUT_RECIPIENT_TRANSFORM), target, occluder
        )
        self.assertTrue(helper["geometrically_visible"])
        self.assertFalse(helper["occluded_by_controlled_truck"])
        self.assertFalse(recipient["geometrically_visible"])
        self.assertTrue(recipient["occluded_by_controlled_truck"])

    def test_phase2_queue_reveal_frozen_geometry_has_distinct_initial_visibility(self):
        carla = __import__("carla")

        def transform(values):
            return carla.Transform(
                carla.Location(x=values[0], y=values[1], z=values[2]),
                carla.Rotation(yaw=values[3]),
            )

        self.assertEqual(
            QUEUE_REVEAL_GEOMETRY_ID,
            "town10hd_opt_queue_reveal_lead_vehicle_v1",
        )
        routes = queue_reveal_frozen_routes()
        self.assertEqual(set(routes), {"recipient", "helper", "occluder"})
        self.assertEqual(
            len(routes["occluder"]), len(QUEUE_REVEAL_OCCLUDER_ROUTE_VALUES)
        )
        target = transform(QUEUE_REVEAL_TARGET_TRANSFORM)
        occluder = transform(QUEUE_REVEAL_OCCLUDER_TRANSFORM)
        helper = cross_traffic_visibility_state(
            transform(QUEUE_REVEAL_HELPER_TRANSFORM), target, occluder
        )
        recipient = cross_traffic_visibility_state(
            transform(QUEUE_REVEAL_RECIPIENT_TRANSFORM), target, occluder
        )
        self.assertTrue(helper["geometrically_visible"])
        self.assertFalse(helper["occluded_by_controlled_truck"])
        self.assertFalse(recipient["geometrically_visible"])
        self.assertTrue(recipient["occluded_by_controlled_truck"])

    def test_review_route_shield_accounts_for_an_angled_vehicle_envelope(self):
        carla = __import__("carla")

        class ActorList:
            def __init__(self, vehicle):
                self.vehicle = vehicle

            def filter(self, pattern):
                return [self.vehicle] if pattern == "vehicle.*" else []

        class World:
            def __init__(self, vehicle):
                self.actors = ActorList(vehicle)

            def get_actors(self):
                return self.actors

        other = SimpleNamespace(
            id=2,
            type_id="vehicle.sprinter.mercedes",
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=3.0, y=1.0)
            ),
        )
        other.get_transform = lambda: carla.Transform(
            carla.Location(x=5.0, y=2.7, z=0.0),
            carla.Rotation(yaw=20.0),
        )
        actor = SimpleNamespace(
            id=1,
            bounding_box=SimpleNamespace(
                extent=SimpleNamespace(x=2.4, y=0.9)
            ),
        )
        actor.get_world = lambda: World(other)
        controller = DirectRouteController.__new__(DirectRouteController)
        controller.actor = actor
        controller.last_yield = None
        ego = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=0.0),
            carla.Rotation(yaw=0.0),
        )

        self.assertTrue(controller._must_yield(ego, speed_mps=3.0))
        self.assertEqual(controller.last_yield["actor_id"], 2)
        other.get_transform = lambda: carla.Transform(
            carla.Location(x=5.0, y=2.7, z=0.0),
            carla.Rotation(yaw=0.0),
        )
        self.assertFalse(controller._must_yield(ego, speed_mps=3.0))

    def test_review_route_shield_predicts_a_moving_pedestrian_crossing(self):
        carla = __import__("carla")

        class ActorList:
            def __init__(self, walker):
                self.walker = walker

            def filter(self, pattern):
                return [self.walker] if pattern == "walker.pedestrian.*" else []

        walker = SimpleNamespace(
            id=2,
            type_id="walker.pedestrian.0001",
            bounding_box=SimpleNamespace(extent=SimpleNamespace(x=0.3, y=0.3)),
        )
        walker.get_transform = lambda: carla.Transform(
            carla.Location(x=10.0, y=4.0), carla.Rotation(yaw=-90.0)
        )
        walker.get_velocity = lambda: carla.Vector3D(x=0.0, y=-1.5)
        actor = SimpleNamespace(
            id=1,
            bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.5, y=1.0)),
        )
        actor.get_world = lambda: SimpleNamespace(
            get_actors=lambda: ActorList(walker)
        )
        controller = DirectRouteController.__new__(DirectRouteController)
        controller.actor = actor
        controller.last_yield = None
        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0), carla.Rotation(yaw=0.0)
        )
        self.assertTrue(controller._must_yield(transform, speed_mps=5.0))
        self.assertLessEqual(
            controller.last_yield["predicted_lateral_m"],
            controller.last_yield["lateral_limit_m"],
        )
        walker.get_velocity = lambda: carla.Vector3D()
        self.assertFalse(controller._must_yield(transform, speed_mps=5.0))

    def test_phase2_midblock_occluder_settles_before_physics_is_frozen(self):
        carla = __import__("carla")
        commanded = carla.Transform(
            carla.Location(x=-6.0, y=72.5, z=0.8),
            carla.Rotation(yaw=0.073),
        )
        transforms = [
            carla.Transform(
                carla.Location(x=-6.0 + min(index, 4) * 0.002, y=72.5, z=z),
                carla.Rotation(yaw=0.073),
            )
            for index, z in enumerate(
                (0.8, 0.5, 0.1, -0.05, -0.056, -0.056, -0.056, -0.056, -0.056)
            )
        ]

        class FakeWorld:
            step = 0

            def tick(self, _timeout_s):
                self.step += 1
                return 1000 + self.step

        class FakeActor:
            def __init__(self, world):
                self.world = world
                self.physics_calls = []

            def set_simulate_physics(self, enabled):
                self.physics_calls.append(bool(enabled))

            def apply_control(self, _control):
                return None

            def get_transform(self):
                return transforms[min(max(self.world.step - 1, 0), len(transforms) - 1)]

            def get_velocity(self):
                return carla.Vector3D(z=0.0)

        world = FakeWorld()
        actor = FakeActor(world)
        result = _settle_parked_occluder(
            world, actor, commanded, timeout_s=1.0
        )
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(result["settled_z_m"], -0.056, places=6)
        self.assertLess(result["xy_drift_m"], 0.01)
        self.assertEqual(actor.physics_calls, [True, False])

    def test_vehicle_v2_locks_recipe_and_regimes_in_every_split(self):
        config = _load_config(CONFIG_PATH)

        self.assertEqual(config["experiment_name"], "policy_corpus_vehicle_v2")
        self.assertEqual(len(config["runs"]), 32)
        for run in [*config["smoke_runs"], *config["runs"]]:
            options = _effective_options(_resolved_run_args(config, run))
            self.assertEqual(options["--npc-pedestrians"], "0")
            self.assertEqual(options["--radar-points-per-second"], "200000")
            self.assertEqual(options["--radar-rasterizer"], "fast")
            self.assertEqual(options["--object-nms-radius-px"], "2")
            self.assertEqual(options["--topk-objects"], "120")

        exact = [
            run for run in config["runs"]
            if run["scenario_family"] == "exact_fast_convoy"
        ]
        counts = {split: sum(run["split"] == split for run in exact)
                  for split in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 4, "validation": 2, "test": 2})
        self.assertTrue(all(int(run["requested_frames"]) == 100 for run in exact))

    def test_contract_rejects_detector_recipe_drift(self):
        config = _load_config(CONFIG_PATH)
        drifted = copy.deepcopy(config)
        index = drifted["common_args"].index("--radar-points-per-second") + 1
        drifted["common_args"][index] = 5000

        with self.assertRaisesRegex(ValueError, "radar-points-per-second"):
            _validate_collection_contract(drifted)

    def test_contract_rejects_seed_reuse_across_trajectory_splits(self):
        config = _load_config(CONFIG_PATH)
        duplicated = copy.deepcopy(config)
        duplicated["runs"][-1]["seed"] = duplicated["runs"][0]["seed"]

        with self.assertRaisesRegex(ValueError, "seeds must be unique"):
            _validate_collection_contract(duplicated)

    def test_existing_v1_and_detection_gate_still_load(self):
        config_dir = CONFIG_PATH.parent
        self.assertEqual(
            _load_config(config_dir / "policy_corpus_v1.yaml")["experiment_name"],
            "policy_corpus_v1",
        )
        self.assertEqual(
            _load_config(config_dir / "detection_ab_gate_v1.yaml")["experiment_name"],
            "detection_ab_gate_v1",
        )

    def test_vehicle_freshness_config_uses_verified_pass_input(self):
        config = _load_freshness_config(
            CONFIG_PATH.parent / "freshness_rescore_vehicle_v2.yaml"
        )
        _validate_reference(config)

        self.assertEqual(config["corpus_scope"], "vehicle_only_track_a")
        self.assertEqual(config["provenance"]["prior_verification_status"], "PASS")

    def test_fast_dwell_breaks_at_timestamp_gap(self):
        dwell = _longest_true_dwell(
            [True, True, True, True, False, True],
            [0.0, 0.1, 0.2, 0.7, 0.8, 0.9],
        )
        self.assertAlmostEqual(dwell, 0.3)


class AdvisorRichCorpusConfigTests(unittest.TestCase):
    def test_advisor_v5_uses_native_training_clock_and_observed_density_gate(self):
        config = _load_advisor_config(ADVISOR_V5_CONFIG_PATH)

        self.assertEqual(config["experiment_name"], "policy_corpus_advisor_rich_v5")
        self.assertEqual(len(config["smoke_runs"]), 3)
        self.assertEqual(len(config["runs"]), 24)
        for run in [*config["smoke_runs"], *config["runs"]]:
            options = _effective_options(_resolved_run_args(config, run))
            expected = {
                "--fps": "10",
                "--world-tick-hz": "10",
                "--sensor-every-tick": "true",
                "--radar-points-per-second": "200000",
                "--camera-width": "1280",
                "--camera-height": "720",
                "--camera-fov": "120",
                "--radar-hfov": "120",
                "--radar-rasterizer": "legacy",
                "--radar-raster-radius-px": "4",
                "--radar-temporal-window-frames": "2",
            }
            self.assertEqual({key: options.get(key) for key in expected}, expected)
            self.assertNotIn("--no-sensor-every-tick", options)
            if run["scenario_family"] == "exact_fast_convoy":
                self.assertEqual(int(run["requested_frames"]), 60)
                self.assertEqual(options["--max-frames"], "60")
        integration = config["advisor_integration"]
        self.assertTrue(str(integration["spawn_blocker_script"]).endswith("spawn_blocker_v4.py"))
        self.assertEqual(
            integration["smoke_gate"]["pedestrian_role_prefix"],
            "pedestrian_blocker_v4",
        )
        self.assertEqual(float(integration["fixed_delta_seconds"]), 0.1)
        self.assertEqual(int(integration["update_hz"]), 10)
        self.assertEqual(
            float(integration["radar_density_gate"]["reference_projected_points_median"]),
            18591.5,
        )
        self.assertEqual(
            int(integration["smoke_gate"]["control_ticks_per_sensor_frame"]), 1
        )
        self.assertEqual(
            float(integration["smoke_gate"]["exact_fast_max_route_offset_m"]),
            4.0,
        )

    def test_exact_fast_scenario_gate_rejects_route_departure_and_walker_impact(self):
        route = pd.DataFrame({"ego_x": [0.0, 10.0], "ego_y": [0.0, 0.0]})
        ground_truth = pd.DataFrame(
            [
                {
                    "actor_id": 1,
                    "role_name": "exact",
                    "class_name": "vehicle",
                    "carla_timestamp": 0.0,
                    "origin_x": 0.0,
                    "origin_y": 0.0,
                },
                {
                    "actor_id": 1,
                    "role_name": "exact",
                    "class_name": "vehicle",
                    "carla_timestamp": 0.1,
                    "origin_x": 10.0,
                    "origin_y": 8.0,
                },
                {
                    "actor_id": 2,
                    "role_name": "walker",
                    "class_name": "pedestrian",
                    "carla_timestamp": 0.0,
                    "origin_x": 1.0,
                    "origin_y": 1.0,
                },
                {
                    "actor_id": 2,
                    "role_name": "walker",
                    "class_name": "pedestrian",
                    "carla_timestamp": 0.1,
                    "origin_x": 2.0,
                    "origin_y": 1.0,
                },
            ]
        )

        summary = _exact_fast_scenario_summary(
            ground_truth,
            route,
            role_name="exact",
            maximum_route_offset_m=4.0,
            pedestrian_speed_max_mps=3.5,
        )

        self.assertFalse(summary["pass"])
        self.assertIn("exact_fast_target_left_authored_route", summary["failures"])
        self.assertIn("exact_fast_pedestrian_impact_signature", summary["failures"])

    def test_all_family_pedestrian_motion_gate_rejects_ego_push_signature(self):
        ground_truth = pd.DataFrame(
            {
                "class_name": ["pedestrian", "pedestrian"],
                "actor_id": [7, 7],
                "carla_timestamp": [0.0, 0.1],
                "origin_x": [0.0, 0.0],
                "origin_y": [0.0, 0.43],
            }
        )
        summary = _pedestrian_motion_summary(
            ground_truth, maximum_speed_mps=3.5
        )
        self.assertFalse(summary["pass"])
        self.assertEqual(summary["pedestrian_speed_rows_above_max"], 1)
        self.assertIn("pedestrian_impact_signature", summary["failures"])

    def test_radar_density_gate_accepts_reference_and_rejects_half_density(self):
        gate = {
            "reference_projected_points_median": 18591.5,
            "relative_tolerance": 0.10,
            "minimum_metric_frames": 3,
        }
        accepted = _radar_density_summary(
            pd.DataFrame({"radar_projected_points": [18300, 18500, 18700]}),
            gate,
        )
        rejected = _radar_density_summary(
            pd.DataFrame({"radar_projected_points": [9500, 9700, 9900]}),
            gate,
        )

        self.assertTrue(accepted["pass"])
        self.assertFalse(rejected["pass"])
        self.assertIn(
            "radar_projected_points_median_outside_contract",
            rejected["failures"],
        )

    def test_npc_shield_forward_corridor_includes_crossing_walker_not_sidewalk(self):
        transform = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0),
            get_forward_vector=lambda: SimpleNamespace(x=1.0, y=0.0),
        )

        self.assertTrue(
            _in_forward_corridor(
                transform,
                SimpleNamespace(x=10.0, y=2.0),
                maximum_forward_m=15.0,
                maximum_lateral_m=3.5,
            )
        )
        self.assertFalse(
            _in_forward_corridor(
                transform,
                SimpleNamespace(x=10.0, y=5.0),
                maximum_forward_m=15.0,
                maximum_lateral_m=3.5,
            )
        )
        self.assertTrue(
            _vehicle_requires_yield(
                transform,
                SimpleNamespace(x=9.0, y=2.5),
                maximum_forward_m=12.0,
            )
        )
        self.assertFalse(
            _vehicle_requires_yield(
                transform,
                SimpleNamespace(x=9.0, y=3.5),
                maximum_forward_m=12.0,
            )
        )

        stationary = SimpleNamespace(x=0.0, y=0.0)
        moving = SimpleNamespace(x=0.0, y=-1.0)
        curb = SimpleNamespace(x=10.0, y=2.7)
        self.assertFalse(_walker_requires_yield(transform, curb, stationary))
        self.assertTrue(_walker_requires_yield(transform, curb, moving))
        registered_waiting = SimpleNamespace(x=10.0, y=4.5)
        self.assertFalse(
            _walker_requires_yield(transform, registered_waiting, stationary)
        )
        self.assertTrue(
            _walker_requires_yield(
                transform,
                registered_waiting,
                stationary,
                registered_crossing=True,
            )
        )
        self.assertFalse(
            _in_forward_corridor(
                transform,
                SimpleNamespace(x=-1.0, y=0.0),
                maximum_forward_m=15.0,
                maximum_lateral_m=3.5,
            )
        )

    def test_pedestrian_gate_excludes_incidental_blockers_in_other_families(self):
        truth = pd.DataFrame(
            {
                "scenario_family": ["mixed_urban", "ped_crossing", "exact_fast_convoy"],
                "class_name": ["pedestrian"] * 3,
                "role_name": ["pedestrian_blocker_v4_1"] * 3,
                "in_camera_frustum": [True] * 3,
                "distance_m": [8.0] * 3,
                "origin_x": [1.0, 2.0, 3.0],
                "origin_y": [4.0, 5.0, 6.0],
            }
        )
        gate = {
            "headline_range_m": 25.0,
            "pedestrian_gate_scenario_family": "ped_crossing",
            "pedestrian_role_prefix": "pedestrian_blocker_v4",
        }

        scoped = _controlled_pedestrian_gate_rows(truth, gate)

        self.assertEqual(len(scoped), 1)
        self.assertEqual(scoped.iloc[0]["scenario_family"], "ped_crossing")

    def test_advisor_v4_inherits_splits_and_locks_on_contract_dual_clock(self):
        config = _load_advisor_config(ADVISOR_V4_CONFIG_PATH)

        self.assertEqual(config["experiment_name"], "policy_corpus_advisor_rich_v4")
        self.assertEqual(len(config["smoke_runs"]), 3)
        self.assertEqual(len(config["runs"]), 24)
        for run in [*config["smoke_runs"], *config["runs"]]:
            options = _effective_options(_resolved_run_args(config, run))
            expected = {
                "--fps": "10",
                "--world-tick-hz": "20",
                "--no-sensor-every-tick": "true",
                "--camera-width": "1280",
                "--camera-height": "720",
                "--camera-fov": "120",
                "--radar-hfov": "120",
                "--radar-points-per-second": "200000",
                "--radar-rasterizer": "legacy",
                "--radar-raster-radius-px": "4",
                "--radar-temporal-window-frames": "2",
                "--object-nms-radius-px": "2",
                "--topk-objects": "120",
            }
            self.assertEqual({key: options.get(key) for key in expected}, expected)
            self.assertNotIn("--sensor-every-tick", options)
        integration = config["advisor_integration"]
        self.assertIs(integration["reload_world_before_run"], True)
        self.assertIn("--safe", integration["common_traffic_args"])
        self.assertEqual(float(integration["tm_distance_to_leading_vehicle_m"]), 12.0)
        self.assertEqual(float(integration["tm_speed_difference_pct"]), 65.0)
        self.assertEqual(float(integration["tm_desired_speed_mps"]), 8.0)
        self.assertEqual(integration["npc_route_mode"], "direct_loop")
        self.assertEqual(float(integration["npc_direct_route_speed_mps"]), 6.0)
        self.assertEqual(
            int(integration["traffic_sanity_gate"]["maximum_collision_incidents"]), 0
        )
        self.assertEqual(
            integration["smoke_gate"]["pedestrian_gate_scenario_family"],
            "ped_crossing",
        )
        mixed = next(
            run for run in config["smoke_runs"] if run["scenario_family"] == "mixed_urban"
        )
        blocker, traffic = _population_commands(config, mixed)
        self.assertEqual(blocker, [])
        self.assertTrue(
            any(value.endswith("run_advisor_generate_traffic.py") for value in traffic)
        )
        self.assertEqual(
            traffic[traffic.index("--vehicle-spawn-clearance-m") + 1], "35.0"
        )
        self.assertEqual(traffic[traffic.index("--maximum-route-offset-m") + 1], "2.0")
        self.assertEqual(
            traffic[traffic.index("--maximum-route-heading-error-deg") + 1], "35.0"
        )
        self.assertEqual(
            traffic[traffic.index("--traffic-speed-difference-pct") + 1], "65.0"
        )
        self.assertEqual(
            traffic[traffic.index("--traffic-desired-speed-mps") + 1], "8.0"
        )
        self.assertIn("--defer-vehicle-control-to-runner", traffic)
        self.assertEqual(
            config["advisor_integration"]["families"]["mixed_urban"][
                "number_of_vehicles"
            ],
            4,
        )
        self.assertIn(
            "--one-shot-pedestrians",
            config["advisor_integration"]["common_blocker_args"],
        )
        for family in ("mixed_urban", "exact_fast_convoy"):
            family_spec = config["advisor_integration"]["families"][family]
            self.assertIn("--no-pedestrian-blockers", family_spec["blocker_args"])
            self.assertNotIn(
                "pedestrian_blocker_v4",
                family_spec["minimum_blocker_role_prefix_counts"],
            )

    def test_one_shot_blocker_retires_completed_crossing_without_respawn(self):
        advisor = blocker_wrapper.advisor_blocker
        original_request = advisor.request_respawn
        original_retire = advisor.retire_state_actors
        original_reset = advisor.reset_activation_fields
        calls = []
        state = SimpleNamespace(
            index=1,
            generation=1,
            actor=SimpleNamespace(id=42),
            state="HOLDING",
            respawn_due=None,
        )

        def retire(fake_state, registry):
            calls.append((fake_state.actor.id, registry))
            fake_state.actor = None

        try:
            advisor.retire_state_actors = retire
            advisor.reset_activation_fields = lambda _state: None
            blocker_wrapper._install_one_shot_pedestrian_lifecycle()
            advisor.request_respawn(
                state,
                12.0,
                "post-event hold completed after near miss",
                object(),
                SimpleNamespace(),
            )
        finally:
            advisor.request_respawn = original_request
            advisor.retire_state_actors = original_retire
            advisor.reset_activation_fields = original_reset

        self.assertEqual(len(calls), 1)
        self.assertEqual(state.state, advisor.STATE_RESPAWN_PENDING)
        self.assertTrue(math.isinf(state.respawn_due))

    def test_traffic_sanity_gate_deduplicates_collision_and_detects_gridlock(self):
        trajectories = []
        for frame_id in range(1, 81):
            for actor_id in range(1, 7):
                trajectories.append(
                    {
                        "frame_id": frame_id,
                        "carla_timestamp": frame_id * 0.05,
                        "actor_id": actor_id,
                        "speed_mps": 0.1,
                    }
                )
        collisions = [
            {"frame_id": 10, "npc_actor_id": 1, "other_actor_id": 2},
            {"frame_id": 10, "npc_actor_id": 2, "other_actor_id": 1},
            {"frame_id": 11, "npc_actor_id": 1, "other_actor_id": 2},
        ]
        gate = {
            "maximum_collision_incidents": 0,
            "minimum_actor_observation_fraction": 0.95,
            "minimum_per_actor_frame_observation_fraction": 0.95,
            "stopped_speed_max_mps": 0.5,
            "gridlock_minimum_npc_count": 5,
            "gridlock_stopped_fraction": 0.8,
            "persistent_gridlock_min_s": 3.0,
        }

        summary = _traffic_sanity_summary(
            pd.DataFrame(trajectories),
            pd.DataFrame(collisions),
            list(range(1, 7)),
            gate,
            expected_frame_count=80,
        )

        self.assertEqual(summary["collision_events"], 1)
        self.assertGreaterEqual(summary["persistent_gridlock_dwell_s"], 3.0)
        self.assertIn("owned_actor_collision_incidents_above_gate", summary["failures"])
        self.assertIn("persistent_network_gridlock", summary["failures"])
        self.assertEqual(1.0, summary["minimum_per_actor_frame_observation_fraction"])

        under_sampled = _traffic_sanity_summary(
            pd.DataFrame(trajectories[:6]),
            pd.DataFrame(),
            list(range(1, 7)),
            gate,
            expected_frame_count=80,
        )
        self.assertIn(
            "insufficient_npc_per_frame_observation", under_sampled["failures"]
        )

    def test_traffic_collision_gate_ignores_settlement_but_keeps_real_contacts(self):
        trajectories = pd.DataFrame(
            [
                {
                    "frame_id": frame_id,
                    "carla_timestamp": frame_id * 0.1,
                    "actor_id": 1,
                    "speed_mps": 2.0,
                }
                for frame_id in range(1, 11)
            ]
        )
        collisions = pd.DataFrame(
            [
                {
                    "frame_id": 2,
                    "npc_actor_id": 1,
                    "other_actor_id": 0,
                    "other_type_id": "static.road",
                    "normal_impulse_x": 8.0,
                    "normal_impulse_y": 2.0,
                    "normal_impulse_z": 600.0,
                },
                {
                    "frame_id": 3,
                    "npc_actor_id": 50,
                    "other_actor_id": 0,
                    "other_type_id": "static.sidewalk",
                    "normal_impulse_x": 0.0,
                    "normal_impulse_y": 0.0,
                    "normal_impulse_z": 0.0,
                },
                {
                    "frame_id": 4,
                    "npc_actor_id": 1,
                    "other_actor_id": 50,
                    "other_type_id": "walker.pedestrian.0001",
                    "normal_impulse_x": 0.0,
                    "normal_impulse_y": 0.0,
                    "normal_impulse_z": 0.0,
                },
                {
                    "frame_id": 8,
                    "npc_actor_id": 1,
                    "other_actor_id": 0,
                    "other_type_id": "static.wall",
                    "normal_impulse_x": 80.0,
                    "normal_impulse_y": 0.0,
                    "normal_impulse_z": 0.0,
                },
            ]
        )
        gate = {
            "maximum_collision_incidents": 0,
            "minimum_static_collision_horizontal_impulse": 50.0,
            "minimum_actor_observation_fraction": 0.95,
            "minimum_per_actor_frame_observation_fraction": 0.95,
            "stopped_speed_max_mps": 0.5,
            "gridlock_minimum_npc_count": 5,
            "gridlock_stopped_fraction": 0.8,
            "persistent_gridlock_min_s": 3.0,
        }
        summary = _traffic_sanity_summary(
            trajectories, collisions, [1], gate, expected_frame_count=10
        )
        self.assertEqual(summary["ignored_static_contact_rows"], 2)
        self.assertEqual(summary["collision_events"], 2)
        self.assertIn("owned_actor_collision_incidents_above_gate", summary["failures"])

    def test_registered_crossing_yield_is_not_misclassified_as_gridlock(self):
        trajectories = []
        for frame_id in range(1, 81):
            for actor_id in range(1, 7):
                trajectories.append(
                    {
                        "frame_id": frame_id,
                        "carla_timestamp": frame_id * 0.1,
                        "actor_id": actor_id,
                        "speed_mps": 0.0,
                        "registered_hazard_yield_active": frame_id <= 60,
                    }
                )
        gate = {
            "maximum_collision_incidents": 0,
            "minimum_actor_observation_fraction": 0.95,
            "minimum_per_actor_frame_observation_fraction": 0.95,
            "stopped_speed_max_mps": 0.5,
            "gridlock_minimum_npc_count": 5,
            "gridlock_stopped_fraction": 0.8,
            "persistent_gridlock_min_s": 3.0,
        }
        summary = _traffic_sanity_summary(
            pd.DataFrame(trajectories),
            pd.DataFrame(),
            list(range(1, 7)),
            gate,
            expected_frame_count=80,
        )
        self.assertGreaterEqual(summary["raw_stopped_network_dwell_s"], 7.0)
        self.assertGreaterEqual(summary["registered_hazard_yield_dwell_s"], 5.0)
        self.assertLess(summary["persistent_gridlock_dwell_s"], 3.0)
        self.assertNotIn("persistent_network_gridlock", summary["failures"])

    def test_one_registered_yield_does_not_mask_other_stalled_npcs(self):
        trajectories = []
        for frame_id in range(1, 81):
            for actor_id in range(1, 7):
                trajectories.append(
                    {
                        "frame_id": frame_id,
                        "carla_timestamp": frame_id * 0.1,
                        "actor_id": actor_id,
                        "speed_mps": 0.0,
                        "registered_hazard_yield_active": actor_id == 1,
                    }
                )
        gate = {
            "maximum_collision_incidents": 0,
            "minimum_actor_observation_fraction": 0.95,
            "minimum_per_actor_frame_observation_fraction": 0.95,
            "stopped_speed_max_mps": 0.5,
            "gridlock_minimum_npc_count": 5,
            "gridlock_stopped_fraction": 0.8,
            "persistent_gridlock_min_s": 3.0,
        }
        summary = _traffic_sanity_summary(
            pd.DataFrame(trajectories),
            pd.DataFrame(),
            list(range(1, 7)),
            gate,
            expected_frame_count=80,
        )
        self.assertGreaterEqual(summary["persistent_gridlock_dwell_s"], 7.0)
        self.assertIn("persistent_network_gridlock", summary["failures"])

    def test_direct_route_vehicle_yield_excludes_adjacent_lane(self):
        carla = __import__("carla")
        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0), carla.Rotation(yaw=0.0)
        )
        self.assertTrue(
            _vehicle_requires_yield(
                transform, carla.Location(x=10.0, y=2.5)
            )
        )
        self.assertFalse(
            _vehicle_requires_yield(
                transform, carla.Location(x=10.0, y=3.5)
            )
        )

    def test_direct_route_vehicle_envelope_catches_crossing_not_adjacent(self):
        carla = __import__("carla")
        own = SimpleNamespace(
            bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.5, y=1.0))
        )
        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0), carla.Rotation(yaw=0.0)
        )

        def other(yaw):
            value = SimpleNamespace(
                bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.5, y=1.0))
            )
            value.get_transform = lambda: carla.Transform(
                carla.Location(x=8.0, y=3.5), carla.Rotation(yaw=yaw)
            )
            return value

        self.assertFalse(
            _vehicle_envelope_requires_yield(
                own, transform, other(0.0), speed_mps=4.0
            )
        )
        self.assertTrue(
            _vehicle_envelope_requires_yield(
                own, transform, other(45.0), speed_mps=4.0
            )
        )

    def test_on_contract_diagnostic_is_exact_and_cannot_schedule_a_full_corpus(self):
        config = _load_advisor_config(ON_CONTRACT_CONFIG_PATH)

        self.assertEqual(config["runs"], [])
        self.assertEqual(len(config["smoke_runs"]), 1)
        run = config["smoke_runs"][0]
        options = _effective_options(_resolved_run_args(config, run))
        expected = {
            "--fps": "10",
            "--camera-width": "1280",
            "--camera-height": "720",
            "--camera-fov": "120",
            "--radar-hfov": "120",
            "--radar-points-per-second": "200000",
            "--radar-rasterizer": "legacy",
            "--radar-raster-radius-px": "4",
            "--radar-temporal-window-frames": "2",
            "--max-objects-drawn": "120",
            "--retain-diagnostic-inputs": "true",
        }
        self.assertEqual({key: options.get(key) for key in expected}, expected)
        self.assertEqual(
            float(config["advisor_integration"]["fixed_delta_seconds"]), 0.1
        )
        self.assertEqual(int(config["advisor_integration"]["update_hz"]), 10)

    def test_target_radar_hits_use_model_scaled_actor_box(self):
        gt = {"bbox_x1": 100.0, "bbox_y1": 50.0, "bbox_x2": 300.0, "bbox_y2": 150.0}
        radar = {
            "display_size": np.asarray([1000, 500]),
            "model_size": np.asarray([500, 250]),
            "points_u": np.asarray([49.0, 50.0, 100.0, 150.0, 151.0]),
            "points_v": np.asarray([50.0, 25.0, 50.0, 75.0, 50.0]),
            "points_valid_projection": np.asarray([1, 1, 0, 1, 1]),
        }

        self.assertEqual(target_radar_hit_count(gt, radar), 2)

    def test_advisor_rich_config_locks_observe_existing_recipe_and_splits(self):
        config = _load_advisor_config(ADVISOR_CONFIG_PATH)

        self.assertEqual(config["experiment_name"], "policy_corpus_advisor_rich_v3")
        self.assertEqual(len(config["smoke_runs"]), 3)
        self.assertEqual(len(config["runs"]), 24)
        for run in [*config["smoke_runs"], *config["runs"]]:
            options = _effective_options(_resolved_run_args(config, run))
            self.assertEqual(options["--npc-vehicles"], "0")
            self.assertEqual(options["--npc-pedestrians"], "0")
            self.assertEqual(options["--tm-port"], "8010")
            self.assertEqual(options["--fps"], "20")
            self.assertEqual(options["--sensor-every-tick"], "true")
            self.assertEqual(options["--radar-points-per-second"], "200000")
            self.assertEqual(options["--radar-rasterizer"], "fast")
            self.assertEqual(options["--object-nms-radius-px"], "2")
            self.assertEqual(options["--topk-objects"], "120")

        integration = config["advisor_integration"]
        self.assertEqual(float(integration["pedestrian_speed_mps"]), 2.0)
        self.assertEqual(float(integration["minimum_pedestrian_speed_mps"]), 1.0)
        self.assertEqual(int(integration["tm_port"]), 8010)
        self.assertEqual(float(integration["fixed_delta_seconds"]), 0.05)

    def test_population_commands_keep_advisor_scripts_passive_and_realistic(self):
        config = _load_advisor_config(ADVISOR_CONFIG_PATH)
        exact = next(
            run for run in config["smoke_runs"]
            if run["scenario_family"] == "exact_fast_convoy"
        )
        blocker, traffic = _population_commands(config, exact)

        self.assertTrue(
            any(value.endswith("run_advisor_spawn_blocker.py") for value in blocker)
        )
        self.assertEqual(blocker[blocker.index("--pedestrian-speed") + 1], "2.0")
        self.assertEqual(blocker[blocker.index("--min-pedestrian-speed") + 1], "1.0")
        self.assertEqual(blocker.count("--pedestrian-location"), 1)
        self.assertIn("--no-vehicle-blockers", blocker)
        self.assertNotIn("--asynch", traffic)
        self.assertEqual(traffic[traffic.index("--tm-port") + 1], "8010")
        self.assertEqual(traffic[traffic.index("--number-of-vehicles") + 1], "0")

    def test_blocker_snapshot_poll_observes_next_frame_without_ticking(self):
        class FakeWorld:
            def __init__(self):
                self.frames = iter((10, 10, 11))

            def get_snapshot(self):
                return type("Snapshot", (), {"frame": next(self.frames)})()

        snapshot = _poll_for_tick(FakeWorld(), 0.1)
        self.assertEqual(snapshot.frame, 11)

    def test_postflight_waits_passively_for_deferred_async_destruction(self):
        class FakeActors:
            def __init__(self, count):
                self.count = count

            def filter(self, _pattern):
                return [object()] * self.count

        class FakeWorld:
            def __init__(self):
                self.snapshots = iter((FakeActors(1), FakeActors(0)))

            def get_actors(self):
                return next(self.snapshots)

            def tick(self, *_args):
                raise AssertionError("postflight must not tick the world")

        self.assertEqual(
            _tick_until_empty(FakeWorld(), 0.2),
            {
                "vehicle.*": 0,
                "walker.pedestrian.*": 0,
                "sensor.*": 0,
                "controller.ai.walker": 0,
            },
        )

    def test_route_is_ui_compatible_loop_through_advisor_blockers(self):
        route_path = (
            REPO_ROOT
            / "data_collection"
            / "routes"
            / "town10hd_opt_advisor_demo_loop_v1.json"
        )
        route = json.loads(route_path.read_text(encoding="utf-8"))
        points = route["planned_path"]
        targets = [
            (19.791866302490234, 32.016666412353516),
            (8.827261924743652, 62.21647644042969),
            (56.35101318359375, 62.8189811706543),
            (47.52729797363281, 40.43889617919922),
            (-13.132160186767578, 28.438270568847656),
        ]

        self.assertTrue(route["loop"])
        self.assertTrue(str(route["map"]).endswith("Town10HD_Opt"))
        self.assertEqual(
            route["ui_selection"]["planner"],
            "physical_ai_scenario_controller_ui_v2.py",
        )
        self.assertGreater(len(points), 100)
        for target in targets:
            minimum = min(
                math.hypot(float(point["x"]) - target[0], float(point["y"]) - target[1])
                for point in points
            )
            self.assertLess(minimum, 5.0)

        progress_path = route_path.with_suffix(".progress.csv")
        with progress_path.open("r", encoding="utf-8", newline="") as stream:
            first_progress = next(csv.DictReader(stream))
        start = route["start"]
        yaw = math.radians(float(start["rotation"]["yaw"]))
        dx = float(first_progress["ego_x"]) - float(start["location"]["x"])
        dy = float(first_progress["ego_y"]) - float(start["location"]["y"])
        forward_m = dx * math.cos(yaw) + dy * math.sin(yaw)
        self.assertGreaterEqual(forward_m, 6.0)

    def test_advisor_contract_rejects_demo_speed_and_tm_drift(self):
        config = _load_advisor_config(ADVISOR_CONFIG_PATH)
        speed_drift = copy.deepcopy(config)
        speed_drift["advisor_integration"]["pedestrian_speed_mps"] = 30.0
        with self.assertRaisesRegex(ValueError, "1-2 m/s"):
            _validate_advisor_contract(speed_drift)

        tm_drift = copy.deepcopy(config)
        tm_drift["advisor_integration"]["tm_port"] = 8000
        with self.assertRaisesRegex(ValueError, "8010"):
            _validate_advisor_contract(tm_drift)

    def test_advisor_freshness_config_is_multiclass_v5_input(self):
        config = _load_freshness_config(
            ADVISOR_CONFIG_PATH.parent / "freshness_rescore_advisor_rich_v5.yaml"
        )
        _validate_reference(config)

        self.assertEqual(config["corpus_scope"], "multiclass_advisor_rich")
        self.assertEqual(config["provenance"]["prior_verification_status"], "PASS")
        self.assertTrue(config["matching"]["use_verification_thresholds"])

    def test_advisor_v5_evaluation_accepts_structural_gates_only(self):
        with (
            ADVISOR_CONFIG_PATH.parent
            / "evaluation_contract_advisor_rich_v5.yaml"
        ).open("r", encoding="utf-8") as stream:
            contract = yaml.safe_load(stream)
        self.assertEqual(contract["recall_gate_mode"], "report_only")
        self.assertEqual(contract["excluded_episode_ids"], ["pcarv5_mixed_va01"])
        self.assertEqual(
            contract["diagnostic_recall_range_m_by_class"],
            {"pedestrian": 12.0, "vehicle": 25.0},
        )


if __name__ == "__main__":
    unittest.main()
