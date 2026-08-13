import copy
import csv
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
    _in_forward_corridor,
    _walker_requires_yield,
    _load_config as _load_advisor_config,
    _population_commands,
    _tick_until_empty,
    _traffic_sanity_summary,
    _validate_advisor_contract,
)
from data_collection.run_advisor_spawn_blocker import _poll_for_tick
from data_collection.replay_on_contract_pedestrian_diagnostic import (
    target_radar_hit_count,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "data_collection" / "configs" / "policy_corpus_vehicle_v2.yaml"
ADVISOR_CONFIG_PATH = (
    REPO_ROOT / "data_collection" / "configs" / "policy_corpus_advisor_rich_v3.yaml"
)
ADVISOR_V4_CONFIG_PATH = (
    REPO_ROOT / "data_collection" / "configs" / "policy_corpus_advisor_rich_v4.yaml"
)
ON_CONTRACT_CONFIG_PATH = (
    REPO_ROOT
    / "data_collection"
    / "configs"
    / "pedestrian_on_contract_diagnostic_v1.yaml"
)


class VehicleCorpusConfigTests(unittest.TestCase):
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

        stationary = SimpleNamespace(x=0.0, y=0.0)
        moving = SimpleNamespace(x=0.0, y=-1.0)
        curb = SimpleNamespace(x=10.0, y=2.7)
        self.assertFalse(_walker_requires_yield(transform, curb, stationary))
        self.assertTrue(_walker_requires_yield(transform, curb, moving))
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
            "stopped_speed_max_mps": 0.5,
            "gridlock_minimum_npc_count": 5,
            "gridlock_stopped_fraction": 0.8,
            "persistent_gridlock_min_s": 3.0,
        }

        summary = _traffic_sanity_summary(
            pd.DataFrame(trajectories), pd.DataFrame(collisions), list(range(1, 7)), gate
        )

        self.assertEqual(summary["collision_events"], 1)
        self.assertGreaterEqual(summary["persistent_gridlock_dwell_s"], 3.0)
        self.assertIn("npc_collision_incidents_above_gate", summary["failures"])
        self.assertIn("persistent_network_gridlock", summary["failures"])

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
            ADVISOR_CONFIG_PATH.parent / "freshness_rescore_advisor_rich_v3.yaml"
        )
        _validate_reference(config)

        self.assertEqual(config["corpus_scope"], "multiclass_advisor_rich")
        self.assertEqual(config["provenance"]["prior_verification_status"], "PASS")


if __name__ == "__main__":
    unittest.main()
