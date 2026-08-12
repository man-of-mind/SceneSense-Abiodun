import copy
import csv
import json
import math
import unittest
from pathlib import Path

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
    _load_config as _load_advisor_config,
    _population_commands,
    _validate_advisor_contract,
)
from data_collection.run_advisor_spawn_blocker import _poll_for_tick


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "data_collection" / "configs" / "policy_corpus_vehicle_v2.yaml"
ADVISOR_CONFIG_PATH = (
    REPO_ROOT / "data_collection" / "configs" / "policy_corpus_advisor_rich_v3.yaml"
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
