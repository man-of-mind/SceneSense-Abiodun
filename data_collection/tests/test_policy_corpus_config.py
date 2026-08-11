import copy
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


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "data_collection" / "configs" / "policy_corpus_vehicle_v2.yaml"


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


if __name__ == "__main__":
    unittest.main()
