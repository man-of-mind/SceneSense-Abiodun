from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from phase2_map_sharing.replay_calibration_grid import (
    _candidate_diagnostics,
    _capture_artifact_fingerprints,
    _enrich_arm_metrics,
    _fingerprint_drift,
    _named_path_fingerprints,
    _paired_role_provenance,
    _result_defining_dependency_paths,
    _validate_output_dir,
    grid_settings,
    load_replay_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "phase2_map_sharing"
    / "configs"
    / "calibration_replay_sufficiency_v1.yaml"
)


class CalibrationGridContractTests(unittest.TestCase):
    def test_checked_in_grid_is_exactly_the_frozen_72_settings(self) -> None:
        config = load_replay_config(CONFIG)
        settings = grid_settings(config)
        self.assertEqual(len(settings), 72)
        self.assertEqual(len({row["setting_id"] for row in settings}), 72)
        self.assertEqual(
            sorted(
                {row["warning_emission_confidence_floor"] for row in settings}
            ),
            [0.05, 0.10, 0.15, 0.20],
        )
        self.assertEqual(
            sorted({row["map_association_gate_m"] for row in settings}),
            [2.0, 3.0, 4.0],
        )
        self.assertEqual(
            sorted({row["map_track_ttl_s"] for row in settings}), [0.5, 1.0]
        )
        self.assertEqual(
            sorted({row["warning_uncertainty_multiplier"] for row in settings}),
            [0.0, 1.0, 2.0],
        )

    def test_obsolete_96_grid_and_source_tracker_tuning_fail_closed(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            stale = copy.deepcopy(payload)
            stale["replay_grid"]["warning_emission_confidence_floors"] = [
                0.01,
                0.05,
                0.10,
                0.20,
            ]
            stale["replay_grid"]["expected_combinations"] = 96
            path.write_text(yaml.safe_dump(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "drifted"):
                load_replay_config(path)

            tuned = copy.deepcopy(payload)
            tuned["source_tracker"]["tuned"] = True
            path.write_text(yaml.safe_dump(tuned), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_tracker.tuned drifted"):
                load_replay_config(path)

    def test_every_frozen_config_leaf_fails_closed_on_drift(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        mutable_prefixes = {("execution", "parallel_workers")}

        def leaves(value, prefix=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, (*prefix, key))
            else:
                yield prefix, value

        def changed(value):
            if isinstance(value, bool):
                return not value
            if isinstance(value, (int, float)):
                return value + 0.25
            if isinstance(value, str):
                return value + "_drift"
            if isinstance(value, list):
                return [*value, value[0] if value else "drift"]
            self.fail(f"unsupported frozen test value: {value!r}")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for key_path, original in leaves(payload):
                if key_path in mutable_prefixes:
                    continue
                candidate = copy.deepcopy(payload)
                target = candidate
                for key in key_path[:-1]:
                    target = target[key]
                target[key_path[-1]] = changed(original)
                path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
                with self.subTest(key_path=".".join(key_path)):
                    with self.assertRaises(ValueError):
                        load_replay_config(path)

            extra = copy.deepcopy(payload)
            extra["unexpected_scientific_knob"] = 1
            path.write_text(yaml.safe_dump(extra), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root keys drifted"):
                load_replay_config(path)

    def test_parallel_worker_type_is_strict_not_coerced(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for invalid in (True, 8.0, "8"):
                candidate = copy.deepcopy(payload)
                candidate["execution"]["parallel_workers"] = invalid
                path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        load_replay_config(path)

    def test_expected_combination_count_is_not_coerced(self) -> None:
        payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            for invalid in (True, 72.0, 72.5, "72"):
                candidate = copy.deepcopy(payload)
                candidate["replay_grid"]["expected_combinations"] = invalid
                path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "expected_combinations"):
                        load_replay_config(path)

    def test_reanalysis_output_must_be_create_only_and_outside_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "capture"
            batch.mkdir()
            with self.assertRaisesRegex(ValueError, "sibling"):
                _validate_output_dir(batch, batch / "evaluation")
            output = root / "analysis" / "new_replay"
            _validate_output_dir(batch, output)
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                _validate_output_dir(batch, output)

    def test_source_tree_fingerprint_covers_truth_and_runtime_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "capture"
            truth = root / "trajectory/scenario/realized_trace.csv"
            runtime = root / "trajectory/recipient/runtime/ego_states.csv"
            truth.parent.mkdir(parents=True)
            runtime.parent.mkdir(parents=True)
            truth.write_text("frame_id,x\n1,2\n", encoding="utf-8")
            runtime.write_text("frame_id,x\n1,3\n", encoding="utf-8")
            before = _capture_artifact_fingerprints(root)
            self.assertIn("trajectory/scenario/realized_trace.csv", before)
            self.assertIn("trajectory/recipient/runtime/ego_states.csv", before)
            truth.write_text("frame_id,x\n1,999\n", encoding="utf-8")
            after = _capture_artifact_fingerprints(root)
            drift = _fingerprint_drift(before, after)
            self.assertEqual(
                drift["changed"], ["trajectory/scenario/realized_trace.csv"]
            )

    def test_result_defining_dependencies_are_complete_and_hashable(self) -> None:
        paths = _result_defining_dependency_paths(CONFIG)
        required = {
            "analysis_config",
            "replay_calibration_grid.py",
            "data_collection/phase2_causal_runtime.py",
            "phase2_map_sharing/causal_contract.py",
            "phase2_map_sharing/replay_paired_pilot.py",
            "phase2_map_sharing/adjudicate_future_hazards.py",
            "phase2_map_sharing/engine_v2.py",
            "phase2_map_sharing/schemas_v2.py",
            "pole_lraspp_multimodal_fusion/object_targets.py",
        }
        self.assertTrue(required.issubset(paths))
        fingerprints = _named_path_fingerprints(paths, require_files=True)
        for label in required:
            self.assertEqual(len(str(fingerprints[label]["sha256"])), 64)

    def test_candidate_diagnostic_uses_declared_cadence(self) -> None:
        metrics = [
            {
                "setting_id": "s",
                "arm_id": "ego_only",
                "scenario_role": "matched_benign_negative",
                "eligible_full_horizon_frame_count": 10,
                "false_warning_active_frame_count": 1,
                "false_warning_episode_count": 1,
                "application_bytes": 0,
                "warning_emission_confidence_floor": 0.05,
                "map_association_gate_m": 2.0,
                "map_track_ttl_s": 0.5,
                "warning_uncertainty_multiplier": 0.0,
            }
        ]
        row = _candidate_diagnostics(metrics, cadence_s=0.2)[0]
        self.assertAlmostEqual(
            row["suite_a_benign_false_warning_episodes_per_minute"], 30.0
        )

    def test_first_target_warning_requires_truth_positive_hazard(self) -> None:
        config = load_replay_config(CONFIG)
        metric = {
            "setting_id": "s",
            "trajectory_id": "positive",
            "arm_id": "ego_only",
            "scenario_role": "controlled_positive_occlusion",
        }
        base = {
            "setting_id": "s",
            "trajectory_id": "positive",
            "arm_id": "ego_only",
            "false_warning_adjudicated": 0,
            "future_label": "truth_hazard_positive",
            "future_truth_censored": 0,
            "current_truth_matched": 1,
            "target_hazard_match_adjudicated": 1,
        }
        adjudicated = [
            {
                **base,
                "frame_id": 1,
                "warning_at_s": 1.0,
                "truth_hazard_positive": 0,
            },
            {
                **base,
                "frame_id": 2,
                "warning_at_s": 2.0,
                "truth_hazard_positive": 1,
            },
        ]
        contexts = {
            "positive": {
                "last_truth_s": 10.0,
                "frame_times": pd.DataFrame(
                    {"frame_id": [1, 2], "carla_timestamp": [1.0, 2.0]}
                ),
                "target_prefix": "phase2_registered_target_",
                "hazard_ego_basis": "test_counterfactual",
                "counterfactual_source_trajectory_id": "benign",
            }
        }

        result = _enrich_arm_metrics([metric], adjudicated, contexts, config)[0]

        self.assertEqual(result["first_registered_target_warning_s"], 2.0)
        self.assertEqual(result["missed_registered_target"], 0)

    def test_helper_and_recipient_keep_distinct_resolved_config_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"shared model")
            role_dirs = {}
            for role in ("helper", "recipient"):
                role_dir = root / role
                manifests = role_dir / "manifests"
                manifests.mkdir(parents=True)
                (manifests / f"{role}_manifest.json").write_text(
                    json.dumps({"checkpoint_path": str(checkpoint)}),
                    encoding="utf-8",
                )
                (manifests / f"{role}_resolved_config.json").write_text(
                    json.dumps({"role": role}), encoding="utf-8"
                )
                role_dirs[role] = role_dir
            provenance = _paired_role_provenance(
                role_dirs["helper"], role_dirs["recipient"]
            )
            self.assertEqual(
                provenance["helper"]["model_sha256"],
                provenance["recipient"]["model_sha256"],
            )
            self.assertNotEqual(
                provenance["helper"]["config_sha256"],
                provenance["recipient"]["config_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
