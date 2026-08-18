from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from data_collection.analyze_phase2_renderer_quality_gate import (
    _one_to_one_counts,
    canonical_prediction_class,
)
from data_collection.run_phase2_renderer_quality_gate import (
    DEFAULT_CONFIG,
    build_launch_spec,
    load_gate_config,
    resolve_stage,
)


class RendererQualityGateTests(unittest.TestCase):
    def test_checked_in_gate_is_short_carla_only_and_exact_contract(self) -> None:
        gate = load_gate_config(DEFAULT_CONFIG)
        self.assertEqual(
            gate["comparison"]["required_quality_levels"], ["Low", "Epic"]
        )
        self.assertFalse(gate["authorization"]["oai_launch"])
        self.assertFalse(gate["authorization"]["full_collection"])
        self.assertEqual(gate["capture"]["frames_per_trajectory"], 120)
        self.assertEqual(gate["capture"]["retained_raw_window_seconds"], 8.0)
        self.assertEqual(gate["capture"]["aggregate_raw_bytes_cap"], 16000000000)

    def test_stage_resolution_records_quality_and_reuses_reviewed_collector(self) -> None:
        _gate, effective, source, contract = resolve_stage(DEFAULT_CONFIG, "Epic")
        self.assertEqual(
            effective["renderer_quality"]["required_server_launch_flag"],
            "-quality-level=Epic",
        )
        self.assertEqual(contract["sensor_contract"], "exact_training_contract")
        self.assertEqual(len(effective["trajectories"]), 2)
        self.assertTrue(
            all(
                item["trajectory_id"].startswith("renderer_epic_")
                for item in effective["trajectories"]
            )
        )
        self.assertIsInstance(source, dict)

    def test_detached_spec_fails_closed_to_one_declared_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = build_launch_spec(
                DEFAULT_CONFIG,
                "Low",
                output_root=Path(temporary),
                timestamp="20260817_230000",
            )
        self.assertEqual(spec["declared_renderer_quality"], "Low")
        self.assertEqual(spec["required_server_launch_flag"], "-quality-level=Low")
        self.assertEqual(spec["trajectory_count"], 2)
        self.assertFalse(spec["quality_empirically_introspected"])
        self.assertIn("--run-live-internal", spec["command"])
        self.assertNotIn("--launch", spec["command"])

    def test_gate_rejects_quality_or_server_flag_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "renderer quality"):
            resolve_stage(DEFAULT_CONFIG, "High")
        payload = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        payload["comparison"]["server_launch_flag_by_quality"]["Epic"] = (
            "-quality-level=High"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "drifted.yaml"
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "launch flags"):
                load_gate_config(path)

    def test_center_matching_is_one_to_one(self) -> None:
        truth = pd.DataFrame(
            [
                {"origin_x": 0.0, "origin_y": 0.0, "origin_z": 0.0},
                {"origin_x": 9.0, "origin_y": 0.0, "origin_z": 0.0},
            ]
        )
        predictions = pd.DataFrame(
            [
                {"world_x": 1.0, "world_y": 0.0, "world_z": 0.0},
                {"world_x": 2.0, "world_y": 0.0, "world_z": 0.0},
            ]
        )
        self.assertEqual(_one_to_one_counts(truth, predictions, 5.0), (1, 1, 1))

    def test_detector_person_label_is_canonicalized_to_pedestrian_gt(self) -> None:
        self.assertEqual(canonical_prediction_class("person"), "pedestrian")
        self.assertEqual(canonical_prediction_class("pedestrian"), "pedestrian")
        self.assertEqual(canonical_prediction_class("vehicle"), "vehicle")
        with self.assertRaisesRegex(ValueError, "unexpected detector class"):
            canonical_prediction_class("cyclist")


if __name__ == "__main__":
    unittest.main()
