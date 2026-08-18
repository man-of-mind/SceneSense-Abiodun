from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_collection.run_phase2_renderer_dense_confirmation import (
    build_launch_spec,
    load_stage_config,
)
from data_collection.run_policy_corpus import _effective_options, _resolved_run_args


class RendererDenseConfirmationTests(unittest.TestCase):
    def test_both_quality_overlays_lock_the_same_short_exact_contract(self) -> None:
        effective_by_quality = {}
        for quality in ("Low", "Epic"):
            _path, config = load_stage_config(quality)
            effective_by_quality[quality] = config
            self.assertEqual(
                config["renderer_quality"]["required_server_launch_flag"],
                f"-quality-level={quality}",
            )
            self.assertTrue(config["carla"]["reload_world_before_run"])
            self.assertEqual(len(config["runs"]), 2)
            self.assertFalse(config["authorization"]["full_collection"])
            for run in config["runs"]:
                options = _effective_options(_resolved_run_args(config, run))
                self.assertEqual(options["--fps"], "10")
                self.assertEqual(options["--world-tick-hz"], "10")
                self.assertEqual(options["--camera-width"], "1280")
                self.assertEqual(options["--camera-height"], "720")
                self.assertEqual(options["--camera-fov"], "120")
                self.assertEqual(options["--radar-points-per-second"], "200000")
                self.assertEqual(options["--radar-raster-radius-px"], "4")
                self.assertEqual(options["--radar-temporal-window-frames"], "2")
                self.assertEqual(options["--object-nms-radius-px"], "2")
                self.assertEqual(options["--topk-objects"], "120")
                self.assertEqual(options["--enable-semantic-gt"], "true")

        low = effective_by_quality["Low"]
        epic = effective_by_quality["Epic"]
        for key in ("common_args", "family_args", "runs", "evaluation"):
            self.assertEqual(low[key], epic[key])

    def test_populations_match_final_mprime_training_lineage(self) -> None:
        _path, config = load_stage_config("Low")
        expected = {"medium": (20, 25), "crowded": (28, 35)}
        for family, counts in expected.items():
            args = [str(value) for value in config["family_args"][family]]
            options = {args[index]: args[index + 1] for index in range(0, len(args), 2)}
            self.assertEqual(
                (int(options["--npc-vehicles"]), int(options["--npc-pedestrians"])),
                counts,
            )

    def test_detached_launch_spec_is_one_quality_and_two_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = build_launch_spec(
                "Low", output_root=Path(temporary), timestamp="20260818_020000"
            )
        self.assertEqual(spec["declared_renderer_quality"], "Low")
        self.assertEqual(spec["run_count"], 2)
        self.assertEqual(spec["scenario_families"], ["medium", "crowded"])
        self.assertIn("--run-live-internal", spec["command"])
        self.assertNotIn("--launch-detached", spec["command"])


if __name__ == "__main__":
    unittest.main()
