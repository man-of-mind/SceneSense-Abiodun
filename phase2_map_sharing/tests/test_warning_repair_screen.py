from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from phase2_map_sharing.replay_calibration_grid import load_replay_config
from phase2_map_sharing.replay_warning_repair_screen import (
    DEFAULT_CONFIG,
    DEFAULT_REPLAY_CONFIG,
    EXPECTED_SETTING_ID,
    _screen_setting,
    load_config,
)


class WarningRepairScreenTests(unittest.TestCase):
    def test_checked_in_config_is_the_single_preregistered_screen(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        replay_config = load_replay_config(DEFAULT_REPLAY_CONFIG)
        setting = _screen_setting(config, replay_config)

        self.assertEqual(setting["setting_id"], EXPECTED_SETTING_ID)
        self.assertEqual(config["source_tracker"]["minimum_confirmation_hits"], 2)
        self.assertFalse(config["source_tracker"]["publish_missed_tracks"])
        self.assertEqual(
            config["screen_gates"][
                "maximum_suite_a_benign_false_warning_active_frame_rate"
            ],
            0.10,
        )

    def test_scope_and_setting_drift_fail_closed(self) -> None:
        payload = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "screen.yaml"
            expanded = copy.deepcopy(payload)
            expanded["authorization"] = "collect_carla"
            path.write_text(yaml.safe_dump(expanded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not authorized"):
                load_config(path)

            tuned = copy.deepcopy(payload)
            tuned["recipient_map"]["association_gate_m"] = 4.0
            path.write_text(yaml.safe_dump(tuned), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recipient_map.*drifted"):
                load_config(path)

            stale = copy.deepcopy(payload)
            stale["source_tracker"]["publish_missed_tracks"] = True
            path.write_text(yaml.safe_dump(stale), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_tracker.*drifted"):
                load_config(path)

            tracker_tuned = copy.deepcopy(payload)
            tracker_tuned["source_tracker"]["velocity_smoothing_alpha"] = 0.25
            path.write_text(yaml.safe_dump(tracker_tuned), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_tracker.*drifted"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
