from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from rl_agent import generate_network_profile_meeting_figures as profiles


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "network_profile_design_v1.json"
)


class NetworkProfileMeetingFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        profiles.validate_config(cls.config)
        cls.traces = {
            profile["profile_id"]: profiles.generate_trace(profile, cls.config)
            for profile in cls.config["profiles"]
        }

    def test_route_has_exactly_624_half_open_100_ms_intervals(self) -> None:
        for trace in self.traces.values():
            self.assertEqual(len(trace["snr"]), 624)
            self.assertEqual(len(trace["times"]), 624)
            self.assertEqual(len(trace["edges"]), 625)
            self.assertAlmostEqual(float(trace["times"][-1]), 62.3)
            self.assertAlmostEqual(float(trace["edges"][-1]), 62.4)

    def test_fixed_seed_traces_are_deterministic_and_bounded(self) -> None:
        lower = float(self.config["target_snr"]["lower_bound_db"])
        upper = float(self.config["target_snr"]["upper_bound_db"])
        for profile in self.config["profiles"]:
            first = self.traces[profile["profile_id"]]
            second = profiles.generate_trace(profile, self.config)
            np.testing.assert_array_equal(first["state_index"], second["state_index"])
            np.testing.assert_array_equal(first["snr"], second["snr"])
            self.assertTrue(np.all(first["snr"] >= lower))
            self.assertTrue(np.all(first["snr"] <= upper))

    def test_shifted_gaussian_parameters_follow_normalized_band_rule(self) -> None:
        target = self.config["target_snr"]
        lower = float(target["lower_bound_db"])
        upper = float(target["upper_bound_db"])
        width = upper - lower
        np.testing.assert_allclose(
            target["state_means_db"],
            [lower + 0.15 * width, lower + 0.50 * width, lower + 0.85 * width],
        )
        np.testing.assert_allclose(target["state_sigma_db"], [0.06 * width] * 3)

    def test_stationary_occupancies_and_expected_dwell_are_exact(self) -> None:
        expected = {
            "FAVORABLE_STABLE": ([0.04, 0.16, 0.80], [0.5, 2.0 / 3.0, 5.0]),
            "MID_VARIABLE": ([0.20, 0.60, 0.20], [2.0 / 3.0, 1.0, 2.0 / 3.0]),
            "ADVERSE_STABLE": ([0.80, 0.16, 0.04], [5.0, 2.0 / 3.0, 0.5]),
            "FADE_RECOVERY": ([0.20, 0.60, 0.20], [10.0 / 3.0, 5.0, 10.0 / 3.0]),
        }
        for profile_id, (stationary, dwell) in expected.items():
            trace = self.traces[profile_id]
            np.testing.assert_allclose(trace["stationary"], stationary, atol=1e-12)
            np.testing.assert_allclose(
                0.1 / (1.0 - np.diag(trace["matrix"])), dwell, atol=1e-12
            )

    def test_mid_and_fade_have_same_marginal_but_five_times_slower_transitions(self) -> None:
        mid = self.traces["MID_VARIABLE"]
        fade = self.traces["FADE_RECOVERY"]
        identity = np.eye(3)
        np.testing.assert_allclose(mid["stationary"], fade["stationary"], atol=1e-12)
        np.testing.assert_allclose(
            fade["matrix"], identity + (mid["matrix"] - identity) / 5.0, atol=1e-12
        )
        target = self.config["target_snr"]
        x = np.linspace(target["lower_bound_db"], target["upper_bound_db"], 1000)
        mid_density, _ = profiles.mixture_pdf(
            x,
            mid["stationary"],
            target["state_means_db"],
            target["state_sigma_db"],
            target["lower_bound_db"],
            target["upper_bound_db"],
        )
        fade_density, _ = profiles.mixture_pdf(
            x,
            fade["stationary"],
            target["state_means_db"],
            target["state_sigma_db"],
            target["lower_bound_db"],
            target["upper_bound_db"],
        )
        np.testing.assert_allclose(mid_density, fade_density, atol=1e-12)

    def test_expected_switch_counts_span_stable_variable_and_fade_cases(self) -> None:
        expected = {
            "FAVORABLE_STABLE": 29.904,
            "MID_VARIABLE": 74.76,
            "ADVERSE_STABLE": 29.904,
            "FADE_RECOVERY": 14.952,
        }
        for profile_id, count in expected.items():
            trace = self.traces[profile_id]
            change_probability = 1.0 - float(
                np.dot(trace["stationary"], np.diag(trace["matrix"]))
            )
            self.assertAlmostEqual(change_probability * 623, count, places=9)


if __name__ == "__main__":
    unittest.main()
