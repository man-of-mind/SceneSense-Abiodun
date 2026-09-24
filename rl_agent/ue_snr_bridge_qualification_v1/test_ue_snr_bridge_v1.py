#!/usr/bin/env python3
"""Offline tests for the UE/gNB SNR bridge qualification.

CPU only. No OAI, no Docker, no CARLA, no network, no live evidence required
except one bounded read-only check of the captured run when it is present.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyzer = _load("bridge_analyzer", "ue_snr_bridge_analyzer_v1.py")
CONFIG = json.loads((HERE / "config_v1.json").read_text())


class LabellingTests(unittest.TestCase):
    """The two link directions must never share a label."""

    def test_labels_name_their_link_direction(self):
        self.assertIn("downlink", analyzer.UE_LABEL.lower())
        self.assertIn("uplink", analyzer.GNB_LABEL.lower())
        self.assertNotEqual(analyzer.UE_LABEL, analyzer.GNB_LABEL)

    def test_neither_label_is_bare_snr(self):
        for label in (analyzer.UE_LABEL, analyzer.GNB_LABEL):
            self.assertNotEqual(label.strip().lower(), "snr")

    def test_ue_label_is_never_called_uplink(self):
        self.assertNotIn("uplink", analyzer.UE_LABEL.lower())


class SnrUnitTests(unittest.TestCase):
    """The x10 asymmetry between the two sources must be handled explicitly."""

    def test_gnb_snrx10_is_divided_by_ten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ttracer/gnb/csv/GNB_MAC_PUSCH_POWER_CONTROL.csv"
            path.parent.mkdir(parents=True)
            path.write_text(
                ",".join(analyzer.PUSCH_HEADER) + "\n"
                "00:00:01.000000,1,0,0,165,0,0,0,0,0,9,0\n")
            samples = analyzer.load_gnb_samples(root, 1_000_000_000, 0)
            self.assertEqual(len(samples), 1)
            self.assertAlmostEqual(samples[0].value, 16.5)

    def test_ue_snr_is_taken_as_plain_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ttracer/ue/csv/UE_PHY_MEAS.csv"
            path.parent.mkdir(parents=True)
            path.write_text(
                ",".join(analyzer.UE_PHY_MEAS_HEADER) + "\n"
                "00:00:01.000000,0,0,0,-1,-183,-93,-90,3,-93,6\n")
            samples = analyzer.load_ue_samples(root, 1_000_000_000, 0)
            self.assertEqual(samples[0].value, -93.0)

    def test_wrong_header_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("time,snr\n00:00:01.000000,5\n")
            with self.assertRaises(analyzer.AnalysisError):
                analyzer.read_exact_csv(path, analyzer.UE_PHY_MEAS_HEADER)


class AlignmentTests(unittest.TestCase):
    def _samples(self, times, values):
        return [analyzer.Sample(wall_ns=t, value=v) for t, v in zip(times, values)]

    def test_pairs_only_inside_the_window(self):
        ue = self._samples([1_000_000_000], [10.0])
        far = self._samples([1_000_000_000 + 10_000_000], [20.0])
        out = analyzer.align_nearest(ue, far)
        self.assertEqual(out["paired"], 0)
        self.assertEqual(out["unmatched_ue_samples"], 1)

    def test_unmatched_samples_are_never_forward_filled(self):
        ue = self._samples([0, 10_000_000_000], [1.0, 2.0])
        gnb = self._samples([0], [5.0])
        out = analyzer.align_nearest(ue, gnb)
        self.assertEqual(out["paired"], 1)
        self.assertEqual(out["unmatched_ue_samples"], 1)
        self.assertFalse(out["forward_fill_used"])

    def test_equidistant_counterparts_are_flagged_ambiguous(self):
        ue = self._samples([1_000_000], [1.0])
        gnb = self._samples([0, 2_000_000], [5.0, 6.0])
        out = analyzer.align_nearest(ue, gnb)
        self.assertEqual(out["ambiguous_matches"], 1)

    def test_pair_distance_is_reported(self):
        ue = self._samples([1_000_000], [1.0])
        gnb = self._samples([1_500_000], [5.0])
        out = analyzer.align_nearest(ue, gnb)
        self.assertEqual(out["pair_distance_ns"]["p50"], 500_000.0)


class AvailabilityTests(unittest.TestCase):
    def test_coverage_counts_only_bins_with_an_observation(self):
        # Two 100 ms bins; only the first holds a sample.
        ue = [analyzer.Sample(wall_ns=50_000_000, value=1.0)]
        out = analyzer.decision_bin_coverage(ue, 0, 200_000_000, 100)
        self.assertEqual(out["bins"], 2)
        self.assertEqual(out["bins_with_observation"], 1)
        self.assertAlmostEqual(out["coverage"], 0.5)

    def test_age_never_uses_a_future_sample(self):
        # The only sample lands after the first boundary, so that boundary has
        # no prior observation and must not borrow the later one.
        ue = [analyzer.Sample(wall_ns=150_000_000, value=1.0)]
        out = analyzer.decision_bin_coverage(ue, 0, 200_000_000, 100)
        self.assertEqual(out["bins_with_no_prior_observation"], 1)
        self.assertEqual(out["observation_age_ms"]["count"], 1)

    def test_empty_window_reports_missing_not_zero(self):
        out = analyzer.decision_bin_coverage([], 10, 10, 100)
        self.assertIsNone(out["coverage"])


class StatisticsTests(unittest.TestCase):
    def test_pearson_is_missing_for_a_constant_series(self):
        self.assertIsNone(analyzer.pearson([1.0] * 10, list(range(10))))

    def test_spearman_recovers_a_monotone_but_nonlinear_relation(self):
        xs = [float(v) for v in range(1, 11)]
        ys = [v ** 3 for v in xs]
        self.assertAlmostEqual(analyzer.spearman(xs, ys), 1.0, places=6)

    def test_block_bootstrap_refuses_a_too_short_series(self):
        out = analyzer.block_bootstrap_ci([1.0, 2.0], [1.0, 2.0],
                                          analyzer.pearson, block=10)
        self.assertIsNone(out["ci_low"])

    def test_block_bootstrap_interval_brackets_the_estimate(self):
        xs = [float(v % 17) for v in range(400)]
        ys = [v + 0.5 for v in xs]
        out = analyzer.block_bootstrap_ci(xs, ys, analyzer.pearson, block=10,
                                          iterations=300)
        self.assertLessEqual(out["ci_low"], 1.0)
        self.assertGreaterEqual(out["ci_high"], 0.9)

    def test_cliffs_delta_separates_and_reports_overlap(self):
        out = analyzer.cliffs_delta([10.0] * 50, [1.0] * 50)
        self.assertEqual(out["delta"], 1.0)
        self.assertEqual(out["interpretation"], "LARGE")
        identical = analyzer.cliffs_delta([5.0] * 50, [5.0] * 50)
        self.assertEqual(identical["delta"], 0.0)
        self.assertEqual(identical["interpretation"], "NEGLIGIBLE")


class ClockTests(unittest.TestCase):
    def test_dateless_tracer_time_is_dated_from_the_anchor(self):
        import datetime as dt
        reference = dt.datetime(2026, 9, 23, 22, 3, 55).astimezone()
        ref_ns = int(reference.timestamp() * 1e9)
        out = analyzer.tracer_time_to_wall_ns("22:03:56.000000", ref_ns, 0)
        self.assertAlmostEqual((out - ref_ns) / 1e9, 1.0, places=3)

    def test_midnight_rollover_picks_the_nearest_day(self):
        import datetime as dt
        reference = dt.datetime(2026, 9, 23, 23, 59, 59).astimezone()
        ref_ns = int(reference.timestamp() * 1e9)
        out = analyzer.tracer_time_to_wall_ns("00:00:01.000000", ref_ns, 0)
        self.assertAlmostEqual((out - ref_ns) / 1e9, 2.0, places=3)


class ProfileDesignTests(unittest.TestCase):
    """The registered profiles must be used, not replacements."""

    def test_exactly_the_four_registered_profiles(self):
        self.assertEqual(CONFIG["replay"]["profile_order"],
                         ["FAVORABLE_STABLE", "MID_VARIABLE",
                          "ADVERSE_STABLE", "FADE_RECOVERY"])

    def test_one_constant_offered_load_for_every_profile(self):
        traffic = CONFIG["traffic"]
        # A single rate pair and datagram size, not a per-profile table.
        for key in ("uplink_mbps", "downlink_mbps", "datagram_bytes"):
            self.assertIsInstance(traffic[key], (int, float))

    def test_measured_window_is_a_short_qualification(self):
        replay = CONFIG["replay"]
        measured_s = replay["measured_samples"] * replay["sample_period_s"]
        self.assertGreaterEqual(measured_s, 20.0)
        self.assertLessEqual(measured_s, 30.0)

    def test_ue_phy_meas_is_recorded(self):
        self.assertIn("UE_PHY_MEAS", CONFIG["telemetry"]["events"]["ue"])


class CapturedRunTests(unittest.TestCase):
    """Bounded read-only checks of the captured evidence, when present."""

    @classmethod
    def setUpClass(cls):
        root = REPO / CONFIG["paths"]["output_root"]
        runs = sorted(p for p in root.glob("*") if (p / "analysis_v1.json").is_file()) \
            if root.is_dir() else []
        cls.run_dir = runs[-1] if runs else None

    def setUp(self):
        if self.run_dir is None:
            self.skipTest("no analyzed run present")

    def test_verdict_is_one_of_the_four_registered_outcomes(self):
        report = json.loads((self.run_dir / "analysis_v1.json").read_text())
        self.assertIn(report["interpretation"]["verdict"], {
            "QUALIFIED_AS_UE_POLICY_CHANNEL_SIGNAL",
            "PROMISING_BUT_REQUIRES_DIRECT_CARRIER_VALIDATION",
            "NOT_SUPPORTED_AS_UPLINK_PREDICTOR",
            "EXPERIMENT_INCONCLUSIVE",
        })

    def test_scope_refuses_a_direct_uplink_measurement_claim(self):
        report = json.loads((self.run_dir / "analysis_v1.json").read_text())
        scope = report["interpretation"]["scope"]
        self.assertIn("NOT a direct measurement", scope)

    def test_channel_was_restored_and_read_back(self):
        restore = json.loads((self.run_dir / "restore_readback.json").read_text())
        self.assertTrue(restore["verified"])
        self.assertEqual(restore["commanded_clean_noise_power_db"],
                         restore["read_back_noise_power_db"])

    def test_host_returned_to_cold_state(self):
        cold = json.loads((self.run_dir / "final_cold_state.json").read_text())
        self.assertTrue(cold["cold"])
        self.assertEqual(cold["residual_ue_tunnels"], [])

    def test_no_profile_clamped_its_targets(self):
        report = json.loads((self.run_dir / "analysis_v1.json").read_text())
        for name, block in report["per_profile"].items():
            self.assertEqual(block["targets_clamped_to_mapping"], 0, name)


if __name__ == "__main__":
    unittest.main()
