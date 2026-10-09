"""Focused tests for the exact-P95 Run-2 presentation renderer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from .render_run2_presentation_v1 import (
        FIXED,
        ORACLE,
        PROFILES,
        RUN1,
        RUN2,
        SEEDS,
        aggregate_row,
        context_rows,
        load_inputs,
        render,
    )
except ImportError:  # Direct file execution.
    from render_run2_presentation_v1 import (  # type: ignore
        FIXED,
        ORACLE,
        PROFILES,
        RUN1,
        RUN2,
        SEEDS,
        aggregate_row,
        context_rows,
        load_inputs,
        render,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    PROJECT_ROOT
    / "experiments/splitfusion_hybrid_sac_run2_evaluation_v2/20260922_exact_p95_run2_v2"
)


class Run2PresentationRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs(SOURCE)

    def test_hash_bound_source_shapes(self) -> None:
        self.assertEqual(len(self.inputs.aggregate), 270)
        self.assertEqual(len(self.inputs.contexts), 13940)
        self.assertEqual(set(self.inputs.source_hashes), {
            "aggregate.csv",
            "per_context.csv",
            "success_tiers.json",
            "adaptation.json",
            "training_metrics_seed_17.csv",
            "training_metrics_seed_29.csv",
            "training_metrics_seed_43.csv",
        })

    def test_registered_headline_values(self) -> None:
        final = aggregate_row(self.inputs, RUN2, -1, 5000, "ALL_PROFILES")
        run1 = aggregate_row(self.inputs, RUN1, -1, 5000, "ALL_PROFILES")
        fixed = aggregate_row(self.inputs, FIXED, -1, -1, "ALL_PROFILES")
        oracle = aggregate_row(self.inputs, ORACLE, -1, -1, "ALL_PROFILES")
        self.assertAlmostEqual(float(final["v2_utility_mean"]), 0.29709034738879575)
        self.assertEqual(int(final["p95_miss_count"]), 23)
        self.assertAlmostEqual(float(run1["p95_miss_rate"]), 775 / 1020)
        self.assertAlmostEqual(float(fixed["v2_utility_mean"]), 0.31921325210920154)
        self.assertAlmostEqual(float(oracle["q_perc_mean"]), 0.6575026841735581)

    def test_final_misses_are_only_adverse_and_near_boundary(self) -> None:
        misses = []
        for profile in PROFILES:
            profile_misses = []
            for seed in SEEDS:
                rows = context_rows(self.inputs, RUN2, seed, 5000, profile)
                profile_misses.extend(
                    float(row["latency_proxy_p95_ms"])
                    for row in rows
                    if int(row["p95_miss"]) == 1
                )
            if profile == "ADVERSE_STABLE":
                misses = profile_misses
                self.assertEqual(len(misses), 23)
            else:
                self.assertEqual(profile_misses, [])
        self.assertAlmostEqual(min(misses), 200.17408313152734)
        self.assertAlmostEqual(max(misses), 205.19562670335006)

    def test_final_run2_mode_is_ae32_uint4_everywhere(self) -> None:
        modes = []
        for profile in PROFILES:
            for seed in SEEDS:
                modes.extend(
                    int(row["executed_mode_id"])
                    for row in context_rows(self.inputs, RUN2, seed, 5000, profile)
                )
        self.assertEqual(len(modes), 1020)
        self.assertEqual(set(modes), {11})

    def test_q_increases_with_channel_severity_for_stable_profiles(self) -> None:
        means = {}
        for profile in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE"):
            values = []
            for seed in SEEDS:
                values.extend(
                    int(row["executed_q_e4"]) / 10000.0
                    for row in context_rows(self.inputs, RUN2, seed, 5000, profile)
                )
            means[profile] = float(np.mean(values))
        self.assertLess(means["FAVORABLE_STABLE"], means["MID_VARIABLE"])
        self.assertLess(means["MID_VARIABLE"], means["ADVERSE_STABLE"])

    def test_renderer_emits_complete_png_pdf_pack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            manifest = render(self.inputs, output)
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["figure_count"], 12)
            self.assertEqual(len(list(output.glob("*.png"))), 12)
            self.assertEqual(len(list(output.glob("*.pdf"))), 12)
            for path in output.glob("*.png"):
                self.assertGreater(path.stat().st_size, 20_000)
            for path in output.glob("*.pdf"):
                self.assertGreater(path.stat().st_size, 5_000)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["run2_final"]["p95_miss_count"], 23)


if __name__ == "__main__":
    unittest.main()
