"""Phase-A tests for the Run-4 near-capacity sweep.

Four things must be true before any radio is started: the design is the
balanced, partitioned design it claims to be; the three actions are exactly the
registered ones; output is create-only; and the completed Run-3 campaign is
byte-identical. Everything here runs offline and starts nothing.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from rl_agent.ue_mcs_backlog_near_capacity_v1 import analysis_spec as S
from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE

ROOT = Path(__file__).resolve().parents[2]
PORTS = {"low": 5401, "medium": 5402, "high": 5403}


def build_plan():
    return C.build_cell_plan(C.resolve_load_tiers(ROOT), ports=PORTS)


class CatalogIdentityTests(unittest.TestCase):
    """The action authority is reconciled, never assumed."""

    def test_catalog_digests_match_the_pinned_authority(self):
        self.assertEqual(
            C.sha256_file(ROOT / C.ACTION_CATALOG_RELPATH),
            C.ACTION_CATALOG_JSON_SHA256)
        self.assertEqual(
            C.sha256_file(ROOT / C.ACTION_CATALOG_CSV_RELPATH),
            C.ACTION_CATALOG_CSV_SHA256)

    def test_the_three_registered_actions_resolve_exactly(self):
        tiers = {t.tier: t for t in C.resolve_load_tiers(ROOT)}
        self.assertEqual(
            {(t.tier, t.action_id, t.profile_id, t.payload_bytes,
              t.chunks_per_frame) for t in tiers.values()},
            {("low", 70, "split_ae32_uint4_q9000", 28_109, 1),
             ("medium", 69, "split_ae32_uint4_q7000", 81_087, 2),
             ("high", 68, "split_ae32_uint4_q5000", 129_707, 3)})

    def test_all_three_share_the_pinned_checkpoint(self):
        for tier in C.resolve_load_tiers(ROOT):
            self.assertEqual(tier.checkpoint_sha256, C.SHARED_CHECKPOINT_SHA256)

    def test_offered_rates_are_the_registered_rates(self):
        rates = {t.tier: round(t.offered_mbps, 2)
                 for t in C.resolve_load_tiers(ROOT)}
        self.assertEqual(rates, {"low": 2.25, "medium": 6.49, "high": 10.38})

    def test_payload_is_strictly_increasing_low_to_high(self):
        tiers = C.resolve_load_tiers(ROOT)
        self.assertEqual([t.tier for t in tiers], ["low", "medium", "high"])
        sizes = [t.payload_bytes for t in tiers]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), 3)

    def test_a_profile_id_mismatch_refuses_rather_than_guesses(self):
        original = dict(C.EXPECTED_TIERS["medium"])
        try:
            C.EXPECTED_TIERS["medium"]["profile_id"] = "split_ae64_uint4_q5000"
            with self.assertRaises(C.ContractError):
                C.resolve_load_tiers(ROOT)
        finally:
            C.EXPECTED_TIERS["medium"].update(original)

    def test_a_payload_mismatch_refuses(self):
        original = dict(C.EXPECTED_TIERS["high"])
        try:
            C.EXPECTED_TIERS["high"]["payload_bytes"] = 129_708
            with self.assertRaises(C.ContractError):
                C.resolve_load_tiers(ROOT)
        finally:
            C.EXPECTED_TIERS["high"].update(original)

    def test_a_chunk_count_mismatch_refuses(self):
        original = dict(C.EXPECTED_TIERS["low"])
        try:
            C.EXPECTED_TIERS["low"]["chunks"] = 2
            with self.assertRaises(C.ContractError):
                C.resolve_load_tiers(ROOT)
        finally:
            C.EXPECTED_TIERS["low"].update(original)

    def test_a_catalog_digest_mismatch_refuses(self):
        original = C.ACTION_CATALOG_JSON_SHA256
        try:
            C.ACTION_CATALOG_JSON_SHA256 = "0" * 64
            with self.assertRaises(C.ContractError):
                C.resolve_load_tiers(ROOT)
        finally:
            C.ACTION_CATALOG_JSON_SHA256 = original


class DesignBalanceTests(unittest.TestCase):
    """Balance is proven from the built plan, not asserted in prose."""

    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan()
        cls.audit = C.audit_cell_plan(cls.plan)

    def test_twelve_cells_and_5400_decisions(self):
        self.assertEqual(self.audit["cells"], 12)
        self.assertEqual(self.audit["decisions"], 5400)
        self.assertEqual(len(self.plan), C.EXPECTED_CELLS)

    def test_all_six_permutations_appear_under_each_channel(self):
        for profile in C.CONTRAST_PROFILE_IDS:
            seqs = {c.sequence for c in self.plan if c.profile_id == profile}
            self.assertEqual(seqs, set(C.PERMUTATIONS))
            self.assertEqual(len(seqs), 6)

    def test_every_cell_contains_every_tier(self):
        self.assertTrue(self.audit["load_is_within_cell"])
        for cell in self.plan:
            self.assertEqual(set(cell.sequence), set(C.TIER_ORDER))

    def test_every_payload_occupies_every_block_position_per_channel(self):
        self.assertTrue(all(self.audit["position_balanced_per_channel"].values()))
        for profile in C.CONTRAST_PROFILE_IDS:
            counts = self.audit["position_counts_per_channel"][profile]
            self.assertEqual(len(counts), 9)
            self.assertEqual(set(counts.values()), {2})

    def test_every_transition_direction_is_represented_twice_per_channel(self):
        self.assertTrue(
            all(self.audit["all_six_transitions_balanced_per_channel"].values()))
        for profile in C.CONTRAST_PROFILE_IDS:
            counts = self.audit["transition_counts_per_channel"][profile]
            self.assertEqual(len(counts), 6)
            self.assertEqual(set(counts.values()), {2})

    def test_fit_and_validation_are_three_and_three_per_channel(self):
        self.assertTrue(all(self.audit["partition_balanced_per_channel"].values()))
        for profile in C.CONTRAST_PROFILE_IDS:
            self.assertEqual(
                self.audit["partition_counts_per_channel"][profile],
                {C.FIT: 3, C.VALIDATION: 3})

    def test_the_registered_fit_orders_are_lmh_mhl_hlm(self):
        self.assertEqual(
            [C.permutation_label(o) for o in C.FIT_PERMUTATIONS],
            ["L-M-H", "M-H-L", "H-L-M"])

    def test_the_validation_orders_are_the_three_reverses(self):
        self.assertEqual(
            sorted(C.permutation_label(o) for o in C.VALIDATION_PERMUTATIONS),
            sorted(["H-M-L", "L-H-M", "M-L-H"]))
        self.assertTrue(self.audit["validation_reverses_fit"])

    def test_fit_and_validation_are_each_a_latin_square(self):
        self.assertTrue(all(self.audit["fit_is_latin_square_per_channel"].values()))
        self.assertTrue(
            all(self.audit["validation_is_latin_square_per_channel"].values()))

    def test_validation_extrapolates_across_transition_direction(self):
        """A declared property of the design, not a surprise found later."""
        self.assertTrue(
            all(self.audit[
                "fit_validation_transitions_disjoint_per_channel"].values()))
        for profile in C.CONTRAST_PROFILE_IDS:
            self.assertEqual(
                set(self.audit["fit_transition_counts_per_channel"][profile]),
                {"low->medium", "medium->high", "high->low"})
            self.assertEqual(
                set(self.audit["validation_transition_counts_per_channel"][profile]),
                {"medium->low", "high->medium", "low->high"})

    def test_plan_is_accepted_by_the_single_runner_predicate(self):
        self.assertTrue(C.plan_is_registered_design(self.audit))

    def test_execution_order_is_seeded_and_reproducible(self):
        again = C.audit_cell_plan(build_plan())
        self.assertEqual([c.cell_id for c in build_plan()],
                         [c.cell_id for c in self.plan])
        self.assertEqual(again["cells"], self.audit["cells"])

    def test_shuffling_never_moves_a_cell_between_partitions(self):
        for cell in self.plan:
            expected = C.PARTITIONS[cell.permutation_index]
            self.assertEqual(cell.partition, expected)
            self.assertEqual(cell.sequence, C.PERMUTATIONS[cell.permutation_index])

    def test_execution_order_interleaves_channels_and_partitions(self):
        """A seeded order is only useful if it is not blocked by design factor."""
        profiles = [c.profile_id for c in self.plan]
        partitions = [c.partition for c in self.plan]
        self.assertNotEqual(profiles, sorted(profiles))
        self.assertNotEqual(partitions, sorted(partitions))

    def test_blocks_carry_the_registered_payloads_and_ports(self):
        tiers = {t.tier: t for t in C.resolve_load_tiers(ROOT)}
        for cell in self.plan:
            self.assertEqual(len(cell.blocks), 3)
            for position, block in enumerate(cell.blocks):
                self.assertEqual(block.block_index, position)
                self.assertEqual(block.tier, cell.sequence[position])
                self.assertEqual(block.payload_bytes,
                                 tiers[block.tier].payload_bytes)
                self.assertEqual(block.action_id, tiers[block.tier].action_id)
                self.assertEqual(block.port, PORTS[block.tier])
                self.assertEqual(block.frames, C.FRAMES_PER_BLOCK)
                self.assertEqual(block.first_frame_index,
                                 position * C.FRAMES_PER_BLOCK)

    def test_cell_ids_are_unique_and_name_their_partition(self):
        ids = [c.cell_id for c in self.plan]
        self.assertEqual(len(set(ids)), 12)
        for cell in self.plan:
            self.assertIn(cell.partition.lower(), cell.cell_id)


class ProtectedEvidenceTests(unittest.TestCase):
    """Run 3 must survive Run 4 byte-for-byte."""

    def test_all_five_protected_files_are_unchanged(self):
        report = PE.audit(ROOT)
        self.assertTrue(report["all_unchanged"], report)
        self.assertEqual(len(report["files"]), 5)
        for name, item in report["files"].items():
            self.assertTrue(item["present"], name)
            self.assertEqual(item["observed_sha256"], item["expected_sha256"], name)

    def test_the_guard_names_the_five_registered_files(self):
        self.assertEqual(set(PE.PROTECTED_SHA256), {
            "manifest.json",
            "INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md",
            "VERIFIER_INPUT_MANIFEST_V2.json",
            "analysis_v2.json",
            "decisions_v2.csv"})

    def test_run3_verdict_is_preserved_as_inconclusive(self):
        self.assertEqual(PE.PROTECTED_VERDICT, "INCONCLUSIVE")
        self.assertIn("4/7", PE.PROTECTED_BOUND_RESULT)
        self.assertIn("12/13", PE.PROTECTED_BOUND_RESULT)

    def test_a_tampered_digest_is_detected_and_refused(self):
        original = PE.PROTECTED_SHA256["analysis_v2.json"]
        try:
            PE.PROTECTED_SHA256["analysis_v2.json"] = "0" * 64  # type: ignore[index]
            self.assertFalse(PE.audit(ROOT)["all_unchanged"])
            with self.assertRaises(PE.ProtectedEvidenceError):
                PE.require_unchanged("test", ROOT)
        finally:
            PE.PROTECTED_SHA256["analysis_v2.json"] = original  # type: ignore[index]

    def test_writing_inside_the_protected_run_is_refused(self):
        protected = ROOT / PE.PROTECTED_RUN_RELPATH
        for candidate in (protected, protected / "analysis_v3.json",
                          protected / "cells" / "anything" / "x.csv"):
            with self.assertRaises(PE.ProtectedEvidenceError):
                PE.assert_outside_protected_run(candidate, ROOT)

    def test_the_run4_output_root_is_outside_the_protected_run(self):
        config = json.loads(
            (Path(__file__).resolve().parent / "config_v1.json").read_text())
        root = ROOT / config["paths"]["output_root"]
        PE.assert_outside_protected_run(root, ROOT)
        self.assertNotIn("20260924_131015", str(root))


class CreateOnlyTests(unittest.TestCase):
    """An existing output root is never reused, extended or overwritten."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nearcap_createonly_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_mkdir_refuses_an_existing_root(self):
        target = self.tmp / "20260924_000000"
        target.mkdir(parents=True, exist_ok=False)
        with self.assertRaises(FileExistsError):
            target.mkdir(parents=True, exist_ok=False)

    def test_runner_main_refuses_an_existing_output_dir(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        target = self.tmp / "existing"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            R.main(["--output-dir", str(target)])
        self.assertEqual(list(target.iterdir()), [],
                         "a refused run must not have written anything")

    def test_runner_main_refuses_a_target_inside_the_protected_run(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        inside = ROOT / PE.PROTECTED_RUN_RELPATH / "run4_attempt"
        with self.assertRaises(PE.ProtectedEvidenceError):
            R.main(["--output-dir", str(inside)])
        self.assertFalse(inside.exists())


class AnalysisSpecTests(unittest.TestCase):
    """The preregistered analysis leaves no post-hoc freedom."""

    def test_payload_levels_are_the_three_registered_payloads(self):
        self.assertEqual(S.PAYLOAD_LEVELS, (28_109, 81_087, 129_707))

    def test_backlog_bin_zero_is_exactly_zero(self):
        self.assertEqual(S.backlog_bin(0), 0)
        self.assertEqual(S.backlog_bin(1), 1)
        self.assertNotEqual(S.backlog_bin(0), S.backlog_bin(1))

    def test_backlog_bins_are_monotone(self):
        values = [0, 1, 500, 5_000, 50_000, 500_000, 5_000_000, 50_000_000]
        bins = [S.backlog_bin(v) for v in values]
        self.assertEqual(bins, sorted(bins))
        self.assertEqual(bins[-1], len(S.BACKLOG_BIN_LABELS) - 1)

    def test_mcs_bins_cover_table0_and_are_monotone(self):
        bins = [S.mcs_bin(v) for v in range(29)]
        self.assertEqual(bins, sorted(bins))
        self.assertEqual(S.mcs_bin(28), len(S.MCS_BIN_LABELS) - 1)

    def test_eight_gates_are_registered_with_the_stated_thresholds(self):
        self.assertEqual([g.number for g in S.GATES], list(range(1, 9)))
        self.assertEqual(S.GATES_BY_KEY["COMPLETE_CAPTURE"].thresholds,
                         {"cells": 12, "decisions": 5400})
        self.assertEqual(
            S.GATES_BY_KEY["VALIDATION_NEXT_BACKLOG_ERROR"].thresholds,
            {"max_nmae": 0.10, "min_improvement_over_persistence": 0.20})
        self.assertEqual(
            S.GATES_BY_KEY["VALIDATION_LATENCY_ERROR"].thresholds,
            {"max_p50_error_ms": 17.0, "max_p95_error_ms": 34.0})
        self.assertEqual(
            S.GATES_BY_KEY["VALIDATION_TRANSPORT_OUTCOME"]
            .thresholds["max_false_success_rate"], 0.05)
        self.assertEqual(
            S.GATES_BY_KEY["MCS_DOES_NOT_HURT_AND_POINTS_THE_RIGHT_WAY"]
            .thresholds["max_brier_degradation"], 0.01)
        self.assertEqual(
            S.GATES_BY_KEY["MONOTONICITY"].thresholds["max_violations"], 0)

    def test_degenerate_arm_never_counts_as_a_pass(self):
        self.assertEqual(
            S.combine_arm_verdicts({"a": S.NON_INFORMATIVE, "b": S.NON_INFORMATIVE}),
            S.INDETERMINATE)
        self.assertEqual(
            S.combine_arm_verdicts({"a": S.NON_INFORMATIVE, "b": S.PASS}), S.PASS)
        self.assertEqual(
            S.combine_arm_verdicts({"a": S.NON_INFORMATIVE, "b": S.FAIL}), S.FAIL)
        self.assertEqual(S.combine_arm_verdicts({"a": S.PASS, "b": S.FAIL}), S.FAIL)

    def test_improvement_over_a_perfect_baseline_is_undefined_not_zero(self):
        self.assertTrue(math.isnan(S.improvement_over_baseline(0.0, 0.0)))
        self.assertAlmostEqual(S.improvement_over_baseline(0.8, 1.0), 0.2)

    def test_nmae_refuses_a_zero_scale(self):
        with self.assertRaises(ValueError):
            S.normalized_median_absolute_error([1.0], [2.0], scale=0.0)

    def test_metrics_on_known_values(self):
        self.assertAlmostEqual(S.brier_score([1.0, 0.0], [1, 0]), 0.0)
        self.assertAlmostEqual(S.brier_score([0.0, 1.0], [1, 0]), 1.0)
        self.assertAlmostEqual(S.percentile([0, 10, 20, 30, 40], 50), 20.0)
        self.assertAlmostEqual(S.percentile([0, 10, 20, 30, 40], 95), 38.0)
        self.assertAlmostEqual(
            S.normalized_median_absolute_error([1, 2, 3], [1, 2, 3], scale=10), 0.0)

    def test_false_success_rate_is_undefined_when_nothing_is_promised(self):
        self.assertTrue(math.isnan(S.false_success_rate([0.1, 0.2], [0, 0])))
        self.assertAlmostEqual(S.false_success_rate([0.9, 0.9], [1, 0]), 0.5)

    def test_queue_recurrence_surfaces_overflow_and_never_goes_negative(self):
        value, overflow = S.next_backlog(0, 28_109, 1e9, backlog_max=5e7)
        self.assertEqual(value, 0.0)
        self.assertFalse(overflow)
        value, overflow = S.next_backlog(5e7, 129_707, 0, backlog_max=5e7)
        self.assertEqual(value, 5e7)
        self.assertTrue(overflow)

    def test_monotonicity_detects_a_wrong_direction_predictor(self):
        grid = dict(payloads=[28_109, 81_087, 129_707],
                    backlogs=[0.0, 1e5, 1e6], mcs_values=[9, 16, 25])
        good = S.monotonicity_violations(
            lambda p, b, m: 1.0 - p / 1e6 - b / 1e8 + m / 1e3,
            higher_is_better=True, **grid)
        self.assertEqual(good, [])
        bad = S.monotonicity_violations(
            lambda p, b, m: p / 1e6, higher_is_better=True, **grid)
        self.assertTrue(any(v["axis"] == "payload" for v in bad))
        worse_mcs = S.monotonicity_violations(
            lambda p, b, m: -m / 1e3, higher_is_better=True, **grid)
        self.assertTrue(any(v["axis"] == "mcs" for v in worse_mcs))


class CapacityPremiseTests(unittest.TestCase):
    """The stated capacity premise is recorded, with its consequence."""

    def test_capacity_anchors_are_recorded_for_both_channels(self):
        self.assertEqual(set(C.MEASURED_CAPACITY_MBPS), set(C.CONTRAST_PROFILE_IDS))
        for stats in C.MEASURED_CAPACITY_MBPS.values():
            self.assertLess(stats["p10"], stats["p50"])
            self.assertLess(stats["p50"], stats["p90"])

    def test_the_adverse_arm_brackets_its_capacity_knee(self):
        ratios = C.expected_load_ratios()["ADVERSE_STABLE"]
        self.assertLess(ratios["low"], 0.25)
        self.assertGreater(ratios["high"], 0.75)
        self.assertLess(ratios["high"], 1.0)

    def test_the_favorable_arm_is_declared_sub_capacity_in_advance(self):
        ratios = C.expected_load_ratios()["FAVORABLE_STABLE"]
        self.assertLess(max(ratios.values()), 0.5)

    def test_pinned_anchors_reproduce_from_run3_evidence(self):
        """The capacity claim is re-derivable, not a remembered number."""
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import (
            capacity_rederivation as CR)
        derived = CR.rederive(ROOT)["capacity_mbps"]
        self.assertEqual(set(derived), set(C.MEASURED_CAPACITY_MBPS))
        for profile, pinned in C.MEASURED_CAPACITY_MBPS.items():
            for key in ("p10", "p50", "p90", "n"):
                self.assertAlmostEqual(
                    derived[profile][key], pinned[key], places=2,
                    msg=f"{profile}.{key} drifted from the pinned anchor")

    def test_the_superseded_6mbps_assumption_is_recorded(self):
        self.assertEqual(C.SUPERSEDED_CAPACITY_ASSUMPTION_MBPS, 6.0)
        for stats in C.MEASURED_CAPACITY_MBPS.values():
            self.assertGreater(stats["p50"], C.SUPERSEDED_CAPACITY_ASSUMPTION_MBPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
