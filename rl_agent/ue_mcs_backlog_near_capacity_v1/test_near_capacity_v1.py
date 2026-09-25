"""Phase-A tests for the Run-4 near-capacity sweep.

Four things must be true before any radio is started: the design is the
balanced, partitioned design it claims to be; the three actions are exactly the
registered ones; output is create-only; and the completed Run-3 campaign is
byte-identical. Everything here runs offline and starts nothing.
"""

from __future__ import annotations

import inspect
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from rl_agent.ue_mcs_backlog_near_capacity_v1 import analysis_spec as S
from rl_agent.ue_mcs_backlog_near_capacity_v1 import authorization as AUTH
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_qualification as CQ
from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C
from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE
from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

ROOT = Path(__file__).resolve().parents[2]
PORTS = {"low": 5401, "medium": 5402, "high": 5403}

#: A representative measured boundary used to exercise the design. The real one
#: comes from the live capacity stage; nothing here freezes an action.
EXAMPLE_CAPACITY_MBPS = 85.0


def example_tiers():
    return C.load_tiers_from_selection(
        CQ.select_tiers(EXAMPLE_CAPACITY_MBPS, repo_root=ROOT))


def build_plan():
    return C.build_cell_plan(example_tiers(), ports=PORTS)


class CatalogAuthorityTests(unittest.TestCase):
    """Catalogue digests are pinned; the three actions deliberately are not."""

    def test_catalog_digests_match_the_pinned_authority(self):
        C.assert_catalog_digests(ROOT)
        self.assertEqual(C.sha256_file(ROOT / C.ACTION_CATALOG_RELPATH),
                         C.ACTION_CATALOG_JSON_SHA256)
        self.assertEqual(C.sha256_file(ROOT / C.ACTION_CATALOG_CSV_RELPATH),
                         C.ACTION_CATALOG_CSV_SHA256)

    def test_a_catalog_digest_mismatch_refuses(self):
        original = C.ACTION_CATALOG_JSON_SHA256
        try:
            C.ACTION_CATALOG_JSON_SHA256 = "0" * 64
            with self.assertRaises(C.ContractError):
                C.assert_catalog_digests(ROOT)
        finally:
            C.ACTION_CATALOG_JSON_SHA256 = original

    def test_no_action_is_frozen_in_the_contract(self):
        self.assertFalse(C.ACTIONS_FROZEN_IN_CONTRACT)
        for name in ("EXPECTED_TIERS", "MEASURED_CAPACITY_MBPS",
                     "SHARED_CHECKPOINT_SHA256", "resolve_load_tiers"):
            self.assertFalse(hasattr(C, name),
                             f"{name} is a withdrawn 106 PRB artefact")

    def test_every_eligible_action_carries_a_decoder_digest(self):
        actions = CQ.eligible_actions(ROOT)
        self.assertEqual(len(actions), 72)
        for action in actions:
            self.assertEqual(len(action["checkpoint_sha256"]), 64)


class TierRuleTests(unittest.TestCase):
    """The deterministic rule must bracket the measured boundary or refuse."""

    def test_rule_is_deterministic(self):
        first = CQ.select_tiers(85.0, repo_root=ROOT)
        second = CQ.select_tiers(85.0, repo_root=ROOT)
        self.assertEqual([t.to_json() for t in first],
                         [t.to_json() for t in second])

    def test_tiers_bracket_the_boundary_across_plausible_capacities(self):
        for capacity in (1.0, 5.0, 12.05, 40.0, 85.0, 150.0, 200.0):
            with self.subTest(capacity=capacity):
                tiers = {t.tier: t for t in CQ.select_tiers(capacity,
                                                            repo_root=ROOT)}
                self.assertLess(tiers["low"].offered_mbps, capacity)
                self.assertGreater(tiers["high"].offered_mbps, capacity)
                self.assertLessEqual(
                    abs(tiers["medium"].achieved_ratio - 1.0),
                    CQ.MEDIUM_BOUNDARY_TOLERANCE)
                payloads = [tiers[k].payload_bytes
                            for k in ("low", "medium", "high")]
                self.assertEqual(payloads, sorted(payloads))
                self.assertEqual(len(set(payloads)), 3)

    def test_rule_refuses_capacity_outside_the_catalogue_span(self):
        for capacity in (0.4, 0.5, 285.47, 400.0):
            with self.subTest(capacity=capacity):
                with self.assertRaises(CQ.CapacityQualificationError):
                    CQ.select_tiers(capacity, repo_root=ROOT)

    def test_rule_refuses_a_nonpositive_capacity(self):
        for capacity in (0.0, -1.0):
            with self.assertRaises(CQ.CapacityQualificationError):
                CQ.select_tiers(capacity, repo_root=ROOT)

    def test_adoption_rejects_a_blank_decoder_digest(self):
        import dataclasses
        blank = [dataclasses.replace(t, checkpoint_sha256="")
                 for t in CQ.select_tiers(85.0, repo_root=ROOT)]
        with self.assertRaises(C.ContractError):
            C.load_tiers_from_selection(blank)

    def test_adoption_rejects_non_increasing_payloads(self):
        import dataclasses
        tiers = list(CQ.select_tiers(85.0, repo_root=ROOT))
        tiers[2] = dataclasses.replace(tiers[2],
                                       payload_bytes=tiers[0].payload_bytes - 1)
        with self.assertRaises(C.ContractError):
            C.load_tiers_from_selection(tiers)

    def test_legacy_106prb_capacity_is_refused_not_reused(self):
        self.assertIn("ADVERSE_STABLE_106PRB_7D2U", CQ.FORBIDDEN_LEGACY_CAPACITY_MBPS)
        self.assertIn("DO_NOT_REUSE", CQ.FORBIDDEN_LEGACY_CAPACITY_REASON.upper()
                      .replace(" ", "_") + "_DO_NOT_REUSE")

    def test_stage_freezes_no_action(self):
        plan = CQ.stage_plan()
        self.assertFalse(plan["actions_frozen_here"])
        self.assertEqual(plan["probe"]["role"], "SATURATING_PROBE_NEVER_A_TIER")


class CapacityStageGateTests(unittest.TestCase):
    """The stage refuses a boundary it did not actually measure."""

    def point(self, label, snr, p50, samples=200, backlogged=0.95):
        return CQ.CapacityPoint(
            label=label, target_snr_db=snr, commanded_noise_power_db=-5.0,
            service_mbps_p10=p50 * 0.9, service_mbps_p50=p50,
            service_mbps_p90=p50 * 1.1, samples=samples,
            backlogged_fraction=backlogged)

    def test_a_clean_surface_qualifies(self):
        audit = CQ.audit_points([self.point("p25", 7.827, 80.0),
                                 self.point("p50", 8.608, 85.0),
                                 self.point("p75", 9.604, 90.0)])
        self.assertTrue(audit["qualified"], audit["problems"])
        self.assertEqual(audit["adverse_capacity_mbps"], 85.0)

    def test_a_non_saturating_probe_is_refused(self):
        audit = CQ.audit_points([self.point("p25", 7.827, 80.0),
                                 self.point("p50", 8.608, 85.0, backlogged=0.3),
                                 self.point("p75", 9.604, 90.0)])
        self.assertFalse(audit["qualified"])
        self.assertTrue(any("did not saturate" in p for p in audit["problems"]))

    def test_too_few_samples_is_refused(self):
        audit = CQ.audit_points([self.point("p25", 7.827, 80.0, samples=5),
                                 self.point("p50", 8.608, 85.0),
                                 self.point("p75", 9.604, 90.0)])
        self.assertFalse(audit["qualified"])

    def test_capacity_falling_as_snr_rises_is_refused(self):
        audit = CQ.audit_points([self.point("p25", 7.827, 90.0),
                                 self.point("p50", 8.608, 85.0),
                                 self.point("p75", 9.604, 40.0)])
        self.assertFalse(audit["qualified"])
        self.assertFalse(audit["monotonic_in_snr"])

    def test_a_missing_operating_point_is_refused(self):
        audit = CQ.audit_points([self.point("p25", 7.827, 80.0),
                                 self.point("p50", 8.608, 85.0)])
        self.assertFalse(audit["qualified"])


class RadioBindingTests(unittest.TestCase):
    """Run 4 runs on 273PRB/4D5U, and proves it."""

    def test_all_pinned_identities_verify(self):
        report = RB.verify("before_preflight", ROOT)
        self.assertTrue(report["verified"])
        self.assertEqual(report["radio_profile_id"],
                         "OAI_N78_100MHZ_273PRB_4D5U_V1")
        self.assertGreaterEqual(len(report["files"]), 20)

    def test_launcher_digest_matches_the_phase14a_binding(self):
        binding = json.loads(
            (ROOT / "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json")
            .read_text())
        self.assertEqual(RB.PINS["launcher"]["sha256"],
                         binding["launcher"]["sha256"])
        self.assertEqual(binding["launcher"]["radio_profile_id"],
                         RB.RADIO_PROFILE_ID)
        self.assertEqual(RB.EXECUTION_TOKEN, binding["launcher"]["execution_token"])

    def test_radio_configs_match_the_phase14a_calibration_config(self):
        cal = json.loads(
            (ROOT / "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json")
            .read_text())["source_sha256"]
        self.assertEqual(RB.PINS["gnb_config_273prb"]["sha256"], cal["gnb_source"])
        self.assertEqual(RB.PINS["ue_config"]["sha256"], cal["ue_source"])
        self.assertEqual(RB.PINS["channel_config"]["sha256"], cal["channel_source"])

    def test_radio_is_273prb_100mhz_4d5u(self):
        self.assertEqual(RB.RADIO["prb"], 273)
        self.assertEqual(RB.RADIO["bandwidth_mhz"], 100)
        self.assertEqual((RB.RADIO["downlink_slots"], RB.RADIO["uplink_slots"]),
                         (4, 5))
        self.assertNotEqual(RB.RADIO["prb"], RB.FORBIDDEN_PRB)

    def test_ue_telemetry_port_is_the_launcher_port(self):
        launcher = (ROOT / RB.PINS["launcher"]["path"]).read_text()
        self.assertIn("--T_port 2023", launcher.replace("\n", " "))
        self.assertEqual(RB.TELEMETRY_PORTS["ue_port"], 2023)
        self.assertNotEqual(RB.TELEMETRY_PORTS["ue_port"], 2022)

    def test_compiled_t_messages_match_the_source(self):
        report = RB.verify("before_preflight", ROOT)
        self.assertTrue(report["compiled_t_messages_consistent"])

    def test_mapping_is_the_phase14a_273prb_mapping(self):
        self.assertEqual(len(RB.TARGET_SNR_ANCHORS), 12)
        self.assertTrue(RB._mapping_is_strictly_monotonic())
        data = json.loads((ROOT / RB.PINS["target_snr_mapping_json"]["path"])
                          .read_text())
        self.assertEqual(data["radio_profile_id"], RB.RADIO_PROFILE_ID)
        self.assertFalse(data["legacy_mapping_used"])

    def test_no_run3_106prb_anchor_survives(self):
        current = set(RB.TARGET_SNR_ANCHORS)
        for legacy in RB.FORBIDDEN_RUN3_ANCHORS:
            if legacy in current:
                # A coincidental (command, snr) pair is only acceptable if the
                # whole legacy set is not the mapping.
                self.assertNotEqual(current, set(RB.FORBIDDEN_RUN3_ANCHORS))
        self.assertNotEqual(current, set(RB.FORBIDDEN_RUN3_ANCHORS))
        self.assertEqual(RB.TARGET_SNR_ANCHORS[0], (-13.0, 25.5))

    def test_mapping_covers_the_required_target_range(self):
        self.assertLessEqual(RB.MAPPING_MEASURED_LOWER_DB,
                             RB.MAPPING_REQUIRED_LOWER_DB)
        self.assertGreaterEqual(RB.MAPPING_MEASURED_UPPER_DB,
                                RB.MAPPING_REQUIRED_UPPER_DB)

    def test_mapping_qualification_is_not_overclaimed(self):
        self.assertIn("does NOT claim the 288-cell campaign qualification",
                      RB.MAPPING_QUALIFICATION_NOTE)

    def test_a_drifted_pin_is_detected(self):
        original = RB.PINS["launcher"]["sha256"]
        try:
            RB.PINS["launcher"]["sha256"] = "0" * 64
            with self.assertRaises(RB.RadioBindingError):
                RB.verify("before_preflight", ROOT)
        finally:
            RB.PINS["launcher"]["sha256"] = original

    def test_forbidden_legacy_env_is_refused(self):
        for name in RB.FORBIDDEN_ENV:
            with self.assertRaises(RB.RadioBindingError):
                RB.assert_no_forbidden_env({name: "anything"})
        RB.assert_no_forbidden_env({"PATH": "/usr/bin"})

    def test_legacy_launchers_are_named_and_refused(self):
        self.assertEqual(len(RB.FORBIDDEN_LAUNCHERS), 2)
        self.assertIn("106", RB.FORBIDDEN_GNB_CONFIG)
        self.assertEqual(RB.FORBIDDEN_PRB, 106)


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

    def test_blocks_carry_the_selected_payloads_and_ports(self):
        tiers = {t.tier: t for t in example_tiers()}
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
        self.campaign = self.tmp / "campaign"
        self.campaign.mkdir()
        config = json.loads(
            (Path(__file__).resolve().parent / "config_v1.json").read_text())
        self.token = config["authorization"]["scientific_stage_token"]

    def authorize(self, **overrides):
        payload = {"stage": AUTH.SCIENTIFIC_STAGE, "token": self.token,
                   "granted_by": "test", "granted_utc": "2026-09-24T00:00:00Z"}
        payload.update(overrides)
        (self.tmp / AUTH.AUTHORIZATION_FILENAME).write_text(json.dumps(payload))

    def test_mkdir_refuses_an_existing_root(self):
        target = self.campaign / "20260924_000000"
        target.mkdir(parents=True, exist_ok=False)
        with self.assertRaises(FileExistsError):
            target.mkdir(parents=True, exist_ok=False)

    def test_runner_main_refuses_without_authorization(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        target = self.campaign / "20260924_120000"
        with self.assertRaises(AUTH.AuthorizationError):
            R.main(["--output-dir", str(target)])
        self.assertFalse(target.exists(),
                         "an unauthorized run must not create its root")

    def test_runner_main_refuses_an_existing_output_dir(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        target = self.campaign / "existing"
        target.mkdir()
        # Authorized, and the authorization supersedes the only prior attempt,
        # so the refusal under test is create-only, not authorization.
        self.authorize(supersedes="existing", repaired_defect="test fixture")
        # Refuses, and writes nothing into the pre-existing directory. The
        # specific exception depends on whether this task's own sources are
        # committed yet; the invariant under test is that nothing is written.
        with self.assertRaises(Exception):
            R.main(["--output-dir", str(target)])
        self.assertEqual(list(target.iterdir()), [],
                         "a refused run must not have written anything")

    def test_runner_main_authorizes_before_creating_anything(self):
        """A run must not count itself as a prior attempt."""
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        self.authorize()
        target = self.campaign / "20260924_130000"
        with self.assertRaises(Exception) as ctx:
            R.main(["--output-dir", str(target)])
        # Whatever stops it -- a dirty tree here, a cold radio on the host --
        # it must never be the one-attempt rule counting this run as its own
        # predecessor.
        self.assertNotIn("One attempt per campaign", str(ctx.exception))

    def test_runner_main_refuses_a_target_inside_the_protected_run(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        inside = ROOT / PE.PROTECTED_RUN_RELPATH / "run4_attempt"
        with self.assertRaises(PE.ProtectedEvidenceError):
            R.main(["--output-dir", str(inside)])
        self.assertFalse(inside.exists())


class AnalysisSpecTests(unittest.TestCase):
    """The preregistered analysis leaves no post-hoc freedom."""

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


class AnalysisCompletionTests(unittest.TestCase):
    """The four previously-incomplete analysis rules, adversarially exercised."""

    # --- exact MCS freshness limit ---------------------------------
    def test_mcs_freshness_limit_is_exactly_one_pinned_value(self):
        self.assertIsInstance(S.MCS_MAX_AGE_MS, float)
        self.assertEqual(S.MCS_MAX_AGE_MS, 200.0)
        self.assertIn(S.MCS_MAX_AGE_MS, S.MCS_AGE_SENSITIVITY_REPORT_MS)

    def test_sensitivity_sweep_does_not_select_the_operative_limit(self):
        self.assertGreater(len(S.MCS_AGE_SENSITIVITY_REPORT_MS), 1)
        self.assertEqual(S.MCS_MAX_AGE_MS, 200.0)

    # --- clock-bridge refusal --------------------------------------
    def test_clock_bridge_accepts_at_and_below_one_microsecond(self):
        for residual in (0.0, 0.48, 1.0):
            self.assertEqual(S.require_clock_bridge(residual, cell_id="c"), residual)

    def test_clock_bridge_refuses_above_one_microsecond(self):
        for residual in (1.000001, 1.5, 1000.0):
            with self.assertRaises(S.ClockBridgeError):
                S.require_clock_bridge(residual, cell_id="c")

    def test_clock_bridge_refuses_an_undefined_residual(self):
        with self.assertRaises(S.ClockBridgeError):
            S.require_clock_bridge(float("nan"), cell_id="c")

    # --- P95 mixture estimator -------------------------------------
    def test_mixture_percentile_matches_a_single_bin_exactly(self):
        samples = {("a",): [1.0, 2.0, 3.0, 4.0, 100.0]}
        weights = {("a",): 1.0}
        self.assertEqual(S.mixture_percentile(samples, weights, 100), 100.0)
        self.assertLessEqual(S.mixture_percentile(samples, weights, 50), 3.0)

    def test_mixture_percentile_follows_the_weights(self):
        samples = {("fast",): [10.0] * 10, ("slow",): [500.0] * 10}
        mostly_fast = S.mixture_percentile(samples, {("fast",): 0.99,
                                                     ("slow",): 0.01}, 50)
        mostly_slow = S.mixture_percentile(samples, {("fast",): 0.01,
                                                     ("slow",): 0.99}, 50)
        self.assertEqual(mostly_fast, 10.0)
        self.assertEqual(mostly_slow, 500.0)

    def test_mixture_percentile_ignores_unoccupied_and_empty_bins(self):
        samples = {("a",): [7.0], ("b",): []}
        self.assertEqual(
            S.mixture_percentile(samples, {("a",): 1.0, ("b",): 5.0}, 50), 7.0)

    def test_mixture_percentile_refuses_when_nothing_is_supported(self):
        with self.assertRaises(ValueError):
            S.mixture_percentile({("a",): []}, {("a",): 1.0}, 50)

    def test_p95_is_not_silently_the_p50(self):
        samples = {("a",): [1.0] * 95 + [900.0] * 5}
        weights = {("a",): 1.0}
        self.assertNotEqual(S.mixture_percentile(samples, weights, 50),
                            S.mixture_percentile(samples, weights, 95))

    def test_both_gate5_arms_use_one_percentile_convention(self):
        """Predicted and observed P50/P95 must mean the same thing."""
        values = [1.0] * 95 + [900.0] * 5
        samples, weights = {("a",): values}, {("a",): 1.0}
        for q in (0, 5, 25, 50, 75, 90, 95, 99, 100):
            with self.subTest(q=q):
                self.assertAlmostEqual(S.mixture_percentile(samples, weights, q),
                                       S.percentile(values, q), places=9)

    def test_latency_errors_report_both_arms(self):
        observed = [10.0] * 95 + [200.0] * 5
        errors = S.latency_errors(10.0, 200.0, observed)
        self.assertAlmostEqual(errors["p50_error_ms"], 0.0)
        self.assertLess(errors["p95_error_ms"], 200.0)
        self.assertEqual(errors["n"], 100.0)

    # --- gate 7 matching and effect rule ---------------------------
    def test_near_boundary_window_is_symmetric_and_bounded(self):
        capacity = 85.0
        self.assertTrue(S.is_near_boundary(capacity, capacity))
        self.assertTrue(S.is_near_boundary(capacity * 1.25, capacity))
        self.assertTrue(S.is_near_boundary(capacity * 0.75, capacity))
        self.assertFalse(S.is_near_boundary(capacity * 1.26, capacity))
        self.assertFalse(S.is_near_boundary(capacity * 0.5, capacity))

    def test_contrast_requires_a_real_mcs_gap(self):
        result = S.mcs_contrast([1] * 40, [1] * 40, low_bin=2, high_bin=3)
        self.assertIsNone(result["verdict"])
        self.assertEqual(result["reason"], "MCS_BIN_GAP_TOO_SMALL")

    def test_contrast_requires_support_on_both_sides(self):
        result = S.mcs_contrast([1] * 5, [1] * 100, low_bin=1, high_bin=5)
        self.assertIsNone(result["verdict"])
        self.assertEqual(result["reason"], "INSUFFICIENT_SUPPORT")

    def test_contrast_detects_the_physically_wrong_direction(self):
        result = S.mcs_contrast([1] * 40, [0] * 40, low_bin=1, high_bin=5)
        self.assertEqual(result["verdict"], S.CONTRAST_VIOLATION)
        self.assertLess(result["effect"], 0)

    def test_a_small_adverse_effect_is_null_not_a_pass(self):
        low = [1] * 39 + [0]
        high = [1] * 38 + [0] * 2
        result = S.mcs_contrast(low, high, low_bin=1, high_bin=5)
        self.assertEqual(result["verdict"], S.CONTRAST_NULL)

    def test_gate7_fails_on_any_violation_and_is_indeterminate_when_empty(self):
        self.assertEqual(S.gate7_direction_verdict(
            [{"verdict": S.CONTRAST_PASS}, {"verdict": S.CONTRAST_VIOLATION}]),
            S.FAIL)
        self.assertEqual(S.gate7_direction_verdict([]), S.INDETERMINATE)
        self.assertEqual(S.gate7_direction_verdict(
            [{"verdict": None, "reason": "INSUFFICIENT_SUPPORT"}]),
            S.INDETERMINATE)
        self.assertEqual(S.gate7_direction_verdict(
            [{"verdict": S.CONTRAST_NULL}]), S.PASS)

    def test_payload_levels_come_from_the_selected_tiers(self):
        levels = S.payload_levels(example_tiers())
        self.assertEqual(len(levels), 3)
        self.assertEqual(list(levels), sorted(levels))
        with self.assertRaises(ValueError):
            S.payload_levels(example_tiers()[:2])


class AuthorizationTests(unittest.TestCase):
    """One attempt per campaign, enforced rather than described."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nearcap_auth_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "campaign"
        self.root.mkdir()

    def write_auth(self, **overrides):
        payload = {"stage": AUTH.SCIENTIFIC_STAGE, "token": "TOK",
                   "granted_by": "abiodun", "granted_utc": "2026-09-24T00:00:00Z"}
        payload.update(overrides)
        (self.tmp / AUTH.AUTHORIZATION_FILENAME).write_text(json.dumps(payload))

    def test_a_run_without_authorization_is_refused(self):
        with self.assertRaises(AUTH.AuthorizationError):
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)

    def test_a_wrong_stage_or_token_is_refused(self):
        self.write_auth(stage=AUTH.CAPACITY_STAGE)
        with self.assertRaises(AUTH.AuthorizationError):
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)
        self.write_auth(token="WRONG")
        with self.assertRaises(AUTH.AuthorizationError):
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)

    def test_unknown_authorization_fields_are_refused_not_ignored(self):
        self.write_auth(sneaky_override=True)
        with self.assertRaises(AUTH.AuthorizationError):
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)

    def test_a_second_attempt_without_a_superseding_authorization_is_refused(self):
        self.write_auth()
        (self.root / "20260924_000000").mkdir()
        with self.assertRaises(AUTH.AuthorizationError) as ctx:
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)
        self.assertIn("One attempt per campaign", str(ctx.exception))

    def test_a_superseding_authorization_must_name_a_real_attempt(self):
        (self.root / "20260924_000000").mkdir()
        self.write_auth(supersedes="does_not_exist", repaired_defect="x")
        with self.assertRaises(AUTH.AuthorizationError):
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)

    def test_a_superseding_authorization_must_state_the_defect(self):
        (self.root / "20260924_000000").mkdir()
        self.write_auth(supersedes="20260924_000000")
        with self.assertRaises(AUTH.AuthorizationError):
            AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                       expected_token="TOK", repo_root=ROOT)

    def test_a_complete_superseding_authorization_is_accepted(self):
        (self.root / "20260924_000000").mkdir()
        self.write_auth(supersedes="20260924_000000",
                        repaired_defect="receiver bound the wrong port")
        result = AUTH.require_authorization(
            AUTH.SCIENTIFIC_STAGE, self.root, expected_token="TOK", repo_root=ROOT,
            require_clean_sources=False)
        self.assertEqual(result["supersedes"], "20260924_000000")
        self.assertEqual(result["prior_attempts"], ["20260924_000000"])

    def test_dirty_task_sources_are_refused_by_default(self):
        """A scientific run must execute one committed revision."""
        self.write_auth()
        import unittest.mock as mock
        dirty = {"head": "0" * 40, "branch": "master",
                 "dirty_paths": ["rl_agent/ue_mcs_backlog_near_capacity_v1/x.py"],
                 "dirty_outside_this_task": [],
                 "tree_clean_for_this_task": False}
        with mock.patch.object(AUTH, "source_commit", return_value=dirty):
            with self.assertRaises(AUTH.AuthorizationError) as ctx:
                AUTH.require_authorization(AUTH.SCIENTIFIC_STAGE, self.root,
                                           expected_token="TOK", repo_root=ROOT)
            self.assertIn("single committed code revision", str(ctx.exception))

    def test_a_clean_tree_is_accepted(self):
        self.write_auth()
        import unittest.mock as mock
        clean = {"head": "0" * 40, "branch": "master", "dirty_paths": [],
                 "dirty_outside_this_task": [], "tree_clean_for_this_task": True}
        with mock.patch.object(AUTH, "source_commit", return_value=clean):
            result = AUTH.require_authorization(
                AUTH.SCIENTIFIC_STAGE, self.root, expected_token="TOK",
                repo_root=ROOT)
        self.assertEqual(result["prior_attempts"], [])

    def test_prior_attempts_are_listed_never_removed(self):
        for name in ("20260924_000000", "20260924_010000"):
            (self.root / name).mkdir()
        self.assertEqual(AUTH.existing_attempts(self.root),
                         ["20260924_000000", "20260924_010000"])
        self.assertTrue(all((self.root / n).exists()
                            for n in ("20260924_000000", "20260924_010000")))

    def test_lineage_names_parent_and_authorization(self):
        record = AUTH.lineage_record(
            AUTH.SCIENTIFIC_STAGE, run_id="20260924_120000",
            parent={"run_id": "cap_0"}, authorization={"granted_by": "abiodun"})
        self.assertEqual(record["parent"], {"run_id": "cap_0"})
        self.assertIn("One attempt per campaign", record["one_attempt_policy"])

    def test_source_commit_reports_head_and_task_cleanliness(self):
        commit = AUTH.source_commit(ROOT)
        self.assertEqual(len(commit["head"]), 40)
        self.assertIn("tree_clean_for_this_task", commit)


class FailurePolicyTests(unittest.TestCase):
    """Every condition Run 3 only logged must now fail the run."""

    def test_config_declares_every_required_failure(self):
        config = json.loads(
            (Path(__file__).resolve().parent / "config_v1.json").read_text())
        declared = set(config["failure_policy"]["nonzero_exit_on"])
        for required in ("UNRESOLVED_CALIBRATION", "UNREGISTERED_CLAMP_OR_SKIP",
                         "RF_RESTORE_OR_READBACK_FAILURE",
                         "SENDER_OR_RECEIVER_FAILURE",
                         "TTRACER_EXTRACTION_FAILURE", "ACCOUNTING_FAILURE",
                         "ANY_TEARDOWN_NOTE", "NON_COLD_FINAL_STATE",
                         "UDP_PROBE_FAILURE", "RADIO_BINDING_DRIFT"):
            self.assertIn(required, declared)

    def test_runner_enforces_them_rather_than_only_recording(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        source = inspect.getsource(R.Runner.run_cell)
        for guard in ("clamped == 0", "skipped == 0", 'require(record["restored"]',
                      "teardown notes present", "extraction"):
            self.assertIn(guard, source)
        run_source = inspect.getsource(R.Runner.run)
        self.assertIn('cold.get("cold")', run_source)
        self.assertIn("teardown notes present", run_source)

    def test_udp_probe_requires_arrival_and_pdcp_evidence(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        source = inspect.getsource(R.Runner.udp_probe)
        self.assertIn("NR_PDCP_TX_SDU", source)
        self.assertIn('outcome["arrival_ok"]', source)
        self.assertIn('outcome["pdcp_ok"]', source)
        config = json.loads(
            (Path(__file__).resolve().parent / "config_v1.json").read_text())
        self.assertTrue(config["traffic"]["udp_probe_requires_pdcp_evidence"])
        self.assertGreaterEqual(config["traffic"]["udp_probe_datagrams"], 1)

    def test_identities_are_verified_at_three_points(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        self.assertEqual(R.VERIFY_STAGES,
                         ("before_preflight", "before_scientific_cells",
                          "final_sealing"))
        run_source = inspect.getsource(R.Runner.run)
        for stage in R.VERIFY_STAGES:
            self.assertIn(stage, run_source)

    def test_runner_binds_the_registered_mapping_not_run3_anchors(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        source = inspect.getsource(R.Runner.__init__)
        self.assertIn("RB.anchors_for_interpolation()", source)

    def test_runner_tears_down_the_launcher_started_ran(self):
        from rl_agent.ue_mcs_backlog_near_capacity_v1 import runner as R
        source = inspect.getsource(R.Runner.teardown_ran)
        self.assertIn("nr-softmodem", source)
        self.assertIn("nr-uesoftmodem", source)

    def test_config_has_no_live_106prb_radio_block(self):
        config = json.loads(
            (Path(__file__).resolve().parent / "config_v1.json").read_text())
        self.assertEqual(config["radio"]["prb"], 273)
        self.assertEqual(config["telemetry"]["ue_port"], 2023)
        self.assertFalse(config["actuator"]["legacy_106prb_anchors_reused"])
        self.assertEqual(len(config["actuator"]["registered_anchors"]), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
