#!/usr/bin/env python3
"""Tests for the UE-state evidence audit.

CPU only, no network, no CUDA, no Docker, no CARLA, no OAI.  Almost everything
runs against temporary synthetic fixtures; one bounded smoke test touches the
real evidence tree read-only and proves it stays byte-identical.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_ue_state_evidence as audit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_EVIDENCE_ROOT = REPO_ROOT / "rl_agent" / "experiments"


def write_csv(path: Path, header, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    lines.extend(",".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n")
    return path


def rlc_row(time, frame, slot, lcid, lcgid, backlog):
    return [time, 1, 0, frame, slot, lcid, lcgid, backlog, 0, -1, 1]


def bsr_row(time, frame, slot, lcg_bytes, sdu_bytes=0, bsr_index=0, bsr_lcg_id=0):
    groups = list(lcg_bytes) + [0] * (audit.LCG_COUNT - len(lcg_bytes))
    return (
        [time, 1, 0, frame, slot, 2, 1, 1, 0, 0, sdu_bytes]
        + groups
        + [bsr_lcg_id, bsr_index]
        + [0] * 8
    )


class TimestampParsingTests(unittest.TestCase):
    def test_parses_microsecond_precision(self):
        self.assertEqual(audit.parse_tracer_time("00:00:00.000000"), 0)
        self.assertEqual(audit.parse_tracer_time("01:02:03.000456"), 3_723_000_456)

    def test_short_fraction_is_left_padded_not_right_shifted(self):
        # ".5" in a microsecond field means 500000 us, not 5 us.
        self.assertEqual(audit.parse_tracer_time("00:00:00.5"), 500_000)

    def test_rejects_malformed_and_out_of_range(self):
        for bad in ("", "12:34", "aa:bb:cc.dddddd", "25:00:00.000000", "00:61:00.000000"):
            with self.assertRaises(audit.SchemaError):
                audit.parse_tracer_time(bad)

    def test_midnight_rollover_adds_one_day_but_jitter_does_not(self):
        day = audit.MICROSECONDS_PER_DAY
        rolled = audit.unwrap_tracer_times([day - 10, day - 5, 3, 8])
        self.assertEqual(rolled, [day - 10, day - 5, day + 3, day + 8])
        # A small backwards step is reordering, not a new day.
        jittered = audit.unwrap_tracer_times([1_000, 999, 1_001])
        self.assertEqual(jittered, [1_000, 999, 1_001])


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_both_schemas(self):
        rlc = write_csv(
            self.tmp / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
            audit.RLC_BUFFER_HEADER,
            [rlc_row("00:00:01.000000", 10, 2, 4, 1, 512)],
        )
        bsr = write_csv(
            self.tmp / "NRUE_MAC_BSR_STATUS.csv",
            audit.BSR_STATUS_HEADER,
            [bsr_row("00:00:01.000010", 10, 2, [0, 512])],
        )
        self.assertEqual(audit.read_rlc_buffer_rows(rlc)[0].bytes_in_buffer, 512)
        self.assertEqual(audit.read_bsr_status_rows(bsr)[0].lcg_total_bytes, 512)

    def test_renamed_or_reordered_column_is_rejected(self):
        header = list(audit.RLC_BUFFER_HEADER)
        header[7] = "bytes_in_queue"
        path = write_csv(self.tmp / "renamed.csv", header, [])
        with self.assertRaises(audit.SchemaError):
            audit.read_rlc_buffer_rows(path)

        swapped = list(audit.BSR_STATUS_HEADER)
        swapped[11], swapped[12] = swapped[12], swapped[11]
        path = write_csv(self.tmp / "swapped.csv", swapped, [])
        with self.assertRaises(audit.SchemaError):
            audit.read_bsr_status_rows(path)

    def test_negative_byte_counts_are_rejected_not_clamped(self):
        path = write_csv(
            self.tmp / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
            audit.RLC_BUFFER_HEADER,
            [rlc_row("00:00:01.000000", 10, 2, 4, 1, -1)],
        )
        with self.assertRaisesRegex(audit.SchemaError, "negative"):
            audit.read_rlc_buffer_rows(path)

        path = write_csv(
            self.tmp / "NRUE_MAC_BSR_STATUS.csv",
            audit.BSR_STATUS_HEADER,
            [bsr_row("00:00:01.000000", 10, 2, [0, -5])],
        )
        with self.assertRaisesRegex(audit.SchemaError, "negative"):
            audit.read_bsr_status_rows(path)

    def test_missing_and_non_numeric_byte_counts_are_rejected(self):
        for bad in ("", "n/a", "512.5"):
            path = write_csv(
                self.tmp / f"bad_{len(bad)}_{bad or 'empty'}.csv".replace("/", "_"),
                audit.RLC_BUFFER_HEADER,
                [rlc_row("00:00:01.000000", 10, 2, 4, 1, bad)],
            )
            with self.assertRaises(audit.SchemaError):
                audit.read_rlc_buffer_rows(path)

    def test_short_row_is_rejected(self):
        path = self.tmp / "short.csv"
        path.write_text(",".join(audit.RLC_BUFFER_HEADER) + "\n1,2,3\n")
        with self.assertRaisesRegex(audit.SchemaError, "fields"):
            audit.read_rlc_buffer_rows(path)


class BacklogTotalTests(unittest.TestCase):
    def test_tick_total_sums_every_logical_channel(self):
        rows = [
            audit.RlcBufferRow(1_000, 10, 2, 1, 0, 0),
            audit.RlcBufferRow(1_000, 10, 2, 2, 0, 128),
            audit.RlcBufferRow(1_001, 10, 2, 4, 1, 512),
        ]
        ticks = audit.rlc_ticks(rows)
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].total_bytes, 640)
        self.assertEqual(ticks[0].per_lcg_bytes[0], 128)
        self.assertEqual(ticks[0].per_lcg_bytes[1], 512)

    def test_one_emitter_call_is_not_split_by_a_microsecond_boundary(self):
        # The three rows of one call can straddle a microsecond tick; grouping
        # on the timestamp would wrongly report two ticks.
        rows = [
            audit.RlcBufferRow(999, 10, 2, 1, 0, 0),
            audit.RlcBufferRow(1_000, 10, 2, 2, 0, 0),
            audit.RlcBufferRow(1_000, 10, 2, 4, 1, 300),
        ]
        self.assertEqual(len(audit.rlc_ticks(rows)), 1)

    def test_repeated_lcid_starts_the_next_tick(self):
        rows = [
            audit.RlcBufferRow(1_000, 10, 2, 4, 1, 300),
            audit.RlcBufferRow(1_500, 10, 2, 4, 1, 100),
        ]
        ticks = audit.rlc_ticks(rows)
        self.assertEqual([tick.total_bytes for tick in ticks], [300, 100])

    def test_changed_frame_slot_starts_the_next_tick(self):
        rows = [
            audit.RlcBufferRow(1_000, 10, 2, 1, 0, 0),
            audit.RlcBufferRow(1_500, 10, 3, 1, 0, 7),
        ]
        self.assertEqual(len(audit.rlc_ticks(rows)), 2)

    def test_zero_backlog_ticks_are_kept_as_measured_zeros(self):
        rows = [audit.RlcBufferRow(1_000 * n, 10, n, 4, 1, 0) for n in range(1, 6)]
        ticks = audit.rlc_ticks(rows)
        stats = audit.describe_distribution([float(t.total_bytes) for t in ticks])
        self.assertEqual(stats.count, 5)
        self.assertEqual(stats.zero_count, 5)
        self.assertEqual(stats.zero_fraction, 1.0)

    def test_bsr_lcg_total_covers_all_eight_groups(self):
        row = audit.BsrStatusRow(
            time_us=0, frame=0, slot=0, bsr_type=2, bsr_sent=1, num_sdus=0,
            sdu_bytes=0, lcg_bytes=tuple(range(8)), bsr_lcg_id=0, bsr_index=0,
        )
        self.assertEqual(row.lcg_total_bytes, 28)


class DistributionTests(unittest.TestCase):
    def test_empty_input_yields_missing_not_zero(self):
        stats = audit.describe_distribution([])
        self.assertEqual(stats.count, 0)
        self.assertIsNone(stats.zero_fraction)
        self.assertIsNone(stats.p50)
        self.assertIsNone(stats.maximum)

    def test_nearest_rank_percentile_returns_an_observed_value(self):
        values = [float(v) for v in range(1, 101)]
        self.assertEqual(audit.percentile(values, 0.50), 50.0)
        self.assertEqual(audit.percentile(values, 0.95), 95.0)
        self.assertEqual(audit.percentile(values, 1.0), 100.0)

    def test_percentile_rejects_out_of_range_quantile(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(audit.AuditError):
                audit.percentile([1.0, 2.0], bad)

    def test_per_run_stats_are_not_pooled(self):
        # A run of 1000 zeros beside a run of 10 large values must keep its own
        # zero fraction; pooling would drown the small run.
        run_a = [0.0] * 1000
        run_b = [10_000.0] * 10
        per_run = [audit.describe_distribution(run_a), audit.describe_distribution(run_b)]
        self.assertEqual(per_run[0].zero_fraction, 1.0)
        self.assertEqual(per_run[1].zero_fraction, 0.0)
        pooled = audit.describe_distribution(run_a + run_b)
        self.assertNotEqual(pooled.zero_fraction, per_run[1].zero_fraction)
        self.assertAlmostEqual(pooled.zero_fraction, 1000 / 1010)


class ClockDomainTests(unittest.TestCase):
    def _records(self, times):
        return [{"k": (1, 1), "t": t} for t in times]

    def test_join_across_domains_without_a_bridge_is_refused(self):
        with self.assertRaisesRegex(audit.ClockDomainError, "measured clock bridge"):
            audit.align_one_to_one(
                left_source="ue",
                right_source="sender",
                left=self._records([0]),
                right=self._records([0]),
                key_of=lambda r: r["k"],
                time_of=lambda r: r["t"],
                join_keys=("k",),
                left_domain=audit.ClockDomain.T_TRACER_REALTIME_LOCAL,
                right_domain=audit.ClockDomain.EPOCH_WALL_SECONDS,
                max_skew_us=100,
                max_skew_justification="test",
            )

    def test_cross_domain_join_is_allowed_with_a_measured_bridge(self):
        result = audit.align_one_to_one(
            left_source="ue",
            right_source="sender",
            left=self._records([1_000]),
            right=self._records([0]),
            key_of=lambda r: r["k"],
            time_of=lambda r: r["t"],
            join_keys=("k",),
            left_domain=audit.ClockDomain.T_TRACER_REALTIME_LOCAL,
            right_domain=audit.ClockDomain.EPOCH_WALL_SECONDS,
            max_skew_us=100,
            max_skew_justification="test",
            bridge_us=1_000,
        )
        self.assertEqual(result.matched_rows, 1)
        self.assertEqual(result.domain_relation, "BRIDGED_BY_MEASURED_OFFSET")

    def test_unresolved_domain_is_still_a_different_domain(self):
        with self.assertRaises(audit.ClockDomainError):
            audit.align_one_to_one(
                left_source="a",
                right_source="b",
                left=self._records([0]),
                right=self._records([0]),
                key_of=lambda r: r["k"],
                time_of=lambda r: r["t"],
                join_keys=("k",),
                left_domain=audit.ClockDomain.T_TRACER_REALTIME_LOCAL,
                right_domain=audit.ClockDomain.UNRESOLVED,
                max_skew_us=10,
                max_skew_justification="test",
            )


class AlignmentTests(unittest.TestCase):
    def _mk(self, key, time):
        return {"k": key, "t": time}

    def _align(self, left, right, max_skew_us):
        return audit.align_one_to_one(
            left_source="L",
            right_source="R",
            left=left,
            right=right,
            key_of=lambda r: r["k"],
            time_of=lambda r: r["t"],
            join_keys=("k",),
            left_domain=audit.ClockDomain.T_TRACER_REALTIME_LOCAL,
            right_domain=audit.ClockDomain.T_TRACER_REALTIME_LOCAL,
            max_skew_us=max_skew_us,
            max_skew_justification="test",
        )

    def test_one_to_one_match_with_skew_accounting(self):
        left = [self._mk((1, 1), 0), self._mk((1, 2), 1_000_000)]
        right = [self._mk((1, 1), 5), self._mk((1, 2), 1_000_010)]
        result = self._align(left, right, 100)
        self.assertEqual(result.matched_rows, 2)
        self.assertEqual(result.unmatched_left_rows, 0)
        self.assertEqual(result.reused_right_rows, 0)
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.skew_p50_us, 5.0)
        self.assertEqual(result.skew_max_us, 10.0)

    def test_rows_outside_the_window_stay_unmatched_rather_than_snapping(self):
        left = [self._mk((1, 1), 0)]
        right = [self._mk((1, 1), 0), self._mk((1, 1), 10_000_000)]
        result = self._align(left, right, 100)
        self.assertEqual(result.matched_rows, 1)
        left_far = [self._mk((1, 1), 5_000_000)]
        far = self._align(left_far, right, 100)
        self.assertEqual(far.matched_rows, 0)
        self.assertEqual(far.unmatched_left_rows, 1)

    def test_ambiguous_key_is_refused_not_resolved_by_proximity(self):
        # Two right rows sit 40 us apart and a 50 us window would reach both.
        # The refusal comes from the measured-recurrence guard, which is the
        # primary defence: it rejects the unsound *window* before any row is
        # matched, so no left row is ever silently snapped to the nearer of two
        # counterparts.
        left = [self._mk((1, 1), 100)]
        right = [self._mk((1, 1), 90), self._mk((1, 1), 130)]
        with self.assertRaisesRegex(audit.AuditError, "recurrence"):
            self._align(left, right, 50)

    def test_narrowing_the_window_resolves_the_same_pair_unambiguously(self):
        left = [self._mk((1, 1), 100)]
        right = [self._mk((1, 1), 90), self._mk((1, 1), 130)]
        result = self._align(left, right, 15)
        self.assertEqual(result.matched_rows, 1)
        self.assertEqual(result.skew_max_us, 10.0)

    def test_every_join_refusal_shares_one_catchable_base(self):
        # A caller that catches AuditError catches the recurrence guard, the
        # per-key backstop and the clock-domain refusal alike.
        for subclass in (
            audit.JoinAmbiguityError,
            audit.ClockDomainError,
            audit.RunIsolationError,
            audit.SchemaError,
            audit.ActionLeakageError,
        ):
            self.assertTrue(issubclass(subclass, audit.AuditError))

    def test_skew_bound_wider_than_half_the_key_recurrence_is_refused(self):
        left = [self._mk((1, 1), 0)]
        right = [self._mk((1, 1), 0), self._mk((1, 1), 1_000)]
        with self.assertRaisesRegex(audit.AuditError, "recurrence"):
            self._align(left, right, 600)

    def test_coverage_is_missing_rather_than_zero_for_an_empty_left_side(self):
        result = self._align([], [self._mk((1, 1), 0)], 10)
        self.assertIsNone(result.coverage)
        self.assertEqual(result.matched_rows, 0)

    def test_reused_right_rows_are_counted(self):
        left = [self._mk((1, 1), 0), self._mk((1, 1), 20)]
        right = [self._mk((1, 1), 10)]
        result = self._align(left, right, 50)
        self.assertEqual(result.matched_rows, 2)
        self.assertEqual(result.reused_right_rows, 1)

    def test_measured_recurrence_is_reported(self):
        left = [self._mk((1, 1), 0)]
        right = [self._mk((1, 1), 0), self._mk((1, 1), 5_000)]
        self.assertEqual(self._align(left, right, 100).min_same_key_recurrence_us, 5_000)

    def test_recurrence_derived_skew_stays_unambiguous(self):
        right = [self._mk((1, 1), 0), self._mk((1, 1), 5_000)]
        bound, reason = audit.recurrence_derived_skew(
            right,
            lambda r: r["k"],
            lambda r: r["t"],
            source_label="R",
            purpose="test purpose",
        )
        self.assertEqual(bound, 2_499)
        self.assertIn("5000 us", reason)
        self.assertIn("test purpose", reason)
        # The derived bound must itself survive the ambiguity check.
        self._align([self._mk((1, 1), 100)], right, bound)


class RunIsolationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_run_root_is_the_parent_of_ttracer(self):
        path = self.tmp / "expA" / "run1" / "ttracer" / "ue" / "csv" / "x.csv"
        self.assertEqual(audit.logical_run_root(path), self.tmp / "expA" / "run1")

    def test_path_without_ttracer_has_no_run_identity(self):
        with self.assertRaisesRegex(audit.RunIsolationError, "ttracer"):
            audit.logical_run_root(self.tmp / "expA" / "run1" / "x.csv")

    def test_cross_run_join_is_refused_even_for_identical_filenames(self):
        left = self.tmp / "expA" / "run1" / "ttracer" / "ue" / "csv" / "a.csv"
        right = self.tmp / "expA" / "run2" / "ttracer" / "ue" / "csv" / "a.csv"
        with self.assertRaisesRegex(audit.RunIsolationError, "cross-run"):
            audit.require_same_run(left, right)

    def test_same_run_join_is_allowed(self):
        base = self.tmp / "expA" / "run1" / "ttracer"
        left = base / "ue" / "csv" / "a.csv"
        right = base / "gnb" / "csv" / "b.csv"
        self.assertEqual(audit.require_same_run(left, right), self.tmp / "expA" / "run1")

    def test_discovery_keys_runs_by_directory_not_by_filename(self):
        for name in ("run1", "run2"):
            write_csv(
                self.tmp / "exp" / name / "ttracer" / "ue" / "csv"
                / "NRUE_MAC_BSR_STATUS.csv",
                audit.BSR_STATUS_HEADER,
                [],
            )
        runs = audit.discover_runs(self.tmp)
        self.assertEqual([r.run_id for r in runs], ["exp/run1", "exp/run2"])


class OrderingAndLeakageTests(unittest.TestCase):
    def test_rlc_buffer_is_classified_pre_multiplex(self):
        self.assertIs(
            audit.RLC_BUFFER_SOURCE.ordering,
            audit.EnqueueOrdering.PRE_MULTIPLEX_WITHIN_MAC_SLOT,
        )

    def test_bsr_status_is_classified_post_multiplex_residual(self):
        self.assertIs(
            audit.BSR_STATUS_SOURCE.ordering,
            audit.EnqueueOrdering.POST_MULTIPLEX_RESIDUAL,
        )

    def test_every_classification_cites_source_code(self):
        for source in audit.CANDIDATE_SOURCES:
            self.assertRegex(source.ordering_evidence, r"\.(c|h|txt):\d+")

    def test_post_multiplex_source_is_blocked_as_policy_state(self):
        with self.assertRaisesRegex(audit.ActionLeakageError, "leaks the action"):
            audit.assert_no_action_leakage(audit.BSR_STATUS_SOURCE)

    def test_gnb_buffer_estimate_is_also_blocked(self):
        with self.assertRaises(audit.ActionLeakageError):
            audit.assert_no_action_leakage(audit.GNB_ESTIMATED_BUFFER_SOURCE)

    def test_pre_multiplex_source_passes_the_leakage_check(self):
        audit.assert_no_action_leakage(audit.RLC_BUFFER_SOURCE)

    def test_pre_multiplex_alone_does_not_qualify_without_enqueue_evidence(self):
        self.assertIs(
            audit.qualify_pre_action_source(
                audit.RLC_BUFFER_SOURCE, enqueue_instant_evidence=False
            ),
            audit.PreActionQualification.UNRESOLVED,
        )

    def test_pre_multiplex_plus_enqueue_evidence_qualifies(self):
        self.assertIs(
            audit.qualify_pre_action_source(
                audit.RLC_BUFFER_SOURCE, enqueue_instant_evidence=True
            ),
            audit.PreActionQualification.QUALIFIED,
        )

    def test_post_multiplex_never_qualifies_even_with_enqueue_evidence(self):
        self.assertIs(
            audit.qualify_pre_action_source(
                audit.BSR_STATUS_SOURCE, enqueue_instant_evidence=True
            ),
            audit.PreActionQualification.DISQUALIFIED_ACTION_CONTAMINATED,
        )

    def test_visibility_separates_ue_from_gnb(self):
        self.assertIs(audit.RLC_BUFFER_SOURCE.visibility, audit.SourceVisibility.UE)
        self.assertIs(
            audit.GNB_ESTIMATED_BUFFER_SOURCE.visibility, audit.SourceVisibility.GNB
        )
        self.assertFalse(audit.GNB_ESTIMATED_BUFFER_SOURCE.ue_runtime_available)


class EnqueueEvidenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, names):
        csv_dir = self.tmp / "exp" / "r" / "ttracer" / "ue" / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (csv_dir / f"{name}.csv").write_text("time\n")
        return audit.LogicalRun(
            run_id="exp/r", root=self.tmp / "exp" / "r",
            ue_csv_dir=csv_dir, gnb_csv_dir=None, traffic_sender=None,
        )

    def test_absent_enqueue_traces_are_detected(self):
        run = self._run(["NRUE_MAC_BSR_STATUS", "NRUE_MAC_RLC_BUFFER_STATUS"])
        self.assertFalse(audit.has_enqueue_instant_evidence(run))

    def test_any_enqueue_trace_counts(self):
        for event in audit.ENQUEUE_INSTANT_EVENTS:
            self.assertTrue(audit.has_enqueue_instant_evidence(self._run([event])))


class NormalizationTests(unittest.TestCase):
    def test_matches_the_deployed_contract_formula(self):
        # state_reward_transition_contract.py:5827
        for byte_count in (0, 1, 2, 10, 12751):
            expected = min(max(math.log1p(float(byte_count)) / 1.0, 0.0), 1.0)
            self.assertEqual(
                audit.scale_bsr_bytes(byte_count, audit.DEPLOYED_BSR_LOG1P_SCALE),
                expected,
            )

    def test_deployed_scale_saturates_at_two_bytes(self):
        self.assertEqual(
            audit.smallest_saturating_bytes(audit.DEPLOYED_BSR_LOG1P_SCALE), 2
        )
        self.assertLess(audit.scale_bsr_bytes(1, 1.0), 1.0)
        self.assertEqual(audit.scale_bsr_bytes(2, 1.0), 1.0)

    def test_deployed_scale_collapses_every_nonzero_queue(self):
        values = [0, 0, 1, 64, 512, 12751, 186_000]
        report = audit.saturation_report(
            values, scale=1.0, label="deployed", status="CURRENTLY_FROZEN_IN_CONTRACT"
        )
        # Only the single 1-byte sample escapes the clip.
        self.assertEqual(report.saturated_count, 4)
        self.assertAlmostEqual(report.saturated_fraction_of_nonzero, 4 / 5)
        self.assertEqual(report.distinct_outputs, 3)

    def test_zero_is_preserved_by_every_candidate_scale(self):
        values = [0, 1, 4096]
        for _, scale, _ in audit.candidate_scales(values):
            self.assertEqual(audit.scale_bsr_bytes(0, scale), 0.0)
        report = audit.saturation_report(
            values, scale=9.0, label="c", status="PROVISIONAL_NOT_FROZEN"
        )
        self.assertEqual(report.zero_preserved_count, 1)

    def test_a_wider_scale_recovers_resolution(self):
        values = [0, 1, 64, 512, 12751]
        narrow = audit.saturation_report(
            values, scale=1.0, label="n", status="x"
        )
        wide = audit.saturation_report(
            values, scale=math.log1p(12751.0), label="w", status="x"
        )
        self.assertGreater(wide.distinct_outputs, narrow.distinct_outputs)
        self.assertLess(wide.saturated_count, narrow.saturated_count)

    def test_every_candidate_other_than_the_deployed_one_is_provisional(self):
        candidates = audit.candidate_scales([0, 1, 100, 12751])
        statuses = {label: status for label, _, status in candidates}
        self.assertEqual(
            statuses["DEPLOYED_bsr_log1p_scale_1.0"], "CURRENTLY_FROZEN_IN_CONTRACT"
        )
        for label, status in statuses.items():
            if label != "DEPLOYED_bsr_log1p_scale_1.0":
                self.assertEqual(status, "PROVISIONAL_NOT_FROZEN")

    def test_candidates_are_derived_from_nonzero_values_only(self):
        # A run that is 99% zero must not produce a degenerate P95 of 0.
        values = [0] * 99 + [4096]
        labels = [label for label, _, _ in audit.candidate_scales(values)]
        self.assertIn("CANDIDATE_log1p_nonzero_P95_4096B", labels)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(audit.AuditError):
            audit.scale_bsr_bytes(-1, 1.0)
        with self.assertRaises(audit.AuditError):
            audit.scale_bsr_bytes(0, 0.0)


class CoverageVerdictTests(unittest.TestCase):
    def _run_audit(self, sizes_per_run, enqueue=False):
        runs = []
        for index, sizes in enumerate(sizes_per_run):
            run = audit.RunAudit(run_id=f"r{index}")
            run.offered_load = audit.OfferedLoadProfile(
                present=True, row_count=10,
                distinct_frame_bytes=tuple(sizes), distinct_period_s=(0.1,),
                reaches_splitfusion_range=max(sizes)
                >= audit.SPLITFUSION_PAYLOAD_MIN_BYTES,
                note="",
            )
            run.enqueue_instant_evidence = enqueue
            run.pre_action_qualification = {
                source.name: audit.qualify_pre_action_source(
                    source, enqueue_instant_evidence=enqueue
                ).value
                for source in audit.CANDIDATE_SOURCES
            }
            runs.append(run)
        return audit.decide_coverage(runs)

    def test_fixed_subrange_traffic_is_insufficient(self):
        verdict, rationale = self._run_audit([[12_500], [25_000]])
        self.assertIs(verdict, audit.CoverageVerdict.INSUFFICIENT)
        self.assertIn("below the action range", rationale)

    def test_varied_traffic_reaching_the_action_range_is_adequate(self):
        verdict, _ = self._run_audit([[49_400, 400_000, 1_000_000]])
        self.assertIs(verdict, audit.CoverageVerdict.ACTION_CONDITIONED)

    def test_fixed_traffic_with_resolved_ordering_is_carrier_audit_only(self):
        verdict, _ = self._run_audit([[12_500]], enqueue=True)
        self.assertIs(verdict, audit.CoverageVerdict.CARRIER_AND_NORMALIZATION_ONLY)

    def test_no_offered_load_evidence_is_insufficient(self):
        run = audit.RunAudit(run_id="r0")
        run.offered_load = audit.OfferedLoadProfile(
            present=False, row_count=0, distinct_frame_bytes=(),
            distinct_period_s=(), reaches_splitfusion_range=False, note="",
        )
        verdict, _ = audit.decide_coverage([run])
        self.assertIs(verdict, audit.CoverageVerdict.INSUFFICIENT)

    def test_varied_traffic_below_the_action_range_does_not_extrapolate_up(self):
        verdict, _ = self._run_audit([[12_500, 25_000]])
        self.assertIs(verdict, audit.CoverageVerdict.INSUFFICIENT)


class AssociationTests(unittest.TestCase):
    def test_associations_are_labelled_descriptive(self):
        ticks = [audit.RlcTick(n * 1_000, 1, n, n * 10, (0,) * 8, ()) for n in range(1, 8)]
        out = audit.descriptive_associations(
            ticks=ticks, bsr_rows=[], granted_tb_bytes=[], same_tick_skew_us=100
        )
        self.assertEqual(out["claim"], "DESCRIPTIVE_ONLY_NOT_CAUSAL")
        self.assertEqual(out["traffic_regime"], "FIXED_RATE_GENERATOR_NO_ACTION_VARIATION")
        self.assertIsNotNone(out["r_backlog_t_vs_backlog_t_plus_1"])

    def test_grant_size_is_kept_separate_from_served_bytes(self):
        out = audit.descriptive_associations(
            ticks=[], bsr_rows=[], granted_tb_bytes=[], same_tick_skew_us=100
        )
        self.assertIn("padding", out["granted_tb_bytes_note"])
        self.assertIsNone(out["r_backlog_vs_served_sdu_bytes"])

    def test_pearson_is_missing_rather_than_zero_for_a_constant_series(self):
        self.assertIsNone(audit._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertIsNone(audit._pearson([1.0], [1.0]))


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_sha256_binds_content_and_changes_with_it(self):
        import hashlib

        path = self.tmp / "a.csv"
        path.write_text("time\n1\n")
        first = audit.sha256_file(path)
        self.assertEqual(first, hashlib.sha256(b"time\n1\n").hexdigest())
        path.write_text("time\n2\n")
        self.assertNotEqual(first, audit.sha256_file(path))

    def test_describe_file_reports_rows_excluding_the_header(self):
        path = write_csv(
            self.tmp / "x.csv", audit.RLC_BUFFER_HEADER,
            [rlc_row("00:00:01.000000", 1, 1, 4, 1, 0)] * 3,
        )
        prov = audit.describe_file(path, self.tmp)
        self.assertEqual(prov.row_count, 3)
        self.assertEqual(prov.header, audit.RLC_BUFFER_HEADER)
        self.assertEqual(prov.sha256, audit.sha256_file(path))

    def test_header_only_file_reports_zero_rows(self):
        path = write_csv(self.tmp / "empty.csv", audit.RLC_BUFFER_HEADER, [])
        self.assertEqual(audit.describe_file(path, self.tmp).row_count, 0)


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        csv_dir = self.tmp / "exp" / "run1" / "ttracer" / "ue" / "csv"
        write_csv(
            csv_dir / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
            audit.RLC_BUFFER_HEADER,
            [
                row
                for n in range(1, 21)
                for row in (
                    rlc_row(f"00:00:0{n // 10}.{n % 10:01d}00000", 10, n, 1, 0, 0),
                    rlc_row(f"00:00:0{n // 10}.{n % 10:01d}00001", 10, n, 4, 1, n * 100),
                )
            ],
        )
        write_csv(
            csv_dir / "NRUE_MAC_BSR_STATUS.csv",
            audit.BSR_STATUS_HEADER,
            [
                bsr_row(f"00:00:0{n // 10}.{n % 10:01d}00002", 10, n, [0, n * 50],
                        sdu_bytes=n * 10)
                for n in range(1, 21)
            ],
        )
        write_csv(
            self.tmp / "exp" / "run1" / "traffic" / "sender.csv",
            audit.SENDER_HEADER,
            [[1.0, 0.0, 0, 0, 12500, 12500, 0.1, 0.0, 0.0]],
        )

    def test_repeated_audits_of_the_same_bytes_agree_exactly(self):
        first = audit.audit_evidence_root(self.tmp).to_json()
        second = audit.audit_evidence_root(self.tmp).to_json()
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )

    def test_report_carries_hashes_and_the_final_verdict(self):
        report = audit.audit_evidence_root(self.tmp)
        self.assertEqual(len(report.runs), 1)
        hashes = {item.sha256 for item in report.runs[0].provenance}
        self.assertTrue(all(len(value) == 64 for value in hashes))
        self.assertIs(report.coverage_verdict, audit.CoverageVerdict.INSUFFICIENT)
        self.assertIs(
            report.pre_action_status, audit.PreActionQualification.UNRESOLVED
        )
        self.assertEqual(report.recommended_pre_action_source, "UNRESOLVED")
        self.assertTrue(report.calibration_required)

    def test_offered_load_is_summarized_but_never_joined(self):
        report = audit.audit_evidence_root(self.tmp)
        run = report.runs[0]
        self.assertEqual(run.offered_load.distinct_frame_bytes, (12500,))
        refused = run.refused_alignments
        self.assertEqual(len(refused), 1)
        self.assertEqual(refused[0]["right_source"], "traffic/sender.csv")
        self.assertEqual(refused[0]["refusal"], "ClockDomainError")
        for alignment in run.alignments:
            self.assertNotIn("sender", alignment.right_source)


class ImportPurityTests(unittest.TestCase):
    def test_import_touches_nothing(self):
        probe = textwrap.dedent(
            """
            import builtins, os, socket, subprocess, sys, glob, pathlib
            calls = []
            def block(name):
                def guard(*a, **k):
                    calls.append((name, a[:2]))
                    raise AssertionError(name + " called at import time")
                return guard
            builtins.open = block("open")
            io_open = None
            os.scandir = block("os.scandir")
            os.listdir = block("os.listdir")
            os.walk = block("os.walk")
            os.system = block("os.system")
            glob.glob = block("glob.glob")
            glob.iglob = block("glob.iglob")
            pathlib.Path.glob = block("Path.glob")
            pathlib.Path.open = block("Path.open")
            pathlib.Path.read_text = block("Path.read_text")
            pathlib.Path.write_text = block("Path.write_text")
            pathlib.Path.iterdir = block("Path.iterdir")
            pathlib.Path.stat = block("Path.stat")
            pathlib.Path.mkdir = block("Path.mkdir")
            subprocess.Popen = block("subprocess.Popen")
            subprocess.run = block("subprocess.run")
            socket.socket = block("socket.socket")
            socket.create_connection = block("socket.create_connection")

            sys.path.insert(0, sys.argv[1])
            import audit_ue_state_evidence  # noqa: F401

            forbidden = [m for m in ("torch", "numpy.cuda", "docker", "carla")
                         if m in sys.modules]
            assert not forbidden, forbidden
            assert not calls, calls
            print("CLEAN")
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "probe.py"
            script.write_text(probe)
            completed = subprocess.run(
                [sys.executable, str(script), str(Path(audit.__file__).parent)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(
            completed.returncode, 0, msg=completed.stdout + completed.stderr
        )
        self.assertIn("CLEAN", completed.stdout)

    def test_module_pulls_in_no_heavy_or_project_runtime_dependency(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, sys.argv[1]);"
                " import audit_ue_state_evidence;"
                " print(sorted(m for m in sys.modules"
                " if m.split('.')[0] in"
                " {'torch','carla','docker','scipy','pandas','sklearn'}))",
                str(Path(audit.__file__).parent),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]")


@unittest.skipUnless(
    REAL_EVIDENCE_ROOT.is_dir(), f"real evidence tree {REAL_EVIDENCE_ROOT} not present"
)
class RealEvidenceSmokeTests(unittest.TestCase):
    """Bounded, read-only checks against the retained traces."""

    @classmethod
    def setUpClass(cls):
        cls.runs = audit.discover_runs(REAL_EVIDENCE_ROOT)

    def test_every_discovered_run_has_both_backlog_traces(self):
        self.assertGreater(len(self.runs), 0)
        for run in self.runs:
            self.assertIsNotNone(run.ue_csv("NRUE_MAC_BSR_STATUS.csv"), run.run_id)
            self.assertIsNotNone(
                run.ue_csv("NRUE_MAC_RLC_BUFFER_STATUS.csv"), run.run_id
            )

    def test_real_headers_match_the_declared_schemas(self):
        for run in self.runs:
            for name, expected in (
                ("NRUE_MAC_RLC_BUFFER_STATUS.csv", audit.RLC_BUFFER_HEADER),
                ("NRUE_MAC_BSR_STATUS.csv", audit.BSR_STATUS_HEADER),
                ("NRUE_MAC_DCI_GRANT.csv", audit.DCI_GRANT_HEADER),
                ("UE_PHY_UL_PAYLOAD_TX_BITS.csv", audit.TX_BITS_HEADER),
            ):
                path = run.ue_csv(name)
                if path is None:
                    continue
                with path.open("r", newline="") as handle:
                    header = tuple(next(__import__("csv").reader(handle)))
                self.assertEqual(header, expected, f"{run.run_id}/{name}")

    def test_no_run_retains_an_application_enqueue_timestamp(self):
        for run in self.runs:
            self.assertFalse(
                audit.has_enqueue_instant_evidence(run),
                f"{run.run_id} unexpectedly retains an enqueue trace; the "
                f"pre-action verdict must be recomputed",
            )

    def test_auditing_a_run_leaves_the_evidence_byte_identical(self):
        smallest = min(
            self.runs,
            key=lambda run: run.ue_csv("NRUE_MAC_RLC_BUFFER_STATUS.csv").stat().st_size,
        )
        watched = sorted(smallest.root.rglob("*.csv"))
        self.assertGreater(len(watched), 0)
        before = {
            path: (audit.sha256_file(path), path.stat().st_size) for path in watched
        }
        audit.audit_run(smallest, REAL_EVIDENCE_ROOT)
        after = {
            path: (audit.sha256_file(path), path.stat().st_size) for path in watched
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
