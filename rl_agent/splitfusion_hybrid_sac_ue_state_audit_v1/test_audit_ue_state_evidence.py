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
    """A2. Qualification must come from parsed content, never a filename."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _build(
        self,
        *,
        event="NR_PDCP_TX_SDU",
        enqueue_rows=((("00:00:01.000000", 1, 0, 0, 1, 1200)),),
        decisions=None,
        with_rlc=True,
        with_decisions=True,
        root="exp/r",
    ):
        """A run that fully qualifies unless a caller removes something."""
        run_root = self.tmp / root
        csv_dir = run_root / "ttracer" / "ue" / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        if with_rlc:
            write_csv(
                csv_dir / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
                audit.RLC_BUFFER_HEADER,
                [rlc_row("00:00:01.000000", 1, 0, 4, 1, 4096)],
            )
        if event is not None:
            header = audit.ENQUEUE_EVENT_HEADERS.get(
                event, audit.SERVICE_DEQUEUE_EVENT_HEADERS.get(event)
            )
            write_csv(csv_dir / f"{event}.csv", header, list(enqueue_rows))
        if with_decisions:
            if decisions is None:
                decisions = [
                    ["d0", 7, 0, 1, 4, 1, 100, 110, 120, 130, 140],
                    ["d1", 8, 0, 1, 4, 1, 200, 200, 200, 230, 240],
                ]
            write_csv(
                run_root / "decisions" / "decision_log.csv",
                audit.DECISION_RECORD_HEADER,
                decisions,
            )
        return audit.LogicalRun(
            run_id=root, root=run_root, ue_csv_dir=csv_dir,
            gnb_csv_dir=None, traffic_sender=None,
        )

    # -- the positive control: the check is satisfiable, not vacuous ----
    def test_a_complete_run_does_qualify(self):
        check = audit.qualify_enqueue_evidence(self._build())
        self.assertTrue(check.qualified, check.first_failure)
        self.assertEqual(check.admitted_decisions, 2)
        self.assertEqual(check.ordered_decisions, 2)
        self.assertIsNone(check.first_failure)

    def test_equal_instants_are_allowed_only_where_the_chain_permits(self):
        # t_measure == t_available == t_state_commit is admissible.
        check = audit.qualify_enqueue_evidence(
            self._build(decisions=[["d0", 1, 0, 1, 4, 1, 5, 5, 5, 6, 7]])
        )
        self.assertTrue(check.qualified, check.first_failure)

    # -- filename existence must not qualify ---------------------------
    def test_header_only_file_does_not_qualify(self):
        check = audit.qualify_enqueue_evidence(self._build(enqueue_rows=()))
        self.assertFalse(check.qualified)
        self.assertIn("NONEMPTY_PARSED_RECORDS", check.first_failure)

    def test_wrong_schema_does_not_qualify(self):
        run = self._build()
        path = run.ue_csv_dir / "NR_PDCP_TX_SDU.csv"
        path.write_text("time,mono_sec,ue_id\n00:00:01.000000,1,0\n")
        check = audit.qualify_enqueue_evidence(run)
        self.assertFalse(check.qualified)
        self.assertIn("SCHEMA_EXACT", check.first_failure)

    def test_blank_source_timestamp_fails_closed(self):
        check = audit.qualify_enqueue_evidence(
            self._build(enqueue_rows=[["00:00:01.000000", "", 0, 0, 1, 1200]])
        )
        self.assertFalse(check.qualified)
        self.assertIn("SOURCE_TIMESTAMP_PRESENT", check.first_failure)

    def test_blank_availability_timestamp_fails_closed(self):
        check = audit.qualify_enqueue_evidence(
            self._build(enqueue_rows=[["", 1, 0, 0, 1, 1200]])
        )
        self.assertFalse(check.qualified)
        self.assertIn("AVAILABILITY_TIMESTAMP_PRESENT", check.first_failure)

    # -- dequeue is not enqueue ----------------------------------------
    def test_dequeue_only_evidence_does_not_qualify_enqueue_ordering(self):
        run = self._build(
            event="NR_RLC_TX_DEQUEUE",
            enqueue_rows=[["00:00:01.000000", 1, 0, 4, 900]],
        )
        check = audit.qualify_enqueue_evidence(run)
        self.assertFalse(check.qualified)
        self.assertIn("FILE_PRESENT", check.first_failure)
        # And the refusal must say *why* rather than claim nothing was kept.
        self.assertIn("service/dequeue", check.first_failure)

    def test_dequeue_is_not_listed_as_an_enqueue_event(self):
        self.assertNotIn("NR_RLC_TX_DEQUEUE", audit.ENQUEUE_INSTANT_EVENTS)
        self.assertIn("NR_RLC_TX_DEQUEUE", audit.SERVICE_DEQUEUE_EVENTS)
        self.assertEqual(
            set(audit.ENQUEUE_INSTANT_EVENTS) & set(audit.SERVICE_DEQUEUE_EVENTS),
            set(),
        )

    # -- identity ------------------------------------------------------
    def test_mismatched_ue_identity_does_not_qualify(self):
        check = audit.qualify_enqueue_evidence(
            # field 3 is ue_id: 99 does not appear in the backlog trace.
            self._build(enqueue_rows=[["00:00:01.000000", 1, 0, 99, 1, 1200]])
        )
        self.assertFalse(check.qualified)
        self.assertIn("UE_AND_RNTI_IDENTITY", check.first_failure)

    def test_blank_bearer_identity_does_not_qualify(self):
        check = audit.qualify_enqueue_evidence(
            self._build(enqueue_rows=[["00:00:01.000000", 1, 0, 0, "", 1200]])
        )
        self.assertFalse(check.qualified)
        self.assertIn("LCID_OR_DRB_IDENTITY", check.first_failure)

    def test_missing_decision_identity_fails_closed(self):
        check = audit.qualify_enqueue_evidence(self._build(with_decisions=False))
        self.assertFalse(check.qualified)
        self.assertIn("DECISION_AND_FRAME_IDENTITY", check.first_failure)

    def test_blank_frame_index_fails_closed(self):
        check = audit.qualify_enqueue_evidence(
            self._build(decisions=[["d0", "", 0, 1, 4, 1, 1, 2, 3, 4, 5]])
        )
        self.assertFalse(check.qualified)
        self.assertIn("DECISION_AND_FRAME_IDENTITY", check.first_failure)

    # -- cross-run contamination ---------------------------------------
    def test_evidence_from_one_run_cannot_qualify_another(self):
        good = self._build(root="exp/a")
        other = self._build(root="exp/b", event=None, with_decisions=False)
        # Point run B at run A's enqueue trace: the run roots differ, so the
        # structural identity check must refuse it.
        with self.assertRaises(audit.RunIsolationError):
            audit.require_same_run(
                other.ue_csv_dir / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
                good.ue_csv_dir / "NR_PDCP_TX_SDU.csv",
            )
        # And run B, which retains no enqueue trace of its own, stays unqualified
        # even though a sibling run under the same tree is fully qualified.
        self.assertTrue(audit.qualify_enqueue_evidence(good).qualified)
        self.assertFalse(audit.qualify_enqueue_evidence(other).qualified)

    def test_one_qualified_run_does_not_qualify_the_others_globally(self):
        self._build(root="exp/a")
        self._build(root="exp/b", event=None, with_decisions=False)
        write_csv(
            self.tmp / "exp" / "b" / "ttracer" / "ue" / "csv"
            / "NRUE_MAC_BSR_STATUS.csv",
            audit.BSR_STATUS_HEADER,
            [bsr_row("00:00:01.000000", 1, 0, [0])],
        )
        write_csv(
            self.tmp / "exp" / "a" / "ttracer" / "ue" / "csv"
            / "NRUE_MAC_BSR_STATUS.csv",
            audit.BSR_STATUS_HEADER,
            [bsr_row("00:00:01.000000", 1, 0, [0])],
        )
        report = audit.audit_evidence_root(self.tmp)
        self.assertEqual(report.total_run_count, 2)
        self.assertEqual(len(report.qualified_run_ids), 1)
        self.assertIs(
            report.pre_action_status, audit.PreActionQualification.UNRESOLVED
        )
        self.assertEqual(report.recommended_pre_action_source, "UNRESOLVED")

    # -- ordering ------------------------------------------------------
    def test_action_after_enqueue_is_refused(self):
        # t_action must strictly precede the payload enqueue it causes.
        ok, why = audit.check_decision_ordering(
            dict(zip(audit.DECISION_RECORD_HEADER,
                     ["d", 1, 0, 1, 4, 1, 1, 2, 3, 40, 30]))
        )
        self.assertFalse(ok)
        self.assertIn("t_action_mono_ns", why)

    def test_state_commit_after_action_is_refused(self):
        ok, why = audit.check_decision_ordering(
            dict(zip(audit.DECISION_RECORD_HEADER,
                     ["d", 1, 0, 1, 4, 1, 1, 2, 30, 3, 40]))
        )
        self.assertFalse(ok)
        self.assertIn("t_state_commit_mono_ns", why)

    def test_measurement_after_availability_is_refused(self):
        ok, why = audit.check_decision_ordering(
            dict(zip(audit.DECISION_RECORD_HEADER,
                     ["d", 1, 0, 1, 4, 1, 50, 2, 60, 70, 80]))
        )
        self.assertFalse(ok)
        self.assertIn("t_measure_mono_ns", why)

    def test_missing_instant_fails_closed_rather_than_defaulting_to_zero(self):
        for index in range(6, 11):
            row = ["d", 1, 0, 1, 4, 1, 10, 20, 30, 40, 50]
            row[index] = ""
            ok, why = audit.check_decision_ordering(
                dict(zip(audit.DECISION_RECORD_HEADER, row))
            )
            self.assertFalse(ok)
            self.assertIn("missing", why)

    def test_ordering_must_hold_for_every_admitted_decision(self):
        # One good decision must not carry a bad one.
        check = audit.qualify_enqueue_evidence(
            self._build(decisions=[
                ["d0", 1, 0, 1, 4, 1, 10, 20, 30, 40, 50],
                ["d1", 2, 0, 1, 4, 1, 10, 20, 30, 90, 50],
            ])
        )
        self.assertFalse(check.qualified)
        self.assertEqual(check.admitted_decisions, 2)
        self.assertEqual(check.ordered_decisions, 1)
        self.assertIn("CAUSAL_ORDERING", check.first_failure)

    def test_absent_enqueue_traces_are_detected(self):
        run = self._build(event=None, with_decisions=False)
        self.assertFalse(audit.has_enqueue_instant_evidence(run))


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
    def _make_run(self, index, sizes, *, ordering_ok):
        run = audit.RunAudit(run_id=f"r{index}")
        run.offered_load = audit.OfferedLoadProfile(
            present=True, row_count=10,
            distinct_frame_bytes=tuple(sorted(sizes)), distinct_period_s=(0.1,),
            within_action_range=any(
                audit.SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES
                <= value
                <= audit.SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES
                for value in sizes
            ),
            within_run3_support=any(
                audit.RUN3_MODELED_SUPPORT_MIN_BYTES
                <= value
                <= audit.RUN3_MODELED_SUPPORT_MAX_BYTES
                for value in sizes
            ),
            varies_within_run=len(set(sizes)) > 1,
            note="",
        )
        run.enqueue_evidence = audit.EnqueueEvidenceCheck(
            run_id=run.run_id,
            event_name="NR_PDCP_TX_SDU" if ordering_ok else None,
            outcomes=(
                audit.RequirementOutcome(
                    audit.EnqueueRequirement.CAUSAL_ORDERING, ordering_ok, ""
                ),
            ),
            admitted_decisions=3 if ordering_ok else 0,
            ordered_decisions=3 if ordering_ok else 0,
        )
        run.enqueue_instant_evidence = run.enqueue_evidence.qualified
        return run

    def _run_audit(self, sizes_per_run, ordering_ok=False):
        runs = [
            self._make_run(index, sizes, ordering_ok=ordering_ok)
            for index, sizes in enumerate(sizes_per_run)
        ]
        return audit.decide_coverage(runs)

    def test_retained_traffic_is_inside_the_low_end_not_below_the_range(self):
        # A1: the corrected authority puts 12.5 kB / 25 kB *inside* the action
        # range. The old "below the action range" claim must be gone.
        verdict, rationale = self._run_audit([[12_500], [25_000]])
        self.assertIs(verdict, audit.CoverageVerdict.INSUFFICIENT)
        self.assertNotIn("below the action range", rationale)
        self.assertIn("inside", rationale)

    def test_fixed_within_run_payload_is_insufficient_even_inside_the_range(self):
        verdict, rationale = self._run_audit([[12_500], [25_000]])
        self.assertIs(verdict, audit.CoverageVerdict.INSUFFICIENT)
        self.assertIn("constant", rationale)

    def test_payload_variation_alone_cannot_reach_action_conditioned(self):
        # A2: causal ordering is unresolved, so no amount of payload spread
        # may promote the verdict.
        verdict, _ = self._run_audit([[6_500, 400_000, 3_000_000]])
        self.assertIs(verdict, audit.CoverageVerdict.INSUFFICIENT)

    def test_action_conditioned_requires_both_variation_and_ordering(self):
        verdict, _ = self._run_audit(
            [[6_500, 400_000, 3_000_000]], ordering_ok=True
        )
        self.assertIs(verdict, audit.CoverageVerdict.ACTION_CONDITIONED)

    def test_one_unresolved_run_blocks_action_conditioned_for_all(self):
        runs = [
            self._make_run(0, [6_500, 400_000, 3_000_000], ordering_ok=True),
            self._make_run(1, [12_500], ordering_ok=False),
        ]
        verdict, rationale = audit.decide_coverage(runs)
        self.assertIsNot(verdict, audit.CoverageVerdict.ACTION_CONDITIONED)
        self.assertIn("unresolved in 1 of 2", rationale)

    def test_fixed_traffic_with_resolved_ordering_is_carrier_audit_only(self):
        verdict, _ = self._run_audit([[12_500]], ordering_ok=True)
        self.assertIs(verdict, audit.CoverageVerdict.CARRIER_AND_NORMALIZATION_ONLY)

    def test_no_offered_load_evidence_is_insufficient(self):
        run = audit.RunAudit(run_id="r0")
        run.offered_load = audit.OfferedLoadProfile(
            present=False, row_count=0, distinct_frame_bytes=(),
            distinct_period_s=(), within_action_range=False,
            within_run3_support=False, varies_within_run=False, note="",
        )
        verdict, _ = audit.decide_coverage([run])
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


class PayloadAuthorityTests(unittest.TestCase):
    """A1. The legacy PERMODEL range must not be able to come back."""

    def test_retired_permodel_constants_are_not_reachable(self):
        for name in ("SPLITFUSION_PAYLOAD_MIN_BYTES", "SPLITFUSION_PAYLOAD_MAX_BYTES"):
            self.assertFalse(
                hasattr(audit, name), f"{name} must stay retired"
            )

    def test_no_payload_authority_constant_holds_a_retired_value(self):
        live = {
            audit.SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES,
            audit.SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES,
            audit.RUN3_MODELED_SUPPORT_MIN_BYTES,
            audit.RUN3_MODELED_SUPPORT_MAX_BYTES,
        }
        self.assertEqual(live & set(audit.RETIRED_PERMODEL_PAYLOAD_BYTES), set())

    def test_constants_match_the_authoritative_files(self):
        # Verified against the catalogue and the Run-3 contract, not hard-coded
        # on trust. This is the check that fails if either file is re-frozen.
        summary = audit.verify_payload_authority(REPO_ROOT)
        self.assertEqual(summary["action_count"], 72)
        self.assertEqual(summary["action_payload_min_bytes"], 6_229)
        self.assertEqual(summary["action_payload_max_bytes"], 3_568_326)
        self.assertEqual(summary["run3_support_min_bytes"], 6_423)
        self.assertEqual(summary["run3_support_max_bytes"], 427_605)

    def test_drift_from_the_catalogue_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / audit.SPLITFUSION_ACTION_CATALOG_RELPATH
            catalog.parent.mkdir(parents=True)
            catalog.write_text(json.dumps({
                "profiles": [
                    {"payload": {"zstd_median_bytes": 1}},
                    {"payload": {"zstd_median_bytes": 2}},
                ]
            }))
            support = root / audit.RUN3_MODELED_SUPPORT_RELPATH
            support.parent.mkdir(parents=True)
            support.write_text(
                f"{audit.RUN3_MODELED_SUPPORT_SYMBOL} = (6423, 427605)\n"
            )
            with self.assertRaises(audit.PayloadAuthorityError):
                audit.verify_payload_authority(root)

    def test_retained_traffic_sits_inside_the_action_range(self):
        # The concrete A1 fact: 12.5 kB and 25 kB are inside the low end.
        for value in (12_500, 25_000):
            self.assertGreaterEqual(
                value, audit.SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES
            )
            self.assertLessEqual(
                value, audit.SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES
            )

    def test_run3_support_is_narrower_than_the_catalogue(self):
        self.assertGreater(
            audit.RUN3_MODELED_SUPPORT_MIN_BYTES,
            audit.SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES,
        )
        self.assertLess(
            audit.RUN3_MODELED_SUPPORT_MAX_BYTES,
            audit.SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES,
        )


class ChannelSignalTaxonomyTests(unittest.TestCase):
    """A4. UE downlink SNR and gNB uplink PUSCH SNR must stay distinct."""

    def test_ue_snr_is_downlink_not_uplink(self):
        self.assertIs(
            audit.UE_PHY_MEAS_SNR.direction, audit.LinkDirection.UE_DOWNLINK_RECEIVE
        )
        with self.assertRaises(audit.AuditError):
            audit.assert_direction_not_mislabelled(
                audit.UE_PHY_MEAS_SNR, audit.LinkDirection.GNB_UPLINK_RECEIVE
            )

    def test_gnb_pusch_snr_cannot_be_labelled_ue_visible(self):
        self.assertIs(
            audit.GNB_PUSCH_SNR.direction, audit.LinkDirection.GNB_UPLINK_RECEIVE
        )
        self.assertFalse(audit.GNB_PUSCH_SNR.ue_runtime_available)
        with self.assertRaises(audit.AuditError):
            audit.assert_ue_observable(audit.GNB_PUSCH_SNR)

    def test_ue_snr_is_ue_observable(self):
        audit.assert_ue_observable(audit.UE_PHY_MEAS_SNR)

    def test_no_candidate_is_labelled_bare_snr(self):
        for candidate in audit.CHANNEL_SIGNAL_CANDIDATES:
            self.assertNotEqual(candidate.display_label.strip().lower(), "snr")
            self.assertTrue(
                candidate.display_label.startswith(("UE downlink", "gNB received"))
            )

    def test_the_two_directions_never_share_a_display_label(self):
        labels = {c.display_label for c in audit.CHANNEL_SIGNAL_CANDIDATES}
        self.assertEqual(len(labels), len(audit.CHANNEL_SIGNAL_CANDIDATES))
        self.assertNotEqual(
            audit.UE_PHY_MEAS_SNR.display_label, audit.GNB_PUSCH_SNR.display_label
        )

    def test_w_cqi_is_not_a_standardized_index_but_csi_rs_cqi_is(self):
        self.assertFalse(audit.UE_PHY_MEAS_W_CQI.standardized_index)
        self.assertTrue(audit.CSI_RS_CQI.standardized_index)
        self.assertIn("not a standardized CQI", audit.UE_PHY_MEAS_W_CQI.caveat)

    def test_units_record_the_x10_conversion_asymmetry(self):
        self.assertIn("NOT scaled by 10", audit.UE_PHY_MEAS_SNR.units)
        self.assertIn("x10", audit.GNB_PUSCH_SNR.units)

    def test_every_candidate_cites_source_code(self):
        for candidate in audit.CHANNEL_SIGNAL_CANDIDATES:
            self.assertTrue(candidate.code_citation.strip())

    def test_ue_phy_meas_schema_matches_the_t_message_format(self):
        self.assertEqual(audit.UE_PHY_MEAS_HEADER[0], "time")
        for field in ("rsrp", "rssi", "snr", "rx_power", "noise_power", "w_cqi"):
            self.assertIn(field, audit.UE_PHY_MEAS_HEADER)


class BsrTemporalUseTests(unittest.TestCase):
    """A3. Post-multiplex is not universally invalid, only same-action."""

    def test_same_action_alignment_is_refused(self):
        with self.assertRaises(audit.ActionLeakageError):
            audit.assert_no_action_leakage(
                audit.BSR_STATUS_SOURCE, aligned_to_same_action=True
            )

    def test_lagged_alignment_is_not_refused(self):
        # A correctly lagged BSR may describe a *previous* action.
        audit.assert_no_action_leakage(
            audit.BSR_STATUS_SOURCE, aligned_to_same_action=False
        )

    def test_classification_names_both_uses(self):
        self.assertIs(
            audit.classify_bsr_temporal_use(
                audit.BSR_STATUS_SOURCE, aligned_to_same_action=True
            ),
            audit.BsrTemporalUse.SAME_ACTION_PRE_STATE,
        )
        self.assertIs(
            audit.classify_bsr_temporal_use(
                audit.BSR_STATUS_SOURCE, aligned_to_same_action=False
            ),
            audit.BsrTemporalUse.LAGGED_PREVIOUS_ACTION,
        )

    def test_rlc_stays_the_preferred_candidate(self):
        self.assertIn("preferred", audit.BSR_VS_RLC_PREFERENCE_NOTE)
        self.assertIn("less quantized", audit.BSR_VS_RLC_PREFERENCE_NOTE)

    def test_pre_multiplex_source_is_never_refused(self):
        for same_action in (True, False):
            audit.assert_no_action_leakage(
                audit.RLC_BUFFER_SOURCE, aligned_to_same_action=same_action
            )

    def test_classification_rejects_a_pre_multiplex_source(self):
        with self.assertRaises(audit.AuditError):
            audit.classify_bsr_temporal_use(
                audit.RLC_BUFFER_SOURCE, aligned_to_same_action=True
            )

    def test_post_multiplex_still_never_qualifies_as_pre_action_state(self):
        self.assertIs(
            audit.qualify_pre_action_source(
                audit.BSR_STATUS_SOURCE, enqueue_instant_evidence=True
            ),
            audit.PreActionQualification.DISQUALIFIED_ACTION_CONTAMINATED,
        )


class RetainedUePhyMeasTests(unittest.TestCase):
    """A4. Does the retained evidence actually contain UE_PHY_MEAS rows?"""

    def test_no_retained_run_holds_ue_phy_meas(self):
        if not REAL_EVIDENCE_ROOT.is_dir():
            self.skipTest("evidence tree not present")
        found = sorted(REAL_EVIDENCE_ROOT.glob("**/ttracer/ue/csv/UE_PHY_MEAS.csv"))
        self.assertEqual(
            found, [], "expectation was that no run retains UE_PHY_MEAS"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
