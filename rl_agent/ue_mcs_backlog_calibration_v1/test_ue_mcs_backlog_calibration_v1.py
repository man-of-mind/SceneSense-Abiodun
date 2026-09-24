#!/usr/bin/env python3
"""Offline tests for the UE-local [previous UL MCS, pre-enqueue backlog] study.

CPU only: no OAI, no Docker, no CARLA, no CUDA, no radio, no network beyond a
loopback socket. The design tests exist to *prove the matrix and ordering*
before any live launch is permitted.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rl_agent.ue_mcs_backlog_calibration_v1 import contract as C  # noqa: E402
from rl_agent.ue_mcs_backlog_calibration_v1 import decision_join as J  # noqa: E402

CONFIG = json.loads((HERE / "config_v1.json").read_text())
PORTS = CONFIG["traffic"]["ports"]


def plan():
    return C.build_cell_plan(C.resolve_load_tiers(REPO), ports=PORTS, seed=20260924)


class LoadTierTests(unittest.TestCase):
    def test_tiers_resolve_from_the_frozen_catalogue(self):
        tiers = {t.tier: t for t in C.resolve_load_tiers(REPO)}
        self.assertEqual(tiers["low"].action_id, 71)
        self.assertEqual(tiers["medium"].action_id, 50)
        self.assertEqual(tiers["high"].action_id, 30)
        self.assertEqual(tiers["low"].payload_bytes, 6_229)
        self.assertEqual(tiers["medium"].payload_bytes, 263_507)
        self.assertEqual(tiers["high"].payload_bytes, 880_567)

    def test_tiers_are_strictly_increasing(self):
        sizes = [t.payload_bytes for t in C.resolve_load_tiers(REPO)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(len(set(sizes)), 3)

    def test_a_tier_whose_identity_does_not_reconcile_is_refused(self):
        original = dict(C.EXPECTED_TIER_PROFILE_IDS)
        try:
            C.EXPECTED_TIER_PROFILE_IDS = {**original, "low": "not_the_real_profile"}
            with self.assertRaises(C.ContractError):
                C.resolve_load_tiers(REPO)
        finally:
            C.EXPECTED_TIER_PROFILE_IDS = original


class DesignMatrixTests(unittest.TestCase):
    """These are the gate: the amended matrix must be provably balanced."""

    def setUp(self):
        self.cells = plan()
        self.audit = C.audit_cell_plan(self.cells)

    def test_matrix_is_three_orders_by_two_channels_by_two_repetitions(self):
        self.assertEqual(len(self.cells), 12)
        self.assertEqual(len(C.BLOCK_ORDERS), 3)
        self.assertEqual(len(C.CONTRAST_PROFILE_IDS), 2)
        self.assertEqual(C.REPETITIONS, 2)
        keys = {(c.profile_id, c.order_index, c.repetition) for c in self.cells}
        self.assertEqual(len(keys), 12)

    def test_load_varies_within_every_cell(self):
        # The defect the amendment exists to fix: load must not be a purely
        # between-cell factor.
        for cell in self.cells:
            self.assertEqual(set(cell.sequence), set(C.TIER_ORDER), cell.cell_id)
            self.assertEqual(len(cell.sequence), 3)
        self.assertTrue(self.audit["load_is_within_cell"])

    def test_every_tier_appears_once_per_cell(self):
        for cell in self.cells:
            self.assertEqual(Counter(cell.sequence),
                             Counter(C.TIER_ORDER), cell.cell_id)

    def test_position_is_balanced_within_each_channel(self):
        for channel in C.CONTRAST_PROFILE_IDS:
            counts: Counter = Counter()
            for cell in self.cells:
                if cell.profile_id != channel:
                    continue
                for position, tier in enumerate(cell.sequence):
                    counts[(tier, position)] += 1
            self.assertEqual(len(counts), 9, channel)
            self.assertEqual(set(counts.values()), {2}, channel)

    def test_all_six_ordered_transitions_occur_equally_in_each_channel(self):
        expected = {f"{a}->{b}" for a in C.TIER_ORDER for b in C.TIER_ORDER if a != b}
        self.assertEqual(len(expected), 6)
        for channel in C.CONTRAST_PROFILE_IDS:
            counts: Counter = Counter()
            for cell in self.cells:
                if cell.profile_id == channel:
                    counts.update("->".join(p) for p in cell.transitions)
            self.assertEqual(set(counts), expected, channel)
            self.assertEqual(set(counts.values()), {2}, channel)

    def test_repetition_one_reverses_repetition_zero(self):
        by_key = {(c.profile_id, c.order_index, c.repetition): c for c in self.cells}
        for channel in C.CONTRAST_PROFILE_IDS:
            for order_index in range(len(C.BLOCK_ORDERS)):
                base = by_key[(channel, order_index, 0)]
                rep = by_key[(channel, order_index, 1)]
                self.assertEqual(rep.sequence, tuple(reversed(base.sequence)))

    def test_block_orders_are_a_latin_square(self):
        for position in range(3):
            self.assertEqual(
                {order[position] for order in C.BLOCK_ORDERS}, set(C.TIER_ORDER))

    def test_run_order_is_seeded_and_reproducible(self):
        again = C.build_cell_plan(C.resolve_load_tiers(REPO), ports=PORTS,
                                  seed=20260924)
        self.assertEqual([c.cell_id for c in self.cells], [c.cell_id for c in again])
        different = C.build_cell_plan(C.resolve_load_tiers(REPO), ports=PORTS,
                                      seed=1)
        self.assertNotEqual([c.cell_id for c in self.cells],
                            [c.cell_id for c in different])

    def test_run_order_does_not_change_the_design(self):
        different = C.build_cell_plan(C.resolve_load_tiers(REPO), ports=PORTS, seed=7)
        self.assertEqual(C.audit_cell_plan(different)["tier_block_counts"],
                         self.audit["tier_block_counts"])

    def test_channels_are_the_two_contrasting_stable_profiles(self):
        self.assertEqual(set(C.CONTRAST_PROFILE_IDS),
                         {"FAVORABLE_STABLE", "ADVERSE_STABLE"})
        self.assertTrue(set(C.CONTRAST_PROFILE_IDS) <= set(C.PROFILE_IDS))

    def test_blocks_carry_distinct_ports_per_tier(self):
        for cell in self.cells:
            ports = {b.tier: b.port for b in cell.blocks}
            self.assertEqual(len(set(ports.values())), 3, cell.cell_id)

    def test_analysis_windows_are_disjoint_inside_a_block(self):
        self.assertLessEqual(
            C.TRANSIENT_DECISIONS + C.STEADY_STATE_DECISIONS, C.FRAMES_PER_BLOCK)

    def test_frames_per_cell_follows_from_the_design(self):
        self.assertEqual(C.FRAMES_PER_CELL,
                         C.FRAMES_PER_BLOCK * C.BLOCKS_PER_CELL)
        for cell in self.cells:
            self.assertEqual(sum(b.frames for b in cell.blocks), C.FRAMES_PER_CELL)

    def test_block_first_frame_indices_are_contiguous(self):
        for cell in self.cells:
            starts = [b.first_frame_index for b in cell.blocks]
            self.assertEqual(starts, [0, C.FRAMES_PER_BLOCK, 2 * C.FRAMES_PER_BLOCK])


class ProfileResolutionTests(unittest.TestCase):
    def test_profiles_resolve_and_match_their_registered_trace_ids(self):
        profiles = {p.profile_id: p
                    for p in C.resolve_profiles(REPO, C.FRAMES_PER_CELL)}
        for name in C.CONTRAST_PROFILE_IDS:
            self.assertIn(name, profiles)
            self.assertEqual(len(profiles[name].samples), C.FRAMES_PER_CELL)
            self.assertTrue(profiles[name].trace_sha256)

    def test_profile_prefix_is_the_registered_values(self):
        profiles = {p.profile_id: p
                    for p in C.resolve_profiles(REPO, C.FRAMES_PER_CELL)}
        adverse = profiles["ADVERSE_STABLE"]
        self.assertEqual([s["step_index"] for s in adverse.samples[:3]], [0, 1, 2])

    def test_requesting_more_samples_than_the_trace_holds_is_refused(self):
        with self.assertRaises(C.ContractError):
            C.resolve_profiles(REPO, 10 ** 7)


class ClockBridgeTests(unittest.TestCase):
    def test_bridge_is_built_from_same_event_pairs(self):
        rows = [{"time": "01:00:00.000000", "mono_sec": "100", "mono_nsec": "0",
                 "ue_id": "0", "rb_id": "1", "sdu_bytes": "10"},
                {"time": "01:00:01.000000", "mono_sec": "101", "mono_nsec": "0",
                 "ue_id": "0", "rb_id": "1", "sdu_bytes": "10"}]
        bridge = J.build_clock_bridge(rows)
        self.assertEqual(bridge.samples, 2)
        tod = J.tracer_time_of_day_ns("01:00:00.000000")
        self.assertEqual(bridge.to_monotonic(tod), 100 * 10 ** 9)

    def test_missing_pdcp_rows_refuse_the_join(self):
        with self.assertRaises(J.JoinError):
            J.build_clock_bridge([])

    def test_residual_spread_is_reported(self):
        rows = [{"time": "01:00:00.000000", "mono_sec": "100", "mono_nsec": "0"},
                {"time": "01:00:01.000000", "mono_sec": "101", "mono_nsec": "500"}]
        bridge = J.build_clock_bridge(rows)
        self.assertGreaterEqual(bridge.residual_max_ns, 0.0)


class CausalJoinTests(unittest.TestCase):
    def _bridge(self):
        return J.ClockBridge(offset_ns=0, samples=10, residual_p50_ns=0.0,
                             residual_p95_ns=0.0, residual_max_ns=0.0, day_wraps=0)

    def _sender_row(self, decision_ns, **over):
        row = {"cell_id": "c", "decision_index": "0", "block_index": "0",
               "tier": "low", "action_id": "71", "frame_index_in_block": "0",
               "is_first_frame_of_block": "True", "decisions_since_transition": "0",
               "previous_tier": "", "decision_monotonic_ns": str(decision_ns),
               "schedule_lag_ms": "0.0", "payload_bytes": "6229",
               "chunks_per_frame": "1", "chunks_sent": "1", "chunks_dropped": "0",
               "terminal_reason": "ALL_CHUNKS_HANDED_TO_SOCKET"}
        row.update(over)
        return row

    def _meta(self):
        return {"cell_id": "c", "profile_id": "ADVERSE_STABLE", "order_index": 0,
                "repetition": 0, "sequence": ("low", "medium", "high")}

    def _grant(self, ns, mcs, harq_round=0):
        return J.UlGrant(monotonic_ns=ns, mcs=mcs, mcs_table=0, rb_size=10,
                         tbs=100, harq_pid=1, ndi=1, rv=0, harq_round=harq_round)

    def test_backlog_and_mcs_must_precede_the_decision(self):
        ticks = [J.BacklogTick(monotonic_ns=1_000, total_bytes=500, per_lcid=()),
                 J.BacklogTick(monotonic_ns=9_000, total_bytes=999, per_lcid=())]
        grants = [self._grant(2_000, 12), self._grant(9_500, 28)]
        out = J.join_cell(cell_meta=self._meta(),
                          sender_rows=[self._sender_row(5_000)],
                          ticks=ticks, grants=grants, arrivals_by_block={},
                          budget_ms=170.0)
        # The 9_000 tick and 9_500 grant are AFTER the decision and must not win.
        self.assertEqual(out[0]["pre_enqueue_backlog_bytes"], 500)
        self.assertEqual(out[0]["previous_ul_mcs"], 12)
        self.assertGreater(out[0]["backlog_age_ms"], 0)

    def test_retransmission_grants_never_enter_the_feature(self):
        rows = [
            {**{k: "0" for k in C.DCI_GRANT_HEADER}, "time": "01:00:00.000000",
             "direction": "1", "mcs": "9", "mcs_table": "0", "round": "0",
             "rb_size": "5", "tbs": "50", "harq_pid": "0", "ndi": "1", "rv": "0"},
            {**{k: "0" for k in C.DCI_GRANT_HEADER}, "time": "01:00:00.100000",
             "direction": "1", "mcs": "2", "mcs_table": "0", "round": "1",
             "rb_size": "5", "tbs": "50", "harq_pid": "0", "ndi": "1", "rv": "2"},
        ]
        grants, counts = J.build_ul_grants(rows, self._bridge())
        self.assertEqual(counts["retransmission"], 1)
        self.assertEqual([g.mcs for g in grants], [9])

    def test_downlink_grants_are_excluded(self):
        rows = [{**{k: "0" for k in C.DCI_GRANT_HEADER},
                 "time": "01:00:00.000000", "direction": "0", "mcs": "20",
                 "mcs_table": "0", "round": "0"}]
        grants, counts = J.build_ul_grants(rows, self._bridge())
        self.assertEqual(grants, [])
        self.assertEqual(counts["non_ul"], 1)

    def test_stale_mcs_becomes_missing_and_is_never_forward_filled(self):
        stale_ns = int((C.MCS_MAX_AGE_MS + 50) * 1e6)
        out = J.join_cell(
            cell_meta=self._meta(),
            sender_rows=[self._sender_row(stale_ns + 1_000)],
            ticks=[J.BacklogTick(monotonic_ns=0, total_bytes=0, per_lcid=())],
            grants=[self._grant(1_000, 14)], arrivals_by_block={}, budget_ms=170.0)
        self.assertIsNone(out[0]["previous_ul_mcs"])
        self.assertEqual(out[0]["mcs_status"], "MISSING_STALE")

    def test_missing_mcs_is_never_coerced_to_zero(self):
        out = J.join_cell(
            cell_meta=self._meta(), sender_rows=[self._sender_row(10)],
            ticks=[], grants=[], arrivals_by_block={}, budget_ms=170.0)
        self.assertIsNone(out[0]["previous_ul_mcs"])
        self.assertEqual(out[0]["mcs_status"], "MISSING_NO_PRIOR_GRANT")
        self.assertNotEqual(out[0]["previous_ul_mcs"], 0)
        audit = J.causal_audit(out)
        self.assertEqual(audit["missing_mcs_coerced_to_zero"], 0)

    def test_mcs_zero_is_preserved_as_a_real_observation(self):
        # MCS 0 is a real modulation index and must survive the missingness rule.
        out = J.join_cell(
            cell_meta=self._meta(), sender_rows=[self._sender_row(2_000)],
            ticks=[], grants=[self._grant(1_000, 0)], arrivals_by_block={},
            budget_ms=170.0)
        self.assertEqual(out[0]["previous_ul_mcs"], 0)
        self.assertEqual(out[0]["mcs_status"], "OBSERVED")

    def test_raw_backlog_bytes_are_retained_unscaled(self):
        out = J.join_cell(
            cell_meta=self._meta(), sender_rows=[self._sender_row(5_000)],
            ticks=[J.BacklogTick(monotonic_ns=1_000, total_bytes=123_456,
                                 per_lcid=())],
            grants=[], arrivals_by_block={}, budget_ms=170.0)
        self.assertEqual(out[0]["pre_enqueue_backlog_bytes"], 123_456)

    def test_causal_audit_flags_nothing_on_a_clean_join(self):
        out = J.join_cell(
            cell_meta=self._meta(), sender_rows=[self._sender_row(5_000)],
            ticks=[J.BacklogTick(monotonic_ns=1_000, total_bytes=10, per_lcid=())],
            grants=[self._grant(2_000, 11)], arrivals_by_block={}, budget_ms=170.0)
        audit = J.causal_audit(out)
        self.assertTrue(audit["all_joined_observations_precede_decision"])
        self.assertEqual(audit["negative_observation_age"], 0)
        self.assertEqual(audit["stale_mcs_used_as_observed"], 0)

    def test_arrivals_are_looked_up_per_block(self):
        # 5.000 ms after the decision, in nanoseconds.
        arrivals = {1: {0: J.FrameArrival(first_monotonic_ns=6_000,
                                          last_monotonic_ns=5_000 + 5_000_000,
                                          unique_chunks=1, complete=True)}}
        row = self._sender_row(5_000, block_index="1")
        out = J.join_cell(cell_meta=self._meta(), sender_rows=[row], ticks=[],
                          grants=[], arrivals_by_block=arrivals, budget_ms=170.0)
        self.assertTrue(out[0]["complete_reassembly"])
        self.assertAlmostEqual(out[0]["uplink_latency_ms"], 5.0)
        self.assertTrue(out[0]["within_transport_budget"])

    def test_incomplete_frames_report_no_latency_rather_than_a_guess(self):
        arrivals = {0: {0: J.FrameArrival(first_monotonic_ns=6_000,
                                          last_monotonic_ns=7_000,
                                          unique_chunks=1, complete=False)}}
        out = J.join_cell(cell_meta=self._meta(),
                          sender_rows=[self._sender_row(5_000)], ticks=[],
                          grants=[], arrivals_by_block=arrivals, budget_ms=170.0)
        self.assertIsNone(out[0]["uplink_latency_ms"])
        self.assertEqual(out[0]["terminal_outcome"], "INCOMPLETE_REASSEMBLY")


class SenderContractTests(unittest.TestCase):
    def test_wire_contract_is_imported_not_restated(self):
        from rl_agent.ue_n3_structured_udp_receiver import HEADER, MAGIC
        from rl_agent.ue_mcs_backlog_calibration_v1 import tagged_sender
        self.assertIs(tagged_sender.HEADER, HEADER)
        self.assertIs(tagged_sender.MAGIC, MAGIC)

    def test_payload_is_deterministic_for_a_tier(self):
        from rl_agent.ue_mcs_backlog_calibration_v1 import tagged_sender
        a = tagged_sender.build_payload(4096, 11)
        b = tagged_sender.build_payload(4096, 11)
        c = tagged_sender.build_payload(4096, 12)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_chunking_matches_the_declared_chunks_per_frame(self):
        from rl_agent.ue_mcs_backlog_calibration_v1 import tagged_sender
        for tier in C.resolve_load_tiers(REPO):
            spans = tagged_sender.chunk_spans(tier.payload_bytes, C.CHUNK_BYTES)
            self.assertEqual(len(spans), tier.chunks_per_frame, tier.tier)
            self.assertEqual(sum(size for _, size in spans), tier.payload_bytes)


class LoopbackBlockTransitionTests(unittest.TestCase):
    """End-to-end over loopback: transitions must be sharp and contiguous."""

    def test_three_block_run_switches_load_without_a_gap(self):
        tiers = {t.tier: t for t in C.resolve_load_tiers(REPO)}
        frames = 8
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blocks = []
            for index, tier_name in enumerate(("low", "medium", "low")):
                tier = tiers[tier_name]
                blocks.append({
                    "block_index": index, "tier": tier_name,
                    "action_id": tier.action_id,
                    "payload_bytes": tier.payload_bytes,
                    "chunks_per_frame": tier.chunks_per_frame,
                    "frames": frames, "first_frame_index": index * frames,
                    "port": 5591 + index,
                })
            plan_path = root / "blocks.json"
            plan_path.write_text(json.dumps(blocks))
            csv_path = root / "decisions.csv"
            summary_path = root / "summary.json"
            result = subprocess.run(
                [sys.executable, "-m",
                 "rl_agent.ue_mcs_backlog_calibration_v1.tagged_sender",
                 "--cell-id", "loopback", "--bind-host", "127.0.0.1",
                 "--remote-host", "127.0.0.1", "--block-plan", str(plan_path),
                 "--payload-seed", "1", "--fps", "20",
                 "--log-csv", str(csv_path), "--summary-json", str(summary_path)],
                cwd=str(REPO), text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-800:])

            import csv as _csv
            with csv_path.open() as handle:
                rows = list(_csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * frames)
            self.assertEqual([r["tier"] for r in rows[:frames]], ["low"] * frames)
            self.assertEqual([r["tier"] for r in rows[frames:2 * frames]],
                             ["medium"] * frames)

            # Decision indices are one contiguous timeline across blocks.
            self.assertEqual([int(r["decision_index"]) for r in rows],
                             list(range(3 * frames)))

            # The transition lands on its scheduled decision, with no extra gap:
            # the step across a boundary is the same period as a step inside a
            # block. A process restart between blocks would show up here.
            stamps = [int(r["decision_monotonic_ns"]) for r in rows]
            steps = [b - a for a, b in zip(stamps, stamps[1:])]
            boundary = [steps[frames - 1], steps[2 * frames - 1]]
            interior = [s for i, s in enumerate(steps)
                        if i not in (frames - 1, 2 * frames - 1)]
            self.assertLess(max(boundary), 2 * (sum(interior) / len(interior)))

            first_of_block = [r for r in rows if r["is_first_frame_of_block"] == "True"]
            self.assertEqual(len(first_of_block), 3)
            self.assertEqual(first_of_block[1]["previous_tier"], "low")
            self.assertEqual(first_of_block[2]["previous_tier"], "medium")

            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["sequence"], ["low", "medium", "low"])
            # The same tier reuses the identical payload, so payload is never a
            # hidden variable across blocks or cells.
            digests = {b["tier"]: b["payload_sha256"] for b in summary["blocks"]}
            self.assertEqual(summary["blocks"][0]["payload_sha256"],
                             summary["blocks"][2]["payload_sha256"])
            self.assertNotEqual(digests["low"], digests["medium"])


class ConfigTests(unittest.TestCase):
    def test_config_pins_one_port_per_tier(self):
        self.assertEqual(set(PORTS), set(C.TIER_ORDER))
        self.assertEqual(len(set(PORTS.values())), 3)

    def test_sinr_policy_is_the_configured_mcs_policy(self):
        self.assertEqual(CONFIG["radio"]["mcs_policy"], "sinr")

    def test_ue_events_cover_every_required_observation(self):
        events = set(CONFIG["telemetry"]["events"]["ue"])
        self.assertIn("NRUE_MAC_DCI_GRANT", events)
        self.assertIn("NRUE_MAC_RLC_BUFFER_STATUS", events)
        self.assertIn("NR_PDCP_TX_SDU", events)

    def test_restore_value_is_the_registered_cold_channel(self):
        self.assertEqual(
            CONFIG["actuator"]["clean_and_restore_commanded_noise_power_db"], "-50")


if __name__ == "__main__":
    unittest.main()
