"""Offline tests for the Phase-2C qualification's pure analysis functions.

No radio, CARLA, CUDA or model process is started.
"""

from __future__ import annotations

import copy
import unittest

from . import phase2_telemetry_qualification as Q
from .test_ue_telemetry_provider_v2 import Harness, _valid_tick


def _raw_rlc(rows):
    return [{"time": t, "rnti": str(r), "ue_id": str(u), "frame": str(f),
             "slot": str(s), "lcid": str(l), "bytes_in_buffer": str(b)}
            for t, r, u, f, s, l, b in rows]


class ParsingTest(unittest.TestCase):
    def test_identity_round_trip_from_live_provider(self) -> None:
        h = Harness()
        h.dci(mcs=13, ndi=0)
        _valid_tick(h, 9, 3, values=(5, 6, 7))
        h.rlc(frame=9, slot=4, lcid=1, bytes_=0)
        grant = h.provider.snapshot().dci[-1]
        parsed = Q.parse_grant_identity(grant.grant_identity)
        self.assertEqual((parsed["dci_frame"], parsed["dci_slot"], parsed["harq_pid"]),
                         (499, 0, 1))
        self.assertRegex(parsed["tod"], r"^\d\d:\d\d:\d\d\.\d{6}$")
        tick = Q.parse_rlc_source(h.provider.snapshot().rlc[-1].source)
        self.assertEqual((tick["frame"], tick["slot"], tick["lcids"]), (9, 3, 3))
        self.assertRegex(tick["tod"], r"^\d\d:\d\d:\d\d\.\d{6}$")


class RawReplayTest(unittest.TestCase):
    def test_raw_tick_replay_drops_open_group_and_filters_ue(self) -> None:
        rows = _raw_rlc([
            ("10:00:00.000001", 7, 0, 1, 1, 1, 10),
            ("10:00:00.000001", 7, 0, 1, 1, 4, 5),
            ("10:00:00.000002", 9, 0, 1, 1, 1, 999),   # other UE
            ("10:00:00.000500", 7, 0, 1, 2, 1, 0),
        ])
        ticks = Q.raw_rlc_ticks(rows, rnti=7, ue_id=0)
        self.assertEqual(ticks, {(1, 1, "10:00:00.000001"):
                                 {"backlog_bytes": 15, "lcids": 2}})

    def test_compare_detects_agreement_and_mismatch(self) -> None:
        decisions = [{"decision_seq": 0,
                      "mcs": {"value": 12, "ndi": 0,
                              "grant_identity": "abcd1234:1:10:00:00.000001:5.2:h3"},
                      "backlog": {"value": 15, "source":
                                  "x:y:tick=1.1:tod=10:00:00.000001:lcids=2"}}]
        dci = [{"time": "10:00:00.000001", "direction": "1", "rnti": "7",
                "dci_frame": "5", "dci_slot": "2", "harq_pid": "3", "mcs": "12",
                "mcs_table": "0", "round": "0", "ndi": "0"}]
        rlc = _raw_rlc([("10:00:00.000001", 7, 0, 1, 1, 1, 10),
                        ("10:00:00.000001", 7, 0, 1, 1, 4, 5),
                        ("10:00:00.000500", 7, 0, 1, 2, 1, 0)])
        ok = Q.compare_with_raw(decisions, dci_rows=dci, rlc_rows=rlc, rnti=7, ue_id=0)
        self.assertEqual((ok["matched_dci"], ok["matched_rlc"], ok["mismatches"]),
                         (1, 1, []))
        self.assertEqual(ok["raw_eligible_round0_ul_grants_by_ndi"], {"0": 1, "1": 0})
        bad = copy.deepcopy(decisions)
        bad[0]["mcs"]["value"] = 13
        bad[0]["backlog"]["value"] = 16
        result = Q.compare_with_raw(bad, dci_rows=dci, rlc_rows=rlc, rnti=7, ue_id=0)
        self.assertEqual(len(result["mismatches"]), 2)


def _decision(seq, *, admitted=True):
    return {"decision_seq": seq, "admitted": admitted, "readers_alive": True,
            "violations": 0, "fallback_raised": not admitted,
            "reasons": [] if admitted else ["X:MISSING"], "invented_zero": False,
            "snapshot_latency_ns": 2_000,
            "cache": {"dci": 8, "dci_capacity": 8, "rlc": 8, "rlc_capacity": 8},
            "mcs": {"x": 1} if admitted else None,
            "backlog": {"x": 1} if admitted else None}


def _run():
    return {"malformed_headers": 0, "pins_before_ok": True, "pins_after_ok": True,
            "bridge_residual_abs_ns": {"n": 100, "p95": 1_000, "max": 3_000},
            "record_alive_after_window": True, "unexpected_reader_eof": 0,
            "rf_restored": True, "channel_state_matches_initial": True,
            "core_stopped": True, "final_cold": True}


class GateTest(unittest.TestCase):
    def _comparison(self, decisions):
        n = sum(1 for d in decisions if d["mcs"])
        return {"matched_dci": n, "matched_rlc": n, "mismatches": []}

    def test_all_gates_pass_on_clean_evidence(self) -> None:
        decisions = [_decision(k) for k in range(300)]
        result = Q.evaluate_gates(decisions=decisions,
                                  comparison=self._comparison(decisions), run=_run())
        self.assertTrue(result["passed"], result["gates"])
        self.assertEqual(set(result["gates"]), set(Q.GATES))

    def test_freshness_threshold_is_strict(self) -> None:
        decisions = [_decision(k, admitted=not (10 <= k < 25)) for k in range(300)]
        result = Q.evaluate_gates(decisions=decisions,
                                  comparison=self._comparison(decisions), run=_run())
        self.assertFalse(result["gates"]["G2_FRESH_COVERAGE"])   # 275/290 < 95%
        decisions = [_decision(k, admitted=not (10 <= k < 24)) for k in range(300)]
        result = Q.evaluate_gates(decisions=decisions,
                                  comparison=self._comparison(decisions), run=_run())
        self.assertTrue(result["gates"]["G2_FRESH_COVERAGE"])    # 276/290

    def test_each_structural_failure_fails_its_gate(self) -> None:
        base = [_decision(k) for k in range(300)]
        cases = []
        d = copy.deepcopy(base); d[50]["violations"] = 1
        cases.append((d, _run(), "G3_ZERO_CAUSALITY_VIOLATIONS"))
        d = copy.deepcopy(base); d[40]["readers_alive"] = False
        cases.append((d, _run(), "G1_READERS_ALIVE_SCHEMAS_AND_PINS"))
        d = copy.deepcopy(base); d[60].update(_decision(60, admitted=False))
        d[60]["fallback_raised"] = False
        cases.append((d, _run(), "G5_EXPLICIT_FALLBACK"))
        d = copy.deepcopy(base)
        for row in d[:10]:
            row["snapshot_latency_ns"] = 2_000_000
        cases.append((d, _run(), "G6_SNAPSHOT_LATENCY"))
        run = _run(); run["bridge_residual_abs_ns"]["p95"] = 6_000
        cases.append((base, run, "G7_CLOCK_BRIDGE_RESIDUAL"))
        d = copy.deepcopy(base); d[3]["cache"] = dict(d[3]["cache"], dci=9)
        cases.append((d, _run(), "G8_BOUNDED_CACHE"))
        run = _run(); run["unexpected_reader_eof"] = 1
        cases.append((base, run, "G9_COEXISTENCE"))
        run = _run(); run["channel_state_matches_initial"] = False
        cases.append((base, run, "G10_RESTORE_AND_COLD"))
        for decisions, run, gate in cases:
            result = Q.evaluate_gates(decisions=decisions,
                                      comparison=self._comparison(decisions), run=run)
            self.assertFalse(result["gates"][gate], gate)
            self.assertFalse(result["passed"], gate)
        mismatch = {"matched_dci": 299, "matched_rlc": 300, "mismatches": [{}]}
        self.assertFalse(Q.evaluate_gates(decisions=base, comparison=mismatch,
                                          run=_run())["gates"]["G4_RAW_REPLAY_AGREEMENT"])

    def test_emitter_pins_and_gate_digest_are_stable(self) -> None:
        self.assertEqual(len(Q.verify_emitter_pins()), 5)
        self.assertEqual(Q.gates_sha256(), Q.canonical_sha256(Q.GATES))


if __name__ == "__main__":
    unittest.main()
