"""Existing Run-5 live sessions and per-session loaders for the RLC backlog transition study (read-only).

Per session (all on the host CLOCK_MONOTONIC_RAW timeline unless stated):
* ``PHASE6_UE_EVIDENCE.json``: per transmitted frame (POLICY_DECISION / POLICY_HOLD / FALLBACK): action_open,
  state_commit, backlog_age_ns (= action_open - selected RLC sample source), first/last packet send.
* ``telemetry_live/NRUE_MAC_RLC_BUFFER_STATUS_live.csv``: every OAI UE MAC buffer-status row
  (col0 host RAW receipt, col1 host REALTIME receipt, then T fields; ``time`` = CLOCK_REALTIME local time of day).
* ``telemetry_live/NR_PDCP_TX_SDU_live.csv``: every UE PDCP TX SDU (dual stamped: T time + CLOCK_MONOTONIC).
* decision table (``DECISIONS_*.jsonl``): recorded pre_action_rlc_backlog_bytes, prior_ul_mcs, payload_bytes, outcome.
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

ABI = Path(__file__).resolve().parents[2]
PROFILER = ABI.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
SESSIONS = {   # name -> (decision table relative to abiodun/, cell relative to PROFILER, partition)
    "pilot_valid02": ("experiments/splitfusion_run5_pilot_live_v1/valid_02/plots/DECISIONS_valid1000.jsonl",
                      "20261006_run5_pilot_live_v1/valid_b1000_02/capture/cells/a71__favorable_stable", "FIT"),
    "pilot_valid03": ("experiments/splitfusion_run5_pilot_live_v1/valid_03/plots/DECISIONS_valid1000_v3.jsonl",
                      "20261006_run5_pilot_live_v1/valid_b1000_03/capture/cells/a71__favorable_stable", "FIT"),
    "full_valid01": ("experiments/splitfusion_run5_full_live_v1/valid_01/plots/DECISIONS_valid1000_full.jsonl",
                     "20261006_run5_full_live_v1/valid_b1000/capture/cells/a71__favorable_stable", "FIT"),
    "gc1_control": ("experiments/splitfusion_run5_full_live_gc_v1/pair_01_analysis/control/DECISIONS_control_b1000.jsonl",
                    "20261006_run5_full_live_gc_v1/control/capture/cells/a71__favorable_stable", "FIT"),
    "gc1_candidate": ("experiments/splitfusion_run5_full_live_gc_v1/pair_01_analysis/candidate/DECISIONS_candidate_b1000.jsonl",
                      "20261006_run5_full_live_gc_v1/candidate/capture/cells/a71__favorable_stable", "FIT"),
    "gcv2": ("experiments/splitfusion_run5_full_live_gc_v2/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
             "20261006_run5_full_live_gc_v2/validation/capture/cells/a71__favorable_stable", "FIT"),
    "retention": ("experiments/splitfusion_run5_full_live_retention_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
                  "20261006_run5_full_live_retention_v1/validation/capture/cells/a71__favorable_stable", "FIT"),
    "fade": ("experiments/splitfusion_run5_channel_response_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
             "20261006_run5_channel_response_v1/validation/capture/cells/a71__fade_recovery", "DIAGNOSTIC_PREVIOUSLY_INSPECTED"),
}


@dataclass(frozen=True)
class Tick:
    source_raw_ns: int        # MAC slot instant mapped to RAW (T realtime - receipt realtime offset)
    available_raw_ns: int     # receipt of the first row of the next tick (completion proof), as in the live provider
    bytes: int                # sum over LCIDs of bytes_in_buffer (latest row per LCID)
    drb_bytes: int            # LCID >= 4 only


def _tod_to_realtime_ns(tod: str, receipt_real_ns: int) -> int:
    """T ``time`` is CLOCK_REALTIME local time of day (us); anchor it to the receipt date (safe across midnight)."""
    receipt = dt.datetime.fromtimestamp(receipt_real_ns / 1e9)
    h, m, s = tod.split(":")
    sec, _, frac = s.partition(".")
    cand = receipt.replace(hour=int(h), minute=int(m), second=int(sec), microsecond=int(frac.ljust(6, "0")[:6]))
    best = min((cand + dt.timedelta(days=d) for d in (-1, 0, 1)), key=lambda c: abs(c.timestamp() * 1e9 - receipt_real_ns))
    return int(round(best.timestamp() * 1e6)) * 1000


def load_rlc_ticks(cell: Path) -> list[Tick]:
    path = cell / "run4_phase6/telemetry_live/NRUE_MAC_RLC_BUFFER_STATUS_live.csv"
    ticks: list[Tick] = []
    open_key, by_lcid, src = None, {}, None
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 13 or not row[3].isdigit():
                continue
            raw, real = int(row[0]), int(row[1])
            key = (row[3], row[4], row[5], row[6])          # rnti, ue_id, frame, slot
            if open_key is not None and key != open_key:
                ticks.append(Tick(src, raw, sum(by_lcid.values()), sum(v for k, v in by_lcid.items() if k >= 4)))
                by_lcid = {}
            if key != open_key:
                open_key = key
                src = None
            source = raw - (real - _tod_to_realtime_ns(row[2], real))
            src = source if src is None else max(src, source)
            by_lcid[int(row[7])] = int(row[9])
    return ticks


def load_pdcp(cell: Path) -> list[tuple[int, int]]:
    """(RAW instant, sdu_bytes) for every UE PDCP TX SDU. MONOTONIC -> RAW by the median receipt offset."""
    path = cell / "run4_phase6/telemetry_live/NR_PDCP_TX_SDU_live.csv"
    rows = []
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 8 or not row[3].isdigit():
                continue
            rows.append((int(row[0]), int(row[3]) * 1_000_000_000 + int(row[4]), int(row[7])))
    offs = sorted(r[0] - r[1] for r in rows)
    off = offs[len(offs) // 2]
    return [(mono + off, size) for _, mono, size in rows]


def load_frames(cell: Path) -> list[dict]:
    d = json.loads((cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_text())
    out = []
    for x in d["decisions"]:
        out.append(dict(frame_id=int(x["frame_id"]), kind=x["kind"], open=int(x["action_open"]["ns"]) if x.get("action_open") else None,
                        commit=int(x["state_commit"]["ns"]) if x.get("state_commit") else None,
                        age=x.get("backlog_age_ns"), first_send=int(x["stages"]["first_packet_send_raw_ns"]),
                        last_send=int(x["stages"]["last_packet_send_raw_ns"]),
                        ledger=int(x["stages"]["ledger_staged_raw_ns"]), mode=x.get("mode_id"), q=x.get("q_e4")))
    out.sort(key=lambda r: r["frame_id"])
    return out


def load_decisions(rel: str) -> dict[int, dict]:
    rows = [json.loads(l) for l in (ABI / rel).read_text().splitlines() if l.strip()]
    return {int(r["frame_id"]): r for r in rows}


def tick_at(ticks: list[Tick], sources: list[int], t_raw: int) -> Tick | None:
    """Latest tick whose source is <= t (the true buffer state at t, ignoring availability)."""
    i = bisect.bisect_right(sources, t_raw) - 1
    return ticks[i] if i >= 0 else None
