"""Create a compact, reproducible V11 capture/accounting/radio summary from retained records only.

The radio route window is the first through last decision capture wall time, which excludes startup and restore.
``estimated_ul_buffer`` is reported only as the gNB's quantised BSR-derived belief; the independent pre-action
backlog field comes from the decision timeline.  Grant utilisation is not treated as peak achievable throughput.
"""
import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import telemetry as T

UL = re.compile(r"ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+), ulsch_DTX (\d+)")
POWER_SNR = re.compile(r"ulsch_rounds .*?SNR ([-\d.]+) \(")


def load(path):
    return json.loads(Path(path).read_bytes())


def stats(values):
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "min": float(a.min()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max()), "mean": float(a.mean()), "sum": float(a.sum())}


def accounting(path):
    record = load(path)
    names = record["caller_names"]
    callers = record["caller"]
    assert len(callers) == len(record["cpu_ns"]) == len(record["begin_raw_ns"]) == len(record["end_raw_ns"])
    assert len(callers) % len(names) == 0
    expected = list(range(len(names)))
    assert all(callers[i:i + len(names)] == expected for i in range(0, len(callers), len(names)))
    cpu = np.asarray(record["cpu_ns"], dtype=float) / 1e6
    wall = (np.asarray(record["end_raw_ns"], dtype=float) - np.asarray(record["begin_raw_ns"], dtype=float)) / 1e6
    per_caller = {}
    for index, name in enumerate(names):
        mask = np.asarray(callers) == index
        per_caller[name] = {"cpu_ms": stats(cpu[mask]), "wall_ms": stats(wall[mask])}
    width = len(names)
    per_decision_cpu = cpu.reshape((-1, width)).sum(axis=1)
    per_decision_wall = wall.reshape((-1, width)).sum(axis=1)
    quarter = len(per_decision_cpu) // 4
    early = slice(0, quarter)
    late = slice(len(per_decision_cpu) - quarter, len(per_decision_cpu))
    stage_windows = {}
    for index, name in enumerate(names):
        values = cpu[index::width]
        stage_windows[name] = {"early_cpu_ms": stats(values[early]), "late_cpu_ms": stats(values[late])}
    return {
        "record": {k: record[k] for k in ("calls", "recorded", "dropped", "errors", "foreign_frames",
                                                   "capacity", "caller_names")},
        "per_caller": per_caller,
        "per_decision": {"cpu_ms": stats(per_decision_cpu), "wall_ms": stats(per_decision_wall)},
        "first_last_quarter": {"decisions_per_window": quarter, "callers": stage_windows},
    }


def radio(capture, analysis):
    with (analysis / "DECISION_TIMELINE.csv").open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    start_ns = int(min(float(r["capture_wall_s"]) for r in decisions) * 1e9)
    end_ns = int(max(float(r["capture_wall_s"]) for r in decisions) * 1e9)
    streams = T.load(capture / "capture/m3_radio/ttracer")
    mcs = [r for r in streams["mcs"]["rows"] if start_ns <= r["wall_ns"] <= end_ns]
    pusch = [r for r in streams["pusch"]["rows"] if start_ns <= r["wall_ns"] <= end_ns]
    outstanding = [r for r in mcs if r["estimated_ul_buffer"] > 0]
    empty = [r for r in mcs if r["estimated_ul_buffer"] == 0]
    lead_slots = [((r["sched_frame"] - r["frame"]) % 1024) * 20 + r["sched_slot"] - r["slot"] for r in mcs]
    lead_counts = collections.Counter(lead_slots)
    mac, power_snr = [], []
    for line in (capture / "capture/m3_radio/gnb_stats_observed.jsonl").read_text().splitlines():
        row = json.loads(line)
        if start_ns <= int(row["observed_wall_ns"]) <= end_ns:
            match = UL.search(row["line"])
            if match:
                mac.append([int(match.group(i)) for i in range(1, 7)])
                power = POWER_SNR.search(row["line"])
                if power:
                    power_snr.append(float(power.group(1)))
    deltas = [b - a for a, b in zip(mac[0], mac[-1])] if len(mac) >= 2 else None
    backlog = [int(float(r["pre_action_rlc_backlog_bytes"])) for r in decisions]
    return {
        "window": {"definition": "first through last decision capture wall time; startup/restore excluded",
                   "start_wall_ns": start_ns, "end_wall_ns": end_ns,
                   "duration_s": (end_ns - start_ns) / 1e9},
        "commanded_snr": {"target_db": 23.5, "semantics": "commanded target / actor-visible proxy; not measured"},
        "calibrated_per_transmission": {"pusch_snr_db": stats(r["pusch_snr_db"] for r in pusch),
                                         "actual_final_mcs": stats(r["final_mcs"] for r in mcs)},
        "grants": {
            "rows": len(mcs),
            "with_gnb_reported_outstanding_data": {
                "definition": "estimated_ul_buffer > 0; quantised BSR-derived gNB belief, not independent UE backlog",
                "count": len(outstanding), "rb_size_final_prb": stats(r["rb_size_final"] for r in outstanding)},
            "with_gnb_reported_empty_buffer": {
                "definition": "estimated_ul_buffer == 0; does not imply zero network delay",
                "count": len(empty), "rb_size_final_prb": stats(r["rb_size_final"] for r in empty)},
            "all_rb_size_final_prb": stats(r["rb_size_final"] for r in mcs),
            "lead_time": {"slot_counts": {str(k): v for k, v in sorted(lead_counts.items())},
                          "slot_duration_ms": 0.5,
                          "lead_ms_counts": {str(k * 0.5): v for k, v in sorted(lead_counts.items())},
                          "basis": "100 MHz / 273 PRB, 30-kHz SCS; sched frame/slot minus decision frame/slot"},
            "warning": "grant utilisation is not a measurement of peak achievable throughput",
        },
        "backlog": {"pre_action_rlc_bytes": stats(backlog), "nonzero_decisions": sum(v > 0 for v in backlog),
                    "denominator_decisions": len(backlog)},
        "mac_log_counters": {
            "samples": len(mac), "rounds_delta": None if deltas is None else deltas[:4],
            "retransmissions_delta": None if deltas is None else sum(deltas[1:4]),
            "ulsch_errors_delta": None if deltas is None else deltas[4],
            "ulsch_dtx_delta": None if deltas is None else deltas[5],
            "power_control_snr_db": stats(power_snr),
            "power_control_snr_semantics": "avg_snr + tpc_in_flight; not calibrated channel SNR",
            "source": "gNB cumulative MAC log counters within the route wall-time window",
        },
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--binned", type=Path, required=True)
    parser.add_argument("--v10-timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    current_timing = args.capture / "gt_gc_trace/storage_accounting_timing.json"
    current = accounting(current_timing)
    reference = accounting(args.v10_timing)
    current_late = current["first_last_quarter"]["callers"]["compute_unbound"]["late_cpu_ms"]["p50"]
    reference_late = reference["first_last_quarter"]["callers"]["compute_unbound"]["late_cpu_ms"]["p50"]
    document = {
        "schema": "splitfusion.m3_route_b_v11.capture_summary.v1",
        "scope": "read-only postflight analysis of the sole V11 live capture",
        "accounting": current,
        "v10_accounting_reference": reference,
        "late_compute_unbound_comparison": {
            "v11_p50_ms": current_late, "v10_p50_ms": reference_late,
            "ratio": current_late / reference_late, "change_fraction": current_late / reference_late - 1,
            "interpretation": "descriptive cross-session comparison; same timing definition, different scene/session/history"},
        "binned": load(args.binned / "SUMMARY.json"),
        "radio_route_window": radio(args.capture, args.analysis),
        "inputs_sha256": {
            "v11_timing": sha(current_timing), "v10_timing": sha(args.v10_timing),
            "decision_timeline": sha(args.analysis / "DECISION_TIMELINE.csv"),
            "metrics": sha(args.analysis / "METRICS.json"), "binned_summary": sha(args.binned / "SUMMARY.json")},
    }
    output = args.out / "ANALYSIS_SUMMARY.json"
    output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps({output.name: sha(output)}, indent=1) + "\n")
    print(json.dumps({"out": str(output), "late_compute_unbound": document["late_compute_unbound_comparison"],
                      "radio_rows": document["radio_route_window"]["grants"]["rows"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
