"""Create a compact, reproducible V16 (M3.6 OPERATING_14DB) capture/accounting/radio summary from retained records only.

Copy of the v11 summary with three additions, none of which changes a v11 definition: the commanded schedule is read
from the capture's own verified registration (not a constant), the accounting reference is the V11 capture, and the
channel-fidelity checks F1-F5 registered in ``registration.channel.fidelity`` BEFORE launch are evaluated on the same
route window (report-only; never an integration criterion; a mismatch is reported, not adjusted).

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


def percentile(values, p):
    a = np.asarray(list(values), dtype=float)
    return None if not a.size else float(np.percentile(a, p))


def histogram(values):
    return {str(k): v for k, v in sorted(collections.Counter(int(x) for x in values).items())}


def fidelity(registration, pusch, mcs, commanded_rows):
    """The registered F1-F5 channel-fidelity checks on the route window (report-only)."""
    channel = registration["channel"]
    checks = channel["fidelity"]["checks"]
    snr = [r["pusch_snr_db"] for r in pusch]
    median, p05, p95 = percentile(snr, 50), percentile(snr, 5), percentile(snr, 95)
    selected = [r["selected_mcs"] for r in mcs]
    mcs_median = percentile(selected, 50)
    schedule = channel["schedule"]
    commanded_ok = bool(commanded_rows) and all(
        float(r["target_snr_db"]) == schedule["target_snr_db"]
        and float(r["mapped_rfsim_command_db"]) == schedule["expected_rfsim_command_db"] for r in commanded_rows)
    out = {
        "F1_pusch_median_db": dict(value=median, low=checks["F1_pusch_median_db"]["low"],
                                   high=checks["F1_pusch_median_db"]["high"],
                                   passed=median is not None and checks["F1_pusch_median_db"]["low"] <= median
                                   <= checks["F1_pusch_median_db"]["high"]),
        "F2_pusch_p05_db": dict(value=p05, low=checks["F2_pusch_p05_db"]["low"],
                                passed=p05 is not None and p05 >= checks["F2_pusch_p05_db"]["low"]),
        "F3_pusch_p95_db": dict(value=p95, high=checks["F3_pusch_p95_db"]["high"],
                                passed=p95 is not None and p95 <= checks["F3_pusch_p95_db"]["high"]),
        "F4_commanded": dict(rows=len(commanded_rows), passed=commanded_ok),
        "F5_selected_mcs_median": dict(value=mcs_median, expected=checks["F5_selected_mcs_median"]["expected"],
                                       passed=mcs_median is not None
                                       and mcs_median == checks["F5_selected_mcs_median"]["expected"]),
    }
    verdict = "CHANNEL_FIDELITY_MATCH" if all(v["passed"] for v in out.values()) else "CHANNEL_FIDELITY_MISMATCH"
    return dict(verdict=verdict, checks=out, samples=dict(pusch=len(snr), mcs=len(selected)),
                anchor=channel["anchor_evidence"], window=channel["fidelity"]["window"],
                on_mismatch=channel["fidelity"]["on_mismatch"])


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
    registration = load(capture / "registration.json")
    schedule = registration["channel"]["schedule"]
    cells = sorted((capture / "capture/cells").glob("*"))
    with (cells[0] / "radio_trace.csv").open(newline="") as handle:
        commanded_all = list(csv.DictReader(handle))
    commanded = [r for r in commanded_all if r.get("command_ack_wall_ns")
                 and start_ns <= int(r["command_ack_wall_ns"]) <= end_ns]
    return {
        "window": {"definition": "first through last decision capture wall time; startup/restore excluded",
                   "start_wall_ns": start_ns, "end_wall_ns": end_ns,
                   "duration_s": (end_ns - start_ns) / 1e9},
        "commanded_snr": {"schedule_id": schedule["schedule_id"], "trace_id": schedule["trace_id"],
                          "target_db": schedule["target_snr_db"],
                          "rfsim_noise_power_db": schedule["expected_rfsim_command_db"],
                          "rows_in_window": len(commanded), "rows_total": len(commanded_all),
                          "timing_status": dict(collections.Counter(r["command_timing_status"] for r in commanded_all)),
                          "semantics": "commanded target / actor-visible proxy; not measured"},
        "calibrated_per_transmission": {"pusch_snr_db": stats(r["pusch_snr_db"] for r in pusch),
                                         "pusch_snr_db_p05": percentile([r["pusch_snr_db"] for r in pusch], 5),
                                         "pusch_snr_db_histogram_x10": histogram(r["snrx10"] for r in pusch),
                                         "scheduler_avg_snr_db": stats(r["scheduler_avg_snr_db"] for r in mcs),
                                         "actual_selected_mcs": stats(r["selected_mcs"] for r in mcs),
                                         "actual_final_mcs": stats(r["final_mcs"] for r in mcs),
                                         "selected_mcs_histogram": histogram(r["selected_mcs"] for r in mcs),
                                         "final_mcs_histogram": histogram(r["final_mcs"] for r in mcs)},
        "channel_fidelity": fidelity(registration, pusch, mcs, commanded),
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
    parser.add_argument("--v11-timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    current_timing = args.capture / "gt_gc_trace/storage_accounting_timing.json"
    current = accounting(current_timing)
    reference = accounting(args.v11_timing)
    current_late = current["first_last_quarter"]["callers"]["compute_unbound"]["late_cpu_ms"]["p50"]
    reference_late = reference["first_last_quarter"]["callers"]["compute_unbound"]["late_cpu_ms"]["p50"]
    document = {
        "schema": "splitfusion.m3_route_b_v16.capture_summary.v1",
        "scope": "read-only postflight analysis of the sole V16 (M3.6 OPERATING_14DB) live capture",
        "accounting": current,
        "v11_accounting_reference": reference,
        "late_compute_unbound_comparison": {
            "v16_p50_ms": current_late, "v11_p50_ms": reference_late,
            "ratio": current_late / reference_late, "change_fraction": current_late / reference_late - 1,
            "interpretation": "descriptive cross-session comparison; same timing definition, different scene/session/history"},
        "binned": load(args.binned / "SUMMARY.json"),
        "radio_route_window": radio(args.capture, args.analysis),
        "inputs_sha256": {
            "v16_timing": sha(current_timing), "v11_timing": sha(args.v11_timing),
            "registration": sha(args.capture / "registration.json"),
            "decision_timeline": sha(args.analysis / "DECISION_TIMELINE.csv"),
            "metrics": sha(args.analysis / "METRICS.json"), "binned_summary": sha(args.binned / "SUMMARY.json")},
    }
    output = args.out / "ANALYSIS_SUMMARY.json"
    output.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps({output.name: sha(output)}, indent=1) + "\n")
    print(json.dumps({"out": str(output), "late_compute_unbound": document["late_compute_unbound_comparison"],
                      "radio_rows": document["radio_route_window"]["grants"]["rows"],
                      "channel_fidelity": document["radio_route_window"]["channel_fidelity"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
