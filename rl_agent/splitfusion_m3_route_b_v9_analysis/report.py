"""M3.1 v9 report assembly (offline, CPU only): the binned tables, the targeted GT-accounting cost, the calibrated
radio summary and the end-of-route drain evidence.

    python3 -B -m rl_agent.splitfusion_m3_route_b_v9_analysis.report --capture <prospective root>
        --analysis <registered analyzer out> [--route-layers <overlay dir>] --out <new dir>

The registered analyzer (``splitfusion_m3_route_b_v3_analysis.analyze``) is unchanged and runs separately; this adds
only what v9 introduced.  Nothing here re-scores, re-admits or re-derives a quality number: it reads the analyzer's
own per-decision rows, the GT prep worker trace, the accounting timer and the T-tracer telemetry.

The late-route growth reference is V8's measured +54 ms FREEZE+BIND rise over the route
(``experiments/splitfusion_m3_route_b_v8/LIVE_CAPTURE_REPORT.md``: FREEZE CPU P50 68 -> ~100 ms, BIND 14 -> ~37 ms).
Cross-session comparisons are descriptive: V9 changes both the accounting implementation and the GOOD_ALL target.
"""
import argparse
import csv
import json
from pathlib import Path

from . import binned as B
from . import telemetry as T

V8_REFERENCE = dict(source="experiments/splitfusion_m3_route_b_v8/LIVE_CAPTURE_REPORT.md",
                    freeze_cpu_p50_ms=[68.0, 100.0], bind_cpu_p50_ms=[14.0, 37.0], combined_rise_ms=54.0,
                    note="measured on the V8 capture (different session, different GOOD_ALL target and accounting "
                         "implementation); quoted as a reference point, not a controlled comparison")
FLAT = ("bin_index", "t_start_s", "t_end_s", "decisions_n", "timely_n", "timely_fraction", "quality_defined_n",
        "accounting_calls_n", "accounting_ms_per_decision")
STAT_COLUMNS = (("prediction_completion_ms", "p50"), ("prediction_completion_ms", "p95"),
                ("feedback_receipt_ms", "p50"), ("feedback_receipt_ms", "p95"), ("gt_published_ms", "p50"),
                ("freeze_cpu_ms", "p50"), ("freeze_cpu_ms", "p95"), ("bind_cpu_ms", "p50"), ("bind_cpu_ms", "p95"),
                ("accounting_cpu_ms", "p50"), ("accounting_cpu_ms", "p95"),
                ("pusch_snr_db", "p50"), ("pusch_snr_db", "p05"), ("scheduler_avg_snr_db", "p50"), ("mcs", "p50"),
                ("commanded_target_db", "p50"), ("prior_ul_mcs", "p50"), ("backlog_bytes", "p95"),
                ("payload_l2_bytes", "p50"), ("q_e4", "p50"), ("camera_si", "p50"), ("radar_p40", "p50"),
                ("map_age_at_install_ms", "p50"))


def growth(document):
    """Late-minus-early CPU, over the bins that actually carry GT work (first and last quarter of them)."""
    bins = [row for row in document["bins"] if row["freeze_cpu_ms"]["n"] and row["bind_cpu_ms"]["n"]]
    if len(bins) < 4:
        return dict(available=False, bins_with_gt_work=len(bins))
    quarter = max(1, len(bins) // 4)
    early, late = bins[:quarter], bins[-quarter:]

    def mean_of(group, key):
        values = [row[key]["p50"] for row in group if row[key].get("p50") is not None]
        return sum(values) / len(values) if values else None

    out = dict(available=True, bins_with_gt_work=len(bins), bins_per_side=quarter,
               early_window_s=[early[0]["t_start_s"], early[-1]["t_end_s"]],
               late_window_s=[late[0]["t_start_s"], late[-1]["t_end_s"]])
    for key in ("freeze_cpu_ms", "bind_cpu_ms", "accounting_cpu_ms"):
        first, last = mean_of(early, key), mean_of(late, key)
        out[key] = dict(early_p50=first, late_p50=last,
                        rise=None if None in (first, last) else last - first)
    rises = [out[key]["rise"] for key in ("freeze_cpu_ms", "bind_cpu_ms")]
    out["freeze_plus_bind_rise_ms"] = None if None in rises else sum(rises)
    out["v8_reference"] = dict(V8_REFERENCE)
    return out


def radio_summary(root):
    directory = B.radio_dir(root)
    streams = T.load(directory / "ttracer")
    record = json.loads((directory / "RADIO_TELEMETRY.json").read_bytes())
    mac = []
    path = directory / "gnb_stats_observed.jsonl"
    if path.is_file():
        import re
        pattern = re.compile(r"ulsch_rounds .*?BLER ([0-9.]+) MCS \(\d+\) (\d+).*?SNR ([\-0-9.]+) \(")
        for line in path.read_text(errors="replace").splitlines():
            match = pattern.search(json.loads(line)["line"]) if line.strip() else None
            if match:
                mac.append(float(match.group(3)))
    return dict(
        collector=dict(started=record.get("started"), error=record.get("error"), cpus=record.get("cpus"),
                       streams=record.get("streams"), relay_port_free_after_stop=record.get("relay_port_free_after_stop")),
        pusch_snr_db=T.summarise([r["pusch_snr_db"] for r in streams["pusch"]["rows"]]),
        scheduler_avg_snr_db=T.summarise([r["scheduler_avg_snr_db"] for r in streams["mcs"]["rows"]]),
        selected_mcs=T.summarise([r["selected_mcs"] for r in streams["mcs"]["rows"]]),
        final_mcs=T.summarise([r["final_mcs"] for r in streams["mcs"]["rows"]]),
        pusch_mcs=T.summarise([r["mcs"] for r in streams["pusch"]["rows"]]),
        pusch_value_counts={str(v): c for v, c in sorted(
            {x: [r["pusch_snr_db"] for r in streams["pusch"]["rows"]].count(x)
             for x in {r["pusch_snr_db"] for r in streams["pusch"]["rows"]}}.items())},
        mac_log_power_control_snr_db=dict(T.summarise(mac), label=T.LABELS["mac_log_snr_db"]),
        labels=dict(T.LABELS))


def drain_evidence(capture_root):
    cells = list((Path(capture_root) / "capture/cells").iterdir())
    if len(cells) != 1:
        return dict(available=False, cells=len(cells))
    path = cells[0] / "m3_route/END_OF_ROUTE_DRAIN.json"
    return json.loads(path.read_bytes()) if path.is_file() else dict(available=False, missing=str(path))


def write_csv(path, document):
    rows = []
    for item in document["bins"]:
        row = {key: item[key] for key in FLAT}
        for name, statistic in STAT_COLUMNS:
            row[f"{name}_{statistic}"] = (item.get(name) or {}).get(statistic)
        rows.append(row)
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [*FLAT])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(capture_root, analysis_dir, out, route_layers=None, bin_s=B.BIN_S):
    capture_root, out = Path(capture_root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    document = B.build(capture_root, analysis_dir, route_layers, bin_s)
    document["accounting_growth"] = growth(document)
    document["radio"] = radio_summary(capture_root)
    document["end_of_route_drain"] = drain_evidence(capture_root)
    (out / "BINNED_REPORT.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    written = write_csv(out / "BINNED_REPORT.csv", document)
    summary = dict(bins=written, bin_seconds=bin_s, decisions=document["decisions_total"],
                   accounting_timing=document["accounting_timing"], growth=document["accounting_growth"],
                   pusch_snr_db=document["radio"]["pusch_snr_db"], final_mcs=document["radio"]["final_mcs"],
                   mac_log=document["radio"]["mac_log_power_control_snr_db"],
                   drain={k: document["end_of_route_drain"].get(k) for k in
                          ("in_flight_completed", "open_at_end", "expired_on_real_clock", "closed_equals_sent",
                           "sent_at_end", "waited_s")})
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--route-layers", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bin-seconds", type=float, default=B.BIN_S)
    args = parser.parse_args(argv)
    run(args.capture, args.analysis, args.out, args.route_layers, args.bin_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
