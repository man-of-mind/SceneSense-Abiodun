"""Create-only, CPU-only GT-accounting growth figure from a sealed binned report (v10).

Two panels, both read from THIS run's ``BINNED_REPORT.csv`` and the run's own ``storage_accounting_timing.json``;
no count, window or reference value is hard-coded from an earlier capture except the explicitly labelled V9
reference curve, which is read from the sealed V9 trace file passed in with ``--v9-timing`` (omit it and the panel
is drawn for this run alone).

  left   per-bin P50 of the accounting CPU per _account call, FREEZE CPU and BIND CPU over elapsed route time,
         with the pre-registered early/late windows shaded
  right  per-decision accounting CPU split by the ProducerOwner stage that called _account, first vs last quarter
         of decisions -- the quantity the corrected implementation acts on

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -B -m \\
      rl_agent.splitfusion_m3_route_b_v10_analysis.render_accounting_growth \\
        --binned <binned report dir> --timing <capture root>/gt_gc_trace/storage_accounting_timing.json \\
        [--v9-timing <v9 capture root>/gt_gc_trace/storage_accounting_timing.json] --out <new dir>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREEZE_STAGES = ("compute_unbound", "stage_products")
BIND_STAGES = ("bind_plan", "set_ready")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _p50(values):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(0.5 * (len(values) - 1)))))
    return values[index]


def per_stage(timing_path):
    """Per-stage accounting CPU (ms), first vs last quarter of decisions, from one run's own timer record."""
    timing = json.loads(Path(timing_path).read_bytes())
    cpu = [value / 1e6 for value in timing["cpu_ns"]]
    names, callers = timing["caller_names"], timing["caller"]
    quarter = len(cpu) // 4
    out = {}
    for index, name in enumerate(names):
        early = [c for c, k in zip(cpu[:quarter], callers[:quarter]) if k == index]
        late = [c for c, k in zip(cpu[-quarter:], callers[-quarter:]) if k == index]
        out[name] = dict(early_p50_ms=_p50(early), late_p50_ms=_p50(late), early_n=len(early), late_n=len(late))
    out["_meta"] = dict(calls=timing["calls"], recorded=timing["recorded"], dropped=timing["dropped"],
                        errors=timing["errors"], foreign_frames=timing["foreign_frames"],
                        wrapped=timing.get("wrapped"), decisions_per_quarter=quarter // len(names))
    return out


def run(binned, timing, out, v9_timing=None):
    binned, out = Path(binned), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with (binned / "BINNED_REPORT.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]
    summary = json.loads((binned / "SUMMARY.json").read_bytes())
    growth = summary["growth"]

    def series(column):
        points = []
        for row in rows:
            try:
                t, value = float(row["t_start_s"]), float(row[column])
            except (TypeError, ValueError):
                continue
            points.append((t, value))
        return [p[0] for p in points], [p[1] for p in points]

    stages = per_stage(timing)
    reference = per_stage(v9_timing) if v9_timing else None

    figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    for column, label, colour in (("accounting_cpu_ms_p50", "accounting CPU per _account call", "#c44e52"),
                                  ("freeze_cpu_ms_p50", "FREEZE CPU", "#4c72b0"),
                                  ("bind_cpu_ms_p50", "BIND CPU", "#55a868")):
        x, y = series(column)
        left.plot(x, y, marker="o", markersize=4, linewidth=1.6, color=colour, label=label)
    for window, shade in ((growth["early_window_s"], "#4c72b0"), (growth["late_window_s"], "#c44e52")):
        left.axvspan(window[0], window[1] + summary["bin_seconds"], color=shade, alpha=0.08)
    left.set_xlabel(f"elapsed route time (s), {summary['bin_seconds']:.0f}-s bins")
    left.set_ylabel("per-bin P50 CPU (ms)")
    left.set_title(f"GT prep CPU over the route\n{summary['decisions']} decisions, "
                   f"{stages['_meta']['calls']} _account calls, {stages['_meta']['dropped']} dropped")
    left.grid(alpha=0.25)
    left.legend(loc="upper left", fontsize=9)
    left.text(0.99, 0.02, "shaded: pre-registered early / late windows", transform=left.transAxes,
              ha="right", va="bottom", fontsize=8, color="#555555")

    order = [name for name in (*FREEZE_STAGES, *BIND_STAGES) if name in stages]
    width = 0.2 if reference else 0.38
    positions = range(len(order))
    bars = []
    if reference:
        bars.append(("V9 early", [reference[n]["early_p50_ms"] for n in order], -1.5, "#aec7e8"))
        bars.append(("V9 late", [reference[n]["late_p50_ms"] for n in order], -0.5, "#7ba3cc"))
        bars.append(("V10 early", [stages[n]["early_p50_ms"] for n in order], 0.5, "#f0a3a3"))
        bars.append(("V10 late", [stages[n]["late_p50_ms"] for n in order], 1.5, "#c44e52"))
    else:
        bars.append(("early", [stages[n]["early_p50_ms"] for n in order], -0.5, "#f0a3a3"))
        bars.append(("late", [stages[n]["late_p50_ms"] for n in order], 0.5, "#c44e52"))
    for label, values, offset, colour in bars:
        right.bar([p + offset * width for p in positions], values, width, label=label, color=colour)
    right.set_xticks(list(positions))
    right.set_xticklabels([f"{n}\n({'FREEZE' if n in FREEZE_STAGES else 'BIND'})" for n in order], fontsize=9)
    right.set_ylabel("P50 accounting CPU per call (ms)")
    right.set_title("Accounting CPU by the ProducerOwner stage that called _account\n"
                    "first vs last quarter of decisions" + (" (V9 reference alongside)" if reference else ""))
    right.grid(alpha=0.25, axis="y")
    right.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(out / "01_gt_accounting_growth.png", dpi=150)
    plt.close(figure)

    provenance = dict(
        schema="splitfusion.m3_route_b_v10.accounting_growth_figure.v1",
        scope="CPU-only presentation of an existing capture; no live run, no model inference, no re-scoring",
        binned_report=str(binned), binned_csv_sha256=sha(binned / "BINNED_REPORT.csv"),
        binned_summary_sha256=sha(binned / "SUMMARY.json"),
        timing=str(timing), timing_sha256=sha(timing),
        v9_timing=None if not v9_timing else str(v9_timing),
        v9_timing_sha256=None if not v9_timing else sha(v9_timing),
        bins=len(rows), bin_seconds=summary["bin_seconds"], decisions=summary["decisions"],
        early_window_s=growth["early_window_s"], late_window_s=growth["late_window_s"],
        stages=stages, v9_stages=reference,
        counts_read_from_this_run=True, hard_coded_counts=[],
        caveat="V9 is one capture of a different session, scene and route length; it is a reference curve, not a "
               "controlled A/B arm")
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(out=str(out), figure="01_gt_accounting_growth.png", bins=len(rows),
                          decisions=summary["decisions"], stages={k: v for k, v in stages.items() if k != "_meta"}),
                     indent=1))
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binned", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--v9-timing", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.binned, args.timing, args.out, args.v9_timing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
