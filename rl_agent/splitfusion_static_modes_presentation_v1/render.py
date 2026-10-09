"""Generate slide PNGs from sealed campaign metrics and registered completion logs.

No inference, scoring, GT reconstruction, source changes, or live services are used.
All quality values come directly from campaign_report_05/CAMPAIGN_REPORT.json.
Completion curves use the campaign's existing, stdlib-only completion accountant.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/splitfusion_static_modes_slide_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

from rl_agent.splitfusion_static_modes_v13_offline_v1 import completion as CP

MAIN = Path(__file__).resolve().parents[2]
REPORT = MAIN / "experiments/splitfusion_static_modes_v13/campaign_report_05"
CAPTURES = MAIN.parent / ("abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/"
                         "splitfusion_single_gpu_profiler_v1/20261009_static_modes_v13")
DEFAULT_OUTPUT = MAIN / "experiments/splitfusion_static_modes_v13/presentation_01"
ORDER = ["agent"] + [f"mode{i:02d}" for i in range(12)]
FAMILIES = ["noAE", "AE128", "AE64", "AE32"]
BITS = [8, 6, 4]
INK, MUTED, GRID = "#172B3B", "#4C5C6A", "#DFE6EC"
GREEN, BLUE, PURPLE, GOLD = "#087F5B", "#2A78D6", "#7651A8", "#B88209"
STATIC, LATE, NONE = "#376B9B", "#E1A846", "#E0E5EB"
QUANT_COLORS = [BLUE, PURPLE, GOLD]
# Gaps separate agent and the four feature families, without random launch ordering.
Y = np.array([0.] + [1.35 + i + .30 * (i // 3) for i in range(12)])
SEPARATORS = [.68, (Y[3] + Y[4]) / 2, (Y[6] + Y[7]) / 2, (Y[9] + Y[10]) / 2]
COMMON = "FAVORABLE_STABLE  |  1,000 transmitted frames per arm  |  Separate live sessions"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def label(arm):
    if arm == "agent":
        return "Agent · adaptive mode + q"
    m = int(arm[4:])
    return f"{m:02d} · {FAMILIES[m // 3]} / {BITS[m % 3]}-bit"


def percent(value, upper=False, sparse=False):
    if value is None:
        return "N/A"
    v = value * 100
    digits = 2 if 0 < v < .1 else 1
    text = f"{v:.{digits}f}%" if v else "0%"
    return ("≤" if upper and value > 0 else "") + text + ("*" if sparse else "")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15, "font.weight": "bold",
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
        "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.titlesize": 19,
        "axes.labelsize": 17, "xtick.color": MUTED, "ytick.color": INK,
        "xtick.labelsize": 13, "ytick.labelsize": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.axisbelow": True,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": 14,
    })


def canvas(title, subtitle, notes):
    fig = plt.figure(figsize=(16, 9), dpi=240)
    fig.text(.035, .956, title, fontsize=29, va="top")
    fig.text(.035, .900, subtitle, fontsize=15, va="top", color=MUTED)
    fig.add_artist(Line2D([.035, .965], [.112, .112], transform=fig.transFigure,
                          color=GRID, linewidth=1))
    for j, note in enumerate(notes):
        fig.text(.035, .087 - .026 * j, note, fontsize=10.8, color=MUTED,
                 fontweight="normal", va="top")
    return fig


def row_axis(ax, labels=True):
    ax.set_ylim(Y[-1] + .65, -.7)
    ax.set_yticks(Y, [label(a) for a in ORDER] if labels else [])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.axhspan(-.48, .48, color=GREEN, alpha=.06, zorder=0)
    for s in SEPARATORS:
        ax.axhline(s, color=GRID, linewidth=.8, zorder=0)
    if labels:
        ax.get_yticklabels()[0].set_color(GREEN)


def save(fig, path, validation):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        if b.x0 < -2 or b.y0 < -2 or b.x1 > width + 2 or b.y1 > height + 2:
            outside.append(t.get_text())
    if outside:
        raise ValueError(f"Text outside {path.name}: {outside}")
    fig.savefig(path, dpi=240, metadata={"Software": "SplitFusion static-mode slide renderer v1"})
    validation[path.name] = {"pixels": [width, height], "text_inside_canvas": True}
    plt.close(fig)


def outcomes(rows, out, validation):
    fig = canvas("Prediction completion across all 12 modes", COMMON, [
        "All fixed modes: q = 0 (no spatial dropping). Denominator: 493 agent decisions; 500 per fixed mode. Holds/fallbacks are separate.",
        "Completion = action-open → edge prediction ready, before feedback. No completion is not a packet-loss classification."])
    ax = fig.add_axes([.255, .185, .545, .595])
    row_axis(ax)
    for arm, y in zip(ORDER, Y):
        c = rows[arm]["completion"]
        n = c["decisions"]
        vals = [c["completed_by_170"], c["completed_later"], c["no_completed_prediction"]]
        left = 0
        for j, (v, color) in enumerate(zip(vals, [GREEN if arm == "agent" else STATIC, LATE, NONE])):
            ax.barh(y, v / n, left=left, height=.68, color=color, linewidth=0)
            if v / n >= .085:
                ax.text(left + v / n / 2, y, f"{100*v/n:.1f}%", va="center", ha="center",
                        fontsize=12, color="white" if j == 0 else INK)
            left += v / n
        ax.text(1.035, y, f"{vals[0]:>3} / {vals[1]:>3} / {vals[2]:>3}", va="center", fontsize=13,
                fontfamily="DejaVu Sans Mono", color=GREEN if arm == "agent" else INK)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Share of all policy decisions", labelpad=8)
    fig.text(.815, .800, "Counts: on time / late / none", fontsize=11.5, color=MUTED)
    fig.legend(handles=[Patch(facecolor=STATIC, label="Prediction ready ≤170 ms"),
                        Patch(facecolor=LATE, label="Prediction ready later"),
                        Patch(facecolor=NONE, label="No completed prediction")],
               loc="center", bbox_to_anchor=(.57, .839), ncol=3, fontsize=13.5)
    save(fig, out / "01_prediction_outcomes.png", validation)


def curves(rows, curves_by_arm, out, validation):
    fig = canvas("When do predictions become available?", COMMON, [
        "Cumulative share over ALL decisions, not only completed predictions: agent n = 493; each fixed mode n = 500. Fixed modes use q = 0.",
        "Dotted line: original 170-ms feedback deadline, shown at this earlier prediction boundary. A flat zero means no completed prediction."])
    grid = fig.add_gridspec(2, 2, left=.075, right=.965, bottom=.21, top=.785, hspace=.50, wspace=.13)
    xmax = max(550., np.ceil(max(max(v or [0.]) for v in curves_by_arm.values()) / 50) * 50)

    def draw(ax, arm, color, lw):
        values = curves_by_arm[arm]
        n = rows[arm]["completion"]["decisions"]
        ax.step([0.] + values + [xmax], [0.] + [i/n for i in range(1, len(values)+1)] + [len(values)/n],
                where="post", color=color, lw=lw, zorder=4 if arm == "agent" else 3)

    for family in range(4):
        ax = fig.add_subplot(grid[family//2, family % 2])
        ax.set_title(FAMILIES[family] + "  ·  " + f"modes {family*3}, {family*3+1}, {family*3+2}", loc="left", pad=10)
        for j in range(3):
            draw(ax, f"mode{family*3+j:02d}", QUANT_COLORS[j], 2.5)
        draw(ax, "agent", GREEN, 3)
        ax.axvline(170, color=MUTED, linestyle=":", linewidth=1.7)
        ax.set(xlim=(0, xmax), ylim=(-.025, 1.06), xticks=[0, 170, 300, 450], yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(color=GRID, linewidth=.8)
        if family % 2 == 0:
            ax.set_ylabel("Decisions completed", fontsize=14)
        else:
            ax.tick_params(labelleft=False)
    fig.text(.52, .140, "Time from action-open to edge prediction ready (ms)", ha="center", fontsize=17)
    fig.legend(handles=[Line2D([], [], color=GREEN, lw=3, label="Agent reference")]+
                        [Line2D([], [], color=c, lw=3, label=f"{b}-bit (fixed mode)") for b,c in zip(BITS, QUANT_COLORS)],
               loc="center", bbox_to_anchor=(.52, .842), ncol=4)
    save(fig, out / "02_prediction_completion_curves.png", validation)


def payload(rows, out, validation):
    fig = canvas("Payload size across all 12 modes", COMMON, [
        "SFD4 application-frame bytes, all 1,000 transmitted frames per arm, including holds/fallbacks. Decimal kB (1 kB = 1,000 bytes).",
        "Fixed modes use q = 0 (no dropping); the agent adapts mode + q. Same byte layer for every bar; linear scale starts at zero."])
    ax = fig.add_axes([.255, .185, .58, .600])
    row_axis(ax)
    for arm, y in zip(ORDER, Y):
        p = rows[arm]["payload"]
        color = GREEN if arm == "agent" else STATIC
        v50, v99 = p["p50"] / 1000, p["p99"] / 1000
        ax.barh(y-.18, v50, height=.31, color=color)
        ax.barh(y+.18, v99, height=.31, facecolor="white", edgecolor=color, linewidth=1.3, hatch="////")
        ax.text(3800, y, f"{v50:,.1f} / {v99:,.1f}", va="center", fontsize=12.7,
                color=GREEN if arm == "agent" else INK)
    ax.set(xlim=(0, 3700), xticks=[0, 1000, 2000, 3000], xlabel="Application-frame payload (kB)")
    fig.text(.847, .804, "P50 / P99 (kB)", fontsize=13, color=MUTED)
    fig.legend(handles=[Patch(facecolor=STATIC, label="P50"),
                        Patch(facecolor="white", edgecolor=STATIC, hatch="////", label="P99")],
               loc="center", bbox_to_anchor=(.72, .844), ncol=2)
    ratio = 1 - rows["agent"]["payload"]["p50"] / rows["mode11"]["payload"]["p50"]
    fig.text(.255, .845, f"Agent median: {ratio:.1%} smaller than mode 11 at q = 0", fontsize=15, color=GREEN)
    save(fig, out / "03_payload_p50_p99.png", validation)


def quality(rows, cls, out, validation):
    name, tolerance = ("Vehicle", "1.0 m") if cls == "vehicle" else ("Pedestrian", "1.2 m")
    fig = canvas(f"{name} quality available by 170 ms", "Fixed modes: no dropping (q = 0)  |  Agent: adaptive mode + q  |  Separate live sessions", [
        "Recall / IoU include late or absent predictions as no contribution; known eligible GT remains in their denominators. ≤ marks an upper bound from missing GT.",
        "Precision uses only timely predictions (3-m matching gate), not all decisions. * Mode 7: just 2 timely frames. N/A ≠ zero; segmentation union may be unmeasured.",
        "Agent: 493 decisions; fixed modes: 500 each. Different scenes / eligible-object counts; no paired superiority claim."])
    grid = fig.add_gridspec(1, 3, left=.255, right=.970, top=.777, bottom=.197, wspace=.24)
    titles = [f"Recall ≤{tolerance}", "Segmentation IoU", "Detection precision"]
    for k, title in enumerate(titles):
        ax = fig.add_subplot(grid[0, k])
        row_axis(ax, labels=k == 0)
        ax.set_title(title, loc="left", fontsize=16, pad=14)
        for arm, y in zip(ORDER, Y):
            q = rows[arm]["quality"][cls]
            if k == 0:
                val, bound = q["recall_deadline"]["value"], q["recall_deadline"]["bound"]
            elif k == 1:
                val, bound = q["seg_iou_deadline"], q["seg_bound"]
            else:
                val, bound = q["precision_timely"], "CONDITIONAL_ON_TIMELY_PREDICTIONS"
            upper = "UPPER_BOUND" in bound
            color = GREEN if arm == "agent" else STATIC
            if val is not None:
                ax.barh(y, val, height=.65, color=color, alpha=.9, edgecolor="white" if upper else color,
                        hatch="////" if upper and val > 0 else None, linewidth=0)
            ax.text((val or 0) + .025, y, percent(val, upper=upper, sparse=k == 2 and arm == "mode07"),
                    va="center", fontsize=11.5, color=color if arm == "agent" else INK)
        ax.set(xlim=(0, 1.27), xticks=[0, .5, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    fig.text(.255, .835, "Agent", color=GREEN, fontsize=15)
    fig.text(.337, .835, "Fixed configurations", color=STATIC, fontsize=15)
    fig.text(.568, .835, "Deadline = prediction completion, not feedback receipt", color=MUTED, fontsize=13)
    save(fig, out / ("04_vehicle_quality_by_deadline.png" if cls == "vehicle" else "05_pedestrian_quality_by_deadline.png"), validation)


def boundaries(rows, out, validation):
    fig = canvas("Prediction ready is not the same as feedback received", COMMON, [
        "Both panels use the same 170-ms boundary and all policy decisions: agent n = 493; fixed modes n = 500. Fixed modes: q = 0.",
        "Feedback additionally waits for GT readiness, scoring and return. These are separate measured boundaries, not interchangeable latency metrics."])
    grid = fig.add_gridspec(1, 2, left=.255, right=.97, top=.776, bottom=.186, wspace=.18)
    for k, title in enumerate(["Edge prediction ready ≤170 ms", "Feedback at UE ≤170 ms"]):
        ax = fig.add_subplot(grid[0, k])
        row_axis(ax, labels=k == 0)
        ax.set_title(title, fontsize=17, loc="left", pad=13)
        for arm, y in zip(ORDER, Y):
            r = rows[arm]
            n = r["completion"]["decisions"]
            count = r["completion"]["completed_by_170"] if k == 0 else r["feedback_by_170"]
            frac = count/n
            color = GREEN if arm == "agent" else STATIC
            ax.barh(y, frac, height=.65, color=color)
            ax.text(frac + .02, y, f"{frac:.1%}", va="center", fontsize=12.5, color=color if arm == "agent" else INK)
        ax.set(xlim=(0, 1.20), xticks=[0, .25, .5, .75, 1.0], xlabel="Share of all policy decisions")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save(fig, out / "06_prediction_vs_feedback.png", validation)


def load_verified():
    manifest_path = REPORT / "OUTPUT_SHA256.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {str(manifest_path): sha(manifest_path)}
    for rel, expected in manifest.items():
        path = REPORT / rel
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Sealed report hash mismatch: {path}")
        inputs[str(path)] = actual
    rows = json.loads((REPORT / "CAMPAIGN_REPORT.json").read_text())["arms"]
    if set(rows) != set(ORDER):
        raise ValueError("Expected exactly agent plus modes 00–11")
    curves_by_arm, decisions, checks = {}, [], {}
    for arm in ORDER:
        row = rows[arm]
        if row["verdict"] != "VALID_MEASUREMENT" or row["payload"]["n"] != 1000:
            raise ValueError(f"Unexpected validity/sample count: {arm}")
        cell = CP.find_cell(CAPTURES / arm)
        for relative in ["CELL_RESULT.json", "numeric_gt/service/FRAMES.jsonl", "profiler/edge/edge.profile.jsonl"]:
            p = cell / relative
            inputs[str(p)] = sha(p)
        result = CP.account(CAPTURES / arm)
        if result["accounting_problems"]:
            raise ValueError(result["accounting_problems"])
        for key in ["decisions", "completed", "completed_by_170", "completed_later", "no_completed_prediction", "no_completion_subcategories"]:
            if result[key] != row["completion"][key]:
                raise ValueError(f"{arm}: {key} differs from sealed report")
        values = result["latency_ms"]
        if any(x < 0 or not np.isfinite(x) for x in values):
            raise ValueError(f"Invalid completion times: {arm}")
        for p in [50, 95, 99]:
            expected = row["completion"]["latency_ms"][f"p{p}"]
            actual = float(np.percentile(values, p)) if values else None
            if expected != actual and not (actual is not None and expected is not None and abs(actual-expected) < 1e-8):
                raise ValueError(f"{arm}: latency P{p} differs")
        curves_by_arm[arm] = values
        for d in result["per_decision"]:
            decisions.append({"arm": arm, "frame_id": d["frame_id"], "state": d["state"], "latency_ms": d["latency_ms"]})
        checks[arm] = {"counts_match": True, "quantiles_match": True, "accounting_problems": 0,
                       "decision_denominator": result["decisions"], "completed": result["completed"]}
        print(f"Validated {arm}: {result['completed']}/{result['decisions']} completions", flush=True)
    for p in [Path(CP.__file__), Path(__file__), Path(__file__).with_name("__init__.py")]:
        inputs[str(p)] = sha(p)
    return rows, curves_by_arm, decisions, checks, inputs


def exports(rows, curves_by_arm, decisions, out):
    dump(out / "PLOT_DATA.json", {"schema": "splitfusion.static_modes.presentation.v1", "order": ORDER,
         "rows_from_sealed_report": rows, "completion_latency_ms": curves_by_arm})
    with (out / "PREDICTION_COMPLETION_DECISIONS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "frame_id", "state", "latency_ms"])
        w.writeheader(); w.writerows(decisions)
    fields = ["arm", "label", "decisions", "completed_by_170", "completed_later", "no_completed_prediction",
              "feedback_by_170", "payload_n", "payload_p50_bytes", "payload_p99_bytes", "class", "recall",
              "recall_numerator", "recall_known_denominator", "recall_bound", "seg_iou", "seg_bound", "precision_timely"]
    with (out / "SLIDE_METRICS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for arm in ORDER:
            r = rows[arm]
            for c in ["vehicle", "person"]:
                q = r["quality"][c]
                w.writerow({"arm": arm, "label": label(arm), **{k:r["completion"][k] for k in fields[2:6]},
                    "feedback_by_170": r["feedback_by_170"], "payload_n": r["payload"]["n"],
                    "payload_p50_bytes": r["payload"]["p50"], "payload_p99_bytes": r["payload"]["p99"],
                    "class": c, "recall": q["recall_deadline"]["value"], "recall_numerator": q["recall_deadline"]["n"],
                    "recall_known_denominator": q["recall_deadline"]["d"], "recall_bound": q["recall_deadline"]["bound"],
                    "seg_iou": q["seg_iou_deadline"], "seg_bound": q["seg_bound"], "precision_timely": q["precision_timely"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, curve_data, decisions, checks, inputs = load_verified()
    exports(rows, curve_data, decisions, out)
    style()
    validation = {}
    outcomes(rows, out, validation)
    curves(rows, curve_data, out, validation)
    payload(rows, out, validation)
    quality(rows, "vehicle", out, validation)
    quality(rows, "person", out, validation)
    boundaries(rows, out, validation)
    dump(out / "INPUT_SHA256.json", inputs)
    dump(out / "VALIDATION.json", {"sealed_report_files_verified": 6, "arms": checks, "plots": validation,
         "quality_source": str(REPORT / "CAMPAIGN_REPORT.json"), "quality_recomputed": False,
         "decisions_exported": len(decisions), "missing_gt_preserved_as_missing": True,
         "no_completion_not_assigned_a_latency": True, "raw_sources_rechecked_after_render": True})
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise ValueError(f"Input changed while rendering: {path}")
    dump(out / "PRESENTATION_GUIDE.json", {
        "slide_order": list(validation), "meaning": {
            "01": "Three mutually exclusive outcomes over all decisions; exact counts at right.",
            "02": "Cumulative completion fraction over all decisions, not a conditional-success latency CDF.",
            "03": "Application-frame payload P50/P99 at the same byte layer, all transmitted frames.",
            "04_05": "Deadline-aware quality only. Precision is conditional on timely predictions; recall/IoU penalize missing predictions.",
            "06": "Prediction completion and feedback receipt are separate system boundaries."},
        "caveats": ["Different live sessions/scenes; same FAVORABLE_STABLE trace. No paired or causal superiority claim.",
                    "All fixed modes use q=0, not the earlier q6000 comparator. No full-local session is mixed in.",
                    "Missing GT makes some quality values upper bounds; zero is never substituted for unknown GT.",
                    "Mode 7 precision rests on two timely predictions; no timely predictions means precision N/A.",
                    "Agent is more timely and smaller than the q=0 modes, but mode11 retains more vehicle quality."],
        "reproduce_from": str(MAIN),
        "command": "env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 ionice -c 3 python3 -B -m rl_agent.splitfusion_static_modes_presentation_v1.render",
        "runtime_versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__}})
    dump(out / "OUTPUT_SHA256.json", {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "OUTPUT_SHA256.json"})
    print(f"Created {len(validation)} slide PNGs in {out}", flush=True)


if __name__ == "__main__":
    main()
