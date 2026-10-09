"""Render only M1 extension plots 01/02, with one series per configuration.

Reads exported metrics only: no inference, runtime import, or metric recomputation.
Full local is the existing 99-frame agent-input offline reference, not a timed arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/splitfusion_m1_baselines_v1"
KEYS = ("full_local", "agent", "arm_a", "action0")
COLORS = dict(zip(KEYS, ("#20364E", "#087F5B", "#B88209", "#66758A")))
LABELS = dict(zip(KEYS, ("Full local (offline)", "Agent split", "AE32 / UINT8 split*", "noAE / UINT8 split")))
CLASSES = ("vehicle", "person")
TOL = {"vehicle": 1.0, "person": 1.2}
INK, MUTED, GRID = "#172B3B", "#435565", "#DCE3E8"
NAMES = ("01_system_localisation_deadline_cdf.png", "02_system_deadline_quality_bars.png")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def load(source):
    paths = [source / "metrics/DEADLINE_AWARE_QUALITY.json",
             source / "metrics/LOCALISATION_CURVES.csv",
             BASE / "final_03/LOCALIZATION_CDF.csv"]
    doc = json.loads(paths[0].read_text())
    assert doc["match_radius_m"] == 3.0
    live_rows, local_rows = read_csv(paths[1]), read_csv(paths[2])
    curves, values = {}, {}
    for key in KEYS:
        for cls in CLASSES:
            if key == "full_local":
                rows = [r for r in local_rows if r["session"] == "agent"
                        and r["branch"] == "full_local" and r["class"] == cls]
                field = "fraction_all_eligible_gt"
                ref = doc["full_local_offline_reference"]["values"]["agent"][cls]["full_local"]
                assert ref["samples"] == 99
                block = {"recall": ref["recall_within_tol"], "iou": ref["seg_iou_pooled"],
                         "precision": ref["precision"], "precision_tp": ref["matched_3m"],
                         "precision_fp": ref["false_positives"],
                         "numerator": ref["within_tol"], "denominator": ref["eligible"],
                         "bound": "OFFLINE_REFERENCE_UNTIMED", "samples": ref["samples"]}
            else:
                rows = [r for r in live_rows if r["session"] == key and r["cls"] == cls
                        and r["convention"] == "deadline_aware"]
                field = "fraction"
                loc = doc["sessions"][key]["localisation"][cls]["deadline_aware"]
                seg = doc["sessions"][key]["segmentation"][cls]["deadline_aware"]
                block = {k: loc[k] for k in ("recall", "numerator", "denominator", "bound")}
                block.update(iou=seg["iou"], segmentation_bound=seg["bound"],
                             precision=loc["precision_timely"], precision_tp=loc["tp_timely"],
                             precision_fp=loc["fp_timely"],
                             samples=doc["sessions"][key]["decisions"])
            assert rows, (key, cls)
            rows.sort(key=lambda r: float(r["threshold_m"]))
            x = np.array([float(r["threshold_m"]) for r in rows])
            y = np.array([float(r[field]) for r in rows])
            assert len(x) == 301 and np.all(np.diff(x) > 0)
            assert np.all(np.diff(y) >= -1e-12) and np.all((y >= 0) & (y <= 1))
            index = np.flatnonzero(np.isclose(x, TOL[cls]))
            assert len(index) == 1 and abs(y[index[0]] - block["recall"]) < 1e-12
            assert abs(block["numerator"] / block["denominator"] - block["recall"]) < 1e-12
            detections = block["precision_tp"] + block["precision_fp"]
            if detections:
                assert abs(block["precision_tp"] / detections - block["precision"]) < 1e-12
            else:
                assert block["precision"] is None
            curves[key, cls], values[key, cls] = (x, y), block
    assert values["action0", "vehicle"]["iou"] is None
    assert values["action0", "person"]["iou"] is None
    return curves, values, {str(p.relative_to(ROOT)): digest(p) for p in paths}


def canvas(title, bars=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 17, "font.weight": "bold",
        "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.titlesize": 22,
        "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#ACB6BF", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 1, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(16, 9), dpi=240)
    fig.text(.05, .95, title, fontsize=30, va="top")
    fig.text(.05, .888, "Split models: predictions ready by 170 ms  |  Full local: offline quality reference",
             fontsize=17, color=MUTED, va="top")
    handles = [Patch(facecolor=COLORS[k], label=LABELS[k]) if bars else
               Line2D([], [], color=COLORS[k], lw=4, label=LABELS[k]) for k in KEYS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.505, .824), ncol=4,
               fontsize=14, handlelength=2.3, columnspacing=1.8)
    fig.add_artist(Line2D([.05, .96], [.16, .16], transform=fig.transFigure, color=GRID, lw=1.2))
    notes = [
        "Late / absent split predictions count as no detections and empty masks; timing ends at edge prediction ready, not feedback.",
        "Separate sessions: agent 495 decisions; static arms 500 each. Full local: 99 retained agent inputs, with no timing claim.",
        "* AE32 values are upper bounds (missing GT on failed frames). noAE: 0 predictions; segmentation unavailable (no GT masks).",
    ]
    if bars:
        notes = [
            "Recall / IoU count late or absent predictions as empty outputs. Precision = TP / (TP + FP) on available detections (3 m gate).",
            "Separate sessions: agent 495 decisions; static arms 500 each. Full local: 99 retained agent inputs, with no timing claim.",
            "* AE32 recall / IoU are upper bounds (missing GT). noAE: 0 predictions; precision and segmentation IoU are unavailable.",
        ]
    for y, note in zip((.122, .084, .046), notes):
        fig.text(.05, y, note, fontsize=11.4, color=MUTED, va="center")
    return fig


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width + 1 or b.y1 > fig.bbox.height + 1:
            raise ValueError(f"Text outside canvas: {text.get_text()}")
    fig.savefig(path, dpi=240, metadata={"Software": "M1 presentation simple v1"})
    plt.close(fig)


def localisation(out, curves):
    fig = canvas("Localisation available by 170 ms")
    for ix, cls in enumerate(CLASSES):
        ax = fig.add_axes([.078 + .474 * ix, .282, .402, .454])
        for key in KEYS:
            x, y = curves[key, cls]
            ax.step(x, y, where="post", color=COLORS[key], lw=3.6, ls="-", zorder=3)
            if key != "action0":
                point = y[np.flatnonzero(np.isclose(x, TOL[cls]))[0]]
                ax.scatter([TOL[cls]], [point], color=COLORS[key], s=42, zorder=4)
        ax.axvline(TOL[cls], color="#87949E", lw=1.4, ls=":", zorder=1)
        ax.text(TOL[cls] + .05, .99, f"Target: {TOL[cls]:g} m", fontsize=14, va="top", color=MUTED)
        ax.text(2.93, .035, "noAE: 0 predictions", color=COLORS["action0"], fontsize=12, ha="right")
        ax.set(xlim=(0, 3), ylim=(-.018, 1.04), xlabel="Localisation error threshold (m)")
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if ix == 0:
            ax.set_ylabel("Recall over eligible GT objects")
        ax.set_title("Vehicles" if cls == "vehicle" else "Pedestrians", loc="left", pad=13)
        ax.grid(True)
    fig.text(.05, .194, "Missed objects stay in the denominator; curves are not normalised to the few predictions that completed.",
             fontsize=13.2, color=MUTED)
    save(fig, out / NAMES[0])


def quality(out, values):
    fig = canvas("Quality available by 170 ms", bars=True)
    for ix, (field, title) in enumerate((("recall", "Localisation recall"),
                                       ("precision", "Detection precision"),
                                       ("iou", "Segmentation IoU"))):
        left = .078 + .315 * ix
        fig.text(left, .767, title, fontsize=19, va="center")
        for row, cls in enumerate(CLASSES):
            ax = fig.add_axes([left, .523 - .246 * row, .252, .197])
            for x, key in enumerate(KEYS):
                value = values[key, cls][field]
                if value is None:
                    ax.text(x, .018, "N/A", ha="center", va="bottom", fontsize=13, color=COLORS[key])
                    continue
                ax.bar(x, value, .75, color=COLORS[key], zorder=3)
                label = ("≤" if key == "arm_a" and field != "precision" else "") + f"{value:.1%}"
                offset = .018
                ax.text(x, value + offset, label, ha="center", va="bottom", fontsize=12.5, color=COLORS[key])
            ax.set(xlim=(-.6, 3.6), ylim=(0, 1.16), xticks=[])
            ax.set_yticks([0, .5, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.tick_params(axis="y", labelsize=14)
            ax.grid(axis="y")
            if ix == 0:
                ax.set_ylabel("Vehicles" if cls == "vehicle" else "Pedestrians", fontsize=18)
                ax.text(.99, -.14, f"Localisation tolerance: ≤{TOL[cls]:g} m", transform=ax.transAxes,
                        ha="right", va="top", fontsize=12, color=MUTED)
    fig.text(.05, .194, "Precision describes the detections that arrived; it does not penalise delivery failures. Read it alongside recall.",
             fontsize=13.2, color=MUTED)
    save(fig, out / NAMES[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "presentation_ext_01")
    parser.add_argument("--plots", type=Path, help="Defaults to the existing presentation_ext_01/plots")
    args = parser.parse_args()
    out = args.plots or args.source / "plots"
    out.mkdir(parents=True, exist_ok=True)
    curves, values, hashes = load(args.source)
    localisation(out, curves)
    quality(out, values)
    note = {
        "schema": "m1_simplified_system_plots_v1",
        "scope": "Only plots 01/02; split curves and bars use deadline_aware, never all_completion",
        "boundary": "action open to successful edge prediction ready <=170 ms; not feedback receipt",
        "full_local": "99 retained agent-input offline frames, untimed; not a whole-session paired comparison",
        "upper_bounds": "AE32/UINT8: missing failed-frame GT; noAE IoU is unavailable, not zero",
        "precision": "3 m detection match gate; TP/(TP+FP) only on <=170 ms predictions for split arms; full local untimed; zero detections => undefined",
        "input_sha256": hashes,
        "renderer": str(Path(__file__).relative_to(ROOT)),
        "renderer_sha256": digest(Path(__file__)),
        "reproduce": "python3 -B rl_agent/splitfusion_m1_presentation_simple_v1/render_system.py",
        "values": {key: {cls: values[key, cls] for cls in CLASSES} for key in KEYS},
        "output_sha256": {name: digest(out / name) for name in NAMES},
    }
    (out / "SIMPLIFIED_SYSTEM_PLOTS.json").write_text(json.dumps(note, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plots": str(out), "validated_curves": len(curves), "output_sha256": note["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
