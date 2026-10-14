"""Presentation-only revision: saved Shapley data, no model queries or old edits.

The main summary deliberately shows one reference background. The alternative
remains in the sealed original figure and the machine-readable slide data.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
GROUPS = ("scene", "history", "channel", "backlog")
WATERFALL_ORDER = ("scene", "channel", "backlog", "history")
COLORS = {"scene": "#D78326", "channel": "#2672AE", "backlog": "#5C7C77", "history": "#98468D"}
LABELS = {"scene": "Scene: SI + P40", "history": "Previous action / outcome",
          "channel": "Channel: SNR + MCS", "backlog": "RLC backlog"}
INK = "#182E3D"
GREY = "#87949E"
LIGHT = "#E4E9EE"
RUNS = (11, 12, 13, 14)
RUN_LABELS = ["Good throughout\n23.5 dB", "Bad throughout\n8.5 dB", "Good → bad\n23.5 → 8.5 dB", "Bad → good\n8.5 → 23.5 dB"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
        "axes.titlesize": 21, "axes.titleweight": "bold", "axes.labelsize": 16,
        "axes.labelweight": "bold", "xtick.labelsize": 14, "ytick.labelsize": 17,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#98A5AF", "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
        "figure.facecolor": "white", "pdf.fonttype": 42})


def load_data():
    hashes = json.loads((BASE / "figures_02/OUTPUT_SHA256.json").read_text())
    paths = {"source_sha256": BASE / "render.py",
             "selection_sha256": BASE / "selection_01/SELECTION.json",
             "results_sha256": BASE / "attribution_01/RESULTS.json",
             "attribution_sha256": BASE / "attribution_01/ATTRIBUTIONS.json.gz"}
    for key, path in paths.items():
        assert sha(path) == hashes[key], path
    originals = {str(path): sha(path) for path in paths.values()}
    for name, digest in hashes["png_sha256"].items():
        path = BASE / "figures_02" / name
        assert sha(path) == digest
        originals[str(path)] = digest
    selection = json.loads(paths["selection_sha256"].read_text())
    result = json.loads(paths["results_sha256"].read_text())
    with gzip.open(paths["attribution_sha256"], "rt") as f:
        rows = json.load(f)
    states = json.loads((BASE / "selection_01/RECORDED_STATES.json").read_text())
    originals[str(BASE / "selection_01/RECORDED_STATES.json")] = sha(BASE / "selection_01/RECORDED_STATES.json")
    # Recompute every summary magnitude used by the figure, including backup values.
    for record in result["summary"]:
        chosen = [r for r in rows if r["run"] == record["run"] and r["question"] == record["question"] and r["representative"]]
        assert len(chosen) == record["n"]
        for group in GROUPS:
            val = sum(abs(r[record["output"] + "_" + group]) for r in chosen) / len(chosen)
            assert abs(val - record["mean_absolute"][group]) < 1e-12
    return selection, result, rows, states, originals


def save_figure(fig, out, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    overflow = []
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text():
            continue
        b = item.get_window_extent(renderer)
        if b.x0 < bounds.x0 - 2 or b.y0 < bounds.y0 - 2 or b.x1 > bounds.x1 + 2 or b.y1 > bounds.y1 + 2:
            overflow.append(item.get_text())
    assert not overflow, (name, "Text outside figure", overflow)
    png = out / (name + ".png")
    pdf = out / (name + ".pdf")
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    with Image.open(png) as im:
        assert im.size == (3840, 2160)
    plt.close(fig)
    return {"png": png.name, "pdf": pdf.name, "pixels": [3840, 2160], "text_bounds": "PASS"}


def draw_summary(result, out):
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8))
    fig.subplots_adjust(left=.151, right=.975, top=.724, bottom=.225, wspace=.21)
    headings = [("margin11", "1. Which mode?", "Preference for mode 11 (AE32 / UINT4)",
                 "Mean absolute attribution\n(mode-preference logit units)"),
                ("q11", "2. How much to drop?", "Proposed q for mode 11",
                 "Mean absolute attribution\n(q units, on a 0–1 scale)")]
    export = []
    for col, (ax, (output, title, subtitle, xlabel)) in enumerate(zip(axes, headings)):
        ax.set_title(title, pad=46, fontsize=23)
        ax.text(.5, 1.045, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=16)
        records = [next(r for r in result["summary"] if r["run"] == v and r["question"] == "common_primary" and r["output"] == output) for v in RUNS]
        maxval = max(r["mean_absolute"][g] for r in records for g in GROUPS)
        for gi, group in enumerate(GROUPS):
            vals = [r["mean_absolute"][group] for r in records]
            y = np.arange(4) + (gi - 1.5) * .175
            ax.barh(y, vals, height=.149, color=COLORS[group])
            for yy, val in zip(y, vals):
                ax.text(val + maxval * .013, yy, f"{val:.3f}", va="center", ha="left", fontsize=14, color=INK)
            for v, val in zip(RUNS, vals):
                alt = next(r for r in result["summary"] if r["run"] == v and r["question"] == "common_alternative" and r["output"] == output)
                export.append({"run": v, "output": output, "group": group, "main": val,
                               "alternative_backup_only": alt["mean_absolute"][group]})
        ax.set_ylim(3.46, -.46)
        ax.set_xlim(0, maxval * 1.19)
        ax.set_yticks(range(4), RUN_LABELS if col == 0 else [""] * 4)
        ax.tick_params(axis="y", length=0, pad=16)
        ax.set_xlabel(xlabel, labelpad=12)
        ax.set_xticks([0, .2, .4, .6, .8] if output == "margin11" else [0, .02, .04, .06, .08, .10])
        ax.grid(axis="x", color=LIGHT)
        ax.set_axisbelow(True)
        for y in (.5, 1.5, 2.5):
            ax.axhline(y, color=LIGHT, linewidth=.85)
    fig.suptitle("What influences the agent’s two outputs?", y=.965, fontsize=29, fontweight="bold")
    fig.text(.5, .907, "Larger bars mean greater average influence relative to the chosen reference states.", ha="center", fontsize=18)
    fig.legend(handles=[Patch(color=COLORS[g], label=LABELS[g]) for g in GROUPS],
               loc="upper center", bbox_to_anchor=(.53, .87), ncol=4, fontsize=16, handlelength=1.3, columnspacing=1.7)
    fig.text(.5, .108, "100 decisions per condition · 8 shared reference states · proposed q is executed only when mode 11 is selected", ha="center", fontsize=15)
    fig.text(.5, .067, "Higher q = more dropping. Bars show attribution magnitudes, not percentages or performance gains. Panel scales differ.", ha="center", fontsize=15)
    fig.text(.5, .029, "Rankings depend on the references; 37–41% of mixed-input states are far from recordings. Sensitivity check retained in backup.", ha="center", fontsize=14)
    # This specific message is valid for GOOD_ALL, not a universal ranking.
    good = [next(r for r in result["summary"] if r["run"] == 11 and r["question"] == "common_primary" and r["output"] == o) for o in ("margin11", "q11")]
    assert all(sorted(GROUPS, key=lambda g: r["mean_absolute"][g], reverse=True) == list(GROUPS) for r in good)
    return save_figure(fig, out, "01_what_influences_agent"), export


def draw_waterfalls(selection, rows, states, out):
    cases = [("G1", "A. Stable good channel: q rises", "Previous action / outcome leads this change"),
             ("G2", "B. Stable good channel: q falls", "RLC backlog leads this change"),
             ("T13I", "C. Good → bad: q initially falls", "Channel and previous outcome both contribute"),
             ("T14I", "D. Bad → good: q rises at a reset", "Cleared history contributes most to the rise")]
    indexed = {r["id"]: r for r in states}
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
    fig.subplots_adjust(left=.063, right=.974, bottom=.152, top=.765, hspace=.72, wspace=.16)
    exported = []
    names = ["Previous\nq", "Scene", "Channel", "RLC\nbacklog", "Previous action\n/ outcome", "New\nq"]
    for ax, (key, title, takeaway) in zip(axes.flat, cases):
        case = next(c for c in selection["cases"] if c["case"] == key)
        row = next(r for r in rows if r["question"] == "adjacent" and r["run"] == case["run"] and r["index"] == case["index"])
        before = indexed[row["reference_ids"][0]]
        after = indexed[f"V{row['run']}:{row['index']}"]
        assert before["mode"] == after["mode"] == 11
        start, end = row["q11_reference"], row["q11_target"]
        assert abs(start - before["q_e4"] / 10000) < 1e-12
        assert abs(end - after["q_e4"] / 10000) < 1e-12
        parts = [row["q11_" + g] for g in WATERFALL_ORDER]
        assert abs(start + sum(parts) - end) < 1e-12
        vals = [start] + parts + [end]
        bases = [0]
        running = start
        for v in parts:
            bases.append(running)
            running += v
        bases.append(0)
        colors = ["#9EA9B2"] + [COLORS[g] for g in WATERFALL_ORDER] + [INK]
        for i, (val, base, color) in enumerate(zip(vals, bases, colors)):
            ax.bar(i, val, width=.64, bottom=base, color=color, zorder=3)
            if 0 < i < 5 and abs(val) < .001:
                label = "≈0" if val != 0 else "0"
            else:
                label = f"{val:+.3f}" if 0 < i < 5 else f"{val:.3f}"
            offset = 5 if val >= 0 else -5
            ax.annotate(label, (i, base + val), xytext=(0, offset), textcoords="offset points",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=17, fontweight="bold", zorder=4)
            if i < 5:
                ax.plot([i + .32, i + .68], [base + val, base + val], color="#9BA8B1", linewidth=1, linestyle=":", zorder=2)
        ax.set_ylim(0, 1)
        ax.set_xlim(-.6, 5.6)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_xticks(range(6), names, fontsize=13)
        ax.tick_params(axis="x", length=0, pad=9)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylabel("Dropping strength q", fontsize=15)
        ax.grid(axis="y", color=LIGHT)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=40, fontsize=19)
        far = row["far_nonendpoint_slots"]
        support = f"{far}/14 mixed states far from recordings"
        ax.text(0, 1.09, takeaway, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, color=INK)
        ax.text(1, .985, support, transform=ax.transAxes, ha="right", va="top", fontsize=11,
                color="#A34B31" if far else "#506570", bbox={"facecolor": "white", "alpha": .95, "edgecolor": "none", "pad": 2})
        exported.append({"case": key, "run": case["run"], "previous_frame": case["reference_frame"], "frame": case["frame"],
                         "q_before": start, "q_after": end, "phi": {g: row["q11_" + g] for g in WATERFALL_ORDER},
                         "far_mixed_states": far, "mixed_states": 14,
                         "actual_mode_before_and_after": 11, "takeaway": takeaway})
    fig.suptitle("Why did the dropping strength change?", y=.967, fontsize=29, fontweight="bold")
    fig.text(.5, .91, "Four recorded examples, all using mode 11:  previous q + input contributions = new q", ha="center", fontsize=18)
    fig.text(.5, .862, "Positive contribution → more dropping     |     Negative contribution → less dropping", ha="center", fontsize=17)
    fig.text(.5, .066, "Previous action / outcome includes mode, q, quality, feedback latency and validity flags—not only previous quality.", ha="center", fontsize=15)
    fig.text(.5, .026, "Far-state counts flag limited support; labels are rounded. These explanations do not establish payload, quality or latency benefits.", ha="center", fontsize=14)
    return save_figure(fig, out, "02_four_q_examples"), exported


def main(out):
    assert not out.exists(), "Create-only output directory; preserve prior figures"
    selection, result, rows, states, originals = load_data()
    out.mkdir()
    style()
    summary_figure, summary = draw_summary(result, out)
    waterfall_figure, cases = draw_waterfalls(selection, rows, states, out)
    save_json(out / "SLIDE_DATA.json", {"summary": summary, "cases": cases,
        "alternative_background": "A different fixed set of eight recorded reference states, used for sensitivity. Not an experiment, confidence interval or error bar. Not displayed on the main slide; original figures_02/01_four_condition_summary.png retains it.",
        "summary_definition": "For each sampled target, average signed grouped contributions across the eight references, then average the absolute values over the 100 targets in each run.",
        "scope": "Actor-output explanation only; no model query, experiment, training or changed scientific conclusion."})
    for path, digest in originals.items():
        assert sha(Path(path)) == digest, "Original changed: " + path
    save_json(out / "CHECKS.json", {"status": "PASS", "input_and_original_sha256": originals,
        "figures": [summary_figure, waterfall_figure], "summary_rows_checked": len(result["summary"]),
        "waterfalls_checked": len(cases), "both_endpoints_use_mode_11": True,
        "q_at_both_endpoints_matches_executed_action": True, "max_case_reconstruction_error": max(abs(c["q_before"] + sum(c["phi"].values()) - c["q_after"]) for c in cases),
        "sources_unchanged": True, "new_actor_queries": 0})
    save_json(out / "OUTPUT_SHA256.json", {"renderer_sha256": sha(Path(__file__)),
        "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({"out": str(out), "status": "PASS", "pngs": 2, "vector_pdfs": 2, "queries": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "presentation_04")
    main(parser.parse_args().out)
