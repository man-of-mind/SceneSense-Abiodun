"""Render the M1 presentation extensions (3840x2160 PNG) from metrics/ and sealed final_03 exports.

No live service, inference or capture. Main system plots: agent, static AE32/UINT8 (a54), static noAE/UINT8.
Fixed mode 11 / q=0.60 appears only in the dedicated agent-versus-fixed figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.text
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

from rl_agent.splitfusion_m1_presentation_ext_v1 import metrics as MX

STORY = MX.MAIN / "experiments/splitfusion_m1_baselines_v1/presentation_story_01"
COLORS = {"agent": "#087F5B", "arm_a": "#B88209", "action0": "#66758A", "arm_b": "#7651A8",
          "full_local": "#20364E", "remote": "#3B6FD4"}
INK, MUTED, GRID = "#172B3B", "#435565", "#DCE3E8"
SHORT = {"agent": "Agent", "arm_a": "Static AE32 / UINT8", "action0": "Static noAE / UINT8",
         "arm_b": "Fixed mode 11, q = 0.60"}
MAIN_KEYS = ("agent", "arm_a", "action0")
CLS_NAME = {"vehicle": "Vehicles", "person": "Pedestrians"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 17, "font.weight": "bold",
        "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.titlesize": 22,
        "axes.labelsize": 19, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#ACB6BF", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.axisbelow": True, "grid.color": GRID,
        "grid.linewidth": 1.0, "lines.linewidth": 3.2, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white"})


def canvas(title, subtitle, footer):
    style()
    fig = plt.figure(figsize=(16, 9), dpi=240)
    fig.text(.05, .95, title, fontsize=30, va="top")
    fig.text(.05, .888, subtitle, fontsize=17, va="top", color=MUTED)
    fig.text(.05, .025, footer, fontsize=12.5, va="bottom", color=MUTED, linespacing=1.5)
    fig.add_artist(Line2D([.05, .96], [.105, .105], transform=fig.transFigure, color=GRID, lw=1.2))
    return fig


def save(fig, path):
    """Save at 3840x2160; refuse if any visible text leaves the canvas or two figure-level texts overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.x1 > width + 1 or box.y0 < -1 or box.y1 > height + 1:
            raise ValueError(f"{Path(path).name}: text outside canvas: {text.get_text()[:60]!r} {box} canvas {width}x{height}")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def pct(value, bound=""):
    return "n/a" if value is None else f"{'≤' if bound.startswith('UPPER') else ''}{value:.1%}"


def load(metrics_dir):
    doc = json.loads((metrics_dir / "DEADLINE_AWARE_QUALITY.json").read_text())
    with (metrics_dir / "LOCALISATION_CURVES.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    final = json.loads((MX.FINAL_03 / "SUMMARY.json").read_text())
    return doc, curves, final


def curve(curves, key, cls, kind):
    rows = [r for r in curves if r["session"] == key and r["cls"] == cls and r["convention"] == kind]
    if not rows or rows[0]["fraction"] in ("", "None"):
        return None, None
    return np.array([float(r["threshold_m"]) for r in rows]), np.array([float(r["fraction"]) for r in rows])


def fig_localisation(out, doc, curves, keys, name, title, subtitle, footer):
    fig = canvas(title, subtitle, footer)
    for ix, cls in enumerate(MX.CLASSES):
        ax = fig.add_axes([.075 + ix * .475, .405, .40, .40])
        tol = MX.TOL[cls]
        for key in keys:
            loc = doc["sessions"][key]["localisation"][cls]
            for kind, ls in (("all_completion", "-"), ("deadline_aware", "--")):
                x, y = curve(curves, key, cls, kind)
                if x is None:
                    continue
                ax.step(x, y, where="post", color=COLORS[key], ls=ls, lw=4.0 if kind == "deadline_aware" else 2.4,
                        alpha=1 if kind == "deadline_aware" else .8)
            da = loc["deadline_aware"]
            ac = loc["all_completion"]
        ax.axvline(tol, color=INK, ls=":", lw=2)
        ax.text(tol + .04, .965, f"{tol:g} m", fontsize=15, va="top")
        ax.set(xlim=(0, 3), ylim=(0, 1.02), xlabel="Localisation error threshold e (m)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(True)
        if ix == 0:
            ax.set_ylabel("Eligible GT objects localised within e")
        ax.set_title(CLS_NAME[cls], loc="left", pad=12)
        lines = []
        for key in keys:
            loc = doc["sessions"][key]["localisation"][cls]
            da, ac = loc["deadline_aware"], loc["all_completion"]
            all_txt = "—" if ac["recall"] is None else f"{ac['numerator']}/{ac['denominator']}"
            lines.append(f"{SHORT[key].replace('Static ', '')}:  ≤170 ms {da['numerator']}/{da['denominator']}"
                         f"{' (upper bound)' if da['bound'].startswith('UPPER') else ''}  ·  all {all_txt}")
        fig.text(.075 + ix * .475, .29, f"Objects within {tol:g} m (≤170 ms · all completions)\n" + "\n".join(lines), fontsize=12.5,
                 va="top", linespacing=1.45)
    handles = [Line2D([], [], color=COLORS[k], lw=4, label=SHORT[k].replace("Static ", "")) for k in keys]
    handles += [Line2D([], [], color=INK, lw=2.4, alpha=.8, label="All completions"),
                Line2D([], [], color=INK, lw=4, ls="--", label="≤170 ms (deadline-aware)")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .135), ncol=len(handles), fontsize=14,
               handlelength=2.6, columnspacing=1.4)
    save(fig, out / name)


def bars_quality(out, doc, final, keys, name, title, subtitle, footer, full_local=False, boot=False):
    fig = canvas(title, subtitle, footer)
    metrics = (("Localisation recall", lambda s, c: s["localisation"][c], "recall"),
               ("Segmentation IoU", lambda s, c: s["segmentation"][c], "iou"))
    width = .8 / (2 * len(keys))
    for ix, (label, get, field) in enumerate(metrics):
        ax = fig.add_axes([.075 + ix * .475, .30, .40, .48])
        for j, key in enumerate(keys):
            s = doc["sessions"][key]
            for k, kind in enumerate(("all_completion", "deadline_aware")):
                xs = np.arange(2) - .4 + width * (2 * j + k) + width / 2
                vals, texts, errs = [], [], []
                for cls in MX.CLASSES:
                    block = get(s, cls)[kind]
                    v = block.get(field)
                    bound = block.get("bound", "")
                    vals.append(0 if v is None else v)
                    if v is None:
                        texts.append("n/a")
                    else:
                        texts.append(pct(v, bound))
                    ci = (s.get("bootstrap_95") or {}).get(
                        f"{cls}_{'recall' if field == 'recall' else 'iou'}_{'all' if k == 0 else 'deadline'}")
                    errs.append(ci)
                bars = ax.bar(xs, vals, width * .92, color=COLORS[key], alpha=.42 if k == 0 else 1,
                              edgecolor=COLORS[key], linewidth=1.5)
                if boot and all(errs):
                    lo = [v - e[0] for v, e in zip(vals, errs)]
                    hi = [e[1] - v for v, e in zip(vals, errs)]
                    ax.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.6, capsize=4)
                for x, v, t in zip(xs, vals, texts):
                    ax.text(x, v + (.06 if boot else .015), t, ha="center", va="bottom", fontsize=11.5, rotation=90)
        if full_local:
            for j, key in enumerate(keys):
                ref = (doc["full_local_offline_reference"]["values"].get(key) or {})
                for c_ix, cls in enumerate(MX.CLASSES):
                    if not ref:
                        continue
                    v = ref[cls]["full_local"]["recall_within_tol" if field == "recall" else "seg_iou_pooled"]
                    x0 = c_ix - .4 + width * 2 * j
                    ax.plot([x0, x0 + 2 * width], [v, v], color=COLORS["full_local"], lw=3, ls=(0, (2, 1.2)))
        ax.set(xticks=[0, 1], xticklabels=[f"Vehicle ({MX.TOL['vehicle']:g} m)" if field == "recall" else "Vehicle",
                                           f"Pedestrian ({MX.TOL['person']:g} m)" if field == "recall" else "Pedestrian"],
               ylim=(0, 1.08))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y")
        ax.set_title(label, loc="left", pad=12)
    handles = [Patch(facecolor=COLORS[k], label=SHORT[k].replace("Static ", "")) for k in keys]
    handles += [Patch(facecolor="#B5C0C9", label="Light: all completions"),
                Patch(facecolor=INK, label="Solid: ≤170 ms (deadline-aware)")]
    if full_local:
        handles.append(Line2D([], [], color=COLORS["full_local"], lw=3, ls=(0, (2, 1.2)),
                              label="Full local, offline, retained frames"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .175), ncol=3, fontsize=14)
    save(fig, out / name)


def table_figure(out, doc, keys, name, title, subtitle, footer):
    fig = canvas(title, subtitle, footer)
    ax = fig.add_axes([.05, .14, .91, .66]); ax.axis("off")
    cols = ["", *[SHORT[k] for k in keys]]
    rows = []

    def get(key):
        return doc["sessions"][key]
    rows.append(["Decisions (opportunities)", *[str(get(k)["decisions"]) for k in keys]])
    rows.append(["Prediction ≤170 ms / later / none",
                 *[f"{get(k)['prediction_states'].get('TIMELY', 0)} / {get(k)['prediction_states'].get('LATE', 0)} / "
                   f"{get(k)['prediction_states'].get('NONE', 0)}" for k in keys]])
    rows.append(["Decisions with GT objects / without GT",
                 *[f"{sum(get(k)['gt_coverage'][x] for x in get(k)['gt_coverage'] if not x.endswith('MISSING'))} / "
                   f"{sum(get(k)['gt_coverage'][x] for x in get(k)['gt_coverage'] if x.endswith('MISSING'))}" for k in keys]])
    for cls in MX.CLASSES:
        tol = MX.TOL[cls]
        for kind, lab in (("all_completion", "all completions"), ("deadline_aware", "≤170 ms")):
            cells = []
            for k in keys:
                b = get(k)["localisation"][cls][kind]
                cells.append("n/a (0 completed)" if b["recall"] is None else
                             f"{pct(b['recall'], b.get('bound', ''))}  ({b['numerator']}/{b['denominator']})")
            rows.append([f"{CLS_NAME[cls][:-1]} recall ≤{tol:g} m · {lab}", *cells])
    for cls in MX.CLASSES:
        for kind, lab in (("all_completion", "all completions"), ("deadline_aware", "≤170 ms")):
            cells = []
            for k in keys:
                b = get(k)["segmentation"][cls][kind]
                bound = b.get("bound", "")
                if b["iou"] is None:
                    cells.append("0 by construction*" if bound.startswith("ZERO") else "n/a (0 completed)")
                else:
                    ev = f" ({b['evaluable_opportunities']}/{b['opportunities']} frames)" if kind == "deadline_aware" else ""
                    cells.append(f"{pct(b['iou'], bound)}{ev}")
            rows.append([f"{CLS_NAME[cls][:-1]} seg IoU · {lab}", *cells])
    for cls in MX.CLASSES:
        cells = []
        for k in keys:
            b = get(k)["localisation"][cls]["all_completion"]
            cells.append("n/a" if b["precision"] is None else f"{b['precision']:.1%}  ({b['tp']}/{b['tp'] + b['fp']})")
        rows.append([f"{CLS_NAME[cls][:-1]} precision · completions (3 m match)", *cells])
    ref = doc["full_local_offline_reference"]["values"]
    for cls in MX.CLASSES:
        cells = []
        for k in keys:
            r = (ref.get(k) or {}).get(cls)
            cells.append("not computed" if not r else
                         f"{r['full_local']['recall_within_tol']:.1%} / {r['full_local']['seg_iou_pooled']:.1%}  "
                         f"(n={r['full_local']['samples']})")
        rows.append([f"Full-local {CLS_NAME[cls][:-1].lower()} recall / seg IoU (offline)", *cells])
    rows.append(["Feedback receipt ≤170 ms (separate)", *[f"{get(k)['feedback_by_170']}/{get(k)['decisions']}" for k in keys]])
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center",
                   colWidths=[.40, *[.60 / len(keys)] * len(keys)])
    tab.auto_set_font_size(False); tab.set_fontsize(12.5); tab.scale(1, 1.55)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("white")
        cell.get_text().set_weight("bold")
        if r == 0:
            cell.set_facecolor("#2F3B46"); cell.get_text().set_color("white")
        else:
            cell.set_facecolor("#F1F3F5" if r % 2 else "#E3E7EB")
            if c == 0:
                cell.get_text().set_ha("left"); cell._loc = "left"
    save(fig, out / name)


def timeline_rows():
    from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as LIVE
    _, rows, _ = LIVE.rows(MX.ROOTS["agent"])
    with (STORY / "ACTION_TIMELINE.csv").open() as stream:
        trace = {int(r["carla_frame_id"]): r for r in csv.DictReader(stream)}
    snr = json.loads((MX.ROOTS["agent"] / "capture/cells/a71__favorable_stable/run5_policy/SNR_EVIDENCE.json").read_text())
    kinds = {d["snr"]["kind"] for d in snr["decision_snr"]}
    assert kinds == {"SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB"}, kinds
    out = []
    for r in rows:
        t = trace[int(r["frame_id"])]
        assert (r["executed_action"]["mode_id"], r["executed_action"]["q_e4"]) == (int(t["mode_id"]), int(t["q_e4"]))
        assert int(t["payload_bytes"]) == r["payload_bytes"]
        out.append(dict(transmitted_frame_ordinal=int(t["transmitted_frame_ordinal"]), carla_frame_id=r["frame_id"],
                        policy_decision_ordinal=int(t["policy_decision_ordinal"]),
                        elapsed_action_open_s=float(t["elapsed_action_open_s"]), mode_id=int(t["mode_id"]),
                        q_drop_fraction=float(t["q_drop_fraction"]), payload_bytes=r["payload_bytes"],
                        snr_proxy_db=r["snr_db"], camera_si=r["camera_si"], radar_p40=r["radar_p40"]))
    assert len(out) == 495 and all(v is not None for row in out for v in row.values())
    return sorted(out, key=lambda r: r["transmitted_frame_ordinal"])


def fig_timeline(out, rows):
    style()
    fig = plt.figure(figsize=(16, 9), dpi=240)
    fig.text(.05, .965, "Agent actions and observed inputs on one frame axis", fontsize=29, va="top")
    fig.text(.05, .91, "M1 agent session · FAVORABLE_STABLE · 495 policy decisions across 1,000 transmitted frames",
             fontsize=16.5, va="top", color=MUTED)
    panels = (("mode_id", "Mode"), ("q_drop_fraction", "Drop q"), ("payload_kb", "Payload\n(kB)"),
              ("snr_proxy_db", "UL SNR\nproxy (dB)"), ("camera_si", "Camera SI"), ("radar_p40", "Radar P40"))
    grid = fig.add_gridspec(len(panels), 1, left=.13, right=.975, top=.865, bottom=.19, hspace=.24)
    x = np.array([r["transmitted_frame_ordinal"] for r in rows])
    color = COLORS["agent"]
    for i, (field, label) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        y = (np.array([r["payload_bytes"] for r in rows]) / 1000 if field == "payload_kb"
             else np.array([r[field] for r in rows], float))
        if field == "mode_id":
            ax.scatter(x, y, s=9, color=color, linewidths=0)
            ax.set(ylim=(-.8, 11.8), yticks=[0, 4, 8, 11])
        else:
            ax.plot(x, y, color=color, lw=1.6)
        if field == "q_drop_fraction":
            ax.set(ylim=(0, 1), yticks=[0, .5, 1])
        if field == "radar_p40":
            ax.set(ylim=(0, 1), yticks=[0, .5, 1])
        if field == "payload_kb":
            ax.set_ylim(0, max(200, y.max() * 1.08))
        if field == "snr_proxy_db":
            ax.set_ylim(0, 27)
        ax.set_ylabel(label, fontsize=15, rotation=0, ha="right", va="center", labelpad=14)
        ax.set_xlim(1, 1000)
        ax.grid(axis="y")
        ax.tick_params(labelsize=13)
        if i < len(panels) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Transmitted-frame number", fontsize=17, labelpad=4)
    fig.text(.05, .02, "Mode 11 = AE32/UINT4 · q = requested spatial drop fraction · payload = SFD4 application bytes "
             "(1 kB = 1,000 B) · holds/fallbacks not plotted.\n"
             "SNR = commanded uplink SNR proxy given to the actor, not measured PUSCH SNR (no PUSCH telemetry retained in M1).\n"
             "Camera SI = std of 3×3 Sobel magnitude on 768×448 luma · radar P40 = mean clip(1 − range/40 m, 0, 1).",
             fontsize=12.5, color=MUTED, linespacing=1.5)
    save(fig, out / "05_agent_action_timeline.png")


def fig_agent_vs_fixed(out, doc, final):
    keys = ("agent", "arm_b")
    fig = canvas("Agent vs fixed mode 11, q = 0.60",
                 "FAVORABLE_STABLE · separate live sessions, not paired scenes · every decision has scored GT in both",
                 "Prediction: action-open → successful edge prediction materialized. Payload: SFD4 application bytes, 1,000 frames each.\n"
                 "Whiskers: 95% frame-cluster bootstrap within each session (descriptive). Deadline-aware: late = no detections, empty mask.")
    ax = fig.add_axes([.13, .60, .25, .20])
    for i, key in enumerate(keys):
        s = doc["sessions"][key]
        n = s["decisions"]; t = s["prediction_states"].get("TIMELY", 0); l = s["prediction_states"].get("LATE", 0)
        none = s["prediction_states"].get("NONE", 0)
        y = 1 - i
        ax.barh(y, t / n, color=COLORS[key], height=.5)
        ax.barh(y, l / n, left=t / n, color=COLORS[key], alpha=.35, height=.5)
        ax.barh(y, none / n, left=(t + l) / n, color="#EDF0F3", hatch="///", edgecolor="#8895A1", height=.5)
        ax.text(.02, y, f"{t}/{n} ≤170 ms · {l} later · {none} none", va="center", color="white", fontsize=13)
    ax.set(yticks=[1, 0], yticklabels=["Agent", "Fixed m11\nq = 0.60"], xlim=(0, 1), ylim=(-.5, 1.5),
           xlabel="Share of policy decisions")
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="x")
    ax.set_title("Prediction completion", loc="left", fontsize=19)
    bx = fig.add_axes([.13, .27, .25, .19])
    for i, key in enumerate(keys):
        p = final["payload"][key]
        y = 1 - i
        for off, stat in ((.14, "p50"), (-.14, "p99")):
            bx.barh(y + off, p[stat] / 1000, height=.25, color=COLORS[key] if stat == "p50" else "white",
                    edgecolor=COLORS[key], hatch="///" if stat == "p99" else None, linewidth=1.5)
            bx.text(p[stat] / 1000 + 2, y + off, f"{stat.upper()} {p[stat] / 1000:.1f} kB", va="center", fontsize=13)
    bx.set(yticks=[1, 0], yticklabels=["Agent", "Fixed m11\nq = 0.60"], xlim=(0, 240), ylim=(-.5, 1.5),
           xlabel="Payload per frame (kB)")
    bx.grid(axis="x")
    bx.set_title("Application payload", loc="left", fontsize=19)
    width = .2
    for ix, (field, label, getter) in enumerate((("recall", "Localisation recall", "localisation"),
                                                  ("iou", "Segmentation IoU", "segmentation"))):
        cx = fig.add_axes([.47 + ix * .265, .27, .225, .53])
        for j, key in enumerate(keys):
            s = doc["sessions"][key]
            for k, kind in enumerate(("all_completion", "deadline_aware")):
                xs = np.arange(2) - .4 + width * (2 * j + k) + width / 2
                vals = [s[getter][c][kind][field] for c in MX.CLASSES]
                ci = [s["bootstrap_95"][f"{c}_{field}_{'all' if k == 0 else 'deadline'}"] for c in MX.CLASSES]
                cx.bar(xs, vals, width * .92, color=COLORS[key], alpha=.42 if k == 0 else 1)
                cx.errorbar(xs, vals, yerr=[[v - e[0] for v, e in zip(vals, ci)], [e[1] - v for v, e in zip(vals, ci)]],
                            fmt="none", ecolor=INK, elinewidth=1.5, capsize=3)
                if k == 1:
                    for x, v in zip(xs, vals):
                        cx.text(x, .02, f"{v:.1%}", rotation=90, ha="center", va="bottom", color="white", fontsize=12)
        cx.set(xticks=[0, 1], xticklabels=["Vehicle", "Pedestrian"], ylim=(0, 1))
        cx.yaxis.set_major_formatter(PercentFormatter(1)); cx.grid(axis="y")
        cx.set_title(label, loc="left", fontsize=19)
    fig.legend(handles=[Patch(facecolor=COLORS["agent"], label="Agent"),
                        Patch(facecolor=COLORS["arm_b"], label="Fixed m11, q = 0.60"),
                        Patch(facecolor="#B5C0C9", label="Light: completed later (left) / all completions (right)"),
                        Patch(facecolor=INK, label="Solid: ≤170 ms")],
               loc="center", bbox_to_anchor=(.5, .14), ncol=4, fontsize=13.5)
    save(fig, out / "06_agent_vs_fixed_deadline_quality.png")


def size_text(b):
    return "0 B" if b == 0 else f"{b / 1e6:.3f} MB" if b >= 1e6 else f"{b / 1e3:.1f} kB"


def fig_payload(out, final):
    fig = canvas("How much data does each configuration carry per frame?",
                 "P50 and P99 per frame · measured split inputs · calculated full-remote input · full-local output kept separate",
                 "Split: measured SFD4 application-frame bytes, all 1,000 transmitted frames per session. Full remote: calculated "
                 "1280×720×4 BGRA + 16 B × two-sweep\nradar returns (n = 1,000), metadata/framing excluded, not a minimum-streaming "
                 "claim. Full local: canonical JSON of emitted detections, not transmitted.")
    ax = fig.add_axes([.24, .25, .44, .53])
    keys = ("full_remote", "action0", "arm_a", "agent")
    labels = {"full_remote": "Full remote\nnative inputs (calc.)", "action0": "Static noAE / UINT8",
              "arm_a": "Static AE32 / UINT8", "agent": "Agent-selected split"}
    for i, key in enumerate(keys):
        y = 3 - i
        color = COLORS["remote" if key == "full_remote" else key]
        p = final["payload"][key]
        for off, stat in ((.17, "p50"), (-.17, "p99")):
            ax.barh(y + off, p[stat] / 1e6, height=.30, color=color if stat == "p50" else "white", edgecolor=color,
                    hatch="///" if stat == "p99" else None, linewidth=1.6)
            ax.text(p[stat] / 1e6 + .06, y + off, f"{stat.upper()} {size_text(p[stat])}", va="center", fontsize=14)
        ax.text(-.04, y - .42, f"n = {p['n']:,}", transform=ax.get_yaxis_transform(), ha="right", fontsize=12.5, color=MUTED)
    ax.set(xlim=(0, 5.9), ylim=(-.6, 3.5), yticks=range(4), yticklabels=[labels[k] for k in keys[::-1]],
           xlabel="Bytes per frame (MB, 10⁶ B)")
    ax.grid(axis="x"); ax.tick_params(axis="y", length=0)
    ax.set_title("Input carried to edge inference", loc="left", fontsize=20)
    bx = fig.add_axes([.79, .27, .17, .51])
    local = final["full_local_detection_output"]["agent"]
    bars = bx.bar([0, 1], [local["p50"] / 1e3, local["p99"] / 1e3], width=.55, color=[COLORS["full_local"], "white"],
                  edgecolor=COLORS["full_local"], linewidth=1.8)
    bars[1].set_hatch("///")
    bx.bar_label(bars, labels=[size_text(local["p50"]), size_text(local["p99"])], padding=6, fontsize=15)
    bx.set(xticks=[0, 1], xticklabels=["P50", "P99"], ylim=(0, 75), ylabel="Detection output (kB)")
    bx.grid(axis="y")
    bx.set_title(f"Full-local output", loc="left", fontsize=20)
    bx.text(.5, -.13, f"n = {local['n']} agent-session inputs\ninput uplink: 0 B", transform=bx.transAxes,
            ha="center", va="top", fontsize=13.5)
    save(fig, out / "07_system_payload.png")


def fig_compute(out, cost):
    parts = cost["parts"]
    fam = {f["family"]: f for f in cost["per_family_totals"]}
    full = parts["ue_front"]["flop_counter_macs"] + parts["edge_tail"]["flop_counter_macs"]
    check = cost["consistency"]["front_plus_tail_equals_full_local"]
    assert check["exact"] and check["full_local_macs"] == full, check
    fig = canvas("Where the perception compute runs",
                 "Deployed FCOS-R50-FPN P2–P7 · 7 × 448 × 768 input · split after ResNet layer1 (C2, 256 × 112 × 192)",
                 "Work per inference (batch 1), not GFLOP/s or latency. Conv/linear MACs; 1 MAC = 2 FLOPs; norms, activations, upsampling\n"
                 "and adds not counted (≈0.2%). Excluded: preprocessing, quantisation, zstd/packing, postprocessing (decode, NMS), RL actor.")
    ax = fig.add_axes([.17, .30, .44, .46])
    rows = (("Full local\n(one device)", full, COLORS["full_local"]),
            ("UE front\n(stem + layer1)", parts["ue_front"]["flop_counter_macs"], COLORS["agent"]),
            ("Edge tail\n(layer2 → heads)", parts["edge_tail"]["flop_counter_macs"], "#4C6A85"))
    for i, (label, macs, color) in enumerate(rows):
        g = macs / 1e9
        ax.barh(2 - i, g, height=.55, color=color)
        label_text = f"{g:.2f} GMACs · {2 * g:.1f} GFLOPs  ({macs / full:.1%})"
        if g > 100:
            ax.text(6, 2 - i, label_text, va="center", fontsize=15, color="white")
        else:
            ax.text(g + 4, 2 - i, label_text, va="center", fontsize=15)
    ax.set(yticks=[2, 1, 0], yticklabels=[r[0] for r in rows], xlim=(0, 290), ylim=(-.6, 2.6),
           xlabel="GMACs per inference")
    ax.grid(axis="x"); ax.tick_params(axis="y", length=0)
    ax.set_title("No AE (identity transport)", loc="left", fontsize=20)
    bx = fig.add_axes([.71, .30, .25, .46])
    names = ("AE32", "AE64", "AE128")
    y = np.arange(3)[::-1]
    enc = [fam[n]["ae_encoder_macs"] / 1e9 for n in names]
    dec = [fam[n]["ae_decoder_macs"] / 1e9 for n in names]
    bx.barh(y + .17, enc, height=.32, color=COLORS["agent"], label="Encoder (UE)")
    bx.barh(y - .17, dec, height=.32, color="#4C6A85", label="Decoder (edge)")
    for yy, e, d, n in zip(y, enc, dec, names):
        bx.text(e + .02, yy + .17, f"{e:.2f}", va="center", fontsize=14)
        bx.text(d + .02, yy - .17, f"{d:.2f}", va="center", fontsize=14)
        bx.text(1.24, yy, f"+{fam[n]['overhead_vs_full_local_q0']:.2%}", va="center", ha="right", fontsize=14,
                color=MUTED)
    bx.set(yticks=y, yticklabels=names, xlim=(0, 1.25), ylim=(-.6, 2.75), xlabel="GMACs per inference")
    bx.grid(axis="x"); bx.tick_params(axis="y", length=0)
    bx.set_title("AE overhead (dense grid)", loc="left", fontsize=20)
    bx.legend(loc="center", bbox_to_anchor=(.45, -.30), ncol=2, fontsize=14)
    bx.text(1.24, 2.5, "vs full local", ha="right", va="center", fontsize=13, color=MUTED)
    fig.text(.17, .19, f"q > 0 adds the UE ranker: {cost['q_dependence']['ranker_macs_when_q_gt0'] / 1e9:.3f} GMACs. "
             "Same weights in every mode; the edge tail dominates every configuration.", fontsize=14)
    save(fig, out / "04_compute_cost.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    doc, curves, final = load(args.out / "metrics")
    plots = args.out / "plots"
    plots.mkdir(exist_ok=False)
    fig_localisation(
        plots, doc, curves, MAIN_KEYS, "01_system_localisation_deadline_cdf.png",
        "Localisation available by 170 ms vs all completed predictions",
        "Eligible GT objects localised within e · misses stay in the denominator · one live session per colour",
        "Deadline-aware: only predictions completed ≤170 ms after action-open count; late or absent predictions count as misses. "
        "Sessions are not paired.\nnoAE completed no prediction (deadline-aware 0, no all-completion curve). AE32 ≤170-ms values "
        "are upper bounds: 4 failed decisions have no GT.")
    bars_quality(
        plots, doc, final, MAIN_KEYS, "02_system_deadline_quality_bars.png",
        "Quality actually available by 170 ms",
        "Light: all completed predictions · solid: deadline-aware (late or absent = no detections, empty mask)",
        "Recall denominators = eligible objects on every decision with GT: agent 1,144 veh / 590 ped; AE32 ≥817 / ≥670; noAE ≥368 / ≥254.\n"
        "≤ upper bound: 30/500 AE32 failures lack GT segmentation, 4 lack all GT. noAE: 0 completions; seg GT not retained (n/a).")
    table_figure(
        plots, doc, MAIN_KEYS, "03_system_metric_table.png", "Delivery-aware quality: denominators and boundaries",
        "Prediction boundary: action-open → successful edge prediction materialized · feedback receipt kept separate",
        "Recall cells: object counts (within tolerance / eligible GT). ≤ = upper bound (failed decisions lacking GT). Precision is "
        "conditional on detections.\n* noAE: intersection 0 on all 500 decisions; GT segmentation not retained. Full local: offline, "
        "retained frames, no runtime. Separate sessions.")
    rows = timeline_rows()
    with (args.out / "metrics" / "AGENT_TIMELINE.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig_timeline(plots, rows)
    fig_agent_vs_fixed(plots, doc, final)
    fig_localisation(
        plots, doc, curves, ("agent", "arm_b"), "06b_agent_vs_fixed_localisation_cdf.png",
        "Agent vs fixed mode 11, q = 0.60: localisation CDFs",
        "Both conventions · separate live sessions · all decisions scored in both",
        "Dashed: deadline-aware (completed ≤170 ms). Solid: all completed predictions. Object counts below each panel.")
    fig_payload(plots, final)
    cost_path = args.out / "compute_cost" / "COMPUTE_COST.json"
    fig_compute(plots, json.loads(cost_path.read_text()))
    print(json.dumps(sorted(p.name for p in plots.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
