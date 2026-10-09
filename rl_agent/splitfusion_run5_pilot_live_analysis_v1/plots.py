"""Figures for a pilot live run (reference palette, light surface). Every figure distinguishes all scored decisions
from timely-only decisions where quality is shown; outcome classes are labelled, never colour-only (markers differ).

01 action heatmap (mode x q bin, decision counts)      05 localization (within-tolerance recall, error CDF)
02 latency / deadline timeline (receipt per decision)   06 segmentation IoU (pooled bars, per-frame distribution)
03 SNR vs latency and quality                           07 freshness (capture-to-feedback age CDF)
04 SI / P40 vs latency and quality                      08 CDF and exceedance of stage boundaries and receipt
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from . import report as RP

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
OUTCOME_STYLE = {"TIMELY_DEFINED": (BLUE, "o", "timely, quality defined"), "LATE": (ORANGE, "^", "late (> 170 ms)"),
                 "TIMELY_UNDEFINED": (AQUA, "s", "timely, quality undefined"), "MISSING": (INK2, "x", "missing reply")}
SUBSET_STYLE = {"all": (BLUE, "all scored decisions"), "timely": (ORANGE, "timely decisions only")}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc2ef", BLUE, "#0f3f80"])
TOL = RP.TOL


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def _save(fig, out, name, title):
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / name, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _timely(r):
    return r["outcome_class"] in ("TIMELY_DEFINED", "TIMELY_UNDEFINED")


def _outcome_scatter(ax, rows, xkey, ykey, xf=lambda v: v):
    for cls, (color, marker, label) in OUTCOME_STYLE.items():
        pts = [(xf(r[xkey]), r[ykey]) for r in rows if r["outcome_class"] == cls and r[xkey] is not None and r[ykey] is not None]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=14, color=color, marker=marker, label=f"{label} (n={len(pts)})", alpha=0.75,
                       linewidths=0.6 if marker == "x" else 0)


def fig_actions(rows, out, tag):
    bins = np.arange(3000, 10001, 250)
    modes = list(RP.SUP.ALLOWED_MODES)
    grid = np.zeros((len(modes), len(bins) - 1))
    for r in rows:
        m, q = r["executed_action"]["mode_id"], r["executed_action"]["q_e4"]
        if m in modes:
            grid[modes.index(m), min(len(bins) - 2, max(0, np.searchsorted(bins, q, side="right") - 1))] += 1
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), facecolor=SURFACE, gridspec_kw=dict(height_ratios=[1, 1.4]))
    im = axes[0].imshow(grid, aspect="auto", cmap=SEQ, extent=[bins[0], bins[-1], len(modes) - 0.5, -0.5])
    axes[0].set_yticks(range(len(modes)), [f"mode {m}" for m in modes])
    for i, m in enumerate(modes):
        lo, hi = RP.SUP.BRACKETS[m]
        axes[0].plot([lo, hi], [i, i], color=INK2, lw=0.6, ls=":")
    _style(axes[0], "Executed policy decisions per mode and q_e4 bin (250); dotted = registered bracket", "q_e4")
    axes[0].grid(False)
    cb = fig.colorbar(im, ax=axes[0], pad=0.01)
    cb.set_label("decisions", color=INK2, fontsize=8)
    seq = [(i, r["executed_action"]["q_e4"], r["outcome_class"]) for i, r in enumerate(rows)]
    for cls, (color, marker, label) in OUTCOME_STYLE.items():
        pts = [(i, q) for i, q, c in seq if c == cls]
        if pts:
            x, y = zip(*pts)
            axes[1].scatter(x, y, s=12, color=color, marker=marker, label=f"{label} (n={len(pts)})", linewidths=0.6 if marker == "x" else 0)
    _style(axes[1], "Executed q_e4 per decision (all mode 11 unless noted), by outcome", "decision index", "q_e4")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _save(fig, out, f"01_action_heatmap_{tag}.png", f"Actions chosen by the frozen pilot actor ({tag})")


def fig_timeline(rows, out, tag):
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    for cls, (color, marker, label) in OUTCOME_STYLE.items():
        pts = [(i, r["ue_receipt"] if r["ue_receipt"] is not None else np.nan) for i, r in enumerate(rows) if r["outcome_class"] == cls]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, [v if not np.isnan(v) else 0 for v in y], s=12, color=color, marker=marker,
                       label=f"{label} (n={len(pts)})", linewidths=0.6 if marker == "x" else 0)
    ax.axhline(170, color=INK, lw=1, ls="--")
    ax.annotate("170 ms deadline (inclusive)", (len(rows) * 0.99, 171), ha="right", va="bottom", color=INK, fontsize=8)
    gt = [r["gt_published"] for r in rows]
    ax.plot(range(len(rows)), gt, color=INK2, lw=0.8, alpha=0.6, label="GT ready (publish)")
    _style(ax, "UE feedback receipt per decision (ms after action-open); missing replies drawn at 0", "decision index", "ms")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=3, loc="upper left")
    _save(fig, out, f"02_latency_deadline_timeline_{tag}.png", f"Latency and deadline timeline ({tag})")


def _quality_scatter(ax, rows, xkey, title, xlabel):
    for subset, (color, label) in SUBSET_STYLE.items():
        sub = [r for r in rows if r["quality_defined"] and r[xkey] is not None and (subset == "all" or _timely(r))]
        if sub:
            ax.scatter([r[xkey] for r in sub], [r["q_perc"] for r in sub], s=12 if subset == "all" else 6, color=color,
                       alpha=0.5 if subset == "all" else 0.9, label=f"{label} (n={len(sub)})")
    _style(ax, title, xlabel, "Q_perc")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)


def fig_snr(rows, out, tag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    _outcome_scatter(axes[0], rows, "snr_db", "ue_receipt")
    axes[0].axhline(170, color=INK, lw=1, ls="--")
    _style(axes[0], "Receipt vs SNR (decision state)", "SNR (dB)", "receipt (ms)")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    _quality_scatter(axes[1], rows, "snr_db", "Q_perc vs SNR", "SNR (dB)")
    _save(fig, out, f"03_snr_latency_quality_{tag}.png", f"SNR vs latency and quality ({tag})")


def fig_scene(rows, out, tag):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    for col, (key, label) in enumerate((("camera_si", "camera SI"), ("radar_p40", "radar P40"))):
        _outcome_scatter(axes[0][col], rows, key, "ue_receipt")
        axes[0][col].axhline(170, color=INK, lw=1, ls="--")
        _style(axes[0][col], f"Receipt vs {label}", label, "receipt (ms)")
        axes[0][col].legend(frameon=False, fontsize=7, labelcolor=INK2)
        _quality_scatter(axes[1][col], rows, key, f"Q_perc vs {label}", label)
    _save(fig, out, f"04_si_p40_latency_quality_{tag}.png", f"Scene descriptors vs latency and quality ({tag})")


def fig_localization(rows, out, tag):
    scored = [r for r in rows if r["service_outcome"] == "SCORED"]
    timely = [r for r in scored if _timely(r)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE)
    width = 0.38
    for k, (subset, sub) in enumerate((("all", scored), ("timely", timely))):
        color, label = SUBSET_STYLE[subset]
        vals = [RP.pooled(sub, c)["recall_within_tolerance"] or 0 for c in ("vehicle", "person")]
        gate = [RP.pooled(sub, c)["recall"] or 0 for c in ("vehicle", "person")]
        x = np.arange(2) + (k - 0.5) * width
        axes[0].bar(x, vals, width * 0.95, color=color, label=f"{label}: within tolerance")
        axes[0].scatter(x, gate, color=INK, marker="_", s=200, label=None if k else "recall at the 3 m gate")
        for xi, v in zip(x, vals):
            axes[0].annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    axes[0].set_xticks(range(2), ["vehicle (1.0 m)", "person (1.2 m)"])
    axes[0].set_ylim(0, 1.05)
    _style(axes[0], "Recall within tolerance (bars) and at the 3 m gate (ticks)", None, "recall")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2, loc="upper right")
    for ax, cls in ((axes[1], "vehicle"), (axes[2], "person")):
        for subset, sub in (("all", scored), ("timely", timely)):
            color, label = SUBSET_STYLE[subset]
            e = np.sort([x for r in sub for x in (r[f"{cls}_matched_xy_errors_m"] or [])])
            if len(e):
                ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=color, lw=2, label=f"{label} (n={len(e)} matches)")
        ax.axvline(TOL[cls], color=INK, lw=1, ls="--")
        ax.annotate(f"{TOL[cls]} m tolerance", (TOL[cls], 1.0), xytext=(4, -2), textcoords="offset points", fontsize=8,
                    color=INK, va="top")
        _style(ax, f"{cls.capitalize()} matched XY error CDF", "error (m)", "fraction of matches")
        ax.legend(frameon=False, fontsize=7, labelcolor=INK2, loc="lower right")
    _save(fig, out, f"05_localization_{tag}.png", f"Localization: vehicle 1.0 m / person 1.2 m ({tag})")


def fig_segmentation(rows, out, tag):
    scored = [r for r in rows if r["service_outcome"] == "SCORED"]
    timely = [r for r in scored if _timely(r)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    width = 0.38
    for k, (subset, sub) in enumerate((("all", scored), ("timely", timely))):
        color, label = SUBSET_STYLE[subset]
        vals = [RP.pooled(sub, c)["seg_iou_pooled"] or 0 for c in ("vehicle", "person")]
        x = np.arange(2) + (k - 0.5) * width
        axes[0].bar(x, vals, width * 0.95, color=color, label=label)
        for xi, v in zip(x, vals):
            axes[0].annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
        per_frame = [r["seg_vehicle_intersection_pixels"] / r["seg_vehicle_union_pixels"] for r in sub
                     if (r["seg_vehicle_union_pixels"] or 0) > 0]
        if per_frame:
            e = np.sort(per_frame)
            axes[1].step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=color, lw=2, label=f"{label} (n={len(e)} frames)")
    axes[0].set_xticks(range(2), ["vehicle", "person"])
    axes[0].set_ylim(0, 1.05)
    _style(axes[0], "Pooled segmentation IoU", None, "IoU")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(axes[1], "Per-frame vehicle IoU CDF", "IoU", "fraction of frames")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    _save(fig, out, f"06_segmentation_{tag}.png", f"Segmentation ({tag})")


def fig_freshness(rows, out, tag):
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    for subset, (color, label) in SUBSET_STYLE.items():
        e = np.sort([r["freshness_capture_to_feedback_ms"] for r in rows if r["freshness_capture_to_feedback_ms"] is not None
                     and (subset == "all" or _timely(r))])
        if len(e):
            ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=color, lw=2,
                    label=f"{label.replace('scored ', '')} (n={len(e)}, P50 {np.median(e):.0f} ms)")
    _style(ax, "Capture-to-feedback age (CARLA capture tick end -> UE receipt)", "age (ms)", "fraction of decisions")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    _save(fig, out, f"07_freshness_{tag}.png", f"Freshness ({tag})")


def fig_cdf(rows, out, tag):
    stages = (("service_prediction_received", "prediction at service", BLUE), ("gt_published", "GT ready (publish)", ORANGE),
              ("scoring_worker_end", "scoring end", AQUA), ("ue_receipt", "UE receipt", INK))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for key, label, color in stages:
        e = np.sort([r[key] for r in rows if r[key] is not None])
        if not len(e):
            continue
        y = np.arange(1, len(e) + 1) / len(e)
        axes[0].step(e, y, where="post", color=color, lw=2, label=f"{label} (P50 {np.median(e):.1f})")
        axes[1].step(e[:-1], 1 - y[:-1], where="post", color=color, lw=2, label=label)   # last point (0) is not drawn
    for ax in axes:
        ax.axvline(170, color=INK, lw=1, ls="--")
    axes[1].set_yscale("log")
    _style(axes[0], "CDF of boundary times (ms after action-open)", "ms", "fraction of decisions")
    _style(axes[1], "Exceedance P(X > t), log scale (to the largest value); dashed = 170 ms", "ms", "exceedance")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    _save(fig, out, f"08_cdf_exceedance_{tag}.png", f"Latency CDF and exceedance ({tag})")


def render(arm_dir, out, tag):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arm, rows, evidence = RP.rows(arm_dir)
    for f in (fig_actions, fig_timeline, fig_snr, fig_scene, fig_localization, fig_segmentation, fig_freshness, fig_cdf):
        f(rows, out, tag)
    with open(out / f"DECISIONS_{tag}.jsonl", "w") as stream:
        for r in rows:
            stream.write(json.dumps(r, default=str, sort_keys=True) + "\n")
    summary = RP.summarize(arm, rows, evidence)
    (out / f"SUMMARY_{tag}.json").write_text(json.dumps(summary, indent=1, default=str, sort_keys=True))
    return summary


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2], sys.argv[3])
