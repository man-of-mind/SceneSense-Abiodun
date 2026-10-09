"""Figures and tables for the reward-trial report (reads runs_01 and eval_01; writes report_01, create-only).

  python3 -m rl_agent.splitfusion_run5_reward_trials_v1.report --runs experiments/splitfusion_run5_reward_trials_v1/runs_01 \
      --eval experiments/splitfusion_run5_reward_trials_v1/eval_01 --out experiments/splitfusion_run5_reward_trials_v1/report_01
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
import numpy as np

ARMS = ("C0", "C1", "C2")
SEEDS = (17, 29, 43)
COLOR = {"C0": "#2a78d6", "C1": "#eb6834", "C2": "#1baf7a", "HIST": "#8a8984"}
MARKER = {"C0": "o", "C1": "s", "C2": "^", "HIST": "D"}
LABEL = {"C0": "C0  Y(0.7+0.3S)", "C1": "C1  Y(0.9+0.1S)", "C2": "C2  (0.75Y+0.25M)(0.9+0.1S)",
         "HIST": "deployed seed-43 (historical, old binding)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"
TRAIL = 200
ENVS = {"E1": "E1 primary: held scenes + held gc_v2 blocks", "E2": "E2 secondary: held scenes + retention session"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(*args, **kw):
    fig, axes = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------ 1. replay reward vs updates
def replay_reward(runs: Path, out: Path) -> dict:
    series, rows = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            m = [json.loads(l) for l in (runs / f"{arm}_seed{seed}/metrics.jsonl").read_text().splitlines()]
            r = np.asarray([x["reward_mean"] for x in m]); u = np.asarray([x["update"] for x in m])
            trail = np.convolve(r, np.ones(TRAIL) / TRAIL, mode="valid")
            series[(arm, seed)] = (u, r, trail)
            for x, t in zip(m, np.concatenate([np.full(TRAIL - 1, np.nan), trail])):
                rows.append(dict(arm=arm, seed=seed, update=x["update"], replay_reward_mean=x["reward_mean"],
                                 trailing_200=None if np.isnan(t) else float(t)))
    with open(out / "REPLAY_REWARD.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axes = _fig(1, 3, figsize=(13, 3.8), sharey=True)
    styles = {17: "-", 29: "--", 43: ":"}
    for ax, arm in zip(axes, ARMS):
        for seed in SEEDS:
            u, r, trail = series[(arm, seed)]
            ax.plot(u, r, color=COLOR[arm], alpha=0.08, linewidth=0.5)
            ax.plot(u[TRAIL - 1:], trail, color=COLOR[arm], linestyle=styles[seed], linewidth=1.6, label=f"seed {seed}")
        ax.set_title(LABEL[arm], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("gradient update", fontsize=8, color=INK2)
        ax.set_xlim(0, 10_000)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_ylabel("replay-minibatch mean reward\n(each arm's own formula)", fontsize=8, color=INK2)
    fig.suptitle("Replay-minibatch reward vs update, by arm (faint = raw per update; lines = 200-update trailing mean). "
                 "Scales differ by formula: do not compare levels across panels.", fontsize=8.5, color=INK, x=0.01, ha="left")
    _save(fig, out / "fig1_replay_reward_by_arm.png")
    return {f"{a}_seed{s}": dict(first_trailing=float(series[(a, s)][2][0]), last_trailing=float(series[(a, s)][2][-1]))
            for a in ARMS for s in SEEDS}


# ------------------------------------------------------------------------ helpers on EVALUATION.json
def _get(doc, label, env, path):
    v = doc["results"][label][env]["pooled"]
    for k in path:
        v = v[k]
    return v


def _points(doc, env, path):
    out = {arm: [_get(doc, f"{arm}_seed{s}", env, path) for s in SEEDS] for arm in ARMS}
    out["HIST"] = [_get(doc, "HIST_seed43", env, path)]
    return out


def _strip(ax, pts, *, tier=None, title="", ylabel=""):
    for i, key in enumerate(("C0", "C1", "C2", "HIST")):
        vals = [v for v in pts[key] if v is not None]
        xs = i + np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [i]
        ax.scatter(xs, vals, s=34, marker=MARKER[key], color=COLOR[key], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if key != "HIST":
            ax.hlines(np.mean(vals), i - 0.28, i + 0.28, color=COLOR[key], linewidth=2, zorder=2)
    if tier is not None:
        ax.axhline(tier, color=INK2, linestyle="--", linewidth=1)
        ax.text(3.45, tier, f"tier {tier:.2f}", fontsize=7, color=INK2, va="bottom", ha="right")
    ax.set_xticks(range(4), ["C0", "C1", "C2", "hist."], fontsize=8)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _legend(fig):
    handles = [plt.Line2D([], [], marker=MARKER[k], color=COLOR[k], linestyle="", markersize=6, label=LABEL[k])
               for k in ("C0", "C1", "C2", "HIST")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))


# ------------------------------------------------------------------------ 2. primary metric + timeliness
def heldout(doc, out):
    fig, axes = _fig(2, 3, figsize=(12, 6.5))
    for row, env in enumerate(ENVS):
        _strip(axes[row, 0], _points(doc, env, ["primary_timely_loc"]), title=f"{env}: timely within-tol localisation P",
               ylabel=ENVS[env].split(":")[0])
        _strip(axes[row, 1], _points(doc, env, ["timely"]), title=f"{env}: timeliness (receipt <= 170 ms)")
        _strip(axes[row, 2], _points(doc, env, ["otr_mean"]), title=f"{env}: per-decision mean D*Y (OTR)")
    fig.suptitle("Held-out comparison across training seeds (dots = seeds, bar = arm mean; same scenes and schedules for every policy)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, out / "fig2_heldout_primary_and_timeliness.png")


# ------------------------------------------------------------------------ 3. per-class quality and retention
def class_quality(doc, out, env="E1"):
    tiers = {"loc": (0.95, 0.90), "seg": (0.85, 0.80), "precision": (0.95, 0.90)}
    names = {"loc": "localisation (recall within tol)", "seg": "segmentation IoU", "precision": "precision"}
    fig, axes = _fig(2, 3, figsize=(12, 6.5))
    for row, cls in enumerate(("vehicle", "person")):
        for col, k in enumerate(("loc", "seg", "precision")):
            _strip(axes[row, col], _points(doc, env, ["retention", cls, k]), tier=tiers[k][row],
                   title=f"{cls}: {names[k]} retention", ylabel="policy / uncompressed-equivalent" if col == 0 else "")
    fig.suptitle(f"{ENVS[env]}: pooled per-class retention vs the same-scene noAE/UINT8 q0 reference "
                 "(all decisions; dashed = moderate tier)", fontsize=9, color=INK, x=0.01, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, out / f"fig3_class_retention_{env}.png")

    fig, axes = _fig(2, 4, figsize=(14, 6.5))
    for row, cls in enumerate(("vehicle", "person")):
        _strip(axes[row, 0], _points(doc, env, ["quality_all", cls, "recall_within_tolerance"]),
               title=f"{cls}: recall within tol (all)", ylabel=cls)
        _strip(axes[row, 1], _points(doc, env, ["quality_timely_hits", cls, "recall_within_tolerance"]),
               title=f"{cls}: timely recall within tol")
        _strip(axes[row, 2], _points(doc, env, ["quality_all", cls, "fp_per_decision"]), title=f"{cls}: FP per decision")
        _strip(axes[row, 3], _points(doc, env, ["quality_all", cls, "seg_iou"]), title=f"{cls}: pooled seg IoU")
        ref = doc["results"]["C0_seed17"][env]["pooled"]["reference"][cls]
        for col, key in ((0, "recall_within_tolerance"), (2, "fp_per_decision"), (3, "seg_iou")):
            axes[row, col].axhline(ref[key], color=INK2, linestyle=":", linewidth=1)
            axes[row, col].text(3.45, ref[key], "reference", fontsize=7, color=INK2, va="bottom", ha="right")
    fig.suptitle(f"{ENVS[env]}: absolute per-class quality (misses count; dotted = uncompressed-equivalent reference on the same scenes)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, out / f"fig4_class_quality_{env}.png")


# ------------------------------------------------------------------------ 4. channel response
PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "FADE_RECOVERY", "ADVERSE_STABLE")


def channel(doc, out, env="E1"):
    fig, axes = _fig(1, 4, figsize=(14, 3.6))
    metrics = (("timely", "timeliness"), ("timely_loc", "timely within-tol localisation P"),
               ("wire_p50", "payload P50 (KB)"), ("q_p50", "q_e4 P50"))
    x = np.arange(len(PROFILES))
    for ax, (key, title) in zip(axes, metrics):
        for arm in ("C0", "C1", "C2", "HIST"):
            labels = [f"{arm}_seed{s}" for s in SEEDS] if arm != "HIST" else ["HIST_seed43"]
            vals = np.asarray([[doc["results"][l][env]["pooled"]["by_profile"][p][key] for p in PROFILES] for l in labels])
            if key == "wire_p50":
                vals = vals / 1000
            mean = vals.mean(0)
            ax.plot(x, mean, color=COLOR[arm], marker=MARKER[arm], linewidth=2, markersize=6, label=LABEL[arm],
                    linestyle="--" if arm == "HIST" else "-")
            if len(labels) > 1:
                ax.fill_between(x, vals.min(0), vals.max(0), color=COLOR[arm], alpha=0.12, linewidth=0)
        ax.set_xticks(x, ["FAV", "MID", "FADE", "ADV"], fontsize=8)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[0].set_xlabel("channel profile (Run-5 registered profiles)", fontsize=8, color=INK2)
    fig.suptitle(f"{ENVS[env]}: channel response (line = seed mean, band = seed range)", fontsize=9, color=INK, x=0.01, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, out / f"fig5_channel_response_{env}.png")


def actions(doc, out, env="E1"):
    fig, axes = _fig(1, 2, figsize=(12, 3.6))
    for arm in ("C0", "C1", "C2", "HIST"):
        labels = [f"{arm}_seed{s}" for s in SEEDS] if arm != "HIST" else ["HIST_seed43"]
        qs = np.asarray([doc["results"][l][env]["pooled"]["q_p5_p50_p95"] for l in labels]).mean(0)
        ws = np.asarray([doc["results"][l][env]["pooled"]["wire_p5_p50_p95"] for l in labels]).mean(0) / 1000
        i = ("C0", "C1", "C2", "HIST").index(arm)
        for ax, v in ((axes[0], qs), (axes[1], ws)):
            ax.vlines(i, v[0], v[2], color=COLOR[arm], linewidth=2)
            ax.scatter([i], [v[1]], marker=MARKER[arm], color=COLOR[arm], s=40, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for ax, t in ((axes[0], "q_e4: P5-P95 (line), P50 (marker), seed-mean"), (axes[1], "payload KB: P5-P95, P50, seed-mean")):
        ax.set_xticks(range(4), ["C0", "C1", "C2", "hist."], fontsize=8)
        ax.set_title(t, fontsize=8.5, color=INK, loc="left")
    fig.suptitle(f"{ENVS[env]}: selected q and payload", fontsize=9, color=INK, x=0.01, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    _save(fig, out / f"fig6_actions_{env}.png")


# ------------------------------------------------------------------------ tables
def table(doc, out):
    rows = []
    for label in sorted(doc["results"]):
        for env in ENVS:
            s = doc["results"][label][env]["pooled"]
            row = dict(policy=label, env=env, primary_P=s["primary_timely_loc"], timely=s["timely"], otr=s["otr_mean"],
                       receipt_p50=s["receipt_p50"], receipt_p95=s["receipt_p95"], wire_p50=s["wire_p5_p50_p95"][1],
                       q_p50=s["q_p5_p50_p95"][1], mode11=s["mode_share"]["11"] if "11" in s["mode_share"] else s["mode_share"][11],
                       excluded=s["excluded_timely_undefined"])
            for cls in ("vehicle", "person"):
                qa, qt, ret = s["quality_all"][cls], s["quality_timely_hits"][cls], s["retention"][cls]
                row.update({f"{cls}_recall_tol": qa["recall_within_tolerance"], f"{cls}_timely_recall_tol": qt["recall_within_tolerance"],
                            f"{cls}_precision": qa["precision"], f"{cls}_fp_per_dec": qa["fp_per_decision"],
                            f"{cls}_seg_iou": qa["seg_iou"], f"{cls}_ret_loc": ret["loc"], f"{cls}_ret_seg": ret["seg"],
                            f"{cls}_ret_precision": ret["precision"]})
            for a, v in s["realized_reward_by_formula"].items():
                row[f"realized_{a}"] = v
            rows.append(row)
    with open(out / "HELDOUT_TABLE.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def main(runs: Path, eval_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    doc = json.loads((eval_dir / "EVALUATION.json").read_text())
    summary = dict(replay=replay_reward(runs, out))
    heldout(doc, out)
    for env in ENVS:
        class_quality(doc, out, env)
        channel(doc, out, env)
        actions(doc, out, env)
    table(doc, out)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.runs, a.eval, a.out)
