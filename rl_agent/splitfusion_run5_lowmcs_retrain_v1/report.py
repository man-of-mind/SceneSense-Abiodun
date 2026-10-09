"""Figures and generated tables for the comparison (reads runs_01, eval_01, fade_01; create-only report dir).

    python3 -m rl_agent.splitfusion_run5_lowmcs_retrain_v1.report --root <ABS experiments/splitfusion_run5_lowmcs_retrain_v1>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import registration as REG

INK, MUTED, GRID = "#2b2b2b", "#77766f", "#e6e5e0"
NEW, C0, HIST, FIX = "#2a78d6", "#eb6834", "#2b2b2b", "#9a998f"     # validated categorical pair + neutral inks
STYLES = {17: "-", 29: "--", 43: ":"}
C0_RUNS = REG.ROOT / "experiments/splitfusion_run5_reward_trials_v1/runs_01"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def runs(root):
    out = {}
    for s in REG.SEEDS:
        out[("NEW", s)] = root / f"runs_01/NEW_seed{s}"
        out[("C0", s)] = C0_RUNS / f"C0_seed{s}"
    return out


def training_curves(root, out):
    keys = (("reward_mean", "replay-minibatch reward (200-update mean)"), ("discrete_entropy", "categorical mode entropy"),
            ("critic_loss", "critic loss"), ("actor_loss", "actor loss"), ("critic_grad_norm", "critic grad norm"),
            ("actor_grad_norm", "actor grad norm"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for (group, s), d in runs(root).items():
        m = _jl(d / "metrics.jsonl")
        seen, rows = set(), []
        for x in m:                                   # smoke re-steps reuse update numbers; keep the first record
            if x["update"] not in seen:
                seen.add(x["update"]); rows.append(x)
        u = np.array([x["update"] for x in rows])
        for ax, (k, title) in zip(axes.flat, keys):
            v = np.convolve([x[k] for x in rows], np.ones(200) / 200, mode="valid")
            ax.plot(u[199:], v, STYLES[s], color=NEW if group == "NEW" else C0, lw=1.6,
                    label=f"{'NEW (corrected env)' if group == 'NEW' else 'C0 (reward trials, old env)'} seed {s}")
            ax.set_title(title, color=INK, fontsize=10, loc="left")
            ax.set_xlabel("update", color=INK)
            _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.005, "Each environment's own reward; NEW and C0 rewards are on the same C0 formula but different "
             "latency models, so levels are not a policy comparison.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig1_training_curves.png", dpi=120)
    plt.close(fig)


def coverage(root, out):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    width = 0.13
    for j, ((group, s), d) in enumerate(runs(root).items()):
        rows = _jl(d / "decisions.jsonl")
        st = [r for r in rows if r["source"] == "STOCHASTIC_ACTOR"][-4000:]
        share = collections.Counter(r["mode_id"] for r in st)
        x = np.arange(12) + (j - 2.5) * width
        axes[0].bar(x, [share.get(m, 0) / len(st) for m in range(12)], width=width * 0.9,
                    color=NEW if group == "NEW" else C0, alpha=[1.0, 0.7, 0.45][REG.SEEDS.index(s)], label=f"{group} s{s}")
    warm = collections.Counter(r["mode_id"] for r in _jl(runs(root)[("NEW", 17)] / "decisions.jsonl")
                               if r["source"] == "STRATIFIED_WARMUP")
    axes[0].plot(range(12), [warm.get(m, 0) / 288 for m in range(12)], "_", ms=16, mew=2, color=INK,
                 label="prescribed warm-up (288)")
    axes[0].set_xticks(range(12))
    axes[0].set_xlabel("mode", color=INK)
    axes[0].set_ylabel("share of last 4,000 stochastic decisions", color=INK)
    axes[0].set_title("Mode coverage (training)", color=INK, fontsize=10, loc="left")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    _style(axes[0])
    strata = (("MCS 9–10", lambda r: r["prior_ul_mcs"] <= 10), ("MCS 11–15", lambda r: 11 <= r["prior_ul_mcs"] <= 15),
              ("MCS ≥ 16", lambda r: r["prior_ul_mcs"] >= 16))
    bins = np.arange(0, 10001, 400)
    for ax, (name, f) in zip(axes[1:], strata):
        for group, col in (("NEW", NEW), ("C0", C0)):
            q = [r["q_e4"] for s in REG.SEEDS for r in _jl(runs(root)[(group, s)] / "decisions.jsonl")[-4000:]
                 if r["mode_id"] == 11 and f(r)]
            ax.hist(q, bins=bins, histtype="step", lw=2, color=col, density=True, label=f"{group} (3 seeds, n={len(q)})")
        ax.set_title(f"Mode-11 q, last 4,000 per seed, {name}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("q_e4 (higher = smaller payload)", color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig2_mode_q_coverage.png", dpi=120)
    plt.close(fig)


def joint(ev, out):
    res = ev["results"]
    strata = (("all", "All decisions"), ("mcs_9_10", "Prior MCS 9–10"), ("mcs_11_15", "Prior MCS 11–15"),
              ("mcs_ge16", "Prior MCS ≥ 16"))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for col, (st, title) in enumerate(strata):
        for label, spec in ev["policies"].items():
            g = spec["group"]
            color = {"NEW": NEW, "C0": C0, "HIST": HIST, "FIXED": FIX}[g]
            marker = {"NEW": "o", "C0": "o", "HIST": "D", "FIXED": "s"}[g]
            for var, filled in (("CORRECTED", True), ("OLD", False)):
                x = res[label]["E1"][var]["pooled"][st]
                kw = dict(ms=9, color=color, mew=2, mfc=color if filled else "none",
                          markeredgecolor="white" if filled else color)
                axes[0, col].plot(x["wire_p50"] / 1e3, x["P"], marker, **kw)
                axes[1, col].plot(x["wire_p50"] / 1e3, x["timely"], marker, **kw)
            if st == "all" or True:
                xc = res[label]["E1"]["CORRECTED"]["pooled"][st]
                if g in ("HIST", "FIXED"):
                    axes[0, col].annotate(label.replace("FIXED_m11_", "").replace("_seed43", ""),
                                          (xc["wire_p50"] / 1e3, xc["P"]), textcoords="offset points", xytext=(6, -3),
                                          fontsize=7, color=INK)
        axes[0, col].set_title(title, color=INK, fontsize=10, loc="left")
        axes[1, col].set_xlabel("wire payload P50 (KB)", color=INK)
        for r in (0, 1):
            _style(axes[r, col])
    axes[0, 0].set_ylabel("P: timely within-tolerance localisation", color=INK)
    axes[1, 0].set_ylabel("feedback on time (≤ 170 ms)", color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=NEW, ms=8, label="NEW seeds (corrected-env training)"),
               Line2D([], [], marker="o", ls="", color=C0, ms=8, label="C0 seeds (reward trials)"),
               Line2D([], [], marker="D", ls="", color=HIST, ms=8, label="deployed seed 43"),
               Line2D([], [], marker="s", ls="", color=FIX, ms=8, label="fixed m11 q7000 / q8000"),
               Line2D([], [], marker="o", ls="", color=MUTED, ms=8, label="filled = CORRECTED env (primary)"),
               Line2D([], [], marker="o", ls="", color=MUTED, mfc="none", ms=8, label="hollow = OLD env (sensitivity)")]
    fig.legend(handles=handles, fontsize=8, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.01, 0.005, "E1 = held-out scenes + HELDOUT_GCV2 latency blocks; 3 eval seeds x 6,000 decisions per point; "
             "common scene and channel streams. Modelled, not live.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig3_joint_latency_quality_payload_E1.png", dpi=120)
    plt.close(fig)


def tables(ev, root, out):
    res, L = ev["results"], []
    order = [f"NEW_seed{s}" for s in REG.SEEDS] + [f"C0_seed{s}" for s in REG.SEEDS] + ["HIST_seed43", *REG.FIXED]
    f3 = lambda v: "—" if v is None else f"{v:.3f}"
    for env in ("E1", "E2"):
        for var in ("CORRECTED", "OLD"):
            for st in ("all", "mcs_9_10", "mcs_11_15", "mcs_ge16", "mcs_9_10|B0", "mcs_11_15|B0", "mcs_ge16|B0",
                       "mcs_9_10|Bpos", "mcs_11_15|Bpos", "mcs_ge16|Bpos"):
                L.append(f"\n### {env} {var} — {st}\n")
                L.append("| policy | n | P | P envelope | timely | receipt P50/P95/P99 | R_v^T / R_p^T | recall all-scored v/p | "
                         "recall on-time-only v/p | seg IoU v/p | precision v/p | retention loc v/p, seg v/p, prec v/p | "
                         "wire P50/P95/P99 KB | q P10/50/90 | mode-11 | flagged | flip-gain |")
                L.append("|" + "---|" * 17)
                for p in order:
                    x = res[p][env][var]["pooled"].get(st, {})
                    if x.get("n", 0) == 0 or "P" not in x:
                        L.append(f"| {p} | {x.get('n', 0)} | (not reported) |" + " |" * 14)
                        continue
                    rt = x["retention"]
                    L.append(
                        f"| {p} | {x['n']} | {f3(x['P'])} | {f3(x['P_envelope'])} | {f3(x['timely'])} | "
                        f"{x['receipt_p50']:.1f}/{x['receipt_p95']:.1f}/{x['receipt_p99']:.1f} | "
                        f"{f3(x['recall_timely']['vehicle'])} / {f3(x['recall_timely']['person'])} | "
                        f"{f3(x['recall_all_scored']['vehicle'])} / {f3(x['recall_all_scored']['person'])} | "
                        f"{f3(x['recall_on_time_only']['vehicle'])} / {f3(x['recall_on_time_only']['person'])} | "
                        f"{f3(x['seg_iou_all']['vehicle'])} / {f3(x['seg_iou_all']['person'])} | "
                        f"{f3(x['precision_all']['vehicle'])} / {f3(x['precision_all']['person'])} | "
                        f"{f3(rt['vehicle']['loc'])}/{f3(rt['person']['loc'])}, {f3(rt['vehicle']['seg'])}/{f3(rt['person']['seg'])}, "
                        f"{f3(rt['vehicle']['precision'])}/{f3(rt['person']['precision'])} | "
                        f"{x['wire_p50']/1e3:.0f}/{x['wire_p95']/1e3:.0f}/{x['wire_p99']/1e3:.0f} | "
                        f"{x['q_p10_p50_p90'][0]:.0f}/{x['q_p10_p50_p90'][1]:.0f}/{x['q_p10_p50_p90'][2]:.0f} | "
                        f"{x['mode_share'].get('11', x['mode_share'].get(11, 0)):.2f} | {x['flagged']:.3f} | {x['flip_gain_rate']:.3f} |")
    L.append("\n## Paired dP (NEW minus reference, common streams; block-bootstrap 95% CI)\n")
    L.append("| pair | stratum | dP [CI] | dP envelope | unflagged contribution | flip-gain contribution (decisions) |")
    L.append("|---|---|---|---|---|---|")
    for k, v in sorted(ev["paired"].items()):
        for st, x in v.items():
            ci = "" if not x.get("ci95") else f" [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
            L.append(f"| {k} | {st} | {x['dP']:+.4f}{ci} | {x['dP_envelope']:+.4f} | {x['dP_unflagged_contribution']:+.4f} | "
                     f"{x['dP_flip_gain_contribution']:+.4f} ({x['decisions_with_flip_gain']}) |")
    L.append("\n## Training exposure (RUN_RESULT.json)\n")
    L.append("| run | phase | decisions | low-MCS | UNRESOLVED_BACKLOG | EXTRAPOLATED_PAYLOAD | EXTRAPOLATED_MCS | any (rate / given low MCS) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in REG.SEEDS:
        rr = json.loads((root / f"runs_01/NEW_seed{s}/RUN_RESULT.json").read_text())
        for ph in ("warmup", "stochastic", "last_4000"):
            e = rr["exposure"][ph]
            L.append(f"| NEW_seed{s} | {ph} | {e['decisions']} | {e['low_mcs']} | {e['UNRESOLVED_BACKLOG']['count']} | "
                     f"{e['EXTRAPOLATED_PAYLOAD']['count']} | {e['EXTRAPOLATED_MCS']['count']} | "
                     f"{e['any']['rate']:.4f} / {e['any']['rate_given_low_mcs']:.4f} |")
    (out / "TABLES.md").write_text("# Generated tables (E = environment pool, CORRECTED/OLD latency variant)\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    out = root / "report_01"
    out.mkdir(exist_ok=False)
    ev = json.loads((root / "eval_01/EVALUATION.json").read_text())
    training_curves(root, out)
    coverage(root, out)
    joint(ev, out)
    tables(ev, root, out)
