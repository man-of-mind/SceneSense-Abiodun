"""Figures + generated tables (create-only report_01).

    python3 -m rl_agent.splitfusion_run5_constrained_v1.report --root <ABS experiments/splitfusion_run5_constrained_v1>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import objective as O
from .evaluate import PROFILES

INK, MUTED, GRID = "#2b2b2b", "#77766f", "#e6e5e0"
SEEDC = {17: "#2a78d6", 29: "#1baf7a", 43: "#4a3aa7"}
LAMC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]


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


def training(root, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for s in (17, 29, 43):
        rows, seen = [], set()
        for x in _jl(root / f"runs_01/CON_seed{s}/metrics.jsonl"):
            if x["update"] not in seen:
                seen.add(x["update"]); rows.append(x)
        u = np.array([x["update"] for x in rows])
        sm = lambda k: np.convolve([x[k] for x in rows], np.ones(200) / 200, mode="valid")
        for ax, k, t in ((axes[0, 0], "reward_mean", "replay-minibatch reward D·Y + λ·g/κ (200-update mean)"),
                         (axes[0, 1], "discrete_entropy", "categorical mode entropy"),
                         (axes[0, 2], "critic_loss", "critic loss"), (axes[1, 2], "actor_grad_norm", "actor grad norm")):
            ax.plot(u[199:], sm(k), color=SEEDC[s], lw=1.5, label=f"seed {s}")
            ax.set_title(t, color=INK, fontsize=10, loc="left")
        lam = np.array([x["lam"] for x in rows])
        ls = {17: "-", 29: "--", 43: ":"}[s]
        for i, name in enumerate(O.CONSTRAINTS):
            axes[1, 0].plot(u, lam[:, i], ls, color=LAMC[i], lw=1.4, label=name if s == 17 else None)
        dual = _jl(root / f"runs_01/CON_seed{s}/DUAL.jsonl")
        if dual:
            du = np.array([d["decision_count"] for d in dual]) / 4.0
            gm = np.array([d["window_mean"] for d in dual])
            for i in range(len(O.CONSTRAINTS)):
                axes[1, 1].plot(du, gm[:, i], ls, color=LAMC[i], lw=1.2)
    axes[1, 0].set_title("multipliers λ_k (solid 17, dashed 29, dotted 43)", color=INK, fontsize=10, loc="left")
    axes[1, 1].axhline(0, color=INK, lw=1)
    axes[1, 1].set_title("window mean g_k/κ_k (≈ retention − target; ≥ 0 = met on FIT)", color=INK, fontsize=10, loc="left")
    axes[1, 0].legend(fontsize=7, frameon=False)
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("update", color=INK)
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig1_training_curves_and_duals.png", dpi=120)
    plt.close(fig)


def profiles(ev, out):
    E1 = ev["results"]["E1|CORRECTED"]
    pols = ["HIST_seed43", "CON_seed17", "CON_seed29", "CON_seed43"]
    cols = {"HIST_seed43": INK, "CON_seed17": SEEDC[17], "CON_seed29": SEEDC[29], "CON_seed43": SEEDC[43]}
    panels = (("P", "timely within-tolerance localisation P", lambda m: m["P"]),
              ("timely", "feedback on time (≤ 170 ms)", lambda m: m["timely"]),
              ("segv", "vehicle segmentation retention", lambda m: m["vehicle"]["retention"]["seg"]),
              ("precp", "person precision retention", lambda m: m["person"]["retention"]["precision"]),
              ("wire", "payload P50 (KB)", lambda m: m["wire_p50_kb"]))
    groups = ("overall",) + PROFILES
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4.6))
    w = 0.2
    for ax, (key, title, f) in zip(axes, panels):
        for j, p in enumerate(pols):
            vals = [f(E1[p][g]) for g in groups]
            ax.bar(np.arange(len(groups)) + (j - 1.5) * w, vals, width=w * 0.9, color=cols[p],
                   label="deployed seed 43" if p.startswith("HIST") else f"constrained {p[4:]}")
        if key == "segv":
            ax.axhline(0.85, color=INK, ls="--", lw=1)
        ax.set_xticks(range(len(groups)), ["overall", "FAV", "MID", "ADV", "FADE"])
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        lo = min(f(E1[p][g]) for p in pols for g in groups)
        hi = max(f(E1[p][g]) for p in pols for g in groups)
        if key != "wire":
            ax.set_ylim(max(0, lo - 0.08), min(1.05, hi + 0.04))
        _style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.04))
    fig.text(0.01, 0.005, "E1-CORRECTED (held-out scenes + HELDOUT_GCV2), 3 eval seeds x 6,000 decisions, identical scene/channel "
             "streams; profiles are the actual hidden channel profile of each decision. Modelled, not live.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig2_profile_comparison_E1.png", dpi=120)
    plt.close(fig)


def tables(ev, out):
    L = []
    f3 = lambda v: "—" if v is None else f"{v:.3f}"
    for envvar in ("E1|CORRECTED", "E2|CORRECTED", "E1|OLD", "E2|OLD"):
        res = ev["results"][envvar]
        for g in ("overall",) + PROFILES + ("mcs_9_10", "mcs_11_15", "mcs_ge16"):
            L.append(f"\n### {envvar} — {g}\n")
            L.append("| policy | n | on time | P | ΔP vs deployed [CI] | R_v@1.0 / R_p@1.2 (all-scored) | retention loc v/p | seg v/p | "
                     "prec v/p | payload P50/P99 KB | q P10/50/90 | mode-11 | SNR P50 | MCS P50 | B>0 |")
            L.append("|" + "---|" * 15)
            for p in ("HIST_seed43", "CON_seed17", "CON_seed29", "CON_seed43"):
                m = res[p][g]
                pv = res[p].get("paired_vs_deployed", {}).get(g)
                dp = "—" if pv is None else f"{pv['dP']:+.4f} [{pv['ci95'][0]:+.4f}, {pv['ci95'][1]:+.4f}]"
                r = lambda c, k: m[c]["retention"][k]
                L.append(f"| {p} | {m['n']} | {f3(m['timely'])} | {f3(m['P'])} | {dp} | "
                         f"{f3(m['vehicle']['recall_within_tol'])} / {f3(m['person']['recall_within_tol'])} | "
                         f"{f3(r('vehicle','loc'))} / {f3(r('person','loc'))} | {f3(r('vehicle','seg'))} / {f3(r('person','seg'))} | "
                         f"{f3(r('vehicle','precision'))} / {f3(r('person','precision'))} | {m['wire_p50_kb']:.0f} / {m['wire_p99_kb']:.0f} | "
                         f"{m['q_p10_p50_p90'][0]:.0f}/{m['q_p10_p50_p90'][1]:.0f}/{m['q_p10_p50_p90'][2]:.0f} | {m['mode11_share']:.2f} | "
                         f"{m['snr_p50']:.1f} | {m['mcs_p50']:.0f} | {m['positive_backlog']} |")
    (out / "TABLES.md").write_text("# Generated tables (all-scored retention vs each set's own reference totals)\n" + "\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    out = root / "report_01"
    out.mkdir(exist_ok=False)
    ev = json.loads((root / "eval_01/EVALUATION.json").read_text())
    training(root, out)
    profiles(ev, out)
    tables(ev, out)
