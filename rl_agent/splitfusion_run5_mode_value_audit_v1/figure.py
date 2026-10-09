"""Figures: fig_favorable_pareto.png (old 288 metric vs current objective, separate panels) and
fig_actor_trace_favorable.png (recorded gc_v2 FAVORABLE states: SNR/MCS, mode probabilities, mode-conditional q)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_run5_full_v1.support import in_support

from . import compare as CP

COL = {11: "#2a78d6", 10: "#eb6834", 8: "#1baf7a", 9: "#eda100"}
MK = {11: "o", 10: "s", 8: "^", 9: "D"}
GRAY = "#9a9a94"
INK, INK2 = "#222220", "#5f5e58"
FAMILY_MODE = {("noAE", "UINT8"): 0, ("noAE", "UINT6"): 1, ("noAE", "UINT4"): 2}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c0")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color="#ecebe5", lw=0.6)
    ax.set_axisbelow(True)


def pareto_fig(root: Path):
    d = json.loads((root / "PART1_PARETO.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    # panel 1: OLD
    LAB_A = {(8, 3000): (6, -12), (8, 9000): (6, 6), (10, 9000): (6, -12), (10, 9800): (6, 6), (0, 0): (-10, 8),
             (8, 0): (-62, 8)}
    for r in d["layer_A_old_288_favorable"]:
        m = r["action_id"] // 6
        ok = in_support(m, r["q_e4"])
        c = COL.get(m, GRAY)
        nd = r["pareto_payload_quality"]
        mk = MK.get(m, "v")
        a1.scatter(r["payload_kb"], r["old_combined_quality"], s=60 if nd else 24, marker=mk,
                   facecolors=c if ok else "white", edgecolors=c, linewidths=1.6 if nd else 0.8,
                   alpha=1 if nd else 0.45, zorder=3 if nd else 2)
        if nd and (m, r["q_e4"]) in LAB_A:
            a1.annotate(f"m{m} q{r['q_e4']}" + ("" if ok else " (unsupported)"), (r["payload_kb"], r["old_combined_quality"]),
                        xytext=LAB_A[(m, r["q_e4"])], textcoords="offset points", fontsize=7, color=INK2)
    a1.set_xscale("log")
    a1.set_xlabel("median live wire payload per frame (KB, log)", fontsize=9, color=INK)
    a1.set_ylabel("OLD combined_quality (offline val set, 3-m gate; presentation coordinate)", fontsize=8.5, color=INK)
    a1.set_title("A. OLD: original 288 campaign, FAVORABLE_STABLE (measured live payload)\n"
                 "large = non-dominated on (payload, old quality); hollow = outside actor q support\n"
                 "old quality is action-level, 3-m gate; no 170-ms clock (500-ms map boundary)", fontsize=9, color=INK, loc="left")
    style(a1)
    # panel 2: CURRENT (training evidence, FAVORABLE decisions)
    B = d["layer_B_current_env_train_evidence"]
    tiers5 = ("loc_vehicle", "loc_person", "seg_vehicle", "seg_person", "precision_vehicle")
    TIER = dict(loc_vehicle=.95, loc_person=.90, seg_vehicle=.85, seg_person=.80, precision_vehicle=.95)
    for k, v in B.items():
        f = v["FAVORABLE_STABLE"]
        m = v["mode"]
        meets = all(f["ret"][t] >= TIER[t] for t in tiers5)
        c = COL.get(m, GRAY)
        a2.scatter(f["wire_p50_kb"], f["P"], s=62 if meets else 26, marker=MK.get(m, "v"), facecolors=c if meets else "white",
                   edgecolors=c, linewidths=1.4 if meets else 0.8, alpha=1 if meets else 0.6, zorder=3 if meets else 2)
        LAB_B = {(11, 3000): (6, 4), (11, 5000): (6, 4), (11, 6000): (6, 4), (11, 7000): (6, 4), (11, 8000): (-46, 6),
                 (10, 8000): (-50, 6), (10, 7000): (8, -2), (8, 8000): (-14, -14)}
        if (m, v["q_e4"]) in LAB_B:
            a2.annotate(f"m{m} q{v['q_e4']}", (f["wire_p50_kb"], f["P"]), xytext=LAB_B[(m, v["q_e4"])], textcoords="offset points",
                        fontsize=7, color=INK2)
    a2.set_xscale("log")
    a2.set_xlabel("modelled wire payload P50 (KB, log)", fontsize=9, color=INK)
    a2.set_ylabel("CURRENT P = timely (<=170 ms) within-tolerance localisation\n0.4 vehicle@1.0 m + 0.6 person@1.2 m, misses count", fontsize=8.5, color=INK)
    a2.set_title("B. CURRENT: corrected latency env, FAVORABLE decisions, training evidence\n"
                 "filled = meets the 5 tiers the deployed actor meets (loc v/p, seg v/p, prec v)\n"
                 "hollow = fails >= 1; person precision >= 0.90 only at m11 q <= 1500", fontsize=9, color=INK, loc="left")
    style(a2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=MK[m], color=COL[m], label=f"mode {m}") for m in (11, 10, 8, 9)]
    h.append(Line2D([], [], ls="", marker="v", color=GRAY, label="other modes (0-7)"))
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(root / "fig_favorable_pareto.png", dpi=150)


def trace_fig(root: Path):
    recs = [json.loads(l) for l in open(root / "part2_trace_FAVORABLE_gc_v2.jsonl")]
    x = np.array([r["decision"] for r in recs])
    P = np.array([r["p"] for r in recs])
    Q = np.array([r["qm"] for r in recs])
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(x, [r["snr_db"] for r in recs], color="#2a78d6", lw=1.2, label="causal SNR (dB)")
    ax0b = ax[0]
    ax0b.plot(x, [r["prior_ul_mcs"] for r in recs], color="#eb6834", lw=1.2, label="prior UL MCS")
    ax[0].set_ylabel("SNR dB / MCS", fontsize=8.5)
    ax[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 0.98))
    for m in (11, 8, 9, 10):
        ax[1].plot(x, P[:, m], color=COL[m], lw=1.2 if m == 11 else 1.0, label=f"P(mode {m})")
    ax[1].set_ylabel("mode probability", fontsize=8.5)
    ax[1].legend(fontsize=7.5, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 0.98))
    for m in (11, 8, 9, 10):
        ax[2].plot(x, Q[:, m], color=COL[m], lw=1.2 if m == 11 else 1.0, label=f"conditional q, mode {m}")
    sel = [i for i, r in enumerate(recs) if r["mode"] != 11]
    ax[2].scatter(x[sel], [recs[i]["q"] for i in sel], s=60, facecolors="none", edgecolors=INK, label="executed non-11 (dec 25, 26)", zorder=4)
    ax[2].set_ylabel("q_e4", fontsize=8.5)
    ax[2].legend(fontsize=7.5, frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 0.98))
    late = np.array([r["outcome"] == "LATE" for r in recs])
    pay = np.array([r["payload_kb"] for r in recs])
    ax[3].scatter(x[~late], pay[~late], s=8, color="#2a78d6", label="timely (observed)")
    ax[3].scatter(x[late], pay[late], s=10, marker="x", color="#eb6834", label="late (observed)")
    ax[3].set_ylabel("payload KB (observed)", fontsize=8.5)
    ax[3].set_xlabel("decision index, gc_v2 1,000-frame FAVORABLE_STABLE validation (recorded states; deployed actor re-queried on CPU, 494/494 reproduced)", fontsize=8.5)
    ax[3].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 0.98))
    for a in ax:
        style(a)
    fig.suptitle("Deployed actor on recorded favourable states: argmax mode 11 on 492/494; runner-up modes 8/9 carry their own much higher q",
                 fontsize=9.5, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(root / "fig_actor_trace_favorable.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    r = p.parse_args().out
    pareto_fig(r)
    trace_fig(r)
