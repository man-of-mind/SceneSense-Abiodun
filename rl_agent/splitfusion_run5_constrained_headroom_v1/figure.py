"""One trade-off figure: (a) P vs vehicle segmentation retention, E1-CORRECTED seed 9017, all 71 grid actions + policies
+ privileged bounds; (b) P by prior-MCS stratum on the full held-out E1-CORRECTED (3 eval seeds)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#2b2b2b", "#77766f", "#e6e5e0"
COL = {"HIST": "#2b2b2b", "C0": "#eb6834", "NEW": "#2a78d6", "RULE": "#1baf7a", "S2": "#e87ba4", "FIX": "#9a998f"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def draw(out, root):
    pv = out["privileged_E1_CORRECTED_seed9017"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for name, m in pv["fixed_actions"].items():
        is11 = name.startswith("m11_")
        a.plot(m["vehicle"]["retention"]["seg"], m["P"], "o", ms=4 if not is11 else 5, color="#c9c8c0" if not is11 else "#9a998f",
               zorder=1)
    a.plot([], [], "o", color="#c9c8c0", ms=5, label="fixed grid actions, modes 0–10")
    a.plot([], [], "o", color="#9a998f", ms=5, label="fixed grid actions, mode 11")
    for p, m in pv["policies"].items():
        g = ("HIST" if p.startswith("HIST") else "C0" if p.startswith("C0") else "NEW" if p.startswith("NEW") else
             "RULE" if p == "RULE_MCS3" else "S2" if p == "RULE_S2" else None)
        if g is None:
            continue
        a.plot(m["vehicle"]["retention"]["seg"], m["P"], "D" if g in ("HIST", "RULE", "S2") else "o", ms=9, color=COL[g],
               markeredgecolor="white", mew=1.5, zorder=3)
    for g, lab in (("HIST", "deployed seed 43"), ("C0", "C0 seeds"), ("NEW", "NEW seeds (corrected training)"),
                   ("RULE", "S1 MCS rule, all six registered tiers (training-chosen)"),
                   ("S2", "S2 MCS rule, deployed-parity person precision (training-chosen)")):
        a.plot([], [], "D" if g in ("HIST", "RULE", "S2") else "o", color=COL[g], ms=8, label=lab)
    bound = pv["rule_class_bound"]
    orc = pv["hindsight_oracle"]
    if bound.get("feasible"):
        a.plot(bound["retention_train"]["seg_vehicle"], bound["P_train"], "*", ms=15, mfc="none", mew=1.6, color=INK, zorder=4,
               label="PRIVILEGED: best MCS rule on this seed, all six tiers")
    pb = pv.get("rule_class_bound_deployed_parity", {})
    if pb.get("feasible"):
        a.plot(pb["retention_train"]["seg_vehicle"], pb["P_train"], "*", ms=15, mfc="none", mew=1.6, color=COL["S2"], zorder=4,
               label="PRIVILEGED: best MCS rule on this seed, deployed-parity quality")
    if orc:
        a.plot(orc["vehicle"]["retention"]["seg"], orc["P"], "P", ms=12, mfc="none", mew=1.6, color=INK, zorder=4,
               label="PRIVILEGED: per-decision hindsight oracle")
    a.axvline(0.85, color=INK, lw=1, ls="--")
    a.text(0.852, a.get_ylim()[0] + 0.005, "registered tier 0.85", fontsize=8, color=INK)
    a.set_xlabel("vehicle segmentation retention (all-scored, vs reference cell)", color=INK)
    a.set_ylabel("P: timely within-tolerance localisation", color=INK)
    a.set_title("E1-CORRECTED, eval seed 9017 (6,000 decisions, identical streams)", color=INK, fontsize=10, loc="left")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")
    _style(a)
    t = out["tables"]["E1|CORRECTED"]["metrics"]
    groups = [("deployed", ["HIST_seed43"], COL["HIST"]), ("C0 (3 seeds)", [f"C0_seed{s}" for s in (17, 29, 43)], COL["C0"]),
              ("NEW (3 seeds)", [f"NEW_seed{s}" for s in (17, 29, 43)], COL["NEW"]), ("S1 rule (all six tiers)", ["RULE_MCS3"], COL["RULE"]),
              ("S2 rule (deployed parity)", ["RULE_S2"], COL["S2"])]
    strata = ("mcs_9_10", "mcs_11_15", "mcs_ge16", "all")
    w = 0.16
    for j, (lab, ps, c) in enumerate(groups):
        vals = [[t[p][st]["P"] for p in ps] for st in strata]
        x = np.arange(len(strata)) + (j - 2) * w
        b.bar(x, [np.mean(v) for v in vals], width=w * 0.9, color=c, label=lab)
        if len(ps) > 1:
            b.errorbar(x, [np.mean(v) for v in vals], yerr=[[np.mean(v) - min(v) for v in vals], [max(v) - np.mean(v) for v in vals]],
                       fmt="none", ecolor=INK, lw=1, capsize=2)
    b.set_xticks(range(len(strata)), ["prior MCS 9–10", "MCS 11–15", "MCS ≥ 16", "all"])
    b.set_ylim(0.25, 0.6)
    b.set_ylabel("P (seed range shown for C0 / NEW)", color=INK)
    b.set_title("Held-out E1-CORRECTED, 3 eval seeds x 6,000 decisions", color=INK, fontsize=10, loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    _style(b)
    fig.text(0.01, 0.005, "Modelled (corrected latency environment + grid quality), not live. Vehicle segmentation is the binding "
             "registered requirement for NEW/C0; person precision 0.90 is attainable only with large payloads (S1 / stars).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(root / "fig_tradeoff.png", dpi=130)
    plt.close(fig)
