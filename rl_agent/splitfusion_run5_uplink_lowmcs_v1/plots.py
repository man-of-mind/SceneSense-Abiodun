"""Two figures: old-versus-candidate calibration (held-out FADE) and the action trade-off panel.

    python -m rl_agent.splitfusion_run5_uplink_lowmcs_v1.plots --run <dir>     # writes <dir>/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data as DT
from . import fit as FT
from . import model as M

INK, MUTED, GRID = "#2b2b2b", "#77766f", "#e6e5e0"
OLD, CAND = "#eb6834", "#2a78d6"            # categorical slots 2 / 1 (validated pair)
LABEL = {"old": "old (binding_02 + T_v2b)", "candidate": "candidate (+ low-MCS term)"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def calibration(run: Path, out: Path):
    ev = json.loads((run / "eval_01/EVALUATION.json").read_text())
    s = ev["summary"]["mcs_le15_B0"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    x = np.arange(3)
    names = ("T1", "T2", "T3")
    meas = [s[t]["measured_on_time"] for t in names]
    lo = [s[t]["measured_on_time"] - s[t]["wilson95"][0] for t in names]
    hi = [s[t]["wilson95"][1] - s[t]["measured_on_time"] for t in names]
    a.errorbar(x, meas, yerr=[lo, hi], fmt="s", ms=9, color=INK, capsize=5, lw=2, label="measured (Wilson 95%)")
    for off, m, c in ((-0.18, "old", OLD), (0.18, "candidate", CAND)):
        a.plot(x + off, [s[t][m]["own_chain_p_timely"] for t in names], "o", ms=9, color=c,
               markeredgecolor="white", markeredgewidth=2, label=f"{LABEL[m]}: own-chain")
        a.plot(x + off, [s[t][m]["env_idle_p_timely"] for t in names], "o", ms=9, mfc="none", mew=2, color=c,
               label=f"{LABEL[m]}: env (5 gc_v2 chains)")
    a.set_xticks(x, [f"{t}\n{s[t]['payload_kb'][0]:.0f}–{s[t]['payload_kb'][2]:.0f} KB\nn={s[t]['n']}" for t in names])
    a.set_ylim(0, 1.02)
    a.set_ylabel("feedback on time (receipt ≤ 170 ms)", color=INK)
    a.set_title("Held-out FADE session, prior MCS ≤ 15, zero backlog", color=INK, loc="left", fontsize=11)
    a.legend(fontsize=8, frameon=False, loc="lower left")
    _style(a)
    # transport at MCS 9-10, zero backlog: measured points (FIT grey, held-out black) and the two mean curves at MCS 9
    transport = FT.load_transport()
    rows, _ = DT.load_all(transport)
    corr = M.LowMcsCorrectionV1.load(run / "fit_01/CANDIDATE.json")
    for part, mk, col, lab in (("FIT", "o", "#b9b8b0", "FIT sessions (7)"), ("VALIDATION", "o", INK, "held-out FADE")):
        g = [r for r in rows if r["partition"] == part and r["mcs"] <= 10 and r["backlog"] == 0]
        b.plot([r["payload"] / 1e3 for r in g], [r["T"] for r in g], mk, ms=4.5 if part == "FIT" else 6, color=col,
               alpha=0.9, label=f"{lab}, MCS 9–10 (n={len(g)})", zorder=2 if part == "FIT" else 3)
    w = np.linspace(20e3, 165e3, 60)
    beta = ev["beta_pool_ms"]
    mu = np.array([transport.predict(pre_enqueue_backlog_bytes=0.0, wire_bytes=int(v), prior_ul_mcs=9).conditional_latency_ms for v in w])
    term = np.array([corr.term_ms(backlog_bytes=0.0, wire_bytes=int(v), prior_ul_mcs=9) for v in w])
    b.plot(w / 1e3, mu + beta, color=OLD, lw=2, label=f"{LABEL['old']} mean, MCS 9")
    b.plot(w / 1e3, mu + term + beta, color=CAND, lw=2, label=f"{LABEL['candidate']} mean, MCS 9")
    b.set_xlabel("L2 payload (KB)", color=INK)
    b.set_ylabel("uplink: last send → complete reassembly (ms)", color=INK)
    b.set_title("Transport target at low MCS (mean curves include the pool offset β)", color=INK, loc="left", fontsize=11)
    b.set_ylim(20, 130)
    b.legend(fontsize=8, frameon=False, loc="upper left")
    _style(b)
    fig.tight_layout()
    fig.savefig(out / "fig1_calibration_old_vs_candidate.png", dpi=130)
    plt.close(fig)


def tradeoff(run: Path, out: Path):
    pn = json.loads((run / "panel_01/PANEL.json").read_text())
    picks = (("138", "dec 138: MCS 9, zero backlog"), ("302", "dec 302: MCS 9, zero backlog"),
             ("294", "dec 294: MCS 13, zero backlog"), ("144", "dec 144: MCS 21 (control)"))
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True)
    for col, (d, title) in enumerate(picks):
        r = pn["results"][d]
        for m, c in (("old", OLD), ("candidate", CAND)):
            acts = r["models"][m]["actions"]
            grid = sorted((a["wire_kb_p50"], a) for n, a in acts.items() if n != "RECORDED")
            xs = [g[0] for g in grid]
            axes[0, col].plot(xs, [g[1]["shifts"]["0.0"]["p_timely"] for g in grid], "-o", color=c, lw=2, ms=6,
                              markeredgecolor="white", label=LABEL[m])
            axes[1, col].plot(xs, [g[1]["shifts"]["0.0"]["e_reward"] for g in grid], "-o", color=c, lw=2, ms=6,
                              markeredgecolor="white")
            rec = acts["RECORDED"]
            for row, key in ((0, "p_timely"), (1, "e_reward")):
                axes[row, col].plot(rec["wire_kb_p50"], rec["shifts"]["0.0"][key], "D", ms=10, mfc="none", mew=2, color=c)
            best = r["models"][m]["best_by_shift"]["0.0"]
            same = best == r["models"]["old"]["best_by_shift"]["0.0"] == r["models"]["candidate"]["best_by_shift"]["0.0"]
            if not (same and m == "old"):
                axes[1, col].annotate("best (both)" if same else f"best ({m})",
                                      (acts[best]["wire_kb_p50"], acts[best]["shifts"]["0.0"]["e_reward"]),
                                      textcoords="offset points", xytext=(0, 8 if m == "candidate" else -14), ha="center",
                                      fontsize=8, color=INK)
        axes[0, col].set_title(title, color=INK, fontsize=10, loc="left")
        axes[0, col].set_ylim(-0.03, 1.05)
        axes[1, col].set_xlabel("wire payload (KB), mode 11, q 9400 → 4000", color=INK)
        for row in (0, 1):
            _style(axes[row, col])
    axes[0, 0].set_ylabel("P(feedback ≤ 170 ms)", color=INK)
    axes[1, 0].set_ylabel("E[reward] = P·Y(0.7+0.3Q_seg)\n(quality: nearest FIT scenes, approx.)", color=INK)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.text(0.01, 0.005, "Diamond = the deployed actor's recorded action. Same TRAIN_GCV2 pool and common random numbers "
             "for both models; score channel idle. Backlog decisions (120/130/299) are unchanged by design.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig2_action_tradeoff_old_vs_candidate.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    run = Path(p.parse_args().run)
    out = run / "figures"
    out.mkdir(exist_ok=False)
    calibration(run, out)
    tradeoff(run, out)
