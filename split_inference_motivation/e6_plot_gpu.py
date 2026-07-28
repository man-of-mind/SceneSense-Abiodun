"""E6 GPU arm figure: FPS vs locked GPU clock, and the split speedup trend across BOTH arms.

Panel A shows the honest negative result: clock-limiting an RTX 5090 never produces a
crossover - full-local clears 30 FPS at every clock down to 210 MHz.
Panel B shows what the GPU arm DOES establish: the split speedup grows as the compute
budget shrinks, in both arms, with no P-core/E-core confound in the GPU one.

Palette: dataviz reference slots 1-2, validated all-pairs (light): CVD dE 24.7, normal 33.6.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=MUTED, alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=MUTED)


def main():
    g = list(csv.DictReader(open(OUT / "E6_gpu_raw.csv")))
    mhz = [float(r["actual_mhz"]) for r in g]
    gfull = [float(r["full_fps"]) for r in g]
    gfront = [float(r["front_fps"]) for r in g]
    gspeed = [f / u for f, u in zip(gfront, gfull)]

    c = [r for r in csv.DictReader(open(OUT / "E6_raw.csv")) if int(r["threads"]) <= 16]
    thr = [int(r["threads"]) for r in c]
    cspeed = [float(r["front_fps"]) / float(r["full_fps"]) for r in c]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: GPU FPS vs clock ----
    for y, col, lbl in ((gfront, S1, "SPLIT-front"), (gfull, S2, "FULL-local")):
        ax1.plot(mhz, y, color=col, lw=2.2, marker="o", markersize=6.5,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label=lbl, zorder=4)
    ax1.axhline(30, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax1.annotate("30 FPS deadline", (215, 31.5), fontsize=8.5, color=INK2, va="bottom")
    # top-left is the only region clear of both curves (they run bottom-left -> top-right)
    ax1.annotate(f"full-local still {gfull[-1]:.0f} FPS at the\nminimum clock — no crossover\nreachable by clock-limiting",
                 (222, 430), fontsize=9, color=INK2, va="center")
    style(ax1)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Locked GPU clock (MHz, log)", color=INK2, fontsize=10)
    ax1.set_ylabel("Sustained throughput (FPS, log)", color=INK2, fontsize=10)
    ax1.set_title("GPU arm: clock-limiting never forces a miss", color=INK,
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax1.set_ylim(24, 1150)
    ax1.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.10),
               labelcolor=INK2)

    # ---- Panel B: speedup vs budget, both arms (normalised x = fraction of max budget) ----
    ax2.plot([m / max(mhz) for m in mhz], gspeed, color=S1, lw=2.2, marker="o", markersize=6.5,
             markeredgecolor=SURFACE, markeredgewidth=1.5, label="GPU arm (clock-limited)", zorder=4)
    ax2.plot([t / max(thr) for t in thr], cspeed, color=S2, lw=2.2, marker="s", markersize=6.5,
             markeredgecolor=SURFACE, markeredgewidth=1.5, label="CPU arm (core-limited)", zorder=4)
    style(ax2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Compute budget (fraction of this device's maximum, log)", color=INK2, fontsize=10)
    ax2.set_ylabel("Split speedup (FRONT FPS / FULL FPS)", color=INK2, fontsize=10)
    ax2.set_title("Both arms: split gains more as the budget shrinks", color=INK,
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)
    ax2.set_ylim(1.2, 3.1)

    fig.text(0.005, 0.012,
             "GPU: RTX 5090, clocks locked with nvidia-smi -lgc (actual achieved clock recorded per point); "
             "15 s sustained per config. CPU arm from E6_raw.csv (24-thread point excluded).",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    p = OUT / "E6_gpu_arm.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
