"""E6 figure: sustained FPS vs on-vehicle compute budget, full-local vs split-front.

Palette: dataviz reference slots 1-2, validated all-pairs (light): CVD dE 24.7,
normal-vision 33.6, both >=3:1 contrast on the surface.

The 24-thread point is EXCLUDED from the curves: pinning 24 torch threads to all 24
cores leaves no core for the OS and produces p95 = 1945 ms against p50 = 30 ms, i.e. a
scheduling pathology rather than a compute measurement. It is annotated, not hidden.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"
DEADLINES = [(10, "10 FPS"), (20, "20 FPS"), (30, "30 FPS")]


def main():
    rows = [r for r in csv.DictReader(open(OUT / "E6_raw.csv"))]
    valid = [r for r in rows if int(r["threads"]) <= 16]
    thr = [int(r["threads"]) for r in valid]
    full = [float(r["full_fps"]) for r in valid]
    front = [float(r["front_fps"]) for r in valid]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    meta = json.loads((OUT / "E6_raw.json").read_text())
    # Shade ONLY the 30 FPS split-only band - the strongest and unambiguous claim.
    # (Unioning all three deadlines' bands covers the whole x-range and says nothing.)
    b30 = [t for t in meta["crossover"]["30"]["crossover_band_threads"] if t <= 16]
    if b30:
        ax.axvspan(min(b30) * 0.88, max(b30) * 1.14, color=S1, alpha=0.07, zorder=0)

    for y, lbl in DEADLINES:
        ax.axhline(y, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.annotate(lbl, (0.90, y), xytext=(0, 3), textcoords="offset points",
                    fontsize=8.5, color=INK2, va="bottom", ha="right")

    for xs, ys, c, lbl in ((thr, front, S1, "SPLIT-front (backbone on car)"),
                           (thr, full, S2, "FULL-local (backbone + heads)")):
        ax.plot(xs, ys, color=c, lw=2.2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.6, label=lbl, zorder=4)
        ax.annotate(f"{ys[-1]:.1f} FPS", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9, fontweight="bold", color=INK)

    ax.set_xscale("log", base=2)
    ax.set_xticks(thr)
    ax.set_xticklabels([str(t) for t in thr])
    ax.set_xlim(0.72, 20.5)
    ax.set_ylim(0, 72)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=MUTED, alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=MUTED)
    ax.set_xlabel("On-vehicle compute budget (pinned CPU cores / intra-op threads)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("Sustained throughput (FPS)", color=INK2, fontsize=10)
    ax.set_title("Full-local misses real-time deadlines that split-front meets",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.annotate("full-local never reaches 30 FPS\non any budget tested;\nsplit-front clears it from 4 cores",
                (4.6, 40.5), fontsize=9, color=INK2, va="center")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK2)

    fig.text(0.005, 0.030,
             "Real 1x7x432x768 input; 20 s sustained window per point; process pinned with "
             "sched_setaffinity. Host: Core Ultra 9 285K (cores 0-7 P @5.0-5.1 GHz, 8-23 E @4.6 GHz),",
             fontsize=7.4, color=MUTED)
    fig.text(0.005, 0.008,
             "so points past 8 add slower E-cores. 24-thread point excluded: full-core "
             "oversubscription gives p95 1945 ms vs p50 30 ms (scheduling stall, not compute).",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    p = OUT / "E6_compute_crossover.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
