"""E4 figure: cooperative coverage and localization error vs two-ego baseline.

Palette = dataviz reference categorical slots 1-4, validated (light mode, adjacent
pairs): worst CVD dE 9.1 (target 8), worst normal-vision dE 22.9 (floor 15).
Slots 3/4 fall below 3:1 contrast -> relief rule applied: every series is direct-labeled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from e4_cooperative_gain import coverage_experiment, load_scene_frames, localization_experiment

OUT = Path(__file__).parent / "results"

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"
LIVE_TRIANG_M = 1.40   # measured live, cooperative_fusion/RESULTS_phase2_two_view.md


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=MUTED, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--depth-std", type=float, default=3.5,
                    help="calibrated to the measured live monocular error (3.56 m)")
    args = ap.parse_args()

    baselines = [4, 8, 14, 20]
    frames = load_scene_frames(args.frames, 3, 40.0)
    cov = {b: coverage_experiment(frames, b, 120.0, 40.0)[0] for b in baselines}
    loc = {b: localization_experiment(frames, b, 120.0, 40.0, args.depth_std, 0.3) for b in baselines}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    fig.patch.set_facecolor(SURFACE)

    # ---------------- panel 1: coverage ----------------
    pct = lambda b, k: 100 * cov[b][k] / cov[b]["objects"]
    series1 = [
        ("Cooperative (either ego)", [pct(b, "union") for b in baselines], S1),
        ("Ego A alone", [pct(b, "A") for b in baselines], S2),
        ("Ego B alone", [pct(b, "B") for b in baselines], S3),
        ("Seen by both (triangulable)", [pct(b, "both") for b in baselines], S4),
    ]
    for label, ys, c in series1:
        ax1.plot(baselines, ys, color=c, linewidth=2.0, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax1.annotate(f"{ys[-1]:.0f}%", (baselines[-1], ys[-1]), textcoords="offset points",
                     xytext=(8, 0), color=INK, fontsize=9, va="center", fontweight="bold")
    style(ax1)
    ax1.set_xlabel("Two-ego baseline (m)", color=INK2, fontsize=10)
    ax1.set_ylabel("Objects localizable (% of in-range GT)", color=INK2, fontsize=10)
    ax1.set_title("Map coverage: two egos see more of the scene", color=INK,
                  fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax1.set_xticks(baselines)
    ax1.set_ylim(38, 100)
    ax1.set_xlim(2.5, 23)
    # sit the legend below the lowest series (52% at 20 m) so it never crosses a line
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.02, 0.02),
               labelcolor=INK2, ncol=2, columnspacing=1.2)

    # ---------------- panel 2: localization ----------------
    series2 = [
        ("Two-view triangulation", [loc[b]["triangulate"]["mae"] for b in baselines], S1),
        ("Single ego A (monocular)", [loc[b]["single_A"]["mae"] for b in baselines], S2),
        ("Two-view mean", [loc[b]["mean_2view"]["mae"] for b in baselines], S3),
    ]
    for label, ys, c in series2:
        ax2.plot(baselines, ys, color=c, linewidth=2.0, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax2.annotate(f"{ys[-1]:.2f} m", (baselines[-1], ys[-1]), textcoords="offset points",
                     xytext=(8, 0), color=INK, fontsize=9, va="center", fontweight="bold")
    ax2.axhline(LIVE_TRIANG_M, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax2.annotate(f"live measured triangulation  {LIVE_TRIANG_M:.2f} m", (4.3, LIVE_TRIANG_M),
                 textcoords="offset points", xytext=(0, 5), color=INK2, fontsize=8.5)
    style(ax2)
    ax2.set_xlabel("Two-ego baseline (m)", color=INK2, fontsize=10)
    ax2.set_ylabel("Localization MAE (m)", color=INK2, fontsize=10)
    ax2.set_title(f"Localization: triangulation needs baseline $\\geq$8 m",
                  color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=10)
    ax2.set_xticks(baselines)
    ax2.set_xlim(2.5, 23.5)
    # keep the legend clear of the right-hand direct labels
    # lower-left is the only region clear of all three series and the 1.40 m reference line
    ax2.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.02, 0.02),
               labelcolor=INK2)

    fig.text(0.005, 0.030,
             f"Real CARLA GT object layouts + real ego-A poses ({len(frames)} test-split frames, "
             f"40 m gate, 120° FOV); ego B synthesized at the stated offset.",
             fontsize=7.4, color=MUTED)
    fig.text(0.005, 0.008,
             f"Monocular depth noise σ={args.depth_std} m (calibrated to the 3.56 m measured live "
             f"single-view error); bearing noise σ=0.3°. Triangulation is bearing-only, so it is "
             f"independent of depth noise.",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.065, 0.995, 1))
    p = OUT / "E4_coverage_localization.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
