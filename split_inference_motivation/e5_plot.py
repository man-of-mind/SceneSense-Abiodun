"""E5 figure: attacker reconstruction quality vs transmitted payload.

The point of the chart: payload spans ~22x across accuracy-preserving profiles, but
attack quality barely moves - EXCEPT for the ROI-drop profile. Log-x because payload
spans an order of magnitude.

Palette: dataviz reference slots 1-2, validated (light, all-pairs): CVD dE 24.7/31.7,
normal-vision 33.6 - well clear of both floors.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
SURFACE = "#fcfcfb"


def main():
    pts = []
    base = json.loads((OUT / "E5_raw_u8.json").read_text())
    floor = base["floor_mean_image"]
    ceil = base["ceiling_architecture_B_jpeg92"]
    pts.append(("noae u8\n(deployed)", 1050.3, base["attack"]["ssim"], False, True))
    try:
        fp32 = json.loads((OUT / "E5_raw_fp32.json").read_text())
        pts.append(("noae fp32\n(control)", 2835.0, fp32["attack"]["ssim"], False, True))
    except FileNotFoundError:
        pass
    for f in sorted(glob.glob(str(OUT / "E5_profile_*.json"))):
        if "temporal" in f:
            continue
        d = json.loads(Path(f).read_text())
        if d["profile"] == "noae__uint8__roi0.0":
            continue
        label = d["profile"].replace("__", "\n")
        pts.append((label, d["payload_kb"], d["attack"]["ssim"], d["roi"] > 0, d["accept"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhspan(0, floor["ssim"], color=MUTED, alpha=0.10, zorder=0)
    ax.axhline(floor["ssim"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no-information floor (mean image)  SSIM {floor['ssim']:.3f}",
                (4000, floor["ssim"]), xytext=(0, 6), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)
    ax.axhline(ceil["ssim"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"architecture B ships the image itself  SSIM {ceil['ssim']:.3f}",
                (4000, ceil["ssim"]), xytext=(0, 6), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK2)

    for lbl, kb, ssim, is_roi, accept in pts:
        c = S2 if is_roi else S1
        ax.plot([kb], [ssim], marker="o", markersize=11, color=c,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4, linestyle="none")
        # anchor the text block's edge (va) rather than its centre, so multi-line
        # labels never grow back over their own marker
        if is_roi:
            ax.annotate(f"{lbl}\n{ssim:.3f}", (kb, ssim), xytext=(0, -13),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8.2, color=INK)
        else:
            ax.annotate(f"{lbl}\n{ssim:.3f}", (kb, ssim), xytext=(0, 13),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.2, color=INK)

    nonroi = sorted([(k, s) for _, k, s, r, _ in pts if not r])
    ax.plot([p[0] for p in nonroi], [p[1] for p in nonroi], color=S1, lw=1.6, alpha=0.55, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(90, 4200)
    ax.set_ylim(0, 1.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=MUTED, alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=MUTED)
    ax.set_xlabel("Transmitted payload per frame (KB, log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("Attacker reconstruction SSIM", color=INK2, fontsize=10)
    ax.set_title("Compressing the split payload does not make it private",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=12)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=S1, markersize=9,
               markeredgecolor=SURFACE, label="quantization / AE bottleneck only"),
        Line2D([], [], marker="o", ls="none", color=S2, markersize=9,
               markeredgecolor=SURFACE, label="with ROI drop (q=0.3)")],
        frameon=False, fontsize=9, loc="center right", labelcolor=INK2)

    fig.text(0.005, 0.012,
             "All profiles hold model accuracy (see E5 table). Attacker: 0.93M-param decoder, identical "
             "15-min budget per profile, trained on the manifest split.",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = OUT / "E5_payload_vs_privacy.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
