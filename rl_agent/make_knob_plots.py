#!/usr/bin/env python3
"""Clean PDF bar charts of the static-knob results for the team presentation.
Each figure: top = task-utility grouped bars (mIoU, obj recall, loc error), bottom = payload.
Colorblind-safe (Okabe-Ito). Direct value labels. Utility metrics: mIoU/recall higher=better,
loc error lower=better (labelled). Payload: entropy-coded KB/frame."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "plots"; OUT.mkdir(exist_ok=True)
# Okabe-Ito colorblind-safe palette
C_MIOU, C_REC, C_LOC, C_PAY = "#0072B2", "#E69F00", "#009E73", "#999999"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})


def chart(fname, title, xlabels, miou, objr, loc, payload, xaxis_title):
    n = len(xlabels)
    x = np.arange(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(7, 1.7 * n + 2.5), 6.6),
                                   gridspec_kw={"height_ratios": [2.1, 1]})
    # ---- top: utility grouped bars ----
    w = 0.26
    b1 = ax1.bar(x - w, miou, w, label="mIoU (↑ better)", color=C_MIOU)
    b2 = ax1.bar(x,      objr, w, label="obj recall (↑ better)", color=C_REC)
    b3 = ax1.bar(x + w,  loc,  w, label="loc error, m (↓ better)", color=C_LOC)
    for bars, fmt in ((b1, "{:.3f}"), (b2, "{:.3f}"), (b3, "{:.2f}")):
        for r in bars:
            ax1.annotate(fmt.format(r.get_height()), (r.get_x() + r.get_width() / 2, r.get_height()),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylabel("metric value")
    ax1.set_ylim(0, max(max(miou), max(objr), max(loc)) * 1.22)
    ax1.set_xticks(x); ax1.set_xticklabels(xlabels)
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=9, ncol=3, frameon=False)
    # ---- bottom: payload ----
    bp = ax2.bar(x, payload, 0.5, color=C_PAY)
    for r in bp:
        ax2.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("payload KB/frame")
    ax2.set_ylim(0, max(payload) * 1.2)
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels)
    ax2.set_xlabel(xaxis_title)
    fig.tight_layout()
    fig.savefig(OUT / fname, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / fname}")


# 1) ENTROPY — utility flat, payload varies (u8, ROI 0, no AE)
chart("knob_entropy.pdf",
      "Entropy coding: task utility unchanged, payload reduced\n(per-channel uint8, no ROI, no AE)",
      ["zlib", "zstd", "none"],
      [0.841, 0.841, 0.841], [0.837, 0.837, 0.837], [1.21, 1.21, 1.21],
      [992.2, 992.8, 1425.3], "entropy coder")

# 2) QUANTIZATION (zlib, ROI 0, no AE)
chart("knob_quant.pdf",
      "Quantization: payload vs task utility\n(zlib, no ROI, no AE)",
      ["uint8", "uint6", "uint4"],
      [0.841, 0.841, 0.840], [0.837, 0.837, 0.817], [1.21, 1.22, 1.32],
      [992.2, 730.0, 358.6], "quantization bit-depth")

# 3) ROI drop (u8, zlib, no AE)
chart("knob_roi.pdf",
      "ROI drop: trades segmentation for payload, detection preserved\n(uint8, zlib, no AE)",
      ["ROI 0.1", "ROI 0.3", "ROI 0.5", "ROI 0.7"],
      [0.833, 0.811, 0.797, 0.779], [0.836, 0.837, 0.832, 0.831], [1.21, 1.21, 1.30, 1.30],
      [898.5, 714.4, 556.6, 380.6], "ROI drop fraction q")
