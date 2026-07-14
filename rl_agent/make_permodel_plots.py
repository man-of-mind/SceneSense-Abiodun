#!/usr/bin/env python3
"""Presentation plots for the PER-MODEL sweep (4 comparable models: no-AE + AE-32/64/128).
3 figures, Okabe-Ito colorblind-safe, direct value labels, single value axis (no dual-axis):
  1) payload_collapse  -- accuracy held while payload crashes (raw -> no-AE -> AE, matched u4/ROI0)
  2) accuracy_vs_payload -- Pareto frontier scatter, colored by model (AE dominates low-payload)
  3) roi_uint4_tradeoff -- at uint4, ROI drop sacrifices segmentation but preserves detection/loc
Values are the verified per-model sweep results (rl_agent/PERMODEL_KNOB_MATRIX.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "plots"; OUT.mkdir(exist_ok=True)
# Okabe-Ito
C_MIOU, C_REC, C_LOC, C_PAY = "#0072B2", "#E69F00", "#009E73", "#999999"
C_VEH = "#CC79A7"
MODEL_C = {"AE-128": "#0072B2", "AE-64": "#E69F00", "AE-32": "#009E73", "no-AE": "#D55E00"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


# ============ PLOT 1: payload collapses, accuracy holds ============
# matched operating points (ROI 0; models at uint4; raw = uncompressed fp16 reference)
p1_lbl = ["Raw\nfp16", "no-AE\nu4·ROI0", "AE-128\nu4·ROI0", "AE-64\nu4·ROI0", "AE-32\nu4·ROI0"]
p1_pay = [2835.0, 387.6, 127.4, 99.3, 87.5]
p1_miou = [0.839, 0.838, 0.819, 0.824, 0.822]
p1_ped = [0.883, 0.843, 0.887, 0.862, 0.860]

def plot1():
    x = np.arange(len(p1_lbl)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 6.8), gridspec_kw={"height_ratios": [1.5, 1]})
    b1 = ax1.bar(x - w/2, p1_miou, w, label="mIoU (↑ better)", color=C_MIOU)
    b2 = ax1.bar(x + w/2, p1_ped, w, label="ped recall (↑ better)", color=C_REC)
    for bars in (b1, b2):
        for r in bars:
            ax1.annotate(f"{r.get_height():.3f}", (r.get_x()+r.get_width()/2, r.get_height()),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, 1.02); ax1.set_ylabel("accuracy")
    ax1.set_xticks(x); ax1.set_xticklabels(p1_lbl)
    ax1.set_title("Accuracy held while payload collapses\n(matched operating points: ROI 0)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="lower center", fontsize=9, ncol=2, frameon=False)
    bp = ax2.bar(x, p1_pay, 0.55, color=C_PAY)
    for r, red in zip(bp, [1.0, 7.3, 22.2, 28.5, 32.4]):
        ax2.annotate(f"{r.get_height():.0f} KB\n({red:.0f}×↓)" if red > 1 else f"{r.get_height():.0f} KB",
                     (r.get_x()+r.get_width()/2, r.get_height()), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylabel("payload KB/frame"); ax2.set_ylim(0, max(p1_pay)*1.22)
    ax2.set_xticks(x); ax2.set_xticklabels(p1_lbl)
    fig.tight_layout(); save(fig, "permodel_payload_collapse")


# ============ PLOT 2: accuracy-vs-payload frontier ============
# (payload KB, mIoU, ROI) per model -- all compressed points
PTS = {
 "AE-128": [(127.4,0.819,0),(95.0,0.684,0.3),(77.7,0.621,0.5),(257.8,0.819,0),(190.9,0.756,0.3),
            (153.2,0.714,0.5),(337.7,0.819,0),(250.3,0.759,0.3),(199.5,0.730,0.5)],
 "AE-64":  [(99.3,0.824,0),(73.9,0.739,0.3),(60.0,0.702,0.5),(200.2,0.825,0),(147.3,0.795,0.3),
            (116.9,0.781,0.5),(262.7,0.825,0),(193.2,0.805,0.3),(152.1,0.793,0.5)],
 "AE-32":  [(87.5,0.822,0),(62.9,0.715,0.3),(48.7,0.656,0.5),(172.7,0.822,0),(123.6,0.786,0.3),
            (94.0,0.759,0.5),(225.0,0.822,0),(161.5,0.799,0.3),(121.8,0.773,0.5)],
 "no-AE":  [(387.6,0.838,0),(285.0,0.719,0.3),(234.8,0.613,0.5),(783.3,0.840,0),(564.9,0.797,0.3),
            (449.0,0.774,0.5),(1052.9,0.840,0),(753.6,0.803,0.3),(586.6,0.782,0.5)],
}

def plot2():
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for m, pts in PTS.items():
        roi0 = [(p, q) for p, q, r in pts if r == 0]
        roiN = [(p, q) for p, q, r in pts if r > 0]
        ax.scatter([p for p, q in roi0], [q for p, q in roi0], s=90, color=MODEL_C[m],
                   edgecolor="white", linewidth=0.7, zorder=3, label=f"{m} (ROI 0)")
        ax.scatter([p for p, q in roiN], [q for p, q in roiN], s=55, facecolors="none",
                   edgecolors=MODEL_C[m], linewidth=1.3, zorder=2)
    ax.axhline(0.839, color="#444444", lw=1, ls="--", zorder=1)
    ax.annotate("clean mIoU 0.839 (no-AE)", (55, 0.842), fontsize=8.5, color="#444444")
    ax.annotate("Pareto pick\nAE-128 u4·ROI0\n127 KB · mIoU 0.819", (127.4, 0.819), xytext=(150, 0.70),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1))
    ax.set_xscale("log"); ax.set_xlabel("payload KB/frame (log)"); ax.set_ylabel("mIoU (↑ better)")
    ax.set_ylim(0.58, 0.87)
    ax.set_title("Accuracy vs payload — AE dominates the low-payload region\n"
                 "(filled = ROI 0; open = ROI 0.3/0.5; each model × quant u4/u6/u8)",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False, ncol=2)
    fig.tight_layout(); save(fig, "permodel_accuracy_vs_payload")


# ============ PLOT 3: ROI drop at uint4 — seg sacrificed, detection preserved (AE-128) ============
p3_lbl = ["ROI 0", "ROI 0.3", "ROI 0.5"]
p3_miou = [0.819, 0.684, 0.621]
p3_veh = [0.913, 0.536, 0.382]
p3_ped = [0.887, 0.886, 0.878]
p3_loc = [0.88, 0.88, 0.94]
p3_pay = [127.4, 95.0, 77.7]

def plot3():
    x = np.arange(len(p3_lbl)); w = 0.2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.0), gridspec_kw={"height_ratios": [1.7, 1]})
    series = [("mIoU / seg (↑)", p3_miou, C_MIOU, -1.5), ("vehicle IoU (↑)", p3_veh, C_VEH, -0.5),
              ("ped recall (↑)", p3_ped, C_REC, 0.5), ("loc error m (↓)", p3_loc, C_LOC, 1.5)]
    for name, vals, col, off in series:
        b = ax1.bar(x + off*w, vals, w, label=name, color=col)
        for r in b:
            ax1.annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax1.set_ylim(0, 1.08); ax1.set_ylabel("metric value")
    ax1.set_xticks(x); ax1.set_xticklabels([])   # ROI labels carried by the bottom (payload) panel
    ax1.set_title("ROI drop at uint4 (AE-128): segmentation sacrificed, detection & loc preserved",
                  fontsize=12.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper center", fontsize=9, ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    bp = ax2.bar(x, p3_pay, 0.5, color=C_PAY)
    for r in bp:
        ax2.annotate(f"{r.get_height():.0f} KB", (r.get_x()+r.get_width()/2, r.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("payload KB/frame"); ax2.set_ylim(0, max(p3_pay)*1.25)
    ax2.set_xticks(x); ax2.set_xticklabels(p3_lbl); ax2.set_xlabel("ROI drop fraction")
    fig.tight_layout(); save(fig, "permodel_roi_uint4_tradeoff")


plot1(); plot2(); plot3()
print("done")
