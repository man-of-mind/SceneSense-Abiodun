#!/usr/bin/env python3
"""Knob -> (accuracy, payload) tradeoff plot for the per-model sweep. Parses PERMODEL_KNOB_MATRIX.md.
Two panels: mIoU-vs-payload (segmentation HAS a payload cost) and ped-recall-vs-payload (detection ~flat).
Color = model, marker = ROI level (filled=0 / half=0.3 / open=0.5) with an explicit legend, Pareto frontier
drawn per panel so 'for target payload P, best achievable accuracy' is readable. Replaces the old scatter."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, re, sys
from pathlib import Path

SRC = Path("rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md")  # accuracy/payload codec-invariant; latency=zstd
OUT = Path("rl_agent/plots"); OUT.mkdir(exist_ok=True)
MODEL_C = {"128": "#0072B2", "64": "#E69F00", "32": "#009E73", "-": "#D55E00"}
MODEL_L = {"128": "AE-128", "64": "AE-64", "32": "AE-32", "-": "no-AE"}
# marker fill by ROI: 0 solid, 0.3 half (top), 0.5 open
ROI_FILL = {0.0: "full", 0.3: "top", 0.5: "none"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})

def parse():
    pts = []  # (model, quant, roi, payKB, miou, ped)
    for ln in SRC.read_text().splitlines():
        if not ln.strip().startswith("|"): continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 12 or c[0].lower() == "profile" or set(c[0]) <= set("-: "): continue
        quant, roi, ae, pay = c[1], c[3], c[4], c[5]
        if quant not in ("per_channel_uint4","per_channel_uint6","per_channel_uint8"): continue
        try: pay_v, miou, ped = float(pay), float(c[7]), float(c[9])
        except: continue
        pts.append((ae, quant, float(roi), pay_v, miou, ped))
    return pts

def frontier(xy):  # non-dominated: min payload for >= accuracy; returns sorted (x,y) upper-left envelope
    s = sorted(xy)  # by payload asc
    best = -1; out = []
    for x, y in s:
        if y > best: out.append((x, y)); best = y
    return out

def panel(ax, pts, yi, ylabel, ymin):
    for (ae, q, roi, pay, miou, ped) in pts:
        y = miou if yi == "miou" else ped
        col = MODEL_C[ae]; fs = ROI_FILL.get(roi, "full")
        kw = dict(markersize=9, markeredgecolor=col, markeredgewidth=1.3, linestyle="none")
        if fs == "full": kw.update(marker="o", markerfacecolor=col)
        elif fs == "top": kw.update(marker="o", fillstyle="top", markerfacecolor=col, markerfacecoloralt="white")
        else: kw.update(marker="o", markerfacecolor="white")
        ax.plot([pay], [y], **kw, zorder=3)
    # pareto frontier over ALL points
    fr = frontier([(p, (m if yi=="miou" else pd)) for (_,_,_,p,m,pd) in pts])
    ax.plot([x for x,_ in fr], [y for _,y in fr], color="#444444", lw=1.2, ls="--", zorder=2, label="Pareto frontier")
    ax.set_xscale("log"); ax.set_ylabel(ylabel); ax.set_ylim(ymin, None)
    ax.set_xlabel("payload KB/frame (log)")

pts = parse()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
panel(ax1, pts, "miou", "mIoU (segmentation, ↑)", 0.58)
panel(ax2, pts, "ped", "pedestrian recall (detection, ↑)", 0.80)
ax1.set_title("Segmentation vs payload — real cost", fontsize=12, fontweight="bold")
ax2.set_title("Detection vs payload — stays ~flat", fontsize=12, fontweight="bold")
# annotate a couple frontier points on the mIoU panel (payload labels)
for (ae,q,roi,pay,miou,ped) in pts:
    if (ae,q,roi) in [("128","per_channel_uint4",0.0),("128","per_channel_uint8",0.0),("-","per_channel_uint8",0.0)]:
        ax1.annotate(f"{pay:.0f}KB", (pay, miou), xytext=(0,7), textcoords="offset points", ha="center", fontsize=8)

# legends: model color + ROI marker
from matplotlib.lines import Line2D
mh = [Line2D([0],[0], marker="o", color="none", markerfacecolor=MODEL_C[k], markeredgecolor=MODEL_C[k], markersize=9, label=MODEL_L[k]) for k in ("128","64","32","-")]
rh = [Line2D([0],[0], marker="o", color="none", markerfacecolor="#666", markeredgecolor="#666", markersize=9, label="ROI 0"),
      Line2D([0],[0], marker="o", color="none", markerfacecolor="#666", fillstyle="top", markeredgecolor="#666", markersize=9, label="ROI 0.3"),
      Line2D([0],[0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#666", markersize=9, label="ROI 0.5"),
      Line2D([0],[0], color="#444", ls="--", lw=1.2, label="Pareto frontier")]
ax1.legend(handles=mh, loc="lower right", fontsize=9, frameon=False, title="model")
ax2.legend(handles=rh, loc="lower left", fontsize=9, frameon=False, title="ROI drop / frontier")
fig.suptitle("Knob → accuracy vs payload: pick a payload, read the accuracy cost (upper-left = best)",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT/"permodel_tradeoff.pdf", format="pdf", bbox_inches="tight")
fig.savefig(OUT/"permodel_tradeoff.png", format="png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/permodel_tradeoff.pdf/.png  ({len(pts)} points)")
