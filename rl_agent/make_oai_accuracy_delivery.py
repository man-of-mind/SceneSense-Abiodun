#!/usr/bin/env python3
"""OAI: accuracy and delivery are SEPARATE axes (honest framing -- no raw x delivery multiply).
Left  = per-frame accuracy (mIoU, ped-recall): set by the compression config, IDENTICAL loopback vs OAI for
        delivered frames -- the channel does not corrupt delivered outputs.
Right = delivery/availability (fresh result per tick): the network's real cost. no-AE baseline drops 25%.
Staleness (late-frame temporal misalignment) is the true latency->accuracy link and is NOT shown here -- a
separate future measurement. Okabe-Ito."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT = Path("rl_agent/plots"); OUT.mkdir(exist_ok=True)
C_MIOU, C_PED, C_DEL, C_LOOP = "#0072B2", "#E69F00", "#009E73", "#D55E00"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True})

lbl   = ["no-AE u8\n1141 KB", "AE-128 u8\n346 KB", "AE-128 u4\n142 KB"]
miou  = [0.840, 0.819, 0.819]
ped   = [0.855, 0.883, 0.887]
deliv = [75.0, 99.3, 99.0]
x = np.arange(len(lbl)); w = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.6))

# ---- Panel A: per-frame accuracy (config-determined, channel-invariant) ----
b1 = ax1.bar(x-w/2, miou, w, label="mIoU (segmentation)", color=C_MIOU)
b2 = ax1.bar(x+w/2, ped,  w, label="pedestrian recall (detection)", color=C_PED)
for bars in (b1, b2):
    for r in bars:
        ax1.annotate(f"{r.get_height():.3f}", (r.get_x()+r.get_width()/2, r.get_height()),
                     xytext=(0,2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
ax1.set_ylabel("accuracy (delivered frames)"); ax1.set_ylim(0.5, 1.02)
ax1.set_xticks(x); ax1.set_xticklabels(lbl, fontsize=9.5)
ax1.set_title("Per-frame accuracy — set by config, NOT the channel\n"
              "(identical loopback vs OAI: a delivered frame decodes to the model's output)",
              fontsize=11.5, fontweight="bold")
ax1.legend(loc="upper center", fontsize=9, ncol=2, frameon=False)

# ---- Panel B: delivery / availability (the network's cost) ----
bd = ax2.bar(x, deliv, 0.5, color=C_DEL)
for r in bd:
    ax2.annotate(f"{r.get_height():.1f}%", (r.get_x()+r.get_width()/2, r.get_height()),
                 xytext=(0,2), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.axhline(100, color=C_LOOP, lw=1.3, ls="--"); ax2.annotate("loopback 100%", (-0.35, 101), ha="left", va="bottom", fontsize=9, color=C_LOOP)
ax2.annotate("25% of ticks:\nno fresh perception", xy=(0, 87), xytext=(0.55, 60), ha="left", fontsize=9,
             color="#8a3b00", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#8a3b00", lw=1.2))
ax2.set_ylabel("delivery — fresh result per tick (%)"); ax2.set_ylim(0, 112)
ax2.set_xticks(x); ax2.set_xticklabels(lbl, fontsize=9.5)
ax2.set_title("Delivery — the channel's real cost (availability)", fontsize=12, fontweight="bold")

fig.suptitle("Accuracy is config-driven & channel-invariant; the network costs AVAILABILITY, not accuracy",
             fontsize=12.5, fontweight="bold", y=1.01)
fig.text(0.5, -0.02, "Not shown: staleness (a late-but-delivered frame is temporally misaligned -> position error) "
         "= the true latency->accuracy link; a separate future measurement.",
         ha="center", fontsize=8.3, style="italic", color="#666666")
fig.tight_layout()
fig.savefig(OUT/"oai_accuracy_delivery.pdf", format="pdf", bbox_inches="tight")
fig.savefig(OUT/"oai_accuracy_delivery.png", format="png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/oai_accuracy_delivery.pdf/.png")
