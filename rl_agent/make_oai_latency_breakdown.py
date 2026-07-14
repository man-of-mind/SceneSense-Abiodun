#!/usr/bin/env python3
"""OAI latency breakdown: end-to-end pipeline split into front (UE compute) / transport / back (edge compute),
per config. Shows compression shrinks the TRANSPORT term (the bottleneck); front/back compute ~flat and the
AE encode/decode adds only ~0.27 ms (annotated). Okabe-Ito, stacked bars."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent / "plots"; OUT.mkdir(exist_ok=True)
C_FRONT, C_TRANS, C_BACK = "#E69F00", "#0072B2", "#009E73"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True})

lbl   = ["no-AE · u8\n1141 KB\n(baseline)", "AE-128 · u8\n346 KB", "AE-128 · u4\n142 KB"]
front = [58.2, 33.6, 28.2]      # UE compute incl. quant+entropy+fragment
trans = [199.5, 107.1, 67.1]    # transport (uplink-dominated; downlink result tiny)
back  = [9.1, 11.0, 10.25]      # edge compute incl. entropy-decode + AE-decode + heads
tot   = [f+t+b for f,t,b in zip(front,trans,back)]
x = np.arange(len(lbl))

fig, ax = plt.subplots(figsize=(8.8, 6.0))
b1 = ax.bar(x, front, 0.55, label="front · UE compute", color=C_FRONT)
b2 = ax.bar(x, trans, 0.55, bottom=front, label="transport · uplink+downlink", color=C_TRANS)
b3 = ax.bar(x, back,  0.55, bottom=[f+t for f,t in zip(front,trans)], label="back · edge compute", color=C_BACK)
for i in range(len(lbl)):
    ax.annotate(f"{front[i]:.0f}", (x[i], front[i]/2), ha="center", va="center", fontsize=8.5, color="white")
    ax.annotate(f"{trans[i]:.0f}", (x[i], front[i]+trans[i]/2), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.annotate(f"{back[i]:.0f}", (x[i], front[i]+trans[i]+back[i]/2), ha="center", va="center", fontsize=8, color="white")
    ax.annotate(f"{tot[i]:.0f} ms", (x[i], tot[i]), xytext=(0,3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("end-to-end pipeline latency (ms)"); ax.set_ylim(0, max(tot)*1.15)
ax.set_xticks(x); ax.set_xticklabels(lbl)
ax.set_title("OAI latency breakdown — compression shrinks the TRANSPORT term\n"
             "(front/back compute ~flat; AE encode+decode adds only ~0.27 ms/frame)",
             fontsize=12.5, fontweight="bold", pad=10)
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
ax.annotate("AE encode 0.18 ms + decode 0.09 ms = 0.27 ms/frame\n(negligible — invisible in these bars)",
            (1.5, 235), ha="center", fontsize=9, style="italic", color="#555555")
fig.tight_layout()
fig.savefig(OUT/"oai_latency_breakdown.pdf", format="pdf", bbox_inches="tight")
fig.savefig(OUT/"oai_latency_breakdown.png", format="png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/oai_latency_breakdown.pdf/.png")
