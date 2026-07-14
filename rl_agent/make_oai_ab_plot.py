#!/usr/bin/env python3
"""OAI A/B plot: compression collapses transport latency + loss on real 5G (single-UE, no impairment).
3 configs (no-AE u8 -> AE-128 u8 -> AE-128 u4). Top: RTT mean+p95 vs loopback floor. Bottom: delivery %.
Okabe-Ito, direct labels."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent / "plots"; OUT.mkdir(exist_ok=True)
C_RTT, C_P95, C_DEL, C_LOOP = "#0072B2", "#56B4E9", "#009E73", "#D55E00"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})

lbl = ["no-AE · u8\n1141 KB\n(baseline)", "AE-128 · u8\n346 KB", "AE-128 · u4\n142 KB"]
rtt_mean = [208.6, 118.1, 77.3]
rtt_p95  = [270.6, 174.0, 138.9]
deliver  = [75.0, 99.3, 99.0]
LOOPBACK = 53.0
x = np.arange(len(lbl)); w = 0.38

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 7.2), gridspec_kw={"height_ratios": [1.5, 1]})
b1 = ax1.bar(x - w/2, rtt_mean, w, label="RTT mean", color=C_RTT)
b2 = ax1.bar(x + w/2, rtt_p95,  w, label="RTT p95", color=C_P95)
for bars in (b1, b2):
    for r in bars:
        ax1.annotate(f"{r.get_height():.0f}", (r.get_x()+r.get_width()/2, r.get_height()),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
ax1.axhline(LOOPBACK, color=C_LOOP, lw=1.4, ls="--")
ax1.annotate(f"loopback floor {LOOPBACK:.0f} ms", (len(lbl)-1, LOOPBACK+6), ha="right", fontsize=9, color=C_LOOP)
ax1.set_ylabel("round-trip latency (ms)"); ax1.set_ylim(0, max(rtt_p95)*1.18)
ax1.set_xticks(x); ax1.set_xticklabels([])
ax1.set_title("Compression collapses OAI transport latency toward the loopback floor\n"
              "(single-UE 5G, 106 PRB / 40 MHz, no channel impairment)",
              fontsize=12.5, fontweight="bold", pad=10)
ax1.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

bd = ax2.bar(x, deliver, 0.5, color=C_DEL)
for r in bd:
    ax2.annotate(f"{r.get_height():.1f}%", (r.get_x()+r.get_width()/2, r.get_height()),
                 xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)
ax2.set_ylabel("delivery (%)"); ax2.set_ylim(0, 108)
ax2.axhline(100, color="#888888", lw=0.8, ls=":")
ax2.set_xticks(x); ax2.set_xticklabels(lbl)
fig.tight_layout()
fig.savefig(OUT / "oai_ab_compression.pdf", format="pdf", bbox_inches="tight")
fig.savefig(OUT / "oai_ab_compression.png", format="png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/oai_ab_compression.pdf/.png")
