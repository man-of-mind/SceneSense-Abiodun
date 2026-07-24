#!/usr/bin/env python3
"""Visualize the UL drain rate: RLC occupancy decay (slope = drain rate),
default QPSK vs forced 64QAM, + the theoretical spectral-efficiency ceiling.

Reads pre-extracted LCID4 occupancy (time_sec,bytes) temp CSVs.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SC = Path("/tmp/claude-200171/-home-shr-aisvcs-workarea-carla-0-10-env-Carla-0-10-0-Linux-Shipping-PythonAPI-neu-collab/745221c3-a3f4-48d1-abf6-c5830356e63b/scratchpad")
OUT = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/oai_layer_latency/plots")
C_DEF, C_FRC = "#D1495B", "#2E86AB"


def load(p):
    t, b = [], []
    with open(p) as f:
        for line in f:
            s, v = line.split(",")
            t.append(float(s)); b.append(int(v) / 1024.0)  # KB
    t = np.array(t); b = np.array(b)
    t = t - t[0]
    return t, b


def window_with_burst(t, b, span=0.45, pre=0.05):
    """Find a short window around a large frame arrival (occupancy > 800 KB)."""
    idx = np.argmax(b > 800)
    t0 = t[idx] - pre
    m = (t >= t0) & (t <= t0 + span)
    return t[m] - t0, b[m]


td, bd = load(SC / "rlc_occ_default.csv")
tf, bf = load(SC / "rlc_occ_forced.csv")
wtd, wbd = window_with_burst(td, bd)
wtf, wbf = window_with_burst(tf, bf)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle("UL drain rate: QPSK (MCS ~4-8) drains a 1 MB frame ~6x slower than 64QAM (MCS 28)\n"
             "UE RLC data-bearer occupancy over time — the decay slope IS the drain rate", fontsize=12)

axA = ax[0]
axA.plot(wtd * 1000, wbd, color=C_DEF, lw=1.5, label="default (QPSK) — slow decay")
axA.plot(wtf * 1000, wbf, color=C_FRC, lw=1.5, label="forced MCS 28 (64QAM) — fast decay")
axA.axhline(1024, color="gray", ls=":", lw=1); axA.text(axA.get_xlim()[1]*0.98, 1035, "~1 MB feature frame", ha="right", fontsize=8, color="gray")
axA.set_xlabel("time within window (ms)"); axA.set_ylabel("UE RLC occupancy (KB)")
axA.set_title("A. A queued 1 MB frame drains far faster at 64QAM"); axA.legend(fontsize=9)

# B. spectral-efficiency ceiling: max UL goodput at 273 PRB, mu=1
# approx bits/RE by modulation*coderate; use nominal MCS spectral efficiency (bits/symbol) from 38.214 table 1
# MCS8 (QPSK, R~0.49)~0.98 b/RE; MCS16(16QAM,R~0.64)~2.57; MCS28(64QAM,R~0.93)~5.55
axB = ax[1]
# usable REs/s uplink: 273 PRB * 12 subcarriers * ~12 symbols/slot * 2000 slots/s (mu=1) * ~0.85 (dmrs/overhead)
re_per_s = 273 * 12 * 12 * 2000 * 0.85
mods = [("QPSK\n(MCS 8)", 0.98, C_DEF), ("16QAM\n(MCS 16)", 2.57, "#EDAE49"), ("64QAM\n(MCS 28)", 5.55, C_FRC)]
labels = [m[0] for m in mods]
mbps = [re_per_s * m[1] / 1e6 for m in mods]
axB.bar(labels, mbps, color=[m[2] for m in mods])
for i, v in enumerate(mbps):
    axB.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)
axB.annotate("64QAM ceiling is\n~5.7x QPSK", xy=(2, mbps[2]), xytext=(0.7, mbps[2]*0.8),
             fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color="k"))
axB.set_ylabel("theoretical max UL goodput @273 PRB (Mbps)")
axB.set_title("B. QPSK caps spectral efficiency (per-burst drain ceiling)")

fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"uplink_drain_rate.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {OUT/'uplink_drain_rate.png'}")
# quantify decay: time from peak (~1MB) back below 100KB
def decay_ms(t, b):
    i = np.argmax(b)
    for j in range(i, len(b)):
        if b[j] < 100:
            return (t[j] - t[i]) * 1000
    return float("nan")
print(f"  default 1MB->'<100KB drain time: {decay_ms(wtd,wbd):.0f} ms")
print(f"  forced  1MB->'<100KB drain time: {decay_ms(wtf,wbf):.0f} ms")
