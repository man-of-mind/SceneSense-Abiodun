#!/usr/bin/env python3
"""Presentation plots for the OAI uplink per-layer latency + MCS investigation.

Panels:
  A. End-to-end round-trip + per-packet RAN uplink transit: default vs forced MCS28.
  B. UL MCS distribution: iperf vs CARLA-default vs forced-MCS28 (the QPSK cap).
  C. Per-packet RAN uplink transit CDF: default vs forced MCS28.
  D. UE RLC data-bearer occupancy (mean / p95 / max): default vs forced MCS28.

Reads only saved artifacts under metrics_logs/scenesense_ttracer + downlink runs.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
TT = AB / "metrics_logs" / "scenesense_ttracer"
RUNS = AB / "downlink_latency_fps" / "runs" / "oai_bw273_mu1_ttracer"
OUT = AB / "oai_layer_latency" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

IPERF = "validate_20260722_175810_iperf_ul"
DEF = "downlink_oai_bw273_mu1_ttracer_fps10_layerinstr_20260722_191024"
FRC = "downlink_oai_bw273_mu1_ttracer_fps10_forcemcs28_20260722_201150"

C_DEF, C_FRC, C_IPF = "#D1495B", "#2E86AB", "#8D99AE"  # default(red), forced(blue), iperf(grey)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


def mcs_hist(rg):
    p = TT / rg / "ue" / "csv" / "NRUE_MAC_DCI_GRANT.csv"
    xs = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("direction") == "1":
                try:
                    xs.append(int(r["mcs"]))
                except ValueError:
                    pass
    return np.array(xs)


def transit_ms(rg):
    def load(ev):
        out = []
        with open(TT / rg / ("gnb" if ev.startswith("GNB") else "ue") / "csv" / f"{ev}.csv") as f:
            for r in csv.DictReader(f):
                try:
                    out.append((int(r["mono_sec"]) + int(r["mono_nsec"]) / 1e9, int(r["sdu_bytes"])))
                except (KeyError, ValueError):
                    pass
        return out
    pin, gout = load("NR_PDCP_TX_SDU"), load("GNB_PDCP_RX_DELIVER")
    n = min(len(pin), len(gout))
    d = [(gout[i][0] - pin[i][0]) * 1000 for i in range(n)
         if pin[i][1] == gout[i][1] and pin[i][1] > 1000 and 0 < (gout[i][0] - pin[i][0]) < 10]
    return np.array(d)


def rtt_p(rg, q=50):
    p = RUNS / f"fps_10_{rg.split('fps10_')[1]}" / "streams" / f"{rg}_metrics.csv"
    xs = []
    with open(p) as f:
        for r in csv.DictReader(f):
            v = r.get("round_trip_result_recv_ms", "")
            if v not in ("", "nan", "NaN", None):
                try:
                    xs.append(float(v))
                except ValueError:
                    pass
    return np.percentile(xs, q) if xs else float("nan")


print("loading MCS ...")
mcs_ipf, mcs_def, mcs_frc = mcs_hist(IPERF), mcs_hist(DEF), mcs_hist(FRC)
print("loading transit ...")
tr_def, tr_frc = transit_ms(DEF), transit_ms(FRC)
print("loading RTT ...")
rtt_def, rtt_frc = rtt_p(DEF), rtt_p(FRC)

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("OAI uplink latency: low UL MCS (QPSK cap) is the bottleneck; forcing MCS 28 (64QAM) fixes it\n"
             "CARLA split-inference, 273PRB, RFsim ideal channel, no-AE/zstd/ROI0", fontsize=13, y=0.99)

# A. latency bars
axA = ax[0, 0]
labels = ["End-to-end RTT\n(front->result, p50)", "RAN uplink transit\n(PDCP->PDCP, p50)", "RAN uplink transit\n(p95)"]
defv = [rtt_def, np.percentile(tr_def, 50), np.percentile(tr_def, 95)]
frcv = [rtt_frc, np.percentile(tr_frc, 50), np.percentile(tr_frc, 95)]
x = np.arange(len(labels)); w = 0.38
axA.bar(x - w/2, defv, w, label="default (MCS ~4-8, QPSK)", color=C_DEF)
axA.bar(x + w/2, frcv, w, label="forced MCS 28 (64QAM)", color=C_FRC)
for i, (d, fv) in enumerate(zip(defv, frcv)):
    axA.text(i - w/2, d + 3, f"{d:.0f}", ha="center", fontsize=9)
    axA.text(i + w/2, fv + 3, f"{fv:.0f}", ha="center", fontsize=9)
axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9)
axA.set_ylabel("latency (ms)"); axA.set_title("A. Latency collapses when MCS is raised"); axA.legend(fontsize=9)

# B. MCS distribution
axB = ax[0, 1]
bins = np.arange(-0.5, 29.5, 1)
for xs, c, lb in [(mcs_ipf, C_IPF, "iperf (smooth)"), (mcs_def, C_DEF, "CARLA default"), (mcs_frc, C_FRC, "CARLA forced 28")]:
    if len(xs):
        axB.hist(xs, bins=bins, density=True, alpha=0.6, color=c, label=lb)
axB.axvspan(-0.5, 9.5, color="grey", alpha=0.08)
axB.text(4.5, axB.get_ylim()[1]*0.9, "QPSK\n(MCS 0-9)", ha="center", fontsize=8, color="dimgray")
axB.axvline(9.5, color="k", ls=":", lw=1)
axB.set_xlabel("UL MCS index (table 1)"); axB.set_ylabel("fraction of grants")
axB.set_title("B. CARLA is pinned in QPSK; iperf reaches MCS 28"); axB.legend(fontsize=9)

# C. transit CDF
axC = ax[1, 0]
for xs, c, lb in [(tr_def, C_DEF, "default (MCS ~4-8)"), (tr_frc, C_FRC, "forced MCS 28")]:
    xs = np.sort(xs); axC.plot(xs, np.linspace(0, 1, len(xs)), color=c, lw=2, label=lb)
axC.set_xlabel("per-packet RAN uplink transit (ms)"); axC.set_ylabel("CDF")
axC.set_xlim(0, 200); axC.set_title("C. Per-packet RAN transit distribution"); axC.legend(fontsize=9)

# D. RLC occupancy (from analyzer summaries)
axD = ax[1, 1]
occ_labels = ["mean", "p95", "max"]
occ_def = [136.9, 942.9, 1105.9]; occ_frc = [19.0, 0.0, 1104.3]
x = np.arange(len(occ_labels))
axD.bar(x - w/2, occ_def, w, label="default", color=C_DEF)
axD.bar(x + w/2, occ_frc, w, label="forced MCS 28", color=C_FRC)
for i, (d, fv) in enumerate(zip(occ_def, occ_frc)):
    axD.text(i - w/2, d + 15, f"{d:.0f}", ha="center", fontsize=9)
    axD.text(i + w/2, fv + 15, f"{fv:.0f}", ha="center", fontsize=9)
axD.set_xticks(x); axD.set_xticklabels(occ_labels)
axD.set_ylabel("UE RLC data-bearer occupancy (KB)")
axD.set_title("D. RLC queue buildup vanishes (peak still ~1 frame, but drains fast)"); axD.legend(fontsize=9)

fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"uplink_mcs_bottleneck_summary.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {OUT/'uplink_mcs_bottleneck_summary.png'}")
print(f"  RTT p50: default={rtt_def:.0f}ms forced={rtt_frc:.0f}ms")
print(f"  transit p50: default={np.percentile(tr_def,50):.0f}ms forced={np.percentile(tr_frc,50):.0f}ms")
print(f"  MCS p50: iperf={np.median(mcs_ipf):.0f} default={np.median(mcs_def):.0f} forced={np.median(mcs_frc):.0f}")
