#!/usr/bin/env python3
"""Radar-pps study figures: accuracy-vs-cost Pareto, person-recall-by-distance, cost-vs-pps.
Reads deployment cost metrics from PPS_DEPLOY_RESULTS.md (parsed) + embedded accuracy from the
ablation. Follows dataviz principles: pps is ORDERED -> sequential (viridis) color; one y-axis per
chart (no dual-axis); direct labels; recessive grid; legend for series. Outputs PNGs + a summary md.

Run after the deployment measurement writes PPS_DEPLOY_RESULTS.md. Accuracy-only figs render even
without cost data.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
OUT = AB / "cooperative_fusion" / "pps_study_figs"; OUT.mkdir(parents=True, exist_ok=True)
PPS = [100000, 150000, 200000, 250000, 300000]
LAB = {p: f"{p//1000}k" for p in PPS}

# ---- accuracy (from PPS_ABLATION_ANALYSIS_20260702.md) ----
ACC = {  # pps: dict
    100000: dict(veh_iou=0.910, miou=0.827, det_f1=0.818, veh_f1=0.875, per_f1=0.718,
                 per_rec={"0-10":0.86,"10-20":None,"20-30":None,"30-40":None}),  # 100k ref (own collection)
    150000: dict(veh_iou=0.943, miou=0.837, det_f1=0.814, veh_f1=0.850, per_f1=0.742,
                 per_rec={"0-10":0.76,"10-20":0.74,"20-30":0.82,"30-40":0.75}),
    200000: dict(veh_iou=0.934, miou=0.837, det_f1=0.846, veh_f1=0.870, per_f1=0.806,
                 per_rec={"0-10":0.94,"10-20":0.88,"20-30":0.87,"30-40":0.80}),
    250000: dict(veh_iou=0.925, miou=0.835, det_f1=0.828, veh_f1=0.851, per_f1=0.783,
                 per_rec={"0-10":0.85,"10-20":0.85,"20-30":0.87,"30-40":0.78}),
    300000: dict(veh_iou=0.939, miou=0.849, det_f1=0.831, veh_f1=0.856, per_f1=0.790,
                 per_rec={"0-10":0.90,"10-20":0.88,"20-30":0.88,"30-40":0.76}),
}
def per_near(p):  # mean of 0-10 & 10-20 where available
    r = ACC[p]["per_rec"]; vals = [r["0-10"], r["10-20"]]; vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None

# ---- parse deployment cost metrics ----
def load_cost():
    f = AB / "PPS_DEPLOY_RESULTS.md"
    cost = {}
    if not f.exists(): return cost
    for line in f.read_text().splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if not m: continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        try:
            pps = int(cells[0])
            # cols: pps|frames|results_n|front_ms|back_ms|RTT|transport_est|payload_comp|payload_uncomp|RTT_p95
            cost[pps] = dict(front_ms=float(cells[3]), back_ms=float(cells[4]), rtt_ms=float(cells[5]),
                             transport_ms=float(cells[6]), payload_comp=float(cells[7]),
                             payload_uncomp=float(cells[8]))
        except Exception: pass
    return cost

# ---- style ----
plt.rcParams.update({"figure.dpi":140,"axes.grid":True,"grid.alpha":0.22,"grid.linewidth":0.6,
                     "axes.spines.top":False,"axes.spines.right":False,"font.size":13,
                     "axes.titlesize":16,"axes.titleweight":"bold","axes.labelsize":13,
                     "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12,
                     "pdf.fonttype":42,"svg.fonttype":"none","figure.constrained_layout.use":False})
cmap = plt.cm.viridis
colors = {p: cmap(i/(len(PPS)-1)) for i,p in enumerate(PPS)}

def save(fig, name):  # vector PDF (crisp at any zoom) + PNG for quick preview
    fig.savefig(OUT/f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT/f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def fig_person_by_distance():
    bins = ["0-10","10-20","20-30","30-40"]; x = np.arange(len(bins))
    clean = [p for p in PPS if all(ACC[p]["per_rec"][b] is not None for b in bins)]  # same-collection set (150-300k)
    w = 0.82/len(clean)
    fig, ax = plt.subplots(figsize=(10,5.6))
    for i,p in enumerate(clean):
        r = ACC[p]["per_rec"]; ys = [r[b] for b in bins]
        xpos = x + (i-(len(clean)-1)/2)*w
        ax.bar(xpos, ys, w, color=colors[p], label=LAB[p], zorder=3)
        for xi,yi in zip(xpos,ys):  # direct value labels
            ax.text(xi, yi+0.012, f"{yi:.2f}", ha="center", va="bottom", fontsize=8.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([f"{b} m" for b in bins])
    ax.set_ylabel("person detection recall"); ax.set_ylim(0,1.05)
    ax.set_xlabel("distance to pedestrian")
    ax.set_title("Pedestrian recall improves with radar pps — strongest in the near field")
    ax.legend(title="radar pps", ncol=len(clean), frameon=False, loc="upper center", bbox_to_anchor=(0.5,-0.11))
    ax.axhline(0.9, color="#888", lw=1, ls="--", zorder=1); ax.text(3.42,0.905,"0.90",color="#888",fontsize=10)
    fig.tight_layout(); save(fig, "person_recall_by_distance")

def fig_pareto(cost):
    # Payload is pps-INDEPENDENT (radar is a fixed-size raster before the split point), so there is no
    # accuracy-cost frontier — the honest chart is a same-cost vertical cluster with accuracy varying by
    # pps. x-axis anchored at 0 so the ~4% payload spread reads as "essentially constant"; NO connecting
    # line (payload does not order with pps).
    if not cost: return None
    fig, ax = plt.subplots(figsize=(8,5.5))
    xs=[cost[p]["payload_comp"] for p in PPS if p in cost]
    for p in PPS:
        if p not in cost: continue
        x = cost[p]["payload_comp"]; y = ACC[p]["per_f1"]
        ax.scatter(x, y, s=150, color=colors[p], zorder=3, edgecolor="white", linewidth=1.2, label=LAB[p])
        ax.annotate(LAB[p], (x,y), textcoords="offset points", xytext=(10,0), va="center", fontsize=10, fontweight="bold")
    xm=max(xs)
    ax.axvspan(min(xs)-5, xm+5, color=cmap(0.5), alpha=0.06, zorder=0)
    ax.set_xlim(0, xm*1.22)
    ax.set_ylim(0.68, 0.83)
    ax.set_xlabel("intermediate-tensor payload per frame (KB, compressed)")
    ax.set_ylabel("pedestrian detection F1")
    ax.set_title("No accuracy–cost tradeoff: payload is pps-independent")
    ax.text(xm*0.5, 0.695, f"payload ≈ {np.mean(xs):.0f} KB for every pps\n(radar rasterized to fixed-size channel before split)",
            fontsize=11, color="#555", ha="center")
    fig.tight_layout(); save(fig, "pareto_accuracy_vs_payload")
    return True

def fig_latency_breakdown(cost):
    """Stacked bar: front + back + transport = total split-inference pipeline, per pps.
    Makes visible what 'RTT' (round_trip = back + transport) covers vs the full front+RTT pipeline."""
    if not cost: return None
    ps=[p for p in PPS if p in cost]; x=np.arange(len(ps))
    front=[cost[p]["front_ms"] for p in ps]; back=[cost[p]["back_ms"] for p in ps]
    trans=[cost[p]["transport_ms"] for p in ps]
    fig, ax = plt.subplots(figsize=(9.5,5.6))
    b1=ax.bar(x, front, 0.6, label="front compute", color=cmap(0.15), zorder=3)
    b2=ax.bar(x, back, 0.6, bottom=front, label="back compute", color=cmap(0.55), zorder=3)
    b3=ax.bar(x, trans, 0.6, bottom=np.array(front)+np.array(back), label="transport (out+back, loopback)", color=cmap(0.85), zorder=3)
    for i,p in enumerate(ps):
        tot=front[i]+back[i]+trans[i]
        ax.text(x[i], tot+1.2, f"{tot:.0f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LAB[p] for p in ps])
    ax.set_xlabel("radar pps"); ax.set_ylabel("latency (ms)")
    ax.set_ylim(0, max(np.array(front)+np.array(back)+np.array(trans))*1.18)
    ax.set_title("Split-inference latency breakdown — flat across pps")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5,-0.11), ncol=3)
    ax.text(0.5,-0.24,'"RTT" (round_trip) = back + transport;  total pipeline = front + RTT',
            transform=ax.transAxes, ha="center", fontsize=10.5, color="#555")
    fig.tight_layout(); save(fig, "latency_breakdown")
    return True

def fig_cost_vs_pps(cost):
    if not cost: return None
    ps=[p for p in PPS if p in cost]; x=[p//1000 for p in ps]
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    axes[0].plot(x,[cost[p]["payload_uncomp"] for p in ps],"--o",color=cmap(0.7),lw=2,ms=7,label="uncompressed")
    axes[0].plot(x,[cost[p]["payload_comp"] for p in ps],"-o",color=cmap(0.3),lw=2.5,ms=9,label="compressed")
    axes[0].set_ylim(0, max(cost[p]["payload_uncomp"] for p in ps)*1.18)  # anchor at 0 so flatness is honest
    axes[0].set_title("Payload — flat across pps"); axes[0].set_xlabel("radar pps (k)"); axes[0].set_ylabel("KB / frame")
    axes[0].legend(frameon=False, loc="center right")
    axes[1].plot(x,[cost[p]["front_ms"] for p in ps],"-o",label="front",color=cmap(0.15),lw=2.5,ms=8)
    axes[1].plot(x,[cost[p]["rtt_ms"] for p in ps],"-o",label="RTT (back+transport)",color=cmap(0.85),lw=2.5,ms=8)
    axes[1].plot(x,[cost[p]["back_ms"] for p in ps],"-o",label="back",color=cmap(0.5),lw=2.5,ms=8)
    axes[1].set_ylim(0, max(cost[p]["front_ms"] for p in ps)*1.25)
    axes[1].set_title("Latency — flat across pps"); axes[1].set_xlabel("radar pps (k)"); axes[1].set_ylabel("ms"); axes[1].legend(frameon=False)
    fig.suptitle("Split-inference cost is independent of radar pps", fontsize=17, fontweight="bold")
    fig.tight_layout(); save(fig, "cost_vs_pps")
    return True

def summary_table(cost):
    lines=["# Radar-pps study — accuracy vs cost (2026-07-02)\n",
           "| pps | veh IoU | mIoU | veh F1 | person F1 | person near-recall | payload KB(comp) | front_ms | back_ms | RTT_ms |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for p in PPS:
        a=ACC[p]; c=cost.get(p,{})
        pn=per_near(p); pn=f"{pn:.2f}" if pn is not None else "n/a"
        def g(k): return f"{c[k]:.1f}" if k in c else "—"
        lines.append(f"| {LAB[p]} | {a['veh_iou']:.3f} | {a['miou']:.3f} | {a['veh_f1']:.3f} | {a['per_f1']:.3f} | {pn} | {g('payload_comp')} | {g('front_ms')} | {g('back_ms')} | {g('rtt_ms')} |")
    lines += [
        "",
        "Accuracy from the ablation (150k–300k same seed/route; 100k is the prior-collection reference).",
        "Cost from loopback split-inference deployment (400 frames = 2 crowded loops, seed 31, same route).",
        "back_ms / RTT are over the frames whose result returned; front_ms & payload are per-frame.",
        "",
        "## Conclusion",
        "",
        "**Higher radar pps buys pedestrian recall up to ~200k — at no transport cost.**",
        "",
        "- **Accuracy:** pedestrian detection is the only radar-limited class. Person F1 climbs 0.72→0.81 and",
        "  near-field recall 0.74 (150k) → 0.90 (200k), then plateaus. Vehicles and segmentation are already",
        "  saturated (flat across pps). So ~200k pps is the accuracy sweet spot; beyond it adds nothing.",
        "- **Transport cost is pps-independent.** Radar is fused as a fixed-size 4-channel raster *before* the",
        "  split point, so the intermediate-tensor shape doesn't depend on point count: uncompressed payload is",
        "  identical (2835 KB) for all five models, compressed varies only ~4% (content entropy, not size), and",
        "  front/back/RTT latency is flat (~49 / ~8 / ~40 ms). There is **no accuracy–bandwidth tradeoff** on the",
        "  wire — the cost of higher pps is entirely front-end (rasterizing more points), not the split link.",
        "- **Bottom line:** run at ~200k pps — you get the full pedestrian-recall gain, pay no extra payload or",
        "  latency for it, and gain nothing by pushing higher.",
        "",
        "**Caveats:** loopback (not the live 5G link) — payload + front/back compute are exact, but the ~40 ms",
        "RTT and transport estimate are localhost, not over-the-air; real 5G RTT is a follow-on with the OAI",
        "stack up (`--role back` receiver). Also, ~14–20% of frames returned a result over UDP (the ~1 MB payload",
        "fragments), so back/RTT means use 56–78 samples/model — enough for a stable mean, and the high UDP drop",
        "rate for MB-scale tensors is itself a deployment note (large intermediate tensors want a reliable transport).",
        "",
        "**Figures (PDF + PNG):** `cooperative_fusion/pps_study_figs/` — person_recall_by_distance,",
        "latency_breakdown, cost_vs_pps, pareto_accuracy_vs_payload.",
    ]
    (AB/"PPS_STUDY_SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines[:len(lines)]))

if __name__ == "__main__":
    cost = load_cost()
    fig_person_by_distance()
    fig_pareto(cost)
    fig_latency_breakdown(cost)
    fig_cost_vs_pps(cost)
    summary_table(cost)
    print(f"\nfigures -> {OUT}  ({'with' if cost else 'NO'} deployment cost data)")
    if not cost: print("(cost figs skipped — run after PPS_DEPLOY_RESULTS.md exists)")
