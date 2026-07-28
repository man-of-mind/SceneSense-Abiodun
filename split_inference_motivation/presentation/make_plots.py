#!/usr/bin/env python3
"""Slide plots for the split-inference motivation study. Values verified from results/ on 2026-07-27.
Okabe-Ito colorblind-safe palette; bold, minimal-chrome, direct-labeled. Outputs PDF (vector) + PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "plots"; OUT.mkdir(exist_ok=True)

# Okabe-Ito
BLACK="#000000"; ORANGE="#E69F00"; SKY="#56B4E9"; GREEN="#009E73"; BLUE="#0072B2"; VERM="#D55E00"; PURPLE="#CC79A7"
INK="#222222"; MUTED="#7a7a7a"; GRID="#d9d9d9"
FULL=VERM      # full-local = the costly one
SPLIT=GREEN    # split = the better-on-compute one

plt.rcParams.update({
    "figure.dpi":120, "savefig.dpi":120, "font.size":15, "font.family":"DejaVu Sans",
    "axes.edgecolor":MUTED, "axes.linewidth":1.0, "axes.labelcolor":INK, "text.color":INK,
    "xtick.color":INK, "ytick.color":INK, "axes.titlesize":18, "axes.titleweight":"bold",
    "axes.labelsize":15, "legend.frameon":False, "figure.facecolor":"white", "axes.facecolor":"white",
})
def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
def save(fig,name,tight=True):
    if tight: fig.tight_layout()
    for ext in ("pdf","png"): fig.savefig(OUT/f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig); print(f"  wrote {name}.pdf/.png")

# ---------- FIG 1: E6 compute crossover (the money plot) ----------
cores=np.array([1,2,4,8,16]); full=np.array([5.49,9.96,17.79,29.49,29.89]); split=np.array([15.68,25.91,44.32,65.19,58.58])
fig,ax=plt.subplots(figsize=(8.4,5.4)); clean(ax)
ax.axhspan(0,10,color=VERM,alpha=0.06,zorder=0)
ax.axhline(10,color=MUTED,ls="--",lw=1.3); ax.axhline(30,color=MUTED,ls=":",lw=1.3)
ax.text(16,10.6,"10 FPS real-time floor",ha="right",va="bottom",color=MUTED,fontsize=12,style="italic")
ax.text(16,30.6,"30 FPS (fast driving)",ha="right",va="bottom",color=MUTED,fontsize=12,style="italic")
ax.plot(cores,split,"-o",color=SPLIT,lw=3,ms=9,label="Split (car runs backbone only)",zorder=5)
ax.plot(cores,full,"-o",color=FULL,lw=3,ms=9,label="Full-local (car runs whole model)",zorder=5)
ax.annotate("misses 10 FPS\nat 1–2 cores",(2,9.96),xytext=(3.2,4.5),color=FULL,fontsize=12,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=FULL,lw=1.5))
ax.annotate("clears 30 FPS\nfrom 4 cores",(4,44.3),xytext=(5,52),color=SPLIT,fontsize=12,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=SPLIT,lw=1.5))
ax.set_xscale("log",base=2); ax.set_xticks(cores); ax.set_xticklabels(cores)
ax.set_xlabel("On-vehicle compute budget  (CPU cores)"); ax.set_ylabel("Sustained speed  (frames / second)")
ax.set_title("Under a tight compute budget, only Split keeps up",loc="left")
ax.set_ylim(0,70); ax.legend(loc="upper left",fontsize=13)
save(fig,"fig1_compute_crossover")

# ---------- FIG 2: E1 where the compute goes ----------
fig,ax=plt.subplots(figsize=(8.6,3.9)); ax.set_axisbelow(True)
ax.barh([0],[24.1],color=SPLIT,zorder=3)
ax.barh([0],[75.9],left=[24.1],color=ORANGE,zorder=3)
ax.text(24.1/2,0,"Backbone\n24%\n(stays on car)",ha="center",va="center",color="white",fontweight="bold",fontsize=12)
ax.text(24.1+75.9/2,0,"Heads — 76% of the model's compute\n(offloaded to the edge)",ha="center",va="center",color="white",fontweight="bold",fontsize=13.5)
ax.set_xlim(0,100); ax.set_ylim(-0.7,0.7); ax.set_yticks([])
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.set_xlabel("Share of total model compute  (% of FLOPs)")
ax.set_title("Split moves the heavy 76% off the vehicle",loc="left",pad=14)
ax.text(0,-0.62,"One head layer (a 1000→128 convolution) alone = 59% of the whole model.",
        color=MUTED,fontsize=11,style="italic")
save(fig,"fig2_where_compute_goes")

# ---------- FIG 3: E3 the cost of split (2 panels: uplink + latency) ----------
fig,(a1,a2)=plt.subplots(1,2,figsize=(12.5,5.4));
for ax in (a1,a2): clean(ax)
# panel 1 — uplink (architecture comparison)
lab1=["Full-local","Split\n(no comp.)","Split\n(+AE comp.)"]; up=[0.18,81.68,11.93]; c1=[SKY,VERM,SPLIT]
b1=a1.bar(lab1,up,color=c1,zorder=3); a1.set_yscale("log"); a1.set_ylim(0.1,320)
a1.axhline(10.9,color=MUTED,ls="--",lw=1.3)
a1.text(-0.42,235,"5G uplink budget (~10.9 Mbps, dashed)",color=MUTED,fontsize=10.5,style="italic",ha="left",va="center")
for r,v in zip(b1,up): a1.text(r.get_x()+r.get_width()/2,v*1.35,f"{v:g} Mbps",ha="center",fontweight="bold",fontsize=11.5)
a1.set_ylabel("Uplink needed  (Mbps, log scale)")
a1.set_title("Bandwidth: split needs the most (~450× A)",loc="left",fontsize=14.5,pad=8)
# panel 2 — latency of the SPLIT pipeline by transport (LOOPBACK-centered), full-local reference band
lab2=["Ideal\nloopback","5G +\ncompression","5G, no\ncompression"]; lat=[46.1,86.5,188.0]; c2=[SPLIT,ORANGE,VERM]
b2=a2.bar(lab2,lat,color=c2,zorder=3); a2.set_ylim(0,212)
a2.axhspan(33,42,color=SKY,alpha=0.20,zorder=1)
a2.text(2.45,37.5,"full-local ≈ 33–42 ms\n(compute-bound)",ha="right",va="center",color="#2b6a8a",fontsize=10.5,style="italic")
for r,v in zip(b2,lat): a2.text(r.get_x()+r.get_width()/2,v+4,f"{v:.0f} ms",ha="center",fontweight="bold",fontsize=13)
a2.set_ylabel("Split latency: capture → result  (ms)")
a2.set_title("Latency: it's the 5G radio, not compute",loc="left",fontsize=14.5,pad=8)
fig.text(0.5,-0.03,"Split's own compute is ~46 ms on ideal transport (≈ full-local). The 188 ms is the 5G radio moving ~1 MB — "
         "compression cuts it to 86 ms.",ha="center",color=INK,fontsize=11.5)
save(fig,"fig3_network_cost")

# ---------- FIG 4: E5 payload is not privacy ----------
pay=np.array([129.2,174.7,1050.3,2835.0]); ssim=np.array([0.703,0.733,0.716,0.72]);
roi_pay,roi_ssim=195.7,0.571; floor,ceil=0.326,0.979
fig,ax=plt.subplots(figsize=(8.4,5.2)); clean(ax); ax.set_xscale("log")
ax.axhline(floor,color=MUTED,ls=":",lw=1.2); ax.text(3350,floor-0.007,"no-information floor (0.33)",va="top",ha="right",color=MUTED,fontsize=10.5)
ax.axhline(ceil,color=MUTED,ls=":",lw=1.2); ax.text(3350,ceil-0.007,"raw image sent (0.98)",va="top",ha="right",color=MUTED,fontsize=10.5)
ax.plot(pay,ssim,"o",color=FULL,ms=11,label="Compression profiles (payload knobs)",zorder=5)
ax.plot([roi_pay],[roi_ssim],"D",color=SPLIT,ms=13,label="ROI-drop (deletes information)",zorder=6)
ax.annotate("22× smaller payload,\nsame ~0.71 recoverability",(pay[0],ssim[0]),xytext=(150,0.83),fontsize=11.5,
            color=FULL,fontweight="bold",arrowprops=dict(arrowstyle="->",color=FULL,lw=1.3))
ax.annotate("only ROI-drop\nlowers it (0.57)",(roi_pay,roi_ssim),xytext=(260,0.45),fontsize=11.5,
            color=SPLIT,fontweight="bold",arrowprops=dict(arrowstyle="->",color=SPLIT,lw=1.3))
ax.set_xlabel("Transmitted payload  (KB, log scale)"); ax.set_ylabel("Attacker's image recovery  (SSIM, 1 = perfect)")
ax.set_title("Compressing features does NOT hide the scene",loc="left")
ax.set_ylim(0.25,1.02); ax.set_xlim(90,3500); ax.legend(loc="lower left",fontsize=12)
save(fig,"fig4_privacy_not_free")

# ---------- FIG 5: three-way scorecard ----------
axes_lbl=["On-vehicle compute","Uplink bandwidth","End-to-end latency","Privacy (no imagery)","Enables cooperation"]
arch=["A. Full-local","B. Full-offload","C. Split (ours)"]
# 2=best/green, 1=ok/grey, 0=worst/red
M=np.array([[0,2,1],[2,0,0],[2,1,0],[1,0,1],[1,1,1]])
txt=[["heavy","none","light"],["tiny","medium","large"],["best","—","worst"],["ok","worst","weak"],["yes*","yes","yes"]]
cmap={2:GREEN,1:"#c9c9c9",0:VERM}
fig,ax=plt.subplots(figsize=(8.6,4.8))
for i in range(len(axes_lbl)):
    for j in range(3):
        ax.add_patch(plt.Rectangle((j,-i),0.96,0.92,color=cmap[M[i,j]],zorder=2))
        ax.text(j+0.48,-i+0.46,txt[i][j],ha="center",va="center",color="white" if M[i,j]!=1 else INK,
                fontweight="bold",fontsize=12.5)
ax.set_xlim(0,3); ax.set_ylim(-len(axes_lbl)+0.04,1.4)
ax.set_xticks([0.48,1.48,2.48]); ax.set_xticklabels(arch,fontweight="bold",fontsize=13)
ax.set_yticks([-i+0.46 for i in range(len(axes_lbl))]); ax.set_yticklabels(axes_lbl,fontsize=13)
ax.xaxis.tick_top(); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("No architecture wins everything — Split trades bandwidth & latency for on-vehicle compute",
             loc="left",fontsize=13.5,pad=26)
ax.text(0,-len(axes_lbl)+0.5,"* cooperation is available to all three; it is not unique to Split.",
        color=MUTED,fontsize=10.5,style="italic")
save(fig,"fig5_scorecard")

# ---------- FIG 6: model-weight scaling (illustrative extrapolation) ----------
k=np.linspace(1,10,50); budget_fps_full=17.79; budget_fps_split=44.32   # at 4 cores, measured 1x point
full_k=budget_fps_full/k; split_k=budget_fps_split/k
fig,ax=plt.subplots(figsize=(8.4,5.2)); clean(ax)
ax.axhspan(0,10,color=VERM,alpha=0.06,zorder=0); ax.axhline(10,color=MUTED,ls="--",lw=1.3)
ax.text(10,10.6,"10 FPS real-time floor",ha="right",color=MUTED,fontsize=11.5,style="italic")
ax.plot(k,split_k,color=SPLIT,lw=3,label="Split (car runs backbone only)")
ax.plot(k,full_k,color=FULL,lw=3,label="Full-local (car runs whole model)")
kf=budget_fps_full/10; ks=budget_fps_split/10
ax.plot([kf],[10],"o",color=FULL,ms=10); ax.plot([ks],[10],"o",color=SPLIT,ms=10)
ax.annotate(f"full-local drops below\nreal-time at ~{kf:.1f}× our model",(kf,10),xytext=(2.4,26),color=FULL,
            fontsize=11.5,fontweight="bold",arrowprops=dict(arrowstyle="->",color=FULL,lw=1.3))
ax.annotate(f"split holds to ~{ks:.1f}×",(ks,10),xytext=(5.2,17),color=SPLIT,fontsize=11.5,fontweight="bold",
            arrowprops=dict(arrowstyle="->",color=SPLIT,lw=1.3))
ax.set_xlabel("Perception model size  (× our lightweight model)",fontsize=13)
ax.set_ylabel("Sustained FPS  (4-core budget)",fontsize=12.5); ax.tick_params(labelsize=12)
ax.set_title("Sustained speed vs. model size (fixed 4-core budget)",loc="left",fontsize=15)
ax.set_ylim(0,48); ax.set_xlim(1,10); ax.legend(loc="upper right",fontsize=11.5)
fig.subplots_adjust(bottom=0.30,top=0.90,left=0.135,right=0.97)
fig.text(0.5,0.05,"Illustrative extrapolation: assumes speed ∝ 1/compute at a fixed budget, anchored to the measured 1× point.\n"
        "Modern BEV / transformer perception models are ~10–100× the FLOPs of ours.",ha="center",color=MUTED,fontsize=10,style="italic")
save(fig,"fig6_model_scaling",tight=False)
# ---------- FIG 7: E2 power / energy per frame ----------
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,5.0))
for ax in (a1,a2): clean(ax)
lbe=["Full-local","Split\n(front only)"]
ge=[0.441,0.290]
b=a1.bar(lbe,ge,color=[FULL,SPLIT],zorder=3); a1.set_ylim(0,0.55)
for r,v in zip(b,ge): a1.text(r.get_x()+r.get_width()/2,v+0.014,f"{v:.2f} J",ha="center",fontweight="bold",fontsize=13)
a1.text(1,0.40,"−34%",ha="center",color=SPLIT,fontweight="bold",fontsize=16)
a1.set_ylabel("GPU energy per frame  (Joules)"); a1.set_title("GPU energy / frame",loc="left",fontsize=14.5,pad=8)
cw=[178.7,64.8]
b=a2.bar(lbe,cw,color=[FULL,SPLIT],zorder=3); a2.set_ylim(0,215)
for r,v in zip(b,cw): a2.text(r.get_x()+r.get_width()/2,v+5,f"{v:.0f}",ha="center",fontweight="bold",fontsize=13)
a2.text(1,140,"−64%",ha="center",color=SPLIT,fontweight="bold",fontsize=16)
a2.set_ylabel("CPU work per frame  (core-milliseconds)"); a2.set_title("CPU work / frame (1 core)",loc="left",fontsize=14.5,pad=8)
fig.text(0.5,-0.04,"Split cuts on-vehicle energy per frame: −34% (GPU) to −64% (CPU). The absolute GPU delta is modest (~1.5 W at 10 FPS) and\n"
         "absolute watts do not transfer from this datacenter GPU to an embedded car chip — the percentage reduction is the transferable result.",
         ha="center",color=MUTED,fontsize=10.5,style="italic")
save(fig,"fig7_power_energy")

# ---------- FIG 8: E6 GPU-clock arm — does a GPU car change the trend? ----------
mhz=[2872,2085,1395,892,592,397,210]
gfull=[539.06,498.32,359.69,236.38,159.98,113.4,54.82]
gsplit=[757.58,760.15,592.96,372.46,264.39,185.59,91.66]
fig,ax=plt.subplots(figsize=(9.0,5.4)); clean(ax); ax.set_yscale("log")
ax.axhspan(1,10,color=VERM,alpha=0.05,zorder=0)
ax.axhline(30,color=MUTED,ls=":",lw=1.3); ax.axhline(10,color=MUTED,ls="--",lw=1.2)
ax.plot(mhz,gsplit,"-o",color=SPLIT,lw=3,ms=8,label="Split (backbone only)",zorder=5)
ax.plot(mhz,gfull,"-o",color=FULL,lw=3,ms=8,label="Full-local (whole model)",zorder=5)
ax.invert_xaxis()
ax.text(2960,31.5,"30 FPS",va="bottom",ha="left",color=MUTED,fontsize=11,style="italic")
ax.text(2960,10.4,"10 FPS",va="bottom",ha="left",color=MUTED,fontsize=11,style="italic")
ax.text(1650,45,"Even at 210 MHz (13× throttle), full-local = 55 FPS —\nstill above 30 FPS: no crossover on this GPU.",
        ha="center",va="center",color=FULL,fontsize=11.5,fontweight="bold")
ax.set_xlabel("GPU clock  (MHz)  —  weaker GPU →"); ax.set_ylabel("Sustained FPS  (log scale)")
ax.set_title("On a GPU, the light model stays real-time even heavily throttled",loc="left",fontsize=14.5)
ax.set_ylim(8,1100); ax.legend(loc="upper right",fontsize=12)
fig.subplots_adjust(bottom=0.26,top=0.90,left=0.10,right=0.97)
fig.text(0.5,0.045,"A throttled datacenter GPU is still far stronger than an embedded car GPU (Jetson). The split advantage still grows as the\n"
         "GPU weakens (1.4×→1.7×), but a real embedded-GPU crossover needs a Jetson run — see JETSON_EXPERIMENT_PLAN.md.",
         ha="center",color=MUTED,fontsize=10,style="italic")
save(fig,"fig8_gpu_budget",tight=False)

# ---------- FIG 9: compute TIME per frame — CPU vs GPU (two LINEAR panels; log would hide the ratio) ----------
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.5,5.0))
for ax in (a1,a2): clean(ax)
lbe9=["Full-local","Split"]
cf=[178.7,64.8]
b=a1.bar(lbe9,cf,color=[FULL,SPLIT],zorder=3); a1.set_ylim(0,208)
for r,v in zip(b,cf): a1.text(r.get_x()+r.get_width()/2,v+4,f"{v:g} ms",ha="center",fontweight="bold",fontsize=13)
a1.text(1,105,"−64%",ha="center",color=SPLIT,fontweight="bold",fontsize=17)
a1.set_ylabel("Compute time per frame  (ms)"); a1.set_title("CPU (1 core)",loc="left",fontsize=15,pad=8)
gf=[1.852,1.302]
b=a2.bar(lbe9,gf,color=[FULL,SPLIT],zorder=3); a2.set_ylim(0,2.15)
for r,v in zip(b,gf): a2.text(r.get_x()+r.get_width()/2,v+0.04,f"{v:g} ms",ha="center",fontweight="bold",fontsize=13)
a2.text(1,1.63,"−30%",ha="center",color=SPLIT,fontweight="bold",fontsize=17)
a2.set_ylabel("Compute time per frame  (ms)"); a2.set_title("GPU",loc="left",fontsize=15,pad=8)
fig.text(0.5,-0.03,"Note the different y-scales — CPU is ~100× slower than GPU. Split cuts CPU time 64% (2.8×) but GPU time only 30%: "
         "the compute\nbenefit concentrates on CPU-bound / weak-accelerator platforms. Clean CPU energy (RAPL) is pending a quiet host.",
         ha="center",color=MUTED,fontsize=10,style="italic")
save(fig,"fig9_compute_time")

# ---------- FIG 10: SCALABILITY — how many real-time (10 FPS) perception streams fit a CPU budget ----------
cores=[1,2,4,8]; full_fps=[5.49,9.96,17.79,29.49]; split_fps=[15.68,25.91,44.32,65.19]
full_s=[f/10 for f in full_fps]; split_s=[f/10 for f in split_fps]
fig,ax=plt.subplots(figsize=(9.0,5.4)); clean(ax)
xg=np.arange(len(cores)); w=0.38
b1=ax.bar(xg-w/2,full_s,w,color=FULL,label="Full-local (whole model)",zorder=3)
b2=ax.bar(xg+w/2,split_s,w,color=SPLIT,label="Split (backbone only)",zorder=3)
ax.axhline(1.0,color=MUTED,ls="--",lw=1.2); ax.text(3.4,1.08,"1 stream",ha="right",color=MUTED,fontsize=10.5,style="italic")
for r,v in zip(b1,full_s): ax.text(r.get_x()+r.get_width()/2,v+0.08,f"{v:.1f}",ha="center",fontweight="bold",fontsize=12,color=FULL)
for r,v in zip(b2,split_s): ax.text(r.get_x()+r.get_width()/2,v+0.08,f"{v:.1f}",ha="center",fontweight="bold",fontsize=12,color=SPLIT)
ax.set_xticks(xg); ax.set_xticklabels([f"{c} core{'s' if c>1 else ''}" for c in cores])
ax.set_ylabel("Concurrent real-time streams  (each ≥10 FPS)"); ax.set_ylim(0,7.2)
ax.set_xlabel("On-device compute budget"); ax.set_title("How many perception streams fit the device? (CPU)",loc="left")
ax.legend(loc="upper left",fontsize=12)
fig.text(0.5,-0.03,"Numbers = sustained FPS ÷ 10 (stream capacity). At 1–2 cores full-local can't sustain even one stream; split sustains 1–2.\n"
         "On a GPU both fit dozens (full 53 / split 76) — scalability bites on compute-poor devices. Upper bound: ignores concurrency overhead.",
         ha="center",color=MUTED,fontsize=10,style="italic")
save(fig,"fig10_scalability_streams")

# ---------- FIG 11: RUNTIME on a fixed compute-energy budget (illustrative multiplier) ----------
fig,ax=plt.subplots(figsize=(8.6,5.2)); clean(ax)
labs=["Full-local","Split\n(GPU)","Split\n(CPU)"]; mult=[1.0,0.441/0.290,178.7/64.8]; cols=[FULL,SPLIT,SPLIT]
b=ax.bar(labs,mult,color=cols,zorder=3); ax.set_ylim(0,3.3)
for r,v in zip(b,mult): ax.text(r.get_x()+r.get_width()/2,v+0.07,f"{v:.1f}×",ha="center",fontweight="bold",fontsize=15)
ax.set_ylabel("Relative operating time",fontsize=13); ax.tick_params(labelsize=12)
ax.set_title("Split runs longer on the same on-device energy budget",loc="left",fontsize=15)
fig.subplots_adjust(bottom=0.30,top=0.90,left=0.12,right=0.96)
fig.text(0.5,0.05,"Same perception for less compute energy (GPU 0.44→0.29 J/frame; CPU 179→65 core-ms) → ~1.5× (GPU) to ~2.8× (CPU) longer.\n"
   "Illustrative: assumes perception compute dominates; absolute hours (AR glasses ~2–5 Wh, drone ~50–100 Wh) need embedded power.",
   ha="center",color=MUTED,fontsize=9.5,style="italic")
save(fig,"fig11_runtime_energy",tight=False)

print("done ->", OUT)
