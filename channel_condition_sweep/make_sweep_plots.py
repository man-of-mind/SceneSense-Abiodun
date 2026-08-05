#!/usr/bin/env python3
"""Channel-sweep presentation plots — clean, bold, minimal text (dataviz method).
Reads the 3 per-payload summary CSVs, emits presentation figures:
standard PNG/PDF plus SVG vector and 600-DPI PNG for PowerPoint.
Palette: reference categorical (P1 blue, P2 orange, P3 aqua) + sequential blue heatmap."""
import os, numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE="uplink_only_spatial_map_pipeline/results/"
OUT="channel_condition_sweep/plots/"; os.makedirs(OUT, exist_ok=True)
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#e6e5e2"
P1,P2,P3="#2a78d6","#eb6834","#1baf7a"
PAYS=[("chsweep_full_p1u8_","1 MB",P1),("chsweep_full_p2u4_","400 KB",P2),("chsweep_full_p3ae_","90 KB",P3)]
RUNGS=["clear","mild","mid15","strong"]
XLAB=["Clear\n50 dB","Mild\n19.5 dB","Mid\n15.6 dB","Strong\n8.2 dB"]
FIGSIZE=(12.8,7.2)  # fixed 16:9 landscape page for slide/PDF import

mpl.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.size":14,"font.family":"DejaVu Sans","text.color":INK,"axes.labelcolor":INK,
    "xtick.color":INK2,"ytick.color":INK2,"axes.edgecolor":GRID,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

def load():
    d={}
    for key,lab,_ in PAYS:
        df=pd.read_csv(BASE+key+".csv").set_index("profile")
        d[lab]=df.reindex(RUNGS)
    return d

def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(length=0); ax.set_axisbelow(True)

def title(fig,t,sub):
    fig.text(0.045,0.935,t,fontsize=18,fontweight="bold",color=INK,ha="left")
    fig.text(0.045,0.890,sub,fontsize=12,color=INK2,ha="left")

def save(fig,name):
    # Do not use bbox_inches="tight": fixed 16:9 page avoids portrait/cropped PDF behavior in slide tools.
    fig.savefig(OUT+name+".png",dpi=220)
    fig.savefig(OUT+name+"_600dpi.png",dpi=600)
    fig.savefig(OUT+name+".pdf")
    fig.savefig(OUT+name+".svg")
    plt.close(fig)
    print("wrote",OUT+name+".{png,pdf,svg} and "+OUT+name+"_600dpi.png")

d=load(); x=np.arange(4)

# ---- FIG 1: latency knee (log-y lines) ----
fig,ax=plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(top=0.76,left=0.12,right=0.86,bottom=0.16)
for lab,col in [("1 MB",P1),("400 KB",P2),("90 KB",P3)]:
    y=d[lab]["capture_to_map_publish_p50_ms"].values
    ax.plot(x,y,color=col,lw=2.6,marker="o",ms=9,mfc=col,mec=SURF,mew=1.6,zorder=3,clip_on=False)
    ax.text(x[-1]+0.10,y[-1],lab,color=col,fontsize=15,fontweight="bold",va="center")
ax.set_yscale("log"); ax.set_ylim(60,30000)
ax.set_yticks([100,1000,10000]); ax.set_yticklabels(["100 ms","1 s","10 s"])
ax.axhspan(60,300,color=P3,alpha=0.06,zorder=0)  # "usable" band, no text clutter
ax.set_xticks(x); ax.set_xticklabels(XLAB); ax.set_xlim(-0.2,3.6)
ax.grid(axis="y",color=GRID,lw=1); style(ax); ax.set_ylabel("capture → map latency (p50)")
title(fig,"Payload latency knee under degrading channel",
      "Uplink-only over 5G · SINR MCS · ~6–8 fps offered · lower is better")
save(fig,"fig1_latency_knee")

# ---- FIG 2: delivery heatmap ----
M=np.array([d[lab]["edge_delivery_pct"].values for lab,_ in [("1 MB",0),("400 KB",0),("90 KB",0)]],dtype=float)
blue=LinearSegmentedColormap.from_list("bl",["#f3f6fb","#2a78d6","#123a6b"])
fig,ax=plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(top=0.76,left=0.14,right=0.92,bottom=0.17)
im=ax.imshow(M,cmap=blue,vmin=0,vmax=100,aspect="auto")
for i in range(3):
    for j in range(4):
        v=M[i,j]; ax.text(j,i,f"{v:.0f}%",ha="center",va="center",fontsize=17,fontweight="bold",
                          color="white" if v>55 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels(XLAB); ax.set_yticks(range(3)); ax.set_yticklabels(["1 MB","400 KB","90 KB"])
ax.tick_params(length=0); [ax.spines[s].set_visible(False) for s in ax.spines]
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02,ticks=[0,50,100]); cb.ax.set_yticklabels(["0%","50%","100%"]); cb.outline.set_visible(False)
title(fig,"Delivery holds only where payload fits",
      "Fresh-frame delivery over 5G · dark = reliable, pale = congestion collapse")
save(fig,"fig2_delivery_heatmap")

# ---- FIG 3: payload budget @10fps ----
cap=np.array([max(d[lab]["ul_sched_mbps"].reindex(RUNGS).values[j] for lab,_ in [("1 MB",0),("400 KB",0),("90 KB",0)]) for j in range(4)])
budget_kb=cap*1e6/8/10/1024  # capacity/10fps -> KB per frame
fig,ax=plt.subplots(figsize=FIGSIZE); fig.subplots_adjust(top=0.76,left=0.12,right=0.86,bottom=0.16)
ax.bar(x,budget_kb,width=0.62,color="#cfe0f4",edgecolor=P1,lw=1.6,zorder=2)
for j in range(4): ax.text(x[j],budget_kb[j]+8,f"{budget_kb[j]:.0f} KB",ha="center",fontsize=13,fontweight="bold",color=P1)
for kb,lab,col in [(1024,"1 MB",P1),(400,"400 KB",P2),(90,"90 KB",P3)]:
    ax.axhline(kb,color=col,lw=2,ls=(0,(5,3)),zorder=1,xmax=0.85)
    ax.text(3.52,kb,lab,color=col,fontsize=13,fontweight="bold",va="center",
            bbox=dict(fc=SURF,ec="none",pad=1.5))
ax.set_ylim(0,1150); ax.set_xticks(x); ax.set_xticklabels(XLAB); ax.set_xlim(-0.5,3.9)
ax.grid(axis="y",color=GRID,lw=1); style(ax); ax.set_ylabel("affordable payload per frame @ 10 fps")
title(fig,"Payload budget @10 fps: 90 KB fits every rung",
      "budget = scheduled-UL capacity ÷ 10 fps · a payload is deliverable only below its line")
save(fig,"fig3_payload_budget")

# ---- FIG 4: summary bars, closed-loop-policy-style ----
def sec_or_ms(v):
    return f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"

def bar_label(col,val):
    if not np.isfinite(val):
        return None
    if col=="delivery_pct":
        return None if val >= 99.5 else f"{val:.0f}%"
    if col=="cap2map_p50_ms":
        return sec_or_ms(val)
    if col in {"app_offered_mbps","sched_ul_mbps"}:
        return f"{val:.0f}" if val >= 10 else f"{val:.1f}"
    if col=="bsr_p95_MiB":
        return f"{val:.0f}" if val >= 10 else f"{val:.1f}"
    return f"{val:.0f}"

surf=pd.read_csv("channel_condition_sweep/combined_surface.csv")
payload_order=["1 MB","400 KB","90 KB"]
payload_cols={"1 MB":P1,"400 KB":P2,"90 KB":P3}
snr_order=[50.3,19.5,15.6,8.2]
snr_labels=["Clear","Mild","Mid","Strong"]

metrics=[
    ("delivery_pct","Delivery","%",False,(0,105)),
    ("cap2map_p50_ms","Latency","p50, log ms",True,(70,30000)),
    ("app_offered_mbps","App offered","Mbps",False,(0,75)),
    ("sched_ul_mbps","Scheduled UL","Mbps",False,(0,42)),
    ("bsr_p95_MiB","UE backlog","BSR p95 MiB",False,(0,52)),
    ("mcs","MCS","p50 index",False,(0,30)),
]
fig,axes=plt.subplots(2,3,figsize=FIGSIZE)
fig.subplots_adjust(top=0.865,left=0.06,right=0.992,bottom=0.085,wspace=0.20,hspace=0.30)
width=0.32
for ax,(col,ttl,ylab,logy,ylim) in zip(axes.flat,metrics):
    for i,payload in enumerate(payload_order):
        vals=[]
        for snr in snr_order:
            rows=surf[(surf["payload"]==payload) & (np.isclose(surf["snr"],snr,atol=0.25))]
            vals.append(float(rows[col].iloc[0]) if len(rows) else np.nan)
        offs=x+(i-1)*width
        bars=ax.bar(offs,vals,width=width,color=payload_cols[payload],label=payload,zorder=3)
        if col != "mcs":
            for bar,val in zip(bars,vals):
                txt=bar_label(col,val)
                if txt is None:
                    continue
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),txt,
                        ha="center",va="bottom",fontsize=7.3,fontweight="bold",
                        rotation=90 if col=="cap2map_p50_ms" else 0,
                        clip_on=False,zorder=5)
    if logy:
        ax.set_yscale("log")
        ax.set_yticks([100,1000,10000])
        ax.set_yticklabels(["100ms","1s","10s"])
    ax.set_ylim(*ylim)
    if col=="mcs":
        for j,snr in enumerate(snr_order):
            rows=surf[np.isclose(surf["snr"],snr,atol=0.25)]
            if len(rows):
                val=float(np.nanmedian(rows["mcs"]))
                ax.text(x[j],val+0.8,f"{val:.0f}",ha="center",va="bottom",
                        fontsize=8.0,fontweight="bold",clip_on=False,zorder=5)
    ax.set_title(ttl,fontsize=14.0,fontweight="bold",pad=8)
    ax.set_ylabel(ylab,fontsize=11.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(snr_labels,fontsize=11.5,fontweight="bold")
    ax.grid(axis="y",color=GRID,lw=1,zorder=0)
    style(ax)
    ax.tick_params(axis="both",labelsize=11.5,width=1.25)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
handles,labels=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(0.64,0.965),ncol=3,
           frameon=False,prop={"weight":"bold","size":12.5},handlelength=1.9,columnspacing=2.6)
fig.text(0.035,0.955,"Channel sweep",fontsize=20,fontweight="bold",color=INK,ha="left",va="top")
save(fig,"fig4_sweep_summary_bars")

# ---- FIG 5: horizontal latency breakdown by payload × channel ----
break_rows=[]
for _,payload,_ in PAYS:
    df=d[payload].copy()
    for rung,xlab in zip(RUNGS,XLAB):
        r=df.loc[rung]
        front=float(r["front_build_p50_ms"])
        transport=float(r["front_to_edge_p50_ms"])
        tail=float(r["tail_p50_ms"])
        total=float(r["capture_to_map_publish_p50_ms"])
        map_pub=max(0.0,total-front-transport-tail)
        break_rows.append({
            "label": f"{payload}\n{xlab.replace(chr(10),' ')}",
            "payload": payload,
            "front build": front,
            "front→edge": transport,
            "edge tail": tail,
            "map publish": map_pub,
            "total": total,
        })
break_df=pd.DataFrame(break_rows)
break_cols=[
    ("front build","#9E9E9E"),
    ("front→edge","#56B4E9"),
    ("edge tail","#CC79A7"),
    ("map publish","#F0E442"),
]
fig,ax=plt.subplots(figsize=FIGSIZE)
fig.subplots_adjust(top=0.895,left=0.105,right=0.992,bottom=0.085)
y=np.arange(len(break_df))
left=np.zeros(len(break_df))
for col,color in break_cols:
    vals=break_df[col].to_numpy()
    ax.barh(y,vals,left=left,height=0.74,color=color,label=col,zorder=3)
    for idx,val in enumerate(vals):
        if col=="front→edge" and val >= 1000:
            ax.text(left[idx]+val/2,idx,f"{val:,.0f}",ha="center",va="center",
                    fontsize=8.5,fontweight="bold",color=INK,zorder=5)
    left += vals
for idx,total in enumerate(break_df["total"].to_numpy()):
    label=sec_or_ms(total)
    ax.text(total+260,idx,label,ha="left",va="center",fontsize=8.6,fontweight="bold",color=INK,zorder=5)
ax.set_yticks(y)
ax.set_yticklabels(break_df["label"],fontsize=9.8,fontweight="bold")
for sep in (3.5,7.5):
    ax.axhline(sep,color="#000000",lw=2.2,ls=(0,(5,4)),alpha=0.95,zorder=1)
ax.invert_yaxis()
ax.set_xlim(0,17000)
ax.set_xlabel("p50 latency component (ms)",fontsize=12,fontweight="bold")
ax.set_title("Latency breakdown by payload and channel",fontsize=18,fontweight="bold",pad=10)
ax.grid(axis="x",color=GRID,lw=1,zorder=0)
ax.legend(loc="lower right",frameon=False,ncol=4,
          prop={"weight":"bold","size":10.5},handlelength=1.7,columnspacing=1.8)
style(ax)
ax.tick_params(axis="both",labelsize=10.5,width=1.25)
for tick in ax.get_xticklabels()+ax.get_yticklabels():
    tick.set_fontweight("bold")
save(fig,"fig5_latency_breakdown_by_payload_channel")
print("done")
