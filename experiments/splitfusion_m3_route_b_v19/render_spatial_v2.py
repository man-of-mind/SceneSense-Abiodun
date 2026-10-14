"""Use the existing presentation Deck; add only the approved boundary and spatial labels."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse
from rl_agent.splitfusion_m3_presentation_v1 import render as P
from rl_agent.splitfusion_m3_route_b_v19 import contract as C


def main(args):
    args.out.mkdir(exist_ok=False)
    C.approved_design(ROOT)
    approved_png=ROOT/C.APPROVED_DIR/'01_proposed_spatial_snr_region.png'
    png_before=C.sha(approved_png)
    P.style()
    settings=SimpleNamespace(analysis=args.analysis,layers=args.layers,out=args.out,
        label='M3 spatial · V19 · one capture',highlight_window=None,gc_trace=None,
        radio_metadata=None,hide_context_markers=True)
    deck=P.Deck(settings)
    regions=json.loads((args.regions/'SUMMARY.json').read_text())
    rows=P.read_csv(args.regions/'DECISIONS_SPATIAL.csv')
    exposure=np.array([r['spatial_region'] for r in rows])
    assert len(rows)==len(deck.dec)
    original_finish=deck.finish
    def finish(fig,name,note):
        names={'01_completion_cdf.png':'03_prediction_feedback_cdf.png',
               '02_map_freshness_cdf.png':'04_map_freshness_cdf.png',
               '11_latency_map_freshness_time.png':'05_latency_map_freshness_time.png'}
        original_finish(fig,names.get(name,name),note)
    deck.finish=finish

    fig=deck.new('Spatial-channel demonstration on Town10',
        'Red: approved low core · Dashed: transition boundary · Points: actual UE feedback outcomes')
    ax=fig.add_axes([.055,.17,.48,.64]);deck.route_base(ax)
    # Include the whole approved outer ellipse, not only the driven route's extent.
    ax.set(xlim=(-165,135),ylim=(-90,210))
    ax.add_patch(Ellipse((-55,140),200,112.5,fill=False,edgecolor='#d28a71',linestyle='--',linewidth=1.6,zorder=4))
    ax.add_patch(Ellipse((-55,140),160,90,facecolor='#d93e3e',edgecolor='none',alpha=.08,zorder=1))
    ax.add_patch(Ellipse((-55,140),160,90,fill=False,edgecolor='#ce3038',linewidth=2.4,zorder=5))
    for mask,color,label in ((deck.timely,P.TEAL,'Feedback ≤170 ms'),(deck.late,P.ORANGE,'Late feedback'),(deck.missing,P.RED,'Missing feedback')):
        ax.scatter(*deck.xy[mask].T,c=color,s=18,lw=.2,edgecolor='white',zorder=3,label=label)
    ax.legend(loc='lower right',fontsize=10)
    ax.text(-55,200,'LOW CORE · 12–14 dB',ha='center',color='#b52a33',size=11,weight='bold')
    fig.text(.57,.78,'REGION PERFORMANCE',size=16,weight='bold')
    fig.text(.57,.73,'All policy decisions in each ACK-pose region',size=12,color=P.MUTED)
    for top,region,color in ((.64,'HIGH',P.TEAL),(.47,'TRANSITION','#ce8846'),(.30,'LOW_CORE','#c93035')):
        g=regions['regions'][region];n=g['n']
        fig.text(.57,top,region.replace('_',' ')+f'  ·  n={n:,}',size=17,weight='bold',color=color)
        fig.text(.57,top-.054,f"Timely feedback: {g['timely_feedback']}/{n} ({g['timely_feedback']/n:.1%})" if n else 'No decisions',size=17)
        fig.text(.57,top-.095,f"Timely SCORED: {g['timely_scored_feedback']}/{n} · Prediction ≤170: {g['timely_predictions']}/{n}",size=11.5,color=P.MUTED)
    deck.finish(fig,'01_spatial_region_outcomes.png',
        'Synthetic position-indexed impairment, not measured coverage or building-derived propagation. Grey: planned route; star: observed start.\nRegions use the causal pose behind each acknowledged command. Feedback deadline and map admission remain independent.')

    fig=deck.new('Channel, actions and payload — aligned',
        f'{len(deck.dec):,} policy decisions · Red shading: LOW_CORE · Tan shading: TRANSITION · No smoothing')
    axes=fig.subplots(6,1,sharex=True,gridspec_kw=dict(left=.15,right=.945,bottom=.13,top=.82,hspace=.24))
    t=deck.t
    for ax in axes:
        for region,color in (('LOW_CORE','#d93e3e'),('TRANSITION','#e7bda8')):
            mask=exposure==region
            starts=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))).reshape(-1,2)
            for first,last in starts:
                right=t[last] if last<len(t) else t[-1]
                ax.axvspan(t[first],right,color=color,alpha=.12,zorder=0)
        ax.grid(color=P.GRID,lw=.6);ax.tick_params(labelsize=10)
    axes[0].scatter(t,deck.mode,s=6,c=P.TEAL);axes[0].set_yticks(sorted(set(deck.mode)))
    axes[1].plot(t,deck.q,color=P.TEAL,lw=.8);axes[1].set(ylim=(0,1),yticks=[0,.5,1])
    axes[2].plot(t,P.array(deck.dec,'payload_l2_bytes')/1000,color=P.BLUE,lw=.8)
    axes[3].scatter(t,deck.snr,s=4,color=P.PURPLE,alpha=.6,label='Measured PUSCH (preceding transmission)')
    axes[3].plot(t,deck.commanded,color=P.INK,lw=1,ls='--',label='ACKed commanded SNR / actor proxy')
    axes[3].set(ylim=(5.5,25.5),yticks=[8.5,14,23.5]);axes[3].legend(loc='lower left',ncol=2,fontsize=8)
    axes[4].scatter(t,deck.mcs,s=4,color=P.PURPLE,label='Measured final grant MCS')
    axes[4].plot(t,P.array(deck.dec,'prior_ul_mcs'),color=P.INK,lw=.6,alpha=.6,label='Actor-observed prior MCS')
    axes[4].set(ylim=(0,29),yticks=[0,14,28]);axes[4].legend(loc='lower left',ncol=2,fontsize=8)
    axes[5].plot(t,deck.feedback,color=P.ORANGE,lw=.7);axes[5].axhline(170,color=P.INK,lw=1,ls='--')
    for ax,label in zip(axes,('Mode\n(category)','Drop fraction\nq','Payload\n(kB)','SNR\n(dB)','MCS\n(index)','UE feedback\n(ms)')):
        ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=14,size=12)
    axes[-1].set(xlim=(0,t[-1]+1),xlabel='Elapsed wall time from first policy-frame capture (s)')
    deck.finish(fig,'02_channel_mode_q_payload.png',
        'Measured PUSCH SNR and grant MCS are diagnostics, not actor inputs; joins use preceding telemetry observations, not packet identity.\nq=q_e4/10,000 (larger means more spatial dropping); kB=1,000 application bytes. Feedback is action-open→UE receipt; dashed line: original 170-ms deadline.')
    deck.completion();deck.freshness_cdf();deck.freshness_timeline()
    deck.paths.extend([args.regions/'SUMMARY.json',args.regions/'DECISIONS_SPATIAL.csv',Path(__file__),approved_png])
    deck.hashes.update({str(p.resolve()):P.sha(p) for p in deck.paths})
    deck.summary['spatial_regions']=regions['regions']
    deck.summary['synthetic_position_indexed_not_coverage']=True
    deck.export()
    assert C.sha(approved_png)==png_before
    print(json.dumps(dict(status='PASS',figures=deck.figs,approved_png_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('analysis','layers','regions','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
