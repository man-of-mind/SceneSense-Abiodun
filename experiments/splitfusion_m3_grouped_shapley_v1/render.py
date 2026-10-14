"""Six create-only 3840x2160 presentation PNGs; no actor queries."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import BASE,old,write
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ORDER=['scene','channel','backlog','history']
LABEL={'scene':'Scene SI + P40','channel':'Channel SNR + MCS','backlog':'RLC backlog','history':'Previous decision / outcome'}
COLOR={'scene':'#D78326','channel':'#2672AE','backlog':'#5C7C77','history':'#98468D'}
TITLE={11:'M3.1  GOOD_ALL',12:'M3.2  BAD_ALL',13:'M3.3  GOOD_TO_BAD',14:'M3.4  BAD_TO_GOOD'}
INK='#142431';LIGHT='#DFE5EA'

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':15,'axes.titleweight':'bold',
        'axes.labelsize':13,'axes.labelweight':'bold','xtick.labelsize':11,'ytick.labelsize':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#7D8C98',
        'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
        'legend.frameon':False,'legend.fontsize':11,'figure.facecolor':'white'})

def save(fig,path):
    fig.savefig(path,dpi=200,facecolor='white');plt.close(fig)
    from PIL import Image
    with Image.open(path) as im:assert im.size==(3840,2160),im.size

def summary_plot(result,out):
    fig,axes=plt.subplots(1,2,figsize=(19.2,10.8))
    fig.subplots_adjust(left=.065,right=.975,bottom=.27,top=.79,wspace=.18)
    for ax,metric,unit in zip(axes,('margin11','q11'),('Mode 11 preference | logit margin','Mode 11 conditional q | fraction')):
        x=np.arange(4);far=[]
        for gi,g in enumerate(ORDER):
            main=[next(s for s in result['summary'] if s['run']==v and s['question']=='common_primary' and s['output']==metric) for v in (11,12,13,14)]
            alt=[next(s for s in result['summary'] if s['run']==v and s['question']=='common_alternative' and s['output']==metric) for v in (11,12,13,14)]
            ax.bar(x+(gi-1.5)*.19,[s['mean_absolute'][g] for s in main],.18,color=COLOR[g])
            ax.scatter(x+(gi-1.5)*.19,[s['mean_absolute'][g] for s in alt],marker='_',s=130,c=INK,zorder=4)
        labs=[f"{TITLE[v].replace('  ', chr(10))}\nn=100; far {100*main[i]['far_slots']/main[i]['slots']:.0f}%" for i,v in enumerate((11,12,13,14))]
        ax.set_xticks(x,labs,fontsize=14);ax.set_ylabel('Mean absolute grouped contribution',fontsize=16)
        ax.set_title(unit,fontsize=19,pad=18);ax.grid(axis='y',color=LIGHT);ax.set_axisbelow(True)
        ax.tick_params(axis='y',labelsize=14)
    fig.suptitle('What changes the frozen actor output across the full Town10 map?',fontsize=27,fontweight='bold',y=.97)
    fig.text(.5,.895,'QUESTION A  •  Output relative to one common recorded-state background',ha='center',fontsize=19)
    handles=[Patch(color=COLOR[g],label=LABEL[g]) for g in ORDER]+[Line2D([0],[0],marker='_',color=INK,lw=0,markersize=12,label='Alternative fixed background')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.11),ncol=3,fontsize=14)
    fig.text(.5,.07,'100 uniformly spaced decisions per condition; 8 equal-weight references: 2 per session, 4 GOOD + 4 BAD. Extras excluded.',ha='center',fontsize=14)
    fig.text(.5,.04,'Mean |background-averaged Shapley contribution|, not percentages of causation. Far = nonendpoint coalition slots outside recorded-state proximity.',ha='center',fontsize=13)
    fig.text(.5,.015,'Both backgrounds are empirical, not training distributions. Conditional q11 is not always executed. No quality, latency or live causal inference.',ha='center',fontsize=13)
    save(fig,out/'01_four_condition_summary.png')

def signed_bars(ax,rs,metric,width):
    t=np.array([r['t_s'] for r in rs]);pos=np.zeros(len(rs));neg=np.zeros(len(rs))
    for g in ORDER:
        y=np.array([r[metric+'_'+g] for r in rs]);base=np.where(y>=0,pos,neg)
        ax.bar(t,y,width=width,bottom=base,color=COLOR[g],align='center',linewidth=0)
        pos+=np.maximum(y,0);neg+=np.minimum(y,0)
    ax.axhline(0,color=INK,lw=.7)
    far=[r['t_s'] for r in rs if r['far_nonendpoint_slots']]
    if far:ax.scatter(far,np.full(len(far),.93),transform=ax.get_xaxis_transform(),c='#B12634',marker='x',s=24,linewidths=1.2,zorder=8)

def timeline(ver,allrecords,rows,selection,out):
    rr=[r for r in allrecords if r['run']==ver];index={r['index']:r for r in rr}
    reps=selection['selection'][str(ver)]['representative'];win=selection['selection'][str(ver)]['transition_window']
    panels=[('Whole capture | 100 representative targets',rr,[r for r in rows if r['run']==ver and r['question']=='adjacent' and r['representative']])]
    if win:
        panels.append(('Recorded transition window | 60 decisions',[index[i] for i in win],
            [r for r in rows if r['run']==ver and r['question']=='adjacent' and r['transition_window']]))
    fig=plt.figure(figsize=(19.2,10.8))
    gs=fig.add_gridspec(9,len(panels),height_ratios=[.44,.8,.7,.65,.7,.43,.65,1.05,1.05],
        width_ratios=[1.65,1] if win else [1],left=.112,right=.951,bottom=.10,top=.825,hspace=.19,wspace=.27)
    labels=['Mode\ncategory','Executed q\n[0–1]','Payload\n[kB]','Commanded\nSNR [dB]','SI / P40\nencoded','Backlog\n[kB]','Previous\nhistory','Δ mode 11\nlogit margin','Δ q11\nfraction']
    trans=selection['transitions'][str(ver)];transition_t=None if trans is None else index[trans]['t_s']
    leftaxs=[]
    for col,(title,rrr,att) in enumerate(panels):
        axs=[]
        for y in range(9):
            ax=fig.add_subplot(gs[y,col],sharex=axs[0] if axs else None);axs.append(ax)
            if col==0:ax.set_ylabel(labels[y],rotation=0,ha='right',va='center',labelpad=21,fontsize=12)
            if y<8:ax.tick_params(labelbottom=False)
            ax.grid(axis='y',color=LIGHT,alpha=.7);ax.set_axisbelow(True);ax.tick_params(axis='y',labelsize=10,pad=2)
        t=np.array([r['t_s'] for r in rrr]);axs[0].set_title(title,fontsize=14,pad=10)
        for mode,colr in [(8,'#458A79'),(9,'#C27229'),(10,'#8155A4'),(11,'#246C9F')]:
            ts=[r['t_s'] for r in rrr if r['mode']==mode]
            axs[0].scatter(ts,[mode-8]*len(ts),s=7 if col==0 else 17,color=colr,marker='s')
        axs[0].set_yticks([0,1,2,3],['8','9','10','11']);axs[0].set_ylim(-.5,3.5)
        axs[1].plot(t,[r['q_e4']/10000 for r in rrr],color='#235F92',lw=.7 if col==0 else 1.2)
        q11=[r for r in att if r['mode']!=11]
        axs[1].scatter([r['t_s'] for r in q11],[r['q11_target'] for r in q11],facecolors='none',edgecolors='#A92F70',marker='D',s=35,lw=1.2,zorder=6)
        axs[1].set_ylim(0,1);axs[1].set_yticks([0,.5,1])
        axs[2].plot(t,np.array([r['payload_sfd4_bytes'] for r in rrr])/1000,color='#586C7C',lw=.75 if col==0 else 1.2)
        axs[2].set_ylim(0,310);axs[2].set_yticks([0,150,300])
        axs[3].step(t,[r['commanded_snr_db'] for r in rrr],where='post',color=COLOR['channel'],lw=1.5)
        axs[3].set_ylim(5,27);axs[3].set_yticks([8.5,23.5])
        mcs=axs[3].twinx();mcs.spines['right'].set_visible(True)
        mcs.plot(t,[r['prior_mcs'] for r in rrr],color='#777D84',ls='--',lw=.75);mcs.set_ylim(0,30);mcs.set_yticks([9,27]);mcs.tick_params(labelsize=10)
        if col==len(panels)-1:mcs.set_ylabel('Prior MCS\n[index]',fontsize=11)
        axs[4].plot(t,[r['features'][0] for r in rrr],color='#D78326',lw=1,label='SI scaled')
        axs[4].plot(t,[r['features'][1] for r in rrr],color='#148B96',lw=1,label='Radar P40')
        if col==0:axs[4].legend(loc='upper right',ncol=2,fontsize=9)
        axs[5].plot(t,np.array([r['backlog_bytes'] for r in rrr])/1000,color=COLOR['backlog'],lw=.8)
        axs[5].set_ylim(0,max(r['backlog_bytes'] for r in rr)/1000*1.15+1);axs[5].set_yticks([0,round(max(r['backlog_bytes'] for r in rr)/1000)])
        hist={'INITIAL_EMPTY':(0,'#686868'),'CLEARED_AFTER_TIMELY_UNDEFINED':(1,'#B46D1D'),
            'TIMEOUT_HISTORY':(2,'#725593'),'DEFINED_HISTORY':(3,'#278476')}
        for name,(y,c) in hist.items():
            ht=[r['t_s'] for r in rrr if r['history']==name];axs[6].scatter(ht,[y]*len(ht),c=c,s=7 if col==0 else 17,marker='s')
        axs[6].set_yticks([0,1,2,3],['Initial','Cleared','Timeout','Defined']);axs[6].set_ylim(-.5,3.5)
        width=(t[-1]-t[0])/max(1,len(att))*.48
        signed_bars(axs[7],att,'margin11',width);signed_bars(axs[8],att,'q11',width)
        axs[8].set_xlabel('Time since first decision [s]',fontsize=12)
        if col==0:leftaxs=axs
        else:
            for k in (4,7,8):
                lo=min(leftaxs[k].get_ylim()[0],axs[k].get_ylim()[0]);hi=max(leftaxs[k].get_ylim()[1],axs[k].get_ylim()[1])
                leftaxs[k].set_ylim(lo,hi);axs[k].set_ylim(lo,hi)
        for ax in axs:
            ax.set_xlim(t[0]-.5,t[-1]+.5)
            if transition_t is not None:ax.axvline(transition_t,color='#25333D',ls=':',lw=1.2,zorder=5)
        if transition_t is not None:
            axs[0].text(transition_t,.98,' first new-SNR decision',transform=axs[0].get_xaxis_transform(),fontsize=9,va='top')
    suffix=' | original registered STOP unchanged' if ver==13 else ''
    fig.suptitle(TITLE[ver]+'  •  full Town10 map'+suffix,fontsize=24,fontweight='bold',y=.98)
    fig.text(.5,.925,f"Observed decisions n={len(rr)}; representative explanations n=100 (99 predecessor contrasts). q11 diamonds only where mode≠11.",ha='center',fontsize=13)
    fig.text(.5,.895,'QUESTION B  •  Signed contributions explain change from the immediately preceding decision, not a global baseline.',ha='center',fontsize=14)
    handles=[Patch(color=COLOR[g],label=LABEL[g]) for g in ORDER]+[Line2D([0],[0],color='#B12634',marker='x',lw=0,label='Far coalition(s) retained')]
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.5,.88),fontsize=11)
    fig.text(.5,.035,'Bars only at explained times; gaps are not interpolated. Commanded SNR is an actor input; measured PUSCH is not shown and is not substituted.',ha='center',fontsize=12)
    fig.text(.5,.012,'Payload is observed SFD4 bytes (kB=1,000 bytes), not an attributed output. Cleared/timeout zero fields are unavailable placeholders. No performance or live causal claim.',ha='center',fontsize=12)
    save(fig,out/f'{ver-9:02d}_M3_{ver-10}_timeline.png')

def waterfalls(rows,selection,out):
    specs=[('G1','Constant GOOD: large q rise'),('G2','Constant GOOD: backlog-associated fall'),
        ('B2','Constant BAD: timeout history'),('G6','Constant GOOD: actual mode 9→8'),
        ('T13I','Immediate GOOD→BAD'),('T14I','Immediate BAD→GOOD and reset')]
    fig,axes=plt.subplots(6,2,figsize=(19.2,10.8));fig.subplots_adjust(left=.09,right=.985,bottom=.085,top=.855,hspace=.82,wspace=.18)
    names=['Ref','Scene','Channel','Backlog','History','Target'];colors=['#9AA6B1']+[COLOR[g] for g in ORDER]+['#263D4E']
    for row,(label,title) in enumerate(specs):
        case=next(c for c in selection['cases'] if c['case']==label)
        r=next(r for r in rows if r['question']=='adjacent' and r['run']==case['run'] and r['index']==case['index'])
        mode=r['mode'];far=r['far_nonendpoint_slots'];prefix='selected_' if mode!=11 else ''
        metrics=['selected_margin','selected_conditional_q'] if mode!=11 else ['margin11','q11']
        for col,metric in enumerate(metrics):
            ax=axes[row,col];start=r[metric+'_reference'];end=r[metric+'_target'];parts=[r[metric+'_'+g] for g in ORDER]
            vals=[start]+parts+[end];bottom=[0];running=start
            for v in parts:bottom.append(running);running+=v
            bottom.append(0)
            for i,(v,b,c) in enumerate(zip(vals,bottom,colors)):
                ax.bar(i,v,.67,bottom=b,color=c)
                y=b+v;ax.annotate((f'{v:+.3f}' if 0<i<5 else f'{v:.3f}'),(i,y),xytext=(0,3 if v>=0 else -3),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=11,fontweight='bold')
            levels=[0,start,end]+[sum([start]+parts[:k]) for k in range(1,5)]
            span=max(levels)-min(levels);pad=max(span*.25,.01);ax.set_ylim(min(levels)-pad,max(levels)+pad)
            ax.set_xticks(range(6),names if row==5 else ['']*6,fontsize=12)
            ax.tick_params(axis='y',labelsize=9);ax.axhline(0,color=INK,lw=.5)
            unit='logit margin' if col==0 else 'conditional q'
            ax.set_title(f"{label} · {title} · mode {mode} {unit}  |  far {far}/14",fontsize=12,color='#AB2738' if far else INK,pad=6)
            if col==0:ax.set_ylabel(f"f{case['reference_frame']}\n→ f{case['frame']}",rotation=0,ha='right',va='center',fontsize=11,labelpad=12)
    fig.suptitle('Exact step explanations: the same actor can respond in different directions',fontsize=25,fontweight='bold',y=.98)
    fig.text(.5,.93,'Reference = preceding recorded decision. Each waterfall adds all four groups to reach the target output.',ha='center',fontsize=16)
    fig.legend(handles=[Patch(color=COLOR[g],label=LABEL[g]) for g in ORDER],ncol=4,loc='upper center',bbox_to_anchor=(.5,.915),fontsize=13)
    fig.text(.5,.045,'Red titles flag weak empirical support, not failed arithmetic. Every coalition stays included. Mode-8 heads are shown for the actual non-11 choice.',ha='center',fontsize=13)
    fig.text(.5,.017,'Conditional q is not necessarily the executed q at the reference. Group contributions do not isolate individual history fields or predict payload, quality or latency.',ha='center',fontsize=12)
    save(fig,out/'06_waterfall_examples.png')

def main(selection_dir,attribution_dir,out):
    out.mkdir(exist_ok=False);style()
    selection=json.loads((selection_dir/'SELECTION.json').read_text());states=json.loads((selection_dir/'RECORDED_STATES.json').read_text())
    result=json.loads((attribution_dir/'RESULTS.json').read_text())
    import gzip
    with gzip.open(attribution_dir/'ATTRIBUTIONS.json.gz','rt') as f:rows=json.load(f)
    summary_plot(result,out)
    for v in (11,12,13,14):timeline(v,states,rows,selection,out)
    waterfalls(rows,selection,out)
    pngs=list(out.glob('*.png'));assert len(pngs)==6
    write(out/'OUTPUT_SHA256.json',dict(source_sha256=old.sha(__file__),
        selection_sha256=old.sha(selection_dir/'SELECTION.json'),results_sha256=old.sha(attribution_dir/'RESULTS.json'),
        attribution_sha256=old.sha(attribution_dir/'ATTRIBUTIONS.json.gz'),
        png_sha256={p.name:old.sha(p) for p in sorted(pngs)},dimensions=[3840,2160]))
    print('Rendered six 3840x2160 PNGs',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection-dir',type=Path,required=True);p.add_argument('--attribution-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.selection_dir,a.attribution_dir,a.out)
