"""Bounded saved-evidence uplink audit. No live objects, production writes, or tests."""
from __future__ import annotations
import ast,collections as C,csv,hashlib,importlib.util,json,math,os,re,sys,time
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2];DEP=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else BASE
REFRESH=len(sys.argv)>2 and sys.argv[2]=='--refresh-unsealed'
RX=BASE.parent/'receiver_boundary_diagnostic_20261013_01'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
R=module(RX/'analyze.py','receiver_audit');H=R.H
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(OUT/'mpl_cache'))
import numpy as np
doc,lines,rows,q=H.doc,H.lines,H.rows,H.q
CAP=DEP/'experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_08'
SINGLE=DEP/'rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_m3_route_b_v11/good_all_prospective_01'
def write(name,obj):
    with (OUT/name).open('w' if REFRESH else 'x') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
def table(name,data):
    fields=list(dict.fromkeys(k for r in data for k in r))
    with (OUT/name).open('w' if REFRESH else 'x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
# Exact retained timestamp parser, streamed for the 92-MB single-UE RLC file.
cp=H.bound(ROOT/'rl_agent/splitfusion_hybrid_sac_live_route_b_v2/ue_telemetry_provider_v2.py')
tree=ast.parse(cp.read_text());ns=dict(re=re,DAY_US=86400000000,MAX_INGEST_US=2000000,
    _TOD=re.compile(r'^([0-2][0-9]):([0-5][0-9]):([0-5][0-9])\.([0-9]{6})$'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'parse_time_of_day_us','reconstruct_event_realtime_ns'}],type_ignores=[]),str(cp),'exec'),ns)
def streamed(path,native=True):
    fields=None
    with H.bound(path).open() as f:
        for a in csv.reader(f):
            if len(a)>2 and a[2]=='time':
                fields=(['ingest_raw_ns','ingest_wall_ns'] if native else ['ingest_wall_ns','ingest_raw_ns'])+a[2:];continue
            if fields is None or len(a)!=len(fields):continue
            r=dict(zip(fields,a));wall=int(r['ingest_wall_ns'])
            r['event_wall_ns']=ns['reconstruct_event_realtime_ns'](r['time'],SimpleNamespace(real_ns=wall,utc_offset_s=-7*3600))
            yield r
def ticks(path):
    groups={};current={};out=[]
    def flush(ue):
        g=groups.get(ue)
        if g:out.append(dict(ue_id=ue,rnti=int(next(iter(g.values()))['rnti']),frame=int(next(iter(g.values()))['frame']),
            slot=int(next(iter(g.values()))['slot']),wall_ns=max(v['event_wall_ns'] for v in g.values()),
            bytes=sum(int(v['bytes_in_buffer']) for v in g.values())))
    for r in streamed(path):
        ue=int(r['ue_id']);key=(int(r['rnti']),r['frame'],r['slot'])
        if current.get(ue)!=key:flush(ue);groups[ue]={};current[ue]=key
        groups[ue][r['lcid']]=r
    # Last tick may be partial: intentionally exclude it.
    for ue in {v['ue_id'] for v in out}:
        seq=sorted((v for v in out if v['ue_id']==ue),key=lambda v:v['wall_ns'])
        offset=0;previous=None
        for v in seq:
            n=20*v['frame']+v['slot']
            if previous is not None and n<previous-10240:offset+=20480
            v['radio_slot']=n+offset;previous=n
    return sorted(out,key=lambda t:t['wall_ns'])
def prior_tick(tt,wall):
    a=[t for t in tt if t['wall_ns']<=wall]
    return a[-1] if a else None
def interval_rate(n,lo,hi):return n*8e3/(hi-lo)
def resources(grants,tt,lo,hi):
    g=[r for r in grants if lo<=r['event_wall_ns']<=hi and r['direction']=='1']
    a=prior_tick(tt,lo);b=prior_tick(tt,hi);assert a and b
    wall=(b['wall_ns']-a['wall_ns'])/1e9;radio=(b['radio_slot']-a['radio_slot'])*.0005
    nnew=sum(r['round']=='0' for r in g);tb=sum(int(r['tbs']) for r in g);newtb=sum(int(r['tbs']) for r in g if r['round']=='0')
    return dict(grants=len(g),new_grants=nnew,retransmission_grants=len(g)-nnew,scheduled_tb_bytes=tb,new_tb_bytes=newtb,
        scheduled_tb_Mbps=interval_rate(tb,lo,hi),new_tb_Mbps=interval_rate(newtb,lo,hi),grants_per_wall_s=len(g)/((hi-lo)/1e9),
        radio_elapsed_s=radio,clock_observation_wall_s=wall,radio_per_wall=radio/wall,
        clock_start_wall_ns=a['wall_ns'],clock_end_wall_ns=b['wall_ns'],
        tb_bytes=q(int(r['tbs']) for r in g),prb=q(int(r['rb_size']) for r in g),
        symbols=dict(C.Counter(r['nr_symbols'] for r in g)),mcs=dict(C.Counter(r['mcs'] for r in g)),
        slots=dict(C.Counter(r['sched_slot'] for r in g)),dci_formats=dict(C.Counter(r['dci_format'] for r in g)),
        mcs_tables=dict(C.Counter(r['mcs_table'] for r in g)),median_full_width_grant_tb_bytes=q(int(r['tbs']) for r in g if r['rb_size']=='273' and r['nr_symbols']=='13')['p50'])
def main():
    start=time.monotonic();os.nice(19);os.sched_setaffinity(0,{23})
    assert not H.live_busy(),H.live_busy()
    OUT.mkdir(exist_ok=True)
    if REFRESH:assert OUT==BASE and not (BASE/'FINAL_SHA256.json').exists()
    assert REFRESH or not (OUT/'SUMMARY.json').exists(),'Refusing to overwrite prior analysis: use a new output directory.'
    for p in [Path(__file__),RX/'analyze.py',RX.parent/'smoke_diagnosis_20261013_01/analyze.py']:H.bound(p)
    frames=rows(RX/'analysis_01/FRAME_TIMING.csv');chunks=rows(RX/'analysis_01/CHUNK_TIMING.csv')
    sources=doc(CAP/'ATTACHED.json')['sources'];audit=lines(CAP/'edge/RECEIVER_TIMING.jsonl')
    sizes={s['stream_id']:doc(CAP/s['stream_id']/'SENT_FRAME_SIZES.json') for s in sources}
    clock=R.Clock(audit,[r['pair'] for v in sizes.values() for r in v])
    for f in frames:
        for k in ('frame_id','capture_wall_ns','send_start_wall_ns','last_packet_send_raw_ns','payload_bytes','chunk_count'):
            f[k]=int(f[k])
        f['last_send_wall_ns']=clock.to_wall(f['last_packet_send_raw_ns'])
        f['udp_bytes']=f['payload_bytes']+8*f['chunk_count']
        f['delay_ms']=float(f['last_send_to_kernel_ms']) if f['last_send_to_kernel_ms'] else None
    lo=min(f['send_start_wall_ns'] for f in frames);hi=max(f['last_send_wall_ns'] for f in frames)
    close=max(r['pair']['wall_ns'] for r in audit if 'pair' in r)
    for c in chunks:
        c['kernel_wall_ns']=int(c['kernel_wall_ns']);c['datagram_bytes']=int(c['datagram_bytes'])
    pusch=H.radio_reader(CAP/'pusch/ttracer/live_pusch_with_ingest.csv')
    grants=H.radio_reader(CAP/'native_radio/NRUE_MAC_DCI_GRANT_live.csv',True)
    sched=H.radio_reader(CAP/'pusch/ttracer/live_mcs_with_ingest.csv');tt=ticks(CAP/'native_radio/NRUE_MAC_RLC_BUFFER_STATUS_live.csv')
    pdcp=list(streamed(CAP/'native_radio/NR_PDCP_TX_SDU_live.csv'))
    bins=[];summaries={};joined_grants=[]
    for source in sources:
        s=source['stream_id'];f=[r for r in frames if r['stream_id']==s];cc=[r for r in chunks if r['stream_id']==s]
        own=[r for r in tt if r['ue_id']==source['ue_id'] and r['rnti']==source['rnti']]
        gg=[r for r in grants if int(r['rnti'])==source['rnti']]
        pp=[r for r in pusch if int(r['rnti'])==source['rnti']]
        mm=[r for r in sched if int(r['rnti'])==source['rnti']]
        sender=doc(CAP/s/'SENDER_BINDING.json');assert sum(r['udp_bytes'] for r in f)==sender['bytes']
        b0=prior_tick(own,lo);b1=prior_tick(own,hi);bc=prior_tick(own,close)
        rx=[r for r in cc if lo<=r['kernel_wall_ns']<=hi]
        pg=[r for r in pp if lo<=r['event_wall_ns']<=hi];mg=[r for r in mm if lo<=r['event_wall_ns']<=hi]
        pd=[r for r in pdcp if int(r['ue_id'])==source['ue_id'] and lo<=r['event_wall_ns']<=hi]
        res=resources(gg,own,lo,hi)
        completed=[r for r in f if r['completing_kernel_wall_ns'] and lo<=int(r['completing_kernel_wall_ns'])<=hi]
        offered=sum(r['udp_bytes'] for r in f);received=sum(r['datagram_bytes'] for r in rx);delta=b1['bytes']-b0['bytes']
        packetbacklog=offered-received
        bykind={k:dict(frames=sum(r['kind']==k for r in f),feature_bytes=sum(r['payload_bytes'] for r in f if r['kind']==k),
                udp_bytes=sum(r['udp_bytes'] for r in f if r['kind']==k)) for k in sorted({r['kind'] for r in f})}
        out=dict(source=source,window_wall_ns=[lo,hi],window_s=(hi-lo)/1e9,sent=len(f),offered_by_kind=bykind,
            feature_bytes=sum(r['payload_bytes'] for r in f),offered_udp_bytes=offered,received_udp_bytes=received,
            received_feature_chunk_bytes=received-8*len(rx),completed_feature_bytes=sum(r['payload_bytes'] for r in completed),
            offered_feature_Mbps=interval_rate(sum(r['payload_bytes'] for r in f),lo,hi),
            offered_udp_Mbps=interval_rate(offered,lo,hi),received_udp_Mbps=interval_rate(received,lo,hi),
            received_feature_chunks_Mbps=interval_rate(received-8*len(rx),lo,hi),
            completed_feature_goodput_Mbps=interval_rate(sum(r['payload_bytes'] for r in completed),lo,hi),
            pdcp_sdu_bytes=sum(int(r['sdu_bytes']) for r in pd),pdcp_events=len(pd),
            app_udp_outstanding_at_window_end_bytes=packetbacklog,rlc_start_bytes=b0['bytes'],rlc_end_bytes=b1['bytes'],
            rlc_delta_bytes=delta,rlc_close_bytes=bc['bytes'],rlc_capture_bytes=q(r['bytes'] for r in own if lo<=r['wall_ns']<=hi),
            rlc_boundary_age_ms=[(lo-b0['wall_ns'])/1e6,(hi-b1['wall_ns'])/1e6],
            outstanding_minus_rlc_delta_bytes=packetbacklog-delta,
            layer_balance_caveat='Different layers, fragmentation/PDCP/RLC/MAC headers, in-flight data and asynchronously sampled endpoints; not an exact per-packet RLC conservation equation.',
            receiver_close_wall_ns=close,received_udp_bytes_by_close=sum(r['datagram_bytes'] for r in cc),
            last_send_to_kernel_ms=q(r['delay_ms'] for r in f),
            resources=res,steady_resources=resources(gg,own,lo+8_000_000_000,hi),
            pusch_snr_db=q(float(r['snrx10'])/10 for r in pg),scheduler_avg_snr_db=q(float(r['avg_snr_x10'])/10 for r in mg),
            pusch_txpower_calc=dict(C.Counter(r['txpower_calc'] for r in pg)),
            pusch_tpc=dict(C.Counter(r['tpc'] for r in pg)),ph=q(r['ph'] for r in mg),
            phr_mcs_reductions=sum(int(r['post_phr_mcs'])<int(r['pre_phr_mcs']) for r in mg),
            force_mcs=dict(C.Counter(r['force_ul_mcs'] for r in mg)),
            pusch_tb_bytes=sum(int(r['tb_size']) for r in pg),pusch_tb_Mbps=interval_rate(sum(int(r['tb_size']) for r in pg),lo,hi),
            selected_vs_final_mcs_changes=sum(r['selected_mcs']!=r['final_mcs'] for r in mg))
        summaries[s]=out
        for r in gg:
            if lo<=r['event_wall_ns']<=hi and r['direction']=='1':joined_grants.append(dict(stream_id=s,**r))
        for j in range(math.ceil((close-lo)/1e9)):
            left=lo+j*1_000_000_000;right=min(close,left+1_000_000_000);t0=prior_tick(own,left);t1=prior_tick(own,right)
            ff=[r for r in f if left<=r['last_send_wall_ns']<right];cr=[r for r in cc if left<=r['kernel_wall_ns']<right]
            gr=[r for r in gg if r['direction']=='1' and left<=r['event_wall_ns']<right]
            cumf=sum(r['udp_bytes'] for r in f if r['last_send_wall_ns']<right)
            cumc=sum(r['datagram_bytes'] for r in cc if r['kernel_wall_ns']<right)
            rr=(t1['radio_slot']-t0['radio_slot'])*.0005/((t1['wall_ns']-t0['wall_ns'])/1e9) if t0 and t1 and t1['wall_ns']!=t0['wall_ns'] else None
            bins.append(dict(stream_id=s,bin_start_wall_ns=left,bin_end_wall_ns=right,seconds=j,
                offered_feature_Mbps=interval_rate(sum(r['payload_bytes'] for r in ff),left,right),
                scheduled_new_tb_Mbps=interval_rate(sum(int(r['tbs']) for r in gr if r['round']=='0'),left,right),
                received_chunk_feature_Mbps=interval_rate(sum(r['datagram_bytes']-8 for r in cr),left,right),
                outstanding_udp_MB=(cumf-cumc)/1e6,rlc_MB=(t1['bytes']-b0['bytes'])/1e6 if t1 else None,
                delay_p50_ms=q(r['delay_ms'] for r in ff).get('p50'),radio_per_wall=rr))
    # Aggregate radio opportunities and per-UE scheduling share, fully covered frames.
    nonempty=[r for r in joined_grants if r['direction']=='1']
    shares=C.Counter();byframe=C.defaultdict(list)
    for r in nonempty:
        # no SFN wrap inside this short recording; assert rather than silently infer
        shares[r['stream_id']]+=int(r['rb_size'])*int(r['nr_symbols'])
        byframe[int(r['sched_frame'])].append(r)
    frameids=sorted(byframe);assert frameids[-1]-frameids[0]<1024
    fullframes=list(range(frameids[0]+1,frameids[-1]))
    slots=[]
    for slot in range(20):
        values=[sum(int(r['rb_size'])*int(r['nr_symbols']) for r in byframe[fr] if int(r['sched_slot'])==slot) for fr in fullframes]
        opportunity=3 if slot%10==4 else 13 if slot%10>=5 else 0
        slots.append(dict(slot=slot,configured_role='MIXED' if slot%10==4 else 'UL' if slot%10>=5 else 'DL',
            max_pusch_prb_symbols=273*opportunity,observed_mean_prb_symbols=float(np.mean(values)),
            occupied_frames=sum(v>0 for v in values),fully_covered_frames=len(fullframes)))
    aggregate=dict(prb_symbol_shares={s:v/sum(shares.values()) for s,v in shares.items()},
        slots=slots,full_width_13_symbol_tbs_MCS25_bytes=22547,reachable_slots_per_radio_frame=10,
        configured_ul_full_slots_per_frame=10,configured_mixed_slots_per_frame=2,
        radio_opportunity_note='Two 5-ms periods/frame; configured 5 full UL + mixed, but k2=6 full-slot TDAs cannot reach first UL slots 5/15.',
        nominal_reachable_saturated_TB_bytes_per_radio_frame=8*22547+2*4867,
        unused_full_slot_uplift_arithmetic_only=(10*22547+2*4867)/(8*22547+2*4867)-1)
    # Compatible single-UE V11: same retained radio profile/mapping, different load/session.
    sc=SINGLE/'capture';cell=sc/'cells/a71__favorable_stable'
    single_frames=[r for r in rows(cell/'per_frame_metrics.csv') if r['prepare_status']=='SENT']
    slo=min(int(float(r['capture_wall_s'])*1e9) for r in single_frames)
    shi=max(int(float(r['capture_wall_s'])*1e9) for r in single_frames)
    sr=H.radio_reader(sc/'m3_radio/ttracer/live_pusch_with_ingest.csv')
    sg=H.radio_reader(cell/'run4_phase6/telemetry_live/NRUE_MAC_DCI_GRANT_live.csv',True)
    st=ticks(cell/'run4_phase6/telemetry_live/NRUE_MAC_RLC_BUFFER_STATUS_live.csv')
    seen={int(r['rnti']) for r in sr if slo<=r['event_wall_ns']<=shi};assert len(seen)==1
    srnti=seen.pop();st=[r for r in st if r['rnti']==srnti];sg=[r for r in sg if int(r['rnti'])==srnti]
    single={}
    for label,end in [('full_capture',shi),('first_matched_wall_duration',slo+hi-lo)]:
        ff=[r for r in single_frames if slo<=float(r['capture_wall_s'])*1e9<=end]
        pp=[r for r in sr if int(r['rnti'])==srnti and slo<=r['event_wall_ns']<=end]
        single[label]=dict(window_wall_ns=[slo,end],duration_s=(end-slo)/1e9,sent=len(ff),
            offered_feature_Mbps=interval_rate(sum(int(r['payload_bytes']) for r in ff),slo,end),
            payload_bytes=q(r['payload_bytes'] for r in ff),pusch_snr_db=q(float(r['snrx10'])/10 for r in pp),
            mcs=q(r['mcs'] for r in pp),rlc_bytes=q(r['bytes'] for r in st if slo<=r['wall_ns']<=end),
            resources=resources(sg,st,slo,end),received_goodput_unknown='No per-datagram edge timestamps in this reference.',
            note='Payload keyed to recorded capture WALL, not exact send WALL; unmatched scene/session/one ego; descriptive only.')
    single['radio_materialization']=doc(sc/'m3_radio/radio_state_copy/radio_materialization.json')
    single['rnti']=srnti
    # Read-only identity checks; RFsim source/library not retrospectively called sealed.
    prep=doc(DEP/'experiments/splitfusion_m4_two_ue_v1/preparation_smoke_20261013_11/PREPARATION.json')
    names=['radio/rfsimulator/simulator.cpp','radio/rfsimulator/apply_channelmod.c',
        'openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_ulsch.c','openair2/LAYER2/NR_MAC_gNB/gNB_scheduler_primitives.c',
        'openair2/LAYER2/NR_MAC_gNB/nr_radio_config.c','openair2/LAYER2/NR_MAC_gNB/main.c',
        'openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c','openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.c',
        'cmake_targets/ran_build/build/nr-softmodem','cmake_targets/ran_build/build/nr-uesoftmodem',
        'cmake_targets/ran_build/build/librfsimulator.so']
    bindings={}
    for name in names:
        name='OAI/openairinterface5g/'+name;a=H.bound(ROOT/name);b=H.bound(DEP/name)
        digest=H.INPUTS[str(a)]['sha256'];assert digest==H.INPUTS[str(b)]['sha256']
        registered=prep['sources'].get(name)
        if registered:assert registered==digest
        bindings[name]=dict(sha256=digest,main_deploy_agree=True,registered_sha256=registered,
            scope='sealed preparation match' if registered else 'current main/deployment agreement; not in historical source inventory')
    for p in [CAP/'gnb.log',CAP/'ue.log',CAP/'gnb.ownership.json',CAP/'ue.ownership.json',CAP/'COMMANDS.jsonl',
        CAP/'radio_work/gnb/nrMAC_stats.log',CAP/'radio_work/gnb/nrL1_stats.log',
        DEP/'experiments/splitfusion_m4_two_ue_v1/preparation_smoke_20261013_11/gnb.conf',
        DEP/'experiments/splitfusion_m4_two_ue_v1/preparation_smoke_20261013_11/ue.conf',
        ROOT/'experiments/splitfusion_phase14a_100mhz_calibration_v1/20260904_phase14a_live_calibration_retry3/target_to_rfsim_mapping.csv']:
        H.bound(p)
    summary=dict(verdict='OFFLINE_SERVICE_DEFICIT_EXPLAINED_NO_SMOKE_REQUIRED',scope='One new offline audit directory; no runtime/configuration changes or live attempt',
        aligned_m4_window_wall_ns=[lo,hi],aligned_m4_window_seconds=(hi-lo)/1e9,
        per_ue=summaries,allocation=aggregate,single_ue_reference=single,clock_uncertainty=clock.summary,
        snr=dict(commanded_db=23.5,per_channel_command=-12.,predicted_two_independent_equal_noise_penalty_db=10*math.log10(2),
            predicted_two_ue_snr_db=23.5-10*math.log10(2),
            mechanism='Source: each connected modeled UE adds independent per-sample Gaussian noise into the same gNB receive sum, including zero-signal intervals. No global noise_power_dBFS enabled.',
            caveat='Source-supported and quantitatively consistent, not isolated by a controlled noise intervention; RFsim shared-library historical hash not retained.'),
        recommendation='One future controlled wall-clock byte-admission cap per UE, keeping only latest unsent frame, below measured service with headroom; do not add another deep queue. No implementation here.',
        unresolved='Why the RFsim/PHY clock is this slow at instruction/thread level; packet-level residence within sender/radio/core; causal effect of a fix.')
    write('SUMMARY.json',summary);write('SOURCE_BINDING.json',bindings)
    table('ALIGNED_1S.csv',bins);table('ALLOCATION_BY_SLOT.csv',slots);table('UL_GRANTS.csv',joined_grants)
    table('FRAME_DELAY.csv',[{k:r[k] for k in ('stream_id','frame_id','kind','payload_bytes','udp_bytes','send_start_wall_ns','last_send_wall_ns','delay_ms')} for r in frames])
    table('PER_UE.csv',[dict(stream_id=s,window_start_wall_ns=lo,window_end_wall_ns=hi,window_s=(hi-lo)/1e9,sent=v['sent'],
        offered_feature_Mbps=v['offered_feature_Mbps'],offered_udp_Mbps=v['offered_udp_Mbps'],
        scheduled_dci_tb_Mbps=v['resources']['scheduled_tb_Mbps'],new_dci_tb_Mbps=v['resources']['new_tb_Mbps'],
        pusch_observed_tb_Mbps=v['pusch_tb_Mbps'],received_chunk_feature_Mbps=v['received_feature_chunks_Mbps'],
        completed_feature_goodput_Mbps=v['completed_feature_goodput_Mbps'],
        rlc_start_MB=v['rlc_start_bytes']/1e6,rlc_end_MB=v['rlc_end_bytes']/1e6,rlc_delta_MB=v['rlc_delta_bytes']/1e6,
        sent_minus_rx_udp_MB=v['app_udp_outstanding_at_window_end_bytes']/1e6,
        radio_seconds_per_wall_second=v['resources']['radio_per_wall'],
        scheduling_prb_symbol_share=aggregate['prb_symbol_shares'][s],pusch_snr_median_db=v['pusch_snr_db']['p50'],
        last_send_kernel_p50_ms=v['last_send_to_kernel_ms']['p50']) for s,v in summaries.items()])
    plots(summary,bins,frames)
    write('INPUT_SHA256.json',H.INPUTS)
    write('REPRODUCIBILITY.json',dict(command='env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B '+str(Path(__file__).relative_to(ROOT))+' '+' '.join(sys.argv[1:]),
        cpu_affinity=sorted(os.sched_getaffinity(0)),elapsed_s=time.monotonic()-start,
        note='Create-only outputs; optional positional argument selects a NEW output directory. No live services imported.'))
    print(json.dumps(dict(verdict=summary['verdict'],per_ue={s:dict(offered=v['offered_feature_Mbps'],radio_rtf=v['resources']['radio_per_wall'],backlog_delta=v['rlc_delta_bytes']) for s,v in summaries.items()},single=single['full_capture']),indent=2))
def plots(summary,bins,frames):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':15,'axes.titlesize':19,'axes.labelsize':15,'legend.fontsize':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#146C9E','#D47720'];lo,hi=summary['aligned_m4_window_wall_ns'];duration=(hi-lo)/1e9
    fig,axes=plt.subplots(3,2,figsize=(19.2,10.8),dpi=200,sharex=True)
    for i,(s,v) in enumerate(summary['per_ue'].items()):
        b=[r for r in bins if r['stream_id']==s];x=[r['seconds']+1 for r in b]
        axes[0,i].plot(x,[r['outstanding_udp_MB'] for r in b],label='Cumulative UDP sent − received',color=colors[i],lw=2.4)
        axes[0,i].plot(x,[r['rlc_MB'] for r in b],label='UE RLC backlog change',color='#303846',ls='--',lw=1.8)
        axes[0,i].set_title(f"UE {i}: offered {v['offered_feature_Mbps']:.2f} Mbit/s; received chunks {v['received_feature_chunks_Mbps']:.2f} Mbit/s",pad=12)
        axes[0,i].set_ylabel('Outstanding data (MB)');axes[0,i].legend(loc='upper left')
        for key,label,color in [('offered_feature_Mbps','Offered feature payload',colors[i]),
            ('scheduled_new_tb_Mbps','Scheduled new-data TB','#888888'),('received_chunk_feature_Mbps','Received feature chunks','#248A66')]:
            axes[1,i].step(x,[r[key] for r in b],where='pre',label=label,color=color,lw=1.7)
        axes[1,i].set_ylabel('Rate (Mbit/s; 1-s bins)');axes[1,i].legend(loc='upper left')
        f=[r for r in frames if r['stream_id']==s and r['delay_ms'] is not None]
        axes[2,i].plot([(r['last_send_wall_ns']-lo)/1e9 for r in f],[r['delay_ms']/1000 for r in f],color=colors[i],lw=2.1)
        axes[2,i].set_ylabel('Last send → kernel RX (s)');axes[2,i].set_xlabel('Wall seconds since first send')
        missing=[r for r in frames if r['stream_id']==s and r['delay_ms'] is None]
        axes[2,i].scatter([(r['last_send_wall_ns']-lo)/1e9 for r in missing],[.03]*len(missing),transform=axes[2,i].get_xaxis_transform(),marker='|',color='#A8324B',label='Incomplete at shutdown (rug)')
        axes[2,i].legend(loc='upper left')
        for ax in axes[:,i]:ax.axvline(duration,ls=':',color='#555555');ax.grid(alpha=.2)
    fig.suptitle('Offered bytes outpace service; the growing deficit appears in UE RLC queues',fontsize=24,y=.985)
    fig.text(.5,.025,'Same WALL-clock windows. Layer byte totals are compared, never added. Scheduled TB is not application goodput.\nDotted line: last send. Subsequent drain is censored at receiver close; missing arrivals are not proven loss.',ha='center',fontsize=14)
    fig.subplots_adjust(left=.075,right=.98,top=.88,bottom=.13,wspace=.22,hspace=.22)
    fig.savefig(OUT/'01_LOAD_BACKLOG_DELAY.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(19.2,10.8),dpi=200)
    slots=summary['allocation']['slots']
    axes[0,0].bar([r['slot'] for r in slots],[r['max_pusch_prb_symbols']/273 for r in slots],color='#DCE1E6',label='Configured PUSCH-compatible symbols')
    axes[0,0].bar([r['slot'] for r in slots],[r['observed_mean_prb_symbols']/273 for r in slots],color='#276B91',width=.55,label='Granted PRB-symbols / 273')
    axes[0,0].set_title('First full UL slot of each period is unreachable');axes[0,0].set_xlabel('Slot within 10-ms NR frame (μ = 1)')
    axes[0,0].set_ylabel('PRB-normalized PUSCH symbols');axes[0,0].set_xticks(range(20));axes[0,0].legend(fontsize=10)
    for slot in (5,15):axes[0,0].annotate('No grant',xy=(slot,.3),xytext=(slot,5),ha='center',color='#A8324B',arrowprops=dict(arrowstyle='->',color='#A8324B'))
    b=[r for r in bins if r['stream_id']=='m4-ue0' and r['radio_per_wall'] is not None]
    axes[0,1].plot([r['seconds'] for r in b],[r['radio_per_wall'] for r in b],color='#276B91',label='Two-UE NR-clock / WALL rate')
    ref=summary['single_ue_reference']['full_capture']['resources']['radio_per_wall']
    axes[0,1].axhline(ref,color='#238965',ls='--',label=f'Single-UE V11 whole-run mean {ref:.3f} (different session)')
    axes[0,1].set_title('Radio time advances far slower than wall time');axes[0,1].set_xlabel('Wall seconds since first send')
    axes[0,1].set_ylabel('NR seconds per wall second');axes[0,1].legend(fontsize=11);axes[0,1].grid(alpha=.2)
    names=[];rates=[];shares=[]
    for s,v in summary['per_ue'].items():
        names.append('UE '+str(v['source']['ue_id']));rates.append(v['resources']['grants_per_wall_s']);shares.append(100*summary['allocation']['prb_symbol_shares'][s])
    axes[1,0].bar(names,rates,color=colors,width=.55)
    for j,n in enumerate(names):axes[1,0].text(j,rates[j]+1,f'{rates[j]:.1f} grants/s\n{shares[j]:.1f}% of PRB-symbols',ha='center',fontsize=17)
    axes[1,0].set_ylim(0,max(rates)*1.3);axes[1,0].set_ylabel('UL grants per wall second')
    axes[1,0].set_title('Approximately equal share, not UE starvation')
    v11=summary['single_ue_reference']['full_capture']
    values=[23.5,v11['pusch_snr_db']['p50'],*[v['pusch_snr_db']['p50'] for v in summary['per_ue'].values()],summary['snr']['predicted_two_ue_snr_db']]
    labels=['Command','V11 single UE','M4 UE0','M4 UE1','Two-noise\nprediction']
    axes[1,1].bar(labels,values,color=['#B7BDC5','#238965',*colors,'#858296'])
    for j,v in enumerate(values):axes[1,1].text(j,v+.2,f'{v:.2f}',ha='center',fontsize=15)
    axes[1,1].set_ylim(0,27);axes[1,1].set_ylabel('SNR (dB)');axes[1,1].set_title('Two per-link noise terms predict −3.01 dB')
    fig.suptitle('Why service is low: slow radio clock, shared grants, and reduced spectral efficiency',fontsize=24,y=.985)
    fig.text(.5,.028,'100 MHz / 273 PRBs is configured bandwidth, not continuous wall-clock service. One layer; MCS 25 dominant.\nNoise mechanism is source-supported, not a controlled live test. Single-UE comparison has different payload, session and duration.',ha='center',fontsize=14)
    fig.subplots_adjust(left=.075,right=.98,top=.88,bottom=.13,wspace=.2,hspace=.45)
    fig.savefig(OUT/'02_RADIO_RESOURCES.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()

