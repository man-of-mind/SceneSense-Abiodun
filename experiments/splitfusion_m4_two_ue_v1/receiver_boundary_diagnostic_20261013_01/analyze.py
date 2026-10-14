"""CPU-only saved-evidence diagnostic; run from repository, optional NEW output path."""
import collections as C, csv, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else BASE/'analysis_01'
REFRESH=len(sys.argv)>2 and sys.argv[2]=='--refresh-unsealed'
CAP=ROOT.parent/'abiodun_run4_single_gpu_profiler_v1/experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_08'
os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
spec=importlib.util.spec_from_file_location('retained_readers',BASE.parent/'smoke_diagnosis_20261013_01/analyze.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
doc,lines,rows,q,ms=H.doc,H.lines,H.rows,H.q,H.ms
import numpy as np
def write(name,value):
    with (OUT/name).open('w' if REFRESH else 'x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
def csvwrite(name,values):
    fields=list(dict.fromkeys(k for r in values for k in r))
    with (OUT/name).open('w' if REFRESH else 'x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
class Clock:
    def __init__(self,audit,sender_pairs):
        pairs=[r['pair'] for r in audit if 'pair' in r]
        pairs=sorted(pairs,key=lambda p:p['raw_before_ns'])
        self.r0=(pairs[0]['raw_before_ns']+pairs[0]['raw_after_ns'])//2
        self.w0=pairs[0]['wall_ns']
        self.x=np.array([((p['raw_before_ns']+p['raw_after_ns'])//2-self.r0) for p in pairs],dtype=np.float64)
        self.y=np.array([p['wall_ns']-self.w0 for p in pairs],dtype=np.float64)
        assert np.all(np.diff(self.x)>0) and np.all(np.diff(self.y)>0)
        train=np.unique(np.r_[np.arange(0,len(pairs),2),len(pairs)-1])
        hold=np.setdiff1d(np.arange(len(pairs)),train)
        residual=self.y[hold]-np.interp(self.x[hold],self.x[train],self.y[train])
        bracket=np.array([p['raw_after_ns']-p['raw_before_ns'] for p in pairs])
        sender_residual=[]
        for p in sender_pairs:
            mid=(p['raw_before_ns']+p['raw_after_ns'])//2
            assert self.x[0]<=mid-self.r0<=self.x[-1]
            sender_residual.append(p['wall_ns']-self.to_wall(mid))
        self.uncertainty_ns=float(max(max(abs(residual)),max(abs(np.array(sender_residual))))+max(bracket)/2)
        off=self.y-self.x
        self.summary=dict(method='piecewise-linear paired REALTIME/RAW interpolation; no clipping; no extrapolation',
            pairs=len(pairs),holdout_pairs=len(hold),holdout_abs_residual_ns=q(abs(residual)),
            independent_sender_pair_abs_residual_ns=q(abs(np.array(sender_residual))),bracket_width_ns=q(bracket),
            offset_span_ns=float(max(off)-min(off)),linear_drift_ppm=float(np.polyfit(self.x,off,1)[0]*1e6),
            maximum_pair_gap_ms=float(max(np.diff(self.x))/1e6),empirical_uncertainty_envelope_ns=self.uncertainty_ns,
            caveat='Empirical cross-check plus worst bracket, not a formal bound on unobserved clock steps or kernel timestamp-generation error')
    def to_raw(self,wall):
        if wall is None:return None
        assert self.y[0]<=wall-self.w0<=self.y[-1],('clock extrapolation',wall)
        return self.r0+int(round(np.interp(wall-self.w0,self.y,self.x)))
    def to_wall(self,raw):
        if raw is None:return None
        assert self.x[0]<=raw-self.r0<=self.x[-1],('clock extrapolation',raw)
        return self.w0+int(round(np.interp(raw-self.r0,self.x,self.y)))
def main():
    begin=time.monotonic();os.nice(15);os.sched_setaffinity(0,{23})
    assert not H.live_busy(),H.live_busy()
    if REFRESH:
        assert OUT.resolve()==(BASE/'analysis_01').resolve() and not (BASE/'FINAL_SHA256.json').exists()
    OUT.mkdir(exist_ok=REFRESH)
    attached=doc(CAP/'ATTACHED.json')['sources'];audit=lines(CAP/'edge/RECEIVER_TIMING.jsonl')
    status=doc(CAP/'edge/RECEIVER_TIMING_STATUS.json');events=lines(CAP/'edge/events.jsonl');maps=lines(CAP/'map/observations.jsonl')
    assessment=doc(BASE/'existing_assessment_01/ASSESSMENT.json')
    ref=doc(CAP/'PREPARATION_REFERENCE.json');reg=doc(ref['path'])
    assert reg['sha256']==ref['sha256']==assessment['preparation_sha256']
    sizes={s['stream_id']:doc(CAP/s['stream_id']/'SENT_FRAME_SIZES.json') for s in attached}
    clock=Clock(audit,[r['pair'] for v in sizes.values() for r in v]);write('CLOCK_VALIDATION.json',clock.summary)
    assert status['counters']['attempted']==status['counters']['enqueued']==status['counters']['written']==len(audit)
    assert not status['errors'] and not status['writer_alive']
    for r in audit:
        if r['stream_id']:
            s=next(s for s in attached if s['stream_id']==r['stream_id'])
            assert r['session_id']==s['session_id']
    pusch=H.radio_reader(CAP/'pusch/ttracer/live_pusch_with_ingest.csv')
    rlc=H.radio_reader(CAP/'native_radio/NRUE_MAC_RLC_BUFFER_STATUS_live.csv',True)
    scheduler=H.radio_reader(CAP/'pusch/ttracer/live_mcs_with_ingest.csv')
    dci=H.radio_reader(CAP/'native_radio/NRUE_MAC_DCI_GRANT_live.csv',True)
    command=lines(CAP/'COMMANDS.jsonl')
    allframes=[];allchunks=[];summaries={};radio_ticks=[];rates=[]
    for source in attached:
        stream=source['stream_id'];directory=CAP/stream
        evidence=doc(directory/'run4_phase6/PHASE6_UE_EVIDENCE.json')
        plans={r['frame_id']:r for r in evidence['decisions']}
        transmitted={r['frame_id']:r for r in evidence['transmitted_identities']}
        size={r['frame_id']:r for r in sizes[stream]}
        assert set(plans)==set(transmitted)==set(size) and len(size)==len(sizes[stream])
        assert all(r['source']==source for r in sizes[stream])
        ledger=rows(directory/'map_feedback.csv')
        terminals={int(r['frame_id']):r for r in ledger if H.yes(r['terminal'])}
        assert set(terminals)==set(plans) and len(terminals)==sum(H.yes(r['terminal']) for r in ledger)
        assert all(r['stream_id']==stream and r['capture_id']==f"{stream}:{r['frame_id']}" for r in ledger)
        by=C.defaultdict(lambda:C.defaultdict(list));old=C.defaultdict(lambda:C.defaultdict(list))
        for r in audit:
            if r['stream_id']==stream and 'frame_id' in r:by[r['frame_id']][r['kind']].append(r)
        for r in events:
            assert r['source'] in attached
            if r['source']==source:
                old[r['frame_id']][r['kind']].append(r)
                if r['kind']=='edge_wire':assert r['action_identity']==transmitted[r['frame_id']]['run4_identity']
        gt={r['frame_id']:r for r in lines(directory/'numeric_gt/service/FRAMES.jsonl')}
        feedback={r['frame_id']:r for r in evidence['feedback_rows']}
        ownmaps={r['wire_document']['frame_id']:r for r in maps if r['source']==source}
        sender=doc(directory/'SENDER_BINDING.json')
        assert sum(r['chunk_count'] for r in size.values())==sender['datagrams']
        assert sum(r['payload_bytes']+8*r['chunk_count'] for r in size.values())==sender['bytes']
        this=[];chunks=[]
        for frame,d in sorted(plans.items()):
            b=by[frame];e=old[frame];z=size[frame];st=d['stages']
            datagrams=sorted(b['datagram'],key=lambda r:r['recv_seq'])
            comp=[r for r in b['assembly'] if r['completed']];assert len(comp)<=1
            comp=comp[0] if comp else None
            last=next((r for r in datagrams if comp and r['recv_seq']==comp['recv_seq']),None)
            pred=next(iter(e['prediction_ready']),None);mp=ownmaps.get(frame);fb=feedback.get(frame)
            verified=sorted(b['verified'],key=lambda r:r['end_raw_ns'])
            ge={r['name']:r['raw_ns'] for r in gt.get(frame,{}).get('events',[])}
            deadline=next(iter(b['deadline_check']),None);binding=next(iter(b['binding_checked']),None)
            seen=set();previous=None;dup=0;reorder=0
            for c in datagrams:
                assert c['chunk_count']==z['chunk_count'] and not c['payload_truncated']
                idx=c['chunk_index'];isdup=idx in seen;dup+=isdup
                outoforder=not isdup and previous is not None and idx<previous;reorder+=outoforder
                seen.add(idx);previous=idx
                kr=clock.to_raw(c['kernel_wall_ns'])
                ch=dict(stream_id=stream,source_session_id=source['session_id'],frame_id=frame,recv_seq=c['recv_seq'],
                    chunk_index=idx,chunk_count=c['chunk_count'],datagram_bytes=c['datagram_bytes'],
                    kernel_wall_ns=c['kernel_wall_ns'],kernel_mapped_raw_ns=kr,recv_return_raw_ns=c['recv_return_raw_ns'],
                    duplicate=isdup,out_of_order=outoforder,timestamp_status=c['timestamp_status'],
                    kernel_to_app_ms=ms(kr,c['recv_return_raw_ns']),
                    send_bracket_lower_ms=ms(st['last_packet_send_raw_ns'],kr),
                    send_bracket_upper_ms=ms(st['first_packet_send_raw_ns'],kr),
                    frame_envelope_binding_verified=bool(binding),
                    clock_empirical_uncertainty_us=clock.uncertainty_ns/1000)
                chunks.append(ch)
            kernelvals=[c['kernel_wall_ns'] for c in datagrams if c['kernel_wall_ns'] is not None]
            complete_kernel=clock.to_raw(last['kernel_wall_ns']) if last else None
            r=dict(stream_id=stream,ue_id=source['ue_id'],rnti=source['rnti'],source_session_id=source['session_id'],
                capture_id=f'{stream}:{frame}',frame_id=frame,kind=d['kind'],mode_id=d['mode_id'],q_e4=d['q_e4'],
                capture_wall_ns=d['capture']['ns'],action_open_raw_ns=d.get('action_open',{}).get('ns'),**st,
                payload_bytes=z['payload_bytes'],chunk_count=z['chunk_count'],received_datagrams=len(datagrams),
                unique_chunks_observed=len(seen),duplicate_chunks=dup,out_of_order_chunks=reorder,
                missing_chunk_indices=';'.join(str(i) for i in range(z['chunk_count']) if i not in seen),
                first_chunk_kernel_wall_ns=min(kernelvals) if kernelvals else None,
                last_chunk_kernel_wall_ns=max(kernelvals) if kernelvals else None,
                completing_recv_seq=last['recv_seq'] if last else None,completing_chunk_index=last['chunk_index'] if last else None,
                completing_kernel_wall_ns=last['kernel_wall_ns'] if last else None,
                completing_kernel_mapped_raw_ns=complete_kernel,
                completing_recv_return_raw_ns=last['recv_return_raw_ns'] if last else None,
                reassembly_start_raw_ns=comp['start_raw_ns'] if comp else None,
                reassembly_complete_raw_ns=comp['end_raw_ns'] if comp else None,
                binding_end_raw_ns=binding['end_raw_ns'] if binding else None,
                verification_end_raw_ns=verified[0]['end_raw_ns'] if verified else None,
                full_verifier_observed_raw_ns=next((v['raw_ns'] for v in e['edge_wire']),None),
                admission_end_raw_ns=deadline['end_raw_ns'] if deadline else None,
                admission_result=deadline['result'] if deadline else 'NOT_COMPLETED',
                edge_terminal_stage=next((r['stage'] for r in e['edge_terminal']),None),
                ledger_terminal=terminals[frame]['outcome'],
                prediction_ready_raw_ns=pred['prediction_ready_raw_ns'] if pred else None,
                prediction_ready_wall_ns=pred['prediction_ready_wall_ns'] if pred else None,
                gt_published_raw_ns=ge.get('published'),gt_ready_raw_ns=ge.get('target_admitted'),
                gt_prediction_received_raw_ns=ge.get('prediction_received'),gt_result_sent_raw_ns=ge.get('result_sent'),
                gt_feedback_emitted_raw_ns=ge.get('feedback_emitted'),
                feedback_receipt_raw_ns=fb['receipt']['ns'] if fb else None,
                feedback_kind=fb['kind'] if fb else None,gt_outcome=gt.get(frame,{}).get('outcome'),
                map_arrival_wall_ns=mp['arrival_wall_ns'] if mp else None,
                map_install_wall_ns=int(mp['install_wall_s']*1e9) if mp and mp['map_feedback']['accepted'] else None,
                edge_compute_ms=pred['compute_ms'] if pred else None,edge_queue_ms=pred['queue_ms'] if pred else None,
                clock_empirical_uncertainty_us=clock.uncertainty_ns/1000)
            if comp:assert comp['payload_bytes']==z['payload_bytes'] and binding['capture_wall_ns']==r['capture_wall_ns']
            pairs=dict(last_send_to_kernel_ms=('last_packet_send_raw_ns','completing_kernel_mapped_raw_ns'),
                first_send_to_kernel_ms=('first_packet_send_raw_ns','completing_kernel_mapped_raw_ns'),
                kernel_to_app_ms=('completing_kernel_mapped_raw_ns','completing_recv_return_raw_ns'),
                app_to_reassembly_ms=('completing_recv_return_raw_ns','reassembly_complete_raw_ns'),
                reassembly_work_ms=('reassembly_start_raw_ns','reassembly_complete_raw_ns'),
                reassembly_to_binding_ms=('reassembly_complete_raw_ns','binding_end_raw_ns'),
                reassembly_to_admission_ms=('reassembly_complete_raw_ns','admission_end_raw_ns'),
                reassembly_to_verified_ms=('reassembly_complete_raw_ns','verification_end_raw_ns'),
                reassembly_to_full_verifier_ms=('reassembly_complete_raw_ns','full_verifier_observed_raw_ns'),
                send_to_full_verifier_ms=('last_packet_send_raw_ns','full_verifier_observed_raw_ns'),
                send_to_verified_ms=('last_packet_send_raw_ns','verification_end_raw_ns'),
                capture_to_prediction_ms=('capture_wall_ns','prediction_ready_wall_ns'),
                action_to_prediction_ms=('action_open_raw_ns','prediction_ready_raw_ns'),
                action_to_feedback_ms=('action_open_raw_ns','feedback_receipt_raw_ns'),
                prediction_to_feedback_ms=('prediction_ready_raw_ns','feedback_receipt_raw_ns'),
                action_to_gt_ready_ms=('action_open_raw_ns','gt_ready_raw_ns'),
                capture_to_install_ms=('capture_wall_ns','map_install_wall_ns'),
                capture_to_first_send_ms=('capture_wall_ns','send_start_wall_ns'),
                rgb_to_send_ms=('rgb_receipt_raw_ns','last_packet_send_raw_ns'),
                front_ranker_ae_codec_ms=('front_start_raw_ns','front_end_raw_ns'),
                sender_calls_ms=('first_packet_send_raw_ns','last_packet_send_raw_ns'))
            r['send_start_wall_ns']=clock.to_wall(st['first_packet_send_raw_ns'])
            for name,(a,b) in pairs.items():r[name]=ms(r.get(a),r.get(b))
            if d['kind']!='POLICY_DECISION':r['action_to_prediction_ms']=r['action_to_feedback_ms']=r['action_to_gt_ready_ms']=None
            r['clock_complete_frame_span_ms']=ms(r['first_chunk_kernel_wall_ns'],r['last_chunk_kernel_wall_ns'])
            r['before_kernel_fraction_of_send_to_verified']=(r['last_send_to_kernel_ms']/r['send_to_full_verifier_ms']) if r['send_to_full_verifier_ms'] and r['send_to_full_verifier_ms']>0 else None
            this.append(r)
        allframes+=this;allchunks+=chunks
        first=min(r['capture_wall_ns'] for r in this);last=max(r['capture_wall_ns'] for r in this)
        drained=max(r['pair']['wall_ns'] for r in audit if 'pair' in r)
        ownradio=[r for r in pusch if int(r['rnti'])==source['rnti']]
        windows={}
        for label,hi in [('capture_only',last),('capture_through_receiver_close',drained)]:
            sel=[r for r in ownradio if first<=r['event_wall_ns']<=hi]
            windows[label]=dict(event_window_wall_ns=[first,hi],snr_db=q(float(r['snrx10'])/10 for r in sel),
                mcs=q(r['mcs'] for r in sel),allocated_prb=q(r['rbSize'] for r in sel),
                scheduled_tb_bytes=sum(int(r['tb_size']) for r in sel),
                scheduled_tb_Mbps=sum(int(r['tb_size']) for r in sel)*8/(hi-first)*1e3,
                ingest_lag_ms=q(r['ingest_lag_ms'] for r in sel))
        ticks=[];group={};current=None
        def flush():
            if group:
                tick=dict(stream_id=stream,event_wall_ns=max(v['event_wall_ns'] for v in group.values()),
                          bytes=sum(int(v['bytes_in_buffer']) for v in group.values()))
                if first<=tick['event_wall_ns']<=drained:ticks.append(tick)
        for r in rlc:
            if int(r['rnti'])!=source['rnti'] or int(r['ue_id'])!=source['ue_id']:continue
            key=(r['frame'],r['slot'])
            if key!=current:flush();group={}
            current=key;group[r['lcid']]=r
        flush();ticks.sort(key=lambda t:t['event_wall_ns'])
        times=np.array([t['event_wall_ns'] for t in ticks],dtype=np.int64)
        for r in this:
            j=np.searchsorted(times,r['send_start_wall_ns'],side='left')-1
            r['rlc_prior_send_bytes']=ticks[j]['bytes'] if j>=0 else None
            r['rlc_prior_send_age_ms']=ms(int(times[j]),r['send_start_wall_ns']) if j>=0 else None
        radio_ticks+=ticks
        # All rate series use the same one-second wall-clock bins; TB != goodput.
        for second in range(int(math.ceil((drained-first)/1e9))):
            lo=first+second*1_000_000_000;hi=lo+1_000_000_000
            offered=[r for r in this if lo<=r['send_start_wall_ns']<hi]
            completed=[r for r in this if r['completing_kernel_wall_ns'] and lo<=r['completing_kernel_wall_ns']<hi]
            rad=[r for r in ownradio if lo<=r['event_wall_ns']<hi]
            rates.append(dict(stream_id=stream,seconds=second,offered_sfd4_Mbps=sum(r['payload_bytes'] for r in offered)*8/1e6,
                completed_sfd4_Mbps=sum(r['payload_bytes'] for r in completed)*8/1e6,
                scheduled_pusch_tb_Mbps=sum(int(r['tb_size']) for r in rad)*8/1e6))
        owncmd=[r for r in command if r['source']==source]
        assert owncmd and all(r['target_snr_db']==23.5 and r['mapped_rfsim_command_db']==-12 for r in owncmd)
        ack=doc(directory/'ACKED_SNR_STATE.json')
        assert len(ack)==sum(r['kind']=='POLICY_DECISION' for r in this)
        for a in ack:
            assert a['observation']['valid'] and a['observation']['value_db']==23.5
            assert a['features']==plans[a['frame_id']]['features']
            assert a['observation']['effective_since_ns']<=plans[a['frame_id']]['action_open']['ns']
        decisions=[r for r in this if r['kind']=='POLICY_DECISION']
        complete=[r for r in this if r['reassembly_complete_raw_ns'] is not None]
        matched=[r for r in this if r['rlc_prior_send_bytes'] is not None and r['last_send_to_kernel_ms'] is not None]
        gates=assessment['per_ue'][stream]
        summ=dict(source=source,sent=len(this),decisions=len(decisions),holds=sum(r['kind']=='POLICY_HOLD' for r in this),
            fallbacks=sum(r['kind']=='FALLBACK' for r in this),predictions=sum(r['prediction_ready_raw_ns'] is not None for r in this),
            accepted_map_installs=sum(r['map_install_wall_ns'] is not None for r in this),
            fully_reassembled=len(complete),after_reassembly_expired=sum(r['admission_result']=='EXPIRED' for r in this),
            admitted=sum(r['admission_result']=='PASS' for r in this),
            incomplete_with_chunks=sum(r['received_datagrams']>0 and r['reassembly_complete_raw_ns'] is None for r in this),
            no_datagram_observed=sum(r['received_datagrams']==0 for r in this),
            exact_sent_datagrams=sender['datagrams'],received_datagrams=len(chunks),
            sent_chunks_not_observed=sender['datagrams']-len(chunks),duplicate_chunks=sum(r['duplicate'] for r in chunks),
            out_of_order_chunks=sum(r['out_of_order'] for r in chunks),unobserved_note='Right-censored at receiver close; not classified as permanent packet loss.',
            ledger_terminals=dict(C.Counter(r['ledger_terminal'] for r in this)),
            terminal_crosswalk=[dict(ledger=a,boundary=b,n=n) for (a,b),n in C.Counter((r['ledger_terminal'],r['edge_terminal_stage'] or ('MAP_ACCEPTED' if r['map_install_wall_ns'] else 'MAP_REJECTED' if r['map_arrival_wall_ns'] else 'NOT_COMPLETED')) for r in this).items()],
            edge_terminal_stages=dict(C.Counter(r['edge_terminal_stage'] for r in this if r['edge_terminal_stage'])),
            gt_outcomes=dict(C.Counter(r['outcome'] for r in gt.values())),scored_feedback=len(feedback),
            timely_policy_feedback=sum(r['action_to_feedback_ms'] is not None and 0<=r['action_to_feedback_ms']<=170 for r in decisions),
            timely_scored_feedback=sum(r['action_to_feedback_ms'] is not None and 0<=r['action_to_feedback_ms']<=170 and r['feedback_kind']=='DELIVERED_SUCCESS' for r in decisions),
            timely_policy_predictions_action_open=sum(r['action_to_prediction_ms'] is not None and 0<=r['action_to_prediction_ms']<=170 for r in decisions),
            duration_ms={k:q(r[k] for r in this) for k in this[0] if k.endswith('_ms')},
            completing_chunk_before_kernel_fraction=q(r['before_kernel_fraction_of_send_to_verified'] for r in this),
            all_datagram_kernel_to_app_ms=q(r['kernel_to_app_ms'] for r in chunks),
            all_datagram_send_bracket_lower_ms=q(r['send_bracket_lower_ms'] for r in chunks),
            negative_send_to_kernel_frame_intervals=sum(r['last_send_to_kernel_ms'] is not None and r['last_send_to_kernel_ms']<0 for r in this),
            payload_bytes=q(r['payload_bytes'] for r in this),q_e4=q(r['q_e4'] for r in this),
            modes=dict(C.Counter(str(r['mode_id']) for r in this)),
            sender_sfd4_Mbps=sum(r['payload_bytes'] for r in this)*8/(last-first)*1e3,
            completed_feature_goodput_Mbps_capture=sum(r['payload_bytes'] for r in complete if r['completing_kernel_wall_ns']<=last)*8/(last-first)*1e3,
            radio=windows,rlc_capture_bytes=q(r['bytes'] for r in ticks if r['event_wall_ns']<=last),
            backlog_delay_pearson=float(np.corrcoef([r['rlc_prior_send_bytes'] for r in matched],[r['last_send_to_kernel_ms'] for r in matched])[0,1]),
            uplink_dci_rounds=dict(C.Counter(r['round'] for r in dci if int(r['rnti'])==source['rnti'] and r['direction']=='1' and first<=r['event_wall_ns']<=last)),
            scheduler_buffer_bytes=q(r['estimated_ul_buffer'] for r in scheduler if int(r['rnti'])==source['rnti'] and first<=r['event_wall_ns']<=last),
            capture_window_wall_ns=[first,last],capture_span_s=(last-first)/1e9,
            actor_replay=dict(n=len(gates['actor_replay']),all_exact=all(r['exact'] for r in gates['actor_replay'])),
            rescoring=dict(n=len(gates['exact_rescoring']),all_exact=all(r['exact'] for r in gates['exact_rescoring'])),
            quality=gates['quality'],scored_undefined_quality=gates['scored_undefined_quality'],
            controller_faulted_field=evidence['faulted'],unresolved_tickets=evidence['unresolved_tickets_at_close'])
        assert summ['sent']==summ['fully_reassembled']+summ['incomplete_with_chunks']+summ['no_datagram_observed']
        assert summ['fully_reassembled']==summ['admitted']+summ['after_reassembly_expired']
        assert sum(summ['ledger_terminals'].values())==summ['sent']
        assert summ['accepted_map_installs']==summ['ledger_terminals'].get('RESULT_INSTALLED',0)
        summaries[stream]=summ
    csvwrite('FRAME_TIMING.csv',allframes);csvwrite('CHUNK_TIMING.csv',allchunks)
    csvwrite('RLC_TICKS.csv',radio_ticks);csvwrite('RATES_1S.csv',rates)
    verdict='A_DELAY_LARGELY_PRECEDES_KERNEL_ARRIVAL' if all(s['completing_chunk_before_kernel_fraction']['p50']>.9 for s in summaries.values()) else 'D_MIXED_OR_UNRESOLVED'
    result=dict(verdict=verdict,functional_smoke=doc(CAP/'SMOKE.json'),preparation_sha256=reg['sha256'],per_ue=summaries,
        clock=clock.summary,instrumentation=status,launch=doc(BASE/'smoke_launch.json'),route=doc(CAP/'TWO_ROUTE_RESULTS.json')['results'],
        cleanup=doc(BASE/'POSTFLIGHT.json'),analysis_seconds_before_plots=time.monotonic()-begin,
        classification_scope='Received, fully reconstructed frames including expired; missing arrivals remain censored. No claim isolating radio from sender/core/virtual network.',
        instrumentation_note='Producer and receiver wall-time counters overlap: never sum. They include descheduling; writer CPU and causal observer effect unmeasured.',
        legacy_assessment_note='Existing full-route analyzer preserved as STOP: short-route gates not met, absent per_frame_metrics offered/sent=0; faulted=None is not False; nonempty zero-count faults dict is truthy. Corrected smoke counts reconciled separately; no production checker changed.')
    write('SUMMARY.json',result)
    csvwrite('RESULTS.csv',[dict(stream=s,**{k:r[k] for k in ('sent','decisions','holds','fallbacks','fully_reassembled','after_reassembly_expired','predictions','accepted_map_installs','scored_feedback','timely_policy_feedback','incomplete_with_chunks','no_datagram_observed')},
        prekernel_p50_ms=r['duration_ms']['last_send_to_kernel_ms']['p50'],kernel_app_p50_ms=r['duration_ms']['kernel_to_app_ms']['p50'],
        app_reassembly_p50_ms=r['duration_ms']['app_to_reassembly_ms']['p50'],reassembly_verify_p50_ms=r['duration_ms']['reassembly_to_verified_ms']['p50'],
        pusch_median_db=r['radio']['capture_only']['snr_db']['p50'],rlc_median_MB=r['rlc_capture_bytes']['p50']/1e6) for s,r in summaries.items()])
    plots(allframes,summaries,radio_ticks,rates)
    write('INPUT_SHA256.json',H.INPUTS)
    write('REPRODUCIBILITY.json',dict(command=f'env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -B {Path(__file__).relative_to(ROOT)} NEW_OUTPUT_DIRECTORY',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),analysis_wall_s=time.monotonic()-begin,cpu_affinity=sorted(os.sched_getaffinity(0)),python=sys.version))
    print(json.dumps(dict(verdict=verdict,results={s:{k:v[k] for k in ('sent','fully_reassembled','predictions','accepted_map_installs')} for s,v in summaries.items()}),indent=2))
def plots(frames,summaries,ticks,rates):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':16,'axes.titlesize':20,'axes.labelsize':16,'legend.fontsize':13,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    colors=['#176B9D','#E07A24']
    metrics=[('last_send_to_kernel_ms','Last send → completing kernel RX'),
        ('kernel_to_app_ms','Matched kernel RX → app receive'),
        ('app_to_reassembly_ms','App receive → reassembly complete'),
        ('reassembly_to_full_verifier_ms','Reassembly complete → verification'),
        ('front_ranker_ae_codec_ms','Frontend/ranker/AE/codec'),
        ('edge_compute_ms','Edge compute [predicted only]'),
        ('action_to_feedback_ms','Action open → scored feedback')]
    fig,axes=plt.subplots(1,2,figsize=(19.2,10.8),dpi=200,sharey=True)
    for i,(stream,s) in enumerate(summaries.items()):
        ax=axes[i]
        for j,(key,label) in enumerate(metrics):
            v=s['duration_ms'][key]
            if not v['n']:continue
            ax.plot([v['p50'],v['p99']],[j,j],color=colors[i],lw=4)
            ax.scatter(v['p50'],j,s=120,color=colors[i],zorder=3)
            ax.scatter(v['p99'],j,s=95,facecolors='white',edgecolors=colors[i],linewidths=2,zorder=3)
            ax.text(1.02,j,f"n={v['n']}",transform=ax.get_yaxis_transform(),va='center',fontsize=14)
        ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(.004,20000)
        ax.set_yticks(range(len(metrics)),[x[1] for x in metrics]);ax.set_ylim(len(metrics)-.5,-.5)
        ax.grid(axis='x',alpha=.25);ax.set_xlabel('Interval duration (ms; log scale above 0.01 ms)')
        ax.set_title(f"UE {i}  |  {s['fully_reassembled']}/{s['sent']} fully reassembled\n{s['after_reassembly_expired']} expired; {s['accepted_map_installs']} maps; 0/{s['decisions']} timely feedback",pad=22)
        ax.axvline(170,color='#5B6170',ls=':',lw=1.4);ax.axvline(500,color='#9D3248',ls='--',lw=1.4)
    fig.suptitle('The multi-second delay is already present at kernel arrival',fontsize=25,y=.97)
    fig.text(.5,.09,'● Median     ○ P99   |   Each row is a separately measured interval; do not add these quantiles.',ha='center',fontsize=15)
    fig.text(.5,.052,'170 ms: feedback deadline from action open.  500 ms: processing horizon from capture.\nVertical lines provide scale only; they are not deadlines for each individual stage.',ha='center',fontsize=14)
    fig.subplots_adjust(left=.255,right=.955,wspace=.26,top=.81,bottom=.19)
    fig.savefig(OUT/'01_DELAY_BREAKDOWN.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(19.2,10.8),dpi=200,sharex='col')
    for i,(stream,s) in enumerate(summaries.items()):
        these=[r for r in frames if r['stream_id']==stream];first=s['capture_window_wall_ns'][0]
        complete=[r for r in these if r['last_send_to_kernel_ms'] is not None]
        x=[(r['send_start_wall_ns']-first)/1e9 for r in complete]
        axes[0,i].plot(x,[r['last_send_to_kernel_ms']/1000 for r in complete],color=colors[i],label='Last send → completing kernel RX',lw=2)
        axes[0,i].plot(x,[r['kernel_to_app_ms']/1000 for r in complete],color='#606A78',label='Kernel RX → app receive',lw=1.5)
        missing=[r for r in these if r['last_send_to_kernel_ms'] is None]
        axes[0,i].scatter([(r['send_start_wall_ns']-first)/1e9 for r in missing],[.035]*len(missing),transform=axes[0,i].get_xaxis_transform(),marker='|',s=80,color='#A12D48',label='No complete frame by close (rug)')
        axes[0,i].set_ylabel('Matched interval (s)')
        axes[0,i].set_title(f"UE {i}  |  commanded 23.5 dB; measured median {s['radio']['capture_only']['snr_db']['p50']:.1f} dB\nPUSCH median MCS {s['radio']['capture_only']['mcs']['p50']:.0f}",pad=12)
        bt=[r for r in ticks if r['stream_id']==stream]
        axes[1,i].fill_between([(r['event_wall_ns']-first)/1e9 for r in bt],[r['bytes']/1e6 for r in bt],color=colors[i],alpha=.3)
        axes[1,i].set_ylabel('UE RLC backlog (MB)')
        rr=[r for r in rates if r['stream_id']==stream]
        for key,label,color in [('offered_sfd4_Mbps','Offered feature payload',colors[i]),('scheduled_pusch_tb_Mbps','Scheduled PUSCH TB','#777777'),('completed_sfd4_Mbps','Completed feature goodput','#238B65')]:
            axes[2,i].step([r['seconds'] for r in rr],[r[key] for r in rr],where='post',label=label,color=color,lw=1.8)
        axes[2,i].set_ylabel('Rate (Mbit/s; 1-s bins)');axes[2,i].set_xlabel('Wall time since this UE’s first capture (s)')
        for j in range(3):
            axes[j,i].grid(alpha=.2);axes[j,i].axvline(s['capture_span_s'],color='#555555',ls=':',lw=1)
        axes[0,i].legend(loc='upper left',fontsize=11)
        axes[2,i].legend(loc='upper left',fontsize=11)
    fig.suptitle('Growing backlog accompanies the pre-kernel delay; radio causality is not isolated',fontsize=23,y=.985)
    fig.text(.5,.025,'Source/RNTI-bound evidence. Scheduled TB volume is not application goodput. Unreceived/incomplete frames remain censored.\nTop-panel x-axis is sender time; RLC and rates use their event times. Association is not a packet-level radio attribution.',ha='center',fontsize=14)
    fig.subplots_adjust(left=.075,right=.98,top=.83,bottom=.13,hspace=.18,wspace=.22)
    fig.savefig(OUT/'02_BACKLOG_AND_DELAY.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()

