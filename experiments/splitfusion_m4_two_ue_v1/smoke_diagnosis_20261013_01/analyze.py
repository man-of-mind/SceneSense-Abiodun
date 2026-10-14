#!/usr/bin/env python3
"""Saved-evidence-only M4 diagnosis. No production imports, GPU, or live commands.

Run with /usr/bin/python3 analyze.py. All writes remain beside this script.
The lightweight supervisor pauses the worker if a live-host lock or radio/CARLA
process appears. No lock is held against a live launcher.
"""
from __future__ import annotations
import ast
import collections as co
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CAP = ROOT.parent / 'abiodun_run4_single_gpu_profiler_v1/experiments/splitfusion_m4_two_ue_v1/smoke_live_20261013_07'
OLD = OUT.parent / 'first_send_marker_repair_20261013_01'
INPUTS = {}
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(OUT / 'mpl_cache'))


def live_busy():
    try:
        with open('/tmp/splitfusion-live-host.lock', 'rb') as h:
            fcntl.flock(h, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(h, fcntl.LOCK_UN)
    except BlockingIOError:
        return 'live-host lock held'
    except FileNotFoundError:
        return 'live-host coordination lock missing'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / 'comm').read_text().strip()
            if comm.startswith(('CarlaUnreal', 'nr-softmodem', 'nr-uesoftmodem')):
                return f'live process {p.name} {comm}'
        except (OSError, ProcessLookupError):
            pass
    return None


def supervise():
    started = time.monotonic()
    child = None
    paused = False
    with (OUT / 'COORDINATION.jsonl').open('a') as log:
        def mark(kind, **kw):
            log.write(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), kind=kind, **kw)) + '\n')
            log.flush()
        mark('START', method='read-only lock/process checks every 100 ms; one CPU, nice 19; no live-host lock retained')
        while child is None or child.poll() is None:
            reason = live_busy()
            if child is None and not reason:
                child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], env=os.environ.copy())
                mark('WORKER_STARTED', pid=child.pid)
            elif reason and not paused:
                if child is not None:
                    child.send_signal(signal.SIGSTOP)
                paused = True
                mark('PAUSED', reason=reason)
            elif not reason and paused:
                if child is not None:
                    child.send_signal(signal.SIGCONT)
                paused = False
                mark('RESUMED')
            time.sleep(.1)
        mark('FINISHED', returncode=child.returncode, wall_seconds=time.monotonic()-started)
    raise SystemExit(child.returncode)


def bound(path):
    path = Path(path).resolve()
    if str(path) not in INPUTS:
        INPUTS[str(path)] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def doc(path):
    return json.loads(bound(path).read_text())


def lines(path):
    return [json.loads(s) for s in bound(path).read_text().splitlines() if s.strip()]


def rows(path):
    with bound(path).open() as h:
        return list(csv.DictReader(h))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(name, values):
    fields = list(dict.fromkeys(k for v in values for k in v))
    with (OUT / name).open('w') as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(values)


def yes(x):
    return str(x).lower() in ('true', '1')


def ms(a, b):
    return None if a is None or b is None else (b-a)/1e6


def q(values):
    import numpy as np
    v = [float(x) for x in values if x is not None and x != '']
    return dict(n=len(v), **({k:float(x) for k,x in zip(('min','p50','p95','p99','max'), np.percentile(v,[0,50,95,99,100]))} if v else {}))


def radio_reader(path, native=False):
    # Reuse the retained reader's header convention and its exact time-of-day
    # reconstruction functions without importing any model/runtime dependencies.
    clock_path = bound(ROOT / 'rl_agent/splitfusion_hybrid_sac_live_route_b_v2/ue_telemetry_provider_v2.py')
    tree = ast.parse(clock_path.read_text())
    names = {'parse_time_of_day_us', 'reconstruct_event_realtime_ns'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(re=re, DAY_US=86_400_000_000, MAX_INGEST_US=2_000_000,
              _TOD=re.compile(r'^([0-2][0-9]):([0-5][0-9]):([0-5][0-9])\.([0-9]{6})$'))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(clock_path),'exec'), ns)
    result=[]; fields=None
    with bound(path).open() as h:
        for a in csv.reader(h):
            if len(a)>2 and a[2]=='time':
                fields=(['ingest_raw_ns','ingest_wall_ns'] if native else ['ingest_wall_ns','ingest_raw_ns'])+a[2:]
                continue
            if fields is None or len(a)!=len(fields):
                continue
            r=dict(zip(fields,a))
            wall=int(r['ingest_wall_ns'])
            # Capture host is America/Los_Angeles, UTC-7 on this date. Independent
            # check below rejects all event/receipt differences outside 0..2 sec.
            r['event_wall_ns']=ns['reconstruct_event_realtime_ns'](r['time'],SimpleNamespace(real_ns=wall,utc_offset_s=-7*3600))
            r['ingest_lag_ms']=(wall-r['event_wall_ns'])/1e6
            result.append(r)
    return result


def main():
    os.nice(19)
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    import numpy as np
    started=time.monotonic()
    saved=doc(OLD/'SMOKE_RESULT.json'); assessment=doc(OLD/'analysis_01/ASSESSMENT.json')
    attached=doc(CAP/'ATTACHED.json')['sources']
    events=lines(CAP/'edge/events.jsonl'); maps=lines(CAP/'map/observations.jsonl')
    runtime=doc(CAP/'RUNTIME.json')
    reference=doc(CAP/'PREPARATION_REFERENCE.json')
    registration=doc(reference['path'])
    assert registration['sha256']==reference['sha256']==runtime['preparation_sha256']
    assert runtime['sources']==attached
    assert runtime['preparation_sha256']==assessment['preparation_sha256']
    # Preserve identity context in each joined row; fail on foreign events.
    for r in events+maps:
        assert r['source'] in attached
    command=lines(CAP/'COMMANDS.jsonl')
    all_rows=[]; summaries={}; offsets=[]; payload_rows=[]; stage_stats=[]
    pusch=radio_reader(CAP/'pusch/ttracer/live_pusch_with_ingest.csv')
    scheduler=radio_reader(CAP/'pusch/ttracer/live_mcs_with_ingest.csv')
    rlc=radio_reader(CAP/'native_radio/NRUE_MAC_RLC_BUFFER_STATUS_live.csv',True)
    dci=radio_reader(CAP/'native_radio/NRUE_MAC_DCI_GRANT_live.csv',True)
    pdcp=radio_reader(CAP/'native_radio/NR_PDCP_TX_SDU_live.csv',True)
    radio_export=[]; backlog_export=[]
    for source in attached:
        s=source['stream_id']; base=CAP/s
        evidence=doc(base/'run4_phase6/PHASE6_UE_EVIDENCE.json')
        planned={r['frame_id']:r for r in evidence['decisions']}
        transmitted={r['frame_id']:r for r in evidence['transmitted_identities']}
        ledger=rows(base/'map_feedback.csv')
        terminals={int(r['frame_id']):r for r in ledger if yes(r['terminal'])}
        assert len(terminals)==sum(yes(r['terminal']) for r in ledger)==len(planned)==208
        assert set(planned)==set(transmitted)==set(terminals)
        assert all(r['stream_id']==s and r['capture_id']==f"{s}:{r['frame_id']}" for r in ledger)
        ev=co.defaultdict(lambda:co.defaultdict(list))
        for r in events:
            if r['source']==source:
                assert r['frame_id'] in planned
                ev[r['frame_id']][r['kind']].append(r)
                offsets.append(r['wall_ns']-r['raw_ns'])
                if r['kind']=='edge_wire':
                    assert r['capture_timestamp_ns']==planned[r['frame_id']]['capture']['ns']
                    assert r['action_identity']==transmitted[r['frame_id']]['run4_identity']
        own_maps={r['wire_document']['frame_id']:r for r in maps if r['source']==source}
        gt={r['frame_id']:r for r in lines(base/'numeric_gt/service/FRAMES.jsonl')}
        feedback={r['frame_id']:r for r in evidence['feedback_rows']}
        scored=doc(CAP/'edge'/f'{s}_GT_FEEDBACK.json')
        ack_snr=doc(base/'ACKED_SNR_STATE.json')
        own_commands=[r for r in command if r['source']==source]
        assert own_commands and all(r['target_snr_db']==23.5 and r['mapped_rfsim_command_db']==-12. for r in own_commands)
        assert len(ack_snr)==sum(r['kind']=='POLICY_DECISION' for r in planned.values())
        for r in ack_snr:
            obs=r['observation'];d=planned[r['frame_id']]
            assert obs['valid'] and obs['value_db']==23.5 and obs['effective_since_ns']<=d['action_open']['ns']
            assert obs['identity']['ue_id']==f"oai-ue{source['ue_id']}-rnti{source['rnti']}"
            assert r['features']==d['features']
        sender=doc(base/'SENDER_BINDING.json')
        assert sender['source']==source
        retained={int(re.search(r'__frame(\d+)__',p.name)[1]):p for p in (base/'paired_retention_v2').glob('*.sfd4')}
        this=[]
        for f,d in sorted(planned.items()):
            st=d['stages']; ee=ev[f]; t=terminals[f]; g=gt.get(f,{}); fb=feedback.get(f)
            wire=next(iter(ee['edge_wire']),None); pred=next(iter(ee['prediction_ready']),None)
            reasm=next(iter(ee['reassembled']),None); term=next(iter(ee['edge_terminal']),None)
            mp=own_maps.get(f); ge={v['name']:v['raw_ns'] for v in g.get('events',[])}
            r=dict(stream_id=s,ue_id=source['ue_id'],rnti=source['rnti'],source_session_id=source['session_id'],
                   capture_id=f'{s}:{f}',frame_id=f,tensor_seq=d['tensor_seq'],kind=d['kind'],mode_id=d['mode_id'],q_e4=d['q_e4'],
                   capture_wall_ns=d['capture']['ns'],action_open_raw_ns=d.get('action_open',{}).get('ns'),**st,
                   edge_verified_raw_ns=wire['raw_ns'] if wire else None,
                   edge_verified_wall_ns=wire['wall_ns'] if wire else None,
                   admission_log_raw_ns=reasm['raw_ns'] if reasm else None,
                   reassembly_last_recv_wall_ns=None,compute_start_wall_ns=None,
                   prediction_ready_wall_ns=pred['prediction_ready_wall_ns'] if pred else None,
                   prediction_ready_raw_ns=pred['prediction_ready_raw_ns'] if pred else None,
                   gt_published_raw_ns=ge.get('published'),gt_ready_score_admitted_raw_ns=ge.get('target_admitted'),
                   gt_prediction_received_raw_ns=ge.get('prediction_received'),gt_result_sent_raw_ns=ge.get('result_sent'),
                   feedback_emitted_raw_ns=ge.get('feedback_emitted'),policy_feedback_raw_ns=fb['receipt']['ns'] if fb else None,
                   policy_feedback_kind=fb['kind'] if fb else None,
                   map_arrival_wall_ns=mp['arrival_wall_ns'] if mp else None,
                   map_install_wall_ns=int(mp['install_wall_s']*1e9) if mp and mp['map_feedback']['accepted'] else None,
                   ledger_terminal=t['outcome'],ledger_receipt_wall_s=t['feedback_received_at'],
                   edge_terminal_stage=term['stage'] if term else None,
                   gt_outcome=g.get('outcome'),payload_sfd4_bytes=reasm['payload_bytes'] if reasm else None,
                   payload_provenance='edge admission log' if reasm else None,
                   exact_reassembly_raw_ns=None,kernel_receive_wall_ns=None,encoding_only_ms=None,gpu_wait_ms=None)
            if f in retained:
                size=bound(retained[f]).stat().st_size
                assert r['payload_sfd4_bytes'] in (None,size)
                r['payload_sfd4_bytes']=size
                r['payload_provenance']='retained SFD4'+(' and edge admission log' if reasm else '')
            if mp:
                et=mp['wire_document']['edge_timing']
                r['reassembly_last_recv_wall_ns']=int(et['reassembly_complete_wall_s']*1e9)
                r['compute_start_wall_ns']=int(et['compute_start_wall_s']*1e9)
            pairs=dict(rgb_to_sensor_prepared_ms=('rgb_receipt_raw_ns','sensor_prepared_raw_ns'),
                       sensor_prepared_to_action_ms=('sensor_prepared_raw_ns','action_open_raw_ns'),
                       input_7ch_ms=('input_7ch_start_raw_ns','front_start_raw_ns'),
                       front_ranker_ae_codec_ms=('front_start_raw_ns','front_end_raw_ns'),
                       front_end_to_send_ms=('front_end_raw_ns','first_packet_send_raw_ns'),
                       packet_send_calls_ms=('first_packet_send_raw_ns','last_packet_send_raw_ns'),
                       rgb_to_send_ms=('rgb_receipt_raw_ns','last_packet_send_raw_ns'),
                       send_to_verified_ms=('last_packet_send_raw_ns','edge_verified_raw_ns'),
                       rgb_to_verified_ms=('rgb_receipt_raw_ns','edge_verified_raw_ns'),
                       capture_to_verified_wall_ms=('capture_wall_ns','edge_verified_wall_ns'),
                       capture_to_prediction_wall_ms=('capture_wall_ns','prediction_ready_wall_ns'),
                       action_to_prediction_raw_ms=('action_open_raw_ns','prediction_ready_raw_ns'),
                       action_to_gt_ready_ms=('action_open_raw_ns','gt_ready_score_admitted_raw_ns'),
                       action_to_feedback_ms=('action_open_raw_ns','policy_feedback_raw_ns'),
                       prediction_to_feedback_ms=('prediction_ready_raw_ns','policy_feedback_raw_ns'),
                       capture_to_map_install_ms=('capture_wall_ns','map_install_wall_ns'))
            for k,(a,b) in pairs.items(): r[k]=ms(r.get(a),r.get(b))
            r['edge_queue_ms']=pred['queue_ms'] if pred else None
            r['edge_compute_ms']=pred['compute_ms'] if pred else None
            r['capture_to_map_ack_ms']=(float(t['feedback_received_at'])*1e9-r['capture_wall_ns'])/1e6 if t['outcome']=='RESULT_INSTALLED' else None
            if d['kind']!='POLICY_DECISION':
                r['action_to_feedback_ms']=r['action_to_prediction_raw_ms']=None
            this.append(r)
            if r['payload_sfd4_bytes'] is not None:
                payload_rows.append({k:r[k] for k in ('stream_id','frame_id','kind','mode_id','q_e4','payload_sfd4_bytes','payload_provenance','edge_terminal_stage')})
        all_rows+=this
        decisions=[r for r in this if r['kind']=='POLICY_DECISION']
        first=min(r['capture_wall_ns'] for r in this);last=max(r['capture_wall_ns'] for r in this)
        # Explicit offered-window and observed-edge-drain-window results; exclude
        # all model prewarm/attach/startup radio samples from the main statistic.
        drain=max(r['wall_ns'] for r in events if r['source']==source)
        own=[r for r in pusch if int(r['rnti'])==source['rnti']]
        windows={}
        for label,lo,hi in [('whole_log',0,10**30),('capture_only',first,last),('capture_through_edge_drain',first,drain)]:
            selected=[r for r in own if lo<=r['event_wall_ns']<=hi]
            windows[label]=dict(n=len(selected),snr_db=q(float(r['snrx10'])/10 for r in selected),
                  mcs=q(r['mcs'] for r in selected),rb=q(r['rbSize'] for r in selected),tb_bytes=q(r['tb_size'] for r in selected),
                  sum_tb_bytes=sum(int(r['tb_size']) for r in selected),ingest_lag_ms=q(r['ingest_lag_ms'] for r in selected),
                  event_window_wall_ns=[lo,hi] if label!='whole_log' else None)
            if label!='whole_log':
                windows[label]['scheduled_tb_Mbps_wall']=sum(int(r['tb_size']) for r in selected)*8/((hi-lo)/1e9)/1e6
        for r in own:
            if first<=r['event_wall_ns']<=last:
                radio_export.append(dict(stream_id=s,**r))
        # RLC ticks: sum the last value for each distinct LCID, not median of
        # per-LCID rows (which would be dominated by zero signalling channels).
        ticks=[];current=None;group={}
        for r in rlc:
            if int(r['rnti'])!=source['rnti'] or int(r['ue_id'])!=source['ue_id']: continue
            key=(r['frame'],r['slot'])
            if key!=current and group:
                tick=dict(stream_id=s,event_wall_ns=max(v['event_wall_ns'] for v in group.values()),
                          bytes=sum(int(v['bytes_in_buffer']) for v in group.values()))
                if first<=tick['event_wall_ns']<=last:ticks.append(tick)
                group={}
            current=key;group[r['lcid']]=r
        backlog_export+=ticks
        # Exact temporal predecessor is only an association, not packet mapping.
        bt=np.array([t['event_wall_ns'] for t in ticks],dtype=np.int64)
        assert np.all(np.diff(bt)>=0)
        bb=np.array([t['bytes'] for t in ticks])
        for r in this:
            j=np.searchsorted(bt,r['capture_wall_ns'],side='left')-1
            r['rlc_at_prior_capture_tick_bytes']=int(bb[j]) if j>=0 else None
        pairs=[(r['rlc_at_prior_capture_tick_bytes'],r['send_to_verified_ms']) for r in this if r['rlc_at_prior_capture_tick_bytes'] is not None and r['send_to_verified_ms'] is not None]
        sched=[r for r in scheduler if int(r['rnti'])==source['rnti'] and first<=r['event_wall_ns']<=last]
        grants=[r for r in dci if int(r['rnti'])==source['rnti'] and r['direction']=='1' and first<=r['event_wall_ns']<=last]
        pd=[r for r in pdcp if int(r['ue_id'])==source['ue_id'] and first<=r['event_wall_ns']<=last]
        durations={k:q(r.get(k) for r in this) for k in this[0] if k.endswith('_ms')}
        for k,v in durations.items():stage_stats.append(dict(stream_id=s,interval=k,**v))
        validgt=[r for r in decisions if r['gt_ready_score_admitted_raw_ns'] is not None]
        joined=[r for r in decisions if r['policy_feedback_raw_ns'] is not None]
        count=co.Counter(r['ledger_terminal'] for r in this)
        observation_stages=co.Counter(('NO_EDGE_EVENT' if not any(ev[r['frame_id']].values()) else
                                      r['edge_terminal_stage'] or ('MAP_ACCEPTED' if r['map_install_wall_ns'] else 'MAP_REJECTED')) for r in this)
        # Cross-table proves whether late NACK/map observations were collapsed
        # into the local watchdog's earlier terminal, without double counting.
        cross=co.Counter((r['ledger_terminal'], 'NO_EDGE_EVENT' if not any(ev[r['frame_id']].values()) else
                         r['edge_terminal_stage'] or ('MAP_ACCEPTED' if r['map_install_wall_ns'] else 'MAP_REJECTED')) for r in this)
        summ=dict(source=source,sent=len(this),offered_assigned_frames=len(planned),
            offered_all_collector_opportunities=None,collector_offer_note='per_frame_metrics.csv not retained; all assigned/transmitted frames recovered, broader offer count unknown',
            capture_window_wall_ns=[first,last],capture_window_seconds=(last-first)/1e9,
            decisions=len(decisions),holds=sum(r['kind']=='POLICY_HOLD' for r in this),fallbacks=sum(r['kind']=='FALLBACK' for r in this),
            predictions=sum(r['prediction_ready_raw_ns'] is not None for r in this),
            decision_predictions=sum(r['prediction_ready_raw_ns'] is not None for r in decisions),
            accepted_map_installs=sum(r['map_install_wall_ns'] is not None for r in this),
            map_observations=len(own_maps),terminals=dict(count),
            fully_reassembled=sum(bool(ev[r['frame_id']]['reassembled']) or r['edge_terminal_stage']=='EDGE_AFTER_REASSEMBLY' for r in this),
            after_reassembly_expired=sum(r['edge_terminal_stage']=='EDGE_AFTER_REASSEMBLY' for r in this),
            no_edge_evidence=[r['frame_id'] for r in this if not any(ev[r['frame_id']].values())],
            edge_stages=dict(observation_stages),terminal_crosswalk=[dict(ledger=a,evidence=b,n=n) for (a,b),n in cross.items()],
            terminal_rows=sum(yes(r['terminal']) for r in ledger),late_nonterminal_rows=sum(not yes(r['terminal']) for r in ledger),
            policy_scored_feedback=len(feedback),timely_policy_feedback=sum(r['action_to_feedback_ms'] is not None and 0<=r['action_to_feedback_ms']<=170 for r in decisions),
            timely_decision_prediction_action_open=sum(r['action_to_prediction_raw_ms'] is not None and 0<=r['action_to_prediction_raw_ms']<=170 for r in decisions),
            policy_reward_resolutions=dict(co.Counter(r['terminal'] for r in evidence['resolutions'])),
            gt_outcomes=dict(co.Counter(r.get('outcome') for r in gt.values())),
            gt_ready=len(validgt),gt_ready_by_170=sum(r['action_to_gt_ready_ms']<=170 for r in validgt),
            scored_gt_ready_before_prediction=sum(r['gt_ready_score_admitted_raw_ns']<=r['prediction_ready_raw_ns'] for r in joined),
            duration_ms=durations,
            modes_all_sent=dict(co.Counter(str(r['mode_id']) for r in this)),modes_policy=dict(co.Counter(str(r['mode_id']) for r in decisions)),
            q_e4_all=q(r['q_e4'] for r in this),q_e4_policy=q(r['q_e4'] for r in decisions),
            payload_known_sfd4_bytes=q(r['payload_sfd4_bytes'] for r in this),
            payload_admitted_sfd4_bytes=q(r['payload_bytes'] for f,v in ev.items() for r in v['reassembled']),
            payload_retained_sfd4_bytes=q(p.stat().st_size for p in retained.values()),
            datagrams=sender['datagrams'],udp_application_bytes=sender['bytes'],
            sfd4_total_bytes=sender['bytes']-8*sender['datagrams'],
            sfd4_mean_bytes=(sender['bytes']-8*sender['datagrams'])/len(this),
            sfd4_full_population_quantiles=None,
            sender_Mbps_over_capture_span=sender['bytes']*8/((last-first)/1e9)/1e6,
            commanded_snr_db=23.5,command_ack_counts=dict(co.Counter(r['command_timing_status'] for r in own_commands)),
            actor_acknowledged_snr_consistent=len(ack_snr),
            radio_windows=windows,rlc_summed_tick_bytes=q(t['bytes'] for t in ticks),
            rlc_nonzero_fraction=sum(t['bytes']>0 for t in ticks)/len(ticks),
            backlog_delay_pearson=float(np.corrcoef(np.array(pairs).T)[0,1]),
            scheduler_estimated_buffer_bytes=q(r['estimated_ul_buffer'] for r in sched),
            uplink_dci_rounds=dict(co.Counter(r['round'] for r in grants)),
            pdcp_capture_sdu_bytes=sum(int(r['sdu_bytes']) for r in pd),pdcp_events=len(pd),
            prior_actor_replay=all(r['exact'] for r in assessment['per_ue'][s]['actor_replay']),
            prior_rescoring=all(r['exact'] for r in assessment['per_ue'][s]['exact_rescoring']),
            prior_quality=assessment['per_ue'][s]['quality'])
        assert summ['policy_scored_feedback']==len(scored)
        assert all(r['kind']=='DELIVERED_SUCCESS' and r['q_perc'] is not None for r in scored)
        assert summ['accepted_map_installs']==count['RESULT_INSTALLED']
        assert summ['fully_reassembled']+len(summ['no_edge_evidence'])==208
        summaries[s]=summ
    write_csv('FRAME_TIMING.csv',all_rows)
    write_csv('INTERVAL_SUMMARY.csv',stage_stats)
    write_csv('KNOWN_PAYLOADS.csv',payload_rows)
    write_csv('PUSCH_CAPTURE_ONLY.csv',radio_export)
    write_csv('RLC_CAPTURE_TICKS.csv',backlog_export)
    result=dict(verdict='FUNCTIONAL_INTEGRATION_PASS__NOT_PERFORMANCE_QUALIFIED',
                scope='Saved smoke only; no causal live intervention',preparation_sha256=assessment['preparation_sha256'],
                per_ue=summaries,clock_pair_offset_span_ns=max(offsets)-min(offsets),
                clocks='WALL and MONOTONIC_RAW kept separate; no MONOTONIC conversion or cross-domain subtraction',
                quantiles='NumPy linear percentile; descriptive conditional subsets; missing completion is never latency',
                radio_time='Exact retained time-of-day reconstructor; UTC-7; source RNTI from ATTACHED; 0..2s ingest guard',
                analysis_seconds=time.monotonic()-started)
    write_json('CORRECTED_ANALYSIS.json',result)
    print(json.dumps({s:{k:a[k] for k in ('sent','predictions','accepted_map_installs','timely_policy_feedback','after_reassembly_expired')} for s,a in summaries.items()},indent=2),flush=True)
    plots(all_rows,summaries)
    # Bind both evidence and the source bodies read to interpret timestamps.
    for name in ('datapath.py','transport.py','edge_entry.py','edge_components.py','frontend.py','sender.py','analysis.py'):
        bound(ROOT/'rl_agent/splitfusion_m4_two_ue_v1'/name)
    for name in ('phase6_ue_runtime_v2.py','run4_live_wire_v2.py','phase6_edge_runtime_v2.py'):
        bound(ROOT/'rl_agent/splitfusion_hybrid_sac_live_route_b_v2'/name)
    bound(ROOT/'rl_agent/ue_route_b_split_cell_adapter_v1.py')
    source_binding={str(Path(p).relative_to(ROOT)):dict(sha256=v['sha256'],registered=registration['sources'].get(str(Path(p).relative_to(ROOT))))
                    for p,v in INPUTS.items() if Path(p).is_relative_to(ROOT/'rl_agent')}
    for p,v in source_binding.items():
        if p!='rl_agent/ue_route_b_split_cell_adapter_v1.py':
            assert v['registered']==v['sha256'],f'Interpretation source changed: {p}'
    # The inherited collector is not separately listed in this M4 inventory.
    # Record current deployment agreement without claiming a missing sealed hash.
    name='rl_agent/ue_route_b_split_cell_adapter_v1.py'
    deployed=bound(ROOT.parent/'abiodun_run4_single_gpu_profiler_v1'/name)
    source_binding[name]['deployment_sha256']=INPUTS[str(deployed)]['sha256']
    assert source_binding[name]['deployment_sha256']==source_binding[name]['sha256']
    source_binding[name]['scope']='current main/deployment byte agreement; not separately hashed in outer registration'
    write_json('SOURCE_BINDING.json',source_binding)
    write_json('INPUT_SHA256.json',INPUTS)
    write_json('REPRODUCIBILITY.json',dict(python=sys.version,argv=sys.argv,cpu_affinity=sorted(os.sched_getaffinity(0)),
        numpy=np.__version__,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],total_seconds=time.monotonic()-started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def plots(data,summaries):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter,ScalarFormatter
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':19,'axes.labelsize':21,'axes.titlesize':23,
                         'xtick.labelsize':18,'ytick.labelsize':18,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    teal='#087E8B';purple='#7356A6';orange='#DD8A2A';red='#BA4747';dark='#23364A'
    def save(fig,name):
        assert tuple(round(x*160) for x in fig.get_size_inches())==(3840,2160)
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=160)
        plt.close(fig)
    fig=plt.figure(figsize=(24,13.5))
    fig.text(.065,.93,'Two-UE integration works; timeliness remains unqualified',fontsize=32,weight='bold',color=dark)
    fig.text(.065,.885,'First passing M4 smoke  •  208 sent frames per UE  •  One terminal per sent frame',fontsize=23,color=dark)
    ax=fig.add_axes([.095,.45,.64,.33])
    items=[('RESULT_INSTALLED','Accepted map installation',teal),('FEEDBACK_TIMEOUT','Local feedback timeout',orange),
           ('OTHER','Other terminal',purple)]
    for y,s in enumerate(('m4-ue0','m4-ue1')):
        a=summaries[s];left=0
        for key,label,color in items:
            n=a['terminals'].get(key,0) if key!='OTHER' else a['sent']-a['accepted_map_installs']-a['terminals'].get('FEEDBACK_TIMEOUT',0)
            ax.barh(y,n,left=left,height=.55,color=color,edgecolor='white',linewidth=1.5)
            if n>3:ax.text(left+n/2,y,f'{n}\n{n/208:.1%}',ha='center',va='center',fontsize=23,color='white',weight='bold')
            else:ax.annotate(str(n),xy=(left+n/2,y),xytext=(225,y),ha='left',va='center',fontsize=23,color=color,
                              arrowprops=dict(arrowstyle='-',color=color))
            left+=n
        ax.text(247,y,'1 superseded' if y==0 else '2 stale at map',va='center',fontsize=20,color=purple)
    ax.set_yticks([0,1],['UE0','UE1']);ax.invert_yaxis();ax.set_xlim(0,280)
    ax.set_xticks([0,52,104,156,208]);ax.set_xlabel('Sent-frame terminal counts (n = 208 per UE)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=c,label=l) for _,l,c in items],loc='upper left',bbox_to_anchor=(0,-.26),ncol=2,frameon=False,fontsize=19)
    fig.text(.77,.75,'POLICY FEEDBACK',fontsize=21,color=dark,weight='bold')
    fig.text(.77,.65,'0 / 103     UE0\n0 / 100     UE1',fontsize=29,color=red,weight='bold',linespacing=1.5)
    fig.text(.77,.50,'Within 170 ms of action-open\n14 / 11 scored replies arrived,\nall late. Not map ACK latency.',fontsize=20,color=dark,linespacing=1.45)
    fig.text(.065,.25,'Timeout does not mean packet loss',fontsize=25,color=dark,weight='bold')
    fig.text(.065,.185,'172 / 179 frames fully reassembled but expired at the 500-ms edge check.\n33 / 28 predictions completed; 29 / 26 were accepted by the map.',fontsize=23,color=dark,linespacing=1.55)
    fig.text(.065,.085,'Caption: terminal bars are disjoint; downstream observations are not additional frames.\nUE0 frame 52 has no edge event; its cause is unknown. Source isolation and shutdown passed.',fontsize=18,color='#52606D')
    save(fig,'01_frame_outcomes')

    fig=plt.figure(figsize=(24,13.5))
    fig.text(.06,.94,'Delay accumulates before edge inference; exact cause is not isolated',fontsize=29,weight='bold',color=dark)
    ax=fig.add_axes([.075,.51,.405,.31]);bx=fig.add_axes([.565,.51,.385,.31])
    for s,color in [('m4-ue0',teal),('m4-ue1',purple)]:
        own=[r for r in data if r['stream_id']==s];ue=s[-1]
        x=sorted(r['capture_to_verified_wall_ms'] for r in own if r['capture_to_verified_wall_ms'] is not None)
        ax.step([0]+x+[6500],[0]+list(np.arange(1,len(x)+1)/208)+[len(x)/208],where='post',color=color,lw=3,label=f'UE{ue}: {len(x)}/208 observed')
        dec=[r for r in own if r['kind']=='POLICY_DECISION']
        for key,ls,label in [('action_to_prediction_raw_ms','-',f'UE{ue} prediction'),('action_to_feedback_ms','--',f'UE{ue} scored feedback')]:
            x=sorted(r[key] for r in dec if r[key] is not None)
            bx.step([0]+x+[700],[0]+list(np.arange(1,len(x)+1)/len(dec))+[len(x)/len(dec)],where='post',color=color,ls=ls,lw=3,label=f'{label} ({len(x)}/{len(dec)})')
    ax.axvline(500,color=red,lw=2,ls=':');ax.text(650,.63,'500-ms processing horizon\n(capture → admission)',color=red,fontsize=18)
    ax.set(xlim=(0,6500),ylim=(0,1.02),xlabel='Capture → edge verification observation (ms)',ylabel='Fraction of all sent frames')
    ax.set_title('Includes stale frames; verification is after reassembly',loc='left',pad=18,fontsize=21)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='lower right',frameon=False,fontsize=17);ax.grid(alpha=.15)
    bx.axvline(170,color=red,lw=2,ls=':');bx.text(184,.175,'170-ms original\nfeedback deadline',color=red,fontsize=18)
    bx.set(xlim=(0,700),ylim=(0,.22),xlabel='Action-open → completion / feedback receipt (ms)',ylabel='Fraction of all policy decisions')
    bx.set_title('Incomplete predictions remain in the denominator',loc='left',pad=18,fontsize=21)
    bx.yaxis.set_major_formatter(PercentFormatter(1));bx.legend(loc='lower right',frameon=False,fontsize=15);bx.grid(alpha=.15)
    cx=fig.add_axes([.27,.15,.56,.22])
    keys=[('rgb_to_send_ms','RGB receipt → last send'),('send_to_verified_ms','Last send → verified frame'),
          ('edge_queue_ms','Edge queue (survivors only)'),('edge_compute_ms','Edge compute (survivors only)'),
          ('prediction_to_feedback_ms','Prediction → scored reply')]
    for y,(key,label) in enumerate(keys):
        for s,color,dy in [('m4-ue0',teal,-.14),('m4-ue1',purple,.14)]:
            a=summaries[s]['duration_ms'][key]; yy=y+dy
            cx.plot([a['p50'],a['p95']],[yy,yy],color=color,lw=3)
            cx.plot(a['p50'],yy,'o',color=color,ms=6);cx.plot(a['p95'],yy,'|',color=color,ms=9)
            cx.text(9500,yy,f"n={a['n']}; {a['p50']:.1f} / {a['p95']:.1f} ms",color=color,va='center',fontsize=14)
    cx.set_yticks(range(len(keys)),[v[1] for v in keys]);cx.invert_yaxis();cx.set_xscale('log');cx.set_xlim(.1,7500)
    cx.set_xticks([.1,1,10,100,1000]);cx.get_xaxis().set_major_formatter(ScalarFormatter());cx.grid(axis='x',alpha=.15)
    cx.set_xlabel('Separate interval P50 ● — P95 | (ms, logarithmic axis); not a summed total',fontsize=18)
    fig.text(.065,.052,'Caption: edge verification is a logged upper-bound observation, not a kernel-arrival timestamp. Wall-clock ages (left)\nand MONOTONIC_RAW policy times (right) use distinct anchors. Post-send delay includes radio, host buffering and receiver work.',fontsize=17,color='#52606D')
    save(fig,'02_delay_localization')


if __name__=='__main__':
    if '--worker' in sys.argv:main()
    else:supervise()
