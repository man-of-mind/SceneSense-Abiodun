"""Registration-bound M4 readers. Per UE first; combined counts are observations."""
import collections
import csv
import json
import math
from pathlib import Path
import statistics

from . import configuration as C
from .identity import require


def document(path, default=None):
    return json.loads(path.read_bytes()) if path.is_file() else default


def lines(path):
    if not path.is_file(): return []
    with path.open() as stream: return [json.loads(row) for row in stream if row.strip()]


def csv_rows(path):
    if not path.is_file(): return []
    with path.open() as stream: return list(csv.DictReader(stream))


def quantiles(values, unit):
    vals=sorted(float(x) for x in values if x not in ('',None))
    require(all(math.isfinite(x) for x in vals),'nonfinite analysis measurement')
    return dict(n=len(vals),unit=unit,p50=None if not vals else statistics.median(vals),
                p99=None if not vals else vals[min(len(vals)-1,math.ceil(.99*len(vals))-1)])


def truth(value): return str(value).lower() in ('true','1')


def pusch_rows(path):
    if not path.is_file(): return []
    text=path.read_text().splitlines()
    header=next((i for i,r in enumerate(text) if ',time,rnti,frame,slot,snrx10,' in r),None)
    if header is None: return []
    fields=['ingest_wall_ns','ingest_raw_ns']+text[header].split(',')[2:]
    return [r for r in csv.DictReader(text[header+1:],fieldnames=fields) if r.get('rnti','').isdigit()]


def clean_report(value):
    if isinstance(value,list): return all(clean_report(v) for v in value)
    if not isinstance(value,dict): return True
    return (not any(value.get(k) for k in ('error','errors','cleanup_error','failure','worker_alive'))
            and value.get('shutdown_verified',True) is True
            and all(clean_report(v) for v in value.values()))


def smoke_assessment(capture):
    """A bounded data-path check; never substitutes for full-route acceptance."""
    capture=Path(capture)
    route=document(capture/'TWO_ROUTE_RESULTS.json',{})
    maps=lines(capture/'map/observations.jsonl')
    events=lines(capture/'edge/events.jsonl')
    sources=document(capture/'ATTACHED.json',{}).get('sources',[])
    gates={}
    for source in sources:
        stream=source['stream_id']; result=route.get('results',{}).get(stream,{})
        bounded=result.get('completed') or ('exceeded the 20 s simulated budget' in result.get('abort_reason',''))
        gates[stream]=bool(bounded and not result.get('collision_count') and not result.get('watchdog_aborted')
            and not result.get('intervention_count') and route.get('drained',{}).get(stream)
            and any(r['source']==source and r['kind']=='prediction_ready' for r in events)
            and any(r['source']==source and r['map_feedback']['accepted'] for r in maps)
            and document(capture/'edge'/f'{stream}_GT_FEEDBACK.json',[]))
    return dict(scope='20-s two-source attach/data-path smoke ONLY; no full-route or calibration qualification',
                gates=gates,pass_=len(gates)==2 and all(gates.values()))


def analyze(preparation,capture,output):
    reg=C.validate(preparation); capture=Path(capture); output=Path(output)
    require(not output.exists(),'analysis output already exists')
    reference=document(capture/'PREPARATION_REFERENCE.json',{})
    require(reference.get('sha256')==reg['sha256'],'capture belongs to a different preparation')
    run=document(capture/'RUN_RESULT.json',{})
    attached=document(capture/'ATTACHED.json',{})
    sources={v['stream_id']:v for v in attached.get('sources',[])}
    events=lines(capture/'edge/events.jsonl'); maps=lines(capture/'map/observations.jsonl')
    radio=pusch_rows(capture/'pusch/ttracer/live_pusch_with_ingest.csv')
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import rescore
    from rl_agent.splitfusion_m3_route_b_v20.postflight_p3 import install_p3
    from rl_agent.splitfusion_m3_route_b_v3_analysis import route_geometry as RG
    from rl_agent.splitfusion_m3_route_b_v3 import contract as M3
    actor=load_registered_actor(); per_ue={}; table=[]
    for expected in reg['config']['sources']:
        stream=expected['stream_id']; directory=capture/stream
        evidence=document(directory/'run4_phase6/PHASE6_UE_EVIDENCE.json',{})
        frames=csv_rows(directory/'per_frame_metrics.csv')
        plans=evidence.get('decisions',[])
        decisions=[r for r in plans if r['kind']=='POLICY_DECISION']
        decision_by_frame={r['frame_id']:r for r in decisions}
        replay=[]
        for row in decisions:
            actual=actor.act_on_vector(tuple(row['features']))
            replay.append(dict(frame_id=row['frame_id'],exact=(actual.mode_id==row['mode_id'] and actual.q_e4==row['q_e4'])))
        ready=[r for r in events if r['source']['stream_id']==stream and r['kind']=='prediction_ready']
        ready_ms=[(r['prediction_ready_wall_ns']-r['capture_timestamp_ns'])/1e6 for r in ready]
        ready_decision=[r for r in ready if r['frame_id'] in decision_by_frame]
        feedback=evidence.get('feedback_rows',[])
        feedback_ms=[]; timely_scored=0; timely_feedback=0
        for row in feedback:
            planned=decision_by_frame.get(row['frame_id'])
            if planned is None: continue
            elapsed=(row['receipt']['ns']-planned['action_open']['ns'])/1e6
            feedback_ms.append(elapsed)
            if 0<=elapsed<=170:
                timely_feedback+=1
                timely_scored+=int(row.get('kind')=='DELIVERED_SUCCESS' and row.get('q_perc') is not None)
        service=directory/'numeric_gt/service'
        gt=lines(service/'FRAMES.jsonl')
        measures=[]; rescored=[]; scored_undefined=0
        for path in sorted((service/'archive').glob('frame_*/scorer/numerical_score.json')):
            value=document(path); measures.append(value['measurement'])
            scored_undefined+=not value['quality_defined']
            with install_p3():
                rescored.append(dict(frame_id=value['measurement']['frame_id'],exact=bool(rescore(path.parent))))
        quality={}
        for cls,threshold in (('vehicle',1.),('person',1.2)):
            eligible=sum(m['loc_'+cls+'_eligible_gt'] for m in measures)
            matched=[float(e) for m in measures for e in m['loc_'+cls+'_matched_xy_errors_m']]
            intersection=sum(m['seg_'+cls+'_intersection_pixels'] for m in measures)
            union=sum(m['seg_'+cls+'_union_pixels'] for m in measures)
            quality[cls]=dict(scored_frames=len(measures),eligible_gt_objects=eligible,matched_objects=len(matched),
                misses=sum(m['loc_'+cls+'_fn'] for m in measures),false_positives=sum(m['loc_'+cls+'_fp'] for m in measures),
                localisation_threshold_m=threshold,within_threshold=sum(e<=threshold for e in matched),
                conservative_fraction=None if not eligible else sum(e<=threshold for e in matched)/eligible,
                zero_eligible_scored_frames=sum(m['loc_'+cls+'_eligible_gt']==0 for m in measures),
                segmentation_intersection_pixels=intersection,segmentation_union_pixels=union,
                segmentation_iou=None if not union else intersection/union)
        installed=[r for r in maps if r['source']['stream_id']==stream and r['map_feedback']['accepted']]
        freshness=[(r['install_wall_s']*1e9-r['wire_document']['capture_timestamp_ns'])/1e6 for r in installed]
        own_radio=[r for r in radio if stream in sources and int(r['rnti'])==sources[stream]['rnti']]
        score_frames={m['frame_id'] for m in measures}
        terminals=[r for r in csv_rows(directory/'map_feedback.csv') if truth(r.get('terminal'))]
        counts=collections.Counter(int(r['frame_id']) for r in terminals)
        sent_frames={int(r['frame_id']) for r in frames if r.get('prepare_status')=='SENT'}
        source_terminals=all(r['stream_id']==stream for r in terminals)
        pose_path=directory/'owned_pose/pose_evidence.jsonl'
        poses=RG.poses_from_archive(pose_path) if pose_path.is_file() else []
        coverage=RG.ordered_coverage([(r[1],r[2]) for r in poses],RG.load_route(C.ROOT))
        coverage.pop('matched_index',None)
        operations=[]
        for frame in gt:
            queued=collections.defaultdict(collections.deque)
            for row in frame.get('m4_enqueue',[]): queued[row['worker'],row['op']].append(row['raw_ns'])
            for row in frame.get('requests',[]):
                timing=row.get('timing') or {}; key=row['worker'],row['op']
                if timing and queued[key]:
                    operations.append(dict(frame_id=frame['frame_id'],worker=key[0],op=key[1],
                        queue_ms=(row['begin']-queued[key].popleft())/1e6,cpu_ms=timing['thread_cpu_ns']/1e6))
        result=dict(source=sources.get(stream),empty_or_incomplete=not plans or not run or bool(run.get('error')),
            offered=len(frames),sent=sum(r.get('prepare_status')=='SENT' for r in frames),
            completed=len(ready),decision_predictions_completed=len(ready_decision),decisions=len(decisions),
            holds=sum(r['kind']=='POLICY_HOLD' for r in plans),fallbacks=sum(r['kind']=='FALLBACK' for r in plans),
            timely_prediction_capture_to_ready=sum(0<=v<=170 for v in ready_ms),
            timely_decision_prediction_capture_to_ready=sum(0<=(r['prediction_ready_wall_ns']-r['capture_timestamp_ns'])/1e6<=170 for r in ready_decision),
            decision_completion_fraction=None if not decisions else len(ready_decision)/len(decisions),
            timely_feedback_action_open_to_receipt=timely_feedback,timely_scored_feedback=timely_scored,
            prediction_ready_latency=quantiles(ready_ms,'ms, capture WALL -> prediction-ready WALL'),
            feedback_latency=quantiles(feedback_ms,'ms, action-open MONOTONIC_RAW -> R4FB receipt MONOTONIC_RAW'),
            feedback_outcomes=dict(collections.Counter(r.get('kind') for r in feedback)),
            gt_scored=len(score_frames),gt_missing_scored_decisions=len(decision_by_frame.keys()-score_frames),
            scored_undefined_quality=scored_undefined,
            one_terminal_per_sent_frame=bool(sent_frames) and source_terminals and set(counts)==sent_frames
                and all(v==1 for v in counts.values()),
            controller_clean=bool(evidence) and evidence.get('faulted') is False
                and evidence.get('unresolved_tickets_at_close')==0,
            transport_terminal_outcomes=dict(collections.Counter(r['outcome'] for r in terminals)),
            ordered_route_coverage=coverage,
            gt_outcomes=dict(collections.Counter(r.get('outcome') for r in gt)),quality=quality,
            payload=quantiles([r['payload_bytes'] for r in frames if r.get('prepare_status')=='SENT'],'bytes, transmitted feature payload'),
            modes=dict(collections.Counter(str(r['mode_id']) for r in decisions)),
            q_e4=quantiles([r['q_e4'] for r in decisions],'q_e4, policy decisions'),
            map_installs=len(installed),map_freshness=quantiles(freshness,'ms, capture -> actual installation'),
            commanded_snr_db=23.5,measured_pusch_snr=quantiles([float(r['snrx10'])/10 for r in own_radio],'dB, per transmission'),
            measured_pusch_mcs=quantiles([r['mcs'] for r in own_radio],'MCS index, per transmission'),
            measured_pusch_allocated_rb=quantiles([r['rbSize'] for r in own_radio],'PRB, per transmission'),
            measured_pusch_tb_bytes=quantiles([r['tb_size'] for r in own_radio],'bytes, per transmission'),
            harq_retransmissions=None,harq_note='Retained PUSCH event does not expose a HARQ round; no inferred count',
            edge_queue=quantiles([r['queue_ms'] for r in ready],'ms, reassembly -> compute start'),
            edge_compute=quantiles([r['compute_ms'] for r in ready],'ms, compute start -> CPU prediction ready'),
            two_ue_calibration_claim=False,gt_queue_and_cpu=operations,
            actor_replay=replay,exact_rescoring=rescored,
            latency_rule='STALE/timeout receipts never count as completed predictions; missing completions stay in decision denominator')
        per_ue[stream]=result
        table.append({k:result[k] for k in ('offered','sent','completed','decisions','holds','fallbacks',
                     'timely_prediction_capture_to_ready','timely_feedback_action_open_to_receipt',
                     'timely_scored_feedback','gt_scored','gt_missing_scored_decisions','map_installs')}|dict(stream=stream))
    route=document(capture/'TWO_ROUTE_RESULTS.json',{})
    density=document(capture/'route_metrics_summary.json',{})
    cleanup=document(capture/'CLEANUP.json',{})
    gates=dict(registration=True,run_ended=run.get('status')=='CAPTURE_ENDED' and not run.get('error') and run.get('returncode')==0,
        both_routes=set(route.get('completed',{}))==set(per_ue) and all(route.get('completed',{}).values()),
        both_drained=set(route.get('drained',{}))==set(per_ue) and all(route.get('drained',{}).values()),
        retained_scenario_qualification=density.get('status')=='PASS',
        both_ordered_route_coverage=all(r['ordered_route_coverage']['coverage']>=M3.PROGRESS_COVERAGE_MIN for r in per_ue.values()),
        both_transport_accounting=all(r['one_terminal_per_sent_frame'] for r in per_ue.values()),
        both_controllers_clean=all(r['controller_clean'] for r in per_ue.values()),
        retention_finalized=all(document(capture/s/'RETENTION_RESULT.json',{}).get('finalized') is True
            and not document(capture/s/'RETENTION_RESULT.json',{}).get('faults') for s in per_ue),
        both_have_predictions=all(r['completed']>0 for r in per_ue.values()),
        both_actor_replays_exact=all(r['actor_replay'] and all(v['exact'] for v in r['actor_replay']) for r in per_ue.values()),
        exact_rescoring=all(r['exact_rescoring'] and all(v['exact'] for v in r['exact_rescoring']) for r in per_ue.values()),
        both_pusch_measured=all(r['measured_pusch_snr']['n']>0 for r in per_ue.values()),
        shared_edge_closed=bool(document(capture/'edge/CLOSED.json')) and clean_report(document(capture/'edge/CLOSED.json')),
        cleanup_present=bool(cleanup) and clean_report(cleanup))
    combined={key:sum(row[key] for row in table) for key in table[0] if key!='stream'}
    result=dict(schema='splitfusion.m4.assessment.v1',preparation_sha256=reg['sha256'],
        verdict='INTEGRATION_PASS_CHANNEL_FIDELITY_REVIEW_REQUIRED' if all(gates.values()) else 'STOP',
        gates=gates,per_ue=per_ue,combined=combined,
        interpretation='Combined counts are source observations, not unique objects or cooperative-perception gains. No two-UE channel calibration or timeliness performance is assumed.',
        denominator='Each UE: all offered frames; policy fractions use all POLICY_DECISION frames. Quality uses only scored frames, with every eligible GT object including misses. Missing GT is not zero-eligible GT.')
    output.mkdir(parents=True); C.write_new(output/'ASSESSMENT.json',result)
    with (output/'PER_UE.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return dict(verdict=result['verdict'],gates=gates,output=str(output),combined=combined)
