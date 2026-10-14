"""Compact spatial summaries from existing M3 readers and recorded numeric scores; no inference."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from rl_agent.splitfusion_m3_route_b_v17_analysis import analyze as A,telemetry as T
from rl_agent.splitfusion_fixed_action_screen_analysis_v2.report import quality_fields
from rl_agent.splitfusion_m3_route_b_v19.spatial_field import field
from rl_agent.splitfusion_m3_route_b_v19 import contract as C


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=1,sort_keys=True,allow_nan=False)
def number(x):return None if x in ('',None) else float(x)
def truth(x):return x is True or x=='True'


def summarize(rows):
    n=len(rows);scored=[r for r in rows if r['service_outcome']=='SCORED']
    quality={}
    for cls in ('vehicle','person'):
        q=A.pooled(scored,cls);q.pop('errors')
        timely_hits=sum(sum(e<=A.TOL[cls] for e in r[cls+'_matched_xy_errors_m'] or []) for r in scored if r['timely_feedback'])
        q.update(feedback_timely_hits=timely_hits,feedback_aware_recall=A.ratio(timely_hits,q['eligible_gt']))
        quality[cls]=q
    return dict(n=n,predictions_completed=sum(r['prediction_completed'] for r in rows),
        timely_predictions=sum(r['prediction_completion_ms'] is not None and r['prediction_completion_ms']<=170 for r in rows),
        feedback_received=sum(r['feedback_receipt_ms'] is not None for r in rows),
        timely_feedback=sum(r['timely_feedback'] for r in rows),
        timely_scored_feedback=sum(r['timely_feedback'] for r in scored),
        timely_defined_feedback=sum(r['timely_feedback'] and r['quality_defined'] for r in scored),
        scored_gt=len(scored),scored_zero_eligible=sum((r['vehicle_eligible_gt'] or 0)+(r['person_eligible_gt'] or 0)==0 for r in scored),
        unavailable_gt=dict(Counter(r['service_outcome'] for r in rows if r['service_outcome']!='SCORED')),
        prediction_ms=A.q([r['prediction_completion_ms'] for r in rows]),feedback_ms=A.q([r['feedback_receipt_ms'] for r in rows]),
        payload_bytes=A.q([r['payload_l2_bytes'] for r in rows]),
        map_installed=sum(r['map_outcome']=='RESULT_INSTALLED' for r in rows),
        decision_map_age_ms=A.q([r['map_age_at_install_ms'] for r in rows]),
        target_snr_db=A.q([r['snr_proxy_db'] for r in rows]),measured_pusch_at_decision_db=A.q([r['measured_pusch_snr_db'] for r in rows]),
        prior_ul_mcs=A.q([r['prior_ul_mcs'] for r in rows]),mode_counts=dict(Counter(str(r['mode_id']) for r in rows)),
        q11=A.q([r['q_e4']/10000 for r in rows if r['mode_id']==11]),
        quality=quality,denominator='ALL policy decisions in the ACK-pose region; localisation uses all known eligible GT including misses')


def main(capture,analysis,layers,out):
    out.mkdir(exist_ok=False)
    cell=next((capture/'capture/cells').iterdir())
    timeline=A.rows_csv(analysis/'DECISION_TIMELINE.csv')
    layer={int(r['frame_id']):r for r in A.rows_csv(layers/'ROUTE_LAYERS_JOINED.csv')}
    snr={r['frame_id']:r for r in read(cell/'run5_policy/SNR_EVIDENCE.json')['decision_snr']}
    raw=lines(cell/'numeric_gt/service/FRAMES.jsonl');frames={r['frame_id']:r for r in raw if r.get('decision')}
    radio=A.rows_csv(cell/'radio_trace.csv')
    commands={f"ack-{int(r['step_index'])+1}":r for r in radio if r['command_timing_status'].startswith('ACK_')}
    acks=[r for r in lines(cell/'run5_policy/CONTROLLER_ACKS.jsonl') if r.get('valid') and r['command_id']==f"ack-{r['sequence']}"]
    assert len(commands)==len(acks)
    ack_times=np.array([r['effective_raw_ns'] for r in acks],dtype=np.int64)
    rows=[]
    for t in timeline:
        fid=int(t['frame_id']);f=frames[fid];s=snr[fid];cmd=commands[s['snr']['active_command_id']]
        assert s['admitted'] and float(t['snr_proxy_db'])==s['snr']['value_db']==float(cmd['target_snr_db'])
        assert t['service_outcome']==f['outcome']
        r=dict(t,frame_id=fid,decision_index=int(t['decision_index']),command_id=s['snr']['active_command_id'],
            spatial_region=cmd['region'],action_pose_region=field(float(t['ego_x']),float(t['ego_y']))['region'],
            command_pose_x=float(cmd['ego_x']),command_pose_y=float(cmd['ego_y']),command_pose_frame=int(cmd['source_frame_id']),
            mode_id=int(t['mode_id']),q_e4=int(t['q_e4']),
            prediction_completed=truth(t['prediction_completed']),timely_feedback=truth(t['timely_170']),
            quality_defined=truth(t['quality_defined']),**quality_fields((f.get('result') or {}).get('measurement')))
        for k in ('prediction_completion_ms','feedback_receipt_ms','payload_l2_bytes','map_age_at_install_ms','snr_proxy_db','prior_ul_mcs','capture_wall_s','t_since_first_capture_s','ego_x','ego_y'):
            r[k]=number(t[k])
        r['measured_pusch_snr_db']=number(layer[fid]['measured_pusch_snr_db'])
        r['measured_final_mcs']=number(layer[fid]['measured_final_mcs'])
        rows.append(r)
    assert len(rows)==len(frames)
    groups={region:summarize([r for r in rows if r['spatial_region']==region]) for region in ('HIGH','TRANSITION','LOW_CORE')}
    core=[r['decision_index'] for r in rows if r['spatial_region']=='LOW_CORE']
    context={}
    if core:
        context=dict(high_before_entry=summarize([r for r in rows if r['spatial_region']=='HIGH' and r['decision_index']<min(core)]),
                     high_after_exit=summarize([r for r in rows if r['spatial_region']=='HIGH' and r['decision_index']>max(core)]),
                     core_first_decision=min(core),core_last_decision=max(core))
    # Exact retained telemetry reader; observer-time attribution, not per-payload packet/HARQ identity.
    streams=T.load(capture/'capture/m3_radio/ttracer')
    first,last=min(r['capture_wall_s'] for r in rows),max(r['capture_wall_s'] for r in rows)
    measured=[];unmatched=0
    for p in streams['pusch']['rows']:
        if not first<=p['wall_ns']/1e9<=last:continue
        i=int(np.searchsorted(ack_times,p['raw_ns'],side='right')-1)
        if i<0:unmatched+=1;continue
        a=acks[i];cmd=commands[a['command_id']]
        measured.append(dict(wall_ns=p['wall_ns'],raw_ns=p['raw_ns'],command_id=a['command_id'],region=cmd['region'],
            target_db=float(cmd['target_snr_db']),pusch_snr_db=p['pusch_snr_db'],mcs=p['mcs'],
            residual_measured_minus_target_db=p['pusch_snr_db']-float(cmd['target_snr_db']),
            ack_age_ms=(p['raw_ns']-a['effective_raw_ns'])/1e6))
    fidelity={}
    for region in ('ALL','HIGH','TRANSITION','LOW_CORE'):
        rr=[r for r in measured if region=='ALL' or r['region']==region]
        fidelity[region]=dict(transmissions=len(rr),target_db=A.q([r['target_db'] for r in rr],(5,50,95)),
            measured_pusch_db=A.q([r['pusch_snr_db'] for r in rr],(5,50,95)),mcs=A.q([r['mcs'] for r in rr]),
            mcs_histogram=dict(Counter(str(r['mcs']) for r in rr)),
            measured_minus_target_db=A.q([r['residual_measured_minus_target_db'] for r in rr],(5,50,95)),
            absolute_residual_db=A.q([abs(r['residual_measured_minus_target_db']) for r in rr]),
            ack_age_ms=A.q([r['ack_age_ms'] for r in rr]))
    # AoI time samples are grouped by contemporaneous ACK region, not by decision count.
    ack_wall=np.array([int(commands[a['command_id']]['command_ack_wall_ns'])/1e9 for a in acks])
    aoi_groups={k:[] for k in ('HIGH','TRANSITION','LOW_CORE','UNJOINED')}
    for r in A.rows_csv(analysis/'MAP_AOI_TIMELINE.csv'):
        i=int(np.searchsorted(ack_wall,float(r['wall_s']),side='right')-1)
        region='UNJOINED' if i<0 else commands[acks[i]['command_id']]['region']
        aoi_groups[region].append(float(r['aoi_ms']))
    for region,g in groups.items():g['map_aoi_time_samples_ms']=A.q(aoi_groups[region])
    table=[]
    for region,g in groups.items():
        table.append(dict(region=region,decisions=g['n'],prediction_by170=g['timely_predictions'],feedback_by170=g['timely_feedback'],
            scored_feedback_by170=g['timely_scored_feedback'],scored_gt=g['scored_gt'],zero_eligible=g['scored_zero_eligible'],
            gt_unavailable=sum(g['unavailable_gt'].values()),prediction_p50_ms=g['prediction_ms']['p50'],prediction_p99_ms=g['prediction_ms']['p99'],
            feedback_p50_ms=g['feedback_ms']['p50'],feedback_p99_ms=g['feedback_ms']['p99'],
            payload_p50_kB=None if g['payload_bytes']['p50'] is None else g['payload_bytes']['p50']/1000,
            payload_p99_kB=None if g['payload_bytes']['p99'] is None else g['payload_bytes']['p99']/1000,
            vehicle_recall_1m=g['quality']['vehicle']['recall_within_tolerance'],pedestrian_recall_1p2m=g['quality']['person']['recall_within_tolerance'],
            vehicle_iou=g['quality']['vehicle']['seg_iou_pooled'],pedestrian_iou=g['quality']['person']['seg_iou_pooled'],
            decision_map_age_p50_ms=g['decision_map_age_ms']['p50'],decision_map_age_p99_ms=g['decision_map_age_ms']['p99'],
            measured_pusch_p50_db=fidelity[region]['measured_pusch_db']['p50']))
    A.write_csv(out/'REGIONS.csv',table);A.write_csv(out/'DECISIONS_SPATIAL.csv',rows);A.write_csv(out/'PUSCH_SPATIAL.csv',measured)
    save(out/'SUMMARY.json',dict(whole=summarize(rows),regions=groups,context=context,fidelity=fidelity,
         action_pose_vs_ack_region_mismatches=sum(r['action_pose_region']!=r['spatial_region'] for r in rows),
         measured_transmissions_without_preceding_ack=unmatched,unjoined_aoi_samples=len(aoi_groups['UNJOINED']),
         region_definition='Causal observed pose behind the ACKed command admitted to the actor; current action-position region also retained',
         measurement_alignment='Latest preceding ACK in host RAW clock at telemetry ingest; not packet identity or zero-lag propagation proof',
         quality_definition='All SCORED records including zero-eligible pixels/false positives; localisation denominator all known eligible GT. Missing/refused GT excluded as unknown, counted separately.',
         no_endpoint_search=True,perception_runs=0,rescore_runs=0))
    inputs=[cell/'numeric_gt/service/FRAMES.jsonl',cell/'run5_policy/SNR_EVIDENCE.json',cell/'run5_policy/CONTROLLER_ACKS.jsonl',
            cell/'radio_trace.csv',analysis/'DECISION_TIMELINE.csv',analysis/'MAP_AOI_TIMELINE.csv',layers/'ROUTE_LAYERS_JOINED.csv',Path(__file__)]
    inputs.extend(Path(v['file']) for v in streams.values())
    save(out/'INPUT_SHA256.json',{str(p.resolve()):C.sha(p) for p in inputs})
    save(out/'OUTPUT_SHA256.json',{p.name:C.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(decisions=len(rows),regions={k:g['n'] for k,g in groups.items()},fidelity_samples=len(measured))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--layers',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.capture,a.analysis,a.layers,a.out)
