"""Focused spatial field/controller/ACK/checker proof; qualified GT proofs reused."""
import argparse
import ast
import copy
import csv
import importlib.util
import inspect
import json
import math
from pathlib import Path
import time
import uuid

from . import contract as C, host as H, entries as E, runner as R
from . import spatial_controller as SC
from .spatial_field import field
from .spatial_pose import Publisher, atomic_json


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=1,sort_keys=True,allow_nan=False)


def run(name):
    root = C.repo_root()
    out = root/'experiments/splitfusion_m3_route_b_v19'/name
    out.mkdir(parents=True,exist_ok=False)
    checks, info = {}, {}
    approved = C.approved_design(root)
    checks['approved_json_render_and_png_hashes'] = True
    path = root/'experiments/splitfusion_m3_spatial_snr_plan_v1/render_plan.py'
    spec = importlib.util.spec_from_file_location('approved_spatial_plan',path)
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)    # no render/outputs; only the approved pure function
    import numpy as np
    x,y = np.meshgrid(np.linspace(-170,150,101),np.linspace(-100,210,101))
    expected, _, _ = plan.field(x,y)
    actual = np.array([field(a,b)['target_snr_db'] for a,b in zip(x.flat,y.flat)]).reshape(x.shape)
    error = float(np.max(np.abs(expected-actual)))
    checks['exact_approved_formula_grid'] = error < 1e-12
    info['grid_max_error_db'] = error
    points = [(float(a),float(b)) for a,b in zip(x.flat,y.flat)]
    values = [field(a,b) for a,b in points]
    checks['finite_bounds_by_region'] = all(12 <= v['target_snr_db'] <= 23.5 and
        (v['region'] != 'LOW_CORE' or v['target_snr_db'] <= 14) and
        (v['region'] != 'HIGH' or v['target_snr_db'] >= 20) for v in values)
    checks['deterministic_and_order_independent'] = values[::-1] == [field(a,b) for a,b in points[::-1]]
    continuity = max(abs(field(-55+80*(r-1e-8)*math.cos(a),140+45*(r-1e-8)*math.sin(a))['target_snr_db']-
                         field(-55+80*(r+1e-8)*math.cos(a),140+45*(r+1e-8)*math.sin(a))['target_snr_db'])
                     for r in (1,1.25) for a in np.linspace(0,2*math.pi,500))
    checks['both_boundaries_continuous'] = continuity < 1e-5
    info['maximum_boundary_jump_db'] = continuity
    channel = C.channel_registration(root)
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    replay = base.load_replay_module()
    mapping = base.load_mapping(root/channel['mapping_csv'],replay)
    checks['existing_calibration_anchors'] = [replay.inverse_interpolate(t,mapping,.25) for t in (12,14,23.5)] == [-6,-7,-12]
    checks['interpolation_preserves_quarter_db_commands'] = all(abs(replay.inverse_interpolate(v['target_snr_db'],mapping,.25)*4-round(replay.inverse_interpolate(v['target_snr_db'],mapping,.25)*4))<1e-10 for v in values[::101])
    checks['native_runner_host_contract_identities'] = R.C is C and H.C is C and E.C is C and SC.C is C
    checks['registered_role_constants'] = all(C.role_constant_audit().values())
    checks['unchanged_containment_accounting_workers'] = C.SERVICE_ENTRY==C.V17.SERVICE_ENTRY and C.WORKER_ENTRY==C.V17.WORKER_ENTRY and C.accounting_registration()==C.V17.accounting_registration()
    from rl_agent.splitfusion_m3_route_b_v17 import entries as OLD
    def scopes(fn):
        return [[ast.unparse(i) for i in w.items] for w in ast.walk(ast.parse(inspect.getsource(fn))) if isinstance(w,ast.With)]
    before, after = scopes(OLD._child), scopes(E._child)
    expected_scopes = copy.deepcopy(before)
    expected_scopes[0][0] = 'install_spatial_controller(attempt)'
    expected_scopes[0].insert(1,'install_spatial_pose(attempt)')
    checks['child_scopes_only_spatial_addition'] = after == expected_scopes
    checks['edge_scopes_unchanged'] = scopes(E.edge) == scopes(OLD.edge)
    # Exercise actual pose writer, sequence, CSV writer and existing causal publisher/provider.
    fixture = out/'fixture';fixture.mkdir()
    publisher = Publisher(fixture)
    seq = SC.SpatialSequence(channel['schedule'],publisher.snapshot,fixture/'SAMPLES.jsonl')
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_target_snr_runtime as legacy
    from rl_agent.splitfusion_run5_live_avo_v1 import snr_live as S
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
    controller, session = str(uuid.uuid4()), str(uuid.uuid4())
    ack = S.ControllerPublisher(fixture/'unused.sock',controller,audit_path=fixture/'ACKS.jsonl')
    provider = S.LiveSnrProvider(fixture/'unused.sock',controller,'test-ue',start_receiver=False)
    provider.adopt_session(session)
    decisions=[]; old_fields=legacy.FIELDS
    try:
        legacy.FIELDS=(*old_fields,*SC.POSE_FIELDS)
        write = SC.make_spatial_writer(legacy.write_row,[seq])
        with (fixture/'RADIO.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=legacy.FIELDS);writer.writeheader()
            for i,(a,b) in enumerate([(0,0),(-55,140),(35,140)]):
                publisher.publish(100+i,a,b)
                _,target=seq.next_sample();sample=seq.last
                mono=time.monotonic_ns()
                row=dict(step_index=i,target_snr_db=target,profile_id=C.SCHEDULE['schedule_id'],
                         trace_id=C.SCHEDULE['trace_id'],seed=0,mapped_rfsim_command_db=replay.inverse_interpolate(target,mapping,.25),
                         scheduled_monotonic_ns=mono,interval_end_monotonic_ns=mono+100000000,
                         command_send_monotonic_ns=mono,command_ack_monotonic_ns=mono+1,
                         command_timing_status='ACK_ON_TIME')
                write(f,writer,row);ack.on_row(row)
                arow=ack.current;receipt=arow.heartbeat_raw_ns+1
                assert provider.accept_packet(arow.encode(),receipt)
                boundary=R4.DecisionBoundaryV1(R4.DecisionIdentityV1(session,'test-ue',i),receipt+1,receipt+2,S.CLOCK_DOMAIN)
                observation=provider.observe(boundary)
                obs=dict(active_command_id=observation.active_command_id,value_db=observation.value_db,
                         effective_since_ns=observation.effective_since_ns,heartbeat_ns=observation.heartbeat_ns)
                decisions.append(dict(admitted=observation.valid,snr=obs,snr_scaled=(observation.value_db-5.5)/19,
                                      boundary=dict(state_commit_timestamp_ns=receipt+1,action_open_timestamp_ns=receipt+2)))
        publisher.close('ROUTE_COMPLETED');seq.close()
        with (fixture/'RADIO.csv').open() as f: radio=list(csv.DictReader(f))
        samples=[json.loads(l) for l in (fixture/'SAMPLES.jsonl').read_text().splitlines() if '"kind"' not in l]
        poses=[json.loads(l) for l in publisher.trace_path.read_text().splitlines()]
        acks=[json.loads(l) for l in (fixture/'ACKS.jsonl').read_text().splitlines()]
        snr=dict(decision_snr=decisions,provider=provider.evidence())
        from rl_agent.splitfusion_m3_route_b_v19_offline_v1.spatial_checks import check_records
        checked=check_records(channel['schedule'],radio,samples,poses,acks,snr,mapping,replay)
        checks['actual_writer_output_passes_spatial_checker'] = checked['rows']==3 and set(checked['regions'])=={'HIGH','TRANSITION','LOW_CORE'}
        checks['acknowledged_actor_input_consistency'] = all(d['admitted'] for d in decisions)
        negative={}
        for name,mutate in {
            'wrong_target':lambda rr,ss:rr[1].update(target_snr_db=23.5),
            'wrong_position':lambda rr,ss:rr[1].update(ego_x=0),
            'wrong_region':lambda rr,ss:rr[1].update(region='HIGH'),
            'wrong_mapping':lambda rr,ss:rr[1].update(mapped_rfsim_command_db=-12),
            'one_ns_source_drift':lambda rr,ss:rr[1].update(source_wall_ns=int(rr[1]['source_wall_ns'])+1),
            'unacked_actor_value':lambda rr,ss:ss['decision_snr'][1]['snr'].update(value_db=14),
        }.items():
            rr,ss=copy.deepcopy(radio),copy.deepcopy(snr);mutate(rr,ss)
            try:check_records(channel['schedule'],rr,samples,poses,acks,ss,mapping,replay)
            except (ValueError,RuntimeError):negative[name]=True
            else:negative[name]=False
        checks.update({f'negative_{k}':v for k,v in negative.items()})
        # Startup/stale/future/abort/nonfinite and bounded terminal behaviour in isolated snapshots.
        for name,change in {
            'stale':dict(status='ACTIVE',source_monotonic_ns=time.monotonic_ns()-2000000000),
            'future':dict(status='ACTIVE',source_monotonic_ns=time.monotonic_ns()+10000000000),
            'aborted':dict(status='ABORTED'), 'nonfinite':dict(ego_x='nan',status='ACTIVE'),
            'terminal_expired':dict(status='ROUTE_COMPLETED',source_monotonic_ns=time.monotonic_ns()-21000000000),
        }.items():
            path=fixture/(name+'.json');snapshot=dict(publisher.last,**change);save(path,snapshot)
            bad=SC.SpatialSequence(channel['schedule'],path,fixture/(name+'_trace.jsonl'))
            bad.last=seq.last
            try:bad.next_sample()
            except (ValueError,RuntimeError):checks['reject_'+name]=True
            else:checks['reject_'+name]=False
            finally:bad.close()
        missing=SC.SpatialSequence(channel['schedule'],fixture/'absent.json',fixture/'missing_trace.jsonl')
        try:missing.next_sample()
        except RuntimeError:checks['missing_startup_pose_never_synthesized']=True
        else:checks['missing_startup_pose_never_synthesized']=False
        finally:missing.close()
    finally:
        legacy.FIELDS=old_fields;ack.close();provider.close()
        publisher.close('ABORTED');seq.close()
    result=dict(schema=H.REGRESSION_SCHEMA,status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                info=info,sources=C.exact_sources(root),launcher_binding=C.launcher_binding(),live_services_started=False,
                carla_started=False,oai_started=False,live_inference_started=False)
    save(out/'RESULT.json',result)
    save(out/'OUTPUT_SHA256.json',{str(p.relative_to(out)):C.sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(status=result['status'],checks=checks,info=info),sort_keys=True))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    raise SystemExit(run(p.parse_args().name))
