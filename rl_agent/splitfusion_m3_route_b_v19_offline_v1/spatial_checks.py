"""Spatial registration-bound row/pose/ACK/actor checks, independent of performance."""
import csv
import json
import math
from pathlib import Path

from rl_agent import ue_target_snr_cell_runtime_v1 as BASE
from rl_agent.splitfusion_m3_route_b_v19.spatial_field import field


def check_records(schedule, radio, samples, poses, acknowledgements, snr, mapping, replay):
    if not radio or len(samples) != len(radio):
        raise ValueError('missing or unjoined spatial/radio rows')
    by_frame = {p['source_frame_id']: p for p in poses if p.get('status') == 'ACTIVE'}
    if len(by_frame) != sum(p.get('status') == 'ACTIVE' for p in poses):
        raise ValueError('duplicate source pose frame')
    ack_by_sequence = {a['sequence']: a for a in acknowledgements}
    ack_commands = {}
    regions, maximum_age = {}, 0.0
    for i, (r, sample) in enumerate(zip(radio, samples)):
        if (int(r['step_index']) != i or sample['sequence_index'] != i
                or r['profile_id'] != schedule['schedule_id'] or r['trace_id'] != schedule['trace_id']):
            raise ValueError('spatial trace identity/order mismatch')
        p = by_frame.get(int(r['source_frame_id']))
        if p is None:
            raise ValueError('spatial command lacks an observed route pose')
        for key in ('ego_x','ego_y','source_frame_id','source_monotonic_ns','source_monotonic_raw_ns','source_wall_ns'):
            convert = float if key in ('ego_x','ego_y') else int
            if convert(r[key]) != convert(sample[key]) or convert(r[key]) != convert(p[key]):
                raise ValueError('spatial pose provenance mismatch: '+key)
        if r['pose_status'] != sample['pose_status']:
            raise ValueError('spatial pose status mismatch')
        expected = field(float(r['ego_x']), float(r['ego_y']))
        if any(abs(float(r[k])-expected[k]) > 1e-12 for k in ('target_snr_db','rho','low_weight')):
            raise ValueError('spatial formula mismatch')
        if r['region'] != sample['region'] or r['region'] != expected['region']:
            raise ValueError('spatial region mismatch')
        if sample['target_snr_db'] != float(r['target_snr_db']):
            raise ValueError('selected target differs from actuator row')
        age = (int(r['selected_monotonic_ns'])-int(r['source_monotonic_ns']))/1e6
        bound = schedule['max_pose_sample_age_ms'] if r['pose_status'] == 'ACTIVE' else schedule['terminal_drain_ms']
        if r['pose_status'] not in ('ACTIVE','ROUTE_COMPLETED','ROUTE_INCOMPLETE') or not 0 <= age <= bound:
            raise ValueError('spatial source freshness/status failure')
        if abs(age-float(r['pose_age_ms'])) > 1e-9:
            raise ValueError('spatial age accounting mismatch')
        mapped = replay.inverse_interpolate(expected['target_snr_db'], mapping, 0.25)
        if not math.isclose(mapped, float(r['mapped_rfsim_command_db']), abs_tol=1e-12):
            raise ValueError('spatial calibration mapping mismatch')
        status = r['command_timing_status']
        if status not in ('ACK_ON_TIME','ACK_LATE','SKIP_OBSOLETE_NEVER_BURST'):
            raise ValueError('unvalidated spatial command')
        a = ack_by_sequence.get(i+1)
        if a is None:
            raise ValueError('missing causal controller publication')
        if status.startswith('ACK_'):
            sent, ack = int(r['command_send_monotonic_ns']), int(r['command_ack_monotonic_ns'])
            if not int(r['selected_monotonic_ns']) <= sent <= ack:
                raise ValueError('spatial command chronology failure')
            if r['pose_status'] == 'ACTIVE' and (sent-int(r['source_monotonic_ns']))/1e6 > schedule['max_pose_to_command_ms']:
                raise ValueError('spatial command uses stale pose')
            if (a['valid'] is not True or a['target_snr_db'] != expected['target_snr_db']
                    or a['command_id'] != f'ack-{i+1}' or a['effective_raw_ns'] < int(r['selected_monotonic_raw_ns'])):
                raise ValueError('ACK publication differs from effective spatial target')
            ack_commands[a['command_id']] = a
        elif a['valid'] and (a['command_id'] not in ack_commands
                             or a['target_snr_db'] != ack_commands[a['command_id']]['target_snr_db']):
            raise ValueError('skipped command replaced acknowledged target')
        regions[r['region']] = regions.get(r['region'],0)+1
        maximum_age = max(maximum_age,age)
    admitted = [d for d in snr['decision_snr'] if d.get('admitted')]
    if not admitted:
        raise ValueError('no admitted actor SNR observations')
    receipts = snr['provider']['received_snapshots']
    for d in admitted:
        obs, boundary = d['snr'], d['boundary']
        a = ack_commands.get(obs['active_command_id'])
        if (a is None or obs['value_db'] != a['target_snr_db']
                or obs['effective_since_ns'] != a['effective_raw_ns']
                or abs(d['snr_scaled']-(obs['value_db']-5.5)/19) > 1e-12):
            raise ValueError('actor input is not the acknowledged spatial target')
        cutoff = boundary['state_commit_timestamp_ns']
        selected = next((r for r in reversed(receipts) if r['receipt_raw_ns'] < cutoff and r['heartbeat_raw_ns'] < cutoff), None)
        if (selected is None or selected['command_id'] != obs['active_command_id']
                or selected['heartbeat_raw_ns'] != obs['heartbeat_ns']
                or not 0 < boundary['action_open_timestamp_ns']-obs['heartbeat_ns'] <= 200000000):
            raise ValueError('actor spatial target failed retained causal lease/receipt rules')
    return dict(rows=len(radio), selected_samples=len(samples), observed_pose_frames=len(by_frame),
                regions=regions, admitted_decisions=len(admitted),
                not_admitted=sum(not d.get('admitted') for d in snr['decision_snr']),
                maximum_pose_age_ms=maximum_age, actor_inputs='ACKed commanded SNR only; no position/zone',
                commanded_not_measured=True)


def evaluate(cell, binding, root):
    cell, root = Path(cell), Path(root)
    def lines(path):
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    with (cell/'radio_trace.csv').open() as f:
        radio = list(csv.DictReader(f))
    sequence = lines(cell/'run5_policy/SPATIAL_SAMPLES.jsonl')
    samples = [x for x in sequence if x.get('kind') != 'summary']
    summaries = [x for x in sequence if x.get('kind') == 'summary']
    if len(summaries) != 1 or summaries[0]['samples'] != len(samples):
        raise ValueError('spatial writer summary missing or inconsistent')
    poses = lines(cell/'m3_route/SPATIAL_POSE_TRACE.jsonl')
    summary = json.loads((cell/'m3_route/SPATIAL_POSE_SUMMARY.json').read_text())
    if summary.get('status') != 'ROUTE_COMPLETED' or summary.get('error'):
        raise ValueError('spatial pose source did not close with route completion')
    from rl_agent.splitfusion_m3_route_b_v19 import contract as C
    channel = C.channel_registration(root)
    replay = BASE.load_replay_module()
    mapping = BASE.load_mapping(root/channel['mapping_csv'], replay)
    result = check_records(binding.SCHEDULE, radio, samples, poses,
                           lines(cell/'run5_policy/CONTROLLER_ACKS.jsonl'),
                           json.loads((cell/'run5_policy/SNR_EVIDENCE.json').read_text()), mapping, replay)
    if set(result['regions']) != {'HIGH','TRANSITION','LOW_CORE'}:
        raise ValueError('full spatial demonstration did not cover all three regions')
    launch = json.loads((cell/'run5_policy/CONTROLLER_LAUNCH.json').read_text())
    if binding.CONTROLLER_MODULE not in launch['argv']:
        raise ValueError('unregistered spatial controller module')
    return result
