"""Approved spatial sequence on the unchanged 100-ms actuator and causal ACK IPC."""
import json
import math
from pathlib import Path
import sys
import time

from . import contract as C
from .spatial_field import field

SCHEMA = 'splitfusion.m3_route_b_v19.spatial_controller.v1'
POSE_FIELDS = ('ego_x', 'ego_y', 'rho', 'region', 'low_weight', 'pose_status', 'pose_age_ms',
               'source_frame_id', 'source_monotonic_ns', 'source_monotonic_raw_ns', 'source_wall_ns',
               'selected_monotonic_ns', 'selected_monotonic_raw_ns')


def argument(argv, name):
    hits = [i for i, x in enumerate(argv) if x == name]
    if len(hits) != 1 or hits[0] + 1 >= len(argv):
        raise ValueError('expected exactly one '+name)
    return Path(argv[hits[0]+1])


class SpatialSequence:
    def __init__(self, schedule, pose_path, trace_path, *, monotonic_ns=time.monotonic_ns,
                 raw_ns=lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)):
        self.schedule, self.pose_path = schedule, Path(pose_path)
        self.monotonic_ns, self.raw_ns = monotonic_ns, raw_ns
        self.trace = Path(trace_path).open('x')
        self.samples, self.last_frame = 0, -1
        self.last = None

    def next_sample(self):
        now = self.monotonic_ns()
        # Called only after the retained first-capture gate. No pose -> no command.
        try:
            pose = json.loads(self.pose_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise RuntimeError('observed ego pose absent/unreadable after first-capture gate') from exc
        if (pose.get('schema') != self.schedule['pose_schema'] or pose.get('source') != self.schedule['pose_source']
                or pose.get('clock_domain') != 'CLOCK_MONOTONIC'):
            raise RuntimeError('spatial pose identity/clock mismatch')
        required = ('source_frame_id','source_monotonic_ns','source_monotonic_raw_ns','source_wall_ns','ego_x','ego_y')
        if any(k not in pose for k in required):
            raise RuntimeError('spatial pose missing required fields')
        age = (now-int(pose['source_monotonic_ns']))/1e6
        frame, status = int(pose['source_frame_id']), pose.get('status')
        if age < 0 or frame < self.last_frame:
            raise RuntimeError('future/reordered spatial pose')
        if status == 'ACTIVE':
            if age > self.schedule['max_pose_sample_age_ms']:
                raise RuntimeError('stale active spatial pose')
        elif status in ('ROUTE_COMPLETED','ROUTE_INCOMPLETE') and self.last is not None:
            if age > self.schedule['terminal_drain_ms']:
                raise RuntimeError('spatial terminal pose exceeded bounded drain')
        else:
            raise RuntimeError('spatial pose aborted or terminal before first active sample')
        value = field(pose['ego_x'], pose['ego_y'])
        value.update(schema=SCHEMA, sequence_index=self.samples, pose_status=status, pose_age_ms=age,
                     selected_monotonic_ns=now, selected_monotonic_raw_ns=self.raw_ns(),
                     **{k:pose[k] for k in required if k not in ('ego_x','ego_y')})
        self.trace.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        self.trace.flush()
        self.last_frame, self.last = frame, value
        self.samples += 1
        return 1, value['target_snr_db']  # state index is a discarded trace label, never an actor input

    def close(self):
        if not self.trace.closed:
            self.trace.write(json.dumps(dict(schema=SCHEMA, kind='summary', samples=self.samples,
                                            last_frame=self.last_frame), sort_keys=True)+'\n')
            self.trace.close()


def build_prepare(original, registration, pose_path, trace_path, holder):
    schedule = registration['schedule']

    def prepare_sequence(*, campaign, profile_id):
        if profile_id != schedule['schedule_id']:
            raise ValueError('foreign spatial controller profile')
        original(campaign=campaign, profile_id=C.ORIGINAL_PROFILE)
        if int(campaign['network']['prefix_samples']) != schedule['carrier_validation_prefix_samples']:
            raise ValueError('carrier prefix validation drift')
        if C.schedule_definition_sha256(C.repo_root()) != schedule['trace_sha256']:
            raise ValueError('approved spatial definition digest drift')
        sequence = SpatialSequence(schedule, pose_path, trace_path)
        holder.append(sequence)
        frozen = dict(profile_id=schedule['schedule_id'], trace_id=schedule['trace_id'], seed=0,
                      trace_sha256=schedule['trace_sha256'])
        return sequence, [], frozen
    return prepare_sequence


def rewrite_profile(argv):
    argv = list(argv)
    hits = [i for i, x in enumerate(argv) if x == '--profile-id']
    if len(hits) != 1 or hits[0]+1 >= len(argv) or argv[hits[0]+1] != C.ORIGINAL_PROFILE:
        raise ValueError('expected exactly one --profile-id FAVORABLE_STABLE carrier')
    argv[hits[0]+1] = C.SCHEDULE['schedule_id']
    return argv


def make_spatial_writer(writer, sequences):
    def spatial_write(handle, csv_writer, row):
        sample = sequences[0].last
        if sample is None or sample['sequence_index'] != row['step_index'] or sample['target_snr_db'] != row['target_snr_db']:
            raise RuntimeError('spatial sample/actuator row mismatch')
        if row.get('command_send_monotonic_ns') and sample['pose_status'] == 'ACTIVE':
            if (row['command_send_monotonic_ns']-sample['source_monotonic_ns'])/1e6 > C.POSE_TO_COMMAND_MAX_MS:
                raise RuntimeError('spatial pose-to-command age exceeded existing safeguard')
        writer(handle, csv_writer, dict(row, **{k:sample[k] for k in POSE_FIELDS}))
    return spatial_write


def main(argv=None):
    from rl_agent import ue_target_snr_cell_runtime_v1 as base
    from rl_agent.splitfusion_run5_live_avo_v1 import snr_live
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_target_snr_runtime as legacy
    argv = rewrite_profile(sys.argv[1:] if argv is None else argv)
    audit = argument(argv, '--snr-audit')
    pose_path = audit.parent.parent/C.POSE_SNAPSHOT_RELATIVE
    sequences = []
    prepare, writer, fields = base.prepare_sequence, legacy.write_row, legacy.FIELDS
    base.prepare_sequence = build_prepare(prepare, C.channel_registration(C.repo_root()), pose_path,
                                         audit.parent/'SPATIAL_SAMPLES.jsonl', sequences)
    legacy.FIELDS = (*fields, *POSE_FIELDS)
    legacy.write_row = make_spatial_writer(writer, sequences)
    try:
        return snr_live.controller_main(argv)
    finally:
        base.prepare_sequence, legacy.write_row, legacy.FIELDS = prepare, writer, fields
        for sequence in sequences:
            sequence.close()


if __name__ == '__main__':
    raise SystemExit(main())
