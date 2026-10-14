"""OFFLINE evaluation input only: per-frame GT positions for the replay clips.

Never served to or imported by the UI. Used solely to score what the display
shows (coverage / unsupported glyphs / stationary wobble), before and after.
Reads the registered capture's archived exact-GT targets read-only:
  numeric_gt/service/archive/frame_<id>/scorer/target_metadata.json
plus the capture's static-environment catalogue (parked cars/trucks/buses).
"""
import argparse
import csv
import json
from pathlib import Path

from .recording import Recording

VEHICLE_STATIC = {'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle'}


def extract(recording, clip_names=('departure', 'turn', 'stop', 'avenue', 'ped_pause', 'ped_standing')):
    cell = recording.cell
    archive = cell / 'numeric_gt/service/archive'
    static = []
    with (cell / 'parity_static_catalogue/static_environment_objects.csv').open() as f:
        for row in csv.DictReader(f):
            if row['semantic_class'] in VEHICLE_STATIC and row['enabled'] == 'true':
                static.append([float(row['bbox_center_x_m']), float(row['bbox_center_y_m'])])
    out = dict(schema='replay_gt_audit_input.v3', use='OFFLINE_DISPLAY_EVALUATION_ONLY',
               source=str(archive), static_vehicles=static, clips={})
    for name in clip_names:
        frames = {}
        for event in recording.clip(name)['events']:
            path = archive / f'frame_{event["frame_id"]}' / 'scorer/target_metadata.json'
            if not path.exists():
                continue
            t = json.loads(path.read_text())
            positives = [dict(id=b['gt_actor_id'], label=b['label'], x=float(b['object_world_x']), y=float(b['object_world_y']),
                              speed=float(b['object_speed_mps'] or 0), yaw=float(b['object_yaw_deg'] or 0),
                              distance=float(b['gt_distance_m']))
                         for b in t['object_boxes'] if b.get('contract_state') == 'POSITIVE']
            actors = [[a['world_x'], a['world_y']] for a in t['avo']['qualified'] + t['avo']['structural_ignored']]
            frames[event['frame_id']] = dict(capture_ms=event['capture_ms'], arrival_ms=event['arrival_ms'],
                                             positives=positives, actors=actors)
        out['clips'][name] = frames
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = extract(Recording())
    args.out.write_text(json.dumps(data))
    print({k: len(v) for k, v in data['clips'].items()}, len(data['static_vehicles']))


if __name__ == '__main__':
    main()
