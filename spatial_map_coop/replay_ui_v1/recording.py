"""Bounded, read-only M3 installed-event replay and exact pose-sidecar adapter.

The frozen production wire protocol is not changed. These are already-installed
map records, NOT production wire messages. Original object dictionaries remain
intact under `raw`; display association uses a separate ConservativeTrackStore.
"""
from __future__ import annotations

import ast
import csv
from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from spatial_map_coop.coop_input_v1.pose_sidecar_v1 import parse_pose_sidecar
from spatial_map_coop.multi_ue_v1.association import (
    AssociationPolicy, ConservativeTrackStore, associate_snapshot,
)
from spatial_map_coop.multi_ue_v1.buffer import MultiUESnapshot, SourceFrame
from spatial_map_coop.multi_ue_v1.contracts import ObservationBatch, _normalise_record

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / 'experiments/splitfusion_m3_route_b_v10/preregistration_01/PREREGISTRATION.json'
STATIC = ROOT / 'spatial_map_coop/recordings/two_ego_occl.jsonl.static.json'
MAX_EVENTS = 256
MAX_INDEX = 8000
AUDITED_RUN = 'splitfusion_run4_phase6_v2/20261009T031008Z'
AUDITED_CELL = 'run4p6_20261009T031008Z_a71__favorable_stable'
POLICY = AssociationPolicy(750_000_000, 100_000_000, 100_000_000, 3., .65, 3., 750_000_000,
                           minimum_confidence=0., confidence_provenance='NO_ADDITIONAL_SCORE_GATE')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def calibration():
    # Read the frozen numeric registry without importing torch / the live runtime.
    source = ROOT / 'rl_agent/splitfusion_live_dispatch_v1/frame_context.py'
    values = {}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    'CAMERA_MODEL_NUMERIC_FIELDS', 'CAMERA_MOUNT_NUMERIC_FIELDS',
                    'STATIC_CAMERA_MODEL_SHA256', 'STATIC_CAMERA_MOUNT_SHA256'}:
                    values[target.id] = ast.literal_eval(node.value)
    from decimal import Decimal
    for kind in ('MODEL', 'MOUNT'):
        fields = values[f'CAMERA_{kind}_NUMERIC_FIELDS']
        doc = {'schema': f'scenesense.static_camera_{kind.lower()}.v1',
               'numeric_fields': [[k, format(Decimal(v).normalize(), 'f')] for k, v in fields]}
        if digest(doc) != values[f'STATIC_CAMERA_{kind}_SHA256']:
            raise ValueError('camera numeric registry hash mismatch')
    intr = dict((k, float(v)) for k, v in values['CAMERA_MODEL_NUMERIC_FIELDS'])
    sensor_source = ROOT / 'carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py'
    ranges = []
    for node in ast.walk(ast.parse(sensor_source.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == 'add_argument' and node.args and
                isinstance(node.args[0], ast.Constant) and node.args[0].value == '--radar-range'):
            ranges.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg == 'default')
    if ranges != [120.0]:
        raise ValueError('audited radar range configuration changed; no range fallback')
    return {
        'model_sha256': values['STATIC_CAMERA_MODEL_SHA256'],
        'mount_sha256': values['STATIC_CAMERA_MOUNT_SHA256'],
        'fov_deg': math.degrees(2 * math.atan(intr['camera_width'] / (2 * intr['camera_fx']))),
        'intrinsic': intr, 'registry': str(source.relative_to(ROOT)),
        # This is the registered radar acquisition / preprocessing envelope,
        # not a claim that every camera object within 120 m is observable.
        'range_m': ranges[0], 'range_kind': 'configured radar envelope; not a visibility guarantee',
        'range_source': 'rl_agent/ue_route_b_split_cell_adapter_v1.py:1568; '
                        'carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py:478,863',
    }


def bind_pose(record, wire, owned, calib, install_binding=None):
    try:
        return _bind_pose_checked(record, wire, owned, calib, install_binding)
    except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None, 'INVALID_POSE_SIDECAR'


def _bind_pose_checked(record, wire, owned, calib, install_binding):
    """Validate full run/cell/source/frame/capture identity and sensor hash binding.

    UE_WIRE context supplies the capture pose. OWNED_SOURCE is retained sensor
    synchronization evidence, never a new actor query. Float seconds in the
    installed API must be the exact original conversion of wire nanoseconds.
    """
    if not wire or not owned:
        return None, 'MISSING_FRAME_POSE'
    c, identity, source = wire['context'], wire['gt_identity'], owned['source']
    if not install_binding:
        return None, 'MISSING_INSTALL_IDENTITY'
    for field in ('run_id', 'cell_id', 'stream_id'):
        if identity[field] != install_binding[field]:
            return None, 'INSTALL_IDENTITY_MISMATCH'
    if (int(install_binding['frame_id']) != record['frame_id'] or
            float(install_binding['capture_timestamp']) != record['capture_timestamp'] or
            float(install_binding['install_timestamp']) != record['install_timestamp']):
        return None, 'INSTALL_TIMESTAMP_MISMATCH'
    for key in ('stream_id', 'frame_id'):
        if not (record[key] == c[key] == identity[key] == source[key]):
            return None, 'POSE_IDENTITY_MISMATCH'
    ns = c['capture_timestamp_ns']
    if identity['capture_timestamp_ns'] != ns or record['capture_timestamp'] != ns / 1e9:
        return None, 'POSE_TIMESTAMP_MISMATCH'
    if wire['source_sha256'] != owned['source_sha256'] or digest(source) != owned['source_sha256']:
        return None, 'POSE_SOURCE_HASH_MISMATCH'
    if c['ego_world'] != source['ego_world']:
        return None, 'POSE_CONTEXT_MISMATCH'
    for field, expected in [('camera_model_sha256', calib['model_sha256']),
                            ('camera_mount_sha256', calib['mount_sha256'])]:
        if c[field] != expected or source[field] != expected:
            return None, 'UNREGISTERED_CALIBRATION'
    sidecar = dict(schema='scenesense.coop_pose_sidecar.v1', sidecar_version=1,
                   run_id=identity['run_id'], cell_id=identity['cell_id'],
                   stream_id=c['stream_id'], frame_id=c['frame_id'], capture_timestamp_ns=ns,
                   ego_world=dict(zip(('x', 'y', 'z', 'pitch', 'yaw', 'roll'), c['ego_world'])),
                   horizontal_fov_deg=calib['fov_deg'],
                   pose_provenance='SFD1_FRAME_CONTEXT_V1_EGO_WORLD')
    validated = parse_pose_sidecar(sidecar)
    matrix = np.asarray(source['camera_matrix'], dtype=float)
    ego_matrix = np.asarray(source['ego_matrix'], dtype=float)
    if (matrix.shape != (4, 4) or ego_matrix.shape != (4, 4) or
            not np.isfinite(matrix).all() or not np.isfinite(ego_matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1]) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)):
        return None, 'INVALID_SENSOR_MATRIX'
    # Verify mount and ego translation as well as hashes; refuse corrupted pose data.
    relative = np.linalg.inv(ego_matrix) @ matrix
    # Same degree-based CARLA rotation convention as the frozen frame-context
    # module, without importing its torch/live-dispatch dependency.
    pitch, yaw, roll = map(math.radians, c['ego_world'][3:])
    cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
    expected_rotation = np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                                  [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                                  [sp, -cp*sr, cp*cr]])
    if not np.allclose(ego_matrix[:3, :3], expected_rotation, atol=1e-5, rtol=0):
        return None, 'EGO_ROTATION_MISMATCH'
    a = math.radians(-4.)
    expected_mount = np.array([[math.cos(a), 0, -math.sin(a), 1.8],
                              [0, 1, 0, 0], [math.sin(a), 0, math.cos(a), 1.55],
                              [0, 0, 0, 1]])
    if (not np.allclose(relative, expected_mount, atol=3e-5) or
            not np.allclose(ego_matrix[:3, 3], c['ego_world'][:3], atol=1e-5)):
        return None, 'SENSOR_MOUNT_MISMATCH'
    return {
        'x': validated.pose.x, 'y': validated.pose.y, 'z': validated.pose.z,
        'yaw': validated.pose.yaw, 'frame_id': c['frame_id'], 'source': c['stream_id'],
        'capture_ns': str(ns), 'binding': list(validated.key[:-1]) + [str(ns)],
        'sensor': {'x': float(matrix[0, 3]), 'y': float(matrix[1, 3]),
                   'yaw': math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))},
        'camera_matrix': matrix.tolist(), 'fov_deg': calib['fov_deg'],
        'range_m': calib['range_m'], 'sidecar': sidecar,
    }, None


class Recording:
    def __init__(self, capture=None):
        self.root = Path(capture) if capture else Path(json.loads(REGISTRATION.read_text())['prospective'])
        cells = list((self.root / 'capture/cells').glob('*'))
        if len(cells) != 1:
            raise ValueError('select a sealed M3 capture with exactly one cell')
        self.cell = cells[0]
        self.path = self.cell / 'm3_map/installed_map_records.jsonl'
        self.calib = calibration()
        self.install_bindings = {}
        with (self.cell / 'direct_edge_map/direct_map_ingest.csv').open() as f:
            for row in csv.DictReader(f):
                if row['outcome'] == 'RESULT_INSTALLED':
                    if (row['run_id'], row['cell_id']) != (AUDITED_RUN, AUDITED_CELL):
                        raise ValueError('No validated clock/calibration contract for this capture; this adapter is scoped to M3 V10')
                    key = (row['stream_id'], int(row['frame_id']))
                    if key in self.install_bindings:
                        raise ValueError('conflicting installed frame identity')
                    self.install_bindings[key] = row
                    if len(self.install_bindings) > MAX_INDEX:
                        raise ValueError('recording exceeds bounded install index')
        self.wires, self.owned = {}, {}
        with (self.cell / 'pose_archive/pose_evidence.jsonl').open() as pose_file:
            for line in pose_file:
                d = json.loads(line)
                if d['kind'] == 'UE_WIRE':
                    c = d['context']; key = (c['stream_id'], c['frame_id'])
                    if key in self.wires and self.wires[key] != d:
                        raise ValueError('conflicting wire pose')
                    self.wires[key] = d
                elif d['kind'] == 'OWNED_SOURCE':
                    self.owned[d['source_sha256']] = d
                if len(self.wires) > MAX_INDEX or len(self.owned) > MAX_INDEX * 2:
                    raise ValueError('recording exceeds bounded pose index')
        self.index = []
        with self.path.open('rb') as f:
            while True:
                offset = f.tell(); line = f.readline()
                if not line:
                    break
                d = json.loads(line)
                if d['outcome'] != 'OK':
                    continue
                r = d['map_document']['record']
                self.index.append((r['install_timestamp'], offset, r['frame_id']))
                if len(self.index) > MAX_INDEX:
                    raise ValueError('recording exceeds bounded event index')
        if not self.index:
            raise ValueError('no installed events')
        self.start = min(x[0] for x in self.index)
        self.index.sort()  # deterministic recorded installation order
        self.retained = {}
        manifest = self.root / 'paired_retention_v2/MANIFEST.json'
        if manifest.exists():
            for r in json.loads(manifest.read_text())['records']:
                if r['status'] == 'RETAINED_COMPLETE':
                    self.retained[r['frame_context']['frame_id']] = r
        self.clips = [dict(id='departure', name='01 · Departure', start_s=0., duration_s=18.),
                      dict(id='turn', name='02 · Junction & turn', start_s=30., duration_s=18.),
                      dict(id='stop', name='03 · Stop & nearby traffic', start_s=76., duration_s=18.),
                      dict(id='avenue', name='04 · Avenue & turn', start_s=180., duration_s=18.),
                      # Pedestrian-motion review segments (frames ~2810-3100 and ~3220-3510).
                      dict(id='ped_pause', name='05 · Pedestrians: slow down, pause, resume', start_s=84., duration_s=18.),
                      dict(id='ped_standing', name='06 · Pedestrians standing, ego approaching', start_s=110., duration_s=18.)]

    def meta(self):
        return dict(title='Town10 / recorded M3 V10', clips=self.clips,
                    calibration=self.calib, event_count=len(self.index), max_events=MAX_EVENTS,
                    clock='unix_wall_ns (same recorded host); browser mapped to replay elapsed time',
                    visibility='Unknown: no qualified object visibility evidence retained in this adapter',
                    capture=str(self.root), source_policy='selected source only; no added confidence gate')

    @lru_cache(maxsize=3)
    def clip(self, name):
        spec = next((c for c in self.clips if c['id'] == name), None)
        if spec is None:
            raise KeyError(name)
        base = self.start + spec['start_s']
        selected = [row for row in self.index if base - 1 <= row[0] <= base + spec['duration_s']]
        if len(selected) > MAX_EVENTS:
            raise ValueError('clip exceeds bounded event window')
        stores, events, errors = {}, [], Counter()
        with self.path.open('rb') as f:
            for installed, offset, _ in selected:
                f.seek(offset); envelope = json.loads(f.readline())
                r = envelope['map_document']['record']
                key = (r['stream_id'], r['frame_id'])
                wire = self.wires.get(key)
                owned = self.owned.get(wire['source_sha256']) if wire else None
                pose, error = bind_pose(r, wire, owned, self.calib, self.install_bindings.get(key))
                if error:
                    errors[error] += 1
                capture_ns = wire['context']['capture_timestamp_ns'] if pose else round(r['capture_timestamp'] * 1e9)
                # Subtract integer nanoseconds before JS conversion; absolute ns travel as strings.
                base_ns = round(base * 1e9)
                capture_ms = (capture_ns - base_ns) / 1e6
                arrival_ms = (round(installed * 1e9) - base_ns) / 1e6
                session = wire['gt_identity']['run_id'] + '/' + wire['gt_identity']['cell_id'] if wire else str(self.root)
                source = r['stream_id']
                batch = ObservationBatch(source, source, session, 'unix_wall_ns', r['frame_id'],
                                         capture_ns, 'unix_wall_ns', round(installed * 1e9), None,
                                         r['schema'], digest(r),
                                         tuple(_normalise_record(o, i) for i, o in enumerate(r['objects'])))
                snapshot = MultiUESnapshot('unix_wall_ns', capture_ns, (SourceFrame(batch, 0),),
                                           0, POLICY.alignment_tolerance_ns, True)
                associations = associate_snapshot(snapshot, POLICY)
                store = stores.setdefault((session, source), ConservativeTrackStore(POLICY))
                tracks = store.update(associations)
                current = {t.selected_observation_id: t for t in tracks
                           if t.selected_frame_id == r['frame_id']}
                from .upstream_v2 import join_predictions
                joined, join_error = join_predictions(self, r, pose)
                objects = []
                for raw in r['objects']:
                    track = current.get(raw['id'])
                    objects.append(dict(raw=raw, track_id=track.track_id if track else raw['id'],
                                        source=raw['source_stream_id'], frame_id=raw['frame_id'],
                                        capture_ms=capture_ms, capture_ns=str(capture_ns),
                                        heading_valid=joined.get(raw['id'], {}).get('heading', {}).get('valid', False),
                                        heading_yaw_deg=joined.get(raw['id'], {}).get('heading', {}).get('yaw_deg'),
                                        heading_reason=join_error or joined[raw['id']]['heading']['reason'],
                                        upstream=joined.get(raw['id']),
                                        visibility=dict(state='unknown', reason='NO_QUALIFIED_VISIBILITY_EVIDENCE')))
                events.append(dict(session=session, source=source, frame_id=r['frame_id'],
                                   capture_ms=capture_ms, arrival_ms=arrival_ms, capture_ns=str(capture_ns),
                                   clock_domain='unix_wall_ns', age_mapping='RECORDED_SAME_HOST_WALL',
                                   pose=pose, pose_error=error, objects=objects,
                                   raw_sha256=envelope['raw_sha256'], camera_available=r['frame_id'] in self.retained))
        return dict(**spec, events=events, errors=dict(errors), sources=sorted({e['source'] for e in events}),
                    camera_frames=[dict(frame_id=e['frame_id'], time_ms=e['arrival_ms'])
                                   for e in events if e['camera_available']],
                    # Tail allows interruption/expiry inspection without inventing an update.
                    end_ms=spec['duration_s'] * 1000 + 1500)

    @lru_cache(maxsize=3)
    def camera(self, frame):
        retained = self.retained[frame]
        c = retained['frame_context']
        wire = self.wires[(c['stream_id'], frame)]
        if c != wire['context']:
            raise ValueError('camera/context mismatch')
        path = self.root / 'paired_retention_v2' / retained['input_file']
        if path.parent != self.root / 'paired_retention_v2' or sha_file(path) != retained['input_sha256']:
            raise ValueError('camera tensor hash mismatch')
        a = np.load(path, mmap_mode='r', allow_pickle=False)
        if a.shape != (1, 7, 448, 768) or a.dtype != np.float32 or np.any(a[:, :, 432:, :]):
            raise ValueError('unexpected retained RGB layout / padding')
        rgb = a[0, :3, :432].transpose(1, 2, 0)
        rgb = np.rint(np.clip(rgb * [.229, .224, .225] + [.485, .456, .406], 0, 1) * 255).astype('uint8')
        output = io.BytesIO(); Image.fromarray(rgb).save(output, format='PNG')
        return output.getvalue()
