"""Offline, read-only V20 tap replay. Never starts LiveFeed or changes capture timestamps.

The full route is streamed for diagnostics; browser clips are capped at 256 events.
Pose and heading qualification reuse the live viewer's exact binding functions.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .recording import ROOT, bind_pose, calibration, sha_file
from .live_feed_v1 import MAP_SCHEMA, POSE_SCHEMA, join_producer

DEFAULT_CAPTURE = ROOT.parent / ('abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/'
    'splitfusion_single_gpu_profiler_v1/20261012_m3_route_b_v20/good_all_live_view_01')
EVIDENCE = ROOT / 'experiments/splitfusion_m3_route_b_v20/live_view_01'
# Times are relative to the first installed frame's capture stamp, not video start.
CLIPS = [('parked', 'V20 · Parked headings', 145000, 159000),
         ('turn', 'V20 · Moving vehicle / turn', 233000, 259000),
         ('building', 'V20 · Approximate building edge', 191000, 201000),
         # Added for the v20.2 orientation review: densest v20.1 diagonal-parked window, and pedestrians.
         ('parked_avenue', 'V20 · Parked vehicles (avenue)', 306000, 320000),
         ('pedestrians', 'V20 · Pedestrians', 21000, 35000)]


class TapRecording:
    def __init__(self, capture=None):
        self.root = Path(capture or DEFAULT_CAPTURE)
        dirs = list(self.root.glob('capture/cells/*/live_view'))
        if len(dirs) != 1:
            raise ValueError('Select one recorded cell with V20 tap evidence')
        self.map_path, self.pose_path = (dirs[0] / n for n in ('map_installs.jsonl', 'ue_pose.jsonl'))
        self.calib = calibration()
        self.wires, self.owned = {}, {}
        for _, d in self._lines(self.pose_path):
            if d.get('schema') != POSE_SCHEMA:
                raise ValueError('Unexpected pose schema')
            if d['kind'] == 'UE_WIRE':
                key = (d['context']['stream_id'], d['context']['frame_id'])
                if key in self.wires and self.wires[key] != d:
                    raise ValueError('Conflicting pose identity')
                self.wires[key] = d
            elif d['kind'] == 'OWNED_SOURCE':
                self.owned[d['source_sha256']] = d
            if len(self.wires) > 8000 or len(self.owned) > 16000:
                raise ValueError('Pose index bound exceeded')
        self.index, self.base_ns, previous, session = [], None, None, None
        for offset, d in self._lines(self.map_path):
            if d.get('schema') != MAP_SCHEMA or not d.get('record_available'):
                raise ValueError('Missing recorded map')
            i = d['identity']; key = (int(i['frame_id']), int(i['capture_timestamp_ns']))
            s = (i['run_id'], i['cell_id'], i['stream_id'])
            if session is not None and session != s:
                raise ValueError('Mixed session')
            if previous and (key[0] <= previous[0] or key[1] <= previous[1]):
                raise ValueError('Duplicate / out of order map')
            session, previous = s, key
            if self.base_ns is None:
                self.base_ns = key[1]
            self.index.append((offset, key[0], (key[1]-self.base_ns)/1e6))
            if len(self.index) > 8000:
                raise ValueError('Map index bound exceeded')
        self.session = '/'.join(session[:2]); self.source = session[2]

    @staticmethod
    def _lines(path):
        with path.open('rb') as f:
            while True:
                offset = f.tell(); line = f.readline()
                if not line:
                    break
                yield offset, json.loads(line)

    def events(self, start=-float('inf'), end=float('inf')):
        with self.map_path.open('rb') as f:
            for offset, frame, capture_ms in self.index:
                if not start <= capture_ms <= end:
                    continue
                f.seek(offset); d = json.loads(f.readline()); r, i = d['record'], d['identity']
                wire = self.wires.get((self.source, frame))
                owned = self.owned.get(wire['source_sha256']) if wire else None
                binding = dict(run_id=i['run_id'], cell_id=i['cell_id'], stream_id=i['stream_id'],
                    frame_id=frame, capture_timestamp=r['capture_timestamp'], install_timestamp=d['install']['install_timestamp'])
                pose, error = bind_pose(r, wire, owned, self.calib, binding)
                joined, join_error = join_producer(r, d['producer_records'], pose)
                # Actual installation timing retained. Transport/render delay is not reconstructed.
                arrival_ms = (round(d['install']['install_timestamp']*1e9)-self.base_ns)/1e6
                objects = []
                for raw in r['objects']:
                    info = joined.get(raw['id'], {}); h = info.get('heading', {})
                    objects.append(dict(raw=raw, track_id=raw['id'], source=self.source, frame_id=frame,
                        capture_ms=capture_ms, capture_ns=str(i['capture_timestamp_ns']),
                        heading_valid=bool(h.get('valid', False)), heading_yaw_deg=h.get('yaw_deg'),
                        heading_reason=join_error or h.get('reason'), upstream=info or None,
                        visibility=dict(state='unknown', reason='NO_QUALIFIED_VISIBILITY_EVIDENCE')))
                yield dict(session=self.session, source=self.source, frame_id=frame,
                    capture_ms=capture_ms, arrival_ms=arrival_ms, capture_ns=str(i['capture_timestamp_ns']),
                    age_mapping='RECORDED_SAME_HOST_WALL', clock_domain='unix_wall_ns',
                    pose=pose, pose_error=error, heading_join_error=join_error, objects=objects)

    def meta(self):
        return dict(clips=[dict(id=k, name=n) for k,n,_,_ in CLIPS], calibration=self.calib,
                    label='RECORDED M3 V20 · TOWN10')

    def clip(self, name):
        spec = next((c for c in CLIPS if c[0] == name), None)
        if spec is None:
            raise KeyError(name)
        key, title, start, end = spec
        events = list(self.events(start, end))
        if len(events) > 256:
            raise ValueError('Replay clip exceeds 256 events')
        for e in events:
            e['capture_ms'] -= start; e['arrival_ms'] -= start
            for o in e['objects']:
                o['capture_ms'] -= start
        return dict(id=key, name=title, start_ms=0, end_ms=end-start+750, events=events,
                    sources=[self.source], camera_frames=[], errors=dict(Counter(e['pose_error'] for e in events if e['pose_error'])),
                    recording_start_ms=start, base_ns=str(self.base_ns), timing='recorded installation; no browser transport delay')

    def camera(self, frame):
        raise KeyError(frame)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path); p.add_argument('--export', type=Path)
    p.add_argument('--port', type=int, default=8776)
    args = p.parse_args(); recording = TapRecording(args.capture)
    if args.export:
        from .static_geometry_v2 import validated_geometry
        args.export.mkdir(parents=True, exist_ok=True)
        errors = Counter()
        with (args.export/'events.jsonl').open('x') as f:
            for e in recording.events():
                errors.update([e['pose_error'] or 'POSE_OK', e['heading_join_error'] or 'JOIN_OK'])
                f.write(json.dumps(e, separators=(',', ':'), allow_nan=False)+'\n')
        (args.export/'geometry.json').write_text(json.dumps(validated_geometry(recording)))
        (args.export/'INPUTS.json').write_text(json.dumps(dict(events=len(recording.index), bindings=dict(errors),
            base_ns=str(recording.base_ns), files={str(p):sha_file(p) for p in (recording.map_path, recording.pose_path)}), indent=2)+'\n')
        print(dict(events=len(recording.index), bindings=dict(errors)), flush=True)
        return
    from .server import create_app
    from flask import abort, send_file
    app = create_app(recording)
    baseline = Path(__file__).parent/'baseline_v3_1'

    @app.get('/heading-before')
    def heading_before():
        return (baseline/'replay_v3.html').read_text().replace('/assets/', '/heading-before-assets/').replace('RECORDED M3 V10', 'RECORDED M3 V20')

    @app.get('/heading-before-assets/<name>')
    def heading_asset(name):
        if name not in json.loads((baseline/'manifest.json').read_text()):
            abort(404)
        return send_file(baseline/name)

    @app.get('/compare-heading')
    def heading_compare():
        return '<!doctype html><title>V20 heading comparison</title><style>body{margin:0;background:#0a1016;color:white;font:16px sans-serif}header{height:32px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 36px);border:0}</style><header><span>BEFORE · preserved v3.1 (historical; includes hypothesis haze and labels)</span><span>AFTER · current display</span></header><iframe id=before src=/heading-before></iframe><iframe id=after src=/></iframe>'

    orientation_baseline = Path(__file__).parent/'baseline_v20_1'

    @app.get('/orientation-before')
    def orientation_before():
        # Frozen v20.1 (the display reviewed at /compare-heading on 12 Oct), served unchanged.
        return (orientation_baseline/'replay_v3.html').read_text().replace('/assets/', '/orientation-before-assets/').replace('RECORDED M3 V10', 'RECORDED M3 V20 · v20.1')

    @app.get('/orientation-before-assets/<name>')
    def orientation_asset(name):
        if name not in json.loads((orientation_baseline/'manifest.json').read_text()):
            abort(404)
        return send_file(orientation_baseline/name)

    @app.get('/compare-orientation')
    def orientation_compare():
        return '<!doctype html><title>V20 vehicle orientation v20.1 / v20.2</title><style>body{margin:0;background:#0a1016;color:white;font:16px sans-serif}header{height:32px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 36px);border:0}</style><header><span>BEFORE · v20.1 (previous refinement, frozen)</span><span>AFTER · v20.2 display orientation + clean default view</span></header><iframe id=before src=/orientation-before></iframe><iframe id=after src=/></iframe>'

    layout_baseline = Path(__file__).parent/'baseline_v20_2'

    @app.get('/layout-before')
    def layout_before():
        # Frozen v20.2 (accepted display before the v20.3 layout change), served unchanged.
        return (layout_baseline/'replay_v3.html').read_text().replace('/assets/', '/layout-before-assets/').replace('RECORDED M3 V10', 'RECORDED M3 V20 · v20.2')

    @app.get('/layout-before-assets/<name>')
    def layout_asset(name):
        if name not in json.loads((layout_baseline/'manifest.json').read_text()):
            abort(404)
        return send_file(layout_baseline/name)

    @app.get('/compare-layout')
    def layout_compare():
        return '<!doctype html><title>V20 layout v20.2 / v20.3</title><style>body{margin:0;background:#0a1016;color:white;font:16px sans-serif}header{height:32px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 36px);border:0}</style><header><span>BEFORE · v20.2 (accepted display, frozen)</span><span>AFTER · v20.3 map-freshness strip + legend below the map</span></header><iframe id=before src=/layout-before></iframe><iframe id=after src=/></iframe>'

    @app.get('/motion-candidate')
    def motion_candidate():
        # EVALUATION ONLY: the current page with the rejected EMA position-correction candidate loaded over the engine.
        page = (ROOT/'spatial_map_coop/viewer_assets/replay_v3.html').read_text()
        tag = '<script src="/assets/replay_tracks_v3.js"></script>'
        return page.replace(tag, tag+'<script src="/eval-assets/motion_ema_candidate_v1.js"></script>').replace(
            'EGO TRAFFIC VIEW', 'EGO TRAFFIC VIEW · REJECTED EMA CANDIDATE')

    @app.get('/eval-assets/motion_ema_candidate_v1.js')
    def motion_candidate_asset():
        return send_file(Path(__file__).parent/'motion_ema_candidate_v1.js')

    @app.get('/compare-motion')
    def motion_compare():
        return '<!doctype html><title>V20 motion: current / rejected EMA candidate</title><style>body{margin:0;background:#0a1016;color:white;font:16px sans-serif}header{height:32px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 36px);border:0}</style><header><span>CURRENT motion (kept)</span><span>REJECTED · causal EMA of position corrections (evaluation only)</span></header><iframe id=before src=/></iframe><iframe id=after src=/motion-candidate></iframe>'

    print(f'Offline V20 replay: http://127.0.0.1:{args.port}/compare-layout (v20.2 | v20.3), /compare-motion (current | rejected EMA), '
          f'/compare-orientation (v20.1 | current) and /compare-heading (v3.1 | current)', flush=True)
    app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
