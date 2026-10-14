"""Local replay viewer; optionally also the live view of ONE running capture (``--live-root``).

Replay starts no CARLA, inference, experiment or ingest connection.  Live mode only READS the display-only tap files
the running pipeline writes (``live_feed_v1``); it never writes upstream and the pipeline never waits for it.
"""
import argparse
from pathlib import Path
from flask import Flask, Response, jsonify, request, send_file, abort
import io

from .recording import Recording, STATIC, ROOT


def create_app(recording, live=None):
    app = Flask(__name__, static_folder=None)
    assets = ROOT / 'spatial_map_coop/viewer_assets'
    baseline_assets = Path(__file__).parent / 'baseline_v1'
    from .static_geometry_v2 import validated_geometry
    geometry = validated_geometry(recording)

    @app.get('/')
    def index():
        # v3 ego traffic view (tiered display tracks). Previous UIs stay served.
        return send_file(assets / 'replay_v3.html')

    @app.get('/v2')
    def v2():
        return send_file(assets / 'replay_v1.html')

    @app.get('/before')
    def before():
        return (baseline_assets / 'replay_v1.html').read_text().replace('/assets/', '/before-assets/').replace('/api/replay/', '/api/replay/')

    @app.get('/before-assets/<name>')
    def before_asset(name):
        if name not in {'presentation_predictor_v1.js','presentation_inventory_v1.js','replay_policy_v1.js','replay_app_v1.js','replay_v1.css'}: abort(404)
        if name == 'replay_app_v1.js':
            return (baseline_assets/name).read_text().replace("'/api/replay/clip/'+", "'/api/replay/before-clip/'+"), 200, {'Content-Type':'application/javascript'}
        return send_file(baseline_assets/name)

    @app.get('/assets/<name>')
    def asset(name):
        allowed = {'presentation_predictor_v1.js', 'presentation_inventory_v1.js',
                   'replay_policy_v1.js', 'replay_app_v1.js', 'replay_v1.css',
                   'replay_motion_v2.js', 'replay_geometry_v2.js',
                   'replay_tracks_v3.js', 'replay_app_v3.js', 'replay_v3.css',
                   'live_app_v1.js', 'live_v1.css', 'replay_qos_v1.js'}
        if name not in allowed:
            abort(404)
        return send_file(assets / name)

    @app.get('/api/replay/meta')
    def meta():
        return jsonify(recording.meta())

    @app.get('/api/replay/clip/<name>')
    def clip(name):
        try:
            return jsonify(recording.clip(name))
        except KeyError:
            abort(404)

    @app.get('/api/replay/before-clip/<name>')
    def before_clip(name):
        # Same raw clip, disable new joined heading qualification for baseline.
        import copy
        try: data = copy.deepcopy(recording.clip(name))
        except KeyError: abort(404)
        for event in data['events']:
            for obj in event['objects']:
                obj['heading_valid'] = False
                obj.pop('upstream', None)
        return jsonify(data)

    @app.get('/compare')
    def compare():
        return '<!doctype html><title>Replay before / after</title><style>body{margin:0;background:#0c1620;color:white;font:18px sans-serif}header{height:36px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 40px);border:0}</style><header><span>BEFORE · preserved replay v1</span><span>AFTER · heading / buildings / motion</span></header><iframe id=before src=/before></iframe><iframe id=after src=/v2></iframe>'

    @app.get('/compare-ped')
    def compare_ped():
        return '<!doctype html><title>Pedestrian motion v3.0 / v3.1</title><style>body{margin:0;background:#0a1016;color:white;font:16px sans-serif}header{height:36px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 40px);border:0}</style><header><span>BEFORE · v3.0 pedestrian motion (speed &gt; 1 m/s and 1.5 m)</span><span>AFTER · v3.1 uncertainty-aware pedestrian motion</span></header><iframe id=before src="/?ped=v3.0"></iframe><iframe id=after src=/></iframe>'

    @app.get('/compare-v3')
    def compare_v3():
        return '<!doctype html><title>Replay v2 / v3</title><style>body{margin:0;background:#0a1016;color:white;font:16px sans-serif}header{height:36px;display:flex;align-items:center}header span{width:50%;text-align:center}iframe{width:50%;height:calc(100vh - 40px);border:0}</style><header><span>BEFORE · v2 (every candidate drawn)</span><span>AFTER · v3 ego traffic view (tiered display tracks)</span></header><iframe id=before src=/v2></iframe><iframe id=after src=/></iframe>'

    @app.get('/api/replay/static')
    def static():
        return jsonify(geometry)

    @app.get('/api/replay/camera/<int:frame>.png')
    def camera(frame):
        try:
            return send_file(io.BytesIO(recording.camera(frame)), mimetype='image/png')
        except KeyError:
            abort(404)

    @app.get('/live')
    def live_page():
        if live is None:
            abort(404)
        return send_file(assets / 'live_v1.html')

    @app.get('/api/live/meta')
    def live_meta():
        if live is None:
            abort(404)
        return jsonify(dict(epoch=live.epoch, label=live.label, calibration=recording.calib,
                            status=live.status(), age_mapping='LIVE_SAME_HOST_WALL',
                            clock='capture and installation stamps: host unix wall ns; browser maps its clock to '
                                  'the server clock from every message (minimum-delay estimate)'))

    @app.get('/api/live/status')
    def live_status():
        if live is None:
            abort(404)
        return jsonify(live.status())

    @app.get('/api/live/stream')
    def live_stream():
        if live is None:
            abort(404)
        client = live.connect(request.remote_addr)
        if client is None:
            abort(503)
        return Response(live.stream(client), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    @app.post('/api/live/report')
    def live_report():
        if live is None or not live.client_report(request.get_json(silent=True)):
            abort(400)
        return ('', 204)

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, help='sealed M3 prospective directory (default: registered V10)')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--open', action='store_true', help='open the local replay in your default browser')
    p.add_argument('--live-root', type=Path, help='prospective root (or cell directory) of ONE running capture; '
                                                   'serves the live view at /live')
    p.add_argument('--live-log', type=Path, help='create-only compact viewer event log (JSONL)')
    args = p.parse_args()
    recording = Recording(args.capture)
    # Validate the initial replay before serving; no writes to capture files.
    first = recording.clip('departure')
    print(f'Replay ready: {len(first["events"])} initial events; pose limitations: {first["errors"]}', flush=True)
    live = None
    if args.live_root is not None:
        from .live_feed_v1 import LiveFeed
        live = LiveFeed(args.live_root, log_path=args.live_log).start()
        print(f'Live view: http://127.0.0.1:{args.port}/live  (reading {args.live_root})', flush=True)
    print(f'Open http://127.0.0.1:{args.port} — replay', flush=True)
    if args.open:
        import threading, webbrowser
        threading.Timer(.75, lambda: webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    try:
        create_app(recording, live).run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)
    finally:
        if live is not None:
            live.close()


if __name__ == '__main__':
    main()
