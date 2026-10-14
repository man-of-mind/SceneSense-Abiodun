"""Actual retained UDP map ingestion with source-local endpoints and journal.

No Flask, renderer or cross-source association. The unchanged retained ingest
checks the 500-ms horizon and ordering; 170 ms is never an admission test.
"""
import copy
import threading
import time
import types
import json
from collections import OrderedDict

from .binding import bind
from .identity import require
from .map_adapter import Run4MapJournal
from rl_agent.splitfusion_direct_edge_map_v1 import map_ingest as MI, protocol as DP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_m3_route_b_v3.map_class import convert


class ObservationStore:
    """Per-stream observations, no association or unique-object interpretation."""
    def __init__(self, *, history_size):
        require(history_size > 0, 'positive retained map history capacity required')
        self.limit = history_size
        self.latest, self.history = {}, {}
        self.lock = threading.Lock()
        from rl_agent.splitfusion_direct_edge_map_v1 import spatial_map_direct_server_v1 as server
        # Keep the actual retained normalizer and installation bytecode. Each
        # source gets its own retained history capacity, not a shared eviction
        # budget that lets one UE remove the other's observations.
        self._server = server
        self._installers = {}
        config = types.SimpleNamespace(default_action_id='',min_object_score=0.,object_yaw_map_offset_deg=90.)
        yaw = bind(server.baseline._object_map_yaw_deg,_config=lambda:config)
        obj = bind(server.baseline._normalize_object,_config=lambda:config,_object_map_yaw_deg=yaw)
        self._normalize = bind(server.baseline._normalize_packet,_config=lambda:config,_normalize_object=obj)

    def install(self, document, ingest_at):
        value = convert(document)  # copy-only class conversion, originals in journal
        stream = value['stream_id']
        if stream not in self._installers:
            history = self.history.setdefault(stream, OrderedDict())
            baseline = types.SimpleNamespace(_normalize_packet=self._normalize,
                state_lock=self.lock, latest_streams=self.latest, installed_frame_history=history,
                _config=lambda: types.SimpleNamespace(installed_frame_history_size=self.limit))
            self._installers[stream] = bind(self._server.install_under_state_lock, baseline=baseline)
        return self._installers[stream](MP.run4_install_document(value), ingest_at)


def service_class(contract):
    proxy = MP.MapProtocolProxyV2(DP, contract=contract)
    class SourceMapService(MI.DirectMapIngestService):
        # These functions are the retained bytecode, with a private Run4
        # protocol binding. Never mutate MI.protocol or another map service.
        ingest = bind(MI.DirectMapIngestService.ingest, protocol=proxy)
        _respond_retained = bind(MI.DirectMapIngestService._respond, protocol=proxy)
        _handle_complete = bind(MI.DirectMapIngestService._handle_complete, protocol=proxy)

        def __init__(self, *, source, journal, pose_lookup, **kwargs):
            self.source, self.journal, self.pose_lookup = source, journal, pose_lookup
            super().__init__(feedback_host=source.ip, feedback_port=source.feedback_port,
                expected_run_id=source.run_id, expected_cell_id=source.cell_id, allowed_action_ids=(), **kwargs)

        def _validate_binding(self, document):
            super()._validate_binding(document)
            require(document['stream_id'] == self.source.stream_id, 'wrong-source map endpoint')
            self.source.capture(document)

        def _respond(self, document, **kwargs):
            # Retain actual ACK send timestamps, send-failure counters and
            # terminal accounting. Journal the same terminal without restamping.
            result = self._respond_retained(document, **kwargs)
            if result.get('feedback') and result['feedback']['terminal']:
                pose = self.pose_lookup(self.source, document['frame_id'])
                self.journal.record(self.source, document, arrival_ns=int(kwargs['map_ingest_at'] * 1e9),
                                    feedback=result['feedback'], pose=pose)
            return result
    return SourceMapService


class HeadlessMaps:
    def __init__(self, *, sources, contract, output, bind_host, ports, pose_lookup,
                 history_size, service_factory=None, cpus=''):
        self.journal = Run4MapJournal(sources, contract, output / 'observations.jsonl')
        self.store = ObservationStore(history_size=history_size)
        self.services = {}
        cls = service_factory or service_class(contract)
        try:
            for source in sources.sources:
                directory = output / source.stream_id
                directory.mkdir()
                self.services[source.stream_id] = cls(source=source, journal=self.journal,
                    pose_lookup=pose_lookup, bind_host=bind_host, bind_port=ports[source.stream_id],
                    install=self.store.install, ingest_csv=directory / 'ingest.csv',
                    processing_horizon_s=.5, ingest_cpus=cpus, receive_cpus=cpus)
        except BaseException:
            self.close()
            raise

    def start(self):
        for service in self.services.values():
            service.start()

    def close(self):
        results = {}
        try:
            for stream, service in self.services.items():
                try:
                    results[stream] = service.close()
                except BaseException as error:
                    results[stream] = dict(error=repr(error))
        finally:
            self.journal.close()
        return results


class ExactRecordAPI:
    """Only the retained collector's exact-record read API; no UI or renderer."""
    def __init__(self, store, port):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import unquote
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parts = self.path.split('/')
                if len(parts) != 6 or parts[1:4] != ['api','fusion_streams','installed']:
                    self.send_error(404); return
                try:
                    stream, frame = unquote(parts[4]), int(parts[5])
                except ValueError:
                    self.send_error(400); return
                with store.lock:
                    record = store.history.get(stream, {}).get((stream, frame))
                    body = None if record is None else json.dumps(dict(status='INSTALLED', record=record)).encode()
                if body is None:
                    self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *args): pass
        self.server = ThreadingHTTPServer(('127.0.0.1',port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name='m4-exact-map-api')
    def start(self): self.thread.start()
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=5)
