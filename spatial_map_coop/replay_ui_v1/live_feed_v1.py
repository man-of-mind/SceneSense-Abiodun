"""Live adapter for the v3.1 viewer: tails the display-only live-view tap files of ONE running capture.

Inputs (written by ``rl_agent.splitfusion_live_view_tap_v1.tap`` inside the pipeline, off the decision path):
  <cell>/live_view/map_installs.jsonl  one line per SUCCESSFUL map installation: the installed (normalised) record,
                                       the edge document identity and the producer records (p025 world-yaw pair)
  <cell>/live_view/ue_pose.jsonl       the UE pose-evidence documents (OWNED_SOURCE, UE_WIRE) for each frame

Each installed record is bound to its frame-bound pose sidecar with the replay's own ``recording.bind_pose`` (full
run/cell/stream/frame/capture-ns identity, sensor hash, calibration hashes, ego rotation, mount); heading comes only
from the producer world-yaw pair through ``heading_ingest_v2.heading`` after an exact per-record join (original index,
XYZ, dimensions, score, class).  The legacy installed ``yaw_deg`` (always 90) is never used.  GT is never read.

The viewer is a separate process: the pipeline never waits for it.  Every buffer here is bounded; a slow or absent
browser only drops its own queued messages (counted).  Events are emitted in installation order; duplicates and
out-of-order frames are refused before they reach any client, and clients refuse them again.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import json
from pathlib import Path
import threading
import time
import uuid

from .heading_ingest_v2 import heading
from .recording import bind_pose, calibration, digest

MAP_SCHEMA = "scenesense.live_view.map_install.v1"
POSE_SCHEMA = "scenesense.live_view.ue_pose.v1"
LIVE_DIR = "live_view"
MAP_FILE = "map_installs.jsonl"
POSE_FILE = "ue_pose.jsonl"
POLL_S = 0.010
HEARTBEAT_S = 0.25
POSE_WAIT_S = 0.40               # an install whose pose has not arrived yet waits at most this long, then is emitted
MAX_PENDING = 64                 #   with an explicit pose error (never bound to a nearby/inferred pose)
MAX_WIRES = 4096
MAX_OWNED = 8192
CLIENT_QUEUE = 32
MAX_CLIENTS = 8
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_INSTALL_AGE_NS = 2_000_000_000   # an installation read later than this is history, not live
AGE_MAPPING = "LIVE_SAME_HOST_WALL"
CLASS = {"vehicle": "Vehicle", "person": "Pedestrian"}


def now_ns():
    return time.time_ns()


class Tail:
    """Incremental reader of one append-only JSONL file; only complete lines are returned.

    A file that already existed before this viewer started (viewer restarted mid-run) is NOT read from its start:
    ``backlog_bytes`` = 0 opens at its end; > 0 keeps only that many trailing bytes (pose index warm-up)."""

    def __init__(self, path, *, started_ns, backlog_bytes=0):
        self.path, self.handle, self.partial, self.lines, self.errors = Path(path), None, b"", 0, 0
        self.started_ns, self.backlog_bytes, self.skipped_bytes = started_ns, backlog_bytes, 0

    def read(self, limit=256):
        if self.handle is None:
            if not self.path.is_file():
                return []
            self.handle = self.path.open("rb")
            stat = self.path.stat()
            if stat.st_mtime_ns < self.started_ns and stat.st_size:
                start = max(0, stat.st_size - self.backlog_bytes)
                self.handle.seek(start)
                self.skipped_bytes = start
                if start:
                    self.handle.readline()          # drop the partial line at the cut
        out = []
        while len(out) < limit:
            chunk = self.handle.readline()
            if not chunk:
                break
            self.partial += chunk
            if not self.partial.endswith(b"\n"):
                if len(self.partial) > MAX_LINE_BYTES:
                    self.partial, self.errors = b"", self.errors + 1
                continue
            raw, self.partial = self.partial, b""
            try:
                out.append(json.loads(raw))
                self.lines += 1
            except ValueError:
                self.errors += 1
        return out


def join_producer(record, producer, pose):
    """Exact producer-record join for each installed object (heading only).  Returns ({object id: info}, error)."""
    if not pose:
        return {}, "NO_BOUND_POSE"
    if not producer:
        return {}, "NO_PRODUCER_RECORDS"
    prefix = f"{record['stream_id']}:{record['frame_id']}:"
    result = {}
    for raw in record["objects"]:
        oid = str(raw.get("id", ""))
        if not oid.startswith(prefix) or not oid[len(prefix):].isdigit():
            return {}, "UNEXPECTED_INSTALLED_OBJECT_ID"
        index = int(oid[len(prefix):])
        if index >= len(producer):
            return {}, "PRODUCER_INDEX_OUT_OF_RANGE"
        p = producer[index]
        if (p.get("stream_id", record["stream_id"]) != record["stream_id"]
                or p.get("frame_id", record["frame_id"]) != record["frame_id"]
                or str(p.get("capture_timestamp_ns", pose["capture_ns"])) != pose["capture_ns"]
                or p.get("score") != raw["score"] or CLASS.get(p.get("class_name")) != raw["type"]
                or any(p.get("world_" + a) != raw["location"][a] for a in "xyz")
                or any(max(0.05, p.get("size_" + a, 0.0)) != raw["dimensions"][b]
                       for a, b in zip("xyz", ("length", "width", "height")))):
            return {}, "PRODUCER_RECORD_MISMATCH"
        result[oid] = dict(heading=heading(p), candidate={k: p.get(k) for k in (
            "candidate_identity", "prediction_index", "fpn_level", "point_index", "internal_class",
            "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "local_x", "local_y", "local_z")},
            provenance=dict(source="LIVE_MAP_INSTALL_TAP", record_sha256=digest(p), binding=pose["binding"],
                            producer_index=index, join="EXACT_INDEX_XYZ_DIMENSIONS_SCORE_CLASS"))
    return result, None


class Client:
    def __init__(self, ident, address):
        self.id, self.address = ident, address
        self.queue, self.cond = deque(maxlen=CLIENT_QUEUE), threading.Condition()
        self.dropped, self.sent, self.connected_ns, self.closed = 0, 0, now_ns(), False

    def push(self, message):
        with self.cond:
            if len(self.queue) == self.queue.maxlen:
                self.dropped += 1            # oldest dropped by the bounded deque
            self.queue.append(message)
            self.cond.notify()

    def pop(self, timeout):
        with self.cond:
            if not self.queue:
                self.cond.wait(timeout)
            return self.queue.popleft() if self.queue else None


class EventLog:
    """Compact viewer-side timeline (one JSON line per emitted frame / connection / client report / flag)."""

    def __init__(self, path):
        self.path, self.handle, self.lock = (Path(path) if path else None), None, threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("x", buffering=1, encoding="utf-8")

    def write(self, kind, **value):
        if self.handle is None:
            return
        line = json.dumps(dict(kind=kind, wall_ns=now_ns(), **value), sort_keys=True, separators=(",", ":"),
                          default=str)
        with self.lock:
            self.handle.write(line + "\n")

    def close(self):
        if self.handle is not None:
            with self.lock:
                self.handle.close()
                self.handle = None


class LiveFeed:
    """One running capture -> bounded live events for the browser.  Start with ``start()``; never writes upstream."""

    def __init__(self, root, *, log_path=None, label="full Town10 map"):
        self.root, self.label = Path(root), label
        self.epoch = uuid.uuid4().hex[:12]
        self.started_ns = now_ns()
        self.calib = calibration()
        self.log = EventLog(log_path)
        self.directory = None
        self.map_tail = self.pose_tail = None
        self.wires, self.owned = OrderedDict(), OrderedDict()
        self.pending = deque()
        self.clients, self.clients_lock = {}, threading.Lock()
        self.session, self.base_ns, self.last_key = None, None, None
        self.seq = 0
        self.counts = dict(map_lines=0, pose_lines=0, emitted=0, duplicate=0, out_of_order=0, pose_timeout=0,
                           pose_errors=0, heading_join_errors=0, bad_lines=0, sessions=0, client_drops=0,
                           pending_overflow=0, record_unavailable=0, other_session=0, stale_at_read=0)
        self.last_install_ns = None
        self.last_emit_ns = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="live-feed", daemon=True)
        self.log.write("feed_start", epoch=self.epoch, root=str(self.root), label=label)

    # ---- discovery ------------------------------------------------------------------------------------------------
    def _discover(self):
        if self.directory is not None:
            return True
        candidates = [self.root / LIVE_DIR] if (self.root / LIVE_DIR).is_dir() else \
            sorted(self.root.glob("capture/cells/*/" + LIVE_DIR))
        if len(candidates) == 1:
            self.directory = candidates[0]
            self.map_tail = Tail(self.directory / MAP_FILE, started_ns=self.started_ns)
            self.pose_tail = Tail(self.directory / POSE_FILE, started_ns=self.started_ns, backlog_bytes=4 << 20)
            self.log.write("feed_directory", directory=str(self.directory))
            return True
        return False

    # ---- main loop ------------------------------------------------------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(2)
        self.log.write("feed_stop", counts=dict(self.counts))
        self.log.close()

    def _run(self):
        next_heartbeat = 0.0
        while not self.stop.is_set():
            try:
                if self._discover():
                    for line in self.pose_tail.read():
                        self._pose(line)
                    for line in self.map_tail.read(64):
                        self._install(line)
                    self._flush_pending()
                if time.monotonic() >= next_heartbeat:
                    next_heartbeat = time.monotonic() + HEARTBEAT_S
                    self._broadcast(self._heartbeat())
            except Exception as exc:  # noqa: BLE001 - the viewer must keep serving; error is logged and shown
                self.counts["bad_lines"] += 1
                self.log.write("feed_error", error=f"{type(exc).__name__}: {exc}"[:300])
            self.stop.wait(POLL_S)

    def _pose(self, line):
        if line.get("schema") != POSE_SCHEMA:
            self.counts["bad_lines"] += 1
            return
        self.counts["pose_lines"] += 1
        if line["kind"] == "UE_WIRE":
            c = line["context"]
            self.wires[(c["stream_id"], c["frame_id"])] = line
            self.wires.move_to_end((c["stream_id"], c["frame_id"]))
            while len(self.wires) > MAX_WIRES:
                self.wires.popitem(last=False)
        elif line["kind"] == "OWNED_SOURCE":
            self.owned[line["source_sha256"]] = line
            while len(self.owned) > MAX_OWNED:
                self.owned.popitem(last=False)

    def _install(self, line):
        if line.get("schema") != MAP_SCHEMA:
            self.counts["bad_lines"] += 1
            return
        self.counts["map_lines"] += 1
        if not line.get("record_available"):
            self.counts["record_unavailable"] += 1
            self.log.write("install_without_record", seq=line.get("seq"), frame_id=line["identity"].get("frame_id"))
            return
        self.last_install_ns = round(float(line["install"]["install_timestamp"]) * 1e9)
        if now_ns() - self.last_install_ns > MAX_INSTALL_AGE_NS:
            # Historical lines (e.g. a backlog after a viewer stall) are never shown as live objects.
            self.counts["stale_at_read"] += 1
            self.log.write("refused", reason="STALE_AT_READ", frame_id=line["identity"].get("frame_id"),
                           install_age_ms=(now_ns() - self.last_install_ns) / 1e6)
            return
        self.pending.append((line, time.monotonic(), now_ns()))
        if len(self.pending) > MAX_PENDING:
            self.counts["pending_overflow"] += 1
            self._emit(*self.pending.popleft(), force=True)

    def _flush_pending(self):
        while self.pending:
            line, since, read_ns = self.pending[0]
            ident = line["identity"]
            ready = (ident["stream_id"], ident["frame_id"]) in self.wires
            if not ready and time.monotonic() - since < POSE_WAIT_S:
                return               # preserve installation order; the head waits for its own frame's pose only
            self.pending.popleft()
            self._emit(line, since, read_ns, force=not ready)

    # ---- event construction ----------------------------------------------------------------------------------------
    def _emit(self, line, since, read_ns, force=False):
        ident, record = line["identity"], line["record"]
        session = f"{ident['run_id']}/{ident['cell_id']}"
        key = (int(record["frame_id"]), int(ident["capture_timestamp_ns"]))
        if self.session is not None and session != self.session:
            # One feed serves ONE capture: a line naming another run/cell can never enter this view.
            self.counts["other_session"] += 1
            self.log.write("refused", reason="OTHER_SESSION", session=session, frame_id=key[0])
            return
        if self.session is None:
            self.session, self.base_ns, self.last_key = session, int(ident["capture_timestamp_ns"]), None
            self.counts["sessions"] += 1
            self.log.write("session", session=session, base_ns=self.base_ns, epoch=self.epoch)
            self._broadcast(json.dumps(dict(type="session", session=session, base_ns=str(self.base_ns),
                                            epoch=self.epoch, server_ns=str(now_ns()))))
        if self.last_key is not None and (key[0] == self.last_key[0] or key[1] == self.last_key[1]):
            self.counts["duplicate"] += 1
            self.log.write("refused", reason="DUPLICATE", frame_id=key[0])
            return
        if self.last_key is not None and (key[0] < self.last_key[0] or key[1] < self.last_key[1]):
            self.counts["out_of_order"] += 1
            self.log.write("refused", reason="OUT_OF_ORDER", frame_id=key[0])
            return
        self.last_key = key
        install_binding = dict(run_id=ident["run_id"], cell_id=ident["cell_id"], stream_id=ident["stream_id"],
                               frame_id=ident["frame_id"], capture_timestamp=record["capture_timestamp"],
                               install_timestamp=line["install"]["install_timestamp"])
        wire = self.wires.get((record["stream_id"], record["frame_id"]))
        owned = self.owned.get(wire["source_sha256"]) if wire else None
        pose, error = bind_pose(record, wire, owned, self.calib, install_binding)
        if force and not wire:
            self.counts["pose_timeout"] += 1
        if error:
            self.counts["pose_errors"] += 1
        joined, join_error = join_producer(record, line.get("producer_records") or [], pose)
        if join_error and pose:
            self.counts["heading_join_errors"] += 1
        capture_ns = int(ident["capture_timestamp_ns"])
        rel = lambda ns: None if ns is None else (int(ns) - self.base_ns) / 1e6  # noqa: E731
        install_ns = round(float(line["install"]["install_timestamp"]) * 1e9)
        objects = []
        for raw in record["objects"]:
            info = joined.get(raw["id"], {})
            h = info.get("heading", {})
            objects.append(dict(raw=raw, track_id=raw["id"], source=raw["source_stream_id"], frame_id=raw["frame_id"],
                                capture_ms=rel(capture_ns), capture_ns=str(capture_ns),
                                heading_valid=bool(h.get("valid", False)), heading_yaw_deg=h.get("yaw_deg"),
                                heading_reason=join_error or h.get("reason"), upstream=info or None,
                                visibility=dict(state="unknown", reason="NO_QUALIFIED_VISIBILITY_EVIDENCE")))
        self.seq += 1
        emit_ns = now_ns()
        event = dict(type="event", seq=self.seq, epoch=self.epoch, session=session, source=record["stream_id"],
                     frame_id=record["frame_id"], capture_ms=rel(capture_ns), capture_ns=str(capture_ns),
                     clock_domain="unix_wall_ns", age_mapping=AGE_MAPPING, pose=pose, pose_error=error,
                     objects=objects, camera_available=False,
                     timeline=dict(install_ms=rel(install_ns), tap_ms=rel(line.get("tap_wall_ns")),
                                   tap_written_ms=rel(line.get("written_wall_ns")), viewer_read_ms=rel(read_ns),
                                   emit_ms=rel(emit_ns)),
                     action=dict(action_id=ident.get("action_id"), profile_id=ident.get("profile_id"),
                                 sequence_id=ident.get("sequence_id")),
                     tap=dict(seq=line.get("seq"), counts=line.get("tap_counts")),
                     server_ns=str(emit_ns))
        self.counts["emitted"] += 1
        self.last_emit_ns = emit_ns
        message = json.dumps(event, separators=(",", ":"), allow_nan=False)
        self._broadcast(message)
        ms = lambda a, b: round((a - b) / 1e6, 3)  # noqa: E731
        self.log.write("frame", seq=self.seq, frame_id=record["frame_id"], capture_ns=capture_ns,
                       capture_to_install_ms=ms(install_ns, capture_ns),
                       install_to_tap_written_ms=ms(int(line.get("written_wall_ns", install_ns)), install_ns),
                       capture_to_viewer_read_ms=ms(read_ns, capture_ns), capture_to_emit_ms=ms(emit_ns, capture_ns),
                       objects=len(objects), pose_error=error, heading_join=join_error or "EXACT",
                       action_id=ident.get("action_id"), bytes=len(message))

    # ---- fan-out --------------------------------------------------------------------------------------------------
    def _heartbeat(self):
        status = self.status()
        return json.dumps(dict(type="heartbeat", server_ns=str(now_ns()), epoch=self.epoch, session=self.session,
                               base_ns=None if self.base_ns is None else str(self.base_ns), status=status),
                          separators=(",", ":"))

    def status(self):
        n = now_ns()
        return dict(directory=None if self.directory is None else str(self.directory),
                    skipped_backlog_bytes=dict(map=self.map_tail.skipped_bytes if self.map_tail else 0,
                                               pose=self.pose_tail.skipped_bytes if self.pose_tail else 0),
                    tap_files=dict(map=bool(self.map_tail and self.map_tail.path.is_file()),
                                   pose=bool(self.pose_tail and self.pose_tail.path.is_file())),
                    counts=dict(self.counts), pending=len(self.pending), wires=len(self.wires),
                    since_last_install_ms=None if self.last_install_ns is None else (n - self.last_install_ns) / 1e6,
                    since_last_emit_ms=None if self.last_emit_ns is None else (n - self.last_emit_ns) / 1e6,
                    clients=len(self.clients), label=self.label)

    def _broadcast(self, message):
        with self.clients_lock:
            clients = list(self.clients.values())
        for client in clients:
            before = client.dropped
            client.push(message)
            self.counts["client_drops"] += client.dropped - before

    def connect(self, address):
        with self.clients_lock:
            if len(self.clients) >= MAX_CLIENTS:
                return None
            client = Client(uuid.uuid4().hex[:8], address)
            self.clients[client.id] = client
        self.log.write("client_connect", client=client.id, address=address)
        # Only the current session identity; never a backlog of earlier objects or frames.
        client.push(json.dumps(dict(type="hello", epoch=self.epoch, session=self.session, client=client.id,
                                    base_ns=None if self.base_ns is None else str(self.base_ns),
                                    server_ns=str(now_ns()), label=self.label)))
        return client

    def disconnect(self, client):
        client.closed = True
        with self.clients_lock:
            self.clients.pop(client.id, None)
        self.log.write("client_disconnect", client=client.id, sent=client.sent, dropped=client.dropped,
                       duration_s=(now_ns() - client.connected_ns) / 1e9)

    def stream(self, client):
        """SSE generator for one browser; the thread serving it is the only thing that waits on it."""
        try:
            while not self.stop.is_set():
                message = client.pop(1.0)
                if message is None:
                    continue
                client.sent += 1
                yield "data: " + message + "\n\n"
        finally:
            self.disconnect(client)

    def client_report(self, value):
        """Small render/flag reports posted by the browser; logged only."""
        if not isinstance(value, dict) or len(json.dumps(value)) > 16384:
            return False
        self.log.write("client_report", **{k: v for k, v in value.items() if k != "kind"},
                       report_kind=str(value.get("kind", "stats"))[:32])
        return True
