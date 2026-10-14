"""Live-view taps: bounded, non-blocking copies of already-produced evidence for the live spatial-map viewer.

Display only.  Two taps, each one bounded queue drained by one daemon thread that appends JSON lines to a
create-only file under ``<attempt>/live_view/``:

  * map process (``install_map_tap``): after a SUCCESSFUL ``install_under_state_lock`` returns, references to the
    installed (baseline-normalised) record, the edge document identity and the edge's producer records, which still
    carry the p025 world-yaw pair that the frozen baseline normaliser replaces with ``yaw_deg`` 90;
  * UE child (``install_ue_pose_tap``): the ``OWNED_SOURCE`` / ``UE_WIRE`` documents the retained pose archive already
    records for every prepared/sent frame (the frame-bound pose sidecar the replay binds offline).

The producing thread only appends a tuple of references to a bounded ``queue.Queue`` (``put_nowait``); a full queue
drops the item and counts it.  Serialisation and file I/O happen on the drain thread.  A tap failure is counted and
recorded; it never propagates into the install, the pose archive or any decision.  There is no socket, no HTTP and no
viewer dependency: the pipeline does not know whether a viewer exists.  Nothing here changes what is installed,
scored, archived or acknowledged; GT is never read.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import queue
import threading
import time

SCHEMA_MAP = "scenesense.live_view.map_install.v1"
SCHEMA_POSE = "scenesense.live_view.ue_pose.v1"
SCHEMA_STATUS = "scenesense.live_view.tap_status.v1"
LIVE_DIR = "live_view"
MAP_FILE = "map_installs.jsonl"
POSE_FILE = "ue_pose.jsonl"
QUEUE_CAPACITY = 256
CLOSE_TIMEOUT_S = 2.0
POSE_KINDS = ("OWNED_SOURCE", "UE_WIRE")
IDENTITY_FIELDS = ("schema", "run_id", "cell_id", "stream_id", "frame_id", "sequence_id", "action_id", "profile_id",
                   "decoder_identity", "capture_timestamp_ns", "service_deadline_at", "ack_timeout_at")
PRODUCER_FIELDS = ("candidate_identity", "prediction_index", "class_name", "internal_class", "score",
                   "world_x", "world_y", "world_z", "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos",
                   "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1", "local_x", "local_y", "local_z",
                   "fpn_level", "point_index", "stream_id", "frame_id", "capture_timestamp_ns")


class BoundedJsonlTap:
    """One bounded queue + one daemon writer thread + one create-only line-buffered JSONL file."""

    def __init__(self, path, *, role, build, capacity=QUEUE_CAPACITY):
        self.path, self.role, self._build = Path(path), str(role), build
        self.capacity = int(capacity)
        self._queue = queue.Queue(maxsize=self.capacity)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stream = None
        self._thread = None
        self._seq = 0
        self.counts = dict(offered=0, enqueued=0, dropped_queue_full=0, dropped_not_running=0, written=0,
                           build_errors=0, write_errors=0, offer_errors=0)
        self.first_error = None
        self.started_wall_ns = None
        self.closed_wall_ns = None
        self.status = None

    def _note(self, key, exc=None):
        with self._lock:
            self.counts[key] += 1
            if exc is not None and self.first_error is None:
                self.first_error = f"{key}: {type(exc).__name__}: {exc}"[:300]

    def start(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path, "x", buffering=1, encoding="utf-8")
            self.started_wall_ns = time.time_ns()
            self._thread = threading.Thread(target=self._run, name="live-view-" + self.role, daemon=True)
            self._thread.start()
        except Exception as exc:  # noqa: BLE001 - a display tap must never stop the pipeline
            self._stream = None
            self._note("write_errors", exc)
        return self

    @property
    def running(self):
        return self._stream is not None and not self._stop.is_set()

    def offer(self, item):
        """Non-blocking; never raises.  Called on the producing (pipeline) thread."""
        try:
            with self._lock:
                self.counts["offered"] += 1
            if not self.running:
                self._note("dropped_not_running")
                return False
            self._queue.put_nowait((time.time_ns(), item))
            with self._lock:
                self.counts["enqueued"] += 1
            return True
        except queue.Full:
            self._note("dropped_queue_full")
            return False
        except Exception as exc:  # noqa: BLE001
            self._note("offer_errors", exc)
            return False

    def snapshot_counts(self):
        with self._lock:
            return dict(self.counts)

    def _run(self):
        while True:
            try:
                tap_ns, item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            self._seq += 1
            try:
                line = json.dumps(self._build(self._seq, tap_ns, item, self), sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
            except Exception as exc:  # noqa: BLE001
                self._note("build_errors", exc)
                continue
            try:
                self._stream.write(line + "\n")
                with self._lock:
                    self.counts["written"] += 1
            except Exception as exc:  # noqa: BLE001
                self._note("write_errors", exc)

    def close(self, timeout_s=CLOSE_TIMEOUT_S):
        """Stop accepting, drain for at most ``timeout_s``, close the file and write the status record once."""
        if self.status is not None:
            return self.status
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        alive = bool(self._thread is not None and self._thread.is_alive())
        try:
            if self._stream is not None and not alive:
                self._stream.close()
        except Exception as exc:  # noqa: BLE001
            self._note("write_errors", exc)
        self.closed_wall_ns = time.time_ns()
        self.status = dict(schema=SCHEMA_STATUS, role=self.role, path=str(self.path), capacity=self.capacity,
                           counts=self.snapshot_counts(), undrained=self._queue.qsize(), writer_alive_at_close=alive,
                           first_error=self.first_error, started_wall_ns=self.started_wall_ns,
                           closed_wall_ns=self.closed_wall_ns, decision_path=False, gt_read=False)
        try:
            status_path = self.path.with_name(self.role + "_tap_status.json")
            with open(status_path, "x", encoding="utf-8") as handle:
                json.dump(self.status, handle, indent=1, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            self._note("write_errors", exc)
        return self.status


# ------------------------------------------------------------------------------------------------- map process
def map_line(seq, tap_ns, item, tap):
    document, records, installed, result = item
    producer = [{k: record[k] for k in PRODUCER_FIELDS if k in record} for record in records]
    run4 = document.get("run4_identity")
    return dict(schema=SCHEMA_MAP, seq=seq, tap_wall_ns=tap_ns, written_wall_ns=time.time_ns(),
                identity={k: document.get(k) for k in IDENTITY_FIELDS},
                run4_identity=run4 if isinstance(run4, dict) else None,
                install=result, record=installed, record_available=installed is not None,
                producer_records=producer, producer_record_count=len(producer),
                tap_counts=tap.snapshot_counts())


def install_map_tap():
    """Wrap ``phase6_map_server_v2.install`` (called by ``map_profiled_v1.main``) in THIS map process.

    Installed after ``map_class.install_map_class_conversion`` so the offer runs only after the class-converted
    document was installed successfully.  The installed record is read from the baseline's own installed-frame history
    on the ingest thread that has just written it (the only writer), without taking the map lock."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_map_server_v2 as M6
    original_install = M6.install
    if getattr(original_install, "__live_view_tap__", False):
        raise RuntimeError("live-view map tap already installed")
    state = dict(tap=None, error=None)

    def install(contract):
        server = original_install(contract)
        inner = server.install_under_state_lock
        baseline = server.baseline

        def install_with_live_view(document, ingest_at):
            result = inner(document, ingest_at)
            tap = state["tap"]
            if tap is not None:
                try:
                    key = (str(document["stream_id"]), int(document["frame_id"]))
                    tap.offer((document, list(document.get("records") or ()),
                               baseline.installed_frame_history.get(key), dict(result)))
                except Exception as exc:  # noqa: BLE001
                    tap._note("offer_errors", exc)
            return result

        server.install_under_state_lock = install_with_live_view
        service = server.DirectMapIngestService

        class LiveViewMapIngestService(service):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                try:
                    ingest_csv = getattr(self, "ingest_csv", None)
                    if ingest_csv is None:
                        raise ValueError("map ingest CSV path unavailable; live view tap disabled")
                    state["tap"] = BoundedJsonlTap(Path(ingest_csv).parent.parent / LIVE_DIR / MAP_FILE,
                                                   role="map", build=map_line).start()
                except Exception as exc:  # noqa: BLE001
                    state["error"] = f"{type(exc).__name__}: {exc}"[:300]

            def close(self):
                try:
                    return super().close()
                finally:
                    if state["tap"] is not None:
                        state["tap"].close()

        server.DirectMapIngestService = LiveViewMapIngestService
        return server

    install.__live_view_tap__ = True
    M6.install = install
    return state


# ------------------------------------------------------------------------------------------------- UE child
def pose_line(seq, tap_ns, item, tap):
    kind, value = item
    return dict(value, schema=SCHEMA_POSE, seq=seq, kind=kind, tap_wall_ns=tap_ns, written_wall_ns=time.time_ns())


@contextmanager
def install_ue_pose_tap(attempt):
    """Wrap the retained ``PoseArchive.add`` for the scope: the retained add runs first and unchanged (its exceptions
    propagate as before); afterwards UE-side OWNED_SOURCE/UE_WIRE documents are offered to the bounded tap.  Must be
    entered after the C5 capacity substitution (which replaces ``PoseArchive.__init__`` only) and before
    ``pose.install_ue_pose`` creates the archive."""
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose as POSE
    archive_class = POSE.PoseArchive
    original_add = archive_class.add
    if getattr(original_add, "__live_view_tap__", False):
        raise RuntimeError("live-view pose tap already installed")
    tap = BoundedJsonlTap(Path(attempt) / LIVE_DIR / POSE_FILE, role="ue_pose", build=pose_line).start()

    def add(self, kind, **value):
        original_add(self, kind, **value)
        if kind in POSE_KINDS and getattr(self, "side", None) == "ue":
            tap.offer((kind, value))

    add.__live_view_tap__ = True
    archive_class.add = add
    try:
        yield tap
    finally:
        archive_class.add = original_add
        tap.close()
