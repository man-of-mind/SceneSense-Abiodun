"""Durable map journal: the exact installed map records, as the map server returned them.

The retained UE adapter already reads every ACK-installed frame back from the map's bounded installed-frame history
(``GET /api/fusion_streams/installed/<stream>/<frame>``) on its exact-record retrieval worker and keeps only derived
metrics. This module tees the raw response bytes of THAT request (thread-local; no second request, no change to the
parse or its outcome classification) and persists:

  installed_map_records.jsonl  one line per retrieval attempt: frame_id, outcome, retrieval wall times, sha256 of the
                               raw bytes and, for INSTALLED records, the full map document (status + record with the
                               measured world coordinates, scores, identities and map install stamps)
  map_state_latest.json        atomically replaced after every INSTALLED record: the most recent installed map state
                               of the (single) stream, i.e. the persisted/updated map JSON
  JOURNAL_SUMMARY.json         counts at close

Writes happen on the retrieval worker after the HTTP response, never on the decision, send or map-install path. A
journal I/O failure is counted and reported; it never changes the retrieval outcome.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import urllib.request

_LOCAL = threading.local()
_STATE = {"original": None}


class _TeeResponse:
    def __init__(self, response, sink):
        self._response, self._sink = response, sink

    def read(self, *args):
        data = self._response.read(*args)
        self._sink.append(bytes(data))
        return data

    def __enter__(self):
        self._response.__enter__()
        return self

    def __exit__(self, *exc):
        return self._response.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._response, name)


def _urlopen(*args, **kwargs):
    response = _STATE["original"](*args, **kwargs)
    sink = getattr(_LOCAL, "sink", None)
    return response if sink is None else _TeeResponse(response, sink)


def install_tee():
    if _STATE["original"] is not None:
        raise RuntimeError("map journal tee already installed")
    _STATE["original"] = urllib.request.urlopen
    urllib.request.urlopen = _urlopen


def uninstall_tee():
    if _STATE["original"] is not None:
        urllib.request.urlopen = _STATE["original"]
        _STATE["original"] = None


def begin_capture():
    sink = []
    _LOCAL.sink = sink
    return sink


def end_capture(sink):
    _LOCAL.sink = None
    return b"".join(sink)


class MapJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._stream = None
        self.counts = dict(attempts=0, installed=0, other_outcomes=0, write_errors=0, parse_errors=0)
        self.outcomes = {}
        self.first_error = None
        self.latest_frame = None

    def _open(self):
        if self._stream is None:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._stream = open(self.directory / "installed_map_records.jsonl", "x", buffering=1)

    def record(self, *, frame_id, outcome, raw, retrieval_start_wall_ns, retrieval_end_wall_ns):
        try:
            document = None
            if outcome == "OK" and raw:
                try:
                    document = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    self.counts["parse_errors"] += 1
            row = dict(frame_id=int(frame_id), outcome=outcome, retrieval_start_wall_ns=int(retrieval_start_wall_ns),
                       retrieval_end_wall_ns=int(retrieval_end_wall_ns), raw_bytes=len(raw),
                       raw_sha256=hashlib.sha256(raw).hexdigest() if raw else None, map_document=document,
                       coordinate_semantics="MEASURED_EDGE_OUTPUT_AS_INSTALLED (no display prediction)")
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            with self._lock:
                self._open()
                self._stream.write(line + "\n")
                self.counts["attempts"] += 1
                self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
                if document is not None:
                    self.counts["installed"] += 1
                    self.latest_frame = int(frame_id)
                    latest = self.directory / "map_state_latest.json"
                    temporary = latest.with_name(latest.name + ".partial")
                    temporary.write_text(json.dumps(dict(row, journal_installed_index=self.counts["installed"]),
                                                    sort_keys=True))
                    os.replace(temporary, latest)
                else:
                    self.counts["other_outcomes"] += 1
        except Exception as exc:  # noqa: BLE001 - evidence I/O must never change the retrieval outcome
            with self._lock:
                self.counts["write_errors"] += 1
                if self.first_error is None:
                    self.first_error = f"{type(exc).__name__}: {exc}"[:300]

    def close(self):
        with self._lock:
            if self._stream is not None:
                self._stream.flush()
                os.fsync(self._stream.fileno())
                self._stream.close()
                self._stream = None
            summary = dict(schema="splitfusion.m3_route_b_v1.map_journal.v1", counts=dict(self.counts),
                           outcomes=dict(self.outcomes), first_error=self.first_error, latest_frame=self.latest_frame,
                           directory=str(self.directory))
            if self.directory.is_dir():
                path = self.directory / "JOURNAL_SUMMARY.json"
                if not path.exists():
                    path.write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")
            return summary
