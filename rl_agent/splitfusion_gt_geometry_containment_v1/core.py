"""``PrepCoreContained``: the FREEZE-lifetime PrepCore whose engine is adopted as ``ContainedEngine``, plus bounded
diagnostics for each contained refusal.  Prep worker only; installed inside the unchanged lifetime-repair context.

Every operation is the inherited ``PrepCoreLifetime`` / ``PrepCore`` implementation (FREEZE exception lifetime,
fatal snapshot, BIND refusal mapping, cancels, publication).  After a FREEZE that the engine contained, while the
sealed source fd is still open (the worker closes it after ``handle`` returns), one bounded record is written:

  <trace>/geometry_refusal_v1/REFUSALS.jsonl           <= MAX_RECORDS lines (then only counters advance)
  <trace>/geometry_refusal_v1/source_NNN/              first MAX_SOURCES refusals: ERROR/REQUEST/SOURCE/READY
  <trace>/geometry_refusal_v1/COUNTERS.json             atomically replaced after each refusal

A diagnostics write error is counted and reported on stderr; it never changes the frame's outcome or the worker.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as R
from rl_agent.splitfusion_live_gt_optimization_v4 import memfd_transport as M

from . import classify as CL, engine as E

MAX_RECORDS = 128
MAX_SOURCES = 2
SCHEMA = CL.VERSION + ".diagnostics"


class Diagnostics:
    def __init__(self, directory):
        self.root = None if directory is None else Path(directory) / "geometry_refusal_v1"
        self.counters = dict(schema=SCHEMA, refusals=0, records_written=0, records_dropped=0, sources_written=0,
                             sources_skipped=0, write_errors=0, last_error=None)

    def _write_new(self, path, data):
        R._create_file(path, data)

    def _source(self, index, args, fd, record):
        out = self.root / f"source_{index:03d}"
        out.mkdir(mode=0o700)
        request = json.dumps(args, sort_keys=True, allow_nan=False).encode()
        if len(request) > R.MAX_DIAGNOSTIC_BYTES:
            raise ValueError("refusal request exceeds the diagnostic bound")
        self._write_new(out / "ERROR.json", json.dumps(record, indent=1, sort_keys=True).encode())
        self._write_new(out / "REQUEST.json", request)
        hashes = {"ERROR.json": hashlib.sha256((out / "ERROR.json").read_bytes()).hexdigest(),
                  "REQUEST.json": hashlib.sha256(request).hexdigest()}
        size = os.fstat(fd).st_size
        M.verify(fd, size)
        data = bytearray()
        while len(data) < size:                       # pread: the worker's descriptor offset is not moved
            piece = os.pread(fd, min(1024 * 1024, size - len(data)), len(data))
            if not piece:
                raise OSError("truncated sealed refusal source")
            data.extend(piece)
        self._write_new(out / "SOURCE.bin", data)
        hashes["SOURCE.bin"] = hashlib.sha256(data).hexdigest()
        self._write_new(out / "READY.json", json.dumps(dict(complete=True, sha256=hashes), indent=1).encode())

    def record(self, args, fd, record, job):
        self.counters["refusals"] += 1
        if self.root is None:
            return
        try:
            self.root.mkdir(mode=0o700, exist_ok=True)
            n = self.counters["refusals"]
            line = dict(record, index=n, job=job)
            if self.counters["records_written"] < MAX_RECORDS:
                with open(self.root / "REFUSALS.jsonl", "a") as stream:
                    stream.write(json.dumps(line, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                self.counters["records_written"] += 1
            else:
                self.counters["records_dropped"] += 1
            if self.counters["sources_written"] < MAX_SOURCES and fd is not None:
                self._source(self.counters["sources_written"] + 1, args, fd, line)
                self.counters["sources_written"] += 1
            else:
                self.counters["sources_skipped"] += 1
        except BaseException as error:
            self.counters["write_errors"] += 1
            self.counters["last_error"] = f"{type(error).__name__}: {error}"[:300]
            try:
                print(f"GEOMETRY_REFUSAL_DIAGNOSTIC_WRITE_FAILED: {self.counters['last_error']}", file=sys.stderr,
                      flush=True)
            except BaseException:
                pass
        finally:
            try:
                if self.root is not None and self.root.is_dir():
                    temporary = self.root / ".COUNTERS.json.tmp"
                    temporary.write_text(json.dumps(self.counters, indent=1, sort_keys=True) + "\n")
                    os.replace(temporary, self.root / "COUNTERS.json")
            except BaseException:
                self.counters["write_errors"] += 1


class PrepCoreContained(R.PrepCoreLifetime):
    """Unchanged PrepCoreLifetime; the constructed engine is adopted, contained FREEZEs leave bounded evidence."""

    diagnostics_directory = None

    def __init__(self, config):
        super().__init__(config)
        E.adopt(self.engine)
        self.geometry_diagnostics = Diagnostics(self.diagnostics_directory)

    def handle(self, op, args, fd):
        result = super().handle(op, args, fd)
        if op == "FREEZE":
            job = self.jobs[result[0]["job"]]
            record = self.engine.contained_refusal(job["handle"])
            if record is not None:
                self.geometry_diagnostics.record(args, fd, record, result[0]["job"])
        return result


@contextmanager
def install(diagnostics_directory):
    """Inside ``repair.install``: PrepCore resolves to PrepCoreContained for this process; restored on exit."""
    if PC.PrepCore is not R.PrepCoreLifetime:
        raise RuntimeError("containment installs only on top of the FREEZE lifetime repair")
    previous = PrepCoreContained.diagnostics_directory
    PrepCoreContained.diagnostics_directory = None if diagnostics_directory is None else str(diagnostics_directory)
    PC.PrepCore = PrepCoreContained
    try:
        yield
    finally:
        PC.PrepCore = R.PrepCoreLifetime
        PrepCoreContained.diagnostics_directory = previous
