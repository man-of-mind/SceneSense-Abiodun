"""Narrow E3 process-scope evidence for a numerical-GT engineering run.

The retained inventory remains authoritative for every ordinary process. A
numeric worker is exempted from the baseline thread-cap environment only when
its sampled host PID, live NSpid/starttime observation, source-bound READY and
clean CLOSE all reconcile. A command-line candidate alone is never exempted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time


SCHEMA = "splitfusion.numeric_gt.e3_scope.v1"
MAPPING_SCHEMA = "splitfusion.numeric_gt.e3_pid_mapping.v1"
WORKER_SCOPE_SCHEMA = "splitfusion.numeric_gt.worker_scope.v1"
MAPPING_FILENAME = "numeric_pid_mapping.jsonl"
CAP_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
WORKERS = frozenset(("producer", "scorer", "archive"))
PROCESS_MODULE = "rl_agent.splitfusion_numeric_gt_runtime_v1.process"
ARCHIVE_MODULE = "rl_agent.splitfusion_live_gt_handoff_v1.archive"
PROCESS_CANDIDATE = "numeric_process_candidate"
ARCHIVE_CANDIDATE = "numeric_archive_candidate"
_MODULE_PATTERN = re.compile(r"(?:^|\s)-m\s+([^\s]+)(?:\s|$)")


class ProcessScopeError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise ProcessScopeError(message)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _is_digest(value):
    return type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def candidate_kind(command: str) -> str | None:
    """Exact module candidate detection; never an exemption decision."""
    if type(command) is not str:
        return None
    match = _MODULE_PATTERN.search(command)
    module = None if match is None else match.group(1)
    return (PROCESS_CANDIDATE if module == PROCESS_MODULE else
            ARCHIVE_CANDIDATE if module == ARCHIVE_MODULE else None)


def _stat_fields(raw: bytes):
    text = raw.decode("utf-8", "replace")
    _require(")" in text and "(" in text, "malformed process stat")
    tail = text.rpartition(")")[2].split()
    _require(len(tail) > 19, "process stat lacks starttime")
    return int(tail[1]), int(tail[19])


def _nspid(raw: bytes):
    rows = [line.split()[1:] for line in raw.decode("utf-8", "replace").splitlines()
            if line.startswith("NSpid:")]
    _require(len(rows) == 1 and rows[0], "process status lacks one NSpid vector")
    values = [int(value) for value in rows[0]]
    _require(all(value > 0 for value in values), "invalid NSpid vector")
    return values


def observe_pid_mapping(host_pid: int, *, proc_root="/proc") -> dict:
    """Read one candidate's namespace and process-start identity while alive."""
    _require(type(host_pid) is int and host_pid > 1, "positive host PID required")
    root = Path(proc_root)
    path = root / str(host_pid)
    before = (path / "stat").read_bytes()
    status = (path / "status").read_bytes()
    command_raw = (path / "cmdline").read_bytes()
    after = (path / "stat").read_bytes()
    ppid, start = _stat_fields(before)
    _require(_stat_fields(after) == (ppid, start) and start > 0,
             "PID changed identity during scope observation")
    values = _nspid(status)
    _require(values[0] == host_pid, "host PID differs from first NSpid")
    command = command_raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    kind = candidate_kind(command)
    _require(kind is not None, "PID is not an exact numerical child candidate")
    parent_status = (root / str(ppid) / "status").read_bytes()
    parent_ns = _nspid(parent_status)
    _require(parent_ns[0] == ppid, "parent host PID differs from NSpid")
    return dict(schema=MAPPING_SCHEMA, host_pid=host_pid, host_parent_pid=ppid,
        nspid=values, parent_nspid=parent_ns, proc_start_ticks=start,
        observed_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        candidate_kind=kind, status_sha256=_digest(status), stat_sha256=_digest(before),
        cmdline_sha256=_digest(command_raw), inventory_sampled=False,
        inventory_t_raw_ns=None, error=None)


def wrap_inventory_class(base_class):
    """Preserve retained rows/checks, adding exact candidates and NSpid sidecar.

    Install before the retained runner constructs its inventory:
    ``TCAP.ProcessEnvironmentInventory = wrap_inventory_class(TCAP.ProcessEnvironmentInventory)``.
    The final E3 decision must use :func:`verify_scope`, not the old summary's
    unqualified candidate-name violations or a blanket role exemption.
    """
    class NumericInventory(base_class):
        def __init__(self, *args, classify, **kwargs):
            def augmented(comm, command):
                return candidate_kind(command) or classify(comm, command)
            super().__init__(*args, classify=augmented, **kwargs)
            self.numeric_mapping_path = self.path.with_name(MAPPING_FILENAME)
            self._numeric_mapping = self.numeric_mapping_path.open("x", encoding="utf-8")
            self.numeric_mappings = []

        def _examine(self, pid):
            command_path = Path(self._proc) / str(pid) / "cmdline"
            try:
                command = command_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                command = ""
            kind = candidate_kind(command)
            mapping = None
            if kind is not None:
                try:
                    mapping = observe_pid_mapping(pid, proc_root=self._proc)
                except (OSError, ProcessScopeError, ValueError) as exc:
                    mapping = dict(schema=MAPPING_SCHEMA, host_pid=pid, candidate_kind=kind,
                        inventory_sampled=False, error=f"{type(exc).__name__}: {exc}"[:256])
            before = len(self._rows)
            super()._examine(pid)
            if mapping is not None:
                rows = self._rows[before:]
                matched = [row for row in rows if row.get("pid") == pid
                           and row.get("role") == kind]
                if len(matched) == 1 and mapping.get("error") is None:
                    mapping["inventory_sampled"] = True
                    mapping["inventory_t_raw_ns"] = matched[0].get("t_raw_ns")
                self.numeric_mappings.append(mapping)
                self._numeric_mapping.write(json.dumps(mapping, sort_keys=True) + "\n")
                self._numeric_mapping.flush()

        def stop(self):
            try:
                return super().stop()
            finally:
                self._numeric_mapping.close()

    return NumericInventory


def _worker_ready(record, source_digest, source_count):
    role, ready, close, exitcode = (record.get(name) for name in
        ("role", "ready", "close", "exitcode"))
    _require(role in WORKERS and type(ready) is dict and type(close) is dict
             and type(exitcode) is int and exitcode == 0,
             "worker needs actual READY, clean CLOSE and exit zero")
    if role in ("producer", "scorer"):
        _require(ready.get("status") == "READY" and ready.get("role") == role
                 and ready.get("authentication") ==
                     "INHERITED_SOCKET_CAPABILITY_CREATOR_UID_PID_AND_NONCE"
                 and ready.get("source_files") == source_count
                 and ready.get("cuda_initialized") is False
                 and close.get("closed") is True and close.get("valid") is True
                 and close.get("retained_jobs") == 0 and close.get("failure") is None,
                 "numeric worker READY/CLOSE/source count differs")
        return ready.get("pid"), ready.get("parent_pid"), PROCESS_CANDIDATE
    context = ready.get("context", {})
    binding = ready.get("config", {}).get("binding", {})
    _require(ready.get("status") == "READY" and type(context) is dict
             and context.get("parent_death_signal") == "SIGTERM"
             and context.get("numerical_imports") == []
             and type(binding) is dict
             and binding.get("numeric_worker_source_digest") == source_digest
             and close.get("child_pid") == context.get("pid")
             and close.get("child_returncode") == 0
             and close.get("failed") is False and close.get("fault") is None
             and close.get("outstanding_jobs") == 0,
             "archive READY/CLOSE/source binding differs")
    return context.get("pid"), context.get("parent_pid"), ARCHIVE_CANDIDATE


def verify_scope(*, inventory_rows, mappings, worker_scope_records,
                 baseline, expected_sources) -> dict:
    """Recompute E3 with exactly three identity-bound exceptions, fail closed."""
    _require(type(inventory_rows) is list and len(inventory_rows) <= 200000
             and type(mappings) is list and len(mappings) <= 10000
             and type(worker_scope_records) is list and len(worker_scope_records) == 3,
             "bounded original inventory, mapping and worker records required")
    _require(type(baseline) is dict and set(baseline) == set(CAP_VARIABLES)
             and type(expected_sources) is dict and 1 <= len(expected_sources) <= 1000
             and all(type(name) is str and _is_digest(digest)
                     for name, digest in expected_sources.items()),
             "exact baseline and pinned source map required")
    source_digest = _digest(_canonical(expected_sources))
    records = {}
    for item in worker_scope_records:
        _require(type(item) is dict and set(item) == {
            "role", "ready", "close", "exitcode", "source_sha256"}
            and item["role"] not in records and item["source_sha256"] == expected_sources,
            "duplicate or foreign source-bound worker record")
        records[item["role"]] = item
    _require(set(records) == WORKERS, "producer, scorer and archive records required")
    sampled = {row["pid"]: row for row in inventory_rows
               if type(row) is dict and type(row.get("pid")) is int}
    _require(len(sampled) == len([row for row in inventory_rows if type(row) is dict
                                  and type(row.get("pid")) is int]),
             "duplicate host PID inventory rows cannot establish scope")
    mapping_by_pid = {row["host_pid"]: row for row in mappings
                      if type(row) is dict and type(row.get("host_pid")) is int}
    _require(len(mapping_by_pid) == len(mappings), "duplicate or malformed PID mapping evidence")
    issues, exempt = [], {}
    for role, item in sorted(records.items()):
        try:
            namespace_pid, parent_pid, kind = _worker_ready(item, source_digest, len(expected_sources))
            _require(type(namespace_pid) is int and namespace_pid > 1
                     and type(parent_pid) is int and parent_pid > 1,
                     "worker READY namespace/parent PID required")
            candidates = [row for row in mappings if row.get("candidate_kind") == kind
                          and row.get("error") is None and row.get("inventory_sampled") is True
                          and type(row.get("nspid")) is list and row["nspid"][-1] == namespace_pid
                          and type(row.get("parent_nspid")) is list
                          and row["parent_nspid"][-1] == parent_pid]
            _require(len(candidates) == 1, "worker lacks unique sampled host/namespace PID mapping")
            mapping = candidates[0]
            pid = mapping["host_pid"]
            row = sampled.get(pid)
            _require(row is not None and row.get("role") == kind
                     and row.get("in_container") == (len(mapping["nspid"]) > 1)
                     and row.get("t_raw_ns") == mapping.get("inventory_t_raw_ns")
                     and type(mapping.get("proc_start_ticks")) is int
                     and mapping["proc_start_ticks"] > 0
                     and mapping["nspid"][0] == pid
                     and mapping["parent_nspid"][0] == mapping["host_parent_pid"]
                     and type(mapping.get("observed_raw_ns")) is int
                     and abs(mapping["observed_raw_ns"] - row["t_raw_ns"]) < 10_000_000_000
                     and all(_is_digest(mapping.get(name)) for name in
                             ("status_sha256", "stat_sha256", "cmdline_sha256")),
                     "worker host PID/starttime evidence differs from original inventory")
            variables = row.get("variables")
            _require(type(variables) is dict and
                     all(variables.get(name) == "1" for name in CAP_VARIABLES),
                     "worker environment unreadable or differs from bounded CPU-child scope")
            _require(pid not in exempt, "worker host PID reused by another role")
            exempt[pid] = role
        except (ProcessScopeError, TypeError, KeyError, IndexError) as exc:
            issues.append(dict(role=role, issue=str(exc)))
    violations, unreadable, ue_rows = [], [], 0
    for row in inventory_rows:
        if type(row) is not dict or type(row.get("pid")) is not int:
            continue
        pid, role, variables = row["pid"], row.get("role"), row.get("variables")
        if role == "ue":
            ue_rows += 1
            continue
        if pid in exempt:
            continue
        if role in (PROCESS_CANDIDATE, ARCHIVE_CANDIDATE):
            violations.append(dict(pid=pid, role=role, reason="UNBOUND_NUMERIC_CANDIDATE"))
        if variables is None:
            unreadable.append(dict(pid=pid, role=role, reason=row.get("environ_source")))
            continue
        for name in CAP_VARIABLES:
            if variables.get(name) != baseline[name]:
                violations.append(dict(pid=pid, role=role, variable=name,
                    value=variables.get(name), baseline=baseline[name]))
    if not ue_rows:
        issues.append(dict(role="ue", issue="retained E3 UE process was not sampled"))
    return dict(schema=SCHEMA, source_digest=source_digest, inventory_rows=len(sampled),
        sampled_ue_processes=ue_rows, exempt_host_pids=exempt,
        worker_issues=issues, scope_violations=violations, unreadable=unreadable,
        scope_ok=not issues and not violations)


def verify_scope_files(*, capture_root, cell_root, baseline, expected_sources) -> dict:
    """Load exact retained E3 and three live worker records, then recompute scope.

    The inventory is at the retained capture root. The UE scope is copied into
    the cell; the edge scope is under its copied ``profiler/edge`` evidence.
    Unique constrained suffixes avoid assuming a mutable attempt ID.
    """
    capture = Path(capture_root).resolve(strict=True)
    cell = Path(cell_root).resolve(strict=True)
    _require(cell.is_relative_to(capture), "worker scopes belong to another capture")
    inventory = capture / "threadcap" / "PROCESS_ENV_INVENTORY.jsonl"
    mapping = inventory.with_name(MAPPING_FILENAME)
    _require(inventory.is_file() and mapping.is_file()
             and not inventory.is_symlink() and not mapping.is_symlink(),
             "retained inventory and live mapping evidence required")

    def selected(name, *, edge=False):
        candidates = []
        for path in cell.rglob(name):
            if (not path.is_file() or path.is_symlink()
                    or any(parent.is_symlink() for parent in path.parents if parent != cell)
                    or not path.resolve().is_relative_to(cell)):
                continue
            if edge and path.parts[-3:] != ("profiler", "edge", name):
                continue
            candidates.append(path)
        _require(len(candidates) == 1, "one exact " + name + " in cell required")
        return candidates[0]

    ue_path = selected("numeric_ue_scope.json")
    edge_path = selected("numeric_edge_scope.json", edge=True)

    def lines(path, max_lines):
        _require(path.stat().st_size <= 64 * 1024 * 1024, "bounded scope JSONL required")
        with path.open("r", encoding="utf-8") as stream:
            values = [json.loads(line) for line in stream if line.strip()]
        _require(len(values) <= max_lines, "scope JSONL row limit exceeded")
        return values

    inventory_lines = lines(inventory, 200001)
    mapping_lines = lines(mapping, 10000)
    _require(len(inventory_lines) >= 2 and type(inventory_lines[-1]) is dict
             and set(inventory_lines[-1]) == {"summary"}
             and type(inventory_lines[-1]["summary"]) is dict,
             "retained inventory summary footer required")
    rows, summary = inventory_lines[:-1], inventory_lines[-1]["summary"]
    _require(summary.get("rows") == len(rows) and summary.get("errors") == 0,
             "retained process inventory failed or missed rows")

    def scope(path, roles):
        _require(path.stat().st_size <= 2 * 1024 * 1024, "bounded worker scope record required")
        value = json.loads(path.read_bytes())
        _require(type(value) is dict and value.get("schema") == WORKER_SCOPE_SCHEMA
                 and type(value.get("workers")) is list
                 and {item.get("role") for item in value["workers"]} == roles
                 and len(value["workers"]) == len(roles),
                 "exact source-bound worker scope roles required")
        return value["workers"]

    workers = scope(ue_path, {"producer"}) + scope(edge_path, {"scorer", "archive"})
    result = verify_scope(inventory_rows=rows, mappings=mapping_lines,
        worker_scope_records=workers, baseline=baseline, expected_sources=expected_sources)
    return dict(result, files={str(path): _digest(path.read_bytes()) for path in
        (inventory, mapping, ue_path, edge_path)}, inventory_footer=summary)


__all__ = ["ProcessScopeError", "candidate_kind", "observe_pid_mapping",
           "wrap_inventory_class", "verify_scope", "verify_scope_files",
           "MAPPING_FILENAME", "WORKER_SCOPE_SCHEMA"]
