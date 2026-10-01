#!/usr/bin/env python3
"""Frozen analysis export of one profiled attempt (read-only over the evidence).

Turns the raw profiler JSONL, the sampler series and the unchanged Phase-6
evidence into flat CSV tables plus a hash manifest, so independent analyses
(e.g. Claude and Codex bottleneck studies) start from byte-identical inputs and
neither re-implements the record parsing.  Nothing here interprets, filters
for significance or ranks bottlenecks; every row of the source is kept, and
derived columns are plain differences on one clock.

Output (create-only directory):

==================  =========================================================
``frames.csv``      one row per transmitted frame: identity, kind, payload,
                    map outcome, ticket outcome, and per registered stage the
                    first live span's ``start_raw_ns``, wall ``dur_ms`` and
                    ``cuda_ms``
``spans.csv``       every span (all processes): thread, parent, stage, phase,
                    raw boundaries, duration, CUDA time/status, identity
``marks.csv``       every mark (receipts, reassembly, queue, terminals)
``gauges.csv``      every queue-depth gauge sample
``gpu.csv``         sampler: device and host series (one row per sample)
``procs.csv``       sampler: per role process CPU/RSS/threads/context switches
``threads.csv``     sampler: per-thread CPU deltas (UE, edge, map), with the
                    container-namespace TID for the edge
``gpu_procs.csv``   sampler: per-process GPU memory and SM/memory utilization
``tickets.csv``     Phase-6 timing characterization, one row per reward ticket
``MANIFEST.json``   schema, source and export SHA-256, row counts, clocks,
                    claim scope
==================  =========================================================

All ``*_raw_ns`` are ``CLOCK_MONOTONIC_RAW`` on one host-wide clock (verified
per attempt by the analyzer).  ``capture_wall_ns`` is realtime and must not be
subtracted from raw stamps without the per-process offsets in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import recorder_v1 as RV
from . import stages_v1 as SV

SCHEMA = "scenesense.single_gpu_profiler.export.v1"
IDENTITY = ("frame_id", "tensor_seq", "session_uuid", "decision_seq", "ticket_seq",
            "reward_requested", "frame_kind", "mode_id", "q_e4")
SPAN_STAGES = tuple(name for name, spec in SV.STAGES.items() if spec["kind"] == "span")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> list[dict]:
    return list(RV.iter_records(path)) if Path(path).is_file() else []


def _write(path: Path, header: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in header})
            count += 1
    return count


def span_rows(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict]:
    rows = []
    for process, records in streams.items():
        threads = {r["tid"]: r.get("name") for r in records if r.get("r") == "thread"}
        spans = {r["seq"]: r for r in records if r.get("r") == "span"}
        cuda = {r["span_seq"]: r for r in records if r.get("r") == "cuda"}
        header = next((r for r in records if r.get("r") == "header"), {})
        for seq, span in sorted(spans.items()):
            parent = spans.get(span.get("parent")) if span.get("parent") is not None else None
            resolved = cuda.get(seq) or {}
            rows.append({"process": process, "pid": header.get("pid"), "tid": span.get("tid"),
                         "thread_name": threads.get(span.get("tid")), "seq": seq,
                         "parent_seq": span.get("parent"),
                         "parent_stage": None if parent is None else parent.get("stage"),
                         "stage": span.get("stage"), "phase": span.get("ph"),
                         "derived": bool(span.get("derived")), "ok": span.get("ok"),
                         "start_raw_ns": span["start_raw_ns"], "end_raw_ns": span["end_raw_ns"],
                         "dur_ms": (span["end_raw_ns"] - span["start_raw_ns"]) / 1e6,
                         "cuda_ms": resolved.get("gpu_ms"),
                         "cuda_status": resolved.get("status"),
                         **{k: span.get(k) for k in IDENTITY}})
    return rows


SPAN_HEADER = ("process", "pid", "tid", "thread_name", "seq", "parent_seq", "parent_stage",
               "stage", "phase", "derived", "ok", "start_raw_ns", "end_raw_ns", "dur_ms",
               "cuda_ms", "cuda_status", *IDENTITY)


def frame_rows(spans: Sequence[Mapping[str, Any]], ue: Mapping[str, Any],
               per_frame: Mapping[int, Mapping[str, str]],
               map_outcomes: Mapping[int, str], tickets: Mapping[int, Mapping[str, Any]],
               feedback: Mapping[int, int]) -> list[dict]:
    """One row per transmitted frame; first live span per stage (pure)."""
    first: dict[tuple[int, str], Mapping[str, Any]] = {}
    for span in sorted(spans, key=lambda s: (s["start_raw_ns"], s["seq"])):
        if span.get("phase") != "live" or span.get("frame_id") in (None, ""):
            continue
        first.setdefault((int(span["frame_id"]), span["stage"]), span)
    decisions = {int(d["frame_id"]): d for d in ue.get("decisions") or ()}
    rows = []
    for item in sorted(ue.get("transmitted_identities") or (), key=lambda x: x["frame_id"]):
        frame = int(item["frame_id"])
        identity = dict(item["run4_identity"])
        decision = decisions.get(frame, {})
        stages = dict(decision.get("stages") or {})
        pf = per_frame.get(frame, {})
        ticket = tickets.get(frame, {})
        row = {**{k: identity.get(k) for k in IDENTITY}, "frame_id": frame,
               "capture_wall_ns": (decision.get("capture") or {}).get("ns"),
               "rgb_receipt_raw_ns": stages.get("rgb_receipt_raw_ns"),
               "action_open_raw_ns": (decision.get("action_open") or {}).get("ns"),
               "first_datagram_raw_ns": stages.get("first_packet_send_raw_ns"),
               "last_datagram_raw_ns": stages.get("last_packet_send_raw_ns"),
               "feedback_receipt_raw_ns": feedback.get(frame),
               "payload_bytes": pf.get("payload_bytes"), "datagrams": pf.get("payload_chunks"),
               "map_outcome": map_outcomes.get(frame),
               "ticket_terminal": ticket.get("resolution_terminal"),
               "ticket_reward": ticket.get("reward"),
               "action_open_to_feedback_ms": ticket.get("action_open_to_feedback_ms")}
        for stage in SPAN_STAGES:
            span = first.get((frame, stage))
            row[f"{stage}__start_raw_ns"] = None if span is None else span["start_raw_ns"]
            row[f"{stage}__dur_ms"] = None if span is None else span["dur_ms"]
            row[f"{stage}__cuda_ms"] = None if span is None else span["cuda_ms"]
        rows.append(row)
    return rows


FRAME_HEADER = (*IDENTITY, "capture_wall_ns", "rgb_receipt_raw_ns", "action_open_raw_ns",
                "first_datagram_raw_ns", "last_datagram_raw_ns", "feedback_receipt_raw_ns",
                "payload_bytes", "datagrams", "map_outcome", "ticket_terminal", "ticket_reward",
                "action_open_to_feedback_ms",
                *[f"{s}__{c}" for s in SPAN_STAGES for c in ("start_raw_ns", "dur_ms", "cuda_ms")])


def _csv(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def export(output_root: Path, target: Path,
           result: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    output_root = Path(output_root)
    cells = sorted((output_root / "cells").iterdir())
    if len(cells) != 1:
        raise RuntimeError("expected exactly one cell")
    cell = cells[0]
    sources = {"ue": cell / "profiler" / "ue.profile.jsonl",
               "edge": cell / "profiler" / "edge" / "edge.profile.jsonl",
               "map": cell / "profiler" / "map.profile.jsonl",
               "sampler": output_root / "profiler" / "sampler.jsonl",
               "phase6_ue": cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json",
               "per_frame_metrics": cell / "per_frame_metrics.csv",
               "map_feedback": cell / "map_feedback.csv",
               "timing": output_root / "PHASE6_TIMING_CHARACTERIZATION.json",
               "analysis": output_root / "PROFILER_ANALYSIS.json",
               "result": output_root / "QUALIFICATION_RESULT.json"}
    missing = [name for name, path in sources.items() if not path.is_file()
               and not (name == "result" and result is not None)]
    if missing:
        raise RuntimeError(f"evidence missing: {missing}")
    streams = {name: _load(sources[name]) for name in ("ue", "edge", "map")}
    sampler = _load(sources["sampler"])
    ue = json.loads(sources["phase6_ue"].read_text(encoding="utf-8"))
    timing = json.loads(sources["timing"].read_text(encoding="utf-8"))
    analysis = json.loads(sources["analysis"].read_text(encoding="utf-8"))
    target = Path(target)
    target.mkdir(parents=False, exist_ok=False)                 # create-only
    spans = span_rows(streams)
    per_frame = {int(r["frame_id"]): r for r in _csv(sources["per_frame_metrics"])
                 if r.get("frame_id")}
    map_outcomes = {int(r["frame_id"]): r.get("outcome") for r in _csv(sources["map_feedback"])
                    if str(r.get("terminal", "")).lower() in ("true", "1")}
    tickets = {int(t["frame_id"]): t for t in timing.get("tickets") or ()}
    feedback = {int(f["frame_id"]): int(f["receipt"]["ns"]) for f in ue.get("feedback_rows") or ()}
    counts = {}
    counts["frames.csv"] = _write(target / "frames.csv", FRAME_HEADER,
                                  frame_rows(spans, ue, per_frame, map_outcomes, tickets, feedback))
    counts["spans.csv"] = _write(target / "spans.csv", SPAN_HEADER, spans)
    marks = []
    for process, records in streams.items():
        for r in records:
            if r.get("r") == "mark":
                extra = {k: v for k, v in r.items()
                         if k not in ("r", "seq", "stage", "tid", "t_raw_ns", "ph", *IDENTITY)}
                marks.append({"process": process, "tid": r.get("tid"), "seq": r.get("seq"),
                              "stage": r.get("stage"), "phase": r.get("ph"),
                              "t_raw_ns": r.get("t_raw_ns"), **{k: r.get(k) for k in IDENTITY},
                              "extra_json": json.dumps(extra, sort_keys=True)})
    counts["marks.csv"] = _write(target / "marks.csv",
                                 ("process", "tid", "seq", "stage", "phase", "t_raw_ns",
                                  *IDENTITY, "extra_json"), marks)
    gauges = [{"process": process, "t_raw_ns": r["t_raw_ns"], "name": name, "value": value}
              for process, records in streams.items() for r in records if r.get("r") == "gauge"
              for name, value in (r.get("values") or {}).items()]
    counts["gauges.csv"] = _write(target / "gauges.csv", ("process", "t_raw_ns", "name", "value"),
                                  gauges)
    samples = [s for s in sampler if s.get("r") == "sample"]
    gpu_keys = ("util_gpu_pct", "util_mem_pct", "mem_used_bytes", "mem_total_bytes",
                "clock_sm_mhz", "clock_mem_mhz", "power_mw", "temp_c", "pstate")
    host_keys = ("load1", "load5", "load15", "runnable_entities", "total_entities",
                 "cpu_busy_pct", "procs_running", "procs_blocked")
    counts["gpu.csv"] = _write(
        target / "gpu.csv", ("seq", "t_raw_ns", "t_wall_ns", "late_ns", "cost_ns", *gpu_keys,
                             *host_keys),
        ({"seq": s.get("seq"), "t_raw_ns": s["t_raw_ns"], "t_wall_ns": s.get("t_wall_ns"),
          "late_ns": s.get("late_ns"), "cost_ns": s.get("cost_ns"),
          **(s.get("gpu") or {}), **(s.get("host") or {})} for s in samples))
    counts["procs.csv"] = _write(
        target / "procs.csv", ("t_raw_ns", "role", "pid", "comm", "cpu_pct", "rss_bytes",
                               "threads", "vcsw", "nvcsw", "processor", "state"),
        ({"t_raw_ns": s["t_raw_ns"], **p} for s in samples for p in s.get("procs") or ()))
    counts["threads.csv"] = _write(
        target / "threads.csv", ("t_raw_ns", "role", "pid", "tid", "comm", "delta_ticks",
                                 "ns_tid"),
        ({"t_raw_ns": s["t_raw_ns"], "role": p["role"], "pid": p["pid"], "tid": t[0],
          "comm": t[1], "delta_ticks": t[2], "ns_tid": t[3]}
         for s in samples for p in s.get("procs") or () for t in p.get("thread_deltas") or ()))
    gpu_procs = []
    for s in samples:
        mem = s.get("gpu_proc_mem") or {}
        for kind in ("compute", "graphics"):
            for row in mem.get(kind) or ():
                gpu_procs.append({"t_raw_ns": s["t_raw_ns"], "kind": f"mem_{kind}",
                                  "pid": row["pid"], "used_bytes": row["used_bytes"]})
        for row in s.get("gpu_proc_util") or ():
            gpu_procs.append({"t_raw_ns": s["t_raw_ns"], "kind": "util", "pid": row["pid"],
                              "sm_pct": row["sm_pct"], "mem_pct": row["mem_pct"],
                              "sample_ts_us": row["ts_us"]})
    counts["gpu_procs.csv"] = _write(target / "gpu_procs.csv",
                                     ("t_raw_ns", "kind", "pid", "used_bytes", "sm_pct",
                                      "mem_pct", "sample_ts_us"), gpu_procs)
    ticket_rows = list(timing.get("tickets") or ())
    ticket_header = sorted({k for row in ticket_rows for k in row})
    counts["tickets.csv"] = _write(target / "tickets.csv", ticket_header, ticket_rows)
    roles = {}
    for s in samples:
        for p in s.get("procs") or ():
            roles.setdefault(p["role"], set()).add(p["pid"])
    if result is None:
        result = json.loads(sources["result"].read_text(encoding="utf-8"))
    else:
        sources = {k: v for k, v in sources.items() if k != "result"}   # written after export
    manifest = {
        "schema": SCHEMA, "output_root": str(output_root), "cell": cell.name,
        "claim_scope": result.get("claim_scope"),
        "verdict": result.get("verdict"), "gates": result.get("gates"),
        "registration": result.get("registration"),
        "clock_domain": "CLOCK_MONOTONIC_RAW (host-wide); capture_wall_ns is realtime",
        "clock_offsets": (analysis.get("clocks") or {}).get("per_process"),
        "single_raw_domain": (analysis.get("clocks") or {}).get("single_raw_domain"),
        "stage_registry": {name: {"parents": None if spec["parents"] is None
                                  else sorted(map(str, spec["parents"])),
                                  "scope": spec["scope"], "cuda": spec["cuda"],
                                  "kind": spec["kind"]} for name, spec in SV.STAGES.items()},
        "sampler_roles": {role: sorted(pids) for role, pids in roles.items()},
        "row_counts": counts,
        "source_sha256": {name: _sha(path) for name, path in sources.items()},
        "export_sha256": {name: _sha(target / name) for name in counts},
        "notes": ["cuda_ms is GPU execution on the current stream between two events; "
                  "side-stream work shows only as wall time",
                  "NVML utilization is the driver's sampled window, not instantaneous",
                  "profiler hot-path cost is inside every span it brackets and is reported "
                  "per thread in the analysis overhead section; it is not subtracted",
                  "MPS control daemon was pre-existing on the host; see the preflight"]}
    with (target / "MANIFEST.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    return manifest


def verify(target: Path) -> dict[str, Any]:
    """Re-hash an export against its manifest (for each independent analysis)."""
    manifest = json.loads((Path(target) / "MANIFEST.json").read_text(encoding="utf-8"))
    mismatched = [name for name, digest in manifest["export_sha256"].items()
                  if _sha(Path(target) / name) != digest]
    return {"files": len(manifest["export_sha256"]), "mismatched": mismatched,
            "verified": not mismatched}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--target", type=Path, default=None,
                        help="default: <output_root>/export_v1 (create-only)")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    target = args.target or (args.output_root / "export_v1")
    if args.verify_only:
        print(json.dumps(verify(target), indent=1))
        return 0
    manifest = export(args.output_root, target)
    print(json.dumps({"target": str(target), "row_counts": manifest["row_counts"]}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
