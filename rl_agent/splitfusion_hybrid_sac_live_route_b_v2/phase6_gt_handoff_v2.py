#!/usr/bin/env python3
"""Addendum 6: exact GT handoff diagnostics and the one-decision handshake verdict.

* :class:`GtWriteRecorderV2` wraps the UE-side GT writers (process-local) and
  appends one JSON line per written component: host path, identity, write
  time, size, SHA-256.
* :func:`preserve_directory` copies the edge GT scratch directory create-only
  into the attempt directory and verifies a SHA-256 manifest before the
  temporary source may be deleted.
* :func:`handoff_report` reconciles, per reward ticket and per component
  (``objects.json``, ``semantic.npy``, ``semantic.json``), the UE host write,
  the preserved copy, the edge container view (first observed, digest at read,
  exact read errors) and the actual bind mounts. It never collapses failures
  into "GT unavailable".
* :func:`handshake_verdict` applies the preregistered one-decision criteria.

Importing this module performs no I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

COMPONENTS = ("objects.json", "semantic.npy", "semantic.json")
CONTAINER_EVIDENCE_ROOT = "/work/torch_cache"
EVALUATOR_WAIT_BOUND_MS = 250.0
GT_LOCAL_EXPIRY_S = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class GtWriteRecorderV2:
    """Append-only JSONL record of every UE-side GT component write."""

    def __init__(self, path: Path, *, ticket_log: Any = None) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.ticket_log = ticket_log          # addendum 7: object-write timing

    def _append(self, row: Mapping[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def record(self, kind: str, paths: Sequence[Path], identity: Mapping[str, Any]) -> None:
        written = time.time_ns()
        for path in paths:
            path = Path(path)
            entry = {"kind": kind, "host_path": str(path), "name": path.name,
                     "written_wall_ns": written, "identity": dict(identity)}
            if path.is_file():
                entry.update(size_bytes=path.stat().st_size, sha256=_sha256(path))
            else:
                entry.update(size_bytes=None, sha256=None, error="MISSING_AFTER_WRITE")
            self._append(entry)

    def wrap(self, write_objects: Callable[..., Any], write_semantic: Callable[..., Any]):
        def objects(directory, *, identity, **kwargs):
            frame_id = int(identity["frame_id"])
            if self.ticket_log is not None:
                self.ticket_log.write_started(frame_id, time.time_ns())
            try:
                result = write_objects(directory, identity=identity, **kwargs)
            except Exception as exc:
                self._append({"kind": "objects", "error": f"{type(exc).__name__}: {exc}"[:300],
                              "identity": dict(identity), "written_wall_ns": time.time_ns()})
                raise
            self.record("objects", [Path(result)], identity)
            if self.ticket_log is not None:
                path = Path(result)
                self.ticket_log.write_finished(
                    frame_id, time.time_ns(), object_count=len(kwargs.get("rows") or ()),
                    size_bytes=path.stat().st_size if path.is_file() else None,
                    sha256=_sha256(path) if path.is_file() else None, identity=identity)
            return result

        def semantic(directory, *, identity, **kwargs):
            try:
                result = write_semantic(directory, identity=identity, **kwargs)
            except Exception as exc:
                self._append({"kind": "semantic", "error": f"{type(exc).__name__}: {exc}"[:300],
                              "identity": dict(identity), "written_wall_ns": time.time_ns()})
                raise
            paths = list(result) if isinstance(result, (tuple, list)) else [Path(result)]
            self.record("semantic", paths, identity)
            return result
        return objects, semantic


def preserve_directory(source: Path, destination: Path) -> dict[str, Any]:
    """Create-only copy of ``source`` with a verified SHA-256 manifest."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"preservation target exists: {destination}")
    manifest: dict[str, Any] = {"schema": "scenesense.run4_live_v2.gt_scratch_manifest.v1",
                                "source": str(source), "destination": str(destination),
                                "source_present": source.is_dir(), "files": {}}
    destination.mkdir(parents=True)
    if source.is_dir():
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest["files"][str(relative)] = {"sha256": _sha256(path),
                                                "size_bytes": path.stat().st_size}
    manifest["verified"] = all(
        _sha256(destination / name) == entry["sha256"]
        for name, entry in manifest["files"].items())
    manifest["preserved_wall_ns"] = time.time_ns()
    with (destination.parent / (destination.name + ".manifest.json")).open(
            "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    return manifest


def _load(path: Path, default: Any = None) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else default


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def handoff_report(cell: Path) -> dict[str, Any]:
    cell = Path(cell)
    evidence = cell / "run4_phase6"
    edge = _load(evidence / "edge_report.json", {}) or {}
    launch = _load(evidence / "edge_image_launch.json", {}) or {}
    manifest = _load(evidence / "gt_scratch_preserved.manifest.json", {}) or {}
    writes = _jsonl(evidence / "gt_handoff_ue.jsonl")
    mounts = ((launch.get("post_create_container") or {}).get("mounts") or {})
    state_source = (mounts.get("state") or {}).get("source")
    tickets = []
    for record in edge.get("evaluations") or ():
        timing = record.get("timing") or {}
        handoff = timing.get("gt_handoff") or {}
        stem = handoff.get("stem")
        components = {}
        for name in COMPONENTS:
            container = (handoff.get("components") or {}).get(name) or {}
            relative = f"segmentation_evidence/{stem}.{name}" if stem else None
            host_expected = (None if not (state_source and relative)
                             else str(Path(state_source) / relative))
            write = next((w for w in writes if w.get("name") == f"{stem}.{name}"), None)
            preserved = (manifest.get("files") or {}).get(f"{stem}.{name}") if stem else None
            components[name] = {
                "container_path": container.get("container_path"),
                "host_path_expected": host_expected,
                "host_write": write,
                "preserved_copy": preserved,
                "container_first_observed_wall_ns": container.get("first_observed_wall_ns"),
                "container_sha256_at_read": container.get("sha256_at_read"),
                "container_size_bytes": container.get("size_bytes"),
                "host_container_match": bool(
                    write and write.get("sha256") and container.get("sha256_at_read")
                    and write["sha256"] == container["sha256_at_read"]),
                "preserved_match": bool(write and preserved and write.get("sha256")
                                        == preserved.get("sha256")),
            }
        tickets.append({
            "frame_id": record.get("frame_id"), "kind": record.get("kind"),
            "reason": record.get("reason"), "stem": stem,
            "identity": handoff.get("identity"),
            "enqueued_wall_ns": timing.get("enqueued_wall_ns"),
            "gt_ready_detected_wall_ns": timing.get("gt_ready_detected_wall_ns"),
            "read_success_wall_ns": handoff.get("read_success_wall_ns"),
            "evaluator_start_wall_ns": timing.get("evaluator_start_wall_ns"),
            "evaluator_end_wall_ns": timing.get("evaluator_end_wall_ns"),
            "emit_wall_ns": record.get("emit_wall_ns"),
            "gt_expired_wall_ns": timing.get("gt_expired_wall_ns"),
            "read_errors": handoff.get("read_errors"),
            "gt_bundle_sha256": timing.get("gt_bundle_sha256"),
            "components": components,
        })
    return {"schema": "scenesense.run4_live_v2.gt_handoff_report.v1",
            "bind_mounts": mounts, "evidence_container_root":
                f"{CONTAINER_EVIDENCE_ROOT}/segmentation_evidence",
            "scratch_manifest": {k: manifest.get(k) for k in (
                "source", "destination", "source_present", "verified")}
                | {"file_count": len(manifest.get("files") or {})},
            "ue_writes": len(writes), "ue_write_errors": [w for w in writes if w.get("error")],
            "tickets": tickets}


def handshake_verdict(cell: Path, *, actor_audit_before: Mapping[str, Any],
                      actor_audit_after: Mapping[str, Any]) -> dict[str, Any]:
    cell = Path(cell)
    report = handoff_report(cell)
    ue = _load(cell / "run4_phase6" / "PHASE6_UE_EVIDENCE.json", {}) or {}
    result = _load(cell / "CELL_RESULT.json", {}) or {}
    cleanup = result.get("cleanup") or {}
    tickets = report["tickets"]
    ticket = tickets[0] if len(tickets) == 1 else None
    rows = ue.get("feedback_rows") or []
    frames = ue.get("frames") or []
    decisions = {}
    for frame in frames:
        decisions.setdefault((frame["session_uuid"], frame["ticket_seq"]), []).append(frame)
    comps = (ticket or {}).get("components") or {}
    wait_ms = None
    if ticket and ticket.get("gt_ready_detected_wall_ns") and ticket.get(
            "evaluator_start_wall_ns"):
        wait_ms = (ticket["evaluator_start_wall_ns"] - ticket["gt_ready_detected_wall_ns"]) / 1e6
    read_elapsed_s = None
    if ticket and ticket.get("read_success_wall_ns") and ticket.get("enqueued_wall_ns"):
        read_elapsed_s = (ticket["read_success_wall_ns"] - ticket["enqueued_wall_ns"]) / 1e9
    radio = cleanup.get("radio") or {}
    checks = {
        "exactly_one_reward_ticket": ticket is not None,
        "all_three_components_host_and_container_match": bool(comps) and all(
            c["host_container_match"] for c in comps.values()),
        "scratch_preserved_and_verified": bool(report["scratch_manifest"].get("verified"))
            and all(c["preserved_match"] for c in comps.values()),
        "read_before_local_gt_expiry": read_elapsed_s is not None
            and read_elapsed_s < GT_LOCAL_EXPIRY_S,
        "gt_ready_and_evaluator_start_recorded": wait_ms is not None,
        "evaluator_wait_within_250ms": wait_ms is not None
            and wait_ms <= EVALUATOR_WAIT_BOUND_MS,
        "feedback_accepted_over_downlink": len(rows) == 1
            and rows[0].get("class") == "ACCEPTED"
            and rows[0].get("reason") != "GROUND_TRUTH_UNAVAILABLE"
            and bool(((result.get("phase6") or {}).get("gates") or {}).get(
                "P7_FEEDBACK_OVER_DOWNLINK")),
        "decision_two_tensors_one_reward_request": len(decisions) == 1 and all(
            len(v) >= 2 and sum(1 for f in v if f["reward_requested"]) == 1
            for v in decisions.values()),
        "zero_unresolved_tickets": ue.get("unresolved_tickets_at_close") == 0
            and len(ue.get("resolutions") or []) == len(decisions),
        "actor_hashes_unchanged": (actor_audit_before.get("verdict") == "PASS"
                                   and actor_audit_after.get("verdict") == "PASS"
                                   and actor_audit_before.get("weights_file_sha256")
                                   == actor_audit_after.get("weights_file_sha256")),
        "cold_host_and_channel_restored": bool(cleanup.get("all_gates_passed"))
            and bool(radio.get("noise_power_db_restored_and_read_back")),
    }
    missing = [name for name, c in comps.items() if not c.get("host_write")]
    return {"schema": "scenesense.run4_live_v2.phase6_gt_handshake.v1",
            "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "evaluator_wait_ms": wait_ms,
            "read_elapsed_s": read_elapsed_s,
            "components_never_written_on_host": missing,
            "handoff": report}


def _csv(path: Path) -> list[dict[str, Any]]:
    import csv

    if not Path(path).is_file():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def handshake_verdict_v2(cell: Path, *, actor_audit_before: Mapping[str, Any],
                         actor_audit_after: Mapping[str, Any]) -> dict[str, Any]:
    """Addendum-7 prospective one-decision handshake criteria (not yet executed)."""
    cell = Path(cell)
    evidence = cell / "run4_phase6"
    base = handshake_verdict(cell, actor_audit_before=actor_audit_before,
                             actor_audit_after=actor_audit_after)
    ue = _load(evidence / "PHASE6_UE_EVIDENCE.json", {}) or {}
    edge = _load(evidence / "edge_report.json", {}) or {}
    result = _load(cell / "CELL_RESULT.json", {}) or {}
    edge_warm = _load(evidence / "gt_scratch_preserved" / "run4_phase6_prewarm_edge.json", {}) or {}
    reward_frames = [f for f in ue.get("frames") or () if f.get("reward_requested")]
    reward_frame = reward_frames[0]["frame_id"] if len(reward_frames) == 1 else None
    terminals = _csv(cell / "map_feedback.csv")
    superseded = any(int(r["frame_id"]) == reward_frame and r.get("outcome") == "SUPERSEDED_PENDING"
                     for r in terminals) if reward_frame is not None else False
    evaluation = next((r for r in edge.get("evaluations") or ()
                       if r.get("frame_id") == reward_frame), None)
    feedback = next((r for r in ue.get("feedback_rows") or ()
                     if r.get("frame_id") == reward_frame), None)
    gt_ticket = next((t for t in (ue.get("gt_objects") or {}).get("tickets") or ()
                      if t.get("frame_id") == reward_frame), None)
    resolutions = ue.get("resolutions") or []
    resolution = resolutions[0] if len(resolutions) == 1 else {}
    refresh = ue.get("gt_refresh_startup") or {}
    first_capture = min((int(d["capture"]["ns"]) for d in ue.get("decisions") or ()
                         if isinstance(d.get("capture"), Mapping)), default=None)
    classes = [r.get("class") for c in ue.get("feedback_ledgers") or () for r in c]
    warmed = (bool(ue.get("prewarm_ue_completed")) and bool(edge_warm.get("completed"))
              and edge_warm.get("modes_warmed") == list(range(12)))
    latency = resolution.get("latency_ms")
    checks = {
        "warmup_completed_before_admission": warmed,
        "refresh_static_completed_before_admission": bool(refresh.get("completed"))
            and first_capture is not None and int(refresh["end_wall_ns"]) < first_capture,
        "one_rewarded_decision_plus_hold": base["checks"]["decision_two_tensors_one_reward_request"],
        "reward_frame_reached_inference_and_evaluator": evaluation is not None,
        "reward_frame_not_superseded": reward_frame is not None and not superseded,
        "gt_components_identity_matched_and_readable": (
            base["checks"]["all_three_components_host_and_container_match"]
            and base["checks"]["read_before_local_gt_expiry"]),
        "object_gt_queue_wait_and_compute_reported": bool(gt_ticket)
            and gt_ticket.get("queue_class") == "HIGH"
            and gt_ticket.get("queue_wait_ms") is not None
            and gt_ticket.get("object_rows_ms") is not None,
        "exact_q_perc_through_ue_feedback": bool(feedback and evaluation)
            and feedback.get("class") == "ACCEPTED" and evaluation.get("kind") == "DELIVERED_SUCCESS"
            and feedback.get("q_perc") == evaluation.get("q_perc")
            and resolution.get("q_perc") == evaluation.get("q_perc"),
        "action_open_to_feedback_within_170ms": latency is not None and float(latency) <= 170.0,
        "zero_unresolved_orphans_conflicts_faults": (
            ue.get("unresolved_tickets_at_close") == 0 and "UNKNOWN_ORPHAN" not in classes
            and ue.get("faulted") is None
            and bool(((result.get("phase6") or {}).get("gates") or {}).get(
                "P8_NO_INFRASTRUCTURE_FAULT"))),
        "no_missing_high_gt_output": not ue.get("gt_missing_high_outputs"),
        "actor_hashes_unchanged": base["checks"]["actor_hashes_unchanged"],
        "cold_host_and_channel_restored": base["checks"]["cold_host_and_channel_restored"],
    }
    if warmed and superseded:
        classification = "REWARD_FRAME_PROTECTION_REQUIRED"
    elif all(checks.values()):
        classification = "PASS"
    else:
        classification = "FAIL"
    excess = None
    if latency is not None and float(latency) > 170.0:
        decision = next((d for d in ue.get("decisions") or ()
                         if d.get("frame_id") == reward_frame), {})
        stages = decision.get("stages") or {}
        opened = (decision.get("action_open") or {}).get("ns")
        timing = (evaluation or {}).get("timing") or {}
        ingest = next((r for r in _csv(cell / "direct_edge_map" / "direct_map_ingest.csv")
                       if int(r["frame_id"]) == reward_frame), {})

        def span(a, b):
            return None if a is None or b is None else (float(b) - float(a)) / 1e6

        excess = {
            "excess_ms": float(latency) - 170.0,
            "ue_raw_clock_ms": {
                "action_open_to_7ch": span(opened, stages.get("input_7ch_start_raw_ns")),
                "7ch_to_front_end": span(stages.get("input_7ch_start_raw_ns"),
                                         stages.get("front_end_raw_ns")),
                "front_end_to_first_send": span(stages.get("front_end_raw_ns"),
                                                stages.get("first_packet_send_raw_ns"))},
            "edge_wall_clock_ms": {
                "reassembly_to_compute_start": span(
                    float(ingest["edge_reassembly_complete_wall_s"]) * 1e9
                    if ingest.get("edge_reassembly_complete_wall_s") else None,
                    float(ingest["edge_compute_start_wall_s"]) * 1e9
                    if ingest.get("edge_compute_start_wall_s") else None),
                "evaluator_enqueue_to_gt_ready": span(timing.get("enqueued_wall_ns"),
                                                      timing.get("gt_ready_detected_wall_ns")),
                "gt_ready_to_evaluator_start": span(timing.get("gt_ready_detected_wall_ns"),
                                                    timing.get("evaluator_start_wall_ns")),
                "evaluator_compute": span(timing.get("evaluator_start_wall_ns"),
                                          timing.get("evaluator_end_wall_ns"))},
            "note": "UE stages use CLOCK_MONOTONIC_RAW and edge stages wall time; they are "
                    "never subtracted across domains."}
    return {"schema": "scenesense.run4_live_v2.phase6_gt_handshake.v2",
            "classification": classification, "checks": checks,
            "reward_frame": reward_frame, "latency_ms": latency,
            "stage_excess": excess, "object_gt_ticket": gt_ticket,
            "handoff": base["handoff"]}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--actor-audit-before", type=Path, required=True)
    parser.add_argument("--actor-audit-after", type=Path, required=True)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    verdict = handshake_verdict_v2(args.cell,
                                   actor_audit_before=_load(args.actor_audit_before),
                                   actor_audit_after=_load(args.actor_audit_after))
    text = json.dumps(verdict, indent=1, sort_keys=True, default=str) + "\n"
    if args.write:
        with args.write.open("x", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0 if verdict["classification"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
