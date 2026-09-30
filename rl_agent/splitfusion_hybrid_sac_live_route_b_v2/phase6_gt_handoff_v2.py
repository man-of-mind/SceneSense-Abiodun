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

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

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
            try:
                result = write_objects(directory, identity=identity, **kwargs)
            except Exception as exc:
                self._append({"kind": "objects", "error": f"{type(exc).__name__}: {exc}"[:300],
                              "identity": dict(identity), "written_wall_ns": time.time_ns()})
                raise
            self.record("objects", [Path(result)], identity)
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("--actor-audit-before", type=Path, required=True)
    parser.add_argument("--actor-audit-after", type=Path, required=True)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    verdict = handshake_verdict(args.cell, actor_audit_before=_load(args.actor_audit_before),
                                actor_audit_after=_load(args.actor_audit_after))
    text = json.dumps(verdict, indent=1, sort_keys=True, default=str) + "\n"
    if args.write:
        with args.write.open("x", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
