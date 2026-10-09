"""Fail-closed numerical-GT engineering postflight over observed evidence.

This supplements, never rewrites, the retained profiler's G1--G9 report.
The old GT-function coverage is not a requirement: the numerical producer and
scorer have separate actual worker ledgers and durable archive proofs.
"""
from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
import hashlib
import json

from .live_process_scope import verify_scope_files, WORKER_SCOPE_SCHEMA


SCHEMA = "splitfusion.numeric_gt.engineering_postflight.v1"
SCORED = "LIVE_SCORED_DURABLE"
UNSCORED = frozenset(("LIVE_UNSCORED_TIMEOUT_DURABLE", "LIVE_UNSCORED_FAULT_DURABLE"))
PROFILE_STAGES = ("ue.worker_frame", "ue.codec.total_ue_preparation",
                  "edge.compute", "edge.codec.frozen_tail")


class PostflightError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise PostflightError(message)


def _one(root: Path, name: str, *, edge=False) -> Path:
    paths = [path for path in root.rglob(name)
             if path.is_file() and not path.is_symlink()
             and path.resolve().is_relative_to(root)
             and (not edge or path.parts[-3:] == ("profiler", "edge", name))]
    _require(len(paths) == 1, "one exact " + name + " required")
    return paths[0]


def _read(path: Path, *, limit=16 * 1024 * 1024):
    _require(path.is_file() and not path.is_symlink()
             and path.stat().st_size <= limit, "bounded real evidence file required: " + str(path))
    return json.loads(path.read_bytes())


def verify_archive_readback(*, edge_scope_path, edge_scope):
    """Hash the evidence copy of every file named by actual durable receipts."""
    root = Path(edge_scope_path).parent / "numeric_runtime" / "archive"
    _require(root.is_dir() and not root.is_symlink(), "copied numeric archive root required")
    workers = {row.get("role"): row for row in edge_scope.get("workers", [])}
    completions = workers.get("archive", {}).get("close", {}).get("completions", [])
    _require(type(completions) is list and len(completions) <= 1000,
             "bounded actual archive completion rows required")
    verified, seen, errors = 0, set(), []
    for receipt in completions:
        written = receipt.get("written") if type(receipt) is dict else None
        _require(type(written) is list and len(written) <= 4096,
                 "bounded durable written inventory required")
        for item in written:
            relative = item.get("relative") if type(item) is dict else None
            path = PurePosixPath(relative) if type(relative) is str else None
            _require(path is not None and bool(path.parts) and not path.is_absolute()
                     and str(path) == relative and all(part not in (".", "..") for part in path.parts)
                     and item.get("root") == 0 and item.get("ready") is False
                     and relative not in seen,
                     "foreign, duplicate or mutable archive inventory path")
            seen.add(relative)
            target = root.joinpath(*path.parts)
            if (not target.is_file() or target.is_symlink()
                    or any(parent.is_symlink() for parent in target.parents if parent != root)
                    or not target.resolve().is_relative_to(root)):
                errors.append(dict(relative=relative, problem="MISSING_OR_SYMBOLIC"))
                continue
            digest = hashlib.sha256()
            size = 0
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if size != item.get("bytes") or digest.hexdigest() != item.get("sha256"):
                errors.append(dict(relative=relative, problem="BYTE_OR_HASH_DRIFT"))
            else:
                verified += 1
    return dict(root=str(root), files=len(seen), verified=verified, errors=errors,
                passed=bool(seen) and verified == len(seen) and not errors)


def evaluate_reports(*, ue_scope, edge_scope, profiler, timing, qualification,
                     process_scope, transmitted_budget):
    """Check actual workers, targets, archive receipts and retained profile evidence."""
    _require(type(transmitted_budget) is int and 1 <= transmitted_budget <= 1000,
             "bounded registered frame budget required")
    _require(ue_scope.get("schema") == WORKER_SCOPE_SCHEMA
             and edge_scope.get("schema") == WORKER_SCOPE_SCHEMA
             and len(ue_scope.get("workers", [])) == 1
             and len(edge_scope.get("workers", [])) == 2,
             "actual UE and edge worker scopes required")
    producer = ue_scope["workers"][0]
    edge_workers = {item.get("role"): item for item in edge_scope["workers"]}
    _require(producer.get("role") == "producer" and set(edge_workers) == {"scorer", "archive"},
             "exact producer/scorer/archive records required")
    p, s, a = producer["close"], edge_workers["scorer"]["close"], edge_workers["archive"]["close"]
    state = edge_scope.get("state") or {}
    inbox = state.get("inbox") or {}
    reports = state.get("evaluator_reports") or []
    events = p.get("events") or {}
    _require(type(events) is dict and len(events) <= 1000,
             "bounded actual producer event ledger required")
    published = {row.get("frame_id"): row for row in events.values()
                 if type(row) is dict and "ready_published_raw_ns" in row}
    outcomes = [row.get("outcome") for row in published.values()]
    scored = outcomes.count(SCORED)
    unscored = sum(outcome in UNSCORED for outcome in outcomes)
    archive_rows = [item for report in reports for item in report.get("archive_records", [])]
    scored_archives = sum(item.get("status") == "SCORED_DURABLE" for item in archive_rows)
    unscored_archives = sum(item.get("status") == "UNSCORED_DURABLE" for item in archive_rows)
    completed = a.get("completions") or []
    archive_frames = [item.get("identity", {}).get("frame_id") for item in completed]
    sent = (timing.get("counts") or {}).get("sent_frames")
    counts = profiler.get("counts") or {}
    validity = profiler.get("validity") or {}
    coverage = profiler.get("coverage") or {}
    semantics = profiler.get("semantics") or {}
    identity = profiler.get("identity") or {}
    sampler = profiler.get("sampler") or {}
    old_gates = qualification.get("gates") or {}
    checks = {
        "N1_RETAINED_FRAME_ACCOUNTING":
            sent in (transmitted_budget, transmitted_budget - 1)
            and counts.get("transmitted") == sent
            and old_gates.get("G1_FRAME_ACCOUNTING") is True,
        "N2_PRODUCER_DRAIN_AND_TARGET_COUNT":
            p.get("closed") is True and p.get("valid") is True and p.get("failure") is None
            and p.get("retained_jobs") == 0 and p.get("issued") == p.get("retired") == len(events)
            and len(published) > 0 and len(published) == scored + unscored
            and len({row.get("frame_id") for row in published.values()}) == len(published)
            and inbox.get("targets") == inbox.get("terminals") == len(published)
            and inbox.get("failure") is None and inbox.get("worker_alive") is False,
        "N3_SCORER_DRAIN":
            s.get("closed") is True and s.get("valid") is True and s.get("failure") is None
            and s.get("retained_jobs") == 0 and s.get("scored") == s.get("released") == scored,
        "N4_EXACT_DURABLE_ARCHIVES":
            a.get("failed") is False and a.get("fault") is None
            and a.get("outstanding_jobs") == 0 and a.get("submitted") == a.get("completed") == len(published)
            and len(completed) == len(published) and all(item.get("durable") is True
                and item.get("status") == "COMPLETE" and item.get("error") is None for item in completed)
            and set(archive_frames) == set(published) and len(archive_frames) == len(set(archive_frames))
            and scored_archives == scored and unscored_archives == unscored
            and len(archive_rows) == len(published)
            and type(reports) is list and bool(reports)
            and all(report.get("archive_worker_alive") is False
                    and report.get("failure") is None for report in reports),
        "N5_SOURCE_BOUND_PROCESS_SCOPE": process_scope.get("scope_ok") is True,
        "N6_REAL_PROFILER_COVERAGE":
            bool(validity) and all(row.get("passed") is True for row in validity.values())
            and all((coverage.get(name) or {}).get("records", 0) > 0 for name in PROFILE_STAGES)
            and sampler.get("present") is True and sampler.get("footer_present") is True
            and sampler.get("dropped_samples") == 0,
        "N7_IDENTITIES_ACTOR_AND_REWARD":
            old_gates.get("G2_ACTOR_HASH_UNCHANGED") is True
            and all(row.get("conflicts", 0) == 0 for row in identity.values())
            and all((semantics.get(name) or {}).get("passed") is True
                    for name in ("actor_replay", "reward", "receipt_marks")),
        "N8_CLEAN_EDGE_AND_HOST":
            state.get("installed") is True and not state.get("errors")
            and old_gates.get("G9_HOST_RESTORED_CLEAN") is True,
    }
    return dict(schema=SCHEMA, checks=checks, passed=all(checks.values()),
        verdict="NUMERIC_GT_ENGINEERING_POSTFLIGHT_PASS" if all(checks.values())
            else "NUMERIC_GT_ENGINEERING_POSTFLIGHT_FAIL",
        counts=dict(transmitted=sent, producer_issued=p.get("issued"), published=len(published),
                    scored=scored, unscored=unscored, scored_archives=scored_archives,
                    unscored_archives=unscored_archives, archive_completions=len(completed)),
        retained_gates_report_only=old_gates, old_gt_stage_coverage_claim=False,
        policy_performance_claim=False, live_qualified=False)


def evaluate_files(*, capture_root, cell_root, registration, baseline):
    """Read exact retained files and numerical scopes after teardown."""
    capture = Path(capture_root).resolve(strict=True)
    cell = Path(cell_root).resolve(strict=True)
    _require(cell.is_relative_to(capture), "cell belongs to another capture")
    scope = verify_scope_files(capture_root=capture, cell_root=cell, baseline=baseline,
        expected_sources=registration["numeric_worker_sources"])
    edge_path = _one(cell, "numeric_edge_scope.json", edge=True)
    edge = _read(edge_path)
    result = evaluate_reports(ue_scope=_read(_one(cell, "numeric_ue_scope.json")),
        edge_scope=edge,
        profiler=_read(capture / "PROFILER_ANALYSIS.json"),
        timing=_read(capture / "PHASE6_TIMING_CHARACTERIZATION.json"),
        qualification=_read(capture / "QUALIFICATION_RESULT.json"),
        process_scope=scope, transmitted_budget=registration["transmitted_budget"])
    readback = verify_archive_readback(edge_scope_path=edge_path, edge_scope=edge)
    result["checks"]["N9_HASHED_ARCHIVE_READBACK"] = readback["passed"]
    result["archive_readback"] = readback
    result["passed"] = all(result["checks"].values())
    result["verdict"] = ("NUMERIC_GT_ENGINEERING_POSTFLIGHT_PASS" if result["passed"]
                         else "NUMERIC_GT_ENGINEERING_POSTFLIGHT_FAIL")
    return result


__all__ = ["PostflightError", "evaluate_reports", "evaluate_files",
           "verify_archive_readback", "SCHEMA"]
