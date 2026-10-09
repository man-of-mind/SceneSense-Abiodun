"""verify_archive_v2 + paired-replay admission. Offline only; refuses rather than repairs.

``verify_archive_v2`` checks the externally pinned manifest digest, the pre-launch plan digest (and that
the plan is the registered rule applied to its own run-length inputs), the exact file inventory and
hashes, every retained tensor/wire, all plan/wire/GT identities and the status ledger. With
``strict=True`` (the default, and always for admission) any sampled ordinal that is not
``RETAINED_COMPLETE`` refuses the archive; only ``NOT_REACHED`` ordinals of a declared shorter run are
allowed. ``admit_for_paired_replay`` additionally requires a postflight GT-reference document bound to
this manifest and plan, a RESOLVED row with exact rescoring and all identity joins for every retained
record, and re-hashes every referenced numeric-GT archive file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .plan import INPUT_BYTES, INPUT_SHAPE, MAX_WIRE_BYTES, PlanError, canonical, load_plan, validate_plan
from . import observer as OB
from . import writer as WR

ALLOWED_FINAL = ("RETAINED_COMPLETE", "BUDGET_EXHAUSTED", "NOT_REACHED")


class RetentionAdmissionError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise RetentionAdmissionError(message)


def _safe_dir(directory: Path) -> Path:
    directory = Path(directory).absolute()
    _require(not any(p.is_symlink() for p in (directory, *directory.parents)), "unsafe (symlinked) archive path")
    _require(directory.is_dir(), "archive directory missing")
    return directory


def _status_ok(status: str) -> bool:
    return status in ALLOWED_FINAL or status.startswith("INCOMPLETE_") or status.startswith("FAULT_")


def verify_archive_v2(directory: Path, *, expected_manifest_sha256: str, expected_plan_sha256: str,
                      plan_path: Optional[Path] = None, strict: bool = True,
                      check_sources: bool = True) -> dict:
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    directory = _safe_dir(directory)
    path = directory / "MANIFEST.json"
    _require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 8 * 1024 * 1024,
             "missing or unsafe manifest")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, "unbound or modified retention manifest")
    manifest = json.loads(raw)
    _require(manifest.get("schema") == OB.SCHEMA and manifest.get("complete") is True, "foreign/unfinished archive")
    # ---- plan binding
    _require(manifest.get("plan_sha256") == expected_plan_sha256, "manifest sample plan digest differs from pre-launch plan")
    try:
        plan = validate_plan(manifest["plan"])
        if plan_path is not None:
            load_plan(plan_path, expected_plan_sha256)
    except PlanError as exc:
        raise RetentionAdmissionError(f"invalid sample plan: {exc}") from exc
    _require(hashlib.sha256(canonical(plan)).hexdigest() == expected_plan_sha256, "embedded plan differs from its digest")
    if check_sources:
        _require(manifest.get("sources") == OB.package_source_digests(), "retention sources drifted since capture")
    # ---- status ledger
    ordinals = plan["ordinals"]
    statuses = manifest.get("ordinal_status", {})
    _require(set(statuses) == {str(o) for o in ordinals}, "status ledger does not cover exactly the planned ordinals")
    _require(all(isinstance(s, str) and _status_ok(s) for s in statuses.values()), "unknown retention status")
    counts: dict = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    _require(manifest.get("status_counts") == counts, "status counts differ from the ledger")
    observed = manifest.get("observed_policy_decisions")
    _require(type(observed) is int and observed >= 0, "observed decision count missing")
    planned = plan["planned_policy_decisions"]
    _require(manifest.get("run_length") == ("SHORTER_THAN_PLANNED" if observed < planned else
             "LONGER_THAN_PLANNED" if observed > planned else "EQUAL_TO_PLANNED"), "run-length outcome differs")
    for o in ordinals:
        _require((statuses[str(o)] == "NOT_REACHED") == (o >= observed), "NOT_REACHED inconsistent with run length")
    counters = manifest.get("counters", {})
    _require(counters.get("policy_decisions_observed") == observed, "observed counter differs")
    faulted_sampled = sum(statuses[str(o)].startswith("FAULT_") for o in ordinals if o < observed)
    _require(counters.get("sampled") == sum(o < observed for o in ordinals) - faulted_sampled,
             "sampled counter inconsistent with the ledger")
    _require(counters.get("skipped_off_sample", 0) + counters.get("skipped_beyond_plan", 0)
             + counters.get("sampled", 0) + faulted_sampled + counters.get("faulted_unsampled", 0) == observed,
             "decision accounting does not sum to the observed policy decisions")
    # ---- inventory and hashes
    hashes = manifest.get("sha256", {})
    _require(set(p.name for p in directory.iterdir()) == set(hashes) | {"MANIFEST.json"}, "archive inventory mismatch")
    records = manifest.get("records", [])
    _require({e["ordinal"] for e in records} == {o for o in ordinals if statuses[str(o)] == "RETAINED_COMPLETE"}
             and len(records) == counts.get("RETAINED_COMPLETE", 0), "retained records differ from the ledger")
    _require(len(records) <= plan["bounds"]["max_records"], "retained count exceeds plan cap")
    referenced, keys, storage = set(), set(), 0
    for entry in records:
        npy, sfd4 = entry["input_file"], entry["wire_file"]
        _require(npy in hashes and sfd4 in hashes and npy != sfd4 and not ({npy, sfd4} & referenced),
                 "unbound record members")
        referenced.update((npy, sfd4))
        for name in (npy, sfd4):
            member = directory / name
            _require(Path(name).name == name and not member.is_symlink() and member.is_file()
                     and member.stat().st_size <= INPUT_BYTES + 4096
                     and WR.sha256_file(member) == hashes[name], "missing, unsafe or modified record member")
        _require(entry["input_sha256"] == hashes[npy] and entry["wire_sha256"] == hashes[sfd4], "record digest differs")
        _require(INPUT_BYTES < (directory / npy).stat().st_size <= INPUT_BYTES + 4096, "invalid NPY size")
        array = np.load(directory / npy, allow_pickle=False, mmap_mode="r")
        _require(array.shape == INPUT_SHAPE and array.dtype == np.float32 and bool(np.isfinite(array).all()),
                 "invalid exact model input")
        data = (directory / sfd4).read_bytes()
        _require(0 < len(data) <= MAX_WIRE_BYTES and len(data) == entry["wire_bytes"], "invalid wire size")
        envelope, context = W.unpack_sfd4(data)
        identity = MP.run4_identity(envelope)
        MP.validate_run4_identity(identity)
        key = entry["join_key"]
        _require(identity == entry["identity"] and envelope.reward_requested
                 and identity["frame_kind"] == "POLICY_DECISION", "wire/action identity mismatch")
        _require(key == {"session_uuid": envelope.session_uuid, "tensor_seq": envelope.tensor_seq,
                         "frame_id": envelope.frame_id, "capture_timestamp_ns": envelope.capture_timestamp_ns,
                         "decision_ordinal": entry["ordinal"]}, "join key mismatch")
        _require(entry["action"] == {"mode_id": envelope.mode_id, "q_e4": envelope.q_e4,
                 "anchor_action_id": envelope.anchor_action_id, "kind": identity["frame_kind"],
                 "execution_bundle_sha256": envelope.execution_bundle_sha256}, "action identity mismatch")
        _require(entry["frame_context"] == WR.context_document(context), "wire frame context mismatch")
        try:
            WR.check_gt_context(entry["gt_identity"], envelope, context)
        except WR.RetentionWriteError as exc:
            raise RetentionAdmissionError("GT/wire identity mismatch") from exc
        _require(not plan["require_cuda_transfer"] or entry["transfer_device"] in ("cuda", "cuda:0"),
                 "not the original CUDA inference path")
        jk = (key["session_uuid"], key["tensor_seq"], key["frame_id"])
        _require(jk not in keys, "duplicate retained identity")
        keys.add(jk)
        storage += entry["input_file_bytes"] + entry["wire_bytes"]
    _require(referenced == set(hashes), "unreferenced archive data")
    _require(storage == manifest.get("storage_bytes") and storage <= plan["bounds"]["max_storage_bytes"],
             "storage accounting mismatch or cap exceeded")
    host = manifest.get("host_bytes", {})
    _require(host.get("cap") == plan["bounds"]["max_host_bytes"] and 0 <= host.get("peak", -1) <= host["cap"],
             "host-memory bound violated")
    _require(manifest.get("drain") and all(v.get("alive") is False for v in manifest["drain"].values()),
             "producer drain not proven")
    if strict:
        bad = {s: n for s, n in counts.items() if s not in ("RETAINED_COMPLETE", "NOT_REACHED")}
        _require(not bad, f"incomplete retention refused: {bad}")
        _require(counts.get("RETAINED_COMPLETE", 0) >= 1, "no retained records")
    return manifest


def admit_for_paired_replay(directory: Path, *, expected_manifest_sha256: str, expected_plan_sha256: str,
                            references_path: Path, expected_references_sha256: str,
                            plan_path: Optional[Path] = None, rehash_gt: bool = True) -> list[dict]:
    """Return one admitted row per retained record, or refuse the whole cohort."""
    from .gt_adapter import TARGET_HASH_DOMAIN, REFERENCES_SCHEMA, REFERENCED_FILES
    manifest = verify_archive_v2(directory, expected_manifest_sha256=expected_manifest_sha256,
                                 expected_plan_sha256=expected_plan_sha256, plan_path=plan_path, strict=True)
    references_path = Path(references_path)
    _require(not references_path.is_symlink() and references_path.is_file(), "missing GT references")
    raw = references_path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == expected_references_sha256, "unbound or modified GT references")
    refs = json.loads(raw)
    _require(refs.get("schema") == REFERENCES_SCHEMA and refs.get("target_hash_domain") == TARGET_HASH_DOMAIN,
             "foreign GT reference document or undeclared target hash domain")
    _require(refs.get("retention_manifest_sha256") == expected_manifest_sha256
             and refs.get("plan_sha256") == expected_plan_sha256, "GT references bound to another manifest/plan")
    rows = {r["ordinal"]: r for r in refs.get("rows", [])}
    _require(set(rows) == {e["ordinal"] for e in manifest["records"]}, "GT references do not cover the retained cohort")
    archive_root = Path(refs["archive_root"])
    admitted = []
    for entry in manifest["records"]:
        row = rows[entry["ordinal"]]
        _require(row.get("status") == "RESOLVED", f"ordinal {entry['ordinal']}: GT {row.get('status')}")
        _require(row.get("target_hash_domain") == TARGET_HASH_DOMAIN
                 and row.get("edge_target_bundle_sha256_consulted") is False, "silent target-hash substitution")
        _require(row.get("joins") and all(row["joins"].values()), f"ordinal {entry['ordinal']}: identity join failed")
        _require(row.get("rescore", {}).get("exact") is True, f"ordinal {entry['ordinal']}: rescore not exact")
        _require(set(REFERENCED_FILES) <= set(row["files"]), "GT reference inventory incomplete")
        _require(row["frame_id"] == entry["join_key"]["frame_id"], "GT reference frame differs")
        if rehash_gt:
            frame_dir = archive_root / row["frame_dir"]
            for name, digest in row["files"].items():
                member = frame_dir / name
                _require(not member.is_symlink() and member.is_file() and WR.sha256_file(member) == digest,
                         f"ordinal {entry['ordinal']}: referenced GT file missing or modified: {name}")
        admitted.append({"ordinal": entry["ordinal"], "join_key": entry["join_key"], "action": entry["action"],
                         "input_file": str(Path(directory) / entry["input_file"]),
                         "wire_file": str(Path(directory) / entry["wire_file"]),
                         "gt_frame_dir": str(archive_root / row["frame_dir"]), "gt_files": row["files"],
                         "prediction": row["prediction"], "q_perc": row["q_perc"]})
    return admitted
