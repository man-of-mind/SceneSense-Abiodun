"""Numeric-GT service archive -> exact rescoring inputs, retained-evaluator inputs and GT references.

Each live archive frame ``numeric_gt/service/archive/frame_NNNN/`` holds the scorer's exact admitted
target (``target_metadata.json`` + ``segmentation.u8`` + ``object_ignore.u8``), the native predicted
labels (``predicted_labels.u8``) and ``numerical_score.json``, which carries the complete native
prediction records (``prediction_records``), the prediction identity and digest
(sha256(labels || canonical(records))), the measurement, Q_perc, matches and confusion matrix.
Per-object predictions ARE archived, so exact rescoring needs nothing beyond the archive.

Two independent reproductions are checked before any use (no tolerances):
  1. the unchanged numeric scorer (``ready._admit`` -> ``scoring.admit_prediction`` ->
     ``join_admitted_inputs`` -> ``NumericalScorer.score`` -> registered quality formula ->
     ``live_archive._scored_values``) must rebuild ``numerical_score.json`` BYTE-FOR-BYTE;
  2. the retained evaluator ``splitfusion_live_parity_repair_v1.evaluation.evaluate_prepared``, fed
     a PreparedEvaluationTarget adapted from the same archive bytes, must return the identical
     measurement.

GT is referenced (paths + SHA-256), never copied or rebuilt. The GT is bound by the numeric ready
target (hash domain ``NUMERIC_GT_READY_TARGET_V2``); the edge ``timing.target_bundle_sha256`` field is
NOT consulted (in v11 it carried the scoring-packet digest; see ``edge_target_hash_crosscheck``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

MAIN = Path(__file__).resolve().parents[2]
TARGET_HASH_DOMAIN = "NUMERIC_GT_READY_TARGET_V2"
REFERENCES_SCHEMA = "splitfusion.paired_retention.gt_references.v2"
MAX_FILE_BYTES = 16 * 1024 * 1024
REFERENCED_FILES = ("scorer/target_metadata.json", "scorer/segmentation.u8", "scorer/object_ignore.u8",
                    "scorer/numerical_score.json", "scorer/predicted_labels.u8", "producer/numerical_avo.json")
OPTIONAL_FILES = ("scorer/anchor_provenance.json",)
IDENTITY_KEYS = ("capture_identity", "action_identity", "plan_sha256", "run_binding", "producer_binding",
                 "frame_context", "prediction_contract")
# Every module on the rescoring path; a live run's worktree copies must match these before use.
SCORER_MODULES = (
    "rl_agent/splitfusion_numeric_gt_integration_v1/contract.py",
    "rl_agent/splitfusion_numeric_gt_integration_v2/ready.py",
    "rl_agent/splitfusion_numeric_gt_integration_v2/scoring.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/scorer_core.py",
    "rl_agent/splitfusion_exact_gt_service_proto_v2/deferred.py",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py",
    "rl_agent/splitfusion_minimal_quality_scorer_v1/scorer.py",
    "rl_agent/splitfusion_minimal_quality_scorer_v1/shared_admission.py",
    "rl_agent/splitfusion_numeric_gt_runtime_v1/live_archive.py",
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/quality.py",
    "rl_agent/splitfusion_live_parity_repair_v2/prediction_archive.py",
    "rl_agent/splitfusion_live_parity_repair_v1/evaluation.py",
    "rl_agent/splitfusion_live_target_parity_v1/adapter.py",
)
EDGE_TARGET_HASH_DOMAINS = {
    "V11_SCORING_PACKET_SHA256": "edge timing.target_bundle_sha256 is the scoring-packet digest (v11); "
                                 "it must equal timing.scoring_packet_sha256 and is not a target-bundle digest",
    "COMPLETE_TARGET_BUNDLE_MANIFEST_SHA256": "timing.target_bundle_sha256 is the parity_complete_targets "
                                              "bundle manifest digest",
}


class GtAdapterError(ValueError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise GtAdapterError("GT_FILE_MISSING_OR_UNSAFE", str(path))
    return path.read_bytes()


def scorer_source_digests(root: Path = MAIN) -> dict[str, str]:
    return {m: _sha((Path(root) / m).read_bytes()) for m in SCORER_MODULES}


def edge_target_hash_crosscheck(timing: dict, *, declared_domain: str,
                                complete_bundle_manifest_sha256: Optional[str] = None) -> str:
    """Explicit, pinned interpretation of an edge ``timing.target_bundle_sha256``; never silent."""
    if declared_domain not in EDGE_TARGET_HASH_DOMAINS:
        raise GtAdapterError("UNDECLARED_EDGE_TARGET_HASH_DOMAIN", str(declared_domain))
    value = timing.get("target_bundle_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise GtAdapterError("EDGE_TARGET_HASH_MISSING")
    if declared_domain == "V11_SCORING_PACKET_SHA256":
        if value != timing.get("scoring_packet_sha256"):
            raise GtAdapterError("EDGE_TARGET_HASH_NOT_SCORING_PACKET")
        if complete_bundle_manifest_sha256 is not None and value == complete_bundle_manifest_sha256:
            raise GtAdapterError("EDGE_TARGET_HASH_DOMAIN_AMBIGUOUS")
    elif value != complete_bundle_manifest_sha256:
        raise GtAdapterError("EDGE_TARGET_HASH_NOT_COMPLETE_BUNDLE")
    return declared_domain


@dataclass(frozen=True)
class ArchivedFrame:
    frame_dir: Path
    raw: dict
    metadata: dict
    score: dict
    target: Any
    target_doc: dict
    anchor_join: Optional[dict]
    received_identity: dict      # prediction identity as the scorer received it (edge/UE identity)


def frame_directory(archive_root: Path, frame_id: int) -> Path:
    names = {f"frame_{int(frame_id):04d}", f"frame_{int(frame_id)}"}
    found = [Path(archive_root) / n for n in sorted(names) if (Path(archive_root) / n).is_dir()]
    if len(found) != 1:
        raise GtAdapterError("GT_FRAME_DIRECTORY_MISSING", f"frame {frame_id}")
    return found[0]


def load_frame(frame_dir: Path) -> ArchivedFrame:
    """Rebuild the scorer's exact ReadyTarget from archive bytes and resolve the received identity."""
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
    from rl_agent.splitfusion_exact_gt_service_proto_v2.scorer_core import expected_from_identity, join_identity
    from rl_agent.splitfusion_exact_gt_service_proto_v2.deferred import validate_anchor_provenance
    frame_dir = Path(frame_dir)
    raw = {name: _read(frame_dir / name) for name in REFERENCED_FILES}
    for name in OPTIONAL_FILES:
        if (frame_dir / name).exists():
            raw[name] = _read(frame_dir / name)
    metadata = C._parse(raw["scorer/target_metadata.json"])
    score = json.loads(raw["scorer/numerical_score.json"])
    identity = C.canonical({k: metadata[k] for k in IDENTITY_KEYS})
    target_doc, expected = expected_from_identity(identity)
    target = R._admit(metadata, raw["scorer/segmentation.u8"], raw["scorer/object_ignore.u8"], **expected)
    if target.metadata_bytes != raw["scorer/target_metadata.json"]:
        raise GtAdapterError("GT_TARGET_REBUILD_DIFFERS")
    anchor = None
    if "scorer/anchor_provenance.json" in raw:
        anchor = json.loads(raw["scorer/anchor_provenance.json"])
        mapping = validate_anchor_provenance(anchor["provenance"], target_identity=target_doc)
        derived, mode = join_identity(anchor["prediction_identity_as_received"], mapping,
                                      target_doc["capture_identity"])
        if not (mode == anchor["join_mode"] == "ANCHOR_PROVENANCE_V1"
                and C.canonical(derived) == C.canonical(anchor["join_identity"])
                == C.canonical(score["prediction_identity"])):
            raise GtAdapterError("GT_ANCHOR_PROVENANCE_JOIN_DIFFERS")
        received = anchor["prediction_identity_as_received"]
    else:
        if C.canonical(score["prediction_identity"]["gt_identity"]) != C.canonical(target_doc["capture_identity"]):
            raise GtAdapterError("GT_PREDICTION_CAPTURE_IDENTITY_DIFFERS")
        received = score["prediction_identity"]
    avo_doc = json.loads(raw["producer/numerical_avo.json"])
    if C.canonical(avo_doc) != C.canonical({k: metadata["avo"][k] for k in avo_doc}):
        raise GtAdapterError("GT_PRODUCER_AVO_DIFFERS_FROM_TARGET")
    return ArchivedFrame(frame_dir, raw, metadata, score, target, target_doc, anchor, received)


def _labels(frame: ArchivedFrame) -> np.ndarray:
    return np.frombuffer(frame.raw["scorer/predicted_labels.u8"], np.uint8).reshape(720, 1280)


def _numeric_score(frame: ArchivedFrame, identity: dict, labels: np.ndarray, records: list):
    from rl_agent.splitfusion_numeric_gt_integration_v2 import scoring as Q
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
    spec = W.load_run4_quality_spec(MAIN)
    scorer = Q.NumericalScorer()
    prediction = Q.admit_prediction(scorer, identity=identity, labels=labels, records=records,
                                    prediction_contract=frame.metadata["prediction_contract"])
    result = scorer.score(Q.join_admitted_inputs(scorer, target=frame.target, prediction=prediction))
    value = quality.evaluate_exact_quality(spec, result.measurement)
    return result, (None if value is None else float(value.q_perc)), spec


def evaluator_inputs(frame: ArchivedFrame, labels: Optional[np.ndarray] = None,
                     records: Optional[list] = None) -> dict:
    """Keyword arguments for ``evaluation.evaluate_prepared`` built from the archived exact GT."""
    import cv2
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    from rl_agent.splitfusion_live_target_parity_v1 import adapter as T
    meta = frame.metadata
    seg = np.frombuffer(frame.raw["scorer/segmentation.u8"], np.uint8).reshape(432, 768)
    ign = np.frombuffer(frame.raw["scorer/object_ignore.u8"], np.uint8).reshape(432, 768)
    seg_png, ign_png = (bytes(cv2.imencode(".png", a)[1]) for a in (seg, ign))
    sample, frame_id = meta["sample_id"], meta["capture_identity"]["frame_id"]
    target = {"contract": T.CONTRACT, "provenance": C.thaw(meta["provenance"]),
              "segmentation": seg.copy(), "object_ignore": ign.copy(),
              "segmentation_png": seg_png, "object_ignore_png": ign_png,
              "tables": {"object_boxes.csv": C.thaw(meta["object_boxes"]),
                         "object_ignore_regions.csv": C.thaw(meta["object_ignore_regions"]),
                         "target_manifest.csv": [{
                             "sample_id": sample,
                             "segmentation_mask_path": "numeric_gt/scorer/segmentation.u8",
                             "segmentation_mask_sha256": _sha(seg_png),
                             "object_ignore_mask_path": "numeric_gt/scorer/object_ignore.u8",
                             "object_ignore_mask_sha256": _sha(ign_png)}]}}
    prepared = E.restore_prepared_target(frame_id=frame_id, sample_id=sample, target=target,
                                         content_sha256=E._target_digest(frame_id, sample, target))
    return dict(identity=C.thaw(meta["action_identity"]), prepared=prepared,
                predicted_labels=(_labels(frame) if labels is None else labels).copy(),
                predictions=frame.score["prediction_records"] if records is None else records,
                avo=C.thaw(meta["avo"]))


def rescore(frame: ArchivedFrame, *, with_evaluator: bool = True) -> dict:
    """Field-by-field exact reproduction of the archived numerical_score.json (no tolerances)."""
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_numeric_gt_runtime_v1 import live_archive as LA
    score, labels = frame.score, _labels(frame)
    result, q_perc, spec = _numeric_score(frame, score["prediction_identity"], labels, score["prediction_records"])
    _, rebuilt = LA._scored_values(frame.target, result, q_perc, spec)
    document = json.loads(rebuilt)
    checks = {key: C.canonical(document.get(key)) == C.canonical(score.get(key))
              for key in sorted(set(document) | set(score))}
    checks["numerical_score_json_bytes"] = rebuilt == frame.raw["scorer/numerical_score.json"]
    checks["predicted_labels_sha256"] = _sha(frame.raw["scorer/predicted_labels.u8"]) == score["predicted_labels"]["sha256"]
    checks["prediction_digest"] = score["prediction_identity"]["prediction_digest"] == _sha(
        frame.raw["scorer/predicted_labels.u8"] + C.canonical(score["prediction_records"]))
    checks["ready_semantic_sha256"] = frame.target.semantic_sha256 == score["ready_semantic_sha256"]
    if with_evaluator:
        from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
        evaluated = E.evaluate_prepared(**evaluator_inputs(frame))
        checks["evaluate_prepared_measurement"] = (
            C.canonical(dict(evaluated.measurement)) == C.canonical(score["measurement"]))
    return {"frame_id": frame.metadata["capture_identity"]["frame_id"], "checks": checks,
            "exact": all(checks.values()), "q_perc": score["q_perc"],
            "anchor_provenance": frame.anchor_join is not None}


def score_prediction(frame: ArchivedFrame, labels: np.ndarray, records: list) -> dict:
    """Score another branch's native prediction (e.g. full-local, UINT8) against the same exact GT."""
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    identity = dict(frame.score["prediction_identity"],
                    prediction_digest=_sha(labels.tobytes(order="C") + C.canonical(records)))
    result, q_perc, _ = _numeric_score(frame, identity, labels, records)
    evaluated = E.evaluate_prepared(**evaluator_inputs(frame, labels, records))
    return {"measurement": C._json_owned(result.measurement), "q_perc": q_perc,
            "quality_defined": q_perc is not None, "prediction_digest": identity["prediction_digest"],
            "evaluate_prepared_measurement_equal":
                C.canonical(dict(evaluated.measurement)) == C.canonical(result.measurement)}


def join_record(entry: dict, frame: ArchivedFrame) -> dict:
    """Explicit identity joins: retained input/wire/action record <-> archived prediction <-> GT."""
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    received, meta = frame.received_identity, frame.metadata
    key = entry["join_key"]
    return {
        "action_identity": C.canonical(received["action_identity"]) == C.canonical(entry["identity"]),
        "gt_identity": C.canonical(received["gt_identity"]) == C.canonical(entry["gt_identity"]),
        "capture_timestamp_ns": received["capture_timestamp_ns"] == key["capture_timestamp_ns"]
                                == meta["capture_identity"]["capture_timestamp_ns"],
        "session_tensor_frame": (received["action_identity"]["session_uuid"], received["action_identity"]["tensor_seq"],
                                 received["action_identity"]["frame_id"])
                                == (key["session_uuid"], key["tensor_seq"], key["frame_id"]),
        "wire_frame_context_equals_target_frame_context":
            C.canonical(entry["frame_context"]) == C.canonical(meta["frame_context"]),
        "target_frame_kind_policy": meta["action_identity"]["frame_kind"] == "POLICY_DECISION",
    }


def resolve_gt_references(retention_dir: Path, *, manifest_sha256: str, plan_sha256: str,
                          archive_root: Path, output_path: Path, root: Path = MAIN) -> dict:
    """Postflight, read-only on the capture: bind every retained record to its exact GT by reference."""
    from . import validate as V
    manifest = V.verify_archive_v2(retention_dir, expected_manifest_sha256=manifest_sha256,
                                   expected_plan_sha256=plan_sha256, strict=False)
    archive_root = Path(archive_root).resolve()
    directories = sorted(p for p in archive_root.iterdir() if p.is_dir() and p.name.startswith("frame_"))
    rank = None
    if len(directories) == manifest["observed_policy_decisions"]:
        tensors = sorted(json.loads(_read(d / "scorer/numerical_score.json"))["prediction_identity"]
                         ["action_identity"]["tensor_seq"] for d in directories)
        rank = {t: i for i, t in enumerate(tensors)}
    rows = []
    for entry in manifest["records"]:
        row = {"ordinal": entry["ordinal"], "frame_id": entry["join_key"]["frame_id"],
               "target_hash_domain": TARGET_HASH_DOMAIN, "edge_target_bundle_sha256_consulted": False}
        try:
            frame_dir = frame_directory(archive_root, entry["join_key"]["frame_id"])
            frame = load_frame(frame_dir)
            joins = join_record(entry, frame)
            if not all(joins.values()):
                raise GtAdapterError("INCOMPLETE_GT_JOIN_IDENTITY_MISMATCH",
                                     ",".join(k for k, v in joins.items() if not v))
            check = rescore(frame)
            if not check["exact"]:
                raise GtAdapterError("INCOMPLETE_GT_RESCORE_NOT_EXACT",
                                     ",".join(k for k, v in check["checks"].items() if not v))
            row.update(status="RESOLVED", frame_dir=frame_dir.name, joins=joins, rescore=check,
                       files={name: _sha(raw) for name, raw in sorted(frame.raw.items())},
                       prediction={"predicted_labels_sha256": frame.score["predicted_labels"]["sha256"],
                                   "prediction_digest": frame.score["prediction_identity"]["prediction_digest"],
                                   "records": len(frame.score["prediction_records"]),
                                   "source": "NUMERIC_GT_SCORER_ARCHIVE"},
                       ready_semantic_sha256=frame.score["ready_semantic_sha256"],
                       q_perc=frame.score["q_perc"],
                       ordinal_crosscheck=(None if rank is None else
                                           rank.get(entry["join_key"]["tensor_seq"]) == entry["ordinal"]))
        except GtAdapterError as exc:
            row.update(status=exc.reason if exc.reason.startswith("INCOMPLETE_") else "INCOMPLETE_" + exc.reason,
                       detail=str(exc)[:300])
        except Exception as exc:   # admission must see a classified failure, never a silent pass
            row.update(status="INCOMPLETE_GT_ADAPTER_EXCEPTION", detail=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    document = {"schema": REFERENCES_SCHEMA, "retention_manifest_sha256": manifest_sha256,
                "plan_sha256": plan_sha256, "archive_root": str(archive_root),
                "target_hash_domain": TARGET_HASH_DOMAIN,
                "ordinal_crosscheck_available": rank is not None,
                "scorer_sources": scorer_source_digests(root), "rows": rows,
                "status_counts": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}}
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with Path(output_path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(document, references_sha256=_sha(raw))


def check_archive(archive_root: Path, frame_ids: Optional[list] = None, *, with_evaluator: bool = True) -> dict:
    archive_root = Path(archive_root)
    directories = sorted(p for p in archive_root.iterdir() if p.is_dir() and p.name.startswith("frame_"))
    if frame_ids is not None:
        directories = [frame_directory(archive_root, f) for f in frame_ids]
    rows, failures = [], {}
    for directory in directories:
        try:
            row = rescore(load_frame(directory), with_evaluator=with_evaluator)
        except Exception as exc:
            row = {"frame_dir": directory.name, "exact": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(row)
        for key, ok in row.get("checks", {}).items():
            if not ok:
                failures[key] = failures.get(key, 0) + 1
    return {"archive_root": str(archive_root.resolve()), "frames": len(rows),
            "exact": sum(bool(r["exact"]) for r in rows), "field_failures": failures,
            "anchor_provenance_frames": sum(bool(r.get("anchor_provenance")) for r in rows),
            "quality_undefined_frames": sum(r.get("q_perc", 0) is None for r in rows),
            "checked_fields": sorted(rows[0]["checks"]) if rows and "checks" in rows[0] else [],
            "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Exact offline rescoring of a numeric-GT service archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, default=None,
                        help="checkout that produced the archive; scorer modules must be byte-identical")
    args = parser.parse_args(argv)
    sources = scorer_source_digests(MAIN)
    identical = None
    if args.worktree_root is not None:
        identical = {m: _sha((args.worktree_root / m).read_bytes()) == d for m, d in sources.items()}
        if not all(identical.values()):
            raise SystemExit("scorer modules differ from the producing checkout: "
                             + ",".join(m for m, ok in identical.items() if not ok))
    summary = check_archive(args.archive)
    summary.update(scorer_sources=sources, worktree_sources_identical=identical)
    raw = json.dumps(summary, sort_keys=True, indent=1, allow_nan=False).encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({k: summary[k] for k in ("frames", "exact", "field_failures", "anchor_provenance_frames",
                                              "quality_undefined_frames")}))
    return 0 if summary["exact"] == summary["frames"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
