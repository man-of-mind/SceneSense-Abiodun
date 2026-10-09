"""Deferred, byte-exact reconstruction of the original artifacts from the payload.

None of this runs on the scoring-readiness path. Given a staged frame, its
numeric target and AVO result, these functions rebuild exactly what the pinned
modules used to produce inline: the complete-input evidence document, the
``build_complete_v010`` target dictionary (CSV-roundtripped tables, PNG bytes,
summary with file hashes), the ``PreparedEvaluationTarget`` through the
unchanged ``restore_prepared_target`` validator, and the two AVO documents.
A binding record ties the new semantic digest to every legacy artifact hash.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v1.source_geometry import json_value
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG

from . import originals as O
from . import payload as P

BINDING_SCHEMA = "splitfusion.numeric_gt_producer.deferred_artifact_binding.v1"
AVO_CLOCKS = ("start_raw_ns", "write_raw_ns", "prescore_input_ready_raw_ns", "scoring_start_raw_ns",
              "end_raw_ns", "ready_wall_ns", "sync_observed_raw_ns")


class ArtifactError(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value) -> bytes:
    return A.canonical(json_value(value))


def png_bytes(array: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", np.ascontiguousarray(array))
    if not success:
        raise ArtifactError("PNG encoding failed")
    return encoded.tobytes()


def legacy_target(staged, target, *, sources: Mapping[str, Mapping[str, str]] | None = None) -> dict:
    """Exact ``A.build_complete_v010`` result dictionary for this frame."""
    builder = O.bound().builder
    inputs = staged.inputs
    if inputs is None:
        raise ArtifactError("staged frame lacks validated complete inputs")
    sample_id = inputs.frame["sample_id"]
    seg_png, obj_png = png_bytes(target.segmentation), png_bytes(target.object_ignore)
    seg_hash, obj_hash = _sha(seg_png), _sha(obj_png)
    target_manifest = [{"sample_id": sample_id, "segmentation_mask_path": target.seg_rel,
        "segmentation_mask_sha256": seg_hash, "object_ignore_mask_path": target.obj_rel,
        "object_ignore_mask_sha256": obj_hash}]
    manifest_fields = ("sample_id", "segmentation_mask_path", "segmentation_mask_sha256",
                       "object_ignore_mask_path", "object_ignore_mask_sha256")
    csv_bytes = {"object_boxes.csv": P.legacy_csv_bytes(target.positive_rows, builder.OBJECT_FIELDS),
                 "object_ignore_regions.csv": P.legacy_csv_bytes(target.ignore_rows, builder.IGNORE_FIELDS),
                 "target_manifest.csv": P.legacy_csv_bytes(target_manifest, manifest_fields)}
    tables = {"object_boxes.csv": P.legacy_rows(target.positive_rows, builder.OBJECT_FIELDS),
              "object_ignore_regions.csv": P.legacy_rows(target.ignore_rows, builder.IGNORE_FIELDS),
              "target_manifest.csv": P.legacy_rows(target_manifest, manifest_fields)}
    summary = {
        "frames": 1, "positive_records": len(target.positive_rows), "ignore_records": len(target.ignore_rows),
        "positive_by_class_source": {f"{k[0]}:{k[1]}": v for k, v in sorted(target.object_record_counts.items())},
        "ignore_by_class_source_reason": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in sorted(target.ignore_record_counts.items())},
        "segmentation_pixels": dict(target.pixel_counts),
        "object_supervision_cells_two_classes": dict(target.object_cell_counts),
        "object_boxes_sha256": _sha(csv_bytes["object_boxes.csv"]),
        "object_ignore_regions_sha256": _sha(csv_bytes["object_ignore_regions.csv"]),
        "target_manifest_sha256": _sha(csv_bytes["target_manifest.csv"]),
        "target_payload_hash": hashlib.sha256("".join(
            row["segmentation_mask_sha256"] + row["object_ignore_mask_sha256"] for row in target_manifest).encode("ascii")).hexdigest()}
    pins = O.bound().pins
    if sources is None:
        sources = {name: dict(pins[name]) for name in A.SOURCES}
    segmentation = cv2.imdecode(np.frombuffer(seg_png, np.uint8), cv2.IMREAD_UNCHANGED)
    object_ignore = cv2.imdecode(np.frombuffer(obj_png, np.uint8), cv2.IMREAD_UNCHANGED)
    return {"contract": A.CONTRACT, "live_qualified": False, "sources": sources,
            "provenance": dict(inputs.provenance), "summary": summary, "tables": tables,
            "segmentation": segmentation, "object_ignore": object_ignore,
            "segmentation_png": seg_png, "object_ignore_png": obj_png, "_csv_bytes": csv_bytes}


def prepared_target(staged, legacy: Mapping[str, Any]) -> E.PreparedEvaluationTarget:
    """Run the unchanged frozen validator on the reconstructed artifact."""
    target = {k: v for k, v in legacy.items() if not k.startswith("_")}
    frame_id, sample_id = int(staged.frame["frame_id"]), str(staged.frame["sample_id"])
    return E.restore_prepared_target(frame_id=frame_id, sample_id=sample_id, target=target,
        content_sha256=E._target_digest(frame_id, sample_id, target))


def avo_documents(staged, avo, *, identity: Mapping[str, Any], clocks: Mapping[str, int],
                  payload_filenames: Mapping[str, str], prescore_filename: str) -> dict:
    """Original pre-score and scored AVO documents from numeric values plus clocks."""
    for name in AVO_CLOCKS:
        if type(clocks.get(name)) is not int:
            raise ArtifactError("all seven AVO clocks are required to rebuild the documents: " + name)
    # The original AVO writer observes source synchronization with its own clock.
    sync = dict(staged.sync, observed_raw_ns=clocks["sync_observed_raw_ns"])
    height, width = staged.depth_raw.shape[:2]
    payloads = {}
    for name, value in (("depth_bgra", staged.depth_raw), ("rawsemantic", staged.semantic)):
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        payloads[name] = {"filename": payload_filenames[name], "sha256": _sha(buffer.getvalue())}
    prescore = {"schema": LG.SCHEMA, "record_kind": "AVO_PRE_SCORE_INPUTS", "mask_contract": LG.MASK_CONTRACT,
                "identity": dict(identity), "source_frame_id": staged.frame_id, "sensor_timestamp": float(staged.stamp),
                "native_shape": [height, width], "camera_matrix": staged.camera_matrix.tolist(),
                "camera_inverse": staged.camera_inverse.tolist(), "intrinsics": avo.intrinsics.tolist(),
                "source_synchronization": sync, "raw_pedestrian_rows": list(avo.rows), "payloads": payloads,
                "start_raw_ns": clocks["start_raw_ns"], "write_raw_ns": clocks["write_raw_ns"]}
    prescore_bytes = LG._canonical(prescore)
    input_binding = {"filename": prescore_filename, "sha256": _sha(prescore_bytes)}
    document = {**prescore, "record_kind": "AVO_SCORED_FRAME", "avo": json_value(avo.avo),
                "prescore_input": input_binding, "prescore_input_ready_raw_ns": clocks["prescore_input_ready_raw_ns"],
                "scoring_start_raw_ns": clocks["scoring_start_raw_ns"], "end_raw_ns": clocks["end_raw_ns"],
                "ready_wall_ns": clocks["ready_wall_ns"]}
    document_bytes = LG._canonical(document)
    return {"prescore": prescore, "prescore_bytes": prescore_bytes, "prescore_sha256": input_binding["sha256"],
            "document": document, "document_bytes": document_bytes, "document_sha256": _sha(document_bytes)}


def binding(payload: P.NumericGtPayload, legacy: Mapping[str, Any], prepared: E.PreparedEvaluationTarget,
            *, avo_document_sha256: str | None = None, owned_source_sha256: str | None = None) -> dict:
    """Bind the semantic digest to every original artifact hash (never relabelled)."""
    P.verify_payload(payload)
    if not np.array_equal(payload.segmentation, legacy["segmentation"]) or not np.array_equal(payload.object_ignore, legacy["object_ignore"]):
        raise ArtifactError("legacy PNG decode differs from the numeric masks")
    if list(map(dict, payload.object_boxes)) != legacy["tables"]["object_boxes.csv"] \
            or list(map(dict, payload.object_ignore_regions)) != legacy["tables"]["object_ignore_regions.csv"]:
        raise ArtifactError("legacy tables differ from the payload rows")
    return {"schema": BINDING_SCHEMA, "semantic_sha256": payload.semantic_sha256,
            "prepared_content_sha256": prepared.content_sha256,
            "input_content_sha256": payload.input_content_sha256,
            "owned_source_sha256": owned_source_sha256 if owned_source_sha256 is not None
                                   else payload.provenance["evidence_sha256"]["owned_source"],
            "segmentation_png_sha256": _sha(legacy["segmentation_png"]),
            "object_ignore_png_sha256": _sha(legacy["object_ignore_png"]),
            "csv_sha256": {name: _sha(raw) for name, raw in legacy["_csv_bytes"].items()},
            "target_payload_hash": legacy["summary"]["target_payload_hash"],
            "avo_document_sha256": avo_document_sha256,
            "prepared_digest_schema": "legacy PNG/CSV-dependent E._target_digest",
            "semantic_digest_schema": P.SCHEMA}
