"""Complete-v010 measurement repair; no policy, reward, or timer changes.

Target construction belongs in the GT producer. The edge evaluates already
prepared targets. Evidence payloads are returned to the caller, never written
or queued on this module's authority. No synthetic GT or missing-input default
is permitted. ``live_qualified`` is deliberately never asserted here.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as targets
from rl_agent.splitfusion_run5_live_avo_v1 import quality_adapter as people

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "splitfusion.complete_live_evaluation.v1"
VEHICLE_SCORE_THRESHOLD = .20
MAX_RECORDS = 512
DETECTION_SOURCE = "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_targeted_refinement_v1/audit_v1.py"
DETECTION_SHA256 = "64d7d677480328d64176a96b2e7bf7d1eda2e9eb12fc5aa6a74762f381c7727b"
PERSON_SOURCE = "rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py"
PERSON_SHA256 = "fa4b5817ba69eb0a53a8897abb7ba6a089e46bd6d045bcf7d785b752ac4a20a3"
IDENTITY_FIELDS = ("frame_id", "session_uuid", "decision_seq", "tensor_seq", "ticket_seq")


class EvaluationInputError(ValueError):
    pass


def _copy(value):
    return json.loads(targets.canonical(value))


def _digest(value):
    return hashlib.sha256(targets.canonical(value)).hexdigest()


def verify_sources():
    result = targets.verify_sources(ROOT)
    for label, relative, expected in (("detection", DETECTION_SOURCE, DETECTION_SHA256),
                                      ("person_avo", PERSON_SOURCE, PERSON_SHA256)):
        actual = targets.sha256(ROOT / relative)
        if actual != expected:
            raise EvaluationInputError(f"original {label} source pin changed")
        result[label] = {"path": relative, "sha256": actual}
    return result


def _target_digest(frame_id, sample_id, target):
    return _digest({"frame_id": frame_id, "sample_id": sample_id,
        "contract": target["contract"], "provenance": target["provenance"],
        "tables": target["tables"],
        "segmentation_png_sha256": hashlib.sha256(target["segmentation_png"]).hexdigest(),
        "object_ignore_png_sha256": hashlib.sha256(target["object_ignore_png"]).hexdigest()})


@dataclass(frozen=True)
class PreparedEvaluationTarget:
    frame_id: int
    sample_id: str
    target: Mapping[str, Any]
    content_sha256: str


@dataclass(frozen=True)
class EvaluationResult:
    measurement: Mapping[str, Any]
    evidence: Mapping[str, Any]
    predicted_labels: np.ndarray

    @property
    def predicted_png(self) -> bytes:
        """Deferred encoding: only the post-feedback archive should access this."""
        success, encoded = cv2.imencode(".png", self.predicted_labels)
        if not success:
            raise EvaluationInputError("prediction PNG encoding failed")
        return bytes(encoded)


def archive_payloads(result: EvaluationResult) -> dict[str, bytes]:
    """Encode durable evidence after feedback, never on the scoring branch."""
    png = result.predicted_png
    evidence = _copy(result.evidence)
    evidence["predicted_labels"]["png_sha256"] = hashlib.sha256(png).hexdigest()
    return {"predicted_labels.png": png,
            "evaluation.json": targets.canonical(evidence)}


def restore_prepared_target(*, frame_id: int, sample_id: str,
                            target: Mapping[str, Any], content_sha256: str) -> PreparedEvaluationTarget:
    """Validate complete target restored by an identity/hash-bound transport.

    The transport must validate its outer session/tensor/ticket identity first.
    This validator checks numerical payloads and the complete-input declaration;
    it cannot independently establish a collector's completeness assertion.
    """
    verify_sources()
    if type(frame_id) is not int or frame_id < 0 or not isinstance(sample_id, str) or not sample_id:
        raise EvaluationInputError("invalid prepared target frame/sample identity")
    if target.get("contract") != targets.CONTRACT:
        raise EvaluationInputError("complete v010 target required")
    provenance = target.get("provenance", {})
    if provenance.get("sample_id") != sample_id or provenance.get("schema") != targets.SCHEMA:
        raise EvaluationInputError("prepared target provenance/sample mismatch")
    if not provenance.get("scope") or any(provenance.get("complete", {}).get(k) is not True
                                           for k in targets.COMPLETENESS_KEYS):
        raise EvaluationInputError("incomplete prepared target provenance")
    for name, allowed in (("segmentation", {0, 1, 2, 255}), ("object_ignore", {0, 255})):
        array, encoded = target.get(name), target.get(name + "_png")
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.shape != (432, 768):
            raise EvaluationInputError(f"invalid prepared {name} grid")
        if not set(np.unique(array)).issubset(allowed) or not isinstance(encoded, bytes):
            raise EvaluationInputError(f"invalid prepared {name} labels/PNG")
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
        if not np.array_equal(decoded, array):
            raise EvaluationInputError(f"prepared {name} PNG/array mismatch")
    table = target.get("tables", {})
    if not all(k in table for k in ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv")):
        raise EvaluationInputError("missing full v010 target tables")
    for name, rows in table.items():
        if any(str(row.get("sample_id")) != sample_id for row in rows):
            raise EvaluationInputError("foreign sample in target tables")
        # Literal original target_manifest has only sample_id, not frame_id.
        if name != "target_manifest.csv" and any(str(row.get("frame_id")) != str(frame_id) for row in rows):
            raise EvaluationInputError("foreign frame in target tables")
    if _target_digest(frame_id, sample_id, target) != content_sha256:
        raise EvaluationInputError("prepared target content digest mismatch")
    own = {k: v for k, v in target.items() if k not in ("segmentation", "object_ignore")}
    own["provenance"] = _copy(target["provenance"])
    own["tables"] = _copy(table)
    for name in ("segmentation", "object_ignore"):
        own[name] = target[name].copy()
        own[name].flags.writeable = False
    return PreparedEvaluationTarget(frame_id, sample_id, own, content_sha256)


def prepare_target(inputs: targets.CompleteTargetInputs) -> PreparedEvaluationTarget:
    """GT-worker operation; invokes the unchanged full builder exactly once."""
    target = targets.build_complete_v010(inputs)
    frame_id, sample_id = int(inputs.frame["frame_id"]), str(inputs.frame["sample_id"])
    return restore_prepared_target(frame_id=frame_id, sample_id=sample_id, target=target,
        content_sha256=_target_digest(frame_id, sample_id, target))


@lru_cache(maxsize=1)
def _detection_code():
    verify_sources()
    path = ROOT / DETECTION_SOURCE
    parsed = ast.parse(path.read_text(), filename=str(path))
    names = {"load_gt", "load_predictions", "prediction_center", "match_frame", "is_neutral", "score_arm"}
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    constants = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {
                    "MODEL_SIZE", "SOURCE_SIZE", "CLASSES", "MATCH_RADIUS_M", "PREDICTION_FIELDS"}:
                    constants[name.id] = ast.literal_eval(node.value)
    if len(nodes) != len(names) or constants["MATCH_RADIUS_M"] != 3.0:
        raise EvaluationInputError("frozen detection API changed")
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), constants


def _detection(rows, object_boxes):
    """Private original-function globals per call: no shared mutable seams."""
    code, constants = _detection_code()
    prediction_path, gt_path = Path("/prediction/records.csv"), Path("/complete/contracts/v010/val/object_boxes.csv")
    def read_csv(path):
        if Path(path) == prediction_path:
            return rows
        if Path(path) == gt_path:
            return object_boxes
        raise EvaluationInputError("unexpected original scorer file read")
    ns = {"Any": Any, "Mapping": Mapping, "Sequence": Sequence, "Path": Path,
          "math": math, "defaultdict": defaultdict, "read_csv": read_csv, **constants}
    exec(code, ns)
    return ns, prediction_path


def _identity(identity, prepared, avo):
    if any(k not in identity for k in IDENTITY_FIELDS):
        raise EvaluationInputError("full frame/session/decision/tensor/ticket identity required")
    for key in ("frame_id", "decision_seq", "tensor_seq", "ticket_seq"):
        if type(identity[key]) is not int or identity[key] < 0:
            raise EvaluationInputError(f"invalid identity {key}")
    if not isinstance(identity["session_uuid"], str) or not identity["session_uuid"]:
        raise EvaluationInputError("missing session UUID")
    if identity["frame_id"] != prepared.frame_id or avo.get("frame_id") != prepared.frame_id:
        raise EvaluationInputError("prediction/target/AVO frame mismatch")
    if any(k not in avo for k in ("qualified", "structural_ignored")):
        raise EvaluationInputError("complete AVO qualified/structural lists required")
    seen = set()
    for row in [*avo["qualified"], *avo["structural_ignored"]]:
        if row.get("frame_id", prepared.frame_id) != prepared.frame_id:
            raise EvaluationInputError("foreign AVO row frame")
        key = str(row["gt_actor_id"])
        if key in seen:
            raise EvaluationInputError("duplicate AVO actor identity")
        seen.add(key)


def evaluate_prepared(*, identity: Mapping[str, Any], prepared: PreparedEvaluationTarget,
                      predicted_labels: np.ndarray, predictions: Sequence[Mapping[str, Any]],
                      avo: Mapping[str, Any]) -> EvaluationResult:
    """Return exact sufficient statistics plus complete, immutable evidence.

    The existing quality-feedback caller consumes ``measurement`` unchanged.
    Do not delay ACK emission to persist ``evidence``/``predicted_png``; the
    caller must use its separately bounded archive and fault accounting.
    """
    verify_sources()
    _identity(identity, prepared, avo)
    prepared = restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
        target=prepared.target, content_sha256=prepared.content_sha256)
    if len(predictions) > MAX_RECORDS:
        raise EvaluationInputError("prediction record bound exceeded")
    raw = _copy(list(predictions))
    for row in raw:
        if row.get("class_name") not in ("vehicle", "person"):
            raise EvaluationInputError("unknown prediction class")
        if int(row.get("frame_id", -1)) != prepared.frame_id:
            raise EvaluationInputError("foreign/missing prediction frame")
        if not math.isfinite(float(row["score"])) or not 0 <= float(row["score"]) <= 1:
            raise EvaluationInputError("invalid calibrated prediction score")
    scorer_rows = [dict(row, sample_id=prepared.sample_id) for row in raw]
    scorer, prediction_path = _detection(scorer_rows, prepared.target["tables"]["object_boxes.csv"])
    grouped, missing = scorer["load_predictions"](prediction_path)
    if missing:
        raise EvaluationInputError("incomplete full prediction record: " + ", ".join(missing[:4]))
    gt, _ = scorer["load_gt"](Path("/complete"), "v010")
    detection = scorer["score_arm"](experiment=Path("/complete"), contract="v010",
        frame_ids=[prepared.sample_id], predictions=grouped, gt=gt,
        threshold=VEHICLE_SCORE_THRESHOLD, ignore_cache={prepared.sample_id: prepared.target["object_ignore"]})
    ordered_indices = sorted(range(len(raw)), key=lambda i: (-float(raw[i]["score"]), str(raw[i]["class_name"])))
    scored_indices = [i for i in ordered_indices if float(raw[i]["score"]) >= VEHICLE_SCORE_THRESHOLD]
    scored_predictions = [grouped[prepared.sample_id][j] for j, i in enumerate(ordered_indices)
                          if float(raw[i]["score"]) >= VEHICLE_SCORE_THRESHOLD]
    frame_gt = gt.get(prepared.sample_id, [])
    used_pred, used_gt, matched = scorer["match_frame"](scored_predictions, frame_gt)
    vehicle_pairs = []
    for p, g in sorted(matched.items()):
        if frame_gt[g]["class_name"] != "vehicle":
            continue
        error = math.hypot(scored_predictions[p]["world_x"] - frame_gt[g]["world_x"],
                           scored_predictions[p]["world_y"] - frame_gt[g]["world_y"])
        vehicle_pairs.append({"prediction_record_index": scored_indices[p], "target_index": g,
            "target_source_identity": frame_gt[g]["source_identity"], "error_m": error})
    pixel = targets.score_complete_v010(prepared.target, predicted_labels)
    measurement = {"frame_id": prepared.frame_id}
    for cls in ("vehicle", "person"):
        for key, value in pixel["quality_grid_segmentation_counts"][cls].items():
            measurement[f"seg_{cls}_{key}"] = value
    vehicle = detection["classes"]["vehicle"]
    for key in ("eligible_gt", "tp", "fp", "fn", "ignored_predictions"):
        measurement[f"loc_vehicle_{key}"] = vehicle[key]
    measurement["loc_vehicle_matched_xy_errors_m"] = [p["error_m"] for p in vehicle_pairs]
    measurement.update(people.person_avo_measurement(raw, avo))
    assert len(vehicle_pairs) == measurement["loc_vehicle_tp"]
    # Person match/order is unchanged from the qualified live adapter.
    person_indices = [i for i, p in enumerate(raw) if p["class_name"] == "person" and float(p["score"]) >= .25]
    person_predictions = [raw[i] for i in person_indices]
    eligible = [r for r in avo["qualified"] if float(r["avo"]) >= .65]
    person_matches, _ = people.greedy_match(person_predictions, eligible)
    person_pairs = [{"prediction_record_index": person_indices[p], "target_index": g,
        "target_actor_id": str(eligible[g]["gt_actor_id"]),
        "error_m": math.hypot(float(person_predictions[p]["world_x"]) - float(eligible[g]["world_x"]),
                              float(person_predictions[p]["world_y"]) - float(eligible[g]["world_y"]))}
                    for p, g in sorted(person_matches.items())]
    assert [p["error_m"] for p in person_pairs] == measurement["loc_person_matched_xy_errors_m"]
    labels = np.ascontiguousarray(predicted_labels).copy()
    labels.flags.writeable = False
    evidence = {"schema": SCHEMA, "identity": _copy(identity), "live_qualified": False,
        "measurement": _copy(measurement), "source_pins": verify_sources(),
        "target_content_sha256": prepared.content_sha256,
        "target_provenance": _copy(prepared.target["provenance"]),
        "prediction_records": raw, "prediction_records_sha256": _digest(raw),
        "predicted_labels": {"shape": list(predicted_labels.shape), "dtype": str(predicted_labels.dtype),
            "array_sha256": hashlib.sha256(labels.tobytes(order="C")).hexdigest()},
        "vehicle_targets": _copy(frame_gt), "person_avo": _copy(avo),
        "matched_pairs": {"vehicle": vehicle_pairs, "person": person_pairs},
        "scoring": {"vehicle_score_threshold": .20, "person_score_threshold": .25,
            "person_avo_threshold": .65, "match_radius_m": 3.0,
            "vehicle_detection": detection["classes"]["vehicle"],
            "segmentation": pixel["original_score"], "reward_and_timeout": "UNCHANGED_CALLER_IMPLEMENTATION"}}
    return EvaluationResult(_copy(measurement), evidence, labels)
