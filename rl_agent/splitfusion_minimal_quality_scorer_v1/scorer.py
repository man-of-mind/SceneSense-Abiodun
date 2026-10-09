"""One numerical pass; original evidence is an explicit deferred operation.

Only offline_admission can produce this package's private admitted fixture.
That fixture is NOT the coordinator's forthcoming DTO. No live call site is
installed. Python reflection/hostile in-process code is not a trust boundary.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_ISSUER = object()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _frozen_rows(rows):
    # Rows here contain only scalar numbers/strings and immutable box tuples.
    return tuple(MappingProxyType(dict(row)) for row in rows)


@dataclass(frozen=True, slots=True, init=False)
class _AdmittedFrame:
    issuer: object
    owner: object
    frame_id: int
    segmentation: bytes
    object_ignore: bytes
    labels: bytes
    labels_shape: tuple
    predictions: tuple
    ground_truth: tuple
    detection_indices: tuple
    person_indices: tuple
    eligible: tuple
    avo_ignored: tuple
    structural: tuple
    eligible_indices: tuple
    ignored_indices: tuple
    evidence_inputs: bytes
    source_pins: bytes
    original_content_sha256: str
    shared_context: object

    def __init__(self, *args, **kwargs):
        raise TypeError("admitted fixtures can only be issued by the private admission adapter")

    def label_view(self):
        return np.frombuffer(self.labels, np.uint8).reshape(self.labels_shape)


def _issue_frame(*values):
    result = object.__new__(_AdmittedFrame)
    if len(values) != len(fields(_AdmittedFrame)):
        raise TypeError("complete private fixture required")
    for field, value in zip(fields(_AdmittedFrame), values):
        object.__setattr__(result, field.name, value)
    return result


@dataclass(frozen=True, slots=True)
class Associations:
    # p is the original prediction record index. Detection g is the full
    # target-table index. Person g is the relevant filtered-list index.
    detection_distance_order: tuple
    vehicle_prediction_order: tuple
    person_prediction_order: tuple
    person_avo_ignored: tuple
    person_structural_ignored: tuple
    vehicle_neutral: tuple


@dataclass(frozen=True, slots=True)
class ScoreResult:
    _frame: _AdmittedFrame
    _measurement: tuple
    associations: Associations
    confusion: tuple
    ignored_pixels: int
    vehicle_detection: tuple

    @property
    def measurement(self):
        # New caller-owned dictionary/lists, exactly the legacy representation.
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self._measurement}


def _compile_functions(payload, path, names, namespace):
    parsed = ast.parse(payload, filename=str(path))
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError("pinned numerical function inventory changed")
    futures = [node for node in parsed.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    exec(compile(ast.Module(body=futures + nodes, type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)
    return namespace


class NumericalScorer:
    """Startup verifies and compiles the same exact bytes, once per instance.

    Matching comes directly from frozen source ASTs in private namespaces;
    no imports of the CLI, no pandas and no disk-reading callback in scoring.
    The code objects continue to refer to verified bytes if a file changes
    later. Coordinator must bind this lifecycle to its source admission.
    """

    def __init__(self):
        from .offline_admission import verify_fixture_validator_sources
        verify_fixture_validator_sources()
        from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
        from rl_agent.splitfusion_numeric_gt_integration_v1.sources import verify_sources_once
        self._owner = object()
        self._source_pins = canonical(E.verify_sources())
        pins = json.loads(self._source_pins)
        self._verified_sources = verify_sources_once(ROOT,
            {name: (item["path"], item["sha256"]) for name, item in pins.items()})
        blobs = {name: self._verified_sources.source(name).raw for name in pins}
        namespace = dict(Any=Any, Mapping=Mapping, Sequence=Sequence, Path=Path,
                         math=math, defaultdict=defaultdict)
        for node in ast.parse(blobs["detection"]).body:
            if isinstance(node, ast.Assign):
                for key in node.targets:
                    if isinstance(key, ast.Name) and key.id in (
                        "MODEL_SIZE", "SOURCE_SIZE", "CLASSES", "MATCH_RADIUS_M", "PREDICTION_FIELDS"):
                        namespace[key.id] = ast.literal_eval(node.value)
        names = ("load_gt", "load_predictions", "prediction_center", "match_frame", "is_neutral")
        self._detection = _compile_functions(blobs["detection"], E.DETECTION_SOURCE, names, namespace)
        self._person = _compile_functions(blobs["person_avo"], E.PERSON_SOURCE,
                                         ("greedy_match",), {"math": math})

    def score(self, frame):
        """Admitted numerical call: no decoding, file I/O, hashes or JSON work."""
        if type(frame) is not _AdmittedFrame or frame.issuer is not _ISSUER or frame.owner is not self._owner:
            raise TypeError("private immutable frame from this scorer's offline admission required")
        truth = np.frombuffer(frame.segmentation, np.uint8).reshape(432, 768)
        ignore = np.frombuffer(frame.object_ignore, np.uint8).reshape(432, 768)
        prediction = frame.label_view()
        if prediction.shape != truth.shape:
            prediction = cv2.resize(prediction, (768, 432), interpolation=cv2.INTER_NEAREST)
        valid = truth != 255
        confusion = np.bincount(truth[valid].astype(np.int64) * 3
            + prediction[valid].astype(np.int64), minlength=9).reshape(3, 3)
        measurement = {"frame_id": frame.frame_id}
        for name, index in (("vehicle", 1), ("person", 2)):
            intersection = int(confusion[index, index])
            gt, pred = int(confusion[index, :].sum()), int(confusion[:, index].sum())
            for key, value in (("gt_pixels", gt), ("pred_pixels", pred),
                               ("intersection_pixels", intersection), ("union_pixels", gt + pred - intersection)):
                measurement[f"seg_{name}_{key}"] = value

        detection = tuple(frame.predictions[i] for i in frame.detection_indices)
        used, used_gt, matches = self._detection["match_frame"](detection, frame.ground_truth)
        distance_order = tuple((frame.detection_indices[p], g, _distance(detection[p], frame.ground_truth[g]))
                               for p, g in matches.items())
        vehicle_pairs = tuple((frame.detection_indices[p], g, _distance(detection[p], frame.ground_truth[g]))
            for p, g in sorted(matches.items()) if frame.ground_truth[g]["class_name"] == "vehicle")
        vehicle_xy = [error for p, g, error in distance_order if frame.ground_truth[g]["class_name"] == "vehicle"]
        neutral, fp = [], 0
        for p, pred in enumerate(detection):
            if p not in used and pred["class_name"] == "vehicle":
                if self._detection["is_neutral"](pred, ignore):
                    neutral.append(frame.detection_indices[p])
                else:
                    fp += 1
        eligible = sum(g["class_name"] == "vehicle" for g in frame.ground_truth)
        tp, fn = len(vehicle_pairs), eligible - len(vehicle_pairs)
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        vehicle = dict(eligible_gt=eligible, tp=tp, fp=fp, fn=fn, ignored_predictions=len(neutral),
            precision=precision, recall=recall, f1=2.0 * precision * recall / max(1e-12, precision + recall),
            xy_mae_m=sum(vehicle_xy) / len(vehicle_xy) if vehicle_xy else None)
        for key in ("eligible_gt", "tp", "fp", "fn", "ignored_predictions"):
            measurement["loc_vehicle_" + key] = vehicle[key]
        measurement["loc_vehicle_matched_xy_errors_m"] = tuple(error for _, _, error in vehicle_pairs)

        people = tuple(frame.predictions[i] for i in frame.person_indices)
        pmatch, pused = self._person["greedy_match"](people, frame.eligible)
        remaining = set(range(len(people))) - set(pmatch)
        imatch, _ = self._person["greedy_match"](people, frame.avo_ignored, remaining)
        remaining -= set(imatch)
        smatch, _ = self._person["greedy_match"](people, frame.structural, remaining)
        remaining -= set(smatch)
        def pairs(matched, targets):
            return tuple((frame.person_indices[p], g, _distance(people[p], targets[g]))
                         for p, g in sorted(matched.items()))
        person_pairs = pairs(pmatch, frame.eligible)
        measurement.update(loc_person_eligible_gt=len(frame.eligible), loc_person_tp=len(pused),
            loc_person_fp=len(remaining), loc_person_fn=len(frame.eligible) - len(pused),
            loc_person_ignored_predictions=len(imatch) + len(smatch),
            loc_person_avo_ignored_predictions=len(imatch), loc_person_structural_ignored_predictions=len(smatch),
            loc_person_matched_xy_errors_m=tuple(error for _, _, error in person_pairs))
        return ScoreResult(frame, tuple(measurement.items()), Associations(distance_order, vehicle_pairs,
            person_pairs, pairs(imatch, frame.avo_ignored), pairs(smatch, frame.structural), tuple(neutral)),
            tuple(tuple(int(x) for x in row) for row in confusion), int(np.count_nonzero(~valid)), tuple(vehicle.items()))


def _distance(prediction, target):
    return math.hypot(float(prediction["world_x"]) - float(target["world_x"]),
                      float(prediction["world_y"]) - float(target["world_y"]))


def reconstruct_evidence(result, *, legacy_reference=None):
    """Deferred exact legacy evidence; uses associations, never re-matches."""
    if type(result) is not ScoreResult:
        raise TypeError("scoring result required")
    frame, association = result._frame, result.associations
    if frame.shared_context is None:
        inputs = json.loads(frame.evidence_inputs)
        original_digest = frame.original_content_sha256
    else:
        from .shared_admission import deferred_inputs
        inputs, original_digest = deferred_inputs(frame, legacy_reference)
    confusion = np.asarray(result.confusion, dtype=np.int64)
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    iou = intersection / np.maximum(union, 1.0)
    segmentation = dict(vehicle_iou=float(iou[1]), person_box_mask_iou=float(iou[2]),
        foreground_miou=float((iou[1] + iou[2]) / 2.0), background_iou=float(iou[0]),
        background_iou_role="diagnostic_only", ignored_pixels=result.ignored_pixels,
        confusion_matrix=confusion.tolist())
    return dict(schema="splitfusion.complete_live_evaluation.v1", identity=inputs["action_identity"],
        live_qualified=False, measurement=result.measurement, source_pins=json.loads(frame.source_pins),
        target_content_sha256=original_digest,
        target_provenance=inputs["target"]["provenance"], prediction_records=inputs["records"],
        prediction_records_sha256=hashlib.sha256(canonical(inputs["records"])).hexdigest(),
        predicted_labels=dict(shape=list(frame.labels_shape), dtype="uint8",
                              array_sha256=hashlib.sha256(frame.labels).hexdigest()),
        vehicle_targets=[dict(row, box=list(row["box"])) for row in frame.ground_truth],
        person_avo=inputs["avo_document"]["avo"],
        matched_pairs=dict(
            vehicle=[dict(prediction_record_index=p, target_index=g,
                target_source_identity=frame.ground_truth[g]["source_identity"], error_m=error)
                for p, g, error in association.vehicle_prediction_order],
            person=[dict(prediction_record_index=p, target_index=g,
                target_actor_id=str(frame.eligible[g]["gt_actor_id"]), error_m=error)
                for p, g, error in association.person_prediction_order]),
        scoring=dict(vehicle_score_threshold=.20, person_score_threshold=.25, person_avo_threshold=.65,
                     match_radius_m=3., vehicle_detection=dict(result.vehicle_detection),
                     segmentation=segmentation, reward_and_timeout="UNCHANGED_CALLER_IMPLEMENTATION"))


def archive_payloads(result, *, legacy_reference=None):
    """Caller owns bounded queue, durability and explicit archive-fault handling."""
    success, encoded = cv2.imencode(".png", result._frame.label_view())
    if not success:
        raise ValueError("prediction PNG encoding failed")
    png = bytes(encoded)
    evidence = reconstruct_evidence(result, legacy_reference=legacy_reference)
    evidence["predicted_labels"]["png_sha256"] = hashlib.sha256(png).hexdigest()
    return {"predicted_labels.png": png, "evaluation.json": canonical(evidence)}
