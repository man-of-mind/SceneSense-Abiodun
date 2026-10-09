"""Offline component adapter to the coordinator's published immutable DTO.

No wire is defined here. Prediction admission retains every frozen record
field and native calibration contract. Runtime authentication, FrameContext
pose reconciliation and accepted-history transactions remain coordinator work.
Legacy target artifact completion is not a scoring prerequisite.
"""
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

from . import scorer as S

_PREDICTION_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class _AdmittedPrediction:
    issuer: object
    owner: object
    identity: object
    labels: bytes
    shape: tuple
    records: tuple
    raw_records: bytes
    contract: object
    detection_indices: tuple
    person_indices: tuple

    def __init__(self, *args, **kwargs):
        raise TypeError("use admit_prediction")


def admit_prediction(scorer, *, identity, labels, records, prediction_contract):
    """Actual prediction trust boundary; never skip numerical validation."""
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    C.require(type(scorer) is S.NumericalScorer, "exact scorer context required")
    owned_identity = C.thaw(identity)
    capture, action = owned_identity["gt_identity"], owned_identity["action_identity"]
    C._capture(capture)
    MP.validate_run4_identity(action)
    C._policy({key: action[key] for key in C.POLICY_FIELDS}, capture["frame_id"])
    C.require(action["reward_requested"] is True and owned_identity["capture_timestamp_ns"] == capture["capture_timestamp_ns"],
              "reward ticket and capture timestamp required")
    contract = PA.PredictionContract(prediction_contract["kind"], *prediction_contract["shape"],
        prediction_contract["camera_model_sha256"], prediction_contract["camera_mount_sha256"])
    C.require(contract.document() == C.thaw(prediction_contract), "prediction contract differs")
    PA.validate_labels(labels, contract)
    raw = C.thaw(records)
    C.require(type(raw) is list and len(raw) <= 512, "bounded full prediction records required")
    encoded = S.canonical(raw)
    label_bytes = labels.tobytes(order="C")
    C.require(owned_identity.get("prediction_digest") == hashlib.sha256(label_bytes + encoded).hexdigest(),
              "prediction content digest differs")
    for row in raw:
        C.require(row.get("class_name") in ("vehicle", "person"), "unknown prediction class")
        C.require(int(row.get("frame_id", -1)) == capture["frame_id"], "foreign/missing prediction frame")
        C.require(math.isfinite(float(row["score"])) and 0 <= float(row["score"]) <= 1,
                  "invalid calibrated prediction score")
    sample = "private_prediction_admission"
    namespace = dict(scorer._detection, read_csv=lambda _: [dict(row, sample_id=sample) for row in raw])
    grouped, missing = FunctionType(namespace["load_predictions"].__code__, namespace)(Path("/offline/predictions.csv"))
    C.require(not missing, "incomplete full prediction record: " + ", ".join(missing[:4]))
    ordered = sorted(range(len(raw)), key=lambda i: (-float(raw[i]["score"]), str(raw[i]["class_name"])))
    numeric = [None] * len(raw)
    for i, row in zip(ordered, grouped.get(sample, [])):
        numeric[i] = row
    value = _AdmittedPrediction.__new__(_AdmittedPrediction)
    members = dict(issuer=_PREDICTION_ISSUER, owner=scorer._owner, identity=C._freeze(owned_identity),
        labels=label_bytes, shape=tuple(labels.shape), records=S._frozen_rows(numeric), raw_records=encoded,
        contract=contract,
        detection_indices=tuple(i for i in ordered if float(raw[i]["score"]) >= .20),
        person_indices=tuple(i for i, row in enumerate(raw) if row["class_name"] == "person" and float(row["score"]) >= .25))
    for field in fields(_AdmittedPrediction):
        object.__setattr__(value, field.name, members[field.name])
    return value


def join_admitted_inputs(scorer, *, target, prediction):
    """No PNG, file reads, hashing or evidence packaging at this internal seam.

    Converts exact legacy CSV values already in immutable memory into the
    frozen numerical row representation once, before the numerical call.
    No full table restoration, source reads, CSV parser or artifact is used.
    """
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
    from rl_agent.splitfusion_numeric_gt_integration_v1.oracle_bridge import scorer_inputs
    target = C.require_admitted(target)
    C.require(type(prediction) is _AdmittedPrediction and prediction.issuer is _PREDICTION_ISSUER
              and prediction.owner is scorer._owner, "private immutable prediction from this scorer required")
    view = scorer_inputs(target)
    capture, action = prediction.identity["gt_identity"], prediction.identity["action_identity"]
    C.require(capture == target.metadata["capture_identity"]
              and all(action[key] == view["identity"][key] for key in C.POLICY_FIELDS),
              "prediction/numerical target complete identity differs")
    namespace = dict(scorer._detection, read_csv=lambda _: view["object_boxes"])
    grouped, _ = FunctionType(namespace["load_gt"].__code__, namespace)(Path("/offline"), "v010")
    gt = grouped.get(view["sample_id"], [])
    avo = view["avo"]
    eligible_indices = tuple(i for i, row in enumerate(avo["qualified"]) if float(row["avo"]) >= .65)
    ignored_indices = tuple(i for i, row in enumerate(avo["qualified"]) if float(row["avo"]) < .65)
    def actors(rows):
        return S._frozen_rows({k: row[k] for k in ("world_x", "world_y", "gt_actor_id")} for row in rows)
    return S._issue_frame(S._ISSUER, scorer._owner, capture["frame_id"], target.segmentation_bytes,
        target.object_ignore_bytes, prediction.labels, prediction.shape, prediction.records, S._frozen_rows(gt),
        prediction.detection_indices, prediction.person_indices,
        actors(avo["qualified"][i] for i in eligible_indices), actors(avo["qualified"][i] for i in ignored_indices),
        actors(avo["structural_ignored"]), eligible_indices, ignored_indices,
        b"", scorer._source_pins, None, (target, prediction))


def deferred_inputs(frame, legacy_reference):
    """Require the coordinator's source/artifact attachment only after scoring.

    The legacy reference carries a producer-verified old digest; this function
    checks its complete semantic/source/table binding. Artifact byte parity is
    the deferred producer's responsibility and independently tested offline.
    """
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C, evidence as V
    target, prediction = frame.shared_context
    C.require(type(legacy_reference) is V.LegacyArtifactReference, "deferred legacy artifact reference required")
    C.require(legacy_reference.semantic_sha256 == target.semantic_sha256, "foreign numerical artifact reference")
    checked = V.own_legacy_reference(target, legacy_reference.source,
        target_tables=legacy_reference.target_tables, old_prepared_sha256=legacy_reference.old_prepared_sha256)
    # Source attachment retains the complete original camera, AVO and metadata.
    return dict(action_identity=C.thaw(prediction.identity["action_identity"]),
                records=json.loads(prediction.raw_records), target=dict(provenance=C.thaw(target.metadata["provenance"])),
                avo_document=C.thaw(checked.source.avo_document)), checked.old_prepared_sha256
