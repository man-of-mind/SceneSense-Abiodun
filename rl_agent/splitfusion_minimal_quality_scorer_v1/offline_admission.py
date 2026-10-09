"""Private oracle-fixture admission, deliberately NOT a live DTO or protocol.

Restoration here is allowed solely to derive independent numerical fixtures.
The coordinator replaces this seam with shared admitted DTO integration; the
scorer itself does not accept a public 'already_validated' flag. Original full
validation runs before freezing bytes and numerical rows. No producer needed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

import numpy as np

from . import scorer as S

VALIDATOR_PINS = {
    "rl_agent/splitfusion_live_parity_repair_v2/prediction_archive.py": "5e992c47f726ca693272da2561fa8d3ca7837a9242233ddd7ec7b7e25ec84ae6",
    "rl_agent/splitfusion_live_dispatch_v1/frame_context.py": "845c3886f8862ce3f1cd618323b5b7c8baba05b043e5709ff0cd83d36fc33676",
    "rl_agent/splitfusion_live_parity_repair_v1/evaluation.py": "7ea46360ab7eae5ec0ea85bcebc6b70a52da0635e6049c08d878f03a180f6938",
    "rl_agent/splitfusion_live_gt_handoff_v1/scoring_packet.py": "e00a19f455d17f5d87d37e0bda7004917af1941cae7b2e6d15ccacf518fbef59",
    "rl_agent/splitfusion_live_parity_repair_v1/target_transport.py": "3c36cc9dd319fd3c67db753705e890f74a1db1bf3542d9b033921a0db3f7cedb",
    "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_map_protocol_v2.py": "ef6489f266ef3ca523b596edcece445f55b78eb24367a2789b4697089d02a6b5",
}


def verify_fixture_validator_sources():
    for path, expected in VALIDATOR_PINS.items():
        if hashlib.sha256((S.ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("offline fixture validator source changed: " + path)
    return dict(VALIDATOR_PINS)


def require(value, message):
    if not value:
        raise ValueError(message)


def admit_oracle_fixture(scorer, *, prepared, prediction_identity, predicted_labels,
                         predictions, avo_document, input_metadata, run_binding, prediction_contract):
    """Validate a complete offline oracle input, then detach all mutable aliases.

    Caller verifies retained outer file manifests and native input content.
    This adapter additionally verifies the prepared digest, full prediction
    digest, all record fields, packet provenance/metadata/AVO, complete capture
    and action identities, exact retained pose, and numerical admissibility.
    The run binding is retained evidence, not a new semantic payload digest.
    """
    from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T
    from rl_agent.splitfusion_live_gt_handoff_v1 import scoring_packet as P
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
    require(type(scorer) is S.NumericalScorer, "exact scorer context required")
    identity = json.loads(S.canonical(prediction_identity))
    metadata = json.loads(S.canonical(input_metadata))
    avo_document = P._json(S.canonical(avo_document))
    raw = json.loads(S.canonical(list(predictions)))
    require(T._digest(run_binding), "run evidence binding required")
    gt_identity = T.validate_identity(identity["gt_identity"])
    action = identity["action_identity"]
    MP.validate_run4_identity(action)
    require(action["reward_requested"] is True, "held/fallback frame is not a scoring ticket")
    require(identity["capture_timestamp_ns"] == gt_identity["capture_timestamp_ns"], "capture timestamp differs")
    E._identity(action, prepared, avo_document["avo"])
    require(gt_identity["frame_id"] == prepared.frame_id, "capture/action frame differs")
    owned = E.restore_prepared_target(frame_id=prepared.frame_id, sample_id=prepared.sample_id,
        target=prepared.target, content_sha256=prepared.content_sha256)
    P._provenance(owned.target, gt_identity, owned.sample_id)
    P._input_metadata(metadata, gt_identity, owned.target, owned.sample_id)
    P._avo(avo_document, gt_identity)
    native_inverse = np.asarray(P._json(metadata["frame"]["camera_inverse_matrix_json"]), dtype=np.float64)
    avo_inverse = np.asarray(avo_document["camera_inverse"], dtype=np.float64)
    require(native_inverse.shape == avo_inverse.shape == (4, 4)
            and np.isfinite(native_inverse).all() and np.array_equal(native_inverse, avo_inverse),
            "native/AVO exact camera inverse differs")
    # Preserve matrix and intrinsics; no pose restoration inside scoring.
    matrix = np.asarray(avo_document["camera_matrix"], dtype=np.float64)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), "invalid camera matrix")
    shape = prediction_contract["shape"]
    contract = PA.PredictionContract(prediction_contract["kind"], *shape,
        prediction_contract["camera_model_sha256"], prediction_contract["camera_mount_sha256"])
    require(contract.document() == prediction_contract, "prediction contract differs")
    PA.validate_labels(predicted_labels, contract)
    if contract.kind == "NATIVE_CALIBRATED":
        require((int(metadata["frame"]["camera_height"]), int(metadata["frame"]["camera_width"])) == contract.shape,
                "native prediction/target camera dimensions differ")
    label_bytes = predicted_labels.tobytes(order="C")
    require(identity.get("prediction_digest") == hashlib.sha256(label_bytes + S.canonical(raw)).hexdigest(),
            "prediction content digest differs")
    require(len(raw) <= E.MAX_RECORDS, "prediction record bound exceeded")
    for row in raw:
        require(row.get("class_name") in ("vehicle", "person"), "unknown prediction class")
        require(int(row.get("frame_id", -1)) == owned.frame_id, "foreign/missing prediction frame")
        require(math.isfinite(float(row["score"])) and 0 <= float(row["score"]) <= 1,
                "invalid calibrated prediction score")
    # Execute the frozen loaders with isolated in-memory rows, not file handles.
    prediction_path = Path("/private_offline_fixture/predictions.csv")
    scorer_rows = [dict(row, sample_id=owned.sample_id) for row in raw]
    def read_rows(path):
        return scorer_rows if path == prediction_path else owned.target["tables"]["object_boxes.csv"]
    namespace = dict(scorer._detection, read_csv=read_rows)
    load_predictions = FunctionType(namespace["load_predictions"].__code__, namespace)
    load_gt = FunctionType(namespace["load_gt"].__code__, namespace)
    grouped, missing = load_predictions(prediction_path)
    require(not missing, "incomplete full prediction record: " + ", ".join(missing[:4]))
    gt, _ = load_gt(Path("/complete"), "v010")
    ground_truth = gt.get(owned.sample_id, [])
    for row in ground_truth:
        require(row["class_name"] in ("vehicle", "person"), "unknown target class")
        require(all(math.isfinite(v) for v in (row["world_x"], row["world_y"], row["area_px"], *row["box"])),
                "nonfinite localization target")
    ordered = sorted(range(len(raw)), key=lambda i: (-float(raw[i]["score"]), str(raw[i]["class_name"])))
    numeric = [None] * len(raw)
    for index, row in zip(ordered, grouped.get(owned.sample_id, [])):
        numeric[index] = row
    avo = avo_document["avo"]
    eligible_indices = tuple(i for i, row in enumerate(avo["qualified"]) if float(row["avo"]) >= .65)
    ignored_indices = tuple(i for i, row in enumerate(avo["qualified"]) if float(row["avo"]) < .65)
    def actors(rows):
        # Full rows/reasons remain immutable JSON evidence. Numerical rows need
        # only these immutable scalars; list aliases never enter the scorer.
        return S._frozen_rows({k: row[k] for k in ("world_x", "world_y", "gt_actor_id")} for row in rows)
    target = {k: v for k, v in owned.target.items() if k not in
              ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png")}
    evidence = dict(action_identity=action, prediction_identity=identity, target=target,
        records=raw, avo_document=avo_document, input_metadata=metadata, run_binding=run_binding,
        prediction_contract=contract.document(),
        artifact_hashes={name: hashlib.sha256(owned.target[name + "_png"]).hexdigest()
                        for name in ("segmentation", "object_ignore")})
    return S._issue_frame(S._ISSUER, scorer._owner, owned.frame_id,
        owned.target["segmentation"].tobytes(order="C"), owned.target["object_ignore"].tobytes(order="C"),
        label_bytes, tuple(predicted_labels.shape), S._frozen_rows(numeric), S._frozen_rows(ground_truth),
        tuple(i for i in ordered if float(raw[i]["score"]) >= .20),
        tuple(i for i, row in enumerate(raw) if row["class_name"] == "person" and float(row["score"]) >= .25),
        actors(avo["qualified"][i] for i in eligible_indices), actors(avo["qualified"][i] for i in ignored_indices),
        actors(avo["structural_ignored"]), eligible_indices, ignored_indices, S.canonical(evidence),
        scorer._source_pins, owned.content_sha256, None)
