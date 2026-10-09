"""Adapter to a caller-pinned RewardSpecV1 and protocol-v2 quality formula.

The repository intentionally has no production calibration.  A supplied spec
is a hash-bound, recomputable interpretation of the retained raw sufficient
statistics; it does not make the scalar reward weights selected or validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .. import protocol_v2_contract as protocol_v2
from ..state_reward_transition_contract import LocalizationCombiner, RewardSpecV1
from .contract import (
    FROZEN_QUALITY_CALIBRATION,
    PROVISIONAL_QUALITY_CALIBRATION,
    REGISTERED_FROZEN_REWARD_SPEC_SHA256,
    OfflineGridContractError,
    canonical_json_bytes,
    sha256_file,
)

_SPEC_CONSTRUCTOR_FIELDS = (
    "spec_id",
    "spec_version",
    "w_loc_person",
    "w_loc_vehicle",
    "tau_person_m",
    "tau_vehicle_m",
    "localization_combiner",
    "w_seg_person",
    "w_seg_vehicle",
    "seg_reference_person_iou",
    "seg_reference_vehicle_iou",
    "segmentation_modulation_beta",
    "w_quality",
    "w_latency",
    "lambda_mode",
    "lambda_q",
    "r_registered_failure",
    "gamma_per_tensor",
    "provenance",
)


def reward_spec_from_document(document: Mapping[str, Any]) -> RewardSpecV1:
    """Construct the real registered type, then demand canonical round-trip."""

    if document.get("record") != "reward_spec_v1":
        raise OfflineGridContractError("reward spec has the wrong record tag")
    try:
        combiner = LocalizationCombiner(str(document["localization_combiner"]))
        kwargs = {name: document[name] for name in _SPEC_CONSTRUCTOR_FIELDS}
    except (KeyError, ValueError) as exc:
        raise OfflineGridContractError(f"malformed RewardSpecV1 document: {exc}") from exc
    kwargs["localization_combiner"] = combiner
    spec = RewardSpecV1(**kwargs)
    if spec.to_canonical_dict() != dict(document):
        raise OfflineGridContractError(
            "reward spec document is not exactly RewardSpecV1.to_canonical_dict()"
        )
    return spec


def load_reward_spec(path: Path, expected_sha256: str) -> RewardSpecV1:
    """Load a caller-pinned canonical spec; no calibration default is permitted."""

    path = Path(path)
    if len(expected_sha256) != 64:
        raise OfflineGridContractError("--reward-spec-sha256 must be a 64-hex digest")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineGridContractError("reward spec is not UTF-8 JSON") from exc
    if type(document) is not dict or raw != canonical_json_bytes(document):
        raise OfflineGridContractError(
            "reward spec file must be exact canonical JSON with no trailing newline"
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise OfflineGridContractError(
            f"reward spec SHA-256 drift: expected {expected_sha256}, observed {observed}"
        )
    spec = reward_spec_from_document(document)
    if spec.canonical_sha256() != expected_sha256:
        raise OfflineGridContractError("RewardSpecV1 canonical digest/file digest mismatch")
    return spec


def calibration_status(spec: RewardSpecV1) -> str:
    """Return frozen only for a source-reviewed, explicitly registered hash."""

    if spec.canonical_sha256() in REGISTERED_FROZEN_REWARD_SPEC_SHA256:
        return FROZEN_QUALITY_CALIBRATION
    return PROVISIONAL_QUALITY_CALIBRATION


def _seg_class(measurement: Mapping[str, Any], name: str) -> dict[str, int]:
    return {
        "gt_pixels": int(measurement[f"seg_{name}_gt_pixels"]),
        "pred_pixels": int(measurement[f"seg_{name}_pred_pixels"]),
        "intersection_pixels": int(measurement[f"seg_{name}_intersection_pixels"]),
        "union_pixels": int(measurement[f"seg_{name}_union_pixels"]),
    }


def _loc_class(measurement: Mapping[str, Any], name: str) -> dict[str, Any]:
    eligible = int(measurement[f"loc_{name}_eligible_gt"])
    errors = [float(value) for value in measurement[f"loc_{name}_matched_xy_errors_m"]]
    return {
        "eligible_actor_ids": list(range(eligible)),
        "eligible_gt_instances": eligible,
        "tp": int(measurement[f"loc_{name}_tp"]),
        "fn": int(measurement[f"loc_{name}_fn"]),
        "matched_xy_errors_m": errors,
    }


def evaluate_exact_quality(
    spec: RewardSpecV1, measurement: Mapping[str, Any]
) -> protocol_v2.ExactPerceptionQualityV2 | None:
    """Reuse protocol-v2's exact derivation; undefined support remains ``None``.

    The protocol-v2 formula uses the median of the retained matched-error list.
    That list is therefore a required row field; a mean alone is not accepted as
    sufficient evidence.
    """

    if type(spec) is not RewardSpecV1:
        raise OfflineGridContractError("quality requires an exact RewardSpecV1")
    if not any(int(measurement[f"loc_{name}_eligible_gt"]) > 0 for name in ("vehicle", "person")):
        return None
    zero = "0" * 64
    segmentation_document = {
        "record": "segmentation_sufficient_counts_v2",
        "carla_frame_id": int(measurement["frame_id"]),
        "camera_calibration_sha256": zero,
        "carla_semantic_label_sha256": zero,
        "prediction_segmentation_label_sha256": zero,
        "segmentation_eligibility_mask_sha256": zero,
        "eligibility_contract_sha256": zero,
        "vehicle": _seg_class(measurement, "vehicle"),
        "person": _seg_class(measurement, "person"),
    }
    localization_document = {
        "record": "localization_match_ledger_v2",
        "carla_frame_id": int(measurement["frame_id"]),
        "actor_projection_eligibility_sha256": zero,
        "actor_snapshot_sha256": zero,
        "camera_calibration_sha256": zero,
        "eligibility_contract_sha256": zero,
        "prediction_object_records_sha256": zero,
        "vehicle": _loc_class(measurement, "vehicle"),
        "person": _loc_class(measurement, "person"),
    }
    # This is the registered implementation, not a restatement/interpolator.
    return protocol_v2._derive_exact_quality(  # noqa: SLF001 - reviewed adapter seam
        spec, segmentation_document, localization_document
    )
