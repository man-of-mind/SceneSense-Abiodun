"""Strict per-frame/mode/q row schema and validators."""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, MutableMapping, Sequence

from ..state_reward_transition_contract import RewardSpecV1
from .contract import (
    CONTRACT_SHA256,
    DEPLOYMENT_CLAIM,
    EVIDENCE_LABEL,
    FROZEN_QUALITY_CALIBRATION,
    FAMILIES,
    Q_E4_GRID,
    QUANTIZERS,
    PROVISIONAL_QUALITY_CALIBRATION,
    REGISTERED_FROZEN_REWARD_SPEC_SHA256,
    SCHEMA_ID,
    SCHEMA_VERSION,
    OfflineGridContractError,
    canonical_sha256,
    datagram_count,
    keep_count,
    mode_inventory,
    sfd1_accounting_identity,
    sfd1_accounting_stream_id,
    SFD1_COMMON_HEADER_BYTES,
    SFD1_CONTEXT_FIXED_BYTES,
    SFD1_CONTEXT_PROTOCOL_VERSION,
    UDP_CHUNK_HEADER_BYTES,
    UDP_CHUNK_BYTES_INCLUDING_HEADER,
)
from .quality import calibration_status, evaluate_exact_quality

CLASS_NAMES = ("vehicle", "person")
SEGMENTATION_FIELDS = tuple(
    f"seg_{name}_{suffix}"
    for name in CLASS_NAMES
    for suffix in (
        "gt_pixels",
        "pred_pixels",
        "intersection_pixels",
        "union_pixels",
        "iou",
        "valid",
        "status",
    )
)
LOCALIZATION_FIELDS = tuple(
    f"loc_{name}_{suffix}"
    for name in CLASS_NAMES
    for suffix in (
        "eligible_gt",
        "tp",
        "fp",
        "fn",
        "ignored_predictions",
        "matched_xy_count",
        "matched_xy_errors_m",
        "matched_xy_sum_m",
        "matched_xy_sum_sq_m2",
        "matched_xy_mean_m",
        "matched_xy_median_m",
        "matched_xy_max_m",
        "recall",
        "valid",
        "status",
    )
)
ROW_FIELDS = (
    "schema",
    "schema_version",
    "evidence_label",
    "deployment_claim",
    "contract_sha256",
    "run_binding_sha256",
    "reward_spec_sha256",
    "selection_manifest_sha256",
    "episode_manifest_sha256",
    "source_binding_sha256",
    "episode_id",
    "sample_id",
    "frame_id",
    "timestamp",
    "grid_split",
    "selection_rank_within_split",
    "inclusion_probability",
    "sampling_weight",
    "mode_id",
    "family",
    "quantizer",
    "q_e4",
    "keep_count",
    "camera_si",
    "camera_si_valid",
    "camera_si_status",
    "radar_p40",
    "radar_p40_valid",
    "radar_p40_status",
    "scientific_inner_payload_bytes",
    "scientific_inner_payload_sha256",
    "sfd1_protocol_version",
    "sfd1_action_id_field",
    "sfd1_action_identity_status",
    "sfd1_stream_id",
    "sfd1_common_header_bytes",
    "sfd1_frame_context_bytes",
    "sfd1_outer_envelope_bytes",
    "total_transmitted_bytes",
    "total_transmitted_sha256",
    "datagram_count",
    "udp_chunk_bytes_including_header",
    "udp_chunk_header_bytes_per_datagram",
    "udp_chunk_header_bytes_total",
    "udp_application_bytes",
) + SEGMENTATION_FIELDS + LOCALIZATION_FIELDS + (
    "q_seg",
    "q_loc",
    "q_perc",
    "quality_valid",
    "quality_status",
    "quality_calibration_status",
    "raw_sufficient_statistics_are_primary",
    "scalar_reward_weights_used_by_extraction",
    "row_key_sha256",
    "row_sha256",
)

_MODE_BY_ID = {
    mode_id: (family, quantizer)
    for mode_id, family, quantizer in mode_inventory()
}


def row_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "sample_id": row["sample_id"],
        "frame_id": row["frame_id"],
        "grid_split": row["grid_split"],
        "mode_id": row["mode_id"],
        "family": row["family"],
        "quantizer": row["quantizer"],
        "q_e4": row["q_e4"],
    }


def expected_row_keys(selection_manifest: Mapping[str, Any]) -> frozenset[str]:
    frames = selection_manifest.get("selected_frames")
    if not isinstance(frames, list):
        raise OfflineGridContractError("selection manifest has no selected frame list")
    keys = frozenset(
        canonical_sha256(
            row_key(
                {
                    **frame,
                    "mode_id": mode_id,
                    "family": family,
                    "quantizer": quantizer,
                    "q_e4": q_e4,
                }
            )
        )
        for frame in frames
        for mode_id, family, quantizer in mode_inventory()
        for q_e4 in Q_E4_GRID
    )
    from .contract import EXPECTED_GRID_ROWS

    if len(keys) != EXPECTED_GRID_ROWS:
        raise OfflineGridContractError(
            f"expected row-key cardinality {len(keys)} != {EXPECTED_GRID_ROWS}"
        )
    return keys


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OfflineGridContractError(f"{field} must be a non-negative exact int")
    return value


def _require_optional_finite(value: Any, field: str, low: float = 0.0) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfflineGridContractError(f"{field} must be numeric or null")
    if not math.isfinite(float(value)) or float(value) < low:
        raise OfflineGridContractError(f"{field} is non-finite or below {low}")


def _validate_transport_accounting(row: Mapping[str, Any]) -> None:
    inner = _require_nonnegative_int(
        row.get("scientific_inner_payload_bytes"), "scientific_inner_payload_bytes"
    )
    outer = _require_nonnegative_int(
        row.get("sfd1_outer_envelope_bytes"), "sfd1_outer_envelope_bytes"
    )
    total = _require_nonnegative_int(
        row.get("total_transmitted_bytes"), "total_transmitted_bytes"
    )
    expected_action, expected_status = sfd1_accounting_identity(
        int(row["mode_id"]), int(row["q_e4"])
    )
    expected_stream = sfd1_accounting_stream_id(
        str(row["episode_id"]), int(row["mode_id"])
    )
    if (
        inner <= 0
        or row.get("sfd1_protocol_version") != SFD1_CONTEXT_PROTOCOL_VERSION
        or row.get("sfd1_action_id_field") != expected_action
        or row.get("sfd1_action_identity_status") != expected_status
        or row.get("sfd1_stream_id") != expected_stream
        or row.get("sfd1_common_header_bytes") != SFD1_COMMON_HEADER_BYTES
        or row.get("sfd1_frame_context_bytes")
        != SFD1_CONTEXT_FIXED_BYTES + len(expected_stream.encode("utf-8"))
        or outer
        != row.get("sfd1_common_header_bytes") + row.get("sfd1_frame_context_bytes")
        or total != inner + outer
        or row.get("datagram_count") != datagram_count(total)
    ):
        raise OfflineGridContractError("SFD1 payload/envelope/datagram accounting drift")
    if row.get("udp_chunk_bytes_including_header") != UDP_CHUNK_BYTES_INCLUDING_HEADER:
        raise OfflineGridContractError("UDP chunk-size accounting drift")
    if (
        row.get("udp_chunk_header_bytes_per_datagram") != UDP_CHUNK_HEADER_BYTES
        or row.get("udp_chunk_header_bytes_total")
        != row.get("datagram_count") * UDP_CHUNK_HEADER_BYTES
        or row.get("udp_application_bytes")
        != total + row.get("udp_chunk_header_bytes_total")
    ):
        raise OfflineGridContractError("UDP application byte accounting drift")
    for name in ("scientific_inner_payload_sha256", "total_transmitted_sha256"):
        digest = row.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise OfflineGridContractError(f"{name} is missing")


def _populate_class_statistics(row: MutableMapping[str, Any], name: str) -> None:
    gt = _require_nonnegative_int(row[f"seg_{name}_gt_pixels"], f"seg_{name}_gt_pixels")
    pred = _require_nonnegative_int(row[f"seg_{name}_pred_pixels"], f"seg_{name}_pred_pixels")
    intersection = _require_nonnegative_int(
        row[f"seg_{name}_intersection_pixels"], f"seg_{name}_intersection_pixels"
    )
    union = _require_nonnegative_int(row[f"seg_{name}_union_pixels"], f"seg_{name}_union_pixels")
    if intersection > min(gt, pred) or union != gt + pred - intersection:
        raise OfflineGridContractError(f"{name} segmentation counts do not reconcile")
    if union == 0:
        row[f"seg_{name}_iou"] = None
        row[f"seg_{name}_valid"] = False
        row[f"seg_{name}_status"] = "UNDEFINED_BOTH_MASKS_EMPTY"
    else:
        row[f"seg_{name}_iou"] = intersection / union
        row[f"seg_{name}_valid"] = True
        row[f"seg_{name}_status"] = "VALID"

    eligible = _require_nonnegative_int(row[f"loc_{name}_eligible_gt"], f"loc_{name}_eligible_gt")
    tp = _require_nonnegative_int(row[f"loc_{name}_tp"], f"loc_{name}_tp")
    fp = _require_nonnegative_int(row[f"loc_{name}_fp"], f"loc_{name}_fp")
    fn = _require_nonnegative_int(row[f"loc_{name}_fn"], f"loc_{name}_fn")
    _require_nonnegative_int(
        row[f"loc_{name}_ignored_predictions"], f"loc_{name}_ignored_predictions"
    )
    if tp + fn != eligible:
        raise OfflineGridContractError(f"{name} TP+FN does not equal eligible GT")
    errors_raw = row[f"loc_{name}_matched_xy_errors_m"]
    if not isinstance(errors_raw, (list, tuple)):
        raise OfflineGridContractError(f"loc_{name}_matched_xy_errors_m must be a list")
    errors = [float(value) for value in errors_raw]
    if len(errors) != tp or any(not math.isfinite(value) or value < 0 for value in errors):
        raise OfflineGridContractError(f"{name} matched XY errors do not match TP")
    row[f"loc_{name}_matched_xy_errors_m"] = errors
    row[f"loc_{name}_matched_xy_count"] = len(errors)
    row[f"loc_{name}_matched_xy_sum_m"] = sum(errors)
    row[f"loc_{name}_matched_xy_sum_sq_m2"] = sum(value * value for value in errors)
    row[f"loc_{name}_matched_xy_mean_m"] = (
        sum(errors) / len(errors) if errors else None
    )
    row[f"loc_{name}_matched_xy_median_m"] = (
        statistics.median(errors) if errors else None
    )
    row[f"loc_{name}_matched_xy_max_m"] = max(errors) if errors else None
    if eligible == 0:
        row[f"loc_{name}_recall"] = None
        row[f"loc_{name}_valid"] = False
        row[f"loc_{name}_status"] = "UNDEFINED_NO_ELIGIBLE_GT"
    else:
        row[f"loc_{name}_recall"] = tp / eligible
        row[f"loc_{name}_valid"] = True
        row[f"loc_{name}_status"] = (
            "VALID" if tp else "VALID_COMPLETE_MISS_NO_MATCHED_XY_ERROR"
        )


def finalize_row(
    partial: Mapping[str, Any], *, reward_spec: RewardSpecV1
) -> dict[str, Any]:
    """Derive every count-dependent/status/quality field, then validate/hash."""

    row: dict[str, Any] = dict(partial)
    row.update(
        {
            "schema": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "evidence_label": EVIDENCE_LABEL,
            "deployment_claim": DEPLOYMENT_CLAIM,
            "contract_sha256": CONTRACT_SHA256,
            "reward_spec_sha256": reward_spec.canonical_sha256(),
            "row_key_sha256": "",
            "row_sha256": "",
        }
    )
    mode_id = row.get("mode_id")
    if mode_id not in _MODE_BY_ID or _MODE_BY_ID[mode_id] != (
        row.get("family"),
        row.get("quantizer"),
    ):
        raise OfflineGridContractError("mode/family/quantizer identity mismatch")
    if row.get("q_e4") not in Q_E4_GRID:
        raise OfflineGridContractError("row q_e4 is off the frozen grid")
    if row.get("keep_count") != keep_count(row["q_e4"]):
        raise OfflineGridContractError("row keep_count drift")
    _validate_transport_accounting(row)
    for name in CLASS_NAMES:
        _populate_class_statistics(row, name)
    quality = evaluate_exact_quality(reward_spec, row)
    if quality is None:
        row.update(
            q_seg=None,
            q_loc=None,
            q_perc=None,
            quality_valid=False,
            quality_status="UNDEFINED_NO_LOCALIZATION_ELIGIBLE_GT",
        )
    else:
        row.update(
            q_seg=quality.q_seg,
            q_loc=quality.q_loc,
            q_perc=quality.q_perc,
            quality_valid=True,
            quality_status="VALID_EXACT_FOR_CALLER_PINNED_REWARD_SPEC",
        )
    row["quality_calibration_status"] = calibration_status(reward_spec)
    row["raw_sufficient_statistics_are_primary"] = True
    row["scalar_reward_weights_used_by_extraction"] = False
    row["row_key_sha256"] = canonical_sha256(row_key(row))
    missing = sorted(set(ROW_FIELDS).difference(row))
    foreign = sorted(set(row).difference(ROW_FIELDS))
    if missing or foreign:
        raise OfflineGridContractError(
            f"row fields mismatch: missing={missing}, foreign={foreign}"
        )
    row["row_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    validate_row(row)
    return row


def validate_row(row: Mapping[str, Any]) -> None:
    if set(row) != set(ROW_FIELDS):
        raise OfflineGridContractError("quality-grid row does not have the exact schema")
    if row["schema"] != SCHEMA_ID or row["schema_version"] != SCHEMA_VERSION:
        raise OfflineGridContractError("quality-grid row schema drift")
    if row["evidence_label"] != EVIDENCE_LABEL or row["deployment_claim"] != DEPLOYMENT_CLAIM:
        raise OfflineGridContractError("evidence/deployment label drift")
    mode_id = row["mode_id"]
    if mode_id not in _MODE_BY_ID or _MODE_BY_ID[mode_id] != (
        row["family"], row["quantizer"]
    ):
        raise OfflineGridContractError("stored row mode identity drift")
    if row["q_e4"] not in Q_E4_GRID or row["keep_count"] != keep_count(row["q_e4"]):
        raise OfflineGridContractError("stored row q/keep-count drift")
    _validate_transport_accounting(row)
    for name in (
        "run_binding_sha256", "reward_spec_sha256", "selection_manifest_sha256",
        "episode_manifest_sha256", "source_binding_sha256",
        "scientific_inner_payload_sha256", "total_transmitted_sha256",
    ):
        if not isinstance(row[name], str) or len(row[name]) != 64:
            raise OfflineGridContractError(f"stored row {name} is not a SHA-256 hex digest")
        try:
            int(row[name], 16)
        except ValueError as exc:
            raise OfflineGridContractError(f"stored row {name} is not hexadecimal") from exc
    recomputed = dict(row)
    for name in CLASS_NAMES:
        _populate_class_statistics(recomputed, name)
    for name in SEGMENTATION_FIELDS + LOCALIZATION_FIELDS:
        if recomputed[name] != row[name]:
            raise OfflineGridContractError(f"stored row derived statistic drift: {name}")
    if row["raw_sufficient_statistics_are_primary"] is not True:
        raise OfflineGridContractError("stored row does not preserve raw-evidence primacy")
    if row["scalar_reward_weights_used_by_extraction"] is not False:
        raise OfflineGridContractError("stored row claims scalar reward use")
    if row["quality_calibration_status"] not in (
        PROVISIONAL_QUALITY_CALIBRATION, FROZEN_QUALITY_CALIBRATION
    ):
        raise OfflineGridContractError("stored row quality-calibration status drift")
    if (
        row["quality_calibration_status"] == FROZEN_QUALITY_CALIBRATION
        and row["reward_spec_sha256"] not in REGISTERED_FROZEN_REWARD_SPEC_SHA256
    ):
        raise OfflineGridContractError("unregistered reward spec claims frozen calibration")
    if row["quality_valid"] is True:
        if row["q_loc"] is None or row["q_perc"] is None:
            raise OfflineGridContractError("valid quality row omits Q_loc/Q_perc")
        if row["quality_status"] != "VALID_EXACT_FOR_CALLER_PINNED_REWARD_SPEC":
            raise OfflineGridContractError("valid quality row status drift")
    elif row["quality_valid"] is False:
        if any(row[name] is not None for name in ("q_seg", "q_loc", "q_perc")):
            raise OfflineGridContractError("undefined quality row carries a Q value")
        if row["quality_status"] != "UNDEFINED_NO_LOCALIZATION_ELIGIBLE_GT":
            raise OfflineGridContractError("undefined quality row status drift")
    else:
        raise OfflineGridContractError("quality_valid must be an exact bool")
    if canonical_sha256(row_key(row)) != row["row_key_sha256"]:
        raise OfflineGridContractError("row key digest drift")
    expected = canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    if expected != row["row_sha256"]:
        raise OfflineGridContractError("row content digest drift")
    for name in ("camera_si", "radar_p40", "q_seg", "q_loc", "q_perc"):
        _require_optional_finite(row[name], name)
    for name in ("q_seg", "q_loc", "q_perc"):
        if row[name] is not None and float(row[name]) > 1.0:
            raise OfflineGridContractError(f"{name} must lie in [0,1]")
    for name in CLASS_NAMES:
        _require_optional_finite(row[f"seg_{name}_iou"], f"seg_{name}_iou")
        _require_optional_finite(row[f"loc_{name}_recall"], f"loc_{name}_recall")
