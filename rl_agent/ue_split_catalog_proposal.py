#!/usr/bin/env python3
"""Build the offline UE SPLIT candidate-catalog proposal.

The proposal consumes the immutable Stage-A review and a separately approved
decision record.  It cannot run CARLA, OAI, inference, training, or policy
code.  Difficult-object evidence is intentionally fail-closed, so this module
never emits a final action catalog or a completion marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


DECISION_SCHEMA = "scenesense.ue_split_catalog_decision.v1"
MANIFEST_SCHEMA = "scenesense.ue_split_catalog_proposal_manifest.v1"
MARKER_SCHEMA = "scenesense.ue_split_catalog_candidate_review.v1"
VERDICT = "PASS_CANDIDATE_PROPOSAL_REVIEW_REQUIRED_NO_RUN_AUTHORITY"


class CatalogProposalError(RuntimeError):
    """Raised when a candidate proposal cannot be reproduced safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    result = frame.copy()
    columns = result.select_dtypes(include=["float32", "float64"]).columns
    result[columns] = result[columns].round(8)
    result.to_csv(path, index=False, lineterminator="\n")


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()
        return sum(1 for _ in handle)


def _resolve_root(decision_path: Path, decision: Mapping[str, Any]) -> Path:
    return (decision_path.parent / str(decision.get("repository_root", "../.."))).resolve()


def load_decision(decision_path: Path) -> dict[str, Any]:
    decision = yaml.safe_load(decision_path.read_text())
    if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA:
        raise CatalogProposalError(f"decision schema must be exactly {DECISION_SCHEMA}")
    if decision.get("decision_status") != "approved_for_candidate_proposal":
        raise CatalogProposalError("decision is not approved for a candidate proposal")
    if decision.get("supervisor_review_status") not in {"pending", "approved"}:
        raise CatalogProposalError("supervisor review status is invalid")
    if set(decision.get("approved_by", [])) != {"Abiodun", "Codex"}:
        raise CatalogProposalError("decision approval must name Abiodun and Codex")

    authority = decision.get("authority", {})
    if authority.get("evidence_reuse_only") is not True:
        raise CatalogProposalError("candidate proposal requires evidence_reuse_only=true")
    prohibited = (
        "new_carla_run",
        "new_oai_run",
        "model_inference",
        "model_training",
        "policy_training",
        "measurement_authorized",
        "final_catalog_freeze",
    )
    enabled = [name for name in prohibited if authority.get(name) is not False]
    if enabled:
        raise CatalogProposalError(f"candidate proposal prohibits authority fields: {enabled}")

    output = decision.get("output", {})
    if output.get("decision_state") != "CANDIDATE_REVIEW_REQUIRED":
        raise CatalogProposalError("output state must be CANDIDATE_REVIEW_REQUIRED")
    if output.get("write_completed_marker") is not False:
        raise CatalogProposalError("candidate proposal cannot write COMPLETED.json")
    if output.get("eligible_action_count") is not None:
        raise CatalogProposalError("candidate proposal cannot predeclare eligible actions")

    service = decision.get("service_contract", {})
    if service.get("id") != "OBJECT_MAP_V1":
        raise CatalogProposalError("service contract must be OBJECT_MAP_V1")
    required_outputs = {
        "vehicle_or_pedestrian_class",
        "confidence_score",
        "world_location_xy",
        "source_capture_identity",
        "valid_empty_vs_missing_update",
    }
    if set(service.get("required_outputs", [])) != required_outputs or len(
        service.get("required_outputs", [])
    ) != len(required_outputs):
        raise CatalogProposalError("OBJECT_MAP_V1 required outputs are not exact")
    if service.get("segmentation_output_role") != "secondary_diagnostic":
        raise CatalogProposalError("segmentation role must remain secondary_diagnostic")
    if set(service.get("segmentation_metrics", [])) != {
        "miou",
        "iou_vehicle",
        "iou_person",
    }:
        raise CatalogProposalError("segmentation diagnostic metrics are not exact")
    if service.get("segmentation_is_eligibility_veto") is not False:
        raise CatalogProposalError("segmentation must remain a non-vetoing diagnostic")
    if service.get("world_location_semantics") != (
        "predicted_actor_reference_location_not_mask_centroid"
    ):
        raise CatalogProposalError("world-location semantics are not frozen")

    floor = decision.get("quality_floor", {})
    normal = floor.get("normal", {})
    expected_keys = {
        "recall_vehicle_min",
        "recall_pedestrian_min",
        "precision_vehicle_min",
        "precision_pedestrian_min",
        "xy_mae_vehicle_max_m",
        "xy_mae_pedestrian_max_m",
        "fp_per_frame_max",
    }
    if set(normal) != expected_keys:
        raise CatalogProposalError("normal quality-floor fields are not exact")
    numeric_floor = {name: float(value) for name, value in normal.items()}
    if not all(math.isfinite(value) for value in numeric_floor.values()):
        raise CatalogProposalError("normal quality-floor values must be finite")
    for name in (
        "recall_vehicle_min",
        "recall_pedestrian_min",
        "precision_vehicle_min",
        "precision_pedestrian_min",
    ):
        if not 0.0 <= numeric_floor[name] <= 1.0:
            raise CatalogProposalError(f"normal quality-floor value is out of range: {name}")
    for name in ("xy_mae_vehicle_max_m", "xy_mae_pedestrian_max_m", "fp_per_frame_max"):
        if numeric_floor[name] < 0.0:
            raise CatalogProposalError(f"normal quality-floor value must be nonnegative: {name}")
    preferred = floor.get("preferred_non_gating", {})
    if set(preferred) != {"recall_vehicle_min"}:
        raise CatalogProposalError("preferred non-gating fields are not exact")
    preferred_recall = float(preferred["recall_vehicle_min"])
    if not math.isfinite(preferred_recall) or not (
        numeric_floor["recall_vehicle_min"] <= preferred_recall <= 1.0
    ):
        raise CatalogProposalError("preferred vehicle-recall target is invalid")
    if floor.get("roi_incremental_floor_id") != "prior_reference_exploratory":
        raise CatalogProposalError("ROI-incremental floor is not the approved one")
    if floor.get("threshold_comparison") != "inclusive":
        raise CatalogProposalError("quality thresholds must be inclusive")
    rescue = decision.get("degraded_rescue", {})
    if rescue.get("enabled") is not True:
        raise CatalogProposalError("the approved degraded rescue must remain enabled")
    if set(rescue.get("allowed_normal_gate_exceptions", [])) != {
        "recall_pedestrian_min"
    }:
        raise CatalogProposalError("rescue may except only normal pedestrian recall")
    if set(rescue.get("rescue_minimums", {})) != {"recall_pedestrian_min"}:
        raise CatalogProposalError("rescue minimum fields are not exact")
    rescue_recall = float(rescue["rescue_minimums"]["recall_pedestrian_min"])
    if not math.isfinite(rescue_recall) or not 0.0 <= rescue_recall <= 1.0:
        raise CatalogProposalError("rescue pedestrian-recall minimum is invalid")
    activation = rescue.get("activation_contract", {})
    expected_activation = {
        "only_when_no_normal_action_physically_feasible": True,
        "counts_as_normal_service_success": False,
        "emits_service_debt": True,
        "requires_network_feasibility": True,
        "requires_final_evidence_review": True,
    }
    if activation != expected_activation:
        raise CatalogProposalError("degraded-rescue activation contract drift")

    bins = decision.get("supplemental_object_detail", {}).get("range_bins_m", [])
    if len(bins) != 3:
        raise CatalogProposalError("horizontal-range audit must use exactly three bins")
    previous_upper = 0.0
    for index, bin_spec in enumerate(bins):
        lower = float(bin_spec.get("min_inclusive"))
        upper_raw = bin_spec.get("max_exclusive")
        if not math.isfinite(lower) or not math.isclose(lower, previous_upper):
            raise CatalogProposalError("horizontal-range bins must be finite and contiguous")
        if index < len(bins) - 1:
            upper = float(upper_raw)
            if not math.isfinite(upper) or upper <= lower:
                raise CatalogProposalError("horizontal-range bin upper bound is invalid")
            previous_upper = upper
        elif upper_raw is not None:
            raise CatalogProposalError("final horizontal-range bin must be open-ended")
    return decision


def _validate_parent(
    root: Path, decision: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Path], list[dict[str, Any]]]:
    parent = decision["parent_review"]
    parent_dir = (root / str(parent["path"])).resolve()
    manifest_path = parent_dir / "manifest.json"
    marker_path = parent_dir / "REVIEW_REQUIRED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise CatalogProposalError("parent review bundle is incomplete")
    if _sha256_file(manifest_path) != parent["manifest_sha256"]:
        raise CatalogProposalError("parent manifest hash drift")
    if _sha256_file(marker_path) != parent["marker_sha256"]:
        raise CatalogProposalError("parent review marker hash drift")

    manifest = json.loads(manifest_path.read_text())
    marker = json.loads(marker_path.read_text())
    if manifest.get("decision_state") != "REVIEW_REQUIRED":
        raise CatalogProposalError("parent manifest is not REVIEW_REQUIRED")
    if marker.get("decision_state") != "REVIEW_REQUIRED":
        raise CatalogProposalError("parent marker is not REVIEW_REQUIRED")
    if marker.get("manifest_sha256") != parent["manifest_sha256"]:
        raise CatalogProposalError("parent marker does not bind its manifest")
    if marker.get("verdict") != parent["expected_verdict"]:
        raise CatalogProposalError("parent verdict drift")
    if marker.get("no_run_authority") is not True:
        raise CatalogProposalError("parent review marker lost no-run authority")

    output_paths: dict[str, Path] = {}
    output_entries: list[dict[str, Any]] = []
    for item in manifest.get("outputs", []):
        path = parent_dir / str(item["path"])
        if not path.is_file() or _sha256_file(path) != item["sha256"]:
            raise CatalogProposalError(f"parent output drift: {item.get('path')}")
        if path.suffix == ".csv" and _csv_rows(path) != int(item["rows"]):
            raise CatalogProposalError(f"parent output row-count drift: {item.get('path')}")
        output_paths[str(item["artifact_id"])] = path
        output_entries.append(dict(item))
    required = {
        "ue_split_evidence_pool",
        "ue_split_quality_strata",
        "ue_split_quality_floor_sensitivity",
        "ue_split_profile_regime_screen",
    }
    if required - set(output_paths):
        raise CatalogProposalError("parent manifest omits required Stage-A outputs")
    for prohibited in ("FROZEN.json", "COMPLETED.json", "ue_split_action_catalog.csv"):
        if (parent_dir / prohibited).exists():
            raise CatalogProposalError(f"parent unexpectedly contains {prohibited}")
    return parent_dir, manifest, output_paths, output_entries


def apply_quality_gates(
    catalog: pd.DataFrame,
    sensitivity: pd.DataFrame,
    decision: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply the approved absolute and same-q0 gates to all 72 profiles."""

    floor = decision["quality_floor"]
    normal = floor["normal"]
    metric_columns = (
        "recall_vehicle",
        "recall_pedestrian",
        "precision_vehicle",
        "precision_pedestrian",
        "xy_mae_vehicle_m",
        "xy_mae_pedestrian_m",
        "fp_per_frame",
    )
    missing = sorted(set(metric_columns) - set(catalog.columns))
    if missing:
        raise CatalogProposalError(f"catalog is missing quality columns: {missing}")
    values = catalog.loc[:, metric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CatalogProposalError("catalog contains nonfinite gate metrics")
    if len(catalog) != 72 or catalog["profile_id"].duplicated().any():
        raise CatalogProposalError("Stage-A catalog must contain 72 unique profiles")

    selected = sensitivity.loc[
        sensitivity["sensitivity_floor_id"] == floor["roi_incremental_floor_id"]
    ].copy()
    if len(selected) != len(catalog) or selected["profile_id"].duplicated().any():
        raise CatalogProposalError("selected ROI-incremental screen is not one row/profile")
    if not pd.api.types.is_bool_dtype(selected["roi_incremental_screen_pass"]):
        raise CatalogProposalError("ROI-incremental screen must contain strict booleans")
    incremental = selected.set_index("profile_id")["roi_incremental_screen_pass"]
    if set(incremental.index) != set(catalog["profile_id"]):
        raise CatalogProposalError("ROI-incremental profile set differs from catalog")

    result = catalog.copy()
    inherited = {
        column: f"stage_a_{column}"
        for column in (
            "eligibility_status",
            "exclusion_reason",
            "object_detail_evidence",
        )
        if column in result.columns
    }
    result = result.rename(columns=inherited)
    result["gate_recall_vehicle"] = result["recall_vehicle"] >= float(
        normal["recall_vehicle_min"]
    )
    result["gate_recall_pedestrian"] = result["recall_pedestrian"] >= float(
        normal["recall_pedestrian_min"]
    )
    result["gate_precision_vehicle"] = result["precision_vehicle"] >= float(
        normal["precision_vehicle_min"]
    )
    result["gate_precision_pedestrian"] = result["precision_pedestrian"] >= float(
        normal["precision_pedestrian_min"]
    )
    result["gate_xy_mae_vehicle"] = result["xy_mae_vehicle_m"] <= float(
        normal["xy_mae_vehicle_max_m"]
    )
    result["gate_xy_mae_pedestrian"] = result["xy_mae_pedestrian_m"] <= float(
        normal["xy_mae_pedestrian_max_m"]
    )
    result["gate_fp_per_frame"] = result["fp_per_frame"] <= float(
        normal["fp_per_frame_max"]
    )
    result["gate_roi_incremental"] = result["profile_id"].map(incremental).astype(bool)
    gate_columns = [column for column in result.columns if column.startswith("gate_")]
    result["absolute_quality_pass"] = result[
        [column for column in gate_columns if column != "gate_roi_incremental"]
    ].all(axis=1)
    result["normal_candidate"] = result[gate_columns].all(axis=1)
    result["preferred_vehicle_recall_target"] = result["recall_vehicle"] >= float(
        floor["preferred_non_gating"]["recall_vehicle_min"]
    )
    result["segmentation_used_as_veto"] = False

    reason_names = {
        "gate_recall_vehicle": "recall_vehicle",
        "gate_recall_pedestrian": "recall_pedestrian",
        "gate_precision_vehicle": "precision_vehicle",
        "gate_precision_pedestrian": "precision_pedestrian",
        "gate_xy_mae_vehicle": "xy_mae_vehicle",
        "gate_xy_mae_pedestrian": "xy_mae_pedestrian",
        "gate_fp_per_frame": "fp_per_frame",
        "gate_roi_incremental": "roi_incremental",
    }
    result["normal_failure_reasons"] = result.apply(
        lambda row: "|".join(
            reason_names[column] for column in gate_columns if not bool(row[column])
        ),
        axis=1,
    )
    return result


def _rescue_row(gates: pd.DataFrame, decision: Mapping[str, Any]) -> pd.Series:
    rescue = decision["degraded_rescue"]
    matches = gates.loc[gates["profile_id"] == rescue["profile_id"]]
    if len(matches) != 1:
        raise CatalogProposalError("approved rescue profile is missing or duplicated")
    row = matches.iloc[0]
    if bool(row["normal_candidate"]):
        raise CatalogProposalError("rescue unexpectedly passes the normal service floor")
    failed = set(filter(None, str(row["normal_failure_reasons"]).split("|")))
    if failed != {"recall_pedestrian"}:
        raise CatalogProposalError(f"rescue fails unapproved normal gates: {sorted(failed)}")
    if float(row["recall_pedestrian"]) < float(
        rescue["rescue_minimums"]["recall_pedestrian_min"]
    ):
        raise CatalogProposalError("rescue fails its bounded pedestrian-recall minimum")
    if not bool(row["gate_roi_incremental"]):
        raise CatalogProposalError("rescue fails the approved ROI-incremental screen")
    return row


def _strict_dominance(candidate: pd.DataFrame) -> pd.DataFrame:
    maximize = ("recall_vehicle", "recall_pedestrian", "precision_vehicle", "precision_pedestrian")
    minimize = (
        "payload_bytes_p95",
        "xy_mae_vehicle_m",
        "xy_mae_pedestrian_m",
        "fp_per_frame",
    )
    dominators: dict[str, list[str]] = {str(value): [] for value in candidate["profile_id"]}
    rows = list(candidate.to_dict("records"))
    for target in rows:
        for other in rows:
            if target["profile_id"] == other["profile_id"]:
                continue
            weak = all(float(other[key]) >= float(target[key]) for key in maximize) and all(
                float(other[key]) <= float(target[key]) for key in minimize
            )
            strict = any(float(other[key]) > float(target[key]) for key in maximize) or any(
                float(other[key]) < float(target[key]) for key in minimize
            )
            if weak and strict:
                dominators[str(target["profile_id"])].append(str(other["profile_id"]))
    result = candidate.copy()
    result["strictly_dominated_within_normal"] = result["profile_id"].map(
        lambda value: bool(dominators[str(value)])
    )
    result["strict_dominator_profile_ids"] = result["profile_id"].map(
        lambda value: "|".join(sorted(dominators[str(value)]))
    )
    return result


def _candidate_catalog(
    gates: pd.DataFrame, decision: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal = _strict_dominance(gates.loc[gates["normal_candidate"]].copy())
    normal["candidate_tier"] = "NORMAL"
    normal["candidate_status"] = "AGGREGATE_PASS_DIFFICULT_REVIEW_REQUIRED"
    normal["service_debt_on_use"] = False
    rescue = _rescue_row(gates, decision).to_frame().T
    rescue["strictly_dominated_within_normal"] = False
    rescue["strict_dominator_profile_ids"] = ""
    rescue["candidate_tier"] = "DEGRADED_RESCUE"
    rescue["candidate_status"] = "PROVISIONAL_RESCUE_DIFFICULT_REVIEW_REQUIRED"
    rescue["service_debt_on_use"] = True
    result = pd.concat([normal, rescue], ignore_index=True)
    is_rescue = result["candidate_tier"] == "DEGRADED_RESCUE"
    result["eligibility_status"] = result["candidate_status"]
    result["exclusion_reason"] = np.where(
        is_rescue,
        "PENDING_RESCUE_DIFFICULT_OBJECT_AND_FINAL_ACTIVATION_REVIEW",
        "PENDING_DIFFICULT_OBJECT_AND_CATALOG_BUDGET_REVIEW",
    )
    pinned_detail = {
        str(item["display_profile_id"])
        for item in decision["supplemental_object_detail"]["inputs"]
    }
    result["object_detail_evidence"] = np.select(
        [
            result["display_profile_id"].isin(pinned_detail),
            result["roi_drop_fraction"].astype(float) > 0.5,
        ],
        [
            "PINNED_REPRODUCED_HORIZONTAL_RANGE_SMALL_UNRESOLVED",
            "ABSENT_HIGH_ROI_PER_OBJECT_ROWS",
        ],
        default="AVAILABLE_HISTORICAL_NOT_PINNED_IN_PROPOSAL",
    )
    result["final_eligible"] = False
    result["measurement_authorized"] = False
    result["network_certification"] = "UNRESOLVED"
    result["small_object_status"] = "UNRESOLVED_SOURCE_GT_ABSENT_FN_BOXES_MISSING"
    result["world_location_semantics"] = decision["service_contract"][
        "world_location_semantics"
    ]
    result["segmentation_role"] = "SECONDARY_NON_VETOING"
    activation = decision["degraded_rescue"]["activation_contract"]
    result["activation_only_when_no_normal_feasible"] = np.where(
        is_rescue,
        activation["only_when_no_normal_action_physically_feasible"],
        False,
    )
    result["requires_network_feasibility"] = True
    result["counts_as_normal_service_success"] = ~is_rescue
    result["requires_final_evidence_review"] = True
    result = result.sort_values(
        ["candidate_tier", "payload_bytes_p95", "profile_id"]
    ).reset_index(drop=True)

    roles = {
        item["display_profile_id"]: item["role"]
        for item in decision["diagnostic_shortlist"]["profiles"]
    }
    roles[decision["degraded_rescue"]["display_profile_id"]] = "degraded_rescue"
    shortlist = result.loc[result["display_profile_id"].isin(roles)].copy()
    shortlist["diagnostic_role"] = shortlist["display_profile_id"].map(roles)
    shortlist["shortlist_status"] = "NONBINDING_AUDIT_PRIORITY_NOT_FINAL_CATALOG"
    expected = len(roles)
    if len(shortlist) != expected:
        raise CatalogProposalError("diagnostic shortlist does not resolve exact candidate profiles")
    return result, shortlist.sort_values("payload_bytes_p95").reset_index(drop=True)


def _manifest_input_path(
    root: Path, parent_manifest: Mapping[str, Any], artifact_id: str
) -> tuple[Path, dict[str, Any]]:
    matches = [item for item in parent_manifest.get("inputs", []) if item.get("artifact_id") == artifact_id]
    if len(matches) != 1:
        raise CatalogProposalError(f"parent input {artifact_id} is missing or duplicated")
    item = dict(matches[0])
    path = (root / str(item["path"])).resolve()
    if not path.is_file() or _sha256_file(path) != item["sha256"]:
        raise CatalogProposalError(f"parent input drift: {artifact_id}")
    return path, item


def _range_audit(
    root: Path,
    parent_manifest: Mapping[str, Any],
    evidence_pool: pd.DataFrame,
    decision: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    context_path, context_entry = _manifest_input_path(root, parent_manifest, "frame_context")
    context = pd.read_csv(context_path)
    required_context = {"sample_id", "camera_x", "camera_y"}
    if required_context - set(context.columns) or context["sample_id"].duplicated().any():
        raise CatalogProposalError("frame context cannot support horizontal-range audit")
    context = context.set_index("sample_id")

    input_entries = [context_entry]
    range_rows: list[dict[str, Any]] = []
    reproduction: dict[str, Any] = {}
    source = decision["supplemental_object_detail"]
    bins = source["range_bins_m"]
    catalog_index = evidence_pool.set_index("display_profile_id")
    for spec in source["inputs"]:
        display = str(spec["display_profile_id"])
        if display not in catalog_index.index:
            raise CatalogProposalError(f"object-detail profile not found: {display}")
        path = (root / str(spec["path"])).resolve()
        if not path.is_file() or _sha256_file(path) != spec["sha256"]:
            raise CatalogProposalError(f"object-detail hash drift: {display}")
        input_entries.append(
            {
                "artifact_id": f"object_detail_{display.replace('/', '_').replace('.', 'p')}",
                "role": "supplemental_object_detail",
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
                "bytes": path.stat().st_size,
                "rows": _csv_rows(path),
            }
        )
        detail = pd.read_csv(path)
        required = {
            "sample_id",
            "match_status",
            "gt_class_name",
            "gt_world_x",
            "gt_world_y",
            "gt_bbox_h",
            "gt_bbox_w",
            "global_xy_error_m",
        }
        if required - set(detail.columns):
            raise CatalogProposalError(f"object-detail schema drift: {display}")
        if not set(detail["sample_id"].dropna().astype(str)).issubset(set(context.index.astype(str))):
            raise CatalogProposalError(f"object-detail sample set differs: {display}")
        aggregate = catalog_index.loc[display]
        counts = detail["match_status"].value_counts().to_dict()
        if (
            int(counts.get("tp", 0)) != int(aggregate["tp"])
            or int(counts.get("fp", 0)) != int(aggregate["fp"])
            or int(counts.get("fn", 0)) != int(aggregate["fn"])
        ):
            raise CatalogProposalError(f"object-detail aggregate count mismatch: {display}")
        tp_error_sum = float(
            detail.loc[detail["match_status"] == "tp", "global_xy_error_m"].sum()
        )
        expected_error_sum = float(aggregate["xy_mae_m"]) * int(aggregate["tp"])
        if not math.isclose(tp_error_sum, expected_error_sum, abs_tol=1e-5, rel_tol=1e-8):
            raise CatalogProposalError(f"object-detail localization mismatch: {display}")

        gt = detail.loc[detail["match_status"].isin(["tp", "fn"])].copy()
        if gt[["gt_world_x", "gt_world_y", "gt_class_name"]].isna().any().any():
            raise CatalogProposalError(f"object-detail GT fields are incomplete: {display}")
        joined = context.loc[gt["sample_id"].astype(str)]
        gt["horizontal_range_m"] = np.hypot(
            gt["gt_world_x"].to_numpy(dtype=float) - joined["camera_x"].to_numpy(dtype=float),
            gt["gt_world_y"].to_numpy(dtype=float) - joined["camera_y"].to_numpy(dtype=float),
        )
        missing_boxes = int(gt[["gt_bbox_h", "gt_bbox_w"]].isna().any(axis=1).sum())
        reproduction[display] = {
            "tp": int(counts.get("tp", 0)),
            "fp": int(counts.get("fp", 0)),
            "fn": int(counts.get("fn", 0)),
            "gt_rows_missing_bbox": missing_boxes,
            "status": "PASS_COUNTS_AND_LOCALIZATION_REPRODUCED",
        }
        for bin_spec in bins:
            lower = float(bin_spec["min_inclusive"])
            upper_value = bin_spec.get("max_exclusive")
            mask = gt["horizontal_range_m"] >= lower
            if upper_value is not None:
                mask &= gt["horizontal_range_m"] < float(upper_value)
            subset = gt.loc[mask]
            for class_name in ("vehicle", "person"):
                class_rows = subset.loc[subset["gt_class_name"] == class_name]
                tp_rows = class_rows.loc[class_rows["match_status"] == "tp"]
                n_gt = len(class_rows)
                range_rows.append(
                    {
                        "display_profile_id": display,
                        "profile_id": aggregate["profile_id"],
                        "range_bin": bin_spec["id"],
                        "class_name": class_name,
                        "n_gt": n_gt,
                        "tp": len(tp_rows),
                        "fn": int((class_rows["match_status"] == "fn").sum()),
                        "recall": float(len(tp_rows) / n_gt) if n_gt else np.nan,
                        "xy_mae_m": float(tp_rows["global_xy_error_m"].mean())
                        if len(tp_rows)
                        else np.nan,
                        "gt_rows_missing_bbox": int(
                            class_rows[["gt_bbox_h", "gt_bbox_w"]].isna().any(axis=1).sum()
                        ),
                        "precision_by_range_status": "NOT_DEFINED_FP_HAS_NO_GT_RANGE",
                        "small_object_status": "UNRESOLVED_FN_BOXES_MISSING",
                        "evidence_status": "EXACT_RETAINED_OBJECT_DETAIL_NO_INFERENCE",
                    }
                )
    ranges = pd.DataFrame(range_rows).sort_values(
        ["display_profile_id", "range_bin", "class_name"]
    ).reset_index(drop=True)
    for display, aggregate in catalog_index.loc[
        [str(item["display_profile_id"]) for item in source["inputs"]]
    ].iterrows():
        profile_ranges = ranges.loc[ranges["display_profile_id"] == display]
        expected_by_class = {
            "vehicle": int(aggregate["n_gt_vehicle"]),
            "person": int(aggregate["n_gt_pedestrian"]),
        }
        for class_name, expected_count in expected_by_class.items():
            observed = int(
                profile_ranges.loc[
                    profile_ranges["class_name"] == class_name, "n_gt"
                ].sum()
            )
            if observed != expected_count:
                raise CatalogProposalError(
                    f"horizontal-range bins do not partition {display}/{class_name} GT"
                )

    comparison_rows: list[dict[str, Any]] = []
    parsed = ranges.copy()
    parsed[["model", "quant", "roi"]] = parsed["display_profile_id"].str.split("/", expand=True)
    for _, row in parsed.iterrows():
        if row["roi"] == "q0":
            continue
        baseline_display = f"{row['model']}/{row['quant']}/q0"
        base = parsed.loc[
            (parsed["display_profile_id"] == baseline_display)
            & (parsed["range_bin"] == row["range_bin"])
            & (parsed["class_name"] == row["class_name"])
        ]
        if len(base) != 1:
            raise CatalogProposalError(f"missing range baseline for {row['display_profile_id']}")
        base_row = base.iloc[0]
        comparison_rows.append(
            {
                "display_profile_id": row["display_profile_id"],
                "baseline_display_profile_id": baseline_display,
                "range_bin": row["range_bin"],
                "class_name": row["class_name"],
                "recall": row["recall"],
                "baseline_recall": base_row["recall"],
                "recall_drop": float(base_row["recall"] - row["recall"]),
                "xy_mae_m": row["xy_mae_m"],
                "baseline_xy_mae_m": base_row["xy_mae_m"],
                "xy_mae_increase_m": float(row["xy_mae_m"] - base_row["xy_mae_m"]),
                "decision_status": "DIAGNOSTIC_ONLY_NO_APPROVED_RANGE_GATE",
            }
        )
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["display_profile_id", "range_bin", "class_name"]
    ).reset_index(drop=True)
    return ranges, comparisons, input_entries, reproduction


def _candidate_surfaces(
    candidates: pd.DataFrame,
    strata: pd.DataFrame,
    screen: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tier = candidates.set_index("profile_id")["candidate_tier"]
    profile_ids = set(tier.index)
    candidate_strata = strata.loc[strata["profile_id"].isin(profile_ids)].copy()
    candidate_strata["candidate_tier"] = candidate_strata["profile_id"].map(tier)
    candidate_screen = screen.loc[screen["profile_id"].isin(profile_ids)].copy()
    candidate_screen["candidate_tier"] = candidate_screen["profile_id"].map(tier)
    candidate_screen["measurement_authorized"] = False
    candidate_screen["candidate_network_status"] = "PROVISIONAL_UNRESOLVED_NO_RUN_AUTHORITY"
    if len(candidate_strata) != len(candidates) * 9:
        raise CatalogProposalError("candidate strata is not exactly 9 rows/profile")
    if len(candidate_screen) != len(candidates) * 4:
        raise CatalogProposalError("candidate network screen is not exactly 4 rows/profile")
    return candidate_strata, candidate_screen


def _unresolved(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (1, "approve_catalog_equivalence_or_budget_rule", "human_decision", "PENDING", False, "NONE"),
        (2, "small_object_source_gt", "quality_evidence", "SOURCE_GT_ABSENT_FN_BOXES_MISSING", False, "RESTORE_FROZEN_SOURCE_ARTIFACT"),
        (3, "normal_high_roi_object_detail", "quality_evidence", "Q0P7_MATCH_ROWS_ABSENT", True, "OFFLINE_INFERENCE"),
        (4, "degraded_rescue_object_detail", "quality_evidence", "Q0P9_MATCH_ROWS_ABSENT", True, "OFFLINE_INFERENCE"),
        (5, "parallel_decoder_audit", "model_evidence", "SEPARATE_VERSIONED_WORKSTREAM_PENDING", True, "SEPARATE_OFFLINE_WORKSTREAM"),
        (6, "transport_replay_sequence", "transport_provenance", "UNSELECTED_PENDING_FINAL_N", True, "OFFLINE_ARTIFACT_GENERATION"),
        (7, "fixed_10hz_boundaries", "network_measurement", "NOT_AUTHORIZED_PENDING_FINAL_N", True, "OAI_MEASUREMENT"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "priority",
            "item_id",
            "category",
            "status",
            "new_evidence_generation_required",
            "required_work_type",
        ],
    )
    frame["new_run_authorized"] = False
    frame["measurement_authorized"] = False
    frame["candidate_count_normal"] = int((candidates["candidate_tier"] == "NORMAL").sum())
    frame["candidate_count_rescue"] = int(
        (candidates["candidate_tier"] == "DEGRADED_RESCUE").sum()
    )
    return frame


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return {"git_commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "UNAVAILABLE", "dirty": None}


def _output_entry(path: Path, schema_id: str) -> dict[str, Any]:
    return {
        "artifact_id": path.stem,
        "path": path.name,
        "schema_id": schema_id,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": _csv_rows(path) if path.suffix == ".csv" else None,
    }


def _report(
    decision: Mapping[str, Any],
    gates: pd.DataFrame,
    candidates: pd.DataFrame,
    shortlist: pd.DataFrame,
    ranges: pd.DataFrame,
) -> str:
    normal = candidates.loc[candidates["candidate_tier"] == "NORMAL"]
    rescue = candidates.loc[candidates["candidate_tier"] == "DEGRADED_RESCUE"].iloc[0]
    lines = [
        "# UE SPLIT candidate-catalog proposal",
        "",
        f"**Verdict:** `{VERDICT}`",
        "",
        "This artifact records the approved aggregate quality floor and proposes",
        "candidates using only retained evidence. It is not a final action catalog,",
        "does not authorize a measurement, and cannot launch CARLA/OAI/training.",
        "The point estimates are catalog-development evidence from the frozen offline",
        "set, not live/deployment certification; independent bounded validation remains",
        "required after the final catalog is selected.",
        "",
        "## Approved service decision",
        "",
        "- Service: `OBJECT_MAP_V1`.",
        "- Required object position: predicted actor-reference world XY (not a mask centroid).",
        "- Segmentation IoU remains secondary and cannot veto object-map eligibility.",
        f"- Absolute floor passes {int(gates['absolute_quality_pass'].sum())}/72 profiles.",
        f"- Absolute plus same-q0 screen yields {len(normal)} normal aggregate candidates.",
        f"- Rescue candidate: `{rescue['display_profile_id']}` at "
        f"{float(rescue['payload_bytes_p95']) / 1024:.1f} KiB P95.",
        "",
        "## Nonbinding audit-priority shortlist",
        "",
        "| Tier | Role | Profile | P95 KiB | Veh R | Ped R | mIoU |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in shortlist.iterrows():
        lines.append(
            f"| {row['candidate_tier']} | {row['diagnostic_role']} | "
            f"`{row['display_profile_id']}` | {float(row['payload_bytes_p95']) / 1024:.1f} | "
            f"{float(row['recall_vehicle']):.3f} | {float(row['recall_pedestrian']):.3f} | "
            f"{float(row['miou']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The shortlist is an audit priority only. Exact eight-metric dominance leaves",
            f"{int((~normal['strictly_dominated_within_normal']).sum())}/{len(normal)} normal "
            "candidates non-dominated, so reducing to a final N needs an approved",
            "equivalence/catalog-budget rule rather than post-hoc preference.",
            "",
            "## Difficult-object evidence",
            "",
            "Retained per-object rows reproduce aggregate counts/localization for the",
            "q<=0.5 diagnostic profiles. Horizontal range is auditable; precision by",
            "range is not defined because false positives have no GT range. Small-object",
            "recall remains unresolved because every FN lacks GT box size and the source",
            "dataset is absent.",
            "",
            "| Profile | Class | 30m+ recall | 30m+ XY MAE (m) |",
            "|---|---|---:|---:|",
        ]
    )
    far = ranges.loc[ranges["range_bin"] == "far_30_to_evaluator_limit"]
    for _, row in far.iterrows():
        lines.append(
            f"| `{row['display_profile_id']}` | {row['class_name']} | "
            f"{float(row['recall']):.3f} | {float(row['xy_mae_m']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Required review before final N",
            "",
            "1. Approve an equivalence/catalog-budget rule; do not equate this proposal with N=27.",
            "2. Decide whether q0.7 compact evidence warrants one bounded offline detail regeneration.",
            "3. Keep the q0.9 rescue provisional and service-debt-labelled.",
            "4. Treat small-object certification as unavailable unless the frozen source GT is restored.",
            "5. Only a later, separately approved freeze may create the N-action catalog or OAI boundary plan.",
        ]
    )
    return "\n".join(lines) + "\n"


def assemble_proposal(
    decision_path: Path,
    output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    decision_path = decision_path.resolve()
    decision = load_decision(decision_path)
    root = _resolve_root(decision_path, decision)
    parent_dir, parent_manifest, parent_outputs, parent_output_entries = _validate_parent(
        root, decision
    )

    evidence_pool = pd.read_csv(parent_outputs["ue_split_evidence_pool"])
    sensitivity = pd.read_csv(parent_outputs["ue_split_quality_floor_sensitivity"])
    strata = pd.read_csv(parent_outputs["ue_split_quality_strata"])
    screen = pd.read_csv(parent_outputs["ue_split_profile_regime_screen"])
    gates = apply_quality_gates(evidence_pool, sensitivity, decision)
    candidates, shortlist = _candidate_catalog(gates, decision)
    candidate_strata, candidate_screen = _candidate_surfaces(candidates, strata, screen)
    ranges, range_comparisons, supplemental_inputs, reproduction = _range_audit(
        root, parent_manifest, evidence_pool, decision
    )
    unresolved = _unresolved(candidates)

    if int(gates["absolute_quality_pass"].sum()) != 28:
        raise CatalogProposalError("absolute quality-pass count drift")
    if int((candidates["candidate_tier"] == "NORMAL").sum()) != 26:
        raise CatalogProposalError("normal candidate count drift")
    if int((candidates["candidate_tier"] == "DEGRADED_RESCUE").sum()) != 1:
        raise CatalogProposalError("rescue candidate count drift")

    tracked_inputs: dict[Path, str] = {
        decision_path: _sha256_file(decision_path),
        parent_dir / "manifest.json": decision["parent_review"]["manifest_sha256"],
        parent_dir / "REVIEW_REQUIRED.json": decision["parent_review"]["marker_sha256"],
    }
    assembler_path = Path(__file__).resolve()
    tracked_inputs[assembler_path] = _sha256_file(assembler_path)
    for item in parent_output_entries:
        tracked_inputs[parent_dir / str(item["path"])] = str(item["sha256"])
    for item in supplemental_inputs:
        tracked_inputs[(root / str(item["path"])).resolve()] = str(item["sha256"])
    for path, expected in tracked_inputs.items():
        if _sha256_file(path) != expected:
            raise CatalogProposalError(f"input changed before proposal write: {path}")

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_root = (root / decision["output"]["root"]).resolve()
        final_dir = output_root / f"{timestamp}_candidate"
    else:
        final_dir = output_dir.resolve()
        output_root = final_dir.parent
    output_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise CatalogProposalError(f"refusing to overwrite output: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=output_root))
    try:
        resolved = temporary / "resolved_decision.yaml"
        resolved.write_text(yaml.safe_dump(decision, sort_keys=False))
        assembler_snapshot = temporary / "ASSEMBLER_SNAPSHOT.py"
        shutil.copy2(assembler_path, assembler_snapshot)
        _write_csv(temporary / "ue_split_absolute_quality_gate.csv", gates)
        _write_csv(temporary / "ue_split_candidate_catalog.csv", candidates)
        _write_csv(temporary / "ue_split_audit_priority_shortlist.csv", shortlist)
        _write_csv(temporary / "ue_split_candidate_quality_strata.csv", candidate_strata)
        _write_csv(temporary / "ue_split_candidate_profile_regime_screen.csv", candidate_screen)
        _write_csv(temporary / "ue_split_range_audit.csv", ranges)
        _write_csv(temporary / "ue_split_range_comparison.csv", range_comparisons)
        _write_csv(temporary / "ue_split_unresolved_evidence.csv", unresolved)
        report_path = temporary / "REPORT.md"
        report_path.write_text(_report(decision, gates, candidates, shortlist, ranges))

        schemas = {
            "resolved_decision.yaml": DECISION_SCHEMA,
            "ue_split_absolute_quality_gate.csv": "scenesense.ue_split_absolute_quality_gate.v1",
            "ue_split_candidate_catalog.csv": "scenesense.ue_split_candidate_catalog.v1",
            "ue_split_audit_priority_shortlist.csv": "scenesense.ue_split_audit_priority_shortlist.v1",
            "ue_split_candidate_quality_strata.csv": "scenesense.ue_split_candidate_quality_strata.v1",
            "ue_split_candidate_profile_regime_screen.csv": "scenesense.ue_split_candidate_profile_regime_screen.v1",
            "ue_split_range_audit.csv": "scenesense.ue_split_range_audit.v1",
            "ue_split_range_comparison.csv": "scenesense.ue_split_range_comparison.v1",
            "ue_split_unresolved_evidence.csv": "scenesense.ue_split_candidate_unresolved.v1",
            "REPORT.md": "scenesense.ue_split_catalog_proposal_report.v1",
            "ASSEMBLER_SNAPSHOT.py": "scenesense.ue_split_catalog_proposal_source.v1",
        }
        output_entries = [
            _output_entry(temporary / name, schema) for name, schema in schemas.items()
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "experiment_id": final_dir.name,
            "created_utc": (now or datetime.now(timezone.utc)).isoformat(),
            "decision_state": "CANDIDATE_REVIEW_REQUIRED",
            "verdict": VERDICT,
            "service_contract": {
                "id": "OBJECT_MAP_V1",
                "quality_floor_id": decision["quality_floor"]["id"],
                "required_outputs": decision["service_contract"]["required_outputs"],
                "segmentation_role": "SECONDARY_NON_VETOING",
                "world_location_semantics": decision["service_contract"][
                    "world_location_semantics"
                ],
            },
            "parent": {
                "path": str(decision["parent_review"]["path"]),
                "manifest_sha256": decision["parent_review"]["manifest_sha256"],
                "marker_sha256": decision["parent_review"]["marker_sha256"],
            },
            "decision": {
                "path": str(decision_path.relative_to(root)),
                "sha256": _sha256_file(decision_path),
                "approved_by": decision["approved_by"],
                "supervisor_review_status": decision["supervisor_review_status"],
            },
            "counts": {
                "evidence_pool_profiles": len(gates),
                "absolute_quality_pass": int(gates["absolute_quality_pass"].sum()),
                "normal_candidates": int((candidates["candidate_tier"] == "NORMAL").sum()),
                "rescue_candidates": int(
                    (candidates["candidate_tier"] == "DEGRADED_RESCUE").sum()
                ),
                "final_eligible_actions": None,
                "candidate_profile_regime_rows": len(candidate_screen),
            },
            "degraded_rescue": {
                "profile_id": decision["degraded_rescue"]["profile_id"],
                "activation_contract": decision["degraded_rescue"][
                    "activation_contract"
                ],
                "candidate_count": 1,
                "included_in_normal_candidate_count": False,
            },
            "inputs": [
                {
                    "artifact_id": "decision",
                    "path": str(decision_path.relative_to(root)),
                    "sha256": _sha256_file(decision_path),
                },
                {
                    "artifact_id": "parent_manifest",
                    "path": str((parent_dir / "manifest.json").relative_to(root)),
                    "sha256": decision["parent_review"]["manifest_sha256"],
                },
                {
                    "artifact_id": "parent_review_marker",
                    "path": str((parent_dir / "REVIEW_REQUIRED.json").relative_to(root)),
                    "sha256": decision["parent_review"]["marker_sha256"],
                },
                *supplemental_inputs,
            ],
            "outputs": output_entries,
            "range_reproduction": reproduction,
            "audit": {
                "verdict": "PASS",
                "tests": [
                    {"id": "PARENT_HASH_CHAIN", "status": "PASS"},
                    {"id": "ABSOLUTE_FLOOR", "status": "PASS", "profiles": 28},
                    {"id": "NORMAL_CANDIDATES", "status": "PASS", "profiles": 26},
                    {"id": "RESCUE_SEPARATION", "status": "PASS", "profiles": 1},
                    {"id": "SEGMENTATION_NON_VETO", "status": "PASS"},
                    {"id": "RANGE_REPRODUCTION", "status": "PASS", "profiles": len(reproduction)},
                    {"id": "NO_RUN_AUTHORITY", "status": "PASS"},
                ],
                "warnings": [
                    "No final N is selected.",
                    "Offline quality evidence informed candidate selection; reserve independent bounded validation for performance claims.",
                    "Small-object certification is unresolved because FN boxes/source GT are absent.",
                    "High-ROI q0.7/q0.9 profiles lack retained per-object match rows.",
                    "All network rows remain provisional and unauthorized.",
                ],
            },
            "repository": {
                **_git_state(root),
                "assembler_path": str(assembler_path.relative_to(root)),
                "assembler_sha256": _sha256_file(assembler_path),
                "assembler_snapshot_path": assembler_snapshot.name,
                "assembler_snapshot_sha256": _sha256_file(assembler_snapshot),
            },
        }
        manifest_path = temporary / "manifest.json"
        _json_dump(manifest_path, manifest)
        marker = {
            "schema": MARKER_SCHEMA,
            "decision_state": "CANDIDATE_REVIEW_REQUIRED",
            "verdict": VERDICT,
            "manifest_sha256": _sha256_file(manifest_path),
            "normal_candidate_count": 26,
            "rescue_candidate_count": 1,
            "eligible_action_count": None,
            "measurement_authorized": False,
            "no_run_authority": True,
            "next_action": "review_catalog_equivalence_and_difficult_evidence_before_final_N",
        }
        _json_dump(temporary / "CANDIDATE_REVIEW_REQUIRED.json", marker)

        for path, expected in tracked_inputs.items():
            if _sha256_file(path) != expected:
                raise CatalogProposalError(f"input changed during proposal assembly: {path}")
        temporary.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output_dir": str(final_dir),
        "verdict": VERDICT,
        "normal_candidates": 26,
        "rescue_candidates": 1,
        "measurement_authorized": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path("rl_agent/decisions/ue_split_object_map_v1_floor_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = assemble_proposal(args.decision, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
