#!/usr/bin/env python3
"""Assemble the reuse-only Stage-A UE SPLIT evidence sheet.

This module is intentionally offline.  It reads existing CSV/JSON/checkpoint
artifacts, validates the frozen 72-profile factorial evidence, and writes an
immutable review bundle.  It cannot launch CARLA, OAI, inference, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml


CONFIG_SCHEMA = "scenesense.ue_split_stage_a_config.v1"
MANIFEST_SCHEMA = "scenesense.ue_split_evidence_manifest.v1"
REVIEW_SCHEMA = "scenesense.ue_split_stage_a_review_required.v1"
VERDICT = "PASS_EVIDENCE_ASSEMBLED_REVIEW_REQUIRED_NO_RUN_AUTHORITY"


class StageAError(RuntimeError):
    """Raised when frozen Stage-A evidence fails closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_sample_hash(sample_ids: Iterable[str]) -> str:
    return _sha256_text("".join(f"{value}\n" for value in sample_ids))


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _finite_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _quant_short(quant: str) -> str:
    mapping = {
        "per_channel_uint8": "u8",
        "per_channel_uint6": "u6",
        "per_channel_uint4": "u4",
    }
    try:
        return mapping[quant]
    except KeyError as exc:
        raise StageAError(f"unknown quantizer: {quant}") from exc


def _roi_text(roi: float) -> str:
    if math.isclose(roi, 0.0, abs_tol=1e-12):
        return "0"
    return f"{roi:.2f}".rstrip("0").rstrip(".")


def canonical_profile_id(
    model: str,
    quant: str,
    roi: float,
    entropy_level: int,
    checkpoint_sha256: str,
) -> str:
    return (
        f"{model}__{_quant_short(quant)}__q{_roi_text(roi)}"
        f"__zstd{entropy_level}__ckpt{checkpoint_sha256[:12]}"
    )


def display_profile_id(model: str, quant: str, roi: float) -> str:
    return f"{model}/{_quant_short(quant)}/q{_roi_text(roi)}"


def load_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise StageAError(f"config schema must be exactly {CONFIG_SCHEMA}")
    authority = config.get("authority", {})
    if authority.get("evidence_reuse_only") is not True:
        raise StageAError("Stage A requires evidence_reuse_only=true")
    prohibited = (
        "new_carla_run",
        "new_oai_run",
        "model_inference",
        "model_training",
        "policy_training",
        "profile_eligibility_freeze",
    )
    enabled = [name for name in prohibited if authority.get(name) is not False]
    if enabled:
        raise StageAError(f"offline Stage A prohibits authority fields: {enabled}")
    if config.get("output", {}).get("decision_state") != "REVIEW_REQUIRED":
        raise StageAError("initial Stage A decision_state must be REVIEW_REQUIRED")
    if config.get("output", {}).get("write_completed_marker") is not False:
        raise StageAError("REVIEW_REQUIRED assembly cannot write COMPLETED.json")
    service = config.get("service_contract", {})
    if service.get("id") != "OBJECT_MAP_V1":
        raise StageAError("Stage A service contract must be OBJECT_MAP_V1")
    if service.get("quality_floor_id") is not None:
        raise StageAError("initial Stage A must not preselect a quality floor")
    return config


def _resolve_repo_root(config_path: Path, config: Mapping[str, Any]) -> Path:
    value = str(config.get("repository_root", "../.."))
    return (config_path.parent / value).resolve()


def _artifact(
    root: Path,
    artifact_id: str,
    role: str,
    relative_path: str,
    expected_sha256: str,
) -> dict[str, Any]:
    path = (root / relative_path).resolve()
    if not path.is_file():
        raise StageAError(f"missing {role} artifact: {relative_path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise StageAError(
            f"{artifact_id} hash drift: expected {expected_sha256}, got {actual}"
        )
    rows: int | None = None
    schema_sha256: str | None = None
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\r\n")
            rows = sum(1 for _ in handle)
        schema_sha256 = _sha256_text(header)
    return {
        "artifact_id": artifact_id,
        "role": role,
        "path": relative_path,
        "sha256": actual,
        "bytes": path.stat().st_size,
        "rows": rows,
        "schema_sha256": schema_sha256,
        "provenance_status": "PASS",
        "_resolved_path": path,
    }


def collect_inputs(
    root: Path, config_path: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    artifacts: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}

    def add(artifact_id: str, role: str, path: str, digest: str) -> None:
        item = _artifact(root, artifact_id, role, path, digest)
        paths[artifact_id] = item.pop("_resolved_path")
        artifacts.append(item)

    factors = config["factor_contract"]
    for model, spec in factors["models"].items():
        add(
            f"per_frame_{model}",
            "primary_per_frame",
            spec["per_frame_path"],
            spec["per_frame_sha256"],
        )
        add(
            f"checkpoint_{model}",
            "checkpoint",
            spec["checkpoint_path"],
            spec["checkpoint_sha256"],
        )

    quality = config["quality_evidence"]
    for artifact_id, role, path_key, hash_key in (
        ("eval_settings", "settings", "eval_settings_path", "eval_settings_sha256"),
        (
            "analysis_settings",
            "settings",
            "analysis_settings_path",
            "analysis_settings_sha256",
        ),
        (
            "saved_gate_report",
            "validation",
            "saved_gate_report_path",
            "saved_gate_report_sha256",
        ),
        ("evaluator", "evaluation_code", "evaluator_path", "evaluator_sha256"),
        ("run_log", "provenance_record", "run_log_path", "run_log_sha256"),
        (
            "frame_context",
            "quality_context",
            "frame_context_path",
            "frame_context_sha256",
        ),
        (
            "aggregate_check",
            "derived_check",
            "aggregate_check_path",
            "aggregate_check_sha256",
        ),
    ):
        add(artifact_id, role, quality[path_key], quality[hash_key])
    for split_name, split_spec in quality["split_manifests"].items():
        for model, relative_path in split_spec["paths"].items():
            add(
                f"split_{split_name}_{model}",
                "dataset_split",
                relative_path,
                split_spec["sha256"],
            )

    network = config["network_evidence"]
    add(
        "network_summary",
        "network_summary",
        network["summary_path"],
        network["summary_sha256"],
    )
    for source in network["transport_sources"]:
        add(source["id"], "network_transport", source["path"], source["sha256"])

    stale = config["staleness_evidence"]
    for artifact_id, role, path_key, hash_key in (
        ("aoi_budget", "staleness", "budget_path", "budget_sha256"),
        (
            "latency_anchors",
            "staleness",
            "latency_anchors_path",
            "latency_anchors_sha256",
        ),
        (
            "direct_latency_summary",
            "staleness",
            "direct_latency_summary_path",
            "direct_latency_summary_sha256",
        ),
        (
            "error_sensitivity",
            "staleness",
            "error_sensitivity_path",
            "error_sensitivity_sha256",
        ),
    ):
        add(artifact_id, role, stale[path_key], stale[hash_key])

    config_item = {
        "artifact_id": "stage_a_config",
        "role": "config",
        "path": str(config_path.resolve().relative_to(root)),
        "sha256": _sha256_file(config_path.resolve()),
        "bytes": config_path.stat().st_size,
        "rows": None,
        "schema_sha256": None,
        "provenance_status": "PASS",
    }
    artifacts.append(config_item)
    paths["stage_a_config"] = config_path.resolve()
    return artifacts, paths


PROFILE_COLUMNS = {
    "model",
    "ae_bottleneck",
    "quant",
    "roi",
    "sample_id",
    "frame_id",
    "n_inview",
    "n_inview_veh",
    "n_inview_ped",
    "payload_bytes",
    "n_pred",
    "tp",
    "fp",
    "fn",
    "tp_veh",
    "fp_veh",
    "fn_veh",
    "tp_ped",
    "fp_ped",
    "fn_ped",
    "loc_err_sum",
    "loc_err_sq_sum",
    "loc_err_sum_veh",
    "loc_err_sum_ped",
    *(f"conf_{row}{column}" for row in range(3) for column in range(3)),
}


def _validate_eval_settings(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    settings = json.loads(path.read_text())
    knobs = settings.get("eval_knobs", {})
    expected = config["quality_evidence"]["expected_eval_settings"]
    actual = {
        "object_score_threshold": knobs.get("object_score_threshold"),
        "object_nms_radius_px": knobs.get("object_nms_radius_px"),
        "topk_objects": knobs.get("topk_objects"),
        "match_distance_m": knobs.get("match_distance_m"),
        "max_gt_distance_m": knobs.get("max_gt_distance_m"),
        "min_gt_area_px": settings.get("min_gt_area_px"),
    }
    if actual != expected:
        raise StageAError(f"evaluation settings drift: {actual} != {expected}")
    factors = config["factor_contract"]
    if settings.get("entropy_coder") != factors["entropy_coder"]:
        raise StageAError("entropy coder drift")
    if settings.get("zstd_level") != factors["entropy_level"]:
        raise StageAError("entropy level drift")
    return settings


def load_and_validate_profiles(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    factors = config["factor_contract"]
    frames: list[pd.DataFrame] = []
    ordered_samples: list[str] | None = None
    for model in factors["models"]:
        frame = pd.read_csv(paths[f"per_frame_{model}"])
        missing = sorted(PROFILE_COLUMNS - set(frame.columns))
        if missing:
            raise StageAError(f"{model} per-frame schema missing columns: {missing}")
        if len(frame) != int(factors["expected_rows_per_model"]):
            raise StageAError(f"{model} has {len(frame)} rows")
        if set(frame["model"].unique()) != {model}:
            raise StageAError(f"{model} file contains wrong model labels")
        this_order = frame["sample_id"].drop_duplicates().astype(str).tolist()
        if ordered_samples is None:
            ordered_samples = this_order
        elif this_order != ordered_samples:
            raise StageAError(f"{model} ordered sample set differs")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    if len(data) != int(factors["expected_total_profile_frames"]):
        raise StageAError("combined profile-frame row count drift")
    if set(data["quant"].unique()) != set(factors["quantizers"]):
        raise StageAError("quantizer factor grid drift")
    actual_rois = sorted(float(value) for value in data["roi"].unique())
    expected_rois = sorted(float(value) for value in factors["roi_drop_fractions"])
    if not np.allclose(actual_rois, expected_rois, atol=1e-12, rtol=0):
        raise StageAError(f"ROI factor grid drift: {actual_rois}")
    profile_counts = data.groupby(["model", "quant", "roi"], sort=False).size()
    if len(profile_counts) != int(factors["expected_profiles"]):
        raise StageAError(f"expected 72 profiles, found {len(profile_counts)}")
    if set(profile_counts.unique()) != {int(factors["expected_frames_per_profile"])}:
        raise StageAError("profiles do not all contain exactly 2,162 frames")
    key = ["model", "quant", "roi", "sample_id"]
    if data.duplicated(key).any():
        raise StageAError("duplicate model/quant/roi/sample key")

    count_columns = [
        "n_inview",
        "n_inview_veh",
        "n_inview_ped",
        "payload_bytes",
        "n_pred",
        "tp",
        "fp",
        "fn",
        "tp_veh",
        "fp_veh",
        "fn_veh",
        "tp_ped",
        "fp_ped",
        "fn_ped",
        *(f"conf_{row}{column}" for row in range(3) for column in range(3)),
    ]
    if data[count_columns].isna().any().any():
        raise StageAError("nonfinite count/payload fields")
    if (data[count_columns] < 0).any().any() or (data["payload_bytes"] <= 0).any():
        raise StageAError("negative count or nonpositive payload")
    integer_values = data[count_columns].to_numpy(dtype=float)
    if not np.allclose(integer_values, np.rint(integer_values), atol=0, rtol=0):
        raise StageAError("count/payload fields must be integer-valued")
    if not (data["n_inview"] == data["n_inview_veh"] + data["n_inview_ped"]).all():
        raise StageAError("n_inview class identity failed")
    if not (data["n_pred"] == data["tp"] + data["fp"]).all():
        raise StageAError("n_pred identity failed")
    if not (data["n_inview"] == data["tp"] + data["fn"]).all():
        raise StageAError("GT identity failed")
    for stem in ("tp", "fp", "fn"):
        if not (data[stem] == data[f"{stem}_veh"] + data[f"{stem}_ped"]).all():
            raise StageAError(f"{stem} class identity failed")
    loc_columns = [
        "loc_err_sum",
        "loc_err_sq_sum",
        "loc_err_sum_veh",
        "loc_err_sum_ped",
    ]
    if data[loc_columns].isna().any().any() or (data[loc_columns] < 0).any().any():
        raise StageAError("invalid localization aggregates")

    gt_columns = ["frame_id", "n_inview", "n_inview_veh", "n_inview_ped"]
    if int(data.groupby("sample_id")[gt_columns].nunique().to_numpy().max()) != 1:
        raise StageAError("GT frame/counts differ across profiles")
    confusion_columns = [f"conf_{row}{column}" for row in range(3) for column in range(3)]
    total_pixels = data[confusion_columns].sum(axis=1)
    if int(data.assign(_pixels=total_pixels).groupby("sample_id")["_pixels"].nunique().max()) != 1:
        raise StageAError("segmentation pixel total differs across profiles")

    context = pd.read_csv(paths["frame_context"])
    required_context = {
        "sample_id",
        "frame_id",
        "n_inview",
        "n_inview_veh",
        "n_inview_ped",
        "density_bin",
    }
    if required_context - set(context.columns):
        raise StageAError("frame-context schema drift")
    if len(context) != int(factors["expected_frames_per_profile"]):
        raise StageAError("frame-context row count drift")
    if context["sample_id"].duplicated().any():
        raise StageAError("duplicate frame-context sample_id")
    if context["sample_id"].astype(str).tolist() != ordered_samples:
        raise StageAError("frame context and per-profile order differ")
    expected_hash = config["quality_evidence"]["ordered_test_set_sha256"]
    actual_hash = _ordered_sample_hash(context["sample_id"].astype(str))
    if actual_hash != expected_hash:
        raise StageAError(f"ordered test-set hash drift: {actual_hash}")

    split_lists: dict[str, list[str]] = {}
    for split_name, split_spec in config["quality_evidence"]["split_manifests"].items():
        family_lists: list[list[str]] = []
        for model in factors["models"]:
            values = paths[f"split_{split_name}_{model}"].read_text().splitlines()
            if len(values) != int(split_spec["expected_count"]):
                raise StageAError(f"{model} {split_name} split count drift")
            if len(set(values)) != len(values):
                raise StageAError(f"{model} {split_name} split contains duplicates")
            family_lists.append(values)
        if any(values != family_lists[0] for values in family_lists[1:]):
            raise StageAError(f"model families use different {split_name} splits")
        split_lists[split_name] = family_lists[0]
    if set(split_lists) != {"train", "val", "test"}:
        raise StageAError("split manifest must contain train/val/test")
    if set(split_lists["train"]) & set(split_lists["val"]):
        raise StageAError("train/val identifier overlap")
    if set(split_lists["train"]) & set(split_lists["test"]):
        raise StageAError("train/test identifier overlap")
    if set(split_lists["val"]) & set(split_lists["test"]):
        raise StageAError("val/test identifier overlap")
    if split_lists["test"] != context["sample_id"].astype(str).tolist():
        raise StageAError("retained test split differs from evaluated frame order")
    base = data.loc[data["model"] == next(iter(factors["models"]))]
    base = base.drop_duplicates("sample_id").set_index("sample_id")
    joined = context.set_index("sample_id")
    for column in ("frame_id", "n_inview", "n_inview_veh", "n_inview_ped"):
        if not base[column].sort_index().equals(joined[column].sort_index()):
            raise StageAError(f"frame-context {column} mismatch")

    settings = _validate_eval_settings(paths["eval_settings"], config)
    audit = {
        "profile_count": int(len(profile_counts)),
        "profile_frame_rows": int(len(data)),
        "frames_per_profile": int(factors["expected_frames_per_profile"]),
        "ordered_test_set_sha256": actual_hash,
        "source_dataset_path": settings.get("dataset"),
        "source_dataset_present": bool(Path(str(settings.get("dataset", ""))).is_dir()),
        "split_identifier_disjointness": "PASS",
        "train_split_count": len(split_lists["train"]),
        "val_split_count": len(split_lists["val"]),
        "test_split_count": len(split_lists["test"]),
    }
    return data, context, audit


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    sums = frame[
        [
            "tp",
            "fp",
            "fn",
            "tp_veh",
            "fp_veh",
            "fn_veh",
            "tp_ped",
            "fp_ped",
            "fn_ped",
            "loc_err_sum",
            "loc_err_sq_sum",
            "loc_err_sum_veh",
            "loc_err_sum_ped",
        ]
    ].sum()
    tp, fp, fn = (float(sums[name]) for name in ("tp", "fp", "fn"))
    tp_v, fp_v, fn_v = (float(sums[name]) for name in ("tp_veh", "fp_veh", "fn_veh"))
    tp_p, fp_p, fn_p = (float(sums[name]) for name in ("tp_ped", "fp_ped", "fn_ped"))
    recall = _finite_ratio(tp, tp + fn)
    precision = _finite_ratio(tp, tp + fp)
    recall_v = _finite_ratio(tp_v, tp_v + fn_v)
    precision_v = _finite_ratio(tp_v, tp_v + fp_v)
    recall_p = _finite_ratio(tp_p, tp_p + fn_p)
    precision_p = _finite_ratio(tp_p, tp_p + fp_p)
    confusion = np.array(
        [
            [frame[f"conf_{row}{column}"].sum() for column in range(3)]
            for row in range(3)
        ],
        dtype=float,
    )
    ious: list[float] = []
    for index in range(3):
        union = confusion[index, :].sum() + confusion[:, index].sum() - confusion[index, index]
        ious.append(_finite_ratio(confusion[index, index], union))
    valid_ious = [value for value in ious if math.isfinite(value)]
    f1 = _finite_ratio(2 * precision * recall, precision + recall)
    f1_v = _finite_ratio(2 * precision_v * recall_v, precision_v + recall_v)
    f1_p = _finite_ratio(2 * precision_p * recall_p, precision_p + recall_p)
    return {
        "n_frames": int(len(frame)),
        "n_gt": int(tp + fn),
        "n_gt_vehicle": int(tp_v + fn_v),
        "n_gt_pedestrian": int(tp_p + fn_p),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tp_vehicle": int(tp_v),
        "fp_vehicle": int(fp_v),
        "fn_vehicle": int(fn_v),
        "tp_pedestrian": int(tp_p),
        "fp_pedestrian": int(fp_p),
        "fn_pedestrian": int(fn_p),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "recall_vehicle": recall_v,
        "precision_vehicle": precision_v,
        "f1_vehicle": f1_v,
        "recall_pedestrian": recall_p,
        "precision_pedestrian": precision_p,
        "f1_pedestrian": f1_p,
        "fp_per_frame": _finite_ratio(fp, len(frame)),
        "fp_vehicle_per_frame": _finite_ratio(fp_v, len(frame)),
        "fp_pedestrian_per_frame": _finite_ratio(fp_p, len(frame)),
        "xy_mae_m": _finite_ratio(float(sums["loc_err_sum"]), tp),
        "xy_rmse_m": math.sqrt(_finite_ratio(float(sums["loc_err_sq_sum"]), tp)),
        "xy_mae_vehicle_m": _finite_ratio(float(sums["loc_err_sum_veh"]), tp_v),
        "xy_mae_pedestrian_m": _finite_ratio(float(sums["loc_err_sum_ped"]), tp_p),
        "payload_bytes_mean": float(frame["payload_bytes"].mean()),
        "payload_bytes_p50": float(frame["payload_bytes"].quantile(0.50)),
        "payload_bytes_p95": float(frame["payload_bytes"].quantile(0.95)),
        "payload_bytes_min": int(frame["payload_bytes"].min()),
        "payload_bytes_max": int(frame["payload_bytes"].max()),
        "miou": float(np.mean(valid_ious)) if valid_ious else float("nan"),
        "iou_background": ious[0],
        "iou_vehicle": ious[1],
        "iou_person": ious[2],
    }


def build_profile_tables(
    data: pd.DataFrame,
    context: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = config["factor_contract"]
    rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    context_index = context.set_index("sample_id")
    checkpoints = {
        model: spec["checkpoint_sha256"] for model, spec in factors["models"].items()
    }
    for (model, quant, roi), frame in data.groupby(
        ["model", "quant", "roi"], sort=False
    ):
        checkpoint = checkpoints[str(model)]
        identity = {
            "profile_id": canonical_profile_id(
                str(model), str(quant), float(roi), int(factors["entropy_level"]), checkpoint
            ),
            "display_profile_id": display_profile_id(str(model), str(quant), float(roi)),
            "model_family": str(model),
            "ae_bottleneck": int(frame["ae_bottleneck"].iloc[0]),
            "quantization_mode": str(quant),
            "roi_drop_fraction": float(roi),
            "entropy_coder": factors["entropy_coder"],
            "entropy_level": int(factors["entropy_level"]),
            "checkpoint_sha256": checkpoint,
            "eval_set_sha256": config["quality_evidence"]["ordered_test_set_sha256"],
        }
        row = {**identity, **_aggregate(frame)}
        row.update(
            {
                "evidence_pool_version": factors["evidence_pool_version"],
                "quality_set_id": config["quality_evidence"]["quality_set_id"],
                "quality_frame_count": int(
                    factors["expected_frames_per_profile"]
                ),
                "perception_evidence": "registered_historical_integrated_checkpoint_same_test_set",
                "difficult_object_evidence": config["quality_evidence"]
                ["difficult_object_evidence_status"],
                "object_detail_evidence": (
                    "SUPPLEMENTAL_HISTORICAL_NOT_PINNED_SMALL_FAR_STILL_UNRESOLVED"
                    if float(roi) <= 0.5
                    else "FRAME_AGGREGATE_ONLY_NO_PER_OBJECT_MATCH_ROWS"
                ),
                "segmentation_role": "secondary_diagnostic",
                "provenance_status": "PASS_REGISTERED_HISTORICAL_LINEAGE_NOT_ROW_EMBEDDED",
                "evidence_source_id": f"per_frame_{model}",
                "exclusion_reason": "PENDING_ABSOLUTE_FLOOR_AND_DIFFICULT_OBJECT_EVIDENCE",
                "eligibility_status": "REVIEW_REQUIRED_NO_SELECTED_QUALITY_FLOOR",
            }
        )
        rows.append(row)

        sample_ids = frame["sample_id"].astype(str)
        joined_context = context_index.loc[sample_ids].reset_index(drop=False)
        local = frame.reset_index(drop=True).copy()
        local["density_bin"] = joined_context["density_bin"].astype(str).to_numpy()
        stratum_masks: list[tuple[str, str, pd.Series]] = [
            ("all", "ALL", pd.Series(True, index=local.index)),
            ("object_presence", "nonempty", local["n_inview"] > 0),
            ("class_presence", "vehicle_positive", local["n_inview_veh"] > 0),
            ("class_presence", "pedestrian_positive", local["n_inview_ped"] > 0),
            (
                "class_presence",
                "both_classes",
                (local["n_inview_veh"] > 0) & (local["n_inview_ped"] > 0),
            ),
        ]
        for value in ("0", "1-2", "3-4", "5+"):
            stratum_masks.append(("density_bin", value, local["density_bin"] == value))
        for axis, value, mask in stratum_masks:
            subset = local.loc[mask]
            if subset.empty:
                continue
            strata_rows.append(
                {
                    "profile_id": identity["profile_id"],
                    "display_profile_id": identity["display_profile_id"],
                    "stratum_axis": axis,
                    "stratum_value": value,
                    "evidence_level": "frame_aggregate",
                    **_aggregate(subset),
                }
            )

    catalog = pd.DataFrame(rows).sort_values(
        ["model_family", "quantization_mode", "roi_drop_fraction"]
    ).reset_index(drop=True)
    if len(catalog) != int(factors["expected_profiles"]):
        raise StageAError("catalog row count is not 72")
    if catalog["profile_id"].duplicated().any():
        raise StageAError("canonical profile ID collision")
    strata = pd.DataFrame(strata_rows).sort_values(
        ["profile_id", "stratum_axis", "stratum_value"]
    ).reset_index(drop=True)

    sensitivity_rows: list[dict[str, Any]] = []
    baseline = catalog.loc[catalog["roi_drop_fraction"] == 0].set_index(
        ["model_family", "quantization_mode"]
    )
    for _, row in catalog.iterrows():
        base = baseline.loc[(row["model_family"], row["quantization_mode"])]
        deltas = {
            "recall_drop": float(base["recall"] - row["recall"]),
            "vehicle_recall_drop": float(
                base["recall_vehicle"] - row["recall_vehicle"]
            ),
            "pedestrian_recall_drop": float(
                base["recall_pedestrian"] - row["recall_pedestrian"]
            ),
            "localization_increase_m": float(row["xy_mae_m"] - base["xy_mae_m"]),
            "vehicle_localization_increase_m": float(
                row["xy_mae_vehicle_m"] - base["xy_mae_vehicle_m"]
            ),
            "pedestrian_localization_increase_m": float(
                row["xy_mae_pedestrian_m"] - base["xy_mae_pedestrian_m"]
            ),
            "vehicle_precision_drop": float(
                base["precision_vehicle"] - row["precision_vehicle"]
            ),
            "pedestrian_precision_drop": float(
                base["precision_pedestrian"] - row["precision_pedestrian"]
            ),
            "fp_per_frame_increase": float(
                row["fp_per_frame"] - base["fp_per_frame"]
            ),
            "vehicle_fp_per_frame_increase": float(
                row["fp_vehicle_per_frame"] - base["fp_vehicle_per_frame"]
            ),
            "pedestrian_fp_per_frame_increase": float(
                row["fp_pedestrian_per_frame"] - base["fp_pedestrian_per_frame"]
            ),
            "miou_drop_secondary": float(base["miou"] - row["miou"]),
        }
        for floor in config["quality_evidence"]["sensitivity_floors"]:
            gated_values = [
                deltas["recall_drop"],
                deltas["vehicle_recall_drop"],
                deltas["pedestrian_recall_drop"],
                deltas["localization_increase_m"],
                deltas["fp_per_frame_increase"],
            ]
            finite = all(math.isfinite(value) for value in gated_values)
            screen_pass = finite and (
                deltas["recall_drop"] <= float(floor["max_recall_drop"])
                and deltas["vehicle_recall_drop"]
                <= float(floor["max_class_recall_drop"])
                and deltas["pedestrian_recall_drop"]
                <= float(floor["max_class_recall_drop"])
                and deltas["localization_increase_m"]
                <= float(floor["max_localization_increase_m"])
                and deltas["fp_per_frame_increase"]
                <= float(floor["max_fp_per_frame_increase"])
            )
            recall_pass = finite and deltas["recall_drop"] <= float(
                floor["max_recall_drop"]
            )
            vehicle_recall_pass = finite and deltas[
                "vehicle_recall_drop"
            ] <= float(floor["max_class_recall_drop"])
            pedestrian_recall_pass = finite and deltas[
                "pedestrian_recall_drop"
            ] <= float(floor["max_class_recall_drop"])
            aggregate_localization_pass = finite and deltas[
                "localization_increase_m"
            ] <= float(floor["max_localization_increase_m"])
            aggregate_fp_pass = finite and deltas[
                "fp_per_frame_increase"
            ] <= float(floor["max_fp_per_frame_increase"])
            sensitivity_rows.append(
                {
                    "profile_id": row["profile_id"],
                    "display_profile_id": row["display_profile_id"],
                    "sensitivity_floor_id": floor["id"],
                    "quality_floor_config_sha256": _sha256_text(
                        json.dumps(floor, sort_keys=True, separators=(",", ":"))
                    ),
                    "max_recall_drop": float(floor["max_recall_drop"]),
                    "max_class_recall_drop": float(
                        floor["max_class_recall_drop"]
                    ),
                    "max_localization_increase_m": float(
                        floor["max_localization_increase_m"]
                    ),
                    "max_fp_per_frame_increase": float(
                        floor["max_fp_per_frame_increase"]
                    ),
                    **deltas,
                    "screen_basis": "ROI_INCREMENTAL_VS_SAME_MODEL_QUANT_Q0_NOT_ABSOLUTE_SERVICE_FLOOR",
                    "incremental_recall_gate_pass": bool(recall_pass),
                    "incremental_vehicle_recall_gate_pass": bool(
                        vehicle_recall_pass
                    ),
                    "incremental_pedestrian_recall_gate_pass": bool(
                        pedestrian_recall_pass
                    ),
                    "incremental_aggregate_localization_gate_pass": bool(
                        aggregate_localization_pass
                    ),
                    "incremental_aggregate_fp_gate_pass": bool(aggregate_fp_pass),
                    "roi_incremental_screen_pass": bool(screen_pass),
                    "absolute_object_quality_gate_status": "UNRESOLVED_NO_SELECTED_ABSOLUTE_FLOOR",
                    "class_precision_localization_gate_status": "REPORTED_NOT_GATED_IN_INCREMENTAL_SCREEN",
                    "difficult_object_gate_status": "UNRESOLVED_NOT_USED_TO_PASS",
                    "difficult_nonempty_gate_status": "UNRESOLVED_NO_SELECTED_ABSOLUTE_FLOOR",
                    "quality_gate_status": "REVIEW_REQUIRED",
                    "quality_gate_reason": "INCREMENTAL_SCREEN_ONLY_ABSOLUTE_AND_DIFFICULT_OBJECT_GATES_UNRESOLVED",
                    "segmentation_used_as_veto": False,
                    "final_eligible": False,
                    "decision_status": "EXPLORATORY_ONLY_REVIEW_REQUIRED",
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        ["sensitivity_floor_id", "profile_id"]
    ).reset_index(drop=True)
    return catalog, strata, sensitivity


def _check_registered_density_summary(
    catalog: pd.DataFrame,
    strata: pd.DataFrame,
    registered_path: Path,
) -> dict[str, Any]:
    registered = pd.read_csv(registered_path)
    observed = strata.loc[strata["stratum_axis"].isin(["all", "density_bin"])].copy()
    observed["density_bin"] = observed["stratum_value"]
    observed = observed.merge(
        catalog[["profile_id", "model_family", "quantization_mode", "roi_drop_fraction"]],
        on="profile_id",
        validate="many_to_one",
    )
    registered_metrics = [
        "frames",
        "gt_objs",
        "payload_kb",
        "recall",
        "precision",
        "recall_veh",
        "recall_ped",
        "loc_m",
        "fp_per_frame",
        "miou",
        "iou_bg",
        "veh_iou",
        "person_iou",
    ]
    observed_metrics = [
        "n_frames",
        "n_gt",
        "payload_bytes_mean",
        "recall",
        "precision",
        "recall_vehicle",
        "recall_pedestrian",
        "xy_mae_m",
        "fp_per_frame",
        "miou",
        "iou_background",
        "iou_vehicle",
        "iou_person",
    ]
    registered = registered.rename(
        columns={name: f"reg_{name}" for name in registered_metrics}
    )
    observed = observed.rename(
        columns={name: f"obs_{name}" for name in observed_metrics}
    )
    merged = registered.merge(
        observed,
        left_on=["model", "quant", "roi", "density_bin"],
        right_on=["model_family", "quantization_mode", "roi_drop_fraction", "density_bin"],
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all() or len(merged) != 360:
        raise StageAError("registered density aggregate key mismatch")
    exact_pairs = (("frames", "n_frames"), ("gt_objs", "n_gt"))
    for registered_name, recomputed_name in exact_pairs:
        if not (
            merged[f"reg_{registered_name}"].astype(int)
            == merged[f"obs_{recomputed_name}"].astype(int)
        ).all():
            raise StageAError(f"registered {registered_name} mismatch")
    approximate_pairs = (
        ("payload_kb", "payload_bytes_mean", 1024.0),
        ("recall", "recall", 1.0),
        ("precision", "precision", 1.0),
        ("recall_veh", "recall_vehicle", 1.0),
        ("recall_ped", "recall_pedestrian", 1.0),
        ("loc_m", "xy_mae_m", 1.0),
        ("fp_per_frame", "fp_per_frame", 1.0),
        ("miou", "miou", 1.0),
        ("iou_bg", "iou_background", 1.0),
        ("veh_iou", "iou_vehicle", 1.0),
        ("person_iou", "iou_person", 1.0),
    )
    max_difference = 0.0
    for registered_name, recomputed_name, divisor in approximate_pairs:
        left = merged[f"reg_{registered_name}"].astype(float)
        right = merged[f"obs_{recomputed_name}"].astype(float) / divisor
        difference = np.nanmax(np.abs(left.to_numpy() - right.to_numpy()))
        max_difference = max(max_difference, float(difference))
        if difference > 0.0011:
            raise StageAError(
                f"registered {registered_name} differs by {difference:.6g}"
            )
    return {"rows": 360, "max_absolute_rounded_difference": max_difference}


def build_network_tables(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    network = config["network_evidence"]
    regime_by_label = {
        item["source_label"]: item for item in network["regimes"]
    }
    combined = pd.read_csv(paths["network_summary"])
    transport_frames: list[pd.DataFrame] = []
    for source in network["transport_sources"]:
        frame = pd.read_csv(paths[source["id"]]).copy()
        if set(frame["label"]) != set(regime_by_label):
            raise StageAError(f"network labels drift in {source['id']}")
        frame["transport_source_id"] = source["id"]
        frame["evidence_status"] = source["evidence_kind"]
        frame["combined_payload_label"] = source["combined_payload_label"]
        transport_frames.append(frame)
    raw = pd.concat(transport_frames, ignore_index=True)
    if len(raw) != 12:
        raise StageAError("expected exactly 12 historical transport rows")
    raw["network_regime"] = raw["label"].map(
        lambda value: regime_by_label[value]["id"]
    )
    raw["historical_network_config_id"] = raw["label"].map(
        lambda value: regime_by_label[value]["historical_id"]
    )
    raw = raw.merge(
        combined[["payload", "snr", "mcs", "send_fps", "app_offered_mbps"]],
        left_on=["combined_payload_label"],
        right_on=["payload"],
        how="left",
        suffixes=("", "_combined"),
    )
    # The convenience surface has four rows per payload; align by the regime's
    # achieved MCS rather than relying on row order.
    raw = raw.loc[
        np.isclose(raw["mcs"].astype(float), raw["mcs_p50"].astype(float))
    ].copy()
    if len(raw) != 12 or raw["send_fps"].isna().any():
        raise StageAError("could not join historical achieved send rate")
    raw["target_offer_hz"] = float(network["target_offer_hz"])
    raw["target_rate_match"] = False
    raw["map_update_done_observed"] = False
    raw["map_acceptance_observed"] = False
    raw["aoi_observed"] = False
    raw["directly_measured_at_target_10hz"] = False
    raw["historical_endpoint"] = "edge_reassembly_and_publish_enqueue_only"
    raw["radio_provenance_status"] = "SUMMARY_ONLY_RAW_TTRACER_ABSENT"
    transport_columns = [
        "transport_source_id",
        "run_group",
        "network_regime",
        "historical_network_config_id",
        "evidence_status",
        "payload_p50_kib",
        "attempted_frames",
        "edge_frames",
        "edge_delivery_pct",
        "send_fps",
        "target_offer_hz",
        "target_rate_match",
        "snr_p05_db",
        "snr_p50_db",
        "snr_p95_db",
        "mcs_p50",
        "ul_sched_mbps",
        "bsr_lcg_p95_kib",
        "front_to_edge_p50_ms",
        "front_to_edge_p95_ms",
        "map_update_done_observed",
        "map_acceptance_observed",
        "aoi_observed",
        "directly_measured_at_target_10hz",
        "historical_endpoint",
        "radio_provenance_status",
    ]
    transport = raw[transport_columns].sort_values(
        ["network_regime", "payload_p50_kib"]
    ).reset_index(drop=True)

    uncertainty = float(network["capacity_projection_uncertainty_fraction"])
    regime_rows: list[dict[str, Any]] = []
    for item in network["regimes"]:
        subset = raw.loc[raw["network_regime"] == item["id"]]
        capacity = float(subset["ul_sched_mbps"].max())
        budget_bytes = capacity * 1e6 / (8 * float(network["target_offer_hz"]))
        regime_rows.append(
            {
                "network_regime": item["id"],
                "historical_network_config_id": item["historical_id"],
                "achieved_snr_db_median": float(subset["snr_p50_db"].median()),
                "mcs_median": float(subset["mcs_p50"].median()),
                "capacity_reference_mbps": capacity,
                "capacity_reference_low_mbps": capacity * (1 - uncertainty),
                "capacity_reference_high_mbps": capacity * (1 + uncertainty),
                "capacity_equivalent_payload_10hz_bytes": budget_bytes,
                "capacity_equivalent_payload_10hz_kib": budget_bytes / 1024.0,
                "projection_uncertainty_fraction": uncertainty,
                "evidence_status": "existing_reused_capacity_projection",
                "exact_10hz_certification": False,
                "authoritative_map_update_done": False,
                "radio_provenance_status": "SUMMARY_ONLY_RAW_TTRACER_ABSENT",
            }
        )
    regimes = pd.DataFrame(regime_rows)
    if set(regimes["network_regime"]) != {"clear", "mild", "mid", "poor"}:
        raise StageAError("network regime catalog drift")

    screen_rows: list[dict[str, Any]] = []
    lower_ratio = float(network["screening_load_ratio"]["below_capacity_max"])
    upper_ratio = float(network["screening_load_ratio"]["above_capacity_min"])
    packet = network["packetization_proxy"]
    datagram_bytes = int(packet["max_udp_datagram_bytes"])
    custom_header_bytes = int(packet["custom_chunk_header_bytes"])
    udp_ipv4_header_bytes = int(packet["udp_ipv4_header_bytes"])
    chunk_payload_bytes = datagram_bytes - custom_header_bytes
    if chunk_payload_bytes <= 0:
        raise StageAError("invalid packetization proxy")
    for _, profile in catalog.iterrows():
        feature_bytes = float(profile["payload_bytes_p95"])
        chunks = max(1, math.ceil(feature_bytes / chunk_payload_bytes))
        udp_ip_bytes = feature_bytes + chunks * (
            custom_header_bytes + udp_ipv4_header_bytes
        )
        feature_offered = (
            feature_bytes
            * 8
            * float(network["target_offer_hz"])
            / 1e6
        )
        udp_ip_offered = (
            udp_ip_bytes * 8 * float(network["target_offer_hz"]) / 1e6
        )
        for _, regime in regimes.iterrows():
            ratio = udp_ip_offered / float(regime["capacity_reference_mbps"])
            if ratio <= lower_ratio:
                prediction = "projected_below_capacity"
            elif ratio >= upper_ratio:
                prediction = "projected_above_capacity"
            else:
                prediction = "projected_boundary"
            if udp_ip_offered < float(regime["capacity_reference_low_mbps"]):
                relation = "below_uncertainty_band"
            elif udp_ip_offered > float(regime["capacity_reference_high_mbps"]):
                relation = "above_uncertainty_band"
            else:
                relation = "inside_uncertainty_band"
            screen_rows.append(
                {
                    "profile_id": profile["profile_id"],
                    "display_profile_id": profile["display_profile_id"],
                    "network_regime": regime["network_regime"],
                    "profile_payload_p95_bytes": profile["payload_bytes_p95"],
                    "estimated_chunk_count_p95": chunks,
                    "estimated_udp_ip_bytes_p95": udp_ip_bytes,
                    "target_offer_hz": float(network["target_offer_hz"]),
                    "target_feature_offer_mbps": feature_offered,
                    "estimated_udp_ip_offer_mbps": udp_ip_offered,
                    "lower_layer_onwire_offer_status": packet[
                        "lower_layer_overhead_status"
                    ],
                    "capacity_reference_mbps": regime["capacity_reference_mbps"],
                    "load_ratio": ratio,
                    "screening_prediction": prediction,
                    "capacity_uncertainty_relation": relation,
                    "evidence_status": "composed_projection_not_fixed_10hz_measurement",
                    "directly_measured": False,
                    "network_certification": "UNRESOLVED",
                    "profile_quality_status": "REVIEW_REQUIRED",
                }
            )
    screen = pd.DataFrame(screen_rows).sort_values(
        ["network_regime", "profile_payload_p95_bytes", "profile_id"]
    ).reset_index(drop=True)
    if len(screen) != len(catalog) * 4:
        raise StageAError("provisional profile/regime screen is not 72 x 4")

    boundary_rows: list[dict[str, Any]] = []
    for _, regime in regimes.iterrows():
        subset = screen.loc[screen["network_regime"] == regime["network_regime"]]
        center = float(regime["capacity_reference_mbps"])
        lower = subset.loc[subset["estimated_udp_ip_offer_mbps"] <= center]
        upper = subset.loc[subset["estimated_udp_ip_offer_mbps"] > center]
        if not lower.empty:
            candidate = lower.loc[
                (center - lower["estimated_udp_ip_offer_mbps"]).idxmin()
            ]
            boundary_rows.append(
                {
                    **candidate.to_dict(),
                    "boundary_role": "nearest_at_or_below_feature_payload_proxy",
                    "measurement_authorized": False,
                }
            )
        if not upper.empty:
            candidate = upper.loc[
                (upper["estimated_udp_ip_offer_mbps"] - center).idxmin()
            ]
            boundary_rows.append(
                {
                    **candidate.to_dict(),
                    "boundary_role": "nearest_above_feature_payload_proxy",
                    "measurement_authorized": False,
                }
            )
    boundaries = pd.DataFrame(boundary_rows).sort_values(
        ["network_regime", "boundary_role"]
    ).reset_index(drop=True)
    return regimes, transport, screen, boundaries


def build_unresolved(
    profile_audit: Mapping[str, Any], boundaries: pd.DataFrame
) -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "item_id": "select_object_map_quality_floor",
            "category": "human_decision",
            "status": "REVIEW_REQUIRED",
            "resolution": "Abiodun and supervisor select one absolute object-quality floor after reviewing ROI-incremental sensitivity and absolute catalog metrics.",
            "new_run_required": False,
        },
        {
            "priority": 2,
            "item_id": "difficult_object_certification",
            "category": "quality_evidence",
            "status": "UNRESOLVED_ALL_PROFILES_HIGH_ROI_HAS_ADDITIONAL_DETAIL_GAP",
            "resolution": "All survivors require difficult-object review; high-ROI profiles additionally lack per-object match rows. Restore source GT only for decision-relevant survivors.",
            "new_run_required": False,
        },
        {
            "priority": 3,
            "item_id": "quality_source_dataset_restore",
            "category": "reproducibility",
            "status": (
                "AVAILABLE" if profile_audit["source_dataset_present"] else "SOURCE_DATASET_ABSENT"
            ),
            "resolution": "Restore the frozen source dataset before any fresh profile or ROI evaluation.",
            "new_run_required": False,
        },
        {
            "priority": 4,
            "item_id": "select_transport_replay_sequence",
            "category": "transport_provenance",
            "status": "UNSELECTED",
            "resolution": "Select and hash one retained ordered tensor sequence only after the eligible catalog is known.",
            "new_run_required": False,
        },
        {
            "priority": 5,
            "item_id": "integrated_bundle_compute_timing",
            "category": "compute_evidence",
            "status": "LEGACY_ARCHITECTURE_PROXY_ONLY",
            "resolution": "Remeasure only eligible or decision-relevant integrated bundles if exact compute timing is missing.",
            "new_run_required": False,
        },
        {
            "priority": 6,
            "item_id": "fixed_10hz_map_update_boundaries",
            "category": "network_measurement",
            "status": "PENDING_AFTER_CATALOG_FREEZE",
            "resolution": "Measure at most the nearest decision-relevant profile per regime, then close only the adjacent bracket needed.",
            "new_run_required": True,
        },
        {
            "priority": 7,
            "item_id": "select_service_tolerance",
            "category": "freshness_decision",
            "status": "SENSITIVITY_ONLY",
            "resolution": "Choose a localization/service tolerance after reviewing the historical latency proxy; derive AoI from later accepted-update events.",
            "new_run_required": False,
        },
    ]
    for _, boundary in boundaries.iterrows():
        rows.append(
            {
                "priority": 8,
                "item_id": (
                    f"candidate_{boundary['network_regime']}_{boundary['boundary_role']}"
                ),
                "category": "candidate_only",
                "status": "NOT_AUTHORIZED_PENDING_ELIGIBLE_CATALOG",
                "resolution": boundary["display_profile_id"],
                "new_run_required": False,
            }
        )
    result = pd.DataFrame(rows)
    result["measurement_authorized"] = False
    return result


def _round_float_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    float_columns = result.select_dtypes(include=["float32", "float64"]).columns
    result[float_columns] = result[float_columns].round(8)
    return result


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _round_float_columns(frame).to_csv(path, index=False, lineterminator="\n")


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "UNAVAILABLE", None
    return {"git_commit": commit, "dirty": dirty}


def _output_entry(path: Path, schema_id: str) -> dict[str, Any]:
    rows: int | None = None
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            handle.readline()
            rows = sum(1 for _ in handle)
    return {
        "artifact_id": path.stem,
        "path": path.name,
        "schema_id": schema_id,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _report_text(
    catalog: pd.DataFrame,
    sensitivity: pd.DataFrame,
    regimes: pd.DataFrame,
    boundaries: pd.DataFrame,
    latency_anchors: pd.DataFrame,
    profile_audit: Mapping[str, Any],
) -> str:
    lines = [
        "# UE SPLIT Stage-A evidence report",
        "",
        f"**Verdict:** `{VERDICT}`",
        "",
        "This is a reuse-only evidence assembly. It authorizes no CARLA/OAI run,",
        "profile freeze, controller implementation, or training.",
        "",
        "## What passed",
        "",
        f"- {len(catalog)} unique measured profiles over {profile_audit['frames_per_profile']:,} identical frames each.",
        f"- {profile_audit['profile_frame_rows']:,} profile-frame rows passed the frozen grid and count identities.",
        "- All four registered integrated checkpoints and primary evidence files independently match their pinned SHA-256 values.",
        "- The retained run record names those checkpoints, but the CSV rows do not embed a cryptographic checkpoint lineage field.",
        "- Object detection/localization are primary; segmentation is reported only as a secondary diagnostic.",
        "",
        "## Exploratory quality sensitivity",
        "",
        "These ROI-incremental counts compare each profile with its same-model/",
        "same-quantization q=0 baseline. They do not apply an absolute service floor,",
        "are not final eligibility decisions, and do not certify missing small/far strata.",
        "",
        "| Sensitivity floor | ROI-incremental profiles passing |",
        "|---|---:|",
    ]
    counts = sensitivity.groupby("sensitivity_floor_id")[
        "roi_incremental_screen_pass"
    ].sum()
    for floor, count in counts.items():
        lines.append(f"| `{floor}` | {int(count)} / {len(catalog)} |")
    lines.extend(
        [
            "",
            "## Existing network capacity projections",
            "",
            "The historical runs achieved about 5.8--8.0 sends/s and did not log",
            "authoritative map update-done events. The 10-Hz values below are capacity",
            "projections with a registered engineering uncertainty of plus/minus 30%,",
            "not direct action certifications.",
            "",
            "| Regime | Historical ID | SNR (dB) | MCS | Capacity (Mbps) | 10-Hz equivalent (KiB/frame) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in regimes.iterrows():
        lines.append(
            f"| {row['network_regime']} | `{row['historical_network_config_id']}` | "
            f"{row['achieved_snr_db_median']:.1f} | {row['mcs_median']:.0f} | "
            f"{row['capacity_reference_mbps']:.2f} | "
            f"{row['capacity_equivalent_payload_10hz_kib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Existing staleness evidence",
            "",
            "Direct loopback records do reach `map_update_done`; they provide capture-to-map",
            "latency anchors, not profile-specific OAI AoI. The historical localization budget",
            "is emitted separately as a fixed-floor latency-tolerance proxy and cannot select",
            "`AoI_max` without later accepted-update events.",
            "",
            "| Anchor | Evidence | P50 (ms) | P95 (ms) |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in latency_anchors.iterrows():
        lines.append(
            f"| `{row['anchor_id']}` | {row['evidence_kind']} | "
            f"{float(row['latency_p50_ms']):.1f} | "
            f"{float(row['latency_p95_ms']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Provisional payload-boundary candidates",
            "",
            "These are mechanically nearest to each projected capacity center. They",
            "are not approved measurements and may disappear once the quality floor",
            "freezes the eligible catalog. The ratio includes estimated custom/UDP/IPv4",
            "overhead but not GTP, PDCP, RLC, MAC, or scheduling overhead, so below/above",
            "remains only a feature-payload proxy.",
            "",
            "| Regime | Role | Profile | P95 feature payload (KiB) | UDP/IP proxy load ratio |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in boundaries.iterrows():
        lines.append(
            f"| {row['network_regime']} | {row['boundary_role']} | "
            f"`{row['display_profile_id']}` | "
            f"{float(row['profile_payload_p95_bytes']) / 1024:.1f} | "
            f"{float(row['load_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Required decision before measurements",
            "",
            "1. Review the object-quality sensitivity and missing difficult-object evidence.",
            "2. Select and record one absolute `OBJECT_MAP_V1` quality floor.",
            "3. Form N aggregate candidates, then resolve required difficult-object evidence for the survivors.",
            "4. Freeze only fully supported eligible actions in a new immutable sibling bundle.",
            "5. Recompute the exact 4N logical surface and authorize only remaining boundary cells.",
            "",
            "No `COMPLETED.json` or action catalog is written in this review bundle.",
        ]
    )
    return "\n".join(lines) + "\n"


def assemble(
    config_path: Path,
    output_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    root = _resolve_repo_root(config_path, config)
    inputs, input_paths = collect_inputs(root, config_path, config)
    before_hashes = {artifact_id: _sha256_file(path) for artifact_id, path in input_paths.items()}

    data, context, profile_audit = load_and_validate_profiles(config, input_paths)
    catalog, strata, sensitivity = build_profile_tables(data, context, config)
    aggregate_check = _check_registered_density_summary(
        catalog, strata, input_paths["aggregate_check"]
    )
    regimes, transport, screen, boundaries = build_network_tables(
        catalog, config, input_paths
    )
    latency_proxy = pd.read_csv(input_paths["aoi_budget"]).copy()
    allowed_tolerances = set(float(value) for value in config["staleness_evidence"]["tolerance_m"])
    if set(float(value) for value in latency_proxy["eps_m"].unique()) != allowed_tolerances:
        raise StageAError("historical latency-tolerance grid drift")
    latency_proxy.insert(0, "source_id", "registered_uplink_only_staleness_budget")
    latency_proxy["metric_kind"] = "historical_capture_to_map_latency_tolerance_proxy"
    latency_proxy["historical_fixed_floor_m"] = float(
        config["staleness_evidence"]["historical_base_localization_floor_m"]
    )
    latency_proxy["profile_specific"] = False
    latency_proxy["aoi_observed"] = False
    latency_proxy["selected_service_tolerance"] = False
    latency_proxy["decision_status"] = "SENSITIVITY_ONLY_NOT_AN_AOI_SELECTION"

    fresh = pd.read_csv(input_paths["direct_latency_summary"])
    direct_rows = []
    for _, row in fresh.iterrows():
        direct_rows.append(
            {
                "anchor_id": f"fresh_{row['condition']}",
                "source_id": "fresh_L_by_condition",
                "evidence_kind": "direct_capture_to_map_update_done",
                "n_frames": int(row["n_frames"]),
                "latency_p50_ms": float(row["L_p50_ms"]),
                "latency_p95_ms": float(row["L_p95_ms"]),
                "profile_specific": False,
                "map_update_done_observed": True,
                "note": "fresh uplink-only loopback condition",
            }
        )
    registered_anchors = pd.read_csv(input_paths["latency_anchors"])
    for _, row in registered_anchors.iterrows():
        direct_rows.append(
            {
                "anchor_id": str(row["anchor"]),
                "source_id": "registered_L_anchors",
                "evidence_kind": "registered_historical_latency_anchor",
                "n_frames": np.nan,
                "latency_p50_ms": float(row["L_p50_ms"]),
                "latency_p95_ms": float(row["L_p95_ms"]),
                "profile_specific": False,
                "map_update_done_observed": True,
                "note": str(row["note"]),
            }
        )
    latency_anchors = pd.DataFrame(direct_rows)
    error_sensitivity = pd.read_csv(input_paths["error_sensitivity"]).copy()
    error_sensitivity.insert(0, "source_id", "registered_error_vs_L_by_speed")
    error_sensitivity["profile_specific"] = False
    error_sensitivity["aoi_observed"] = False
    error_sensitivity["evidence_role"] = "historical_localization_error_vs_imposed_lag"
    unresolved = build_unresolved(profile_audit, boundaries)

    after_hashes = {artifact_id: _sha256_file(path) for artifact_id, path in input_paths.items()}
    if after_hashes != before_hashes:
        raise StageAError("source artifact changed during Stage-A assembly")

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_root = (root / config["output"]["root"]).resolve()
        final_dir = output_root / f"{timestamp}_review"
    else:
        final_dir = output_dir.resolve()
        output_root = final_dir.parent
    output_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        raise StageAError(f"refusing to overwrite output: {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=output_root))
    try:
        resolved_config = temporary / "resolved_config.yaml"
        resolved_config.write_text(yaml.safe_dump(config, sort_keys=False))
        _json_dump(temporary / "input_manifest.json", {"inputs": inputs})
        _write_csv(temporary / "ue_split_evidence_pool.csv", catalog)
        _write_csv(temporary / "ue_split_quality_strata.csv", strata)
        _write_csv(temporary / "ue_split_quality_floor_sensitivity.csv", sensitivity)
        _write_csv(temporary / "ue_split_network_regimes.csv", regimes)
        _write_csv(temporary / "ue_split_transport_evidence.csv", transport)
        _write_csv(temporary / "ue_split_profile_regime_screen.csv", screen)
        _write_csv(temporary / "ue_split_boundary_candidates.csv", boundaries)
        _write_csv(
            temporary / "ue_split_latency_tolerance_proxy.csv", latency_proxy
        )
        _write_csv(
            temporary / "ue_split_staleness_latency_anchors.csv", latency_anchors
        )
        _write_csv(
            temporary / "ue_split_staleness_error_sensitivity.csv",
            error_sensitivity,
        )
        _write_csv(temporary / "ue_split_unresolved_measurements.csv", unresolved)
        report_path = temporary / "REPORT.md"
        report_path.write_text(
            _report_text(
                catalog,
                sensitivity,
                regimes,
                boundaries,
                latency_anchors,
                profile_audit,
            )
        )

        output_schemas = {
            "resolved_config.yaml": CONFIG_SCHEMA,
            "input_manifest.json": "scenesense.ue_split_input_manifest.v1",
            "ue_split_evidence_pool.csv": "scenesense.ue_split_evidence_pool.v1",
            "ue_split_quality_strata.csv": "scenesense.ue_split_quality_strata.v1",
            "ue_split_quality_floor_sensitivity.csv": "scenesense.ue_split_quality_floor_sensitivity.v1",
            "ue_split_network_regimes.csv": "scenesense.ue_split_network_regimes.v1",
            "ue_split_transport_evidence.csv": "scenesense.ue_split_transport_evidence.v1",
            "ue_split_profile_regime_screen.csv": "scenesense.ue_split_profile_regime_screen.v1",
            "ue_split_boundary_candidates.csv": "scenesense.ue_split_boundary_candidates.v1",
            "ue_split_latency_tolerance_proxy.csv": "scenesense.ue_split_latency_tolerance_proxy.v1",
            "ue_split_staleness_latency_anchors.csv": "scenesense.ue_split_staleness_latency_anchors.v1",
            "ue_split_staleness_error_sensitivity.csv": "scenesense.ue_split_staleness_error_sensitivity.v1",
            "ue_split_unresolved_measurements.csv": "scenesense.ue_split_unresolved_measurements.v1",
            "REPORT.md": "scenesense.ue_split_stage_a_report.v1",
        }
        outputs = [
            _output_entry(temporary / name, schema)
            for name, schema in output_schemas.items()
        ]
        audit_tests = [
            {"id": "A_GRID", "status": "PASS", **profile_audit},
            {
                "id": "A_SPLIT_DISJOINTNESS",
                "status": "PASS",
                "train": profile_audit["train_split_count"],
                "val": profile_audit["val_split_count"],
                "test": profile_audit["test_split_count"],
            },
            {"id": "A_AGGREGATE_REPRODUCTION", "status": "PASS", **aggregate_check},
            {
                "id": "A_CHECKPOINT_REGISTRY",
                "status": "PASS",
                "models": 4,
                "lineage_kind": "registered_historical_not_row_embedded",
            },
            {"id": "N_REGIME_CATALOG", "status": "PASS", "regimes": 4},
            {
                "id": "N_NO_FALSE_DIRECT_CLAIM",
                "status": "PASS",
                "direct_fixed_10hz_cells": 0,
            },
            {
                "id": "N_FIXED_10HZ_BOUNDARY",
                "status": "PENDING",
                "direct_fixed_10hz_cells": 0,
            },
            {
                "id": "S_LATENCY_PROXY_INTEGRITY",
                "status": "PASS",
                "rows": len(latency_proxy),
                "aoi_observed": False,
            },
            {
                "id": "S_DIRECT_LATENCY_ANCHORS",
                "status": "PASS",
                "rows": len(latency_anchors),
            },
            {"id": "O_NO_RUN_AUTHORITY", "status": "PASS"},
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "experiment_id": final_dir.name,
            "created_utc": (now or datetime.now(timezone.utc)).isoformat(),
            "audit_state": "PASS",
            "decision_state": "REVIEW_REQUIRED",
            "verdict": VERDICT,
            "service_contract": {
                "service_contract_id": "OBJECT_MAP_V1",
                "segmentation_role": "secondary_diagnostic",
                "quality_floor_id": None,
                "quality_floor_config_sha256": None,
                "approval_reference": None,
            },
            "factor_contract": {
                "evidence_pool_version": config["factor_contract"]["evidence_pool_version"],
                "expected_profile_count": 72,
                "actual_profile_count": len(catalog),
                "network_regimes": ["clear", "mild", "mid", "poor"],
                "eligible_action_count": None,
                "expected_logical_surface_rows": None,
            },
            "datasets": {
                "quality_set": {
                    "quality_set_id": config["quality_evidence"]["quality_set_id"],
                    "split": "test",
                    "sample_count": profile_audit["frames_per_profile"],
                    "ordered_sample_ids_sha256": profile_audit["ordered_test_set_sha256"],
                    "source_dataset_present": profile_audit["source_dataset_present"],
                    "training_disjointness_status": config["quality_evidence"]
                    ["training_disjointness_status"],
                },
                "transport_replay": config["transport_replay"],
            },
            "repository": {
                **_git_state(root),
                "repository_diff_sha256": "UNAVAILABLE_DIRTY_SHARED_WORKTREE_NOT_SNAPSHOTTED",
                "assembler_path": str(Path(__file__).resolve().relative_to(root)),
                "assembler_sha256": _sha256_file(Path(__file__).resolve()),
                "resolved_config_sha256": _sha256_file(resolved_config),
            },
            "inputs": inputs,
            "outputs": outputs,
            "evidence_counts": {
                "profile_rows_existing_reused": len(catalog),
                "transport_rows_existing_reused_or_proxy": len(transport),
                "provisional_composed_profile_regime_rows": len(screen),
                "direct_new": 0,
                "direct_fixed_10hz": 0,
                "transport_status_counts": {
                    str(key): int(value)
                    for key, value in transport["evidence_status"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "screen_status_counts": {
                    str(key): int(value)
                    for key, value in screen["evidence_status"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            },
            "unresolved_count": int(len(unresolved)),
            "audit": {
                "verdict": "PASS",
                "tests": audit_tests,
                "fatal_errors": [],
                "warnings": [
                    "Final profile eligibility is not selected.",
                    "High-ROI small/far object certification is unresolved.",
                    "Historical OAI cells are not fixed-10-Hz accepted-map measurements.",
                    "Historical radio fields are summary-only because raw T-tracer directories are absent.",
                    "The source evaluation dataset is currently absent; raw evidence remains audit-ready.",
                ],
            },
        }
        manifest_path = temporary / "manifest.json"
        _json_dump(manifest_path, manifest)
        review = {
            "schema": REVIEW_SCHEMA,
            "verdict": VERDICT,
            "decision_state": "REVIEW_REQUIRED",
            "profile_count": len(catalog),
            "quality_floor_id": None,
            "eligible_action_count": None,
            "logical_surface_rows": None,
            "provisional_profile_regime_screen_rows": len(screen),
            "manifest_sha256": _sha256_file(manifest_path),
            "next_action": "review_quality_sensitivity_and_freeze_object_map_v1_floor",
            "no_run_authority": True,
        }
        _json_dump(temporary / "REVIEW_REQUIRED.json", review)
        if (temporary / "COMPLETED.json").exists():
            raise StageAError("REVIEW_REQUIRED output must not contain COMPLETED.json")
        os.replace(temporary, final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "verdict": VERDICT,
        "output_dir": str(final_dir),
        "profile_count": len(catalog),
        "profile_regime_screen_rows": len(screen),
        "quality_floor_id": None,
        "eligible_action_count": None,
        "next_action": "review_quality_sensitivity_and_freeze_object_map_v1_floor",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("rl_agent/configs/ue_split_stage_a_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = assemble(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
