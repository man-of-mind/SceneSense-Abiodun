"""Train-only distance failure audit for the frozen SplitFusion-FCOS p025 person path.

The audit answers one bounded question: do visible 30-40 m pedestrians already exist
among the raw FCOS person candidates, and which stage of the frozen pipeline removes
them? It reuses the registered train-only consolidation cache, the frozen train-holdout
actor-volume-observability (AVO) table, and the frozen consolidation rule. No training,
no cache rebuild, no model forward pass, no CUDA, and no validation or test access.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    CANONICAL_SCORE_THRESHOLD,
    consolidate_person_candidates,
    validate_configuration,
)

from .policy import PERSON_SCORE_THRESHOLD
from .qualification import (
    AVO_THRESHOLD,
    CACHE_ROOT,
    FEASIBILITY_PATH,
    MATCH_RADIUS_M,
    MAX_DISTANCE_M,
    OUTPUT_DIR as P025_OUTPUT_DIR,
    REFERENCE_ROOT,
    REFERENCE_RECORDS,
    REPO_ROOT,
    SELECTED_RULE,
    cache_hashes,
    greedy_match,
    load_contract,
    load_holdout_cache,
    load_holdout_raw,
    read_json,
    sha256_file,
    truth,
)

AUDIT_OUTPUT_DIR = REPO_ROOT / "experiments/splitfusion_fcos_person_distance_failure_audit_v1"
AVO_TABLE_PATH = P025_OUTPUT_DIR / "holdout_actor_volume_observability_table.csv"
P025_QUALIFICATION_PATH = P025_OUTPUT_DIR / "train_holdout_qualification.json"
P025_INPUT_HASHES_PATH = P025_OUTPUT_DIR / "INPUT_HASHES.json"

# Registered SHA-256 values, restated so any input drift fails closed. Sources:
# splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/FINAL_REPORT.md and the
# Phase-5 holdout evaluation record for the AVO table itself.
REGISTERED_HASHES = {
    "feasibility_result": "a1bb8b2b7062abc2d0ef4c5cbc715154c5a4e9f1da64e050547de14c56bdddde",
    "cache_manifest": "6e9386a6ee1d87cb19685ae0afb1c54cc6b9406bfae4ccf01d9e804578ddcc4c",
    "cache_shard_hash_map": "b599e883affc13ec1fd723c42e3901423af6062c2ac495e433254d2bdeef0d4b",
    "training_support_records": "8755b1904c821e6942197a3d41abb18806d049131a764ccb9f6100ab80493faf",
    "training_reference_json": "a825cffac4a060ee422951bb7d5af0b10d15eb39a347c081af836de35e6c1fff",
    "holdout_avo_table": "114514b645e8016f6ef5117e1b2da14b86fa3087efcb583985adb3096b711009",
}

# Registered train-holdout cardinality invariants (contract.py and the frozen record).
REGISTERED_COUNTS = {
    "raw_person_actor_frames": 25757,
    "canonically_qualified_actor_frames": 4703,
    "observable_actor_frames": 2556,
    "structural_ignored_actor_frames": 21054,
    "cache_frames": 3284,
    "cache_person_candidates": 222096,
    "retained_person_outputs_p020": 3460,
    "retained_person_outputs_p025": 3217,
}

# Fixed GT-distance bands. Every band is right-open except the last, which is closed at
# 40 m so that it matches the frozen `distance_bin` treatment of MAX_DISTANCE_M.
BANDS = (
    ("00_10m", 0.0, 10.0),
    ("10_20m", 10.0, 20.0),
    ("20_30m", 20.0, 30.0),
    ("30_35m", 30.0, 35.0),
    ("35_40m", 35.0, 40.0),
)
OVERFLOW_BAND = "gte_40m"
PREDICTION_BANDS = tuple(name for name, _lo, _hi in BANDS) + (OVERFLOW_BAND,)
LONG_RANGE_BANDS = ("30_35m", "35_40m")

# The frozen 30_40m bin is exactly the union of the two long-range audit bands.
FROZEN_BIN_UNION = {
    "00_10m": ("00_10m",),
    "10_20m": ("10_20m",),
    "20_30m": ("20_30m",),
    "30_40m": LONG_RANGE_BANDS,
}

# Nested stage decomposition of the frozen person path. Each stage is expressed as a
# preregistered consolidation-grid configuration so no selection logic is re-implemented.
STAGES = (
    {
        "key": "s1_raw_fcos_person_candidates",
        "title": "raw FCOS person candidates (post-NMS)",
        "rule": None,
        "score_threshold": None,
    },
    {
        "key": "s2_score_filter",
        "title": "after score filtering (>= 0.20)",
        "rule": {"grid_index": 0, "semantic_support_threshold": None, "group_box_iou_threshold": None},
        "score_threshold": None,
    },
    {
        "key": "s3_semantic_support_filter",
        "title": "after semantic-support filtering (>= 0.10)",
        "rule": {"grid_index": 24, "semantic_support_threshold": 0.10, "group_box_iou_threshold": None},
        "score_threshold": None,
    },
    {
        "key": "s4_instance_grouping",
        "title": "after instance grouping/consolidation (box IoU >= 0.20)",
        "rule": dict(SELECTED_RULE),
        "score_threshold": None,
    },
    {
        "key": "s5_person_p025",
        "title": "after the final p025 threshold (>= 0.25)",
        "rule": dict(SELECTED_RULE),
        "score_threshold": PERSON_SCORE_THRESHOLD,
    },
)

# Hard reproduction gates: the last two stages must reproduce the frozen p020 and p025
# train-holdout views exactly, otherwise the staged re-derivation is not faithful.
REPRODUCTION_GATES = {"s4_instance_grouping": "p020", "s5_person_p025": "p025"}

SUCCESS = "PERSON_P025_DISTANCE_FAILURE_AUDIT_COMPLETE"


class AuditError(RuntimeError):
    """Fail-closed audit input, contract, or accounting error."""


def band_of(distance_m: float, *, allow_overflow: bool) -> str:
    """Assign one distance to a fixed audit band."""
    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise AuditError(f"non-finite or negative distance: {distance_m!r}")
    for name, lower, upper in BANDS:
        if lower <= distance_m < upper:
            return name
    if distance_m <= MAX_DISTANCE_M:
        return BANDS[-1][0]
    if allow_overflow:
        return OVERFLOW_BAND
    raise AuditError(f"ground-truth distance outside [0,40]: {distance_m}")


def verify_hashes() -> dict[str, str]:
    """Verify every frozen input against its registered SHA-256."""
    manifest = read_json(CACHE_ROOT / "cache_manifest.json")
    exact = cache_hashes(manifest)
    observed = {
        "feasibility_result": sha256_file(FEASIBILITY_PATH),
        "cache_manifest": exact["cache_manifest_sha256"],
        "cache_shard_hash_map": exact["shard_hash_map_sha256"],
        "training_support_records": sha256_file(REFERENCE_RECORDS),
        "training_reference_json": sha256_file(REFERENCE_ROOT / "training_reference.json"),
        "holdout_avo_table": sha256_file(AVO_TABLE_PATH),
    }
    drift = sorted(name for name, value in observed.items() if value != REGISTERED_HASHES[name])
    if drift:
        raise AuditError(f"frozen input hash drift: {drift}")
    registered_raw = read_json(P025_INPUT_HASHES_PATH)["raw_holdout_metadata_sha256"]
    return {**observed, "raw_holdout_metadata_registered": registered_raw}


def verify_raw_metadata(raw_hashes: Mapping[str, str], registered: Mapping[str, str]) -> None:
    if dict(raw_hashes) != dict(registered):
        raise AuditError("raw holdout metadata hash drift against the frozen p025 record")


def load_avo_table() -> list[dict[str, Any]]:
    # The frozen table was written with `repr` from values that pandas' default (fast)
    # float parser produced when reading the raw GT. Re-reading it with `round_trip`
    # recovers those exact doubles, so the join below can demand bitwise agreement with
    # the unchanged frozen raw-GT load path.
    frame = pd.read_csv(AVO_TABLE_PATH, dtype={"gt_actor_id": str}, float_precision="round_trip")
    rows = frame.to_dict(orient="records")
    if len(rows) != REGISTERED_COUNTS["canonically_qualified_actor_frames"]:
        raise AuditError("frozen AVO table cardinality drift")
    return rows


def join_ground_truth(
    table: Sequence[Mapping[str, Any]],
    raw: Mapping[str, Any],
    frame_set: set[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Join the frozen AVO table to the raw holdout GT on the exact sample/actor identity."""
    qualified_by_key = {
        (str(row["experiment_id"]), str(row["sample_id"]), str(row["gt_actor_id"])): row
        for row in raw["qualified"]
    }
    table_keys = {
        (str(row["episode_id"]), str(row["sample_id"]), str(row["gt_actor_id"])) for row in table
    }
    if len(table_keys) != len(table):
        raise AuditError("frozen AVO table has duplicate actor-frame keys")
    if table_keys != set(qualified_by_key):
        raise AuditError("frozen AVO table is not an exact join with the canonically eligible GT")

    manifest_by_sample = raw["manifest_by_sample"]
    observable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    avo_ignored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    max_plane_error = 0.0
    plane_error_sum = 0.0
    for row in table:
        key = (str(row["episode_id"]), str(row["sample_id"]), str(row["gt_actor_id"]))
        source = qualified_by_key[key]
        distance = float(row["distance_m"])
        world_x, world_y = float(row["world_x"]), float(row["world_y"])
        if (
            world_x != float(source["object_world_x"])
            or world_y != float(source["object_world_y"])
            or distance != float(source["gt_distance_m"])
        ):
            raise AuditError(f"frozen AVO row disagrees with the raw GT record: {key}")
        sample_id = str(row["sample_id"])
        if sample_id not in frame_set:
            raise AuditError(f"AVO sample is absent from the candidate cache: {sample_id}")
        meta = manifest_by_sample[sample_id]
        plane = math.hypot(world_x - float(meta["camera_x"]), world_y - float(meta["camera_y"]))
        plane_error_sum += abs(plane - distance)
        max_plane_error = max(max_plane_error, abs(plane - distance))
        avo = float(row["actor_volume_observability"])
        target = {
            "world_x": world_x,
            "world_y": world_y,
            "distance_m": distance,
            "band": band_of(distance, allow_overflow=False),
            "frozen_bin": str(row["distance_bin"]),
            "avo": avo,
            "no_support": truth(row["no_support"]),
        }
        (observable if avo >= AVO_THRESHOLD else avo_ignored)[sample_id].append(target)

    for row in table:
        band = band_of(float(row["distance_m"]), allow_overflow=False)
        if band not in FROZEN_BIN_UNION[str(row["distance_bin"])]:
            raise AuditError("audit band is inconsistent with the frozen distance_bin")

    structural: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw["structural"]:
        structural[str(row["sample_id"])].append(
            {"world_x": float(row["object_world_x"]), "world_y": float(row["object_world_y"])}
        )
    observable_total = sum(len(rows) for rows in observable.values())
    structural_total = sum(len(rows) for rows in structural.values())
    if (
        observable_total != REGISTERED_COUNTS["observable_actor_frames"]
        or structural_total != REGISTERED_COUNTS["structural_ignored_actor_frames"]
        or len(raw["all_people"]) != REGISTERED_COUNTS["raw_person_actor_frames"]
    ):
        raise AuditError("train-holdout GT cardinality drift")
    diagnostics = {
        "avo_table_rows": len(table),
        "observable_gt_actor_frames": observable_total,
        "avo_ignored_gt_actor_frames": sum(len(rows) for rows in avo_ignored.values()),
        "structural_ignored_gt_actor_frames": structural_total,
        "raw_person_actor_frames": len(raw["all_people"]),
        "join_key": "(episode_id, sample_id, gt_actor_id)",
        "join_tolerance": "bitwise equality on world_x, world_y, and distance_m",
        "gt_distance_convention": "gt_distance_m = 3D camera-origin radial distance",
        "prediction_distance_convention": (
            "world-plane radial distance from (camera_x, camera_y); the cache stores only "
            "world_xyz[:, :2] for candidates, so no 3D predicted distance is available"
        ),
        "gt_plane_vs_3d_distance_max_abs_error_m": max_plane_error,
        "gt_plane_vs_3d_distance_mean_abs_error_m": plane_error_sum / max(1, len(table)),
    }
    return dict(observable), dict(avo_ignored), dict(structural), diagnostics


def stage_positions(frame: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return the nested surviving candidate positions for every audited stage."""
    scores = frame["scores"].detach().float().cpu()
    original = frame["original_indices"].detach().long().cpu()
    count = scores.numel()
    if count and not bool((original[1:] > original[:-1]).all() if count > 1 else True):
        raise AuditError("cached person candidates are not in ascending original order")
    positions: dict[str, torch.Tensor] = {}
    for stage in STAGES:
        rule = stage["rule"]
        if rule is None:
            selected = torch.arange(count, dtype=torch.long)
        else:
            validate_configuration(rule)
            selected = consolidate_person_candidates(
                scores=frame["scores"],
                boxes=frame["boxes"],
                world_xy=frame["world_xy"],
                component_ids=frame["component_ids"],
                semantic_support=frame["semantic_support"],
                original_indices=original,
                semantic_support_threshold=rule["semantic_support_threshold"],
                group_box_iou_threshold=rule["group_box_iou_threshold"],
            )
        if stage["score_threshold"] is not None:
            keep = scores.index_select(0, selected) >= float(stage["score_threshold"])
            selected = selected.index_select(0, torch.where(keep)[0])
        positions[stage["key"]] = selected
    previous: set[int] | None = None
    for stage in STAGES:
        current = set(positions[stage["key"]].tolist())
        if previous is not None and not current.issubset(previous):
            raise AuditError(f"stage {stage['key']} is not a subset of its predecessor")
        previous = current
    return positions


def prediction_records(
    frame: Mapping[str, Any], positions: torch.Tensor, camera_xy: tuple[float, float]
) -> list[dict[str, Any]]:
    scores = frame["scores"].detach().float().cpu()
    world = frame["world_xy"].detach().double().cpu()
    support = frame["semantic_support"].detach().float().cpu()
    original = frame["original_indices"].detach().long().cpu()
    records = []
    for position in positions.tolist():
        world_x, world_y = float(world[position, 0]), float(world[position, 1])
        records.append(
            {
                "score": float(scores[position]),
                "semantic_support": float(support[position]),
                "world_x": world_x,
                "world_y": world_y,
                "predicted_distance_m": math.hypot(world_x - camera_xy[0], world_y - camera_xy[1]),
                "original_index": int(original[position]),
            }
        )
    return records


def maximum_matching(reachable: Sequence[Sequence[int]]) -> list[int]:
    """Return the GT indices covered by one maximum bipartite matching (Kuhn's algorithm)."""
    assignment: dict[int, int] = {}

    def augment(gt_index: int, seen: set[int]) -> bool:
        for candidate in reachable[gt_index]:
            if candidate in seen:
                continue
            seen.add(candidate)
            holder = assignment.get(candidate)
            if holder is None or augment(holder, seen):
                assignment[candidate] = gt_index
                return True
        return False

    covered = [gt_index for gt_index in range(len(reachable)) if augment(gt_index, set())]
    return covered


def new_bucket() -> dict[str, Any]:
    return {
        "eligible_gt": 0,
        "tp_gt_band": 0,
        "fn": 0,
        "reachable_gt": 0,
        "max_matching_gt": 0,
        "tp_pred_band": 0,
        "fp": 0,
        "avo_ignored_predictions": 0,
        "structural_ignored_predictions": 0,
        "_xy_gt": [],
        "_xy_pred": [],
    }


def score_stage(
    *,
    frames: Sequence[Mapping[str, Any]],
    stage_key: str,
    stage_positions_by_sample: Mapping[str, Mapping[str, torch.Tensor]],
    observable_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    avo_ignored_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    structural_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    camera_xy_by_sample: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Score one stage, preserving the frozen observable-first matching order."""
    totals = new_bucket()
    bands = {name: new_bucket() for name in PREDICTION_BANDS}
    retained_predictions = 0
    for frame in frames:
        sample_id = str(frame["sample_id"])
        predictions = prediction_records(
            frame, stage_positions_by_sample[sample_id][stage_key], camera_xy_by_sample[sample_id]
        )
        retained_predictions += len(predictions)
        eligible = list(observable_gt.get(sample_id, []))
        ignored = list(avo_ignored_gt.get(sample_id, []))
        structural = list(structural_gt.get(sample_id, []))

        for target in eligible:
            bands[target["band"]]["eligible_gt"] += 1
            totals["eligible_gt"] += 1

        reachable: list[list[int]] = []
        for target in eligible:
            near = [
                index
                for index, prediction in enumerate(predictions)
                if math.hypot(
                    prediction["world_x"] - target["world_x"],
                    prediction["world_y"] - target["world_y"],
                )
                <= MATCH_RADIUS_M
            ]
            reachable.append(near)
            if near:
                bands[target["band"]]["reachable_gt"] += 1
                totals["reachable_gt"] += 1
        for gt_index in maximum_matching(reachable):
            bands[eligible[gt_index]["band"]]["max_matching_gt"] += 1
            totals["max_matching_gt"] += 1

        matched, used_eligible = greedy_match(predictions, eligible)
        used_predictions = set(matched)
        for pred_index, gt_index in matched.items():
            prediction, target = predictions[pred_index], eligible[gt_index]
            error = math.hypot(
                prediction["world_x"] - target["world_x"], prediction["world_y"] - target["world_y"]
            )
            gt_band = target["band"]
            pred_band = band_of(prediction["predicted_distance_m"], allow_overflow=True)
            bands[gt_band]["tp_gt_band"] += 1
            bands[gt_band]["_xy_gt"].append(error)
            bands[pred_band]["tp_pred_band"] += 1
            bands[pred_band]["_xy_pred"].append(error)
            totals["tp_gt_band"] += 1
            totals["tp_pred_band"] += 1
            totals["_xy_gt"].append(error)
            totals["_xy_pred"].append(error)
        for gt_index, target in enumerate(eligible):
            if gt_index not in used_eligible:
                bands[target["band"]]["fn"] += 1
                totals["fn"] += 1

        remaining = set(range(len(predictions))) - used_predictions
        matched_avo, _ = greedy_match(predictions, ignored, remaining)
        remaining -= set(matched_avo)
        matched_structural, _ = greedy_match(predictions, structural, remaining)
        remaining -= set(matched_structural)
        for pred_index in matched_avo:
            band = band_of(predictions[pred_index]["predicted_distance_m"], allow_overflow=True)
            bands[band]["avo_ignored_predictions"] += 1
            totals["avo_ignored_predictions"] += 1
        for pred_index in matched_structural:
            band = band_of(predictions[pred_index]["predicted_distance_m"], allow_overflow=True)
            bands[band]["structural_ignored_predictions"] += 1
            totals["structural_ignored_predictions"] += 1
        for pred_index in remaining:
            band = band_of(predictions[pred_index]["predicted_distance_m"], allow_overflow=True)
            bands[band]["fp"] += 1
            totals["fp"] += 1

    def finalize(bucket: Mapping[str, Any]) -> dict[str, Any]:
        eligible_gt = int(bucket["eligible_gt"])
        tp_gt, tp_pred, fp, fn = (
            int(bucket["tp_gt_band"]),
            int(bucket["tp_pred_band"]),
            int(bucket["fp"]),
            int(bucket["fn"]),
        )
        if tp_gt + fn != eligible_gt:
            raise AuditError("per-band TP+FN denominator failure")
        precision = tp_pred / (tp_pred + fp) if tp_pred + fp else 0.0
        recall = tp_gt / eligible_gt if eligible_gt else 0.0
        return {
            "eligible_gt": eligible_gt,
            "tp_gt_band": tp_gt,
            "tp_pred_band": tp_pred,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "xy_mae_m": (sum(bucket["_xy_gt"]) / len(bucket["_xy_gt"])) if bucket["_xy_gt"] else None,
            "xy_mae_m_pred_band": (
                sum(bucket["_xy_pred"]) / len(bucket["_xy_pred"]) if bucket["_xy_pred"] else None
            ),
            "reachable_gt": int(bucket["reachable_gt"]),
            "candidate_recall_ceiling": (
                int(bucket["reachable_gt"]) / eligible_gt if eligible_gt else None
            ),
            "max_matching_gt": int(bucket["max_matching_gt"]),
            "max_matching_ceiling": (
                int(bucket["max_matching_gt"]) / eligible_gt if eligible_gt else None
            ),
            "avo_ignored_predictions": int(bucket["avo_ignored_predictions"]),
            "structural_ignored_predictions": int(bucket["structural_ignored_predictions"]),
        }

    overall = finalize(totals)
    per_band = {name: finalize(bands[name]) for name in PREDICTION_BANDS}
    checks = {
        "eligible_gt_sums": sum(row["eligible_gt"] for row in per_band.values()) == overall["eligible_gt"],
        "tp_gt_band_sums": sum(row["tp_gt_band"] for row in per_band.values()) == overall["tp_gt_band"],
        "tp_pred_band_sums": sum(row["tp_pred_band"] for row in per_band.values()) == overall["tp_pred_band"],
        "fp_sums": sum(row["fp"] for row in per_band.values()) == overall["fp"],
        "fn_sums": sum(row["fn"] for row in per_band.values()) == overall["fn"],
        "tp_assignments_agree_in_total": overall["tp_gt_band"] == overall["tp_pred_band"],
        "overflow_band_has_no_eligible_gt": per_band[OVERFLOW_BAND]["eligible_gt"] == 0,
    }
    if not all(checks.values()):
        raise AuditError(f"distance-band accounting failure: {sorted(k for k, v in checks.items() if not v)}")
    return {
        "stage": stage_key,
        "retained_predictions": retained_predictions,
        "overall": overall,
        "bands": per_band,
        "accounting_checks": checks,
    }


def add_stage_losses(views: Mapping[str, Mapping[str, Any]]) -> None:
    """Attach the per-band GT loss attributable to each stage transition."""
    keys = [stage["key"] for stage in STAGES]
    for index, key in enumerate(keys):
        previous = views[keys[index - 1]] if index else None
        for scope, row in [("overall", views[key]["overall"])] + [
            (name, views[key]["bands"][name]) for name in PREDICTION_BANDS
        ]:
            if previous is None:
                row["reachable_gt_lost_since_previous_stage"] = 0
                row["reachable_gt_lost_pct_of_eligible"] = 0.0
                row["reachable_gt_lost_pct_of_previous_reachable"] = 0.0
                row["tp_change_since_previous_stage"] = 0
                continue
            prior = previous["overall"] if scope == "overall" else previous["bands"][scope]
            lost = int(prior["reachable_gt"]) - int(row["reachable_gt"])
            if lost < 0:
                raise AuditError("candidate reachability increased across a nested stage")
            row["reachable_gt_lost_since_previous_stage"] = lost
            row["reachable_gt_lost_pct_of_eligible"] = (
                100.0 * lost / row["eligible_gt"] if row["eligible_gt"] else 0.0
            )
            row["reachable_gt_lost_pct_of_previous_reachable"] = (
                100.0 * lost / prior["reachable_gt"] if prior["reachable_gt"] else 0.0
            )
            row["tp_change_since_previous_stage"] = int(row["tp_gt_band"]) - int(prior["tp_gt_band"])


def reproduction_gates(views: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require the last two stages to reproduce the frozen p020/p025 views exactly."""
    frozen = read_json(P025_QUALIFICATION_PATH)
    gates: dict[str, Any] = {}
    for stage_key, view_name in REPRODUCTION_GATES.items():
        expected = frozen["train_holdout"][view_name]["overall"]
        observed = views[stage_key]["overall"]
        gates[f"{stage_key}_reproduces_frozen_{view_name}"] = bool(
            observed["eligible_gt"] == expected["observable_gt"]
            and observed["tp_gt_band"] == expected["tp"]
            and observed["fp"] == expected["fp"]
            and observed["fn"] == expected["fn"]
            and observed["avo_ignored_predictions"] == expected["avo_ignored_predictions"]
            and observed["structural_ignored_predictions"] == expected["structural_ignored_predictions"]
            and abs(observed["precision"] - expected["precision"]) <= 1e-12
            and abs(observed["recall"] - expected["recall"]) <= 1e-12
            and abs(float(observed["xy_mae_m"]) - float(expected["xy_mae_m"])) <= 1e-12
        )
    invariants = frozen["output_invariants"]
    gates["s4_retained_count_matches_frozen_p020"] = (
        views["s4_instance_grouping"]["retained_predictions"]
        == invariants["retained_person_outputs_p020"]
        == REGISTERED_COUNTS["retained_person_outputs_p020"]
    )
    gates["s5_retained_count_matches_frozen_p025"] = (
        views["s5_person_p025"]["retained_predictions"]
        == invariants["retained_person_outputs_p025"]
        == REGISTERED_COUNTS["retained_person_outputs_p025"]
    )
    if not all(gates.values()):
        raise AuditError(f"frozen reproduction failure: {sorted(k for k, v in gates.items() if not v)}")
    return gates


def candidate_headroom(
    *,
    frames: Sequence[Mapping[str, Any]],
    stage_positions_by_sample: Mapping[str, Mapping[str, torch.Tensor]],
    observable_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    camera_xy_by_sample: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Characterize the best raw candidate near each observable GT, per distance band."""
    per_band = {
        name: {
            "eligible_gt": 0,
            "reachable_raw": 0,
            "best_raw_score_ge_0_20": 0,
            "best_raw_score_ge_0_25": 0,
            "supported_raw_candidate_present": 0,
            "supported_raw_score_ge_0_20": 0,
            "supported_raw_score_ge_0_25": 0,
            "_best_scores": [],
            "_best_supports": [],
        }
        for name, _lo, _hi in BANDS
    }
    for frame in frames:
        sample_id = str(frame["sample_id"])
        eligible = observable_gt.get(sample_id, [])
        if not eligible:
            continue
        raw = prediction_records(
            frame,
            stage_positions_by_sample[sample_id]["s1_raw_fcos_person_candidates"],
            camera_xy_by_sample[sample_id],
        )
        for target in eligible:
            row = per_band[target["band"]]
            row["eligible_gt"] += 1
            near = [
                candidate
                for candidate in raw
                if math.hypot(
                    candidate["world_x"] - target["world_x"], candidate["world_y"] - target["world_y"]
                )
                <= MATCH_RADIUS_M
            ]
            if not near:
                continue
            row["reachable_raw"] += 1
            best_score = max(candidate["score"] for candidate in near)
            best_support = max(candidate["semantic_support"] for candidate in near)
            row["_best_scores"].append(best_score)
            row["_best_supports"].append(best_support)
            row["best_raw_score_ge_0_20"] += int(best_score >= CANONICAL_SCORE_THRESHOLD)
            row["best_raw_score_ge_0_25"] += int(best_score >= PERSON_SCORE_THRESHOLD)
            supported = [
                candidate
                for candidate in near
                if candidate["semantic_support"] >= float(SELECTED_RULE["semantic_support_threshold"])
            ]
            if supported:
                row["supported_raw_candidate_present"] += 1
                best_supported = max(candidate["score"] for candidate in supported)
                row["supported_raw_score_ge_0_20"] += int(best_supported >= CANONICAL_SCORE_THRESHOLD)
                row["supported_raw_score_ge_0_25"] += int(best_supported >= PERSON_SCORE_THRESHOLD)

    def quantiles(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        def at(fraction: float) -> float:
            return ordered[min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))]
        return {"p10": at(0.10), "p50": at(0.50), "p90": at(0.90)}

    result = {}
    for name, row in per_band.items():
        result[name] = {
            **{key: value for key, value in row.items() if not key.startswith("_")},
            "best_raw_score_quantiles": quantiles(row["_best_scores"]),
            "best_raw_semantic_support_quantiles": quantiles(row["_best_supports"]),
        }
    return result


def first_responsible_stage(views: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Identify the stage removing the largest share of long-range observable GT."""
    summary = {}
    for band in LONG_RANGE_BANDS + ("30_40m_union",):
        if band == "30_40m_union":
            losses = {
                stage["key"]: sum(
                    views[stage["key"]]["bands"][name]["reachable_gt_lost_since_previous_stage"]
                    for name in LONG_RANGE_BANDS
                )
                for stage in STAGES[1:]
            }
            eligible = sum(views[STAGES[0]["key"]]["bands"][name]["eligible_gt"] for name in LONG_RANGE_BANDS)
            raw_reachable = sum(
                views[STAGES[0]["key"]]["bands"][name]["reachable_gt"] for name in LONG_RANGE_BANDS
            )
            final_recall_tp = sum(
                views[STAGES[-1]["key"]]["bands"][name]["tp_gt_band"] for name in LONG_RANGE_BANDS
            )
            final_reachable = sum(
                views[STAGES[-1]["key"]]["bands"][name]["reachable_gt"] for name in LONG_RANGE_BANDS
            )
        else:
            losses = {
                stage["key"]: views[stage["key"]]["bands"][band]["reachable_gt_lost_since_previous_stage"]
                for stage in STAGES[1:]
            }
            eligible = views[STAGES[0]["key"]]["bands"][band]["eligible_gt"]
            raw_reachable = views[STAGES[0]["key"]]["bands"][band]["reachable_gt"]
            final_recall_tp = views[STAGES[-1]["key"]]["bands"][band]["tp_gt_band"]
            final_reachable = views[STAGES[-1]["key"]]["bands"][band]["reachable_gt"]
        total_lost = sum(losses.values())
        # losses is built in stage order, so max() breaks ties toward the earliest stage.
        dominant = max(losses.items(), key=lambda item: item[1])
        summary[band] = {
            "eligible_gt": eligible,
            "raw_candidate_recall_ceiling": raw_reachable / eligible if eligible else None,
            "final_stage_recall": final_recall_tp / eligible if eligible else None,
            "final_stage_candidate_recall_ceiling": final_reachable / eligible if eligible else None,
            "reachable_gt_lost_by_stage": losses,
            "total_reachable_gt_lost": total_lost,
            "first_responsible_stage": dominant[0],
            "first_responsible_stage_share_of_losses": (
                dominant[1] / total_lost if total_lost else None
            ),
            "recoverable_headroom_gt": raw_reachable - final_recall_tp,
            "recoverable_headroom_recall_points": (
                (raw_reachable - final_recall_tp) / eligible if eligible else None
            ),
        }
    return summary


def markdown_report(result: Mapping[str, Any]) -> str:
    views = result["stage_views"]
    lines = [
        "# Train-only distance failure audit — frozen SplitFusion-FCOS p025 person path",
        "",
        "One bounded, train-only diagnostic. No training, no cache rebuild, no model forward",
        "pass, no CUDA, and no validation or test access. Every input was verified against its",
        "registered SHA-256, and stages 4 and 5 reproduce the frozen p020 and p025 train-holdout",
        "views exactly (counts, precision, recall, and XY MAE), which validates the",
        "re-derivation.",
        "",
        "## Conventions",
        "",
        "- Ground truth is assigned to a band by `gt_distance_m` (3D camera-origin radial distance).",
        "- Predictions are assigned to a band by their predicted radial distance, computed on the",
        "  world plane from `(camera_x, camera_y)`; the cache stores only `world_xyz[:, :2]` for",
        f"  candidates. Over the {result['ground_truth']['avo_table_rows']} eligible GT rows the plane-vs-3D convention gap is at most",
        f"  {result['ground_truth']['gt_plane_vs_3d_distance_max_abs_error_m']:.4f} m (mean {result['ground_truth']['gt_plane_vs_3d_distance_mean_abs_error_m']:.4f} m).",
        "- Per-band recall uses the GT assignment; per-band precision uses the prediction",
        "  assignment; per-band F1 combines the two and is therefore a mixed-assignment quantity.",
        "- Matching order is unchanged: observable GT first, then AVO-ignore, then structural-ignore,",
        f"  greedy by ascending world distance inside a {MATCH_RADIUS_M:.1f} m radius, AVO >= {AVO_THRESHOLD}.",
        "- `candidate recall ceiling` = share of eligible GT with at least one surviving candidate",
        "  within the match radius. `max_matching_ceiling` is the same quantity under one maximum",
        "  bipartite matching, so it is the attainable recall of a perfect downstream selector.",
        "",
        "## Scope",
        "",
        f"- Episodes: the two registered train-holdout episodes, {result['cache_counts']['frames']} frames.",
        f"- Observable GT actor-frames at AVO >= {AVO_THRESHOLD}: {result['ground_truth']['observable_gt_actor_frames']}.",
        "- Stage 1 is the earliest cached stage: post-NMS person candidates (head score > 0.02,",
        "  per-level top-1000, class-wise NMS at IoU 0.60, top-100 detections per image). Pre-NMS",
        "  candidates are not cached, so any loss inside NMS or the top-100 cap is upstream of this",
        "  audit and is not measured.",
        "",
    ]
    for stage in STAGES:
        view = views[stage["key"]]
        lines.extend(
            [
                f"## {stage['key']} — {stage['title']}",
                "",
                f"Retained person predictions: {view['retained_predictions']}.",
                "",
                "| band | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m | ceiling | max-match ceiling | GT lost | GT lost % |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in PREDICTION_BANDS + ("overall",):
            row = view["overall"] if name == "overall" else view["bands"][name]
            mae = "n/a" if row["xy_mae_m"] is None else f"{row['xy_mae_m']:.6f}"
            ceiling = "n/a" if row["candidate_recall_ceiling"] is None else f"{row['candidate_recall_ceiling']:.6f}"
            max_ceiling = (
                "n/a" if row["max_matching_ceiling"] is None else f"{row['max_matching_ceiling']:.6f}"
            )
            lines.append(
                f"| {name} | {row['eligible_gt']} | {row['tp_gt_band']} | {row['tp_pred_band']} | {row['fp']} | {row['fn']} | "
                f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {mae} | {ceiling} | "
                f"{max_ceiling} | {row['reachable_gt_lost_since_previous_stage']} | "
                f"{row['reachable_gt_lost_pct_of_eligible']:.3f} |"
            )
        lines.append("")
        lines.append(
            "`TP(gt)` and `FN` drive recall and `XY MAE m`; `TP(pred)` and `FP` drive precision. "
            "The two TP assignments differ per band and are equal in total."
        )
        lines.append("")

    lines.extend(["## First stage responsible for the 30-40 m losses", ""])
    lines.append("| band | eligible GT | raw ceiling | final recall | final ceiling | dominant stage | share | headroom GT | headroom recall pts |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|")
    for band, row in result["long_range_attribution"].items():
        share = "n/a" if row["first_responsible_stage_share_of_losses"] is None else f"{row['first_responsible_stage_share_of_losses']:.4f}"
        lines.append(
            f"| {band} | {row['eligible_gt']} | {row['raw_candidate_recall_ceiling']:.6f} | "
            f"{row['final_stage_recall']:.6f} | {row['final_stage_candidate_recall_ceiling']:.6f} | "
            f"{row['first_responsible_stage']} | {share} | {row['recoverable_headroom_gt']} | "
            f"{row['recoverable_headroom_recall_points']:.6f} |"
        )
    lines.extend(["", "## Raw-candidate headroom near observable GT", ""])
    lines.append("| band | eligible GT | reachable raw | best raw score >= 0.20 | >= 0.25 | supported candidate present | supported >= 0.20 | supported >= 0.25 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, _lo, _hi in BANDS:
        row = result["raw_candidate_headroom"][name]
        lines.append(
            f"| {name} | {row['eligible_gt']} | {row['reachable_raw']} | {row['best_raw_score_ge_0_20']} | "
            f"{row['best_raw_score_ge_0_25']} | {row['supported_raw_candidate_present']} | "
            f"{row['supported_raw_score_ge_0_20']} | {row['supported_raw_score_ge_0_25']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "No policy was tuned in this phase. The ceilings above bound what any range-aware",
            "post-processing over the already-cached candidates could reach; they are train-holdout",
            "numbers and carry no validation or test claim.",
            "",
            SUCCESS,
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUDIT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise AuditError('refusing to run without CUDA_VISIBLE_DEVICES=""')
    output = args.output.resolve()
    if output != AUDIT_OUTPUT_DIR.resolve():
        raise AuditError("output must be the registered audit directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    hashes = verify_hashes()
    _feasibility, cache_manifest, episodes = load_contract()
    frames, cache_counts = load_holdout_cache(cache_manifest, episodes)
    if (
        cache_counts["frames"] != REGISTERED_COUNTS["cache_frames"]
        or cache_counts["person_candidates"] != REGISTERED_COUNTS["cache_person_candidates"]
    ):
        raise AuditError("holdout candidate-cache cardinality drift")
    frame_set = {str(frame["sample_id"]) for frame in frames}
    raw, raw_hashes = load_holdout_raw(frame_set, episodes)
    verify_raw_metadata(raw_hashes, hashes.pop("raw_holdout_metadata_registered"))
    # The exclusion reasons are non-exclusive, so assert the property the audit relies on:
    # every structurally ignored person is beyond 40 m, hence no in-range GT was dropped
    # before AVO scoring.
    if (
        int(raw["exclusion_reasons"].get("distance_gt_40m", 0)) != len(raw["structural"])
        or any(float(row["gt_distance_m"]) <= MAX_DISTANCE_M for row in raw["structural"])
        or any(float(row["gt_distance_m"]) > MAX_DISTANCE_M for row in raw["qualified"])
    ):
        raise AuditError("structural-ignore composition drift")

    table = load_avo_table()
    observable_gt, avo_ignored_gt, structural_gt, gt_diagnostics = join_ground_truth(
        table, raw, frame_set
    )
    camera_xy_by_sample = {
        sample_id: (float(meta["camera_x"]), float(meta["camera_y"]))
        for sample_id, meta in raw["manifest_by_sample"].items()
    }
    stage_positions_by_sample = {
        str(frame["sample_id"]): stage_positions(frame) for frame in frames
    }

    views = {
        stage["key"]: score_stage(
            frames=frames,
            stage_key=stage["key"],
            stage_positions_by_sample=stage_positions_by_sample,
            observable_gt=observable_gt,
            avo_ignored_gt=avo_ignored_gt,
            structural_gt=structural_gt,
            camera_xy_by_sample=camera_xy_by_sample,
        )
        for stage in STAGES
    }
    add_stage_losses(views)
    gates = reproduction_gates(views)
    headroom = candidate_headroom(
        frames=frames,
        stage_positions_by_sample=stage_positions_by_sample,
        observable_gt=observable_gt,
        camera_xy_by_sample=camera_xy_by_sample,
    )
    attribution = first_responsible_stage(views)

    result = {
        "schema": "splitfusion_fcos_person_p025_distance_failure_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": SUCCESS,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_run": False,
        "cache_rebuilt": False,
        "model_inference_run": False,
        "validation_accessed": False,
        "test_accessed": False,
        "policy_tuned": False,
        "split": "train_holdout_only",
        "holdout_episodes": list(episodes),
        "avo_threshold": AVO_THRESHOLD,
        "match_radius_m": MATCH_RADIUS_M,
        "matching_order": "observable_gt_then_avo_ignored_gt_then_structural_ignored_gt",
        "selected_consolidation_rule": dict(SELECTED_RULE),
        "distance_bands": [
            {"band": name, "lower_inclusive_m": lower, "upper_exclusive_m": upper}
            for name, lower, upper in BANDS
        ],
        "prediction_overflow_band": OVERFLOW_BAND,
        "band_assignment": {
            "recall": "gt_distance_m",
            "precision": "predicted world-plane radial distance from the camera origin",
        },
        "earliest_cached_stage_scope": (
            "stage 1 is post-NMS: head score > 0.02, per-level top-1000, class-wise NMS at "
            "IoU 0.60, top-100 detections per image; pre-NMS candidates are not cached"
        ),
        "stages": [
            {"key": stage["key"], "title": stage["title"], "rule": stage["rule"],
             "score_threshold": stage["score_threshold"]}
            for stage in STAGES
        ],
        "cache_counts": cache_counts,
        "ground_truth": gt_diagnostics,
        "stage_views": views,
        "long_range_attribution": attribution,
        "raw_candidate_headroom": headroom,
        "frozen_reproduction_gates": gates,
        "input_hashes": {**hashes, "raw_holdout_metadata_sha256": raw_hashes},
        "registered_counts": REGISTERED_COUNTS,
    }
    result["runtime_seconds"] = time.perf_counter() - started
    with (output / "distance_failure_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    (output / "DISTANCE_FAILURE_AUDIT.md").write_text(markdown_report(result), encoding="utf-8")
    (output / SUCCESS).write_text(SUCCESS + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal": SUCCESS,
        "runtime_seconds": result["runtime_seconds"],
        "frozen_reproduction_gates": gates,
        "long_range_attribution": attribution,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
