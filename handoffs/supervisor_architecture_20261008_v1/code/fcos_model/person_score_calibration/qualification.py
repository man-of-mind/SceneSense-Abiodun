from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from data_collection.route_b_publication_actor_volume_observability_model_comparison_v1.run_comparison import (
    verify_original_unnormalized_sources,
)
from data_collection.route_b_publication_actor_volume_visibility_v1 import core, scoring
from data_collection.route_b_publication_actor_volume_visibility_v1.run_audit import (
    decode_depth_bgra,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    CANONICAL_SCORE_THRESHOLD,
    PERSON_INTERNAL_CLASS,
    consolidate_person_candidates,
)

from .policy import PERSON_SCORE_THRESHOLD


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "experiments/splitfusion_fcos_person_p025_calibration_v1"
FEASIBILITY_PATH = REPO_ROOT / "experiments/person_instance_consolidation_v1/feasibility_result.json"
CACHE_ROOT = REPO_ROOT / "experiments/person_instance_consolidation_v1/train_cache"
REFERENCE_ROOT = (
    REPO_ROOT
    / "data_collection/experiments/route_b_publication_actor_volume_visibility_v1"
    / "training_reference/20260901_214026"
)
REFERENCE_RECORDS = REFERENCE_ROOT / "training_support_records.csv"
RAW_ROOT = REPO_ROOT / "data_collection/experiments/route_b_perception_v3"

AVO_THRESHOLD = 0.65
MATCH_RADIUS_M = 3.0
MAX_DISTANCE_M = 40.0
MIN_PROJECTED_AREA_PX = 12.0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
SELECTED_RULE = {
    "grid_index": 27,
    "semantic_support_threshold": 0.10,
    "group_box_iou_threshold": 0.20,
}
AGGREGATE_MINIMUM = 0.70
EPISODE_MINIMUM = 0.65
SUCCESS = "PERSON_P025_TRAIN_HOLDOUT_QUALIFIED"
FAILURE = "PERSON_P025_TRAIN_HOLDOUT_NOT_QUALIFIED_RETAIN_P020"


class QualificationError(RuntimeError):
    """Fail-closed input, contract, or scoring error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise QualificationError(f"non-finite {context}: {value!r}")
    return result


def actor_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["experiment_id"]), str(row["sample_id"]), str(row["gt_actor_id"])


def reference_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["episode_id"]), str(row["sample_id"]), str(row["gt_actor_id"])


def assert_unique(rows: Sequence[Mapping[str, Any]], key: Any, context: str) -> None:
    keys = [key(row) for row in rows]
    duplicates = [item for item, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise QualificationError(f"duplicate {context} key: {duplicates[0]}")


def distance_bin(distance_m: float) -> str:
    if 0.0 <= distance_m < 10.0:
        return "00_10m"
    if 10.0 <= distance_m < 20.0:
        return "10_20m"
    if 20.0 <= distance_m < 30.0:
        return "20_30m"
    if 30.0 <= distance_m <= MAX_DISTANCE_M:
        return "30_40m"
    raise QualificationError(f"qualified person distance outside [0,40]: {distance_m}")


def load_contract() -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    feasibility = read_json(FEASIBILITY_PATH)
    manifest = read_json(CACHE_ROOT / "cache_manifest.json")
    holdout = tuple(str(value) for value in feasibility.get("holdout_episodes", []))
    if not (
        feasibility.get("schema")
        == "splitfusion_fcos_person_instance_consolidation_result_v1"
        and feasibility.get("status") == "holdout_feasible"
        and feasibility.get("validation_or_test_accessed") is False
        and feasibility.get("selected_fit", {}).get("grid_index") == 27
        and all(
            feasibility["selected_fit"].get(name) == value
            for name, value in SELECTED_RULE.items()
        )
        and len(holdout) == 2
        and len(set(holdout)) == 2
    ):
        raise QualificationError("frozen consolidation feasibility contract drift")
    if not (
        manifest.get("schema")
        == "splitfusion_fcos_person_instance_consolidation_cache_v1"
        and manifest.get("split") == "train"
        and manifest.get("pass_count") == 1
        and manifest.get("validation_or_test_accessed") is False
        and manifest.get("canonical_person_threshold") == CANONICAL_SCORE_THRESHOLD
        and manifest.get("canonical_world_match_radius_m") == MATCH_RADIUS_M
        and tuple(manifest.get("episode_split", {}).get("holdout", [])) == holdout
        and manifest.get("stored_candidate_class") == "person_only"
        and manifest.get("original_candidate_labels_stored") is False
    ):
        raise QualificationError("frozen consolidation cache contract drift")
    return feasibility, manifest, holdout


def cache_hashes(manifest: Mapping[str, Any]) -> dict[str, Any]:
    shards = {
        str(item["path"]): sha256_file(CACHE_ROOT / str(item["path"]))
        for item in manifest["shards"]
    }
    encoded = json.dumps(shards, sort_keys=True, separators=(",", ":")).encode()
    return {
        "cache_manifest_sha256": sha256_file(CACHE_ROOT / "cache_manifest.json"),
        "shard_sha256": shards,
        "shard_hash_map_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def load_holdout_cache(
    manifest: Mapping[str, Any], holdout: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    holdout_set = set(holdout)
    frames: list[dict[str, Any]] = []
    candidate_count = 0
    for item in manifest["shards"]:
        payload = torch.load(
            CACHE_ROOT / str(item["path"]), map_location="cpu", weights_only=True
        )
        shard_frames = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(shard_frames, list):
            raise QualificationError(f"invalid cache shard: {item['path']}")
        for frame in shard_frames:
            if str(frame["experiment_id"]) not in holdout_set:
                continue
            frames.append(frame)
            candidate_count += int(frame["scores"].numel())
    sample_ids = [str(frame["sample_id"]) for frame in frames]
    expected = manifest["partition_counts"]["holdout"]
    if not (
        len(sample_ids) == int(expected["frames"])
        and len(sample_ids) == len(set(sample_ids))
        and candidate_count == int(expected["person_candidates"])
        and set(str(frame["experiment_id"]) for frame in frames) == holdout_set
    ):
        raise QualificationError("holdout cache cardinality or identity drift")
    return frames, {"frames": len(frames), "person_candidates": candidate_count}


def load_holdout_raw(
    frame_set: set[str], holdout: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    manifests: list[dict[str, Any]] = []
    boxes: list[dict[str, Any]] = []
    visibility: list[dict[str, Any]] = []
    depth_frames: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for episode in holdout:
        root = RAW_ROOT / episode
        paths = {
            "manifest": root / "manifest.csv",
            "object_boxes": root / "object_boxes.csv",
            "object_visibility": root / "object_visibility.csv",
            "depth_frames": root / "depth_frames.csv",
        }
        for path in paths.values():
            hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
        episode_manifests = pd.read_csv(paths["manifest"]).to_dict(orient="records")
        episode_boxes = pd.read_csv(
            paths["object_boxes"], dtype={"gt_actor_id": str}
        ).to_dict(orient="records")
        episode_visibility = pd.read_csv(
            paths["object_visibility"], dtype={"gt_actor_id": str}
        ).to_dict(orient="records")
        episode_depth = pd.read_csv(paths["depth_frames"]).to_dict(orient="records")
        manifests.extend(row for row in episode_manifests if str(row["sample_id"]) in frame_set)
        boxes.extend(row for row in episode_boxes if str(row["sample_id"]) in frame_set)
        visibility.extend(row for row in episode_visibility if str(row["sample_id"]) in frame_set)
        depth_frames.extend(row for row in episode_depth if str(row["sample_id"]) in frame_set)

    assert_unique(manifests, lambda row: str(row["sample_id"]), "raw manifest")
    assert_unique(depth_frames, lambda row: str(row["sample_id"]), "raw depth")
    assert_unique(boxes, actor_key, "raw box")
    assert_unique(visibility, actor_key, "raw visibility")
    if set(str(row["sample_id"]) for row in manifests) != frame_set:
        raise QualificationError("raw manifests do not exactly cover holdout cache frames")
    if set(str(row["sample_id"]) for row in depth_frames) != frame_set:
        raise QualificationError("raw depth records do not exactly cover holdout cache frames")

    manifest_by_sample = {str(row["sample_id"]): row for row in manifests}
    depth_by_sample = {str(row["sample_id"]): row for row in depth_frames}
    visibility_by_key = {actor_key(row): row for row in visibility}
    people = [row for row in boxes if str(row["label"]) == "person"]
    people_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qualified: list[dict[str, Any]] = []
    structural: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    for box in people:
        sample_id = str(box["sample_id"])
        people_by_sample[sample_id].append(box)
        key = actor_key(box)
        if key not in visibility_by_key:
            raise QualificationError(f"raw person has no visibility record: {key}")
        vis = visibility_by_key[key]
        meta = manifest_by_sample[sample_id]
        depth = depth_by_sample[sample_id]
        width, height = int(meta["camera_width"]), int(meta["camera_height"])
        if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            raise QualificationError(f"camera shape drift: {sample_id}")

        failures: list[str] = []
        if finite(box["gt_distance_m"], "person distance") > MAX_DISTANCE_M:
            failures.append("distance_gt_40m")
        center_x = finite(box["gt_center_x"], "projected center x")
        center_y = finite(box["gt_center_y"], "projected center y")
        if not (0.0 <= center_x < width and 0.0 <= center_y < height):
            failures.append("projected_box_center_outside_image")
        if (
            finite(box["gt_bbox_area_px"], "raw projected area") < MIN_PROJECTED_AREA_PX
            or finite(vis["clipped_projected_area_px"], "clipped projected area")
            < MIN_PROJECTED_AREA_PX
        ):
            failures.append("projected_area_lt_12px")
        if not truth(vis["geometry_qualified_v2"]):
            failures.append("registered_geometry_unqualified")
        if finite(box["object_sensor_x"], "camera-forward actor center") <= 0.0:
            failures.append("camera_forward_center_nonpositive")

        frame_id = int(meta["frame_id"])
        timestamp = finite(meta["timestamp"], "manifest timestamp")
        synchronized = (
            int(box["frame_id"]) == frame_id
            and int(vis["frame_id"]) == frame_id
            and int(vis["depth_frame_id"]) == frame_id
            and int(depth["frame_id"]) == frame_id
            and int(depth["rgb_frame_id"]) == frame_id
            and int(depth["semantic_frame_id"]) == frame_id
            and int(depth["depth_frame_id"]) == frame_id
            and int(depth["radar_frame_id"]) == frame_id
            and finite(box["timestamp"], "box timestamp") == timestamp
            and finite(vis["timestamp"], "visibility timestamp") == timestamp
            and finite(vis["depth_timestamp_s"], "visibility depth timestamp") == timestamp
            and finite(depth["rgb_timestamp_s"], "depth rgb timestamp") == timestamp
            and finite(depth["semantic_timestamp_s"], "depth semantic timestamp") == timestamp
            and finite(depth["depth_timestamp_s"], "depth timestamp") == timestamp
            and finite(depth["radar_timestamp_s"], "depth radar timestamp") == timestamp
            and finite(depth["max_timestamp_delta_s"], "max sync delta") == 0.0
            and truth(depth["depth_finite"])
            and truth(depth["depth_physically_plausible"])
            and str(vis["depth_path"]) == str(depth["depth_path"])
        )
        if not synchronized:
            failures.append("invalid_depth_synchronization")
        depth_path = RAW_ROOT / str(box["experiment_id"]) / str(vis["depth_path"])
        if not depth_path.is_file():
            failures.append("missing_depth_payload")

        if failures:
            structural.append(box)
            exclusion_reasons.update(failures)
        else:
            qualified.append(box)
    return {
        "manifest_by_sample": manifest_by_sample,
        "depth_by_sample": depth_by_sample,
        "visibility_by_key": visibility_by_key,
        "people_by_sample": dict(people_by_sample),
        "all_people": people,
        "qualified": qualified,
        "structural": structural,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
    }, hashes


def load_reference(holdout: Sequence[str]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    rows = pd.read_csv(REFERENCE_RECORDS, dtype={"gt_actor_id": str}).to_dict(
        orient="records"
    )
    selected = [row for row in rows if str(row["episode_id"]) in set(holdout)]
    assert_unique(selected, reference_key, "training actor-volume reference")
    by_key = {reference_key(row): row for row in selected}
    return by_key, {
        "training_support_records_sha256": sha256_file(REFERENCE_RECORDS),
        "training_reference_json_sha256": sha256_file(REFERENCE_ROOT / "training_reference.json"),
        "reference_hashes_json_sha256": sha256_file(REFERENCE_ROOT / "REFERENCE_HASHES.json"),
        "selected_holdout_rows": len(selected),
    }


def missing_actor_volume(
    missing: Sequence[Mapping[str, Any]], raw: Mapping[str, Any]
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for target in missing:
        grouped[str(target["sample_id"])].append(target)
    results: dict[tuple[str, str, str], dict[str, Any]] = {}
    depth_hashes: dict[str, str] = {}
    min_depth, max_depth = math.inf, -math.inf
    max_inverse_error = 0.0
    for sample_id in sorted(grouped):
        meta = raw["manifest_by_sample"][sample_id]
        depth_record = raw["depth_by_sample"][sample_id]
        episode = str(meta["experiment_id"])
        depth_path = RAW_ROOT / episode / str(depth_record["depth_path"])
        raw_bgra = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_bgra is None or raw_bgra.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 4):
            raise QualificationError(f"invalid saved depth payload: {depth_path}")
        depth_m = decode_depth_bgra(raw_bgra).astype(np.float64)
        if not np.all(np.isfinite(depth_m)):
            raise QualificationError(f"non-finite saved depth: {depth_path}")
        current_min, current_max = float(depth_m.min()), float(depth_m.max())
        if current_min <= 0.0 or current_max > core.CARLA_MAX_DEPTH_M:
            raise QualificationError(f"implausible saved depth range: {depth_path}")
        min_depth, max_depth = min(min_depth, current_min), max(max_depth, current_max)
        depth_hashes[str(depth_path.relative_to(REPO_ROOT))] = sha256_file(depth_path)

        camera_matrix = np.asarray(json.loads(meta["camera_matrix_json"]), dtype=np.float64)
        camera_inverse = np.asarray(
            json.loads(meta["camera_inverse_matrix_json"]), dtype=np.float64
        )
        intrinsics = np.asarray(
            [
                [float(meta["camera_fx"]), 0.0, float(meta["camera_cx"])],
                [0.0, float(meta["camera_fy"]), float(meta["camera_cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        identity_error = float(np.abs(camera_matrix @ camera_inverse - np.eye(4)).max())
        if not (
            np.all(np.isfinite(camera_matrix))
            and np.all(np.isfinite(camera_inverse))
            and np.all(np.isfinite(intrinsics))
            and identity_error < 1e-4
        ):
            raise QualificationError(f"invalid camera calibration: {sample_id}")
        max_inverse_error = max(max_inverse_error, identity_error)

        pedestrians = [
            {
                "key": str(person["gt_actor_id"]),
                "centre": (
                    float(person["object_world_x"]),
                    float(person["object_world_y"]),
                    float(person["object_world_z"]),
                ),
                "extent": (
                    float(person["gt_extent_x_m"]),
                    float(person["gt_extent_y_m"]),
                    float(person["gt_extent_z_m"]),
                ),
                "yaw_deg": float(person["object_yaw_deg"]),
            }
            for person in raw["people_by_sample"][sample_id]
        ]
        if len({person["key"] for person in pedestrians}) != len(pedestrians):
            raise QualificationError(f"duplicate person actor within frame: {sample_id}")
        for target in grouped[sample_id]:
            result = scoring.score_actor_frame(
                depth_m=depth_m,
                camera_matrix=camera_matrix,
                camera_inverse=camera_inverse,
                intrinsics=intrinsics,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                target_key=str(target["gt_actor_id"]),
                target_centre=(
                    float(target["object_world_x"]),
                    float(target["object_world_y"]),
                    float(target["object_world_z"]),
                ),
                target_extent=(
                    float(target["gt_extent_x_m"]),
                    float(target["gt_extent_y_m"]),
                    float(target["gt_extent_z_m"]),
                ),
                target_yaw_deg=float(target["object_yaw_deg"]),
                pedestrian_boxes=pedestrians,
            )
            if not (
                result["algorithm_version"] == core.ALGORITHM_VERSION
                and result["actor_volume_tolerance_m"] == 0.05
                and result["ground_reject_margin_m"] == 0.03
            ):
                raise QualificationError("actor-volume implementation constant drift")
            results[actor_key(target)] = {
                "avo": float(result["visibility"]),
                "no_support": bool(result["no_support"]),
            }
    return results, {
        "missing_rows_computed": len(missing),
        "depth_images_opened": len(grouped),
        "minimum_decoded_depth_m": min_depth if grouped else None,
        "maximum_decoded_depth_m": max_depth if grouped else None,
        "maximum_calibration_identity_error": max_inverse_error,
        "opened_depth_sha256": depth_hashes,
    }


def build_avo_table(
    raw: Mapping[str, Any], reference: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qualified_keys = {actor_key(row) for row in raw["qualified"]}
    if not set(reference).issubset(qualified_keys):
        unexpected = sorted(set(reference) - qualified_keys)[0]
        raise QualificationError(f"reference row is not canonically eligible: {unexpected}")
    missing = [row for row in raw["qualified"] if actor_key(row) not in reference]
    computed, computation = missing_actor_volume(missing, raw)
    rows: list[dict[str, Any]] = []
    reuse_by_episode: Counter[str] = Counter()
    missing_by_episode: Counter[str] = Counter()
    for target in raw["qualified"]:
        key = actor_key(target)
        distance = float(target["gt_distance_m"])
        if key in reference:
            source = reference[key]
            avo = float(source["raw_box_visibility"])
            no_support = truth(source["no_support"])
            reuse_by_episode[key[0]] += 1
            origin = "existing_train_reference"
        else:
            avo = float(computed[key]["avo"])
            no_support = bool(computed[key]["no_support"])
            missing_by_episode[key[0]] += 1
            origin = "saved_holdout_depth_missing_row_only"
        if not (math.isfinite(avo) and 0.0 <= avo <= 1.0):
            raise QualificationError(f"invalid AVO value: {key}/{avo}")
        rows.append(
            {
                "episode_id": key[0],
                "sample_id": key[1],
                "frame_id": int(target["frame_id"]),
                "gt_actor_id": key[2],
                "world_x": float(target["object_world_x"]),
                "world_y": float(target["object_world_y"]),
                "distance_m": distance,
                "distance_bin": distance_bin(distance),
                "actor_volume_observability": avo,
                "no_support": no_support,
                "source": origin,
            }
        )
    assert_unique(rows, reference_key, "holdout AVO table")
    return rows, {
        "all_raw_person_actor_frames": len(raw["all_people"]),
        "canonically_qualified_actor_frames": len(raw["qualified"]),
        "structurally_ignored_actor_frames": len(raw["structural"]),
        "reference_rows_reused": sum(reuse_by_episode.values()),
        "reference_rows_reused_by_episode": dict(sorted(reuse_by_episode.items())),
        "missing_rows_by_episode": dict(sorted(missing_by_episode.items())),
        "exclusion_reasons_nonexclusive": raw["exclusion_reasons"],
        **computation,
    }


def reconstruct_predictions(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    retained_020 = 0
    retained_025 = 0
    retained_by_episode_020: Counter[str] = Counter()
    retained_by_episode_025: Counter[str] = Counter()
    fields = ("original_indices", "boxes", "world_xy", "component_ids", "semantic_support")
    for frame in frames:
        selected = consolidate_person_candidates(
            scores=frame["scores"],
            boxes=frame["boxes"],
            world_xy=frame["world_xy"],
            component_ids=frame["component_ids"],
            semantic_support=frame["semantic_support"],
            original_indices=frame["original_indices"],
            semantic_support_threshold=SELECTED_RULE["semantic_support_threshold"],
            group_box_iou_threshold=SELECTED_RULE["group_box_iou_threshold"],
        )
        selected_scores = frame["scores"].index_select(0, selected).float()
        if selected_scores.numel() and not bool(
            (selected_scores >= CANONICAL_SCORE_THRESHOLD).all()
        ):
            raise QualificationError("consolidation retained a person below 0.20")
        p025_mask = selected_scores >= PERSON_SCORE_THRESHOLD
        selected_025 = selected.index_select(0, torch.where(p025_mask)[0])
        expected_025 = selected[torch.where(p025_mask)[0]]
        if not torch.equal(selected_025, expected_025):
            raise QualificationError("p025 is not an exact p020 position subset")
        for name in fields:
            p020_values = frame[name].index_select(0, selected)
            p025_values = frame[name].index_select(0, selected_025)
            if not torch.equal(p025_values, p020_values[p025_mask]):
                raise QualificationError(f"retained person field changed: {name}")
        if not torch.equal(
            frame["scores"].index_select(0, selected_025), selected_scores[p025_mask]
        ):
            raise QualificationError("retained person score changed")

        sample_id = str(frame["sample_id"])
        episode = str(frame["experiment_id"])
        predictions[sample_id] = [
            {
                "score": float(frame["scores"][position]),
                "world_x": float(frame["world_xy"][position, 0]),
                "world_y": float(frame["world_xy"][position, 1]),
                "original_index": int(frame["original_indices"][position]),
            }
            for position in selected.tolist()
        ]
        retained_020 += int(selected.numel())
        retained_025 += int(selected_025.numel())
        retained_by_episode_020[episode] += int(selected.numel())
        retained_by_episode_025[episode] += int(selected_025.numel())
    return predictions, {
        "retained_person_outputs_p020": retained_020,
        "retained_person_outputs_p025": retained_025,
        "retained_p020_by_episode": dict(sorted(retained_by_episode_020.items())),
        "retained_p025_by_episode": dict(sorted(retained_by_episode_025.items())),
        "p025_exact_subset_of_p020": True,
        "retained_person_non_score_fields_unchanged": True,
        "retained_person_scores_unchanged": True,
        "vehicle_outputs_exactly_unchanged": True,
        "vehicle_invariance_basis": (
            "the post-consolidation policy retains every non-person index and "
            "asserts every retained vehicle tensor field bitwise equal"
        ),
    }


def greedy_match(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    available: set[int] | None = None,
) -> tuple[dict[int, int], set[int]]:
    remaining = set(range(len(predictions))) if available is None else set(available)
    candidates: list[tuple[float, int, int]] = []
    for pred_index in sorted(remaining):
        for gt_index, target in enumerate(targets):
            distance = math.hypot(
                float(predictions[pred_index]["world_x"]) - float(target["world_x"]),
                float(predictions[pred_index]["world_y"]) - float(target["world_y"]),
            )
            if distance <= MATCH_RADIUS_M:
                candidates.append((distance, pred_index, gt_index))
    matched: dict[int, int] = {}
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for _distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_predictions or gt_index in used_targets:
            continue
        matched[pred_index] = gt_index
        used_predictions.add(pred_index)
        used_targets.add(gt_index)
    return matched, used_targets


def score_view(
    *,
    frame_ids: Sequence[str],
    episodes: Sequence[str],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    qualified_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    structural_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_by_sample: Mapping[str, str],
    detection_threshold: float,
) -> dict[str, Any]:
    def bucket() -> dict[str, Any]:
        return {
            "observable_gt": 0,
            "avo_ignored_gt": 0,
            "structural_ignored_gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "avo_ignored_predictions": 0,
            "structural_ignored_predictions": 0,
            "xy": [],
        }

    totals = bucket()
    episode_totals = {episode: bucket() for episode in episodes}
    for sample_id in frame_ids:
        episode = episode_by_sample[sample_id]
        current = episode_totals[episode]
        qualified = list(qualified_gt.get(sample_id, []))
        eligible = [
            row for row in qualified
            if float(row["actor_volume_observability"]) >= AVO_THRESHOLD
        ]
        avo_ignored = [
            row for row in qualified
            if float(row["actor_volume_observability"]) < AVO_THRESHOLD
        ]
        structural = list(structural_gt.get(sample_id, []))
        frame_predictions = [
            row for row in predictions.get(sample_id, [])
            if float(row["score"]) >= detection_threshold
        ]

        matched, used_eligible = greedy_match(frame_predictions, eligible)
        used_predictions = set(matched)
        for pred_index, gt_index in matched.items():
            target = eligible[gt_index]
            error = math.hypot(
                float(frame_predictions[pred_index]["world_x"]) - float(target["world_x"]),
                float(frame_predictions[pred_index]["world_y"]) - float(target["world_y"]),
            )
            totals["tp"] += 1
            totals["xy"].append(error)
            current["tp"] += 1
            current["xy"].append(error)
        remaining = set(range(len(frame_predictions))) - used_predictions
        matched_avo, _ = greedy_match(frame_predictions, avo_ignored, remaining)
        remaining -= set(matched_avo)
        matched_structural, _ = greedy_match(frame_predictions, structural, remaining)
        remaining -= set(matched_structural)

        counts = {
            "observable_gt": len(eligible),
            "avo_ignored_gt": len(avo_ignored),
            "structural_ignored_gt": len(structural),
            "fn": len(eligible) - len(used_eligible),
            "fp": len(remaining),
            "avo_ignored_predictions": len(matched_avo),
            "structural_ignored_predictions": len(matched_structural),
        }
        for name, count in counts.items():
            totals[name] += count
            current[name] += count

    def finalize(values: Mapping[str, Any]) -> dict[str, Any]:
        tp, fp, fn = int(values["tp"]), int(values["fp"]), int(values["fn"])
        observable = int(values["observable_gt"])
        if tp + fn != observable:
            raise QualificationError("TP+FN denominator failure")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / observable if observable else 0.0
        return {
            **{name: value for name, value in values.items() if name != "xy"},
            "ignored_predictions": int(values["avo_ignored_predictions"])
            + int(values["structural_ignored_predictions"]),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "xy_mae_m": sum(values["xy"]) / len(values["xy"])
            if values["xy"]
            else None,
        }

    return {
        "avo_threshold": AVO_THRESHOLD,
        "detection_score_threshold": detection_threshold,
        "overall": finalize(totals),
        "episodes": {
            episode: finalize(episode_totals[episode]) for episode in episodes
        },
        "matching_order": "observable_gt_then_avo_ignored_gt_then_structural_ignored_gt",
    }


def grouped_gt(
    table: Sequence[Mapping[str, Any]], raw: Mapping[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    qualified: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structural: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        qualified[str(row["sample_id"])].append(dict(row))
    for row in raw["structural"]:
        structural[str(row["sample_id"])].append(
            {
                "world_x": float(row["object_world_x"]),
                "world_y": float(row["object_world_y"]),
            }
        )
    return dict(qualified), dict(structural)


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv_x(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "episode_id", "sample_id", "frame_id", "gt_actor_id", "world_x", "world_y",
        "distance_m", "distance_bin", "actor_volume_observability", "no_support", "source",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def failure_report(result: Mapping[str, Any]) -> str:
    p025 = result["train_holdout"]["p025"]
    lines = [
        "# Person p025 train-holdout qualification",
        "",
        "The fixed 0.25 person score threshold did not satisfy every registered train-only gate. ",
        "The accepted p020 service remains unchanged; validation predictions were not accessed.",
        "",
        f"- Aggregate precision: {p025['overall']['precision']:.6f}",
        f"- Aggregate recall: {p025['overall']['recall']:.6f}",
    ]
    for episode, metrics in p025["episodes"].items():
        lines.append(
            f"- {episode}: precision {metrics['precision']:.6f}, recall {metrics['recall']:.6f}"
        )
    lines.extend(("", FAILURE, ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise QualificationError('refusing to run without CUDA_VISIBLE_DEVICES=""')
    output = args.output.resolve()
    if output != OUTPUT_DIR.resolve():
        raise QualificationError("output must be the registered calibration directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    original_avo = verify_original_unnormalized_sources()
    feasibility, cache_manifest, episodes = load_contract()
    exact_cache_hashes = cache_hashes(cache_manifest)
    frames, cache_counts = load_holdout_cache(cache_manifest, episodes)
    frame_ids = [str(frame["sample_id"]) for frame in frames]
    frame_set = set(frame_ids)
    raw, raw_hashes = load_holdout_raw(frame_set, episodes)
    reference, reference_hashes = load_reference(episodes)
    table, avo_diagnostics = build_avo_table(raw, reference)
    predictions, invariants = reconstruct_predictions(frames)
    qualified_gt, structural_gt = grouped_gt(table, raw)
    episode_by_sample = {
        str(frame["sample_id"]): str(frame["experiment_id"]) for frame in frames
    }
    p020 = score_view(
        frame_ids=frame_ids,
        episodes=episodes,
        predictions=predictions,
        qualified_gt=qualified_gt,
        structural_gt=structural_gt,
        episode_by_sample=episode_by_sample,
        detection_threshold=CANONICAL_SCORE_THRESHOLD,
    )
    p025 = score_view(
        frame_ids=frame_ids,
        episodes=episodes,
        predictions=predictions,
        qualified_gt=qualified_gt,
        structural_gt=structural_gt,
        episode_by_sample=episode_by_sample,
        detection_threshold=PERSON_SCORE_THRESHOLD,
    )

    gates = {
        "aggregate_precision_gte_0_70": p025["overall"]["precision"] >= AGGREGATE_MINIMUM,
        "aggregate_recall_gte_0_70": p025["overall"]["recall"] >= AGGREGATE_MINIMUM,
        "each_episode_precision_gte_0_65": all(
            row["precision"] >= EPISODE_MINIMUM for row in p025["episodes"].values()
        ),
        "each_episode_recall_gte_0_65": all(
            row["recall"] >= EPISODE_MINIMUM for row in p025["episodes"].values()
        ),
        "vehicle_outputs_exactly_unchanged": invariants["vehicle_outputs_exactly_unchanged"],
        "retained_person_non_score_fields_unchanged": invariants[
            "retained_person_non_score_fields_unchanged"
        ],
        "p025_exact_subset_of_p020": invariants["p025_exact_subset_of_p020"],
    }
    qualified = all(gates.values())
    elapsed = time.perf_counter() - started
    hashes = {
        "feasibility_result_sha256": sha256_file(FEASIBILITY_PATH),
        "cache": exact_cache_hashes,
        "reference": reference_hashes,
        "raw_holdout_metadata_sha256": raw_hashes,
        "actor_volume_source": original_avo,
    }
    result = {
        "schema": "splitfusion_fcos_person_p025_train_holdout_qualification_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": SUCCESS if qualified else FAILURE,
        "qualified": qualified,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "carla_started": False,
        "model_inference_run": False,
        "cache_rebuilt": False,
        "training_run": False,
        "validation_accessed": False,
        "test_accessed": False,
        "thresholds_evaluated": [CANONICAL_SCORE_THRESHOLD, PERSON_SCORE_THRESHOLD],
        "avo_threshold": AVO_THRESHOLD,
        "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
        "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
        "selected_consolidation_rule": SELECTED_RULE,
        "holdout_episodes": list(episodes),
        "cache_counts": cache_counts,
        "avo_table": avo_diagnostics,
        "output_invariants": invariants,
        "qualification_gates": gates,
        "train_holdout": {"p020": p020, "p025": p025},
        "input_hashes": hashes,
        "runtime_seconds": elapsed,
        "phase2_authorized": qualified,
    }
    write_csv_x(output / "holdout_actor_volume_observability_table.csv", table)
    write_json_x(output / "INPUT_HASHES.json", hashes)
    write_json_x(output / "train_holdout_qualification.json", result)
    if qualified:
        (output / SUCCESS).write_text(SUCCESS + "\n", encoding="utf-8")
    else:
        (output / "FINAL_REPORT.md").write_text(failure_report(result), encoding="utf-8")
        (output / FAILURE).write_text(FAILURE + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal": result["terminal"],
        "runtime_seconds": elapsed,
        "gates": gates,
        "p020": p020,
        "p025": p025,
        "avo_table": avo_diagnostics,
    }, indent=2, sort_keys=True))
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
