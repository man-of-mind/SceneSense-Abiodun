#!/usr/bin/env python3
"""One-shot CPU-only AVO comparison of three frozen validation prediction sets.

The visibility definition is the original unnormalized actor-volume score from
commit dc5238d.  Human annotations are not loaded by this program; only the
already-published aggregate human-band model comparison is reused in the final
context table after all AVO scoring has completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import struct
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import core, scoring
from data_collection.route_b_publication_actor_volume_visibility_v1.run_audit import (
    assert_registered_decoder,
    decode_depth_bgra,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PARENT = REPO_ROOT / "experiments/actor_volume_observability_model_comparison_v1"
RAW_ROOT = REPO_ROOT / "data_collection/experiments/route_b_perception_v3"
CANONICAL_EXPERIMENT = (
    REPO_ROOT
    / "experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151"
)
PILOT_ROOT = (
    REPO_ROOT
    / "data_collection/experiments/route_b_publication_actor_volume_visibility_v1"
    / "20260901_191239"
)
HUMAN_AGGREGATE = (
    REPO_ROOT
    / "data_collection/experiments/route_b_publication_human_occlusion_pilot_v1"
    / "20260901_030234_seed20260831/human_visibility_band_model_comparison_v1.json"
)

EPISODES = (
    "canonical_v3_05_val_30_30_s601_tm1601",
    "canonical_v3_06_val_50_50_s602_tm1602",
)
AVO_THRESHOLDS = (0.10, 0.25, 0.50, 0.65, 0.70, 0.85)
DETECTION_THRESHOLDS = (0.20, 0.02)
HUMAN_SUPPORTED_AVO_THRESHOLD = 0.65
MAX_DISTANCE_M = 40.0
MIN_PROJECTED_AREA_PX = 12.0
MATCH_RADIUS_M = 3.0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
EXPECTED_VALIDATION_FRAMES = 3345
EXPECTED_PILOT_ROWS = 100
EXPECTED_PILOT_BALANCED_ACCURACY = 0.8522727272727273
TERMINAL = "ACTOR_VOLUME_OBSERVABILITY_MODEL_COMPARISON_COMPLETE"

ORIGINAL_UNNORMALIZED_SOURCE_HASHES = {
    "data_collection/route_b_publication_actor_volume_visibility_v1/core.py":
        "02d39fd1d15c31ab8323f0ece09e951e8fc0e42aecf5de06a120918276f1d3e4",
    "data_collection/route_b_publication_actor_volume_visibility_v1/scoring.py":
        "3942b8c35c990c27676fc66c13f80ed8011e2a2ddc0768cac5c6e531f85b00dd",
    "data_collection/route_b_publication_actor_volume_visibility_v1/run_audit.py":
        "cc6a0afa9997d6d1ccc603ec990046e8c6b0fcbe9ba9b442abaad0589efa7701",
}

DISTANCE_BINS = (
    ("00_10m", 0.0, 10.0),
    ("10_20m", 10.0, 20.0),
    ("20_30m", 20.0, 30.0),
    ("30_40m", 30.0, 40.0000000001),
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "splitfusion_fcos": {
        "display_name": "SplitFusion-FCOS",
        "prediction_root": REPO_ROOT / "experiments/splitfusion_fcos_service_candidate_v1/predictions",
        "canonical_result": REPO_ROOT / "experiments/splitfusion_fcos_service_candidate_v1/predictions/evaluation_v010.json",
        "expected_detection_sha256": "a682a1fc5eabb2e59e07449a8c6b5fc604077b40ef094b57dc30c5a18d7ec260",
        "human_key": "splitfusion_fcos_epoch26_service_candidate",
    },
    "joint_lraspp": {
        "display_name": "Joint LR-ASPP",
        "prediction_root": REPO_ROOT / "experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_060656/predictions/epoch_010",
        "canonical_result": REPO_ROOT / "experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_060656/evaluation/epoch_010.json",
        "expected_detection_sha256": "49830ff0bd0d77468efa0443f47474c7581c6b8442b5eeb339822fb12b7ab292",
        "human_key": "joint_lraspp_epoch10",
    },
    "two_stage_lraspp": {
        "display_name": "Two-stage LR-ASPP",
        "prediction_root": REPO_ROOT / "experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/stage2/predictions/epoch_030",
        "canonical_result": REPO_ROOT / "experiments/route_b_v3_1_depth_aware_lraspp_two_stage_v1/20260829_184743/stage2/evaluation/epoch_030.json",
        "expected_detection_sha256": "21295344987b37b0995520caacb98188ae017de62bd90de841f73a2bb162f92b",
        "human_key": "two_stage_lraspp_epoch30",
    },
}

TABLE_FIELDS = (
    "sample_id", "episode_id", "frame_id", "gt_actor_id", "label",
    "object_world_x", "object_world_y", "object_world_z", "distance_m",
    "distance_bin", "raw_gt_center_x", "raw_gt_center_y",
    "raw_projected_area_px", "depth_path", "depth_frame_id",
    "depth_timestamp_s", "algorithm_version", "actor_volume_tolerance_m",
    "ground_reject_margin_m", "unclipped_bbox_x", "unclipped_bbox_y",
    "unclipped_bbox_w", "unclipped_bbox_h", "unclipped_projected_area_px",
    "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h",
    "clipped_projected_area_px", "actor_near_depth_m", "actor_far_depth_m",
    "sampled_roi_px", "valid_depth_px", "retained_actor_point_count",
    "competing_actor_boxes", "actor_volume_observability", "truncation",
    "no_support", "visible_bbox_x", "visible_bbox_y", "visible_bbox_w",
    "visible_bbox_h", "visible_box_area_px", "visible_box_raster_ratio",
    "degenerate_visible_box", "visible_box_height_ratio",
    "visible_box_width_ratio", "actor_point_surface_occupancy",
    "visible_box_fill_ratio", "retained_local_z_min_above_floor_m",
    "retained_max_abs_local_excess_m",
)


class QualificationError(RuntimeError):
    """Fail-closed provenance, reproduction, or evaluator qualification error."""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_csv_x(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_float(value: Any, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise QualificationError(f"non-finite {context}: {value!r}")
    return number


def distance_bin(distance_m: float) -> str:
    for name, low, high in DISTANCE_BINS:
        if low <= distance_m < high:
            return name
    raise QualificationError(f"qualified distance outside registered bins: {distance_m}")


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["sample_id"]), str(row["gt_actor_id"]), str(row["label"])


def assert_unique(rows: Sequence[Mapping[str, Any]], key_fn: Any, context: str) -> None:
    keys = [key_fn(row) for row in rows]
    if len(keys) != len(set(keys)):
        duplicate = next(key for key, count in Counter(keys).items() if count > 1)
        raise QualificationError(f"duplicate {context} key: {duplicate}")


def exact_double_equal(left: Any, right: Any) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def verify_original_unnormalized_sources() -> dict[str, Any]:
    actual = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in ORIGINAL_UNNORMALIZED_SOURCE_HASHES
    }
    if actual != ORIGINAL_UNNORMALIZED_SOURCE_HASHES:
        raise QualificationError(
            "original unnormalized actor-volume implementation differs from commit dc5238d"
        )
    decoder = assert_registered_decoder()
    return {
        "source_commit": "dc5238d",
        "source_hashes": actual,
        "normalized_variants_imported": False,
        "registered_decoder": decoder,
    }


def load_raw_sources() -> dict[str, Any]:
    canonical_manifest_path = CANONICAL_EXPERIMENT / "dataset/manifest.csv"
    canonical_rows = [
        row for row in read_csv(canonical_manifest_path) if row["split"] == "val"
    ]
    frame_ids = [row["sample_id"] for row in canonical_rows]
    if len(frame_ids) != EXPECTED_VALIDATION_FRAMES or len(set(frame_ids)) != len(frame_ids):
        raise QualificationError("frozen validation frame count/uniqueness drift")
    frame_set = set(frame_ids)

    manifests: list[dict[str, str]] = []
    boxes: list[dict[str, str]] = []
    visibility: list[dict[str, str]] = []
    depth_frames: list[dict[str, str]] = []
    input_hashes: dict[str, str] = {
        str(canonical_manifest_path.relative_to(REPO_ROOT)): sha256_file(canonical_manifest_path)
    }
    raw_frame_counts: dict[str, dict[str, int]] = {}
    for episode in EPISODES:
        episode_root = RAW_ROOT / episode
        paths = {
            "manifest": episode_root / "manifest.csv",
            "object_boxes": episode_root / "object_boxes.csv",
            "object_visibility": episode_root / "object_visibility.csv",
            "depth_frames": episode_root / "depth_frames.csv",
        }
        for path in paths.values():
            input_hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
        episode_manifest = read_csv(paths["manifest"])
        retained = [row for row in episode_manifest if row["sample_id"] in frame_set]
        manifests.extend(retained)
        boxes.extend(row for row in read_csv(paths["object_boxes"]) if row["sample_id"] in frame_set)
        visibility.extend(
            row for row in read_csv(paths["object_visibility"]) if row["sample_id"] in frame_set
        )
        depth_frames.extend(
            row for row in read_csv(paths["depth_frames"]) if row["sample_id"] in frame_set
        )
        raw_frame_counts[episode] = {
            "raw": len(episode_manifest),
            "frozen_validation": len(retained),
            "excluded_before_table": len(episode_manifest) - len(retained),
        }

    assert_unique(manifests, lambda row: row["sample_id"], "raw manifest")
    assert_unique(boxes, row_key, "raw actor box")
    assert_unique(visibility, row_key, "raw actor visibility")
    assert_unique(depth_frames, lambda row: row["sample_id"], "raw depth-frame")
    if set(row["sample_id"] for row in manifests) != frame_set:
        raise QualificationError("raw episode manifests do not exactly cover frozen validation")
    if set(row["sample_id"] for row in depth_frames) != frame_set:
        raise QualificationError("raw depth-frame records do not exactly cover frozen validation")
    if set(frame_ids) != set(row["sample_id"] for row in canonical_rows):
        raise QualificationError("canonical manifest reconciliation failure")

    manifest_by_sample = {row["sample_id"]: row for row in manifests}
    depth_by_sample = {row["sample_id"]: row for row in depth_frames}
    visibility_by_key = {row_key(row): row for row in visibility}
    all_people = [row for row in boxes if row["label"] == "person"]
    people_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for person in all_people:
        people_by_sample[person["sample_id"]].append(person)

    qualified: list[dict[str, str]] = []
    structural_ignored: list[dict[str, str]] = []
    exclusion_reasons: Counter[str] = Counter()
    for box in all_people:
        key = row_key(box)
        if key not in visibility_by_key:
            raise QualificationError(f"person box has no raw visibility row: {key}")
        vis = visibility_by_key[key]
        meta = manifest_by_sample[box["sample_id"]]
        depth = depth_by_sample[box["sample_id"]]
        width, height = int(meta["camera_width"]), int(meta["camera_height"])
        if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            raise QualificationError(f"camera shape drift for {box['sample_id']}")

        failures: list[str] = []
        if finite_float(box["gt_distance_m"], "person distance") > MAX_DISTANCE_M:
            failures.append("distance_gt_40m")
        center_x = finite_float(box["gt_center_x"], "projected center x")
        center_y = finite_float(box["gt_center_y"], "projected center y")
        if not (0.0 <= center_x < width and 0.0 <= center_y < height):
            failures.append("projected_box_center_outside_image")
        if (
            finite_float(box["gt_bbox_area_px"], "raw projected area")
            < MIN_PROJECTED_AREA_PX
            or finite_float(vis["clipped_projected_area_px"], "visibility projected area")
            < MIN_PROJECTED_AREA_PX
        ):
            failures.append("projected_area_lt_12px")
        if not truth(vis["geometry_qualified_v2"]):
            failures.append("registered_geometry_unqualified")
        if finite_float(box["object_sensor_x"], "camera-forward actor center") <= 0.0:
            failures.append("camera_forward_center_nonpositive")

        frame_id = int(meta["frame_id"])
        timestamp = finite_float(meta["timestamp"], "manifest timestamp")
        sync_ok = (
            int(box["frame_id"]) == frame_id
            and int(vis["frame_id"]) == frame_id
            and int(vis["depth_frame_id"]) == frame_id
            and int(depth["frame_id"]) == frame_id
            and int(depth["rgb_frame_id"]) == frame_id
            and int(depth["semantic_frame_id"]) == frame_id
            and int(depth["depth_frame_id"]) == frame_id
            and int(depth["radar_frame_id"]) == frame_id
            and finite_float(box["timestamp"], "box timestamp") == timestamp
            and finite_float(vis["timestamp"], "visibility timestamp") == timestamp
            and finite_float(vis["depth_timestamp_s"], "visibility depth timestamp") == timestamp
            and finite_float(depth["rgb_timestamp_s"], "depth rgb timestamp") == timestamp
            and finite_float(depth["semantic_timestamp_s"], "depth semantic timestamp") == timestamp
            and finite_float(depth["depth_timestamp_s"], "depth timestamp") == timestamp
            and finite_float(depth["radar_timestamp_s"], "depth radar timestamp") == timestamp
            and finite_float(depth["max_timestamp_delta_s"], "max sync delta") == 0.0
            and truth(depth["depth_finite"])
            and truth(depth["depth_physically_plausible"])
            and vis["depth_path"] == depth["depth_path"]
        )
        if not sync_ok:
            failures.append("invalid_depth_synchronization")
        depth_path = RAW_ROOT / box["experiment_id"] / vis["depth_path"]
        if not depth_path.is_file():
            failures.append("missing_depth_payload")

        if failures:
            structural_ignored.append(box)
            exclusion_reasons.update(failures)
        else:
            qualified.append(box)

    return {
        "frame_ids": frame_ids,
        "frame_set": frame_set,
        "manifest_by_sample": manifest_by_sample,
        "depth_by_sample": depth_by_sample,
        "visibility_by_key": visibility_by_key,
        "people_by_sample": dict(people_by_sample),
        "all_people": all_people,
        "qualified": qualified,
        "structural_ignored": structural_ignored,
        "raw_frame_counts": raw_frame_counts,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "input_hashes": input_hashes,
    }


def score_avo_table(raw: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qualified_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw["qualified"]:
        qualified_by_sample[row["sample_id"]].append(row)

    rows: list[dict[str, Any]] = []
    depth_images_opened = 0
    max_calibration_identity_error = 0.0
    min_depth_m = math.inf
    max_depth_m = -math.inf
    episode_counts: Counter[str] = Counter()
    for frame_index, sample_id in enumerate(raw["frame_ids"], 1):
        targets = qualified_by_sample.get(sample_id, [])
        if not targets:
            continue
        meta = raw["manifest_by_sample"][sample_id]
        depth_record = raw["depth_by_sample"][sample_id]
        episode_id = str(meta["experiment_id"])
        depth_path = RAW_ROOT / episode_id / depth_record["depth_path"]
        raw_bgra = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_bgra is None or raw_bgra.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 4):
            raise QualificationError(f"invalid depth payload: {depth_path}")
        depth_m = decode_depth_bgra(raw_bgra).astype(np.float64)
        if not np.all(np.isfinite(depth_m)):
            raise QualificationError(f"non-finite decoded depth: {depth_path}")
        current_min, current_max = float(depth_m.min()), float(depth_m.max())
        if current_min <= 0.0 or current_max > core.CARLA_MAX_DEPTH_M:
            raise QualificationError(f"implausible decoded depth range: {depth_path}")
        depth_images_opened += 1
        min_depth_m = min(min_depth_m, current_min)
        max_depth_m = max(max_depth_m, current_max)

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
        if not (
            np.all(np.isfinite(camera_matrix))
            and np.all(np.isfinite(camera_inverse))
            and np.all(np.isfinite(intrinsics))
        ):
            raise QualificationError(f"non-finite calibration: {sample_id}")
        identity_error = float(np.abs(camera_matrix @ camera_inverse - np.eye(4)).max())
        if identity_error >= 1e-4:
            raise QualificationError(f"invalid calibration inverse: {sample_id}/{identity_error}")
        max_calibration_identity_error = max(max_calibration_identity_error, identity_error)

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
            raise QualificationError(f"duplicate pedestrian actor ID within frame: {sample_id}")

        for target in targets:
            key = row_key(target)
            vis = raw["visibility_by_key"][key]
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
            distance = float(target["gt_distance_m"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "episode_id": episode_id,
                    "frame_id": int(target["frame_id"]),
                    "gt_actor_id": str(target["gt_actor_id"]),
                    "label": "person",
                    "object_world_x": float(target["object_world_x"]),
                    "object_world_y": float(target["object_world_y"]),
                    "object_world_z": float(target["object_world_z"]),
                    "distance_m": distance,
                    "distance_bin": distance_bin(distance),
                    "raw_gt_center_x": float(target["gt_center_x"]),
                    "raw_gt_center_y": float(target["gt_center_y"]),
                    "raw_projected_area_px": float(target["gt_bbox_area_px"]),
                    "depth_path": str(Path(episode_id) / vis["depth_path"]),
                    "depth_frame_id": int(vis["depth_frame_id"]),
                    "depth_timestamp_s": float(vis["depth_timestamp_s"]),
                    **{
                        ("actor_volume_observability" if name == "visibility" else name): value
                        for name, value in result.items()
                        if name != "visibility_band"
                    },
                }
            )
            episode_counts[episode_id] += 1
        if frame_index % 500 == 0:
            print(
                f"[AVO table] frame {frame_index}/{len(raw['frame_ids'])}; rows={len(rows)}",
                flush=True,
            )

    if len(rows) != len(raw["qualified"]):
        raise QualificationError("AVO table cardinality mismatch")
    if len({(row["sample_id"], row["gt_actor_id"]) for row in rows}) != len(rows):
        raise QualificationError("AVO table actor-frame uniqueness failure")
    diagnostics = {
        "qualified_person_actor_frames": len(rows),
        "structurally_ignored_raw_person_actor_frames": len(raw["structural_ignored"]),
        "all_raw_person_actor_frames_on_validation_frames": len(raw["all_people"]),
        "qualified_by_episode": dict(sorted(episode_counts.items())),
        "depth_images_opened": depth_images_opened,
        "minimum_decoded_depth_m": min_depth_m,
        "maximum_decoded_depth_m": max_depth_m,
        "maximum_calibration_identity_error": max_calibration_identity_error,
        "no_support_count": sum(bool(row["no_support"]) for row in rows),
        "truncated_count": sum(float(row["truncation"]) > 0.0 for row in rows),
        "exclusion_reasons_nonexclusive": raw["exclusion_reasons"],
    }
    return rows, diagnostics


def verify_pilot_reproduction(table: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pilot_path = PILOT_ROOT / "actor_volume_visibility_scores.csv"
    metadata_path = PILOT_ROOT / "RUN_METADATA.json"
    pilot = read_csv(pilot_path)
    metadata = read_json(metadata_path)
    if len(pilot) != EXPECTED_PILOT_ROWS:
        raise QualificationError(f"pilot row count drift: {len(pilot)}")
    balanced_accuracy = float(
        metadata["agreement"]["actor_volume"]["balanced_accuracy"]
    )
    exact_four_band_agreement = float(
        metadata["agreement"]["actor_volume"]["exact_agreement"]
    )
    linear_weighted_kappa = float(
        metadata["agreement"]["actor_volume"]["linear_weighted_cohen_kappa"]
    )
    if not exact_double_equal(balanced_accuracy, EXPECTED_PILOT_BALANCED_ACCURACY):
        raise QualificationError("registered pilot balanced-accuracy drift")

    by_key = {(str(row["sample_id"]), str(row["gt_actor_id"])): row for row in table}
    fields = (
        "retained_actor_point_count",
        "unclipped_projected_area_px",
        "clipped_projected_area_px",
        "visible_box_area_px",
        "actor_volume_observability",
        "truncation",
        "no_support",
    )
    mismatches: list[dict[str, Any]] = []
    for expected in pilot:
        key = (expected["sample_id"], expected["gt_actor_id"])
        actual = by_key.get(key)
        if actual is None:
            mismatches.append({"key": key, "field": "row", "expected": "present", "actual": "missing"})
            continue
        comparisons = {
            "retained_actor_point_count": int(expected["retained_actor_point_count"])
            == int(actual["retained_actor_point_count"]),
            "unclipped_projected_area_px": exact_double_equal(
                expected["unclipped_projected_area_px"], actual["unclipped_projected_area_px"]
            ),
            "clipped_projected_area_px": exact_double_equal(
                expected["clipped_projected_area_px"], actual["clipped_projected_area_px"]
            ),
            "visible_box_area_px": exact_double_equal(
                expected["visible_box_area_px"], actual["visible_box_area_px"]
            ),
            "actor_volume_observability": exact_double_equal(
                expected["visibility"], actual["actor_volume_observability"]
            ),
            "truncation": exact_double_equal(expected["truncation"], actual["truncation"]),
            "no_support": truth(expected["no_support"]) == bool(actual["no_support"]),
        }
        for field in fields:
            if not comparisons[field]:
                expected_field = "visibility" if field == "actor_volume_observability" else field
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": expected[expected_field],
                        "actual": actual[field],
                    }
                )
    result = {
        "pilot_rows": len(pilot),
        "fields": list(fields),
        "exact_matches": len(pilot) * len(fields) - len(mismatches),
        "expected_comparisons": len(pilot) * len(fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "pilot_balanced_accuracy": balanced_accuracy,
        "pilot_balanced_accuracy_rounded": round(balanced_accuracy, 4),
        "pilot_exact_four_band_agreement": exact_four_band_agreement,
        "pilot_linear_weighted_four_band_kappa": linear_weighted_kappa,
        "pilot_scores_sha256": sha256_file(pilot_path),
        "pilot_metadata_sha256": sha256_file(metadata_path),
        "passed": not mismatches,
    }
    if mismatches:
        raise QualificationError(f"exact 100-person pilot reproduction failed: {mismatches[:3]}")
    return result


def load_person_predictions(path: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "class_name", "score", "world_x", "world_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise QualificationError(f"prediction schema drift: {path}")
        for row in reader:
            total_rows += 1
            if row["class_name"] != "person":
                continue
            item = {
                "score": finite_float(row["score"], "prediction score"),
                "world_x": finite_float(row["world_x"], "prediction world_x"),
                "world_y": finite_float(row["world_y"], "prediction world_y"),
                "prediction_index": int(row.get("prediction_index", len(grouped[row["sample_id"]]))),
            }
            grouped[row["sample_id"]].append(item)
    for values in grouped.values():
        values.sort(key=lambda item: -float(item["score"]))
    return dict(grouped), total_rows


def gt_from_table(table: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        grouped[str(row["sample_id"])].append(
            {
                "sample_id": str(row["sample_id"]),
                "episode_id": str(row["episode_id"]),
                "gt_actor_id": str(row["gt_actor_id"]),
                "world_x": float(row["object_world_x"]),
                "world_y": float(row["object_world_y"]),
                "distance_m": float(row["distance_m"]),
                "distance_bin": str(row["distance_bin"]),
                "avo": float(row["actor_volume_observability"]),
                "no_support": bool(row["no_support"]),
                "qualified": True,
            }
        )
    return dict(grouped)


def structural_gt(raw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw["structural_ignored"]:
        grouped[row["sample_id"]].append(
            {
                "sample_id": row["sample_id"],
                "episode_id": row["experiment_id"],
                "gt_actor_id": str(row["gt_actor_id"]),
                "world_x": float(row["object_world_x"]),
                "world_y": float(row["object_world_y"]),
                "qualified": False,
            }
        )
    return dict(grouped)


def greedy_match(
    predictions: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    available_prediction_indices: set[int] | None = None,
) -> tuple[dict[int, int], set[int]]:
    available = (
        set(range(len(predictions)))
        if available_prediction_indices is None
        else set(available_prediction_indices)
    )
    candidates: list[tuple[float, int, int]] = []
    for pred_index in sorted(available):
        prediction = predictions[pred_index]
        for gt_index, target in enumerate(targets):
            distance = math.hypot(
                float(prediction["world_x"]) - float(target["world_x"]),
                float(prediction["world_y"]) - float(target["world_y"]),
            )
            if distance <= MATCH_RADIUS_M:
                candidates.append((distance, pred_index, gt_index))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matched: dict[int, int] = {}
    for _distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_predictions or gt_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(gt_index)
        matched[pred_index] = gt_index
    return matched, used_targets


def empty_slice() -> dict[str, Any]:
    return {"eligible_gt": 0, "tp": 0, "fn": 0, "xy": []}


def finalize_recall_slice(bucket: Mapping[str, Any]) -> dict[str, Any]:
    eligible, tp, fn = int(bucket["eligible_gt"]), int(bucket["tp"]), int(bucket["fn"])
    if tp + fn != eligible:
        raise QualificationError("slice TP+FN denominator failure")
    return {
        "eligible_gt": eligible,
        "tp": tp,
        "fn": fn,
        "recall": tp / eligible if eligible else None,
        "xy_mae_m": sum(bucket["xy"]) / len(bucket["xy"]) if bucket["xy"] else None,
    }


def score_person_view(
    *,
    frame_ids: Sequence[str],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    qualified_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    structural_ignored_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    episode_by_sample: Mapping[str, str],
    avo_threshold: float,
    detection_threshold: float,
) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "observable_gt": 0,
        "avo_ignored_gt": 0,
        "structural_ignored_gt": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ignored_predictions": 0,
        "avo_ignored_predictions": 0,
        "structural_ignored_predictions": 0,
        "xy": [],
    }
    episode_buckets: dict[str, dict[str, Any]] = {
        episode: {
            "observable_gt": 0,
            "avo_ignored_gt": 0,
            "structural_ignored_gt": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ignored_predictions": 0,
            "avo_ignored_predictions": 0,
            "structural_ignored_predictions": 0,
            "xy": [],
        }
        for episode in EPISODES
    }
    distance_buckets = {name: empty_slice() for name, _low, _high in DISTANCE_BINS}

    for sample_id in frame_ids:
        qualified = list(qualified_gt.get(sample_id, []))
        eligible = [row for row in qualified if float(row["avo"]) >= avo_threshold]
        avo_ignored = [row for row in qualified if float(row["avo"]) < avo_threshold]
        structural = list(structural_ignored_gt.get(sample_id, []))
        episode = episode_by_sample[sample_id]
        if episode not in episode_buckets:
            raise QualificationError(f"unexpected episode identity: {episode}")
        ep = episode_buckets[str(episode)]
        frame_predictions = [
            row
            for row in predictions.get(sample_id, [])
            if float(row["score"]) >= detection_threshold
        ]

        matched, used_eligible = greedy_match(frame_predictions, eligible)
        used_prediction_indices = set(matched)
        for pred_index, gt_index in matched.items():
            target = eligible[gt_index]
            distance = math.hypot(
                float(frame_predictions[pred_index]["world_x"]) - float(target["world_x"]),
                float(frame_predictions[pred_index]["world_y"]) - float(target["world_y"]),
            )
            totals["tp"] += 1
            totals["xy"].append(distance)
            ep["tp"] += 1
            ep["xy"].append(distance)
            band = distance_buckets[str(target["distance_bin"])]
            band["tp"] += 1
            band["xy"].append(distance)

        remaining = set(range(len(frame_predictions))) - used_prediction_indices
        matched_avo_ignore, _ = greedy_match(frame_predictions, avo_ignored, remaining)
        remaining -= set(matched_avo_ignore)
        matched_structural_ignore, _ = greedy_match(frame_predictions, structural, remaining)
        remaining -= set(matched_structural_ignore)

        ignored_avo_count = len(matched_avo_ignore)
        ignored_structural_count = len(matched_structural_ignore)
        fp_count = len(remaining)
        totals["avo_ignored_predictions"] += ignored_avo_count
        totals["structural_ignored_predictions"] += ignored_structural_count
        totals["ignored_predictions"] += ignored_avo_count + ignored_structural_count
        totals["fp"] += fp_count
        ep["avo_ignored_predictions"] += ignored_avo_count
        ep["structural_ignored_predictions"] += ignored_structural_count
        ep["ignored_predictions"] += ignored_avo_count + ignored_structural_count
        ep["fp"] += fp_count

        totals["observable_gt"] += len(eligible)
        totals["avo_ignored_gt"] += len(avo_ignored)
        totals["structural_ignored_gt"] += len(structural)
        totals["fn"] += len(eligible) - len(used_eligible)
        ep["observable_gt"] += len(eligible)
        ep["avo_ignored_gt"] += len(avo_ignored)
        ep["structural_ignored_gt"] += len(structural)
        ep["fn"] += len(eligible) - len(used_eligible)
        for index, target in enumerate(eligible):
            band = distance_buckets[str(target["distance_bin"])]
            band["eligible_gt"] += 1
            if index not in used_eligible:
                band["fn"] += 1

    if totals["tp"] + totals["fn"] != totals["observable_gt"]:
        raise QualificationError("overall TP+FN denominator failure")
    if totals["observable_gt"] + totals["avo_ignored_gt"] != sum(
        len(values) for values in qualified_gt.values()
    ):
        raise QualificationError("AVO eligibility partition failure")
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / totals["observable_gt"] if totals["observable_gt"] else 0.0
    overall = {
        key: value for key, value in totals.items() if key != "xy"
    }
    overall.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "xy_mae_m": sum(totals["xy"]) / len(totals["xy"]) if totals["xy"] else None,
        }
    )
    episodes: dict[str, Any] = {}
    for episode, bucket in episode_buckets.items():
        tp, fp, fn = int(bucket["tp"]), int(bucket["fp"]), int(bucket["fn"])
        observable = int(bucket["observable_gt"])
        if tp + fn != observable:
            raise QualificationError(f"episode TP+FN denominator failure: {episode}")
        ep_precision = tp / (tp + fp) if tp + fp else 0.0
        ep_recall = tp / observable if observable else 0.0
        episodes[episode] = {
            **{key: value for key, value in bucket.items() if key != "xy"},
            "precision": ep_precision,
            "recall": ep_recall,
            "f1": 2.0 * ep_precision * ep_recall / (ep_precision + ep_recall)
            if ep_precision + ep_recall
            else 0.0,
            "xy_mae_m": sum(bucket["xy"]) / len(bucket["xy"]) if bucket["xy"] else None,
        }
    return {
        "avo_threshold": avo_threshold,
        "detection_score_threshold": detection_threshold,
        "overall": overall,
        "episodes": episodes,
        "distance_bins": {
            name: finalize_recall_slice(bucket) for name, bucket in distance_buckets.items()
        },
    }


def canonical_person_result(canonical: Mapping[str, Any]) -> dict[str, Any]:
    primary = canonical["primary_v010"]
    at_020 = primary["0.20"]["classes"]["person"]
    at_002 = primary["0.02"]["classes"]["person"]
    return {
        **at_020,
        "recall_at_detection_score_0_02": at_002["recall"],
    }


def score_models(
    raw: Mapping[str, Any], table: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, str]]:
    qualified_gt = gt_from_table(table)
    ignored_gt = structural_gt(raw)
    episode_by_sample = {
        sample_id: str(meta["experiment_id"])
        for sample_id, meta in raw["manifest_by_sample"].items()
    }
    output: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for model_key, spec in MODEL_SPECS.items():
        prediction_root = Path(spec["prediction_root"])
        detections = prediction_root / "detections.csv"
        manifest_path = prediction_root / "inference_manifest.json"
        canonical_path = Path(spec["canonical_result"])
        detection_hash = sha256_file(detections)
        manifest = read_json(manifest_path)
        canonical = read_json(canonical_path)
        expected = str(spec["expected_detection_sha256"])
        if not (
            detection_hash == expected
            and manifest["detections_sha256"] == expected
            and canonical["detections_sha256"] == expected
            and int(manifest["inference_pass_count"]) == 1
            and int(manifest["validation_frames"]) == EXPECTED_VALIDATION_FRAMES
        ):
            raise QualificationError(f"frozen prediction provenance failure: {model_key}")
        print(f"[predictions] loading frozen {spec['display_name']}", flush=True)
        predictions, total_predictions = load_person_predictions(detections)
        if set(predictions) - raw["frame_set"]:
            raise QualificationError(f"prediction contains non-validation frame: {model_key}")

        views: dict[str, Any] = {}
        for avo_threshold in AVO_THRESHOLDS:
            key = f"{avo_threshold:.2f}"
            at_020 = score_person_view(
                frame_ids=raw["frame_ids"],
                predictions=predictions,
                qualified_gt=qualified_gt,
                structural_ignored_gt=ignored_gt,
                episode_by_sample=episode_by_sample,
                avo_threshold=avo_threshold,
                detection_threshold=0.20,
            )
            at_002 = score_person_view(
                frame_ids=raw["frame_ids"],
                predictions=predictions,
                qualified_gt=qualified_gt,
                structural_ignored_gt=ignored_gt,
                episode_by_sample=episode_by_sample,
                avo_threshold=avo_threshold,
                detection_threshold=0.02,
            )
            at_020["overall"]["recall_at_detection_score_0_02"] = at_002["overall"]["recall"]
            for episode in EPISODES:
                at_020["episodes"][episode]["recall_at_detection_score_0_02"] = at_002[
                    "episodes"
                ][episode]["recall"]
            for band, values in at_020["distance_bins"].items():
                values["recall_at_detection_score_0_02"] = at_002["distance_bins"][band][
                    "recall"
                ]
            at_020["diagnostic_detection_score_0_02"] = at_002
            at_020["observable_no_support_gt"] = sum(
                bool(row["no_support"])
                and float(row["actor_volume_observability"]) >= avo_threshold
                for row in table
            )
            at_020["qualified_table_no_support_gt"] = sum(
                bool(row["no_support"]) for row in table
            )
            views[key] = at_020
        canonical_person = canonical_person_result(canonical)
        output[model_key] = {
            "display_name": spec["display_name"],
            "prediction_root": str(prediction_root.relative_to(REPO_ROOT)),
            "detection_predictions_total": total_predictions,
            "person_prediction_frames": len(predictions),
            "detections_sha256": detection_hash,
            "prediction_set_sha256": manifest["prediction_set_sha256"],
            "canonical_v010_person": canonical_person,
            "reused_canonical_vehicle": canonical["primary_v010"]["0.20"]["classes"]["vehicle"],
            "reused_canonical_segmentation": canonical["segmentation_v010"],
            "avo_thresholds": views,
        }
        for path in (detections, manifest_path, canonical_path):
            input_hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
        print(f"[score] completed all six AVO views for {spec['display_name']}", flush=True)
    return output, input_hashes


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build_summary_rows(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model in models.items():
        canonical = model["canonical_v010_person"]
        for threshold in AVO_THRESHOLDS:
            view = model["avo_thresholds"][f"{threshold:.2f}"]
            metric = view["overall"]
            rows.append(
                {
                    "model": model_key,
                    "model_display_name": model["display_name"],
                    "avo_threshold": threshold,
                    "human_supported_binary_operating_point": threshold == 0.65,
                    "observable_gt": metric["observable_gt"],
                    "avo_ignored_gt": metric["avo_ignored_gt"],
                    "structural_ignored_gt": metric["structural_ignored_gt"],
                    "tp": metric["tp"],
                    "fp": metric["fp"],
                    "fn": metric["fn"],
                    "ignored_predictions": metric["ignored_predictions"],
                    "avo_ignored_predictions": metric["avo_ignored_predictions"],
                    "structural_ignored_predictions": metric["structural_ignored_predictions"],
                    "precision": metric["precision"],
                    "recall": metric["recall"],
                    "f1": metric["f1"],
                    "xy_mae_m": metric["xy_mae_m"],
                    "recall_at_detection_score_0_02": metric[
                        "recall_at_detection_score_0_02"
                    ],
                    "observable_no_support_gt": view["observable_no_support_gt"],
                    "qualified_table_no_support_gt": view["qualified_table_no_support_gt"],
                    "canonical_v010_gt": canonical["eligible_gt"],
                    "canonical_v010_precision": canonical["precision"],
                    "canonical_v010_recall": canonical["recall"],
                    "canonical_v010_f1": canonical["f1"],
                    "canonical_v010_xy_mae_m": canonical["xy_mae_m"],
                    "canonical_v010_recall_at_detection_score_0_02": canonical[
                        "recall_at_detection_score_0_02"
                    ],
                }
            )
    return rows


def build_episode_rows(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model in models.items():
        for threshold in AVO_THRESHOLDS:
            view = model["avo_thresholds"][f"{threshold:.2f}"]
            for episode, metric in view["episodes"].items():
                rows.append(
                    {
                        "model": model_key,
                        "model_display_name": model["display_name"],
                        "avo_threshold": threshold,
                        "episode_id": episode,
                        **metric,
                    }
                )
    return rows


def build_distance_rows(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, model in models.items():
        for threshold in AVO_THRESHOLDS:
            view = model["avo_thresholds"][f"{threshold:.2f}"]
            for band, metric in view["distance_bins"].items():
                rows.append(
                    {
                        "model": model_key,
                        "model_display_name": model["display_name"],
                        "avo_threshold": threshold,
                        "distance_bin": band,
                        **metric,
                    }
                )
    return rows


def report_markdown(
    *,
    run_id: str,
    table_hash: str,
    table_diagnostics: Mapping[str, Any],
    pilot: Mapping[str, Any],
    models: Mapping[str, Any],
    human: Mapping[str, Any],
) -> str:
    lines = [
        "# Actor-volume observability frozen-model comparison",
        "",
        f"Run: `{run_id}`  ",
        f"Immutable AVO table SHA-256: `{table_hash}`",
        "",
        "## Scope and qualification",
        "",
        "This is the single registered CPU-only, read-only retrospective comparison. It used "
        "the original unnormalized actor-volume implementation from commit `dc5238d`; it did "
        "not train, run inference, load a checkpoint, import torch, use CUDA, invoke CARLA/Epic, "
        "open test data, tune a threshold, select a model, or change a service verdict.",
        "",
        f"The frozen evaluator universe contains {EXPECTED_VALIDATION_FRAMES:,} frames from the "
        f"two raw validation episodes. The immutable table contains "
        f"{table_diagnostics['qualified_person_actor_frames']:,} person actor-frames satisfying "
        "distance ≤40 m, projected-box center inside the 1280×720 image, projected area ≥12 px, "
        "positive camera-forward geometry, and exact synchronized depth. "
        f"{table_diagnostics['structurally_ignored_raw_person_actor_frames']:,} other raw person "
        "actor-frames are structural ignores. Truncation remains a separate diagnostic.",
        "",
        f"Before any prediction was opened, all {pilot['expected_comparisons']} registered pilot "
        "comparisons reproduced bit-exactly: retained point counts; unclipped, clipped, and "
        "visible-box areas; unnormalized AVO; truncation; and no-support status. The table has "
        f"{table_diagnostics['no_support_count']} no-support records.",
        "",
        "`actor_volume_observability = area(B_visible) / area(B_full_clipped)`. It uses unchanged "
        "depth back-projection, oriented actor-volume containment, 0.05 m containment tolerance, "
        "bottom +0.03 m ground rejection, deterministic overlap assignment, and a no-support "
        "score of 0. It is not an exact visible-silhouette percentage.",
        "",
        "## Complete model × AVO-threshold comparison",
        "",
        "Detection score is fixed at 0.20. `R@.02` is diagnostic. `Ign pred` includes matches to "
        "AVO-below-cutoff and structural-ignore person GT. The six AVO thresholds are "
        "supplementary sensitivity views, not literal percentages of the pedestrian silhouette.",
        "",
        "| Model | AVO≥ | Obs GT | AVO-ignored GT | Obs no-support GT | TP | FP | FN | Ign pred | Precision | Recall | F1 | XY MAE m | R@.02 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        for threshold in AVO_THRESHOLDS:
            view = model["avo_thresholds"][f"{threshold:.2f}"]
            metric = view["overall"]
            mark = " **(human-supported)**" if threshold == 0.65 else ""
            lines.append(
                f"| {model['display_name']} | {threshold:.2f}{mark} | {metric['observable_gt']} | "
                f"{metric['avo_ignored_gt']} | {view['observable_no_support_gt']} | "
                f"{metric['tp']} | {metric['fp']} | {metric['fn']} | "
                f"{metric['ignored_predictions']} | {fmt(metric['precision'])} | "
                f"{fmt(metric['recall'])} | {fmt(metric['f1'])} | {fmt(metric['xy_mae_m'], 3)} | "
                f"{fmt(metric['recall_at_detection_score_0_02'])} |"
            )

    lines += [
        "",
        "## Highlighted AVO≥0.65 view and frozen references",
        "",
        "AVO≥0.65 is highlighted because it was independently compared with the 100-person "
        "human pilot and achieved 0.8523 balanced accuracy. It is the only human-supported "
        "binary AVO operating point here. The existing human-band recall is a separate "
        "target-stratified reference and is not part of the AVO calculation.",
        "",
        "| Model | View | GT/N | TP | FP | FN | Precision | Recall | F1 | XY MAE m | R@.02 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, model in models.items():
        canonical = model["canonical_v010_person"]
        avo = model["avo_thresholds"]["0.65"]["overall"]
        human_row = human["models"][MODEL_SPECS[model_key]["human_key"]][
            "primary_ge65_nonsevere"
        ]
        lines.append(
            f"| {model['display_name']} | canonical v0.10 | {canonical['eligible_gt']} | "
            f"{canonical['tp']} | {canonical['fp']} | {canonical['fn']} | "
            f"{fmt(canonical['precision'])} | {fmt(canonical['recall'])} | "
            f"{fmt(canonical['f1'])} | {fmt(canonical['xy_mae_m'], 3)} | "
            f"{fmt(canonical['recall_at_detection_score_0_02'])} |"
        )
        lines.append(
            f"| {model['display_name']} | **AVO≥0.65** | {avo['observable_gt']} | {avo['tp']} | "
            f"{avo['fp']} | {avo['fn']} | {fmt(avo['precision'])} | {fmt(avo['recall'])} | "
            f"{fmt(avo['f1'])} | {fmt(avo['xy_mae_m'], 3)} | "
            f"{fmt(avo['recall_at_detection_score_0_02'])} |"
        )
        lines.append(
            f"| {model['display_name']} | human bands ≥65, non-severe | {human_row['n']} | "
            f"{human_row['tp']} | NA | {human_row['fn']} | NA | {fmt(human_row['recall'])} | "
            f"NA | {fmt(human_row['xy_mae_m'], 3)} | NA |"
        )

    lines += [
        "",
        "## Per-model precision-recall trends",
        "",
        "These are denominator changes under successively stricter supplementary eligibility "
        "views, not model improvements and not a threshold-selection exercise.",
        "",
    ]
    for model in models.values():
        trend = []
        for threshold in AVO_THRESHOLDS:
            metric = model["avo_thresholds"][f"{threshold:.2f}"]["overall"]
            trend.append(
                f"AVO≥{threshold:.2f}: n={metric['observable_gt']}, P={metric['precision']:.4f}, "
                f"R={metric['recall']:.4f}"
            )
        lines.append(f"- {model['display_name']}: " + "; ".join(trend) + ".")

    lines += [
        "",
        "## Direct comparison with each frozen canonical v0.10 result",
        "",
        "Canonical values below are reused artifacts, not rescored values. Differences reflect "
        "eligibility denominators and ignore assignment; they are not model changes.",
        "",
        "| Model | AVO≥ | Canonical GT | Canonical P/R/F1 | AVO GT | AVO P/R/F1 | Canonical/AVO XY MAE m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        canonical = model["canonical_v010_person"]
        for threshold in AVO_THRESHOLDS:
            metric = model["avo_thresholds"][f"{threshold:.2f}"]["overall"]
            lines.append(
                f"| {model['display_name']} | {threshold:.2f} | {canonical['eligible_gt']} | "
                f"{fmt(canonical['precision'])}/{fmt(canonical['recall'])}/{fmt(canonical['f1'])} | "
                f"{metric['observable_gt']} | "
                f"{fmt(metric['precision'])}/{fmt(metric['recall'])}/{fmt(metric['f1'])} | "
                f"{fmt(canonical['xy_mae_m'], 3)}/{fmt(metric['xy_mae_m'], 3)} |"
            )

    lines += [
        "",
        "## Reused canonical vehicle and segmentation evidence",
        "",
        "These frozen canonical v0.10 values were copied from each existing evaluation artifact; "
        "vehicle and segmentation were not rescored.",
        "",
        "| Model | Vehicle GT | Vehicle TP/FP/FN | Vehicle P/R/F1 | Vehicle XY MAE m | Vehicle IoU | Person box-mask IoU | Foreground mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        vehicle = model["reused_canonical_vehicle"]
        segmentation = model["reused_canonical_segmentation"]
        lines.append(
            f"| {model['display_name']} | {vehicle['eligible_gt']} | "
            f"{vehicle['tp']}/{vehicle['fp']}/{vehicle['fn']} | "
            f"{fmt(vehicle['precision'])}/{fmt(vehicle['recall'])}/{fmt(vehicle['f1'])} | "
            f"{fmt(vehicle['xy_mae_m'], 3)} | {fmt(segmentation['vehicle_iou'])} | "
            f"{fmt(segmentation['person_box_mask_iou'])} | "
            f"{fmt(segmentation['foreground_miou'])} |"
        )

    lines += [
        "",
        "## Results by validation episode",
        "",
        "| Model | AVO≥ | Episode | Obs GT | AVO-ignored GT | TP | FP | FN | Ign pred | Precision | Recall | F1 | XY MAE m | R@.02 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        for threshold in AVO_THRESHOLDS:
            for episode in EPISODES:
                metric = model["avo_thresholds"][f"{threshold:.2f}"]["episodes"][episode]
                lines.append(
                    f"| {model['display_name']} | {threshold:.2f} | {episode} | "
                    f"{metric['observable_gt']} | {metric['avo_ignored_gt']} | "
                    f"{metric['tp']} | {metric['fp']} | {metric['fn']} | "
                    f"{metric['ignored_predictions']} | {fmt(metric['precision'])} | "
                    f"{fmt(metric['recall'])} | {fmt(metric['f1'])} | "
                    f"{fmt(metric['xy_mae_m'], 3)} | "
                    f"{fmt(metric['recall_at_detection_score_0_02'])} |"
                )

    lines += [
        "",
        "## Distance-bin recall and localization",
        "",
        "| Model | AVO≥ | Distance | Obs GT | TP/FN | Recall | XY MAE m | R@.02 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for model in models.values():
        for threshold in AVO_THRESHOLDS:
            for band, _low, _high in DISTANCE_BINS:
                metric = model["avo_thresholds"][f"{threshold:.2f}"]["distance_bins"][band]
                lines.append(
                    f"| {model['display_name']} | {threshold:.2f} | {band} | "
                    f"{metric['eligible_gt']} | {metric['tp']}/{metric['fn']} | "
                    f"{fmt(metric['recall'])} | {fmt(metric['xy_mae_m'], 3)} | "
                    f"{fmt(metric['recall_at_detection_score_0_02'])} |"
                )

    lines += [
        "",
        "## Interpretation",
        "",
        "AVO≥0.65 achieved 0.8523 balanced accuracy against the human pilot. This is a binary "
        "observability sensitivity analysis. The actor-volume score failed fine-grained "
        f"four-band agreement (exact agreement {pilot['pilot_exact_four_band_agreement']:.4f}; "
        f"linear-weighted kappa {pilot['pilot_linear_weighted_four_band_kappa']:.4f}, below 0.60), "
        "so human bands remain the fine-grained visibility reference. "
        "AVO is a bounding-box extent statistic derived from actor-volume-supported depth "
        "points; it is not an exact visible-silhouette percentage.",
        "",
        "The comparison is retrospective and supplementary. It does not alter checkpoint "
        "selection, the supervisor-approved SplitFusion-FCOS service decision, canonical v0.10 "
        "results, vehicle results, segmentation results, service gates, or model selection. "
        "Historical depth-consistent occupancy is retained only as an internal sensitivity. "
        "Increasing precision or recall under a stricter AVO eligibility view must not be "
        "interpreted as model improvement.",
        "",
        "The six AVO thresholds are supplementary sensitivity views. AVO≥0.65 is highlighted "
        "because it was independently compared with the human pilot; none of the thresholds "
        "changes the canonical or supervisor-approved service result.",
        "",
        TERMINAL,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="unused create-only output directory name")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise QualificationError('refusing to run without CUDA_VISIBLE_DEVICES=""')
    if "torch" in sys.modules:
        raise QualificationError("torch is imported; checkpoint/model execution is prohibited")
    if tuple(AVO_THRESHOLDS) != (0.10, 0.25, 0.50, 0.65, 0.70, 0.85):
        raise QualificationError("registered AVO threshold set/order drift")

    started = time.perf_counter()
    run_dir = OUTPUT_PARENT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        source_qualification = verify_original_unnormalized_sources()
        print("[qualification] loading raw validation sources", flush=True)
        raw = load_raw_sources()
        print("[qualification] building the one AVO table", flush=True)
        table, table_diagnostics = score_avo_table(raw)
        print("[qualification] exact 100-person pilot reproduction", flush=True)
        pilot = verify_pilot_reproduction(table)

        table_path = run_dir / "actor_volume_observability_table.csv"
        write_csv_x(table_path, TABLE_FIELDS, table)
        table_hash = sha256_file(table_path)
        table_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        reloaded = read_csv(table_path)
        if len(reloaded) != len(table) or sha256_file(table_path) != table_hash:
            raise QualificationError("immutable AVO table read-back failure")

        # This is deliberately after the exact pilot gate and immutable table write.
        print("[evaluation] opening frozen predictions after qualification passed", flush=True)
        models, prediction_hashes = score_models(raw, table)
        human = read_json(HUMAN_AGGREGATE)
        human_hash = sha256_file(HUMAN_AGGREGATE)

        summary_rows = build_summary_rows(models)
        episode_rows = build_episode_rows(models)
        distance_rows = build_distance_rows(models)
        summary_path = run_dir / "model_threshold_summary.csv"
        episode_path = run_dir / "model_threshold_episode_metrics.csv"
        distance_path = run_dir / "model_threshold_distance_metrics.csv"
        write_csv_x(summary_path, tuple(summary_rows[0]), summary_rows)
        write_csv_x(episode_path, tuple(episode_rows[0]), episode_rows)
        write_csv_x(distance_path, tuple(distance_rows[0]), distance_rows)

        results = {
            "schema": "actor_volume_observability_model_comparison_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "terminal": TERMINAL,
            "run_id": args.run_id,
            "cpu_only": True,
            "torch_imported": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inference_runs": 0,
            "checkpoint_files_opened": 0,
            "carla_runs": 0,
            "test_rows_opened": 0,
            "avo_thresholds": list(AVO_THRESHOLDS),
            "human_supported_binary_avo_threshold": HUMAN_SUPPORTED_AVO_THRESHOLD,
            "detection_score_threshold": 0.20,
            "diagnostic_detection_score_threshold": 0.02,
            "match_radius_m": MATCH_RADIUS_M,
            "matching": (
                "canonical eligible nearest-first same-class world-XY match within 3m; "
                "then unmatched predictions matched nearest-first to AVO or structural ignored "
                "person GT; all remaining predictions are FP"
            ),
            "visibility": {
                "name": "actor_volume_observability",
                "formula": "area(B_visible) / area(B_full_clipped)",
                "algorithm_version": core.ALGORITHM_VERSION,
                "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
                "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
                "no_support_score": 0.0,
                "truncation_separate": True,
                "not_exact_visible_silhouette_percentage": True,
                "source_qualification": source_qualification,
            },
            "pilot_reproduction": pilot,
            "avo_table": {
                **table_diagnostics,
                "path": str(table_path.relative_to(REPO_ROOT)),
                "sha256": table_hash,
                "read_only_mode": oct(table_path.stat().st_mode & 0o777),
            },
            "models": models,
            "human_band_aggregate_evidence": human,
            "human_band_aggregate_sha256": human_hash,
            "vehicle_and_segmentation": "reused from canonical results; not rescored",
            "interpretation_guards": {
                "binary_observability_sensitivity_analysis": True,
                "fine_grained_four_band_agreement_failed": True,
                "retrospective_and_supplementary": True,
                "human_bands_remain_fine_grained_reference": True,
                "historical_depth_consistent_occupancy_internal_sensitivity_only": True,
                "checkpoint_selection_unchanged": True,
                "supervisor_approved_fcos_service_decision_unchanged": True,
                "canonical_results_unchanged": True,
                "no_best_threshold_selected": True,
                "stricter_threshold_metrics_are_not_model_improvement": True,
            },
            "wall_seconds": time.perf_counter() - started,
        }
        result_path = run_dir / "model_threshold_results.json"
        write_json_x(result_path, results)
        report = report_markdown(
            run_id=args.run_id,
            table_hash=table_hash,
            table_diagnostics=table_diagnostics,
            pilot=pilot,
            models=models,
            human=human,
        )
        report_path = run_dir / "FINAL_REPORT.md"
        write_text_x(report_path, report)

        hashes = {
            "inputs": {
                **raw["input_hashes"],
                **prediction_hashes,
                str(HUMAN_AGGREGATE.relative_to(REPO_ROOT)): human_hash,
            },
            "outputs": {
                str(path.relative_to(run_dir)): sha256_file(path)
                for path in (
                    table_path,
                    summary_path,
                    episode_path,
                    distance_path,
                    result_path,
                    report_path,
                )
            },
        }
        write_json_x(run_dir / "ARTIFACT_HASHES.json", hashes)
        write_json_x(
            run_dir / "RUN_METADATA.json",
            {
                "schema": "actor_volume_observability_model_comparison_run_v1",
                "created_utc": results["created_utc"],
                "run_id": args.run_id,
                "terminal": TERMINAL,
                "wall_seconds": results["wall_seconds"],
                "pilot_reproduction_passed": True,
                "avo_table_sha256": table_hash,
                "model_count": len(models),
                "avo_threshold_count": len(AVO_THRESHOLDS),
                "orchestrated_model_threshold_views": len(models) * len(AVO_THRESHOLDS),
            },
        )
        write_text_x(run_dir / TERMINAL, TERMINAL + "\n")
        print(
            json.dumps(
                {
                    "terminal": TERMINAL,
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "avo_table_rows": len(table),
                    "avo_table_sha256": table_hash,
                    "pilot_exact": pilot["passed"],
                    "models": list(models),
                    "avo_thresholds": list(AVO_THRESHOLDS),
                    "wall_seconds": results["wall_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        failure = {
            "terminal": "ACTOR_VOLUME_OBSERVABILITY_MODEL_COMPARISON_FAILED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "models_scored_before_failure": 0,
        }
        failure_path = run_dir / "FAILURE.json"
        if not failure_path.exists():
            write_json_x(failure_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
