"""Part 1: build the immutable training-only expected-clear-support reference.

Runs the *unchanged* actor-volume extraction over every qualifying training
person GT and writes a hashed reference artifact.  No human annotation, no
validation row, no test row, no model artifact and no CUDA are touched; this
script deliberately never opens the human pilot directory at all, so the
reference provably cannot have been shaped by the labels it is later scored
against.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 -m \
        data_collection.route_b_publication_actor_volume_visibility_v1.build_training_reference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import core, scoring, training_reference as tref
from .run_audit import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DATASET_DIR,
    EPISODE_ROOT,
    OUTPUT_PARENT,
    REPO_ROOT,
    AuditError,
    assert_registered_decoder,
    decode_depth_bgra,
    sha256_file,
)

REFERENCE_DIR_NAME = "training_reference"
TERMINAL = "TRAIN_NORMALIZED_ACTOR_VOLUME_REFERENCE_BUILT"


def training_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every training person GT that passes the locked geometric filter."""
    manifest = pd.read_csv(DATASET_DIR / "manifest.csv")
    split_counts = manifest.split.value_counts().to_dict()
    if "test" in split_counts:
        raise AuditError("dataset manifest unexpectedly contains a test split")
    train = manifest[manifest.split == "train"]
    val_sample_ids = set(manifest[manifest.split != "train"].sample_id)
    episodes = sorted(train.experiment_id.unique())
    train_sample_ids = set(train.sample_id)

    frames: list[pd.DataFrame] = []
    for episode in episodes:
        boxes = pd.read_csv(
            EPISODE_ROOT / episode / "object_boxes.csv", dtype={"gt_actor_id": str}
        )
        visibility = pd.read_csv(
            EPISODE_ROOT / episode / "object_visibility.csv", dtype={"gt_actor_id": str}
        )
        people = boxes[boxes.label == "person"]
        merged = people.merge(
            visibility[
                [
                    "sample_id",
                    "gt_actor_id",
                    "unclipped_projected_area_px",
                    "clipped_projected_area_px",
                ]
            ],
            on=["sample_id", "gt_actor_id"],
            how="left",
            validate="one_to_one",
        )
        merged["episode_id"] = episode
        frames.append(merged)
    population = pd.concat(frames, ignore_index=True)

    # Split membership is enforced by set intersection with the manifest, not by
    # trusting the episode name.
    population = population[population.sample_id.isin(train_sample_ids)].copy()
    # In-frame fraction is recomputed from the recorded projected areas rather
    # than read from a stored visibility field, so no historical visibility or
    # eligibility flag enters the filter.
    population["in_frame_fraction"] = (
        population.clipped_projected_area_px / population.unclipped_projected_area_px
    )
    before = len(population)
    qualified = population[
        (population.gt_distance_m <= tref.MAX_DISTANCE_M)
        & (population.in_frame_fraction >= tref.MIN_IN_FRAME_FRACTION)
        & np.isfinite(population.gt_distance_m)
        & np.isfinite(population.object_world_x)
        & np.isfinite(population.object_world_y)
        & np.isfinite(population.object_world_z)
        & np.isfinite(population.object_yaw_deg)
        & (population.gt_extent_x_m > 0.0)
        & (population.gt_extent_y_m > 0.0)
        & (population.gt_extent_z_m > 0.0)
    ].copy()

    provenance = {
        "dataset_manifest_split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "train_episodes": episodes,
        "train_frames_in_manifest": int(len(train)),
        "person_rows_in_train_episodes": int(before),
        "person_rows_qualified": int(len(qualified)),
        "filter": {
            "max_distance_m": tref.MAX_DISTANCE_M,
            "min_in_frame_fraction": tref.MIN_IN_FRAME_FRACTION,
            "requires_finite_geometry": True,
            "uses_any_visibility_or_eligibility_flag": False,
        },
        "validation_sample_ids_in_population": int(
            len(set(qualified.sample_id) & val_sample_ids)
        ),
        "test_rows_read": 0,
    }
    return qualified.sort_values(["episode_id", "sample_id", "gt_actor_id"]), provenance


def extract_records(qualified: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the unchanged actor-volume extraction frame by frame."""
    import cv2

    records: list[dict[str, Any]] = []
    stats = {
        "frames_opened": 0,
        "actors_attempted": 0,
        "actors_extracted": 0,
        "actors_skipped_geometry": 0,
        "actors_zero_support": 0,
        "actors_invalid_support": 0,
        "skip_reasons": {},
    }

    for (episode, sample_id), group in qualified.groupby(
        ["episode_id", "sample_id"], sort=True
    ):
        manifest_row = _manifest_row(episode, sample_id)
        depth_path = EPISODE_ROOT / episode / "depth" / f"{sample_id}.png"
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise AuditError(f"unreadable training depth image {depth_path}")
        depth_m = decode_depth_bgra(raw).astype(np.float64)
        if not np.all(np.isfinite(depth_m)):
            raise AuditError(f"non-finite training depth in {depth_path}")
        stats["frames_opened"] += 1

        intrinsics = np.asarray(
            [
                [float(manifest_row.camera_fx), 0.0, float(manifest_row.camera_cx)],
                [0.0, float(manifest_row.camera_fy), float(manifest_row.camera_cy)],
                [0.0, 0.0, 1.0],
            ]
        )
        camera_matrix = np.asarray(
            json.loads(manifest_row.camera_matrix_json), dtype=np.float64
        )
        camera_inverse = np.asarray(
            json.loads(manifest_row.camera_inverse_matrix_json), dtype=np.float64
        )
        camera_position = camera_matrix[:3, 3]

        # All persons in the frame are candidates for the deterministic
        # overlapping-actor assignment, exactly as in the pilot.
        pedestrians = [
            {
                "key": str(row.gt_actor_id),
                "centre": (row.object_world_x, row.object_world_y, row.object_world_z),
                "extent": (row.gt_extent_x_m, row.gt_extent_y_m, row.gt_extent_z_m),
                "yaw_deg": float(row.object_yaw_deg),
            }
            for row in _frame_people(episode, sample_id).itertuples()
        ]

        for row in group.itertuples():
            stats["actors_attempted"] += 1
            try:
                result = scoring.score_actor_frame(
                    depth_m=depth_m,
                    camera_matrix=camera_matrix,
                    camera_inverse=camera_inverse,
                    intrinsics=intrinsics,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    target_key=str(row.gt_actor_id),
                    target_centre=(row.object_world_x, row.object_world_y, row.object_world_z),
                    target_extent=(row.gt_extent_x_m, row.gt_extent_y_m, row.gt_extent_z_m),
                    target_yaw_deg=float(row.object_yaw_deg),
                    pedestrian_boxes=pedestrians,
                )
                angle = tref.folded_view_angle_deg(
                    (row.object_world_x, row.object_world_y, row.object_world_z),
                    float(row.object_yaw_deg),
                    camera_position,
                )
                density = tref.support_density(
                    int(result["retained_actor_point_count"]),
                    float(result["clipped_projected_area_px"]),
                )
            except ValueError as exc:
                stats["actors_skipped_geometry"] += 1
                reason = str(exc).split("(")[0][:80]
                stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1
                continue
            if not np.isfinite(density) or density < 0.0:
                stats["actors_invalid_support"] += 1
                continue
            if density == 0.0:
                stats["actors_zero_support"] += 1
            stats["actors_extracted"] += 1
            records.append(
                {
                    "episode_id": episode,
                    "sample_id": sample_id,
                    "gt_actor_id": str(row.gt_actor_id),
                    "actor_type": str(row.gt_actor_type_id),
                    "distance_m": float(row.gt_distance_m),
                    "folded_view_angle_deg": angle,
                    "angle_bin": tref.angle_bin(angle),
                    "clipped_bbox_h": float(result["clipped_bbox_h"]),
                    "height_bin": tref.height_bin(float(result["clipped_bbox_h"])),
                    "clipped_projected_area_px": float(result["clipped_projected_area_px"]),
                    "retained_actor_point_count": int(result["retained_actor_point_count"]),
                    "support_density": density,
                }
            )
    return records, stats


_MANIFEST_CACHE: dict[str, pd.DataFrame] = {}
_PEOPLE_CACHE: dict[str, dict[str, pd.DataFrame]] = {}


def _non_train_sample_ids() -> set[str]:
    """Every sample id the dataset manifest does not mark as training."""
    manifest = pd.read_csv(DATASET_DIR / "manifest.csv", usecols=["sample_id", "split"])
    return set(manifest[manifest.split != "train"].sample_id)


def _manifest_row(episode: str, sample_id: str):
    if episode not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[episode] = pd.read_csv(
            EPISODE_ROOT / episode / "manifest.csv"
        ).set_index("sample_id")
    return _MANIFEST_CACHE[episode].loc[sample_id]


def _frame_people(episode: str, sample_id: str) -> pd.DataFrame:
    if episode not in _PEOPLE_CACHE:
        boxes = pd.read_csv(
            EPISODE_ROOT / episode / "object_boxes.csv", dtype={"gt_actor_id": str}
        )
        people = boxes[boxes.label == "person"]
        _PEOPLE_CACHE[episode] = {sid: g for sid, g in people.groupby("sample_id")}
    return _PEOPLE_CACHE[episode][sample_id]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise AuditError('refusing to run without CUDA_VISIBLE_DEVICES="" (CPU-only)')
    if "torch" in sys.modules:
        raise AuditError("torch is imported; this build must stay prediction-blind")

    started = time.perf_counter()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_PARENT / REFERENCE_DIR_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # create-only

    decoder = assert_registered_decoder()
    qualified, provenance = training_population()
    if provenance["validation_sample_ids_in_population"] != 0:
        raise AuditError("validation rows leaked into the training population")

    records, stats = extract_records(qualified)
    reference = tref.build_reference(records)

    records_frame = pd.DataFrame(records)
    records_path = run_dir / "training_support_records.csv"
    records_frame.to_csv(records_path, index=False)

    non_positive = {
        tier: sorted(
            key
            for key, group in table.items()
            if not np.isfinite(group["expected_clear_support_density"])
            or group["expected_clear_support_density"] <= 0.0
        )
        for tier, table in reference["tables"].items()
    }

    artifact = {
        "schema": "route_b_train_normalized_actor_volume_reference_v1",
        "terminal": TERMINAL,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "human_annotation_files_read": 0,
        "human_pilot_directory_opened": False,
        "model_or_prediction_artifacts_read": 0,
        "test_rows_read": 0,
        "carla_started": False,
        "decoder": decoder,
        "extraction_constants": {
            "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
            "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
            "algorithm_version": core.ALGORITHM_VERSION,
            "unchanged_from_pilot": True,
        },
        "training_provenance": provenance,
        "training_source_hashes": {
            f"{episode}/{name}": sha256_file(EPISODE_ROOT / episode / name)
            for episode in provenance["train_episodes"]
            for name in ("manifest.csv", "object_boxes.csv", "object_visibility.csv")
        },
        "dataset_manifest_sha256": sha256_file(DATASET_DIR / "manifest.csv"),
        "extraction_stats": stats,
        "leakage_proof": {
            "episodes_used": provenance["train_episodes"],
            "all_episodes_are_train_split": True,
            "validation_sample_ids_in_population": provenance[
                "validation_sample_ids_in_population"
            ],
            "validation_sample_ids_in_records": int(
                len(set(records_frame.sample_id) & _non_train_sample_ids())
            ),
            "unique_sample_ids_used": int(records_frame.sample_id.nunique()),
            "unique_actors_used": int(
                records_frame.groupby(["sample_id", "gt_actor_id"]).ngroups
            ),
            "human_annotation_files_read": 0,
            "test_rows_read": 0,
        },
        "reference": reference,
        "non_positive_reference_groups": non_positive,
        "group_counts": {
            tier: len(table) for tier, table in reference["tables"].items()
        },
        "wall_seconds": time.perf_counter() - started,
    }
    reference_path = run_dir / "training_reference.json"
    reference_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    reference_hash = sha256_file(reference_path)
    (run_dir / "REFERENCE_HASHES.json").write_text(
        json.dumps(
            {
                "training_reference.json": reference_hash,
                "training_support_records.csv": sha256_file(records_path),
            },
            indent=2,
        )
    )
    (run_dir / TERMINAL).write_text(
        f"{TERMINAL}\n{reference_hash}\n{datetime.now(timezone.utc).isoformat()}\n"
    )
    # Make the artifact immutable on disk.
    for path in (reference_path, records_path):
        path.chmod(0o444)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "training_reference_sha256": reference_hash,
                "records": len(records),
                "group_counts": artifact["group_counts"],
                "extraction_stats": stats,
                "wall_seconds": round(artifact["wall_seconds"], 1),
            },
            indent=2,
        )
    )
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
