"""Bounded, prediction-blind, CPU-only feasibility audit of the actor-volume
pedestrian visibility metric on the 100 human-annotated validation panels.

The audit trains nothing, reads no model prediction or checkpoint, opens no test
rows, never imports torch, and never starts CARLA.  It reads only the frozen
validation dataset geometry, the recorded depth PNGs, and annotator A's bands.

Usage:
    CUDA_VISIBLE_DEVICES="" python3 -m \
        data_collection.route_b_publication_actor_volume_visibility_v1.run_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import agreement as agreement_mod
from . import core, scoring

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    REPO_ROOT
    / "experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151"
)
DATASET_DIR = DATASET_ROOT / "dataset"
HUMAN_PILOT_DIR = (
    REPO_ROOT
    / "data_collection/experiments/route_b_publication_human_occlusion_pilot_v1"
    / "20260901_030234_seed20260831"
)
# The pilot registered its geometry provenance by sha256 against the frozen
# validation view; `sample_manifest.csv` carries those hashes.  The view files
# are therefore the authoritative box/visibility source for these 100 rows.
VIEW_DIR = (
    REPO_ROOT
    / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/views/val"
)
EPISODE_ROOT = REPO_ROOT / "data_collection/experiments/route_b_perception_v3"
OUTPUT_PARENT = (
    REPO_ROOT
    / "data_collection/experiments/route_b_publication_actor_volume_visibility_v1"
)

CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720

# --- pre-registered qualification thresholds --------------------------------
MAX_BOX_RECONSTRUCTION_ERROR_PX = 1e-3
# The recorded camera matrices are stored as float32, so `camera_matrix @
# camera_inverse` departs from the identity at the 1e-6 level.  A projection /
# back-projection round trip through the two recorded matrices inherits that
# floor; 0.01 px is two orders of magnitude below one pixel and two orders above
# the storage floor.
MAX_ROUND_TRIP_ERROR_PX = 1e-2
MAX_CALIBRATION_IDENTITY_ERROR = 1e-4
MAX_TIMESTAMP_DELTA_S = 0.0
EXPECTED_SAMPLE_COUNT = 100

# --- pre-registered decision thresholds -------------------------------------
MIN_WEIGHTED_KAPPA = 0.60
MIN_BALANCED_ACCURACY = 0.80

DISTANCE_BANDS = (
    ("00_10m", 0.0, 10.0),
    ("10_20m", 10.0, 20.0),
    ("20_30m", 20.0, 30.0),
    ("30_40m", 30.0, 40.0),
)

TERMINAL_FEASIBLE = "ACTOR_VOLUME_VISIBILITY_PILOT_FEASIBLE_AWAITING_FULL_RESCORE"
TERMINAL_NOT_FEASIBLE = "ACTOR_VOLUME_VISIBILITY_PILOT_NOT_FEASIBLE_RETAIN_HUMAN_BANDS"
TERMINAL_INVALID = "ACTOR_VOLUME_VISIBILITY_PILOT_IMPLEMENTATION_INVALID"


class AuditError(RuntimeError):
    """Raised when a qualification check fails; never caught to weaken a gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_depth_bgra(raw_bgra: np.ndarray) -> np.ndarray:
    """The registered CARLA BGRA depth decoder.

    Byte-identical in behaviour to
    ``data_collection/route_b_perception_v3/visibility_v1.decode_depth_bgra``;
    equality against that frozen implementation is asserted at run time by
    :func:`assert_registered_decoder`.
    """
    raw = np.asarray(raw_bgra)
    if raw.ndim != 3 or raw.shape[2] != 4 or raw.dtype != np.uint8:
        raise AuditError(f"expected HxWx4 uint8 BGRA depth, got {raw.shape}/{raw.dtype}")
    values = raw.astype(np.float32, copy=False)
    blue, green, red = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    normalized = (red + green * 256.0 + blue * 256.0 * 256.0) / (256.0**3 - 1.0)
    return (core.CARLA_MAX_DEPTH_M * normalized).astype(np.float32, copy=False)


def assert_registered_decoder() -> dict[str, Any]:
    """Prove this audit decodes depth exactly like the frozen collector."""
    import importlib.util

    path = REPO_ROOT / "data_collection/route_b_perception_v3/visibility_v1.py"
    spec = importlib.util.spec_from_file_location("_registered_visibility_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(20260901)
    probe = rng.integers(0, 256, size=(37, 41, 4), dtype=np.uint8)
    mine = decode_depth_bgra(probe)
    theirs = module.decode_depth_bgra(probe)
    if not np.array_equal(mine, theirs):
        raise AuditError("depth decoder diverges from the registered collector decoder")
    return {
        "registered_module": str(path.relative_to(REPO_ROOT)),
        "registered_module_sha256": sha256_file(path),
        "registered_algorithm_version": module.ALGORITHM_VERSION,
        "decoder_bit_identical": True,
    }


def load_inputs() -> dict[str, Any]:
    sample_manifest = pd.read_csv(
        HUMAN_PILOT_DIR / "sample_manifest.csv", dtype={"gt_actor_id": str}
    )
    human = pd.read_csv(
        HUMAN_PILOT_DIR / "annotator_A_visibility_bands.csv", dtype=str
    )
    dataset_manifest = pd.read_csv(DATASET_DIR / "manifest.csv")
    view_manifest = pd.read_csv(VIEW_DIR / "manifest.csv")
    object_boxes = pd.read_csv(VIEW_DIR / "object_boxes_all.csv", dtype={"gt_actor_id": str})
    object_visibility = pd.read_csv(
        VIEW_DIR / "object_visibility_all.csv", dtype={"gt_actor_id": str}
    )
    episode_visibility = pd.concat(
        [
            pd.read_csv(
                EPISODE_ROOT / episode / "object_visibility.csv",
                dtype={"gt_actor_id": str},
            )
            for episode in sorted(sample_manifest["episode_id"].unique())
        ],
        ignore_index=True,
    )
    dataset_object_boxes = pd.read_csv(
        DATASET_DIR / "object_boxes.csv", dtype={"gt_actor_id": str}
    )
    return {
        "sample_manifest": sample_manifest,
        "human": human,
        "dataset_manifest": dataset_manifest,
        "view_manifest": view_manifest,
        "object_boxes": object_boxes,
        "object_visibility": object_visibility,
        "episode_visibility": episode_visibility,
        "dataset_object_boxes": dataset_object_boxes,
    }


def qualify_provenance(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks that can be made before any depth image is opened."""
    checks: list[dict[str, Any]] = []
    sm = data["sample_manifest"]
    human = data["human"]
    key = ["sample_id", "gt_actor_id"]
    target_keys = set(zip(sm.sample_id, sm.gt_actor_id))

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    record(
        "sample_count_is_100",
        len(sm) == EXPECTED_SAMPLE_COUNT
        and len(target_keys) == EXPECTED_SAMPLE_COUNT
        and sm.sample_id.nunique() == EXPECTED_SAMPLE_COUNT,
        {"rows": int(len(sm)), "unique_actor_frames": len(target_keys)},
    )
    record(
        "annotator_A_covers_every_sample_exactly_once",
        len(human) == EXPECTED_SAMPLE_COUNT
        and set(human.sample_id) == set(sm.sample_id)
        and human.sample_id.is_unique,
        {"rows": int(len(human))},
    )

    # Registered provenance hashes.
    hashes = {
        "manifest": (sha256_file(VIEW_DIR / "manifest.csv"), sm.source_manifest_sha256.unique().tolist()),
        "object_boxes": (
            sha256_file(VIEW_DIR / "object_boxes_all.csv"),
            sm.source_object_boxes_sha256.unique().tolist(),
        ),
        "depth_diagnostics": (
            sha256_file(VIEW_DIR / "object_visibility_all.csv"),
            sm.source_depth_diagnostics_sha256.unique().tolist(),
        ),
    }
    record(
        "geometry_sources_match_registered_pilot_hashes",
        all(len(expected) == 1 and actual == expected[0] for actual, expected in hashes.values()),
        {name: {"actual": a, "registered": e} for name, (a, e) in hashes.items()},
    )

    # Exact joins.  A left join on the pilot keys plus an explicit match count is
    # used so that a target missing from a source frame surfaces as zero matches
    # rather than silently appearing as one all-NaN row.
    for name, frame in (
        ("object_boxes", data["object_boxes"]),
        ("object_visibility", data["object_visibility"]),
        ("episode_object_visibility", data["episode_visibility"]),
    ):
        sizes = frame.groupby(key, dropna=False).size().reset_index(name="matches")
        counted = sm[key].merge(sizes, on=key, how="left")
        counted["matches"] = counted["matches"].fillna(0).astype(int)
        record(
            f"exact_1to1_join_{name}",
            len(counted) == EXPECTED_SAMPLE_COUNT and bool((counted.matches == 1).all()),
            {
                "rows": int(len(counted)),
                "missing_targets": int((counted.matches == 0).sum()),
                "duplicated_targets": int((counted.matches > 1).sum()),
            },
        )
    dm = data["dataset_manifest"]
    dm_targets = dm[dm.sample_id.isin(sm.sample_id)]
    record(
        "exact_1to1_join_dataset_manifest",
        len(dm_targets) == EXPECTED_SAMPLE_COUNT and dm_targets.sample_id.is_unique,
        {"rows": int(len(dm_targets))},
    )

    # The dataset object_boxes.csv is a POSITIVE-contract subset; record its
    # coverage explicitly rather than silently substituting a different source.
    dob = data["dataset_object_boxes"]
    dob_keys = set(zip(dob.sample_id, dob.gt_actor_id))
    record(
        "dataset_object_boxes_coverage_reported",
        True,
        {
            "targets_present": len(target_keys & dob_keys),
            "targets_absent": len(target_keys - dob_keys),
            "note": (
                "dataset/object_boxes.csv keeps only contract-POSITIVE rows and omits "
                "some pilot targets; the hash-registered frozen validation view is used "
                "for box geometry instead"
            ),
        },
    )

    # Calibration identity between the named dataset manifest and the view.
    calib_cols = [
        "camera_fx", "camera_fy", "camera_cx", "camera_cy",
        "camera_matrix_json", "camera_inverse_matrix_json",
        "camera_width", "camera_height", "rgb_path", "frame_id", "timestamp",
    ]
    vm = data["view_manifest"]
    merged = dm_targets[["sample_id"] + calib_cols].merge(
        vm[vm.sample_id.isin(sm.sample_id)][["sample_id"] + calib_cols],
        on="sample_id", suffixes=("_dataset", "_view"),
    )
    identical = {
        column: bool((merged[column + "_dataset"] == merged[column + "_view"]).all())
        for column in calib_cols
    }
    record(
        "dataset_and_view_calibration_identical",
        all(identical.values()) and len(merged) == EXPECTED_SAMPLE_COUNT,
        identical,
    )

    # Box geometry identity between the view and the raw episode files.
    eb = pd.concat(
        [
            pd.read_csv(EPISODE_ROOT / episode / "object_boxes.csv", dtype={"gt_actor_id": str})
            for episode in sorted(sm.episode_id.unique())
        ],
        ignore_index=True,
    )
    geom_cols = [
        "gt_bbox_x", "gt_bbox_y", "gt_bbox_w", "gt_bbox_h",
        "object_world_x", "object_world_y", "object_world_z",
        "gt_extent_x_m", "gt_extent_y_m", "gt_extent_z_m",
        "object_yaw_deg", "gt_distance_m",
    ]
    pair = (
        sm[key]
        .merge(data["object_boxes"][key + geom_cols], on=key)
        .merge(eb[key + geom_cols], on=key, suffixes=("_view", "_episode"))
    )
    deltas = {
        column: float(
            np.abs(
                pair[column + "_view"].astype(float) - pair[column + "_episode"].astype(float)
            ).max()
        )
        for column in geom_cols
    }
    record(
        "view_and_episode_box_geometry_identical",
        len(pair) == EXPECTED_SAMPLE_COUNT and max(deltas.values()) == 0.0,
        deltas,
    )

    record(
        "every_target_is_labelled_person",
        bool(
            (sm[key].merge(data["object_boxes"], on=key).label == "person").all()
        ),
        sm[key].merge(data["object_boxes"], on=key).label.value_counts().to_dict(),
    )
    return checks


def frame_identity_checks(sm: pd.DataFrame, dm: pd.DataFrame, ov: pd.DataFrame) -> dict[str, Any]:
    """Assert the depth image, the RGB image and the metadata are the same frame."""
    key = ["sample_id", "gt_actor_id"]
    # Columns are renamed explicitly rather than relying on merge suffixes, so a
    # future schema change cannot silently repoint one of these comparisons.
    dataset = dm[["sample_id", "frame_id", "timestamp"]].rename(
        columns={"frame_id": "ds_frame_id", "timestamp": "ds_timestamp_s"}
    )
    vis = ov[key + ["frame_id", "timestamp", "depth_frame_id", "depth_timestamp_s"]].rename(
        columns={
            "frame_id": "ov_frame_id",
            "timestamp": "ov_timestamp_s",
            "depth_frame_id": "ov_depth_frame_id",
            "depth_timestamp_s": "ov_depth_timestamp_s",
        }
    )
    joined = sm.merge(dataset, on="sample_id", how="left", validate="one_to_one").merge(
        vis, on=key, how="left", validate="one_to_one"
    )
    parsed = joined.sample_id.str.rsplit("frame", n=1).str[-1].astype(int)
    return {
        "rows": int(len(joined)),
        "sample_id_frame_suffix_matches_frame_id": bool((parsed == joined.frame_id).all()),
        "rgb_frame_id_equals_frame_id": bool((joined.rgb_frame_id == joined.frame_id).all()),
        "depth_frame_id_equals_frame_id": bool(
            (joined.depth_frame_id == joined.frame_id).all()
            and (joined.ov_depth_frame_id == joined.frame_id).all()
        ),
        "dataset_manifest_frame_id_matches": bool((joined.ds_frame_id == joined.frame_id).all()),
        "visibility_frame_id_matches": bool((joined.ov_frame_id == joined.frame_id).all()),
        "max_rgb_depth_timestamp_delta_s": float(
            joined.rgb_depth_timestamp_delta_s.abs().max()
        ),
        "max_dataset_vs_depth_timestamp_delta_s": float(
            (joined.ds_timestamp_s - joined.ov_depth_timestamp_s).abs().max()
        ),
        "max_dataset_vs_visibility_timestamp_delta_s": float(
            (joined.ds_timestamp_s - joined.ov_timestamp_s).abs().max()
        ),
        "max_pilot_vs_dataset_depth_timestamp_delta_s": float(
            (joined.depth_timestamp_s - joined.ds_timestamp_s).abs().max()
        ),
    }


def score_all(data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    import cv2

    sm = data["sample_manifest"]
    key = ["sample_id", "gt_actor_id"]
    boxes = data["object_boxes"]
    visibility = data["object_visibility"]
    manifest = data["dataset_manifest"].set_index("sample_id")

    people = boxes[boxes.label == "person"]
    by_sample = {sid: group for sid, group in people.groupby("sample_id")}

    rows: list[dict[str, Any]] = []
    diagnostics = {
        "max_clipped_box_reconstruction_error_px": 0.0,
        "max_unclipped_box_reconstruction_error_px": 0.0,
        "max_round_trip_error_px": 0.0,
        "max_calibration_identity_error": 0.0,
        "min_depth_m": math.inf,
        "max_depth_m": -math.inf,
        "non_finite_depth_images": 0,
        "non_finite_calibration": 0,
        "accepted_points_outside_actor_volume": 0,
        "accepted_points_below_ground_margin": 0,
        "depth_images_opened": 0,
    }

    for _, sample in sm.iterrows():
        sid, aid = sample.sample_id, sample.gt_actor_id
        meta = manifest.loc[sid]
        target = boxes[(boxes.sample_id == sid) & (boxes.gt_actor_id == aid)].iloc[0]
        recorded = visibility[
            (visibility.sample_id == sid) & (visibility.gt_actor_id == aid)
        ].iloc[0]

        intrinsics = np.asarray(
            [
                [float(meta.camera_fx), 0.0, float(meta.camera_cx)],
                [0.0, float(meta.camera_fy), float(meta.camera_cy)],
                [0.0, 0.0, 1.0],
            ]
        )
        camera_matrix = np.asarray(json.loads(meta.camera_matrix_json), dtype=np.float64)
        camera_inverse = np.asarray(
            json.loads(meta.camera_inverse_matrix_json), dtype=np.float64
        )
        if not (
            np.all(np.isfinite(intrinsics))
            and np.all(np.isfinite(camera_matrix))
            and np.all(np.isfinite(camera_inverse))
        ):
            diagnostics["non_finite_calibration"] += 1
            raise AuditError(f"non-finite calibration for {sid}")
        identity_error = float(np.abs(camera_matrix @ camera_inverse - np.eye(4)).max())
        diagnostics["max_calibration_identity_error"] = max(
            diagnostics["max_calibration_identity_error"], identity_error
        )

        depth_path = DATASET_DIR / sample.depth_source_path
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise AuditError(f"unreadable depth image {depth_path}")
        depth_m = decode_depth_bgra(raw).astype(np.float64)
        diagnostics["depth_images_opened"] += 1
        if not np.all(np.isfinite(depth_m)):
            diagnostics["non_finite_depth_images"] += 1
            raise AuditError(f"non-finite depth in {depth_path}")
        if depth_m.shape != (CAMERA_HEIGHT, CAMERA_WIDTH):
            raise AuditError(f"unexpected depth shape {depth_m.shape} for {sid}")
        diagnostics["min_depth_m"] = min(diagnostics["min_depth_m"], float(depth_m.min()))
        diagnostics["max_depth_m"] = max(diagnostics["max_depth_m"], float(depth_m.max()))
        if float(depth_m.min()) <= 0.0 or float(depth_m.max()) > core.CARLA_MAX_DEPTH_M:
            raise AuditError(f"implausible depth range in {depth_path}")

        pedestrians = [
            {
                "key": str(person.gt_actor_id),
                "centre": (person.object_world_x, person.object_world_y, person.object_world_z),
                "extent": (person.gt_extent_x_m, person.gt_extent_y_m, person.gt_extent_z_m),
                "yaw_deg": float(person.object_yaw_deg),
            }
            for person in by_sample[sid].itertuples()
        ]
        result = scoring.score_actor_frame(
            depth_m=depth_m,
            camera_matrix=camera_matrix,
            camera_inverse=camera_inverse,
            intrinsics=intrinsics,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            target_key=str(aid),
            target_centre=(target.object_world_x, target.object_world_y, target.object_world_z),
            target_extent=(target.gt_extent_x_m, target.gt_extent_y_m, target.gt_extent_z_m),
            target_yaw_deg=float(target.object_yaw_deg),
            pedestrian_boxes=pedestrians,
        )

        clipped_error = max(
            abs(result["clipped_bbox_x"] - float(recorded.clipped_bbox_x)),
            abs(result["clipped_bbox_y"] - float(recorded.clipped_bbox_y)),
            abs(result["clipped_bbox_w"] - float(recorded.clipped_bbox_w)),
            abs(result["clipped_bbox_h"] - float(recorded.clipped_bbox_h)),
        )
        unclipped_error = max(
            abs(result["unclipped_bbox_x"] - float(recorded.unclipped_bbox_x)),
            abs(result["unclipped_bbox_y"] - float(recorded.unclipped_bbox_y)),
            abs(result["unclipped_bbox_w"] - float(recorded.unclipped_bbox_w)),
            abs(result["unclipped_bbox_h"] - float(recorded.unclipped_bbox_h)),
        )
        diagnostics["max_clipped_box_reconstruction_error_px"] = max(
            diagnostics["max_clipped_box_reconstruction_error_px"], clipped_error
        )
        diagnostics["max_unclipped_box_reconstruction_error_px"] = max(
            diagnostics["max_unclipped_box_reconstruction_error_px"], unclipped_error
        )

        # Independent re-derivation of the accepted set: round trip, actor-volume
        # membership and the ground rule are all re-asserted from scratch here.
        bounds = core.roi_pixel_bounds(
            {
                "clipped_bbox_x": result["clipped_bbox_x"],
                "clipped_bbox_y": result["clipped_bbox_y"],
                "clipped_bbox_w": result["clipped_bbox_w"],
                "clipped_bbox_h": result["clipped_bbox_h"],
            },
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
        )
        roi = core.back_project_roi(depth_m, bounds, camera_matrix, intrinsics)
        u_back, v_back, _ = core.project_points(roi["world"], camera_inverse, intrinsics)
        round_trip = float(
            max(np.abs(u_back - roi["u"]).max(), np.abs(v_back - roi["v"]).max())
        )
        diagnostics["max_round_trip_error_px"] = max(
            diagnostics["max_round_trip_error_px"], round_trip
        )
        owned = core.assign_competing_pedestrians(roi["world"], str(aid), pedestrians)["owned"]
        if int(np.count_nonzero(owned)) != int(result["retained_actor_point_count"]):
            raise AuditError(f"retained-point re-derivation mismatch for {sid}")
        if np.any(owned):
            local = core.actor_local_points(
                roi["world"][owned],
                (target.object_world_x, target.object_world_y, target.object_world_z),
                float(target.object_yaw_deg),
            )
            half = np.asarray(
                [target.gt_extent_x_m, target.gt_extent_y_m, target.gt_extent_z_m],
                dtype=np.float64,
            )
            diagnostics["accepted_points_outside_actor_volume"] += int(
                np.count_nonzero(
                    np.any(np.abs(local) > half + core.ACTOR_VOLUME_TOLERANCE_M, axis=1)
                )
            )
            diagnostics["accepted_points_below_ground_margin"] += int(
                np.count_nonzero(local[:, 2] <= -half[2] + core.GROUND_REJECT_MARGIN_M)
            )

        rows.append(
            {
                "sample_id": sid,
                "gt_actor_id": aid,
                "episode_id": sample.episode_id,
                "frame_id": int(sample.frame_id),
                "distance_m": float(sample.distance_m),
                "distance_band": sample.distance_band,
                "rgb_source_path": sample.rgb_source_path,
                "depth_source_path": sample.depth_source_path,
                **{k: v for k, v in result.items()},
                "old_depth_interval_visible_fraction": float(recorded.visible_fraction),
                "old_depth_interval_native_visible_px": int(recorded.native_visible_px),
                "old_depth_interval_sampled_roi_px": int(recorded.sampled_roi_px),
                "old_depth_interval_tier": recorded.visibility_tier,
                "recorded_clipped_box_error_px": clipped_error,
                "recorded_unclipped_box_error_px": unclipped_error,
                "round_trip_error_px": round_trip,
                "calibration_identity_error": identity_error,
            }
        )

    scored = pd.DataFrame(rows)
    scored["old_depth_interval_band"] = [
        core.band_for_score(core.clamp_unit(v))
        for v in scored.old_depth_interval_visible_fraction
    ]
    return scored, diagnostics


def qualify_geometry(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "check": "reconstructed_projected_box_matches_recorded",
            "passed": max(
                diagnostics["max_clipped_box_reconstruction_error_px"],
                diagnostics["max_unclipped_box_reconstruction_error_px"],
            )
            < MAX_BOX_RECONSTRUCTION_ERROR_PX,
            "detail": {
                "max_clipped_error_px": diagnostics["max_clipped_box_reconstruction_error_px"],
                "max_unclipped_error_px": diagnostics["max_unclipped_box_reconstruction_error_px"],
                "threshold_px": MAX_BOX_RECONSTRUCTION_ERROR_PX,
            },
        },
        {
            "check": "projection_back_projection_round_trip",
            "passed": diagnostics["max_round_trip_error_px"] < MAX_ROUND_TRIP_ERROR_PX,
            "detail": {
                "max_error_px": diagnostics["max_round_trip_error_px"],
                "threshold_px": MAX_ROUND_TRIP_ERROR_PX,
            },
        },
        {
            "check": "calibration_matrices_finite_and_mutually_inverse",
            "passed": diagnostics["non_finite_calibration"] == 0
            and diagnostics["max_calibration_identity_error"] < MAX_CALIBRATION_IDENTITY_ERROR,
            "detail": {
                "non_finite": diagnostics["non_finite_calibration"],
                "max_identity_error": diagnostics["max_calibration_identity_error"],
                "threshold": MAX_CALIBRATION_IDENTITY_ERROR,
            },
        },
        {
            "check": "depth_finite_and_physically_plausible",
            "passed": diagnostics["non_finite_depth_images"] == 0
            and diagnostics["min_depth_m"] > 0.0
            and diagnostics["max_depth_m"] <= core.CARLA_MAX_DEPTH_M,
            "detail": {
                "images_opened": diagnostics["depth_images_opened"],
                "min_depth_m": diagnostics["min_depth_m"],
                "max_depth_m": diagnostics["max_depth_m"],
                "non_finite_images": diagnostics["non_finite_depth_images"],
            },
        },
        {
            "check": "every_accepted_point_inside_actor_volume",
            "passed": diagnostics["accepted_points_outside_actor_volume"] == 0,
            "detail": {"violations": diagnostics["accepted_points_outside_actor_volume"]},
        },
        {
            "check": "no_accepted_point_violates_ground_rejection",
            "passed": diagnostics["accepted_points_below_ground_margin"] == 0,
            "detail": {"violations": diagnostics["accepted_points_below_ground_margin"]},
        },
    ]


def build_agreement(
    scored: pd.DataFrame, human: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = scored.merge(
        human[["panel_number", "sample_id", "visibility_band", "truncation_label"]],
        on="sample_id",
    ).rename(columns={"visibility_band_y": "human_band", "visibility_band_x": "auto_band"})
    scoreable = merged[merged.human_band != "ambiguous"].copy()

    report: dict[str, Any] = {
        "annotated_samples": int(len(merged)),
        "ambiguous_excluded": int((merged.human_band == "ambiguous").sum()),
        "scoreable_samples": int(len(scoreable)),
        "human_band_counts": merged.human_band.value_counts().to_dict(),
        "actor_volume": agreement_mod.evaluate(
            scoreable.human_band.tolist(),
            scoreable.visibility.tolist(),
            scoreable.auto_band.tolist(),
        ),
        "old_depth_interval": agreement_mod.evaluate(
            scoreable.human_band.tolist(),
            scoreable.old_depth_interval_visible_fraction.tolist(),
            scoreable.old_depth_interval_band.tolist(),
        ),
    }

    by_band: dict[str, Any] = {}
    for band in core.BAND_ORDER:
        subset = scoreable[scoreable.human_band == band]
        by_band[band] = {
            "n": int(len(subset)),
            "actor_volume": {
                "median": float(subset.visibility.median()) if len(subset) else float("nan"),
                "mean": float(subset.visibility.mean()) if len(subset) else float("nan"),
                "p25": float(subset.visibility.quantile(0.25)) if len(subset) else float("nan"),
                "p75": float(subset.visibility.quantile(0.75)) if len(subset) else float("nan"),
                "min": float(subset.visibility.min()) if len(subset) else float("nan"),
                "max": float(subset.visibility.max()) if len(subset) else float("nan"),
            },
            "old_depth_interval": {
                "median": float(subset.old_depth_interval_visible_fraction.median())
                if len(subset) else float("nan"),
                "mean": float(subset.old_depth_interval_visible_fraction.mean())
                if len(subset) else float("nan"),
            },
            "no_support_count": int(subset.no_support.sum()) if len(subset) else 0,
        }
    report["score_distribution_by_human_band"] = by_band

    medians = [by_band[band]["actor_volume"]["median"] for band in core.BAND_ORDER]
    report["median_monotonic_increasing"] = bool(
        all(
            math.isfinite(a) and math.isfinite(b) and b > a
            for a, b in zip(medians, medians[1:])
        )
    )
    report["medians_in_band_order"] = medians

    by_distance: dict[str, Any] = {}
    for name, low, high in DISTANCE_BANDS:
        subset = scoreable[(scoreable.distance_m >= low) & (scoreable.distance_m < high)]
        if len(subset) == 0:
            by_distance[name] = {"n": 0}
            continue
        by_distance[name] = {
            "n": int(len(subset)),
            "actor_volume": agreement_mod.evaluate(
                subset.human_band.tolist(), subset.visibility.tolist(), subset.auto_band.tolist()
            ),
            "old_depth_interval": agreement_mod.evaluate(
                subset.human_band.tolist(),
                subset.old_depth_interval_visible_fraction.tolist(),
                subset.old_depth_interval_band.tolist(),
            ),
            "median_actor_volume_visibility": float(subset.visibility.median()),
        }
    report["by_distance_band"] = by_distance
    return report, merged


def decide(qualification: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    new = report["actor_volume"]
    old = report["old_depth_interval"]
    gates = {
        "all_qualification_checks_pass": all(check["passed"] for check in qualification),
        "median_visibility_monotonic": bool(report["median_monotonic_increasing"]),
        "weighted_kappa_at_least_0_60": bool(
            math.isfinite(new["linear_weighted_cohen_kappa"])
            and new["linear_weighted_cohen_kappa"] >= MIN_WEIGHTED_KAPPA
        ),
        "balanced_accuracy_at_least_0_80": bool(
            math.isfinite(new["balanced_accuracy"])
            and new["balanced_accuracy"] >= MIN_BALANCED_ACCURACY
        ),
        "not_worse_than_old_on_kappa": bool(
            new["linear_weighted_cohen_kappa"] >= old["linear_weighted_cohen_kappa"]
        ),
        "not_worse_than_old_on_balanced_accuracy": bool(
            new["balanced_accuracy"] >= old["balanced_accuracy"]
        ),
    }
    feasible = all(gates.values())
    return {
        "gates": gates,
        "thresholds": {
            "min_weighted_kappa": MIN_WEIGHTED_KAPPA,
            "min_balanced_accuracy": MIN_BALANCED_ACCURACY,
        },
        "feasible": feasible,
        "terminal": TERMINAL_FEASIBLE if feasible else TERMINAL_NOT_FEASIBLE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="explicit create-only run directory name")
    args = parser.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise AuditError(
            'refusing to run without CUDA_VISIBLE_DEVICES="" (CPU-only audit)'
        )
    if "torch" in sys.modules:
        raise AuditError("torch is imported; this audit must stay prediction-blind")

    started = time.perf_counter()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_PARENT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # create-only

    decoder = assert_registered_decoder()
    data = load_inputs()
    qualification = qualify_provenance(data)
    identity = frame_identity_checks(
        data["sample_manifest"], data["dataset_manifest"], data["object_visibility"]
    )
    qualification.append(
        {
            "check": "depth_rgb_metadata_frame_identity",
            "passed": bool(
                identity["sample_id_frame_suffix_matches_frame_id"]
                and identity["rgb_frame_id_equals_frame_id"]
                and identity["depth_frame_id_equals_frame_id"]
                and identity["dataset_manifest_frame_id_matches"]
                and identity["visibility_frame_id_matches"]
                and identity["max_rgb_depth_timestamp_delta_s"] <= MAX_TIMESTAMP_DELTA_S
                and identity["max_dataset_vs_depth_timestamp_delta_s"] <= MAX_TIMESTAMP_DELTA_S
                and identity["max_dataset_vs_visibility_timestamp_delta_s"] <= MAX_TIMESTAMP_DELTA_S
                and identity["max_pilot_vs_dataset_depth_timestamp_delta_s"] <= MAX_TIMESTAMP_DELTA_S
            ),
            "detail": identity,
        }
    )

    provenance_failures = [c["check"] for c in qualification if not c["passed"]]
    if provenance_failures:
        payload = {
            "terminal": TERMINAL_INVALID,
            "failed_checks": provenance_failures,
            "qualification": qualification,
        }
        (run_dir / "QUALIFICATION_FAILED.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        print(TERMINAL_INVALID)
        return 2

    scored, diagnostics = score_all(data)
    qualification.extend(qualify_geometry(diagnostics))

    geometry_failures = [c["check"] for c in qualification if not c["passed"]]
    if geometry_failures:
        payload = {
            "terminal": TERMINAL_INVALID,
            "failed_checks": geometry_failures,
            "qualification": qualification,
        }
        (run_dir / "QUALIFICATION_FAILED.json").write_text(json.dumps(payload, indent=2))
        scored.to_csv(run_dir / "actor_volume_visibility_scores.csv", index=False)
        print(json.dumps(payload, indent=2))
        print(TERMINAL_INVALID)
        return 2

    report, merged = build_agreement(scored, data["human"])
    decision = decide(qualification, report)

    scores_path = run_dir / "actor_volume_visibility_scores.csv"
    merged_path = run_dir / "actor_volume_visibility_with_human_bands.csv"
    scored.to_csv(scores_path, index=False)
    merged.to_csv(merged_path, index=False)

    from .contact_sheet import build_contact_sheet

    sheet = build_contact_sheet(
        merged, DATASET_DIR, run_dir, VIEW_DIR / "object_boxes_all.csv"
    )

    metadata = {
        "schema": "route_b_publication_actor_volume_visibility_v1",
        "algorithm_version": core.ALGORITHM_VERSION,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_imported": "torch" in sys.modules,
        "model_or_prediction_artifacts_read": 0,
        "test_rows_read": 0,
        "carla_started": False,
        "depth_images_opened": diagnostics["depth_images_opened"],
        "decoder": decoder,
        "constants": {
            "actor_volume_tolerance_m": core.ACTOR_VOLUME_TOLERANCE_M,
            "ground_reject_margin_m": core.GROUND_REJECT_MARGIN_M,
            "binary_decision_threshold": core.BINARY_DECISION_THRESHOLD,
            "band_edges": [list(edge) for edge in core.BAND_EDGES],
        },
        "inputs": {
            "dataset_root": str(DATASET_ROOT.relative_to(REPO_ROOT)),
            "human_pilot_dir": str(HUMAN_PILOT_DIR.relative_to(REPO_ROOT)),
            "geometry_view_dir": str(VIEW_DIR.relative_to(REPO_ROOT)),
            "episode_root": str(EPISODE_ROOT.relative_to(REPO_ROOT)),
        },
        "input_hashes": {
            "sample_manifest.csv": sha256_file(HUMAN_PILOT_DIR / "sample_manifest.csv"),
            "annotator_A_visibility_bands.csv": sha256_file(
                HUMAN_PILOT_DIR / "annotator_A_visibility_bands.csv"
            ),
            "dataset/manifest.csv": sha256_file(DATASET_DIR / "manifest.csv"),
            "dataset/object_boxes.csv": sha256_file(DATASET_DIR / "object_boxes.csv"),
            "view/manifest.csv": sha256_file(VIEW_DIR / "manifest.csv"),
            "view/object_boxes_all.csv": sha256_file(VIEW_DIR / "object_boxes_all.csv"),
            "view/object_visibility_all.csv": sha256_file(
                VIEW_DIR / "object_visibility_all.csv"
            ),
        },
        "code_hashes": {
            name: sha256_file(Path(__file__).parent / name)
            for name in ("core.py", "scoring.py", "agreement.py", "run_audit.py", "contact_sheet.py")
        },
        "qualification": qualification,
        "frame_identity": identity,
        "geometry_diagnostics": diagnostics,
        "agreement": report,
        "decision": decision,
        "contact_sheet": sheet,
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "RUN_METADATA.json").write_text(json.dumps(metadata, indent=2, default=str))

    artifacts = {
        "actor_volume_visibility_scores.csv": sha256_file(scores_path),
        "actor_volume_visibility_with_human_bands.csv": sha256_file(merged_path),
        "contact_sheet.png": sha256_file(run_dir / "contact_sheet.png"),
        "RUN_METADATA.json": sha256_file(run_dir / "RUN_METADATA.json"),
    }
    (run_dir / "ARTIFACT_HASHES.json").write_text(json.dumps(artifacts, indent=2))
    (run_dir / decision["terminal"]).write_text(
        f"{decision['terminal']}\n{datetime.now(timezone.utc).isoformat()}\n"
    )

    print(json.dumps({"run_dir": str(run_dir), "decision": decision}, indent=2))
    print(decision["terminal"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
