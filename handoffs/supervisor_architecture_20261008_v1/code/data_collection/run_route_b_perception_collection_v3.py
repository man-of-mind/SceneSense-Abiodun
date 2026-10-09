#!/usr/bin/env python3
"""Collect one Route B perception v3 episode by extending the frozen v2 path.

Only depth persistence, depth-derived visibility metadata, visible-person masks,
v3 gates, and deterministic review artifacts are new.  Route driving, traffic,
radar, cadence, intervention, renderer, and cleanup behavior are inherited from
``run_route_b_perception_collection_v2``.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_collection.run_route_b_perception_collection_v2 as v2  # noqa: E402
from data_collection.route_b_perception_v3.visibility_v1 import (  # noqa: E402
    ALGORITHM_VERSION,
    CLEAR_THRESHOLD,
    DEPTH_ENCODING,
    DEPTH_TOLERANCE_M,
    MAX_DISTANCE_M,
    MIN_MODEL_VISIBLE_PX,
    MIN_PROJECTED_AREA_PX,
    TIER_CLEAR,
    TIER_MARGINAL,
    TIER_UNOBSERVABLE,
    VISIBLE_THRESHOLD,
    decode_depth_bgra,
    depth_image_bgra,
    depth_is_plausible,
    eligibility_flags,
    reconstruct_consistent_mask,
    visibility_from_corners,
    visibility_tier,
)


CONFIG_PATH = HERE / "configs" / "route_b_perception_v3.yaml"
SCHEMA_PATH = HERE / "route_b_perception_v3" / "SCHEMA.md"
MIN_EXERCISED_PERSON_ROWS = 4
REVIEW_CASES = 32
CANONICAL_EPISODE_KEYS = {
    ("train", "traffic_30_30", 501, 1501),
    ("train", "traffic_50_50", 502, 1502),
    ("train", "traffic_30_30", 503, 1503),
    ("train", "traffic_50_50", 504, 1504),
    ("val", "traffic_30_30", 601, 1601),
    ("val", "traffic_50_50", 602, 1602),
    ("test", "traffic_30_30", 701, 1701),
    ("test", "traffic_50_50", 702, 1702),
}

# Six additional independent train-only episodes, registered for the Route B v3.1
# expanded training view.  Purely additive: the canonical eight above are untouched and
# every other bound - 25 km/h, fast rasterizer, 2.0 s replenish, 600 s budget, roadblock
# clearing, no hybrid physics - still applies unchanged to these tuples.
ADDITIONAL_TRAIN_EPISODE_KEYS = {
    ("train", "traffic_30_30", 801, 1801),
    ("train", "traffic_50_50", 802, 1802),
    ("train", "traffic_30_30", 803, 1803),
    ("train", "traffic_50_50", 804, 1804),
    ("train", "traffic_30_30", 805, 1805),
    ("train", "traffic_50_50", 806, 1806),
}
REGISTERED_EPISODE_KEYS = CANONICAL_EPISODE_KEYS | ADDITIONAL_TRAIN_EPISODE_KEYS

VISIBILITY_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "timestamp", "gt_actor_id", "label",
    "gt_actor_type_id", "depth_path", "depth_frame_id", "depth_timestamp_s",
    "unclipped_bbox_x", "unclipped_bbox_y", "unclipped_bbox_w", "unclipped_bbox_h",
    "clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h",
    "unclipped_projected_area_px", "clipped_projected_area_px",
    "projected_box_in_frame_fraction", "gt_distance_m", "actor_near_depth_m",
    "actor_far_depth_m", "sampled_roi_px", "native_visible_px",
    "model_input_visible_px", "visible_fraction", "occluder_closer_fraction",
    "background_farther_fraction", "geometry_qualified_v2",
    "eligible_visible_v010", "eligible_clear_v025", "visibility_tier",
    "person_mask_candidate_px", "person_mask_painted_px", "frame_person_mask_px",
    "depth_tolerance_m", "visibility_algorithm_version", "low_support_fallback_used",
)

DEPTH_FRAME_FIELDS = (
    "experiment_id", "sample_id", "frame_id", "rgb_frame_id", "semantic_frame_id",
    "depth_frame_id", "radar_frame_id", "rgb_timestamp_s", "semantic_timestamp_s",
    "depth_timestamp_s", "radar_timestamp_s", "max_timestamp_delta_s", "depth_path",
    "depth_bytes", "depth_min_m", "depth_max_m", "depth_finite",
    "depth_physically_plausible", "encoding", "algorithm_version",
)


def _append_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summary(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None,
                "p50": None, "p90": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "minimum": float(array.min()),
        "mean": float(array.mean()), "maximum": float(array.max()),
        "p50": float(np.quantile(array, 0.50)), "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _distance_bin(distance_m: float) -> str:
    value = float(distance_m)
    if value <= 10.0:
        return "0_10m"
    if value <= 20.0:
        return "10_20m"
    if value <= 30.0:
        return "20_30m"
    if value <= 40.0:
        return "30_40m"
    return "over_40m"


def _select_review_rows(rows: list[dict[str, Any]], maximum: int = REVIEW_CASES) -> list[dict[str, Any]]:
    """Deterministic round-robin selection over actual strata; never fabricates one."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        edge = "edge" if float(row["projected_box_in_frame_fraction"]) < 0.95 else "interior"
        key = ":".join((str(row["label"]), str(row["visibility_tier"]),
                        _distance_bin(float(row["gt_distance_m"])), edge))
        buckets[key].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: (
            str(row["sample_id"]), int(row["gt_actor_id"]), int(row["frame_id"])))
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    depth = 0
    while len(selected) < maximum:
        added = False
        for key in keys:
            values = buckets[key]
            if depth < len(values):
                selected.append({**values[depth], "review_stratum": key})
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        depth += 1
    return selected


def _write_review_artifacts(collector: "PerceptionCollectorV3") -> dict[str, Any]:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    selected = _select_review_rows(collector.visibility_rows)
    selection_path = collector.output_dir / "manual_review_selection.csv"
    selection_fields = (
        "review_index", "review_stratum", "sample_id", "frame_id", "gt_actor_id", "label",
        "visibility_tier", "gt_distance_m", "visible_fraction", "model_input_visible_px",
        "projected_box_in_frame_fraction", "eligible_visible_v010", "eligible_clear_v025",
        "panel_path",
    )
    panels_dir = collector.output_dir / "manual_review_panels"
    panels_dir.mkdir(exist_ok=False)
    selection_rows: list[dict[str, Any]] = []
    sheet_items: list[tuple[np.ndarray, dict[str, Any]]] = []

    for index, row in enumerate(selected):
        rgb_path = collector.output_dir / "rgb" / f"{row['sample_id']}.jpg"
        mask_path = collector.output_dir / "masks" / f"{row['sample_id']}.png"
        depth_path = collector.output_dir / str(row["depth_path"])
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        combined = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or combined is None or raw_depth is None:
            raise v2.PilotError(f"manual-review source missing for {row['sample_id']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_m = decode_depth_bgra(raw_depth)
        consistent = reconstruct_consistent_mask(
            depth_m, row, width=collector.args.camera_width, height=collector.args.camera_height)

        x0 = max(0, int(math.floor(float(row["clipped_bbox_x"]))))
        y0 = max(0, int(math.floor(float(row["clipped_bbox_y"]))))
        x1 = min(collector.args.camera_width, max(x0 + 1, int(math.ceil(
            float(row["clipped_bbox_x"]) + float(row["clipped_bbox_w"])))))
        y1 = min(collector.args.camera_height, max(y0 + 1, int(math.ceil(
            float(row["clipped_bbox_y"]) + float(row["clipped_bbox_h"])))))
        pad_x, pad_y = max(12, (x1 - x0) // 2), max(12, (y1 - y0) // 3)
        sx0, sy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        sx1, sy1 = min(collector.args.camera_width, x1 + pad_x), min(collector.args.camera_height, y1 + pad_y)
        old = np.zeros_like(consistent)
        old[y0:y1, x0:x1] = True
        new = (combined == 2) & consistent if row["label"] == "person" else consistent

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
        axes[0].imshow(rgb)
        axes[0].add_patch(Rectangle((float(row["clipped_bbox_x"]), float(row["clipped_bbox_y"])),
                                    float(row["clipped_bbox_w"]), float(row["clipped_bbox_h"]),
                                    fill=False, edgecolor="magenta", linewidth=2))
        axes[0].set_title("RGB + projected actor box")
        axes[0].axis("off")
        axes[1].imshow(depth_m[sy0:sy1, sx0:sx1], cmap="viridis", vmin=0,
                       vmax=min(60.0, max(1.0, float(row["actor_far_depth_m"]) + 10.0)))
        axes[1].set_title("decoded depth (m), zoom")
        axes[1].axis("off")
        overlay = rgb[sy0:sy1, sx0:sx1].copy()
        overlay[consistent[sy0:sy1, sx0:sx1]] = (
            0.35 * overlay[consistent[sy0:sy1, sx0:sx1]] + 0.65 * np.array([255, 0, 255])
        ).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title("counted depth-consistent pixels")
        axes[2].axis("off")
        comparison = np.zeros((sy1 - sy0, 2 * (sx1 - sx0), 3), dtype=np.uint8)
        old_crop, new_crop = old[sy0:sy1, sx0:sx1], new[sy0:sy1, sx0:sx1]
        comparison[:, :sx1 - sx0, 0][old_crop] = 255
        comparison[:, sx1 - sx0:, 1][new_crop] = 255
        axes[3].imshow(comparison)
        axes[3].set_title("old filled box (red) | v3 visible mask (green)")
        axes[3].axis("off")
        axes[4].axis("off")
        axes[4].text(0.0, 1.0, (
            f"sample: {row['sample_id']}\nactor: {row['gt_actor_id']} ({row['label']})\n"
            f"distance: {float(row['gt_distance_m']):.2f} m\n"
            f"visible fraction: {float(row['visible_fraction']):.4f}\n"
            f"native/model px: {row['native_visible_px']}/{row['model_input_visible_px']}\n"
            f"in-frame fraction: {float(row['projected_box_in_frame_fraction']):.4f}\n"
            f"tier: {row['visibility_tier']}\n"
            f"eligible v010/v025: {bool(row['eligible_visible_v010'])}/{bool(row['eligible_clear_v025'])}\n"
            "depth-derived visible-region approximation;\nnot an anatomical silhouette"
        ), va="top", family="monospace", fontsize=9)
        panel_rel = Path("manual_review_panels") / f"case_{index:02d}_{row['sample_id']}_actor{row['gt_actor_id']}.png"
        fig.tight_layout()
        fig.savefig(collector.output_dir / panel_rel, dpi=110, bbox_inches="tight")
        plt.close(fig)
        selection_rows.append({
            "review_index": index, "panel_path": str(panel_rel),
            **{field: row.get(field, "") for field in selection_fields
               if field not in {"review_index", "panel_path"}},
        })
        sheet_items.append((rgb, row))

    _append_csv(selection_path, selection_fields, selection_rows)
    if sheet_items:
        fig, axes = plt.subplots(8, 4, figsize=(16, 18))
        flat = list(np.asarray(axes).flat)
        colours = {TIER_CLEAR: "lime", TIER_MARGINAL: "orange", TIER_UNOBSERVABLE: "red"}
        for axis, (rgb, row) in zip(flat, sheet_items):
            axis.imshow(rgb)
            axis.add_patch(Rectangle((float(row["clipped_bbox_x"]), float(row["clipped_bbox_y"])),
                                     float(row["clipped_bbox_w"]), float(row["clipped_bbox_h"]),
                                     fill=False, edgecolor=colours[str(row["visibility_tier"])], linewidth=2))
            axis.set_title(f"{row['label']} {row['visibility_tier']} vf={float(row['visible_fraction']):.3f}", fontsize=7)
            axis.axis("off")
        for axis in flat[len(sheet_items):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(collector.output_dir / "manual_review_contact_sheet.png", dpi=100)
        plt.close(fig)

    actual = Counter(str(row["review_stratum"]) for row in selected)
    return {
        "requested_cases": REVIEW_CASES,
        "actual_cases": len(selected),
        "selection_artifact": str(selection_path),
        "panels_directory": str(panels_dir),
        "full_frame_contact_sheet": str(collector.output_dir / "manual_review_contact_sheet.png"),
        "actual_selected_strata": dict(sorted(actual.items())),
    }


class PerceptionCollectorV3(v2.PerceptionCollectorV2):
    """v2 collector with reversible visibility evidence and no mask fallback."""

    def __init__(self, **kwargs: Any) -> None:
        self.visibility_rows: list[dict[str, Any]] = []
        self.depth_frame_rows: list[dict[str, Any]] = []
        self.depth_alignment_records: list[dict[str, Any]] = []
        self.depth_decode_wall_s: list[float] = []
        self.visibility_wall_s: list[float] = []
        self.v3_prepare_wall_s: list[float] = []
        self.depth_bytes = 0
        self.depth_roundtrip_verified = False
        self.depth_invalid_frames: list[int] = []
        self.visibility_incomplete_rows = 0
        self.visibility_flag_mismatches = 0
        self.visibility_object_rows_total = 0
        self.person_mask_subset_failures = 0
        self.ineligible_person_paint_events = 0
        self.vehicle_mask_preservation_failures = 0
        self.low_support_fallback_events = 0
        self.depth_hidden_ticks = 0
        self._waited: dict[str, Any] = {}
        self._depth_cache: dict[str, Any] = {}
        super().__init__(**kwargs)

    def _write_metadata(self) -> None:
        self.depth_dir = self.output_dir / "depth"
        self.depth_dir.mkdir(exist_ok=False)
        self.visibility_path = self.output_dir / "object_visibility.csv"
        self.depth_frames_path = self.output_dir / "depth_frames.csv"
        super()._write_metadata()
        metadata_path = self.output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({
            "schema": "scenesense_moving_ego_fusion_training_data.v3",
            "description": "Route B v3: v2 contract plus synchronized depth visibility evidence.",
            "versioned_parent": "data_collection/run_route_b_perception_collection_v2.py",
            "v3_config": str(CONFIG_PATH),
            "v3_schema": str(SCHEMA_PATH),
            "depth_visibility": {
                "blueprint": str(self.depth_camera.type_id),
                "configured_attributes": {
                    key: str(self.depth_camera.attributes.get(key))
                    for key in ("image_size_x", "image_size_y", "fov", "sensor_tick")
                },
                "colocated_with_rgb": True,
                "encoding": DEPTH_ENCODING,
                "decode": "(R + G*256 + B*256^2)/(256^3-1) * 1000 m",
                "lossy": False,
                "depth_tolerance_m": DEPTH_TOLERANCE_M,
                "algorithm_version": ALGORITHM_VERSION,
                "thresholds": {
                    "max_distance_m": MAX_DISTANCE_M,
                    "min_projected_area_px": MIN_PROJECTED_AREA_PX,
                    "min_model_input_visible_px": MIN_MODEL_VISIBLE_PX,
                    "eligible_visible_v010": VISIBLE_THRESHOLD,
                    "eligible_clear_v025": CLEAR_THRESHOLD,
                },
                "visibility_csv": "object_visibility.csv",
                "depth_frame_manifest": "depth_frames.csv",
                "raw_depth_directory": "depth/",
                "semantic_walker_tags_used": False,
                "low_support_full_box_or_ellipse_fallback": False,
                "person_mask_description": (
                    "depth-derived visible-region approximation; not a guaranteed anatomical silhouette"
                ),
            },
        })
        self.parked.save_json(metadata_path, metadata)
        self.parked.save_json(self.output_dir / "resolved_config.json", {
            "schema": "scenesense_route_b_perception_collection.resolved.v3",
            "parent_metadata": metadata,
            "density": self.density,
            "split": self.split,
            "target_speed_kph": self.target_speed_kph,
            "hybrid_physics": self.hybrid_physics,
            "radar_points_per_second": self.args.radar_points_per_second,
            "rasterizer": self.args.radar_rasterizer,
            "create_only_output": True,
        })

    def _wait_exact(self, name: str, frame_id: int) -> Any:
        item = super()._wait_exact(name, frame_id)
        if name in v2.CAMERA_NAMES:
            self._waited[name] = item
        if name == "depth":
            started = time.perf_counter()
            raw = depth_image_bgra(item)
            depth_m = decode_depth_bgra(raw)
            elapsed = time.perf_counter() - started
            self.depth_decode_wall_s.append(elapsed)
            if not depth_is_plausible(depth_m):
                self.depth_invalid_frames.append(int(frame_id))
                raise v2.PilotError(f"invalid decoded depth at frame {frame_id}")
            self._depth_cache = {
                "frame_id": int(frame_id), "measurement": item,
                "raw_bgra": raw, "depth_m": depth_m,
            }
        return item

    def prepare_input(self, frame_id: int, route_tick: int, radar_measurement: Any,
                      sweep_index: int) -> None:
        self._waited = {}
        started = time.perf_counter()
        super().prepare_input(frame_id, route_tick, radar_measurement, sweep_index)
        elapsed = time.perf_counter() - started
        self.v3_prepare_wall_s.append(elapsed)
        rgb, semantic, depth = (self._waited.get(name) for name in v2.CAMERA_NAMES)
        if rgb is None or semantic is None or depth is None:
            raise v2.PilotError(f"incomplete v3 camera ownership at prepared frame {frame_id}")
        ids = [int(rgb.frame), int(semantic.frame), int(depth.frame), int(radar_measurement.frame)]
        timestamps = [float(rgb.timestamp), float(semantic.timestamp), float(depth.timestamp),
                      float(radar_measurement.timestamp)]
        delta = max(timestamps) - min(timestamps)
        transform_delta = max(
            self._transform_delta_m(rgb.transform, depth.transform),
            abs(float(rgb.transform.rotation.pitch) - float(depth.transform.rotation.pitch)),
            abs(float(rgb.transform.rotation.yaw) - float(depth.transform.rotation.yaw)),
            abs(float(rgb.transform.rotation.roll) - float(depth.transform.rotation.roll)),
        )
        persisted = bool(self.sample_stats and int(self.sample_stats[-1]["frame_id"]) == int(frame_id))
        self.depth_alignment_records.append({
            "frame_id": int(frame_id), "route_tick": int(route_tick),
            "sensor_frame_ids": ids, "frame_ids_exact": len(set(ids + [int(frame_id)])) == 1,
            "sensor_timestamps_s": timestamps, "timestamp_delta_s": delta,
            "timestamp_exact": delta == 0.0, "rgb_depth_transform_delta": transform_delta,
            "persisted": persisted,
        })

    def _persist(self, *, frame_id: int, route_tick: int, image: Any,
                 semantic_image: Any, radar_measurement: Any, radar_tensor: Any,
                 radar_points: dict[str, Any], radar_summary: dict[str, float],
                 camera_matrix: Any, camera_inverse: Any, radar_matrix: Any,
                 radar_inverse: Any, window_meta: dict[str, Any], timestamp_delta: float,
                 prepared_timestamp_s: float, ego_speed_mps: float,
                 ego_velocity_mps: tuple[float, float, float]) -> None:
        import cv2

        if int(self._depth_cache.get("frame_id", -1)) != int(frame_id):
            raise v2.PilotError(f"saved frame {frame_id} does not own its depth measurement")
        depth_image = self._depth_cache["measurement"]
        depth_raw = self._depth_cache["raw_bgra"]
        depth_m = self._depth_cache["depth_m"]
        sample_id = f"{self.experiment_id}_{self.saved:06d}_frame{frame_id}"

        radar_points = dict(radar_points)
        radar_points.update(window_meta["raw_provenance"])
        radar_points["prepared_timestamp_s"] = self.np.asarray(
            [prepared_timestamp_s], dtype=self.np.float64)
        radar_points["ego_speed_mps"] = self.np.asarray([ego_speed_mps], dtype=self.np.float32)
        radar_points["ego_velocity_mps"] = self.np.asarray(ego_velocity_mps, dtype=self.np.float32)
        file_paths, mask = self.parked.save_sample_files(
            dataset_dir=self.output_dir, dirs=self.dirs, sample_id=sample_id,
            image=image, semantic_image=semantic_image, radar_tensor=radar_tensor,
            radar_points=radar_points, jpeg_quality=self.args.jpeg_quality)
        depth_path = self.depth_dir / f"{sample_id}.png"
        if not cv2.imwrite(str(depth_path), depth_raw, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
            raise v2.PilotError(f"failed to write lossless depth PNG at frame {frame_id}")
        file_paths["depth_path"] = depth_path
        if not self.depth_roundtrip_verified:
            decoded_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if decoded_raw is None or not np.array_equal(decoded_raw, depth_raw):
                raise v2.PilotError("lossless depth PNG round-trip verification failed")
            self.depth_roundtrip_verified = True

        sample_base = {
            "experiment_id": self.experiment_id, "sample_id": sample_id,
            "frame_id": frame_id, "timestamp": float(image.timestamp),
            "traffic_light_id": "", "scenario_id": self.scenario_id,
            "view_id": "qualified_route_b_controller",
        }
        visibility_started = time.perf_counter()
        object_rows = self.parked.build_object_rows(
            world=self.world, ego_vehicle=self.ego, sample_base=sample_base,
            camera_location=self.camera.get_transform().location,
            camera_matrix=camera_matrix, camera_inverse_matrix=camera_inverse,
            intrinsics=self.intrinsics_full, width=self.args.camera_width,
            height=self.args.camera_height, max_distance_m=self.args.gt_max_distance_m,
            radar_world_xyz=self.np.asarray(radar_points["world_xyz"], dtype=self.np.float32),
            stationary_tracker=self.actor_tracker, include_pedestrians=True,
            radar_support_margin_m=self.args.radar_support_margin_m,
            radar_person_support_mode=self.args.radar_person_support_mode,
            radar_person_support_radius_m=self.args.radar_person_support_radius_m,
            radar_person_support_z_down_m=self.args.radar_person_support_z_down_m,
            radar_person_support_z_up_m=self.args.radar_person_support_z_up_m)
        self.visibility_object_rows_total += len(object_rows)
        frame_visibility: list[dict[str, Any]] = []
        person_masks: list[tuple[dict[str, Any], np.ndarray]] = []
        depth_rel = str(depth_path.relative_to(self.output_dir))
        for object_row in object_rows:
            actor = self.world.get_actor(int(object_row["gt_actor_id"]))
            if actor is None:
                self.visibility_incomplete_rows += 1
                raise v2.PilotError(
                    f"object actor {object_row['gt_actor_id']} disappeared at frame {frame_id}")
            _center, corners = self.parked.actor_bbox_world_points(actor)
            try:
                metrics, consistent = visibility_from_corners(
                    corners, camera_inverse, self.intrinsics_full, depth_m,
                    distance_m=float(object_row["gt_distance_m"]),
                    width=self.args.camera_width, height=self.args.camera_height,
                    model_width=self.args.model_input_width,
                    model_height=self.args.model_input_height,
                    tolerance_m=DEPTH_TOLERANCE_M)
            except (ValueError, FloatingPointError) as exc:
                self.visibility_incomplete_rows += 1
                raise v2.PilotError(
                    f"visibility failed for {sample_id}/actor{object_row['gt_actor_id']}: {exc}") from exc
            for source_key, metric_key in (
                ("gt_bbox_x", "clipped_bbox_x"), ("gt_bbox_y", "clipped_bbox_y"),
                ("gt_bbox_w", "clipped_bbox_w"), ("gt_bbox_h", "clipped_bbox_h")):
                if abs(float(object_row[source_key]) - float(metrics[metric_key])) > 1e-6:
                    raise v2.PilotError(
                        f"v2/v3 projection mismatch at {sample_id}/actor{object_row['gt_actor_id']}")
            expected = eligibility_flags(
                distance_m=float(object_row["gt_distance_m"]),
                projected_area_px=float(metrics["clipped_projected_area_px"]),
                model_visible_px=int(metrics["model_input_visible_px"]),
                visible_fraction=float(metrics["visible_fraction"]))
            if expected != (bool(metrics["eligible_visible_v010"]),
                            bool(metrics["eligible_clear_v025"])):
                self.visibility_flag_mismatches += 1
            row = {
                "experiment_id": self.experiment_id, "sample_id": sample_id,
                "frame_id": frame_id, "timestamp": float(image.timestamp),
                "gt_actor_id": object_row["gt_actor_id"], "label": object_row["label"],
                "gt_actor_type_id": object_row["gt_actor_type_id"],
                "depth_path": depth_rel, "depth_frame_id": int(depth_image.frame),
                "depth_timestamp_s": float(depth_image.timestamp),
                "gt_distance_m": float(object_row["gt_distance_m"]),
                **metrics,
                "person_mask_candidate_px": 0, "person_mask_painted_px": 0,
                "frame_person_mask_px": 0, "low_support_fallback_used": False,
            }
            frame_visibility.append(row)
            if object_row.get("label") == "person":
                person_masks.append((row, consistent))

        vehicle_pixels_before = int(np.count_nonzero(mask == 1))
        mask[mask == 2] = 0
        eligible_union = np.zeros(mask.shape, dtype=bool)
        for row, consistent in person_masks:
            if not bool(row["eligible_visible_v010"]):
                row["person_mask_candidate_px"] = int(np.count_nonzero(consistent))
                row["person_mask_painted_px"] = 0
                continue
            candidate = consistent & (mask != 1)
            row["person_mask_candidate_px"] = int(np.count_nonzero(consistent))
            row["person_mask_painted_px"] = int(np.count_nonzero(candidate))
            eligible_union |= candidate
            mask[candidate] = 2
        frame_person_pixels = int(np.count_nonzero(mask == 2))
        for row in frame_visibility:
            row["frame_person_mask_px"] = frame_person_pixels
        if np.any((mask == 2) & ~eligible_union):
            self.person_mask_subset_failures += 1
            raise v2.PilotError(f"person mask escaped depth-consistent union at frame {frame_id}")
        if any(not bool(row["eligible_visible_v010"]) and int(row["person_mask_painted_px"]) > 0
               for row, _consistent in person_masks):
            self.ineligible_person_paint_events += 1
            raise v2.PilotError(f"ineligible person painted at frame {frame_id}")
        if int(np.count_nonzero(mask == 1)) != vehicle_pixels_before:
            self.vehicle_mask_preservation_failures += 1
            raise v2.PilotError(f"vehicle semantic mask changed at frame {frame_id}")
        if not cv2.imwrite(str(file_paths["mask_path"]), mask):
            raise v2.PilotError(f"failed to write v3 person mask at frame {frame_id}")
        self.visibility_wall_s.append(time.perf_counter() - visibility_started)

        manifest_row = self.parked.build_manifest_row(
            args=self.args, dataset_dir=self.output_dir, experiment_id=self.experiment_id,
            sample_id=sample_id, split=self.split, file_paths=file_paths, image=image,
            semantic_image=semantic_image, radar_measurement=radar_measurement, mask=mask,
            world=self.world, camera=self.camera, radar=self.radar, ego_vehicle=self.ego,
            camera_matrix=camera_matrix, camera_inverse_matrix=camera_inverse,
            radar_matrix=radar_matrix, radar_inverse_matrix=radar_inverse,
            intrinsics_full=self.intrinsics_full, radar_summary=radar_summary)
        manifest_row["scenario_id"] = self.scenario_id
        manifest_row["view_id"] = "qualified_route_b_controller"
        manifest_row["vehicle_pixels"] = int(np.count_nonzero(mask == 1))
        manifest_row["person_pixels"] = frame_person_pixels
        counts = self._density_counts(object_rows)

        missing = [str(path) for path in file_paths.values() if not Path(path).is_file()]
        if missing:
            raise v2.PilotError(f"saved record missing at frame {frame_id}: {missing}")
        sample_bytes = sum(Path(path).stat().st_size for path in file_paths.values())
        depth_size = int(depth_path.stat().st_size)
        self.saved_bytes += sample_bytes
        self.depth_bytes += depth_size
        self.parked.append_manifest_rows(self.manifest_path, [manifest_row])
        self.parked.append_object_box_rows(self.object_boxes_path, object_rows)
        _append_csv(self.visibility_path, VISIBILITY_FIELDS, frame_visibility)

        sensor_items = (image, semantic_image, depth_image, radar_measurement)
        sensor_timestamps = [float(item.timestamp) for item in sensor_items]
        depth_frame_row = {
            "experiment_id": self.experiment_id, "sample_id": sample_id, "frame_id": frame_id,
            "rgb_frame_id": int(image.frame), "semantic_frame_id": int(semantic_image.frame),
            "depth_frame_id": int(depth_image.frame), "radar_frame_id": int(radar_measurement.frame),
            "rgb_timestamp_s": float(image.timestamp),
            "semantic_timestamp_s": float(semantic_image.timestamp),
            "depth_timestamp_s": float(depth_image.timestamp),
            "radar_timestamp_s": float(radar_measurement.timestamp),
            "max_timestamp_delta_s": max(sensor_timestamps) - min(sensor_timestamps),
            "depth_path": depth_rel, "depth_bytes": depth_size,
            "depth_min_m": float(depth_m.min()), "depth_max_m": float(depth_m.max()),
            "depth_finite": bool(np.all(np.isfinite(depth_m))),
            "depth_physically_plausible": depth_is_plausible(depth_m),
            "encoding": DEPTH_ENCODING, "algorithm_version": ALGORITHM_VERSION,
        }
        _append_csv(self.depth_frames_path, DEPTH_FRAME_FIELDS, [depth_frame_row])
        self.visibility_rows.extend(frame_visibility)
        self.depth_frame_rows.append(depth_frame_row)

        self.runtime_provenance_rows.append({
            "sample_id": sample_id, "frame_id": frame_id,
            "prepared_timestamp_s": round(float(prepared_timestamp_s), 6),
            "ego_speed_mps": round(float(ego_speed_mps), 6),
            "ego_velocity_x_mps": round(float(ego_velocity_mps[0]), 6),
            "ego_velocity_y_mps": round(float(ego_velocity_mps[1]), 6),
            "ego_velocity_z_mps": round(float(ego_velocity_mps[2]), 6),
            "radar_window_returns": int(window_meta["returns"]),
            "sweep_index": int(window_meta["sweep_indices"][-1]),
        })
        population = self._live_population()
        controller_health = self._controller_health()
        self.population_samples.append({
            "sample_id": sample_id, "frame_id": frame_id,
            "timestamp_s": float(image.timestamp), **population})
        self.controller_health_samples.append({
            "sample_id": sample_id, "frame_id": frame_id,
            "timestamp_s": float(image.timestamp), **controller_health})
        self.saved += 1
        tier_counts = Counter(str(row["visibility_tier"]) for row in frame_visibility)
        self.sample_stats.append({
            "sample_id": sample_id, "frame_id": frame_id, "timestamp_s": float(image.timestamp),
            "route_tick": route_tick, "sample_bytes": int(sample_bytes),
            "timestamp_delta_s": float(timestamp_delta),
            "radar_window_returns": int(window_meta["returns"]),
            "raw_vehicle_count": sum(row.get("label") == "vehicle" for row in object_rows),
            "raw_person_count": sum(row.get("label") == "person" for row in object_rows),
            "visibility_clear_count": tier_counts[TIER_CLEAR],
            "visibility_marginal_count": tier_counts[TIER_MARGINAL],
            "visibility_unobservable_count": tier_counts[TIER_UNOBSERVABLE],
            "person_mask_pixels": frame_person_pixels, "depth_bytes": depth_size,
            **population, **counts, **controller_health,
        })
        if self.saved == 1 or self.saved % 100 == 0:
            print(
                f"v3 perception saved={self.saved} prepared={self.plan.prepared} frame={frame_id} "
                f"objects={len(object_rows)} tiers={dict(tier_counts)} depth_bytes={depth_size} "
                f"person_mask_px={frame_person_pixels} timestamp_delta_s={timestamp_delta:.9f}",
                flush=True)

    def write_summary(self, route_result: dict[str, Any] | None, error: str = "", *,
                      replenish_interval_s: float = 20.0) -> None:
        super().write_summary(route_result, error, replenish_interval_s=replenish_interval_s)
        summary_path = self.output_dir / "route_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        v2_gates = dict(summary["gates"])
        required_fields = set(VISIBILITY_FIELDS) - {
            "person_mask_candidate_px", "person_mask_painted_px", "frame_person_mask_px"}
        complete = all(all(field in row and row[field] != "" for field in required_fields)
                       for row in self.visibility_rows)
        reproduced = all(
            eligibility_flags(
                distance_m=float(row["gt_distance_m"]),
                projected_area_px=float(row["clipped_projected_area_px"]),
                model_visible_px=int(row["model_input_visible_px"]),
                visible_fraction=float(row["visible_fraction"]))
            == (bool(row["eligible_visible_v010"]), bool(row["eligible_clear_v025"]))
            and visibility_tier(float(row["visible_fraction"]), int(row["model_input_visible_px"]))
            == str(row["visibility_tier"])
            for row in self.visibility_rows)
        alignment = self.depth_alignment_records
        saved_alignment = [row for row in alignment if row["persisted"]]
        metadata = json.loads((self.output_dir / "metadata.json").read_text(encoding="utf-8"))
        depth_meta = metadata.get("depth_visibility", {})
        v3_gates = {
            "depth_callback_frame_timestamp_alignment_exact_prepared_and_saved": bool(alignment)
                and len(alignment) == self.plan.prepared
                and len(saved_alignment) == self.saved
                and all(row["frame_ids_exact"] and row["timestamp_exact"] for row in alignment),
            "no_missing_or_invalid_depth_frame": self.saved > 0
                and len(self.depth_frame_rows) == self.saved and not self.depth_invalid_frames,
            "decoded_depth_finite_and_physically_plausible": bool(self.depth_frame_rows)
                and all(row["depth_finite"] and row["depth_physically_plausible"]
                        for row in self.depth_frame_rows),
            "rgb_depth_colocation_exact": bool(alignment)
                and all(float(row["rgb_depth_transform_delta"]) <= 1e-6 for row in alignment),
            "visibility_fields_complete_for_every_object_row": bool(self.visibility_rows)
                and len(self.visibility_rows) == self.visibility_object_rows_total
                and complete and self.visibility_incomplete_rows == 0,
            "visibility_flags_and_tiers_reproduce_registered_functions": reproduced
                and self.visibility_flag_mismatches == 0,
            "person_mask_pixels_subset_of_depth_consistent_pixels":
                self.person_mask_subset_failures == 0,
            "no_person_paint_for_ineligible_v010": self.ineligible_person_paint_events == 0,
            "vehicle_semantic_mask_preserved": self.vehicle_mask_preservation_failures == 0,
            "no_full_box_or_ellipse_fallback": self.low_support_fallback_events == 0,
            "visibility_rows_reconcile_with_object_rows":
                len(self.visibility_rows) == self.visibility_object_rows_total,
            "depth_frame_tick_ownership_exact_no_hidden_ticks": self.depth_hidden_ticks == 0
                and len(alignment) == self.plan.prepared,
            "depth_lossless_png_roundtrip_verified": self.depth_roundtrip_verified,
            "v3_manifest_config_complete": (
                depth_meta.get("blueprint") == "sensor.camera.depth"
                and depth_meta.get("encoding") == DEPTH_ENCODING
                and depth_meta.get("depth_tolerance_m") == DEPTH_TOLERANCE_M
                and depth_meta.get("algorithm_version") == ALGORITHM_VERSION
                and (self.output_dir / "resolved_config.json").is_file()
            ),
        }

        tier_counts = Counter((str(row["label"]), str(row["visibility_tier"]),
                               _distance_bin(float(row["gt_distance_m"])))
                              for row in self.visibility_rows)
        geometry_person = [row for row in self.visibility_rows
                           if row["label"] == "person" and bool(row["geometry_qualified_v2"])]
        person_marginal = sum(row["label"] == "person" and row["visibility_tier"] == TIER_MARGINAL
                              for row in self.visibility_rows)
        person_unobservable = sum(row["label"] == "person" and row["visibility_tier"] == TIER_UNOBSERVABLE
                                  for row in self.visibility_rows)
        visibility_exercised = (
            person_marginal >= MIN_EXERCISED_PERSON_ROWS
            and person_unobservable >= MIN_EXERCISED_PERSON_ROWS)

        review: dict[str, Any] = {"written": False}
        smoke_mode = self.split == "smoke"
        technical_pass_before_review = all(v2_gates.values()) and all(v3_gates.values())
        if technical_pass_before_review and smoke_mode:
            try:
                review = {"written": True, **_write_review_artifacts(self)}
            except Exception as exc:  # noqa: BLE001
                review = {"written": False, "error": f"{type(exc).__name__}: {exc}"}
        elif technical_pass_before_review:
            review = {
                "written": False,
                "not_required_for_canonical_episode": True,
                "accepted_smoke_review": (
                    "data_collection/experiments/route_b_perception_v3/"
                    "20260827_103139_traffic_30_30_smoke/"
                    "ROUTE_B_V3_30_30_SMOKE_REPORT.md"
                ),
            }
        if smoke_mode:
            v3_gates["manual_review_artifacts_written"] = bool(review.get("written"))
        technical_pass = all(v2_gates.values()) and all(v3_gates.values())
        if not technical_pass:
            terminal = "ROUTE_B_V3_COLLECTION_FAILED"
        elif not smoke_mode:
            terminal = "ROUTE_B_V3_CANONICAL_EPISODE_PASSED"
        elif not visibility_exercised:
            terminal = "ROUTE_B_V3_VISIBILITY_NOT_EXERCISED"
        else:
            terminal = "ROUTE_B_V3_30_30_READY_FOR_MANUAL_REVIEW"

        depth_sizes = [int(row["depth_bytes"]) for row in self.depth_frame_rows]
        mask_pixels = [int(row["person_mask_pixels"]) for row in self.sample_stats]
        baseline_core = [max(0.0, float(total) - float(decode))
                         for total, decode in zip(self.prepare_wall_s, self.depth_decode_wall_s)]
        summary.update({
            "schema": "scenesense_moving_ego_fusion_training_data.v3.route_summary",
            "terminal": terminal,
            "v2_gates": v2_gates,
            "v3_gates": v3_gates,
            "gates": {**v2_gates, **v3_gates},
            "status": "COLLECTION_EPISODE_PASSED" if technical_pass else "COLLECTION_EPISODE_FAILED",
            "depth_visibility": {
                "algorithm_version": ALGORITHM_VERSION,
                "depth_tolerance_m": DEPTH_TOLERANCE_M,
                "encoding": DEPTH_ENCODING,
                "visibility_rows": len(self.visibility_rows),
                "object_rows_reconciled": self.visibility_object_rows_total,
                "tier_counts_by_class_and_distance": {
                    f"{label}:{tier}:{distance}": count
                    for (label, tier, distance), count in sorted(tier_counts.items())
                },
                "geometry_qualified_person_rows": len(geometry_person),
                "geometry_qualified_person_retained_v010": sum(
                    bool(row["eligible_visible_v010"]) for row in geometry_person),
                "geometry_qualified_person_retained_v025": sum(
                    bool(row["eligible_clear_v025"]) for row in geometry_person),
                "geometry_qualified_person_retained_v010_percent": (
                    100.0 * sum(bool(row["eligible_visible_v010"]) for row in geometry_person)
                    / max(1, len(geometry_person))),
                "geometry_qualified_person_retained_v025_percent": (
                    100.0 * sum(bool(row["eligible_clear_v025"]) for row in geometry_person)
                    / max(1, len(geometry_person))),
                "person_marginal_rows": person_marginal,
                "person_unobservable_rows": person_unobservable,
                "visibility_exercised": visibility_exercised,
                "minimum_rows_per_nonclear_person_tier": MIN_EXERCISED_PERSON_ROWS,
                "person_mask_pixels_per_frame": _summary(mask_pixels),
                "person_mask_pixels_per_actor": _summary([
                    int(row["person_mask_painted_px"]) for row in self.visibility_rows
                    if row["label"] == "person"]),
                "alignment": {
                    "prepared_records": len(alignment), "saved_records": len(saved_alignment),
                    "maximum_timestamp_delta_s": max(
                        (float(row["timestamp_delta_s"]) for row in alignment), default=None),
                    "maximum_rgb_depth_transform_delta": max(
                        (float(row["rgb_depth_transform_delta"]) for row in alignment), default=None),
                },
            },
            "v3_runtime": {
                "v2_core_prepare_before_depth_estimate_s": _summary(baseline_core),
                "depth_decode_s": _summary(self.depth_decode_wall_s),
                "saved_frame_visibility_and_mask_s": _summary(self.visibility_wall_s),
                "v3_prepare_including_persistence_s": _summary(self.v3_prepare_wall_s),
            },
            "v3_storage": {
                "raw_depth_saved_bytes": self.depth_bytes,
                "raw_depth_saved_gib": self.depth_bytes / (1024 ** 3),
                "raw_depth_bytes_per_frame": _summary(depth_sizes),
                "projected_six_episode_depth_bytes": self.depth_bytes * 6,
                "projected_eight_episode_depth_bytes": self.depth_bytes * 8,
                "future_canonical_recommendation": (
                    "After v3 manual qualification, raw depth may be replaced by compact derived "
                    "visibility masks plus a bounded lossless raw-depth provenance sample; do not "
                    "apply that optimization to this retained smoke."
                ),
            },
            "manual_review": review,
            "resource_usage": {
                "status": "populated by the supervised launcher after CARLA shutdown",
                "peak_client_rss_kib": None, "peak_host_ram_used_kib": None,
                "peak_gpu_memory_used_mib": None,
            },
        })
        summary["storage"]["per_frame_bytes_note"] = (
            "v2 rgb/mask/semantic/radar payloads + lossless raw CARLA encoded-depth PNG"
        )
        self.parked.save_json(summary_path, summary)
        self._write_episode_report(summary)
        failed = sorted(name for name, passed in summary["gates"].items() if not passed)
        print(json.dumps({
            "terminal": terminal, "technical_pass": technical_pass,
            "visibility_exercised": visibility_exercised, "failed_gates": failed,
            "review_cases": review.get("actual_cases", 0),
        }, indent=2), flush=True)

    def _write_episode_report(self, summary: dict[str, Any]) -> None:
        route = summary.get("route_result") or {}
        visibility = summary["depth_visibility"]
        storage = summary["v3_storage"]
        runtime = summary["v3_runtime"]
        failed = sorted(name for name, value in summary["gates"].items() if not value)
        depth_mean = float(storage["raw_depth_bytes_per_frame"]["mean"] or 0.0)
        decode_mean = float(runtime["depth_decode_s"]["mean"] or 0.0)
        visibility_mean = float(runtime["saved_frame_visibility_and_mask_s"]["mean"] or 0.0)
        baseline_mean = float(runtime["v2_core_prepare_before_depth_estimate_s"]["mean"] or 0.0)
        report_title = (
            "Route B v3 30/30 smoke report" if self.split == "smoke"
            else "Route B v3 canonical episode report"
        )
        report = f"""# {report_title}

Terminal: `{summary['terminal']}`

- Density/split: `{self.density}` / `{self.split}`; target speed {self.target_speed_kph:.1f} km/h; hybrid physics `{self.hybrid_physics}`.
- Route: completed `{bool(route.get('completed'))}`; simulated {float(route.get('simulation_duration_s', 0.0)):.2f} s; client wall {float(route.get('wall_clock_duration_s', 0.0)):.2f} s.
- Frames: raw radar callbacks {summary['cadence']['raw_callbacks']}; prepared {summary['prepared_inputs']}; saved {summary['saved_samples']}.
- Exact synchronized depth: {visibility['alignment']['prepared_records']} prepared / {visibility['alignment']['saved_records']} saved; maximum timestamp delta {visibility['alignment']['maximum_timestamp_delta_s']} s.
- Visibility rows/object rows: {visibility['visibility_rows']}/{visibility['object_rows_reconciled']}.
- Geometry-qualified person retention: v0.10 {visibility['geometry_qualified_person_retained_v010']}/{visibility['geometry_qualified_person_rows']} ({visibility['geometry_qualified_person_retained_v010_percent']:.2f}%); v0.25 {visibility['geometry_qualified_person_retained_v025']}/{visibility['geometry_qualified_person_rows']} ({visibility['geometry_qualified_person_retained_v025_percent']:.2f}%).
- Person mask pixels/frame: median {visibility['person_mask_pixels_per_frame']['p50']}, p90 {visibility['person_mask_pixels_per_frame']['p90']}; per actor median {visibility['person_mask_pixels_per_actor']['p50']}.
- Raw lossless depth: {storage['raw_depth_saved_bytes']} bytes total; mean {depth_mean:.1f} bytes/frame; projected six/eight episodes {storage['projected_six_episode_depth_bytes']}/{storage['projected_eight_episode_depth_bytes']} bytes.
- CPU depth decode mean {decode_mean:.6f} s; saved-frame visibility/mask mean {visibility_mean:.6f} s; estimated pre-depth core preparation mean {baseline_mean:.6f} s.
- Manual review cases: {summary['manual_review'].get('actual_cases', 0)}; marginal person rows {visibility['person_marginal_rows']}; unobservable person rows {visibility['person_unobservable_rows']}.
- Failed gates: {failed or 'none'}.
- Sensor cleanup: `{summary['sensor_cleanup_succeeded']}`. CARLA process shutdown and peak RAM/VRAM are appended by the supervised launcher.

Vehicle masks preserve CARLA semantic pixels. Person masks are depth-derived visible-region approximations, not guaranteed anatomical silhouettes. No semantic walker tags, filled-box fallback, or ellipse fallback are used.

Future canonical storage recommendation: {storage['future_canonical_recommendation']}
"""
        report_name = (
            "ROUTE_B_V3_30_30_SMOKE_REPORT.md"
            if self.split == "smoke" else "ROUTE_B_V3_EPISODE_REPORT.md"
        )
        (self.output_dir / report_name).write_text(
            report, encoding="utf-8")


def build_parser():
    """Expose the inherited v2 CLI unchanged so density arguments forward exactly."""
    parser = v2.build_parser()
    parser.description = __doc__
    return parser


def main(argv: list[str] | None = None) -> int:
    # Parse once here to admit only the reviewed smoke or one of the eight
    # registered canonical episode tuples before v2 can connect or create data.
    preview = build_parser().parse_args(argv)
    requested = (
        str(preview.split), str(preview.density),
        int(preview.scenario_seed), int(preview.tm_seed),
    )
    smoke_request = requested == ("smoke", "traffic_30_30", 101, 1101)
    canonical_request = requested in REGISTERED_EPISODE_KEYS
    if not smoke_request and not canonical_request:
        print("v3 collector request is not the reviewed smoke or a registered canonical tuple",
              file=sys.stderr)
        return 2
    if preview.hybrid_physics:
        print("v3 bounded collector requires --no-hybrid-physics", file=sys.stderr)
        return 2
    if float(preview.target_speed_kph) != 25.0:
        print("v3 bounded collector requires target speed 25 km/h", file=sys.stderr)
        return 2
    if str(preview.rasterizer) != "fast" or float(preview.replenish_interval_s) != 2.0:
        print("v3 bounded collector requires fast rasterizer and 2.0 s replenish interval", file=sys.stderr)
        return 2
    if float(preview.maximum_loop_sim_s) != 600.0 or not preview.allow_roadblock_clearing:
        print("v3 bounded collector requires 600 s maximum and roadblock clearing", file=sys.stderr)
        return 2

    original = v2.PerceptionCollectorV2
    original_allowed_seeds = set(v2.ALLOWED_SEED_BUNDLES)
    v2.ALLOWED_SEED_BUNDLES.update((key[2], key[3]) for key in REGISTERED_EPISODE_KEYS)
    v2.PerceptionCollectorV2 = PerceptionCollectorV3
    try:
        code = v2.main(argv)
    finally:
        v2.PerceptionCollectorV2 = original
        v2.ALLOWED_SEED_BUNDLES.clear()
        v2.ALLOWED_SEED_BUNDLES.update(original_allowed_seeds)
    summary_path = Path(preview.output_dir).resolve() / "route_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(json.dumps({"v3_terminal": summary.get("terminal"),
                          "summary": str(summary_path)}, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
