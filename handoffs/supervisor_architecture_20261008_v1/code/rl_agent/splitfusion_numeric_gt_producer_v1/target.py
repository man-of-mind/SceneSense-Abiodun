"""Numerical v010 target for one frame: masks and ordered rows, no PNG/CSV.

This is the per-frame body of the pinned ``build_contract`` with the original
helper functions (``roi_mask``, ``source_state``, ``actor_positive_row``,
``static_positive_row``, ``paint_scaled_box``, ``finite_number``) and the
original gates. Removed from readiness: image file reads, a second depth decode
and plausibility check on the identical array, per-box recomputation of the
full-image vehicle mask, PNG encoding + file hashing, CSV writing + re-parsing
and PNG decoding of the just-written masks. Those are reconstructed exactly by
``artifacts`` when evidence is produced.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A

from . import originals as O

TABLES = ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv")


@dataclass(frozen=True)
class NumericTarget:
    """Python-valued (pre-CSV) outputs of the original per-frame builder body."""
    segmentation: np.ndarray
    object_ignore: np.ndarray
    positive_rows: tuple
    ignore_rows: tuple
    object_record_counts: Mapping
    ignore_record_counts: Mapping
    pixel_counts: Mapping[str, int]
    object_cell_counts: Mapping[str, int]
    seg_rel: str
    obj_rel: str


def numeric_target(*, frame: Mapping[str, Any], split: str, depth: np.ndarray, semantic: np.ndarray,
                   original_mask: np.ndarray, boxes: Sequence[Mapping[str, Any]], visibility: Sequence[Mapping[str, Any]],
                   static_rows: Sequence[Mapping[str, Any]], quarantine: Mapping[str, Any],
                   vehicle_mask: np.ndarray | None = None, contract: str = "v010") -> NumericTarget:
    bound = O.bound()
    builder, recon = bound.builder, bound.recon
    VEHICLE_TAGS, MODEL_SIZE = builder.VEHICLE_TAGS, builder.MODEL_SIZE
    sample_id = frame["sample_id"]
    source_w, source_h = int(frame["camera_width"]), int(frame["camera_height"])
    if semantic.shape != (source_h, source_w) or original_mask.shape != (source_h, source_w):
        raise RuntimeError(f"target shape mismatch: {sample_id}")
    if depth.shape != (source_h, source_w):
        raise RuntimeError(f"decoded depth shape mismatch: {sample_id}")
    if vehicle_mask is None:
        vehicle_mask = np.isin(semantic, list(VEHICLE_TAGS))
    elif vehicle_mask.shape != semantic.shape or vehicle_mask.dtype != np.bool_:
        raise RuntimeError("hoisted vehicle mask must be a full-image boolean array")
    visibility_by_key = {(row["sample_id"], row["gt_actor_id"], row["label"]): row for row in visibility}
    positive_rows, ignore_rows = [], []
    object_record_counts, ignore_record_counts = Counter(), Counter()
    duplicate_positive_keys = set()
    positive_native = np.zeros((source_h, source_w), dtype=np.uint8)
    ignore_native = np.zeros((source_h, source_w), dtype=np.uint8)
    object_ignore = np.zeros((MODEL_SIZE[1], MODEL_SIZE[0]), dtype=np.uint8)
    frame_positive_centers, all_source_masks = [], []

    for box_row in boxes:
        if box_row["sample_id"] != sample_id:
            raise RuntimeError("foreign sample in actor boxes")
        key = (sample_id, box_row["gt_actor_id"], box_row["label"])
        vis = visibility_by_key[key]
        record = recon.roi_mask(depth, vis, source_w, source_h)
        all_source_masks.append(record)
        x0, y0, x1, y1, visible = record
        if box_row["label"] == "vehicle":
            support = int(np.count_nonzero(visible & vehicle_mask[y0:y1, x0:x1]))
        else:
            support = int(np.count_nonzero(visible & (original_mask[y0:y1, x0:x1] == 2)))
        identity = f"{box_row['experiment_id']}:actor:{box_row['gt_actor_id']}"
        state, reason = builder.source_state(contract=contract, split=split, sample_id=sample_id, source_kind="actor",
            source_identity=identity, class_name=box_row["label"], visibility=vis, semantic_support=support,
            quarantined=(sample_id, identity) in quarantine["dynamic_actor_keys"])
        box_xywh = builder.actor_bbox(vis)
        class_pixels = vehicle_mask[y0:y1, x0:x1] if box_row["label"] == "vehicle" else original_mask[y0:y1, x0:x1] == 2
        if state == "POSITIVE":
            positive_rows.append(builder.actor_positive_row(box_row, contract))
            object_record_counts[(box_row["label"], "actor")] += 1
            positive_native[y0:y1, x0:x1][visible & class_pixels] = 1 if box_row["label"] == "vehicle" else 2
            cx = int(round(float(box_row["gt_center_x"]) * MODEL_SIZE[0] / source_w))
            cy = int(round(float(box_row["gt_center_y"]) * MODEL_SIZE[1] / source_h))
            frame_positive_centers.append((cx, cy))
        else:
            ignore_rows.append({
                "contract": contract, "split": split, "experiment_id": frame["experiment_id"],
                "sample_id": sample_id, "frame_id": frame["frame_id"], "class_name": box_row["label"],
                "source_kind": "actor", "source_identity": identity,
                "bbox_x": box_xywh[0], "bbox_y": box_xywh[1], "bbox_w": box_xywh[2], "bbox_h": box_xywh[3],
                "reason": reason, "object_ignore": 1, "segmentation_ignore": 1,
                "source_record": f"actor:{box_row['gt_actor_id']}"})
            ignore_record_counts[(box_row["label"], "actor", reason)] += 1
            builder.paint_scaled_box(object_ignore, box_xywh, (source_w, source_h))
            ignore_native[y0:y1, x0:x1][visible & class_pixels] = 1

    for static in static_rows:
        if static["sample_id"] != sample_id:
            raise RuntimeError("foreign sample in static rows")
        static_id = static["environment_object_id"]
        if static_id in quarantine["confirmed_secondary"]:
            continue
        record = recon.roi_mask(depth, static, source_w, source_h)
        all_source_masks.append(record)
        x0, y0, x1, y1, visible = record
        support = int(static["semantic_vehicle_visible_px"])
        panel_quarantine = (sample_id, static_id) in quarantine["panel_static_keys"]
        state, reason = builder.source_state(contract=contract, split=split, sample_id=sample_id,
            source_kind="environment_static", source_identity=static_id, class_name="vehicle", visibility=static,
            semantic_support=support,
            quarantined=(static_id in quarantine["manual_static_ids"] or (sample_id, static_id) in quarantine["dynamic_static_keys"]),
            panel_quarantine=panel_quarantine)
        box_xywh = tuple(float(static[key]) for key in ("clipped_bbox_x", "clipped_bbox_y", "clipped_bbox_w", "clipped_bbox_h"))
        if state == "POSITIVE":
            positive_rows.append(builder.static_positive_row(static, contract))
            object_record_counts[("vehicle", "environment_static")] += 1
            positive_native[y0:y1, x0:x1][visible & vehicle_mask[y0:y1, x0:x1]] = 1
            cx = int(round((box_xywh[0] + box_xywh[2] / 2.0) * MODEL_SIZE[0] / source_w))
            cy = int(round((box_xywh[1] + box_xywh[3] / 2.0) * MODEL_SIZE[1] / source_h))
            frame_positive_centers.append((cx, cy))
            dup_key = (sample_id, static_id, contract)
            if dup_key in duplicate_positive_keys:
                raise RuntimeError(f"duplicate static positive: {dup_key}")
            duplicate_positive_keys.add(dup_key)
        else:
            ignore_rows.append({
                "contract": contract, "split": split, "experiment_id": frame["experiment_id"],
                "sample_id": sample_id, "frame_id": frame["frame_id"], "class_name": "vehicle",
                "source_kind": "environment_static", "source_identity": static_id,
                "bbox_x": box_xywh[0], "bbox_y": box_xywh[1], "bbox_w": box_xywh[2], "bbox_h": box_xywh[3],
                "reason": reason, "object_ignore": 1, "segmentation_ignore": 1,
                "source_record": f"environment_static:{static_id}"})
            ignore_record_counts[("vehicle", "environment_static", reason)] += 1
            builder.paint_scaled_box(object_ignore, box_xywh, (source_w, source_h))
            ignore_native[y0:y1, x0:x1][visible & vehicle_mask[y0:y1, x0:x1]] = 1

    vehicle_binary = vehicle_mask.astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(vehicle_binary, connectivity=8)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        component = labels == component_id
        explained = False
        for x0, y0, x1, y1, visible in all_source_masks:
            overlap = int(np.count_nonzero(component[y0:y1, x0:x1] & visible))
            if overlap >= 12 and overlap / area >= 0.10:
                explained = True
                break
        if explained:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT]); y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH]); h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        reason = "unresolved_semantic_vehicle_component"
        identity = f"{sample_id}:semantic_component:{component_id}"
        ignore_rows.append({
            "contract": contract, "split": split, "experiment_id": frame["experiment_id"],
            "sample_id": sample_id, "frame_id": frame["frame_id"], "class_name": "vehicle",
            "source_kind": "semantic_component", "source_identity": identity,
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "reason": reason, "object_ignore": 1, "segmentation_ignore": 1,
            "source_record": f"semantic_component:{component_id}"})
        ignore_record_counts[("vehicle", "semantic_component", reason)] += 1
        builder.paint_scaled_box(object_ignore, (x, y, w, h), (source_w, source_h))
        ignore_native[component] = 1

    target_native = np.zeros((source_h, source_w), dtype=np.uint8)
    target_native[positive_native == 1] = 1
    target_native[positive_native == 2] = 2
    target_native[(ignore_native > 0) & (positive_native == 0)] = 255
    target = cv2.resize(target_native, MODEL_SIZE, interpolation=cv2.INTER_NEAREST)
    for cx, cy in frame_positive_centers:
        if 0 <= cx < MODEL_SIZE[0] and 0 <= cy < MODEL_SIZE[1]:
            object_ignore[cy, cx] = 0
    counts = np.bincount(target.ravel(order="C"), minlength=256)
    pixel_counts = {"background": int(counts[0]), "vehicle_positive": int(counts[1]),
                    "person_positive": int(counts[2]), "ignore": int(counts[255])}
    ignored_cells = int(np.count_nonzero(object_ignore))
    object_cell_counts = {"ignore": ignored_cells * 2, "background": (MODEL_SIZE[0] * MODEL_SIZE[1] - ignored_cells) * 2}
    # ---- original post-loop gates (unchanged) ----
    for row in positive_rows:
        for field_name in builder.OBJECT_NUMERIC_FIELDS:
            builder.finite_number(row[field_name], f"{row['sample_id']}/{row['source_identity']}/{field_name}")
        if not row.get("contract_reason") or not row.get("source_kind") or not row.get("source_identity"):
            raise RuntimeError("positive provenance/reason gate failure")
    if any(not row.get("reason") or not row.get("source_kind") or not row.get("source_identity") for row in ignore_rows):
        raise RuntimeError("ignore provenance/reason gate failure")
    quarantined_identities = set(quarantine["manual_static_ids"])
    if any(row["source_identity"] in quarantined_identities for row in positive_rows):
        raise RuntimeError("positive overlaps globally quarantined static identity")
    for pair in quarantine["confirmed_pairs"]:
        preferred, duplicate = pair["preferred_environment_object_id"], pair["duplicate_environment_object_id"]
        by_frame = Counter(row["sample_id"] for row in positive_rows
                           if row["source_kind"] == "environment_static" and row["source_identity"] in {preferred, duplicate})
        if any(value > 1 for value in by_frame.values()):
            raise RuntimeError("confirmed static duplicate contributes more than one positive per frame")
    target.flags.writeable = False
    object_ignore.flags.writeable = False
    return NumericTarget(segmentation=target, object_ignore=object_ignore, positive_rows=tuple(positive_rows),
        ignore_rows=tuple(ignore_rows), object_record_counts=dict(object_record_counts),
        ignore_record_counts=dict(ignore_record_counts), pixel_counts=pixel_counts, object_cell_counts=object_cell_counts,
        seg_rel=str(Path("segmentation_masks") / f"{sample_id}.png"), obj_rel=str(Path("object_ignore_masks") / f"{sample_id}.png"))


def numeric_target_from_inputs(inputs: A.CompleteTargetInputs, *, depth: np.ndarray | None = None) -> NumericTarget:
    """Numeric target from already validated complete inputs (historical/synthetic fixtures)."""
    bound = O.bound()
    if depth is None:
        depth = bound.visibility.decode_depth_bgra(inputs.depth_bgra)
        if not bound.visibility.depth_is_plausible(depth):
            raise RuntimeError(f"implausible retained depth: {inputs.frame['sample_id']}")
    return numeric_target(frame=dict(inputs.frame), split=inputs.split, depth=depth, semantic=inputs.raw_semantic,
        original_mask=inputs.original_v3_mask, boxes=[dict(r) for r in inputs.actor_boxes],
        visibility=[dict(r) for r in inputs.actor_visibility], static_rows=[dict(r) for r in inputs.static_candidates],
        quarantine=inputs.quarantine)
