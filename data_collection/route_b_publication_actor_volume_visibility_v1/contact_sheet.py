"""Contact sheet overlay for the actor-volume visibility audit.

One panel per pilot sample showing the RGB view, the full projected box, the
retained actor-volume depth points, the derived visible box, the automatic score
and band, the human band, truncation, and a disagreement marker.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import core

PANEL_IMAGE_W, PANEL_IMAGE_H = 480, 400
PANEL_TEXT_H = 96
PANEL_W, PANEL_H = PANEL_IMAGE_W, PANEL_IMAGE_H + PANEL_TEXT_H
GRID_COLS = 10
CROP_MARGIN_FRACTION = 0.55
CROP_MIN_PX = 150

# BGR, matching OpenCV.
COLOUR_FULL_BOX = (0, 220, 255)      # amber: full clipped projected box
COLOUR_VISIBLE_BOX = (255, 220, 0)   # cyan: derived visible box
COLOUR_POINTS = (60, 230, 60)        # green: retained actor-volume points
COLOUR_AGREE = (90, 210, 90)
COLOUR_DISAGREE = (60, 60, 235)
COLOUR_AMBIGUOUS = (170, 170, 170)
COLOUR_TEXT = (245, 245, 245)
COLOUR_PANEL_BG = (28, 28, 28)

BAND_SHORT = {
    core.BAND_NOT_OBSERVABLE: "not-obs",
    core.BAND_HEAVY: "heavy",
    core.BAND_PARTIAL: "partial",
    core.BAND_BARE: "bare",
    "ambiguous": "ambiguous",
}


def _retained_mask(row: pd.Series, dataset_dir: Path) -> np.ndarray | None:
    """Re-derive the retained pixel mask for one row, for display only."""
    import cv2
    import json

    depth_raw = cv2.imread(str(dataset_dir / row.depth_source_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    values = depth_raw.astype(np.float32)
    normalized = (
        values[:, :, 2] + values[:, :, 1] * 256.0 + values[:, :, 0] * 256.0 * 256.0
    ) / (256.0**3 - 1.0)
    depth_m = (core.CARLA_MAX_DEPTH_M * normalized).astype(np.float64)

    intrinsics = np.asarray(
        [[row.fx, 0.0, row.cx], [0.0, row.fy, row.cy], [0.0, 0.0, 1.0]]
    )
    camera_matrix = np.asarray(json.loads(row.camera_matrix_json), dtype=np.float64)
    bounds = core.roi_pixel_bounds(
        {
            "clipped_bbox_x": row.clipped_bbox_x,
            "clipped_bbox_y": row.clipped_bbox_y,
            "clipped_bbox_w": row.clipped_bbox_w,
            "clipped_bbox_h": row.clipped_bbox_h,
        },
        width=depth_m.shape[1],
        height=depth_m.shape[0],
    )
    roi = core.back_project_roi(depth_m, bounds, camera_matrix, intrinsics)
    owned = core.assign_competing_pedestrians(
        roi["world"], str(row.gt_actor_id), row.pedestrian_boxes
    )["owned"]
    mask = np.zeros(depth_m.shape, dtype=bool)
    mask[roi["row"][owned], roi["col"][owned]] = True
    return mask


def _render_panel(row: pd.Series, dataset_dir: Path, index: int) -> np.ndarray:
    import cv2

    panel = np.full((PANEL_H, PANEL_W, 3), COLOUR_PANEL_BG, dtype=np.uint8)
    rgb = cv2.imread(str(dataset_dir / row.rgb_source_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return panel

    mask = _retained_mask(row, dataset_dir)
    canvas = rgb.copy()
    if mask is not None and mask.any():
        overlay = canvas.copy()
        overlay[mask] = COLOUR_POINTS
        canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0)

    fx0, fy0 = float(row.clipped_bbox_x), float(row.clipped_bbox_y)
    fx1, fy1 = fx0 + float(row.clipped_bbox_w), fy0 + float(row.clipped_bbox_h)
    cv2.rectangle(
        canvas, (int(round(fx0)), int(round(fy0))), (int(round(fx1)), int(round(fy1))),
        COLOUR_FULL_BOX, 2,
    )
    if not bool(row.no_support) and float(row.visible_bbox_w) > 0.0:
        vx0, vy0 = float(row.visible_bbox_x), float(row.visible_bbox_y)
        vx1, vy1 = vx0 + float(row.visible_bbox_w), vy0 + float(row.visible_bbox_h)
        cv2.rectangle(
            canvas, (int(round(vx0)), int(round(vy0))), (int(round(vx1)), int(round(vy1))),
            COLOUR_VISIBLE_BOX, 1,
        )

    # Crop around the target with margin, preserving aspect for the panel.
    height, width = canvas.shape[:2]
    box_w, box_h = fx1 - fx0, fy1 - fy0
    margin = max(CROP_MIN_PX, CROP_MARGIN_FRACTION * max(box_w, box_h))
    centre_x, centre_y = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
    half_w = 0.5 * box_w + margin
    half_h = 0.5 * box_h + margin
    aspect = PANEL_IMAGE_W / PANEL_IMAGE_H
    if half_w / half_h < aspect:
        half_w = half_h * aspect
    else:
        half_h = half_w / aspect
    x0 = int(max(0, min(width - 1, math.floor(centre_x - half_w))))
    x1 = int(max(x0 + 1, min(width, math.ceil(centre_x + half_w))))
    y0 = int(max(0, min(height - 1, math.floor(centre_y - half_h))))
    y1 = int(max(y0 + 1, min(height, math.ceil(centre_y + half_h))))
    crop = canvas[y0:y1, x0:x1]
    panel[:PANEL_IMAGE_H, :PANEL_IMAGE_W] = cv2.resize(
        crop, (PANEL_IMAGE_W, PANEL_IMAGE_H), interpolation=cv2.INTER_NEAREST
    )

    human = str(row.human_band)
    auto = str(row.auto_band)
    if human == "ambiguous":
        marker, colour = "AMBIG (excluded)", COLOUR_AMBIGUOUS
    elif human == auto:
        marker, colour = "AGREE", COLOUR_AGREE
    else:
        marker, colour = "DISAGREE", COLOUR_DISAGREE
    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_IMAGE_H - 1), colour, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"#{index:03d}  {row.distance_band}  {float(row.distance_m):.1f}m"
        + ("  NO_SUPPORT" if bool(row.no_support) else ""),
        f"auto {float(row.visibility):.3f} -> {BAND_SHORT[auto]}   human {BAND_SHORT[human]}",
        f"trunc {float(row.truncation):.3f} ({row.truncation_label})  pts {int(row.retained_actor_point_count)}",
        marker,
    ]
    for offset, text in enumerate(lines):
        cv2.putText(
            panel, text, (8, PANEL_IMAGE_H + 20 + offset * 20), font, 0.42,
            colour if offset == len(lines) - 1 else COLOUR_TEXT, 1, cv2.LINE_AA,
        )
    return panel


def build_contact_sheet(
    merged: pd.DataFrame, dataset_dir: Path, run_dir: Path, view_boxes_path: Path
) -> dict[str, Any]:
    """Render all panels plus one tiled contact sheet; returns a small summary."""
    import cv2

    # Attach the per-sample calibration and pedestrian candidate boxes needed to
    # re-derive the display mask.
    manifest = pd.read_csv(dataset_dir / "manifest.csv").set_index("sample_id")
    boxes = pd.read_csv(view_boxes_path, dtype={"gt_actor_id": str})
    people = {sid: group for sid, group in boxes[boxes.label == "person"].groupby("sample_id")}

    panels_dir = run_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    ordered = merged.sort_values("panel_number").reset_index(drop=True)
    rendered = _render_rows(ordered, manifest, people, dataset_dir, panels_dir)
    sheet = _tile(rendered, GRID_COLS)
    sheet_path = run_dir / "contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    rows = math.ceil(len(rendered) / GRID_COLS)

    return {
        "panels": int(len(rendered)),
        "grid": [rows, GRID_COLS],
        "panel_size_px": [PANEL_W, PANEL_H],
        "contact_sheet_px": [int(sheet.shape[1]), int(sheet.shape[0])],
        "panels_dir": "panels",
        "legend": {
            "amber_box": "full clipped projected box (B_full_clipped)",
            "green_overlay": "retained actor-volume depth points",
            "cyan_box": "derived visible box (B_visible)",
            "border": "green agree / red disagree / grey ambiguous-excluded",
        },
    }


def _render_rows(ordered, manifest, people, dataset_dir, panels_dir):
    """Enrich each row with its calibration and candidate boxes, then render."""
    import cv2

    rendered: list[np.ndarray] = []
    for _position, row in ordered.iterrows():
        meta = manifest.loc[row.sample_id]
        enriched = row.copy()
        enriched["fx"] = float(meta.camera_fx)
        enriched["fy"] = float(meta.camera_fy)
        enriched["cx"] = float(meta.camera_cx)
        enriched["cy"] = float(meta.camera_cy)
        enriched["camera_matrix_json"] = meta.camera_matrix_json
        enriched["pedestrian_boxes"] = [
            {
                "key": str(person.gt_actor_id),
                "centre": (person.object_world_x, person.object_world_y, person.object_world_z),
                "extent": (person.gt_extent_x_m, person.gt_extent_y_m, person.gt_extent_z_m),
                "yaw_deg": float(person.object_yaw_deg),
            }
            for person in people[row.sample_id].itertuples()
        ]
        panel = _render_panel(enriched, dataset_dir, int(row.panel_number))
        if panels_dir is not None:
            cv2.imwrite(str(panels_dir / f"panel_{int(row.panel_number):03d}.png"), panel)
        rendered.append(panel)
    return rendered


def _tile(rendered, columns):
    rows = math.ceil(len(rendered) / columns) if rendered else 0
    sheet = np.full(
        (max(1, rows) * PANEL_H, columns * PANEL_W, 3), COLOUR_PANEL_BG, dtype=np.uint8
    )
    for position, panel in enumerate(rendered):
        r, c = divmod(position, columns)
        sheet[r * PANEL_H : (r + 1) * PANEL_H, c * PANEL_W : (c + 1) * PANEL_W] = panel
    return sheet


def build_disagreement_sheet(
    rows: pd.DataFrame,
    dataset_dir: Path,
    output_path: Path,
    view_boxes_path: Path,
    *,
    columns: int = 5,
) -> dict[str, Any]:
    """Render an ordered subset (largest disagreements first) as one sheet."""
    import cv2

    manifest = pd.read_csv(dataset_dir / "manifest.csv").set_index("sample_id")
    boxes = pd.read_csv(view_boxes_path, dtype={"gt_actor_id": str})
    people = {sid: g for sid, g in boxes[boxes.label == "person"].groupby("sample_id")}
    rendered = _render_rows(rows.reset_index(drop=True), manifest, people, dataset_dir, None)
    sheet = _tile(rendered, columns)
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    return {
        "panels": int(len(rendered)),
        "grid": [math.ceil(len(rendered) / columns), columns],
        "image_px": [int(sheet.shape[1]), int(sheet.shape[0])],
    }
