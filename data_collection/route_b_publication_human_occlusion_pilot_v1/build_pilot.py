#!/usr/bin/env python3
"""Build a create-only, prediction-blind 100-person RGB annotation pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_SOURCE_ROOT = (
    ROOT / "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/views/val"
)
DEFAULT_SEED = 20260831
EXPECTED_SAMPLE_COUNT = 100
DISTANCE_BANDS = (
    ("00_10m", 0.0, 10.0),
    ("10_20m", 10.0, 20.0),
    ("20_30m", 20.0, 30.0),
    ("30_40m", 30.0, 40.0),
)
ANNOTATION_FIELDS = ("sample_id", "visibility_label", "truncation_label", "notes")
MANIFEST_FIELDS = (
    "selection_ordinal",
    "selection_seed",
    "sample_id",
    "gt_actor_id",
    "frame_id",
    "episode_id",
    "distance_m",
    "distance_band",
    "selection_group",
    "proxy_c_stratum",
    "proxy_s_stratum",
    "selection_stratum",
    "projected_bbox_x",
    "projected_bbox_y",
    "projected_bbox_w",
    "projected_bbox_h",
    "rgb_source_path",
    "depth_source_path",
    "rgb_frame_id",
    "depth_frame_id",
    "rgb_timestamp_s",
    "depth_timestamp_s",
    "rgb_depth_timestamp_delta_s",
    "projected_box_line_of_sight_clearance_C",
    "depth_order_surface_ratio_S",
    "depth_order_surface_ratio_zero_denominator",
    "actor_depth_consistent_pixels",
    "closer_depth_pixels",
    "sampled_projected_box_pixels",
    "source_manifest_sha256",
    "source_object_boxes_sha256",
    "source_depth_diagnostics_sha256",
)
PANEL_INDEX_FIELDS = (
    "selection_ordinal",
    "sample_id",
    "panel_path",
    "panel_sha256",
    "contact_sheet_row",
    "contact_sheet_column",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_x(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def write_bytes_x(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def write_json_x(path: Path, value: Any) -> None:
    write_text_x(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite(value: str, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {field}: {value!r}")
    return result


def distance_band(distance_m: float) -> str:
    for index, (name, lower, upper) in enumerate(DISTANCE_BANDS):
        if lower <= distance_m < upper or (index == len(DISTANCE_BANDS) - 1 and distance_m == upper):
            return name
    raise RuntimeError(f"distance outside pilot range: {distance_m}")


def proxy_stratum(value: float | None) -> str:
    if value is None:
        return "zero_denominator"
    if value < 0.25:
        return "q0_000_025"
    if value < 0.50:
        return "q1_025_050"
    if value < 0.75:
        return "q2_050_075"
    return "q3_075_100"


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode("utf-8")).hexdigest()


def load_population(source_root: Path = DEFAULT_SOURCE_ROOT) -> tuple[list[dict[str, Any]], dict[str, str]]:
    source_root = source_root.resolve(strict=True)
    manifest_path = source_root / "manifest.csv"
    boxes_path = source_root / "object_boxes_all.csv"
    diagnostics_path = source_root / "object_visibility_all.csv"
    source_hashes = {
        "source_manifest_sha256": sha256(manifest_path),
        "source_object_boxes_sha256": sha256(boxes_path),
        "source_depth_diagnostics_sha256": sha256(diagnostics_path),
    }
    manifest_rows = read_csv(manifest_path)
    if len(manifest_rows) != 3345 or {row.get("split") for row in manifest_rows} != {"val"}:
        raise RuntimeError("frozen validation manifest count/split drift")
    manifests = {row["sample_id"]: row for row in manifest_rows}
    if len(manifests) != len(manifest_rows):
        raise RuntimeError("duplicate validation sample IDs")
    episodes = tuple(dict.fromkeys(row["experiment_id"] for row in manifest_rows))
    if len(episodes) != 2:
        raise RuntimeError(f"expected two validation episodes, got {episodes}")

    box_rows = read_csv(boxes_path)
    boxes: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in box_rows:
        key = (row["experiment_id"], row["sample_id"], row["gt_actor_id"], row["label"])
        if key in boxes:
            raise RuntimeError(f"duplicate GT actor-frame key: {key}")
        boxes[key] = row

    population: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for diagnostic in read_csv(diagnostics_path):
        if diagnostic["label"] != "person":
            continue
        sample_id = diagnostic["sample_id"]
        frame = manifests.get(sample_id)
        if frame is None or frame["experiment_id"] != diagnostic["experiment_id"]:
            raise RuntimeError(f"diagnostic/manifest mismatch: {sample_id}")
        key = (
            diagnostic["experiment_id"],
            sample_id,
            diagnostic["gt_actor_id"],
            diagnostic["label"],
        )
        box = boxes.get(key)
        if box is None:
            raise RuntimeError(f"diagnostic/GT mismatch: {key}")
        distance = finite(diagnostic["gt_distance_m"], "gt_distance_m")
        x = finite(diagnostic["clipped_bbox_x"], "clipped_bbox_x")
        y = finite(diagnostic["clipped_bbox_y"], "clipped_bbox_y")
        width = finite(diagnostic["clipped_bbox_w"], "clipped_bbox_w")
        height = finite(diagnostic["clipped_bbox_h"], "clipped_bbox_h")
        camera_width = int(frame["camera_width"])
        camera_height = int(frame["camera_height"])
        valid_intersection = (
            width > 0.0
            and height > 0.0
            and x < camera_width
            and y < camera_height
            and x + width > 0.0
            and y + height > 0.0
        )
        if not (0.0 <= distance <= 40.0 and valid_intersection):
            continue
        frame_id = str(frame["frame_id"])
        if str(box["frame_id"]) != frame_id or str(diagnostic["frame_id"]) != frame_id:
            raise RuntimeError(f"RGB/GT frame mismatch: {key}")
        if str(diagnostic["depth_frame_id"]) != frame_id:
            raise RuntimeError(f"RGB/depth frame mismatch: {key}")
        rgb_timestamp = finite(frame["timestamp"], "rgb_timestamp")
        depth_timestamp = finite(diagnostic["depth_timestamp_s"], "depth_timestamp_s")
        diagnostic_timestamp = finite(diagnostic["timestamp"], "diagnostic_timestamp")
        timestamp_delta = abs(rgb_timestamp - depth_timestamp)
        if timestamp_delta > 1e-6 or abs(rgb_timestamp - diagnostic_timestamp) > 1e-6:
            raise RuntimeError(f"RGB/depth timestamp mismatch: {key}")
        rgb_path = source_root / frame["rgb_path"]
        depth_path = source_root / diagnostic["depth_path"]
        if not rgb_path.is_file() or not depth_path.is_file():
            raise RuntimeError(f"missing synchronized source payload: {key}")

        roi_pixels = int(diagnostic["sampled_roi_px"])
        consistent_pixels = int(diagnostic["native_visible_px"])
        closer_fraction = finite(diagnostic["occluder_closer_fraction"], "closer_depth_fraction")
        if roi_pixels <= 0 or not 0.0 <= closer_fraction <= 1.0:
            raise RuntimeError(f"invalid depth-count metadata: {key}")
        closer_pixels = int(round(closer_fraction * roi_pixels))
        if abs(closer_fraction - closer_pixels / roi_pixels) > 1e-12:
            raise RuntimeError(f"closer-depth fraction does not reconstruct an integer count: {key}")
        clearance = 1.0 - closer_fraction
        denominator = consistent_pixels + closer_pixels
        surface_ratio = consistent_pixels / denominator if denominator > 0 else None
        actor_frame_key = (sample_id, diagnostic["gt_actor_id"])
        if actor_frame_key in seen:
            raise RuntimeError(f"duplicate eligible actor-frame: {actor_frame_key}")
        seen.add(actor_frame_key)
        band = distance_band(distance)
        c_stratum = proxy_stratum(clearance)
        s_stratum = proxy_stratum(surface_ratio)
        population.append({
            "sample_id": sample_id,
            "gt_actor_id": diagnostic["gt_actor_id"],
            "frame_id": frame_id,
            "episode_id": diagnostic["experiment_id"],
            "distance_m": distance,
            "distance_band": band,
            "selection_group": diagnostic["experiment_id"] + "|" + band,
            "proxy_c_stratum": c_stratum,
            "proxy_s_stratum": s_stratum,
            "selection_stratum": c_stratum + "|" + s_stratum,
            "projected_bbox_x": x,
            "projected_bbox_y": y,
            "projected_bbox_w": width,
            "projected_bbox_h": height,
            "rgb_source_path": str(Path(frame["rgb_path"])),
            "depth_source_path": str(Path(diagnostic["depth_path"])),
            "rgb_frame_id": frame_id,
            "depth_frame_id": diagnostic["depth_frame_id"],
            "rgb_timestamp_s": rgb_timestamp,
            "depth_timestamp_s": depth_timestamp,
            "rgb_depth_timestamp_delta_s": timestamp_delta,
            "projected_box_line_of_sight_clearance_C": clearance,
            "depth_order_surface_ratio_S": surface_ratio,
            "depth_order_surface_ratio_zero_denominator": denominator == 0,
            "actor_depth_consistent_pixels": consistent_pixels,
            "closer_depth_pixels": closer_pixels,
            "sampled_projected_box_pixels": roi_pixels,
            "_rgb_path": rgb_path,
            **source_hashes,
        })
    if not population:
        raise RuntimeError("eligible validation pedestrian population is empty")
    return population, source_hashes


def group_quotas(population: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], int]:
    episodes = sorted({str(row["episode_id"]) for row in population})
    if len(episodes) != 2:
        raise RuntimeError("pilot requires exactly two validation episodes")
    quotas: dict[tuple[str, str], int] = {}
    band_names = [band[0] for band in DISTANCE_BANDS]
    for episode in episodes:
        for band_index, band in enumerate(band_names):
            quotas[(episode, band)] = 13 if band_index < 2 else 12
    if sum(quotas.values()) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("internal quota error")
    return quotas


def select_examples(
    population: Sequence[Mapping[str, Any]], seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    quotas = group_quotas(population)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in population:
        grouped[(str(row["episode_id"]), str(row["distance_band"]))].append(row)
    selected: list[dict[str, Any]] = []
    used_frames: set[str] = set()
    for group in sorted(quotas):
        quota = quotas[group]
        candidates = grouped.get(group, [])
        unique_frames = {str(row["sample_id"]) for row in candidates}
        if len(unique_frames) < quota:
            raise RuntimeError(f"insufficient unique frames for {group}: {len(unique_frames)} < {quota}")
        by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_stratum[str(row["selection_stratum"])].append(row)
        stratum_order = sorted(
            by_stratum,
            key=lambda value: stable_key(seed, group[0], group[1], value),
        )
        queues: dict[str, list[Mapping[str, Any]]] = {}
        for stratum in stratum_order:
            queues[stratum] = sorted(
                by_stratum[stratum],
                key=lambda row: stable_key(
                    seed,
                    str(row["sample_id"]),
                    str(row["gt_actor_id"]),
                ),
            )
        group_selected = 0
        while group_selected < quota:
            progress = False
            for stratum in stratum_order:
                queue = queues[stratum]
                while queue and str(queue[0]["sample_id"]) in used_frames:
                    queue.pop(0)
                if not queue:
                    continue
                row = dict(queue.pop(0))
                used_frames.add(str(row["sample_id"]))
                selected.append(row)
                group_selected += 1
                progress = True
                if group_selected == quota:
                    break
            if not progress:
                raise RuntimeError(f"one-person-per-frame selection exhausted for {group}")
    if len(selected) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError(f"selected {len(selected)}, expected {EXPECTED_SAMPLE_COUNT}")
    if len({row["sample_id"] for row in selected}) != EXPECTED_SAMPLE_COUNT:
        raise RuntimeError("selected frames are not unique")
    selected.sort(key=lambda row: (
        str(row["episode_id"]),
        [band[0] for band in DISTANCE_BANDS].index(str(row["distance_band"])),
        stable_key(seed, str(row["sample_id"]), str(row["gt_actor_id"])),
    ))
    for ordinal, row in enumerate(selected, 1):
        row["selection_ordinal"] = ordinal
        row["selection_seed"] = seed
    return selected


def crop_box(image: np.ndarray, box: Sequence[float], scale: float) -> np.ndarray:
    x, y, width, height = map(float, box)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    crop_width = max(8.0, width * scale)
    crop_height = max(8.0, height * scale)
    x0 = max(0, int(math.floor(center_x - crop_width / 2.0)))
    y0 = max(0, int(math.floor(center_y - crop_height / 2.0)))
    x1 = min(image.shape[1], int(math.ceil(center_x + crop_width / 2.0)))
    y1 = min(image.shape[0], int(math.ceil(center_y + crop_height / 2.0)))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("empty RGB crop")
    return image[y0:y1, x0:x1].copy()


def fit_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.size == 0:
        raise RuntimeError("cannot fit an empty image")
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
    return canvas


def write_png_x(path: Path, image: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
        raise RuntimeError(f"failed writing {path}")


def make_panel(row: Mapping[str, Any]) -> np.ndarray:
    image = cv2.imread(str(row["_rgb_path"]), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise RuntimeError(f"invalid RGB frame: {row['_rgb_path']}")
    box = (
        float(row["projected_bbox_x"]),
        float(row["projected_bbox_y"]),
        float(row["projected_bbox_w"]),
        float(row["projected_bbox_h"]),
    )
    annotated = image.copy()
    x0 = max(0, min(image.shape[1] - 1, int(math.floor(box[0]))))
    y0 = max(0, min(image.shape[0] - 1, int(math.floor(box[1]))))
    x1 = max(0, min(image.shape[1] - 1, int(math.ceil(box[0] + box[2]))))
    y1 = max(0, min(image.shape[0] - 1, int(math.ceil(box[1] + box[3]))))
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 3, cv2.LINE_8)
    context = crop_box(image, box, 2.0)
    close = crop_box(image, box, 1.15)

    canvas = np.full((880, 1580, 3), 24, dtype=np.uint8)
    canvas[20:820, 20:1020] = fit_nearest(annotated, 1000, 800)
    canvas[20:405, 1040:1560] = fit_nearest(context, 520, 385)
    canvas[435:820, 1040:1560] = fit_nearest(close, 520, 385)
    sample_id = str(row["sample_id"])
    scale = 0.55
    while cv2.getTextSize(sample_id, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > 1530:
        scale *= 0.9
    cv2.putText(
        canvas,
        sample_id,
        (20, 860),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def annotation_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"sample_id": str(row["sample_id"]), "visibility_label": "", "truncation_label": "", "notes": ""}
        for row in selected
    ]


def manifest_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in selected:
        item = {field: row.get(field, "") for field in MANIFEST_FIELDS}
        if row["depth_order_surface_ratio_S"] is None:
            item["depth_order_surface_ratio_S"] = ""
        output.append(item)
    return output


def png_has_text_chunks(path: Path) -> bool:
    with path.open("rb") as stream:
        if stream.read(8) != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"invalid PNG signature: {path}")
        while True:
            raw_length = stream.read(4)
            if len(raw_length) != 4:
                raise RuntimeError(f"truncated PNG: {path}")
            length = int.from_bytes(raw_length, "big")
            chunk = stream.read(4)
            if chunk in {b"tEXt", b"zTXt", b"iTXt"}:
                return True
            stream.seek(length + 4, os.SEEK_CUR)
            if chunk == b"IEND":
                return False


def validate_annotator_assets(output: Path) -> None:
    forbidden = (
        "projected_box_line_of_sight_clearance_C",
        "depth_order_surface_ratio_S",
        "occluder_closer_fraction",
        "visible_fraction",
        "distance_m",
    )
    for name in ("ANNOTATION_RUBRIC.md", "annotator_A.csv", "annotator_B.csv", "panel_index.csv"):
        text = (output / name).read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            raise RuntimeError(f"hidden diagnostic leaked into annotator asset: {name}")
    png_paths = sorted((output / "panels").glob("*.png")) + [output / "panel_contact_sheet.png"]
    if len(png_paths) != EXPECTED_SAMPLE_COUNT + 1:
        raise RuntimeError("panel/contact-sheet count drift")
    if any(png_has_text_chunks(path) for path in png_paths):
        raise RuntimeError("PNG textual metadata is forbidden in annotator assets")


def make_contact_sheet(output: Path, selected: Sequence[Mapping[str, Any]]) -> None:
    columns = 4
    rows = math.ceil(len(selected) / columns)
    cell_width, cell_height = 400, 250
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 24, dtype=np.uint8)
    for index, row in enumerate(selected):
        panel = cv2.imread(str(output / "panels" / f"{row['sample_id']}.png"), cv2.IMREAD_COLOR)
        if panel is None:
            raise RuntimeError(f"missing panel for contact sheet: {row['sample_id']}")
        thumbnail = fit_nearest(panel, 380, 214)
        grid_row, column = divmod(index, columns)
        y0, x0 = grid_row * cell_height, column * cell_width
        sheet[y0:y0 + 214, x0 + 10:x0 + 390] = thumbnail
        text = str(row["sample_id"])
        scale = 0.30
        while cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > 380:
            scale *= 0.9
        cv2.putText(sheet, text, (x0 + 10, y0 + 238), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (235, 235, 235), 1, cv2.LINE_AA)
    write_png_x(output / "panel_contact_sheet.png", sheet)


def build(output: Path, seed: int, source_root: Path = DEFAULT_SOURCE_ROOT) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite create-only run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name("." + output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    started = time.monotonic()
    staging.mkdir()
    try:
        population, source_hashes = load_population(source_root)
        selected = select_examples(population, seed)
        write_csv_x(staging / "sample_manifest.csv", MANIFEST_FIELDS, manifest_rows(selected))
        write_csv_x(staging / "annotator_A.csv", ANNOTATION_FIELDS, annotation_rows(selected))
        write_csv_x(staging / "annotator_B.csv", ANNOTATION_FIELDS, annotation_rows(selected))
        write_bytes_x(staging / "ANNOTATION_RUBRIC.md", (PACKAGE_ROOT / "ANNOTATION_RUBRIC.md").read_bytes())
        write_bytes_x(staging / "score_agreement.py", (PACKAGE_ROOT / "score_agreement.py").read_bytes())

        panel_rows: list[dict[str, Any]] = []
        for index, row in enumerate(selected):
            panel_path = staging / "panels" / f"{row['sample_id']}.png"
            write_png_x(panel_path, make_panel(row))
            grid_row, column = divmod(index, 4)
            panel_rows.append({
                "selection_ordinal": row["selection_ordinal"],
                "sample_id": row["sample_id"],
                "panel_path": str(Path("panels") / panel_path.name),
                "panel_sha256": sha256(panel_path),
                "contact_sheet_row": grid_row + 1,
                "contact_sheet_column": column + 1,
            })
        write_csv_x(staging / "panel_index.csv", PANEL_INDEX_FIELDS, panel_rows)
        make_contact_sheet(staging, selected)
        validate_annotator_assets(staging)

        episode_counts = Counter(str(row["episode_id"]) for row in selected)
        distance_counts = Counter(str(row["distance_band"]) for row in selected)
        group_counts = Counter(str(row["selection_group"]) for row in selected)
        stratum_coverage = {
            group: len({str(row["selection_stratum"]) for row in selected if row["selection_group"] == group})
            for group in sorted(group_counts)
        }
        artifacts = {
            "sample_manifest_sha256": sha256(staging / "sample_manifest.csv"),
            "rubric_sha256": sha256(staging / "ANNOTATION_RUBRIC.md"),
            "annotator_A_template_sha256": sha256(staging / "annotator_A.csv"),
            "annotator_B_template_sha256": sha256(staging / "annotator_B.csv"),
            "panel_index_sha256": sha256(staging / "panel_index.csv"),
            "panel_contact_sheet_sha256": sha256(staging / "panel_contact_sheet.png"),
            "agreement_utility_sha256": sha256(staging / "score_agreement.py"),
        }
        result = {
            "schema": "route_b_publication_human_occlusion_pilot_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "terminal": "HUMAN_OCCLUSION_PILOT_READY_FOR_TWO_INDEPENDENT_ANNOTATORS",
            "selection_seed": seed,
            "source_root": str(source_root.resolve(strict=True)),
            "source_hashes": source_hashes,
            "eligible_population_count": len(population),
            "selected_sample_count": len(selected),
            "selected_unique_frames": len({row["sample_id"] for row in selected}),
            "selected_unique_actor_frames": len({(row["sample_id"], row["gt_actor_id"]) for row in selected}),
            "allocation_by_episode": dict(sorted(episode_counts.items())),
            "allocation_by_distance_band": dict(sorted(distance_counts.items())),
            "allocation_by_episode_and_distance": dict(sorted(group_counts.items())),
            "proxy_stratum_coverage_by_group": stratum_coverage,
            "selection_uses_existing_occupancy_threshold": False,
            "annotation_material_contains_hidden_diagnostics": False,
            "model_or_service_artifacts_read": 0,
            "test_rows_read": 0,
            "depth_images_opened": 0,
            "panel_count": len(panel_rows),
            "artifacts": artifacts,
            "wall_seconds": time.monotonic() - started,
        }
        write_json_x(staging / "RUN_METADATA.json", result)
        write_text_x(staging / "PILOT_READY", result["terminal"] + "\n")
        (staging / "sample_manifest.csv").chmod(0o444)
        staging.rename(output)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    build(args.output, args.seed, args.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
