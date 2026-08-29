"""Deterministic canonical-world and diagnostic image-space matching."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


MODEL_WIDTH, MODEL_HEIGHT = 768, 432
MATCH_DEFINITIONS = ("FULL_BOX_CENTER", "FULL_BOX_IOU_050", "FULL_BOX_IOU_030")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_frame_ids(dataset_root: Path) -> list[str]:
    return [row["sample_id"] for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "val"]


def load_person_gt(dataset_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, str]], set[tuple[str, str]]]:
    frame_geometry = {
        row["sample_id"]: (int(row["camera_width"]), int(row["camera_height"]))
        for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "val"
    }
    clear_keys = {
        (row["sample_id"], row["source_identity"])
        for row in read_csv(dataset_root / "contracts/v025/val/object_boxes.csv")
        if row["label"] == "person"
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict[str, str]] = {}
    stable = 0
    for row in read_csv(dataset_root / "contracts/v010/val/object_boxes.csv"):
        if row["label"] != "person":
            continue
        key = (row["sample_id"], row["source_identity"])
        if key in metadata:
            raise RuntimeError(f"duplicate sample/source GT metadata key: {key}")
        metadata[key] = row
        if row["sample_id"] not in frame_geometry:
            raise RuntimeError(f"GT sample absent from frozen validation manifest: {row['sample_id']}")
        source_w, source_h = frame_geometry[row["sample_id"]]
        sx, sy = MODEL_WIDTH / source_w, MODEL_HEIGHT / source_h
        x0 = float(row["gt_bbox_x"]) * sx
        y0 = float(row["gt_bbox_y"]) * sy
        x1 = (float(row["gt_bbox_x"]) + float(row["gt_bbox_w"])) * sx
        y1 = (float(row["gt_bbox_y"]) + float(row["gt_bbox_h"])) * sy
        grouped[row["sample_id"]].append({
            "stable_row": stable,
            "sample_id": row["sample_id"],
            "source_identity": row["source_identity"],
            "world_x": float(row["object_world_x"]),
            "world_y": float(row["object_world_y"]),
            "distance_m": float(row["gt_distance_m"]),
            "area_px": float(row["gt_bbox_area_px"]),
            "bbox": (x0, y0, x1, y1),
            "center": ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            "radar_supported": float(row.get("radar_support_points", "0") or 0) > 0,
            "clear_v025": key in clear_keys,
        })
        stable += 1
    return grouped, metadata, clear_keys


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stable, row in enumerate(read_csv(path)):
        item: dict[str, Any] = {
            "stable_row": stable,
            "sample_id": row["sample_id"],
            "class_name": row["class_name"],
            "score": float(row["score"]),
            "world_x": float(row["world_x"]),
            "world_y": float(row["world_y"]),
            "center": (float(row["center_x_px"]), float(row["center_y_px"])),
            "bbox": tuple(float(row[key]) for key in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")),
        }
        if not all(math.isfinite(value) for value in (
            item["score"], item["world_x"], item["world_y"], *item["center"], *item["bbox"]
        )):
            raise RuntimeError(f"nonfinite prediction at stable row {stable}")
        grouped[row["sample_id"]].append(item)
    # This reproduces the canonical scorer's frame-local score/class ordering.
    for items in grouped.values():
        items.sort(key=lambda item: (-item["score"], item["class_name"], item["stable_row"]))
    return grouped


def annotate_neutral_predictions(
    predictions: Mapping[str, Sequence[dict[str, Any]]], dataset_root: Path, frame_ids: Sequence[str]
) -> None:
    for sample_id in frame_ids:
        ignore_path = dataset_root / f"contracts/v010/val/object_ignore_masks/{sample_id}.png"
        ignore = cv2.imread(str(ignore_path), cv2.IMREAD_UNCHANGED)
        if ignore is None or ignore.shape != (MODEL_HEIGHT, MODEL_WIDTH):
            raise RuntimeError(f"invalid object ignore mask: {sample_id}")
        for prediction in predictions.get(sample_id, ()):
            cx, cy = prediction["center"]
            ix, iy = int(round(cx)), int(round(cy))
            prediction["neutral"] = bool(
                0 <= ix < MODEL_WIDTH and 0 <= iy < MODEL_HEIGHT and int(ignore[iy, ix]) != 0
            )


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-12, precision + recall),
    }


def canonical_world_match(
    frame_ids: Sequence[str], gt: Mapping[str, Sequence[dict[str, Any]]],
    predictions: Mapping[str, Sequence[dict[str, Any]]], threshold: float,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    unmatched_gt: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for sample_id in frame_ids:
        targets = list(gt.get(sample_id, ()))
        preds = [item for item in predictions.get(sample_id, ())
                 if item["class_name"] == "person" and item["score"] >= threshold]
        candidates: list[tuple[float, int, int]] = []
        for pi, prediction in enumerate(preds):
            for gi, target in enumerate(targets):
                distance = math.hypot(prediction["world_x"] - target["world_x"],
                                      prediction["world_y"] - target["world_y"])
                if distance <= 3.0:
                    candidates.append((distance, pi, gi))
        used_p: set[int] = set()
        used_g: set[int] = set()
        for distance, pi, gi in sorted(candidates):
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            matches.append({"sample_id": sample_id, "prediction": preds[pi], "gt": targets[gi],
                            "world_error_m": distance})
        for gi, target in enumerate(targets):
            if gi not in used_g:
                unmatched_gt.append(target)
        for pi, prediction in enumerate(preds):
            if pi in used_p:
                continue
            (ignored if prediction["neutral"] else unmatched_predictions).append(prediction)
    values = _metrics(len(matches), len(unmatched_predictions), len(unmatched_gt))
    if values["tp"] + values["fn"] != sum(len(gt.get(sample, ())) for sample in frame_ids):
        raise RuntimeError("canonical TP+FN denominator failure")
    return {
        **values,
        "eligible_gt": values["tp"] + values["fn"],
        "ignored_predictions": len(ignored),
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_predictions": unmatched_predictions,
        "ignored": ignored,
    }


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-12, area_a + area_b - intersection)


def _candidate_value(prediction: Mapping[str, Any], target: Mapping[str, Any], definition: str) -> float | None:
    if definition == "FULL_BOX_CENTER":
        x, y = prediction["center"]
        x0, y0, x1, y1 = target["bbox"]
        if not (x0 <= x < x1 and y0 <= y < y1):
            return None
        width, height = max(1e-12, x1 - x0), max(1e-12, y1 - y0)
        gx, gy = target["center"]
        return math.hypot((x - gx) / width, (y - gy) / height)
    iou = box_iou(prediction["bbox"], target["bbox"])
    floor = 0.50 if definition == "FULL_BOX_IOU_050" else 0.30
    return iou if iou >= floor else None


def image_match(
    frame_ids: Sequence[str], gt: Mapping[str, Sequence[dict[str, Any]]],
    predictions: Mapping[str, Sequence[dict[str, Any]]], threshold: float, definition: str,
) -> dict[str, Any]:
    if definition not in MATCH_DEFINITIONS:
        raise ValueError(definition)
    matches: list[dict[str, Any]] = []
    unmatched_gt: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    potential_by_gt: dict[int, set[int]] = defaultdict(set)
    potential_by_pred: dict[int, set[int]] = defaultdict(set)
    class_confusion_gt: set[int] = set()
    for sample_id in frame_ids:
        targets = list(gt.get(sample_id, ()))
        preds = [item for item in predictions.get(sample_id, ())
                 if item["class_name"] == "person" and item["score"] >= threshold]
        candidates: list[tuple[float, float, int, int, int, int, float]] = []
        for pi, prediction in enumerate(preds):
            for gi, target in enumerate(targets):
                value = _candidate_value(prediction, target, definition)
                if value is None:
                    continue
                primary = value if definition == "FULL_BOX_CENTER" else -value
                candidates.append((primary, -prediction["score"], prediction["stable_row"],
                                   target["stable_row"], pi, gi, value))
                potential_by_gt[target["stable_row"]].add(prediction["stable_row"])
                potential_by_pred[prediction["stable_row"]].add(target["stable_row"])
        # Diagnostic class-confusion count: a non-person prediction geometrically supports a person GT.
        other_preds = [item for item in predictions.get(sample_id, ())
                       if item["class_name"] != "person" and item["score"] >= threshold]
        for target in targets:
            if any(_candidate_value(item, target, definition) is not None for item in other_preds):
                class_confusion_gt.add(target["stable_row"])
        used_p: set[int] = set()
        used_g: set[int] = set()
        for _primary, _neg_score, _pr, _gr, pi, gi, value in sorted(candidates):
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            prediction, target = preds[pi], targets[gi]
            error = math.hypot(prediction["world_x"] - target["world_x"],
                               prediction["world_y"] - target["world_y"])
            matches.append({"sample_id": sample_id, "prediction": prediction, "gt": target,
                            "match_value": value, "world_error_m": error})
        for gi, target in enumerate(targets):
            if gi not in used_g:
                unmatched_gt.append(target)
        for pi, prediction in enumerate(preds):
            if pi in used_p:
                continue
            (ignored if prediction["neutral"] else unmatched_predictions).append(prediction)
    values = _metrics(len(matches), len(unmatched_predictions), len(unmatched_gt))
    return {
        **values,
        "eligible_gt": values["tp"] + values["fn"],
        "ignored_predictions": len(ignored),
        "matches": matches,
        "unmatched_gt": unmatched_gt,
        "unmatched_predictions": unmatched_predictions,
        "ignored": ignored,
        "potential_by_gt": potential_by_gt,
        "potential_by_pred": potential_by_pred,
        "class_confusion_gt_count": len(class_confusion_gt),
        "contended_gt_count": sum(len(values) > 1 for values in potential_by_gt.values()),
        "contended_prediction_count": sum(len(values) > 1 for values in potential_by_pred.values()),
    }


def assignment_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[int, int]:
    def gt_map(result: Mapping[str, Any]) -> dict[int, int]:
        return {pair["gt"]["stable_row"]: pair["prediction"]["stable_row"] for pair in result["matches"]}
    ma, mb = gt_map(a), gt_map(b)
    keys = set(ma) | set(mb)
    changed_gt = sum(ma.get(key) != mb.get(key) for key in keys)
    pairs_a, pairs_b = set(ma.items()), set(mb.items())
    return changed_gt, len(pairs_a ^ pairs_b)


def summarize_conditional(
    model: str, threshold: float, definition: str, result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pairs = list(result["matches"])
    subsets: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", pairs)]
    distance_edges = (0.0, 10.0, 20.0, 30.0, 40.000001)
    area_edges = (0.0, 400.0, 1600.0, 6400.0, 1.0e9)
    for left, right in zip(distance_edges[:-1], distance_edges[1:]):
        subsets.append(("distance_m", f"[{left:g},{right:g})",
                        [pair for pair in pairs if left <= pair["gt"]["distance_m"] < right]))
    for left, right in zip(area_edges[:-1], area_edges[1:]):
        subsets.append(("area_px", f"[{left:g},{right:g})",
                        [pair for pair in pairs if left <= pair["gt"]["area_px"] < right]))
    subsets += [
        ("visibility_contract", "clear_v025", [pair for pair in pairs if pair["gt"]["clear_v025"]]),
        ("visibility_contract", "primary_v010_only", [pair for pair in pairs if not pair["gt"]["clear_v025"]]),
        ("radar_support", "supported", [pair for pair in pairs if pair["gt"]["radar_supported"]]),
        ("radar_support", "unsupported", [pair for pair in pairs if not pair["gt"]["radar_supported"]]),
    ]
    rows: list[dict[str, Any]] = []
    for kind, label, subset in subsets:
        errors = np.asarray([pair["world_error_m"] for pair in subset], dtype=np.float64)
        row: dict[str, Any] = {
            "model": model, "threshold": threshold, "match_definition": definition,
            "subset_kind": kind, "subset_label": label, "matched_pairs": len(subset),
            "within_1m_fraction": float(np.mean(errors <= 1.0)) if len(errors) else "",
            "within_2m_fraction": float(np.mean(errors <= 2.0)) if len(errors) else "",
            "within_3m_fraction": float(np.mean(errors <= 3.0)) if len(errors) else "",
            "within_5m_fraction": float(np.mean(errors <= 5.0)) if len(errors) else "",
            "outside_3m_count": int(np.count_nonzero(errors > 3.0)),
            "mean_m": float(np.mean(errors)) if len(errors) else "",
            "median_m": float(np.median(errors)) if len(errors) else "",
            "p75_m": float(np.percentile(errors, 75)) if len(errors) else "",
            "p90_m": float(np.percentile(errors, 90)) if len(errors) else "",
            "p95_m": float(np.percentile(errors, 95)) if len(errors) else "",
            "class_confusion_gt_count": result["class_confusion_gt_count"] if kind == "overall" else "",
            "contended_gt_count": result["contended_gt_count"] if kind == "overall" else "",
            "contended_prediction_count": result["contended_prediction_count"] if kind == "overall" else "",
        }
        rows.append(row)
    return rows


def build_taxonomy(
    model: str, threshold: float, canonical: Mapping[str, Any], primary_2d: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_gt_to_pred = {pair["gt"]["stable_row"]: pair["prediction"]["stable_row"] for pair in canonical["matches"]}
    image_gt_to_pair = {pair["gt"]["stable_row"]: pair for pair in primary_2d["matches"]}
    canonical_fn_ids = {target["stable_row"] for target in canonical["unmatched_gt"]}
    all_gt = [pair["gt"] for pair in canonical["matches"]] + list(canonical["unmatched_gt"])
    all_counts = {label: 0 for label in (
        "NO_2D_PERSON_SUPPORT", "TWO_D_MATCH_WORLD_ERROR_GT_3M", "TWO_D_AND_WORLD_MATCH",
        "MATCHING_CONTENTION", "IGNORED_NEUTRAL")}
    fn_counts = dict(all_counts)
    fn_details: list[dict[str, Any]] = []
    for target in all_gt:
        gid = target["stable_row"]
        pair = image_gt_to_pair.get(gid)
        potential = primary_2d["potential_by_gt"].get(gid, set())
        if pair is None:
            label = "MATCHING_CONTENTION" if potential else "NO_2D_PERSON_SUPPORT"
        elif pair["world_error_m"] > 3.0:
            label = "TWO_D_MATCH_WORLD_ERROR_GT_3M"
        elif gid in canonical_gt_to_pred:
            label = "TWO_D_AND_WORLD_MATCH"
        else:
            label = "MATCHING_CONTENTION"
        all_counts[label] += 1
        if gid in canonical_fn_ids:
            fn_counts[label] += 1
            fn_details.append({"gt": target, "label": label, "pair": pair,
                               "has_valid_one_to_one_2d_match": pair is not None})
    all_counts["IGNORED_NEUTRAL"] = len(canonical["ignored"])
    rows: list[dict[str, Any]] = []
    for scope, counts, denominator in (
        ("all_gt_plus_ignored_neutral", all_counts, len(all_gt) + len(canonical["ignored"])),
        ("canonical_joint_fn", fn_counts, len(canonical["unmatched_gt"])),
    ):
        if sum(counts.values()) != denominator:
            raise RuntimeError(f"taxonomy denominator failure: {model}/{threshold}/{scope}")
        for label, count in counts.items():
            rows.append({"model": model, "threshold": threshold, "scope": scope,
                         "label": label, "count": count, "denominator": denominator,
                         "fraction": count / max(1, denominator)})
    fn_den = len(canonical["unmatched_gt"])
    localization = sum(item["label"] == "TWO_D_MATCH_WORLD_ERROR_GT_3M" for item in fn_details)
    lacks_2d = sum(not item["has_valid_one_to_one_2d_match"] for item in fn_details)
    decision = {
        "canonical_joint_fn_denominator": fn_den,
        "valid_2d_but_world_error_gt_3m": localization,
        "valid_2d_but_world_error_gt_3m_fraction": localization / max(1, fn_den),
        "lacks_valid_one_to_one_2d_match": lacks_2d,
        "lacks_valid_one_to_one_2d_match_fraction": lacks_2d / max(1, fn_den),
        "matching_contention": fn_counts["MATCHING_CONTENTION"],
    }
    return rows, decision


def threshold_grid() -> list[float]:
    values = {round(0.02 + 0.005 * index, 6) for index in range(int(round((1.0 - 0.02) / 0.005)) + 1)}
    values.update((0.02, 0.20, 0.40, 0.50, 1.0))
    return sorted(values)


def trapezoid_auprc(rows: Sequence[Mapping[str, Any]]) -> float:
    points = sorted({(float(row["recall"]), float(row["precision"])) for row in rows})
    if len(points) < 2:
        return 0.0
    area = 0.0
    for (r0, p0), (r1, p1) in zip(points[:-1], points[1:]):
        area += (r1 - r0) * (p0 + p1) / 2.0
    return area


def score_summary(predictions: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, Any]:
    values = np.asarray([item["score"] for items in predictions.values() for item in items
                         if item["class_name"] == "person"], dtype=np.float64)
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)), "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def pairs_by_stable(matches: Iterable[Mapping[str, Any]]) -> set[tuple[int, int]]:
    return {(pair["prediction"]["stable_row"], pair["gt"]["stable_row"]) for pair in matches}
