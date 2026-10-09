"""Prediction-blind retained fixtures; never reads held/test scene partitions."""
from __future__ import annotations

import csv
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2

from .adapter import (COMPLETENESS_KEYS, ROOT, SCHEMA, CompleteTargetInputs,
                      TargetInputError, _original_modules, input_content_sha256, sha256)

FROZEN = "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/views/val"
FINAL = "experiments/route_b_v3_1_clean_base_v1/20260828_012309/contracts/v010/val"
STATIC = "experiments/route_b_v3_static_vehicle_gt_audit_v1/20260827_225706/static_vehicle_visibility.csv"
FIXTURES = {
    "canonical_v3_06_val_50_50_s602_tm1602_001177_frame4935": {
        "segmentation": "cdc07d4e727e65157683e92ca1d97a5b7a971ae6de165cedd5f9b95eb766344e",
        "object_ignore": "e1f2291c0eab6daad23b7dc8886b4bbbc70fe1ed556afc8a5c871a717ad590e2",
    },
    "canonical_v3_05_val_30_30_s601_tm1601_000027_frame357": {
        "segmentation": "9dfb833f343adb9ca7810509e78384770f56e1f8162bdbc337e6c6c465ee2480",
        "object_ignore": "d7fddd8bf3de5732969cdbb5ef3d25a06c434fde9329d979dca45d3ccd6cee2d",
    },
}


@lru_cache(maxsize=32)
def _fixture_rows(path: Path):
    with path.open(newline="") as stream:
        return tuple(row for row in csv.DictReader(stream) if row.get("sample_id") in FIXTURES)


def load_retained_fixture(sample_id: str, *, root: Path = ROOT):
    if sample_id not in FIXTURES:
        raise TargetInputError("not a registered retained validation fixture")
    builder, visibility, _recon = _original_modules(root)
    source = root / FROZEN
    paths = {"manifest": source / "manifest.csv", "boxes": source / "object_boxes_all.csv",
             "visibility": source / "object_visibility_all.csv", "static_candidates": root / STATIC}
    selected = {name: [row for row in _fixture_rows(path) if row["sample_id"] == sample_id]
                for name, path in paths.items()}
    if len(selected["manifest"]) != 1:
        raise TargetInputError("fixture manifest identity mismatch")
    frame = selected["manifest"][0]
    paths.update(depth=source / frame["experiment_id"] / "depth" / f"{sample_id}.png",
                 semantic=source / frame["instance_raw_path"], original_v3_mask=source / frame["mask_path"])
    for name in ("static_static_conflicts.csv", "static_dynamic_conflicts.csv", "manual_review_selection.csv"):
        paths[f"quarantine/{name}"] = builder.RECON / name
    paths["static_catalogue"] = builder.CATALOG_PATH
    provenance = {"schema": SCHEMA, "scope": "HISTORICAL_REGISTERED_V010",
                  "sample_id": sample_id, "complete": dict.fromkeys(COMPLETENESS_KEYS, True),
                  "evidence_sha256": {name: sha256(path) for name, path in paths.items()},
                  "evidence_paths": {name: str(path.relative_to(root)) for name, path in paths.items()}}
    value = CompleteTargetInputs(frame=frame, split="val",
        depth_bgra=cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED),
        raw_semantic=cv2.imread(str(paths["semantic"]), cv2.IMREAD_UNCHANGED),
        original_v3_mask=cv2.imread(str(paths["original_v3_mask"]), cv2.IMREAD_UNCHANGED),
        actor_boxes=selected["boxes"], actor_visibility=selected["visibility"],
        static_candidates=selected["static_candidates"], quarantine=builder.registered_quarantines(),
        provenance=provenance)
    value = replace(value, provenance=dict(provenance, input_content_sha256=input_content_sha256(value)))
    final = root / FINAL
    targets = {name: final / directory / f"{sample_id}.png" for name, directory in
               (("segmentation", "segmentation_masks"), ("object_ignore", "object_ignore_masks"))}
    for name, path in targets.items():
        if sha256(path) != FIXTURES[sample_id][name]:
            raise TargetInputError(f"registered {name} reference changed")
    for name in ("object_boxes.csv", "object_ignore_regions.csv"):
        targets[name] = [row for row in _fixture_rows(final / name) if row["sample_id"] == sample_id]
    return value, targets
