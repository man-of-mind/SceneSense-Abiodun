#!/usr/bin/env python3
"""Create-only camera-plane localization contracts derived from Route B v3.1.

Raw RGB/radar/depth/semantic payloads are never copied. Segmentation targets are
directory symlinks to the immutable v3.1 targets. Existing object-ignore masks are
symlinked for unaffected frames; only frames containing a transitioned object receive
a newly derived mask. Original positive rows remain immutable and every transition is
preserved in the provenance CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
CONTRACTS = ("v010", "v025")
SPLITS = ("train", "val")
REASON = "CAMERA_PLANE_STRADDLING_CENTER_NONPOSITIVE_DEPTH"
FINAL_INVALID = "LRASPP_CAMERA_PLANE_CONTRACT_INVALID"

PROVENANCE_FIELDS = (
    "contract", "split", "sample_id", "frame_id", "source_identity", "source_type",
    "camera_forward_depth_m", "projected_bbox_x", "projected_bbox_y",
    "projected_bbox_w", "projected_bbox_h", "eligible_v010", "eligible_v025",
    "original_status", "derived_localization_status", "ignore_reason",
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


def write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_x(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def paint_scaled_box(mask: np.ndarray, row: Mapping[str, Any], source_size: tuple[int, int]) -> int:
    source_w, source_h = source_size
    x, y, width, height = (
        float(row["gt_bbox_x"]), float(row["gt_bbox_y"]),
        float(row["gt_bbox_w"]), float(row["gt_bbox_h"]),
    )
    x0 = max(0, min(mask.shape[1], int(math.floor(x * mask.shape[1] / source_w))))
    y0 = max(0, min(mask.shape[0], int(math.floor(y * mask.shape[0] / source_h))))
    x1 = max(0, min(mask.shape[1], int(math.ceil((x + width) * mask.shape[1] / source_w))))
    y1 = max(0, min(mask.shape[0], int(math.ceil((y + height) * mask.shape[0] / source_h))))
    before = int(np.count_nonzero(mask))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"empty projected transition box: {row['sample_id']}/{row['source_identity']}")
    mask[y0:y1, x0:x1] = 255
    return int(np.count_nonzero(mask)) - before


def write_png_x(path: Path, image: np.ndarray) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
        raise RuntimeError(f"failed to write {path}")
    return sha256(path)


def transition_key(row: Mapping[str, str]) -> tuple[str, str]:
    return str(row["sample_id"]), str(row["source_identity"])


def build_one(
    *, source: Path, output: Path, contract: str, split: str,
    eligibility: Mapping[tuple[str, str], Mapping[str, bool]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = source / "contracts" / contract / split
    output_dir = output / "contracts" / contract / split
    output_dir.mkdir(parents=True, exist_ok=False)
    source_boxes = read_csv(source_dir / "object_boxes.csv")
    source_ignore = read_csv(source_dir / "object_ignore_regions.csv")
    source_targets = read_csv(source_dir / "target_manifest.csv")
    manifest = [row for row in read_csv(source / "dataset/manifest.csv") if row["split"] == split]
    manifest_by_sample = {row["sample_id"]: row for row in manifest}
    if len(manifest_by_sample) != len(manifest):
        raise RuntimeError(f"duplicate sample IDs in {split}")

    transitions = [row for row in source_boxes if float(row["object_sensor_x"]) <= 0.0]
    remaining = [row for row in source_boxes if float(row["object_sensor_x"]) > 0.0]
    if any(float(row["object_sensor_x"]) <= 0.0 for row in remaining):
        raise RuntimeError(f"non-positive depth remained in {contract}/{split}")
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    provenance: list[dict[str, Any]] = []
    added_ignore: list[dict[str, Any]] = []
    for row in transitions:
        by_sample[row["sample_id"]].append(row)
        eligible = eligibility[transition_key(row)]
        provenance.append({
            "contract": contract, "split": split, "sample_id": row["sample_id"],
            "frame_id": row["frame_id"], "source_identity": row["source_identity"],
            "source_type": row["source_kind"],
            "camera_forward_depth_m": row["object_sensor_x"],
            "projected_bbox_x": row["gt_bbox_x"], "projected_bbox_y": row["gt_bbox_y"],
            "projected_bbox_w": row["gt_bbox_w"], "projected_bbox_h": row["gt_bbox_h"],
            "eligible_v010": int(bool(eligible["v010"])),
            "eligible_v025": int(bool(eligible["v025"])),
            "original_status": row["contract_state"],
            "derived_localization_status": "IGNORE", "ignore_reason": REASON,
        })
        added_ignore.append({
            "contract": contract, "split": split, "experiment_id": row["experiment_id"],
            "sample_id": row["sample_id"], "frame_id": row["frame_id"],
            "class_name": row["label"], "source_kind": row["source_kind"],
            "source_identity": row["source_identity"],
            "bbox_x": row["gt_bbox_x"], "bbox_y": row["gt_bbox_y"],
            "bbox_w": row["gt_bbox_w"], "bbox_h": row["gt_bbox_h"],
            "reason": REASON, "object_ignore": 1, "segmentation_ignore": 0,
            "source_record": f"camera_plane_transition:{row['source_identity']}",
        })

    box_fields = list(source_boxes[0]) if source_boxes else []
    ignore_fields = list(source_ignore[0]) if source_ignore else [
        "contract", "split", "experiment_id", "sample_id", "frame_id", "class_name",
        "source_kind", "source_identity", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "reason", "object_ignore", "segmentation_ignore", "source_record",
    ]
    write_csv_x(output_dir / "object_boxes.csv", box_fields, remaining)
    write_csv_x(output_dir / "object_ignore_regions.csv", ignore_fields, [*source_ignore, *added_ignore])
    (output_dir / "segmentation_masks").symlink_to(
        (source_dir / "segmentation_masks").resolve(strict=True), target_is_directory=True
    )
    masks_dir = output_dir / "object_ignore_masks"
    masks_dir.mkdir()
    derived_targets: list[dict[str, Any]] = []
    changed_mask_count = 0
    symlink_mask_count = 0
    added_ignore_pixels = 0
    for target in source_targets:
        sample_id = target["sample_id"]
        source_mask = source_dir / target["object_ignore_mask_path"]
        destination = masks_dir / f"{sample_id}.png"
        rows = by_sample.get(sample_id, [])
        if rows:
            mask = cv2.imread(str(source_mask), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != (432, 768):
                raise RuntimeError(f"invalid source ignore mask: {sample_id}")
            frame = manifest_by_sample[sample_id]
            for row in rows:
                added_ignore_pixels += paint_scaled_box(
                    mask, row, (int(frame["camera_width"]), int(frame["camera_height"]))
                )
            object_hash = write_png_x(destination, mask)
            changed_mask_count += 1
        else:
            destination.symlink_to(source_mask.resolve(strict=True))
            object_hash = sha256(destination)
            symlink_mask_count += 1
        segmentation = source_dir / target["segmentation_mask_path"]
        segmentation_hash = sha256(segmentation)
        if segmentation_hash != target["segmentation_mask_sha256"]:
            raise RuntimeError(f"source segmentation hash drift: {sample_id}")
        derived_targets.append({
            "sample_id": sample_id,
            "segmentation_mask_path": target["segmentation_mask_path"],
            "segmentation_mask_sha256": segmentation_hash,
            "object_ignore_mask_path": f"object_ignore_masks/{sample_id}.png",
            "object_ignore_mask_sha256": object_hash,
        })
    write_csv_x(
        output_dir / "target_manifest.csv",
        ("sample_id", "segmentation_mask_path", "segmentation_mask_sha256",
         "object_ignore_mask_path", "object_ignore_mask_sha256"),
        derived_targets,
    )

    source_seg_payload_hash = hashlib.sha256("".join(
        row["segmentation_mask_sha256"] for row in source_targets
    ).encode("ascii")).hexdigest()
    derived_seg_payload_hash = hashlib.sha256("".join(
        row["segmentation_mask_sha256"] for row in derived_targets
    ).encode("ascii")).hexdigest()
    transition_mask_gate = all(
        np.any(cv2.imread(str(masks_dir / f"{sample_id}.png"), cv2.IMREAD_UNCHANGED) != 0)
        for sample_id in by_sample
    )
    return {
        "source_positive_records": len(source_boxes),
        "derived_positive_records": len(remaining),
        "transition_records": len(transitions),
        "transition_class_counts": dict(sorted(Counter(row["label"] for row in transitions).items())),
        "transition_source_counts": dict(sorted(Counter(row["source_kind"] for row in transitions).items())),
        "transition_unique_identities": len({row["source_identity"] for row in transitions}),
        "transition_frames": len(by_sample),
        "changed_object_ignore_masks": changed_mask_count,
        "symlinked_object_ignore_masks": symlink_mask_count,
        "added_object_ignore_pixels": added_ignore_pixels,
        "all_remaining_depth_positive": all(float(row["object_sensor_x"]) > 0.0 for row in remaining),
        "every_transition_has_nonbackground_ignore_mask": transition_mask_gate,
        "source_segmentation_payload_hash": source_seg_payload_hash,
        "derived_segmentation_payload_hash": derived_seg_payload_hash,
        "segmentation_hashes_unchanged": source_seg_payload_hash == derived_seg_payload_hash,
        "source_object_boxes_sha256": sha256(source_dir / "object_boxes.csv"),
        "derived_object_boxes_sha256": sha256(output_dir / "object_boxes.csv"),
        "derived_object_ignore_regions_sha256": sha256(output_dir / "object_ignore_regions.csv"),
        "derived_target_manifest_sha256": sha256(output_dir / "target_manifest.csv"),
    }, provenance


def materialize_dataset(source: Path, output: Path) -> dict[str, Any]:
    source_dataset = source / "dataset"
    destination = output / "dataset"
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_csv(source_dataset / "manifest.csv")
    test_rows = [row for row in manifest if row.get("split") == "test"]
    if test_rows:
        raise RuntimeError("test rows present in v3.1 source view")
    episodes = sorted({row["experiment_id"] for row in manifest})
    for episode in episodes:
        (destination / episode).symlink_to(
            (source_dataset / episode).resolve(strict=True), target_is_directory=True
        )
    target_by_key = {
        (split, row["sample_id"]): row
        for split in SPLITS
        for row in read_csv(output / "contracts/v010" / split / "target_manifest.csv")
    }
    derived_manifest: list[dict[str, Any]] = []
    for row in manifest:
        item = dict(row)
        target = target_by_key[(row["split"], row["sample_id"])]
        item["mask_path"] = f"../contracts/v010/{row['split']}/{target['segmentation_mask_path']}"
        item["object_ignore_mask_path"] = f"../contracts/v010/{row['split']}/{target['object_ignore_mask_path']}"
        item["gt_contract"] = "route_b_v3_1_camera_plane_v010"
        derived_manifest.append(item)
    write_csv_x(destination / "manifest.csv", list(derived_manifest[0]), derived_manifest)
    boxes = [
        row for split in SPLITS
        for row in read_csv(output / "contracts/v010" / split / "object_boxes.csv")
    ]
    write_csv_x(destination / "object_boxes.csv", list(boxes[0]), boxes)
    train_ids = {row["sample_id"] for row in derived_manifest if row["split"] == "train"}
    val_ids = {row["sample_id"] for row in derived_manifest if row["split"] == "val"}
    return {
        "frames": len(derived_manifest),
        "train_frames": len(train_ids), "validation_frames": len(val_ids),
        "episode_symlinks": len(episodes), "raw_corpus_files_copied": 0,
        "test_rows": len(test_rows), "train_validation_disjoint": not bool(train_ids & val_ids),
        "manifest_sha256": sha256(destination / "manifest.csv"),
        "object_boxes_sha256": sha256(destination / "object_boxes.csv"),
    }


def schema_markdown() -> str:
    return f"""# Route B v3.1 camera-plane localization contract

This view changes object-centre localization eligibility only. A positive whose
physical centre has CARLA camera-forward depth `<= 0.0` becomes localization `IGNORE`
with reason `{REASON}`. Its source row remains immutable and is preserved in
`provenance/camera_plane_exclusions.csv`.

Segmentation targets are unchanged directory symlinks. The transitioned visible 2D box
is painted into the existing object-ignore mask, so object loss supplies neither a
positive nor background supervision there. During scoring, an unmatched prediction
whose model-space centre lands in that region is neutral and is neither TP nor FP.

Primary `v010` and sensitivity `v025` masks are separate files and must be cached by
contract key. The dataset contains train and validation only; it is deterministic and
does not inspect or materialize locked-test records.
"""


def run(output: Path, config_path: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = (ROOT / config["source_experiment"]).resolve(strict=True)
    source_summary = json.loads((source / "GT_CONTRACT_SUMMARY.json").read_text(encoding="utf-8"))
    source_manifest_hash_before = sha256(source / "dataset/manifest.csv")

    source_boxes = {
        (contract, split): read_csv(source / "contracts" / contract / split / "object_boxes.csv")
        for contract in CONTRACTS for split in SPLITS
    }
    eligibility: dict[tuple[str, str], dict[str, bool]] = defaultdict(
        lambda: {"v010": False, "v025": False}
    )
    for contract in CONTRACTS:
        for split in SPLITS:
            for row in source_boxes[(contract, split)]:
                eligibility[transition_key(row)][contract] = True

    summaries: dict[str, dict[str, Any]] = {contract: {} for contract in CONTRACTS}
    provenance: list[dict[str, Any]] = []
    for contract in CONTRACTS:
        for split in SPLITS:
            summary, rows = build_one(
                source=source, output=output, contract=contract, split=split,
                eligibility=eligibility,
            )
            summaries[contract][split] = summary
            provenance.extend(rows)
    write_csv_x(output / "provenance/camera_plane_exclusions.csv", PROVENANCE_FIELDS, provenance)
    dataset = materialize_dataset(source, output)
    write_json_x(output / "resolved_config.json", config)
    write_text_x(output / "CAMERA_PLANE_CONTRACT_SCHEMA.md", schema_markdown())

    primary_val = summaries["v010"]["val"]
    primary_train = summaries["v010"]["train"]
    expected = config["expected_v010_validation_transition"]
    # Byte-identical segmentation payload hashes prove both the masks and every
    # per-class pixel count are unchanged; no derived segmentation file is written.
    seg_counts_match = all(
        summaries[contract][split]["segmentation_hashes_unchanged"]
        for contract in CONTRACTS for split in SPLITS
    )
    gates = {
        "v010_validation_transition_exactly_34": primary_val["transition_records"] == expected["total"],
        "v010_validation_composition_26_actor_8_static_11_identities": (
            primary_val["transition_source_counts"] == {
                "actor": expected["actor"], "environment_static": expected["environment_static"]
            }
            and primary_val["transition_unique_identities"] == expected["unique_identities"]
        ),
        "v010_validation_zero_person_transitions": primary_val["transition_class_counts"].get("person", 0) == 0,
        "all_remaining_localization_positive_depths_positive": all(
            summaries[contract][split]["all_remaining_depth_positive"]
            for contract in CONTRACTS for split in SPLITS
        ),
        "segmentation_masks_counts_and_hashes_unchanged": seg_counts_match,
        "no_transition_became_background": all(
            summaries[contract][split]["every_transition_has_nonbackground_ignore_mask"]
            for contract in CONTRACTS for split in SPLITS
        ),
        "train_validation_sample_ids_disjoint": dataset["train_validation_disjoint"],
        "test_rows_and_payloads_absent": dataset["test_rows"] == 0,
        "no_raw_corpus_copied": dataset["raw_corpus_files_copied"] == 0,
    }
    source_manifest_hash_after = sha256(source / "dataset/manifest.csv")
    if source_manifest_hash_before != source_manifest_hash_after:
        raise RuntimeError("source manifest changed during derived-contract build")
    if not all(gates.values()):
        raise RuntimeError(f"hard contract gate failure: {gates}")
    result = {
        "schema": "route_b_v3_1_camera_plane_contract_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_terminal": "CAMERA_PLANE_CONTRACT_READY",
        "rule": config["rule"], "summaries": summaries, "dataset": dataset,
        "hard_gates": gates,
        "source_provenance": {
            "source_experiment": str(source),
            "source_manifest_sha256": source_manifest_hash_before,
            "source_summary_sha256": sha256(source / "GT_CONTRACT_SUMMARY.json"),
            "source_files_modified": 0,
        },
        "segmentation_pixel_counts_unchanged": {
            contract: {
                split: source_summary["summaries"][contract][split]["segmentation_pixels"]
                for split in SPLITS
            }
            for contract in CONTRACTS
        },
        "primary_v010_counts": {
            "train_source_positive": primary_train["source_positive_records"],
            "train_localization_positive": primary_train["derived_positive_records"],
            "train_localization_ignore_transition": primary_train["transition_records"],
            "validation_source_positive": primary_val["source_positive_records"],
            "validation_localization_positive": primary_val["derived_positive_records"],
            "validation_localization_ignore_transition": primary_val["transition_records"],
        },
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(output / "CAMERA_PLANE_CONTRACT_SUMMARY.json", result)
    write_text_x(output / "PHASE_A_COMPLETE", "CAMERA_PLANE_CONTRACT_READY\n")
    print(json.dumps({
        "phase_terminal": result["phase_terminal"], "hard_gates": gates,
        "primary_v010_counts": result["primary_v010_counts"],
        "sensitivity_v025_counts": {
            split: summaries["v025"][split]["transition_records"] for split in SPLITS
        },
        "wall_seconds": result["wall_seconds"],
    }, indent=2, allow_nan=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/camera_plane_contract_v1.json",
    )
    args = parser.parse_args()
    try:
        run(args.output.resolve(), args.config.resolve())
        return 0
    except Exception as exc:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if not (output / "TERMINAL_VERDICT.txt").exists():
            write_text_x(output / "TERMINAL_VERDICT.txt", FINAL_INVALID + "\n")
            write_json_x(output / "contract_failure.json", {
                "terminal": FINAL_INVALID, "error": f"{type(exc).__name__}: {exc}",
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
