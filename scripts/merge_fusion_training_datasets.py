#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PATH_FIELDS = (
    "rgb_path",
    "mask_path",
    "instance_raw_path",
    "radar_tensor_path",
    "radar_points_path",
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_metadata(dataset_dir: Path) -> Dict[str, object]:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"metadata_parse_error": str(path)}


def ensure_empty_output(output_dir: Path) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    existing = [path for path in output_dir.iterdir() if path.name not in {".gitkeep"}]
    if existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new folder so an old dataset is not mixed accidentally."
        )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def all_fields(rows: Sequence[Dict[str, str]]) -> List[str]:
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def merge_datasets(input_dirs: Sequence[Path], output_dir: Path, link_mode: str) -> Dict[str, object]:
    ensure_empty_output(output_dir)
    merged_manifest: List[Dict[str, str]] = []
    merged_objects: List[Dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    source_summaries: List[Dict[str, object]] = []
    sample_ids: set[str] = set()

    for dataset_dir in input_dirs:
        dataset_dir = dataset_dir.expanduser().resolve()
        manifest_path = dataset_dir / "manifest.csv"
        object_boxes_path = dataset_dir / "object_boxes.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        if not object_boxes_path.exists():
            raise FileNotFoundError(object_boxes_path)

        rows = read_csv(manifest_path)
        object_rows = read_csv(object_boxes_path)
        metadata = load_metadata(dataset_dir)
        source_summaries.append(
            {
                "dataset_dir": str(dataset_dir),
                "manifest_rows": len(rows),
                "object_rows": len(object_rows),
                "metadata": metadata,
            }
        )

        for row in rows:
            sample_id = row.get("sample_id", "")
            if not sample_id:
                raise ValueError(f"Manifest row without sample_id in {manifest_path}")
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id across datasets: {sample_id}")
            sample_ids.add(sample_id)
            split_counts[str(row.get("split", ""))] += 1
            merged_row = dict(row)
            for field in PATH_FIELDS:
                rel = str(row.get(field, "")).strip()
                if not rel:
                    continue
                src = Path(rel)
                if not src.is_absolute():
                    src = dataset_dir / src
                dst = output_dir / rel
                link_or_copy(src, dst, link_mode)
            merged_manifest.append(merged_row)

        merged_objects.extend(object_rows)

    manifest_fields = all_fields(merged_manifest)
    object_fields = all_fields(merged_objects)
    write_csv(output_dir / "manifest.csv", merged_manifest, manifest_fields)
    write_csv(output_dir / "object_boxes.csv", merged_objects, object_fields)

    summary = {
        "schema": "scenesense_merged_fusion_training_dataset.v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "link_mode": link_mode,
        "source_count": len(input_dirs),
        "sources": source_summaries,
        "manifest_rows": len(merged_manifest),
        "object_rows": len(merged_objects),
        "split_counts": dict(sorted(split_counts.items())),
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SceneSense fusion training datasets into one dataset root.")
    parser.add_argument("output_dir", help="Output merged dataset directory.")
    parser.add_argument("input_dirs", nargs="+", help="Input dataset directories containing manifest.csv/object_boxes.csv.")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize sample files in the merged directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_datasets(
        [Path(path) for path in args.input_dirs],
        Path(args.output_dir).expanduser().resolve(),
        str(args.link_mode),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
