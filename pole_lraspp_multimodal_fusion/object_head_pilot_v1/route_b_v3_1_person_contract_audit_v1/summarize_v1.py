#!/usr/bin/env python3
"""Create-only aggregate slice tables for completed person-contract audit rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CATEGORIES = (
    "ON_OWN_VISIBLE_MASK", "ON_CLOSER_OCCLUDER", "INSIDE_PROJECTED_BOX_BUT_NOT_VISIBLE",
    "OUTSIDE_OWN_PROJECTED_BOX", "OUTSIDE_FRAME",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_typed(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in source.items():
                if value == "":
                    row[key] = ""
                    continue
                try:
                    number = float(value)
                    row[key] = int(number) if math.isfinite(number) and number.is_integer() else number
                except ValueError:
                    row[key] = value
            result.append(row)
    return result


def write_csv_x(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty summary: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def visibility_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    dimensions = (
        ("overall", lambda row: "all"),
        ("visibility_tier", lambda row: str(row["visibility_tier"])),
        ("distance_m", lambda row: str(row["distance_bin_m"])),
        ("area_px", lambda row: str(row["area_bin_px"])),
        ("radar_support", lambda row: "supported" if int(row["radar_supported"]) else "unsupported"),
        ("episode", lambda row: str(row["episode_id"])),
    )
    for contract in ("v010", "v025"):
        for split in ("train", "val"):
            population = [row for row in rows if row["contract"] == contract and row["split"] == split]
            for dimension, labeler in dimensions:
                grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for row in population:
                    grouped[labeler(row)].append(row)
                for group, subset in sorted(grouped.items()):
                    counts = Counter(str(row["category"]) for row in subset)
                    denominator = len(subset)
                    if sum(counts.values()) != denominator:
                        raise RuntimeError(f"visibility summary denominator failure: {contract}/{split}/{dimension}/{group}")
                    for category in CATEGORIES:
                        count = counts[category]
                        output.append({
                            "contract": contract, "split": split, "slice_dimension": dimension,
                            "slice_value": group, "category": category, "count": count,
                            "denominator": denominator, "percentage": 100.0 * count / denominator,
                            "off_own_visible_mask_count": denominator - counts["ON_OWN_VISIBLE_MASK"],
                            "off_own_visible_mask_percentage": 100.0 * (denominator - counts["ON_OWN_VISIBLE_MASK"]) / denominator,
                        })
    return output


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data)


def _median(values: Iterable[float]) -> float:
    data = sorted(values)
    midpoint = len(data) // 2
    return data[midpoint] if len(data) % 2 else (data[midpoint - 1] + data[midpoint]) / 2.0


def gaussian_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    dimensions = (
        ("overall", lambda row: "all"),
        ("distance_m", lambda row: str(row["distance_bin_m"])),
        ("area_px", lambda row: str(row["area_bin_px"])),
    )
    for split in ("train", "val"):
        for implementation in ("current", "reference"):
            population = [row for row in rows if row["split"] == split and row["implementation"] == implementation]
            for dimension, labeler in dimensions:
                grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for row in population:
                    grouped[labeler(row)].append(row)
                for group, subset in sorted(grouped.items()):
                    denominator = len(subset)
                    radius1 = sum(int(row["integer_radius"]) == 1 for row in subset)
                    output.append({
                        "split": split, "implementation": implementation,
                        "slice_dimension": dimension, "slice_value": group,
                        "denominator": denominator, "radius_1_count": radius1,
                        "radius_1_percentage": 100.0 * radius1 / denominator,
                        "raw_radius_mean": _mean(float(row["raw_radius"]) for row in subset),
                        "raw_radius_median": _median(float(row["raw_radius"]) for row in subset),
                        "integer_radius_mean": _mean(float(row["integer_radius"]) for row in subset),
                        "positive_cells_mean": _mean(float(row["positive_cells"]) for row in subset),
                        "support_cells_mean": _mean(float(row["support_cells"]) for row in subset),
                        "mean_abs_left_right_asymmetry_cells": _mean(abs(float(row["left_right_asymmetry_cells"])) for row in subset),
                        "mean_abs_up_down_asymmetry_cells": _mean(abs(float(row["up_down_asymmetry_cells"])) for row in subset),
                    })
    return output


def markdown(visibility: Sequence[Mapping[str, Any]], gaussian: Sequence[Mapping[str, Any]], hashes: Mapping[str, str]) -> str:
    lines = [
        "# Person contract audit slice supplement", "",
        "This create-only supplement materializes the requested aggregate slices from the completed per-object audit rows. It changes no metric, decision, or existing artifact.", "",
        f"- `target_visibility_audit.csv` SHA-256: `{hashes['target_visibility_audit.csv']}`",
        f"- `gaussian_radius_comparison.csv` SHA-256: `{hashes['gaussian_radius_comparison.csv']}`", "",
        "## Target-centre observability", "",
        "| contract | split | denominator | on own | closer occluder | inside box, not visible | off-own total |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for contract in ("v010", "v025"):
        for split in ("train", "val"):
            rows = [row for row in visibility if row["contract"] == contract and row["split"] == split
                    and row["slice_dimension"] == "overall"]
            by_category = {row["category"]: row for row in rows}
            denominator = int(rows[0]["denominator"])
            def cell(category: str) -> str:
                item = by_category[category]
                return f"{int(item['count'])} ({float(item['percentage']):.3f}%)"
            lines.append(f"| {contract} | {split} | {denominator} | {cell('ON_OWN_VISIBLE_MASK')} | {cell('ON_CLOSER_OCCLUDER')} | {cell('INSIDE_PROJECTED_BOX_BUT_NOT_VISIBLE')} | {int(rows[0]['off_own_visible_mask_count'])} ({float(rows[0]['off_own_visible_mask_percentage']):.3f}%) |")
    lines += ["", "Full counts and percentages by visibility tier, distance, area, radar support, split, and episode are in `target_visibility_summary.csv`.", "",
              "## Gaussian radius", "",
              "| split | implementation | denominator | radius 1 | raw mean / median | support cells mean |",
              "|---|---|---:|---:|---:|---:|"]
    for split in ("train", "val"):
        for implementation in ("current", "reference"):
            row = next(item for item in gaussian if item["split"] == split and item["implementation"] == implementation and item["slice_dimension"] == "overall")
            lines.append(f"| {split} | {implementation} | {int(row['denominator'])} | {int(row['radius_1_count'])} ({float(row['radius_1_percentage']):.3f}%) | {float(row['raw_radius_mean']):.4f} / {float(row['raw_radius_median']):.4f} | {float(row['support_cells_mean']):.3f} |")
    lines += ["", "Full radius, support, and forced-floor asymmetry summaries by distance and projected area are in `gaussian_radius_summary.csv`.", ""]
    return "\n".join(lines)


def run(output: Path) -> int:
    source_paths = {
        "target_visibility_audit.csv": output / "target_visibility_audit.csv",
        "gaussian_radius_comparison.csv": output / "gaussian_radius_comparison.csv",
    }
    if not all(path.is_file() for path in source_paths.values()):
        raise RuntimeError("completed raw audit rows are absent")
    destinations = (output / "target_visibility_summary.csv", output / "gaussian_radius_summary.csv",
                    output / "AUDIT_SLICE_SUPPLEMENT.md")
    if any(path.exists() for path in destinations):
        raise FileExistsError("create-only slice supplement already exists")
    visibility = visibility_summary(read_typed(source_paths["target_visibility_audit.csv"]))
    gaussian = gaussian_summary(read_typed(source_paths["gaussian_radius_comparison.csv"]))
    write_csv_x(destinations[0], visibility)
    write_csv_x(destinations[1], gaussian)
    hashes = {name: sha256(path) for name, path in source_paths.items()}
    destinations[2].write_text(markdown(visibility, gaussian, hashes), encoding="utf-8")
    print(f"wrote {len(visibility)} visibility slice rows and {len(gaussian)} Gaussian slice rows")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
