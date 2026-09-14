#!/usr/bin/env python3
"""Plot and diagnose capture-callback to action-start delay for all 288 cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/splitfusion-capture-action-mpl")

import matplotlib.pyplot as plt

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288 as source


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260914_capture_vs_action_clock_v4"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260914_capture_to_action_start_diagnostic_v2"
)
SCHEMA = "scenesense.splitfusion.capture_to_action_start_diagnostic.v1"
TERMINAL = "SPLITFUSION_CAPTURE_TO_ACTION_START_DIAGNOSTIC_COMPLETE"
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "FADE_RECOVERY",
    "ADVERSE_STABLE",
)
PROFILE_LABELS = {
    "FAVORABLE_STABLE": "Favorable",
    "MID_VARIABLE": "Mid-variable",
    "FADE_RECOVERY": "Fade/recovery",
    "ADVERSE_STABLE": "Adverse",
}
STAGES = (
    "sensor_wait_ms",
    "radar_window_ms",
    "radar_prepare_ms",
    "rgb_convert_ms",
    "scene_snapshot_ms",
    "pre_front_compute_ms",
)
PDF_METADATA = {
    "Creator": "SplitFusion capture-to-action diagnostic",
    "Producer": "matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


class DiagnosticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def wall_ns(value: str) -> int:
    return int(
        (Decimal(value) * Decimal(1_000_000_000)).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    require(bool(ordered), "percentile input is empty")
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), "refusing to write empty CSV")
    fields = list(rows[0])
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_source() -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads((SOURCE / "artifact_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        require(sha256(SOURCE / relative) == expected, f"source hash drift: {relative}")
    analysis = json.loads((SOURCE / "analysis.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in read_csv(SOURCE / "action_network_dual_clock_freshness.csv")
        if row["queue_policy"] == "LATEST_ONLY_NO_EXPIRY"
    ]
    require(len(rows) == 288, "source does not contain 288 latest-only cells")
    return analysis, rows


def ordered_cell_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    by_key = {
        (row["network_profile"], int(row["action_id"])): row for row in rows
    }
    require(len(by_key) == 288, "duplicate action/profile cell")
    output: list[dict[str, Any]] = []
    for index, profile in enumerate(PROFILE_ORDER):
        for action in range(72):
            row = by_key[(profile, action)]
            output.append(
                {
                    "cell_index": index * 72 + action,
                    "cell_id": row["cell_id"],
                    "action_id": action,
                    "network_profile": profile,
                    "capture_to_action_start_ms_median": float(
                        row["capture_to_action_start_ms_median"]
                    ),
                    "capture_to_action_start_ms_p95": float(
                        row["capture_to_action_start_ms_p95"]
                    ),
                    "capture_to_action_start_ms_p99": float(
                        row["capture_to_action_start_ms_p99"]
                    ),
                }
            )
    return output


def raw_tail_diagnostic(
    analysis: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bridge_ns = int(analysis["clock_bridge"]["global_median_offset_ns"])
    threshold = float(analysis["capture_to_action_start_all_sent_ms"]["p99"])
    cell_table = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    delays: list[float] = []
    stage_all: dict[str, list[float]] = {stage: [] for stage in STAGES}
    stage_tail: dict[str, list[float]] = {stage: [] for stage in STAGES}
    source_hashes = 0
    for number, cell in enumerate(cell_table, start=1):
        attempt = source._attempt(cell)
        source._attempt_manifest_hash(attempt, "per_frame_metrics.csv")
        source_hashes += 1
        for row in source._sent_rows(attempt):
            delay = (
                int(row["capture_started_ns"])
                + bridge_ns
                - wall_ns(row["capture_wall_s"])
            ) / 1e6
            delays.append(delay)
            for stage in STAGES:
                try:
                    value = float(row[stage])
                except (KeyError, TypeError, ValueError):
                    continue
                stage_all[stage].append(value)
                if delay >= threshold:
                    stage_tail[stage].append(value)
        if number % 72 == 0:
            print(f"raw diagnostic: {number}/288 cells", flush=True)
    require(len(delays) == 896_856, "sent-frame total drift")
    bins = (
        ("LT_50", lambda value: value < 50),
        ("LT_25", lambda value: value < 25),
        ("GE_25_LT_50", lambda value: 25 <= value < 50),
        ("GE_50_LT_75", lambda value: 50 <= value < 75),
        ("GE_75_LT_100", lambda value: 75 <= value < 100),
        ("GE_100_LT_125", lambda value: 100 <= value < 125),
        ("GE_125_LT_150", lambda value: 125 <= value < 150),
        ("GE_150", lambda value: value >= 150),
    )
    return {
        "schema": f"{SCHEMA}.raw_tail",
        "sent_frames": len(delays),
        "source_per_frame_hashes_verified": source_hashes,
        "boundary": {
            "start": "capture_wall_s: host receipt of raw RGB callback",
            "end": "capture_started_ns: entry to seven-channel tensor assembly",
            "clock_bridge": "same-event UE result-receive monotonic/wall pair",
        },
        "distribution_ms": {
            "minimum": min(delays),
            "median": statistics.median(delays),
            "p95": percentile(delays, 0.95),
            "p99": percentile(delays, 0.99),
            "maximum": max(delays),
        },
        "bins": {
            label: {
                "count": sum(predicate(value) for value in delays),
                "fraction": sum(predicate(value) for value in delays) / len(delays),
            }
            for label, predicate in bins
        },
        "p99_tail_threshold_ms": threshold,
        "p99_tail_count": sum(value >= threshold for value in delays),
        "component_distributions": {
            stage: {
                "all_frames_median_ms": statistics.median(stage_all[stage]),
                "all_frames_p95_ms": percentile(stage_all[stage], 0.95),
                "p99_delay_tail_median_ms": statistics.median(stage_tail[stage]),
                "p99_delay_tail_p95_ms": percentile(stage_tail[stage], 0.95),
            }
            for stage in STAGES
        },
        "cell_p99_range_ms": {
            "minimum": min(row["capture_to_action_start_ms_p99"] for row in cells),
            "median": statistics.median(
                row["capture_to_action_start_ms_p99"] for row in cells
            ),
            "maximum": max(row["capture_to_action_start_ms_p99"] for row in cells),
        },
    }


def line_plot(rows: Sequence[Mapping[str, Any]], output: Path) -> list[Path]:
    figure, axis = plt.subplots(figsize=(16.0, 6.3))
    x = [int(row["cell_index"]) for row in rows]
    for profile_index in range(4):
        left = profile_index * 72 - 0.5
        right = (profile_index + 1) * 72 - 0.5
        if profile_index % 2:
            axis.axvspan(left, right, color="#eeeeee", alpha=0.48, zorder=0)
        axis.text(
            (left + right) / 2,
            0.965,
            PROFILE_LABELS[PROFILE_ORDER[profile_index]],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        if profile_index:
            axis.axvline(left, color="#666666", linewidth=0.9, alpha=0.6)
    specifications = (
        ("capture_to_action_start_ms_median", "Median", "#2a6fbb", 1.7),
        ("capture_to_action_start_ms_p95", "P95", "#e68613", 1.5),
        ("capture_to_action_start_ms_p99", "P99", "#c23b3b", 1.4),
    )
    for field, label, color, width in specifications:
        axis.plot(
            x,
            [float(row[field]) for row in rows],
            label=label,
            color=color,
            linewidth=width,
        )
    axis.axhline(
        100,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="100 ms source interval",
    )
    ticks = list(range(0, 288, 12)) + [287]
    labels = [str(tick % 72) for tick in ticks[:-1]] + ["71"]
    axis.set_xticks(ticks, labels)
    axis.set_xlabel("Action ID within each network-profile block")
    axis.set_ylabel("RGB callback to tensor-assembly start (ms)")
    axis.set_ylim(20, 190)
    figure.suptitle(
        "Capture-to-action-start delay across all 288 cells",
        y=0.995,
        fontweight="bold",
    )
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=4, loc="upper left")
    axis.tick_params(axis="both", labelsize=9, width=1.1)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_fontweight("bold")
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    figure.tight_layout()
    paths = [
        output / "01_capture_to_action_start_all_288_cells.png",
        output / "01_capture_to_action_start_all_288_cells.pdf",
    ]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], metadata=PDF_METADATA, bbox_inches="tight")
    plt.close(figure)
    return paths


def report(diagnostic: Mapping[str, Any]) -> str:
    dist = diagnostic["distribution_ms"]
    stages = diagnostic["component_distributions"]
    return "\n".join(
        [
            "# Capture-to-action-start diagnostic",
            "",
            "`capture_wall_s` is not a timestamp for a fully prepared seven-channel",
            "tensor. It is host wall time when the raw RGB callback reaches the UE",
            "process. `capture_started_ns` is recorded later, at entry to RGB/radar",
            "tensor assembly immediately before the chosen action's UE dispatch.",
            "",
            "Between those boundaries, the worker may wait for the matching radar",
            "record, extract the rolling radar window, transform/rasterize radar,",
            "convert RGB, freeze the evaluation snapshot, and wait for worker access.",
            "",
            "## Distribution",
            "",
            f"Across {diagnostic['sent_frames']:,} sent frames: median {dist['median']:.1f} ms,",
            f"p95 {dist['p95']:.1f} ms, p99 {dist['p99']:.1f} ms and maximum",
            f"{dist['maximum']:.1f} ms. {100 * diagnostic['bins']['LT_50']['fraction']:.1f}%",
            "of frames start action processing within 50 ms of the RGB callback.",
            "",
            "## Why the p99 reaches about 143 ms",
            "",
            "| Component | All-frame median | All-frame p95 | Slowest-1% median | Slowest-1% p95 |",
            "|---|---:|---:|---:|---:|",
            *[
                f"| {stage} | {values['all_frames_median_ms']:.1f} ms | "
                f"{values['all_frames_p95_ms']:.1f} ms | "
                f"{values['p99_delay_tail_median_ms']:.1f} ms | "
                f"{values['p99_delay_tail_p95_ms']:.1f} ms |"
                for stage, values in stages.items()
            ],
            "",
            "The tail is primarily preparation work, especially radar rasterization,",
            "not a claim that CARLA needs 143 ms to generate every frame. In the",
            "slowest 1%, radar preparation rises from a 23.2 ms overall median to",
            "about 93.7 ms, and total pre-front compute rises from 33.3 to 134.5 ms.",
            "",
            "The 100 ms line is the nominal source interval, not a per-frame action",
            "deadline. A p99 above it means a small fraction of preparation operations",
            "overlap or miss the next source opportunity; it does not change the",
            "physical capture timestamp used for map AoI.",
            "",
        ]
    )


def run(output: Path) -> None:
    require(not output.exists(), f"create-only output exists: {output}")
    analysis, source_rows = verify_source()
    cells = ordered_cell_rows(source_rows)
    diagnostic = raw_tail_diagnostic(analysis, cells)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "capture_to_action_start_by_cell.csv", cells)
    atomic_json(output / "capture_to_action_start_tail_diagnostic.json", diagnostic)
    atomic_text(output / "REPORT.md", report(diagnostic))
    plots = line_plot(cells, output)
    primary = [
        output / "capture_to_action_start_by_cell.csv",
        output / "capture_to_action_start_tail_diagnostic.json",
        output / "REPORT.md",
        *plots,
    ]
    hashes = {path.name: sha256(path) for path in primary}
    atomic_json(
        output / "artifact_manifest.json",
        {
            "schema": f"{SCHEMA}.artifacts",
            "status": "COMPLETE",
            "source_dual_clock_manifest_sha256": sha256(SOURCE / "artifact_manifest.json"),
            "sha256": hashes,
        },
    )
    atomic_text(output / TERMINAL, TERMINAL + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.output.resolve())
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
