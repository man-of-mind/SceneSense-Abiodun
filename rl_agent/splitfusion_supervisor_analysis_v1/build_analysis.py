#!/usr/bin/env python3
"""Build the post-direct-map supervisor analysis from immutable evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288 as source
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288_final_v3 as final_v3
from rl_agent.splitfusion_edge_freshness_scheduler_v1 import counterfactual_288_direct_map_v1 as direct
from rl_agent.splitfusion_edge_freshness_scheduler_v1.two_stage_simulator import (
    TwoStageConfig,
    simulate_two_stage,
)
from rl_agent.splitfusion_map_freshness_analysis_v1 import analyze_288 as prior
from rl_agent.splitfusion_map_freshness_analysis_v1 import analyze_timing_boundaries as timing


ROOT = Path(__file__).resolve().parents[2]
DIRECT_ROOT = ROOT / (
    "experiments/splitfusion_direct_edge_map_v1/"
    "20260914_direct_map_288_counterfactual"
)
LIVE_DIRECT_ROOT = ROOT / (
    "experiments/splitfusion_direct_edge_map_v1/"
    "20260914_live_validation_retry1"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_supervisor_analysis_v1/"
    "20260914_latency_quality_and_sensor_v1"
)
SCHEMA = "scenesense.splitfusion.supervisor_analysis.v1"
TERMINAL = "SPLITFUSION_SUPERVISOR_ANALYSIS_COMPLETE"
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
PROFILE_LABEL = {
    "FAVORABLE_STABLE": "Favorable Stable",
    "MID_VARIABLE": "Mid Variable",
    "ADVERSE_STABLE": "Adverse Stable",
    "FADE_RECOVERY": "Fade Recovery",
}
FAMILY_COLOR = {
    "noAE": "#4C78A8",
    "AE128": "#F58518",
    "AE64": "#54A24B",
    "AE32": "#E45756",
}
STAGES = {
    "pure_front": "Model front backbone",
    "ue_action": "UE action path",
    "network": "Feature uplink",
    "edge_map": "Edge-to-map service",
    "total": "Action-to-map total",
}
SCATTER_STAGES = ("pure_front", "network", "edge_map", "total")
PERCENTILES = (0.50, 0.95, 0.99)


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    require(all(list(row) == fields for row in rows), f"column drift: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finite(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def percentile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def stats(values: Sequence[float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_p50_ms": percentile(values, 0.50),
        f"{prefix}_p95_ms": percentile(values, 0.95),
        f"{prefix}_p99_ms": percentile(values, 0.99),
    }


def quality_score(row: Mapping[str, Any]) -> tuple[float, float]:
    segmentation = float(row["val_segmentation_miou"])
    vehicle_iou = float(row["val_vehicle_iou"])
    person_iou = float(row["val_person_box_mask_iou"])
    localization = math.sqrt(max(0.0, vehicle_iou) * max(0.0, person_iou))
    combined = math.sqrt(max(0.0, segmentation) * localization)
    return localization, combined


def useful_outcomes(result: Any) -> list[Any]:
    installed = sorted(
        (item for item in result.outcomes if item.install_ns is not None),
        key=lambda item: (int(item.install_ns), int(item.frame.sequence_id)),
    )
    useful: list[Any] = []
    newest_capture = -1
    for item in installed:
        capture = int(item.frame.capture_ns)
        if capture > newest_capture and int(item.install_ns) < int(result.observation_end_ns):
            useful.append(item)
            newest_capture = capture
    return useful


def sensor_components(rows: Sequence[Mapping[str, str]], bridge_ns: int) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        capture_wall_ns = timing.wall_seconds_to_ns(row["capture_wall_s"])
        action_start_ns = int(row["capture_started_ns"]) + bridge_ns
        capture_to_action = (action_start_ns - capture_wall_ns) / 1e6
        prefront = float(row["pre_front_compute_ms"])
        callback_to_worker = capture_to_action - prefront
        radar_window = float(row["radar_window_ms"])
        radar_prepare = float(row["radar_prepare_ms"])
        rgb_convert = float(row["rgb_convert_ms"])
        scene_snapshot = float(row["scene_snapshot_ms"])
        residual = prefront - radar_window - radar_prepare - rgb_convert - scene_snapshot
        require(capture_to_action >= -0.01, "capture-to-action interval is negative")
        require(callback_to_worker >= -0.01, "callback-to-worker interval is negative")
        require(residual >= -0.01, "pre-front residual is negative")
        values["capture_to_action_ms"].append(max(0.0, capture_to_action))
        values["callback_to_worker_ms"].append(max(0.0, callback_to_worker))
        values["radar_window_ms"].append(radar_window)
        values["radar_prepare_ms"].append(radar_prepare)
        values["rgb_convert_ms"].append(rgb_convert)
        values["evaluation_snapshot_ms"].append(scene_snapshot)
        values["prefront_residual_ms"].append(max(0.0, residual))
        # Operational wait is diagnostic, not additive to capture-based age:
        # its start can precede the RGB capture event used as time zero.
        values["sensor_wait_nonadditive_ms"].append(float(row["sensor_wait_ms"]))
    return values


def pareto_ids(rows: Sequence[Mapping[str, Any]], x_key: str) -> set[int]:
    points = [
        (float(row[x_key]), float(row["combined_quality"]), int(row["action_id"]))
        for row in rows
        if row.get(x_key) not in (None, "")
    ]
    frontier: set[int] = set()
    best_quality = -math.inf
    for latency, quality, action_id in sorted(points):
        if quality > best_quality + 1e-12:
            frontier.add(action_id)
            best_quality = quality
    return frontier


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_quality_latency(rows: Sequence[Mapping[str, Any]], output: Path) -> list[str]:
    configure_plot()
    names: list[str] = []
    for number, stage in enumerate(SCATTER_STAGES, start=1):
        label = STAGES[stage]
        key = f"{stage}_p50_ms"
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharey=True)
        for ax, profile in zip(axes.flat, PROFILE_ORDER):
            selected = [row for row in rows if row["network_profile"] == profile]
            frontier = pareto_ids(selected, key)
            for family, color in FAMILY_COLOR.items():
                group = [row for row in selected if row["family"] == family and row[key] != ""]
                ax.scatter(
                    [float(row[key]) for row in group],
                    [100.0 * float(row["combined_quality"]) for row in group],
                    s=30,
                    alpha=0.78,
                    c=color,
                    edgecolors="white",
                    linewidths=0.4,
                    label=family,
                )
            for row in selected:
                if int(row["action_id"]) in frontier and row[key] != "":
                    ax.annotate(
                        str(row["action_id"]),
                        (float(row[key]), 100.0 * float(row["combined_quality"])),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=7,
                        fontweight="bold",
                    )
            ax.set_title(PROFILE_LABEL[profile])
            ax.set_xlabel(f"{label} P50 (ms)")
            ax.set_ylabel("Combined validation quality (%)")
            ax.grid(alpha=0.25)
            ax.tick_params(axis="both", labelsize=9, width=1.2)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(f"Validation quality vs {label.lower()} by network profile", y=0.995)
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        name = f"{number:02d}_quality_vs_{stage}_p50"
        save_figure(fig, output / name)
        names.extend([name + ".png", name + ".pdf"])
    return names


def action_balanced_latency(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        selected = [row for row in rows if row["network_profile"] == profile]
        for stage, label in STAGES.items():
            record: dict[str, Any] = {
                "network_profile": profile,
                "stage": stage,
                "stage_label": label,
                "actions_total": len(selected),
            }
            for probability in PERCENTILES:
                suffix = int(probability * 100)
                values = [
                    float(row[f"{stage}_p{suffix}_ms"])
                    for row in selected
                    if row[f"{stage}_p{suffix}_ms"] != ""
                ]
                record[f"action_balanced_p{suffix}_ms"] = percentile(values, 0.50)
                record[f"actions_with_p{suffix}"] = len(values)
            output.append(record)
    return output


def plot_latency_bars(rows: Sequence[Mapping[str, Any]], output: Path) -> list[str]:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey=False)
    components = ("pure_front", "network", "edge_map")
    colors = ("#4C78A8", "#F58518", "#54A24B")
    x = np.arange(len(components))
    width = 0.23
    for ax, profile in zip(axes.flat, PROFILE_ORDER):
        selected = {row["stage"]: row for row in rows if row["network_profile"] == profile}
        for offset, suffix in enumerate((50, 95, 99)):
            heights = [float(selected[stage][f"action_balanced_p{suffix}_ms"]) for stage in components]
            bars = ax.bar(x + (offset - 1) * width, heights, width, color=colors[offset], label=f"P{suffix}")
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2, fontweight="bold")
        ax.set_xticks(x, [STAGES[stage] for stage in components])
        ax.set_ylabel("Latency (ms)")
        ax.set_title(PROFILE_LABEL[profile])
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Action-balanced latency percentiles by causal stage", y=0.995)
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    name = "05_latency_percentiles_by_profile"
    save_figure(fig, output / name)
    return [name + ".png", name + ".pdf"]


def sensor_profile_rows(cell_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stages = (
        "callback_to_worker",
        "radar_window",
        "radar_prepare",
        "rgb_convert",
        "evaluation_snapshot",
        "prefront_residual",
        "capture_to_action",
        "sensor_wait_nonadditive",
    )
    result: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        selected = [row for row in cell_rows if row["network_profile"] == profile]
        for stage in stages:
            record: dict[str, Any] = {
                "network_profile": profile,
                "stage": stage,
                "actions_total": len(selected),
            }
            for suffix in (50, 95, 99):
                values = [float(row[f"{stage}_p{suffix}_ms"]) for row in selected]
                record[f"action_balanced_p{suffix}_ms"] = percentile(values, 0.50)
            result.append(record)
    return result


def plot_sensor_breakdown(rows: Sequence[Mapping[str, Any]], output: Path) -> list[str]:
    configure_plot()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), sharey=True)
    stages = (
        "callback_to_worker",
        "radar_window",
        "radar_prepare",
        "rgb_convert",
        "evaluation_snapshot",
        "prefront_residual",
    )
    labels = ("Callback→worker", "Radar window", "Radar prepare", "RGB convert", "Eval snapshot", "Residual")
    x = np.arange(len(stages))
    width = 0.23
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for ax, profile in zip(axes.flat, PROFILE_ORDER):
        selected = {row["stage"]: row for row in rows if row["network_profile"] == profile}
        for offset, suffix in enumerate((50, 95, 99)):
            heights = [float(selected[stage][f"action_balanced_p{suffix}_ms"]) for stage in stages]
            ax.bar(x + (offset - 1) * width, heights, width, color=colors[offset], label=f"P{suffix}")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(PROFILE_LABEL[profile])
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Retained pre-action timing breakdown", y=0.995)
    fig.legend(
        handles,
        legend_labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    name = "06_sensor_preparation_retained_breakdown"
    save_figure(fig, output / name)
    return [name + ".png", name + ".pdf"]


def make_report(
    rows: Sequence[Mapping[str, Any]],
    latency_rows: Sequence[Mapping[str, Any]],
    sensor_rows: Sequence[Mapping[str, Any]],
    bridge_audit: Mapping[str, Any],
) -> str:
    by_stage = {
        (row["network_profile"], row["stage"]): row for row in latency_rows
    }
    lines = [
        "# SplitFusion latency, quality, and sensor analysis",
        "",
        "This report uses the corrected direct edge-to-map counterfactual. It is",
        "an offline causal replay, not a live remeasurement of the 288 cells.",
        "",
        "## Main interpretation",
        "",
        "- Validation quality is action-dependent and therefore repeats across network profiles; latency and survival move with the network.",
        "- A point is absent from an edge/total plot when that cell produced no useful map installation. Absence is not encoded as zero latency.",
        "- The combined score is a provisional presentation coordinate, not the PPO reward.",
        "- Physical map installation ends at the edge-host map. Compact controller feedback to the UE is a later observation and is excluded from map AoI.",
        "",
        "## Action-balanced stage percentiles",
        "",
        "| Profile | Stage | P50 (ms) | P95 (ms) | P99 (ms) | Actions represented |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile in PROFILE_ORDER:
        for stage in ("pure_front", "ue_action", "network", "edge_map", "total"):
            row = by_stage[(profile, stage)]
            lines.append(
                f"| {PROFILE_LABEL[profile]} | {STAGES[stage]} | "
                f"{float(row['action_balanced_p50_ms']):.1f} | "
                f"{float(row['action_balanced_p95_ms']):.1f} | "
                f"{float(row['action_balanced_p99_ms']):.1f} | "
                f"{int(row['actions_with_p50'])}/72 |"
            )
    lines.extend(
        [
            "",
            "Percentile aggregation is action-balanced: each displayed value is the median of the corresponding per-action cell percentile. It is not a pooled-frame percentile dominated by high-throughput actions.",
            "",
            "## Sensor preparation boundary",
            "",
            "The existing 288 evidence supports a trustworthy breakdown only to the retained function boundaries: callback-to-worker scheduling, radar-window extraction, aggregate radar preparation, RGB conversion, and evaluation-snapshot capture. The inner radar projection/tracking/rasterization functions were not individually timed, so this report does not invent them.",
            "",
            "Camera and radar callbacks are distinct CARLA callback threads, but the numerical RGB conversion and radar-window/raster preparation used by one frame run sequentially inside the single `route-b-split-front` worker. Evaluation is already dispatched separately after its immutable scene snapshot is captured.",
            "",
            "`sensor_wait_ms` is shown only as a non-additive operational diagnostic because that wait can begin before the RGB capture event used as the action-age boundary.",
            "",
            "## Clock and denominator integrity",
            "",
            f"The same-host clock bridge used {int(bridge_audit['anchor_count']):,} anchors; its absolute error P99 was {float(bridge_audit['absolute_deviation_ms_p99']):.6f} ms.",
            "Every stage carries its own count. Pure model-front and complete UE action timing use all sent frames; feature-uplink timing uses only retained observed complete edge receipts (never imputed arrivals); edge and total timing use useful direct-map installations.",
            "",
            "## Quality definition",
            "",
            "$$",
            "Q_{\\mathrm{loc}}=\\sqrt{\\mathrm{IoU}_{\\mathrm{vehicle}}\\,\\mathrm{IoU}_{\\mathrm{person}}},",
            "\\qquad",
            "Q_{\\mathrm{joint}}=\\sqrt{mIoU_{\\mathrm{seg}}\\,Q_{\\mathrm{loc}}}.",
            "$$",
            "",
            "All terms are frozen validation metrics in $[0,1]$. The geometric mean is conservative: an action cannot appear strong merely because one quality dimension hides a weak one.",
            "",
            "## Figure guide",
            "",
            "Figures 01–04 show all available action/profile points and label Pareto-frontier action IDs. Figure 05 gives the requested P50/P95/P99 bars for model front, feature uplink and edge-to-map service. Figure 06 exposes the retained sensor-preparation boundaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output exists: {output}")
    direct_manifest = json.loads((DIRECT_ROOT / "artifact_manifest.json").read_text())
    for name, expected in direct_manifest["sha256"].items():
        require(sha256(DIRECT_ROOT / name) == expected, f"direct artifact hash drift: {name}")
    direct_rows = read_csv(DIRECT_ROOT / "direct_map_288_cell_summary.csv")
    direct_index = {row["cell_id"]: row for row in direct_rows}
    require(len(direct_index) == 288, "direct counterfactual is not 288 cells")

    cells = source._read_csv(source.CONSOLIDATION / "campaign_288_cell_table.csv")
    bridge_ns, bridge_audit, bridge_hashes = timing.collect_clock_bridge(cells)
    final_v3._verify_final_sources()
    calibration, publication_samples = final_v3._final_calibration()
    direct_samples, delay_provenance = direct.load_direct_delay_samples(LIVE_DIRECT_ROOT)
    pooling = direct.choose_pooling(direct_samples)
    quality = source._quality_by_action()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        pool_hashes,
    ) = source._build_empirical_pools(cells)
    require(pool_hashes == bridge_hashes, "per-frame source hash verification drift")

    rows: list[dict[str, Any]] = []
    sensor_cell_rows: list[dict[str, Any]] = []
    observed_network_intervals = 0
    for number, cell in enumerate(cells, start=1):
        attempt = source._attempt(cell)
        sent = sorted(source._sent_rows(attempt), key=lambda row: float(row["capture_wall_s"]))
        family = cell["family"]
        frames, counters = source._candidate_frames(
            cell=cell,
            rows=sent,
            family_calibration=calibration[family],
            publication_samples=publication_samples[family],
            action_service_pool=action_services.get(int(cell["action_id"]), ()),
            family_service_pool=family_services[family],
            profile_delay_pool=profile_delays[cell["network_profile"]],
            action_profile_arrival_pool=action_profile_arrivals.get((int(cell["action_id"]), cell["network_profile"]), ()),
            profile_arrival_pool=profile_arrivals[cell["network_profile"]],
        )
        delay_pool = direct.delay_pool_for(family, direct_samples, pooling)
        frames, _ = direct._direct_frames(frames, cell_id=cell["cell_id"], delay_pool=delay_pool)
        publication_ns = int(round(calibration[family]["optimized_publication_ms_median"] * 1e6))
        total_ns = int(round(calibration[family]["optimized_total_edge_processing_ms_median"] * 1e6))
        result = simulate_two_stage(
            frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=None,
                processing_horizon_ns=source.HORIZON_NS,
                service_target_ns=source.SERVICE_TARGET_NS,
                predicted_compute_ns=max(1, total_ns - publication_ns),
                predicted_publication_ns=publication_ns,
                predicted_post_publication_install_ns=int(statistics.median(delay_pool)),
            ),
        )
        summary = result.summary()
        reference = direct_index[cell["cell_id"]]
        require(int(summary["ack_installed_frames"]) == int(reference["ack_installed_frames"]), f"{cell['cell_id']}: direct installed count drift")

        ue_action_ms: list[float] = []
        pure_front_ms: list[float] = []
        for row in sent:
            ue_action_ms.append((int(row["ue_prepare_finished_ns"]) - int(row["capture_started_ns"])) / 1e6)
            parsed = ast.literal_eval(row["front_timing_ns"])
            if finite(parsed.get("front_backbone")) is not None:
                pure_front_ms.append(float(parsed["front_backbone"]) / 1e6)
        network_ms: list[float] = []
        for frame, row in zip(frames, sent):
            # The published replay imputes enough arrivals to reproduce the
            # measured aggregate admission counts. Those synthetic arrivals
            # are necessary for scheduling, but are not network measurements.
            # Plot only the retained, same-clock-bridge edge receipt here.
            if frame.arrival_ns is None or not row.get("edge_receipt_wall_s"):
                continue
            start_ns = int(row["ue_prepare_finished_ns"]) + bridge_ns
            value = (int(frame.arrival_ns) - start_ns) / 1e6
            require(value >= -0.001, f"{cell['cell_id']}: causal uplink interval is negative")
            network_ms.append(max(0.0, value))
        observed_network_intervals += len(network_ms)
        edge_map_ms: list[float] = []
        total_ms: list[float] = []
        for item in useful_outcomes(result):
            edge_map_ms.append((int(item.install_ns) - int(item.frame.arrival_ns)) / 1e6)
            row = sent[int(item.frame.sequence_id)]
            action_start_ns = int(row["capture_started_ns"]) + bridge_ns
            total_ms.append((int(item.install_ns) - action_start_ns) / 1e6)

        qloc, qjoint = quality_score(quality[int(cell["action_id"])])
        record: dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "action_id": int(cell["action_id"]),
            "profile_id": cell["profile_id"],
            "network_profile": cell["network_profile"],
            "family": family,
            "quantizer": cell["quantizer"],
            "q": float(cell["q"]),
            "median_payload_bytes": percentile([float(row["payload_bytes"]) for row in sent], 0.50),
            **quality[int(cell["action_id"])],
            "localization_overlap": qloc,
            "combined_quality": qjoint,
            "rate_reassembled_per_sent": int(counters["measured_reassemblies"]) / len(sent),
            "rate_admitted_per_sent": int(counters["measured_edge_admissions"]) / len(sent),
            "rate_installed_per_sent": int(summary["ack_installed_frames"]) / len(sent),
            "network_latency_observed_only": True,
            **stats(ue_action_ms, "ue_action"),
            **stats(pure_front_ms, "pure_front"),
            **stats(network_ms, "network"),
            **stats(edge_map_ms, "edge_map"),
            **stats(total_ms, "total"),
        }
        # CSV uses an empty string, rather than a fabricated zero, for an absent stage.
        record = {key: "" if value is None else value for key, value in record.items()}
        rows.append(record)

        components = sensor_components(sent, bridge_ns)
        sensor_record: dict[str, Any] = {
            "cell_id": cell["cell_id"],
            "action_id": int(cell["action_id"]),
            "network_profile": cell["network_profile"],
        }
        for name, values in components.items():
            sensor_record.update(stats(values, name.removesuffix("_ms")))
        sensor_cell_rows.append(sensor_record)
        if number % 24 == 0:
            print(f"supervisor analysis: {number}/288 cells", flush=True)

    require(len(rows) == 288 and len(sensor_cell_rows) == 288, "analysis inventory drift")
    latency_rows = action_balanced_latency(rows)
    sensor_rows = sensor_profile_rows(sensor_cell_rows)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "action_profile_quality_latency.csv", rows)
    write_csv(output / "latency_percentiles_by_profile.csv", latency_rows)
    write_csv(output / "sensor_preparation_cell_percentiles.csv", sensor_cell_rows)
    write_csv(output / "sensor_preparation_profile_percentiles.csv", sensor_rows)
    figure_names = []
    figure_names.extend(plot_quality_latency(rows, output))
    figure_names.extend(plot_latency_bars(latency_rows, output))
    figure_names.extend(plot_sensor_breakdown(sensor_rows, output))
    atomic_json(
        output / "analysis_summary.json",
        {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "scientific_status": "OFFLINE_COUNTERFACTUAL_NOT_LIVE_REMEASUREMENT",
            "inventory": {"cells": 288, "actions": 72, "profiles": 4},
            "source_bindings": {
                "builder_sha256": sha256(Path(__file__).resolve()),
                "direct_counterfactual_manifest_sha256": sha256(DIRECT_ROOT / "artifact_manifest.json"),
                "direct_live_validation_manifest_sha256": sha256(LIVE_DIRECT_ROOT / "artifact_manifest.json"),
                "campaign_cell_table_sha256": sha256(source.CONSOLIDATION / "campaign_288_cell_table.csv"),
            },
            "component_boundaries": {
                "ue_action": "capture_started_ns to ue_prepare_finished_ns",
                "pure_front": "front_timing_ns.front_backbone only",
                "network": "ue_prepare_finished_ns to complete edge reassembly",
                "edge_map": "complete edge reassembly to direct spatial-map install",
                "total": "capture_started_ns to direct spatial-map install",
                "physical_map_aoi_excludes_controller_feedback": True,
            },
            "quality": {
                "localization_overlap": "sqrt(vehicle_iou * person_box_mask_iou)",
                "combined_quality": "sqrt(segmentation_miou * localization_overlap)",
                "role": "provisional presentation coordinate, not PPO reward",
            },
            "denominators": {
                "ue_action": "all sent frames",
                "pure_front": "all sent frames",
                "network": "frames with retained observed complete edge receipt",
                "edge_map": "useful direct-map installations",
                "total": "useful direct-map installations",
            },
            "clock_bridge": bridge_audit,
            "observed_network_intervals": observed_network_intervals,
            "imputed_arrivals_excluded_from_network_latency": True,
            "direct_delay_provenance": delay_provenance,
            "direct_delay_pooling": pooling,
            "sensor_threading_audit": {
                "carla_callbacks": "camera and radar callbacks are distinct",
                "numerical_preparation": "radar and RGB preparation are sequential in one route-b-split-front worker",
                "evaluation": "separate bounded evaluation worker after immutable snapshot capture",
                "inner_radar_limitation": "not separately timed in retained 288 evidence",
            },
        },
    )
    atomic_text(output / "REPORT.md", make_report(rows, latency_rows, sensor_rows, bridge_audit))
    primary = (
        "action_profile_quality_latency.csv",
        "latency_percentiles_by_profile.csv",
        "sensor_preparation_cell_percentiles.csv",
        "sensor_preparation_profile_percentiles.csv",
        "analysis_summary.json",
        "REPORT.md",
        *figure_names,
    )
    atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "status": "COMPLETE", "sha256": {name: sha256(output / name) for name in primary}},
    )
    atomic_text(output / TERMINAL, TERMINAL + "\n")
    return {"output": str(output), "figures": figure_names}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv).output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
