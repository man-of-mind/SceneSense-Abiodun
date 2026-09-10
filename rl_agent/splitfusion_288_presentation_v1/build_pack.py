#!/usr/bin/env python3
"""Build a presentation-only pack from bound SplitFusion evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd


EXPECTED_SHA256 = {
    "action": "250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88",
    "action_profile": "280b372ed8b6a52bb3e5f1f69e6cbdc02dc851a6a597565fcab9ca9652f5998e",
    "cell": "3f3067d4c9d0ef3c0d2661c3d19d04306bcbd18ef6d21fa426148e0ee24d2e0d",
    "analysis": "48358c2f27ffc2f3da8bf0f821c733914a03267f50da7adab4cbf70cbe239690",
    "baseline_timing": "7b4113f721dfdebc92df5c6fc110c359c779fc14d9d87cfa65cf59c07bf58176",
    "baseline_result": "98a807e69c73190a7157dfb32293928f15aba23d3d0f634bd5af3104cb4299ab",
    "optimized_timing": "bdc3b59894b7eb649ad9f80de731ad0af8a43249d585eeb5f06936f377c155cf",
    "optimized_result": "5003cc70144154ba5e96995ae4f0d6533d34aad99549f0ee33dda1afe760db95",
}

NETWORKS = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
NETWORK_LABELS = {
    "FAVORABLE_STABLE": "Favorable stable",
    "MID_VARIABLE": "Mid variable",
    "ADVERSE_STABLE": "Adverse stable",
    "FADE_RECOVERY": "Fade recovery",
}
NETWORK_COLORS = {
    "FAVORABLE_STABLE": "#4C78A8",
    "MID_VARIABLE": "#F58518",
    "ADVERSE_STABLE": "#E45756",
    "FADE_RECOVERY": "#54A24B",
}
FAMILIES = ("noAE", "AE128", "AE64", "AE32")
FAMILY_COLORS = {
    "noAE": "#3366CC",
    "AE128": "#DC3912",
    "AE64": "#109618",
    "AE32": "#990099",
}
QUANTIZERS = ("UINT8", "UINT6", "UINT4")
QUANT_COLORS = {"UINT8": "#1f77b4", "UINT6": "#ff7f0e", "UINT4": "#2ca02c"}
QUANT_MARKERS = {"UINT8": "o", "UINT6": "s", "UINT4": "^"}
SHORTLIST = (12, 44, 49, 50, 51, 66, 69, 70, 71)
DIAGNOSTIC_ACTIONS = (30, 15, 50, 71)
DIAGNOSTIC_LABELS = {
    30: "A30\nAE128 / UINT4 / q=0",
    15: "A15\nnoAE / UINT4 / q=.70",
    50: "A50\nAE64 / UINT4 / q=.50",
    71: "A71\nAE32 / UINT4 / q=.98",
}
PAYLOAD_TICKS_KIB = (6, 10, 30, 100, 300, 1024, 3072)
PAYLOAD_TICK_LABELS = ("6 KiB", "10 KiB", "30 KiB", "100 KiB", "300 KiB", "1 MiB", "3 MiB")


class PresentationError(RuntimeError):
    """Raised when an evidence or plotting contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected:
        raise PresentationError(f"{label} hash mismatch: {observed} != {expected}")
    return resolved


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
        }
    )


def bold_axes(axis: plt.Axes) -> None:
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    axis.title.set_fontweight("bold")
    axis.tick_params(axis="both", width=1.1)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontweight("bold")


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for axis in fig.axes:
        bold_axes(axis)
    fig.savefig(
        output / f"{stem}.png",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "SplitFusion 288 presentation pack v6"},
    )
    fig.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "SplitFusion 288 presentation pack v6", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def payload_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(PAYLOAD_TICKS_KIB))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: PAYLOAD_TICK_LABELS[PAYLOAD_TICKS_KIB.index(int(value))] if int(value) in PAYLOAD_TICKS_KIB else "")
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(4.8, 4200)
    axis.set_xlabel("Feature payload per frame (log scale)")


def payload_y_axis(axis: plt.Axes) -> None:
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(FixedLocator(PAYLOAD_TICKS_KIB))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: PAYLOAD_TICK_LABELS[PAYLOAD_TICKS_KIB.index(int(value))] if int(value) in PAYLOAD_TICKS_KIB else "")
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_ylim(4.8, 4200)


def annotate_actions(axis: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    for _, row in frame[frame["action_id"].isin(SHORTLIST)].iterrows():
        if pd.notna(row[x]) and pd.notna(row[y]):
            axis.annotate(
                f"a{int(row['action_id'])}",
                (row[x], row[y]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#222222",
            )


def scatter_actions(axis: plt.Axes, frame: pd.DataFrame, x: str, y: str, *, annotate: bool = True) -> None:
    for family in FAMILIES:
        for quantizer in QUANTIZERS:
            part = frame[(frame["family"] == family) & (frame["quantizer"] == quantizer)]
            axis.scatter(
                part[x],
                part[y],
                c=FAMILY_COLORS[family],
                marker=QUANT_MARKERS[quantizer],
                s=40,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.45,
            )
    if annotate:
        annotate_actions(axis, frame, x, y)


def family_quant_legend(axis: plt.Axes) -> None:
    family = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=name)
        for name, color in FAMILY_COLORS.items()
    ]
    quantizer = [
        Line2D([0], [0], marker=QUANT_MARKERS[name], linestyle="", markerfacecolor="none", color="#333333", label=name)
        for name in QUANTIZERS
    ]
    first = axis.legend(handles=family, title="Feature family", loc="best", frameon=True)
    axis.add_artist(first)
    axis.legend(handles=quantizer, title="Quantizer", loc="lower right", frameon=True)


def plot_quality(action: pd.DataFrame, output: Path) -> None:
    frame = action.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    panels = (
        ("val_vehicle_f1", "Vehicle F1", "Vehicle detection F1"),
        ("val_person_avo_f1", "Person F1", "Person detection F1"),
        ("val_foreground_miou", "Segmentation mIoU", "Vehicle/person pixel-class mIoU"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), constrained_layout=True)
    for axis, (metric, ylabel, title) in zip(axes, panels):
        scatter_actions(axis, frame, "payload_kib", metric)
        payload_axis(axis)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    family_quant_legend(axes[0])
    fig.suptitle("Feature compression versus frozen validation quality", fontsize=15, fontweight="bold")
    save(fig, output, "01_payload_vs_validation_quality")


def plot_localization(action: pd.DataFrame, output: Path) -> None:
    frame = action.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    panels = (
        ("val_vehicle_xy_mae_m", "Vehicle XY MAE (m)", "Vehicle position error — lower is better"),
        ("val_person_avo_xy_mae_m", "Person XY MAE (m)", "Person position error — lower is better"),
        ("val_vehicle_iou", "Vehicle IoU", "Vehicle localization overlap — higher is better"),
        ("val_person_box_mask_iou", "Person box-mask IoU", "Person localization overlap — higher is better"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.2), constrained_layout=True)
    for axis, (metric, ylabel, title) in zip(axes.flat, panels):
        scatter_actions(axis, frame, "payload_kib", metric)
        payload_axis(axis)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    family_quant_legend(axes.flat[0])
    fig.suptitle("Feature payload versus localization quality", fontsize=15, fontweight="bold")
    save(fig, output, "02_payload_vs_localization_quality")


def plot_q_sweep(action: pd.DataFrame, output: Path) -> None:
    frame = action.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), constrained_layout=True)
    twins = []
    for axis, family in zip(axes.flat, FAMILIES):
        part = frame[frame["family"] == family]
        quality_axis = axis.twinx()
        twins.append(quality_axis)
        for quantizer in QUANTIZERS:
            qpart = part[part["quantizer"] == quantizer].sort_values("q")
            axis.plot(qpart["q"], qpart["payload_kib"], color=QUANT_COLORS[quantizer], marker=QUANT_MARKERS[quantizer], linewidth=2)
            quality_axis.plot(qpart["q"], qpart["val_person_avo_f1"], color=QUANT_COLORS[quantizer], linestyle="--", linewidth=1.6, alpha=0.72)
        payload_y_axis(axis)
        axis.set_xlabel("Dropped-channel fraction q")
        axis.set_ylabel("Feature payload per frame (solid)")
        axis.set_xticks([0.0, 0.3, 0.5, 0.7, 0.9, 0.98], ["0", ".30", ".50", ".70", ".90", ".98"])
        axis.set_title(family)
        quality_axis.set_ylabel("Person F1 (dashed)")
        quality_axis.set_ylim(0.28, 0.75)
        quality_axis.grid(False)
    quant_handles = [Line2D([0], [0], color=QUANT_COLORS[q], marker=QUANT_MARKERS[q], linewidth=2, label=q) for q in QUANTIZERS]
    meaning_handles = [
        Line2D([0], [0], color="#333333", linewidth=2, label="Payload"),
        Line2D([0], [0], color="#333333", linewidth=1.6, linestyle="--", label="Person F1"),
    ]
    fig.legend(handles=quant_handles + meaning_handles, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Increasing q reduces payload but eventually sacrifices person quality", fontsize=15, fontweight="bold")
    save(fig, output, "03_q_sweep_payload_and_person_quality")


def plot_install_rate(action_profile: pd.DataFrame, output: Path) -> None:
    frame = action_profile.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median"] / 1024.0
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3), constrained_layout=True)
    for axis, network in zip(axes.flat, NETWORKS):
        part = frame[frame["network_profile"] == network]
        scatter_actions(axis, part, "payload_kib", "rate_installed_per_sent")
        payload_axis(axis)
        axis.set_ylim(-0.025, 0.72)
        axis.set_yticks(np.arange(0.0, 0.71, 0.1))
        axis.set_ylabel("Map-install rate (ACK-installed / sent)")
        axis.set_title(NETWORK_LABELS[network])
    family_quant_legend(axes.flat[0])
    fig.suptitle("Payload controls successful map installation under every channel process", fontsize=15, fontweight="bold")
    save(fig, output, "04_payload_vs_map_install_rate")


def feature_delivery_table(action_profile: pd.DataFrame, cell: pd.DataFrame) -> pd.DataFrame:
    frame = action_profile[["cell_id", "action_id", "network_profile", "live_scientific_inner_bytes_median"]].merge(
        cell[["cell_id", "frames_sent", "edge_complete_reassemblies", "edge_feature_datagrams_received", "ue_feature_datagrams_transmitted"]],
        on="cell_id",
        validate="one_to_one",
    )
    if len(frame) != 288:
        raise PresentationError(f"feature-delivery join has {len(frame)} rows")
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median"] / 1024.0
    frame["complete_message_rate"] = frame["edge_complete_reassemblies"] / frame["frames_sent"]
    frame["datagram_receive_fraction"] = frame["edge_feature_datagrams_received"] / frame["ue_feature_datagrams_transmitted"]
    return frame


def plot_feature_delivery(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), constrained_layout=True)
    for network in NETWORKS:
        part = frame[frame["network_profile"] == network]
        for axis, metric in zip(axes, ("complete_message_rate", "datagram_receive_fraction")):
            axis.scatter(part["payload_kib"], part[metric], s=36, alpha=0.82, color=NETWORK_COLORS[network], edgecolors="white", linewidths=0.4, label=NETWORK_LABELS[network])
    for axis in axes:
        payload_axis(axis)
        axis.set_ylim(-0.025, 1.03)
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axes[0].set_ylabel("Complete feature messages / sent frames")
    axes[0].set_title("Complete application-message reassembly")
    axes[1].set_ylabel("UDP datagrams received / UDP datagrams sent")
    axes[1].set_title("Datagram reception fraction (no retransmission)")
    axes[0].legend(loc="lower left")
    axes[1].legend(loc="lower left")
    fig.suptitle("Larger fragmented features are more likely to arrive incomplete", fontsize=15, fontweight="bold")
    save(fig, output, "05_payload_vs_feature_delivery")


def plot_heatmap(action: pd.DataFrame, output: Path) -> None:
    ordered = action.sort_values("action_id")
    values = np.asarray([[row[f"rate_installed_per_sent__{network}"] for network in NETWORKS] for _, row in ordered.iterrows()], dtype=float)
    fig, axis = plt.subplots(figsize=(8.8, 13.8), constrained_layout=True)
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=0.70)
    axis.set_xticks(range(4), [NETWORK_LABELS[n] for n in NETWORKS], rotation=18, ha="right")
    ticks = list(range(0, 72, 3)) + [71]
    axis.set_yticks(ticks, [f"a{i:02d}" for i in ticks])
    axis.set_xlabel("Network profile")
    axis.set_ylabel("Action ID (family → quantizer → q catalog order)")
    for boundary in (17.5, 35.5, 53.5):
        axis.axhline(boundary, color="white", linewidth=1.7)
    for start, family in zip((0, 18, 36, 54), FAMILIES):
        axis.text(4.12, start + 8.5, family, va="center", color=FAMILY_COLORS[family], fontweight="bold")
    colorbar = fig.colorbar(image, ax=axis, pad=0.10)
    colorbar.set_label("Map-install rate (ACK-installed / sent)", fontweight="bold")
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    axis.set_title("All 72 actions × 4 network profiles", fontweight="bold")
    save(fig, output, "06_action_network_map_install_heatmap")


def network_summary(cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for network in NETWORKS:
        part = cell[cell["network_profile"] == network]
        sent = int(part["frames_sent"].sum())
        rows.append(
            {
                "network_profile": network,
                "frames_sent": sent,
                "maps_installed": int(part["maps_installed"].sum()),
                "map_install_rate": float(part["maps_installed"].sum() / sent),
                "complete_feature_rate": float(part["edge_complete_reassemblies"].sum() / sent),
                "udp_datagram_receive_fraction": float(part["edge_feature_datagrams_received"].sum() / part["ue_feature_datagrams_transmitted"].sum()),
                "actions_with_zero_map_install": int((part["maps_installed"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_network_summary(frame: pd.DataFrame, output: Path) -> None:
    labels = [NETWORK_LABELS[n] for n in frame["network_profile"]]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), constrained_layout=True)
    bars = axes[0].bar(x, frame["map_install_rate"], color=[NETWORK_COLORS[n] for n in frame["network_profile"]])
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 0.52)
    axes[0].set_yticks(np.arange(0.0, 0.51, 0.1))
    axes[0].set_ylabel("Campaign-wide map-install rate")
    axes[0].set_title("Successful map installation")
    for bar, value in zip(bars, frame["map_install_rate"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{value:.3f}", ha="center", fontweight="bold")
    zero_bars = axes[1].bar(x, frame["actions_with_zero_map_install"], color=[NETWORK_COLORS[n] for n in frame["network_profile"]])
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylim(0, 32)
    axes[1].set_yticks(np.arange(0, 31, 5))
    axes[1].set_ylabel("Actions with zero map installations (of 72)")
    axes[1].set_title("Action infeasibility grows as the channel worsens")
    for bar, value in zip(zero_bars, frame["actions_with_zero_map_install"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7, f"{int(value)}", ha="center", fontweight="bold")
    fig.suptitle("Network condition changes the feasible action region", fontsize=15, fontweight="bold")
    save(fig, output, "07_network_profile_summary")


def plot_total_aoi(action_profile: pd.DataFrame, output: Path) -> None:
    frame = action_profile.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median"] / 1024.0
    metric = "s13_capture_to_install_aoi_ms__median"
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3), constrained_layout=True)
    for axis, network in zip(axes.flat, NETWORKS):
        part = frame[(frame["network_profile"] == network) & frame[metric].notna()]
        scatter_actions(axis, part, "payload_kib", metric)
        payload_axis(axis)
        axis.set_ylim(0, 520)
        axis.set_yticks(np.arange(0, 501, 100))
        axis.axhline(100, color="#222222", linestyle="--", linewidth=1.2)
        axis.set_ylabel("Median capture-to-map-install AoI (ms)")
        axis.set_title(f"{NETWORK_LABELS[network]} ({len(part)}/72 delivered)")
    family_quant_legend(axes.flat[0])
    fig.suptitle("Installed-map latency versus payload — zero-delivery cells have no AoI point", fontsize=15, fontweight="bold")
    save(fig, output, "08_payload_vs_total_installed_map_aoi")


def plot_original_stages(timing: pd.DataFrame, output: Path) -> None:
    indexed = timing.set_index("action_id")
    stage_specs = (
        ("Sensor preparation compute", lambda r: float(r["prep_pre_front_compute_ms_median"]), "#72B7B2"),
        ("UE split dispatch", lambda r: float(r["ue_front_ms_median"]), "#4C78A8"),
        ("Feature uplink through OAI", lambda r: float(r["application_feature_uplink_ms_median"]), "#F58518"),
        ("Edge queue", lambda r: float(r["edge_queue_wait_ms_median"]), "#E45756"),
        ("Feature reconstruction", lambda r: sum(float(r[k]) for k in ("edge_zstd_decompression_ms_median", "edge_unpack_dequantize_ms_median", "edge_ae_decode_ms_median")), "#B279A2"),
        ("FCOS tail inference", lambda r: float(r["decode_tail_inference_block_ms_median"]), "#54A24B"),
        ("Postprocess + p025 filter", lambda r: float(r["post_processing_ms_median"]), "#EECA3B"),
        ("Compact-result serialization", lambda r: float(r["tail_output_serialization_ms_median"]), "#9D755D"),
    )
    fig, axis = plt.subplots(figsize=(14.6, 6.7), constrained_layout=True)
    y = np.arange(len(DIAGNOSTIC_ACTIONS))
    left = np.zeros(len(DIAGNOSTIC_ACTIONS), dtype=float)
    for label, getter, color in stage_specs:
        values = np.asarray([getter(indexed.loc[action]) for action in DIAGNOSTIC_ACTIONS])
        bars = axis.barh(y, values, left=left, color=color, label=label)
        for bar, value in zip(bars, values):
            if value >= 12:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", ha="center", va="center", fontsize=8, fontweight="bold")
        left += values
    axis.set_yticks(y, [DIAGNOSTIC_LABELS[a] for a in DIAGNOSTIC_ACTIONS])
    axis.invert_yaxis()
    axis.set_xlabel("Descriptive sum of median stage spans (ms)")
    axis.set_ylabel("Action")
    axis.set_title("Original four-action path: sensor compute → UE dispatch → uplink → edge service")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=True)
    fig.suptitle("Pre-optimization latency-stage breakdown (300 live CARLA frames per action)", fontsize=15, fontweight="bold")
    save(fig, output, "09_original_four_action_latency_breakdown")


def comparison_rows(path: Path) -> dict[int, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["action_id"]): row for row in document["comparisons"]["per_action"]}


def stage(row: dict, key: str) -> float:
    if key == "edge_queue_wait_ms":
        return float(row[key]["median"])
    return float(row["live_stage_group_medians_ms"][key])


def edge_stage_values(row: pd.Series) -> dict[str, float]:
    return {
        "edge_queue": float(row["edge_queue_wait_ms_median"]),
        "feature_reconstruction": sum(
            float(row[key])
            for key in (
                "edge_zstd_decompression_ms_median",
                "edge_unpack_dequantize_ms_median",
                "edge_ae_decode_ms_median",
            )
        ),
        "tail_inference": float(row["decode_tail_inference_block_ms_median"]),
        "postprocess_p025": float(row["post_processing_ms_median"]),
        "serialization": float(row["tail_output_serialization_ms_median"]),
    }


def plot_optimization(
    before_timing: pd.DataFrame,
    after_timing: pd.DataFrame,
    before_path: Path,
    after_path: Path,
    output: Path,
) -> pd.DataFrame:
    before = comparison_rows(before_path)
    after = comparison_rows(after_path)
    before_indexed = before_timing.set_index("action_id")
    after_indexed = after_timing.set_index("action_id")
    records = []
    for action in DIAGNOSTIC_ACTIONS:
        b = before[action]
        a = after[action]
        before_stages = edge_stage_values(before_indexed.loc[action])
        after_stages = edge_stage_values(after_indexed.loc[action])
        record = {
            "action_id": action,
            "profile_id": a["profile_id"],
            "before_edge_service_ms": float(b["deployed_tail_service_ms"]["median"]),
            "after_edge_service_ms": float(a["deployed_tail_service_ms"]["median"]),
            "direct_saving_ms": float(b["deployed_tail_service_ms"]["median"] - a["deployed_tail_service_ms"]["median"]),
        }
        for name, value in before_stages.items():
            record[f"before_{name}_ms"] = value
        for name, value in after_stages.items():
            record[f"after_{name}_ms"] = value
        records.append(record)
    table = pd.DataFrame(records)
    stage_specs = (
        ("Edge queue", "edge_queue", "#E45756"),
        ("Feature reconstruction", "feature_reconstruction", "#B279A2"),
        ("FCOS tail inference", "tail_inference", "#54A24B"),
        ("Postprocess + p025 filter", "postprocess_p025", "#EECA3B"),
        ("Compact-result serialization", "serialization", "#9D755D"),
    )
    fig, axis = plt.subplots(figsize=(14.6, 8.0), constrained_layout=True)
    y = np.asarray(
        [group * 2.4 + offset for group in range(len(DIAGNOSTIC_ACTIONS)) for offset in (0.0, 0.82)]
    )
    labels = [
        f"A{action}  {version}"
        for action in DIAGNOSTIC_ACTIONS
        for version in ("Before", "After")
    ]
    left = np.zeros(len(y), dtype=float)
    for label, key, color in stage_specs:
        values = np.asarray(
            [
                float(table.loc[table["action_id"] == action, f"{version}_{key}_ms"].iloc[0])
                for action in DIAGNOSTIC_ACTIONS
                for version in ("before", "after")
            ]
        )
        bars = axis.barh(y, values, left=left, height=0.66, color=color, label=label)
        for bar, value in zip(bars, values):
            if value >= 10:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
        left += values
    for group, action in enumerate(DIAGNOSTIC_ACTIONS):
        record = table[table["action_id"] == action].iloc[0]
        after_index = group * 2 + 1
        axis.text(
            left[after_index] + 3,
            y[after_index],
            f"direct service −{record['direct_saving_ms']:.1f} ms",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#176B87",
        )
        if group < len(DIAGNOSTIC_ACTIONS) - 1:
            axis.axhline(group * 2.4 + 1.62, color="#999999", linewidth=0.7, alpha=0.35)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(left) + 38)
    axis.set_xlabel("Descriptive sum of median edge-stage spans (ms)")
    axis.set_ylabel("Action and implementation")
    axis.set_title("Same edge stages and colors as Figure 09; sensor, UE dispatch and OAI uplink excluded")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True)
    fig.suptitle("Output-preserving edge optimization: exact stage-for-stage comparison", fontsize=15, fontweight="bold")
    save(fig, output, "10_edge_optimization_before_after")
    return table


def write_talking_points(output: Path, network: pd.DataFrame, optimization: pd.DataFrame) -> None:
    savings = optimization["direct_saving_ms"]
    text = rf"""# SplitFusion 288-cell results — presentation talking points

## Start with the experimental question

We measured 72 split-inference actions under four time-varying channel
processes. Each action chooses a feature family, a quantizer and a dropped-cell
fraction q. The presentation asks which action preserves useful perception
while producing a payload that the current channel can deliver freshly.

## Figure 01 — payload versus validation quality

- The x-axis is logarithmic and uses explicit byte units: 10 KiB to 100 KiB is
  a tenfold increase, as is 100 KiB to 1 MiB. Equal horizontal distance means
  an equal multiplicative change, not an equal number of bytes.
- Vehicle F1, person F1 and segmentation mIoU are three different tasks.
- Segmentation mIoU is the arithmetic mean of the **vehicle pixel-class IoU**
  and **person pixel-class IoU**. It is not the average of the vehicle-detection
  IoU and person box-mask IoU shown in Figure 02.
- Therefore a high-q action near 100 KiB can have vehicle F1 around 0.8,
  person F1 around 0.5 and segmentation mIoU around 0.4 without contradiction.
- Agent connection: these curves provide the quality consequence of each
  action. Low bit width is usually cheap; extreme q buys payload at a real
  person/segmentation cost.

## Figure 02 — payload versus localization quality

- XY MAE is the mean planar Euclidean distance between a matched prediction's
  world-XY centroid and the corresponding ground-truth centroid. Matching is
  class-specific and limited to 3 m; lower is better.
- Vehicle IoU and person box-mask IoU are aggregate semantic pixel-overlap
  scores, not centroid distances and not per-object detection-box IoU. The
  person ground truth is a filled projected box rather than a silhouette.
- Person box-mask IoU genuinely peaks at 0.528 across the 72 actions. Thin,
  small person regions make pixel overlap more sensitive to boundary errors,
  and the filled-box target also limits how a predicted person silhouette can
  overlap it. The registered service threshold was 0.50, not 0.60.
- Agent connection: quality reward should not use F1 alone. A small, deliverable
  action can still incur localization or mask-quality debt.

## Figure 03 — q sweep

- Solid lines are feature payload; dashed lines are person F1. The legend now
  states both line meanings explicitly.
- q is the fraction of ranked feature cells dropped. q=0 keeps all cells and
  bypasses the ranker; q=.98 retains only about two percent.
- Agent connection: q is the strongest dynamic lever. The policy should raise
  it under congestion or stale-map pressure, but avoid permanent high-q use.

## Figure 04 — payload versus map-install rate

- Map-install rate means authoritative `MAP_INSTALLED` acknowledgements divided
  by split frames sent by the UE.
- It proves the complete feature was processed, its compact result reached the
  map service, and installation was acknowledged. It does **not** mean the
  update met the separate 100 ms reference.
- The best favorable-stable action is still below 70%. This campaign used the
  pre-optimization edge path: radio loss, fragmented-message loss, latest-frame
  replacement and an edge service slower than the 100 ms arrival interval all
  reduce installations.
- Agent connection: this is the empirical action-conditional probability of
  obtaining a usable map update under each channel state.

## Figure 05 — feature delivery

- UDP did not retransmit. “UDP datagrams received / sent” is a reception
  fraction: denominator is the UE's original datagrams and numerator is what
  the edge observed.
- A feature message is useful only when all of its fragments are reassembled.
  A modest datagram loss therefore hurts large multi-datagram actions much
  more than a one-datagram action.
- Agent connection: payload size is not just an airtime cost; it changes the
  probability of complete delivery.

## Figure 06 — action × network heatmap

- Each row is one action and each column is one network process. The color is
  map-install rate.
- The structured change across columns is the reason for adaptive selection:
  the best action is conditional on channel history rather than globally fixed.

## Figure 07 — network summary

- Campaign-wide map-install rate falls from {network.iloc[0]['map_install_rate']:.3f}
  in favorable stable to {network.iloc[2]['map_install_rate']:.3f} in adverse
  stable.
- Zero-delivery actions rise from {int(network.iloc[0]['actions_with_zero_map_install'])}
  to {int(network.iloc[2]['actions_with_zero_map_install'])} of 72.
- Agent connection: network state changes the feasible action set, but a
  zero-delivery observation is a measured bad outcome—not a corrupt action.

## Figure 08 — payload versus total installed-map latency

- This is the registered end-to-end AoI from camera capture to authoritative
  map installation. It includes sensor/preparation delay, UE dispatch, radio,
  edge queue/service, compact result delivery and map installation.
- Only installed frames have an AoI. A zero-delivery action has no point and
  must receive no quality credit merely because its hypothetical output is
  accurate.
- No installed update met 100 ms. The 100 ms line is a reference borrowed from
  stringent teleoperation practice, not a claim that this full perception and
  map pipeline already meets it.
- Agent connection: AoI/map freshness—not an isolated transport timer—is the
  direct state and reward quantity.

## Figure 09 — original four-action latency breakdown

- Sensor preparation compute ends before UE split dispatch begins.
- UE split dispatch includes seven-channel input assembly, FCOS front, ranker,
  optional AE encoding, quantization, zstd and chunk preparation.
- Feature uplink is the same-clock application interval from first UE datagram
  send to complete edge reassembly through OAI.
- Edge queue is waiting after reassembly. Feature reconstruction is zstd
  decompression + unpack/dequantization + optional AE decode.
- FCOS tail inference is only the model's tail launch/completion. Postprocessing
  and p025 filtering are separate.
- The old 111.6 ms “FCOS tail service span” was a misleading name for the whole
  frozen tail adapter: model tail + camera-aware postprocessing + p025 filtering
  + segmentation construction. Pure tail inference was about 21 ms by CUDA
  events and 27–29 ms by wall timing.
- The stack is a descriptive sum of per-stage medians; medians from different
  frames are not mathematically additive. Figure 08 is the authoritative E2E
  latency.

## Figure 10 — optimization result

- Before and after now use the same edge-stage categories, colors and action
  ordering as Figure 09. Upstream sensor preparation, UE dispatch and OAI
  transport are excluded because the optimization did not change them.
- Output-preserving optimization reduced direct edge service by
  {savings.min():.1f}–{savings.max():.1f} ms across all four actions, with a
  mean saving of {savings.mean():.1f} ms.
- The direct-service saving is worker start to result publication. Edge queue
  is shown as a system consequence but is not included in that direct-service
  number.
- Perception tensors, p025 selections, segmentation labels and serialized
  records remained exact. Feature payloads and the radio bridge were unchanged.
- Camera-aware postprocessing originally decoded geometry for candidates that
  NMS later discarded; it now performs identical score/box/NMS decisions first
  and computes geometry only for survivors. Serialization originally caused
  repeated device-to-host scalar synchronizations; it now makes one aligned
  tensor transfer before constructing the same records.
- The p025 stage builds a person semantic mask, finds connected components,
  associates person boxes with them, consolidates duplicate person candidates,
  applies the locked 0.25 person threshold and calibrates vehicle scores. Its
  own saving was modest; most of the improvement came from postprocessing and
  serialization.
- Do not subtract a constant from every historical AoI. Faster service changes
  queue replacement nonlinearly. The RL simulator should retain the 288-cell
  radio/delivery evidence and replay it with the optimized service-time
  distributions.

## Transition to the agent discussion

The measurements establish three coupled consequences of an action:

1. perception and localization quality if an update succeeds;
2. payload-dependent delivery probability under the current channel;
3. resulting map freshness after queueing and processing.

That motivates a recurrent policy that observes recent channel/delivery/map
state and selects family, quantizer and q to maximize useful fresh-map utility,
not simply accuracy and not simply minimum payload.

## Compact split-action reward

Use the following display-math form in the presentation:

$$
r_t = I_t\,Q(a_t)\,\exp\!\left(-\frac{{\operatorname{{AoI}}_t}}{{\tau}}\right)
      - \lambda_B\frac{{B(a_t)}}{{B_{{\max}}}}
$$

- $I_t$ is 1 when the selected split update is installed in the spatial map
  and 0 otherwise. An undelivered prediction therefore receives no perception
  utility.
- $Q(a_t)$ is the normalized perception/localization quality associated with
  the selected action.
- $\exp(-\operatorname{{AoI}}_t/\tau)$ is a smooth freshness discount. It is
  1 for a new update, about 0.368 when AoI equals $\tau$, and about 0.135 at
  twice $\tau$. A smaller $\tau$ represents a freshness-sensitive application;
  a larger $\tau$ tolerates older map information.
- $B(a_t)$ is the selected action's feature payload. $B_{{\max}}$ is a fixed
  normalization constant, not the instantaneous network capacity. For this
  catalog it is the largest registered median feature payload: 3,580,215 bytes
  (about 3.41 MiB, action 0). Consequently $B(a_t)/B_{{\max}}$ is dimensionless
  and lies in approximately $[0,1]$.
- $\lambda_B$ controls how strongly the agent trades perception utility for
  lower communication cost.

Speaker summary: an action is valuable only if it produces an installed update;
its value then decreases smoothly as that update becomes older, while larger
feature payloads pay an explicit communication penalty.
"""
    (output / "TALKING_POINTS.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    evidence = root / "experiments/splitfusion_288_offline_rl_dataset_v1/20260909_offline_consolidation_v1"
    baseline = root / "experiments/splitfusion_timing_diagnostic_v1/20260909_live_carla_actions30_15_50_71_retry3"
    optimized = root / "experiments/splitfusion_edge_optimization_v1/20260909_live_actions30_15_50_71"
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=Path, default=evidence / "action_72_summary.csv")
    parser.add_argument("--action-profile", type=Path, default=evidence / "action_72x4_summary.csv")
    parser.add_argument("--cell", type=Path, default=evidence / "campaign_288_cell_table.csv")
    parser.add_argument("--analysis", type=Path, default=evidence / "analysis_summary.json")
    parser.add_argument("--baseline-timing", type=Path, default=baseline / "action_summary.csv")
    parser.add_argument("--baseline-result", type=Path, default=baseline / "LIVE_DIAGNOSTIC_RESULTS.json")
    parser.add_argument("--optimized-timing", type=Path, default=optimized / "action_summary.csv")
    parser.add_argument("--optimized-result", type=Path, default=optimized / "LIVE_DIAGNOSTIC_RESULTS.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "action": require(args.action, EXPECTED_SHA256["action"], "action summary"),
        "action_profile": require(args.action_profile, EXPECTED_SHA256["action_profile"], "action/profile summary"),
        "cell": require(args.cell, EXPECTED_SHA256["cell"], "cell table"),
        "analysis": require(args.analysis, EXPECTED_SHA256["analysis"], "analysis summary"),
        "baseline_timing": require(args.baseline_timing, EXPECTED_SHA256["baseline_timing"], "baseline timing"),
        "baseline_result": require(args.baseline_result, EXPECTED_SHA256["baseline_result"], "baseline result"),
        "optimized_timing": require(args.optimized_timing, EXPECTED_SHA256["optimized_timing"], "optimized timing"),
        "optimized_result": require(args.optimized_result, EXPECTED_SHA256["optimized_result"], "optimized result"),
    }
    output = args.output.resolve(strict=False)
    if output.exists():
        raise PresentationError(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    action = pd.read_csv(paths["action"])
    action_profile = pd.read_csv(paths["action_profile"])
    cell = pd.read_csv(paths["cell"])
    timing = pd.read_csv(paths["baseline_timing"])
    optimized_timing = pd.read_csv(paths["optimized_timing"])
    if len(action) != 72 or len(action_profile) != 288 or len(cell) != 288 or len(timing) != 4 or len(optimized_timing) != 4:
        raise PresentationError("unexpected evidence row count")
    if sorted(action["action_id"].astype(int)) != list(range(72)):
        raise PresentationError("action inventory is not 0..71")
    if set(action_profile["network_profile"]) != set(NETWORKS):
        raise PresentationError("network inventory mismatch")
    if int(action_profile["installed_within_100ms_service_reference"].sum()) != 0:
        raise PresentationError("100-ms campaign result changed")

    configure_style()
    plot_quality(action, output)
    plot_localization(action, output)
    plot_q_sweep(action, output)
    plot_install_rate(action_profile, output)
    feature = feature_delivery_table(action_profile, cell)
    plot_feature_delivery(feature, output)
    plot_heatmap(action, output)
    network = network_summary(cell)
    plot_network_summary(network, output)
    plot_total_aoi(action_profile, output)
    plot_original_stages(timing, output)
    optimization = plot_optimization(
        timing,
        optimized_timing,
        paths["baseline_result"],
        paths["optimized_result"],
        output,
    )

    network.to_csv(output / "network_profile_summary.csv", index=False, lineterminator="\n")
    optimization.to_csv(output / "edge_optimization_summary.csv", index=False, lineterminator="\n")
    write_talking_points(output, network, optimization)

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "scenesense.splitfusion.288_results_presentation_pack.v2",
        "status": "COMPLETE",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "row_counts": {"actions": len(action), "action_network_cells": len(action_profile), "campaign_cells": len(cell), "diagnostic_actions": len(timing)},
        "presentation_rules": [
            "LOCALHOST_CONTROL_OMITTED",
            "PAYLOAD_AXES_USE_EXPLICIT_BINARY_UNITS_AND_LOG_SCALE",
            "ALL_SUBPLOTS_SHOW_BOTH_AXIS_LABELS_AND_VALUES",
            "UDP_DATAGRAM_COUNTS_DO_NOT_IMPLY_RETRANSMISSION",
            "INSTALL_AOI_IS_CONDITIONAL_ON_SUCCESSFUL_INSTALLATION",
            "FOUR_ACTION_STAGE_MEDIANS_ARE_DESCRIPTIVE_NOT_ADDITIVE_E2E",
            "EDGE_OPTIMIZATION_DOES_NOT_CHANGE_FEATURE_TRANSPORT",
        ],
        "artifacts": artifacts,
    }
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SPLITFUSION_288_RESULTS_PRESENTATION_PACK_COMPLETE")
    print(json.dumps({"output": str(output), "artifact_count": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
