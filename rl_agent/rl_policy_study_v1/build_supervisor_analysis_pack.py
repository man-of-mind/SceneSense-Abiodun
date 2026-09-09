#!/usr/bin/env python3
"""Build a presentation-ready, evidence-bound SplitFusion analysis pack.

The plots deliberately keep three evidence domains separate:

* Phase-13C localhost software timing (no sensor capture or radio),
* frozen validation quality, and
* live 288-cell OAI/CARLA delivery and AoI.

No missing one-way uplink latency is estimated and no aggregate cell table is
treated as an on-policy RL trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


EXPECTED_SHA256 = {
    "action_summary": "250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88",
    "action_profile_summary": "280b372ed8b6a52bb3e5f1f69e6cbdc02dc851a6a597565fcab9ca9652f5998e",
    "cell_table": "3f3067d4c9d0ef3c0d2661c3d19d04306bcbd18ef6d21fa426148e0ee24d2e0d",
    "analysis_summary": "48358c2f27ffc2f3da8bf0f821c733914a03267f50da7adab4cbf70cbe239690",
    "localhost_summary": "8021cada5cc10d70a8125bd2327e491f30bd2e32c744fd559cca56cc934c7031",
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
FAMILIES = ("noAE", "AE128", "AE64", "AE32")
FAMILY_COLORS = {
    "noAE": "#3366CC",
    "AE128": "#DC3912",
    "AE64": "#109618",
    "AE32": "#990099",
}
QUANTIZERS = ("UINT8", "UINT6", "UINT4")
QUANT_MARKERS = {"UINT8": "o", "UINT6": "s", "UINT4": "^"}
QUANT_COLORS = {"UINT8": "#1f77b4", "UINT6": "#ff7f0e", "UINT4": "#2ca02c"}
SHORTLIST = (12, 44, 49, 50, 51, 66, 69, 70, 71)


class PackError(RuntimeError):
    """Raised when an input or output violates this analysis contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected:
        raise PackError(f"{label} sha256 mismatch: {observed} != {expected}")
    return resolved


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    if path.suffix != ".png":
        raise PackError(f"figure path must end in .png: {path}")
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "SplitFusion personal analysis pack v2"},
    )
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "SplitFusion personal analysis pack v2",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def family_quant_legend(ax: plt.Axes) -> None:
    family_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    quant_handles = [
        Line2D(
            [0],
            [0],
            marker=QUANT_MARKERS[q],
            linestyle="",
            markerfacecolor="none",
            color="#333333",
            label=q,
        )
        for q in QUANTIZERS
    ]
    first = ax.legend(
        handles=family_handles,
        title="Feature family",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(first)
    ax.legend(
        handles=quant_handles,
        title="Quantizer",
        loc="lower right",
        frameon=True,
    )


def annotate_actions(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str) -> None:
    for _, row in frame[frame["action_id"].isin(SHORTLIST)].iterrows():
        if pd.notna(row[x]) and pd.notna(row[y]):
            ax.annotate(
                f"a{int(row['action_id'])}",
                (row[x], row[y]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
                color="#222222",
            )


def scatter_by_action(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    annotate: bool = False,
) -> None:
    for family in FAMILIES:
        for quantizer in QUANTIZERS:
            part = frame[
                (frame["family"] == family) & (frame["quantizer"] == quantizer)
            ]
            ax.scatter(
                part[x],
                part[y],
                c=FAMILY_COLORS[family],
                marker=QUANT_MARKERS[quantizer],
                s=38,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.45,
            )
    if annotate:
        annotate_actions(ax, frame, x, y)


def plot_localhost(local: pd.DataFrame, output: Path) -> None:
    local = local.copy()
    local["payload_kib"] = local["scientific_inner_bytes_median"] / 1024.0
    local["q"] = local["q_e4"] / 10000.0
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for ax, y, title, ylabel in (
        (
            axes[0],
            "capture_to_edge_result_ms_median",
            "Localhost end-to-end software time",
            "Median time (ms)",
        ),
        (
            axes[1],
            "ue_prepare_ms_median",
            "UE dispatch preparation",
            "Median UE preparation (ms)",
        ),
        (
            axes[2],
            "measured_sequential_throughput_frames_per_second",
            "Sequential software throughput",
            "Frames/s",
        ),
    ):
        scatter_by_action(ax, local, "payload_kib", y, annotate=True)
        ax.set_xscale("log")
        ax.set_xlabel("Feature payload (KiB, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    family_quant_legend(axes[0])
    fig.suptitle(
        "36×300 localhost control: compression reduces software time, but this is not radio latency",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output / "01_localhost_payload_vs_software_latency.png")


def plot_quality(action: pd.DataFrame, output: Path) -> None:
    frame = action.copy()
    frame["payload_kib"] = (
        frame["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    )
    panels = (
        ("val_vehicle_f1", "Vehicle F1"),
        ("val_person_avo_f1", "AVO-person F1"),
        ("val_foreground_miou", "Foreground mIoU"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharex=True)
    for ax, (metric, label) in zip(axes, panels):
        scatter_by_action(ax, frame, "payload_kib", metric, annotate=True)
        ax.set_xscale("log")
        ax.set_xlabel("Feature payload (KiB, log scale)")
        ax.set_ylabel(label)
        ax.set_title(label + " vs payload")
    family_quant_legend(axes[0])
    fig.suptitle(
        "Compression–quality frontier: bit width is comparatively cheap; extreme q is not",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output / "02_payload_vs_validation_quality.png")


def plot_q_effect(action: pd.DataFrame, output: Path) -> None:
    frame = action.copy()
    frame["payload_kib"] = (
        frame["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
    for ax, family in zip(axes.flat, FAMILIES):
        part = frame[frame["family"] == family]
        quality_ax = ax.twinx()
        for quantizer in QUANTIZERS:
            qpart = part[part["quantizer"] == quantizer].sort_values("q")
            ax.plot(
                qpart["q"],
                qpart["payload_kib"],
                color=QUANT_COLORS[quantizer],
                marker=QUANT_MARKERS[quantizer],
                linewidth=2,
                label=f"{quantizer} payload",
            )
            quality_ax.plot(
                qpart["q"],
                qpart["val_person_avo_f1"],
                color=QUANT_COLORS[quantizer],
                linestyle="--",
                alpha=0.66,
                linewidth=1.4,
            )
        ax.set_yscale("log")
        ax.set_title(family)
        ax.set_ylabel("Payload (KiB, log)")
        quality_ax.set_ylabel("AVO-person F1 (dashed)")
        quality_ax.set_ylim(0.28, 0.75)
        ax.set_xlabel("Dropped-channel fraction q")
        if family == "noAE":
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "q is the strongest knob: payload falls smoothly, while quality collapses at q=.90/.98",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output / "03_q_sweep_payload_and_person_quality.png")


def plot_live_delivery(action_profile: pd.DataFrame, output: Path) -> None:
    frame = action_profile.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median"] / 1024.0
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True, sharey=True)
    for ax, network in zip(axes.flat, NETWORKS):
        part = frame[frame["network_profile"] == network]
        scatter_by_action(ax, part, "payload_kib", "rate_installed_per_sent", annotate=True)
        ax.set_xscale("log")
        ax.set_ylim(-0.025, 0.72)
        ax.set_title(NETWORK_LABELS[network])
        ax.set_xlabel("Feature payload (KiB, log scale)")
        ax.set_ylabel("Installed / sent")
    family_quant_legend(axes.flat[0])
    fig.suptitle(
        "OAI/CARLA: payload strongly controls delivery, and the boundary shifts with channel process",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output / "04_live_payload_vs_delivery_by_network.png")


def plot_transport_residual(action_profile: pd.DataFrame, output: Path) -> None:
    frame = action_profile.copy()
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median"] / 1024.0
    latency = "s5_derived_roundtrip_residual_ms_RESIDUAL__median"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    finite = frame[pd.to_numeric(frame[latency], errors="coerce").notna()].copy()

    network_colors = {
        "FAVORABLE_STABLE": "#4C78A8",
        "MID_VARIABLE": "#F58518",
        "ADVERSE_STABLE": "#E45756",
        "FADE_RECOVERY": "#54A24B",
    }
    for network in NETWORKS:
        part = finite[finite["network_profile"] == network].sort_values("action_id")
        axes[0].plot(
            part["action_id"],
            part[latency],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=network_colors[network],
            label=NETWORK_LABELS[network],
        )
        axes[1].scatter(
            part["payload_kib"],
            part[latency],
            s=34,
            alpha=0.8,
            color=network_colors[network],
            edgecolors="white",
            linewidths=0.4,
            label=NETWORK_LABELS[network],
        )
    axes[0].set_xlabel("Action ID")
    axes[0].set_ylabel("Transport residual (ms)")
    axes[0].set_title("Action ordering")
    for boundary in (17.5, 35.5, 53.5):
        axes[0].axvline(boundary, color="#777777", linewidth=0.8, alpha=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Feature payload (KiB, log scale)")
    axes[1].set_ylabel("Transport residual (ms)")
    axes[1].set_title("Payload relationship")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.98,
        0.03,
        "Uplink + edge queue + compact downlink\nOne-way uplink is not separately available",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )
    fig.suptitle(
        "Action versus compatible-clock transport residual (delivering cells only)",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output / "05_action_vs_transport_residual.png")


def plot_heatmap(action: pd.DataFrame, output: Path) -> None:
    ordered = action.sort_values("action_id")
    values = np.array(
        [
            [row[f"rate_installed_per_sent__{network}"] for network in NETWORKS]
            for _, row in ordered.iterrows()
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.5, 14))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=0.70)
    ax.set_xticks(range(4), [NETWORK_LABELS[n] for n in NETWORKS], rotation=18, ha="right")
    ticks = list(range(0, 72, 3)) + [71]
    ax.set_yticks(ticks, [f"a{i:02d}" for i in ticks])
    ax.set_ylabel("Action ID (family → quantizer → q catalog order)")
    for boundary in (17.5, 35.5, 53.5):
        ax.axhline(boundary, color="white", linewidth=1.8)
    for start, family in zip((0, 18, 36, 54), FAMILIES):
        ax.text(
            4.13,
            start + 8.5,
            family,
            va="center",
            color=FAMILY_COLORS[family],
            fontweight="bold",
        )
    cbar = fig.colorbar(image, ax=ax, pad=0.10)
    cbar.set_label("Installed / sent")
    ax.set_title("All 72 actions × 4 network profiles", fontweight="bold")
    save_figure(fig, output / "06_action_network_delivery_heatmap.png")


def weighted_network_table(
    cell_table: pd.DataFrame, action_profile: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for network in NETWORKS:
        part = cell_table[cell_table["network_profile"] == network]
        profile_part = action_profile[action_profile["network_profile"] == network]
        sent = int(part["frames_sent"].sum())
        installed = int(part["maps_installed"].sum())
        ack = int(part["ack_within_500ms_timeout"].sum())
        reassembled = int(part["edge_complete_reassemblies"].sum())
        datagrams_tx = int(part["ue_feature_datagrams_transmitted"].sum())
        datagrams_rx = int(part["edge_feature_datagrams_received"].sum())
        rows.append(
            {
                "network_profile": network,
                "frames_sent": sent,
                "maps_installed": installed,
                "installed_per_sent": installed / sent,
                "ack_500ms_per_sent": ack / sent,
                "reassembled_per_sent": reassembled / sent,
                "datagram_rx_per_tx": datagrams_rx / datagrams_tx,
                "zero_delivery_actions": int((part["maps_installed"] == 0).sum()),
                "transport_residual_ms_median_of_cells": float(
                    profile_part[
                        "s5_derived_roundtrip_residual_ms_RESIDUAL__median"
                    ].median(skipna=True)
                ),
                "action_path_ms_median_of_cells": float(
                    profile_part[
                        "analytic_model_front_transport_back_feedback_ms__median"
                    ].median(skipna=True)
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_network_summary(network: pd.DataFrame, output: Path) -> None:
    labels = [NETWORK_LABELS[n] for n in network["network_profile"]]
    x = np.arange(len(labels))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for index, (column, label, color) in enumerate(
        (
            ("datagram_rx_per_tx", "Datagram rx/tx", "#4C78A8"),
            ("reassembled_per_sent", "Complete frame/sent", "#F58518"),
            ("installed_per_sent", "Installed/sent", "#54A24B"),
            ("ack_500ms_per_sent", "ACK≤500 ms/sent", "#B279A2"),
        )
    ):
        axes[0].bar(x + (index - 1.5) * width, network[column], width, label=label, color=color)
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(0.0, 0.82)
    axes[0].set_ylabel("Campaign-wide rate")
    axes[0].set_title("Channel process changes every survival stage")
    axes[0].legend(ncol=2)
    bars = axes[1].bar(
        x, network["transport_residual_ms_median_of_cells"], color="#E45756"
    )
    axes[1].axhline(100.0, color="#222222", linestyle="--", linewidth=1.2, label="100 ms reference")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Transport residual (ms)")
    axes[1].set_title("Compatible-clock network-path comparison")
    for bar, zero in zip(bars, network["zero_delivery_actions"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{int(zero)} zero\nactions",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].legend()
    fig.suptitle("Network-profile behavior is structured, not noise", fontsize=14, fontweight="bold")
    save_figure(fig, output / "07_network_profile_aggregate.png")


def plot_latency_boundaries(
    summary: dict, action_profile: pd.DataFrame, output: Path
) -> None:
    stages = dict(summary["aggregates"]["stage_medians"])
    selected = (
        ("UE front path (`front_ms`)", "s3_front_ms_FULL_UE_DISPATCH_SPAN"),
        ("Transport residual", "s5_derived_roundtrip_residual_ms_RESIDUAL"),
        ("Model back (`frozen_tail`)", "s8_frozen_tail_ms"),
    )
    values = []
    for label, key in selected:
        record = stages[key]
        values.append(float(record["median_of_medians"]))
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    y = np.arange(len(selected))
    colors = ["#4C78A8", "#F58518", "#B279A2"]
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y, [item[0] for item in selected])
    ax.invert_yaxis()
    ax.axvline(
        100.0,
        color="#111111",
        linestyle="--",
        linewidth=1.3,
        label="100 ms service reference",
    )
    ax.set_xlabel("Median of per-cell medians (ms)")
    ax.set_title("Action-controlled latency breakdown (sensor preparation excluded)", fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(value + 4, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")
    sensor_median = float(stages["s1_pre_front_compute_ms"]["median_of_medians"])
    sensor_p95 = float(action_profile["s1_pre_front_compute_ms__p95"].median())
    ax.text(
        0.98,
        0.04,
        (
            f"Sensor preparation excluded: median {sensor_median:.1f} ms; "
            f"median cell p95 {sensor_p95:.1f} ms\n"
            "Transport residual = uplink + edge queue + compact downlink; "
            "one-way uplink unavailable"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )
    ax.legend()
    save_figure(fig, output / "08_action_controlled_latency_breakdown.png")


def markdown_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                text = "—"
            elif column in formats:
                text = formats[column].format(value)
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_shortlist(
    action: pd.DataFrame, action_profile: pd.DataFrame
) -> pd.DataFrame:
    selected = action[action["action_id"].isin(SHORTLIST)].copy()
    selected["action"] = "a" + selected["action_id"].astype(int).astype(str)
    selected["configuration"] = (
        selected["family"] + "/" + selected["quantizer"] + "/q=" + selected["q"].map(lambda x: f"{x:.2f}")
    )
    selected["payload_KiB"] = (
        selected["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    )
    for network in NETWORKS:
        selected[NETWORK_LABELS[network]] = selected[f"rate_installed_per_sent__{network}"]
    selected["worst_install_rate"] = selected[
        [NETWORK_LABELS[n] for n in NETWORKS]
    ].min(axis=1)
    latency_rows = []
    for action_id in SHORTLIST:
        part = action_profile[action_profile["action_id"] == action_id]
        transport = part[
            "s5_derived_roundtrip_residual_ms_RESIDUAL__median"
        ].dropna()
        action_path = part[
            "analytic_model_front_transport_back_feedback_ms__median"
        ].dropna()
        latency_rows.append(
            {
                "action_id": action_id,
                "transport_residual_range_ms": (
                    f"{transport.min():.0f}–{transport.max():.0f}"
                    if len(transport)
                    else "—"
                ),
                "action_path_range_ms": (
                    f"{action_path.min():.0f}–{action_path.max():.0f}"
                    if len(action_path)
                    else "—"
                ),
            }
        )
    selected = selected.merge(pd.DataFrame(latency_rows), on="action_id", how="left")
    return selected[
        [
            "action",
            "configuration",
            "payload_KiB",
            "val_vehicle_f1",
            "val_person_avo_f1",
            "val_person_avo_recall_0_30m",
            "val_foreground_miou",
            "val_absolute_service_gates_passed",
            "val_localization_gates_passed",
            "val_segmentation_installable",
            *[NETWORK_LABELS[n] for n in NETWORKS],
            "worst_install_rate",
            "transport_residual_range_ms",
            "action_path_range_ms",
        ]
    ].sort_values("action", key=lambda s: s.str[1:].astype(int))


def build_q_effects(action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = action.set_index(["family", "quantizer", "q"])
    for q in sorted(action["q"].unique()):
        ratios = []
        vehicle_delta = []
        person_delta = []
        miou_delta = []
        part = action[action["q"] == q]
        for _, row in part.iterrows():
            base = indexed.loc[(row["family"], row["quantizer"], 0.0)]
            ratios.append(
                row["live_scientific_inner_bytes_median__across_profile_mean"]
                / base["live_scientific_inner_bytes_median__across_profile_mean"]
            )
            vehicle_delta.append(row["val_vehicle_f1"] - base["val_vehicle_f1"])
            person_delta.append(row["val_person_avo_f1"] - base["val_person_avo_f1"])
            miou_delta.append(row["val_foreground_miou"] - base["val_foreground_miou"])
        rows.append(
            {
                "q": q,
                "median_payload_ratio_vs_matched_q0": float(np.median(ratios)),
                "median_delta_vehicle_f1_vs_q0": float(np.median(vehicle_delta)),
                "median_delta_person_avo_f1_vs_q0": float(np.median(person_delta)),
                "median_delta_foreground_miou_vs_q0": float(np.median(miou_delta)),
                **{
                    f"mean_install_rate_{network}": float(
                        part[f"rate_installed_per_sent__{network}"].mean()
                    )
                    for network in NETWORKS
                },
            }
        )
    return pd.DataFrame(rows)


def build_quantizer_effects(action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = action.set_index(["family", "quantizer", "q"])
    for quantizer in QUANTIZERS:
        ratios = []
        deltas = {"vehicle": [], "person": [], "miou": []}
        for _, row in action[action["quantizer"] == quantizer].iterrows():
            base = indexed.loc[(row["family"], "UINT8", row["q"])]
            ratios.append(
                row["live_scientific_inner_bytes_median__across_profile_mean"]
                / base["live_scientific_inner_bytes_median__across_profile_mean"]
            )
            deltas["vehicle"].append(row["val_vehicle_f1"] - base["val_vehicle_f1"])
            deltas["person"].append(row["val_person_avo_f1"] - base["val_person_avo_f1"])
            deltas["miou"].append(row["val_foreground_miou"] - base["val_foreground_miou"])
        rows.append(
            {
                "quantizer": quantizer,
                "median_payload_ratio_vs_matched_UINT8": float(np.median(ratios)),
                "median_delta_vehicle_f1_vs_UINT8": float(np.median(deltas["vehicle"])),
                "median_delta_person_avo_f1_vs_UINT8": float(np.median(deltas["person"])),
                "median_delta_foreground_miou_vs_UINT8": float(np.median(deltas["miou"])),
            }
        )
    return pd.DataFrame(rows)


def build_family_effects(action: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = action.set_index(["family", "quantizer", "q"])
    for family in FAMILIES:
        ratios = []
        vehicle = []
        person = []
        miou = []
        for _, row in action[action["family"] == family].iterrows():
            base = indexed.loc[("noAE", row["quantizer"], row["q"])]
            ratios.append(
                row["live_scientific_inner_bytes_median__across_profile_mean"]
                / base["live_scientific_inner_bytes_median__across_profile_mean"]
            )
            vehicle.append(row["val_vehicle_f1"] - base["val_vehicle_f1"])
            person.append(row["val_person_avo_f1"] - base["val_person_avo_f1"])
            miou.append(row["val_foreground_miou"] - base["val_foreground_miou"])
        rows.append(
            {
                "family": family,
                "median_payload_ratio_vs_matched_noAE": float(np.median(ratios)),
                "median_delta_vehicle_f1_vs_noAE": float(np.median(vehicle)),
                "median_delta_person_avo_f1_vs_noAE": float(np.median(person)),
                "median_delta_foreground_miou_vs_noAE": float(np.median(miou)),
            }
        )
    return pd.DataFrame(rows)


def zero_delivery_boundaries(action: pd.DataFrame) -> pd.DataFrame:
    frame = action.copy()
    frame["payload_KiB"] = (
        frame["live_scientific_inner_bytes_median__across_profile_mean"] / 1024.0
    )
    rows = []
    for network in NETWORKS:
        rate = f"rate_installed_per_sent__{network}"
        positive = frame[frame[rate] > 0]
        zero = frame[frame[rate] == 0]
        rows.append(
            {
                "network_profile": network,
                "positive_delivery_actions": len(positive),
                "zero_delivery_actions": len(zero),
                "largest_payload_with_delivery_KiB": float(positive["payload_KiB"].max()),
                "smallest_zero_delivery_payload_KiB": float(zero["payload_KiB"].min()),
                "spearman_payload_vs_install_rate": float(
                    frame["payload_KiB"].corr(frame[rate], method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def write_interpretation(
    output: Path,
    network: pd.DataFrame,
    q_effects: pd.DataFrame,
    quantizer: pd.DataFrame,
    family: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> None:
    q_fmt = {
        "q": "{:.2f}",
        "median_payload_ratio_vs_matched_q0": "{:.3f}",
        "median_delta_vehicle_f1_vs_q0": "{:+.3f}",
        "median_delta_person_avo_f1_vs_q0": "{:+.3f}",
        "median_delta_foreground_miou_vs_q0": "{:+.3f}",
        **{f"mean_install_rate_{n}": "{:.3f}" for n in NETWORKS},
    }
    quant_fmt = {c: ("{:.3f}" if "ratio" in c else "{:+.3f}") for c in quantizer.columns if c != "quantizer"}
    family_fmt = {c: ("{:.3f}" if "ratio" in c else "{:+.3f}") for c in family.columns if c != "family"}
    network_fmt = {
        "installed_per_sent": "{:.3f}",
        "ack_500ms_per_sent": "{:.3f}",
        "reassembled_per_sent": "{:.3f}",
        "datagram_rx_per_tx": "{:.3f}",
        "transport_residual_ms_median_of_cells": "{:.1f}",
        "action_path_ms_median_of_cells": "{:.1f}",
    }
    boundary_fmt = {
        "largest_payload_with_delivery_KiB": "{:.1f}",
        "smallest_zero_delivery_payload_KiB": "{:.1f}",
        "spearman_payload_vs_install_rate": "{:.3f}",
    }
    text = f"""# SplitFusion evidence interpretation for policy design

This pack joins three **separate** evidence domains: frozen model validation,
36×300 localhost software timing, and the 288-cell live OAI/CARLA campaign. It
does not manufacture the unavailable one-way feature-uplink latency and does
not treat the 288 aggregate table as an on-policy PPO trajectory.

## Discussion in five figures

1. `01_localhost_payload_vs_software_latency.pdf`: the clean software control.
   Smaller features reduce UE preparation and improve sequential throughput,
   but this is localhost—not radio latency.
2. `02_payload_vs_validation_quality.pdf`: compression versus validated model
   quality. UINT4 often cuts bytes sharply with very little change from the
   matched UINT8 action; high q is the expensive quality decision.
3. `04_live_payload_vs_delivery_by_network.pdf`: the radio frontier. The
   payload at which delivery collapses shifts substantially by channel process.
4. `05_action_vs_transport_residual.pdf`: action and payload against the
   compatible-clock residual containing uplink, edge queue and compact-result
   downlink. It is deliberately not called one-way uplink latency.
5. `06_action_network_delivery_heatmap.pdf`: all 288 cells without averaging
   away action-by-network interactions.

Use `07_network_profile_aggregate.pdf` and
`08_action_controlled_latency_breakdown.pdf` as appendix slides.

## What each compression knob is doing

### q, matched against q=0 within family and quantizer

{markdown_table(q_effects, q_fmt)}

The median is over the 12 matched family/quantizer branches. q gives a smooth
payload ladder, but q=.90/.98 are emergency actions because person and
segmentation quality fall sharply. The controller should therefore learn when
freshness debt justifies that quality sacrifice; high q should not be declared
technically invalid.

### Quantizer, matched against UINT8 within family and q

{markdown_table(quantizer, quant_fmt)}

The median is over 24 matched family/q branches. The low-bit quantizers are the
cheapest compression lever in quality terms: the byte reduction is large while
median F1/mIoU changes are near zero. This argues for learning quantizer and q
as separate categorical decisions rather than folding them into one opaque ID.

### AE family, matched against noAE within quantizer and q

{markdown_table(family, family_fmt)}

The AE bottleneck is a second, distinct payload lever. AE32 is smallest, but it
has a clearer person/segmentation penalty than AE64/AE128. Family therefore
belongs in the policy hierarchy instead of being inferred from payload alone.

## Network process changes the feasible action region

{markdown_table(network, network_fmt)}

{markdown_table(boundaries, boundary_fmt)}

The boundary values are descriptive, not hard capacity estimates: there is one
route realization per action/profile and the adverse profile has a small overlap
between the largest successful and smallest zero-delivery payload. Still, the
Spearman relationship is strong in every profile. A policy must observe causal
channel/history features; a global static action is leaving substantial quality
or delivery performance on the table.

## Reward and constraint interpretation

Use the measurements in different roles:

| Evidence | Correct role | Incorrect role |
| --- | --- | --- |
| Validation quality | action prior / emulator quality consequence after a successful install | automatic reward merely because an action was selected |
| Installed/sent and reassembly | stochastic transition and map-update probability | technical invalid-action mask |
| Install AoI | post-outcome map freshness cost | one-way uplink latency |
| 500 ms ACK | accounting/observability timeout | service-success target |
| 100 ms reference | normalization and reporting reference | binary reward (it is zero for all cells) |
| Payload/datagrams/airtime | radio resource cost | perception quality |
| Emergency tier | learned quality trade-off / optional safety shield | corruption or impossible action |

A useful constrained objective remains

```text
map-state utility
  − λ_air × radio resource cost
  − λ_compute × compute cost
  − λ_switch × action switching cost,
```

with separate cost critics for coverage debt and localization/freshness debt.
When delivery fails, the old map ages; the action receives **no** credit for the
quality it would have produced. Do not separately award a full delivery bonus
and a full AoI bonus if both encode the same state improvement.

For action comparison, the presentation excludes sensor preparation and uses
the measured UE front path, compatible-clock transport residual, frozen model
back, and compact map-feedback terms. Sensor preparation remains reported as a
separate workload fact. The transport residual includes uplink, edge queueing
and compact-result downlink because true one-way uplink is not recoverable from
the registered clock domains.

## Training readiness

The policy architecture can be instantiated and smoke-tested immediately. A
scientifically valid PPO run should begin only after four short data-engineering
gates are closed:

1. freeze the seven causal observation reducers and their decision cutoff;
2. audit which fields are genuinely UE-visible online;
3. materialize a per-decision transition table from the retained per-frame data;
4. fit/validate an action-conditional stochastic emulator using temporal blocks
   and independent trace seeds.

The 288 aggregate table alone is not a PPO trajectory: each route held one
action fixed, behavior probabilities are absent, and switching counterfactuals
were not measured. Starting PPO directly from the 288 rows would therefore be
fast but scientifically wrong.
"""
    (output / "INTERPRETATION_AND_REWARD_GUIDE.md").write_text(text, encoding="utf-8")


def write_presentation_guide(output: Path) -> None:
    text = """# Suggested discussion presentation order

## Slide 1 — Experimental surface

- 72 actions = 4 feature families × 3 quantizers × 6 q values.
- 4 Markov/Gaussian channel processes = 288 live CARLA/OAI cells.
- Keep frozen validation quality, localhost timing and live network outcomes
  visibly separate.

## Slide 2 — What the knobs buy

Show `02_payload_vs_validation_quality.pdf` and
`03_q_sweep_payload_and_person_quality.pdf`.

Message: lower bit width is the low-cost compression lever; AE width changes
both payload and representation quality; q creates the strongest compression
ladder and the clearest emergency-quality penalty.

## Slide 3 — Software control

Show `01_localhost_payload_vs_software_latency.pdf`.

Message: smaller payloads improve a radio-free 36×300 control, so the later live
patterns are not explained solely by model compute. Do not call localhost
delivery radio latency.

## Slide 4 — Action × network interaction

Show `04_live_payload_vs_delivery_by_network.pdf` and, if room permits,
`06_action_network_delivery_heatmap.pdf`.

Message: the feasible payload region contracts from favorable to adverse. This
is the empirical justification for an adaptive policy rather than one fixed
compression setting.

## Slide 5 — Latency scope

Show `05_action_vs_transport_residual.pdf` and use
`08_action_controlled_latency_breakdown.pdf` to define the action-controlled
front, transport-residual, back and feedback terms.

Message: one-way feature-uplink latency is unavailable. The compatible-clock
transport residual is the fair action comparison, but it also contains edge
queueing and the compact downlink. Sensor preparation is reported separately.

## Slide 6 — Initial policy implication

- Keep all 72 actions technically available.
- Start split-only with a recurrent, factorized categorical PPO policy.
- Let the policy trade post-install map utility against radio, compute and
  switching costs.
- Treat coverage and localization/freshness debt as constrained costs.
- Use high-q actions only when channel/freshness conditions justify their model
  quality loss.
- Defer LOCAL/SKIP and continuous q until their transitions are measured.

## Slide 7 — What is ready today

- Architecture/formulation and measured action surface: ready.
- Network instantiation and synthetic smoke: can start today.
- Full PPO training: wait until causal reducers, UE-visible state audit,
  per-decision transitions and emulator validation are complete.
"""
    (output / "PRESENTATION_GUIDE.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-summary", type=Path, required=True)
    parser.add_argument("--action-profile-summary", type=Path, required=True)
    parser.add_argument("--cell-table", type=Path, required=True)
    parser.add_argument("--analysis-summary", type=Path, required=True)
    parser.add_argument("--localhost-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        "action_summary": require_file(
            args.action_summary, EXPECTED_SHA256["action_summary"], "action summary"
        ),
        "action_profile_summary": require_file(
            args.action_profile_summary,
            EXPECTED_SHA256["action_profile_summary"],
            "action/profile summary",
        ),
        "cell_table": require_file(
            args.cell_table, EXPECTED_SHA256["cell_table"], "288-cell table"
        ),
        "analysis_summary": require_file(
            args.analysis_summary,
            EXPECTED_SHA256["analysis_summary"],
            "analysis summary",
        ),
        "localhost_summary": require_file(
            args.localhost_summary,
            EXPECTED_SHA256["localhost_summary"],
            "localhost summary",
        ),
    }
    output = args.output.resolve(strict=False)
    if output.exists():
        raise PackError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    action = pd.read_csv(inputs["action_summary"])
    action_profile = pd.read_csv(inputs["action_profile_summary"])
    cell_table = pd.read_csv(inputs["cell_table"])
    local = pd.read_csv(inputs["localhost_summary"])
    summary = json.loads(inputs["analysis_summary"].read_text(encoding="utf-8"))
    if (
        len(action) != 72
        or len(action_profile) != 288
        or len(cell_table) != 288
        or len(local) != 36
    ):
        raise PackError(
            "unexpected row counts: "
            f"{len(action)}, {len(action_profile)}, {len(cell_table)}, {len(local)}"
        )
    if sorted(action["action_id"].astype(int).tolist()) != list(range(72)):
        raise PackError("action inventory is not exactly 0..71")
    if set(action_profile["network_profile"]) != set(NETWORKS):
        raise PackError("network-profile inventory differs from the frozen contract")
    if int(action_profile["installed_within_100ms_service_reference"].sum()) != 0:
        raise PackError("the 100-ms outcome no longer matches the consolidated evidence")

    configure_style()
    plot_localhost(local, output)
    plot_quality(action, output)
    plot_q_effect(action, output)
    plot_live_delivery(action_profile, output)
    plot_transport_residual(action_profile, output)
    plot_heatmap(action, output)
    network = weighted_network_table(cell_table, action_profile)
    plot_network_summary(network, output)
    plot_latency_boundaries(summary, action_profile, output)

    shortlist = build_shortlist(action, action_profile)
    q_effects = build_q_effects(action)
    quantizer = build_quantizer_effects(action)
    family = build_family_effects(action)
    boundaries = zero_delivery_boundaries(action)
    for name, frame in (
        ("action_shortlist.csv", shortlist),
        ("network_profile_summary.csv", network),
        ("q_matched_effects.csv", q_effects),
        ("quantizer_matched_effects.csv", quantizer),
        ("family_matched_effects.csv", family),
        ("zero_delivery_boundaries.csv", boundaries),
    ):
        frame.to_csv(output / name, index=False, lineterminator="\n")

    shortlist_formats = {
        "payload_KiB": "{:.1f}",
        "val_vehicle_f1": "{:.3f}",
        "val_person_avo_f1": "{:.3f}",
        "val_person_avo_recall_0_30m": "{:.3f}",
        "val_foreground_miou": "{:.3f}",
        **{NETWORK_LABELS[n]: "{:.3f}" for n in NETWORKS},
        "worst_install_rate": "{:.3f}",
    }
    (output / "ACTION_SHORTLIST.md").write_text(
        "# Evidence-based discussion shortlist\n\n"
        "This is a discussion aid, not a pruned action catalog. All 72 actions "
        "remain technically valid.\n\n"
        + markdown_table(shortlist, shortlist_formats),
        encoding="utf-8",
    )
    write_interpretation(output, network, q_effects, quantizer, family, boundaries)
    write_presentation_guide(output)

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "schema": "scenesense.splitfusion.personal_analysis_pack.v2",
        "status": "COMPLETE",
        "source_sha256": {key: sha256_file(path) for key, path in inputs.items()},
        "source_paths": {key: str(path) for key, path in inputs.items()},
        "row_counts": {
            "actions": len(action),
            "action_network_cells": len(action_profile),
            "reconciled_campaign_cells": len(cell_table),
            "localhost_profiles": len(local),
        },
        "scientific_limits": [
            "ONE_WAY_FEATURE_UPLINK_LATENCY_UNAVAILABLE_NOT_ESTIMATED",
            "ZERO_100MS_INSTALLS_BINARY_100MS_REWARD_UNINFORMATIVE",
            "VALIDATION_QUALITY_IS_NOT_CREDITED_WHEN_DELIVERY_FAILS",
            "AGGREGATE_288_TABLE_IS_NOT_A_PPO_TRAJECTORY",
            "ONE_ROUTE_REALIZATION_PER_ACTION_NETWORK_CELL_NO_REPLICATE_UNCERTAINTY",
        ],
        "artifacts": artifacts,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SPLITFUSION_PERSONAL_ANALYSIS_PACK_COMPLETE")
    print(json.dumps({"output": str(output), "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
