#!/usr/bin/env python3
"""Build the final presentation pack with the calibrated 288-cell simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import build_pack as base


ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATION = ROOT / (
    "experiments/splitfusion_288_offline_rl_dataset_v1/"
    "20260909_offline_consolidation_v1"
)
BASELINE = ROOT / (
    "experiments/splitfusion_timing_diagnostic_v1/"
    "20260909_live_carla_actions30_15_50_71_retry3"
)
SIMULATION = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260911_final_v3_predicted_horizon_288"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_rl_policy_design_v1/"
    "20260911_288_results_final_simulated_pack_v3"
)
FINAL_TIMING_PATHS = {
    30: ROOT / (
        "experiments/splitfusion_edge_optimization_v3/"
        "20260910_live_v2_vs_v3_actions30_50_71_retry1/"
        "action30__v3_overlapped_final/action_summary.csv"
    ),
    15: ROOT / (
        "experiments/splitfusion_edge_optimization_v3/"
        "20260910_live_v2_vs_v3_actions30_15_50_71/"
        "action15__v2_predicted_install_horizon/action_summary.csv"
    ),
    50: ROOT / (
        "experiments/splitfusion_edge_optimization_v3/"
        "20260910_live_v2_vs_v3_actions30_50_71_retry1/"
        "action50__v3_overlapped_final/action_summary.csv"
    ),
    71: ROOT / (
        "experiments/splitfusion_edge_optimization_v3/"
        "20260910_live_v2_vs_v3_actions30_50_71_retry1/"
        "action71__v3_overlapped_final/action_summary.csv"
    ),
}
EXPECTED = {
    "action": base.EXPECTED_SHA256["action"],
    "action_profile": base.EXPECTED_SHA256["action_profile"],
    "cell": base.EXPECTED_SHA256["cell"],
    "baseline_timing": base.EXPECTED_SHA256["baseline_timing"],
    "sim_manifest": "a0f8705833a64000d272fa2e1f81e06263856ccc0cb5930cdacd0c65580ac981",
    "sim_result": "2dff447678c8e265d175d7e8335bcf9f0e6492e28d296e0f207c5f32a33a81ba",
    "sim_cells": "a39b5f03a9282176616f2b99b03972deedeec46a026a0e268d58ea621097ba1c",
    "final_timing_30": "6840b80401693888cbefaabb97d22553810bb822d8b26dc3b81ae82c00dee858",
    "final_timing_15": "857a75b0364d87be387d2a8f6dc32f2366ae663df18eec4aa269dd308fdfe4fe",
    "final_timing_50": "56e0315bc30a7271ea80b0569e20cff53c8939d7b6c73345ff93182d89a448b3",
    "final_timing_71": "d58daa801144abd951f271ec8c2d467eefdab39cb8c759e22d78312bfd887c59",
}
QUALITY_PANELS = (
    ("val_vehicle_f1", "Vehicle F1"),
    ("val_person_avo_f1", "Person F1"),
    ("val_foreground_miou", "Segmentation mIoU"),
)
LOCALIZATION_PANELS = (
    ("val_vehicle_xy_mae_m", "Vehicle XY MAE (m)"),
    ("val_person_avo_xy_mae_m", "Person XY MAE (m)"),
    ("val_vehicle_iou", "Vehicle localization IoU"),
    ("val_person_box_mask_iou", "Person box-mask IoU"),
)


def _paths() -> dict[str, Path]:
    paths = {
        "action": base.require(
            CONSOLIDATION / "action_72_summary.csv", EXPECTED["action"], "action summary"
        ),
        "action_profile": base.require(
            CONSOLIDATION / "action_72x4_summary.csv",
            EXPECTED["action_profile"],
            "action/profile summary",
        ),
        "cell": base.require(
            CONSOLIDATION / "campaign_288_cell_table.csv", EXPECTED["cell"], "cell table"
        ),
        "baseline_timing": base.require(
            BASELINE / "action_summary.csv",
            EXPECTED["baseline_timing"],
            "original timing summary",
        ),
        "sim_manifest": base.require(
            SIMULATION / "artifact_manifest.json", EXPECTED["sim_manifest"], "simulator manifest"
        ),
        "sim_result": base.require(
            SIMULATION / "counterfactual_results.json", EXPECTED["sim_result"], "simulator result"
        ),
        "sim_cells": base.require(
            SIMULATION / "final_v3_288_cell_summary.csv", EXPECTED["sim_cells"], "simulator cells"
        ),
    }
    for action_id, path in FINAL_TIMING_PATHS.items():
        key = f"final_timing_{action_id}"
        paths[key] = base.require(path, EXPECTED[key], f"final timing action {action_id}")
    manifest = json.loads(paths["sim_manifest"].read_text(encoding="utf-8"))
    for name, key in (
        ("counterfactual_results.json", "sim_result"),
        ("final_v3_288_cell_summary.csv", "sim_cells"),
    ):
        observed = manifest["sha256"].get(name)
        if observed != EXPECTED[key]:
            raise base.PresentationError(f"simulator manifest binding drift: {name}")
    return paths


def _joined(action_profile: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    sim = simulated[
        [
            "cell_id",
            "action_id",
            "network_profile",
            "rate_installed_per_sent",
            "ack_installed_frames",
            "install_aoi_ms_median",
            "time_weighted_map_aoi_ms",
            "edge_target_variant",
        ]
    ].rename(
        columns={
            "action_id": "sim_action_id",
            "network_profile": "sim_network_profile",
            "rate_installed_per_sent": "sim_rate_installed_per_sent",
            "ack_installed_frames": "sim_ack_installed_frames",
            "install_aoi_ms_median": "sim_install_aoi_ms_median",
            "time_weighted_map_aoi_ms": "sim_time_weighted_map_aoi_ms",
        }
    )
    frame = action_profile.merge(sim, on="cell_id", how="inner", validate="one_to_one")
    if len(frame) != 288:
        raise base.PresentationError(f"action/profile simulator join has {len(frame)} rows")
    if not (frame["action_id"] == frame["sim_action_id"]).all():
        raise base.PresentationError("simulator action identity mismatch")
    if not (frame["network_profile"] == frame["sim_network_profile"]).all():
        raise base.PresentationError("simulator profile identity mismatch")
    frame["payload_kib"] = frame["live_scientific_inner_bytes_median"] / 1024.0
    return frame


def _facet_scatter(axis: plt.Axes, frame: pd.DataFrame, metric: str) -> None:
    for family in base.FAMILIES:
        for quantizer in base.QUANTIZERS:
            part = frame[
                (frame["family"] == family) & (frame["quantizer"] == quantizer)
            ]
            axis.scatter(
                part["payload_kib"],
                part[metric],
                s=54.0,
                c=base.FAMILY_COLORS[family],
                marker=base.QUANT_MARKERS[quantizer],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
            )
    base.annotate_actions(axis, frame, "payload_kib", metric)


def _figure_legend(fig: plt.Figure, y: float) -> None:
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=family)
        for family, color in base.FAMILY_COLORS.items()
    ]
    handles += [
        Line2D(
            [0],
            [0],
            marker=base.QUANT_MARKERS[quantizer],
            linestyle="",
            markerfacecolor="none",
            color="#333333",
            label=quantizer,
        )
        for quantizer in base.QUANTIZERS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        bbox_to_anchor=(0.5, y),
    )


def plot_profile_quality(frame: pd.DataFrame, output: Path) -> None:
    for letter, network in zip("abcd", base.NETWORKS):
        part = frame[frame["network_profile"] == network]
        fig, axes = plt.subplots(1, 3, figsize=(16.6, 6.2))
        fig.subplots_adjust(left=0.055, right=0.99, top=0.79, bottom=0.22, wspace=0.16)
        for axis, (metric, ylabel) in zip(axes, QUALITY_PANELS):
            _facet_scatter(axis, part, metric)
            base.payload_axis(axis)
            axis.set_ylabel(ylabel)
            axis.set_title(ylabel)
            axis.set_ylim(0.25, 1.0)
        _figure_legend(fig, 0.01)
        fig.suptitle(
            f"Measured payload versus frozen validation quality — "
            f"{base.NETWORK_LABELS[network]}",
            fontsize=15,
            fontweight="bold",
            y=0.97,
        )
        base.save(
            fig,
            output,
            f"01{letter}_{network.lower()}_payload_vs_validation_quality",
        )


def plot_profile_localization(frame: pd.DataFrame, output: Path) -> None:
    for letter, network in zip("abcd", base.NETWORKS):
        part = frame[frame["network_profile"] == network]
        fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.7))
        fig.subplots_adjust(
            left=0.07,
            right=0.99,
            top=0.84,
            bottom=0.14,
            hspace=0.31,
            wspace=0.16,
        )
        for axis, (metric, ylabel) in zip(axes.flat, LOCALIZATION_PANELS):
            _facet_scatter(axis, part, metric)
            base.payload_axis(axis)
            axis.set_ylabel(ylabel)
            axis.set_title(ylabel)
        _figure_legend(fig, 0.005)
        fig.suptitle(
            f"Measured payload versus frozen localization quality — "
            f"{base.NETWORK_LABELS[network]}",
            fontsize=15,
            fontweight="bold",
            y=0.97,
        )
        base.save(
            fig,
            output,
            f"02{letter}_{network.lower()}_payload_vs_localization_quality",
        )


def plot_sim_install(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3), constrained_layout=True)
    for axis, network in zip(axes.flat, base.NETWORKS):
        part = frame[frame["network_profile"] == network]
        base.scatter_actions(axis, part, "payload_kib", "sim_rate_installed_per_sent")
        base.payload_axis(axis)
        axis.set_ylim(-0.025, 1.025)
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.set_ylabel("Simulated map-install rate (installed / sent)")
        axis.set_title(base.NETWORK_LABELS[network])
    base.family_quant_legend(axes.flat[0])
    fig.suptitle("Final-edge simulated map installation by network profile", fontsize=15, fontweight="bold")
    base.save(fig, output, "04_payload_vs_map_install_rate")


def _sim_action_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for action_id, part in frame.groupby("action_id", sort=True):
        record = {"action_id": int(action_id)}
        for network in base.NETWORKS:
            row = part[part["network_profile"] == network]
            if len(row) != 1:
                raise base.PresentationError(f"action {action_id}/{network}: not unique")
            record[f"rate_installed_per_sent__{network}"] = float(
                row.iloc[0]["sim_rate_installed_per_sent"]
            )
        records.append(record)
    return pd.DataFrame(records)


def sim_network_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for network in base.NETWORKS:
        part = frame[frame["network_profile"] == network]
        sent = int(part["frames_sent"].sum())
        installed = int(part["sim_ack_installed_frames"].sum())
        records.append(
            {
                "network_profile": network,
                "frames_sent": sent,
                "simulated_maps_installed": installed,
                "simulated_map_install_rate": installed / sent,
                "actions_with_zero_simulated_install": int(
                    (part["sim_ack_installed_frames"] == 0).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def plot_sim_network(frame: pd.DataFrame, output: Path) -> None:
    labels = [base.NETWORK_LABELS[n] for n in frame["network_profile"]]
    colors = [base.NETWORK_COLORS[n] for n in frame["network_profile"]]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), constrained_layout=True)
    bars = axes[0].bar(x, frame["simulated_map_install_rate"], color=colors)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 0.85)
    axes[0].set_ylabel("Simulated map-install rate")
    axes[0].set_title("Successful map installation")
    for bar, value in zip(bars, frame["simulated_map_install_rate"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{value:.3f}", ha="center", fontweight="bold")
    zeros = axes[1].bar(x, frame["actions_with_zero_simulated_install"], color=colors)
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Actions with zero simulated installations (of 72)")
    axes[1].set_title("Remaining action infeasibility")
    ceiling = max(5, int(frame["actions_with_zero_simulated_install"].max()) + 4)
    axes[1].set_ylim(0, ceiling)
    for bar, value in zip(zeros, frame["actions_with_zero_simulated_install"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{int(value)}", ha="center", fontweight="bold")
    fig.suptitle("Final-edge simulator: network condition still changes the feasible actions", fontsize=15, fontweight="bold")
    base.save(fig, output, "07_network_profile_summary")


def plot_sim_aoi(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.3), constrained_layout=True)
    for axis, network in zip(axes.flat, base.NETWORKS):
        part = frame[
            (frame["network_profile"] == network)
            & frame["sim_install_aoi_ms_median"].notna()
        ]
        base.scatter_actions(axis, part, "payload_kib", "sim_install_aoi_ms_median")
        base.payload_axis(axis)
        axis.set_ylabel("Simulated median capture-to-install AoI (ms)")
        axis.set_title(f"{base.NETWORK_LABELS[network]} ({len(part)}/72 installed)")
        axis.set_ylim(0, 550)
        axis.set_yticks(np.arange(0, 551, 100))
    base.family_quant_legend(axes.flat[0])
    fig.suptitle("Final-edge simulated installed-map latency versus payload", fontsize=15, fontweight="bold")
    base.save(fig, output, "08_payload_vs_total_installed_map_aoi")


def _final_edge_stages(row: pd.Series) -> dict[str, float]:
    """Return stage fields that remain interpretable across v2/v3 timing."""
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
        # CUDA-event time keeps the FCOS computation definition stable when
        # v3 overlaps its host launch span with downstream CPU work.
        "tail_inference": float(row["decode_tail_cuda_ms_median"]),
        "postprocess_p025": float(row["post_processing_ms_median"]),
        "serialization": float(row["edge_output_serialization_ms_median"]),
    }


def _original_edge_stages(row: pd.Series) -> dict[str, float]:
    values = base.edge_stage_values(row)
    values["tail_inference"] = float(row["decode_tail_cuda_ms_median"])
    return values


def plot_final_edge_comparison(
    baseline: pd.DataFrame, final_timing: pd.DataFrame, output: Path
) -> pd.DataFrame:
    original = baseline.set_index("action_id")
    final = final_timing.set_index("action_id")
    if set(base.DIAGNOSTIC_ACTIONS) != set(final.index.astype(int)):
        raise base.PresentationError("final timing does not contain the four diagnostic actions")
    records = []
    upstream_fields = {
        "sensor_preparation": "prep_pre_front_compute_ms_median",
        "ue_split_dispatch": "ue_front_ms_median",
        "feature_uplink": "application_feature_uplink_ms_median",
    }
    for action_id in base.DIAGNOSTIC_ACTIONS:
        before_row = original.loc[action_id]
        after_row = final.loc[action_id]
        before_edge = _original_edge_stages(before_row)
        after_edge = _final_edge_stages(after_row)
        record = {
            "action_id": action_id,
            "profile_id": str(before_row["profile_id"]),
            "final_variant": "v2 live-valid" if action_id == 15 else "v3 final",
            "before_edge_processing_ms": float(before_row["edge_total_edge_processing_ms_median"]),
            "final_edge_processing_ms": float(after_row["edge_total_edge_processing_ms_median"]),
        }
        record["edge_saving_ms"] = (
            record["before_edge_processing_ms"] - record["final_edge_processing_ms"]
        )
        for name, source in upstream_fields.items():
            record[f"before_{name}_ms"] = float(before_row[source])
            record[f"final_{name}_ms"] = float(before_row[source])
        for name, value in before_edge.items():
            record[f"before_{name}_ms"] = value
        for name, value in after_edge.items():
            record[f"final_{name}_ms"] = value
        records.append(record)
    table = pd.DataFrame(records)
    stage_specs = (
        ("Sensor preparation compute", "sensor_preparation", "#72B7B2"),
        ("UE split dispatch", "ue_split_dispatch", "#4C78A8"),
        ("Feature uplink through OAI", "feature_uplink", "#F58518"),
        ("Edge queue", "edge_queue", "#E45756"),
        ("Feature reconstruction", "feature_reconstruction", "#B279A2"),
        ("FCOS CUDA inference", "tail_inference", "#54A24B"),
        ("Postprocess + p025 filter", "postprocess_p025", "#EECA3B"),
        ("Compact-result serialization", "serialization", "#9D755D"),
    )
    y = np.asarray(
        [group * 2.4 + offset for group in range(len(base.DIAGNOSTIC_ACTIONS)) for offset in (0.0, 0.82)]
    )
    labels = [
        f"A{action}  {version}"
        for action in base.DIAGNOSTIC_ACTIONS
        for version in ("Before", "Final")
    ]
    left = np.zeros(len(y), dtype=float)
    fig, axis = plt.subplots(figsize=(15.4, 9.0), constrained_layout=True)
    for label, key, color in stage_specs:
        values = np.asarray(
            [
                float(table.loc[table["action_id"] == action, f"{version}_{key}_ms"].iloc[0])
                for action in base.DIAGNOSTIC_ACTIONS
                for version in ("before", "final")
            ]
        )
        bars = axis.barh(y, values, left=left, height=0.66, color=color, label=label)
        for bar, value in zip(bars, values):
            if value >= 16:
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
    for group, action_id in enumerate(base.DIAGNOSTIC_ACTIONS):
        row = table[table["action_id"] == action_id].iloc[0]
        final_index = group * 2 + 1
        axis.text(
            left[final_index] + 4,
            y[final_index],
            f"edge −{row['edge_saving_ms']:.1f} ms",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#176B87",
        )
        if group < len(base.DIAGNOSTIC_ACTIONS) - 1:
            axis.axhline(group * 2.4 + 1.62, color="#999999", linewidth=0.7, alpha=0.35)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(left) + 45)
    axis.set_xlabel("Descriptive sum of median stage spans (ms)")
    axis.set_ylabel("Action and implementation")
    axis.set_title("Same eight stages and colors as Figure 09; upstream medians held fixed")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=True)
    fig.suptitle(
        "Before versus final edge implementation (A15 v2; A30/A50/A71 v3)",
        fontsize=15,
        fontweight="bold",
    )
    base.save(fig, output, "10_edge_optimization_before_after")
    return table


def write_reward_dictionary(output: Path) -> None:
    text = r"""# Reward formulation — symbol dictionary

The initial split-only policy uses installed-map utility change plus explicit
resource costs:

$$
r_t = (U_{t+1}-U_t)
      - \lambda_B\frac{B_t}{B_{\max}}
      - \lambda_C\frac{C_t}{C_{\max}}
      - \lambda_S\mathbf{1}[a_t \ne a_{t-1}],
$$

with

$$
U_t =
\frac{\sum_{i\in\mathcal O_t} w_i Q_{i,t}
      \exp\!\left(-A_{i,t}/\tau_i\right)}
     {\sum_{i\in\mathcal O_t} w_i + \varepsilon}.
$$

| Symbol | Meaning |
|---|---|
| $t$ | Decision epoch. |
| $a_t$ | One selected split action among the 72 registered profiles. |
| $r_t$ | Immediate reward assigned to the transition from epoch $t$ to $t+1$. |
| $U_t$ | Utility of the currently installed spatial map before the new transition. |
| $U_{t+1}-U_t$ | Actual improvement or degradation in installed-map utility; an undelivered update does not create fictitious quality credit. |
| $\mathcal O_t$ | Objects currently represented in the map utility calculation. |
| $i$ | One tracked map object. |
| $w_i$ | Application importance of object $i$, for example a larger weight for a vulnerable road user on the ego path. |
| $Q_{i,t}$ | Normalized quality/utility contribution of the installed observation for object $i$. This is distinct from the action's compression knob $q$. |
| $A_{i,t}$ | Age of information of object $i$: current time minus capture time of its newest installed observation. |
| $\tau_i$ | Freshness tolerance for object $i$; utility falls to $e^{-1}\approx0.368$ when $A_{i,t}=\tau_i$. |
| $\varepsilon$ | Small positive constant preventing division by zero when the map contains no weighted objects. |
| $B_t$ | Feature bytes actually charged to action $a_t$. |
| $B_{\max}$ | Fixed payload normalizer: the largest registered median action payload, not instantaneous channel capacity. |
| $C_t$ | Compute consumed by the action, including spent work on an intentionally superseded frame. |
| $C_{\max}$ | Fixed compute-cost normalizer. |
| $\lambda_B$ | Weight on communication cost. |
| $\lambda_C$ | Weight on compute cost. |
| $\lambda_S$ | Penalty for switching actions too frequently. |
| $\mathbf{1}[a_t\ne a_{t-1}]$ | Indicator equal to 1 when the action changes and 0 otherwise. |

The training return remains

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k},
$$

where $\gamma\in[0,1)$ controls how much future reward matters. The auxiliary
next-SNR forecast improves the recurrent representation; it does not replace
the PPO reward and never supplies future SNR to the acting policy.
"""
    (output / "REWARD_PARAMETER_DICTIONARY.md").write_text(text, encoding="utf-8")


def write_talking_points(output: Path, result: dict, network: pd.DataFrame) -> None:
    overall = result["aggregate"]["overall"]
    text = f"""# Final 288-cell presentation talking points

## What changed

The original campaign remains immutable. The final-edge simulator preserves
the measured payload, radio reassembly and edge-admission outcomes, then
causally replays the depth-one latest-only scheduler with the final edge
calibration. Therefore simulated installation and AoI are counterfactual—not
new live measurements.

Overall installed/sent rises from {overall['source_rate_installed_per_sent']:.3f}
to {overall['rate_installed_per_sent']:.3f}. Median cell capture-to-install AoI
falls from {overall['source_install_aoi_ms_cell_median']:.1f} ms to
{overall['install_aoi_ms_cell_median']:.1f} ms. Time-weighted map AoI falls from
{overall['source_time_weighted_map_aoi_ms_cell_median']:.1f} ms to
{overall['time_weighted_map_aoi_ms_cell_median']:.1f} ms.

## Figures 01a–01d and 02a–02d

The validation and localization scores are frozen action-level measurements;
they were not measured four times and are not medians across network profiles.
Each network profile now has its own readable page: three quality panels in
Figure 01 and four localization panels in Figure 02. Each page uses that
profile's measured payload. Every action uses the same marker size: no
simulated installation or AoI value is encoded in these two figures. The
network-dependent counterfactual outcomes begin at Figure 04.

## Figure 03

Unchanged: q controls the intrinsic payload/quality tradeoff. Solid lines are
payload; dashed lines are person F1.

## Figure 04

This is simulated installed/sent after final edge optimization. The profile
rates are {', '.join(f"{base.NETWORK_LABELS[row.network_profile]} {row.simulated_map_install_rate:.3f}" for row in network.itertuples())}.
The numerator is an authoritative simulated map installation; the denominator
is every measured UE-sent feature.

## Figure 05

Unchanged measured radio evidence. UDP does not retransmit. Datagram reception
means datagrams observed at the edge divided by datagrams sent; complete
reassembly requires every fragment of a feature message.

## Figures 06–07

The simulator expands the feasible action region, but the network profile still
matters. This is precisely why the agent should condition decisions on causal
channel and recent-delivery history rather than select one globally fixed
action.

## Figure 08

Shows simulated capture-to-authoritative-install AoI only for actions with at
least one installation in that profile. The former 100-ms line was removed as
requested. It remains a separately reportable reference, not the optimization
or admission horizon.

## Figure 09

Unchanged original live four-action breakdown. It explains why edge service
was optimized, but its component medians are descriptive and are not an exact
additive reconstruction of Figure 08.

## Figure 10

Uses the same eight stage labels and colors as Figure 09, with paired `Before`
and `Final` bars for each action. Sensor preparation, UE dispatch and OAI
uplink are held at their original medians because this intervention changed
only the edge path. A30/A50/A71 use final v3; A15 uses its last live-valid v2
because its v3 run failed closed on non-finite camera-aware geometry. The
stage medians are descriptive rather than an additive end-to-end identity;
the annotation beside each final bar reports the directly measured edge-total
saving.

## Agent transition

The architecture document follows the LR-ASPP/FCOS documentation style: a
portable SVG followed by the Mermaid source. The LSTM keeps a compact memory
of recent channel, delivery, map-freshness and action history. Its state feeds
the PPO policy and critics, while a separate auxiliary head learns to forecast
the next observed SNR. That forecast shapes useful temporal features during
training; future SNR is never supplied to the acting policy.
"""
    (output / "TALKING_POINTS.md").write_text(text, encoding="utf-8")


def run(output: Path) -> dict:
    paths = _paths()
    output = output.resolve(strict=False)
    if output.exists():
        raise base.PresentationError(f"create-only output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    action = pd.read_csv(paths["action"])
    action_profile = pd.read_csv(paths["action_profile"])
    cell = pd.read_csv(paths["cell"])
    baseline = pd.read_csv(paths["baseline_timing"])
    final_timing = pd.concat(
        [pd.read_csv(paths[f"final_timing_{action_id}"]) for action_id in base.DIAGNOSTIC_ACTIONS],
        ignore_index=True,
    )
    simulated = pd.read_csv(paths["sim_cells"])
    result = json.loads(paths["sim_result"].read_text(encoding="utf-8"))
    if len(action) != 72 or len(action_profile) != 288 or len(simulated) != 288:
        raise base.PresentationError("unexpected action or cell inventory")
    frame = _joined(action_profile, simulated)
    base.configure_style()
    plot_profile_quality(frame, output)
    plot_profile_localization(frame, output)
    base.plot_q_sweep(action, output)
    plot_sim_install(frame, output)
    feature = base.feature_delivery_table(action_profile, cell)
    base.plot_feature_delivery(feature, output)
    base.plot_heatmap(_sim_action_table(frame), output)
    network = sim_network_summary(frame)
    plot_sim_network(network, output)
    plot_sim_aoi(frame, output)
    base.plot_original_stages(baseline, output)
    optimization = plot_final_edge_comparison(baseline, final_timing, output)
    architecture_dir = ROOT / "rl_agent/rl_policy_study_v1"
    shutil.copy2(
        architecture_dir / "AGENT_ARCHITECTURE_V1.md",
        output / "AGENT_ARCHITECTURE_V1.md",
    )
    shutil.copy2(
        architecture_dir / "AGENT_ARCHITECTURE_V1.svg",
        output / "AGENT_ARCHITECTURE_V1.svg",
    )
    write_reward_dictionary(output)
    write_talking_points(output, result, network)
    network.to_csv(output / "network_profile_summary.csv", index=False, lineterminator="\n")
    optimization.to_csv(output / "edge_optimization_summary.csv", index=False, lineterminator="\n")
    frame[
        [
            "cell_id",
            "action_id",
            "network_profile",
            "live_scientific_inner_bytes_median",
            "sim_rate_installed_per_sent",
            "sim_ack_installed_frames",
            "sim_install_aoi_ms_median",
            "sim_time_weighted_map_aoi_ms",
            "edge_target_variant",
        ]
    ].to_csv(output / "simulated_action_profile_summary.csv", index=False, lineterminator="\n")
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": base.sha256_file(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = {
        "schema": "scenesense.splitfusion.final_simulated_presentation_pack.v3",
        "status": "COMPLETE",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: base.sha256_file(path) for name, path in paths.items()},
        "row_counts": {"actions": len(action), "action_profiles": len(frame), "simulated_cells": len(simulated)},
        "simulation_status": result["scientific_status"],
        "presentation_rules": [
            "VALIDATION_QUALITY_IS_NETWORK_INDEPENDENT_AND_FROZEN",
            "FIGURES_01_02_USE_UNIFORM_ACTION_MARKERS_WITH_NO_SIMULATION_ENCODING",
            "MEASURED_TRANSPORT_AND_REASSEMBLY_ARE_UNCHANGED",
            "FIGURES_04_06_07_08_USE_FINAL_EDGE_COUNTERFACTUAL",
            "FIGURES_03_05_09_RETAIN_MEASURED_ORIGINAL_VALUES",
            "FIGURE_08_HAS_NO_100MS_REFERENCE_LINE",
            "FIGURE_10_HAS_NO_INTERMEDIATE_OPTIMIZATION_VARIANTS",
            "FIGURE_10_REUSES_FIGURE_09_STAGE_DEFINITIONS_AND_COLORS",
            "FIGURE_10_HOLDS_NON_EDGE_STAGE_MEDIANS_FIXED",
            "NOAE_FINAL_TARGET_IS_LIVE_VALID_V2_NOT_FAILED_V3",
            "ARCHITECTURE_MD_EMBEDS_STATIC_SVG_AND_MERMAID_SOURCE",
        ],
        "artifacts": artifacts,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve(strict=False)
    manifest = run(output)
    print(
        json.dumps(
            {"output": str(output), "artifacts": len(manifest["artifacts"])},
            indent=2,
        )
    )
    print("SPLITFUSION_FINAL_SIMULATED_PRESENTATION_PACK_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
