#!/usr/bin/env python3
"""Build the final presentation pack with the calibrated 288-cell simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    "20260911_288_results_final_simulated_pack_v1"
)
EXPECTED = {
    "action": base.EXPECTED_SHA256["action"],
    "action_profile": base.EXPECTED_SHA256["action_profile"],
    "cell": base.EXPECTED_SHA256["cell"],
    "baseline_timing": base.EXPECTED_SHA256["baseline_timing"],
    "sim_manifest": "a0f8705833a64000d272fa2e1f81e06263856ccc0cb5930cdacd0c65580ac981",
    "sim_result": "2dff447678c8e265d175d7e8335bcf9f0e6492e28d296e0f207c5f32a33a81ba",
    "sim_cells": "a39b5f03a9282176616f2b99b03972deedeec46a026a0e268d58ea621097ba1c",
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
                s=18.0 + 105.0 * part["sim_rate_installed_per_sent"],
                c=base.FAMILY_COLORS[family],
                marker=base.QUANT_MARKERS[quantizer],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
            )
    base.annotate_actions(axis, frame, "payload_kib", metric)


def _figure_legend(fig: plt.Figure) -> None:
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
    handles += [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=np.sqrt(size),
            color="#777777",
            label=f"{int(rate * 100)}% simulated install",
        )
        for rate, size in ((0.1, 18 + 105 * 0.1), (0.5, 18 + 105 * 0.5), (0.9, 18 + 105 * 0.9))
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.025),
    )


def plot_profile_quality(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20.5, 13.0), constrained_layout=True)
    for row, (metric, ylabel) in enumerate(QUALITY_PANELS):
        for column, network in enumerate(base.NETWORKS):
            axis = axes[row, column]
            part = frame[frame["network_profile"] == network]
            _facet_scatter(axis, part, metric)
            base.payload_axis(axis)
            axis.set_ylabel(ylabel)
            axis.set_title(base.NETWORK_LABELS[network])
            axis.set_ylim(0.25, 1.0)
    _figure_legend(fig)
    fig.suptitle(
        "Frozen validation quality by measured network profile\n"
        "marker size = simulated map-install probability",
        fontsize=15,
        fontweight="bold",
    )
    base.save(fig, output, "01_payload_vs_validation_quality")


def plot_profile_localization(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(20.5, 16.0), constrained_layout=True)
    for row, (metric, ylabel) in enumerate(LOCALIZATION_PANELS):
        for column, network in enumerate(base.NETWORKS):
            axis = axes[row, column]
            part = frame[frame["network_profile"] == network]
            _facet_scatter(axis, part, metric)
            base.payload_axis(axis)
            axis.set_ylabel(ylabel)
            axis.set_title(base.NETWORK_LABELS[network])
    _figure_legend(fig)
    fig.suptitle(
        "Frozen localization quality by measured network profile\n"
        "marker size = simulated map-install probability",
        fontsize=15,
        fontweight="bold",
    )
    base.save(fig, output, "02_payload_vs_localization_quality")


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


def plot_final_edge_comparison(
    baseline: pd.DataFrame, result: dict, output: Path
) -> pd.DataFrame:
    original = baseline.set_index("action_id")
    calibration = result["family_calibration"]
    records = []
    for action_id in base.DIAGNOSTIC_ACTIONS:
        family = str(original.loc[action_id]["profile_id"]).split("_")[1]
        family = {"noae": "noAE", "ae128": "AE128", "ae64": "AE64", "ae32": "AE32"}[family]
        row = calibration[family]
        records.append(
            {
                "action_id": action_id,
                "family": family,
                "original_edge_processing_ms": float(row["baseline_total_edge_processing_ms_median"]),
                "final_edge_processing_ms": float(row["optimized_total_edge_processing_ms_median"]),
                "saving_ms": float(row["total_edge_processing_reduction_ms"]),
                "final_variant": row["target_variant"],
            }
        )
    table = pd.DataFrame(records)
    x = np.arange(len(table))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12.8, 6.2), constrained_layout=True)
    before = axis.bar(x - width / 2, table["original_edge_processing_ms"], width, label="Original", color="#9C9C9C")
    after = axis.bar(x + width / 2, table["final_edge_processing_ms"], width, label="Final qualified target", color="#2A9D8F")
    axis.set_xticks(x, [base.DIAGNOSTIC_LABELS[a] for a in base.DIAGNOSTIC_ACTIONS])
    axis.set_ylabel("Median edge processing, worker start → publication (ms)")
    axis.set_ylim(0, 185)
    axis.set_title(
        "Original versus final output-preserving edge implementation\n"
        "A15 retains live-valid v2; A30/A50/A71 use final v3"
    )
    axis.legend()
    for bars in (before, after):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f"{bar.get_height():.1f}", ha="center", fontweight="bold")
    for index, row in table.iterrows():
        axis.text(index, 6, f"−{row['saving_ms']:.1f} ms", ha="center", color="white", fontweight="bold", fontsize=8)
    base.save(fig, output, "10_edge_optimization_before_after")
    return table


def plot_agent_architecture(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(16.2, 7.4), constrained_layout=True)
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 8)
    axis.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", linewidth=1.8, edgecolor=color, facecolor=color + "18")
        axis.add_patch(patch)
        axis.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontweight="bold", fontsize=11, color=color)
        axis.text(x + w / 2, y + h / 2 - 0.12, body, ha="center", va="center", fontsize=9)

    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#333333"))
        if label:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, label, ha="center", fontsize=8, fontweight="bold")

    box(0.25, 2.55, 2.65, 3.05, "Causal observation", "SNR / MCS / throughput\nmap AoI + risk\nrecent delivery outcomes\nprevious action + costs", "#3366CC")
    box(3.55, 3.0, 2.05, 2.15, "Encoder", "availability flags\nLayerNorm\nMLP", "#7A5195")
    box(6.25, 2.8, 2.35, 2.55, "LSTM memory", "history state $(h_t,c_t)$\nchannel dynamics\npartial observability", "#EF5675")
    box(9.45, 5.45, 2.45, 1.25, "Policy head", "72 split-action logits", "#2A9D8F")
    box(9.45, 3.75, 2.45, 1.25, "Value head", "expected return", "#4C78A8")
    box(9.45, 2.05, 2.45, 1.25, "Cost critics", "bytes / compute / switching", "#F58518")
    box(9.45, 0.35, 2.45, 1.25, "Forecast head", "next-SNR prediction\n(auxiliary loss only)", "#54A24B")
    box(12.75, 4.55, 2.85, 1.65, "Split action", "family + quantizer + q\n$a_t \in \{0,\ldots,71\}$", "#264653")
    box(12.75, 1.55, 2.85, 1.75, "Environment feedback", "installed-map utility\nAoI / terminal outcome\nbytes + compute charged", "#8C564B")
    arrow(2.9, 4.05, 3.55, 4.05)
    arrow(5.6, 4.05, 6.25, 4.05)
    for target_y in (6.08, 4.38, 2.68, 0.98):
        arrow(8.6, 4.05, 9.45, target_y)
    arrow(11.9, 6.08, 12.75, 5.38, "masked sample")
    arrow(14.15, 4.55, 14.15, 3.3)
    axis.plot([12.75, 12.35, 8.15], [2.35, 0.10, 0.10], color="#333333", linewidth=1.5)
    arrow(8.15, 0.10, 7.45, 2.8)
    axis.text(10.25, 0.20, "next observation + reward", ha="center", fontsize=8, fontweight="bold")
    axis.text(8.0, 7.55, "Split-only recurrent PPO architecture", ha="center", fontsize=16, fontweight="bold")
    axis.text(8.0, 7.08, "Future SNR and current-frame outcomes are never policy inputs", ha="center", fontsize=10, color="#8B1E3F", fontweight="bold")
    base.save(fig, output, "agent_architecture")


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

## Figures 01–02

The validation and localization scores are frozen action-level measurements;
they were not measured four times and are not medians across network profiles.
Each column now uses that profile's measured payload. Marker size represents
the simulated installation probability, which is the network-dependent part.
The same intrinsically good action can therefore be useful under favorable
conditions but rarely installed under adverse conditions.

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

Compares only the original edge processing with the final target. A30/A50/A71
use final v3. NoAE uses the last live-valid v2 because action 15 failed v3
closed on non-finite camera-aware geometry. The plot deliberately omits
intermediate implementations. V3 overlaps GPU/CPU stages, so component times
cannot be presented as an additive stack; worker-start-to-publication is the
scientifically valid before/after total.

## Agent transition

The policy diagram connects the evidence to split-only recurrent PPO. The
action determines intrinsic quality and payload; the channel and edge state
determine whether the update arrives and remains useful; installed-object AoI
determines freshness utility. The next-SNR head is an auxiliary training task,
not access to future channel information.
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
    optimization = plot_final_edge_comparison(baseline, result, output)
    plot_agent_architecture(output)
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
        "schema": "scenesense.splitfusion.final_simulated_presentation_pack.v1",
        "status": "COMPLETE",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": {name: base.sha256_file(path) for name, path in paths.items()},
        "row_counts": {"actions": len(action), "action_profiles": len(frame), "simulated_cells": len(simulated)},
        "simulation_status": result["scientific_status"],
        "presentation_rules": [
            "VALIDATION_QUALITY_IS_NETWORK_INDEPENDENT_AND_FROZEN",
            "PROFILE_FACETS_ENCODE_SIMULATED_INSTALL_PROBABILITY_BY_MARKER_SIZE",
            "MEASURED_TRANSPORT_AND_REASSEMBLY_ARE_UNCHANGED",
            "FIGURES_04_06_07_08_USE_FINAL_EDGE_COUNTERFACTUAL",
            "FIGURES_03_05_09_RETAIN_MEASURED_ORIGINAL_VALUES",
            "FIGURE_08_HAS_NO_100MS_REFERENCE_LINE",
            "FIGURE_10_HAS_NO_INTERMEDIATE_OPTIMIZATION_VARIANTS",
            "NOAE_FINAL_TARGET_IS_LIVE_VALID_V2_NOT_FAILED_V3",
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
