#!/usr/bin/env python3
"""Create-only analysis of the existing six q anchors.

The dense-q inference preflight is intentionally separate from the analysis:
if the immutable dataset or CUDA is unavailable, this script records the stop
and analyzes only already-persisted per-frame evidence. It never imports CARLA,
starts OAI, edits a registry/runtime, or writes outside its own run directory.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
RAW = ROOT / "rl_agent" / "density_knob" / "raw"
DATASET = ROOT / "fusion_training_data" / "moving_ego_pps200000_merged_8loops_stride2"

FAMILIES = ("noae", "ae32", "ae64", "ae128")
QUANTS = ("uint8", "uint6", "uint4")
MEASURED_Q = (0.0, 0.3, 0.5, 0.7, 0.9, 0.98)
DENSE_Q = tuple(round(index * 0.05, 2) for index in range(17)) + (0.9, 0.98)
ID_Q_MAX = 0.8
N_LOW = 54 * 96
N_HIGH = 27 * 48
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260820

CHECKPOINTS = {
    "noae": ROOT / "experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt",
    "ae32": ROOT / "experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt",
    "ae64": ROOT / "experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt",
    "ae128": ROOT / "experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt",
}

SAMPLE_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)_frame(?P<frame>\d+)$")

SUM_COLUMNS = [
    "payload_bytes", "n_pred", "tp", "fp", "fn",
    "tp_veh", "fp_veh", "fn_veh", "tp_ped", "fp_ped", "fn_ped",
    "loc_err_sum", "loc_err_sq_sum", "loc_err_sum_veh", "loc_err_sum_ped",
] + [f"conf_{i}{j}" for i in range(3) for j in range(3)]

BOOT_METRICS = (
    "payload_bytes_mean", "veh_precision", "veh_recall", "ped_precision",
    "ped_recall", "fp_per_frame", "xy_mae_m", "xy_rmse_m", "miou",
    "iou_vehicle", "iou_person",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def parse_sample_id(sample_id: str) -> Tuple[str, int, int]:
    match = SAMPLE_RE.match(str(sample_id))
    if not match:
        raise ValueError(f"Unrecognized sample identifier: {sample_id}")
    return match.group("prefix"), int(match.group("index")), int(match.group("frame"))


def split_assignment(sample_id: str) -> Tuple[str, str, int]:
    prefix, collection_index, _ = parse_sample_id(sample_id)
    block = collection_index // 25
    token = f"continuous-roi-v1|{prefix}|{block}"
    value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    split = "audit_validation" if value % 5 in (0, 1) else "audit_test"
    return split, prefix, block


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def f1_value(precision: float, recall: float) -> float:
    if not math.isfinite(precision) or not math.isfinite(recall):
        return float("nan")
    return safe_div(2.0 * precision * recall, precision + recall)


def ious_from_confusion(confusion: np.ndarray) -> Tuple[float, float, float, float]:
    values = []
    for index in range(3):
        intersection = float(confusion[index, index])
        union = float(confusion[index, :].sum() + confusion[:, index].sum() - confusion[index, index])
        values.append(safe_div(intersection, union))
    return float(np.nanmean(values)), values[0], values[1], values[2]


def metrics_from_sums(values: Mapping[str, float], frame_count: int) -> Dict[str, float]:
    veh_p = safe_div(values["tp_veh"], values["tp_veh"] + values["fp_veh"])
    veh_r = safe_div(values["tp_veh"], values["tp_veh"] + values["fn_veh"])
    ped_p = safe_div(values["tp_ped"], values["tp_ped"] + values["fp_ped"])
    ped_r = safe_div(values["tp_ped"], values["tp_ped"] + values["fn_ped"])
    confusion = np.asarray([[values[f"conf_{i}{j}"] for j in range(3)] for i in range(3)], dtype=np.float64)
    miou, iou_bg, iou_veh, iou_person = ious_from_confusion(confusion)
    tp = float(values["tp"])
    return {
        "payload_bytes_mean": safe_div(values["payload_bytes"], frame_count),
        "veh_precision": veh_p,
        "veh_recall": veh_r,
        "veh_f1": f1_value(veh_p, veh_r),
        "ped_precision": ped_p,
        "ped_recall": ped_r,
        "ped_f1": f1_value(ped_p, ped_r),
        "fp_per_frame": safe_div(values["fp"], frame_count),
        "xy_mae_m": safe_div(values["loc_err_sum"], tp),
        "xy_rmse_m": math.sqrt(safe_div(values["loc_err_sq_sum"], tp)) if tp else float("nan"),
        "xy_mae_vehicle_m": safe_div(values["loc_err_sum_veh"], values["tp_veh"]),
        "xy_mae_person_m": safe_div(values["loc_err_sum_ped"], values["tp_ped"]),
        "miou": miou,
        "iou_background": iou_bg,
        "iou_vehicle": iou_veh,
        "iou_person": iou_person,
    }


def load_evidence() -> pd.DataFrame:
    tables = []
    for family in FAMILIES:
        path = RAW / f"perframe_{family}.csv"
        table = pd.read_csv(path)
        table["family"] = family
        table["quantizer"] = table["quant"].str.replace("per_channel_", "", regex=False)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    expected_rows = len(FAMILIES) * len(QUANTS) * len(MEASURED_Q) * 2162
    if len(result) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} persisted rows, found {len(result)}")
    combos = result.groupby(["family", "quantizer", "roi"]).sample_id.nunique()
    if len(combos) != 72 or not (combos == 2162).all():
        raise AssertionError("Existing 72-profile evidence is incomplete")
    return result


def make_split_manifest(frame_density: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in frame_density.to_dict("records"):
        sample_id = str(record["sample_id"])
        split, prefix, block = split_assignment(sample_id)
        _, collection_index, parsed_frame = parse_sample_id(sample_id)
        rows.append({
            "sample_id": sample_id,
            "frame_id": int(record.get("frame_id", parsed_frame)),
            "source_prefix": prefix,
            "collection_index": collection_index,
            "trajectory_block": block,
            "block_key": f"{prefix}|{block}",
            "audit_split": split,
            "density_bin": str(record.get("density_bin", "")),
        })
    result = pd.DataFrame(rows).sort_values(["source_prefix", "collection_index", "sample_id"])
    if result.sample_id.duplicated().any():
        raise AssertionError("Duplicate sample IDs in frame density")
    val = set(result.loc[result.audit_split == "audit_validation", "sample_id"])
    test = set(result.loc[result.audit_split == "audit_test", "sample_id"])
    if val & test:
        raise AssertionError("Audit identifier overlap")
    block_splits = result.groupby(["source_prefix", "trajectory_block"]).audit_split.nunique()
    if int(block_splits.max()) != 1:
        raise AssertionError("Trajectory block split overlap")
    return result


def summarize_per_q(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in evidence.groupby(["audit_split", "family", "quantizer", "roi"], sort=True):
        split, family, quantizer, q = keys
        sums = {column: float(group[column].sum()) for column in SUM_COLUMNS}
        metrics = metrics_from_sums(sums, len(group))
        rows.append({
            "split": split,
            "family": family,
            "quantizer": quantizer,
            "q": float(q),
            "q_support": "in_distribution" if float(q) <= ID_Q_MAX else "measured_extrapolation",
            "unique_frames": int(group.sample_id.nunique()),
            "profile_frames": int(len(group)),
            "payload_bytes_median": float(group.payload_bytes.median()),
            "payload_bytes_p90": float(group.payload_bytes.quantile(0.90)),
            "payload_bytes_p95": float(group.payload_bytes.quantile(0.95)),
            "payload_bytes_max": int(group.payload_bytes.max()),
            "tp_vehicle": int(group.tp_veh.sum()),
            "fp_vehicle": int(group.fp_veh.sum()),
            "fn_vehicle": int(group.fn_veh.sum()),
            "tp_person": int(group.tp_ped.sum()),
            "fp_person": int(group.fp_ped.sum()),
            "fn_person": int(group.fn_ped.sum()),
            "prediction_count": int(group.n_pred.sum()),
            **metrics,
        })
    # Full is a descriptive aggregate only; audit decisions use the frozen split rows.
    for keys, group in evidence.groupby(["family", "quantizer", "roi"], sort=True):
        family, quantizer, q = keys
        sums = {column: float(group[column].sum()) for column in SUM_COLUMNS}
        metrics = metrics_from_sums(sums, len(group))
        rows.append({
            "split": "full_descriptive",
            "family": family,
            "quantizer": quantizer,
            "q": float(q),
            "q_support": "in_distribution" if float(q) <= ID_Q_MAX else "measured_extrapolation",
            "unique_frames": int(group.sample_id.nunique()),
            "profile_frames": int(len(group)),
            "payload_bytes_median": float(group.payload_bytes.median()),
            "payload_bytes_p90": float(group.payload_bytes.quantile(0.90)),
            "payload_bytes_p95": float(group.payload_bytes.quantile(0.95)),
            "payload_bytes_max": int(group.payload_bytes.max()),
            "tp_vehicle": int(group.tp_veh.sum()), "fp_vehicle": int(group.fp_veh.sum()), "fn_vehicle": int(group.fn_veh.sum()),
            "tp_person": int(group.tp_ped.sum()), "fp_person": int(group.fp_ped.sum()), "fn_person": int(group.fn_ped.sum()),
            "prediction_count": int(group.n_pred.sum()),
            **metrics,
        })
    return pd.DataFrame(rows).sort_values(["split", "family", "quantizer", "q"])


def action_quantization() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for q in DENSE_Q:
        k_low = int(round(q * N_LOW))
        k_high = int(round(q * N_HIGH))
        rows.append({
            "q": q,
            "q_support": "in_distribution" if q <= ID_Q_MAX else "measured_extrapolation",
            "low_total_cells": N_LOW,
            "low_dropped_cells": k_low,
            "low_retained_cells": N_LOW - k_low,
            "low_actual_drop_fraction": k_low / N_LOW,
            "high_total_cells": N_HIGH,
            "high_dropped_cells": k_high,
            "high_retained_cells": N_HIGH - k_high,
            "high_actual_drop_fraction": k_high / N_HIGH,
            "q_equals_one_forbidden": False,
        })
    table = pd.DataFrame(rows)
    low_boundaries = {Fraction(2 * k + 1, 2 * N_LOW) for k in range(N_LOW) if Fraction(2 * k + 1, 2 * N_LOW) < Fraction(4, 5)}
    high_boundaries = {Fraction(2 * k + 1, 2 * N_HIGH) for k in range(N_HIGH) if Fraction(2 * k + 1, 2 * N_HIGH) < Fraction(4, 5)}
    boundaries = sorted(low_boundaries | high_boundaries)
    points = [Fraction(0, 1), *boundaries, Fraction(4, 5)]
    widths = [float(right - left) for left, right in zip(points[:-1], points[1:])]
    summary = {
        "semantics": "drop round(q*N) lowest-ranked objectness cells independently at native low/high resolutions",
        "not_a_score_threshold": True,
        "rounding": "Python round, ties to even",
        "n_low": N_LOW,
        "n_high": N_HIGH,
        "in_distribution_interval": [0.0, 0.8],
        "joint_transition_count_open_interval": len(boundaries),
        "joint_plateau_count": len(boundaries) + 1,
        "minimum_plateau_width_q": min(widths),
        "maximum_plateau_width_q": max(widths),
        "low_only_transition_count": len(low_boundaries),
        "high_only_transition_count": len(high_boundaries),
        "coincident_transition_count": len(low_boundaries & high_boundaries),
        "q_one_forbidden": True,
    }
    return table, summary


def payload_pairing(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = list(zip(MEASURED_Q[:-1], MEASURED_Q[1:]))
    for keys, branch in evidence.groupby(["audit_split", "family", "quantizer"]):
        split, family, quantizer = keys
        for q0, q1 in pairs:
            left = branch.loc[np.isclose(branch.roi, q0), ["sample_id", "payload_bytes"]].rename(columns={"payload_bytes": "payload_q0"})
            right = branch.loc[np.isclose(branch.roi, q1), ["sample_id", "payload_bytes"]].rename(columns={"payload_bytes": "payload_q1"})
            paired = left.merge(right, on="sample_id", validate="one_to_one")
            delta = paired.payload_q1 - paired.payload_q0
            rows.append({
                "split": split, "family": family, "quantizer": quantizer,
                "q0": q0, "q1": q1,
                "interval_role": "in_distribution" if q1 <= ID_Q_MAX else "extrapolation_reference",
                "paired_frames": len(paired),
                "nonincreasing_or_tied_fraction": float((delta <= 0).mean()),
                "strict_increase_count": int((delta > 0).sum()),
                "mean_delta_bytes": float(delta.mean()),
                "median_delta_bytes": float(delta.median()),
                "max_increase_bytes": int(max(0, delta.max())),
            })
    return pd.DataFrame(rows)


def block_bootstrap(evidence: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pairs = list(zip(MEASURED_Q[:-1], MEASURED_Q[1:]))
    rows = []
    for keys, branch in evidence.groupby(["audit_split", "family", "quantizer"]):
        split, family, quantizer = keys
        block_sums = branch.groupby(["source_prefix", "block_key", "roi"], as_index=False)[SUM_COLUMNS].sum()
        block_counts = branch.groupby(["source_prefix", "block_key", "roi"]).size().rename("frame_count").reset_index()
        block_sums = block_sums.merge(block_counts, on=["source_prefix", "block_key", "roi"], validate="one_to_one")
        vector_columns = [*SUM_COLUMNS, "frame_count"]
        for q0, q1 in pairs:
            q_tables = {}
            prefix_keys: Dict[str, List[str]] = {}
            for q in (q0, q1):
                current = block_sums.loc[np.isclose(block_sums.roi, q)].copy()
                current["key"] = current.source_prefix + "||" + current.block_key
                q_tables[q] = {str(row.key): np.asarray([getattr(row, column) for column in vector_columns], dtype=np.float64)
                               for row in current.itertuples()}
                if not prefix_keys:
                    for prefix, group in current.groupby("source_prefix"):
                        prefix_keys[str(prefix)] = list(group.key.astype(str))
            if set(q_tables[q0]) != set(q_tables[q1]):
                raise AssertionError("Incomplete q pairing in block bootstrap")

            def evaluate(q: float, sampled_keys: Sequence[str]) -> Dict[str, float]:
                vector = np.sum([q_tables[q][key] for key in sampled_keys], axis=0)
                values = {column: float(vector[index]) for index, column in enumerate(vector_columns)}
                return metrics_from_sums(values, int(values["frame_count"]))

            all_keys = [key for keys_for_prefix in prefix_keys.values() for key in keys_for_prefix]
            observed0 = evaluate(q0, all_keys)
            observed1 = evaluate(q1, all_keys)
            distributions: Dict[str, List[float]] = {metric: [] for metric in BOOT_METRICS}
            for _ in range(BOOTSTRAP_REPS):
                sampled: List[str] = []
                for available in prefix_keys.values():
                    sampled.extend(rng.choice(available, size=len(available), replace=True).tolist())
                metrics0 = evaluate(q0, sampled)
                metrics1 = evaluate(q1, sampled)
                for metric in BOOT_METRICS:
                    distributions[metric].append(metrics1[metric] - metrics0[metric])
            for metric in BOOT_METRICS:
                values = np.asarray([value for value in distributions[metric] if math.isfinite(value)], dtype=np.float64)
                rows.append({
                    "split": split, "family": family, "quantizer": quantizer,
                    "q0": q0, "q1": q1,
                    "interval_role": "in_distribution" if q1 <= ID_Q_MAX else "extrapolation_reference",
                    "metric": metric,
                    "value_q0": observed0[metric], "value_q1": observed1[metric],
                    "observed_delta": observed1[metric] - observed0[metric],
                    "delta_ci95_low": float(np.percentile(values, 2.5)),
                    "delta_ci95_high": float(np.percentile(values, 97.5)),
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "resampling_unit": "trajectory_block_with_all_q_rows",
                })
    return pd.DataFrame(rows)


def coarse_interpolation(per_q: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "payload_bytes_mean", "veh_precision", "veh_recall", "ped_precision", "ped_recall",
        "fp_per_frame", "xy_mae_m", "xy_rmse_m", "miou", "iou_vehicle", "iou_person",
    ]
    protocols = ((0.0, 0.3, 0.5), (0.3, 0.5, 0.7))
    rows = []
    source = per_q.loc[per_q.split.isin(["audit_validation", "audit_test"])]
    for keys, branch in source.groupby(["split", "family", "quantizer"]):
        split, family, quantizer = keys
        by_q = branch.set_index("q")
        for q_left, q_target, q_right in protocols:
            weight = (q_target - q_left) / (q_right - q_left)
            for metric in metrics:
                left, actual, right = (float(by_q.loc[q, metric]) for q in (q_left, q_target, q_right))
                predicted = left + weight * (right - left)
                error = actual - predicted
                rows.append({
                    "split": split, "family": family, "quantizer": quantizer,
                    "protocol": "coarse_leave_one_current_anchor_out",
                    "q_left": q_left, "q_target": q_target, "q_right": q_right,
                    "metric": metric, "actual": actual, "linear_prediction": predicted,
                    "signed_error": error, "absolute_error": abs(error),
                    "relative_absolute_error": safe_div(abs(error), abs(actual)),
                    "decision_eligible": False,
                    "reason_not_decision_eligible": "target is an existing coarse anchor; dense midpoint q outputs are missing",
                })
    return pd.DataFrame(rows)


def curve_diagnostics(per_q: pd.DataFrame, interpolation: pd.DataFrame) -> pd.DataFrame:
    metric_directions = {
        "payload_bytes_mean": "decreasing", "veh_precision": "none", "veh_recall": "none",
        "ped_precision": "none", "ped_recall": "none", "fp_per_frame": "none",
        "xy_mae_m": "none", "xy_rmse_m": "none", "miou": "none",
    }
    rows = []
    source = per_q.loc[(per_q.split == "audit_test") & (per_q.q <= ID_Q_MAX)]
    for keys, branch in source.groupby(["family", "quantizer"]):
        family, quantizer = keys
        branch = branch.sort_values("q")
        q = branch.q.to_numpy(dtype=float)
        for metric, expected_direction in metric_directions.items():
            values = branch[metric].to_numpy(dtype=float)
            slopes = np.diff(values) / np.diff(q)
            direction_changes = int(np.sum(np.sign(slopes[1:]) * np.sign(slopes[:-1]) < 0)) if len(slopes) > 1 else 0
            monotonic_inc = bool(np.all(np.diff(values) >= -1e-12))
            monotonic_dec = bool(np.all(np.diff(values) <= 1e-12))
            interp = interpolation.loc[(interpolation.split == "audit_test") &
                                       (interpolation.family == family) &
                                       (interpolation.quantizer == quantizer) &
                                       (interpolation.metric == metric)]
            rows.append({
                "family": family, "quantizer": quantizer, "metric": metric,
                "measured_in_distribution_q_count": len(q),
                "expected_direction": expected_direction,
                "monotonic_increasing": monotonic_inc,
                "monotonic_decreasing": monotonic_dec,
                "slope_direction_changes": direction_changes,
                "largest_absolute_slope_per_q": float(np.max(np.abs(slopes))),
                "coarse_loo_max_absolute_error": float(interp.absolute_error.max()) if not interp.empty else float("nan"),
                "dense_smoothness_test_available": False,
            })
    return pd.DataFrame(rows)


def crossings(per_q: pd.DataFrame) -> pd.DataFrame:
    metrics = ["payload_bytes_mean", "veh_precision", "veh_recall", "ped_precision", "ped_recall",
               "fp_per_frame", "xy_mae_m", "miou"]
    source = per_q.loc[per_q.split == "audit_test"]
    rows = []

    def scan(scope: str, fixed_name: str, fixed_value: str, item_name: str, items: Sequence[str], table: pd.DataFrame) -> None:
        for item_a, item_b in itertools.combinations(items, 2):
            a = table.loc[table[item_name] == item_a].set_index("q")
            b = table.loc[table[item_name] == item_b].set_index("q")
            shared = sorted(set(a.index) & set(b.index))
            for metric in metrics:
                for q0, q1 in zip(shared[:-1], shared[1:]):
                    delta0 = float(a.loc[q0, metric] - b.loc[q0, metric])
                    delta1 = float(a.loc[q1, metric] - b.loc[q1, metric])
                    if delta0 == 0.0 or delta1 == 0.0 or delta0 * delta1 < 0.0:
                        rows.append({
                            "scope": scope, "fixed_factor": fixed_name, "fixed_value": fixed_value,
                            "compared_factor": item_name, "item_a": item_a, "item_b": item_b,
                            "metric": metric, "q0": q0, "q1": q1,
                            "delta_at_q0": delta0, "delta_at_q1": delta1,
                            "interval_role": "in_distribution" if q1 <= ID_Q_MAX else "extrapolation_reference",
                        })

    for family in FAMILIES:
        scan("within_family_quantizer", "family", family, "quantizer", QUANTS, source.loc[source.family == family])
    for quantizer in QUANTS:
        scan("within_quantizer_family", "quantizer", quantizer, "family", FAMILIES, source.loc[source.quantizer == quantizer])
    return pd.DataFrame(rows)


COLORS = {"uint8": "#0072B2", "uint6": "#E69F00", "uint4": "#6A3D9A"}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def make_figures(per_q: pd.DataFrame, action_table: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=False)
    test = per_q.loc[per_q.split == "audit_test"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.step(action_table.q, action_table.low_dropped_cells, where="post", label="low: N=5,184", color="#0072B2")
    ax.step(action_table.q, action_table.high_dropped_cells, where="post", label="high: N=1,296", color="#E69F00")
    ax.axvspan(0.8, 0.98, color="#999999", alpha=0.18, label="measured extrapolation")
    ax.set(xlabel="q", ylabel="Dropped feature cells", title="Rank-drop action is integer and piecewise constant")
    ax.grid(True, alpha=0.25); ax.legend()
    ax.text(0.01, 0.01, "continuous_roi_control_audit_v1/20260820_203411", transform=ax.transAxes, fontsize=7, color="#555555")
    save_figure(fig, "q_action_piecewise")

    def family_grid(metric: str, ylabel: str, stem: str, scale: float = 1.0) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        for family, ax in zip(FAMILIES, axes.flat):
            for quantizer in QUANTS:
                branch = test.loc[(test.family == family) & (test.quantizer == quantizer)].sort_values("q")
                ax.plot(branch.q, branch[metric] / scale, marker="o", linewidth=1.7,
                        color=COLORS[quantizer], label=quantizer)
            ax.axvspan(0.8, 0.98, color="#999999", alpha=0.16)
            ax.set_title(family); ax.grid(True, alpha=0.25)
        axes[1, 0].set_xlabel("q"); axes[1, 1].set_xlabel("q")
        axes[0, 0].set_ylabel(ylabel); axes[1, 0].set_ylabel(ylabel)
        axes[0, 0].legend(ncol=3, fontsize=8)
        fig.suptitle(f"Audit-test existing-anchor {ylabel} curves")
        save_figure(fig, stem)

    family_grid("payload_bytes_mean", "Mean payload (KiB)", "payload_anchor_curves", scale=1024.0)
    family_grid("veh_precision", "Vehicle precision", "vehicle_precision_anchor_curves")
    family_grid("veh_recall", "Vehicle recall", "vehicle_recall_anchor_curves")
    family_grid("ped_precision", "Person precision", "person_precision_anchor_curves")
    family_grid("ped_recall", "Person recall", "person_recall_anchor_curves")
    family_grid("xy_mae_m", "World-XY MAE (m)", "xy_mae_anchor_curves")
    family_grid("fp_per_frame", "False positives / frame", "fp_per_frame_anchor_curves")
    family_grid("miou", "Segmentation mIoU (secondary)", "segmentation_miou_anchor_curves")


def markdown_table(table: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    if table.empty:
        return "(no rows)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def preflight() -> Dict[str, Any]:
    required_dataset = [DATASET / "manifest.csv", DATASET / "object_boxes.csv"]
    checkpoint_status = [{"family": family, "path": str(path), "exists": path.exists(),
                          "sha256": sha256_file(path) if path.exists() else None}
                         for family, path in CHECKPOINTS.items()]
    cuda_available = bool(torch.cuda.is_available())
    reasons = []
    for path in required_dataset:
        if not path.exists():
            reasons.append(f"missing_required_dataset_file:{path}")
    if not cuda_available:
        reasons.append("cuda_unavailable:no_cpu_full_run_fallback_per_frozen_plan")
    return {
        "checked_at_unix": time.time(),
        "dataset_root": str(DATASET),
        "dataset_root_exists": DATASET.exists(),
        "required_dataset_files": [{"path": str(path), "exists": path.exists()} for path in required_dataset],
        "checkpoints": checkpoint_status,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "stop_rule_triggered": bool(reasons),
        "stop_reasons": reasons,
        "dense_inference_started": False,
        "carla_started": False,
        "oai_started": False,
    }


def input_entries() -> List[Dict[str, Any]]:
    inputs: List[Tuple[str, Path]] = []
    inputs.extend(("per_frame_q_evidence", RAW / f"perframe_{family}.csv") for family in FAMILIES)
    inputs.extend([
        ("frame_density", RAW / "frame_density.csv"),
        ("eval_settings", RAW / "eval_settings.json"),
        ("registry_config_read_only", ROOT / "rl_agent/configs/ue_split_profile_registry_v1.json"),
        ("density_evaluator", ROOT / "rl_agent/density_knob/density_knob_eval.py"),
        ("model_q_semantics", ROOT / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/model.py"),
        ("evaluator_q_semantics", ROOT / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/evaluate_fusion.py"),
        ("live_q_semantics_read_only", ROOT / "uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py"),
        ("split_runtime", ROOT / "pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/split_runtime.py"),
    ])
    inputs.extend(("checkpoint", path) for path in CHECKPOINTS.values())
    rows = []
    for role, path in inputs:
        exists = path.exists()
        stat = path.stat() if exists else None
        rows.append({"role": role, "path": str(path.relative_to(ROOT)), "exists": exists,
                     "size_bytes": stat.st_size if stat else None,
                     "mtime_ns": stat.st_mtime_ns if stat else None,
                     "sha256": sha256_file(path) if exists else None})
    for role, path in (("dataset_manifest_missing", DATASET / "manifest.csv"),
                       ("dataset_objects_missing", DATASET / "object_boxes.csv")):
        rows.append({"role": role, "path": str(path.relative_to(ROOT)), "exists": path.exists(),
                     "size_bytes": None, "mtime_ns": None, "sha256": None})
    return rows


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main() -> int:
    expected_absent = [
        "resolved_config.json", "preflight.json", "q_action_quantization.csv",
        "q_action_structural_summary.json", "audit_split_manifest.csv", "per_q_results.csv",
        "payload_frame_monotonicity.csv", "paired_anchor_bootstrap.csv",
        "coarse_anchor_interpolation.csv", "anchor_curve_diagnostics.csv", "branch_crossings.csv",
        "latency_measurement.csv", "REPORT.md", "RESULTS_SUMMARY.json", "REVIEW_REQUIRED.json", "manifest.json",
    ]
    collisions = [name for name in expected_absent if (OUT / name).exists()]
    if collisions:
        raise FileExistsError(f"Create-only output collision: {collisions}")

    resolved = {
        "audit_id": "continuous_roi_control_audit_v1/20260820_203411",
        "families": list(FAMILIES), "quantizers": list(QUANTS),
        "measured_q_anchors": list(MEASURED_Q), "planned_dense_q_grid": list(DENSE_Q),
        "primary_in_distribution_interval": [0.0, ID_Q_MAX], "q_one_forbidden": True,
        "feature_cells": {"low": N_LOW, "high": N_HIGH},
        "split": {"block_size": 25, "hash_salt": "continuous-roi-v1", "validation_modulo": [0, 1], "modulo": 5},
        "bootstrap": {"replicates": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED,
                      "unit": "trajectory block with all q outcomes"},
        "dense_profile_count": len(FAMILIES) * len(QUANTS) * len(DENSE_Q),
        "dense_profile_frame_count": len(FAMILIES) * len(QUANTS) * len(DENSE_Q) * 2162,
        "compute_budget": {"gpu_hours": 6, "wall_hours": 8, "new_artifacts_gb": 100, "cpu_full_fallback": False},
        "terminal_rule": "INSUFFICIENT_EVIDENCE when dense q inference or latency is missing",
    }
    (OUT / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    preflight_result = preflight()
    (OUT / "preflight.json").write_text(json.dumps(preflight_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not preflight_result["stop_rule_triggered"]:
        raise RuntimeError("This analysis-only driver must not silently begin inference; use a separately reviewed dense-q runner")

    action_table, action_summary = action_quantization()
    action_table.to_csv(OUT / "q_action_quantization.csv", index=False)
    (OUT / "q_action_structural_summary.json").write_text(json.dumps(action_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frame_density = pd.read_csv(RAW / "frame_density.csv")
    split_manifest = make_split_manifest(frame_density)
    split_manifest.to_csv(OUT / "audit_split_manifest.csv", index=False)
    evidence = load_evidence().merge(
        split_manifest[["sample_id", "source_prefix", "block_key", "trajectory_block", "audit_split", "density_bin"]],
        on="sample_id", how="left", validate="many_to_one",
    )
    if evidence.audit_split.isna().any():
        raise AssertionError("Evidence includes identifiers outside split manifest")

    per_q = summarize_per_q(evidence)
    per_q.to_csv(OUT / "per_q_results.csv", index=False)
    payload_pairs = payload_pairing(evidence)
    payload_pairs.to_csv(OUT / "payload_frame_monotonicity.csv", index=False)
    paired = block_bootstrap(evidence)
    paired.to_csv(OUT / "paired_anchor_bootstrap.csv", index=False)
    interpolation = coarse_interpolation(per_q)
    interpolation.to_csv(OUT / "coarse_anchor_interpolation.csv", index=False)
    diagnostics = curve_diagnostics(per_q, interpolation)
    diagnostics.to_csv(OUT / "anchor_curve_diagnostics.csv", index=False)
    crossing_table = crossings(per_q)
    crossing_table.to_csv(OUT / "branch_crossings.csv", index=False)

    latency = pd.DataFrame([
        {"stage": stage, "status": "NOT_MEASURED", "reason": "dense inference preflight stopped: dataset missing and CUDA unavailable",
         "p50_ms": float("nan"), "p90_ms": float("nan"), "p95_ms": float("nan"), "max_ms": float("nan")}
        for stage in ("objectness_plus_rank", "cached_rank_mask_apply", "full_q_gate", "integrated_ae", "quantize_serialize_zstd3", "gate_to_serialized_payload")
    ])
    latency.to_csv(OUT / "latency_measurement.csv", index=False)
    make_figures(per_q, action_table)

    test_q = per_q.loc[per_q.split == "audit_test"]
    test_id = test_q.loc[test_q.q <= ID_Q_MAX]
    payload_branch_monotone = (test_id.sort_values("q").groupby(["family", "quantizer"])
                               .payload_bytes_mean.apply(lambda s: bool(np.all(np.diff(s.to_numpy()) <= 0))))
    paired_id = payload_pairs.loc[(payload_pairs.split == "audit_test") & (payload_pairs.q1 <= ID_Q_MAX)]
    min_frame_monotonic = float(paired_id.nonincreasing_or_tied_fraction.min())
    interpolation_test = interpolation.loc[interpolation.split == "audit_test"]
    interpolation_max = (interpolation_test.groupby("metric").absolute_error.max().reset_index()
                         .sort_values("metric"))
    crossing_id_count = int((crossing_table.interval_role == "in_distribution").sum()) if not crossing_table.empty else 0
    split_counts = split_manifest.audit_split.value_counts().to_dict()

    branch_snapshot = test_q.loc[np.isclose(test_q.q, 0.7), [
        "family", "quantizer", "payload_bytes_mean", "veh_precision", "veh_recall",
        "ped_precision", "ped_recall", "fp_per_frame", "xy_mae_m", "xy_rmse_m", "miou",
    ]].sort_values(["family", "quantizer"])

    report = f"""# Continuous ROI-Drop Control Audit

Audit: `continuous_roi_control_audit_v1/20260820_203411`  
Terminal: **`INSUFFICIENT_EVIDENCE`**  
Controller recommendation: **keep measured discrete q anchors** pending dense-q evidence.

## Answer

The current 72-profile evidence does **not** establish q as a valid continuous
control in any of the 12 fixed family/quantizer branches. It contains only four
in-distribution q anchors (`0,.3,.5,.7`), so it cannot resolve local behavior at
the planned 0.05 scale or score held-out midpoint q values. This is not a finding
that q is intrinsically discrete-only: the preregistered `DISCRETE_ONLY` terminal
also requires a complete dense-q run. The defensible status is
`INSUFFICIENT_EVIDENCE`, with the controller remaining discrete.

## Preflight and stop rule

The frozen plan was written before inference. Preflight then triggered its stop
rule because the recorded dataset manifest/object file are absent and PyTorch
reports CUDA unavailable. No CPU full-run fallback was allowed. Dense inference,
CARLA, and OAI were not started; all results below derive from immutable existing
anchor CSVs.

- Planned grid: 19 q values; 228 profiles; 492,936 profile-frame evaluations.
- Audit validation: {int(split_counts.get('audit_validation', 0)):,} unique frames.
- Frozen audit test: {int(split_counts.get('audit_test', 0)):,} unique frames.
- Identifier and trajectory-block overlap: zero.

## Production q semantics and piecewise action

Production and evaluator code both use rank drop: independently for the native
low/high feature maps, compute objectness ordering and zero the
`round(q*N)` lowest-ranked cells. q is **not** a score threshold. With 5,184 low
cells and 1,296 high cells, `[0,.8]` has
**{action_summary['joint_plateau_count']:,} joint mask-count plateaus** separated
by {action_summary['joint_transition_count_open_interval']:,} transitions. Plateau
widths range from {action_summary['minimum_plateau_width_q']:.7f} to
{action_summary['maximum_plateau_width_q']:.7f} in q. Thus a float-valued API
produces a fine but integer, piecewise-constant actuator. The planned 0.05 grid
tests macro smoothness; it does not prove single-cell smoothness.

## Existing-anchor evidence

All 72 profiles are complete: 2,162 unique frames at six q anchors for every
branch. Aggregate audit-test payload is non-increasing over the measured
in-distribution anchors in **{int(payload_branch_monotone.sum())}/12 branches**.
The worst frame-paired non-increasing/tied rate over those large anchor gaps is
**{min_frame_monotonic:.2%}**. These are coarse payload facts, not continuous
quality validation.

At q=0.7, the 12 separate branch outcomes are:

{markdown_table(branch_snapshot, list(branch_snapshot.columns), 4)}

Object quality is not assumed monotonic. `anchor_curve_diagnostics.csv` records
slope reversals per branch/metric, while `paired_anchor_bootstrap.csv` provides
2,000 trajectory-block paired confidence intervals. The audit found
**{crossing_id_count} in-distribution ordering crossings** across family or
quantizer comparisons, reinforcing that branch factors are not safely separable.

## Interpolation and smoothness limit

Only a coarse leave-one-current-anchor-out diagnostic is possible: predict q=.3
from q=0/.5 and q=.5 from q=.3/.7. Its audit-test maximum errors are:

{markdown_table(interpolation_max, ['metric', 'absolute_error'], 5)}

Those targets are existing anchors and the gaps are 0.2--0.3 wide. They are not
the preregistered held-out `.05,.15,...,.75` midpoint test and cannot earn a
continuous terminal. No dense local discontinuity test was run. `.9/.98` remain
measured extrapolation references and are excluded from the continuous decision.

## Latency

q-gating and serialization latency is `NOT_MEASURED`. Historical end-to-end or
technical-smoke numbers do not isolate objectness/ranking, mask application,
integrated AE, quantization, serialization, and zstd-3 as required by the frozen
plan, so they were not substituted.

## Hybrid-action implication

Even if q later passes, the action is not plain continuous SAC/TD3: it is a
categorical choice among 12 `{{family, quantizer}}` branches plus a conditional
bounded `q in [0,.8]`. It needs an explicit hierarchy, parameterized-action
critic, or categorical branch policy with a conditional q actor. This audit adds
no q-selection reward and implements no RL agent.

## Smallest next experiment

Restore the exact hashed dataset and a CUDA device, then run **AE64/uint6 on the
822 audit-validation frames across all 19 q values**: 15,618 profile-frame
evaluations, reusing backbone features/ranking per frame. Measure all required
quality, payload, and stage latency outputs. Stop there if midpoint interpolation
or local-jump criteria fail.

Before promotion, evidence must expand to all 12 branches on the frozen 1,340-frame
audit test, pass the preregistered branch criteria with trajectory-block paired
uncertainty, include production-equivalent gate/serialization timing, and show
12/12 branch support. Until then: no q promotion, no registry change, and no
continuous/hybrid policy implementation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    summary = {
        "audit_id": "continuous_roi_control_audit_v1/20260820_203411",
        "terminal": "INSUFFICIENT_EVIDENCE",
        "controller_recommendation": "KEEP_MEASURED_DISCRETE_Q_ANCHORS",
        "dense_inference_started": False,
        "measured_profiles_complete": 72,
        "planned_dense_profiles": 228,
        "planned_dense_profile_frames": 492936,
        "structural_action": action_summary,
        "audit_split": {"validation_frames": int(split_counts.get("audit_validation", 0)),
                        "test_frames": int(split_counts.get("audit_test", 0)),
                        "identifier_overlap": 0, "block_overlap": 0},
        "existing_anchor_diagnostics": {
            "payload_monotone_branches": int(payload_branch_monotone.sum()),
            "branches": 12,
            "minimum_frame_paired_payload_nonincrease_fraction": min_frame_monotonic,
            "in_distribution_ordering_crossings": crossing_id_count,
        },
        "missing_evidence": ["13 unmeasured planned q points", "dense midpoint interpolation",
                             "0.05-step local discontinuity tests", "q gate and serialization latency",
                             "original dataset files", "CUDA device"],
        "next_experiment": {"branch": "ae64/uint6", "split": "audit_validation", "frames": int(split_counts.get("audit_validation", 0)),
                            "q_points": 19, "profile_frame_evaluations": int(split_counts.get("audit_validation", 0)) * 19},
        "promotion_allowed": False,
    }
    (OUT / "RESULTS_SUMMARY.json").write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review = {
        "schema": "scenesense.continuous_roi_control_review_required.v1",
        "terminal": "REVIEW_REQUIRED",
        "analysis_conclusion": "INSUFFICIENT_EVIDENCE",
        "controller_action": "KEEP_MEASURED_DISCRETE_Q_ANCHORS",
        "promotion_allowed": False,
        "registry_changed": False, "runtime_changed": False, "controller_changed": False,
        "carla_run": False, "oai_run": False, "rl_agent_implemented": False,
        "review_questions": [
            "Restore the exact dataset path or approve an equivalently hashed immutable cache.",
            "Provide CUDA capacity for the AE64/uint6 19-q validation pilot.",
            "Review the preregistered interpolation and local-jump tolerances before any promotion run.",
        ],
    }
    (OUT / "REVIEW_REQUIRED.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    input_manifest = input_entries()
    output_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts)
    manifest = {
        "schema": "scenesense.continuous_roi_control_audit_manifest.v1",
        "audit_id": "continuous_roi_control_audit_v1/20260820_203411",
        "created_at_unix": time.time(),
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "create_only": True,
        "terminal": "REVIEW_REQUIRED",
        "analysis_conclusion": "INSUFFICIENT_EVIDENCE",
        "inputs": input_manifest,
        "outputs": [{"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in output_files],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"terminal": "REVIEW_REQUIRED", "conclusion": "INSUFFICIENT_EVIDENCE",
                      "output_dir": str(OUT), "outputs": len(output_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
