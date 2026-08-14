#!/usr/bin/env python3
"""Task A: held-out scene-context rank-reversal screen over 36 measured profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "argmax_stability_v1.yaml"
PROFILE_RE = re.compile(r"^(noae|ae32|ae64|ae128)__uint(4|6|8)__roi(0\.0|0\.3|0\.5)$")
SAMPLE_RE = re.compile(r"^(.*)_([0-9]+)_frame([0-9]+)$")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_id(model: str, quant: str, roi: float) -> str:
    quant_short = str(quant).replace("per_channel_uint", "uint")
    return f"{model}__{quant_short}__roi{float(roi):.1f}"


def parse_matrix_payloads(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "__uint" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 17 or PROFILE_RE.match(cells[0]) is None:
            continue
        rows.append({"profile_id": cells[0], "payload_kib": float(cells[5].replace("~", ""))})
    frame = pd.DataFrame(rows).drop_duplicates("profile_id").sort_values("payload_kib")
    if len(frame) != 36:
        raise ValueError(f"expected 36 published profiles, parsed {len(frame)} from {path}")
    return frame.reset_index(drop=True)


def per_frame_miou(frame: pd.DataFrame) -> np.ndarray:
    confusion = np.stack(
        [frame[[f"conf_{i}{j}" for i in range(3) for j in range(3)]].to_numpy(dtype=float)[:, k]
         for k in range(9)],
        axis=1,
    ).reshape((-1, 3, 3))
    diagonal = np.diagonal(confusion, axis1=1, axis2=2)
    unions = confusion.sum(axis=1) + confusion.sum(axis=2) - diagonal
    iou = np.divide(diagonal, unions, out=np.full_like(diagonal, np.nan), where=unions > 0)
    return np.nanmean(iou, axis=1)


def add_utilities(frame: pd.DataFrame, utility: Mapping[str, object]) -> pd.DataFrame:
    result = frame.copy()
    required_seg = {f"conf_{i}{j}" for i in range(3) for j in range(3)}
    if not required_seg.issubset(result.columns):
        raise ValueError("per-frame segmentation confusion columns are missing")
    result["miou_frame"] = per_frame_miou(result)
    ped_den = result["tp_ped"] + result["fn_ped"]
    veh_den = result["tp_veh"] + result["fn_veh"]
    result["pedestrian_recall_frame"] = np.divide(
        result["tp_ped"], ped_den, out=np.full(len(result), np.nan), where=ped_den > 0
    )
    result["vehicle_recall_frame"] = np.divide(
        result["tp_veh"], veh_den, out=np.full(len(result), np.nan), where=veh_den > 0
    )
    weights = utility["weights"]
    refs = utility["references"]
    numerator = float(weights["miou"]) * result["miou_frame"] / float(refs["miou"])
    denominator = np.full(len(result), float(weights["miou"]))
    for metric, present in (
        ("pedestrian_recall", ped_den > 0),
        ("vehicle_recall", veh_den > 0),
    ):
        column = f"{metric}_frame"
        numerator += np.where(
            present,
            float(weights[metric]) * result[column].fillna(0.0) / float(refs[metric]),
            0.0,
        )
        denominator += np.where(present, float(weights[metric]), 0.0)
    result["utility_v5_frame"] = numerator / denominator

    det_num = np.zeros(len(result), dtype=float)
    det_den = np.zeros(len(result), dtype=float)
    for metric, present in (
        ("pedestrian_recall", ped_den > 0),
        ("vehicle_recall", veh_den > 0),
    ):
        det_num += np.where(
            present,
            float(weights[metric]) * result[f"{metric}_frame"].fillna(0.0) / float(refs[metric]),
            0.0,
        )
        det_den += np.where(present, float(weights[metric]), 0.0)
    result["utility_detection_only_frame"] = np.divide(
        det_num, det_den, out=np.full(len(result), np.nan), where=det_den > 0
    )
    return result


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    ordered = valid[np.argsort(values[valid])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def cluster_bootstrap_mean(
    differences: np.ndarray,
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    unique = np.unique(groups)
    if len(unique) < 2:
        return math.nan, math.nan, math.nan
    sums = np.array([differences[groups == group].sum() for group in unique], dtype=float)
    counts = np.array([(groups == group).sum() for group in unique], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    p_one_sided = (1.0 + float((boot <= 0.0).sum())) / (replicates + 1.0)
    return float(low), float(high), p_one_sided


def _profile_metric_path(root: Path, profile: str) -> Path:
    return root / profile / "metrics" / "test_learned_object_metrics.csv"


def load_common_ids(root: Path, profiles: Iterable[str], expected: int) -> set[str]:
    sets = []
    for profile in profiles:
        path = _profile_metric_path(root, profile)
        frame = pd.read_csv(path, usecols=["sample_id"])
        sets.append(set(frame["sample_id"].dropna().astype(str)))
    common = set.intersection(*sets)
    if len(common) != expected:
        raise ValueError(f"expected {expected} common sample IDs, found {len(common)}")
    return common


def load_profile_frames(config: Mapping[str, object], profiles: pd.DataFrame, common: set[str]) -> pd.DataFrame:
    paths = sorted(_resolve(config["inputs"]["perframe_glob"]).parent.glob(Path(config["inputs"]["perframe_glob"]).name))
    frames = []
    allowed_roi = {float(value) for value in config["inputs"]["allowed_roi_q"]}
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame["roi"].astype(float).isin(allowed_roi)].copy()
        frame["profile_id"] = [
            profile_id(model, quant, roi)
            for model, quant, roi in zip(frame["model"], frame["quant"], frame["roi"])
        ]
        frames.append(frame[frame["sample_id"].isin(common)])
    result = pd.concat(frames, ignore_index=True)
    counts = result.groupby(["sample_id", "profile_id"]).size()
    if len(counts) != len(common) * len(profiles) or not (counts == 1).all():
        raise ValueError("per-frame table is not a complete one-row sample x profile grid")
    return result.merge(profiles, on="profile_id", how="left", validate="many_to_one")


def build_context(config: Mapping[str, object], common: set[str]) -> pd.DataFrame:
    spec = config["contexts"]
    frame = pd.read_csv(_resolve(config["inputs"]["frame_context_csv"]))
    frame = frame[frame["sample_id"].isin(common)].copy()
    if len(frame) != len(common):
        raise ValueError("frame context does not cover the common sample set exactly once")
    parsed = frame["sample_id"].str.extract(SAMPLE_RE)
    if parsed.isna().any().any():
        raise ValueError("sample IDs do not satisfy the acquisition-window contract")
    frame["regime"] = parsed[0]
    frame["sample_index"] = parsed[1].astype(int)
    loops = int(config["split"]["declared_loops_per_regime"])
    max_index = frame.groupby("regime")["sample_index"].transform("max") + 1
    frame["trajectory_window_index"] = np.minimum(
        loops - 1, np.floor(frame["sample_index"] * loops / max_index).astype(int)
    )
    frame["trajectory_group"] = frame["regime"] + "::loop" + frame["trajectory_window_index"].astype(str)
    frame["partition"] = np.where(frame["trajectory_window_index"] % 2 == 0, "discovery", "confirmation")

    nv, nped = frame["n_inview_veh"].astype(int), frame["n_inview_ped"].astype(int)
    frame["class_mix"] = np.select(
        [(nv == 0) & (nped == 0), (nv > 0) & (nped == 0), (nv == 0) & (nped > 0)],
        ["empty", "vehicle_only", "pedestrian_only"],
        default="mixed",
    )
    frame["vulnerable_present"] = np.where(nped > 0, "present", "absent")
    near, mid = [float(value) for value in spec["nearest_range_edges_m"]]
    distance = pd.to_numeric(frame["gt_dist_min_m"], errors="coerce")
    frame["nearest_range"] = np.select(
        [distance.isna(), distance <= near, distance <= mid],
        ["empty", f"le_{near:g}m", f"{near:g}_{mid:g}m"],
        default=f"gt_{mid:g}m",
    )

    reference = str(config["inputs"]["reference_profile"])
    objects = pd.read_csv(_profile_metric_path(_resolve(config["inputs"]["per_object_root"]), reference))
    objects = objects[objects["sample_id"].isin(common)].copy()
    gt = objects[objects["match_status"].isin(["tp", "fn"])].copy()
    low_threshold = float(spec["low_confidence_threshold"])
    gt["low_or_missed"] = (gt["match_status"] == "fn") | (pd.to_numeric(gt["score"], errors="coerce") < low_threshold)
    gt["small"] = (
        pd.to_numeric(gt["gt_bbox_h"], errors="coerce")
        * pd.to_numeric(gt["gt_bbox_w"], errors="coerce")
        < float(spec["small_object_area_px2"])
    )
    margin = float(spec["edge_margin_px"])
    x0 = gt["gt_center_x"] - gt["gt_bbox_w"] / 2.0
    x1 = gt["gt_center_x"] + gt["gt_bbox_w"] / 2.0
    y0 = gt["gt_center_y"] - gt["gt_bbox_h"] / 2.0
    y1 = gt["gt_center_y"] + gt["gt_bbox_h"] / 2.0
    gt["truncated"] = (x0 <= margin) | (y0 <= margin) | (x1 >= gt["orig_w"] - margin) | (y1 >= gt["orig_h"] - margin)
    diagnostics = gt.groupby("sample_id").agg(
        reference_low_confidence_or_miss=("low_or_missed", "max"),
        small_object_present=("small", "max"),
        edge_truncated_present=("truncated", "max"),
    )
    frame = frame.merge(diagnostics, on="sample_id", how="left", validate="one_to_one")
    for column in ("reference_low_confidence_or_miss", "small_object_present", "edge_truncated_present"):
        frame[column] = frame[column].eq(True).map({True: "present", False: "absent"})
    return frame


def _winner(means: pd.Series, payloads: Mapping[str, float]) -> str:
    return max(means.index, key=lambda value: (float(means[value]), -payloads[value], value))


def evaluate_family(
    utility_wide: pd.DataFrame,
    context: pd.DataFrame,
    payloads: Mapping[str, float],
    family: str,
    config: Mapping[str, object],
    metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gate = config["positive_gate"]
    merged = context[["sample_id", "partition", "trajectory_group", family]].merge(
        utility_wide, on="sample_id", how="inner", validate="one_to_one"
    )
    discovery = merged[merged["partition"] == "discovery"]
    confirmation = merged[merged["partition"] == "confirmation"]
    budget_rows: list[dict] = []
    winner_rows: list[dict] = []
    budgets = sorted(set(payloads.values()))
    for budget_index, budget in enumerate(budgets):
        feasible = sorted(profile for profile, payload in payloads.items() if payload <= budget + 1e-12)
        global_means = discovery[feasible].mean()
        global_winner = _winner(global_means, payloads)
        mapping: Dict[str, str] = {}
        for category, group in discovery.groupby(family, dropna=False):
            winner = _winner(group[feasible].mean(), payloads)
            mapping[str(category)] = winner
            winner_rows.append(
                {
                    "metric": metric,
                    "context_family": family,
                    "budget_kib": budget,
                    "context_value": str(category),
                    "discovery_frames": len(group),
                    "global_profile": global_winner,
                    "context_profile": winner,
                }
            )
        selected = confirmation[family].astype(str).map(mapping).fillna(global_winner)
        row_index = np.arange(len(confirmation))
        profile_columns = {name: confirmation[name].to_numpy(dtype=float) for name in feasible}
        contextual_values = np.array([profile_columns[name][i] for i, name in enumerate(selected)], dtype=float)
        global_values = profile_columns[global_winner]
        differences = contextual_values - global_values
        changed = selected.to_numpy() != global_winner
        finite = np.isfinite(differences)
        groups = confirmation["trajectory_group"].to_numpy()[finite]
        diff = differences[finite]
        low, high, p_value = cluster_bootstrap_mean(
            diff,
            groups,
            int(config["bootstrap_replicates"]),
            int(config["seed"]) + budget_index + int(hashlib.sha256(family.encode()).hexdigest()[:6], 16),
        )
        budget_rows.append(
            {
                "metric": metric,
                "context_family": family,
                "budget_kib": budget,
                "feasible_profile_count": len(feasible),
                "global_profile": global_winner,
                "confirmation_frames": int(finite.sum()),
                "confirmation_trajectory_groups": int(len(np.unique(groups))),
                "action_change_fraction": float(changed[finite].mean()) if finite.any() else math.nan,
                "mean_global_utility": float(np.nanmean(global_values[finite])) if finite.any() else math.nan,
                "mean_contextual_utility": float(np.nanmean(contextual_values[finite])) if finite.any() else math.nan,
                "mean_utility_lift": float(np.nanmean(diff)) if finite.any() else math.nan,
                "ci95_low": low,
                "ci95_high": high,
                "p_one_sided": p_value,
                "enough_frames": int(finite.sum()) >= int(gate["minimum_confirmation_frames"]),
                "enough_groups": len(np.unique(groups)) >= int(gate["minimum_confirmation_trajectory_groups"]),
            }
        )
    result = pd.DataFrame(budget_rows)
    eligible = result["enough_frames"] & result["enough_groups"]
    result["p_holm"] = math.nan
    result.loc[eligible, "p_holm"] = holm_adjust(result.loc[eligible, "p_one_sided"])
    result["practical_reversal"] = (
        eligible
        & (result["action_change_fraction"] >= float(gate["minimum_action_change_fraction"]))
        & (result["mean_utility_lift"] >= float(gate["minimum_absolute_utility_lift"]))
        & (result["ci95_low"] > 0.0)
        & (result["p_holm"] <= float(gate["holm_familywise_alpha"]))
    )
    return result, pd.DataFrame(winner_rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def run(config_path: Path = DEFAULT_CONFIG) -> Path:
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Task A config schema_version must be 1")
    matrix_path = _resolve(config["inputs"]["matrix_markdown"])
    profiles = parse_matrix_payloads(matrix_path)
    object_root = _resolve(config["inputs"]["per_object_root"])
    common = load_common_ids(
        object_root,
        profiles["profile_id"],
        int(config["inputs"]["expected_common_sample_ids"]),
    )
    frames = load_profile_frames(config, profiles, common)
    frames = add_utilities(frames, config["utility"])
    context = build_context(config, common)
    frames = frames.merge(
        context[["sample_id", "partition", "trajectory_group"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    payloads = dict(zip(profiles["profile_id"], profiles["payload_kib"]))

    all_results = []
    all_winners = []
    families = list(config["contexts"]["primary_families"]) + list(config["contexts"]["supporting_families"])
    for metric in ("utility_v5_frame", "utility_detection_only_frame"):
        wide = frames.pivot(index="sample_id", columns="profile_id", values=metric).reset_index()
        for family in families:
            result, winners = evaluate_family(wide, context, payloads, family, config, metric)
            all_results.append(result)
            all_winners.append(winners)
    results = pd.concat(all_results, ignore_index=True)
    winners = pd.concat(all_winners, ignore_index=True)
    primary = set(config["contexts"]["primary_families"])
    primary_pass = results[
        (results["metric"] == "utility_v5_frame")
        & results["context_family"].isin(primary)
        & results["practical_reversal"]
    ]
    verdict = (
        "POSITIVE_CONTEXTUAL_OPPORTUNITY"
        if not primary_pass.empty
        else "NO_PRACTICAL_REVERSAL_ON_AVAILABLE_CONTEXTS"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "rl_agent" / "contextual_knob" / "experiments" / stamp
    (run_dir / "figures").mkdir(parents=True)
    profiles.to_csv(run_dir / "profile_costs.csv", index=False)
    context.to_csv(run_dir / "frame_context.csv", index=False)
    results.to_csv(run_dir / "budget_context_results.csv", index=False)
    winners.to_csv(run_dir / "context_winners.csv", index=False)
    config_copy = dict(config)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config_copy, sort_keys=False), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    primary_frame = results[
        (results["metric"] == "utility_v5_frame") & results["context_family"].isin(primary)
    ]
    for family, group in primary_frame.groupby("context_family"):
        axis.plot(group["budget_kib"], group["mean_utility_lift"], marker="o", markersize=3, label=family)
    axis.axhline(float(config["positive_gate"]["minimum_absolute_utility_lift"]), color="#C44E52", linestyle="--", label="practical gate")
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Fixed payload budget (KiB/frame)")
    axis.set_ylabel("Held-out contextual minus global utility")
    axis.set_title("Task A: held-out scene-conditioned lookup lift")
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(run_dir / "figures" / f"contextual_lookup_lift.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(figure)

    best = primary_frame.sort_values("mean_utility_lift", ascending=False).head(10).copy()
    display = best[
        ["context_family", "budget_kib", "global_profile", "action_change_fraction", "mean_utility_lift", "ci95_low", "ci95_high", "p_holm", "practical_reversal"]
    ].round(5)
    report = "\n".join(
        [
            "# Task A — argmax-stability / rank-reversal result",
            "",
            f"**Verdict:** `{verdict}`.",
            "",
            f"The exact registered intersection contains **{len(common):,} sample IDs** and all 36 published profiles. "
            f"Per-frame segmentation was already present and validated structurally, so incremental segmentation re-evaluation cost was **0 GPU-minutes**. "
            "A clean 36-profile regeneration is estimated at 35–45 GPU-minutes from the recorded 72-profile runtime.",
            "",
            "The primary result is seg-inclusive reward-v5 utility. Detection-only results are retained as a diagnostic, not allowed to close Phase 1 on their own.",
            "",
            "## Strongest primary-family cells",
            "",
            _markdown_table(display),
            "",
            "## Interpretation",
            "",
            (
                "At least one pre-registered class-mix/range lookup cleared the held-out practical gate. This establishes a scene-conditioned profile-selection opportunity, but does not establish sequential value or justify RL. Proceed to the registered three-way lookup ladder."
                if verdict == "POSITIVE_CONTEXTUAL_OPPORTUNITY"
                else
                "No available primary context cleared the pre-registered held-out practical gate. Because segmentation was included, this is stronger than a detection-only null, but remains scoped to the available class/range contexts and measured profiles. True occlusion, cyclists, and broader scenarios were not tested."
            ),
            "",
            "`edge_truncated_present` is an image-boundary truncation proxy, not an occlusion label. `reference_low_confidence_or_miss` uses the full-quality reference output and is supporting/non-deployable.",
            "",
            "## Artifacts",
            "",
            "See `budget_context_results.csv`, `context_winners.csv`, `frame_context.csv`, `profile_costs.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/`.",
            "",
        ]
    )
    (run_dir / "TASK_A_RESULTS.md").write_text(report, encoding="utf-8")
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(run_dir))] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    source_hashes = {
        str(matrix_path.relative_to(REPO_ROOT)): _sha256(matrix_path),
        str(config_path.relative_to(REPO_ROOT)): _sha256(config_path),
        str(Path(__file__).relative_to(REPO_ROOT)): _sha256(Path(__file__)),
        str(_resolve(config["inputs"]["frame_context_csv"]).relative_to(REPO_ROOT)): _sha256(
            _resolve(config["inputs"]["frame_context_csv"])
        ),
    }
    perframe_pattern = _resolve(config["inputs"]["perframe_glob"])
    for path in sorted(perframe_pattern.parent.glob(perframe_pattern.name)):
        source_hashes[str(path.relative_to(REPO_ROOT))] = _sha256(path)
    for profile in profiles["profile_id"]:
        path = _profile_metric_path(object_root, profile)
        source_hashes[str(path.relative_to(REPO_ROOT))] = _sha256(path)
    manifest = {
        "schema_version": 1,
        "analysis_id": config["analysis_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "common_sample_ids": len(common),
        "profile_count": len(profiles),
        "primary_practical_reversal_cells": len(primary_pass),
        "preregistration_file": "rl_agent/contextual_knob/TASK_A_PREREGISTRATION.md",
        "preregistration_sha256": _sha256(REPO_ROOT / "rl_agent/contextual_knob/TASK_A_PREREGISTRATION.md"),
        "config_sha256": _sha256(config_path),
        "source_hashes": source_hashes,
        "files": files,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
