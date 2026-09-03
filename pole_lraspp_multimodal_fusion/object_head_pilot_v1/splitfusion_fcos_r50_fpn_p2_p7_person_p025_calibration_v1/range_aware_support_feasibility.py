"""Train-holdout feasibility study for a range-aware person semantic-support gate.

The deployed person path gates every candidate at `semantic_support >= 0.10`. The
completed distance audit showed that this gate, not the detector score, removes 88% of
the recoverable 30-40 m pedestrians. This study asks one bounded question: does relaxing
the support threshold only for candidates whose predicted radial distance is >= 30 m
recover long-range recall without damaging the near field?

Everything else is frozen and unchanged: the candidate cache, the AVO table, the
observable-first matching order, `score >= 0.20` before consolidation, the grouping
configuration, and the final p025 threshold. Episode 03 is the development episode;
episode 04 is scored exactly once and only if episode 03 yields a feasible policy.

Feasibility only. No training, no cache rebuild, no model forward pass, no CUDA, no
validation or test access, and no change to any deployed artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    CANONICAL_SCORE_THRESHOLD,
    consolidate_person_candidates,
    validate_configuration,
)

from .distance_failure_audit import (
    OVERFLOW_BAND,
    PREDICTION_BANDS,
    REGISTERED_COUNTS,
    join_ground_truth,
    load_avo_table,
    score_stage,
    verify_hashes,
    verify_raw_metadata,
)
from .policy import PERSON_SCORE_THRESHOLD
from .qualification import (
    AVO_THRESHOLD,
    MATCH_RADIUS_M,
    OUTPUT_DIR as P025_OUTPUT_DIR,
    REPO_ROOT,
    SELECTED_RULE,
    load_contract,
    load_holdout_cache,
    load_holdout_raw,
    read_json,
)

FEASIBILITY_OUTPUT_DIR = (
    REPO_ROOT / "experiments/splitfusion_fcos_person_range_aware_support_feasibility_v1"
)
P025_QUALIFICATION_PATH = P025_OUTPUT_DIR / "train_holdout_qualification.json"

DEVELOPMENT_EPISODE = "canonical_v3_03_train_30_30_s503_tm1503"
CONFIRMATION_EPISODE = "canonical_v3_04_train_50_50_s504_tm1504"

# The range-aware rule. Candidates below the boundary keep the deployed threshold; only
# candidates at or beyond it see the relaxed one. The boundary uses predicted radial
# distance, which is a runtime-computable candidate property, never ground truth.
RANGE_BOUNDARY_M = 30.0
NEAR_SUPPORT_THRESHOLD = 0.10

# The grouping stage is applied to the admitted subset through the frozen preregistered
# configuration (None, 0.20); the support gate is applied beforehand, per candidate.
GROUPING_RULE = {
    "grid_index": 3,
    "semantic_support_threshold": None,
    "group_box_iou_threshold": SELECTED_RULE["group_box_iou_threshold"],
}

BASELINE_KEY = "baseline"
POLICIES = (
    {"key": BASELINE_KEY, "label": "unchanged 0.10 everywhere", "long_range_support_threshold": 0.10},
    {"key": "A", "label": ">= 30 m support 0.075", "long_range_support_threshold": 0.075},
    {"key": "B", "label": ">= 30 m support 0.050", "long_range_support_threshold": 0.050},
    {"key": "C", "label": ">= 30 m support 0.025", "long_range_support_threshold": 0.025},
)
CANDIDATE_KEYS = tuple(policy["key"] for policy in POLICIES if policy["key"] != BASELINE_KEY)

NEAR_BANDS = ("00_10m", "10_20m", "20_30m")
LONG_RANGE_BANDS = ("30_35m", "35_40m")

LONG_RANGE_PRECISION_MINIMUM = 0.70
LONG_RANGE_RECALL_MINIMUM = 0.70
AGGREGATE_PRECISION_MINIMUM = 0.70
AGGREGATE_RECALL_MINIMUM = 0.70
NEAR_DEGRADATION_MAXIMUM = 0.01

FEASIBLE = "RANGE_AWARE_PERSON_SUPPORT_HOLDOUT_FEASIBLE"
NOT_FEASIBLE = "RANGE_AWARE_PERSON_SUPPORT_NOT_FEASIBLE_RETAIN_30M_PRIMARY_RANGE"


class FeasibilityError(RuntimeError):
    """Fail-closed feasibility input, contract, or accounting error."""


def predicted_distances(frame: Mapping[str, Any], camera_xy: tuple[float, float]) -> torch.Tensor:
    world = frame["world_xy"].detach().double().cpu()
    offset = world - torch.tensor(camera_xy, dtype=torch.float64)
    return torch.linalg.vector_norm(offset, dim=1)


def policy_positions(
    frame: Mapping[str, Any], camera_xy: tuple[float, float], long_range_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retained p025 positions and the admitted candidate set under one range-aware gate.

    The per-candidate support gate is applied first, then the frozen consolidation runs on
    the admitted subset under the preregistered (None, 0.20) configuration. That is exactly
    equivalent to a per-candidate support threshold inside the frozen rule: `score >= 0.20`
    still selects the eligible set, and grouping, winner choice, and ordering are the
    unmodified implementation.
    """
    scores = frame["scores"].detach().float().cpu()
    support = frame["semantic_support"].detach().float().cpu()
    original = frame["original_indices"].detach().long().cpu()
    distance = predicted_distances(frame, camera_xy)
    threshold = torch.where(
        distance >= RANGE_BOUNDARY_M,
        torch.tensor(float(long_range_threshold), dtype=torch.float32),
        torch.tensor(float(NEAR_SUPPORT_THRESHOLD), dtype=torch.float32),
    )
    admitted = torch.where(support >= threshold)[0]

    validate_configuration(GROUPING_RULE)
    within = consolidate_person_candidates(
        scores=scores.index_select(0, admitted),
        boxes=frame["boxes"].index_select(0, admitted),
        world_xy=frame["world_xy"].index_select(0, admitted),
        component_ids=frame["component_ids"].index_select(0, admitted),
        semantic_support=support.index_select(0, admitted),
        original_indices=original.index_select(0, admitted),
        semantic_support_threshold=GROUPING_RULE["semantic_support_threshold"],
        group_box_iou_threshold=GROUPING_RULE["group_box_iou_threshold"],
    )
    selected = admitted.index_select(0, within)
    retained = scores.index_select(0, selected)
    if retained.numel() and not bool((retained >= CANONICAL_SCORE_THRESHOLD).all()):
        raise FeasibilityError("consolidation retained a person below 0.20")
    return selected.index_select(0, torch.where(retained >= PERSON_SCORE_THRESHOLD)[0]), admitted


def frozen_p025_positions(frame: Mapping[str, Any]) -> torch.Tensor:
    """The unchanged deployed selection, used as the per-frame equivalence reference."""
    scores = frame["scores"].detach().float().cpu()
    selected = consolidate_person_candidates(
        scores=frame["scores"],
        boxes=frame["boxes"],
        world_xy=frame["world_xy"],
        component_ids=frame["component_ids"],
        semantic_support=frame["semantic_support"],
        original_indices=frame["original_indices"],
        semantic_support_threshold=SELECTED_RULE["semantic_support_threshold"],
        group_box_iou_threshold=SELECTED_RULE["group_box_iou_threshold"],
    )
    retained = scores.index_select(0, selected)
    return selected.index_select(0, torch.where(retained >= PERSON_SCORE_THRESHOLD)[0])


def build_positions(
    frames: Sequence[Mapping[str, Any]],
    camera_xy_by_sample: Mapping[str, tuple[float, float]],
    policy_keys: Sequence[str],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Per-frame retained positions per policy, with the baseline equivalence gate.

    Relaxing the gate is monotone on the *admitted* candidate set, and that is asserted.
    It is deliberately not asserted on the retained set: a newly admitted higher-scoring
    candidate can merge with a baseline retention under the frozen grouping rule and win
    the group, replacing it. That displacement is the frozen consolidation behaving as
    designed on a larger admitted set, so it is counted and reported rather than rejected.
    """
    selected_policies = [policy for policy in POLICIES if policy["key"] in set(policy_keys)]
    positions: dict[str, dict[str, torch.Tensor]] = {}
    displacement = {
        key: {"baseline_retentions_displaced": 0, "newly_retained": 0, "frames_affected": 0}
        for key in policy_keys
        if key != BASELINE_KEY
    }
    for frame in frames:
        sample_id = str(frame["sample_id"])
        camera_xy = camera_xy_by_sample[sample_id]
        retained: dict[str, torch.Tensor] = {}
        admitted: dict[str, torch.Tensor] = {}
        for policy in selected_policies:
            retained[policy["key"]], admitted[policy["key"]] = policy_positions(
                frame, camera_xy, policy["long_range_support_threshold"]
            )
        if not torch.equal(retained[BASELINE_KEY], frozen_p025_positions(frame)):
            raise FeasibilityError(
                f"baseline range-aware path is not bitwise equal to the frozen p025 selection: {sample_id}"
            )
        baseline_admitted = set(admitted[BASELINE_KEY].tolist())
        baseline_retained = set(retained[BASELINE_KEY].tolist())
        for key in displacement:
            if not baseline_admitted.issubset(set(admitted[key].tolist())):
                raise FeasibilityError(
                    f"relaxed policy {key} admitted fewer candidates than the baseline: {sample_id}"
                )
            lost = baseline_retained - set(retained[key].tolist())
            gained = set(retained[key].tolist()) - baseline_retained
            displacement[key]["baseline_retentions_displaced"] += len(lost)
            displacement[key]["newly_retained"] += len(gained)
            displacement[key]["frames_affected"] += int(bool(lost or gained))
        positions[sample_id] = retained
    return positions, displacement


def combine_bands(view: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Aggregate finalized per-band counts into one cumulative range scope."""
    bands = view["bands"]
    totals = {
        name: sum(int(bands[band][name]) for band in names)
        for name in ("eligible_gt", "tp_gt_band", "tp_pred_band", "fp", "fn")
    }
    tp_gt, tp_pred = totals["tp_gt_band"], totals["tp_pred_band"]
    fp, fn, eligible = totals["fp"], totals["fn"], totals["eligible_gt"]
    if tp_gt + fn != eligible:
        raise FeasibilityError("cumulative TP+FN denominator failure")
    precision = tp_pred / (tp_pred + fp) if tp_pred + fp else 0.0
    recall = tp_gt / eligible if eligible else 0.0
    # Each band's XY MAE is the mean over its GT-assigned matched pairs, so the cumulative
    # mean is the tp_gt_band-weighted mean of the band means.
    weighted = sum(
        float(bands[band]["xy_mae_m"]) * int(bands[band]["tp_gt_band"])
        for band in names
        if bands[band]["xy_mae_m"] is not None
    )
    return {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "xy_mae_m": weighted / tp_gt if tp_gt else None,
    }


def check_band_recombination(view: Mapping[str, Any]) -> None:
    """Recombining every band must reproduce the aggregate the frozen scorer computed."""
    combined = combine_bands(view, PREDICTION_BANDS)
    overall = view["overall"]
    for name in ("eligible_gt", "tp_gt_band", "tp_pred_band", "fp", "fn"):
        if int(combined[name]) != int(overall[name]):
            raise FeasibilityError(f"band recombination disagrees with the aggregate: {name}")
    if abs(float(combined["xy_mae_m"]) - float(overall["xy_mae_m"])) > 1e-9:
        raise FeasibilityError("band recombination disagrees with the aggregate XY MAE")


def summarize(view: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    check_band_recombination(view)
    keep = (
        "eligible_gt", "tp_gt_band", "tp_pred_band", "fp", "fn", "precision", "recall", "f1",
        "xy_mae_m", "reachable_gt", "candidate_recall_ceiling",
    )
    summary: dict[str, Any] = {
        "retained_predictions": int(view["retained_predictions"]),
        "bands": {
            band: {name: view["bands"][band][name] for name in keep} for band in PREDICTION_BANDS
        },
        "cumulative_le_30m": combine_bands(view, NEAR_BANDS),
        "cumulative_30_40m": combine_bands(view, LONG_RANGE_BANDS),
        "aggregate_avo": {name: view["overall"][name] for name in keep},
    }
    if baseline is None:
        summary["versus_baseline"] = None
        return summary
    scopes = {
        "aggregate_avo": ("aggregate_avo", "aggregate_avo"),
        "cumulative_le_30m": ("cumulative_le_30m", "cumulative_le_30m"),
        "cumulative_30_40m": ("cumulative_30_40m", "cumulative_30_40m"),
    }
    deltas = {}
    for name, (left, right) in scopes.items():
        current, reference = summary[left], baseline[right]
        deltas[name] = {
            "recovered_gt": int(current["tp_gt_band"]) - int(reference["tp_gt_band"]),
            "additional_fp": int(current["fp"]) - int(reference["fp"]),
            "precision_delta": float(current["precision"]) - float(reference["precision"]),
            "recall_delta": float(current["recall"]) - float(reference["recall"]),
            "f1_delta": float(current["f1"]) - float(reference["f1"]),
            "xy_mae_delta_m": (
                float(current["xy_mae_m"]) - float(reference["xy_mae_m"])
                if current["xy_mae_m"] is not None and reference["xy_mae_m"] is not None
                else None
            ),
        }
    deltas["retained_predictions_delta"] = (
        summary["retained_predictions"] - int(baseline["retained_predictions"])
    )
    summary["versus_baseline"] = deltas
    return summary


def feasibility_conditions(
    summary: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    long_range = summary["cumulative_30_40m"]
    aggregate = summary["aggregate_avo"]
    near, near_baseline = summary["cumulative_le_30m"], baseline["cumulative_le_30m"]
    precision_degradation = float(near_baseline["precision"]) - float(near["precision"])
    recall_degradation = float(near_baseline["recall"]) - float(near["recall"])
    conditions = {
        "long_range_precision_gte_0_70": float(long_range["precision"]) >= LONG_RANGE_PRECISION_MINIMUM,
        "long_range_recall_gte_0_70": float(long_range["recall"]) >= LONG_RANGE_RECALL_MINIMUM,
        "aggregate_precision_gte_0_70": float(aggregate["precision"]) >= AGGREGATE_PRECISION_MINIMUM,
        "aggregate_recall_gte_0_70": float(aggregate["recall"]) >= AGGREGATE_RECALL_MINIMUM,
        "near_precision_degradation_lte_0_01": precision_degradation <= NEAR_DEGRADATION_MAXIMUM,
        "near_recall_degradation_lte_0_01": recall_degradation <= NEAR_DEGRADATION_MAXIMUM,
    }
    return {
        "conditions": conditions,
        "all_passed": all(conditions.values()),
        "near_precision_degradation": precision_degradation,
        "near_recall_degradation": recall_degradation,
        "long_range_min_precision_recall": min(
            float(long_range["precision"]), float(long_range["recall"])
        ),
        "long_range_f1": float(long_range["f1"]),
    }


def episode_views(
    *,
    frames: Sequence[Mapping[str, Any]],
    policy_keys: Sequence[str],
    positions: Mapping[str, Mapping[str, torch.Tensor]],
    observable_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    avo_ignored_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    structural_gt: Mapping[str, Sequence[Mapping[str, Any]]],
    camera_xy_by_sample: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    views = {
        key: score_stage(
            frames=frames,
            stage_key=key,
            stage_positions_by_sample=positions,
            observable_gt=observable_gt,
            avo_ignored_gt=avo_ignored_gt,
            structural_gt=structural_gt,
            camera_xy_by_sample=camera_xy_by_sample,
        )
        for key in policy_keys
    }
    baseline = summarize(views[BASELINE_KEY], None)
    summaries = {BASELINE_KEY: baseline}
    for key in policy_keys:
        if key != BASELINE_KEY:
            summaries[key] = summarize(views[key], baseline)
    return summaries


def check_baseline_against_frozen(summary: Mapping[str, Any], episode: str) -> dict[str, Any]:
    """The baseline must reproduce the frozen per-episode p025 record exactly."""
    expected = read_json(P025_QUALIFICATION_PATH)["train_holdout"]["p025"]["episodes"][episode]
    aggregate = summary["aggregate_avo"]
    gate = bool(
        int(aggregate["eligible_gt"]) == int(expected["observable_gt"])
        and int(aggregate["tp_gt_band"]) == int(expected["tp"])
        and int(aggregate["fp"]) == int(expected["fp"])
        and int(aggregate["fn"]) == int(expected["fn"])
        and abs(float(aggregate["precision"]) - float(expected["precision"])) <= 1e-12
        and abs(float(aggregate["recall"]) - float(expected["recall"])) <= 1e-12
        and abs(float(aggregate["xy_mae_m"]) - float(expected["xy_mae_m"])) <= 1e-12
    )
    if not gate:
        raise FeasibilityError(f"baseline does not reproduce the frozen p025 record for {episode}")
    return {"episode": episode, "baseline_reproduces_frozen_p025": gate}


def rank_feasible(
    summaries: Mapping[str, Any], assessments: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Rank by min(long-range P,R), then long-range F1, then the higher support threshold."""
    thresholds = {policy["key"]: policy["long_range_support_threshold"] for policy in POLICIES}
    ranked = [
        {
            "policy": key,
            "long_range_support_threshold": thresholds[key],
            "long_range_min_precision_recall": assessments[key]["long_range_min_precision_recall"],
            "long_range_f1": assessments[key]["long_range_f1"],
        }
        for key in CANDIDATE_KEYS
        if assessments[key]["all_passed"]
    ]
    ranked.sort(
        key=lambda row: (
            -row["long_range_min_precision_recall"],
            -row["long_range_f1"],
            -row["long_range_support_threshold"],
        )
    )
    return ranked


def frontier(summaries: Mapping[str, Any], assessments: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = {policy["key"]: policy["long_range_support_threshold"] for policy in POLICIES}
    rows = []
    for key in CANDIDATE_KEYS:
        long_range = summaries[key]["cumulative_30_40m"]
        rows.append(
            {
                "policy": key,
                "long_range_support_threshold": thresholds[key],
                "long_range_precision": float(long_range["precision"]),
                "long_range_recall": float(long_range["recall"]),
                "long_range_f1": float(long_range["f1"]),
                "aggregate_precision": float(summaries[key]["aggregate_avo"]["precision"]),
                "aggregate_recall": float(summaries[key]["aggregate_avo"]["recall"]),
                "near_precision_degradation": assessments[key]["near_precision_degradation"],
                "near_recall_degradation": assessments[key]["near_recall_degradation"],
                "failed_conditions": sorted(
                    name for name, passed in assessments[key]["conditions"].items() if not passed
                ),
                "all_passed": assessments[key]["all_passed"],
            }
        )
    return rows


def _scope_row(label: str, row: Mapping[str, Any]) -> str:
    mae = "n/a" if row["xy_mae_m"] is None else f"{row['xy_mae_m']:.6f}"
    return (
        f"| {label} | {row['eligible_gt']} | {row['tp_gt_band']} | {row['tp_pred_band']} | "
        f"{row['fp']} | {row['fn']} | {row['precision']:.6f} | {row['recall']:.6f} | "
        f"{row['f1']:.6f} | {mae} |"
    )


def _policy_tables(summaries: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    lines: list[str] = []
    labels = {policy["key"]: policy["label"] for policy in POLICIES}
    thresholds = {policy["key"]: policy["long_range_support_threshold"] for policy in POLICIES}
    for key in keys:
        summary = summaries[key]
        lines.extend(
            [
                f"### {key} — {labels[key]} (>= 30 m support {thresholds[key]})",
                "",
                f"Retained p025 person predictions: {summary['retained_predictions']}.",
                "",
                "| scope | eligible GT | TP(gt) | TP(pred) | FP | FN | precision | recall | F1 | XY MAE m |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for band in PREDICTION_BANDS:
            lines.append(_scope_row(band, summary["bands"][band]))
        lines.append(_scope_row("cumulative <=30 m", summary["cumulative_le_30m"]))
        lines.append(_scope_row("cumulative 30-40 m", summary["cumulative_30_40m"]))
        lines.append(_scope_row("aggregate AVO", summary["aggregate_avo"]))
        lines.append("")
        deltas = summary["versus_baseline"]
        if deltas is not None:
            lines.extend(
                [
                    "Relative to the unchanged baseline:",
                    "",
                    "| scope | recovered GT | additional FP | precision delta | recall delta |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for scope in ("cumulative_30_40m", "cumulative_le_30m", "aggregate_avo"):
                row = deltas[scope]
                lines.append(
                    f"| {scope} | {row['recovered_gt']:+d} | {row['additional_fp']:+d} | "
                    f"{row['precision_delta']:+.6f} | {row['recall_delta']:+.6f} |"
                )
            lines.append("")
            lines.append(
                f"Retained prediction delta: {deltas['retained_predictions_delta']:+d}."
            )
            lines.append("")
    return lines


def reading_section(result: Mapping[str, Any]) -> list[str]:
    """State what the numbers mean, including where the gate failed to discriminate."""
    development = result["development_episode_03"]
    baseline_03 = development["policies"][BASELINE_KEY]["cumulative_30_40m"]
    lines = [
        "## Reading",
        "",
        "The protocol's terminal is decided on episode 03, and by that rule the answer is",
        f"`{result['terminal']}`. Three caveats belong with it.",
        "",
        "**The 30-40 m recall gate did not discriminate on episode 03.** The unchanged baseline",
        f"already scores {baseline_03['recall']:.6f} there, above the 0.70 requirement, before any policy is",
        "applied. On episode 03 the gates therefore mostly test whether relaxation damages",
        "precision, not whether it delivers recall.",
        "",
        "**The recovery is small in absolute terms.** Against"
        f" {int(baseline_03['fn'])} baseline long-range misses on episode 03,"
        " policy A recovers"
        f" {development['policies']['A']['versus_baseline']['cumulative_30_40m']['recovered_gt']}"
        " and adds"
        f" {development['policies']['A']['versus_baseline']['cumulative_30_40m']['additional_fp']} false positives;"
        " even C, the most permissive threshold tested, recovers only"
        f" {development['policies']['C']['versus_baseline']['cumulative_30_40m']['recovered_gt']}.",
        "",
    ]
    confirmation = result["confirmation_episode_04"]
    if confirmation is not None:
        baseline_04 = confirmation["policies"][BASELINE_KEY]["cumulative_30_40m"]
        selected_04 = confirmation["policies"][result["selected_policy"]]["cumulative_30_40m"]
        lines.extend(
            [
                "**Episode 04 does not corroborate.** Its unchanged baseline sits at"
                f" {baseline_04['recall']:.6f} long-range recall, already below the 0.70 gate, and the frozen"
                f" policy reaches only {selected_04['recall']:.6f}. The gate is not reachable on that episode by"
                " this family of policies at all, so the episode-03 pass reflects episode difficulty"
                " more than policy strength.",
                "",
            ]
        )
    lines.extend(
        [
            "**Threshold relaxation is the wrong lever for the audited headroom.** The completed",
            "distance audit put the 30-40 m candidate recall ceiling at 0.6878 for the deployed p025",
            "candidate set and 0.9638 when only `score >= 0.20` is applied with no support gate.",
            "Dropping the long-range support threshold all the way to 0.025 moves measured recall",
            "very little, so nearly all of that headroom sits in candidates whose semantic support is",
            "below 0.025 — the person segmentation yields no usable component at range rather than a",
            "merely weak one. Recovering it needs support-independent evidence, not a lower threshold.",
            "",
        ]
    )
    return lines


def markdown_report(result: Mapping[str, Any]) -> str:
    development = result["development_episode_03"]
    lines = [
        "# Range-aware person semantic-support gate — train-holdout feasibility",
        "",
        "One bounded feasibility study. No training, no cache rebuild, no model forward pass,",
        "no CUDA, no validation or test access, and no deployed artifact was modified.",
        "",
        "## Rule under test",
        "",
        f"- Candidates with predicted radial distance < {RANGE_BOUNDARY_M:.0f} m keep the deployed",
        f"  `semantic_support >= {NEAR_SUPPORT_THRESHOLD}` gate, unchanged.",
        f"- Candidates at or beyond {RANGE_BOUNDARY_M:.0f} m use a relaxed threshold: A = 0.075,",
        "  B = 0.050, C = 0.025. No other threshold, range boundary, score threshold, grouping",
        "  parameter, or p025 value was varied.",
        "- `score >= 0.20` before consolidation, the frozen grouping configuration, and the final",
        "  p025 threshold are unchanged. The support gate is applied per candidate and the frozen",
        "  `consolidate_person_candidates` then runs on the admitted subset under the",
        "  preregistered `(None, 0.20)` configuration, so no selection logic is re-implemented.",
        "- The boundary uses predicted radial distance, a runtime-computable candidate property.",
        "  Ground-truth distance is used only to bin recall, exactly as in the completed audit.",
        "",
        "## Validity",
        "",
        "- Every frozen input verified against its registered SHA-256, plus the raw holdout",
        "  metadata hashes against the frozen p025 `INPUT_HASHES.json`.",
        "- Per frame, the baseline path is **bitwise equal** to the frozen p025 selection, and",
        "  every relaxed policy admits a superset of the baseline candidates. Retention is",
        "  deliberately *not* asserted to be a superset: under the frozen grouping rule a newly",
        "  admitted higher-scoring candidate can merge with a baseline retention and win the",
        "  group, replacing it. Those displacements are counted below rather than rejected.",
        "- Each scored baseline reproduces the frozen per-episode p025 record exactly (observable",
        "  GT, TP, FP, FN, precision, recall, XY MAE).",
        "- Per-band counts recombine to the aggregate the frozen scorer computed, including XY MAE.",
        "- Recall bins by `gt_distance_m`; precision bins by predicted radial distance. Matching",
        f"  order preserved: observable GT, then AVO-ignore, then structural-ignore, greedy inside",
        f"  {MATCH_RADIUS_M:.1f} m at AVO >= {AVO_THRESHOLD}. The `{OVERFLOW_BAND}` row holds predictions beyond 40 m;",
        "  it carries no eligible GT but does contribute to aggregate precision.",
        "",
        "## Development on episode 03",
        "",
        f"Episode: `{DEVELOPMENT_EPISODE}`, {development['frames']} frames.",
        "",
    ]
    lines.extend(_policy_tables(development["policies"], [BASELINE_KEY, *CANDIDATE_KEYS]))
    lines.extend(
        [
            "### Feasibility frontier on episode 03",
            "",
            "Gates: 30-40 m precision >= 0.70, 30-40 m recall >= 0.70, aggregate AVO precision",
            ">= 0.70, aggregate AVO recall >= 0.70, and cumulative <=30 m precision and recall each",
            "degrading by at most 0.01 absolute from the unchanged baseline.",
            "",
            "| policy | >=30 m support | 30-40 m P | 30-40 m R | 30-40 m F1 | agg P | agg R | <=30 m P loss | <=30 m R loss | passed | failed conditions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in development["frontier"]:
        failed = ", ".join(row["failed_conditions"]) if row["failed_conditions"] else "—"
        lines.append(
            f"| {row['policy']} | {row['long_range_support_threshold']} | "
            f"{row['long_range_precision']:.6f} | {row['long_range_recall']:.6f} | "
            f"{row['long_range_f1']:.6f} | {row['aggregate_precision']:.6f} | "
            f"{row['aggregate_recall']:.6f} | {row['near_precision_degradation']:+.6f} | "
            f"{row['near_recall_degradation']:+.6f} | {'yes' if row['all_passed'] else 'no'} | {failed} |"
        )
    lines.extend(
        [
            "",
            "### Grouping displacement on episode 03",
            "",
            "| policy | frames affected | newly retained | baseline retentions displaced |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in CANDIDATE_KEYS:
        row = development["grouping_displacement"][key]
        lines.append(
            f"| {key} | {row['frames_affected']} | {row['newly_retained']} | "
            f"{row['baseline_retentions_displaced']} |"
        )
    lines.append("")

    confirmation = result["confirmation_episode_04"]
    if confirmation is None:
        lines.extend(
            [
                "## Episode 04",
                "",
                "**Not accessed.** No episode-03 policy was feasible, so no episode-04 metric was",
                "computed and no policy was frozen.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Confirmation on episode 04",
                "",
                f"Frozen policy `{result['selected_policy']}` was chosen on episode 03 alone and scored",
                f"exactly once on `{CONFIRMATION_EPISODE}` ({confirmation['frames']} frames). The choice",
                "was not revised afterwards.",
                "",
            ]
        )
        lines.extend(_policy_tables(confirmation["policies"], [BASELINE_KEY, result["selected_policy"]]))
        assessment = confirmation["assessment"]
        lines.extend(
            [
                "### Whether the episode-03 conditions also hold on episode 04",
                "",
                "| condition | holds |",
                "|---|---|",
            ]
        )
        for name, passed in sorted(assessment["conditions"].items()):
            lines.append(f"| {name} | {'yes' if passed else 'no'} |")
        lines.append("")
        lines.append(
            "This is a report-only confirmation, not a second selection gate."
            if assessment["all_passed"]
            else "**The frozen policy does not satisfy every condition on episode 04.** The choice "
            "stands as made on episode 03; this disagreement is the finding, not a reason to retune."
        )
        lines.append("")

    lines.extend(reading_section(result))
    lines.extend(
        [
            "## Scope limits",
            "",
            "- Feasibility only, on two train-holdout episodes. No validation or test claim, and",
            "  nothing here authorizes a runtime, threshold, checkpoint, forward-lock, segmentation,",
            "  AE, or Phase-10B change.",
            "- Only the three registered alternatives were evaluated. The range boundary, score",
            "  threshold, grouping parameters, and p025 value were not searched.",
            "",
            result["terminal"],
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FEASIBILITY_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FeasibilityError('refusing to run without CUDA_VISIBLE_DEVICES=""')
    output = args.output.resolve()
    if output != FEASIBILITY_OUTPUT_DIR.resolve():
        raise FeasibilityError("output must be the registered feasibility directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    hashes = verify_hashes()
    _feasibility, cache_manifest, episodes = load_contract()
    if set(episodes) != {DEVELOPMENT_EPISODE, CONFIRMATION_EPISODE}:
        raise FeasibilityError("train-holdout episode identity drift")
    frames, cache_counts = load_holdout_cache(cache_manifest, episodes)
    if (
        cache_counts["frames"] != REGISTERED_COUNTS["cache_frames"]
        or cache_counts["person_candidates"] != REGISTERED_COUNTS["cache_person_candidates"]
    ):
        raise FeasibilityError("holdout candidate-cache cardinality drift")
    frame_set = {str(frame["sample_id"]) for frame in frames}
    raw, raw_hashes = load_holdout_raw(frame_set, episodes)
    verify_raw_metadata(raw_hashes, hashes.pop("raw_holdout_metadata_registered"))
    table = load_avo_table()
    observable_gt, avo_ignored_gt, structural_gt, gt_diagnostics = join_ground_truth(
        table, raw, frame_set
    )
    camera_xy_by_sample = {
        sample_id: (float(meta["camera_x"]), float(meta["camera_y"]))
        for sample_id, meta in raw["manifest_by_sample"].items()
    }
    by_episode = {
        episode: [frame for frame in frames if str(frame["experiment_id"]) == episode]
        for episode in episodes
    }

    development_frames = by_episode[DEVELOPMENT_EPISODE]
    development_positions, development_displacement = build_positions(
        development_frames, camera_xy_by_sample, [BASELINE_KEY, *CANDIDATE_KEYS]
    )
    development_summaries = episode_views(
        frames=development_frames,
        policy_keys=[BASELINE_KEY, *CANDIDATE_KEYS],
        positions=development_positions,
        observable_gt=observable_gt,
        avo_ignored_gt=avo_ignored_gt,
        structural_gt=structural_gt,
        camera_xy_by_sample=camera_xy_by_sample,
    )
    development_gate = check_baseline_against_frozen(
        development_summaries[BASELINE_KEY], DEVELOPMENT_EPISODE
    )
    baseline_assessment = feasibility_conditions(
        development_summaries[BASELINE_KEY], development_summaries[BASELINE_KEY]
    )
    assessments = {
        key: feasibility_conditions(development_summaries[key], development_summaries[BASELINE_KEY])
        for key in CANDIDATE_KEYS
    }
    ranked = rank_feasible(development_summaries, assessments)
    selected = ranked[0]["policy"] if ranked else None

    confirmation: dict[str, Any] | None = None
    if selected is not None:
        confirmation_frames = by_episode[CONFIRMATION_EPISODE]
        confirmation_positions, confirmation_displacement = build_positions(
            confirmation_frames, camera_xy_by_sample, [BASELINE_KEY, selected]
        )
        confirmation_summaries = episode_views(
            frames=confirmation_frames,
            policy_keys=[BASELINE_KEY, selected],
            positions=confirmation_positions,
            observable_gt=observable_gt,
            avo_ignored_gt=avo_ignored_gt,
            structural_gt=structural_gt,
            camera_xy_by_sample=camera_xy_by_sample,
        )
        confirmation_gate = check_baseline_against_frozen(
            confirmation_summaries[BASELINE_KEY], CONFIRMATION_EPISODE
        )
        confirmation = {
            "episode": CONFIRMATION_EPISODE,
            "frames": len(confirmation_frames),
            "policies": confirmation_summaries,
            "assessment": feasibility_conditions(
                confirmation_summaries[selected], confirmation_summaries[BASELINE_KEY]
            ),
            "baseline_gate": confirmation_gate,
            "grouping_displacement": confirmation_displacement,
            "evaluations": 1,
            "choice_revised_after_seeing_episode_04": False,
        }

    result = {
        "schema": "splitfusion_fcos_person_range_aware_support_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": FEASIBLE if selected is not None else NOT_FEASIBLE,
        "selected_policy": selected,
        "study_type": "train_holdout_feasibility_only",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_run": False,
        "cache_rebuilt": False,
        "model_inference_run": False,
        "validation_accessed": False,
        "test_accessed": False,
        "deployed_runtime_modified": False,
        "rule": {
            "range_boundary_m": RANGE_BOUNDARY_M,
            "boundary_quantity": "predicted radial distance from the camera origin, world plane",
            "near_support_threshold": NEAR_SUPPORT_THRESHOLD,
            "score_threshold_before_consolidation": CANONICAL_SCORE_THRESHOLD,
            "grouping_rule": GROUPING_RULE,
            "frozen_grouping_rule": dict(SELECTED_RULE),
            "person_output_threshold": PERSON_SCORE_THRESHOLD,
            "policies": list(POLICIES),
        },
        "gates": {
            "long_range_precision_minimum": LONG_RANGE_PRECISION_MINIMUM,
            "long_range_recall_minimum": LONG_RANGE_RECALL_MINIMUM,
            "aggregate_precision_minimum": AGGREGATE_PRECISION_MINIMUM,
            "aggregate_recall_minimum": AGGREGATE_RECALL_MINIMUM,
            "near_degradation_maximum": NEAR_DEGRADATION_MAXIMUM,
        },
        "band_assignment": {
            "recall": "gt_distance_m",
            "precision": "predicted world-plane radial distance from the camera origin",
        },
        "matching_order": "observable_gt_then_avo_ignored_gt_then_structural_ignored_gt",
        "avo_threshold": AVO_THRESHOLD,
        "match_radius_m": MATCH_RADIUS_M,
        "cache_counts": cache_counts,
        "ground_truth": gt_diagnostics,
        "development_episode_03": {
            "episode": DEVELOPMENT_EPISODE,
            "frames": len(development_frames),
            "policies": development_summaries,
            "baseline_assessment": baseline_assessment,
            "assessments": assessments,
            "frontier": frontier(development_summaries, assessments),
            "ranked_feasible": ranked,
            "baseline_gate": development_gate,
            "grouping_displacement": development_displacement,
        },
        "confirmation_episode_04": confirmation,
        "episode_04_accessed": confirmation is not None,
        "input_hashes": {**hashes, "raw_holdout_metadata_sha256": raw_hashes},
    }
    result["runtime_seconds"] = time.perf_counter() - started
    with (output / "range_aware_support_feasibility.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    (output / "RANGE_AWARE_SUPPORT_FEASIBILITY.md").write_text(
        markdown_report(result), encoding="utf-8"
    )
    (output / result["terminal"]).write_text(result["terminal"] + "\n", encoding="utf-8")
    print(json.dumps({
        "terminal": result["terminal"],
        "selected_policy": selected,
        "runtime_seconds": result["runtime_seconds"],
        "frontier": result["development_episode_03"]["frontier"],
        "ranked_feasible": ranked,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
