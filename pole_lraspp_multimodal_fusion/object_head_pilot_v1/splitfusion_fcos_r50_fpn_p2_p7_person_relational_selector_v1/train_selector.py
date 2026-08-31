from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.core import (
    rematch_person_frame,
)

from .cache_join import JoinedFrame, iter_joined_frames, pad_frames
from .provenance import (
    CONSOLIDATION_MANIFEST_SHA256,
    FROZEN_CHECKPOINT_SHA256,
    HOLDOUT_EXPERIMENT_IDS,
    ROI_MANIFEST_SHA256,
    LockedCaches,
    load_locked_caches,
)
from .selector import (
    ARCHITECTURE,
    PersonRelationalSelector,
    build_selector_optimizer,
    refined_person_logits,
    refined_person_scores,
)

EPOCHS = 5
BATCH_FRAMES = 16
NEGATIVES_PER_POSITIVE = 3
SEED = 20260831
CANONICAL_THRESHOLD = 0.20


def _require_finite_parameters(selector: PersonRelationalSelector, *, gradients: bool) -> None:
    for name, parameter in selector.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None or not bool(torch.isfinite(value).all()):
            kind = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"non-finite selector {kind}: {name}")


def _frame_key(frame: JoinedFrame) -> tuple[str, str]:
    return frame.sample_id, frame.experiment_id


def _sampling_plans(
    caches: LockedCaches,
) -> tuple[list[dict[tuple[str, str], torch.Tensor]], int, int]:
    """Create all fixed epoch plans from one metadata/label-only cache scan."""
    positive_by_frame: list[tuple[tuple[str, str], torch.Tensor]] = []
    negative_by_frame: list[tuple[tuple[str, str], torch.Tensor]] = []
    seen: set[tuple[str, str]] = set()
    for frame in iter_joined_frames(caches, include_features=False):
        if frame.partition != 0:
            continue
        key = _frame_key(frame)
        if key in seen:
            raise RuntimeError("fit frame identity is not unique")
        seen.add(key)
        positive_by_frame.append((key, torch.where(frame.labels == 1)[0]))
        negative_by_frame.append((key, torch.where(frame.labels == 0)[0]))
    positive_count = sum(indices.numel() for _key, indices in positive_by_frame)
    negative_count = sum(indices.numel() for _key, indices in negative_by_frame)
    sampled_negative_count = NEGATIVES_PER_POSITIVE * positive_count
    if positive_count == 0 or negative_count < sampled_negative_count:
        raise RuntimeError("fit cache cannot support the fixed 1:3 positive/negative loss sample")

    epoch_plans: list[dict[tuple[str, str], torch.Tensor]] = []
    for epoch in range(1, EPOCHS + 1):
        generator = torch.Generator().manual_seed(SEED + epoch)
        selected_global = torch.randperm(negative_count, generator=generator)[:sampled_negative_count]
        plans = {key: positive.clone() for key, positive in positive_by_frame}
        offset = 0
        for key, negative in negative_by_frame:
            selected = selected_global[
                (selected_global >= offset) & (selected_global < offset + negative.numel())
            ] - offset
            sampled = negative.index_select(0, selected)
            plans[key] = torch.cat((plans[key], sampled)).sort().values
            offset += negative.numel()
        if sum(indices.numel() for indices in plans.values()) != positive_count + sampled_negative_count:
            raise RuntimeError("deterministic 1:3 sampling plan did not reconcile")
        epoch_plans.append(plans)
    return epoch_plans, positive_count, sampled_negative_count


def _optimize_batch(
    selector: PersonRelationalSelector,
    optimizer: torch.optim.Optimizer,
    frames: list[JoinedFrame],
    plans: Mapping[tuple[str, str], torch.Tensor],
    device: torch.device,
) -> tuple[float, int]:
    features, base_scores, labels, padding = pad_frames(frames, device)
    loss_mask = torch.zeros_like(padding)
    for batch_index, frame in enumerate(frames):
        selected = plans[_frame_key(frame)].to(device)
        loss_mask[batch_index, selected] = True
    if bool((loss_mask & padding).any()) or not bool(loss_mask.any()):
        raise RuntimeError("loss sampling mask is empty or selects padding")
    optimizer.zero_grad(set_to_none=True)
    residual = selector(features, padding)
    logits = refined_person_logits(base_scores, residual)
    loss = F.binary_cross_entropy_with_logits(logits[loss_mask], labels[loss_mask].float())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite relational-selector loss")
    loss.backward()
    _require_finite_parameters(selector, gradients=True)
    optimizer.step()
    _require_finite_parameters(selector, gradients=False)
    count = int(loss_mask.sum())
    return float(loss.detach()) * count, count


def _train_epoch(
    selector: PersonRelationalSelector,
    optimizer: torch.optim.Optimizer,
    caches: LockedCaches,
    plans: Mapping[tuple[str, str], torch.Tensor],
    device: torch.device,
) -> float:
    selector.train()
    batch: list[JoinedFrame] = []
    loss_sum = 0.0
    example_count = 0
    visited: set[tuple[str, str]] = set()
    for frame in iter_joined_frames(caches):
        if frame.partition != 0:
            continue
        key = _frame_key(frame)
        selected = plans.get(key)
        if selected is None:
            raise RuntimeError("fit sampling plan/frame identity mismatch")
        visited.add(key)
        if selected.numel() == 0:
            continue
        # Every candidate in this frame remains in attention context; only the
        # loss mask is subsampled and ignored labels are never selected.
        batch.append(frame)
        if len(batch) == BATCH_FRAMES:
            weighted, count = _optimize_batch(selector, optimizer, batch, plans, device)
            loss_sum += weighted
            example_count += count
            batch.clear()
    if batch:
        weighted, count = _optimize_batch(selector, optimizer, batch, plans, device)
        loss_sum += weighted
        example_count += count
    if visited != set(plans) or example_count != sum(value.numel() for value in plans.values()):
        raise RuntimeError("fit epoch did not consume its exact sampling plan")
    return loss_sum / example_count


@dataclass(frozen=True)
class ScoredHoldoutFrame:
    sample_id: str
    experiment_id: str
    original_indices: torch.Tensor
    boxes: torch.Tensor
    world_xy: torch.Tensor
    ignore_flags: torch.Tensor
    gt_world_xy: torch.Tensor
    base_scores: torch.Tensor
    residual_logits: torch.Tensor
    refined_scores: torch.Tensor

    @property
    def candidate_count(self) -> int:
        return int(self.refined_scores.numel())


def _holdout_outputs(
    selector: PersonRelationalSelector,
    caches: LockedCaches,
    device: torch.device,
) -> list[ScoredHoldoutFrame]:
    """Compute each holdout candidate score once and retain rematching fields."""
    scored: list[ScoredHoldoutFrame] = []
    eligible_by_episode = {experiment_id: 0 for experiment_id in HOLDOUT_EXPERIMENT_IDS}
    selector.eval()
    with torch.inference_mode():
        for frame in iter_joined_frames(caches):
            if frame.partition != 1:
                continue
            if frame.experiment_id not in eligible_by_episode:
                raise RuntimeError("unexpected holdout episode")
            eligible_by_episode[frame.experiment_id] += frame.eligible_positive_count
            if frame.candidate_count:
                if frame.features is None:
                    raise RuntimeError("holdout frame is missing relational features")
                features = frame.features.unsqueeze(0).to(device)
                padding = torch.zeros((1, frame.candidate_count), dtype=torch.bool, device=device)
                base = frame.base_scores.to(device)
                residual = selector(features, padding)[0]
                scores = refined_person_scores(base, residual)
            else:
                base = torch.empty(0, dtype=torch.float32, device=device)
                residual = torch.empty(0, dtype=torch.float32, device=device)
                scores = torch.empty(0, dtype=torch.float32, device=device)
            scored.append(ScoredHoldoutFrame(
                sample_id=frame.sample_id,
                experiment_id=frame.experiment_id,
                original_indices=frame.original_indices.cpu(),
                boxes=frame.boxes.cpu(),
                world_xy=frame.world_xy.cpu(),
                ignore_flags=frame.ignore_flags.cpu(),
                gt_world_xy=frame.gt_world_xy.cpu(),
                base_scores=base.cpu(),
                residual_logits=residual.cpu(),
                refined_scores=scores.cpu(),
            ))
    if (set(frame.experiment_id for frame in scored) != set(HOLDOUT_EXPERIMENT_IDS)
            or any(count <= 0 for count in eligible_by_episode.values())):
        raise RuntimeError("both fixed holdout episodes must contain eligible people")
    return scored


def _logit(value: float) -> float:
    epsilon = torch.finfo(torch.float64).eps
    clamped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clamped / (1.0 - clamped))


COUNT_FIELDS = ("tp", "fp", "fn", "ignored")


def _rematch_counts(frame: ScoredHoldoutFrame, retained: torch.Tensor) -> dict[str, int]:
    retained = retained.detach().long().cpu()
    if (retained.ndim != 1 or bool((retained < 0).any())
            or bool((retained >= frame.candidate_count).any())
            or (retained.numel() > 1 and not bool((retained[1:] > retained[:-1]).all()))):
        raise RuntimeError("holdout rematch did not preserve original candidate order")
    if retained.numel() > 1:
        original = frame.original_indices.index_select(0, retained)
        if not bool((original[1:] > original[:-1]).all()):
            raise RuntimeError("holdout rematch original indices are not increasing")
    labels, summary = rematch_person_frame({
        "boxes": frame.boxes,
        "world_xy": frame.world_xy,
        "ignore_flags": frame.ignore_flags,
        "gt_world_xy": frame.gt_world_xy,
    }, retained)
    return {
        "tp": int(summary["tp"]),
        "fp": int((labels == 0).sum()),
        "fn": int(summary["fn"]),
        "ignored": int((labels == -1).sum()),
    }


def _add_counts(target: dict[str, int], value: Mapping[str, int], scale: int = 1) -> None:
    for name in COUNT_FIELDS:
        target[name] += scale * int(value[name])


def _metrics(counts: Mapping[str, int], threshold: float) -> dict[str, Any]:
    tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored": int(counts["ignored"]),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def _rematched_metrics_at_threshold(
    frames: list[ScoredHoldoutFrame],
    scores_by_frame: list[torch.Tensor],
    threshold: float,
) -> dict[str, Any]:
    if len(frames) != len(scores_by_frame) or not frames:
        raise ValueError("holdout frames and score vectors are empty or misaligned")
    episodes = {
        name: {field: 0 for field in COUNT_FIELDS}
        for name in sorted({frame.experiment_id for frame in frames})
    }
    for frame, scores in zip(frames, scores_by_frame, strict=True):
        scores = scores.detach().float().cpu()
        if scores.shape != (frame.candidate_count,) or not bool(torch.isfinite(scores).all()):
            raise RuntimeError("holdout rematch score/frame alignment drift")
        retained = torch.where(scores >= float(threshold))[0]
        _add_counts(episodes[frame.experiment_id], _rematch_counts(frame, retained))
    aggregate = {field: sum(value[field] for value in episodes.values()) for field in COUNT_FIELDS}
    return {
        "aggregate": _metrics(aggregate, threshold),
        "episodes": {name: _metrics(value, threshold) for name, value in episodes.items()},
    }


def _passes_all_gates(metrics: Mapping[str, Any]) -> bool:
    episodes = metrics.get("episodes", {})
    reports = [metrics.get("aggregate", {}), *episodes.values()]
    return (set(episodes) == set(HOLDOUT_EXPERIMENT_IDS)
            and all(float(report.get("precision", -1.0)) >= 0.80
                    and float(report.get("recall", -1.0)) >= 0.80 for report in reports))


def _same_rematched_counts(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if set(left.get("episodes", {})) != set(right.get("episodes", {})):
        return False
    scopes = [(left["aggregate"], right["aggregate"])] + [
        (left["episodes"][name], right["episodes"][name]) for name in left["episodes"]
    ]
    return all(all(int(a[field]) == int(b[field]) for field in COUNT_FIELDS) for a, b in scopes)


def _rematched_holdout_frontier(frames: list[ScoredHoldoutFrame]) -> dict[str, Any]:
    """Build the exact tied-score frontier with per-frame canonical rematching."""
    if set(frame.experiment_id for frame in frames) != set(HOLDOUT_EXPERIMENT_IDS):
        raise RuntimeError("exact frontier requires both fixed holdout episodes")
    scores_by_frame = [frame.refined_scores.detach().float().cpu() for frame in frames]
    if not scores_by_frame or sum(scores.numel() for scores in scores_by_frame) == 0:
        raise RuntimeError("holdout frontier contains no candidates")
    flat_scores = torch.cat(scores_by_frame)
    if not bool(torch.isfinite(flat_scores).all()):
        raise FloatingPointError("non-finite holdout score")
    flat_frame = torch.cat([
        torch.full((scores.numel(),), index, dtype=torch.long)
        for index, scores in enumerate(scores_by_frame)
    ])
    flat_position = torch.cat([torch.arange(scores.numel()) for scores in scores_by_frame])
    order = torch.argsort(flat_scores, descending=True, stable=True)
    ordered_scores = flat_scores[order].double()
    ordered_frame = flat_frame[order]
    ordered_position = flat_position[order]
    unique_scores, group_counts = torch.unique_consecutive(ordered_scores, return_counts=True)

    active = [torch.zeros(frame.candidate_count, dtype=torch.bool) for frame in frames]
    frame_counts = [
        {"tp": 0, "fp": 0, "fn": int(frame.gt_world_xy.shape[0]), "ignored": 0}
        for frame in frames
    ]
    episode_counts = {
        name: {field: 0 for field in COUNT_FIELDS} for name in HOLDOUT_EXPERIMENT_IDS
    }
    for frame, counts in zip(frames, frame_counts, strict=True):
        _add_counts(episode_counts[frame.experiment_id], counts)
    maximum_precision_at_recall = {"aggregate": 0.0, **{name: 0.0 for name in HOLDOUT_EXPERIMENT_IDS}}
    maximum_recall_at_precision = {"aggregate": 0.0, **{name: 0.0 for name in HOLDOUT_EXPERIMENT_IDS}}
    jointly_feasible: list[bool] = []

    offset = 0
    for boundary, group_count in zip(unique_scores.tolist(), group_counts.tolist(), strict=True):
        stop = offset + int(group_count)
        affected: set[int] = set()
        for frame_index, position in zip(
            ordered_frame[offset:stop].tolist(), ordered_position[offset:stop].tolist(), strict=True,
        ):
            active[frame_index][position] = True
            affected.add(frame_index)
        for frame_index in sorted(affected):
            frame = frames[frame_index]
            _add_counts(episode_counts[frame.experiment_id], frame_counts[frame_index], scale=-1)
            retained = torch.where(active[frame_index])[0]
            frame_counts[frame_index] = _rematch_counts(frame, retained)
            _add_counts(episode_counts[frame.experiment_id], frame_counts[frame_index])
        aggregate_counts = {
            field: sum(value[field] for value in episode_counts.values()) for field in COUNT_FIELDS
        }
        reports = {
            "aggregate": _metrics(aggregate_counts, boundary),
            **{name: _metrics(episode_counts[name], boundary) for name in HOLDOUT_EXPERIMENT_IDS},
        }
        for name, report in reports.items():
            if float(report["recall"]) >= 0.80:
                maximum_precision_at_recall[name] = max(
                    maximum_precision_at_recall[name], float(report["precision"]),
                )
            if float(report["precision"]) >= 0.80:
                maximum_recall_at_precision[name] = max(
                    maximum_recall_at_precision[name], float(report["recall"]),
                )
        jointly_feasible.append(all(
            float(report["precision"]) >= 0.80 and float(report["recall"]) >= 0.80
            for report in reports.values()
        ))
        offset = stop

    feasible_indices = [index for index, feasible in enumerate(jointly_feasible) if feasible]
    selected_interval: dict[str, float] | None = None
    if feasible_indices:
        start = end = feasible_indices[0]
        for index in feasible_indices[1:]:
            if index != end + 1:
                break
            end = index
        upper = float(unique_scores[start])
        lower_exclusive = float(unique_scores[end + 1]) if end + 1 < unique_scores.numel() else 0.0
        midpoint_logit = 0.5 * (_logit(lower_exclusive) + _logit(upper))
        selected_interval = {
            "lower_score_exclusive": lower_exclusive,
            "upper_score_inclusive": upper,
            "midpoint_logit": midpoint_logit,
            "selected_threshold": 1.0 / (1.0 + math.exp(-midpoint_logit)),
        }
    at_canonical = _rematched_metrics_at_threshold(frames, scores_by_frame, CANONICAL_THRESHOLD)
    return {
        "candidate_scores_computed_once": True,
        "tie_processing": "all_equal_scores_added_before_affected_frames_are_rematched",
        "score_boundaries": int(unique_scores.numel()),
        "aggregate": {
            "at_0_20": at_canonical["aggregate"],
            "maximum_precision_at_recall_gte_0_80": maximum_precision_at_recall["aggregate"],
            "maximum_recall_at_precision_gte_0_80": maximum_recall_at_precision["aggregate"],
        },
        "episodes": {
            name: {
                "at_0_20": at_canonical["episodes"][name],
                "maximum_precision_at_recall_gte_0_80": maximum_precision_at_recall[name],
                "maximum_recall_at_precision_gte_0_80": maximum_recall_at_precision[name],
            }
            for name in HOLDOUT_EXPERIMENT_IDS
        },
        "joint_precision_recall_0_80_exists": bool(feasible_indices),
        "selected_interval": selected_interval,
        "selected_interval_rule": "highest-score contiguous jointly feasible rematched interval",
    }


def _holdout_calibration_result(frames: list[ScoredHoldoutFrame]) -> dict[str, Any]:
    frontier = _rematched_holdout_frontier(frames)
    interval = frontier["selected_interval"]
    attempted_bias: float | None = None
    selected_metrics: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None
    counts_agree = False
    allowed = False
    if interval is not None:
        selected_threshold = float(interval["selected_threshold"])
        raw_scores = [frame.refined_scores for frame in frames]
        selected_metrics = _rematched_metrics_at_threshold(frames, raw_scores, selected_threshold)
        attempted_bias = _logit(CANONICAL_THRESHOLD) - float(interval["midpoint_logit"])
        calibrated_scores = [
            refined_person_scores(frame.base_scores, frame.residual_logits, attempted_bias)
            for frame in frames
        ]
        deployment = _rematched_metrics_at_threshold(
            frames, calibrated_scores, CANONICAL_THRESHOLD,
        )
        counts_agree = _same_rematched_counts(selected_metrics, deployment)
        allowed = (_passes_all_gates(selected_metrics)
                   and _passes_all_gates(deployment)
                   and counts_agree)
    return {
        "before_calibration": frontier,
        "joint_feasible_interval": interval,
        "selected_threshold_metrics": selected_metrics,
        "attempted_calibration_bias": attempted_bias,
        "calibration_bias": attempted_bias if allowed else 0.0,
        "deployment_at_0_20": deployment,
        "selected_deployment_counts_agree": counts_agree,
        "status": "train_feasible" if allowed else "train_infeasible",
        "validation_allowed": allowed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the fixed per-frame person relational selector")
    parser.add_argument("--roi-cache", type=Path)
    parser.add_argument("--consolidation-cache", type=Path)
    parser.add_argument("--consolidation-evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    caches = load_locked_caches(args.roi_cache, args.consolidation_cache, args.consolidation_evidence)
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    selector = PersonRelationalSelector().to(device)
    optimizer = build_selector_optimizer(selector)
    _require_finite_parameters(selector, gradients=False)

    epoch_losses: list[float] = []
    epoch_sampling: list[dict[str, int]] = []
    all_epoch_plans, positives, negatives = _sampling_plans(caches)
    for epoch, plans in enumerate(all_epoch_plans, start=1):
        loss = _train_epoch(selector, optimizer, caches, plans, device)
        epoch_losses.append(loss)
        epoch_sampling.append({"positive": positives, "negative": negatives})
        print(json.dumps({"epoch": epoch, "loss": loss, "positive": positives, "negative": negatives}),
              flush=True)

    episodes = _holdout_outputs(selector, caches, device)
    calibration = _holdout_calibration_result(episodes)
    checkpoint = {
        "schema": "splitfusion_fcos_person_relational_selector_v1",
        "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "roi_manifest_sha256": ROI_MANIFEST_SHA256,
        "consolidation_manifest_sha256": CONSOLIDATION_MANIFEST_SHA256,
        "architecture": ARCHITECTURE,
        "selector": {name: value.detach().cpu() for name, value in selector.state_dict().items()},
        "training": {
            "epochs": EPOCHS,
            "selected_epoch": EPOCHS,
            "batch_frames": BATCH_FRAMES,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "positive_to_negative_loss_sampling": "1:3",
            "all_candidates_retained_in_attention_context": True,
            "ignored_labels_excluded_from_loss": True,
            "sampling_plan_scans": 1,
            "fit_episodes": list(caches.roi_manifest["episode_split"]["fit"]),
            "holdout_episodes": list(HOLDOUT_EXPERIMENT_IDS),
            "seed": SEED,
            "epoch_losses": epoch_losses,
            "epoch_sampling": epoch_sampling,
        },
        "holdout": {
            "threshold_source": "one fixed epoch and one joint two-episode holdout threshold",
            **{name: value for name, value in calibration.items()
               if name not in {"status", "validation_allowed"}},
        },
        "status": calibration["status"],
        "validation_allowed": calibration["validation_allowed"],
        "validation_or_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(json.dumps({
        "selector_checkpoint": str(output),
        "status": checkpoint["status"],
        "validation_allowed": checkpoint["validation_allowed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
