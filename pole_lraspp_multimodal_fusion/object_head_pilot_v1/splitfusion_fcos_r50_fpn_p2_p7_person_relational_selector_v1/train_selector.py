from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.verifier import (
    exact_pr_report,
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
    caches: LockedCaches, *, seed: int,
) -> tuple[dict[tuple[str, str], torch.Tensor], int, int]:
    positive_by_frame: list[tuple[tuple[str, str], torch.Tensor]] = []
    negative_by_frame: list[tuple[tuple[str, str], torch.Tensor]] = []
    seen: set[tuple[str, str]] = set()
    for frame in iter_joined_frames(caches):
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

    generator = torch.Generator().manual_seed(seed)
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
    return plans, positive_count, sampled_negative_count


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


def _holdout_outputs(
    selector: PersonRelationalSelector,
    caches: LockedCaches,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    outputs = {
        experiment_id: {
            "base_scores": [], "residual_logits": [], "labels": [], "eligible_positive": 0,
        }
        for experiment_id in HOLDOUT_EXPERIMENT_IDS
    }
    selector.eval()
    with torch.inference_mode():
        for frame in iter_joined_frames(caches):
            if frame.partition != 1:
                continue
            if frame.experiment_id not in outputs:
                raise RuntimeError("unexpected holdout episode")
            episode = outputs[frame.experiment_id]
            episode["eligible_positive"] += frame.eligible_positive_count
            if frame.candidate_count == 0:
                continue
            features, base_scores, labels, padding = pad_frames([frame], device)
            residual = selector(features, padding)[0, :frame.candidate_count]
            valid = labels[0, :frame.candidate_count] >= 0
            episode["base_scores"].append(base_scores[0, :frame.candidate_count][valid].cpu())
            episode["residual_logits"].append(residual[valid].cpu())
            episode["labels"].append(labels[0, :frame.candidate_count][valid].cpu())
    finalized: dict[str, dict[str, Any]] = {}
    for experiment_id, episode in outputs.items():
        if not episode["base_scores"] or int(episode["eligible_positive"]) <= 0:
            raise RuntimeError(f"holdout episode has no scored candidates or eligible people: {experiment_id}")
        finalized[experiment_id] = {
            "base_scores": torch.cat(episode["base_scores"]),
            "residual_logits": torch.cat(episode["residual_logits"]),
            "labels": torch.cat(episode["labels"]),
            "eligible_positive": int(episode["eligible_positive"]),
        }
    return finalized


def _logit(value: float) -> float:
    epsilon = torch.finfo(torch.float64).eps
    clamped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clamped / (1.0 - clamped))


def _joint_feasible_interval(
    scores_by_episode: Mapping[str, torch.Tensor],
    labels_by_episode: Mapping[str, torch.Tensor],
    eligible_by_episode: Mapping[str, int],
) -> dict[str, float] | None:
    episode_names = tuple(sorted(scores_by_episode))
    scores = torch.cat([scores_by_episode[name].detach().float().cpu() for name in episode_names])
    labels = torch.cat([labels_by_episode[name].detach().long().cpu() for name in episode_names])
    episode_index = torch.cat([
        torch.full((scores_by_episode[name].numel(),), index, dtype=torch.long)
        for index, name in enumerate(episode_names)
    ])
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_scores = scores[order].double()
    ordered_labels = labels[order]
    ordered_episode = episode_index[order]
    unique_scores, group_counts = torch.unique_consecutive(ordered_scores, return_counts=True)
    group_ends = group_counts.cumsum(0) - 1

    predicted = torch.arange(1, scores.numel() + 1, dtype=torch.long)[group_ends]
    tp = ordered_labels.eq(1).long().cumsum(0)[group_ends]
    eligible_total = sum(int(eligible_by_episode[name]) for name in episode_names)
    feasible = (tp.double() / predicted.double() >= 0.80) & (tp.double() / eligible_total >= 0.80)
    for episode_number, name in enumerate(episode_names):
        member = ordered_episode.eq(episode_number)
        episode_predicted = member.long().cumsum(0)[group_ends]
        episode_tp = (member & ordered_labels.eq(1)).long().cumsum(0)[group_ends]
        precision = episode_tp.double() / episode_predicted.clamp_min(1).double()
        recall = episode_tp.double() / int(eligible_by_episode[name])
        feasible &= (precision >= 0.80) & (recall >= 0.80)
    indices = torch.where(feasible)[0].tolist()
    if not indices:
        return None
    start = end = indices[0]
    for index in indices[1:]:
        if index != end + 1:
            break
        end = index
    upper = float(unique_scores[start])
    lower_exclusive = float(unique_scores[end + 1]) if end + 1 < unique_scores.numel() else 0.0
    midpoint_logit = 0.5 * (_logit(lower_exclusive) + _logit(upper))
    return {
        "lower_score_exclusive": lower_exclusive,
        "upper_score_inclusive": upper,
        "midpoint_logit": midpoint_logit,
        "selected_threshold": 1.0 / (1.0 + math.exp(-midpoint_logit)),
    }


def _metric_at_0_20(scores: torch.Tensor, labels: torch.Tensor, eligible: int) -> dict[str, Any]:
    return exact_pr_report(
        scores, labels, eligible_positive_count=eligible, canonical_threshold=CANONICAL_THRESHOLD,
    )["at_0_20"]


def _holdout_calibration_result(episodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    raw_scores = {
        name: refined_person_scores(value["base_scores"], value["residual_logits"])
        for name, value in episodes.items()
    }
    labels = {name: value["labels"] for name, value in episodes.items()}
    eligible = {name: int(value["eligible_positive"]) for name, value in episodes.items()}
    aggregate_scores = torch.cat([raw_scores[name] for name in sorted(raw_scores)])
    aggregate_labels = torch.cat([labels[name] for name in sorted(labels)])
    aggregate_eligible = sum(eligible.values())
    before = {
        "aggregate": exact_pr_report(
            aggregate_scores, aggregate_labels, eligible_positive_count=aggregate_eligible,
            canonical_threshold=CANONICAL_THRESHOLD,
        ),
        "episodes": {
            name: exact_pr_report(
                raw_scores[name], labels[name], eligible_positive_count=eligible[name],
                canonical_threshold=CANONICAL_THRESHOLD,
            )
            for name in sorted(raw_scores)
        },
    }
    interval = _joint_feasible_interval(raw_scores, labels, eligible)
    attempted_bias: float | None = None
    deployment: dict[str, Any] | None = None
    allowed = False
    if interval is not None:
        attempted_bias = _logit(CANONICAL_THRESHOLD) - float(interval["midpoint_logit"])
        calibrated = {
            name: refined_person_scores(
                episodes[name]["base_scores"], episodes[name]["residual_logits"], attempted_bias,
            )
            for name in episodes
        }
        deployment = {
            "aggregate": _metric_at_0_20(
                torch.cat([calibrated[name] for name in sorted(calibrated)]),
                aggregate_labels,
                aggregate_eligible,
            ),
            "episodes": {
                name: _metric_at_0_20(calibrated[name], labels[name], eligible[name])
                for name in sorted(calibrated)
            },
        }
        reports = [deployment["aggregate"], *deployment["episodes"].values()]
        allowed = all(
            float(report["precision"]) >= 0.80 and float(report["recall"]) >= 0.80
            for report in reports
        )
    return {
        "before_calibration": before,
        "joint_feasible_interval": interval,
        "attempted_calibration_bias": attempted_bias,
        "calibration_bias": attempted_bias if allowed else 0.0,
        "deployment_at_0_20": deployment,
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
    for epoch in range(1, EPOCHS + 1):
        plans, positives, negatives = _sampling_plans(caches, seed=SEED + epoch)
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
