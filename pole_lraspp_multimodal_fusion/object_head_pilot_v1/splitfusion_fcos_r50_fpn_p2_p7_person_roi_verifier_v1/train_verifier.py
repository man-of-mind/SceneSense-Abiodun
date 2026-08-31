from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .runtime import FROZEN_CHECKPOINT_SHA256, require_device
from .verifier import (
    FEATURE_DIM,
    HOLDOUT_EXPERIMENT_IDS,
    ROI_DESCRIPTOR_DIM,
    SCALAR_FEATURE_NAMES,
    VERIFIER_ARCHITECTURE,
    PersonVerifier,
    build_verifier_optimizer,
    calibration_bias_for_interval,
    exact_pr_report,
    fp16_round_trip_roi_descriptors,
    partition_experiment_ids,
    refined_person_logits,
    refined_person_scores,
)

EPOCHS = 5
BATCH_SIZE = 1024
SEED = 20260830
NEGATIVES_PER_POSITIVE = 3
CANONICAL_THRESHOLD = 0.20


def _require_finite_parameters(head: PersonVerifier, *, gradients: bool) -> None:
    for name, parameter in head.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None or not bool(torch.isfinite(value).all()):
            kind = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"non-finite verifier {kind}: {name}")


def _load_shard(cache: Path, shard: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(cache / str(shard["path"]), map_location="cpu", weights_only=True)
    count = payload["labels"].numel()
    if (payload["roi_descriptors"].shape != (count, ROI_DESCRIPTOR_DIM)
            or payload["scalar_features"].shape != (count, len(SCALAR_FEATURE_NAMES))
            or payload["base_scores"].shape != (count,)
            or payload["partitions"].shape != (count,)
            or len(payload["experiment_ids"]) != count):
        raise RuntimeError(f"person ROI cache shard shape drift: {shard['path']}")
    return payload


def _epoch_sampling_plan(
    cache: Path, shards: list[dict[str, Any]], generator: torch.Generator,
) -> tuple[list[torch.Tensor], int, int]:
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    for shard in shards:
        payload = _load_shard(cache, shard)
        labels = payload["labels"].long()
        fit = payload["partitions"].long() == 0
        positives.append(torch.where(fit & (labels == 1))[0])
        negatives.append(torch.where(fit & (labels == 0))[0])
    positive_count = sum(value.numel() for value in positives)
    negative_count = sum(value.numel() for value in negatives)
    sampled_negative_count = NEGATIVES_PER_POSITIVE * positive_count
    if positive_count == 0 or negative_count < sampled_negative_count:
        raise RuntimeError("fit cache cannot support the fixed 1:3 positive-to-negative sampling ratio")

    selected_global = torch.randperm(negative_count, generator=generator)[:sampled_negative_count]
    plans: list[torch.Tensor] = []
    offset = 0
    for positive, negative in zip(positives, negatives):
        in_shard = selected_global[(selected_global >= offset) & (selected_global < offset + negative.numel())] - offset
        sampled_negative = negative.index_select(0, in_shard)
        selected = torch.cat((positive, sampled_negative))
        plans.append(selected[torch.randperm(selected.numel(), generator=generator)])
        offset += negative.numel()
    if sum(plan.numel() for plan in plans) != positive_count + sampled_negative_count:
        raise RuntimeError("deterministic 1:3 sampling plan did not reconcile")
    return plans, positive_count, sampled_negative_count


def _holdout_outputs(
    head: PersonVerifier, cache: Path, shards: list[dict[str, Any]], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_score_parts: list[torch.Tensor] = []
    delta_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    head.eval()
    with torch.inference_mode():
        for shard in shards:
            payload = _load_shard(cache, shard)
            labels = payload["labels"].long()
            selected = torch.where((payload["partitions"].long() == 1) & (labels >= 0))[0]
            if selected.numel() == 0:
                continue
            descriptors = fp16_round_trip_roi_descriptors(
                payload["roi_descriptors"].index_select(0, selected),
            ).to(device)
            scalars = payload["scalar_features"].index_select(0, selected).float().to(device)
            base_scores = payload["base_scores"].index_select(0, selected).float().to(device)
            features = torch.cat((descriptors, scalars), dim=1)
            delta = head(features)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError("non-finite holdout verifier delta")
            base_score_parts.append(base_scores.cpu())
            delta_parts.append(delta.cpu())
            label_parts.append(labels.index_select(0, selected))
    if not base_score_parts:
        raise RuntimeError("holdout cache contains no non-ignored person candidates")
    return torch.cat(base_score_parts), torch.cat(delta_parts), torch.cat(label_parts)


def _deployment_calibration_result(
    base_scores: torch.Tensor,
    verifier_delta: torch.Tensor,
    labels: torch.Tensor,
    eligible_positive_count: int,
    interval: dict[str, float],
) -> dict[str, Any]:
    attempted_bias = calibration_bias_for_interval(interval, CANONICAL_THRESHOLD)
    calibrated_scores = refined_person_scores(base_scores, verifier_delta, attempted_bias)
    report = exact_pr_report(
        calibrated_scores,
        labels,
        eligible_positive_count=eligible_positive_count,
        canonical_threshold=CANONICAL_THRESHOLD,
    )
    canonical = report["at_0_20"]
    allowed = bool(canonical["precision"] >= 0.80 and canonical["recall"] >= 0.80)
    return {
        "attempted_calibration_bias": attempted_bias,
        "calibration_bias": attempted_bias if allowed else 0.0,
        "after_calibration": report,
        "status": "train_feasible" if allowed else "train_infeasible",
        "validation_allowed": allowed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the fixed person ROI verifier on fit episodes only")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cache = args.cache.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    episode_split = manifest.get("episode_split", {})
    fit_ids, holdout_ids = partition_experiment_ids(
        [*episode_split.get("fit", []), *episode_split.get("holdout", [])],
    )
    if (manifest.get("schema") != "splitfusion_fcos_person_roi_cache_v1"
            or manifest.get("split") != "train"
            or int(manifest.get("pass_count", -1)) != 1
            or int(manifest.get("feature_dim", -1)) != FEATURE_DIM
            or int(manifest.get("roi_descriptor_dim", -1)) != ROI_DESCRIPTOR_DIM
            or episode_split != {"fit": list(fit_ids), "holdout": list(holdout_ids)}
            or set(holdout_ids) != set(HOLDOUT_EXPERIMENT_IDS)
            or manifest.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or manifest.get("validation_or_test_accessed") is not False):
        raise RuntimeError("person ROI cache contract drift")

    torch.manual_seed(SEED)
    device = require_device(args.device)
    head = PersonVerifier().to(device)
    optimizer = build_verifier_optimizer(head)
    head_ids = {id(parameter) for parameter in head.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != head_ids:
        raise RuntimeError("optimizer contains a non-verifier parameter")
    _require_finite_parameters(head, gradients=False)

    epoch_losses: list[float] = []
    epoch_sampling: list[dict[str, int]] = []
    shards = list(manifest["shards"])
    holdout_eligible_person_gt = int(manifest["partition_counts"]["holdout"]["eligible_person_gt"])
    if holdout_eligible_person_gt <= 0:
        raise RuntimeError("holdout cache contains no eligible person GT")
    for epoch in range(1, EPOCHS + 1):
        generator = torch.Generator().manual_seed(SEED + epoch)
        plans, positive_count, negative_count = _epoch_sampling_plan(cache, shards, generator)
        shard_order = torch.randperm(len(shards), generator=generator).tolist()
        loss_sum, example_count = 0.0, 0
        head.train()
        for shard_index in shard_order:
            selected = plans[shard_index]
            if selected.numel() == 0:
                continue
            payload = _load_shard(cache, shards[shard_index])
            for start in range(0, selected.numel(), BATCH_SIZE):
                indices = selected[start:start + BATCH_SIZE]
                descriptors = fp16_round_trip_roi_descriptors(
                    payload["roi_descriptors"].index_select(0, indices),
                ).to(device)
                scalars = payload["scalar_features"].index_select(0, indices).float().to(device)
                base_scores = payload["base_scores"].index_select(0, indices).float().to(device)
                labels = payload["labels"].index_select(0, indices).float().to(device)
                features = torch.cat((descriptors, scalars), dim=1)
                optimizer.zero_grad(set_to_none=True)
                logits = refined_person_logits(base_scores, head(features))
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite verifier BCE loss")
                loss.backward()
                _require_finite_parameters(head, gradients=True)
                optimizer.step()
                _require_finite_parameters(head, gradients=False)
                batch_count = indices.numel()
                loss_sum += float(loss.detach()) * batch_count
                example_count += batch_count
        expected_examples = positive_count + negative_count
        if example_count != expected_examples:
            raise RuntimeError("verifier epoch sampling count drift")
        epoch_loss = loss_sum / example_count
        epoch_losses.append(epoch_loss)
        epoch_sampling.append({"positive": positive_count, "negative": negative_count})
        print(json.dumps({"epoch": epoch, "epochs": EPOCHS, "bce": epoch_loss,
                          "positive": positive_count, "negative": negative_count}), flush=True)

    holdout_base_scores, holdout_delta, holdout_labels = _holdout_outputs(head, cache, shards, device)
    holdout_base_scores = holdout_base_scores.to(device)
    holdout_delta = holdout_delta.to(device)
    holdout_scores = refined_person_scores(holdout_base_scores, holdout_delta)
    holdout_before = exact_pr_report(
        holdout_scores,
        holdout_labels,
        eligible_positive_count=holdout_eligible_person_gt,
        canonical_threshold=CANONICAL_THRESHOLD,
    )
    if holdout_before["joint_precision_recall_0_80_exists"]:
        selected_interval = holdout_before["selected_interval"]
        if selected_interval is None:
            raise RuntimeError("joint-feasible holdout has no threshold interval")
        calibration = _deployment_calibration_result(
            holdout_base_scores,
            holdout_delta,
            holdout_labels,
            holdout_eligible_person_gt,
            selected_interval,
        )
        attempted_calibration_bias = calibration["attempted_calibration_bias"]
        calibration_bias = calibration["calibration_bias"]
        holdout_after = calibration["after_calibration"]
        status = calibration["status"]
        validation_allowed = calibration["validation_allowed"]
    else:
        attempted_calibration_bias = None
        calibration_bias = 0.0
        holdout_after = None
        status, validation_allowed = "train_infeasible", False

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "splitfusion_fcos_person_roi_verifier_v1",
        "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "verifier": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "architecture": VERIFIER_ARCHITECTURE,
        "training": {
            "fit_episodes": list(fit_ids),
            "holdout_episodes": list(holdout_ids),
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": BATCH_SIZE,
            "loss": "binary_cross_entropy",
            "positive_to_negative_sampling": "1:3",
            "ignored_labels_excluded": True,
            "seed": SEED,
            "epoch_losses": epoch_losses,
            "epoch_sampling": epoch_sampling,
        },
        "holdout": {
            "threshold_source": "two untouched training holdout episodes",
            "before_calibration": holdout_before,
            "attempted_calibration_bias": attempted_calibration_bias,
            "calibration_bias": calibration_bias,
            "after_calibration": holdout_after,
        },
        "status": status,
        "validation_allowed": validation_allowed,
        "validation_or_test_accessed": False,
        "canonical_evaluator_threshold": CANONICAL_THRESHOLD,
        "cache_manifest": str(cache / "cache_manifest.json"),
    }
    torch.save(checkpoint, output)
    print(json.dumps({
        "verifier_checkpoint": str(output),
        "status": status,
        "validation_allowed": validation_allowed,
        "calibration_bias": calibration_bias,
        "holdout": holdout_before,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
