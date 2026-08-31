from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .quality import FEATURE_DIM, QualityMLP, build_quality_optimizer, refined_logits, sigmoid_focal_loss
from .runtime import FROZEN_CHECKPOINT_SHA256, require_device

EPOCHS = 5
BATCH_SIZE = 2048
SEED = 20260830
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
METRIC_THRESHOLD = 0.20


def _require_finite_quality_parameters(head: QualityMLP, *, gradients: bool) -> None:
    for name, parameter in head.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None or not bool(torch.isfinite(value).all()):
            kind = "gradient" if gradients else "parameter"
            raise FloatingPointError(f"non-finite quality-head {kind}: {name}")


def _train_cache_metrics(
    head: QualityMLP, cache: Path, shards: list[dict[str, object]], device: torch.device,
) -> dict[str, dict[str, float | int]]:
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("vehicle", "person")}
    head.eval()
    with torch.inference_mode():
        for shard in shards:
            payload = torch.load(cache / str(shard["path"]), map_location="cpu", weights_only=True)
            labels = payload["labels"].long()
            selected = torch.where(labels >= 0)[0]
            features = payload["features"][selected].float().to(device)
            base_scores = payload["base_scores"][selected].float().to(device)
            classes = payload["classes"][selected].long().to(device)
            labels = labels[selected].to(device)
            logits = refined_logits(base_scores, head(features))
            scores = torch.sigmoid(logits)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("non-finite refined train-cache metric output")
            predicted = scores >= METRIC_THRESHOLD
            for class_index, class_name in enumerate(("vehicle", "person")):
                class_mask = classes == class_index
                positive = labels == 1
                totals[class_name]["tp"] += int((class_mask & predicted & positive).sum())
                totals[class_name]["fp"] += int((class_mask & predicted & ~positive).sum())
                totals[class_name]["fn"] += int((class_mask & ~predicted & positive).sum())
    metrics: dict[str, dict[str, float | int]] = {}
    for class_name, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        metrics[class_name] = {
            **counts,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
        }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train only the quality MLP from a frozen train cache")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cache = args.cache.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "splitfusion_fcos_candidate_cache_v1"
            or manifest.get("split") != "train"
            or int(manifest.get("pass_count", -1)) != 1
            or int(manifest.get("feature_dim", -1)) != FEATURE_DIM
            or manifest.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or manifest.get("validation_or_test_accessed") is not False):
        raise RuntimeError("candidate cache contract drift")

    torch.manual_seed(SEED)
    device = require_device(args.device)
    head = QualityMLP(normalize=False).to(device)
    optimizer = build_quality_optimizer(head)
    head_parameter_ids = {id(parameter) for parameter in head.parameters()}
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_parameter_ids != head_parameter_ids:
        raise RuntimeError("optimizer contains a non-quality-head parameter")
    _require_finite_quality_parameters(head, gradients=False)

    epoch_losses: list[float] = []
    shards = list(manifest["shards"])
    for epoch in range(1, EPOCHS + 1):
        generator = torch.Generator().manual_seed(SEED + epoch)
        shard_order = torch.randperm(len(shards), generator=generator).tolist()
        loss_sum, example_count = 0.0, 0
        head.train()
        for shard_index in shard_order:
            payload = torch.load(cache / shards[shard_index]["path"], map_location="cpu", weights_only=True)
            features = payload["features"]
            base_scores = payload["base_scores"]
            labels = payload["labels"].long()
            selected = torch.where(labels >= 0)[0]
            selected = selected[torch.randperm(selected.numel(), generator=generator)]
            for start in range(0, selected.numel(), BATCH_SIZE):
                indices = selected[start:start + BATCH_SIZE]
                batch_features = features[indices].float().to(device)
                batch_base_scores = base_scores[indices].float().to(device)
                batch_labels = labels[indices].float().to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = refined_logits(batch_base_scores, head(batch_features))
                loss = sigmoid_focal_loss(logits, batch_labels, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite quality-head training loss")
                loss.backward()
                _require_finite_quality_parameters(head, gradients=True)
                optimizer.step()
                _require_finite_quality_parameters(head, gradients=False)
                batch_count = indices.numel()
                loss_sum += float(loss.detach()) * batch_count
                example_count += batch_count
        if example_count == 0:
            raise RuntimeError("candidate cache contains no labeled candidates")
        epoch_loss = loss_sum / example_count
        epoch_losses.append(epoch_loss)
        print(json.dumps({"epoch": epoch, "epochs": EPOCHS, "focal_loss": epoch_loss}), flush=True)

    train_cache_metrics = _train_cache_metrics(head, cache, shards, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "splitfusion_fcos_candidate_quality_head_v1",
        "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "quality_head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "architecture": {"normalize": False, "input": FEATURE_DIM, "hidden": 64, "output": 1},
        "training": {
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "batch_size": BATCH_SIZE,
            "focal_alpha": FOCAL_ALPHA,
            "focal_gamma": FOCAL_GAMMA,
            "seed": SEED,
            "epoch_losses": epoch_losses,
            "train_cache_metrics": {
                "threshold": METRIC_THRESHOLD,
                "ignored_labels_excluded": True,
                "classes": train_cache_metrics,
            },
        },
        "cache_manifest": str(cache / "cache_manifest.json"),
    }
    torch.save(checkpoint, output)
    print(json.dumps({"quality_checkpoint": str(output), "epochs": EPOCHS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
