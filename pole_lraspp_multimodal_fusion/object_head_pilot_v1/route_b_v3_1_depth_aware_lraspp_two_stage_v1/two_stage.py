from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch

from common import load_json, sha256, tensor_state_hash

REPRESENTATION_PREFIXES = ("backbone.", "depth_neck.", "segmentation.", "dense_depth.")
OBJECT_PREFIXES = ("vehicle.", "person.")


def is_representation(name: str) -> bool:
    return name.startswith(REPRESENTATION_PREFIXES)


def is_object(name: str) -> bool:
    return name.startswith(OBJECT_PREFIXES)


def parameter_allowlist(model: torch.nn.Module, stage: str) -> list[str]:
    predicate = is_representation if stage == "stage1" else is_object
    result = [name for name, _ in model.named_parameters() if predicate(name)]
    if len(result) != len(set(result)) or not result:
        raise RuntimeError(f"invalid {stage} parameter allowlist")
    return result


def named_state(model: torch.nn.Module, predicate) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().cpu().clone())
                       for name, value in model.state_dict().items() if predicate(name))


def representation_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return named_state(model, is_representation)


def object_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return named_state(model, is_object)


def state_hash(model: torch.nn.Module, predicate) -> str:
    return tensor_state_hash(named_state(model, predicate))


def assert_allowlist(model: torch.nn.Module, stage: str, registered: list[str]) -> None:
    actual = [name for name, value in model.named_parameters() if value.requires_grad]
    expected = parameter_allowlist(model, stage)
    if actual != expected or actual != registered:
        raise RuntimeError(f"{stage} trainable parameter allowlist drift")


def build_optimizer(model: torch.nn.Module, stage: str) -> torch.optim.Optimizer:
    expected = set(parameter_allowlist(model, stage))
    groups: OrderedDict[str, list[torch.nn.Parameter]] = OrderedDict()
    if stage == "stage1":
        names = ("new_decay", "new_no_decay", "pretrained_decay", "pretrained_no_decay")
    else:
        names = ("object_decay", "object_no_decay")
    for name in names:
        groups[name] = []
    seen = set()
    for name, parameter in model.named_parameters():
        if name not in expected:
            continue
        no_decay = parameter.ndim == 1 or name.endswith(".bias")
        if stage == "stage1":
            # Only the radar half of the stem is new. The RGB half and all other
            # MobileNet tensors came from the official ImageNet state.
            pretrained = name.startswith("backbone.") and not name.startswith("backbone.0.radar_conv.")
            key = ("pretrained" if pretrained else "new") + ("_no_decay" if no_decay else "_decay")
        else:
            key = "object" + ("_no_decay" if no_decay else "_decay")
        groups[key].append(parameter); seen.add(name)
    if seen != expected:
        raise RuntimeError(f"optimizer membership drift for {stage}")
    specs = [{"params": values, "name": name, "lr": 0.0,
              "weight_decay": 0.0 if name.endswith("no_decay") else 1e-4}
             for name, values in groups.items() if values]
    return torch.optim.AdamW(specs, betas=(0.9, 0.999), eps=1e-8)


def scheduled_lr(stage: str, epoch: int, update: int, updates: int) -> tuple[float, float]:
    peak = 3e-4
    if epoch == 1:
        fraction = max(0.0, min(1.0, (update - 1) / max(1, updates - 1)))
        new = peak * (0.1 + 0.9 * fraction)
    else:
        final_epoch = 20 if stage == "stage1" else 30
        total = (final_epoch - 1) * updates
        index = (epoch - 2) * updates + (update - 1)
        fraction = max(0.0, min(1.0, index / max(1, total - 1)))
        new = peak * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * fraction)))
    return new, new * 0.1


def set_lrs(optimizer: torch.optim.Optimizer, stage: str, new: float, pretrained: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = pretrained if stage == "stage1" and str(group["name"]).startswith("pretrained") else new


def checkpoint_valid(path: Path, expected_config_hash: str | None = None) -> bool:
    sidecar = path.with_suffix(".json")
    try:
        record = load_json(sidecar)
        valid = (record["complete"] is True and int(record["bytes"]) == path.stat().st_size
                 and record["sha256"] == sha256(path))
        if expected_config_hash is not None:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            valid = valid and payload["resolved_config_sha256"] == expected_config_hash
        return bool(valid)
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def latest_checkpoint(directory: Path, prefix: str, expected_config_hash: str) -> Path | None:
    for candidate in sorted(directory.glob(f"{prefix}_*.pt"), reverse=True):
        if checkpoint_valid(candidate, expected_config_hash):
            return candidate
    return None


def model_finite(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all().item() for value in model.state_dict().values()
               if value.is_floating_point() or value.is_complex())


def optimizer_finite(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                if not torch.isfinite(value).all().item():
                    return False
    return True


def parameter_counts(model: torch.nn.Module, stage: str) -> dict[str, int]:
    allow = set(parameter_allowlist(model, stage))
    trainable = sum(value.numel() for name, value in model.named_parameters() if name in allow)
    total = sum(value.numel() for value in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total,
            "trainable_tensors": len(allow), "frozen_tensors": len(list(model.named_parameters())) - len(allow)}
