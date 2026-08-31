from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
BASE_EXPERIMENT = ROOT / "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1/20260829_214123"
FROZEN_CHECKPOINT = ROOT / (
    "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/"
    "20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt"
)
FROZEN_CHECKPOINT_SHA256 = "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenRuntime:
    base: Any
    model: torch.nn.Module
    dataset_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str


def load_frozen_runtime(device: torch.device) -> FrozenRuntime:
    """Load only the exact frozen recovered epoch-26 base model."""
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.base_runtime import load_base
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.recovery_model import build_recovery_model

    checkpoint_path = FROZEN_CHECKPOINT.resolve(strict=True)
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("frozen epoch-26 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema") != "splitfusion_fcos_numerical_recovery_atomic_checkpoint_v1"
            or int(checkpoint.get("epoch", -1)) != 26
            or checkpoint.get("validation_accessed") is not False):
        raise RuntimeError("frozen checkpoint provenance drift")
    base = load_base()
    priors = base.common.load_json(BASE_EXPERIMENT / "TRAIN_ONLY_PRIORS.json")
    model, _report = build_recovery_model(priors, float(checkpoint["recovery"]["selected_tau"]), device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model was not completely frozen")
    config = base.common.load_json(base.common.CONFIG_PATH)
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    return FrozenRuntime(base, model, dataset_root, checkpoint_path, checkpoint_hash)


def require_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return device
