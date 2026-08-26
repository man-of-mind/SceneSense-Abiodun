#!/usr/bin/env python3
"""Phase B launch check for the LR-ASPP/CenterFusion hybrid.

Runs exactly one real q=0 AMP forward/backward step on a real Route B training
batch and requires finite, non-zero gradient in every branch that has to learn:

    rgb_backbone, radar_encoder, vehicle_heatmap, person_heatmap,
    regression_refinement, segmentation

It also records the output schema (14 channels at input resolution), the
parameter counts, the split-boundary shapes and the peak VRAM of the step, and
verifies the refinement branch is genuinely wired (a non-zero residual).

Nothing is trained and nothing is written into the checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hybrid_model_v1 import install  # noqa: E402

from pole_lraspp_multimodal_fusion.common import load_config, read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from pole_lraspp_multimodal_fusion.train_fusion import (  # noqa: E402
    FusionPoleMultiTaskDataset,
    _deep_merge_dicts,
    _freeze_batch_norm,
    _move_object_targets,
    compute_losses,
    split_rows,
)

# name-prefix -> required branch. Checked against parameters that require grad.
BRANCH_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "rgb_backbone": ("backbone.",),
    "radar_encoder": ("radar_encoder.",),
    "vehicle_heatmap": ("object_head.vehicle_heatmap_head.", "refine_vehicle_heatmap_head."),
    "person_heatmap": ("object_head.person_heatmap_head.", "refine_person_heatmap_head."),
    "regression_refinement": (
        "object_head.shared_trunk.", "object_head.regression_head.",
        "lat16.", "lat8.", "lat4.", "norm16.", "norm8.", "norm4.",
        "reduce8.", "smooth4.", "refine_trunk.", "refine_regression_head.",
    ),
    "segmentation": ("classifier.",),
}


def _branch_of(name: str) -> str:
    for branch, prefixes in BRANCH_PREFIXES.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return branch
    return "unassigned"


def _install_target_cap(trial: dict) -> str:
    """Match the baseline's object targets exactly.

    The fixed baseline (curriculum_stage2_joint_v1 epoch 13) was trained with the
    vehicle-only adaptive-radius cap installed. Installing the same cap here keeps
    the target tensors identical, so any delta is attributable to the architecture
    and not to a different supervision signal.
    """
    from object_head_pilot_v1.target_variants_v1 import assert_control_parity, install as install_cap
    import numpy as np

    rng = np.random.default_rng(20260824)
    objects = []
    for _ in range(24):
        width = float(rng.uniform(8.0, 400.0))
        height = float(rng.uniform(8.0, 300.0))
        objects.append({
            "class_index": int(rng.integers(0, 2)),
            "center_x": float(rng.uniform(0.0, 1280.0)),
            "center_y": float(rng.uniform(0.0, 720.0)),
            "bbox_w": width, "bbox_h": height, "area": width * height,
            "local_x": 1.0, "local_y": 2.0, "local_z": 0.3,
            "size_x": 4.0, "size_y": 2.0, "size_z": 1.5,
            "yaw_sin": 0.1, "yaw_cos": 0.9, "parked": 0.0, "radar_support": 1.0,
            "world_x": 10.0, "world_y": 5.0, "world_z": 0.2,
        })
    assert_control_parity({
        "objects": objects, "original_size": (1280, 720), "input_size": (768, 432),
        "heatmap_radius_px": 4, "max_objects": 64, "predict_bbox2d": True,
        "adaptive_heatmap_radius": True,
    })
    return install_cap(trial.get("object_heads", {}).get("vehicle_heatmap_radius_cap_px"))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = the trial's batch size")
    args = parser.parse_args(argv)

    install()
    from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp

    config = load_config(args.config)
    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    train_cfg = config["training"]
    fusion_cfg = config.get("fusion", {})
    object_cfg = _deep_merge_dicts(dict(config.get("object_heads", {})), trial.get("object_heads"))
    loss_weights = _deep_merge_dicts(dict(train_cfg.get("loss_weights", {})), trial.get("loss_weights"))

    input_width, input_height = [int(v) for v in trial["input_size"]]
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(fusion_cfg.get("radar_channels", 4))
    object_classes = tuple(object_cfg.get("object_classes", ("vehicle", "person")))
    predict_bbox2d = bool(object_cfg.get("predict_bbox2d", True))
    object_channels = len(object_classes) + (12 if predict_bbox2d else 10)

    dataset_dir = Path(args.experiment_dir).resolve() / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    splits = split_rows(rows)
    if splits["test"]:
        raise RuntimeError("locked Route B test split is present in this view; refusing to run")
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    target_arm = _install_target_cap(trial)
    batch_size = int(args.batch_size or trial.get("batch_size", 16))
    train_ds = FusionPoleMultiTaskDataset(
        dataset_dir, splits["train"], object_rows, (input_width, input_height), object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
    )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    tensors, masks, object_targets = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the launch check requires CUDA (training runs on the sm_120 GPU)")
    model = build_multitask_fusion_lraspp(
        num_classes=num_classes,
        radar_channels=radar_channels,
        pretrained=False,
        init_checkpoint=str(trial.get("init_rgb_checkpoint", "")),
        object_channels=object_channels,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(object_cfg.get("fuse_low_feature", True)),
        head_arch=str(object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(object_cfg.get("head_depth", 3)),
        predict_bbox2d=predict_bbox2d,
        use_groundplane_prior=bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params={},
        device=device,
    ).to(device)
    warm_start = getattr(model, "warm_start_report", None)
    if bool(trial.get("freeze_bn", False)):
        _freeze_batch_norm(model)

    tensors = tensors.to(device)
    masks = masks.to(device)
    object_targets = _move_object_targets(object_targets, device)

    # Split-boundary shapes + a direct check that the refinement is wired.
    model.eval()
    with torch.no_grad():
        fused = model.encode_front(tensors[:, :3], tensors[:, 3: 3 + radar_channels])
        tail = model.decode_tail(fused, (int(tensors.shape[-2]), int(tensors.shape[-1])))
        coarse = model.object_head(torch.cat(
            [fused["low"], model._match(fused["high"], fused["low"])], dim=1))
        coarse_full = F.interpolate(coarse, size=tail["object"].shape[-2:],
                                    mode="bilinear", align_corners=False)
        refine_residual = (tail["object"] - coarse_full).abs().max().item()
    split_shapes = {key: list(value.shape) for key, value in fused.items()}

    model.train()
    if bool(trial.get("freeze_bn", False)):
        _freeze_batch_norm(model)
    amp_enabled = bool(train_cfg.get("amp", True))
    torch.cuda.reset_peak_memory_stats(device)

    # One real AMP forward/backward. torch.amp.GradScaler starts at 65536 and is
    # *designed* to overflow on the first steps of any run, skip them and halve
    # the scale; the production loop does exactly that. Reproducing that settling
    # here (instead of testing at the arbitrary initial scale) is what makes the
    # finite-gradient assertion below a statement about the model rather than
    # about the scaler's warm-up.
    scale = 65536.0 if amp_enabled else 1.0
    scale_reductions = 0
    while True:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled, cache_enabled=False):
            loss, parts, _ = compute_losses(
                model, tensors, masks, object_targets, num_classes, loss_weights,
                class_weights=torch.tensor([float(w) for w in trial["class_loss_weights"]],
                                           dtype=torch.float32, device=device),
                lovasz_weight=float(trial.get("lovasz_weight", 0.0)),
                feature_drop_fraction=0.0,   # q = 0, the clean path
            )
        (loss * scale).backward()
        overflowed = any(
            not bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        if not overflowed or scale_reductions >= 24:
            break
        scale /= 2.0
        scale_reductions += 1
    # Gradients still carry the AMP loss scale; undo it before any magnitude test
    # so "non-zero" means non-zero in real units.
    inv_scale = 1.0 / float(scale)

    branches: Dict[str, Dict[str, Any]] = {
        name: {"tensors": 0, "grad_sq": 0.0, "nonfinite_tensors": 0, "zero_grad_tensors": [],
               "missing_grad_tensors": []}
        for name in list(BRANCH_PREFIXES) + ["unassigned"]
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        record = branches[_branch_of(name)]
        record["tensors"] += 1
        if param.grad is None:
            record["missing_grad_tensors"].append(name)
            continue
        grad = param.grad.detach().float() * inv_scale
        if not bool(torch.isfinite(grad).all().item()):
            record["nonfinite_tensors"] += 1
            continue
        norm = float(grad.norm().item())
        record["grad_sq"] += norm * norm
        if norm == 0.0:
            record["zero_grad_tensors"].append(name)

    failures: List[str] = []
    for name in BRANCH_PREFIXES:
        record = branches[name]
        record["grad_norm"] = math.sqrt(record.pop("grad_sq"))
        if record["tensors"] == 0:
            failures.append(f"{name}: no trainable tensors")
        if record["missing_grad_tensors"]:
            failures.append(f"{name}: {len(record['missing_grad_tensors'])} tensors received no grad")
        if record["nonfinite_tensors"]:
            failures.append(f"{name}: {record['nonfinite_tensors']} tensors have non-finite grad")
        if not (record["grad_norm"] > 0.0 and math.isfinite(record["grad_norm"])):
            failures.append(f"{name}: grad norm is {record['grad_norm']}")
        if record["zero_grad_tensors"]:
            failures.append(f"{name}: {len(record['zero_grad_tensors'])} tensors have exactly zero grad")
    branches["unassigned"]["grad_norm"] = math.sqrt(branches["unassigned"].pop("grad_sq"))
    if branches["unassigned"]["tensors"]:
        failures.append(f"unassigned trainable tensors: {branches['unassigned']['tensors']}")

    object_shape = list(tail["object"].shape)
    if object_shape[1] != object_channels:
        failures.append(f"object output has {object_shape[1]} channels, expected {object_channels}")
    if tuple(object_shape[2:]) != (input_height, input_width):
        failures.append(f"object output grid {object_shape[2:]} != input {(input_height, input_width)}")
    if not math.isfinite(float(loss.item())):
        failures.append(f"loss is not finite: {loss.item()}")
    if not (refine_residual > 0.0):
        failures.append("refinement branch contributes an identically zero residual")

    total_params = sum(int(p.numel()) for p in model.parameters())
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    result = {
        "check": "hybrid_centerfusion_v1_launch_check",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "amp": amp_enabled,
        "amp_loss_scale_settled": float(scale),
        "amp_scale_reductions": int(scale_reductions),
        "feature_drop_fraction": 0.0,
        "batch_size": int(tensors.shape[0]),
        "loss": float(loss.item()),
        "loss_parts": {k: float(v) for k, v in parts.items()},
        "output_schema": {"segmentation": list(tail["out"].shape), "object": object_shape,
                          "object_channels_expected": object_channels},
        "split_boundary": {"encode_front": split_shapes,
                           "decode_tail": ["out", "object"]},
        "refinement_residual_max_abs": float(refine_residual),
        "branch_gradients": branches,
        "parameters": {"total": total_params, "trainable": trainable_params},
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2),
        "peak_vram_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 2),
        "warm_start": warm_start,
        "splits": {k: len(v) for k, v in splits.items()},
        "object_target_arm": target_arm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    printable = {k: v for k, v in result.items() if k != "warm_start"}
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
