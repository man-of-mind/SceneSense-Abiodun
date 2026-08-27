#!/usr/bin/env python3
"""One real q=0 AMP launch batch for the clean CenterNet model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent.parent, HERE.parent.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from centernet_model_v1 import install  # noqa: E402
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

BRANCH_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "rgb_encoder": ("backbone.", "rgb_fpn."),
    "radar_encoder": ("radar_encoder.", "radar_fpn."),
    "primary_vehicle_heatmap": ("object_head.vehicle_heatmap_head.",),
    "primary_person_heatmap": ("object_head.person_heatmap_head.",),
    "regression_refinement": (
        "object_head.shared_trunk.",
        "object_head.regression_head.",
        "refinement_head.",
    ),
    "segmentation": ("fusion_projection.", "classifier."),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _branch(name: str) -> str:
    for branch, prefixes in BRANCH_PREFIXES.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return branch
    return "unassigned"


def _audit_view(dataset_dir: Path, rows: List[dict]) -> Dict[str, object]:
    missing: List[str] = []
    frame_mismatches: List[str] = []
    timestamp_mismatches: List[str] = []
    for row in rows:
        sample = str(row.get("sample_id", ""))
        for field in ("rgb_path", "mask_path", "radar_tensor_path"):
            path = dataset_dir / str(row.get(field, ""))
            if not path.is_file() and len(missing) < 20:
                missing.append(f"{sample}:{field}:{path}")
        if str(row.get("frame_id", "")) != str(row.get("radar_frame_id", "")):
            if len(frame_mismatches) < 20:
                frame_mismatches.append(sample)
        try:
            delta = abs(float(row["timestamp"]) - float(row["radar_timestamp"]))
        except (KeyError, TypeError, ValueError):
            delta = math.inf
        if delta > 1e-6 and len(timestamp_mismatches) < 20:
            timestamp_mismatches.append(sample)
    return {
        "missing_modality_paths": missing,
        "frame_id_mismatches": frame_mismatches,
        "timestamp_mismatches": timestamp_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int, choices=(16, 24))
    args = parser.parse_args()

    install()
    from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp

    config = load_config(args.config)
    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    train_cfg = config["training"]
    object_cfg = _deep_merge_dicts(config.get("object_heads", {}), trial["object_heads"])
    loss_weights = _deep_merge_dicts(train_cfg.get("loss_weights", {}), trial["loss_weights"])
    width, height = (int(value) for value in trial["input_size"])
    dataset_dir = args.experiment_dir.resolve() / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    splits = split_rows(rows)
    view_audit = _audit_view(dataset_dir, rows)
    failures: List[str] = []
    if {name: len(values) for name, values in splits.items()} != {
        "train": 6600, "val": 3588, "test": 0
    }:
        failures.append(f"unexpected split counts: { {k: len(v) for k, v in splits.items()} }")
    for name, values in view_audit.items():
        if values:
            failures.append(f"{name}: {len(values)} examples retained in audit output")
    if failures:
        raise RuntimeError("; ".join(failures))

    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    dataset = FusionPoleMultiTaskDataset(
        dataset_dir,
        splits["train"],
        object_rows,
        (width, height),
        object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    tensors, masks, targets = next(iter(loader))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real AMP launch batch")
    device = torch.device("cuda")
    model = build_multitask_fusion_lraspp(
        num_classes=int(train_cfg.get("num_classes", 3)),
        radar_channels=int(config.get("fusion", {}).get("radar_channels", 4)),
        pretrained=True,
        object_channels=14,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        head_arch=str(object_cfg["head_arch"]),
        predict_bbox2d=True,
        device=device,
    ).to(device)
    if bool(trial.get("freeze_bn", False)):
        _freeze_batch_norm(model)

    tensors = tensors.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    targets = _move_object_targets(targets, device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
        bundle = model.encode_front(tensors[:, :3], tensors[:, 3:7])
        direct = model.decode_tail(bundle, (height, width))
        complete = model(tensors, feature_drop_fraction=0.0)
    exact_split = all(torch.equal(direct[key], complete[key]) for key in ("out", "object"))
    split_shapes = {name: list(value.shape) for name, value in bundle.items()}

    model.train()
    if bool(trial.get("freeze_bn", False)):
        _freeze_batch_norm(model)
    torch.cuda.reset_peak_memory_stats(device)
    scale = 65536.0
    reductions = 0
    while True:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
            loss, parts, _ = compute_losses(
                model,
                tensors,
                masks,
                targets,
                3,
                loss_weights,
                class_weights=torch.tensor(trial["class_loss_weights"], device=device),
                lovasz_weight=float(trial["lovasz_weight"]),
                feature_drop_fraction=0.0,
            )
        (loss * scale).backward()
        overflow = any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters() if p.requires_grad
        )
        if not overflow or reductions >= 24:
            break
        scale /= 2.0
        reductions += 1

    branch_records: Dict[str, Dict[str, object]] = {
        name: {"tensors": 0, "gradient_norm": 0.0, "missing": [], "nonfinite": []}
        for name in (*BRANCH_PREFIXES, "unassigned")
    }
    inv_scale = 1.0 / scale
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        record = branch_records[_branch(name)]
        record["tensors"] = int(record["tensors"]) + 1
        if parameter.grad is None:
            record["missing"].append(name)
            continue
        gradient = parameter.grad.detach().float() * inv_scale
        if not bool(torch.isfinite(gradient).all().item()):
            record["nonfinite"].append(name)
            continue
        record["gradient_norm"] = float(record["gradient_norm"]) + float(gradient.square().sum().item())
    for name, record in branch_records.items():
        record["gradient_norm"] = math.sqrt(float(record["gradient_norm"]))
        if name in BRANCH_PREFIXES and not (
            math.isfinite(float(record["gradient_norm"])) and float(record["gradient_norm"]) > 0
        ):
            failures.append(f"{name} gradient norm is {record['gradient_norm']}")
        if name in BRANCH_PREFIXES and (record["missing"] or record["nonfinite"]):
            failures.append(
                f"{name} missing={len(record['missing'])} nonfinite={len(record['nonfinite'])}"
            )
    if branch_records["unassigned"]["tensors"]:
        failures.append(f"unassigned trainable tensors={branch_records['unassigned']['tensors']}")
    if not exact_split:
        failures.append("encode_front/decode_tail is not equal to the q=0 forward path")
    if list(direct["object"].shape) != [args.batch_size, 14, height, width]:
        failures.append(f"object shape mismatch: {list(direct['object'].shape)}")
    if list(direct["out"].shape) != [args.batch_size, 3, height // 4, width // 4]:
        failures.append(f"segmentation shape mismatch: {list(direct['out'].shape)}")
    if not math.isfinite(float(loss.item())):
        failures.append(f"loss is nonfinite: {loss.item()}")

    weight_path = Path.home() / ".cache/torch/hub/checkpoints/resnet34-b627a593.pth"
    result = {
        "check": "route_b_clean_centernet_launch_gate_v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "device": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "batch_size": int(args.batch_size),
        "q": 0.0,
        "amp": True,
        "amp_loss_scale_settled": scale,
        "amp_scale_reductions": reductions,
        "loss": float(loss.item()),
        "loss_parts": parts,
        "splits": {name: len(values) for name, values in splits.items()},
        "view_audit": view_audit,
        "output_shapes": {name: list(value.shape) for name, value in direct.items()},
        "split_boundary_shapes": split_shapes,
        "split_forward_exact": exact_split,
        "branch_gradients": branch_records,
        "parameters": {
            "total": sum(int(p.numel()) for p in model.parameters()),
            "trainable": sum(int(p.numel()) for p in model.parameters() if p.requires_grad),
        },
        "peak_vram_mib": float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2),
        "peak_vram_reserved_mib": float(torch.cuda.max_memory_reserved(device)) / (1024 ** 2),
        "pretrained_weight": {
            "path": str(weight_path),
            "url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "sha256": _sha256(weight_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

