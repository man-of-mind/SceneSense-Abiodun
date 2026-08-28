#!/usr/bin/env python3
"""Qualify the FP32 repair at v2's deterministic failing batch and its successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
V2_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_factorized_localization_v2"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(V2_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from losses_v3 import factorized_localization_loss  # noqa: E402
from model_v3 import (  # noqa: E402
    build_factorized_model, freeze_for_localization, load_native_warm_start,
    localization_parameters,
)
from targets_v2 import FactorizedLocalizationDataset  # noqa: E402

QUALIFY_EPOCH = 2
QUALIFY_BATCHES = (134, 135)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_values = value[finite]
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "all_finite": bool(finite.all().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
        "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
    }


def module_gradients(model: torch.nn.Module) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("localization_trunk", "log_depth_head", "projected_3d_center_offset_head"):
        module = getattr(model, name)
        gradients = [parameter.grad for parameter in module.parameters()]
        output[name] = {
            "all_present": all(value is not None for value in gradients),
            "all_finite": all(
                value is not None and bool(torch.isfinite(value).all().item())
                for value in gradients
            ),
            "absolute_sum": sum(
                float(value.detach().float().abs().sum().item())
                for value in gradients if value is not None
            ),
        }
    return output


class WithSampleId(Dataset):
    def __init__(self, dataset: FactorizedLocalizationDataset,
                 rows: list[dict[str, str]]) -> None:
        self.dataset = dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        tensors, masks, targets = self.dataset[index]
        return tensors, masks, targets, self.rows[index]["sample_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--contract-experiment", required=True, type=Path)
    parser.add_argument("--root-cause", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root_cause = json.loads(args.root_cause.read_text(encoding="utf-8"))
    expected_failure = root_cause["root_cause"]
    if expected_failure["epoch"] != QUALIFY_EPOCH or expected_failure["batch"] != 134:
        raise RuntimeError("registered deterministic failure location drift")
    expected_failure_ids = list(expected_failure["sample_ids"])

    set_reproducible_seeds(int(config["training_seed"]))
    device = torch.device("cuda")
    checkpoint = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    if sha256(checkpoint) != config["warm_start_sha256"]:
        raise RuntimeError("warm-start SHA mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    object_cfg = dict(payload["config"]["object_heads"])
    dataset_dir = args.contract_experiment.resolve() / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    train_rows = [row for row in rows if row.get("split") == "train"]
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    dataset = FactorizedLocalizationDataset(
        dataset_dir, train_rows, object_rows, tuple(config["input_size"]), object_cfg,
        augment_strength=str(config["augment_strength"]),
        geometric_augment=bool(config["geometric_augment"]),
    )
    loader = DataLoader(
        WithSampleId(dataset, train_rows), batch_size=int(config["batch_size"]),
        shuffle=True, drop_last=False, num_workers=int(config["num_workers"]),
        pin_memory=True, persistent_workers=bool(config["persistent_workers"]),
        prefetch_factor=int(config["prefetch_factor"]),
    )

    model = build_factorized_model(
        num_classes=int(payload["config"]["training"].get("num_classes", 3)),
        radar_channels=int(payload["radar_channels"]),
        hidden_channels=int(payload["object_hidden_channels"]),
        head_depth=int(payload["object_head_depth"]),
        localization_hidden=int(config["localization_hidden_channels"]), device=device,
    )
    load_native_warm_start(model, checkpoint, device=device)
    freeze_for_localization(model)
    trainable = localization_parameters(model)
    trainable_ids = {id(parameter) for parameter in trainable}
    optimizer = torch.optim.AdamW(
        trainable, lr=float(config["localization_lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))
    captured: dict[str, Any] = {}
    capture_key: str | None = None

    def first_conv_hook(_module, inputs, result):
        if capture_key is not None:
            captured[capture_key] = {
                "input": tensor_stats(inputs[0]), "output": tensor_stats(result),
            }

    hook = model.localization_trunk[0].register_forward_hook(first_conv_hook)
    records: list[dict[str, Any]] = []
    try:
        for epoch in range(1, QUALIFY_EPOCH + 1):
            model.eval()
            model.localization_trunk.train()
            model.log_depth_head.train()
            model.projected_3d_center_offset_head.train()
            for batch_number, (tensors, _masks, targets, sample_ids) in enumerate(loader, 1):
                tensors = tensors.to(device, non_blocking=True)
                targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                capture_key = f"epoch_{epoch}_batch_{batch_number}" if (
                    epoch == QUALIFY_EPOCH and batch_number in QUALIFY_BATCHES
                ) else None
                with torch.autocast(
                    device_type="cuda", enabled=scaler.is_enabled(),
                    cache_enabled=bool(config["autocast_cache_enabled"]),
                ):
                    outputs = model.localization_training_outputs(tensors)
                loss, parts = factorized_localization_loss(
                    outputs["localization"], outputs["object"], targets, config["losses"]
                )
                if outputs["localization"].dtype != torch.float32:
                    raise RuntimeError("localization output escaped FP32 boundary")
                if not math.isfinite(float(loss.detach().item())):
                    raise RuntimeError(f"non-finite repaired loss epoch={epoch} batch={batch_number}")
                scaler.scale(loss).backward()
                if epoch == QUALIFY_EPOCH and batch_number in QUALIFY_BATCHES:
                    gradients = module_gradients(model)
                    frozen_gradient_tensors = sum(
                        parameter.grad is not None for parameter in model.parameters()
                        if id(parameter) not in trainable_ids
                    )
                    record = {
                        "epoch": epoch, "batch": batch_number,
                        "sample_ids": list(sample_ids),
                        "matches_registered_failure_batch": (
                            batch_number == 134 and list(sample_ids) == expected_failure_ids
                        ),
                        "localization_output": tensor_stats(outputs["localization"]),
                        "loss_dtype": str(loss.dtype), "loss": float(loss.detach().item()),
                        "loss_parts": parts, "first_convolution": captured[capture_key],
                        "gradients": gradients,
                        "frozen_gradient_tensors": frozen_gradient_tensors,
                        "grad_scaler_scale_before_step": float(scaler.get_scale()),
                    }
                    records.append(record)
                scaler.step(optimizer)
                scaler.update()
                if epoch == QUALIFY_EPOCH and batch_number == QUALIFY_BATCHES[-1]:
                    break
            if epoch == QUALIFY_EPOCH:
                break
            scheduler.step()
    finally:
        hook.remove()

    gates = {
        "registered_failure_batch_identity_reproduced": bool(
            records and records[0]["matches_registered_failure_batch"]
        ),
        "failing_and_successor_localization_fp32": len(records) == 2 and all(
            record["localization_output"]["dtype"] == "torch.float32"
            and record["loss_dtype"] == "torch.float32" for record in records
        ),
        "failing_and_successor_first_conv_finite": len(records) == 2 and all(
            record["first_convolution"]["input"]["all_finite"]
            and record["first_convolution"]["output"]["all_finite"] for record in records
        ),
        "failing_and_successor_loss_finite": len(records) == 2 and all(
            math.isfinite(record["loss"]) for record in records
        ),
        "all_new_component_gradients_present_finite_nonzero": len(records) == 2 and all(
            all(detail["all_present"] and detail["all_finite"] and detail["absolute_sum"] > 0
                for detail in record["gradients"].values()) for record in records
        ),
        "frozen_components_receive_no_gradients": len(records) == 2 and all(
            record["frozen_gradient_tensors"] == 0 for record in records
        ),
        "trainable_parameters_finite_after_successor": all(
            bool(torch.isfinite(parameter).all().item()) for parameter in trainable
        ),
    }
    result = {
        "schema": "route_b_v3_1_factorized_localization_fp32_qualification_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "warm_start_sha256": sha256(checkpoint),
        "repair": "localization trunk, heads, decode, unprojection, and losses in FP32",
        "records": records, "gates": gates,
        "all_gates_pass": all(gates.values()),
        "wall_seconds": time.monotonic() - started,
    }
    with (output / "FP32_QUALIFICATION.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    (output / "QUALIFICATION_COMPLETE").write_text(
        "PASS\n" if result["all_gates_pass"] else "FAIL\n", encoding="utf-8"
    )
    print(json.dumps({
        "all_gates_pass": result["all_gates_pass"], "gates": gates,
        "records": records, "output": str(output),
        "wall_seconds": result["wall_seconds"],
    }, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
