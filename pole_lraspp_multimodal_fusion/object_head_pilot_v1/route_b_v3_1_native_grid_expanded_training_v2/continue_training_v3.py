#!/usr/bin/env python3
"""Epoch-boundary resumable worker for the authorized epoch-10 continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from continuation_policy_v3 import (  # noqa: E402
    all_finite, catastrophic_regression, decorate, rank_key, service_targets,
)
from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    load_config, read_manifest,
)
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402


def _load_native() -> Any:
    spec = importlib.util.spec_from_file_location(
        "continuation_registered_native_trainer_v3", NATIVE_PACKAGE / "train_native_v1.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load registered native trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = _load_native()

TRAINING_FIELDS = (
    "epoch", "attempt", "stage", "object_lr_start", "object_lr_end",
    "inherited_lr_start", "inherited_lr_end", "train_total_loss",
    "validation_total_loss", "center_loss", "offset_loss", "loc_loss",
    "bbox2d_loss", "dim_loss", "yaw_loss", "parked_loss",
    "radar_support_loss", "seg_loss", "ce_loss", "lovasz_loss",
    "vehicle_iou", "person_box_mask_iou", "foreground_miou", "finite",
    "optimizer_steps", "epoch_seconds", "cuda_allocated_peak_mib",
    "cuda_reserved_peak_mib", "created_utc",
)
DECODE_FIELDS = (
    "epoch", "vehicle_precision", "vehicle_recall", "vehicle_f1",
    "vehicle_recall_002", "vehicle_xy_mae_m", "vehicle_dimension_mae_m",
    "vehicle_yaw_mae_deg", "person_precision", "person_recall", "person_f1",
    "person_recall_002", "person_xy_mae_m", "person_dimension_mae_m",
    "person_yaw_mae_deg", "vehicle_duplicate_fp", "person_heatmap_center_miss",
    "vehicle_iou", "person_box_mask_iou", "foreground_miou",
    "checkpoint_sha256", "prediction_set_sha256",
)
PROGRESS_FIELDS = (
    "created_utc", "attempt", "phase", "epoch", "optimizer_steps", "detail",
)


class ContinuationStateInvalid(RuntimeError):
    pass


class CatastrophicRegression(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_progress(path: Path, attempt: int, phase: str, epoch: int | None,
                    optimizer_steps: int | None, detail: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS).writerow({
            "created_utc": utc_now(), "attempt": attempt, "phase": phase,
            "epoch": "" if epoch is None else epoch,
            "optimizer_steps": "" if optimizer_steps is None else optimizer_steps,
            "detail": detail,
        })


class RegisteredSchedule:
    """Exact absolute-step implementation of the registered H2/J2 schedule."""

    def __init__(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int,
                 config: dict[str, Any]) -> None:
        self.optimizer = optimizer
        self.steps_per_epoch = int(steps_per_epoch)
        self.config = config
        self.optimizer_steps = 0
        self.last_lr = {"object": 0.0, "inherited": 0.0}
        self.trace: list[dict[str, Any]] = []

    def _values(self, epoch: int, batch_index: int) -> dict[str, float]:
        h2, j2 = self.config["stage_h2"], self.config["stage_j2"]
        if epoch <= int(h2["last_epoch"]):
            h2_step = (epoch - 1) * self.steps_per_epoch + batch_index + 1
            factor = min(1.0, h2_step / float(h2["warmup_optimizer_steps"]))
            return {"inherited": 0.0, "object": float(h2["object_peak_lr"]) * factor}
        if epoch <= int(j2["warmup_last_epoch"]):
            warm_step = (epoch - int(j2["warmup_first_epoch"])) * self.steps_per_epoch + batch_index + 1
            warm_steps = (int(j2["warmup_last_epoch"]) - int(j2["warmup_first_epoch"]) + 1) * self.steps_per_epoch
            factor = min(1.0, warm_step / float(warm_steps))
            return {
                "inherited": float(j2["inherited_peak_lr"]) * factor,
                "object": float(j2["object_peak_lr"]) * factor,
            }
        decay_index = (epoch - int(j2["cosine_first_epoch"])) * self.steps_per_epoch + batch_index
        decay_steps = (int(j2["cosine_last_epoch"]) - int(j2["cosine_first_epoch"]) + 1) * self.steps_per_epoch
        progress = decay_index / float(max(1, decay_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = float(j2["final_lr_ratio"]) + (1.0 - float(j2["final_lr_ratio"])) * cosine
        return {
            "inherited": float(j2["inherited_peak_lr"]) * ratio,
            "object": float(j2["object_peak_lr"]) * ratio,
        }

    def step(self, epoch: int, batch_index: int) -> dict[str, float]:
        expected_step = (epoch - 1) * self.steps_per_epoch + batch_index + 1
        if expected_step != self.optimizer_steps + 1:
            raise ContinuationStateInvalid(
                f"scheduler discontinuity: expected absolute step {expected_step}, state has {self.optimizer_steps}"
            )
        values = self._values(epoch, batch_index)
        for group in self.optimizer.param_groups:
            group["lr"] = values[str(group["name"])]
        self.optimizer_steps += 1
        self.last_lr = dict(values)
        if batch_index in {0, self.steps_per_epoch - 1}:
            self.trace.append({
                "epoch": epoch, "batch_index_zero_based": batch_index,
                "optimizer_step": self.optimizer_steps, **values,
            })
        return values

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            state.get("schema") != "registered_h2_j2_warmup_cosine_v2"
            or state.get("steps_per_epoch") != self.steps_per_epoch
            or state.get("stage_h2") != self.config["stage_h2"]
            or state.get("stage_j2") != self.config["stage_j2"]
        ):
            raise ContinuationStateInvalid("scheduler contract mismatch")
        self.optimizer_steps = int(state["optimizer_steps"])
        self.last_lr = {key: float(value) for key, value in state["last_lr"].items()}
        self.trace = list(state.get("trace", []))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "registered_h2_j2_warmup_cosine_v2",
            "steps_per_epoch": self.steps_per_epoch,
            "optimizer_steps": self.optimizer_steps,
            "last_lr": self.last_lr,
            "trace": self.trace,
            "stage_h2": self.config["stage_h2"],
            "stage_j2": self.config["stage_j2"],
        }


def stage_for_epoch(epoch: int) -> dict[str, Any]:
    return {"name": "J2", "freeze_backbone": False, "freeze_classifier": False}


def rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(states: dict[str, Any]) -> None:
    if set(states) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ContinuationStateInvalid("incomplete RNG state")
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    torch.cuda.set_rng_state_all(states["torch_cuda"])


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: RegisteredSchedule, scaler: torch.amp.GradScaler, epoch: int,
    training: dict[str, Any], native_config: dict[str, Any], source: dict[str, Any],
    continuation: dict[str, Any], input_size: tuple[int, int], radar_channels: int,
    object_cfg: dict[str, Any], amp_patch: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": "route_b_v3_1_native_grid_expanded_continuation_checkpoint_v3",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "grad_scaler": scaler.state_dict(),
        "epoch": epoch, "rng_states": rng_states(), "resolved_config": training,
        "training_view_hashes": source["training_view_hashes"],
        "source_hashes": source["source_hashes"],
        "resume_origin": continuation["resume_checkpoint"],
        "resume_origin_sha256": continuation["resume_checkpoint_sha256"],
        "continuation_contract": continuation,
        "amp_numerical_patch": amp_patch,
        "warm_start": source.get("warm_start", source.get("resume_origin")),
        "warm_start_sha256": training["warm_start_sha256"],
        "config": native_config, "trial": training, "input_size": list(input_size),
        "radar_channels": radar_channels,
        "object_class_names": list(object_cfg["object_classes"]),
        "object_output_channels": native.OUTPUT_CHANNELS,
        "native_stride": int(object_cfg["native_stride"]),
        "native_grid": list(native.NATIVE_GRID),
        "object_hidden_channels": int(object_cfg.get("hidden_channels", 128)),
        "object_head_depth": int(object_cfg.get("head_depth", 3)),
        "model_task": "segmentation_plus_native_grid_object_localization",
    }


def save_create_only(path: Path, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ContinuationStateInvalid(f"refusing to overwrite checkpoint {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise ContinuationStateInvalid(f"stale partial checkpoint {temporary}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    digest = sha256(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "grad_scaler", "rng_states", "epoch"}
    integrity = {
        "path": str(path), "sha256": digest, "epoch": loaded.get("epoch"),
        "required_state_present": required.issubset(loaded),
        "optimizer_steps": loaded.get("scheduler", {}).get("optimizer_steps"),
        "finite_model": all(bool(torch.isfinite(value).all().item()) for value in loaded["model"].values()),
    }
    integrity["pass"] = (
        integrity["required_state_present"] and integrity["epoch"] == payload["epoch"]
        and integrity["optimizer_steps"] == payload["scheduler"]["optimizer_steps"]
        and integrity["finite_model"]
    )
    if not integrity["pass"]:
        raise CatastrophicRegression(f"checkpoint integrity failure at epoch {payload['epoch']}")
    return digest, integrity


def tensor_tree_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(tensor_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_tree_finite(item) for item in value)
    return True


def patched_loss(
    model: torch.nn.Module, tensors: torch.Tensor, masks: torch.Tensor,
    targets: dict[str, torch.Tensor], loss_weights: dict[str, Any],
    class_weights: torch.Tensor, lovasz_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
        outputs = model(tensors, feature_drop_fraction=0.0)
    with torch.autocast(device_type="cuda", enabled=False):
        seg_loss, seg_parts, seg_logits = native.segmentation_loss(
            outputs["out"].float(), masks, class_weights=class_weights,
            lovasz_weight=lovasz_weight,
        )
        object_loss, object_parts = native.native_object_loss(
            outputs["object"].float(), targets, loss_weights.get("object", {})
        )
        total = (
            float(loss_weights.get("segmentation", 0.3)) * seg_loss
            + float(loss_weights.get("object_total", 1.0)) * object_loss
        )
    parts = {**seg_parts, **object_parts, "object_loss": float(object_loss.detach().item())}
    return total, parts, seg_logits


def regular_loss(
    model: torch.nn.Module, tensors: torch.Tensor, masks: torch.Tensor,
    targets: dict[str, torch.Tensor], loss_weights: dict[str, Any],
    class_weights: torch.Tensor, lovasz_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
        return native.compute_batch_losses(
            model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
        )


def full_fp32_loss(
    model: torch.nn.Module, tensors: torch.Tensor, masks: torch.Tensor,
    targets: dict[str, torch.Tensor], loss_weights: dict[str, Any],
    class_weights: torch.Tensor, lovasz_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    with torch.autocast(device_type="cuda", enabled=False):
        return native.compute_batch_losses(
            model, tensors.float(), masks, targets, loss_weights, class_weights, lovasz_weight
        )


def diagnose_amp_overflow(
    model: torch.nn.Module, tensors: torch.Tensor, masks: torch.Tensor,
    targets: dict[str, torch.Tensor], loss_weights: dict[str, Any],
    class_weights: torch.Tensor, lovasz_weight: float,
) -> dict[str, Any]:
    inputs_finite = tensor_tree_finite((tensors, masks, targets))
    parameters_finite = all(bool(torch.isfinite(value).all().item()) for value in model.parameters())
    with torch.no_grad():
        amp_loss, amp_parts, _ = regular_loss(
            model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
        )
        fp32_loss, fp32_parts, _ = patched_loss(
            model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
        )
        full_loss, full_parts, _ = full_fp32_loss(
            model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
        )
    detail = {
        "inputs_finite": inputs_finite, "parameters_finite": parameters_finite,
        "amp_loss": float(amp_loss.detach().item()), "amp_parts": amp_parts,
        "patched_operation_loss": float(fp32_loss.detach().item()),
        "patched_operation_parts": fp32_parts,
        "full_fp32_loss": float(full_loss.detach().item()), "full_fp32_parts": full_parts,
    }
    detail["eligible"] = (
        inputs_finite and parameters_finite
        and not math.isfinite(detail["amp_loss"])
        and math.isfinite(detail["patched_operation_loss"])
        and math.isfinite(detail["full_fp32_loss"])
        and all(math.isfinite(float(value)) for value in fp32_parts.values())
        and all(math.isfinite(float(value)) for value in full_parts.values())
    )
    return detail


def run_inference(experiment: Path, checkpoint: Path, checkpoint_hash: str,
                  epoch: int, tag: str | None = None) -> Path:
    actual_tag = tag or f"continued_epoch_{epoch:03d}"
    prediction_root = experiment / "predictions" / actual_tag
    if (prediction_root / "INFERENCE_COMPLETE").is_file():
        manifest = json.loads((prediction_root / "inference_manifest.json").read_text())
        if manifest["checkpoint_sha256"] != checkpoint_hash:
            raise ContinuationStateInvalid(f"inference checkpoint hash drift: {actual_tag}")
        return prediction_root
    if prediction_root.exists():
        raise RuntimeError(f"incomplete create-only prediction directory: {prediction_root}")
    command = [
        sys.executable, str(NATIVE_PACKAGE / "infer_native_v1.py"),
        "--experiment", str(experiment), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", checkpoint_hash, "--tag", actual_tag,
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError(f"native inference failed at epoch {epoch}")
    return prediction_root


def run_score(command: list[str], output: Path) -> dict[str, Any]:
    if output.is_file():
        return json.loads(output.read_text())
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError(f"scoring failed: {' '.join(command[1:4])}")
    return json.loads(output.read_text())


def append_decode(path: Path, record: dict[str, Any]) -> None:
    metric = record["metrics"]
    row = {
        "epoch": record["epoch"],
        **{key: metric.get(key, "") for key in DECODE_FIELDS if key in metric},
        "vehicle_duplicate_fp": record["vehicle_duplicate_fp"],
        "person_heatmap_center_miss": record["person_heatmap_center_miss"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "prediction_set_sha256": record["prediction_set_sha256"],
    }
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=DECODE_FIELDS).writerow(row)


def update_status(experiment: Path, **values: Any) -> None:
    current = json.loads((experiment / "STATUS.json").read_text())
    current.update(values)
    current["updated_utc"] = utc_now()
    write_json_atomic(experiment / "STATUS.json", current)


def validate_resume_payload(
    payload: dict[str, Any], digest: str, expected_digest: str,
    training: dict[str, Any], steps_per_epoch: int,
) -> None:
    required = {"model", "optimizer", "scheduler", "grad_scaler", "rng_states", "epoch"}
    epoch = int(payload.get("epoch", -1))
    scheduler = payload.get("scheduler", {})
    if digest != expected_digest:
        raise ContinuationStateInvalid(f"resume SHA mismatch: {digest} != {expected_digest}")
    if not required.issubset(payload):
        raise ContinuationStateInvalid("resume checkpoint omits required state")
    if epoch < 10 or epoch > 40:
        raise ContinuationStateInvalid(f"illegal resume epoch {epoch}")
    if scheduler.get("steps_per_epoch") != steps_per_epoch:
        raise ContinuationStateInvalid("resume steps-per-epoch mismatch")
    if scheduler.get("optimizer_steps") != epoch * steps_per_epoch:
        raise ContinuationStateInvalid("resume optimizer-step mismatch")
    if payload.get("resolved_config") != training:
        raise ContinuationStateInvalid("resume resolved training config drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--continuation-config", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--resume-sha256", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    continuation = json.loads(args.continuation_config.read_text())
    training = json.loads(args.training_config.read_text())
    attempt = args.attempt
    started = time.monotonic()
    progress = experiment / "PROGRESS.csv"
    try:
        if sys.executable != "/usr/bin/python3":
            raise ContinuationStateInvalid(f"required /usr/bin/python3, got {sys.executable}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        preflight = json.loads((experiment / "PREFLIGHT.json").read_text())
        if not preflight["all_pass"]:
            raise ContinuationStateInvalid("preflight is not green")

        device = torch.device("cuda")
        native_config = load_config(NATIVE_PACKAGE / "configs/route_b_v3_1_native_grid_v1.yaml")
        object_cfg = dict(native_config["object_heads"])
        input_size = tuple(int(value) for value in training["input_size"])
        radar_channels = int(native_config["fusion"]["radar_channels"])
        rows = read_manifest(experiment / "dataset/manifest.csv")
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        if len(train_rows) != 16827 or len(val_rows) != 3345 or {row["split"] for row in rows} != {"train", "val"}:
            raise ContinuationStateInvalid("expanded manifest count/split drift")
        object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
        loader_options = {
            "num_workers": int(training["num_workers"]), "pin_memory": True,
            "persistent_workers": bool(training["persistent_workers"]),
            "prefetch_factor": int(training["prefetch_factor"]),
        }
        train_loader = DataLoader(
            native.NativeGridDataset(
                experiment / "dataset", train_rows, object_rows, input_size, object_cfg,
                augment_strength=str(training["augment_strength"]),
                geometric_augment=bool(training["geometric_augment"]),
            ),
            batch_size=int(training["batch_size"]), shuffle=True, drop_last=False,
            **loader_options,
        )
        val_loader = DataLoader(
            native.NativeGridDataset(
                experiment / "dataset", val_rows, object_rows, input_size, object_cfg,
                augment_strength="off", geometric_augment=False,
            ),
            batch_size=int(training["batch_size"]), shuffle=False, drop_last=False,
            **loader_options,
        )
        if len(train_loader) != int(continuation["resume_steps_per_epoch"]):
            raise ContinuationStateInvalid(f"steps per epoch drift: {len(train_loader)}")

        resume_path = args.resume_checkpoint.resolve(strict=True)
        resume_hash = sha256(resume_path)
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_resume_payload(resume, resume_hash, args.resume_sha256, training, len(train_loader))
        resume_epoch = int(resume["epoch"])
        model = native.build_native_grid_model(
            num_classes=int(native_config["training"].get("num_classes", 3)),
            radar_channels=radar_channels,
            hidden_channels=int(object_cfg.get("hidden_channels", 128)),
            head_depth=int(object_cfg.get("head_depth", 3)), device=device,
        )
        model.load_state_dict(resume["model"], strict=True)
        optimizer = torch.optim.AdamW([
            {"params": native.inherited_parameters(model), "lr": 0.0, "name": "inherited"},
            {"params": native.object_parameters(model), "lr": 0.0, "name": "object"},
        ], lr=0.0, weight_decay=float(training["weight_decay"]))
        optimizer.load_state_dict(resume["optimizer"])
        scheduler = RegisteredSchedule(optimizer, len(train_loader), training)
        scheduler.load_state_dict(resume["scheduler"])
        scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
        scaler.load_state_dict(resume["grad_scaler"])
        class_weights = torch.tensor(training["class_loss_weights"], dtype=torch.float32, device=device)
        loss_weights = dict(training["loss_weights"])
        lovasz_weight = float(training["lovasz_weight"])
        amp_patch = resume.get("amp_numerical_patch")
        restore_rng(resume["rng_states"])

        resume_verification = {
            "schema": "route_b_v3_1_native_grid_resume_verification_v3",
            "created_utc": utc_now(), "attempt": attempt,
            "checkpoint": str(resume_path), "checkpoint_sha256": resume_hash,
            "epoch": resume_epoch, "next_epoch": resume_epoch + 1,
            "optimizer_steps": scheduler.optimizer_steps,
            "optimizer_group_lrs": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "grad_scaler": scaler.state_dict(), "rng_keys": sorted(resume["rng_states"]),
            "model_state_strict": True, "optimizer_state_loaded": True,
            "scheduler_state_loaded": True, "amp_patch": amp_patch,
        }
        write_json_x(experiment / f"RESUME_VERIFICATION_ATTEMPT_{attempt}.json", resume_verification)
        append_progress(progress, attempt, "resume_verified", resume_epoch,
                        scheduler.optimizer_steps, str(resume_path))
        update_status(
            experiment, phase="training", attempt=attempt, last_safe_epoch=resume_epoch,
            next_epoch=resume_epoch + 1, resume_checkpoint=str(resume_path),
        )

        metrics_dir = experiment / "metrics"
        decisions_dir = experiment / "decisions"
        checkpoint_dir = experiment / "checkpoints" / continuation["name"]
        recovery_dir = experiment / "recovery_checkpoints"
        for directory in (metrics_dir, decisions_dir, checkpoint_dir, recovery_dir, experiment / "predictions"):
            directory.mkdir(parents=True, exist_ok=True)
        training_csv = metrics_dir / "epoch_metrics.csv"
        decode_csv = metrics_dir / "decode_metrics.csv"
        if attempt == 1:
            with training_csv.open("x", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=TRAINING_FIELDS).writeheader()
            with decode_csv.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=DECODE_FIELDS)
                writer.writeheader()
                epoch10 = json.loads((ROOT / continuation["resume_epoch10_evidence"]).read_text())
                row = {
                    "epoch": 10,
                    **{key: epoch10["metrics"].get(key, "") for key in DECODE_FIELDS if key in epoch10["metrics"]},
                    "vehicle_duplicate_fp": epoch10["vehicle_duplicate_fp"],
                    "person_heatmap_center_miss": epoch10["person_heatmap_center_miss"],
                    "checkpoint_sha256": epoch10["checkpoint_sha256"],
                    "prediction_set_sha256": epoch10["prediction_set_sha256"],
                }
                writer.writerow(row)
            write_json_x(decisions_dir / "epoch_010_decode_carried.json", epoch10)
            write_json_x(experiment / "TRAINING_STARTED.json", {
                "schema": "route_b_v3_1_native_grid_expanded_continuation_started_v3",
                "created_utc": utc_now(), "first_epoch": 11, "maximum_epoch": 40,
                "resume_checkpoint": str(resume_path), "resume_checkpoint_sha256": resume_hash,
                "resume_epoch": resume_epoch, "attempt": attempt,
            })

        epoch_rows = [
            json.loads(path.read_text())
            for path in sorted(metrics_dir.glob("epoch_???_training.json"))
        ]
        if epoch_rows and int(epoch_rows[-1]["epoch"]) != resume_epoch:
            raise ContinuationStateInvalid(
                f"progress/resume mismatch: last row {epoch_rows[-1]['epoch']} vs checkpoint {resume_epoch}"
            )
        decoded = [json.loads(path.read_text()) for path in sorted(decisions_dir.glob("epoch_0[234]0_decode.json"))]
        peak_allocated = max([float(row["cuda_allocated_peak_mib"]) for row in epoch_rows] or [0.0])
        peak_reserved = max([float(row["cuda_reserved_peak_mib"]) for row in epoch_rows] or [0.0])
        prior_latest = json.loads((experiment / "LATEST_SAFE.json").read_text()) if (experiment / "LATEST_SAFE.json").is_file() else {
            "epoch": resume_epoch, "path": str(resume_path), "sha256": resume_hash,
        }
        early_stop_reason: str | None = None
        catastrophic_detail: dict[str, Any] | None = None
        patch_verifications: list[dict[str, Any]] = (
            list(amp_patch.get("failing_batch_and_successor_verified", []))
            if amp_patch is not None else []
        )

        # A transient can occur after a designated checkpoint is safely written but
        # before its inference, scoring, or policy action completes. Finish that
        # create-only action before advancing to the next epoch on the one retry.
        if resume_epoch in set(continuation["decode_epochs"]):
            resume_record = next(
                (record for record in decoded if int(record["epoch"]) == resume_epoch), None
            )
            if resume_record is None:
                prediction_root = run_inference(
                    experiment, resume_path, resume_hash, resume_epoch
                )
                score_output = decisions_dir / f"epoch_{resume_epoch:03d}_decode.json"
                resume_record = run_score([
                    sys.executable, str(PACKAGE_ROOT / "score_continuation_v3.py"),
                    "--mode", "primary", "--experiment", str(experiment),
                    "--prediction-root", str(prediction_root), "--checkpoint", str(resume_path),
                    "--checkpoint-sha256", resume_hash, "--epoch", str(resume_epoch),
                    "--output", str(score_output),
                ], score_output)
                resume_record["checkpoint_state_integrity"] = True
                decoded.append(resume_record)
            with decode_csv.open("r", encoding="utf-8", newline="") as stream:
                csv_epochs = {int(row["epoch"]) for row in csv.DictReader(stream)}
            if resume_epoch not in csv_epochs:
                append_decode(decode_csv, resume_record)
            if resume_epoch in {20, 30}:
                policy_path = decisions_dir / f"EPOCH_{resume_epoch}_POLICY.json"
                if policy_path.is_file():
                    policy = json.loads(policy_path.read_text())
                else:
                    catastrophic = catastrophic_regression(
                        resume_record, training["baseline"], continuation
                    )
                    services = service_targets(resume_record, continuation)
                    policy = {
                        "epoch": resume_epoch, "catastrophic_guards": catastrophic,
                        "catastrophic": not all(catastrophic.values()),
                        "service_targets": services, "service_ready": all(services.values()),
                        "advisory": {
                            "vehicle_duplicate_fp": resume_record["vehicle_duplicate_fp"],
                            "validation_loss": epoch_rows[-1]["validation_total_loss"],
                            "absolute_service_targets_met": sum(services.values()),
                            "duplicate_fp_can_stop": False,
                        },
                    }
                    write_json_x(policy_path, policy)
                if policy["catastrophic"]:
                    catastrophic_detail = policy
                    early_stop_reason = f"catastrophic regression at epoch {resume_epoch}"
                elif policy["service_ready"]:
                    early_stop_reason = f"all nine service targets met at epoch {resume_epoch}"

        epochs_to_run = (
            range(resume_epoch + 1, int(continuation["maximum_epoch"]) + 1)
            if early_stop_reason is None else ()
        )
        for epoch in epochs_to_run:
            epoch_started = time.monotonic()
            stage = stage_for_epoch(epoch)
            native.apply_stage(model, stage, bool(training["freeze_batch_norm"]))
            native.stage_train_mode(model, stage, bool(training["freeze_batch_norm"]))
            torch.cuda.reset_peak_memory_stats(device)
            train_sums: dict[str, float] = {}
            first_lr: dict[str, float] | None = None
            last_lr: dict[str, float] | None = None
            batches = 0
            for batch_index, (tensors, masks, targets) in enumerate(train_loader):
                current_lr = scheduler.step(epoch, batch_index)
                first_lr = first_lr or dict(current_lr)
                last_lr = dict(current_lr)
                tensors = tensors.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                loss_fn = patched_loss if amp_patch is not None else regular_loss
                loss, parts, _logits = loss_fn(
                    model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
                )
                if not math.isfinite(float(loss.detach().item())):
                    if amp_patch is not None:
                        raise CatastrophicRegression(f"nonfinite loss with numerical patch epoch={epoch} batch={batch_index + 1}")
                    diagnosis = diagnose_amp_overflow(
                        model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
                    )
                    if not diagnosis["eligible"]:
                        raise CatastrophicRegression(
                            f"nonfinite training loss epoch={epoch} batch={batch_index + 1}"
                        )
                    amp_patch = {
                        "operation": "segmentation_loss", "precision": "FP32",
                        "epoch": epoch, "batch_one_based": batch_index + 1,
                        "diagnosis": diagnosis, "created_utc": utc_now(),
                    }
                    loss, parts, _logits = patched_loss(
                        model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
                    )
                old_scale = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < old_scale:
                    # A finite-loss GradScaler backoff is normal registered AMP
                    # behavior: scaler.step skips the unsafe update and scaler.update
                    # lowers the scale. It is neither a nonfinite model/loss/metric
                    # nor evidence that one deterministic FP16 operation overflowed.
                    append_progress(
                        progress, attempt, "amp_scaler_backoff", epoch,
                        scheduler.optimizer_steps,
                        f"batch={batch_index + 1};scale={old_scale}->{float(scaler.get_scale())}",
                    )
                if amp_patch is not None and len(patch_verifications) < 2:
                    patch_verifications.append({
                        "epoch": epoch, "batch_one_based": batch_index + 1,
                        "loss": float(loss.detach().item()), "finite": True,
                    })
                    if len(patch_verifications) == 2:
                        amp_patch["failing_batch_and_successor_verified"] = list(patch_verifications)
                        write_json_x(experiment / "AMP_NUMERICAL_RECOVERY.json", amp_patch)
                train_sums["total_loss"] = train_sums.get("total_loss", 0.0) + float(loss.detach().item())
                for key, value in parts.items():
                    train_sums[key] = train_sums.get(key, 0.0) + float(value)
                batches += 1

            validation = native.evaluate(
                model, val_loader, device, 3, loss_weights, class_weights, lovasz_weight
            )
            assert first_lr is not None and last_lr is not None
            allocated = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
            finite_values = [train_sums["total_loss"] / batches, float(validation["loss"])]
            finite_values.extend(float(validation[key]) for key in (
                "center_loss", "offset_loss", "loc_loss", "bbox2d_loss", "dim_loss",
                "yaw_loss", "parked_loss", "radar_support_loss", "seg_loss", "ce_loss",
                "lovasz_loss", "vehicle_iou", "person_iou", "miou",
            ))
            finite = all(math.isfinite(value) for value in finite_values)
            if not finite or not all(bool(torch.isfinite(value).all().item()) for value in model.state_dict().values()):
                raise CatastrophicRegression(f"nonfinite model/loss/metrics at epoch {epoch}")
            row = {
                "epoch": epoch, "attempt": attempt, "stage": "J2",
                "object_lr_start": first_lr["object"], "object_lr_end": last_lr["object"],
                "inherited_lr_start": first_lr["inherited"], "inherited_lr_end": last_lr["inherited"],
                "train_total_loss": train_sums["total_loss"] / batches,
                "validation_total_loss": validation["loss"],
                "center_loss": validation["center_loss"], "offset_loss": validation["offset_loss"],
                "loc_loss": validation["loc_loss"], "bbox2d_loss": validation["bbox2d_loss"],
                "dim_loss": validation["dim_loss"], "yaw_loss": validation["yaw_loss"],
                "parked_loss": validation["parked_loss"],
                "radar_support_loss": validation["radar_support_loss"],
                "seg_loss": validation["seg_loss"], "ce_loss": validation["ce_loss"],
                "lovasz_loss": validation["lovasz_loss"],
                "vehicle_iou": validation["vehicle_iou"],
                "person_box_mask_iou": validation["person_iou"],
                "foreground_miou": (validation["vehicle_iou"] + validation["person_iou"]) / 2.0,
                "finite": True, "optimizer_steps": scheduler.optimizer_steps,
                "epoch_seconds": time.monotonic() - epoch_started,
                "cuda_allocated_peak_mib": allocated, "cuda_reserved_peak_mib": reserved,
                "created_utc": utc_now(),
            }
            with training_csv.open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=TRAINING_FIELDS).writerow(row)
            write_json_x(metrics_dir / f"epoch_{epoch:03d}_training.json", row)
            epoch_rows.append(row)

            designated = epoch in set(continuation["checkpoint_epochs"])
            should_decode = epoch in set(continuation["decode_epochs"])
            checkpoint = (
                checkpoint_dir / f"epoch_{epoch:03d}.pt"
                if designated else recovery_dir / f"epoch_{epoch:03d}.pt"
            )
            source = {
                "training_view_hashes": resume["training_view_hashes"],
                "source_hashes": resume["source_hashes"],
                "warm_start": resume.get("warm_start"),
            }
            checkpoint_hash, integrity = save_create_only(
                checkpoint,
                checkpoint_payload(
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=epoch, training=training, native_config=native_config,
                    source=source, continuation=continuation, input_size=input_size,
                    radar_channels=radar_channels, object_cfg=object_cfg, amp_patch=amp_patch,
                ),
            )
            write_json_x(decisions_dir / f"epoch_{epoch:03d}_checkpoint_integrity.json", integrity)
            latest = {"epoch": epoch, "path": str(checkpoint), "sha256": checkpoint_hash,
                      "integrity": integrity, "created_utc": utc_now()}
            write_json_atomic(experiment / "LATEST_SAFE.json", latest)
            old_path = Path(prior_latest["path"])
            if old_path.parent == recovery_dir and old_path != checkpoint and old_path.is_file():
                old_path.unlink()
            prior_latest = latest
            append_progress(progress, attempt, "epoch_complete", epoch,
                            scheduler.optimizer_steps, f"checkpoint={checkpoint_hash}")
            update_status(
                experiment, phase="training", attempt=attempt, last_safe_epoch=epoch,
                next_epoch=(epoch + 1 if epoch < 40 else None),
                latest_safe_checkpoint=str(checkpoint), latest_safe_sha256=checkpoint_hash,
            )
            print(
                f"[continuation] epoch={epoch}/40 train={row['train_total_loss']:.6f} "
                f"val={row['validation_total_loss']:.6f} object_lr={last_lr['object']:.10g} "
                f"inherited_lr={last_lr['inherited']:.10g}", flush=True,
            )

            if should_decode:
                prediction_root = run_inference(experiment, checkpoint, checkpoint_hash, epoch)
                score_output = decisions_dir / f"epoch_{epoch:03d}_decode.json"
                record = run_score([
                    sys.executable, str(PACKAGE_ROOT / "score_continuation_v3.py"),
                    "--mode", "primary", "--experiment", str(experiment),
                    "--prediction-root", str(prediction_root), "--checkpoint", str(checkpoint),
                    "--checkpoint-sha256", checkpoint_hash, "--epoch", str(epoch),
                    "--output", str(score_output),
                ], score_output)
                record["checkpoint_state_integrity"] = integrity["pass"]
                decoded.append(record)
                append_decode(decode_csv, record)
                append_progress(progress, attempt, "primary_decode_complete", epoch,
                                scheduler.optimizer_steps, f"duplicate_fp={record['vehicle_duplicate_fp']}")
                update_status(experiment, phase="checkpoint_policy", last_decoded_epoch=epoch)
                if epoch in {20, 30}:
                    catastrophic = catastrophic_regression(record, training["baseline"], continuation)
                    services = service_targets(record, continuation)
                    policy = {
                        "epoch": epoch, "catastrophic_guards": catastrophic,
                        "catastrophic": not all(catastrophic.values()),
                        "service_targets": services, "service_ready": all(services.values()),
                        "advisory": {
                            "vehicle_duplicate_fp": record["vehicle_duplicate_fp"],
                            "validation_loss": row["validation_total_loss"],
                            "absolute_service_targets_met": sum(services.values()),
                            "duplicate_fp_can_stop": False,
                        },
                    }
                    write_json_x(decisions_dir / f"EPOCH_{epoch}_POLICY.json", policy)
                    if policy["catastrophic"]:
                        catastrophic_detail = policy
                        early_stop_reason = f"catastrophic regression at epoch {epoch}"
                        break
                    if policy["service_ready"]:
                        early_stop_reason = f"all nine service targets met at epoch {epoch}"
                        break

        write_json_x(experiment / "LR_SCHEDULE_TRACE.json", scheduler.state_dict())
        if amp_patch is not None and not (experiment / "AMP_NUMERICAL_RECOVERY.json").is_file():
            raise CatastrophicRegression("AMP numerical patch did not verify failing batch and successor")

        # The person-refinement program first needs an exact, retained epoch-40
        # engineering base.  In that registered recovery-only mode the continuation
        # worker stops here: it neither runs the old checkpoint-selection policy nor
        # performs the old selected-only v0.25 pass.  Checkpoint retention and the
        # sole requested epoch-40 primary decode are controlled independently by
        # checkpoint_epochs/decode_epochs above.
        if bool(continuation.get("recovery_only", False)):
            if [int(record["epoch"]) for record in decoded] != list(continuation["decode_epochs"]):
                raise ContinuationStateInvalid("recovery-only decode set drift")
            retained = []
            for retained_epoch in continuation["checkpoint_epochs"]:
                path = checkpoint_dir / f"epoch_{int(retained_epoch):03d}.pt"
                if not path.is_file():
                    raise ContinuationStateInvalid(f"missing retained Pareto checkpoint {path}")
                retained.append({
                    "epoch": int(retained_epoch), "path": str(path), "sha256": sha256(path),
                })
            decision = {
                "schema": "route_b_v3_1_native_grid_pareto_recovery_decision_v1",
                "created_utc": utc_now(), "phase": "base_recovery_complete",
                "epochs_completed": [row["epoch"] for row in epoch_rows],
                "decoded_epochs": [record["epoch"] for record in decoded],
                "training_rows": epoch_rows, "decode_records": decoded,
                "retained_checkpoints": retained,
                "wall_seconds": time.monotonic() - started,
                "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
                "automatic_recovery": amp_patch,
            }
            write_json_x(experiment / "BASE_RECOVERY_DECISION.json", decision)
            write_json_x(experiment / "WORKER_COMPLETE.json", {
                "phase": "base_recovery_complete", "attempt": attempt,
                "created_utc": utc_now(), "retained_checkpoints": retained,
            })
            update_status(experiment, phase="base_recovery_complete", last_decoded_epoch=40)
            append_progress(progress, attempt, "base_recovery_complete", 40,
                            scheduler.optimizer_steps, retained[-1]["sha256"])
            print(json.dumps(decision, indent=2), flush=True)
            return 0

        if catastrophic_detail is not None:
            terminal = "LRASPP_EXPANDED_LONGTRAIN_CATASTROPHIC_REGRESSION"
            decision = {
                "schema": "route_b_v3_1_native_grid_expanded_continuation_decision_v3",
                "created_utc": utc_now(), "terminal": terminal,
                "epochs_completed": [row["epoch"] for row in epoch_rows],
                "decoded_epochs": [10] + [record["epoch"] for record in decoded],
                "early_stop_reason": early_stop_reason,
                "catastrophic_detail": catastrophic_detail, "selected": None,
                "retained_checkpoint": continuation["resume_checkpoint"],
                "retained_checkpoint_sha256": continuation["resume_checkpoint_sha256"],
                "training_rows": epoch_rows, "decode_records": decoded,
                "wall_seconds": time.monotonic() - started,
                "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
                "automatic_recovery": amp_patch,
            }
            write_json_x(experiment / "DECISION.json", decision)
            write_json_x(experiment / "WORKER_COMPLETE.json", {
                "terminal": terminal, "attempt": attempt, "created_utc": utc_now(),
            })
            return 0

        baseline_path = (ROOT / continuation["amended_baseline"]).resolve(strict=True)
        amended = json.loads(baseline_path.read_text())
        baseline_prediction_root = Path(amended["retained_predictions"]).parent
        baseline_output = decisions_dir / "amended_baseline_primary.json"
        baseline_record = run_score([
            sys.executable, str(PACKAGE_ROOT / "score_continuation_v3.py"),
            "--mode", "baseline-primary", "--experiment", str(experiment),
            "--prediction-root", str(baseline_prediction_root),
            "--amended-baseline", str(baseline_path), "--output", str(baseline_output),
        ], baseline_output)
        epoch10 = json.loads((decisions_dir / "epoch_010_decode_carried.json").read_text())
        epoch10.update({"label": "epoch_010", "selection_order": 10,
                        "checkpoint_state_integrity": True})
        baseline_record = decorate(baseline_record, training["baseline"], continuation)
        epoch10 = decorate(epoch10, training["baseline"], continuation)
        candidates = [baseline_record, epoch10]
        for record in decoded:
            record.update({"label": f"epoch_{record['epoch']:03d}",
                           "selection_order": int(record["epoch"])})
            candidates.append(decorate(record, training["baseline"], continuation))
        eligible = [record for record in candidates if record["eligible"]]
        ranked = sorted(eligible, key=lambda record: rank_key(record, continuation))
        if not ranked:
            raise ContinuationStateInvalid("amended baseline unexpectedly ineligible")
        selected = ranked[0]
        selected_epoch = selected.get("epoch")
        if selected["label"] == "amended_baseline":
            terminal = "LRASPP_EXPANDED_LONGTRAIN_NO_GAIN"
            selected_prediction_root = baseline_prediction_root
        else:
            terminal = (
                "LRASPP_EXPANDED_LONGTRAIN_SERVICE_READY"
                if all(selected["service_targets"].values())
                else "LRASPP_EXPANDED_LONGTRAIN_IMPROVED_NOT_SERVICE_READY"
            )
            selected_prediction_root = Path(selected.get("prediction_root", ""))
            if not (selected_prediction_root / "INFERENCE_COMPLETE").is_file():
                selected_prediction_root = run_inference(
                    experiment, Path(selected["checkpoint"]), selected["checkpoint_sha256"],
                    int(selected_epoch), tag=f"selected_sensitivity_epoch_{int(selected_epoch):03d}",
                )
        sensitivity_output = decisions_dir / "SELECTED_V025_SENSITIVITY.json"
        sensitivity = run_score([
            sys.executable, str(PACKAGE_ROOT / "score_continuation_v3.py"),
            "--mode", "sensitivity", "--experiment", str(experiment),
            "--prediction-root", str(selected_prediction_root),
            "--output", str(sensitivity_output),
        ], sensitivity_output)

        decision = {
            "schema": "route_b_v3_1_native_grid_expanded_continuation_decision_v3",
            "created_utc": utc_now(), "terminal": terminal,
            "epochs_completed": [row["epoch"] for row in epoch_rows],
            "decoded_epochs": [10] + [record["epoch"] for record in decoded],
            "early_stop_reason": early_stop_reason or "epoch 40 maximum reached; final selection performed",
            "training_rows": epoch_rows, "decode_records": candidates,
            "eligible_labels": [record["label"] for record in eligible],
            "ranking": [{
                "label": record["label"], "service_target_count": record["service_target_count"],
                "minimum_class_recall": record["metrics"]["minimum_class_recall"],
                "mean_class_f1": record["metrics"]["mean_class_f1"],
                "mean_xy_mae_m": record["metrics"]["mean_xy_mae_m"],
                "vehicle_duplicate_fp": record["vehicle_duplicate_fp"],
            } for record in ranked],
            "selected": {
                "label": selected["label"], "epoch": selected_epoch,
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "metrics_v010": selected["metrics"],
                "taxonomy_v010": selected["taxonomy_v010"],
                "eligibility_gates": selected["eligibility_gates"],
                "service_targets": selected["service_targets"],
                "service_target_count": selected["service_target_count"],
            },
            "sensitivity_v025": sensitivity,
            "baseline": training["baseline"], "baseline_v025": training["baseline_v025"],
            "retained_checkpoint": selected["checkpoint"],
            "retained_checkpoint_sha256": selected["checkpoint_sha256"],
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
            "automatic_recovery": amp_patch,
        }
        write_json_x(experiment / "DECISION.json", decision)
        write_json_x(experiment / "WORKER_COMPLETE.json", {
            "terminal": terminal, "attempt": attempt, "created_utc": utc_now(),
            "selected": decision["selected"],
        })
        update_status(
            experiment, phase="worker_complete", terminal=terminal,
            selected_checkpoint=selected["checkpoint"],
            selected_checkpoint_sha256=selected["checkpoint_sha256"],
        )
        append_progress(progress, attempt, "worker_complete", selected_epoch,
                        scheduler.optimizer_steps, terminal)
        print(json.dumps({
            "terminal": terminal, "selected": decision["selected"],
            "decoded_epochs": decision["decoded_epochs"],
        }, indent=2), flush=True)
        return 0
    except ContinuationStateInvalid as caught:
        kind, code = "state_invalid", 20
        error_text = f"{type(caught).__name__}: {caught}"
    except CatastrophicRegression as caught:
        kind, code = "catastrophic", 21
        error_text = f"{type(caught).__name__}: {caught}"
    except Exception as caught:
        kind, code = "runtime", 22
        error_text = f"{type(caught).__name__}: {caught}"
    failure = {
        "schema": "route_b_v3_1_native_grid_expanded_continuation_worker_failure_v3",
        "created_utc": utc_now(), "attempt": attempt, "kind": kind,
        "error": error_text, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / f"WORKER_FAILURE_ATTEMPT_{attempt}.json", failure)
    append_progress(progress, attempt, f"worker_{kind}", None, None, failure["error"])
    update_status(experiment, phase=f"worker_{kind}", attempt=attempt, error=failure["error"])
    print(json.dumps(failure, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
