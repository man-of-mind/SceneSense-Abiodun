#!/usr/bin/env python3
"""One gated 40-epoch native-grid run on the registered expanded training view."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
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

from pole_lraspp_multimodal_fusion.common import (  # noqa: E402
    load_config, read_manifest, set_reproducible_seeds,
)
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from scoring_v2 import (  # noqa: E402
    epoch10_gate, epoch20_gate, material_gain, primary_eligibility, rank_key,
    sensitivity_no_reversal, service_targets,
)

TRAINING_FIELDS = (
    "epoch", "stage", "object_lr_start", "object_lr_end", "inherited_lr_start",
    "inherited_lr_end", "train_total_loss", "validation_total_loss", "center_loss",
    "offset_loss", "loc_loss", "bbox2d_loss", "dim_loss", "yaw_loss", "parked_loss",
    "radar_support_loss", "seg_loss", "ce_loss", "lovasz_loss", "vehicle_iou",
    "person_box_mask_iou", "foreground_miou", "finite", "optimizer_steps",
    "epoch_seconds", "cuda_allocated_peak_mib", "cuda_reserved_peak_mib", "created_utc",
)
DECODE_FIELDS = (
    "epoch", "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_recall_002",
    "vehicle_xy_mae_m", "person_precision", "person_recall", "person_f1",
    "person_recall_002", "person_xy_mae_m", "vehicle_duplicate_fp",
    "person_heatmap_center_miss", "vehicle_iou", "person_box_mask_iou",
    "foreground_miou", "checkpoint_sha256", "prediction_set_sha256",
)


def _load_native_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "route_b_native_grid_registered_trainer_v1", NATIVE_PACKAGE / "train_native_v1.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("unable to load registered native-grid trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = _load_native_trainer()


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RegisteredSchedule:
    """Exact per-step H2 warm-up/hold and J2 warm-up/cosine schedule."""

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
            warm_steps = (
                int(j2["warmup_last_epoch"]) - int(j2["warmup_first_epoch"]) + 1
            ) * self.steps_per_epoch
            factor = min(1.0, warm_step / float(warm_steps))
            return {
                "inherited": float(j2["inherited_peak_lr"]) * factor,
                "object": float(j2["object_peak_lr"]) * factor,
            }
        decay_index = (epoch - int(j2["cosine_first_epoch"])) * self.steps_per_epoch + batch_index
        decay_steps = (
            int(j2["cosine_last_epoch"]) - int(j2["cosine_first_epoch"]) + 1
        ) * self.steps_per_epoch
        progress = decay_index / float(max(1, decay_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = float(j2["final_lr_ratio"]) + (1.0 - float(j2["final_lr_ratio"])) * cosine
        return {
            "inherited": float(j2["inherited_peak_lr"]) * ratio,
            "object": float(j2["object_peak_lr"]) * ratio,
        }

    def step(self, epoch: int, batch_index: int) -> dict[str, float]:
        values = self._values(epoch, batch_index)
        for group in self.optimizer.param_groups:
            group["lr"] = values[str(group["name"])]
        self.optimizer_steps += 1
        self.last_lr = dict(values)
        notable = (
            self.optimizer_steps in {1, 500}
            or batch_index == 0 or batch_index == self.steps_per_epoch - 1
            or (epoch, batch_index) in {(6, 0), (8, 0), (40, self.steps_per_epoch - 1)}
        )
        if notable:
            self.trace.append({
                "epoch": epoch, "batch_index_zero_based": batch_index,
                "optimizer_step": self.optimizer_steps, **values,
            })
        return values

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
    if epoch <= 5:
        return {"name": "H2", "freeze_backbone": True, "freeze_classifier": True}
    return {"name": "J2", "freeze_backbone": False, "freeze_classifier": False}


def rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: RegisteredSchedule, scaler: torch.amp.GradScaler, epoch: int,
    config: dict[str, Any], native_config: dict[str, Any], preflight: dict[str, Any],
    warm_mapping: dict[str, Any], input_size: tuple[int, int], radar_channels: int,
    object_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "route_b_v3_1_native_grid_expanded_training_checkpoint_v2",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "epoch": epoch,
        "rng_states": rng_states(),
        "resolved_config": config,
        "training_view_hashes": preflight["training_view_hashes"],
        "source_hashes": preflight["source_hashes"],
        "warm_start": warm_mapping["checkpoint"],
        "warm_start_sha256": config["warm_start_sha256"],
        # Compatibility fields consumed by the unchanged native inference loader.
        "config": native_config,
        "trial": config,
        "input_size": list(input_size),
        "radar_channels": radar_channels,
        "object_class_names": list(object_cfg["object_classes"]),
        "object_output_channels": native.OUTPUT_CHANNELS,
        "native_stride": int(object_cfg["native_stride"]),
        "native_grid": list(native.NATIVE_GRID),
        "object_hidden_channels": int(object_cfg.get("hidden_channels", 128)),
        "object_head_depth": int(object_cfg.get("head_depth", 3)),
        "model_task": "segmentation_plus_native_grid_object_localization",
    }


def append_decode(path: Path, record: dict[str, Any]) -> None:
    metric = record["metrics"]
    row = {
        "epoch": record["epoch"],
        **{key: metric[key] for key in (
            "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_recall_002",
            "vehicle_xy_mae_m", "person_precision", "person_recall", "person_f1",
            "person_recall_002", "person_xy_mae_m", "vehicle_iou",
            "person_box_mask_iou", "foreground_miou",
        )},
        "vehicle_duplicate_fp": record["vehicle_duplicate_fp"],
        "person_heatmap_center_miss": record["person_heatmap_center_miss"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "prediction_set_sha256": record["prediction_set_sha256"],
    }
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=DECODE_FIELDS).writerow(row)


def run_inference(experiment: Path, checkpoint: Path, checkpoint_hash: str, epoch: int) -> Path:
    tag = f"expanded_epoch_{epoch:03d}"
    prediction_root = experiment / "predictions" / tag
    command = [
        sys.executable, str(NATIVE_PACKAGE / "infer_native_v1.py"),
        "--experiment", str(experiment), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", checkpoint_hash, "--tag", tag,
    ]
    print(f"[expanded evaluation] authorized inference epoch={epoch}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"native inference failed at epoch {epoch}: exit {completed.returncode}")
    return prediction_root


def run_primary_scoring(
    experiment: Path, prediction_root: Path, checkpoint: Path,
    checkpoint_hash: str, epoch: int, output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable, str(PACKAGE_ROOT / "score_checkpoint_v2.py"),
        "--mode", "primary", "--experiment", str(experiment),
        "--prediction-root", str(prediction_root), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", checkpoint_hash, "--epoch", str(epoch),
        "--output", str(output),
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError(f"registered primary scoring failed at epoch {epoch}")
    return json.loads(output.read_text())


def run_sensitivity_scoring(
    experiment: Path, prediction_root: Path, output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable, str(PACKAGE_ROOT / "score_checkpoint_v2.py"),
        "--mode", "sensitivity", "--experiment", str(experiment),
        "--prediction-root", str(prediction_root), "--output", str(output),
    ]
    if subprocess.run(command, check=False).returncode != 0:
        raise RuntimeError("registered selected-only v0.25 scoring failed")
    return json.loads(output.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    config = json.loads(args.config.read_text())
    started = time.monotonic()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    preflight = json.loads((experiment / "PREFLIGHT.json").read_text())
    if not preflight["all_pass"]:
        raise RuntimeError("preflight gate is not green")

    set_reproducible_seeds(int(config["training_seed"]))
    device = torch.device("cuda")
    native_config = load_config(NATIVE_PACKAGE / "configs/route_b_v3_1_native_grid_v1.yaml")
    object_cfg = dict(native_config["object_heads"])
    input_size = tuple(int(value) for value in config["input_size"])
    radar_channels = int(native_config["fusion"]["radar_channels"])
    rows = read_manifest(experiment / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    if len(train_rows) != 16827 or len(val_rows) != 3345 or any(
        row["split"] not in {"train", "val"} for row in rows
    ):
        raise RuntimeError("expanded train/validation manifest drift")
    object_rows = load_object_boxes(experiment / "dataset/object_boxes.csv")
    loader_options = {
        "num_workers": int(config["num_workers"]), "pin_memory": True,
        "persistent_workers": bool(config["persistent_workers"]),
        "prefetch_factor": int(config["prefetch_factor"]),
    }
    train_loader = DataLoader(
        native.NativeGridDataset(
            experiment / "dataset", train_rows, object_rows, input_size, object_cfg,
            augment_strength=str(config["augment_strength"]),
            geometric_augment=bool(config["geometric_augment"]),
        ),
        batch_size=int(config["batch_size"]), shuffle=True, drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(
        native.NativeGridDataset(
            experiment / "dataset", val_rows, object_rows, input_size, object_cfg,
            augment_strength="off", geometric_augment=False,
        ),
        batch_size=int(config["batch_size"]), shuffle=False, drop_last=False,
        **loader_options,
    )
    model = native.build_native_grid_model(
        num_classes=int(native_config["training"].get("num_classes", 3)),
        radar_channels=radar_channels,
        hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        head_depth=int(object_cfg.get("head_depth", 3)), device=device,
    )
    warm_checkpoint = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    warm_mapping = native.load_warm_start(model, warm_checkpoint, device=device)
    if not warm_mapping["missing_keys_are_new_only"] or warm_mapping["incompatible_count"]:
        raise RuntimeError("native epoch-15 warm-start mapping failure")

    optimizer = torch.optim.AdamW(
        [
            {"params": native.inherited_parameters(model), "lr": 0.0, "name": "inherited"},
            {"params": native.object_parameters(model), "lr": 0.0, "name": "object"},
        ],
        lr=0.0, weight_decay=float(config["weight_decay"]),
    )
    scheduler = RegisteredSchedule(optimizer, len(train_loader), config)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["amp"]))
    class_weights = torch.tensor(
        config["class_loss_weights"], dtype=torch.float32, device=device
    )
    loss_weights = dict(config["loss_weights"])
    lovasz_weight = float(config["lovasz_weight"])

    metrics_dir = experiment / "metrics"
    metrics_dir.mkdir(exist_ok=False)
    training_csv = metrics_dir / "epoch_metrics.csv"
    decode_csv = metrics_dir / "decode_metrics.csv"
    with training_csv.open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=TRAINING_FIELDS).writeheader()
    with decode_csv.open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=DECODE_FIELDS).writeheader()
    checkpoint_dir = experiment / "checkpoints" / config["name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = experiment / "predictions"
    predictions_dir.mkdir(exist_ok=False)
    decisions_dir = experiment / "decisions"
    decisions_dir.mkdir(exist_ok=False)
    write_json_x(experiment / "TRAINING_STARTED.json", {
        "schema": "route_b_v3_1_native_grid_expanded_training_started_v2",
        "created_utc": utc_now(), "single_authorized_training_launch": True,
        "config": config, "warm_start_mapping": warm_mapping,
        "steps_per_epoch": len(train_loader),
    })

    decoded: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    peak_allocated = peak_reserved = 0.0
    terminal: str | None = None
    epochs_completed = 0

    for epoch in range(1, int(config["total_epochs"]) + 1):
        epoch_started = time.monotonic()
        stage = stage_for_epoch(epoch)
        native.apply_stage(model, stage, bool(config["freeze_batch_norm"]))
        native.stage_train_mode(model, stage, bool(config["freeze_batch_norm"]))
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
            with torch.autocast(
                device_type="cuda", enabled=scaler.is_enabled(),
                cache_enabled=bool(config["autocast_cache_enabled"]),
            ):
                loss, parts, _logits = native.compute_batch_losses(
                    model, tensors, masks, targets, loss_weights, class_weights, lovasz_weight
                )
            if not math.isfinite(float(loss.detach().item())):
                raise RuntimeError(f"nonfinite training loss epoch={epoch} batch={batch_index + 1}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
        row = {
            "epoch": epoch, "stage": stage["name"],
            "object_lr_start": first_lr["object"], "object_lr_end": last_lr["object"],
            "inherited_lr_start": first_lr["inherited"],
            "inherited_lr_end": last_lr["inherited"],
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
            "finite": finite,
            "optimizer_steps": scheduler.optimizer_steps,
            "epoch_seconds": time.monotonic() - epoch_started,
            "cuda_allocated_peak_mib": allocated, "cuda_reserved_peak_mib": reserved,
            "created_utc": utc_now(),
        }
        if not finite:
            raise RuntimeError(f"nonfinite epoch metrics at epoch {epoch}")
        training_rows.append(row)
        with training_csv.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=TRAINING_FIELDS).writerow(row)
        epochs_completed = epoch
        print(
            f"[expanded train] epoch={epoch}/40 stage={stage['name']} "
            f"train={row['train_total_loss']:.5f} val={row['validation_total_loss']:.5f} "
            f"lr_object={last_lr['object']:.9g} lr_inherited={last_lr['inherited']:.9g}",
            flush=True,
        )

        if epoch in set(config["checkpoint_epochs"]):
            checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            if checkpoint.exists():
                raise FileExistsError(f"refusing to overwrite {checkpoint}")
            torch.save(checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, config=config, native_config=native_config, preflight=preflight,
                warm_mapping=warm_mapping, input_size=input_size,
                radar_channels=radar_channels, object_cfg=object_cfg,
            ), checkpoint)
            checkpoint_hash = sha256(checkpoint)
            checkpoints[epoch] = {"path": str(checkpoint), "sha256": checkpoint_hash}
            prediction_root = run_inference(experiment, checkpoint, checkpoint_hash, epoch)
            score_output = decisions_dir / f"epoch_{epoch:03d}_decode.json"
            record = run_primary_scoring(
                experiment, prediction_root, checkpoint, checkpoint_hash, epoch, score_output
            )
            decoded.append(record)
            append_decode(decode_csv, record)
            print(json.dumps({
                "epoch": epoch, "vehicle_f1": record["metrics"]["vehicle_f1"],
                "person_f1": record["metrics"]["person_f1"],
                "vehicle_xy": record["metrics"]["vehicle_xy_mae_m"],
                "person_xy": record["metrics"]["person_xy_mae_m"],
                "duplicates": record["vehicle_duplicate_fp"],
                "heatmap_misses": record["person_heatmap_center_miss"],
            }, sort_keys=True), flush=True)

            if epoch == 10:
                gate = epoch10_gate(record, config)
                write_json_x(decisions_dir / "EPOCH_10_GATE.json", {
                    "epoch": 10, "gates": gate, "pass": all(gate.values()),
                    "terminal_on_fail": "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH10_INSTABILITY",
                })
                if not all(gate.values()):
                    terminal = "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH10_INSTABILITY"
                    break
            if epoch == 20:
                gate = epoch20_gate(record, config)
                write_json_x(decisions_dir / "EPOCH_20_GATE.json", {
                    "epoch": 20, "gates": gate, "pass": all(gate.values()),
                    "terminal_on_fail": "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH20_NO_PROGRESS",
                })
                if not all(gate.values()):
                    terminal = "LRASPP_EXPANDED_TRAINING_STOPPED_EPOCH20_NO_PROGRESS"
                    break

    write_json_x(experiment / "LR_SCHEDULE_TRACE.json", scheduler.state_dict())
    loss_candidates = [
        row for row in training_rows if row["epoch"] in {record["epoch"] for record in decoded}
    ]
    loss_best = min(loss_candidates, key=lambda row: row["validation_total_loss"])
    primary_candidates = []
    for record in decoded:
        guards = primary_eligibility(record, config)
        record["primary_eligibility_gates"] = guards
        record["primary_eligible"] = all(guards.values())
        if record["primary_eligible"]:
            primary_candidates.append(record)
    ranked_primary = sorted(primary_candidates, key=rank_key)
    best_ranked = sorted(decoded, key=rank_key)[0]
    selected = None
    sensitivity = None
    sensitivity_gates = None
    provisional = ranked_primary[0] if ranked_primary else None
    if terminal is None and provisional is not None:
        sensitivity = run_sensitivity_scoring(
            experiment, Path(provisional["prediction_root"]),
            decisions_dir / "SELECTED_V025_SENSITIVITY.json",
        )
        sensitivity_gates = sensitivity_no_reversal(sensitivity, config)
        if all(sensitivity_gates.values()):
            selected = provisional

    material = None
    service = None
    if terminal is None:
        if selected is not None:
            material = material_gain(
                selected, config, selected["primary_eligibility_gates"], sensitivity_gates
            )
            service = service_targets(selected)
        if selected is not None and material is not None and all(material.values()):
            terminal = (
                "LRASPP_EXPANDED_LONGTRAIN_SERVICE_READY"
                if service is not None and all(service.values())
                else "LRASPP_EXPANDED_LONGTRAIN_IMPROVED_NOT_SERVICE_READY"
            )
        else:
            terminal = "LRASPP_EXPANDED_LONGTRAIN_NO_GAIN"

    result = {
        "schema": "route_b_v3_1_native_grid_expanded_training_decision_v2",
        "created_utc": utc_now(), "terminal": terminal,
        "epochs_completed": epochs_completed,
        "decoded_epochs": [record["epoch"] for record in decoded],
        "training_rows": training_rows,
        "decode_records": decoded,
        "epoch10_gate": (
            json.loads((decisions_dir / "EPOCH_10_GATE.json").read_text())
            if (decisions_dir / "EPOCH_10_GATE.json").is_file() else None
        ),
        "epoch20_gate": (
            json.loads((decisions_dir / "EPOCH_20_GATE.json").read_text())
            if (decisions_dir / "EPOCH_20_GATE.json").is_file() else None
        ),
        "primary_eligible_epochs": [record["epoch"] for record in ranked_primary],
        "primary_ranking": [record["epoch"] for record in ranked_primary],
        "provisional_selected_epoch": provisional["epoch"] if provisional else None,
        "sensitivity_v025": sensitivity,
        "sensitivity_no_reversal_gates": sensitivity_gates,
        "selected": ({
            "epoch": selected["epoch"], "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "metrics_v010": selected["metrics"],
            "taxonomy_v010": selected["taxonomy_v010"],
        } if selected else None),
        "best_ranked_regardless_of_eligibility": {
            "epoch": best_ranked["epoch"], "checkpoint": best_ranked["checkpoint"],
            "checkpoint_sha256": best_ranked["checkpoint_sha256"],
        },
        "loss_best_checkpoint": {
            "epoch": loss_best["epoch"],
            "validation_total_loss": loss_best["validation_total_loss"],
            **checkpoints[loss_best["epoch"]],
            "auto_promoted": False,
        },
        "material_gain_gates": material,
        "service_targets": service,
        "baseline": config["baseline"], "baseline_v025": config["baseline_v025"],
        "checkpoint_hashes": checkpoints,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
    }
    write_json_x(experiment / "DECISION.json", result)
    write_json_x(experiment / "TRAINING_COMPLETE.json", {
        "terminal": terminal, "epochs_completed": epochs_completed,
        "optimizer_steps": scheduler.optimizer_steps,
        "wall_seconds": result["wall_seconds"],
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
        "checkpoint_hashes": checkpoints,
    })
    print(json.dumps({
        "terminal": terminal, "epochs_completed": epochs_completed,
        "decoded_epochs": result["decoded_epochs"],
        "selected_epoch": selected["epoch"] if selected else None,
        "best_ranked_epoch": best_ranked["epoch"],
        "loss_best_epoch": loss_best["epoch"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
