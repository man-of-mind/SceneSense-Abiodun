from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

from .base_runtime import load_base
from .contracts import (atomic_json, atomic_text, canonical_hash, current_commit, load_json, package_hashes,
                        require_qualified, resolve_repo_path, sha256, verify_original_provenance)
from .guards import PreStepBreaker
from .recovery_losses import compute_loss_groups
from .recovery_model import build_recovery_model
from .runner import run_guarded_epoch
from .state_guard import DiagnosticStateGuard


def _checkpoint(base: Any, model: torch.nn.Module, optimizer: torch.optim.Optimizer, *, epoch: int,
                global_update: int, sampler: Any, lrs: Mapping[str, float], qualified: Mapping[str, Any],
                qualification_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {"schema": "splitfusion_fcos_numerical_recovery_atomic_checkpoint_v1",
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": {"kind": "absolute_epoch_multistep", "lrs": dict(lrs),
                          "milestones_after_epochs": [16, 22], "gamma": 0.1},
            "amp": {"enabled": True, "dtype": "bfloat16", "grad_scaler": None},
            "epoch": int(epoch), "global_optimizer_update": int(global_update), "rng": base.common.capture_rng(),
            "sampler": sampler.state_dict(), "validation_accessed": False,
            "recovery": {"source_commit": current_commit(), "source_files_sha256": canonical_hash(package_hashes()),
                         "qualified_config_sha256": canonical_hash(qualified), "selected_tau": qualified["selected_tau"],
                         "qualification_artifact_hashes": dict(qualification_hashes),
                         "original_checkpoint_sha256": qualified["original_checkpoint_sha256"],
                         "original_registration_sha256": qualified["original_registration_sha256"],
                         "ceilings": qualified["ceilings"], "explicit_original_epoch9_resume": True}}


def _require_runtime_source_binding(qualified: Mapping[str, Any]) -> None:
    if (current_commit() != qualified["source_commit"]
            or canonical_hash(package_hashes()) != qualified["source_files_sha256"]):
        raise RuntimeError("source changed after qualification")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualified epoch-9 to epoch-26 continuation")
    parser.add_argument("--qualification-dir", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-scientific-continuation", required=True,
                        choices=("AUTHORIZED_EPOCH9_TO_EPOCH26_RECOVERY",))
    args = parser.parse_args()
    qualified, qualification = require_qualified(args.qualification_dir, args.authorization)
    _require_runtime_source_binding(qualified)
    provenance = verify_original_provenance(checkpoint_metadata=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("recovery experiment is create-only")
    base = load_base(); immutable = load_json(Path(__file__).with_name("recovery_config.json"))
    original_experiment = resolve_repo_path(immutable["original"]["experiment"])
    config = load_json(resolve_repo_path(immutable["original"]["config"]))
    runtime = load_json(original_experiment / "QUALIFIED_RUNTIME.json")
    calibration = load_json(original_experiment / "LOSS_CALIBRATION.json")
    priors = load_json(original_experiment / "TRAIN_ONLY_PRIORS.json")
    if (runtime["physical_batch"], runtime["gradient_accumulation"]) != (4, 4):
        raise RuntimeError("locked physical/accumulation batch drift")
    if not torch.cuda.is_available():
        raise RuntimeError("scientific continuation requires CUDA")
    device = torch.device("cuda:0"); total_memory = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 12288 * 2**20 / total_memory), device)
    model, build_report = build_recovery_model(priors, float(qualified["selected_tau"]), device)
    optimizer = base.train.build_optimizer(model, config)
    state = torch.load(Path(provenance["checkpoint"]), map_location="cpu", weights_only=False)
    if int(state["epoch"]) != 9 or int(state["global_optimizer_update"]) != 9468:
        raise RuntimeError("only the explicit complete epoch-9 checkpoint is admissible")
    model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
    base.common.restore_rng(state["rng"])
    if not base.train.all_model_finite(model) or not base.train.optimizer_finite(optimizer):
        raise FloatingPointError("epoch-9 recovery state is nonfinite")
    dataset_root = (base.common.ROOT / config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache((base.common.ROOT / config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = base.data.RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), cache, augment=True)
    qualification_root = args.qualification_dir.resolve(strict=True)
    qualification_hashes = {
        "RECOVERY_QUALIFICATION.json": sha256(qualification_root / "RECOVERY_QUALIFICATION.json"),
        "QUALIFIED_RECOVERY_CONFIG.json": sha256(qualification_root / "QUALIFIED_RECOVERY_CONFIG.json"),
        "QUALIFIED_TO_TRAIN": sha256(qualification_root / "QUALIFIED_TO_TRAIN"),
        "INDEPENDENT_SOURCE_REVIEW.json": sha256(qualification_root / "INDEPENDENT_SOURCE_REVIEW.json"),
        "INDEPENDENT_QUALIFICATION_REVIEW.json": sha256(
            qualification_root / "INDEPENDENT_QUALIFICATION_REVIEW.json"),
        "USER_SCIENTIFIC_AUTHORIZATION.json": sha256(args.authorization.resolve(strict=True)),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "training_metrics", "gradient_telemetry", "numerical_failures"):
        (output / name).mkdir()
    atomic_json(output / "RECOVERY_PROVENANCE.json", {"schema": "splitfusion_fcos_scientific_recovery_v1",
        "source_commit": current_commit(), "source_files": package_hashes(), "source_files_sha256": canonical_hash(package_hashes()),
        "original_epoch9": provenance, "original_epochs_10_26": "CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE",
        "qualified_config": qualified, "qualification_sha256": canonical_hash(qualification),
        "qualification_artifact_hashes": qualification_hashes, "build_report": build_report,
        "start_epoch": 10, "validation_accessed": False})
    breaker = PreStepBreaker(qualified["ceilings"], output / "numerical_failures")
    global_update = 9468
    registration = load_json(original_experiment / "SCIENTIFIC_REGISTRATION.json")
    diagnostic_sets = [registration["calibration_batches"][0]["indices"], registration["calibration_batches"][-1]["indices"]]
    for epoch in range(10, 27):
        _require_runtime_source_binding(qualified)
        base.model.configure_trainability(model, epoch); dataset.set_epoch(epoch)
        loader, sampler = base.train.dataloader(dataset, int(config["scientific_seed"]), epoch, 4,
                                                 int(config["training"]["workers"]))
        global_update, summary = run_guarded_epoch(base, model, optimizer, loader, config,
            calibration["multipliers"], epoch, global_update, 4, breaker)
        with DiagnosticStateGuard(model, optimizer) as state_guard:
            original_binding = base.train.compute_loss_groups
            base.train.compute_loss_groups = compute_loss_groups
            try:
                telemetry = base.train.diagnostic_telemetry(model, dataset, diagnostic_sets, calibration["multipliers"], 4)
            finally:
                base.train.compute_loss_groups = original_binding
        telemetry["full_state_guard"] = state_guard.report
        atomic_json(output / "training_metrics" / f"epoch_{epoch:03d}.json", summary)
        atomic_json(output / "gradient_telemetry" / f"epoch_{epoch:03d}.json", telemetry)
        payload = _checkpoint(base, model, optimizer, epoch=epoch, global_update=global_update,
                              sampler=sampler, lrs=summary["last_lrs"], qualified=qualified,
                              qualification_hashes=qualification_hashes)
        checkpoint = output / "checkpoints" / f"epoch_{epoch:03d}.pt"
        base.common.atomic_torch(checkpoint, payload)
        atomic_json(output / "checkpoints" / f"epoch_{epoch:03d}.json",
                    {"epoch": epoch, "path": str(checkpoint), "sha256": sha256(checkpoint),
                     "global_optimizer_update": global_update})
        atomic_json(output / "STATUS.json", {"state": "training", "epoch_complete": epoch,
                                              "global_optimizer_update": global_update, "validation_accessed": False}, overwrite=True)
    atomic_text(output / "TRAINING_COMPLETE", "EXACT_EPOCH9_RECOVERY_THROUGH_EPOCH26_COMPLETE\n")
    atomic_json(output / "TRAINING_COMPLETE.json", {"epochs_recovered": [10, 26], "global_optimizer_update": global_update,
                                                     "validation_accessed_during_training": False})
    atomic_json(output / "STATUS.json", {"state": "complete", "epoch_complete": 26,
                                          "global_optimizer_update": global_update, "validation_accessed": False}, overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
