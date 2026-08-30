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


def _qualification_artifact_hashes(qualification_root: Path, authorization: Path) -> dict[str, str]:
    return {
        "RECOVERY_QUALIFICATION.json": sha256(qualification_root / "RECOVERY_QUALIFICATION.json"),
        "QUALIFIED_RECOVERY_CONFIG.json": sha256(qualification_root / "QUALIFIED_RECOVERY_CONFIG.json"),
        "QUALIFIED_TO_TRAIN": sha256(qualification_root / "QUALIFIED_TO_TRAIN"),
        "INDEPENDENT_SOURCE_REVIEW.json": sha256(qualification_root / "INDEPENDENT_SOURCE_REVIEW.json"),
        "INDEPENDENT_QUALIFICATION_REVIEW.json": sha256(
            qualification_root / "INDEPENDENT_QUALIFICATION_REVIEW.json"),
        "USER_SCIENTIFIC_AUTHORIZATION.json": sha256(authorization),
    }


def _recovery_binding(qualified: Mapping[str, Any], qualification_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {"source_commit": current_commit(), "source_files_sha256": canonical_hash(package_hashes()),
            "qualified_config_sha256": canonical_hash(qualified), "selected_tau": qualified["selected_tau"],
            "qualification_artifact_hashes": dict(qualification_hashes),
            "original_checkpoint_sha256": qualified["original_checkpoint_sha256"],
            "original_registration_sha256": qualified["original_registration_sha256"],
            "ceilings": qualified["ceilings"], "explicit_original_epoch9_resume": True}


def _expected_scheduler(config: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    decay = 0 if epoch <= 16 else 1 if epoch <= 22 else 2
    lrs = {name: float(value) * float(config["training"]["gamma"]) ** decay
           for name, value in config["training"]["base_lrs"].items()}
    return {"kind": "absolute_epoch_multistep", "lrs": lrs,
            "milestones_after_epochs": [16, 22], "gamma": 0.1}


def _select_verified_resume_checkpoint(output: Path, *, expected_recovery: Mapping[str, Any],
                                       config: Mapping[str, Any], train_frames: int,
                                       effective_batch: int) -> tuple[int, int, Mapping[str, Any]]:
    """Return only the latest contiguous, hash-bound, complete epoch checkpoint."""
    checkpoints = output / "checkpoints"
    if not checkpoints.is_dir():
        raise RuntimeError("resume output lacks checkpoint directory")
    hidden_partials = sorted(path.name for path in checkpoints.iterdir() if path.name.startswith(".epoch_"))
    if hidden_partials:
        raise RuntimeError(f"partial atomic checkpoint files require intervention: {hidden_partials}")
    updates_per_epoch = (int(train_frames) + int(effective_batch) - 1) // int(effective_batch)
    latest: tuple[int, int, Mapping[str, Any]] | None = None
    missing_seen = False
    for epoch in range(10, 27):
        checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
        sidecar_path = checkpoints / f"epoch_{epoch:03d}.json"
        metrics_path = output / "training_metrics" / f"epoch_{epoch:03d}.json"
        telemetry_path = output / "gradient_telemetry" / f"epoch_{epoch:03d}.json"
        present = (checkpoint.is_file(), sidecar_path.is_file(), metrics_path.is_file(), telemetry_path.is_file())
        if not any(present):
            missing_seen = True
            continue
        if missing_seen or not all(present):
            raise RuntimeError(f"partial or noncontiguous recovered epoch {epoch} artifacts require intervention")
        if any(path.is_symlink() for path in (checkpoint, sidecar_path, metrics_path, telemetry_path)):
            raise RuntimeError(f"recovered epoch {epoch} artifacts must be regular files in the same output")
        sidecar = load_json(sidecar_path)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected_global = 9468 + (epoch - 9) * updates_per_epoch
        expected_scheduler = _expected_scheduler(config, epoch)
        expected_sampler = {"length": int(train_frames), "seed": int(config["scientific_seed"]),
                            "epoch": epoch, "start_index": 0}
        optimizer_param_groups = state.get("optimizer", {}).get("param_groups", [])
        optimizer_group_names = [str(group.get("name")) for group in optimizer_param_groups]
        optimizer_groups = {str(group.get("name")): float(group["lr"]) for group in optimizer_param_groups}
        checks = (
            sidecar.get("epoch") == epoch,
            Path(sidecar.get("path", "")).resolve() == checkpoint.resolve(),
            sidecar.get("sha256") == sha256(checkpoint),
            int(sidecar.get("global_optimizer_update", -1)) == expected_global,
            state.get("schema") == "splitfusion_fcos_numerical_recovery_atomic_checkpoint_v1",
            int(state.get("epoch", -1)) == epoch,
            int(state.get("global_optimizer_update", -1)) == expected_global,
            state.get("scheduler") == expected_scheduler,
            state.get("sampler") == expected_sampler,
            set(state.get("rng", {})) == {"python", "numpy", "torch", "cuda"},
            state.get("validation_accessed") is False,
            state.get("recovery") == dict(expected_recovery),
            optimizer_group_names == list(expected_scheduler["lrs"]),
            optimizer_groups == expected_scheduler["lrs"],
            state.get("amp") == {"enabled": True, "dtype": "bfloat16", "grad_scaler": None},
            all(key in state for key in ("model", "optimizer", "rng")),
            int(load_json(metrics_path).get("epoch", -1)) == epoch,
            int(load_json(telemetry_path).get("epoch", -1)) == epoch,
        )
        if not all(checks):
            raise RuntimeError(f"recovered epoch {epoch} checkpoint/provenance verification failed")
        latest = (epoch, expected_global, state)
    if latest is None:
        raise RuntimeError("resume requires at least one fully verified recovered epoch checkpoint")
    failures = list((output / "numerical_failures").glob("BREAKER_*.json"))
    if failures:
        raise RuntimeError("resume forbidden while a numerical breaker failure artifact exists")
    marker = output / "TRAINING_COMPLETE"
    completion = output / "TRAINING_COMPLETE.json"
    if marker.exists() or completion.exists():
        if not (marker.is_file() and completion.is_file() and latest[0] == 26):
            raise RuntimeError("partial or inconsistent terminal artifacts require intervention")
        raise RuntimeError("recovered scientific continuation is already complete")
    status_path = output / "STATUS.json"
    if status_path.is_file():
        status = load_json(status_path)
        status_epoch = int(status.get("epoch_complete", -1))
        expected_status_global = 9468 + (status_epoch - 9) * updates_per_epoch
        if status.get("validation_accessed") is not False:
            raise RuntimeError("resume status reports validation access")
        if (status.get("state") != "training" or not 10 <= status_epoch <= latest[0]
                or int(status.get("global_optimizer_update", -1)) != expected_status_global):
            raise RuntimeError("resume status is inconsistent with verified checkpoints")
    return latest


def _verify_resume_provenance(output: Path, *, qualified: Mapping[str, Any],
                              qualification: Mapping[str, Any], provenance: Mapping[str, Any],
                              qualification_hashes: Mapping[str, str]) -> None:
    saved = load_json(output / "RECOVERY_PROVENANCE.json")
    checks = (
        saved.get("schema") == "splitfusion_fcos_scientific_recovery_v1",
        saved.get("source_commit") == current_commit(),
        saved.get("source_files") == package_hashes(),
        saved.get("source_files_sha256") == canonical_hash(package_hashes()),
        saved.get("original_epoch9") == provenance,
        saved.get("original_epochs_10_26") == "CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE",
        saved.get("qualified_config") == qualified,
        saved.get("qualification_sha256") == canonical_hash(qualification),
        saved.get("qualification_artifact_hashes") == dict(qualification_hashes),
        saved.get("start_epoch") == 10,
        saved.get("validation_accessed") is False,
    )
    if not all(checks):
        raise RuntimeError("recovery experiment provenance does not bind the current qualified source")


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
            "recovery": _recovery_binding(qualified, qualification_hashes)}


def _require_runtime_source_binding(qualified: Mapping[str, Any]) -> None:
    if (current_commit() != qualified["source_commit"]
            or canonical_hash(package_hashes()) != qualified["source_files_sha256"]):
        raise RuntimeError("source changed after qualification")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualified epoch-9 to epoch-26 continuation")
    parser.add_argument("--qualification-dir", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-existing-epoch-boundary", action="store_true")
    parser.add_argument("--execute-scientific-continuation", required=True,
                        choices=("AUTHORIZED_EPOCH9_TO_EPOCH26_RECOVERY",))
    args = parser.parse_args()
    qualified, qualification = require_qualified(args.qualification_dir, args.authorization)
    _require_runtime_source_binding(qualified)
    provenance = verify_original_provenance(checkpoint_metadata=True)
    output = args.output.resolve()
    if output.exists() and not args.resume_existing_epoch_boundary:
        raise FileExistsError("new recovery experiment is create-only; explicit epoch-boundary resume was not requested")
    if not output.exists() and args.resume_existing_epoch_boundary:
        raise FileNotFoundError("epoch-boundary resume requires the existing recovery output directory")
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
    qualification_root = args.qualification_dir.resolve(strict=True)
    qualification_hashes = _qualification_artifact_hashes(
        qualification_root, args.authorization.resolve(strict=True))
    if args.resume_existing_epoch_boundary:
        _verify_resume_provenance(output, qualified=qualified, qualification=qualification,
                                  provenance=provenance, qualification_hashes=qualification_hashes)
        completed_epoch, global_update, state = _select_verified_resume_checkpoint(
            output, expected_recovery=_recovery_binding(qualified, qualification_hashes), config=config,
            train_frames=int(immutable["locked_science"]["train_frames"]),
            effective_batch=int(immutable["locked_science"]["effective_batch"]))
        start_epoch = completed_epoch + 1
    else:
        state = torch.load(Path(provenance["checkpoint"]), map_location="cpu", weights_only=False)
        if int(state["epoch"]) != 9 or int(state["global_optimizer_update"]) != 9468:
            raise RuntimeError("only the explicit complete epoch-9 checkpoint is admissible for a new run")
        global_update = 9468
        start_epoch = 10
    model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
    base.common.restore_rng(state["rng"])
    if not base.train.all_model_finite(model) or not base.train.optimizer_finite(optimizer):
        raise FloatingPointError("epoch-9 recovery state is nonfinite")
    dataset_root = (base.common.ROOT / config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache((base.common.ROOT / config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = base.data.RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), cache, augment=True)
    if not args.resume_existing_epoch_boundary:
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
    registration = load_json(original_experiment / "SCIENTIFIC_REGISTRATION.json")
    diagnostic_sets = [registration["calibration_batches"][0]["indices"], registration["calibration_batches"][-1]["indices"]]
    for epoch in range(start_epoch, 27):
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
        sidecar = output / "checkpoints" / f"epoch_{epoch:03d}.json"
        if checkpoint.exists() or sidecar.exists():
            raise FileExistsError(f"refusing to overwrite recovered epoch {epoch} checkpoint artifacts")
        base.common.atomic_torch(checkpoint, payload)
        atomic_json(sidecar,
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
