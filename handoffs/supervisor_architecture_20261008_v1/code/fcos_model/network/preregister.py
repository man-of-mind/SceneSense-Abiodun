from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Any

import torch

from common import (CONFIG_PATH, EXPERIMENT_FAMILY, PACKAGE, ROOT, atomic_json, atomic_text,
                    canonical_hash, desktop_notify, load_json, named_tensor_hash, package_hashes,
                    read_csv, seed_everything, sha256, utc_now)
from data import (DepthCache, FrozenEpochSampler, RouteBDataset, load_split_rows,
                  training_priors)
from model import build_model, configure_trainability, parameter_inventory

START_CAPTURE_UTC = "2026-08-29T21:33:49Z"
START_HEAD = "35384e0106d61021459c30df20c8560eb7f9e131"
START_SUPER_STATUS = (
    "# branch.oid 35384e0106d61021459c30df20c8560eb7f9e131\n"
    "# branch.head master\n# branch.upstream origin/master\n# branch.ab +7 -0\n"
    "1 .M S.MU 160000 160000 160000 7473cdb52e1cf3c40e1e1f189f03b2785bf15610 "
    "7473cdb52e1cf3c40e1e1f189f03b2785bf15610 OAI/openairinterface5g\n"
)


def command(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


PHASE_A_FILES = {
    "NOTIFICATION_PHASE_A.json", "PREREGISTRATION_COMPLETE", "REGISTRATION_HASHES.json",
    "SCIENTIFIC_REGISTRATION.json", "SCIENTIFIC_REGISTRATION.md", "STARTING_WORKTREE.json",
    "STATUS.json", "TRAIN_ONLY_PRIORS.json",
}


def validate_experiment(path: Path, refresh: bool) -> Path:
    path = path.resolve()
    family = EXPERIMENT_FAMILY.resolve()
    if path.parent != family or re.fullmatch(r"[0-9]{8}_[0-9]{6}", path.name) is None:
        raise RuntimeError(f"experiment must be one timestamp directly under {family}")
    if refresh:
        if not path.is_dir() or not (path / "PREREGISTRATION_COMPLETE").is_file():
            raise RuntimeError("refresh requires an existing completed Phase A directory")
        observed = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
        unexpected = observed - PHASE_A_FILES
        if unexpected:
            raise RuntimeError(f"refresh forbidden after downstream artifacts: {sorted(unexpected)}")
        prior_status = load_json(path / "STATUS.json")
        prior_registration = load_json(path / "SCIENTIFIC_REGISTRATION.json")
        if (prior_status.get("phase") != "A" or prior_status.get("optimizer_steps") != 0
                or prior_registration.get("optimizer_steps_before_registration") != 0
                or prior_registration.get("optimizer_constructed") is not False):
            raise RuntimeError("refresh forbidden: Phase A zero-optimizer invariant is not intact")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def source_inventory(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if name and not list(module.children()):
            rows.append({
                "name": name, "type": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "parameters": sum(value.numel() for value in module.parameters(recurse=False)),
                "buffers": sum(value.numel() for value in module.buffers(recurse=False)),
            })
    return rows


def activation_inventory(model: torch.nn.Module, value: torch.Tensor) -> list[dict[str, Any]]:
    rows, handles = [], []
    def shapes(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return {"shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, (list, tuple)):
            return [shapes(child) for child in item]
        if isinstance(item, dict):
            return {str(key): shapes(child) for key, child in item.items()}
        return type(item).__name__
    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(
                lambda _module, inputs, output, n=name: rows.append({"name": n, "input": shapes(inputs), "output": shapes(output)})
            ))
    try:
        with torch.inference_mode():
            model(value, dense=True)
    finally:
        for handle in handles:
            handle.remove()
    return rows


def registration_markdown(registration: dict[str, Any]) -> str:
    return f"""# Scientific registration: SplitFusion FCOS R50 FPN P2-P7 V1

Created before every optimizer step: `{registration['created_utc']}`.

This is the only authorized architecture and scientific recipe. It uses one seven-channel normalized/padded tensor, a mathematically single split-parameter ResNet-50 convolution, an exact raw FP32 C2 noAE boundary, the official pretrained ResNet C3-C5/FPN P3-P7/FCOS towers, one new P2 path, a two-output car/person classifier transfer, and task-private semantic, dense-depth, and factorized geometry heads.

The train/validation populations are 16,827 frames from 10 episodes and 3,345 frames from two disjoint episodes. Only v0.10 determines selection; v0.25 is selected-checkpoint sensitivity. Locked test is absent and unopened.

The FCOS point sizes are 4/8/16/32/64/128 for P2-P7. Scale intervals are P2 (0,32), P3 (0,64), P4 (64,128), P5 (128,256), P6 (256,512), and P7 (512,infinity), with radius 1.5, box-inside, and minimum-area conflict resolution. P2 overlaps P3 without changing P3 positives.

Four fixed loss groups are calibrated on eight hashed train-only batches: D is fixed to 1.0; G/S/A use the preregistered clipped gradient-norm equations. The optimizer is SGD(momentum=0.9, weight_decay=1e-4), effective batch 16, epochs 1-3 new-module warm-up with 1,000-update linear LR warm-up, then differential joint LRs. Decays take effect after epochs 16 and 22. Evaluation is deferred until epoch 26 completes and then runs exactly epochs 3, 8, 16, 22, and 26.

Inference is one pass at score floor 0.02, standard FCOS square-root score, top-1000 per level, classwise 2-D NMS 0.60, and top-100 detections. Score 0.20 metrics derive from retained predictions. The registered selection order and nine service gates are recorded verbatim in `config.json` and the JSON registration.

Initial model state SHA-256: `{registration['initial_model']['state_sha256']}`. Source-state hash: `{registration['source_state']['canonical_sha256']}`. No optimizer existed while this document and its JSON companion were created.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--refresh-before-optimizer", action="store_true")
    args = parser.parse_args()
    refresh = bool(args.refresh_before_optimizer)
    experiment = validate_experiment(args.experiment, refresh)
    prior_hashes = load_json(experiment / "REGISTRATION_HASHES.json") if refresh else None
    config = load_json(CONFIG_PATH)
    if command("git", "branch", "--show-current").strip() != "master":
        raise RuntimeError("must remain on local master")
    if command("git", "rev-parse", "HEAD").strip() != START_HEAD:
        raise RuntimeError("starting master changed before preregistration")
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    train_cache_path = (ROOT / config["train_depth_cache"]).resolve(strict=True)
    val_cache_path = (ROOT / config["validation_depth_cache"]).resolve(strict=True)
    if sha256(dataset_root / "dataset/manifest.csv") != config["data"]["manifest_sha256"]:
        raise RuntimeError("dataset manifest hash drift")
    manifest = read_csv(dataset_root / "dataset/manifest.csv")
    splits_present = {row["split"] for row in manifest}
    if splits_present != {"train", "val"}:
        raise RuntimeError(f"locked-test/split drift: {splits_present}")
    for contract, split, expected in (("v010", "train", config["data"]["v010_train_objects_sha256"]),
                                      ("v010", "val", config["data"]["v010_val_objects_sha256"]),
                                      ("v025", "train", config["data"]["v025_train_objects_sha256"]),
                                      ("v025", "val", config["data"]["v025_val_objects_sha256"])):
        if sha256(dataset_root / f"contracts/{contract}/{split}/object_boxes.csv") != expected:
            raise RuntimeError(f"{contract}/{split} object contract drift")
    train_rows = load_split_rows(dataset_root, "train")
    val_rows = load_split_rows(dataset_root, "val")
    train_episodes = {row["experiment_id"] for row in train_rows}
    val_episodes = {row["experiment_id"] for row in val_rows}
    if len(train_episodes) != 10 or len(val_episodes) != 2 or train_episodes & val_episodes:
        raise RuntimeError("episode disjointness drift")
    DepthCache(train_cache_path, train_rows)
    DepthCache(val_cache_path, val_rows)
    priors = training_priors(dataset_root)
    atomic_json(experiment / "TRAIN_ONLY_PRIORS.json", priors, overwrite=refresh)
    seed = int(config["scientific_seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transfer = build_model(priors, device)
    configure_trainability(model, 1)
    state_hash = named_tensor_hash(model.state_dict().items())
    train_dataset = RouteBDataset(dataset_root, "train", seed, depth_cache=None, augment=False)
    value = train_dataset[0]["input"].unsqueeze(0).to(device)
    activations = activation_inventory(model, value)
    permutation = FrozenEpochSampler(len(train_rows), seed, epoch=1).order().tolist()
    starts = torch.linspace(0, len(permutation) - 16, 8).round().long().tolist()
    calibration_batches = []
    for batch_index, start in enumerate(starts):
        indices = permutation[start:start + 16]
        ids = [train_rows[index]["sample_id"] for index in indices]
        calibration_batches.append({"batch": batch_index, "sampler_start": start, "indices": indices,
                                    "sample_ids": ids, "sha256": canonical_hash(ids)})
    oai_status = command("git", "status", "--porcelain=v2", "--branch", "--untracked-files=all",
                         cwd=ROOT / "OAI/openairinterface5g")
    starting = {
        "schema": "splitfusion_fcos_starting_worktree_v1", "captured_utc": START_CAPTURE_UTC,
        "starting_head": START_HEAD, "branch": "master", "upstream_delta": {"ahead": 7, "behind": 0},
        "superproject_porcelain_v2": START_SUPER_STATUS,
        "preexisting_dirty_paths": ["OAI/openairinterface5g"],
        "oai_head": command("git", "rev-parse", "HEAD", cwd=ROOT / "OAI/openairinterface5g").strip(),
        "oai_porcelain_v2": oai_status, "oai_porcelain_sha256": canonical_hash(oai_status),
        "note": "The exact state was captured read-only before source creation and is materialized here later without reinterpretation."
    }
    if refresh:
        if load_json(experiment / "STARTING_WORKTREE.json") != starting:
            raise RuntimeError("starting-worktree evidence drift during Phase A refresh")
    else:
        atomic_json(experiment / "STARTING_WORKTREE.json", starting, overwrite=False)
    source_state = {"files": package_hashes()}
    source_state["canonical_sha256"] = canonical_hash(source_state["files"])
    source_paths = {
        "fcos": "/home/shr_aisvcs/.local/lib/python3.10/site-packages/torchvision/models/detection/fcos.py",
        "backbone_utils": "/home/shr_aisvcs/.local/lib/python3.10/site-packages/torchvision/models/detection/backbone_utils.py",
        "fpn": "/home/shr_aisvcs/.local/lib/python3.10/site-packages/torchvision/ops/feature_pyramid_network.py",
    }
    environment = {
        "python": platform.python_version(), "pytorch": torch.__version__,
        "torchvision": __import__("torchvision").__version__, "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else None,
        "source_paths": source_paths, "source_sha256": {name: sha256(Path(path)) for name, path in source_paths.items()},
    }
    registration = {
        "schema": "splitfusion_fcos_scientific_registration_v1", "created_utc": utc_now(),
        "optimizer_steps_before_registration": 0, "optimizer_constructed": False,
        "phase_a_refresh": {"performed": refresh, "reason": "complete source tree hash before any optimizer",
                            "superseded_registration_hashes": prior_hashes},
        "config": config, "config_sha256": sha256(CONFIG_PATH), "source_state": source_state,
        "starting_worktree": starting, "environment": environment,
        "data_audit": {"train_frames": len(train_rows), "validation_frames": len(val_rows),
                       "train_episodes": sorted(train_episodes), "validation_episodes": sorted(val_episodes),
                       "disjoint": not bool(train_episodes & val_episodes), "splits_present": sorted({row['split'] for row in manifest}),
                       "locked_test_accessed": False},
        "train_only_priors": priors, "train_only_priors_sha256": sha256(experiment / "TRAIN_ONLY_PRIORS.json"),
        "calibration_batches": calibration_batches,
        "calibration_batches_sha256": canonical_hash(calibration_batches),
        "initial_model": {"state_sha256": state_hash, "parameter_inventory": parameter_inventory(model),
                          "module_inventory": source_inventory(model), "activation_inventory": activations,
                          "transfer": transfer},
        "loss_equations": {
            "D": "L_focal_sigmoid_sum/N_fg + L_GIoU_sum/N_fg + L_BCE_centerness_sum/N_fg",
            "G": "actor mean of 1.5 CE(depth-bin including overflow) + 0.75 SmoothL1(0.5*tanh(residual),target) + 0.1 SmoothL1(analytic local XYZ/3,target/3) + SmoothL1(ray/stride,target) + 0.6 SmoothL1(log-dim,target-log-dim) + 0.15 SmoothL1(normalize(yaw),target)",
            "S": "weighted CE([0.5,1,4],ignore=-100) + 0.5 Lovasz-Softmax",
            "A": "SmoothL1(predicted log1p surface depth,target log1p depth) + 0.5 SmoothL1(radar-sampled predicted log1p depth,radar log1p depth)",
            "calibration": "wG=clip(0.50*gD/max(gG,eps),0.05,10); wS=clip(0.25*gD/max(gS,eps),0.05,10); wA=clip(0.10*gD/max(gA,eps),0.05,10)",
        },
        "padding_contract": {"content": [7, 432, 768], "network": [7, 448, 768],
                             "normalized_padding_value": 0.0, "semantic_ignore": -100,
                             "dense_valid_padding": False, "boxes_intrinsics_unchanged_by_bottom_padding": True},
        "transport_schema": load_json(PACKAGE / "transport_schema.json"),
        "validation_used_for_design": False, "validation_accessed": False,
        "excluded": ["q", "quantization", "AE", "hybrid-q", "live deployment", "CARLA", "OAI mutation", "locked test", "288-cell campaign"],
    }
    atomic_json(experiment / "SCIENTIFIC_REGISTRATION.json", registration, overwrite=refresh)
    atomic_text(experiment / "SCIENTIFIC_REGISTRATION.md", registration_markdown(registration), overwrite=refresh)
    hashes = {"json_sha256": sha256(experiment / "SCIENTIFIC_REGISTRATION.json"),
              "markdown_sha256": sha256(experiment / "SCIENTIFIC_REGISTRATION.md"),
              "source_state_sha256": source_state["canonical_sha256"], "created_utc": utc_now()}
    atomic_json(experiment / "REGISTRATION_HASHES.json", hashes, overwrite=refresh)
    atomic_json(experiment / "STATUS.json", {"phase": "A", "state": "complete", "created_utc": utc_now(),
                                              "optimizer_steps": 0, "validation_accessed": False})
    atomic_text(experiment / "PREREGISTRATION_COMPLETE", "PHASE_A_PREREGISTERED_BEFORE_OPTIMIZER\n", overwrite=refresh)
    atomic_json(experiment / "NOTIFICATION_PHASE_A.json", desktop_notify(
        "SplitFusion FCOS", "Phase A preregistration complete; no optimizer step or validation access."), overwrite=refresh)
    print(json.dumps({"experiment": str(experiment), "registration_hashes": hashes,
                      "model_parameters": registration["initial_model"]["parameter_inventory"]["total_parameters"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
