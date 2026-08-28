#!/usr/bin/env python3
"""Fail-closed state, data, source, and CUDA preflight for epoch-10 continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_payload_hashes(contracts: Path) -> dict[str, Any]:
    checked = 0
    failures: list[dict[str, str]] = []
    cache: dict[Path, str] = {}
    manifests: dict[str, str] = {}
    for visibility in ("v010", "v025"):
        for split in ("train", "val"):
            split_root = contracts / visibility / split
            manifest = split_root / "target_manifest.csv"
            manifests[f"{visibility}/{split}"] = sha256(manifest)
            for row in read_csv(manifest):
                for path_key, hash_key in (
                    ("segmentation_mask_path", "segmentation_mask_sha256"),
                    ("object_ignore_mask_path", "object_ignore_mask_sha256"),
                ):
                    payload = split_root / row[path_key]
                    resolved = payload.resolve(strict=True)
                    actual = cache.get(resolved)
                    if actual is None:
                        actual = sha256(payload)
                        cache[resolved] = actual
                    checked += 1
                    if actual != row[hash_key] and len(failures) < 20:
                        failures.append({
                            "sample_id": row["sample_id"], "path": str(payload),
                            "expected": row[hash_key], "actual": actual,
                        })
    return {
        "references_checked": checked,
        "unique_payloads_hashed": len(cache),
        "target_manifest_sha256": manifests,
        "failures": failures,
        "pass": not failures,
    }


def expected_lr(config: dict[str, Any], epoch: int, batch_index: int, steps: int) -> dict[str, float]:
    stage = config["stage_j2"]
    decay_index = (epoch - int(stage["cosine_first_epoch"])) * steps + batch_index
    decay_steps = (int(stage["cosine_last_epoch"]) - int(stage["cosine_first_epoch"]) + 1) * steps
    progress = decay_index / float(decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = float(stage["final_lr_ratio"]) + (1.0 - float(stage["final_lr_ratio"])) * cosine
    return {
        "inherited": float(stage["inherited_peak_lr"]) * ratio,
        "object": float(stage["object_peak_lr"]) * ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--continuation-config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    started = time.monotonic()
    contract = json.loads(args.continuation_config.read_text())
    training_path = (ROOT / contract["registered_training_config"]).resolve(strict=True)
    training = json.loads(training_path.read_text())
    checkpoint_path = (ROOT / contract["resume_checkpoint"]).resolve(strict=True)
    amended_path = (ROOT / contract["amended_baseline"]).resolve(strict=True)
    epoch10_path = (ROOT / contract["resume_epoch10_evidence"]).resolve(strict=True)
    checks: list[dict[str, Any]] = []

    sources = sorted(set(PACKAGE_ROOT.glob("*continuation*v3.py")) | {PACKAGE_ROOT / "score_continuation_v3.py"})
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, sources)],
        capture_output=True, text=True, check=False,
    )
    checks.append({
        "name": "continuation_sources_compile", "pass": compiled.returncode == 0,
        "sources": [str(path) for path in sources], "stderr": compiled.stderr[-2000:],
    })

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    checks.append({
        "name": "required_local_master_start", "pass": (
            branch == "master" and head == contract["required_starting_head"]
        ), "branch": branch, "head": head,
    })

    immutable_hashes = {
        "training_config": sha256(training_path),
        "resume_checkpoint": sha256(checkpoint_path),
        "epoch10_evidence": sha256(epoch10_path),
        "amended_baseline": sha256(amended_path),
        "dimension_yaw_scorer": sha256(ROOT / contract["dimension_yaw_scorer"]),
    }
    expected_immutable = {
        "training_config": contract["registered_training_config_sha256"],
        "resume_checkpoint": contract["resume_checkpoint_sha256"],
        "epoch10_evidence": contract["resume_epoch10_evidence_sha256"],
        "amended_baseline": contract["amended_baseline_sha256"],
        "dimension_yaw_scorer": contract["dimension_yaw_scorer_sha256"],
    }
    checks.append({
        "name": "immutable_inputs", "pass": immutable_hashes == expected_immutable,
        "actual": immutable_hashes, "expected": expected_immutable,
    })

    view = (ROOT / training["training_view"]).resolve(strict=True)
    camera_contract = (ROOT / training["expanded_contract"]).resolve(strict=True)
    base_contract = (ROOT / training["expanded_base_contract"]).resolve(strict=True)
    view_summary_path = view / "EXPANDED_TRAIN_VIEW_SUMMARY.json"
    camera_summary_path = camera_contract / "CAMERA_PLANE_CONTRACT_SUMMARY.json"
    base_summary_path = base_contract / "GT_CONTRACT_SUMMARY.json"
    view_summary = json.loads(view_summary_path.read_text())
    camera_summary = json.loads(camera_summary_path.read_text())
    base_summary = json.loads(base_summary_path.read_text())
    manifest = read_csv(view / "dataset/manifest.csv")
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    data_csv_hashes: dict[str, str] = {
        "dataset/manifest.csv": sha256(view / "dataset/manifest.csv"),
        "dataset/object_boxes.csv": sha256(view / "dataset/object_boxes.csv"),
    }
    csv_hashes_ok = (
        data_csv_hashes["dataset/manifest.csv"] == camera_summary["dataset"]["manifest_sha256"]
        and data_csv_hashes["dataset/object_boxes.csv"] == camera_summary["dataset"]["object_boxes_sha256"]
    )
    for visibility in ("v010", "v025"):
        for split in ("train", "val"):
            summary = camera_summary["summaries"][visibility][split]
            base_one = base_summary["summaries"][visibility][split]
            for filename, key in (
                ("object_boxes.csv", "derived_object_boxes_sha256"),
                ("object_ignore_regions.csv", "derived_object_ignore_regions_sha256"),
                ("target_manifest.csv", "derived_target_manifest_sha256"),
            ):
                relative = f"contracts/{visibility}/{split}/{filename}"
                actual = sha256(camera_contract / relative)
                data_csv_hashes[relative] = actual
                csv_hashes_ok = csv_hashes_ok and actual == summary[key]
            for filename, key in (
                ("object_boxes.csv", "object_boxes_sha256"),
                ("object_ignore_regions.csv", "object_ignore_regions_sha256"),
                ("target_manifest.csv", "target_manifest_sha256"),
            ):
                relative = f"base/{visibility}/{split}/{filename}"
                actual = sha256(base_contract / "contracts" / visibility / split / filename)
                data_csv_hashes[relative] = actual
                csv_hashes_ok = csv_hashes_ok and actual == base_one[key]
    payloads = verify_payload_hashes(camera_contract / "contracts")
    data_ok = (
        sha256(view_summary_path) == training["view_contract"]["summary_sha256"]
        and sha256(camera_summary_path) == training["view_contract"]["camera_plane_summary_sha256"]
        and sha256(base_summary_path) == camera_summary["source_provenance"]["source_summary_sha256"]
        and len(train_rows) == 16827 and len(val_rows) == 3345
        and {row["split"] for row in manifest} == {"train", "val"}
        and all(view_summary["verification"]["checks"].values())
        and all(camera_summary["hard_gates"].values())
        and csv_hashes_ok and payloads["pass"]
    )
    checks.append({
        "name": "frozen_view_gt_ignore_camera_plane_hashes", "pass": data_ok,
        "train_frames": len(train_rows), "validation_frames": len(val_rows),
        "csv_hashes": data_csv_hashes, "payload_verification": payloads,
        "summary_hashes": {
            "view": sha256(view_summary_path), "camera_plane": sha256(camera_summary_path),
            "base_gt": sha256(base_summary_path),
        },
        "independent_ignore_cache_roots": {
            "v010": str((camera_contract / "contracts/v010").resolve()),
            "v025": str((camera_contract / "contracts/v025").resolve()),
            "distinct": (camera_contract / "contracts/v010").resolve() != (camera_contract / "contracts/v025").resolve(),
        },
    })

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {"model", "optimizer", "scheduler", "grad_scaler", "rng_states", "epoch"}
    scheduler = checkpoint.get("scheduler", {})
    optimizer = checkpoint.get("optimizer", {})
    rng = checkpoint.get("rng_states", {})
    expected_end_lr = expected_lr(
        training, contract["resume_epoch"], contract["resume_steps_per_epoch"] - 1,
        contract["resume_steps_per_epoch"],
    )
    groups = {group.get("name"): group for group in optimizer.get("param_groups", [])}
    state_ok = (
        required_keys.issubset(checkpoint)
        and checkpoint["epoch"] == contract["resume_epoch"]
        and scheduler.get("schema") == "registered_h2_j2_warmup_cosine_v2"
        and scheduler.get("steps_per_epoch") == contract["resume_steps_per_epoch"]
        and scheduler.get("optimizer_steps") == contract["resume_optimizer_steps"]
        and scheduler.get("stage_j2") == training["stage_j2"]
        and set(groups) == {"inherited", "object"}
        and math.isclose(groups["inherited"]["lr"], expected_end_lr["inherited"], rel_tol=0, abs_tol=1e-16)
        and math.isclose(groups["object"]["lr"], expected_end_lr["object"], rel_tol=0, abs_tol=1e-16)
        and scheduler.get("last_lr") == expected_end_lr
        and bool(checkpoint["grad_scaler"])
        and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        and len(rng["torch_cuda"]) == 1
        and checkpoint.get("resolved_config") == training
        and checkpoint.get("training_view_hashes", {}).get("manifest") == data_csv_hashes["dataset/manifest.csv"]
        and checkpoint.get("training_view_hashes", {}).get("object_boxes") == data_csv_hashes["dataset/object_boxes.csv"]
    )
    checks.append({
        "name": "exact_epoch10_resume_payload", "pass": state_ok,
        "required_keys": sorted(required_keys), "present_keys": sorted(checkpoint),
        "epoch": checkpoint.get("epoch"), "optimizer_steps": scheduler.get("optimizer_steps"),
        "steps_per_epoch": scheduler.get("steps_per_epoch"),
        "optimizer_group_lrs": {name: group.get("lr") for name, group in groups.items()},
        "expected_end_lr": expected_end_lr,
        "optimizer_state_entries": len(optimizer.get("state", {})),
        "grad_scaler": checkpoint.get("grad_scaler"), "rng_keys": sorted(rng),
    })

    source_hashes = {
        relative: sha256(NATIVE_PACKAGE / relative)
        for relative in {**training["native_sources"], **training["native_config_sources"]}
    }
    source_ok = (
        source_hashes == {**training["native_sources"], **training["native_config_sources"]}
        and source_hashes == checkpoint.get("source_hashes")
    )
    checks.append({
        "name": "frozen_model_loss_decoder_sources", "pass": source_ok,
        "source_hashes": source_hashes,
    })

    cuda_detail: dict[str, Any] = {}
    cuda_ok = False
    try:
        if sys.executable != "/usr/bin/python3":
            raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda")
        x = torch.randn(2, 4, 16, 16, device=device)
        conv = torch.nn.Conv2d(4, 8, 3, padding=1).to(device)
        y = conv(x)
        if not bool(torch.isfinite(y).all().item()):
            raise RuntimeError("minimal CUDA convolution produced nonfinite output")
        native_config_path = NATIVE_PACKAGE / "configs/route_b_v3_1_native_grid_v1.yaml"
        from pole_lraspp_multimodal_fusion.common import load_config
        native_config = load_config(native_config_path)
        object_cfg = native_config["object_heads"]
        model = native.build_native_grid_model(
            num_classes=int(native_config["training"].get("num_classes", 3)),
            radar_channels=int(native_config["fusion"]["radar_channels"]),
            hidden_channels=int(object_cfg.get("hidden_channels", 128)),
            head_depth=int(object_cfg.get("head_depth", 3)), device=device,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optim = torch.optim.AdamW([
            {"params": native.inherited_parameters(model), "lr": 0.0, "name": "inherited"},
            {"params": native.object_parameters(model), "lr": 0.0, "name": "object"},
        ], lr=0.0, weight_decay=float(training["weight_decay"]))
        optim.load_state_dict(checkpoint["optimizer"])
        scaler = torch.amp.GradScaler("cuda", enabled=bool(training["amp"]))
        scaler.load_state_dict(checkpoint["grad_scaler"])
        cuda_detail = {
            "interpreter": sys.executable, "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "architecture_list": torch.cuda.get_arch_list(),
            "convolution_shape": list(y.shape), "convolution_finite": True,
            "model_state_strict": True, "optimizer_state_loaded": True,
            "grad_scaler_state_loaded": True,
        }
        cuda_ok = (
            torch.cuda.get_device_capability() == (12, 0)
            and "sm_120" in torch.cuda.get_arch_list()
        )
    except Exception as exc:
        cuda_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append({"name": "sm120_cuda_conv_and_state_load", "pass": cuda_ok, "detail": cuda_detail})

    test_absent = (
        {row["split"] for row in manifest} == {"train", "val"}
        and view_summary["test"]["present"] is False
        and view_summary["test"]["rows"] == 0
        and view_summary["test"]["payload_references"] == 0
        and view_summary["verification"]["checks"]["no_locked_test_token_or_path"]
        and camera_summary["hard_gates"]["test_rows_and_payloads_absent"]
    )
    checks.append({
        "name": "locked_test_absent_without_access", "pass": test_absent,
        "manifest_splits": sorted({row["split"] for row in manifest}),
        "locked_directories_listed_or_read": 0,
    })

    result = {
        "schema": "route_b_v3_1_native_grid_expanded_continuation_preflight_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "all_pass": all(check["pass"] for check in checks),
        "checks": checks,
        "checkpoint_sha256": immutable_hashes["resume_checkpoint"],
        "resume_state": checks[4],
        "training_view_hashes": checks[3]["summary_hashes"],
        "source_hashes": source_hashes,
        "wall_seconds": time.monotonic() - started,
    }
    output = args.output.resolve() if args.output is not None else experiment / "PREFLIGHT.json"
    write_json_x(output, result)
    if result["all_pass"] and args.output is None:
        with (experiment / "PREFLIGHT_COMPLETE").open("x", encoding="utf-8") as stream:
            stream.write("PASS\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["all_pass"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
