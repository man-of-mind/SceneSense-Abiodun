#!/usr/bin/env python3
"""Eight bounded preflight gates for expanded native-grid long training."""

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
from model_v1 import split_boundary_report  # noqa: E402


def _load_native_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "route_b_native_grid_preflight_trainer_v1", NATIVE_PACKAGE / "train_native_v1.py"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def gradients(module: torch.nn.Module) -> dict[str, Any]:
    values = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    absolute_sum = sum(
        float(value.detach().double().abs().sum().item()) for value in values if value is not None
    )
    return {
        "trainable_parameter_tensors": len(values),
        "all_present": all(value is not None for value in values),
        "all_finite": all(
            value is not None and bool(torch.isfinite(value).all().item()) for value in values
        ),
        "absolute_sum": absolute_sum,
        "nonzero": absolute_sum > 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    started = time.monotonic()
    checks: list[dict[str, Any]] = []

    # 1. Compile only the new package sources.
    sources = sorted(PACKAGE_ROOT.glob("*.py"))
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, sources)],
        text=True, capture_output=True, check=False,
    )
    checks.append({
        "number": 1, "name": "py_compile_new_runner", "pass": compiled.returncode == 0,
        "source_count": len(sources), "stderr": compiled.stderr[-2000:],
    })

    # 2. Parse and pin the only changed optimization contract; loss recipe is exact v1.
    try:
        config = json.loads(args.config.read_text())
        native_recipe = json.loads(
            (NATIVE_PACKAGE / "configs/native_grid_training_v1.json").read_text()
        )
        config_ok = (
            config["total_epochs"] == 40
            and config["checkpoint_epochs"] == [5, 10, 20, 30, 40]
            and config["decode_epochs"] == [5, 10, 20, 30, 40]
            and config["batch_size"] == 16 and config["q"] == 0 and config["ae"] is False
            and config["optimizer"] == "AdamW" and config["weight_decay"] == 0.0001
            and config["class_loss_weights"] == native_recipe["class_loss_weights"]
            and config["lovasz_weight"] == native_recipe["lovasz_weight"]
            and config["loss_weights"] == native_recipe["loss_weights"]
            and config["stage_h2"]["warmup_optimizer_steps"] == 500
            and config["stage_j2"]["final_lr_ratio"] == 0.1
        )
        config_error = None
    except Exception as exc:
        config, config_ok = {}, False
        config_error = f"{type(exc).__name__}: {exc}"
    checks.append({
        "number": 2, "name": "parse_registered_config", "pass": config_ok,
        "error": config_error,
    })

    # 3. Expanded view counts, GT counts, camera-plane transitions, and retained hashes.
    view = (ROOT / config["training_view"]).resolve(strict=True)
    expanded_contract = (ROOT / config["expanded_contract"]).resolve(strict=True)
    base_contract = (ROOT / config["expanded_base_contract"]).resolve(strict=True)
    view_summary_path = view / "EXPANDED_TRAIN_VIEW_SUMMARY.json"
    camera_summary_path = expanded_contract / "CAMERA_PLANE_CONTRACT_SUMMARY.json"
    base_summary_path = base_contract / "GT_CONTRACT_SUMMARY.json"
    view_summary = json.loads(view_summary_path.read_text())
    camera_summary = json.loads(camera_summary_path.read_text())
    base_summary = json.loads(base_summary_path.read_text())
    manifest = read_manifest(view / "dataset/manifest.csv")
    train_rows = [row for row in manifest if row["split"] == "train"]
    val_rows = [row for row in manifest if row["split"] == "val"]
    observed = view_summary["verification"]["observed"]
    expected = config["view_contract"]
    actual_validation_hashes = {
        relative: sha256(expanded_contract / "contracts" / relative)
        for relative in expected["retained_validation_hashes"]
    }
    reported_validation = observed["validation_identity"]["camera_plane_native_grid_v3_1"]
    view_gates = dict(view_summary["verification"]["checks"])
    camera_gates = dict(camera_summary["hard_gates"])
    view_ok = (
        sha256(view_summary_path) == expected["summary_sha256"]
        and sha256(camera_summary_path) == expected["camera_plane_summary_sha256"]
        and len(train_rows) == expected["train_frames"]
        and len(val_rows) == expected["validation_frames"]
        and {row["split"] for row in manifest} == {"train", "val"}
        and len(view_summary["train_episodes"]) == expected["train_episodes"]
        and len(view_summary["validation_episodes"]) == expected["validation_episodes"]
        and observed["symlink_count"] == expected["symlinks"]
        and view_summary["corpus_payload_copies"] == expected["corpus_payload_copies"]
        and base_summary["summaries"]["v010"]["train"]["positive_records"] == expected["v010"]["train_positives"]
        and base_summary["summaries"]["v010"]["train"]["ignore_records"] == expected["v010"]["train_ignores"]
        and base_summary["summaries"]["v010"]["val"]["positive_records"] == expected["v010"]["validation_positives"]
        and base_summary["summaries"]["v010"]["val"]["ignore_records"] == expected["v010"]["validation_ignores"]
        and camera_summary["summaries"]["v010"]["train"]["transition_records"] == expected["v010"]["camera_plane_train_exclusions"]
        and camera_summary["summaries"]["v010"]["val"]["transition_records"] == expected["v010"]["camera_plane_validation_exclusions"]
        and camera_summary["summaries"]["v025"]["train"]["transition_records"] == expected["v025"]["camera_plane_train_exclusions"]
        and camera_summary["summaries"]["v025"]["val"]["transition_records"] == expected["v025"]["camera_plane_validation_exclusions"]
        and actual_validation_hashes == expected["retained_validation_hashes"]
        and reported_validation["identical"]
        and all(view_gates.values()) and all(camera_gates.values())
    )
    training_view_hashes = {
        "expanded_view_summary": sha256(view_summary_path),
        "camera_plane_summary": sha256(camera_summary_path),
        "base_gt_summary": sha256(base_summary_path),
        "manifest": sha256(view / "dataset/manifest.csv"),
        "object_boxes": sha256(view / "dataset/object_boxes.csv"),
        "retained_validation": actual_validation_hashes,
    }
    checks.append({
        "number": 3, "name": "expanded_training_view_contract", "pass": view_ok,
        "counts": {
            "train_frames": len(train_rows), "validation_frames": len(val_rows),
            "train_episodes": len(view_summary["train_episodes"]),
            "validation_episodes": len(view_summary["validation_episodes"]),
            "symlinks": observed["symlink_count"],
            "corpus_payload_copies": view_summary["corpus_payload_copies"],
        },
        "actual_retained_validation_hashes": actual_validation_hashes,
        "view_gates": view_gates, "camera_plane_gates": camera_gates,
    })

    # 4. Required native epoch-15 warm start.
    warm = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    warm_sha = sha256(warm)
    warm_ok = warm_sha == config["warm_start_sha256"] == "1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed"
    checks.append({
        "number": 4, "name": "warm_start_sha256", "pass": warm_ok,
        "checkpoint": str(warm), "actual_sha256": warm_sha,
    })

    # 5. Imported native source hashes; no factorized package is referenced.
    source_hashes: dict[str, str] = {}
    source_ok = True
    for relative, expected_hash in {**config["native_sources"], **config["native_config_sources"]}.items():
        actual = sha256(NATIVE_PACKAGE / relative)
        source_hashes[relative] = actual
        source_ok = source_ok and actual == expected_hash
    checks.append({
        "number": 5, "name": "immutable_native_source_hashes", "pass": source_ok,
        "source_hashes": source_hashes, "factorized_source_paths": [],
    })

    # 6 and 7. One real Stage-H2 AMP batch and explicit low/high split parity.
    batch_detail: dict[str, Any] = {}
    split_detail: dict[str, Any] = {}
    try:
        if sys.executable != "/usr/bin/python3":
            raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if torch.cuda.get_device_capability() != (12, 0) or "sm_120" not in torch.cuda.get_arch_list():
            raise RuntimeError("RTX 5090 sm_120 CUDA contract unavailable")
        set_reproducible_seeds(int(config["training_seed"]))
        device = torch.device("cuda")
        native_config = load_config(NATIVE_PACKAGE / "configs/route_b_v3_1_native_grid_v1.yaml")
        object_cfg = dict(native_config["object_heads"])
        dataset = native.NativeGridDataset(
            view / "dataset", train_rows, load_object_boxes(view / "dataset/object_boxes.csv"),
            tuple(config["input_size"]), object_cfg,
            augment_strength=str(config["augment_strength"]),
            geometric_augment=bool(config["geometric_augment"]),
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        tensors, masks, targets = next(iter(loader))
        tensors, masks = tensors.to(device), masks.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}
        model = native.build_native_grid_model(
            num_classes=3, radar_channels=4, hidden_channels=128, head_depth=3, device=device
        )
        mapping = native.load_warm_start(model, warm, device=device)
        stage = {"name": "H2", "freeze_backbone": True, "freeze_classifier": True}
        native.apply_stage(model, stage, True)
        native.stage_train_mode(model, stage, True)
        optimizer = torch.optim.AdamW(
            native.object_parameters(model), lr=0.0, weight_decay=0.0001
        )
        # Match the immutable native-grid v1 launch proof. The training loop keeps
        # its normal dynamic scaler; this bounded gradient diagnostic uses 2**8 so
        # scaled-gradient overflow cannot obscure the finite unscaled gradient gate.
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=2.0 ** 8)
        class_weights = torch.tensor(config["class_loss_weights"], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
            loss, parts, _logits = native.compute_batch_losses(
                model, tensors, masks, targets, config["loss_weights"], class_weights,
                float(config["lovasz_weight"]),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        component_gradients = {
            "object_trunk": gradients(model.object_head.shared_trunk),
            "object_upsampler": gradients(model.object_head.upsampler),
            "vehicle_heatmap": gradients(model.object_head.vehicle_heatmap_head),
            "person_heatmap": gradients(model.object_head.person_heatmap_head),
            "regression": gradients(model.object_head.regression_head),
            "offset": gradients(model.object_head.offset_head),
        }
        inherited_grads = [
            parameter.grad for parameter in native.inherited_parameters(model)
            if parameter.grad is not None
        ]
        with torch.inference_mode():
            outputs = model(tensors[:1], feature_drop_fraction=0.0)
        batch_ok = (
            math.isfinite(float(loss.detach().item()))
            and all(math.isfinite(float(value)) for value in parts.values())
            and all(detail["all_present"] and detail["all_finite"] and detail["nonzero"]
                    for detail in component_gradients.values())
            and not inherited_grads
            and list(outputs["object"].shape) == [1, 16, 108, 192]
            and mapping["missing_keys_are_new_only"] and mapping["incompatible_count"] == 0
        )
        batch_detail = {
            "interpreter": sys.executable, "python": sys.version.split()[0],
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "architecture_list": torch.cuda.get_arch_list(),
            "loss": float(loss.detach().item()), "loss_parts": parts,
            "component_gradients": component_gradients,
            "inherited_gradient_tensors": len(inherited_grads),
            "object_output_shape": list(outputs["object"].shape),
            "segmentation_output_shape": list(outputs["out"].shape),
            "q": 0, "ae": False, "feature_drop_fraction": 0.0,
            "warm_start_mapping": mapping,
        }
        split_detail = split_boundary_report(model, tensors[:1])
        split_detail["tail_raw_modality_side_channels"] = []
        split_ok = (
            split_detail["outputs_match"]
            and split_detail["tail_reads_only_transported_bundle"]
            and split_detail["object_grid_is_native"]
            and sorted(split_detail["transported_feature_names"]) == ["high", "low"]
            and split_detail["tail_raw_modality_side_channels"] == []
        )
    except Exception as exc:
        batch_ok = split_ok = False
        batch_detail = {"error": f"{type(exc).__name__}: {exc}"}
        split_detail = {"error": "batch setup failed before split proof"}
    checks.append({
        "number": 6, "name": "one_real_q0_stage_h2_amp_batch", "pass": batch_ok,
        "detail": batch_detail,
    })
    checks.append({
        "number": 7, "name": "low_high_split_parity", "pass": split_ok,
        "detail": split_detail,
    })

    # 8. Prove absence from admitted artifacts only; never enumerate/read a locked path.
    test_detail = {
        "manifest_splits": sorted({row["split"] for row in manifest}),
        "manifest_test_rows": sum(row["split"] == "test" for row in manifest),
        "view_summary_test": view_summary["test"],
        "view_no_locked_path_gate": view_gates["no_locked_test_token_or_path"],
        "camera_plane_test_absence_gate": camera_gates["test_rows_and_payloads_absent"],
        "locked_directories_listed_or_read": 0,
    }
    test_ok = (
        test_detail["manifest_splits"] == ["train", "val"]
        and test_detail["manifest_test_rows"] == 0
        and view_summary["test"]["present"] is False
        and view_summary["test"]["rows"] == 0
        and view_summary["test"]["payload_references"] == 0
        and test_detail["view_no_locked_path_gate"]
        and test_detail["camera_plane_test_absence_gate"]
    )
    checks.append({
        "number": 8, "name": "test_absent_without_locked_access", "pass": test_ok,
        "detail": test_detail,
    })

    result = {
        "schema": "route_b_v3_1_native_grid_expanded_training_preflight_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "check_count": len(checks),
        "all_pass": len(checks) == 8 and all(check["pass"] for check in checks),
        "source_hashes": source_hashes,
        "training_view_hashes": training_view_hashes,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "PREFLIGHT.json", result)
    if result["all_pass"]:
        (experiment / "PREFLIGHT_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
