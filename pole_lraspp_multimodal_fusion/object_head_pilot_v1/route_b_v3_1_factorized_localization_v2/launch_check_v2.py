#!/usr/bin/env python3
"""Run exactly the eight preregistered factorized-localization launch checks."""

from __future__ import annotations

import argparse
import hashlib
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
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pole_lraspp_multimodal_fusion.common import read_manifest, set_reproducible_seeds  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from losses_v2 import factorized_localization_loss  # noqa: E402
from model_v2 import (  # noqa: E402
    build_factorized_model, freeze_for_localization, load_native_warm_start,
    localization_parameters, parameter_report, split_boundary_report,
)
from targets_v2 import FactorizedLocalizationDataset, synthetic_projection_case  # noqa: E402


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


def grad_report(module: torch.nn.Module) -> dict[str, Any]:
    values = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    finite = all(value is not None and bool(torch.isfinite(value).all()) for value in values)
    norm = sum(float(value.detach().float().abs().sum().item()) for value in values if value is not None)
    return {"parameter_tensors": len(values), "all_finite": finite, "absolute_gradient_sum": norm,
            "nonzero": norm > 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--selection-contract", required=True, type=Path)
    parser.add_argument("--contract-experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    contract_experiment = args.contract_experiment.resolve()
    started = time.monotonic()
    checks: list[dict[str, Any]] = []

    # 1. py_compile on every new source file, once.
    source_files = sorted(PACKAGE_ROOT.glob("*.py"))
    compile_run = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, source_files)],
        text=True, capture_output=True, check=False,
    )
    checks.append({"number": 1, "name": "py_compile_new_files", "pass": compile_run.returncode == 0,
                   "file_count": len(source_files), "stderr": compile_run.stderr[-2000:]})

    # 2. Parse the create-only resolved copies, and assert the fixed experiment recipe.
    try:
        training = json.loads(args.training_config.read_text(encoding="utf-8"))
        selection = json.loads(args.selection_contract.read_text(encoding="utf-8"))
        resolved_ok = (
            training["schema"] == "route_b_v3_1_factorized_localization_training_v2"
            and training["epochs"] == 12 and training["checkpoint_epochs"] == [4, 8, 12]
            and training["batch_size"] == 16 and training["q"] == 0 and training["ae"] is False
            and selection["evaluated_epochs"] == [4, 8, 12]
            and selection["status"] == "REGISTERED BEFORE FIRST CANDIDATE EVALUATION"
        )
        config_error = None
    except Exception as exc:  # pragma: no cover - recorded fail-closed
        training, selection, resolved_ok = {}, {}, False
        config_error = f"{type(exc).__name__}: {exc}"
    checks.append({"number": 2, "name": "parse_resolved_configs", "pass": resolved_ok,
                   "error": config_error})

    # 3. Verify the retained epoch-15 native-grid warm start.
    checkpoint = ROOT / str(training.get("warm_start_checkpoint", "__missing__"))
    expected_sha = str(training.get("warm_start_sha256", ""))
    actual_sha = sha256(checkpoint) if checkpoint.is_file() else None
    warm_ok = actual_sha == expected_sha == "1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed"
    checks.append({"number": 3, "name": "verify_warm_start_sha256", "pass": warm_ok,
                   "checkpoint": str(checkpoint), "expected_sha256": expected_sha,
                   "actual_sha256": actual_sha})

    # 4. Verify the completed Phase-A contract, including every hard gate.
    contract_path = contract_experiment / "CAMERA_PLANE_CONTRACT_SUMMARY.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        hard_gates = dict(contract["hard_gates"])
        contract_ok = len(hard_gates) == 9 and all(hard_gates.values())
    except Exception as exc:  # pragma: no cover - recorded fail-closed
        contract, hard_gates, contract_ok = {}, {}, False
        contract_error = f"{type(exc).__name__}: {exc}"
    else:
        contract_error = None
    checks.append({"number": 4, "name": "verify_phase_a_contract_gates", "pass": contract_ok,
                   "hard_gates": hard_gates, "error": contract_error})

    # 5. Positive-depth camera projection/unprojection round-trip.
    try:
        synthetic = synthetic_projection_case()
        synthetic_ok = bool(synthetic["pass"])
    except Exception as exc:  # pragma: no cover - recorded fail-closed
        synthetic, synthetic_ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    checks.append({"number": 5, "name": "synthetic_positive_depth_projection", "pass": synthetic_ok,
                   "detail": synthetic})

    model = None
    tensors = None
    batch_detail: dict[str, Any] = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        set_reproducible_seeds(int(training["training_seed"]))
        device = torch.device("cuda")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        object_cfg = dict(payload["config"]["object_heads"])
        dataset_dir = contract_experiment / "dataset"
        rows = read_manifest(dataset_dir / "manifest.csv")
        train_rows = [row for row in rows if row.get("split") == "train"]
        object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
        dataset = FactorizedLocalizationDataset(
            dataset_dir, train_rows, object_rows, tuple(training["input_size"]), object_cfg,
            augment_strength="off", geometric_augment=False,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        tensors, _masks, targets = next(iter(loader))
        tensors = tensors.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}
        model = build_factorized_model(
            num_classes=int(payload["config"]["training"].get("num_classes", 3)),
            radar_channels=int(payload["radar_channels"]),
            hidden_channels=int(payload["object_hidden_channels"]),
            head_depth=int(payload["object_head_depth"]),
            localization_hidden=int(training["localization_hidden_channels"]), device=device,
        )
        mapping = load_native_warm_start(model, checkpoint, device=device)
        freeze_for_localization(model)
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=256.0)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True, cache_enabled=False):
            outputs = model.localization_training_outputs(tensors)
        with torch.autocast(device_type="cuda", enabled=False):
            loss, loss_parts = factorized_localization_loss(
                outputs["localization"].float(), outputs["object"].float(), targets,
                training["losses"],
            )
        scaler.scale(loss).backward()
        scaler.unscale_(torch.optim.AdamW(localization_parameters(model), lr=0.0))
        component_grads = {
            "localization_trunk": grad_report(model.localization_trunk),
            "log_depth_head": grad_report(model.log_depth_head),
            "projected_3d_center_offset_head": grad_report(model.projected_3d_center_offset_head),
        }
        trainable_ids = {id(parameter) for parameter in localization_parameters(model)}
        frozen_grad_tensors = [
            parameter.grad for parameter in model.parameters()
            if id(parameter) not in trainable_ids and parameter.grad is not None
        ]
        frozen_grad_abs_sum = sum(float(value.detach().float().abs().sum().item())
                                  for value in frozen_grad_tensors)
        batch_ok = (
            math.isfinite(float(loss.detach().item()))
            and all(item["all_finite"] and item["nonzero"] for item in component_grads.values())
            and frozen_grad_abs_sum == 0.0
            and all(not parameter.requires_grad for parameter in model.parameters()
                    if id(parameter) not in trainable_ids)
        )
        batch_detail = {
            "batch_size": int(tensors.shape[0]), "q": 0, "amp": True,
            "autocast_cache_enabled": False, "loss": float(loss.detach().item()),
            "loss_parts": loss_parts, "component_gradients": component_grads,
            "frozen_gradient_tensor_count": len(frozen_grad_tensors),
            "frozen_gradient_absolute_sum": frozen_grad_abs_sum,
            "warm_start_mapping": mapping, "parameters": parameter_report(model),
        }
    except Exception as exc:  # pragma: no cover - recorded fail-closed
        batch_ok = False
        batch_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append({"number": 6, "name": "one_real_v3_1_q0_amp_batch", "pass": batch_ok,
                   "detail": batch_detail})

    # 7. The tail remains a pure function of the unchanged low/high split bundle.
    try:
        if model is None or tensors is None:
            raise RuntimeError("real-batch model unavailable")
        split = split_boundary_report(model, tensors[:1])
        split_ok = (split["tail_reads_only_low_high"] and not split["tail_raw_modality_side_channels"]
                    and split["outputs_bit_identical"])
    except Exception as exc:  # pragma: no cover - recorded fail-closed
        split, split_ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    checks.append({"number": 7, "name": "unchanged_low_high_split_bundle", "pass": split_ok,
                   "detail": split})

    # 8. Retained prediction provenance and the amended re-score establish decoder parity.
    amended_path = contract_experiment / "AMENDED_BASELINE.json"
    try:
        amended = json.loads(amended_path.read_text(encoding="utf-8"))
        retained = Path(amended["retained_predictions"])
        manifest_path = retained.parent / "inference_manifest.json"
        retained_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parity_ok = (
            sha256(retained) == amended["retained_detections_sha256"]
            == retained_manifest["detections_sha256"]
            and amended["new_inference_passes"] == 0
            and all(amended["explanation_gates"].values())
            and amended["checkpoint_sha256"] == actual_sha
        )
        parity = {
            "retained_predictions": str(retained),
            "retained_detections_sha256": sha256(retained),
            "new_inference_passes": amended["new_inference_passes"],
            "amended_v010_flat": amended["amended"]["v010"]["flat"],
            "explanation_gates": amended["explanation_gates"],
        }
    except Exception as exc:  # pragma: no cover - recorded fail-closed
        parity, parity_ok = {"error": f"{type(exc).__name__}: {exc}"}, False
    checks.append({"number": 8, "name": "initial_legacy_decoder_retained_prediction_parity",
                   "pass": parity_ok, "detail": parity})

    result = {
        "schema": "route_b_v3_1_factorized_localization_launch_checks_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "check_count": len(checks), "checks": checks,
        "all_pass": len(checks) == 8 and all(item["pass"] for item in checks),
        "contract_valid": bool(contract_ok), "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "LAUNCH_CHECKS.json", result)
    if result["all_pass"]:
        (experiment / "LAUNCH_CHECKS_COMPLETE").write_text("EIGHT_LAUNCH_CHECKS_PASS\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["all_pass"] else (2 if not contract_ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
