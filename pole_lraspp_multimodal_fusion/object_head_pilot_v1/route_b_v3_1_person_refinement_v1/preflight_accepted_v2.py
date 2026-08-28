#!/usr/bin/env python3
"""Fail-closed preflight for the accepted recovered epoch-40 continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parents[2]
RECOVERY_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_expanded_training_v2"
NATIVE_PACKAGE = PACKAGE_ROOT.parent / "route_b_v3_1_native_grid_v1"
FUSION_ROOT = ROOT / "pole_lraspp_multimodal_fusion"
for path in (str(PACKAGE_ROOT), str(NATIVE_PACKAGE), str(FUSION_ROOT), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_recovery_preflight() -> Any:
    path = RECOVERY_PACKAGE / "preflight_continuation_v3.py"
    spec = importlib.util.spec_from_file_location("accepted_recovery_preflight_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError("unable to load frozen recovery preflight helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recovery_preflight = load_recovery_preflight()


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--person-config", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve()
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    config = json.loads(args.person_config.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    checks.append({
        "name": "required_local_master_start", "pass": (
            branch == "master" and head == acceptance["required_starting_head"]
        ), "branch": branch, "head": head,
    })

    historical = (ROOT / acceptance["historical_experiment"]).resolve(strict=True)
    terminal_path = historical / "TERMINAL_VERDICT.txt"
    sentinel_path = historical / "COMPLETION_SENTINEL"
    historical_ok = (
        terminal_path.read_text().strip() == acceptance["historical_terminal"]
        and sentinel_path.read_text().strip() == acceptance["historical_terminal"]
        and sha256(terminal_path) == acceptance["historical_terminal_sha256"]
        and sha256(sentinel_path) == acceptance["historical_terminal_sha256"]
    )
    checks.append({
        "name": "historical_failure_terminal_preserved", "pass": historical_ok,
        "terminal": terminal_path.read_text().strip(),
        "terminal_sha256": sha256(terminal_path), "sentinel_sha256": sha256(sentinel_path),
    })

    accepted_paths = {
        "checkpoint": (ROOT / acceptance["recovered_checkpoint"]).resolve(strict=True),
        "primary_record": (ROOT / acceptance["recovered_primary_record"]).resolve(strict=True),
        "inference_manifest": (ROOT / acceptance["recovered_prediction_root"] / "inference_manifest.json").resolve(strict=True),
        "detections": (ROOT / acceptance["recovered_prediction_root"] / "detections.csv").resolve(strict=True),
        "segmentation_manifest": (ROOT / acceptance["recovered_prediction_root"] / "segmentation_manifest.csv").resolve(strict=True),
        "historical_reconciliation": (ROOT / acceptance["historical_reconciliation"]).resolve(strict=True),
    }
    expected_hashes = {
        "checkpoint": acceptance["recovered_checkpoint_sha256"],
        "primary_record": acceptance["recovered_primary_record_sha256"],
        "inference_manifest": acceptance["recovered_inference_manifest_sha256"],
        "detections": acceptance["recovered_detections_sha256"],
        "segmentation_manifest": acceptance["recovered_segmentation_manifest_sha256"],
        "historical_reconciliation": acceptance["historical_reconciliation_sha256"],
    }
    actual_hashes = {name: sha256(path) for name, path in accepted_paths.items()}
    checkpoint = torch.load(accepted_paths["checkpoint"], map_location="cpu", weights_only=False)
    inference = json.loads(accepted_paths["inference_manifest"].read_text())
    primary = json.loads(accepted_paths["primary_record"].read_text())
    reconciliation = json.loads(accepted_paths["historical_reconciliation"].read_text())
    required_state = {"model", "optimizer", "scheduler", "grad_scaler", "rng_states", "epoch"}
    acceptance_ok = (
        acceptance["decision"] == "RECOVERED_EPOCH40_ACCEPTED_WITH_FAVORABLE_LOW_THRESHOLD_VARIATION"
        and actual_hashes == expected_hashes
        and int(checkpoint["epoch"]) == 40 and required_state.issubset(checkpoint)
        and inference["checkpoint_sha256"] == expected_hashes["checkpoint"]
        and primary["checkpoint_sha256"] == expected_hashes["checkpoint"]
        and inference["validation_frames"] == 3345 and inference["inference_pass_count"] == 1
        and reconciliation["all_pass"] is False
        and sum(bool(value) for value in reconciliation["gates"].values()) == 13
        and reconciliation["gates"]["person_recall_002"] is False
        and acceptance["accepted_variation"]["delta"] > 0.0
        and acceptance["accepted_variation"]["additional_true_positives"] == 14
        and acceptance["repeat_epochs_11_through_40"] is False
    )
    checks.append({
        "name": "accepted_epoch40_checkpoint_predictions_and_decision", "pass": acceptance_ok,
        "actual_hashes": actual_hashes, "expected_hashes": expected_hashes,
        "checkpoint_epoch": checkpoint.get("epoch"), "required_state_present": required_state.issubset(checkpoint),
        "accepted_variation": acceptance["accepted_variation"],
        "comparison_baseline": acceptance["comparison_baseline"],
    })

    rows = read_csv(experiment / "dataset/manifest.csv")
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    test_tokens = [row["sample_id"] for row in rows
                   if "canonical_v3_07" in row["sample_id"] or "canonical_v3_08" in row["sample_id"]]
    historical_preflight = json.loads((historical / "PREFLIGHT.json").read_text())
    historical_data = next(check for check in historical_preflight["checks"]
                           if check["name"] == "frozen_view_gt_ignore_camera_plane_hashes")
    current_csv_hashes: dict[str, str] = {}
    for relative in historical_data["csv_hashes"]:
        if relative.startswith("dataset/"):
            path = experiment / relative
        elif relative.startswith("contracts/"):
            path = experiment / relative
        elif relative.startswith("base/"):
            training = json.loads((ROOT / config["registered_training_config"]).read_text())
            base_contract = (ROOT / training["expanded_base_contract"] / "contracts").resolve(strict=True)
            path = base_contract / relative[len("base/"):]
        else:
            raise RuntimeError(f"unknown frozen CSV provenance path {relative}")
        current_csv_hashes[relative] = sha256(path)
    camera_contract = (ROOT / json.loads((ROOT / config["registered_training_config"]).read_text())["expanded_contract"]).resolve(strict=True)
    payloads = recovery_preflight.verify_payload_hashes(camera_contract / "contracts")
    data_ok = (
        len(train_rows) == 16827 and len(val_rows) == 3345
        and {row["split"] for row in rows} == {"train", "val"} and not test_tokens
        and current_csv_hashes == historical_data["csv_hashes"]
        and payloads["pass"]
        and historical_preflight["all_pass"]
    )
    checks.append({
        "name": "frozen_v31_train_validation_camera_contracts", "pass": data_ok,
        "train_frames": len(train_rows), "validation_frames": len(val_rows), "test_frames": 0,
        "locked_test_token_matches": len(test_tokens), "locked_test_paths_read": 0,
        "current_csv_hashes": current_csv_hashes,
        "payload_references_checked": payloads["references_checked"],
        "unique_payloads_hashed": payloads["unique_payloads_hashed"],
        "payload_failures": payloads["failures"],
        "v010_v025_contract_roots_distinct": (
            (experiment / "contracts/v010").resolve() != (experiment / "contracts/v025").resolve()
        ),
    })

    source_paths = sorted(PACKAGE_ROOT.glob("*.py"))
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, source_paths)],
        capture_output=True, text=True, check=False,
    )
    checks.append({
        "name": "continuation_sources_compile", "pass": compile_result.returncode == 0,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "stderr": compile_result.stderr[-2000:],
    })

    cuda_ok = False
    cuda_detail: dict[str, Any]
    try:
        if sys.executable != "/usr/bin/python3" or not torch.cuda.is_available():
            raise RuntimeError("required /usr/bin/python3 CUDA environment unavailable")
        device = torch.device("cuda")
        conv = torch.nn.Conv2d(4, 8, 3, padding=1).to(device)
        value = conv(torch.randn(2, 4, 16, 16, device=device))
        cuda_detail = {
            "interpreter": sys.executable, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "architecture_list": torch.cuda.get_arch_list(), "finite_convolution": bool(value.isfinite().all()),
        }
        cuda_ok = (
            cuda_detail["compute_capability"] == [12, 0]
            and "sm_120" in cuda_detail["architecture_list"] and cuda_detail["finite_convolution"]
        )
    except Exception as exc:
        cuda_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append({"name": "sm120_cuda_runtime", "pass": cuda_ok, "detail": cuda_detail})

    result = {
        "schema": "route_b_v3_1_recovered_epoch40_accepted_preflight_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "all_pass": all(check["pass"] for check in checks), "checks": checks,
        "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "ACCEPTED_BASE_PREFLIGHT.json", result)
    if result["all_pass"]:
        (experiment / "ACCEPTED_BASE_PREFLIGHT_COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["all_pass"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
