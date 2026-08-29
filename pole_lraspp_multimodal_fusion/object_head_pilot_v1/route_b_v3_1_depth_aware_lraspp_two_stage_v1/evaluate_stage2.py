from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from common import CONFIG_PATH, load_json, read_csv, sha256, utc_now, write_json_x, write_text_x
from data import InferenceDataset
from model import build_model, freeze_bn_running_state
from two_stage import is_representation, state_hash

PACKAGE = Path(__file__).resolve().parent
SCORING_PATH = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2/scoring_v2.py"
MATCHING_PATH = PACKAGE.parent / "route_b_v3_1_person_contract_audit_v1/matching_v1.py"
OLD_EVALUATE = PACKAGE.parent / "route_b_v3_1_depth_aware_lraspp_v1/evaluate.py"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def finite(value: Any) -> bool:
    if isinstance(value, Mapping): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    if isinstance(value, (int, float)): return math.isfinite(float(value))
    return True


def representation_output_audit(config: dict[str, Any], root: Path, selected_path: Path,
                                candidate_path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    device = torch.device("cuda"); weight = Path(config["pretrained"]["path"])
    selected, _ = build_model(weight, device); candidate, _ = build_model(weight, device)
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    selected.load_state_dict(selected_payload["model"], strict=True); candidate.load_state_dict(candidate_payload["model"], strict=True)
    selected.eval(); candidate.eval(); freeze_bn_running_state(selected); freeze_bn_running_state(candidate)
    selected_hash = state_hash(selected, is_representation); candidate_hash = state_hash(candidate, is_representation)
    dataset = InferenceDataset(root, rows); equal = True; compared = 0
    digest = hashlib.sha256(); started = time.monotonic()
    with torch.inference_mode():
        for index in range(len(dataset)):
            value, _row = dataset[index]; value = value.unsqueeze(0).to(device)
            left = selected.representation_outputs(value); right = candidate.representation_outputs(value)
            for name in ("out", "dense_depth_log1p"):
                same = torch.equal(left[name], right[name]); equal = equal and same
                if not same: raise RuntimeError(f"Stage-2 {candidate_path.name} changed {name} at validation index {index}")
                digest.update(name.encode()); digest.update(right[name].cpu().contiguous().numpy().tobytes())
            compared += 1
            if (index + 1) % 500 == 0: print(f"[frozen output audit {candidate_path.stem}] {index+1}/{len(dataset)}", flush=True)
    del selected, candidate; torch.cuda.empty_cache()
    return {"frames_compared": compared, "segmentation_dense_bit_identical": equal,
            "selected_representation_hash": selected_hash, "candidate_representation_hash": candidate_hash,
            "representation_hash_equal": selected_hash == candidate_hash, "candidate_output_sha256": digest.hexdigest(),
            "wall_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True); started = time.monotonic()
    if not (experiment / "stage2/TRAINING_COMPLETE").is_file(): raise RuntimeError("Stage-2 training incomplete")
    config = load_json(CONFIG_PATH); root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(root / "dataset/manifest.csv") if row["split"] == "val"]
    stage1 = load_json(experiment / "STAGE1_SELECTION.json"); transition = load_json(experiment / "STAGE2_TRANSITION.json")
    selected_stage1 = Path(stage1["selected_checkpoint"]); frozen_hash = transition["frozen_representation_hash"]
    # Audit every Stage-2 epoch boundary record before opening predictions.
    boundary_audit = []
    for epoch in range(1, 31):
        metric = load_json(experiment / f"stage2/training_metrics/epoch_{epoch:03d}.json")
        passed = metric["frozen_representation_hash"] == frozen_hash
        boundary_audit.append({"epoch": epoch, "hash": metric["frozen_representation_hash"], "pass": passed})
    if not all(item["pass"] for item in boundary_audit): raise RuntimeError("Stage-2 frozen boundary hash audit failed")
    scoring = load_module("two_stage_scoring", SCORING_PATH); matching = load_module("two_stage_matching", MATCHING_PATH)
    helpers = load_module("two_stage_depth_helpers", OLD_EVALUATE)
    evaluation = experiment / "stage2/evaluation"; evaluation.mkdir(parents=True, exist_ok=True)
    records = []
    for epoch in (10, 20, 30):
        prediction = experiment / f"stage2/predictions/epoch_{epoch:03d}"
        checkpoint = experiment / f"stage2/checkpoints/stage2_epoch_{epoch:03d}.pt"
        checkpoint_hash = sha256(checkpoint); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        inference = load_json(prediction / "inference_manifest.json")
        output_audit = representation_output_audit(config, root, selected_stage1, checkpoint, rows)
        record = scoring.score_primary(root, prediction, checkpoint, checkpoint_hash, epoch)
        record["person_iou_diagnostics"] = helpers.person_iou_diagnostics(root, prediction / "detections.csv", matching)
        record["actor_depth_diagnostics"] = helpers.actor_depth_diagnostics(root, prediction / "detections.csv")
        record["frozen_output_audit"] = output_audit
        record["frozen_representation_hash"] = payload["frozen_representation_hash"]
        record["inference_contract"] = {"single_floor_pass": inference["inference_pass_count"] == 1,
            "score_floor": inference["score_floor"], "derived_threshold": inference["derived_threshold"],
            "split_parity": inference["split_report"]["all_raw_equal"],
            "no_inference_depth": inference["depth_labels_used"] is False and inference["depth_paths_opened"] == 0,
            "latency": inference["latency"], "peak_allocated_mib": inference["peak_allocated_mib"],
            "peak_reserved_mib": inference["peak_reserved_mib"], "transport_shapes": inference["transport_shapes"],
            "transport_dtypes": inference["transport_dtypes"], "raw_transport_bytes": inference["raw_transport_bytes"],
            "identity_serialized_transport_bytes": inference["identity_serialized_transport_bytes"]}
        record["all_finite"] = finite(record)
        metrics = record["metrics"]
        eligibility = {"stage1_gates_preserved": True,
            "frozen_representation_hash": payload["frozen_representation_hash"] == frozen_hash,
            "segmentation_dense_outputs_exact": output_audit["segmentation_dense_bit_identical"],
            "vehicle_f1": metrics["vehicle_f1"] >= config["stage2_gates"]["eligibility"]["vehicle_f1_min"],
            "vehicle_recall": metrics["vehicle_recall"] >= config["stage2_gates"]["eligibility"]["vehicle_recall_min"],
            "all_outputs_finite": record["all_finite"], "split_runtime_contract": (
                record["inference_contract"]["single_floor_pass"] and record["inference_contract"]["split_parity"]
                and record["inference_contract"]["no_inference_depth"])}
        record["eligibility"] = eligibility; record["eligible"] = all(eligibility.values())
        record["normalized_service_attainment"] = min(metrics["vehicle_precision"]/.80,
            metrics["vehicle_recall"]/.85, metrics["person_precision"]/.80, metrics["person_recall"]/.80)
        write_json_x(evaluation / f"epoch_{epoch:03d}.json", record); records.append(record)
        print(json.dumps({"epoch": epoch, "eligible": record["eligible"],
                          "attainment": record["normalized_service_attainment"], "metrics": metrics}), flush=True)
    eligible = [record for record in records if record["eligible"]]
    ranked = sorted(eligible, key=lambda item: (-item["normalized_service_attainment"],
        -(item["metrics"]["vehicle_f1"] + item["metrics"]["person_f1"])/2,
        max(item["metrics"]["vehicle_xy_mae_m"], item["metrics"]["person_xy_mae_m"]), item["epoch"]))
    selected = ranked[0] if ranked else None
    sensitivity = scoring.score_sensitivity(root, Path(selected["prediction_root"])) if selected else None
    if sensitivity is not None: write_json_x(evaluation / "SELECTED_V025_SENSITIVITY.json", sensitivity)
    service = material = None
    if selected:
        m = selected["metrics"]
        service = {"vehicle_precision": m["vehicle_precision"] >= .80, "vehicle_recall": m["vehicle_recall"] >= .85,
            "person_precision": m["person_precision"] >= .80, "person_recall": m["person_recall"] >= .80,
            "vehicle_xy_mae": m["vehicle_xy_mae_m"] <= 1., "person_xy_mae": m["person_xy_mae_m"] <= 1.2,
            "vehicle_iou": m["vehicle_iou"] >= .85, "person_box_mask_iou": m["person_box_mask_iou"] >= .50,
            "foreground_miou": m["foreground_miou"] >= .675}
        material = {"person_f1": m["person_f1"] >= .577617, "person_recall": m["person_recall"] >= .568079,
            "person_xy": m["person_xy_mae_m"] <= 1.341153, "vehicle_eligibility": selected["eligible"],
            "stage1_gates": True}
    if selected and all(service.values()): terminal = "TWO_STAGE_LRASPP_SERVICE_READY"
    elif selected and all(material.values()): terminal = "TWO_STAGE_LRASPP_IMPROVED_NOT_SERVICE_READY"
    else: terminal = "TWO_STAGE_LRASPP_STAGE2_OBJECT_FAILED_CLOSE_LRASPP"
    decision = {"schema": "two_stage_lraspp_stage2_selection_v1", "created_utc": utc_now(),
        "evaluated_epochs": [10,20,30], "eligible_epochs": [item["epoch"] for item in eligible],
        "ranking": [{"epoch": item["epoch"], "minimum_normalized_service_attainment": item["normalized_service_attainment"],
            "mean_f1": (item["metrics"]["vehicle_f1"]+item["metrics"]["person_f1"])/2,
            "worst_xy_mae_m": max(item["metrics"]["vehicle_xy_mae_m"],item["metrics"]["person_xy_mae_m"])} for item in ranked],
        "selected_epoch": selected["epoch"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "service_gates": service, "service_ready": bool(service and all(service.values())),
        "material_gates": material, "material_improvement": bool(material and all(material.values())),
        "v025_sensitivity_run": sensitivity is not None, "v025_sensitivity": sensitivity,
        "frozen_boundary_audit": boundary_audit, "terminal": terminal, "wall_seconds": time.monotonic()-started}
    write_json_x(experiment / "STAGE2_SELECTION.json", decision)
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal+"\n"); write_text_x(experiment / "EVALUATION_COMPLETE", terminal+"\n")
    print(json.dumps(decision, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
