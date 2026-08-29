#!/usr/bin/env python3
"""Fixed canonical/2D evaluation, epoch-12 gate, and final candidate selection."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
EXPANDED_PACKAGE = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2"
AUDIT_PACKAGE = PACKAGE.parent / "route_b_v3_1_person_contract_audit_v1"
for path in (str(PACKAGE), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
if str(PACKAGE) in sys.path:
    sys.path.remove(str(PACKAGE))
sys.path.insert(0, str(PACKAGE))

from common_v1 import read_csv, sha256, utc_now, write_json_x, write_text_x  # noqa: E402

_SCORING_SPEC = importlib.util.spec_from_file_location(
    "route_b_visible_anchor_frozen_scoring_v2", EXPANDED_PACKAGE / "scoring_v2.py",
)
if _SCORING_SPEC is None or _SCORING_SPEC.loader is None:
    raise ImportError("unable to load frozen canonical scorer")
scoring = importlib.util.module_from_spec(_SCORING_SPEC); _SCORING_SPEC.loader.exec_module(scoring)

_MATCH_SPEC = importlib.util.spec_from_file_location(
    "route_b_visible_anchor_person_matching_v1", AUDIT_PACKAGE / "matching_v1.py",
)
if _MATCH_SPEC is None or _MATCH_SPEC.loader is None:
    raise ImportError("unable to load fixed person diagnostics")
matching = importlib.util.module_from_spec(_MATCH_SPEC); _MATCH_SPEC.loader.exec_module(matching)

EPOCHS = (6, 12, 18, 24)


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _vehicle_rows(path: Path) -> list[tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(name for name in (reader.fieldnames or ()) if name != "prediction_index")
        return [tuple(row[name] for name in fields) for row in reader if row["class_name"] == "vehicle"]


def _segmentation_hashes(path: Path) -> dict[str, str]:
    return {row["sample_id"]: row["sha256"] for row in read_csv(path)}


def _diagnostics(experiment: Path, detections: Path, model_name: str) -> dict[str, Any]:
    frame_ids = matching.load_frame_ids(experiment)
    gt, _metadata, _clear = matching.load_person_gt(experiment)
    predictions = matching.load_predictions(detections)
    matching.annotate_neutral_predictions(predictions, experiment, frame_ids)
    two_d: dict[str, Any] = {}
    conditional_rows: list[dict[str, Any]] = []
    for threshold in (0.20, 0.02):
        threshold_key = f"{threshold:.2f}"
        two_d[threshold_key] = {}
        for definition in matching.MATCH_DEFINITIONS:
            result = matching.image_match(frame_ids, gt, predictions, threshold, definition)
            two_d[threshold_key][definition] = {
                key: result[key] for key in (
                    "tp", "fp", "fn", "precision", "recall", "f1", "eligible_gt",
                    "ignored_predictions", "class_confusion_gt_count", "contended_gt_count",
                    "contended_prediction_count",
                )
            }
            conditional_rows.extend(
                matching.summarize_conditional(model_name, threshold, definition, result)
            )
    return {"two_d": two_d, "conditional_localization": conditional_rows}


def _overall_conditional(diagnostics: Mapping[str, Any], threshold: float,
                         definition: str) -> Mapping[str, Any]:
    return next(row for row in diagnostics["conditional_localization"]
                if float(row["threshold"]) == threshold
                and row["match_definition"] == definition
                and row["subset_kind"] == "overall")


def _preservation(candidate_root: Path, baseline_root: Path,
                  candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    candidate_vehicle = _vehicle_rows(candidate_root / "detections.csv")
    baseline_vehicle = _vehicle_rows(baseline_root / "detections.csv")
    candidate_seg = _segmentation_hashes(candidate_root / "segmentation_manifest.csv")
    baseline_seg = _segmentation_hashes(baseline_root / "segmentation_manifest.csv")
    vehicle_keys = (
        "vehicle_tp", "vehicle_fp", "vehicle_fn", "vehicle_precision", "vehicle_recall",
        "vehicle_f1", "vehicle_recall_002", "vehicle_xy_mae_m", "vehicle_iou",
    )
    segmentation_keys = ("foreground_miou", "vehicle_iou", "person_box_mask_iou")
    metric_deltas = {key: float(candidate["metrics"][key]) - float(baseline["metrics"][key])
                     for key in vehicle_keys if key in candidate["metrics"]}
    segmentation_deltas = {
        key: float(candidate["metrics"][key]) - float(baseline["metrics"][key])
        for key in segmentation_keys
    }
    return {
        "vehicle_detection_rows_candidate": len(candidate_vehicle),
        "vehicle_detection_rows_baseline": len(baseline_vehicle),
        "vehicle_detection_csv_fields_bit_identical_excluding_artifact_prediction_index": (
            candidate_vehicle == baseline_vehicle
        ),
        "segmentation_frames_candidate": len(candidate_seg),
        "segmentation_frames_baseline": len(baseline_seg),
        "segmentation_png_hashes_bit_identical": candidate_seg == baseline_seg,
        "vehicle_metric_deltas": metric_deltas,
        "segmentation_metric_deltas": segmentation_deltas,
        "canonical_vehicle_metrics_exact": all(value == 0.0 for value in metric_deltas.values()),
        "segmentation_metrics_exact": all(value == 0.0 for value in segmentation_deltas.values()),
        "all_preserved": (
            candidate_vehicle == baseline_vehicle and candidate_seg == baseline_seg
            and all(value == 0.0 for value in metric_deltas.values())
            and all(value == 0.0 for value in segmentation_deltas.values())
        ),
    }


def _service_targets(metrics: Mapping[str, float], config: Mapping[str, Any]) -> dict[str, bool]:
    gate = config["service_targets"]
    return {
        "vehicle_precision": metrics["vehicle_precision"] >= gate["vehicle_precision_min"],
        "vehicle_recall": metrics["vehicle_recall"] >= gate["vehicle_recall_min"],
        "person_precision": metrics["person_precision"] >= gate["person_precision_min"],
        "person_recall": metrics["person_recall"] >= gate["person_recall_min"],
        "vehicle_xy": metrics["vehicle_xy_mae_m"] <= gate["vehicle_xy_mae_max_m"],
        "person_xy": metrics["person_xy_mae_m"] <= gate["person_xy_mae_max_m"],
        "vehicle_iou": metrics["vehicle_iou"] >= gate["vehicle_iou_min"],
        "person_box_mask_iou": metrics["person_box_mask_iou"] >= gate["person_box_mask_iou_min"],
        "foreground_miou": metrics["foreground_miou"] >= gate["foreground_miou_min"],
    }


def _gates(record: Mapping[str, Any], baseline: Mapping[str, Any],
           config: Mapping[str, Any]) -> dict[str, Any]:
    metrics, base = record["metrics"], baseline["metrics"]
    diagnostic = record["diagnostics"]
    iou50_f1 = diagnostic["two_d"]["0.20"]["FULL_BOX_IOU_050"]["f1"]
    base_iou50_f1 = baseline["diagnostics"]["two_d"]["0.20"]["FULL_BOX_IOU_050"]["f1"]
    conditional = float(_overall_conditional(diagnostic, 0.02, "FULL_BOX_IOU_050")["within_3m_fraction"])
    base_conditional = float(_overall_conditional(
        baseline["diagnostics"], 0.02, "FULL_BOX_IOU_050",
    )["within_3m_fraction"])
    eligibility = {
        "vehicle_and_segmentation_exact": bool(record["preservation"]["all_preserved"]),
        "all_finite": bool(record["all_finite"]),
        "person_precision": metrics["person_precision"] >= base["person_precision"] - 0.03,
        "person_xy": metrics["person_xy_mae_m"] <= base["person_xy_mae_m"] + 0.05,
        "target_geometry_decoder_contract": True,
    }
    route_a = {
        "person_f1": metrics["person_f1"] >= base["person_f1"] + 0.03,
        "person_recall": metrics["person_recall"] >= base["person_recall"] + 0.04,
        "person_precision": metrics["person_precision"] >= base["person_precision"] - 0.02,
        "person_recall_002": metrics["person_recall_002"] >= base["person_recall_002"] + 0.05,
        "person_xy": metrics["person_xy_mae_m"] <= base["person_xy_mae_m"] + 0.05,
    }
    route_b = {
        "person_f1": metrics["person_f1"] >= base["person_f1"] + 0.015,
        "person_iou50_f1_020": iou50_f1 >= base_iou50_f1 + 0.05,
        "person_recall_002": metrics["person_recall_002"] >= base["person_recall_002"] + 0.05,
        "conditional_iou50_within_3m_002": conditional >= base_conditional - 0.02,
        "person_precision": metrics["person_precision"] >= base["person_precision"] - 0.03,
    }
    absolute = {
        "person_precision_ge_0_80": metrics["person_precision"] >= 0.80,
        "person_recall_ge_0_80": metrics["person_recall"] >= 0.80,
        "person_xy_le_1_20m": metrics["person_xy_mae_m"] <= 1.20,
    }
    return {
        "eligibility": eligibility, "eligible": all(eligibility.values()),
        "material_route_a": route_a, "material_route_a_pass": all(route_a.values()),
        "material_route_b": route_b, "material_route_b_pass": all(route_b.values()),
        "material_gain_pass": all(eligibility.values()) and (all(route_a.values()) or all(route_b.values())),
        "absolute_person_targets": absolute,
        "absolute_person_targets_met": sum(absolute.values()),
        "iou50_f1_020": iou50_f1, "conditional_iou50_within_3m_002": conditional,
    }


def _ensure_baseline(experiment: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = experiment / "evaluation/BASELINE_REPRODUCTION.json"
    if path.is_file():
        return json.loads(path.read_text())
    baseline_root = (ROOT / config["baseline_prediction_root"]).resolve(strict=True)
    checkpoint = (ROOT / config["warm_start_checkpoint"]).resolve(strict=True)
    record = scoring.score_primary(
        experiment, baseline_root, checkpoint, sha256(checkpoint), 40,
    )
    record["diagnostics"] = _diagnostics(experiment, baseline_root / "detections.csv", "base_epoch_040")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_x(path, record)
    return record


def _evaluate_epoch(experiment: Path, config: Mapping[str, Any], epoch: int,
                    baseline: Mapping[str, Any]) -> dict[str, Any]:
    output = experiment / f"evaluation/epoch_{epoch:03d}.json"
    if output.is_file():
        return json.loads(output.read_text())
    checkpoint = experiment / f"checkpoints/{config['name']}/epoch_{epoch:03d}.pt"
    checkpoint_hash = sha256(checkpoint)
    tag = f"visible_anchor_epoch_{epoch:03d}"
    prediction_root = experiment / "predictions" / tag
    if not (prediction_root / "INFERENCE_COMPLETE").is_file():
        command = [
            sys.executable, str(PACKAGE / "infer_v1.py"), "--experiment", str(experiment),
            "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_hash,
            "--tag", tag,
        ]
        if subprocess.run(command).returncode != 0:
            raise RuntimeError(f"inference failed for epoch {epoch}")
    record = scoring.score_primary(
        experiment, prediction_root, checkpoint, checkpoint_hash, epoch,
    )
    record["diagnostics"] = _diagnostics(
        experiment, prediction_root / "detections.csv", f"candidate_epoch_{epoch:03d}",
    )
    baseline_root = (ROOT / config["baseline_prediction_root"]).resolve(strict=True)
    record["preservation"] = _preservation(prediction_root, baseline_root, record, baseline)
    record["all_finite"] = _finite(record["metrics"]) and _finite(record["diagnostics"])
    record["gates"] = _gates(record, baseline, config)
    record["service_targets"] = _service_targets(record["metrics"], config)
    write_json_x(output, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--phase", choices=("catastrophic", "final"), required=True)
    args = parser.parse_args()
    started = time.monotonic(); experiment = args.experiment.resolve(strict=True)
    config = json.loads((experiment / "RESOLVED_CONFIG.json").read_text())
    evaluation_dir = experiment / "evaluation"; evaluation_dir.mkdir(exist_ok=True)
    baseline = _ensure_baseline(experiment, config)
    if args.phase == "catastrophic":
        record = _evaluate_epoch(experiment, config, 12, baseline)
        metrics, base = record["metrics"], baseline["metrics"]
        gates = {
            "all_finite": bool(record["all_finite"]),
            "person_f1_not_more_than_0_05_below_baseline": (
                metrics["person_f1"] >= base["person_f1"] - 0.05
            ),
            "person_xy_not_more_than_0_20m_worse": (
                metrics["person_xy_mae_m"] <= base["person_xy_mae_m"] + 0.20
            ),
            "vehicle_and_segmentation_preserved": bool(record["preservation"]["all_preserved"]),
        }
        result = {
            "schema": "route_b_v3_1_person_visible_anchor_epoch12_catastrophic_gate_v1",
            "created_utc": utc_now(), "epoch": 12, "gates": gates,
            "pass": all(gates.values()), "record": str(evaluation_dir / "epoch_012.json"),
            "inference_passes_for_epoch12": 1, "wall_seconds": time.monotonic() - started,
        }
        write_json_x(experiment / "EPOCH12_CATASTROPHIC_GATE.json", result)
        write_text_x(experiment / "EPOCH12_CATASTROPHIC_GATE_COMPLETE",
                     "PASS\n" if result["pass"] else "FAIL\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0 if result["pass"] else 3

    if not (experiment / "TRAINING_COMPLETE").is_file():
        raise RuntimeError("final evaluation requires complete 24-epoch training")
    gate = json.loads((experiment / "EPOCH12_CATASTROPHIC_GATE.json").read_text())
    if not gate["pass"]:
        raise RuntimeError("cannot finalize a run stopped by the epoch-12 catastrophic gate")
    records = [_evaluate_epoch(experiment, config, epoch, baseline) for epoch in EPOCHS]
    eligible = [record for record in records if record["gates"]["eligible"]]
    ranked = sorted(eligible, key=lambda record: (
        -int(record["gates"]["absolute_person_targets_met"]),
        -float(record["metrics"]["person_f1"]),
        -float(record["metrics"]["person_recall"]),
        float(record["metrics"]["person_xy_mae_m"]),
        -float(record["gates"]["iou50_f1_020"]),
        int(record["epoch"]),
    ))
    selected = ranked[0] if ranked else None
    sensitivity = None
    material = bool(selected and selected["gates"]["material_gain_pass"])
    if material and selected is not None:
        sensitivity = scoring.score_sensitivity(
            experiment, Path(selected["prediction_root"]),
        )
        write_json_x(experiment / "evaluation/SELECTED_V025_SENSITIVITY.json", sensitivity)
    if selected is None or not material:
        terminal = "LRASPP_VISIBLE_ANCHOR_NO_GAIN"
    elif all(selected["gates"]["absolute_person_targets"].values()):
        terminal = "LRASPP_VISIBLE_ANCHOR_PERSON_TARGETS_MET"
    else:
        terminal = "LRASPP_VISIBLE_ANCHOR_MATERIAL_GAIN"
    service = selected["service_targets"] if selected else None
    structural = {
        "vehicle_precision_target_unreachable_with_frozen_vehicle_path": not baseline["metrics"]["vehicle_precision"] >= 0.80,
        "vehicle_recall_target_unreachable_with_frozen_vehicle_path": not baseline["metrics"]["vehicle_recall"] >= 0.85,
        "person_box_mask_iou_target_unreachable_with_frozen_segmentation": not baseline["metrics"]["person_box_mask_iou"] >= 0.50,
        "foreground_miou_target_unreachable_with_frozen_segmentation": not baseline["metrics"]["foreground_miou"] >= 0.675,
    }
    result = {
        "schema": "route_b_v3_1_person_visible_anchor_evaluation_selection_v1",
        "created_utc": utc_now(), "evaluated_epochs": list(EPOCHS),
        "records": [str(evaluation_dir / f"epoch_{epoch:03d}.json") for epoch in EPOCHS],
        "eligible_epochs": [int(record["epoch"]) for record in eligible],
        "ranking": [{
            "epoch": int(record["epoch"]),
            "absolute_person_targets_met": int(record["gates"]["absolute_person_targets_met"]),
            "person_f1": record["metrics"]["person_f1"],
            "person_recall": record["metrics"]["person_recall"],
            "person_xy_mae_m": record["metrics"]["person_xy_mae_m"],
            "iou50_f1_020": record["gates"]["iou50_f1_020"],
        } for record in ranked],
        "selected_epoch": int(selected["epoch"]) if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
        "selected_material_gain": material, "selected_service_targets": service,
        "full_service_ready": bool(service and all(service.values())),
        "structurally_unreachable_frozen_service_gates": structural,
        "v025_sensitivity_run": sensitivity is not None, "v025_sensitivity": sensitivity,
        "terminal": terminal, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "SELECTION_DECISION.json", result)
    write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
    write_text_x(experiment / "EVALUATION_COMPLETE", terminal + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
