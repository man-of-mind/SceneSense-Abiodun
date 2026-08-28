#!/usr/bin/env python3
"""Registered final eligibility, material-gain, Pareto, and ranking policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def vehicle_rows(inference_path: Path, inference: dict[str, Any]) -> int:
    if "vehicle_detection_rows" in inference:
        return int(inference["vehicle_detection_rows"])
    with (inference_path.parent / "detections.csv").open("r", encoding="utf-8", newline="") as stream:
        return sum(row["class_name"] == "vehicle" for row in csv.DictReader(stream))


def service_targets(metrics: dict[str, float], targets: dict[str, float]) -> dict[str, bool]:
    return {
        "vehicle_precision": metrics["vehicle_precision"] >= targets["vehicle_precision_min"],
        "vehicle_recall": metrics["vehicle_recall"] >= targets["vehicle_recall_min"],
        "person_precision": metrics["person_precision"] >= targets["person_precision_min"],
        "person_recall": metrics["person_recall"] >= targets["person_recall_min"],
        "vehicle_xy_mae": metrics["vehicle_xy_mae_m"] <= targets["vehicle_xy_mae_max_m"],
        "person_xy_mae": metrics["person_xy_mae_m"] <= targets["person_xy_mae_max_m"],
        "vehicle_iou": metrics["vehicle_iou"] >= targets["vehicle_iou_min"],
        "person_box_mask_iou": metrics["person_box_mask_iou"] >= targets["person_box_mask_iou_min"],
        "foreground_miou": metrics["foreground_miou"] >= targets["foreground_miou_min"],
    }


def eligibility(record: dict[str, Any], base: dict[str, Any], limits: dict[str, float],
                base_vehicle_rows: int) -> dict[str, bool]:
    metrics, reference = record["metrics"], base["metrics"]
    vehicle_rows = int(record["vehicle_detection_rows"])
    count_tolerance = float(limits["vehicle_count_relative_tolerance"])
    return {
        "vehicle_tp_within_0_2pct": abs(int(metrics["vehicle_tp"]) - int(reference["vehicle_tp"]))
                                    / max(1, int(reference["vehicle_tp"])) <= count_tolerance,
        "vehicle_fp_within_0_2pct": abs(int(metrics["vehicle_fp"]) - int(reference["vehicle_fp"]))
                                    / max(1, int(reference["vehicle_fp"])) <= count_tolerance,
        "vehicle_fn_within_0_2pct": abs(int(metrics["vehicle_fn"]) - int(reference["vehicle_fn"]))
                                    / max(1, int(reference["vehicle_fn"])) <= count_tolerance,
        "vehicle_raw_detection_rows_unchanged": vehicle_rows == base_vehicle_rows,
        "vehicle_f1_delta": metrics["vehicle_f1"] - reference["vehicle_f1"] >= limits["vehicle_f1_delta_min"],
        "vehicle_recall_delta": metrics["vehicle_recall"] - reference["vehicle_recall"] >= limits["vehicle_recall_delta_min"],
        "vehicle_xy_increase": metrics["vehicle_xy_mae_m"] - reference["vehicle_xy_mae_m"] <= limits["vehicle_xy_delta_max_m"],
        "vehicle_iou_delta": metrics["vehicle_iou"] - reference["vehicle_iou"] >= limits["vehicle_iou_delta_min"],
        "person_precision_delta": metrics["person_precision"] - reference["person_precision"] >= limits["person_precision_delta_min"],
        "person_recall_delta": metrics["person_recall"] - reference["person_recall"] >= limits["person_recall_delta_min"],
        "person_xy_increase": metrics["person_xy_mae_m"] - reference["person_xy_mae_m"] <= limits["person_xy_delta_max_m"],
        "person_iou_delta": metrics["person_box_mask_iou"] - reference["person_box_mask_iou"] >= limits["person_iou_delta_min"],
    }


def material(metrics: dict[str, float], base: dict[str, float], contract: dict[str, Any]) -> dict[str, Any]:
    delta = {key: metrics[key] - base[key] for key in (
        "person_f1", "person_recall", "person_precision", "person_box_mask_iou"
    )}
    xy_improvement = base["person_xy_mae_m"] - metrics["person_xy_mae_m"]
    a = contract["A"]
    gates_a = {
        "person_f1": delta["person_f1"] >= a["person_f1_delta_min"],
        "person_recall": delta["person_recall"] >= a["person_recall_delta_min"],
        "person_precision": delta["person_precision"] >= a["person_precision_delta_min"],
        "person_xy_improvement": xy_improvement >= a["person_xy_improvement_min_m"],
    }
    b = contract["B"]
    gates_b = {
        "person_f1": delta["person_f1"] >= b["person_f1_delta_min"],
        "person_xy_improvement": xy_improvement >= b["person_xy_improvement_min_m"],
        "person_iou": delta["person_box_mask_iou"] >= b["person_iou_delta_min"],
    }
    return {
        "deltas": delta, "person_xy_improvement_m": xy_improvement,
        "A": {"gates": gates_a, "pass": all(gates_a.values())},
        "B": {"gates": gates_b, "pass": all(gates_b.values())},
        "pass": all(gates_a.values()) or all(gates_b.values()),
    }


def deficit(metrics: dict[str, float]) -> float:
    return (
        max(0.0, 0.80 - metrics["person_precision"]) / 0.80
        + max(0.0, 0.80 - metrics["person_recall"]) / 0.80
        + max(0.0, metrics["person_xy_mae_m"] - 1.20) / 1.20
        + max(0.0, 0.50 - metrics["person_box_mask_iou"]) / 0.50
    )


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ma, mb = a["metrics"], b["metrics"]
    better_equal = (
        ma["person_f1"] >= mb["person_f1"]
        and ma["person_recall"] >= mb["person_recall"]
        and ma["person_precision"] >= mb["person_precision"]
        and ma["person_xy_mae_m"] <= mb["person_xy_mae_m"]
        and ma["person_box_mask_iou"] >= mb["person_box_mask_iou"]
    )
    strict = (
        ma["person_f1"] > mb["person_f1"]
        or ma["person_recall"] > mb["person_recall"]
        or ma["person_precision"] > mb["person_precision"]
        or ma["person_xy_mae_m"] < mb["person_xy_mae_m"]
        or ma["person_box_mask_iou"] > mb["person_box_mask_iou"]
    )
    return better_equal and strict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-record", required=True, type=Path)
    parser.add_argument("--base-inference", required=True, type=Path)
    parser.add_argument("--candidate-record", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base = json.loads(args.base_record.read_text())
    base_inference = json.loads(args.base_inference.read_text())
    base.update({
        "label": "epoch_040_base", "selection_order": 0,
        "vehicle_detection_rows": vehicle_rows(args.base_inference, base_inference),
    })
    records = [base]
    for path in args.candidate_record:
        record = json.loads(path.read_text())
        inference = json.loads((Path(record["prediction_root"]) / "inference_manifest.json").read_text())
        inference_path = Path(record["prediction_root"]) / "inference_manifest.json"
        record.update({
            "label": f"person_epoch_{int(record['epoch']):03d}",
            "selection_order": int(record["epoch"]),
            "vehicle_detection_rows": vehicle_rows(inference_path, inference),
        })
        records.append(record)
    if not all(record.get("all_metrics_finite") and all(
        not isinstance(value, float) or math.isfinite(value) for value in record["metrics"].values()
    ) for record in records):
        raise RuntimeError("nonfinite final scoring record")
    for record in records:
        gates = eligibility(
            record, base, config["final_eligibility"], int(base["vehicle_detection_rows"]),
        )
        record["eligibility_gates"] = gates
        record["eligible"] = all(gates.values())
        record["material_gain"] = material(record["metrics"], base["metrics"], config["material_gain"])
        record["normalized_person_deficit"] = deficit(record["metrics"])
        record["service_targets"] = service_targets(record["metrics"], config["service_targets"])
        record["service_ready"] = all(record["service_targets"].values())
    candidate_records = records[1:]
    eligible = [record for record in candidate_records if record["eligible"]]
    if not eligible:
        raise RuntimeError("no eligible person-refinement checkpoint")
    ranked = sorted(eligible, key=lambda record: (
        record["normalized_person_deficit"], -record["metrics"]["person_f1"],
        -record["metrics"]["person_recall"], record["metrics"]["person_xy_mae_m"],
        record["selection_order"],
    ))
    selected = ranked[0]
    nondominated = [
        record["label"] for record in candidate_records
        if not any(dominates(other, record) for other in candidate_records if other is not record)
    ]
    if selected["service_ready"]:
        terminal = "LRASPP_PERSON_REFINEMENT_SERVICE_READY"
    elif selected["material_gain"]["pass"]:
        terminal = "LRASPP_PERSON_REFINEMENT_MATERIAL_GAIN"
    else:
        terminal = "LRASPP_PERSON_REFINEMENT_NO_GAIN"
    result = {
        "schema": "route_b_v3_1_person_refinement_selection_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_label": "epoch_040_base", "records": records,
        "eligible_labels": [record["label"] for record in eligible],
        "ranking": [{
            "label": record["label"], "normalized_person_deficit": record["normalized_person_deficit"],
            "person_f1": record["metrics"]["person_f1"],
            "person_recall": record["metrics"]["person_recall"],
            "person_xy_mae_m": record["metrics"]["person_xy_mae_m"],
        } for record in ranked],
        "nondominated_labels": nondominated,
        "material_labels": [record["label"] for record in candidate_records if record["material_gain"]["pass"]],
        "selected": selected, "terminal": terminal,
    }
    write_json_x(args.output, result)
    print(json.dumps({
        "terminal": terminal, "selected": selected["label"],
        "eligible": result["eligible_labels"], "nondominated": nondominated,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
