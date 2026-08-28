#!/usr/bin/env python3
"""Score the authorized checkpoints, rank them, and emit the exact final terminal."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from score_contract_v1 import score_model


EPOCHS = (5, 10, 15, 20, 25)
CLASSES = ("vehicle", "person")


def write_json_x(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def primary(model: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    contract = model["contracts"]["v010"]
    return contract["thresholds"]["0.20"]["classes"], contract["thresholds"]["0.02"]["classes"], contract["segmentation"]


def delta(current: float | None, reference: float | None) -> float | None:
    return None if current is None or reference is None else float(current) - float(reference)


def comparison_snapshot(model: Mapping[str, Any]) -> dict[str, float | None]:
    metrics, ceiling, segmentation = primary(model)
    output: dict[str, float | None] = {}
    for class_name in CLASSES:
        for key in ("precision", "recall", "f1", "xy_mae_m", "dimension_mae_m", "yaw_mae_deg"):
            output[f"{class_name}_{key}"] = metrics[class_name][key]
        output[f"{class_name}_recall_002"] = ceiling[class_name]["recall"]
    for key in ("vehicle_iou", "person_box_mask_iou", "foreground_miou", "background_iou"):
        output[key] = segmentation[key]
    return output


def snapshot_delta(current: Mapping[str, float | None], reference: Mapping[str, float | None]) -> dict[str, float | None]:
    return {key: delta(current.get(key), reference.get(key)) for key in current}


def checkpoint_summary(epoch: int, model: Mapping[str, Any], baseline: Mapping[str, Any], feasibility_cfg: Mapping[str, Any], material_cfg: Mapping[str, Any]) -> dict[str, Any]:
    metrics, ceiling, segmentation = primary(model)
    baseline_metrics, _baseline_ceiling, baseline_segmentation = primary(baseline)
    mean_f1 = sum(float(metrics[name]["f1"]) for name in CLASSES) / 2.0
    baseline_mean_f1 = sum(float(baseline_metrics[name]["f1"]) for name in CLASSES) / 2.0
    min_recall = min(float(metrics[name]["recall"]) for name in CLASSES)
    mean_xy = sum(float(metrics[name]["xy_mae_m"]) for name in CLASSES) / 2.0
    feasibility = {
        "vehicle_iou": float(segmentation["vehicle_iou"]) - float(baseline_segmentation["vehicle_iou"]) >= float(feasibility_cfg["vehicle_iou_min_delta"]),
        "person_box_mask_iou": float(segmentation["person_box_mask_iou"]) - float(baseline_segmentation["person_box_mask_iou"]) >= float(feasibility_cfg["person_box_mask_iou_min_delta"]),
        "vehicle_xy_mae": float(metrics["vehicle"]["xy_mae_m"]) - float(baseline_metrics["vehicle"]["xy_mae_m"]) <= float(feasibility_cfg["vehicle_xy_mae_max_increase_m"]),
        "person_xy_mae": float(metrics["person"]["xy_mae_m"]) - float(baseline_metrics["person"]["xy_mae_m"]) <= float(feasibility_cfg["person_xy_mae_max_increase_m"]),
    }
    material = {
        "mean_class_f1": mean_f1 - baseline_mean_f1 >= float(material_cfg["mean_class_f1_min_delta"]),
        "vehicle_f1": float(metrics["vehicle"]["f1"]) - float(baseline_metrics["vehicle"]["f1"]) >= float(material_cfg["per_class_f1_min_delta"]),
        "person_f1": float(metrics["person"]["f1"]) - float(baseline_metrics["person"]["f1"]) >= float(material_cfg["per_class_f1_min_delta"]),
        "foreground_miou": float(segmentation["foreground_miou"]) - float(baseline_segmentation["foreground_miou"]) >= float(material_cfg["foreground_miou_min_delta"]),
    }
    return {
        "epoch": epoch, "mean_class_f1": mean_f1, "mean_class_f1_delta": mean_f1 - baseline_mean_f1,
        "minimum_class_recall": min_recall, "mean_xy_mae_m": mean_xy,
        "foreground_miou": segmentation["foreground_miou"],
        "feasibility_gates": feasibility, "feasible": all(feasibility.values()),
        "material_gain_gates": material, "material_gain": all(material.values()),
        "score_0_20": metrics, "score_0_02": ceiling, "segmentation": segmentation,
    }


def service_results(summary: Mapping[str, Any], targets: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    metrics = summary["score_0_20"]
    segmentation = summary["segmentation"]
    values = {
        "vehicle_precision": float(metrics["vehicle"]["precision"]),
        "vehicle_recall": float(metrics["vehicle"]["recall"]),
        "person_precision": float(metrics["person"]["precision"]),
        "person_recall": float(metrics["person"]["recall"]),
        "vehicle_xy_mae": float(metrics["vehicle"]["xy_mae_m"]),
        "person_xy_mae": float(metrics["person"]["xy_mae_m"]),
        "vehicle_iou": float(segmentation["vehicle_iou"]),
        "person_box_mask_iou": float(segmentation["person_box_mask_iou"]),
        "foreground_miou": float(segmentation["foreground_miou"]),
    }
    output = {}
    for name, value in values.items():
        target_key = f"{name}_max_m" if name.endswith("xy_mae") else f"{name}_min"
        target = float(targets[target_key])
        passes = value <= target if name.endswith("xy_mae") else value >= target
        output[name] = {"value": value, "target": target, "comparison": "<=" if name.endswith("xy_mae") else ">=", "pass": passes}
    return output


def f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def report_markdown(result: Mapping[str, Any]) -> str:
    selected = result.get("selected")
    lines = [
        "# Route B v3.1 clean noAE LR-ASPP report", "",
        f"Terminal: `{result['terminal']}`", "",
        "## GT contract", "",
        "| split | positives | ignores | vehicle actor | vehicle static | person |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        item = result["gt_counts"][split]
        lines.append(f"| {split} | {item['positive_records']} | {item['ignore_records']} | {item['positive_by_class_source']['vehicle:actor']} | {item['positive_by_class_source']['vehicle:environment_static']} | {item['positive_by_class_source']['person:actor']} |")
    lines += ["", "## Frozen baseline and selected checkpoint", ""]
    if selected is None:
        lines.append("No checkpoint was promoted.")
    else:
        lines += [
            f"Selected epoch: **{selected['epoch']}**  ",
            f"Checkpoint: `{selected['checkpoint']}`  ",
            f"SHA-256: `{selected['checkpoint_sha256']}`", "",
        ]
    lines += [
        "| model | class | TP / FP / FN | precision | recall | F1 | recall@0.02 | XY MAE m | dim MAE m | yaw MAE deg | class IoU | foreground mIoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("epoch13_warm_start", "selected"):
        model = result["frozen_epoch13"] if name == "epoch13_warm_start" else selected
        if model is None:
            continue
        metrics = model["score_0_20"]
        ceiling = model["score_0_02"]
        segmentation = model["segmentation"]
        for class_name in CLASSES:
            item = metrics[class_name]
            class_iou = segmentation["vehicle_iou"] if class_name == "vehicle" else segmentation["person_box_mask_iou"]
            lines.append(
                f"| {name} | {class_name} | {item['tp']} / {item['fp']} / {item['fn']} | {f(item['precision'])} | {f(item['recall'])} | {f(item['f1'])} | {f(ceiling[class_name]['recall'])} | {f(item['xy_mae_m'], 3)} | {f(item['dimension_mae_m'], 3)} | {f(item['yaw_mae_deg'], 2)} | {f(class_iou)} | {f(segmentation['foreground_miou'])} |"
            )
    lines += ["", "## Authorized checkpoint ranking", "",
              "| rank | epoch | feasible | material | mean F1 | min recall | mean XY MAE | foreground mIoU |",
              "|---:|---:|---|---|---:|---:|---:|---:|"]
    for rank, item in enumerate(result["ranked_feasible"], 1):
        lines.append(f"| {rank} | {item['epoch']} | yes | {'yes' if item['material_gain'] else 'no'} | {f(item['mean_class_f1'])} | {f(item['minimum_class_recall'])} | {f(item['mean_xy_mae_m'], 3)} | {f(item['foreground_miou'])} |")
    loss_best = result["loss_best"]
    lines += ["", f"Loss-best epoch: **{loss_best['epoch']}** (reported separately; not auto-promoted), val loss `{loss_best['val_loss']:.6f}`.", ""]
    if selected is not None:
        lines += ["## Deltas against frozen context", "", "| reference | mean F1 delta | vehicle F1 delta | person F1 delta | mean XY MAE delta m | foreground mIoU delta |", "|---|---:|---:|---:|---:|---:|"]
        for reference, values in result["deltas_against"].items():
            mean_f1_delta = (float(values["vehicle_f1"]) + float(values["person_f1"])) / 2.0
            mean_xy_delta = (float(values["vehicle_xy_mae_m"]) + float(values["person_xy_mae_m"])) / 2.0
            lines.append(f"| {reference} | {f(mean_f1_delta)} | {f(values['vehicle_f1'])} | {f(values['person_f1'])} | {f(mean_xy_delta, 3)} | {f(values['foreground_miou'])} |")
        sensitivity = result["selected_v025"]
        lines += ["", "## v0.25 sensitivity", "", "| class | precision | recall | F1 | recall@0.02 | XY MAE m |", "|---|---:|---:|---:|---:|---:|"]
        for class_name in CLASSES:
            item = sensitivity["score_0_20"][class_name]
            lines.append(f"| {class_name} | {f(item['precision'])} | {f(item['recall'])} | {f(item['f1'])} | {f(sensitivity['score_0_02'][class_name]['recall'])} | {f(item['xy_mae_m'], 3)} |")
        lines.append(f"\nSegmentation v0.25: vehicle IoU `{f(sensitivity['segmentation']['vehicle_iou'])}`, person box-mask IoU `{f(sensitivity['segmentation']['person_box_mask_iou'])}`, foreground mIoU `{f(sensitivity['segmentation']['foreground_miou'])}`, background IoU diagnostic `{f(sensitivity['segmentation']['background_iou'])}`.")
        lines += ["## Service targets", "", "| target | value | requirement | pass |", "|---|---:|---:|---|"]
        for name, item in result["service_targets"].items():
            lines.append(f"| {name} | {f(item['value'])} | {item['comparison']} {f(item['target'])} | {'PASS' if item['pass'] else 'FAIL'} |")
    lines += [
        "", "## Runtime", "",
        f"Training wall time: `{result['runtime']['training_wall_seconds']:.1f} s`; evaluation wall time: `{result['runtime']['evaluation_wall_seconds']:.1f} s`; peak allocated VRAM: `{result['runtime']['peak_allocated_mib']:.1f} MiB` (peak reserved `{result['runtime']['peak_reserved_mib']:.1f} MiB`).", "",
        f"# {result['terminal']}", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--selection-config", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if (experiment / "TERMINAL_VERDICT.txt").exists():
        raise FileExistsError("terminal already exists")
    started = time.monotonic()
    try:
        config = json.loads(args.selection_config.read_text(encoding="utf-8"))
        baseline_all = json.loads((experiment / "FROZEN_BASELINE_RECONCILIATION.json").read_text(encoding="utf-8"))
        baseline = baseline_all["models"]["epoch13_warm_start"]
        trained_models = {}
        summaries = []
        authorized = json.loads((experiment / "AUTHORIZED_EVALUATIONS_COMPLETE.json").read_text(encoding="utf-8"))
        records = {int(row["epoch"]): row for row in authorized["records"]}
        if tuple(records) != EPOCHS:
            raise RuntimeError(f"authorized epoch set mismatch: {tuple(records)}")
        for epoch in EPOCHS:
            tag = records[epoch]["tag"]
            prediction_root = experiment / "predictions" / tag
            model = score_model(experiment, tag, prediction_root, records[epoch]["checkpoint_sha256"])
            trained_models[str(epoch)] = model
            summaries.append(checkpoint_summary(
                epoch, model, baseline, config["feasibility"], config["material_gain"]
            ))
        feasible = [item for item in summaries if item["feasible"]]
        ranked = sorted(feasible, key=lambda item: (
            -float(item["mean_class_f1"]), -float(item["minimum_class_recall"]),
            float(item["mean_xy_mae_m"]), -float(item["foreground_miou"]), int(item["epoch"]),
        ))
        selected_candidate = ranked[0] if ranked else None
        selected = None
        service = {}
        if selected_candidate is not None and selected_candidate["material_gain"]:
            record = records[int(selected_candidate["epoch"])]
            selected = {**selected_candidate, "checkpoint": record["checkpoint"], "checkpoint_sha256": record["checkpoint_sha256"]}
            service = service_results(selected, config["service_targets"])
            terminal = "LRASPP_V3_1_CLEAN_BASE_READY" if all(item["pass"] for item in service.values()) else "LRASPP_V3_1_IMPROVED_NOT_SERVICE_READY"
        else:
            terminal = "LRASPP_V3_1_NO_MATERIAL_GAIN"
        training = json.loads((experiment / "TRAINING_COMPLETE.json").read_text(encoding="utf-8"))
        training_rows = []
        with (experiment / "metrics/route_b_v3_1_clean_noae_stage2_v1_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
            training_rows = list(csv.DictReader(stream))
        loss_best_row = min(training_rows, key=lambda row: float(row["val_loss"]))
        loss_best_epoch = int(loss_best_row["epoch"])
        if loss_best_epoch != int(authorized["loss_best_epoch"]):
            raise RuntimeError("loss-best epoch drift")
        gt = json.loads((experiment / "GT_CONTRACT_SUMMARY.json").read_text(encoding="utf-8"))["summaries"]["v010"]
        baseline_inference = json.loads((experiment / "predictions/epoch13_warm_start/inference_manifest.json").read_text(encoding="utf-8"))
        evaluation_wall = (
            float(baseline_inference["wall_seconds"])
            + float(baseline_all["wall_seconds"])
            + float(authorized["wall_seconds"])
            + (time.monotonic() - started)
        )
        peak_allocated = max(
            float(training["peak_allocated_mib"]), float(baseline_inference["peak_allocated_mib"]),
            *(float(row["peak_allocated_mib"]) for row in authorized["records"]),
        )
        peak_reserved = max(
            float(training["peak_reserved_mib"]), float(baseline_inference["peak_reserved_mib"]),
            *(float(row["peak_reserved_mib"]) for row in authorized["records"]),
        )
        epoch13_metrics, epoch13_ceiling, epoch13_seg = primary(baseline)
        context_names = ("epoch13_warm_start", "lraspp_mprime_noae", "fasterrcnn_radar_roi_v1_epoch12")
        reference_models = {
            "epoch13_warm_start": baseline,
            "lraspp_mprime_noae": baseline_all["models"]["lraspp_mprime_noae"],
            "fasterrcnn_radar_roi_v1_epoch12": baseline_all["models"]["fasterrcnn_radar_roi_v1_epoch12"],
        }
        deltas_against = {}
        selected_v025 = None
        if selected is not None:
            selected_model = trained_models[str(selected["epoch"])]
            current_snapshot = comparison_snapshot(selected_model)
            deltas_against = {
                name: snapshot_delta(current_snapshot, comparison_snapshot(reference_models[name]))
                for name in context_names
            }
            v025 = selected_model["contracts"]["v025"]
            selected_v025 = {
                "score_0_20": v025["thresholds"]["0.20"]["classes"],
                "score_0_02": v025["thresholds"]["0.02"]["classes"],
                "segmentation": v025["segmentation"],
            }
        result = {
            "schema": "route_b_v3_1_clean_base_selection_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "terminal": terminal, "gt_counts": gt, "evaluated_epochs": list(EPOCHS),
            "loss_best": {"epoch": loss_best_epoch, "val_loss": float(loss_best_row["val_loss"]), "auto_promoted": False},
            "checkpoint_summaries": summaries, "ranked_feasible": ranked,
            "selected_candidate_epoch": selected_candidate["epoch"] if selected_candidate else None,
            "selected": selected, "service_targets": service,
            "selected_v025": selected_v025, "deltas_against": deltas_against,
            "frozen_epoch13": {"score_0_20": epoch13_metrics, "score_0_02": epoch13_ceiling, "segmentation": epoch13_seg},
            "context_baselines": {
                name: baseline_all["models"][name] for name in ("lraspp_mprime_noae", "fasterrcnn_radar_roi_v1_epoch12")
            },
            "trained_models": trained_models,
            "runtime": {
                "training_wall_seconds": float(training["wall_seconds"]),
                "evaluation_wall_seconds": evaluation_wall,
                "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
            },
            "selection_config": config,
        }
        write_json_x(experiment / "FINAL_SELECTION.json", result)
        (experiment / "ROUTE_B_V3_1_CLEAN_BASE_REPORT.md").write_text(report_markdown(result), encoding="utf-8")
        (experiment / "TERMINAL_VERDICT.txt").write_text(terminal + "\n", encoding="utf-8")
        (experiment / "FINAL_COMPLETE").write_text(terminal + "\n", encoding="utf-8")
        print(json.dumps({
            "terminal": terminal, "selected_epoch": selected["epoch"] if selected else None,
            "selected_checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
            "ranked_epochs": [item["epoch"] for item in ranked], "service_targets": service,
        }, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        (experiment / "TERMINAL_VERDICT.txt").write_text("LRASPP_V3_1_RUNTIME_FAILURE\n", encoding="utf-8")
        write_json_x(experiment / "selection_failure.json", {
            "terminal": "LRASPP_V3_1_RUNTIME_FAILURE", "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
