#!/usr/bin/env python3
"""Reconcile retained predictions and freeze per-GT transitions before dense sampling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
VISIBLE = PACKAGE.parent / "route_b_v3_1_person_visible_anchor_v1"
AUDIT = PACKAGE.parent / "route_b_v3_1_person_contract_audit_v1"
EXPANDED = PACKAGE.parent / "route_b_v3_1_native_grid_expanded_training_v2"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from common_v1 import sha256, utc_now, write_csv_x, write_json_x, write_text_x  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matching = load_module("localizer_counterfactual_matching_v1", AUDIT / "matching_v1.py")
scoring = load_module("localizer_counterfactual_scoring_v2", EXPANDED / "scoring_v2.py")


def overall_conditional(diagnostics: Mapping[str, Any], threshold: float) -> Mapping[str, Any]:
    return next(row for row in diagnostics["conditional_localization"]
                if float(row["threshold"]) == threshold
                and row["match_definition"] == "FULL_BOX_IOU_050"
                and row["subset_kind"] == "overall")


def diagnostics(dataset_root: Path, detections: Path, model: str) -> tuple[dict[str, Any], dict[float, Any]]:
    frames = matching.load_frame_ids(dataset_root)
    gt, _metadata, _clear = matching.load_person_gt(dataset_root)
    predictions = matching.load_predictions(detections)
    matching.annotate_neutral_predictions(predictions, dataset_root, frames)
    two_d: dict[str, Any] = {}
    conditional: list[dict[str, Any]] = []
    raw: dict[float, Any] = {}
    for threshold in (0.02, 0.20):
        key = f"{threshold:.2f}"
        two_d[key] = {}
        for definition in matching.MATCH_DEFINITIONS:
            result = matching.image_match(frames, gt, predictions, threshold, definition)
            two_d[key][definition] = {name: result[name] for name in (
                "tp", "fp", "fn", "precision", "recall", "f1", "eligible_gt",
                "ignored_predictions", "class_confusion_gt_count", "contended_gt_count",
                "contended_prediction_count",
            )}
            conditional.extend(matching.summarize_conditional(model, threshold, definition, result))
            if definition == "FULL_BOX_IOU_050":
                raw[threshold] = result
    return {"two_d": two_d, "conditional_localization": conditional}, raw


def metric_values(record: Mapping[str, Any], diag: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record["metrics"]
    conditional = overall_conditional(diag, 0.02)
    taxonomy = record["taxonomy_v010"]["person_fn_at_0_02"]["counts"]
    return {
        "person_precision_020": metrics["person_precision"],
        "person_recall_020": metrics["person_recall"],
        "person_f1_020": metrics["person_f1"],
        "person_recall_002": metrics["person_recall_002"],
        "person_xy_mae_m_020": metrics["person_xy_mae_m"],
        "iou50_f1_020": diag["two_d"]["0.20"]["FULL_BOX_IOU_050"]["f1"],
        "iou50_recall_002": diag["two_d"]["0.02"]["FULL_BOX_IOU_050"]["recall"],
        "iou50_conditional_within_3m_002": conditional["within_3m_fraction"],
        "center_present_world_wrong_002": taxonomy["CENTER_PRESENT_WORLD_WRONG"],
        "iou50_pairs_002": conditional["matched_pairs"],
        "iou50_within_3m_002": conditional["matched_pairs"] - conditional["outside_3m_count"],
    }


def maps(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(pair["gt"]["stable_row"]): pair for pair in result["matches"]}


def state_transition(before: bool, after: bool, positive: str, negative: str) -> str:
    if before and after:
        return f"remained_{positive}"
    if before and not after:
        return f"{positive}_to_{negative}"
    if not before and after:
        return f"{negative}_to_{positive}"
    return f"remained_{negative}"


def build_transitions(dataset_root: Path, base_detections: Path,
                      candidate_detections: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames = matching.load_frame_ids(dataset_root)
    gt, metadata, _clear = matching.load_person_gt(dataset_root)
    base = matching.load_predictions(base_detections)
    candidate = matching.load_predictions(candidate_detections)
    matching.annotate_neutral_predictions(base, dataset_root, frames)
    matching.annotate_neutral_predictions(candidate, dataset_root, frames)
    results: dict[str, dict[float, Any]] = {"base": {}, "candidate": {}}
    canonical: dict[str, dict[float, Any]] = {"base": {}, "candidate": {}}
    center: dict[str, Any] = {}
    for threshold in (0.02, 0.20):
        for name, values in (("base", base), ("candidate", candidate)):
            results[name][threshold] = matching.image_match(
                frames, gt, values, threshold, "FULL_BOX_IOU_050",
            )
            canonical[name][threshold] = matching.canonical_world_match(
                frames, gt, values, threshold,
            )
    for name, values in (("base", base), ("candidate", candidate)):
        center[name] = matching.image_match(frames, gt, values, 0.02, "FULL_BOX_CENTER")
    maps_2d = {name: {threshold: maps(result) for threshold, result in arms.items()}
               for name, arms in results.items()}
    maps_world = {name: {threshold: maps(result) for threshold, result in arms.items()}
                  for name, arms in canonical.items()}
    maps_center = {name: maps(result) for name, result in center.items()}
    rows: list[dict[str, Any]] = []
    for sample_id in frames:
        for target in gt.get(sample_id, ()):
            gid = int(target["stable_row"])
            source = metadata[(sample_id, target["source_identity"])]
            row: dict[str, Any] = {
                "sample_id": sample_id, "source_identity": target["source_identity"],
                "gt_stable_row": gid, "distance_m": target["distance_m"],
                "area_px": target["area_px"],
                "visible_fraction": float(source.get("visible_fraction", 0.0) or 0.0),
                "radar_supported": int(target["radar_supported"]),
                "clear_v025": int(target["clear_v025"]),
            }
            for threshold in (0.02, 0.20):
                suffix = "002" if threshold == 0.02 else "020"
                b2, c2 = maps_2d["base"][threshold].get(gid), maps_2d["candidate"][threshold].get(gid)
                bw, cw = maps_world["base"][threshold].get(gid), maps_world["candidate"][threshold].get(gid)
                row.update({
                    f"base_iou50_match_{suffix}": int(b2 is not None),
                    f"candidate_iou50_match_{suffix}": int(c2 is not None),
                    f"iou50_transition_{suffix}": state_transition(
                        b2 is not None, c2 is not None, "2d_match", "2d_miss",
                    ),
                    f"base_canonical_tp_{suffix}": int(bw is not None),
                    f"candidate_canonical_tp_{suffix}": int(cw is not None),
                    f"canonical_transition_{suffix}": state_transition(
                        bw is not None, cw is not None, "canonical_tp", "canonical_fn",
                    ),
                    f"base_iou50_world_error_m_{suffix}": (
                        b2["world_error_m"] if b2 is not None else ""
                    ),
                    f"candidate_iou50_world_error_m_{suffix}": (
                        c2["world_error_m"] if c2 is not None else ""
                    ),
                    f"candidate_2d_match_world_fail_{suffix}": int(
                        c2 is not None and float(c2["world_error_m"]) > 3.0
                    ),
                    f"remained_2d_matched_candidate_world_fail_{suffix}": int(
                        b2 is not None and c2 is not None and float(c2["world_error_m"]) > 3.0
                    ),
                    f"base_matching_contention_{suffix}": int(
                        len(results["base"][threshold]["potential_by_gt"].get(gid, ())) > 1
                    ),
                    f"candidate_matching_contention_{suffix}": int(
                        len(results["candidate"][threshold]["potential_by_gt"].get(gid, ())) > 1
                    ),
                })
            base_center = maps_center["base"].get(gid)
            candidate_center = maps_center["candidate"].get(gid)
            row["base_center_present_world_wrong_002"] = int(
                base_center is not None and float(base_center["world_error_m"]) > 3.0
            )
            row["candidate_center_present_world_wrong_002"] = int(
                candidate_center is not None and float(candidate_center["world_error_m"]) > 3.0
            )
            row["candidate_confidence_loss_iou50"] = int(
                row["candidate_iou50_match_002"] and not row["candidate_iou50_match_020"]
            )
            rows.append(row)
    summary: dict[str, Any] = {
        "schema": "route_b_v3_1_localizer_counterfactual_transition_summary_v1",
        "created_utc": utc_now(), "gt_rows": len(rows), "thresholds": {},
    }
    for suffix in ("002", "020"):
        summary["thresholds"][suffix] = {
            "iou50_transitions": dict(Counter(row[f"iou50_transition_{suffix}"] for row in rows)),
            "canonical_transitions": dict(Counter(row[f"canonical_transition_{suffix}"] for row in rows)),
            "candidate_2d_match_world_fail": sum(row[f"candidate_2d_match_world_fail_{suffix}"] for row in rows),
            "remained_2d_matched_candidate_world_fail": sum(
                row[f"remained_2d_matched_candidate_world_fail_{suffix}"] for row in rows
            ),
            "base_contended_gt": sum(row[f"base_matching_contention_{suffix}"] for row in rows),
            "candidate_contended_gt": sum(row[f"candidate_matching_contention_{suffix}"] for row in rows),
        }
    summary["center_present_world_wrong_per_gt"] = {
        "base": sum(row["base_center_present_world_wrong_002"] for row in rows),
        "candidate": sum(row["candidate_center_present_world_wrong_002"] for row in rows),
        "delta": sum(row["candidate_center_present_world_wrong_002"]
                     - row["base_center_present_world_wrong_002"] for row in rows),
    }
    summary["candidate_confidence_loss_iou50_count"] = sum(
        row["candidate_confidence_loss_iou50"] for row in rows
    )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    experiment = args.experiment.resolve(strict=True)
    config = json.loads((experiment / "RESOLVED_CONFIG.json").read_text())
    registration = json.loads((experiment / "REGISTERED_AUDIT_PLAN.json").read_text())
    if (registration["counterfactual_results_examined_before_registration"] is not False
            or registration["optimizer_steps_before_registration"] != 0
            or registration["config_sha256"] != sha256(experiment / "RESOLVED_CONFIG.json")):
        raise RuntimeError("registration is not closed")
    dataset_root = (ROOT / config["dataset_root"]).resolve(strict=True)
    base_root = (ROOT / config["base_predictions"]).resolve(strict=True)
    candidate_root = (ROOT / config["candidate_predictions"]).resolve(strict=True)
    base_checkpoint = (ROOT / config["base_checkpoint"]).resolve(strict=True)
    candidate_checkpoint = (ROOT / config["candidate_checkpoint"]).resolve(strict=True)
    base_record = scoring.score_primary(
        dataset_root, base_root, base_checkpoint, config["base_checkpoint_sha256"], 40,
    )
    candidate_record = scoring.score_primary(
        dataset_root, candidate_root, candidate_checkpoint, config["candidate_checkpoint_sha256"], 18,
    )
    base_diag, _base_raw = diagnostics(
        dataset_root, base_root / "detections.csv", "base_epoch_040",
    )
    candidate_diag, _candidate_raw = diagnostics(
        dataset_root, candidate_root / "detections.csv", "visible_anchor_epoch_018",
    )
    actual = {
        "base": metric_values(base_record, base_diag),
        "candidate": metric_values(candidate_record, candidate_diag),
    }
    tolerance = float(config["reconciliation"]["absolute_tolerance"])
    checks: list[dict[str, Any]] = []
    for arm in ("base", "candidate"):
        for metric, expected in config["reconciliation"][arm].items():
            value = actual[arm][metric]
            passed = (value == expected if isinstance(expected, int)
                      else math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=tolerance))
            checks.append({"arm": arm, "metric": metric, "expected": expected,
                           "actual": value, "absolute_delta": abs(float(value) - float(expected)),
                           "pass": passed})
    published_candidate = json.loads((ROOT / config["published_evaluation"]).read_text())
    preservation = published_candidate["preservation"]
    checks.extend([
        {"arm": "preservation", "metric": "vehicle_rows_bit_identical",
         "expected": True,
         "actual": preservation["vehicle_detection_csv_fields_bit_identical_excluding_artifact_prediction_index"],
         "absolute_delta": 0, "pass": preservation["vehicle_detection_csv_fields_bit_identical_excluding_artifact_prediction_index"] is True},
        {"arm": "preservation", "metric": "segmentation_hashes_bit_identical",
         "expected": True, "actual": preservation["segmentation_png_hashes_bit_identical"],
         "absolute_delta": 0, "pass": preservation["segmentation_png_hashes_bit_identical"] is True},
    ])
    reconciliation = {
        "schema": "route_b_v3_1_localizer_counterfactual_reconciliation_v1",
        "created_utc": utc_now(), "checks": checks, "all_pass": all(item["pass"] for item in checks),
        "actual": actual, "base_record_hashes": {
            "detections": base_record["detections_sha256"],
            "prediction_set": base_record["prediction_set_sha256"],
        }, "candidate_record_hashes": {
            "detections": candidate_record["detections_sha256"],
            "prediction_set": candidate_record["prediction_set_sha256"],
        }, "wall_seconds": time.monotonic() - started,
    }
    write_json_x(experiment / "RECONCILIATION.json", reconciliation)
    if not reconciliation["all_pass"]:
        terminal = "LRASPP_LOCALIZER_COUNTERFACTUAL_CONTRACT_INVALID"
        write_text_x(experiment / "TERMINAL_VERDICT.txt", terminal + "\n")
        write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
        print(json.dumps(reconciliation, indent=2, sort_keys=True))
        return 3
    rows, summary = build_transitions(
        dataset_root, base_root / "detections.csv", candidate_root / "detections.csv",
    )
    write_csv_x(experiment / "PER_GT_TRANSITIONS.csv", rows)
    write_json_x(experiment / "TRANSITION_SUMMARY.json", summary)
    write_text_x(experiment / "RECONCILIATION_COMPLETE", "PASS\n")
    print(json.dumps({"reconciliation": "PASS", "transitions": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
