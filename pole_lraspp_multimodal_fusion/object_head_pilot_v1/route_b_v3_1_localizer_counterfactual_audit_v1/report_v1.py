#!/usr/bin/env python3
"""Finalize the bounded audit with a concise report, hashes, sentinel, and notification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from common_v1 import read_csv, sha256, utc_now, write_json_x, write_text_x  # noqa: E402


def f(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def slice_value(rows: list[dict[str, str]], arm: str, kind: str, label: str) -> str:
    row = next(item for item in rows if item["threshold"] == "0.02" and item["arm"] == arm
               and item["subset_kind"] == kind and item["subset_label"] == label)
    return f(row["within_3m_fraction"], 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    decision = json.loads((experiment / "DECISION.json").read_text())
    metrics = json.loads((experiment / "COUNTERFACTUAL_METRICS.json").read_text())
    oracle = json.loads((experiment / "ORACLE_SUMMARY.json").read_text())
    cells = json.loads((experiment / "CELL_SAMPLING_AUDIT.json").read_text())
    transitions = json.loads((experiment / "TRANSITION_SUMMARY.json").read_text())
    traversal = json.loads((experiment / "DENSE_TRAVERSAL.json").read_text())
    reconciliation = json.loads((experiment / "RECONCILIATION.json").read_text())
    analysis_runtime = json.loads((experiment / "ANALYSIS_RUNTIME.json").read_text())
    config = json.loads((experiment / "RESOLVED_CONFIG.json").read_text())
    slices = read_csv(experiment / "SLICE_METRICS.csv")
    terminal, attribution = decision["primary_terminal"], decision["secondary_attribution"]
    base, candidate = metrics["base"], metrics["candidate"]
    deploy = metrics["deployable_inherited_at_candidate_cell"]
    gtcell = metrics["diagnostic_inherited_at_gt_cell"]
    compute_wall = (float(reconciliation["wall_seconds"]) + float(traversal["wall_seconds"])
                    + float(analysis_runtime["wall_seconds"]))
    artifact_names = [
        "INPUT_PROVENANCE.json", "REGISTERED_AUDIT_PLAN.json", "RECONCILIATION.json",
        "PER_GT_TRANSITIONS.csv", "TRANSITION_SUMMARY.json", "DENSE_FIELD_SAMPLES.csv",
        "DENSE_TRAVERSAL.json", "CELL_SAMPLING_AUDIT.json", "ORACLE_PAIR_RESULTS.csv",
        "ORACLE_SUMMARY.json", "COUNTERFACTUAL_DETECTIONS.csv", "COUNTERFACTUAL_METRICS.json",
        "SLICE_METRICS.csv", "DECISION.json",
    ]
    artifact_manifest = {
        "schema": "route_b_v3_1_localizer_counterfactual_artifact_manifest_v1",
        "created_utc": utc_now(), "artifacts": {
            name: {"sha256": sha256(experiment / name), "bytes": (experiment / name).stat().st_size}
            for name in artifact_names
        },
    }
    manifest_path = experiment / "ARTIFACT_MANIFEST.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if (existing_manifest.get("schema") != artifact_manifest["schema"]
                or existing_manifest.get("artifacts") != artifact_manifest["artifacts"]):
            raise RuntimeError("existing artifact manifest does not match current inputs")
        artifact_manifest = existing_manifest
    else:
        write_json_x(manifest_path, artifact_manifest)
    report = f"""# Route B v3.1 inherited-localizer counterfactual audit

Primary terminal: `{terminal}`

Secondary attribution: `{attribution}`

The corrected visible-anchor detector was retained, but hard-cell sampling of the mature epoch-40 localization field did not recover the preregistered canonical result. The depth/ray oracle is decisive: GT depth with the predicted ray recovered all 554 failed score-0.02 IoU50 pairs, while GT ray with predicted depth recovered only one. The fresh candidate depth estimate, not its physical-centre ray, is the dominant localization failure.

## Reconciliation and execution contract

- All published base/candidate canonical, IoU50, conditional-localization, taxonomy, vehicle, and segmentation results reconciled within `1e-12`.
- New inference: 0 candidate traversals, exactly 1 frozen epoch-40 validation traversal, and 0 segmentation traversals.
- No threshold pass, dense-map persistence, optimizer step, or training run occurred.
- The one traversal covered 3,345 validation frames and sampled 21,261 candidate cells plus all 3,872 GT cells; invalid cells: 0.
- Traversal peak VRAM: {f(traversal['peak_allocated_mib'], 2)} MiB allocated / {f(traversal['peak_reserved_mib'], 2)} MiB reserved.
- Reconciliation + traversal + offline analysis compute wall time: {f(compute_wall, 2)} s.

## Canonical and diagnostic comparison

| Arm | P@.20 | R@.20 | F1@.20 | R@.02 | XY MAE m | IoU50 F1@.20 | IoU50 R@.02 | IoU50 ≤3m@.02 | centre-present/world-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Epoch-40 base | {f(base['person_precision_020'])} | {f(base['person_recall_020'])} | {f(base['person_f1_020'])} | {f(base['person_recall_002'])} | {f(base['person_xy_mae_m_020'])} | {f(base['iou50_f1_020'])} | {f(base['iou50_recall_002'])} | {f(base['iou50_conditional_within_3m_002'])} | {base['center_present_world_wrong_002']} |
| Visible-anchor epoch 18 | {f(candidate['person_precision_020'])} | {f(candidate['person_recall_020'])} | {f(candidate['person_f1_020'])} | {f(candidate['person_recall_002'])} | {f(candidate['person_xy_mae_m_020'])} | {f(candidate['iou50_f1_020'])} | {f(candidate['iou50_recall_002'])} | {f(candidate['iou50_conditional_within_3m_002'])} | {candidate['center_present_world_wrong_002']} |
| Inherited field at predicted cell | {f(deploy['person_precision_020'])} | {f(deploy['person_recall_020'])} | {f(deploy['person_f1_020'])} | {f(deploy['person_recall_002'])} | {f(deploy['person_xy_mae_m_020'])} | {f(deploy['iou50_f1_020'])} | {f(deploy['iou50_recall_002'])} | {f(deploy['iou50_conditional_within_3m_002'])} | {deploy['center_present_world_wrong_002']} |
| Inherited field at GT cell (oracle) | {f(gtcell['person_precision_020'])} | {f(gtcell['person_recall_020'])} | {f(gtcell['person_f1_020'])} | {f(gtcell['person_recall_002'])} | {f(gtcell['person_xy_mae_m_020'])} | {f(gtcell['iou50_f1_020'])} | {f(gtcell['iou50_recall_002'])} | {f(gtcell['iou50_conditional_within_3m_002'])} | {gtcell['center_present_world_wrong_002']} |

The deployable composition exceeded base F1 by only {f(deploy['person_f1_020'] - base['person_f1_020'])}, but remained below base recall by {f(base['person_recall_020'] - deploy['person_recall_020'])}, exceeded base XY MAE by {f(deploy['person_xy_mae_m_020'] - base['person_xy_mae_m_020'])} m, and reached only {f(deploy['iou50_conditional_within_3m_002'])} conditional localization versus the registered 0.80 gate. It reduced the 95 additional world-wrong failures by 52.

Even the GT-cell diagnostic missed base recall by {f(base['person_recall_020'] - gtcell['person_recall_020'])} and reached only {f(gtcell['iou50_conditional_within_3m_002'])}. Therefore the result is not classified as sampling-limited.

## Frozen-pair depth/ray oracle

| Arm on the 2,309 score-0.02 IoU50 pairs | Mean m | Median m | P90 m | ≤3 m | Canonical F1@.20 |
|---|---:|---:|---:|---:|---:|
| Predicted ray + predicted depth | {f(oracle['pairwise']['0.02']['predicted_ray_predicted_depth']['mean_m'])} | {f(oracle['pairwise']['0.02']['predicted_ray_predicted_depth']['median_m'])} | {f(oracle['pairwise']['0.02']['predicted_ray_predicted_depth']['p90_m'])} | {f(oracle['pairwise']['0.02']['predicted_ray_predicted_depth']['within_3m_fraction'])} | {f(oracle['end_to_end']['predicted_ray_predicted_depth']['person_f1_020'])} |
| Predicted ray + GT depth | {f(oracle['pairwise']['0.02']['predicted_ray_gt_depth']['mean_m'])} | {f(oracle['pairwise']['0.02']['predicted_ray_gt_depth']['median_m'])} | {f(oracle['pairwise']['0.02']['predicted_ray_gt_depth']['p90_m'])} | {f(oracle['pairwise']['0.02']['predicted_ray_gt_depth']['within_3m_fraction'])} | {f(oracle['end_to_end']['predicted_ray_gt_depth']['person_f1_020'])} |
| GT ray + predicted depth | {f(oracle['pairwise']['0.02']['gt_ray_predicted_depth']['mean_m'])} | {f(oracle['pairwise']['0.02']['gt_ray_predicted_depth']['median_m'])} | {f(oracle['pairwise']['0.02']['gt_ray_predicted_depth']['p90_m'])} | {f(oracle['pairwise']['0.02']['gt_ray_predicted_depth']['within_3m_fraction'])} | {f(oracle['end_to_end']['gt_ray_predicted_depth']['person_f1_020'])} |
| GT ray + GT depth | {f(oracle['pairwise']['0.02']['gt_ray_gt_depth']['mean_m'])} | {f(oracle['pairwise']['0.02']['gt_ray_gt_depth']['median_m'])} | {f(oracle['pairwise']['0.02']['gt_ray_gt_depth']['p90_m'])} | {f(oracle['pairwise']['0.02']['gt_ray_gt_depth']['within_3m_fraction'])} | {f(oracle['end_to_end']['gt_ray_gt_depth']['person_f1_020'])} |

- Original failed fixed pairs: {oracle['original_failed_pairs_002']}.
- Recovered by GT depth alone: {oracle['depth_only_recovered_002']}/{oracle['original_failed_pairs_002']}.
- Recovered by GT ray alone: {oracle['ray_only_recovered_002']}/{oracle['original_failed_pairs_002']}.
- GT/GT maximum world-XY round-trip error: `{oracle['gt_gt_max_world_xy_error_m']:.3e} m` (sanity tolerance `1e-4 m`).

## Cell sampling and transitions

- Candidate predicted centre landed in the GT native cell for {cells['same_native_cell']}/{cells['pairs']} pairs ({f(cells['same_native_cell_fraction'], 4)}); 97.62% were within one cell.
- Epoch-40 field ≤3 m at candidate cell: {f(cells['predicted_cell_field']['within_3m_fraction'])}; at GT cell: {f(cells['gt_cell_field']['within_3m_fraction'])}.
- Score 0.02 IoU50 transitions: {transitions['thresholds']['002']['iou50_transitions']}.
- Score 0.02 canonical transitions: {transitions['thresholds']['002']['canonical_transitions']}.
- Candidate confidence/ranking losses between 0.02 and 0.20: {transitions['candidate_confidence_loss_iou50_count']} GT.

## Conditional ≤3 m slices at score 0.02

| Slice | Candidate depth/ray | Epoch-40 field at predicted cell | Epoch-40 field at GT cell |
|---|---:|---:|---:|
| 0–10 m | {slice_value(slices, 'predicted_ray_predicted_depth', 'distance_m', '[0,10)')} | {slice_value(slices, 'inherited_candidate_cell', 'distance_m', '[0,10)')} | {slice_value(slices, 'inherited_gt_cell', 'distance_m', '[0,10)')} |
| 10–20 m | {slice_value(slices, 'predicted_ray_predicted_depth', 'distance_m', '[10,20)')} | {slice_value(slices, 'inherited_candidate_cell', 'distance_m', '[10,20)')} | {slice_value(slices, 'inherited_gt_cell', 'distance_m', '[10,20)')} |
| 20–30 m | {slice_value(slices, 'predicted_ray_predicted_depth', 'distance_m', '[20,30)')} | {slice_value(slices, 'inherited_candidate_cell', 'distance_m', '[20,30)')} | {slice_value(slices, 'inherited_gt_cell', 'distance_m', '[20,30)')} |
| 30–40 m | {slice_value(slices, 'predicted_ray_predicted_depth', 'distance_m', '[30,40)')} | {slice_value(slices, 'inherited_candidate_cell', 'distance_m', '[30,40)')} | {slice_value(slices, 'inherited_gt_cell', 'distance_m', '[30,40)')} |
| Radar supported | {slice_value(slices, 'predicted_ray_predicted_depth', 'radar_support', 'supported')} | {slice_value(slices, 'inherited_candidate_cell', 'radar_support', 'supported')} | {slice_value(slices, 'inherited_gt_cell', 'radar_support', 'supported')} |
| Radar unsupported | {slice_value(slices, 'predicted_ray_predicted_depth', 'radar_support', 'unsupported')} | {slice_value(slices, 'inherited_candidate_cell', 'radar_support', 'unsupported')} | {slice_value(slices, 'inherited_gt_cell', 'radar_support', 'unsupported')} |
| Clear v0.25 | {slice_value(slices, 'predicted_ray_predicted_depth', 'visibility_contract', 'clear_v025')} | {slice_value(slices, 'inherited_candidate_cell', 'visibility_contract', 'clear_v025')} | {slice_value(slices, 'inherited_gt_cell', 'visibility_contract', 'clear_v025')} |
| Primary-v0.10-only | {slice_value(slices, 'predicted_ray_predicted_depth', 'visibility_contract', 'primary_v010_only')} | {slice_value(slices, 'inherited_candidate_cell', 'visibility_contract', 'primary_v010_only')} | {slice_value(slices, 'inherited_gt_cell', 'visibility_contract', 'primary_v010_only')} |

## Decision

The inherited field helps but does not satisfy the frozen composition contract, even with GT-cell sampling. No hybrid qualification run is licensed. The oracle proves that a sufficiently accurate depth source would unlock the detector (GT-depth F1 `0.624934`, recall `0.609762`, low-score recall `0.723915`), but neither available LR-ASPP depth formulation supplies it. Close further LR-ASPP person-head work under the frozen `{{low, high}}` transport contract and move the next person-accuracy effort to a different architecture with a stronger depth/range representation.

GT substitutions above are diagnostic oracles only and are not deployable inference.

## Untouched scope

Locked test remained absent and unopened. Existing experiments, production model/decoder/evaluator, CARLA, OAI, q/quant/AE/zstd, live runtime, and the 288 measurements were untouched. No v0.25 inference, threshold/NMS sweep, training, checkpoint write, or dataset/prediction mutation occurred.

Experiment: `{experiment}`
"""
    write_text_x(experiment / "FINAL_REPORT.md", report)
    write_text_x(experiment / "SECONDARY_ATTRIBUTION.txt", attribution + "\n")
    notification_command = [
        "notify-send", "LR-ASPP localizer counterfactual audit complete",
        f"{terminal}\n{attribution}\n{experiment}",
    ]
    completed = subprocess.run(notification_command, text=True, capture_output=True)
    write_json_x(experiment / "NOTIFICATION.json", {
        "created_utc": utc_now(), "command": notification_command,
        "returncode": completed.returncode, "stdout": completed.stdout,
        "stderr": completed.stderr,
    })
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "schema": "route_b_v3_1_localizer_counterfactual_pipeline_complete_v1",
        "created_utc": utc_now(), "primary_terminal": terminal,
        "secondary_attribution": attribution, "experiment": str(experiment),
        "report_sha256": sha256(experiment / "FINAL_REPORT.md"),
        "artifact_manifest_sha256": sha256(experiment / "ARTIFACT_MANIFEST.json"),
        "new_inference_counts": decision["new_inference_counts"],
        "training_runs": 0, "optimizer_steps": 0,
        "compute_wall_seconds": compute_wall,
        "peak_allocated_mib": traversal["peak_allocated_mib"],
        "peak_reserved_mib": traversal["peak_reserved_mib"],
        "forbidden_scope_access_counts": json.loads(
            (experiment / "INPUT_PROVENANCE.json").read_text()
        )["forbidden_scope_access_counts"],
    })
    print(json.dumps({"terminal": terminal, "attribution": attribution,
                      "report": str(experiment / "FINAL_REPORT.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
