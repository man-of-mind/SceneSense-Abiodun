#!/usr/bin/env python3
"""Render the recovery-v2 final report from the produced artifacts. No metric is recomputed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common_v2 as C

ARM_ORDER = ("RAW_PRIMARY", "RECALL_PRESERVING_WORLD_NMS", "CALIBRATED_WORLD_NMS", "DIAGNOSTIC_CEILING_S002")


def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    args = parser.parse_args()
    exp = args.experiment_dir.resolve()
    summary = json.loads((exp / "validation_v2" / "all_epochs_metrics.json").read_text())
    registered = json.loads((exp / "registered_config.json").read_text())
    runtime = json.loads((exp / "training_runtime.json").read_text())
    chain = (exp / "CHAIN_STATUS.txt").read_text().strip()
    results = summary["results"]
    order = sorted(results, key=lambda key: results[key]["epoch"])

    lines = []
    add = lines.append
    add("# Route B Faster R-CNN recovery-v2 — final bounded run\n")
    add(f"> ## `{summary['terminal']}`\n")
    add("One bounded 12-epoch run. Architecture FIXED. Warm start = the ORIGINAL "
        "`route_b_fasterrcnn_radar_roi_v1/20260826_224720` epoch-12 checkpoint, SHA-256 "
        f"`{registered['parent_checkpoint_sha256']}` verified in-run before any weight moved. "
        "The locked test split is absent and was never opened. No validation row and no validation "
        "FP row was used as a training example.\n")

    add("## Run cost\n")
    add(f"| quantity | value |\n|---|---|")
    add(f"| training wall time | {runtime['runtime_seconds']:.1f} s ({runtime['runtime_seconds']/60:.2f} min) |")
    add(f"| training peak VRAM allocated | {runtime['peak_allocated_mib']:.1f} MiB |")
    add(f"| training peak VRAM reserved | {runtime['peak_reserved_mib']:.1f} MiB |")
    eval_seconds = sum(results[key]["runtime_seconds"] for key in order)
    eval_peak = max(results[key]["peak_allocated_mib"] for key in order)
    add(f"| validation wall time (3 checkpoints) | {eval_seconds:.1f} s ({eval_seconds/60:.2f} min) |")
    add(f"| validation peak VRAM allocated | {eval_peak:.1f} MiB |")
    add(f"| chain status | `{chain.replace(chr(10), ' | ')}` |")
    audit = registered["freeze_audit"]
    add(f"| trainable parameters | {audit['trainable_parameters']:,} / {audit['total_parameters']:,} "
        f"({100*audit['trainable_parameters']/audit['total_parameters']:.2f}%) |")
    add(f"| trainable modules | {audit['trainable_module_count']} |")
    add(f"| frozen modules | {audit['frozen_module_count']} |")
    add(f"| batch size | {registered['config']['batch_size']} (already-proven; no sweep) |\n")

    add("## Freeze audit — protected branches carry zero trainable parameters\n")
    add("| branch | parameters | trainable |\n|---|---:|---:|")
    for name, item in audit["protected_branches"].items():
        add(f"| `{name}` (frozen) | {item['params']:,} | **{item['trainable']}** |")
    if audit.get("frozen_box_head_copy"):
        copy = audit["frozen_box_head_copy"]
        add(f"| frozen deepcopy of the original `box_head` | {copy['params']:,} | **{copy['trainable']}** |")
    for name, item in audit["trainable_branches"].items():
        add(f"| `{name}` (trainable) | {item['params']:,} | {item['trainable']:,} |")
    add(f"\nUnclassified parameters: `{audit['unclassified_parameter_names'] or 'none'}`. "
        f"Frozen BatchNorm affine parameters: {audit.get('frozen_batchnorm_affine_parameters', 0):,} "
        "(held frozen exactly as in the v1 run that produced the warm start).\n")

    add("## Registered anchor decision\n")
    anchor = registered["anchor_registration"]
    add(f"- before: sizes `{anchor['before']['sizes']}`, {anchor['before']['anchors_per_location']} anchors/location")
    add(f"- after: sizes `{anchor['after']['sizes']}`, {anchor['after']['anchors_per_location']} anchors/location "
        f"(verified {anchor['anchors_per_location_verified']})")
    add(f"- **only new anchor scale pyramid-wide: {anchor['new_scales_introduced']} at {anchor['new_scale_level']}**")
    add(f"- reason: {anchor['reason']}")
    add(f"- anchor sweep performed: `{anchor['sweep_performed']}`\n")

    add("## Detection / localization by epoch and registered operating point\n")
    add("| epoch | arm | class | P | R | F1 | XY MAE m | dim MAE m | yaw MAE deg | dup FP/frame | FP/frame |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in order:
        result = results[key]
        for arm in ARM_ORDER:
            for class_name in C.CLASSES:
                m = result["arms"][arm]["classes"][class_name]
                add(f"| {result['epoch']} | {arm} | {class_name} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                    f"{fmt(m['f1'])} | {fmt(m['xy_mae_m'])} | {fmt(m['dimension_mae_m'])} | "
                    f"{fmt(m['yaw_mae_deg'],2)} | {fmt(m['duplicate_fp_per_frame'])} | {fmt(m['fp_per_frame'])} |")
    add("")

    add("## Segmentation, proposal recall and detection ceiling\n")
    add("| epoch | veh IoU | person box-mask IoU | bg IoU | mIoU | pixel acc | prop recall veh @0.5 | "
        "prop recall per @0.5 | ceiling veh @0.02 | ceiling per @0.02 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in order:
        r = results[key]
        s, p, c = r["segmentation"], r["rpn_proposal_recall_class_agnostic"], r["final_detection_recall_ceiling_score002"]
        add(f"| {r['epoch']} | {fmt(s['vehicle_iou'])} | {fmt(s['person_box_mask_iou'])} | {fmt(s['background_iou'])} | "
            f"{fmt(s['miou'])} | {fmt(s['pixel_accuracy'])} | {fmt(p['vehicle']['iou0.5'])} | "
            f"{fmt(p['person']['iou0.5'])} | {fmt(c['vehicle'])} | {fmt(c['person'])} |")
    add("\nProposal recall is class-agnostic RPN coverage at image IoU 0.5 and is **diagnostic evidence, "
        "not an exact mathematical ceiling** for the 3 m world metric.\n")
    add("| epoch | class | prop recall @0.3 | @0.5 | @0.7 | eligible GT |")
    add("|---|---|---:|---:|---:|---:|")
    for key in order:
        r = results[key]
        for class_name in C.CLASSES:
            p = r["rpn_proposal_recall_class_agnostic"][class_name]
            add(f"| {r['epoch']} | {class_name} | {fmt(p['iou0.3'])} | {fmt(p['iou0.5'])} | {fmt(p['iou0.7'])} | "
                f"{p['eligible_gt']} |")
    add("")

    add("## Gate evaluation — 9 pre-registered operating points (3 arms x 3 epochs)\n")
    gate_names = None
    for key in order:
        for arm in ("RAW_PRIMARY", "RECALL_PRESERVING_WORLD_NMS", "CALIBRATED_WORLD_NMS"):
            gate_names = gate_names or list(results[key]["gate_evaluation"][arm]["checks"])
    add("| epoch | arm | " + " | ".join(gate_names) + " | ALL |")
    add("|---|---|" + "---:|" * len(gate_names) + "---|")
    for key in order:
        r = results[key]
        for arm in ("RAW_PRIMARY", "RECALL_PRESERVING_WORLD_NMS", "CALIBRATED_WORLD_NMS"):
            gate = r["gate_evaluation"][arm]
            cells = []
            for name in gate_names:
                check = gate["checks"][name]
                cells.append(f"{fmt(check['value'],4)} {'PASS' if check['pass'] else '**FAIL**'}")
            add(f"| {r['epoch']} | {arm} | " + " | ".join(cells) +
                f" | {'**PASS**' if gate['all_gates_pass'] else 'FAIL'} |")
    add("")
    add(f"Passing operating points: `{summary['passing_operating_points'] or 'none'}`\n")
    add(f"Grid extended after seeing results: `{summary['grid_extended_after_results']}`. "
        f"Candidate operating points fixed before training: {summary['candidate_operating_points_evaluated']}.\n")

    collision = summary["collision_window_sensitivity"]
    add("## Collision-window-excluded sensitivity\n")
    add(f"Status: **{collision['status']}**. {collision['note']}\n")

    add("## Objective registration (fixed before training)\n")
    add("```json")
    add(json.dumps(registered["objective_registration"], indent=2, sort_keys=True))
    add("```\n")

    if summary["terminal"] == "FRCNN_BASE_VALIDATION_CANDIDATE_READY_FOR_LOCKED_TEST":
        add("## Candidate for the locked test (NOT opened here)\n")
        for checkpoint_key, arm in summary["passing_operating_points"]:
            r = results[checkpoint_key]
            add(f"- checkpoint `{r['checkpoint']}`")
            add(f"  - SHA-256 `{r['checkpoint_sha256']}`")
            add(f"  - epoch {r['epoch']}, operating point `{arm}`")
            spec = summary["registered_operating_points"][arm]
            add(f"  - decoder settings: score floor {registered['config']['evaluation']['score_floor']}, "
                f"class-aware box NMS IoU {registered['config']['evaluation']['box_nms_iou']}, "
                f"detections/image {registered['config']['evaluation']['detections_per_image']}, "
                f"arm thresholds {spec['thresholds']}, world NMS {spec['world_nms']}")
            for class_name in C.CLASSES:
                m = r["arms"][arm]["classes"][class_name]
                add(f"  - {class_name}: P={fmt(m['precision'])} R={fmt(m['recall'])} F1={fmt(m['f1'])} "
                    f"XY MAE={fmt(m['xy_mae_m'])} m")
            s = r["segmentation"]
            add(f"  - segmentation: vehicle IoU={fmt(s['vehicle_iou'])}, person box-mask IoU="
                f"{fmt(s['person_box_mask_iou'])}, mIoU={fmt(s['miou'])}")
        add("\nThe locked test remains absent and unopened.\n")
    else:
        add("## Terminal consequence\n")
        add("No checkpoint/operating point passes every gate. Per the bounded plan this stops completely: "
            "no new architecture, no second run, no relaxed threshold, no q/AE training and no additional "
            "diagnostic proposal. The locked test remains absent and unopened.\n")

    (exp / "FRCNN_RECOVERY_V2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (exp / "TERMINAL_VERDICT.txt").write_text(summary["terminal"] + "\n", encoding="utf-8")
    print(f"wrote {exp / 'FRCNN_RECOVERY_V2_REPORT.md'}")
    print(f"TERMINAL {summary['terminal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
