#!/usr/bin/env python3
"""Build the required final clean-model report from immutable run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

FILES = [
    "__init__.py",
    "centernet_model_v1.py",
    "launch_check_v1.py",
    "train_entry_v1.py",
    "eval_entry_v1.py",
    "evaluate_checkpoint_v1.py",
    "gate_and_select_v1.py",
    "make_trial_v1.py",
    "report_v1.py",
    "run_clean_chain_v1.sh",
    "configs/route_b_centernet_clean_v1.yaml",
    "configs/resnet34_fpn_centerfusion_v1.json",
    "PROVENANCE.md",
    "licenses/torchvision-BSD-3-Clause.txt",
]


def _metrics_rows(exp_dir: Path) -> List[Dict[str, str]]:
    path = exp_dir / "metrics" / "resnet34_fpn_centerfusion_v1_metrics.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _epoch_row(epoch: int, s20: Dict[str, Any], s02: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "score_0.20": {
            cls: {
                key: s20[f"{cls}_{key}"]
                for key in ("precision", "recall", "f1", "xy_mae_m", "dimension_mae_m")
            }
            for cls in ("vehicle", "person")
        },
        "score_0.02_recall_ceiling": {
            "vehicle": s02["vehicle_recall"],
            "person": s02["person_recall"],
        },
        "segmentation": {
            key: s20[key] for key in ("vehicle_iou", "person_iou", "miou")
        },
        "checkpoint": s20["checkpoint"],
        "checkpoint_sha256": s20["checkpoint_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    exp_dir = args.experiment_dir.resolve()
    verdict = (exp_dir / "TERMINAL_VERDICT.txt").read_text(encoding="utf-8").strip()
    if verdict == "IMPLEMENTATION_BLOCKED" and not (exp_dir / "decision" / "pilot_gate_v1.json").is_file():
        reason_path = exp_dir / "BLOCKED_REASON.txt"
        blocked_report = {
            "verdict": verdict,
            "reason": reason_path.read_text(encoding="utf-8").strip() if reason_path.is_file() else "unknown",
            "files_created_or_changed": FILES,
            "locked_test_opened": False,
            "q_ae_work_may_begin": False,
            "q_ae_statement": "No; implementation did not pass the clean launch/pilot boundary.",
        }
        args.output_json.write_text(json.dumps(blocked_report, indent=2, sort_keys=True), encoding="utf-8")
        args.output_md.write_text(
            f"# IMPLEMENTATION_BLOCKED\n\n{blocked_report['reason']}\n\n"
            "The locked test split remained absent and unopened. q/AE work may not begin.\n",
            encoding="utf-8",
        )
        return 0
    launch = json.loads((exp_dir / "launch_gate.json").read_text(encoding="utf-8"))
    final_path = exp_dir / "decision" / "final_selection_v1.json"
    pilot_path = exp_dir / "decision" / "pilot_gate_v1.json"
    decision = json.loads(
        (final_path if final_path.is_file() else pilot_path).read_text(encoding="utf-8")
    )
    if final_path.is_file():
        epoch_records = [
            _epoch_row(int(row["epoch"]), row["s020"], row["s002"])
            for row in decision["epochs"]
        ]
        selected_epoch = int(decision["selected_epoch"])
        selected_checkpoint = decision["selected_checkpoint"]
        selected_sha = decision["selected_checkpoint_sha256"]
        service_targets = decision["service_targets"]
    else:
        epoch_records = [_epoch_row(4, decision["epoch4_s020"], decision["epoch4_s002"])]
        selected_epoch = None
        selected_checkpoint = decision["epoch4_s020"]["checkpoint"]
        selected_sha = decision["epoch4_s020"]["checkpoint_sha256"]
        service_targets = "not reached because the four-epoch continuation gate failed"

    metric_rows = _metrics_rows(exp_dir)
    training_seconds = sum(float(row.get("epoch_seconds") or 0.0) for row in metric_rows)
    peak_train_allocated = max(
        [float(row.get("cuda_max_memory_allocated_mib") or 0.0) for row in metric_rows] or [0.0]
    )
    peak_train_reserved = max(
        [float(row.get("cuda_max_memory_reserved_mib") or 0.0) for row in metric_rows] or [0.0]
    )
    started = float((exp_dir / "chain_started_unix.txt").read_text(encoding="utf-8"))
    report = {
        "verdict": verdict,
        "architecture": {
            "name": "resnet34_fpn_centerfusion_v1",
            "rgb": "ImageNet-pretrained ResNet34 C2-C5 plus 128-channel FPN; primary CenterNet head at input stride 4",
            "radar": "separate four-channel Conv-GN radar pyramid/FPN; no radar-as-colour stem concatenation",
            "fusion": "radar-conditioned second-stage refinement of both heatmaps and all XYZ/dimension/yaw/parked/radar-support/2D-size regression maps",
            "segmentation": "fused stride-4 lightweight decoder for background/vehicle/person",
            "object_channel_order": [
                "vehicle_heatmap", "person_heatmap", "local_x", "local_y", "local_z",
                "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos", "parked",
                "radar_support", "normalized_2d_width", "normalized_2d_height",
            ],
            "ue_edge_split": "UE encode_front(rgb, radar) -> {rgb_p2, radar_p2}; edge decode_tail(feature_bundle, out_hw) -> {out, object}. No modality side channel.",
        },
        "provenance": {
            "source": "torchvision",
            "version": "0.25.0.dev20251117+cu128",
            "git_revision": "4efae90d072d0d11e244d6e213208b357f89efe7",
            "weight_enum": "ResNet34_Weights.IMAGENET1K_V1",
            "weight_url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "weight_sha256": "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
            "license": "BSD-3-Clause (notice retained in licenses/torchvision-BSD-3-Clause.txt)",
            "centerfusion_code_or_weights_reused": False,
        },
        "files_created_or_changed": FILES,
        "parameters": launch["parameters"],
        "launch_gate": launch,
        "evaluated_epochs": epoch_records,
        "selection": {
            "rule": decision.get("selection_rule", "pilot stopped; no final checkpoint selection"),
            "selected_epoch": selected_epoch,
            "checkpoint": selected_checkpoint,
            "checkpoint_sha256": selected_sha,
        },
        "service_targets": service_targets,
        "runtime": {
            "wall_seconds": time.time() - started,
            "training_epoch_seconds_sum": training_seconds,
            "peak_vram_mib": max(float(launch["peak_vram_mib"]), peak_train_allocated),
            "peak_vram_reserved_mib": max(float(launch["peak_vram_reserved_mib"]), peak_train_reserved),
        },
        "q_ae_work_may_begin": verdict == "CENTERNET_BASE_SERVICE_READY",
        "q_ae_statement": (
            "Yes; the clean model met every service-quality target."
            if verdict == "CENTERNET_BASE_SERVICE_READY"
            else "No; q/AE work remains stopped pending review because the clean model did not meet every service-quality target."
        ),
        "locked_test_opened": False,
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# {verdict}", "", "## Architecture and split", "",
        report["architecture"]["rgb"] + ".", report["architecture"]["radar"] + ".",
        report["architecture"]["fusion"] + ".", report["architecture"]["ue_edge_split"], "",
        "## Evaluated epochs", "",
        "| Epoch | Vehicle P/R/F1 | Person P/R/F1 | Recall ceiling V/P | XY MAE V/P (m) | Dim MAE V/P (m) | IoU V/P | mIoU |",
        "|---:|---|---|---|---|---|---|---:|",
    ]
    for row in epoch_records:
        v, p = row["score_0.20"]["vehicle"], row["score_0.20"]["person"]
        ceiling, seg = row["score_0.02_recall_ceiling"], row["segmentation"]
        lines.append(
            f"| {row['epoch']} | {v['precision']:.4f}/{v['recall']:.4f}/{v['f1']:.4f} | "
            f"{p['precision']:.4f}/{p['recall']:.4f}/{p['f1']:.4f} | "
            f"{ceiling['vehicle']:.4f}/{ceiling['person']:.4f} | "
            f"{v['xy_mae_m']:.4f}/{p['xy_mae_m']:.4f} | "
            f"{v['dimension_mae_m']:.4f}/{p['dimension_mae_m']:.4f} | "
            f"{seg['vehicle_iou']:.4f}/{seg['person_iou']:.4f} | {seg['miou']:.4f} |"
        )
    lines.extend([
        "", "## Selection and resources", "",
        f"- Checkpoint: `{selected_checkpoint}`",
        f"- SHA-256: `{selected_sha}`",
        f"- Trainable parameters: {launch['parameters']['trainable']:,}",
        f"- Wall runtime: {report['runtime']['wall_seconds']:.1f} s",
        f"- Peak VRAM allocated/reserved: {report['runtime']['peak_vram_mib']:.1f}/{report['runtime']['peak_vram_reserved_mib']:.1f} MiB",
        "", "## Provenance and licence", "",
        f"torchvision `{report['provenance']['version']}` at `{report['provenance']['git_revision']}`; "
        f"official ResNet34 weight SHA-256 `{report['provenance']['weight_sha256']}`; BSD-3-Clause notice retained.",
        "", "## q/AE decision", "", report["q_ae_statement"], "",
        "The locked test split remained absent and unopened.", "",
    ])
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "report": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
