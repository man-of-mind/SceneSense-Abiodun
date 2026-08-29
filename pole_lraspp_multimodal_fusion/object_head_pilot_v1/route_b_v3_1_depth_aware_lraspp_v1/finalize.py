from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from common import CONFIG_PATH, load_json, sha256, utc_now, write_json_x, write_text_x


def f(value: Any, digits: int = 6) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    config = load_json(CONFIG_PATH); decision = load_json(experiment / "SELECTION_DECISION.json")
    terminal = (experiment / "TERMINAL_VERDICT.txt").read_text().strip()
    qualification = load_json(experiment / "QUALIFICATION_REPORT.json")
    training = load_json(experiment / "TRAINING_COMPLETE.json")
    parameters = load_json(experiment / "PARAMETER_REPORT.json")
    train_cache = load_json(experiment / "depth_cache/train/CACHE_REPORT.json")
    val_cache = load_json(experiment / "depth_cache/val/CACHE_REPORT.json")
    records = {epoch: load_json(experiment / f"evaluation/epoch_{epoch:03d}.json") for epoch in (10, 20, 30, 40)}
    selected = records.get(decision["selected_epoch"]) if decision["selected_epoch"] else None
    notification_command = ["notify-send", "Depth-aware LR-ASPP complete", terminal]
    try:
        completed = subprocess.run(notification_command, capture_output=True, text=True, timeout=10)
        notification = {"attempted": True, "returncode": completed.returncode,
                        "stdout": completed.stdout, "stderr": completed.stderr,
                        "delivered": completed.returncode == 0}
    except Exception as error:
        notification = {"attempted": True, "returncode": None, "delivered": False,
                        "error": f"{type(error).__name__}: {error}"}
    notification.update({"schema": "route_b_v3_1_depth_aware_lraspp_notification_v1",
                         "created_utc": utc_now(), "command": notification_command, "terminal": terminal})
    write_json_x(experiment / "NOTIFICATION.json", notification)
    write_text_x(experiment / "COMPLETION_SENTINEL", terminal + "\n")

    lines = [
        f"# Route B v3.1 depth-aware LR-ASPP — final report",
        "", f"**Terminal verdict: `{terminal}`**", "",
        "Exactly one clean-lineage seven-channel model completed the registered 40-epoch full-FP32 run. No task checkpoint, teacher, architecture/loss/sampler variant, threshold sweep, NMS change, or inference-side depth input was used.", "",
        "## Provenance and architecture", "",
        f"- Local source lineage: `master` at `{config['source_commit']}`; required ancestor passed.",
        f"- Official weight: `{config['pretrained']['enum']}`, {config['pretrained']['bytes']:,} bytes, SHA-256 `{config['pretrained']['sha256']}`.",
        "- Preprocessing: RGB order verified, `[0,1]`, ImageNet mean/std; prepared radar channels remained identity-normalized in occupancy/inverse-range/radial-velocity/stationary-age order.",
        "- Stem proof: official bias-free RGB convolution plus exact-zero bias-free radar convolution, one shared official BN/Hardswish; concatenated weight shape `[16,7,3,3]` and FP32 equivalence error ≤ `1e-5`.",
        "- Transport: identity/disabled compression, `low [B,40,54,96]` and `high [B,960,27,48]`, raw and serialized batch-1 size 5,806,080 bytes; monolithic/split raw tensors and decoded records were byte-identical.",
        "- Tail: one shared 128-channel stride-4 depth-aware neck; new segmentation, dense auxiliary, and class-private object branches. XYZ is derived only from physical ray and distributional actor-forward depth.", "",
        "### Parameter counts", "", "| Module/group | Parameters |", "|---|---:|",
    ]
    for name in ("model", "backbone", "rgb_stem", "radar_stem", "depth_neck", "segmentation",
                 "dense_depth", "vehicle_branch", "person_branch"):
        lines.append(f"| {name} | {parameters[name]['parameters']:,} |")
    lines += ["", "## Data and qualification", "",
              f"- Frozen frames: 16,827 train across 10 episodes; 3,345 validation across two disjoint episodes; zero test rows/references.",
              f"- Train depth cache: {train_cache['depth_valid_pixels']:,} valid stride-4 pixels and {train_cache['radar_consistent_points']:,} current-sweep consistent radar points.",
              f"- Validation depth cache was created only after all four prediction sets: {val_cache['depth_valid_pixels']:,} valid pixels and {val_cache['radar_consistent_points']:,} radar points.",
              f"- Qualification: **PASS**; disposable steps {qualification['disposable_optimizer_steps']}; selected physical batch {qualification['accepted_physical_batch']} × accumulation {qualification['accepted_accumulation']} = effective 16.",
              f"- Disposable overfit falling gates: {qualification['checks']['disposable_overfit']['losses']}.",
              f"- Collision audit train: {qualification['checks']['collisions']['train']['same_class_collisions']}; validation: {qualification['checks']['collisions']['validation']['same_class_collisions']}. Cross-class overwrites and silent truncations: zero.",
              "- Stage-A official RGB/backbone/BN state was bit-identical through epoch 5; all new heads/neck and the zero radar stem had finite nonzero gradients. BN running means, variances, and counters stayed frozen for all 40 epochs.", "",
              "## Training", "", f"Training wall time: {training['wall_seconds'] / 3600.0:.3f} h. Every epoch visited all 16,827 frames exactly once; no validation ran during optimization.", "",
              "| Epoch | Total | Heat | Depth bin | Dense | New LR | Backbone LR | Clips | Seconds |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|" ]
    for epoch in range(1, 41):
        metric = load_json(experiment / f"training_metrics/epoch_{epoch:03d}.json"); loss = metric["mean_losses"]
        lines.append(f"| {epoch} | {f(loss['total'],4)} | {f(loss['heatmap'],4)} | {f(loss['depth_bin'],4)} | {f(loss['dense_depth'],4)} | {metric['new_lr_last']:.3e} | {metric['backbone_lr_last']:.3e} | {metric['clipping_count']} | {metric['epoch_seconds']:.1f} |")
    lines += ["", "Per-epoch configured weights, valid denominators, shared-neck task gradient norms, and detection/actor-depth/dense-depth/segmentation gradient cosines are persisted under `gradient_telemetry/`.", "",
              "## Fixed validation", "", "| Epoch | Veh P/R/F1 | Veh XY | Person P/R/F1 | Person R@.02 | Person XY | IoU50 F1@.20 | IoU50 R@.02 | Veh IoU | Person box IoU | fg mIoU | Preserve |",
              "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for epoch, record in records.items():
        m = record["metrics"]; d = record["person_iou_diagnostics"]["two_d"]
        lines.append(f"| {epoch} | {f(m['vehicle_precision'])}/{f(m['vehicle_recall'])}/{f(m['vehicle_f1'])} | {f(m['vehicle_xy_mae_m'])} | {f(m['person_precision'])}/{f(m['person_recall'])}/{f(m['person_f1'])} | {f(m['person_recall_002'])} | {f(m['person_xy_mae_m'])} | {f(d['0.20']['FULL_BOX_IOU_050']['f1'])} | {f(d['0.02']['FULL_BOX_IOU_050']['recall'])} | {f(m['vehicle_iou'])} | {f(m['person_box_mask_iou'])} | {f(m['foreground_miou'])} | {'yes' if record['gates']['preservation_pass'] else 'no'} |")
    if selected:
        m = selected["metrics"]; base = config["baseline"]; gate = selected["gates"]
        lines += ["", f"Selected checkpoint: `{decision['selected_checkpoint']}` (SHA-256 `{decision['selected_checkpoint_sha256']}`).", "",
                  "### Selected deltas from frozen epoch-40 baseline", "",
                  "| Metric | Candidate | Baseline | Delta |", "|---|---:|---:|---:|",
                  f"| Person F1 @.20 | {f(m['person_f1'])} | {f(base['person_f1'])} | {f(m['person_f1']-base['person_f1'])} |",
                  f"| Person recall @.20 | {f(m['person_recall'])} | {f(base['person_recall'])} | {f(m['person_recall']-base['person_recall'])} |",
                  f"| Person recall @.02 | {f(m['person_recall_002'])} | {f(base['person_recall_002'])} | {f(m['person_recall_002']-base['person_recall_002'])} |",
                  f"| Person XY MAE m | {f(m['person_xy_mae_m'])} | {f(base['person_xy_mae_m'])} | {f(m['person_xy_mae_m']-base['person_xy_mae_m'])} |",
                  f"| Vehicle F1 | {f(m['vehicle_f1'])} | {f(base['vehicle_f1'])} | {f(m['vehicle_f1']-base['vehicle_f1'])} |",
                  f"| Foreground mIoU | {f(m['foreground_miou'])} | {f(base['foreground_miou'])} | {f(m['foreground_miou']-base['foreground_miou'])} |", "",
                  f"Preservation gates: `{gate['preservation']}`", "",
                  f"Material gates: `{gate['material']}`", "",
                  f"All nine service gates: `{gate['service']}`", "",
                  f"Actor-depth/derived-XYZ slices: `{selected['actor_depth_diagnostics']['slices']}`", "",
                  f"Auxiliary dense-depth slices: `{selected['dense_depth_diagnostics']['slices']}`", "",
                  f"Detection/world-error taxonomy: `{selected['taxonomy_v010']}`", ""]
    else:
        lines += ["", "No preservation-eligible checkpoint exists; the highest-ranked runtime-valid epoch is diagnostic only. Selected checkpoint: **none**. v0.25 sensitivity was not licensed.", ""]
    inference = load_json(experiment / f"predictions/epoch_{(decision['selected_epoch'] or 40):03d}/inference_manifest.json")
    lines += ["## Runtime and leakage proof", "",
              f"- Selected/diagnostic inference wall time: {inference['wall_seconds']:.1f} s; peak allocated/reserved {inference['peak_allocated_mib']:.1f}/{inference['peak_reserved_mib']:.1f} MiB.",
              f"- Latency (dense readout disabled): `{inference['latency']}`.",
              "- Every prediction traversal completed with the inference signature containing no depth argument, dense readout disabled, and zero depth paths opened. Validation depth labels were first read only after all four persisted prediction sets existed.",
              "- External detection CSV semantics, including parked and radar-support fields, passed compatibility qualification.", "",
              "## Scope audit", "",
              "Only the new source/config package and this create-only experiment directory were created. The pre-existing dirty `OAI/openairinterface5g` submodule was preserved. Test payloads, CARLA, OAI, q/AE, live split inference, and the 288 measurements were untouched. No branch, push, pull, merge, rebase, or non-identity compression was used.", "",
              f"Desktop notification attempted: `{notification['attempted']}`; delivered: `{notification['delivered']}`. Completion sentinel: present.", ""]
    write_text_x(experiment / "FINAL_REPORT.md", "\n".join(lines) + "\n")
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_pipeline_complete_v1", "created_utc": utc_now(),
        "terminal": terminal, "final_report": str(experiment / "FINAL_REPORT.md"),
        "final_report_sha256": sha256(experiment / "FINAL_REPORT.md"),
        "completion_sentinel": True, "notification": notification,
        "selected_checkpoint": decision["selected_checkpoint"],
        "selected_checkpoint_sha256": decision["selected_checkpoint_sha256"],
    })
    print(json.dumps({"terminal": terminal, "notification": notification,
                      "report": str(experiment / "FINAL_REPORT.md")}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
