from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from common import CONFIG_PATH, load_json, read_csv, sha256, utc_now, write_json_x, write_text_x
from data import DepthCache, TrainingDataset, collate_training, load_objects, load_visible_anchors


TERMINAL = "DEPTH_AWARE_RUNTIME_FAILURE"
FAILED_EPOCH = 1
FAILED_BATCH = 14


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True,
    ).stdout.strip()


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()) if value.dtype.is_floating_point else True,
    }
    if value.numel() and value.dtype.is_floating_point:
        result.update({"minimum": float(value.min()), "maximum": float(value.max())})
    return result


def audit_failed_batch(experiment: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read targets only; deliberately do not instantiate or advance an optimizer."""
    dataset_root = (Path.cwd() / config["dataset_root"]).resolve(strict=True)
    rows = [row for row in read_csv(dataset_root / "dataset/manifest.csv") if row["split"] == "train"]
    generator = torch.Generator().manual_seed(int(config["scientific_seed"]) + FAILED_EPOCH)
    permutation = torch.randperm(len(rows), generator=generator).tolist()
    start = (FAILED_BATCH - 1) * 16
    indices = permutation[start:start + 16]
    cache = DepthCache(experiment / "depth_cache/train", rows)
    dataset = TrainingDataset(
        dataset_root, rows, load_objects(dataset_root),
        load_visible_anchors(Path(config["visible_anchor_cache"])), cache,
        int(config["scientific_seed"]),
    )
    dataset.set_epoch(FAILED_EPOCH)
    batch = collate_training([dataset[index] for index in indices])
    owner_summaries = {
        class_name: {name: tensor_summary(value) for name, value in fields.items()}
        for class_name, fields in batch["owners"].items()
    }
    all_finite = all(
        tensor_summary(batch[name])["finite"]
        for name in ("input", "heatmap", "dense_depth")
    )
    all_finite = all_finite and all(
        bool(torch.isfinite(points).all()) for points in batch["radar_points"]
    )
    all_finite = all_finite and all(
        summary["finite"] for fields in owner_summaries.values() for summary in fields.values()
    )
    return {
        "schema": "route_b_v3_1_depth_aware_lraspp_failed_batch_input_audit_v1",
        "created_utc": utc_now(),
        "method": "deterministic sampler reconstruction plus read-only dataset/target loading; no model, backward, or optimizer step",
        "sampler_seed": int(config["scientific_seed"]) + FAILED_EPOCH,
        "epoch": FAILED_EPOCH,
        "batch": FAILED_BATCH,
        "dataset_indices": indices,
        "sample_ids": list(batch["sample_id"]),
        "all_inputs_and_targets_finite": all_finite,
        "input": tensor_summary(batch["input"]),
        "segmentation_shape": list(batch["segmentation"].shape),
        "segmentation_labels": sorted(torch.unique(batch["segmentation"]).tolist()),
        "heatmap": tensor_summary(batch["heatmap"]),
        "dense_depth": tensor_summary(batch["dense_depth"]),
        "dense_valid_pixels": int(batch["dense_valid"].sum()),
        "radar_points": sum(len(points) for points in batch["radar_points"]),
        "radar_points_all_finite": all(bool(torch.isfinite(points).all()) for points in batch["radar_points"]),
        "owners": owner_summaries,
        "same_class_collisions_in_batch": len(batch["collisions"]),
        "validation_depth_opened": False,
    }


def markdown_report(
    experiment: Path,
    config: dict[str, Any],
    provenance: dict[str, Any],
    qualification: dict[str, Any],
    parameters: dict[str, Any],
    cache: dict[str, Any],
    failure: dict[str, Any],
    notification: dict[str, Any],
) -> str:
    checks = qualification["checks"]
    split = checks["split"]
    collisions = checks["collisions"]
    overfit = checks["disposable_overfit"]["losses"]
    package = CONFIG_PATH.parent
    source_files = sorted(
        path.relative_to(Path.cwd()).as_posix()
        for path in package.iterdir()
        if path.is_file() and path.name != "__pycache__"
    )
    lines = [
        "# Route B v3.1 depth-aware LR-ASPP — terminal report",
        "",
        f"**Terminal verdict: `{TERMINAL}`**",
        "",
        "Exactly one clean-lineage scientific model was started. It became non-finite at epoch 1, batch 14, before the first epoch checkpoint. The registered recovery exception therefore cannot apply: there is no prior exact checkpoint and the failed in-memory optimizer state no longer exists. No retry or scientific change was made. This runtime-failed attempt is not scientific evidence against LR-ASPP.",
        "",
        "## Lineage, model, and pretrained weight",
        "",
        f"- Frozen source lineage: local `master` at `{config['source_commit']}`; required-ancestor check passed.",
        f"- Official backbone: `{config['pretrained']['enum']}` from `{config['pretrained']['url']}`; {config['pretrained']['bytes']:,} bytes; SHA-256 `{config['pretrained']['sha256']}`.",
        f"- Software: Python `{provenance['software']['python']}`, PyTorch `{provenance['software']['pytorch']}`, torchvision `{provenance['software']['torchvision']}`.",
        "- Architecture: one dilated MobileNetV3-Large fused trunk; a shared low/high depth-aware stride-4 neck; segmentation and training-only dense-depth readouts; private vehicle/person heatmap and factorized geometry branches. Actor XYZ is derived only from physical-centre ray plus the 32-bin log-depth distribution—there is no learned XYZ head.",
        "",
        "| Module | Parameters |",
        "|---|---:|",
    ]
    for name in ("model", "backbone", "rgb_stem", "radar_stem", "depth_neck", "segmentation", "dense_depth", "vehicle_branch", "person_branch"):
        lines.append(f"| {name} | {parameters[name]['parameters']:,} |")
    lines.extend(["", "| Optimizer group | Tensors | Parameters |", "|---|---:|---:|"])
    for name, record in parameters["optimizer_groups"].items():
        lines.append(f"| {name} | {record['tensors']:,} | {record['parameters']:,} |")
    lines.extend([
        "",
        "## Input, stem, and split proofs",
        "",
        "- Deployable input is seven channels: RGB in RGB order, scaled to `[0,1]` and ImageNet-normalized, followed by the prepared identity-normalized radar occupancy, inverse-range, radial-velocity, and stationary-age channels. The real-sample PIL RGB versus OpenCV BGR-reversal check passed.",
        f"- The bias-free official RGB convolution and exact-zero bias-free radar convolution concatenate to `{checks['stem_equivalence']['weight_shape']}`. Direct and concatenated FP32 outputs were equal with maximum absolute delta `{checks['stem_equivalence']['max_abs_delta']}`.",
        f"- Transport is identity/disabled: `low {split['shapes']['low']}` and `high {split['shapes']['high']}`, both FP32; raw and serialized batch-1 sizes are each {split['raw_bytes']:,} bytes.",
        f"- Tail input keys were exactly `{split['tail_inputs']}`; all raw outputs were `torch.equal`; {checks['decoded_parity']['records']} decoded records were byte-identical and externally schema-compatible.",
        "- The inference dataset/model signatures have no depth-label input. A nonexistent in-memory depth-path sentinel caused zero open attempts and byte-identical input/prediction behavior.",
        "",
        "## Data, cache, radar, and qualification",
        "",
        f"- Authoritative manifest SHA-256 `{provenance['manifest_sha256']}`: {provenance['train_frames']:,} train frames/10 episodes and {provenance['validation_frames']:,} validation frames/two disjoint episodes; zero test rows.",
        f"- v0.10 object hashes: train `{provenance['v010_train_objects']['sha256']}`, validation `{provenance['v010_validation_objects']['sha256']}`. Visible-anchor SHA-256 `{provenance['visible_anchor_cache']['sha256']}`.",
        f"- Train cache: {cache['entries']:,} `sample_id` entries, {cache['depth_valid_pixels']:,} finite valid depth cells, and {cache['radar_consistent_points']:,} retained current-sweep radar consistency points. Depth F16 SHA-256 `{cache['files']['depth_forward_f16.bin']['sha256']}`; valid-mask SHA-256 `{cache['files']['valid_u8.bin']['sha256']}`; radar-cache SHA-256 `{cache['files']['radar_consistency_f32.bin']['sha256']}`.",
        f"- Depth synchronization had exact frame IDs and zero timestamp delta. Retained radar `camera_depth_m` delta was `{cache['radar_camera_depth_max_abs_delta_m']}` m; current-sweep transform max delta was `{cache['radar_current_sweep_transform_max_abs_delta_m']:.8f}` m.",
        f"- Qualification passed in {qualification['wall_seconds']:.1f} s. Physical batch {qualification['accepted_physical_batch']} × accumulation {qualification['accepted_accumulation']} was accepted at {checks['memory']['attempts'][0]['allocated_mib']:.1f} MiB allocated/{checks['memory']['attempts'][0]['reserved_mib']:.1f} MiB reserved.",
        f"- The disposable 80-step overfit gates fell: person heatmap `{overfit['person_heatmap']['first5_mean']:.6f}` → `{overfit['person_heatmap']['last5_mean']:.6f}`, person actor depth `{overfit['person_actor_depth']['first5_mean']:.6f}` → `{overfit['person_actor_depth']['last5_mean']:.6f}`, dense depth `{overfit['dense_depth']['first5_mean']:.6f}` → `{overfit['dense_depth']['last5_mean']:.6f}`. All disposable state was discarded.",
        f"- Stage-A clone proof kept official state bit-identical and gave finite nonzero gradients to every required new group. Geometry round-trip maximum error was `{checks['actor_geometry']['max_roundtrip_abs_error_m']:.3e}` m.",
        f"- Same-class collisions: train person {collisions['train']['same_class_collisions']['person']}/{collisions['train']['eligible']['person']} and vehicle {collisions['train']['same_class_collisions']['vehicle']}/{collisions['train']['eligible']['vehicle']}; validation person {collisions['validation']['same_class_collisions']['person']}/{collisions['validation']['eligible']['person']} and vehicle {collisions['validation']['same_class_collisions']['vehicle']}/{collisions['validation']['eligible']['vehicle']}. Cross-class overwrites and silent truncations were zero.",
        "",
        "## Scientific runtime failure",
        "",
        f"- Exact exception: `{failure['exception']}`.",
        f"- Completed epoch boundaries: **0**. Optimizer updates before the failed forward/loss check: **13** (control-flow inference). Atomic checkpoints: **0**. Exact resume possible: **no**.",
        f"- A read-only reconstruction of the failed batch found all inputs and targets finite: {failure['failed_batch_input_audit']['all_inputs_and_targets_finite']}; {len(failure['failed_batch_input_audit']['sample_ids'])} samples, {failure['failed_batch_input_audit']['dense_valid_pixels']:,} valid dense cells, and {failure['failed_batch_input_audit']['radar_points']:,} consistent radar points. It performed no forward, backward, or optimizer step.",
        "- No per-epoch loss, denominator, LR, clipping, or gradient-telemetry record exists because the exception preceded the first epoch boundary. Training wall time was not durably instrumented before the exception; the terminalization upper bound from `TRAINING_STARTED` creation is recorded in `SCIENTIFIC_RUNTIME_FAILURE.json`.",
        "",
        "## Validation, selection, and gates",
        "",
        "| Epoch | Prediction | v0.10 evaluation | Reason |",
        "|---:|---|---|---|",
        "| 10 | not run | unavailable | checkpoint absent |",
        "| 20 | not run | unavailable | checkpoint absent |",
        "| 30 | not run | unavailable | checkpoint absent |",
        "| 40 | not run | unavailable | checkpoint absent |",
        "",
        "Baseline/reference deltas, actor-depth and derived-XYZ slices, auxiliary dense-depth slices, detection/world-error taxonomy, latency, and the nine service/material gates are unavailable because there is no completed checkpoint or prediction. Forty-epoch preservation eligibility is false by construction. Selected checkpoint: **none**. v0.25 sensitivity was not licensed or run. Validation depth was never opened and no validation cache was built.",
        "",
        "## Scope and durable completion",
        "",
        f"- Current branch at terminalization: `{git('branch', '--show-current')}`. Protected pre-existing dirty path `OAI/openairinterface5g` remains the sole out-of-scope status entry and was not modified by this work.",
        "- Test payloads, CARLA, OAI execution, q/AE, live split inference, and the 288 measurements were untouched. No branch, push, pull, merge, rebase, architecture/loss/sampler variant, scientific retry, or non-identity compression was used.",
        "- Commit allowlist (and nothing else):",
        "",
    ])
    lines.extend(f"  - `{path}`" for path in source_files)
    lines.append(f"  - `{experiment.relative_to(Path.cwd()).as_posix()}/FINAL_REPORT.md`")
    lines.extend([
        "",
        "The terminal report is included in the local master commit; its commit hash is reported in the external handoff to avoid a self-referential hash. Checkpoints, predictions, caches, and JSON payloads are intentionally uncommitted.",
        "",
        f"Desktop notification attempted: `{notification['attempted']}`; delivered: `{notification['delivered']}`; return code: `{notification['returncode']}`. Completion sentinel: present.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--notification-returncode", required=True, type=int)
    parser.add_argument("--notification-stderr", default="")
    args = parser.parse_args()
    experiment = args.experiment.resolve(strict=True)
    required = ("QUALIFICATION_COMPLETE", "QUALIFICATION_REPORT.json", "TRAINING_STARTED.json")
    if not all((experiment / name).exists() for name in required):
        raise RuntimeError("runtime-failure finalization requires qualified, started scientific attempt")
    checkpoints = sorted((experiment / "checkpoints").glob("epoch_*.pt"))
    metrics = sorted((experiment / "training_metrics").glob("epoch_*.json"))
    predictions = sorted((experiment / "predictions").glob("**/*")) if (experiment / "predictions").exists() else []
    if checkpoints or metrics or predictions:
        raise RuntimeError("failure classification expects no completed epoch checkpoint, metric, or prediction")
    terminal_artifacts = (
        "SCIENTIFIC_RUNTIME_FAILURE.json", "TRAINING_COMPLETE.json", "SELECTION_DECISION.json",
        "TERMINAL_VERDICT.txt", "NOTIFICATION.json", "COMPLETION_SENTINEL", "FINAL_REPORT.md",
        "PIPELINE_COMPLETE.json",
    )
    existing = [name for name in terminal_artifacts if (experiment / name).exists()]
    if existing:
        raise FileExistsError(f"create-only terminal artifacts already exist: {existing}")

    config = load_json(CONFIG_PATH)
    provenance = load_json(experiment / "INPUT_PROVENANCE.json")
    qualification = load_json(experiment / "QUALIFICATION_REPORT.json")
    parameters = load_json(experiment / "PARAMETER_REPORT.json")
    cache = load_json(experiment / "depth_cache/train/CACHE_REPORT.json")
    failed_batch = audit_failed_batch(experiment, config)
    if not failed_batch["all_inputs_and_targets_finite"]:
        raise RuntimeError("failed-batch audit found a data/target non-finite; runtime terminal is not valid")
    started = load_json(experiment / "TRAINING_STARTED.json")
    started_time = datetime.fromisoformat(started["created_utc"])
    terminal_time = datetime.fromisoformat(utc_now())
    failure = {
        "schema": "route_b_v3_1_depth_aware_lraspp_scientific_runtime_failure_v1",
        "created_utc": terminal_time.isoformat(),
        "terminal": TERMINAL,
        "exception_type": "FloatingPointError",
        "exception": "nonfinite scientific loss epoch=1 batch=14",
        "epoch": FAILED_EPOCH,
        "batch": FAILED_BATCH,
        "completed_epochs": 0,
        "optimizer_updates_completed_inferred": 13,
        "optimizer_update_count_is_control_flow_inference": True,
        "durable_checkpoints": [],
        "exact_resume_possible": False,
        "scientific_runs_started": 1,
        "scientific_retries": 0,
        "scientific_variants": 0,
        "contract_qualification_passed": True,
        "classification_reason": "full-FP32 scientific loss became non-finite before the first exact epoch checkpoint; inputs/targets are finite and no frozen-contract violation was found; scientific retry/change is forbidden",
        "training_wall_seconds": None,
        "training_wall_time_note": "exact subprocess wall time was not persisted before the uncaught exception",
        "terminalization_elapsed_upper_bound_seconds_from_started_artifact": (terminal_time - started_time).total_seconds(),
        "failed_batch_input_audit": failed_batch,
        "validation_predictions_created": False,
        "validation_depth_opened": False,
        "test_payload_enumerated": False,
    }
    write_json_x(experiment / "SCIENTIFIC_RUNTIME_FAILURE.json", failure)
    write_json_x(experiment / "TRAINING_COMPLETE.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_training_terminal_v1",
        "created_utc": utc_now(),
        "complete": False,
        "terminal": TERMINAL,
        "epochs_required": 40,
        "epochs_completed": 0,
        "scientific_runs": 1,
        "scientific_retries": 0,
        "optimizer_updates_completed_inferred": 13,
        "checkpoint_epochs": [],
        "evaluation_during_training": 0,
        "failure_artifact": "SCIENTIFIC_RUNTIME_FAILURE.json",
    })
    write_json_x(experiment / "SELECTION_DECISION.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_selection_decision_v1",
        "created_utc": utc_now(),
        "terminal": TERMINAL,
        "selected_epoch": None,
        "selected_checkpoint": None,
        "selected_checkpoint_sha256": None,
        "preservation_eligible_epochs": [],
        "diagnostic_epoch": None,
        "v025_sensitivity_licensed": False,
        "reason": "no epoch completed and no checkpoint exists after unrecoverable non-finite loss",
    })
    write_text_x(experiment / "TERMINAL_VERDICT.txt", TERMINAL + "\n")
    notification = {
        "schema": "route_b_v3_1_depth_aware_lraspp_notification_v1",
        "created_utc": utc_now(),
        "terminal": TERMINAL,
        "attempted": True,
        "command": ["notify-send", "Depth-aware LR-ASPP terminal", TERMINAL],
        "returncode": args.notification_returncode,
        "stderr": args.notification_stderr,
        "delivered": args.notification_returncode == 0,
    }
    write_json_x(experiment / "NOTIFICATION.json", notification)
    write_text_x(experiment / "COMPLETION_SENTINEL", TERMINAL + "\n")
    report = markdown_report(
        experiment, config, provenance, qualification, parameters, cache, failure, notification,
    )
    write_text_x(experiment / "FINAL_REPORT.md", report)
    write_json_x(experiment / "PIPELINE_COMPLETE.json", {
        "schema": "route_b_v3_1_depth_aware_lraspp_pipeline_complete_v1",
        "created_utc": utc_now(),
        "terminal": TERMINAL,
        "scientific_training_complete": False,
        "final_report": str(experiment / "FINAL_REPORT.md"),
        "final_report_sha256": sha256(experiment / "FINAL_REPORT.md"),
        "completion_sentinel": True,
        "notification": notification,
        "selected_checkpoint": None,
        "selected_checkpoint_sha256": None,
    })
    print(json.dumps({
        "terminal": TERMINAL,
        "report": str(experiment / "FINAL_REPORT.md"),
        "notification": notification,
        "failed_batch_all_finite": failed_batch["all_inputs_and_targets_finite"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
