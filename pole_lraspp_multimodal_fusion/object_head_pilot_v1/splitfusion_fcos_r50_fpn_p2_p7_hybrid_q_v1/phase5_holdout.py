"""Phase-5 train-holdout evaluation, preservation gates and checkpoint selection.

Runs the exact p025 service pipeline over the two reserved train-holdout episodes at
the frozen q=0 baseline and at every registered checkpoint/q pair, scores them with
the existing frozen vehicle, segmentation and p025 AVO scorers, applies the
registered preservation gates and selects a checkpoint. Validation and test are not
opened; q=0.90 and q=0.98 are not evaluated in this phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from . import codec, contract, guards
from .gpu_qualification import build_train_dataset, load_frozen_perception, sha256_file
from .phase5_common import (
    bind_inputs,
    build_contract_alias,
    cross_check_person_avo,
    load_frozen_scorers,
    load_holdout_person_truth,
    load_p025_qualification,
    score_person_avo,
    source_delta,
)
from .ranker import build_ranker
from .selection import apply_selection, select_cells
from .teacher_cache import build_split_partition

EXECUTE_TOKEN = "HYBRID_Q_PHASE5_HOLDOUT_EVALUATION"
SCHEMA = "splitfusion_fcos_hybrid_q_phase5_holdout_v1"
DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8
# Framed encode/decode is deterministic and was proved bit-exact in Phase 3; a
# bounded per-pass re-verification keeps the check without a 22 MB device-to-host
# copy of every frame purely for comparison.
ROUNDTRIP_VERIFY_FRAMES = 8


def _collate(items: Sequence[tuple]) -> tuple[torch.Tensor, list, list]:
    return (
        torch.stack([item[0] for item in items]),
        [dict(item[1]) for item in items],
        [item[2] for item in items],
    )


# ---------------------------------------------------------------------------
# One transport configuration over the reserved holdout episodes
# ---------------------------------------------------------------------------


def run_pass(
    *, model: torch.nn.Module, base: Any, ranker: torch.nn.Module | None, q: float,
    dataset: Any, positions: Sequence[int], frame_ids: Sequence[str],
    device: torch.device, output: Path, workers: int, limit: int | None = None,
) -> dict[str, Any]:
    """Encode, transport, decode and serve one q over the holdout, writing predictions."""
    value = guards.require_valid_q(q)
    if value != contract.HOLDOUT_BASELINE_Q and ranker is None:
        raise guards.HybridQConfigError("a nonzero q requires a trained ranker")
    if value == contract.HOLDOUT_BASELINE_Q and ranker is not None:
        raise guards.HybridQConfigError("the q=0 baseline must not invoke a ranker")

    output.mkdir(parents=True, exist_ok=False)
    segmentation_dir = output / "segmentation"
    segmentation_dir.mkdir()
    detections_path = output / "detections.csv"
    manifest_path = output / "segmentation_manifest.csv"

    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=INFERENCE_BATCH, shuffle=False, num_workers=workers,
        collate_fn=_collate, drop_last=False, pin_memory=False,
    )
    expected_keep = contract.keep_count(value)
    payload_bytes: set[int] = set()
    keep_counts: set[int] = set()
    roundtrip_exact = True
    roundtrip_verified = 0
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    detection_count = 0
    person_count = 0
    vehicle_count = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base.infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for batch_index, (fused, rows, calibrations) in enumerate(loader):
                if limit is not None and batch_index >= limit:
                    break
                inputs = fused.to(device, non_blocking=True)
                c2 = model.encode_front(inputs).float()
                guards.require_frozen_batched_c2(c2, what="frozen holdout C2")

                transported = []
                for index in range(c2.shape[0]):
                    frame = c2[index]
                    if value == contract.HOLDOUT_BASELINE_Q:
                        payload = codec.encode(frame, value)
                        selection = None
                    else:
                        scores = ranker.score_cells(frame)
                        selection = select_cells(scores, value)
                        payload = codec.encode(apply_selection(frame, selection), value, selection)
                    decoded, decoded_q = codec.decode(payload)
                    if decoded_q != value:
                        raise guards.HybridQPayloadError("decoded q drift")
                    keep = expected_keep if selection is None else int(selection.keep_count)
                    keep_counts.add(keep)
                    payload_bytes.add(int(payload.total_bytes))
                    if roundtrip_verified < ROUNDTRIP_VERIFY_FRAMES:
                        cpu_frame = frame.detach().cpu()
                        if selection is None:
                            roundtrip_exact &= bool(torch.equal(decoded, cpu_frame))
                        else:
                            mask = selection.keep_mask.unsqueeze(0).expand_as(cpu_frame).cpu()
                            roundtrip_exact &= bool(torch.equal(decoded[mask], cpu_frame[mask]))
                            roundtrip_exact &= bool((decoded[~mask] == 0).all())
                        roundtrip_verified += 1
                        del cpu_frame
                    transported.append(decoded.to(device))

                hybrid = torch.stack(transported)
                outputs = model.decode_tail(hybrid, dense=False)
                calibration_gpu = [
                    {name: tensor.to(device) for name, tensor in calibration.items()}
                    for calibration in calibrations
                ]
                detections = model.postprocess(outputs, calibration_gpu)
                for index, row in enumerate(rows):
                    frame_view = {"semantic_logits": outputs["semantic_logits"][index:index + 1]}
                    served, original_indices = apply_p025_service_policy(
                        frame_view, detections[index]
                    )
                    records = combined_records(base, row, served, original_indices)
                    for record in records:
                        writer.writerow(record)
                        if record["class_name"] == "person":
                            person_count += 1
                        else:
                            vehicle_count += 1
                    detection_count += len(records)
                    observed_ids.append(str(row["sample_id"]))

                    source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                    labels = F.interpolate(
                        outputs["semantic_logits"][index:index + 1].float(),
                        size=source_hw, mode="bilinear", align_corners=False,
                    ).argmax(1)[0]
                    array = labels.cpu().numpy().astype(np.uint8)
                    relative = Path("segmentation") / f"{row['sample_id']}.png"
                    if not cv2.imwrite(str(output / relative), array):
                        raise RuntimeError(f"failed segmentation write {relative}")
                    segmentation_rows.append({
                        "sample_id": row["sample_id"], "prediction_path": str(relative),
                        "width": array.shape[1], "height": array.shape[0],
                    })
                del c2, hybrid, outputs, detections, transported

    with manifest_path.open("x", encoding="utf-8", newline="") as stream:
        manifest_writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height")
        )
        manifest_writer.writeheader()
        manifest_writer.writerows(segmentation_rows)

    del loader
    if limit is None:
        if observed_ids != list(frame_ids):
            raise guards.HybridQConfigError("holdout inference order drift")
        if len(set(observed_ids)) != contract.TRAIN_HOLDOUT_FRAMES:
            raise guards.HybridQConfigError("holdout frame coverage drift")
    if keep_counts != {expected_keep}:
        raise guards.HybridQPayloadError(f"observed keep counts {sorted(keep_counts)}")
    if not roundtrip_exact:
        raise guards.HybridQPayloadError("framed encode/decode was not exact")
    observed_payload = sorted(payload_bytes)
    if len(observed_payload) != 1:
        raise guards.HybridQPayloadError(f"non-constant framed payload {observed_payload}")

    return {
        "q": value,
        "frames": len(observed_ids),
        "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "detections": detection_count,
        "person_service_outputs": person_count,
        "vehicle_service_outputs": vehicle_count,
        "retained_cells": expected_keep,
        "dropped_cells": contract.drop_count(value),
        "framed_payload_bytes": observed_payload[0],
        "framed_payload_ratio": contract.framed_payload_ratio(observed_payload[0]),
        "raw_fp32_ratio": contract.raw_fp32_ratio(observed_payload[0]),
        "framed_encode_decode_exact": True,
        "framed_encode_decode_frames_verified": roundtrip_verified,
        "ranker_invoked": ranker is not None,
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
    }


def person_predictions(detections_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Person service rows in written order, which is ascending post-NMS index."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    with detections_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["class_name"] != "person":
                continue
            grouped.setdefault(row["sample_id"], []).append({
                "score": float(row["score"]),
                "world_x": float(row["world_x"]),
                "world_y": float(row["world_y"]),
                "prediction_index": int(row["prediction_index"]),
            })
    for values in grouped.values():
        if [item["prediction_index"] for item in values] != sorted(
            item["prediction_index"] for item in values
        ):
            raise guards.HybridQConfigError("person service rows are not in post-NMS order")
    return grouped


def score_pass(
    *, result: Mapping[str, Any], scorers: Any, qualification: Any, truth: Any,
    alias_root: Path, frame_ids: Sequence[str], gt: Mapping[str, Any],
    ignore_cache: dict[str, Any], cross_check: bool, require_defined: bool = True,
) -> dict[str, Any]:
    """Existing frozen vehicle and segmentation scoring plus the frozen p025 AVO view."""
    prediction_root = Path(result["prediction_root"])
    detections_path = prediction_root / "detections.csv"
    predictions, missing = scorers.load_predictions(detections_path)
    if missing:
        raise guards.HybridQNumericalError(f"missing/nonfinite prediction fields: {missing[:5]}")
    arm = scorers.score_arm(
        experiment=alias_root, contract=contract.PRIMARY_CONTRACT, frame_ids=list(frame_ids),
        predictions=predictions, gt=gt, threshold=contract.VEHICLE_SCORE_THRESHOLD,
        ignore_cache=ignore_cache,
    )
    segmentation = scorers.score_segmentation(
        alias_root, contract.PRIMARY_CONTRACT, list(frame_ids), prediction_root,
        prediction_root / "segmentation_manifest.csv",
    )
    people = person_predictions(detections_path)
    person = score_person_avo(
        frame_ids=frame_ids, predictions=people, truth=truth, qualification=qualification,
    )
    agreement = None
    if cross_check:
        agreement = cross_check_person_avo(
            frame_ids=frame_ids, predictions=people, truth=truth,
            qualification=qualification, observed=person,
        )
    vehicle = arm["classes"]["vehicle"]
    if require_defined:
        for name in ("precision", "recall", "f1", "xy_mae_m"):
            if vehicle[name] is None:
                raise guards.HybridQNumericalError(f"undefined vehicle {name}")
        if person["xy_mae_m"] is None:
            raise guards.HybridQNumericalError("undefined person AVO xy_mae_m")
    metrics = {
        "vehicle_precision": float(vehicle["precision"]),
        "vehicle_recall": float(vehicle["recall"]),
        "vehicle_f1": float(vehicle["f1"]),
        "vehicle_xy_mae_m": float("nan") if vehicle["xy_mae_m"] is None
        else float(vehicle["xy_mae_m"]),
        "person_avo_precision": float(person["precision"]),
        "person_avo_recall": float(person["recall"]),
        "person_avo_f1": float(person["f1"]),
        "person_avo_xy_mae_m": float("nan") if person["xy_mae_m"] is None
        else float(person["xy_mae_m"]),
        "vehicle_iou": float(segmentation["vehicle_iou"]),
        "person_box_mask_iou": float(segmentation["person_box_mask_iou"]),
        "foreground_miou": float(segmentation["foreground_miou"]),
        "person_avo_recall_20_40m": float(person["recall_20_40m"]),
    }
    if set(metrics) != set(contract.PROTECTED_METRICS):
        raise guards.HybridQConfigError("protected metric set drift")
    return {
        **dict(result),
        "metrics": metrics,
        "vehicle_detail": arm["classes"]["vehicle"],
        "person_canonical_v010_detail": arm["classes"]["person"],
        "person_avo_detail": person,
        "segmentation_detail": {
            name: segmentation[name] for name in
            ("vehicle_iou", "person_box_mask_iou", "foreground_miou",
             "background_iou", "background_iou_role", "ignored_pixels", "confusion_matrix")
        },
        "frozen_p025_agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Gates and selection
# ---------------------------------------------------------------------------


def evaluate_gates(baseline: Mapping[str, float], candidate: Mapping[str, float]) -> dict[str, Any]:
    rows = {}
    for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES:
        degradation = contract.gate_degradation(name, baseline[name], candidate[name])
        rows[name] = {
            "direction": direction,
            "bound": bound,
            "baseline": baseline[name],
            "candidate": candidate[name],
            "degradation": degradation,
            "passed": degradation <= bound,
        }
    degradations = [row["degradation"] for row in rows.values()]
    return {
        "gates": rows,
        "all_passed": all(row["passed"] for row in rows.values()),
        "failed": sorted(name for name, row in rows.items() if not row["passed"]),
        "worst_degradation": max(degradations),
        "worst_absolute_degradation": max(abs(value) for value in degradations),
    }


def select_checkpoint(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Largest passing q; ties by smallest worst absolute degradation; then earlier epoch."""
    passing = [row for row in evaluations if row["gate_result"]["all_passed"]]
    if not passing:
        return {
            "selected": None,
            "terminal": contract.PHASE5_TERMINAL_NOT_SAFE,
            "rule": "no checkpoint/q pair passed every registered preservation gate",
            "passing_pairs": [],
        }
    best_q = max(float(row["q"]) for row in passing)
    at_q = [row for row in passing if float(row["q"]) == best_q]
    minimum = min(row["gate_result"]["worst_absolute_degradation"] for row in at_q)
    tied = [
        row for row in at_q
        if row["gate_result"]["worst_absolute_degradation"] == minimum
    ]
    chosen = min(tied, key=lambda row: int(row["epoch"]))
    return {
        "selected": {
            "epoch": int(chosen["epoch"]),
            "q": float(chosen["q"]),
            "checkpoint": chosen["checkpoint"],
            "checkpoint_sha256": chosen["checkpoint_sha256"],
            "metrics": chosen["metrics"],
            "gate_result": chosen["gate_result"],
        },
        "terminal": contract.PHASE5_TERMINAL_SELECTED,
        "rule": (
            "1) largest q among 0.30/0.50/0.70 passing every holdout preservation gate; "
            "2) smallest worst absolute degradation across the protected metrics at that q; "
            "3) earlier checkpoint"
        ),
        "largest_passing_q": best_q,
        "candidates_at_selected_q": [
            {
                "epoch": int(row["epoch"]),
                "worst_absolute_degradation": row["gate_result"]["worst_absolute_degradation"],
                "worst_degradation": row["gate_result"]["worst_degradation"],
            }
            for row in sorted(at_q, key=lambda item: int(item["epoch"]))
        ],
        "tie_broken_on_epoch": len(tied) > 1,
        "passing_pairs": sorted(
            (int(row["epoch"]), float(row["q"])) for row in passing
        ),
        "selection_inputs_excluded": [
            "teacher loss", "training loss", "isolated metric improvements",
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid-q Phase-5 train-holdout evaluation")
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--keep-segmentation", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()

    training_dir = args.training.resolve(strict=True)
    if not (training_dir / "TRAINING_COMPLETE").is_file():
        raise guards.HybridQConfigError("holdout evaluation requires a complete training run")
    training_report = json.loads(
        (training_dir / "training_report.json").read_text(encoding="utf-8")
    )
    output = training_dir / "holdout"
    if output.exists():
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-5 holdout evaluation requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)

    binding = bind_inputs()
    delta = source_delta(binding)
    for name, digest in training_report["candidate_checkpoints"].items():
        if sha256_file(training_dir / "checkpoints" / name) != digest:
            raise guards.HybridQConfigError(f"candidate checkpoint {name} sha256 drift")
    if training_report["binding"]["hybrid_q_locked_config"]["sha256"] != \
            binding["hybrid_q_locked_config"]["sha256"]:
        raise guards.HybridQConfigError("training ran under a different locked configuration")

    model, base, perception = load_frozen_perception(device)
    frozen_snapshot = guards.snapshot_module_state(model)
    route = build_train_dataset(base)
    partition = build_split_partition(route)
    frame_ids = list(partition.holdout_sample_ids)

    root = contract.repository_root()
    config = json.loads(
        (root / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
         "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json").read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    inference = base.data.InferenceDataset(dataset_root, "train")
    position_by_id = {row["sample_id"]: index for index, row in enumerate(inference.rows)}
    if len(position_by_id) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError("inference dataset train frame count drift")
    positions = [position_by_id[sample_id] for sample_id in frame_ids]
    if len(positions) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("holdout position count drift")

    output.mkdir(parents=True, exist_ok=False)
    alias_root = build_contract_alias(dataset_root, output / "train_contract_alias")
    scorers = load_frozen_scorers()
    qualification = load_p025_qualification()
    truth = load_holdout_person_truth(frame_ids)
    gt, gt_states = scorers.load_gt(alias_root, contract.PRIMARY_CONTRACT)
    holdout_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    ignore_cache: dict[str, Any] = {}
    print(f"[phase5-holdout] {len(frame_ids)} reserved frames; "
          f"{truth.diagnostics['observable_actor_frames']} AVO-observable person actor-frames; "
          f"{sum(1 for rows in holdout_gt.values() for row in rows if row['class_name'] == 'vehicle')}"
          " v010 vehicle GT rows", flush=True)

    limit = args.smoke_batches or None
    scored_ids = frame_ids if limit is None else frame_ids[:limit * INFERENCE_BATCH]
    predictions_root = output / "predictions"
    predictions_root.mkdir()

    baseline_raw = run_pass(
        model=model, base=base, ranker=None, q=contract.HOLDOUT_BASELINE_Q,
        dataset=inference, positions=positions, frame_ids=frame_ids, device=device,
        output=predictions_root / "q0_baseline", workers=int(args.workers), limit=limit,
    )
    guards.require_module_state_unchanged(model, frozen_snapshot)
    baseline = score_pass(
        result=baseline_raw, scorers=scorers, qualification=qualification, truth=truth,
        alias_root=alias_root, frame_ids=scored_ids, gt=holdout_gt,
        ignore_cache=ignore_cache, cross_check=True, require_defined=limit is None,
    )
    baseline["configuration"] = "q0_frozen_baseline"
    baseline["epoch"] = None
    baseline["checkpoint"] = None
    print(json.dumps({"pass": "q0_baseline", "metrics": baseline["metrics"]}, indent=2), flush=True)

    evaluations: list[dict[str, Any]] = []
    for epoch in contract.HOLDOUT_CANDIDATE_EPOCHS:
        name = f"ranker_epoch_{epoch:02d}.pt"
        path = training_dir / "checkpoints" / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != epoch:
            raise guards.HybridQConfigError(f"candidate {name} epoch drift")
        if int(payload["parameter_count"]) != contract.RANKER_PARAMETER_COUNT:
            raise guards.HybridQConfigError(f"candidate {name} parameter count drift")
        ranker = build_ranker()
        ranker.load_state_dict(payload["ranker"])
        ranker = ranker.to(device).eval()
        for parameter in ranker.parameters():
            parameter.requires_grad_(False)
        digest = sha256_file(path)
        for q in contract.HOLDOUT_EVALUATION_Q_VALUES:
            raw = run_pass(
                model=model, base=base, ranker=ranker, q=q, dataset=inference,
                positions=positions, frame_ids=frame_ids, device=device,
                output=predictions_root / f"epoch{epoch:02d}_q{int(round(q * 100)):02d}",
                workers=int(args.workers), limit=limit,
            )
            guards.require_module_state_unchanged(model, frozen_snapshot)
            scored = score_pass(
                result=raw, scorers=scorers, qualification=qualification, truth=truth,
                alias_root=alias_root, frame_ids=scored_ids, gt=holdout_gt,
                ignore_cache=ignore_cache, cross_check=False, require_defined=limit is None,
            )
            scored["configuration"] = f"epoch{epoch:02d}_q{q:.2f}"
            scored["epoch"] = epoch
            scored["checkpoint"] = name
            scored["checkpoint_sha256"] = digest
            scored["gate_result"] = evaluate_gates(baseline["metrics"], scored["metrics"])
            evaluations.append(scored)
            print(json.dumps({
                "pass": scored["configuration"],
                "all_gates_passed": scored["gate_result"]["all_passed"],
                "failed": scored["gate_result"]["failed"],
                "worst_absolute_degradation": scored["gate_result"]["worst_absolute_degradation"],
            }), flush=True)
        del ranker

    passes = [baseline] + evaluations
    decision = select_checkpoint(evaluations)

    if not args.keep_segmentation:
        for entry in passes:
            directory = Path(entry["prediction_root"]) / "segmentation"
            if directory.is_dir():
                shutil.rmtree(directory)
            entry["segmentation_masks_removed_after_scoring"] = True

    report = {
        "schema": SCHEMA,
        "terminal": decision["terminal"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "evaluated_split": "reserved train-holdout episodes",
            "holdout_episodes": list(contract.TRAIN_HOLDOUT_EPISODES),
            "holdout_frames": contract.TRAIN_HOLDOUT_FRAMES,
            "validation_or_test_accessed": False,
            "evaluated_q_values": [contract.HOLDOUT_BASELINE_Q]
            + list(contract.HOLDOUT_EVALUATION_Q_VALUES),
            "stress_q_values_not_evaluated": list(contract.EVALUATION_STRESS_Q_VALUES),
            "payload_or_zstd_acceptance": "deferred; Phase 5 establishes perception preservation only",
            "carla_launched": False,
            "quantization_or_zstd_run": False,
            "training_run_here": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "inference_precision": "fp32 inference_mode, no autocast",
        },
        "binding": {k: v for k, v in binding.items() if k != "teacher_cache_shards"},
        "source_delta": delta,
        "perception_binding": perception,
        "training_run": {
            "path": str(training_dir),
            "terminal": training_report["terminal"],
            "candidate_checkpoints": training_report["candidate_checkpoints"],
            "optimizer_updates": training_report["totals"]["optimizer_updates"],
        },
        "service_pipeline": {
            "policy": "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.apply_p025_service_policy",
            "person_output_threshold": contract.PERSON_SERVICE_SCORE_THRESHOLD,
            "vehicle_score_point": contract.VEHICLE_SCORE_THRESHOLD,
            "vehicle_score_point_note": (
                "the canonical 0.20 applied to the calibrated service score, i.e. the "
                "locked base operating point 0.5224518340619145; unchanged by Phase 5"
            ),
            "thresholds_or_postprocessing_changed": False,
        },
        "scoring": {
            "vehicle_and_segmentation": "frozen v3.1 audit_v1.score_arm and score_contract_v1.score_segmentation",
            "frozen_scorer_sha256": scorers.sha256,
            "person": "frozen p025 AVO>=0.65 view (qualification.greedy_match cascade)",
            "contract": contract.PRIMARY_CONTRACT,
            "train_contract_alias": (
                "contracts/<contract>/val symlinked to the dataset train contract so the "
                "frozen scorers read the holdout split without any scorer edit"
            ),
            "gt_contract_states": gt_states,
            "person_truth": truth.diagnostics,
        },
        "q0_baseline": baseline,
        "checkpoint_q_evaluations": evaluations,
        "preservation_gates": [
            {"metric": name, "direction": direction, "bound": bound}
            for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
        ],
        "selection": decision,
        "frozen_state_unchanged_at_end": True,
    }
    guards.require_module_state_unchanged(model, frozen_snapshot)
    (output / "holdout_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output / "SELECTION_DECISION.json").write_text(
        json.dumps({
            "schema": "splitfusion_fcos_hybrid_q_phase5_selection_v1",
            "generated_utc": report["generated_utc"],
            "terminal": decision["terminal"],
            "q0_baseline_metrics": baseline["metrics"],
            "decision": decision,
            "all_pairs": [
                {
                    "epoch": row["epoch"], "q": row["q"], "metrics": row["metrics"],
                    "gate_result": row["gate_result"],
                }
                for row in evaluations
            ],
        }, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output / decision["terminal"]).write_text(
        f"{decision['terminal']} {report['generated_utc']}\n", encoding="utf-8"
    )
    print(json.dumps({"terminal": decision["terminal"],
                      "selected": decision.get("selected"),
                      "output": str(output)}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - runner entry point
    raise SystemExit(main())
