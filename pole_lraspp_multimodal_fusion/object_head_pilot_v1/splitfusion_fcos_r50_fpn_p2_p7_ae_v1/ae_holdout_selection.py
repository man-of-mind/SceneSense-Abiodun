"""Phase-9C AE128 train-holdout evaluation and preregistered checkpoint selection.

A separate command from training, deliberately: nothing here can run while the
trainer runs, and the trainer refuses to start if this module is even imported.

After a separately authorized completed training run this evaluates exactly the
three candidate epochs {4, 8, 12} at exactly q = {0, 0.30, 0.50, 0.70}, one
holdout inference/evaluation pass per checkpoint/q pair, over the 3,284 reserved
train-holdout frames. Validation and test are never opened.

Transport at this phase is **FP32 AE latent reconstruction only**: the AE encodes,
the ranker's keep mask drops cells, the AE decodes, and the unchanged frozen
perception tail plus the unchanged p025/AVO scoring run on the result. There is
no UINT8, zstd, UINT6 or UINT4 here; the deployment-path quantizer validation is
a later phase on whichever checkpoint this selects.

Each AE checkpoint/q result is compared against the corresponding **frozen noAE
same-q** holdout result, so the reported degradation isolates the AE rather than
re-measuring the ROI drop the noAE path already pays.

Selecting a checkpoint is not a service-ready claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import time
from dataclasses import dataclass, field
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

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
    build_contract_alias,
    load_frozen_scorers,
    load_holdout_person_truth,
    load_p025_qualification,
    source_delta,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_holdout import (
    _collate,
    score_pass,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.teacher_cache import (
    build_split_partition,
)
from . import ae_composition, ae_loss, ae_training_common as common
from .ae_gpu_qualification import per_frame_errors

EXECUTE_TOKEN = "SPLITFUSION_AE128_PHASE9C_HOLDOUT_SELECTION"
TERMINAL = "SPLITFUSION_AE128_HOLDOUT_CHECKPOINT_SELECTED"
SCHEMA = common.AE_HOLDOUT_SCHEMA
DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8
GATE_COUNT = len(contract.HOLDOUT_PRESERVATION_GATES)

# The preregistered ranking, stated once and applied verbatim.
RANKING_RULE = (
    "1) maximize the minimum number of same-q preservation gates passed across "
    "q in {0.00, 0.30, 0.50, 0.70}; "
    "2) then maximize the total preservation gates passed across all four q; "
    "3) then minimize the worst normalized protected-metric degradation "
    "(degradation divided by that metric's registered gate bound); "
    "4) then minimize the mean over the four q of the global holdout task-aware "
    "reconstruction loss, where global means each ratio is a total numerator "
    "over a total denominator accumulated across all holdout frames rather than "
    "an unweighted mean of per-batch ratios; "
    "5) then prefer the earlier epoch"
)
RANKING_CRITERIA = (
    "min_same_q_gates_passed",
    "total_gates_passed",
    "worst_normalized_degradation",
    "mean_holdout_reconstruction_loss",
    "epoch",
)


# ---------------------------------------------------------------------------
# One AE checkpoint/q holdout pass
# ---------------------------------------------------------------------------


@dataclass
class HoldoutReconstructionTotals:
    """Batching-independent global reconstruction ratios over the whole holdout.

    The committed task-aware loss is a *ratio of sums*: plain is the total
    squared error over the total reference energy, and combined-importance is the
    same ratio under the per-frame L1-normalized importance weights. An
    unweighted mean of per-batch ratios is therefore not the loss over the
    holdout — it depends on how the frames happened to be grouped, and the short
    final batch is over-weighted. This accumulates the four sums per frame and
    divides once at the end, so regrouping the same frames into different batch
    sizes cannot move the number that criterion 4 ranks on.
    """

    plain_numerator: list[float] = field(default_factory=list)
    plain_denominator: list[float] = field(default_factory=list)
    weighted_numerator: list[float] = field(default_factory=list)
    weighted_denominator: list[float] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return len(self.plain_numerator)

    def observe(
        self,
        c2: torch.Tensor,
        reconstructed: torch.Tensor,
        teacher: ae_loss.CachedTeacherBatch,
    ) -> None:
        """Accumulate one batch's per-frame numerators and denominators."""
        with torch.no_grad():
            target = c2.detach().float()
            estimate = reconstructed.detach().float()
            frames = int(target.shape[0])
            if teacher.frames != frames:
                raise guards.HybridQPayloadError(
                    f"teacher batch covers {teacher.frames} frames, C2 batch has {frames}"
                )
            cell_error = (estimate - target).pow(2).sum(dim=1)
            cell_energy = target.pow(2).sum(dim=1)
            # Exactly the loss's own weights: per-frame L1-normalized, which is
            # what makes the four sums additive across frames.
            weights = teacher.importance.detach().to(
                device=target.device, dtype=torch.float32
            )
            mass = weights.reshape(frames, -1).sum(dim=1)
            if not bool((mass > 0).all()):
                raise guards.HybridQNumericalError(
                    "a cached importance map has no positive mass"
                )
            weights = weights / mass.reshape(frames, 1, 1)

            # Each per-frame spatial reduction is taken in float64 over that
            # frame's cells alone, so a frame contributes the same four numbers
            # whatever else shares its batch.
            rows = {
                "plain_numerator": cell_error.reshape(frames, -1).double().sum(dim=1),
                "plain_denominator": cell_energy.reshape(frames, -1).double().sum(dim=1),
                "weighted_numerator": (weights * cell_error)
                .reshape(frames, -1)
                .double()
                .sum(dim=1),
                "weighted_denominator": (weights * cell_energy)
                .reshape(frames, -1)
                .double()
                .sum(dim=1),
            }
            for name, values in rows.items():
                guards.require_finite(values, f"per-frame {name}")
                getattr(self, name).extend(float(value) for value in values.cpu())

    def totals(self) -> dict[str, float]:
        """The four exact sums and the three global ratios they define."""
        if self.frames == 0:
            raise guards.HybridQConfigError("no holdout frame was accumulated")
        plain_numerator = math.fsum(self.plain_numerator)
        plain_denominator = math.fsum(self.plain_denominator)
        weighted_numerator = math.fsum(self.weighted_numerator)
        weighted_denominator = math.fsum(self.weighted_denominator)
        if plain_denominator <= 0.0:
            raise guards.HybridQNumericalError(
                "the holdout reference C2 has zero energy"
            )
        if weighted_denominator <= 0.0:
            raise guards.HybridQNumericalError(
                "the holdout importance mass sits where the reference C2 has no energy"
            )
        global_plain = plain_numerator / plain_denominator
        global_combined = weighted_numerator / weighted_denominator
        return {
            "frames": self.frames,
            "plain_squared_error_numerator": plain_numerator,
            "plain_reference_energy_denominator": plain_denominator,
            "combined_importance_numerator": weighted_numerator,
            "combined_importance_reference_energy_denominator": weighted_denominator,
            "global_plain_reconstruction": global_plain,
            "global_combined_importance_reconstruction": global_combined,
            "global_total_loss": global_plain + global_combined,
        }


def run_pass(
    *,
    model: torch.nn.Module,
    base: Any,
    ranker: torch.nn.Module,
    autoencoder: torch.nn.Module,
    q: float,
    dataset: Any,
    positions: Sequence[int],
    frame_ids: Sequence[str],
    store: common.AeTeacherStore,
    device: torch.device,
    output: Path,
    workers: int,
    limit: int | None = None,
) -> dict[str, Any]:
    """Encode, drop, AE-reconstruct in FP32 and serve one q over the holdout."""
    plan = continuous_q.quantize_q(q)
    if plan.wire_q not in {
        continuous_q.quantize_q(float(value)).wire_q for value in common.AE_HOLDOUT_Q_VALUES
    }:
        raise guards.HybridQConfigError(
            f"q={plan.wire_q!r} is not a registered Phase-9C selection q"
        )

    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    detections_path = output / "detections.csv"
    manifest_path = output / "segmentation_manifest.csv"

    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=INFERENCE_BATCH,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate,
        drop_last=False,
        pin_memory=False,
    )
    keep_counts: set[int] = set()
    ranker_invocations = 0
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    batch_totals: list[float] = []
    batch_plain: list[float] = []
    batch_combined: list[float] = []
    frame_plain: list[float] = []
    frame_weighted: list[float] = []
    totals = HoldoutReconstructionTotals()
    min_valid_groups = len(contract.TEACHER_GROUPS)
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
                sample_ids = [str(row["sample_id"]) for row in rows]
                inputs = fused.to(device, non_blocking=True)
                c2 = model.encode_front(inputs).float()
                guards.require_frozen_batched_c2(c2, what="frozen holdout C2")

                reconstructed = []
                for index in range(c2.shape[0]):
                    composition = ae_composition.compose(
                        c2[index], autoencoder, ranker, plan.wire_q
                    )
                    if composition.ranker_used:
                        ranker_invocations += 1
                    keep_counts.add(composition.keep_count)
                    frame = autoencoder.decode(
                        composition.masked_latent, composition.keep_mask
                    )
                    guards.require_frozen_c2(frame, what="reconstructed holdout C2")
                    reconstructed.append(frame)
                    del composition

                hat = torch.stack(reconstructed)
                teacher = store.batch(sample_ids)
                loss = ae_loss.task_aware_reconstruction_loss(c2, hat, teacher)
                report = loss.report()
                batch_totals.append(report["total"])
                batch_plain.append(report["plain_reconstruction"])
                batch_combined.append(report["combined_importance_reconstruction"])
                min_valid_groups = min(
                    min_valid_groups, int(report["min_valid_groups_observed"])
                )
                totals.observe(c2, hat, teacher)
                errors = per_frame_errors(c2, hat, teacher)
                frame_plain.extend(errors["plain"])
                frame_weighted.extend(errors["importance_weighted"])
                del teacher, loss, errors

                outputs = model.decode_tail(hat, dense=False)
                calibration_gpu = [
                    {name: tensor.to(device) for name, tensor in calibration.items()}
                    for calibration in calibrations
                ]
                detections = model.postprocess(outputs, calibration_gpu)
                for index, row in enumerate(rows):
                    frame_view = {
                        "semantic_logits": outputs["semantic_logits"][index : index + 1]
                    }
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
                        outputs["semantic_logits"][index : index + 1].float(),
                        size=source_hw,
                        mode="bilinear",
                        align_corners=False,
                    ).argmax(1)[0]
                    array = labels.cpu().numpy().astype(np.uint8)
                    relative = Path("segmentation") / f"{row['sample_id']}.png"
                    if not cv2.imwrite(str(output / relative), array):
                        raise RuntimeError(f"failed segmentation write {relative}")
                    segmentation_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "prediction_path": str(relative),
                            "width": array.shape[1],
                            "height": array.shape[0],
                        }
                    )
                del c2, hat, outputs, detections, reconstructed

    with manifest_path.open("x", encoding="utf-8", newline="") as stream:
        manifest_writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height")
        )
        manifest_writer.writeheader()
        manifest_writer.writerows(segmentation_rows)

    del loader
    expected_keep = contract.keep_count(plan.wire_q)
    if limit is None:
        if observed_ids != list(frame_ids):
            raise guards.HybridQConfigError("holdout inference order drift")
        if len(set(observed_ids)) != contract.TRAIN_HOLDOUT_FRAMES:
            raise guards.HybridQConfigError("holdout frame coverage drift")
    if keep_counts != {expected_keep}:
        raise guards.HybridQPayloadError(f"observed keep counts {sorted(keep_counts)}")
    expected_invocations = 0 if plan.is_bypass else len(observed_ids)
    if ranker_invocations != expected_invocations:
        raise guards.HybridQConfigError(
            f"ranker invoked {ranker_invocations} times, expected {expected_invocations}"
        )

    def _summarize(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95.0)),
            "max": float(array.max()),
            "min": float(array.min()),
        }

    return {
        "q": plan.wire_q,
        "q_e4": plan.q_e4,
        "frames": len(observed_ids),
        "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "detections": detection_count,
        "person_service_outputs": person_count,
        "vehicle_service_outputs": vehicle_count,
        "retained_cells": expected_keep,
        "dropped_cells": contract.drop_count(plan.wire_q),
        "ranker_invoked": not plan.is_bypass,
        "ranker_invocations": ranker_invocations,
        "transport": common.AE_HOLDOUT_QUANTIZER,
        "reconstruction": {
            **totals.totals(),
            "definition": (
                "the committed task-aware loss evaluated once over the whole "
                "holdout: each ratio divides a total numerator by a total "
                "denominator accumulated per frame, so it does not depend on how "
                "the frames were batched"
            ),
            "batch_diagnostics": {
                "mean_total_loss": float(np.mean(batch_totals)),
                "mean_plain_reconstruction": float(np.mean(batch_plain)),
                "mean_combined_importance_reconstruction": float(np.mean(batch_combined)),
                "batches": len(batch_totals),
                "note": (
                    "unweighted mean of per-batch ratios; diagnostic only, and "
                    "not used by the checkpoint ranking, because it depends on "
                    "the batch grouping and over-weights the short final batch"
                ),
            },
            "per_frame_plain": _summarize(frame_plain),
            "per_frame_importance_weighted": _summarize(frame_weighted),
            "per_frame_note": "per-frame errors are normalized within each frame",
            "min_valid_groups_observed": min_valid_groups,
        },
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
    }


# ---------------------------------------------------------------------------
# Same-q preservation gates against the frozen noAE reference
# ---------------------------------------------------------------------------


# The registered same-q preservation gates live in the shared module, so a later
# phase that compares against a different frozen noAE measurement reuses the one
# definition instead of importing this train-holdout selection runner into a
# validation process. Re-exported under the name this runner and its report use.
evaluate_same_q_gates = common.evaluate_same_q_gates


# ---------------------------------------------------------------------------
# Preregistered checkpoint ranking
# ---------------------------------------------------------------------------


def aggregate_by_checkpoint(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse the per-q rows of each candidate epoch into one ranking record."""
    wanted = tuple(
        continuous_q.quantize_q(float(value)).wire_q for value in common.AE_HOLDOUT_Q_VALUES
    )
    by_epoch: dict[int, dict[float, Mapping[str, Any]]] = {}
    for record in records:
        epoch = int(record["epoch"])
        q = continuous_q.quantize_q(float(record["q"])).wire_q
        if q in by_epoch.setdefault(epoch, {}):
            raise guards.HybridQConfigError(f"epoch {epoch} carries q={q!r} twice")
        by_epoch[epoch][q] = record

    aggregated: list[dict[str, Any]] = []
    for epoch in sorted(by_epoch):
        rows = by_epoch[epoch]
        if tuple(sorted(rows)) != tuple(sorted(wanted)):
            raise guards.HybridQConfigError(
                f"epoch {epoch} covers q {sorted(rows)}, the rule needs {sorted(wanted)}"
            )
        per_q_passed = {
            f"{q:.2f}": int(rows[q]["gate_result"]["gates_passed"]) for q in wanted
        }
        losses = [
            float(rows[q]["reconstruction"]["global_total_loss"]) for q in wanted
        ]
        aggregated.append(
            {
                "epoch": epoch,
                "per_q_gates_passed": per_q_passed,
                "min_same_q_gates_passed": min(per_q_passed.values()),
                "total_gates_passed": sum(per_q_passed.values()),
                "gates_available_per_q": GATE_COUNT,
                "worst_normalized_degradation": max(
                    float(rows[q]["gate_result"]["worst_normalized_degradation"])
                    for q in wanted
                ),
                "mean_holdout_reconstruction_loss": float(np.mean(losses)),
                "per_q_global_reconstruction_loss": {
                    f"{q:.2f}": float(rows[q]["reconstruction"]["global_total_loss"])
                    for q in wanted
                },
                "all_gates_passed_at_every_q": all(
                    bool(rows[q]["gate_result"]["all_passed"]) for q in wanted
                ),
            }
        )
    return aggregated


def ranking_key(row: Mapping[str, Any]) -> tuple[int, int, float, float, int]:
    """The preregistered ordering key; smaller sorts better."""
    return (
        -int(row["min_same_q_gates_passed"]),
        -int(row["total_gates_passed"]),
        float(row["worst_normalized_degradation"]),
        float(row["mean_holdout_reconstruction_loss"]),
        int(row["epoch"]),
    )


def rank_checkpoints(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered rule; deterministic and total by construction."""
    aggregated = aggregate_by_checkpoint(records)
    if not aggregated:
        raise guards.HybridQConfigError("checkpoint ranking needs at least one candidate")
    ordered = sorted(aggregated, key=ranking_key)
    best, *rest = ordered
    decided_at = RANKING_CRITERIA[-1]
    if rest:
        best_key = ranking_key(best)
        runner_key = ranking_key(rest[0])
        for position, name in enumerate(RANKING_CRITERIA):
            if best_key[position] != runner_key[position]:
                decided_at = name
                break
    return {
        "rule": RANKING_RULE,
        "criteria_in_order": list(RANKING_CRITERIA),
        "reconstruction_loss_definition": (
            "criterion 4 uses global_total_loss: the plain and "
            "combined-importance ratios are each formed from numerator and "
            "denominator sums accumulated over every holdout frame, so the value "
            "is independent of the inference batch grouping. The per-batch means "
            "are retained as diagnostics only"
        ),
        "normalized_degradation_definition": (
            "signed protected-metric degradation divided by that metric's "
            "registered preservation-gate bound, so metrics with different "
            "bounds are comparable"
        ),
        "ranking": ordered,
        "selected_epoch": int(best["epoch"]),
        "selected": dict(best),
        "decided_at_criterion": decided_at,
        "selection_is_a_service_ready_claim": False,
        "selection_purpose": (
            "chooses the one AE128 checkpoint that later deployment-path "
            "UINT8+zstd validation will use; it does not accept AE128 for service"
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-9C AE128 train-holdout evaluation and checkpoint selection"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--keep-segmentation", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()

    training_dir = args.training.resolve(strict=True)
    if not (training_dir / "TRAINING_COMPLETE").is_file():
        raise guards.HybridQConfigError(
            "holdout selection requires a separately authorized completed training run"
        )
    training_report = json.loads(
        (training_dir / "training_report.json").read_text(encoding="utf-8")
    )
    if training_report["schema"] != common.AE_TRAINING_SCHEMA:
        raise guards.HybridQConfigError("training report schema drift")
    if training_report["terminal"] != "SPLITFUSION_AE128_TRAINING_COMPLETE":
        raise guards.HybridQConfigError("the training run did not complete")
    if training_report["configuration"] != common.training_configuration():
        raise guards.HybridQConfigError(
            "the training run used a different locked configuration"
        )
    output = training_dir / "holdout_selection"
    if output.exists():
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-9C holdout selection requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)

    binding = common.bind_frozen_inputs()
    delta = source_delta(binding)
    reference = common.load_noae_holdout_reference()
    expected_candidates = {
        common.candidate_filename(epoch): None for epoch in common.AE_CANDIDATE_EPOCHS
    }
    if set(training_report["candidate_checkpoints"]) != set(expected_candidates):
        raise guards.HybridQConfigError("candidate checkpoint set drift")
    for name, digest in training_report["candidate_checkpoints"].items():
        if sha256_file(training_dir / "checkpoints" / name) != digest:
            raise guards.HybridQConfigError(f"candidate checkpoint {name} sha256 drift")

    model, base, perception = load_frozen_perception(device)
    common.freeze(model)
    ranker = common.load_stable_ranker(device)
    guards.require_frozen_perception([model, ranker])
    guards.require_eval_mode([model, ranker])
    frozen_model_state = guards.snapshot_module_state(model)
    frozen_ranker_state = guards.snapshot_module_state(ranker)

    route = build_train_dataset(base)
    partition = build_split_partition(route)
    frame_ids = list(partition.holdout_sample_ids)
    store = common.load_ae_teacher_store(partition, "holdout")
    if store.split != "holdout" or store.frames != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("the selection teacher store is not the holdout")

    root = contract.repository_root()
    config = json.loads(
        (
            root
            / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json"
        ).read_text(encoding="utf-8")
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
    gt, _gt_states = scorers.load_gt(alias_root, contract.PRIMARY_CONTRACT)
    holdout_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    ignore_cache: dict[str, Any] = {}

    limit = args.smoke_batches or None
    scored_ids = frame_ids if limit is None else frame_ids[: limit * INFERENCE_BATCH]
    predictions_root = output / "predictions"
    predictions_root.mkdir()

    evaluations: list[dict[str, Any]] = []
    for epoch in common.AE_CANDIDATE_EPOCHS:
        name = common.candidate_filename(epoch)
        path = training_dir / "checkpoints" / name
        digest = sha256_file(path)
        autoencoder, metadata = common.load_candidate(path, epoch, device, binding)
        for q in common.AE_HOLDOUT_Q_VALUES:
            plan = continuous_q.quantize_q(float(q))
            raw = run_pass(
                model=model,
                base=base,
                ranker=ranker,
                autoencoder=autoencoder,
                q=plan.wire_q,
                dataset=inference,
                positions=positions,
                frame_ids=frame_ids,
                store=store,
                device=device,
                output=predictions_root / f"epoch{epoch:02d}_q{plan.q_e4:04d}",
                workers=int(args.workers),
                limit=limit,
            )
            guards.require_module_state_unchanged(model, frozen_model_state)
            guards.require_module_state_unchanged(ranker, frozen_ranker_state)
            scored = score_pass(
                result=raw,
                scorers=scorers,
                qualification=qualification,
                truth=truth,
                alias_root=alias_root,
                frame_ids=scored_ids,
                gt=holdout_gt,
                ignore_cache=ignore_cache,
                cross_check=False,
                require_defined=limit is None,
            )
            scored["configuration"] = f"ae128_epoch{epoch:02d}_q{plan.wire_q:.2f}"
            scored["epoch"] = epoch
            scored["checkpoint"] = name
            scored["checkpoint_sha256"] = digest
            scored["checkpoint_metadata"] = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            scored["noae_same_q_reference"] = reference[plan.wire_q]
            scored["gate_result"] = evaluate_same_q_gates(
                reference[plan.wire_q]["metrics"], scored["metrics"]
            )
            evaluations.append(scored)
            print(
                json.dumps(
                    {
                        "pass": scored["configuration"],
                        "gates_passed": scored["gate_result"]["gates_passed"],
                        "of": GATE_COUNT,
                        "failed": scored["gate_result"]["failed"],
                        "worst_normalized_degradation": round(
                            scored["gate_result"]["worst_normalized_degradation"], 4
                        ),
                        "global_reconstruction_loss": round(
                            scored["reconstruction"]["global_total_loss"], 6
                        ),
                    }
                ),
                flush=True,
            )
        del autoencoder

    decision = rank_checkpoints(evaluations)

    if not args.keep_segmentation:
        for entry in evaluations:
            directory = Path(entry["prediction_root"]) / "segmentation"
            if directory.is_dir():
                shutil.rmtree(directory)
            entry["segmentation_masks_removed_after_scoring"] = True

    report = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "family": "AE128",
            "evaluated_split": "reserved train-holdout episodes",
            "holdout_episodes": list(contract.TRAIN_HOLDOUT_EPISODES),
            "holdout_frames": contract.TRAIN_HOLDOUT_FRAMES,
            "evaluated_epochs": list(common.AE_CANDIDATE_EPOCHS),
            "evaluated_q_values": [float(q) for q in common.AE_HOLDOUT_Q_VALUES],
            "passes": len(evaluations),
            "one_pass_per_checkpoint_q_pair": True,
            "transport": common.AE_HOLDOUT_QUANTIZER,
            "uint8_zstd_uint6_uint4_run": False,
            "stress_q_values_not_evaluated": list(common.AE_EXCLUDED_Q),
            "validation_or_test_accessed": False,
            "fit_teacher_shards_deserialized": 0,
            "training_run_here": False,
            "threshold_calibration_nms_or_visibility_changed": False,
            "carla_launched": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "binding": binding,
        "perception_binding": perception,
        "hybrid_q_source_delta_since_phase4": delta,
        "teacher_store": store.provenance(),
        "training_run": {
            "path": str(training_dir),
            "candidate_checkpoints": dict(training_report["candidate_checkpoints"]),
            "configuration": training_report["configuration"],
        },
        "noae_reference": {
            "path": common.NOAE_HOLDOUT_RELPATH,
            "sha256": common.NOAE_HOLDOUT_SHA256,
            "ranker_epoch_for_q_above_zero": common.NOAE_REFERENCE_RANKER_EPOCH,
            "per_q": reference,
        },
        "preservation_gates": [
            {"metric": name, "direction": direction, "bound": bound}
            for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
        ],
        "service_pipeline": {
            "vehicle_score_threshold": contract.VEHICLE_SCORE_THRESHOLD,
            "person_service_score_threshold": contract.PERSON_SERVICE_SCORE_THRESHOLD,
            "person_avo_threshold": contract.PERSON_AVO_THRESHOLD,
            "scorer_sha256": scorers.sha256,
            "unchanged": True,
        },
        "checkpoint_q_evaluations": evaluations,
        "selection": decision,
        "frozen_state_unchanged_at_end": True,
    }
    report_path = output / "holdout_selection.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output / TERMINAL).write_text(f"{sha256_file(report_path)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_epoch": decision["selected_epoch"],
                "decided_at_criterion": decision["decided_at_criterion"],
                "report_sha256": sha256_file(report_path),
            }
        )
    )
    print(TERMINAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
