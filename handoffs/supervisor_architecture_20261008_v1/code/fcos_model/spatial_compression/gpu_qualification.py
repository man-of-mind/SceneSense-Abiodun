"""Phase-3 bounded train-only GPU qualification for hybrid-q.

Loads the frozen epoch-26 SplitFusion-FCOS checkpoint behind the p025 forward
lock, keeps every perception parameter frozen and in eval mode, and qualifies
the hybrid-q transport path on a handful of deterministic training batches.

Scope, deliberately: no teacher cache, no epoch, no validation or test access,
no evaluation, no CARLA. Exactly one distillation update and three q-aware
updates are taken on a *disposable* ranker whose state is discarded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1 import (
    recovery_losses,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.base_runtime import (
    load_base,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.recovery_model import (
    build_recovery_model,
)

from . import codec, contract, guards, training
from .ranker import build_ranker
from .selection import select_and_apply, select_cells

EXECUTE_TOKEN = "HYBRID_Q_PHASE3_TRAIN_ONLY_GPU_QUALIFICATION"
PHYSICAL_BATCH_LADDER = (16, 8, 4)
LATENCY_REPETITIONS = 20
LATENCY_WARMUP = 5
QUALIFICATION_BATCHES = 4
UPDATE_Q_ORDER = contract.Q_AWARE_TRAINING_CYCLE  # 0.30, 0.50, 0.70
NESTED_Q_VALUES = (0.30, 0.50, 0.70, 0.90, 0.98)


class OutOfMemory(RuntimeError):
    """CUDA ran out of memory before any optimizer step was taken."""


def _is_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package_source_hashes() -> dict[str, str]:
    root = contract.package_root()
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


# ---------------------------------------------------------------------------
# Frozen perception binding
# ---------------------------------------------------------------------------


def load_frozen_perception(device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Load the accepted epoch-26 checkpoint into the numerical-recovery runtime."""
    lock_path = contract.perception_lock_path()
    lock_hash = sha256_file(lock_path)
    if lock_hash != contract.PERCEPTION_LOCK_SHA256:
        raise guards.HybridQConfigError("perception forward lock sha256 drift")
    lock = contract.load_perception_lock()

    checkpoint_path = (contract.repository_root() / lock["base_checkpoint"]["path"]).resolve(strict=True)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != contract.FROZEN_CHECKPOINT_SHA256:
        raise guards.HybridQConfigError("frozen checkpoint sha256 drift")

    base = load_base()  # verifies the immutable original source hashes
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "splitfusion_fcos_numerical_recovery_atomic_checkpoint_v1":
        raise guards.HybridQConfigError("frozen checkpoint schema drift")
    if int(checkpoint["epoch"]) != int(lock["base_checkpoint"]["epoch"]):
        raise guards.HybridQConfigError("frozen checkpoint epoch drift")
    recovery = checkpoint["recovery"]
    if recovery["source_commit"] != lock["base_checkpoint"]["training_source_commit"]:
        raise guards.HybridQConfigError("frozen checkpoint training-source commit drift")

    original_experiment = contract.repository_root() / (
        "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1/20260829_214123"
    )
    priors = json.loads((original_experiment / "TRAIN_ONLY_PRIORS.json").read_text(encoding="utf-8"))
    tau = float(recovery["selected_tau"])
    model, build_report = build_recovery_model(priors, tau, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    guards.require_frozen_perception([model])
    guards.require_eval_mode([model])

    binding = {
        "perception_forward_lock_path": contract.PERCEPTION_LOCK_RELPATH,
        "perception_forward_lock_sha256": lock_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(contract.repository_root())),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_training_source_commit": recovery["source_commit"],
        "selected_yaw_tau": tau,
        "runtime": "splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.build_recovery_model",
        "recovery_build_report": {k: v for k, v in build_report.items() if isinstance(v, (str, int, float, bool))},
        "service_policy": "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.apply_p025_service_policy",
        "person_output_threshold": 0.25,
        "eval_mode": True,
        "perception_parameters_requires_grad": False,
    }
    del checkpoint
    return model, base, binding


# ---------------------------------------------------------------------------
# Deterministic train-only batch selection
# ---------------------------------------------------------------------------


def build_train_dataset(base: Any) -> Any:
    config = json.loads(
        (contract.repository_root() / contract.PERCEPTION_LOCK_RELPATH).parent.parent
        .joinpath("splitfusion_fcos_r50_fpn_p2_p7_v1/config.json").read_text(encoding="utf-8")
    )
    dataset_root = (base.common.ROOT / config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache(
        (base.common.ROOT / config["train_depth_cache"]).resolve(strict=True), rows
    )
    dataset = base.data.RouteBDataset(
        dataset_root, "train", contract.RANKER_INIT_SEED, cache, augment=False
    )
    if dataset.augment:
        raise guards.HybridQConfigError("augmentation must be disabled for qualification")
    return dataset


def select_batch_indices(dataset: Any, batch_size: int, batches: int) -> list[list[int]]:
    """Deterministic seeded pick of train frames carrying both vehicle and person GT."""
    eligible = []
    for index, row in enumerate(dataset.rows):
        labels = {entry["label"] for entry in dataset.objects.get(row["sample_id"], ())}
        if "vehicle" in labels and "person" in labels:
            eligible.append(index)
    needed = batch_size * batches
    if len(eligible) < needed:
        raise guards.HybridQConfigError("too few train frames with both vehicle and person GT")
    generator = torch.Generator().manual_seed(contract.RANKER_INIT_SEED)
    order = torch.randperm(len(eligible), generator=generator)[:needed].tolist()
    picked = [eligible[position] for position in order]
    return [picked[start:start + batch_size] for start in range(0, needed, batch_size)]


def collate_batch(base: Any, dataset: Any, indices: Sequence[int]) -> dict[str, Any]:
    return base.data.collate([dataset[index] for index in indices])


def batch_gt_counts(batch: Mapping[str, Any]) -> dict[str, int]:
    vehicles = sum(int((target["labels"] == 0).sum()) for target in batch["targets"])
    persons = sum(int((target["labels"] == 1).sum()) for target in batch["targets"])
    return {"vehicle_gt": vehicles, "person_gt": persons}


# ---------------------------------------------------------------------------
# Frozen forward and the registered D/G/S/A decomposition
# ---------------------------------------------------------------------------


def encode_front(model: torch.nn.Module, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        inputs = batch["input"].to(device, non_blocking=True)
        c2 = model.encode_front(inputs)
    guards.require_frozen_batched_c2(c2, what="frozen C2")
    return c2.detach().float()


def loss_groups_from_c2(
    model: torch.nn.Module, base: Any, c2: torch.Tensor, batch: Mapping[str, Any], *, use_amp: bool
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Registered D/G/S/A groups computed from a caller-supplied C2 tensor.

    This is exactly `recovery_losses.compute_loss_groups` with the trunk forward
    replaced by the hybrid-q C2 boundary; the loss functions themselves are the
    immutable originals (geometry via the numerical-recovery yaw map).
    """
    targets = batch["targets"]
    amp = c2.device.type == "cuda" and use_amp
    with torch.autocast(device_type=c2.device.type, dtype=torch.bfloat16, enabled=amp):
        outputs = model.decode_tail(c2, dense=True)
    with torch.autocast(device_type=c2.device.type, enabled=False):
        detection, _, matched, _ = base.losses.detection_losses(model, outputs, targets)
        geometry, _, _ = recovery_losses.geometry_losses(model, outputs, targets, matched)
        semantic, _ = base.losses.semantic_loss(outputs["semantic_logits"], targets)
        auxiliary, _, _ = base.losses.dense_losses(outputs, batch)
    return outputs, {"D": detection, "G": geometry, "S": semantic, "A": auxiliary}


def service_outputs(model: torch.nn.Module, base: Any, outputs: Mapping[str, Any],
                    batch: Mapping[str, Any], device: torch.device) -> list[dict[str, torch.Tensor]]:
    """Frozen postprocess plus the locked p025 service policy, per frame."""
    calibrations = [
        {"intrinsic": target["intrinsic"].to(device), "extrinsic": target["extrinsic"].to(device)}
        for target in batch["targets"]
    ]
    detections = model.postprocess(outputs, calibrations)
    results = []
    for index, detection in enumerate(detections):
        frame_view = {"semantic_logits": outputs["semantic_logits"][index:index + 1]}
        filtered, _ = apply_p025_service_policy(frame_view, detection)
        results.append(filtered)
    return results


# ---------------------------------------------------------------------------
# 4. q=0 parity
# ---------------------------------------------------------------------------


PARITY_TENSOR_KEYS = (
    "detection.cls_logits", "detection.bbox_regression", "detection.bbox_ctrness",
    "semantic_logits", "semantic_logits_stride4",
    "dense_depth_log1p", "dense_depth_log1p_stride4",
)


def flatten_outputs(outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    for name in ("cls_logits", "bbox_regression", "bbox_ctrness"):
        flat[f"detection.{name}"] = outputs["detection"][name]
    for name in ("semantic_logits", "semantic_logits_stride4",
                 "dense_depth_log1p", "dense_depth_log1p_stride4"):
        flat[name] = outputs[name]
    for level_index, level in enumerate(outputs["geometry"]):
        for name, value in level.items():
            flat[f"geometry.l{level_index}.{name}"] = value
    return flat


def q0_parity(model: torch.nn.Module, base: Any, ranker: torch.nn.Module,
              batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    ranker_calls = {"count": 0}

    def _count(_module, _args, _output):
        ranker_calls["count"] += 1

    handle = ranker.register_forward_hook(_count)
    try:
        with torch.inference_mode():
            inputs = batch["input"].to(device, non_blocking=True)
            reference = model(inputs, dense=True)  # existing noAE identity path
            reference_service = service_outputs(model, base, reference, batch, device)

            c2 = model.encode_front(inputs).float()
            frames, bypassed, roundtrip_exact = [], [], []
            payload_bytes = []
            for index in range(c2.shape[0]):
                frame = c2[index].clone()
                masked, selection = select_and_apply(frame, ranker, contract.PARITY_Q)
                bypassed.append(masked is frame and selection is None)
                payload = codec.encode(masked, contract.PARITY_Q)
                payload_bytes.append(payload.total_bytes)
                decoded, decoded_q = codec.decode(payload)
                roundtrip_exact.append(
                    decoded_q == contract.PARITY_Q
                    and torch.equal(decoded, masked.detach().cpu())
                )
                frames.append(decoded.to(device))
            hybrid_c2 = torch.stack(frames)
            hybrid = model.decode_tail(hybrid_c2, dense=True)
            hybrid_service = service_outputs(model, base, hybrid, batch, device)
    finally:
        handle.remove()

    reference_flat, hybrid_flat = flatten_outputs(reference), flatten_outputs(hybrid)
    if set(reference_flat) != set(hybrid_flat):
        raise guards.HybridQPayloadError("q=0 parity output key set drift")
    mismatched = sorted(name for name in reference_flat
                        if not torch.equal(reference_flat[name], hybrid_flat[name]))
    anchors_equal = all(torch.equal(a, b) for a, b in zip(reference["anchors"], hybrid["anchors"]))

    service_mismatch: list[str] = []
    for index, (want, got) in enumerate(zip(reference_service, hybrid_service)):
        if set(want) != set(got):
            service_mismatch.append(f"frame{index}:keys")
            continue
        for name in sorted(want):
            if not torch.equal(want[name], got[name]):
                service_mismatch.append(f"frame{index}:{name}")

    return {
        "ranker_invocations_at_q0": ranker_calls["count"],
        "ranker_bypassed_every_frame": all(bypassed),
        "framed_encode_decode_bit_identical": all(roundtrip_exact),
        "framed_q0_payload_bytes": sorted(set(payload_bytes)),
        "compared_output_tensors": len(reference_flat),
        "raw_output_tensors_bit_identical": not mismatched,
        "mismatched_output_tensors": mismatched,
        "anchors_bit_identical": anchors_equal,
        "p025_service_frames_compared": len(reference_service),
        "p025_service_outputs_bit_identical": not service_mismatch,
        "p025_service_mismatched_fields": service_mismatch,
        "p025_service_detection_counts": [int(item["scores"].numel()) for item in hybrid_service],
        "precision": "fp32_no_autocast_inference_mode",
    }


# ---------------------------------------------------------------------------
# 5. Teacher-map qualification
# ---------------------------------------------------------------------------


def teacher_maps_for_batch(model: torch.nn.Module, base: Any, c2: torch.Tensor,
                           batch: Mapping[str, Any], *, use_amp: bool) -> dict[str, Any]:
    """Per-frame D/G/S/A importance maps from a detached FP32 C2 leaf."""
    leaf = c2.detach().clone().float().requires_grad_(True)
    _outputs, groups = loss_groups_from_c2(model, base, leaf, batch, use_amp=use_amp)

    group_losses: dict[str, float] = {}
    group_grads: dict[str, torch.Tensor | None] = {}
    disconnected: dict[str, str] = {}
    remaining = [name for name in contract.TEACHER_GROUPS if groups.get(name) is not None]
    for position, name in enumerate(remaining):
        loss = groups[name]
        guards.require_finite(loss.detach(), f"group {name} loss")
        group_losses[name] = float(loss.detach())
        grad, = torch.autograd.grad(
            loss, leaf, retain_graph=position < len(remaining) - 1, allow_unused=True
        )
        # An absent group is recorded, never replaced by a substitute gradient.
        if grad is None:
            disconnected[name] = "no_gradient_path_to_c2"
            group_grads[name] = None
            continue
        group_grads[name] = grad.detach()
    del _outputs, groups

    per_frame = []
    for index in range(leaf.shape[0]):
        frame_c2 = leaf[index].detach()
        result = training.build_teacher_maps(
            frame_c2,
            {name: (None if grad is None else grad[index]) for name, grad in group_grads.items()},
            task_losses=group_losses,
        )
        if not result.is_supervisable:
            raise guards.HybridQNumericalError("frame has no valid teacher group")
        importance = result.importance
        guards.require_finite(importance, "combined teacher map")
        if bool((importance < 0).any()) or float(importance.sum()) <= 0.0:
            raise guards.HybridQNumericalError("combined teacher map is not positive")
        per_frame.append(result)

    valid = per_frame[0].valid_groups
    if any(item.valid_groups != valid for item in per_frame):
        raise guards.HybridQNumericalError("teacher group validity differs across frames")
    return {
        "teacher": per_frame,
        "task_losses": group_losses,
        "valid_groups": list(valid),
        "excluded_groups": dict(per_frame[0].excluded_groups),
        "gradient_mass": {name: float(value) for name, value in per_frame[0].gradient_mass.items()},
        "disconnected_groups": disconnected,
        "importance": [item.importance.detach() for item in per_frame],
    }


# ---------------------------------------------------------------------------
# 7. Mask nesting and transport
# ---------------------------------------------------------------------------


def mask_and_transport(c2_frame: torch.Tensor, scores: torch.Tensor) -> dict[str, Any]:
    rows = []
    previous_mask: torch.Tensor | None = None
    nested = True
    for q in NESTED_Q_VALUES:
        selection = select_cells(scores, q)
        masked = c2_frame * selection.keep_mask.unsqueeze(0).to(c2_frame.dtype)
        payload = codec.encode(masked, q, selection)
        decoded, decoded_q = codec.decode(payload)
        decoded = decoded.to(c2_frame.device)
        keep_mask = selection.keep_mask.unsqueeze(0).expand_as(c2_frame)
        retained_exact = bool(torch.equal(decoded[keep_mask], c2_frame[keep_mask]))
        dropped_zero = bool((decoded[~keep_mask] == 0).all())
        if previous_mask is not None:
            nested = nested and bool((selection.keep_mask & ~previous_mask).sum() == 0)
        previous_mask = selection.keep_mask
        rows.append({
            "q": q,
            "keep_count": selection.keep_count,
            "registered_keep_count": contract.keep_count(q),
            "keep_count_exact": selection.keep_count == contract.keep_count(q),
            "drop_count": selection.drop_count,
            "framed_payload_bytes": payload.total_bytes,
            "framed_ratio": payload.framed_ratio,
            "decoded_q": decoded_q,
            "retained_values_bit_identical": retained_exact,
            "dropped_cells_exact_zero": dropped_zero,
        })
    return {
        "masks_nested_over_increasing_q": nested,
        "keep_counts_exact": all(row["keep_count_exact"] for row in rows),
        "codec_roundtrip_exact": all(
            row["retained_values_bit_identical"] and row["dropped_cells_exact_zero"] for row in rows
        ),
        "per_q": rows,
    }


# ---------------------------------------------------------------------------
# 8. Bounded latency measurement
# ---------------------------------------------------------------------------


def _timed(function, repetitions: int, warmup: int) -> tuple[list[float], Any]:
    result = None
    for _ in range(warmup):
        result = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, result


def _stats(samples: Sequence[float]) -> dict[str, float]:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {"median_ms": statistics.median(ordered), "p95_ms": ordered[index], "n": len(ordered)}


def measure_latency(ranker: torch.nn.Module, c2_frame: torch.Tensor, device: torch.device) -> dict[str, Any]:
    with torch.no_grad():
        ranker_samples, scores = _timed(
            lambda: ranker.score_cells(c2_frame), LATENCY_REPETITIONS, LATENCY_WARMUP
        )
        scores = scores.detach()

        stages: dict[str, Any] = {}
        payload_table = []
        for q in contract.REGISTERED_Q_VALUES:
            if contract.drop_count(q) == 0:
                selection_samples = [0.0]
                selected = None
                masked = c2_frame
            else:
                selection_samples, selected = _timed(
                    lambda: select_cells(scores, q), LATENCY_REPETITIONS, LATENCY_WARMUP
                )
                masked = c2_frame * selected.keep_mask.unsqueeze(0).to(c2_frame.dtype)
            pack_samples, payload = _timed(
                lambda: codec.encode(masked, q, selected), LATENCY_REPETITIONS, LATENCY_WARMUP
            )
            unpack_samples, _ = _timed(
                lambda: codec.decode(payload)[0].to(device), LATENCY_REPETITIONS, LATENCY_WARMUP
            )
            total = [
                r + s + p + u for r, s, p, u in zip(
                    ranker_samples,
                    selection_samples * len(ranker_samples) if len(selection_samples) == 1 else selection_samples,
                    pack_samples,
                    unpack_samples,
                )
            ]
            stages[f"q={q:.2f}"] = {
                "deterministic_selection": _stats(selection_samples),
                "gpu_to_cpu_transfer_and_packing": _stats(pack_samples),
                "unpacking_and_cpu_to_gpu_reconstruction": _stats(unpack_samples),
                "total_hybrid_q_preparation_overhead": _stats(total),
            }
            payload_table.append({
                "q": q,
                "keep_count": contract.keep_count(q),
                "framed_payload_bytes": payload.total_bytes,
                "framed_ratio": payload.framed_ratio,
                "header_bytes": payload.header_bytes,
                "mask_bytes": payload.mask_bytes,
                "value_bytes": payload.value_bytes,
            })
    return {
        "note": "single-frame diagnostic; no tuning or selection rewrite is authorized on it",
        "repetitions": LATENCY_REPETITIONS,
        "warmup": LATENCY_WARMUP,
        "ranker_gpu_time": _stats(ranker_samples),
        "per_q": stages,
        "framed_payload_bytes": payload_table,
        "zstd_applied": False,
    }


# ---------------------------------------------------------------------------
# 6. Disposable optimizer qualification
# ---------------------------------------------------------------------------


def disposable_updates(model: torch.nn.Module, base: Any, batches: Sequence[Mapping[str, Any]],
                       teachers: Sequence[Mapping[str, Any]], references: training.ReferenceMedians,
                       device: torch.device, *, use_amp: bool,
                       frozen_snapshot: Mapping[str, torch.Tensor],
                       steps_taken: list[int]) -> dict[str, Any]:
    ranker = build_ranker().to(device)
    optimizer = training.build_ranker_optimizer(ranker, frozen_modules=[model])
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    guards.require_optimizer_owns_only(optimizer, ranker.parameters())
    trainable = sum(p.numel() for p in ranker.parameters() if p.requires_grad)
    ranker_ids = {id(p) for p in ranker.parameters()}
    if trainable != contract.RANKER_PARAMETER_COUNT or any(id(p) not in ranker_ids for p in owned):
        raise guards.HybridQOwnershipError("trainable parameter set is not exactly the ranker")

    qualification = training.GradientQualification.for_module(ranker, window=4)
    records = []

    def _step(loss: torch.Tensor, label: str, q: float | None) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nonzero = qualification.observe(ranker, loss=loss)
        # clip_grad_norm_ returns the total norm observed *before* clipping.
        norm = training.clip_ranker_gradients(ranker)
        grad_norms = {
            name: float(p.grad.detach().norm()) for name, p in ranker.named_parameters()
        }
        optimizer.step()
        steps_taken.append(1)
        training.require_post_step_health(ranker, optimizer)
        guards.require_module_state_unchanged(model, frozen_snapshot)
        if any(p.grad is not None for p in model.parameters()):
            raise guards.HybridQOwnershipError("frozen perception parameter received a gradient")
        records.append({
            "update": len(records) + 1,
            "kind": label,
            "q": q,
            "loss": float(loss.detach()),
            "global_grad_norm_pre_clip": float(norm),
            "grad_clip_applied": float(norm) > contract.GRAD_CLIP_GLOBAL_NORM,
            "all_tensors_nonzero": bool(nonzero),
            "per_tensor_grad_norm_post_clip": grad_norms,
        })

    # 1. listwise distillation
    batch, teacher = batches[0], teachers[0]
    c2 = encode_front(model, batch, device)
    scores = ranker(c2)
    distillation = torch.stack([
        training.ranker_distillation_loss(scores[i], teacher["importance"][i])
        for i in range(c2.shape[0])
    ]).mean()
    _step(distillation, "listwise_distillation", None)
    del c2, scores, distillation

    # 2-4. q-aware updates over the locked cycle
    for position, q in enumerate(UPDATE_Q_ORDER):
        if contract.drop_count(q) == 0:
            raise guards.HybridQConfigError("q=0 must never be a training update")
        batch, teacher = batches[position + 1], teachers[position + 1]
        c2 = encode_front(model, batch, device)
        scores = ranker(c2)
        masked = torch.stack([
            training.masked_c2_forward(c2[i], training.straight_through_mask(scores[i], q))
            for i in range(c2.shape[0])
        ])
        _outputs, groups = loss_groups_from_c2(model, base, masked, batch, use_amp=use_amp)
        del _outputs
        distillation = torch.stack([
            training.ranker_distillation_loss(scores[i], teacher["importance"][i])
            for i in range(c2.shape[0])
        ]).mean()
        objective = training.q_aware_objective(
            {name: groups[name] for name in references.medians}, distillation, references
        )
        _step(objective, "q_aware", q)
        del c2, scores, masked, groups, distillation, objective

    qualification.require_qualified()
    summary = {
        "trainable_parameters": trainable,
        "all_trainable_parameters_belong_to_ranker": True,
        "named_ranker_tensors": list(qualification.parameter_names),
        "named_ranker_tensor_count": len(qualification.parameter_names),
        "updates": records,
        "window_complete": qualification.window_complete(),
        "disconnected_tensors": list(qualification.disconnected()),
        "never_nonzero_tensors": list(qualification.never_nonzero()),
        "zero_gradient_batches": [
            {"update": index, "tensors": list(names)}
            for index, names in qualification.zero_gradient_batches
        ],
        "missing_gradient_batches": [
            {"update": index, "tensors": list(names)}
            for index, names in qualification.missing_gradient_batches
        ],
        "qualified": qualification.qualified(),
        "q0_used_as_training_update": False,
        "optimizer": "AdamW lr=1e-3 wd=1e-4 constant, clip 5.0, ranker parameters only",
    }
    del ranker, optimizer
    torch.cuda.empty_cache()
    summary["disposable_ranker_retained"] = False
    summary["disposable_ranker_state_discarded"] = True
    return summary


# ---------------------------------------------------------------------------
# One bounded attempt at a physical batch size
# ---------------------------------------------------------------------------


def run_attempt(model: torch.nn.Module, base: Any, dataset: Any, batch_size: int,
                device: torch.device, *, use_amp: bool) -> dict[str, Any]:
    steps_taken: list[int] = []
    frozen_snapshot = guards.snapshot_module_state(model)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        index_batches = select_batch_indices(dataset, batch_size, QUALIFICATION_BATCHES)
        batches = [collate_batch(base, dataset, indices) for indices in index_batches]
        sample_ids = [list(batch["sample_ids"]) for batch in batches]

        parity_ranker = build_ranker().to(device).eval()
        parity = q0_parity(model, base, parity_ranker, batches[0], device)
        if not (parity["ranker_invocations_at_q0"] == 0
                and parity["ranker_bypassed_every_frame"]
                and parity["framed_encode_decode_bit_identical"]
                and parity["raw_output_tensors_bit_identical"]
                and parity["anchors_bit_identical"]
                and parity["p025_service_outputs_bit_identical"]):
            raise guards.HybridQPayloadError(f"q=0 parity is not exact: {parity}")
        guards.require_module_state_unchanged(model, frozen_snapshot)
        parity["frozen_state_unchanged"] = True

        teachers = [
            teacher_maps_for_batch(model, base, encode_front(model, batch, device), batch, use_amp=use_amp)
            for batch in batches
        ]
        torch.cuda.empty_cache()

        medians = {}
        excluded_scales = {}
        for group in contract.TEACHER_GROUPS:
            values = [t["task_losses"][group] for t in teachers if group in t["task_losses"]]
            if not values or statistics.median(values) <= 0.0:
                excluded_scales[group] = "absent_or_non_positive_q0_task_loss"
                continue
            medians[group] = statistics.median(values)
        references = training.ReferenceMedians(medians=medians)

        updates = disposable_updates(
            model, base, batches, teachers, references, device,
            use_amp=use_amp, frozen_snapshot=frozen_snapshot, steps_taken=steps_taken,
        )

        c2_frame = encode_front(model, batches[0], device)[0]
        with torch.no_grad():
            scores = build_ranker().to(device).eval().score_cells(c2_frame)
        transport = mask_and_transport(c2_frame, scores)
        latency = measure_latency(build_ranker().to(device).eval(), c2_frame, device)
    except Exception as error:  # noqa: BLE001 - retry policy is batch sizing only
        if _is_oom(error) and not steps_taken:
            torch.cuda.empty_cache()
            raise OutOfMemory(str(error)) from error
        raise

    guards.require_module_state_unchanged(model, frozen_snapshot)
    return {
        "physical_batch": batch_size,
        "batch_frame_indices": index_batches,
        "sample_ids": sample_ids,
        "batch_gt_counts": [batch_gt_counts(batch) for batch in batches],
        "parity": parity,
        "teacher": {
            "valid_groups": teachers[0]["valid_groups"],
            "per_batch": [
                {
                    "valid_groups": t["valid_groups"],
                    "excluded_groups": t["excluded_groups"],
                    "task_losses": t["task_losses"],
                    "gradient_mass": t["gradient_mass"],
                    "disconnected_groups": t["disconnected_groups"],
                    "combined_map_min": float(min(m.min() for m in t["importance"])),
                    "combined_map_sum": [float(m.sum()) for m in t["importance"]],
                }
                for t in teachers
            ],
            "normalization": contract.TEACHER_NORMALIZATION,
            "combination": contract.TEACHER_GROUP_COMBINATION,
            "cache_written": False,
        },
        "disposable_reference_scales": {
            "values": medians,
            "excluded": excluded_scales,
            "source": "q=0 task losses on the same train-only qualification batches",
            "disposable": True,
            "must_not_become_phase4_frozen_medians": True,
        },
        "updates": updates,
        "mask_and_transport": transport,
        "latency": latency,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
        "frozen_state_unchanged_at_end": True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid-q Phase-3 bounded train-only GPU qualification")
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the Phase-3 qualification requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)

    locked = contract.load_locked_config()
    if int(locked["ranker"]["parameter_count"]) != 2144:
        raise guards.HybridQConfigError("locked ranker parameter count is not 2144")

    model, base, binding = load_frozen_perception(device)
    dataset = build_train_dataset(base)

    attempts = []
    result = None
    for batch_size in PHYSICAL_BATCH_LADDER:
        try:
            result = run_attempt(model, base, dataset, batch_size, device, use_amp=True)
            attempts.append({"physical_batch": batch_size, "outcome": "ok"})
            break
        except OutOfMemory as error:
            attempts.append({"physical_batch": batch_size, "outcome": "cuda_oom_before_optimizer_step",
                             "detail": str(error).splitlines()[0][:300]})
            gc.collect()
            torch.cuda.empty_cache()
    if result is None:
        raise RuntimeError("no physical batch size in the ladder survived without CUDA OOM")

    report = {
        "schema": "splitfusion_fcos_hybrid_q_phase3_gpu_qualification_v1",
        "terminal": "HYBRID_Q_PHASE3_QUALIFIED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "train_only": True,
            "augmentation": False,
            "validation_or_test_accessed": False,
            "teacher_cache_written": False,
            "epochs_trained": 0,
            "evaluation_run": False,
            "carla_launched": False,
            "zstd_run": False,
            "seed": contract.RANKER_INIT_SEED,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "training_precision": "bf16 autocast tail, fp32 C2 boundary and losses (registered)",
        },
        "perception_binding": binding,
        "hybrid_q_source_sha256": package_source_hashes(),
        "ranker_correction": {
            "change": "final Conv2d(8,1,kernel_size=1) now has bias=False",
            "parameter_count": contract.RANKER_PARAMETER_COUNT,
            "mac_count_112x192": contract.ranker_mac_count(),
            "reason": "a global scalar score bias cannot alter cell ranking; listwise softmax "
                      "distillation is invariant to it and a straight-through gradient on it "
                      "would not correspond to a change in the hard mask",
        },
        "batch_size_qualification": {
            "ladder": list(PHYSICAL_BATCH_LADDER),
            "attempts": attempts,
            "selected_physical_batch": result["physical_batch"],
            "note": "runtime sizing only; the effective scientific batch is unchanged",
        },
        "result": result,
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "qualification_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"terminal": report["terminal"],
                      "physical_batch": result["physical_batch"],
                      "peak_allocated_vram_mib": result["peak_allocated_vram_mib"],
                      "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
