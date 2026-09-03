"""Phase-9B bounded GPU qualification of the SplitFusion AE128 implementation.

Qualification only. This runner takes exactly five *disposable* updates on a
freshly constructed AE128 -- one Stage-A update at q=0 and one Stage-B cycle
over q in {0, 0.30, 0.50, 0.70} -- purely to show that the Phase-9A
implementation executes correctly on the accelerator. It is **not** scientific
AE training: the optimizer settings are diagnostic, the AE and optimizer are
discarded at the end, and no checkpoint is written.

Scope, deliberately: fit frames only, no reserved train-holdout, no validation,
no test, no accuracy scoring, no AE64/AE32, no CARLA. The frozen perception
model and the stable epoch-4 ranker are loaded, hash-bound, frozen and proven
unchanged; the Phase-4 teacher cache is read, never rebuilt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
    phase5_common,
    teacher_cache,
    training,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    collate_batch,
    encode_front,
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.ranker import build_ranker
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
    ZstdWireCodec,
    implementation_report,
)
from . import ae_composition, ae_contract, ae_family_dispatch, ae_loss, ae_uint8_transport
from .ae_model import ae_parameters, build_split_feature_ae


EXECUTE_TOKEN = "SPLITFUSION_AE128_PHASE9B_GPU_QUALIFICATION"
TERMINAL_QUALIFIED = "SPLITFUSION_AE128_GPU_QUALIFIED_AWAITING_TRAINING_REVIEW"
TERMINAL_FAILED = "SPLITFUSION_AE128_GPU_QUALIFICATION_FAILED"
SCHEMA = "splitfusion_fcos_ae128_phase9b_gpu_qualification_v1"

QUALIFICATION_BOTTLENECK = 128
PRIMARY_BATCH = 16
FALLBACK_BATCH = 8
VRAM_BUDGET_GIB = 30.0
VRAM_BUDGET_BYTES = int(VRAM_BUDGET_GIB * (1024 ** 3))

# One Stage-A update at q=0, then exactly one Stage-B cycle.
STAGE_A_Q = ae_contract.AE_STAGE_A_Q
STAGE_B_CYCLE = ae_contract.AE_STAGE_B_Q_CYCLE
UPDATE_COUNT = 1 + len(STAGE_B_CYCLE)

TAIL_Q_VALUES = (0.00, 0.70)
TRANSPORT_Q_VALUES = (0.00, 0.30)

LATENCY_WARMUP = 20
LATENCY_REPETITIONS = 100

# Diagnostic only. These are not the final scientific training settings.
DIAGNOSTIC_LR = 1e-3
DIAGNOSTIC_WEIGHT_DECAY = 1e-4

# The routing tag is a routing discriminator, not a checkpoint identity, and no
# AE checkpoint exists yet; this qualification tag exists only so the deployable
# encode/dispatch paths (which refuse an unbound tag) can be exercised at all.
ROUTING_TAG_LABEL = "splitfusion_fcos_ae128_phase9b_gpu_qualification_routing_tag_v1"

# The four bindings this qualification was authorized against, restated so a
# contract edit cannot silently move them.
REQUESTED_PERCEPTION_CHECKPOINT_SHA256 = (
    "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f"
)
REQUESTED_STABLE_RANKER_SHA256 = (
    "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5"
)
REQUESTED_P025_FORWARD_LOCK_SHA256 = (
    "86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1"
)
REQUESTED_HYBRID_Q_LOCKED_CONFIG_SHA256 = (
    "b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6"
)


class OutOfMemory(RuntimeError):
    """CUDA ran out of memory during one bounded attempt."""


class VramBudgetExceeded(RuntimeError):
    """The attempt completed but exceeded the registered peak-VRAM budget."""


def _is_oom(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


class CountingWireCodec(ZstdWireCodec):
    """Counts decompressions so the receive path can be shown to do exactly one."""

    def __init__(self) -> None:
        super().__init__()
        self.decompressions = 0

    def decompress(self, frame: bytes, *, expected_bytes: int | None = None) -> bytes:
        self.decompressions += 1
        return super().decompress(frame, expected_bytes=expected_bytes)

    def decompress_bytes(self, frame: bytes) -> bytes:
        self.decompressions += 1
        return super().decompress_bytes(frame)


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def ae_package_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def bind_inputs() -> dict[str, Any]:
    """Hash-bind every frozen input, reusing the Phase-5 binding verbatim."""
    binding = phase5_common.bind_inputs()
    delta = phase5_common.source_delta(binding)

    root = contract.repository_root()
    ranker_path = (root / contract.VALIDATION_RANKER_RELPATH).resolve(strict=True)
    ranker_hash = sha256_file(ranker_path)
    if ranker_hash != contract.VALIDATION_RANKER_SHA256:
        raise guards.HybridQConfigError("stable epoch-4 ranker sha256 drift")

    authorized = {
        "frozen_perception_checkpoint": (
            binding["frozen_checkpoint"]["sha256"],
            REQUESTED_PERCEPTION_CHECKPOINT_SHA256,
        ),
        "stable_epoch4_ranker": (ranker_hash, REQUESTED_STABLE_RANKER_SHA256),
        "p025_forward_lock": (
            binding["perception_forward_lock"]["sha256"],
            REQUESTED_P025_FORWARD_LOCK_SHA256,
        ),
        "hybrid_q_locked_config": (
            binding["hybrid_q_locked_config"]["sha256"],
            REQUESTED_HYBRID_Q_LOCKED_CONFIG_SHA256,
        ),
    }
    for name, (observed, requested) in authorized.items():
        if observed != requested:
            raise guards.HybridQConfigError(
                f"{name} sha256 is not the authorized Phase-9B binding"
            )

    return {
        **binding,
        "stable_epoch4_ranker": {
            "path": contract.VALIDATION_RANKER_RELPATH,
            "sha256": ranker_hash,
            "epoch": contract.VALIDATION_RANKER_EPOCH,
        },
        "hybrid_q_source_delta_since_phase4": delta,
        "ae_package_source_sha256": ae_package_source_hashes(),
        "authorized_bindings_match_request": {
            name: True for name in authorized
        },
    }


def state_hashes(module: torch.nn.Module) -> tuple[dict[str, str], str]:
    """Per-tensor and aggregate sha256 over every parameter *and* buffer."""
    per_tensor: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for name, tensor in sorted(guards.snapshot_module_state(module).items()):
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()
        per_tensor[name] = digest
        aggregate.update(name.encode("utf-8"))
        aggregate.update(bytes.fromhex(digest))
    return per_tensor, aggregate.hexdigest()


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


# ---------------------------------------------------------------------------
# Fit-frame selection and the exact teacher-cache join
# ---------------------------------------------------------------------------


def select_fit_frames(
    partition: teacher_cache.SplitPartition, frames: int
) -> list[dict[str, Any]]:
    """Deterministic seeded pick of fit frames; the holdout is never considered."""
    if frames > len(partition.fit_indices):
        raise guards.HybridQConfigError("more qualification frames than fit frames")
    generator = torch.Generator().manual_seed(ae_contract.AE_INIT_BASE_SEED)
    order = torch.randperm(len(partition.fit_indices), generator=generator)[:frames]
    picked = []
    for position in order.tolist():
        picked.append(
            {
                "fit_position": int(position),
                "dataset_index": int(partition.fit_indices[position]),
                "sample_id": str(partition.fit_sample_ids[position]),
            }
        )
    return picked


def join_teacher_records(
    selected: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Join every selected sample ID to its cached record by identity.

    The fit partition was written to the cache first and in registered order, so
    the fit position gives the shard to open; but the record is then looked up in
    that shard's own `sample_ids` **by identity**, and a frame whose id is absent
    from the shard the position pointed at fails closed. Positional ordering is
    a hint here, never the join key.
    """
    cache_root = phase5_common.teacher_cache_root()
    entries = [
        entry for entry in manifest["shards"]["entries"] if entry["split"] == "fit"
    ]
    if not entries:
        raise guards.HybridQConfigError("the teacher cache holds no fit shard")

    by_shard: dict[int, list[Mapping[str, Any]]] = {}
    for frame in selected:
        position = int(frame["fit_position"])
        hits = [
            index
            for index, entry in enumerate(entries)
            if int(entry["cache_index_start"]) <= position < int(entry["cache_index_end"])
        ]
        if len(hits) != 1:
            raise guards.HybridQConfigError(
                f"fit position {position} does not fall in exactly one fit shard"
            )
        by_shard.setdefault(hits[0], []).append(frame)

    joined: dict[str, dict[str, Any]] = {}
    for shard_index, frames in sorted(by_shard.items()):
        entry = entries[shard_index]
        payload = teacher_cache.load_shard(cache_root / entry["path"])
        if payload["schema"] != teacher_cache.SHARD_SCHEMA:
            raise guards.HybridQConfigError(f"{entry['path']} schema drift")
        if payload["perception_checkpoint_sha256"] != contract.FROZEN_CHECKPOINT_SHA256:
            raise guards.HybridQConfigError(f"{entry['path']} checkpoint binding drift")
        offsets = {
            str(sample_id): offset
            for offset, sample_id in enumerate(payload["sample_ids"])
        }
        for frame in frames:
            sample_id = str(frame["sample_id"])
            offset = offsets.get(sample_id)
            if offset is None:
                raise guards.HybridQConfigError(
                    f"{sample_id} is absent from {entry['path']}; the cache join is "
                    "not an identity match"
                )
            label = str(payload["splits"][offset])
            if label != "fit":
                raise guards.HybridQOwnershipError(
                    f"{sample_id} is labelled {label}; only fit frames may be used"
                )
            if contract.split_for_episode(str(payload["episode_ids"][offset])) != "fit":
                raise guards.HybridQOwnershipError(
                    f"{sample_id} belongs to a non-fit episode"
                )
            importance = payload["importance"][offset]
            if importance.dtype is not torch.float32:
                raise guards.HybridQPayloadError(f"{sample_id} cached map is not FP32")
            if tuple(importance.shape) != contract.SPLIT_SPATIAL_SHAPE:
                raise guards.HybridQPayloadError(f"{sample_id} cached map shape drift")
            valid = tuple(str(name) for name in payload["valid_groups"][offset])
            if len(valid) < ae_contract.AE_MIN_VALID_TASK_GROUPS:
                raise guards.HybridQConfigError(
                    f"{sample_id} carries {len(valid)} valid teacher groups, "
                    f"below the required {ae_contract.AE_MIN_VALID_TASK_GROUPS}"
                )
            joined[sample_id] = {
                "shard": entry["path"],
                "shard_sha256": entry["sha256"],
                "offset_in_shard": int(offset),
                "cache_index": int(entry["cache_index_start"]) + int(offset),
                "fit_position": int(frame["fit_position"]),
                "valid_groups": valid,
                "excluded_groups": {
                    str(key): str(value)
                    for key, value in dict(payload["excluded_groups"][offset]).items()
                },
                "importance": importance.detach().clone().contiguous(),
            }
        del payload
    if len(joined) != len(selected):
        raise guards.HybridQConfigError("teacher join did not cover every frame")
    return joined


def teacher_batch(
    sample_ids: Sequence[str], joined: Mapping[str, Mapping[str, Any]]
) -> ae_loss.CachedTeacherBatch:
    rows = [joined[str(sample_id)] for sample_id in sample_ids]
    return ae_loss.CachedTeacherBatch(
        importance=torch.stack([row["importance"] for row in rows]).contiguous(),
        valid_groups=tuple(tuple(row["valid_groups"]) for row in rows),
        excluded_groups=tuple(dict(row["excluded_groups"]) for row in rows),
    )


# ---------------------------------------------------------------------------
# Per-frame reconstruction error and gradient reporting
# ---------------------------------------------------------------------------


def _summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(array.max()),
        "min": float(array.min()),
        "mean": float(array.mean()),
    }


def per_frame_errors(
    c2: torch.Tensor, reconstructed: torch.Tensor, teacher: ae_loss.CachedTeacherBatch
) -> dict[str, Any]:
    """Per-frame plain and importance-weighted normalized reconstruction error.

    Exactly the two components of `ae_loss.task_aware_reconstruction_loss`, but
    normalized within each frame instead of over the batch, so a single frame's
    error is reported on its own scale.
    """
    with torch.no_grad():
        target = c2.detach()
        estimate = reconstructed.detach()
        frames = int(target.shape[0])
        cell_error = (estimate - target).pow(2).sum(dim=1)
        cell_energy = target.pow(2).sum(dim=1)

        weights = teacher.importance.detach().to(
            device=target.device, dtype=torch.float32
        )
        mass = weights.reshape(frames, -1).sum(dim=1).reshape(frames, 1, 1)
        weights = weights / mass

        plain = cell_error.reshape(frames, -1).sum(dim=1) / cell_energy.reshape(
            frames, -1
        ).sum(dim=1)
        weighted = (weights * cell_error).reshape(frames, -1).sum(dim=1) / (
            weights * cell_energy
        ).reshape(frames, -1).sum(dim=1)
        guards.require_finite(plain, "per-frame plain reconstruction error")
        guards.require_finite(weighted, "per-frame weighted reconstruction error")

    plain_values = [float(value) for value in plain]
    weighted_values = [float(value) for value in weighted]
    return {
        "plain": plain_values,
        "importance_weighted": weighted_values,
        "plain_summary": _summarize(plain_values),
        "importance_weighted_summary": _summarize(weighted_values),
    }


def gradient_norms(autoencoder: torch.nn.Module) -> dict[str, Any]:
    """Global, per-module and per-tensor gradient norms for one update."""
    per_tensor: dict[str, float] = {}
    total = 0.0
    for name, parameter in autoencoder.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm())
        per_tensor[name] = value
        total += value ** 2
    modules = ("project", "latent_context", "expand", "spatial_context")
    per_module = {
        module: float(
            np.sqrt(
                sum(
                    value ** 2
                    for name, value in per_tensor.items()
                    if name.split(".")[0] == module
                )
            )
        )
        for module in modules
    }
    encoder = float(np.sqrt(per_module["project"] ** 2 + per_module["latent_context"] ** 2))
    decoder = float(np.sqrt(per_module["expand"] ** 2 + per_module["spatial_context"] ** 2))
    return {
        "global": float(np.sqrt(total)),
        "per_module": per_module,
        "encoder": encoder,
        "decoder": decoder,
        "per_tensor": per_tensor,
    }


def _finite_state(autoencoder: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, bool]:
    guards.require_module_parameters_finite(autoencoder, "AE128")
    guards.require_finite_optimizer_state(optimizer)
    return {"ae_parameters_finite": True, "optimizer_state_finite": True}


# ---------------------------------------------------------------------------
# Frozen tail structural check
# ---------------------------------------------------------------------------


def shape_signature(value: Any, path: str = "output") -> dict[str, list[int]]:
    signature: dict[str, list[int]] = {}
    if isinstance(value, torch.Tensor):
        signature[path] = list(value.shape)
    elif isinstance(value, Mapping):
        for name, child in value.items():
            signature.update(shape_signature(child, f"{path}.{name}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            signature.update(shape_signature(child, f"{path}[{index}]"))
    return signature


def require_tree_finite(value: Any, path: str = "output") -> int:
    checked = 0
    if isinstance(value, torch.Tensor):
        guards.require_finite(value, path)
        return 1
    if isinstance(value, Mapping):
        for name, child in value.items():
            checked += require_tree_finite(child, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            checked += require_tree_finite(child, f"{path}[{index}]")
    return checked


def frozen_tail_check(
    model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    ranker: torch.nn.Module,
    c2_frame: torch.Tensor,
) -> dict[str, Any]:
    """Run the unchanged frozen tail once per q on AE-reconstructed C2.

    Structural only: shapes must match the signature the same tail produces from
    the original C2, and every output tensor must be finite. No accuracy metric
    is computed and no prediction is scored.
    """
    with torch.no_grad():
        reference = model.decode_tail(c2_frame.unsqueeze(0), dense=True)
        reference_signature = shape_signature(reference)
        del reference

        rows = []
        for q in TAIL_Q_VALUES:
            composition = ae_composition.compose(c2_frame, autoencoder, ranker, q)
            reconstructed = autoencoder.decode(
                composition.masked_latent, composition.keep_mask
            )
            guards.require_frozen_c2(reconstructed, what="reconstructed C2")
            outputs = model.decode_tail(reconstructed.unsqueeze(0), dense=True)
            signature = shape_signature(outputs)
            tensors = require_tree_finite(outputs)
            rows.append(
                {
                    "q": composition.plan.wire_q,
                    "keep_count": composition.keep_count,
                    "ranker_used": composition.ranker_used,
                    "reconstructed_c2_shape": list(reconstructed.shape),
                    "tail_output_tensors": tensors,
                    "all_tail_outputs_finite": True,
                    "shape_signature_matches_original_c2": signature
                    == reference_signature,
                }
            )
            del composition, reconstructed, outputs
    if not all(row["shape_signature_matches_original_c2"] for row in rows):
        raise guards.HybridQPayloadError(
            "the frozen tail produced an unexpected output shape on reconstructed C2"
        )
    return {
        "reference_signature_tensor_count": len(reference_signature),
        "accuracy_scored": False,
        "per_q": rows,
    }


# ---------------------------------------------------------------------------
# Raw-byte transport round trip
# ---------------------------------------------------------------------------


def transport_round_trip(
    autoencoder: torch.nn.Module, ranker: torch.nn.Module, c2_frame: torch.Tensor
) -> dict[str, Any]:
    """One AE128 UINT8 + mandatory-zstd round trip, dispatched on raw bytes."""
    preloaded = ae_family_dispatch.PreloadedAeDecoders([autoencoder])
    identity = autoencoder.wire_identity()
    rows = []
    with torch.no_grad():
        for q in TRANSPORT_Q_VALUES:
            encode_wire = ZstdWireCodec()
            transport = ae_uint8_transport.encode_frame(
                c2_frame, autoencoder, ranker, q, wire_codec=encode_wire
            )
            packet = transport.packet

            receive_wire = CountingWireCodec()
            received = preloaded.receive(packet.data, wire_codec=receive_wire)

            # The authoritative provenance is the inner header of the bytes that
            # actually crossed, not the local packet dataclass.
            header = ae_uint8_transport.inspect(
                ZstdWireCodec().decompress(
                    packet.data, expected_bytes=packet.uncompressed_bytes
                )
            )
            analytical = ae_uint8_transport.analytical_size(
                header.q, QUALIFICATION_BOTTLENECK
            )
            rows.append(
                {
                    "q": header.q,
                    "header_family_id": header.family_id,
                    "header_family_name": ae_contract.family_name(header.family_id),
                    "header_latent_channels": header.bottleneck,
                    "header_routing_tag": header.routing_tag,
                    "decoder_family_id": int(identity["family_id"]),
                    "decoder_latent_channels": int(identity["bottleneck"]),
                    "decoder_routing_tag": int(identity["routing_tag"]),
                    "family_latent_and_tag_agree": (
                        header.family_id == int(identity["family_id"])
                        and header.bottleneck == int(identity["bottleneck"])
                        and header.routing_tag == int(identity["routing_tag"])
                    ),
                    "decoder_selected_from_received_header": (
                        received.family.family_id == header.family_id
                        and received.family.routing_tag == header.routing_tag
                        and received.family.transported_channels == header.bottleneck
                    ),
                    "decompressions_in_receive": receive_wire.decompressions,
                    "keep_count": int(header.header.keep_count),
                    "uncompressed_bytes": packet.uncompressed_bytes,
                    "analytical_uncompressed_bytes": analytical.total_bytes,
                    "analytical_size_agrees": packet.uncompressed_bytes
                    == analytical.total_bytes,
                    "compressed_bytes": packet.compressed_bytes,
                    "zstd_ratio": packet.compressed_bytes / packet.uncompressed_bytes,
                    "output_c2_shape": list(received.c2.shape),
                    "output_c2_device": str(received.c2.device),
                    "output_c2_finite": bool(torch.isfinite(received.c2).all()),
                    "output_c2_shape_correct": tuple(received.c2.shape)
                    == contract.SPLIT_SHAPE,
                }
            )
            del transport, packet, received
    for row in rows:
        if not (
            row["family_latent_and_tag_agree"]
            and row["decoder_selected_from_received_header"]
            and row["decompressions_in_receive"] == 1
            and row["analytical_size_agrees"]
            and row["output_c2_finite"]
            and row["output_c2_shape_correct"]
        ):
            raise guards.HybridQPayloadError(f"AE128 raw-byte round trip failed: {row}")
    return {
        "routing_tag_bound": True,
        "routing_tag": int(identity["routing_tag"]),
        "mandatory_zstd": implementation_report(),
        "per_q": rows,
    }


# ---------------------------------------------------------------------------
# Batch-1 encoder / decoder latency (CUDA events)
# ---------------------------------------------------------------------------


def _event_stats(samples: Sequence[float]) -> dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "mean_ms": float(array.mean()),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "repetitions": int(array.size),
    }


def _cuda_event_timing(function) -> list[float]:
    for _ in range(LATENCY_WARMUP):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(LATENCY_REPETITIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def measure_latency(autoencoder: torch.nn.Module, c2_frame: torch.Tensor) -> dict[str, Any]:
    """Batch-1 AE128 encoder (UE side) and decoder (edge side) GPU latency."""
    with torch.no_grad():
        latent = autoencoder.encode(c2_frame).detach()
        keep_mask = ae_composition.all_keep_mask(device=c2_frame.device)
        encoder_samples = _cuda_event_timing(lambda: autoencoder.encode(c2_frame))
        decoder_samples = _cuda_event_timing(
            lambda: autoencoder.decode(latent, keep_mask)
        )
    return {
        "method": "torch.cuda.Event elapsed_time, batch 1, single frame",
        "warmups": LATENCY_WARMUP,
        "repetitions": LATENCY_REPETITIONS,
        "mask_condition": "dense all-one keep mask (q=0); the decoder cost does "
        "not depend on q because the latent is always dense at its input",
        "encoder_ue_side": _event_stats(encoder_samples),
        "decoder_edge_side": _event_stats(decoder_samples),
    }


# ---------------------------------------------------------------------------
# One bounded attempt at one physical batch size
# ---------------------------------------------------------------------------


def run_attempt(
    model: torch.nn.Module,
    base: Any,
    dataset: Any,
    partition: teacher_cache.SplitPartition,
    manifest: Mapping[str, Any],
    ranker: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    *,
    frozen_model_state: Mapping[str, torch.Tensor],
    frozen_ranker_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    selected = select_fit_frames(partition, batch_size * UPDATE_COUNT)
    joined = join_teacher_records(selected, manifest)
    index_batches = [
        selected[start : start + batch_size]
        for start in range(0, len(selected), batch_size)
    ]

    autoencoder = build_split_feature_ae(QUALIFICATION_BOTTLENECK).to(device)
    routing_tag = ae_contract.routing_tag_from_sha256(
        hashlib.sha256(ROUTING_TAG_LABEL.encode("utf-8")).hexdigest()
    )
    autoencoder.bind_routing_tag(routing_tag)

    optimizer = torch.optim.AdamW(
        ae_parameters(autoencoder),
        lr=DIAGNOSTIC_LR,
        weight_decay=DIAGNOSTIC_WEIGHT_DECAY,
    )
    ae_loss.require_ae_only_optimizer(optimizer, autoencoder)
    ae_loss.require_frozen_companions([model, ranker])

    qualification = training.GradientQualification.for_module(
        autoencoder, window=UPDATE_COUNT
    )
    schedule = [("stage_a", float(STAGE_A_Q))] + [
        ("stage_b", float(ae_loss.stage_b_q_for_update(index)))
        for index in range(len(STAGE_B_CYCLE))
    ]
    if [q for _, q in schedule[1:]] != [float(q) for q in STAGE_B_CYCLE]:
        raise guards.HybridQConfigError("the Stage-B cycle is not the locked cycle")

    updates: list[dict[str, Any]] = []
    for update_index, (stage, q) in enumerate(schedule):
        ae_loss.require_optimization_q(q)
        frames = index_batches[update_index]
        sample_ids = [frame["sample_id"] for frame in frames]
        batch = collate_batch(base, dataset, [frame["dataset_index"] for frame in frames])
        if list(batch["sample_ids"]) != sample_ids:
            raise guards.HybridQConfigError("collated batch identity drift")

        started = time.perf_counter()
        c2 = encode_front(model, batch, device)
        composition = ae_composition.compose_batch(c2, autoencoder, ranker, q)
        reconstructed = autoencoder.decode(
            composition.masked_latent, composition.keep_mask
        )
        teacher = teacher_batch(sample_ids, joined)
        loss = ae_loss.task_aware_reconstruction_loss(c2, reconstructed, teacher)

        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        all_nonzero = qualification.observe(autoencoder, loss=loss.total)
        norms = gradient_norms(autoencoder)
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - started

        health = _finite_state(autoencoder, optimizer)
        guards.require_module_state_unchanged(model, frozen_model_state)
        guards.require_module_state_unchanged(ranker, frozen_ranker_state)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise guards.HybridQOwnershipError(
                "a frozen perception parameter received a gradient"
            )
        if any(parameter.grad is not None for parameter in ranker.parameters()):
            raise guards.HybridQOwnershipError("the stable ranker received a gradient")

        keep_counts = [
            int(value)
            for value in composition.keep_mask.reshape(len(frames), -1).sum(dim=1)
        ]
        plan = composition.plan
        if set(keep_counts) != {plan.keep_count}:
            raise guards.HybridQPayloadError(
                f"per-frame keep counts {sorted(set(keep_counts))} != {plan.keep_count}"
            )
        report = loss.report()
        updates.append(
            {
                "update": update_index + 1,
                "stage": stage,
                "q": plan.wire_q,
                "q_e4": plan.q_e4,
                "frames": len(frames),
                "sample_ids": sample_ids,
                "ranker_used": composition.selections is not None,
                "expected_keep_count": plan.keep_count,
                "per_frame_keep_counts": keep_counts,
                "per_frame_keep_counts_identical": True,
                "loss_total": report["total"],
                "loss_plain_reconstruction": report["plain_reconstruction"],
                "loss_combined_importance_reconstruction": report[
                    "combined_importance_reconstruction"
                ],
                "teacher_group_availability": report["group_availability"],
                "teacher_excluded_groups": report["excluded_groups"],
                "min_valid_groups_observed": report["min_valid_groups_observed"],
                "per_frame_error": per_frame_errors(c2, reconstructed, teacher),
                "gradient_norms": norms,
                "all_ae_tensors_nonzero_this_update": bool(all_nonzero),
                **health,
                "frozen_perception_state_unchanged": True,
                "frozen_ranker_state_unchanged": True,
                "no_gradient_on_frozen_perception": True,
                "no_gradient_on_ranker": True,
                "step_seconds": step_seconds,
                "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device)
                / (1024 ** 3),
            }
        )
        del c2, composition, reconstructed, teacher, loss, batch
        torch.cuda.empty_cache()

    qualification.require_qualified()
    gradient_report = {
        "named_ae_tensors": list(qualification.parameter_names),
        "named_ae_tensor_count": len(qualification.parameter_names),
        "window": qualification.window,
        "updates_observed": qualification.seen,
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
        "every_named_tensor_finite_and_nonzero_at_least_once": qualification.qualified(),
    }

    # One frame for the structural, transport and latency checks.
    probe_frames = index_batches[0][:1]
    probe_batch = collate_batch(
        base, dataset, [frame["dataset_index"] for frame in probe_frames]
    )
    c2_frame = encode_front(model, probe_batch, device)[0]
    tail = frozen_tail_check(model, autoencoder, ranker, c2_frame)
    transport = transport_round_trip(autoencoder, ranker, c2_frame)
    latency = measure_latency(autoencoder, c2_frame)
    del probe_batch, c2_frame

    guards.require_module_state_unchanged(model, frozen_model_state)
    guards.require_module_state_unchanged(ranker, frozen_ranker_state)
    peak_allocated = torch.cuda.max_memory_allocated(device)

    parameter_count = autoencoder.parameter_count()
    complexity = autoencoder.complexity()
    del optimizer, autoencoder
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "physical_batch": batch_size,
        "frames_used": len(selected),
        "selected_frames": [
            {
                "sample_id": frame["sample_id"],
                "dataset_index": frame["dataset_index"],
                "fit_position": frame["fit_position"],
                "cache_index": joined[frame["sample_id"]]["cache_index"],
                "shard": joined[frame["sample_id"]]["shard"],
                "offset_in_shard": joined[frame["sample_id"]]["offset_in_shard"],
                "valid_groups": list(joined[frame["sample_id"]]["valid_groups"]),
                "excluded_groups": joined[frame["sample_id"]]["excluded_groups"],
            }
            for frame in selected
        ],
        "teacher_join": {
            "join_key": "sample_id",
            "positional_ordering_alone_relied_on": False,
            "frames_joined": len(joined),
            "shards_opened": sorted({row["shard"] for row in joined.values()}),
            "one_fp32_combined_importance_map_per_frame": True,
            "valid_and_excluded_groups_present": True,
            "min_valid_groups_over_selected_frames": min(
                len(row["valid_groups"]) for row in joined.values()
            ),
            "required_min_valid_groups": ae_contract.AE_MIN_VALID_TASK_GROUPS,
            "holdout_or_reserved_frames_read": 0,
        },
        "autoencoder": {
            "family": ae_contract.family_name(
                ae_contract.family_for_bottleneck(QUALIFICATION_BOTTLENECK)
            ),
            "bottleneck": QUALIFICATION_BOTTLENECK,
            "init_seed": ae_contract.ae_init_seed(QUALIFICATION_BOTTLENECK),
            "deterministic_committed_initialization": True,
            "parameter_count": parameter_count,
            "encoder_parameters": complexity.encoder_parameters,
            "decoder_parameters": complexity.decoder_parameters,
            "routing_tag": routing_tag,
        },
        "optimizer": {
            "type": "AdamW",
            "lr": DIAGNOSTIC_LR,
            "weight_decay": DIAGNOSTIC_WEIGHT_DECAY,
            "owns_ae_parameters_only": True,
            "diagnostic_only": True,
            "is_the_final_scientific_training_configuration": False,
        },
        "updates": updates,
        "gradient_reachability": gradient_report,
        "frozen_tail": tail,
        "raw_byte_transport": transport,
        "latency": latency,
        "peak_allocated_vram_bytes": int(peak_allocated),
        "peak_allocated_vram_gib": peak_allocated / (1024 ** 3),
        "peak_reserved_vram_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
        "vram_budget_gib": VRAM_BUDGET_GIB,
        "within_vram_budget": peak_allocated <= VRAM_BUDGET_BYTES,
        "ae_and_optimizer_discarded": True,
        "checkpoint_written": False,
    }
    if peak_allocated > VRAM_BUDGET_BYTES:
        raise VramBudgetExceeded(
            f"peak allocated {peak_allocated / (1024 ** 3):.2f} GiB exceeds the "
            f"{VRAM_BUDGET_GIB} GiB budget at batch {batch_size}"
        )
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _atomic_json(path: Path, document: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    text = json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    with staging.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-9B bounded GPU qualification of SplitFusion AE128"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the Phase-9B AE qualification requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(ae_contract.AE_INIT_BASE_SEED)

    started = time.time()
    binding = bind_inputs()
    manifest = json.loads(
        (phase5_common.teacher_cache_root() / "teacher_cache_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    model, base, perception = load_frozen_perception(device)
    freeze(model)
    ranker_payload = torch.load(
        contract.repository_root() / contract.VALIDATION_RANKER_RELPATH,
        map_location="cpu",
        weights_only=False,
    )
    if int(ranker_payload["epoch"]) != contract.VALIDATION_RANKER_EPOCH:
        raise guards.HybridQConfigError("stable ranker epoch drift")
    if int(ranker_payload["parameter_count"]) != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQConfigError("stable ranker parameter-count drift")
    ranker = build_ranker()
    ranker.load_state_dict(ranker_payload["ranker"])
    ranker = ranker.to(device)
    freeze(ranker)
    del ranker_payload

    guards.require_frozen_perception([model, ranker])
    guards.require_eval_mode([model, ranker])
    model_hashes, model_aggregate = state_hashes(model)
    ranker_hashes, ranker_aggregate = state_hashes(ranker)
    frozen_model_state = guards.snapshot_module_state(model)
    frozen_ranker_state = guards.snapshot_module_state(ranker)

    dataset = build_train_dataset(base)
    partition = teacher_cache.build_split_partition(dataset)

    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    for batch_size in (PRIMARY_BATCH, FALLBACK_BATCH):
        try:
            result = run_attempt(
                model,
                base,
                dataset,
                partition,
                manifest,
                ranker,
                device,
                batch_size,
                frozen_model_state=frozen_model_state,
                frozen_ranker_state=frozen_ranker_state,
            )
            attempts.append({"physical_batch": batch_size, "outcome": "ok"})
            break
        except VramBudgetExceeded as error:
            attempts.append(
                {
                    "physical_batch": batch_size,
                    "outcome": "peak_vram_above_budget",
                    "detail": str(error),
                }
            )
        except Exception as error:  # noqa: BLE001 - retry policy is batch sizing only
            if not _is_oom(error):
                raise
            attempts.append(
                {
                    "physical_batch": batch_size,
                    "outcome": "cuda_oom",
                    "detail": str(error).splitlines()[0][:300],
                }
            )
        gc.collect()
        torch.cuda.empty_cache()
    if result is None:
        raise RuntimeError(
            "neither the primary nor the fallback batch size completed the "
            "AE128 qualification"
        )

    guards.require_module_state_unchanged(model, frozen_model_state)
    guards.require_module_state_unchanged(ranker, frozen_ranker_state)
    model_hashes_after, model_aggregate_after = state_hashes(model)
    ranker_hashes_after, ranker_aggregate_after = state_hashes(ranker)
    frozen_equal = (
        model_hashes == model_hashes_after
        and ranker_hashes == ranker_hashes_after
        and model_aggregate == model_aggregate_after
        and ranker_aggregate == ranker_aggregate_after
    )
    if not frozen_equal:
        raise guards.HybridQOwnershipError("a frozen parameter or buffer hash changed")

    report = {
        "schema": SCHEMA,
        "terminal": TERMINAL_QUALIFIED,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": time.time() - started,
        "scope": {
            "qualification_only": True,
            "scientific_ae_training_started": False,
            "checkpoint_written": False,
            "families_qualified": ["AE128"],
            "ae64_or_ae32_trained": False,
            "fit_frames_only": True,
            "train_holdout_accessed": False,
            "validation_accessed": False,
            "test_accessed": False,
            "accuracy_scored": False,
            "teacher_cache_written": False,
            "augmentation": False,
            "carla_launched": False,
            "epochs_trained": 0,
            "disposable_updates": UPDATE_COUNT,
        },
        "environment": {
            "python": platform.python_version(),
            "executable": "/usr/bin/python3",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "total_gpu_gib": torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3),
        },
        "binding": binding,
        "perception_binding": perception,
        "frozen_state": {
            "perception_parameter_and_buffer_sha256": model_hashes,
            "perception_aggregate_sha256": model_aggregate,
            "perception_tensor_count": len(model_hashes),
            "ranker_parameter_and_buffer_sha256": ranker_hashes,
            "ranker_aggregate_sha256": ranker_aggregate,
            "ranker_tensor_count": len(ranker_hashes),
            "recorded_before_qualification": True,
            "unchanged_after_qualification": frozen_equal,
            "eval_mode": True,
            "requires_grad": False,
        },
        "split": {
            "fit_frames": len(partition.fit_indices),
            "holdout_frames": len(partition.holdout_indices),
            "fit_sample_id_sha256": contract.sample_id_digest(partition.fit_sample_ids),
            "holdout_frames_used": 0,
        },
        "batch_attempts": attempts,
        "qualification": result,
    }
    output = Path(args.output)
    report_hash = _atomic_json(output / "ae128_gpu_qualification.json", report)
    (output / TERMINAL_QUALIFIED).write_text(f"{report_hash}\n", encoding="utf-8")
    print(json.dumps({"report_sha256": report_hash, "output": str(output)}))
    print(TERMINAL_QUALIFIED)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
