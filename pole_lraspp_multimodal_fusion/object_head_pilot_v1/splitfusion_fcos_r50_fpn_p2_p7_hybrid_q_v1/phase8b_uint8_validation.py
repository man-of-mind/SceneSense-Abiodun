"""Phase 8B: frozen validation of per-channel UINT8 plus mandatory zstd-1.

This runner measures the six registered q anchors once each on the same 3,345
validation frames and frozen scoring paths as Phase 6.  It trains and tunes
nothing.  Every completed q is persisted atomically, so rerunning the command
after interruption skips completed settings and resumes at the next q.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
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

from . import contract, continuous_q, guards, uint8_codec
from .gpu_qualification import load_frozen_perception, sha256_file
from .low_q_validation import PHASE6_CURVE_RELPATH, PHASE6_CURVE_SHA256
from .phase5_common import load_frozen_scorers
from .phase6_validation import (
    _collate,
    _person_only,
    evaluate_preservation_gates,
    load_validation_person_truth,
    score_validation_pass,
)
from .phase7_zstd_measurement import TORCH_CPU_THREADS, _latency_stats, _sync, host_report
from .ranker import build_ranker
from .selection import CellSelection, select_cells
from .zstd_transport import ZstdWireCodec, frame_content_size, implementation_report


EXECUTE_TOKEN = "HYBRID_Q_PHASE8B_UINT8_VALIDATION"
TERMINAL = "HYBRID_Q_UINT8_VALIDATION_COMPLETE"
SETTING_TERMINAL = "HYBRID_Q_UINT8_Q_SETTING_COMPLETE"
SMOKE_TERMINAL = "HYBRID_Q_UINT8_VALIDATION_SMOKE_COMPLETE"
SCHEMA = "splitfusion_fcos_hybrid_q_phase8b_uint8_validation_v1"
SETTING_SCHEMA = "splitfusion_fcos_hybrid_q_phase8b_uint8_setting_v1"
SMOKE_SCHEMA = "splitfusion_fcos_hybrid_q_phase8b_uint8_smoke_v1"

Q_VALUES = tuple(contract.REGISTERED_Q_VALUES)
DATALOADER_WORKERS = 8
INFERENCE_BATCH = 8
IMPLEMENTATION_BASE_COMMIT = "bdd95b1221e24ac113d1f26842f9be21648621d8"

# The narrow inference-mode compatibility fix changes no quantization or wire
# semantics; these exact source hashes bind what this validation executes.
UINT8_CODEC_SHA256 = "b6f07723860821e93c0dc2aeec456073617942b5ed8446f4555fe87b2c7ca803"
UINT8_ZSTD_TRANSPORT_SHA256 = (
    "db1f79e3786f2d5a12e367b688fd79401bc5e5c8f10523a7bbf7418f14aaa11d"
)


def _q_slug(q: float) -> str:
    return f"q{continuous_q.quantize_q(q).q_e4:04d}"


def _atomic_write(path: Path, text: str) -> str:
    """Durably replace one compact result beside its final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    with staging.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> str:
    return _atomic_write(
        path, json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    )


def _identity_digest(document: Mapping[str, Any]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(relative: str, expected: str) -> dict[str, str]:
    path = (contract.repository_root() / relative).resolve(strict=True)
    observed = sha256_file(path)
    if observed != expected:
        raise guards.HybridQConfigError(f"{relative} sha256 drift")
    return {"path": relative, "sha256": observed}


def bind_inputs() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Hash-bind every frozen model, ranker, lock, codec and FP32 reference."""
    root = contract.repository_root()
    lock_binding = _require_hash(
        contract.PERCEPTION_LOCK_RELPATH, contract.PERCEPTION_LOCK_SHA256
    )
    lock = contract.load_perception_lock()
    checkpoint_binding = _require_hash(
        str(lock["base_checkpoint"]["path"]), contract.FROZEN_CHECKPOINT_SHA256
    )
    config_binding = _require_hash(
        str(contract.locked_config_path().relative_to(root)),
        contract.LOCKED_CONFIG_SHA256,
    )
    contract.load_locked_config()
    ranker_binding = _require_hash(
        contract.VALIDATION_RANKER_RELPATH, contract.VALIDATION_RANKER_SHA256
    )
    phase6_binding = _require_hash(PHASE6_CURVE_RELPATH, PHASE6_CURVE_SHA256)
    uint8_binding = {
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "codec": _require_hash(
            str((contract.package_root() / "uint8_codec.py").relative_to(root)),
            UINT8_CODEC_SHA256,
        ),
        "mandatory_zstd_wrapper": _require_hash(
            str(
                (contract.package_root() / "uint8_zstd_transport.py").relative_to(root)
            ),
            UINT8_ZSTD_TRANSPORT_SHA256,
        ),
    }

    phase6 = json.loads((root / PHASE6_CURVE_RELPATH).read_text(encoding="utf-8"))
    if phase6.get("schema") != contract.PHASE6_SCHEMA:
        raise guards.HybridQConfigError("Phase-6 FP32 curve schema drift")
    if phase6.get("terminal") != contract.PHASE6_TERMINAL:
        raise guards.HybridQConfigError("Phase-6 FP32 curve is incomplete")
    scope = phase6["scope"]
    if int(scope["validation_frames"]) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("Phase-6 validation frame count drift")
    if bool(scope["test_accessed"]):
        raise guards.HybridQConfigError("Phase-6 reference reports test access")

    references: dict[int, dict[str, Any]] = {}
    for row in phase6["curve"]:
        plan = continuous_q.quantize_q(float(row["q"]))
        if plan.q_e4 in references:
            raise guards.HybridQConfigError("duplicate q in Phase-6 FP32 curve")
        references[plan.q_e4] = dict(row)
    expected_q = {continuous_q.quantize_q(q).q_e4 for q in Q_VALUES}
    if set(references) != expected_q:
        raise guards.HybridQConfigError("Phase-6 FP32 q ladder drift")
    for q_e4, row in references.items():
        if int(row["retained_cells"]) != continuous_q.quantize_q(q_e4 / 10000).keep_count:
            raise guards.HybridQConfigError("Phase-6 FP32 keep-count drift")
        if set(row["metrics"]) != set(contract.PROTECTED_METRICS):
            raise guards.HybridQConfigError("Phase-6 protected metric set drift")

    protected_hashes = {
        "ranker.py": "462536991f195651a1ee641f8e83444882ec370a8dffab72f13f0d770422b353",
        "selection.py": "ccc2b12919b078eac7af6131418989567d618d42f7b908b2db74df42e0342a71",
        "codec.py": "7b3833398a84fea31f65b86ec294c6675727390035b5761d372fd5a3cbba7b79",
        "guards.py": "77d8d8bfd168e74a7f0b6a7e3c8e7abc4c3549a86cda392feb394d7580a33031",
        "continuous_q.py": "8ea72faed324c29b7106bd5f6277699bd7e7ba16a66073905f9ff28c69bab23c",
        "zstd_transport.py": "57d1846b3fdc4084266e5a8adcc7abf99556ed2b187befec729003fcdb77edec",
        "locked_config.json": contract.LOCKED_CONFIG_SHA256,
    }
    protected = {
        name: _require_hash(
            str((contract.package_root() / name).relative_to(root)), expected
        )["sha256"]
        for name, expected in protected_hashes.items()
    }
    binding = {
        "perception_forward_lock": lock_binding,
        "frozen_perception_checkpoint": checkpoint_binding,
        "stable_epoch4_ranker": ranker_binding,
        "hybrid_q_locked_config": config_binding,
        "phase6_fp32_validation_curve": phase6_binding,
        "uint8_implementation": uint8_binding,
        "protected_source_sha256": protected,
    }
    return binding, references


def _run_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "schema": SCHEMA,
        "q_e4": [continuous_q.quantize_q(q).q_e4 for q in Q_VALUES],
        "validation_frames": contract.VALIDATION_FRAMES,
        "binding": dict(binding),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    return {**identity, "sha256": _identity_digest(identity)}


def _require_tree_finite(value: Any, path: str = "output") -> int:
    checked = 0
    if isinstance(value, torch.Tensor):
        guards.require_finite(value, path)
        return 1
    if isinstance(value, Mapping):
        for name, child in value.items():
            checked += _require_tree_finite(child, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            checked += _require_tree_finite(child, f"{path}[{index}]")
    return checked


def _load_runtime(device: torch.device) -> dict[str, Any]:
    """Load the one frozen model/ranker and the registered validation ordering."""
    model, base, perception = load_frozen_perception(device)
    frozen_snapshot = guards.snapshot_module_state(model)

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
    ranker = ranker.to(device).eval()
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    ranker_snapshot = guards.snapshot_module_state(ranker)
    del ranker_payload

    root = contract.repository_root()
    config = json.loads(
        (
            root
            / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json"
        ).read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    truth = load_validation_person_truth()
    frame_ids = list(truth["frame_ids"])
    inference = base.data.InferenceDataset(dataset_root, "val")
    position_by_id = {
        str(row["sample_id"]): index for index, row in enumerate(inference.rows)
    }
    if len(position_by_id) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation inference row count drift")
    positions = [position_by_id[sample_id] for sample_id in frame_ids]
    if len(set(positions)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation position mapping is not one-to-one")

    return {
        "model": model,
        "base": base,
        "perception": perception,
        "model_snapshot": frozen_snapshot,
        "ranker": ranker,
        "ranker_snapshot": ranker_snapshot,
        "device": device,
        "dataset_root": dataset_root,
        "truth": truth,
        "frame_ids": frame_ids,
        "inference": inference,
        "positions": positions,
    }


def _require_state_unchanged(runtime: Mapping[str, Any]) -> None:
    guards.require_module_state_unchanged(
        runtime["model"], runtime["model_snapshot"]
    )
    guards.require_module_state_unchanged(
        runtime["ranker"], runtime["ranker_snapshot"]
    )


def _transport_one(
    *,
    frame: torch.Tensor,
    ranker: torch.nn.Module,
    plan: continuous_q.ContinuousQ,
    wire: ZstdWireCodec,
    device: torch.device,
    prepared_frame: uint8_codec.PreparedUint8Frame | None = None,
    inspect_payload: bool = False,
) -> tuple[
    torch.Tensor,
    CellSelection | None,
    dict[str, Any],
    uint8_codec.InspectedUint8Payload | None,
]:
    """Execute and time only the locked UINT8/zstd codec stages for one C2."""
    guards.require_frozen_c2(frame, what="original validation FP32 C2")

    if prepared_frame is None:
        _sync(device)
        started = time.perf_counter_ns()
        prepared = uint8_codec.prepare(frame)
        _sync(device)
        range_ns = time.perf_counter_ns() - started
        ranges_computed_here = True
    else:
        if prepared_frame.c2 is not frame:
            raise guards.HybridQPayloadError("prepared ranges belong to a different C2")
        prepared = prepared_frame
        range_ns = 0
        ranges_computed_here = False

    if plan.is_bypass:
        selection = None
        ranker_invocations = 0
    else:
        scores = ranker.score_cells(frame)
        guards.require_frozen_scores(scores)
        selection = select_cells(scores, plan.wire_q)
        ranker_invocations = 1
        del scores

    _sync(device)
    started = time.perf_counter_ns()
    sparse = uint8_codec.encode(prepared, plan.wire_q, selection)
    _sync(device)
    quantize_frame_ns = time.perf_counter_ns() - started

    started = time.perf_counter_ns()
    compressed = wire.compress_bytes(sparse.data)
    zstd_compress_ns = time.perf_counter_ns() - started
    if frame_content_size(compressed) != sparse.total_bytes:
        raise guards.HybridQPayloadError("zstd frame content-size drift")

    started = time.perf_counter_ns()
    restored = wire.decompress_bytes(compressed)
    zstd_decompress_ns = time.perf_counter_ns() - started
    if restored != sparse.data:
        raise guards.HybridQPayloadError("zstd did not restore sparse bytes exactly")

    started = time.perf_counter_ns()
    decoded, decoded_q = uint8_codec.decode(restored)
    dequantize_scatter_ns = time.perf_counter_ns() - started
    if continuous_q.quantize_q(decoded_q).q_e4 != plan.q_e4:
        raise guards.HybridQPayloadError("decoded UINT8 q drift")
    guards.require_frozen_c2(decoded, what="decoded UINT8 validation C2")
    parsed = uint8_codec.inspect(restored) if inspect_payload else None
    if parsed is not None and (
        parsed.header.q_e4 != plan.q_e4
        or parsed.header.keep_count != plan.keep_count
    ):
        raise guards.HybridQPayloadError("UINT8 header q/cardinality drift")

    row = {
        "sparse_bytes": sparse.total_bytes,
        "compressed_bytes": len(compressed),
        "range_calculation_ns": range_ns,
        "quantization_framing_ns": quantize_frame_ns,
        "zstd_compression_ns": zstd_compress_ns,
        "zstd_decompression_ns": zstd_decompress_ns,
        "dequantization_scatter_ns": dequantize_scatter_ns,
        "ranker_invocations": ranker_invocations,
        "ranges_computed_here": ranges_computed_here,
        "selection_before_quantization": True,
        "zstd_round_trip_exact": True,
    }
    return decoded, selection, row, parsed


def _retained_error_check(
    original: torch.Tensor,
    decoded: torch.Tensor,
    parsed: uint8_codec.InspectedUint8Payload,
) -> dict[str, Any]:
    """Strict half-step check used by the one-frame qualification smoke."""
    source = original.detach().cpu().reshape(
        contract.SPLIT_CHANNELS, contract.SPLIT_CELLS
    )
    restored = decoded.reshape(contract.SPLIT_CHANNELS, contract.SPLIT_CELLS)
    indices = parsed.keep_indices
    wanted = source.index_select(1, indices)
    got = restored.index_select(1, indices)
    per_channel_error = (got - wanted).abs().amax(dim=1)
    minima = parsed.channel_ranges[:, 0]
    maxima = parsed.channel_ranges[:, 1]
    spans = maxima - minima
    constant = spans <= uint8_codec.CONSTANT_SPAN_EPSILON
    magnitude = torch.maximum(minima.abs(), maxima.abs()).clamp_min(1.0)
    tolerance = 8.0 * torch.finfo(torch.float32).eps * magnitude
    bounds = spans / (2.0 * 255.0) + tolerance
    if bool((per_channel_error[~constant] > bounds[~constant]).any()):
        raise guards.HybridQNumericalError("retained UINT8 error exceeds half-step")
    if bool(constant.any()):
        expected = minima[constant].unsqueeze(1).expand_as(got[constant])
        if not torch.equal(got[constant], expected):
            raise guards.HybridQNumericalError("constant channel did not decode to minimum")

    keep_mask = torch.zeros(contract.SPLIT_CELLS, dtype=torch.bool)
    keep_mask[indices] = True
    dropped = restored[:, ~keep_mask]
    if int(torch.count_nonzero(dropped)) != 0:
        raise guards.HybridQNumericalError("dropped cell did not decode to exact zero")
    return {
        "half_step_bound_passed": True,
        "nonconstant_channels": int((~constant).sum()),
        "constant_channels": int(constant.sum()),
        "max_retained_absolute_error": float(per_channel_error.max()),
        "max_half_step_plus_fp32_tolerance": float(bounds.max()),
        "dropped_cells_exact_zero": True,
    }


def run_smoke(
    *, output: Path, binding: Mapping[str, Any], identity: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """One validation C2, q=0 and q=0.30, with all qualification invariants."""
    destination = output / "qualification_smoke.json"
    if destination.exists():
        raise guards.HybridQConfigError(f"create-only smoke artifact exists: {destination}")
    runtime = _load_runtime(device)
    model = runtime["model"]
    ranker = runtime["ranker"]
    inference = runtime["inference"]
    position = runtime["positions"][0]
    fused, row, calibration = inference[position]
    wire = ZstdWireCodec()
    results: list[dict[str, Any]] = []
    parsed_by_q: dict[int, uint8_codec.InspectedUint8Payload] = {}
    event_order: list[str] = []

    with torch.inference_mode():
        inputs = fused.unsqueeze(0).to(device)
        c2 = model.encode_front(inputs).float()
        guards.require_frozen_batched_c2(c2, what="smoke C2")
        frame = c2[0]
        original_pointer = int(frame.data_ptr())
        prepared = uint8_codec.prepare(frame)
        event_order.append("full_original_fp32_c2_ranges_computed_once")

        for q in (0.00, 0.30):
            plan = continuous_q.quantize_q(q)
            decoded, selection, timing, parsed = _transport_one(
                frame=frame,
                ranker=ranker,
                plan=plan,
                wire=wire,
                device=device,
                prepared_frame=prepared,
                inspect_payload=True,
            )
            if parsed is None:
                raise guards.HybridQPayloadError("smoke payload was not inspected")
            if plan.is_bypass:
                if selection is not None or timing["ranker_invocations"] != 0:
                    raise guards.HybridQPayloadError("q=0 did not bypass ranker")
                event_order.append("q0:ranker_bypassed")
            else:
                if selection is None or timing["ranker_invocations"] != 1:
                    raise guards.HybridQPayloadError("q=0.30 did not select cells")
                if frame.dtype is not torch.float32 or int(frame.data_ptr()) != original_pointer:
                    raise guards.HybridQPayloadError(
                        "q=0.30 selection did not use original FP32 C2"
                    )
                event_order.append("q3000:rank_original_fp32_then_quantize")

            error = _retained_error_check(frame, decoded, parsed)
            dense = decoded.unsqueeze(0).to(device)
            outputs = model.decode_tail(dense, dense=False)
            output_tensors = _require_tree_finite(outputs, f"q={q:.2f} model output")
            calibration_gpu = {
                name: tensor.to(device) for name, tensor in calibration.items()
            }
            detections = model.postprocess(outputs, [calibration_gpu])
            detection_tensors = _require_tree_finite(
                detections, f"q={q:.2f} postprocess output"
            )
            frame_view = {"semantic_logits": outputs["semantic_logits"][0:1]}
            served, _ = apply_p025_service_policy(frame_view, detections[0])
            service_tensors = _require_tree_finite(served, f"q={q:.2f} service output")
            results.append(
                {
                    "q": plan.wire_q,
                    "q_e4": plan.q_e4,
                    "retained_cells": plan.keep_count,
                    "ranker_invocations": timing["ranker_invocations"],
                    "selection_on_original_fp32_before_quantization": True,
                    "zstd_round_trip_exact": timing["zstd_round_trip_exact"],
                    "sparse_bytes": timing["sparse_bytes"],
                    "compressed_bytes": timing["compressed_bytes"],
                    "range_pairs_finite": bool(
                        torch.isfinite(parsed.channel_ranges).all()
                    ),
                    "decoded_c2_finite": bool(torch.isfinite(decoded).all()),
                    "model_outputs_finite": True,
                    "output_tensors_checked": (
                        output_tensors + detection_tensors + service_tensors
                    ),
                    "error": error,
                }
            )
            parsed_by_q[plan.q_e4] = parsed
            del decoded, dense, outputs, detections, served

        dense_codes = parsed_by_q[0]
        sparse_codes = parsed_by_q[3000]
        positions = torch.searchsorted(
            dense_codes.keep_indices, sparse_codes.keep_indices
        )
        shared_codes_identical = torch.equal(
            dense_codes.values.index_select(0, positions), sparse_codes.values
        )
        if not shared_codes_identical:
            raise guards.HybridQNumericalError("shared q=0/q=0.30 UINT8 codes differ")

    _require_state_unchanged(runtime)
    document = {
        "schema": SMOKE_SCHEMA,
        "terminal": SMOKE_TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        "runner_sha256": identity["runner_sha256"],
        "binding": dict(binding),
        "validation_frame": str(row["sample_id"]),
        "frames": 1,
        "q_values": [0.0, 0.3],
        "event_order": event_order,
        "q0_ranker_bypassed": True,
        "q030_selection_on_original_fp32_before_quantization": True,
        "ranges_computed_once_and_reused": True,
        "shared_uint8_codes_identical": bool(shared_codes_identical),
        "zstd_round_trips_exact": True,
        "retained_half_step_bounds_passed": True,
        "dropped_cells_exact_zero": True,
        "all_ranges_decoded_c2_and_outputs_finite": True,
        "frozen_model_state_unchanged": True,
        "frozen_ranker_state_unchanged": True,
        "results": results,
        "training_or_tuning": False,
        "test_or_carla_access": False,
    }
    digest = _atomic_json(destination, document)
    print(
        json.dumps(
            {
                "qualification": SMOKE_TERMINAL,
                "frame": document["validation_frame"],
                "q_values": document["q_values"],
                "artifact": str(destination),
                "sha256": digest,
            }
        ),
        flush=True,
    )
    return document


def _compressed_stats(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
    }


def run_validation_pass(
    *, runtime: Mapping[str, Any], q: float, output: Path,
    workers: int, wire: ZstdWireCodec,
) -> dict[str, Any]:
    """One and only one complete UINT8 inference pass for one registered q."""
    plan = continuous_q.quantize_q(q)
    if not plan.is_registered:
        raise guards.HybridQConfigError("Phase 8B accepts registered q only")
    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    detections_path = output / "detections.csv"
    segmentation_manifest = output / "segmentation_manifest.csv"

    loader = DataLoader(
        Subset(runtime["inference"], list(runtime["positions"])),
        batch_size=INFERENCE_BATCH,
        shuffle=False,
        num_workers=workers,
        collate_fn=_collate,
        drop_last=False,
        pin_memory=False,
    )
    stage_samples: dict[str, list[int]] = {
        "range_calculation": [],
        "quantization_framing": [],
        "zstd_compression": [],
        "zstd_decompression": [],
        "dequantization_scatter": [],
    }
    sparse_sizes: set[int] = set()
    compressed_sizes: list[int] = []
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    detection_count = person_count = vehicle_count = 0
    ranker_invocations = 0
    exact_round_trips = 0
    output_tensors_checked = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device=runtime["device"])

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime["base"].infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for fused, rows, calibrations in loader:
                inputs = fused.to(runtime["device"], non_blocking=True)
                c2 = runtime["model"].encode_front(inputs).float()
                guards.require_frozen_batched_c2(c2, what="frozen validation C2")
                transported: list[torch.Tensor] = []
                for index in range(c2.shape[0]):
                    decoded, selection, timing, _parsed = _transport_one(
                        frame=c2[index],
                        ranker=runtime["ranker"],
                        plan=plan,
                        wire=wire,
                        device=runtime["device"],
                    )
                    if plan.is_bypass and selection is not None:
                        raise guards.HybridQPayloadError("q=0 emitted a selection")
                    if not plan.is_bypass and selection is None:
                        raise guards.HybridQPayloadError("q>0 omitted its selection")
                    ranker_invocations += int(timing["ranker_invocations"])
                    exact_round_trips += int(timing["zstd_round_trip_exact"])
                    sparse_sizes.add(int(timing["sparse_bytes"]))
                    compressed_sizes.append(int(timing["compressed_bytes"]))
                    for stage in stage_samples:
                        stage_samples[stage].append(int(timing[f"{stage}_ns"]))
                    transported.append(decoded.to(runtime["device"]))

                hybrid = torch.stack(transported)
                outputs = runtime["model"].decode_tail(hybrid, dense=False)
                output_tensors_checked += _require_tree_finite(
                    outputs, "frozen model output"
                )
                calibration_gpu = [
                    {
                        name: tensor.to(runtime["device"])
                        for name, tensor in calibration.items()
                    }
                    for calibration in calibrations
                ]
                detections = runtime["model"].postprocess(outputs, calibration_gpu)
                output_tensors_checked += _require_tree_finite(
                    detections, "frozen postprocess output"
                )
                for index, row in enumerate(rows):
                    frame_view = {
                        "semantic_logits": outputs["semantic_logits"][index:index + 1]
                    }
                    served, original_indices = apply_p025_service_policy(
                        frame_view, detections[index]
                    )
                    output_tensors_checked += _require_tree_finite(
                        served, "p025 service output"
                    )
                    records = combined_records(
                        runtime["base"], row, served, original_indices
                    )
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
                del c2, hybrid, outputs, detections, transported

    with segmentation_manifest.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height")
        )
        writer.writeheader()
        writer.writerows(segmentation_rows)
    del loader

    if observed_ids != list(runtime["frame_ids"]):
        raise guards.HybridQConfigError("validation inference order/coverage drift")
    if len(set(observed_ids)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("validation frame uniqueness drift")
    if sparse_sizes != {uint8_codec.analytical_size(q).total_bytes}:
        raise guards.HybridQPayloadError(f"pre-zstd payload-size drift: {sparse_sizes}")
    if len(compressed_sizes) != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("compressed payload count drift")
    if exact_round_trips != contract.VALIDATION_FRAMES:
        raise guards.HybridQPayloadError("not every zstd round trip was exact")
    expected_ranker_calls = 0 if plan.is_bypass else contract.VALIDATION_FRAMES
    if ranker_invocations != expected_ranker_calls:
        raise guards.HybridQPayloadError("ranker invocation count drift")
    if any(len(samples) != contract.VALIDATION_FRAMES for samples in stage_samples.values()):
        raise guards.HybridQPayloadError("codec latency sample count drift")

    return {
        "q": plan.wire_q,
        "q_e4": plan.q_e4,
        "frames": len(observed_ids),
        "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "segmentation_manifest_sha256": sha256_file(segmentation_manifest),
        "detections": detection_count,
        "person_service_outputs": person_count,
        "vehicle_service_outputs": vehicle_count,
        "retained_cells": plan.keep_count,
        "dropped_cells": plan.drop_count,
        "analytical_uint8_sparse_bytes": uint8_codec.analytical_size(q).total_bytes,
        "measured_uint8_sparse_bytes": next(iter(sparse_sizes)),
        "compressed_zstd_bytes": _compressed_stats(compressed_sizes),
        "codec_latency": {
            stage: _latency_stats(samples) for stage, samples in stage_samples.items()
        },
        "codec_latency_excludes": [
            "RGB/radar loading",
            "frozen backbone inference",
            "stable ranker and selection",
            "CPU-to-GPU reconstructed-C2 transfer",
            "frozen edge-tail inference",
            "postprocessing and scoring",
        ],
        "ranker_invocations": ranker_invocations,
        "q0_ranker_bypassed": plan.is_bypass,
        "selection_on_original_fp32_before_quantization": True,
        "full_c2_ranges_computed_before_selection": True,
        "zstd_round_trips_exact": exact_round_trips,
        "all_outputs_finite": True,
        "output_tensors_checked": output_tensors_checked,
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": (
            torch.cuda.max_memory_allocated(runtime["device"]) / 2 ** 20
        ),
        "peak_reserved_vram_mib": (
            torch.cuda.max_memory_reserved(runtime["device"]) / 2 ** 20
        ),
    }


def _setting_document(
    *, raw: Mapping[str, Any], scored: Mapping[str, Any],
    fp32: Mapping[str, Any], identity: Mapping[str, Any],
) -> dict[str, Any]:
    q = float(raw["q"])
    deltas = {
        name: {
            "fp32_same_q": float(fp32["metrics"][name]),
            "uint8_same_q": float(scored["metrics"][name]),
            "delta_uint8_minus_fp32": (
                float(scored["metrics"][name]) - float(fp32["metrics"][name])
            ),
            "absolute_delta": abs(
                float(scored["metrics"][name]) - float(fp32["metrics"][name])
            ),
        }
        for name in contract.PROTECTED_METRICS
    }
    preservation = evaluate_preservation_gates(fp32["metrics"], scored["metrics"])
    metrics_finite = all(
        math.isfinite(float(value))
        for value in (
            list(scored["metrics"].values())
            + list(scored["canonical_person_metrics"].values())
        )
    )
    if not metrics_finite:
        raise guards.HybridQNumericalError("scored UINT8 metric is non-finite")
    emergency = q in contract.EVALUATION_STRESS_Q_VALUES
    return {
        "schema": SETTING_SCHEMA,
        "terminal": SETTING_TERMINAL,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity["sha256"],
        **dict(raw),
        "metrics": dict(scored["metrics"]),
        "canonical_person_metrics": dict(scored["canonical_person_metrics"]),
        "absolute_service_gates": dict(scored["absolute_service_gates"]),
        "quantization_preservation_vs_same_q_fp32": preservation,
        "protected_metric_deltas_vs_same_q_fp32": deltas,
        "same_q_fp32_reference": {
            "phase6_curve_sha256": PHASE6_CURVE_SHA256,
            "q": float(fp32["q"]),
            "retained_cells": int(fp32["retained_cells"]),
            "metrics": dict(fp32["metrics"]),
            "canonical_person_metrics": dict(fp32["canonical_person_metrics"]),
        },
        "emergency_mode_status": {
            "is_emergency_anchor": emergency,
            "designation": "emergency-mode anchor" if emergency else "primary anchor",
            "executable": True,
            "removed_for_service_gate_miss": False,
        },
        "all_outputs_and_metrics_finite": True,
        "frozen_model_state_unchanged": True,
        "stable_ranker_state_unchanged": True,
        "inference_passes_for_this_q": 1,
        "training_or_tuning": False,
        "threshold_or_gate_change": False,
        "test_or_carla_access": False,
    }


def _load_completed(path: Path, q: float, identity: Mapping[str, Any]) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    plan = continuous_q.quantize_q(q)
    if document.get("schema") != SETTING_SCHEMA or document.get("terminal") != SETTING_TERMINAL:
        raise guards.HybridQConfigError(f"incomplete setting artifact {path}")
    if int(document.get("q_e4", -1)) != plan.q_e4:
        raise guards.HybridQConfigError(f"q mismatch in completed setting {path}")
    if document.get("run_identity_sha256") != identity["sha256"]:
        raise guards.HybridQConfigError(f"run identity mismatch in {path}")
    if int(document.get("frames", -1)) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError(f"frame count mismatch in {path}")
    if int(document.get("inference_passes_for_this_q", -1)) != 1:
        raise guards.HybridQConfigError(f"inference count mismatch in {path}")
    if not bool(document.get("all_outputs_and_metrics_finite")):
        raise guards.HybridQConfigError(f"non-finite completed setting {path}")
    return document


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    metric_columns = list(contract.PROTECTED_METRICS)
    canonical_columns = (
        "person_precision", "person_recall", "person_f1", "person_xy_mae_m"
    )
    columns = [
        "q", "q_e4", "retained_cells", "analytical_uint8_sparse_bytes",
        "measured_uint8_sparse_bytes", "zstd_bytes_median", "zstd_bytes_p95",
        "zstd_bytes_min", "zstd_bytes_max", "ratio_vs_framed_fp32_q0",
        "ratio_vs_compressed_uint8_q0", *metric_columns, *canonical_columns,
        "absolute_service_pass_count", "quantization_preservation_pass_count",
        "quantization_preservation_all_passed", "emergency_mode",
        "all_outputs_finite",
    ]
    columns += [f"delta_{name}" for name in metric_columns]
    columns += [f"absolute_delta_{name}" for name in metric_columns]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        deltas = row["protected_metric_deltas_vs_same_q_fp32"]
        writer.writerow(
            {
                "q": f"{row['q']:.2f}",
                "q_e4": row["q_e4"],
                "retained_cells": row["retained_cells"],
                "analytical_uint8_sparse_bytes": row["analytical_uint8_sparse_bytes"],
                "measured_uint8_sparse_bytes": row["measured_uint8_sparse_bytes"],
                "zstd_bytes_median": row["compressed_zstd_bytes"]["median"],
                "zstd_bytes_p95": row["compressed_zstd_bytes"]["p95"],
                "zstd_bytes_min": row["compressed_zstd_bytes"]["minimum"],
                "zstd_bytes_max": row["compressed_zstd_bytes"]["maximum"],
                "ratio_vs_framed_fp32_q0": row["ratio_vs_framed_fp32_q0"],
                "ratio_vs_compressed_uint8_q0": row["ratio_vs_compressed_uint8_q0"],
                **{name: row["metrics"][name] for name in metric_columns},
                **{
                    name: row["canonical_person_metrics"][name]
                    for name in canonical_columns
                },
                "absolute_service_pass_count": row["absolute_service_gates"]["pass_count"],
                "quantization_preservation_pass_count": row[
                    "quantization_preservation_vs_same_q_fp32"
                ]["pass_count"],
                "quantization_preservation_all_passed": row[
                    "quantization_preservation_vs_same_q_fp32"
                ]["all_passed"],
                "emergency_mode": row["emergency_mode_status"]["is_emergency_anchor"],
                "all_outputs_finite": row["all_outputs_and_metrics_finite"],
                **{
                    f"delta_{name}": deltas[name]["delta_uint8_minus_fp32"]
                    for name in metric_columns
                },
                **{
                    f"absolute_delta_{name}": deltas[name]["absolute_delta"]
                    for name in metric_columns
                },
            }
        )
    return stream.getvalue()


def _report_text(document: Mapping[str, Any]) -> str:
    rows = document["curve"]
    lines = [
        "# Phase 8B — frozen noAE UINT8 + zstd validation",
        "",
        f"Generated {document['generated_utc']} · terminal `{TERMINAL}`",
        "",
        "This is one frozen-validation measurement. No training, tuning, threshold ",
        "change, test access, CARLA access, Raspberry Pi claim, or OAI latency claim was made.",
        "",
        "## Locked pipeline",
        "",
        "```text",
        "FP32 C2 -> q selection -> per-channel UINT8 framing -> mandatory zstd-1",
        "          -> zstd decode -> dequantize/zero-scatter -> FP32 C2 -> frozen tail",
        "```",
        "",
        "## Payload and host codec cost",
        "",
        "| q | analytical pre-zstd B | measured pre-zstd B | zstd median B | p95 | min | max | vs framed FP32 q0 | vs compressed UINT8 q0 | range ms | quant/frame ms | zstd comp ms | zstd decomp ms | dequant/scatter ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        latency = row["codec_latency"]
        lines.append(
            f"| {row['q']:.2f} | {row['analytical_uint8_sparse_bytes']:,} | "
            f"{row['measured_uint8_sparse_bytes']:,} | "
            f"{row['compressed_zstd_bytes']['median']:,.0f} | "
            f"{row['compressed_zstd_bytes']['p95']:,.0f} | "
            f"{row['compressed_zstd_bytes']['minimum']:,} | "
            f"{row['compressed_zstd_bytes']['maximum']:,} | "
            f"{row['ratio_vs_framed_fp32_q0']:.6f} | "
            f"{row['ratio_vs_compressed_uint8_q0']:.6f} | "
            f"{latency['range_calculation']['median_ms']:.3f} | "
            f"{latency['quantization_framing']['median_ms']:.3f} | "
            f"{latency['zstd_compression']['median_ms']:.3f} | "
            f"{latency['zstd_decompression']['median_ms']:.3f} | "
            f"{latency['dequantization_scatter']['median_ms']:.3f} |"
        )
    lines += [
        "",
        "Ratios use median compressed bytes. Codec latency excludes backbone, ranker/selection, "
        "C2 upload, frozen tail, postprocessing, and scoring; it is current-host evidence only.",
        "",
        "## Accuracy and independent decisions",
        "",
        "| q | vehicle P/R/F1/XY | canonical-p025 person P/R/F1/XY | AVO>=0.65 person P/R/F1/XY | person 20–40 m recall | vehicle IoU | person box-mask IoU | foreground mIoU | service gates | quantization gates vs same-q FP32 | emergency status | finite |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        m = row["metrics"]
        c = row["canonical_person_metrics"]
        preservation = row["quantization_preservation_vs_same_q_fp32"]
        lines.append(
            f"| {row['q']:.2f} | {m['vehicle_precision']:.6f}/{m['vehicle_recall']:.6f}/"
            f"{m['vehicle_f1']:.6f}/{m['vehicle_xy_mae_m']:.6f} | "
            f"{c['person_precision']:.6f}/{c['person_recall']:.6f}/"
            f"{c['person_f1']:.6f}/{c['person_xy_mae_m']:.6f} | "
            f"{m['person_avo_precision']:.6f}/{m['person_avo_recall']:.6f}/"
            f"{m['person_avo_f1']:.6f}/{m['person_avo_xy_mae_m']:.6f} | "
            f"{m['person_avo_recall_20_40m']:.6f} | {m['vehicle_iou']:.6f} | "
            f"{m['person_box_mask_iou']:.6f} | {m['foreground_miou']:.6f} | "
            f"{row['absolute_service_gates']['pass_count']}/9 | "
            f"{preservation['pass_count']}/12 | "
            f"{row['emergency_mode_status']['designation']} | yes |"
        )
    lines += [
        "",
        "Quantization preservation, absolute service-gate attainment, and emergency-anchor "
        "designation are separate. No executable q was removed for missing service gates.",
        "",
        "## Absolute protected-metric deltas from same-q FP32",
        "",
        "Values below are `abs(UINT8 - FP32)` at the same q.",
        "",
        "| q | " + " | ".join(contract.PROTECTED_METRICS) + " |",
        "| ---: | " + " | ".join("---:" for _ in contract.PROTECTED_METRICS) + " |",
    ]
    for row in rows:
        deltas = row["protected_metric_deltas_vs_same_q_fp32"]
        lines.append(
            f"| {row['q']:.2f} | "
            + " | ".join(
                f"{deltas[name]['absolute_delta']:.9f}"
                for name in contract.PROTECTED_METRICS
            )
            + " |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- validation frames per q: {contract.VALIDATION_FRAMES:,}",
        f"- q settings completed exactly once: {len(rows)}/6",
        "- every UINT8 sparse payload was zstd-wrapped and restored byte-for-byte",
        "- all ranges, reconstructed C2 tensors, model outputs, and reported metrics were finite",
        "- q=0 bypassed the ranker but remained UINT8-quantized",
        "- model and stable-ranker state remained unchanged",
        "- predictions were removed after scoring; only compact evidence is retained",
        "",
    ]
    return "\n".join(lines)


def finalize(
    *, output: Path, rows: list[dict[str, Any]], binding: Mapping[str, Any],
    identity: Mapping[str, Any], runtime: Mapping[str, Any],
    default_cpu_threads: int, started: float,
) -> dict[str, Any]:
    q0_median = float(rows[0]["compressed_zstd_bytes"]["median"])
    for row in rows:
        median = float(row["compressed_zstd_bytes"]["median"])
        row["ratio_vs_framed_fp32_q0"] = median / contract.FRAMED_Q0_PAYLOAD_BYTES
        row["ratio_vs_compressed_uint8_q0"] = median / q0_median
    _require_state_unchanged(runtime)
    document = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "isolate per-channel UINT8 quantization from sparsification by comparing "
            "each measured row with Phase-6 FP32 at the same q"
        ),
        "scope": {
            "validation_frames_per_q": contract.VALIDATION_FRAMES,
            "validation_episodes": list(contract.VALIDATION_EPISODES),
            "q_values": list(Q_VALUES),
            "completed_settings": len(rows),
            "inference_passes_per_q": 1,
            "training_or_tuning": False,
            "threshold_or_gate_change": False,
            "test_accessed": False,
            "carla_launched": False,
            "prediction_directories_retained": False,
        },
        "run_identity": dict(identity),
        "binding": dict(binding),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(runtime["device"]),
            "torch_cpu_threads": TORCH_CPU_THREADS,
            "torch_default_cpu_threads": default_cpu_threads,
            "inference_precision": "fp32 inference_mode, no autocast",
        },
        "perception_binding": runtime["perception"],
        "transport": {
            "pipeline": [
                "FP32 C2", "q sparsification", "per-channel UINT8 framing",
                "mandatory zstd-1", "zstd decompression", "UINT8 dequantization",
                "zero scatter to FP32 C2", "unchanged frozen tail",
            ],
            "zstd": implementation_report(),
            "zstd_mandatory": True,
            "snap_continuous_q_called": False,
            "ranges": "per frame/channel from complete original FP32 C2",
        },
        "same_q_fp32_reference": {
            "path": PHASE6_CURVE_RELPATH,
            "sha256": PHASE6_CURVE_SHA256,
            "preservation_gates": [
                {"metric": name, "direction": direction, "bound": bound}
                for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
            ],
        },
        "decision_separation": {
            "quantization_preservation_relative_to_same_q_fp32": True,
            "absolute_service_gate_status": True,
            "emergency_mode_status_for_q090_q098": True,
            "q_removed_for_service_gate_miss": False,
        },
        "curve": rows,
        "settings": {
            _q_slug(row["q"]): {
                "path": f"settings/{_q_slug(row['q'])}.json",
                "sha256": sha256_file(output / "settings" / f"{_q_slug(row['q'])}.json"),
            }
            for row in rows
        },
        "qualification_smoke": {
            "path": "qualification_smoke.json",
            "sha256": sha256_file(output / "qualification_smoke.json"),
            "terminal": SMOKE_TERMINAL,
        },
        "integrity": {
            "zstd_round_trips_exact": sum(
                int(row["zstd_round_trips_exact"]) for row in rows
            ),
            "required_zstd_round_trips": len(rows) * contract.VALIDATION_FRAMES,
            "all_outputs_and_metrics_finite": all(
                row["all_outputs_and_metrics_finite"] for row in rows
            ),
            "q0_ranker_bypassed": rows[0]["ranker_invocations"] == 0,
            "model_state_unchanged": True,
            "ranker_state_unchanged": True,
        },
        "wall_seconds_this_invocation": time.time() - started,
    }
    if document["integrity"]["zstd_round_trips_exact"] != document["integrity"]["required_zstd_round_trips"]:
        raise guards.HybridQPayloadError("final zstd round-trip count drift")
    if not document["integrity"]["all_outputs_and_metrics_finite"]:
        raise guards.HybridQNumericalError("final result contains a non-finite row")

    _atomic_json(output / "phase8b_uint8_validation.json", document)
    _atomic_write(output / "phase8b_uint8_validation.csv", _csv_text(rows))
    _atomic_write(output / "PHASE8B_UINT8_VALIDATION_REPORT.md", _report_text(document))
    _atomic_write(
        output / TERMINAL,
        f"{TERMINAL} {document['generated_utc']}\n",
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-8B frozen per-channel UINT8 + zstd validation"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 8B requires the Phase-7 CUDA runtime")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)
    default_cpu_threads = torch.get_num_threads()
    torch.set_num_threads(TORCH_CPU_THREADS)
    started = time.time()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    binding, references = bind_inputs()
    identity = _run_identity(binding)

    if args.smoke:
        run_smoke(output=output, binding=binding, identity=identity, device=device)
        return 0

    smoke_path = output / "qualification_smoke.json"
    if not smoke_path.is_file():
        raise guards.HybridQConfigError("qualification smoke must complete first")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke.get("terminal") != SMOKE_TERMINAL
        or smoke.get("run_identity_sha256") != identity["sha256"]
        or not smoke.get("all_ranges_decoded_c2_and_outputs_finite")
    ):
        raise guards.HybridQConfigError("qualification smoke binding/status drift")

    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity": identity,
        "q_values": list(Q_VALUES),
        "resume_rule": "skip exact complete setting; rerun only an unfinished q",
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("run_identity") != identity:
            raise guards.HybridQConfigError("existing run manifest identity drift")
    else:
        _atomic_json(manifest_path, manifest)

    settings_dir = output / "settings"
    work_dir = output / "working_predictions"
    settings_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)
    runtime = _load_runtime(device)
    scorers = load_frozen_scorers()
    gt, gt_states = scorers.load_gt(runtime["dataset_root"], contract.PRIMARY_CONTRACT)
    validation_gt = {
        sample_id: gt.get(sample_id, []) for sample_id in runtime["frame_ids"]
    }
    person_gt = _person_only(validation_gt)
    ignore_cache: dict[str, Any] = {}
    wire = ZstdWireCodec()
    completed_rows: list[dict[str, Any]] = []

    for q in Q_VALUES:
        slug = _q_slug(q)
        setting_path = settings_dir / f"{slug}.json"
        if setting_path.exists():
            completed_rows.append(_load_completed(setting_path, q, identity))
            leftover = work_dir / slug
            if leftover.exists():
                shutil.rmtree(leftover)
            print(json.dumps({"reused_completed_q": q, "setting": str(setting_path)}), flush=True)
            continue

        prediction_root = work_dir / slug
        if prediction_root.exists():
            shutil.rmtree(prediction_root)
        raw = run_validation_pass(
            runtime=runtime,
            q=q,
            output=prediction_root,
            workers=int(args.workers),
            wire=wire,
        )
        _require_state_unchanged(runtime)
        scored = score_validation_pass(
            result=raw,
            scorers=scorers,
            truth=runtime["truth"],
            experiment=runtime["dataset_root"],
            frame_ids=runtime["frame_ids"],
            gt=validation_gt,
            person_gt=person_gt,
            ignore_cache=ignore_cache,
        )
        fp32 = references[continuous_q.quantize_q(q).q_e4]
        setting = _setting_document(
            raw=raw, scored=scored, fp32=fp32, identity=identity
        )
        shutil.rmtree(prediction_root)
        setting["prediction_artifacts_removed_after_scoring"] = True
        digest = _atomic_json(setting_path, setting)
        completed_rows.append(setting)
        print(
            json.dumps(
                {
                    "completed_q": q,
                    "frames": setting["frames"],
                    "zstd_bytes_median": setting["compressed_zstd_bytes"]["median"],
                    "service_gates": setting["absolute_service_gates"]["pass_count"],
                    "quantization_gates_vs_same_q_fp32": setting[
                        "quantization_preservation_vs_same_q_fp32"
                    ]["pass_count"],
                    "setting": str(setting_path),
                    "sha256": digest,
                }
            ),
            flush=True,
        )

    if [row["q_e4"] for row in completed_rows] != [
        continuous_q.quantize_q(q).q_e4 for q in Q_VALUES
    ]:
        raise guards.HybridQConfigError("completed q order drift")
    _require_state_unchanged(runtime)
    document = finalize(
        output=output,
        rows=completed_rows,
        binding=binding,
        identity=identity,
        runtime=runtime,
        default_cpu_threads=default_cpu_threads,
        started=started,
    )
    if work_dir.exists() and not any(work_dir.iterdir()):
        work_dir.rmdir()
    print(
        json.dumps(
            {
                "terminal": TERMINAL,
                "output": str(output),
                "settings": len(document["curve"]),
                "all_finite": document["integrity"]["all_outputs_and_metrics_finite"],
                "zstd_round_trips": document["integrity"]["zstd_round_trips_exact"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
