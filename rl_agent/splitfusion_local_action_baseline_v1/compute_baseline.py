#!/usr/bin/env python3
"""Measure full-local FCOS compute and compact object-result size.

The module deliberately excludes sensor preparation and radio transport.  It
starts when one normalized seven-channel tensor is available on the host and
ends when the current, optimized p025 object result has been serialized and
compressed for a one-shot LOCAL upload.  Dense segmentation never enters that
upload and earns no edge-map credit in the LOCAL action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from rl_agent.splitfusion_edge_optimization_v1.detached_edge_preload_v3 import (
    preload_detached_optimized_edge_v3,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    build_frame_context_v1,
)
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for
from rl_agent.splitfusion_timing_diagnostic_v1 import diagnostic_common as common
from rl_agent.splitfusion_timing_diagnostic_v1 import runner as timing


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SCHEMA = "scenesense.splitfusion.local_object_result.v1"
LOCAL_PROFILE_ID = "local_fcos_r50_fpn_p2_p7_p025_v1"
FIT_FRAMES = 300
WARMUP_REPETITIONS = 10
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_SEED = 2026091301
ZLIB_LEVEL = 1


class LocalBaselineError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalBaselineError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_rank(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = max(1, math.ceil(float(probability) * len(ordered))) - 1
    return ordered[min(index, len(ordered) - 1)]


def distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    require(bool(finite), "cannot summarize an empty distribution")
    return {
        "count": len(finite),
        "minimum": finite[0],
        "p50": nearest_rank(finite, 0.50),
        "p90": nearest_rank(finite, 0.90),
        "p95": nearest_rank(finite, 0.95),
        "p99": nearest_rank(finite, 0.99),
        "maximum": finite[-1],
        "mean": statistics.fmean(finite),
    }


def bootstrap_p95_ci(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    require(len(finite) >= 20, "p95 confidence interval requires at least 20 values")
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = [
        float(nearest_rank([finite[rng.randrange(len(finite))] for _ in finite], 0.95))
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]
    return {
        "method": "deterministic_nonparametric_bootstrap",
        "seed": BOOTSTRAP_SEED,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "lower_95": float(nearest_rank(estimates, 0.025)),
        "upper_95": float(nearest_rank(estimates, 0.975)),
    }


def build_local_payload(
    *,
    run_id: str,
    frame_id: int,
    capture_timestamp_ns: int,
    local_result_available_ns: int,
    records: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    stream_id: str | None = None,
) -> bytes:
    """Build the exact compact LOCAL object result sent toward the map."""

    require(frame_id >= 0, "LOCAL frame ID must be non-negative")
    require(capture_timestamp_ns > 0, "LOCAL capture timestamp must be positive")
    require(
        local_result_available_ns >= capture_timestamp_ns,
        "LOCAL completion predates capture",
    )
    document = {
        "schema": LOCAL_SCHEMA,
        "profile_id": LOCAL_PROFILE_ID,
        "run_id": str(run_id),
        "stream_id": str(stream_id) if stream_id is not None else f"{run_id}/local",
        "frame_id": int(frame_id),
        "capture_timestamp_ns": int(capture_timestamp_ns),
        "local_result_available_ns": int(local_result_available_ns),
        "checkpoint_sha256": str(checkpoint_sha256),
        "objects": [dict(record) for record in records],
        "segmentation": {
            "edge_map_credit": False,
            "transported": False,
            "reason": "LOCAL_COMPACT_OBJECT_RESULT_ONLY",
        },
    }
    raw = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return zlib.compress(raw, level=ZLIB_LEVEL)


def decode_local_payload(payload: bytes) -> dict[str, Any]:
    document = json.loads(zlib.decompress(payload).decode("utf-8"))
    require(document.get("schema") == LOCAL_SCHEMA, "LOCAL payload schema drift")
    require(document.get("profile_id") == LOCAL_PROFILE_ID, "LOCAL profile drift")
    require(isinstance(document.get("objects"), list), "LOCAL objects are absent")
    segmentation = document.get("segmentation") or {}
    require(
        segmentation.get("transported") is False
        and segmentation.get("edge_map_credit") is False,
        "LOCAL payload unexpectedly transports segmentation",
    )
    return document


def quality_binding() -> dict[str, Any]:
    """Bind full-local q0 quality and the closest split anchor on identical validation."""

    fp32_path = ROOT / (
        "experiments/splitfusion_fcos_hybrid_q_v1/"
        "20260902_182401_phase6_validation_curve/validation_curve.json"
    )
    split_path = ROOT / (
        "experiments/splitfusion_288_offline_rl_dataset_v1/"
        "20260909_offline_consolidation_v1/action_72_summary.csv"
    )
    document = json.loads(fp32_path.read_text(encoding="utf-8"))
    matches = [row for row in document["curve"] if float(row["q"]) == 0.0]
    require(len(matches) == 1, "full-local FP32 q0 validation row is not unique")
    full_local = dict(matches[0]["metrics"])

    import csv

    with split_path.open(newline="", encoding="utf-8") as handle:
        split_matches = [
            row for row in csv.DictReader(handle) if int(row["action_id"]) == 0
        ]
    require(len(split_matches) == 1, "split action-0 quality row is not unique")
    split = split_matches[0]
    fields = {
        "vehicle_f1": "val_vehicle_f1",
        "vehicle_iou": "val_vehicle_iou",
        "vehicle_xy_mae_m": "val_vehicle_xy_mae_m",
        "person_avo_f1": "val_person_avo_f1",
        "person_avo_xy_mae_m": "val_person_avo_xy_mae_m",
        "person_box_mask_iou": "val_person_box_mask_iou",
        "foreground_miou": "val_foreground_miou",
    }
    split_quality = {name: float(split[column]) for name, column in fields.items()}
    return {
        "identity": "same frozen validation inputs and p025 service",
        "full_local_fp32_q0": full_local,
        "split_action_0_uint8_q0": split_quality,
        "delta_local_minus_split0": {
            name: float(full_local[name]) - split_quality[name] for name in fields
        },
        "full_local_source": {
            "path": str(fp32_path.relative_to(ROOT)),
            "sha256": sha256_file(fp32_path),
        },
        "split_source": {
            "path": str(split_path.relative_to(ROOT)),
            "sha256": sha256_file(split_path),
        },
    }


@dataclass
class ComputeMeasurement:
    rows: list[dict[str, Any]]
    payloads: list[bytes]
    summary: dict[str, Any]
    sample: dict[str, Any]
    model_binding: dict[str, Any]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_compute_outputs(output: Path, measurement: ComputeMeasurement) -> None:
    require(not output.exists(), f"create-only output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for source in measurement.rows:
        row = dict(source)
        for field in ("tail_stage_wall_ns", "tail_stage_cuda_ms"):
            row[field] = json.dumps(
                row[field], sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        rows.append(row)
    fields = list(rows[0])
    csv_path = output / "local_compute_per_frame.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **measurement.summary,
        "sample_manifest_sha256": measurement.sample["sample_manifest_sha256"],
        "selected_sample_id_sha256": measurement.sample["selected_sample_id_sha256"],
        "model_binding": measurement.model_binding,
        "raw_inputs_retained": False,
        "raw_predictions_retained": False,
        "dense_segmentation_retained": False,
    }
    atomic_json(output / "local_compute_summary.json", summary)
    hashes = {
        name: sha256_file(output / name)
        for name in ("local_compute_per_frame.csv", "local_compute_summary.json")
    }
    atomic_json(
        output / "artifact_manifest.json",
        {
            "schema": "scenesense.splitfusion.local_compute_artifacts.v1",
            "status": "COMPLETE",
            "sha256": hashes,
        },
    )
    atomic_text(
        output / "SPLITFUSION_LOCAL_COMPUTE_BASELINE_COMPLETE",
        json.dumps(
            {
                "status": "COMPLETE",
                "summary_sha256": hashes["local_compute_summary.json"],
            },
            sort_keys=True,
        )
        + "\n",
    )


def _context(selected: Mapping[str, Any], frame_id: int, capture_ns: int, run_id: str) -> Any:
    row = selected["source_row"]
    return build_frame_context_v1(
        stream_id=f"{run_id}/local",
        frame_id=int(frame_id),
        sequence_id=int(frame_id),
        capture_timestamp_ns=int(capture_ns),
        ego_world_x=float(row["anchor_x"]),
        ego_world_y=float(row["anchor_y"]),
        ego_world_z=float(row["anchor_z"]),
        ego_world_pitch=float(row["anchor_pitch"]),
        ego_world_yaw=float(row["anchor_yaw"]),
        ego_world_roll=float(row["anchor_roll"]),
    )


def _run_one(
    *,
    edge: Any,
    input_cpu: torch.Tensor,
    selected: Mapping[str, Any],
    frame_id: int,
    run_id: str,
) -> tuple[dict[str, Any], bytes]:
    capture_ns = time.time_ns()
    context = _context(selected, frame_id, capture_ns, run_id)
    reference_profile = edge.registry.resolve(0)
    metadata = metadata_for(
        reference_profile,
        sequence_id=frame_id,
        capture_timestamp_ns=capture_ns,
        frame_context=context,
    )
    device = edge.device
    total_started = time.perf_counter_ns()
    h2d_started = time.perf_counter_ns()
    input_gpu = input_cpu.unsqueeze(0).to(device=device, non_blocking=False)
    torch.cuda.synchronize(device)
    h2d_ms = (time.perf_counter_ns() - h2d_started) / 1e6

    front_start = torch.cuda.Event(enable_timing=True)
    front_end = torch.cuda.Event(enable_timing=True)
    front_wall_started = time.perf_counter_ns()
    front_start.record()
    with torch.inference_mode():
        c2 = edge.model.encode_front(input_gpu)
    front_end.record()
    front_end.synchronize()
    front_wall_ms = (time.perf_counter_ns() - front_wall_started) / 1e6
    front_cuda_ms = float(front_start.elapsed_time(front_end))

    edge.tail.begin_frame()
    tail_started = time.perf_counter_ns()
    with torch.inference_mode():
        work = edge.tail.compute_product(c2, metadata)
    tail_wall_ms = (time.perf_counter_ns() - tail_started) / 1e6
    tail_trace = edge.tail.resolve_frame()
    serialization_started = time.perf_counter_ns()
    serialized = edge.tail.serialize_product(work)
    serialization_ms = (time.perf_counter_ns() - serialization_started) / 1e6
    available_ns = capture_ns + (time.perf_counter_ns() - total_started)
    payload = build_local_payload(
        run_id=run_id,
        frame_id=frame_id,
        capture_timestamp_ns=capture_ns,
        local_result_available_ns=available_ns,
        records=serialized.records,
        checkpoint_sha256=edge.perception_binding["checkpoint_sha256"],
    )
    total_ms = (time.perf_counter_ns() - total_started) / 1e6
    decoded = decode_local_payload(payload)
    require(int(decoded["frame_id"]) == frame_id, "LOCAL payload frame drift")
    require(
        len(decoded["objects"]) == int(serialized.record_count),
        "LOCAL record count drift",
    )
    row = {
        "frame_id": frame_id,
        "sample_id": str(selected["sample_id"]),
        "dataset_index": int(selected["dataset_index"]),
        "h2d_ms": h2d_ms,
        "front_cuda_ms": front_cuda_ms,
        "front_wall_ms": front_wall_ms,
        "optimized_tail_compute_wall_ms": tail_wall_ms,
        "compact_serialization_wall_ms": serialization_ms,
        "local_result_available_ms": total_ms,
        "object_count": int(serialized.record_count),
        "serialized_records_bytes": len(serialized.serialized_records),
        "local_payload_bytes": len(payload),
        "local_payload_sha256": sha256_bytes(payload),
        "tail_stage_wall_ns": tail_trace["wall_ns"],
        "tail_stage_cuda_ms": tail_trace["cuda_ms"],
        "segmentation_transported": False,
    }
    del input_gpu, c2, work, serialized
    return row, payload


def measure(run_id: str) -> ComputeMeasurement:
    require(not torch.cuda.is_initialized(), "fit sampling must precede CUDA")
    sample, sample_context = timing.construct_registered_sample()
    require(int(sample["selected_frame_count"]) == FIT_FRAMES, "fit sample size drift")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.get_device_name(0) == common.DEVICE_NAME, "GPU identity drift")
    device = torch.device("cuda:0")
    edge = preload_detached_optimized_edge_v3(device)
    inference = edge.base.data.InferenceDataset(sample_context["dataset_root"], "train")

    warm = sample["warmup"]
    warm_input, warm_row, _ = inference[int(warm["dataset_index"])]
    require(warm_row["sample_id"] == warm["sample_id"], "warm-up sample drift")
    for repeat in range(WARMUP_REPETITIONS):
        _run_one(
            edge=edge,
            input_cpu=warm_input,
            selected=warm,
            frame_id=1_000_000 + repeat,
            run_id=f"{run_id}/warmup",
        )
    del warm_input
    torch.cuda.synchronize(device)

    rows: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    for ordinal, selected in enumerate(sample["selected_rows"]):
        fused, source_row, _ = inference[int(selected["dataset_index"])]
        require(
            source_row["sample_id"] == selected["sample_id"],
            "measured sample identity drift",
        )
        row, payload = _run_one(
            edge=edge,
            input_cpu=fused,
            selected=selected,
            frame_id=ordinal,
            run_id=run_id,
        )
        rows.append(row)
        payloads.append(payload)
        if (ordinal + 1) % 50 == 0:
            print(f"full-local compute: {ordinal + 1}/{FIT_FRAMES}", flush=True)

    require(len(rows) == len(payloads) == FIT_FRAMES, "LOCAL compute row count drift")
    timing_fields = (
        "h2d_ms",
        "front_cuda_ms",
        "front_wall_ms",
        "optimized_tail_compute_wall_ms",
        "compact_serialization_wall_ms",
        "local_result_available_ms",
    )
    summary = {
        "schema": "scenesense.splitfusion.local_compute_summary.v1",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "target_device_status": "DESKTOP_RTX5090_PROXY_NOT_VEHICLE_HARDWARE",
        "frames": len(rows),
        "warmup_frames": WARMUP_REPETITIONS,
        "timing_ms": {
            field: {
                **distribution([float(row[field]) for row in rows]),
                "p95_confidence_interval": bootstrap_p95_ci(
                    [float(row[field]) for row in rows]
                ),
            }
            for field in timing_fields
        },
        "sustainable_sequential_fps_from_median": (
            1000.0
            / float(nearest_rank([row["local_result_available_ms"] for row in rows], 0.5))
        ),
        "payload_bytes": distribution([float(row["local_payload_bytes"]) for row in rows]),
        "serialized_record_bytes": distribution(
            [float(row["serialized_records_bytes"]) for row in rows]
        ),
        "object_count": distribution([float(row["object_count"]) for row in rows]),
        "segmentation_transported": False,
        "payload_unique_sha256": len({row["local_payload_sha256"] for row in rows}),
        "quality": quality_binding(),
    }
    return ComputeMeasurement(
        rows=rows,
        payloads=payloads,
        summary=summary,
        sample=sample,
        model_binding=dict(edge.perception_binding),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(
        args.execute == "SPLITFUSION_LOCAL_FCOS_COMPUTE_BASELINE",
        "execution token mismatch",
    )
    output = args.output.resolve()
    measurement = measure(output.name)
    write_compute_outputs(output, measurement)
    print(json.dumps(measurement.summary, indent=2, sort_keys=True))
    print("SPLITFUSION_LOCAL_COMPUTE_BASELINE_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
