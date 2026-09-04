"""One-shot SFD1-v2 dynamic-localization qualification for SplitFusion."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)

from . import phase13b_qualification as phase13b
from . import phase13c_measurement as phase13c
from .envelope import (
    CONTEXT_FIXED_BYTES,
    CONTEXT_PROTOCOL_VERSION,
    HEADER_BYTES,
    PROTOCOL_VERSION,
    pack_envelope,
    unpack_envelope,
)
from .frame_context import (
    STATIC_CAMERA_MODEL_SHA256,
    STATIC_CAMERA_MOUNT_SHA256,
    STATIC_INTRINSIC_TENSOR_SHA256,
    camera_world_matrix,
)
from .registry import ActionProfile, SplitActionRegistry


EXECUTE_TOKEN = "SPLITFUSION_FRAME_CONTEXT_BINDING_QUALIFICATION"
SCHEMA = "scenesense.splitfusion_frame_context_binding_qualification.v1"
TERMINAL = "SPLITFUSION_FRAME_CONTEXT_BINDING_QUALIFIED"
STARTING_HEAD = "cc6bc92f6f8224fb7ab5a37c31b01a0383f7cae0"
OUTPUT_RELPATH = (
    "experiments/splitfusion_live_dispatch_v1/"
    "20260904_phase13c_frame_context_qualification"
)
ACTION_IDS = (0, 20, 46, 71)
TRANSACTIONS = 32
MATRIX_ERROR_BOUND = 1e-4
WORLD_ERROR_BOUND_M = 1e-3
EXPECTED_ACTIONS = {
    0: ("noAE", "UINT8", 0),
    20: ("AE128", "UINT8", 5000),
    46: ("AE64", "UINT6", 9000),
    71: ("AE32", "UINT4", 9800),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(*arguments: str) -> str:
    return phase13c._git_output(*arguments)


def _verify_git_state() -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    parent = _git_output("rev-parse", "HEAD^")
    _require(
        parent == STARTING_HEAD,
        f"frame-context implementation parent is {parent}, expected {STARTING_HEAD}",
    )
    _require(
        _git_output("merge-base", "--is-ancestor", STARTING_HEAD, head) == "",
        "required frame-context starting commit is not an ancestor",
    )
    lines = _git_output(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    paths: list[str] = []
    for line in lines:
        _require(len(line) >= 4, f"unparseable git status line: {line!r}")
        path = line[3:]
        _require(" -> " not in path, "renamed dirty paths are not authorized")
        paths.append(path)
    _require(
        frozenset(paths) == phase13b.EXPECTED_DIRTY_PATHS
        and len(paths) == len(phase13b.EXPECTED_DIRTY_PATHS),
        f"unexpected dirty paths: observed={sorted(paths)} "
        f"expected={sorted(phase13b.EXPECTED_DIRTY_PATHS)}",
    )
    return {
        "implementation_commit": head,
        "implementation_parent": parent,
        "expected_user_owned_dirty_paths": sorted(paths),
    }


def _select_actions(registry: SplitActionRegistry) -> tuple[ActionProfile, ...]:
    profiles = tuple(registry.resolve(action_id) for action_id in ACTION_IDS)
    _require(
        all(
            (profile.family, profile.quantizer, profile.q_e4)
            == EXPECTED_ACTIONS[profile.action_id]
            for profile in profiles
        ),
        "representative action identity drift",
    )
    _require(
        all(
            profile.execution_mode == "SPLIT"
            and profile.zstd_level == 1
            and profile.wire.layout == "CURRENT_CELL_MAJOR"
            for profile in profiles
        ),
        "representative action transport contract drift",
    )
    return profiles


def _select_eight(sample: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    selected = []
    for episode_id in sample["partition"]["fit_episodes"]:
        candidates = [
            row for row in sample["selected_rows"] if row["episode_id"] == episode_id
        ]
        _require(bool(candidates), f"frozen sample lacks fit episode {episode_id}")
        selected.append(candidates[(len(candidates) - 1) // 2])
    sample_ids = [row["sample_id"] for row in selected]
    _require(
        len(selected) == 8
        and len(set(sample_ids)) == 8
        and all(
            row["registered_split"] == "fit"
            and row["source_row"]["split"] == "train"
            for row in selected
        ),
        "eight-frame fit-only selection failed",
    )
    return tuple(selected)


def _output_candidate() -> Path:
    experiments = (_root() / "experiments").resolve(strict=True)
    candidate = (_root() / OUTPUT_RELPATH).resolve(strict=False)
    try:
        candidate.relative_to(experiments)
    except ValueError as exc:
        raise RuntimeError("frame-context output escapes experiments root") from exc
    _require(not candidate.exists(), f"create-only output exists: {candidate}")
    return candidate


def _create_output(candidate: Path) -> Path:
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=False, exist_ok=False)
    resolved = candidate.resolve(strict=True)
    experiments = (_root() / "experiments").resolve(strict=True)
    try:
        resolved.relative_to(experiments)
    except ValueError as exc:
        raise RuntimeError("created frame-context output escapes experiments root") from exc
    return resolved


def _counter_delta(after: Counter[str], before: Counter[str]) -> dict[str, int]:
    return phase13c._counter_delta(after, before)


def _operation_delta(after: Any, before: Any) -> dict[str, int]:
    return phase13c._operation_delta(after, before)


def _max_world_error(left: torch.Tensor, right: torch.Tensor) -> float:
    _require(left.shape == right.shape, "world-coordinate shape drift")
    if left.numel() == 0:
        return 0.0
    distances = torch.linalg.vector_norm(left.double() - right.double(), dim=-1)
    _require(bool(torch.isfinite(distances).all()), "world-coordinate error is non-finite")
    return float(distances.max().item())


def _tail_parity(live: Any, direct: Any) -> dict[str, Any]:
    _require(set(live.perception) == set(direct.perception), "p025 field set drift")
    for name in live.perception:
        if name == "world_xyz":
            continue
        _require(
            torch.equal(live.perception[name], direct.perception[name]),
            f"non-world p025 field differs: {name}",
        )
    _require(
        torch.equal(live.perception["local_xyz"], direct.perception["local_xyz"]),
        "camera-relative local_xyz is not bit-identical",
    )
    _require(
        torch.equal(live.original_indices, direct.original_indices),
        "p025 candidate ordering differs",
    )
    _require(
        torch.equal(live.semantic_logits, direct.semantic_logits),
        "segmentation logits differ",
    )
    _require(
        torch.equal(live.semantic_labels, direct.semantic_labels),
        "segmentation labels differ",
    )
    world_error = _max_world_error(
        live.perception["world_xyz"], direct.perception["world_xyz"]
    )
    _require(
        world_error <= WORLD_ERROR_BOUND_M,
        f"p025 world-coordinate error {world_error} exceeds {WORLD_ERROR_BOUND_M}",
    )
    live_records = live.records or ()
    direct_records = direct.records or ()
    _require(len(live_records) == len(direct_records), "service-record count drift")
    record_world_error = 0.0
    ignored = {
        "stream_id",
        "capture_timestamp_ns",
        "sample_id",
        "world_x",
        "world_y",
        "world_z",
    }
    for live_record, direct_record in zip(live_records, direct_records, strict=True):
        for name, value in direct_record.items():
            if name not in ignored:
                _require(
                    live_record.get(name) == value,
                    f"non-world service-record field differs: {name}",
                )
        live_world = torch.tensor(
            [live_record[name] for name in ("world_x", "world_y", "world_z")],
            dtype=torch.float64,
        )
        direct_world = torch.tensor(
            [direct_record[name] for name in ("world_x", "world_y", "world_z")],
            dtype=torch.float64,
        )
        difference = float(torch.linalg.vector_norm(live_world - direct_world).item())
        _require(math.isfinite(difference), "service-record world error is non-finite")
        record_world_error = max(record_world_error, difference)
    _require(
        record_world_error <= WORLD_ERROR_BOUND_M,
        f"service-record world error {record_world_error} exceeds {WORLD_ERROR_BOUND_M}",
    )
    return {
        "detection_count": int(live.perception["scores"].numel()),
        "local_xyz_bit_identical": True,
        "classes_scores_order_and_nonworld_fields_bit_identical": True,
        "segmentation_logits_bit_identical": True,
        "segmentation_labels_bit_identical": True,
        "maximum_world_xyz_error_m": world_error,
        "maximum_service_record_world_error_m": record_world_error,
        "service_record_count": len(live_records),
    }


def _qualify_transaction(
    runtime: Mapping[str, Any],
    *,
    profile: ActionProfile,
    selected_row: Mapping[str, Any],
    message_id: int,
) -> dict[str, Any]:
    ledger: phase13b.CallLedger = runtime["ledger"]
    input_7ch, inference_row = phase13c._load_input(runtime, selected_row)
    _require(
        inference_row["sample_id"] == selected_row["sample_id"],
        "loaded inference identity drift",
    )
    context = phase13c._context_for(selected_row, profile, message_id)
    source = selected_row["source_row"]
    expected_pose = phase13c._pose(source, "anchor")
    _require(context.ego_world == expected_pose, "transmitted ego pose differs from fit metadata")
    _require(
        context.frame_id == int(source["frame_id"])
        and context.capture_timestamp_ns
        == int(round(float(source["timestamp"]) * 1_000_000_000)),
        "frame-context identifiers differ from fit metadata",
    )
    direct_calibration = {
        "intrinsic": runtime["base"].data.model_intrinsic(source).to(runtime["device"]),
        "extrinsic": runtime["base"].data.camera_extrinsic(source).to(runtime["device"]),
    }
    _require(
        phase13c._tensor_digest(direct_calibration["intrinsic"])
        == STATIC_INTRINSIC_TENSOR_SHA256,
        "direct-reference intrinsic hash drift",
    )
    direct_tail = phase13b.FrozenP025TailAdapter(
        model=runtime["model"],
        base=runtime["base"],
        row=source,
        calibration=direct_calibration,
        ledger=ledger,
    )
    live_before = ledger.snapshot("live")
    direct_before = ledger.snapshot("direct")
    ue_before = runtime["ue"].counters
    edge_before = runtime["edge"].counters
    runtime["timers"].reset()
    with phase13b._hot_path_guard():
        with ledger.section("live"):
            prepared = runtime["ue"].prepare(
                profile.action_id,
                input_7ch,
                sequence_id=context.sequence_id,
                capture_timestamp_ns=context.capture_timestamp_ns,
                frame_context=context,
            )
            runtime["front"].release_c2()
            delivery = runtime["loopback"].roundtrip(
                prepared.wire_bytes, message_id=message_id
            )
            live_result = runtime["edge"].process(
                delivery.payload, transmitted_action_id=profile.action_id
            )
            inspected, decoded = runtime["edge_codec"].take()
            live_tail = runtime["tail"].take_snapshot()
        with ledger.section("direct"):
            direct_perception = direct_tail(decoded.c2, live_result.metadata)
            direct_tail.serialize(direct_perception)
            direct_snapshot = direct_tail.take_snapshot()
    del input_7ch

    outer = unpack_envelope(prepared.wire_bytes)
    _require(
        outer.protocol_version == CONTEXT_PROTOCOL_VERSION
        and outer.frame_context == context
        and live_result.metadata.frame_context == context,
        "SFD1 v2 frame-context propagation drift",
    )
    _require(
        outer.action_id == profile.action_id
        and inspected.identity.family == profile.family
        and inspected.identity.family_id == profile.family_id
        and inspected.identity.quantizer == profile.quantizer
        and inspected.identity.bit_width == profile.bit_width
        and inspected.identity.q_e4 == profile.q_e4
        and inspected.identity.keep_count == profile.keep_count
        and inspected.identity.routing_tag == profile.routing_tag
        and inspected.identity.transported_channels == profile.transported_channels
        and inspected.identity.latent_width == profile.latent_width
        and inspected.identity.wire_codec_id == profile.wire.codec_id
        and inspected.identity.wire_version == profile.wire.version,
        "action/SFD1/catalog routing drift",
    )
    _require(
        outer.inner_payload == prepared.wire_bytes[outer.header_bytes:],
        "SFD1 v2 changed the scientific inner bytes",
    )
    historical_v1 = pack_envelope(
        outer.inner_payload,
        action_id=profile.action_id,
        sequence_id=context.sequence_id,
        capture_timestamp_ns=context.capture_timestamp_ns,
    )
    v1 = unpack_envelope(historical_v1)
    _require(
        v1.protocol_version == PROTOCOL_VERSION
        and v1.header_bytes == HEADER_BYTES
        and v1.inner_payload == outer.inner_payload,
        "historical SFD1 v1 compatibility or inner-byte invariance failed",
    )
    _require(
        delivery.payload == prepared.wire_bytes
        and delivery.sfd1_application_bytes == len(prepared.wire_bytes)
        and delivery.duplicate_datagrams == 0,
        "localhost UDP reassembly drift",
    )
    _require(
        decoded.finite
        and decoded.device == runtime["device"]
        and decoded.c2.device == runtime["device"]
        and decoded.c2.dtype is torch.float32
        and tuple(decoded.c2.shape) == (256, 112, 192)
        and bool(torch.isfinite(decoded.c2).all()),
        "reconstructed C2 contract drift",
    )
    static = runtime["camera_registry"].resolve(
        context.camera_model_sha256, context.camera_mount_sha256
    )
    reconstructed = torch.tensor(
        camera_world_matrix(context, static), dtype=torch.float64, device=runtime["device"]
    )
    matrix_error = float(
        torch.max(torch.abs(reconstructed - direct_calibration["extrinsic"].double())).item()
    )
    _require(
        math.isfinite(matrix_error) and matrix_error <= MATRIX_ERROR_BOUND,
        f"camera matrix error {matrix_error} exceeds {MATRIX_ERROR_BOUND}",
    )
    _require(
        torch.equal(reconstructed, live_tail.camera_world),
        "tail did not consume the reconstructed camera matrix",
    )
    parity = _tail_parity(live_tail, direct_snapshot)
    for record in live_tail.records or ():
        _require(
            record["stream_id"] == context.stream_id
            and int(record["frame_id"]) == context.frame_id
            and int(record["capture_timestamp_ns"]) == context.capture_timestamp_ns,
            "service identifiers did not come from frame context",
        )

    live_calls = _counter_delta(ledger.snapshot("live"), live_before)
    direct_calls = _counter_delta(ledger.snapshot("direct"), direct_before)
    phase13c._validate_calls(profile, live_calls)
    _require(
        direct_calls == {"service_record_serialization": 1, "tail": 1},
        f"direct diagnostic call drift: {direct_calls}",
    )
    ue_delta = _operation_delta(runtime["ue"].counters, ue_before)
    edge_delta = _operation_delta(runtime["edge"].counters, edge_before)
    _require(
        ue_delta.get("frames_attempted") == 1
        and ue_delta.get("frames_completed") == 1
        and edge_delta.get("frames_attempted") == 1
        and edge_delta.get("frames_completed") == 1
        and edge_delta.get("tail_dispatches") == 1,
        "UE/edge transaction counter drift",
    )
    _require(
        runtime["ue"].counters.hot_path_model_load_operations == 0
        and runtime["ue"].counters.hot_path_model_construction_operations == 0
        and runtime["edge"].counters.hot_path_model_load_operations == 0
        and runtime["edge"].counters.hot_path_model_construction_operations == 0,
        "hot-path model load or construction occurred",
    )
    result = {
        "action_id": profile.action_id,
        "profile_id": profile.profile_id,
        "family": profile.family,
        "quantizer": profile.quantizer,
        "q_e4": profile.q_e4,
        "keep_count": profile.keep_count,
        "sample_id": selected_row["sample_id"],
        "episode_id": selected_row["episode_id"],
        "frame_id": context.frame_id,
        "stream_id": context.stream_id,
        "sequence_id": context.sequence_id,
        "capture_timestamp_ns": context.capture_timestamp_ns,
        "static_camera_model_sha256": context.camera_model_sha256,
        "static_camera_mount_sha256": context.camera_mount_sha256,
        "camera_matrix_max_abs_error": matrix_error,
        "scientific_inner_bytes": outer.inner_payload_length,
        "sfd1_v1_overhead_bytes": HEADER_BYTES,
        "sfd1_v2_overhead_bytes": outer.header_bytes,
        "frame_context_incremental_overhead_bytes": outer.header_bytes - HEADER_BYTES,
        "complete_sfd1_v2_bytes": len(prepared.wire_bytes),
        "udp_datagrams": delivery.datagrams,
        "udp_chunk_header_bytes": delivery.chunk_header_bytes,
        "udp_application_bytes": delivery.udp_application_bytes,
        "estimated_ip_udp_bytes": delivery.estimated_ip_udp_bytes,
        "estimated_on_wire_bytes": delivery.estimated_on_wire_bytes,
        "sfd1_sha256": phase13c._digest_bytes(prepared.wire_bytes),
        "inner_sha256": phase13c._digest_bytes(outer.inner_payload),
        "reassembled_exact": True,
        "reconstructed_c2_device": str(decoded.c2.device),
        "reconstructed_c2_dtype": str(decoded.c2.dtype).replace("torch.", ""),
        "reconstructed_c2_finite": True,
        "live_calls": live_calls,
        "direct_diagnostic_calls": direct_calls,
        "ue_counter_delta": ue_delta,
        "edge_counter_delta": edge_delta,
        **parity,
    }
    del (
        prepared,
        delivery,
        live_result,
        inspected,
        decoded,
        live_tail,
        direct_snapshot,
        direct_perception,
        direct_calibration,
        direct_tail,
        reconstructed,
    )
    return result


def _aggregate_calls(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row[field])
    return dict(sorted(counts.items()))


def _report(document: Mapping[str, Any]) -> str:
    lines = [
        "# SplitFusion SFD1 v2 frame-context qualification",
        "",
        f"Status: `{document['terminal']}`.",
        "",
        "This bounded run qualified dynamic world localization; it is not a latency or FPS study.",
        "",
        "| action | family | quantizer | q | frames | max matrix error | max world error (m) |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for result in document["action_results"]:
        lines.append(
            f"| {result['action_id']} | {result['family']} | {result['quantizer']} | "
            f"{result['q_e4'] / 10000:.2f} | {result['transactions']} | "
            f"{result['maximum_camera_matrix_error']:.9g} | "
            f"{result['maximum_world_xyz_error_m']:.9g} |"
        )
    lines.extend(
        [
            "",
            "All 32 messages used SFD1 v2, exact localhost UDP reassembly, edge-resident static calibration, and per-frame transmitted ego pose. The direct frozen reference used each recorded camera matrix only for qualification comparison.",
            "",
            "No prediction tensors, inner payloads, SFD1 frames, datagrams, or serialized service payloads are retained.",
        ]
    )
    return "\n".join(lines) + "\n"


def _close(runtime: Mapping[str, Any] | None) -> None:
    phase13c._close_runtime(runtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded SFD1-v2 per-frame-context CUDA/localhost qualification"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.parse_args()
    runtime: dict[str, Any] | None = None
    output: Path | None = None
    outcomes: list[dict[str, Any]] = []
    operation = "preflight_interpreter"
    current_action: int | None = None
    current_sample: str | None = None
    started = time.perf_counter()
    try:
        _require(
            Path(sys.executable).resolve(strict=True)
            == Path("/usr/bin/python3").resolve(strict=True),
            f"qualification requires /usr/bin/python3, observed {sys.executable}",
        )
        operation = "preflight_git"
        git = _verify_git_state()
        operation = "preflight_frame_context_binding"
        frame_context_binding = phase13c._verify_frame_context_binding()
        operation = "preflight_phase13a"
        phase13a_binding = phase13b._verify_phase13a_terminal()
        operation = "preflight_phase13b"
        phase13b_binding = phase13c._verify_phase13b_artifacts()
        operation = "preflight_registry"
        registry = SplitActionRegistry.from_runtime_binding(
            verify_runtime_artifacts=False
        )
        profiles = _select_actions(registry)
        operation = "preflight_historical_ae"
        historical = phase11b.phase11b_preflight()
        operation = "preflight_fit_sample"
        sample, sample_context = phase13c._construct_sample()
        selected = _select_eight(sample)
        _require(
            sample["selected_sample_id_sha256"] == phase13c.AUDITED_SAMPLE_ID_SHA256,
            "frozen 300-frame sample digest drift",
        )
        operation = "preflight_output_absence"
        output_candidate = _output_candidate()
        operation = "preflight_cuda_and_workload"
        gpu = phase13b._gpu_preflight()
        device = torch.device("cuda:0")
        operation = "preload_frozen_runtime"
        runtime = phase13c._load_models(
            device=device,
            registry=registry,
            historical=historical,
            sample_context=sample_context,
            expected_messages=TRANSACTIONS,
        )
        runtime["sample"] = sample
        operation = "create_output"
        output = _create_output(output_candidate)
        torch.cuda.reset_peak_memory_stats(device)
        operation = "frame_context_32_transaction_qualification"
        message_id = 1
        for profile in profiles:
            runtime["edge"].reset_context_session()
            for selected_row in selected:
                current_action = profile.action_id
                current_sample = selected_row["sample_id"]
                outcomes.append(
                    _qualify_transaction(
                        runtime,
                        profile=profile,
                        selected_row=selected_row,
                        message_id=message_id,
                    )
                )
                message_id += 1
        _require(len(outcomes) == TRANSACTIONS, "transaction count is not 32")
        operation = "frozen_state_check"
        state_after, frozen_equal = phase13c._state_after(runtime)
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        peak_vram = int(torch.cuda.max_memory_allocated(device))
        maximum_matrix_error = max(
            row["camera_matrix_max_abs_error"] for row in outcomes
        )
        maximum_world_error = max(
            row["maximum_world_xyz_error_m"] for row in outcomes
        )
        maximum_record_world_error = max(
            row["maximum_service_record_world_error_m"] for row in outcomes
        )
        _require(maximum_matrix_error <= MATRIX_ERROR_BOUND, "matrix gate failed")
        _require(
            max(maximum_world_error, maximum_record_world_error)
            <= WORLD_ERROR_BOUND_M,
            "world-coordinate gate failed",
        )
        action_results = []
        for profile in profiles:
            rows = [row for row in outcomes if row["action_id"] == profile.action_id]
            _require(len(rows) == 8, "per-action transaction count drift")
            action_results.append(
                {
                    "action_id": profile.action_id,
                    "profile_id": profile.profile_id,
                    "family": profile.family,
                    "quantizer": profile.quantizer,
                    "q_e4": profile.q_e4,
                    "transactions": len(rows),
                    "maximum_camera_matrix_error": max(
                        row["camera_matrix_max_abs_error"] for row in rows
                    ),
                    "maximum_world_xyz_error_m": max(
                        row["maximum_world_xyz_error_m"] for row in rows
                    ),
                    "maximum_service_record_world_error_m": max(
                        row["maximum_service_record_world_error_m"] for row in rows
                    ),
                    "scientific_inner_bytes": sum(
                        row["scientific_inner_bytes"] for row in rows
                    ),
                    "complete_sfd1_v2_bytes": sum(
                        row["complete_sfd1_v2_bytes"] for row in rows
                    ),
                    "estimated_on_wire_bytes": sum(
                        row["estimated_on_wire_bytes"] for row in rows
                    ),
                    "udp_datagrams": sum(row["udp_datagrams"] for row in rows),
                }
            )
        live_calls = _aggregate_calls(outcomes, "live_calls")
        _require(
            live_calls.get("live_ue_zstd_compressions") == TRANSACTIONS
            and live_calls.get("live_edge_zstd_decompressions") == TRANSACTIONS
            and live_calls.get("tail") == TRANSACTIONS
            and live_calls.get("ranker") == 24,
            f"aggregate live call counts drift: {live_calls}",
        )
        udp_totals = {
            "messages": len(outcomes),
            "datagrams": sum(row["udp_datagrams"] for row in outcomes),
            "scientific_inner_bytes": sum(
                row["scientific_inner_bytes"] for row in outcomes
            ),
            "sfd1_v1_overhead_bytes": TRANSACTIONS * HEADER_BYTES,
            "sfd1_v2_overhead_bytes": sum(
                row["sfd1_v2_overhead_bytes"] for row in outcomes
            ),
            "frame_context_incremental_overhead_bytes": sum(
                row["frame_context_incremental_overhead_bytes"] for row in outcomes
            ),
            "complete_sfd1_v2_bytes": sum(
                row["complete_sfd1_v2_bytes"] for row in outcomes
            ),
            "udp_chunk_header_bytes": sum(
                row["udp_chunk_header_bytes"] for row in outcomes
            ),
            "udp_application_bytes": sum(
                row["udp_application_bytes"] for row in outcomes
            ),
            "estimated_ip_udp_bytes": sum(
                row["estimated_ip_udp_bytes"] for row in outcomes
            ),
            "estimated_on_wire_bytes": sum(
                row["estimated_on_wire_bytes"] for row in outcomes
            ),
            "duplicate_datagrams": 0,
            "reassembly_exact": True,
        }
        selected_summary = [
            {
                "episode_id": row["episode_id"],
                "sample_id": row["sample_id"],
                "frame_id": int(row["frame_id"]),
                "dataset_index": int(row["dataset_index"]),
                "episode_selection_ordinal": int(row["episode_selection_ordinal"]),
                "source_row_sha256": row["source_row_sha256"],
            }
            for row in selected
        ]
        document = {
            "schema": SCHEMA,
            "terminal": TERMINAL,
            "status": "32_OF_32_DYNAMIC_LOCALIZATION_TRANSACTIONS_QUALIFIED",
            "implementation": git,
            "frame_context_binding": frame_context_binding,
            "phase13a_binding": phase13a_binding,
            "phase13b_binding": phase13b_binding,
            "registry_startup_audit": asdict(registry.startup_audit),
            "environment": gpu,
            "audit_bindings": {
                "frozen_300_sample_id_sha256": phase13c.AUDITED_SAMPLE_ID_SHA256,
                "static_camera_model_sha256": STATIC_CAMERA_MODEL_SHA256,
                "static_camera_mount_sha256": STATIC_CAMERA_MOUNT_SHA256,
                "static_intrinsic_tensor_sha256": STATIC_INTRINSIC_TENSOR_SHA256,
                "ordered_ego_pose_digest": phase13c.AUDITED_EGO_POSE_DIGEST,
                "ordered_camera_pose_digest": phase13c.AUDITED_CAMERA_POSE_DIGEST,
                "legacy_full_calibration_identities": sample["calibration"][
                    "legacy_full_calibration_identity_count"
                ],
            },
            "selected_eight": {
                "rule": "lower middle frozen-sample row within each canonical fit episode",
                "fit_only": True,
                "holdout_validation_test_rows_read": 0,
                "rows": selected_summary,
            },
            "wire_schema": {
                "magic": "SFD1",
                "historical_v1_protocol_version": PROTOCOL_VERSION,
                "context_protocol_version": CONTEXT_PROTOCOL_VERSION,
                "base_header_format": "<4sHHIQQQ",
                "base_header_bytes": HEADER_BYTES,
                "context_fixed_format": "<HHIQQQ6d32s32s",
                "context_fixed_bytes": CONTEXT_FIXED_BYTES,
                "context_fields": [
                    "context_version:uint16",
                    "context_bytes:uint16",
                    "stream_id_bytes:uint32",
                    "canonical_frame_id:uint64",
                    "sequence_id_copy:uint64",
                    "capture_timestamp_ns_copy:uint64",
                    "ego_world_x_y_z_pitch_yaw_roll:6*float64",
                    "camera_model_sha256:32 raw bytes",
                    "camera_mount_sha256:32 raw bytes",
                    "stream_id:strict UTF-8 bytes",
                ],
                "self_delimiting_and_strictly_length_validated": True,
                "inner_codec_bytes_unchanged": True,
            },
            "gates": {
                "transactions_completed": len(outcomes),
                "expected_transactions": TRANSACTIONS,
                "maximum_camera_matrix_error": maximum_matrix_error,
                "camera_matrix_error_bound": MATRIX_ERROR_BOUND,
                "maximum_world_xyz_error_m": maximum_world_error,
                "maximum_service_record_world_error_m": maximum_record_world_error,
                "world_coordinate_error_bound_m": WORLD_ERROR_BOUND_M,
                "context_identifiers_from_source_and_service_records": True,
                "local_xyz_bit_identical": True,
                "classes_scores_order_nonworld_fields_bit_identical": True,
                "segmentation_logits_and_labels_bit_identical": True,
                "all_reconstructed_c2_finite_fp32_cuda0": True,
                "all_udp_reassembly_exact": True,
                "stream_session_reset_between_profile_replays": True,
                "frozen_state_equal": frozen_equal,
            },
            "frozen_state": {
                "before": runtime["state_before"],
                "after": state_after,
            },
            "action_results": action_results,
            "live_call_totals": live_calls,
            "direct_diagnostic_call_totals": _aggregate_calls(
                outcomes, "direct_diagnostic_calls"
            ),
            "udp_accounting": udp_totals,
            "transactions": outcomes,
            "resource_use": {
                "wall_seconds_including_preflight_and_model_load": wall_seconds,
                "peak_cuda_memory_allocated_bytes_after_preload_reset": peak_vram,
                "latency_or_fps_claim": False,
            },
            "scope": {
                "fit_rgb_frames_read": TRANSACTIONS,
                "fit_radar_frames_read": TRANSACTIONS,
                "holdout_frames_read": 0,
                "validation_frames_read": 0,
                "test_frames_read": 0,
                "training_tuning_scoring": False,
                "carla_oai_rfsim_launched": False,
                "phase13c_36x300_launched": False,
                "retained_predictions_payloads_or_datagrams": False,
            },
        }
        qualification_path = output / "qualification.json"
        qualification_hash = phase13c._atomic_create_json(
            qualification_path, document
        )
        report_hash = phase13c._atomic_create_text(
            output / "REPORT.md", _report(document)
        )
        terminal_hash = phase13c._atomic_create_text(
            output / TERMINAL, f"{TERMINAL} {qualification_hash}\n"
        )
        print(
            json.dumps(
                {
                    "terminal": TERMINAL,
                    "qualification_sha256": qualification_hash,
                    "report_sha256": report_hash,
                    "terminal_sha256": terminal_hash,
                    "transactions": len(outcomes),
                    "maximum_camera_matrix_error": maximum_matrix_error,
                    "maximum_world_xyz_error_m": maximum_world_error,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "terminal": "SPLITFUSION_FRAME_CONTEXT_BINDING_FAILED",
                    "operation": operation,
                    "action_id": current_action,
                    "sample_id": current_sample,
                    "completed_transactions": len(outcomes),
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_created": str(output) if output is not None else None,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        _close(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
