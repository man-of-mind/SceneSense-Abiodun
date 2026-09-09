#!/usr/bin/env python3
"""SplitFusion timing diagnostic v1: one four-action decomposition run.

Objective: measure, without conflation, application-level one-way feature
uplink handling through the live OAI 5G path and every edge-side stage from
datagram receipt to compact result serialization, including pure
``model.decode_tail`` CUDA time.

This is not a Route-B campaign. It never launches CARLA, never trains, tunes
or evaluates, and never touches holdout or test frames. Its sample is exactly
the registered Phase-13C fit-only 300-frame sample, reconstructed
deterministically and bound by that sample's published digest.

Per action the lifecycle is: cold host -> the qualified 100 MHz/273 PRB/4D5U
launcher brings up CN5G/gNB/UE -> verified clean ``noise_power_dB=-50`` ->
FAVORABLE_STABLE profile actuation on the registered 100 ms schedule ->
measurement -> restore and read back -50 -> complete teardown -> independent
cold verification. Four actions therefore mean four fresh radio lifecycles.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from phase2_map_sharing.transport import CHUNK_HEADER, ChunkReassembler, chunk_payload
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
)
from rl_agent.splitfusion_live_dispatch_v1 import phase13c_measurement as p13c
from rl_agent.splitfusion_live_dispatch_v1.envelope import pack_envelope, unpack_envelope
from rl_agent.splitfusion_live_dispatch_v1.frame_context import (
    STATIC_CAMERA_MODEL_SHA256,
    STATIC_CAMERA_MOUNT_SHA256,
    build_frame_context_v1,
)
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import (
    ack_timeout_s,
    deadline_at_s,
    service_deadline_s,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.timing import EDGE_STAGES, StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    ProductionSplitCodec,
    require_inner_agreement,
)
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import metadata_for

from . import diagnostic_common as common
from .diagnostic_common import DiagnosticError, ROOT, repo_path, require
from .edge_preload import preload_instrumented_edge, preload_ue
from .instrumented_tail import (
    SERIALIZE_STAGE,
    TAIL_STAGES,
    TOTAL_STAGE,
    assert_parent_equivalence,
)


EXECUTE_TOKEN = "SPLITFUSION_TIMING_DIAGNOSTIC_V1_EXECUTE"
EDGE_MODULE = "rl_agent.splitfusion_timing_diagnostic_v1.edge_service"
TARGET_SNR_MODULE = "rl_agent.splitfusion_live_dispatch_v1.live_pilot_target_snr_runtime"
EDGE_EVIDENCE_LEAF = "diagnostic_edge_state"
WARMUP_ITERATIONS = 12
FIRST_MEASURED_SEQUENCE_ID = 1000
TRANSMIT_ANCHOR_SLACK_NS = 5_000_000_000
# Fixed, action-independent gap between arming the target-SNR actuator and the
# first transmitted slot, so every action's frame i lands on the same frozen
# profile target index regardless of how long its payloads took to materialize.
ACTUATION_LEAD_NS = 200_000_000
DRAIN_SECONDS = 4.0
MICROBENCH_WARMUP = 8


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"git {' '.join(arguments)} failed: {completed.stderr.strip()}",
    )
    return completed.stdout


def _git(*arguments: str) -> str:
    return _git_output(*arguments).strip()


def _porcelain_paths() -> list[str]:
    """Parse ``git status --porcelain`` without disturbing its status columns.

    The two status characters are followed by one space, so the path begins at
    index 3 of each *unmodified* line. Stripping the whole command output
    would remove the first line's leading status space and shift that one path
    by a character, which is exactly the defect this replaces.
    """

    paths: list[str] = []
    pending_rename = False
    for line in _git_output("status", "--porcelain", "-z").split("\0"):
        if not line:
            continue
        if pending_rename:
            # ``-z`` emits a rename/copy origin as its own field.
            pending_rename = False
            continue
        require(
            len(line) > 3 and line[2] == " ",
            f"unparsable git porcelain entry: {line!r}",
        )
        pending_rename = line[0] in {"R", "C"} or line[1] in {"R", "C"}
        paths.append(line[3:])
    return paths


def verify_git_state() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    dirty = sorted(path for path in _porcelain_paths() if path)
    unexpected = sorted(set(dirty) - set(common.EXPECTED_DIRTY_PATHS))
    require(
        not unexpected,
        "unexpected dirty worktree paths would be at risk: " + ", ".join(unexpected),
    )
    return {
        "head": head,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": dirty,
        "preserved_user_owned_dirty_paths": list(common.EXPECTED_DIRTY_PATHS),
    }


def verify_bound_inputs() -> dict[str, Any]:
    bindings: dict[str, str] = {}
    for relative, expected in sorted(common.BOUND_INPUTS.items()):
        path = repo_path(relative)
        require(path.is_file(), f"bound input is missing: {relative}")
        observed = common.sha256_file(path)
        require(
            observed == expected,
            f"bound input hash drift: {relative} {observed} != {expected}",
        )
        bindings[relative] = observed
    diagnostic_sources = {
        relative: common.sha256_file(repo_path(relative))
        for relative in sorted(
            str(path.relative_to(ROOT))
            for path in (ROOT / "rl_agent/splitfusion_timing_diagnostic_v1").rglob("*.py")
            if "__pycache__" not in path.parts
        )
    }
    return {
        "bound_inputs": bindings,
        "diagnostic_implementation_sources": diagnostic_sources,
    }


def verify_cuda() -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "expected exactly one visible CUDA device")
    name = torch.cuda.get_device_name(0)
    require(name == common.DEVICE_NAME, f"CUDA device identity drift: {name}")
    capability = torch.cuda.get_device_capability(0)
    return {
        "device_count": 1,
        "device_name": name,
        "capability": f"{capability[0]}.{capability[1]}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def verify_container_runtime() -> dict[str, Any]:
    completed = subprocess.run(
        (
            "sudo", "-n", "docker", "info", "--format",
            "{{range $name, $runtime := .Runtimes}}{{$name}}\n{{end}}",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(completed.returncode == 0, f"cannot query docker runtimes: {completed.stderr.strip()}")
    runtimes = sorted({line.strip() for line in completed.stdout.splitlines() if line.strip()})
    require("nvidia" in runtimes, f"docker does not report the nvidia runtime: {runtimes}")
    image = subprocess.run(
        ("sudo", "-n", "docker", "image", "inspect", "-f", "{{.Id}}", "oai-perception-rx:latest"),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    return {
        "docker_runtimes": runtimes,
        "nvidia_container_runtime_present": True,
        "edge_image_id": image.stdout.strip() if image.returncode == 0 else "",
    }


def _process_rows() -> list[dict[str, Any]]:
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a

    return list(phase14a.process_table())


def _listening_ports(protocol: str) -> set[int]:
    arguments = ("ss", "-H", "-ltnp") if protocol == "tcp" else ("ss", "-H", "-lunp")
    completed = subprocess.run(
        arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False,
    )
    require(completed.returncode == 0, f"cannot audit {protocol} listeners")
    ports: set[int] = set()
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 5:
            candidate = fields[3].rsplit(":", 1)[-1]
            if candidate.isdigit():
                ports.add(int(candidate))
    return ports


def verify_cold_host(campaign: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Independent proof that no radio, edge, CARLA or foreign cell survives."""

    runtime = campaign["runtime"]
    softmodems = [
        {"pid": int(row["pid"]), "name": str(row.get("command_name", ""))}
        for row in _process_rows()
        if str(row.get("command_name", "")) in {"nr-softmodem", "nr-uesoftmodem"}
    ]
    require(not softmodems, f"[{label}] a softmodem is still active: {softmodems}")
    owned_sources = {
        str(repo_path(str(runtime[name])))
        for name in (
            "required_route_b_split_cell_adapter",
            "map_install_runtime",
            "target_snr_runtime",
            "live_dispatch_bridge",
        )
    }
    owned_sources.add(str(repo_path("rl_agent/splitfusion_timing_diagnostic_v1/edge_service.py")))
    active = [
        {"pid": int(row["pid"]), "executable": Path(str(row.get("executable", ""))).name}
        for row in _process_rows()
        if Path(str(row.get("executable", ""))).name.startswith("CarlaUnreal")
        or any(source in str(row.get("command", "")) for source in owned_sources)
        or EDGE_MODULE in str(row.get("command", ""))
    ]
    require(not active, f"[{label}] stale application processes exist: {active}")
    edge = subprocess.run(
        ("sudo", "-n", "docker", "container", "inspect", "oai-perception-rx"),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    require(edge.returncode != 0, f"[{label}] a stale edge container exists")
    tunnel = subprocess.run(
        ("ip", "link", "show", "oaitun_ue1"),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    require(tunnel.returncode != 0, f"[{label}] a stale oaitun_ue1 interface exists")
    requested_tcp = {2000, 8010, 35001, 9090, 2021, 2031}
    requested_udp = {
        int(runtime["edge_receive_port"]), int(runtime["edge_source_port"]),
        int(runtime["camera_result_port"]), int(runtime["map_ingest_port"]), 39401,
    }
    occupied_tcp = sorted(requested_tcp & _listening_ports("tcp"))
    occupied_udp = sorted(requested_udp & _listening_ports("udp"))
    require(not occupied_tcp, f"[{label}] conflicting TCP listeners: {occupied_tcp}")
    require(not occupied_udp, f"[{label}] conflicting UDP listeners: {occupied_udp}")
    stale_tmp = sorted(
        path.name
        for prefix in (
            "ue_288_cell_runtime_*", "ue_288_seg_eval_*", "splitfusion_pilot_*",
            "splitfusion_live_edge_*", "splitfusion_timing_diagnostic_*",
        )
        for path in Path("/tmp").glob(prefix)
    )
    require(not stale_tmp, f"[{label}] stale temporary runtime paths: {stale_tmp}")
    stale_shm = sorted(
        path.name
        for path in Path("/dev/shm").iterdir()
        if any(token in path.name.casefold() for token in ("carla", "splitfusion", "oai"))
    )
    require(not stale_shm, f"[{label}] stale shared-memory objects: {stale_shm}")
    return {
        "label": label,
        "softmodems": [],
        "application_processes": [],
        "edge_container_absent": True,
        "ue_tunnel_absent": True,
        "conflicting_tcp_ports": [],
        "conflicting_udp_ports": [],
        "temporary_runtime_paths": [],
        "shared_memory_objects": [],
        "load_average": os.getloadavg(),
        "verified_at_unix_s": time.time(),
    }


def verify_actions(registry: SplitActionRegistry) -> dict[str, Any]:
    rows = []
    for action_id, profile_id in common.DIAGNOSTIC_ACTIONS:
        profile = registry.resolve(int(action_id))
        require(
            profile.profile_id == profile_id,
            f"action {action_id} resolves to {profile.profile_id}, expected {profile_id}",
        )
        require(
            profile.execution_mode == "SPLIT",
            f"action {action_id} is not a SPLIT action: {profile.execution_mode}",
        )
        require(profile.zstd_level == 1, f"action {action_id} entropy codec drift")
        rows.append(
            {
                "action_id": int(profile.action_id),
                "profile_id": profile.profile_id,
                "family": profile.family,
                "family_id": int(profile.family_id),
                "quantizer": profile.quantizer,
                "bit_width": int(profile.bit_width),
                "q_e4": int(profile.q_e4),
                "keep_count": int(profile.keep_count),
                "routing_tag": int(profile.routing_tag),
                "latent_width": profile.latent_width,
                "transported_channels": int(profile.transported_channels),
                "segmentation_installable": bool(profile.segmentation_installable),
                "segmentation_behavior": profile.segmentation_behavior,
                "entropy_coder": "zstd",
                "zstd_level": int(profile.zstd_level),
                "execution_mode": profile.execution_mode,
            }
        )
    require(
        len({row["action_id"] for row in rows}) == 4,
        "the diagnostic requires exactly four distinct actions",
    )
    return {"count": len(rows), "actions": rows}


# --------------------------------------------------------------------------
# fit-only sample and immutable payload precomputation
# --------------------------------------------------------------------------


def construct_registered_sample() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the exact registered Phase-13C fit-only 300-frame sample.

    Phase-13C guards its fit sampling as a CPU-only deterministic step, so
    this must run before anything initializes CUDA -- including the device
    identity preflight and every model preload. The ordering is asserted here
    rather than only inside Phase-13C so a reordering fails with a message
    that names the cause.
    """

    require(
        not torch.cuda.is_initialized(),
        "the registered Phase-13C fit sampling must run before CUDA is "
        "initialized; verify_cuda() and every model preload must follow it",
    )
    sample, context = p13c._construct_sample()
    require(
        sample["schema"] == p13c.SAMPLE_SCHEMA,
        "reconstructed sample schema drift",
    )
    require(
        int(sample["selected_frame_count"]) == common.FRAMES,
        f"registered sample must contain exactly {common.FRAMES} frames",
    )
    require(
        sample["sample_manifest_sha256"] == common.PHASE13C_SAMPLE_MANIFEST_SHA256,
        "reconstructed sample does not match the registered Phase-13C digest: "
        f"{sample['sample_manifest_sha256']}",
    )
    published = common.load_json(
        repo_path(common.PHASE13C_EVIDENCE_RELPATH) / "run_manifest.json"
    )["sample_manifest"]
    require(
        published["sample_manifest_sha256"] == sample["sample_manifest_sha256"]
        and published["selected_sample_id_sha256"] == sample["selected_sample_id_sha256"],
        "registered Phase-13C sample digests disagree with the published manifest",
    )
    require(
        all(row["registered_split"] == "fit" for row in sample["selected_rows"]),
        "diagnostic access is fit-only; a non-fit row appeared",
    )
    require(
        sample["access_scope"]["boxes_read"] is False
        and sample["access_scope"]["semantic_ground_truth_read"] is False
        and sample["access_scope"]["evaluation_records_read"] is False,
        "diagnostic must not read evaluation ground truth",
    )
    return sample, context


def _stream_id(run_id: str, profile_id: str) -> str:
    return f"{run_id}/{profile_id}"


def precompute_payloads(
    *,
    ue: Any,
    inference: Any,
    sample: Mapping[str, Any],
    profile: Any,
    run_id: str,
) -> dict[str, Any]:
    """Compute this action's 300 immutable inner payloads before any timing.

    Dataset loading, the front/backbone, the ranker, the AE encoder, quantize
    and pack and the zstd frame all happen here, so none of them can leak into
    the uplink measurement. Only bytes leave this function.
    """

    rows: list[dict[str, Any]] = []
    stream = _stream_id(run_id, profile.profile_id)
    for ordinal, selected in enumerate(sample["selected_rows"]):
        dataset_index = int(selected["dataset_index"])
        fused, row, calibration_cpu = inference[dataset_index]
        require(
            row["sample_id"] == selected["sample_id"]
            and int(row["frame_id"]) == int(selected["frame_id"]),
            "registered sample row drift during payload precomputation",
        )
        require(tuple(fused.shape) == (7, 448, 768), "fused input shape drift")
        require(
            fused.dtype is torch.float32 and bool(torch.isfinite(fused).all()),
            "fused input is not finite FP32",
        )
        input_7ch = fused.unsqueeze(0).to(ue.device)
        del fused, calibration_cpu
        sequence_id = FIRST_MEASURED_SEQUENCE_ID + ordinal
        started_ns = time.time_ns()
        with torch.inference_mode():
            prepared = ue.prepare(
                profile.action_id,
                input_7ch,
                sequence_id=sequence_id,
                capture_timestamp_ns=started_ns,
                frame_context=build_frame_context_v1(
                    stream_id=stream,
                    frame_id=sequence_id,
                    sequence_id=sequence_id,
                    capture_timestamp_ns=started_ns,
                    ego_world_x=float(selected["source_row"]["anchor_x"]),
                    ego_world_y=float(selected["source_row"]["anchor_y"]),
                    ego_world_z=float(selected["source_row"]["anchor_z"]),
                    ego_world_pitch=float(selected["source_row"]["anchor_pitch"]),
                    ego_world_yaw=float(selected["source_row"]["anchor_yaw"]),
                    ego_world_roll=float(selected["source_row"]["anchor_roll"]),
                ),
            )
        ready_ns = time.time_ns()
        del input_7ch
        # Only the scientific inner frame is retained; the envelope is packed
        # again against the real transmission schedule immediately before the
        # timed loop, so the wire bytes stay immutable and precomputed.
        inner = bytes(prepared.wire_bytes[-int(prepared.inner_payload_bytes) :])
        require(
            len(inner) == int(prepared.inner_payload_bytes),
            "inner payload extraction drift",
        )
        rows.append(
            {
                "ordinal": ordinal,
                "sequence_id": sequence_id,
                "frame_id": sequence_id,
                "sample_id": str(selected["sample_id"]),
                "episode_id": str(selected["episode_id"]),
                "registered_frame_id": int(selected["frame_id"]),
                "dataset_index": dataset_index,
                "inner": inner,
                "inner_payload_bytes": int(prepared.inner_payload_bytes),
                "ue_inner_payload_ready_wall_ns": ready_ns,
                "ue_prepare_stage_ns": {
                    str(boundary.name): int(
                        boundary.finished_monotonic_ns - boundary.started_monotonic_ns
                    )
                    for boundary in prepared.timing.boundaries
                },
                "ego_pose": (
                    float(selected["source_row"]["anchor_x"]),
                    float(selected["source_row"]["anchor_y"]),
                    float(selected["source_row"]["anchor_z"]),
                    float(selected["source_row"]["anchor_pitch"]),
                    float(selected["source_row"]["anchor_yaw"]),
                    float(selected["source_row"]["anchor_roll"]),
                ),
            }
        )
    warmup = sample["warmup"]
    warmup_index = int(warmup["dataset_index"])
    fused, row, _calibration = inference[warmup_index]
    require(
        row["sample_id"] == warmup["sample_id"],
        "registered warm-up row drift",
    )
    warmup_input = fused.unsqueeze(0).to(ue.device)
    del fused
    warmup_payloads: list[bytes] = []
    warmup_base_ns = time.time_ns()
    for index in range(WARMUP_ITERATIONS):
        sequence_id = index + 1
        capture_ns = warmup_base_ns + index * 1_000_000
        with torch.inference_mode():
            prepared = ue.prepare(
                profile.action_id,
                warmup_input,
                sequence_id=sequence_id,
                capture_timestamp_ns=capture_ns,
                frame_context=build_frame_context_v1(
                    stream_id=stream,
                    frame_id=sequence_id,
                    sequence_id=sequence_id,
                    capture_timestamp_ns=capture_ns,
                    ego_world_x=float(warmup["source_row"]["anchor_x"]),
                    ego_world_y=float(warmup["source_row"]["anchor_y"]),
                    ego_world_z=float(warmup["source_row"]["anchor_z"]),
                    ego_world_pitch=float(warmup["source_row"]["anchor_pitch"]),
                    ego_world_yaw=float(warmup["source_row"]["anchor_yaw"]),
                    ego_world_roll=float(warmup["source_row"]["anchor_roll"]),
                ),
            )
        warmup_payloads.append(bytes(prepared.wire_bytes))
    del warmup_input
    payload_bytes = {row["inner_payload_bytes"] for row in rows}
    return {
        "stream_id": stream,
        "rows": rows,
        "warmup_payloads": warmup_payloads,
        "distinct_inner_payload_sizes": sorted(payload_bytes),
    }


def materialize_datagrams(
    *,
    payloads: Mapping[str, Any],
    profile: Any,
    anchor_wall_ns: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    """Pack every envelope and datagram against the real 10 Hz schedule.

    Runs entirely before the timed transmission, so the loop performs no
    encoding, packing or allocation -- it only writes precomputed datagrams to
    the socket.
    """

    started_ns = time.time_ns()
    for row in payloads["rows"]:
        capture_ns = anchor_wall_ns + row["ordinal"] * common.SCHEDULE_PERIOD_NS
        context = build_frame_context_v1(
            stream_id=payloads["stream_id"],
            frame_id=int(row["frame_id"]),
            sequence_id=int(row["sequence_id"]),
            capture_timestamp_ns=capture_ns,
            ego_world_x=row["ego_pose"][0],
            ego_world_y=row["ego_pose"][1],
            ego_world_z=row["ego_pose"][2],
            ego_world_pitch=row["ego_pose"][3],
            ego_world_yaw=row["ego_pose"][4],
            ego_world_roll=row["ego_pose"][5],
        )
        wire = pack_envelope(
            row["inner"],
            action_id=int(profile.action_id),
            sequence_id=int(row["sequence_id"]),
            capture_timestamp_ns=capture_ns,
            frame_context=context,
        )
        datagrams = chunk_payload(
            wire, message_id=int(row["frame_id"]), chunk_bytes=chunk_bytes
        )
        row["capture_timestamp_ns"] = capture_ns
        row["scheduled_send_wall_ns"] = capture_ns
        row["sfd1_bytes"] = len(wire)
        row["sfd1_overhead_bytes"] = len(wire) - row["inner_payload_bytes"]
        row["datagrams"] = datagrams
        row["datagram_count"] = len(datagrams)
        row["udp_application_bytes"] = sum(len(item) for item in datagrams)
        row["estimated_wire_bytes"] = sum(len(item) + 28 for item in datagrams)
        row["ue_datagrams_ready_wall_ns"] = time.time_ns()
    finished_ns = time.time_ns()
    elapsed_ns = finished_ns - started_ns
    require(
        finished_ns < anchor_wall_ns - ACTUATION_LEAD_NS,
        "datagram materialization overran the transmission anchor; "
        f"needed {elapsed_ns / 1e6:.1f} ms of "
        f"{(TRANSMIT_ANCHOR_SLACK_NS - ACTUATION_LEAD_NS) / 1e6:.0f} ms slack",
    )
    return {
        "materialization_ms": elapsed_ns / 1e6,
        "anchor_wall_ns": int(anchor_wall_ns),
        "actuation_lead_ms": ACTUATION_LEAD_NS / 1e6,
        "slack_remaining_ms": (anchor_wall_ns - ACTUATION_LEAD_NS - finished_ns) / 1e6,
    }


# --------------------------------------------------------------------------
# radio lifecycle (the qualified launcher owns CN5G, gNB and the UE)
# --------------------------------------------------------------------------


def _radio_modules() -> tuple[Any, Any]:
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a
    from rl_agent import (
        splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b,
    )

    return phase14a, phase14b


def start_radio(
    campaign: Mapping[str, Any], *, cell_id: str, service_log_dir: Path
) -> tuple[Path, Path, dict[str, Any], Mapping[str, Any]]:
    phase14a, phase14b = _radio_modules()
    runtime = campaign["runtime"]
    base = phase14a.load_json(repo_path(str(runtime["phase14a_config"])))
    namespace = (
        ROOT
        / "experiments/splitfusion_oai_100mhz_4d5u_v1"
        / "splitfusion_timing_diagnostic_v1_radio_scratch"
        / cell_id
    )
    radio_state = namespace / f"00_{common.NETWORK_PROFILE_ID}"
    require(not radio_state.exists(), f"radio scratch already exists: {radio_state}")
    phase14b.require_cold_profile_runtime(base, radio_state)
    launcher = repo_path(str(runtime["oai_registered_profile_launcher"]))
    try:
        with (service_log_dir / "oai_launcher.log").open("xb") as stream:
            launched = subprocess.run(
                [
                    str(launcher), "--execute", "SPLITFUSION_OAI_100MHZ_4D5U_ATTACH",
                    "--output", str(radio_state),
                ],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, check=False, timeout=300.0,
            )
        require(
            launched.returncode == 0,
            f"qualified OAI launcher failed rc={launched.returncode}",
        )
        attached_path = radio_state / "ATTACHED_RADIO_STATE.json"
        require(attached_path.is_file(), "launcher omitted the attached-radio snapshot")
        attached = common.load_json(attached_path)
        require(
            attached.get("status") == "ATTACHED_STABLE_100MHZ_4D5U_ONE_UE",
            "radio attachment status drift",
        )
        for name in ("gnb", "ue"):
            topology = attached.get(f"{name}_process_topology", {})
            require(
                topology.get("endpoint_roles_verified") is True
                and int(topology.get("same_executable_process_count", -1)) == 3
                and len(topology.get("worker_pids", ())) == 2,
                f"qualified {name} three-process topology drift",
            )
        clean = phase14b.restore_interrupted_radio(base)
        require(
            clean is not None
            and clean.get("verified") is True
            and float(clean.get("noise_power_db", float("nan")))
            == common.CLEAN_NOISE_POWER_DB,
            "attached RFsim channel is not verified at noise_power_dB=-50",
        )
        return namespace, radio_state, {**attached, "clean_noise_preflight": clean}, base
    except BaseException as exc:
        restore_error = ""
        restored = None
        try:
            restored = phase14b.restore_interrupted_radio(base)
        except BaseException as restore_exc:
            restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
        cleanup = phase14b.teardown_profile_runtime(
            base, radio_state, namespace, None,
            restore_verified=bool(
                not restore_error and (restored is None or restored.get("verified"))
            ),
        )
        raise DiagnosticError(
            f"{type(exc).__name__}: {exc}; partial restore_error={restore_error!r} cleanup={cleanup}"
        ) from exc


def stop_radio(
    base: Mapping[str, Any],
    namespace: Path,
    radio_state: Path,
    attached: Mapping[str, Any] | None,
    *,
    actuator_restore_verified: bool,
) -> dict[str, Any]:
    """Restore -50 dB, then tear down unconditionally and prove cold state.

    Teardown runs even when the restore read-back cannot be taken, because
    refusing to tear down would leak the gNB/UE process groups. The restore
    verdict is asserted only after the radio is provably gone.
    """

    _phase14a, phase14b = _radio_modules()
    restored: Mapping[str, Any] | None = None
    restore_error = ""
    try:
        restored = phase14b.restore_interrupted_radio(base)
    except BaseException as exc:
        restore_error = f"{type(exc).__name__}: {exc}"
    verified = bool(
        restored is not None
        and restored.get("verified") is True
        and float(restored.get("noise_power_db", float("nan")))
        == common.CLEAN_NOISE_POWER_DB
    )
    # The target-SNR actuator already commanded and read back -50 dB before it
    # exited; an absent telnet actuator here therefore still leaves a verified
    # clean channel, and that is recorded rather than assumed.
    restore_status = (
        "READ_BACK_VERIFIED"
        if verified
        else (
            "RESTORE_ALREADY_VERIFIED_BY_ACTUATOR"
            if restored is None and not restore_error and actuator_restore_verified
            else "NOT_VERIFIED"
        )
    )
    report = phase14b.teardown_profile_runtime(
        base, radio_state, namespace, attached,
        restore_verified=verified or restore_status == "RESTORE_ALREADY_VERIFIED_BY_ACTUATOR",
    )
    require(
        bool(report.get("all_lifecycle_gates_passed")),
        "radio teardown/cold proof failed",
    )
    require(
        restore_status != "NOT_VERIFIED",
        "final RFsim restore to noise_power_dB=-50 was not verified: "
        f"restored={restored} error={restore_error!r}",
    )
    return {
        "final_restore": restored,
        "final_restore_status": restore_status,
        "final_restore_error": restore_error,
        "actuator_restore_verified": bool(actuator_restore_verified),
        "teardown": report,
    }


# --------------------------------------------------------------------------
# achieved SNR / MCS telemetry (aggregated only; no raw tracer log retained)
# --------------------------------------------------------------------------


class RadioTelemetry:
    """Bounded T-tracer PUSCH/MCS collection, summarized and then discarded."""

    PUSCH_FIELDS = (
        "time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
        "txpower_calc", "rbSize", "mcs", "rssi",
    )
    MCS_FIELDS = (
        "time", "rnti", "frame", "slot", "sched_frame", "sched_slot",
        "avg_snr_x10", "mcs_table", "ul_bler_mcs_before", "selected_mcs",
        "pre_phr_mcs", "post_phr_mcs", "final_mcs", "estimated_ul_buffer",
        "sched_ul_bytes", "B", "min_rb", "available_rb_before",
        "available_rb_after", "ph", "pcmax", "rb_size_final", "tbs_final",
        "force_ul_mcs",
    )

    def __init__(self, base: Mapping[str, Any], scratch: Path) -> None:
        self._base = base
        self._scratch = scratch
        self._processes: list[subprocess.Popen[str]] = []
        self._pusch: list[str] = []
        self._mcs: list[str] = []
        self._threads: list[threading.Thread] = []
        self._handles: list[Any] = []
        self.status = "NOT_STARTED"
        self.error = ""

    def _drain(self, process: subprocess.Popen[str], sink: list[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text and text[0].isdigit():
                sink.append(text)

    def start(self) -> None:
        from rl_agent import ue_n2_oai_ul_calibration_smoke as n2

        telemetry = self._base["telemetry"]
        tracer = repo_path("OAI/openairinterface5g/common/utils/T/tracer")
        messages = repo_path(str(self._base["paths"]["t_messages"]))
        relay_port = int(telemetry["gnb_relay_port"])
        try:
            require(
                n2.port_is_free(relay_port),
                f"T-tracer relay port {relay_port} is occupied",
            )
            relay_log = (self._scratch / "gnb_relay.log").open("wb")
            self._handles.append(relay_log)
            relay = subprocess.Popen(
                [
                    str(tracer / "multi"), "-d", str(messages), "-ip", "127.0.0.1",
                    "-p", str(telemetry["gnb_port"]), "-lp", str(relay_port),
                ],
                cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=relay_log,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            self._processes.append(relay)
            n2.wait_tcp(relay_port, 15)
            common_argv = [
                str(tracer / "csv"), "-d", str(messages), "-ip", "127.0.0.1",
                "-p", str(relay_port), "-f", "-s", ",", "-t", "time",
            ]
            for event, fields, sink in (
                ("GNB_MAC_PUSCH_POWER_CONTROL", self.PUSCH_FIELDS, self._pusch),
                ("GNB_MAC_UL_MCS_DECISION", self.MCS_FIELDS, self._mcs),
            ):
                process = subprocess.Popen(
                    [*common_argv, event, *fields], cwd=str(ROOT),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, bufsize=1,
                    start_new_session=True,
                )
                self._processes.append(process)
                thread = threading.Thread(
                    target=self._drain, args=(process, sink), daemon=True
                )
                thread.start()
                self._threads.append(thread)
            self.status = "COLLECTING"
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "TRACER_UNAVAILABLE"
            self.stop()

    def stop(self) -> None:
        for process in self._processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                except (ProcessLookupError, PermissionError):
                    try:
                        process.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass
        for process in self._processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        for thread in self._threads:
            thread.join(timeout=2.0)
        for handle in self._handles:
            try:
                handle.close()
            except OSError:
                pass
        self._processes.clear()
        self._threads.clear()
        self._handles.clear()

    def summary(self) -> dict[str, Any]:
        def column(rows: Sequence[str], fields: Sequence[str], name: str) -> list[float]:
            index = list(fields).index(name)
            values: list[float] = []
            for row in rows:
                parts = row.split(",")
                if len(parts) <= index:
                    continue
                try:
                    values.append(float(parts[index]))
                except ValueError:
                    continue
            return values

        pusch_snr = [value / 10.0 for value in column(self._pusch, self.PUSCH_FIELDS, "snrx10")]
        pusch_mcs = column(self._pusch, self.PUSCH_FIELDS, "mcs")
        selected_mcs = column(self._mcs, self.MCS_FIELDS, "selected_mcs")
        final_mcs = column(self._mcs, self.MCS_FIELDS, "final_mcs")
        mcs_snr = [value / 10.0 for value in column(self._mcs, self.MCS_FIELDS, "avg_snr_x10")]
        if self.status == "COLLECTING":
            self.status = "COLLECTED" if (pusch_snr or selected_mcs) else "NO_SAMPLES"
        return {
            "status": self.status,
            "error": self.error,
            "pusch_samples": len(self._pusch),
            "mcs_samples": len(self._mcs),
            "achieved_pusch_snr_db": common.summarize(pusch_snr),
            "achieved_pusch_mcs": common.summarize(pusch_mcs),
            "scheduler_avg_snr_db": common.summarize(mcs_snr),
            "scheduler_selected_ul_mcs": common.summarize(selected_mcs),
            "scheduler_final_ul_mcs": common.summarize(final_mcs),
            "raw_tracer_rows_retained": False,
        }


# --------------------------------------------------------------------------
# diagnostic edge container lifecycle
# --------------------------------------------------------------------------


def _edge_running() -> bool:
    completed = subprocess.run(
        ("sudo", "-n", "docker", "container", "inspect", "-f", "{{.State.Running}}", "oai-perception-rx"),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def request_edge_shutdown(
    stop_host: Path, summary_host: Path, *, timeout_s: float = 60.0
) -> dict[str, Any]:
    """Ask the edge to publish its final summary before the container stops.

    ``docker compose down`` terminates the edge process, so the final summary
    has to be requested and observed first. This writes the stop-file the edge
    watches and waits for ``final`` to appear, rather than weakening the
    integrity gate that requires it.
    """

    stop_host.touch(exist_ok=False)
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if summary_host.is_file():
            try:
                summary = common.load_json(summary_host)
            except (OSError, json.JSONDecodeError):
                summary = {}
            if bool(summary.get("final")):
                return {
                    "graceful_shutdown_observed": True,
                    "terminal_reason": summary.get("terminal_reason", ""),
                    "waited_s": round(
                        float(timeout_s) - (deadline - time.monotonic()), 3
                    ),
                }
        time.sleep(0.25)
    return {
        "graceful_shutdown_observed": False,
        "terminal_reason": "",
        "waited_s": float(timeout_s),
    }


def _stop_edge_container() -> bool:
    subprocess.run(
        [
            "sudo", "-n", "docker", "compose", "-f", "docker-compose.yaml",
            "-f", "docker-compose.fusion-back.yaml", "down", "--remove-orphans",
        ],
        cwd=str(ROOT / "receiver_container"), stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    return not _edge_running()


def _bounded_tail(path: Path, limit: int = 6144) -> str:
    if not path.is_file():
        return ""
    return path.read_bytes()[-limit:].decode("utf-8", errors="replace")


def start_edge_container(
    *,
    campaign: Mapping[str, Any],
    campaign_path: Path,
    profile: Any,
    warmup_payloads: Sequence[bytes],
    temporary_dir: Path,
    run_id: str,
    cell_id: str,
    warmup_stream_id: str = "",
) -> tuple[Path, dict[str, Any]]:
    runtime = campaign["runtime"]
    require(not _edge_running(), "a previous edge container is still running")
    state_root = (temporary_dir / EDGE_EVIDENCE_LEAF).resolve()
    state_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(state_root, 0o777)
    checkpoint_dir = state_root / "hub/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    record = campaign["deployment"]["fcos_constructor_weights"]
    source = repo_path(str(record["path"]))
    require(
        common.sha256_file(source) == str(record["sha256"]),
        "FCOS constructor-weight cache hash drift",
    )
    destination = checkpoint_dir / source.name
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    require(
        common.sha256_file(destination) == str(record["sha256"]),
        "seeded FCOS cache hash drift",
    )
    # Transient warm-up payloads live only on this cell's scratch mount and are
    # deleted with it; no payload blob is ever retained as evidence.
    warmup_host = state_root / "warmup_payloads.bin"
    offsets: list[int] = []
    with warmup_host.open("xb") as handle:
        for payload in warmup_payloads:
            offsets.append(handle.tell())
            handle.write(payload)
    warmup_index = state_root / "warmup_index.json"
    common.atomic_create_json(
        warmup_index,
        {
            "count": len(warmup_payloads),
            "offsets": offsets,
            "sizes": [len(payload) for payload in warmup_payloads],
        },
    )
    os.chmod(warmup_host, 0o666)
    os.chmod(warmup_index, 0o666)

    container_state = Path("/work/torch_cache")
    ready_host = state_root / "ready.json"
    records_host = state_root / "edge_records.jsonl"
    summary_host = state_root / "edge_summary.json"
    env = os.environ.copy()
    env.update(
        {
            "FUSION_BACK_DUAL": "0",
            "FUSION_BACK_BIND_HOST": "0.0.0.0",
            "FUSION_BACK_REMOTE_HOST": str(runtime["ue_bind_host"]),
            "FUSION_BACK_REMOTE_HOST_1": str(runtime["ue_bind_host"]),
            "FUSION_BACK_DEVICE": "cuda",
            "SPLITFUSION_EDGE_STATE_ROOT": str(state_root),
            "SPLITFUSION_FCOS_WEIGHT_PATH": str(source),
            "FUSION_BACK_SCRIPT": f"-m {EDGE_MODULE}",
            "FUSION_REMOTE_PORT_1": str(runtime["edge_receive_port"]),
            "FUSION_REMOTE_SOURCE_PORT_1": str(runtime["edge_source_port"]),
            "FUSION_CAMERA_RESULT_PORT_1": str(runtime["camera_result_port"]),
            "FUSION_BACK_EXTRA_ARGS": " ".join(
                (
                    "--diagnostic-edge",
                    "--config", str(Path("/work/abiodun") / campaign_path.relative_to(ROOT)),
                    "--action-id", str(profile.action_id),
                    "--allowed-action-ids", str(profile.action_id),
                    "--ready-file", str(container_state / "ready.json"),
                    "--records-file", str(container_state / "edge_records.jsonl"),
                    "--summary-file", str(container_state / "edge_summary.json"),
                    "--warmup-payload", str(container_state / "warmup_payloads.bin"),
                    "--stop-file", str(container_state / "stop_edge"),
                    "--warmup-iterations", str(len(warmup_payloads)),
                    "--first-measured-sequence-id", str(FIRST_MEASURED_SEQUENCE_ID),
                    "--warmup-stream-id", str(warmup_stream_id),
                    "--edge-port", str(runtime["edge_receive_port"]),
                    "--result-host", str(runtime["ue_bind_host"]),
                    "--result-port", str(runtime["camera_result_port"]),
                    "--run-id", run_id,
                    "--cell-id", cell_id,
                )
            ),
        }
    )
    launcher_log = temporary_dir / "edge_launcher.log"
    try:
        with launcher_log.open("xb") as stream:
            completed = subprocess.run(
                [str(ROOT / "scripts/receiver_container_fusion_back_up.sh")],
                cwd=str(ROOT), env=env, check=False, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, timeout=300.0,
            )
        require(
            completed.returncode == 0,
            f"diagnostic edge container startup failed rc={completed.returncode}; "
            f"launcher_tail={_bounded_tail(launcher_log)!r}",
        )
        deadline = time.monotonic() + 420.0
        while time.monotonic() < deadline:
            require(_edge_running(), "diagnostic edge container exited before readiness")
            if ready_host.is_file():
                ready = common.load_json(ready_host)
                require(
                    ready.get("schema") == common.EDGE_READY_SCHEMA
                    and int(ready.get("action_id", -1)) == int(profile.action_id)
                    and ready.get("allowed_action_ids") == [int(profile.action_id)]
                    and ready.get("tail_device") == "cuda:0"
                    and ready.get("dense_label_map_on_radio") is False,
                    "diagnostic edge readiness identity drift",
                )
                equivalence = ready.get("parent_equivalence") or {}
                require(
                    equivalence.get("perception_bitwise_identical") is True
                    and equivalence.get("service_records_byte_identical") is True
                    and equivalence.get("segmentation_labels_bitwise_identical") is True,
                    "instrumented tail did not prove equivalence to the production tail",
                )
                return state_root, {
                    "ready": ready,
                    "records_host": str(records_host),
                    "summary_host": str(summary_host),
                    "stop_host": str(state_root / "stop_edge"),
                }
            time.sleep(0.25)
        raise DiagnosticError("diagnostic edge did not become ready")
    except BaseException as exc:
        logs = subprocess.run(
            ("sudo", "-n", "docker", "logs", "--tail", "120", "oai-perception-rx"),
            cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        ).stdout[-8192:]
        _stop_edge_container()
        raise DiagnosticError(
            f"{type(exc).__name__}: {exc}; edge_container_tail={logs!r}"
        ) from exc


def inspect_edge_mounts(state_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ("sudo", "-n", "docker", "inspect", "-f", "{{json .Mounts}}", "oai-perception-rx"),
        cwd=str(ROOT), check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(completed.returncode == 0, "cannot inspect the diagnostic edge mounts")
    mounts = json.loads(completed.stdout)

    def selected(destination: str) -> Mapping[str, Any]:
        rows = [row for row in mounts if row.get("Destination") == destination]
        require(len(rows) == 1, f"edge mount is not unique: {destination}")
        return rows[0]

    state = selected("/work/torch_cache")
    repository = selected("/work/abiodun")
    require(
        Path(str(state["Source"])).resolve(strict=True) == state_root.resolve(strict=True)
        and bool(state.get("RW")) is True,
        "edge state mount source/mode drift",
    )
    require(
        Path(str(repository["Source"])).resolve(strict=True) == ROOT
        and bool(repository.get("RW")) is False,
        "edge repository mount must be the read-only project root",
    )
    return {
        "state_mount_source": str(state["Source"]),
        "state_mount_rw": True,
        "repository_mount_source": str(repository["Source"]),
        "repository_mount_rw": False,
    }


# --------------------------------------------------------------------------
# UE transmission on the registered 10 Hz schedule
# --------------------------------------------------------------------------


class ResultReceiver:
    """Count the compact results the edge returns; never on the send path."""

    def __init__(self, bind_host: str, port: int, buffer_bytes: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((bind_host, int(port)))
        self.socket.settimeout(0.2)
        self.reassembler = ChunkReassembler(timeout_s=2.0, max_chunks=4096)
        self.results: dict[int, dict[str, Any]] = {}
        self.datagrams = 0
        self.malformed = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="diag-ue-results", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                datagram, address = self.socket.recvfrom(65535)
            except socket.timeout:
                self.reassembler.expire(time.monotonic())
                continue
            except OSError:
                return
            self.datagrams += 1
            try:
                complete = self.reassembler.ingest(
                    str(address), datagram, received_at_s=time.monotonic()
                )
            except ValueError:
                self.malformed += 1
                continue
            if complete is None:
                continue
            received_ns = time.time_ns()
            try:
                value = json.loads(complete.payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.malformed += 1
                continue
            value["ue_result_received_wall_ns"] = received_ns
            self.results[int(value.get("frame_id", -1))] = value

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.socket.close()


def transmit_schedule(
    *,
    payloads: Mapping[str, Any],
    remote: tuple[str, int],
    bind_host: str,
    buffer_bytes: int,
    anchor_wall_ns: int,
) -> dict[str, Any]:
    """Send every precomputed message on its own 100 ms slot; never burst."""

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
    sender.bind((bind_host, 0))
    reported = sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    sent = 0
    obsolete = 0
    datagrams = 0
    try:
        for row in payloads["rows"]:
            scheduled = int(row["scheduled_send_wall_ns"])
            while True:
                remaining_ns = scheduled - time.time_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1e9, 0.005))
            now_ns = time.time_ns()
            if now_ns >= scheduled + common.SCHEDULE_PERIOD_NS:
                # SKIP_OBSOLETE_NEVER_BURST: the slot is gone, so this capture
                # is abandoned rather than transmitted late in a burst.
                row["send_status"] = "OBSOLETE_SKIPPED"
                row["ue_first_send_wall_ns"] = 0
                row["ue_final_send_wall_ns"] = 0
                row["slot_lateness_ms"] = (now_ns - scheduled) / 1e6
                obsolete += 1
                continue
            first_send_ns = time.time_ns()
            for datagram in row["datagrams"]:
                sender.sendto(datagram, remote)
            final_send_ns = time.time_ns()
            require(
                final_send_ns >= first_send_ns,
                "negative UE send-loop interval",
            )
            row["send_status"] = "SENT"
            row["ue_first_send_wall_ns"] = first_send_ns
            row["ue_final_send_wall_ns"] = final_send_ns
            row["slot_lateness_ms"] = (first_send_ns - scheduled) / 1e6
            sent += 1
            datagrams += row["datagram_count"]
    finally:
        sender.close()
    return {
        "messages_sent": sent,
        "obsolete_skipped": obsolete,
        "feature_datagrams_transmitted": datagrams,
        "ue_send_buffer_reported_bytes": int(reported),
        "anchor_wall_ns": int(anchor_wall_ns),
    }


# --------------------------------------------------------------------------
# controlled local tail microbenchmark (300 observations per action)
# --------------------------------------------------------------------------


def _repack(
    row: Mapping[str, Any],
    *,
    stream_id: str,
    action_id: int,
    capture_timestamp_ns: int,
) -> bytes:
    """Re-pack one precomputed inner payload into its SFD1 v2 envelope.

    The scientific inner bytes are the immutable precomputed frame; only the
    envelope's stream/capture identity is rebound, and the frozen tail depends
    on the transported ego pose rather than on the capture timestamp, so the
    reconstructed C2 and every model computation are unchanged.
    """

    context = build_frame_context_v1(
        stream_id=stream_id,
        frame_id=int(row["frame_id"]),
        sequence_id=int(row["sequence_id"]),
        capture_timestamp_ns=int(capture_timestamp_ns),
        ego_world_x=row["ego_pose"][0],
        ego_world_y=row["ego_pose"][1],
        ego_world_z=row["ego_pose"][2],
        ego_world_pitch=row["ego_pose"][3],
        ego_world_yaw=row["ego_pose"][4],
        ego_world_roll=row["ego_pose"][5],
    )
    return pack_envelope(
        row["inner"],
        action_id=int(action_id),
        sequence_id=int(row["sequence_id"]),
        capture_timestamp_ns=int(capture_timestamp_ns),
        frame_context=context,
    )


def tail_microbenchmark(
    *,
    edge: Any,
    payloads: Mapping[str, Any],
    profile: Any,
    warmup_frames: int = MICROBENCH_WARMUP,
) -> dict[str, Any]:
    """Decompose the resident edge path over all 300 reconstructed samples.

    Independent of live delivery, so tail timing always has exactly 300
    observations per action. Nothing is loaded, constructed, moved or set to
    eval here: the resident edge from startup is reused.
    """

    action_id = int(profile.action_id)
    stream_id = f"{payloads['stream_id']}/microbenchmark"
    edge.runtime.reset_context_session()
    counters_before = dict(edge.runtime.counters.__dict__)

    # Self-contained monotone schedule so the controlled microbenchmark never
    # depends on the live transmission anchor.
    anchor_ns = time.time_ns()
    wires = [
        _repack(
            row,
            stream_id=stream_id,
            action_id=action_id,
            capture_timestamp_ns=anchor_ns + int(row["ordinal"]) * common.SCHEDULE_PERIOD_NS,
        )
        for row in payloads["rows"]
    ]
    for index in range(min(warmup_frames, len(wires))):
        edge.tail.begin_frame()
        result = edge.runtime.process(wires[index], transmitted_action_id=action_id)
        edge.tail.resolve_frame()
        edge.tail.take_snapshot()
        del result
    torch.cuda.synchronize(edge.device)
    edge.runtime.reset_context_session()

    observations: list[dict[str, Any]] = []
    for row, wire in zip(payloads["rows"], wires):
        started_ns = time.perf_counter_ns()
        edge.tail.begin_frame()
        result = edge.runtime.process(wire, transmitted_action_id=action_id)
        finished_ns = time.perf_counter_ns()
        stages = edge.tail.resolve_frame()
        snapshot = edge.tail.take_snapshot()
        resolved_ns = time.perf_counter_ns()
        require(
            finished_ns >= started_ns and resolved_ns >= finished_ns,
            "negative microbenchmark interval",
        )
        require(
            str(result.metadata.profile_id) == profile.profile_id,
            "microbenchmark action/profile binding drift",
        )
        require(
            tuple(snapshot.semantic_labels.shape) == (720, 1280),
            f"segmentation label shape drift: {tuple(snapshot.semantic_labels.shape)}",
        )
        observations.append(
            {
                "ordinal": int(row["ordinal"]),
                "sequence_id": int(row["sequence_id"]),
                "edge_process_wall_ms": (finished_ns - started_ns) / 1e6,
                "cuda_settled_wall_ms": (resolved_ns - started_ns) / 1e6,
                "edge_stage_ns": {
                    str(boundary.name): int(
                        boundary.finished_monotonic_ns - boundary.started_monotonic_ns
                    )
                    for boundary in result.timing.boundaries
                },
                "tail_stage_wall_ns": stages["wall_ns"],
                "tail_stage_cuda_ms": stages["cuda_ms"],
                "service_record_count": len(snapshot.records or ()),
                "service_record_bytes": len(snapshot.serialized_records or b""),
                "finite_output_tensor_count": int(snapshot.output_tensor_count),
                "scientific_inner_payload_bytes": int(
                    result.scientific_inner_payload_bytes
                ),
            }
        )
        del snapshot, result
    del wires
    counters_after = dict(edge.runtime.counters.__dict__)
    require(
        len(observations) == common.FRAMES,
        f"microbenchmark produced {len(observations)} of {common.FRAMES} observations",
    )
    require(
        int(counters_after.get("hot_path_model_load_operations", 0)) == 0
        and int(counters_after.get("hot_path_model_construction_operations", 0)) == 0,
        "microbenchmark performed a hot-path model load or construction",
    )
    return {
        "observations": observations,
        "counters_before": counters_before,
        "counters_after": counters_after,
        "warmup_frames": int(min(warmup_frames, common.FRAMES)),
    }


# --------------------------------------------------------------------------
# one action: one fresh radio lifecycle and one measurement
# --------------------------------------------------------------------------


def _stage_ms(record: Mapping[str, Any], key: str, stage: str) -> float | None:
    value = record.get(key, {})
    if stage not in value:
        return None
    raw = value[stage]
    return float(raw) / 1e6 if key.endswith("_ns") else float(raw)


def join_records(
    *,
    payloads: Mapping[str, Any],
    edge_records: Sequence[Mapping[str, Any]],
    results: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join UE and edge observations on the transported message identity."""

    by_sequence = {int(record["sequence_id"]): record for record in edge_records}
    require(
        len(by_sequence) == len(edge_records),
        "edge records contain duplicate sequence identities",
    )
    rows: list[dict[str, Any]] = []
    for row in payloads["rows"]:
        sequence_id = int(row["sequence_id"])
        edge = by_sequence.get(sequence_id)
        result = results.get(int(row["frame_id"]))
        joined: dict[str, Any] = {
            "ordinal": int(row["ordinal"]),
            "sequence_id": sequence_id,
            "frame_id": int(row["frame_id"]),
            "sample_id": row["sample_id"],
            "episode_id": row["episode_id"],
            "registered_frame_id": int(row["registered_frame_id"]),
            "dataset_index": int(row["dataset_index"]),
            "inner_payload_bytes": int(row["inner_payload_bytes"]),
            "sfd1_overhead_bytes": int(row["sfd1_overhead_bytes"]),
            "sfd1_bytes": int(row["sfd1_bytes"]),
            "datagram_count": int(row["datagram_count"]),
            "udp_application_bytes": int(row["udp_application_bytes"]),
            "estimated_wire_bytes": int(row["estimated_wire_bytes"]),
            "capture_timestamp_ns": int(row["capture_timestamp_ns"]),
            "scheduled_send_wall_ns": int(row["scheduled_send_wall_ns"]),
            "ue_inner_payload_ready_wall_ns": int(row["ue_inner_payload_ready_wall_ns"]),
            "ue_payload_ready_wall_ns": int(row["ue_datagrams_ready_wall_ns"]),
            "send_status": str(row.get("send_status", "NOT_ATTEMPTED")),
            "slot_lateness_ms": row.get("slot_lateness_ms"),
            "ue_first_send_wall_ns": int(row.get("ue_first_send_wall_ns", 0)) or None,
            "ue_final_send_wall_ns": int(row.get("ue_final_send_wall_ns", 0)) or None,
            "ue_prepare_total_ms": row["ue_prepare_stage_ns"].get(
                "total_ue_preparation", 0
            ) / 1e6,
            "delivered": edge is not None,
            "result_returned": result is not None,
        }
        first = joined["ue_first_send_wall_ns"]
        final = joined["ue_final_send_wall_ns"]
        joined["ue_send_loop_ms"] = (
            (final - first) / 1e6 if first and final else None
        )
        if edge is not None:
            reassembled = int(edge["edge_complete_reassembly_wall_ns"])
            first_datagram = int(edge["edge_first_datagram_wall_ns"])
            worker_start = int(edge["edge_worker_start_wall_ns"])
            joined.update(
                {
                    "edge_first_datagram_wall_ns": first_datagram,
                    "edge_complete_reassembly_wall_ns": reassembled,
                    "edge_admitted_wall_ns": int(edge["edge_admitted_wall_ns"]),
                    "edge_worker_start_wall_ns": worker_start,
                    "edge_tail_finished_wall_ns": int(edge["edge_tail_finished_wall_ns"]),
                    "edge_feature_datagrams": int(edge["feature_datagrams"]),
                    "edge_duplicate_datagrams": int(edge["duplicate_datagrams"]),
                    "edge_service_wall_ms": float(edge["edge_service_wall_ms"]),
                    "edge_result_finite_check_ms": float(
                        edge["edge_result_finite_check_ms"]
                    ),
                    "service_record_count": int(edge["service_record_count"]),
                    "service_record_bytes": int(edge["service_record_bytes"]),
                    "service_target_met": bool(edge["service_target_met"]),
                    "processing_horizon_met": bool(edge["processing_horizon_met"]),
                    "edge_first_to_complete_reassembly_ms": (
                        reassembled - first_datagram
                    ) / 1e6,
                    "edge_queue_wait_ms": (worker_start - reassembled) / 1e6,
                }
            )
            if first:
                joined["application_feature_uplink_ms"] = (reassembled - first) / 1e6
            if final:
                joined["post_send_to_reassembly_ms"] = (reassembled - final) / 1e6
            for stage in EDGE_STAGES:
                joined[f"edge_{stage}_ms"] = _stage_ms(edge, "edge_stage_ns", stage)
            for stage in (*TAIL_STAGES, SERIALIZE_STAGE, TOTAL_STAGE):
                joined[f"tail_{stage}_ms"] = _stage_ms(edge, "tail_stage_wall_ns", stage)
                joined[f"tail_{stage}_cuda_ms"] = (edge.get("tail_stage_cuda_ms") or {}).get(
                    stage
                )
            joined["decode_tail_cuda_ms"] = (edge.get("tail_stage_cuda_ms") or {}).get(
                "decode_tail_cuda"
            )
            launch = joined.get("tail_decode_tail_launch_ms")
            settle = joined.get("tail_finite_check_outputs_ms")
            if launch is not None and settle is not None:
                joined["decode_tail_inference_block_ms"] = launch + settle
        if result is not None:
            joined["ue_result_received_wall_ns"] = int(
                result["ue_result_received_wall_ns"]
            )
            if final:
                joined["ue_round_trip_ms"] = (
                    int(result["ue_result_received_wall_ns"]) - final
                ) / 1e6
        rows.append(joined)
    return rows


def attach_microbenchmark(
    rows: Sequence[dict[str, Any]], microbenchmark: Mapping[str, Any]
) -> None:
    by_sequence = {
        int(item["sequence_id"]): item for item in microbenchmark["observations"]
    }
    for row in rows:
        observation = by_sequence.get(int(row["sequence_id"]))
        require(observation is not None, "microbenchmark observation is missing")
        row["mb_edge_process_wall_ms"] = observation["edge_process_wall_ms"]
        row["mb_cuda_settled_wall_ms"] = observation["cuda_settled_wall_ms"]
        row["mb_service_record_count"] = observation["service_record_count"]
        row["mb_service_record_bytes"] = observation["service_record_bytes"]
        for stage in EDGE_STAGES:
            row[f"mb_edge_{stage}_ms"] = _stage_ms(observation, "edge_stage_ns", stage)
        for stage in (*TAIL_STAGES, SERIALIZE_STAGE, TOTAL_STAGE):
            row[f"mb_tail_{stage}_ms"] = _stage_ms(
                observation, "tail_stage_wall_ns", stage
            )
            row[f"mb_tail_{stage}_cuda_ms"] = observation["tail_stage_cuda_ms"].get(stage)
        row["mb_decode_tail_cuda_ms"] = observation["tail_stage_cuda_ms"].get(
            "decode_tail_cuda"
        )
        launch = row.get("mb_tail_decode_tail_launch_ms")
        settle = row.get("mb_tail_finite_check_outputs_ms")
        if launch is not None and settle is not None:
            row["mb_decode_tail_inference_block_ms"] = launch + settle


def _start_actuator(
    *, campaign_path: Path, temporary_dir: Path, start_file: Path
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    output = temporary_dir / "radio_trace.csv"
    stop_file = temporary_dir / "stop_target_snr"
    process = subprocess.Popen(
        [
            sys.executable, "-m", TARGET_SNR_MODULE,
            "--campaign", str(campaign_path),
            "--profile-id", common.NETWORK_PROFILE_ID,
            "--output", str(output),
            "--stop-file", str(stop_file),
            "--start-file", str(start_file),
        ],
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
    )
    return process, output, stop_file


def _stop_actuator(
    process: subprocess.Popen[Any], output: Path, stop_file: Path
) -> dict[str, Any]:
    stop_file.touch(exist_ok=False)
    try:
        returncode = process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        try:
            returncode = process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=10.0)
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary = common.load_json(summary_path) if summary_path.is_file() else {}
    rows: list[dict[str, str]] = []
    if output.is_file():
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    require(int(returncode) == 0, f"target-SNR actuator exited rc={returncode}")
    require(
        summary.get("clean_restore_verified") is True
        and float(summary.get("clean_restore_noise_power_db", float("nan")))
        == common.CLEAN_NOISE_POWER_DB
        and not summary.get("error"),
        f"target-SNR actuator did not verify the clean -50 dB restore: {summary}",
    )
    require(
        str(summary.get("profile_id")) == common.NETWORK_PROFILE_ID
        and str(summary.get("trace_id")) == "GM_V2_FAVORABLE_STABLE_SEED_2026082101",
        "actuated network profile identity drift",
    )
    require(
        bool(summary.get("same_sequence_continued_after_prefix")) is False,
        "the diagnostic must stay inside the frozen 4200-sample prefix; "
        "no wrapping, hold or reseed is permitted",
    )
    applied = [row for row in rows if row.get("command_timing_status") != "SKIP_OBSOLETE_NEVER_BURST"]
    targets = [float(row["target_snr_db"]) for row in rows if row.get("target_snr_db")]
    commands = [
        float(row["mapped_rfsim_command_db"])
        for row in rows
        if row.get("mapped_rfsim_command_db")
    ]
    latencies = [
        float(row["command_latency_ms"]) for row in applied if row.get("command_latency_ms")
    ]
    late = [row for row in applied if row.get("command_timing_status") == "ACK_LATE"]
    first_targets = targets[: common.FRAMES]
    return {
        "summary": summary,
        "rows": len(rows),
        "commands_applied": len(applied),
        "obsolete_command_skips": len(rows) - len(applied),
        "late_command_acks": len(late),
        "profile_id": common.NETWORK_PROFILE_ID,
        "trace_id": str(summary.get("trace_id", "")),
        "seed": int(summary.get("seed", 0)),
        "sample_period_ms": 100,
        "target_snr_db": common.summarize(targets),
        "first_300_target_snr_db": common.summarize(first_targets),
        "first_300_target_digest": common.digest_bytes(
            common.canonical_bytes([f"{value:.12g}" for value in first_targets])
        ),
        "mapped_rfsim_command_db": common.summarize(commands),
        "command_latency_ms": common.summarize(latencies),
        "clean_restore_verified": True,
        "returncode": int(returncode),
    }


def run_action(
    *,
    action_id: int,
    campaign: Mapping[str, Any],
    campaign_path: Path,
    sample: Mapping[str, Any],
    ue: Any,
    inference: Any,
    host_edge: Any,
    registry: SplitActionRegistry,
    run_id: str,
) -> dict[str, Any]:
    profile = registry.resolve(int(action_id))
    cell_id = f"diag_a{action_id:02d}__{common.NETWORK_PROFILE_ID.lower()}"
    runtime = campaign["runtime"]
    started_at = time.time()
    print(f"[{cell_id}] cold preflight", flush=True)
    cold_before = verify_cold_host(campaign, label=f"{cell_id}/before")

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f"splitfusion_timing_diagnostic_{cell_id}_")
    )
    radio_namespace: Path | None = None
    radio_state: Path | None = None
    attached: Mapping[str, Any] | None = None
    radio_base: Mapping[str, Any] | None = None
    telemetry: RadioTelemetry | None = None
    actuator: subprocess.Popen[bytes] | None = None
    actuator_output: Path | None = None
    actuator_stop: Path | None = None
    edge_state: Path | None = None
    receiver: ResultReceiver | None = None
    payloads: dict[str, Any] | None = None
    edge_records: list[dict[str, Any]] = []
    edge_meta: dict[str, Any] = {}
    report: dict[str, Any] = {
        "cell_id": cell_id,
        "action_id": int(action_id),
        "profile_id": profile.profile_id,
        "network_profile_id": common.NETWORK_PROFILE_ID,
        "started_at_unix_s": started_at,
        "cold_before": cold_before,
    }
    try:
        print(f"[{cell_id}] precomputing {common.FRAMES} immutable payloads", flush=True)
        payloads = precompute_payloads(
            ue=ue, inference=inference, sample=sample, profile=profile, run_id=run_id
        )
        report["payload_precomputation"] = {
            "frames": len(payloads["rows"]),
            "warmup_payloads": len(payloads["warmup_payloads"]),
            "distinct_inner_payload_sizes": payloads["distinct_inner_payload_sizes"],
            "inner_payload_bytes": common.summarize(
                [row["inner_payload_bytes"] for row in payloads["rows"]]
            ),
        }

        print(f"[{cell_id}] starting the qualified 100 MHz/273 PRB/4D5U radio", flush=True)
        radio_namespace, radio_state, attached, radio_base = start_radio(
            campaign, cell_id=cell_id, service_log_dir=temporary_dir
        )
        report["radio_attachment"] = {
            "status": attached.get("status"),
            "clean_noise_preflight": attached.get("clean_noise_preflight"),
            "radio_profile_id": "OAI_N78_100MHZ_273PRB_4D5U_V1",
        }

        telemetry = RadioTelemetry(radio_base, temporary_dir)
        telemetry.start()

        print(f"[{cell_id}] starting the diagnostic edge and warming it up", flush=True)
        edge_state, edge_meta = start_edge_container(
            campaign=campaign,
            campaign_path=campaign_path,
            profile=profile,
            warmup_payloads=payloads["warmup_payloads"],
            temporary_dir=temporary_dir,
            run_id=run_id,
            cell_id=cell_id,
        )
        report["edge_ready"] = edge_meta["ready"]
        report["edge_mounts"] = inspect_edge_mounts(edge_state)

        receiver = ResultReceiver(
            str(runtime["ue_bind_host"]),
            int(runtime["camera_result_port"]),
            int(runtime["socket_buffer_request_bytes"]),
        )

        start_file = temporary_dir / "start_target_snr"
        actuator, actuator_output, actuator_stop = _start_actuator(
            campaign_path=campaign_path, temporary_dir=temporary_dir, start_file=start_file
        )
        time.sleep(1.0)
        require(actuator.poll() is None, "target-SNR actuator exited before actuation")

        ue_anchor_start = common.clock_anchor("ue_process_start")
        anchor_wall_ns = time.time_ns() + TRANSMIT_ANCHOR_SLACK_NS
        materialization = materialize_datagrams(
            payloads=payloads,
            profile=profile,
            anchor_wall_ns=anchor_wall_ns,
            chunk_bytes=int(runtime["udp_chunk_bytes"]),
        )
        report["datagram_materialization"] = materialization
        # Arm the actuator a fixed 200 ms before slot 0 for every action, so
        # frame i always corresponds to the same frozen profile target index.
        while True:
            remaining_ns = (anchor_wall_ns - ACTUATION_LEAD_NS) - time.time_ns()
            if remaining_ns <= 0:
                break
            time.sleep(min(remaining_ns / 1e9, 0.005))
        start_file.touch(exist_ok=False)
        print(
            f"[{cell_id}] transmitting {common.FRAMES} messages on the 100 ms schedule",
            flush=True,
        )
        transmission = transmit_schedule(
            payloads=payloads,
            remote=(str(runtime["edge_remote_host"]), int(runtime["edge_receive_port"])),
            bind_host=str(runtime["ue_bind_host"]),
            buffer_bytes=int(runtime["socket_buffer_request_bytes"]),
            anchor_wall_ns=anchor_wall_ns,
        )
        report["transmission"] = transmission
        time.sleep(DRAIN_SECONDS)
        ue_anchor_end = common.clock_anchor("ue_process_end")
        report["ue_clock_anchors"] = [ue_anchor_start, ue_anchor_end]

        report["radio_actuation"] = _stop_actuator(actuator, actuator_output, actuator_stop)
        actuator = None
        telemetry.stop()
        report["radio_telemetry"] = telemetry.summary()
        telemetry = None

        require(_edge_running(), "the diagnostic edge exited during measurement")
        edge_records_path = Path(edge_meta["records_host"])
        edge_summary_path = Path(edge_meta["summary_host"])
        report["edge_shutdown"] = request_edge_shutdown(
            Path(edge_meta["stop_host"]), edge_summary_path
        )
        require(
            bool(report["edge_shutdown"]["graceful_shutdown_observed"]),
            "the diagnostic edge did not publish its final summary before shutdown",
        )
        require(_stop_edge_container(), "the diagnostic edge container did not stop")
        require(edge_summary_path.is_file(), "the edge published no final summary")
        edge_summary = common.load_json(edge_summary_path)
        require(bool(edge_summary.get("final")), "the edge summary is not the final one")
        edge_records = []
        if edge_records_path.is_file():
            for line in edge_records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    edge_records.append(json.loads(line))
        require(
            all(row.get("phase") == "MEASURED" for row in edge_records),
            "a warm-up record leaked into the measured edge records",
        )
        require(
            all(int(row["action_id"]) == int(action_id) for row in edge_records),
            "an edge record carries a foreign action",
        )
        require(
            not edge_summary.get("failures"),
            f"the edge reported failures: {edge_summary.get('failures')}",
        )
        report["edge_summary"] = edge_summary
        for anchor in (
            edge_summary["start_clock_anchor"],
            edge_summary["clock_anchor"],
        ):
            skew_ns = abs(
                int(anchor["wall_minus_monotonic_ns"])
                - int(ue_anchor_start["wall_minus_monotonic_ns"])
            )
            require(
                skew_ns < 5_000_000,
                "edge container and UE host do not share one wall-clock domain: "
                f"{skew_ns} ns of wall/monotonic offset skew",
            )
        report["clock_domain"] = {
            "ue_anchors": [ue_anchor_start, ue_anchor_end],
            "edge_anchors": [
                edge_summary["start_clock_anchor"],
                edge_summary["clock_anchor"],
            ],
            "shared_wall_clock_domain_verified": True,
            "maximum_wall_minus_monotonic_skew_ns": max(
                abs(
                    int(anchor["wall_minus_monotonic_ns"])
                    - int(ue_anchor_start["wall_minus_monotonic_ns"])
                )
                for anchor in (
                    edge_summary["start_clock_anchor"],
                    edge_summary["clock_anchor"],
                )
            ),
        }
    finally:
        if actuator is not None and actuator_output is not None and actuator_stop is not None:
            try:
                report.setdefault("radio_actuation_error", "")
                report["radio_actuation"] = _stop_actuator(
                    actuator, actuator_output, actuator_stop
                )
            except BaseException as exc:
                report["radio_actuation_error"] = f"{type(exc).__name__}: {exc}"
        if telemetry is not None:
            telemetry.stop()
            report["radio_telemetry"] = telemetry.summary()
        if receiver is not None:
            receiver.close()
            report["result_downlink"] = {
                "result_datagrams_received": receiver.datagrams,
                "malformed_result_datagrams": receiver.malformed,
                "compact_results_returned": len(receiver.results),
            }
        if _edge_running():
            report["edge_forced_stop"] = _stop_edge_container()
        if radio_base is not None and radio_namespace is not None and radio_state is not None:
            report["radio_teardown"] = stop_radio(
                radio_base, radio_namespace, radio_state, attached,
                actuator_restore_verified=bool(
                    report.get("radio_actuation", {}).get("clean_restore_verified")
                ),
            )
        # This cell's scratch holds its payload blobs, its warm-up blob and the
        # service logs, and its name carries the very prefix the cold proof
        # treats as stale. Remove it here, before that proof and on every exit
        # path, so no payload blob outlives a failed cell and the proof covers
        # this cell's own state rather than tripping over it.
        shutil.rmtree(temporary_dir, ignore_errors=True)
        report["cell_scratch_removed"] = not temporary_dir.exists()

    require(
        bool(report.get("cell_scratch_removed")),
        "the cell scratch directory holding payload blobs survived cleanup",
    )
    report["cold_after"] = verify_cold_host(campaign, label=f"{cell_id}/after")
    results = dict(receiver.results) if receiver is not None else {}
    rows = join_records(
        payloads=payloads, edge_records=edge_records, results=results
    )
    for row in payloads["rows"]:
        row.pop("datagrams", None)
    print(f"[{cell_id}] controlled tail microbenchmark over {common.FRAMES} samples", flush=True)
    microbenchmark = tail_microbenchmark(
        edge=host_edge, payloads=payloads, profile=profile
    )
    attach_microbenchmark(rows, microbenchmark)
    report["microbenchmark"] = {
        "observations": len(microbenchmark["observations"]),
        "warmup_frames": microbenchmark["warmup_frames"],
        "counters_after": microbenchmark["counters_after"],
    }
    report["per_frame_rows"] = rows
    report["finished_at_unix_s"] = time.time()
    report["wall_seconds"] = report["finished_at_unix_s"] - started_at
    del payloads
    return report


# --------------------------------------------------------------------------
# statistics, evidence and reporting
# --------------------------------------------------------------------------

# The wall-clock stage groups that partition the frozen tail's service span.
STAGE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("camera_pose_reconstruct", ("camera_pose_reconstruct",)),
    ("tail_inference_block", ("decode_tail_launch", "finite_check_outputs")),
    (
        "camera_aware_postprocess",
        ("camera_aware_postprocess", "finite_check_postprocess"),
    ),
    ("p025_service_filter", ("p025_service_filter", "finite_check_p025")),
    ("segmentation_upsample_argmax", ("segmentation_upsample_argmax",)),
    ("compact_result_serialization", (SERIALIZE_STAGE,)),
)
GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
GROUP_LABELS = (
    "camera pose",
    "decode_tail + completion",
    "camera-aware postproc.",
    "p025 filter",
    "seg. upsample+argmax",
    "result serialization",
)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7975"
GRID = "#e5e4e0"

LIVE_METRICS: tuple[str, ...] = (
    "ue_send_loop_ms",
    "application_feature_uplink_ms",
    "post_send_to_reassembly_ms",
    "edge_first_to_complete_reassembly_ms",
    "edge_queue_wait_ms",
    "edge_zstd_decompression_ms",
    "edge_unpack_dequantize_ms",
    "edge_ae_decode_ms",
    "edge_frozen_tail_ms",
    "edge_output_serialization_ms",
    "edge_total_edge_processing_ms",
    "edge_service_wall_ms",
    "edge_result_finite_check_ms",
    "tail_camera_pose_reconstruct_ms",
    "tail_decode_tail_launch_ms",
    "tail_finite_check_outputs_ms",
    "tail_camera_aware_postprocess_ms",
    "tail_finite_check_postprocess_ms",
    "tail_p025_service_filter_ms",
    "tail_finite_check_p025_ms",
    "tail_segmentation_upsample_argmax_ms",
    "tail_output_serialization_ms",
    "tail_tail_call_total_ms",
    "decode_tail_cuda_ms",
    "decode_tail_inference_block_ms",
    "ue_round_trip_ms",
)
MICROBENCH_METRICS: tuple[str, ...] = tuple(
    f"mb_{name}"
    for name in (
        "edge_process_wall_ms",
        "cuda_settled_wall_ms",
        "edge_zstd_decompression_ms",
        "edge_unpack_dequantize_ms",
        "edge_ae_decode_ms",
        "edge_frozen_tail_ms",
        "edge_output_serialization_ms",
        "edge_total_edge_processing_ms",
        "tail_camera_pose_reconstruct_ms",
        "tail_decode_tail_launch_ms",
        "tail_finite_check_outputs_ms",
        "tail_camera_aware_postprocess_ms",
        "tail_finite_check_postprocess_ms",
        "tail_p025_service_filter_ms",
        "tail_finite_check_p025_ms",
        "tail_segmentation_upsample_argmax_ms",
        "tail_output_serialization_ms",
        "tail_tail_call_total_ms",
        "decode_tail_cuda_ms",
        "decode_tail_inference_block_ms",
    )
) + tuple(
    f"mb_tail_{stage}_cuda_ms"
    for stage in (*TAIL_STAGES, SERIALIZE_STAGE, TOTAL_STAGE)
)


def _group_totals(row: Mapping[str, Any], prefix: str) -> dict[str, float | None]:
    totals: dict[str, float | None] = {}
    for name, stages in STAGE_GROUPS:
        values = [row.get(f"{prefix}tail_{stage}_ms") for stage in stages]
        totals[name] = (
            sum(float(value) for value in values)
            if all(value is not None for value in values)
            else None
        )
    return totals


def summarize_action(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = report["per_frame_rows"]
    sent = [row for row in rows if row["send_status"] == "SENT"]
    delivered = [row for row in rows if row.get("delivered")]
    edge_summary = report.get("edge_summary", {})
    counters = dict(edge_summary.get("counters", {}))
    metrics: dict[str, Any] = {}
    for name in LIVE_METRICS:
        metrics[name] = common.summarize(
            [row[name] for row in rows if row.get(name) is not None]
        )
    for name in MICROBENCH_METRICS:
        metrics[name] = common.summarize(
            [row[name] for row in rows if row.get(name) is not None]
        )
    live_groups = {
        name: common.summarize(
            [
                value
                for row in delivered
                if (value := _group_totals(row, "")[name]) is not None
            ]
        )
        for name, _stages in STAGE_GROUPS
    }
    controlled_groups = {
        name: common.summarize(
            [
                value
                for row in rows
                if (value := _group_totals(row, "mb_")[name]) is not None
            ]
        )
        for name, _stages in STAGE_GROUPS
    }
    controlled_total = sum(
        float(value["median"])
        for value in controlled_groups.values()
        if value["median"] is not None
    )
    return {
        "cell_id": report["cell_id"],
        "action_id": int(report["action_id"]),
        "profile_id": report["profile_id"],
        "network_profile_id": report["network_profile_id"],
        "wall_seconds": report["wall_seconds"],
        "counts": {
            "frames_attempted": len(rows),
            "messages_sent": len(sent),
            "obsolete_skipped": int(report["transmission"]["obsolete_skipped"]),
            "feature_datagrams_transmitted": int(
                report["transmission"]["feature_datagrams_transmitted"]
            ),
            "feature_datagrams_received_edge": int(
                counters.get("feature_datagrams_received", 0)
            ),
            "complete_reassemblies": int(counters.get("feature_messages_reassembled", 0)),
            "incomplete_reassemblies_expired": int(
                edge_summary.get("incomplete_reassemblies_expired", 0)
            ),
            "duplicate_datagrams": int(counters.get("feature_datagrams_duplicate", 0)),
            "queue_admissions": int(counters.get("edge_queue_admissions", 0)),
            "queue_replacements": int(counters.get("edge_pending_replacements", 0)),
            "queue_admission_refused_not_freshest": int(
                counters.get("edge_admission_refused_not_freshest", 0)
            ),
            "edge_process_starts": int(counters.get("edge_process_starts", 0)),
            "tail_completions": int(counters.get("tail_completions", 0)),
            "compact_results_transmitted": int(
                counters.get("compact_results_transmitted", 0)
            ),
            "compact_results_returned_to_ue": int(
                report.get("result_downlink", {}).get("compact_results_returned", 0)
            ),
            "delivered_and_measured": len(delivered),
            "service_target_met": sum(
                1 for row in delivered if row.get("service_target_met")
            ),
            "processing_horizon_met": sum(
                1 for row in delivered if row.get("processing_horizon_met")
            ),
            "microbenchmark_observations": int(report["microbenchmark"]["observations"]),
        },
        "payload": {
            "scientific_inner_payload_bytes": common.summarize(
                [row["inner_payload_bytes"] for row in rows]
            ),
            "sfd1_overhead_bytes": int(rows[0]["sfd1_overhead_bytes"]) if rows else None,
            "sfd1_total_bytes": common.summarize([row["sfd1_bytes"] for row in rows]),
            "udp_application_bytes": common.summarize(
                [row["udp_application_bytes"] for row in rows]
            ),
            "datagrams_per_message": common.summarize(
                [row["datagram_count"] for row in rows]
            ),
        },
        "timing": metrics,
        "live_stage_groups_ms": live_groups,
        "controlled_stage_groups_ms": controlled_groups,
        "controlled_stage_group_total_median_ms": controlled_total,
        "radio_actuation": report.get("radio_actuation", {}),
        "radio_telemetry": report.get("radio_telemetry", {}),
        "clock_domain": report.get("clock_domain", {}),
        "edge_deadline_policy": edge_summary.get("deadline_policy"),
        "parent_equivalence": edge_summary.get("parent_equivalence"),
    }


PER_FRAME_FIELDS: tuple[str, ...] = (
    "action_id", "profile_id", "cell_id", "network_profile_id",
    "ordinal", "sequence_id", "frame_id", "sample_id", "episode_id",
    "registered_frame_id", "dataset_index",
    "inner_payload_bytes", "sfd1_overhead_bytes", "sfd1_bytes",
    "datagram_count", "udp_application_bytes", "estimated_wire_bytes",
    "capture_timestamp_ns", "scheduled_send_wall_ns",
    "ue_inner_payload_ready_wall_ns", "ue_payload_ready_wall_ns",
    "send_status", "slot_lateness_ms",
    "ue_first_send_wall_ns", "ue_final_send_wall_ns", "ue_prepare_total_ms",
    "delivered", "result_returned",
    "edge_first_datagram_wall_ns", "edge_complete_reassembly_wall_ns",
    "edge_admitted_wall_ns", "edge_worker_start_wall_ns",
    "edge_tail_finished_wall_ns", "ue_result_received_wall_ns",
    "edge_feature_datagrams", "edge_duplicate_datagrams",
    "service_record_count", "service_record_bytes",
    "service_target_met", "processing_horizon_met",
    *LIVE_METRICS,
    "mb_service_record_count", "mb_service_record_bytes",
    *MICROBENCH_METRICS,
)


def write_per_frame_csv(path: Path, report: Mapping[str, Any]) -> str:
    rows = report["per_frame_rows"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_FRAME_FIELDS))
        writer.writeheader()
        for row in rows:
            enriched = {
                **row,
                "action_id": int(report["action_id"]),
                "profile_id": report["profile_id"],
                "cell_id": report["cell_id"],
                "network_profile_id": report["network_profile_id"],
            }
            writer.writerow(
                {field: enriched.get(field, "") for field in PER_FRAME_FIELDS}
            )
    return common.sha256_file(path)


def _flatten(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": ("" if value.get(key) is None else value.get(key))
        for key in ("count", "median", "p90", "p95", "minimum", "maximum")
    }


def write_action_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row: dict[str, Any] = {
            "action_id": summary["action_id"],
            "profile_id": summary["profile_id"],
            "cell_id": summary["cell_id"],
            "network_profile_id": summary["network_profile_id"],
            "wall_seconds": round(float(summary["wall_seconds"]), 3),
        }
        row.update({f"count_{key}": value for key, value in summary["counts"].items()})
        for name, value in summary["payload"].items():
            if isinstance(value, Mapping):
                row.update(_flatten(f"payload_{name}", value))
            else:
                row[f"payload_{name}"] = "" if value is None else value
        for name, value in summary["timing"].items():
            row.update(_flatten(name, value))
        for name, value in summary["live_stage_groups_ms"].items():
            row.update(_flatten(f"live_group_{name}", value))
        for name, value in summary["controlled_stage_groups_ms"].items():
            row.update(_flatten(f"controlled_group_{name}", value))
        telemetry = summary.get("radio_telemetry", {})
        row["radio_telemetry_status"] = telemetry.get("status", "")
        for name in (
            "achieved_pusch_snr_db", "achieved_pusch_mcs",
            "scheduler_avg_snr_db", "scheduler_selected_ul_mcs",
            "scheduler_final_ul_mcs",
        ):
            if isinstance(telemetry.get(name), Mapping):
                row.update(_flatten(f"radio_{name}", telemetry[name]))
        actuation = summary.get("radio_actuation", {})
        row["actuated_trace_id"] = actuation.get("trace_id", "")
        row["actuated_commands_applied"] = actuation.get("commands_applied", "")
        row["actuated_first_300_target_digest"] = actuation.get(
            "first_300_target_digest", ""
        )
        for name in ("target_snr_db", "first_300_target_snr_db", "command_latency_ms"):
            if isinstance(actuation.get(name), Mapping):
                row.update(_flatten(f"actuated_{name}", actuation[name]))
        rows.append(row)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return common.sha256_file(path)


def build_comparisons(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The five comparisons the diagnostic was commissioned to answer."""

    def median(summary: Mapping[str, Any], name: str) -> float | None:
        value = summary["timing"].get(name, {})
        return None if value.get("median") is None else float(value["median"])

    per_action = []
    for summary in summaries:
        controlled_total = float(summary["controlled_stage_group_total_median_ms"])
        cuda = median(summary, "mb_decode_tail_cuda_ms")
        groups = {
            name: (
                None
                if summary["controlled_stage_groups_ms"][name]["median"] is None
                else float(summary["controlled_stage_groups_ms"][name]["median"])
            )
            for name, _stages in STAGE_GROUPS
        }
        inference_block = groups["tail_inference_block"]
        non_inference = (
            None
            if inference_block is None
            else controlled_total - inference_block
        )
        per_action.append(
            {
                "action_id": summary["action_id"],
                "profile_id": summary["profile_id"],
                "payload_median_bytes": summary["payload"][
                    "scientific_inner_payload_bytes"
                ]["median"],
                "datagrams_per_message_median": summary["payload"][
                    "datagrams_per_message"
                ]["median"],
                "decode_tail_cuda_ms_median": cuda,
                "controlled_tail_service_span_ms_median": controlled_total,
                "controlled_stage_group_medians_ms": groups,
                "non_inference_overhead_ms_median": non_inference,
                "non_inference_overhead_fraction": (
                    None
                    if non_inference is None or controlled_total <= 0
                    else non_inference / controlled_total
                ),
                "delta_vs_phase13c_controlled_tail_ms": (
                    None if cuda is None else cuda - common.PHASE13C_CONTROLLED_TAIL_MS
                ),
                "delta_vs_phase15_live_frozen_tail_ms": (
                    None if cuda is None else cuda - common.PHASE15_LIVE_FROZEN_TAIL_MS
                ),
                "live_frozen_tail_ms_median": median(summary, "edge_frozen_tail_ms"),
                "live_decode_tail_cuda_ms_median": median(summary, "decode_tail_cuda_ms"),
                "application_feature_uplink_ms": summary["timing"][
                    "application_feature_uplink_ms"
                ],
                "ue_send_loop_ms": summary["timing"]["ue_send_loop_ms"],
                "edge_first_to_complete_reassembly_ms": summary["timing"][
                    "edge_first_to_complete_reassembly_ms"
                ],
                "edge_queue_wait_ms": summary["timing"]["edge_queue_wait_ms"],
                "edge_service_wall_ms": summary["timing"]["edge_service_wall_ms"],
                "complete_reassemblies": summary["counts"]["complete_reassemblies"],
                "messages_sent": summary["counts"]["messages_sent"],
            }
        )

    queue_pairs = [
        (row["edge_queue_wait_ms"]["median"], row["edge_service_wall_ms"]["median"])
        for row in per_action
        if row["edge_queue_wait_ms"]["median"] is not None
        and row["edge_service_wall_ms"]["median"] is not None
    ]
    queue_relationship: dict[str, Any] = {
        "paired_actions": len(queue_pairs),
        "queue_wait_ms_medians": [pair[0] for pair in queue_pairs],
        "edge_service_wall_ms_medians": [pair[1] for pair in queue_pairs],
    }
    if len(queue_pairs) >= 2:
        ratios = [
            wait / service for wait, service in queue_pairs if service > 0
        ]
        queue_relationship["queue_wait_over_service_ratio"] = ratios
        queue_relationship["queue_wait_tracks_service_time"] = bool(
            ratios and min(ratios) >= 0.5 and max(ratios) <= 2.0
        )
    else:
        queue_relationship["queue_wait_tracks_service_time"] = None

    uplink_versus_payload = sorted(
        (
            {
                "action_id": row["action_id"],
                "profile_id": row["profile_id"],
                "payload_median_bytes": row["payload_median_bytes"],
                "datagrams_per_message_median": row["datagrams_per_message_median"],
                "messages_sent": row["messages_sent"],
                "complete_reassemblies": row["complete_reassemblies"],
                "complete_reassembly_fraction": (
                    None
                    if not row["messages_sent"]
                    else row["complete_reassemblies"] / row["messages_sent"]
                ),
                "application_feature_uplink_ms_median": row[
                    "application_feature_uplink_ms"
                ]["median"],
                "application_feature_uplink_ms_p95": row[
                    "application_feature_uplink_ms"
                ]["p95"],
                "ue_send_loop_ms_median": row["ue_send_loop_ms"]["median"],
            }
            for row in per_action
        ),
        key=lambda item: -(item["payload_median_bytes"] or 0),
    )
    return {
        "reference_points": {
            "phase13c_controlled_tail_gpu_ms_median": common.PHASE13C_CONTROLLED_TAIL_MS,
            "phase13c_scope": (
                "CUDA-event span over the whole ContextualFrozenP025TailAdapter call "
                "(pose + decode_tail + finite checks + postprocess + p025 + "
                "segmentation), 36 profiles x 300 frames, localhost, no OAI"
            ),
            "phase15_live_frozen_tail_ms_median": common.PHASE15_LIVE_FROZEN_TAIL_MS,
            "phase15_scope": (
                "monotonic wall span of the edge runtime's frozen_tail stage over "
                "the same adapter call, live CARLA/OAI Route-B cells"
            ),
        },
        "per_action": per_action,
        "uplink_versus_payload": uplink_versus_payload,
        "queue_wait_versus_tail_service": queue_relationship,
    }


def write_figure(
    base_path: Path, summaries: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Any]
) -> dict[str, str]:
    """Two panels: the controlled tail decomposition and uplink versus payload.

    Static print artifact for the report, so there is no hover layer; the
    relief obligation for the low-contrast categorical slots is met by the
    direct segment labels here and by the full table view in
    ``action_summary.csv`` and ``REPORT.md``.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ordered = sorted(summaries, key=lambda item: -(
        item["payload"]["scientific_inner_payload_bytes"]["median"] or 0
    ))
    labels = [
        f"a{item['action_id']:02d}  {item['profile_id'].replace('split_', '')}"
        for item in ordered
    ]
    positions = list(range(len(ordered)))[::-1]

    figure, (upper, lower) = plt.subplots(
        2, 1, figsize=(12.0, 9.2), gridspec_kw={"height_ratios": [1.2, 1.0]}
    )
    figure.patch.set_facecolor(SURFACE)
    for axis in (upper, lower):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3, width=0.8)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)

    # -- panel A: controlled tail service span, decomposed -------------------
    gap_ms = 0.0
    totals = [
        float(item["controlled_stage_group_total_median_ms"]) for item in ordered
    ]
    span = max(totals) if totals else 1.0
    gap_ms = span * 0.004  # ~2 px surface gap between stacked segments
    for index, (name, _stages) in enumerate(STAGE_GROUPS):
        left = []
        widths = []
        for item in ordered:
            offset = sum(
                float(item["controlled_stage_groups_ms"][earlier]["median"] or 0.0)
                + gap_ms
                for earlier, _s in STAGE_GROUPS[:index]
            )
            value = float(item["controlled_stage_groups_ms"][name]["median"] or 0.0)
            left.append(offset)
            widths.append(value)
        bars = upper.barh(
            positions,
            widths,
            left=left,
            height=0.52,
            color=GROUP_COLORS[index],
            edgecolor=SURFACE,
            linewidth=0.0,
            label=GROUP_LABELS[index],
        )
        for bar, value in zip(bars, widths):
            if value >= span * 0.055:
                upper.text(
                    bar.get_x() + value / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.1f}",
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if index in (0, 5) else TEXT_PRIMARY,
                )
    for position, item, total in zip(positions, ordered, totals):
        cuda = item["timing"]["mb_decode_tail_cuda_ms"]["median"]
        annotation = f"span {total:.1f} ms"
        if cuda is not None:
            annotation += f"   ·   pure decode_tail CUDA {float(cuda):.1f} ms"
        upper.text(
            total + span * 0.025,
            position,
            annotation,
            ha="left", va="center", fontsize=8, color=TEXT_SECONDARY,
        )
    for reference, text in (
        (common.PHASE13C_CONTROLLED_TAIL_MS, "Phase-13C tail_gpu 73.4"),
        (common.PHASE15_LIVE_FROZEN_TAIL_MS, "Phase-15 live frozen_tail 111.6"),
    ):
        upper.axvline(reference, color=TEXT_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        upper.text(
            reference,
            len(ordered) - 0.42,
            f"  {text}",
            fontsize=8, color=TEXT_MUTED, ha="left", va="bottom",
        )
    upper.set_yticks(positions)
    upper.set_yticklabels(labels, fontsize=9, color=TEXT_PRIMARY)
    upper.set_xlim(0, span * 1.62)
    upper.set_ylim(-0.55, len(ordered) - 0.2)
    upper.set_xlabel("milliseconds (median of 300 controlled observations)", fontsize=9,
                     color=TEXT_SECONDARY)
    upper.set_title(
        "Frozen p025 tail service span, decomposed  ·  RTX 5090, resident models",
        fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10,
    )
    upper.legend(
        handles=[
            Patch(facecolor=GROUP_COLORS[index], edgecolor=SURFACE, label=GROUP_LABELS[index])
            for index in range(len(STAGE_GROUPS))
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False,
        fontsize=8.5, ncol=3, labelcolor=TEXT_SECONDARY, handlelength=1.4,
        columnspacing=1.6,
    )

    # -- panel B: application feature uplink versus payload ------------------
    uplink = comparisons["uplink_versus_payload"]
    medians = [row["application_feature_uplink_ms_median"] for row in uplink]
    p95s = [row["application_feature_uplink_ms_p95"] for row in uplink]
    finite = [value for value in medians if value is not None]
    scale = max([*finite, *[value for value in p95s if value is not None], 1.0])
    positions_b = list(range(len(uplink)))[::-1]
    labels_b = [
        f"a{row['action_id']:02d}  {(row['payload_median_bytes'] or 0) / 1024:,.0f} KiB"
        f"  ·  {int(row['datagrams_per_message_median'] or 0)} dgram"
        for row in uplink
    ]
    lower.barh(
        positions_b,
        [value if value is not None else 0.0 for value in medians],
        height=0.46,
        color=GROUP_COLORS[0],
        edgecolor=SURFACE,
        linewidth=0.0,
    )
    for position, row, value, p95 in zip(positions_b, uplink, medians, p95s):
        if value is None:
            lower.text(
                scale * 0.01, position,
                "no complete reassembly arrived  ·  uplink-limited",
                ha="left", va="center", fontsize=9, color="#e34948",
            )
            continue
        lower.text(
            value + scale * 0.014, position, f"{value:,.0f} ms",
            ha="left", va="center", fontsize=9, color=TEXT_PRIMARY,
        )
        if p95 is not None:
            lower.plot(
                [p95], [position], marker="o", markersize=6.5,
                markerfacecolor=SURFACE, markeredgecolor=GROUP_COLORS[0],
                markeredgewidth=2.0, linestyle="none",
            )
            lower.text(
                p95 + scale * 0.014, position - 0.29, f"p95 {p95:,.0f}",
                ha="left", va="center", fontsize=8, color=TEXT_SECONDARY,
            )
    lower.set_yticks(positions_b)
    lower.set_yticklabels(labels_b, fontsize=9, color=TEXT_PRIMARY)
    lower.set_xlim(0, scale * 1.34)
    lower.set_ylim(-0.6, len(uplink) - 0.3)
    lower.set_xlabel(
        "milliseconds, UE first datagram send -> edge complete message reassembly",
        fontsize=9, color=TEXT_SECONDARY,
    )
    lower.set_title(
        "Application-level feature-uplink handling through live OAI, by payload"
        "  ·  hollow dot = p95",
        fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10,
    )
    figure.tight_layout(pad=1.4, h_pad=4.2)
    digests: dict[str, str] = {}
    for suffix in ("pdf", "png"):
        path = base_path.with_suffix(f".{suffix}")
        options: dict[str, Any] = {
            "format": suffix, "facecolor": SURFACE, "bbox_inches": "tight",
        }
        if suffix == "png":
            options["dpi"] = 200
        figure.savefig(path, **options)
        digests[path.name] = common.sha256_file(path)
    plt.close(figure)
    return digests


def _ms(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _stage_table(summary: Mapping[str, Any], names: Sequence[tuple[str, str]]) -> str:
    lines = [
        "| stage | n | median | p90 | p95 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in names:
        value = summary["timing"].get(key, {})
        lines.append(
            f"| {label} | {value.get('count', 0)} | {_ms(value.get('median'))} | "
            f"{_ms(value.get('p90'))} | {_ms(value.get('p95'))} | "
            f"{_ms(value.get('minimum'))} | {_ms(value.get('maximum'))} |"
        )
    return "\n".join(lines)


LIVE_TABLE = (
    ("ue_send_loop_ms", "UE send loop (first->final datagram)"),
    ("application_feature_uplink_ms", "application feature uplink (UE first send->edge reassembled)"),
    ("post_send_to_reassembly_ms", "post-send to reassembly (UE final send->edge reassembled)"),
    ("edge_first_to_complete_reassembly_ms", "edge first datagram->complete reassembly"),
    ("edge_queue_wait_ms", "edge queue wait (reassembled->worker start)"),
    ("edge_zstd_decompression_ms", "zstd decompression"),
    ("edge_unpack_dequantize_ms", "unpack / dequantization"),
    ("edge_ae_decode_ms", "AE decode"),
    ("edge_frozen_tail_ms", "frozen_tail stage (deployed definition)"),
    ("decode_tail_cuda_ms", "pure decode_tail CUDA"),
    ("edge_output_serialization_ms", "compact result serialization"),
    ("edge_total_edge_processing_ms", "total edge processing"),
    ("edge_service_wall_ms", "complete edge-service wall time"),
    ("ue_round_trip_ms", "UE round trip (final send->compact result)"),
)
CONTROLLED_TABLE = (
    ("mb_tail_camera_pose_reconstruct_ms", "camera-pose / calibration reconstruction"),
    ("mb_tail_decode_tail_launch_ms", "decode_tail launch (CPU)"),
    ("mb_decode_tail_cuda_ms", "decode_tail CUDA (device, pure)"),
    ("mb_tail_finite_check_outputs_ms", "output finite check (absorbs tail completion)"),
    ("mb_decode_tail_inference_block_ms", "decode_tail inference block (launch + completion)"),
    ("mb_tail_camera_aware_postprocess_ms", "camera-aware post-processing"),
    ("mb_tail_finite_check_postprocess_ms", "post-processing finite check"),
    ("mb_tail_p025_service_filter_ms", "p025 service filtering"),
    ("mb_tail_finite_check_p025_ms", "p025 finite check"),
    ("mb_tail_segmentation_upsample_argmax_ms", "720x1280 segmentation interpolate + argmax"),
    ("mb_tail_output_serialization_ms", "compact object-result serialization"),
    ("mb_tail_tail_call_total_ms", "tail adapter call total"),
    ("mb_edge_zstd_decompression_ms", "zstd decompression"),
    ("mb_edge_unpack_dequantize_ms", "unpack / dequantization"),
    ("mb_edge_ae_decode_ms", "AE decode"),
    ("mb_edge_total_edge_processing_ms", "total edge processing"),
    ("mb_edge_process_wall_ms", "complete edge-service wall time"),
)


def write_report(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    runtime_seconds: float,
) -> str:
    lines: list[str] = []
    lines.append("# SplitFusion timing diagnostic v1 — measured decomposition")
    lines.append("")
    lines.append(
        f"Run `{manifest['run_id']}` · implementation commit "
        f"`{manifest['git']['head']}` · {runtime_seconds / 60.0:.1f} min wall · "
        f"{manifest['environment']['device_name']}."
    )
    lines.append("")
    lines.append(
        "Four catalog actions, four fresh OAI lifecycles, one four-action execution. "
        "Sample: the registered Phase-13C fit-only 300-frame sample "
        f"(`sample_manifest_sha256={manifest['sample']['sample_manifest_sha256']}`), "
        "identical for every action. Radio: FAVORABLE_STABLE on the qualified "
        "100 MHz / 273 PRB / 4D5U profile at the registered 100 ms schedule, "
        "inside the frozen 4200-sample prefix."
    )
    lines.append("")
    lines.append("## What each number is, and is not")
    lines.append("")
    lines.append(
        "- **`application_feature_uplink_ms` = edge complete reassembly − UE first "
        "datagram send.** This is *application-level feature-uplink handling "
        "through OAI*, **not** PHY/RLC latency. It includes the host UDP send "
        "path, the UE tunnel, the 5G core and RAN transport, edge socket "
        "receipt, IP/UDP fragmentation and application message reassembly."
    )
    lines.append(
        "- The earlier round-trip residual is **not** reused or relabelled "
        "anywhere. Every uplink quantity here is a one-way difference between "
        "two same-host `time.time_ns()` boundaries."
    )
    lines.append(
        "- `decode_tail_cuda_ms` is a dedicated CUDA event pair with nothing but "
        "`model.decode_tail(batch, dense=False)` between the two records; camera "
        "pose, finite checks, post-processing, p025 filtering, segmentation "
        "construction and serialization are all measured separately."
    )
    lines.append("")
    clock = summaries[0].get("clock_domain", {})
    lines.append(
        "Clock domain: UE host and edge container were verified to share one "
        "wall-clock domain from paired `time.time_ns()` / `time.monotonic_ns()` "
        "anchors at both ends of both processes (worst wall−monotonic offset skew "
        f"{clock.get('maximum_wall_minus_monotonic_skew_ns', 'n/a')} ns). No "
        "derived interval in any artifact is negative."
    )
    lines.append("")

    lines.append("## Commissioned comparisons")
    lines.append("")
    reference = comparisons["reference_points"]
    lines.append(
        f"**1 & 2 — pure `decode_tail` CUDA time versus the two published spans.** "
        f"The ~{reference['phase13c_controlled_tail_gpu_ms_median']} ms Phase-13C "
        f"`tail_gpu_ms` and the ~{reference['phase15_live_frozen_tail_ms_median']} ms "
        "Phase-15 live `frozen_tail` are both spans over the *whole* tail adapter "
        "call, not over `decode_tail`."
    )
    lines.append("")
    lines.append(
        "| action | profile | pure decode_tail CUDA (ms) | controlled tail span (ms) "
        "| Δ vs 73.4 | Δ vs 111.6 | live frozen_tail (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in comparisons["per_action"]:
        lines.append(
            f"| {row['action_id']} | `{row['profile_id']}` | "
            f"{_ms(row['decode_tail_cuda_ms_median'])} | "
            f"{_ms(row['controlled_tail_service_span_ms_median'])} | "
            f"{_ms(row['delta_vs_phase13c_controlled_tail_ms'])} | "
            f"{_ms(row['delta_vs_phase15_live_frozen_tail_ms'])} | "
            f"{_ms(row['live_frozen_tail_ms_median'])} |"
        )
    lines.append("")
    lines.append(
        "**3 — where the difference goes.** Wall-clock stage groups partition the "
        "controlled tail service span (medians, ms):"
    )
    lines.append("")
    header = "| action | " + " | ".join(name for name, _s in STAGE_GROUPS) + " | span | non-inference share |"
    lines.append(header)
    lines.append("|---" * (len(STAGE_GROUPS) + 3) + "|")
    for row in comparisons["per_action"]:
        cells = " | ".join(
            _ms(row["controlled_stage_group_medians_ms"][name])
            for name, _s in STAGE_GROUPS
        )
        share = row["non_inference_overhead_fraction"]
        lines.append(
            f"| {row['action_id']} | {cells} | "
            f"{_ms(row['controlled_tail_service_span_ms_median'])} | "
            + ("n/a" if share is None else f"{share * 100:.1f}%")
            + " |"
        )
    lines.append("")
    lines.append(
        "**4 — application uplink latency versus payload** (descending payload):"
    )
    lines.append("")
    lines.append(
        "| action | payload (B) | datagrams/msg | sent | complete reassemblies | "
        "complete fraction | uplink median (ms) | uplink p95 (ms) | UE send loop median (ms) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in comparisons["uplink_versus_payload"]:
        fraction = row["complete_reassembly_fraction"]
        lines.append(
            f"| {row['action_id']} | {int(row['payload_median_bytes'] or 0):,} | "
            f"{int(row['datagrams_per_message_median'] or 0)} | {row['messages_sent']} | "
            f"{row['complete_reassemblies']} | "
            + ("n/a" if fraction is None else f"{fraction * 100:.1f}%")
            + f" | {_ms(row['application_feature_uplink_ms_median'], 1)} | "
            f"{_ms(row['application_feature_uplink_ms_p95'], 1)} | "
            f"{_ms(row['ue_send_loop_ms_median'])} |"
        )
    lines.append("")
    queue = comparisons["queue_wait_versus_tail_service"]
    lines.append(
        "**5 — does edge queue wait track tail service time?** "
        f"Paired actions: {queue['paired_actions']}. Queue-wait medians "
        f"{[None if v is None else round(float(v), 2) for v in queue['queue_wait_ms_medians']]} ms "
        f"against edge-service medians "
        f"{[None if v is None else round(float(v), 2) for v in queue['edge_service_wall_ms_medians']]} ms. "
        f"Tracks service time: {queue['queue_wait_tracks_service_time']}."
    )
    lines.append("")

    for summary in summaries:
        lines.append(f"## Action {summary['action_id']} — `{summary['profile_id']}`")
        lines.append("")
        counts = summary["counts"]
        payload = summary["payload"]
        lines.append(
            f"Scientific payload "
            f"{int(payload['scientific_inner_payload_bytes']['median'] or 0):,} B "
            f"median (SFD1 overhead {payload['sfd1_overhead_bytes']} B), "
            f"{int(payload['datagrams_per_message']['median'] or 0)} datagrams/message. "
            f"Sent {counts['messages_sent']}/{counts['frames_attempted']} "
            f"({counts['obsolete_skipped']} obsolete slot skips). Edge: "
            f"{counts['feature_datagrams_received_edge']:,} datagrams received, "
            f"{counts['complete_reassemblies']} complete reassemblies, "
            f"{counts['incomplete_reassemblies_expired']} incomplete expiries, "
            f"{counts['queue_admissions']} queue admissions "
            f"({counts['queue_replacements']} replacements), "
            f"{counts['tail_completions']} tail completions, "
            f"{counts['compact_results_transmitted']} compact results transmitted "
            f"({counts['compact_results_returned_to_ue']} returned to the UE). "
            f"Controlled microbenchmark observations: "
            f"{counts['microbenchmark_observations']}."
        )
        lines.append("")
        telemetry = summary.get("radio_telemetry", {})
        actuation = summary.get("radio_actuation", {})
        snr = telemetry.get("achieved_pusch_snr_db", {})
        mcs = telemetry.get("scheduler_final_ul_mcs", {})
        lines.append(
            f"Radio: target SNR median "
            f"{_ms(actuation.get('first_300_target_snr_db', {}).get('median'))} dB over "
            f"the identical frozen first-300 prefix "
            f"(digest `{str(actuation.get('first_300_target_digest', ''))[:16]}`), "
            f"{actuation.get('commands_applied', 'n/a')} commands applied on the "
            f"100 ms schedule, "
            f"clean −50 dB restore verified. Achieved telemetry status "
            f"`{telemetry.get('status', 'n/a')}`"
            + (
                f": PUSCH SNR median {_ms(snr.get('median'))} dB "
                f"({snr.get('count', 0)} samples), final UL MCS median "
                f"{_ms(mcs.get('median'), 1)} ({mcs.get('count', 0)} samples)."
                if telemetry.get("status") == "COLLECTED"
                else f" (error: {telemetry.get('error', '') or 'no samples'})."
            )
        )
        lines.append("")
        lines.append("### Live path through OAI")
        lines.append("")
        lines.append(_stage_table(summary, LIVE_TABLE))
        lines.append("")
        lines.append("### Controlled local decomposition (300 observations)")
        lines.append("")
        lines.append(_stage_table(summary, CONTROLLED_TABLE))
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Replay capture semantics.** Frames come from the registered fit-only "
        "sample, not a live sensor, so `capture_timestamp_ns` is the *planned "
        "replay offer instant* on the 10 Hz schedule. The 100 ms service target "
        "and 500 ms processing horizon are therefore measured from the offer, not "
        "from a CARLA sensor capture. No preparation-coverage or age-of-information "
        "claim can be read off this run."
    )
    lines.append(
        "- **The diagnostic edge classifies deadlines instead of dropping.** The "
        "deployed edge refuses a late frame at four gates; dropping would delete "
        "the very stage decomposition being measured, so each frame carries "
        "`service_target_met` / `processing_horizon_met` and is still processed. "
        "Model, codec, threshold, action and wire bytes are unchanged, and the "
        "instrumented tail was proved bit-identical to the production tail before "
        "any measurement was recorded."
    )
    lines.append(
        "- **Per-stage CUDA and wall times overlap by construction.** "
        "`decode_tail_cuda_ms` is device time for the tail kernels; most of that "
        "same time appears as wall time in the immediately following finite check, "
        "which is where the deployed path first synchronizes. The additive "
        "wall-clock partition is the stage-group table; the CUDA columns are the "
        "device-side attribution of the same work."
    )
    lines.append(
        "- **No optimization was attempted.** This run only measures. It carries "
        "no CARLA, no Route B, no training, tuning, holdout or test evaluation, "
        "no 16-cell pilot and no 288-cell campaign."
    )
    lines.append(
        "- Single execution per action; the medians are within-run, so no "
        "run-to-run variance is characterized."
    )
    if any(
        summary.get("radio_telemetry", {}).get("status") != "COLLECTED"
        for summary in summaries
    ):
        lines.append(
            "- Achieved SNR/MCS telemetry did not collect for at least one action; "
            "the commanded target-SNR trace is still fully recorded and verified "
            "for every action."
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(
        "`RUN_MANIFEST.json` (immutable pre-run binding), "
        "`per_frame/action_*.csv` (one per action, 300 rows), "
        "`action_summary.csv`, `DIAGNOSTIC_RESULTS.json`, `REPORT.md`, "
        "`timing_breakdown.pdf`, `timing_breakdown.png`, "
        "`ARTIFACT_MANIFEST.json`, terminal file. No RGB or radar frame, C2 "
        "tensor, compressed payload blob, prediction or raw OAI tracer log is "
        "retained."
    )
    lines.append("")
    return common.atomic_create_text(path, "\n".join(lines) + "\n")


def build_manifest(
    *,
    run_id: str,
    output: Path,
    git_state: Mapping[str, Any],
    bindings: Mapping[str, Any],
    cuda: Mapping[str, Any],
    container: Mapping[str, Any],
    cold: Mapping[str, Any],
    actions: Mapping[str, Any],
    sample: Mapping[str, Any],
    campaign: Mapping[str, Any],
    synthetic_tests: Mapping[str, Any],
) -> dict[str, Any]:
    return common.seal(
        {
            "schema": common.MANIFEST_SCHEMA,
            "run_id": run_id,
            "execution_token": EXECUTE_TOKEN,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "objective": (
                "Decompose, without conflation, application-level one-way feature "
                "uplink handling through the live OAI 5G path and every edge stage "
                "from datagram receipt through compact result serialization, "
                "including pure model.decode_tail CUDA time."
            ),
            "scope": {
                "actions": [action_id for action_id, _profile in common.DIAGNOSTIC_ACTIONS],
                "frames_per_action": common.FRAMES,
                "network_profile_id": common.NETWORK_PROFILE_ID,
                "radio_lifecycles": len(common.DIAGNOSTIC_ACTIONS),
                "schedule_period_ms": common.SCHEDULE_PERIOD_NS // 1_000_000,
                "carla_launched": False,
                "route_b": False,
                "training_or_tuning": False,
                "holdout_or_test_frames_read": False,
                "sixteen_cell_pilot": False,
                "two_hundred_eighty_eight_cell_campaign": False,
                "dataset_access": "fit_only_registered_phase13c_300_frame_sample",
                "optimization_attempted": False,
            },
            "output_relpath": str(output.relative_to(ROOT)),
            "git": git_state,
            "bindings": bindings,
            "environment": {
                **cuda,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "hostname": platform.node(),
                **container,
            },
            "cold_host_preflight": cold,
            "action_binding": actions,
            "sample": {
                "schema": sample["schema"],
                "sample_manifest_sha256": sample["sample_manifest_sha256"],
                "selected_sample_id_sha256": sample["selected_sample_id_sha256"],
                "selected_frame_count": int(sample["selected_frame_count"]),
                "registered_split": "fit",
                "phase13c_evidence_relpath": common.PHASE13C_EVIDENCE_RELPATH,
                "access_scope": sample["access_scope"],
                "warmup_sample_id": sample["warmup"]["sample_id"],
                "warmup_excluded_from_measurement": True,
            },
            "transport_contract": {
                "sfd1_protocol_version": int(campaign["runtime"]["sfd1_protocol_version"]),
                "udp_fragment_header": campaign["runtime"]["udp_fragment_header"],
                "udp_chunk_bytes": int(campaign["runtime"]["udp_chunk_bytes"]),
                "socket_buffer_request_bytes": int(
                    campaign["runtime"]["socket_buffer_request_bytes"]
                ),
                "retransmission": bool(campaign["runtime"]["retransmission"]),
                "edge_receive_port": int(campaign["runtime"]["edge_receive_port"]),
                "camera_result_port": int(campaign["runtime"]["camera_result_port"]),
                "ue_bind_host": campaign["runtime"]["ue_bind_host"],
                "edge_remote_host": campaign["runtime"]["edge_remote_host"],
                "entropy_coder": "zstd",
                "inner_bytes_modified": False,
            },
            "radio_contract": {
                "radio_profile_id": "OAI_N78_100MHZ_273PRB_4D5U_V1",
                "launcher": campaign["runtime"]["oai_registered_profile_launcher"],
                "launcher_status": campaign["runtime"][
                    "oai_registered_profile_launcher_status"
                ],
                "cn5g_owner": "qualified_launcher",
                "clean_restore_noise_power_db": common.CLEAN_NOISE_POWER_DB,
                "mapping_csv": campaign["network"]["mapping_csv"],
                "mapping_sha256": campaign["network"]["mapping_sha256"],
                "prefix_samples": int(campaign["network"]["prefix_samples"]),
                "forbidden_after_prefix": list(
                    campaign["network"]["forbidden_after_prefix"]
                ),
                "catch_up_policy": campaign["network"]["catch_up_policy"],
            },
            "instrumentation": {
                "production_tail_source": "rl_agent/splitfusion_live_dispatch_v1/context_tail.py",
                "production_tail_sha256": common.BOUND_INPUTS[
                    "rl_agent/splitfusion_live_dispatch_v1/context_tail.py"
                ],
                "approach": (
                    "subclass of ContextualFrozenP025TailAdapter reproducing the "
                    "production operation sequence with deferred-read CUDA event "
                    "pairs and perf_counter boundaries; proved bit-identical to "
                    "the production adapter before measurement"
                ),
                "hash_bound_production_files_modified": [],
                "edge_deadline_policy": "CLASSIFY_AND_RECORD_NEVER_DROP",
                "clock": "time.time_ns for cross-process joins; time.perf_counter_ns for stages",
            },
            "synthetic_tests": synthetic_tests,
            "reference_points": {
                "phase13c_controlled_tail_gpu_ms": common.PHASE13C_CONTROLLED_TAIL_MS,
                "phase15_live_frozen_tail_ms": common.PHASE15_LIVE_FROZEN_TAIL_MS,
            },
        },
        "run_manifest_sha256",
    )


def run_synthetic_tests() -> dict[str, Any]:
    from .tests import test_diagnostic_contract as suite

    return suite.run_all()


def write_artifact_manifest(output: Path) -> tuple[Path, dict[str, str]]:
    digests: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {
            "ARTIFACT_MANIFEST.json",
            common.TERMINAL_SUCCESS,
            common.TERMINAL_FAILURE,
        }:
            digests[str(path.relative_to(output))] = common.sha256_file(path)
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    common.atomic_create_json(
        manifest_path,
        {
            "schema": "scenesense.splitfusion_timing_diagnostic_artifact_manifest.v1",
            "artifact_count": len(digests),
            "sha256": digests,
            "excluded_by_contract": [
                "rgb_frames", "radar_frames", "c2_tensors",
                "compressed_payload_blobs", "predictions", "raw_oai_tracer_logs",
            ],
        },
    )
    return manifest_path, digests


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--output", default=common.OUTPUT_RELPATH)
    args = parser.parse_args(list(argv) if argv is not None else None)
    require(args.execute == EXECUTE_TOKEN, "execution token mismatch")

    started = time.time()
    output = (ROOT / args.output).resolve()
    require(not output.exists(), f"create-only output already exists: {output}")

    campaign_path = repo_path(common.PILOT_CONFIG_RELPATH)
    campaign = common.load_json(campaign_path)

    print("preflight: git, bindings, container runtime, cold host", flush=True)
    git_state = verify_git_state()
    bindings = verify_bound_inputs()
    container = verify_container_runtime()
    cold = verify_cold_host(campaign, label="run/preflight")
    synthetic = run_synthetic_tests()
    require(
        bool(synthetic.get("all_passed")),
        f"focused synthetic tests failed: {synthetic}",
    )

    registry = SplitActionRegistry.from_runtime_binding()
    actions = verify_actions(registry)
    # The fit sampling is CPU-only by Phase-13C contract, so it precedes the
    # CUDA identity preflight and every model preload.
    print("preflight: reconstructing the registered Phase-13C fit sample", flush=True)
    sample, sample_context = construct_registered_sample()
    print("preflight: CUDA device identity", flush=True)
    cuda = verify_cuda()

    run_id = output.name
    output.mkdir(parents=True, exist_ok=False)
    (output / "per_frame").mkdir(parents=False, exist_ok=False)
    manifest = build_manifest(
        run_id=run_id, output=output, git_state=git_state, bindings=bindings,
        cuda=cuda, container=container, cold=cold, actions=actions, sample=sample,
        campaign=campaign, synthetic_tests=synthetic,
    )
    common.atomic_create_json(output / "RUN_MANIFEST.json", manifest)
    print(f"manifest sealed: {manifest['run_manifest_sha256']}", flush=True)

    device = torch.device("cuda:0")
    print("loading resident UE and edge models once", flush=True)
    ue, ue_ledger, ue_models, base, _registry = preload_ue(device)
    host_edge = preload_instrumented_edge(device)
    inference = base.data.InferenceDataset(sample_context["dataset_root"], "train")
    require(
        len(inference.rows) == contract.TRAIN_TOTAL_FRAMES,
        "deployable train inference row count drift",
    )

    reports: list[dict[str, Any]] = []
    status = common.TERMINAL_FAILURE
    failure = ""
    try:
        for action_id, _profile_id in common.DIAGNOSTIC_ACTIONS:
            reports.append(
                run_action(
                    action_id=action_id, campaign=campaign, campaign_path=campaign_path,
                    sample=sample, ue=ue, inference=inference, host_edge=host_edge,
                    registry=registry, run_id=run_id,
                )
            )
        status = common.TERMINAL_SUCCESS
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if reports:
            summaries = [summarize_action(report) for report in reports]
            comparisons = build_comparisons(summaries)
            artifacts: dict[str, str] = {}
            for report in reports:
                name = f"action_{report['action_id']:02d}_{report['profile_id']}.csv"
                artifacts[f"per_frame/{name}"] = write_per_frame_csv(
                    output / "per_frame" / name, report
                )
            artifacts["action_summary.csv"] = write_action_summary_csv(
                output / "action_summary.csv", summaries
            )
            try:
                artifacts.update(
                    write_figure(output / "timing_breakdown", summaries, comparisons)
                )
            except Exception as exc:
                artifacts["figure_error"] = f"{type(exc).__name__}: {exc}"
            runtime_seconds = time.time() - started
            results = {
                "schema": common.SCHEMA,
                "run_id": run_id,
                "status": status,
                "failure": failure,
                "run_manifest_sha256": manifest["run_manifest_sha256"],
                "runtime_seconds": runtime_seconds,
                "actions_executed": [summary["action_id"] for summary in summaries],
                "comparisons": comparisons,
                "action_summaries": summaries,
                "lifecycle": [
                    {
                        "cell_id": report["cell_id"],
                        "action_id": report["action_id"],
                        "cold_before": report["cold_before"],
                        "cold_after": report.get("cold_after"),
                        "radio_attachment": report.get("radio_attachment"),
                        "radio_teardown": report.get("radio_teardown", {}).get(
                            "teardown", {}
                        ).get("all_lifecycle_gates_passed"),
                        "final_restore_noise_power_db": report.get("radio_teardown", {})
                        .get("final_restore", {})
                        .get("noise_power_db"),
                        "transmission": report.get("transmission"),
                        "datagram_materialization": report.get("datagram_materialization"),
                        "wall_seconds": report.get("wall_seconds"),
                    }
                    for report in reports
                ],
            }
            common.atomic_create_json(output / "DIAGNOSTIC_RESULTS.json", results)
            write_report(
                output / "REPORT.md",
                manifest=manifest, summaries=summaries, comparisons=comparisons,
                runtime_seconds=runtime_seconds,
            )
            write_artifact_manifest(output)
            common.atomic_create_json(
                output / status,
                {
                    "terminal": status,
                    "run_id": run_id,
                    "actions": [summary["action_id"] for summary in summaries],
                    "failure": failure,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            common.atomic_create_json(
                output / "NO_SCIENTIFIC_ROWS.json",
                {
                    "run_id": run_id,
                    "failure": failure,
                    "reason": "no action completed, so no scientific row exists",
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            write_artifact_manifest(output)
            common.atomic_create_json(
                output / common.TERMINAL_FAILURE,
                {
                    "terminal": common.TERMINAL_FAILURE,
                    "run_id": run_id,
                    "actions": [],
                    "failure": failure,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    del ue, ue_models, ue_ledger, host_edge
    print(f"terminal: {status}", flush=True)
    return 0 if status == common.TERMINAL_SUCCESS else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as error:
        print(f"diagnostic contract error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
