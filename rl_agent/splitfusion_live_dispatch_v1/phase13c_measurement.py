"""Phase-13C 36-profile fit-only CUDA/localhost replay measurement.

The runner deliberately delegates action resolution, numerical transport, SFD1
validation, reconstruction and the frozen p025 tail to the qualified Phase-13A/B
runtime.  This file adds only deterministic fit sampling, a preregistered rotated
schedule, stage measurement, durable profile records and fail-closed resume.
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
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import torch

from phase2_map_sharing.transport import CHUNK_HEADER, ChunkReassembler, chunk_payload
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    guards,
    teacher_cache,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.base_runtime import (
    load_base,
)

from . import phase13b_qualification as phase13b
from .edge_runtime import PreloadedSplitEdgeRuntime
from .envelope import HEADER_BYTES, PROTOCOL_VERSION, unpack_envelope
from .registry import FAMILIES, QUANTIZERS, ActionProfile, SplitActionRegistry, sha256_file
from .timing import EDGE_STAGES, UE_STAGES, TimingTrace
from .ue_runtime import DispatchMetadata, PreloadedSplitUERuntime


EXECUTE_TOKEN = "SPLITFUSION_PHASE13C_36X300_LOCALHOST_MEASUREMENT"
SCHEMA = "scenesense.splitfusion_phase13c_36x300_localhost_measurement.v1"
MANIFEST_SCHEMA = "scenesense.splitfusion_phase13c_run_manifest.v1"
SAMPLE_SCHEMA = "scenesense.splitfusion_phase13c_fit_sample.v1"
SCHEDULE_SCHEMA = "scenesense.splitfusion_phase13c_rotated_schedule.v1"
PROFILE_RECORD_SCHEMA = "scenesense.splitfusion_phase13c_profile_record.v1"
TERMINAL = "SPLITFUSION_PHASE13C_36X300_LOCALHOST_MEASUREMENT_COMPLETE"
STARTING_HEAD = "99ab864d2962ea0742f33ea704bb967cd03ac458"
OUTPUT_RELPATH = (
    "experiments/splitfusion_live_dispatch_v1/"
    "20260904_phase13c_36x300_localhost_measurement"
)
DEVICE_NAME = "NVIDIA GeForce RTX 5090"
FRAMES = 300
PROFILE_COUNT = 36
TRANSACTIONS = FRAMES * PROFILE_COUNT
MEASURED_Q_E4 = (0, 3000, 5000)
CHUNK_BYTES = 12_500
SOCKET_BUFFER_REQUEST_BYTES = 8 * 1024 * 1024
EXPECTED_DIRTY_PATHS = phase13b.EXPECTED_DIRTY_PATHS
PHASE13B_ARTIFACTS = MappingProxyType(
    {
        "qualification": {
            "path": (
                "experiments/splitfusion_live_dispatch_v1/"
                "20260904_phase13b_four_action_gpu_loopback_qualification/qualification.json"
            ),
            "sha256": "780d5bd162d2474282074b680cf0fccaf39f1376511c70fb4ec25015e317e03a",
        },
        "report": {
            "path": (
                "experiments/splitfusion_live_dispatch_v1/"
                "20260904_phase13b_four_action_gpu_loopback_qualification/REPORT.md"
            ),
            "sha256": "1c6109c8df897b0388ae1cd45bd092a77581362b1e2df4499398b876554afe86",
        },
        "terminal": {
            "path": (
                "experiments/splitfusion_live_dispatch_v1/"
                "20260904_phase13b_four_action_gpu_loopback_qualification/"
                "SPLITFUSION_LIVE_DISPATCH_FOUR_ACTION_GPU_LOOPBACK_QUALIFIED"
            ),
            "sha256": "dcc37db9e923837d8bfe433df040b83365adb74a79b16e352b71bf6713ab35bc",
        },
    }
)
LATENCY_FIELDS = (
    "front_gpu_ms",
    "ranker_gpu_ms",
    "ae_encode_gpu_ms",
    "quantize_pack_ms",
    "zstd_compress_ms",
    "sfd1_fragment_send_ms",
    "ue_prepare_ms",
    "localhost_delivery_ms",
    "envelope_validate_ms",
    "zstd_decompress_ms",
    "dequant_scatter_ms",
    "ae_decode_gpu_ms",
    "tail_gpu_ms",
    "p025_serialize_ms",
    "back_ms",
    "capture_partition_ue_call_ms",
    "capture_partition_ue_to_first_send_ms",
    "capture_partition_reassembly_to_edge_call_ms",
    "capture_partition_edge_call_ms",
    "capture_to_edge_result_ms",
)
PAYLOAD_FIELDS = (
    "scientific_inner_bytes",
    "sfd1_bytes",
    "udp_application_bytes",
    "estimated_on_wire_bytes",
    "datagram_count",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(relative: str) -> Path:
    root = _root().resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository: {relative}") from exc
    return path


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    prefix = f"{str(tensor.dtype)}:{list(tensor.shape)}:".encode("ascii")
    return _digest_bytes(prefix + tensor.numpy().tobytes(order="C"))


def _seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(document)
    _require(field not in sealed, f"seal field already exists: {field}")
    sealed[field] = _digest_bytes(_canonical_bytes(sealed))
    return sealed


def _verify_seal(document: Mapping[str, Any], field: str) -> None:
    copy = dict(document)
    observed = str(copy.pop(field, ""))
    _require(len(observed) == 64, f"sealed document lacks {field}")
    _require(
        observed == _digest_bytes(_canonical_bytes(copy)),
        f"sealed document hash mismatch: {field}",
    )


def _atomic_create_text(path: Path, text: str) -> str:
    _require(not path.exists(), f"refusing to overwrite existing artifact: {path}")
    staging = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with staging.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    return sha256_file(path)


def _atomic_create_json(path: Path, value: Any) -> str:
    return _atomic_create_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_root(),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.rstrip("\r\n")


def _verify_git_state() -> dict[str, Any]:
    head = _git_output("rev-parse", "HEAD")
    parent = _git_output("rev-parse", "HEAD^")
    _require(
        parent == STARTING_HEAD,
        f"Phase-13C implementation parent is {parent}, expected {STARTING_HEAD}",
    )
    _require(
        _git_output("merge-base", "--is-ancestor", STARTING_HEAD, head) == "",
        "required Phase-13C starting commit is not an ancestor",
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
        frozenset(paths) == EXPECTED_DIRTY_PATHS
        and len(paths) == len(EXPECTED_DIRTY_PATHS),
        f"unexpected dirty paths: observed={sorted(paths)} "
        f"expected={sorted(EXPECTED_DIRTY_PATHS)}",
    )
    source = _repo_path("rl_agent/splitfusion_live_dispatch_v1/phase13c_measurement.py")
    return {
        "head": head,
        "starting_head": STARTING_HEAD,
        "implementation_parent": parent,
        "source_path": str(source.relative_to(_root())),
        "source_sha256": sha256_file(source),
        "expected_user_owned_dirty_paths": sorted(paths),
        "phase13b_porcelain_parser_reused": True,
    }


def _verify_phase13b_artifacts() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, binding in PHASE13B_ARTIFACTS.items():
        path = _repo_path(binding["path"])
        observed = sha256_file(path)
        _require(observed == binding["sha256"], f"Phase-13B {name} hash drift")
        result[name] = {"path": binding["path"], "sha256": observed}
    qualification = json.loads(
        _repo_path(PHASE13B_ARTIFACTS["qualification"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    _require(
        qualification.get("schema")
        == "scenesense.splitfusion_live_dispatch_phase13b_qualification.v1"
        and qualification.get("terminal")
        == "SPLITFUSION_LIVE_DISPATCH_FOUR_ACTION_GPU_LOOPBACK_QUALIFIED",
        "Phase-13B qualification schema or terminal drift",
    )
    terminal = _repo_path(PHASE13B_ARTIFACTS["terminal"]["path"])
    _require(
        terminal.read_text(encoding="utf-8").strip()
        == (
            "SPLITFUSION_LIVE_DISPATCH_FOUR_ACTION_GPU_LOOPBACK_QUALIFIED "
            + PHASE13B_ARTIFACTS["qualification"]["sha256"]
        ),
        "Phase-13B terminal does not bind the qualification JSON",
    )
    _require(
        qualification.get("integrity", {}).get("all_direct_path_parity_exact") is True
        and all(
            qualification.get("integrity", {}).get("frozen_state_equal", {}).values()
        ),
        "Phase-13B functional or frozen-state qualification is not complete",
    )
    result["qualified_action_ids"] = qualification["integrity"][
        "requested_actions_resolved_from_catalog"
    ]
    result["qualification_implementation_commit"] = qualification[
        "implementation_commit"
    ]
    return result


def _profile_record(profile: ActionProfile) -> dict[str, Any]:
    return {
        "action_id": profile.action_id,
        "profile_id": profile.profile_id,
        "execution_mode": profile.execution_mode,
        "family": profile.family,
        "family_id": profile.family_id,
        "quantizer": profile.quantizer,
        "bit_width": profile.bit_width,
        "q": profile.q,
        "q_e4": profile.q_e4,
        "keep_count": profile.keep_count,
        "drop_count": profile.drop_count,
        "transported_channels": profile.transported_channels,
        "latent_width": profile.latent_width,
        "routing_tag": profile.routing_tag,
        "decoder_identity": profile.decoder_identity,
        "zstd_level": profile.zstd_level,
        "wire": asdict(profile.wire),
        "segmentation_installable": profile.segmentation_installable,
        "segmentation_behavior": profile.segmentation_behavior,
    }


def _select_profiles(registry: SplitActionRegistry) -> tuple[ActionProfile, ...]:
    profiles = tuple(
        registry.find(family, quantizer, q_e4)
        for family in FAMILIES
        for quantizer in QUANTIZERS
        for q_e4 in MEASURED_Q_E4
    )
    identities = {(p.family, p.quantizer, p.q_e4) for p in profiles}
    _require(len(profiles) == PROFILE_COUNT, "Phase-13C profile count is not 36")
    _require(len(identities) == PROFILE_COUNT, "Phase-13C profile identities repeat")
    _require(
        len({p.action_id for p in profiles}) == PROFILE_COUNT,
        "Phase-13C action IDs repeat",
    )
    _require(
        all(
            p.execution_mode == "SPLIT"
            and p.q_e4 in MEASURED_Q_E4
            and p.zstd_level == 1
            and p.wire.layout == "CURRENT_CELL_MAJOR"
            for p in profiles
        ),
        "Phase-13C catalog subset violates the locked transport contract",
    )
    return profiles


def _dataset_root(base: Any) -> Path:
    config_path = (
        contract.perception_lock_path().parent.parent
        / "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json"
    ).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return (base.common.ROOT / config["dataset_root"]).resolve(strict=True)


def _calibration_binding(base: Any, row: Mapping[str, str]) -> dict[str, Any]:
    calibration = {
        "intrinsic": base.data.model_intrinsic(row),
        "extrinsic": base.data.camera_extrinsic(row),
    }
    return {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "sha256": _tensor_digest(value),
        }
        for name, value in sorted(calibration.items())
    }


def _construct_sample() -> tuple[dict[str, Any], dict[str, Any]]:
    _require(not torch.cuda.is_initialized(), "CUDA initialized before fit sampling")
    base = load_base()
    dataset_root = _dataset_root(base)
    rows = base.data.load_split_rows(dataset_root, "train")
    partition = teacher_cache.build_split_partition(SimpleNamespace(rows=rows))
    _require(
        len(contract.TRAIN_FIT_EPISODES) == 8,
        "immutable Phase-4 partition must contain exactly eight fit episodes",
    )
    _require(
        len(partition.fit_indices) == 13_543
        and len(partition.holdout_indices) == 3_284,
        "immutable Phase-4 frame counts drift",
    )
    fit_index_set = set(partition.fit_indices)
    holdout_id_set = set(partition.holdout_sample_ids)
    by_episode: dict[str, list[tuple[int, Mapping[str, str]]]] = {
        episode: [] for episode in contract.TRAIN_FIT_EPISODES
    }
    for index in partition.fit_indices:
        row = rows[index]
        _require(row["split"] == "train", "fit row is not registered train")
        by_episode[row["experiment_id"]].append((index, row))

    allocation_rows: list[dict[str, Any]] = []
    for episode in contract.TRAIN_FIT_EPISODES:
        count = len(by_episode[episode])
        _require(count > 0, f"fit episode is empty: {episode}")
        numerator = FRAMES * count
        floor_value, remainder_numerator = divmod(numerator, contract.TRAIN_FIT_FRAMES)
        allocation_rows.append(
            {
                "episode_id": episode,
                "registered_frame_count": count,
                "ideal_fraction": f"{numerator}/{contract.TRAIN_FIT_FRAMES}",
                "ideal_decimal": f"{numerator / contract.TRAIN_FIT_FRAMES:.12f}",
                "floor_allocation": floor_value,
                "remainder_fraction": (
                    f"{remainder_numerator}/{contract.TRAIN_FIT_FRAMES}"
                ),
                "remainder_decimal": (
                    f"{remainder_numerator / contract.TRAIN_FIT_FRAMES:.12f}"
                ),
                "remainder_numerator": remainder_numerator,
                "final_allocation": floor_value,
            }
        )
    remaining = FRAMES - sum(row["floor_allocation"] for row in allocation_rows)
    _require(0 <= remaining < len(allocation_rows), "largest-remainder count invalid")
    priority = sorted(
        allocation_rows,
        key=lambda row: (-row["remainder_numerator"], row["episode_id"]),
    )
    for row in priority[:remaining]:
        row["final_allocation"] += 1
    _require(
        len(allocation_rows) == 8
        and sum(row["final_allocation"] for row in allocation_rows) == FRAMES
        and all(row["final_allocation"] >= 1 for row in allocation_rows),
        "largest-remainder allocation contract failed",
    )

    allowed_row_fields = tuple(base.data.InferenceDataset._ROW_FIELDS) + (
        "experiment_id",
        "split",
        "timestamp",
    )
    selected: list[dict[str, Any]] = []
    for allocation in allocation_rows:
        episode = allocation["episode_id"]
        episode_rows = sorted(
            by_episode[episode],
            key=lambda item: (str(item[1]["sample_id"]), int(item[1]["frame_id"])),
        )
        count = len(episode_rows)
        allocation_count = int(allocation["final_allocation"])
        _require(allocation_count >= 2, f"episode allocation lacks two endpoints: {episode}")
        indices = [
            (position * (count - 1)) // (allocation_count - 1)
            for position in range(allocation_count)
        ]
        _require(
            len(indices) == len(set(indices))
            and indices[0] == 0
            and indices[-1] == count - 1,
            f"endpoint-inclusive selection failed: {episode}",
        )
        allocation["selected_episode_indices"] = indices
        for selected_ordinal, episode_index in enumerate(indices):
            dataset_index, source_row = episode_rows[episode_index]
            _require(dataset_index in fit_index_set, "selected index escaped the fit split")
            public_row = {field: source_row[field] for field in allowed_row_fields}
            selected.append(
                {
                    "sample_ordinal": len(selected),
                    "episode_selection_ordinal": selected_ordinal,
                    "episode_sorted_index": episode_index,
                    "dataset_index": dataset_index,
                    "sample_id": source_row["sample_id"],
                    "frame_id": int(source_row["frame_id"]),
                    "episode_id": episode,
                    "registered_split": "fit",
                    "source_row": public_row,
                    "source_row_sha256": _digest_bytes(_canonical_bytes(source_row)),
                }
            )
    sample_ids = [row["sample_id"] for row in selected]
    _require(len(selected) == FRAMES, "fit sample does not contain 300 rows")
    _require(len(set(sample_ids)) == FRAMES, "fit sample IDs are not unique")
    _require(not (set(sample_ids) & holdout_id_set), "fit sample overlaps holdout")
    _require(
        all(row["source_row"]["split"] == "train" for row in selected),
        "selected sample includes a non-train source row",
    )

    calibration_identities = {
        _digest_bytes(_canonical_bytes(_calibration_binding(base, row["source_row"])))
        for row in selected
    }
    _require(
        len(calibration_identities) == 1,
        f"selected replay has multiple calibration identities: {sorted(calibration_identities)}",
    )
    stream_calibration = _calibration_binding(base, selected[0]["source_row"])
    selected_set = set(sample_ids)
    warmup_candidates = sorted(
        (
            (index, rows[index])
            for index in partition.fit_indices
            if rows[index]["sample_id"] not in selected_set
        ),
        key=lambda item: (str(item[1]["sample_id"]), int(item[1]["frame_id"])),
    )
    _require(bool(warmup_candidates), "no distinct fit warm-up frame is available")
    warmup_index, warmup_source = warmup_candidates[0]
    warmup_public = {field: warmup_source[field] for field in allowed_row_fields}
    _require(
        _calibration_binding(base, warmup_public) == stream_calibration,
        "warm-up calibration differs from the bound replay calibration",
    )
    sample = _seal(
        {
            "schema": SAMPLE_SCHEMA,
            "partition": {
                "authority": (
                    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.contract Phase-4"
                ),
                "train_manifest_sha256": contract.TRAIN_MANIFEST_SHA256,
                "fit_episodes": list(contract.TRAIN_FIT_EPISODES),
                "fit_episode_count": 8,
                "fit_registered_frames": len(partition.fit_indices),
                "fit_sample_id_sha256": contract.TRAIN_FIT_SAMPLE_ID_SHA256,
                "holdout_episodes": list(contract.TRAIN_HOLDOUT_EPISODES),
                "holdout_registered_but_unread_frames": len(partition.holdout_indices),
                "holdout_sample_id_sha256": contract.TRAIN_HOLDOUT_SAMPLE_ID_SHA256,
                "holdout_sensor_or_label_rows_read": 0,
                "validation_sensor_or_label_rows_read": 0,
                "test_sensor_or_label_rows_read": 0,
            },
            "allocation": {
                "total": FRAMES,
                "denominator_fit_frames": contract.TRAIN_FIT_FRAMES,
                "rule": (
                    "floor(300*n_i/N), then largest fractional remainders; "
                    "exact ties by ascending canonical episode_id"
                ),
                "remaining_after_floors": remaining,
                "episodes": allocation_rows,
            },
            "within_episode_selection": {
                "sort_key": ["sample_id ascending", "integer frame_id ascending"],
                "rule": "floor(j*(episode_frame_count-1)/(allocation_i-1))",
                "endpoint_inclusive": True,
                "selected_indices_unique": True,
                "selected_sample_ids_unique_within_and_across_episodes": True,
            },
            "selected_frame_count": len(selected),
            "selected_sample_id_sha256": contract.sample_id_digest(sample_ids),
            "selected_rows": selected,
            "warmup": {
                "dataset_index": warmup_index,
                "sample_id": warmup_source["sample_id"],
                "frame_id": int(warmup_source["frame_id"]),
                "episode_id": warmup_source["experiment_id"],
                "registered_split": "fit",
                "source_row": warmup_public,
                "source_row_sha256": _digest_bytes(_canonical_bytes(warmup_source)),
                "excluded_from_measurement_sample": True,
            },
            "calibration": {
                "identity_sha256": next(iter(calibration_identities)),
                "tensors": stream_calibration,
                "identical_across_selected_stream": True,
                "resident_at_edge_not_transmitted": True,
            },
            "access_scope": {
                "sensor_modalities": ["RGB", "radar", "camera_calibration"],
                "boxes_read": False,
                "semantic_ground_truth_read": False,
                "avo_read": False,
                "depth_ground_truth_read": False,
                "evaluation_records_read": False,
            },
        },
        "sample_manifest_sha256",
    )
    _require(not torch.cuda.is_initialized(), "CUDA initialized while constructing sample")
    return sample, {
        "base": base,
        "dataset_root": dataset_root,
        "train_rows": rows,
    }


def _construct_schedule(
    sample: Mapping[str, Any], profiles: Sequence[ActionProfile]
) -> dict[str, Any]:
    action_ids = [profile.action_id for profile in profiles]
    frames: list[dict[str, Any]] = []
    flat = []
    for frame_ordinal, row in enumerate(sample["selected_rows"]):
        rotation = frame_ordinal % len(action_ids)
        ordered = action_ids[rotation:] + action_ids[:rotation]
        frames.append(
            {
                "frame_ordinal": frame_ordinal,
                "sample_id": row["sample_id"],
                "rotation": rotation,
                "starting_action_id": ordered[0],
                "ordered_action_ids": ordered,
            }
        )
        flat.extend(
            {
                "transaction_ordinal": len(flat),
                "frame_ordinal": frame_ordinal,
                "sample_id": row["sample_id"],
                "action_id": action_id,
            }
            for action_id in ordered
        )
    _require(len(flat) == TRANSACTIONS, "schedule transaction count drift")
    counts = Counter(item["action_id"] for item in flat)
    _require(
        counts == Counter({action_id: FRAMES for action_id in action_ids}),
        "rotated schedule does not contain 300 measurements per action",
    )
    return _seal(
        {
            "schema": SCHEDULE_SCHEMA,
            "profile_order": action_ids,
            "profile_order_basis": (
                "catalog fields: family registry order, quantizer registry order, "
                "q_e4=(0,3000,5000)"
            ),
            "rotation_rule": "frame_ordinal modulo 36",
            "source_frame_count": FRAMES,
            "profile_count": PROFILE_COUNT,
            "transaction_count": TRANSACTIONS,
            "frames": frames,
            "flat_schedule_sha256": _digest_bytes(_canonical_bytes(flat)),
            "per_action_counts": {str(key): counts[key] for key in action_ids},
        },
        "schedule_manifest_sha256",
    )


def _output_candidate(*, resume: bool) -> Path:
    experiments = (_root() / "experiments").resolve(strict=True)
    candidate = (_root() / OUTPUT_RELPATH).resolve(strict=False)
    try:
        candidate.relative_to(experiments)
    except ValueError as exc:
        raise RuntimeError("Phase-13C output escapes experiments root") from exc
    if resume:
        _require(candidate.is_dir(), f"resume output directory does not exist: {candidate}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(experiments)
        except ValueError as exc:
            raise RuntimeError("resolved resume output escapes experiments root") from exc
        return resolved
    _require(not candidate.exists(), f"create-only output already exists: {candidate}")
    return candidate


def _create_output(candidate: Path) -> Path:
    experiments = (_root() / "experiments").resolve(strict=True)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=False, exist_ok=False)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(experiments)
    except ValueError as exc:
        raise RuntimeError("created Phase-13C output escapes experiments root") from exc
    (resolved / "profiles").mkdir(parents=False, exist_ok=False)
    return resolved


class CudaStageTimers:
    """Reusable CUDA-event pairs; every recorded stage synchronizes explicitly."""

    STAGES = (
        "front_gpu_ms",
        "ranker_gpu_ms",
        "ae_encode_gpu_ms",
        "ae_decode_gpu_ms",
        "tail_gpu_ms",
    )

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._events = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in self.STAGES
        }
        self._values: dict[str, float] = {}

    def reset(self) -> None:
        self._values.clear()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        _require(name in self._events, f"unknown CUDA timing stage: {name}")
        _require(name not in self._values, f"CUDA timing stage repeated: {name}")
        start, finish = self._events[name]
        with torch.cuda.device(self._device):
            start.record()
            try:
                yield
            finally:
                finish.record()
                finish.synchronize()
                value = float(start.elapsed_time(finish))
                _require(math.isfinite(value) and value >= 0.0, f"invalid CUDA timing: {name}")
                self._values[name] = value

    def value(self, name: str, *, bypass: bool) -> float:
        if bypass:
            _require(name not in self._values, f"bypassed CUDA stage executed: {name}")
            return 0.0
        _require(name in self._values, f"CUDA timing stage missing: {name}")
        return self._values[name]


class TimedFront:
    def __init__(
        self, model: torch.nn.Module, ledger: phase13b.CallLedger, timers: CudaStageTimers
    ) -> None:
        self._delegate = phase13b.FrozenFrontAdapter(model, ledger)
        self._timers = timers

    def __call__(self, input_7ch: torch.Tensor) -> torch.Tensor:
        with self._timers.measure("front_gpu_ms"):
            return self._delegate(input_7ch)

    def release_c2(self) -> None:
        value = self._delegate.take_c2()
        del value


class TimedRanker:
    def __init__(
        self,
        ranker: torch.nn.Module,
        ledger: phase13b.CallLedger,
        timers: CudaStageTimers,
    ) -> None:
        self._ranker = ranker
        self._ledger = ledger
        self._timers = timers

    def score_cells(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump("ranker")
        with self._timers.measure("ranker_gpu_ms"):
            return self._ranker.score_cells(c2)


class TimedAutoencoder:
    """Identity-preserving timing proxy around one already-frozen AE family."""

    def __init__(
        self,
        family: str,
        autoencoder: Any,
        ledger: phase13b.CallLedger,
        timers: CudaStageTimers,
    ) -> None:
        self.family = family
        self.family_id = int(autoencoder.family_id)
        self.bottleneck = int(autoencoder.bottleneck)
        self.routing_tag = int(autoencoder.routing_tag)
        self._autoencoder = autoencoder
        self._ledger = ledger
        self._timers = timers

    def encode(self, c2: torch.Tensor) -> torch.Tensor:
        self._ledger.bump(f"ae_encoder_{self.family}")
        with self._timers.measure("ae_encode_gpu_ms"):
            return self._autoencoder.encode(c2)

    def decode(self, latent: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        self._ledger.bump(f"ae_decoder_{self.family}")
        with self._timers.measure("ae_decode_gpu_ms"):
            return self._autoencoder.decode(latent, keep_mask)


class TimedTail:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        base: Any,
        initial_row: Mapping[str, str],
        calibration: Mapping[str, torch.Tensor],
        calibration_identity: Mapping[str, Any],
        ledger: phase13b.CallLedger,
        timers: CudaStageTimers,
    ) -> None:
        self._delegate = phase13b.FrozenP025TailAdapter(
            model=model,
            base=base,
            row=initial_row,
            calibration=calibration,
            ledger=ledger,
        )
        self._base = base
        self._calibration_identity = dict(calibration_identity)
        self._timers = timers

    def bind_row(self, row: Mapping[str, str]) -> None:
        _require(self._delegate._last is None, "tail row changed with an unconsumed output")
        observed = _calibration_binding(self._base, row)
        _require(observed == self._calibration_identity, "per-frame calibration identity drift")
        self._delegate._row = MappingProxyType(dict(row))

    def __call__(
        self, c2: torch.Tensor, metadata: DispatchMetadata
    ) -> Mapping[str, torch.Tensor]:
        with self._timers.measure("tail_gpu_ms"):
            return self._delegate(c2, metadata)

    def serialize(self, perception: Mapping[str, torch.Tensor]) -> bytes:
        return self._delegate.serialize(perception)

    def take_snapshot(self) -> phase13b.TailSnapshot:
        return self._delegate.take_snapshot()


@dataclass(frozen=True)
class TimedUdpDelivery:
    payload: bytes
    message_id: int
    datagrams: int
    duplicate_datagrams: int
    sfd1_application_bytes: int
    chunk_header_bytes: int
    udp_application_bytes: int
    estimated_ip_udp_bytes: int
    estimated_on_wire_bytes: int
    fragment_started_perf_counter_ns: int
    first_send_started_perf_counter_ns: int
    last_send_finished_perf_counter_ns: int
    reassembly_completed_perf_counter_ns: int
    fragment_send_ms: float
    localhost_delivery_ms: float


class TimedRawUdpLoopback:
    """The qualified raw ``!IHH`` route with host timing at its true boundaries."""

    def __init__(self, expected_messages: int) -> None:
        self._expected = int(expected_messages)
        self._delivered: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._receiver.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_REQUEST_BYTES
        )
        self._sender.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_REQUEST_BYTES
        )
        self._receiver.bind(("127.0.0.1", 0))
        self._receiver.settimeout(120.0)
        self._destination = self._receiver.getsockname()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._receive,
            name="phase13c-raw-udp-receiver",
            daemon=True,
        )
        self.requested_buffer_bytes = SOCKET_BUFFER_REQUEST_BYTES
        self.reported_receive_buffer_bytes = int(
            self._receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        )
        self.reported_send_buffer_bytes = int(
            self._sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        )
        try:
            self._thread.start()
            _require(self._ready.wait(timeout=5.0), "UDP receiver did not become ready")
        except BaseException:
            self.close()
            raise

    def _receive(self) -> None:
        reassembler = ChunkReassembler(timeout_s=120.0, max_chunks=4096)
        completed = 0
        self._ready.set()
        try:
            while completed < self._expected:
                datagram, source = self._receiver.recvfrom(65_535)
                result = reassembler.ingest(
                    f"{source[0]}:{source[1]}",
                    datagram,
                    received_at_s=time.monotonic(),
                )
                if result is None:
                    continue
                completed_ns = time.perf_counter_ns()
                datagrams = int(result.chunk_count)
                application = len(result.payload)
                chunk_headers = datagrams * CHUNK_HEADER.size
                udp_application = application + chunk_headers
                ip_udp = datagrams * 28
                self._delivered.put(
                    {
                        "payload": result.payload,
                        "message_id": int(result.message_id),
                        "datagrams": datagrams,
                        "duplicate_datagrams": int(result.duplicate_chunks),
                        "sfd1_application_bytes": application,
                        "chunk_header_bytes": chunk_headers,
                        "udp_application_bytes": udp_application,
                        "estimated_ip_udp_bytes": ip_udp,
                        "estimated_on_wire_bytes": udp_application + ip_udp,
                        "reassembly_completed_perf_counter_ns": completed_ns,
                    }
                )
                completed += 1
        except BaseException as exc:
            if not self._closed:
                self._delivered.put(exc)

    def roundtrip(self, payload: bytes, *, message_id: int) -> TimedUdpDelivery:
        _require(not self._closed, "UDP loopback is closed")
        fragment_started = time.perf_counter_ns()
        chunks = chunk_payload(payload, message_id=message_id, chunk_bytes=CHUNK_BYTES)
        delivery_started = time.perf_counter_ns()
        for datagram in chunks:
            sent = self._sender.sendto(datagram, self._destination)
            _require(sent == len(datagram), "localhost UDP datagram send was truncated")
        send_finished = time.perf_counter_ns()
        delivered = self._delivered.get(timeout=125.0)
        if isinstance(delivered, BaseException):
            raise RuntimeError("localhost UDP receiver failed") from delivered
        _require(delivered["message_id"] == message_id, "reassembled message ID drift")
        _require(delivered["datagrams"] == len(chunks), "UDP datagram count drift")
        _require(delivered["duplicate_datagrams"] == 0, "duplicate UDP datagram observed")
        _require(delivered["payload"] == payload, "reassembled SFD1 bytes differ")
        reassembly_finished = int(delivered["reassembly_completed_perf_counter_ns"])
        _require(reassembly_finished >= delivery_started, "negative UDP delivery time")
        return TimedUdpDelivery(
            **delivered,
            fragment_started_perf_counter_ns=fragment_started,
            first_send_started_perf_counter_ns=delivery_started,
            last_send_finished_perf_counter_ns=send_finished,
            fragment_send_ms=(send_finished - fragment_started) / 1_000_000.0,
            localhost_delivery_ms=(reassembly_finished - delivery_started) / 1_000_000.0,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._receiver.close()
        self._sender.close()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _trace_durations(trace: TimingTrace, stages: tuple[str, ...]) -> dict[str, float]:
    phase13b._require_timing(trace, stages)
    return {
        boundary.name: (
            boundary.finished_monotonic_ns - boundary.started_monotonic_ns
        )
        / 1_000_000.0
        for boundary in trace.boundaries
    }


def _trace_boundary(trace: TimingTrace, name: str) -> Any:
    matches = [boundary for boundary in trace.boundaries if boundary.name == name]
    _require(len(matches) == 1, f"timing trace lacks one {name} boundary")
    return matches[0]


def _counter_delta(after: Counter[str], before: Counter[str]) -> dict[str, int]:
    keys = sorted(set(after) | set(before))
    return {
        name: int(after[name] - before[name])
        for name in keys
        if after[name] != before[name]
    }


def _operation_delta(after: Any, before: Any) -> dict[str, int]:
    after_values = asdict(after)
    before_values = asdict(before)
    return {
        name: int(after_values[name] - before_values[name])
        for name in after_values
        if isinstance(after_values[name], int) and after_values[name] != before_values[name]
    }


def _profile_path(output: Path, action_id: int) -> Path:
    return output / "profiles" / f"action_{action_id:02d}.json"


def _validate_profile_record(
    document: Mapping[str, Any],
    *,
    manifest_sha256: str,
    profile: ActionProfile,
    sample_ids: Sequence[str],
) -> None:
    _verify_seal(document, "profile_record_sha256")
    _require(document.get("schema") == PROFILE_RECORD_SCHEMA, "profile record schema drift")
    _require(
        document.get("run_manifest_sha256") == manifest_sha256,
        "profile record manifest binding drift",
    )
    _require(document.get("profile") == _profile_record(profile), "profile identity drift")
    rows = document.get("measurements")
    _require(isinstance(rows, list) and len(rows) == FRAMES, "profile row count is not 300")
    _require(
        [row.get("sample_id") for row in rows] == list(sample_ids),
        "profile sample order or identity drift",
    )
    _require(
        all(
            row.get("action_id") == profile.action_id
            and all(
                isinstance(row.get(name), (int, float))
                and math.isfinite(float(row[name]))
                and float(row[name]) >= 0.0
                for name in (*LATENCY_FIELDS, *PAYLOAD_FIELDS)
            )
            for row in rows
        ),
        "profile scalar measurement contract drift",
    )
    counts = document.get("counts", {})
    _require(
        counts.get("attempted") == FRAMES
        and counts.get("delivered") == FRAMES
        and counts.get("decoded") == FRAMES
        and counts.get("completed") == FRAMES,
        "profile completion accounting drift",
    )


def _load_resume_records(
    output: Path,
    *,
    manifest_sha256: str,
    profiles: Sequence[ActionProfile],
    sample_ids: Sequence[str],
) -> dict[int, dict[str, Any]]:
    known = {profile.action_id: profile for profile in profiles}
    records: dict[int, dict[str, Any]] = {}
    observed_paths = sorted((output / "profiles").glob("*.json"))
    for path in observed_paths:
        _require(path.name.startswith("action_") and path.name.endswith(".json"), f"unknown profile artifact: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        action_id = int(document.get("profile", {}).get("action_id", -1))
        _require(action_id in known, f"unregistered durable action record: {action_id}")
        _require(action_id not in records, f"duplicate durable action record: {action_id}")
        _require(path == _profile_path(output, action_id), "durable action filename drift")
        _validate_profile_record(
            document,
            manifest_sha256=manifest_sha256,
            profile=known[action_id],
            sample_ids=sample_ids,
        )
        records[action_id] = document
    return records


def _historical_binding(historical: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frozen_input_hashes": historical["frozen_hashes"],
        "live_ae_package_source_hashes": historical[
            "live_ae_package_source_hashes"
        ],
        "historical_source_bindings": historical["historical_source_bindings"],
        "selection_bindings": historical["selection_bindings"],
        "phase11b_device_repair_source_transition": historical[
            "phase11b_device_repair_source_transition"
        ],
        "historical_provenance_checks_passed": True,
    }


def _runtime_source_binding() -> dict[str, Any]:
    paths = (
        "rl_agent/splitfusion_live_dispatch_v1/registry.py",
        "rl_agent/splitfusion_live_dispatch_v1/transport.py",
        "rl_agent/splitfusion_live_dispatch_v1/ue_runtime.py",
        "rl_agent/splitfusion_live_dispatch_v1/edge_runtime.py",
        "rl_agent/splitfusion_live_dispatch_v1/envelope.py",
        "rl_agent/splitfusion_live_dispatch_v1/timing.py",
        "rl_agent/splitfusion_live_dispatch_v1/phase13b_qualification.py",
        "rl_agent/splitfusion_live_dispatch_v1/phase13c_measurement.py",
        "phase2_map_sharing/transport.py",
    )
    return {
        path: sha256_file(_repo_path(path))
        for path in paths
    }


def _build_manifest(
    *,
    git: Mapping[str, Any],
    phase13a: Mapping[str, Any],
    phase13b_binding: Mapping[str, Any],
    registry: SplitActionRegistry,
    profiles: Sequence[ActionProfile],
    sample: Mapping[str, Any],
    schedule: Mapping[str, Any],
    gpu: Mapping[str, Any],
    historical: Mapping[str, Any],
    perception_binding: Mapping[str, Any],
    state_before: Mapping[str, Any],
    loopback: TimedRawUdpLoopback,
) -> dict[str, Any]:
    runtime_binding_path = _repo_path(
        "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json"
    )
    runtime_binding = json.loads(runtime_binding_path.read_text(encoding="utf-8"))
    return _seal(
        {
            "schema": MANIFEST_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PREREGISTERED_BEFORE_WARMUP_AND_RECORDED_MEASUREMENT",
            "implementation": dict(git),
            "provenance": {
                "phase13a": dict(phase13a),
                "phase13b": dict(phase13b_binding),
                "runtime_binding": {
                    "path": str(runtime_binding_path.relative_to(_root())),
                    "sha256": sha256_file(runtime_binding_path),
                    "inputs": runtime_binding["inputs"],
                    "selected_checkpoints": runtime_binding[
                        "selected_checkpoints"
                    ],
                    "startup_artifacts": runtime_binding["startup_artifacts"],
                },
                "active_runtime_source_sha256": _runtime_source_binding(),
                "phase11b_historical_checks": _historical_binding(historical),
                "perception": dict(perception_binding),
            },
            "registry_startup_audit": asdict(registry.startup_audit),
            "profile_inventory": [_profile_record(profile) for profile in profiles],
            "profile_selection": {
                "source": "immutable 72-action catalog fields, not action IDs",
                "families": list(FAMILIES),
                "quantizers": list(QUANTIZERS),
                "q_e4": list(MEASURED_Q_E4),
                "profile_count": PROFILE_COUNT,
                "excluded_q_e4": [7000, 9000, 9800],
            },
            "sample_manifest": dict(sample),
            "schedule_manifest": dict(schedule),
            "execution": {
                "warmup_profiles": PROFILE_COUNT,
                "warmup_frame_in_measurement_sample": False,
                "measured_frames_per_profile": FRAMES,
                "measured_profile_frames": TRANSACTIONS,
                "one_transaction_at_a_time": True,
                "batching": False,
                "pipelining": False,
                "c2_reuse": False,
                "raw_input_may_remain_resident_within_source_frame": True,
                "transport_layout": "CURRENT_CELL_MAJOR",
                "zstd_level": 1,
                "measurement_class": "live CUDA/localhost replay",
                "excluded": [
                    "sensor capture",
                    "disk input loading",
                    "CARLA",
                    "OAI",
                    "RFsim",
                    "perception scoring",
                    "training or tuning",
                ],
            },
            "timing": {
                "host_clock": "time.perf_counter_ns",
                "gpu_clock": "torch.cuda.Event(enable_timing=True)",
                "gpu_event_synchronization": "finish event synchronized for every GPU stage",
                "latency_fields": list(LATENCY_FIELDS),
                "quantile_convention": (
                    "nearest-rank: sorted_values[ceil(p*n)-1] for p=0.50 and p=0.95"
                ),
                "component_definitions": {
                    "front_gpu_ms": "CUDA-event frozen encode_front duration",
                    "ranker_gpu_ms": "CUDA-event ranker score_cells duration; exact zero for q=0 bypass",
                    "ae_encode_gpu_ms": "CUDA-event selected AE encode duration; exact zero for noAE",
                    "quantize_pack_ms": "existing UE quantize_pack host boundary",
                    "zstd_compress_ms": "existing UE mandatory-zstd compression host boundary",
                    "sfd1_fragment_send_ms": "host chunk construction through last UDP send completion",
                    "ue_prepare_ms": "existing total UE prepare boundary including SFD1 construction",
                    "localhost_delivery_ms": "immediately before first send through receiver reassembly completion",
                    "envelope_validate_ms": "combined SFD1/catalog prefix before zstd-decompression boundary",
                    "zstd_decompress_ms": "existing edge zstd-decompression host boundary",
                    "dequant_scatter_ms": "existing edge unpack_dequantize host boundary",
                    "ae_decode_gpu_ms": "CUDA-event selected AE decode duration; exact zero for noAE",
                    "tail_gpu_ms": "combined frozen tail, camera postprocess and p025 CUDA-event duration",
                    "p025_serialize_ms": "existing output_serialization host boundary",
                    "back_ms": "existing total edge-processing boundary",
                    "capture_partition_ue_call_ms": "exclusive host interval from capture start through UE prepare return",
                    "capture_partition_ue_to_first_send_ms": "exclusive host interval from UE return through fragment creation to immediately before first send",
                    "capture_partition_reassembly_to_edge_call_ms": "exclusive host interval from receiver reassembly completion through queue handoff to edge call start",
                    "capture_partition_edge_call_ms": "exclusive host interval from edge call start through serialized result return",
                    "capture_to_edge_result_ms": "host start immediately before UE front path through serialized edge return",
                },
                "mutual_exclusion": {
                    "capture_to_edge_result_ms": [
                        "capture_partition_ue_call_ms",
                        "capture_partition_ue_to_first_send_ms",
                        "localhost_delivery_ms",
                        "capture_partition_reassembly_to_edge_call_ms",
                        "capture_partition_edge_call_ms",
                    ],
                    "partition_verified_per_transaction": True,
                    "overlap": (
                        "localhost_delivery_ms begins before UDP sending completes and therefore overlaps "
                        "the send portion of sfd1_fragment_send_ms"
                    ),
                    "other_component_fields": (
                        "diagnostic boundaries not added to the exclusive partition"
                    ),
                    "no_double_counted_sum_published": True,
                },
            },
            "udp": {
                "transport": "real localhost UDP",
                "fragmentation_header": "!IHH",
                "chunk_bytes_including_header": CHUNK_BYTES,
                "requested_socket_buffer_bytes": loopback.requested_buffer_bytes,
                "reported_receive_buffer_bytes": loopback.reported_receive_buffer_bytes,
                "reported_send_buffer_bytes": loopback.reported_send_buffer_bytes,
                "sender_sockets": 1,
                "receiver_sockets": 1,
                "receiver_ready_before_transmission": True,
                "raw_sfd1_only": True,
                "secondary_compression": False,
                "retransmission": False,
                "localhost_diagnostic_not_oai_binding": True,
            },
            "environment": {
                **dict(gpu),
                "sys_executable": sys.executable,
                "platform": platform.platform(),
            },
            "startup_frozen_state": dict(state_before),
            "durability": {
                "profile_record_schema": PROFILE_RECORD_SCHEMA,
                "atomic_record_after_300_complete_rows": True,
                "resume_reuses_only_hash_valid_complete_records": True,
                "invalid_record_overwritten_or_remeasured": False,
            },
            "scope": {
                "fit_sensor_frames_selected": FRAMES,
                "distinct_fit_warmup_frames": 1,
                "holdout_sensor_frames_read": 0,
                "validation_sensor_frames_read": 0,
                "test_sensor_frames_read": 0,
                "boxes_semantic_gt_avo_depth_gt_or_evaluation_records_read": 0,
                "accuracy_rescored": False,
                "oai_campaign_launched": False,
            },
        },
        "run_manifest_sha256",
    )


def _load_or_write_manifest(
    output: Path, expected: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    path = output / "run_manifest.json"
    if not resume:
        _atomic_create_json(path, expected)
        return dict(expected)
    _require(path.is_file(), "resume output lacks run_manifest.json")
    observed = json.loads(path.read_text(encoding="utf-8"))
    _verify_seal(observed, "run_manifest_sha256")
    # Environment/process identity and creation time remain bound to the first run;
    # every immutable scientific/source field must match the current construction.
    for field in (
        "schema",
        "implementation",
        "provenance",
        "registry_startup_audit",
        "profile_inventory",
        "profile_selection",
        "sample_manifest",
        "schedule_manifest",
        "execution",
        "timing",
        "udp",
        "startup_frozen_state",
        "durability",
        "scope",
    ):
        _require(observed.get(field) == expected.get(field), f"resume manifest drift: {field}")
    return observed


def _load_models(
    *,
    device: torch.device,
    registry: SplitActionRegistry,
    historical: dict[str, Any],
    sample_context: Mapping[str, Any],
    expected_messages: int,
) -> dict[str, Any]:
    model, base, perception_binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    ranker_model = phase11b._load_ranker(device)
    autoencoders = {
        family_name: phase11b._load_selected_autoencoder(
            family_name,
            bottleneck,
            phase11b.FROZEN_INPUTS[family_name],
            historical["checkpoint_payloads"][family_name],
            device,
        )
        for family_name, _family_id, bottleneck in phase11b.FAMILIES
        if bottleneck is not None
    }
    historical["checkpoint_payloads"].clear()
    modules = [model, ranker_model, *autoencoders.values()]
    guards.require_frozen_perception(modules)
    guards.require_eval_mode(modules)
    state_before = {
        "perception": phase13b._state(model),
        "ranker": phase13b._state(ranker_model),
        **{
            family: phase13b._state(autoencoder)
            for family, autoencoder in autoencoders.items()
        },
    }

    timers = CudaStageTimers(device)
    ledger = phase13b.CallLedger()
    front = TimedFront(model, ledger, timers)
    ranker = TimedRanker(ranker_model, ledger, timers)
    timed_autoencoders = {
        family: TimedAutoencoder(family, autoencoder, ledger, timers)
        for family, autoencoder in autoencoders.items()
    }
    sample = sample_context["sample"]
    initial_row = sample["warmup"]["source_row"]
    calibration_cpu = {
        "intrinsic": base.data.model_intrinsic(initial_row),
        "extrinsic": base.data.camera_extrinsic(initial_row),
    }
    calibration = {name: value.to(device) for name, value in calibration_cpu.items()}
    tail = TimedTail(
        model=model,
        base=base,
        initial_row=initial_row,
        calibration=calibration,
        calibration_identity=sample["calibration"]["tensors"],
        ledger=ledger,
        timers=timers,
    )
    ue_codec = phase13b.AuditedUECodec(
        phase13b.AuditedWireCodec(ledger, "live_ue"), ledger
    )
    edge_codec = phase13b.AuditedEdgeCodec(
        phase13b.AuditedWireCodec(ledger, "live_edge"), ledger
    )
    ue = PreloadedSplitUERuntime(
        registry,
        front=front,
        ranker=ranker,
        ae_encoders=timed_autoencoders,
        device=device,
        codec=ue_codec,
        prepare_modules=False,
        startup_model_load_operations=5,
        startup_model_construction_operations=5,
    )
    edge = PreloadedSplitEdgeRuntime(
        registry,
        frozen_p025_tail=tail,
        ae_decoders=timed_autoencoders,
        tail_device=device,
        codec=edge_codec,
        output_serializer=tail.serialize,
        prepare_modules=False,
        startup_model_load_operations=4,
        startup_model_construction_operations=4,
    )
    inference = base.data.InferenceDataset(sample_context["dataset_root"], "train")
    _require(
        len(inference.rows) == contract.TRAIN_TOTAL_FRAMES,
        "deployable train inference row count drift",
    )
    loopback = TimedRawUdpLoopback(expected_messages)
    ledger.arm()
    return {
        "model": model,
        "base": base,
        "perception_binding": perception_binding,
        "ranker_model": ranker_model,
        "autoencoders": autoencoders,
        "timers": timers,
        "ledger": ledger,
        "front": front,
        "tail": tail,
        "ue": ue,
        "edge": edge,
        "edge_codec": edge_codec,
        "inference": inference,
        "loopback": loopback,
        "state_before": state_before,
        "device": device,
    }


def _load_input(
    runtime: Mapping[str, Any], selected_row: Mapping[str, Any]
) -> tuple[torch.Tensor, Mapping[str, str]]:
    dataset_index = int(selected_row["dataset_index"])
    fused, row, calibration_cpu = runtime["inference"][dataset_index]
    _require(row["sample_id"] == selected_row["sample_id"], "inference sample drift")
    _require(int(row["frame_id"]) == int(selected_row["frame_id"]), "inference frame drift")
    _require(tuple(fused.shape) == (7, 448, 768), "inference input shape drift")
    _require(fused.dtype is torch.float32 and bool(torch.isfinite(fused).all()), "input is not finite FP32")
    observed_calibration = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "sha256": _tensor_digest(value),
        }
        for name, value in sorted(calibration_cpu.items())
    }
    _require(
        observed_calibration
        == runtime["sample"]["calibration"]["tensors"],
        "loaded frame calibration differs from bound resident calibration",
    )
    input_7ch = fused.unsqueeze(0).to(runtime["device"])
    del fused, calibration_cpu
    return input_7ch, row


def _metadata_values(selected_row: Mapping[str, Any], message_id: int) -> tuple[int, int]:
    sequence_id = int(message_id)
    capture_timestamp_ns = int(
        round(float(selected_row["source_row"]["timestamp"]) * 1_000_000_000)
    )
    _require(sequence_id > 0 and capture_timestamp_ns >= 0, "invalid SFD1 metadata")
    return sequence_id, capture_timestamp_ns


def _validate_calls(profile: ActionProfile, calls: Mapping[str, int]) -> None:
    expected_ranker = 0 if profile.q_e4 == 0 else 1
    expected_ae = 0 if profile.family == "noAE" else 1
    for name in (
        "live_ue_zstd_compressions",
        "live_edge_zstd_decompressions",
        "phase13a_codec_encode",
        "phase13a_codec_inspect",
        "phase13a_codec_decode",
        "front",
        "tail",
        "service_record_serialization",
    ):
        _require(calls.get(name, 0) == 1, f"transaction call count drift: {name}")
    _require(calls.get("ranker", 0) == expected_ranker, "ranker call count drift")
    _require(
        calls.get(f"ae_encoder_{profile.family}", 0) == expected_ae,
        "selected AE encoder count drift",
    )
    _require(
        calls.get(f"ae_decoder_{profile.family}", 0) == expected_ae,
        "selected AE decoder count drift",
    )
    _require(
        sum(value for key, value in calls.items() if key.startswith("ae_encoder_"))
        == expected_ae,
        "wrong AE encoder route executed",
    )
    _require(
        sum(value for key, value in calls.items() if key.startswith("ae_decoder_"))
        == expected_ae,
        "wrong AE decoder route executed",
    )


def _transaction(
    runtime: Mapping[str, Any],
    *,
    profile: ActionProfile,
    selected_row: Mapping[str, Any],
    input_7ch: torch.Tensor,
    message_id: int,
    transaction_ordinal: int,
    warmup: bool,
) -> dict[str, Any] | None:
    ledger: phase13b.CallLedger = runtime["ledger"]
    timers: CudaStageTimers = runtime["timers"]
    ue_before = runtime["ue"].counters
    edge_before = runtime["edge"].counters
    calls_before = ledger.snapshot("live")
    sequence_id, capture_timestamp_ns = _metadata_values(selected_row, message_id)
    timers.reset()

    capture_started = time.perf_counter_ns()
    with phase13b._hot_path_guard():
        with ledger.section("live"):
            prepared = runtime["ue"].prepare(
                profile.action_id,
                input_7ch,
                sequence_id=sequence_id,
                capture_timestamp_ns=capture_timestamp_ns,
            )
            ue_call_finished = time.perf_counter_ns()
            runtime["front"].release_c2()
            delivery = runtime["loopback"].roundtrip(
                prepared.wire_bytes, message_id=message_id
            )
            edge_call_started = time.perf_counter_ns()
            result = runtime["edge"].process(
                delivery.payload, transmitted_action_id=profile.action_id
            )
            capture_finished = time.perf_counter_ns()
            inspected, decoded = runtime["edge_codec"].take()
            snapshot = runtime["tail"].take_snapshot()

    outer = unpack_envelope(prepared.wire_bytes)
    _require(
        outer.protocol_version == PROTOCOL_VERSION
        and outer.action_id == profile.action_id
        and result.metadata.action_id == profile.action_id,
        "action/SFD1/catalog identity drift",
    )
    _require(
        outer.sequence_id == sequence_id == result.metadata.sequence_id
        and outer.capture_timestamp_ns
        == capture_timestamp_ns
        == result.metadata.capture_timestamp_ns,
        "SFD1 sequence or capture timestamp drift",
    )
    _require(
        outer.inner_payload_length
        == prepared.inner_payload_bytes
        == result.scientific_inner_payload_bytes,
        "scientific inner byte count drift",
    )
    _require(
        outer.control_overhead_bytes
        == prepared.outer_envelope_bytes
        == result.framing_control_overhead_bytes
        == HEADER_BYTES,
        "SFD1 control overhead drift",
    )
    _require(
        outer.total_transmitted_bytes
        == prepared.total_transmitted_bytes
        == result.total_received_bytes
        == delivery.sfd1_application_bytes,
        "SFD1 total byte count drift",
    )
    _require(
        inspected.identity.family == profile.family
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
        "inner header/catalog agreement drift",
    )
    _require(
        decoded.finite
        and decoded.device == runtime["device"]
        and decoded.c2.dtype is torch.float32
        and tuple(decoded.c2.shape) == (256, 112, 192)
        and bool(torch.isfinite(decoded.c2).all()),
        "reconstructed C2 contract drift",
    )
    _require(
        result.perception is snapshot.perception
        and result.serialized_output == snapshot.serialized_records,
        "p025 output/serialization handoff drift",
    )
    _require(snapshot.output_tensor_count > 0, "p025 tail produced no finite tensors")
    _require(snapshot.records is not None, "p025 service records missing")

    calls = _counter_delta(ledger.snapshot("live"), calls_before)
    _validate_calls(profile, calls)
    ue_delta = _operation_delta(runtime["ue"].counters, ue_before)
    edge_delta = _operation_delta(runtime["edge"].counters, edge_before)
    _require(
        ue_delta.get("frames_attempted") == 1
        and ue_delta.get("frames_completed") == 1,
        "UE transaction counters drift",
    )
    _require(
        edge_delta.get("frames_attempted") == 1
        and edge_delta.get("frames_completed") == 1
        and edge_delta.get("tail_dispatches") == 1,
        "edge transaction counters drift",
    )
    _require(
        runtime["ue"].counters.hot_path_model_load_operations == 0
        and runtime["ue"].counters.hot_path_model_construction_operations == 0
        and runtime["edge"].counters.hot_path_model_load_operations == 0
        and runtime["edge"].counters.hot_path_model_construction_operations == 0,
        "hot-path model load/construction/device move occurred",
    )

    ue_timing = _trace_durations(prepared.timing, UE_STAGES)
    edge_timing = _trace_durations(result.timing, EDGE_STAGES)
    edge_total = _trace_boundary(result.timing, "total_edge_processing")
    edge_zstd = _trace_boundary(result.timing, "zstd_decompression")
    envelope_validate_ms = (
        edge_zstd.started_monotonic_ns - edge_total.started_monotonic_ns
    ) / 1_000_000.0
    _require(envelope_validate_ms >= 0.0, "negative envelope validation time")
    exclusive_ns = {
        "capture_partition_ue_call_ms": ue_call_finished - capture_started,
        "capture_partition_ue_to_first_send_ms": (
            delivery.first_send_started_perf_counter_ns - ue_call_finished
        ),
        "localhost_delivery_ms": (
            delivery.reassembly_completed_perf_counter_ns
            - delivery.first_send_started_perf_counter_ns
        ),
        "capture_partition_reassembly_to_edge_call_ms": (
            edge_call_started - delivery.reassembly_completed_perf_counter_ns
        ),
        "capture_partition_edge_call_ms": capture_finished - edge_call_started,
    }
    _require(
        all(value >= 0 for value in exclusive_ns.values()),
        f"negative exclusive capture partition: {exclusive_ns}",
    )
    _require(
        sum(exclusive_ns.values()) == capture_finished - capture_started,
        "exclusive capture partition does not equal end-to-end interval",
    )
    record = {
        "transaction_ordinal": transaction_ordinal,
        "frame_ordinal": int(selected_row.get("sample_ordinal", -1)),
        "sample_id": selected_row["sample_id"],
        "episode_id": selected_row["episode_id"],
        "frame_id": int(selected_row["frame_id"]),
        "action_id": profile.action_id,
        "family": profile.family,
        "quantizer": profile.quantizer,
        "q_e4": profile.q_e4,
        "front_gpu_ms": timers.value("front_gpu_ms", bypass=False),
        "ranker_gpu_ms": timers.value(
            "ranker_gpu_ms", bypass=profile.q_e4 == 0
        ),
        "ae_encode_gpu_ms": timers.value(
            "ae_encode_gpu_ms", bypass=profile.family == "noAE"
        ),
        "quantize_pack_ms": ue_timing["quantize_pack"],
        "zstd_compress_ms": ue_timing["zstd_compression"],
        "sfd1_fragment_send_ms": delivery.fragment_send_ms,
        "ue_prepare_ms": ue_timing["total_ue_preparation"],
        "localhost_delivery_ms": delivery.localhost_delivery_ms,
        "envelope_validate_ms": envelope_validate_ms,
        "zstd_decompress_ms": edge_timing["zstd_decompression"],
        "dequant_scatter_ms": edge_timing["unpack_dequantize"],
        "ae_decode_gpu_ms": timers.value(
            "ae_decode_gpu_ms", bypass=profile.family == "noAE"
        ),
        "tail_gpu_ms": timers.value("tail_gpu_ms", bypass=False),
        "p025_serialize_ms": edge_timing["output_serialization"],
        "back_ms": edge_timing["total_edge_processing"],
        "capture_partition_ue_call_ms": exclusive_ns[
            "capture_partition_ue_call_ms"
        ] / 1_000_000.0,
        "capture_partition_ue_to_first_send_ms": exclusive_ns[
            "capture_partition_ue_to_first_send_ms"
        ] / 1_000_000.0,
        "capture_partition_reassembly_to_edge_call_ms": exclusive_ns[
            "capture_partition_reassembly_to_edge_call_ms"
        ] / 1_000_000.0,
        "capture_partition_edge_call_ms": exclusive_ns[
            "capture_partition_edge_call_ms"
        ] / 1_000_000.0,
        "capture_to_edge_result_ms": (
            capture_finished - capture_started
        ) / 1_000_000.0,
        "scientific_inner_bytes": prepared.inner_payload_bytes,
        "sfd1_bytes": prepared.total_transmitted_bytes,
        "udp_application_bytes": delivery.udp_application_bytes,
        "estimated_on_wire_bytes": delivery.estimated_on_wire_bytes,
        "datagram_count": delivery.datagrams,
        "detection_count": int(snapshot.perception["scores"].numel()),
        "service_record_count": len(snapshot.records),
        "duplicate_datagrams": delivery.duplicate_datagrams,
        "compression_calls": calls["live_ue_zstd_compressions"],
        "decompression_calls": calls["live_edge_zstd_decompressions"],
        "ranker_calls": calls.get("ranker", 0),
        "encoder_calls": sum(
            value for key, value in calls.items() if key.startswith("ae_encoder_")
        ),
        "decoder_calls": sum(
            value for key, value in calls.items() if key.startswith("ae_decoder_")
        ),
        "tail_calls": calls["tail"],
        "delivered": 1,
        "decoded": 1,
        "completed": 1,
    }
    _require(
        all(
            math.isfinite(float(record[field])) and float(record[field]) >= 0.0
            for field in (*LATENCY_FIELDS, *PAYLOAD_FIELDS)
        ),
        "non-finite or negative scalar measurement",
    )
    if profile.q_e4 == 0:
        _require(record["ranker_gpu_ms"] == 0.0, "q=0 did not record exact ranker bypass")
    if profile.family == "noAE":
        _require(
            record["ae_encode_gpu_ms"] == 0.0
            and record["ae_decode_gpu_ms"] == 0.0,
            "noAE did not record exact AE bypass",
        )
    del prepared, delivery, result, inspected, decoded, snapshot, outer
    return None if warmup else record


def _nearest_rank(values: Sequence[int | float], probability: float) -> int | float:
    _require(bool(values), "cannot summarize an empty sample")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _summarize_profile(
    profile: ActionProfile,
    measurements: Sequence[Mapping[str, Any]],
    *,
    run_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(len(measurements) == FRAMES, "cannot summarize an incomplete profile")
    latency = {
        field: {
            "median": float(_nearest_rank([float(row[field]) for row in measurements], 0.50)),
            "p95": float(_nearest_rank([float(row[field]) for row in measurements], 0.95)),
        }
        for field in LATENCY_FIELDS
    }
    payload = {
        field: {
            "median": int(_nearest_rank([int(row[field]) for row in measurements], 0.50)),
            "p95": int(_nearest_rank([int(row[field]) for row in measurements], 0.95)),
        }
        for field in PAYLOAD_FIELDS
    }
    call_counts = {
        field: sum(int(row[field]) for row in measurements)
        for field in (
            "ranker_calls",
            "encoder_calls",
            "decoder_calls",
            "compression_calls",
            "decompression_calls",
            "tail_calls",
        )
    }
    expected_ranker = 0 if profile.q_e4 == 0 else FRAMES
    expected_ae = 0 if profile.family == "noAE" else FRAMES
    _require(call_counts["ranker_calls"] == expected_ranker, "profile ranker count drift")
    _require(call_counts["encoder_calls"] == expected_ae, "profile encoder count drift")
    _require(call_counts["decoder_calls"] == expected_ae, "profile decoder count drift")
    _require(call_counts["compression_calls"] == FRAMES, "profile compression count drift")
    _require(call_counts["decompression_calls"] == FRAMES, "profile decompression count drift")
    _require(call_counts["tail_calls"] == FRAMES, "profile tail count drift")
    measured_wall_seconds = sum(
        float(row["capture_to_edge_result_ms"]) for row in measurements
    ) / 1000.0
    _require(measured_wall_seconds > 0.0, "profile measured wall time is not positive")
    counts = {
        "attempted": FRAMES,
        "delivered": sum(int(row["delivered"]) for row in measurements),
        "decoded": sum(int(row["decoded"]) for row in measurements),
        "completed": sum(int(row["completed"]) for row in measurements),
        "duplicate_or_corrupt_messages": sum(
            int(row["duplicate_datagrams"]) for row in measurements
        ),
        "delivery_rate": sum(int(row["delivered"]) for row in measurements) / FRAMES,
        "completion_rate": sum(int(row["completed"]) for row in measurements) / FRAMES,
    }
    _require(
        counts["delivered"] == FRAMES
        and counts["decoded"] == FRAMES
        and counts["completed"] == FRAMES
        and counts["duplicate_or_corrupt_messages"] == 0,
        "profile delivery/completion integrity drift",
    )
    summary = {
        "profile": _profile_record(profile),
        "counts": counts,
        "latency_ms": latency,
        "payload_bytes_or_counts": payload,
        "call_counts": call_counts,
        "detection_count_median": int(
            _nearest_rank(
                [int(row["detection_count"]) for row in measurements], 0.50
            )
        ),
        "profile_measured_wall_seconds": measured_wall_seconds,
        "measured_sequential_throughput_frames_per_second": (
            FRAMES / measured_wall_seconds
        ),
    }
    record = _seal(
        {
            "schema": PROFILE_RECORD_SCHEMA,
            "run_manifest_sha256": run_manifest_sha256,
            "profile": _profile_record(profile),
            "counts": counts,
            "summary": summary,
            "measurements": list(measurements),
        },
        "profile_record_sha256",
    )
    return record, summary


def _summary_csv(summaries: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "action_id",
        "profile_id",
        "family",
        "quantizer",
        "q_e4",
        "attempted",
        "delivered",
        "decoded",
        "completed",
        "delivery_rate",
        "completion_rate",
        "profile_measured_wall_seconds",
        "measured_sequential_throughput_frames_per_second",
        "detection_count_median",
        "ranker_calls",
        "encoder_calls",
        "decoder_calls",
        "compression_calls",
        "decompression_calls",
        "tail_calls",
    ]
    fields.extend(
        f"{field}_{stat}"
        for field in LATENCY_FIELDS
        for stat in ("median", "p95")
    )
    fields.extend(
        f"{field}_{stat}"
        for field in PAYLOAD_FIELDS
        for stat in ("median", "p95")
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for summary in summaries:
        profile = summary["profile"]
        row = {
            "action_id": profile["action_id"],
            "profile_id": profile["profile_id"],
            "family": profile["family"],
            "quantizer": profile["quantizer"],
            "q_e4": profile["q_e4"],
            **summary["counts"],
            "profile_measured_wall_seconds": summary[
                "profile_measured_wall_seconds"
            ],
            "measured_sequential_throughput_frames_per_second": summary[
                "measured_sequential_throughput_frames_per_second"
            ],
            "detection_count_median": summary["detection_count_median"],
            **summary["call_counts"],
        }
        for field in LATENCY_FIELDS:
            for stat in ("median", "p95"):
                row[f"{field}_{stat}"] = summary["latency_ms"][field][stat]
        for field in PAYLOAD_FIELDS:
            for stat in ("median", "p95"):
                row[f"{field}_{stat}"] = summary["payload_bytes_or_counts"][field][stat]
        writer.writerow({name: row[name] for name in fields})
    return stream.getvalue()


def _report(document: Mapping[str, Any]) -> str:
    integrity = document["integrity"]
    lines = [
        "# Phase 13C 36-profile × 300-frame localhost replay measurement",
        "",
        f"Status: `{document['terminal']}`.",
        "",
        "This is a live CUDA/localhost software replay measurement. Sensor capture and disk input loading are excluded. Localhost timing is not OAI, RFsim, radio, or Raspberry Pi latency.",
        "",
        f"The immutable Phase-4 fit sample is `{document['sample_manifest_sha256']}`; the preregistered rotated schedule is `{document['schedule_manifest_sha256']}`.",
        "",
        "Perception accuracy was not rescored. No holdout, validation, or test sensor frame and no box, semantic-GT, AVO, depth-GT, or evaluation record was opened.",
        "",
        "| action | family | quantizer | q | complete | E2E median ms | E2E p95 ms | inner median B | SFD1 median B | on-wire median B | datagrams median | seq. fps |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in document["profile_summaries"]:
        profile = summary["profile"]
        latency = summary["latency_ms"]["capture_to_edge_result_ms"]
        payload = summary["payload_bytes_or_counts"]
        lines.append(
            f"| {profile['action_id']} | {profile['family']} | {profile['quantizer']} | "
            f"{profile['q']:.2f} | {summary['counts']['completed']} | "
            f"{latency['median']:.3f} | {latency['p95']:.3f} | "
            f"{payload['scientific_inner_bytes']['median']} | "
            f"{payload['sfd1_bytes']['median']} | "
            f"{payload['estimated_on_wire_bytes']['median']} | "
            f"{payload['datagram_count']['median']} | "
            f"{summary['measured_sequential_throughput_frames_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"All {integrity['completed_profile_frames']:,} transactions completed with exact reassembly, one compression, one decompression and one frozen tail call each. Frozen perception, ranker, and AE states were unchanged.",
            "",
            "Component timings are diagnostic boundaries. `localhost_delivery_ms` overlaps the send portion of `sfd1_fragment_send_ms`; no double-counted component sum is presented. End-to-end time is measured independently from frozen-front start through serialized edge return.",
            "",
            "Remaining blockers are 100-MHz RFsim calibration and the 16-cell OAI pilot. The 288-cell OAI campaign was not launched.",
        ]
    )
    return "\n".join(lines) + "\n"


def _state_after(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    after = {
        "perception": phase13b._state(runtime["model"]),
        "ranker": phase13b._state(runtime["ranker_model"]),
        **{
            family: phase13b._state(autoencoder)
            for family, autoencoder in runtime["autoencoders"].items()
        },
    }
    equal = {
        name: runtime["state_before"][name] == after[name]
        for name in runtime["state_before"]
    }
    _require(all(equal.values()), "frozen model/ranker/AE state changed")
    _require(
        all(
            parameter.grad is None
            for module in (
                runtime["model"],
                runtime["ranker_model"],
                *runtime["autoencoders"].values(),
            )
            for parameter in module.parameters()
        ),
        "a frozen parameter received a gradient",
    )
    return after, equal


def _close_runtime(runtime: Mapping[str, Any] | None) -> None:
    if runtime is None:
        return
    loopback = runtime.get("loopback")
    if loopback is not None:
        loopback.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-13C 36-profile by 300-fit-frame CUDA/localhost measurement"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only complete hash-valid per-profile records in the bound run directory",
    )
    args = parser.parse_args()

    runtime: dict[str, Any] | None = None
    output: Path | None = None
    operation = "preflight_interpreter"
    current_action: int | None = None
    current_sample: str | None = None
    completed_records: dict[int, dict[str, Any]] = {}
    progress: dict[int, dict[str, int]] = {}
    process_started = time.perf_counter()
    try:
        _require(
            Path(sys.executable).resolve(strict=True)
            == Path("/usr/bin/python3").resolve(strict=True),
            f"Phase-13C requires /usr/bin/python3, observed {sys.executable}",
        )
        operation = "preflight_git"
        git = _verify_git_state()
        operation = "preflight_phase13a"
        phase13a_binding = phase13b._verify_phase13a_terminal()
        operation = "preflight_phase13b"
        phase13b_binding = _verify_phase13b_artifacts()
        operation = "preflight_registry"
        registry = SplitActionRegistry.from_runtime_binding(
            verify_runtime_artifacts=True
        )
        profiles = _select_profiles(registry)
        by_action = {profile.action_id: profile for profile in profiles}
        operation = "preflight_historical_ae"
        historical = phase11b.phase11b_preflight()
        operation = "preflight_fit_sample"
        sample, sample_context = _construct_sample()
        operation = "preflight_schedule"
        schedule = _construct_schedule(sample, profiles)
        _verify_seal(sample, "sample_manifest_sha256")
        _verify_seal(schedule, "schedule_manifest_sha256")
        _require(
            not torch.cuda.is_initialized(),
            "CUDA initialized before sample and schedule hashes were complete",
        )
        operation = "preflight_output"
        output_candidate = _output_candidate(resume=args.resume)
        operation = "preflight_cuda_and_workload"
        gpu = phase13b._gpu_preflight()
        _require(gpu["device_name"] == DEVICE_NAME, "cuda:0 device identity drift")
        device = torch.device("cuda:0")

        existing_count = 0
        if args.resume:
            manifest_path = output_candidate / "run_manifest.json"
            _require(manifest_path.is_file(), "resume output lacks run manifest")
            provisional_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _verify_seal(provisional_manifest, "run_manifest_sha256")
            existing_profiles = sorted((output_candidate / "profiles").glob("*.json"))
            existing_count = len(existing_profiles)
            _require(existing_count <= PROFILE_COUNT, "too many resume profile records")
        expected_messages = PROFILE_COUNT + FRAMES * (PROFILE_COUNT - existing_count)
        operation = "preload_frozen_runtime"
        runtime = _load_models(
            device=device,
            registry=registry,
            historical=historical,
            sample_context={**sample_context, "sample": sample},
            expected_messages=expected_messages,
        )
        runtime["sample"] = sample
        runtime["profiles"] = profiles
        operation = "create_or_bind_output"
        output = output_candidate if args.resume else _create_output(output_candidate)
        manifest_expected = _build_manifest(
            git=git,
            phase13a=phase13a_binding,
            phase13b_binding=phase13b_binding,
            registry=registry,
            profiles=profiles,
            sample=sample,
            schedule=schedule,
            gpu=gpu,
            historical=historical,
            perception_binding=runtime["perception_binding"],
            state_before=runtime["state_before"],
            loopback=runtime["loopback"],
        )
        manifest = _load_or_write_manifest(output, manifest_expected, resume=args.resume)
        manifest_sha256 = str(manifest["run_manifest_sha256"])
        sample_ids = [row["sample_id"] for row in sample["selected_rows"]]
        completed_records = _load_resume_records(
            output,
            manifest_sha256=manifest_sha256,
            profiles=profiles,
            sample_ids=sample_ids,
        )
        _require(
            len(completed_records) == existing_count,
            "resume durable record inventory changed after preflight",
        )
        progress = {
            profile.action_id: {
                "attempted": FRAMES if profile.action_id in completed_records else 0,
                "delivered": FRAMES if profile.action_id in completed_records else 0,
                "decoded": FRAMES if profile.action_id in completed_records else 0,
                "completed": FRAMES if profile.action_id in completed_records else 0,
            }
            for profile in profiles
        }

        operation = "warmup_36_profiles"
        warmup = sample["warmup"]
        warmup_input, _warmup_row = _load_input(runtime, warmup)
        runtime["tail"].bind_row(warmup["source_row"])
        for warmup_ordinal, profile in enumerate(profiles, start=1):
            current_action = profile.action_id
            current_sample = warmup["sample_id"]
            result = _transaction(
                runtime,
                profile=profile,
                selected_row=warmup,
                input_7ch=warmup_input,
                message_id=warmup_ordinal,
                transaction_ordinal=-1,
                warmup=True,
            )
            _require(result is None, "warm-up transaction leaked a scientific row")
        torch.cuda.synchronize(device)
        del warmup_input, _warmup_row
        torch.cuda.reset_peak_memory_stats(device)

        operation = "measured_36x300_transactions"
        measurement_started = time.perf_counter()
        accumulated: dict[int, list[dict[str, Any]]] = {
            profile.action_id: []
            for profile in profiles
            if profile.action_id not in completed_records
        }
        transaction_ordinal = 0
        next_message_id = PROFILE_COUNT + 1
        for frame_schedule, selected_row in zip(
            schedule["frames"], sample["selected_rows"], strict=True
        ):
            _require(
                frame_schedule["sample_id"] == selected_row["sample_id"],
                "schedule/sample binding drift",
            )
            input_7ch, _inference_row = _load_input(runtime, selected_row)
            runtime["tail"].bind_row(selected_row["source_row"])
            for action_id in frame_schedule["ordered_action_ids"]:
                profile = by_action[action_id]
                current_action = action_id
                current_sample = selected_row["sample_id"]
                if action_id in completed_records:
                    transaction_ordinal += 1
                    continue
                progress[action_id]["attempted"] += 1
                row = _transaction(
                    runtime,
                    profile=profile,
                    selected_row=selected_row,
                    input_7ch=input_7ch,
                    message_id=next_message_id,
                    transaction_ordinal=transaction_ordinal,
                    warmup=False,
                )
                _require(row is not None, "measured transaction produced no scalar row")
                next_message_id += 1
                transaction_ordinal += 1
                progress[action_id]["delivered"] += int(row["delivered"])
                progress[action_id]["decoded"] += int(row["decoded"])
                progress[action_id]["completed"] += int(row["completed"])
                accumulated[action_id].append(row)
                if len(accumulated[action_id]) == FRAMES:
                    record, _summary = _summarize_profile(
                        profile,
                        accumulated[action_id],
                        run_manifest_sha256=manifest_sha256,
                    )
                    path = _profile_path(output, action_id)
                    _atomic_create_json(path, record)
                    _validate_profile_record(
                        record,
                        manifest_sha256=manifest_sha256,
                        profile=profile,
                        sample_ids=sample_ids,
                    )
                    completed_records[action_id] = record
                    del accumulated[action_id]
            del input_7ch, _inference_row
        torch.cuda.synchronize(device)
        measurement_wall_seconds = time.perf_counter() - measurement_started
        _require(transaction_ordinal == TRANSACTIONS, "schedule cursor did not reach 10,800")
        _require(not accumulated, "one or more profiles did not reach 300 rows")
        _require(len(completed_records) == PROFILE_COUNT, "not all durable profiles exist")

        operation = "final_integrity"
        ordered_records: list[dict[str, Any]] = []
        durable_files: list[dict[str, Any]] = []
        for profile in profiles:
            path = _profile_path(output, profile.action_id)
            document = json.loads(path.read_text(encoding="utf-8"))
            _validate_profile_record(
                document,
                manifest_sha256=manifest_sha256,
                profile=profile,
                sample_ids=sample_ids,
            )
            ordered_records.append(document)
            durable_files.append(
                {
                    "path": str(path.relative_to(output)),
                    "sha256": sha256_file(path),
                    "profile_record_sha256": document["profile_record_sha256"],
                }
            )
        summaries = [record["summary"] for record in ordered_records]
        total_counts = {
            name: sum(int(summary["counts"][name]) for summary in summaries)
            for name in ("attempted", "delivered", "decoded", "completed")
        }
        _require(
            total_counts
            == {
                "attempted": TRANSACTIONS,
                "delivered": TRANSACTIONS,
                "decoded": TRANSACTIONS,
                "completed": TRANSACTIONS,
            },
            "global 10,800 transaction accounting drift",
        )
        total_calls = {
            name: sum(int(summary["call_counts"][name]) for summary in summaries)
            for name in (
                "ranker_calls",
                "encoder_calls",
                "decoder_calls",
                "compression_calls",
                "decompression_calls",
                "tail_calls",
            )
        }
        _require(
            total_calls
            == {
                "ranker_calls": 7_200,
                "encoder_calls": 8_100,
                "decoder_calls": 8_100,
                "compression_calls": TRANSACTIONS,
                "decompression_calls": TRANSACTIONS,
                "tail_calls": TRANSACTIONS,
            },
            f"global call count drift: {total_calls}",
        )
        state_after, frozen_equal = _state_after(runtime)
        process_wall_seconds = time.perf_counter() - process_started
        document = {
            "schema": SCHEMA,
            "terminal": TERMINAL,
            "status": "PHASE13C_36X300_LOCALHOST_MEASUREMENT_COMPLETE",
            "implementation_commit": git["head"],
            "run_manifest": {
                "path": "run_manifest.json",
                "sha256": sha256_file(output / "run_manifest.json"),
                "sealed_sha256": manifest_sha256,
            },
            "sample_manifest_sha256": sample["sample_manifest_sha256"],
            "selected_sample_id_sha256": sample["selected_sample_id_sha256"],
            "schedule_manifest_sha256": schedule["schedule_manifest_sha256"],
            "flat_schedule_sha256": schedule["flat_schedule_sha256"],
            "profile_inventory": [_profile_record(profile) for profile in profiles],
            "profile_summaries": summaries,
            "durable_profile_records": durable_files,
            "integrity": {
                **{f"{name}_profile_frames": value for name, value in total_counts.items()},
                "profiles": PROFILE_COUNT,
                "frames_per_profile": FRAMES,
                "complete_sfd1_reassembly_rate": 1.0,
                "duplicate_or_corrupt_messages": 0,
                "action_catalog_header_agreement": True,
                "q_and_keep_counts_exact": True,
                "q0_ranker_bypass": True,
                "ae_noae_routing_exact": True,
                "call_counts": total_calls,
                "reconstructed_c2": {
                    "shape": [256, 112, 192],
                    "dtype": "float32",
                    "device": "cuda:0",
                    "finite_every_transaction": True,
                },
                "p025_outputs_finite_and_schema_valid": True,
                "hot_path_model_load_construct_move_eval_or_mutate": 0,
                "frozen_state_before": runtime["state_before"],
                "frozen_state_after": state_after,
                "frozen_state_equal": frozen_equal,
                "all_gradients_absent": True,
                "calibration_resident_not_transmitted": True,
                "retained_feature_payload_datagram_prediction_blobs": 0,
            },
            "scope": {
                "fit_sensor_frames_read": FRAMES + 1,
                "holdout_sensor_frames_read": 0,
                "validation_sensor_frames_read": 0,
                "test_sensor_frames_read": 0,
                "gt_or_evaluation_records_read": 0,
                "perception_accuracy_rescored": False,
                "training_or_tuning": False,
                "carla_oai_or_rfsim_used": False,
                "oai_campaign_launched": False,
                "localhost_is_not_oai_or_pi_latency": True,
            },
            "resume": {
                "requested": bool(args.resume),
                "reused_valid_profile_records": existing_count,
                "new_profile_records": PROFILE_COUNT - existing_count,
            },
            "resources": {
                "measurement_wall_seconds_including_input_load_and_record_writes": measurement_wall_seconds,
                "total_process_wall_seconds": process_wall_seconds,
                "peak_allocated_bytes_after_warmup_reset": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes_after_warmup_reset": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
            "claims": {
                "measurement": "live CUDA/localhost software replay floor",
                "not_claimed": [
                    "perception accuracy",
                    "OAI bandwidth or latency",
                    "radio latency",
                    "Raspberry Pi latency",
                ],
                "remaining_blockers": [
                    "100-MHz RFsim calibration",
                    "16-cell OAI pilot",
                ],
            },
        }
        operation = "write_compact_final_evidence"
        qualification_hash = _atomic_create_json(output / "qualification.json", document)
        csv_hash = _atomic_create_text(output / "profile_summary.csv", _summary_csv(summaries))
        report_hash = _atomic_create_text(output / "REPORT.md", _report(document))
        terminal_hash = _atomic_create_text(
            output / TERMINAL, f"{TERMINAL} {qualification_hash}\n"
        )
        print(
            json.dumps(
                {
                    "terminal": TERMINAL,
                    "output": str(output.relative_to(_root())),
                    "qualification_sha256": qualification_hash,
                    "profile_summary_csv_sha256": csv_hash,
                    "report_sha256": report_hash,
                    "terminal_sha256": terminal_hash,
                    "profiles": PROFILE_COUNT,
                    "completed_profile_frames": TRANSACTIONS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        if output is not None and output.exists():
            failure = {
                "schema": "scenesense.splitfusion_phase13c_failure.v1",
                "status": "FAILED_NO_RETRY_AUTHORIZED",
                "operation": operation,
                "action_id": current_action,
                "sample_id": current_sample,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "durable_action_ids": sorted(completed_records),
                "progress": {str(key): value for key, value in sorted(progress.items())},
            }
            failure_path = output / "FAILURE.json"
            if not failure_path.exists():
                try:
                    _atomic_create_json(failure_path, failure)
                except Exception:
                    pass
        raise
    finally:
        _close_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
