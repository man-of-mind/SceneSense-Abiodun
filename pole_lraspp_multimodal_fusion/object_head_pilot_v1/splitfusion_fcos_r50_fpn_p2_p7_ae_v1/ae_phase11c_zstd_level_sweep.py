"""Phase-11C lossless zstd level sweep over real 72-profile inner payloads.

This bounded host benchmark reuses the Phase-7 128-frame fit sample and the
registered UINT8/UINT6/UINT4 payload writers.  It does not choose a production
level, decode an edge tail, score perception, or open validation/test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import zstandard

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
    phase7_zstd_measurement as phase7,
    uint8_codec,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    collate_batch,
    encode_front,
    load_frozen_perception,
    sha256_file,
)
from . import (
    ae_contract,
    ae_phase11b_gpu_qualification as phase11b,
    ae_training_common as common,
    ae_uint8_transport,
    lowbit_transport,
)
from .ae_model import SplitFeatureAE


EXECUTE_TOKEN = "HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP"
TERMINAL = "HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE"
SCHEMA = "splitfusion_fcos_phase11c_zstd_level_sweep_v1"
OUTPUT_RELPATH = (
    "experiments/splitfusion_fcos_ae_v1/"
    "20260903_phase11c_zstd_level_sweep"
)

Q_VALUES = (0.00, 0.30, 0.50, 0.70, 0.90, 0.98)
LEVELS = (1, 3, 5)
QUANTIZERS = (("UINT8", 8), ("UINT6", 6), ("UINT4", 4))
FAMILIES = (
    ("noAE", ae_contract.AE_FAMILY_NOAE, None),
    ("AE128", ae_contract.AE_FAMILY_AE128, 128),
    ("AE64", ae_contract.AE_FAMILY_AE64, 64),
    ("AE32", ae_contract.AE_FAMILY_AE32, 32),
)
FRAMES = phase7.SAMPLE_FRAMES
PROFILES = len(FAMILIES) * len(QUANTIZERS) * len(Q_VALUES)
REQUIRED_ROUND_TRIPS = FRAMES * PROFILES * len(LEVELS)

PHASE11B_ARTIFACTS = {
    "report_json": {
        "path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase11b_lowbit_gpu_qualification/"
            "phase11b_lowbit_gpu_qualification.json"
        ),
        "sha256": "379aa07148e3e47384cfbebbe0ede5990c07f11b8a4bdef056d6a533cee5fc01",
    },
    "report_markdown": {
        "path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase11b_lowbit_gpu_qualification/"
            "PHASE11B_LOWBIT_GPU_QUALIFICATION_REPORT.md"
        ),
        "sha256": "230f4ec30177c4aadb349084721f6515c319285c24ef59a0e4864063dcdfa768",
    },
    "terminal": {
        "path": (
            "experiments/splitfusion_fcos_ae_v1/"
            "20260903_phase11b_lowbit_gpu_qualification/"
            "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED"
        ),
        "sha256": "83f41560a3327c4207834f5725e5e313ceb6b3e0f9e22ea1f8c37b6dcf0b56e2",
    },
}

PHASE7_SAMPLE_ARTIFACTS = {
    "report_json": {
        "path": (
            "experiments/splitfusion_fcos_hybrid_q_v1/"
            "20260902_212403_phase7_zstd_measurement/phase7_zstd_measurement.json"
        ),
        "sha256": "bfee4a9e6328e588143cf56f7483d5557de0a2e60dddc80d1a4b83a864d2a5ea",
    },
    "terminal": {
        "path": (
            "experiments/splitfusion_fcos_hybrid_q_v1/"
            "20260902_212403_phase7_zstd_measurement/"
            "HYBRID_Q_PHASE7_ZSTD_MEASUREMENT_COMPLETE"
        ),
        "sha256": "8e2ef6e1ac1c03cf885cf425429b514a5365d5b25ea8189b86c74e32928a8779",
    },
}

TRANSPORT_SOURCES = {
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/uint8_codec.py": (
        "b6f07723860821e93c0dc2aeec456073617942b5ed8446f4555fe87b2c7ca803"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/uint8_zstd_transport.py": (
        "db1f79e3786f2d5a12e367b688fd79401bc5e5c8f10523a7bbf7418f14aaa11d"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/zstd_transport.py": (
        "57d1846b3fdc4084266e5a8adcc7abf99556ed2b187befec729003fcdb77edec"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_uint8_transport.py": (
        "4162b162d554f764332d469b6ca7b5038298e77a563a9de73c1622cd99531423"
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py": (
        "c708389982b10968978002d2d8423984d857229021149d51ec0d135619d69f12"
    ),
}

ZSTD_OPTIONS = {
    "threads": 0,
    "dict_data": None,
    "write_checksum": True,
    "write_content_size": True,
    "write_dict_id": False,
}
ZSTD_SETTINGS = {
    **ZSTD_OPTIONS,
    "one_independent_frame_per_camera_payload": True,
}


@dataclass(frozen=True)
class PayloadDescriptor:
    family: str
    family_id: int
    quantizer: str
    bit_width: int
    q: float
    q_e4: int
    keep_count: int
    bottleneck: int | None
    routing_tag: int
    inner: bytes

    @property
    def profile_key(self) -> str:
        return (
            f"{self.family}_{self.quantizer}_q{self.q_e4:05d}"
        )


@dataclass
class Measurements:
    pre_zstd_bytes: list[int]
    compressed_bytes: list[int]
    compression_ns: list[int]
    decompression_ns: list[int]
    inner_digest: Any
    round_trips: int = 0
    header_checks: int = 0


def _repository_path(relative: str) -> Path:
    return (contract.repository_root() / relative).resolve(strict=True)


def _require_exact_hash(item: Mapping[str, str]) -> dict[str, str]:
    path = _repository_path(item["path"])
    digest = sha256_file(path)
    if digest != item["sha256"]:
        raise guards.HybridQConfigError(
            f"Phase-11C input hash drift: {item['path']}"
        )
    return {"path": item["path"], "sha256": digest}


def _verify_phase11b_artifacts() -> dict[str, Any]:
    bound = {name: _require_exact_hash(item) for name, item in PHASE11B_ARTIFACTS.items()}
    document = json.loads(_repository_path(PHASE11B_ARTIFACTS["report_json"]["path"]).read_text(encoding="utf-8"))
    if document.get("terminal") != phase11b.TERMINAL:
        raise guards.HybridQConfigError("Phase-11B report terminal drift")
    terminal = _repository_path(PHASE11B_ARTIFACTS["terminal"]["path"]).read_text(
        encoding="utf-8"
    )
    expected = f"{phase11b.TERMINAL} {PHASE11B_ARTIFACTS['report_json']['sha256']}\n"
    if terminal != expected:
        raise guards.HybridQConfigError("Phase-11B terminal no longer binds its report")
    return {"artifacts": bound, "terminal": document["terminal"]}


def _verify_phase7_sample_artifacts() -> tuple[dict[str, Any], Mapping[str, Any]]:
    bound = {name: _require_exact_hash(item) for name, item in PHASE7_SAMPLE_ARTIFACTS.items()}
    document = json.loads(
        _repository_path(PHASE7_SAMPLE_ARTIFACTS["report_json"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    terminal = _repository_path(PHASE7_SAMPLE_ARTIFACTS["terminal"]["path"]).read_text(
        encoding="utf-8"
    )
    if document.get("terminal") != phase7.TERMINAL or not terminal.startswith(
        phase7.TERMINAL + " "
    ):
        raise guards.HybridQConfigError("Phase-7 sample terminal drift")
    sample = document.get("sample")
    if not isinstance(sample, Mapping) or int(sample.get("total_frames", -1)) != FRAMES:
        raise guards.HybridQConfigError("Phase-7 sample is not the required 128-frame set")
    if int(sample.get("frames_per_episode", -1)) != phase7.FRAMES_PER_EPISODE:
        raise guards.HybridQConfigError("Phase-7 per-episode sample cardinality drift")
    if int(sample.get("holdout_validation_test_frames", -1)) != 0:
        raise guards.HybridQConfigError("Phase-7 sample is not train-fit only")
    frames = sample.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAMES:
        raise guards.HybridQConfigError("Phase-7 sample frame manifest drift")
    return {"artifacts": bound, "terminal": document["terminal"]}, document


def _verify_transport_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in TRANSPORT_SOURCES.items():
        digest = sha256_file(_repository_path(relative))
        if digest != expected:
            raise guards.HybridQConfigError(
                f"Phase-11C transport source drift: {relative}"
            )
        observed[relative] = digest
    return observed


def phase11c_preflight() -> dict[str, Any]:
    """Fail closed on all frozen/provenance inputs before CUDA is queried."""
    phase7_binding = phase7.bind_phase7_inputs()
    phase11b_preflight = phase11b.phase11b_preflight()
    phase11b_artifacts = _verify_phase11b_artifacts()
    phase7_artifacts, phase7_document = _verify_phase7_sample_artifacts()
    transports = _verify_transport_sources()
    phase7_recorded = phase7_document.get("binding", {})
    for name in (
        "frozen_checkpoint",
        "stable_ranker",
        "perception_forward_lock",
        "hybrid_q_locked_config",
        "train_manifest",
    ):
        if phase7_recorded.get(name) != phase7_binding.get(name):
            raise guards.HybridQConfigError(
                f"Phase-7 recorded binding drift for {name}"
            )
    return {
        "phase7_input_binding": phase7_binding,
        "phase7_sample_artifacts": phase7_artifacts,
        "phase7_sample_document": phase7_document,
        "phase11b_artifacts": phase11b_artifacts,
        "phase11b_frozen_inputs": phase11b_preflight["frozen_hashes"],
        "phase11b_selection_bindings": phase11b_preflight["selection_bindings"],
        "phase11b_historical_source_bindings": phase11b_preflight[
            "historical_source_bindings"
        ],
        "phase11b_device_repair_source_transition": phase11b_preflight[
            "phase11b_device_repair_source_transition"
        ],
        "checkpoint_payloads": phase11b_preflight["checkpoint_payloads"],
        "transport_source_sha256": transports,
    }


def _phase7_frames(dataset: Any, phase7_document: Mapping[str, Any]) -> list[phase7.SelectedFrame]:
    """Reconstruct and compare the immutable Phase-7 sample exactly."""
    frames = phase7.select_measurement_frames(dataset)
    actual = [asdict(frame) for frame in frames]
    sample = phase7_document["sample"]
    if actual != sample["frames"]:
        raise guards.HybridQConfigError("reconstructed Phase-7 frame manifest differs")
    if phase7.selected_id_digest(frames) != sample["selected_sample_id_sha256"]:
        raise guards.HybridQConfigError("Phase-7 selected-ID digest drift")
    if phase7.selected_row_digest(frames) != sample["selected_row_sha256"]:
        raise guards.HybridQConfigError("Phase-7 selected-row digest drift")
    for episode in contract.TRAIN_FIT_EPISODES:
        selected = [frame for frame in frames if frame.episode == episode]
        if len(selected) != phase7.FRAMES_PER_EPISODE:
            raise guards.HybridQConfigError(f"Phase-7 episode count drift for {episode}")
        if selected[0].episode_position != 0 or selected[-1].episode_position != 15:
            raise guards.HybridQConfigError(f"Phase-7 route endpoints missing for {episode}")
    return frames


def _selection_map(
    c2: torch.Tensor, ranker: torch.nn.Module
) -> tuple[dict[float, Any], torch.Tensor, torch.Tensor]:
    """One score field and one stable order feed all six q payloads."""
    scores = ranker.score_cells(c2)
    order = torch.argsort(
        scores.reshape(-1).detach().to(torch.float32), descending=True, stable=True
    )
    selections: dict[float, Any] = {0.0: None}
    for q in Q_VALUES[1:]:
        plan = continuous_q.quantize_q(q)
        selections[q] = phase7._selection_from_order(
            order, plan, contract.SPLIT_SPATIAL_SHAPE
        )
    return selections, scores, order


def _payloads_for_frame(
    c2: torch.Tensor,
    ranker: torch.nn.Module,
    autoencoders: Mapping[str, SplitFeatureAE],
) -> Iterator[PayloadDescriptor]:
    """Yield only real framed inner payloads; no quantizer/packer is restated."""
    guards.require_frozen_c2(c2, what="Phase-11C frozen C2")
    selections, scores, order = _selection_map(c2, ranker)
    noae_uint8 = uint8_codec.prepare(c2)
    noae_lowbit = lowbit_transport.prepare_feature(
        c2,
        family_id=ae_contract.AE_FAMILY_NOAE,
        routing_tag=ae_contract.AE_UNBOUND_ROUTING_TAG,
    )
    prepared: dict[str, tuple[Any, Any, int, int]] = {
        "noAE": (
            noae_uint8,
            noae_lowbit,
            ae_contract.AE_FAMILY_NOAE,
            ae_contract.AE_UNBOUND_ROUTING_TAG,
        )
    }
    latents: list[torch.Tensor] = []
    for family, family_id, bottleneck in FAMILIES:
        if bottleneck is None:
            continue
        autoencoder = autoencoders[family]
        latent = autoencoder.encode(c2)
        latents.append(latent)
        prepared[family] = (
            ae_uint8_transport.prepare(latent),
            lowbit_transport.prepare_feature(
                latent,
                family_id=family_id,
                routing_tag=int(autoencoder.routing_tag),
            ),
            family_id,
            int(autoencoder.routing_tag),
        )

    try:
        for family, family_id, bottleneck in FAMILIES:
            uint8_prepared, lowbit_prepared, _prepared_family, routing_tag = prepared[
                family
            ]
            for quantizer, bit_width in QUANTIZERS:
                for q in Q_VALUES:
                    plan = continuous_q.quantize_q(q)
                    selection = selections[q]
                    if quantizer == "UINT8":
                        if family == "noAE":
                            payload = uint8_codec.encode(uint8_prepared, q, selection)
                            expected_bytes = uint8_codec.analytical_size(q).total_bytes
                        else:
                            payload = ae_uint8_transport.encode_sparse(
                                uint8_prepared,
                                q,
                                selection,
                                routing_tag=routing_tag,
                            )
                            expected_bytes = ae_uint8_transport.analytical_size(
                                q, int(bottleneck)
                            ).total_bytes
                    else:
                        payload = lowbit_transport.encode_sparse(
                            lowbit_prepared, q, bit_width, selection
                        )
                        expected_bytes = lowbit_transport.analytical_size(
                            q, family_id, bit_width
                        ).total_bytes
                    if len(payload.data) != expected_bytes:
                        raise guards.HybridQPayloadError(
                            f"{family} {quantizer} q={q}: analytical payload length drift"
                        )
                    yield PayloadDescriptor(
                        family=family,
                        family_id=family_id,
                        quantizer=quantizer,
                        bit_width=bit_width,
                        q=plan.wire_q,
                        q_e4=plan.q_e4,
                        keep_count=plan.keep_count,
                        bottleneck=bottleneck,
                        routing_tag=routing_tag,
                        inner=payload.data,
                    )
                    del payload
    finally:
        del selections, scores, order, noae_uint8, noae_lowbit, prepared, latents


def _verify_header(descriptor: PayloadDescriptor, restored: bytes) -> None:
    """Use the real wire inspectors; this benchmark owns no header parser."""
    if descriptor.quantizer == "UINT8" and descriptor.family == "noAE":
        parsed = uint8_codec.inspect(restored)
        header = parsed.header
        if (
            header.codec_id != uint8_codec.CODEC_ID_PER_CHANNEL_UINT8
            or header.q_e4 != descriptor.q_e4
            or header.keep_count != descriptor.keep_count
            or header.channels != contract.SPLIT_CHANNELS
        ):
            raise guards.HybridQPayloadError("noAE UINT8 header drift")
    elif descriptor.quantizer == "UINT8":
        parsed = ae_uint8_transport.inspect(restored)
        header = parsed.header
        if (
            header.family_id != descriptor.family_id
            or header.q_e4 != descriptor.q_e4
            or header.keep_count != descriptor.keep_count
            or header.bottleneck != descriptor.bottleneck
            or header.routing_tag != descriptor.routing_tag
        ):
            raise guards.HybridQPayloadError("AE UINT8 header drift")
    else:
        parsed = lowbit_transport.inspect(restored)
        header = parsed.header
        if (
            header.family_id != descriptor.family_id
            or header.bit_width != descriptor.bit_width
            or header.q_e4 != descriptor.q_e4
            or header.keep_count != descriptor.keep_count
            or header.routing_tag != descriptor.routing_tag
        ):
            raise guards.HybridQPayloadError("low-bit header drift")
    del parsed


def _contexts() -> dict[int, tuple[Any, Any]]:
    return {
        level: (
            zstandard.ZstdCompressor(level=level, **ZSTD_OPTIONS),
            zstandard.ZstdDecompressor(),
        )
        for level in LEVELS
    }


def _measure_descriptor(
    descriptor: PayloadDescriptor,
    contexts: Mapping[int, tuple[Any, Any]],
    measurements: Mapping[tuple[str, int], Measurements],
    *,
    warmup: bool,
) -> None:
    """One unreported warm-up then one timed exact round trip per level."""
    payload = descriptor.inner
    for level in LEVELS:
        compressor, decompressor = contexts[level]
        accumulator = measurements[(descriptor.profile_key, level)]
        if warmup:
            warm_frame = compressor.compress(payload)
            warm_restored = decompressor.decompress(warm_frame)
            if warm_restored != payload:
                raise guards.HybridQPayloadError("zstd warm-up was not byte-exact")
            del warm_frame, warm_restored

        compression_start = time.perf_counter_ns()
        frame = compressor.compress(payload)
        compression_ns = time.perf_counter_ns() - compression_start
        decompression_start = time.perf_counter_ns()
        restored = decompressor.decompress(frame)
        decompression_ns = time.perf_counter_ns() - decompression_start

        if int(zstandard.frame_content_size(frame)) != len(payload):
            raise guards.HybridQPayloadError("zstd frame content-size binding drift")
        if restored != payload:
            raise guards.HybridQPayloadError("zstd round trip was not byte-exact")
        _verify_header(descriptor, restored)
        accumulator.pre_zstd_bytes.append(len(payload))
        accumulator.compressed_bytes.append(len(frame))
        accumulator.compression_ns.append(compression_ns)
        accumulator.decompression_ns.append(decompression_ns)
        accumulator.inner_digest.update(hashlib.sha256(payload).digest())
        accumulator.round_trips += 1
        accumulator.header_checks += 1
        del frame, restored


def _make_measurements() -> dict[tuple[str, int], Measurements]:
    table: dict[tuple[str, int], Measurements] = {}
    for family, _family_id, _bottleneck in FAMILIES:
        for quantizer, _bits in QUANTIZERS:
            for q in Q_VALUES:
                q_e4 = continuous_q.quantize_q(q).q_e4
                profile = f"{family}_{quantizer}_q{q_e4:05d}"
                for level in LEVELS:
                    table[(profile, level)] = Measurements([], [], [], [], hashlib.sha256())
    if len(table) != PROFILES * len(LEVELS):
        raise guards.HybridQConfigError("Phase-11C profile registry cardinality drift")
    return table


def _stats(values: Sequence[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _ms_stats(values: Sequence[int]) -> dict[str, float]:
    stats = _stats(values)
    return {name + "_ms": value / 1e6 for name, value in stats.items()}


def _profile_rows(
    measurements: Mapping[tuple[str, int], Measurements]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, family_id, bottleneck in FAMILIES:
        for quantizer, bit_width in QUANTIZERS:
            for q in Q_VALUES:
                plan = continuous_q.quantize_q(q)
                profile = f"{family}_{quantizer}_q{plan.q_e4:05d}"
                level_one = measurements[(profile, 1)]
                level_one_total = sum(level_one.compressed_bytes)
                for level in LEVELS:
                    entry = measurements[(profile, level)]
                    if len(entry.pre_zstd_bytes) != FRAMES or entry.round_trips != FRAMES:
                        raise guards.HybridQPayloadError(
                            f"{profile} level {level} did not measure every frame"
                        )
                    pre = _stats(entry.pre_zstd_bytes)
                    compressed = _stats(entry.compressed_bytes)
                    comp_ms = _ms_stats(entry.compression_ns)
                    decomp_ms = _ms_stats(entry.decompression_ns)
                    total_pre = sum(entry.pre_zstd_bytes)
                    total_compressed = sum(entry.compressed_bytes)
                    rows.append(
                        {
                            "profile": profile,
                            "family": family,
                            "family_id": family_id,
                            "quantizer": quantizer,
                            "bit_width": bit_width,
                            "q": plan.wire_q,
                            "q_e4": plan.q_e4,
                            "keep_count": plan.keep_count,
                            "bottleneck": bottleneck,
                            "zstd_level": level,
                            "frames": FRAMES,
                            "exact_round_trips": entry.round_trips,
                            "header_checks": entry.header_checks,
                            "inner_payload_digest_sha256": entry.inner_digest.hexdigest(),
                            "pre_zstd_bytes": pre,
                            "compressed_bytes": compressed,
                            "total_pre_zstd_bytes": total_pre,
                            "total_compressed_bytes": total_compressed,
                            "compressed_to_pre_zstd_ratio": total_compressed / total_pre,
                            "compressed_ratio_relative_to_level1": (
                                total_compressed / level_one_total
                            ),
                            "compression": comp_ms,
                            "decompression": decomp_ms,
                            "compression_MBps": {
                                "median": np.median(
                                    np.asarray(entry.pre_zstd_bytes, dtype=np.float64)
                                    / (np.asarray(entry.compression_ns) / 1e9)
                                    / 1e6
                                ).item(),
                                "aggregate": total_pre
                                / (sum(entry.compression_ns) / 1e9)
                                / 1e6,
                            },
                            "decompression_MBps": {
                                "median": np.median(
                                    np.asarray(entry.pre_zstd_bytes, dtype=np.float64)
                                    / (np.asarray(entry.decompression_ns) / 1e9)
                                    / 1e6
                                ).item(),
                                "aggregate": total_pre
                                / (sum(entry.decompression_ns) / 1e9)
                                / 1e6,
                            },
                        }
                    )
    return rows


def _break_even(
    measurements: Mapping[tuple[str, int], Measurements]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, _family_id, _bottleneck in FAMILIES:
        for quantizer, _bits in QUANTIZERS:
            for q in Q_VALUES:
                q_e4 = continuous_q.quantize_q(q).q_e4
                profile = f"{family}_{quantizer}_q{q_e4:05d}"
                for lower, higher in ((1, 3), (1, 5), (3, 5)):
                    base = measurements[(profile, lower)]
                    candidate = measurements[(profile, higher)]
                    values: list[float] = []
                    dominates = 0
                    never = 0
                    for index in range(FRAMES):
                        saved_bytes = (
                            base.compressed_bytes[index]
                            - candidate.compressed_bytes[index]
                        )
                        extra_ns = (
                            candidate.compression_ns[index]
                            + candidate.decompression_ns[index]
                            - base.compression_ns[index]
                            - base.decompression_ns[index]
                        )
                        if saved_bytes > 0 and extra_ns > 0:
                            values.append(saved_bytes * 8.0 * 1000.0 / extra_ns)
                        elif saved_bytes >= 0 and extra_ns <= 0:
                            dominates += 1
                        else:
                            never += 1
                    rows.append(
                        {
                            "profile": profile,
                            "family": family,
                            "quantizer": quantizer,
                            "q": q,
                            "q_e4": q_e4,
                            "baseline_level": lower,
                            "candidate_level": higher,
                            "paired_frames": FRAMES,
                            "positive_break_even_frames": len(values),
                            "candidate_dominates_frames": dominates,
                            "candidate_never_breaks_even_frames": never,
                            "break_even_Mbps": (
                                {
                                    "median": float(np.median(values)),
                                    "p95": float(np.percentile(values, 95.0)),
                                    "min": float(min(values)),
                                    "max": float(max(values)),
                                }
                                if values
                                else None
                            ),
                        }
                    )
    return rows


def _aggregate_comparisons(
    measurements: Mapping[tuple[str, int], Measurements]
) -> dict[str, Any]:
    totals: dict[int, dict[str, float | int]] = {}
    for level in LEVELS:
        entries = [entry for (_profile, entry_level), entry in measurements.items() if entry_level == level]
        totals[level] = {
            "round_trips": sum(entry.round_trips for entry in entries),
            "pre_zstd_bytes": sum(sum(entry.pre_zstd_bytes) for entry in entries),
            "compressed_bytes": sum(sum(entry.compressed_bytes) for entry in entries),
            "compression_ms": sum(sum(entry.compression_ns) for entry in entries) / 1e6,
            "decompression_ms": sum(sum(entry.decompression_ns) for entry in entries) / 1e6,
        }
        totals[level]["codec_ms"] = (
            float(totals[level]["compression_ms"])
            + float(totals[level]["decompression_ms"])
        )
    comparisons: list[dict[str, Any]] = []
    for baseline, candidate in ((1, 3), (1, 5), (3, 5)):
        before = totals[baseline]
        after = totals[candidate]
        comparisons.append(
            {
                "baseline_level": baseline,
                "candidate_level": candidate,
                "incremental_size_saving_bytes": int(before["compressed_bytes"])
                - int(after["compressed_bytes"]),
                "incremental_size_saving_fraction": 1.0
                - float(after["compressed_bytes"]) / float(before["compressed_bytes"]),
                "incremental_compression_ms": float(after["compression_ms"])
                - float(before["compression_ms"]),
                "incremental_decompression_ms": float(after["decompression_ms"])
                - float(before["decompression_ms"]),
                "incremental_codec_ms": float(after["codec_ms"])
                - float(before["codec_ms"]),
            }
        )
    return {"per_level": totals, "comparisons": comparisons}


def _conclusion(
    rows: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Any]
) -> dict[str, str]:
    candidates: list[int] = []
    for candidate in LEVELS:
        candidate_rows = {row["profile"]: row for row in rows if row["zstd_level"] == candidate}
        dominates_all = True
        strictly_better = False
        for other in LEVELS:
            if other == candidate:
                continue
            other_rows = {row["profile"]: row for row in rows if row["zstd_level"] == other}
            for profile, candidate_row in candidate_rows.items():
                other_row = other_rows[profile]
                candidate_codec = (
                    candidate_row["compression"]["median_ms"]
                    + candidate_row["decompression"]["median_ms"]
                )
                other_codec = (
                    other_row["compression"]["median_ms"]
                    + other_row["decompression"]["median_ms"]
                )
                if (
                    candidate_row["total_compressed_bytes"] > other_row["total_compressed_bytes"]
                    or candidate_codec > other_codec
                ):
                    dominates_all = False
                    break
                strictly_better = strictly_better or (
                    candidate_row["total_compressed_bytes"] < other_row["total_compressed_bytes"]
                    or candidate_codec < other_codec
                )
            if not dominates_all:
                break
        if dominates_all and strictly_better:
            candidates.append(candidate)
    if candidates:
        return {
            "classification": "a level strictly dominates",
            "statement": f"level {candidates[0]} is no larger and no slower in every profile",
        }
    return {
        "classification": "workload/network dependent",
        "statement": (
            "no level is both no larger and no slower in every profile; select no "
            "production level here and seek Raspberry Pi/OAI confirmation"
        ),
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = [
        "profile", "family", "quantizer", "bit_width", "q", "q_e4", "keep_count",
        "zstd_level", "frames", "exact_round_trips", "pre_zstd_median_bytes",
        "pre_zstd_p95_bytes", "compressed_median_bytes", "compressed_p95_bytes",
        "compressed_to_pre_zstd_ratio", "compressed_ratio_relative_to_level1",
        "compression_median_ms", "compression_p95_ms", "decompression_median_ms",
        "decompression_p95_ms", "compression_median_MBps", "decompression_median_MBps",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "profile": row["profile"],
                "family": row["family"],
                "quantizer": row["quantizer"],
                "bit_width": row["bit_width"],
                "q": f"{row['q']:.2f}",
                "q_e4": row["q_e4"],
                "keep_count": row["keep_count"],
                "zstd_level": row["zstd_level"],
                "frames": row["frames"],
                "exact_round_trips": row["exact_round_trips"],
                "pre_zstd_median_bytes": f"{row['pre_zstd_bytes']['median']:.1f}",
                "pre_zstd_p95_bytes": f"{row['pre_zstd_bytes']['p95']:.1f}",
                "compressed_median_bytes": f"{row['compressed_bytes']['median']:.1f}",
                "compressed_p95_bytes": f"{row['compressed_bytes']['p95']:.1f}",
                "compressed_to_pre_zstd_ratio": f"{row['compressed_to_pre_zstd_ratio']:.8f}",
                "compressed_ratio_relative_to_level1": f"{row['compressed_ratio_relative_to_level1']:.8f}",
                "compression_median_ms": f"{row['compression']['median_ms']:.6f}",
                "compression_p95_ms": f"{row['compression']['p95_ms']:.6f}",
                "decompression_median_ms": f"{row['decompression']['median_ms']:.6f}",
                "decompression_p95_ms": f"{row['decompression']['p95_ms']:.6f}",
                "compression_median_MBps": f"{row['compression_MBps']['median']:.3f}",
                "decompression_median_MBps": f"{row['decompression_MBps']['median']:.3f}",
            }
        )
    return stream.getvalue()


def _report_text(document: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 11C — zstd level 1/3/5 sweep",
        "",
        f"Terminal: `{TERMINAL}`",
        "",
        "This is a lossless host codec comparison over real 72-profile inner payloads. "
        "It performs no perception scoring or accuracy measurement; byte-exact recovery "
        "means zstd level cannot change perception accuracy. Timings are not Raspberry Pi "
        "or OAI latency claims.",
        "",
        "## Aggregate comparison",
        "",
        "| level | round trips | compressed bytes | compression ms | decompression ms | codec ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    aggregate = document["aggregate_comparisons"]["per_level"]
    for level in LEVELS:
        row = aggregate[str(level)]
        lines.append(
            f"| {level} | {row['round_trips']:,} | {row['compressed_bytes']:,} | "
            f"{row['compression_ms']:.3f} | {row['decompression_ms']:.3f} | "
            f"{row['codec_ms']:.3f} |"
        )
    lines += [
        "",
        "| comparison | size saving | compression Δ ms | decompression Δ ms | codec Δ ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in document["aggregate_comparisons"]["comparisons"]:
        lines.append(
            f"| L{row['candidate_level']} vs L{row['baseline_level']} | "
            f"{row['incremental_size_saving_bytes']:,} "
            f"({row['incremental_size_saving_fraction']:.3%}) | "
            f"{row['incremental_compression_ms']:.3f} | "
            f"{row['incremental_decompression_ms']:.3f} | "
            f"{row['incremental_codec_ms']:.3f} |"
        )
    integrity = document["integrity"]
    lines += [
        "",
        "## Integrity and scope",
        "",
        f"- exact byte-equal round trips: {integrity['exact_round_trips']:,}/"
        f"{integrity['required_round_trips']:,}",
        f"- headers checked: {integrity['header_checks']:,}",
        f"- frozen state unchanged: {integrity['all_frozen_states_unchanged']}",
        f"- payload blobs retained: {integrity['payload_blobs_retained_after_measurement']}",
        "- train-fit sample: 128 frames, 16 deterministic endpoint-inclusive frames from each "
        "of eight fit episodes; holdout/validation/test frames read: 0",
        "- registered network bandwidth projections: none; no exact rates exist in the repository",
        f"- conclusion: {document['conclusion']['classification']} — "
        f"{document['conclusion']['statement']}",
        "",
        "The JSON carries all paired per-frame break-even summaries. The CSV carries every "
        "family × quantizer × q × level row.",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _runtime(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "executable": "/usr/bin/python3",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "zstandard": str(zstandard.__version__),
        "libzstd": ".".join(str(value) for value in zstandard.ZSTD_VERSION),
        "zstandard_backend": str(zstandard.backend),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase-11C bounded zstd level 1/3/5 comparison"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    args = parser.parse_args()

    output = contract.repository_root() / OUTPUT_RELPATH
    if output.exists():
        raise guards.HybridQConfigError(f"create-only output already exists: {output}")

    # This CPU-only step intentionally precedes even a CUDA availability query.
    preflight = phase11c_preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-11C requires CUDA on cuda:0")
    device = torch.device("cuda:0")
    started = time.perf_counter()

    checkpoint_payloads = preflight["checkpoint_payloads"]
    model, base, perception_binding = load_frozen_perception(device)
    common.freeze(model)
    ranker = phase11b._load_ranker(device)
    autoencoders = {
        family: phase11b._load_selected_autoencoder(
            family,
            int(bottleneck),
            phase11b.FROZEN_INPUTS[family],
            checkpoint_payloads[family],
            device,
        )
        for family, _family_id, bottleneck in FAMILIES
        if bottleneck is not None
    }
    checkpoint_payloads.clear()
    guards.require_frozen_perception([model, ranker, *autoencoders.values()])
    guards.require_eval_mode([model, ranker, *autoencoders.values()])
    frozen_before = {
        "perception": guards.snapshot_module_state(model),
        "ranker": guards.snapshot_module_state(ranker),
        **{
            family: guards.snapshot_module_state(autoencoder)
            for family, autoencoder in autoencoders.items()
        },
    }

    dataset = build_train_dataset(base)
    frames = _phase7_frames(dataset, preflight["phase7_sample_document"])
    contexts = _contexts()
    measurements = _make_measurements()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for position, selected in enumerate(frames):
            batch = collate_batch(base, dataset, [selected.dataset_index])
            c2_batch = encode_front(model, batch, device)
            if int(c2_batch.shape[0]) != 1:
                raise guards.HybridQPayloadError("Phase-11C front emitted more than one C2")
            c2 = c2_batch[0].detach()
            del batch, c2_batch
            frame_profiles = 0
            for descriptor in _payloads_for_frame(c2, ranker, autoencoders):
                _measure_descriptor(
                    descriptor,
                    contexts,
                    measurements,
                    warmup=position == 0,
                )
                del descriptor
                frame_profiles += 1
            if frame_profiles != PROFILES:
                raise guards.HybridQPayloadError(
                    f"frame {position} produced {frame_profiles} profiles, expected {PROFILES}"
                )
            del c2
            if (position + 1) % 16 == 0:
                print(f"measured {position + 1}/{FRAMES} frames", flush=True)

    for name, module in {
        "perception": model,
        "ranker": ranker,
        **autoencoders,
    }.items():
        guards.require_module_state_unchanged(module, frozen_before[name])
    if any(
        parameter.grad is not None
        for module in (model, ranker, *autoencoders.values())
        for parameter in module.parameters()
    ):
        raise guards.HybridQOwnershipError("a frozen Phase-11C module received a gradient")

    rows = _profile_rows(measurements)
    for family, _family_id, _bottleneck in FAMILIES:
        for quantizer, _bits in QUANTIZERS:
            for q in Q_VALUES:
                profile = (
                    f"{family}_{quantizer}_q{continuous_q.quantize_q(q).q_e4:05d}"
                )
                digests = {
                    measurements[(profile, level)].inner_digest.hexdigest()
                    for level in LEVELS
                }
                if len(digests) != 1:
                    raise guards.HybridQPayloadError(
                        f"zstd levels disagree on inner payload bytes for {profile}"
                    )
    exact_round_trips = sum(row["exact_round_trips"] for row in rows)
    header_checks = sum(row["header_checks"] for row in rows)
    if exact_round_trips != REQUIRED_ROUND_TRIPS or header_checks != REQUIRED_ROUND_TRIPS:
        raise guards.HybridQPayloadError(
            f"Phase-11C integrity count {exact_round_trips}/{header_checks} != "
            f"{REQUIRED_ROUND_TRIPS}"
        )
    break_even = _break_even(measurements)
    aggregate_comparisons = _aggregate_comparisons(measurements)
    conclusion = _conclusion(rows, aggregate_comparisons)
    torch.cuda.synchronize(device)
    resources = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "wall_seconds": time.perf_counter() - started,
    }
    sample = preflight["phase7_sample_document"]["sample"]
    document = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "starting_phase11b_terminal": phase11b.TERMINAL,
        "input_bindings": {
            "phase7": preflight["phase7_input_binding"],
            "phase7_sample_artifacts": preflight["phase7_sample_artifacts"],
            "phase11b_artifacts": preflight["phase11b_artifacts"],
            "phase11b_frozen_inputs": preflight["phase11b_frozen_inputs"],
            "phase11b_selection_bindings": preflight["phase11b_selection_bindings"],
            "phase11b_historical_source_bindings": preflight[
                "phase11b_historical_source_bindings"
            ],
            "phase11b_device_repair_source_transition": preflight[
                "phase11b_device_repair_source_transition"
            ],
            "transport_source_sha256": preflight["transport_source_sha256"],
        },
        "perception_binding": perception_binding,
        "sample": {
            "reused_phase7_sample": True,
            "total_frames": FRAMES,
            "frames_per_episode": phase7.FRAMES_PER_EPISODE,
            "episodes": list(contract.TRAIN_FIT_EPISODES),
            "selected_sample_id_sha256": sample["selected_sample_id_sha256"],
            "selected_row_sha256": sample["selected_row_sha256"],
            "canonical_order": True,
            "route_endpoints_included": True,
            "holdout_validation_test_frames_read": 0,
            "frames_retained": 0,
        },
        "scope": {
            "zstd_only": True,
            "lossless": True,
            "front_forwards": FRAMES,
            "front_forwards_per_frame": 1,
            "ranker_score_and_full_ordering_per_frame": 1,
            "ae_encodes_per_family_per_frame": 1,
            "q_masks_derived_from_same_ordering": True,
            "tail_forwards": 0,
            "fcos_heads": 0,
            "segmentation": 0,
            "localization": 0,
            "matching": 0,
            "perception_scoring": 0,
            "training": False,
            "validation_frames_read": 0,
            "test_frames_read": 0,
            "payload_blobs_retained": False,
        },
        "zstd": {
            "levels": list(LEVELS),
            "settings_except_level": ZSTD_SETTINGS,
            "warmup_unreported_per_profile_level": 1,
            "measured_round_trips_per_profile_level": FRAMES,
            "independent_frames": True,
        },
        "runtime": _runtime(device),
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "registered_network_bandwidths": {
            "found": False,
            "projections": [],
            "reason": "no exact registered four-profile bandwidth source exists in the repository",
        },
        "profiles": rows,
        "paired_break_even_bandwidths": break_even,
        "aggregate_comparisons": {
            "per_level": {
                str(level): value
                for level, value in aggregate_comparisons["per_level"].items()
            },
            "comparisons": aggregate_comparisons["comparisons"],
        },
        "conclusion": conclusion,
        "integrity": {
            "required_round_trips": REQUIRED_ROUND_TRIPS,
            "exact_round_trips": exact_round_trips,
            "header_checks": header_checks,
            "all_three_levels_recover_identical_inner_bytes": True,
            "all_headers_preserve_family_quantizer_q_and_keep_count": True,
            "all_frozen_states_unchanged": True,
            "payload_blobs_retained_after_measurement": 0,
        },
        "resources": resources,
    }

    output.mkdir(parents=True, exist_ok=False)
    report_hash = _atomic_write(
        output / "phase11c_zstd_level_sweep.json",
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output / "phase11c_zstd_level_sweep.csv", _csv_text(rows))
    _atomic_write(output / "PHASE11C_ZSTD_LEVEL_SWEEP_REPORT.md", _report_text(document))
    _atomic_write(output / TERMINAL, f"{TERMINAL} {report_hash}\n")
    print(
        json.dumps(
            {
                "terminal": TERMINAL,
                "output": str(output),
                "exact_round_trips": f"{exact_round_trips}/{REQUIRED_ROUND_TRIPS}",
                "report_sha256": report_hash,
            }
        )
    )
    print(TERMINAL)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
