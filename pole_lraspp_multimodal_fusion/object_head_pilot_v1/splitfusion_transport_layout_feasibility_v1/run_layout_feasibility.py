"""One bounded lossless layout feasibility study over the Phase-11C catalog.

This package is intentionally outside the locked AE and hybrid-q packages.  It
does not create a production wire version: the layout identifier exists only in
this benchmark, while every inner payload is produced by the existing public
UINT8/UINT6/UINT4 writers.  The frozen tail and all perception scorers are out
of scope.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import zstandard

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards, phase7_zstd_measurement as phase7
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    collate_batch,
    encode_front,
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import implementation_report
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import ae_contract, ae_phase11b_gpu_qualification as phase11b
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import ae_phase11c_zstd_level_sweep as phase11c
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import ae_training_common as common
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import ae_uint8_transport, lowbit_transport
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import ae_uint8_transport as ae_uint8
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import lowbit_transport as lowbit
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import uint8_codec
from ..splitfusion_fcos_r50_fpn_p2_p7_ae_v1.ae_model import SplitFeatureAE
from .layout_transforms import Layout, ValueBlockPlan, inverse, transform


EXECUTE_TOKEN = "TRANSPORT_LAYOUT_FEASIBILITY_STUDY"
TERMINAL = "TRANSPORT_LAYOUT_FEASIBILITY_COMPLETE"
SCHEMA = "splitfusion_transport_layout_feasibility_v1"
STARTING_HEAD = "447eb0205f258016189fc20564eacd0211e863cf"
OUTPUT_RELPATH = (
    "experiments/splitfusion_fcos_transport_layout_feasibility_v1/"
    "20260903_layout_feasibility_once"
)
FRAMES = phase7.SAMPLE_FRAMES
PROFILES = phase11c.PROFILES
LAYOUTS = (Layout.CURRENT_CELL_MAJOR, Layout.CHANNEL_MAJOR, Layout.CHANNEL_MAJOR_MODULAR_DELTA)
REQUIRED_ROUND_TRIPS = FRAMES * PROFILES * len(LAYOUTS)
ZSTD_OPTIONS = {
    "threads": 0,
    "dict_data": None,
    "write_checksum": True,
    "write_content_size": True,
    "write_dict_id": False,
}


@dataclass
class Samples:
    inner_bytes: list[int]
    compressed_bytes: list[int]
    transform_ns: list[int]
    compression_ns: list[int]
    decompression_ns: list[int]
    inverse_ns: list[int]
    round_trips: int = 0
    prefix_checks: int = 0


def _repository_root(repository_root: Path | None = None) -> Path:
    """Return the existing repository root; inputs never use a lax resolver."""
    root = contract.repository_root() if repository_root is None else repository_root
    return Path(root).resolve(strict=True)


def _root_path(relative: str, *, repository_root: Path | None = None) -> Path:
    """Resolve an existing input artifact beneath an existing repository root."""
    return (_repository_root(repository_root) / relative).resolve(strict=True)


def _is_beneath(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _create_output_directory(
    relative: str = OUTPUT_RELPATH, *, repository_root: Path | None = None
) -> Path:
    """Create the requested run leaf once, without weakening input resolution.

    ``relative`` is resolved non-strictly only while its final leaf does not
    exist.  The existing experiments root and the completed directory are both
    strictly resolved and containment-checked.
    """
    root = _repository_root(repository_root)
    experiments = (root / "experiments").resolve(strict=True)
    candidate = (root / relative).resolve(strict=False)
    if candidate == experiments or not _is_beneath(candidate, experiments):
        raise guards.HybridQConfigError("layout output must be beneath experiments root")
    if candidate.exists():
        raise guards.HybridQConfigError(f"create-only output already exists: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    parent = candidate.parent.resolve(strict=True)
    if not _is_beneath(parent, experiments):
        raise guards.HybridQConfigError("layout output parent escaped experiments root")
    try:
        candidate.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise guards.HybridQConfigError(f"create-only output already exists: {candidate}") from exc
    created = candidate.resolve(strict=True)
    if not _is_beneath(created, experiments):
        raise guards.HybridQConfigError("created layout output escaped experiments root")
    return created


def _require_hash(relative: str, expected: str) -> dict[str, str]:
    observed = sha256_file(_root_path(relative))
    if observed != expected:
        raise guards.HybridQConfigError(f"layout feasibility provenance drift: {relative}")
    return {"path": relative, "sha256": observed}


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=contract.repository_root(), text=True
    ).strip()


def preflight() -> dict[str, Any]:
    """Verify frozen inputs and current source bytes before any CUDA query."""
    inherited = phase11c.phase11c_preflight()
    p11c_report = _require_hash(
        phase11c.OUTPUT_RELPATH + "/phase11c_zstd_level_sweep.json",
        "4bcd2eddff502cc55d799bfdf5af920ccc1378ae87bd2c0d6017d3d2586c7b2d",
    )
    p11c_terminal = _require_hash(
        phase11c.OUTPUT_RELPATH + "/" + phase11c.TERMINAL,
        "198585a7384382b6c858a4f17595b132b008e7ef8f9a58d2c595032acdb1b5c2",
    )
    terminal_text = _root_path(p11c_terminal["path"]).read_text(encoding="utf-8")
    if terminal_text != f"{phase11c.TERMINAL} {p11c_report['sha256']}\n":
        raise guards.HybridQConfigError("Phase-11C terminal does not bind report")
    zstd = implementation_report()
    settings = zstd.get("settings", {})
    expected = {"level": 1, **ZSTD_OPTIONS, "one_frame_per_camera_frame": True}
    if any(settings.get(name) != value for name, value in expected.items()):
        raise guards.HybridQConfigError("selected zstd L1 configuration drift")
    return {
        "starting_head_required": STARTING_HEAD,
        "current_head": _head(),
        "phase11c_preflight": inherited,
        "phase11c_evidence": {"report": p11c_report, "terminal": p11c_terminal},
        "zstd_level1": zstd,
        "layouts": [layout.value for layout in LAYOUTS],
    }


def _value_plan(descriptor: phase11c.PayloadDescriptor) -> ValueBlockPlan:
    """Read offsets and K/C from the existing codec inspectors only."""
    inner = descriptor.inner
    if descriptor.quantizer == "UINT8" and descriptor.family == "noAE":
        parsed = uint8_codec.inspect(inner)
        offset = uint8_codec.HEADER_BYTES + parsed.header.mask_bytes + parsed.header.range_bytes
        plan = ValueBlockPlan(offset, int(parsed.header.keep_count), int(parsed.header.channels), 8)
    elif descriptor.quantizer == "UINT8":
        parsed = ae_uint8.inspect(inner)
        offset = ae_uint8.HEADER_BYTES + parsed.header.mask_bytes + parsed.header.range_bytes
        plan = ValueBlockPlan(offset, int(parsed.header.keep_count), int(parsed.header.bottleneck), 8)
    else:
        parsed = lowbit.inspect(inner)
        offset = lowbit.HEADER_BYTES + parsed.header.mask_bytes + parsed.header.range_bytes
        plan = ValueBlockPlan(offset, int(parsed.header.keep_count), int(parsed.header.channels), int(parsed.header.bit_width))
    if plan.keep_count != descriptor.keep_count or plan.channels <= 0 or plan.bit_width != descriptor.bit_width:
        raise guards.HybridQPayloadError("existing inner codec metadata drift")
    return plan


def _measure(
    *, descriptor: phase11c.PayloadDescriptor, plan: ValueBlockPlan, layout: Layout,
    compressor: zstandard.ZstdCompressor, decompressor: zstandard.ZstdDecompressor,
    samples: Samples, warmup: bool,
) -> None:
    """One unreported warm-up and one timed byte-exact layout/zstd round trip."""
    original = descriptor.inner

    def round_trip(*, record: bool) -> None:
        started = time.perf_counter_ns()
        transformed = transform(original, plan, layout)
        transform_ns = time.perf_counter_ns() - started
        if transformed[: plan.value_offset] != original[: plan.value_offset]:
            raise guards.HybridQPayloadError("layout changed header/mask/range bytes")
        started = time.perf_counter_ns()
        compressed = compressor.compress(transformed)
        compression_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        decompressed = decompressor.decompress(compressed)
        decompression_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        restored = inverse(decompressed, plan, layout)
        inverse_ns = time.perf_counter_ns() - started
        if int(zstandard.frame_content_size(compressed)) != len(transformed):
            raise guards.HybridQPayloadError("zstd content-size binding drift")
        if restored != original:
            raise guards.HybridQPayloadError("layout/zstd round trip was not byte-exact")
        if record:
            samples.inner_bytes.append(len(original))
            samples.compressed_bytes.append(len(compressed))
            samples.transform_ns.append(transform_ns)
            samples.compression_ns.append(compression_ns)
            samples.decompression_ns.append(decompression_ns)
            samples.inverse_ns.append(inverse_ns)
            samples.round_trips += 1
            samples.prefix_checks += 1
        del transformed, compressed, decompressed, restored

    if warmup:
        round_trip(record=False)
    round_trip(record=True)


def _table() -> dict[tuple[str, Layout], Samples]:
    table: dict[tuple[str, Layout], Samples] = {}
    for family, _family_id, _bottleneck in phase11c.FAMILIES:
        for quantizer, _bits in phase11c.QUANTIZERS:
            for q in phase11c.Q_VALUES:
                key = f"{family}_{quantizer}_q{int(round(q * 10000)):05d}"
                for layout in LAYOUTS:
                    table[(key, layout)] = Samples([], [], [], [], [], [])
    if len(table) != PROFILES * len(LAYOUTS):
        raise guards.HybridQConfigError("layout catalog cardinality drift")
    return table


def _stats(values: Sequence[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(array)), "p95": float(np.percentile(array, 95.0)), "mean": float(np.mean(array))}


def _ms(values: Sequence[int]) -> dict[str, float]:
    return {name + "_ms": value / 1e6 for name, value in _stats(values).items()}


def _break_even(base: Samples, candidate: Samples) -> dict[str, Any]:
    values: list[float] = []
    dominates = never = 0
    for index in range(FRAMES):
        saved = base.compressed_bytes[index] - candidate.compressed_bytes[index]
        extra_ns = (
            candidate.transform_ns[index] + candidate.compression_ns[index]
            + candidate.decompression_ns[index] + candidate.inverse_ns[index]
            - base.compression_ns[index] - base.decompression_ns[index]
        )
        if saved > 0 and extra_ns > 0:
            values.append(saved * 8000.0 / extra_ns)
        elif saved >= 0 and extra_ns <= 0:
            dominates += 1
        else:
            never += 1
    return {
        "paired_frames": FRAMES, "positive_break_even_frames": len(values),
        "candidate_dominates_frames": dominates, "candidate_never_breaks_even_frames": never,
        "break_even_Mbps": ({"median": float(np.median(values)), "p95": float(np.percentile(values, 95.0)), "min": float(min(values)), "max": float(max(values))} if values else None),
    }


def _rows(table: Mapping[tuple[str, Layout], Samples]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family, family_id, bottleneck in phase11c.FAMILIES:
        for quantizer, bits in phase11c.QUANTIZERS:
            for q in phase11c.Q_VALUES:
                plan = phase11c.continuous_q.quantize_q(q)
                profile = f"{family}_{quantizer}_q{plan.q_e4:05d}"
                baseline = table[(profile, Layout.CURRENT_CELL_MAJOR)]
                for layout in LAYOUTS:
                    sample = table[(profile, layout)]
                    if sample.round_trips != FRAMES or sample.prefix_checks != FRAMES:
                        raise guards.HybridQPayloadError(f"{profile} {layout.value} measurement count drift")
                    total_codec = [a + b + c + d for a, b, c, d in zip(sample.transform_ns, sample.compression_ns, sample.decompression_ns, sample.inverse_ns)]
                    total_base = sum(baseline.compressed_bytes)
                    total_current = sum(sample.compressed_bytes)
                    out.append({
                        "profile": profile, "family": family, "family_id": family_id, "quantizer": quantizer,
                        "bit_width": bits, "q": plan.wire_q, "q_e4": plan.q_e4, "keep_count": plan.keep_count,
                        "bottleneck": bottleneck, "layout": layout.value, "frames": FRAMES,
                        "exact_round_trips": sample.round_trips, "original_inner_bytes": _stats(sample.inner_bytes),
                        "compressed_bytes": _stats(sample.compressed_bytes),
                        "compression_ratio": sum(sample.compressed_bytes) / sum(sample.inner_bytes),
                        "forward_layout": _ms(sample.transform_ns), "zstd_compression": _ms(sample.compression_ns),
                        "zstd_decompression": _ms(sample.decompression_ns), "inverse_layout": _ms(sample.inverse_ns),
                        "complete_codec": _ms(total_codec),
                        "size_change_relative_to_current": {"bytes": total_current - total_base, "fraction": total_current / total_base - 1.0},
                        "latency_change_relative_to_current": {"median_ms": _ms(total_codec)["median_ms"] - _ms([a + b for a, b in zip(baseline.compression_ns, baseline.decompression_ns)])["median_ms"], "p95_ms": _ms(total_codec)["p95_ms"] - _ms([a + b for a, b in zip(baseline.compression_ns, baseline.decompression_ns)])["p95_ms"]},
                        "paired_break_even": None if layout is Layout.CURRENT_CELL_MAJOR else _break_even(baseline, sample),
                    })
    return out


def _aggregate(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        current = [row for row in group if row["layout"] == Layout.CURRENT_CELL_MAJOR.value][0:]
        # Each layout group is compared to the matching current rows by profile.
        current_by_profile = {row["profile"]: row for row in rows if row["layout"] == Layout.CURRENT_CELL_MAJOR.value}
        compressed = sum(float(row["compressed_bytes"]["mean"]) * FRAMES for row in group)
        base = sum(float(current_by_profile[row["profile"]]["compressed_bytes"]["mean"]) * FRAMES for row in group)
        codec = sum(float(row["complete_codec"]["mean_ms"]) * FRAMES for row in group)
        base_codec = sum(float(current_by_profile[row["profile"]]["complete_codec"]["mean_ms"]) * FRAMES for row in group)
        out.append({**dict(zip(fields, key)), "profiles": len(group), "compressed_bytes": compressed, "size_change_relative_to_current": {"bytes": compressed - base, "fraction": compressed / base - 1.0}, "complete_codec_change_ms": codec - base_codec})
    return out


def _classification(rows: Sequence[Mapping[str, Any]], quantizer_aggregate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_layout = {row["layout"]: row for row in _aggregate(rows, ("layout",))}
    candidates = [row for name, row in by_layout.items() if name != Layout.CURRENT_CELL_MAJOR.value]
    best = min(candidates, key=lambda row: float(row["size_change_relative_to_current"]["fraction"]))
    reduction = -float(best["size_change_relative_to_current"]["fraction"])
    quantizer_reductions = [
        -float(row["size_change_relative_to_current"]["fraction"])
        for row in quantizer_aggregate if row["layout"] == best["layout"]
    ]
    credible = float(best["complete_codec_change_ms"]) > 0 and any(value > 0 for value in quantizer_reductions)
    if reduction < 0.05 or all(value <= 0 for value in quantizer_reductions):
        label = "NOT_USEFUL"
    elif credible and all(value >= 0.05 for value in quantizer_reductions):
        label = "PROMISING"
    else:
        label = "MIXED"
    return {
        "classification": label, "best_layout_by_aggregate_bytes": best["layout"],
        "best_aggregate_byte_reduction_fraction": reduction,
        "credible_latency_tradeoff": credible,
        "quantizer_reduction_fractions_for_best_layout": quantizer_reductions,
        "production_layout_change_recommended": False,
        "decision": "evidence for review only; no production wire or codec is changed automatically",
    }


def _csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = ("profile", "family", "quantizer", "bit_width", "q", "q_e4", "layout", "original_inner_median_bytes", "compressed_median_bytes", "compressed_p95_bytes", "compression_ratio", "size_change_fraction", "forward_layout_median_ms", "zstd_compression_median_ms", "zstd_decompression_median_ms", "inverse_layout_median_ms", "complete_codec_median_ms", "codec_latency_change_median_ms")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({"profile": row["profile"], "family": row["family"], "quantizer": row["quantizer"], "bit_width": row["bit_width"], "q": f"{row['q']:.2f}", "q_e4": row["q_e4"], "layout": row["layout"], "original_inner_median_bytes": row["original_inner_bytes"]["median"], "compressed_median_bytes": row["compressed_bytes"]["median"], "compressed_p95_bytes": row["compressed_bytes"]["p95"], "compression_ratio": row["compression_ratio"], "size_change_fraction": row["size_change_relative_to_current"]["fraction"], "forward_layout_median_ms": row["forward_layout"]["median_ms"], "zstd_compression_median_ms": row["zstd_compression"]["median_ms"], "zstd_decompression_median_ms": row["zstd_decompression"]["median_ms"], "inverse_layout_median_ms": row["inverse_layout"]["median_ms"], "complete_codec_median_ms": row["complete_codec"]["median_ms"], "codec_latency_change_median_ms": row["latency_change_relative_to_current"]["median_ms"]})
    return stream.getvalue()


def _markdown(document: Mapping[str, Any]) -> str:
    aggregate = document["aggregate_by_layout"]
    lines = ["# Transport layout feasibility", "", f"Terminal: `{TERMINAL}`", "", "| layout | profiles | size change vs current | codec latency Δ ms |", "| --- | ---: | ---: | ---: |"]
    for row in aggregate:
        lines.append(f"| {row['layout']} | {row['profiles']} | {row['size_change_relative_to_current']['fraction']:.3%} | {row['complete_codec_change_ms']:.3f} |")
    lines += ["", "## Per quantizer", "", "| quantizer | layout | size change vs current | codec latency Δ ms |", "| --- | --- | ---: | ---: |"]
    for row in document["aggregate_by_quantizer"]:
        lines.append(f"| {row['quantizer']} | {row['layout']} | {row['size_change_relative_to_current']['fraction']:.3%} | {row['complete_codec_change_ms']:.3f} |")
    verdict = document["classification"]
    integrity = document["integrity"]
    lines += ["", "## Result", "", f"- classification: **{verdict['classification']}**", f"- production layout change recommended: {verdict['production_layout_change_recommended']}", f"- exact layout/zstd round trips: {integrity['exact_round_trips']:,}/{integrity['required_round_trips']:,}", f"- frozen state unchanged: {integrity['frozen_state_unchanged']}", "- network profiles with explicit usable bandwidth: none found; paired break-even bandwidths are reported in JSON only.", "- no tail, heads, p025, evaluator, validation/test frame or payload blob was retained.", ""]
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="bounded lossless transport layout feasibility")
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    args = parser.parse_args()
    output = _create_output_directory()
    binding = preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("layout feasibility study requires CUDA on cuda:0")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    checkpoint_payloads = binding["phase11c_preflight"]["checkpoint_payloads"]
    model, base, perception = load_frozen_perception(device)
    common.freeze(model)
    ranker = phase11b._load_ranker(device)
    autoencoders: dict[str, SplitFeatureAE] = {
        family: phase11b._load_selected_autoencoder(family, int(bottleneck), phase11b.FROZEN_INPUTS[family], checkpoint_payloads[family], device)
        for family, _family_id, bottleneck in phase11c.FAMILIES if bottleneck is not None
    }
    checkpoint_payloads.clear()
    guards.require_frozen_perception([model, ranker, *autoencoders.values()])
    guards.require_eval_mode([model, ranker, *autoencoders.values()])
    snapshots = {"perception": guards.snapshot_module_state(model), "ranker": guards.snapshot_module_state(ranker), **{name: guards.snapshot_module_state(module) for name, module in autoencoders.items()}}
    dataset = build_train_dataset(base)
    frames = phase11c._phase7_frames(dataset, binding["phase11c_preflight"]["phase7_sample_document"])
    contexts = {layout: (zstandard.ZstdCompressor(level=1, **ZSTD_OPTIONS), zstandard.ZstdDecompressor()) for layout in LAYOUTS}
    table = _table()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for position, selected in enumerate(frames):
            batch = collate_batch(base, dataset, [selected.dataset_index])
            c2_batch = encode_front(model, batch, device)
            if int(c2_batch.shape[0]) != 1:
                raise guards.HybridQPayloadError("front did not emit exactly one C2")
            c2 = c2_batch[0].detach()
            del batch, c2_batch
            profile_count = 0
            for descriptor in phase11c._payloads_for_frame(c2, ranker, autoencoders):
                # Existing inspectors bind family, quantizer, q and keep count
                # before the external layout experiment sees the value block.
                phase11c._verify_header(descriptor, descriptor.inner)
                plan = _value_plan(descriptor)
                for layout in LAYOUTS:
                    compressor, decompressor = contexts[layout]
                    _measure(descriptor=descriptor, plan=plan, layout=layout, compressor=compressor, decompressor=decompressor, samples=table[(descriptor.profile_key, layout)], warmup=position == 0)
                del descriptor
                profile_count += 1
            if profile_count != PROFILES:
                raise guards.HybridQPayloadError("frame did not generate exactly 72 profiles")
            del c2
            if (position + 1) % 16 == 0:
                print(f"measured {position + 1}/{FRAMES} fit frames", flush=True)
    for name, module in {"perception": model, "ranker": ranker, **autoencoders}.items():
        guards.require_module_state_unchanged(module, snapshots[name])
    if any(parameter.grad is not None for module in (model, ranker, *autoencoders.values()) for parameter in module.parameters()):
        raise guards.HybridQOwnershipError("a frozen module received a gradient")
    rows = _rows(table)
    exact = sum(row["exact_round_trips"] for row in rows)
    if exact != REQUIRED_ROUND_TRIPS:
        raise guards.HybridQPayloadError(f"round-trip integrity {exact} != {REQUIRED_ROUND_TRIPS}")
    by_layout = _aggregate(rows, ("layout",))
    by_family = _aggregate(rows, ("family", "layout"))
    by_quantizer = _aggregate(rows, ("quantizer", "layout"))
    by_q = _aggregate(rows, ("q", "q_e4", "layout"))
    classification = _classification(rows, by_quantizer)
    torch.cuda.synchronize(device)
    document = {
        "schema": SCHEMA, "terminal": TERMINAL, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_bindings": binding, "runner_source_sha256": sha256_file(Path(__file__)),
        "scope": {"fit_frames": FRAMES, "fit_frames_per_episode": phase7.FRAMES_PER_EPISODE, "profiles": PROFILES, "layouts": [layout.value for layout in LAYOUTS], "one_unreported_warmup_per_profile_layout": True, "measured_each_frame_once_per_profile_layout": True, "tail_forwards": 0, "perception_scoring": 0, "validation_frames_read": 0, "test_frames_read": 0, "payload_blobs_retained": False},
        "timing_scope": {"included": ["forward layout", "zstd compression", "zstd decompression", "inverse layout"], "excluded": ["model inference", "AE encoding", "ranker", "quantization", "packing", "report writing"], "clock": "time.perf_counter_ns"},
        "network_profiles": {"explicit_usable_bandwidth_values_found": False, "hash_bound_sources": [], "projection_policy": "paired break-even bandwidths only"},
        "profiles": rows, "aggregate_by_layout": by_layout, "aggregate_by_family": by_family, "aggregate_by_quantizer": by_quantizer, "aggregate_by_q": by_q,
        "classification": classification,
        "integrity": {"required_round_trips": REQUIRED_ROUND_TRIPS, "exact_round_trips": exact, "header_mask_range_bytes_unchanged_checks": exact, "frozen_state_unchanged": True, "payload_blobs_retained_after_measurement": 0, "production_codec_modified": False, "phase11d_modified": False},
        "resources": {"wall_seconds": time.perf_counter() - started, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "device": torch.cuda.get_device_name(device), "python": platform.python_version(), "torch": torch.__version__, "zstandard": zstandard.__version__},
    }
    report_hash = _atomic_write(output / "transport_layout_feasibility.json", json.dumps(document, indent=2, sort_keys=True) + "\n")
    _atomic_write(output / "transport_layout_feasibility.csv", _csv(rows))
    _atomic_write(output / "TRANSPORT_LAYOUT_FEASIBILITY_REPORT.md", _markdown(document))
    _atomic_write(output / TERMINAL, f"{TERMINAL} {report_hash}\n")
    print(json.dumps({"terminal": TERMINAL, "round_trips": f"{exact}/{REQUIRED_ROUND_TRIPS}", "classification": classification["classification"], "report_sha256": report_hash}), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
