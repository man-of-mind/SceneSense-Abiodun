#!/usr/bin/env python3
"""Replay all 288 measured cells through the qualified edge candidate.

This program preserves measured capture times, feature bytes and complete
reassembly times.  It replaces only the old edge-service duration with a
family-calibrated optimized duration, then reruns the qualified two-stage,
depth-one latest-only scheduler with the provisionally selected 25 ms
pre-compute expiry ceiling.  The result is a counterfactual model, never a
replacement for the immutable live measurements.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .two_stage_simulator import (
    TwoStageConfig,
    TwoStageFrame,
    TwoStageReason,
    simulate_two_stage,
)


REPO = Path(__file__).resolve().parents[2]
CONSOLIDATION = REPO / (
    "experiments/splitfusion_288_offline_rl_dataset_v1/"
    "20260909_offline_consolidation_v1"
)
BASELINE = REPO / (
    "experiments/splitfusion_timing_diagnostic_v1/"
    "20260909_live_carla_actions30_15_50_71_retry3"
)
OPTIMIZED = REPO / (
    "experiments/splitfusion_edge_optimization_v1/"
    "20260909_live_actions30_15_50_71"
)
SCHEDULER_ANALYSIS = REPO / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_live_actions50_71_two_policies_analysis_v2"
)
DEFAULT_OUTPUT = REPO / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_counterfactual_288_v4"
)
SCHEMA = "scenesense.splitfusion_counterfactual_288.v4"
TERMINAL = "SPLITFUSION_COUNTERFACTUAL_288_COMPLETE"
WAIT_BUDGET_NS = 25_000_000
HORIZON_NS = 500_000_000
SERVICE_TARGET_NS = 100_000_000
FAMILY_ANCHORS = {"noAE": 15, "AE128": 30, "AE64": 50, "AE32": 71}
CONSOLIDATION_HASHES = {
    "campaign_288_cell_table.csv": (
        "3f3067d4c9d0ef3c0d2661c3d19d04306bcbd18ef6d21fa426148e0ee24d2e0d"
    ),
    "action_72x4_summary.csv": (
        "280b372ed8b6a52bb3e5f1f69e6cbdc02dc851a6a597565fcab9ca9652f5998e"
    ),
    "action_72_summary.csv": (
        "250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88"
    ),
    "analysis_summary.json": (
        "48358c2f27ffc2f3da8bf0f821c733914a03267f50da7adab4cbf70cbe239690"
    ),
}


class CounterfactualError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CounterfactualError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def _i(value: Any) -> int | None:
    number = _f(value)
    return None if number is None else int(number)


def _pydict(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    parsed = ast.literal_eval(value)
    _require(isinstance(parsed, dict), "timing field is not a dictionary")
    return parsed


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def _linear_interpolate(known: Sequence[tuple[int, int]], query: int) -> int:
    _require(bool(known), "cannot interpolate an empty service series")
    ordered = sorted(known)
    positions = [item[0] for item in ordered]
    return _linear_interpolate_ordered(ordered, positions, query)


def _linear_interpolate_ordered(
    ordered: Sequence[tuple[int, int]], positions: Sequence[int], query: int
) -> int:
    _require(bool(ordered), "cannot interpolate an empty service series")
    _require(len(ordered) == len(positions), "interpolation index drift")
    offset = bisect.bisect_left(positions, query)
    if offset <= 0:
        return int(ordered[0][1])
    if offset >= len(ordered):
        return int(ordered[-1][1])
    left_x, left_y = ordered[offset - 1]
    right_x, right_y = ordered[offset]
    if left_x == right_x:
        return int(left_y)
    fraction = (query - left_x) / (right_x - left_x)
    return int(round(left_y + fraction * (right_y - left_y)))


def _deterministic_sample(values: Sequence[int], identity: str) -> int:
    _require(bool(values), "cannot sample an empty empirical pool")
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(values)
    return int(values[index])


def _systematic_select(indices: Sequence[int], count: int) -> set[int]:
    """Select a deterministic, approximately even subset including endpoints."""
    _require(0 <= count <= len(indices), "systematic sample count is invalid")
    if count == 0:
        return set()
    if count == len(indices):
        return set(indices)
    if count == 1:
        return {indices[len(indices) // 2]}
    selected = {
        indices[(position * (len(indices) - 1) + (count - 1) // 2) // (count - 1)]
        for position in range(count)
    }
    _require(len(selected) == count, "systematic selection produced duplicates")
    return selected


def _verify_manifest(root: Path, name: str = "ARTIFACT_MANIFEST.json") -> int:
    manifest = _load_json(root / name)
    hashes = manifest["sha256"]
    for relative, expected in hashes.items():
        path = root / relative
        _require(path.is_file(), f"manifest artifact absent: {path}")
        _require(_sha256(path) == expected, f"manifest artifact drift: {path}")
    return len(hashes)


def _verify_sources() -> dict[str, Any]:
    verified = 0
    for relative, expected in CONSOLIDATION_HASHES.items():
        path = CONSOLIDATION / relative
        _require(path.is_file(), f"consolidation artifact absent: {path}")
        _require(_sha256(path) == expected, f"consolidation artifact drift: {path}")
        verified += 1
    verified += _verify_manifest(BASELINE)
    verified += _verify_manifest(OPTIMIZED)
    verified += _verify_manifest(SCHEDULER_ANALYSIS, "artifact_manifest.json")
    scheduler = _load_json(SCHEDULER_ANALYSIS / "analysis.json")
    selection = scheduler["provisional_policy_selection"]
    _require(selection["policy"] == "LATEST_ONLY_25_MS", "scheduler selection drift")
    return {
        "implementation": {
            str(Path(__file__).resolve().relative_to(REPO)): _sha256(
                Path(__file__).resolve()
            ),
            "rl_agent/splitfusion_edge_freshness_scheduler_v1/two_stage_simulator.py": (
                _sha256(
                    REPO
                    / "rl_agent/splitfusion_edge_freshness_scheduler_v1/"
                    "two_stage_simulator.py"
                )
            ),
        },
        "verified_compact_artifacts": verified,
        "consolidation": {
            name: {"path": str((CONSOLIDATION / name).relative_to(REPO)), "sha256": digest}
            for name, digest in CONSOLIDATION_HASHES.items()
        },
        "baseline_manifest_sha256": _sha256(BASELINE / "ARTIFACT_MANIFEST.json"),
        "optimized_manifest_sha256": _sha256(OPTIMIZED / "ARTIFACT_MANIFEST.json"),
        "scheduler_analysis_sha256": _sha256(SCHEDULER_ANALYSIS / "analysis.json"),
        "scheduler_analysis_manifest_sha256": _sha256(
            SCHEDULER_ANALYSIS / "artifact_manifest.json"
        ),
        "policy_selection": selection,
    }


def _action_summaries(path: Path) -> dict[int, Mapping[str, Any]]:
    document = _load_json(path)
    return {int(item["action_id"]): item for item in document["action_summaries"]}


def _calibration() -> tuple[dict[str, dict[str, Any]], dict[str, list[int]]]:
    before = _action_summaries(BASELINE / "LIVE_DIAGNOSTIC_RESULTS.json")
    after = _action_summaries(OPTIMIZED / "LIVE_DIAGNOSTIC_RESULTS.json")
    family: dict[str, dict[str, Any]] = {}
    publication_samples: dict[str, list[int]] = {}
    for family_name, action_id in FAMILY_ANCHORS.items():
        old = before[action_id]["timing"]["edge_total_edge_processing_ms"]
        new = after[action_id]["timing"]["edge_total_edge_processing_ms"]
        delta_ms = float(old["median"]) - float(new["median"])
        _require(delta_ms > 0.0, f"non-positive optimization delta for {family_name}")
        per_frame = next((OPTIMIZED / "per_frame").glob(f"action_{action_id}_*.csv"))
        samples: list[int] = []
        for row in _read_csv(per_frame):
            value = _f(row.get("tail_output_serialization_ms", ""))
            if value is not None and value > 0:
                samples.append(max(1, int(round(value * 1_000_000))))
        _require(bool(samples), f"no publication samples for {family_name}")
        publication_samples[family_name] = samples
        family[family_name] = {
            "anchor_action_id": action_id,
            "baseline_total_edge_processing_ms_median": float(old["median"]),
            "optimized_total_edge_processing_ms_median": float(new["median"]),
            "total_edge_processing_reduction_ms": delta_ms,
            "optimized_publication_ms_median": statistics.median(samples) / 1e6,
            "optimized_publication_samples": len(samples),
            "application": (
                "subtract the family median reduction from each original per-frame "
                "total_edge_processing duration, then reschedule; never subtract from AoI"
            ),
        }
    return family, publication_samples


def _attempt_manifest_hash(attempt: Path, relative: str) -> str:
    manifest = _load_json(attempt / "manifest.json")
    matches = [item for item in manifest["files"] if item["path"] == relative]
    _require(len(matches) == 1, f"{attempt}: manifest lacks unique {relative}")
    path = attempt / relative
    _require(_sha256(path) == matches[0]["sha256"], f"{path}: hash drift")
    return str(matches[0]["sha256"])


def _attempt(cell: Mapping[str, str]) -> Path:
    path = REPO / cell["source_campaign_root"] / cell["source_attempt"]
    _require(path.is_dir(), f"source attempt absent: {path}")
    return path


def _sent_rows(attempt: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(attempt / "per_frame_metrics.csv")
        if row.get("prepare_status") == "SENT"
    ]


def _old_service_ns(row: Mapping[str, str]) -> int | None:
    timing = _pydict(row.get("edge_timing_ns", ""))
    value = _i(timing.get("total_edge_processing", ""))
    return value if value is not None and value > 0 else None


def _post_install_ns(row: Mapping[str, str]) -> int | None:
    installed = _f(row.get("map_installed_at", ""))
    tail_complete = _f(row.get("edge_tail_complete_wall_s", ""))
    if installed is None or tail_complete is None:
        return None
    value = int(round((installed - tail_complete) * 1e9))
    return value if 0 <= value <= 1_000_000_000 else None


def _measured_install_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    captures = [int(round(float(row["capture_wall_s"]) * 1e9)) for row in rows]
    _require(bool(captures), "cannot summarize an empty measured cell")
    events: list[tuple[int, int]] = []
    for row, capture_ns in zip(rows, captures):
        installed = _f(row.get("map_installed_at", ""))
        if installed is not None:
            events.append((int(round(installed * 1e9)), capture_ns))
    events.sort()
    useful: list[tuple[int, int]] = []
    newest_capture = -1
    for install_ns, capture_ns in events:
        if capture_ns > newest_capture:
            useful.append((install_ns, capture_ns))
            newest_capture = capture_ns
    install_aoi_ms = [(install - capture) / 1e6 for install, capture in events]
    observation_end = max(captures) + HORIZON_NS
    relevant = [event for event in useful if event[0] < observation_end]
    if not relevant:
        weighted_aoi_ms = None
        above_fraction = None
        observation_s = 0.0
    else:
        evaluation_start = max(min(captures), relevant[0][0])
        area_ns2 = 0.0
        above_ns = 0
        for index, (install_ns, capture_ns) in enumerate(relevant):
            interval_start = max(evaluation_start, install_ns)
            interval_end = (
                min(observation_end, relevant[index + 1][0])
                if index + 1 < len(relevant)
                else observation_end
            )
            if interval_end <= interval_start:
                continue
            start_age = interval_start - capture_ns
            duration = interval_end - interval_start
            area_ns2 += float(start_age) * duration + 0.5 * float(duration) ** 2
            crossing = capture_ns + SERVICE_TARGET_NS
            above_ns += max(0, interval_end - max(interval_start, crossing))
        observation_ns = observation_end - evaluation_start
        weighted_aoi_ms = area_ns2 / observation_ns / 1e6
        above_fraction = above_ns / observation_ns
        observation_s = observation_ns / 1e9
    return {
        "installed": len(events),
        "useful_newer_map_installations": len(useful),
        "installed_within_100ms": sum(value <= 100.0 for value in install_aoi_ms),
        "installed_within_500ms": sum(value <= 500.0 for value in install_aoi_ms),
        "install_aoi_ms_median": _percentile(install_aoi_ms, 0.5),
        "install_aoi_ms_p95": _percentile(install_aoi_ms, 0.95),
        "time_weighted_map_aoi_ms": weighted_aoi_ms,
        "map_time_above_service_target_fraction": above_fraction,
        "map_aoi_observation_duration_s": observation_s,
    }


def _build_empirical_pools(
    cells: Sequence[Mapping[str, str]],
) -> tuple[
    dict[int, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[tuple[int, str], list[int]],
    dict[str, list[int]],
    dict[str, str],
]:
    action_services: dict[int, list[int]] = defaultdict(list)
    family_services: dict[str, list[int]] = defaultdict(list)
    profile_install_delays: dict[str, list[int]] = defaultdict(list)
    action_profile_arrival_delays: dict[tuple[int, str], list[int]] = defaultdict(list)
    profile_arrival_delays: dict[str, list[int]] = defaultdict(list)
    per_frame_hashes: dict[str, str] = {}
    for number, cell in enumerate(cells, start=1):
        attempt = _attempt(cell)
        per_frame_hashes[cell["cell_id"]] = _attempt_manifest_hash(
            attempt, "per_frame_metrics.csv"
        )
        action_id = int(cell["action_id"])
        profile = cell["network_profile"]
        for row in _sent_rows(attempt):
            service = _old_service_ns(row)
            if service is not None:
                action_services[action_id].append(service)
                family_services[cell["family"]].append(service)
            delay = _post_install_ns(row)
            if delay is not None:
                profile_install_delays[profile].append(delay)
            receipt = _f(row.get("edge_receipt_wall_s", ""))
            capture = _f(row.get("capture_wall_s", ""))
            if receipt is not None and capture is not None and receipt >= capture:
                arrival_delay = int(round((receipt - capture) * 1e9))
                action_profile_arrival_delays[(action_id, profile)].append(
                    arrival_delay
                )
                profile_arrival_delays[profile].append(arrival_delay)
        if number % 24 == 0:
            print(f"pool pass: {number}/{len(cells)} cells", flush=True)
    _require(len(family_services) == 4, "not every family has an observed service pool")
    _require(len(profile_install_delays) == 4, "not every profile has install-delay data")
    _require(len(profile_arrival_delays) == 4, "not every profile has arrival-delay data")
    return (
        action_services,
        family_services,
        profile_install_delays,
        action_profile_arrival_delays,
        profile_arrival_delays,
        per_frame_hashes,
    )


def _candidate_frames(
    *,
    cell: Mapping[str, str],
    rows: Sequence[Mapping[str, str]],
    family_calibration: Mapping[str, Any],
    publication_samples: Sequence[int],
    action_service_pool: Sequence[int],
    family_service_pool: Sequence[int],
    profile_delay_pool: Sequence[int],
    action_profile_arrival_pool: Sequence[int],
    profile_arrival_pool: Sequence[int],
) -> tuple[list[TwoStageFrame], dict[str, int]]:
    ordered = sorted(rows, key=lambda row: float(row["capture_wall_s"]))
    local_services = [
        (int(round(float(row["capture_wall_s"]) * 1e9)), service)
        for row in ordered
        if (service := _old_service_ns(row)) is not None
    ]
    local_service_positions = [item[0] for item in local_services]
    delta_ns = int(
        round(float(family_calibration["total_edge_processing_reduction_ms"]) * 1e6)
    )
    frames: list[TwoStageFrame] = []
    counters = {
        "service_observed": 0,
        "service_imputed_within_cell": 0,
        "service_imputed_same_action": 0,
        "service_imputed_same_family": 0,
        "install_delay_observed": 0,
        "install_delay_imputed": 0,
        "arrival_observed": 0,
        "arrival_imputed_within_cell": 0,
        "arrival_imputed_same_action_profile": 0,
        "arrival_imputed_same_profile": 0,
        "measured_reassemblies": int(cell["edge_complete_reassemblies"]),
        "measured_edge_admissions": int(cell["edge_admissions"]),
    }
    measured_reassemblies = counters["measured_reassemblies"]
    measured_admissions = counters["measured_edge_admissions"]
    _require(
        0 <= measured_admissions <= measured_reassemblies <= len(ordered),
        f"{cell['cell_id']}: upstream count ordering is invalid",
    )
    known_arrivals = {
        index
        for index, row in enumerate(ordered)
        if _f(row.get("edge_receipt_wall_s", "")) is not None
    }
    _require(
        len(known_arrivals) <= measured_admissions,
        f"{cell['cell_id']}: observed arrivals exceed admissions",
    )
    unknown = [index for index in range(len(ordered)) if index not in known_arrivals]
    imputed_admissions = _systematic_select(
        unknown, measured_admissions - len(known_arrivals)
    )
    remaining = [index for index in unknown if index not in imputed_admissions]
    measured_pre_queue_rejections = _systematic_select(
        remaining, measured_reassemblies - measured_admissions
    )
    local_arrival_pool = [
        int(
            round(
                (
                    float(row["edge_receipt_wall_s"])
                    - float(row["capture_wall_s"])
                )
                * 1e9
            )
        )
        for row in ordered
        if _f(row.get("edge_receipt_wall_s", "")) is not None
    ]
    for sequence, row in enumerate(ordered):
        capture_ns = int(round(float(row["capture_wall_s"]) * 1e9))
        arrival_s = _f(row.get("edge_receipt_wall_s", ""))
        pre_scheduler_reason: TwoStageReason | None = None
        if arrival_s is not None:
            arrival_ns = int(round(arrival_s * 1e9))
            counters["arrival_observed"] += 1
        elif sequence in imputed_admissions:
            if action_profile_arrival_pool:
                arrival_pool = action_profile_arrival_pool
                arrival_source = "arrival_imputed_same_action_profile"
            else:
                arrival_pool = profile_arrival_pool
                arrival_source = "arrival_imputed_same_profile"
            if local_arrival_pool:
                arrival_pool = local_arrival_pool
                arrival_source = "arrival_imputed_within_cell"
            arrival_ns = capture_ns + _deterministic_sample(
                arrival_pool,
                f"arrival:{cell['cell_id']}:{row['frame_id']}",
            )
            counters[arrival_source] += 1
        elif sequence in measured_pre_queue_rejections:
            arrival_ns = None
            pre_scheduler_reason = TwoStageReason.MEASURED_PRE_QUEUE_REJECTION
        else:
            arrival_ns = None
            pre_scheduler_reason = TwoStageReason.TRANSPORT_INCOMPLETE
        if arrival_ns is None:
            compute_ns = 1
            publication_ns = 1
            install_delay = 0
            observed_service = False
            observed_install = False
        else:
            old_service = _old_service_ns(row)
            observed_service = old_service is not None
            if old_service is None:
                if local_services:
                    old_service = _linear_interpolate_ordered(
                        local_services, local_service_positions, capture_ns
                    )
                    service_source = "service_imputed_within_cell"
                elif action_service_pool:
                    old_service = _deterministic_sample(
                        action_service_pool,
                        f"service:{cell['cell_id']}:{row['frame_id']}",
                    )
                    service_source = "service_imputed_same_action"
                else:
                    old_service = _deterministic_sample(
                        family_service_pool,
                        f"family-service:{cell['cell_id']}:{row['frame_id']}",
                    )
                    service_source = "service_imputed_same_family"
            else:
                service_source = "service_observed"
            publication_ns = _deterministic_sample(
                publication_samples,
                f"publication:{cell['cell_id']}:{row['frame_id']}",
            )
            candidate_total = max(
                publication_ns + 1_000_000, int(old_service) - delta_ns
            )
            compute_ns = candidate_total - publication_ns
            install_delay = _post_install_ns(row)
            observed_install = install_delay is not None
            if install_delay is None:
                install_delay = _deterministic_sample(
                    profile_delay_pool,
                    f"install:{cell['cell_id']}:{row['frame_id']}",
                )
            counters[service_source] += 1
            counters[
                "install_delay_observed"
                if observed_install
                else "install_delay_imputed"
            ] += 1
        frames.append(
            TwoStageFrame(
                frame_id=int(row["frame_id"]),
                sequence_id=sequence,
                capture_ns=capture_ns,
                arrival_ns=arrival_ns,
                compute_ns=compute_ns,
                publication_ns=publication_ns,
                post_publication_install_ns=int(install_delay),
                feature_bytes=int(row["payload_bytes"]),
                pre_scheduler_reason=pre_scheduler_reason,
                service_observed=observed_service,
                install_delay_observed=observed_install,
            )
        )
    return frames, counters


def _quality_by_action() -> dict[int, dict[str, Any]]:
    rows = _read_csv(CONSOLIDATION / "action_72_summary.csv")
    fields = (
        "val_vehicle_precision",
        "val_vehicle_recall",
        "val_vehicle_f1",
        "val_vehicle_xy_mae_m",
        "val_vehicle_iou",
        "val_canonical_person_precision",
        "val_canonical_person_recall",
        "val_canonical_person_f1",
        "val_canonical_person_xy_mae_m",
        "val_person_box_mask_iou",
        "val_segmentation_miou",
    )
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        result[int(row["action_id"])] = {field: _f(row.get(field, "")) for field in fields}
    _require(len(result) == 72, "quality table does not contain 72 actions")
    return result


def _flatten_cell(
    cell: Mapping[str, str],
    summary: Mapping[str, Any],
    counters: Mapping[str, int],
    quality: Mapping[str, Any],
    measured_install: Mapping[str, Any],
    per_frame_sha256: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "cell_id": cell["cell_id"],
        "action_id": int(cell["action_id"]),
        "profile_id": cell["profile_id"],
        "network_profile": cell["network_profile"],
        "family": cell["family"],
        "quantizer": cell["quantizer"],
        "q": float(cell["q"]),
        "q_e4": int(cell["q_e4"]),
        "keep_count": int(cell["keep_count"]),
        "source_per_frame_sha256": per_frame_sha256,
        "source_frames_sent": int(cell["frames_sent"]),
        "source_maps_installed": int(cell["maps_installed"]),
        "source_rate_installed_per_sent": _f(cell["rate_installed_per_sent"]),
        **counters,
        **quality,
    }
    for key, value in measured_install.items():
        output[f"source_{key}"] = value
    for key, value in summary.items():
        if key == "reason_counts":
            for reason, count in value.items():
                output[f"reason_{reason}"] = count
        else:
            output[key] = value
    output["rate_published_per_sent"] = (
        output["edge_results_published"] / output["input_frames"]
    )
    output["rate_installed_per_sent"] = (
        output["ack_installed_frames"] / output["input_frames"]
    )
    output["rate_useful_install_per_sent"] = (
        output["useful_newer_map_installations"] / output["input_frames"]
    )
    output["source_rate_useful_install_per_sent"] = (
        output["source_useful_newer_map_installations"] / output["input_frames"]
    )
    output["delta_rate_installed_per_sent"] = (
        output["rate_installed_per_sent"] - output["source_rate_installed_per_sent"]
    )
    source_aoi = output["source_time_weighted_map_aoi_ms"]
    candidate_aoi = output["time_weighted_map_aoi_ms"]
    output["delta_time_weighted_map_aoi_ms"] = (
        None
        if source_aoi is None or candidate_aoi is None
        else candidate_aoi - source_aoi
    )
    return output


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def block(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        sent = sum(int(item["input_frames"]) for item in items)
        installed = sum(int(item["ack_installed_frames"]) for item in items)
        useful = sum(int(item["useful_newer_map_installations"]) for item in items)
        source_installed = sum(int(item["source_installed"]) for item in items)
        source_useful = sum(
            int(item["source_useful_newer_map_installations"]) for item in items
        )
        return {
            "cells": len(items),
            "sent": sent,
            "transport_incomplete": sum(
                int(item["reason_TRANSPORT_INCOMPLETE"]) for item in items
            ),
            "measured_pre_queue_rejections": sum(
                int(item["reason_MEASURED_PRE_QUEUE_REJECTION"])
                for item in items
            ),
            "published": sum(int(item["edge_results_published"]) for item in items),
            "ack_installed": installed,
            "useful_newer_map_installations": useful,
            "installed_within_100ms": sum(
                int(item["installed_within_100ms"]) for item in items
            ),
            "installed_within_500ms": sum(
                int(item["installed_within_500ms"]) for item in items
            ),
            "rate_installed_per_sent": installed / sent if sent else None,
            "rate_useful_install_per_sent": useful / sent if sent else None,
            "source_ack_installed": source_installed,
            "source_useful_newer_map_installations": source_useful,
            "source_installed_within_100ms": sum(
                int(item["source_installed_within_100ms"]) for item in items
            ),
            "source_installed_within_500ms": sum(
                int(item["source_installed_within_500ms"]) for item in items
            ),
            "source_rate_installed_per_sent": (
                source_installed / sent if sent else None
            ),
            "source_rate_useful_install_per_sent": (
                source_useful / sent if sent else None
            ),
            "time_weighted_map_aoi_ms_cell_median": _percentile(
                [
                    float(item["time_weighted_map_aoi_ms"])
                    for item in items
                    if item["time_weighted_map_aoi_ms"] is not None
                ],
                0.5,
            ),
            "install_aoi_ms_cell_median": _percentile(
                [
                    float(item["install_aoi_ms_median"])
                    for item in items
                    if item["install_aoi_ms_median"] is not None
                ],
                0.5,
            ),
            "source_time_weighted_map_aoi_ms_cell_median": _percentile(
                [
                    float(item["source_time_weighted_map_aoi_ms"])
                    for item in items
                    if item["source_time_weighted_map_aoi_ms"] is not None
                ],
                0.5,
            ),
            "source_install_aoi_ms_cell_median": _percentile(
                [
                    float(item["source_install_aoi_ms_median"])
                    for item in items
                    if item["source_install_aoi_ms_median"] is not None
                ],
                0.5,
            ),
            "feature_bytes_charged": sum(
                int(item["feature_bytes_charged"]) for item in items
            ),
            "feature_bytes_without_useful_install": sum(
                int(item["feature_bytes_without_useful_install"]) for item in items
            ),
            "arrival_timestamps_observed": sum(
                int(item["arrival_observed"]) for item in items
            ),
            "arrival_timestamps_imputed": sum(
                int(item["arrival_imputed_within_cell"])
                + int(item["arrival_imputed_same_action_profile"])
                + int(item["arrival_imputed_same_profile"])
                for item in items
            ),
            "edge_service_durations_observed": sum(
                int(item["service_observed"]) for item in items
            ),
            "edge_service_durations_imputed": sum(
                int(item["service_imputed_within_cell"])
                + int(item["service_imputed_same_action"])
                + int(item["service_imputed_same_family"])
                for item in items
            ),
        }

    profiles = sorted({str(item["network_profile"]) for item in rows})
    return {
        "overall": block(rows),
        "by_network_profile": {
            profile: block([item for item in rows if item["network_profile"] == profile])
            for profile in profiles
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "refusing to write an empty CSV")
    columns = list(rows[0])
    _require(all(list(row) == columns for row in rows), "CSV columns drift")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, document: Any) -> None:
    _atomic_text(
        path,
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _report(document: Mapping[str, Any]) -> str:
    overall = document["aggregate"]["overall"]
    lines = [
        "# SplitFusion post-optimization 288-cell counterfactual",
        "",
        "All 288 measured fixed-action cells were replayed offline. Measured feature",
        "capture times, bytes, and per-cell reassembly/admission totals were preserved.",
        "Where frame-level arrival identities were not retained, they were imputed",
        "deterministically. The edge service component was then replaced with the",
        "measured family calibration. The qualified detached two-stage latest-only",
        "scheduler and 25 ms",
        "pre-compute expiry ceiling were simulated.",
        "",
        "This is a counterfactual training model, not a new live measurement and not",
        "evidence of 100 ms service readiness.",
        "",
        "## Aggregate",
        "",
        "| case | installed | useful installs | install AoI <=100 ms | install AoI <=500 ms | install/sent | map AoI cell median |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| measured source | {overall['source_ack_installed']} | "
            f"{overall['source_useful_newer_map_installations']} | "
            f"{overall['source_installed_within_100ms']} | "
            f"{overall['source_installed_within_500ms']} | "
            f"{overall['source_rate_installed_per_sent']:.4f} | "
            f"{overall['source_time_weighted_map_aoi_ms_cell_median']:.1f} ms |"
        ),
        (
            f"| counterfactual | {overall['ack_installed']} | "
            f"{overall['useful_newer_map_installations']} | "
            f"{overall['installed_within_100ms']} | "
            f"{overall['installed_within_500ms']} | "
            f"{overall['rate_installed_per_sent']:.4f} | "
            f"{overall['time_weighted_map_aoi_ms_cell_median']:.1f} ms |"
        ),
        "",
        "Upstream accounting held fixed: 896,856 sent, 184,124 transport",
        "incomplete, and 110,417 measured pre-queue rejections.",
        (
            f"Of {overall['arrival_timestamps_observed'] + overall['arrival_timestamps_imputed']:,} "
            f"edge admissions, {overall['arrival_timestamps_observed']:,} had retained "
            "frame-level arrival timestamps and "
            f"{overall['arrival_timestamps_imputed']:,} were imputed."
        ),
        "",
        "## Network-profile behavior",
        "",
        "| profile | measured install/sent | candidate install/sent | measured map AoI | candidate map AoI |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile, item in document["aggregate"]["by_network_profile"].items():
        aoi = item["time_weighted_map_aoi_ms_cell_median"]
        lines.append(
            f"| {profile} | {item['source_rate_installed_per_sent']:.4f} | "
            f"{item['rate_installed_per_sent']:.4f} | "
            f"{item['source_time_weighted_map_aoi_ms_cell_median']:.1f} ms | "
            f"{'—' if aoi is None else f'{aoi:.1f} ms'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Optimization is applied to the measured edge service duration, never",
            "  by subtracting a constant from measured AoI.",
            "- The family median service reduction is extrapolated from one live anchor",
            "  per family; quantizer/q-specific optimization variance is not measured.",
            "- Missing old service durations and compact-result install delays are",
            "  deterministically imputed from within-action/profile empirical pools.",
            "- Per-cell reassembly and edge-admission counts are preserved exactly.",
            "  Missing admitted identities and arrival delays are deterministically",
            "  imputed and never presented as measured observations.",
            "- Perception quality is the immutable validation quality of each action;",
            "  failed or superseded frames receive no installation utility.",
            "- The four live scheduling cells selected 25 ms provisionally. They do not",
            "  establish run-to-run variance or universal optimality.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output: Path) -> dict[str, Any]:
    _require(not output.exists(), f"create-only output exists: {output}")
    provenance = _verify_sources()
    family, publication_samples = _calibration()
    cells = _read_csv(CONSOLIDATION / "campaign_288_cell_table.csv")
    _require(len(cells) == 288, "campaign table is not 288 cells")
    _require(len({row["cell_id"] for row in cells}) == 288, "duplicate cell ID")
    _require(
        {(int(row["action_id"]), row["network_profile"]) for row in cells}
        == {(action, profile) for action in range(72) for profile in (
            "FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"
        )},
        "72x4 inventory drift",
    )
    quality = _quality_by_action()
    (
        action_services,
        family_services,
        profile_delays,
        action_profile_arrivals,
        profile_arrivals,
        source_hashes,
    ) = _build_empirical_pools(cells)

    output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for number, cell in enumerate(cells, start=1):
        attempt = _attempt(cell)
        sent = _sent_rows(attempt)
        _require(len(sent) == int(cell["frames_sent"]), f"{cell['cell_id']}: sent drift")
        measured_install = _measured_install_summary(sent)
        _require(
            measured_install["installed"] == int(cell["maps_installed"]),
            f"{cell['cell_id']}: measured install count drift",
        )
        family_name = cell["family"]
        frames, counters = _candidate_frames(
            cell=cell,
            rows=sent,
            family_calibration=family[family_name],
            publication_samples=publication_samples[family_name],
            action_service_pool=action_services.get(int(cell["action_id"]), ()),
            family_service_pool=family_services[family_name],
            profile_delay_pool=profile_delays[cell["network_profile"]],
            action_profile_arrival_pool=action_profile_arrivals.get(
                (int(cell["action_id"]), cell["network_profile"]), ()
            ),
            profile_arrival_pool=profile_arrivals[cell["network_profile"]],
        )
        result = simulate_two_stage(
            frames,
            config=TwoStageConfig(
                queue_wait_budget_ns=WAIT_BUDGET_NS,
                processing_horizon_ns=HORIZON_NS,
                service_target_ns=SERVICE_TARGET_NS,
            ),
        )
        summary = result.summary()
        _require(summary["input_frames"] == len(sent), "input count drift")
        _require(sum(summary["reason_counts"].values()) == len(sent), "terminal drift")
        rows.append(
            _flatten_cell(
                cell,
                summary,
                counters,
                quality[int(cell["action_id"])],
                measured_install,
                source_hashes[cell["cell_id"]],
            )
        )
        flattened = rows[-1]
        _require(
            flattened["edge_scheduler_input_frames"]
            == flattened["measured_edge_admissions"],
            f"{cell['cell_id']}: scheduler admission count drift",
        )
        _require(
            flattened["reason_TRANSPORT_INCOMPLETE"]
            == flattened["input_frames"] - flattened["measured_reassemblies"],
            f"{cell['cell_id']}: transport-incomplete count drift",
        )
        _require(
            flattened["reason_MEASURED_PRE_QUEUE_REJECTION"]
            == flattened["measured_reassemblies"]
            - flattened["measured_edge_admissions"],
            f"{cell['cell_id']}: pre-queue rejection count drift",
        )
        _require(
            flattened["source_installed"] == flattened["source_maps_installed"],
            f"{cell['cell_id']}: source-install reconciliation drift",
        )
        if number % 24 == 0:
            print(f"simulation pass: {number}/{len(cells)} cells", flush=True)

    aggregate = _aggregate(rows)
    result_document = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scientific_status": "COUNTERFACTUAL_MODEL_NOT_LIVE_MEASUREMENT",
        "policy": {
            "scheduler": "LATEST_ONLY_DEPTH_ONE_TWO_STAGE",
            "pre_compute_queue_expiry_ms": 25,
            "expiry_is_hold": False,
            "compute_workers": 1,
            "publication_workers": 1,
            "running_cuda_preempted": False,
            "processing_horizon_ms": 500,
            "service_reference_ms": 100,
        },
        "provenance": provenance,
        "family_calibration": family,
        "inventory": {
            "cells": len(rows),
            "actions": len({row["action_id"] for row in rows}),
            "network_profiles": sorted({row["network_profile"] for row in rows}),
            "terminal_reconciled_cells": sum(
                sum(row[f"reason_{reason.value}"] for reason in TwoStageReason)
                == row["input_frames"]
                for row in rows
            ),
            "source_per_frame_hashes_verified": len(source_hashes),
            "actions_without_observed_service_samples": sorted(
                set(range(72)) - set(action_services)
            ),
        },
        "aggregate": aggregate,
        "limitations": [
            "family-level optimization calibration is extrapolated from four favorable-profile anchors",
            "missing source service and post-publication install delays are deterministic empirical imputations",
            "per-cell reassembly/admission counts are exact but missing admitted identities and arrival delays are deterministic imputations",
            "this fixed-action counterfactual surface is not yet a switching-policy trajectory",
            "no 100 ms service-readiness claim",
        ],
    }
    _write_csv(output / "counterfactual_288_cell_summary.csv", rows)
    _atomic_json(output / "counterfactual_results.json", result_document)
    _atomic_text(output / "REPORT.md", _report(result_document))
    artifact_hashes = {
        name: _sha256(output / name)
        for name in (
            "counterfactual_288_cell_summary.csv",
            "counterfactual_results.json",
            "REPORT.md",
        )
    }
    _atomic_json(
        output / "artifact_manifest.json",
        {"schema": f"{SCHEMA}.artifacts", "sha256": artifact_hashes},
    )
    _atomic_text(
        output / TERMINAL,
        json.dumps(
            {
                "schema": f"{SCHEMA}.terminal",
                "status": "COMPLETE",
                "results_sha256": artifact_hashes["counterfactual_results.json"],
                "cell_summary_sha256": artifact_hashes[
                    "counterfactual_288_cell_summary.csv"
                ],
            },
            sort_keys=True,
        )
        + "\n",
    )
    return result_document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document = run(args.output.resolve())
    print(json.dumps(document["aggregate"], indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
