#!/usr/bin/env python3
"""Reproduce and sweep the freshness scheduler on live four-action evidence."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .simulator import (
    QueuePolicy,
    SimulationConfig,
    SimulationFrame,
    SimulationReason,
    simulate,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / (
    "experiments/splitfusion_edge_optimization_v1/"
    "20260909_live_actions30_15_50_71"
)
DEFAULT_OUTPUT = REPO / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_live_diagnostic_sweep_v1"
)
SCHEMA = "scenesense.splitfusion.edge_freshness_sweep.v1"
TERMINAL = "SPLITFUSION_EDGE_FRESHNESS_SCHEDULER_SWEEP_COMPLETE"
PROCESSING_HORIZON_NS = 500_000_000
SERVICE_TARGET_NS = 100_000_000
WAIT_BUDGETS_MS = (0, 5, 10, 20, 25)
REPRODUCTION_TOLERANCE_FRAMES = 15


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _linear_interpolate(
    known: list[tuple[int, int]], query: int
) -> int:
    _require(bool(known), "cannot interpolate without measured values")
    positions = [item[0] for item in known]
    offset = bisect.bisect_left(positions, query)
    if offset <= 0:
        return int(known[0][1])
    if offset >= len(known):
        return int(known[-1][1])
    left_x, left_y = known[offset - 1]
    right_x, right_y = known[offset]
    if right_x == left_x:
        return int(left_y)
    fraction = (query - left_x) / (right_x - left_x)
    return int(round(left_y + fraction * (right_y - left_y)))


def _evenly_selected_indices(count: int, selected_count: int) -> set[int]:
    _require(0 <= selected_count <= count, "invalid deterministic selection")
    if selected_count == 0:
        return set()
    if selected_count == count:
        return set(range(count))
    # Midpoint-stratified positions avoid assigning all unknown outcomes to one
    # end of the trace. This is an explicit imputation rule, not observed truth.
    selected = {
        min(count - 1, (2 * rank + 1) * count // (2 * selected_count))
        for rank in range(selected_count)
    }
    _require(len(selected) == selected_count, "deterministic selection collided")
    return selected


def _float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "").strip()
    return None if not value else float(value)


def _int(row: dict[str, str], name: str) -> int | None:
    value = row.get(name, "").strip()
    return None if not value else int(value)


def _load_action_frames(
    path: Path, *, summary: dict[str, Any]
) -> tuple[list[SimulationFrame], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["ue_first_send_wall_ns"]]
    expected = int(summary["counts"]["transmitted_frames"])
    complete_count = int(summary["counts"]["complete_reassemblies"])
    _require(len(rows) == expected, f"{path}: transmitted row count drift")
    rows.sort(key=lambda row: int(row["ue_first_send_wall_ns"]))

    measured_delay: list[tuple[int, int]] = []
    measured_service: list[tuple[int, int]] = []
    missing_positions: list[int] = []
    for position, row in enumerate(rows):
        send_ns = int(row["ue_first_send_wall_ns"])
        arrival_ns = _int(row, "edge_complete_reassembly_wall_ns")
        if arrival_ns is None:
            missing_positions.append(position)
        else:
            measured_delay.append((send_ns, arrival_ns - send_ns))
        service_ms = _float(row, "edge_service_wall_ms")
        if service_ms is not None:
            measured_service.append((send_ns, int(round(service_ms * 1_000_000))))
    measured_delay.sort()
    measured_service.sort()
    _require(bool(measured_delay) and bool(measured_service), f"{path}: no timing")
    imputed_complete_count = complete_count - len(measured_delay)
    _require(
        0 <= imputed_complete_count <= len(missing_positions),
        f"{path}: complete-reassembly counter cannot reconcile",
    )
    selected_missing_offsets = _evenly_selected_indices(
        len(missing_positions), imputed_complete_count
    )
    selected_positions = {
        missing_positions[offset] for offset in selected_missing_offsets
    }

    frames: list[SimulationFrame] = []
    transport_incomplete: list[int] = []
    transport_incomplete_bytes = 0
    imputed_arrival_ids: list[int] = []
    imputed_service_ids: list[int] = []
    for sequence, row in enumerate(rows):
        send_ns = int(row["ue_first_send_wall_ns"])
        frame_id = int(row["frame_id"])
        arrival_ns = _int(row, "edge_complete_reassembly_wall_ns")
        arrival_observed = arrival_ns is not None
        if arrival_ns is None and sequence not in selected_positions:
            transport_incomplete.append(frame_id)
            transport_incomplete_bytes += int(row["payload_bytes"])
            continue
        if arrival_ns is None:
            delay_ns = max(0, _linear_interpolate(measured_delay, send_ns))
            arrival_ns = send_ns + delay_ns
            imputed_arrival_ids.append(frame_id)
        service_ms = _float(row, "edge_service_wall_ms")
        service_observed = service_ms is not None
        if service_ms is None:
            service_ns = max(1, _linear_interpolate(measured_service, send_ns))
            imputed_service_ids.append(frame_id)
        else:
            service_ns = max(1, int(round(service_ms * 1_000_000)))
        capture_ns = int(round(float(row["capture_wall_s"]) * 1_000_000_000))
        frames.append(
            SimulationFrame(
                frame_id=frame_id,
                sequence_id=sequence,
                capture_ns=capture_ns,
                arrival_ns=max(capture_ns, int(arrival_ns)),
                service_ns=service_ns,
                feature_bytes=int(row["payload_bytes"]),
                arrival_observed=arrival_observed,
                service_observed=service_observed,
            )
        )
    _require(len(frames) == complete_count, f"{path}: modeled arrivals drift")
    _require(
        len(transport_incomplete) == expected - complete_count,
        f"{path}: transport-incomplete attribution drift",
    )
    return frames, {
        "transmitted_frames": expected,
        "complete_reassemblies": complete_count,
        "transport_incomplete_count": len(transport_incomplete),
        "transport_incomplete_bytes": transport_incomplete_bytes,
        "transport_incomplete_frame_ids_assigned_by_rule": transport_incomplete,
        "observed_arrivals": len(measured_delay),
        "imputed_arrivals": len(imputed_arrival_ids),
        "imputed_arrival_frame_ids_sha256": hashlib.sha256(
            json.dumps(imputed_arrival_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "observed_services": len(measured_service),
        "imputed_services": len(imputed_service_ids),
        "imputed_service_frame_ids_sha256": hashlib.sha256(
            json.dumps(imputed_service_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "imputation_rule": (
            "linear interpolation over send time within the same action; "
            "unknown transport-incomplete identities selected by deterministic "
            "midpoint stratification"
        ),
    }


def _policy_name(policy: QueuePolicy, wait_ms: int | None) -> str:
    if policy is QueuePolicy.FIFO:
        return "fifo"
    return "latest_only" if wait_ms is None else f"latest_only_wait_{wait_ms}ms"


def _run_policies(
    frames: list[SimulationFrame], *, transmitted_frames: int
) -> list[dict[str, Any]]:
    definitions: list[tuple[QueuePolicy, int | None]] = [
        (QueuePolicy.FIFO, None),
        (QueuePolicy.LATEST_ONLY, None),
        *((QueuePolicy.LATEST_ONLY, value) for value in WAIT_BUDGETS_MS),
    ]
    rows: list[dict[str, Any]] = []
    for policy, wait_ms in definitions:
        result = simulate(
            frames,
            config=SimulationConfig(
                queue_policy=policy,
                queue_wait_budget_ns=(None if wait_ms is None else wait_ms * 1_000_000),
                processing_horizon_ns=PROCESSING_HORIZON_NS,
                service_target_ns=SERVICE_TARGET_NS,
            ),
        )
        summary = result.summary()
        summary["transmitted_frames"] = int(transmitted_frames)
        summary["transport_incomplete_frames"] = int(transmitted_frames) - len(frames)
        summary["installed_per_transmitted"] = (
            summary["installed_frames"] / transmitted_frames
        )
        rows.append({"policy_id": _policy_name(policy, wait_ms), **summary})
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "reason_counts"}
        for reason, count in row["reason_counts"].items():
            item[f"reason_{reason}"] = count
        flattened.append(item)
    _require(bool(flattened), "refusing to write empty CSV")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(flattened[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(flattened)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _report(document: dict[str, Any]) -> str:
    lines = [
        "# SplitFusion edge freshness scheduling sweep",
        "",
        "This is a counterfactual discrete-event simulation over the four-action",
        "live timing diagnostic. It does not modify or replace the measured",
        "288-cell evidence.",
        "",
        "## Evidence and reproduction",
        "",
        "Missing timestamps belong primarily to frames replaced by the live",
        "depth-one pending slot. Their arrivals and service times are explicitly",
        "imputed by the frozen within-action interpolation rule. Therefore the",
        "sweep is suitable for policy screening, not a new measurement claim.",
        "",
        "| action | measured completions | reproduced | measured replacements | reproduced | pass |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for action in document["actions"]:
        reproduction = action["reproduction"]
        lines.append(
            "| {action_id} | {measured_tail_completions} | {simulated_completions} | "
            "{measured_queue_replacements} | {simulated_replacements} | {passed_text} |".format(
                **reproduction,
                action_id=action["action_id"],
                passed_text="yes" if reproduction["passed"] else "no",
            )
        )
    lines.extend(("", "## Policy sweep", ""))
    for action in document["actions"]:
        lines.extend(
            (
                f"### Action {action['action_id']} — `{action['profile_id']}`",
                "",
                "| policy | installed | superseded | wait-expired | median AoI (ms) | map AoI (ms) | updates/s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in action["policies"]:
            counts = row["reason_counts"]
            lines.append(
                "| {policy_id} | {installed_frames} | {superseded} | {expired} | "
                "{median} | {map_aoi} | {rate} |".format(
                    policy_id=row["policy_id"],
                    installed_frames=row["installed_frames"],
                    superseded=counts[SimulationReason.SUPERSEDED_PENDING.value],
                    expired=counts[
                        SimulationReason.QUEUE_WAIT_BUDGET_EXCEEDED.value
                    ],
                    median=(
                        "—"
                        if row["install_aoi_ms_median"] is None
                        else f"{row['install_aoi_ms_median']:.1f}"
                    ),
                    map_aoi=(
                        "—"
                        if row["time_weighted_map_aoi_ms"] is None
                        else f"{row['time_weighted_map_aoi_ms']:.1f}"
                    ),
                    rate=(
                        "—"
                        if row["installed_updates_per_s"] is None
                        else f"{row['installed_updates_per_s']:.2f}"
                    ),
                )
            )
        lines.append("")
    lines.extend(
        (
            "## Interpretation boundary",
            "",
            "A lower queue budget may reduce update count while improving the",
            "freshness of accepted work. No policy is selected from installed",
            "count alone. Promotion requires joint consideration of time-weighted",
            "AoI, useful update rate, supersession/waste, and later GPU/live parity.",
            "",
        )
    )
    return "\n".join(lines)


def run(input_root: Path, output: Path) -> dict[str, Any]:
    _require(input_root.is_dir(), f"input evidence is absent: {input_root}")
    _require(not output.exists(), f"create-only output already exists: {output}")
    results_path = input_root / "LIVE_DIAGNOSTIC_RESULTS.json"
    manifest_path = input_root / "ARTIFACT_MANIFEST.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = source_manifest.get("sha256")
    _require(isinstance(registered, dict) and registered, "source manifest drift")
    for relative, expected_hash in registered.items():
        artifact = input_root / str(relative)
        _require(artifact.is_file(), f"registered source artifact absent: {relative}")
        _require(
            _sha256(artifact) == str(expected_hash),
            f"registered source artifact hash mismatch: {relative}",
        )
    source = json.loads(results_path.read_text(encoding="utf-8"))
    summaries = {int(item["action_id"]): item for item in source["action_summaries"]}
    _require(set(summaries) == {15, 30, 50, 71}, "four-action inventory drift")

    actions: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for per_frame_path in sorted((input_root / "per_frame").glob("action_*.csv")):
        with per_frame_path.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        action_id = int(first["action_id"])
        summary = summaries[action_id]
        frames, reconstruction = _load_action_frames(per_frame_path, summary=summary)
        reproduction_result = simulate(
            frames,
            config=SimulationConfig(QueuePolicy.LATEST_ONLY, None, None),
        ).summary()
        measured_completions = int(summary["counts"]["tail_completions"])
        measured_replacements = int(summary["counts"]["queue_replacements"])
        simulated_replacements = int(
            reproduction_result["reason_counts"][
                SimulationReason.SUPERSEDED_PENDING.value
            ]
            + reproduction_result["reason_counts"][
                SimulationReason.OUT_OF_ORDER_ARRIVAL.value
            ]
        )
        reproduction = {
            "measured_tail_completions": measured_completions,
            "simulated_completions": int(reproduction_result["installed_frames"]),
            "measured_queue_replacements": measured_replacements,
            "simulated_replacements": simulated_replacements,
        }
        reproduction["passed"] = (
            abs(reproduction["simulated_completions"] - measured_completions)
            <= REPRODUCTION_TOLERANCE_FRAMES
            and abs(simulated_replacements - measured_replacements)
            <= REPRODUCTION_TOLERANCE_FRAMES
        )
        policies = _run_policies(
            frames, transmitted_frames=reconstruction["transmitted_frames"]
        )
        for row in policies:
            all_rows.append(
                {
                    "action_id": action_id,
                    "profile_id": summary["profile_id"],
                    **row,
                }
            )
        actions.append(
            {
                "action_id": action_id,
                "profile_id": summary["profile_id"],
                "input_csv": str(per_frame_path.relative_to(REPO)),
                "input_csv_sha256": _sha256(per_frame_path),
                "reconstruction": reconstruction,
                "reproduction": reproduction,
                "policies": policies,
            }
        )
    _require(len(actions) == 4, "per-frame action inventory drift")
    reproduction_passed = all(item["reproduction"]["passed"] for item in actions)

    document = {
        "schema": SCHEMA,
        "status": "COMPLETE" if reproduction_passed else "REPRODUCTION_FAILED",
        "scope": "COUNTERFACTUAL_POLICY_SCREEN_NOT_LIVE_MEASUREMENT",
        "input_root": str(input_root.relative_to(REPO)),
        "input_results_sha256": _sha256(results_path),
        "input_artifact_manifest_sha256": _sha256(manifest_path),
        "input_artifacts_verified": len(registered),
        "implementation_hashes": {
            relative: _sha256(REPO / relative)
            for relative in (
                "rl_agent/splitfusion_edge_freshness_scheduler_v1/scheduler.py",
                "rl_agent/splitfusion_edge_freshness_scheduler_v1/simulator.py",
                "rl_agent/splitfusion_edge_freshness_scheduler_v1/run_sweep.py",
            )
        },
        "processing_horizon_ms": PROCESSING_HORIZON_NS / 1_000_000,
        "service_target_ms": SERVICE_TARGET_NS / 1_000_000,
        "queue_wait_budgets_ms": list(WAIT_BUDGETS_MS),
        "latest_only_without_wait_budget_included": True,
        "reproduction_tolerance_frames": REPRODUCTION_TOLERANCE_FRAMES,
        "reproduction_passed": reproduction_passed,
        "actions": actions,
    }
    report = _report(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".partial")
    _require(not staging.exists(), f"staging output already exists: {staging}")
    staging.mkdir()
    _atomic_json(staging / "sweep_results.json", document)
    _write_csv(staging / "policy_sweep.csv", all_rows)
    _atomic_text(staging / "REPORT.md", report)
    terminal = {
        "schema": "scenesense.splitfusion.edge_freshness_sweep_terminal.v1",
        "status": document["status"],
        "reproduction_passed": reproduction_passed,
        "results_sha256": _sha256(staging / "sweep_results.json"),
        "policy_sweep_sha256": _sha256(staging / "policy_sweep.csv"),
        "report_sha256": _sha256(staging / "REPORT.md"),
        "terminal": TERMINAL if reproduction_passed else "REPRODUCTION_FAILED",
    }
    _atomic_json(staging / TERMINAL, terminal)
    os.replace(staging, output)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = run(args.input.resolve(), args.output.resolve())
    print(document["status"])
    return 0 if document["reproduction_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
