#!/usr/bin/env python3
"""Two-cell live qualification of the optimized sensor-preparation path.

This is the final bounded check before reviewing readiness for the 288-cell
measurement campaign. It runs only actions 20 and 71 under FAVORABLE_STABLE,
with the already-qualified fresh OAI/CARLA lifecycle. It never launches the
288-cell campaign.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent import splitfusion_phase15_realtime_recovery_v1 as recovery
from rl_agent import ue_288_campaign_supervisor as supervisor


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "SPLITFUSION_PHASE15_PREPARATION_OPTIMIZATION"
SCHEMA = "scenesense.splitfusion_phase15_preparation_optimization.v1"
TERMINAL_READY = "SPLITFUSION_PHASE15_PREPARATION_PATH_READY"
TERMINAL_NOT_READY = "SPLITFUSION_PHASE15_PREPARATION_PATH_NOT_READY"
MATRIX = ((20, "FAVORABLE_STABLE"), (71, "FAVORABLE_STABLE"))
SOURCE_RECOVERY = (
    "experiments/splitfusion_phase15_realtime_recovery_v1/"
    "20260906_actions20_71_favorable_fade_retry1_reevaluated"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise supervisor.CampaignError(message)


def select_cells(cells: Sequence[supervisor.Cell]) -> list[supervisor.Cell]:
    by_key = {(cell.action_id, cell.network_profile_id): cell for cell in cells}
    selected = []
    for action_id, network in MATRIX:
        cell = by_key.get((action_id, network))
        require(cell is not None, f"registered cell missing: {action_id}/{network}")
        profile, family, _quantizer, _q_e4 = recovery.EXPECTED_IDENTITY[action_id]
        require(
            cell.profile_id == profile and cell.model_family == family,
            f"action identity drift for {action_id}",
        )
        selected.append(cell)
    require(len(selected) == 2, "preparation qualification must contain two cells")
    return selected


def bind_inputs() -> dict[str, Any]:
    inherited = recovery.bind_immutable_inputs()
    source = (ROOT / SOURCE_RECOVERY).resolve(strict=True)
    terminal = source / recovery.TERMINAL_VALIDATED
    require(terminal.is_file(), "completed real-time recovery terminal missing")
    return {
        **inherited,
        "source_recovery": SOURCE_RECOVERY,
        "source_recovery_evaluation_sha256": supervisor.sha256_file(
            source / "PROSPECTIVE_EVALUATION.json"
        ),
        "source_recovery_terminal_sha256": supervisor.sha256_file(terminal),
    }


def evaluate_readiness(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, holds: bool, detail: Any) -> None:
        checks[name] = {"holds": bool(holds), "detail": detail}

    add(
        "both_cells_pass_structural_and_teardown_checks",
        all(
            cell.get("terminal_status") == "PASSED"
            and cell.get("structural_status") == "PASS"
            and cell.get("cold_teardown_verified") is True
            for cell in cells
        ),
        {
            cell.get("cell_id"): [
                cell.get("terminal_status"), cell.get("structural_status"),
                cell.get("cold_teardown_verified"),
            ]
            for cell in cells
        },
    )
    add(
        "preparation_timing_is_complete",
        all(
            cell.get("median_radar_prepare_ms") is not None
            and cell.get("p95_radar_prepare_ms") is not None
            and cell.get("median_pre_front_compute_ms") is not None
            and cell.get("p95_pre_front_compute_ms") is not None
            and cell.get("median_sensor_wait_ms") is not None
            for cell in cells
        ),
        {
            cell.get("cell_id"): {
                "sensor_wait_median_ms": cell.get("median_sensor_wait_ms"),
                "radar_prepare_median_ms": cell.get("median_radar_prepare_ms"),
                "radar_prepare_p95_ms": cell.get("p95_radar_prepare_ms"),
                "pre_front_compute_median_ms": cell.get("median_pre_front_compute_ms"),
                "pre_front_compute_p95_ms": cell.get("p95_pre_front_compute_ms"),
            }
            for cell in cells
        },
    )
    add(
        "avoidable_pre_front_compute_keeps_up_with_100ms_arrival_period",
        all(
            cell.get("p95_pre_front_compute_ms") is not None
            and float(cell["p95_pre_front_compute_ms"]) < 100.0
            for cell in cells
        ),
        {cell.get("cell_id"): cell.get("p95_pre_front_compute_ms") for cell in cells},
    )
    add(
        "bounded_latest_frame_queues_remain_valid",
        all(
            cell.get("queue_depth_high_water_edge") is not None
            and int(cell["queue_depth_high_water_edge"]) <= 1
            and cell.get("counter_reconciliation_holds") is True
            for cell in cells
        ),
        {
            cell.get("cell_id"): [
                cell.get("queue_depth_high_water_edge"),
                cell.get("counter_reconciliation_holds"),
            ]
            for cell in cells
        },
    )
    add(
        "service_and_feedback_boundaries_are_distinct",
        all(
            float(cell.get("service_deadline_ms") or 0.0) == 100.0
            and float(cell.get("ack_timeout_ms") or 0.0) == 500.0
            for cell in cells
        ),
        {
            cell.get("cell_id"): [
                cell.get("service_deadline_ms"), cell.get("ack_timeout_ms")
            ]
            for cell in cells
        },
    )

    coverage = {
        cell.get("cell_id"): {
            "coverage": cell.get("preparation_coverage"),
            "target": cell.get("minimum_sensor_preparation_coverage"),
            "target_met": cell.get("preparation_coverage_met"),
            "sustainable_fps": cell.get("sustainable_preparation_fps"),
            "queue_replacements": cell.get("queue_replacement_frames"),
        }
        for cell in cells
    }
    performance = {
        "preparation_coverage_target_0_95": {
            "met_in_both_cells": all(
                bool(cell.get("preparation_coverage_met")) for cell in cells
            ),
            "target_weakened": False,
            "detail": coverage,
        },
        "service_target_100ms": {
            "met_in_both_cells": all(
                int(cell.get("service_on_time_installations") or 0) > 0
                for cell in cells
            ),
            "detail": {
                cell.get("cell_id"): {
                    "within_100ms": cell.get("service_on_time_installations"),
                    "within_500ms_ack_timeout": cell.get(
                        "ack_within_timeout_installations"
                    ),
                    "installed": cell.get("installed_frames_with_aoi"),
                }
                for cell in cells
            },
        },
    }
    failed = [name for name, value in checks.items() if not value["holds"]]
    return {
        "checks": checks,
        "failed_checks": failed,
        "preparation_path_ready": not failed,
        "performance_outcomes_not_validity_gates": performance,
        "campaign_rate_interpretation": (
            "If the unchanged 0.95 target is missed while the pre-front p95 "
            "keeps up with the 100 ms arrival period, report the measured "
            "sustainable synchronized CARLA rate; do not relabel it as 10 FPS."
        ),
    }


def write_report(
    path: Path, status: str, cells: Sequence[Mapping[str, Any]], readiness: Mapping[str, Any]
) -> None:
    lines = [
        "# Phase-15 sensor-preparation optimization qualification",
        "",
        f"- Terminal: `{status}`",
        "- Scope: actions 20 and 71 under `FAVORABLE_STABLE` only.",
        "- 288-cell campaign: not launched.",
        "- Service target: 100 ms; feedback timeout: 500 ms (reported separately).",
        "",
        "| Cell | Coverage | Sustainable FPS | Radar prepare med/p95 ms | Pre-front med/p95 ms | <=100 ms | <=500 ms | Median AoI ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell.get('cell_id')} | {cell.get('preparation_coverage')} | "
            f"{cell.get('sustainable_preparation_fps')} | "
            f"{cell.get('median_radar_prepare_ms')} / {cell.get('p95_radar_prepare_ms')} | "
            f"{cell.get('median_pre_front_compute_ms')} / {cell.get('p95_pre_front_compute_ms')} | "
            f"{cell.get('service_on_time_installations')} | "
            f"{cell.get('ack_within_timeout_installations')} | "
            f"{cell.get('median_install_aoi_ms')} |"
        )
    lines += ["", "## Readiness checks", ""]
    for name, value in readiness["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if value['holds'] else 'FAIL'}")
    lines += [
        "",
        "The 0.95 preparation target and 100 ms service target remain measured "
        "performance outcomes. They were not weakened or replaced by the 500 ms "
        "feedback timeout.",
    ]
    supervisor.write_create_only(path, "\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--carla-port", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.execute == TOKEN, "exact preparation-optimization token required")
    config_path = args.config.resolve(strict=True)
    config, cells, _hashes = supervisor.validate_static(config_path)
    require(
        config.get("campaign_kind") == "live_pilot_16"
        and config.get("authorization", {}).get("campaign_288_authorized") is False,
        "only the non-288 live-pilot contract is allowed",
    )
    immutable = bind_inputs()
    worktree = supervisor.verify_live_pilot_worktree()
    live_qualification = supervisor.verify_phase15_qualification(
        args.qualification_root
    )
    supervisor._phase15_gpu_audit()
    supervisor._require_phase15_application_cold(config)
    catalog = supervisor.read_catalog(config)
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_resolved_models(config, catalog)
    adapter = supervisor.repo_path(supervisor.adapter_value(config))
    selected = select_cells(cells)

    output = args.output_root.resolve(strict=False)
    experiments = (ROOT / "experiments").resolve(strict=True)
    try:
        output.relative_to(experiments)
    except ValueError as exc:
        raise supervisor.CampaignError("output must remain under experiments") from exc
    require(not output.exists(), f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)

    manifest = {
        "schema": SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": supervisor.sha256_file(config_path),
        "git": worktree,
        "phase15_live_qualification": live_qualification,
        "immutable_inputs": immutable,
        "matrix": [
            {
                "action_id": cell.action_id,
                "network_profile_id": cell.network_profile_id,
                "cell_id": cell.cell_id,
                "profile_id": cell.profile_id,
            }
            for cell in selected
        ],
        "required_cells": 2,
        "preregistered_interpretation": {
            "service_deadline_ms": 100,
            "ack_timeout_ms": 500,
            "preparation_coverage_target": 0.95,
            "avoidable_pre_front_p95_ceiling_ms": 100.0,
            "performance_thresholds_weakened": False,
            "full_288_campaign_authorized": False,
        },
        "started_at_unix_s": time.time(),
    }
    supervisor.write_create_only(
        output / "run_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    ledger_path = output / str(config["cell"]["resume_ledger"])
    ledger = supervisor.load_ledger(
        ledger_path, str(config["campaign_id"]), manifest["config_sha256"]
    )
    attempts = []
    for cell in selected:
        rows = ledger["cells"].setdefault(cell.cell_id, [])
        result = supervisor.run_one_cell(
            config=config, cell=cell, adapter=adapter, campaign_root=output,
            ledger_rows=rows, port=int(args.carla_port),
        )
        rows.append(result)
        ledger["updated_at_unix_s"] = time.time()
        supervisor.atomic_json(ledger_path, ledger)
        attempts.append((cell, result))

    evaluated = []
    for cell, result in attempts:
        attempt = output / "cells" / cell.cell_id / "attempts" / (
            f"attempt_{int(result['attempt']):04d}"
        )
        evaluated.append(recovery.evaluate_cell(attempt))
    readiness = evaluate_readiness(evaluated)
    status = TERMINAL_READY if readiness["preparation_path_ready"] else TERMINAL_NOT_READY
    document = {
        "schema": SCHEMA,
        "status": status,
        "classification": "PRE_288_PREPARATION_PATH_QUALIFICATION",
        "readiness": readiness,
        "cells": evaluated,
        "immutable_inputs": immutable,
        "full_288_campaign_authorized_or_launched": False,
        "finished_at_unix_s": time.time(),
    }
    result_path = output / "qualification.json"
    supervisor.write_create_only(
        result_path, json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    recovery.write_summary_csv(output / "cell_summary.csv", evaluated)
    write_report(output / "REPORT.md", status, evaluated, readiness)
    artifacts = [
        output / "run_manifest.json", ledger_path, result_path,
        output / "cell_summary.csv", output / "REPORT.md",
    ]
    supervisor.write_create_only(
        output / "artifact_manifest.json",
        json.dumps(
            {
                "schema": "scenesense.splitfusion_phase15_preparation_artifacts.v1",
                "files": [
                    {
                        "path": path.name,
                        "sha256": supervisor.sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in artifacts
                ],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
    )
    supervisor.write_create_only(output / status, status + "\n")
    print(json.dumps({"status": status, "failed_checks": readiness["failed_checks"]}, indent=2))
    return 0 if status == TERMINAL_READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
