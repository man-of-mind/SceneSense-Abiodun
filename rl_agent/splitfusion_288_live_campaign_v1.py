#!/usr/bin/env python3
"""Final fail-closed gate for the qualified 288-cell SplitFusion campaign."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent import ue_288_campaign_supervisor as supervisor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_288_campaign_v1.yaml"
DEFAULT_BINDING = ROOT / "rl_agent/configs/splitfusion_288_live_campaign_binding_v1.json"
TOKEN = "SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN"
BINDING_SCHEMA = "scenesense.splitfusion_288_live_campaign_binding.v1"
MANIFEST_SCHEMA = "scenesense.splitfusion_288_live_campaign_manifest.v1"
CONTINUATION_SCHEMA = supervisor.CONTINUATION_SCHEMA
COMPLETION_SCHEMA = "scenesense.splitfusion_288_live_campaign_completion.v1"
TERMINAL = "SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN_COMPLETE"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise supervisor.CampaignError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be a mapping: {path}")
    return value


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_record(name: str, record: Mapping[str, Any]) -> Path:
    path = repo_path(str(record.get("path", "")))
    require(path.is_file(), f"bound artifact is missing: {name}: {path}")
    require(
        supervisor.sha256_file(path) == str(record.get("sha256", "")),
        f"bound artifact SHA-256 drift: {name}",
    )
    return path


def verify_campaign_deployment(config: Mapping[str, Any]) -> None:
    """Validate the exact top-level map consumed by the live cell adapter."""

    deployment = config.get("deployment")
    require(isinstance(deployment, dict), "full campaign deployment bindings are missing")
    required = {
        "phase15_supervisor",
        "phase15_qualification_runner",
        "frozen_phase14_target_runtime",
        "perception_train_only_priors",
        "fcos_constructor_weights",
        "carla_lifecycle",
        "edge_image_dockerfile",
        "edge_image_entrypoint",
        "edge_base_compose",
        "edge_split_compose",
        "edge_launcher",
    }
    require(set(deployment) == required, "full campaign deployment inventory drift")
    for name, record in deployment.items():
        require(isinstance(record, dict), f"invalid campaign deployment binding: {name}")
        verify_record(f"deployment.{name}", record)
    weights = deployment["fcos_constructor_weights"]
    require(
        Path(str(weights["path"])).name == "fcos_resnet50_fpn_coco-99b0c9b7.pth",
        "FCOS constructor-weight cache filename drift",
    )


def verify_binding(config_path: Path, binding_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = load_json(binding_path)
    require(binding.get("schema") == BINDING_SCHEMA, "288 campaign binding schema drift")
    require(binding.get("status") == "READY_FOR_EXPLICIT_AUTHORIZATION", "288 campaign is not ready")
    require(binding.get("campaign_288_authorized") is False, "binding must not self-authorize the campaign")
    require(binding.get("execution_token") == TOKEN, "288 campaign execution token drift")

    configured = binding.get("campaign_config")
    require(isinstance(configured, dict), "campaign config binding is missing")
    require(repo_path(str(configured.get("path", ""))) == config_path, "wrong campaign config selected")
    verify_record("campaign_config", configured)

    implementation = binding.get("implementation")
    require(isinstance(implementation, dict), "campaign implementation bindings are missing")
    for name, record in implementation.items():
        require(isinstance(record, dict), f"invalid implementation binding: {name}")
        verify_record(name, record)

    evidence = binding.get("evidence")
    require(isinstance(evidence, dict), "campaign evidence bindings are missing")
    evidence_paths: dict[str, Path] = {}
    for name, record in evidence.items():
        require(isinstance(record, dict), f"invalid evidence binding: {name}")
        evidence_paths[name] = verify_record(name, record)

    pilot = load_json(evidence_paths["phase15_structural_pilot_qualification"])
    require(
        pilot.get("status") == "SPLITFUSION_16_CELL_LIVE_CARLA_OAI_PILOT_COMPLETE"
        and int(pilot.get("required_cells", -1)) == 16
        and int(pilot.get("valid_completed_cells", -1)) == 16
        and pilot.get("full_288_campaign_authorized") is False,
        "16-cell structural pilot evidence is incomplete",
    )
    pilot_cells = pilot.get("cells")
    require(
        isinstance(pilot_cells, list)
        and len(pilot_cells) == 16
        and all(row.get("cold_cleanup_verified") is True for row in pilot_cells),
        "16-cell pilot lifecycle/teardown evidence is incomplete",
    )

    recovery = load_json(evidence_paths["phase15_realtime_recovery_evaluation"])
    require(
        recovery.get("status") == "SPLITFUSION_PHASE15_REALTIME_RECOVERY_VALIDATED"
        and len(recovery.get("cells", [])) == 4
        and all(
            row.get("structural_status") == "PASS"
            and row.get("terminal_status") == "PASSED"
            and row.get("cold_teardown_verified") is True
            for row in recovery.get("cells", [])
        ),
        "real-time recovery evidence is incomplete",
    )

    preparation = load_json(evidence_paths["phase15_preparation_qualification"])
    prep_cells = preparation.get("cells")
    require(
        preparation.get("status") == "SPLITFUSION_PHASE15_PREPARATION_PATH_READY"
        and isinstance(prep_cells, list)
        and {(int(row["action_id"]), str(row["network_profile_id"])) for row in prep_cells}
        == {(20, "FAVORABLE_STABLE"), (71, "FAVORABLE_STABLE")}
        and all(
            row.get("structural_status") == "PASS"
            and row.get("terminal_status") == "PASSED"
            and row.get("preparation_coverage_met") is True
            and float(row.get("preparation_coverage", 0.0)) >= 0.95
            and row.get("cold_teardown_verified") is True
            for row in prep_cells
        ),
        "preparation-path qualification evidence is incomplete",
    )
    require(
        all(int(row.get("service_on_time_installations", -1)) == 0 for row in prep_cells),
        "100-ms outcome changed; this binding must preserve the measured non-service-ready result",
    )
    require(
        binding.get("scientific_interpretation", {}).get("missed_deadlines_are_measured_outcomes") is True
        and binding.get("scientific_interpretation", {}).get("claims_100ms_service_ready") is False,
        "campaign interpretation must not turn performance misses into integrity failures or service claims",
    )
    return binding, preparation


def validate(config_path: Path, binding_path: Path) -> dict[str, Any]:
    config, cells, trace_hashes = supervisor.validate_static(config_path)
    require(config.get("campaign_kind") == "full_288", "config is not the full 288-cell campaign")
    require(len(cells) == 288, "campaign does not enumerate exactly 288 cells")
    require(len({cell.action_id for cell in cells}) == 72, "campaign does not contain 72 actions")
    require(len({cell.network_profile_id for cell in cells}) == 4, "campaign does not contain four profiles")
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_resolved_models(config, supervisor.read_catalog(config))
    verify_campaign_deployment(config)
    binding, preparation = verify_binding(config_path, binding_path)
    return {
        "status": "SPLITFUSION_288_CELL_LIVE_CAMPAIGN_OFFLINE_PREFLIGHT_PASS",
        "external_processes_started": 0,
        "campaign_cells": len(cells),
        "actions": len({cell.action_id for cell in cells}),
        "network_profiles": len({cell.network_profile_id for cell in cells}),
        "cell_mapping_sha256": supervisor.cell_mapping_sha256(cells),
        "trace_prefix_hashes": trace_hashes,
        "binding_sha256": supervisor.sha256_file(binding_path),
        "campaign_288_authorized": False,
        "service_deadline_ms": int(config["cell"]["service_deadline_ms"]),
        "ack_timeout_ms": int(config["cell"]["ack_timeout_ms"]),
        "claims_100ms_service_ready": False,
        "qualified_preparation_cells": len(preparation["cells"]),
        "minimum_qualified_preparation_coverage": min(
            float(row["preparation_coverage"]) for row in preparation["cells"]
        ),
        "binding_status": binding["status"],
    }



def build_continuation(
    config: Mapping[str, Any],
    cells: Sequence[Any],
    output: Path,
    reuse_root: Path,
    rerun_cell_ids: Sequence[str],
    config_path: Path,
    binding_path: Path,
    amended_head: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Bind an immutable earlier campaign as the source of reused cells.

    Reused cells are referenced, never copied or rewritten. Each reference
    carries the source root, attempt directory, that campaign's own registered
    output set, and the terminal and manifest hashes, all revalidated before
    reuse. Cells named for rerun are deliberately excluded so they execute fresh
    under the amended contract.
    """

    reuse_root = reuse_root.resolve(strict=True)
    experiments = (ROOT / "experiments").resolve(strict=True)
    try:
        reuse_root.relative_to(experiments)
    except ValueError as exc:
        raise supervisor.CampaignError("reuse root must remain beneath experiments") from exc
    require(reuse_root != output.resolve(strict=False), "reuse root must differ from the fresh output")
    source_manifest_path = reuse_root / "campaign_manifest.json"
    source_ledger_path = reuse_root / str(config["cell"]["resume_ledger"])
    for artifact in (source_manifest_path, source_ledger_path):
        require(artifact.is_file(), f"reuse root artifact missing: {artifact}")
    source_manifest = load_json(source_manifest_path)
    source_ledger = load_json(source_ledger_path)
    require(
        source_manifest.get("campaign_id") == config["campaign_id"]
        and source_manifest.get("cell_mapping_sha256") == supervisor.cell_mapping_sha256(cells)
        and int(source_manifest.get("required_cells", -1)) == 288,
        "reuse root does not describe the same 288-cell campaign",
    )
    require(source_ledger.get("schema") == supervisor.LEDGER_SCHEMA, "reuse ledger schema drift")
    rerun = {str(value) for value in rerun_cell_ids}
    known = {cell.cell_id for cell in cells}
    require(rerun <= known, f"unknown --rerun-cell-id values: {sorted(rerun - known)}")
    seeded: dict[str, list[dict[str, Any]]] = {}
    reused: list[dict[str, Any]] = []
    for cell in cells:
        if cell.cell_id in rerun:
            continue
        rows = source_ledger.get("cells", {}).get(cell.cell_id) or []
        chosen = None
        for row in rows:
            if row.get("status") != "PASSED":
                continue
            attempt_dir = reuse_root / str(row["attempt_dir"])
            manifest_path = attempt_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            attempt_manifest = load_json(manifest_path)
            registered = list(attempt_manifest.get("registered_outputs") or [])
            if not registered:
                continue
            chosen = {
                "status": supervisor.REUSED_STATUS,
                "attempt": int(row.get("attempt", 1)),
                "source_campaign_root": str(reuse_root.relative_to(ROOT)),
                "source_attempt_dir": str(row["attempt_dir"]),
                "source_expected_outputs": registered,
                "source_manifest_sha256": supervisor.sha256_file(manifest_path),
                # Named source_* so no reader resolves it against the fresh root.
                "source_terminal": str(row.get("terminal", "")),
                "terminal_sha256": str(row.get("terminal_sha256", "")),
                "route_metrics_summary_present": bool(
                    (attempt_dir / supervisor.ROUTE_SUMMARY_NAME).is_file()
                ),
            }
            break
        if chosen is None:
            continue
        require(
            supervisor.reused_attempt_valid([chosen]),
            f"reused cell failed revalidation before reuse: {cell.cell_id}",
        )
        seeded[cell.cell_id] = [chosen]
        reused.append({"cell_id": cell.cell_id, **chosen})
    continuation = {
        "schema": CONTINUATION_SCHEMA,
        "source_campaign_root": str(reuse_root.relative_to(ROOT)),
        "source_campaign_manifest_sha256": supervisor.sha256_file(source_manifest_path),
        "source_campaign_ledger_sha256": supervisor.sha256_file(source_ledger_path),
        "source_starting_head": str(source_manifest.get("starting_head", "")),
        "source_campaign_config_sha256": str(source_manifest.get("campaign_config_sha256", "")),
        "source_binding_sha256": str(source_manifest.get("binding_sha256", "")),
        "amended_starting_head": str(amended_head),
        "amended_campaign_config_sha256": supervisor.sha256_file(config_path),
        "amended_binding_sha256": supervisor.sha256_file(binding_path),
        "amended_supervisor_sha256": supervisor.sha256_file(
            (ROOT / "rl_agent/ue_288_campaign_supervisor.py").resolve(strict=True)
        ),
        "amended_cell_adapter_sha256": supervisor.sha256_file(
            repo_path(str(config["runtime"]["required_route_b_split_cell_adapter"]))
        ),
        "amended_expected_outputs": list(config["cell"]["expected_outputs"]),
        "reused_cells": len(reused),
        "rerun_cell_ids": sorted(rerun),
        "reused_cells_predate_durable_route_summary_requirement": True,
        "reused_cell_inventory": reused,
    }
    return continuation, seeded

def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve(strict=True)
    binding_path = args.binding.resolve(strict=True)
    report = validate(config_path, binding_path)
    require(args.execute == TOKEN, "exact 288-cell campaign execution token is required")
    require(args.authorize_full_sweep, "full campaign requires --authorize-full-sweep")
    require(args.qualification_root is not None, "Phase-15 qualification root is required")
    require(args.pilot_ledger is not None, "16-cell pilot ledger is required")

    output = args.output_root.resolve(strict=False)
    experiments = (ROOT / "experiments").resolve(strict=True)
    try:
        output.relative_to(experiments)
    except ValueError as exc:
        raise supervisor.CampaignError("campaign output must remain beneath experiments") from exc
    worktree = supervisor.verify_live_pilot_worktree(owned_resume_root=output if args.resume else None)
    qualification = supervisor.verify_phase15_qualification(args.qualification_root)
    supervisor.verify_pilot_gate(args.pilot_ledger.resolve(strict=True))
    supervisor._phase15_gpu_audit()
    config = supervisor.load_yaml(config_path)
    supervisor._require_phase15_application_cold(config)

    manifest_path = output / "campaign_manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": supervisor.sha256_file(config_path),
        "binding_sha256": supervisor.sha256_file(binding_path),
        "runner_sha256": supervisor.sha256_file(Path(__file__).resolve()),
        "starting_head": worktree["head"],
        "declared_user_owned_dirty_paths": worktree["dirty_paths"],
        "cell_mapping_sha256": report["cell_mapping_sha256"],
        "required_cells": 288,
        "service_deadline_ms": report["service_deadline_ms"],
        "ack_timeout_ms": report["ack_timeout_ms"],
        "claims_100ms_service_ready": False,
        "phase15_live_qualification": qualification,
        "registered_outputs": list(config["cell"]["expected_outputs"]),
        "reuse_campaign_root": (
            str(args.reuse_campaign_root.resolve(strict=True).relative_to(ROOT))
            if args.reuse_campaign_root is not None else ""
        ),
        "rerun_cell_ids": sorted(str(value) for value in args.rerun_cell_id),
        "created_at_unix_s": None,
    }
    if args.resume:
        require(output.is_dir() and manifest_path.is_file(), "resume output/manifest is missing")
        existing = load_json(manifest_path)
        expected = dict(manifest)
        expected["created_at_unix_s"] = existing.get("created_at_unix_s")
        require(existing == expected, "immutable 288-cell campaign manifest drift")
    else:
        require(not output.exists(), f"create-only campaign output exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=False, exist_ok=False)
        manifest["created_at_unix_s"] = time.time()
        supervisor.write_create_only(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if args.reuse_campaign_root is not None:
            cells = supervisor.enumerate_cells(config)
            continuation, seeded = build_continuation(
                config, cells, output, args.reuse_campaign_root,
                args.rerun_cell_id, config_path, binding_path, worktree["head"],
            )
            supervisor.write_create_only(
                output / "campaign_continuation_binding.json",
                json.dumps(continuation, indent=2, sort_keys=True) + "\n",
            )
            supervisor.write_create_only(
                output / str(config["cell"]["resume_ledger"]),
                json.dumps(
                    {
                        "schema": supervisor.LEDGER_SCHEMA,
                        "campaign_id": str(config["campaign_id"]),
                        "config_sha256": supervisor.sha256_file(config_path),
                        "created_at_unix_s": time.time(),
                        "updated_at_unix_s": time.time(),
                        "continuation_binding_sha256": supervisor.sha256_file(
                            output / "campaign_continuation_binding.json"
                        ),
                        "cells": seeded,
                    },
                    indent=2, sort_keys=True,
                ) + "\n",
            )

    completion_path = output / "campaign_completion.json"
    terminal_path = output / TERMINAL
    if completion_path.exists() or terminal_path.exists():
        require(args.resume and completion_path.is_file() and terminal_path.is_file(), "partial/foreign campaign completion record")
        completion = load_json(completion_path)
        terminal = load_json(terminal_path)
        completed_ledger = output / str(config["cell"]["resume_ledger"])
        require(
            completion.get("schema") == COMPLETION_SCHEMA
            and completion.get("status") == TERMINAL
            and int(completion.get("completed_cells", -1)) == 288
            and completion.get("campaign_manifest_sha256") == supervisor.sha256_file(manifest_path)
            and completed_ledger.is_file()
            and completion.get("campaign_ledger_sha256") == supervisor.sha256_file(completed_ledger)
            and terminal.get("status") == TERMINAL
            and terminal.get("completion_sha256") == supervisor.sha256_file(completion_path),
            "completed campaign record failed validation",
        )
        return 0

    forwarded = argparse.Namespace(
        config=config_path,
        output_root=output,
        model=[],
        route_b_split_cell_adapter=None,
        carla_port=args.carla_port,
        authorize_full_sweep=True,
        pilot_ledger=args.pilot_ledger.resolve(strict=True),
        cell_id=None,
        maximum_loop_sim_s=None,
        execute=TOKEN,
        qualification_root=args.qualification_root.resolve(strict=True),
        resume=args.resume,
    )
    result = int(supervisor.run_campaign(forwarded))
    if result != 0:
        return result

    ledger_path = output / str(config["cell"]["resume_ledger"])
    ledger = load_json(ledger_path)
    cells = supervisor.enumerate_cells(config)
    expected_outputs = config["cell"]["expected_outputs"]
    require(
        set(ledger.get("cells", {})) == {cell.cell_id for cell in cells},
        "campaign ledger contains duplicate, missing or foreign cells",
    )
    executed = 0
    reused = 0
    for cell in cells:
        rows = ledger["cells"].get(cell.cell_id, [])
        if supervisor.passed_attempt_exists(output, rows, expected_outputs):
            executed += 1
        elif supervisor.reused_attempt_valid(rows):
            reused += 1
    passed = executed + reused
    require(passed == 288 and len(ledger.get("cells", {})) == 288, "campaign completion inventory drift")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": TERMINAL,
        "completed_cells": passed,
        "executed_cells": executed,
        "reused_cells": reused,
        "cell_mapping_sha256": report["cell_mapping_sha256"],
        "campaign_manifest_sha256": supervisor.sha256_file(manifest_path),
        "campaign_ledger_sha256": supervisor.sha256_file(ledger_path),
        "low_delivery_or_missed_deadline_is_a_measured_outcome": True,
        "claims_100ms_service_ready": False,
        "finished_at_unix_s": time.time(),
    }
    supervisor.write_create_only(completion_path, json.dumps(completion, indent=2, sort_keys=True) + "\n")
    supervisor.write_create_only(
        terminal_path,
        json.dumps({
            "status": TERMINAL,
            "completion_sha256": supervisor.sha256_file(completion_path),
        }, indent=2, sort_keys=True) + "\n",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="offline-only binding reconciliation")
    launch = subparsers.add_parser("run", help="launch or resume the explicitly authorized campaign")
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--qualification-root", type=Path, required=True)
    launch.add_argument("--pilot-ledger", type=Path, required=True)
    launch.add_argument("--execute", required=True)
    launch.add_argument("--authorize-full-sweep", action="store_true")
    launch.add_argument("--resume", action="store_true")
    launch.add_argument("--reuse-campaign-root", type=Path, default=None)
    launch.add_argument("--rerun-cell-id", action="append", default=[])
    launch.add_argument("--carla-port", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(validate(args.config.resolve(strict=True), args.binding.resolve(strict=True)), indent=2, sort_keys=True))
            return 0
        return run(args)
    except supervisor.CampaignError as exc:
        print(f"campaign contract error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
