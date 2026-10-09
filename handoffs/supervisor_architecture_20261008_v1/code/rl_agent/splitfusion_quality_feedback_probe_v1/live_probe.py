#!/usr/bin/env python3
"""Bounded live exact-quality and compact-ACK timing probe.

Default matrix: action 50 under FAVORABLE_STABLE and ADVERSE_STABLE, 300
successfully transmitted live CARLA frames per cell.  Every cell gets a fresh
qualified CN5G/gNB/UE lifecycle and a fresh Epic CARLA server.  The full
Route-B loop is intentionally not completed.

This harness does not authorize a new 288-cell campaign.  It is create-only,
fail-closed, and requires an explicit execution token.  ``--preflight`` is
offline and starts no CARLA, Docker, OAI, CUDA or model process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent import ue_288_campaign_supervisor as supervisor
from rl_agent.splitfusion_quality_feedback_probe_v1.packet_evidence import (
    QualityAckCapture,
    render_packet_evidence,
)
from rl_agent.splitfusion_quality_feedback_probe_v1.analyze_live_probe import (
    analyze_attempt,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
)
EXECUTE_TOKEN = "SPLITFUSION_QUALITY_FEEDBACK_LIVE_PROBE_V1_EXECUTE"
SUCCESS_TERMINAL = "SPLITFUSION_QUALITY_FEEDBACK_LIVE_PROBE_V1_COMPLETE"
FAILURE_TERMINAL = "SPLITFUSION_QUALITY_FEEDBACK_LIVE_PROBE_V1_FAILED"
CHILD_MODULE = "rl_agent.splitfusion_quality_feedback_probe_v1.live_cell_child"
DEFAULT_PROFILES = ("FAVORABLE_STABLE", "ADVERSE_STABLE")


class LiveProbeError(RuntimeError):
    """The bounded live-probe contract is not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveProbeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_binding() -> dict[str, Any]:
    package = ROOT / "rl_agent/splitfusion_quality_feedback_probe_v1"
    sources = {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(package.glob("*.py"))
    }
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(head.returncode == 0, "cannot bind the launch git HEAD")
    require(status.returncode == 0, "cannot record launch worktree state")
    return {
        "git_head": head.stdout.strip(),
        "git_status_short": status.stdout.splitlines(),
        "source_sha256": sources,
    }


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"create-only output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=1, default=str) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise LiveProbeError(f"invalid action list: {value!r}") from exc
    require(parsed, "at least one action is required")
    require(len(parsed) == len(set(parsed)), "action list contains duplicates")
    return parsed


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    require(parsed, "at least one network profile is required")
    require(len(parsed) == len(set(parsed)), "network profile list contains duplicates")
    return parsed


def _registered_cells(
    config: Mapping[str, Any],
    *,
    action_ids: Sequence[int],
    profile_ids: Sequence[str],
) -> list[supervisor.Cell]:
    """Select only cells already pinned by the supplied campaign config."""

    enumerated = supervisor.enumerate_cells(config)
    by_key = {
        (int(cell.action_id), str(cell.network_profile_id)): cell
        for cell in enumerated
    }
    missing = [
        (int(action_id), str(profile_id))
        for profile_id in profile_ids
        for action_id in action_ids
        if (int(action_id), str(profile_id)) not in by_key
    ]
    require(
        not missing,
        "requested action/profile cells are not pinned by the config: "
        + ", ".join(f"a{action}/{profile}" for action, profile in missing),
    )
    return [
        by_key[(int(action_id), str(profile_id))]
        for profile_id in profile_ids
        for action_id in action_ids
    ]


def _probe_campaign(
    base_config: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    """Build the single-action probe campaign without qualification semantics.

    ``_qualification`` is a reserved field in the pinned collector: its mere
    presence selects the historical 20-frame, four-action qualification
    contract.  A probe cell needs no multi-action edge allow-list because the
    direct edge already defaults to the cell's selected action.  Remove the
    reserved field defensively so an inherited config can never activate that
    unrelated collector mode.
    """

    campaign = deepcopy(dict(base_config))
    campaign.pop("_qualification", None)
    campaign["campaign_id"] = f"splitfusion_quality_feedback_probe_v1/{run_id}"
    return campaign


def offline_preflight(
    config_path: Path,
    *,
    action_ids: Sequence[int],
    profile_ids: Sequence[str],
    transmitted_budget: int,
    safety_timeout_s: float,
    output_root: Path | None,
) -> tuple[dict[str, Any], list[supervisor.Cell], dict[str, Any]]:
    require(config_path.is_file(), f"config is missing: {config_path}")
    require(int(transmitted_budget) > 0, "transmitted budget must be positive")
    require(float(safety_timeout_s) > 0.0, "safety timeout must be positive")
    if output_root is not None:
        require(not output_root.exists(), f"output root already exists: {output_root}")

    config = supervisor.load_json(config_path.resolve(strict=True))
    require(
        str(config.get("runtime", {}).get("architecture"))
        == "DIRECT_EDGE_TO_MAP_V1",
        "probe requires the direct edge-to-map architecture",
    )
    require(
        config.get("runtime", {}).get("object_records_on_radio") is False,
        "direct architecture unexpectedly puts object records on the radio",
    )
    require(
        str(
            (config.get("direct_edge_map") or {})
            .get("cpu_reservation", {})
            .get("map_render", "")
        ).casefold()
        == "off",
        "measurement config must keep the nonessential map renderer off",
    )

    # These are the same static bindings enforced before a normal live cell.
    supervisor.verify_file_hashes(config)
    supervisor.verify_radio_baseline(config)
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_route_contract(config)
    cells = _registered_cells(
        config, action_ids=action_ids, profile_ids=profile_ids
    )
    require(
        len(cells) == len(action_ids) * len(profile_ids),
        "selected cell Cartesian product is incomplete",
    )
    require(
        Path(ROOT / "rl_agent/splitfusion_quality_feedback_probe_v1/adapter_quality_v1.py").is_file(),
        "quality adapter is not present",
    )
    require(
        Path(ROOT / "rl_agent/splitfusion_quality_feedback_probe_v1/live_cell_child.py").is_file(),
        "bounded live child is not present",
    )

    report = {
        "schema": "scenesense.quality_feedback_probe_preflight.v1",
        "status": "PASS",
        "external_processes_started": 0,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "transmitted_budget_per_cell": int(transmitted_budget),
        "safety_timeout_s": float(safety_timeout_s),
        "full_route_b_completion_claimed": False,
        "cells": [supervisor.cell_to_dict(cell) for cell in cells],
        "fresh_oai_per_cell": True,
        "fresh_carla_per_cell": True,
        "evaluation_queues_discarded": False,
        "implementation_binding": _implementation_binding(),
        "action_extension_rule": (
            "an action is accepted only when its profile is selected and pinned "
            "by the supplied config"
        ),
    }
    return config, cells, report


def _bounded_log_record(path: Path, *, tail_bytes: int = 8192) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.name, "present": False}
    data = path.read_bytes()
    return {
        "path": path.name,
        "present": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "tail": data[-tail_bytes:].decode("utf-8", errors="replace"),
    }


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_child(child: subprocess.Popen[Any] | None, pgid: int | None) -> None:
    """Stop the child's whole process group, including orphaned descendants."""

    if child is None or pgid is None:
        return
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15.0
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5.0
        while _process_group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        child.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        pass


def _quality_files(attempt_dir: Path) -> list[dict[str, Any]]:
    """Inventory durable quality evidence without assuming its final names."""

    records: list[dict[str, Any]] = []
    for path in sorted(attempt_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(attempt_dir)
        text = str(relative).casefold()
        if "quality" not in text and "direct_edge" not in text:
            continue
        records.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def run_one_cell(
    *,
    base_config: Mapping[str, Any],
    registered: supervisor.Cell,
    output_root: Path,
    run_id: str,
    ordinal: int,
    transmitted_budget: int,
    safety_timeout_s: float,
    carla_port: int,
    child_timeout_s: float,
) -> dict[str, Any]:
    """Run one fresh OAI/CARLA cell and prove teardown before returning."""

    unique_cell_id = f"qfb_{run_id}_{registered.cell_id}"
    cell = supervisor.Cell(
        cell_id=unique_cell_id,
        action_index=registered.action_index,
        action_id=registered.action_id,
        profile_id=registered.profile_id,
        model_family=registered.model_family,
        network_profile_id=registered.network_profile_id,
        trace_id=registered.trace_id,
        seed=registered.seed,
    )
    attempt_dir = output_root / "cells" / registered.cell_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = attempt_dir / "probe_artifacts"
    artifacts_dir.mkdir(parents=False, exist_ok=False)
    service_dir = Path(
        tempfile.mkdtemp(prefix=f"splitfusion_quality_probe_{unique_cell_id}_")
    )
    lifecycle = supervisor.import_lifecycle_helper(base_config)
    campaign = _probe_campaign(base_config, run_id=run_id)
    campaign_json = service_dir / "campaign.json"
    cell_json = service_dir / "cell.json"
    campaign_json.write_text(
        json.dumps(campaign, sort_keys=True, indent=1) + "\n", encoding="utf-8"
    )
    cell_json.write_text(
        json.dumps(supervisor.cell_to_dict(cell), sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )

    report: dict[str, Any] = {
        "schema": "scenesense.quality_feedback_probe_live_cell.v1",
        "cell_id": unique_cell_id,
        "registered_cell_id": registered.cell_id,
        "action_id": cell.action_id,
        "profile_id": cell.profile_id,
        "network_profile_id": cell.network_profile_id,
        "ordinal": int(ordinal),
        "started_at_unix_s": time.time(),
        "status": "FAILED",
        "full_route_b_completion_claimed": False,
        "cleanup": {},
        "error": "",
    }
    radio_namespace: Path | None = None
    radio_state: Path | None = None
    attached: Mapping[str, Any] | None = None
    server: Any = None
    pgid: int | None = None
    child: subprocess.Popen[Any] | None = None
    child_pgid: int | None = None
    packet_capture: QualityAckCapture | None = None
    child_log = attempt_dir / "child_stdout_stderr.log"
    try:
        report["cold_before"] = supervisor._require_phase15_application_cold(
            campaign
        )
        radio_namespace, radio_state, attached = supervisor._start_live_radio(
            campaign, cell, 1, service_dir
        )
        report["radio_attachment"] = {
            "status": attached.get("status"),
            "clean_noise_preflight": attached.get("clean_noise_preflight"),
        }
        server, pgid = lifecycle.start_carla(
            carla_port, service_dir / "carla_server.log"
        )
        version = lifecycle.wait_for_rpc(carla_port, 180.0)
        require(version is not None, "fresh Epic CARLA did not become RPC-ready")
        report["carla"] = {
            "rpc_port": int(carla_port),
            "server_version": str(version),
        }

        runtime = dict(campaign["runtime"])
        packet_capture = QualityAckCapture(
            attempt_dir,
            interface="oaitun_ue1",
            ue_host=str(runtime["ue_bind_host"]),
            ue_port=int(runtime["ue_control_port"]),
        )
        packet_capture.start()

        argv = [
            sys.executable,
            "-m",
            CHILD_MODULE,
            "--campaign-json",
            str(campaign_json),
            "--cell-json",
            str(cell_json),
            "--attempt-dir",
            str(attempt_dir),
            "--temporary-dir",
            str(service_dir),
            "--artifacts-dir",
            str(artifacts_dir),
            "--carla-port",
            str(carla_port),
            "--transmitted-budget",
            str(transmitted_budget),
            "--safety-timeout-s",
            str(safety_timeout_s),
        ]
        with child_log.open("xb") as stream:
            child = subprocess.Popen(
                argv,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=lifecycle.child_env(),
                start_new_session=True,
            )
            child_pgid = os.getpgid(child.pid)
            try:
                child_returncode = int(child.wait(timeout=float(child_timeout_s)))
            except subprocess.TimeoutExpired as exc:
                _stop_child(child, child_pgid)
                raise LiveProbeError(
                    f"bounded child exceeded {child_timeout_s:.0f}s"
                ) from exc

        result_path = artifacts_dir / "child_result.json"
        require(
            result_path.is_file(),
            f"child left no durable result (rc={child_returncode})",
        )
        child_result = json.loads(result_path.read_text(encoding="utf-8"))
        report["child_returncode"] = child_returncode
        report["child"] = child_result
        collector = dict(child_result.get("collector") or {})
        require(child_returncode == 0, f"child failed: {child_result.get('error')}")
        require(
            int(collector.get("transmitted_frames", -1))
            == int(transmitted_budget),
            "child did not stop at the exact transmitted-frame budget",
        )
        require(
            collector.get("stop_reason") == "TRANSMITTED_BUDGET_REACHED",
            "child stopped for a reason other than the exact frame budget",
        )
        require(
            collector.get("evaluation_queues_discarded") is False
            and collector.get("quality_and_evaluation_drain_complete") is True,
            "quality/evaluation work was discarded or did not drain",
        )
        require(
            bool(collector.get("collector_cleanup_ok")),
            "collector cleanup/drain gate failed",
        )
        packet_capture.stop()
        report["quality_ack_packet_evidence"] = render_packet_evidence(
            attempt_dir,
            ue_host=str(runtime["ue_bind_host"]),
            ue_port=int(runtime["ue_control_port"]),
        )
        report["quality_feedback_analysis"] = analyze_attempt(attempt_dir)
        report["quality_evidence"] = _quality_files(attempt_dir)
        require(
            any(
                "quality" in str(item["path"]).casefold()
                for item in report["quality_evidence"]
            ),
            "no durable quality evidence was copied before edge teardown",
        )
        report["status"] = "PASSED"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "operator interrupt"
        raise
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if packet_capture is not None:
            try:
                packet_capture.stop()
            except BaseException as exc:
                report["cleanup"]["packet_capture_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        _stop_child(child, child_pgid)
        cleanup = report["cleanup"]
        try:
            cleanup["application"] = supervisor._stop_phase15_application(campaign)
        except BaseException as exc:
            cleanup["application_error"] = f"{type(exc).__name__}: {exc}"
        if server is not None and pgid is not None:
            try:
                cleanup["carla"] = lifecycle.stop_carla(server, pgid, carla_port)
            except BaseException as exc:
                cleanup["carla_error"] = f"{type(exc).__name__}: {exc}"
        if radio_namespace is not None and radio_state is not None:
            try:
                cleanup["radio"] = supervisor._stop_live_radio(
                    campaign, radio_namespace, radio_state, attached
                )
                cleanup["radio_shutdown_verified"] = True
            except BaseException as exc:
                cleanup["radio_error"] = f"{type(exc).__name__}: {exc}"
                cleanup["radio_shutdown_verified"] = False
        report["service_logs"] = [
            _bounded_log_record(service_dir / name)
            for name in ("oai_launcher.log", "carla_server.log", "edge_launcher.log")
        ]
        shutil.rmtree(service_dir, ignore_errors=True)
        cleanup["service_scratch_removed"] = not service_dir.exists()
        try:
            cleanup["cold_after"] = supervisor._require_phase15_application_cold(
                campaign
            )
        except BaseException as exc:
            cleanup["cold_after_error"] = f"{type(exc).__name__}: {exc}"

        cleanup_ok = bool(
            not cleanup.get("application_error")
            and not cleanup.get("carla_error")
            and not cleanup.get("packet_capture_error")
            and cleanup.get("radio_shutdown_verified")
            and cleanup.get("service_scratch_removed")
            and "cold_after" in cleanup
            and (
                server is None
                or bool((cleanup.get("carla") or {}).get("shutdown_verified"))
            )
        )
        cleanup["all_gates_passed"] = cleanup_ok
        if not cleanup_ok and report["status"] == "PASSED":
            report["status"] = "FAILED"
            report["error"] = "parent lifecycle cleanup/cold-postflight failed"
        report["finished_at_unix_s"] = time.time()
        report["wall_seconds"] = (
            report["finished_at_unix_s"] - report["started_at_unix_s"]
        )
        _atomic_create_json(attempt_dir / "CELL_RESULT.json", report)
        terminal_name = {
            "PASSED": "PASSED.json",
            "INTERRUPTED": "INTERRUPTED.json",
        }.get(str(report["status"]), "FAILED.json")
        _atomic_create_json(
            attempt_dir / terminal_name,
            {
                "status": report["status"],
                "cell_id": unique_cell_id,
                "cell_result_sha256": _sha256(attempt_dir / "CELL_RESULT.json"),
                "finished_at_unix_s": report["finished_at_unix_s"],
            },
        )
    return report


def _artifact_manifest(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    actions = _parse_csv_ints(args.actions)
    profiles = _parse_csv_strings(args.network_profiles)
    config, cells, preflight = offline_preflight(
        config_path,
        action_ids=actions,
        profile_ids=profiles,
        transmitted_budget=int(args.transmitted_budget),
        safety_timeout_s=float(args.safety_timeout_s),
        output_root=output_root,
    )
    if args.preflight:
        print(json.dumps(preflight, sort_keys=True, indent=2))
        print("SPLITFUSION_QUALITY_FEEDBACK_LIVE_PROBE_V1_PREFLIGHT_PASS")
        return 0
    require(
        args.execute == EXECUTE_TOKEN,
        f"live launch requires --execute {EXECUTE_TOKEN}",
    )

    output_root.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _atomic_create_json(output_root / "preflight.json", preflight)
    reports: list[dict[str, Any]] = []
    status = "PASSED"
    error = ""
    try:
        for ordinal, cell in enumerate(cells):
            report = run_one_cell(
                base_config=config,
                registered=cell,
                output_root=output_root,
                run_id=run_id,
                ordinal=ordinal,
                transmitted_budget=int(args.transmitted_budget),
                safety_timeout_s=float(args.safety_timeout_s),
                carla_port=int(args.carla_port),
                child_timeout_s=float(args.child_timeout_s),
            )
            reports.append(report)
            if report["status"] != "PASSED":
                status = "FAILED"
                error = f"cell failed: {report['registered_cell_id']}"
                break
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        error = "operator interrupt"
    except BaseException as exc:
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"

    summary = {
        "schema": "scenesense.quality_feedback_probe_live_run.v1",
        "status": status,
        "run_id": run_id,
        "error": error,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "requested_cells": len(cells),
        "completed_cells": len(reports),
        "passed_cells": sum(row["status"] == "PASSED" for row in reports),
        "transmitted_budget_per_cell": int(args.transmitted_budget),
        "full_route_b_completion_claimed": False,
        "cells": [
            {
                "registered_cell_id": row["registered_cell_id"],
                "cell_id": row["cell_id"],
                "status": row["status"],
                "action_id": row["action_id"],
                "network_profile_id": row["network_profile_id"],
                "cell_result": f"cells/{row['registered_cell_id']}/CELL_RESULT.json",
            }
            for row in reports
        ],
        "finished_at_unix_s": time.time(),
    }
    _atomic_create_json(output_root / "RUN_SUMMARY.json", summary)
    manifest = {
        "schema": "scenesense.quality_feedback_probe_manifest.v1",
        "files": _artifact_manifest(output_root),
    }
    _atomic_create_json(output_root / "manifest.json", manifest)
    terminal = SUCCESS_TERMINAL if status == "PASSED" else FAILURE_TERMINAL
    _atomic_create_json(
        output_root / terminal,
        {
            "status": terminal,
            "run_summary_sha256": _sha256(output_root / "RUN_SUMMARY.json"),
            "manifest_sha256": _sha256(output_root / "manifest.json"),
        },
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    print(terminal)
    return 0 if status == "PASSED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", default="")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--actions", default="50")
    parser.add_argument(
        "--network-profiles", default=",".join(DEFAULT_PROFILES)
    )
    parser.add_argument("--transmitted-budget", type=int, default=300)
    parser.add_argument("--safety-timeout-s", type=float, default=90.0)
    parser.add_argument("--child-timeout-s", type=float, default=900.0)
    parser.add_argument("--carla-port", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return run(args)
    except (LiveProbeError, supervisor.CampaignError) as exc:
        print(f"quality feedback probe refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
