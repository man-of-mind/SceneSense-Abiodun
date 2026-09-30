#!/usr/bin/env python3
"""Bounded Phase-6 edge-startup qualification (setup-repair addendum 3).

Starts only what the edge container needs: the OAI core compose, which
creates ``oai-cn5g-public-net`` (``--pull never``), and the edge through the
Phase-6 no-build launch path. There is no gNB, UE, CARLA, map service or
scientific frame. The edge must reach the existing ready record with:
- the exact admitted image ID;
- exact mounts;
- ``tail_device == cuda:0``;
- the pinned FCOS checkpoint;
- full preload.

Everything is then torn down, and the host must be cold again.

    env -u PYTHONPATH python3 -m \
      rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_startup_qualification_v2 \
      --output-root <new dir> --execute PHASE6_EDGE_STARTUP_QUALIFICATION_V2_EXECUTE
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
EXECUTE_TOKEN = "PHASE6_EDGE_STARTUP_QUALIFICATION_V2_EXECUTE"
CN_DIR = ROOT / "OAI" / "oai-cn5g"


def _sudo_compose(args: Sequence[str], cwd: Path, log: Path, timeout: float) -> int:
    with log.open("ab") as stream:
        return subprocess.run(["sudo", "-n", "docker", "compose", *args], cwd=str(cwd),
                              stdin=subprocess.DEVNULL, stdout=stream,
                              stderr=subprocess.STDOUT, check=False,
                              timeout=timeout).returncode


def _running_containers() -> list[str]:
    out = subprocess.run(["sudo", "-n", "docker", "ps", "--format", "{{.Names}}"],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True,
                         check=False, timeout=30.0)
    return [line for line in out.stdout.splitlines() if line.strip()]


def _network_present(name: str) -> bool:
    return subprocess.run(["sudo", "-n", "docker", "network", "inspect", name],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False,
                          timeout=30.0).returncode == 0


def _tunnel_present() -> bool:
    return subprocess.run(["ip", "-br", "link", "show", "oaitun_ue1"],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode == 0


def build_cache_total() -> str:
    out = subprocess.run(["sudo", "-n", "docker", "buildx", "du"], stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, check=False, timeout=60.0)
    lines = [line for line in out.stdout.splitlines() if line.startswith("Total:")]
    return lines[-1].split(":", 1)[1].strip() if lines else ""


def cold_snapshot(supervisor: Any, campaign: Any) -> dict[str, Any]:
    return {"application": supervisor._require_phase15_application_cold(campaign),
            "running_containers": _running_containers(),
            "core_network_present": _network_present("oai-cn5g-public-net"),
            "ue_tunnel_present": _tunnel_present()}


def is_cold(snapshot: dict[str, Any]) -> bool:
    return (not snapshot["running_containers"] and not snapshot["core_network_present"]
            and not snapshot["ue_tunnel_present"])


def run(output_root: Path) -> dict[str, Any]:  # pragma: no cover - live
    from rl_agent import ue_288_campaign_supervisor as supervisor
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP

    from . import phase6_edge_launch_v2 as EL
    from . import phase6_live_child_nobuild_v2 as NB
    from . import phase6_live_runner_v2 as RUN
    from . import phase6_ue_runtime_v2 as U

    output_root.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config, cells, preflight = RUN.offline_preflight(
        RUN.DEFAULT_CONFIG, output_root=None, transmitted_budget=300)
    registered = cells[0]
    campaign = LP._probe_campaign(config, run_id=run_id)
    campaign["campaign_id"] = f"splitfusion_run4_phase6_edge_startup_v2/{run_id}"
    cell = supervisor.Cell(
        cell_id=f"run4p6edge_{run_id}_{registered.cell_id}",
        action_index=registered.action_index, action_id=registered.action_id,
        profile_id=registered.profile_id, model_family=registered.model_family,
        network_profile_id=registered.network_profile_id, trace_id=registered.trace_id,
        seed=registered.seed)
    cell_dict = supervisor.cell_to_dict(cell)
    service = Path(tempfile.mkdtemp(prefix=f"run4_phase6_edge_startup_{run_id}_"))
    artifacts = output_root / "artifacts"
    artifacts.mkdir()
    log = output_root / "infrastructure.log"
    report: dict[str, Any] = {
        "schema": "scenesense.run4_live_v2.phase6_edge_startup_qualification.v1",
        "addendum": "PHASE6_SETUP_REPAIR_ADDENDUM_3_NO_BUILD_EDGE", "run_id": run_id,
        "status": "FAILED", "error": "", "checks": {}, "cleanup": {},
        "scientific_frames_sent": 0, "carla_started": False, "radio_started": False,
        "preflight": {k: preflight.get(k) for k in (
            "readiness_manifest_sha256", "actor_boundary_sha256", "setup_addendum",
            "edge_image", "child_module", "external_processes_started")},
        "started_at_unix_s": time.time()}
    core_started = False
    edge_scratch = None
    try:
        report["cold_before"] = cold_snapshot(supervisor, campaign)
        EL.require(is_cold(report["cold_before"]), "host is not cold before startup")
        report["edge_image_prelaunch"] = EL.resolve_admitted_image()
        report["build_cache_total_before"] = build_cache_total()
        core_started = True
        rc = _sudo_compose(["up", "-d", "--pull", "never", "--force-recreate",
                            "--remove-orphans"], CN_DIR, log, 240.0)
        EL.require(rc == 0, f"core compose up failed rc={rc}")
        deadline = time.monotonic() + 60.0
        while not _network_present(EL.NETWORK) and time.monotonic() < deadline:
            time.sleep(0.5)
        EL.require(_network_present(EL.NETWORK), "core network did not appear")
        bindings = U.Run4LiveBindingsV2(
            controller_lineage_sha256=RUN.controller_lineage_sha256(),
            tracer_dir=Path(RUN.telemetry_bindings()["tracer_dir"]),
            t_messages=Path(RUN.telemetry_bindings()["t_messages"]),
            ue_relay_port=int(RUN.telemetry_bindings()["ue_relay_port"]),
            evidence_dir=output_root / "run4_phase6")
        NB.install_run4_seams_nobuild(
            campaign, cell=cell_dict, attempt_dir=output_root, artifacts_dir=artifacts,
            bindings=bindings, transmitted_budget=300, safety_timeout_s=120.0)
        edge_scratch = pinned.start_live_edge(campaign, cell_dict, service)
        ready = json.loads((Path(edge_scratch) / "ready.json").read_text(encoding="utf-8"))
        (output_root / "edge_ready_record.json").write_text(
            json.dumps(ready, sort_keys=True, indent=1) + "\n", encoding="utf-8")
        launch = json.loads((output_root / NB.EVIDENCE_RELPATH).read_text(encoding="utf-8"))
        record = campaign["deployment"]["fcos_constructor_weights"]
        container = EL.inspect_container(
            state_root=Path(edge_scratch),
            fcos_source=(ROOT / str(record["path"])).resolve(strict=True),
            fcos_sha256=str(record["sha256"]))
        mounts = pinned.inspect_live_edge_mounts(Path(edge_scratch))
        report["container_after_ready"] = container
        report["pinned_mounts"] = mounts
        report["ready_record"] = ready
        checks = report["checks"]
        checks["prelaunch_image_is_admitted"] = (
            report["edge_image_prelaunch"]["id"] == EL.ADMITTED_IMAGE_ID)
        checks["launch_verdict_admitted"] = launch.get("verdict") == "ADMITTED_IMAGE_LAUNCHED"
        checks["post_create_image_is_admitted"] = (
            launch["post_create_container"]["image"] == EL.ADMITTED_IMAGE_ID)
        checks["after_ready_image_is_admitted"] = container["image"] == EL.ADMITTED_IMAGE_ID
        checks["no_build_no_pull_command"] = (
            not EL.forbidden_operations(launch["compose_command"])
            and list(launch["compose_command"][-6:]) == list(EL.UP_ARGUMENTS))
        checks["legacy_launcher_not_invoked"] = launch.get("legacy_launcher_invoked") is False
        checks["mounts_exact"] = (container["mounts"]["repository"]["rw"] is False
                                  and container["mounts"]["state"]["rw"] is True
                                  and container["mounts"]["fcos"]["rw"] is False)
        checks["fcos_sha256_pinned"] = (container["mounts"]["fcos"]["sha256"]
                                        == str(record["sha256"]))
        checks["tail_device_cuda0"] = ready.get("tail_device") == "cuda:0"
        checks["ready_schema"] = ready.get("schema") == "splitfusion_direct_live_edge_ready.v1"
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D

        from . import phase6_live_child_v2 as C

        expected_evidence = C.edge_config(campaign, cell_dict,
                                          pinned.EDGE_EVIDENCE_LEAF)["evidence_dir"]
        runtime = campaign["runtime"]
        checks["ready_architecture"] = ready.get("architecture") == "DIRECT_EDGE_TO_MAP_V1"
        checks["ready_dense_label_map_off_radio"] = ready.get("dense_label_map_on_radio") is False
        checks["ready_object_records_off_radio"] = ready.get("object_records_on_radio") is False
        checks["ready_evidence_dir_exact"] = (
            ready.get("evaluation_evidence_dir") == expected_evidence
            == str(Path("/work/torch_cache") / pinned.EDGE_EVIDENCE_LEAF))
        checks["ready_map_endpoint"] = (
            str(ready.get("direct_map_host")) == str(D._ENDPOINT["endpoint"].host)
            and int(ready.get("direct_map_port")) == int(runtime["direct_map_ingest_port"]))
        checks["ready_endpoints_distinct"] = (
            (ready.get("direct_map_host"), ready.get("direct_map_port"))
            != (ready.get("ue_control_host"), ready.get("ue_control_port")))
        checks["ready_quality_spec"] = (ready.get("quality_spec_sha256")
                                        == preflight.get("quality_spec_sha256"))
        report["expected_evidence_dir"] = expected_evidence
        checks["no_scientific_frame"] = True
        report["status"] = "PASSED" if all(checks.values()) else "FAILED"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = report["cleanup"]
        try:
            cleanup["edge_stopped"] = bool(pinned.stop_live_edge(edge_scratch))
        except BaseException as exc:
            cleanup["edge_stop_failure"] = f"{type(exc).__name__}: {exc}"
        try:
            cleanup["application"] = supervisor._stop_phase15_application(campaign)
        except BaseException as exc:
            cleanup["application_failure"] = f"{type(exc).__name__}: {exc}"
        if core_started:
            cleanup["core_down_returncode"] = _sudo_compose(
                ["down", "--remove-orphans"], CN_DIR, log, 240.0)
        shutil.rmtree(service, ignore_errors=True)
        try:
            cleanup["edge_image_after_teardown"] = EL.resolve_admitted_image()
        except BaseException as exc:
            cleanup["edge_image_after_teardown_failure"] = f"{type(exc).__name__}: {exc}"
        cleanup["build_cache_total_after"] = build_cache_total()
        try:
            cleanup["cold_after"] = cold_snapshot(supervisor, campaign)
            cleanup["cold"] = is_cold(cleanup["cold_after"])
        except BaseException as exc:
            cleanup["cold"] = False
            cleanup["cold_after_failure"] = f"{type(exc).__name__}: {exc}"
        if report["status"] == "PASSED" and not cleanup.get("cold"):
            report["status"] = "FAILED"
            report["error"] = report["error"] or "host not cold after teardown"
        if report["status"] == "PASSED" and (
                (cleanup.get("edge_image_after_teardown") or {}).get("id")
                != EL.ADMITTED_IMAGE_ID):
            report["status"] = "FAILED"
            report["error"] = "admitted image identity not verified after teardown"
        if report["status"] == "PASSED" and (
                cleanup.get("build_cache_total_after") != report.get("build_cache_total_before")):
            report["status"] = "FAILED"
            report["error"] = "build cache changed during startup (a build occurred)"
        report["finished_at_unix_s"] = time.time()
        LP._atomic_create_json(output_root / "EDGE_STARTUP_QUALIFICATION.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.execute != EXECUTE_TOKEN:
        print(f"refused: requires --execute {EXECUTE_TOKEN}", file=sys.stderr)
        return 2
    report = run(args.output_root.resolve())
    print(json.dumps({k: report.get(k) for k in ("status", "error", "checks", "cleanup")},
                     sort_keys=True, indent=1, default=str))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
