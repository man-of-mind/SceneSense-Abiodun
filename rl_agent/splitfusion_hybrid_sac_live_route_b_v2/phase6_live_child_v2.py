#!/usr/bin/env python3
"""Phase 6 bounded live child: the Run-4 counterpart of ``live_cell_child``.

The lifecycle parent owns OAI, the T-tracer relay/recorder and CARLA.  This
child owns the direct map process, the edge container, the target-SNR actuator
and the CARLA client, exactly like the qualified quality-feedback child, and
runs the unchanged ``ue_route_b_split_cell_adapter_v1.run_route_b``.

Seams installed (process-local; no legacy file is edited):

* ``adapter_direct_v1.install_direct_seams`` (direct map, direct edge, ledger);
* edge module -> ``phase6_edge_runtime_v2``; map server -> ``phase6_map_server_v2``;
* ``LivePilotCellRuntime`` -> the Run-4 UE runtime with Run-4 ledgers;
* collector -> exact-tick quality collector + Run-4 sensor facts + the
  qualified exact-transmission budget;
* the edge config is seeded next to the qualified edge state.

Before the route starts the child proves that the UE feedback endpoint is
distinct from the direct-map endpoint and that the edge container routes the
UE address through the UPF (the OAI downlink), not the host bridge.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

UPF_ADDRESS = "192.168.70.134"   # receiver_container/entrypoint.sh, oai-cn5g compose
RUN4_EDGE_MODULE = "-m rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2"
RUN4_MAP_SERVER = "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_map_server_v2.py"


class ChildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChildError(message)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True, indent=1,
                                    default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


ROUTE_DETAIL_NAME = "route_detail.json"
POPULATION_GLOB = "ue_route_b_metrics_*population_events.jsonl"


def record_route_detail(result: dict, detail: Any, artifacts_dir: Path, *,
                        since_unix_s: float, tmp_root: Path = Path("/tmp")) -> dict[str, Any]:
    """Keep the complete ``run_route_b`` detail and route artifacts (create-only).

    Called immediately after ``run_route_b`` returns, so a ``collector=None``
    failure still retains the original route error, return code, density
    status, summary retention/parse information and outcome. Population-event
    files created by this route are copied without changing route behaviour.
    """
    safe = json.loads(json.dumps(dict(detail or {}), sort_keys=True, default=str))
    result["route_detail"] = safe
    path = Path(artifacts_dir) / ROUTE_DETAIL_NAME
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(safe, sort_keys=True, indent=1) + "\n")
    copied = []
    for source in sorted(Path(tmp_root).glob(POPULATION_GLOB)):
        try:
            if source.is_file() and source.stat().st_mtime >= float(since_unix_s):
                target = Path(artifacts_dir) / source.name
                with source.open("rb") as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer)
                copied.append(source.name)
        except OSError:
            continue
    result["route_population_artifacts"] = copied
    return safe


def edge_config(campaign: Mapping[str, Any], cell: Mapping[str, Any],
                evidence_leaf: str) -> dict[str, Any]:
    evidence = Path("/work/torch_cache") / evidence_leaf
    return {
        "schema": "scenesense.run4_live_v2.phase6_edge_config.v1",
        "run_id": str(campaign["campaign_id"]), "cell_id": str(cell["cell_id"]),
        "evidence_dir": str(evidence),
        "report_path": str(evidence / "run4_phase6_edge_report.json"),
        "match_distance_m": float(campaign["measurement_contract"]["match_distance_m"]),
        "gt_timeout_s": 2.0, "queue_depth": 64,
    }


def verify_feedback_path(*, map_host: str, map_port: int, ue_host: str, ue_port: int,
                         run: Any = subprocess.run) -> dict[str, Any]:
    """Distinct endpoints and a UE route through the UPF (OAI downlink)."""
    require((str(map_host), int(map_port)) != (str(ue_host), int(ue_port)),
            "reward-feedback endpoint equals the direct-map endpoint")
    require(str(map_host) != str(ue_host), "direct map must not be the UE address")
    route = run(["sudo", "-n", "docker", "exec", "oai-perception-rx", "ip", "route",
                 "get", str(ue_host)], stdin=subprocess.DEVNULL, capture_output=True,
                text=True, timeout=10.0, check=False)
    text = (route.stdout or "").strip()
    require(route.returncode == 0 and f"via {UPF_ADDRESS}" in text,
            f"edge->UE feedback would not traverse the OAI downlink: {text!r}")
    return {"map_endpoint": f"{map_host}:{map_port}", "ue_endpoint": f"{ue_host}:{ue_port}",
            "edge_route_to_ue": text, "via_upf": True}


def install_run4_seams(campaign: Mapping[str, Any], *, cell: Mapping[str, Any],
                       attempt_dir: Path, artifacts_dir: Path, bindings: Any,
                       transmitted_budget: int, safety_timeout_s: float) -> dict[str, Any]:
    from rl_agent import ue_map_install_feedback_v1 as pinned_feedback
    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
    from rl_agent.splitfusion_quality_feedback_probe_v1 import adapter_quality_v1 as Q
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_cell_child as LC

    from . import phase6_ue_runtime_v2 as U
    from . import run4_map_protocol_v2 as MP
    from . import run4_ue_ledger_v2 as UL

    D._ENDPOINT["cell_id"] = str(cell["cell_id"])
    D._ENDPOINT["attempt_dir"] = str(Path(attempt_dir).resolve())
    D._ENDPOINT["config_relpath"] = str(campaign["runtime"]["direct_edge_config_relpath"])
    endpoint = D.install_direct_seams(campaign)
    D.DIRECT_EDGE_MODULE = RUN4_EDGE_MODULE
    D.DIRECT_MAP_SERVER = RUN4_MAP_SERVER
    direct_stop = pinned.stop_tail
    original_seed = pinned.seed_cell_edge_state
    state: dict[str, Any] = {}

    def seed(campaign_: Mapping[str, Any], edge_scratch: Path) -> None:
        original_seed(campaign_, edge_scratch)
        (Path(edge_scratch) / "run4_phase6_edge_config.json").write_text(
            json.dumps(edge_config(campaign_, cell, pinned.EDGE_EVIDENCE_LEAF),
                       sort_keys=True, indent=1) + "\n", encoding="utf-8")
        state["edge_scratch"] = str(edge_scratch)

    def stop_tail() -> bool:
        stopped = True
        if pinned.tail_running():
            stopped = subprocess.run(
                ["sudo", "docker", "stop", "--time", "30", "oai-perception-rx"],
                cwd=str(pinned.ROOT), check=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=40.0).returncode == 0
        copied = False
        scratch = state.get("edge_scratch")
        if scratch:
            source = (Path(scratch) / pinned.EDGE_EVIDENCE_LEAF
                      / "run4_phase6_edge_report.json")
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not source.is_file():
                time.sleep(0.05)
            if source.is_file():
                destination = Path(attempt_dir) / "run4_phase6" / "edge_report.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                copied = True
        return bool(stopped and copied and direct_stop())

    runtime_class = U.build_run4_runtime_class(bindings)

    def runtime_factory(*, campaign: Mapping[str, Any], cell: Mapping[str, Any],
                        attempt_dir: Path, map_host: str, map_port: int,
                        evidence_dir: Path) -> Any:
        del map_host, map_port
        ledger = UL.Run4TerminalLedgerV2(
            output_csv=Path(attempt_dir) / "map_feedback.csv",
            experiment_id=str(campaign["campaign_id"]), cell_id=str(cell["cell_id"]))
        compat = UL.Run4CompatLedgerV2(ledger)
        D._PENDING_LEDGER["ledger"] = compat
        return runtime_class(campaign=campaign, cell=cell, attempt_dir=Path(attempt_dir),
                             evidence_dir=Path(evidence_dir),
                             ue_control_port=int(campaign["runtime"]["ue_control_port"]),
                             ledger=compat)

    pinned.seed_cell_edge_state = seed
    pinned.stop_tail = stop_tail
    pinned.LivePilotCellRuntime = runtime_factory
    pinned.SceneSnapshotSource = Q.QualitySceneSnapshotSource
    pinned.PassiveSplitCollector = LC.build_bounded_collector_class(
        U.build_run4_collector_class(Q.QualityPassiveSplitCollector),
        transmitted_budget=int(transmitted_budget),
        safety_timeout_s=float(safety_timeout_s), artifacts_dir=Path(artifacts_dir))
    pinned_feedback.InstallFeedbackLedger = D.direct_ledger_factory
    return {"endpoint": endpoint, "state": state}


def run(args: argparse.Namespace) -> int:  # pragma: no cover - live
    import yaml

    from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D

    from . import phase6_ue_runtime_v2 as U

    campaign = json.loads(Path(args.campaign_json).read_text(encoding="utf-8"))
    cell = json.loads(Path(args.cell_json).read_text(encoding="utf-8"))
    attempt_dir = Path(args.attempt_dir).resolve(strict=True)
    temporary_dir = Path(args.temporary_dir).resolve(strict=True)
    artifacts_dir = Path(args.artifacts_dir).resolve(strict=True)
    bound = json.loads(Path(args.bindings_json).read_text(encoding="utf-8"))
    bindings = U.Run4LiveBindingsV2(
        controller_lineage_sha256=str(bound["controller_lineage_sha256"]),
        tracer_dir=Path(bound["tracer_dir"]), t_messages=Path(bound["t_messages"]),
        ue_relay_port=int(bound["ue_relay_port"]),
        evidence_dir=attempt_dir / "run4_phase6")
    row = pinned.action_row(campaign, int(cell["action_id"]))
    require(int(cell["action_id"]) == 71 and str(row["profile_id"])
            == "split_ae32_uint4_q9800", "carrier cell must be the registered fallback anchor")
    result: dict[str, Any] = {"schema": "scenesense.run4_live_v2.phase6_child.v1",
                              "cell_id": str(cell["cell_id"]), "error": "", "cleanup": {},
                              "started_at_unix_s": time.time()}
    seams = install_run4_seams(campaign, cell=cell, attempt_dir=attempt_dir,
                               artifacts_dir=artifacts_dir, bindings=bindings,
                               transmitted_budget=int(args.transmitted_budget),
                               safety_timeout_s=float(args.safety_timeout_s))
    map_process = edge_scratch = target_process = collector = None
    target_output = target_stop = Path()
    return_code = 2
    try:
        map_process = pinned.start_map_process(
            campaign, temporary_dir=temporary_dir, action_id=str(cell["action_id"]),
            carla_host="127.0.0.1", carla_port=int(args.carla_port),
            api_port=int(args.map_api_port), udp_port=int(args.spatial_map_port),
            feedback_port=int(args.feedback_port))
        edge_scratch = pinned.start_live_edge(campaign, cell, temporary_dir)
        runtime = campaign["runtime"]
        result["feedback_path"] = verify_feedback_path(
            map_host=str(D._ENDPOINT["endpoint"].host),
            map_port=int(runtime["direct_map_ingest_port"]),
            ue_host=str(runtime["ue_bind_host"]), ue_port=int(runtime["ue_control_port"]))
        isolated = temporary_dir / "run4_phase6_campaign.yaml"
        target_start = temporary_dir / "run4_phase6_target_start"
        campaign["_target_start_file"] = str(target_start)
        isolated.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
        target_process, target_output, target_stop = pinned.start_target_snr(
            campaign, campaign_path=isolated, profile_id=str(cell["network_profile_id"]),
            temporary_dir=temporary_dir, start_file=target_start)
        time.sleep(1.0)
        require(target_process.poll() is None, "target-SNR actuator exited during startup")
        _ok, detail, collector = pinned.run_route_b(
            campaign=campaign, cell=cell, row=row,
            binding={"dispatcher": "run4_phase6_sfd4"}, attempt_dir=attempt_dir,
            carla_host="127.0.0.1", carla_port=int(args.carla_port),
            map_api_port=int(args.map_api_port), feedback_port=int(args.feedback_port),
            edge_evidence_dir=edge_scratch / pinned.EDGE_EVIDENCE_LEAF,
            maximum_loop_sim_s=float(args.safety_timeout_s))
        # Addendum 8: persist the complete route detail before any assertion.
        record_route_detail(result, detail, artifacts_dir,
                            since_unix_s=float(result["started_at_unix_s"]))
        require(collector is not None, "route never entered the collector")
        result["collector"] = collector.probe_summary()
        stop_reason = str(collector.probe_stop_reason)
        # Addendum 6: a stop at a closed k_min decision cycle is registered;
        # it can only end at or below the transmitted budget.
        if stop_reason == "DECISION_CYCLE_BOUNDARY":
            require(1 <= int(collector.sent) <= int(args.transmitted_budget),
                    f"cycle-boundary stop outside the budget ({int(collector.sent)})")
        else:
            require(int(collector.sent) == int(args.transmitted_budget),
                    f"only {int(collector.sent)}/{int(args.transmitted_budget)} frames sent")
            require(stop_reason == "TRANSMITTED_BUDGET_REACHED",
                    f"unexpected stop reason {collector.probe_stop_reason!r}")
        result["stop_reason"] = stop_reason
        require(bool(collector.cleanup_ok) and not collector.failures,
                f"collector failures: {collector.failures[:4]}")
        return_code = 0
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = result["cleanup"]
        if target_process is not None:
            try:
                cleanup["target_snr_restored"] = bool(pinned.stop_target_snr(
                    target_process, target_output, target_stop,
                    attempt_dir / "radio_trace.csv"))
            except BaseException as exc:
                cleanup["target_snr_error"] = f"{type(exc).__name__}: {exc}"
        cleanup["map_stopped"] = bool(pinned.stop_process(map_process))
        cleanup["edge_stopped"] = bool(pinned.stop_live_edge(edge_scratch))
        if return_code == 0 and not all(cleanup.get(k) for k in (
                "target_snr_restored", "map_stopped", "edge_stopped")):
            result["error"] = "child-owned resource cleanup failed"
            return_code = 2
        result["return_code"] = return_code
        result["finished_at_unix_s"] = time.time()
        _atomic_json(artifacts_dir / "child_result.json", result)
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--campaign-json", "--cell-json", "--attempt-dir", "--temporary-dir",
                 "--artifacts-dir", "--bindings-json"):
        parser.add_argument(name, required=True)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--map-api-port", type=int, default=35001)
    parser.add_argument("--spatial-map-port", type=int, default=39310)
    parser.add_argument("--feedback-port", type=int, default=39401)
    parser.add_argument("--transmitted-budget", type=int, default=300)
    parser.add_argument("--safety-timeout-s", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
