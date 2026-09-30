#!/usr/bin/env python3
"""Phase 6 lifecycle parent for the 300-frame Run-4 live qualification.

NOT EXECUTED by this package.  ``--preflight`` is offline (no CARLA, OAI,
Docker, CUDA or network).  The live path requires an explicit execute token
and a separate authorization.

Lifecycle (the qualified quality-probe parent, plus the telemetry relay):

1. cold check -> qualified OAI launcher attach (``_start_live_radio``);
2. UE T-tracer: OAI ``multi`` relays 2023 -> 2123 and the durable ``record``
   client writes ``ue.raw`` (the Phase-2C-qualified topology);
3. fresh Epic CARLA; packet capture on ``oaitun_ue1`` for the UE control port
   (proves R4FB reward feedback arrives over the OAI downlink);
4. the bounded Phase-6 child (map, edge, target-SNR, CARLA client, route);
5. teardown, always and in reverse order: child, capture, tracer, application,
   CARLA, radio; then the application-cold postflight.

The gates are evaluated by :func:`evaluate_phase6` from durable evidence only.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
EXECUTE_TOKEN = "SPLITFUSION_RUN4_PHASE6_LIVE_QUALIFICATION_V2_EXECUTE"
# Addendum 3: the unchanged child plus the no-build, image-bound edge launch seam.
CHILD_MODULE = "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_live_child_nobuild_v2"
DEFAULT_CONFIG = ROOT / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
TELEMETRY_CONFIG = ROOT / "rl_agent/ue_production_queue_capture_v1/config_v1.json"
FALLBACK_ACTION = 71
PROFILE = "FAVORABLE_STABLE"
UE_EVENTS = ("NRUE_MAC_DCI_GRANT", "NRUE_MAC_RLC_BUFFER_STATUS", "NR_PDCP_TX_SDU",
             "NR_RLC_TX_SDU", "NR_RLC_TX_DEQUEUE")


class Phase6RunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase6RunnerError(message)


def controller_lineage_sha256() -> str:
    from . import frozen_actor_v2 as FA

    manifest = FA.load_json(FA.TRACKED_BINDING_PATH)
    return hashlib.sha256(json.dumps({
        "actor_boundary_sha256": manifest["actor"]["boundary_sha256"],
        "actor_manifest_sha256": manifest["manifest_sha256"],
        "controller": "RewardHoldControllerV2", "engine": "Run4DecisionEngineV2",
        "fallback": "mode11_q9800_a71"}, sort_keys=True).encode()).hexdigest()


def telemetry_bindings() -> dict[str, Any]:
    from rl_agent.ue_production_queue_capture_v1 import config as CFG

    runtime = CFG.effective_runtime_config(TELEMETRY_CONFIG)
    return {"tracer_dir": str(ROOT / runtime["paths"]["t_tracer_dir"]),
            "t_messages": str(ROOT / runtime["paths"]["t_messages"]),
            "ue_port": int(runtime["telemetry"]["ue_port"]),
            "ue_relay_port": int(runtime["telemetry"]["ue_relay_port"])}


ADDENDUM_PATH = Path(__file__).resolve().with_name("phase6_prospective_addendum_2.json")
SETUP_ADDENDUM_PATH = Path(__file__).resolve().with_name("phase6_setup_repair_addendum_3.json")


def verify_setup_addendum(path: Path = SETUP_ADDENDUM_PATH) -> dict[str, Any]:
    """Addendum 3 must bind exactly the admitted image ID the launcher enforces."""
    from . import phase6_edge_launch_v2 as EL

    addendum = json.loads(Path(path).read_text(encoding="utf-8"))
    require(addendum.get("id") == "PHASE6_SETUP_REPAIR_ADDENDUM_3_NO_BUILD_EDGE",
            "foreign Phase-6 setup addendum")
    require(addendum.get("admitted_image_id") == EL.ADMITTED_IMAGE_ID,
            "setup addendum image ID differs from the launcher")
    require(addendum.get("scientific_protocol_changed") is False,
            "setup addendum must not change the scientific protocol")
    return {"id": addendum["id"], "admitted_image_id": EL.ADMITTED_IMAGE_ID,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def verify_addendum(path: Path = ADDENDUM_PATH) -> dict[str, Any]:
    """Addendum 2 (option c) must bind the unchanged plan and the decision memo."""
    from . import phase6_result_reporting_v2 as REP

    addendum = json.loads(Path(path).read_text(encoding="utf-8"))
    require(addendum.get("id") == REP.ADDENDUM_ID, "foreign Phase-6 addendum")
    amends = addendum["amends"]
    for key in ("plan", "memo"):
        actual = hashlib.sha256((ROOT / amends[key]).read_bytes()).hexdigest()
        require(actual == amends[f"{key}_sha256"], f"addendum {key} digest differs")
    return {"id": addendum["id"], "claim_scope": REP.CLAIM_SCOPE,
            "policy_performance_claim": False,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def offline_preflight(config_path: Path, *, output_root: Path | None,
                      transmitted_budget: int) -> tuple[dict, list, dict]:
    """Everything checkable without starting a process."""
    import torch

    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

    from . import frozen_actor_v2 as FA
    from . import phase2_telemetry_qualification as Q2
    from . import phase6_decision_engine_v2 as E
    from . import readiness_v2 as RD
    from . import run4_live_wire_v2 as W
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as dec

    config, cells, report = LP.offline_preflight(
        config_path, action_ids=(FALLBACK_ACTION,), profile_ids=(PROFILE,),
        transmitted_budget=int(transmitted_budget), safety_timeout_s=120.0,
        output_root=output_root)
    readiness = RD.verify_manifest()
    actor = FA.load_registered_actor()
    contract = dec.load_dynamic_execution_contract()
    fallback = contract.resolve_q_e4(E.FALLBACK["mode_id"], E.FALLBACK["q_e4"])
    require((fallback.action_id, fallback.profile_id)
            == (E.FALLBACK["anchor_action_id"], E.FALLBACK["profile_id"]),
            "fixed fallback does not reconcile with the catalog")
    spec = W.load_run4_quality_spec(ROOT)
    addendum = verify_addendum()
    setup_addendum = verify_setup_addendum()
    from . import phase6_edge_launch_v2 as EL

    edge_image = EL.resolve_admitted_image()     # read-only docker image inspect
    report.update({
        "schema": "scenesense.run4_live_v2.phase6_preflight.v1",
        "readiness_manifest_sha256": readiness["manifest_sha256"],
        "actor_boundary_sha256": actor.boundary_sha256,
        "fallback": dict(E.FALLBACK),
        "fallback_execution_bundle_sha256": fallback.execution_bundle_sha256,
        "quality_spec_sha256": spec.canonical_sha256(),
        "radio_binding": RB.verify("before_preflight", ROOT)["verified"],
        "emitter_pins": Q2.verify_emitter_pins(ROOT),
        "controller_lineage_sha256": controller_lineage_sha256(),
        "telemetry": telemetry_bindings(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "live_command_executed": False,
        "addendum": addendum,
        "setup_addendum": setup_addendum,
        "edge_image": edge_image,
        "child_module": CHILD_MODULE,
    })
    require(not report["cuda_initialized"], "preflight initialized CUDA")
    return config, cells, report


# ---------------------------------------------------------------------------
# Gates (pure; evaluated from durable evidence)
# ---------------------------------------------------------------------------


def r4fb_digests_in_pcap(path: Path) -> list[str]:
    """SHA-256 of every R4FB UDP payload in a classic pcap (UE tunnel capture)."""
    import struct

    from rl_agent.splitfusion_quality_feedback_probe_v1 import packet_evidence as PE

    from . import run4_live_wire_v2 as W

    data = Path(path).read_bytes()
    formats = {b"\xd4\xc3\xb2\xa1": "<", b"\xa1\xb2\xc3\xd4": ">",
               b"\x4d\x3c\xb2\xa1": "<", b"\xa1\xb2\x3c\x4d": ">"}
    require(len(data) >= 24 and data[:4] in formats, "unsupported or truncated pcap")
    endian = formats[data[:4]]
    linktype = struct.unpack(endian + "IHHIIII", data[:24])[6]
    cursor, digests = 24, []
    while cursor + 16 <= len(data):
        captured = struct.unpack(endian + "IIII", data[cursor:cursor + 16])[2]
        packet = data[cursor + 16:cursor + 16 + captured]
        cursor += 16 + captured
        ip = PE._ipv4_payload(packet, int(linktype))
        if ip is None or len(ip) < 28 or ip[9] != 17:
            continue
        ihl = (ip[0] & 0x0F) * 4
        length = struct.unpack("!H", ip[ihl + 4:ihl + 6])[0]
        payload = ip[ihl + 8:ihl + length]
        if payload[:4] == W.MAGIC_FB:
            digests.append(hashlib.sha256(payload).hexdigest())
    return digests


def evaluate_phase6(*, ue: Mapping[str, Any], edge: Mapping[str, Any],
                    map_identity_rows: Sequence[Mapping[str, Any]],
                    feedback_packets_on_ue_tunnel: Sequence[str],
                    cleanup_ok: bool) -> dict[str, Any]:
    frames = list(ue.get("frames") or ())
    sent_identities = {json.dumps(f.get("run4_identity"), sort_keys=True)
                       for f in ue.get("transmitted_identities", ())}
    counters = dict(ue.get("counters") or {})
    per_decision: dict[tuple, list] = {}
    for frame in frames:
        per_decision.setdefault((frame["ticket_seq"], frame.get("session_uuid", "")),
                                []).append(frame)
    reward_counts = [sum(1 for f in group if f["reward_requested"])
                     for group in per_decision.values()]
    map_ok = all(json.dumps(json.loads(row["run4_identity_json"]), sort_keys=True)
                 in sent_identities for row in map_identity_rows) if sent_identities else False
    edge_counters = dict(edge.get("counters") or {})
    resolutions = list(ue.get("resolutions") or ())
    gates = {
        "P0_POLICY_COVERAGE": (ue.get("coverage") or {}).get("verdict") == "PASS",
        "P1_IDENTITY": (map_ok and edge_counters.get("sfd4_rejected", 0) == 0
                        and edge_counters.get("edge_processing_failed", 0) == 0),
        "P3_HOLD_TICKET": (bool(per_decision)
                           and all(len(g) >= 2 for g in per_decision.values())
                           and all(c == 1 for c in reward_counts)),
        "P4_ACCOUNTING": (counters.get("policy_decisions", -1)
                          - len(resolutions) in (0, 1)),
        "P5_NO_ACTOR_AFTER_REFUSAL": (counters.get("actor_calls")
                                      == counters.get("policy_decisions")),
        "P7_FEEDBACK_OVER_DOWNLINK": _three_way(
            [row.get("feedback_sha256") for row in edge.get("evaluations") or ()],
            list(feedback_packets_on_ue_tunnel),
            [row.get("sha256") for row in ue.get("feedback_rows") or ()]),
        "P8_NO_INFRASTRUCTURE_FAULT": ue.get("faulted") is None,
        "P6_RESTORE_COLD": bool(cleanup_ok),
    }
    verdict = ("PASSED" if all(gates.values())
               else "INCONCLUSIVE_OR_FAILED" if not gates["P0_POLICY_COVERAGE"]
               else "FAILED")
    from . import phase6_result_reporting_v2 as REP

    # Addendum 2 (option c): reporting only; gates and verdict are unchanged.
    return {"gates": gates, "verdict": verdict,
            "claim_scope": REP.CLAIM_SCOPE,
            "policy_performance_claim": False,
            "verdict_statement": REP.PASS_STATEMENT,
            "result_summary": REP.result_summary(ue),
            "reported_not_gated": {
                "fallback_fraction": (ue.get("coverage") or {}).get("fallback_fraction"),
                "session_rollovers": counters.get("session_rollovers"),
                "terminals": _histogram(r.get("terminal") for r in resolutions),
                "reward_mean": _mean(r.get("reward") for r in resolutions
                                     if r.get("reward") is not None),
                "reward_mean_label": REP.CONDITIONAL_LABEL}}


def _three_way(edge: Sequence[Any], tunnel: Sequence[Any], ue: Sequence[Any]) -> bool:
    from collections import Counter

    return bool(edge) and Counter(edge) == Counter(tunnel) == Counter(ue)


def evaluate_attempt(attempt: Path, *, cleanup_ok: bool) -> dict[str, Any]:
    """Load the durable Phase-6 evidence of one attempt and apply the gates."""
    import csv as _csv

    evidence = attempt / "run4_phase6"
    ue = json.loads((evidence / "PHASE6_UE_EVIDENCE.json").read_text(encoding="utf-8"))
    edge = json.loads((evidence / "edge_report.json").read_text(encoding="utf-8"))
    with (attempt / "direct_edge_map" / "run4_map_identity.csv").open(newline="") as handle:
        rows = list(_csv.DictReader(handle))
    tunnel = r4fb_digests_in_pcap(attempt / "quality_ack_oaitun_ue1.pcap")
    return evaluate_phase6(ue=ue, edge=edge, map_identity_rows=rows,
                           feedback_packets_on_ue_tunnel=tunnel, cleanup_ok=cleanup_ok)


def write_result_summary(attempt: Path, evaluation: Mapping[str, Any]) -> Path:
    """Create-only human-readable summary led by the systems-integration scope."""
    from . import phase6_result_reporting_v2 as REP

    path = Path(attempt) / "PHASE6_RESULT_SUMMARY.md"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(REP.render_markdown(evaluation))
    return path


def _histogram(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def _mean(values) -> float | None:
    data = [float(v) for v in values]
    return sum(data) / len(data) if data else None


# ---------------------------------------------------------------------------
# Live lifecycle (not executed here)
# ---------------------------------------------------------------------------


def start_ue_tracer(bindings: Mapping[str, Any], raw_path: Path, log_dir: Path,
                    popen: Callable[..., Any] = subprocess.Popen) -> list[Any]:
    """The Phase-2C topology: multi relay + durable record client."""
    tracer = Path(bindings["tracer_dir"])
    msgs = str(bindings["t_messages"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    multi = popen([str(tracer / "multi"), "-d", msgs, "-ip", "127.0.0.1", "-p",
                   str(bindings["ue_port"]), "-lp", str(bindings["ue_relay_port"])],
                  stdout=(log_dir / "ue_relay.log").open("x"), stderr=subprocess.STDOUT,
                  start_new_session=True)
    time.sleep(1.0)
    argv = [str(tracer / "record"), "-d", msgs, "-o", str(raw_path), "-OFF"]
    for event in UE_EVENTS:
        argv += ["-on", event]
    argv += ["-ip", "127.0.0.1", "-p", str(bindings["ue_relay_port"])]
    record = popen(argv, stdout=(log_dir / "ue_record.log").open("x"),
                   stderr=subprocess.STDOUT, start_new_session=True)
    return [record, multi]


def stop_processes(processes: Sequence[Any], timeout_s: float = 5.0) -> bool:
    ok = True
    for process in processes:
        if process is None or process.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=timeout_s)
        except Exception:  # noqa: BLE001
            try:
                process.kill()
                process.wait(timeout=timeout_s)
            except Exception:  # noqa: BLE001
                ok = False
    return ok and all(p is None or p.poll() is not None for p in processes)


def run_one_cell(*, base_config: Mapping[str, Any], registered: Any, output_root: Path,
                 run_id: str, transmitted_budget: int, safety_timeout_s: float,
                 carla_port: int, child_timeout_s: float,
                 supervisor: Any = None, capture_class: Any = None,
                 image_resolver: Callable[[], dict] | None = None,
                 stop_after_decisions: int | None = None) -> dict[str, Any]:
    """One fresh OAI/CARLA cell; teardown is attempted for every resource."""
    from rl_agent import ue_288_campaign_supervisor as default_supervisor
    from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP
    from rl_agent.splitfusion_quality_feedback_probe_v1.packet_evidence import (
        QualityAckCapture,
    )

    supervisor = supervisor or default_supervisor
    capture_class = capture_class or QualityAckCapture
    cell = supervisor.Cell(
        cell_id=f"run4p6_{run_id}_{registered.cell_id}", action_index=registered.action_index,
        action_id=registered.action_id, profile_id=registered.profile_id,
        model_family=registered.model_family,
        network_profile_id=registered.network_profile_id,
        trace_id=registered.trace_id, seed=registered.seed)
    attempt = output_root / "cells" / registered.cell_id
    artifacts = attempt / "phase6_artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    service = Path(tempfile.mkdtemp(prefix=f"run4_phase6_{cell.cell_id}_"))
    lifecycle = supervisor.import_lifecycle_helper(base_config)
    campaign = LP._probe_campaign(base_config, run_id=run_id)
    campaign["campaign_id"] = f"splitfusion_run4_phase6_v2/{run_id}"
    files = {"campaign": service / "campaign.json", "cell": service / "cell.json",
             "bindings": service / "bindings.json"}
    files["campaign"].write_text(json.dumps(campaign, sort_keys=True), encoding="utf-8")
    files["cell"].write_text(json.dumps(supervisor.cell_to_dict(cell), sort_keys=True),
                             encoding="utf-8")
    tel = telemetry_bindings()
    files["bindings"].write_text(json.dumps({
        "controller_lineage_sha256": controller_lineage_sha256(),
        "tracer_dir": tel["tracer_dir"], "t_messages": tel["t_messages"],
        "ue_relay_port": tel["ue_relay_port"]}, sort_keys=True), encoding="utf-8")
    report: dict[str, Any] = {"schema": "scenesense.run4_live_v2.phase6_cell.v1",
                              "cell_id": cell.cell_id, "status": "FAILED", "cleanup": {},
                              "started_at_unix_s": time.time(), "error": ""}
    namespace = radio_state = attached = server = pgid = child = child_pgid = None
    capture = None
    tracer: list[Any] = []
    try:
        report["cold_before"] = supervisor._require_phase15_application_cold(campaign)
        # Addendum 3: refuse before any OAI/CARLA start if the image is missing/drifted.
        from . import phase6_edge_launch_v2 as EL

        resolve_image = image_resolver or EL.resolve_admitted_image
        report["edge_image_prelaunch"] = resolve_image()
        LP._atomic_create_json(attempt / "edge_image_prelaunch.json",
                               report["edge_image_prelaunch"])
        namespace, radio_state, attached = supervisor._start_live_radio(
            campaign, cell, 1, service)
        tracer = start_ue_tracer({**tel}, attempt / "ttracer" / "ue" / "ue.raw",
                                 attempt / "logs")
        server, pgid = lifecycle.start_carla(carla_port, service / "carla_server.log")
        require(lifecycle.wait_for_rpc(carla_port, 180.0) is not None, "CARLA not ready")
        runtime = dict(campaign["runtime"])
        capture = capture_class(attempt, interface="oaitun_ue1",
                                ue_host=str(runtime["ue_bind_host"]),
                                ue_port=int(runtime["ue_control_port"]))
        capture.start()
        argv = [sys.executable, "-m", CHILD_MODULE, "--campaign-json", str(files["campaign"]),
                "--cell-json", str(files["cell"]), "--attempt-dir", str(attempt),
                "--temporary-dir", str(service), "--artifacts-dir", str(artifacts),
                "--bindings-json", str(files["bindings"]), "--carla-port", str(carla_port),
                "--transmitted-budget", str(transmitted_budget),
                "--safety-timeout-s", str(safety_timeout_s)]
        if stop_after_decisions is not None:      # addendum 6: stop at a closed cycle
            argv += ["--stop-after-decisions", str(int(stop_after_decisions))]
        with (attempt / "child_stdout_stderr.log").open("xb") as stream:
            child = subprocess.Popen(argv, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     env=lifecycle.child_env(), start_new_session=True)
            child_pgid = os.getpgid(child.pid)
            try:
                rc = int(child.wait(timeout=float(child_timeout_s)))
            except subprocess.TimeoutExpired as exc:
                LP._stop_child(child, child_pgid)
                raise Phase6RunnerError("bounded child exceeded its timeout") from exc
        result = json.loads((artifacts / "child_result.json").read_text(encoding="utf-8"))
        report["child"] = result
        require(rc == 0, f"child failed: {result.get('error')}")
        report["status"] = "COLLECTED"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        raise
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = report["cleanup"]
        LP._stop_child(child, child_pgid)
        if capture is not None:
            try:
                capture.stop()
            except BaseException as exc:
                cleanup["capture_error"] = f"{type(exc).__name__}: {exc}"
        cleanup["tracer_stopped"] = stop_processes(tracer)
        try:
            cleanup["application"] = supervisor._stop_phase15_application(campaign)
        except BaseException as exc:
            cleanup["application_error"] = f"{type(exc).__name__}: {exc}"
        if server is not None and pgid is not None:
            try:
                cleanup["carla"] = lifecycle.stop_carla(server, pgid, carla_port)
            except BaseException as exc:
                cleanup["carla_error"] = f"{type(exc).__name__}: {exc}"
        if namespace is not None and radio_state is not None:
            try:
                cleanup["radio"] = supervisor._stop_live_radio(
                    campaign, namespace, radio_state, attached)
                cleanup["radio_shutdown_verified"] = True
            except BaseException as exc:
                cleanup["radio_error"] = f"{type(exc).__name__}: {exc}"
                cleanup["radio_shutdown_verified"] = False
        shutil.rmtree(service, ignore_errors=True)
        try:
            cleanup["cold_after"] = supervisor._require_phase15_application_cold(campaign)
        except BaseException as exc:
            cleanup["cold_after_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from . import phase6_edge_launch_v2 as EL

            cleanup["edge_image_post_run"] = (image_resolver or EL.resolve_admitted_image)()
        except BaseException as exc:
            cleanup["edge_image_post_run_failure"] = f"{type(exc).__name__}: {exc}"
        cleanup["all_gates_passed"] = bool(
            not any(k.endswith("_error") for k in cleanup)
            and cleanup.get("tracer_stopped")
            and (namespace is None or cleanup.get("radio_shutdown_verified"))
            and "cold_after" in cleanup
            and (server is None or bool((cleanup.get("carla") or {}).get("shutdown_verified"))))
        if report["status"] == "COLLECTED":
            try:
                verdict = evaluate_attempt(attempt, cleanup_ok=cleanup["all_gates_passed"])
                report["phase6"] = verdict
                report["status"] = verdict["verdict"]
                report["claim_scope"] = verdict["claim_scope"]
                write_result_summary(attempt, verdict)
            except BaseException as exc:
                report["status"] = "FAILED"
                report["error"] = f"gate evaluation: {type(exc).__name__}: {exc}"
        report["finished_at_unix_s"] = time.time()
        LP._atomic_create_json(attempt / "CELL_RESULT.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", default="")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transmitted-budget", type=int, default=300)
    parser.add_argument("--safety-timeout-s", type=float, default=120.0)
    parser.add_argument("--child-timeout-s", type=float, default=1200.0)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--stop-after-decisions", type=int, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_root = args.output_root.resolve()
    config, cells, preflight = offline_preflight(
        args.config.resolve(strict=True), output_root=output_root,
        transmitted_budget=int(args.transmitted_budget))
    if args.preflight:
        print(json.dumps(preflight, sort_keys=True, indent=2, default=str))
        print("SPLITFUSION_RUN4_PHASE6_PREFLIGHT_PASS")
        return 0
    require(args.execute == EXECUTE_TOKEN, f"live launch requires --execute {EXECUTE_TOKEN}")
    require(len(cells) == 1, "exactly one registered carrier cell is required")
    output_root.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = run_one_cell(base_config=config, registered=cells[0], output_root=output_root,
                          run_id=run_id, transmitted_budget=int(args.transmitted_budget),
                          safety_timeout_s=float(args.safety_timeout_s),
                          carla_port=int(args.carla_port),
                          child_timeout_s=float(args.child_timeout_s),
                          stop_after_decisions=args.stop_after_decisions)
    print(json.dumps({"status": report["status"], "claim_scope": report.get("claim_scope"),
                      "phase6": report.get("phase6")},
                     sort_keys=True, indent=2, default=str))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
