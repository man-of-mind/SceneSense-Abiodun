"""Single-host Run-4B 300-frame runner over the proven Phase-6 lifecycle.

``phase6_live_runner_v2.run_one_cell`` is reused unchanged (cold-host check,
admitted edge image, OAI radio, UE T-tracer, CARLA, the UE-tunnel capture,
the bounded child, full teardown and the post-run cold check).  This module
binds the Run-4B child, the Run-4B controller lineage and GT-free integration
gates; the full Phase-6 offline preflight still runs first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R

from . import ack_wire_v1 as AW

EXECUTE_TOKEN = "SPLITFUSION_RUN4B_SINGLE_HOST_300_V1_EXECUTE"
CHILD_MODULE = "rl_agent.splitfusion_run4b_single_host_v1.child_v1"
CLAIM_SCOPE = "SINGLE_HOST_RUN4B_300_FRAME_CHARACTERIZATION__NOT_POLICY_QUALIFICATION"
TRANSMITTED_BUDGET = 300


class Run4BRunnerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Run4BRunnerError(message)


def payload_digests_in_pcap(path: Path) -> dict[str, list[str]]:
    """SHA-256 of every operational-ACK and R4FB UDP payload in a classic pcap."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_quality_feedback_probe_v1 import packet_evidence as PE

    data = Path(path).read_bytes()
    formats = {b"\xd4\xc3\xb2\xa1": "<", b"\xa1\xb2\xc3\xd4": ">",
               b"\x4d\x3c\xb2\xa1": "<", b"\xa1\xb2\x3c\x4d": ">"}
    require(len(data) >= 24 and data[:4] in formats, "unsupported or truncated pcap")
    endian = formats[data[:4]]
    linktype = struct.unpack(endian + "IHHIIII", data[:24])[6]
    cursor = 24
    found: dict[str, list[str]] = {"ack": [], "r4fb": []}
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
        if AW.is_ack(payload):
            found["ack"].append(hashlib.sha256(payload).hexdigest())
        elif payload[:4] == W.MAGIC_FB:
            found["r4fb"].append(hashlib.sha256(payload).hexdigest())
    return found


def evaluate_run4b(*, ue: Mapping[str, Any], edge: Mapping[str, Any],
                   map_identity_rows: Sequence[Mapping[str, Any]],
                   tunnel: Mapping[str, Sequence[str]], cleanup_ok: bool,
                   transmitted_budget: int = TRANSMITTED_BUDGET) -> dict[str, Any]:
    """Integration gates only.  Timeouts are measured outcomes, never gated."""
    frames = list(ue.get("frames") or ())
    sent = list(ue.get("transmitted_identities") or ())
    sent_identities = {json.dumps(f.get("run4_identity"), sort_keys=True) for f in sent}
    counters = dict(ue.get("counters") or {})
    per_decision: dict[tuple, list] = {}
    for frame in frames:
        per_decision.setdefault((frame["ticket_seq"], frame.get("session_uuid", "")),
                                []).append(frame)
    reward_counts = [sum(1 for f in group if f["reward_requested"])
                     for group in per_decision.values()]
    edge_counters = dict(edge.get("counters") or {})
    evaluator = dict(edge.get("evaluator") or {})
    resolutions = list(ue.get("resolutions") or ())
    ack_rows = [r for r in ue.get("feedback_rows") or ()
                if r.get("kind") == "OPERATIONAL_ACK"]
    map_ok = bool(sent_identities) and all(
        json.dumps(json.loads(row["run4_identity_json"]), sort_keys=True)
        in sent_identities for row in map_identity_rows)
    gates = {
        "G0_TRANSMITTED_BUDGET": len(sent) == int(transmitted_budget),
        "G1_IDENTITY": (map_ok and edge_counters.get("sfd4_rejected", 0) == 0
                        and edge_counters.get("frame_identity_rejected", 0) == 0
                        and edge_counters.get("edge_processing_failed", 0) == 0),
        "G2_HOLD_TICKET": (bool(per_decision)
                           and all(len(g) >= 2 for g in per_decision.values())
                           and all(c == 1 for c in reward_counts)),
        "G3_EXACT_TERMINALS": (counters.get("policy_decisions", -1) == len(resolutions)
                               and ue.get("unresolved_tickets_at_close") == 0),
        "G4_NO_ACTOR_AFTER_REFUSAL": (counters.get("actor_calls")
                                      == counters.get("policy_decisions")),
        "G5_ACK_OVER_DOWNLINK": (
            bool(ack_rows)
            and Counter(r.get("ack_sha256") for r in edge.get("evaluations") or ()
                        if r.get("ack_sha256"))
            == Counter(tunnel.get("ack") or ()) == Counter(r["sha256"] for r in ack_rows)),
        "G6_GT_FREE_LIVE_PATH": (not tunnel.get("r4fb")
                                 and evaluator.get("live_quality_evaluations", 0) == 0
                                 and all(r.get("q_perc") is None
                                         for r in ue.get("feedback_rows") or ())),
        "G7_NO_INFRASTRUCTURE_FAULT": ue.get("faulted") is None,
        "G8_RESTORE_COLD": bool(cleanup_ok),
    }
    terminals = Counter(r.get("terminal") for r in resolutions)
    return {
        "gates": gates,
        "verdict": "INTEGRATION_COMPLETE" if all(gates.values()) else "FAILED",
        "claim_scope": CLAIM_SCOPE, "policy_performance_claim": False,
        "reported_not_gated": {
            "transmitted_frames": len(sent),
            "policy_decisions": counters.get("policy_decisions"),
            "policy_holds": counters.get("policy_holds"),
            "fallbacks": counters.get("fallbacks"),
            "terminals": dict(terminals),
            "ack_rows": Counter(r.get("class") for r in ack_rows),
            "coverage": ue.get("coverage")}}


def evaluate_attempt(attempt: Path, *, cleanup_ok: bool) -> dict[str, Any]:
    import csv

    evidence = Path(attempt) / "run4_phase6"
    ue = json.loads((evidence / "PHASE6_UE_EVIDENCE.json").read_text(encoding="utf-8"))
    edge = json.loads((evidence / "edge_report.json").read_text(encoding="utf-8"))
    with (Path(attempt) / "direct_edge_map" / "run4_map_identity.csv").open(
            newline="") as handle:
        rows = list(csv.DictReader(handle))
    tunnel = payload_digests_in_pcap(Path(attempt) / "quality_ack_oaitun_ue1.pcap")
    return evaluate_run4b(ue=ue, edge=edge, map_identity_rows=rows, tunnel=tunnel,
                          cleanup_ok=cleanup_ok)


def write_result_summary(attempt: Path, evaluation: Mapping[str, Any]) -> Path:
    path = Path(attempt) / "RUN4B_RESULT_SUMMARY.md"
    lines = [f"# Single-host Run-4B 300-frame run: {evaluation['verdict']}", "",
             f"Claim scope: `{evaluation['claim_scope']}`. This is not a "
             "policy-performance result.", "", "| Gate | Pass |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in evaluation["gates"].items()]
    lines += ["", "```", json.dumps(evaluation["reported_not_gated"], indent=1,
                                    sort_keys=True, default=str), "```", ""]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def run4b_preflight(config_path: Path, *, output_root: Path | None,
                    transmitted_budget: int) -> tuple[dict, list, dict]:
    """The full Phase-6 preflight plus the Run-4B bindings (no CUDA, no I/O)."""
    import importlib.util

    import torch

    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        frozen_actor_v2 as FA,
        live_state_v2 as LS,
        phase6_decision_engine_v2 as E,
        phase6_edge_runtime_v2 as E6,
        phase6_live_child_v2 as C,
        phase6_ue_runtime_v2 as U,
    )
    from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4B

    from . import ue_v1 as UE

    require(int(transmitted_budget) == TRANSMITTED_BUDGET,
            "the Run-4B run is exactly 300 transmitted frames")
    config, cells, report = R.offline_preflight(
        config_path, output_root=output_root, transmitted_budget=transmitted_budget)
    actor = UE.load_run4b_actor()
    genesis = UE.Run4BFeatureVectorV1(C4B.build_features(
        C4B.ObservationV1(camera_si=117.0, radar_p40=0.5, prior_ul_mcs=20,
                          pre_action_rlc_backlog_bytes=0),
        C4B.OperationalPriorV1.genesis(), UE.SCALING))
    decision = actor.act(genesis)
    sample = AW.decode(AW.encode(AW.OperationalAckV1(
        session_uuid="00000000-0000-4000-8000-000000000000",
        controller_lineage_sha256=UE.controller_lineage_sha256(), decision_seq=0,
        ticket_seq=0, frame_id=1, tensor_seq=0, capture_timestamp_ns=1, mode_id=0,
        q_e4=0, keep_count=1, execution_bundle_sha256="0" * 64, anchor_action_id=None,
        reward_requested=True, tail_output_sha256="0" * 64, record_count=0,
        reassembly_complete_wall_s=0.0, compute_start_wall_s=0.0,
        tail_ready_wall_ns=1, ack_send_wall_ns=1)))
    for name, owner, attribute in (
            ("actor seam", FA, "load_registered_actor"),
            ("state seam", LS, "build_live_state"),
            ("engine seam", E, "Run4DecisionEngineV2"),
            ("runtime seam", U, "build_run4_runtime_class"),
            ("edge module seam", C, "RUN4_EDGE_MODULE"),
            ("edge processor seam", E6, "Run4EdgeProcessorV2"),
            ("edge evaluator seam", E6, "Run4EvaluatorV2")):
        require(hasattr(owner, attribute), f"Phase-6 {name} is absent")
    for module in (CHILD_MODULE, "rl_agent.splitfusion_run4b_single_host_v1.edge_runtime_v1"):
        require(importlib.util.find_spec(module) is not None, f"{module} is absent")
    # The base report names the registered Run-4 actor/lineage; keep them under
    # explicit names so they cannot be read as the Run-4B identities.
    report["phase6_registered_run4_actor_boundary_sha256"] = report.pop(
        "actor_boundary_sha256")
    report["phase6_run4_controller_lineage_sha256"] = report.pop(
        "controller_lineage_sha256")
    report.update({
        "schema": "scenesense.splitfusion.run4b_single_host.preflight.v1",
        "child_module": CHILD_MODULE, "claim_scope": CLAIM_SCOPE,
        "run4b_actor": {"actor_state_dict_sha256": UE.ACTOR_STATE_DICT_SHA256,
                        "actor_tree_sha256": actor.boundary_sha256,
                        "seed": actor.manifest["seed"], "update": actor.manifest["update"],
                        "feature_schema_sha256": C4B.FEATURE_SCHEMA_SHA256,
                        "genesis_fixture_action": [decision.mode_id, decision.q_e4]},
        "run4b_scaling_sha256": UE.SCALING.sha256,
        "run4b_controller_lineage_sha256": UE.controller_lineage_sha256(),
        "operational_ack": {"schema": AW.SCHEMA, "semantics": AW.SEMANTICS,
                            "round_trip": sample.frame_id == 1},
        "live_qperc_computed": False,
        "cuda_initialized": torch.cuda.is_initialized()})
    require(not report["cuda_initialized"], "preflight initialized CUDA")
    return config, cells, report


def bind() -> None:
    """Bind the Run-4B child/lineage/gates into the unchanged lifecycle."""
    from . import ue_v1 as UE

    R.CHILD_MODULE = CHILD_MODULE
    R.controller_lineage_sha256 = UE.controller_lineage_sha256
    R.evaluate_attempt = evaluate_attempt
    R.write_result_summary = write_result_summary


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", default="")
    parser.add_argument("--config", type=Path, default=R.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--safety-timeout-s", type=float, default=120.0)
    parser.add_argument("--child-timeout-s", type=float, default=1200.0)
    parser.add_argument("--carla-port", type=int, default=2000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_root = args.output_root.resolve()
    config, cells, preflight = run4b_preflight(
        args.config.resolve(strict=True), output_root=output_root,
        transmitted_budget=TRANSMITTED_BUDGET)
    if args.preflight:
        print(json.dumps(preflight, sort_keys=True, indent=2, default=str))
        print("SPLITFUSION_RUN4B_SINGLE_HOST_PREFLIGHT_PASS")
        return 0
    require(args.execute == EXECUTE_TOKEN, f"live launch requires --execute {EXECUTE_TOKEN}")
    require(len(cells) == 1, "exactly one registered carrier cell is required")
    bind()
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "RUN4B_PREFLIGHT.json").write_text(
        json.dumps(preflight, sort_keys=True, indent=1, default=str), encoding="utf-8")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = R.run_one_cell(base_config=config, registered=cells[0],
                            output_root=output_root, run_id=run_id,
                            transmitted_budget=TRANSMITTED_BUDGET,
                            safety_timeout_s=float(args.safety_timeout_s),
                            carla_port=int(args.carla_port),
                            child_timeout_s=float(args.child_timeout_s))
    print(json.dumps({"status": report["status"], "claim_scope": report.get("claim_scope"),
                      "run4b": report.get("phase6"), "error": report.get("error")},
                     sort_keys=True, indent=2, default=str))
    return 0 if report["status"] == "INTEGRATION_COMPLETE" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
