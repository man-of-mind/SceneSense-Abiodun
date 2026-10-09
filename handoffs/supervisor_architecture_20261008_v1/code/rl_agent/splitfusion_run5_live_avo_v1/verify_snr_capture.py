"""Read-only postflight: original Run-5 identity and causal live-SNR selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import actor_binding as A


def verify_capture(capture_root: Path) -> dict:
    capture = Path(capture_root)
    cells = list((capture / "cells").iterdir())
    if len(cells) != 1:
        raise ValueError("Expected exactly one captured cell")
    cell = cells[0]
    evidence = json.loads((cell / "run5_policy/SNR_EVIDENCE.json").read_text())
    ue = json.loads((cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_text())
    analysis = json.loads((capture / "PROFILER_ANALYSIS.json").read_text())
    selected = [row for row in evidence["decision_snr"] if row["admitted"]]
    decisions = [row for row in ue["decisions"] if row.get("kind") == "POLICY_DECISION"]
    snapshots = evidence["provider"]["received_snapshots"]
    decision_map = {int(row["frame_id"]): row for row in decisions}
    transmitted = [row["run4_identity"] for row in ue.get("transmitted_identities", [])
                   if row["run4_identity"].get("frame_kind") == "POLICY_DECISION"]
    transmitted_map = {int(row["frame_id"]): row for row in transmitted}
    selected_frames = [int(row["frame_id"]) for row in selected]
    checks = {
        "original_run5_actor": evidence["actor_tree_sha256"] == A.ACTOR_TREE_SHA256
                               and evidence["weights_sha256"] == A.WEIGHTS_SHA256,
        "22_features_recorded": evidence["state_dim"] == 22
             and analysis["policy_runtime"]["observed_state_dim"] == 22,
        "all_decisions_have_snr": len(selected) == len(decisions) and len(selected) > 0,
        "exact_actor_replay": analysis["semantics"]["actor_replay"]["passed"] is True,
        "controller_started": evidence["controller_started"] is True,
        "provider_bound_to_runtime": evidence["runtime_bound"] is True,
        "unique_exact_decision_population": (len(decision_map) == len(decisions)
            and len(transmitted_map) == len(transmitted)
            and len(set(selected_frames)) == len(selected_frames)
            and set(selected_frames) == set(decision_map) == set(transmitted_map)),
    }
    errors = []
    join_errors = []
    for row in selected:
        boundary, snr = row["boundary"], row["snr"]
        cutoff, opened = boundary["state_commit_timestamp_ns"], boundary["action_open_timestamp_ns"]
        fid = int(row["frame_id"])
        decision, wire = decision_map.get(fid, {}), transmitted_map.get(fid, {})
        ident = row["decision_identity"]
        join_ok = (bool(decision) and bool(wire)
            and boundary.get("identity") == ident
            and ident.get("session_uuid") == decision.get("decision_session_uuid") == wire.get("session_uuid")
            and ident.get("decision_seq") == wire.get("decision_seq")
            and decision.get("tensor_seq") == wire.get("tensor_seq")
            and decision.get("state_commit") == {"domain": boundary["clock_domain"], "ns": cutoff}
            and decision.get("action_open") == {"domain": boundary["clock_domain"], "ns": opened}
            and cutoff <= opened
            and len(decision.get("features", [])) == 22
            and decision["features"][21] == row["snr_scaled"]
            and snr.get("identity", {}).get("ue_id") == ident.get("ue_id")
            and snr.get("identity", {}).get("session_uuid") == ident.get("session_uuid"))
        if not join_ok:
            join_errors.append({"frame_id": fid, "reason": "SNR/UE_DECISION_IDENTITY_OR_BOUNDARY_MISMATCH"})
        candidates = [s for s in snapshots if s["receipt_raw_ns"] < cutoff and s["heartbeat_raw_ns"] < cutoff]
        latest = max(candidates, key=lambda s: s["sequence"], default=None)
        okay = (latest is not None and latest["valid"] is True and snr["valid"] is True
                and latest["controller_uuid"] == evidence["provider"]["controller_uuid"]
                and latest["command_id"] == snr["active_command_id"] == snr["heartbeat_command_id"]
                and latest["target_snr_db"] == snr["value_db"]
                and latest["effective_raw_ns"] == snr["effective_since_ns"] < cutoff
                and latest["heartbeat_raw_ns"] == snr["heartbeat_ns"] < cutoff
                and snr["clock_domain"] == boundary["clock_domain"] == "CLOCK_MONOTONIC_RAW"
                and 0 < opened - snr["heartbeat_ns"] <= A.LEASE_MAX_HEARTBEAT_AGE_NS
                and 5.5 <= snr["value_db"] <= 24.5
                and row["snr_scaled"] == (snr["value_db"] - 5.5) / 19.0
                and snr["controller_session_uuid"] == row["decision_identity"]["session_uuid"])
        if not okay:
            errors.append({"frame_id": row["frame_id"], "latest": latest, "observation": snr})
    checks["all_admitted_snr_is_latest_received_causal_lease"] = not errors
    checks["snr_is_bound_to_exact_executed_decision_and_state_feature"] = not join_errors
    return {"schema": "splitfusion.run5_live_avo.snr_postflight.v1", "passed": all(checks.values()),
            "checks": checks, "admitted_decisions": len(selected), "recorded_decisions": len(decisions),
            "received_controller_snapshots": len(snapshots), "errors": errors, "identity_errors": join_errors,
            "claim_scope": "CAUSAL_INPUT_AND_ACTOR_IDENTITY_NOT_POLICY_PERFORMANCE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    args = parser.parse_args()
    result = verify_capture(args.capture_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)
