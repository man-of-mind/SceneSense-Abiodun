"""Outcome-aware verdict for an external fixed M1 capture.

The retained generic qualification expects an actor proposal/replay.  This
verdict keeps all infrastructure, identity, accounting, reward, transport,
GC, and cleanup checks while marking actor-only gates inapplicable.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

from rl_agent.splitfusion_run5_full_live_retention_v1_analysis.checks import check as gc_check


EXPECTED_ROUTE_STOP = "TypeError: 'bool' object does not support item assignment"
CORE_GATES = (
    "G1_FRAME_ACCOUNTING", "G2_ACTOR_HASH_UNCHANGED",
    "G3_ZERO_IDENTITY_CONFLICTS", "G4_COVERAGE_REPORTED",
    "G5_SAMPLER_REPORTED", "G6_NO_PROFILER_INDUCED_FAULT",
    "G7_INSTRUMENTATION_EVIDENCE_VALID", "G9_HOST_RESTORED_CLEAN")
PHASE_GATES = ("P1_IDENTITY", "P3_HOLD_TICKET", "P4_ACCOUNTING",
               "P6_RESTORE_COLD", "P7_FEEDBACK_OVER_DOWNLINK",
               "P8_NO_INFRASTRUCTURE_FAULT")


def one(root, launch_returncode):
    root = Path(root)
    capture = root / "capture"

    def find(name):
        values = list(capture.rglob(name))
        if len(values) != 1:
            raise ValueError(f"expected one {name}, found {len(values)}")
        return values[0]

    def load(name):
        return json.loads(find(name).read_bytes())

    registration = json.loads((root / "registration.json").read_bytes())
    expected = registration["external_fixed_action"]
    evidence = load("PHASE6_UE_EVIDENCE.json")
    edge = load("edge_report.json")
    fixed = load("FIXED_ACTION_SUMMARY.json")
    qualification = load("QUALIFICATION_RESULT.json")
    cell = load("CELL_RESULT.json")
    analysis = json.loads((capture / "PROFILER_ANALYSIS.json").read_bytes())
    retention = json.loads((root / "RETENTION_RESULT.json").read_bytes())
    profile_path = find("edge.profile.jsonl")
    spans = [json.loads(line) for line in profile_path.read_text().splitlines()
             if line.strip()]
    compute = [row for row in spans if row.get("r") == "span"
               and row.get("stage") == "edge.compute"
               and row.get("ph") == "live"]
    successful_frames = [row for row in compute if row.get("ok") is True]
    successful_predictions = [row for row in successful_frames
                              if row.get("reward_requested") is True]
    failed_predictions = [row for row in compute if row.get("reward_requested") is True
                          and row.get("ok") is not True]
    decisions = evidence["counters"]["policy_decisions"]
    route_csv = find("per_frame_metrics.csv")
    with route_csv.open(newline="") as stream:
        route_rows = list(csv.DictReader(stream))
    route_counts = dict(collections.Counter(row["prepare_status"] for row in route_rows))
    resolutions = dict(collections.Counter(row.get("terminal")
                                           for row in evidence.get("resolutions", [])))
    edge_terminals = {key: value for key, value in edge.get("counters", {}).items()
                      if key.startswith("edge_terminal_")}
    quality = analysis.get("semantics") or {}
    fixed_counts = fixed["counters"]
    counters = evidence["counters"]
    selection_pass = (
        fixed["fixed_action"] == expected
        and counters["policy_decisions"] == fixed_counts["decisions"] == 500
        and counters["policy_holds"] == fixed_counts["holds"] == 500
        and counters["fallbacks"] == fixed_counts["fallbacks"] == 0
        and counters["actor_calls"] == fixed_counts["actor_calls"] == 0
        and fixed_counts["executed_mismatch"] == 0)
    identity_pass = all(
        (int(row["mode_id"]), int(row["q_e4"])) ==
        (int(expected["mode_id"]), int(expected["q_e4"]))
        for row in compute)
    gates = qualification.get("gates") or {}
    phase = (qualification.get("phase6_report_only") or {}).get("gates") or {}
    actor_only = dict(
        G8_REWARD_TIMEOUT_SEMANTICS=dict(
            status="NOT_APPLICABLE_ACTOR_REPLAY_COMPONENT",
            retained_value=gates.get("G8_REWARD_TIMEOUT_SEMANTICS"),
            reward_semantics_pass=(quality.get("reward") or {}).get("passed"),
            receipt_marks_pass=(quality.get("receipt_marks") or {}).get("passed"),
            actor_replay=(quality.get("actor_replay") or {})),
        P0_POLICY_COVERAGE=dict(status="NOT_APPLICABLE_EXTERNAL_FIXED_SELECTOR",
                                retained_value=phase.get("P0_POLICY_COVERAGE")),
        P5_NO_ACTOR_AFTER_REFUSAL=dict(status="NOT_APPLICABLE_ACTOR_MUST_NOT_BE_CALLED",
                                       retained_value=phase.get("P5_NO_ACTOR_AFTER_REFUSAL")))
    child = cell.get("child") or {}
    cleanup = cell.get("cleanup") or {}
    route = child.get("route_detail") or {}
    route_stop_pass = (route.get("error") == EXPECTED_ROUTE_STOP
                       and (child.get("collector") or {}).get("stop_reason") ==
                           "TRANSMITTED_BUDGET_REACHED")
    protocol_faults = []
    if evidence.get("faulted") is not None:
        protocol_faults.append(evidence["faulted"])
    protocol_faults.extend(edge.get("failures") or [])
    for early in capture.rglob("EDGE_EARLY_GT.json"):
        value = json.loads(early.read_bytes())
        if value.get("error") is not None:
            protocol_faults.append(value["error"])
    checks = dict(
        selection_identity=selection_pass and identity_pass,
        frame_accounting=(len(evidence["transmitted_identities"]) == 1000
                          and decisions == 500 and counters["policy_holds"] == 500),
        all_non_actor_generic_gates=all(gates.get(key) is True for key in CORE_GATES),
        all_non_actor_phase_gates=all(phase.get(key) is True for key in PHASE_GATES),
        reward_semantics=(quality.get("reward") or {}).get("passed") is True,
        receipt_marks=(quality.get("receipt_marks") or {}).get("passed") is True,
        no_protocol_faults=not protocol_faults,
        gc_and_producer=gc_check(root)["passed"],
        cleanup=(child.get("return_code") == 0
                 and all((child.get("cleanup") or {}).get(key) is True
                         for key in ("edge_stopped", "map_stopped",
                                     "target_snr_restored"))
                 and cleanup.get("all_gates_passed") is True
                 and not qualification.get("leftover_processes")),
        registered_budget_stop=route_stop_pass,
        retention=(retention.get("status_counts") == {"RETAINED_COMPLETE": 100}),
    )
    passed = all(checks.values())
    return dict(
        schema="splitfusion.m1_baselines_v1.fixed_capture_verdict.v3",
        verdict=("VALID_M1_EXTERNAL_FIXED_MEASUREMENT" if passed
                 else "INVALID_M1_EXTERNAL_FIXED_MEASUREMENT"),
        passed=passed, checks=checks, launch_returncode=launch_returncode,
        raw_generic_qualification=dict(
            verdict=qualification.get("verdict"), passed=qualification.get("passed"),
            preserved=True, actor_only_gates=actor_only,
            interpretation=("generic actor-driven qualification returns non-pass because the "
                            "registered external selector never calls/replays the actor")),
        startup="EDGE_READY_AND_1000_FRAMES_TRANSMITTED",
        registered_action=expected,
        frame_accounting=dict(
            offered=1000, transmitted=len(evidence["transmitted_identities"]),
            successfully_processed=len(successful_frames),
            map_publications=int(edge.get("publications", 0)),
            policy_decisions=decisions, policy_holds=counters["policy_holds"],
            fallbacks=counters["fallbacks"], actor_calls=counters["actor_calls"],
            route_input_observations=route_counts),
        prediction_completion=dict(
            denominator_policy_decisions=decisions,
            successful_edge_compute_with_cpu_prediction=len(successful_predictions),
            incomplete=decisions - len(successful_predictions),
            exception_ending_prediction_spans=len(failed_predictions),
            rule=("successful live edge.compute span with reward_requested and CPU "
                  "prediction materialized; map publication is separate")),
        controller_outcomes=resolutions, edge_typed_terminals=edge_terminals,
        protocol_faults=protocol_faults,
        quality_evaluations=len(edge.get("evaluations", [])),
        retention=dict(status_counts=retention.get("status_counts"),
                       observed_policy_decisions=retention.get("observed_policy_decisions"),
                       storage_bytes=retention.get("storage_bytes")),
        route_summary_limitation=dict(
            route_metrics_summary_available=False,
            retained_post_budget_stop_signature=route.get("error"),
            capture_budget_reached_before_stop=route_stop_pass,
            use="per-frame route positions/input outcomes only; no durable route-summary claim"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch-returncode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = one(args.root, args.launch_returncode)
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(verdict=result["verdict"], passed=result["passed"],
                          completion=result["prediction_completion"],
                          frames=result["frame_accounting"]), sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
