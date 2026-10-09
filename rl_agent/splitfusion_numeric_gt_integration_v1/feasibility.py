"""Read-only v11 critical-path accounting; never a measured intervention.

Only the requested, newly created output directory is written. No retained code
is imported, and no service, GPU, model, replay, or timing benchmark is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIAL = ROOT / "experiments/splitfusion_gt_main_thread_live_trial_v1/20261005_v11"
LABEL = "FIXED_SPAN_COUNTERFACTUAL_NOT_MEASURED_INTERVENTION"
DEADLINE_NS = 170_000_000
BUDGETS_MS = (0, 5, 10, 15, 20)
PROFILE_IDENTITY = ("frame_id", "tensor_seq", "session_uuid", "decision_seq",
                    "ticket_seq", "mode_id", "q_e4", "reward_requested")
STAGES = ("edge.compute", "edge.evaluator_submit", "edge.qperc_evaluate", "edge.feedback_emit")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evidence:
    """Hash precisely the files used, rather than duplicating a provenance tree."""

    def __init__(self):
        self.hashes = {}

    def read(self, path, *, expected=None, lines=False):
        path = Path(path).absolute()
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                "missing or symlinked evidence: " + str(path))
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(expected is None or digest == expected, "evidence hash mismatch: " + str(path))
        self.hashes[str(path)] = digest
        return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)

    def verify_unchanged(self):
        for path, digest in self.hashes.items():
            require(sha256(path) == digest, "input changed during analysis: " + path)


def index(rows, label, nested=None):
    result = {}
    for row in rows:
        frame = (row[nested] if nested else row)["frame_id"]
        require(type(frame) is int and frame not in result, "duplicate/invalid frame: " + label)
        result[frame] = row
    return result


def raw_stamp(value):
    require(value.get("domain") == "CLOCK_MONOTONIC_RAW"
            and type(value.get("ns")) is int, "stamp is not CLOCK_MONOTONIC_RAW")
    return value["ns"]


def verify_clock(ue_profile, edge_profile):
    headers = [[row for row in rows if row.get("r") == "header"]
               for rows in (ue_profile, edge_profile)]
    require(all(len(rows) == 1 for rows in headers), "exactly one clock header required")
    ue, edge = [rows[0] for rows in headers]
    require(ue.get("role") == "ue" and edge.get("role") == "edge", "profiler roles differ")
    require(ue.get("boot_id") and ue["boot_id"] == edge.get("boot_id")
            and ue.get("clock_domain") == edge.get("clock_domain") == "CLOCK_MONOTONIC_RAW",
            "shared CLOCK_MONOTONIC_RAW boot proof absent")
    return ue["boot_id"]


def accounting(*, action, compute_start, compute_end, submit, admitted, evaluate_start,
               emit_start, emit_end, receipt):
    """All arithmetic is exact integer nanoseconds before display conversion."""
    stamps = (action, compute_start, compute_end, submit, admitted, evaluate_start,
              emit_start, emit_end, receipt)
    require(all(type(stamp) is int and stamp >= 0 for stamp in stamps), "invalid timestamp")
    require(action <= compute_start <= compute_end <= submit <= evaluate_start
            <= emit_start <= emit_end <= receipt, "noncausal prediction/evaluation/feedback path")
    require(action <= admitted <= evaluate_start, "noncausal GT admission")
    joined = max(submit, admitted)
    post_join = receipt - joined
    measured = receipt - action
    reconstructed = joined - action + post_join
    require(reconstructed == measured, "critical-path reconstruction differs")
    floor = compute_end - action + receipt - emit_end
    return {
        "action_to_feedback_ns": measured,
        "action_to_prediction_ready_ns": compute_end - action,
        "edge_compute_ns": compute_end - compute_start,
        "prediction_ready_to_evaluator_submit_ns": submit - compute_end,
        "action_to_evaluator_submit_ns": submit - action,
        "action_to_gt_admission_ns": admitted - action,
        "gt_later_than_evaluator_submit": admitted > submit,
        "gt_critical_wait_ns": max(0, admitted - submit),
        "join_to_evaluation_start_ns": evaluate_start - joined,
        "evaluation_start_to_emit_start_ns": emit_start - evaluate_start,
        "feedback_emit_ns": emit_end - emit_start,
        "post_emit_to_ue_receipt_ns": receipt - emit_end,
        "downstream_residual_ns": post_join,
        "reconstructed_feedback_ns": reconstructed,
        "reconstruction_error_ns": reconstructed - measured,
        "gt_ready_by_submit_fixed_residual_ns": submit - action + post_join,
        "ideal_gt_zero_bridge_floor_ns": floor,
        "remaining_bridge_budget_at_170ms_ns": DEADLINE_NS - floor,
        **{f"ideal_gt_bridge_{b}ms_ns": floor + b * 1_000_000 for b in BUDGETS_MS},
    }


def stats_ns(values):
    values = sorted(values)
    require(values, "empty population")

    def percentile(q):
        rank = (len(values) - 1) * q
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        return (values[lower] + (values[upper] - values[lower]) * (rank - lower)) / 1e6

    return dict(n=len(values), min=values[0] / 1e6, p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), max=values[-1] / 1e6,
                mean=statistics.mean(values) / 1e6,
                inclusive_170ms_count=sum(value <= DEADLINE_NS for value in values),
                inclusive_170ms_fraction=sum(value <= DEADLINE_NS for value in values) / len(values))


def analyze_data(arm, comparison, ue, edge, process, handoff, ue_profile, edge_profile):
    """Reject incomplete/ambiguous joins; never quietly drop a policy frame."""
    boot_id = verify_clock(ue_profile, edge_profile)
    require(process.get("clock") == "CLOCK_MONOTONIC_RAW" and process.get("complete") is True
            and process.get("dropped") == 0 and not process.get("errors"), "GT timeline incomplete")
    require(handoff.get("complete") is True and not handoff.get("errors"), "handoff incomplete")
    require(comparison.get("valid_for_comparison") is True, "retained comparison is not valid")
    old = index(comparison["rows"], "comparison")
    policies = index([row for row in ue["decisions"] if row["kind"] == "POLICY_DECISION"], "policy")
    cohort = set(policies)
    require(cohort and set(old) == cohort, "comparison/policy cohort differs")
    all_sent = index(ue["transmitted_identities"], "transmission")
    sent = {frame: row for frame, row in all_sent.items()
            if row["run4_identity"]["frame_kind"] == "POLICY_DECISION"}
    indexes = {
        "transmission": sent,
        "feedback": index(ue["feedback_rows"], "feedback"),
        "evaluation": index(edge["evaluations"], "evaluation"),
        "request": index(process["requests"], "request", "identity"),
        "broker": index(handoff["broker"]["ledger"], "broker", "identity"),
    }
    for stage in STAGES:
        indexes[stage] = index([row for row in edge_profile if row.get("stage") == stage
                               and row.get("ph") == "live" and row.get("reward_requested") is True
                               and row.get("r") in ("span", "mark")], stage)
    for label, indexed in indexes.items():
        require(set(indexed) == cohort, "missing/foreign policy frames: " + label)

    rows = []
    for frame in sorted(cohort, key=lambda f: raw_stamp(policies[f]["action_open"])):
        policy, retained = policies[frame], old[frame]
        identity = sent[frame]["run4_identity"]
        request, broker = indexes["request"][frame], indexes["broker"][frame]
        evaluation, reply = indexes["evaluation"][frame], indexes["feedback"][frame]
        capture = request["identity"]
        require(identity["reward_requested"] is True and identity["frame_id"] == frame
                and identity["session_uuid"] == policy["decision_session_uuid"]
                and all(identity[key] == policy[key] for key in ("tensor_seq", "mode_id", "q_e4")),
                "policy/transmission identity differs")
        require(capture == broker["identity"] == evaluation["timing"]["gt_handoff"]["identity"]
                and capture["capture_timestamp_ns"] == policy["capture"]["ns"]
                and capture["run_id"] == edge["run_id"] == retained["run_id"]
                and capture["cell_id"] == edge["cell_id"] == retained["cell_id"]
                and request["ipc"]["frame_id"] == frame, "capture identity differs")
        require(policy["capture"]["domain"] == "PHYSICAL_CAPTURE_WALL_CLOCK_REALTIME",
                "capture clock domain differs")
        require(all(retained[key] == identity[key] for key in PROFILE_IDENTITY
                    if key != "reward_requested") and evaluation["run4_identity"] == identity,
                "evaluator/comparison identity differs")
        require(evaluation["feedback_sha256"] == reply["sha256"], "feedback bytes binding differs")
        require(evaluation["kind"] == reply["kind"] == retained["feedback_kind"]
                and evaluation["reason"] == reply["reason"] == retained["feedback_reason"],
                "feedback outcome differs")
        require(evaluation.get("q_perc") == reply.get("q_perc") == retained.get("q_perc"),
                "quality value differs")
        component = evaluation["timing"]["gt_handoff"]["components"]["scoring_packet"]
        require(component["packet_sha256"] == broker["packet_sha256"]
                and component["ready_raw_ns"] == broker["admitted_raw_ns"],
                "GT admission binding differs")
        for stage in STAGES:
            profile_row = indexes[stage][frame]
            require(all(profile_row.get(key) == identity[key] for key in PROFILE_IDENTITY),
                    "profiler identity differs: " + stage)
            if profile_row["r"] == "span":
                require(profile_row.get("ok") is True, "failed profiler span: " + stage)
        compute, submit, evaluate, emit = [indexes[stage][frame] for stage in STAGES]
        action, receipt = raw_stamp(policy["action_open"]), raw_stamp(reply["receipt"])
        require(action == retained["action_open_raw_ns"]
                and receipt == retained["feedback_receipt_raw_ns"]
                and broker["admitted_raw_ns"] == retained["broker_admitted_raw_ns"]
                and request["ipc"]["start_raw_ns"] == retained["request_start_raw_ns"],
                "retained raw timestamps differ")
        values = accounting(action=action, compute_start=compute["start_raw_ns"],
                            compute_end=compute["end_raw_ns"], submit=submit["t_raw_ns"],
                            admitted=broker["admitted_raw_ns"], evaluate_start=evaluate["start_raw_ns"],
                            emit_start=emit["start_raw_ns"], emit_end=emit["end_raw_ns"], receipt=receipt)
        for computed, previous in (("action_to_feedback_ns", "action_to_feedback_ms"),
                                   ("edge_compute_ns", "edge_compute_ms"),
                                   ("post_emit_to_ue_receipt_ns", "feedback_emit_end_to_ue_receipt_ms")):
            require(values[computed] / 1e6 == retained[previous], "retained interval differs: " + previous)
        undefined = evaluation["reason"] == "QUALITY_UNDEFINED_NO_ELIGIBLE_GT"
        require(undefined == (evaluation.get("q_perc") is None), "undefined-quality status differs")
        row = dict(arm=arm, scenario_status=LABEL, **capture,
                   **{key: value for key, value in identity.items() if key != "frame_id"},
                   action_open_raw_ns=action, compute_start_raw_ns=compute["start_raw_ns"],
                   prediction_cpu_materialized_raw_ns=compute["end_raw_ns"],
                   evaluator_submit_raw_ns=submit["t_raw_ns"],
                   gt_admitted_raw_ns=broker["admitted_raw_ns"],
                   evaluation_start_raw_ns=evaluate["start_raw_ns"],
                   feedback_emit_start_raw_ns=emit["start_raw_ns"],
                   feedback_emit_end_raw_ns=emit["end_raw_ns"], feedback_receipt_raw_ns=receipt,
                   quality_undefined=undefined, q_perc=evaluation.get("q_perc"),
                   feedback_kind=reply["kind"], feedback_reason=reply["reason"],
                   feedback_sha256=reply["sha256"], scoring_packet_sha256=broker["packet_sha256"],
                   **values)
        rows.append(row)
    summary = dict(policy_frames=len(rows), output_rows=len(rows), excluded_policy_frames=0,
                   undefined_quality_frames=sum(row["quality_undefined"] for row in rows),
                   shared_raw_boot_id=boot_id, all_identity_and_clock_joins_verified=True,
                   gt_later_than_submit_count=sum(row["gt_later_than_evaluator_submit"] for row in rows),
                   maximum_absolute_reconstruction_error_ns=max(abs(row["reconstruction_error_ns"]) for row in rows),
                   observed_feedback_ms=stats_ns([row["action_to_feedback_ns"] for row in rows]),
                   gt_ready_by_submit_fixed_residual_ms=stats_ns([
                       row["gt_ready_by_submit_fixed_residual_ns"] for row in rows]),
                   ideal_gt_bridge_scenarios_ms={str(b): stats_ns([
                       row[f"ideal_gt_bridge_{b}ms_ns"] for row in rows]) for b in BUDGETS_MS},
                   observed_component_ms={key[:-3]: stats_ns([row[key] for row in rows]) for key in (
                       "action_to_prediction_ready_ns", "prediction_ready_to_evaluator_submit_ns",
                       "gt_critical_wait_ns", "join_to_evaluation_start_ns", "downstream_residual_ns",
                       "evaluation_start_to_emit_start_ns", "feedback_emit_ns", "post_emit_to_ue_receipt_ns")})
    # Retained NumPy computes quantiles after float conversion to milliseconds;
    # this diagnostic computes them before conversion. Match at sub-ns display
    # precision only; timestamp and per-frame interval checks above remain exact.
    require(math.isclose(summary["observed_feedback_ms"]["p50"],
                         comparison["metrics_ms"]["action_to_feedback"]["p50"],
                         rel_tol=0, abs_tol=1e-10),
            "retained median differs")
    return rows, summary


def analyze_trial(trial=DEFAULT_TRIAL):
    trial = Path(trial).absolute()
    evidence = Evidence()
    output_pins = evidence.read(trial / "comparison/OUTPUT_SHA256.json")
    comparison = evidence.read(trial / "comparison/RESULT.json", expected=output_pins["RESULT.json"])
    all_rows, arms = [], {}
    for arm in ("treatment", "control"):
        retained = comparison["arms"][arm]
        cell = trial / arm / "capture/cells/a71__favorable_stable"
        require(str(cell) == retained["cell"], "retained cell location differs")

        def read(relative, lines=False):
            path = cell / relative
            require(str(path) in retained["input_sha256"], "input absent from retained pins")
            return evidence.read(path, expected=retained["input_sha256"][str(path)], lines=lines)

        ue = read("run4_phase6/PHASE6_UE_EVIDENCE.json")
        edge = read("run4_phase6/edge_report.json")
        process = read("CPU_GT_PROCESS_TIMELINE.json")
        handoff = read("profiler/edge/EDGE_HANDOFF_SCOPE.json")
        ue_profile = read("profiler/ue.profile.jsonl", True)
        edge_profile = read("profiler/edge/edge.profile.jsonl", True)
        source = ROOT / "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_edge_runtime_v2.py"
        digest = sha256(source)
        require(digest == edge["source_sha256"][source.name], "compute source binding differs")
        evidence.hashes[str(source)] = digest
        rows, summary = analyze_data(arm, retained, ue, edge, process, handoff, ue_profile, edge_profile)
        arms[arm] = summary
        all_rows.extend(rows)
    evidence.hashes[str(Path(__file__).resolve())] = sha256(__file__)
    test_path = Path(__file__).with_name("test_feasibility.py")
    evidence.hashes[str(test_path)] = sha256(test_path)
    evidence.verify_unchanged()
    return all_rows, dict(
        schema="splitfusion.numeric_gt_integration.fixed_span_feasibility.v1", status=LABEL,
        measured_intervention=False, deadline_qualification_claim=False, no_live_actions=True,
        no_benchmark=True, source_trial=str(trial), inclusive_deadline_ms=170,
        percentile_method="linear interpolation at (n-1)*q; per-frame arithmetic before percentiles",
        formulae={
            "observed": "max(evaluator_submit, GT_admitted) - action_open + observed_downstream_residual",
            "downstream_residual": "UE_receipt - max(evaluator_submit, GT_admitted)",
            "gt_ready_by_submit": "evaluator_submit - action_open + observed_downstream_residual",
            "ideal_gt_bridge_b": "CPU_materialized_prediction - action_open + (UE_receipt - feedback_emit.end) + b",
        },
        assumptions=[
            "Recorded prediction path and post-emit-to-UE span remain fixed under each counterfactual.",
            "GT is admitted by prediction readiness for the ideal-GT bridge scenarios.",
            "b is a hypothetical total prediction-ready-to-emit-complete bridge budget, not measured scorer time.",
            "b includes all remaining metadata, GT admission, validation, scoring, scheduling and emission work.",
            "GT-ready-by-submit preserves the observed downstream residual, including recorded wakeup/decode/scoring overhead.",
            "No workload-contention, queueing, overlap, uplink or prediction changes are simulated.",
            "Zero bridge is an optimistic fixed-span floor, not a physical lower bound under a changed system.",
            "All undefined-quality replies stay in the deadline denominator; their numerical quality remains undefined.",
            "Control and treatment are separate unpaired runs; no frame pairing or causal improvement is claimed.",
        ],
        compute_endpoint=dict(source=str(source), source_sha256=digest, source_lines=[596, 603],
                              meaning="edge.compute return after CPU materialization of records and uint8 labels; not CUDA-end",
                              raw_stage="edge.compute.end_raw_ns"),
        arms=arms, input_sha256=evidence.hashes)


def write_results(output, rows, summary):
    output = Path(output).absolute()
    require(not output.exists(), "output already exists; refusing overwrite: " + str(output))
    require(not any(parent.is_symlink() for parent in output.parents), "symlinked output parent")
    output.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    with (output / "per_frame.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "RESULT.json").open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (output / "OUTPUT_SHA256.json").open("x") as stream:
        json.dump({name: sha256(output / name) for name in ("RESULT.json", "per_frame.csv")},
                  stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; never overwritten")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists; refusing overwrite")
    rows, summary = analyze_trial(args.trial)
    write_results(args.output, rows, summary)
    print(json.dumps({"status": LABEL, "output": str(args.output), "arms": {
        arm: {"policy_frames": value["policy_frames"], "undefined_quality_frames": value["undefined_quality_frames"],
              "observed_p50_ms": value["observed_feedback_ms"]["p50"],
              "gt_ready_by_submit_p50_ms": value["gt_ready_by_submit_fixed_residual_ms"]["p50"],
              "bridge_budget_inclusive_170ms_counts": {b: stats["inclusive_170ms_count"]
                  for b, stats in value["ideal_gt_bridge_scenarios_ms"].items()}}
        for arm, value in summary["arms"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
