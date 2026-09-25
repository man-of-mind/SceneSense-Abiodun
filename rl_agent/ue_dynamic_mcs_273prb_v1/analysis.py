"""Causal analysis for the target-radio dynamic prior-MCS capture."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_dynamic_mcs_273prb_v1 import contract as C
from rl_agent.ue_dynamic_mcs_273prb_v1.probe_sender import FIELDS as SENDER_FIELDS
from rl_agent.ue_mcs_backlog_calibration_v1 import contract as BC
from rl_agent.ue_mcs_backlog_calibration_v1 import decision_join as DJ


PROFILE_SCHEDULE_FIELDS = (
    "profile_id", "trace_id", "step_index", "target_snr_db",
    "mapped_noise_power_db", "scheduled_action_open_monotonic_ns",
    "command_due_monotonic_ns", "command_send_monotonic_ns",
    "command_ack_monotonic_ns", "command_ack_latency_ms", "command_status",
    "command_precedes_action_open", "clamped",
)

OBSERVATION_FIELDS = (
    "profile_id", "trace_id", "decision_index", "partition",
    "scheduled_action_open_monotonic_ns", "actual_send_open_monotonic_ns",
    "schedule_lag_ms", "mcs_status", "prior_ul_mcs_index", "mcs_age_ms",
    "source_grant_monotonic_ns", "source_grant_rnti",
    "source_grant_dci_frame", "source_grant_dci_slot",
    "source_grant_sched_frame", "source_grant_sched_slot",
    "source_grant_mcs_table", "source_grant_round", "source_grant_ndi",
    "source_provenance_sha256", "policy_feature_json",
    "target_snr_db_verifier_only", "hidden_profile_verifier_only",
)

TRANSITION_FIELDS = (
    "profile_id", "partition", "current_decision_index",
    "successor_decision_index", "duration_tensors", "scheduled_delta_ns",
    "current_mcs_status", "current_prior_ul_mcs_index",
    "successor_mcs_status", "successor_prior_ul_mcs_index",
    "learning_eligible", "reset_required_after_current",
)


class AnalysisError(RuntimeError):
    """Evidence cannot be analyzed without violating the preregistration."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _read_exact(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        require(tuple(header) == tuple(fields), f"{path}: header drift: {header}")
        rows = []
        for line, values in enumerate(reader, 2):
            require(len(values) == len(fields), f"{path}:{line}: ragged CSV row")
            rows.append(dict(zip(fields, values)))
        return rows


def _write_csv_x(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_json_x(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _grant_digest(grant: DJ.UeUlGrant) -> str:
    value = {
        "monotonic_ns": grant.monotonic_ns,
        "rnti": grant.rnti,
        "dci_frame": grant.dci_frame,
        "dci_slot": grant.dci_slot,
        "sched_frame": grant.sched_frame,
        "sched_slot": grant.sched_slot,
        "mcs": grant.mcs,
        "mcs_table": grant.mcs_table,
        "harq_pid": grant.harq_pid,
        "ndi": grant.ndi,
        "rv": grant.rv,
        "round": grant.harq_round,
    }
    return C.canonical_json_sha256(value)


def _select_prior(
    grants: Sequence[DJ.UeUlGrant], times: Sequence[int], decision_ns: int,
) -> tuple[DJ.UeUlGrant | None, str, float | None]:
    index = bisect.bisect_left(times, decision_ns) - 1
    if index < 0:
        return None, "MISSING_NO_PRIOR_GRANT", None
    grant = grants[index]
    age_ns = decision_ns - grant.monotonic_ns
    require(age_ns > 0, "strictly-prior grant selection produced nonpositive age")
    if age_ns > C.MCS_MAX_AGE_NS:
        return grant, "STALE", age_ns / 1e6
    return grant, "VALID", age_ns / 1e6


def _cell_paths(cell_dir: Path) -> dict[str, Path]:
    return {
        "sender": cell_dir / "sender_decisions.csv",
        "schedule": cell_dir / "profile_schedule.csv",
        "pdcp": cell_dir / "ttracer/ue/csv/NR_PDCP_TX_SDU.csv",
        "dci": cell_dir / "ttracer/ue/csv/NRUE_MAC_DCI_GRANT.csv",
        "gnb": cell_dir / "ttracer/gnb/csv/GNB_MAC_UL_MCS_DECISION.csv",
        "record": cell_dir / "cell_record.json",
    }


def analyze_cell(cell_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    paths = _cell_paths(cell_dir)
    for label, path in paths.items():
        require(path.is_file(), f"{label} evidence missing: {path}")

    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    require(record.get("status") == "CAPTURED", f"cell did not capture: {record}")
    profile_id = str(record["profile_id"])
    require(profile_id in C.PROFILE_IDS, f"unregistered profile {profile_id}")
    plan_by_profile = {plan.profile_id: plan for plan in C.build_plan()}
    plan = plan_by_profile[profile_id]
    require(
        record.get("radio_profile_id") == C.RADIO_PROFILE_ID,
        "cell record radio identity drifted",
    )
    require(record.get("trace_id") == plan.trace_id, "cell trace ID drifted")
    require(
        record.get("trace_sha256") == plan.trace_sha256,
        "cell trace digest drifted",
    )

    sender = _read_exact(paths["sender"], SENDER_FIELDS)
    schedule = _read_exact(paths["schedule"], PROFILE_SCHEDULE_FIELDS)
    pdcp = DJ.read_exact_csv(paths["pdcp"], BC.PDCP_TX_SDU_HEADER)
    dci = DJ.read_exact_csv(paths["dci"], BC.DCI_GRANT_HEADER)
    gnb = DJ.read_exact_csv(paths["gnb"], BC.GNB_MCS_DECISION_HEADER)

    require(len(sender) == C.FRAMES_PER_PROFILE, "sender decision count drifted")
    require(len(schedule) == C.FRAMES_PER_PROFILE, "profile schedule count drifted")
    require(
        [int(row["decision_index"]) for row in sender]
        == list(range(C.FRAMES_PER_PROFILE)),
        "sender indices are not the exact registered grid",
    )
    require(
        [int(row["step_index"]) for row in schedule]
        == list(range(C.FRAMES_PER_PROFILE)),
        "profile indices are not the exact registered grid",
    )
    require(
        all(
            row["profile_id"] == plan.profile_id
            and row["trace_id"] == plan.trace_id
            and int(row["action_id"]) == C.PROBE_ACTION_ID
            and row["partition"] == C.partition_for(int(row["decision_index"]))
            and int(row["payload_bytes"]) == C.PROBE_PAYLOAD_BYTES
            and int(row["chunks_per_frame"]) == C.PROBE_CHUNKS_PER_FRAME
            for row in sender
        ),
        "sender evidence identity/action/payload drifted",
    )
    require(
        all(
            row["profile_id"] == plan.profile_id
            and row["trace_id"] == plan.trace_id
            for row in schedule
        ),
        "RF schedule profile/trace identity drifted",
    )

    bridge = DJ.build_clock_bridge(pdcp)
    window = DJ.audit_bridge_window(bridge, dci, sender, slack_s=30.0)
    require(window["verified"], "converted DCI events do not land in the cell window")
    all_grants, grant_counts = DJ.build_ul_grants(dci, bridge)
    table0 = [grant for grant in all_grants if grant.mcs_table == 0]
    excluded_table = len(all_grants) - len(table0)
    require(bool(table0), "no UE-decoded round-0/table-0 UL grants")
    require(all(0 <= grant.mcs <= 28 for grant in table0), "MCS outside table-0 domain")
    rntis = {grant.rnti for grant in table0}
    require(len(rntis) == 1, f"expected exactly one UE RNTI, observed {sorted(rntis)}")
    times = [grant.monotonic_ns for grant in table0]

    gnb_rows = DJ.build_gnb_mcs_decisions(gnb, bridge)
    provenance = DJ.audit_ue_gnb_mcs_provenance(
        table0, [row for row in gnb_rows if row.mcs_table == 0], max_delta_ms=10.0
    )

    observations: list[dict[str, Any]] = []
    for action, command in zip(sender, schedule):
        index = int(action["decision_index"])
        require(int(command["step_index"]) == index, "schedule/action index mismatch")
        decision_ns = int(action["scheduled_monotonic_ns"])
        require(
            int(command["scheduled_action_open_monotonic_ns"]) == decision_ns,
            "RF schedule and action-open grid differ",
        )
        grant, status, age_ms = _select_prior(table0, times, decision_ns)
        valid = status == "VALID"
        observations.append({
            "profile_id": profile_id,
            "trace_id": action["trace_id"],
            "decision_index": index,
            "partition": C.partition_for(index),
            "scheduled_action_open_monotonic_ns": decision_ns,
            "actual_send_open_monotonic_ns": int(action["decision_monotonic_ns"]),
            "schedule_lag_ms": float(action["schedule_lag_ms"]),
            "mcs_status": status,
            "prior_ul_mcs_index": grant.mcs if valid and grant else None,
            "mcs_age_ms": age_ms,
            "source_grant_monotonic_ns": grant.monotonic_ns if grant else None,
            "source_grant_rnti": grant.rnti if grant else None,
            "source_grant_dci_frame": grant.dci_frame if grant else None,
            "source_grant_dci_slot": grant.dci_slot if grant else None,
            "source_grant_sched_frame": grant.sched_frame if grant else None,
            "source_grant_sched_slot": grant.sched_slot if grant else None,
            "source_grant_mcs_table": grant.mcs_table if grant else None,
            "source_grant_round": grant.harq_round if grant else None,
            "source_grant_ndi": grant.ndi if grant else None,
            "source_provenance_sha256": _grant_digest(grant) if grant else None,
            "policy_feature_json": (
                json.dumps({"prior_ul_mcs_index": grant.mcs}, separators=(",", ":"))
                if valid and grant else "{}"
            ),
            # Verifier-only metadata is explicitly named and never enters the
            # policy feature JSON or the Run-4 state vector.
            "target_snr_db_verifier_only": float(command["target_snr_db"]),
            "hidden_profile_verifier_only": profile_id,
        })

    by_index = {int(row["decision_index"]): row for row in observations}
    transitions: list[dict[str, Any]] = []
    expected_pairs = C.registered_transition_indices()
    for current_index, next_index in expected_pairs:
        current, successor = by_index[current_index], by_index[next_index]
        delta = (
            int(successor["scheduled_action_open_monotonic_ns"])
            - int(current["scheduled_action_open_monotonic_ns"])
        )
        require(delta == C.SUCCESSOR_DELTA_NS, "duration-aware successor grid drifted")
        require(current["partition"] == successor["partition"], "partition crossed")
        eligible = (
            current["mcs_status"] == "VALID"
            and successor["mcs_status"] == "VALID"
        )
        transitions.append({
            "profile_id": profile_id,
            "partition": current["partition"],
            "current_decision_index": current_index,
            "successor_decision_index": next_index,
            "duration_tensors": C.ACTION_HOLD_DURATION_TENSORS,
            "scheduled_delta_ns": delta,
            "current_mcs_status": current["mcs_status"],
            "current_prior_ul_mcs_index": current["prior_ul_mcs_index"],
            "successor_mcs_status": successor["mcs_status"],
            "successor_prior_ul_mcs_index": successor["prior_ul_mcs_index"],
            "learning_eligible": eligible,
            "reset_required_after_current": False,
        })

    # Explicitly record the indices at which a duration-2 successor would
    # cross the fit/validation boundary or leave the profile.
    reset_indices = []
    for index in range(C.FRAMES_PER_PROFILE):
        try:
            C.successor_index(index)
        except C.ContractError:
            reset_indices.append(index)

    lags = [float(row["schedule_lag_ms"]) for row in observations]
    partitions: dict[str, Any] = {}
    for partition in (C.FIT, C.INTERNAL_VALIDATION):
        rows = [row for row in observations if row["partition"] == partition]
        valid = [row for row in rows if row["mcs_status"] == "VALID"]
        partitions[partition] = {
            "rows": len(rows),
            "valid": len(valid),
            "missing": sum(row["mcs_status"].startswith("MISSING") for row in rows),
            "stale": sum(row["mcs_status"] == "STALE" for row in rows),
            "valid_fraction": len(valid) / len(rows),
            "unique_valid_mcs": sorted(
                {int(row["prior_ul_mcs_index"]) for row in valid}
            ),
        }

    sender_ok = all(
        row["terminal_reason"] == "ALL_CHUNKS_HANDED_TO_SOCKET"
        and int(row["chunks_sent"]) == C.PROBE_CHUNKS_PER_FRAME
        and int(row["chunks_dropped"]) == 0
        for row in sender
    )
    command_ok = all(
        row["clamped"] in ("False", "false", "0")
        and row["command_precedes_action_open"] in ("True", "true", "1")
        and row["command_status"] in ("PRIMED", "HOLD", "ACK_ON_TIME")
        for row in schedule
    )
    unique_mcs = sorted(
        {int(row["prior_ul_mcs_index"]) for row in observations
         if row["mcs_status"] == "VALID"}
    )

    gates = {
        "exact_decision_grid": len(observations) == C.FRAMES_PER_PROFILE,
        "sender_complete_without_backpressure": sender_ok,
        "rf_commands_causal_unclamped": command_ok,
        "schedule_lag_p99": (
            (_percentile(lags, 0.99) or 0.0) <= C.MAX_SCHEDULE_LAG_P99_MS
        ),
        "clock_bridge": (
            bridge.residual_p95_ns / 1000
            <= C.MAX_CLOCK_BRIDGE_RESIDUAL_P95_US
        ),
        "fit_mcs_coverage": (
            partitions[C.FIT]["valid_fraction"]
            >= C.MIN_VALID_MCS_FRACTION_PER_PARTITION
        ),
        "validation_mcs_coverage": (
            partitions[C.INTERNAL_VALIDATION]["valid_fraction"]
            >= C.MIN_VALID_MCS_FRACTION_PER_PARTITION
        ),
        "mcs_informative": len(unique_mcs) >= C.MIN_UNIQUE_MCS_PER_PROFILE,
        "gnb_provenance": (
            provenance["coverage"] is not None
            and provenance["coverage"] >= C.MIN_GNB_PROVENANCE_COVERAGE
            and provenance["ue_final_mismatches"] == 0
        ),
        "single_ue_rnti": len(rntis) == 1,
        "only_table0_new_data_grants": excluded_table == 0,
        "duration2_successors_exact": (
            len(transitions) == len(expected_pairs)
            and all(row["scheduled_delta_ns"] == C.SUCCESSOR_DELTA_NS
                    for row in transitions)
        ),
        "no_cross_partition_transition": all(
            by_index[int(row["current_decision_index"])]["partition"]
            == by_index[int(row["successor_decision_index"])]["partition"]
            for row in transitions
        ),
    }
    report = {
        "profile_id": profile_id,
        "trace_id": record["trace_id"],
        "radio_profile_id": record["radio_profile_id"],
        "rows": len(observations),
        "clock_bridge": bridge.to_json(),
        "bridge_window": window,
        "grant_counts": {**grant_counts, "excluded_non_table0": excluded_table},
        "gnb_provenance": provenance,
        "partitions": partitions,
        "unique_valid_mcs": unique_mcs,
        "schedule_lag_ms": {
            "p50": _percentile(lags, 0.50),
            "p95": _percentile(lags, 0.95),
            "p99": _percentile(lags, 0.99),
            "max": max(lags),
        },
        "duration2_transitions": len(transitions),
        "duration2_learning_eligible": sum(
            bool(row["learning_eligible"]) for row in transitions
        ),
        "reset_required_indices": reset_indices,
        "gates": gates,
        "passed": all(gates.values()),
    }
    return observations, transitions, report


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    analysis_dir = run_dir / "analysis"
    require(not analysis_dir.exists(), f"analysis output already exists: {analysis_dir}")
    all_observations: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    for plan in C.build_plan():
        cell_dir = run_dir / "cells" / f"{plan.run_index:02d}__{plan.profile_id.lower()}"
        observations, transitions, report = analyze_cell(cell_dir)
        all_observations.extend(observations)
        all_transitions.extend(transitions)
        profiles[plan.profile_id] = report

    aggregate_gates = {
        "two_registered_profiles": set(profiles) == set(C.PROFILE_IDS),
        "all_profile_gates": all(report["passed"] for report in profiles.values()),
        "exact_observation_count": (
            len(all_observations) == len(C.PROFILE_IDS) * C.FRAMES_PER_PROFILE
        ),
        "policy_feature_excludes_hidden_metadata": all(
            set(json.loads(row["policy_feature_json"]))
            <= {"prior_ul_mcs_index"}
            for row in all_observations
        ),
    }
    result = {
        "schema": "scenesense.ue_dynamic_mcs_273prb.analysis.v1",
        "contract_id": C.CONTRACT_ID,
        "claim_boundary": C.CLAIM_BOUNDARY,
        "radio_profile_id": C.RADIO_PROFILE_ID,
        "design_sha256": C.canonical_json_sha256(C.design_record()),
        "profiles": profiles,
        "aggregate_gates": aggregate_gates,
        "passed": all(aggregate_gates.values()),
        "policy_input_boundary": (
            "ONLY policy_feature_json; target_snr_db_verifier_only, profile_id, "
            "frame/timestamps and gNB data are forbidden actor inputs"
        ),
    }
    _write_csv_x(analysis_dir / "mcs_observations.csv", OBSERVATION_FIELDS,
                 all_observations)
    _write_csv_x(analysis_dir / "duration2_transitions.csv", TRANSITION_FIELDS,
                 all_transitions)
    _write_json_x(analysis_dir / "analysis.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = analyze_run(build_parser().parse_args(argv).run_dir)
    except (AnalysisError, DJ.JoinError, C.ContractError) as exc:
        print(f"dynamic-MCS analysis refused: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
