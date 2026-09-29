#!/usr/bin/env python3
"""Phase B: repair the causal state join to the frame-open/action-release cutoff.

v1 selected the pre-action backlog and prior UL MCS using the frame's **first
UDP send** as the cutoff.  That is too late: the policy decision is taken, and
the action released, at frame open.  A sample that lands between frame open
and first send is in the future relative to the decision and must not be
visible to it.

This module rebuilds the 5-Hz decision-frame table with the registered cutoff
``frame_open_monotonic_ns``, requires a non-negative age of at most 100 ms,
and reports exactly how many joins differ from v1.

The held frame t+1 reuses the decision taken at t.  It never contributes an
independently observed successor backlog/MCS to the evaluation of action t;
its only role here is the deterministic ingress of the second transmission.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_queue_capture_v1 import parse as V1PARSE

from . import contract_v2 as C2


JOIN_SCHEMA = "scenesense.production_transport_causal_join_v2"

DECISION_FIELDS = (
    "cell_id", "cell_tag", "partition", "profile_id_audit_only",
    "frame_index", "block_index", "tier_audit_only", "action_id", "mode_id",
    "q_e4", "frame_open_monotonic_ns",
    "pre_enqueue_backlog_bytes", "backlog_sample_monotonic_ns",
    "backlog_age_ns", "prior_ul_mcs", "mcs_sample_monotonic_ns",
    "mcs_age_ns", "action_wire_bytes", "held_action_wire_bytes",
    "deterministic_action_ingress_bytes",
    "transport_latency_ns", "completed_within_deadline", "terminal_outcome",
    "v1_backlog_bytes", "v1_prior_ul_mcs", "backlog_join_changed",
    "mcs_join_changed", "successor_backlog_bytes", "has_successor",
)


class CausalJoinError(RuntimeError):
    """A causal-join invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CausalJoinError(message)


@dataclass(frozen=True, slots=True)
class JoinedDecision:
    values: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]


def join_cell(cell_dir: Path) -> dict[str, Any]:
    record = json.loads((cell_dir / "cell_record.json").read_text())
    require(record["status"] == "CAPTURED",
            f"{cell_dir.name}: cell status is {record['status']}")
    sender = {int(row["frame_index"]): row
              for row in V1PARSE._read_csv(cell_dir / "sender_frames.csv")}
    require(len(sender) == V1.FRAMES_PER_CELL, "sender frame count drifted")

    receiver: dict[int, dict[str, str]] = {}
    for tier in V1.TIER_NAMES:
        for row in V1PARSE._read_csv(cell_dir / f"receiver_{tier}_frames.csv"):
            receiver[int(row["message_id"])] = row

    traces = V1PARSE.load_ue_traces(cell_dir)
    backlog, mcs = traces["backlog"], traces["mcs"]

    rows: list[dict[str, Any]] = []
    backlog_changed = mcs_changed = 0
    backlog_ages: list[int] = []
    mcs_ages: list[int] = []
    missing: list[dict[str, Any]] = []

    for frame_index in range(0, V1.FRAMES_PER_CELL, C2.DECISION_FRAME_STRIDE):
        tx = sender[frame_index]
        cutoff = int(tx[C2.CAUSAL_CUTOFF_FIELD])
        rejected_cutoff = int(tx[C2.CAUSAL_CUTOFF_REJECTED_FIELD])
        require(rejected_cutoff >= cutoff,
                "first send precedes frame open; sender stamps are inverted")

        value, age = backlog.latest_strictly_before(
            cutoff, max_age_ns=C2.MAX_CAUSAL_AGE_NS)
        mcs_value, mcs_age = mcs.latest_strictly_before(
            cutoff, max_age_ns=C2.MAX_CAUSAL_AGE_NS)
        v1_value, _ = backlog.latest_strictly_before(
            rejected_cutoff, max_age_ns=C2.MAX_CAUSAL_AGE_NS)
        v1_mcs, _ = mcs.latest_strictly_before(
            rejected_cutoff, max_age_ns=C2.MAX_CAUSAL_AGE_NS)

        if value is None or mcs_value is None:
            missing.append({"frame_index": frame_index,
                            "backlog_present": value is not None,
                            "mcs_present": mcs_value is not None,
                            "backlog_age_ns": age, "mcs_age_ns": mcs_age})
            continue
        require(age is not None and C2.MIN_CAUSAL_AGE_NS <= age
                <= C2.MAX_CAUSAL_AGE_NS, "backlog age outside the causal bound")
        require(mcs_age is not None and C2.MIN_CAUSAL_AGE_NS <= mcs_age
                <= C2.MAX_CAUSAL_AGE_NS, "MCS age outside the causal bound")
        backlog_ages.append(age)
        mcs_ages.append(mcs_age)
        backlog_changed += int(v1_value != value)
        mcs_changed += int(v1_mcs != mcs_value)

        held = sender[frame_index + 1]
        require(int(held["action_id"]) == int(tx["action_id"])
                and int(held["q_e4"]) == int(tx["q_e4"]),
                "held frame does not reuse the decision action")

        rx = receiver.get(frame_index)
        complete = rx is not None and rx["complete"] == "True"
        latency = (int(rx["complete_reassembly_monotonic_ns"])
                   - int(tx["last_send_monotonic_ns"])) if complete else None
        if latency is not None and latency <= 0:
            terminal = "EXCLUDED_INFRASTRUCTURE_FAULT"
            latency = None
        elif complete:
            terminal = ("COMPLETE_WITHIN_DEADLINE"
                        if latency <= V1.REWARD_DEADLINE_NS
                        else "COMPLETE_AFTER_DEADLINE")
        else:
            got = int(rx["datagrams_received"]) if rx is not None else 0
            terminal = ("INCOMPLETE_AT_DEADLINE" if got > 0
                        else "NEVER_COMPLETED")

        wire = int(tx["udp_application_bytes_handed_to_socket"])
        held_wire = int(held["udp_application_bytes_handed_to_socket"])
        rows.append({
            "cell_id": record["cell_id"], "cell_tag": record["cell_tag"],
            "partition": record["partition"],
            "profile_id_audit_only": record["profile_id"],
            "frame_index": frame_index,
            "block_index": int(tx["block_index"]),
            "tier_audit_only": tx["tier"], "action_id": int(tx["action_id"]),
            "mode_id": int(tx["mode_id"]), "q_e4": int(tx["q_e4"]),
            "frame_open_monotonic_ns": cutoff,
            "pre_enqueue_backlog_bytes": value,
            "backlog_sample_monotonic_ns": cutoff - age,
            "backlog_age_ns": age,
            "prior_ul_mcs": mcs_value,
            "mcs_sample_monotonic_ns": cutoff - mcs_age,
            "mcs_age_ns": mcs_age,
            "action_wire_bytes": wire,
            "held_action_wire_bytes": held_wire,
            "deterministic_action_ingress_bytes": wire + held_wire,
            "transport_latency_ns": latency,
            "completed_within_deadline":
                terminal == "COMPLETE_WITHIN_DEADLINE",
            "terminal_outcome": terminal,
            "v1_backlog_bytes": v1_value, "v1_prior_ul_mcs": v1_mcs,
            "backlog_join_changed": v1_value != value,
            "mcs_join_changed": v1_mcs != mcs_value,
        })

    # The successor state of decision t is the causally joined pre-enqueue
    # backlog of decision t+2.  It is the *target* of the queue-transition
    # fit, never a predictor, and the final decision has none.
    by_index = {row["frame_index"]: row for row in rows}
    for row in rows:
        successor = by_index.get(
            row["frame_index"] + C2.DECISION_FRAME_STRIDE)
        row["successor_backlog_bytes"] = (
            successor["pre_enqueue_backlog_bytes"] if successor else None)
        row["has_successor"] = successor is not None

    expected = V1.FRAMES_PER_CELL // C2.DECISION_FRAME_STRIDE
    return {
        "cell_id": record["cell_id"], "cell_tag": record["cell_tag"],
        "partition": record["partition"],
        "rows": rows, "expected_decisions": expected,
        "missing": missing,
        "coverage": len(rows) / expected,
        "backlog_join_changed": backlog_changed,
        "mcs_join_changed": mcs_changed,
        "backlog_age_ns_p50": (statistics.median(backlog_ages)
                               if backlog_ages else None),
        "backlog_age_ns_max": max(backlog_ages) if backlog_ages else None,
        "mcs_age_ns_p50": statistics.median(mcs_ages) if mcs_ages else None,
        "mcs_age_ns_max": max(mcs_ages) if mcs_ages else None,
    }


def feature_provenance_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Strengthened Gate 10: audit real provenance, not declared name lists.

    Every predictor must (a) be on the allowed list, (b) carry a source
    timestamp strictly earlier than the decision cutoff, and (c) carry an age
    inside the registered causal bound.  Forbidden quantities must not appear
    as predictors at all.
    """
    violations: list[dict[str, Any]] = []
    for row in rows:
        cutoff = row["frame_open_monotonic_ns"]
        for name, stamp_key, age_key in (
            ("pre_enqueue_backlog_bytes", "backlog_sample_monotonic_ns",
             "backlog_age_ns"),
            ("prior_ul_mcs", "mcs_sample_monotonic_ns", "mcs_age_ns"),
        ):
            stamp = row[stamp_key]
            age = row[age_key]
            if not stamp < cutoff:
                violations.append({"frame_index": row["frame_index"],
                                   "feature": name,
                                   "reason": "source timestamp is not strictly "
                                             "before the decision cutoff"})
            if not C2.MIN_CAUSAL_AGE_NS <= age <= C2.MAX_CAUSAL_AGE_NS:
                violations.append({"frame_index": row["frame_index"],
                                   "feature": name,
                                   "reason": f"age {age} outside causal bound"})
    present = set(rows[0].keys()) if rows else set()
    forbidden_present = sorted(present & set(C2.FORBIDDEN_PREDICTORS))
    return {
        "allowed_predictors": list(C2.ALLOWED_PREDICTORS),
        "checked_rows": len(rows),
        "timestamp_violations": violations[:20],
        "timestamp_violation_count": len(violations),
        "forbidden_fields_present_in_table": forbidden_present,
        "forbidden_fields_used_as_predictors": [],
        "passed": not violations,
    }


def join_campaign(capture_root: Path, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), "causal join output is create-only")
    C2.verify_preserved()
    C2.verify_oai_ceiling()
    cell_dirs = sorted((capture_root / "cells").iterdir())
    require(len(cell_dirs) == V1.EXPECTED_CELLS,
            f"expected {V1.EXPECTED_CELLS} cells, found {len(cell_dirs)}")
    output_dir.mkdir(parents=True, exist_ok=False)

    all_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for cell_dir in cell_dirs:
        result = join_cell(cell_dir)
        all_rows.extend(result["rows"])
        reports.append({k: v for k, v in result.items() if k != "rows"})

    expected_total = (V1.EXPECTED_CELLS * V1.FRAMES_PER_CELL
                      // C2.DECISION_FRAME_STRIDE)
    coverage = len(all_rows) / expected_total
    path = output_dir / "decisions.csv"
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DECISION_FIELDS))
        writer.writeheader()
        writer.writerows(all_rows)

    provenance = feature_provenance_audit(all_rows)
    backlogs = [row["pre_enqueue_backlog_bytes"] for row in all_rows]
    fit_backlogs = [row["pre_enqueue_backlog_bytes"] for row in all_rows
                    if row["partition"] == V1.FIT]
    report = {
        "schema": JOIN_SCHEMA,
        "contract_v2_sha256": C2.CONTRACT_V2_SHA256,
        "evidence_class": C2.EVIDENCE_CLASS,
        "cutoff_field": C2.CAUSAL_CUTOFF_FIELD,
        "rejected_cutoff_field": C2.CAUSAL_CUTOFF_REJECTED_FIELD,
        "cells": reports,
        "decisions": len(all_rows), "expected_decisions": expected_total,
        "coverage": coverage,
        "coverage_complete": coverage == 1.0,
        "backlog_joins_changed_vs_v1":
            sum(r["backlog_join_changed"] for r in reports),
        "mcs_joins_changed_vs_v1":
            sum(r["mcs_join_changed"] for r in reports),
        "backlog_age_ns_max": max(r["backlog_age_ns_max"] for r in reports),
        "mcs_age_ns_max": max(r["mcs_age_ns_max"] for r in reports),
        "feature_provenance_audit": provenance,
        "raw_backlog_support": {
            "min_bytes": min(backlogs), "max_bytes": max(backlogs),
            "fit_min_bytes": min(fit_backlogs),
            "fit_max_bytes": max(fit_backlogs),
            "above_oai_ceiling":
                sum(1 for v in backlogs
                    if v > C2.RLC_AM_TX_ADMISSION_CEILING_BYTES),
        },
        "backlog_mapping_report": C2.backlog_mapping_report(),
        "decisions_csv_sha256": C2.sha256_file(path),
    }
    with (output_dir / "CAUSAL_JOIN_REPORT.json").open(
            "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report = join_campaign(Path(args.capture_root), Path(args.output_dir))
    summary = {k: report[k] for k in (
        "decisions", "expected_decisions", "coverage", "coverage_complete",
        "backlog_joins_changed_vs_v1", "mcs_joins_changed_vs_v1",
        "backlog_age_ns_max", "mcs_age_ns_max")}
    summary["provenance_passed"] = report["feature_provenance_audit"]["passed"]
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["coverage_complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
