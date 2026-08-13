#!/usr/bin/env python3
"""Analyze DG-A raw artifacts and run the provisional DG-A.1 scale screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .endpoint import frame_onwire_bytes


def atomic_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def p95(values: Iterable[float]) -> float:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    return float(np.percentile(array, 95)) if array.size else float("nan")


def max_starvation_ms(
    completion_ns: Sequence[int], duration_s: float, start_ns: int, end_ns: int
) -> float:
    points = [start_ns, *sorted(int(value) for value in completion_ns), end_ns]
    if len(points) < 2:
        return duration_s * 1000.0
    return max((right - left) / 1e6 for left, right in zip(points[:-1], points[1:]))


def trial_metrics(run_dir: Path, trial: Mapping[str, object], deadlines: Sequence[float]) -> dict:
    trial_dir = run_dir / "runs" / str(trial["id"])
    demand = pd.read_csv(trial_dir / "sender_demands.csv")
    frames = pd.read_csv(trial_dir / "receiver_frames.csv")
    sender = json.loads((trial_dir / "sender_summary.json").read_text())
    trial_start_ns = int(sender["start_raw_ns"])
    trial_end_ns = trial_start_ns + int(float(sender["duration_target_s"]) * 1e9)
    delivered = frames[["ue_id", "frame_id", "complete_raw_ns", "complete_latency_ms", "onwire_bytes"]].copy()
    merged = demand.merge(delivered, on=["ue_id", "frame_id"], how="left", suffixes=("_send", "_recv"))
    duration_s = float(trial["duration_s"])
    per_ue = {}
    for ue_id, group in merged.groupby("ue_id", sort=True):
        complete = group["complete_raw_ns"].notna()
        row = {
            "demand_frames": int(len(group)),
            "complete_frames": int(complete.sum()),
            "complete_ratio": safe_div(float(complete.sum()), len(group)),
            "complete_latency_p95_ms": p95(group.loc[complete, "complete_latency_ms"]),
            "max_starvation_ms": max_starvation_ms(
                group.loc[complete, "complete_raw_ns"].astype("int64").tolist(),
                duration_s,
                trial_start_ns,
                trial_end_ns,
            ),
        }
        for deadline in deadlines:
            delivered_in_time = complete & (group["complete_latency_ms"] <= deadline * 1000.0)
            row[f"deadline_{deadline:.2f}s_fraction"] = safe_div(float(delivered_in_time.sum()), len(group))
        per_ue[str(int(ue_id))] = row
    goodput_mbps = frames["onwire_bytes"].sum() * 8 / max(duration_s, 1e-9) / 1e6
    return {
        "trial_id": trial["id"],
        "controller": trial.get("controller", "open_loop"),
        "demand_trace_sha256": sender["demand_trace_sha256"],
        "aggregate_complete_goodput_mbps": float(goodput_mbps),
        "worst_complete_latency_p95_ms": max(
            (float(row["complete_latency_p95_ms"]) for row in per_ue.values()), default=float("nan")
        ),
        "worst_max_starvation_ms": max(
            (float(row["max_starvation_ms"]) for row in per_ue.values()), default=float("nan")
        ),
        **{
            f"worst_deadline_{deadline:.2f}s_fraction": min(
                (float(row[f"deadline_{deadline:.2f}s_fraction"]) for row in per_ue.values()),
                default=float("nan"),
            )
            for deadline in deadlines
        },
        "per_ue": per_ue,
    }


def pair_effect(greedy: Mapping[str, object], central: Mapping[str, object], config: Mapping[str, object]) -> dict:
    deadlines = [float(value) for value in config["deadline_s"]]
    deadline_lifts = {
        f"{deadline:.2f}": 100.0
        * (
            float(central[f"worst_deadline_{deadline:.2f}s_fraction"])
            - float(greedy[f"worst_deadline_{deadline:.2f}s_fraction"])
        )
        for deadline in deadlines
    }
    greedy_latency = float(greedy["worst_complete_latency_p95_ms"])
    central_latency = float(central["worst_complete_latency_p95_ms"])
    latency_abs = greedy_latency - central_latency
    latency_rel = safe_div(latency_abs, greedy_latency)
    greedy_starvation = float(greedy["worst_max_starvation_ms"])
    central_starvation = float(central["worst_max_starvation_ms"])
    starvation_abs = greedy_starvation - central_starvation
    starvation_rel = safe_div(starvation_abs, greedy_starvation)
    goodput_loss = safe_div(
        float(greedy["aggregate_complete_goodput_mbps"])
        - float(central["aggregate_complete_goodput_mbps"]),
        float(greedy["aggregate_complete_goodput_mbps"]),
    )
    deadline_pass = all(
        value >= float(config["minimum_deadline_lift_pp"]) for value in deadline_lifts.values()
    )
    latency_pass = (
        latency_rel >= float(config["minimum_latency_relative_reduction"])
        and latency_abs >= float(config["minimum_latency_absolute_ms"])
    )
    starvation_pass = (
        starvation_rel >= float(config["minimum_starvation_relative_reduction"])
        and starvation_abs >= float(config["minimum_starvation_absolute_ms"])
    )
    no_goodput_regression = goodput_loss <= float(config["maximum_goodput_loss_fraction"])
    return {
        "greedy_trial": greedy["trial_id"],
        "central_trial": central["trial_id"],
        "demand_hash_match": greedy["demand_trace_sha256"] == central["demand_trace_sha256"],
        "deadline_lift_pp": deadline_lifts,
        "latency_reduction_ms": latency_abs,
        "latency_reduction_fraction": latency_rel,
        "starvation_reduction_ms": starvation_abs,
        "starvation_reduction_fraction": starvation_rel,
        "aggregate_goodput_loss_fraction": goodput_loss,
        "deadline_pass": deadline_pass,
        "latency_pass": latency_pass,
        "starvation_pass": starvation_pass,
        "no_goodput_regression": no_goodput_regression,
        "meaningful_gap": bool(
            (deadline_pass or latency_pass or starvation_pass) and no_goodput_regression
        ),
    }


def scheduler_redistribution(run_dir: Path, mu_hat: float, decision: Mapping[str, object]) -> dict:
    trial_dir = run_dir / "runs" / "A4"
    sender = json.loads((trial_dir / "sender_summary.json").read_text())
    start_ns = int(sender["start_raw_ns"])
    end_ns = start_ns + int(float(sender["duration_target_s"]) * 1e9)
    per_ue_bytes: Dict[int, int] = {0: 0, 1: 0}
    with (trial_dir / "receiver_chunks.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            when = int(row["recv_raw_ns"])
            if start_ns <= when <= end_ns:
                per_ue_bytes[int(row["ue_id"])] += int(row["onwire_bytes"])
    duration = float(sender["duration_target_s"])
    service = {ue: value * 8 / duration / 1e6 for ue, value in per_ue_bytes.items()}
    heavy, light = 0, 1
    aggregate = sum(service.values())
    residual = max(0.0, mu_hat - service[light])
    heavy_residual_fraction = safe_div(service[heavy], residual)
    return {
        "per_ue_service_mbps": {str(key): value for key, value in service.items()},
        "aggregate_service_mbps": aggregate,
        "aggregate_to_mu_fraction": safe_div(aggregate, mu_hat),
        "heavy_residual_fraction": heavy_residual_fraction,
        "redistribution_present": bool(
            aggregate >= float(decision["scheduler_aggregate_floor_fraction"]) * mu_hat
            and heavy_residual_fraction >= float(decision["scheduler_residual_to_heavy_floor_fraction"])
        ),
    }


def service_family(name: str, n: int, mu1: float, mu2: float, cap: float) -> float:
    if name == "constant_ceiling":
        value = mu2
    elif name == "saturating":
        mu_inf = 2.0 * mu2 - mu1
        if mu_inf <= 0:
            return float("nan")
        value = mu_inf + (mu1 - mu_inf) / n
    elif name == "power_law":
        if mu1 <= 0 or mu2 <= 0:
            return float("nan")
        beta = math.log(mu2 / mu1, 2)
        value = mu2 * (n / 2.0) ** beta
    else:
        raise ValueError(name)
    return min(cap, value)


def demand_weights(name: str, n: int) -> List[float]:
    if name in {"equal", "synchronized_burst"}:
        return [1.0 / n] * n
    if name == "hot20_traffic80":
        hot = max(1, round(0.2 * n))
        cold = n - hot
        return [0.8 / hot] * hot + ([0.2 / cold] * cold if cold else [])
    raise ValueError(name)


def allocate_equal_ratio(demand: Sequence[float], budget: float, efficiency: float) -> List[float]:
    total = sum(demand)
    ratio = min(1.0, max(0.0, budget * efficiency) / max(total, 1e-12))
    return [value * ratio for value in demand]


def scale_cell(
    *,
    n: int,
    mu: float,
    demand_name: str,
    payload_bytes: int,
    decision: Mapping[str, object],
    central_efficiency: float,
) -> dict:
    rho = 1.30 if demand_name == "synchronized_burst" else 1.10
    total_demand = rho * mu
    weights = demand_weights(demand_name, n)
    demand = [weight * total_demand for weight in weights]
    local_budget = float(decision.get("c1_pessimism_factor", 0.70)) * mu / n
    local = [min(value, local_budget) for value in demand]
    central = allocate_equal_ratio(demand, 0.70 * mu, central_efficiency)
    onwire = frame_onwire_bytes(payload_bytes, 60000) * 8 / 1e6

    def summarize(allocation: Sequence[float]) -> dict:
        ratios = [safe_div(value, need) for value, need in zip(allocation, demand)]
        intervals = [onwire / max(value, 1e-12) * 1000.0 for value in allocation]
        return {
            "worst_delivery_fraction": min(ratios),
            "worst_inter_delivery_ms": max(intervals),
            "aggregate_admitted_mbps": sum(allocation),
        }

    local_summary = summarize(local)
    central_summary = summarize(central)
    starvation_abs = local_summary["worst_inter_delivery_ms"] - central_summary["worst_inter_delivery_ms"]
    starvation_rel = safe_div(starvation_abs, local_summary["worst_inter_delivery_ms"])
    lift_pp = 100.0 * (
        central_summary["worst_delivery_fraction"] - local_summary["worst_delivery_fraction"]
    )
    meaningful = (
        lift_pp >= float(decision["minimum_deadline_lift_pp"])
        or (
            starvation_rel >= float(decision["minimum_starvation_relative_reduction"])
            and starvation_abs >= float(decision["minimum_starvation_absolute_ms"])
        )
    )
    return {
        "n": n,
        "mu_mbps": mu,
        "demand": demand_name,
        "payload_bytes": payload_bytes,
        "central_efficiency": central_efficiency,
        "local": local_summary,
        "central": central_summary,
        "worst_delivery_lift_pp": lift_pp,
        "starvation_reduction_ms": starvation_abs,
        "starvation_reduction_fraction": starvation_rel,
        "meaningful_gap": meaningful,
    }


def large_n_screen(
    stage: Mapping[str, object], config: Mapping[str, object], redistribution: Mapping[str, object]
) -> dict:
    decision = dict(config["decision"])
    decision["c1_pessimism_factor"] = float(config["c1"]["pessimism_factor"])
    conversions = [float(block["service_conversion"]) for block in stage["blocks"].values()]
    mu1 = float(decision["historical_n1_strong_sched_mbps"]) * float(np.mean(conversions))
    mu2_by_block = {
        str(name): float(block["mu_hat_mbps"]) for name, block in stage["blocks"].items()
    }
    mu2_values = [mu2_by_block[name] for name in sorted(mu2_by_block)]
    cap = float(decision["physical_service_cap_mbps"])
    measured_efficiency = min(1.0, max(0.0, float(redistribution["heavy_residual_fraction"])))
    efficiencies = sorted({1.0, measured_efficiency})
    rows: List[dict] = []
    for block_name in sorted(mu2_by_block):
        mu2 = mu2_by_block[block_name]
        for family in decision["large_n_service_families"]:
            for n in decision["large_n"]:
                mu = service_family(str(family), int(n), mu1, mu2, cap)
                if not math.isfinite(mu) or mu <= 0:
                    rows.append(
                        {
                            "block": block_name,
                            "family": family,
                            "n": n,
                            "valid": False,
                            "reason": "nonpositive_service_family",
                        }
                    )
                    continue
                for demand_name in decision["large_n_demands"]:
                    for payload in (92160, 132301, 409600):
                        for efficiency in efficiencies:
                            rows.append(
                                {
                                    "block": block_name,
                                    "family": family,
                                    "valid": True,
                                    **scale_cell(
                                        n=int(n),
                                        mu=mu,
                                        demand_name=str(demand_name),
                                        payload_bytes=payload,
                                        decision=decision,
                                        central_efficiency=efficiency,
                                    ),
                                }
                            )
    valid_rows = [row for row in rows if row.get("valid")]
    robust_scenarios = []
    keys = {
        (row["n"], row["demand"], row["payload_bytes"], row["central_efficiency"])
        for row in valid_rows
    }
    for key in sorted(keys):
        cells = [
            row
            for row in valid_rows
            if (row["n"], row["demand"], row["payload_bytes"], row["central_efficiency"]) == key
        ]
        expected = len(stage["blocks"]) * len(decision["large_n_service_families"])
        if len(cells) == expected and all(bool(row["meaningful_gap"]) for row in cells):
            robust_scenarios.append(
                {
                    "n": key[0],
                    "demand": key[1],
                    "payload_bytes": key[2],
                    "central_efficiency": key[3],
                    "minimum_delivery_lift_pp": min(row["worst_delivery_lift_pp"] for row in cells),
                    "minimum_starvation_reduction_ms": min(
                        row["starvation_reduction_ms"] for row in cells
                    ),
                }
            )
    return {
        "provenance": "MODEL-BASED provisional N=50/100 screen fitted on N=1 historical + N=2 DG-A; not measured",
        "mu1_mapped_mbps": mu1,
        "mu2_block_mbps": mu2_values,
        "central_efficiency_envelope": efficiencies,
        "rows": rows,
        "robust_scenarios": robust_scenarios,
        "robust_n50_gap": any(int(row["n"]) == 50 for row in robust_scenarios),
    }


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    pairs = summary["raw_pairs"]
    large = summary["large_n_screen"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DG-A + DG-A.1 Decision Summary\n\n")
        handle.write(f"**Decision:** `{summary['decision']}`\n\n")
        handle.write("DG-B was **not launched**. This result requires human review.\n\n")
        handle.write("## Raw N=2 gate\n\n")
        handle.write(f"- Scheduler redistribution: `{summary['scheduler_redistribution']['redistribution_present']}`\n")
        for pair in pairs:
            handle.write(
                f"- {pair['greedy_trial']} vs {pair['central_trial']}: meaningful_gap="
                f"`{pair['meaningful_gap']}`, deadline lifts={pair['deadline_lift_pp']}, "
                f"latency reduction={pair['latency_reduction_ms']:.3f} ms, "
                f"starvation reduction={pair['starvation_reduction_ms']:.3f} ms.\n"
            )
        handle.write("\n## Provisional scale screen\n\n")
        handle.write(f"- {large['provenance']}\n")
        handle.write(f"- Robust N=50 candidate gap: `{large['robust_n50_gap']}`\n")
        handle.write(f"- Robust scenarios: {len(large['robust_scenarios'])}\n")
        handle.write("\nAll skipped/replaced/timeout demand remains in deadline denominators.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = yaml.safe_load(Path(args.config).read_text())
    stage = json.loads((run_dir / "stage_manifest.json").read_text())
    trial_map = {str(row["id"]): row for row in stage["trials"]}
    deadlines = [float(value) for value in config["decision"]["deadline_s"]]
    metrics = {
        trial_id: trial_metrics(run_dir, trial_map[trial_id], deadlines)
        for trial_id in [f"A{index}" for index in range(1, 10)]
    }
    for left, right in (("A6", "A7"), ("A8", "A9")):
        if metrics[left]["demand_trace_sha256"] != metrics[right]["demand_trace_sha256"]:
            raise SystemExit(f"paired demand hash mismatch: {left}/{right}")
    raw_pairs = [
        pair_effect(metrics["A6"], metrics["A7"], config["decision"]),
        pair_effect(metrics["A8"], metrics["A9"], config["decision"]),
    ]
    raw_replicated_gap = all(bool(pair["meaningful_gap"]) for pair in raw_pairs)
    mu_a = float(stage["blocks"]["A"]["mu_hat_mbps"])
    redistribution = scheduler_redistribution(run_dir, mu_a, config["decision"])
    large = large_n_screen(stage, config, redistribution)
    if raw_replicated_gap or large["robust_n50_gap"]:
        decision = "CANDIDATE_GO_DG_B_HUMAN_REVIEW_REQUIRED"
    else:
        decision = "STOP_CHEAP_NO"
    summary = {
        "schema_version": "scenesense.multiue_oai.dg_a.decision.v1",
        "decision": decision,
        "next_stage_launched": False,
        "raw_replicated_gap": raw_replicated_gap,
        "raw_pairs": raw_pairs,
        "scheduler_redistribution": redistribution,
        "trial_metrics": metrics,
        "large_n_screen": large,
    }
    atomic_json(run_dir / "results_summary.json", summary)
    rows = large["rows"]
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "large_n_sensitivity.csv", index=False)
    write_markdown(run_dir / "DG_A_DECISION.md", summary)
    print(json.dumps({"decision": decision, "next_stage_launched": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
