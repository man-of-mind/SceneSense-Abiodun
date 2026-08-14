#!/usr/bin/env python3
"""Corrected, immutable sibling reanalysis for the DG-A.1 scale screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml

from .analyze import (
    pair_effect,
    p95,
    safe_div,
    scheduler_redistribution,
    service_family,
    trial_metrics,
)
from .endpoint import frame_onwire_bytes, staggered_arrival_credits


@dataclass
class SimDemand:
    demand_id: int
    ue_id: int
    arrival_tick: int
    status: str = "scheduled"
    admitted_tick: int | None = None
    completion_s: float | None = None


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_input_hashes(paths: Sequence[Path], root: Path) -> Dict[str, str]:
    """Hash every source artifact consumed by the reanalysis."""
    result: Dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            label = str(resolved.relative_to(root.resolve()))
        except ValueError:
            label = str(resolved)
        result[label] = sha256(resolved)
    return result


def max_min_allocate(demand: Sequence[float], budget: float) -> List[float]:
    """Work-conserving max-min allocation capped by each UE's demand."""
    requested = [max(0.0, float(value)) for value in demand]
    available = max(0.0, float(budget))
    allocation = [0.0] * len(requested)
    active = set(range(len(requested)))
    while active and available > 1e-12:
        share = available / len(active)
        satisfied = [index for index in active if requested[index] <= share + 1e-12]
        if not satisfied:
            for index in active:
                allocation[index] = share
            available = 0.0
            break
        for index in satisfied:
            allocation[index] = requested[index]
            available -= requested[index]
            active.remove(index)
    return allocation


def demand_weights(name: str, n: int) -> List[float]:
    if name in {"equal", "synchronized_burst"}:
        return [1.0 / n] * n
    if name == "hot20_traffic80":
        hot = max(1, round(0.2 * n))
        cold = n - hot
        return [0.8 / hot] * hot + ([0.2 / cold] * cold if cold else [])
    raise ValueError(f"unsupported demand pattern: {name}")


def arrival_blueprint(
    demand_mbps: Sequence[float],
    *,
    onwire_bytes: int,
    tick_s: float,
    minimum_arrivals_per_ue: int,
    demand_seed: int,
    synchronized: bool,
    maximum_demands: int,
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Create exact token-credit arrival ticks without iterating empty ticks."""
    if any(float(rate) <= 0 for rate in demand_mbps):
        raise ValueError("every simulated UE must have positive demand")
    ue_ids = list(range(len(demand_mbps)))
    credits = (
        {ue_id: 0.0 for ue_id in ue_ids}
        if synchronized
        else staggered_arrival_credits(ue_ids, int(demand_seed))
    )
    increments = {
        ue_id: float(rate) * 1e6 / 8.0 / float(onwire_bytes) * float(tick_s)
        for ue_id, rate in enumerate(demand_mbps)
    }

    def tick_for(ue_id: int, sequence: int) -> int:
        raw = (float(sequence) - credits[ue_id]) / increments[ue_id]
        return max(0, int(math.ceil(raw - 1e-12)) - 1)

    end_tick = max(tick_for(ue_id, int(minimum_arrivals_per_ue)) for ue_id in ue_ids)
    blueprint: List[Tuple[int, int, int]] = []
    demand_id = 0
    for ue_id in ue_ids:
        count = int(
            math.floor(credits[ue_id] + (end_tick + 1) * increments[ue_id] + 1e-12)
        )
        for sequence in range(1, count + 1):
            blueprint.append((tick_for(ue_id, sequence), ue_id, demand_id))
            demand_id += 1
    if len(blueprint) > int(maximum_demands):
        raise ValueError(
            f"cell generated {len(blueprint)} demands, above frozen maximum {maximum_demands}"
        )
    blueprint.sort(key=lambda row: (row[0], row[1], row[2]))
    return blueprint, end_tick


def _ticks_until_full(token: float, increment: float, capacity: float) -> int:
    if increment <= 0:
        raise ValueError("token increment must be positive")
    return max(1, int(math.ceil((capacity - token) / increment - 1e-12)))


def simulate_admission(
    blueprint: Sequence[Tuple[int, int, int]],
    *,
    n: int,
    end_tick: int,
    tick_s: float,
    onwire_bytes: int,
    effective_mu_mbps: float,
    pessimism_factor: float,
    controller: str,
) -> List[SimDemand]:
    """Replay newest-pending token admission at exact controller ticks."""
    demands = {
        demand_id: SimDemand(demand_id, ue_id, arrival_tick)
        for arrival_tick, ue_id, demand_id in blueprint
    }
    arrivals: Dict[int, List[int]] = defaultdict(list)
    for arrival_tick, _ue_id, demand_id in blueprint:
        arrivals[int(arrival_tick)].append(int(demand_id))
    arrival_ticks = sorted(arrivals)
    arrival_position = 0
    pending: Dict[int, int | None] = {ue_id: None for ue_id in range(n)}
    capacity = float(onwire_bytes)
    current_tick = -1

    if controller == "decentralized_c1":
        tokens = {ue_id: capacity for ue_id in range(n)}
        increment = (
            float(pessimism_factor)
            * float(effective_mu_mbps)
            / n
            * 1e6
            / 8.0
            * float(tick_s)
        )
    elif controller == "centralized_observable":
        aggregate_token = capacity
        increment = (
            float(pessimism_factor)
            * float(effective_mu_mbps)
            * 1e6
            / 8.0
            * float(tick_s)
        )
    else:
        raise ValueError(f"unsupported controller: {controller}")

    while True:
        next_arrival = (
            arrival_ticks[arrival_position]
            if arrival_position < len(arrival_ticks)
            else math.inf
        )
        next_ready = math.inf
        if controller == "decentralized_c1":
            for ue_id, demand_id in pending.items():
                if demand_id is not None:
                    next_ready = min(
                        next_ready,
                        current_tick + _ticks_until_full(tokens[ue_id], increment, capacity),
                    )
        elif any(value is not None for value in pending.values()):
            next_ready = current_tick + _ticks_until_full(
                aggregate_token, increment, capacity
            )
        next_tick = min(next_arrival, next_ready)
        if not math.isfinite(next_tick) or int(next_tick) > int(end_tick):
            break
        next_tick = int(next_tick)
        elapsed_ticks = next_tick - current_tick
        if elapsed_ticks <= 0:
            raise RuntimeError("admission simulator did not advance")
        if controller == "decentralized_c1":
            for ue_id in tokens:
                tokens[ue_id] = min(capacity, tokens[ue_id] + elapsed_ticks * increment)
        else:
            aggregate_token = min(capacity, aggregate_token + elapsed_ticks * increment)
        current_tick = next_tick

        if next_arrival == next_tick:
            for demand_id in arrivals[next_tick]:
                demand = demands[demand_id]
                old_id = pending[demand.ue_id]
                if old_id is not None:
                    demands[old_id].status = "replaced"
                demand.status = "pending"
                pending[demand.ue_id] = demand_id
            arrival_position += 1

        if controller == "decentralized_c1":
            for ue_id in range(n):
                demand_id = pending[ue_id]
                if demand_id is not None and tokens[ue_id] >= capacity - 1e-9:
                    demands[demand_id].status = "admitted"
                    demands[demand_id].admitted_tick = current_tick
                    tokens[ue_id] -= capacity
                    pending[ue_id] = None
        else:
            choices = [demands[value] for value in pending.values() if value is not None]
            if choices and aggregate_token >= capacity - 1e-9:
                chosen = min(choices, key=lambda row: (row.arrival_tick, row.ue_id))
                chosen.status = "admitted"
                chosen.admitted_tick = current_tick
                aggregate_token -= capacity
                pending[chosen.ue_id] = None

    for demand_id in pending.values():
        if demand_id is not None:
            demands[demand_id].status = "skipped_end"
    return [demands[index] for index in sorted(demands)]


def simulate_max_min_service(
    demands: Sequence[SimDemand],
    *,
    n: int,
    tick_s: float,
    onwire_bytes: int,
    service_mbps: float,
) -> None:
    """Drain admitted per-UE FIFO queues with fluid max-min service."""
    capacity_bps = float(service_mbps) * 1e6
    if capacity_bps <= 0:
        raise ValueError("service rate must be positive")
    by_id = {row.demand_id: row for row in demands}
    admitted = sorted(
        (row for row in demands if row.admitted_tick is not None),
        key=lambda row: (int(row.admitted_tick or 0), row.ue_id, row.demand_id),
    )
    queues = {ue_id: deque() for ue_id in range(n)}
    remaining: Dict[int, float] = {}
    now = 0.0

    def active_ues() -> List[int]:
        return [ue_id for ue_id, queue in queues.items() if queue]

    def advance(target: float | None) -> None:
        nonlocal now
        while True:
            active = active_ues()
            if not active:
                if target is not None:
                    now = max(now, target)
                return
            per_ue_bps = capacity_bps / len(active)
            next_completion = min(
                remaining[queues[ue_id][0]] / per_ue_bps for ue_id in active
            )
            if target is not None and now + next_completion > target + 1e-12:
                elapsed = max(0.0, target - now)
                for ue_id in active:
                    remaining[queues[ue_id][0]] -= per_ue_bps * elapsed
                now = target
                return
            elapsed = max(0.0, next_completion)
            for ue_id in active:
                remaining[queues[ue_id][0]] -= per_ue_bps * elapsed
            now += elapsed
            completed = [
                ue_id
                for ue_id in active
                if remaining[queues[ue_id][0]] <= 1e-5
            ]
            if not completed:
                raise RuntimeError("max-min service failed to complete a head frame")
            for ue_id in completed:
                demand_id = queues[ue_id].popleft()
                remaining.pop(demand_id, None)
                by_id[demand_id].completion_s = now

    position = 0
    while position < len(admitted):
        tick = int(admitted[position].admitted_tick or 0)
        when = tick * float(tick_s)
        advance(when)
        while position < len(admitted) and int(admitted[position].admitted_tick or 0) == tick:
            demand = admitted[position]
            queues[demand.ue_id].append(demand.demand_id)
            remaining[demand.demand_id] = float(onwire_bytes) * 8.0
            position += 1
    advance(None)


def simulated_trial_metrics(
    demands: Sequence[SimDemand],
    *,
    controller: str,
    tick_s: float,
    end_tick: int,
    onwire_bytes: int,
    deadlines: Sequence[float],
) -> dict:
    horizon_s = (int(end_tick) + 1) * float(tick_s)
    per_ue: Dict[str, dict] = {}
    completed_total = 0
    for ue_id in sorted({row.ue_id for row in demands}):
        rows = [row for row in demands if row.ue_id == ue_id]
        completed = [row for row in rows if row.completion_s is not None]
        completed_total += len(completed)
        latencies_ms = [
            (float(row.completion_s) - row.arrival_tick * float(tick_s)) * 1000.0
            for row in completed
        ]
        in_horizon = sorted(
            float(row.completion_s)
            for row in completed
            if float(row.completion_s) <= horizon_s + 1e-12
        )
        points = [0.0, *in_horizon, horizon_s]
        starvation_ms = max(
            (right - left) * 1000.0 for left, right in zip(points[:-1], points[1:])
        )
        metrics = {
            "demand_frames": len(rows),
            "admitted_frames": sum(row.admitted_tick is not None for row in rows),
            "replaced_frames": sum(row.status == "replaced" for row in rows),
            "skipped_end_frames": sum(row.status == "skipped_end" for row in rows),
            "complete_frames": len(completed),
            "complete_ratio": safe_div(len(completed), len(rows)),
            "complete_latency_p95_ms": p95(latencies_ms),
            "max_starvation_ms": starvation_ms,
        }
        for deadline in deadlines:
            within = sum(
                (float(row.completion_s) - row.arrival_tick * float(tick_s)) <= float(deadline) + 1e-12
                for row in completed
            )
            metrics[f"deadline_{deadline:.2f}s_fraction"] = safe_div(within, len(rows))
        per_ue[str(ue_id)] = metrics

    finite_latencies = [
        float(row["complete_latency_p95_ms"])
        for row in per_ue.values()
        if math.isfinite(float(row["complete_latency_p95_ms"]))
    ]
    return {
        "trial_id": f"sim_{controller}",
        "controller": controller,
        "demand_trace_sha256": "shared_simulated_blueprint",
        "aggregate_complete_goodput_mbps": (
            completed_total * int(onwire_bytes) * 8.0 / max(horizon_s, 1e-12) / 1e6
        ),
        "worst_complete_latency_p95_ms": max(finite_latencies, default=float("nan")),
        "worst_max_starvation_ms": max(
            (float(row["max_starvation_ms"]) for row in per_ue.values()),
            default=float("nan"),
        ),
        **{
            f"worst_deadline_{deadline:.2f}s_fraction": min(
                float(row[f"deadline_{deadline:.2f}s_fraction"])
                for row in per_ue.values()
            )
            for deadline in deadlines
        },
        "horizon_s": horizon_s,
        "per_ue": per_ue,
    }


def pareto_review(
    local: Mapping[str, object],
    central: Mapping[str, object],
    *,
    deadlines: Sequence[float],
    config: Mapping[str, object],
) -> dict:
    deadline_regressions = {
        f"{deadline:.2f}": 100.0
        * max(
            0.0,
            float(local[f"worst_deadline_{deadline:.2f}s_fraction"])
            - float(central[f"worst_deadline_{deadline:.2f}s_fraction"]),
        )
        for deadline in deadlines
    }
    latency_regression = safe_div(
        float(central["worst_complete_latency_p95_ms"])
        - float(local["worst_complete_latency_p95_ms"]),
        float(local["worst_complete_latency_p95_ms"]),
    )
    starvation_regression = safe_div(
        float(central["worst_max_starvation_ms"])
        - float(local["worst_max_starvation_ms"]),
        float(local["worst_max_starvation_ms"]),
    )
    goodput_loss = safe_div(
        float(local["aggregate_complete_goodput_mbps"])
        - float(central["aggregate_complete_goodput_mbps"]),
        float(local["aggregate_complete_goodput_mbps"]),
    )
    passed = (
        all(
            value <= float(config["maximum_deadline_regression_pp"]) + 1e-12
            for value in deadline_regressions.values()
        )
        and latency_regression <= float(config["maximum_latency_regression_fraction"])
        and starvation_regression <= float(config["maximum_starvation_regression_fraction"])
        and goodput_loss <= float(config["maximum_goodput_loss_fraction"])
    )
    return {
        "pass": bool(passed),
        "deadline_regression_pp": deadline_regressions,
        "latency_regression_fraction": latency_regression,
        "starvation_regression_fraction": starvation_regression,
        "goodput_loss_fraction": goodput_loss,
        "criterion_status": "post_registration_conservative_review",
    }


def allocation_audit(
    demand: Sequence[float], local: Sequence[float], central: Sequence[float]
) -> dict:
    ratios_local = [safe_div(value, need) for value, need in zip(local, demand)]
    ratios_central = [safe_div(value, need) for value, need in zip(central, demand)]
    return {
        "aggregate_demand_mbps": sum(demand),
        "aggregate_local_mbps": sum(local),
        "aggregate_central_mbps": sum(central),
        "worst_local_delivery_fraction": min(ratios_local),
        "worst_central_delivery_fraction": min(ratios_central),
        "central_throttles_any_below_local": any(
            central_value + 1e-12 < local_value
            for local_value, central_value in zip(local, central)
        ),
    }


def simulate_cell(
    *,
    n: int,
    mu_mbps: float,
    demand_pattern: str,
    payload_bytes: int,
    envelope_name: str,
    envelope_efficiency: float,
    model: Mapping[str, object],
) -> dict:
    rho = (
        float(model["synchronized_rho"])
        if demand_pattern == "synchronized_burst"
        else float(model["steady_rho"])
    )
    weights = demand_weights(demand_pattern, n)
    demand_rates = [weight * rho * float(mu_mbps) for weight in weights]
    pessimism = float(model["controller_model"]["pessimism_factor"])
    effective_mu = float(mu_mbps) * float(envelope_efficiency)
    local_rate = pessimism * effective_mu / n
    fluid_local = [min(value, local_rate) for value in demand_rates]
    fluid_central = max_min_allocate(demand_rates, pessimism * effective_mu)
    onwire_bytes = frame_onwire_bytes(int(payload_bytes), int(model["chunk_bytes"]))
    blueprint, end_tick = arrival_blueprint(
        demand_rates,
        onwire_bytes=onwire_bytes,
        tick_s=float(model["tick_s"]),
        minimum_arrivals_per_ue=int(model["minimum_arrivals_per_ue"]),
        demand_seed=int(model["demand_seed"]) + int(n),
        synchronized=demand_pattern == "synchronized_burst",
        maximum_demands=int(model["maximum_generated_demands_per_cell"]),
    )
    metrics = {}
    for controller in ("decentralized_c1", "centralized_observable"):
        demands = simulate_admission(
            blueprint,
            n=n,
            end_tick=end_tick,
            tick_s=float(model["tick_s"]),
            onwire_bytes=onwire_bytes,
            effective_mu_mbps=effective_mu,
            pessimism_factor=pessimism,
            controller=controller,
        )
        simulate_max_min_service(
            demands,
            n=n,
            tick_s=float(model["tick_s"]),
            onwire_bytes=onwire_bytes,
            service_mbps=effective_mu,
        )
        metrics[controller] = simulated_trial_metrics(
            demands,
            controller=controller,
            tick_s=float(model["tick_s"]),
            end_tick=end_tick,
            onwire_bytes=onwire_bytes,
            deadlines=[float(value) for value in model["decision"]["deadline_s"]],
        )
    effect = pair_effect(
        metrics["decentralized_c1"],
        metrics["centralized_observable"],
        model["decision"],
    )
    pareto = pareto_review(
        metrics["decentralized_c1"],
        metrics["centralized_observable"],
        deadlines=[float(value) for value in model["decision"]["deadline_s"]],
        config=model["pareto_review"],
    )
    return {
        "n": int(n),
        "mu_mbps": float(mu_mbps),
        "effective_mu_mbps": effective_mu,
        "demand": demand_pattern,
        "payload_bytes": int(payload_bytes),
        "onwire_bytes": int(onwire_bytes),
        "allocation_envelope": envelope_name,
        "allocation_efficiency": float(envelope_efficiency),
        "generated_demands": len(blueprint),
        "allocation_audit": allocation_audit(demand_rates, fluid_local, fluid_central),
        "decentralized": metrics["decentralized_c1"],
        "centralized": metrics["centralized_observable"],
        "effect": effect,
        "pareto_review": pareto,
        "candidate_cell": bool(effect["meaningful_gap"] and pareto["pass"]),
    }


def validate_model_config(model: Mapping[str, object], source: Mapping[str, object]) -> None:
    if model.get("schema_version") != "scenesense.multiue_oai.dg_a_reanalysis.v2":
        raise ValueError("wrong reanalysis schema")
    exact = {
        "tick_s": float(source["transport"]["tick_s"]),
        "chunk_bytes": int(source["transport"]["chunk_bytes"]),
        "large_n": list(source["decision"]["large_n"]),
        "service_families": list(source["decision"]["large_n_service_families"]),
        "demand_patterns": list(source["decision"]["large_n_demands"]),
        "physical_service_cap_mbps": float(source["decision"]["physical_service_cap_mbps"]),
        "historical_n1_strong_sched_mbps": float(
            source["decision"]["historical_n1_strong_sched_mbps"]
        ),
    }
    for key, expected in exact.items():
        if model.get(key) != expected:
            raise ValueError(f"reanalysis {key} differs from source decision config")
    if float(model["controller_model"]["pessimism_factor"]) != float(
        source["c1"]["pessimism_factor"]
    ):
        raise ValueError("reanalysis pessimism factor differs from source C1 contract")
    for key in (
        "deadline_s",
        "minimum_deadline_lift_pp",
        "minimum_latency_relative_reduction",
        "minimum_latency_absolute_ms",
        "minimum_starvation_relative_reduction",
        "minimum_starvation_absolute_ms",
        "maximum_goodput_loss_fraction",
    ):
        if model["decision"].get(key) != source["decision"].get(key):
            raise ValueError(f"reanalysis decision.{key} differs from registered gate")
    envelopes = [row["name"] for row in model["allocation_envelopes"]]
    if envelopes != ["ideal_max_min", "measured_residual_max_min"]:
        raise ValueError("reanalysis must preserve both registered allocation envelopes")
    if int(model["minimum_arrivals_per_ue"]) < 2:
        raise ValueError("minimum arrivals per UE is too small for a screen")


def corrected_large_n_screen(
    stage: Mapping[str, object],
    source_config: Mapping[str, object],
    model: Mapping[str, object],
    redistribution: Mapping[str, object],
) -> dict:
    conversions = [float(block["service_conversion"]) for block in stage["blocks"].values()]
    mu1 = float(model["historical_n1_strong_sched_mbps"]) * sum(conversions) / len(conversions)
    mu2_by_block = {
        str(name): float(block["mu_hat_mbps"]) for name, block in stage["blocks"].items()
    }
    measured_efficiency = min(
        1.0, max(0.0, float(redistribution["heavy_residual_fraction"]))
    )
    envelope_efficiencies = {
        "ideal_max_min": 1.0,
        "measured_residual_max_min": measured_efficiency,
    }
    rows: List[dict] = []
    for block_name in sorted(mu2_by_block):
        mu2 = mu2_by_block[block_name]
        for family in model["service_families"]:
            for n in model["large_n"]:
                mu = service_family(
                    str(family),
                    int(n),
                    mu1,
                    mu2,
                    float(model["physical_service_cap_mbps"]),
                )
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
                for demand_pattern in model["demand_patterns"]:
                    for payload_bytes in model["payload_bytes"]:
                        for envelope_name, efficiency in envelope_efficiencies.items():
                            rows.append(
                                {
                                    "block": block_name,
                                    "family": family,
                                    "valid": True,
                                    **simulate_cell(
                                        n=int(n),
                                        mu_mbps=mu,
                                        demand_pattern=str(demand_pattern),
                                        payload_bytes=int(payload_bytes),
                                        envelope_name=envelope_name,
                                        envelope_efficiency=efficiency,
                                        model=model,
                                    ),
                                }
                            )
    valid_rows = [row for row in rows if row.get("valid")]
    expected = (
        len(stage["blocks"])
        * len(model["service_families"])
        * len(model["allocation_envelopes"])
    )
    robust_scenarios = []
    keys = {(row["n"], row["demand"], row["payload_bytes"]) for row in valid_rows}
    for key in sorted(keys):
        cells = [
            row
            for row in valid_rows
            if (row["n"], row["demand"], row["payload_bytes"]) == key
        ]
        if len(cells) == expected and all(bool(row["candidate_cell"]) for row in cells):
            deadline_keys = [f"{float(value):.2f}" for value in model["decision"]["deadline_s"]]
            robust_scenarios.append(
                {
                    "n": key[0],
                    "demand": key[1],
                    "payload_bytes": key[2],
                    "minimum_deadline_lift_pp": {
                        deadline: min(
                            float(row["effect"]["deadline_lift_pp"][deadline]) for row in cells
                        )
                        for deadline in deadline_keys
                    },
                    "minimum_latency_reduction_ms": min(
                        float(row["effect"]["latency_reduction_ms"]) for row in cells
                    ),
                    "minimum_starvation_reduction_ms": min(
                        float(row["effect"]["starvation_reduction_ms"]) for row in cells
                    ),
                    "cells": len(cells),
                }
            )
    return {
        "schema_version": "scenesense.multiue_oai.dg_a1_queue_screen.v2",
        "provenance": (
            "MODEL-BASED corrected queue/deadline screen fitted on N=1 historical + "
            "N=2 DG-A; N=50/100 are not measured"
        ),
        "mu1_mapped_mbps": mu1,
        "mu2_block_mbps": [mu2_by_block[name] for name in sorted(mu2_by_block)],
        "allocation_envelope_efficiencies": envelope_efficiencies,
        "rows": rows,
        "robust_scenarios": robust_scenarios,
        "robust_n50_gap": any(int(row["n"]) == 50 for row in robust_scenarios),
    }


def flatten_rows(rows: Iterable[Mapping[str, object]]) -> List[dict]:
    flattened = []
    for row in rows:
        base = {
            key: value
            for key, value in row.items()
            if key not in {"allocation_audit", "decentralized", "centralized", "effect", "pareto_review"}
        }
        if not row.get("valid"):
            flattened.append(base)
            continue
        audit = row["allocation_audit"]
        local = row["decentralized"]
        central = row["centralized"]
        effect = row["effect"]
        pareto = row["pareto_review"]
        flattened.append(
            {
                **base,
                **{f"allocation_{key}": value for key, value in audit.items()},
                "local_goodput_mbps": local["aggregate_complete_goodput_mbps"],
                "central_goodput_mbps": central["aggregate_complete_goodput_mbps"],
                "local_worst_latency_p95_ms": local["worst_complete_latency_p95_ms"],
                "central_worst_latency_p95_ms": central["worst_complete_latency_p95_ms"],
                "local_worst_starvation_ms": local["worst_max_starvation_ms"],
                "central_worst_starvation_ms": central["worst_max_starvation_ms"],
                **{
                    f"deadline_lift_{deadline}_pp": value
                    for deadline, value in effect["deadline_lift_pp"].items()
                },
                "latency_reduction_ms": effect["latency_reduction_ms"],
                "starvation_reduction_ms": effect["starvation_reduction_ms"],
                "registered_meaningful_gap": effect["meaningful_gap"],
                "pareto_safe": pareto["pass"],
            }
        )
    return flattened


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    screen = summary["large_n_screen"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DG-A corrected sibling reanalysis v2\n\n")
        handle.write(f"**Decision:** `{summary['decision']}`\n\n")
        handle.write("The source experiment was not modified. DG-B and all later stages were not launched.\n\n")
        handle.write("## Measured N=2\n\n")
        handle.write(f"- Replicated meaningful gap: `{summary['raw_replicated_gap']}`\n")
        handle.write(
            f"- Scheduler redistribution present: "
            f"`{summary['scheduler_redistribution']['redistribution_present']}`\n"
        )
        handle.write("\n## Corrected provisional N=50/100 screen\n\n")
        handle.write(f"- {screen['provenance']}\n")
        handle.write(f"- Robust N=50 gap: `{screen['robust_n50_gap']}`\n")
        handle.write(f"- Robust scenarios: {len(screen['robust_scenarios'])}\n")
        handle.write(
            "- Required across both blocks, all service families, both named allocation envelopes, "
            "and the post-registration Pareto review.\n"
        )
        handle.write("\nAll scheduled arrivals remain in deadline denominators.\n")


def artifact_manifest(output_dir: Path) -> dict:
    names = (
        "source_provenance.json",
        "resolved_model_config.yaml",
        "results_summary.json",
        "DG_A_REANALYSIS_DECISION.md",
        "large_n_sensitivity.csv",
    )
    return {
        "schema_version": "scenesense.multiue_oai.dg_a_reanalysis.artifacts.v2",
        "artifacts": {
            name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)}
            for name in names
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="immutable completed DG-A source")
    parser.add_argument("--config", required=True, help="source DG-A config")
    parser.add_argument("--model-config", required=True, help="frozen v2 reanalysis config")
    parser.add_argument("--output-dir", required=True, help="new sibling output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    source_config_path = Path(args.config).resolve()
    model_config_path = Path(args.model_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise SystemExit("output directory must be a new sibling, never the source or its child")
    if output_dir.exists():
        raise SystemExit(f"immutable output already exists: {output_dir}")
    completed = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    if completed.get("status") != "DG_A_COMPLETE_HUMAN_REVIEW_REQUIRED":
        raise SystemExit("source is not a completed DG-A human-review artifact")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    model = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    validate_model_config(model, source_config)
    stage = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))

    trial_map = {str(row["id"]): row for row in stage["trials"]}
    trial_ids = [f"A{index}" for index in range(1, 10)]
    consumed_source_paths = [
        run_dir / "COMPLETED.json",
        run_dir / "stage_manifest.json",
        run_dir / "results_summary.json",
        source_config_path,
        model_config_path,
        Path(__file__).resolve(),
    ]
    for trial_id in trial_ids:
        trial_dir = run_dir / "runs" / trial_id
        consumed_source_paths.extend(
            [
                trial_dir / "sender_demands.csv",
                trial_dir / "receiver_frames.csv",
                trial_dir / "sender_summary.json",
            ]
        )
    consumed_source_paths.append(run_dir / "runs" / "A4" / "receiver_chunks.csv")
    before_hashes = source_input_hashes(consumed_source_paths, run_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "resolved_model_config.yaml").write_text(
        yaml.safe_dump(model, sort_keys=False), encoding="utf-8"
    )
    atomic_json(
        output_dir / "source_provenance.json",
        {
            "schema_version": "scenesense.multiue_oai.dg_a_reanalysis.source.v2",
            "source_run": str(run_dir),
            "source_completed_sha256": sha256(run_dir / "COMPLETED.json"),
            "source_stage_manifest_sha256": sha256(run_dir / "stage_manifest.json"),
            "source_results_summary_sha256": sha256(run_dir / "results_summary.json"),
            "source_config": str(source_config_path),
            "source_config_sha256": sha256(source_config_path),
            "model_config": str(model_config_path),
            "model_config_sha256": sha256(model_config_path),
            "source_mutated": False,
            "consumed_input_sha256_before": before_hashes,
        },
    )

    deadlines = [float(value) for value in source_config["decision"]["deadline_s"]]
    metrics = {
        trial_id: trial_metrics(run_dir, trial_map[trial_id], deadlines)
        for trial_id in trial_ids
    }
    for left, right in (("A6", "A7"), ("A8", "A9")):
        if metrics[left]["demand_trace_sha256"] != metrics[right]["demand_trace_sha256"]:
            raise SystemExit(f"paired demand hash mismatch: {left}/{right}")
    raw_pairs = [
        pair_effect(metrics["A6"], metrics["A7"], source_config["decision"]),
        pair_effect(metrics["A8"], metrics["A9"], source_config["decision"]),
    ]
    raw_replicated_gap = all(bool(row["meaningful_gap"]) for row in raw_pairs)
    redistribution = scheduler_redistribution(
        run_dir, float(stage["blocks"]["A"]["mu_hat_mbps"]), source_config["decision"]
    )
    screen = corrected_large_n_screen(stage, source_config, model, redistribution)
    decision = (
        "CANDIDATE_GO_DG_B_HUMAN_REVIEW_REQUIRED"
        if raw_replicated_gap or screen["robust_n50_gap"]
        else "STOP_CHEAP_NO"
    )
    summary = {
        "schema_version": "scenesense.multiue_oai.dg_a.decision.v2",
        "decision": decision,
        "supersedes_source_decision_for_scientific_use": True,
        "source_decision": completed.get("decision"),
        "source_run": str(run_dir),
        "source_mutated": False,
        "next_stage_launched": False,
        "raw_replicated_gap": raw_replicated_gap,
        "raw_pairs": raw_pairs,
        "scheduler_redistribution": redistribution,
        "trial_metrics": metrics,
        "large_n_screen": screen,
    }
    atomic_json(output_dir / "results_summary.json", summary)
    pd.DataFrame(flatten_rows(screen["rows"])).to_csv(
        output_dir / "large_n_sensitivity.csv", index=False
    )
    write_markdown(output_dir / "DG_A_REANALYSIS_DECISION.md", summary)
    after_hashes = source_input_hashes(consumed_source_paths, run_dir)
    if after_hashes != before_hashes:
        raise RuntimeError("a consumed source input changed during immutable reanalysis")
    provenance_path = output_dir / "source_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["consumed_input_sha256_after"] = after_hashes
    provenance["source_hashes_verified_unchanged"] = True
    atomic_json(provenance_path, provenance)
    atomic_json(output_dir / "artifact_manifest.json", artifact_manifest(output_dir))
    atomic_json(
        output_dir / "COMPLETED.json",
        {
            "schema_version": "scenesense.multiue_oai.dg_a_reanalysis.completion.v2",
            "status": "DG_A_REANALYSIS_COMPLETE_HUMAN_REVIEW_REQUIRED",
            "decision": decision,
            "source_mutated": False,
            "source_hashes_verified_unchanged": True,
            "oai_started": False,
            "next_stage_launched": False,
        },
    )
    print(json.dumps({"decision": decision, "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
