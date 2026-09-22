"""Exact offline SPLIT-action oracles on the frozen fit-validation panel.

This analysis is deliberately narrower than training or deployment.  It
enumerates every *executable* ``(mode_id, q_e4)`` in the registered modeled-
smoke support for each of the 340 frozen fit-validation contexts.  Although
the actor requests a continuous q, the wire contract first rounds it to the
integer ``q_e4``.  Consequently these inclusive integer intervals are the
complete quotient action space: there are no additional executable actions
between adjacent q_e4 values.

The empirical same-frame surface and network surrogate remain exactly the
registered models.  Vector evaluation only accelerates their piecewise-linear
formulas; every reported oracle winner is re-evaluated through the scalar,
fail-closed public query/session path.  Any support hole, non-finite value, or
scalar/vector discrepancy aborts the run instead of silently approximating.

P95 and P99 below are post-hoc conditional retained-survivor latency proxies.
They are diagnostic counterfactual constraints, not sampled losses, timeout
probabilities, or observations from a live/online system.  The panel is
reward-held fit validation, not untouched final/generalization evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .action_contract import default_contract
from .empirical_contextual_baseline_runner import REGISTERED_BASELINE_CONFIG
from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    PILOT_UTILITY_SPEC,
    PILOT_UTILITY_SPEC_SHA256,
    fixed_stage_latency_ms,
    require_supported_action,
)
from .empirical_contextual_fit_validation_evaluator import (
    EVALUATION_PROVENANCE_LABEL,
    EVALUATION_SCOPE_DISCLOSURE,
    FitValidationActorEvaluatorV1,
)
from .empirical_contextual_fit_validation_panel import (
    REGISTERED_FIT_VALIDATION_PANEL_SHA256,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .offline_quality_grid.contract import Q_E4_GRID
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "BUDGETS_MS",
    "CONDITIONAL_PERCENTILES",
    "EXACT_ACTION_COUNT_PER_SCENE",
    "LEARNED_POLICY_CSV_SHA256",
    "OracleAuditError",
    "OracleOutcome",
    "choose_registered_random_action",
    "enumerate_supported_actions",
    "run_exact_split_oracle_audit",
]


SCHEMA = "splitfusion.frozen_fit_validation_exact_split_oracle.v1"
RANDOM_RULE_SCHEMA = "splitfusion.registered_seed_uniform_supported_action.v1"
BUDGETS_MS: Tuple[float, ...] = (180.0, 200.0, 220.0, 250.0)
CONDITIONAL_PERCENTILES: Tuple[str, ...] = ("p50", "p95", "p99")
CURRENT_LEARNED_UPDATE = 5000
LEARNED_POLICY_CSV_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_three_seed_checkpoints_v1/fit_validation_per_context.csv"
)
LEARNED_POLICY_MANIFEST_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_fit_validation_v1/"
    "20260921_three_seed_checkpoints_v1/manifest.json"
)
LEARNED_POLICY_CSV_SHA256 = (
    "70c3398c86b54a891dfe736257a9356f6acaf734708695b708be89e0dbdd74be"
)
LEARNED_POLICY_MANIFEST_SHA256 = (
    "4cf7d86229beafafbc9e5555a74c554c77f84245386b823eb282f6841488094b"
)

EXACT_ACTION_COUNT_PER_SCENE = sum(
    upper - lower + 1 for lower, upper in MODELED_SMOKE_MODE_Q_E4_BOUNDS
)
EXPECTED_EXACT_ACTION_COUNT_PER_SCENE = 52_240
SCALAR_VECTOR_ABS_TOLERANCE = 2e-9
CONDITIONAL_FEASIBILITY_SEMANTICS = (
    "MODELED_CONDITIONAL_RETAINED_SURVIVOR_LATENCY_PROXY_LE_BUDGET_"
    "NOT_UNCONDITIONAL_SERVICE_SUCCESS"
)


class OracleAuditError(RuntimeError):
    """The exact audit cannot continue without weakening a frozen contract."""


@dataclass(frozen=True, slots=True)
class OracleOutcome:
    mode_id: int
    q_e4: int
    q_perc: float
    total_transmitted_bytes: float
    datagram_count: int
    p_edge_admission_given_sent: float
    latency_proxy_p50_ms: float
    latency_proxy_p95_ms: float
    latency_proxy_p99_ms: float
    reward: float

    def latency(self, percentile: str) -> float:
        if percentile not in CONDITIONAL_PERCENTILES:
            raise OracleAuditError(f"unknown percentile {percentile!r}")
        return float(getattr(self, f"latency_proxy_{percentile}_ms"))

    def to_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_supported_actions() -> Tuple[Tuple[int, int], ...]:
    """Return every wire-distinct action in canonical mode/q order."""

    actions = tuple(
        (mode_id, q_e4)
        for mode_id, (lower, upper) in enumerate(
            MODELED_SMOKE_MODE_Q_E4_BOUNDS
        )
        for q_e4 in range(lower, upper + 1)
    )
    if len(actions) != EXPECTED_EXACT_ACTION_COUNT_PER_SCENE:
        raise OracleAuditError("registered action-support cardinality drift")
    if len(actions) != len(set(actions)):
        raise OracleAuditError("registered action support contains duplicates")
    for mode_id, q_e4 in actions:
        require_supported_action(mode_id, q_e4)
    return actions


def _uniform_ticket(document: Mapping[str, Any], population: int) -> int:
    """Identity-derived, rejection-sampled uniform integer (no modulo bias)."""

    if type(population) is not int or population <= 0:
        raise OracleAuditError("random-action population must be positive")
    limit = (1 << 256) - ((1 << 256) % population)
    counter = 0
    while True:
        payload = dict(document)
        payload["rejection_counter"] = counter
        value = int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest(), "big")
        if value < limit:
            return value % population
        counter += 1


def choose_registered_random_action(*, seed: int, panel_index: int) -> Tuple[int, int]:
    """Select uniformly from all supported pairs using a registered baseline seed."""

    if type(seed) is not int or seed not in REGISTERED_BASELINE_CONFIG.seeds:
        raise OracleAuditError("random baseline seed is not registered")
    if type(panel_index) is not int or not 0 <= panel_index < 340:
        raise OracleAuditError("panel_index must be an exact integer in [0, 339]")
    ticket = _uniform_ticket(
        {
            "panel_index": panel_index,
            "panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
            "rule_schema": RANDOM_RULE_SCHEMA,
            "seed": seed,
        },
        EXACT_ACTION_COUNT_PER_SCENE,
    )
    for mode_id, (lower, upper) in enumerate(MODELED_SMOKE_MODE_Q_E4_BOUNDS):
        width = upper - lower + 1
        if ticket < width:
            action = (mode_id, lower + ticket)
            require_supported_action(*action)
            return action
        ticket -= width
    raise AssertionError("uniform action ticket escaped the support partition")


def _curve_vector(curve: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vector equivalent of the registered ``_MonotoneCurve.predict``."""

    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise OracleAuditError("curve input must be a finite one-dimensional array")
    if np.any(x < curve.raw_x_min) or np.any(x > curve.raw_x_max):
        raise OracleAuditError("vector curve evaluation would extrapolate")
    centers = np.asarray([block.x_center for block in curve.blocks], dtype=np.float64)
    values = np.asarray([block.value for block in curve.blocks], dtype=np.float64)
    weights = np.asarray([block.weight for block in curve.blocks], dtype=np.float64)
    if (
        len(centers) == 0
        or not np.all(np.isfinite(centers))
        or not np.all(np.diff(centers) > 0.0)
    ):
        raise OracleAuditError("curve block centers are not finite and strictly ordered")

    result = np.empty_like(x)
    support = np.empty_like(x)
    low = x <= centers[0]
    high = x >= centers[-1]
    interior = ~(low | high)
    result[low] = values[0]
    support[low] = weights[0]
    result[high] = values[-1]
    support[high] = weights[-1]
    if np.any(interior):
        xi = x[interior]
        right = np.searchsorted(centers, xi, side="left")
        left = right - 1
        fraction = (xi - centers[left]) / (centers[right] - centers[left])
        result[interior] = values[left] + fraction * (
            values[right] - values[left]
        )
        support[interior] = np.minimum(weights[left], weights[right])
    if not np.all(np.isfinite(result)) or not np.all(support > 0.0):
        raise OracleAuditError("curve vector evaluation produced invalid values")
    return result, support


def _local_latency_support_vector(model: Any, x: np.ndarray) -> np.ndarray:
    knots_x = np.asarray([pair[0] for pair in model.latency_support_knots], dtype=np.float64)
    knots_n = np.asarray([pair[1] for pair in model.latency_support_knots], dtype=np.float64)
    if len(knots_x) == 0 or not np.all(np.diff(knots_x) > 0.0):
        # Duplicate coordinates make the scalar equality/interval convention
        # order-sensitive; refuse instead of inventing a vector convention.
        raise OracleAuditError("latency support knots are not strictly ordered")
    if np.any(x < knots_x[0]) or np.any(x > knots_x[-1]):
        raise OracleAuditError("latency support query escaped its raw envelope")
    right = np.searchsorted(knots_x, x, side="left")
    result = np.empty_like(x)
    exact = (right < len(knots_x)) & (x == knots_x[np.minimum(right, len(knots_x) - 1)])
    result[exact] = knots_n[right[exact]]
    nonexact = ~exact
    if np.any(nonexact):
        r = right[nonexact]
        if np.any(r <= 0) or np.any(r >= len(knots_x)):
            raise OracleAuditError("latency support interval was not bracketed")
        result[nonexact] = np.minimum(knots_n[r - 1], knots_n[r])
    return result


def _surface_mode_vector(surface: Any, sample_id: str, mode_id: int) -> Dict[str, np.ndarray]:
    lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
    q = np.arange(lower, upper + 1, dtype=np.int64)
    rows = surface._rows_for(sample_id, mode_id)
    anchor_q = np.asarray([row.q_e4 for row in rows], dtype=np.int64)
    if tuple(int(value) for value in anchor_q) != Q_E4_GRID:
        raise OracleAuditError("quality-surface q grid drift")
    anchor_payload = np.asarray(
        [row.total_transmitted_bytes for row in rows], dtype=np.float64
    )
    components = [row.component(DIRECT_QUALITY_COMPONENT) for row in rows]
    if any(not component.valid or component.value is None for component in components):
        raise OracleAuditError("q_perc has undefined interpolation endpoints")
    anchor_quality = np.asarray([component.value for component in components], dtype=np.float64)
    if not np.all(np.diff(anchor_payload) < 0.0):
        raise OracleAuditError("payload anchors are not strictly decreasing")

    # np.interp has the same endpoint and adjacent-endpoint linear semantics as
    # the registered surface.  Support lies inside [0, 9800], so no extrapolation.
    payload = np.interp(q, anchor_q, anchor_payload)
    quality = np.interp(q, anchor_q, anchor_quality)
    if (
        not np.all(np.isfinite(payload))
        or not np.all(payload > 0.0)
        or not np.all(np.isfinite(quality))
        or np.any(quality < 0.0)
        or np.any(quality > 1.0)
    ):
        raise OracleAuditError("surface vector produced invalid payload/quality")
    datagrams = np.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES).astype(np.int64)
    return {"q": q, "payload": payload, "quality": quality, "datagrams": datagrams}


def _network_vector(network: Any, profile: str, payload: np.ndarray, datagrams: np.ndarray) -> Dict[str, np.ndarray]:
    model = network.profile_models.get(profile)
    if model is None:
        raise OracleAuditError(f"unknown network profile {profile!r}")
    if payload.ndim != 1 or datagrams.shape != payload.shape:
        raise OracleAuditError("payload/datagram vector shape mismatch")
    expected_datagrams = np.ceil(payload / network.contract.udp_payload_capacity_bytes).astype(np.int64)
    if not np.array_equal(datagrams, expected_datagrams):
        raise OracleAuditError("payload/datagram relation drift")
    if np.any(datagrams < model.datagram_min) or np.any(datagrams > model.datagram_max):
        raise OracleAuditError("datagram count escaped profile support")

    x = np.log(payload)
    reassembly, _ = _curve_vector(model.reassembly_curve, x)
    admission, _ = _curve_vector(model.admission_curve, x)
    reassembly = np.clip(reassembly, 0.0, 1.0)
    admission = np.clip(admission, 0.0, 1.0)
    p_admit = reassembly * admission

    local_support = _local_latency_support_vector(model, x)
    latency_values: Dict[str, np.ndarray] = {}
    effective_support = local_support.copy()
    for name in CONDITIONAL_PERCENTILES:
        values, support = _curve_vector(model.latency_curves[name], x)
        latency_values[name] = values
        effective_support = np.minimum(effective_support, support)
    if np.any(effective_support < network.contract.latency_min_support):
        raise OracleAuditError("an enumerated action lacks qualified latency support")

    p50 = latency_values["p50"]
    p95 = np.maximum(p50, latency_values["p95"])
    p99 = np.maximum(p95, latency_values["p99"])
    fixed = fixed_stage_latency_ms()
    return {
        "p_admit": p_admit,
        "p50": fixed + p50,
        "p95": fixed + p95,
        "p99": fixed + p99,
    }


def _reward_vector(p_admit: np.ndarray, quality: np.ndarray, latency_p50: np.ndarray) -> np.ndarray:
    admitted = (
        PILOT_UTILITY_SPEC.quality_weight * quality
        - PILOT_UTILITY_SPEC.latency_weight
        * (latency_p50 / PILOT_UTILITY_SPEC.deadline_ms)
    )
    result = p_admit * admitted + (1.0 - p_admit) * PILOT_UTILITY_SPEC.service_non_admission_utility
    if not np.all(np.isfinite(result)):
        raise OracleAuditError("reward vector contains non-finite values")
    return result


def _outcome_from_vectors(mode_id: int, index: int, surface: Mapping[str, np.ndarray], network: Mapping[str, np.ndarray], reward: np.ndarray) -> OracleOutcome:
    return OracleOutcome(
        mode_id=mode_id,
        q_e4=int(surface["q"][index]),
        q_perc=float(surface["quality"][index]),
        total_transmitted_bytes=float(surface["payload"][index]),
        datagram_count=int(surface["datagrams"][index]),
        p_edge_admission_given_sent=float(network["p_admit"][index]),
        latency_proxy_p50_ms=float(network["p50"][index]),
        latency_proxy_p95_ms=float(network["p95"][index]),
        latency_proxy_p99_ms=float(network["p99"][index]),
        reward=float(reward[index]),
    )


def _best_index(objective: np.ndarray, latency: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[int]:
    if mask is None:
        eligible = np.arange(len(objective), dtype=np.int64)
    else:
        eligible = np.flatnonzero(mask)
    if len(eligible) == 0:
        return None
    best_value = np.max(objective[eligible])
    eligible = eligible[objective[eligible] == best_value]
    best_latency = np.min(latency[eligible])
    eligible = eligible[latency[eligible] == best_latency]
    return int(eligible[0])  # q is ascending, completing the stable tie rule.


def _better(candidate: OracleOutcome, incumbent: Optional[OracleOutcome], *, objective: str, tie_percentile: str) -> bool:
    if incumbent is None:
        return True
    left = float(getattr(candidate, objective))
    right = float(getattr(incumbent, objective))
    if left != right:
        return left > right
    left_latency = candidate.latency(tie_percentile)
    right_latency = incumbent.latency(tie_percentile)
    if left_latency != right_latency:
        return left_latency < right_latency
    return (candidate.mode_id, candidate.q_e4) < (incumbent.mode_id, incumbent.q_e4)


def _authoritative_outcome(evaluator: FitValidationActorEvaluatorV1, entry: Any, mode_id: int, q_e4: int) -> OracleOutcome:
    require_supported_action(mode_id, q_e4)
    query = evaluator.environment._surface.query_fit_q_e4(
        entry.scene_sample_id, mode_id, q_e4
    )
    quality_component = query.policy.component(DIRECT_QUALITY_COMPONENT)
    if not quality_component.valid or quality_component.value is None:
        raise OracleAuditError("authoritative winner q_perc is undefined")
    payload = float(query.policy.payload.total_transmitted_bytes)
    datagrams = math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES)
    prediction = evaluator.environment._prediction_session.predict(
        network_profile=entry.network_profile,
        payload_bytes=payload,
        datagram_count=datagrams,
    )
    latency = prediction.conditional_retained_survivor_latency_model()
    fixed = fixed_stage_latency_ms()
    p_admit = prediction.p_edge_admission_given_sent
    reward = PILOT_UTILITY_SPEC.expected_utility(
        p_edge_admission_given_sent=p_admit,
        q_perc=float(quality_component.value),
        latency_proxy_ms=fixed + latency.p50_ms,
    )
    return OracleOutcome(
        mode_id=mode_id,
        q_e4=q_e4,
        q_perc=float(quality_component.value),
        total_transmitted_bytes=payload,
        datagram_count=datagrams,
        p_edge_admission_given_sent=p_admit,
        latency_proxy_p50_ms=fixed + latency.p50_ms,
        latency_proxy_p95_ms=fixed + latency.p95_ms,
        latency_proxy_p99_ms=fixed + latency.p99_ms,
        reward=reward,
    )


def _assert_same_outcome(vector: OracleOutcome, scalar: OracleOutcome) -> None:
    if (vector.mode_id, vector.q_e4, vector.datagram_count) != (
        scalar.mode_id,
        scalar.q_e4,
        scalar.datagram_count,
    ):
        raise OracleAuditError("scalar/vector winner identity mismatch")
    for name in (
        "q_perc",
        "total_transmitted_bytes",
        "p_edge_admission_given_sent",
        "latency_proxy_p50_ms",
        "latency_proxy_p95_ms",
        "latency_proxy_p99_ms",
        "reward",
    ):
        if not math.isclose(
            float(getattr(vector, name)),
            float(getattr(scalar, name)),
            rel_tol=0.0,
            abs_tol=SCALAR_VECTOR_ABS_TOLERANCE,
        ):
            raise OracleAuditError(f"scalar/vector winner mismatch for {name}")


def _load_current_learned_rows(root: Path, panel: Any) -> Tuple[Dict[str, Any], ...]:
    csv_path = root / LEARNED_POLICY_CSV_RELATIVE_PATH
    manifest_path = root / LEARNED_POLICY_MANIFEST_RELATIVE_PATH
    if _sha256_file(csv_path) != LEARNED_POLICY_CSV_SHA256:
        raise OracleAuditError("learned-policy CSV SHA-256 drift")
    if _sha256_file(manifest_path) != LEARNED_POLICY_MANIFEST_SHA256:
        raise OracleAuditError("learned-policy manifest SHA-256 drift")
    # The content hash inside the manifest is authoritative.  The literal file
    # hash is retained to detect formatting/source replacement as well.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_files", {}).get("fit_validation_per_context.csv") != LEARNED_POLICY_CSV_SHA256:
        raise OracleAuditError("learned-policy manifest/CSV binding drift")
    if manifest.get("fit_validation_panel_sha256") != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
        raise OracleAuditError("learned-policy manifest/panel binding drift")
    if manifest.get("status") != "COMPLETE":
        raise OracleAuditError("learned-policy campaign is not complete")
    if tuple(manifest.get("seeds", ())) != tuple(REGISTERED_BASELINE_CONFIG.seeds):
        raise OracleAuditError("learned-policy seed inventory drift")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            if int(raw["update_index"]) != CURRENT_LEARNED_UPDATE:
                continue
            row = dict(raw)
            for name in (
                "seed", "update_index", "panel_index", "scene_rank", "frame_id",
                "radio_csv_row_number", "executed_mode_id", "executed_q_e4",
                "datagram_count",
            ):
                row[name] = int(row[name])
            for name in (
                "q_perc", "total_transmitted_bytes",
                "p_edge_admission_given_sent", "latency_proxy_ms",
                "latency_proxy_p95_ms", "latency_proxy_p99_ms", "reward",
            ):
                row[name] = float(row[name])
            rows.append(row)
    expected = len(panel.entries) * len(REGISTERED_BASELINE_CONFIG.seeds)
    if len(rows) != expected:
        raise OracleAuditError(f"current learned row count {len(rows)} != {expected}")
    seen = {(row["seed"], row["panel_index"]) for row in rows}
    expected_identities = {
        (seed, entry.panel_index)
        for seed in REGISTERED_BASELINE_CONFIG.seeds
        for entry in panel.entries
    }
    if seen != expected_identities:
        raise OracleAuditError("duplicate/missing learned seed-panel identities")
    by_index = {entry.panel_index: entry for entry in panel.entries}
    for row in rows:
        entry = by_index[row["panel_index"]]
        if (
            row["scene_rank"] != entry.scene_rank
            or row["sample_id"] != entry.scene_sample_id
            or row["episode_id"] != entry.scene_episode_id
            or row["frame_id"] != entry.scene_frame_id
            or row["network_profile"] != entry.network_profile
            or row["radio_csv_row_number"] != entry.radio_csv_row_number
            or row["radio_row_sha256"] != entry.radio_row_sha256
        ):
            raise OracleAuditError("learned row failed panel identity join")
        if row.get("result_status") != "MODELED_EXPECTED_UTILITY_DEFINED":
            raise OracleAuditError("learned row has a non-rewardable result status")
        require_supported_action(row["executed_mode_id"], row["executed_q_e4"])
    return tuple(rows)


def _fixed_actions() -> Tuple[Tuple[int, int], ...]:
    contract = default_contract()
    actions = []
    for anchor in contract.anchors:
        action = (anchor.mode.mode_id, anchor.q_e4)
        try:
            require_supported_action(*action)
        except ValueError:
            continue
        actions.append(action)
    result = tuple(sorted(set(actions)))
    if not result:
        raise OracleAuditError("no registered catalog anchor lies in modeled support")
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise OracleAuditError(f"refusing to write empty table {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise OracleAuditError(f"non-rectangular table {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "mean": mean(numeric),
        "median": median(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _baseline_summary_rows(records: Sequence[Mapping[str, Any]], *, comparator: str, seed: Optional[int] = None, mode_id: Optional[int] = None, q_e4: Optional[int] = None) -> list[Dict[str, Any]]:
    output: list[Dict[str, Any]] = []
    profiles = sorted({str(row["network_profile"]) for row in records})
    for profile in profiles:
        group = [row for row in records if row["network_profile"] == profile]
        for percentile in CONDITIONAL_PERCENTILES:
            latency_name = f"latency_proxy_{percentile}_ms"
            for budget in BUDGETS_MS:
                available = [row for row in group if row.get("action_available", True)]
                hits = [row for row in available if float(row[latency_name]) <= budget]
                output.append(
                    {
                        "comparator": comparator,
                        "seed": "" if seed is None else seed,
                        "mode_id": "" if mode_id is None else mode_id,
                        "q_e4": "" if q_e4 is None else q_e4,
                        "network_profile": profile,
                        "constraint_percentile": percentile,
                        "budget_ms": budget,
                        "population_count": len(group),
                        "action_available_count": len(available),
                        "deadline_hit_count": len(hits),
                        "deadline_miss_count": len(available) - len(hits),
                        "no_feasible_action_count": len(group) - len(available),
                        "constraint_nonattainment_count": len(group) - len(hits),
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                        "mean_reward": "" if not available else mean(float(row["reward"]) for row in available),
                        "mean_q_perc": "" if not available else mean(float(row["q_perc"]) for row in available),
                        "mean_constraint_latency_ms": "" if not available else mean(float(row[latency_name]) for row in available),
                    }
                )
    return output


def run_exact_split_oracle_audit(*, output_dir: Path, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run the bounded CPU-only audit and write create-only artifacts."""

    if EXACT_ACTION_COUNT_PER_SCENE != EXPECTED_EXACT_ACTION_COUNT_PER_SCENE:
        raise OracleAuditError("exact action count constant drift")
    enumerate_supported_actions()  # complete membership/cardinality proof check
    root = _project_root() if project_root is None else Path(project_root).resolve(strict=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    cuda_before = torch.cuda.is_initialized()

    context_unconstrained: list[Dict[str, Any]] = []
    context_constraints: list[Dict[str, Any]] = []
    comparator_records: Dict[str, list[Dict[str, Any]]] = {}
    learned_comparisons: list[Dict[str, Any]] = []
    fixed_summary_rows: list[Dict[str, Any]] = []
    exact_fixed_search_rows: list[Dict[str, Any]] = []

    with FitValidationActorEvaluatorV1(project_root=root) as evaluator:
        panel = evaluator.panel
        if panel.canonical_sha256() != REGISTERED_FIT_VALIDATION_PANEL_SHA256:
            raise OracleAuditError("validation panel identity drift")
        learned_rows = _load_current_learned_rows(root, panel)
        learned_by_panel: Dict[int, list[Dict[str, Any]]] = {}
        for row in learned_rows:
            learned_by_panel.setdefault(row["panel_index"], []).append(row)

        oracle_by_panel: Dict[int, OracleOutcome] = {}
        constraints_by_panel: Dict[Tuple[int, str, float], Tuple[Optional[OracleOutcome], Optional[OracleOutcome]]] = {}
        exact_fixed_reward_sums = {
            mode_id: np.zeros(upper - lower + 1, dtype=np.float64)
            for mode_id, (lower, upper) in enumerate(
                MODELED_SMOKE_MODE_Q_E4_BOUNDS
            )
        }
        vector_action_evaluations = 0

        for entry in panel.entries:
            best_reward: Optional[OracleOutcome] = None
            constrained: Dict[Tuple[str, float], list[Optional[OracleOutcome]]] = {
                (percentile, budget): [None, None]
                for percentile in CONDITIONAL_PERCENTILES
                for budget in BUDGETS_MS
            }
            for mode_id in range(len(MODELED_SMOKE_MODE_Q_E4_BOUNDS)):
                surface_values = _surface_mode_vector(
                    evaluator.environment._surface, entry.scene_sample_id, mode_id
                )
                network_values = _network_vector(
                    evaluator.environment._network,
                    entry.network_profile,
                    surface_values["payload"],
                    surface_values["datagrams"],
                )
                rewards = _reward_vector(
                    network_values["p_admit"], surface_values["quality"], network_values["p50"]
                )
                exact_fixed_reward_sums[mode_id] += rewards
                vector_action_evaluations += len(rewards)

                index = _best_index(rewards, network_values["p50"])
                assert index is not None
                candidate = _outcome_from_vectors(mode_id, index, surface_values, network_values, rewards)
                if _better(candidate, best_reward, objective="reward", tie_percentile="p50"):
                    best_reward = candidate

                for percentile in CONDITIONAL_PERCENTILES:
                    latency_values = network_values[percentile]
                    for budget in BUDGETS_MS:
                        mask = latency_values <= budget
                        quality_index = _best_index(surface_values["quality"], latency_values, mask)
                        reward_index = _best_index(rewards, latency_values, mask)
                        if (quality_index is None) != (reward_index is None):
                            raise OracleAuditError("constraint feasibility differs by objective")
                        if quality_index is None:
                            continue
                        quality_candidate = _outcome_from_vectors(mode_id, quality_index, surface_values, network_values, rewards)
                        reward_candidate = _outcome_from_vectors(mode_id, int(reward_index), surface_values, network_values, rewards)
                        pair = constrained[(percentile, budget)]
                        if _better(quality_candidate, pair[0], objective="q_perc", tie_percentile=percentile):
                            pair[0] = quality_candidate
                        if _better(reward_candidate, pair[1], objective="reward", tie_percentile=percentile):
                            pair[1] = reward_candidate

            if best_reward is None:
                raise OracleAuditError("unconstrained oracle found no action")
            scalar = _authoritative_outcome(evaluator, entry, best_reward.mode_id, best_reward.q_e4)
            _assert_same_outcome(best_reward, scalar)
            best_reward = scalar
            oracle_by_panel[entry.panel_index] = best_reward
            context_unconstrained.append(
                {
                    "panel_index": entry.panel_index,
                    "scene_rank": entry.scene_rank,
                    "sample_id": entry.scene_sample_id,
                    "network_profile": entry.network_profile,
                    **best_reward.to_dict("oracle_"),
                }
            )

            for percentile in CONDITIONAL_PERCENTILES:
                for budget in BUDGETS_MS:
                    quality_best, reward_best = constrained[(percentile, budget)]
                    feasible = quality_best is not None
                    if feasible:
                        assert reward_best is not None
                        quality_scalar = _authoritative_outcome(evaluator, entry, quality_best.mode_id, quality_best.q_e4)
                        reward_scalar = _authoritative_outcome(evaluator, entry, reward_best.mode_id, reward_best.q_e4)
                        _assert_same_outcome(quality_best, quality_scalar)
                        _assert_same_outcome(reward_best, reward_scalar)
                        if quality_scalar.latency(percentile) > budget or reward_scalar.latency(percentile) > budget:
                            raise OracleAuditError("authoritative constrained winner misses budget")
                        quality_best, reward_best = quality_scalar, reward_scalar
                    constraints_by_panel[(entry.panel_index, percentile, budget)] = (quality_best, reward_best)
                    row: Dict[str, Any] = {
                        "panel_index": entry.panel_index,
                        "scene_rank": entry.scene_rank,
                        "sample_id": entry.scene_sample_id,
                        "network_profile": entry.network_profile,
                        "constraint_percentile": percentile,
                        "budget_ms": budget,
                        "split_action_feasible": feasible,
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                    }
                    empty = {name: "" for name in OracleOutcome.__dataclass_fields__}
                    row.update((quality_best.to_dict("best_quality_") if quality_best else {f"best_quality_{k}": v for k, v in empty.items()}))
                    row.update((reward_best.to_dict("best_reward_") if reward_best else {f"best_reward_{k}": v for k, v in empty.items()}))
                    context_constraints.append(row)

        expected_evaluations = len(panel.entries) * EXACT_ACTION_COUNT_PER_SCENE
        if vector_action_evaluations != expected_evaluations:
            raise OracleAuditError("exhaustive action-context evaluation count drift")

        # Validate and classify the three current learned actors.
        learned_records: list[Dict[str, Any]] = []
        for row in learned_rows:
            entry = panel.entries[row["panel_index"]]
            scalar = _authoritative_outcome(evaluator, entry, row["executed_mode_id"], row["executed_q_e4"])
            frozen = OracleOutcome(
                mode_id=row["executed_mode_id"], q_e4=row["executed_q_e4"],
                q_perc=row["q_perc"], total_transmitted_bytes=row["total_transmitted_bytes"],
                datagram_count=row["datagram_count"],
                p_edge_admission_given_sent=row["p_edge_admission_given_sent"],
                latency_proxy_p50_ms=row["latency_proxy_ms"],
                latency_proxy_p95_ms=row["latency_proxy_p95_ms"],
                latency_proxy_p99_ms=row["latency_proxy_p99_ms"], reward=row["reward"],
            )
            _assert_same_outcome(frozen, scalar)
            oracle = oracle_by_panel[row["panel_index"]]
            regret = oracle.reward - scalar.reward
            if regret < -SCALAR_VECTOR_ABS_TOLERANCE:
                raise OracleAuditError("learned reward exceeds exhaustive reward oracle")
            learned_records.append({
                "panel_index": row["panel_index"], "network_profile": entry.network_profile,
                "seed": row["seed"], "action_available": True, **scalar.to_dict(),
            })
            for percentile in CONDITIONAL_PERCENTILES:
                for budget in BUDGETS_MS:
                    feasible = constraints_by_panel[(entry.panel_index, percentile, budget)][0] is not None
                    miss = scalar.latency(percentile) > budget
                    classification = "HIT" if not miss else ("AVOIDABLE_MISS" if feasible else "UNAVOIDABLE_MISS")
                    learned_comparisons.append({
                        "seed": row["seed"], "panel_index": entry.panel_index,
                        "scene_rank": entry.scene_rank, "sample_id": entry.scene_sample_id,
                        "network_profile": entry.network_profile,
                        "constraint_percentile": percentile, "budget_ms": budget,
                        "learned_mode_id": scalar.mode_id, "learned_q_e4": scalar.q_e4,
                        "learned_latency_proxy_ms": scalar.latency(percentile),
                        "learned_reward": scalar.reward,
                        "unconstrained_reward_oracle": oracle.reward,
                        "current_reward_oracle_regret": max(0.0, regret),
                        "split_action_feasible": feasible,
                        "miss_classification": classification,
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                    })
        comparator_records["LEARNED_UPDATE_5000"] = learned_records

        # Registered-seed random baseline, uniformly over every supported pair.
        for seed in REGISTERED_BASELINE_CONFIG.seeds:
            name = f"REGISTERED_SEEDED_RANDOM_{seed}"
            records = []
            for entry in panel.entries:
                action = choose_registered_random_action(seed=seed, panel_index=entry.panel_index)
                outcome = _authoritative_outcome(evaluator, entry, *action)
                records.append({"panel_index": entry.panel_index, "network_profile": entry.network_profile, "seed": seed, "action_available": True, **outcome.to_dict()})
            comparator_records[name] = records
        comparator_records["REGISTERED_SEEDED_RANDOM_POOLED"] = [
            row
            for seed in REGISTERED_BASELINE_CONFIG.seeds
            for row in comparator_records[f"REGISTERED_SEEDED_RANDOM_{seed}"]
        ]

        # Every supported catalog anchor is reported; "best fixed" is the
        # transparent post-hoc maximizer of mean current reward over all 340.
        fixed_records: Dict[Tuple[int, int], list[Dict[str, Any]]] = {}
        for action in _fixed_actions():
            records = []
            for entry in panel.entries:
                outcome = _authoritative_outcome(evaluator, entry, *action)
                records.append({"panel_index": entry.panel_index, "network_profile": entry.network_profile, "action_available": True, **outcome.to_dict()})
            fixed_records[action] = records
        best_catalog_fixed_action = min(
            fixed_records,
            key=lambda action: (-mean(float(row["reward"]) for row in fixed_records[action]), action),
        )
        comparator_records["POSTHOC_BEST_SUPPORTED_CATALOG_FIXED"] = fixed_records[best_catalog_fixed_action]
        for action, records in sorted(fixed_records.items()):
            for profile in tuple(panel.profile_order) + ("ALL",):
                group = records if profile == "ALL" else [row for row in records if row["network_profile"] == profile]
                fixed_summary_rows.append({
                    "mode_id": action[0], "q_e4": action[1], "network_profile": profile,
                    "context_count": len(group),
                    "mean_reward": mean(float(row["reward"]) for row in group),
                    "mean_q_perc": mean(float(row["q_perc"]) for row in group),
                    "is_posthoc_best_catalog_fixed_overall": action == best_catalog_fixed_action,
                })

        # Fair context-independent comparator with the same fine-grained q_e4
        # access as the learned policy.  Its single action is selected by an
        # exhaustive post-hoc search over all 52,240 executable pairs.
        best_exact_fixed_action: Optional[Tuple[int, int]] = None
        best_exact_fixed_sum = -math.inf
        for mode_id, sums in exact_fixed_reward_sums.items():
            lower, _upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
            for index, reward_sum_value in enumerate(sums):
                action = (mode_id, lower + index)
                reward_sum = float(reward_sum_value)
                exact_fixed_search_rows.append({
                    "mode_id": action[0],
                    "q_e4": action[1],
                    "context_count": len(panel.entries),
                    "mean_current_reward": reward_sum / len(panel.entries),
                })
                if (
                    reward_sum > best_exact_fixed_sum
                    or (
                        reward_sum == best_exact_fixed_sum
                        and (
                            best_exact_fixed_action is None
                            or action < best_exact_fixed_action
                        )
                    )
                ):
                    best_exact_fixed_sum = reward_sum
                    best_exact_fixed_action = action
        if (
            best_exact_fixed_action is None
            or len(exact_fixed_search_rows) != EXACT_ACTION_COUNT_PER_SCENE
        ):
            raise OracleAuditError("exact-support fixed-action search was incomplete")
        exact_fixed_records = []
        for entry in panel.entries:
            outcome = _authoritative_outcome(
                evaluator, entry, *best_exact_fixed_action
            )
            exact_fixed_records.append({
                "panel_index": entry.panel_index,
                "network_profile": entry.network_profile,
                "action_available": True,
                **outcome.to_dict(),
            })
        scalar_exact_fixed_mean = mean(
            float(row["reward"]) for row in exact_fixed_records
        )
        if not math.isclose(
            scalar_exact_fixed_mean,
            best_exact_fixed_sum / len(panel.entries),
            rel_tol=0.0,
            abs_tol=SCALAR_VECTOR_ABS_TOLERANCE,
        ):
            raise OracleAuditError(
                "exact-support fixed winner failed authoritative scalar validation"
            )
        comparator_records["POSTHOC_BEST_EXACT_SUPPORT_FIXED"] = (
            exact_fixed_records
        )

        # Oracle comparator records.
        comparator_records["UNCONSTRAINED_CURRENT_REWARD_ORACLE"] = [
            {"panel_index": entry.panel_index, "network_profile": entry.network_profile, "action_available": True, **oracle_by_panel[entry.panel_index].to_dict()}
            for entry in panel.entries
        ]

        comparator_summary: list[Dict[str, Any]] = []
        comparator_summary.extend(_baseline_summary_rows(comparator_records["LEARNED_UPDATE_5000"], comparator="LEARNED_UPDATE_5000"))
        for seed in REGISTERED_BASELINE_CONFIG.seeds:
            name = f"REGISTERED_SEEDED_RANDOM_{seed}"
            comparator_summary.extend(_baseline_summary_rows(comparator_records[name], comparator=name, seed=seed))
        comparator_summary.extend(
            _baseline_summary_rows(
                comparator_records["REGISTERED_SEEDED_RANDOM_POOLED"],
                comparator="REGISTERED_SEEDED_RANDOM_POOLED",
            )
        )
        for action, records in sorted(fixed_records.items()):
            comparator_summary.extend(
                _baseline_summary_rows(
                    records,
                    comparator="FIXED_SUPPORTED_CATALOG_ANCHOR",
                    mode_id=action[0],
                    q_e4=action[1],
                )
            )
        comparator_summary.extend(_baseline_summary_rows(comparator_records["POSTHOC_BEST_SUPPORTED_CATALOG_FIXED"], comparator="POSTHOC_BEST_SUPPORTED_CATALOG_FIXED", mode_id=best_catalog_fixed_action[0], q_e4=best_catalog_fixed_action[1]))
        comparator_summary.extend(_baseline_summary_rows(comparator_records["POSTHOC_BEST_EXACT_SUPPORT_FIXED"], comparator="POSTHOC_BEST_EXACT_SUPPORT_FIXED", mode_id=best_exact_fixed_action[0], q_e4=best_exact_fixed_action[1]))
        comparator_summary.extend(_baseline_summary_rows(comparator_records["UNCONSTRAINED_CURRENT_REWARD_ORACLE"], comparator="UNCONSTRAINED_CURRENT_REWARD_ORACLE"))

        # Constrained reward oracle gets one row per declared constraint, since
        # its selected action changes with percentile and budget.
        for profile in panel.profile_order:
            entries = [entry for entry in panel.entries if entry.network_profile == profile]
            for percentile in CONDITIONAL_PERCENTILES:
                for budget in BUDGETS_MS:
                    available = [constraints_by_panel[(entry.panel_index, percentile, budget)][1] for entry in entries]
                    outcomes = [item for item in available if item is not None]
                    comparator_summary.append({
                        "comparator": "BUDGET_CONSTRAINED_CURRENT_REWARD_ORACLE",
                        "seed": "", "mode_id": "", "q_e4": "",
                        "network_profile": profile,
                        "constraint_percentile": percentile, "budget_ms": budget,
                        "population_count": len(entries),
                        "action_available_count": len(outcomes),
                        "deadline_hit_count": len(outcomes),
                        "deadline_miss_count": 0,
                        "no_feasible_action_count": len(entries) - len(outcomes),
                        "constraint_nonattainment_count": len(entries) - len(outcomes),
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                        "mean_reward": "" if not outcomes else mean(item.reward for item in outcomes),
                        "mean_q_perc": "" if not outcomes else mean(item.q_perc for item in outcomes),
                        "mean_constraint_latency_ms": "" if not outcomes else mean(item.latency(percentile) for item in outcomes),
                    })

        # Profile-specific feasibility, maximum-quality, and learned miss audit.
        profile_summary: list[Dict[str, Any]] = []
        for profile in panel.profile_order:
            profile_entries = [entry for entry in panel.entries if entry.network_profile == profile]
            for percentile in CONDITIONAL_PERCENTILES:
                for budget in BUDGETS_MS:
                    pairs = [constraints_by_panel[(entry.panel_index, percentile, budget)] for entry in profile_entries]
                    quality_outcomes = [pair[0] for pair in pairs if pair[0] is not None]
                    learned_group = [row for row in learned_comparisons if row["network_profile"] == profile and row["constraint_percentile"] == percentile and row["budget_ms"] == budget]
                    counts = {name: sum(row["miss_classification"] == name for row in learned_group) for name in ("HIT", "AVOIDABLE_MISS", "UNAVOIDABLE_MISS")}
                    quality_stats = _summarize([item.q_perc for item in quality_outcomes])
                    regret_stats = _summarize([float(row["current_reward_oracle_regret"]) for row in learned_group])
                    profile_summary.append({
                        "network_profile": profile, "constraint_percentile": percentile,
                        "budget_ms": budget, "context_count": len(profile_entries),
                        "feasible_context_count": len(quality_outcomes),
                        "infeasible_context_count": len(profile_entries) - len(quality_outcomes),
                        "contextwise_max_quality_mean": quality_stats["mean"],
                        "contextwise_max_quality_median": quality_stats["median"],
                        "contextwise_max_quality_min": quality_stats["min"],
                        "contextwise_max_quality_max": quality_stats["max"],
                        "learned_decision_count": len(learned_group),
                        "learned_hit_count": counts["HIT"],
                        "learned_avoidable_miss_count": counts["AVOIDABLE_MISS"],
                        "learned_unavoidable_miss_count": counts["UNAVOIDABLE_MISS"],
                        "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
                        "learned_current_reward_oracle_regret_mean": regret_stats["mean"],
                        "learned_current_reward_oracle_regret_median": regret_stats["median"],
                        "learned_current_reward_oracle_regret_max": regret_stats["max"],
                    })

        evaluator._assert_no_runtime_side_effects()

    if not cuda_before and torch.cuda.is_initialized():
        raise OracleAuditError("CPU-only oracle unexpectedly initialized CUDA")

    _write_csv(destination / "context_unconstrained_reward_oracle.csv", context_unconstrained)
    _write_csv(destination / "context_constrained_oracles.csv", context_constraints)
    _write_csv(destination / "learned_policy_comparison.csv", learned_comparisons)
    _write_csv(destination / "profile_summary.csv", profile_summary)
    _write_csv(destination / "comparator_summary.csv", comparator_summary)
    _write_csv(destination / "fixed_action_baselines.csv", fixed_summary_rows)
    _write_csv(destination / "exact_support_fixed_search.csv", exact_fixed_search_rows)

    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files[path.name] = _sha256_file(path)
    summary = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "scope": {
            "evidence": EVALUATION_PROVENANCE_LABEL,
            "disclosure": EVALUATION_SCOPE_DISCLOSURE,
            "claims_excluded": [
                "LIVE_OR_ONLINE_PERFORMANCE", "UNTOUCHED_FINAL_TEST",
                "GENERALIZATION", "DEPLOYMENT", "TIMEOUT_PROBABILITY",
            ],
            "tail_latency_semantics": (
                "P95_P99_ARE_POSTHOC_CONDITIONAL_RETAINED_SURVIVOR_PROXY_"
                "DIAGNOSTICS_NOT_SAMPLED_LOSSES"
            ),
        },
        "exactness": {
            "action_domain": "ALL_INCLUSIVE_INTEGER_Q_E4_VALUES_IN_EACH_REGISTERED_MODE_SUPPORT",
            "continuous_q_quotient_reason": "WIRE_ROUND_HALF_UP_TO_INTEGER_Q_E4_MAKES_EACH_CELL_EXECUTION_EQUIVALENT",
            "mode_q_e4_bounds": [list(pair) for pair in MODELED_SMOKE_MODE_Q_E4_BOUNDS],
            "actions_per_context": EXACT_ACTION_COUNT_PER_SCENE,
            "panel_contexts": 340,
            "exhaustive_action_context_evaluations": 340 * EXACT_ACTION_COUNT_PER_SCENE,
            "winner_validation": "EVERY_REPORTED_ORACLE_WINNER_REEVALUATED_BY_AUTHORITATIVE_SCALAR_SURFACE_AND_NETWORK_SESSION",
            "scalar_vector_absolute_tolerance": SCALAR_VECTOR_ABS_TOLERANCE,
            "tie_rule": "MAX_OBJECTIVE_THEN_MIN_CONSTRAINT_LATENCY_THEN_MIN_MODE_ID_THEN_MIN_Q_E4",
        },
        "bindings": {
            "fit_validation_panel_sha256": REGISTERED_FIT_VALIDATION_PANEL_SHA256,
            "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
            "pilot_utility_spec_sha256": PILOT_UTILITY_SPEC_SHA256,
            "learned_policy_csv_relative_path": LEARNED_POLICY_CSV_RELATIVE_PATH,
            "learned_policy_csv_sha256": LEARNED_POLICY_CSV_SHA256,
            "learned_policy_manifest_relative_path": LEARNED_POLICY_MANIFEST_RELATIVE_PATH,
            "learned_policy_manifest_sha256": LEARNED_POLICY_MANIFEST_SHA256,
            "learned_policy_update": CURRENT_LEARNED_UPDATE,
            "oracle_implementation_sha256": _sha256_file(Path(__file__)),
        },
        "constraints": {
            "budgets_ms": list(BUDGETS_MS),
            "percentiles": list(CONDITIONAL_PERCENTILES),
            "comparison": "LATENCY_PROXY_LE_BUDGET",
            "feasibility_semantics": CONDITIONAL_FEASIBILITY_SEMANTICS,
        },
        "random_baseline": {
            "rule_schema": RANDOM_RULE_SCHEMA,
            "registered_seeds": list(REGISTERED_BASELINE_CONFIG.seeds),
            "sampling": "SHA256_IDENTITY_DERIVED_REJECTION_SAMPLED_UNIFORM_OVER_ALL_SUPPORTED_ACTION_PAIRS",
            "rule_sha256": canonical_sha256({
                "fields": ["panel_index", "panel_sha256", "rejection_counter", "rule_schema", "seed"],
                "population": EXACT_ACTION_COUNT_PER_SCENE,
                "rule_schema": RANDOM_RULE_SCHEMA,
            }),
        },
        "catalog_fixed_baseline": {
            "candidate_definition": "EVERY_REGISTERED_72_CATALOG_ANCHOR_THAT_IS_INSIDE_MODELED_SMOKE_SUPPORT",
            "candidate_count": len(_fixed_actions()),
            "posthoc_best_action": {"mode_id": best_catalog_fixed_action[0], "q_e4": best_catalog_fixed_action[1]},
            "selection_metric": "MAXIMUM_MEAN_CURRENT_REWARD_OVER_ALL_340_PANEL_CONTEXTS",
        },
        "exact_support_fixed_baseline": {
            "candidate_definition": "EVERY_ONE_OF_THE_52240_REGISTERED_EXECUTABLE_ACTION_PAIRS",
            "candidate_count": EXACT_ACTION_COUNT_PER_SCENE,
            "posthoc_best_action": {
                "mode_id": best_exact_fixed_action[0],
                "q_e4": best_exact_fixed_action[1],
            },
            "mean_current_reward": scalar_exact_fixed_mean,
            "selection_metric": "MAXIMUM_MEAN_CURRENT_REWARD_OVER_ALL_340_PANEL_CONTEXTS",
        },
        "row_counts": {
            "context_unconstrained_reward_oracle": len(context_unconstrained),
            "context_constrained_oracles": len(context_constraints),
            "learned_policy_comparison": len(learned_comparisons),
            "profile_summary": len(profile_summary),
            "comparator_summary": len(comparator_summary),
            "fixed_action_baselines": len(fixed_summary_rows),
            "exact_support_fixed_search": len(exact_fixed_search_rows),
        },
        "files": files,
        "profile_summary": profile_summary,
    }
    summary["canonical_content_sha256"] = canonical_sha256(summary)
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    complete = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "summary_sha256": _sha256_file(summary_path),
        "files": {**files, "summary.json": _sha256_file(summary_path)},
    }
    (destination / "COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = run_exact_split_oracle_audit(output_dir=args.output, project_root=args.project_root)
    print(json.dumps({
        "status": summary["status"],
        "actions_per_context": summary["exactness"]["actions_per_context"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
