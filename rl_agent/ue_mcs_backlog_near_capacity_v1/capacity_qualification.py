"""Bounded capacity-qualification stage under the exact 273PRB/4D5U radio.

Run 4 originally froze three "near-capacity" actions by re-deriving capacity
from Run-3 evidence. That evidence was collected on the legacy 40 MHz /
106 PRB / 7D2U radio, which this run rejects, so those numbers cannot site
tiers here: 273 PRB is 2.58x the bandwidth and 4D5U gives 5 uplink slots per 9
against 7D2U's 2 per 10, so the uplink is expected to be several times faster.
Any tier chosen from the 106 PRB figure would land in the wrong regime, which
is precisely the failure that made Run 3 ``INCONCLUSIVE``.

So: **no action is frozen here.** This stage measures the adverse-channel
uplink capacity under the exact radio, and the deterministic rule below then
selects the three tiers from the frozen catalogue. The stage is separately
authorized and runs before, and apart from, the scientific cells.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.ue_mcs_backlog_near_capacity_v1 import radio_binding as RB

ROOT = RB.ROOT

STAGE_ID = "ue_mcs_backlog_near_capacity_capacity_qualification_v1"
STAGE_AUTHORIZATION_TOKEN = "AUTHORIZE_NEAR_CAPACITY_CAPACITY_QUALIFICATION"

#: Refused outright rather than reused. Listed so a test can prove absence.
FORBIDDEN_LEGACY_CAPACITY_MBPS: Mapping[str, float] = {
    "ADVERSE_STABLE_106PRB_7D2U": 12.05,
    "FAVORABLE_STABLE_106PRB_7D2U": 42.13,
}
FORBIDDEN_LEGACY_CAPACITY_REASON = (
    "measured on OAI_N78_40MHZ_106PRB_7D2U_LEGACY; the radio lock records the "
    "legacy mapping as DO_NOT_REUSE_AS_100MHZ_EVIDENCE"
)


# --------------------------------------------------------------------------
# Operating points: the adverse trace's own core, held constant.
#
# Percentiles of the registered ADVERSE_STABLE target-SNR trace (450 samples).
# Held FIXED during qualification -- this stage measures a capacity surface, it
# does not replay a profile.
# --------------------------------------------------------------------------

ADVERSE_OPERATING_POINTS_DB: Mapping[str, float] = {
    "p25": 7.827,
    "p50": 8.608,
    "p75": 9.604,
}
#: The point whose measured capacity defines the tier boundary.
BOUNDARY_OPERATING_POINT = "p50"

#: The saturating probe. Deliberately the largest eligible catalogue action, so
#: the queue is certainly backlogged at every operating point and the measured
#: rate is capacity rather than offered load. It is a *probe*, never a tier.
PROBE_ACTION_ID = 0
PROBE_PROFILE_ID = "split_noae_uint8_q0000"
PROBE_PAYLOAD_BYTES = 3_568_326
# Exact catalogue-derived value: payload * 8 bits * 10 fps / 1e6.  This is
# deliberately not presentation-rounded because verify_probe_action is an
# execution-identity gate.
PROBE_OFFERED_MBPS = 285.46608

SETTLE_S = 3.0
MEASURE_S = 10.0
SAMPLE_PERIOD_S = 0.1
MIN_SERVICE_SAMPLES = 60
MIN_PUSCH_SNR_SAMPLES = 30
MAX_ACHIEVED_TARGET_SNR_ERROR_DB = 1.0
DRAIN_QUIET_INTERVAL_S = 0.5
#: Fraction of measured intervals that must show a continuously non-empty queue
#: for the probe to count as saturating.
MIN_BACKLOGGED_FRACTION = 0.80
#: Higher SNR must not measure as lower capacity by more than this fraction.
MONOTONICITY_TOLERANCE = 0.10

#: The capacity estimate is useful only if ordinary sampling uncertainty does
#: not change the three catalogue actions chosen by the deterministic rule.
#: These values are registered before collection and are deliberately fixed.
BOOTSTRAP_SEED = 2026092403
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_CONFIDENCE = 0.95

#: Bounded worst-case budget: one RAN, three five-second future-epoch guards,
#: three point windows plus receiver tails, and four twenty-second drain proofs.
EXPECTED_STAGE_RUNTIME_S = 320.0


class CapacityQualificationError(RuntimeError):
    """Raised when the stage cannot produce a usable capacity boundary."""


@dataclass(frozen=True)
class CapacityPoint:
    """Measured uplink service at one held target SNR."""

    achieved_pusch_snr_db_p50: float
    achieved_pusch_snr_samples: int
    label: str
    target_snr_db: float
    commanded_noise_power_db: float
    service_mbps_p10: float
    service_mbps_p50: float
    service_mbps_p90: float
    samples: int
    backlogged_fraction: float

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label, "target_snr_db": self.target_snr_db,
            "commanded_noise_power_db": self.commanded_noise_power_db,
            "achieved_pusch_snr_db_p50": self.achieved_pusch_snr_db_p50,
            "achieved_pusch_snr_samples": self.achieved_pusch_snr_samples,
            "service_mbps_p10": self.service_mbps_p10,
            "service_mbps_p50": self.service_mbps_p50,
            "service_mbps_p90": self.service_mbps_p90,
            "samples": self.samples,
            "backlogged_fraction": self.backlogged_fraction,
        }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Deterministic Type-7 percentile used by the qualification gate."""
    if not values:
        raise CapacityQualificationError("cannot take a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise CapacityQualificationError(
            f"percentile probability must be in [0,1], got {probability}")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise CapacityQualificationError("percentile input contains non-finite values")
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_median_interval(
    samples: Sequence[float], *, seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS, confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Fixed-seed non-parametric interval for the p50 capacity estimate.

    This is a pure diagnostic: it resamples the registered 100-ms service
    bins, never cells or tracer events, and returns the median interval used by
    :func:`tier_stability_gate`.
    """
    values = tuple(float(value) for value in samples)
    if not values:
        raise CapacityQualificationError("bootstrap needs at least one sample")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise CapacityQualificationError(
            "bootstrap service samples must be finite and non-negative")
    if not isinstance(draws, int) or draws < 100:
        raise CapacityQualificationError("bootstrap draws must be an integer >= 100")
    if not 0.0 < confidence < 1.0:
        raise CapacityQualificationError("bootstrap confidence must be in (0,1)")
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(draws)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "seed": seed,
        "draws": draws,
        "confidence": confidence,
        "sample_count": count,
        "point_estimate_mbps": float(statistics.median(values)),
        "lower_mbps": _percentile(medians, alpha),
        "upper_mbps": _percentile(medians, 1.0 - alpha),
    }


def audit_points(
    points: Sequence[CapacityPoint], *,
    boundary_service_samples: Sequence[float] | None = None,
    actions: Sequence[Mapping[str, Any]] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Gate the measured surface before any tier is derived from it."""
    labels = [point.label for point in points]
    by_label = {point.label: point for point in points}
    problems: list[str] = []

    required_labels = tuple(ADVERSE_OPERATING_POINTS_DB)
    missing = [label for label in required_labels if label not in by_label]
    if missing:
        problems.append(f"missing operating point(s): {missing}")
    unknown = sorted(set(labels) - set(required_labels))
    if unknown:
        problems.append(f"unknown operating point(s): {unknown}")
    duplicates = sorted(label for label in set(labels) if labels.count(label) != 1)
    if duplicates:
        problems.append(f"duplicate operating point(s): {duplicates}")
    if len(points) != len(required_labels):
        problems.append(
            f"expected exactly {len(required_labels)} operating points, got "
            f"{len(points)}")

    for point in points:
        numeric = (
            point.target_snr_db, point.commanded_noise_power_db,
            point.achieved_pusch_snr_db_p50,
            point.service_mbps_p10, point.service_mbps_p50,
            point.service_mbps_p90, point.backlogged_fraction,
        )
        if not all(math.isfinite(value) for value in numeric):
            problems.append(f"{point.label}: non-finite numeric field")
            continue
        expected_target = ADVERSE_OPERATING_POINTS_DB.get(point.label)
        if expected_target is not None and point.target_snr_db != expected_target:
            problems.append(
                f"{point.label}: target SNR {point.target_snr_db!r} != registered "
                f"{expected_target!r}")
        pusch_samples_valid = (
            type(point.achieved_pusch_snr_samples) is int
            and point.achieved_pusch_snr_samples >= 0
        )
        if not pusch_samples_valid:
            problems.append(f"{point.label}: PUSCH sample count is invalid")
        elif point.achieved_pusch_snr_samples < MIN_PUSCH_SNR_SAMPLES:
            problems.append(
                f"{point.label}: {point.achieved_pusch_snr_samples} PUSCH SNR "
                f"samples < {MIN_PUSCH_SNR_SAMPLES}")
        if expected_target is not None and abs(
            point.achieved_pusch_snr_db_p50 - expected_target
        ) > MAX_ACHIEVED_TARGET_SNR_ERROR_DB:
            problems.append(
                f"{point.label}: achieved PUSCH SNR "
                f"{point.achieved_pusch_snr_db_p50:.3f} dB differs from target "
                f"{expected_target:.3f} dB by more than "
                f"{MAX_ACHIEVED_TARGET_SNR_ERROR_DB:.1f} dB")
        samples_valid = isinstance(point.samples, int) and not isinstance(
            point.samples, bool)
        if not samples_valid:
            problems.append(f"{point.label}: samples is not an integer")
        elif point.samples < 0:
            problems.append(f"{point.label}: samples is negative")
        if samples_valid and point.samples < MIN_SERVICE_SAMPLES:
            problems.append(
                f"{point.label}: {point.samples} service samples < "
                f"{MIN_SERVICE_SAMPLES}")
        if not 0.0 <= point.backlogged_fraction <= 1.0:
            problems.append(
                f"{point.label}: backlogged fraction "
                f"{point.backlogged_fraction} is outside [0,1]")
        if point.backlogged_fraction < MIN_BACKLOGGED_FRACTION:
            problems.append(
                f"{point.label}: backlogged only {point.backlogged_fraction:.3f} "
                f"of intervals < {MIN_BACKLOGGED_FRACTION}; the probe did not "
                f"saturate, so this is offered load, not capacity")
        if not (0.0 <= point.service_mbps_p10 <= point.service_mbps_p50
                <= point.service_mbps_p90):
            problems.append(
                f"{point.label}: service percentiles are not ordered "
                f"p10<=p50<=p90")
        if point.service_mbps_p50 <= 0.0:
            problems.append(f"{point.label}: non-positive median service")

    ordered = [by_label[k] for k in ("p25", "p50", "p75") if k in by_label]
    monotone = True
    achieved_ordered = all(
        low.achieved_pusch_snr_db_p50 < high.achieved_pusch_snr_db_p50
        for low, high in zip(ordered, ordered[1:])
    )
    if len(ordered) == 3 and not achieved_ordered:
        problems.append(
            "achieved PUSCH SNR medians do not strictly resolve p25<p50<p75")
    for low, high in zip(ordered, ordered[1:]):
        if high.service_mbps_p50 < low.service_mbps_p50 * (1 - MONOTONICITY_TOLERANCE):
            monotone = False
            problems.append(
                f"capacity fell from {low.label} ({low.service_mbps_p50:.2f}) to "
                f"{high.label} ({high.service_mbps_p50:.2f}) Mbps as SNR rose, "
                f"beyond the {MONOTONICITY_TOLERANCE:.0%} tolerance")

    stability: dict[str, Any] | None = None
    if boundary_service_samples is None:
        problems.append(
            "boundary service samples absent; bootstrap tier-stability gate "
            "cannot be evaluated")
    else:
        try:
            stability = tier_stability_gate(
                boundary_service_samples, actions=actions, repo_root=repo_root)
            if not stability["stable"]:
                problems.extend(
                    f"tier stability: {problem}" for problem in stability["problems"])
        except (CapacityQualificationError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"tier stability could not be evaluated: {exc}")

    boundary = by_label.get(BOUNDARY_OPERATING_POINT)
    return {
        "points": [p.to_json() for p in points],
        "boundary_operating_point": BOUNDARY_OPERATING_POINT,
        "adverse_capacity_mbps": boundary.service_mbps_p50 if boundary else None,
        "achieved_pusch_snr_db_p50_by_point": {
            point.label: point.achieved_pusch_snr_db_p50 for point in points
        },
        "achieved_snr_strictly_ordered": len(ordered) == 3 and achieved_ordered,
        "monotonic_in_snr": monotone,
        "tier_stability": stability,
        "problems": problems,
        "qualified": not problems,
    }


# --------------------------------------------------------------------------
# Deterministic tier rule.
#
# Given the measured boundary C and the frozen catalogue, the three tiers are a
# pure function -- no judgement, no post-hoc choice. Registered before the
# boundary is measured.
# --------------------------------------------------------------------------

#: Offered load as a multiple of the measured adverse capacity boundary.
TIER_TARGET_RATIOS: Mapping[str, float] = {
    "low": 0.50,
    "medium": 1.00,
    "high": 1.40,
}
#: The medium tier must actually sit near the boundary.
MEDIUM_BOUNDARY_TOLERANCE = 0.25
FPS = 10.0


@dataclass(frozen=True)
class SelectedTier:
    tier: str
    action_id: int
    profile_id: str
    payload_bytes: int
    checkpoint_sha256: str
    offered_mbps: float
    target_ratio: float
    achieved_ratio: float

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier, "action_id": self.action_id,
            "profile_id": self.profile_id, "payload_bytes": self.payload_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "offered_mbps": self.offered_mbps,
            "target_ratio": self.target_ratio,
            "achieved_ratio": self.achieved_ratio,
        }


def _select_tiers_from_actions(
    adverse_capacity_mbps: float, actions: Sequence[Mapping[str, Any]],
) -> tuple[SelectedTier, ...]:
    """Pure implementation of the registered nearest-catalogue rule."""
    if not math.isfinite(adverse_capacity_mbps) or adverse_capacity_mbps <= 0:
        raise CapacityQualificationError(
            f"adverse capacity must be finite and positive, got "
            f"{adverse_capacity_mbps}")
    normalized: list[dict[str, Any]] = []
    for raw in actions:
        normalized.append({
            "action_id": int(raw["action_id"]),
            "profile_id": str(raw["profile_id"]),
            "payload_bytes": int(raw["payload_bytes"]),
            "checkpoint_sha256": str(raw["checkpoint_sha256"]),
            "offered_mbps": float(raw["offered_mbps"]),
        })
    if not normalized:
        raise CapacityQualificationError("no eligible catalogue actions")
    if any(not math.isfinite(row["offered_mbps"])
           or row["offered_mbps"] <= 0 for row in normalized):
        raise CapacityQualificationError("eligible action has invalid offered rate")
    normalized.sort(key=lambda row: (row["payload_bytes"], row["action_id"]))

    span_low = min(row["offered_mbps"] for row in normalized)
    span_high = max(row["offered_mbps"] for row in normalized)
    if adverse_capacity_mbps >= span_high:
        raise CapacityQualificationError(
            f"measured adverse capacity {adverse_capacity_mbps:.2f} Mbps is at or "
            f"above the catalogue's largest offered rate {span_high:.2f} Mbps; no "
            f"action can exceed capacity, so the boundary cannot be bracketed. "
            f"Refusing to freeze tiers.")
    if adverse_capacity_mbps <= span_low:
        raise CapacityQualificationError(
            f"measured adverse capacity {adverse_capacity_mbps:.2f} Mbps is at or "
            f"below the catalogue's smallest offered rate {span_low:.2f} Mbps. "
            f"Refusing to freeze tiers.")

    chosen: list[SelectedTier] = []
    for tier in ("low", "medium", "high"):
        ratio = TIER_TARGET_RATIOS[tier]
        target = ratio * adverse_capacity_mbps
        best = min(normalized,
                   key=lambda row: (abs(row["offered_mbps"] - target),
                                    row["action_id"]))
        chosen.append(SelectedTier(
            tier=tier, action_id=best["action_id"],
            profile_id=best["profile_id"], payload_bytes=best["payload_bytes"],
            checkpoint_sha256=best["checkpoint_sha256"],
            offered_mbps=best["offered_mbps"], target_ratio=ratio,
            achieved_ratio=best["offered_mbps"] / adverse_capacity_mbps))

    by_tier = {tier.tier: tier for tier in chosen}
    ids = [tier.action_id for tier in chosen]
    payloads = [tier.payload_bytes for tier in chosen]
    problems: list[str] = []
    if len(set(ids)) != 3:
        problems.append(f"tiers are not distinct actions: {ids}")
    if payloads != sorted(payloads) or len(set(payloads)) != 3:
        problems.append(f"payloads are not strictly increasing: {payloads}")
    if not by_tier["low"].offered_mbps < adverse_capacity_mbps:
        problems.append(
            f"low tier {by_tier['low'].offered_mbps:.2f} Mbps is not strictly "
            f"below the {adverse_capacity_mbps:.2f} Mbps boundary")
    if not by_tier["high"].offered_mbps > adverse_capacity_mbps:
        problems.append(
            f"high tier {by_tier['high'].offered_mbps:.2f} Mbps is not strictly "
            f"above the {adverse_capacity_mbps:.2f} Mbps boundary")
    if abs(by_tier["medium"].achieved_ratio - 1.0) > MEDIUM_BOUNDARY_TOLERANCE:
        problems.append(
            f"medium tier is {by_tier['medium'].achieved_ratio:.3f}x the boundary, "
            f"outside +/-{MEDIUM_BOUNDARY_TOLERANCE:.0%}")
    if problems:
        raise CapacityQualificationError(
            "deterministic tier rule refused: " + "; ".join(problems))
    return tuple(chosen)


def tier_stability_gate(
    service_samples: Sequence[float], *,
    actions: Sequence[Mapping[str, Any]] | None = None,
    repo_root: Path | None = None,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Require point/lower/upper capacity estimates to choose identical tiers."""
    interval = bootstrap_median_interval(
        service_samples, seed=seed, draws=draws, confidence=confidence)
    catalogue = list(actions) if actions is not None else eligible_actions(repo_root)
    selections: dict[str, list[int] | None] = {}
    selection_rows: dict[str, list[dict[str, Any]] | None] = {}
    problems: list[str] = []
    for name, capacity in (
            ("point", interval["point_estimate_mbps"]),
            ("lower", interval["lower_mbps"]),
            ("upper", interval["upper_mbps"])):
        try:
            selected = _select_tiers_from_actions(float(capacity), catalogue)
            selections[name] = [tier.action_id for tier in selected]
            selection_rows[name] = [tier.to_json() for tier in selected]
        except CapacityQualificationError as exc:
            selections[name] = None
            selection_rows[name] = None
            problems.append(f"{name} estimate cannot select tiers: {exc}")
    triplets = [tuple(value) for value in selections.values() if value is not None]
    stable = len(triplets) == 3 and len(set(triplets)) == 1
    if not stable and not problems:
        problems.append(
            "bootstrap capacity interval changes the deterministic tier triplet")
    return {
        "interval": interval,
        "tier_action_ids": selections,
        "tier_selections": selection_rows,
        "stable": stable,
        "problems": problems,
    }


def eligible_actions(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Catalogue actions usable as a tier, from the digest-checked catalogue."""
    from rl_agent.ue_mcs_backlog_near_capacity_v1 import contract as C

    root = repo_root or ROOT
    C.assert_catalog_digests(root)
    data = json.loads((root / C.ACTION_CATALOG_RELPATH).read_text())
    actions = []
    for entry in data["profiles"]:
        caps = entry["capabilities"]
        if not (caps["transport_valid"] and caps["agent_action_enabled"]):
            continue
        payload = int(entry["payload"]["zstd_median_bytes"])
        actions.append({
            "action_id": int(entry["action_id"]),
            "profile_id": str(entry["profile_id"]),
            "payload_bytes": payload,
            "checkpoint_sha256": str(entry["checkpoint_sha256"]),
            "offered_mbps": payload * 8 * FPS / 1e6,
        })
    actions.sort(key=lambda a: (a["payload_bytes"], a["action_id"]))
    return actions


def verify_probe_action(repo_root: Path | None = None) -> dict[str, Any]:
    """Reconcile the saturating probe constants to the pinned catalogue."""

    actions = eligible_actions(repo_root)
    matches = [row for row in actions if row["action_id"] == PROBE_ACTION_ID]
    if len(matches) != 1:
        raise CapacityQualificationError("probe action ID is absent or duplicated")
    probe = matches[0]
    largest = max(
        actions, key=lambda row: (row["payload_bytes"], -row["action_id"])
    )
    expected = (
        PROBE_PROFILE_ID,
        PROBE_PAYLOAD_BYTES,
        PROBE_OFFERED_MBPS,
    )
    observed = (
        probe["profile_id"], probe["payload_bytes"], probe["offered_mbps"]
    )
    if observed != expected or probe != largest:
        raise CapacityQualificationError(
            "registered saturating probe no longer equals the largest eligible "
            f"catalogue action: expected {expected}, observed {observed}")
    return {**probe, "verified_largest_eligible_action": True}


def select_tiers(
    adverse_capacity_mbps: float, *, repo_root: Path | None = None
) -> tuple[SelectedTier, ...]:
    """The registered deterministic tier rule.

    For each tier the rule takes the eligible catalogue action whose offered
    rate is closest to ``ratio * capacity``, breaking ties toward the smaller
    action id. It then refuses unless the result actually brackets the measured
    boundary: low strictly below, high strictly above, payloads strictly
    increasing, three distinct actions, and medium within
    ``MEDIUM_BOUNDARY_TOLERANCE`` of the boundary.

    Refusing is a legitimate outcome. If the catalogue cannot bracket the
    measured capacity, no tier is frozen and the decision goes back to Abiodun.
    """
    actions = eligible_actions(repo_root)
    return _select_tiers_from_actions(adverse_capacity_mbps, actions)


def stage_plan() -> dict[str, Any]:
    """The registered, bounded stage description. Pure; launches nothing."""
    return {
        "stage_id": STAGE_ID,
        "authorization_token": STAGE_AUTHORIZATION_TOKEN,
        "radio_profile_id": RB.RADIO_PROFILE_ID,
        "operating_points_db": dict(ADVERSE_OPERATING_POINTS_DB),
        "boundary_operating_point": BOUNDARY_OPERATING_POINT,
        "probe": {
            "action_id": PROBE_ACTION_ID, "profile_id": PROBE_PROFILE_ID,
            "payload_bytes": PROBE_PAYLOAD_BYTES,
            "offered_mbps": PROBE_OFFERED_MBPS,
            "role": "SATURATING_PROBE_NEVER_A_TIER",
        },
        "settle_s": SETTLE_S, "measure_s": MEASURE_S,
        "sample_period_s": SAMPLE_PERIOD_S,
        "min_service_samples": MIN_SERVICE_SAMPLES,
        "min_pusch_snr_samples": MIN_PUSCH_SNR_SAMPLES,
        "max_achieved_target_snr_error_db": MAX_ACHIEVED_TARGET_SNR_ERROR_DB,
        "min_backlogged_fraction": MIN_BACKLOGGED_FRACTION,
        "monotonicity_tolerance": MONOTONICITY_TOLERANCE,
        "primary_service_measurement": (
            "EXT_DN_UNIQUE_SSBURST_APPLICATION_PAYLOAD_BYTES_PER_FIXED_"
            "100MS_MONOTONIC_WINDOW"),
        "pusch_tb_is_primary": False,
        "corroboration": ["NR_RLC_TX_DEQUEUE", "NR_RLC_TX_SDU_RECURRENCE",
                           "GNB_PDCP_RX_DELIVER"],
        "inter_point_drain": {
            "probe_stopped_before_proof": True,
            "consecutive_zero_ticks": 5,
            "pdcp_rlc_quiet_interval_s": DRAIN_QUIET_INTERVAL_S,
            "timeout_s": 20.0,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "gate": "POINT_LOWER_UPPER_SELECT_IDENTICAL_TIER_TRIPLET",
        },
        "tier_target_ratios": dict(TIER_TARGET_RATIOS),
        "medium_boundary_tolerance": MEDIUM_BOUNDARY_TOLERANCE,
        "expected_runtime_s": EXPECTED_STAGE_RUNTIME_S,
        "forbidden_legacy_capacity_mbps": dict(FORBIDDEN_LEGACY_CAPACITY_MBPS),
        "forbidden_legacy_capacity_reason": FORBIDDEN_LEGACY_CAPACITY_REASON,
        "actions_frozen_here": False,
        "note": ("This stage measures capacity and applies the registered "
                 "deterministic tier rule. It selects no action by judgement "
                 "and freezes nothing until the rule's bracketing checks pass."),
    }
