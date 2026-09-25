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
PROBE_OFFERED_MBPS = 285.47

SETTLE_S = 3.0
MEASURE_S = 10.0
SAMPLE_PERIOD_S = 0.1
MIN_SERVICE_SAMPLES = 60
#: Fraction of measured intervals that must show a continuously non-empty queue
#: for the probe to count as saturating.
MIN_BACKLOGGED_FRACTION = 0.80
#: Higher SNR must not measure as lower capacity by more than this fraction.
MONOTONICITY_TOLERANCE = 0.10

#: Total live budget for the stage: one RAN instance, three short windows.
EXPECTED_STAGE_RUNTIME_S = 3 * (SETTLE_S + MEASURE_S) + 180.0


class CapacityQualificationError(RuntimeError):
    """Raised when the stage cannot produce a usable capacity boundary."""


@dataclass(frozen=True)
class CapacityPoint:
    """Measured uplink service at one held target SNR."""

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
            "service_mbps_p10": self.service_mbps_p10,
            "service_mbps_p50": self.service_mbps_p50,
            "service_mbps_p90": self.service_mbps_p90,
            "samples": self.samples,
            "backlogged_fraction": self.backlogged_fraction,
        }


def audit_points(points: Sequence[CapacityPoint]) -> dict[str, Any]:
    """Gate the measured surface before any tier is derived from it."""
    by_label = {p.label: p for p in points}
    problems: list[str] = []

    missing = [k for k in ADVERSE_OPERATING_POINTS_DB if k not in by_label]
    if missing:
        problems.append(f"missing operating point(s): {missing}")

    for point in points:
        if point.samples < MIN_SERVICE_SAMPLES:
            problems.append(
                f"{point.label}: {point.samples} service samples < "
                f"{MIN_SERVICE_SAMPLES}")
        if point.backlogged_fraction < MIN_BACKLOGGED_FRACTION:
            problems.append(
                f"{point.label}: backlogged only {point.backlogged_fraction:.3f} "
                f"of intervals < {MIN_BACKLOGGED_FRACTION}; the probe did not "
                f"saturate, so this is offered load, not capacity")
        if point.service_mbps_p50 <= 0:
            problems.append(f"{point.label}: non-positive median service")

    ordered = [by_label[k] for k in ("p25", "p50", "p75") if k in by_label]
    monotone = True
    for low, high in zip(ordered, ordered[1:]):
        if high.service_mbps_p50 < low.service_mbps_p50 * (1 - MONOTONICITY_TOLERANCE):
            monotone = False
            problems.append(
                f"capacity fell from {low.label} ({low.service_mbps_p50:.2f}) to "
                f"{high.label} ({high.service_mbps_p50:.2f}) Mbps as SNR rose, "
                f"beyond the {MONOTONICITY_TOLERANCE:.0%} tolerance")

    boundary = by_label.get(BOUNDARY_OPERATING_POINT)
    return {
        "points": [p.to_json() for p in points],
        "boundary_operating_point": BOUNDARY_OPERATING_POINT,
        "adverse_capacity_mbps": boundary.service_mbps_p50 if boundary else None,
        "monotonic_in_snr": monotone,
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
    if not adverse_capacity_mbps > 0:
        raise CapacityQualificationError(
            f"adverse capacity must be positive, got {adverse_capacity_mbps}")

    actions = eligible_actions(repo_root)
    if not actions:
        raise CapacityQualificationError("no eligible catalogue actions")

    span_low, span_high = actions[0]["offered_mbps"], actions[-1]["offered_mbps"]
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
        best = min(actions,
                   key=lambda a: (abs(a["offered_mbps"] - target), a["action_id"]))
        chosen.append(SelectedTier(
            tier=tier, action_id=best["action_id"],
            profile_id=best["profile_id"], payload_bytes=best["payload_bytes"],
            checkpoint_sha256=best["checkpoint_sha256"],
            offered_mbps=best["offered_mbps"], target_ratio=ratio,
            achieved_ratio=best["offered_mbps"] / adverse_capacity_mbps))

    by_tier = {t.tier: t for t in chosen}
    ids = [t.action_id for t in chosen]
    payloads = [t.payload_bytes for t in chosen]
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
        "min_backlogged_fraction": MIN_BACKLOGGED_FRACTION,
        "monotonicity_tolerance": MONOTONICITY_TOLERANCE,
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
