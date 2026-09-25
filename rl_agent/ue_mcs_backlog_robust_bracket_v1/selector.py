"""Deterministic robust selector for a prospective byte-only queue design.

This module is deliberately offline and import-pure.  It reads no files,
starts no process, imports no model framework, and performs no network I/O.
Callers supply the already digest-checked action-catalogue JSON object.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from decimal import Decimal
from typing import Any, Mapping, Sequence


SELECTION_MODE = "ROBUST_EXACT_IDENTITY_GROUP_V1"
UNCERTAINTY_INTERPRETATION = (
    "RETAINED_EMPIRICAL_BOOTSTRAP_UNCERTAINTY_SET_NOT_POPULATION_"
    "CONFIDENCE_INTERVAL"
)
LOWER_MBPS = Decimal("28.512")
POINT_MBPS = Decimal("30.576")
UPPER_MBPS = Decimal("31.584")
EXTERIOR_MARGIN = Decimal("0.10")
FPS = Decimal("10")

TIER_TARGET_RATIOS: Mapping[str, Decimal] = {
    "low": Decimal("0.5"),
    "medium": Decimal("1.0"),
    "high": Decimal("1.4"),
}
TIER_ORDER = ("low", "medium", "high")
GROUP_IDENTITY_FIELDS = (
    "family",
    "quantizer",
    "checkpoint_sha256",
    "decoder_identity",
    "wire_identity_when_available",
)
EXPECTED_ACTION_IDS = {"low": 40, "medium": 39, "high": 38}
EXPECTED_PAYLOAD_BYTES = {
    "low": 126_237,
    "medium": 374_264,
    "high": 619_563,
}
EXPECTED_OFFERED_MBPS = {
    "low": Decimal("10.09896"),
    "medium": Decimal("29.94112"),
    "high": Decimal("49.56504"),
}


class RobustSelectionError(RuntimeError):
    """The catalogue cannot support the registered robust exact-mode rule."""


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON representation used for all semantic digests."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_number(value: Decimal) -> float:
    """Return a JSON number after all comparisons have used exact decimals."""

    return float(value)


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def registered_uncertainty_set() -> dict[str, Any]:
    return {
        "lower_mbps": _json_number(LOWER_MBPS),
        "point_mbps": _json_number(POINT_MBPS),
        "upper_mbps": _json_number(UPPER_MBPS),
        "interpretation": UNCERTAINTY_INTERPRETATION,
        "population_confidence_interval": False,
        "retained_from_preserved_attempt": True,
    }


def registered_rule() -> dict[str, Any]:
    low_limit = LOWER_MBPS * (Decimal("1") - EXTERIOR_MARGIN)
    high_limit = UPPER_MBPS * (Decimal("1") + EXTERIOR_MARGIN)
    return {
        "selection_mode": SELECTION_MODE,
        "cross_group_mixing_allowed": False,
        "group_identity_fields": list(GROUP_IDENTITY_FIELDS),
        "wire_identity_policy": (
            "INCLUDE_CODEC_AND_CANONICAL_FULL_WIRE_DESCRIPTOR_WHEN_AVAILABLE"
        ),
        "eligibility": {
            "transport_valid": True,
            "agent_action_enabled": True,
        },
        "exterior_margin_fraction": _json_number(EXTERIOR_MARGIN),
        "tier_admissibility": {
            "low": {
                "rule": "offered_mbps <= lower_mbps * (1 - margin)",
                "maximum_mbps": _json_number(low_limit),
            },
            "medium": {
                "rule": "lower_mbps <= offered_mbps <= upper_mbps",
                "minimum_mbps": _json_number(LOWER_MBPS),
                "maximum_mbps": _json_number(UPPER_MBPS),
            },
            "high": {
                "rule": "offered_mbps >= upper_mbps * (1 + margin)",
                "minimum_mbps": _json_number(high_limit),
            },
        },
        "target_ratios_to_point": {
            tier: _json_number(TIER_TARGET_RATIOS[tier]) for tier in TIER_ORDER
        },
        "target_mbps": {
            tier: _json_number(POINT_MBPS * TIER_TARGET_RATIOS[tier])
            for tier in TIER_ORDER
        },
        "within_group_action_ranking": [
            "ABS_DISTANCE_TO_TIER_TARGET_ASCENDING",
            "ACTION_ID_ASCENDING",
        ],
        "feasible_group_ranking": [
            "MEDIUM_ABS_DISTANCE_TO_POINT_ASCENDING",
            "LOW_ABS_DISTANCE_TO_HALF_POINT_ASCENDING",
            "HIGH_ABS_DISTANCE_TO_1_4_POINT_ASCENDING",
            "MEDIUM_ACTION_ID_ASCENDING",
            "LOW_ACTION_ID_ASCENDING",
            "HIGH_ACTION_ID_ASCENDING",
            "GROUP_IDENTITY_SHA256_ASCENDING",
        ],
    }


def _require_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise RobustSelectionError(f"{label} must be a non-empty string")
    return value


def _normalize_wire(value: Any, *, action_id: int) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict or not value:
        raise RobustSelectionError(
            f"action {action_id}: wire identity must be a non-empty object")
    # Round-trip through canonical JSON to reject non-JSON or non-finite data
    # and to detach the result from the caller's mutable catalogue object.
    try:
        descriptor = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RobustSelectionError(
            f"action {action_id}: wire identity is not canonical JSON: {exc}") from exc
    codec_id = descriptor.get("codec_id")
    if codec_id is not None and type(codec_id) not in (str, int):
        raise RobustSelectionError(
            f"action {action_id}: wire codec_id has an invalid type")
    return {
        "codec_id": codec_id,
        "descriptor": descriptor,
        "descriptor_sha256": canonical_sha256(descriptor),
    }


def _normalize_action(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        action_id = raw["action_id"]
        profile_id = raw["profile_id"]
        family = raw["family"]
        quantizer = raw["quantizer"]
        checkpoint = raw["checkpoint_sha256"]
        decoder = raw["decoder_identity"]
        payload = raw["payload"]["zstd_median_bytes"]
        capabilities = raw["capabilities"]
    except (KeyError, TypeError) as exc:
        raise RobustSelectionError(
            f"eligible catalogue action is missing an exact-mode field: {exc}") from exc
    if type(action_id) is not int or action_id < 0:
        raise RobustSelectionError("eligible action_id must be a non-negative integer")
    if (type(payload) not in (int, float) or not math.isfinite(float(payload))
            or float(payload) <= 0 or not float(payload).is_integer()):
        raise RobustSelectionError(
            f"action {action_id}: median payload must be a positive integer")
    payload = int(payload)
    checkpoint = _require_string(checkpoint, f"action {action_id} checkpoint")
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
        raise RobustSelectionError(
            f"action {action_id}: checkpoint is not a lowercase SHA-256")
    if type(capabilities) is not dict:
        raise RobustSelectionError(f"action {action_id}: capabilities is not an object")
    rate = Decimal(payload) * Decimal("8") * FPS / Decimal("1000000")
    wire = _normalize_wire(raw.get("wire"), action_id=action_id)
    return {
        "action_id": action_id,
        "profile_id": _require_string(profile_id, f"action {action_id} profile_id"),
        "family": _require_string(family, f"action {action_id} family"),
        "quantizer": _require_string(quantizer, f"action {action_id} quantizer"),
        "checkpoint_sha256": checkpoint,
        "decoder_identity": _require_string(
            decoder, f"action {action_id} decoder_identity"),
        "wire_identity": wire,
        "execution_mode": raw.get("execution_mode"),
        "payload_bytes": payload,
        "offered_mbps": _json_number(rate),
        "source_contract_tier": capabilities.get("source_contract_tier"),
        "perception_capability_caveat": {
            "absolute_service_ready": capabilities.get("absolute_service_ready"),
            "localization_requirements_passed": capabilities.get(
                "localization_requirements_passed"),
            "relative_preservation_passed": capabilities.get(
                "relative_preservation_passed"),
            "segmentation_installable": capabilities.get("segmentation_installable"),
            "stress_or_emergency_anchor": capabilities.get(
                "stress_or_emergency_anchor"),
        },
    }


def eligible_action_universe(catalogue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every transport-valid, agent-enabled action in exact ID order."""

    profiles = catalogue.get("profiles")
    if type(profiles) is not list or not profiles:
        raise RobustSelectionError("catalogue profiles must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for raw in profiles:
        if type(raw) is not dict:
            raise RobustSelectionError("catalogue profile must be an object")
        caps = raw.get("capabilities")
        if type(caps) is not dict:
            raise RobustSelectionError("catalogue profile capabilities must be an object")
        if not (caps.get("transport_valid") is True
                and caps.get("agent_action_enabled") is True):
            continue
        rows.append(_normalize_action(raw))
    rows.sort(key=lambda row: row["action_id"])
    ids = [row["action_id"] for row in rows]
    profiles_ids = [row["profile_id"] for row in rows]
    if not rows:
        raise RobustSelectionError("no eligible catalogue actions")
    if len(ids) != len(set(ids)):
        raise RobustSelectionError("eligible catalogue action IDs are duplicated")
    if len(profiles_ids) != len(set(profiles_ids)):
        raise RobustSelectionError("eligible catalogue profile IDs are duplicated")
    return rows


def _group_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": row["family"],
        "quantizer": row["quantizer"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "decoder_identity": row["decoder_identity"],
        "wire_identity": row["wire_identity"],
    }


def _rate(row: Mapping[str, Any]) -> Decimal:
    return Decimal(str(row["offered_mbps"]))


def _choose(
    candidates: Sequence[Mapping[str, Any]], target: Decimal,
) -> Mapping[str, Any]:
    return min(
        candidates,
        key=lambda row: (abs(_rate(row) - target), row["action_id"]),
    )


def _selection_row(tier: str, row: Mapping[str, Any]) -> dict[str, Any]:
    rate = _rate(row)
    selected = dict(row)
    selected.update({
        "tier": tier,
        "target_ratio_to_point": _json_number(TIER_TARGET_RATIOS[tier]),
        "target_mbps": _json_number(POINT_MBPS * TIER_TARGET_RATIOS[tier]),
        "absolute_target_distance_mbps": _json_number(
            abs(rate - POINT_MBPS * TIER_TARGET_RATIOS[tier])),
        "ratio_to_point": _json_number(rate / POINT_MBPS),
    })
    if tier == "low":
        selected["exterior_margin_fraction"] = _json_number(
            (LOWER_MBPS - rate) / LOWER_MBPS)
    elif tier == "medium":
        selected["inside_retained_uncertainty_set"] = True
    else:
        selected["exterior_margin_fraction"] = _json_number(
            (rate - UPPER_MBPS) / UPPER_MBPS)
    return selected


def select_robust_bracket(catalogue: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the registered exact-group rule to the complete candidate set.

    A feasible group must supply all three tiers without mixing family,
    quantizer, checkpoint, decoder or available wire/codec identity.  Low and
    high are at least ten percent outside the retained empirical uncertainty
    set.  Medium lies inside it.  The deterministic group ranking is recorded
    verbatim in :func:`registered_rule`.
    """

    universe = eligible_action_universe(catalogue)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: dict[str, dict[str, Any]] = {}
    for row in universe:
        identity = _group_identity(row)
        identity_sha = canonical_sha256(identity)
        identities[identity_sha] = identity
        grouped[identity_sha].append(row)

    low_limit = LOWER_MBPS * (Decimal("1") - EXTERIOR_MARGIN)
    high_limit = UPPER_MBPS * (Decimal("1") + EXTERIOR_MARGIN)
    evaluations: list[dict[str, Any]] = []
    feasible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for identity_sha in sorted(grouped):
        members = sorted(grouped[identity_sha], key=lambda row: row["action_id"])
        low = [row for row in members if _rate(row) <= low_limit]
        medium = [row for row in members if LOWER_MBPS <= _rate(row) <= UPPER_MBPS]
        high = [row for row in members if _rate(row) >= high_limit]
        evaluation: dict[str, Any] = {
            "identity_sha256": identity_sha,
            "identity": identities[identity_sha],
            "member_action_ids": [row["action_id"] for row in members],
            "candidate_action_ids": {
                "low": [row["action_id"] for row in low],
                "medium": [row["action_id"] for row in medium],
                "high": [row["action_id"] for row in high],
            },
            "feasible": bool(low and medium and high),
            "selected_action_ids": None,
            "ranking_key": None,
        }
        if evaluation["feasible"]:
            chosen = {
                "low": _choose(low, POINT_MBPS * TIER_TARGET_RATIOS["low"]),
                "medium": _choose(
                    medium, POINT_MBPS * TIER_TARGET_RATIOS["medium"]),
                "high": _choose(high, POINT_MBPS * TIER_TARGET_RATIOS["high"]),
            }
            distances = {
                tier: abs(
                    _rate(chosen[tier])
                    - POINT_MBPS * TIER_TARGET_RATIOS[tier])
                for tier in TIER_ORDER
            }
            rank = (
                distances["medium"], distances["low"], distances["high"],
                chosen["medium"]["action_id"], chosen["low"]["action_id"],
                chosen["high"]["action_id"], identity_sha,
            )
            evaluation["selected_action_ids"] = {
                tier: chosen[tier]["action_id"] for tier in TIER_ORDER
            }
            evaluation["ranking_key"] = [
                _decimal_text(distances["medium"]),
                _decimal_text(distances["low"]),
                _decimal_text(distances["high"]),
                chosen["medium"]["action_id"],
                chosen["low"]["action_id"],
                chosen["high"]["action_id"],
                identity_sha,
            ]
            feasible.append((rank, {tier: dict(chosen[tier]) for tier in TIER_ORDER}))
        evaluations.append(evaluation)

    if not feasible:
        raise RobustSelectionError(
            "no exact identity group robustly brackets the retained uncertainty set")
    feasible.sort(key=lambda item: item[0])
    winning_rank, winning = feasible[0]
    winning_identity = _group_identity(winning["medium"])
    selected = [_selection_row(tier, winning[tier]) for tier in TIER_ORDER]

    # This amendment is intentionally specific.  A catalogue that produces a
    # different result must cause a new version and review, not silent drift.
    observed_ids = {row["tier"]: row["action_id"] for row in selected}
    observed_payloads = {row["tier"]: row["payload_bytes"] for row in selected}
    observed_rates = {
        row["tier"]: Decimal(str(row["offered_mbps"])) for row in selected
    }
    if observed_ids != EXPECTED_ACTION_IDS:
        raise RobustSelectionError(
            f"registered catalogue no longer selects {EXPECTED_ACTION_IDS}: "
            f"observed {observed_ids}")
    if observed_payloads != EXPECTED_PAYLOAD_BYTES:
        raise RobustSelectionError(
            "selected median payloads differ from the registered result")
    if observed_rates != EXPECTED_OFFERED_MBPS:
        raise RobustSelectionError(
            "selected offered rates differ from the registered result")
    if winning_identity["family"] != "AE64" or winning_identity["quantizer"] != "UINT8":
        raise RobustSelectionError("winning exact identity is not AE64 UINT8")
    if any(row["source_contract_tier"] != "EMERGENCY_ONLY" for row in selected):
        raise RobustSelectionError(
            "selected actions no longer carry the required EMERGENCY_ONLY caveat")

    return {
        "schema": "scenesense.robust_exact_bracket_selection.v1",
        "selection_mode": SELECTION_MODE,
        "uncertainty_set": registered_uncertainty_set(),
        "rule": registered_rule(),
        "candidate_universe": universe,
        "candidate_universe_sha256": canonical_sha256(universe),
        "eligible_action_count": len(universe),
        "identity_group_count": len(evaluations),
        "feasible_group_count": len(feasible),
        "group_evaluations": evaluations,
        "winning_group_identity": winning_identity,
        "winning_group_identity_sha256": canonical_sha256(winning_identity),
        "winning_ranking_key": [
            _decimal_text(winning_rank[0]),
            _decimal_text(winning_rank[1]),
            _decimal_text(winning_rank[2]),
            winning_rank[3], winning_rank[4], winning_rank[5], winning_rank[6],
        ],
        "selected_tiers": selected,
        "selected_action_ids": observed_ids,
        "selected_payload_bytes": observed_payloads,
        "selected_offered_mbps": {
            tier: _json_number(observed_rates[tier]) for tier in TIER_ORDER
        },
        "cross_group_mixing_used": False,
        "catalogue_contract_tier": "EMERGENCY_ONLY",
        "byte_only_queue_design": True,
        "perception_endorsement": False,
    }

