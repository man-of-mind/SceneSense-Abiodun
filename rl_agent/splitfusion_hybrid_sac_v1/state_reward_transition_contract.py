"""Versioned causal-state, reward-measurement and replay-transition contract.

Phase 4a of the SplitFusion conditional Hybrid-SAC foundation.  This module is
pure, deterministic, in-memory contract code over the four already-frozen
dependencies.  It defines *what a transition is* and *how its reward is
measured*; it does not learn, store, sample, simulate or communicate.

Scope
-----
Three contracts:

A. :class:`CausalStateV1` -- the causal observation available **before** a new
   policy decision, plus :class:`StateNormalizationSpecV1`,
   :class:`StateFreshnessPolicyV1` and the frozen
   :data:`POLICY_FEATURE_ORDER` mapping into a fixed-width feature vector.
B. :class:`QualityComponentsV1` and :class:`RewardSpecV1` -- raw CARLA
   ground-truth perception components, their registered normalization, the
   derived scalar reward and the separately preserved constraint signals.
C. :class:`ReplayTransitionV1` -- one exactly identified SMDP transition with
   its realized hold duration and discount multiplier.

Deliberately **not** implemented: any Gym/Gymnasium environment, simulator or
interpolator, replay-buffer storage, actor/critic/SAC network, recurrent memory,
optimizer or training loop, UDP/OAI/CARLA/Docker/CUDA integration, live runtime,
and the ``LOCAL``/``SKIP`` top-level modes.  The action remains exactly
``(joint mode among 12, continuous q)`` and the initial policy is feed-forward.

Action-space note
-----------------
The older four-AE / fixed-quantizer / recurrent-agent sketch in
``split_fusion_data_structure.md`` is **not** the contract here.  All 12
family-quantizer joint modes are retained and ``q`` is continuous, exactly as
the Phase-1 catalog adapter and DESIGN.md section 2 freeze them.

What may never enter the causal state
-------------------------------------
No image or radar tensor, no dynamic fraction, no temporal-information
descriptor, no object count, no ground truth, no network-profile label, no
``action_id`` anchor lookup, and no measurement that postdates the decision.
Session/decision/tensor/frame identifiers are **join and audit metadata only**:
they are never numerical policy features and never reward terms.
:func:`assert_policy_features_exclude_forbidden_fields` restates this as an
executable check over :data:`POLICY_FEATURE_ORDER`.

Ground-truth boundary
---------------------
Every quality component is ``CARLA_GT_EXACT``: a **privileged, non-deployable**
oracle that exists only inside the training/testbed instrument.  The label is a
typed :class:`GroundTruthSource`, not a free string, and it reports
``privileged=True`` / ``deployable=False``.  This schema makes no claim that
exact per-frame segmentation or localization accuracy is observable in physical
deployment; a deployed system has no such oracle.

Quality-ACK binding: anchor-only, and never snapped
---------------------------------------------------
The deployed evidence carrier ``sf_priv_quality_ack.v1`` identifies a decision
by ``action_id`` *and* ``profile_id`` and validates ``0 <= action_id < 72``, so
it can only ever describe one of the 72 **registered anchor** actions.  Hybrid
SAC emits arbitrary continuous ``q``, whose
:class:`~.transaction_identity.ExecutedActionIdentity` legitimately carries
``action_id=None``.

This module therefore treats that ACK as an **anchor-only adapter**:
:class:`QualityAckBindingV1` **fails closed** for an off-anchor action and
never, under any circumstance, snaps it to a nearest anchor.  The core reward
evidence is defined around the *full* ``ExecutedActionIdentity`` -- carried here
as its canonical SHA-256 -- so live arbitrary-``q`` support requires a future
**protocol-v2 ACK that carries that identity or its canonical hash**.  Until
that exists, off-anchor ``q`` has no deployable quality-evidence path, and this
contract says so by raising rather than by approximating.

Detection-metric boundary
-------------------------
The v1 ACK carries ``vehicle_tp``/``vehicle_fn``/``person_tp``/``person_fn``
but **no false-positive counts**.  Recall is therefore computable and precision,
F1 and any "comprehensive detection quality" claim are **not**.  Nothing in this
module derives or implies them.

Undefined-class masks
---------------------
A class with no ground-truth support is masked as undefined, never scored.  In
particular a GT-absent localization component can neither be supplied as a
zero error nor be rewarded as perfect recall: validity requires positive GT
support, and an invalid component is excluded from renormalization entirely.

Feature status
--------------
``camera_si`` and ``radar_p40`` are **candidate** state features.  They are
retained here because they are cheap, causal and measurable -- not because they
are established predictors.  Their predictive value still requires the later
ablation and SHAP validation, and this contract must not be read as having
settled it.

Fail-closed policy
------------------
Missing, malformed or stale SI, P40, SNR, BSR or MCS raises.  Nothing is ever
silently replaced by zero, by a nearest value or by a sentinel: the caller's
external runtime guard owns selecting and logging the registered fallback
action.  Absence is always carried as an explicit flag, never as a magic number.

Importing this module performs no filesystem, network, CUDA, CARLA or OAI
access.  The locked action catalog is read only if a caller explicitly asks the
Phase-1 contract for it.  The schema hash below is computed from in-module
literals plus the four dependency hashes.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from .action_contract import (
    CATALOG_SCHEMA,
    CATALOG_SHA256,
    EXECUTION_MODE,
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)
from .scene_descriptors import (
    P40_HORIZON_M,
    SCHEMA_ID as SCENE_SCHEMA_ID,
    SCHEMA_SHA256 as SCENE_SCHEMA_SHA256,
    SCHEMA_VERSION as SCENE_SCHEMA_VERSION,
    SceneDescriptorSample,
)
from .transaction_identity import (
    ACTION_IDENTITY_SCHEMA_ID,
    ACTION_IDENTITY_SCHEMA_SHA256,
    ActionHoldManifest,
    ExecutedActionIdentity,
    MINIMUM_HOLD_TENSORS,
    SCHEMA_ID as TRANSACTION_SCHEMA_ID,
    SCHEMA_SHA256 as TRANSACTION_SCHEMA_SHA256,
    SCHEMA_VERSION as TRANSACTION_SCHEMA_VERSION,
    canonical_json_bytes,
    canonical_sha256,
)
from .reward_ticket_controller import (
    B_REWARD_DEADLINE_NS,
    CONTROLLER_SCHEMA_ID,
    CONTROLLER_SCHEMA_SHA256,
    CONTROLLER_SCHEMA_VERSION,
    CompletedTicket,
    K_MIN_TENSORS,
    TERMINAL_LEARNING_DISPOSITION,
    TerminalClass,
)

__all__ = [
    # errors
    "StateRewardContractError",
    "CausalStateError",
    "StaleTelemetryError",
    "NormalizationSpecError",
    "QualityContractError",
    "OffAnchorQualityAckError",
    "UndefinedClassSupportError",
    "InsufficientQualitySupportError",
    "RewardSpecError",
    "AdjudicationError",
    "TransitionIdentityError",
    # schema
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SCHEMA_DESCRIPTOR",
    "SCHEMA_SHA256",
    "POLICY_FEATURE_ORDER",
    "POLICY_FEATURE_COUNT",
    "QUALITY_ACK_SCHEMA",
    "QUALITY_ACK_FAILURE_SCHEMA",
    "QUALITY_ACK_PROTOCOL_VERSION",
    "QUALITY_ACK_SOURCE",
    "QUALITY_ACK_REQUIRED_ANCHOR_FIELDS",
    "QUALITY_ACK_ANCHOR_ACTION_COUNT",
    "QUALITY_ACK_IS_ANCHOR_ONLY",
    "QUALITY_ACK_TIMING_CLOCK_DOMAIN",
    "QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE",
    "QUALITY_ACK_DERIVABLE_DETECTION_METRICS",
    "QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS",
    "REWARD_LATENCY_CLOCK_DOMAIN",
    "FORBIDDEN_REWARD_LATENCY_SOURCES",
    "PROTOCOL_V2_REQUIREMENT",
    "CANDIDATE_STATE_FEATURES",
    "CANDIDATE_FEATURE_VALIDATION_REQUIRED",
    "FORBIDDEN_POLICY_FEATURE_SUBSTRINGS",
    "assert_policy_features_exclude_forbidden_fields",
    # A: state
    "PreviousOutcomeV1",
    "CausalStateV1",
    "StateNormalizationSpecV1",
    "StateFreshnessPolicyV1",
    "PolicyFeatureVectorV1",
    "build_policy_features",
    # B: quality and reward
    "GroundTruthSource",
    "QualityAckBindingV1",
    "QualityEvidenceV1",
    "QualityComponentsV1",
    "QualityEvaluationV1",
    "LatencyMeasurementV1",
    "RewardSpecV1",
    "ConstraintCostsV1",
    "Adjudication",
    "AdjudicationRecordV1",
    "LearningEligibility",
    "DecisionOutcomeV1",
    "evaluate_completed_decision",
    # C: transition
    "ReplayTransitionV1",
    "build_replay_transition",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class StateRewardContractError(ValueError):
    """Base class for every Phase-4a contract violation."""


class CausalStateError(StateRewardContractError):
    """A causal-state field is missing, malformed or out of range."""


class StaleTelemetryError(CausalStateError):
    """A required observation is older than its registered freshness bound.

    This is a runtime condition for the external registered-fallback guard, not
    a numeric value: nothing is substituted for the stale measurement.
    """


class NormalizationSpecError(StateRewardContractError):
    """A normalization or freshness specification lacks explicit provenance."""


class QualityContractError(StateRewardContractError):
    """A raw quality component is malformed or contradicts its validity flag."""


class OffAnchorQualityAckError(QualityContractError):
    """An off-anchor continuous ``q`` was offered to the anchor-only v1 ACK.

    ``sf_priv_quality_ack.v1`` identifies a decision by ``action_id`` and
    ``profile_id`` and validates ``0 <= action_id < 72``, so it can only ever
    describe one of the 72 registered anchors.  An arbitrary continuous ``q``
    has no such identity.  This is raised instead of snapping the executed
    action to a nearest anchor -- which would silently attribute one action's
    measured quality to a different action -- and instead of fabricating an
    identity the wire format cannot carry.  Live arbitrary-``q`` evaluation
    needs the protocol-v2 ACK described by
    :data:`PROTOCOL_V2_REQUIREMENT`.
    """


class UndefinedClassSupportError(QualityContractError):
    """A component claims validity without positive ground-truth support.

    A class with no GT support is undefined, not perfect: it must be masked out
    rather than entered as a zero localization error or a full recall.
    """


class InsufficientQualitySupportError(QualityContractError):
    """Too few valid components exist to compute the registered quality."""


class RewardSpecError(StateRewardContractError):
    """A reward-specification value is missing or outside its declared domain."""


class AdjudicationError(StateRewardContractError):
    """An adjudication record is absent, misapplied or internally inconsistent."""


class TransitionIdentityError(StateRewardContractError):
    """A replay transition violates an exact-identity or causality invariant."""


# --------------------------------------------------------------------------- #
# Local helpers: fail closed, never normalize
# --------------------------------------------------------------------------- #


def _deep_freeze(value: Any) -> Any:
    """Recursively freeze a literal into read-only mappings and tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _exact_int(value: Any, name: str, error: type = CausalStateError) -> int:
    """Validate an exact Python ``int``; ``bool`` is rejected as a distinct type."""
    if isinstance(value, bool):
        raise error(f"{name} must be an int, not a bool: {value!r}")
    if type(value) is not int:
        raise error(
            f"{name} must be an exact int, got {type(value).__name__}: {value!r}"
        )
    return value


def _non_negative_int(value: Any, name: str, error: type = CausalStateError) -> int:
    _exact_int(value, name, error)
    if value < 0:
        raise error(f"{name} must be >= 0, got {value}")
    return value


def _positive_int(value: Any, name: str, error: type = CausalStateError) -> int:
    _exact_int(value, name, error)
    if value <= 0:
        raise error(f"{name} must be > 0, got {value}")
    return value


def _finite_float(value: Any, name: str, error: type = CausalStateError) -> float:
    """Validate a finite real scalar; ``bool`` and non-numerics are rejected."""
    if isinstance(value, bool):
        raise error(f"{name} must be a real scalar, not a bool: {value!r}")
    if not isinstance(value, (int, float)):
        raise error(
            f"{name} must be a finite real scalar, got "
            f"{type(value).__name__}: {value!r}"
        )
    as_float = float(value)
    if not math.isfinite(as_float):
        raise error(f"{name} must be finite, got {value!r}")
    return as_float


def _finite_in(
    value: Any,
    name: str,
    low: float,
    high: float,
    error: type = CausalStateError,
) -> float:
    as_float = _finite_float(value, name, error)
    if not low <= as_float <= high:
        raise error(f"{name} must lie in [{low}, {high}], got {as_float!r}")
    return as_float


def _exact_bool(value: Any, name: str, error: type = CausalStateError) -> bool:
    if type(value) is not bool:
        raise error(
            f"{name} must be a bool, got {type(value).__name__}: {value!r}"
        )
    return value


def _non_empty_str(value: Any, name: str, error: type = CausalStateError) -> str:
    if not isinstance(value, str) or value == "":
        raise error(
            f"{name} must be a non-empty str, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def _sha256_hex(value: Any, name: str, error: type = CausalStateError) -> str:
    if not isinstance(value, str):
        raise error(
            f"{name} must be a str digest, got {type(value).__name__}: {value!r}"
        )
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise error(f"{name} must be 64 lowercase hex characters, got {value!r}")
    return value


def _canonical_uuid(value: Any, error: type = CausalStateError) -> str:
    if not isinstance(value, str):
        raise error(
            f"session_uuid must be a str, got {type(value).__name__}: {value!r}"
        )
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise error(f"session_uuid is not a parsable UUID: {value!r}") from exc
    if str(parsed) != value:
        raise error(
            f"session_uuid must be the canonical lowercase hyphenated form "
            f"{str(parsed)!r}, got {value!r}"
        )
    return value


def _frozen_str_mapping(
    value: Any,
    name: str,
    error: type = StateRewardContractError,
) -> Mapping[str, str]:
    """Validate and deep-freeze a non-empty ``str -> str`` provenance mapping."""
    if not isinstance(value, Mapping) or not value:
        raise error(
            f"{name} must be a non-empty mapping of str to str, got "
            f"{type(value).__name__}: {value!r}"
        )
    for key, item in value.items():
        _non_empty_str(key, f"{name} key", error)
        _non_empty_str(item, f"{name}[{key!r}]", error)
    return MappingProxyType(dict(value))


# --------------------------------------------------------------------------- #
# Frozen policy-feature order
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Declared bindings to the privileged quality-ACK wire contract
# --------------------------------------------------------------------------- #
#
# These mirror rl_agent/splitfusion_quality_feedback_probe_v1/protocol.py.  They
# are *declared literals* rather than imports on purpose: that module resolves
# its own dependency through an absolute ``rl_agent.*`` import rooted at a
# different path than this package, so importing it here would require mutating
# ``sys.path`` at import time -- itself a side effect this phase promises not to
# have.  The accompanying test AST-parses the real protocol file and asserts
# every literal below still matches, so the binding cannot drift silently.

#: The privileged exact-quality ACK schema this adapter reads.
QUALITY_ACK_SCHEMA: str = "sf_priv_quality_ack.v1"
QUALITY_ACK_FAILURE_SCHEMA: str = "sf_priv_quality_fail.v1"
QUALITY_ACK_PROTOCOL_VERSION: int = 1
QUALITY_ACK_SOURCE: str = "EDGE_CARLA_GT"

#: The ACK identifies a decision by anchor fields and validates
#: ``0 <= action_id < 72``, so it is structurally anchor-only.
QUALITY_ACK_REQUIRED_ANCHOR_FIELDS: Tuple[str, ...] = ("action_id", "profile_id")
QUALITY_ACK_ANCHOR_ACTION_COUNT: int = 72
QUALITY_ACK_IS_ANCHOR_ONLY: bool = True

#: Every ACK timing field is a *wall* clock reading.  None of them may be used
#: for reward latency; see :data:`REWARD_LATENCY_CLOCK_DOMAIN`.
QUALITY_ACK_TIMING_CLOCK_DOMAIN: str = "WALL"

#: The v1 ACK carries TP and FN but no false positives.
QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE: bool = False
QUALITY_ACK_DERIVABLE_DETECTION_METRICS: Tuple[str, ...] = ("recall",)
QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS: Tuple[str, ...] = (
    "precision",
    "f1",
    "average_precision",
    "comprehensive_detection_quality",
)

#: Reward latency is UE-local and monotonic, from the policy decision to exact
#: feedback receipt.  It is never the map-install ACK latency and never mixes a
#: wall clock in: the two controller timestamps it is derived from come from one
#: injected monotonic source.
REWARD_LATENCY_CLOCK_DOMAIN: str = "UE_LOCAL_MONOTONIC"
FORBIDDEN_REWARD_LATENCY_SOURCES: Tuple[str, ...] = (
    "map_install_ack_latency",
    "quality_ack_wall_timings",
    "wall_clock",
    "mixed_wall_and_monotonic",
)

#: What a future ACK must carry before arbitrary continuous q can be evaluated
#: live.  Recorded in the schema so the gap is explicit rather than folklore.
PROTOCOL_V2_REQUIREMENT: str = (
    "a protocol-v2 quality ACK must carry the full ExecutedActionIdentity or "
    "its canonical SHA-256, because an arbitrary continuous q has no action_id "
    "or profile_id and must never be snapped to a nearest anchor"
)

#: SI and P40 are candidates pending ablation/SHAP validation, not established
#: predictors.
CANDIDATE_STATE_FEATURES: Tuple[str, ...] = ("camera_si", "radar_p40")
CANDIDATE_FEATURE_VALIDATION_REQUIRED: str = (
    "camera_si and radar_p40 are candidate state features retained for being "
    "cheap, causal and measurable; their predictive value still requires the "
    "registered ablation and SHAP validation and is NOT established here"
)


#: One-hot width of the previous joint mode, bound to the Phase-1 catalog.
_PREVIOUS_MODE_ONEHOT_WIDTH = EXPECTED_MODE_COUNT

#: The single frozen policy-feature order.  Fixed width, fixed names, fixed
#: positions.  Every entry is a scaled *observation* or an explicit
#: presence/validity mask; no identifier, ground truth, current-decision
#: outcome or future measurement appears.  Measurement ages deliberately do not
#: appear: they gate admission through :class:`StateFreshnessPolicyV1` (a stale
#: state cannot be vectorized at all) and no age scaling constant has been
#: measured, so adding one here would invent a fitted value.
POLICY_FEATURE_ORDER: Tuple[str, ...] = (
    "scene_camera_si_scaled",
    "scene_radar_p40",
    "radio_achieved_snr_db_scaled",
    "radio_bsr_log1p_scaled",
    "radio_mcs_index_scaled",
) + tuple(
    f"prev_joint_mode_onehot_{index:02d}"
    for index in range(_PREVIOUS_MODE_ONEHOT_WIDTH)
) + (
    "prev_q_normalized",
    "prev_quality_normalized",
    "prev_latency_normalized",
    "prev_present_mask",
    "prev_quality_valid_mask",
    "prev_latency_valid_mask",
)

POLICY_FEATURE_COUNT: int = len(POLICY_FEATURE_ORDER)

#: Substrings that must never appear in a policy-feature name.  Identifiers are
#: join metadata; ground truth and the current decision's outcome are not
#: causally available; ``action_id``/``profile_id`` would smuggle in an anchor
#: lookup that an arbitrary continuous ``q`` does not have.
FORBIDDEN_POLICY_FEATURE_SUBSTRINGS: Tuple[str, ...] = (
    "session",
    "uuid",
    "decision_seq",
    "tensor_seq",
    "frame_id",
    "carla_frame",
    "action_id",
    "profile_id",
    "ground_truth",
    "gt_",
    "iou",
    "localization_error",
    "reward",
    "next_",
    "future",
    "profile_label",
    "network_profile",
    "object_count",
    "dynamic_fraction",
    "temporal_information",
    "age_ns",
)


def assert_policy_features_exclude_forbidden_fields() -> None:
    """Fail closed if the frozen feature order ever gains a forbidden field.

    Called at import time so a future edit to
    :data:`POLICY_FEATURE_ORDER` cannot quietly introduce an identifier, a
    ground-truth term, the current decision's own outcome, an anchor lookup or a
    future measurement.
    """
    if len(set(POLICY_FEATURE_ORDER)) != len(POLICY_FEATURE_ORDER):
        raise StateRewardContractError(
            "POLICY_FEATURE_ORDER contains duplicate feature names"
        )
    for name in POLICY_FEATURE_ORDER:
        _non_empty_str(name, "policy feature name", StateRewardContractError)
        for forbidden in FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
            if forbidden in name:
                raise StateRewardContractError(
                    f"policy feature {name!r} contains the forbidden substring "
                    f"{forbidden!r}: identifiers, ground truth, the current "
                    f"decision's outcome, anchor lookups, measurement ages and "
                    f"future telemetry are not causal policy inputs"
                )


assert_policy_features_exclude_forbidden_fields()


# --------------------------------------------------------------------------- #
# Canonical schema descriptor
# --------------------------------------------------------------------------- #

SCHEMA_ID: str = "splitfusion_hybrid_sac_state_reward_transition_v1"
SCHEMA_VERSION: int = 1

SCHEMA_DESCRIPTOR: Mapping[str, Any] = _deep_freeze(
    {
        "schema_id": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "phase": (
            "pure in-memory causal-state, reward-measurement and "
            "replay-transition contract; no environment, storage, network, "
            "optimizer or live integration"
        ),
        "design_reference": (
            "DESIGN.md section 3 runtime control contract, section 4 minimal "
            "causal state, section 7 localization-prioritized perception "
            "quality, section 8 initial reward and terminal accounting, "
            "section 9 transaction identity and replay record"
        ),
        "action_space": (
            "(joint mode among 12 family-quantizer pairs, continuous q); the "
            "older four-AE fixed-quantizer recurrent sketch is superseded"
        ),
        "policy_class": "feed-forward; no recurrent memory in v1",
        "dependencies": {
            "action_catalog": {
                "schema": CATALOG_SCHEMA,
                "sha256": CATALOG_SHA256,
                "execution_mode": EXECUTION_MODE,
                "joint_mode_count": EXPECTED_MODE_COUNT,
                "q_e4_bounds": [Q_E4_MIN, Q_E4_MAX],
            },
            "executed_action_identity": {
                "schema_id": ACTION_IDENTITY_SCHEMA_ID,
                "schema_sha256": ACTION_IDENTITY_SCHEMA_SHA256,
            },
            "transaction_identity": {
                "schema_id": TRANSACTION_SCHEMA_ID,
                "schema_sha256": TRANSACTION_SCHEMA_SHA256,
                "schema_version": TRANSACTION_SCHEMA_VERSION,
                "minimum_hold_tensors": MINIMUM_HOLD_TENSORS,
            },
            "scene_descriptors": {
                "schema_id": SCENE_SCHEMA_ID,
                "schema_sha256": SCENE_SCHEMA_SHA256,
                "schema_version": SCENE_SCHEMA_VERSION,
                "descriptors": ["camera_si", "radar_p40"],
                "p40_horizon_m": P40_HORIZON_M,
            },
            "reward_ticket_controller": {
                "schema_id": CONTROLLER_SCHEMA_ID,
                "schema_sha256": CONTROLLER_SCHEMA_SHA256,
                "schema_version": CONTROLLER_SCHEMA_VERSION,
                "reward_deadline_ns": B_REWARD_DEADLINE_NS,
                "minimum_hold_tensors": K_MIN_TENSORS,
            },
        },
        "causal_state": {
            "rule": (
                "only information available before the new policy decision; no "
                "image or radar tensor, dynamic fraction, temporal-information "
                "descriptor, object count, ground truth, network-profile "
                "label, action_id anchor lookup or future measurement"
            ),
            "identifiers": (
                "session_uuid, tensor_seq and carla_frame_id are exact join "
                "and audit metadata only; they are never policy features and "
                "never reward terms"
            ),
            "missingness": (
                "missing, malformed or stale SI, P40, SNR, BSR or MCS fails "
                "closed for the external registered-fallback guard; zero is "
                "never substituted and no numeric sentinel is used"
            ),
            "previous_outcome": (
                "optionally absent only at episode start, carried as an "
                "explicit presence flag with independent quality and latency "
                "validity flags"
            ),
            "candidate_features": CANDIDATE_STATE_FEATURES,
            "candidate_feature_status": CANDIDATE_FEATURE_VALIDATION_REQUIRED,
            "policy_feature_order": POLICY_FEATURE_ORDER,
            "policy_feature_count": POLICY_FEATURE_COUNT,
            "forbidden_feature_substrings": FORBIDDEN_POLICY_FEATURE_SUBSTRINGS,
            "measurement_ages": (
                "retained in the state and enforced as an admission gate; "
                "deliberately not policy features in v1 because no age "
                "scaling constant has been measured"
            ),
            "normalization": (
                "every empirical scaling value is constructor supplied and "
                "carries a train-split identifier, fit population count and "
                "fit-config SHA-256; this module invents no fitted constant"
            ),
        },
        "quality": {
            "ground_truth_source": (
                "CARLA_GT_EXACT: exact CARLA ground truth, explicitly "
                "privileged and non-deployable, available only in the "
                "training/testbed instrument"
            ),
            "gt_privileged": True,
            "gt_deployable": False,
            "quality_ack": {
                "schema": QUALITY_ACK_SCHEMA,
                "failure_schema": QUALITY_ACK_FAILURE_SCHEMA,
                "protocol_version": QUALITY_ACK_PROTOCOL_VERSION,
                "source": QUALITY_ACK_SOURCE,
                "anchor_only": QUALITY_ACK_IS_ANCHOR_ONLY,
                "required_anchor_fields": QUALITY_ACK_REQUIRED_ANCHOR_FIELDS,
                "anchor_action_count": QUALITY_ACK_ANCHOR_ACTION_COUNT,
                "off_anchor_rule": (
                    "an off-anchor continuous q has action_id=None and is "
                    "REFUSED with OffAnchorQualityAckError; it is never snapped "
                    "to a nearest anchor"
                ),
                "core_evidence_key": (
                    "the canonical SHA-256 of the full ExecutedActionIdentity, "
                    "which is well defined on and off anchor"
                ),
                "protocol_v2_requirement": PROTOCOL_V2_REQUIREMENT,
                "bound_hashes": [
                    "raw_quality_ack_sha256",
                    "detailed_evidence_sha256",
                ],
                "timing_clock_domain": QUALITY_ACK_TIMING_CLOCK_DOMAIN,
                "false_positive_counts_available": (
                    QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE
                ),
                "derivable_detection_metrics": (
                    QUALITY_ACK_DERIVABLE_DETECTION_METRICS
                ),
                "underivable_detection_metrics": (
                    QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS
                ),
                "detection_claim_limit": (
                    "the v1 ACK carries tp and fn but no false positives, so "
                    "recall is derivable and precision, F1 and any "
                    "comprehensive detection-quality claim are NOT"
                ),
            },
            "undefined_class_mask": (
                "a class may be marked valid only with strictly positive "
                "ground-truth support, so a GT-absent class is masked as "
                "undefined and is never entered as a zero localization error "
                "nor rewarded as perfect recall"
            ),
            "deployment_claim": (
                "this schema does NOT claim that exact per-frame segmentation "
                "or localization accuracy exists in physical deployment"
            ),
            "localization_normalization": "exp(-error_m / tau_class_m)",
            "renormalization": (
                "Q_seg and Q_loc are each renormalized over the weights of "
                "their valid components only; an absent class is neither "
                "scored zero nor scored perfect"
            ),
            "top_level_mix": (
                "Q = beta * Q_seg + (1 - beta) * Q_loc with 0 <= beta < 0.5, "
                "so localization always carries the larger share; the "
                "top-level mix is NOT renormalized, so at least one valid "
                "segmentation component and one valid localization component "
                "are structurally required"
            ),
        },
        "reward": {
            "scalar": "r = w_quality * Q - w_latency * normalized_latency",
            "latency": (
                "L_ns = CompletedTicket.resolution_ns - "
                "CompletedTicket.opened_ns, derived from controller "
                "timestamps and never accepted from a caller; "
                "normalized_latency = L_ns / reward_deadline_ns"
            ),
            "latency_clock_domain": REWARD_LATENCY_CLOCK_DOMAIN,
            "latency_semantics": (
                "UE-local monotonic, from the policy decision that opened the "
                "ticket to exact feedback receipt at the UE"
            ),
            "forbidden_latency_sources": FORBIDDEN_REWARD_LATENCY_SOURCES,
            "constraints": [
                "c_deadline",
                "c_latency_excess = max(0, normalized_latency - 1)",
                "c_authoritative_failure",
            ],
            "constraint_note": (
                "the controller refuses feedback after the deadline, so on any "
                "feedback-resolved ticket normalized_latency <= 1 and "
                "c_latency_excess is structurally 0; a deadline violation "
                "appears only as FEEDBACK_TIMEOUT, whose latency excess is "
                "unmeasurable without a receipt and is therefore null"
            ),
            "terminal_handling": {
                TerminalClass.REWARD_FINAL_EXACT.value: (
                    "learning eligible; exact Q and L both required"
                ),
                TerminalClass.ACTION_PATH_FAILURE.value: (
                    "registered negative failure outcome; Q is never fabricated"
                ),
                TerminalClass.FEEDBACK_TIMEOUT.value: (
                    "censored pending authoritative reconciliation; not "
                    "automatically punished, because feedback-only control "
                    "loss and service failure are indistinguishable online"
                ),
                TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED.value: (
                    "excluded from learning; never converted into an agent "
                    "penalty"
                ),
            },
            "adjudication": (
                "a censored timeout becomes an authoritative negative only "
                "through an explicit AUTHORITATIVE_SERVICE_FAILURE "
                "adjudication record; a proven FEEDBACK_ONLY_LOSS never "
                "penalizes the action"
            ),
            "controller_dispositions": {
                terminal.value: disposition
                for terminal, disposition in TERMINAL_LEARNING_DISPOSITION.items()
            },
        },
        "transition": {
            "smdp": (
                "d = completed hold tensor count; discount multiplier = "
                "gamma_per_tensor ** d; both derived, never caller supplied"
            ),
            "invariants": [
                "exactly one session and one decision throughout",
                "the executed action equals every held tensor's action",
                "the state joins the ticket's reward-requested tensor and frame",
                "the next state is causally later and never crosses a session",
                "every quality-bearing transition binds raw_quality_ack_sha256 "
                "and detailed_evidence_sha256",
                "no scalar reward for a censored or excluded transition",
                "an infrastructure fault is never a failure reward",
                "canonical JSON serialization and SHA-256 are deterministic",
                "records are immutable or returned as defensive copies",
            ],
        },
    }
)

SCHEMA_SHA256: str = canonical_sha256(SCHEMA_DESCRIPTOR)


# --------------------------------------------------------------------------- #
# A. Raw causal state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PreviousOutcomeV1:
    """The previous *completed* decision, as seen before the next decision.

    Carries the exact catalog-reconciled
    :class:`~.transaction_identity.ExecutedActionIdentity`, the controller's
    terminal classification, and -- only when learning-valid -- the exact
    normalized quality and normalized action-to-feedback latency.

    Validity is always an explicit flag.  A field that is not valid is ``None``,
    never ``0.0`` and never any other sentinel.  Because only
    ``REWARD_FINAL_EXACT`` yields an exact Q and L, any other terminal class
    must carry both validity flags false; that is the exact agreement with
    :mod:`.reward_ticket_controller`.
    """

    action: ExecutedActionIdentity
    terminal_class: TerminalClass
    decision_seq: int
    quality_valid: bool
    latency_valid: bool
    quality_normalized: Optional[float] = None
    latency_normalized: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, ExecutedActionIdentity):
            raise CausalStateError(
                f"action must be a Phase-2 ExecutedActionIdentity, got "
                f"{type(self.action).__name__}"
            )
        # The previous action must still be catalog reconciled: a fabricated
        # identity cannot enter the state, and serialization would fail anyway.
        self.action.require_reconciled()
        if not isinstance(self.terminal_class, TerminalClass):
            raise CausalStateError(
                f"terminal_class must be a controller TerminalClass, got "
                f"{type(self.terminal_class).__name__}: {self.terminal_class!r}"
            )
        _non_negative_int(self.decision_seq, "decision_seq")
        _exact_bool(self.quality_valid, "quality_valid")
        _exact_bool(self.latency_valid, "latency_valid")

        if self.quality_valid:
            if self.quality_normalized is None:
                raise CausalStateError(
                    "quality_valid is true but quality_normalized is absent"
                )
            _finite_in(self.quality_normalized, "quality_normalized", 0.0, 1.0)
        elif self.quality_normalized is not None:
            raise CausalStateError(
                f"quality_valid is false, so quality_normalized must be None "
                f"rather than a sentinel; got {self.quality_normalized!r}"
            )

        if self.latency_valid:
            if self.latency_normalized is None:
                raise CausalStateError(
                    "latency_valid is true but latency_normalized is absent"
                )
            value = _finite_float(self.latency_normalized, "latency_normalized")
            if value < 0.0:
                raise CausalStateError(
                    f"latency_normalized must be >= 0, got {value}"
                )
        elif self.latency_normalized is not None:
            raise CausalStateError(
                f"latency_valid is false, so latency_normalized must be None "
                f"rather than a sentinel; got {self.latency_normalized!r}"
            )

        if self.terminal_class is not TerminalClass.REWARD_FINAL_EXACT and (
            self.quality_valid or self.latency_valid
        ):
            raise CausalStateError(
                f"terminal class {self.terminal_class.value} yields no exact "
                f"quality or latency, so both validity flags must be false; "
                f"got quality_valid={self.quality_valid}, "
                f"latency_valid={self.latency_valid}"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "decision_seq": self.decision_seq,
            "executed_action": self.action.to_canonical_dict(),
            "latency_normalized": (
                None
                if self.latency_normalized is None
                else float(self.latency_normalized)
            ),
            "latency_valid": self.latency_valid,
            "quality_normalized": (
                None
                if self.quality_normalized is None
                else float(self.quality_normalized)
            ),
            "quality_valid": self.quality_valid,
            "record": "previous_outcome_v1",
            "terminal_class": self.terminal_class.value,
        }


@dataclass(frozen=True, slots=True)
class CausalStateV1:
    """One immutable causal observation, complete and structurally valid.

    Structural validity is enforced here.  *Freshness* is a separate, explicitly
    parameterized decision: see :class:`StateFreshnessPolicyV1`, which
    :func:`build_policy_features` requires, so a stale state can never be
    vectorized for a policy.

    The identity block (``session_uuid``, ``tensor_seq``, ``carla_frame_id``,
    ``observed_ns``) and the provenance block (``scene_source_id``,
    ``network_source_id``) exist for exact joining and audit only.  They are
    excluded from :data:`POLICY_FEATURE_ORDER` by construction and by
    :func:`assert_policy_features_exclude_forbidden_fields`.
    """

    # -- scene (Phase-3 descriptors, already validated by their own record) -- #
    scene: SceneDescriptorSample
    scene_age_ns: int
    # -- UE-side radio telemetry ------------------------------------------- #
    achieved_snr_db: float
    bsr_bytes: int
    mcs_table_id: str
    mcs_index: int
    snr_age_ns: int
    bsr_age_ns: int
    mcs_age_ns: int
    # -- identity / provenance metadata ------------------------------------ #
    session_uuid: str
    observed_ns: int
    tensor_seq: int
    carla_frame_id: int
    scene_source_id: str
    network_source_id: str
    # -- previous completed decision, absent only at episode start --------- #
    previous: Optional[PreviousOutcomeV1] = None

    def __post_init__(self) -> None:
        if not isinstance(self.scene, SceneDescriptorSample):
            raise CausalStateError(
                f"scene must be a Phase-3 SceneDescriptorSample -- the only "
                f"admissible SI/P40 pair -- got {type(self.scene).__name__}"
            )
        # Restated so the fail-closed contract is visible here even though the
        # Phase-3 record already enforces it.
        _finite_float(self.scene.camera_si, "camera_si")
        if float(self.scene.camera_si) < 0.0:
            raise CausalStateError("camera_si cannot be negative")
        _finite_in(self.scene.radar_p40, "radar_p40", 0.0, 1.0)

        _non_negative_int(self.scene_age_ns, "scene_age_ns")
        _finite_float(self.achieved_snr_db, "achieved_snr_db")
        _non_negative_int(self.bsr_bytes, "bsr_bytes")
        _non_empty_str(self.mcs_table_id, "mcs_table_id")
        _non_negative_int(self.mcs_index, "mcs_index")
        _non_negative_int(self.snr_age_ns, "snr_age_ns")
        _non_negative_int(self.bsr_age_ns, "bsr_age_ns")
        _non_negative_int(self.mcs_age_ns, "mcs_age_ns")

        _canonical_uuid(self.session_uuid)
        _non_negative_int(self.observed_ns, "observed_ns")
        _non_negative_int(self.tensor_seq, "tensor_seq")
        _non_negative_int(self.carla_frame_id, "carla_frame_id")
        _non_empty_str(self.scene_source_id, "scene_source_id")
        _non_empty_str(self.network_source_id, "network_source_id")

        if self.previous is not None and not isinstance(
            self.previous, PreviousOutcomeV1
        ):
            raise CausalStateError(
                f"previous must be a PreviousOutcomeV1 or None (explicit "
                f"absence at episode start), got {type(self.previous).__name__}"
            )

    # -- derived ----------------------------------------------------------- #

    @property
    def has_previous_decision(self) -> bool:
        """Explicit presence of a previous completed decision."""
        return self.previous is not None

    @property
    def camera_si(self) -> float:
        return float(self.scene.camera_si)

    @property
    def radar_p40(self) -> float:
        return float(self.scene.radar_p40)

    @classmethod
    def metadata_field_names(cls) -> Tuple[str, ...]:
        """Fields that are join/audit metadata and never policy features."""
        return (
            "session_uuid",
            "observed_ns",
            "tensor_seq",
            "carla_frame_id",
            "scene_source_id",
            "network_source_id",
        )

    @classmethod
    def measurement_age_field_names(cls) -> Tuple[str, ...]:
        """Fields carrying measurement ages, used only as an admission gate."""
        return ("scene_age_ns", "snr_age_ns", "bsr_age_ns", "mcs_age_ns")

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "achieved_snr_db": float(self.achieved_snr_db),
            "bsr_age_ns": self.bsr_age_ns,
            "bsr_bytes": self.bsr_bytes,
            "camera_si": self.camera_si,
            "carla_frame_id": self.carla_frame_id,
            "mcs_age_ns": self.mcs_age_ns,
            "mcs_index": self.mcs_index,
            "mcs_table_id": self.mcs_table_id,
            "network_source_id": self.network_source_id,
            "observed_ns": self.observed_ns,
            "previous": (
                None if self.previous is None else self.previous.to_canonical_dict()
            ),
            "radar_p40": self.radar_p40,
            "record": "causal_state_v1",
            "scene_age_ns": self.scene_age_ns,
            "scene_schema_id": SCENE_SCHEMA_ID,
            "scene_schema_sha256": SCENE_SCHEMA_SHA256,
            "scene_source_id": self.scene_source_id,
            "session_uuid": self.session_uuid,
            "snr_age_ns": self.snr_age_ns,
            "tensor_seq": self.tensor_seq,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class StateFreshnessPolicyV1:
    """Explicit per-source maximum measurement ages.

    There is no default: every bound is an engineering decision that must be
    stated, and a state older than its bound fails closed for the external
    registered-fallback guard rather than being used or patched.
    """

    policy_id: str
    max_scene_age_ns: int
    max_snr_age_ns: int
    max_bsr_age_ns: int
    max_mcs_age_ns: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        _non_empty_str(self.policy_id, "policy_id", NormalizationSpecError)
        for name in (
            "max_scene_age_ns",
            "max_snr_age_ns",
            "max_bsr_age_ns",
            "max_mcs_age_ns",
        ):
            _positive_int(getattr(self, name), name, NormalizationSpecError)
        object.__setattr__(
            self,
            "provenance",
            _frozen_str_mapping(
                self.provenance, "provenance", NormalizationSpecError
            ),
        )

    def assert_fresh(self, state: CausalStateV1) -> None:
        """Fail closed when any required observation is too old.

        Raises:
            StaleTelemetryError: naming the offending source, so the caller's
                guard can select and log the registered fallback action.
        """
        if not isinstance(state, CausalStateV1):
            raise CausalStateError(
                f"state must be a CausalStateV1, got {type(state).__name__}"
            )
        for label, age, bound in (
            ("scene SI/P40", state.scene_age_ns, self.max_scene_age_ns),
            ("achieved SNR", state.snr_age_ns, self.max_snr_age_ns),
            ("BSR", state.bsr_age_ns, self.max_bsr_age_ns),
            ("MCS", state.mcs_age_ns, self.max_mcs_age_ns),
        ):
            if age > bound:
                raise StaleTelemetryError(
                    f"{label} measurement age {age} ns exceeds the registered "
                    f"bound {bound} ns of freshness policy "
                    f"{self.policy_id!r}; this state must not be used and the "
                    f"external guard must select the registered fallback "
                    f"action.  Nothing is substituted for the stale value"
                )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "max_bsr_age_ns": self.max_bsr_age_ns,
            "max_mcs_age_ns": self.max_mcs_age_ns,
            "max_scene_age_ns": self.max_scene_age_ns,
            "max_snr_age_ns": self.max_snr_age_ns,
            "policy_id": self.policy_id,
            "provenance": dict(self.provenance),
            "record": "state_freshness_policy_v1",
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class StateNormalizationSpecV1:
    """Explicitly provenanced scaling constants for the policy feature vector.

    **No fitted constant is invented here.**  Every empirical value is
    constructor supplied and the record refuses to exist without a train-split
    identifier, a positive fit population count and a fit-config SHA-256, so a
    feature vector can always be traced to the population its scaling came from.

    ``mcs_table_id`` is part of the spec, not only of the state: scaling an MCS
    index by the wrong table's maximum would be a silent unit error, so
    :func:`build_policy_features` requires the state's table id to match.
    """

    spec_id: str
    spec_version: int
    train_split_id: str
    fit_population_count: int
    fit_config_sha256: str
    camera_si_clip_min: float
    camera_si_clip_max: float
    achieved_snr_db_clip_min: float
    achieved_snr_db_clip_max: float
    bsr_log1p_scale: float
    mcs_table_id: str
    mcs_table_max_index: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        E = NormalizationSpecError
        _non_empty_str(self.spec_id, "spec_id", E)
        _positive_int(self.spec_version, "spec_version", E)
        _non_empty_str(self.train_split_id, "train_split_id", E)
        _positive_int(self.fit_population_count, "fit_population_count", E)
        _sha256_hex(self.fit_config_sha256, "fit_config_sha256", E)

        si_lo = _finite_float(self.camera_si_clip_min, "camera_si_clip_min", E)
        si_hi = _finite_float(self.camera_si_clip_max, "camera_si_clip_max", E)
        if si_lo < 0.0:
            raise E("camera_si_clip_min must be >= 0 (SI is non-negative)")
        if si_hi <= si_lo:
            raise E(
                f"camera_si_clip_max must exceed camera_si_clip_min; got "
                f"{si_hi} <= {si_lo}"
            )
        snr_lo = _finite_float(
            self.achieved_snr_db_clip_min, "achieved_snr_db_clip_min", E
        )
        snr_hi = _finite_float(
            self.achieved_snr_db_clip_max, "achieved_snr_db_clip_max", E
        )
        if snr_hi <= snr_lo:
            raise E(
                f"achieved_snr_db_clip_max must exceed "
                f"achieved_snr_db_clip_min; got {snr_hi} <= {snr_lo}"
            )
        scale = _finite_float(self.bsr_log1p_scale, "bsr_log1p_scale", E)
        if scale <= 0.0:
            raise E(f"bsr_log1p_scale must be > 0, got {scale}")
        _non_empty_str(self.mcs_table_id, "mcs_table_id", E)
        _positive_int(self.mcs_table_max_index, "mcs_table_max_index", E)
        object.__setattr__(
            self, "provenance", _frozen_str_mapping(self.provenance, "provenance", E)
        )

    # -- named scalar transformations -------------------------------------- #

    def scale_camera_si(self, camera_si: float) -> float:
        """Configured robust clip and scale of raw SI into [0, 1]."""
        lo = float(self.camera_si_clip_min)
        hi = float(self.camera_si_clip_max)
        clipped = min(max(float(camera_si), lo), hi)
        return (clipped - lo) / (hi - lo)

    def scale_achieved_snr_db(self, snr_db: float) -> float:
        """Configured clip and scale of achieved SNR into [0, 1]."""
        lo = float(self.achieved_snr_db_clip_min)
        hi = float(self.achieved_snr_db_clip_max)
        clipped = min(max(float(snr_db), lo), hi)
        return (clipped - lo) / (hi - lo)

    def scale_bsr_bytes(self, bsr_bytes: int) -> float:
        """``log1p`` of the buffer occupancy, then the configured scaling.

        The registered saturation is a clip to [0, 1]: ``bsr_log1p_scale`` is
        expected to be the fitted ``log1p`` of the population maximum, and a
        larger report saturates rather than leaving the unit interval.
        """
        value = math.log1p(float(bsr_bytes)) / float(self.bsr_log1p_scale)
        return min(max(value, 0.0), 1.0)

    def scale_mcs_index(self, mcs_index: int, mcs_table_id: str) -> float:
        """Scale an MCS index by the explicitly supplied table maximum."""
        if mcs_table_id != self.mcs_table_id:
            raise NormalizationSpecError(
                f"state MCS table {mcs_table_id!r} does not match the "
                f"normalization spec's table {self.mcs_table_id!r}; scaling an "
                f"index by another table's maximum would be a silent unit error"
            )
        if mcs_index > self.mcs_table_max_index:
            raise NormalizationSpecError(
                f"mcs_index {mcs_index} exceeds the declared maximum "
                f"{self.mcs_table_max_index} of table {self.mcs_table_id!r}"
            )
        return float(mcs_index) / float(self.mcs_table_max_index)

    @staticmethod
    def normalize_previous_q(action: ExecutedActionIdentity) -> float:
        """``q_e4 / 9800`` -- the registered wire value, not an anchor lookup."""
        return float(action.q_e4) / float(Q_E4_MAX)

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "achieved_snr_db_clip_max": float(self.achieved_snr_db_clip_max),
            "achieved_snr_db_clip_min": float(self.achieved_snr_db_clip_min),
            "bsr_log1p_scale": float(self.bsr_log1p_scale),
            "camera_si_clip_max": float(self.camera_si_clip_max),
            "camera_si_clip_min": float(self.camera_si_clip_min),
            "fit_config_sha256": self.fit_config_sha256,
            "fit_population_count": self.fit_population_count,
            "mcs_table_id": self.mcs_table_id,
            "mcs_table_max_index": self.mcs_table_max_index,
            "policy_feature_order": list(POLICY_FEATURE_ORDER),
            "provenance": dict(self.provenance),
            "record": "state_normalization_spec_v1",
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "train_split_id": self.train_split_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class PolicyFeatureVectorV1:
    """A fixed-width, deterministically ordered policy feature vector."""

    values: Tuple[float, ...]
    state_normalization_spec_sha256: str
    freshness_policy_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise StateRewardContractError(
                f"values must be a tuple, got {type(self.values).__name__}"
            )
        if len(self.values) != POLICY_FEATURE_COUNT:
            raise StateRewardContractError(
                f"expected exactly {POLICY_FEATURE_COUNT} features in the "
                f"frozen order, got {len(self.values)}"
            )
        for name, value in zip(POLICY_FEATURE_ORDER, self.values):
            if type(value) is not float or not math.isfinite(value):
                raise StateRewardContractError(
                    f"feature {name!r} must be a finite float, got "
                    f"{type(value).__name__}: {value!r}"
                )
        _sha256_hex(
            self.state_normalization_spec_sha256,
            "state_normalization_spec_sha256",
            StateRewardContractError,
        )
        _non_empty_str(
            self.freshness_policy_id,
            "freshness_policy_id",
            StateRewardContractError,
        )

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """The frozen order this vector was built in."""
        return POLICY_FEATURE_ORDER

    def as_tuple(self) -> Tuple[float, ...]:
        """The values, in frozen order.  Tuples are already immutable."""
        return self.values

    def as_mapping(self) -> Dict[str, float]:
        """A fresh defensive ``name -> value`` copy; mutating it is harmless."""
        return dict(zip(POLICY_FEATURE_ORDER, self.values))

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "feature_order": list(POLICY_FEATURE_ORDER),
            "freshness_policy_id": self.freshness_policy_id,
            "record": "policy_feature_vector_v1",
            "state_normalization_spec_sha256": (
                self.state_normalization_spec_sha256
            ),
            "values": [float(value) for value in self.values],
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


def build_policy_features(
    state: CausalStateV1,
    normalization: StateNormalizationSpecV1,
    freshness: StateFreshnessPolicyV1,
) -> PolicyFeatureVectorV1:
    """Map a fresh causal state into the frozen policy feature order.

    Freshness is checked first and unconditionally, so a stale observation can
    never reach a policy: there is no code path from a stale
    :class:`CausalStateV1` to a :class:`PolicyFeatureVectorV1`.

    When no previous decision exists, its feature block is written as zeros
    *behind explicit presence and validity masks*.  That is not a silent
    substitution: the masks are themselves features, so absence is
    representable and distinguishable from a real zero.
    """
    if not isinstance(state, CausalStateV1):
        raise CausalStateError(
            f"state must be a CausalStateV1, got {type(state).__name__}"
        )
    if not isinstance(normalization, StateNormalizationSpecV1):
        raise NormalizationSpecError(
            f"normalization must be a StateNormalizationSpecV1 with explicit "
            f"train-fit provenance, got {type(normalization).__name__}"
        )
    if not isinstance(freshness, StateFreshnessPolicyV1):
        raise NormalizationSpecError(
            f"freshness must be a StateFreshnessPolicyV1, got "
            f"{type(freshness).__name__}"
        )
    freshness.assert_fresh(state)

    named: Dict[str, float] = {
        "scene_camera_si_scaled": normalization.scale_camera_si(state.camera_si),
        "scene_radar_p40": float(state.radar_p40),
        "radio_achieved_snr_db_scaled": normalization.scale_achieved_snr_db(
            state.achieved_snr_db
        ),
        "radio_bsr_log1p_scaled": normalization.scale_bsr_bytes(state.bsr_bytes),
        "radio_mcs_index_scaled": normalization.scale_mcs_index(
            state.mcs_index, state.mcs_table_id
        ),
    }

    one_hot = [0.0] * _PREVIOUS_MODE_ONEHOT_WIDTH
    previous = state.previous
    if previous is not None:
        mode_id = previous.action.mode_id
        if not 0 <= mode_id < _PREVIOUS_MODE_ONEHOT_WIDTH:  # pragma: no cover
            raise CausalStateError(
                f"previous mode_id {mode_id} is outside the catalog's "
                f"{_PREVIOUS_MODE_ONEHOT_WIDTH} joint modes"
            )
        one_hot[mode_id] = 1.0
    for index, value in enumerate(one_hot):
        named[f"prev_joint_mode_onehot_{index:02d}"] = value

    named["prev_q_normalized"] = (
        0.0
        if previous is None
        else StateNormalizationSpecV1.normalize_previous_q(previous.action)
    )
    named["prev_quality_normalized"] = (
        0.0
        if previous is None or previous.quality_normalized is None
        else float(previous.quality_normalized)
    )
    named["prev_latency_normalized"] = (
        0.0
        if previous is None or previous.latency_normalized is None
        else float(previous.latency_normalized)
    )
    named["prev_present_mask"] = 0.0 if previous is None else 1.0
    named["prev_quality_valid_mask"] = (
        1.0 if previous is not None and previous.quality_valid else 0.0
    )
    named["prev_latency_valid_mask"] = (
        1.0 if previous is not None and previous.latency_valid else 0.0
    )

    missing = set(POLICY_FEATURE_ORDER) - set(named)
    extra = set(named) - set(POLICY_FEATURE_ORDER)
    if missing or extra:  # pragma: no cover - guarded by the frozen order
        raise StateRewardContractError(
            f"feature mapping disagrees with the frozen order; missing "
            f"{sorted(missing)}, unexpected {sorted(extra)}"
        )
    return PolicyFeatureVectorV1(
        values=tuple(float(named[name]) for name in POLICY_FEATURE_ORDER),
        state_normalization_spec_sha256=normalization.canonical_sha256(),
        freshness_policy_id=freshness.policy_id,
    )


# --------------------------------------------------------------------------- #
# B. Quality components and reward
# --------------------------------------------------------------------------- #


class GroundTruthSource(Enum):
    """The origin of a quality measurement, with its deployability marked.

    ``CARLA_GT_EXACT`` is the only source this phase admits.  It is a
    **privileged, non-deployable** simulator oracle: the edge evaluator reads it
    from CARLA, marks the wire record ``pg=True`` / ``dp=False``, and no
    physically deployed system has any equivalent.  Keeping it as a typed value
    rather than a free string means the non-deployability travels with every
    record that carries it.
    """

    CARLA_GT_EXACT = "CARLA_GT_EXACT"

    @property
    def privileged(self) -> bool:
        """True when the source is an instrument-only oracle."""
        return True

    @property
    def deployable(self) -> bool:
        """Always False: no deployed system observes exact per-frame accuracy."""
        return False


@dataclass(frozen=True, slots=True)
class QualityAckBindingV1:
    """Anchor-only binding to one ``sf_priv_quality_ack.v1`` evidence packet.

    Carries the two hashes that make a transition auditable back to its raw
    evidence: the canonical hash of the ACK datagram itself
    (``raw_quality_ack_sha256``) and the edge-retained detailed scientific row
    it hash-binds through its ``dh`` field (``detailed_evidence_sha256``).

    Construct through :meth:`for_executed_action`, which refuses an off-anchor
    action rather than approximating one.
    """

    raw_quality_ack_sha256: str
    detailed_evidence_sha256: str
    action_id: int
    profile_id: str
    evaluator_mode: str
    ack_schema: str = QUALITY_ACK_SCHEMA
    ack_protocol_version: int = QUALITY_ACK_PROTOCOL_VERSION
    ack_source: str = QUALITY_ACK_SOURCE

    def __post_init__(self) -> None:
        E = QualityContractError
        _sha256_hex(self.raw_quality_ack_sha256, "raw_quality_ack_sha256", E)
        _sha256_hex(self.detailed_evidence_sha256, "detailed_evidence_sha256", E)
        _non_negative_int(self.action_id, "action_id", E)
        if self.action_id >= QUALITY_ACK_ANCHOR_ACTION_COUNT:
            raise E(
                f"action_id {self.action_id} is outside the "
                f"{QUALITY_ACK_ANCHOR_ACTION_COUNT}-anchor catalog the v1 ACK "
                f"validates"
            )
        _non_empty_str(self.profile_id, "profile_id", E)
        _non_empty_str(self.evaluator_mode, "evaluator_mode", E)
        if self.ack_schema != QUALITY_ACK_SCHEMA:
            raise E(
                f"ack_schema must be {QUALITY_ACK_SCHEMA!r}, got "
                f"{self.ack_schema!r}"
            )
        if self.ack_protocol_version != QUALITY_ACK_PROTOCOL_VERSION:
            raise E(
                f"ack_protocol_version must be "
                f"{QUALITY_ACK_PROTOCOL_VERSION}, got "
                f"{self.ack_protocol_version!r}"
            )
        if self.ack_source != QUALITY_ACK_SOURCE:
            raise E(
                f"ack_source must be {QUALITY_ACK_SOURCE!r}, got "
                f"{self.ack_source!r}"
            )

    @classmethod
    def for_executed_action(
        cls,
        action: ExecutedActionIdentity,
        *,
        raw_quality_ack_sha256: str,
        detailed_evidence_sha256: str,
        evaluator_mode: str,
    ) -> "QualityAckBindingV1":
        """Bind an ACK to an executed action, or fail closed off-anchor.

        Raises:
            OffAnchorQualityAckError: if ``action`` is not exactly a registered
                anchor.  No nearest anchor is ever substituted.
        """
        if not isinstance(action, ExecutedActionIdentity):
            raise QualityContractError(
                f"action must be a Phase-2 ExecutedActionIdentity, got "
                f"{type(action).__name__}"
            )
        action.require_reconciled()
        if action.action_id is None or action.profile_id is None:
            raise OffAnchorQualityAckError(
                f"executed action {action.canonical_mode} q_e4={action.q_e4} is "
                f"not a registered anchor, so {QUALITY_ACK_SCHEMA} cannot "
                f"identify it: that wire contract requires "
                f"{list(QUALITY_ACK_REQUIRED_ANCHOR_FIELDS)} and validates "
                f"0 <= action_id < {QUALITY_ACK_ANCHOR_ACTION_COUNT}.  This "
                f"action is refused rather than snapped to a nearest anchor, "
                f"which would attribute another action's measured quality to "
                f"it.  {PROTOCOL_V2_REQUIREMENT}"
            )
        return cls(
            raw_quality_ack_sha256=raw_quality_ack_sha256,
            detailed_evidence_sha256=detailed_evidence_sha256,
            action_id=action.action_id,
            profile_id=action.profile_id,
            evaluator_mode=evaluator_mode,
        )

    def assert_matches_action(self, action: ExecutedActionIdentity) -> None:
        """Fail closed unless this ACK names exactly ``action``'s anchor."""
        if not isinstance(action, ExecutedActionIdentity):
            raise QualityContractError(
                f"action must be an ExecutedActionIdentity, got "
                f"{type(action).__name__}"
            )
        if action.action_id is None or action.profile_id is None:
            raise OffAnchorQualityAckError(
                f"executed action {action.canonical_mode} q_e4={action.q_e4} is "
                f"off-anchor and can never match an anchor-only v1 ACK.  "
                f"{PROTOCOL_V2_REQUIREMENT}"
            )
        if (action.action_id, action.profile_id) != (
            self.action_id,
            self.profile_id,
        ):
            raise QualityContractError(
                f"the ACK names anchor action_id={self.action_id} "
                f"profile_id={self.profile_id!r}, but the executed action is "
                f"action_id={action.action_id} "
                f"profile_id={action.profile_id!r}; evidence is never "
                f"re-attributed between actions"
            )

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "ack_protocol_version": self.ack_protocol_version,
            "ack_schema": self.ack_schema,
            "ack_source": self.ack_source,
            "action_id": self.action_id,
            "anchor_only": QUALITY_ACK_IS_ANCHOR_ONLY,
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "evaluator_mode": self.evaluator_mode,
            "false_positive_counts_available": (
                QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE
            ),
            "profile_id": self.profile_id,
            "raw_quality_ack_sha256": self.raw_quality_ack_sha256,
            "record": "quality_ack_binding_v1",
            "timing_clock_domain": QUALITY_ACK_TIMING_CLOCK_DOMAIN,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class QualityEvidenceV1:
    """Provenance of one quality measurement, keyed on the full action identity.

    The *core* key is ``executed_action_sha256``, the canonical hash of the
    whole :class:`~.transaction_identity.ExecutedActionIdentity`.  That is
    deliberate: it is well defined for every action, on-anchor or not, whereas
    the v1 ACK's anchor fields are not.  A future protocol-v2 ACK is expected to
    carry exactly this identity or this hash, which is what would let arbitrary
    continuous ``q`` be evaluated live.

    ``ack_binding`` is present only when the executed action happens to be one
    of the 72 registered anchors, because the v1 carrier cannot express anything
    else.
    """

    gt_source: GroundTruthSource
    executed_action_sha256: str
    gt_source_detail: str
    ack_binding: Optional[QualityAckBindingV1] = None

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.gt_source, GroundTruthSource):
            raise E(
                f"gt_source must be a GroundTruthSource (the privileged, "
                f"non-deployable oracle label), got "
                f"{type(self.gt_source).__name__}: {self.gt_source!r}"
            )
        _sha256_hex(self.executed_action_sha256, "executed_action_sha256", E)
        _non_empty_str(self.gt_source_detail, "gt_source_detail", E)
        if self.ack_binding is not None and not isinstance(
            self.ack_binding, QualityAckBindingV1
        ):
            raise E(
                f"ack_binding must be a QualityAckBindingV1 or None, got "
                f"{type(self.ack_binding).__name__}"
            )

    @classmethod
    def for_action(
        cls,
        action: ExecutedActionIdentity,
        *,
        gt_source_detail: str,
        ack_binding: Optional[QualityAckBindingV1] = None,
        gt_source: GroundTruthSource = GroundTruthSource.CARLA_GT_EXACT,
    ) -> "QualityEvidenceV1":
        """Key evidence on the full executed-action identity hash."""
        if not isinstance(action, ExecutedActionIdentity):
            raise QualityContractError(
                f"action must be a Phase-2 ExecutedActionIdentity, got "
                f"{type(action).__name__}"
            )
        action.require_reconciled()
        if ack_binding is not None:
            ack_binding.assert_matches_action(action)
        return cls(
            gt_source=gt_source,
            executed_action_sha256=action.canonical_sha256(),
            gt_source_detail=gt_source_detail,
            ack_binding=ack_binding,
        )

    @property
    def is_anchor_evidenced(self) -> bool:
        """True when a v1 anchor-only ACK could carry this action's evidence."""
        return self.ack_binding is not None

    @property
    def raw_quality_ack_sha256(self) -> Optional[str]:
        return (
            None if self.ack_binding is None
            else self.ack_binding.raw_quality_ack_sha256
        )

    @property
    def detailed_evidence_sha256(self) -> Optional[str]:
        return (
            None if self.ack_binding is None
            else self.ack_binding.detailed_evidence_sha256
        )

    @property
    def supports_detection_f1_claim(self) -> bool:
        """Always False: the v1 ACK reports TP and FN but no false positives.

        Recall is derivable; precision, F1 and any comprehensive detection
        quality are not, and must not be claimed from this evidence.
        """
        return QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "ack_binding": (
                None
                if self.ack_binding is None
                else self.ack_binding.to_canonical_dict()
            ),
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "executed_action_sha256": self.executed_action_sha256,
            "gt_deployable": self.gt_source.deployable,
            "gt_privileged": self.gt_source.privileged,
            "gt_source": self.gt_source.value,
            "gt_source_detail": self.gt_source_detail,
            "protocol_v2_requirement": PROTOCOL_V2_REQUIREMENT,
            "raw_quality_ack_sha256": self.raw_quality_ack_sha256,
            "record": "quality_evidence_v1",
            "supports_detection_f1_claim": self.supports_detection_f1_claim,
        }

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class QualityComponentsV1:
    """Raw exact CARLA ground-truth perception components for one decision.

    Four independent components with four independent validity flags:
    vehicle/person segmentation IoU and vehicle/person localization error.  A
    component that has no valid support is ``None`` and is *ignored* by
    renormalization -- it is neither scored zero nor scored perfect.

    ``evidence`` is mandatory: it carries the privileged, non-deployable
    ``CARLA_GT_EXACT`` label, the full executed-action identity hash these
    scores belong to, and -- for an anchor action -- the raw ACK and detailed
    evidence hashes.  These components exist only in the training/testbed
    instrument: a physically deployed system has no exact per-frame oracle.

    Ground-truth support is explicit.  ``vehicle_gt_support`` and
    ``person_gt_support`` are the class's GT instance counts (``tp + fn`` in ACK
    terms).  A class may only be marked valid when its support is **strictly
    positive**, so a GT-absent class is masked as undefined rather than entered
    as a zero localization error or scored as perfect recall.  See
    :attr:`undefined_classes`.

    Detection metrics: no false-positive count exists in the v1 evidence, so
    recall is the only derivable detection quantity and precision/F1 are not
    representable here at all.
    """

    evidence: QualityEvidenceV1
    vehicle_seg_valid: bool
    person_seg_valid: bool
    vehicle_loc_valid: bool
    person_loc_valid: bool
    vehicle_gt_support: Optional[int] = None
    person_gt_support: Optional[int] = None
    vehicle_seg_iou: Optional[float] = None
    person_seg_iou: Optional[float] = None
    vehicle_localization_error_m: Optional[float] = None
    person_localization_error_m: Optional[float] = None

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.evidence, QualityEvidenceV1):
            raise E(
                f"evidence must be a QualityEvidenceV1 carrying the privileged "
                f"ground-truth label and the executed-action identity hash, "
                f"got {type(self.evidence).__name__}"
            )
        for name in (
            "vehicle_seg_valid",
            "person_seg_valid",
            "vehicle_loc_valid",
            "person_loc_valid",
        ):
            _exact_bool(getattr(self, name), name, E)

        for flag, name, value in (
            (self.vehicle_seg_valid, "vehicle_seg_iou", self.vehicle_seg_iou),
            (self.person_seg_valid, "person_seg_iou", self.person_seg_iou),
        ):
            if flag:
                if value is None:
                    raise E(f"{name} support is valid but the value is absent")
                _finite_in(value, name, 0.0, 1.0, E)
            elif value is not None:
                raise E(
                    f"{name} support is invalid, so the value must be None "
                    f"rather than a sentinel; got {value!r}"
                )

        for flag, name, value in (
            (
                self.vehicle_loc_valid,
                "vehicle_localization_error_m",
                self.vehicle_localization_error_m,
            ),
            (
                self.person_loc_valid,
                "person_localization_error_m",
                self.person_localization_error_m,
            ),
        ):
            if flag:
                if value is None:
                    raise E(f"{name} support is valid but the value is absent")
                as_float = _finite_float(value, name, E)
                if as_float < 0.0:
                    raise E(f"{name} must be >= 0, got {as_float}")
            elif value is not None:
                raise E(
                    f"{name} support is invalid, so the value must be None "
                    f"rather than a sentinel; got {value!r}"
                )

        # -- undefined-class masking ---------------------------------------- #
        # A class with no ground-truth support is undefined, never perfect.  Any
        # valid component for a class therefore requires strictly positive GT
        # support, which is what stops a GT-absent localization from being
        # rewarded as a zero error or a full recall.
        for klass, support, has_valid in (
            (
                "vehicle",
                self.vehicle_gt_support,
                self.vehicle_seg_valid or self.vehicle_loc_valid,
            ),
            (
                "person",
                self.person_gt_support,
                self.person_seg_valid or self.person_loc_valid,
            ),
        ):
            if support is not None:
                _non_negative_int(support, f"{klass}_gt_support", E)
            if has_valid:
                if support is None:
                    raise UndefinedClassSupportError(
                        f"{klass} has a valid component but no declared "
                        f"{klass}_gt_support; a class without stated "
                        f"ground-truth support is undefined, not perfect"
                    )
                if support <= 0:
                    raise UndefinedClassSupportError(
                        f"{klass} declares {support} ground-truth instances yet "
                        f"marks a component valid; a GT-absent class must be "
                        f"masked as undefined and must never be scored as a "
                        f"zero localization error or as perfect recall"
                    )

    @property
    def undefined_classes(self) -> Tuple[str, ...]:
        """Classes masked out for lack of any valid, GT-supported component."""
        undefined = []
        if not (self.vehicle_seg_valid or self.vehicle_loc_valid):
            undefined.append("vehicle")
        if not (self.person_seg_valid or self.person_loc_valid):
            undefined.append("person")
        return tuple(undefined)

    @property
    def gt_source(self) -> GroundTruthSource:
        """The privileged, non-deployable ground-truth label."""
        return self.evidence.gt_source

    @property
    def supports_detection_f1_claim(self) -> bool:
        """Always False: the v1 evidence has no false-positive counts."""
        return self.evidence.supports_detection_f1_claim

    @property
    def valid_component_count(self) -> int:
        return sum(
            (
                self.vehicle_seg_valid,
                self.person_seg_valid,
                self.vehicle_loc_valid,
                self.person_loc_valid,
            )
        )

    @property
    def has_segmentation_support(self) -> bool:
        return self.vehicle_seg_valid or self.person_seg_valid

    @property
    def has_localization_support(self) -> bool:
        return self.vehicle_loc_valid or self.person_loc_valid

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "evidence": self.evidence.to_canonical_dict(),
            "person_gt_support": self.person_gt_support,
            "supports_detection_f1_claim": self.supports_detection_f1_claim,
            "undefined_classes": list(self.undefined_classes),
            "vehicle_gt_support": self.vehicle_gt_support,
            "person_localization_error_m": (
                None
                if self.person_localization_error_m is None
                else float(self.person_localization_error_m)
            ),
            "person_loc_valid": self.person_loc_valid,
            "person_seg_iou": (
                None if self.person_seg_iou is None else float(self.person_seg_iou)
            ),
            "person_seg_valid": self.person_seg_valid,
            "record": "quality_components_v1",
            "vehicle_localization_error_m": (
                None
                if self.vehicle_localization_error_m is None
                else float(self.vehicle_localization_error_m)
            ),
            "vehicle_loc_valid": self.vehicle_loc_valid,
            "vehicle_seg_iou": (
                None
                if self.vehicle_seg_iou is None
                else float(self.vehicle_seg_iou)
            ),
            "vehicle_seg_valid": self.vehicle_seg_valid,
        }


@dataclass(frozen=True, slots=True)
class QualityEvaluationV1:
    """The derived scalar quality, with every raw input preserved beside it."""

    components: QualityComponentsV1
    q_seg: float
    q_loc: float
    quality: float
    loc_vehicle_normalized: Optional[float]
    loc_person_normalized: Optional[float]
    segmentation_weights_used: Mapping[str, float]
    localization_weights_used: Mapping[str, float]
    reward_spec_sha256: str

    def __post_init__(self) -> None:
        E = QualityContractError
        if not isinstance(self.components, QualityComponentsV1):
            raise E(
                f"components must be a QualityComponentsV1, got "
                f"{type(self.components).__name__}"
            )
        _finite_in(self.q_seg, "q_seg", 0.0, 1.0, E)
        _finite_in(self.q_loc, "q_loc", 0.0, 1.0, E)
        _finite_in(self.quality, "quality", 0.0, 1.0, E)
        for name in ("loc_vehicle_normalized", "loc_person_normalized"):
            value = getattr(self, name)
            if value is not None:
                _finite_in(value, name, 0.0, 1.0, E)
        for name in ("segmentation_weights_used", "localization_weights_used"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise E(f"{name} must be a non-empty mapping")
            object.__setattr__(
                self, name, MappingProxyType({k: float(v) for k, v in value.items()})
            )
        _sha256_hex(self.reward_spec_sha256, "reward_spec_sha256", E)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "components": self.components.to_canonical_dict(),
            "localization_weights_used": dict(self.localization_weights_used),
            "loc_person_normalized": (
                None
                if self.loc_person_normalized is None
                else float(self.loc_person_normalized)
            ),
            "loc_vehicle_normalized": (
                None
                if self.loc_vehicle_normalized is None
                else float(self.loc_vehicle_normalized)
            ),
            "q_loc": float(self.q_loc),
            "q_seg": float(self.q_seg),
            "quality": float(self.quality),
            "record": "quality_evaluation_v1",
            "reward_spec_sha256": self.reward_spec_sha256,
            "segmentation_weights_used": dict(self.segmentation_weights_used),
        }


@dataclass(frozen=True, slots=True)
class LatencyMeasurementV1:
    """Reward latency: UE-local monotonic policy decision to feedback receipt.

    Both endpoints come from the *same* injected monotonic source inside
    :class:`~.reward_ticket_controller.CompletedTicket`: ``opened_ns`` is the
    policy decision that opened the ticket and ``resolution_ns`` is the exact
    feedback receipt at the UE.  The clock domain is recorded explicitly as
    :data:`REWARD_LATENCY_CLOCK_DOMAIN`.

    This is deliberately **not** the map-install ACK latency, and it never mixes
    clock domains: every timing field of ``sf_priv_quality_ack.v1`` is a *wall*
    clock reading and none of them may enter this measurement.  See
    :data:`FORBIDDEN_REWARD_LATENCY_SOURCES`.
    """

    opened_ns: int
    resolution_ns: int
    l_ns: int
    normalized_latency: float
    deadline_ns: int

    def __post_init__(self) -> None:
        E = StateRewardContractError
        _non_negative_int(self.opened_ns, "opened_ns", E)
        _non_negative_int(self.resolution_ns, "resolution_ns", E)
        _non_negative_int(self.l_ns, "l_ns", E)
        _non_negative_int(self.deadline_ns, "deadline_ns", E)
        if self.l_ns != self.resolution_ns - self.opened_ns:
            raise E(
                f"l_ns {self.l_ns} must equal resolution_ns - opened_ns "
                f"({self.resolution_ns} - {self.opened_ns})"
            )
        expected = float(self.l_ns) / float(B_REWARD_DEADLINE_NS)
        if float(self.normalized_latency) != expected:
            raise E(
                f"normalized_latency must be exactly l_ns / "
                f"{B_REWARD_DEADLINE_NS}; expected {expected}, got "
                f"{self.normalized_latency}"
            )

    @classmethod
    def from_completed_ticket(cls, ticket: CompletedTicket) -> "LatencyMeasurementV1":
        """Derive L exactly from the frozen ticket; never accept a caller's L.

        Raises:
            StateRewardContractError: if the ticket has no exact feedback
                receipt, which is precisely the censored/excluded case.
        """
        if not isinstance(ticket, CompletedTicket):
            raise StateRewardContractError(
                f"ticket must be a controller CompletedTicket, got "
                f"{type(ticket).__name__}"
            )
        if ticket.resolution_ns is None:
            raise StateRewardContractError(
                f"decision {ticket.decision_seq} closed as "
                f"{ticket.terminal_class.value} with no exact feedback receipt, "
                f"so no action-to-feedback latency exists; a latency must never "
                f"be imputed for a censored or excluded transition"
            )
        l_ns = ticket.resolution_ns - ticket.opened_ns
        return cls(
            opened_ns=ticket.opened_ns,
            resolution_ns=ticket.resolution_ns,
            l_ns=l_ns,
            normalized_latency=float(l_ns) / float(B_REWARD_DEADLINE_NS),
            deadline_ns=ticket.deadline_ns,
        )

    @property
    def clock_domain(self) -> str:
        """``UE_LOCAL_MONOTONIC``: one monotonic source, never a wall clock."""
        return REWARD_LATENCY_CLOCK_DOMAIN

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "clock_domain": self.clock_domain,
            "deadline_ns": self.deadline_ns,
            "forbidden_sources": list(FORBIDDEN_REWARD_LATENCY_SOURCES),
            "l_ns": self.l_ns,
            "normalized_latency": float(self.normalized_latency),
            "opened_ns": self.opened_ns,
            "record": "latency_measurement_v1",
            "reward_deadline_ns": B_REWARD_DEADLINE_NS,
            "resolution_ns": self.resolution_ns,
        }


@dataclass(frozen=True, slots=True)
class RewardSpecV1:
    """Every scientific reward value, all constructor supplied.

    There is **no production default** for any weight, time constant, mixing
    coefficient or discount: each is a calibration hypothesis, and baking one in
    here would present an unmeasured choice as a frozen contract.

    ``segmentation_mix_beta`` is constrained to ``[0, 0.5)`` so the
    localization share ``1 - beta`` is always strictly larger, which is the
    localization-prioritized quality of DESIGN.md section 7.
    """

    spec_id: str
    spec_version: int
    w_seg_person: float
    w_seg_vehicle: float
    w_loc_person: float
    w_loc_vehicle: float
    tau_person_m: float
    tau_vehicle_m: float
    segmentation_mix_beta: float
    w_quality: float
    w_latency: float
    r_registered_failure: float
    gamma_per_tensor: float
    min_valid_quality_components: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        E = RewardSpecError
        _non_empty_str(self.spec_id, "spec_id", E)
        _positive_int(self.spec_version, "spec_version", E)
        for name in (
            "w_seg_person",
            "w_seg_vehicle",
            "w_loc_person",
            "w_loc_vehicle",
            "tau_person_m",
            "tau_vehicle_m",
            "w_quality",
        ):
            value = _finite_float(getattr(self, name), name, E)
            if value <= 0.0:
                raise E(f"{name} must be > 0, got {value}")
        beta = _finite_float(self.segmentation_mix_beta, "segmentation_mix_beta", E)
        if not 0.0 <= beta < 0.5:
            raise E(
                f"segmentation_mix_beta must satisfy 0 <= beta < 0.5 so that "
                f"localization keeps the larger top-level share; got {beta}"
            )
        w_latency = _finite_float(self.w_latency, "w_latency", E)
        if w_latency < 0.0:
            raise E(f"w_latency must be >= 0, got {w_latency}")
        failure = _finite_float(self.r_registered_failure, "r_registered_failure", E)
        if failure > 0.0:
            raise E(
                f"r_registered_failure is the registered *negative* service "
                f"outcome and must be <= 0, got {failure}"
            )
        gamma = _finite_float(self.gamma_per_tensor, "gamma_per_tensor", E)
        if not 0.0 < gamma <= 1.0:
            raise E(f"gamma_per_tensor must lie in (0, 1], got {gamma}")
        minimum = _positive_int(
            self.min_valid_quality_components, "min_valid_quality_components", E
        )
        if minimum > 4:
            raise E(
                f"min_valid_quality_components cannot exceed the four "
                f"registered components, got {minimum}"
            )
        object.__setattr__(
            self, "provenance", _frozen_str_mapping(self.provenance, "provenance", E)
        )

    @property
    def localization_mix(self) -> float:
        """``1 - beta``: always strictly greater than the segmentation share."""
        return 1.0 - float(self.segmentation_mix_beta)

    # -- localization normalization ---------------------------------------- #

    def normalize_localization(self, error_m: float, tau_m: float) -> float:
        """``exp(-error_m / tau_m)``: lower error becomes strictly higher quality."""
        value = math.exp(-float(error_m) / float(tau_m))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise QualityContractError(  # pragma: no cover - defensive
                f"normalized localization produced invalid value {value!r}"
            )
        return value

    # -- quality ----------------------------------------------------------- #

    def evaluate_quality(
        self, components: QualityComponentsV1
    ) -> QualityEvaluationV1:
        """Compute Q, renormalizing each family over its valid weights only.

        Raises:
            InsufficientQualitySupportError: if fewer than
                ``min_valid_quality_components`` components are valid, or if
                either family has no valid component at all.  The top-level mix
                ``beta * Q_seg + (1 - beta) * Q_loc`` is deliberately *not*
                renormalized, so both families are structurally required; an
                absent family is never silently folded away.
        """
        if not isinstance(components, QualityComponentsV1):
            raise QualityContractError(
                f"components must be a QualityComponentsV1, got "
                f"{type(components).__name__}"
            )
        if components.valid_component_count < self.min_valid_quality_components:
            raise InsufficientQualitySupportError(
                f"quality needs at least {self.min_valid_quality_components} "
                f"valid components under spec {self.spec_id!r}; only "
                f"{components.valid_component_count} are valid"
            )
        if not components.has_segmentation_support:
            raise InsufficientQualitySupportError(
                "no valid segmentation component, so Q_seg is undefined; the "
                "top-level mix is not renormalized and an absent family is "
                "never treated as zero or perfect"
            )
        if not components.has_localization_support:
            raise InsufficientQualitySupportError(
                "no valid localization component, so Q_loc is undefined; the "
                "top-level mix is not renormalized and an absent family is "
                "never treated as zero or perfect"
            )

        seg_terms: Dict[str, Tuple[float, float]] = {}
        if components.person_seg_valid:
            seg_terms["person"] = (
                float(self.w_seg_person),
                float(components.person_seg_iou),
            )
        if components.vehicle_seg_valid:
            seg_terms["vehicle"] = (
                float(self.w_seg_vehicle),
                float(components.vehicle_seg_iou),
            )

        loc_person: Optional[float] = None
        loc_vehicle: Optional[float] = None
        loc_terms: Dict[str, Tuple[float, float]] = {}
        if components.person_loc_valid:
            loc_person = self.normalize_localization(
                components.person_localization_error_m, self.tau_person_m
            )
            loc_terms["person"] = (float(self.w_loc_person), loc_person)
        if components.vehicle_loc_valid:
            loc_vehicle = self.normalize_localization(
                components.vehicle_localization_error_m, self.tau_vehicle_m
            )
            loc_terms["vehicle"] = (float(self.w_loc_vehicle), loc_vehicle)

        def _renormalized(terms: Mapping[str, Tuple[float, float]]) -> float:
            total_weight = sum(weight for weight, _ in terms.values())
            return sum(weight * value for weight, value in terms.values()) / (
                total_weight
            )

        q_seg = _renormalized(seg_terms)
        q_loc = _renormalized(loc_terms)
        beta = float(self.segmentation_mix_beta)
        quality = beta * q_seg + self.localization_mix * q_loc
        return QualityEvaluationV1(
            components=components,
            q_seg=q_seg,
            q_loc=q_loc,
            quality=quality,
            loc_vehicle_normalized=loc_vehicle,
            loc_person_normalized=loc_person,
            segmentation_weights_used={
                name: weight for name, (weight, _) in seg_terms.items()
            },
            localization_weights_used={
                name: weight for name, (weight, _) in loc_terms.items()
            },
            reward_spec_sha256=self.canonical_sha256(),
        )

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "gamma_per_tensor": float(self.gamma_per_tensor),
            "localization_mix": self.localization_mix,
            "min_valid_quality_components": self.min_valid_quality_components,
            "provenance": dict(self.provenance),
            "r_registered_failure": float(self.r_registered_failure),
            "record": "reward_spec_v1",
            "reward_deadline_ns": B_REWARD_DEADLINE_NS,
            "segmentation_mix_beta": float(self.segmentation_mix_beta),
            "spec_id": self.spec_id,
            "spec_version": self.spec_version,
            "tau_person_m": float(self.tau_person_m),
            "tau_vehicle_m": float(self.tau_vehicle_m),
            "w_latency": float(self.w_latency),
            "w_loc_person": float(self.w_loc_person),
            "w_loc_vehicle": float(self.w_loc_vehicle),
            "w_quality": float(self.w_quality),
            "w_seg_person": float(self.w_seg_person),
            "w_seg_vehicle": float(self.w_seg_vehicle),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


class Adjudication(Enum):
    """Authoritative post-run verdict on a censored deadline expiry.

    Only ``AUTHORITATIVE_SERVICE_FAILURE`` may turn a censored timeout into a
    negative reward, and only ever through an explicit
    :class:`AdjudicationRecordV1`.  ``FEEDBACK_ONLY_LOSS`` is a proven
    control-plane miss and must never penalize the action.
    """

    PENDING = "PENDING"
    FEEDBACK_ONLY_LOSS = "FEEDBACK_ONLY_LOSS"
    AUTHORITATIVE_SERVICE_FAILURE = "AUTHORITATIVE_SERVICE_FAILURE"
    INFRASTRUCTURE_FAULT = "INFRASTRUCTURE_FAULT"


@dataclass(frozen=True, slots=True)
class AdjudicationRecordV1:
    """An explicit, attributed reconciliation verdict with its evidence."""

    verdict: Adjudication
    adjudicator_id: str
    evidence_sha256: str
    adjudicated_ns: int
    detail: str

    def __post_init__(self) -> None:
        E = AdjudicationError
        if not isinstance(self.verdict, Adjudication):
            raise E(
                f"verdict must be an Adjudication, got "
                f"{type(self.verdict).__name__}: {self.verdict!r}"
            )
        _non_empty_str(self.adjudicator_id, "adjudicator_id", E)
        _sha256_hex(self.evidence_sha256, "evidence_sha256", E)
        _non_negative_int(self.adjudicated_ns, "adjudicated_ns", E)
        _non_empty_str(self.detail, "detail", E)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "adjudicated_ns": self.adjudicated_ns,
            "adjudicator_id": self.adjudicator_id,
            "detail": self.detail,
            "evidence_sha256": self.evidence_sha256,
            "record": "adjudication_record_v1",
            "verdict": self.verdict.value,
        }


class LearningEligibility(Enum):
    """Whether and why a transition may contribute to learning."""

    ELIGIBLE = "ELIGIBLE"
    CENSORED_PENDING_ADJUDICATION = "CENSORED_PENDING_ADJUDICATION"
    CENSORED_FEEDBACK_ONLY_LOSS = "CENSORED_FEEDBACK_ONLY_LOSS"
    EXCLUDED_INFRASTRUCTURE_FAULT = "EXCLUDED_INFRASTRUCTURE_FAULT"


@dataclass(frozen=True, slots=True)
class ConstraintCostsV1:
    """Constraint signals kept separate from the scalar reward.

    ``None`` means *unmeasured*, not zero.  On any feedback-resolved ticket the
    controller has already refused post-deadline feedback, so
    ``normalized_latency <= 1`` and ``c_latency_excess`` is structurally ``0``;
    a real deadline violation appears only as a censored ``FEEDBACK_TIMEOUT``,
    whose excess magnitude cannot be measured without a receipt.
    """

    c_deadline: Optional[float]
    c_latency_excess: Optional[float]
    c_authoritative_failure: Optional[float]

    def __post_init__(self) -> None:
        E = StateRewardContractError
        for name in ("c_deadline", "c_authoritative_failure"):
            value = getattr(self, name)
            if value is not None:
                as_float = _finite_float(value, name, E)
                if as_float not in (0.0, 1.0):
                    raise E(f"{name} is an indicator in {{0, 1}}, got {as_float}")
        if self.c_latency_excess is not None:
            value = _finite_float(self.c_latency_excess, "c_latency_excess", E)
            if value < 0.0:
                raise E(f"c_latency_excess must be >= 0, got {value}")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "c_authoritative_failure": (
                None
                if self.c_authoritative_failure is None
                else float(self.c_authoritative_failure)
            ),
            "c_deadline": (
                None if self.c_deadline is None else float(self.c_deadline)
            ),
            "c_latency_excess": (
                None
                if self.c_latency_excess is None
                else float(self.c_latency_excess)
            ),
            "record": "constraint_costs_v1",
        }


@dataclass(frozen=True, slots=True)
class DecisionOutcomeV1:
    """The complete measured outcome of one completed decision."""

    terminal_class: TerminalClass
    eligibility: LearningEligibility
    costs: ConstraintCostsV1
    reward_spec_sha256: str
    quality: Optional[QualityEvaluationV1] = None
    latency: Optional[LatencyMeasurementV1] = None
    scalar_reward: Optional[float] = None
    adjudication: Optional[AdjudicationRecordV1] = None

    def __post_init__(self) -> None:
        E = StateRewardContractError
        if not isinstance(self.terminal_class, TerminalClass):
            raise E("terminal_class must be a controller TerminalClass")
        if not isinstance(self.eligibility, LearningEligibility):
            raise E("eligibility must be a LearningEligibility")
        if not isinstance(self.costs, ConstraintCostsV1):
            raise E("costs must be a ConstraintCostsV1")
        _sha256_hex(self.reward_spec_sha256, "reward_spec_sha256", E)
        if self.scalar_reward is not None:
            _finite_float(self.scalar_reward, "scalar_reward", E)
        if self.eligibility is not LearningEligibility.ELIGIBLE and (
            self.scalar_reward is not None
        ):
            raise E(
                f"eligibility {self.eligibility.value} carries no scalar "
                f"reward; got {self.scalar_reward!r}"
            )
        if self.eligibility is LearningEligibility.ELIGIBLE and (
            self.scalar_reward is None
        ):
            raise E("an eligible outcome must carry a scalar reward")
        if self.terminal_class is TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
            if self.eligibility is not (
                LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT
            ):
                raise E(
                    "an infrastructure fault must be excluded, never scored"
                )
            if self.costs.c_authoritative_failure == 1.0:
                raise E(
                    "an infrastructure fault must never be recorded as an "
                    "authoritative service failure"
                )

    @property
    def learning_eligible(self) -> bool:
        return self.eligibility is LearningEligibility.ELIGIBLE

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "adjudication": (
                None
                if self.adjudication is None
                else self.adjudication.to_canonical_dict()
            ),
            "costs": self.costs.to_canonical_dict(),
            "eligibility": self.eligibility.value,
            "latency": (
                None if self.latency is None else self.latency.to_canonical_dict()
            ),
            "quality": (
                None if self.quality is None else self.quality.to_canonical_dict()
            ),
            "record": "decision_outcome_v1",
            "reward_spec_sha256": self.reward_spec_sha256,
            "scalar_reward": (
                None if self.scalar_reward is None else float(self.scalar_reward)
            ),
            "terminal_class": self.terminal_class.value,
        }


def evaluate_completed_decision(
    ticket: CompletedTicket,
    reward_spec: RewardSpecV1,
    *,
    quality_components: Optional[QualityComponentsV1] = None,
    adjudication: Optional[AdjudicationRecordV1] = None,
) -> DecisionOutcomeV1:
    """Measure one completed decision exactly as the controller classified it.

    Terminal handling agrees with :mod:`.reward_ticket_controller` term for
    term:

    * ``REWARD_FINAL_EXACT`` -- learning eligible; exact Q *and* L are both
      required, and ``r = w_quality * Q - w_latency * normalized_latency``.
    * ``ACTION_PATH_FAILURE`` -- the registered negative service outcome; Q is
      never fabricated, so supplying quality components is an error.
    * ``FEEDBACK_TIMEOUT`` -- censored pending authoritative reconciliation.  It
      is *not* automatically punished, because feedback-only control loss and a
      real service failure are indistinguishable online.  It becomes a negative
      only through an explicit ``AUTHORITATIVE_SERVICE_FAILURE`` adjudication.
    * ``INFRASTRUCTURE_FAULT_EXCLUDED`` -- excluded, never a penalty.

    L is always derived from the ticket's own timestamps and is never accepted
    from the caller.
    """
    if not isinstance(ticket, CompletedTicket):
        raise StateRewardContractError(
            f"ticket must be a controller CompletedTicket, got "
            f"{type(ticket).__name__}"
        )
    if not isinstance(reward_spec, RewardSpecV1):
        raise RewardSpecError(
            f"reward_spec must be a RewardSpecV1 with explicit values, got "
            f"{type(reward_spec).__name__}"
        )
    if quality_components is not None and not isinstance(
        quality_components, QualityComponentsV1
    ):
        raise QualityContractError(
            f"quality_components must be a QualityComponentsV1 or None, got "
            f"{type(quality_components).__name__}"
        )
    if adjudication is not None and not isinstance(
        adjudication, AdjudicationRecordV1
    ):
        raise AdjudicationError(
            f"adjudication must be an AdjudicationRecordV1 or None, got "
            f"{type(adjudication).__name__}"
        )
    terminal = ticket.terminal_class
    spec_sha = reward_spec.canonical_sha256()

    if adjudication is not None and terminal is not TerminalClass.FEEDBACK_TIMEOUT:
        raise AdjudicationError(
            f"an adjudication record applies only to a censored "
            f"{TerminalClass.FEEDBACK_TIMEOUT.value}; decision "
            f"{ticket.decision_seq} closed as {terminal.value}, whose "
            f"classification is already authoritative"
        )

    # -- exact feedback: the only learning-eligible perception outcome ----- #
    if terminal is TerminalClass.REWARD_FINAL_EXACT:
        if quality_components is None:
            raise QualityContractError(
                f"{terminal.value} is learning eligible and requires exact "
                f"quality components; none were supplied for decision "
                f"{ticket.decision_seq}"
            )
        quality = reward_spec.evaluate_quality(quality_components)
        latency = LatencyMeasurementV1.from_completed_ticket(ticket)
        reward = float(reward_spec.w_quality) * quality.quality - float(
            reward_spec.w_latency
        ) * latency.normalized_latency
        excess = max(0.0, latency.normalized_latency - 1.0)
        return DecisionOutcomeV1(
            terminal_class=terminal,
            eligibility=LearningEligibility.ELIGIBLE,
            costs=ConstraintCostsV1(
                c_deadline=1.0 if latency.normalized_latency > 1.0 else 0.0,
                c_latency_excess=excess,
                c_authoritative_failure=0.0,
            ),
            reward_spec_sha256=spec_sha,
            quality=quality,
            latency=latency,
            scalar_reward=reward,
        )

    # -- proven action-path failure: registered negative, no fabricated Q -- #
    if terminal is TerminalClass.ACTION_PATH_FAILURE:
        if quality_components is not None:
            raise QualityContractError(
                f"{terminal.value} receives the registered negative service "
                f"outcome and must not carry a fabricated quality; quality "
                f"components were supplied for decision {ticket.decision_seq}"
            )
        latency = LatencyMeasurementV1.from_completed_ticket(ticket)
        return DecisionOutcomeV1(
            terminal_class=terminal,
            eligibility=LearningEligibility.ELIGIBLE,
            costs=ConstraintCostsV1(
                c_deadline=1.0 if latency.normalized_latency > 1.0 else 0.0,
                c_latency_excess=max(0.0, latency.normalized_latency - 1.0),
                c_authoritative_failure=1.0,
            ),
            reward_spec_sha256=spec_sha,
            quality=None,
            latency=latency,
            scalar_reward=float(reward_spec.r_registered_failure),
        )

    # -- excluded instrument fault ----------------------------------------- #
    if terminal is TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
        if quality_components is not None:
            raise QualityContractError(
                f"{terminal.value} is excluded from learning and carries no "
                f"quality; components were supplied for decision "
                f"{ticket.decision_seq}"
            )
        return DecisionOutcomeV1(
            terminal_class=terminal,
            eligibility=LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
            costs=ConstraintCostsV1(
                c_deadline=None,
                c_latency_excess=None,
                c_authoritative_failure=None,
            ),
            reward_spec_sha256=spec_sha,
            quality=None,
            latency=None,
            scalar_reward=None,
        )

    # -- censored deadline expiry ------------------------------------------ #
    if terminal is not TerminalClass.FEEDBACK_TIMEOUT:  # pragma: no cover
        raise StateRewardContractError(
            f"unhandled terminal class {terminal.value}; every controller "
            f"terminal class must have exactly one documented treatment"
        )
    if quality_components is not None:
        raise QualityContractError(
            f"{terminal.value} received no exact feedback, so no quality "
            f"exists; components were supplied for decision "
            f"{ticket.decision_seq}"
        )
    verdict = Adjudication.PENDING if adjudication is None else adjudication.verdict

    if verdict is Adjudication.AUTHORITATIVE_SERVICE_FAILURE:
        # The only path from a censored timeout to a negative reward, and only
        # with an attributed evidence record.
        return DecisionOutcomeV1(
            terminal_class=terminal,
            eligibility=LearningEligibility.ELIGIBLE,
            costs=ConstraintCostsV1(
                c_deadline=1.0,
                c_latency_excess=None,
                c_authoritative_failure=1.0,
            ),
            reward_spec_sha256=spec_sha,
            quality=None,
            latency=None,
            scalar_reward=float(reward_spec.r_registered_failure),
            adjudication=adjudication,
        )
    if verdict is Adjudication.INFRASTRUCTURE_FAULT:
        return DecisionOutcomeV1(
            terminal_class=terminal,
            eligibility=LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
            costs=ConstraintCostsV1(
                c_deadline=1.0,
                c_latency_excess=None,
                c_authoritative_failure=None,
            ),
            reward_spec_sha256=spec_sha,
            adjudication=adjudication,
        )
    if verdict is Adjudication.FEEDBACK_ONLY_LOSS:
        # A proven control-message miss is not evidence of a bad action, so the
        # transition is censored from the perception/action reward rather than
        # penalized, and the control-plane miss is reported separately.
        return DecisionOutcomeV1(
            terminal_class=terminal,
            eligibility=LearningEligibility.CENSORED_FEEDBACK_ONLY_LOSS,
            costs=ConstraintCostsV1(
                c_deadline=1.0,
                c_latency_excess=None,
                c_authoritative_failure=0.0,
            ),
            reward_spec_sha256=spec_sha,
            adjudication=adjudication,
        )
    return DecisionOutcomeV1(
        terminal_class=terminal,
        eligibility=LearningEligibility.CENSORED_PENDING_ADJUDICATION,
        costs=ConstraintCostsV1(
            c_deadline=1.0,
            c_latency_excess=None,
            c_authoritative_failure=None,
        ),
        reward_spec_sha256=spec_sha,
        adjudication=adjudication,
    )


# --------------------------------------------------------------------------- #
# C. Replay transition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReplayTransitionV1:
    """One exactly identified SMDP transition, ready for a later replay layer.

    This is the *record*, not the storage: nothing here buffers, samples or
    persists.  ``hold_duration_tensors`` and ``discount_multiplier`` are derived
    properties over the frozen hold, never constructor inputs, so a caller
    cannot disagree with the controller about ``d`` or about ``gamma ** d``.
    """

    state: CausalStateV1
    executed_action: ExecutedActionIdentity
    completed_ticket: CompletedTicket
    outcome: DecisionOutcomeV1
    gamma_per_tensor: float
    reward_spec_sha256: str
    state_normalization_spec_sha256: str
    terminated: bool
    truncated: bool
    next_state: Optional[CausalStateV1] = None
    episode_end_reason: Optional[str] = None

    def __post_init__(self) -> None:
        E = TransitionIdentityError
        for name, expected in (
            ("state", CausalStateV1),
            ("executed_action", ExecutedActionIdentity),
            ("completed_ticket", CompletedTicket),
            ("outcome", DecisionOutcomeV1),
        ):
            if not isinstance(getattr(self, name), expected):
                raise E(
                    f"{name} must be a {expected.__name__}, got "
                    f"{type(getattr(self, name)).__name__}"
                )
        gamma = _finite_float(self.gamma_per_tensor, "gamma_per_tensor", E)
        if not 0.0 < gamma <= 1.0:
            raise E(f"gamma_per_tensor must lie in (0, 1], got {gamma}")
        _sha256_hex(self.reward_spec_sha256, "reward_spec_sha256", E)
        _sha256_hex(
            self.state_normalization_spec_sha256,
            "state_normalization_spec_sha256",
            E,
        )
        _exact_bool(self.terminated, "terminated", E)
        _exact_bool(self.truncated, "truncated", E)

        ticket = self.completed_ticket
        self.executed_action.require_reconciled()

        # -- one exact session and decision throughout --------------------- #
        if self.state.session_uuid != ticket.session_uuid:
            raise E(
                f"state session {self.state.session_uuid} does not match the "
                f"ticket's session {ticket.session_uuid}"
            )
        if self.outcome.terminal_class is not ticket.terminal_class:
            raise E(
                f"outcome terminal class {self.outcome.terminal_class.value} "
                f"disagrees with the ticket's {ticket.terminal_class.value}"
            )
        if self.outcome.reward_spec_sha256 != self.reward_spec_sha256:
            raise E(
                "the outcome was measured under a different reward spec than "
                "the transition records"
            )

        # -- the executed action equals every held tensor's action --------- #
        if self.executed_action != ticket.action:
            raise E(
                f"executed action {self.executed_action.canonical_mode} "
                f"q_e4={self.executed_action.q_e4} does not match the hold's "
                f"{ticket.action.canonical_mode} q_e4={ticket.action.q_e4}"
            )
        for member in ticket.hold.tensors:
            if member.action != self.executed_action:
                raise E(
                    f"tensor_seq {member.tensor_seq} carries a different "
                    f"executed action than the transition's"
                )

        # -- the state joins the reward-requested tensor and frame --------- #
        if self.state.tensor_seq != ticket.reward_tensor_seq:
            raise E(
                f"state tensor_seq {self.state.tensor_seq} is not the ticket's "
                f"reward-requested tensor {ticket.reward_tensor_seq}; the "
                f"decision state must join the reward tensor exactly"
            )
        if self.state.carla_frame_id != ticket.reward_carla_frame_id:
            raise E(
                f"state carla_frame_id {self.state.carla_frame_id} is not the "
                f"reward tensor's frame {ticket.reward_carla_frame_id}"
            )

        # -- the previous outcome cannot be this decision's own outcome ---- #
        previous = self.state.previous
        if previous is not None and previous.decision_seq >= ticket.decision_seq:
            raise E(
                f"the state's previous decision {previous.decision_seq} must "
                f"precede this decision {ticket.decision_seq}; a state may "
                f"never carry its own or a future outcome"
            )

        # -- terminal/truncation bookkeeping ------------------------------- #
        if self.terminated and self.truncated:
            raise E("a transition cannot be both terminated and truncated")
        if (self.terminated or self.truncated) and not self.episode_end_reason:
            raise E(
                "a terminated or truncated transition must carry an explicit "
                "episode_end_reason"
            )
        if not (self.terminated or self.truncated) and self.episode_end_reason:
            raise E(
                f"episode_end_reason {self.episode_end_reason!r} was given for "
                f"a transition that neither terminated nor truncated"
            )
        if self.episode_end_reason is not None:
            _non_empty_str(self.episode_end_reason, "episode_end_reason", E)

        # -- the next state is causally later and never crosses a session -- #
        if self.next_state is None:
            if not (self.terminated or self.truncated):
                raise E(
                    "a non-terminal transition requires a next state; only a "
                    "terminated or truncated transition may omit it"
                )
        else:
            if not isinstance(self.next_state, CausalStateV1):
                raise E(
                    f"next_state must be a CausalStateV1 or None, got "
                    f"{type(self.next_state).__name__}"
                )
            if self.next_state.session_uuid != self.state.session_uuid:
                raise E(
                    f"next state session {self.next_state.session_uuid} "
                    f"crosses out of session {self.state.session_uuid}; a "
                    f"transition never spans two sessions"
                )
            if self.next_state.observed_ns < self.state.observed_ns:
                raise E(
                    f"next state was observed at {self.next_state.observed_ns} "
                    f"ns, before this state's {self.state.observed_ns} ns"
                )
            last_governed = ticket.tensor_seqs[-1]
            if self.next_state.tensor_seq <= last_governed:
                raise E(
                    f"next state tensor_seq {self.next_state.tensor_seq} must "
                    f"follow every tensor governed by this decision (last is "
                    f"{last_governed}); tensor_seq is the frozen chronology"
                )

        # -- quality evidence is bound into the transition ----------------- #
        # Every transition that carries a quality must carry the hashes that
        # make it auditable: the raw ACK datagram and the edge-retained detailed
        # scientific row.  A censored or excluded transition has no quality and
        # therefore no evidence.
        quality = self.outcome.quality
        if quality is not None:
            evidence = quality.components.evidence
            if evidence.executed_action_sha256 != (
                self.executed_action.canonical_sha256()
            ):
                raise E(
                    f"the quality evidence is keyed on executed-action hash "
                    f"{evidence.executed_action_sha256}, but this transition "
                    f"executed {self.executed_action.canonical_sha256()}; "
                    f"evidence is never re-attributed between actions"
                )
            if evidence.ack_binding is not None:
                # An anchor action must agree with the anchor the ACK names.
                evidence.ack_binding.assert_matches_action(self.executed_action)
            elif self.executed_action.is_registered_anchor:
                raise E(
                    f"executed action {self.executed_action.profile_id!r} is a "
                    f"registered anchor, so its quality must be bound to a "
                    f"{QUALITY_ACK_SCHEMA} evidence packet; none was supplied"
                )

        # -- reward/eligibility agreement ---------------------------------- #
        if not self.outcome.learning_eligible and (
            self.outcome.scalar_reward is not None
        ):
            raise E(  # pragma: no cover - already guarded in DecisionOutcomeV1
                "a censored or excluded transition carries no scalar reward"
            )
        if ticket.terminal_class is TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
            if self.outcome.scalar_reward is not None:
                raise E(  # pragma: no cover - already guarded upstream
                    "an infrastructure fault must never carry a reward"
                )

    # -- derived identity -------------------------------------------------- #

    @property
    def session_uuid(self) -> str:
        return self.state.session_uuid

    @property
    def decision_seq(self) -> int:
        return self.completed_ticket.decision_seq

    @property
    def reward_tensor_seq(self) -> int:
        return self.completed_ticket.reward_tensor_seq

    @property
    def reward_carla_frame_id(self) -> int:
        return self.completed_ticket.reward_carla_frame_id

    @property
    def hold(self) -> ActionHoldManifest:
        """The frozen Phase-2 hold manifest this transition was measured over."""
        return self.completed_ticket.hold

    @property
    def hold_duration_tensors(self) -> int:
        """The realized ``d``, derived from the frozen hold."""
        return self.completed_ticket.hold_duration_tensors

    @property
    def discount_multiplier(self) -> float:
        """``gamma_per_tensor ** d``: the SMDP discount, always derived."""
        return float(self.gamma_per_tensor) ** self.hold_duration_tensors

    @property
    def scalar_reward(self) -> Optional[float]:
        return self.outcome.scalar_reward

    @property
    def costs(self) -> ConstraintCostsV1:
        return self.outcome.costs

    @property
    def quality(self) -> Optional[QualityEvaluationV1]:
        return self.outcome.quality

    @property
    def latency(self) -> Optional[LatencyMeasurementV1]:
        return self.outcome.latency

    @property
    def eligibility(self) -> LearningEligibility:
        return self.outcome.eligibility

    @property
    def terminal_class(self) -> TerminalClass:
        return self.completed_ticket.terminal_class

    @property
    def quality_evidence(self) -> Optional[QualityEvidenceV1]:
        """Provenance of this transition's quality, if it has one."""
        return None if self.quality is None else self.quality.components.evidence

    @property
    def raw_quality_ack_sha256(self) -> Optional[str]:
        """Hash of the raw ACK datagram; ``None`` off-anchor or without quality."""
        evidence = self.quality_evidence
        return None if evidence is None else evidence.raw_quality_ack_sha256

    @property
    def detailed_evidence_sha256(self) -> Optional[str]:
        """Hash of the edge-retained detailed scientific evidence row."""
        evidence = self.quality_evidence
        return None if evidence is None else evidence.detailed_evidence_sha256

    @property
    def executed_action_sha256(self) -> str:
        """Canonical hash of the full executed-action identity.

        This is the core reward-evidence key: it is well defined whether or not
        the action is one of the 72 anchors, which is exactly what a protocol-v2
        ACK would need to carry.
        """
        return self.executed_action.canonical_sha256()

    @property
    def learning_eligible(self) -> bool:
        return self.outcome.learning_eligible

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic canonical mapping for this whole transition."""
        return {
            "completed_ticket": self.completed_ticket.to_canonical_dict(),
            "decision_seq": self.decision_seq,
            "discount_multiplier": self.discount_multiplier,
            "episode_end_reason": self.episode_end_reason,
            "executed_action": self.executed_action.to_canonical_dict(),
            "gamma_per_tensor": float(self.gamma_per_tensor),
            "hold_duration_tensors": self.hold_duration_tensors,
            "next_state": (
                None
                if self.next_state is None
                else self.next_state.to_canonical_dict()
            ),
            "outcome": self.outcome.to_canonical_dict(),
            "record": "replay_transition_v1",
            "detailed_evidence_sha256": self.detailed_evidence_sha256,
            "executed_action_sha256": self.executed_action_sha256,
            "quality_evidence": (
                None
                if self.quality_evidence is None
                else self.quality_evidence.to_canonical_dict()
            ),
            "raw_quality_ack_sha256": self.raw_quality_ack_sha256,
            "reward_carla_frame_id": self.reward_carla_frame_id,
            "reward_latency_clock_domain": REWARD_LATENCY_CLOCK_DOMAIN,
            "reward_spec_sha256": self.reward_spec_sha256,
            "reward_tensor_seq": self.reward_tensor_seq,
            "schema_id": SCHEMA_ID,
            "schema_sha256": SCHEMA_SHA256,
            "schema_version": SCHEMA_VERSION,
            "session_uuid": self.session_uuid,
            "state": self.state.to_canonical_dict(),
            "state_normalization_spec_sha256": (
                self.state_normalization_spec_sha256
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


def build_replay_transition(
    *,
    state: CausalStateV1,
    next_state: Optional[CausalStateV1],
    completed_ticket: CompletedTicket,
    outcome: DecisionOutcomeV1,
    reward_spec: RewardSpecV1,
    normalization: StateNormalizationSpecV1,
    terminated: bool = False,
    truncated: bool = False,
    episode_end_reason: Optional[str] = None,
) -> ReplayTransitionV1:
    """Assemble a transition, deriving every derivable field from the frozen inputs.

    The executed action, the reward-spec hash, the normalization-spec hash and
    ``gamma_per_tensor`` are all taken from the supplied frozen records rather
    than from loose caller arguments, so the assembled transition cannot
    disagree with the controller or with either specification.
    """
    if not isinstance(reward_spec, RewardSpecV1):
        raise RewardSpecError(
            f"reward_spec must be a RewardSpecV1, got {type(reward_spec).__name__}"
        )
    if not isinstance(normalization, StateNormalizationSpecV1):
        raise NormalizationSpecError(
            f"normalization must be a StateNormalizationSpecV1, got "
            f"{type(normalization).__name__}"
        )
    if not isinstance(completed_ticket, CompletedTicket):
        raise TransitionIdentityError(
            f"completed_ticket must be a controller CompletedTicket, got "
            f"{type(completed_ticket).__name__}"
        )
    return ReplayTransitionV1(
        state=state,
        executed_action=completed_ticket.action,
        completed_ticket=completed_ticket,
        outcome=outcome,
        gamma_per_tensor=float(reward_spec.gamma_per_tensor),
        reward_spec_sha256=reward_spec.canonical_sha256(),
        state_normalization_spec_sha256=normalization.canonical_sha256(),
        terminated=terminated,
        truncated=truncated,
        next_state=next_state,
        episode_end_reason=episode_end_reason,
    )
