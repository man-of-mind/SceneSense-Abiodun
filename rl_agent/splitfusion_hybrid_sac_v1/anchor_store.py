"""Measured-anchor evidence store for the SplitFusion Hybrid-SAC (DESIGN Step 4).

This module implements *only* the read-only evidence store described in
``DESIGN.md`` section 10 and implementation step 4: it binds the two frozen
288-cell measurement artifacts, reconciles their action identity against the
frozen 72-action catalog, and exposes the result as exact-anchor records.

What this module deliberately does **not** do
---------------------------------------------

It contains no reward weights, no ``Q^perc`` computation, no interpolation or
extrapolation over ``q``, no actor/critic, no training loop, no replay storage
and no CARLA/OAI execution.  Those are later, separately gated steps.

The single scientific claim it enforces
---------------------------------------

Every record it returns is labelled :data:`EVIDENCE_CLASS`
(``MEASURED_ANCHOR_AGGREGATE``).  Such a record is an *aggregate over a whole
campaign cell* -- thousands of frames sharing one action and one authored
network profile.  It is **not** a per-frame causal transition: it has no
``session_uuid``/``decision_seq``/``carla_frame_id``, no per-decision
SNR/MCS/BSR state, no hold duration and no terminal classification for an
individual ticket.  It therefore may never be inserted into
``ReplayTransitionV1``; :meth:`MeasuredAnchorRecord.as_replay_transition`
exists solely to refuse that, loudly.

Two levels of evidence, kept apart
----------------------------------

``DESIGN.md`` section 10 warns that the campaign mixes two different things.
This store keeps them structurally separate:

* :class:`ActionQualityAnchor` -- **action-level** perception quality and
  payload identity.  These are properties of the offline-validated action
  alone; they were measured once per action and are byte-identical across all
  four network profiles (the store verifies this rather than assuming it).
* :class:`NetworkProfileOutcome` -- **cell-level** transport, admission,
  delivery and latency outcomes under one authored network profile.  These do
  vary by profile and are the only profile-conditioned evidence here.

Exact anchors only
------------------

Lookup is exact.  ``q`` must equal one of the six registered anchors under
exact decimal semantics; ``0.30001`` and ``0.1 + 0.2`` are *not* ``0.30`` and
raise :class:`UnsupportedCounterfactualError` (code
``UNSUPPORTED_COUNTERFACTUAL``).  Nothing is snapped to a nearest anchor and
no quality, payload or latency value is ever interpolated between anchors.
The interior of the continuous ``q`` surface is unmeasured, and this store
says so instead of inventing it.

Fail-closed evidence binding
----------------------------

Both source files are pinned by SHA-256.  Any hash drift, missing registered
column, malformed numeric field, duplicate/missing/foreign action or cell key,
inventory miscount or cross-source disagreement raises rather than degrading.
Missing measurements stay missing: an unobserved latency percentile is
``None``, never ``0.0``.
"""

from __future__ import annotations

import csv
import hashlib
import numbers
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .action_contract import (
    EXPECTED_MODE_COUNT,
    EXPECTED_PROFILE_COUNT,
    EXPECTED_Q_ANCHOR_COUNT,
    Q_E4_SCALE,
    SPATIAL_CELLS,
    AnchorAction,
    SplitActionContract,
    UnknownJointModeError,
    load_contract,
)
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "ACTION_SUMMARY_RELATIVE_PATH",
    "ACTION_SUMMARY_SHA256",
    "ActionQualityAnchor",
    "AnchorEvidenceStore",
    "AnchorStoreError",
    "EVIDENCE_CLASS",
    "EVIDENCE_USE_RESTRICTION",
    "EXPECTED_CELL_COUNT",
    "EXPECTED_NETWORK_PROFILE_COUNT",
    "EvidenceIntegrityError",
    "EvidenceInventoryError",
    "FORBIDDEN_POLICY_OBSERVATION_FIELDS",
    "LatencyStat",
    "MeasuredAnchorRecord",
    "NETWORK_PROFILE_ORDER",
    "NetworkProfileOutcome",
    "PROFILE_LATENCY_RELATIVE_PATH",
    "PROFILE_LATENCY_SHA256",
    "PolicyObservationLeakError",
    "REGISTERED_Q_ANCHORS_E4",
    "REPLAY_INSERTION_RESTRICTION",
    "ReplayInsertionForbiddenError",
    "STORE_SCHEMA_ID",
    "UNSUPPORTED_COUNTERFACTUAL",
    "UnsupportedCounterfactualError",
    "assert_no_forbidden_policy_observation",
    "default_anchor_store",
    "exact_q_e4",
    "load_anchor_store",
]


# --------------------------------------------------------------------------- #
# Frozen registered constants
# --------------------------------------------------------------------------- #

#: Store schema identifier, carried by every canonical serialization.
STORE_SCHEMA_ID = "splitfusion_measured_anchor_evidence_store_v1"

#: The evidence class label stamped on every returned record.
EVIDENCE_CLASS = "MEASURED_ANCHOR_AGGREGATE"

#: Verbatim restriction text carried alongside :data:`EVIDENCE_CLASS`.
EVIDENCE_USE_RESTRICTION = (
    "Campaign cell aggregate over many frames sharing one action and one "
    "authored network profile. NOT a per-frame causal transition: it carries "
    "no session_uuid, decision_seq, carla_frame_id, per-decision SNR/MCS/BSR "
    "state, hold duration or per-ticket terminal classification. Admissible "
    "for unit checks, outcome-model initialization and conservative critic "
    "warm-start only."
)

#: Why a record may never reach the replay buffer.
REPLAY_INSERTION_RESTRICTION = (
    "MEASURED_ANCHOR_AGGREGATE records must not be inserted into "
    "ReplayTransitionV1. ReplayTransitionV1 requires exact per-frame identity "
    "and a single documented terminal classification, neither of which a "
    "campaign-cell aggregate possesses."
)

#: Machine-readable failure code for any non-anchor ``q`` request.
UNSUPPORTED_COUNTERFACTUAL = "UNSUPPORTED_COUNTERFACTUAL"

#: Source A: action-level offline consolidation (72 rows), project-root relative.
ACTION_SUMMARY_RELATIVE_PATH = (
    "experiments/splitfusion_288_offline_rl_dataset_v1/"
    "20260909_offline_consolidation_v1/action_72_summary.csv"
)

#: Required exact SHA-256 of the source A bytes.
ACTION_SUMMARY_SHA256 = (
    "250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88"
)

#: Source B: per-cell quality/latency analysis v3 (288 rows), project-root relative.
PROFILE_LATENCY_RELATIVE_PATH = (
    "experiments/splitfusion_supervisor_analysis_v1/"
    "20260915_tail_completion_feedback_policy_analysis_v3/"
    "action_profile_quality_latency.csv"
)

#: Required exact SHA-256 of the source B bytes.
PROFILE_LATENCY_SHA256 = (
    "2c0e1e974d612567b74f797b8d6b47733cdb7f138f5061657fe74de93dd423a7"
)

#: Short source tags used in record provenance.
SOURCE_ACTION_SUMMARY = "action_72_summary_v1"
SOURCE_PROFILE_LATENCY = "action_profile_quality_latency_v3"

#: The four authored network profiles, in registered order.
NETWORK_PROFILE_ORDER: Tuple[str, ...] = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)

EXPECTED_NETWORK_PROFILE_COUNT = 4
EXPECTED_CELL_COUNT = EXPECTED_PROFILE_COUNT * EXPECTED_NETWORK_PROFILE_COUNT

#: The six registered ``q`` anchors, on the ``q_e4`` wire scale.
REGISTERED_Q_ANCHORS_E4: Tuple[int, ...] = (0, 3000, 5000, 7000, 9000, 9800)

#: Fields that must never reach a policy observation (DESIGN.md section 4:
#: "Forbidden v1 inputs include the authored network-profile name").
#: ``cell_id`` is included because it embeds the profile name verbatim.
FORBIDDEN_POLICY_OBSERVATION_FIELDS = frozenset({"network_profile", "cell_id"})

#: Action-identity columns of source A, cross-checked against the catalog.
_A_IDENTITY_FIELDS: Tuple[str, ...] = (
    "action_id",
    "profile_id",
    "family",
    "quantizer",
    "q",
    "q_e4",
    "keep_count",
    "drop_count",
)

#: Payload-identity columns of source A (properties of the action's wire form).
_A_PAYLOAD_IDENTITY_FIELDS: Tuple[str, ...] = (
    "keep_count",
    "drop_count",
    "bit_width",
    "latent_width",
    "wire_layout",
    "zstd_level",
    "routing_tag",
)

#: Per-profile metric bases of source A; each appears as ``<base>__<PROFILE>``.
_A_PROFILE_COUNT_FIELDS: Tuple[str, ...] = (
    "frames_sent",
    "edge_complete_reassemblies",
    "maps_installed",
    "installed_within_100ms_service_reference",
    "ack_within_500ms_timeout",
    "terminal_TIMEOUT_NO_ACK",
)

_A_PROFILE_RATE_FIELDS: Tuple[str, ...] = (
    "rate_reassembled_per_sent",
    "rate_installed_per_sent",
    "rate_on_time_100ms_per_sent",
    "rate_ack_500ms_per_sent",
    "rate_datagrams_received_per_transmitted",
)

#: Per-profile measured scalars that may legitimately be absent.
_A_PROFILE_OPTIONAL_FLOAT_FIELDS: Tuple[str, ...] = (
    "live_scientific_inner_bytes_median",
    "live_estimated_wire_bytes_median",
    "live_datagrams_per_message_median",
    "sensor_preparation_coverage",
    "radio_achieved_snr_db_median",
)

#: Cell-level count columns of source B, with ``frames_sent`` as denominator.
_B_COUNT_FIELDS: Tuple[str, ...] = (
    "frames_sent",
    "measured_complete_reassemblies",
    "measured_edge_admissions",
    "scheduler_arrivals_observed",
    "scheduler_arrivals_imputed",
    "simulated_compute_completions",
    "model_ready_timing_samples",
    "simulated_map_installs",
    "simulated_useful_newer_map_installs",
)

_B_RATE_FIELDS: Tuple[str, ...] = (
    "rate_reassembled_per_sent",
    "rate_admitted_per_sent",
    "rate_installed_per_sent",
    "rate_useful_installations_per_sent",
)

#: Cell-level map-service scalars of source B (optional; absent when no install).
_B_MAP_SERVICE_FIELDS: Tuple[str, ...] = (
    "map_observation_duration_s",
    "map_available_fraction",
    "time_weighted_map_aoi_ms_when_available",
    "fresh_map_time_ms_le_150_fraction",
    "fresh_map_time_ms_le_200_fraction",
    "fresh_map_time_ms_le_250_fraction",
    "fresh_map_time_ms_le_300_fraction",
)

#: The twelve mutually exclusive per-cell terminal tallies of source B.
_B_TERMINAL_FIELDS: Tuple[str, ...] = (
    "terminal_measured_pre_queue_rejection",
    "terminal_predicted_map_install_horizon_exceeded",
    "terminal_processing_horizon_expired_after_compute",
    "terminal_processing_horizon_expired_after_publication",
    "terminal_processing_horizon_expired_at_arrival",
    "terminal_processing_horizon_expired_before_compute",
    "terminal_processing_horizon_expired_before_publication",
    "terminal_queue_wait_budget_exceeded",
    "terminal_result_published",
    "terminal_superseded_pending",
    "terminal_superseded_publication_pending",
    "terminal_transport_incomplete",
)

#: Conditional latency stages of source B.  Each has ``<stage>_count`` support
#: and ``<stage>_{p50,p95,p99}_ms`` percentiles.
_B_LATENCY_STAGES: Tuple[str, ...] = (
    "ue_action",
    "sensor_compute",
    "sensor_compute_including_concat",
    "seven_channel_concat",
    "pure_front",
    "network",
    "edge_queue",
    "model_tail",
    "tail_support",
    "map_install",
    "edge_model_ready",
    "sensor_model_ready",
    "edge_map",
    "total",
    "capture_total",
)

_B_LATENCY_PERCENTILES: Tuple[str, ...] = ("p50", "p95", "p99")

#: The one conditional latency stage carried by source A: map install AoI.
#: It reports median and p95 only -- there is no p99 column.
_A_LATENCY_STAGE = "install_aoi_ms"
_A_LATENCY_PERCENTILES: Tuple[str, ...] = ("p50", "p95")

#: Source A column names for the install-AoI stage, by percentile key.
_A_LATENCY_COLUMNS: Mapping[str, str] = MappingProxyType(
    {"p50": "install_aoi_ms_median", "p95": "install_aoi_ms_p95"}
)


# --------------------------------------------------------------------------- #
# Exceptions: fail closed, never normalize
# --------------------------------------------------------------------------- #


class AnchorStoreError(Exception):
    """Base class for every measured-anchor evidence failure."""


class EvidenceIntegrityError(AnchorStoreError):
    """A source file is unreadable, hash-drifted, malformed or incomplete."""


class EvidenceInventoryError(AnchorStoreError):
    """The evidence inventory does not reconcile exactly as registered."""


class UnsupportedCounterfactualError(AnchorStoreError):
    """An off-anchor ``q`` was requested.

    The measured campaign contains six ``q`` anchors per joint mode and nothing
    between them.  Answering an interior ``q`` would require interpolation,
    which would silently manufacture a measurement.  This store refuses.
    """

    #: Stable machine-readable code for callers that branch on the failure.
    code = UNSUPPORTED_COUNTERFACTUAL

    def __init__(self, message: str) -> None:
        super().__init__(f"{UNSUPPORTED_COUNTERFACTUAL}: {message}")


class ReplayInsertionForbiddenError(AnchorStoreError):
    """A measured aggregate was offered to the causal replay buffer."""


class PolicyObservationLeakError(AnchorStoreError):
    """An authored network-profile label reached a policy observation."""


# --------------------------------------------------------------------------- #
# Strict field parsing: missing stays missing
# --------------------------------------------------------------------------- #


def _require(condition: bool, message: str) -> None:
    """Raise :class:`EvidenceIntegrityError` unless ``condition`` holds."""
    if not condition:
        raise EvidenceIntegrityError(message)


def _cell(row: Mapping[str, str], column: str, where: str) -> str:
    """Return the verbatim text of ``column``, or fail if it is not declared."""
    if column not in row:
        raise EvidenceIntegrityError(
            f"{where}: registered column {column!r} is absent from the source"
        )
    value = row[column]
    if value is None:
        raise EvidenceIntegrityError(f"{where}: column {column!r} has no value")
    return value


def _parse_int(row: Mapping[str, str], column: str, where: str) -> int:
    """Parse a required integer field."""
    text = _cell(row, column, where).strip()
    _require(text != "", f"{where}: required integer {column!r} is empty")
    try:
        return int(text)
    except ValueError as exc:
        raise EvidenceIntegrityError(
            f"{where}: {column!r} is not an integer: {text!r}"
        ) from exc


def _parse_optional_int(
    row: Mapping[str, str], column: str, where: str
) -> Optional[int]:
    """Parse an integer field that may legitimately be absent."""
    text = _cell(row, column, where).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise EvidenceIntegrityError(
            f"{where}: {column!r} is not an integer: {text!r}"
        ) from exc


def _finite(value: float, column: str, where: str) -> float:
    """Reject NaN and infinity rather than letting them into a record."""
    if value != value or value in (float("inf"), float("-inf")):
        raise EvidenceIntegrityError(
            f"{where}: {column!r} is not finite: {value!r}"
        )
    return value


def _parse_float(row: Mapping[str, str], column: str, where: str) -> float:
    """Parse a required finite float field."""
    text = _cell(row, column, where).strip()
    _require(text != "", f"{where}: required float {column!r} is empty")
    try:
        return _finite(float(text), column, where)
    except ValueError as exc:
        raise EvidenceIntegrityError(
            f"{where}: {column!r} is not a float: {text!r}"
        ) from exc


def _parse_optional_float(
    row: Mapping[str, str], column: str, where: str
) -> Optional[float]:
    """Parse a float that may legitimately be absent.

    An absent measurement returns ``None``.  It is never coerced to ``0.0``:
    "no sample was observed" and "the observed value was zero" are different
    measured facts and the store keeps them different.
    """
    text = _cell(row, column, where).strip()
    if text == "":
        return None
    try:
        return _finite(float(text), column, where)
    except ValueError as exc:
        raise EvidenceIntegrityError(
            f"{where}: {column!r} is not a float: {text!r}"
        ) from exc


def _parse_bool(row: Mapping[str, str], column: str, where: str) -> bool:
    """Parse a strict ``True``/``False`` field."""
    text = _cell(row, column, where).strip()
    if text == "True":
        return True
    if text == "False":
        return False
    raise EvidenceIntegrityError(
        f"{where}: {column!r} is not a strict boolean: {text!r}"
    )


def _freeze_str_map(pairs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable view of a mapping."""
    return MappingProxyType(dict(pairs))


def exact_q_e4(q: Any) -> Optional[int]:
    """Convert ``q`` to its exact ``q_e4`` integer, or ``None`` if inexact.

    This is deliberately *not* :meth:`SplitActionContract.quality_for`, which
    rounds half-up to the nearest wire integer.  Rounding is correct when
    choosing what to transmit; it is wrong when deciding whether a measurement
    exists.  Here ``0.30001`` must not become the measured ``0.30`` anchor, so
    only an exact decimal multiple of ``1e-4`` yields an integer.

    Returns:
        The exact ``q_e4``, or ``None`` when ``q`` is not an exact multiple of
        ``1e-4`` (and therefore cannot name any registered anchor).
    """
    if isinstance(q, bool) or not isinstance(q, (numbers.Integral, float, Decimal, str)):
        raise UnsupportedCounterfactualError(
            f"q must be an exact numeric value, got {type(q).__name__}: {q!r}"
        )
    try:
        if isinstance(q, Decimal):
            decimal_q = q
        elif isinstance(q, numbers.Integral):
            decimal_q = Decimal(int(q))
        else:
            # ``str(float)`` is the shortest round-tripping decimal literal, so
            # 0.3 gives Decimal('0.3') while 0.1 + 0.2 keeps its real value
            # 0.30000000000000004 and is correctly judged off-anchor.
            decimal_q = Decimal(str(q))
    except (InvalidOperation, ValueError) as exc:
        raise UnsupportedCounterfactualError(
            f"q is not a decodable decimal value: {q!r}"
        ) from exc
    if not decimal_q.is_finite():
        raise UnsupportedCounterfactualError(f"q is not finite: {q!r}")
    scaled = decimal_q * Q_E4_SCALE
    if scaled != scaled.to_integral_value():
        return None
    return int(scaled)


def assert_no_forbidden_policy_observation(mapping: Mapping[str, Any]) -> None:
    """Reject any mapping bound for a policy observation that names a profile.

    ``DESIGN.md`` section 4 forbids the authored network-profile name as a v1
    policy input: it is testbed metadata the deployed UE cannot observe, and a
    policy that reads it would be scored on privileged information.  This store
    exposes no feature-vector builder at all, so this guard is the boundary
    check a later feature builder must call.

    Raises:
        PolicyObservationLeakError: if any forbidden field is present.
    """
    leaked = sorted(FORBIDDEN_POLICY_OBSERVATION_FIELDS.intersection(mapping))
    if leaked:
        raise PolicyObservationLeakError(
            f"authored network-profile identity may not be a policy "
            f"observation; remove {leaked}"
        )


# --------------------------------------------------------------------------- #
# Immutable value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LatencyStat:
    """One conditional latency distribution and the support it is conditional on.

    ``support`` is the number of contributing samples.  When it is zero the
    stage was never observed in this cell and every percentile is ``None`` --
    an unobserved latency is missing, not fast.
    """

    stage: str
    source: str
    support: int
    percentiles_ms: Mapping[str, Optional[float]]

    def __post_init__(self) -> None:
        _require(
            self.support >= 0,
            f"latency stage {self.stage!r}: negative support {self.support}",
        )
        observed = self.support > 0
        for key, value in self.percentiles_ms.items():
            if observed:
                _require(
                    value is not None,
                    f"latency stage {self.stage!r}: support={self.support} but "
                    f"{key} is absent",
                )
            else:
                _require(
                    value is None,
                    f"latency stage {self.stage!r}: zero support but {key} "
                    f"carries the value {value!r}",
                )

    @property
    def observed(self) -> bool:
        """True when at least one sample contributed to this stage."""
        return self.support > 0

    @property
    def p50_ms(self) -> Optional[float]:
        """Median latency in milliseconds, or ``None`` when unobserved."""
        return self.percentiles_ms.get("p50")

    @property
    def p95_ms(self) -> Optional[float]:
        """95th-percentile latency in milliseconds, or ``None``."""
        return self.percentiles_ms.get("p95")

    @property
    def p99_ms(self) -> Optional[float]:
        """99th-percentile latency in milliseconds, or ``None``."""
        return self.percentiles_ms.get("p99")

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic serializable form."""
        return {
            "stage": self.stage,
            "source": self.source,
            "support": self.support,
            "observed": self.observed,
            "percentiles_ms": {
                key: self.percentiles_ms[key] for key in sorted(self.percentiles_ms)
            },
        }


@dataclass(frozen=True, slots=True)
class ActionQualityAnchor:
    """Action-level perception quality and payload identity for one anchor.

    Everything here is a property of the action itself, measured offline
    against CARLA ground truth.  None of it depends on the radio: the store
    verifies that all four network-profile cells report byte-identical quality
    before constructing this record.

    ``raw_quality`` holds every registered ``val_*`` field of the source row
    **verbatim as text**, so the exact recorded evidence -- including gate
    strings such as ``'12/12'``, checkpoint digests and deliberately absent
    fields -- survives without a lossy numeric reinterpretation.
    ``quality_metrics`` is the typed numeric view of the same fields, with
    ``None`` for the ones the source leaves empty.
    """

    action_id: int
    profile_id: str
    mode_id: int
    family: str
    quantizer: str
    q: float
    q_e4: int
    payload_identity: Mapping[str, Any]
    raw_quality: Mapping[str, str]
    quality_metrics: Mapping[str, Optional[float]]
    derived_presentation_quality: Mapping[str, Optional[float]]
    source: str
    source_sha256: str
    source_row_sha256: str

    @property
    def key(self) -> Tuple[str, str, int]:
        """The ``(family, quantizer, q_e4)`` exact-lookup tuple."""
        return self.family, self.quantizer, self.q_e4

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic serializable form."""
        return {
            "action_id": self.action_id,
            "profile_id": self.profile_id,
            "mode_id": self.mode_id,
            "family": self.family,
            "quantizer": self.quantizer,
            "q": self.q,
            "q_e4": self.q_e4,
            "payload_identity": dict(self.payload_identity),
            "raw_quality": dict(self.raw_quality),
            "quality_metrics": dict(self.quality_metrics),
            "derived_presentation_quality": dict(self.derived_presentation_quality),
            "source": self.source,
            "source_sha256": self.source_sha256,
            "source_row_sha256": self.source_row_sha256,
        }


@dataclass(frozen=True, slots=True)
class NetworkProfileOutcome:
    """Transport, admission, delivery and latency outcome for one campaign cell.

    This is the only profile-conditioned evidence in the store.  The authored
    ``network_profile`` label is retained here for analysis and provenance, and
    is simultaneously registered in
    :data:`FORBIDDEN_POLICY_OBSERVATION_FIELDS`: it may describe a measurement
    but may never become a policy feature.  Use :meth:`policy_safe_dict` at any
    boundary that feeds a model.

    Two independent zero-delivery facts are preserved, because the two sources
    measure different things and disagree in count (66 cells versus 46):

    * ``zero_delivery_live_campaign`` -- the live 288-cell campaign installed no
      map at all in this cell, so no end-to-end network latency sample exists.
    * ``zero_admission_replay_v3`` -- the v3 tail-completion replay admitted no
      feature at the edge, so it simulated no map install.

    Both are explicit measured outcomes, not missing data.
    """

    action_id: int
    network_profile: str
    cell_id: str
    counts: Mapping[str, int]
    denominators: Mapping[str, int]
    rates: Mapping[str, Optional[float]]
    measured_payload_bytes: Mapping[str, Optional[float]]
    map_service: Mapping[str, Optional[float]]
    terminal_counts: Mapping[str, int]
    latency: Mapping[str, LatencyStat]
    zero_delivery_live_campaign: bool
    zero_admission_replay_v3: bool
    network_latency_observed_only: bool
    execution_provenance: str
    route_summary_available: bool
    source_sha256: Mapping[str, str]
    source_row_sha256: Mapping[str, str]

    @property
    def frames_sent(self) -> int:
        """The cell denominator: frames the UE actually transmitted."""
        return self.counts["frames_sent"]

    def latency_stat(self, stage: str) -> LatencyStat:
        """Return one conditional latency stage.

        Raises:
            KeyError: if ``stage`` is not a registered stage.
        """
        return self.latency[stage]

    def policy_safe_dict(self) -> Dict[str, Any]:
        """This outcome with every forbidden policy-observation field removed.

        The authored ``network_profile`` and the ``cell_id`` that embeds it are
        stripped.  The result still describes a measured aggregate and remains
        inadmissible as a causal transition.
        """
        payload = self.to_canonical_dict()
        for forbidden in FORBIDDEN_POLICY_OBSERVATION_FIELDS:
            payload.pop(forbidden, None)
        assert_no_forbidden_policy_observation(payload)
        return payload

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic serializable form."""
        return {
            "action_id": self.action_id,
            "network_profile": self.network_profile,
            "cell_id": self.cell_id,
            "counts": dict(self.counts),
            "denominators": dict(self.denominators),
            "rates": dict(self.rates),
            "measured_payload_bytes": dict(self.measured_payload_bytes),
            "map_service": dict(self.map_service),
            "terminal_counts": dict(self.terminal_counts),
            "latency": {
                stage: self.latency[stage].to_canonical_dict()
                for stage in sorted(self.latency)
            },
            "zero_delivery_live_campaign": self.zero_delivery_live_campaign,
            "zero_admission_replay_v3": self.zero_admission_replay_v3,
            "network_latency_observed_only": self.network_latency_observed_only,
            "execution_provenance": self.execution_provenance,
            "route_summary_available": self.route_summary_available,
            "source_sha256": dict(self.source_sha256),
            "source_row_sha256": dict(self.source_row_sha256),
        }


@dataclass(frozen=True, slots=True)
class MeasuredAnchorRecord:
    """One registered anchor: action-level quality plus four cell outcomes.

    Always labelled :data:`EVIDENCE_CLASS` and always inadmissible to
    ``ReplayTransitionV1``.
    """

    quality: ActionQualityAnchor
    profiles: Mapping[str, NetworkProfileOutcome]
    evidence_class: str = EVIDENCE_CLASS
    evidence_use_restriction: str = EVIDENCE_USE_RESTRICTION
    replay_admissible: bool = False
    replay_restriction: str = REPLAY_INSERTION_RESTRICTION

    @property
    def action_id(self) -> int:
        """Catalog action identifier."""
        return self.quality.action_id

    @property
    def key(self) -> Tuple[str, str, int]:
        """The ``(family, quantizer, q_e4)`` exact-lookup tuple."""
        return self.quality.key

    def outcome(self, network_profile: str) -> NetworkProfileOutcome:
        """Return one network profile's measured outcome.

        Raises:
            KeyError: if ``network_profile`` is not one of the four authored
                profiles.
        """
        return self.profiles[network_profile]

    def as_replay_transition(self) -> None:
        """Always refuse: an aggregate is not a causal transition.

        Raises:
            ReplayInsertionForbiddenError: unconditionally.
        """
        raise ReplayInsertionForbiddenError(
            f"{REPLAY_INSERTION_RESTRICTION} Refused for action_id="
            f"{self.action_id} ({self.quality.profile_id})."
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic serializable form."""
        return {
            "schema": STORE_SCHEMA_ID,
            "evidence_class": self.evidence_class,
            "evidence_use_restriction": self.evidence_use_restriction,
            "replay_admissible": self.replay_admissible,
            "replay_restriction": self.replay_restriction,
            "quality": self.quality.to_canonical_dict(),
            "profiles": {
                profile: self.profiles[profile].to_canonical_dict()
                for profile in sorted(self.profiles)
            },
        }

    def canonical_bytes(self) -> bytes:
        """Canonical JSON bytes of :meth:`to_canonical_dict`."""
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """SHA-256 of :meth:`canonical_bytes`."""
        return canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Source binding helpers
# --------------------------------------------------------------------------- #


def default_project_root() -> Path:
    """Resolve the ``abiodun/`` project root relative to this module."""
    return Path(__file__).resolve().parents[2]


def _read_pinned_csv(
    path: Path, expected_sha256: str, label: str
) -> Tuple[str, Tuple[str, ...], Tuple[Mapping[str, str], ...]]:
    """Read a SHA-pinned CSV, returning its digest, header and rows.

    Raises:
        EvidenceIntegrityError: on an unreadable file, a hash drift, a
            duplicated header column or a ragged row.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceIntegrityError(
            f"cannot read {label} evidence at {path}: {exc}"
        ) from exc

    digest = hashlib.sha256(raw).hexdigest()
    _require(
        digest == expected_sha256,
        f"{label} SHA-256 mismatch at {path}: expected {expected_sha256}, "
        f"got {digest}",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceIntegrityError(
            f"{label} at {path} is not valid UTF-8: {exc}"
        ) from exc

    reader = csv.reader(text.splitlines())
    try:
        header = tuple(next(reader))
    except StopIteration as exc:
        raise EvidenceIntegrityError(f"{label} at {path} is empty") from exc
    _require(
        len(set(header)) == len(header),
        f"{label} at {path} declares duplicate columns",
    )

    rows = []
    for index, values in enumerate(reader, start=2):
        _require(
            len(values) == len(header),
            f"{label} at {path} line {index}: expected {len(header)} fields, "
            f"got {len(values)}",
        )
        rows.append(_freeze_str_map(dict(zip(header, values))))
    return digest, header, tuple(rows)


def _row_sha256(source: str, header_sha: str, row: Mapping[str, str]) -> str:
    """Deterministic SHA-256 of one verbatim source row.

    The digest covers the source tag, the source's header digest and the row's
    verbatim ``column -> text`` mapping, canonically serialized.  It therefore
    changes if any recorded character changes, and is independent of column
    order and of the file's line endings.
    """
    return canonical_sha256(
        {"source": source, "header_sha256": header_sha, "row": dict(row)}
    )


def _require_columns(
    header: Sequence[str], required: Sequence[str], label: str
) -> None:
    """Fail closed when a registered column is absent from a source."""
    missing = [name for name in required if name not in header]
    _require(
        not missing,
        f"{label}: registered columns absent from the source: {missing}",
    )


# --------------------------------------------------------------------------- #
# The evidence store
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AnchorEvidenceStore:
    """Strict, read-only view of the measured 72-anchor / 288-cell evidence.

    Construct with :meth:`from_paths` or :func:`load_anchor_store`; the
    constructor is not intended to be called directly.
    """

    action_summary_path: Path
    action_summary_sha256: str
    profile_latency_path: Path
    profile_latency_sha256: str
    contract: SplitActionContract
    records: Tuple[MeasuredAnchorRecord, ...]
    _by_action_id: Mapping[int, MeasuredAnchorRecord]
    _by_key: Mapping[Tuple[str, str, int], MeasuredAnchorRecord]
    _by_mode: Mapping[int, Tuple[MeasuredAnchorRecord, ...]]

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_paths(
        cls,
        action_summary_path: Optional[Path] = None,
        profile_latency_path: Optional[Path] = None,
        contract: Optional[SplitActionContract] = None,
    ) -> "AnchorEvidenceStore":
        """Read, verify and bind both pinned evidence files.

        Nothing is written and no source file is modified.
        """
        root = default_project_root()
        summary_path = (
            Path(action_summary_path)
            if action_summary_path is not None
            else root / ACTION_SUMMARY_RELATIVE_PATH
        )
        latency_path = (
            Path(profile_latency_path)
            if profile_latency_path is not None
            else root / PROFILE_LATENCY_RELATIVE_PATH
        )
        bound_contract = contract if contract is not None else load_contract()

        summary_sha, summary_header, summary_rows = _read_pinned_csv(
            summary_path, ACTION_SUMMARY_SHA256, SOURCE_ACTION_SUMMARY
        )
        latency_sha, latency_header, latency_rows = _read_pinned_csv(
            latency_path, PROFILE_LATENCY_SHA256, SOURCE_PROFILE_LATENCY
        )
        return cls._bind(
            summary_path,
            summary_sha,
            summary_header,
            summary_rows,
            latency_path,
            latency_sha,
            latency_header,
            latency_rows,
            bound_contract,
        )

    @classmethod
    def _bind(
        cls,
        summary_path: Path,
        summary_sha: str,
        summary_header: Sequence[str],
        summary_rows: Sequence[Mapping[str, str]],
        latency_path: Path,
        latency_sha: str,
        latency_header: Sequence[str],
        latency_rows: Sequence[Mapping[str, str]],
        contract: SplitActionContract,
    ) -> "AnchorEvidenceStore":
        """Verify inventory and identity, then build the immutable records."""
        cls._verify_required_columns(summary_header, latency_header)
        summary_by_action = cls._index_action_rows(summary_rows)
        latency_by_cell = cls._index_cell_rows(latency_rows)
        cls._verify_inventory(contract, summary_by_action, latency_by_cell)

        summary_header_sha = canonical_sha256(list(summary_header))
        latency_header_sha = canonical_sha256(list(latency_header))

        records = []
        for anchor in sorted(contract.anchors, key=lambda a: a.action_id):
            summary_row = summary_by_action[anchor.action_id]
            cell_rows = {
                profile: latency_by_cell[(anchor.action_id, profile)]
                for profile in NETWORK_PROFILE_ORDER
            }
            records.append(
                cls._build_record(
                    anchor,
                    summary_row,
                    cell_rows,
                    summary_sha,
                    summary_header_sha,
                    latency_sha,
                    latency_header_sha,
                )
            )

        frozen_records = tuple(records)
        by_action = {record.action_id: record for record in frozen_records}
        by_key = {record.key: record for record in frozen_records}
        _require(
            len(by_key) == EXPECTED_PROFILE_COUNT,
            f"expected {EXPECTED_PROFILE_COUNT} distinct "
            f"(family, quantizer, q_e4) anchors, got {len(by_key)}",
        )
        by_mode: Dict[int, list] = {mode.mode_id: [] for mode in contract.modes}
        for record in frozen_records:
            by_mode[record.quality.mode_id].append(record)

        return cls(
            action_summary_path=summary_path,
            action_summary_sha256=summary_sha,
            profile_latency_path=latency_path,
            profile_latency_sha256=latency_sha,
            contract=contract,
            records=frozen_records,
            _by_action_id=MappingProxyType(by_action),
            _by_key=MappingProxyType(by_key),
            _by_mode=MappingProxyType(
                {mode_id: tuple(items) for mode_id, items in by_mode.items()}
            ),
        )

    # -- verification ------------------------------------------------------ #

    @staticmethod
    def _verify_required_columns(
        summary_header: Sequence[str], latency_header: Sequence[str]
    ) -> None:
        """Require every registered column in both sources."""
        summary_required = list(_A_IDENTITY_FIELDS) + list(
            _A_PAYLOAD_IDENTITY_FIELDS
        )
        per_profile_bases = (
            list(_A_PROFILE_COUNT_FIELDS)
            + list(_A_PROFILE_RATE_FIELDS)
            + list(_A_PROFILE_OPTIONAL_FLOAT_FIELDS)
            + ["zero_delivery", f"{_A_LATENCY_STAGE}_count"]
            + list(_A_LATENCY_COLUMNS.values())
        )
        for base in per_profile_bases:
            summary_required.extend(
                f"{base}__{profile}" for profile in NETWORK_PROFILE_ORDER
            )
        summary_required.extend(
            ["executed_or_reused_by_profile", "route_summary_available_by_profile"]
        )
        quality_columns = [
            name for name in summary_header if name.startswith("val_")
        ]
        _require(
            len(quality_columns) > 0,
            f"{SOURCE_ACTION_SUMMARY}: no raw val_* quality columns present",
        )
        _require_columns(summary_header, summary_required, SOURCE_ACTION_SUMMARY)

        latency_required = [
            "cell_id",
            "action_id",
            "profile_id",
            "network_profile",
            "family",
            "quantizer",
            "q",
            "median_payload_bytes",
            "network_latency_observed_only",
            "scheduler_arrival_causal_floor_frames",
        ]
        latency_required.extend(_B_COUNT_FIELDS)
        latency_required.extend(_B_RATE_FIELDS)
        latency_required.extend(_B_MAP_SERVICE_FIELDS)
        latency_required.extend(_B_TERMINAL_FIELDS)
        for stage in _B_LATENCY_STAGES:
            latency_required.append(f"{stage}_count")
            latency_required.extend(
                f"{stage}_{percentile}_ms" for percentile in _B_LATENCY_PERCENTILES
            )
        _require_columns(latency_header, latency_required, SOURCE_PROFILE_LATENCY)

    @staticmethod
    def _index_action_rows(
        rows: Sequence[Mapping[str, str]],
    ) -> Mapping[int, Mapping[str, str]]:
        """Index source A by ``action_id``, rejecting duplicates."""
        indexed: Dict[int, Mapping[str, str]] = {}
        for offset, row in enumerate(rows):
            where = f"{SOURCE_ACTION_SUMMARY} row {offset + 2}"
            action_id = _parse_int(row, "action_id", where)
            if action_id in indexed:
                raise EvidenceInventoryError(
                    f"{where}: duplicate action_id {action_id}"
                )
            indexed[action_id] = row
        return indexed

    @staticmethod
    def _index_cell_rows(
        rows: Sequence[Mapping[str, str]],
    ) -> Mapping[Tuple[int, str], Mapping[str, str]]:
        """Index source B by ``(action_id, network_profile)``, rejecting dupes."""
        indexed: Dict[Tuple[int, str], Mapping[str, str]] = {}
        for offset, row in enumerate(rows):
            where = f"{SOURCE_PROFILE_LATENCY} row {offset + 2}"
            action_id = _parse_int(row, "action_id", where)
            profile = _cell(row, "network_profile", where).strip()
            if profile not in NETWORK_PROFILE_ORDER:
                raise EvidenceInventoryError(
                    f"{where}: foreign network_profile {profile!r}; registered "
                    f"profiles are {list(NETWORK_PROFILE_ORDER)}"
                )
            key = (action_id, profile)
            if key in indexed:
                raise EvidenceInventoryError(
                    f"{where}: duplicate cell key {key}"
                )
            indexed[key] = row
        return indexed

    @staticmethod
    def _verify_inventory(
        contract: SplitActionContract,
        summary_by_action: Mapping[int, Mapping[str, str]],
        latency_by_cell: Mapping[Tuple[int, str], Mapping[str, str]],
    ) -> None:
        """Verify the exact registered inventory of both sources.

        Checks 72 unique action anchors, 12 family-quantizer modes with six
        ``q`` anchors each, 288 unique action/profile cells, four profiles per
        action, and the absence of any duplicate, missing or foreign key.
        """
        _require(
            len(NETWORK_PROFILE_ORDER) == EXPECTED_NETWORK_PROFILE_COUNT
            and len(set(NETWORK_PROFILE_ORDER)) == EXPECTED_NETWORK_PROFILE_COUNT,
            "the registered network-profile order is not four distinct labels",
        )
        if contract.anchor_count != EXPECTED_PROFILE_COUNT:
            raise EvidenceInventoryError(
                f"the frozen catalog declares {contract.anchor_count} anchors, "
                f"expected {EXPECTED_PROFILE_COUNT}"
            )
        if contract.mode_count != EXPECTED_MODE_COUNT:
            raise EvidenceInventoryError(
                f"the frozen catalog declares {contract.mode_count} joint "
                f"modes, expected {EXPECTED_MODE_COUNT}"
            )

        catalog_ids = {anchor.action_id for anchor in contract.anchors}
        summary_ids = set(summary_by_action)
        if summary_ids != catalog_ids:
            raise EvidenceInventoryError(
                f"{SOURCE_ACTION_SUMMARY} action inventory does not reconcile "
                f"with the frozen catalog: missing "
                f"{sorted(catalog_ids - summary_ids)}, foreign "
                f"{sorted(summary_ids - catalog_ids)}"
            )
        if len(summary_by_action) != EXPECTED_PROFILE_COUNT:
            raise EvidenceInventoryError(
                f"expected {EXPECTED_PROFILE_COUNT} unique action anchors, got "
                f"{len(summary_by_action)}"
            )

        # 12 modes x 6 q anchors, with the registered anchor set per mode.
        per_mode: Dict[Tuple[str, str], set] = {}
        for anchor in contract.anchors:
            per_mode.setdefault(anchor.mode.key, set()).add(anchor.q_e4)
        if len(per_mode) != EXPECTED_MODE_COUNT:
            raise EvidenceInventoryError(
                f"expected {EXPECTED_MODE_COUNT} family-quantizer modes, got "
                f"{len(per_mode)}"
            )
        for mode_key, anchors in sorted(per_mode.items()):
            if len(anchors) != EXPECTED_Q_ANCHOR_COUNT:
                raise EvidenceInventoryError(
                    f"mode {mode_key} has {len(anchors)} q anchors, expected "
                    f"{EXPECTED_Q_ANCHOR_COUNT}"
                )
            if anchors != set(REGISTERED_Q_ANCHORS_E4):
                raise EvidenceInventoryError(
                    f"mode {mode_key} q anchors {sorted(anchors)} do not match "
                    f"the registered {list(REGISTERED_Q_ANCHORS_E4)}"
                )

        expected_cells = {
            (action_id, profile)
            for action_id in catalog_ids
            for profile in NETWORK_PROFILE_ORDER
        }
        observed_cells = set(latency_by_cell)
        if observed_cells != expected_cells:
            raise EvidenceInventoryError(
                f"{SOURCE_PROFILE_LATENCY} cell inventory does not reconcile: "
                f"missing {sorted(expected_cells - observed_cells)}, foreign "
                f"{sorted(observed_cells - expected_cells)}"
            )
        if len(latency_by_cell) != EXPECTED_CELL_COUNT:
            raise EvidenceInventoryError(
                f"expected {EXPECTED_CELL_COUNT} unique action/profile cells, "
                f"got {len(latency_by_cell)}"
            )
        for action_id in sorted(catalog_ids):
            present = {
                profile
                for profile in NETWORK_PROFILE_ORDER
                if (action_id, profile) in latency_by_cell
            }
            if len(present) != EXPECTED_NETWORK_PROFILE_COUNT:
                raise EvidenceInventoryError(
                    f"action_id {action_id} has {len(present)} network "
                    f"profiles, expected {EXPECTED_NETWORK_PROFILE_COUNT}"
                )

    @staticmethod
    def _reconcile_action_identity(
        anchor: AnchorAction,
        summary_row: Mapping[str, str],
        cell_rows: Mapping[str, Mapping[str, str]],
    ) -> None:
        """Reconcile both sources' action identity with the frozen catalog."""
        where = f"{SOURCE_ACTION_SUMMARY} action_id={anchor.action_id}"
        observed = (
            _cell(summary_row, "profile_id", where),
            _cell(summary_row, "family", where),
            _cell(summary_row, "quantizer", where),
            _parse_int(summary_row, "q_e4", where),
            _parse_int(summary_row, "keep_count", where),
            _parse_int(summary_row, "drop_count", where),
        )
        expected = (
            anchor.profile_id,
            anchor.mode.family,
            anchor.mode.quantizer,
            anchor.q_e4,
            anchor.keep_count,
            anchor.drop_count,
        )
        if observed != expected:
            raise EvidenceInventoryError(
                f"{where}: action identity does not reconcile with the frozen "
                f"catalog: source {observed} vs catalog {expected}"
            )
        summary_q = _parse_float(summary_row, "q", where)
        _require(
            exact_q_e4(summary_q) == anchor.q_e4,
            f"{where}: q {summary_q!r} is not the catalog anchor "
            f"q_e4={anchor.q_e4}",
        )
        _require(
            anchor.keep_count + anchor.drop_count == SPATIAL_CELLS,
            f"{where}: keep+drop does not reconcile to {SPATIAL_CELLS} cells",
        )

        for profile, row in cell_rows.items():
            cell_where = (
                f"{SOURCE_PROFILE_LATENCY} action_id={anchor.action_id} "
                f"profile={profile}"
            )
            cell_observed = (
                _cell(row, "profile_id", cell_where),
                _cell(row, "family", cell_where),
                _cell(row, "quantizer", cell_where),
            )
            cell_expected = (
                anchor.profile_id,
                anchor.mode.family,
                anchor.mode.quantizer,
            )
            if cell_observed != cell_expected:
                raise EvidenceInventoryError(
                    f"{cell_where}: action identity does not reconcile with "
                    f"the frozen catalog: source {cell_observed} vs catalog "
                    f"{cell_expected}"
                )
            cell_q = _parse_float(row, "q", cell_where)
            _require(
                exact_q_e4(cell_q) == anchor.q_e4,
                f"{cell_where}: q {cell_q!r} is not the catalog anchor "
                f"q_e4={anchor.q_e4}",
            )

    # -- record construction ----------------------------------------------- #

    @classmethod
    def _build_record(
        cls,
        anchor: AnchorAction,
        summary_row: Mapping[str, str],
        cell_rows: Mapping[str, Mapping[str, str]],
        summary_sha: str,
        summary_header_sha: str,
        latency_sha: str,
        latency_header_sha: str,
    ) -> MeasuredAnchorRecord:
        """Build one fully verified measured-anchor record."""
        cls._reconcile_action_identity(anchor, summary_row, cell_rows)
        quality = cls._build_quality(
            anchor, summary_row, cell_rows, summary_sha, summary_header_sha
        )
        profiles = {
            profile: cls._build_outcome(
                anchor,
                profile,
                summary_row,
                cell_rows[profile],
                summary_sha,
                summary_header_sha,
                latency_sha,
                latency_header_sha,
            )
            for profile in NETWORK_PROFILE_ORDER
        }
        return MeasuredAnchorRecord(
            quality=quality,
            profiles=MappingProxyType(profiles),
        )

    @staticmethod
    def _build_quality(
        anchor: AnchorAction,
        summary_row: Mapping[str, str],
        cell_rows: Mapping[str, Mapping[str, str]],
        summary_sha: str,
        summary_header_sha: str,
    ) -> ActionQualityAnchor:
        """Build the action-level quality record and prove it is profile-free."""
        where = f"{SOURCE_ACTION_SUMMARY} action_id={anchor.action_id}"
        raw_quality = {
            name: text
            for name, text in summary_row.items()
            if name.startswith("val_")
        }

        # Source B repeats a subset of the action-level quality in every cell.
        # Require byte-identical agreement across all four cells and with
        # source A: that is what makes quality action-level rather than
        # profile-conditioned, and it is verified, not assumed.
        shared = sorted(
            name
            for name in raw_quality
            if all(name in row for row in cell_rows.values())
        )
        _require(
            len(shared) > 0,
            f"{where}: the two sources share no quality column to cross-check",
        )
        for profile, row in cell_rows.items():
            for name in shared:
                _require(
                    row[name] == raw_quality[name],
                    f"{where}: quality field {name!r} differs between "
                    f"{SOURCE_ACTION_SUMMARY} ({raw_quality[name]!r}) and "
                    f"{SOURCE_PROFILE_LATENCY} profile {profile} "
                    f"({row[name]!r}); action-level quality must not depend on "
                    f"the network profile",
                )

        quality_metrics = {}
        for name, text in raw_quality.items():
            stripped = text.strip()
            if stripped == "":
                quality_metrics[name] = None
                continue
            try:
                quality_metrics[name] = _finite(float(stripped), name, where)
            except (ValueError, EvidenceIntegrityError):
                # Non-numeric registered evidence (gate strings, digests,
                # paths, booleans) stays in raw_quality only.
                continue

        # Derived presentation quantities from source B.  DESIGN.md and the
        # analysis artifact both label these presentation coordinates, not a
        # calibrated probability and not the reward; they are kept separate
        # from the raw measured fields for exactly that reason.
        derived_names = (
            "localization_overlap",
            "localization_centroid_rms_m",
            "localization_centroid_score",
            "localization_quality",
            "combined_quality",
        )
        reference_row = cell_rows[NETWORK_PROFILE_ORDER[0]]
        derived: Dict[str, Optional[float]] = {}
        for name in derived_names:
            value = _parse_optional_float(reference_row, name, where)
            for profile, row in cell_rows.items():
                other = _parse_optional_float(row, name, where)
                _require(
                    other == value,
                    f"{where}: derived quality {name!r} differs across network "
                    f"profiles ({value!r} vs {other!r} at {profile}); it must "
                    f"be action-level",
                )
            derived[name] = value

        payload_identity: Dict[str, Any] = {
            "spatial_cells": SPATIAL_CELLS,
            "keep_count": anchor.keep_count,
            "drop_count": anchor.drop_count,
            "bit_width": _parse_optional_int(summary_row, "bit_width", where),
            "latent_width": _parse_optional_int(summary_row, "latent_width", where),
            "wire_layout": _cell(summary_row, "wire_layout", where),
            "zstd_level": _parse_int(summary_row, "zstd_level", where),
            "routing_tag": _parse_int(summary_row, "routing_tag", where),
        }

        return ActionQualityAnchor(
            action_id=anchor.action_id,
            profile_id=anchor.profile_id,
            mode_id=anchor.mode.mode_id,
            family=anchor.mode.family,
            quantizer=anchor.mode.quantizer,
            q=anchor.q,
            q_e4=anchor.q_e4,
            payload_identity=_freeze_str_map(payload_identity),
            raw_quality=_freeze_str_map(raw_quality),
            quality_metrics=_freeze_str_map(quality_metrics),
            derived_presentation_quality=_freeze_str_map(derived),
            source=SOURCE_ACTION_SUMMARY,
            source_sha256=summary_sha,
            source_row_sha256=_row_sha256(
                SOURCE_ACTION_SUMMARY, summary_header_sha, summary_row
            ),
        )

    @staticmethod
    def _build_outcome(
        anchor: AnchorAction,
        profile: str,
        summary_row: Mapping[str, str],
        cell_row: Mapping[str, str],
        summary_sha: str,
        summary_header_sha: str,
        latency_sha: str,
        latency_header_sha: str,
    ) -> NetworkProfileOutcome:
        """Build one cell's network-profile outcome from both sources."""
        where = (
            f"{SOURCE_PROFILE_LATENCY} action_id={anchor.action_id} "
            f"profile={profile}"
        )
        a_where = (
            f"{SOURCE_ACTION_SUMMARY} action_id={anchor.action_id} "
            f"profile={profile}"
        )

        frames_sent = _parse_int(cell_row, "frames_sent", where)
        summary_frames_sent = _parse_int(
            summary_row, f"frames_sent__{profile}", a_where
        )
        _require(
            frames_sent == summary_frames_sent,
            f"{where}: frames_sent {frames_sent} disagrees with "
            f"{SOURCE_ACTION_SUMMARY} ({summary_frames_sent})",
        )
        _require(frames_sent > 0, f"{where}: frames_sent must be positive")

        counts: Dict[str, int] = {}
        for name in _B_COUNT_FIELDS:
            counts[f"replay_v3__{name}"] = _parse_int(cell_row, name, where)
        for name in _A_PROFILE_COUNT_FIELDS:
            counts[f"live__{name}"] = _parse_int(
                summary_row, f"{name}__{profile}", a_where
            )
        counts["frames_sent"] = frames_sent

        # Cross-source reconciliation of the one count both sources measure.
        _require(
            counts["replay_v3__measured_complete_reassemblies"]
            == counts["live__edge_complete_reassemblies"],
            f"{where}: complete-reassembly count disagrees between sources "
            f"({counts['replay_v3__measured_complete_reassemblies']} vs "
            f"{counts['live__edge_complete_reassemblies']})",
        )

        for name, value in counts.items():
            _require(value >= 0, f"{where}: count {name!r} is negative: {value}")

        terminal_counts = {
            name: _parse_int(cell_row, name, where) for name in _B_TERMINAL_FIELDS
        }
        terminal_total = sum(terminal_counts.values())
        _require(
            terminal_total == frames_sent,
            f"{where}: terminal tallies sum to {terminal_total} but "
            f"{frames_sent} frames were sent; terminal accounting must be exact",
        )

        denominators = {
            "frames_sent": frames_sent,
            "replay_v3__rate_denominator": frames_sent,
            "live__rate_denominator": frames_sent,
            "terminal_denominator": frames_sent,
        }

        rates: Dict[str, Optional[float]] = {}
        for name in _B_RATE_FIELDS:
            rates[f"replay_v3__{name}"] = _parse_optional_float(
                cell_row, name, where
            )
        for name in _A_PROFILE_RATE_FIELDS:
            rates[f"live__{name}"] = _parse_optional_float(
                summary_row, f"{name}__{profile}", a_where
            )

        measured_payload_bytes: Dict[str, Optional[float]] = {
            "replay_v3__median_payload_bytes": _parse_optional_float(
                cell_row, "median_payload_bytes", where
            )
        }
        for name in _A_PROFILE_OPTIONAL_FLOAT_FIELDS:
            measured_payload_bytes[f"live__{name}"] = _parse_optional_float(
                summary_row, f"{name}__{profile}", a_where
            )

        map_service = {
            name: _parse_optional_float(cell_row, name, where)
            for name in _B_MAP_SERVICE_FIELDS
        }
        map_service["scheduler_arrival_causal_floor_frames"] = (
            _parse_optional_float(
                cell_row, "scheduler_arrival_causal_floor_frames", where
            )
        )

        latency: Dict[str, LatencyStat] = {}
        for stage in _B_LATENCY_STAGES:
            support = _parse_int(cell_row, f"{stage}_count", where)
            latency[stage] = LatencyStat(
                stage=stage,
                source=SOURCE_PROFILE_LATENCY,
                support=support,
                percentiles_ms=_freeze_str_map(
                    {
                        percentile: _parse_optional_float(
                            cell_row, f"{stage}_{percentile}_ms", where
                        )
                        for percentile in _B_LATENCY_PERCENTILES
                    }
                ),
            )
        install_support = _parse_int(
            summary_row, f"{_A_LATENCY_STAGE}_count__{profile}", a_where
        )
        latency[_A_LATENCY_STAGE] = LatencyStat(
            stage=_A_LATENCY_STAGE,
            source=SOURCE_ACTION_SUMMARY,
            support=install_support,
            percentiles_ms=_freeze_str_map(
                {
                    percentile: _parse_optional_float(
                        summary_row,
                        f"{_A_LATENCY_COLUMNS[percentile]}__{profile}",
                        a_where,
                    )
                    for percentile in _A_LATENCY_PERCENTILES
                }
            ),
        )

        zero_delivery_live = _parse_bool(
            summary_row, f"zero_delivery__{profile}", a_where
        )
        zero_admission_replay = counts["replay_v3__measured_edge_admissions"] == 0

        # The two sources' zero-delivery facts must remain mutually consistent.
        _require(
            zero_delivery_live == (counts["live__maps_installed"] == 0),
            f"{a_where}: zero_delivery flag disagrees with maps_installed",
        )
        _require(
            zero_delivery_live == (not latency["network"].observed),
            f"{where}: zero_delivery={zero_delivery_live} disagrees with the "
            f"network latency support ({latency['network'].support})",
        )
        _require(
            install_support == counts["live__maps_installed"],
            f"{a_where}: install-AoI support {install_support} disagrees with "
            f"maps_installed {counts['live__maps_installed']}",
        )
        _require(
            zero_admission_replay
            == (counts["replay_v3__simulated_map_installs"] == 0),
            f"{where}: zero edge admission disagrees with zero simulated map "
            f"installs",
        )
        # The v3 replay is the strictly more permissive path: it models a
        # renderer-off direct map service and an optimized latest-only
        # scheduler, so it legitimately installs in some cells where the live
        # radio-detour campaign installed nothing.  The implication therefore
        # runs one way only -- if even the permissive replay admitted nothing,
        # the live path cannot have delivered anything.
        _require(
            not zero_admission_replay or zero_delivery_live,
            f"{where}: the v3 replay admitted no feature yet the live campaign "
            f"reports delivery; the permissive replay cannot under-report the "
            f"measured live path",
        )

        execution_provenance = _decode_by_profile(
            _cell(summary_row, "executed_or_reused_by_profile", a_where),
            profile,
            a_where,
        )
        route_summary_text = _decode_by_profile(
            _cell(summary_row, "route_summary_available_by_profile", a_where),
            profile,
            a_where,
        )
        _require(
            route_summary_text in ("True", "False"),
            f"{a_where}: route_summary_available is not a strict boolean: "
            f"{route_summary_text!r}",
        )

        return NetworkProfileOutcome(
            action_id=anchor.action_id,
            network_profile=profile,
            cell_id=_cell(cell_row, "cell_id", where),
            counts=_freeze_str_map(counts),
            denominators=_freeze_str_map(denominators),
            rates=_freeze_str_map(rates),
            measured_payload_bytes=_freeze_str_map(measured_payload_bytes),
            map_service=_freeze_str_map(map_service),
            terminal_counts=_freeze_str_map(terminal_counts),
            latency=_freeze_str_map(latency),
            zero_delivery_live_campaign=zero_delivery_live,
            zero_admission_replay_v3=zero_admission_replay,
            network_latency_observed_only=_parse_bool(
                cell_row, "network_latency_observed_only", where
            ),
            execution_provenance=execution_provenance,
            route_summary_available=route_summary_text == "True",
            source_sha256=_freeze_str_map(
                {
                    SOURCE_ACTION_SUMMARY: summary_sha,
                    SOURCE_PROFILE_LATENCY: latency_sha,
                }
            ),
            source_row_sha256=_freeze_str_map(
                {
                    SOURCE_ACTION_SUMMARY: _row_sha256(
                        SOURCE_ACTION_SUMMARY, summary_header_sha, summary_row
                    ),
                    SOURCE_PROFILE_LATENCY: _row_sha256(
                        SOURCE_PROFILE_LATENCY, latency_header_sha, cell_row
                    ),
                }
            ),
        )

    # -- access ------------------------------------------------------------ #

    @property
    def anchor_count(self) -> int:
        """Number of measured anchor records (72)."""
        return len(self.records)

    @property
    def cell_count(self) -> int:
        """Number of measured action/profile cells (288)."""
        return sum(len(record.profiles) for record in self.records)

    def by_action_id(self, action_id: int) -> MeasuredAnchorRecord:
        """Return the record for a catalog ``action_id``.

        Raises:
            KeyError: if ``action_id`` is not one of the 72 registered actions.
        """
        return self._by_action_id[action_id]

    def records_for_mode(self, mode_id: int) -> Tuple[MeasuredAnchorRecord, ...]:
        """Return a joint mode's six measured anchors, in ascending ``q_e4``."""
        found = self._by_mode[self.contract.mode(mode_id).mode_id]
        return tuple(sorted(found, key=lambda record: record.quality.q_e4))

    def lookup(self, family: str, quantizer: str, q: Any) -> MeasuredAnchorRecord:
        """Return the measured record for an exact ``(family, quantizer, q)``.

        Lookup is exact in both arguments.  ``q`` must be exactly one of the six
        registered anchors; nothing is snapped and nothing is interpolated.

        Raises:
            UnknownJointModeError: if ``(family, quantizer)`` is not a declared
                joint mode.  An undeclared mode is a contract violation,
                distinct from an unmeasured ``q``.
            UnsupportedCounterfactualError: if ``q`` is not a registered anchor
                of this mode.  The error code is ``UNSUPPORTED_COUNTERFACTUAL``.
        """
        mode = self.contract.mode_for(family, quantizer)
        q_e4 = exact_q_e4(q)
        if q_e4 is None:
            raise UnsupportedCounterfactualError(
                f"q={q!r} is not an exact multiple of 1e-4 and therefore names "
                f"no measured anchor of {mode.canonical}; the campaign measured "
                f"only q_e4 in {list(REGISTERED_Q_ANCHORS_E4)}. This store does "
                f"not snap, interpolate or extrapolate."
            )
        return self.lookup_by_q_e4(family, quantizer, q_e4)

    def lookup_by_q_e4(
        self, family: str, quantizer: str, q_e4: int
    ) -> MeasuredAnchorRecord:
        """Return the measured record for an exact ``(family, quantizer, q_e4)``.

        Raises:
            UnknownJointModeError: if the joint mode is not declared.
            UnsupportedCounterfactualError: if ``q_e4`` is not a measured
                anchor of that mode.
        """
        mode = self.contract.mode_for(family, quantizer)
        if isinstance(q_e4, bool) or not isinstance(q_e4, numbers.Integral):
            raise UnsupportedCounterfactualError(
                f"q_e4 must be an integer, got {type(q_e4).__name__}: {q_e4!r}"
            )
        key = (mode.family, mode.quantizer, int(q_e4))
        try:
            return self._by_key[key]
        except KeyError:
            raise UnsupportedCounterfactualError(
                f"q_e4={int(q_e4)} is not a measured anchor of "
                f"{mode.canonical}; measured anchors are "
                f"{list(REGISTERED_Q_ANCHORS_E4)}. This store does not snap, "
                f"interpolate or extrapolate."
            ) from None

    # -- serialization ----------------------------------------------------- #

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Deterministic serializable form of the whole bound store."""
        return {
            "schema": STORE_SCHEMA_ID,
            "evidence_class": EVIDENCE_CLASS,
            "evidence_use_restriction": EVIDENCE_USE_RESTRICTION,
            "replay_restriction": REPLAY_INSERTION_RESTRICTION,
            "sources": {
                SOURCE_ACTION_SUMMARY: {
                    "relative_path": ACTION_SUMMARY_RELATIVE_PATH,
                    "sha256": self.action_summary_sha256,
                },
                SOURCE_PROFILE_LATENCY: {
                    "relative_path": PROFILE_LATENCY_RELATIVE_PATH,
                    "sha256": self.profile_latency_sha256,
                },
            },
            "catalog_sha256": self.contract.catalog_sha256,
            "inventory": {
                "anchor_count": self.anchor_count,
                "cell_count": self.cell_count,
                "mode_count": self.contract.mode_count,
                "q_anchor_count": EXPECTED_Q_ANCHOR_COUNT,
                "network_profiles": list(NETWORK_PROFILE_ORDER),
            },
            "records": [
                record.to_canonical_dict()
                for record in sorted(self.records, key=lambda r: r.action_id)
            ],
        }

    def canonical_bytes(self) -> bytes:
        """Canonical JSON bytes of :meth:`to_canonical_dict`."""
        return canonical_json_bytes(self.to_canonical_dict())

    def canonical_sha256(self) -> str:
        """SHA-256 of :meth:`canonical_bytes`."""
        return canonical_sha256(self.to_canonical_dict())


def _decode_by_profile(encoded: str, profile: str, where: str) -> str:
    """Decode a ``A=x|B=y`` per-profile string, returning ``profile``'s value."""
    entries = {}
    for chunk in encoded.split("|"):
        if "=" not in chunk:
            raise EvidenceIntegrityError(
                f"{where}: malformed per-profile field {encoded!r}"
            )
        name, _, value = chunk.partition("=")
        entries[name] = value
    if profile not in entries:
        raise EvidenceIntegrityError(
            f"{where}: per-profile field {encoded!r} has no entry for {profile}"
        )
    return entries[profile]


# --------------------------------------------------------------------------- #
# Module-level entry points (no work happens at import time)
# --------------------------------------------------------------------------- #


def load_anchor_store(
    action_summary_path: Optional[Path] = None,
    profile_latency_path: Optional[Path] = None,
    contract: Optional[SplitActionContract] = None,
) -> AnchorEvidenceStore:
    """Read and verify both pinned sources, returning a fresh bound store."""
    return AnchorEvidenceStore.from_paths(
        action_summary_path, profile_latency_path, contract
    )


@lru_cache(maxsize=1)
def default_anchor_store() -> AnchorEvidenceStore:
    """Return a process-cached store bound to the default evidence paths.

    The sources are read on the first explicit call, never at import time.
    """
    return AnchorEvidenceStore.from_paths()
