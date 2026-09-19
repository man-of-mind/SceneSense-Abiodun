"""Strict offline payload/network surrogate for SplitFusion Hybrid-SAC.

This module turns the *aggregate* 288-cell campaign evidence exposed by
``AnchorEvidenceStore`` into four profile-specific, payload-conditioned
outcome models:

* ``P(complete reassembly | frame sent)``;
* ``P(edge admission | complete reassembly)``;
* their coherent product, ``P(edge admission | frame sent)``; and
* conditional feature-uplink p50/p95/p99 for the admitted, downstream-result-
  retained survivor population.

The surrogate is deliberately narrow.  It is a profile-marginal campaign-cell
aggregate for provisional curriculum/surface modeling, not a per-frame causal
radio model, policy input, or new measurement evidence.  It has no aligned
per-frame CSI and no temporal/channel-memory model.  Every prediction is
labelled :data:`EVIDENCE_CLASS` (``OFFLINE_MODELED_FROM_MEASURED_ANCHORS``).
The authored network profile is privileged simulator context and
:meth:`NetworkSurrogatePrediction.as_policy_observation` always refuses it.

Scientific boundaries
---------------------

The independent variable is the measured median compressed payload in bytes.
The measured UDP datagram count is a second support coordinate: it is checked
exactly against the production 12,500-byte datagram / 8-byte header / 12,492-
byte payload-capacity contract, but is not added as a second regressor because
it is a deterministic quantization of payload size.
Treating the two collinear quantities as independent predictors would imply
support that the 288 cells do not contain.

Rates are fitted with denominator-weighted isotonic regression (PAVA), with
larger payload constrained not to improve delivery.  The three conditional
latency percentiles use support-weighted isotonic fits with larger payload
constrained not to reduce latency.  The interval itself ends at complete edge
reassembly, but the old runtime returned that timestamp to the UE only inside
a later downstream result.  Its observed population is therefore necessarily
edge-admitted *and* downstream-result-retained.  It is not a causal arrival-
latency distribution for every reassembled frame and must never be consumed
before admission in a scheduler simulation.  Delivery failures and cells with
fewer than 100 retained timing samples are excluded rather than zero-imputed.
The fitted distribution is consequently downstream-survivor/missingness
selected.  Queries outside the measured probability or qualified latency
envelope fail closed.

Wilson intervals are labelled only as binomial reference intervals around
the fitted rate.  They are not predictive uncertainty.  The latency result
contains a deterministic held-mode-out residual band.  A block/bootstrap
interval is not implemented because the bound sources contain campaign-cell
aggregates rather than the ordered per-frame blocks required to construct one
honestly.

Validation leaves out an entire joint mode (all six q anchors) at a time.  It
therefore measures interpolation to an unseen family/quantizer mode rather
than leaking another q anchor from the held mode into the fit.  The resulting
metrics are diagnostics of this fixed aggregate data set, not generalization
claims about a live radio.

No function in this module launches CARLA, OAI, Docker, CUDA or a network
service.  The CLI writes nothing; it prints a preflight/report or one modeled
prediction to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .anchor_store import (
    ACTION_SUMMARY_SHA256,
    NETWORK_PROFILE_ORDER,
    PROFILE_LATENCY_SHA256,
    AnchorEvidenceStore,
    load_anchor_store,
)
from .action_contract import CATALOG_SHA256
from .transaction_identity import canonical_sha256

__all__ = [
    "EVIDENCE_CLASS",
    "NETWORK_SURROGATE_SCHEMA",
    "CrossValidationMetric",
    "EvidenceDefinitionError",
    "ExtrapolationRefusedError",
    "LatencyPrediction",
    "NetworkSurrogateError",
    "NetworkSurrogatePrediction",
    "PayloadNetworkSurrogate",
    "PrivilegedContextLeakError",
    "SurrogateContract",
    "SupportSummary",
    "build_payload_network_surrogate",
    "main",
]


NETWORK_SURROGATE_SCHEMA = "splitfusion_payload_network_surrogate_v1"
EVIDENCE_CLASS = "OFFLINE_MODELED_FROM_MEASURED_ANCHORS"
NETWORK_STAGE = "network"
UDP_DATAGRAM_BYTES_INCLUDING_HEADER = 12_500
UDP_CHUNK_HEADER_BYTES = 8
UDP_PAYLOAD_CAPACITY_BYTES = 12_492
LATENCY_MIN_SUPPORT = 100
BINOMIAL_REFERENCE_Z = 1.959963984540054
PRODUCTION_TRANSPORT_RELATIVE_PATH = "phase2_map_sharing/transport.py"
PRODUCTION_TRANSPORT_SHA256 = (
    "c8d8d0b253356c11776e9c35b7d6b1bef009bfc980e4cecfcbba84ae95734a6e"
)
PRODUCTION_RUNTIME_CONTRACT_RELATIVE_PATH = (
    "rl_agent/ue_route_b_split_cell_adapter_v1.py"
)
PRODUCTION_RUNTIME_CONTRACT_SHA256 = (
    "3589c7b1366d7b2309cf142b9210116697f14b4c9ad4f42de80ddc896803ae5b"
)
SOURCE_ANALYSIS_BUILDER_RELATIVE_PATH = (
    "rl_agent/splitfusion_supervisor_analysis_v1/build_analysis.py"
)
SOURCE_ANALYSIS_BUILDER_SHA256 = (
    "f11f6f4baafed85cf6cd00012040604915dab236fa1f4567cf5b9098cec9b127"
)
SOURCE_ANALYSIS_ROOT_RELATIVE_PATH = (
    "experiments/splitfusion_supervisor_analysis_v1/"
    "20260915_tail_completion_feedback_policy_analysis_v3"
)
SOURCE_ANALYSIS_SUMMARY_RELATIVE_PATH = (
    f"{SOURCE_ANALYSIS_ROOT_RELATIVE_PATH}/analysis_summary.json"
)
SOURCE_ANALYSIS_SUMMARY_SHA256 = (
    "6daf9553aef65d3950d22e45714198284e73e08f57d0e11bdde87c358763e3f0"
)
SOURCE_ANALYSIS_MANIFEST_RELATIVE_PATH = (
    f"{SOURCE_ANALYSIS_ROOT_RELATIVE_PATH}/artifact_manifest.json"
)
SOURCE_ANALYSIS_MANIFEST_SHA256 = (
    "e3fededc38ab259a5524a945421fb40c8118e65e6621884eff9f8e48bd5fc588"
)
SURROGATE_IMPLEMENTATION_RELATIVE_PATH = (
    "rl_agent/splitfusion_hybrid_sac_v1/payload_network_surrogate.py"
)
LATENCY_EVENT_DEFINITION = (
    "SEND_FINISHED_TO_COMPLETE_EDGE_REASSEMBLY_BOUNDARY_ON_EDGE_ADMITTED_"
    "DOWNSTREAM_RESULT_RETAINED_SURVIVORS"
)
LATENCY_SELECTION_DISCLOSURE = (
    "INTERVAL_ENDS_AT_COMPLETE_REASSEMBLY_BUT_TIMESTAMP_IS_OBSERVED_ONLY_"
    "WHEN_A_LATER_EDGE_RESULT_RETURNS; POPULATION_IS_NECESSARILY_EDGE_"
    "ADMITTED_AND_DOWNSTREAM_RESULT_RETAINED; NOT_CAUSAL_PRE_ADMISSION_"
    "ARRIVAL_LATENCY; CELL_SUPPORT_MUST_BE_AT_LEAST_100"
)
UNCERTAINTY_DISCLOSURE = (
    "WILSON_VALUES_ARE_BINOMIAL_REFERENCE_INTERVALS_NOT_PREDICTIVE_"
    "UNCERTAINTY; LATENCY_BANDS_ARE_DETERMINISTIC_HELD_MODE_RESIDUAL_"
    "DIAGNOSTICS; BLOCK_BOOTSTRAP_NOT_IMPLEMENTED_BECAUSE_ONLY_AGGREGATE_"
    "CELL_EVIDENCE_IS_BOUND"
)


class NetworkSurrogateError(Exception):
    """Base class for strict surrogate failures."""


class EvidenceDefinitionError(NetworkSurrogateError):
    """The bound evidence cannot support the declared surrogate definition."""


class ExtrapolationRefusedError(NetworkSurrogateError):
    """A query falls outside measured payload/fragmentation support."""


class PrivilegedContextLeakError(NetworkSurrogateError):
    """Authored simulator context was offered to the deployed policy."""


@dataclass(frozen=True, slots=True)
class SurrogateContract:
    """Every non-evidence constant that can affect a modeled output."""

    schema: str
    evidence_class: str
    network_stage: str
    udp_datagram_bytes_including_header: int
    udp_chunk_header_bytes: int
    udp_payload_capacity_bytes: int
    latency_min_support: int
    binomial_reference_z: float
    production_transport_relative_path: str
    production_transport_sha256: str
    production_runtime_contract_relative_path: str
    production_runtime_contract_sha256: str
    source_analysis_builder_relative_path: str
    source_analysis_builder_sha256: str
    source_analysis_summary_relative_path: str
    source_analysis_summary_sha256: str
    source_analysis_manifest_relative_path: str
    source_analysis_manifest_sha256: str
    surrogate_implementation_relative_path: str
    latency_event_definition: str
    latency_selection_disclosure: str
    uncertainty_disclosure: str
    rate_fit: str
    latency_fit: str
    interpolation: str
    extrapolation: str

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in (
                "schema",
                "evidence_class",
                "network_stage",
                "udp_datagram_bytes_including_header",
                "udp_chunk_header_bytes",
                "udp_payload_capacity_bytes",
                "latency_min_support",
                "binomial_reference_z",
                "production_transport_relative_path",
                "production_transport_sha256",
                "production_runtime_contract_relative_path",
                "production_runtime_contract_sha256",
                "source_analysis_builder_relative_path",
                "source_analysis_builder_sha256",
                "source_analysis_summary_relative_path",
                "source_analysis_summary_sha256",
                "source_analysis_manifest_relative_path",
                "source_analysis_manifest_sha256",
                "surrogate_implementation_relative_path",
                "latency_event_definition",
                "latency_selection_disclosure",
                "uncertainty_disclosure",
                "rate_fit",
                "latency_fit",
                "interpolation",
                "extrapolation",
            )
        }


def _registered_contract() -> SurrogateContract:
    return SurrogateContract(
        schema=NETWORK_SURROGATE_SCHEMA,
        evidence_class=EVIDENCE_CLASS,
        network_stage=NETWORK_STAGE,
        udp_datagram_bytes_including_header=(
            UDP_DATAGRAM_BYTES_INCLUDING_HEADER
        ),
        udp_chunk_header_bytes=UDP_CHUNK_HEADER_BYTES,
        udp_payload_capacity_bytes=UDP_PAYLOAD_CAPACITY_BYTES,
        latency_min_support=LATENCY_MIN_SUPPORT,
        binomial_reference_z=BINOMIAL_REFERENCE_Z,
        production_transport_relative_path=PRODUCTION_TRANSPORT_RELATIVE_PATH,
        production_transport_sha256=PRODUCTION_TRANSPORT_SHA256,
        production_runtime_contract_relative_path=(
            PRODUCTION_RUNTIME_CONTRACT_RELATIVE_PATH
        ),
        production_runtime_contract_sha256=(
            PRODUCTION_RUNTIME_CONTRACT_SHA256
        ),
        source_analysis_builder_relative_path=(
            SOURCE_ANALYSIS_BUILDER_RELATIVE_PATH
        ),
        source_analysis_builder_sha256=SOURCE_ANALYSIS_BUILDER_SHA256,
        source_analysis_summary_relative_path=(
            SOURCE_ANALYSIS_SUMMARY_RELATIVE_PATH
        ),
        source_analysis_summary_sha256=SOURCE_ANALYSIS_SUMMARY_SHA256,
        source_analysis_manifest_relative_path=(
            SOURCE_ANALYSIS_MANIFEST_RELATIVE_PATH
        ),
        source_analysis_manifest_sha256=SOURCE_ANALYSIS_MANIFEST_SHA256,
        surrogate_implementation_relative_path=(
            SURROGATE_IMPLEMENTATION_RELATIVE_PATH
        ),
        latency_event_definition=LATENCY_EVENT_DEFINITION,
        latency_selection_disclosure=LATENCY_SELECTION_DISCLOSURE,
        uncertainty_disclosure=UNCERTAINTY_DISCLOSURE,
        rate_fit="DENOMINATOR_WEIGHTED_PAVA_NONINCREASING",
        latency_fit=(
            "SUPPORT_WEIGHTED_PAVA_NONDECREASING_SUPPORT_GE_100"
        ),
        interpolation="LINEAR_BETWEEN_PAVA_BLOCK_CENTERS_ENDPOINT_CONSTANT",
        extrapolation="FAIL_CLOSED",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceDefinitionError(message)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtrapolationRefusedError(
            f"{name} must be a finite non-negative number, got {value!r}"
        )
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ExtrapolationRefusedError(
            f"{name} must be finite and non-negative, got {value!r}"
        )
    return parsed


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_bound_source(
    relative_path: str, expected_sha256: str, label: str
) -> bytes:
    path = _project_root() / relative_path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvidenceDefinitionError(
            f"cannot read {label} source {path}: {exc}"
        ) from exc
    digest = hashlib.sha256(raw).hexdigest()
    _require(
        digest == expected_sha256,
        f"{label} SHA-256 drift: expected {expected_sha256}, got {digest}",
    )
    return raw


def _current_implementation_sha256(contract: SurrogateContract) -> str:
    path = _project_root() / contract.surrogate_implementation_relative_path
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvidenceDefinitionError(
            f"cannot read surrogate implementation {path}: {exc}"
        ) from exc


def _verify_analysis_semantic_sources(contract: SurrogateContract) -> None:
    """Bind the source-B derivation code and its semantic metadata chain."""
    _read_bound_source(
        contract.source_analysis_builder_relative_path,
        contract.source_analysis_builder_sha256,
        "source analysis builder",
    )
    summary_raw = _read_bound_source(
        contract.source_analysis_summary_relative_path,
        contract.source_analysis_summary_sha256,
        "source analysis summary",
    )
    manifest_raw = _read_bound_source(
        contract.source_analysis_manifest_relative_path,
        contract.source_analysis_manifest_sha256,
        "source analysis manifest",
    )
    try:
        summary = json.loads(summary_raw.decode("utf-8"))
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceDefinitionError(
            f"source analysis semantic metadata is malformed: {exc}"
        ) from exc
    _require(
        summary.get("schema")
        == "scenesense.splitfusion.supervisor_analysis.tail_completion_feedback.v3",
        "source analysis summary schema drifted",
    )
    _require(
        (summary.get("source_bindings") or {}).get("builder_sha256")
        == contract.source_analysis_builder_sha256,
        "source analysis summary does not bind the registered builder",
    )
    _require(
        (summary.get("component_boundaries") or {}).get("network")
        == "send_finished_ns to complete edge reassembly, observed receipts only",
        "source analysis network boundary drifted",
    )
    _require(
        (summary.get("denominators") or {}).get("network")
        == "frames with retained observed complete edge receipt",
        "source analysis network denominator drifted",
    )
    manifest_hashes = manifest.get("sha256") or {}
    _require(
        manifest_hashes.get("action_profile_quality_latency.csv")
        == PROFILE_LATENCY_SHA256,
        "source analysis manifest does not bind the registered profile table",
    )
    _require(
        manifest_hashes.get("analysis_summary.json")
        == contract.source_analysis_summary_sha256,
        "source analysis manifest does not bind the registered semantic summary",
    )


def _verify_production_fragmentation(contract: SurrogateContract) -> None:
    """Bind the exact production chunk-header source and 12,500-byte setup."""
    _read_bound_source(
        contract.production_transport_relative_path,
        contract.production_transport_sha256,
        "production transport",
    )
    _read_bound_source(
        contract.production_runtime_contract_relative_path,
        contract.production_runtime_contract_sha256,
        "production runtime contract",
    )
    from phase2_map_sharing.transport import CHUNK_HEADER

    _require(
        CHUNK_HEADER.size == contract.udp_chunk_header_bytes,
        f"production header is {CHUNK_HEADER.size} bytes, contract binds "
        f"{contract.udp_chunk_header_bytes}",
    )
    _require(
        contract.udp_datagram_bytes_including_header
        - contract.udp_chunk_header_bytes
        == contract.udp_payload_capacity_bytes,
        "UDP datagram/header/capacity constants do not reconcile",
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    """Deterministic nearest-rank percentile (no dependency on NumPy)."""
    if not values:
        raise EvidenceDefinitionError("cannot compute a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise EvidenceDefinitionError(f"invalid percentile {probability!r}")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _r_squared(observed: Sequence[float], predicted: Sequence[float]) -> float:
    if len(observed) != len(predicted) or not observed:
        raise EvidenceDefinitionError("R2 inputs must be non-empty and aligned")
    mean = sum(observed) / len(observed)
    total = sum((value - mean) ** 2 for value in observed)
    residual = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(observed, predicted)
    )
    if total == 0.0:
        return 1.0 if residual == 0.0 else float("-inf")
    return 1.0 - residual / total


def _wilson_interval(
    probability: float, support: float, z: float
) -> Tuple[float, float]:
    """Binomial reference interval, not modeled predictive uncertainty."""
    if support <= 0.0:
        raise EvidenceDefinitionError("Wilson support must be positive")
    p = min(1.0, max(0.0, probability))
    z2 = z * z
    denominator = 1.0 + z2 / support
    center = (p + z2 / (2.0 * support)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * support)) / support)
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True, slots=True)
class _Observation:
    action_id: int
    mode_id: int
    network_profile: str
    payload_bytes: float
    datagram_count: int
    frames_sent: int
    complete_reassemblies: int
    edge_admissions: int
    network_support: int
    network_p50_ms: Optional[float]
    network_p95_ms: Optional[float]
    network_p99_ms: Optional[float]

    @property
    def x(self) -> float:
        return math.log(self.payload_bytes)

    @property
    def reassembly_rate(self) -> float:
        return self.complete_reassemblies / self.frames_sent

    @property
    def admission_given_reassembly(self) -> float:
        return self.edge_admissions / self.complete_reassemblies

    @property
    def admission_per_sent(self) -> float:
        return self.edge_admissions / self.frames_sent

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "action_id",
                "mode_id",
                "network_profile",
                "payload_bytes",
                "datagram_count",
                "frames_sent",
                "complete_reassemblies",
                "edge_admissions",
                "network_support",
                "network_p50_ms",
                "network_p95_ms",
                "network_p99_ms",
            )
        }


@dataclass(frozen=True, slots=True)
class _PavaBlock:
    x_min: float
    x_max: float
    weight: float
    weighted_sum: float
    members: int

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def value(self) -> float:
        return self.weighted_sum / self.weight


@dataclass(frozen=True, slots=True)
class _CurveValue:
    value: float
    effective_support: float


@dataclass(frozen=True, slots=True)
class _MonotoneCurve:
    """One denominator/support-weighted isotonic curve."""

    increasing: bool
    raw_x_min: float
    raw_x_max: float
    blocks: Tuple[_PavaBlock, ...]

    @classmethod
    def fit(
        cls,
        points: Iterable[Tuple[float, float, float]],
        *,
        increasing: bool,
    ) -> "_MonotoneCurve":
        ordered = sorted(
            (float(x), float(y), float(weight)) for x, y, weight in points
        )
        _require(len(ordered) >= 2, "a monotone curve needs at least two points")
        blocks = []
        for x, y, weight in ordered:
            _require(math.isfinite(x), f"non-finite curve coordinate {x!r}")
            _require(math.isfinite(y), f"non-finite curve value {y!r}")
            _require(
                math.isfinite(weight) and weight > 0.0,
                f"curve weight must be positive and finite, got {weight!r}",
            )
            blocks.append(_PavaBlock(x, x, weight, weight * y, 1))
            while len(blocks) >= 2:
                left, right = blocks[-2], blocks[-1]
                violation = (
                    left.value > right.value
                    if increasing
                    else left.value < right.value
                )
                if not violation:
                    break
                blocks[-2:] = [
                    _PavaBlock(
                        x_min=left.x_min,
                        x_max=right.x_max,
                        weight=left.weight + right.weight,
                        weighted_sum=left.weighted_sum + right.weighted_sum,
                        members=left.members + right.members,
                    )
                ]
        return cls(
            increasing=increasing,
            raw_x_min=ordered[0][0],
            raw_x_max=ordered[-1][0],
            blocks=tuple(blocks),
        )

    def predict(self, x: float) -> _CurveValue:
        if x < self.raw_x_min or x > self.raw_x_max:
            raise ExtrapolationRefusedError(
                f"log-payload {x:.9f} is outside fitted support "
                f"[{self.raw_x_min:.9f}, {self.raw_x_max:.9f}]"
            )
        if x <= self.blocks[0].x_center:
            block = self.blocks[0]
            return _CurveValue(block.value, block.weight)
        if x >= self.blocks[-1].x_center:
            block = self.blocks[-1]
            return _CurveValue(block.value, block.weight)

        for left, right in zip(self.blocks, self.blocks[1:]):
            if x <= right.x_center:
                span = right.x_center - left.x_center
                fraction = 0.0 if span == 0.0 else (x - left.x_center) / span
                value = left.value + fraction * (right.value - left.value)
                # The smaller adjacent pool is a conservative local support.
                support = min(left.weight, right.weight)
                return _CurveValue(value, support)
        raise AssertionError("reachable x was not bracketed by curve blocks")

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "increasing": self.increasing,
            "raw_log_payload_min": self.raw_x_min,
            "raw_log_payload_max": self.raw_x_max,
            "blocks": [
                {
                    "log_payload_min": block.x_min,
                    "log_payload_max": block.x_max,
                    "weight": block.weight,
                    "value": block.value,
                    "members": block.members,
                }
                for block in self.blocks
            ],
        }


@dataclass(frozen=True, slots=True)
class CrossValidationMetric:
    """Held-entire-mode-out diagnostic for one modeled outcome."""

    name: str
    evaluated: int
    unsupported: int
    mae: float
    p90_absolute_error: float
    r_squared: float

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "evaluated": self.evaluated,
            "unsupported": self.unsupported,
            "mae": self.mae,
            "p90_absolute_error": self.p90_absolute_error,
            "r_squared": self.r_squared,
        }


@dataclass(frozen=True, slots=True)
class SupportSummary:
    """Measured envelope and effective local support for one prediction."""

    payload_min_bytes: float
    payload_max_bytes: float
    datagram_min: int
    datagram_max: int
    reassembly_effective_frames: float
    admission_effective_reassemblies: float
    latency_effective_samples: Optional[float]
    latency_min_support: int
    latency_supported: bool

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "payload_min_bytes": self.payload_min_bytes,
            "payload_max_bytes": self.payload_max_bytes,
            "datagram_min": self.datagram_min,
            "datagram_max": self.datagram_max,
            "reassembly_effective_frames": self.reassembly_effective_frames,
            "admission_effective_reassemblies": (
                self.admission_effective_reassemblies
            ),
            "latency_effective_samples": self.latency_effective_samples,
            "latency_min_support": self.latency_min_support,
            "latency_supported": self.latency_supported,
        }


@dataclass(frozen=True, slots=True)
class LatencyPrediction:
    """Retained-survivor uplink-boundary latency and CV residual diagnostic."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    effective_support: float
    held_mode_absolute_residual_p90_ms: Mapping[str, float]
    held_mode_residual_band_ms: Mapping[str, Tuple[float, float]]

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "condition": LATENCY_EVENT_DEFINITION,
            "selection": LATENCY_SELECTION_DISCLOSURE,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "effective_support": self.effective_support,
            "held_mode_absolute_residual_p90_ms": dict(
                self.held_mode_absolute_residual_p90_ms
            ),
            "held_mode_residual_band_ms": {
                name: list(interval)
                for name, interval in self.held_mode_residual_band_ms.items()
            },
            "residual_band_interpretation": (
                "DETERMINISTIC_HELD_MODE_DIAGNOSTIC_NOT_A_PREDICTIVE_"
                "CONFIDENCE_INTERVAL"
            ),
        }


@dataclass(frozen=True, slots=True)
class NetworkSurrogatePrediction:
    """One modeled transport result under privileged simulator context."""

    network_profile: str
    payload_bytes: float
    datagram_count: int
    p_complete_reassembly_given_sent: float
    p_edge_admission_given_reassembled: float
    p_edge_admission_given_sent: float
    reassembly_binomial_reference_interval95: Tuple[float, float]
    admission_binomial_reference_interval95: Tuple[float, float]
    _latency_on_admitted_retained_survivor: Optional[LatencyPrediction]
    support: SupportSummary
    latency_event_definition: str
    latency_selection_disclosure: str
    uncertainty_disclosure: str
    schema: str = NETWORK_SURROGATE_SCHEMA
    evidence_class: str = EVIDENCE_CLASS
    policy_observation_admissible: bool = False

    def as_policy_observation(self) -> None:
        raise PrivilegedContextLeakError(
            "authored network_profile is privileged offline-simulator context "
            "and must not enter the deployed policy observation"
        )

    def require_latency(
        self,
        *,
        edge_admission_succeeded: bool,
        downstream_result_retained: bool,
    ) -> LatencyPrediction:
        """Return latency only for the population that actually retained it.

        Although the measured interval ends at complete reassembly, the old
        runtime carried that timestamp back only in a later edge result.  A
        reassembly flag alone is therefore insufficient and intentionally is
        not accepted by this API.
        """
        if (
            edge_admission_succeeded is not True
            or downstream_result_retained is not True
        ):
            raise ExtrapolationRefusedError(
                "the retained feature-uplink boundary distribution may be "
                "consumed only for an edge-admitted frame whose downstream "
                "result retained the edge receipt timestamp; it is not causal "
                "pre-admission arrival latency"
            )
        if self._latency_on_admitted_retained_survivor is None:
            raise ExtrapolationRefusedError(
                "payload is inside delivery-rate support but outside the "
                "qualified retained-survivor latency envelope (including the "
                "minimum-support gate); no latency was zero-imputed or "
                "extrapolated"
            )
        return self._latency_on_admitted_retained_survivor

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_class": self.evidence_class,
            "network_profile": self.network_profile,
            "payload_bytes": self.payload_bytes,
            "datagram_count": self.datagram_count,
            "probability": {
                "complete_reassembly_given_sent": (
                    self.p_complete_reassembly_given_sent
                ),
                "edge_admission_given_reassembled": (
                    self.p_edge_admission_given_reassembled
                ),
                "edge_admission_given_sent": self.p_edge_admission_given_sent,
                "reassembly_binomial_reference_interval95": list(
                    self.reassembly_binomial_reference_interval95
                ),
                "admission_binomial_reference_interval95": list(
                    self.admission_binomial_reference_interval95
                ),
                "reference_interval_interpretation": (
                    "BINOMIAL_REFERENCE_ONLY_NOT_PREDICTIVE_UNCERTAINTY"
                ),
            },
            "latency": (
                None
                if self._latency_on_admitted_retained_survivor is None
                else self._latency_on_admitted_retained_survivor.to_canonical_dict()
            ),
            "support": self.support.to_canonical_dict(),
            "latency_event_definition": self.latency_event_definition,
            "latency_selection_disclosure": self.latency_selection_disclosure,
            "uncertainty_disclosure": self.uncertainty_disclosure,
            "policy_observation_admissible": self.policy_observation_admissible,
        }


@dataclass(frozen=True, slots=True)
class _ProfileModel:
    profile: str
    reassembly_curve: _MonotoneCurve
    admission_curve: _MonotoneCurve
    latency_curves: Mapping[str, _MonotoneCurve]
    latency_support_knots: Tuple[Tuple[float, int], ...]
    datagram_min: int
    datagram_max: int
    held_mode_absolute_residual_p90_ms: Mapping[str, float]

    def local_latency_support(self, x: float) -> int:
        """Conservative support: minimum of adjacent measured cells."""
        knots = self.latency_support_knots
        if x < knots[0][0] or x > knots[-1][0]:
            return 0
        if x == knots[0][0]:
            return knots[0][1]
        for left, right in zip(knots, knots[1:]):
            if x == right[0]:
                return right[1]
            if x < right[0]:
                return min(left[1], right[1])
        return knots[-1][1]


@dataclass(frozen=True, slots=True)
class PayloadNetworkSurrogate:
    """Profile-conditioned offline model bound to the measured 288 cells."""

    contract: SurrogateContract
    source_action_summary_sha256: str
    source_profile_latency_sha256: str
    catalog_sha256: str
    surrogate_implementation_sha256: str
    observations: Tuple[_Observation, ...]
    profile_models: Mapping[str, _ProfileModel]
    validation: Mapping[str, CrossValidationMetric]
    evidence_class: str = EVIDENCE_CLASS

    @classmethod
    def from_anchor_store(
        cls, store: Optional[AnchorEvidenceStore] = None
    ) -> "PayloadNetworkSurrogate":
        contract = _registered_contract()
        _verify_production_fragmentation(contract)
        _verify_analysis_semantic_sources(contract)
        bound = load_anchor_store() if store is None else store
        _require(
            bound.action_summary_sha256 == ACTION_SUMMARY_SHA256,
            "action-summary source SHA-256 is not the registered pin",
        )
        _require(
            bound.profile_latency_sha256 == PROFILE_LATENCY_SHA256,
            "profile-latency source SHA-256 is not the registered pin",
        )
        _require(
            bound.contract.catalog_sha256 == CATALOG_SHA256,
            "action-catalog source SHA-256 is not the registered pin",
        )
        observations = _extract_observations(bound, contract)
        validation, held_mode_residual_p90_by_profile = _cross_validate(
            observations, contract
        )
        profile_models: Dict[str, _ProfileModel] = {}
        for profile in NETWORK_PROFILE_ORDER:
            rows = tuple(
                row for row in observations if row.network_profile == profile
            )
            profile_models[profile] = _fit_profile(
                profile,
                rows,
                held_mode_residual_p90_by_profile[profile],
                contract,
            )
        model = cls(
            contract=contract,
            source_action_summary_sha256=bound.action_summary_sha256,
            source_profile_latency_sha256=bound.profile_latency_sha256,
            catalog_sha256=bound.contract.catalog_sha256,
            surrogate_implementation_sha256=(
                _current_implementation_sha256(contract)
            ),
            observations=observations,
            profile_models=MappingProxyType(dict(profile_models)),
            validation=MappingProxyType(dict(validation)),
        )
        model.revalidate()
        return model

    def revalidate(self) -> None:
        """Fail closed if a frozen binding or model invariant was tampered."""
        _require(
            self.contract == _registered_contract(),
            "surrogate contract differs from the registered constants",
        )
        _verify_production_fragmentation(self.contract)
        _verify_analysis_semantic_sources(self.contract)
        _require(
            self.evidence_class == self.contract.evidence_class,
            "model evidence class contradicts its contract",
        )
        _require(
            self.source_action_summary_sha256 == ACTION_SUMMARY_SHA256,
            "action-summary source binding drifted",
        )
        _require(
            self.source_profile_latency_sha256 == PROFILE_LATENCY_SHA256,
            "profile-latency source binding drifted",
        )
        _require(
            self.catalog_sha256 == CATALOG_SHA256,
            "catalog source binding drifted",
        )
        _require(
            self.surrogate_implementation_sha256
            == _current_implementation_sha256(self.contract),
            "surrogate implementation source changed after model binding",
        )
        _require(
            tuple(sorted(self.profile_models))
            == tuple(sorted(NETWORK_PROFILE_ORDER)),
            "profile-model inventory drifted",
        )
        _require(len(self.observations) == 288, "observation inventory drifted")
        for profile, model in self.profile_models.items():
            _require(model.profile == profile, "profile-model identity drifted")
            _require(
                set(model.latency_curves) == {"p50", "p95", "p99"},
                f"profile {profile}: latency-curve inventory drifted",
            )
            _require(
                set(model.held_mode_absolute_residual_p90_ms)
                == {"p50", "p95", "p99"},
                f"profile {profile}: residual diagnostic inventory drifted",
            )
            _require(
                len(model.latency_support_knots) == 72,
                f"profile {profile}: latency-support inventory drifted",
            )
            _require(
                tuple(sorted(model.latency_support_knots))
                == model.latency_support_knots,
                f"profile {profile}: latency-support knots are not sorted",
            )

    def predict(
        self,
        *,
        network_profile: str,
        payload_bytes: float,
        datagram_count: int,
    ) -> NetworkSurrogatePrediction:
        self.revalidate()
        if network_profile not in self.profile_models:
            raise ExtrapolationRefusedError(
                f"unknown network_profile {network_profile!r}; expected one of "
                f"{list(NETWORK_PROFILE_ORDER)!r}"
            )
        payload = _finite_nonnegative(payload_bytes, "payload_bytes")
        if payload <= 0.0:
            raise ExtrapolationRefusedError("payload_bytes must be positive")
        if isinstance(datagram_count, bool) or not isinstance(datagram_count, int):
            raise ExtrapolationRefusedError(
                f"datagram_count must be an integer, got {datagram_count!r}"
            )
        model = self.profile_models[network_profile]
        if not model.datagram_min <= datagram_count <= model.datagram_max:
            raise ExtrapolationRefusedError(
                f"datagram_count {datagram_count} is outside profile support "
                f"[{model.datagram_min}, {model.datagram_max}]"
            )
        nominal_fragments = math.ceil(
            payload / self.contract.udp_payload_capacity_bytes
        )
        if datagram_count != nominal_fragments:
            raise ExtrapolationRefusedError(
                f"payload/datagram pair ({payload:g} bytes, {datagram_count}) "
                f"is outside the measured fragmentation relation: expected "
                f"exactly {nominal_fragments} datagrams using "
                f"{self.contract.udp_payload_capacity_bytes} payload bytes "
                "per datagram"
            )

        x = math.log(payload)
        reassembly = model.reassembly_curve.predict(x)
        admission = model.admission_curve.predict(x)
        p_reassembly = min(1.0, max(0.0, reassembly.value))
        p_admission_conditional = min(1.0, max(0.0, admission.value))

        latency: Optional[LatencyPrediction] = None
        latency_support: Optional[float] = None
        latency_in_envelope = all(
            curve.raw_x_min <= x <= curve.raw_x_max
            for curve in model.latency_curves.values()
        )
        local_raw_latency_support = model.local_latency_support(x)
        latency_supported = (
            latency_in_envelope
            and local_raw_latency_support >= self.contract.latency_min_support
        )
        if latency_in_envelope:
            latency_support = float(local_raw_latency_support)
        if latency_supported:
            estimates = {
                name: curve.predict(x)
                for name, curve in model.latency_curves.items()
            }
            # Independent isotonic fits can cross by a fraction of a
            # millisecond.  Projecting across quantiles preserves the defining
            # p50 <= p95 <= p99 relation without changing payload monotonicity.
            p50 = estimates["p50"].value
            p95 = max(p50, estimates["p95"].value)
            p99 = max(p95, estimates["p99"].value)
            latency_support = min(
                local_raw_latency_support,
                *(estimate.effective_support for estimate in estimates.values()),
            )
            latency_supported = (
                latency_support >= self.contract.latency_min_support
            )
            if latency_supported:
                point = {"p50": p50, "p95": p95, "p99": p99}
                residual = model.held_mode_absolute_residual_p90_ms
                bands = MappingProxyType(
                    {
                        name: (
                            max(0.0, estimate - residual[name]),
                            estimate + residual[name],
                        )
                        for name, estimate in point.items()
                    }
                )
                latency = LatencyPrediction(
                    p50_ms=p50,
                    p95_ms=p95,
                    p99_ms=p99,
                    effective_support=latency_support,
                    held_mode_absolute_residual_p90_ms=residual,
                    held_mode_residual_band_ms=bands,
                )

        return NetworkSurrogatePrediction(
            network_profile=network_profile,
            payload_bytes=payload,
            datagram_count=datagram_count,
            p_complete_reassembly_given_sent=p_reassembly,
            p_edge_admission_given_reassembled=p_admission_conditional,
            p_edge_admission_given_sent=p_reassembly * p_admission_conditional,
            reassembly_binomial_reference_interval95=_wilson_interval(
                p_reassembly,
                reassembly.effective_support,
                self.contract.binomial_reference_z,
            ),
            admission_binomial_reference_interval95=_wilson_interval(
                p_admission_conditional,
                admission.effective_support,
                self.contract.binomial_reference_z,
            ),
            _latency_on_admitted_retained_survivor=latency,
            support=SupportSummary(
                payload_min_bytes=math.exp(model.reassembly_curve.raw_x_min),
                payload_max_bytes=math.exp(model.reassembly_curve.raw_x_max),
                datagram_min=model.datagram_min,
                datagram_max=model.datagram_max,
                reassembly_effective_frames=reassembly.effective_support,
                admission_effective_reassemblies=admission.effective_support,
                latency_effective_samples=latency_support,
                latency_min_support=self.contract.latency_min_support,
                latency_supported=latency_supported,
            ),
            latency_event_definition=self.contract.latency_event_definition,
            latency_selection_disclosure=(
                self.contract.latency_selection_disclosure
            ),
            uncertainty_disclosure=self.contract.uncertainty_disclosure,
            schema=self.contract.schema,
            evidence_class=self.contract.evidence_class,
        )

    def preflight_document(self) -> Dict[str, Any]:
        self.revalidate()
        zero_latency_cells = sum(row.network_support == 0 for row in self.observations)
        low_support_latency_cells = sum(
            0 < row.network_support < self.contract.latency_min_support
            for row in self.observations
        )
        qualified_latency_cells = sum(
            row.network_support >= self.contract.latency_min_support
            for row in self.observations
        )
        return {
            "schema": self.contract.schema,
            "status": "PREFLIGHT_COMPLETE",
            "evidence_class": self.evidence_class,
            "source_binding": {
                "action_summary_sha256": self.source_action_summary_sha256,
                "profile_latency_sha256": self.source_profile_latency_sha256,
                "catalog_sha256": self.catalog_sha256,
                "production_transport_relative_path": (
                    self.contract.production_transport_relative_path
                ),
                "production_transport_sha256": (
                    self.contract.production_transport_sha256
                ),
                "production_runtime_contract_relative_path": (
                    self.contract.production_runtime_contract_relative_path
                ),
                "production_runtime_contract_sha256": (
                    self.contract.production_runtime_contract_sha256
                ),
                "source_analysis_builder_relative_path": (
                    self.contract.source_analysis_builder_relative_path
                ),
                "source_analysis_builder_sha256": (
                    self.contract.source_analysis_builder_sha256
                ),
                "source_analysis_summary_relative_path": (
                    self.contract.source_analysis_summary_relative_path
                ),
                "source_analysis_summary_sha256": (
                    self.contract.source_analysis_summary_sha256
                ),
                "source_analysis_manifest_relative_path": (
                    self.contract.source_analysis_manifest_relative_path
                ),
                "source_analysis_manifest_sha256": (
                    self.contract.source_analysis_manifest_sha256
                ),
                "surrogate_implementation_relative_path": (
                    self.contract.surrogate_implementation_relative_path
                ),
                "surrogate_implementation_sha256": (
                    self.surrogate_implementation_sha256
                ),
            },
            "inventory": {
                "cells": len(self.observations),
                "profiles": list(NETWORK_PROFILE_ORDER),
                "modes": len({row.mode_id for row in self.observations}),
                "zero_latency_support_cells_excluded": zero_latency_cells,
                "latency_observed_cells": len(self.observations)
                - zero_latency_cells,
                "low_support_latency_cells_excluded": low_support_latency_cells,
                "latency_qualified_cells": qualified_latency_cells,
            },
            "contract": self.contract.to_canonical_dict(),
            "model_scope": {
                "primary_coordinate": "log(measured_median_payload_bytes)",
                "fragmentation_support_coordinate": (
                    "measured_udp_datagrams_per_message"
                ),
                "latency_event_definition": (
                    self.contract.latency_event_definition
                ),
                "latency_selection_disclosure": (
                    self.contract.latency_selection_disclosure
                ),
                "uncertainty_disclosure": self.contract.uncertainty_disclosure,
                "latency_interval_endpoint": "COMPLETE_EDGE_REASSEMBLY",
                "latency_observation_population": (
                    "EDGE_ADMITTED_DOWNSTREAM_RESULT_RETAINED_SURVIVORS"
                ),
                "latency_observation_is_post_admission_selected": True,
                "latency_interval_includes_edge_admission_time": False,
                "causal_pre_admission_arrival_latency_available": False,
                "failed_delivery_latency": "MISSING_NOT_ZERO",
                "network_profile_is_policy_state": False,
                "input_granularity": "CAMPAIGN_CELL_AGGREGATE",
                "payload_coordinate": "CELL_MEDIAN_NOT_PER_FRAME_CAUSAL",
                "per_frame_csi_conditioning": False,
                "temporal_channel_correlation_modeled": False,
                "admissible_use": (
                    "PROVISIONAL_PROFILE_MARGINAL_CURRICULUM_SURFACE_ONLY"
                ),
                "inadmissible_use": (
                    "FINAL_PER_FRAME_CHANNEL_ADAPTIVE_CAUSAL_CLAIM"
                ),
            },
            "validation": {
                name: metric.to_canonical_dict()
                for name, metric in sorted(self.validation.items())
            },
        }

    def canonical_sha256(self) -> str:
        self.revalidate()
        document = self.preflight_document()
        document["observation_rows_sha256"] = canonical_sha256(
            [row.to_canonical_dict() for row in self.observations]
        )
        document["profile_models"] = {
            profile: {
                "reassembly": model.reassembly_curve.to_canonical_dict(),
                "admission_given_reassembly": (
                    model.admission_curve.to_canonical_dict()
                ),
                "latency": {
                    name: curve.to_canonical_dict()
                    for name, curve in sorted(model.latency_curves.items())
                },
                "latency_support_knots": [
                    [x, support]
                    for x, support in model.latency_support_knots
                ],
                "datagram_range": [model.datagram_min, model.datagram_max],
                "held_mode_absolute_residual_p90_ms": dict(
                    model.held_mode_absolute_residual_p90_ms
                ),
            }
            for profile, model in sorted(self.profile_models.items())
        }
        return canonical_sha256(document)

    def report_markdown(self) -> str:
        lines = [
            "# Payload/network surrogate preflight",
            "",
            f"Evidence class: `{self.evidence_class}`.",
            "",
            "This is a profile-marginal campaign-cell aggregate for provisional "
            "curriculum/surface modeling, not a measured or counterfactual "
            "per-frame causal transition. It has no aligned per-frame CSI or "
            "temporal/channel-memory model. The authored network profile is "
            "privileged simulator context and is excluded from the policy "
            "observation.",
            "",
            "Latency ends at complete edge reassembly, but its timestamp was "
            "returned only inside a later edge result. The observed population "
            "is therefore necessarily edge-admitted and downstream-result-"
            "retained. It is not causal pre-admission arrival latency. Failed/"
            "missing receipts carry no latency, and cells below "
            f"{self.contract.latency_min_support} retained timing samples are "
            "excluded, so the model is explicitly downstream-survivor/"
            "missingness selected.",
            "",
            "Wilson values are binomial reference intervals only. The latency "
            "band is a held-mode residual diagnostic, not a predictive "
            "confidence interval. Block/bootstrap uncertainty is not claimed "
            "because ordered per-frame blocks are absent from the bound "
            "aggregate evidence.",
            "",
            "## Held-entire-mode-out validation",
            "",
            "| outcome | evaluated | unsupported | MAE | P90 abs. error | R² |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, metric in sorted(self.validation.items()):
            lines.append(
                f"| {name} | {metric.evaluated} | {metric.unsupported} | "
                f"{metric.mae:.6f} | {metric.p90_absolute_error:.6f} | "
                f"{metric.r_squared:.6f} |"
            )
        lines.extend(
            [
                "",
                "Latency metrics are milliseconds and use only held-out cells "
                f"with at least {self.contract.latency_min_support} observed "
                "timing samples. Cells without timing are never assigned zero.",
                "",
                f"Canonical model digest: `{self.canonical_sha256()}`",
            ]
        )
        return "\n".join(lines) + "\n"


def _extract_observations(
    store: AnchorEvidenceStore, contract: SurrogateContract
) -> Tuple[_Observation, ...]:
    rows = []
    for record in store.records:
        for profile in NETWORK_PROFILE_ORDER:
            outcome = record.outcome(profile)
            payload = outcome.measured_payload_bytes[
                "replay_v3__median_payload_bytes"
            ]
            datagrams = outcome.measured_payload_bytes[
                "live__live_datagrams_per_message_median"
            ]
            _require(payload is not None and payload > 0.0, "missing payload bytes")
            _require(
                datagrams is not None and datagrams > 0.0,
                "missing datagram-count evidence",
            )
            _require(
                float(datagrams).is_integer(),
                f"non-integral datagram median {datagrams!r}",
            )
            expected_datagrams = math.ceil(
                float(payload) / contract.udp_payload_capacity_bytes
            )
            _require(
                int(datagrams) == expected_datagrams,
                f"payload/datagram evidence contradicts the production "
                f"fragmentation contract: {payload} bytes requires "
                f"{expected_datagrams}, source records {datagrams}",
            )
            sent = outcome.counts["replay_v3__frames_sent"]
            reassembled = outcome.counts[
                "replay_v3__measured_complete_reassemblies"
            ]
            admitted = outcome.counts["replay_v3__measured_edge_admissions"]
            _require(sent > 0, "frames_sent must be positive")
            # Every registered cell has at least one reassembly.  Refuse a
            # future source where the admission conditional is undefined.
            _require(
                0 < reassembled <= sent,
                "complete reassemblies must be in [1, frames_sent]",
            )
            _require(
                0 <= admitted <= reassembled,
                "edge admissions must not exceed reassemblies",
            )
            latency = outcome.latency_stat(contract.network_stage)
            _require(
                latency.support <= admitted,
                "retained network-timing support exceeds measured edge "
                "admissions; the registered timestamp was returned only on "
                "the downstream admitted-result path",
            )
            if latency.support == 0:
                _require(
                    latency.p50_ms is None
                    and latency.p95_ms is None
                    and latency.p99_ms is None,
                    "zero latency support carried a value",
                )
            else:
                _require(
                    latency.p50_ms is not None
                    and latency.p95_ms is not None
                    and latency.p99_ms is not None,
                    "positive latency support is missing a percentile",
                )
            rows.append(
                _Observation(
                    action_id=record.action_id,
                    mode_id=record.quality.mode_id,
                    network_profile=profile,
                    payload_bytes=float(payload),
                    datagram_count=int(datagrams),
                    frames_sent=sent,
                    complete_reassemblies=reassembled,
                    edge_admissions=admitted,
                    network_support=latency.support,
                    network_p50_ms=latency.p50_ms,
                    network_p95_ms=latency.p95_ms,
                    network_p99_ms=latency.p99_ms,
                )
            )
    _require(len(rows) == 288, f"expected 288 observations, got {len(rows)}")
    return tuple(rows)


def _fit_profile(
    profile: str,
    rows: Sequence[_Observation],
    held_mode_residual_p90_ms: Mapping[str, float],
    contract: SurrogateContract,
) -> _ProfileModel:
    _require(len(rows) >= 2, f"profile {profile} has insufficient observations")
    reassembly = _MonotoneCurve.fit(
        (
            (row.x, row.reassembly_rate, row.frames_sent)
            for row in rows
        ),
        increasing=False,
    )
    admission = _MonotoneCurve.fit(
        (
            (
                row.x,
                row.admission_given_reassembly,
                row.complete_reassemblies,
            )
            for row in rows
        ),
        increasing=False,
    )
    latency_rows = tuple(
        row
        for row in rows
        if row.network_support >= contract.latency_min_support
    )
    _require(
        len(latency_rows) >= 2,
        f"profile {profile} has fewer than two qualified latency cells",
    )
    latency_curves = {
        percentile: _MonotoneCurve.fit(
            (
                (
                    row.x,
                    float(getattr(row, f"network_{percentile}_ms")),
                    row.network_support,
                )
                for row in latency_rows
            ),
            increasing=True,
        )
        for percentile in ("p50", "p95", "p99")
    }
    return _ProfileModel(
        profile=profile,
        reassembly_curve=reassembly,
        admission_curve=admission,
        latency_curves=MappingProxyType(dict(latency_curves)),
        latency_support_knots=tuple(
            sorted((row.x, row.network_support) for row in rows)
        ),
        datagram_min=min(row.datagram_count for row in rows),
        datagram_max=max(row.datagram_count for row in rows),
        held_mode_absolute_residual_p90_ms=MappingProxyType(
            dict(held_mode_residual_p90_ms)
        ),
    )


def _metric(
    name: str,
    observed: Sequence[float],
    predicted: Sequence[float],
    unsupported: int,
) -> CrossValidationMetric:
    _require(len(observed) == len(predicted), f"{name}: CV vectors differ")
    _require(len(observed) > 0, f"{name}: no CV predictions")
    errors = [abs(actual - estimate) for actual, estimate in zip(observed, predicted)]
    return CrossValidationMetric(
        name=name,
        evaluated=len(observed),
        unsupported=unsupported,
        mae=sum(errors) / len(errors),
        p90_absolute_error=_percentile(errors, 0.90),
        r_squared=_r_squared(observed, predicted),
    )


def _cross_validate(
    observations: Sequence[_Observation],
    contract: SurrogateContract,
) -> Tuple[Mapping[str, CrossValidationMetric], Mapping[str, Mapping[str, float]]]:
    """Leave all six q anchors of one mode out, repeated for all 12 modes."""
    modes = sorted({row.mode_id for row in observations})
    _require(modes == list(range(12)), f"unexpected mode inventory {modes!r}")
    accumulator: Dict[str, Tuple[list, list]] = {
        "reassembly_per_sent": ([], []),
        "admission_given_reassembly": ([], []),
        "admission_per_sent_chain": ([], []),
        "network_p50_ms_support_ge_100": ([], []),
        "network_p95_ms_support_ge_100": ([], []),
        "network_p99_ms_support_ge_100": ([], []),
    }
    unsupported = {name: 0 for name in accumulator}
    latency_errors_by_profile: Dict[str, Dict[str, list]] = {
        profile: {name: [] for name in ("p50", "p95", "p99")}
        for profile in NETWORK_PROFILE_ORDER
    }

    for held_mode in modes:
        for profile in NETWORK_PROFILE_ORDER:
            train = tuple(
                row
                for row in observations
                if row.mode_id != held_mode and row.network_profile == profile
            )
            test = tuple(
                row
                for row in observations
                if row.mode_id == held_mode and row.network_profile == profile
            )
            _require(len(train) == 66, "whole-mode CV train fold is not 66 cells")
            _require(len(test) == 6, "whole-mode CV test fold is not six cells")
            _require(
                held_mode not in {row.mode_id for row in train},
                "held mode leaked into its training fold",
            )

            reassembly_curve = _MonotoneCurve.fit(
                (
                    (row.x, row.reassembly_rate, row.frames_sent)
                    for row in train
                ),
                increasing=False,
            )
            admission_curve = _MonotoneCurve.fit(
                (
                    (
                        row.x,
                        row.admission_given_reassembly,
                        row.complete_reassemblies,
                    )
                    for row in train
                ),
                increasing=False,
            )
            latency_train = tuple(
                row
                for row in train
                if row.network_support >= contract.latency_min_support
            )
            latency_curves = {
                percentile: _MonotoneCurve.fit(
                    (
                        (
                            row.x,
                            float(getattr(row, f"network_{percentile}_ms")),
                            row.network_support,
                        )
                        for row in latency_train
                    ),
                    increasing=True,
                )
                for percentile in ("p50", "p95", "p99")
            }

            for row in test:
                try:
                    reassembly = reassembly_curve.predict(row.x).value
                    admission = admission_curve.predict(row.x).value
                except ExtrapolationRefusedError:
                    for name in (
                        "reassembly_per_sent",
                        "admission_given_reassembly",
                        "admission_per_sent_chain",
                    ):
                        unsupported[name] += 1
                else:
                    accumulator["reassembly_per_sent"][0].append(
                        row.reassembly_rate
                    )
                    accumulator["reassembly_per_sent"][1].append(reassembly)
                    accumulator["admission_given_reassembly"][0].append(
                        row.admission_given_reassembly
                    )
                    accumulator["admission_given_reassembly"][1].append(admission)
                    accumulator["admission_per_sent_chain"][0].append(
                        row.admission_per_sent
                    )
                    accumulator["admission_per_sent_chain"][1].append(
                        reassembly * admission
                    )

                if row.network_support < contract.latency_min_support:
                    continue
                for percentile, curve in latency_curves.items():
                    metric_name = (
                        f"network_{percentile}_ms_support_ge_100"
                    )
                    try:
                        estimate = curve.predict(row.x).value
                    except ExtrapolationRefusedError:
                        unsupported[metric_name] += 1
                        continue
                    actual = float(getattr(row, f"network_{percentile}_ms"))
                    accumulator[metric_name][0].append(actual)
                    accumulator[metric_name][1].append(estimate)
                    latency_errors_by_profile[profile][percentile].append(
                        abs(actual - estimate)
                    )

    metrics = {
        name: _metric(name, observed, predicted, unsupported[name])
        for name, (observed, predicted) in accumulator.items()
    }
    latency_p90 = {
        profile: {
            percentile: _percentile(errors, 0.90)
            for percentile, errors in per_percentile.items()
        }
        for profile, per_percentile in latency_errors_by_profile.items()
    }
    return metrics, latency_p90


def build_payload_network_surrogate() -> PayloadNetworkSurrogate:
    """Load the pinned anchor store and deterministically fit the surrogate."""
    model = PayloadNetworkSurrogate.from_anchor_store()
    _require(
        model.source_action_summary_sha256 == ACTION_SUMMARY_SHA256,
        "action-summary source binding drifted",
    )
    _require(
        model.source_profile_latency_sha256 == PROFILE_LATENCY_SHA256,
        "profile-latency source binding drifted",
    )
    return model


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only SplitFusion payload/network surrogate preflight"
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--preflight", action="store_true")
    operation.add_argument("--report", action="store_true")
    operation.add_argument("--predict", action="store_true")
    parser.add_argument("--network-profile", choices=NETWORK_PROFILE_ORDER)
    parser.add_argument("--payload-bytes", type=float)
    parser.add_argument("--datagram-count", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    model = build_payload_network_surrogate()
    if args.preflight:
        print(json.dumps(model.preflight_document(), indent=2, sort_keys=True))
        return 0
    if args.report:
        print(model.report_markdown(), end="")
        return 0
    if (
        args.network_profile is None
        or args.payload_bytes is None
        or args.datagram_count is None
    ):
        raise SystemExit(
            "--predict requires --network-profile, --payload-bytes and "
            "--datagram-count"
        )
    prediction = model.predict(
        network_profile=args.network_profile,
        payload_bytes=args.payload_bytes,
        datagram_count=args.datagram_count,
    )
    print(json.dumps(prediction.to_canonical_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())
