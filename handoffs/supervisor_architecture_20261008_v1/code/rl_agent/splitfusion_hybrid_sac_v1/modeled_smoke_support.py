"""Immutable support contract for the contextual modeled-smoke curriculum.

``MODELED_SMOKE_SUPPORT`` is a deliberately narrow curriculum support set.  It
was derived only from the 512 fit frames of the hash-bound empirical payload
surface by retaining, for each mode, the inclusive ``q_e4`` interval whose
``total_transmitted_bytes`` lies in the all-network-profile common surrogate
support ``[6423, 427605]``.  The 256 held frames are audit-only: they did not
choose the bounds.

This is contextual smoke support, not a deployment action contract.  In
particular it neither changes nor narrows :mod:`action_contract`, and it must
be selected explicitly by a consumer.  Importing this module performs no file
I/O, samples no RNG, and queries no runtime or accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from . import empirical_quality_surface as quality_surface
from . import payload_network_surrogate as network_surrogate
from .transaction_identity import canonical_sha256

__all__ = [
    "ALL_PROFILE_TOTAL_TRANSMITTED_BYTES_SUPPORT",
    "FIT_FRAME_COUNT",
    "HELD_AUDIT_REFUSED_COUNT",
    "HELD_AUDIT_SUPPORTED_COUNT",
    "HELD_AUDIT_TOTAL_COUNT",
    "MODELED_SMOKE_MODE_Q_E4_BOUNDS",
    "MODELED_SMOKE_SUPPORT",
    "MODELED_SMOKE_SUPPORT_SHA256",
    "MODELED_SMOKE_SUPPORT_SCHEMA",
    "ModeledSmokeSupportContract",
    "ModeledSmokeSupportError",
    "SourceShaPin",
    "require_registered_modeled_smoke_support",
]


MODELED_SMOKE_SUPPORT_SCHEMA = "splitfusion.modeled_smoke_support.v1"

# Inclusive bounds, ordered by the frozen joint mode id 0..11.
MODELED_SMOKE_MODE_Q_E4_BOUNDS: Tuple[Tuple[int, int], ...] = (
    (8812, 9800),
    (8493, 9800),
    (7491, 9800),
    (8280, 9800),
    (7827, 9800),
    (5902, 9800),
    (6603, 9800),
    (5682, 9800),
    (1946, 9800),
    (3110, 9800),
    (1217, 9800),
    (0, 9791),
)

ALL_PROFILE_TOTAL_TRANSMITTED_BYTES_SUPPORT = (6423, 427605)
FIT_FRAME_COUNT = 512
HELD_AUDIT_SUPPORTED_COUNT = 13_373_151
HELD_AUDIT_TOTAL_COUNT = 13_373_440
HELD_AUDIT_REFUSED_COUNT = 289


class ModeledSmokeSupportError(ValueError):
    """A support object is malformed or differs from the registered contract."""


@dataclass(frozen=True, slots=True)
class SourceShaPin:
    """One already-registered evidence/source SHA pin used by the derivation."""

    component: str
    relative_path: str
    sha256: str

    def to_canonical_dict(self) -> Dict[str, str]:
        return {
            "component": self.component,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ModeledSmokeSupportContract:
    """Complete, canonical description of the optional smoke support set."""

    schema: str
    evidence_class: str
    use_scope: str
    deployment_action_contract_status: str
    derivation_split: str
    derivation_method: str
    fit_frame_count: int
    payload_coordinate: str
    all_profile_payload_support: Tuple[int, int]
    mode_q_e4_bounds: Tuple[Tuple[int, int], ...]
    bounds_are_inclusive: bool
    held_audit_frame_count: int
    held_audit_supported_count: int
    held_audit_total_count: int
    held_audit_refused_count: int
    source_sha_pins: Tuple[SourceShaPin, ...]

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return the stable JSON-domain document covered by the contract hash."""
        return {
            "all_profile_payload_support": list(self.all_profile_payload_support),
            "bounds_are_inclusive": self.bounds_are_inclusive,
            "deployment_action_contract_status": (
                self.deployment_action_contract_status
            ),
            "derivation_method": self.derivation_method,
            "derivation_split": self.derivation_split,
            "evidence_class": self.evidence_class,
            "fit_frame_count": self.fit_frame_count,
            "held_audit_frame_count": self.held_audit_frame_count,
            "held_audit_refused_count": self.held_audit_refused_count,
            "held_audit_supported_count": self.held_audit_supported_count,
            "held_audit_total_count": self.held_audit_total_count,
            "mode_q_e4_bounds": [list(pair) for pair in self.mode_q_e4_bounds],
            "payload_coordinate": self.payload_coordinate,
            "schema": self.schema,
            "source_sha_pins": [
                pin.to_canonical_dict() for pin in self.source_sha_pins
            ],
            "use_scope": self.use_scope,
        }

    def canonical_sha256(self) -> str:
        """Hash the complete canonical support document."""
        return canonical_sha256(self.to_canonical_dict())


def _source_sha_pins() -> Tuple[SourceShaPin, ...]:
    """Bind the source identities already frozen by the two model modules."""
    surface_root = quality_surface.BUNDLE_RELATIVE_PATH
    return (
        SourceShaPin(
            "quality_surface_complete",
            f"{surface_root}/COMPLETE.json",
            quality_surface.COMPLETE_FILE_SHA256,
        ),
        SourceShaPin(
            "quality_surface_run_manifest_file",
            f"{surface_root}/run_manifest.json",
            quality_surface.RUN_MANIFEST_FILE_SHA256,
        ),
        SourceShaPin(
            "quality_surface_run_manifest_identity",
            f"{surface_root}/run_manifest.json#run_manifest_sha256",
            quality_surface.RUN_MANIFEST_SHA256,
        ),
        SourceShaPin(
            "quality_surface_run_binding",
            f"{surface_root}/run_manifest.json#run_binding_sha256",
            quality_surface.RUN_BINDING_SHA256,
        ),
        SourceShaPin(
            "quality_surface_selection_file",
            f"{surface_root}/selection_manifest.json",
            quality_surface.SELECTION_FILE_SHA256,
        ),
        SourceShaPin(
            "quality_surface_selection_identity",
            f"{surface_root}/selection_manifest.json#selection_manifest_sha256",
            quality_surface.SELECTION_MANIFEST_SHA256,
        ),
        SourceShaPin(
            "quality_surface_reward_spec",
            f"{surface_root}/reward_spec.json",
            quality_surface.REWARD_SPEC_FILE_SHA256,
        ),
        SourceShaPin(
            "quality_surface_database",
            f"{surface_root}/quality_rows.sqlite3",
            quality_surface.DATABASE_FILE_SHA256,
        ),
        SourceShaPin(
            "network_anchor_action_summary",
            "experiments/splitfusion_288_offline_rl_dataset_v1/"
            "20260909_offline_consolidation_v1/action_72_summary.csv",
            network_surrogate.ACTION_SUMMARY_SHA256,
        ),
        SourceShaPin(
            "network_anchor_profile_latency",
            "experiments/splitfusion_supervisor_analysis_v1/"
            "20260915_tail_completion_feedback_policy_analysis_v3/"
            "action_profile_quality_latency.csv",
            network_surrogate.PROFILE_LATENCY_SHA256,
        ),
        SourceShaPin(
            "action_catalog",
            "rl_agent/splitfusion_action_catalog_v1/"
            "splitfusion_72_action_catalog.json",
            network_surrogate.CATALOG_SHA256,
        ),
        SourceShaPin(
            "network_analysis_builder",
            network_surrogate.SOURCE_ANALYSIS_BUILDER_RELATIVE_PATH,
            network_surrogate.SOURCE_ANALYSIS_BUILDER_SHA256,
        ),
        SourceShaPin(
            "network_analysis_summary",
            network_surrogate.SOURCE_ANALYSIS_SUMMARY_RELATIVE_PATH,
            network_surrogate.SOURCE_ANALYSIS_SUMMARY_SHA256,
        ),
        SourceShaPin(
            "network_analysis_manifest",
            network_surrogate.SOURCE_ANALYSIS_MANIFEST_RELATIVE_PATH,
            network_surrogate.SOURCE_ANALYSIS_MANIFEST_SHA256,
        ),
        SourceShaPin(
            "production_transport",
            network_surrogate.PRODUCTION_TRANSPORT_RELATIVE_PATH,
            network_surrogate.PRODUCTION_TRANSPORT_SHA256,
        ),
        SourceShaPin(
            "production_runtime_contract",
            network_surrogate.PRODUCTION_RUNTIME_CONTRACT_RELATIVE_PATH,
            network_surrogate.PRODUCTION_RUNTIME_CONTRACT_SHA256,
        ),
    )


MODELED_SMOKE_SUPPORT = ModeledSmokeSupportContract(
    schema=MODELED_SMOKE_SUPPORT_SCHEMA,
    evidence_class="MODELED_SMOKE_SUPPORT",
    use_scope="CONTEXTUAL_SMOKE_CURRICULUM_ONLY",
    deployment_action_contract_status="NEVER_A_DEPLOYMENT_ACTION_CONTRACT",
    derivation_split="FIT_ONLY_HELD_EXCLUDED_FROM_BOUND_SELECTION",
    derivation_method=(
        "FIT_ONLY_512_FRAME_EXHAUSTIVE_Q_E4_INTERSECTION_WITH_"
        "ALL_PROFILE_COMMON_TOTAL_TRANSMITTED_BYTES_SUPPORT"
    ),
    fit_frame_count=FIT_FRAME_COUNT,
    payload_coordinate="total_transmitted_bytes",
    all_profile_payload_support=ALL_PROFILE_TOTAL_TRANSMITTED_BYTES_SUPPORT,
    mode_q_e4_bounds=MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    bounds_are_inclusive=True,
    held_audit_frame_count=256,
    held_audit_supported_count=HELD_AUDIT_SUPPORTED_COUNT,
    held_audit_total_count=HELD_AUDIT_TOTAL_COUNT,
    held_audit_refused_count=HELD_AUDIT_REFUSED_COUNT,
    source_sha_pins=_source_sha_pins(),
)

# Static pin over every semantic field above.  Keeping this literal rather
# than deriving an "expected" hash at import makes any accidental edit fail
# closed when a consumer opts into the curriculum.
MODELED_SMOKE_SUPPORT_SHA256 = "dc545c8703c9763165240bec5ad3f341979d1dd3b64e5bd5615222adeb24f566"


def require_registered_modeled_smoke_support(
    contract: object,
) -> ModeledSmokeSupportContract:
    """Return only the exact registered support contract; reject all drift."""
    if type(contract) is not ModeledSmokeSupportContract:
        raise ModeledSmokeSupportError(
            "modeled-smoke curriculum requires the exact registered contract type"
        )
    string_fields = (
        "schema",
        "evidence_class",
        "use_scope",
        "deployment_action_contract_status",
        "derivation_split",
        "derivation_method",
        "payload_coordinate",
    )
    integer_fields = (
        "fit_frame_count",
        "held_audit_frame_count",
        "held_audit_supported_count",
        "held_audit_total_count",
        "held_audit_refused_count",
    )
    if any(type(getattr(contract, name)) is not str for name in string_fields):
        raise ModeledSmokeSupportError(
            "modeled-smoke support string field has a foreign type"
        )
    if any(type(getattr(contract, name)) is not int for name in integer_fields):
        raise ModeledSmokeSupportError(
            "modeled-smoke support integer field has a foreign type"
        )
    if type(contract.bounds_are_inclusive) is not bool:
        raise ModeledSmokeSupportError(
            "modeled-smoke bounds_are_inclusive must be an exact bool"
        )
    if (
        type(contract.all_profile_payload_support) is not tuple
        or len(contract.all_profile_payload_support) != 2
        or any(
            type(value) is not int
            for value in contract.all_profile_payload_support
        )
    ):
        raise ModeledSmokeSupportError(
            "all-profile payload support must be an exact integer pair"
        )
    if (
        type(contract.mode_q_e4_bounds) is not tuple
        or len(contract.mode_q_e4_bounds) != 12
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
            for pair in contract.mode_q_e4_bounds
        )
    ):
        raise ModeledSmokeSupportError(
            "mode_q_e4_bounds must be 12 exact integer pairs"
        )
    if (
        type(contract.source_sha_pins) is not tuple
        or any(type(pin) is not SourceShaPin for pin in contract.source_sha_pins)
        or any(
            type(pin.component) is not str
            or type(pin.relative_path) is not str
            or type(pin.sha256) is not str
            or len(pin.sha256) != 64
            or any(character not in "0123456789abcdef" for character in pin.sha256)
            for pin in contract.source_sha_pins
        )
    ):
        raise ModeledSmokeSupportError(
            "source SHA pins are malformed or have foreign types"
        )
    try:
        document = contract.to_canonical_dict()
        observed_sha256 = contract.canonical_sha256()
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModeledSmokeSupportError(
            "modeled-smoke support contract is malformed"
        ) from exc
    if document != MODELED_SMOKE_SUPPORT.to_canonical_dict():
        raise ModeledSmokeSupportError(
            "foreign or modified modeled-smoke support contract refused"
        )
    if observed_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
        raise ModeledSmokeSupportError(
            "modeled-smoke support canonical SHA-256 differs from its static pin"
        )
    if (
        contract.held_audit_supported_count
        + contract.held_audit_refused_count
        != contract.held_audit_total_count
    ):
        raise ModeledSmokeSupportError(
            "held audit supported/refused counts do not sum to the total"
        )
    return contract
