"""Pure causal production-state assembly for Run-4.

This module is the narrow join between three already-separated sources:

* one exact fit-scene draw (camera SI and radar P40 only);
* the current sequential-kernel :class:`RadioQueueStateV1`; and
* the exact previous outcome requested by the persistent environment.

The sequential kernel intentionally retains only radio values and provenance
digests.  It does *not* retain the UL-DCI NDI/grant identity or measurement
timestamps required by :mod:`run4_contract`.  Those facts are therefore
supplied in an explicit, hash-matched provenance envelope.  They are never
reconstructed from a digest or filled with a plausible default.

Freshness and empirical scaling stay external verifier inputs.  Production
authorization is fail-closed until a reviewed prerequisite digest is pinned in
``REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256``.  The test-only factory is
useful for contract tests, but its authorization is structurally ineligible
for replay.

Importing this module performs no I/O and launches no runtime component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from rl_agent.splitfusion_hybrid_sac_run4_v1 import environment
from rl_agent.splitfusion_hybrid_sac_run4_v1 import fit_scene_provider
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import sequential_kernel
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

__all__ = [
    "ProductionStateProviderError",
    "StateProviderEvidenceError",
    "StateProviderAuthorizationError",
    "StateAssemblyFallbackRequired",
    "StateSequenceError",
    "StateProviderAuthorizationClass",
    "ObservationTimingV1",
    "PriorGrantProvenanceV1",
    "BacklogProvenanceV1",
    "StagedDecisionInputsV1",
    "StateProviderPrerequisitesV1",
    "StateProviderAuthorizationV1",
    "StateProviderBindingV1",
    "verify_production_state_provider_prerequisites",
    "authorize_test_only_state_provider",
    "Run4ProductionStateProviderV1",
    "REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256",
]


SCHEMA_ID = "splitfusion_run4_production_state_provider_v1"
SCHEMA_VERSION = 1
PRODUCTION_VERIFICATION_STATUS = "VERIFIED_FOR_RUN4_PRODUCTION_STATE"
TEST_ONLY_VERIFICATION_STATUS = "TEST_ONLY_STRUCTURAL_FIXTURE"

# Set only in a reviewed change after the calibration verifier has attested the
# fit-scene source, radio provenance, freshness policy and empirical scaling.
REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256: Optional[str] = None


class ProductionStateProviderError(ValueError):
    """Base class for state-provider contract failures."""


class StateProviderEvidenceError(ProductionStateProviderError):
    """Evidence identities, timestamps or provenance contradict each other."""


class StateProviderAuthorizationError(ProductionStateProviderError):
    """The supplied verifier/test authorization is absent or ineligible."""


class StateAssemblyFallbackRequired(ProductionStateProviderError):
    """A missing or stale causal input requires the external fallback path."""


class StateSequenceError(ProductionStateProviderError):
    """A staged decision does not continue the provider's exact sequence."""


def _text(value: object, name: str) -> str:
    if type(value) is not str or value == "" or value.strip() != value:
        raise StateProviderEvidenceError(f"{name} must be a non-empty canonical str")
    return value


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise StateProviderEvidenceError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StateProviderEvidenceError(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


def _timing_order(source_ns: int, available_ns: int) -> None:
    if source_ns > available_ns:
        raise StateProviderEvidenceError(
            "source_timestamp_ns must be <= available_timestamp_ns"
        )


class StateProviderAuthorizationClass(str, Enum):
    VERIFIED_EMPIRICAL = "VERIFIED_EMPIRICAL"
    TEST_ONLY = "TEST_ONLY"


@dataclass(frozen=True, slots=True)
class ObservationTimingV1:
    """One explicit source/availability record in the decision clock domain."""

    source: str
    source_timestamp_ns: int
    available_timestamp_ns: int
    clock_domain: str

    def __post_init__(self) -> None:
        _text(self.source, "source")
        source_ns = _exact_int(self.source_timestamp_ns, "source_timestamp_ns")
        available_ns = _exact_int(
            self.available_timestamp_ns, "available_timestamp_ns"
        )
        _timing_order(source_ns, available_ns)
        _text(self.clock_domain, "clock_domain")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_timestamp_ns": self.available_timestamp_ns,
            "clock_domain": self.clock_domain,
            "source": self.source,
            "source_timestamp_ns": self.source_timestamp_ns,
        }


@dataclass(frozen=True, slots=True)
class PriorGrantProvenanceV1:
    """Facts discarded from ``RadioQueueStateV1`` but required by policy state."""

    radio_observation_provenance_sha256: str
    grant_identity: str
    new_data_indicator: int
    harq_round: int
    mcs_table: int
    scheduler_policy_id: str
    timing: ObservationTimingV1

    def __post_init__(self) -> None:
        _digest(
            self.radio_observation_provenance_sha256,
            "radio_observation_provenance_sha256",
        )
        _text(self.grant_identity, "grant_identity")
        if type(self.new_data_indicator) is not int or self.new_data_indicator not in (
            0,
            1,
        ):
            raise StateProviderEvidenceError(
                "new_data_indicator must be exactly 0 or 1"
            )
        if type(self.harq_round) is not int or self.harq_round != 0:
            raise StateProviderEvidenceError("harq_round must be exactly 0")
        if type(self.mcs_table) is not int or self.mcs_table != contract.UL_MCS_TABLE_ID:
            raise StateProviderEvidenceError(
                f"mcs_table must be exactly {contract.UL_MCS_TABLE_ID}"
            )
        if self.scheduler_policy_id != contract.UL_MCS_POLICY_ID:
            raise StateProviderEvidenceError("scheduler_policy_id drifted")
        if type(self.timing) is not ObservationTimingV1:
            raise StateProviderEvidenceError(
                "timing must be exactly ObservationTimingV1"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grant_identity": self.grant_identity,
            "harq_round": self.harq_round,
            "mcs_table": self.mcs_table,
            "new_data_indicator": self.new_data_indicator,
            "radio_observation_provenance_sha256": (
                self.radio_observation_provenance_sha256
            ),
            "scheduler_policy_id": self.scheduler_policy_id,
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class BacklogProvenanceV1:
    """Exact causal timing/provenance for the pre-enqueue queue sample."""

    radio_observation_provenance_sha256: str
    timing: ObservationTimingV1

    def __post_init__(self) -> None:
        _digest(
            self.radio_observation_provenance_sha256,
            "radio_observation_provenance_sha256",
        )
        if type(self.timing) is not ObservationTimingV1:
            raise StateProviderEvidenceError(
                "timing must be exactly ObservationTimingV1"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "radio_observation_provenance_sha256": (
                self.radio_observation_provenance_sha256
            ),
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StagedDecisionInputsV1:
    """All hidden evidence needed to answer one environment state request."""

    identity: contract.DecisionIdentityV1
    boundary: contract.DecisionBoundaryV1
    scene_draw: fit_scene_provider.FitSceneDrawV1
    scene_timing: ObservationTimingV1
    radio_state: sequential_kernel.RadioQueueStateV1
    prior_grant: PriorGrantProvenanceV1
    backlog: BacklogProvenanceV1
    expected_previous_sha256: Optional[str]

    def __post_init__(self) -> None:
        if type(self.identity) is not contract.DecisionIdentityV1:
            raise StateProviderEvidenceError(
                "identity must be exactly DecisionIdentityV1"
            )
        if type(self.boundary) is not contract.DecisionBoundaryV1:
            raise StateProviderEvidenceError(
                "boundary must be exactly DecisionBoundaryV1"
            )
        if self.boundary.identity != self.identity:
            raise StateProviderEvidenceError("boundary/decision identity mismatch")
        if type(self.scene_draw) is not fit_scene_provider.FitSceneDrawV1:
            raise StateProviderEvidenceError(
                "scene_draw must be exactly FitSceneDrawV1"
            )
        if type(self.scene_timing) is not ObservationTimingV1:
            raise StateProviderEvidenceError(
                "scene_timing must be exactly ObservationTimingV1"
            )
        if type(self.radio_state) is not sequential_kernel.RadioQueueStateV1:
            raise StateProviderEvidenceError(
                "radio_state must be exactly RadioQueueStateV1"
            )
        if type(self.prior_grant) is not PriorGrantProvenanceV1:
            raise StateProviderEvidenceError(
                "prior_grant must be exactly PriorGrantProvenanceV1"
            )
        if type(self.backlog) is not BacklogProvenanceV1:
            raise StateProviderEvidenceError(
                "backlog must be exactly BacklogProvenanceV1"
            )
        observed = (
            self.radio_state.session_uuid,
            self.radio_state.ue_id,
            self.radio_state.decision_seq,
        )
        expected = (
            self.identity.session_uuid,
            self.identity.ue_id,
            self.identity.decision_seq,
        )
        if observed != expected:
            raise StateProviderEvidenceError(
                "radio state does not belong to the exact decision identity"
            )
        if self.expected_previous_sha256 is None:
            if self.identity.decision_seq != 0:
                raise StateProviderEvidenceError(
                    "every non-genesis decision requires the exact previous digest"
                )
        else:
            _digest(self.expected_previous_sha256, "expected_previous_sha256")
            if self.identity.decision_seq == 0:
                raise StateProviderEvidenceError(
                    "genesis cannot carry a previous-outcome digest"
                )
        if self.prior_grant.radio_observation_provenance_sha256 != (
            self.radio_state.prior_ul_mcs.provenance_sha256
        ):
            raise StateProviderEvidenceError(
                "prior-grant envelope does not bind the current radio state"
            )
        if self.backlog.radio_observation_provenance_sha256 != (
            self.radio_state.pre_enqueue_backlog_bytes.provenance_sha256
        ):
            raise StateProviderEvidenceError(
                "backlog envelope does not bind the current radio state"
            )
        for name, timing in (
            ("scene", self.scene_timing),
            ("prior_grant", self.prior_grant.timing),
            ("backlog", self.backlog.timing),
        ):
            if timing.clock_domain != self.boundary.clock_domain:
                raise StateProviderEvidenceError(
                    f"{name} timing uses a different clock domain"
                )

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_staged_decision_inputs_v1",
                "value": {
                    "backlog": self.backlog.to_dict(),
                    "boundary_sha256": self.boundary.canonical_sha256(),
                    "expected_previous_sha256": self.expected_previous_sha256,
                    "identity_sha256": self.identity.canonical_sha256(),
                    "prior_grant": self.prior_grant.to_dict(),
                    "radio_state_sha256": self.radio_state.canonical_sha256,
                    "scene_draw_sha256": self.scene_draw.canonical_sha256,
                    "scene_timing": self.scene_timing.to_dict(),
                },
            }
        )


@dataclass(frozen=True, slots=True)
class StateProviderPrerequisitesV1:
    """Future verifier input binding scaling, freshness and evidence sources."""

    fit_scene_provider_binding_sha256: str
    radio_queue_evidence_sha256: str
    calibration_evidence_sha256: str
    verifier_report_sha256: str
    scaling: contract.EmpiricalScalingV2
    freshness: contract.FreshnessPolicyV2
    verification_status: str

    def __post_init__(self) -> None:
        for name in (
            "fit_scene_provider_binding_sha256",
            "radio_queue_evidence_sha256",
            "calibration_evidence_sha256",
            "verifier_report_sha256",
        ):
            _digest(getattr(self, name), name)
        if type(self.scaling) is not contract.EmpiricalScalingV2:
            raise StateProviderEvidenceError(
                "scaling must be exactly EmpiricalScalingV2"
            )
        if type(self.freshness) is not contract.FreshnessPolicyV2:
            raise StateProviderEvidenceError(
                "freshness must be exactly FreshnessPolicyV2"
            )
        # Scaling, freshness and radio/queue calibration can legitimately come
        # from separate verifier inputs.  Their own evidence digests are kept
        # inside the canonical scaling/freshness records; the umbrella
        # calibration and verifier-report digests close the combined join.
        if self.verification_status not in (
            PRODUCTION_VERIFICATION_STATUS,
            TEST_ONLY_VERIFICATION_STATUS,
        ):
            raise StateProviderEvidenceError("unrecognized verification_status")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_evidence_sha256": self.calibration_evidence_sha256,
            "fit_scene_provider_binding_sha256": (
                self.fit_scene_provider_binding_sha256
            ),
            "freshness_sha256": self.freshness.canonical_sha256(),
            "radio_queue_evidence_sha256": self.radio_queue_evidence_sha256,
            "scaling_sha256": self.scaling.canonical_sha256(),
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "verification_status": self.verification_status,
            "verifier_report_sha256": self.verifier_report_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_state_provider_prerequisites_v1",
                "value": self.to_dict(),
            }
        )


class _AuthorizationAttestation:
    __slots__ = ("binding", "nonce")

    def __init__(self, binding: str, nonce: object) -> None:
        self.binding = binding
        self.nonce = nonce


_AUTHORIZATION_NONCE = object()


@dataclass(frozen=True, slots=True)
class StateProviderAuthorizationV1:
    authorization_class: StateProviderAuthorizationClass
    prerequisites_sha256: str
    verifier_report_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.authorization_class, StateProviderAuthorizationClass
        ):
            raise StateProviderAuthorizationError("invalid authorization class")
        _digest(self.prerequisites_sha256, "prerequisites_sha256")
        _digest(self.verifier_report_sha256, "verifier_report_sha256")

    def _binding(self) -> str:
        return canonical_sha256(
            {
                "authorization_class": self.authorization_class.value,
                "prerequisites_sha256": self.prerequisites_sha256,
                "record": "splitfusion_run4_state_provider_authorization_v1",
                "verifier_report_sha256": self.verifier_report_sha256,
            }
        )

    @property
    def is_attested(self) -> bool:
        return (
            type(self._attestation) is _AuthorizationAttestation
            and self._attestation.nonce is _AUTHORIZATION_NONCE
            and self._attestation.binding == self._binding()
        )

    @property
    def replay_eligible(self) -> bool:
        return (
            self.is_attested
            and self.authorization_class
            is StateProviderAuthorizationClass.VERIFIED_EMPIRICAL
        )

    def require_attested(self) -> None:
        if not self.is_attested:
            raise StateProviderAuthorizationError(
                "state-provider authorization is not verifier-attested"
            )

    def require_replay_eligible(self) -> None:
        self.require_attested()
        if not self.replay_eligible:
            raise StateProviderAuthorizationError(
                "test-only state assembly can never authorize replay"
            )


def _issue_authorization(
    prerequisites: StateProviderPrerequisitesV1,
    authorization_class: StateProviderAuthorizationClass,
) -> StateProviderAuthorizationV1:
    candidate = StateProviderAuthorizationV1(
        authorization_class=authorization_class,
        prerequisites_sha256=prerequisites.canonical_sha256,
        verifier_report_sha256=prerequisites.verifier_report_sha256,
    )
    object.__setattr__(
        candidate,
        "_attestation",
        _AuthorizationAttestation(candidate._binding(), _AUTHORIZATION_NONCE),
    )
    return candidate


def verify_production_state_provider_prerequisites(
    prerequisites: StateProviderPrerequisitesV1,
) -> StateProviderAuthorizationV1:
    """Issue production authority only for the future reviewed digest."""

    if type(prerequisites) is not StateProviderPrerequisitesV1:
        raise StateProviderAuthorizationError(
            "prerequisites must be exactly StateProviderPrerequisitesV1"
        )
    if prerequisites.verification_status != PRODUCTION_VERIFICATION_STATUS:
        raise StateProviderAuthorizationError(
            "production verification status is not accepted"
        )
    if REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256 is None:
        raise StateProviderAuthorizationError(
            "no reviewed Run-4 state-provider prerequisites are registered"
        )
    if prerequisites.canonical_sha256 != (
        REGISTERED_STATE_PROVIDER_PREREQUISITES_SHA256
    ):
        raise StateProviderAuthorizationError(
            "state-provider prerequisites do not match the registered digest"
        )
    return _issue_authorization(
        prerequisites, StateProviderAuthorizationClass.VERIFIED_EMPIRICAL
    )


def authorize_test_only_state_provider(
    prerequisites: StateProviderPrerequisitesV1,
) -> StateProviderAuthorizationV1:
    """Issue a structural-test token which is categorically replay-ineligible."""

    if type(prerequisites) is not StateProviderPrerequisitesV1:
        raise StateProviderAuthorizationError(
            "prerequisites must be exactly StateProviderPrerequisitesV1"
        )
    if prerequisites.verification_status != TEST_ONLY_VERIFICATION_STATUS:
        raise StateProviderAuthorizationError(
            "test authorization requires TEST_ONLY_STRUCTURAL_FIXTURE"
        )
    return _issue_authorization(
        prerequisites, StateProviderAuthorizationClass.TEST_ONLY
    )


@dataclass(frozen=True, slots=True)
class StateProviderBindingV1:
    prerequisites_sha256: str
    authorization_class: StateProviderAuthorizationClass
    fit_scene_provider_binding_sha256: str
    scaling_sha256: str
    freshness_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "prerequisites_sha256",
            "fit_scene_provider_binding_sha256",
            "scaling_sha256",
            "freshness_sha256",
        ):
            _digest(getattr(self, name), name)
        if not isinstance(
            self.authorization_class, StateProviderAuthorizationClass
        ):
            raise StateProviderAuthorizationError("invalid authorization class")

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record": "splitfusion_run4_state_provider_binding_v1",
                "value": {
                    "authorization_class": self.authorization_class.value,
                    "fit_scene_provider_binding_sha256": (
                        self.fit_scene_provider_binding_sha256
                    ),
                    "freshness_sha256": self.freshness_sha256,
                    "prerequisites_sha256": self.prerequisites_sha256,
                    "scaling_sha256": self.scaling_sha256,
                },
            }
        )


class Run4ProductionStateProviderV1:
    """One-shot staged provider implementing ``environment.StateProvider``.

    A provider instance owns exactly one monotonically increasing decision
    sequence.  A new episode/session therefore requires a new provider.  This
    prevents a non-genesis continuation from being silently reset to sequence
    zero.
    """

    def __init__(
        self,
        *,
        prerequisites: StateProviderPrerequisitesV1,
        authorization: StateProviderAuthorizationV1,
    ) -> None:
        if type(prerequisites) is not StateProviderPrerequisitesV1:
            raise StateProviderAuthorizationError(
                "prerequisites must be exactly StateProviderPrerequisitesV1"
            )
        if type(authorization) is not StateProviderAuthorizationV1:
            raise StateProviderAuthorizationError(
                "authorization must be exactly StateProviderAuthorizationV1"
            )
        authorization.require_attested()
        if authorization.prerequisites_sha256 != prerequisites.canonical_sha256:
            raise StateProviderAuthorizationError(
                "authorization belongs to different prerequisites"
            )
        if authorization.verifier_report_sha256 != (
            prerequisites.verifier_report_sha256
        ):
            raise StateProviderAuthorizationError(
                "authorization/verifier report mismatch"
            )
        if (
            authorization.authorization_class
            is StateProviderAuthorizationClass.VERIFIED_EMPIRICAL
            and prerequisites.verification_status
            != PRODUCTION_VERIFICATION_STATUS
        ):
            raise StateProviderAuthorizationError(
                "production authorization carries test prerequisites"
            )
        if (
            authorization.authorization_class
            is StateProviderAuthorizationClass.TEST_ONLY
            and prerequisites.verification_status != TEST_ONLY_VERIFICATION_STATUS
        ):
            raise StateProviderAuthorizationError(
                "test authorization carries production prerequisites"
            )
        self._prerequisites = prerequisites
        self._authorization = authorization
        self.binding = StateProviderBindingV1(
            prerequisites_sha256=prerequisites.canonical_sha256,
            authorization_class=authorization.authorization_class,
            fit_scene_provider_binding_sha256=(
                prerequisites.fit_scene_provider_binding_sha256
            ),
            scaling_sha256=prerequisites.scaling.canonical_sha256(),
            freshness_sha256=prerequisites.freshness.canonical_sha256(),
        )
        self._pending: Optional[StagedDecisionInputsV1] = None
        self._last_identity: Optional[contract.DecisionIdentityV1] = None

    @property
    def replay_export_allowed(self) -> bool:
        return self._authorization.replay_eligible

    def require_replay_eligible(self) -> None:
        self._authorization.require_replay_eligible()

    def _require_next_identity(self, identity: contract.DecisionIdentityV1) -> None:
        previous = self._last_identity
        if previous is None:
            if identity.decision_seq != 0:
                raise StateSequenceError(
                    "a fresh provider must start at genesis decision_seq 0"
                )
            return
        if (
            identity.session_uuid != previous.session_uuid
            or identity.ue_id != previous.ue_id
            or identity.decision_seq != previous.decision_seq + 1
        ):
            raise StateSequenceError(
                "provider cannot reset or skip a non-genesis decision sequence"
            )

    def stage_decision(self, staged: StagedDecisionInputsV1) -> None:
        """Stage one exact set of causal inputs for the environment request."""

        if type(staged) is not StagedDecisionInputsV1:
            raise StateProviderEvidenceError(
                "staged must be exactly StagedDecisionInputsV1"
            )
        if self._pending is not None:
            raise StateSequenceError(
                "an unconsumed staged decision cannot be overwritten"
            )
        self._require_next_identity(staged.identity)
        if staged.scene_draw.provider_binding_sha256 != (
            self.binding.fit_scene_provider_binding_sha256
        ):
            raise StateProviderEvidenceError(
                "fit-scene draw/provider binding mismatch"
            )
        if not staged.radio_state.actor_ready:
            raise StateAssemblyFallbackRequired(
                "missing radio state requires external fallback; zero-fill forbidden"
            )
        # Radio values are kept integer all the way into the strict contract.
        mcs = staged.radio_state.prior_ul_mcs.value
        backlog = staged.radio_state.pre_enqueue_backlog_bytes.value
        if type(mcs) is not int or not (
            contract.UL_MCS_INDEX_MIN <= mcs <= contract.UL_MCS_INDEX_MAX
        ):
            raise StateAssemblyFallbackRequired("prior UL MCS is unavailable/invalid")
        if type(backlog) is not int or backlog < 0:
            raise StateAssemblyFallbackRequired(
                "pre-enqueue backlog is unavailable/invalid"
            )
        self._pending = staged

    @staticmethod
    def _metadata(
        *,
        identity: contract.DecisionIdentityV1,
        sample_seq: int,
        kind: contract.MeasurementKind,
        observer: contract.Observer,
        direction: contract.LinkDirection,
        timing: ObservationTimingV1,
    ) -> contract.MeasurementMetadataV1:
        return contract.MeasurementMetadataV1(
            identity=contract.SampleIdentityV1(
                identity.session_uuid, identity.ue_id, sample_seq
            ),
            kind=kind,
            observer=observer,
            link_direction=direction,
            source=timing.source,
            source_timestamp_ns=timing.source_timestamp_ns,
            available_timestamp_ns=timing.available_timestamp_ns,
            clock_domain=timing.clock_domain,
            valid=True,
        )

    def _validate_request(
        self,
        staged: StagedDecisionInputsV1,
        request: environment.DecisionStateRequestV1,
    ) -> None:
        if type(request) is not environment.DecisionStateRequestV1:
            raise StateSequenceError(
                "request must be exactly DecisionStateRequestV1"
            )
        if request.identity != staged.identity:
            raise StateSequenceError("request/staged decision identity mismatch")
        previous_sha = (
            None
            if request.previous is None
            else request.previous.canonical_sha256()
        )
        if previous_sha != staged.expected_previous_sha256:
            raise StateSequenceError(
                "request does not carry the exact staged previous outcome"
            )
        required_open = request.required_action_open_timestamp_ns
        if required_open is not None and required_open != (
            staged.boundary.action_open_timestamp_ns
        ):
            raise StateSequenceError(
                "staged action-open timestamp differs from environment request"
            )
        if staged.boundary.state_commit_timestamp_ns < (
            request.minimum_state_commit_timestamp_ns
        ):
            raise StateSequenceError(
                "state commit predates the required completed outcome"
            )

    def build_state(
        self, request: environment.DecisionStateRequestV1
    ) -> environment.DecisionStateBundleV1:
        """Assemble, guard and scale one exact staged decision atomically."""

        staged = self._pending
        if staged is None:
            raise StateSequenceError("no causal decision inputs are staged")
        self._validate_request(staged, request)
        identity = staged.identity
        radio = staged.radio_state
        scene = staged.scene_draw.policy_scene()

        scene_sample_seq = identity.decision_seq
        camera_metadata = self._metadata(
            identity=identity,
            sample_seq=scene_sample_seq,
            kind=contract.MeasurementKind.CAMERA_SI,
            observer=contract.Observer.SCENE_PIPELINE,
            direction=contract.LinkDirection.NOT_APPLICABLE,
            timing=staged.scene_timing,
        )
        radar_metadata = self._metadata(
            identity=identity,
            sample_seq=scene_sample_seq,
            kind=contract.MeasurementKind.RADAR_P40,
            observer=contract.Observer.SCENE_PIPELINE,
            direction=contract.LinkDirection.NOT_APPLICABLE,
            timing=staged.scene_timing,
        )
        mcs_metadata = self._metadata(
            identity=identity,
            sample_seq=radio.prior_ul_mcs.source_decision_seq,
            kind=contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
            observer=contract.Observer.UE,
            direction=contract.LinkDirection.UPLINK,
            timing=staged.prior_grant.timing,
        )
        backlog_metadata = self._metadata(
            identity=identity,
            sample_seq=radio.pre_enqueue_backlog_bytes.source_decision_seq,
            kind=contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES,
            observer=contract.Observer.UE,
            direction=contract.LinkDirection.UPLINK,
            timing=staged.backlog.timing,
        )
        state = contract.PolicyStateV2(
            identity=identity,
            camera_si=contract.ScalarObservationV1(
                value=scene.camera_si,
                metadata=camera_metadata,
                missing_reason=None,
            ),
            radar_p40=contract.ScalarObservationV1(
                value=scene.radar_p40,
                metadata=radar_metadata,
                missing_reason=None,
            ),
            prior_ul_mcs=contract.PriorUlGrantObservationV1(
                observation=contract.ScalarObservationV1(
                    value=radio.prior_ul_mcs.value,
                    metadata=mcs_metadata,
                    missing_reason=None,
                ),
                mcs_table=staged.prior_grant.mcs_table,
                harq_round=staged.prior_grant.harq_round,
                new_data_indicator=staged.prior_grant.new_data_indicator,
                grant_identity=staged.prior_grant.grant_identity,
                scheduler_policy_id=staged.prior_grant.scheduler_policy_id,
                selection_rule_id=contract.UL_MCS_SELECTION_RULE_ID,
            ),
            pre_action_rlc_backlog=contract.ScalarObservationV1(
                value=radio.pre_enqueue_backlog_bytes.value,
                metadata=backlog_metadata,
                missing_reason=None,
            ),
            previous=request.previous,
        )
        try:
            guarded = contract.guard_state_for_action(
                state, staged.boundary, self._prerequisites.freshness
            )
            features = contract.build_policy_features(
                guarded, self._prerequisites.scaling
            )
            bundle = environment.DecisionStateBundleV1(guarded, features)
        except contract.ExternalFallbackRequired as exc:
            raise StateAssemblyFallbackRequired(str(exc)) from exc

        # Commit only after every contract guard and feature attestation passes.
        self._last_identity = identity
        self._pending = None
        return bundle
