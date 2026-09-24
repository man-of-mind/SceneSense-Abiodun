"""Calibration-independent sequential mechanics for Run-4.

This module deliberately implements only the causal *shape* of one Run-4
decision cycle.  State acquisition and the network/service kernel are injected
interfaces.  Consequently this file contains no fitted queue parameters, no
synthetic radio model presented as empirical evidence, no I/O, and no runtime
launches.

The current implementation is intentionally fail-closed in two important
ways:

* a state provider must return a contract-attested state and feature vector;
  missing/stale observations propagate the contract's external-fallback path
  and are never zero-filled here; and
* until a real calibration binding is registered, the environment can run
  only ``SYNTHETIC_MECHANICS_FIXTURE`` cycles.  Such a cycle is validated with
  :func:`run4_contract.build_transition`, but the transition object is
  immediately discarded.  The returned object contains no bare
  ``SemiMarkovTransitionV2`` and its export method always raises.

That second boundary is structural, not a warning label: a mechanics fixture
cannot be handed directly to the production replay buffer.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Union

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)

__all__ = [
    "EnvironmentError",
    "EnvironmentStateError",
    "KernelResultError",
    "SuccessorUnavailableError",
    "SyntheticEvidenceRejected",
    "CalibrationUnavailableError",
    "EnvironmentEvidenceClass",
    "REGISTERED_CALIBRATION_BINDING_SHA256",
    "CalibrationBindingV1",
    "DecisionStateRequestV1",
    "DecisionStateBundleV1",
    "KernelCycleRequestV1",
    "KernelCycleResultV1",
    "StateProvider",
    "CycleKernel",
    "SyntheticMechanicsCycleV1",
    "ExcludedCycleV1",
    "Run4SequentialEnvironmentV1",
]


class EnvironmentError(RuntimeError):
    """Base class for sequential-environment failures."""


class EnvironmentStateError(EnvironmentError):
    """The environment lifecycle or a provider state is inconsistent."""


class KernelResultError(EnvironmentError):
    """The injected kernel did not describe this exact decision cycle."""


class SuccessorUnavailableError(EnvironmentError):
    """A completed non-boundary cycle has no real causal successor."""


class SyntheticEvidenceRejected(EnvironmentError):
    """Synthetic mechanics evidence cannot be exported to replay."""


class CalibrationUnavailableError(EnvironmentError):
    """No accepted empirical calibration is registered for this shell."""


class EnvironmentEvidenceClass(str, Enum):
    SYNTHETIC_MECHANICS_FIXTURE = "SYNTHETIC_MECHANICS_FIXTURE"
    CALIBRATED_EMPIRICAL = "CALIBRATED_EMPIRICAL"


# TODO(Run-4 calibration): set this only in a reviewed change that pins the
# accepted 12-cell calibration artifact and its verifier output.  Leaving it
# unset makes it impossible to relabel a test kernel as calibrated merely by
# supplying a plausible-looking SHA-256 string.
REGISTERED_CALIBRATION_BINDING_SHA256: Optional[str] = None


def _non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise EnvironmentStateError(f"{name} must be a non-empty str")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise EnvironmentStateError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _exact_non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise EnvironmentStateError(f"{name} must be an exact int >= 0")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvironmentStateError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise EnvironmentStateError(f"{name} must be finite and > 0")
    return result


def _canonical_uuid(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise EnvironmentStateError(f"{name} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise EnvironmentStateError(f"{name} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise EnvironmentStateError(f"{name} must be canonical lowercase UUID")
    return value


@dataclass(frozen=True, slots=True)
class CalibrationBindingV1:
    """Externally supplied description of one qualified empirical kernel.

    Constructing this record is not authorization.  The environment also
    requires its canonical digest to equal
    :data:`REGISTERED_CALIBRATION_BINDING_SHA256`, which is intentionally
    ``None`` until the real calibration is reviewed and pinned.
    """

    calibration_id: str
    calibration_version: int
    evidence_sha256: str
    verifier_report_sha256: str
    kernel_binding_sha256: str
    state_provider_binding_sha256: str

    def __post_init__(self) -> None:
        _non_empty_str(self.calibration_id, "calibration_id")
        if type(self.calibration_version) is not int or self.calibration_version <= 0:
            raise EnvironmentStateError("calibration_version must be an int > 0")
        for name in (
            "evidence_sha256",
            "verifier_report_sha256",
            "kernel_binding_sha256",
            "state_provider_binding_sha256",
        ):
            _sha256(getattr(self, name), name)

    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "calibration_id": self.calibration_id,
                "calibration_version": self.calibration_version,
                "evidence_sha256": self.evidence_sha256,
                "kernel_binding_sha256": self.kernel_binding_sha256,
                "schema": "splitfusion_run4_calibration_binding_v1",
                "state_provider_binding_sha256": (
                    self.state_provider_binding_sha256
                ),
                "verifier_report_sha256": self.verifier_report_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class DecisionStateRequestV1:
    """Causal request passed to the injected state provider."""

    identity: contract.DecisionIdentityV1
    previous: Optional[contract.PreviousOutcomeV1]
    required_action_open_timestamp_ns: Optional[int]
    minimum_state_commit_timestamp_ns: int

    def __post_init__(self) -> None:
        if type(self.identity) is not contract.DecisionIdentityV1:
            raise EnvironmentStateError(
                "identity must be exactly DecisionIdentityV1"
            )
        if self.previous is not None and type(self.previous) is not (
            contract.PreviousOutcomeV1
        ):
            raise EnvironmentStateError(
                "previous must be exactly PreviousOutcomeV1 or None"
            )
        if self.required_action_open_timestamp_ns is not None:
            _exact_non_negative_int(
                self.required_action_open_timestamp_ns,
                "required_action_open_timestamp_ns",
            )
        _exact_non_negative_int(
            self.minimum_state_commit_timestamp_ns,
            "minimum_state_commit_timestamp_ns",
        )


@dataclass(frozen=True, slots=True)
class DecisionStateBundleV1:
    """One already guarded state plus its exact contract feature vector."""

    state: contract.GuardedPolicyStateV2
    features: contract.PolicyFeatureVectorV2

    def __post_init__(self) -> None:
        if type(self.state) is not contract.GuardedPolicyStateV2:
            raise EnvironmentStateError(
                "state must be exactly GuardedPolicyStateV2"
            )
        if type(self.features) is not contract.PolicyFeatureVectorV2:
            raise EnvironmentStateError(
                "features must be exactly PolicyFeatureVectorV2"
            )
        self.state.require_guarded()
        self.features.require_attested()
        if self.features.guarded_state_sha256 != self.state.canonical_sha256():
            raise EnvironmentStateError("features are not bound to this state")


@dataclass(frozen=True, slots=True)
class KernelCycleRequestV1:
    """Exact decision sent to the injected queue/service kernel."""

    state: DecisionStateBundleV1
    action: ExecutedActionIdentity

    def __post_init__(self) -> None:
        if type(self.state) is not DecisionStateBundleV1:
            raise KernelResultError("state must be exactly DecisionStateBundleV1")
        if type(self.action) is not ExecutedActionIdentity:
            raise KernelResultError(
                "action must be exactly ExecutedActionIdentity"
            )
        self.action.require_reconciled()


@dataclass(frozen=True, slots=True)
class KernelCycleResultV1:
    """Terminal result of one injected kernel cycle.

    A pending partial observation has no representation here.  The kernel must
    return only after feedback, a registered timeout/failure, or an excluded
    infrastructure/evaluator fault is known.
    """

    hold: contract.ActionHoldV1
    reward_event: contract.RewardEventV1
    cycle_end_timestamp_ns: int
    episode_boundary: contract.EpisodeBoundary

    def __post_init__(self) -> None:
        if type(self.hold) is not contract.ActionHoldV1:
            raise KernelResultError("hold must be exactly ActionHoldV1")
        if type(self.reward_event) is not contract.RewardEventV1:
            raise KernelResultError("reward_event must be exactly RewardEventV1")
        _exact_non_negative_int(
            self.cycle_end_timestamp_ns, "cycle_end_timestamp_ns"
        )
        if not isinstance(self.episode_boundary, contract.EpisodeBoundary):
            raise KernelResultError(
                "episode_boundary must be EpisodeBoundary"
            )


class StateProvider(Protocol):
    """Injected provider; it must never fabricate missing observations."""

    def build_state(self, request: DecisionStateRequestV1) -> DecisionStateBundleV1:
        ...


class CycleKernel(Protocol):
    """Injected terminal queue/service kernel."""

    def execute_cycle(self, request: KernelCycleRequestV1) -> KernelCycleResultV1:
        ...


@dataclass(frozen=True, slots=True)
class SyntheticMechanicsCycleV1:
    """Non-exportable diagnostics from one structurally validated cycle."""

    evidence_class: EnvironmentEvidenceClass
    identity: contract.DecisionIdentityV1
    terminal: contract.RewardTerminal
    reward: float
    duration: int
    action_sha256: str
    hold_sha256: str
    reward_resolution_sha256: str
    transition_sha256: str
    next_state_sha256: Optional[str]
    reward_request_flags: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.evidence_class is not (
            EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE
        ):
            raise SyntheticEvidenceRejected(
                "SyntheticMechanicsCycleV1 must remain labelled synthetic"
            )
        if type(self.identity) is not contract.DecisionIdentityV1:
            raise SyntheticEvidenceRejected(
                "identity must be exactly DecisionIdentityV1"
            )
        if not isinstance(self.terminal, contract.RewardTerminal):
            raise SyntheticEvidenceRejected("terminal must be RewardTerminal")
        if not math.isfinite(float(self.reward)):
            raise SyntheticEvidenceRejected("reward must be finite")
        if (
            type(self.duration) is not int
            or self.duration < contract.MINIMUM_HOLD_TENSORS
        ):
            raise SyntheticEvidenceRejected(
                "duration must satisfy the registered minimum action hold"
            )
        for name in (
            "action_sha256",
            "hold_sha256",
            "reward_resolution_sha256",
            "transition_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.next_state_sha256 is not None:
            _sha256(self.next_state_sha256, "next_state_sha256")
        if type(self.reward_request_flags) is not tuple or any(
            type(value) is not bool for value in self.reward_request_flags
        ):
            raise SyntheticEvidenceRejected(
                "reward_request_flags must be an exact tuple of bools"
            )

    @property
    def replay_export_allowed(self) -> bool:
        return False

    def export_for_replay(self) -> contract.SemiMarkovTransitionV2:
        raise SyntheticEvidenceRejected(
            "synthetic mechanics output has no replay-exportable transition"
        )


@dataclass(frozen=True, slots=True)
class ExcludedCycleV1:
    """Excluded infrastructure/evaluator fault; never a learning sample."""

    evidence_class: EnvironmentEvidenceClass
    identity: contract.DecisionIdentityV1
    terminal: contract.RewardTerminal
    reward_resolution_sha256: str

    @property
    def replay_export_allowed(self) -> bool:
        return False

    def export_for_replay(self) -> contract.SemiMarkovTransitionV2:
        raise SyntheticEvidenceRejected(
            "excluded infrastructure/evaluator fault has no transition"
        )


CycleResult = Union[SyntheticMechanicsCycleV1, ExcludedCycleV1]


class Run4SequentialEnvironmentV1:
    """Persistent decision-cycle orchestrator with injected causal sources."""

    def __init__(
        self,
        *,
        state_provider: StateProvider,
        kernel: CycleKernel,
        gamma: float,
        evidence_class: EnvironmentEvidenceClass,
        calibration_binding: Optional[CalibrationBindingV1] = None,
    ) -> None:
        if not callable(getattr(state_provider, "build_state", None)):
            raise EnvironmentStateError(
                "state_provider must provide build_state(request)"
            )
        if not callable(getattr(kernel, "execute_cycle", None)):
            raise EnvironmentStateError(
                "kernel must provide execute_cycle(request)"
            )
        self._state_provider = state_provider
        self._kernel = kernel
        self._gamma = _positive_finite(gamma, "gamma")
        if self._gamma > 1.0:
            raise EnvironmentStateError("gamma must lie in (0, 1]")
        if not isinstance(evidence_class, EnvironmentEvidenceClass):
            raise EnvironmentStateError(
                "evidence_class must be EnvironmentEvidenceClass"
            )
        self._evidence_class = evidence_class
        self._calibration_binding = calibration_binding

        if evidence_class is EnvironmentEvidenceClass.CALIBRATED_EMPIRICAL:
            if type(calibration_binding) is not CalibrationBindingV1:
                raise CalibrationUnavailableError(
                    "calibrated mode requires an explicit external "
                    "CalibrationBindingV1"
                )
            if REGISTERED_CALIBRATION_BINDING_SHA256 is None:
                raise CalibrationUnavailableError(
                    "no empirical Run-4 calibration is registered; calibrated "
                    "mode remains fail-closed"
                )
            if calibration_binding.canonical_sha256() != (
                REGISTERED_CALIBRATION_BINDING_SHA256
            ):
                raise CalibrationUnavailableError(
                    "external calibration binding does not match the registered "
                    "reviewed binding"
                )
        elif calibration_binding is not None:
            raise EnvironmentStateError(
                "synthetic mechanics mode must not carry a calibration binding"
            )

        self._current: Optional[DecisionStateBundleV1] = None
        self._requires_reset = False
        self._used_session_uuids: set[str] = set()

    @property
    def evidence_class(self) -> EnvironmentEvidenceClass:
        return self._evidence_class

    @property
    def requires_reset(self) -> bool:
        return self._requires_reset

    @property
    def current_state(self) -> DecisionStateBundleV1:
        if self._current is None or self._requires_reset:
            raise EnvironmentStateError("environment has no active decision state")
        return self._current

    def _validate_bundle(
        self,
        bundle: object,
        request: DecisionStateRequestV1,
    ) -> DecisionStateBundleV1:
        if type(bundle) is not DecisionStateBundleV1:
            raise EnvironmentStateError(
                "state provider must return exactly DecisionStateBundleV1"
            )
        actual_state = bundle.state.state
        actual_boundary = bundle.state.boundary
        if actual_state.identity != request.identity:
            raise EnvironmentStateError(
                "provider returned a different decision identity"
            )
        expected_previous = request.previous
        actual_previous = actual_state.previous
        if expected_previous is None:
            if actual_previous is not None:
                raise EnvironmentStateError(
                    "genesis state must not fabricate a previous outcome"
                )
        elif actual_previous is None or actual_previous.canonical_sha256() != (
            expected_previous.canonical_sha256()
        ):
            raise EnvironmentStateError(
                "successor previous outcome is not the exact completed outcome"
            )
        required_open = request.required_action_open_timestamp_ns
        if required_open is not None and (
            actual_boundary.action_open_timestamp_ns != required_open
        ):
            raise EnvironmentStateError(
                "successor action-open timestamp differs from the requested "
                "cycle end"
            )
        if actual_boundary.state_commit_timestamp_ns < (
            request.minimum_state_commit_timestamp_ns
        ):
            raise EnvironmentStateError(
                "state was committed before the completed outcome was available"
            )
        return bundle

    def reset(self, *, session_uuid: str, ue_id: str) -> DecisionStateBundleV1:
        """Acquire one guarded genesis under a never-before-used session UUID.

        An active sequence cannot be abandoned through ``reset``.  After an
        explicit boundary or fault, the next genesis must use a fresh session
        UUID so sequence zero cannot masquerade as a continuation or retry.
        """

        canonical_session = _canonical_uuid(session_uuid, "session_uuid")
        if self._current is not None and not self._requires_reset:
            raise EnvironmentStateError(
                "cannot reset an active decision sequence without an explicit "
                "boundary or fault"
            )
        if canonical_session in self._used_session_uuids:
            raise EnvironmentStateError(
                "reset requires a fresh session_uuid; a completed, faulted, or "
                "abandoned sequence must never restart at decision_seq 0"
            )
        identity = contract.DecisionIdentityV1(
            canonical_session,
            _non_empty_str(ue_id, "ue_id"),
            0,
        )
        request = DecisionStateRequestV1(
            identity=identity,
            previous=None,
            required_action_open_timestamp_ns=None,
            minimum_state_commit_timestamp_ns=0,
        )
        bundle = self._validate_bundle(
            self._state_provider.build_state(request), request
        )
        self._used_session_uuids.add(canonical_session)
        self._current = bundle
        self._requires_reset = False
        return bundle

    def step(self, action: ExecutedActionIdentity) -> CycleResult:
        """Execute one closed decision cycle transactionally.

        Once the injected kernel has executed, any malformed terminal evidence
        or unavailable successor makes the environment require ``reset()``;
        retrying the same decision identity would duplicate a physical action.
        """

        current = self.current_state
        if type(action) is not ExecutedActionIdentity:
            raise EnvironmentStateError(
                "action must be exactly ExecutedActionIdentity"
            )
        action.require_reconciled()
        request = KernelCycleRequestV1(current, action)
        kernel_executed = False
        try:
            raw = self._kernel.execute_cycle(request)
            kernel_executed = True
            if type(raw) is not KernelCycleResultV1:
                raise KernelResultError(
                    "kernel must return exactly KernelCycleResultV1"
                )
            self._validate_kernel_result(raw, request)
            resolution = contract.resolve_reward(raw.reward_event)

            if not resolution.learning_included:
                self._requires_reset = True
                return ExcludedCycleV1(
                    evidence_class=self._evidence_class,
                    identity=resolution.identity,
                    terminal=resolution.terminal,
                    reward_resolution_sha256=resolution.canonical_sha256(),
                )

            next_bundle: Optional[DecisionStateBundleV1]
            if raw.episode_boundary is contract.EpisodeBoundary.CONTINUES:
                previous = contract.PreviousOutcomeV1.from_resolution(resolution)
                next_request = DecisionStateRequestV1(
                    identity=contract.DecisionIdentityV1(
                        current.state.state.identity.session_uuid,
                        current.state.state.identity.ue_id,
                        current.state.state.identity.decision_seq + 1,
                    ),
                    previous=previous,
                    required_action_open_timestamp_ns=(
                        raw.cycle_end_timestamp_ns
                    ),
                    minimum_state_commit_timestamp_ns=(
                        resolution.resolution_timestamp_ns
                    ),
                )
                try:
                    next_bundle = self._validate_bundle(
                        self._state_provider.build_state(next_request),
                        next_request,
                    )
                except Exception as exc:
                    raise SuccessorUnavailableError(
                        "completed continuing cycle has no valid real successor"
                    ) from exc
            else:
                next_bundle = None

            elapsed = (
                raw.cycle_end_timestamp_ns
                - current.state.boundary.action_open_timestamp_ns
            )
            transition = contract.build_transition(
                state=current.state,
                state_features=current.features,
                action=action,
                hold=raw.hold,
                reward_resolution=resolution,
                next_state=(None if next_bundle is None else next_bundle.state),
                next_state_features=(
                    None if next_bundle is None else next_bundle.features
                ),
                episode_boundary=raw.episode_boundary,
                duration=raw.hold.duration,
                cycle_end_timestamp_ns=raw.cycle_end_timestamp_ns,
                elapsed_virtual_ns=elapsed,
                gamma=self._gamma,
                discount=self._gamma ** raw.hold.duration,
            )

            # Calibrated construction is currently refused in __init__.  Keep
            # this explicit assertion so a future registration change cannot
            # accidentally expose a bare transition without first defining a
            # replay verifier envelope.
            if self._evidence_class is not (
                EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE
            ):
                raise CalibrationUnavailableError(
                    "calibrated transition export is not implemented; wire it "
                    "through the production replay verifier before enabling"
                )

            result = SyntheticMechanicsCycleV1(
                evidence_class=self._evidence_class,
                identity=current.state.state.identity,
                terminal=resolution.terminal,
                reward=float(resolution.reward),
                duration=raw.hold.duration,
                action_sha256=action.canonical_sha256(),
                hold_sha256=raw.hold.canonical_sha256(),
                reward_resolution_sha256=resolution.canonical_sha256(),
                transition_sha256=transition.canonical_sha256(),
                next_state_sha256=(
                    None
                    if next_bundle is None
                    else next_bundle.state.canonical_sha256()
                ),
                reward_request_flags=tuple(
                    tensor.reward_requested for tensor in raw.hold.tensors
                ),
            )

            if next_bundle is None:
                self._current = None
                self._requires_reset = True
            else:
                self._current = next_bundle
            return result
        except Exception:
            if kernel_executed:
                self._requires_reset = True
            raise

    @staticmethod
    def _validate_kernel_result(
        result: KernelCycleResultV1,
        request: KernelCycleRequestV1,
    ) -> None:
        state = request.state.state
        identity = state.state.identity
        boundary = state.boundary
        if result.hold.identity != identity:
            raise KernelResultError("kernel hold has a different decision identity")
        if result.reward_event.identity != identity:
            raise KernelResultError(
                "kernel reward event has a different decision identity"
            )
        if result.hold.action != request.action or (
            result.reward_event.action != request.action
        ):
            raise KernelResultError(
                "the exact executed action must govern every held tensor and "
                "the terminal reward"
            )
        if result.reward_event.action_open_timestamp_ns != (
            boundary.action_open_timestamp_ns
        ):
            raise KernelResultError(
                "reward latency must start at this decision's action-open time"
            )
        if result.reward_event.clock_domain != boundary.clock_domain:
            raise KernelResultError("kernel reward uses a different clock domain")
        if result.cycle_end_timestamp_ns <= boundary.action_open_timestamp_ns:
            raise KernelResultError("cycle end must follow action open")
        if result.reward_event.resolution_timestamp_ns > (
            result.cycle_end_timestamp_ns
        ):
            raise KernelResultError("cycle end cannot precede terminal feedback")
