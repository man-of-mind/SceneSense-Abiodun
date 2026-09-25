"""Persistent, verifier-gated orchestration for the Run-4 SAC pipeline.

The runner owns one session/UE across decisions and joins the registered fit
scene provider, causal state provider, two-tensor action hold, external
empirical queue prediction, sequential environment, exploration gate, replay
and trainer.  It never invents radio observations, queue coefficients or a
network-profile actor input.

The hold proves the registered reward/held ordering and nominal 10-Hz cadence.
It does not relabel that nominal cadence as a measured inter-capture interval;
the current tensor contract carries sequence identities but no capture stamps.

Production remains fail closed until a future composite verifier binds the
near-capacity kernel, scaling, freshness and exploration evidence.  The
private test harness can exercise causality and checkpoint recovery, but it
can never call a gradient update or create production replay eligibility.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)

from . import environment, exploration, fit_scene_provider, held_payload, models
from . import production_state_provider, replay, run4_contract as contract
from . import sequential_kernel, trainer


SCHEMA_ID = "splitfusion.run4.persistent_runner.v1"
SCHEMA_VERSION = 1
REGISTERED_COMPOSITE_PREREQUISITES_SHA256: Optional[str] = None


class PersistentRunnerError(RuntimeError):
    pass


class RunnerAuthorizationError(PersistentRunnerError):
    pass


class RunnerBindingError(PersistentRunnerError):
    pass


class RunnerStateError(PersistentRunnerError):
    pass


class RunnerCheckpointError(PersistentRunnerError):
    pass


class RunnerTrainingUnavailable(PersistentRunnerError):
    pass


class RunnerAuthorizationClass(str, Enum):
    VERIFIED_TRAINING = "VERIFIED_TRAINING"
    TEST_ONLY_MECHANICS = "TEST_ONLY_MECHANICS"


class CausalStateStager(Protocol):
    @property
    def binding_sha256(self) -> str: ...

    def stage(
        self,
        *,
        identity: contract.DecisionIdentityV1,
        scene_draw: fit_scene_provider.FitSceneDrawV1,
        radio_state: sequential_kernel.RadioQueueStateV1,
        previous: Optional[contract.PreviousOutcomeV1],
        minimum_state_commit_timestamp_ns: int,
    ) -> production_state_provider.StagedDecisionInputsV1: ...

    def state_dict(self) -> Any: ...
    def load_state_dict(self, state: Any) -> None: ...


class EmpiricalPredictionProvider(Protocol):
    @property
    def binding_sha256(self) -> str: ...

    def predict(
        self, model_input: sequential_kernel.PredictionModelInputV1
    ) -> sequential_kernel.EmpiricalModelForecastV1: ...

    def state_dict(self) -> Any: ...
    def load_state_dict(self, state: Any) -> None: ...


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RunnerBindingError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RunnerBindingError(f"{name} must be a SHA-256 hex digest") from exc
    return value.lower()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerStateError(f"{name} must be a nonempty string")
    return value


def _exact_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RunnerStateError(f"{name} must be an exact int >= {minimum}")
    return value


def _tensor_fingerprint(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest(),
    }


def _fingerprint(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": _tensor_fingerprint(value)}
    if isinstance(value, Enum):
        return {"enum": type(value).__qualname__, "value": value.value}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunnerCheckpointError("checkpoint contains non-finite float")
        return {"float_hex": value.hex()}
    if isinstance(value, Mapping):
        rows = [(_fingerprint(k), _fingerprint(v)) for k, v in value.items()]
        rows.sort(key=lambda row: json.dumps(row[0], sort_keys=True))
        return {"mapping": rows}
    if isinstance(value, tuple):
        return {"tuple": [_fingerprint(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_fingerprint(item) for item in value]}
    if is_dataclass(value):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _fingerprint(getattr(value, item.name))
                for item in fields(value)
                if item.name != "_attestation"
            },
        }
    canonical = getattr(value, "canonical_sha256", None)
    if callable(canonical):
        canonical = canonical()
    if isinstance(canonical, str) and len(canonical) == 64:
        return {"canonical_type": type(value).__qualname__, "sha256": canonical}
    raise RunnerCheckpointError(
        f"unsupported checkpoint value {type(value).__name__}"
    )


def _state_digest(value: Any) -> str:
    return canonical_sha256(
        {"record": "run4_runner_checkpoint_material_v1", "value": _fingerprint(value)}
    )


def _trainer_config_sha256(config: trainer.TrainerConfigV1) -> str:
    if type(config) is not trainer.TrainerConfigV1:
        raise RunnerBindingError("trainer config must be exact TrainerConfigV1")
    return canonical_sha256(
        {
            "record": "splitfusion_run4_trainer_config_binding_v1",
            "value": {
                "actor_lr": config.actor_lr,
                "alpha_c": config.alpha_c,
                "alpha_d": config.alpha_d,
                "critic_lr": config.critic_lr,
                "float_dtype": str(config.float_dtype),
                "nominal_batch_size": config.nominal_batch_size,
                "tau": config.tau,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class CompositeRunnerPrerequisitesV1:
    fit_scene_provider_binding_sha256: str
    state_provider_binding_sha256: str
    kernel_prerequisites_sha256: str
    kernel_support_sha256: str
    state_stager_binding_sha256: str
    prediction_provider_binding_sha256: str
    replay_binding_sha256: str
    replay_capacity: int
    model_binding_sha256: str
    trainer_config_sha256: str
    warmup_schedule_id: str
    exploration_gate_config_sha256: str
    verifier_manifest_sha256: str
    near_capacity_kernel_validated: bool
    scaling_and_freshness_validated: bool
    prior_outcome_chain_validated: bool

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("sha256") or item.name.endswith("schedule_id"):
                _digest(value, item.name)
            elif item.name == "replay_capacity":
                _exact_int(value, item.name, minimum=1)
            elif type(value) is not bool:
                raise RunnerBindingError(f"{item.name} must be an exact bool")

    def to_dict(self) -> Dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)} | {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
        }

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class _AuthorizationAttestation:
    __slots__ = ("binding", "nonce")

    def __init__(self, binding: str, nonce: object) -> None:
        self.binding = binding
        self.nonce = nonce


_AUTHORIZATION_NONCE = object()


@dataclass(frozen=True, slots=True)
class RunnerAuthorizationV1:
    authorization_class: RunnerAuthorizationClass
    prerequisites_sha256: str
    verifier_manifest_sha256: str
    _attestation: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.authorization_class, RunnerAuthorizationClass):
            raise RunnerAuthorizationError("invalid authorization class")
        _digest(self.prerequisites_sha256, "prerequisites_sha256")
        _digest(self.verifier_manifest_sha256, "verifier_manifest_sha256")

    def _binding(self) -> str:
        return canonical_sha256(
            {
                "authorization_class": self.authorization_class.value,
                "prerequisites_sha256": self.prerequisites_sha256,
                "verifier_manifest_sha256": self.verifier_manifest_sha256,
            }
        )

    def require_attested(self) -> None:
        if not (
            type(self._attestation) is _AuthorizationAttestation
            and self._attestation.nonce is _AUTHORIZATION_NONCE
            and self._attestation.binding == self._binding()
        ):
            raise RunnerAuthorizationError("runner authorization is not attested")

    def require_training_eligible(self) -> None:
        self.require_attested()
        if self.authorization_class is not RunnerAuthorizationClass.VERIFIED_TRAINING:
            raise RunnerTrainingUnavailable(
                "test-only mechanics cannot authorize gradients"
            )


def _issue_authorization(
    prerequisites: CompositeRunnerPrerequisitesV1,
    authorization_class: RunnerAuthorizationClass,
) -> RunnerAuthorizationV1:
    candidate = RunnerAuthorizationV1(
        authorization_class=authorization_class,
        prerequisites_sha256=prerequisites.canonical_sha256,
        verifier_manifest_sha256=prerequisites.verifier_manifest_sha256,
    )
    object.__setattr__(
        candidate,
        "_attestation",
        _AuthorizationAttestation(candidate._binding(), _AUTHORIZATION_NONCE),
    )
    return candidate


def verify_composite_prerequisites(
    prerequisites: CompositeRunnerPrerequisitesV1,
) -> RunnerAuthorizationV1:
    if type(prerequisites) is not CompositeRunnerPrerequisitesV1:
        raise RunnerAuthorizationError("invalid composite prerequisites")
    if not all(
        (
            prerequisites.near_capacity_kernel_validated,
            prerequisites.scaling_and_freshness_validated,
            prerequisites.prior_outcome_chain_validated,
        )
    ):
        raise RunnerAuthorizationError("composite evidence is incomplete")
    if REGISTERED_COMPOSITE_PREREQUISITES_SHA256 is None:
        raise RunnerAuthorizationError(
            "no reviewed composite Run-4 verifier binding is registered"
        )
    if prerequisites.canonical_sha256 != REGISTERED_COMPOSITE_PREREQUISITES_SHA256:
        raise RunnerAuthorizationError("composite prerequisites digest differs")
    return _issue_authorization(
        prerequisites, RunnerAuthorizationClass.VERIFIED_TRAINING
    )


def _authorize_test_only(
    prerequisites: CompositeRunnerPrerequisitesV1,
) -> RunnerAuthorizationV1:
    return _issue_authorization(
        prerequisites, RunnerAuthorizationClass.TEST_ONLY_MECHANICS
    )


@dataclass(frozen=True, slots=True)
class PortableJournalReplayRowV1:
    """Attestation-free inputs for reissuing one durable journal row.

    A portable checkpoint may retain the causal decision, prediction and
    successor inputs, but it must not persist interpreter-private transition
    attestations. The expected digests below are comparison targets only;
    :meth:`_RunnerCore.reissue_portable_journal` rebuilds the transition via
    the normal validated execution path before accepting any of them.
    """

    decision: sequential_kernel.KernelDecisionInputV1
    prediction: sequential_kernel.EmpiricalStepPredictionV1
    successor_staged: production_state_provider.StagedDecisionInputsV1
    expected_transition_sha256: str
    expected_environment_transition_sha256: str
    expected_journal_sha256: str

    def __post_init__(self) -> None:
        if type(self.decision) is not sequential_kernel.KernelDecisionInputV1:
            raise RunnerCheckpointError("portable replay contains foreign decision")
        if type(self.prediction) is not sequential_kernel.EmpiricalStepPredictionV1:
            raise RunnerCheckpointError("portable replay contains foreign prediction")
        if type(self.successor_staged) is not (
            production_state_provider.StagedDecisionInputsV1
        ):
            raise RunnerCheckpointError("portable replay contains foreign successor")
        self.decision.action.require_reconciled()
        for name in (
            "expected_transition_sha256",
            "expected_environment_transition_sha256",
            "expected_journal_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.decision.prediction_request_sha256 != (
            self.prediction.prediction_request_sha256
        ):
            raise RunnerCheckpointError(
                "portable decision/prediction-request digest mismatch"
            )
        expected_identity = contract.DecisionIdentityV1(
            self.decision.identity.session_uuid,
            self.decision.identity.ue_id,
            self.decision.identity.decision_seq + 1,
        )
        if self.successor_staged.identity != expected_identity:
            raise RunnerCheckpointError(
                "portable replay successor identity is not contiguous"
            )
        if self.successor_staged.radio_state.canonical_sha256 != (
            self.prediction.next_state.canonical_sha256
        ):
            raise RunnerCheckpointError(
                "portable replay successor radio state was substituted"
            )
        if self.expected_transition_sha256 != (
            self.expected_environment_transition_sha256
        ):
            raise RunnerCheckpointError(
                "portable replay environment/transition digest mismatch"
            )
        observed_journal = canonical_sha256(
            {
                "decision": self.decision.canonical_sha256,
                "prediction": self.prediction.canonical_sha256,
                "staged": self.successor_staged.canonical_sha256,
                "transition": self.expected_transition_sha256,
            }
        )
        if observed_journal != self.expected_journal_sha256:
            raise RunnerCheckpointError("portable replay journal fingerprint mismatch")


@dataclass(frozen=True, slots=True)
class RunnerStepJournalV1:
    decision: sequential_kernel.KernelDecisionInputV1
    prediction: sequential_kernel.EmpiricalStepPredictionV1
    successor_staged: production_state_provider.StagedDecisionInputsV1
    transition: contract.SemiMarkovTransitionV2
    environment_transition_sha256: str

    def __post_init__(self) -> None:
        if type(self.decision) is not sequential_kernel.KernelDecisionInputV1:
            raise RunnerCheckpointError("foreign decision in journal")
        if type(self.prediction) is not sequential_kernel.EmpiricalStepPredictionV1:
            raise RunnerCheckpointError("foreign prediction in journal")
        if type(self.successor_staged) is not production_state_provider.StagedDecisionInputsV1:
            raise RunnerCheckpointError("foreign successor staging in journal")
        if type(self.transition) is not contract.SemiMarkovTransitionV2:
            raise RunnerCheckpointError("foreign transition in journal")
        self.transition.require_attested()
        _digest(self.environment_transition_sha256, "environment_transition_sha256")
        if (
            self.decision.prediction_request_sha256
            != self.prediction.prediction_request_sha256
        ):
            raise RunnerCheckpointError(
                "decision/prediction-request digest mismatch"
            )
        if self.transition.canonical_sha256() != self.environment_transition_sha256:
            raise RunnerCheckpointError("environment/transition digest mismatch")
        if self.transition.state.state.identity != self.decision.identity:
            raise RunnerCheckpointError("journal decision/state identity mismatch")
        if self.transition.action != self.decision.action:
            raise RunnerCheckpointError("journal decision/transition action mismatch")
        if self.transition.hold.canonical_sha256() != (
            self.decision.hold.canonical_sha256()
        ):
            raise RunnerCheckpointError("journal decision/transition hold mismatch")
        expected_successor = contract.DecisionIdentityV1(
            self.decision.identity.session_uuid,
            self.decision.identity.ue_id,
            self.decision.identity.decision_seq + 1,
        )
        if self.successor_staged.identity != expected_successor:
            raise RunnerCheckpointError("journal successor identity is not contiguous")
        if self.successor_staged.radio_state.canonical_sha256 != (
            self.prediction.next_state.canonical_sha256
        ):
            raise RunnerCheckpointError("journal successor radio state was substituted")
        expected_previous = contract.PreviousOutcomeV1.from_resolution(
            self.transition.reward_resolution
        )
        if self.successor_staged.expected_previous_sha256 != (
            expected_previous.canonical_sha256()
        ):
            raise RunnerCheckpointError("journal successor previous outcome differs")
        if self.successor_staged.boundary.state_commit_timestamp_ns < (
            self.transition.reward_resolution.resolution_timestamp_ns
        ):
            raise RunnerCheckpointError("journal successor predates reward resolution")
        if self.transition.next_state is None or (
            self.transition.next_state.state.identity != expected_successor
        ):
            raise RunnerCheckpointError("journal transition lacks exact successor")
        if self.transition.cycle_end_timestamp_ns != (
            self.successor_staged.boundary.action_open_timestamp_ns
        ):
            raise RunnerCheckpointError(
                "journal transition end is not measured successor action-open"
            )

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "decision": self.decision.canonical_sha256,
                "prediction": self.prediction.canonical_sha256,
                "staged": self.successor_staged.canonical_sha256,
                "transition": self.transition.canonical_sha256(),
            }
        )


@dataclass(frozen=True, slots=True)
class PersistentRunnerCheckpointV1:
    runner_binding_sha256: str
    session_uuid: str
    ue_id: str
    genesis_staged: production_state_provider.StagedDecisionInputsV1
    journal: Tuple[RunnerStepJournalV1, ...]
    fit_scene_state: fit_scene_provider.FitSceneProviderStateV1
    initial_kernel_checkpoint: sequential_kernel.KernelCheckpointV1
    kernel_checkpoint: sequential_kernel.KernelCheckpointV1
    state_stager_state: Any
    prediction_provider_state: Any
    actor_state_dict: Mapping[str, Any]
    critics_state_dict: Mapping[str, Any]
    actor_optimizer_state_dict: Mapping[str, Any]
    critic_optimizer_state_dict: Mapping[str, Any]
    trainer_update_count: int
    decision_q_rng_state: torch.Tensor
    decision_mode_rng_state: torch.Tensor
    replay_rng_state: torch.Tensor
    trainer_target_rng_state: torch.Tensor
    trainer_actor_rng_state: torch.Tensor
    current_state_sha256: str
    exploration_decision_count: int
    replay_accepted_count: int
    replay_evicted_count: int
    replay_seen_digest_count: int
    replay_seen_identity_count: int
    replay_resident_transition_digests: Tuple[str, ...]
    checkpoint_sha256: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        _digest(self.runner_binding_sha256, "runner_binding_sha256")
        try:
            parsed = str(uuid.UUID(self.session_uuid))
        except (ValueError, TypeError, AttributeError) as exc:
            raise RunnerCheckpointError("invalid checkpoint session UUID") from exc
        if parsed != self.session_uuid:
            raise RunnerCheckpointError("checkpoint session UUID is not canonical")
        _text(self.ue_id, "ue_id")
        if type(self.genesis_staged) is not production_state_provider.StagedDecisionInputsV1:
            raise RunnerCheckpointError("foreign genesis staging")
        if type(self.journal) is not tuple or any(
            type(row) is not RunnerStepJournalV1 for row in self.journal
        ):
            raise RunnerCheckpointError("journal must be exact runner records")
        if type(self.fit_scene_state) is not fit_scene_provider.FitSceneProviderStateV1:
            raise RunnerCheckpointError("foreign fit-scene checkpoint")
        for name in ("initial_kernel_checkpoint", "kernel_checkpoint"):
            if type(getattr(self, name)) is not sequential_kernel.KernelCheckpointV1:
                raise RunnerCheckpointError(f"foreign {name}")
        for name in (
            "trainer_update_count",
            "exploration_decision_count",
            "replay_accepted_count",
            "replay_evicted_count",
            "replay_seen_digest_count",
            "replay_seen_identity_count",
        ):
            _exact_int(getattr(self, name), name)
        if type(self.replay_resident_transition_digests) is not tuple:
            raise RunnerCheckpointError("resident replay digests must be a tuple")
        for digest in self.replay_resident_transition_digests:
            _digest(digest, "replay_resident_transition_digest")
        for name in (
            "decision_q_rng_state",
            "decision_mode_rng_state",
            "replay_rng_state",
            "trainer_target_rng_state",
            "trainer_actor_rng_state",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise RunnerCheckpointError(f"{name} must be a CPU tensor")
        _digest(self.current_state_sha256, "current_state_sha256")
        observed = self.compute_sha256()
        if self.checkpoint_sha256 and self.checkpoint_sha256 != observed:
            raise RunnerCheckpointError("checkpoint fingerprint mismatch")
        object.__setattr__(self, "checkpoint_sha256", observed)

    def compute_sha256(self) -> str:
        return canonical_sha256(
            {
                "actor": _state_digest(self.actor_state_dict),
                "actor_optimizer": _state_digest(self.actor_optimizer_state_dict),
                "critics": _state_digest(self.critics_state_dict),
                "critic_optimizer": _state_digest(self.critic_optimizer_state_dict),
                "current_state": self.current_state_sha256,
                "decision_mode_rng": _state_digest(self.decision_mode_rng_state),
                "decision_q_rng": _state_digest(self.decision_q_rng_state),
                "exploration_count": self.exploration_decision_count,
                "fit_scene_state": _state_digest(self.fit_scene_state),
                "genesis_staged": self.genesis_staged.canonical_sha256,
                "initial_kernel": self.initial_kernel_checkpoint.canonical_sha256,
                "journal": [row.canonical_sha256 for row in self.journal],
                "kernel": self.kernel_checkpoint.canonical_sha256,
                "prediction_provider": _state_digest(self.prediction_provider_state),
                "record": "splitfusion_run4_persistent_checkpoint_v1",
                "replay_accepted": self.replay_accepted_count,
                "replay_evicted": self.replay_evicted_count,
                "replay_seen_digests": self.replay_seen_digest_count,
                "replay_seen_identities": self.replay_seen_identity_count,
                "replay_resident": list(self.replay_resident_transition_digests),
                "replay_rng": _state_digest(self.replay_rng_state),
                "runner_binding": self.runner_binding_sha256,
                "session_uuid": self.session_uuid,
                "state_stager": _state_digest(self.state_stager_state),
                "trainer_actor_rng": _state_digest(self.trainer_actor_rng_state),
                "trainer_target_rng": _state_digest(self.trainer_target_rng_state),
                "trainer_updates": self.trainer_update_count,
                "ue_id": self.ue_id,
            }
        )


class _PreparedKernelAdapter:
    """One-shot adapter from the caller-bound empirical kernel to environment."""

    def __init__(self, kernel: sequential_kernel.Run4SequentialRadioQueueKernelV1):
        self.kernel = kernel
        self._decision: Optional[sequential_kernel.KernelDecisionInputV1] = None
        self._result: Optional[sequential_kernel.KernelStepResultV1] = None

    @property
    def pending(self) -> bool:
        return self._result is not None

    def prepare(
        self,
        decision: sequential_kernel.KernelDecisionInputV1,
        prediction: sequential_kernel.EmpiricalStepPredictionV1,
    ) -> sequential_kernel.KernelStepResultV1:
        if self.pending:
            raise RunnerStateError("prepared kernel result was not consumed")
        result = self.kernel.advance(decision=decision, prediction=prediction)
        self._decision = decision
        self._result = result
        return result

    def execute_cycle(
        self, request: environment.KernelCycleRequestV1
    ) -> environment.KernelCycleResultV1:
        decision, result = self._decision, self._result
        if decision is None or result is None:
            raise RunnerStateError("no empirical kernel prediction is prepared")
        boundary = request.state.state.boundary
        if (
            request.state.state.state.identity != decision.identity
            or request.action != decision.action
            or boundary.action_open_timestamp_ns != decision.action_open_timestamp_ns
        ):
            raise RunnerStateError("prepared decision differs from environment request")
        self._decision = None
        self._result = None
        return result.to_environment_result(
            episode_boundary=contract.EpisodeBoundary.CONTINUES
        )


class _RunnerCore:
    def __init__(
        self,
        *,
        prerequisites: CompositeRunnerPrerequisitesV1,
        authorization: RunnerAuthorizationV1,
        fit_provider: fit_scene_provider.Run4FitSceneProviderV1,
        state_provider: production_state_provider.Run4ProductionStateProviderV1,
        state_stager: CausalStateStager,
        kernel: sequential_kernel.Run4SequentialRadioQueueKernelV1,
        prediction_provider: EmpiricalPredictionProvider,
        warmup_schedule: exploration.StratifiedWarmupSchedule,
        coverage_gate: exploration.CoverageGateConfig,
        replay_buffer: replay._ReplayBufferCore,
        model_bundle: models.Run4ModelBundleV1,
        sac_trainer: trainer._Run4TrainerCore,
        decision_q_generator: torch.Generator,
        decision_mode_generator: torch.Generator,
        replay_generator: torch.Generator,
    ) -> None:
        if type(prerequisites) is not CompositeRunnerPrerequisitesV1:
            raise RunnerBindingError("invalid composite prerequisites")
        if type(authorization) is not RunnerAuthorizationV1:
            raise RunnerAuthorizationError("invalid authorization")
        authorization.require_attested()
        if authorization.prerequisites_sha256 != prerequisites.canonical_sha256:
            raise RunnerAuthorizationError("authorization/prerequisites mismatch")
        exact = (
            (fit_provider, fit_scene_provider.Run4FitSceneProviderV1, "fit_provider"),
            (state_provider, production_state_provider.Run4ProductionStateProviderV1, "state_provider"),
            (kernel, sequential_kernel.Run4SequentialRadioQueueKernelV1, "kernel"),
            (warmup_schedule, exploration.StratifiedWarmupSchedule, "warmup_schedule"),
            (coverage_gate, exploration.CoverageGateConfig, "coverage_gate"),
            (model_bundle, models.Run4ModelBundleV1, "model_bundle"),
        )
        for value, wanted, name in exact:
            if type(value) is not wanted:
                raise RunnerBindingError(f"{name} must be exactly {wanted.__name__}")
        for source, name, methods in (
            (state_stager, "state_stager", ("stage", "state_dict", "load_state_dict")),
            (prediction_provider, "prediction_provider", ("predict", "state_dict", "load_state_dict")),
        ):
            _digest(getattr(source, "binding_sha256", None), f"{name}.binding_sha256")
            if any(not callable(getattr(source, method, None)) for method in methods):
                raise RunnerBindingError(f"{name} lacks checkpointable API")
        generators = (
            (decision_q_generator, "decision_q_generator"),
            (decision_mode_generator, "decision_mode_generator"),
            (replay_generator, "replay_generator"),
        )
        for value, name in generators:
            if not isinstance(value, torch.Generator):
                raise RunnerBindingError(f"{name} must be torch.Generator")
            if value is torch.default_generator or value.device.type != "cpu":
                raise RunnerBindingError(f"{name} must be a private CPU generator")
        if len({id(value) for value, _ in generators}) != 3:
            raise RunnerBindingError("runner RNG streams must be distinct")

        self.prerequisites = prerequisites
        self.authorization = authorization
        self.fit_provider = fit_provider
        self.state_provider = state_provider
        self.state_stager = state_stager
        self.kernel = kernel
        self.prediction_provider = prediction_provider
        self.warmup_schedule = warmup_schedule
        self.coverage_gate = coverage_gate
        self.replay_buffer = replay_buffer
        self.model_bundle = model_bundle
        self.trainer = sac_trainer
        self._decision_q_generator = decision_q_generator
        self._decision_mode_generator = decision_mode_generator
        self._replay_generator = replay_generator
        self._catalog = action_contract.load_contract()
        self._adapter = _PreparedKernelAdapter(kernel)
        self.environment = environment.Run4SequentialEnvironmentV1(
            state_provider=state_provider,
            kernel=self._adapter,
            gamma=replay_buffer.binding.gamma,
            evidence_class=environment.EnvironmentEvidenceClass.SYNTHETIC_MECHANICS_FIXTURE,
        )
        self._initial_kernel_checkpoint = kernel.checkpoint()
        self._ledger = self._make_ledger()
        self._session_uuid: Optional[str] = None
        self._ue_id: Optional[str] = None
        self._genesis_staged: Optional[production_state_provider.StagedDecisionInputsV1] = None
        self._current_draw: Optional[fit_scene_provider.FitSceneDrawV1] = None
        self._journal: list[RunnerStepJournalV1] = []
        self._faulted = False
        self._assert_bindings()

    def _make_ledger(self) -> exploration.ExplorationCoverageLedger:
        raise NotImplementedError

    def _record_coverage(self, transition: contract.SemiMarkovTransitionV2) -> None:
        raise NotImplementedError

    def _require_training_authority(self) -> None:
        raise NotImplementedError

    def _require_actor_action_ready(self) -> None:
        raise NotImplementedError

    @property
    def runner_binding_sha256(self) -> str:
        return self.prerequisites.canonical_sha256

    @property
    def started(self) -> bool:
        return self._genesis_staged is not None

    @property
    def decision_count(self) -> int:
        return len(self._journal)

    @property
    def current_features(self) -> Tuple[float, ...]:
        return self.environment.current_state.features.as_tuple()

    @property
    def transition_digests(self) -> Tuple[str, ...]:
        return tuple(row.transition.canonical_sha256() for row in self._journal)

    def _assert_bindings(self) -> None:
        kernel_checkpoint = self.kernel.checkpoint()
        observed = {
            "fit_scene_provider_binding_sha256": self.fit_provider.binding.canonical_sha256,
            "state_provider_binding_sha256": self.state_provider.binding.canonical_sha256,
            "kernel_prerequisites_sha256": kernel_checkpoint.prerequisites_sha256,
            "kernel_support_sha256": kernel_checkpoint.support_sha256,
            "state_stager_binding_sha256": self.state_stager.binding_sha256,
            "prediction_provider_binding_sha256": self.prediction_provider.binding_sha256,
            "replay_binding_sha256": canonical_sha256(
                self.replay_buffer.binding.to_canonical_dict()
            ),
            "replay_capacity": self.replay_buffer.capacity,
            "model_binding_sha256": self.model_bundle.binding_sha256,
            "trainer_config_sha256": _trainer_config_sha256(
                self.trainer.config
            ),
            "warmup_schedule_id": self.warmup_schedule.config.schedule_id,
            "exploration_gate_config_sha256": self.coverage_gate.config_sha256,
        }
        for name, actual in observed.items():
            expected = getattr(self.prerequisites, name)
            if actual != expected:
                raise RunnerBindingError(
                    f"composite binding differs at {name}: {actual} != {expected}"
                )
        self.trainer.expected_binding.assert_exactly(self.replay_buffer.binding)
        models.validate_run4_models(self.model_bundle.actor, self.model_bundle.critics)
        if self.trainer.actor is not self.model_bundle.actor or (
            self.trainer.critics is not self.model_bundle.critics
        ):
            raise RunnerBindingError("trainer does not own supplied model bundle")
        if self.state_provider.binding.fit_scene_provider_binding_sha256 != (
            self.fit_provider.binding.canonical_sha256
        ):
            raise RunnerBindingError("state/scene provider binding mismatch")
        if self.replay_buffer.binding.freshness_policy_sha256 != (
            self.state_provider.binding.freshness_sha256
        ):
            raise RunnerBindingError("replay/state freshness binding mismatch")
        if self.replay_buffer.binding.empirical_scaling_sha256 != (
            self.state_provider.binding.scaling_sha256
        ):
            raise RunnerBindingError("replay/state scaling binding mismatch")

    def start(self, *, session_uuid: str, ue_id: str) -> None:
        if self.started or self._faulted:
            raise RunnerStateError("runner is already started or faulted")
        self._assert_bindings()
        parsed = str(uuid.UUID(session_uuid))
        if parsed != session_uuid:
            raise RunnerStateError("session_uuid must be canonical")
        _text(ue_id, "ue_id")
        radio = self.kernel.current_state
        if (radio.session_uuid, radio.ue_id, radio.decision_seq) != (
            session_uuid,
            ue_id,
            0,
        ):
            raise RunnerStateError("kernel genesis differs from requested session/UE")
        try:
            draw = self.fit_provider.select_scene()
            identity = contract.DecisionIdentityV1(session_uuid, ue_id, 0)
            staged = self.state_stager.stage(
                identity=identity,
                scene_draw=draw,
                radio_state=radio,
                previous=None,
                minimum_state_commit_timestamp_ns=0,
            )
            self._validate_staged_inputs(
                staged=staged,
                identity=identity,
                scene_draw=draw,
                radio_state=radio,
                previous=None,
                minimum_state_commit_timestamp_ns=0,
                minimum_action_open_timestamp_ns=0,
            )
            self.state_provider.stage_decision(staged)
            bundle = self.environment.reset(session_uuid=session_uuid, ue_id=ue_id)
            if bundle.state.state.identity != identity:
                raise RunnerStateError("genesis identity drifted")
            self._session_uuid = session_uuid
            self._ue_id = ue_id
            self._genesis_staged = staged
            self._current_draw = draw
        except Exception:
            # Genesis consumes external/provider state before it becomes a
            # visible active runner.  A failed start must never be retryable
            # against those partially advanced components.
            self._faulted = True
            raise

    @staticmethod
    def _validate_staged_inputs(
        *,
        staged: production_state_provider.StagedDecisionInputsV1,
        identity: contract.DecisionIdentityV1,
        scene_draw: fit_scene_provider.FitSceneDrawV1,
        radio_state: sequential_kernel.RadioQueueStateV1,
        previous: Optional[contract.PreviousOutcomeV1],
        minimum_state_commit_timestamp_ns: int,
        minimum_action_open_timestamp_ns: int,
    ) -> None:
        """Close the stager join over every exact caller-supplied input."""

        minimum_commit = _exact_int(
            minimum_state_commit_timestamp_ns,
            "minimum_state_commit_timestamp_ns",
        )
        minimum_action_open = _exact_int(
            minimum_action_open_timestamp_ns,
            "minimum_action_open_timestamp_ns",
        )
        if type(staged) is not production_state_provider.StagedDecisionInputsV1:
            raise RunnerStateError("state stager returned a foreign record")
        expected_previous = (
            None if previous is None else previous.canonical_sha256()
        )
        if staged.identity != identity:
            raise RunnerStateError("state stager substituted decision identity")
        if staged.scene_draw.canonical_sha256 != scene_draw.canonical_sha256:
            raise RunnerStateError("state stager substituted the selected scene")
        if staged.radio_state.canonical_sha256 != radio_state.canonical_sha256:
            raise RunnerStateError("state stager substituted radio/queue state")
        if staged.expected_previous_sha256 != expected_previous:
            raise RunnerStateError("state stager substituted previous outcome")
        if staged.boundary.state_commit_timestamp_ns < minimum_commit:
            raise RunnerStateError("state stager committed before causal lower bound")
        if staged.boundary.action_open_timestamp_ns < minimum_action_open:
            raise RunnerStateError(
                "successor action-open precedes the completed action-hold cadence"
            )

    def _execution(self, mode_id: int, q_e4: int) -> ExecutedActionIdentity:
        executable = self._catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        action = ExecutedActionIdentity.from_executable_action(
            executable, self._catalog
        )
        if (action.mode_id, action.q_e4) != (mode_id, q_e4):
            raise RunnerStateError("catalog changed selected action")
        return action

    def _actor_action(self) -> ExecutedActionIdentity:
        self._require_actor_action_ready()
        state = torch.tensor((self.current_features,), dtype=torch.float32)
        with torch.no_grad():
            sample = self.model_bundle.actor.sample_all_modes(
                state, generator=self._decision_q_generator
            )
            mode = int(
                torch.multinomial(
                    sample.probs[0], 1, generator=self._decision_mode_generator
                )[0]
            )
            q_e4 = int(sample.q_e4[0, mode])
        return self._execution(mode, q_e4)

    def select_action(self) -> ExecutedActionIdentity:
        if not self.started or self._faulted:
            raise RunnerStateError("runner is not in a healthy active sequence")
        if self.decision_count < len(self.warmup_schedule):
            chosen = self.warmup_schedule.action_at(self.decision_count)
            return self._execution(chosen.mode_id, chosen.q_e4)
        return self._actor_action()

    @staticmethod
    def _coverage_observation(
        guarded: contract.GuardedPolicyStateV2,
    ) -> exploration.CoverageObservation:
        state = guarded.state
        previous = state.previous
        prior = None
        if previous is not None:
            prior = exploration.PreviousDecisionObservation(
                mode_id=previous.action.mode_id,
                q_e4=previous.action.q_e4,
                success=previous.terminal is contract.RewardTerminal.SUCCESS,
                quality=previous.q_perc,
                latency_ms=previous.latency_ms,
            )
        values = (
            state.camera_si.value,
            state.radar_p40.value,
            state.prior_ul_mcs.observation.value,
            state.pre_action_rlc_backlog.value,
        )
        if any(value is None for value in values):
            raise RunnerStateError("guarded actor state contains missing telemetry")
        return exploration.CoverageObservation(
            scene_si=float(values[0]),
            scene_p40=float(values[1]),
            prior_ul_mcs_index=int(values[2]),
            rlc_backlog_bytes=float(values[3]),
            previous=prior,
        )

    def _build_transition(
        self,
        *,
        current: environment.DecisionStateBundleV1,
        action: ExecutedActionIdentity,
        step_result: sequential_kernel.KernelStepResultV1,
        next_bundle: environment.DecisionStateBundleV1,
        transition_cycle_end_timestamp_ns: int,
    ) -> contract.SemiMarkovTransitionV2:
        resolution = contract.resolve_reward(step_result.reward_event)
        elapsed = (
            transition_cycle_end_timestamp_ns
            - current.state.boundary.action_open_timestamp_ns
        )
        return contract.build_transition(
            state=current.state,
            state_features=current.features,
            action=action,
            hold=step_result.hold,
            reward_resolution=resolution,
            next_state=next_bundle.state,
            next_state_features=next_bundle.features,
            episode_boundary=contract.EpisodeBoundary.CONTINUES,
            duration=step_result.hold.duration,
            cycle_end_timestamp_ns=transition_cycle_end_timestamp_ns,
            elapsed_virtual_ns=elapsed,
            gamma=self.replay_buffer.binding.gamma,
            discount=self.replay_buffer.binding.gamma ** step_result.hold.duration,
        )

    def _ingest(self, transition: contract.SemiMarkovTransitionV2) -> None:
        if self._ledger.decision_count < len(self.warmup_schedule):
            self._record_coverage(transition)
        self.replay_buffer.insert(transition)

    def _execute_prebuilt(
        self,
        *,
        decision: sequential_kernel.KernelDecisionInputV1,
        prediction: sequential_kernel.EmpiricalStepPredictionV1,
        successor_staged: production_state_provider.StagedDecisionInputsV1,
        record: bool,
    ) -> RunnerStepJournalV1:
        current = self.environment.current_state
        step_result = self._adapter.prepare(decision, prediction)
        resolution = contract.resolve_reward(step_result.reward_event)
        previous = contract.PreviousOutcomeV1.from_resolution(resolution)
        successor_identity = contract.DecisionIdentityV1(
            decision.identity.session_uuid,
            decision.identity.ue_id,
            decision.identity.decision_seq + 1,
        )
        self._validate_staged_inputs(
            staged=successor_staged,
            identity=successor_identity,
            scene_draw=successor_staged.scene_draw,
            radio_state=step_result.next_radio_state,
            previous=previous,
            minimum_state_commit_timestamp_ns=(
                resolution.resolution_timestamp_ns
            ),
            minimum_action_open_timestamp_ns=(
                current.state.boundary.action_open_timestamp_ns
                + step_result.hold.duration * contract.TRANSMIT_PERIOD_NS
            ),
        )
        self.state_provider.stage_decision(successor_staged)
        diagnostic = self.environment.step(decision.action)
        if type(diagnostic) is not environment.SyntheticMechanicsCycleV1:
            raise RunnerStateError("learning cycle became excluded")
        next_bundle = self.environment.current_state
        transition = self._build_transition(
            current=current,
            action=decision.action,
            step_result=step_result,
            next_bundle=next_bundle,
            transition_cycle_end_timestamp_ns=(
                diagnostic.transition_cycle_end_timestamp_ns
            ),
        )
        journal = RunnerStepJournalV1(
            decision=decision,
            prediction=prediction,
            successor_staged=successor_staged,
            transition=transition,
            environment_transition_sha256=diagnostic.transition_sha256,
        )
        if diagnostic.reward_request_flags != (True, False):
            raise RunnerStateError("decision must contain reward then held tensor")
        self._ingest(transition)
        if record:
            self._journal.append(journal)
        return journal

    def step(self) -> RunnerStepJournalV1:
        if not self.started or self._faulted or self._current_draw is None:
            raise RunnerStateError("runner is not in a healthy active sequence")
        self._assert_bindings()
        current = self.environment.current_state
        identity = current.state.state.identity
        action = self.select_action()
        try:
            reward_tensor = self.fit_provider.reward_tensor(
                draw=self._current_draw,
                action=action,
                tensor_seq=identity.decision_seq * contract.MINIMUM_HOLD_TENSORS,
            )
            held_tensor = self.fit_provider.held_tensor(
                counter=held_payload.HeldSelectionCounterV1(
                    session_id=identity.session_uuid,
                    decision_seq=identity.decision_seq,
                    held_ordinal=1,
                    rng_stream_id=self.fit_provider.binding.held_rng_stream_id,
                ),
                action=action,
                tensor_seq=identity.decision_seq * contract.MINIMUM_HOLD_TENSORS + 1,
            )
            decision = sequential_kernel.KernelDecisionInputV1(
                identity=identity,
                current_radio_state_sha256=self.kernel.current_state.canonical_sha256,
                action=action,
                reward_tensor=reward_tensor,
                held_tensors=(held_tensor,),
                action_open_timestamp_ns=current.state.boundary.action_open_timestamp_ns,
                clock_domain=current.state.boundary.clock_domain,
                calibration_partition=sequential_kernel.KernelCalibrationPartition.FIT,
            )
            prediction_request = decision.to_prediction_request(
                self.kernel.current_state
            )
            forecast = self.prediction_provider.predict(
                prediction_request.model_input
            )
            if type(forecast) is not sequential_kernel.EmpiricalModelForecastV1:
                raise RunnerStateError("prediction provider returned foreign forecast")
            prediction = prediction_request.bind_forecast(forecast)
            step_result = self._adapter.prepare(decision, prediction)
            resolution = contract.resolve_reward(step_result.reward_event)
            previous = contract.PreviousOutcomeV1.from_resolution(resolution)
            next_draw = self.fit_provider.select_scene()
            successor = self.state_stager.stage(
                identity=contract.DecisionIdentityV1(
                    identity.session_uuid, identity.ue_id, identity.decision_seq + 1
                ),
                scene_draw=next_draw,
                radio_state=step_result.next_radio_state,
                previous=previous,
                minimum_state_commit_timestamp_ns=resolution.resolution_timestamp_ns,
            )
            self._validate_staged_inputs(
                staged=successor,
                identity=contract.DecisionIdentityV1(
                    identity.session_uuid, identity.ue_id, identity.decision_seq + 1
                ),
                scene_draw=next_draw,
                radio_state=step_result.next_radio_state,
                previous=previous,
                minimum_state_commit_timestamp_ns=(
                    resolution.resolution_timestamp_ns
                ),
                minimum_action_open_timestamp_ns=(
                    current.state.boundary.action_open_timestamp_ns
                    + step_result.hold.duration * contract.TRANSMIT_PERIOD_NS
                ),
            )
            self.state_provider.stage_decision(successor)
            diagnostic = self.environment.step(action)
            if type(diagnostic) is not environment.SyntheticMechanicsCycleV1:
                raise RunnerStateError("learning cycle became excluded")
            next_bundle = self.environment.current_state
            transition = self._build_transition(
                current=current,
                action=action,
                step_result=step_result,
                next_bundle=next_bundle,
                transition_cycle_end_timestamp_ns=(
                    diagnostic.transition_cycle_end_timestamp_ns
                ),
            )
            journal = RunnerStepJournalV1(
                decision=decision,
                prediction=prediction,
                successor_staged=successor,
                transition=transition,
                environment_transition_sha256=diagnostic.transition_sha256,
            )
            if diagnostic.reward_request_flags != (True, False):
                raise RunnerStateError("decision must contain reward then held tensor")
            self._ingest(transition)
            self._journal.append(journal)
            self._current_draw = next_draw
            return journal
        except Exception:
            self._faulted = True
            raise

    def train_once(self, batch_size: int) -> trainer.UpdateMetricsV1:
        self._require_training_authority()
        self._ledger.require_gradient_start()
        batch = self.replay_buffer.sample(batch_size, self._replay_generator)
        return self.trainer.update_once(batch)

    def checkpoint(self) -> PersistentRunnerCheckpointV1:
        if not self.started or self._faulted:
            raise RunnerCheckpointError("checkpoint requires healthy active runner")
        self._assert_bindings()
        if self._adapter.pending:
            raise RunnerCheckpointError("cannot checkpoint a partial cycle")
        if self._genesis_staged is None or self._current_draw is None:
            raise RunnerCheckpointError("runner lost active sequence material")
        if self.fit_provider.active_draw != self._current_draw:
            raise RunnerCheckpointError("fit-provider active draw drifted")
        return PersistentRunnerCheckpointV1(
            runner_binding_sha256=self.runner_binding_sha256,
            session_uuid=self._session_uuid or "",
            ue_id=self._ue_id or "",
            genesis_staged=self._genesis_staged,
            journal=tuple(self._journal),
            fit_scene_state=self.fit_provider.state_dict(),
            initial_kernel_checkpoint=self._initial_kernel_checkpoint,
            kernel_checkpoint=self.kernel.checkpoint(),
            state_stager_state=copy.deepcopy(self.state_stager.state_dict()),
            prediction_provider_state=copy.deepcopy(
                self.prediction_provider.state_dict()
            ),
            actor_state_dict=copy.deepcopy(self.model_bundle.actor.state_dict()),
            critics_state_dict=copy.deepcopy(self.model_bundle.critics.state_dict()),
            actor_optimizer_state_dict=copy.deepcopy(
                self.trainer.actor_optimizer.state_dict()
            ),
            critic_optimizer_state_dict=copy.deepcopy(
                self.trainer.critic_optimizer.state_dict()
            ),
            trainer_update_count=self.trainer.update_count,
            decision_q_rng_state=self._decision_q_generator.get_state().clone(),
            decision_mode_rng_state=self._decision_mode_generator.get_state().clone(),
            replay_rng_state=self._replay_generator.get_state().clone(),
            trainer_target_rng_state=self.trainer._target_generator.get_state().clone(),
            trainer_actor_rng_state=self.trainer._actor_generator.get_state().clone(),
            current_state_sha256=self.environment.current_state.state.canonical_sha256(),
            exploration_decision_count=self._ledger.decision_count,
            replay_accepted_count=self.replay_buffer.accepted_count,
            replay_evicted_count=self.replay_buffer.evicted_count,
            replay_seen_digest_count=self.replay_buffer.seen_digest_count,
            replay_seen_identity_count=self.replay_buffer.seen_identity_count,
            replay_resident_transition_digests=(
                self.replay_buffer.resident_transition_digests()
            ),
        )

    def reissue_portable_journal(
        self,
        *,
        runner_binding_sha256: str,
        session_uuid: str,
        ue_id: str,
        genesis_staged: production_state_provider.StagedDecisionInputsV1,
        initial_kernel_checkpoint_sha256: str,
        rows: Tuple[PortableJournalReplayRowV1, ...],
    ) -> Tuple[RunnerStepJournalV1, ...]:
        """Rebuild durable rows through the normal causal execution path.

        This is the public process-boundary seam used by checkpoint I/O. It
        accepts no transition object and no attestation token. Every returned
        transition is reconstructed by :meth:`_execute_prebuilt`, re-attested
        by :func:`run4_contract.build_transition`, inserted into replay, and
        compared with all three durable digest targets.

        The receiver must be a pristine runner created by the caller's bound
        factory. Any contradiction permanently faults this disposable
        receiver; a partially replayed runner is never returned as usable.
        """

        _digest(runner_binding_sha256, "runner_binding_sha256")
        _digest(
            initial_kernel_checkpoint_sha256,
            "initial_kernel_checkpoint_sha256",
        )
        if runner_binding_sha256 != self.runner_binding_sha256:
            raise RunnerCheckpointError("portable replay/factory binding mismatch")
        if self.started or self.decision_count or len(self.replay_buffer):
            raise RunnerCheckpointError("portable replay receiver is not pristine")
        if self._faulted or self._adapter.pending:
            raise RunnerCheckpointError("portable replay receiver is not healthy")
        if type(genesis_staged) is not (
            production_state_provider.StagedDecisionInputsV1
        ):
            raise RunnerCheckpointError("portable replay genesis is foreign")
        try:
            parsed_session = str(uuid.UUID(session_uuid))
        except (ValueError, TypeError, AttributeError) as exc:
            raise RunnerCheckpointError("portable replay session UUID is invalid") from exc
        if parsed_session != session_uuid:
            raise RunnerCheckpointError("portable replay session UUID is not canonical")
        _text(ue_id, "ue_id")
        if type(rows) is not tuple or any(
            type(row) is not PortableJournalReplayRowV1 for row in rows
        ):
            raise RunnerCheckpointError(
                "portable replay rows must be an exact tuple of replay records"
            )
        if (
            genesis_staged.identity.session_uuid,
            genesis_staged.identity.ue_id,
            genesis_staged.identity.decision_seq,
        ) != (session_uuid, ue_id, 0):
            raise RunnerCheckpointError("portable replay genesis identity differs")
        initial = self.kernel.checkpoint()
        if initial.canonical_sha256 != initial_kernel_checkpoint_sha256:
            raise RunnerCheckpointError("portable replay initial kernel differs")
        if genesis_staged.radio_state.canonical_sha256 != (
            initial.current_state.canonical_sha256
        ):
            raise RunnerCheckpointError("portable replay genesis radio state differs")
        for index, row in enumerate(rows):
            identity = row.decision.identity
            if (
                identity.session_uuid,
                identity.ue_id,
                identity.decision_seq,
            ) != (session_uuid, ue_id, index):
                raise RunnerCheckpointError(
                    "portable replay decision sequence is not contiguous"
                )

        try:
            self.state_provider.stage_decision(genesis_staged)
            self.environment.reset(session_uuid=session_uuid, ue_id=ue_id)
            self._session_uuid = session_uuid
            self._ue_id = ue_id
            self._genesis_staged = genesis_staged
            for expected in rows:
                observed = self._execute_prebuilt(
                    decision=expected.decision,
                    prediction=expected.prediction,
                    successor_staged=expected.successor_staged,
                    record=True,
                )
                if observed.transition.canonical_sha256() != (
                    expected.expected_transition_sha256
                ):
                    raise RunnerCheckpointError(
                        "portable replay transition digest differs"
                    )
                if observed.environment_transition_sha256 != (
                    expected.expected_environment_transition_sha256
                ):
                    raise RunnerCheckpointError(
                        "portable replay environment digest differs"
                    )
                if observed.canonical_sha256 != expected.expected_journal_sha256:
                    raise RunnerCheckpointError(
                        "portable replay journal digest differs"
                    )
            return tuple(self._journal)
        except Exception:
            self._faulted = True
            raise

    @classmethod
    def restore(
        cls,
        checkpoint: PersistentRunnerCheckpointV1,
        *,
        fresh_factory: Callable[[], "_RunnerCore"],
    ) -> "_RunnerCore":
        """Restore atomically by replaying every causal join into fresh objects."""

        if type(checkpoint) is not PersistentRunnerCheckpointV1:
            raise RunnerCheckpointError("foreign checkpoint type")
        if checkpoint.compute_sha256() != checkpoint.checkpoint_sha256:
            raise RunnerCheckpointError("checkpoint changed after construction")
        candidate = fresh_factory()
        if type(candidate) is not cls:
            raise RunnerCheckpointError("factory returned a different runner type")
        if candidate.started or candidate.decision_count or len(candidate.replay_buffer):
            raise RunnerCheckpointError("factory runner is not pristine")
        if candidate.runner_binding_sha256 != checkpoint.runner_binding_sha256:
            raise RunnerCheckpointError("checkpoint/factory binding mismatch")
        if candidate.kernel.checkpoint().canonical_sha256 != (
            checkpoint.initial_kernel_checkpoint.canonical_sha256
        ):
            raise RunnerCheckpointError("fresh kernel differs from checkpoint genesis")

        # Until this point no supplied live runner was mutated.  Candidate is
        # private to this method and is returned only after every equality gate.
        candidate.state_provider.stage_decision(checkpoint.genesis_staged)
        candidate.environment.reset(
            session_uuid=checkpoint.session_uuid, ue_id=checkpoint.ue_id
        )
        candidate._session_uuid = checkpoint.session_uuid
        candidate._ue_id = checkpoint.ue_id
        candidate._genesis_staged = checkpoint.genesis_staged
        for expected in checkpoint.journal:
            observed = candidate._execute_prebuilt(
                decision=expected.decision,
                prediction=expected.prediction,
                successor_staged=expected.successor_staged,
                record=True,
            )
            if observed.canonical_sha256 != expected.canonical_sha256:
                raise RunnerCheckpointError("causal journal did not replay exactly")

        if candidate.kernel.checkpoint().canonical_sha256 != (
            checkpoint.kernel_checkpoint.canonical_sha256
        ):
            raise RunnerCheckpointError("kernel did not restore exactly")
        candidate.fit_provider.load_state_dict(checkpoint.fit_scene_state)
        candidate._current_draw = candidate.fit_provider.active_draw
        candidate.state_stager.load_state_dict(
            copy.deepcopy(checkpoint.state_stager_state)
        )
        candidate.prediction_provider.load_state_dict(
            copy.deepcopy(checkpoint.prediction_provider_state)
        )
        candidate.model_bundle.actor.load_state_dict(
            copy.deepcopy(checkpoint.actor_state_dict), strict=True
        )
        candidate.model_bundle.critics.load_state_dict(
            copy.deepcopy(checkpoint.critics_state_dict), strict=True
        )
        models.validate_run4_models(
            candidate.model_bundle.actor, candidate.model_bundle.critics
        )
        candidate.trainer.actor_optimizer.load_state_dict(
            copy.deepcopy(checkpoint.actor_optimizer_state_dict)
        )
        candidate.trainer.critic_optimizer.load_state_dict(
            copy.deepcopy(checkpoint.critic_optimizer_state_dict)
        )
        candidate.trainer.update_count = checkpoint.trainer_update_count
        candidate._decision_q_generator.set_state(
            checkpoint.decision_q_rng_state.clone()
        )
        candidate._decision_mode_generator.set_state(
            checkpoint.decision_mode_rng_state.clone()
        )
        candidate._replay_generator.set_state(checkpoint.replay_rng_state.clone())
        candidate.trainer._target_generator.set_state(
            checkpoint.trainer_target_rng_state.clone()
        )
        candidate.trainer._actor_generator.set_state(
            checkpoint.trainer_actor_rng_state.clone()
        )
        if candidate.environment.current_state.state.canonical_sha256() != (
            checkpoint.current_state_sha256
        ):
            raise RunnerCheckpointError("current state did not restore exactly")
        if candidate._ledger.decision_count != checkpoint.exploration_decision_count:
            raise RunnerCheckpointError("exploration ledger did not restore")
        if candidate.replay_buffer.accepted_count != checkpoint.replay_accepted_count:
            raise RunnerCheckpointError("replay buffer did not restore")
        replay_observed = (
            candidate.replay_buffer.evicted_count,
            candidate.replay_buffer.seen_digest_count,
            candidate.replay_buffer.seen_identity_count,
            candidate.replay_buffer.resident_transition_digests(),
        )
        replay_expected = (
            checkpoint.replay_evicted_count,
            checkpoint.replay_seen_digest_count,
            checkpoint.replay_seen_identity_count,
            checkpoint.replay_resident_transition_digests,
        )
        if replay_observed != replay_expected:
            raise RunnerCheckpointError("replay state did not restore exactly")
        if candidate.checkpoint().checkpoint_sha256 != checkpoint.checkpoint_sha256:
            raise RunnerCheckpointError("restored checkpoint fingerprint differs")
        return candidate


class Run4PersistentTrainingRunnerV1(_RunnerCore):
    """Production entry point, deliberately unreachable before verification."""

    def __init__(self, **kwargs: Any) -> None:
        authorization = kwargs.get("authorization")
        if type(authorization) is not RunnerAuthorizationV1:
            raise RunnerAuthorizationError("production authorization is required")
        authorization.require_training_eligible()
        if type(kwargs.get("replay_buffer")) is not replay.ReplayBufferV1:
            raise RunnerAuthorizationError("production replay buffer is required")
        if type(kwargs.get("sac_trainer")) is not trainer.Run4HybridSacTrainerV1:
            raise RunnerAuthorizationError("production trainer is required")
        if not kwargs["state_provider"].replay_export_allowed:
            raise RunnerAuthorizationError("state provider is not replay eligible")
        if not kwargs["kernel"].replay_export_allowed:
            raise RunnerAuthorizationError("kernel is not replay eligible")
        # Even a future token must not silently bypass the existing
        # environment's explicit calibrated-export stop gate.
        raise RunnerAuthorizationError(
            "calibrated transition export awaits composite-verifier integration"
        )

    def _make_ledger(self) -> exploration.ExplorationCoverageLedger:
        return exploration.ExplorationCoverageLedger(
            self.warmup_schedule, self.coverage_gate
        )

    def _record_coverage(self, transition: contract.SemiMarkovTransitionV2) -> None:
        self._ledger.record_transition(transition)

    def _require_training_authority(self) -> None:
        self.authorization.require_training_eligible()

    def _require_actor_action_ready(self) -> None:
        self._require_training_authority()
        self._ledger.require_gradient_start()


class _TestOnlyPersistentRunnerV1(_RunnerCore):
    """Private mechanics harness; never replay/training eligible."""

    def __init__(self, **kwargs: Any) -> None:
        authorization = kwargs.get("authorization")
        if type(authorization) is not RunnerAuthorizationV1:
            raise RunnerAuthorizationError("test authorization is required")
        authorization.require_attested()
        if authorization.authorization_class is not RunnerAuthorizationClass.TEST_ONLY_MECHANICS:
            raise RunnerAuthorizationError("test runner received production authority")
        if type(kwargs.get("replay_buffer")) is not replay._TestOnlyReplayBufferV1:
            raise RunnerAuthorizationError("test-only replay is required")
        if type(kwargs.get("sac_trainer")) is not trainer._TestOnlyRun4HybridSacTrainerV1:
            raise RunnerAuthorizationError("test-only trainer is required")
        if kwargs["state_provider"].replay_export_allowed:
            raise RunnerAuthorizationError("test state provider became replay eligible")
        if kwargs["kernel"].replay_export_allowed:
            raise RunnerAuthorizationError("test kernel became replay eligible")
        super().__init__(**kwargs)

    def _make_ledger(self) -> exploration.ExplorationCoverageLedger:
        return exploration.ExplorationCoverageLedger.for_test_only(
            self.warmup_schedule, self.coverage_gate
        )

    def _record_coverage(self, transition: contract.SemiMarkovTransitionV2) -> None:
        expected = self.warmup_schedule.action_at(self._ledger.decision_count)
        if (transition.action.mode_id, transition.action.q_e4) != (
            expected.mode_id,
            expected.q_e4,
        ):
            raise RunnerStateError("executed warm-up action differs from schedule")
        identity = transition.state.state.identity
        self._ledger.record_test_only_decision(
            decision_identity=canonical_sha256(identity.to_canonical_dict()),
            action=expected,
            observation=self._coverage_observation(transition.state),
        )
        if self._ledger.decision_count == len(self.warmup_schedule):
            assert transition.next_state is not None
            self._ledger.record_test_only_final_feedback_state(
                self._coverage_observation(transition.next_state)
            )

    def _require_training_authority(self) -> None:
        raise RunnerTrainingUnavailable(
            "TEST_ONLY_MECHANICS can exercise persistence but never gradients"
        )

    def _require_actor_action_ready(self) -> None:
        if self._ledger.decision_count != len(self.warmup_schedule):
            raise RunnerTrainingUnavailable(
                "test actor mechanics require a completed warm-up sequence"
            )


__all__ = [
    "CausalStateStager",
    "CompositeRunnerPrerequisitesV1",
    "EmpiricalPredictionProvider",
    "PersistentRunnerCheckpointV1",
    "PortableJournalReplayRowV1",
    "Run4PersistentTrainingRunnerV1",
    "RunnerAuthorizationError",
    "RunnerBindingError",
    "RunnerCheckpointError",
    "RunnerStateError",
    "RunnerStepJournalV1",
    "RunnerTrainingUnavailable",
    "verify_composite_prerequisites",
]
