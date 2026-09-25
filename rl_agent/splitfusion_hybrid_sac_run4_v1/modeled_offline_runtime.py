"""Dedicated offline replay and trainer for MODELED_COMPOSITE evidence.

This module is an intentionally separate learning boundary.  It never
relabels modeled evidence as empirical evidence, never constructs a production
ReplayBufferV1, and never launches CARLA, OAI, CUDA, a container, or a network
service.  Only an exact, attested ModeledCompositeOfflineTransitionV1 can
enter the buffer.

The numerical SAC update is inherited unchanged from the tested Run-4 trainer
core.  All evidence admission, binding, tensorization, and preflight checks are
modeled-specific and remain disjoint from the production empirical path.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields
from itertools import chain
from types import MappingProxyType
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    CATALOG_SHA256,
    EXPECTED_MODE_COUNT,
    Q_E4_MAX,
    Q_E4_MIN,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    MINIMUM_HOLD_TENSORS,
    canonical_sha256,
)

from . import modeled_composite_training as modeled
from . import mcs_transition_acceptance
from . import run4_contract as contract
from . import trainer as run4_trainer
from .models import Run4ModelBundleV1, build_run4_models, validate_run4_models
from .replay import (
    BindingMismatchError,
    DEFAULT_FLOAT_DTYPE,
    DuplicateTransitionError,
    IdentityConflictError,
    NonFiniteInReplayDtypeError,
    Q_CRITIC_NORMALIZER,
    ReplayBufferError,
    ReplaySamplingError,
    TransitionRejectedError,
)

__all__ = [
    "MODELED_OFFLINE_REPLAY_SCHEMA_ID",
    "MODELED_OFFLINE_RUNNER_SCHEMA_ID",
    "ModeledCompositeHybridSacTrainerV1",
    "ModeledCompositeOfflineFactoryV1",
    "ModeledCompositeOfflineRunnerV1",
    "ModeledCompositeReplayBufferV1",
    "ModeledReplayBindingV1",
    "ModeledReplayTensorBatchV1",
]


MODELED_OFFLINE_REPLAY_SCHEMA_ID = (
    "splitfusion.run4.modeled_composite.offline_replay.v1"
)
MODELED_OFFLINE_RUNNER_SCHEMA_ID = (
    "splitfusion.run4.modeled_composite.offline_runner.v1"
)
MODELED_OFFLINE_ELIGIBILITY = "OFFLINE_MODELED_TRAINING_ONLY"
MODELED_EVIDENCE_CLASS = modeled.EVIDENCE_CLASS.value


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ReplayBufferError(
            f"{name} must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayBufferError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayBufferError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ReplayBufferError(f"{name} must be a positive exact int")
    return value


def _assert_finite_float32(values: Sequence[float], name: str) -> None:
    converted = torch.tensor(tuple(values), dtype=DEFAULT_FLOAT_DTYPE)
    finite = torch.isfinite(converted)
    if bool(finite.all()):
        return
    index = int((~finite).nonzero()[0])
    raise NonFiniteInReplayDtypeError(
        f"{name}[{index}] is unsafe after conversion to torch.float32"
    )


_MCS_ACCEPTANCE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _McsAcceptanceAttestationV1:
    result_sha256: str
    model_binding_sha256: str
    source_evidence_sha256: str
    evidence_class: str
    _token: object

    @classmethod
    def load_registered(cls) -> "_McsAcceptanceAttestationV1":
        report = mcs_transition_acceptance.load_registered_acceptance()
        if report.get("accepted_for_offline_run4_mcs_dynamics") is not True:
            raise ReplayBufferError("registered MCS dynamics were not accepted")
        return cls(
            result_sha256=(
                mcs_transition_acceptance
                .REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256
            ),
            model_binding_sha256=_sha256(
                report.get("model_binding_sha256"),
                "MCS model_binding_sha256",
            ),
            source_evidence_sha256=_sha256(
                report.get("source_evidence_sha256"),
                "MCS source_evidence_sha256",
            ),
            evidence_class=str(report.get("evidence_class")),
            _token=_MCS_ACCEPTANCE_TOKEN,
        )

    def require_registered(self) -> None:
        if self._token is not _MCS_ACCEPTANCE_TOKEN:
            raise BindingMismatchError("MCS acceptance attestation is forged")
        if self.result_sha256 != (
            mcs_transition_acceptance.REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256
        ):
            raise BindingMismatchError("MCS acceptance result digest changed")
        _sha256(self.model_binding_sha256, "MCS model binding")
        _sha256(self.source_evidence_sha256, "MCS source evidence")
        if self.evidence_class != mcs_transition_acceptance.EVIDENCE_CLASS:
            raise BindingMismatchError("MCS acceptance evidence class changed")


@dataclass(frozen=True, slots=True)
class ModeledReplayBindingV1:
    """Exact offline modeled learning problem; never an empirical binding."""

    schema_id: str
    schema_version: int
    schema_sha256: str
    feature_schema_sha256: str
    reward_schema_sha256: str
    transition_schema_sha256: str
    catalog_sha256: str
    policy_feature_order: Tuple[str, ...]
    policy_feature_count: int
    gamma: float
    freshness_policy_sha256: str
    empirical_scaling_sha256: str
    modeled_composite_binding_sha256: str
    provider_implementation_sha256: str
    verifier_manifest_sha256: str
    mcs_acceptance_result_sha256: str
    mcs_model_binding_sha256: str
    mcs_source_evidence_sha256: str
    mcs_acceptance_evidence_class: str
    training_evidence_class: str
    evidence_eligibility: str
    offline_training_only: bool
    measured_runtime_evidence: bool
    calibrated_empirical_evidence: bool
    production_authorized: bool
    deployment_claim_allowed: bool
    _modeled_binding: modeled.ModeledCompositeBindingV1
    _mcs_acceptance: _McsAcceptanceAttestationV1

    @classmethod
    def from_modeled_composite(
        cls,
        *,
        modeled_binding: modeled.ModeledCompositeBindingV1,
        gamma: float,
        freshness_policy_sha256: str,
        empirical_scaling_sha256: str,
    ) -> "ModeledReplayBindingV1":
        if type(modeled_binding) is not modeled.ModeledCompositeBindingV1:
            raise ReplayBufferError(
                "modeled_binding must be exact ModeledCompositeBindingV1"
            )
        modeled_binding.__post_init__()
        mcs_acceptance = _McsAcceptanceAttestationV1.load_registered()
        mcs_disclosures = tuple(
            item
            for item in modeled_binding.component_disclosures
            if item.role is modeled.ComponentRole.UL_MCS_TRANSITION
        )
        if len(mcs_disclosures) != 1:
            raise BindingMismatchError(
                "modeled binding lacks one UL_MCS_TRANSITION disclosure"
            )
        mcs_disclosure = mcs_disclosures[0]
        if (
            mcs_disclosure.source_evidence_sha256
            != mcs_acceptance.source_evidence_sha256
            or mcs_disclosure.fit_support_sha256
            != mcs_acceptance.model_binding_sha256
        ):
            raise BindingMismatchError(
                "UL_MCS_TRANSITION disclosure is not the accepted source/model"
            )
        result = cls(
            schema_id=contract.SCHEMA_ID,
            schema_version=contract.SCHEMA_VERSION,
            schema_sha256=contract.SCHEMA_SHA256,
            feature_schema_sha256=contract.FEATURE_SCHEMA_SHA256,
            reward_schema_sha256=contract.REWARD_SCHEMA_SHA256,
            transition_schema_sha256=contract.TRANSITION_SCHEMA_SHA256,
            catalog_sha256=CATALOG_SHA256,
            policy_feature_order=tuple(contract.POLICY_FEATURE_ORDER),
            policy_feature_count=contract.POLICY_FEATURE_COUNT,
            gamma=float(gamma),
            freshness_policy_sha256=freshness_policy_sha256,
            empirical_scaling_sha256=empirical_scaling_sha256,
            modeled_composite_binding_sha256=modeled_binding.canonical_sha256,
            provider_implementation_sha256=(
                modeled_binding.provider_implementation_sha256
            ),
            verifier_manifest_sha256=modeled_binding.verifier_manifest_sha256,
            mcs_acceptance_result_sha256=mcs_acceptance.result_sha256,
            mcs_model_binding_sha256=mcs_acceptance.model_binding_sha256,
            mcs_source_evidence_sha256=(
                mcs_acceptance.source_evidence_sha256
            ),
            mcs_acceptance_evidence_class=mcs_acceptance.evidence_class,
            training_evidence_class=MODELED_EVIDENCE_CLASS,
            evidence_eligibility=MODELED_OFFLINE_ELIGIBILITY,
            offline_training_only=True,
            measured_runtime_evidence=False,
            calibrated_empirical_evidence=False,
            production_authorized=False,
            deployment_claim_allowed=False,
            _modeled_binding=modeled_binding,
            _mcs_acceptance=mcs_acceptance,
        )
        result.revalidate()
        return result

    def revalidate(self) -> None:
        expected = {
            "schema_id": contract.SCHEMA_ID,
            "schema_version": contract.SCHEMA_VERSION,
            "schema_sha256": contract.SCHEMA_SHA256,
            "feature_schema_sha256": contract.FEATURE_SCHEMA_SHA256,
            "reward_schema_sha256": contract.REWARD_SCHEMA_SHA256,
            "transition_schema_sha256": contract.TRANSITION_SCHEMA_SHA256,
            "catalog_sha256": CATALOG_SHA256,
            "policy_feature_order": tuple(contract.POLICY_FEATURE_ORDER),
            "policy_feature_count": contract.POLICY_FEATURE_COUNT,
            "training_evidence_class": MODELED_EVIDENCE_CLASS,
            "evidence_eligibility": MODELED_OFFLINE_ELIGIBILITY,
            "offline_training_only": True,
            "measured_runtime_evidence": False,
            "calibrated_empirical_evidence": False,
            "production_authorized": False,
            "deployment_claim_allowed": False,
        }
        for name, wanted in expected.items():
            if getattr(self, name) != wanted:
                raise BindingMismatchError(
                    f"modeled replay requires {name}={wanted!r}"
                )
        gamma = _finite_float(self.gamma, "gamma")
        if not 0.0 < gamma <= 1.0:
            raise ReplayBufferError("gamma must lie in (0, 1]")
        for name in (
            "freshness_policy_sha256",
            "empirical_scaling_sha256",
            "modeled_composite_binding_sha256",
            "provider_implementation_sha256",
            "verifier_manifest_sha256",
            "mcs_acceptance_result_sha256",
            "mcs_model_binding_sha256",
            "mcs_source_evidence_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.mcs_acceptance_evidence_class != (
            mcs_transition_acceptance.EVIDENCE_CLASS
        ):
            raise BindingMismatchError("MCS acceptance evidence class differs")
        if type(self._modeled_binding) is not modeled.ModeledCompositeBindingV1:
            raise BindingMismatchError(
                "binding lacks exact ModeledCompositeBindingV1 authority"
            )
        self._modeled_binding.__post_init__()
        if (
            self._modeled_binding.canonical_sha256
            != self.modeled_composite_binding_sha256
        ):
            raise BindingMismatchError("modeled binding digest changed")
        if (
            self._modeled_binding.provider_implementation_sha256
            != self.provider_implementation_sha256
            or self._modeled_binding.verifier_manifest_sha256
            != self.verifier_manifest_sha256
        ):
            raise BindingMismatchError(
                "modeled provider/verifier provenance changed"
            )
        if type(self._mcs_acceptance) is not _McsAcceptanceAttestationV1:
            raise BindingMismatchError("MCS acceptance attestation is absent")
        self._mcs_acceptance.require_registered()
        if (
            self.mcs_acceptance_result_sha256
            != self._mcs_acceptance.result_sha256
            or self.mcs_model_binding_sha256
            != self._mcs_acceptance.model_binding_sha256
            or self.mcs_source_evidence_sha256
            != self._mcs_acceptance.source_evidence_sha256
            or self.mcs_acceptance_evidence_class
            != self._mcs_acceptance.evidence_class
        ):
            raise BindingMismatchError("MCS acceptance provenance changed")
        mcs_disclosures = tuple(
            item
            for item in self._modeled_binding.component_disclosures
            if item.role is modeled.ComponentRole.UL_MCS_TRANSITION
        )
        if len(mcs_disclosures) != 1:
            raise BindingMismatchError("modeled MCS disclosure disappeared")
        mcs_disclosure = mcs_disclosures[0]
        if (
            mcs_disclosure.source_evidence_sha256
            != self.mcs_source_evidence_sha256
            or mcs_disclosure.fit_support_sha256
            != self.mcs_model_binding_sha256
        ):
            raise BindingMismatchError(
                "modeled MCS disclosure changed from the accepted model"
            )

    def require_offline_modeled_training(self) -> None:
        self.revalidate()
        if self.evidence_eligibility != MODELED_OFFLINE_ELIGIBILITY:
            raise BindingMismatchError(
                "modeled trainer requires OFFLINE_MODELED_TRAINING_ONLY"
            )

    def assert_exactly(self, other: "ModeledReplayBindingV1") -> None:
        if type(other) is not ModeledReplayBindingV1:
            raise BindingMismatchError(
                f"expected exact ModeledReplayBindingV1, got {type(other).__name__}"
            )
        self.revalidate()
        other.revalidate()
        for item in fields(self):
            mine = getattr(self, item.name)
            theirs = getattr(other, item.name)
            if mine != theirs:
                raise BindingMismatchError(
                    f"modeled replay binding differs at {item.name}"
                )

    def assert_wrapper(
        self, wrapper: modeled.ModeledCompositeOfflineTransitionV1
    ) -> None:
        self.revalidate()
        if type(wrapper) is not modeled.ModeledCompositeOfflineTransitionV1:
            raise TransitionRejectedError(
                "modeled replay accepts only exact "
                "ModeledCompositeOfflineTransitionV1"
            )
        try:
            wrapper.require_attested()
        except Exception as exc:
            raise TransitionRejectedError(
                "modeled offline wrapper is absent, forged, or stale"
            ) from exc
        if wrapper.evidence_class is not modeled.EVIDENCE_CLASS:
            raise BindingMismatchError("wrapper evidence class differs")
        comparisons = {
            "modeled_composite_binding_sha256": wrapper.modeled_binding_sha256,
            "gamma": float(wrapper.gamma),
            "freshness_policy_sha256": wrapper.freshness_policy_sha256,
            "empirical_scaling_sha256": wrapper.empirical_scaling_sha256,
        }
        for name, actual in comparisons.items():
            if getattr(self, name) != actual:
                raise BindingMismatchError(
                    f"modeled wrapper differs at {name}"
                )

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(
            {
                "record_type": "modeled_replay_binding_v1",
                "payload": self.to_canonical_dict(),
            }
        )

    def to_canonical_dict(self) -> Dict[str, Any]:
        self.revalidate()
        return {
            "calibrated_empirical_evidence": self.calibrated_empirical_evidence,
            "catalog_sha256": self.catalog_sha256,
            "deployment_claim_allowed": self.deployment_claim_allowed,
            "empirical_scaling_sha256": self.empirical_scaling_sha256,
            "evidence_eligibility": self.evidence_eligibility,
            "feature_schema_sha256": self.feature_schema_sha256,
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "gamma": self.gamma,
            "measured_runtime_evidence": self.measured_runtime_evidence,
            "mcs_acceptance_evidence_class": (
                self.mcs_acceptance_evidence_class
            ),
            "mcs_acceptance_result_sha256": (
                self.mcs_acceptance_result_sha256
            ),
            "mcs_model_binding_sha256": self.mcs_model_binding_sha256,
            "mcs_source_evidence_sha256": self.mcs_source_evidence_sha256,
            "modeled_composite_binding": self._modeled_binding.to_dict(),
            "modeled_composite_binding_sha256": (
                self.modeled_composite_binding_sha256
            ),
            "offline_training_only": self.offline_training_only,
            "policy_feature_count": self.policy_feature_count,
            "policy_feature_order": list(self.policy_feature_order),
            "production_authorized": self.production_authorized,
            "provider_implementation_sha256": (
                self.provider_implementation_sha256
            ),
            "reward_schema_sha256": self.reward_schema_sha256,
            "schema_id": self.schema_id,
            "schema_sha256": self.schema_sha256,
            "schema_version": self.schema_version,
            "training_evidence_class": self.training_evidence_class,
            "transition_schema_sha256": self.transition_schema_sha256,
            "verifier_manifest_sha256": self.verifier_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class _ModeledStoredRow:
    transition_sha256: str
    wrapper_audit_sha256: str
    logical_key: Tuple[str, str, int]
    episode_boundary: contract.EpisodeBoundary
    executed_action_sha256: str
    state_values: Tuple[float, ...]
    next_state_values: Optional[Tuple[float, ...]]
    mode_id: int
    q_e4: int
    reward: float
    duration: int
    discount: float
    modeled_binding_sha256: str
    source_envelope_sha256: str
    support_use_sha256: str
    latency_projection_sha256: str

    @property
    def has_next_state(self) -> bool:
        return self.next_state_values is not None

    @property
    def bootstrap(self) -> bool:
        return self.has_next_state and (
            self.episode_boundary is contract.EpisodeBoundary.CONTINUES
        )

    @property
    def terminated(self) -> bool:
        return self.episode_boundary is contract.EpisodeBoundary.TERMINATED

    @property
    def truncated(self) -> bool:
        return self.episode_boundary is contract.EpisodeBoundary.TRUNCATED

    def audit_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "decision_seq": self.logical_key[2],
                "discount": self.discount,
                "episode_boundary": self.episode_boundary.value,
                "evidence_class": MODELED_EVIDENCE_CLASS,
                "executed_action_sha256": self.executed_action_sha256,
                "latency_projection_sha256": self.latency_projection_sha256,
                "modeled_binding_sha256": self.modeled_binding_sha256,
                "offline_training_only": True,
                "production_authorized": False,
                "session_uuid": self.logical_key[0],
                "source_envelope_sha256": self.source_envelope_sha256,
                "support_use_sha256": self.support_use_sha256,
                "transition_sha256": self.transition_sha256,
                "ue_id": self.logical_key[1],
                "wrapper_audit_sha256": self.wrapper_audit_sha256,
            }
        )


@dataclass(frozen=True, slots=True, eq=False)
class ModeledReplayTensorBatchV1:
    """Modeled-only tensor batch with immutable provenance and clone accessors."""

    _state: Tensor
    _next_state: Tensor
    _mode_id: Tensor
    _q_e4: Tensor
    _reward: Tensor
    _duration: Tensor
    _discount: Tensor
    _has_next_state: Tensor
    _bootstrap: Tensor
    _terminated: Tensor
    _truncated: Tensor
    _binding: ModeledReplayBindingV1
    _audit: Tuple[Mapping[str, Any], ...]

    _FLOAT_FIELDS = ("_state", "_next_state", "_reward", "_discount")
    _INT_FIELDS = ("_mode_id", "_q_e4", "_duration")
    _BOOL_FIELDS = (
        "_has_next_state",
        "_bootstrap",
        "_terminated",
        "_truncated",
    )
    _TENSOR_FIELDS = _FLOAT_FIELDS + _INT_FIELDS + _BOOL_FIELDS

    def __post_init__(self) -> None:
        if type(self._binding) is not ModeledReplayBindingV1:
            raise ReplayBufferError(
                "modeled batch binding must be exact ModeledReplayBindingV1"
            )
        self._binding.revalidate()
        for name in self._TENSOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                raise ReplayBufferError(f"{name} must be a torch.Tensor")
            if value.device.type != "cpu":
                raise ReplayBufferError(f"{name} must remain on CPU")
            object.__setattr__(self, name, value.detach().clone())
        for name in self._FLOAT_FIELDS:
            if getattr(self, name).dtype is not DEFAULT_FLOAT_DTYPE:
                raise ReplayBufferError(f"{name} must use torch.float32")
        for name in self._INT_FIELDS:
            if getattr(self, name).dtype is not torch.int64:
                raise ReplayBufferError(f"{name} must use torch.int64")
        for name in self._BOOL_FIELDS:
            if getattr(self, name).dtype is not torch.bool:
                raise ReplayBufferError(f"{name} must use torch.bool")
        if self._state.ndim != 2 or (
            self._state.shape[1] != contract.POLICY_FEATURE_COUNT
        ):
            raise ReplayBufferError(
                f"state must have shape [B, {contract.POLICY_FEATURE_COUNT}]"
            )
        if self._next_state.shape != self._state.shape:
            raise ReplayBufferError("next_state shape differs from state")
        size = int(self._state.shape[0])
        for name in self._TENSOR_FIELDS[2:]:
            if getattr(self, name).shape != (size,):
                raise ReplayBufferError(f"{name} must have shape [B]")
        if len(self._audit) != size:
            raise ReplayBufferError("audit row count differs from batch size")
        for row in self._audit:
            if row.get("evidence_class") != MODELED_EVIDENCE_CLASS:
                raise ReplayBufferError("batch audit lost modeled evidence class")
            if row.get("modeled_binding_sha256") != (
                self._binding.modeled_composite_binding_sha256
            ):
                raise ReplayBufferError("batch audit lost modeled binding")
            if row.get("production_authorized") is not False:
                raise ReplayBufferError("modeled batch cannot be production-authorized")
        object.__setattr__(
            self,
            "_audit",
            tuple(MappingProxyType(dict(item)) for item in self._audit),
        )

    @property
    def state(self) -> Tensor:
        return self._state.clone()

    @property
    def next_state(self) -> Tensor:
        return self._next_state.clone()

    @property
    def mode_id(self) -> Tensor:
        return self._mode_id.clone()

    @property
    def q_e4(self) -> Tensor:
        return self._q_e4.clone()

    @property
    def q_normalized_executed(self) -> Tensor:
        return self._q_e4.to(DEFAULT_FLOAT_DTYPE) / float(Q_CRITIC_NORMALIZER)

    @property
    def reward(self) -> Tensor:
        return self._reward.clone()

    @property
    def duration(self) -> Tensor:
        return self._duration.clone()

    @property
    def has_next_state(self) -> Tensor:
        return self._has_next_state.clone()

    @property
    def bootstrap(self) -> Tensor:
        return self._bootstrap.clone()

    @property
    def terminated(self) -> Tensor:
        return self._terminated.clone()

    @property
    def truncated(self) -> Tensor:
        return self._truncated.clone()

    @property
    def batch_size(self) -> int:
        return int(self._state.shape[0])

    @property
    def float_dtype(self) -> torch.dtype:
        return DEFAULT_FLOAT_DTYPE

    @property
    def gamma(self) -> float:
        return self._binding.gamma

    @property
    def binding(self) -> ModeledReplayBindingV1:
        self._binding.revalidate()
        return self._binding

    @property
    def audit(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(dict(item)) for item in self._audit)

    def discount(self) -> Tensor:
        return self._discount.clone()

    def to_canonical_metadata(self) -> Dict[str, Any]:
        return {
            "audit": [dict(item) for item in self._audit],
            "batch_size": self.batch_size,
            "binding": self.binding.to_canonical_dict(),
            "float_dtype": str(DEFAULT_FLOAT_DTYPE),
            "schema": MODELED_OFFLINE_REPLAY_SCHEMA_ID,
        }


class ModeledCompositeReplayBufferV1:
    """Bounded FIFO for exact attested modeled wrappers only."""

    def __init__(
        self, capacity: int, binding: ModeledReplayBindingV1
    ) -> None:
        _positive_int(capacity, "capacity")
        if type(binding) is not ModeledReplayBindingV1:
            raise ReplayBufferError(
                "binding must be exact ModeledReplayBindingV1"
            )
        binding.require_offline_modeled_training()
        contract.assert_policy_feature_schema()
        self._capacity = capacity
        self._binding = binding
        self._rows: Deque[_ModeledStoredRow] = deque()
        self._seen_digests: set[str] = set()
        self._seen_identities: Dict[Tuple[str, str, int], str] = {}
        self._accepted_count = 0
        self._evicted_count = 0

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def binding(self) -> ModeledReplayBindingV1:
        self._binding.revalidate()
        return self._binding

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    @property
    def evicted_count(self) -> int:
        return self._evicted_count

    @property
    def seen_digest_count(self) -> int:
        return len(self._seen_digests)

    @property
    def seen_identity_count(self) -> int:
        return len(self._seen_identities)

    def resident_transition_digests(self) -> Tuple[str, ...]:
        return tuple(row.transition_sha256 for row in self._rows)

    def insert(self, wrapper: Any) -> None:
        row = self._validate(wrapper)
        self._seen_digests.add(row.transition_sha256)
        self._seen_identities[row.logical_key] = row.transition_sha256
        self._rows.append(row)
        self._accepted_count += 1
        while len(self._rows) > self._capacity:
            self._rows.popleft()
            self._evicted_count += 1

    def _validate(self, wrapper: Any) -> _ModeledStoredRow:
        self._binding.assert_wrapper(wrapper)
        assert type(wrapper) is modeled.ModeledCompositeOfflineTransitionV1
        try:
            transition = wrapper._sealed_transition_for_modeled_replay(
                self._binding.modeled_composite_binding_sha256
            )
        except Exception as exc:
            raise TransitionRejectedError(
                "sealed modeled transition hand-off failed"
            ) from exc
        if type(transition) is not contract.SemiMarkovTransitionV2:
            raise TransitionRejectedError(
                "sealed payload must be exact SemiMarkovTransitionV2"
            )
        transition.require_attested()

        transition.state.require_guarded()
        transition.state_features.require_attested()
        if transition.state_features.guarded_state_sha256 != (
            transition.state.canonical_sha256()
        ):
            raise TransitionRejectedError("state features are not bound to state")
        state_values = transition.state_features.as_tuple()
        self._check_features(state_values, "state_features")

        boundary = transition.episode_boundary
        if boundary is contract.EpisodeBoundary.CONTINUES:
            if (
                transition.next_state is None
                or transition.next_state_features is None
            ):
                raise TransitionRejectedError(
                    "CONTINUES requires the real successor and features"
                )
            transition.next_state.require_guarded()
            transition.next_state_features.require_attested()
            if transition.next_state_features.guarded_state_sha256 != (
                transition.next_state.canonical_sha256()
            ):
                raise TransitionRejectedError(
                    "successor features are not bound to successor"
                )
            if transition.next_state.freshness_policy_sha256 != (
                self._binding.freshness_policy_sha256
            ):
                raise BindingMismatchError(
                    "successor freshness-policy hash differs"
                )
            if transition.next_state_features.empirical_scaling_sha256 != (
                self._binding.empirical_scaling_sha256
            ):
                raise BindingMismatchError(
                    "successor empirical-scaling hash differs"
                )
            expected_previous = contract.PreviousOutcomeV1.from_resolution(
                transition.reward_resolution
            )
            actual_previous = transition.next_state.state.previous
            if (
                actual_previous is None
                or actual_previous.canonical_sha256()
                != expected_previous.canonical_sha256()
            ):
                raise TransitionRejectedError(
                    "successor previous outcome differs from current reward"
                )
            next_values: Optional[Tuple[float, ...]] = (
                transition.next_state_features.as_tuple()
            )
            self._check_features(next_values, "next_state_features")
        elif boundary in (
            contract.EpisodeBoundary.TERMINATED,
            contract.EpisodeBoundary.TRUNCATED,
        ):
            if (
                transition.next_state is not None
                or transition.next_state_features is not None
            ):
                raise TransitionRejectedError(
                    "terminal/truncated row cannot fabricate a successor"
                )
            next_values = None
        else:
            raise TransitionRejectedError("unknown episode boundary")

        action = transition.action
        action.require_reconciled()
        mode_id = int(action.mode_id)
        q_e4 = int(action.q_e4)
        if not 0 <= mode_id < EXPECTED_MODE_COUNT:
            raise TransitionRejectedError("mode_id is outside the catalog")
        if not Q_E4_MIN <= q_e4 <= Q_E4_MAX:
            raise TransitionRejectedError("q_e4 is outside execution range")

        reward = float(transition.reward)
        if not math.isfinite(reward):
            raise TransitionRejectedError("reward must be finite")
        duration = int(transition.duration)
        if (
            duration < MINIMUM_HOLD_TENSORS
            or duration != transition.hold.duration
        ):
            raise TransitionRejectedError(
                "duration differs from the exact action hold"
            )
        discount = float(transition.discount)
        if not math.isfinite(discount) or not 0.0 < discount <= 1.0:
            raise TransitionRejectedError("discount must lie in (0, 1]")

        _assert_finite_float32(state_values, "state_features")
        if next_values is not None:
            _assert_finite_float32(next_values, "next_state_features")
        _assert_finite_float32((reward,), "reward")
        _assert_finite_float32((discount,), "discount")
        _assert_finite_float32((float(transition.gamma),), "gamma")
        _assert_finite_float32(
            (q_e4 / float(Q_CRITIC_NORMALIZER),),
            "q_normalized_executed",
        )
        if (
            next_values is not None
            and discount > 0.0
            and float(torch.tensor(discount, dtype=DEFAULT_FLOAT_DTYPE)) == 0.0
        ):
            raise NonFiniteInReplayDtypeError(
                "bootstrap-eligible discount underflows to zero"
            )

        digest = transition.canonical_sha256()
        if digest != wrapper.transition_sha256:
            raise TransitionRejectedError("wrapper transition digest changed")
        identity = transition.state.state.identity
        logical_key = (
            identity.session_uuid,
            identity.ue_id,
            identity.decision_seq,
        )
        known = self._seen_identities.get(logical_key)
        if known is not None and known != digest:
            raise IdentityConflictError(
                f"logical decision {logical_key!r} already has another digest"
            )
        if digest in self._seen_digests:
            raise DuplicateTransitionError(
                f"transition {digest} was already seen"
            )

        wrapper_audit = wrapper.to_audit_dict()
        return _ModeledStoredRow(
            transition_sha256=digest,
            wrapper_audit_sha256=canonical_sha256(
                {
                    "record_type": "modeled_offline_wrapper_audit_v1",
                    "payload": wrapper_audit,
                }
            ),
            logical_key=logical_key,
            episode_boundary=boundary,
            executed_action_sha256=action.canonical_sha256(),
            state_values=tuple(state_values),
            next_state_values=(
                None if next_values is None else tuple(next_values)
            ),
            mode_id=mode_id,
            q_e4=q_e4,
            reward=reward,
            duration=duration,
            discount=discount,
            modeled_binding_sha256=wrapper.modeled_binding_sha256,
            source_envelope_sha256=wrapper.source_envelope_sha256,
            support_use_sha256=wrapper.support_use_sha256,
            latency_projection_sha256=wrapper.latency_projection_sha256,
        )

    @staticmethod
    def _check_features(values: Sequence[float], name: str) -> None:
        if len(values) != contract.POLICY_FEATURE_COUNT:
            raise TransitionRejectedError(
                f"{name} must contain {contract.POLICY_FEATURE_COUNT} values"
            )
        for index, value in enumerate(values):
            if not math.isfinite(float(value)):
                raise TransitionRejectedError(
                    f"{name}[{index}] "
                    f"({contract.POLICY_FEATURE_ORDER[index]}) is not finite"
                )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> ModeledReplayTensorBatchV1:
        if not isinstance(generator, torch.Generator):
            raise ReplaySamplingError(
                "sampling requires an explicit local CPU generator"
            )
        if generator is torch.default_generator:
            raise ReplaySamplingError("global generator is forbidden")
        if generator.device.type != "cpu":
            raise ReplaySamplingError("sampling generator must be on CPU")
        _positive_int(batch_size, "batch_size")
        if batch_size > len(self._rows):
            raise ReplaySamplingError(
                f"cannot sample {batch_size} rows from {len(self._rows)}"
            )
        permutation = torch.randperm(
            len(self._rows), generator=generator, device="cpu"
        )
        rows = [self._rows[int(index)] for index in permutation[:batch_size]]
        return self._build_batch(rows)

    def _build_batch(
        self, rows: Sequence[_ModeledStoredRow]
    ) -> ModeledReplayTensorBatchV1:
        state = torch.tensor(
            [row.state_values for row in rows], dtype=DEFAULT_FLOAT_DTYPE
        )
        next_state = torch.zeros(
            (len(rows), contract.POLICY_FEATURE_COUNT),
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        for index, row in enumerate(rows):
            if row.next_state_values is not None:
                next_state[index] = torch.tensor(
                    row.next_state_values, dtype=DEFAULT_FLOAT_DTYPE
                )
        return ModeledReplayTensorBatchV1(
            _state=state,
            _next_state=next_state,
            _mode_id=torch.tensor(
                [row.mode_id for row in rows], dtype=torch.int64
            ),
            _q_e4=torch.tensor(
                [row.q_e4 for row in rows], dtype=torch.int64
            ),
            _reward=torch.tensor(
                [row.reward for row in rows], dtype=DEFAULT_FLOAT_DTYPE
            ),
            _duration=torch.tensor(
                [row.duration for row in rows], dtype=torch.int64
            ),
            _discount=torch.tensor(
                [row.discount for row in rows], dtype=DEFAULT_FLOAT_DTYPE
            ),
            _has_next_state=torch.tensor(
                [row.has_next_state for row in rows], dtype=torch.bool
            ),
            _bootstrap=torch.tensor(
                [row.bootstrap for row in rows], dtype=torch.bool
            ),
            _terminated=torch.tensor(
                [row.terminated for row in rows], dtype=torch.bool
            ),
            _truncated=torch.tensor(
                [row.truncated for row in rows], dtype=torch.bool
            ),
            _binding=self._binding,
            _audit=tuple(row.audit_record() for row in rows),
        )


class ModeledCompositeHybridSacTrainerV1(run4_trainer._Run4TrainerCore):
    """Hybrid-SAC trainer restricted to modeled-composite offline batches."""

    def __init__(
        self,
        *,
        actor: Any,
        critics: Any,
        config: run4_trainer.TrainerConfigV1,
        expected_binding: ModeledReplayBindingV1,
        target_generator: torch.Generator,
        actor_generator: torch.Generator,
    ) -> None:
        if type(config) is not run4_trainer.TrainerConfigV1:
            raise run4_trainer.TrainerStateError(
                "config must be exact TrainerConfigV1"
            )
        if type(expected_binding) is not ModeledReplayBindingV1:
            raise run4_trainer.TrainerStateError(
                "expected binding must be exact ModeledReplayBindingV1"
            )
        expected_binding.require_offline_modeled_training()
        if (
            expected_binding.policy_feature_count
            != contract.POLICY_FEATURE_COUNT
            or expected_binding.policy_feature_order
            != tuple(contract.POLICY_FEATURE_ORDER)
        ):
            raise run4_trainer.TrainerStateError(
                "modeled binding feature schema differs from Run 4"
            )
        try:
            validate_run4_models(actor, critics)
        except Exception as exc:
            raise run4_trainer.TrainerStateError(
                "models do not satisfy Run-4 binding"
            ) from exc
        for name, generator in (
            ("target_generator", target_generator),
            ("actor_generator", actor_generator),
        ):
            if not isinstance(generator, torch.Generator):
                raise run4_trainer.TrainerStateError(
                    f"{name} must be a torch.Generator"
                )
            if generator is torch.default_generator:
                raise run4_trainer.TrainerStateError(
                    f"{name} must not be the global generator"
                )
            if generator.device.type != "cpu":
                raise run4_trainer.TrainerStateError(
                    f"{name} must be on CPU"
                )
        if target_generator is actor_generator:
            raise run4_trainer.TrainerStateError(
                "target and actor generators must be distinct"
            )

        self.actor = actor
        self.critics = critics
        self.config = config
        self.expected_binding = expected_binding
        self._target_generator = target_generator
        self._actor_generator = actor_generator
        self._online_critic_parameters = list(
            chain(
                critics.critic_1.parameters(),
                critics.critic_2.parameters(),
            )
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self._online_critic_parameters, lr=config.critic_lr
        )
        self.update_count = 0
        self._assert_optimizer_wiring()

    def _require_evidence_class(self, binding: Any) -> None:
        if type(binding) is not ModeledReplayBindingV1:
            raise run4_trainer.TrainerStateError(
                "modeled trainer requires ModeledReplayBindingV1"
            )
        binding.require_offline_modeled_training()

    def _preflight(self, batch: Any) -> None:
        error = run4_trainer.TrainerPreflightError
        self._assert_optimizer_wiring()
        try:
            validate_run4_models(self.actor, self.critics)
        except Exception as exc:
            raise run4_trainer.TrainerStateError(
                "models changed after construction"
            ) from exc
        if type(batch) is not ModeledReplayTensorBatchV1:
            raise error(
                "modeled update requires exact ModeledReplayTensorBatchV1"
            )
        try:
            self._require_evidence_class(batch.binding)
            self.expected_binding.assert_exactly(batch.binding)
        except Exception as exc:
            raise error("modeled batch binding differs") from exc
        if batch.float_dtype is not torch.float32:
            raise error("batch must use CPU float32 tensors")
        size = batch.batch_size
        if size < 1:
            raise error("batch must not be empty")

        floats = {
            "state": (batch.state, (size, contract.POLICY_FEATURE_COUNT)),
            "next_state": (
                batch.next_state,
                (size, contract.POLICY_FEATURE_COUNT),
            ),
            "reward": (batch.reward, (size,)),
            "discount": (batch.discount(), (size,)),
            "q_normalized_executed": (
                batch.q_normalized_executed,
                (size,),
            ),
        }
        for name, (tensor, shape) in floats.items():
            if tuple(tensor.shape) != shape:
                raise error(f"{name} must have shape {shape}")
            if (
                tensor.device.type != "cpu"
                or tensor.dtype is not torch.float32
            ):
                raise error(f"{name} must be a CPU float32 tensor")
            if not bool(torch.isfinite(tensor).all()):
                raise error(f"{name} contains a non-finite value")

        integers = {
            "mode_id": batch.mode_id,
            "q_e4": batch.q_e4,
            "duration": batch.duration,
        }
        for name, tensor in integers.items():
            if tuple(tensor.shape) != (size,):
                raise error(f"{name} must have shape ({size},)")
            if (
                tensor.device.type != "cpu"
                or tensor.dtype is not torch.int64
            ):
                raise error(f"{name} must be a CPU int64 tensor")

        masks = {
            "has_next_state": batch.has_next_state,
            "bootstrap": batch.bootstrap,
            "terminated": batch.terminated,
            "truncated": batch.truncated,
        }
        for name, tensor in masks.items():
            if tuple(tensor.shape) != (size,):
                raise error(f"{name} must have shape ({size},)")
            if (
                tensor.device.type != "cpu"
                or tensor.dtype is not torch.bool
            ):
                raise error(f"{name} must be a CPU bool tensor")

        has_next = batch.has_next_state
        terminated = batch.terminated
        truncated = batch.truncated
        bootstrap = batch.bootstrap
        if bool((terminated & truncated).any()):
            raise error("row cannot be terminated and truncated")
        expected_bootstrap = has_next & (~terminated) & (~truncated)
        if not bool(torch.equal(bootstrap, expected_bootstrap)):
            raise error("bootstrap mask differs from real successor mask")
        if not bool(torch.equal(has_next, ~(terminated | truncated))):
            raise error("successor presence differs from episode boundary")
        absent = (~has_next).nonzero(as_tuple=False).squeeze(1)
        if int(absent.numel()):
            wanted = torch.zeros(
                (int(absent.numel()), contract.POLICY_FEATURE_COUNT),
                dtype=torch.float32,
            )
            if not bool(
                torch.equal(
                    batch.next_state.index_select(0, absent),
                    wanted,
                )
            ):
                raise error(
                    "rows without successors must use the zero sentinel"
                )

        if bool((batch.mode_id < 0).any()) or bool(
            (batch.mode_id >= EXPECTED_MODE_COUNT).any()
        ):
            raise error("mode_id is outside action catalog")
        if bool((batch.q_e4 < Q_E4_MIN).any()) or bool(
            (batch.q_e4 > Q_E4_MAX).any()
        ):
            raise error("q_e4 is outside execution range")
        if bool((batch.duration < MINIMUM_HOLD_TENSORS).any()):
            raise error("duration violates minimum hold")
        discount = batch.discount()
        if bool((discount <= 0.0).any()) or bool((discount > 1.0).any()):
            raise error("discount must lie in (0, 1]")

        for row in batch.audit:
            if row["evidence_class"] != MODELED_EVIDENCE_CLASS:
                raise error("audit lost MODELED_COMPOSITE evidence")
            if row["modeled_binding_sha256"] != (
                self.expected_binding.modeled_composite_binding_sha256
            ):
                raise error("audit modeled binding differs")
            if (
                row["offline_training_only"] is not True
                or row["production_authorized"] is not False
            ):
                raise error("audit acquired an empirical/production claim")


class ModeledCompositeOfflineRunnerV1:
    """Offline ingestion and SAC-update runner; it has no live-service hooks."""

    def __init__(
        self,
        *,
        model_bundle: Run4ModelBundleV1,
        replay_buffer: ModeledCompositeReplayBufferV1,
        trainer: ModeledCompositeHybridSacTrainerV1,
        replay_generator: torch.Generator,
    ) -> None:
        if type(model_bundle) is not Run4ModelBundleV1:
            raise run4_trainer.TrainerStateError(
                "model_bundle must be exact Run4ModelBundleV1"
            )
        if type(replay_buffer) is not ModeledCompositeReplayBufferV1:
            raise run4_trainer.TrainerStateError(
                "replay_buffer has a foreign type"
            )
        if type(trainer) is not ModeledCompositeHybridSacTrainerV1:
            raise run4_trainer.TrainerStateError("trainer has a foreign type")
        replay_buffer.binding.assert_exactly(trainer.expected_binding)
        if (
            trainer.actor is not model_bundle.actor
            or trainer.critics is not model_bundle.critics
        ):
            raise run4_trainer.TrainerStateError(
                "runner model bundle differs from trainer models"
            )
        if not isinstance(replay_generator, torch.Generator):
            raise ReplaySamplingError(
                "replay_generator must be a torch.Generator"
            )
        if (
            replay_generator is torch.default_generator
            or replay_generator.device.type != "cpu"
        ):
            raise ReplaySamplingError(
                "replay_generator must be a private CPU generator"
            )
        if replay_generator in (
            trainer._target_generator,
            trainer._actor_generator,
        ):
            raise run4_trainer.TrainerStateError(
                "replay, target, and actor RNG streams must be distinct"
            )
        self.model_bundle = model_bundle
        self.replay_buffer = replay_buffer
        self.trainer = trainer
        self._replay_generator = replay_generator

    @property
    def binding(self) -> ModeledReplayBindingV1:
        return self.replay_buffer.binding

    def ingest(
        self, wrapper: modeled.ModeledCompositeOfflineTransitionV1
    ) -> None:
        self.replay_buffer.insert(wrapper)

    def ingest_many(
        self, wrappers: Iterable[modeled.ModeledCompositeOfflineTransitionV1]
    ) -> int:
        if isinstance(wrappers, (str, bytes)):
            raise TransitionRejectedError("wrappers must be an iterable")
        count = 0
        for wrapper in wrappers:
            self.ingest(wrapper)
            count += 1
        return count

    def train_once(
        self, batch_size: int
    ) -> run4_trainer.UpdateMetricsV1:
        batch = self.replay_buffer.sample(
            batch_size, generator=self._replay_generator
        )
        return self.trainer.update_once(batch)

    def audit_summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "accepted_count": self.replay_buffer.accepted_count,
                "binding_sha256": self.binding.canonical_sha256,
                "deployment_claim_allowed": False,
                "evidence_class": MODELED_EVIDENCE_CLASS,
                "evicted_count": self.replay_buffer.evicted_count,
                "mcs_acceptance_result_sha256": (
                    self.binding.mcs_acceptance_result_sha256
                ),
                "mcs_model_binding_sha256": (
                    self.binding.mcs_model_binding_sha256
                ),
                "offline_training_only": True,
                "production_authorized": False,
                "replay_resident_count": len(self.replay_buffer),
                "schema": MODELED_OFFLINE_RUNNER_SCHEMA_ID,
                "trainer_update_count": self.trainer.update_count,
            }
        )


class ModeledCompositeOfflineFactoryV1:
    """Construct deterministic CPU-only modeled replay/training components."""

    @staticmethod
    def build(
        *,
        modeled_binding: modeled.ModeledCompositeBindingV1,
        gamma: float,
        freshness_policy_sha256: str,
        empirical_scaling_sha256: str,
        capacity: int,
        trainer_config: run4_trainer.TrainerConfigV1,
        actor_seed: int,
        critic_seed: int,
        replay_seed: int,
        target_seed: int,
        trainer_actor_seed: int,
    ) -> ModeledCompositeOfflineRunnerV1:
        for value, name in (
            (actor_seed, "actor_seed"),
            (critic_seed, "critic_seed"),
            (replay_seed, "replay_seed"),
            (target_seed, "target_seed"),
            (trainer_actor_seed, "trainer_actor_seed"),
        ):
            if type(value) is not int or value < 0:
                raise run4_trainer.TrainerStateError(
                    f"{name} must be a non-negative exact int"
                )
        if len({replay_seed, target_seed, trainer_actor_seed}) != 3:
            raise run4_trainer.TrainerStateError(
                "replay, target, and actor RNG seeds must be distinct"
            )
        binding = ModeledReplayBindingV1.from_modeled_composite(
            modeled_binding=modeled_binding,
            gamma=gamma,
            freshness_policy_sha256=freshness_policy_sha256,
            empirical_scaling_sha256=empirical_scaling_sha256,
        )
        model_bundle = build_run4_models(
            actor_seed=actor_seed,
            critic_seed=critic_seed,
        )
        replay_buffer = ModeledCompositeReplayBufferV1(capacity, binding)
        target_generator = torch.Generator(device="cpu")
        target_generator.manual_seed(target_seed)
        actor_generator = torch.Generator(device="cpu")
        actor_generator.manual_seed(trainer_actor_seed)
        replay_generator = torch.Generator(device="cpu")
        replay_generator.manual_seed(replay_seed)
        trainer = ModeledCompositeHybridSacTrainerV1(
            actor=model_bundle.actor,
            critics=model_bundle.critics,
            config=trainer_config,
            expected_binding=binding,
            target_generator=target_generator,
            actor_generator=actor_generator,
        )
        return ModeledCompositeOfflineRunnerV1(
            model_bundle=model_bundle,
            replay_buffer=replay_buffer,
            trainer=trainer,
            replay_generator=replay_generator,
        )
