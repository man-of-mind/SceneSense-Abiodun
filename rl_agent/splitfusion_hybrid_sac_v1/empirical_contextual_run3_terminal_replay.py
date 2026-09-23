"""Authenticated terminal replay for the bounded Run-3 contextual study.

Run 3 is deliberately a one-step problem.  A replay row contains only the
causal 31-feature state, the executed hybrid action, and the realized scalar
reward.  Network probabilities and latency-quantile parameters remain inside
the immutable simulator audit record; they are never tensorized for learning.
"""

from __future__ import annotations

import hashlib
import math
import struct
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .action_contract import EXPECTED_MODE_COUNT, Q_E4_MAX, round_half_up_q_e4
from .empirical_contextual_contract import EmpiricalActionV1, require_supported_action
from .empirical_contextual_environment import (
    EmpiricalPolicyObservationV1,
    EmpiricalStepResultV1,
)
from .empirical_contextual_fit_partition import (
    EmpiricalFitPartitionV1,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
)
from .empirical_contextual_terminal_replay import EmpiricalTerminalTransitionV1
from .empirical_quality_surface import SurfaceQueryResult
from .empirical_contextual_run3_reward import (
    RUN3_KERNEL_SPEC_SHA256,
    RUN3_REWARD_SPEC_SHA256,
    Run3SimulatedOutcomeV1,
    Run3TerminalOutcome,
)
from .modeled_smoke_support import (
    MODELED_SMOKE_MODE_Q_E4_BOUNDS,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from .state_reward_transition_contract import (
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    assert_policy_features_exclude_forbidden_fields,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "RUN3_REPLAY_PHASE_LABEL",
    "Run3ReplayBindingV1",
    "Run3TerminalBatchV1",
    "Run3TerminalReplayV1",
    "Run3TerminalTransitionV1",
    "Run3ReplayError",
    "Run3TransitionRejected",
    "Run3DuplicateTransition",
    "Run3IdentityConflict",
]


RUN3_REPLAY_PHASE_LABEL = "RUN3_REALIZED_TERMINAL_CONTEXTUAL_REPLAY_V1"
_TRANSITION_SCHEMA = "splitfusion.run3_realized_terminal_transition.v1"
_BINDING_SCHEMA = "splitfusion.run3_realized_terminal_replay_binding.v1"
_BATCH_SCHEMA = "splitfusion.run3_realized_terminal_batch.v1"
Q_CRITIC_NORMALIZER = Q_E4_MAX


class Run3ReplayError(RuntimeError):
    """Base error for the Run-3 replay boundary."""


class Run3TransitionRejected(Run3ReplayError):
    """A row failed authentication or terminal-learning semantics."""


class Run3DuplicateTransition(Run3TransitionRejected):
    """The exact lifetime transition has already been accepted."""


class Run3IdentityConflict(Run3TransitionRejected):
    """One collection identity was reused for different content."""


def _tensor_sha256(value: Tensor) -> str:
    """Hash exact CPU tensor metadata and bytes without changing its value."""

    if type(value) is not Tensor or value.device.type != "cpu":
        raise Run3ReplayError("batch seal accepts exact CPU tensors only")
    contiguous = value.detach().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise Run3TransitionRejected(f"{name} must be an exact finite float")
    return value


def _float32(value: float, name: str) -> float:
    result = torch.tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(result)):
        raise Run3TransitionRejected(f"{name} is not finite in float32")
    return float(result)


def _transition_document(
    *,
    collection_session_uuid: str,
    collection_seq: int,
    decision_key: str,
    observation: EmpiricalPolicyObservationV1,
    action: EmpiricalActionV1,
    requested_q: float,
    source_result: EmpiricalStepResultV1,
    source_d1_transition: EmpiricalTerminalTransitionV1,
    quality_query: SurfaceQueryResult,
    quality_components: Tuple[Tuple[str, Optional[float], bool, str], ...],
    q_loc: float,
    q_seg: float,
    q_perc: float,
    realized: Run3SimulatedOutcomeV1,
    reward64: float,
    emitted_reward_float32: float,
    emitted_reward_float32_bits_hex: str,
    d1_environment_binding_sha256: str,
) -> Dict[str, Any]:
    return {
        "action": asdict(action),
        "collection_seq": collection_seq,
        "collection_session_uuid": collection_session_uuid,
        "d1_environment_binding_sha256": d1_environment_binding_sha256,
        "decision_key": decision_key,
        "emitted_reward_float32": emitted_reward_float32,
        "emitted_reward_float32_bits_hex": emitted_reward_float32_bits_hex,
        "fit_partition_sha256": REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        "kernel_spec_sha256": RUN3_KERNEL_SPEC_SHA256,
        "modeled_smoke_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "observation": asdict(observation),
        "quality_components": [list(item) for item in quality_components],
        "q_loc": q_loc,
        "q_perc": q_perc,
        "q_seg": q_seg,
        "realized": realized.to_canonical_dict(),
        "requested_q": requested_q,
        "reward64": reward64,
        "reward_spec_sha256": RUN3_REWARD_SPEC_SHA256,
        "sampling_split": TRAIN_SPLIT,
        "schema": _TRANSITION_SCHEMA,
        "source_result": asdict(source_result),
        "source_d1_transition_sha256": source_d1_transition.canonical_sha256(),
        "quality_query": {
            "hidden": asdict(quality_query.hidden),
            "policy": quality_query.policy.to_dict(),
        },
    }


@dataclass(frozen=True, slots=True)
class Run3TerminalTransitionV1:
    """One exact learning-eligible realized Run-3 terminal."""

    collection_session_uuid: str
    collection_seq: int
    decision_key: str
    observation: EmpiricalPolicyObservationV1
    action: EmpiricalActionV1
    requested_q: float
    source_result: EmpiricalStepResultV1
    source_d1_transition: EmpiricalTerminalTransitionV1
    quality_query: SurfaceQueryResult
    quality_components: Tuple[Tuple[str, Optional[float], bool, str], ...]
    q_loc: float
    q_seg: float
    q_perc: float
    realized: Run3SimulatedOutcomeV1
    reward64: float
    emitted_reward_float32: float
    emitted_reward_float32_bits_hex: str
    d1_environment_binding_sha256: str
    _attestation_sha256: str = field(repr=False)

    @classmethod
    def issue(
        cls,
        *,
        collection_session_uuid: str,
        collection_seq: int,
        decision_key: str,
        observation: EmpiricalPolicyObservationV1,
        action: EmpiricalActionV1,
        requested_q: float,
        source_result: EmpiricalStepResultV1,
        source_d1_transition: EmpiricalTerminalTransitionV1,
        quality_query: SurfaceQueryResult,
        quality_components: Tuple[Tuple[str, Optional[float], bool, str], ...],
        q_loc: float,
        q_seg: float,
        q_perc: float,
        realized: Run3SimulatedOutcomeV1,
        d1_environment_binding_sha256: str,
    ) -> "Run3TerminalTransitionV1":
        reward64 = realized.reward_result.scalar_reward
        if type(reward64) is not float:
            raise Run3TransitionRejected("learning row has no scalar reward")
        emitted = _float32(reward64, "reward")
        emitted_bits = struct.pack(">f", emitted).hex()
        document = _transition_document(
            collection_session_uuid=collection_session_uuid,
            collection_seq=collection_seq,
            decision_key=decision_key,
            observation=observation,
            action=action,
            requested_q=requested_q,
            source_result=source_result,
            source_d1_transition=source_d1_transition,
            quality_query=quality_query,
            quality_components=quality_components,
            q_loc=q_loc,
            q_seg=q_seg,
            q_perc=q_perc,
            realized=realized,
            reward64=reward64,
            emitted_reward_float32=emitted,
            emitted_reward_float32_bits_hex=emitted_bits,
            d1_environment_binding_sha256=d1_environment_binding_sha256,
        )
        result = cls(
            collection_session_uuid=collection_session_uuid,
            collection_seq=collection_seq,
            decision_key=decision_key,
            observation=observation,
            action=action,
            requested_q=requested_q,
            source_result=source_result,
            source_d1_transition=source_d1_transition,
            quality_query=quality_query,
            quality_components=quality_components,
            q_loc=q_loc,
            q_seg=q_seg,
            q_perc=q_perc,
            realized=realized,
            reward64=reward64,
            emitted_reward_float32=emitted,
            emitted_reward_float32_bits_hex=emitted_bits,
            d1_environment_binding_sha256=d1_environment_binding_sha256,
            _attestation_sha256=canonical_sha256(document),
        )
        result.revalidate()
        return result

    @property
    def logical_key(self) -> Tuple[str, int]:
        return self.collection_session_uuid, self.collection_seq

    @property
    def reward(self) -> float:
        return self.emitted_reward_float32

    @property
    def terminal_outcome(self) -> Run3TerminalOutcome:
        return self.realized.terminal_outcome

    @property
    def latency_ms(self) -> Optional[float]:
        return self.realized.reward_result.latency_ms

    def canonical_sha256(self) -> str:
        return canonical_sha256(
            _transition_document(
                collection_session_uuid=self.collection_session_uuid,
                collection_seq=self.collection_seq,
                decision_key=self.decision_key,
                observation=self.observation,
                action=self.action,
                requested_q=self.requested_q,
                source_result=self.source_result,
                source_d1_transition=self.source_d1_transition,
                quality_query=self.quality_query,
                quality_components=self.quality_components,
                q_loc=self.q_loc,
                q_seg=self.q_seg,
                q_perc=self.q_perc,
                realized=self.realized,
                reward64=self.reward64,
                emitted_reward_float32=self.emitted_reward_float32,
                emitted_reward_float32_bits_hex=(
                    self.emitted_reward_float32_bits_hex
                ),
                d1_environment_binding_sha256=(
                    self.d1_environment_binding_sha256
                ),
            )
        )

    def revalidate(self) -> None:
        try:
            parsed = uuid.UUID(self.collection_session_uuid)
        except (TypeError, ValueError, AttributeError) as exc:
            raise Run3TransitionRejected("collection session is not a UUID") from exc
        if str(parsed) != self.collection_session_uuid:
            raise Run3TransitionRejected("collection session UUID is not canonical")
        if type(self.collection_seq) is not int or self.collection_seq < 0:
            raise Run3TransitionRejected("collection_seq must be non-negative int")
        expected_key = f"run3:{self.collection_session_uuid}:{self.collection_seq}"
        if self.decision_key != expected_key:
            raise Run3TransitionRejected(
                "decision key must depend only on session and collection sequence"
            )
        if type(self.observation) is not EmpiricalPolicyObservationV1:
            raise Run3TransitionRejected("observation has foreign type")
        self.observation.__post_init__()
        if type(self.action) is not EmpiricalActionV1:
            raise Run3TransitionRejected("action has foreign type")
        require_supported_action(self.action.mode_id, self.action.q_e4)
        requested_q = _finite_float(self.requested_q, "requested_q")
        if not 0.0 <= requested_q <= 0.98:
            raise Run3TransitionRejected("requested_q escaped [0,0.98]")
        if round_half_up_q_e4(requested_q) != self.action.q_e4:
            raise Run3TransitionRejected("requested q/canonical executed q_e4 drift")
        if type(self.source_result) is not EmpiricalStepResultV1:
            raise Run3TransitionRejected("source result has foreign type")
        if type(self.source_d1_transition) is not EmpiricalTerminalTransitionV1:
            raise Run3TransitionRejected("authenticated D1 source has foreign type")
        self.source_d1_transition.revalidate()
        if (
            self.source_d1_transition.observation != self.observation
            or self.source_d1_transition.action != self.action
            or self.source_d1_transition.result != self.source_result
            or self.source_d1_transition.d1_binding.canonical_sha256()
            != self.d1_environment_binding_sha256
        ):
            raise Run3TransitionRejected("authenticated D1 source join drift")
        audit = self.source_result.audit
        if (
            audit.executed_mode_id != self.action.mode_id
            or audit.executed_q_e4 != self.action.q_e4
        ):
            raise Run3TransitionRejected("source action join mismatch")
        if type(self.quality_query) is not SurfaceQueryResult:
            raise Run3TransitionRejected("quality query has foreign type")
        if (
            self.quality_query.hidden.sample_id != audit.sample_id
            or self.quality_query.hidden.episode_id != audit.episode_id
            or self.quality_query.hidden.frame_id != audit.frame_id
            or self.quality_query.hidden.grid_split != "fit"
            or self.quality_query.policy.mode_id != self.action.mode_id
            or self.quality_query.policy.q_e4 != self.action.q_e4
            or float(self.quality_query.policy.payload.total_transmitted_bytes)
            != audit.total_transmitted_bytes
        ):
            raise Run3TransitionRejected("quality query/source identity drift")
        if self.observation.environment_binding_sha256 != (
            self.d1_environment_binding_sha256
        ):
            raise Run3TransitionRejected("observation/environment binding mismatch")
        if not _is_sha256(self.d1_environment_binding_sha256):
            raise Run3TransitionRejected("D1 environment binding is malformed")
        for name in ("q_loc", "q_seg", "q_perc"):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise Run3TransitionRejected(f"{name} escaped [0,1]")
            _float32(value, name)
        # Interior q values use the registered direct interpolation of Qperc;
        # recomputing the nonlinear identity from separately interpolated
        # Qloc/Qseg would be a different surface.  The exact source-query join
        # below is therefore the authority for all three values.
        source_q = self.source_result.policy.q_perc
        if type(source_q) is not float or self.q_perc != source_q:
            raise Run3TransitionRejected("source/realized Qperc mismatch")
        if type(self.realized) is not Run3SimulatedOutcomeV1:
            raise Run3TransitionRejected("realized outcome has foreign type")
        self.realized.revalidate()
        if self.realized.random_draws.decision_key != self.decision_key:
            raise Run3TransitionRejected("kernel draw/decision identity mismatch")
        if self.realized.source_q_perc != self.q_perc:
            raise Run3TransitionRejected("kernel/source Qperc mismatch")
        reward = self.realized.reward_result
        if not reward.learning_eligible or reward.scalar_reward is None:
            raise Run3TransitionRejected("excluded fault cannot enter replay")
        if self.realized.terminal_outcome in (
            Run3TerminalOutcome.INFRASTRUCTURE_FAULT_EXCLUDED,
            Run3TerminalOutcome.EVALUATOR_FAULT_EXCLUDED,
        ):
            raise Run3TransitionRejected("excluded fault cannot enter replay")
        if self.reward64 != reward.scalar_reward:
            raise Run3TransitionRejected("reward64 differs from realized reward")
        emitted = _float32(self.reward64, "reward64")
        if self.emitted_reward_float32 != emitted:
            raise Run3TransitionRejected("emitted float32 reward drift")
        if self.emitted_reward_float32_bits_hex != struct.pack(">f", emitted).hex():
            raise Run3TransitionRejected("emitted float32 reward bits drift")
        expected_names = (
            "seg_vehicle_iou",
            "seg_person_iou",
            "vehicle_recall",
            "person_recall",
            "vehicle_xy_error_m",
            "person_xy_error_m",
            "q_seg",
            "q_loc",
            "q_perc",
        )
        if (
            type(self.quality_components) is not tuple
            or tuple(item[0] for item in self.quality_components) != expected_names
        ):
            raise Run3TransitionRejected("quality component inventory drift")
        source_components = tuple(
            (item.name, item.value, item.valid, item.status)
            for item in self.quality_query.policy.quality
        )
        if self.quality_components != source_components:
            raise Run3TransitionRejected("quality components differ from source query")
        by_name = {}
        for item in self.quality_components:
            if type(item) is not tuple or len(item) != 4:
                raise Run3TransitionRejected("quality component record malformed")
            name, value, valid, status = item
            if type(valid) is not bool or type(status) is not str or not status:
                raise Run3TransitionRejected(f"{name} validity/status malformed")
            if valid:
                _finite_float(value, f"quality_components[{name}]")
            elif value is not None:
                raise Run3TransitionRejected(f"invalid {name} carries a value")
            by_name[name] = value
        if by_name["q_loc"] != self.q_loc or by_name["q_seg"] != self.q_seg or by_name["q_perc"] != self.q_perc:
            raise Run3TransitionRejected("separate quality components drift")
        policy = self.source_result.policy
        if (
            type(policy.p_complete_reassembly_given_sent) is not float
            or type(policy.p_edge_admission_given_reassembled) is not float
            or type(policy.latency_proxy_ms) is not float
            or type(policy.latency_proxy_p95_ms) is not float
            or type(policy.latency_proxy_p99_ms) is not float
        ):
            raise Run3TransitionRejected("source kernel inputs are unavailable")
        if self.realized.p_complete_reassembly_given_sent != (
            policy.p_complete_reassembly_given_sent
        ) or self.realized.p_edge_admission_given_reassembled != (
            policy.p_edge_admission_given_reassembled
        ):
            raise Run3TransitionRejected("source/kernel probability mismatch")
        proxy = self.realized.latency_proxy
        if (proxy.p50_ms, proxy.p95_ms, proxy.p99_ms) != (
            policy.latency_proxy_ms,
            policy.latency_proxy_p95_ms,
            policy.latency_proxy_p99_ms,
        ):
            raise Run3TransitionRejected("source/kernel latency proxy mismatch")
        if not _is_sha256(self._attestation_sha256):
            raise Run3TransitionRejected("transition attestation is malformed")
        if self._attestation_sha256 != self.canonical_sha256():
            raise Run3TransitionRejected("transition attestation mismatch")


@dataclass(frozen=True, slots=True)
class Run3ReplayBindingV1:
    d1_environment_binding_sha256: str
    train_scene_inventory_sha256: str
    train_radio_inventory_sha256: str
    fit_partition_sha256: str = REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
    reward_spec_sha256: str = RUN3_REWARD_SPEC_SHA256
    kernel_spec_sha256: str = RUN3_KERNEL_SPEC_SHA256
    modeled_smoke_support_sha256: str = MODELED_SMOKE_SUPPORT_SHA256
    policy_feature_order: Tuple[str, ...] = tuple(POLICY_FEATURE_ORDER)
    policy_feature_count: int = POLICY_FEATURE_COUNT
    float_dtype: str = str(torch.float32)
    terminal_semantics: str = "NEXT_STATE_NONE_TERMINATED_TRUE_DURATION_1_DISCOUNT_0"
    schema: str = _BINDING_SCHEMA

    @classmethod
    def from_transition(
        cls, row: Run3TerminalTransitionV1, partition: EmpiricalFitPartitionV1
    ) -> "Run3ReplayBindingV1":
        row.revalidate()
        if type(partition) is not EmpiricalFitPartitionV1:
            raise Run3TransitionRejected("fit partition has foreign type")
        partition.__post_init__()
        if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
            raise Run3TransitionRejected("fit partition hash drift")
        train_scenes = tuple(sorted(item.sample_id for item in partition.scene_assignments if item.split == TRAIN_SPLIT))
        train_radio = tuple(sorted(item.csv_row_number for item in partition.radio_assignments if item.split == TRAIN_SPLIT))
        if row.source_result.audit.sample_id not in frozenset(train_scenes):
            raise Run3TransitionRejected("non-training scene refused by binding")
        if row.source_result.audit.hidden_radio_csv_row_number not in frozenset(train_radio):
            raise Run3TransitionRejected("non-training radio row refused by binding")
        result = cls(
            d1_environment_binding_sha256=row.d1_environment_binding_sha256,
            train_scene_inventory_sha256=canonical_sha256({"train_scene_ids": list(train_scenes)}),
            train_radio_inventory_sha256=canonical_sha256({"train_radio_rows": list(train_radio)}),
        )
        result.require_valid()
        return result

    def require_valid(self) -> None:
        if not _is_sha256(self.d1_environment_binding_sha256):
            raise Run3TransitionRejected("binding D1 hash malformed")
        if not _is_sha256(self.train_scene_inventory_sha256) or not _is_sha256(self.train_radio_inventory_sha256):
            raise Run3TransitionRejected("binding train inventory hash malformed")
        if self.fit_partition_sha256 != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
            raise Run3TransitionRejected("binding fit partition drift")
        if self.reward_spec_sha256 != RUN3_REWARD_SPEC_SHA256:
            raise Run3TransitionRejected("binding reward spec drift")
        if self.kernel_spec_sha256 != RUN3_KERNEL_SPEC_SHA256:
            raise Run3TransitionRejected("binding kernel spec drift")
        if self.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
            raise Run3TransitionRejected("binding support drift")
        if self.policy_feature_order != tuple(POLICY_FEATURE_ORDER):
            raise Run3TransitionRejected("binding policy feature order drift")
        assert_policy_features_exclude_forbidden_fields()
        if self.policy_feature_count != POLICY_FEATURE_COUNT:
            raise Run3TransitionRejected("binding feature count drift")
        if self.float_dtype != str(torch.float32):
            raise Run3TransitionRejected("binding dtype drift")
        if self.terminal_semantics != (
            "NEXT_STATE_NONE_TERMINATED_TRUE_DURATION_1_DISCOUNT_0"
        ) or self.schema != _BINDING_SCHEMA:
            raise Run3TransitionRejected("binding terminal/schema drift")

    def canonical_sha256(self) -> str:
        self.require_valid()
        return canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class _AdmittedRun3Record:
    """Compact immutable learner/diagnostic view issued after deep admission."""

    row: Run3TerminalTransitionV1 = field(repr=False)
    accepted_digest: str
    logical_key: Tuple[str, int]
    observation_values: Tuple[float, ...]
    mode_id: int
    q_e4: int
    reward: float
    q_loc: float
    q_seg: float
    q_perc: float
    terminal_outcome: Run3TerminalOutcome
    latency_ms: Optional[float]

    @classmethod
    def from_admitted(
        cls, row: Run3TerminalTransitionV1, digest: str
    ) -> "_AdmittedRun3Record":
        return cls(
            row=row,
            accepted_digest=digest,
            logical_key=row.logical_key,
            observation_values=tuple(row.observation.values),
            mode_id=row.action.mode_id,
            q_e4=row.action.q_e4,
            reward=row.reward,
            q_loc=row.q_loc,
            q_seg=row.q_seg,
            q_perc=row.q_perc,
            terminal_outcome=row.terminal_outcome,
            latency_ms=row.latency_ms,
        )

    def seal_document(self) -> Dict[str, Any]:
        return {
            "accepted_digest": self.accepted_digest,
            "latency_ms": self.latency_ms,
            "logical_key": list(self.logical_key),
            "mode_id": self.mode_id,
            "observation_values": list(self.observation_values),
            "q_e4": self.q_e4,
            "q_loc": self.q_loc,
            "q_perc": self.q_perc,
            "q_seg": self.q_seg,
            "reward": self.reward,
            "terminal_outcome": self.terminal_outcome.value,
        }

    def require_compact_valid(self) -> None:
        if type(self.row) is not Run3TerminalTransitionV1:
            raise Run3ReplayError("compact record rich row type drift")
        if not _is_sha256(self.accepted_digest):
            raise Run3ReplayError("compact record digest drift")
        if self.logical_key != self.row.logical_key:
            raise Run3ReplayError("compact record logical key drift")
        if (
            type(self.observation_values) is not tuple
            or len(self.observation_values) != POLICY_FEATURE_COUNT
            or any(type(value) is not float or not math.isfinite(value) for value in self.observation_values)
        ):
            raise Run3ReplayError("compact observation drift")
        if type(self.mode_id) is not int or not 0 <= self.mode_id < EXPECTED_MODE_COUNT:
            raise Run3ReplayError("compact mode drift")
        if type(self.q_e4) is not int:
            raise Run3ReplayError("compact q type drift")
        require_supported_action(self.mode_id, self.q_e4)
        for name in ("reward", "q_loc", "q_seg", "q_perc"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise Run3ReplayError(f"compact {name} drift")
        if not all(0.0 <= value <= 1.0 for value in (self.q_loc, self.q_seg, self.q_perc)):
            raise Run3ReplayError("compact quality range drift")
        if type(self.terminal_outcome) is not Run3TerminalOutcome:
            raise Run3ReplayError("compact outcome drift")
        if self.latency_ms is not None and (
            type(self.latency_ms) is not float or not math.isfinite(self.latency_ms)
        ):
            raise Run3ReplayError("compact latency drift")
        # The accepted rich-row digest binds provenance integrity; every compact
        # field must still join exactly to that rich row so a finite in-range
        # compact mutation cannot be resealed at sampling.
        def same_float(left: float, right: float) -> bool:
            return struct.pack(">d", left) == struct.pack(">d", right)

        rich_observation = tuple(self.row.observation.values)
        floats_match = (
            len(self.observation_values) == len(rich_observation)
            and all(
                same_float(left, right)
                for left, right in zip(self.observation_values, rich_observation)
            )
            and same_float(self.reward, self.row.reward)
            and same_float(self.q_loc, self.row.q_loc)
            and same_float(self.q_seg, self.row.q_seg)
            and same_float(self.q_perc, self.row.q_perc)
            and (
                (self.latency_ms is None and self.row.latency_ms is None)
                or (
                    self.latency_ms is not None
                    and self.row.latency_ms is not None
                    and same_float(self.latency_ms, self.row.latency_ms)
                )
            )
        )
        if (
            not floats_match
            or self.mode_id != self.row.action.mode_id
            or self.q_e4 != self.row.action.q_e4
            or self.terminal_outcome is not self.row.terminal_outcome
        ):
            raise Run3ReplayError("compact record differs from admitted rich row")

    def require_rich_row_untampered(self) -> None:
        self.require_compact_valid()
        if self.row.canonical_sha256() != self.accepted_digest:
            raise Run3ReplayError("resident rich row changed after admission")


def _batch_seal(
    *,
    binding: Run3ReplayBindingV1,
    records: Tuple[_AdmittedRun3Record, ...],
    state: Tensor,
    mode_id: Tensor,
    q_e4: Tensor,
    reward: Tensor,
    schema: str,
) -> str:
    return canonical_sha256(
        {
            "binding_sha256": binding.canonical_sha256(),
            "record_digests": [record.accepted_digest for record in records],
            "records": [record.seal_document() for record in records],
            "schema": schema,
            "tensor_sha256": {
                "mode_id": _tensor_sha256(mode_id),
                "q_e4": _tensor_sha256(q_e4),
                "reward": _tensor_sha256(reward),
                "state": _tensor_sha256(state),
            },
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class Run3TerminalBatchV1:
    """Private learner tensors plus immutable diagnostic source rows."""

    binding: Run3ReplayBindingV1
    partition: EmpiricalFitPartitionV1 = field(repr=False)
    rows: Tuple[Run3TerminalTransitionV1, ...]
    _state: Tensor = field(repr=False)
    _mode_id: Tensor = field(repr=False)
    _q_e4: Tensor = field(repr=False)
    _reward: Tensor = field(repr=False)
    schema: str = _BATCH_SCHEMA
    _records: Tuple[_AdmittedRun3Record, ...] = field(default=(), repr=False)
    _issuer_capability: object = field(default=None, repr=False)
    _partition_identity: object = field(default=None, repr=False)
    _seal_sha256: str = field(default="", repr=False)

    @property
    def batch_size(self) -> int:
        return len(self.rows)

    @property
    def compact_records(self) -> Tuple[_AdmittedRun3Record, ...]:
        return self._records

    @property
    def state(self) -> Tensor:
        return self._state.clone()

    @property
    def mode_id(self) -> Tensor:
        return self._mode_id.clone()

    @property
    def q_e4(self) -> Tensor:
        return self._q_e4.clone()

    @property
    def q_normalized(self) -> Tensor:
        return self._q_e4.to(torch.float32) / float(Q_CRITIC_NORMALIZER)

    @property
    def reward(self) -> Tensor:
        return self._reward.clone()

    @property
    def next_state(self) -> None:
        return None

    @property
    def terminated(self) -> Tensor:
        return torch.ones(self.batch_size, dtype=torch.bool)

    @property
    def bootstrap(self) -> Tensor:
        return torch.zeros(self.batch_size, dtype=torch.bool)

    @property
    def duration(self) -> Tensor:
        return torch.ones(self.batch_size, dtype=torch.int64)

    @property
    def discount(self) -> Tensor:
        return torch.zeros(self.batch_size, dtype=torch.float32)

    def terminal_target(self) -> Tensor:
        return self.reward

    def learner_tensors(self) -> Mapping[str, Tensor]:
        return {
            "state": self.state,
            "mode_id": self.mode_id,
            "q_e4": self.q_e4,
            "q_normalized": self.q_normalized,
            "reward": self.reward,
        }

    def revalidate(self, *, expected_issuer_capability: object) -> None:
        if (
            expected_issuer_capability is None
            or self._issuer_capability is not expected_issuer_capability
        ):
            raise Run3TransitionRejected(
                "batch was not issued by the paired Run-3 replay"
            )
        if type(self.binding) is not Run3ReplayBindingV1:
            raise Run3ReplayError("batch binding has foreign type")
        self.binding.require_valid()
        if self.schema != _BATCH_SCHEMA or not self.rows or len(self._records) != len(self.rows):
            raise Run3ReplayError("batch schema/size drift")
        if (
            type(self.partition) is not EmpiricalFitPartitionV1
            or self.partition is not self._partition_identity
        ):
            raise Run3ReplayError("batch partition type drift")
        for row, record in zip(self.rows, self._records):
            if row is not record.row:
                raise Run3ReplayError("batch rich/compact row identity drift")
            record.require_rich_row_untampered()
        expected_state = torch.tensor(
            [record.observation_values for record in self._records], dtype=torch.float32
        )
        expected_mode = torch.tensor(
            [record.mode_id for record in self._records], dtype=torch.int64
        )
        expected_q = torch.tensor(
            [record.q_e4 for record in self._records], dtype=torch.int64
        )
        expected_reward = torch.tensor(
            [record.reward for record in self._records], dtype=torch.float32
        )
        for name, observed, expected in (
            ("state", self._state, expected_state),
            ("mode_id", self._mode_id, expected_mode),
            ("q_e4", self._q_e4, expected_q),
            ("reward", self._reward, expected_reward),
        ):
            if (
                type(observed) is not Tensor
                or tuple(observed.shape) != tuple(expected.shape)
                or observed.device.type != "cpu"
                or observed.dtype is not expected.dtype
                or (observed.is_floating_point() and not bool(torch.isfinite(observed).all()))
                or not torch.equal(observed, expected)
            ):
                raise Run3ReplayError(f"batch {name} tensor drift")
        expected_seal = _batch_seal(
            binding=self.binding,
            records=self._records,
            state=self._state,
            mode_id=self._mode_id,
            q_e4=self._q_e4,
            reward=self._reward,
            schema=self.schema,
        )
        if not _is_sha256(self._seal_sha256) or self._seal_sha256 != expected_seal:
            raise Run3ReplayError("batch seal mismatch")


class Run3TerminalReplayV1:
    """FIFO storage with lifetime duplicate and identity-conflict indexes."""

    def __init__(self, capacity: int, *, partition: EmpiricalFitPartitionV1) -> None:
        if type(capacity) is not int or capacity < 1:
            raise Run3ReplayError("capacity must be a positive exact integer")
        self.capacity = capacity
        if type(partition) is not EmpiricalFitPartitionV1:
            raise Run3ReplayError("replay requires exact registered fit partition")
        partition.__post_init__()
        if partition.canonical_sha256() != REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256:
            raise Run3ReplayError("replay fit partition hash drift")
        self._partition = partition
        self._train_scene_ids = frozenset(
            item.sample_id for item in partition.scene_assignments if item.split == TRAIN_SPLIT
        )
        self._train_radio_rows = frozenset(
            item.csv_row_number for item in partition.radio_assignments if item.split == TRAIN_SPLIT
        )
        self._train_scene_inventory_sha256 = canonical_sha256(
            {"train_scene_ids": sorted(self._train_scene_ids)}
        )
        self._train_radio_inventory_sha256 = canonical_sha256(
            {"train_radio_rows": sorted(self._train_radio_rows)}
        )
        self._records: Deque[_AdmittedRun3Record] = deque()
        self._lifetime_digests: set[str] = set()
        self._identity_digests: Dict[Tuple[str, int], str] = {}
        self._binding: Optional[Run3ReplayBindingV1] = None
        # Per-replay identity, deliberately absent from checkpoints.  A
        # resumed replay/trainer pair receives a fresh matching capability.
        self._issuer_capability = object()
        self.accepted_count = 0
        self.evicted_count = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def binding(self) -> Optional[Run3ReplayBindingV1]:
        return self._binding

    @property
    def trainer_issuer_capability(self) -> object:
        """Opaque identity passed only to the trainer paired with this replay."""

        return self._issuer_capability

    @property
    def rows(self) -> Tuple[Run3TerminalTransitionV1, ...]:
        return tuple(record.row for record in self._records)

    def insert(self, row: Run3TerminalTransitionV1) -> None:
        if type(row) is not Run3TerminalTransitionV1:
            raise Run3TransitionRejected("replay accepts only exact Run3 rows")
        row.revalidate()
        if row.source_result.audit.sample_id not in self._train_scene_ids:
            raise Run3TransitionRejected("non-training scene refused by replay")
        if row.source_result.audit.hidden_radio_csv_row_number not in self._train_radio_rows:
            raise Run3TransitionRejected("non-training radio row refused by replay")
        digest = row.canonical_sha256()
        prior = self._identity_digests.get(row.logical_key)
        if prior is not None:
            if prior == digest:
                raise Run3DuplicateTransition("duplicate transition")
            raise Run3IdentityConflict("collection identity conflict")
        if digest in self._lifetime_digests:
            raise Run3DuplicateTransition("lifetime digest duplicate")
        binding = Run3ReplayBindingV1(
            d1_environment_binding_sha256=row.d1_environment_binding_sha256,
            train_scene_inventory_sha256=self._train_scene_inventory_sha256,
            train_radio_inventory_sha256=self._train_radio_inventory_sha256,
        )
        binding.require_valid()
        if self._binding is not None and binding != self._binding:
            raise Run3TransitionRejected("replay binding mismatch")
        # All gates above complete before any mutation.
        if self._binding is None:
            self._binding = binding
        self._records.append(_AdmittedRun3Record.from_admitted(row, digest))
        self._lifetime_digests.add(digest)
        self._identity_digests[row.logical_key] = digest
        self.accepted_count += 1
        if len(self._records) > self.capacity:
            self._records.popleft()
            self.evicted_count += 1

    def sample(
        self, batch_size: int, *, generator: torch.Generator
    ) -> Run3TerminalBatchV1:
        if type(batch_size) is not int or batch_size < 1 or batch_size > len(self):
            raise Run3ReplayError("invalid replay batch size")
        if not isinstance(generator, torch.Generator):
            raise Run3ReplayError("an explicit torch.Generator is required")
        if generator is torch.default_generator or generator.device.type != "cpu":
            raise Run3ReplayError("replay generator must be private CPU state")
        generator_state = generator.get_state().clone()
        indices = torch.randperm(len(self), generator=generator)[:batch_size].tolist()
        resident = tuple(self._records)
        records = tuple(resident[index] for index in indices)
        # This is the sampling-time tamper gate.  The trainer later validates
        # the sealed compact copy once, without repeating deep row/D1/partition
        # authentication.
        try:
            for record in records:
                record.require_rich_row_untampered()
        except BaseException:
            # A rejected batch must not perturb the deterministic sample
            # sequence used by a later repaired/resumed attempt.
            generator.set_state(generator_state)
            raise
        rows = tuple(record.row for record in records)
        if self._binding is None:  # pragma: no cover - nonempty implies binding
            raise Run3ReplayError("nonempty replay lacks a binding")
        state = torch.tensor(
            [record.observation_values for record in records], dtype=torch.float32
        ).clone().detach()
        mode_id = torch.tensor(
            [record.mode_id for record in records], dtype=torch.int64
        ).clone().detach()
        q_e4 = torch.tensor(
            [record.q_e4 for record in records], dtype=torch.int64
        ).clone().detach()
        reward = torch.tensor(
            [record.reward for record in records], dtype=torch.float32
        ).clone().detach()
        seal = _batch_seal(
            binding=self._binding,
            records=records,
            state=state,
            mode_id=mode_id,
            q_e4=q_e4,
            reward=reward,
            schema=_BATCH_SCHEMA,
        )
        batch = Run3TerminalBatchV1(
            binding=self._binding,
            partition=self._partition,
            rows=rows,
            _state=state,
            _mode_id=mode_id,
            _q_e4=q_e4,
            _reward=reward,
            _records=records,
            _issuer_capability=self._issuer_capability,
            _partition_identity=self._partition,
            _seal_sha256=seal,
        )
        return batch
