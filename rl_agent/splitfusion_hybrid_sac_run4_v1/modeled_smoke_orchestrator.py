"""Bounded, event-sourced orchestration for the Run-4 modeled SAC smoke.

This module is the only schedule layer for the offline
``MODELED_COMPOSITE_TRAINING`` path.  It deliberately does not weaken or
reuse the production empirical runner.  A caller supplies a typed collector
which emits exact attested :class:`ModeledCompositeOfflineTransitionV1`
records, and this module supplies the frozen registered-seed schedule:

* registered seeds 17, 29 and 43 only,
* 288 balanced warm-up decisions (12 modes x 6 q strata x 4 samples),
* four modeled transitions before every gradient,
* batch size 256, and
* checkpoints at updates 0, 100, 250, 500, 1,500 and 10,000.

``run_to_hard_stop`` remains the mandatory seed-17 update-500 smoke wrapper;
``run_to_registered_update`` is the bounded continuation seam.

Checkpointing is event sourced.  The modeled replay intentionally exposes no
mutable snapshot or RNG accessor.  Reaching into its private deque, lifetime
indexes, or private generators would make a fragile second replay contract.
Instead a durable checkpoint contains the collector's canonical checkpoint,
the exact transition/action ledger, all factory seeds and binding documents,
and hashes of the complete model/optimizer/replay boundary.  Restore builds a
fresh exact :class:`ModeledCompositeOfflineRunnerV1`, asks the collector to
reconstitute the attested history, and replays the exact schedule from
genesis.  This deterministically reconstructs the actor, twin online and
target critics, both optimizers, update count, replay FIFO and lifetime
indexes, decision RNGs, replay RNG, and trainer actor/target RNGs.  The
restored boundary must match every stored hash before it can continue.

Importing this module performs no I/O, initializes no accelerator, samples no
RNG and takes no gradient.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    EXPECTED_MODE_COUNT,
    Q_E4_SCALE,
    load_contract,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    canonical_sha256,
)

from . import exploration
from . import modeled_composite_training as modeled
from . import modeled_offline_runtime as runtime
from . import models
from . import run4_contract as contract
from . import smoke_preregistration
from . import trainer


SCHEMA_ID = "splitfusion.run4.modeled_smoke_orchestrator.v1"
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_ID = "splitfusion.run4.modeled_smoke_checkpoint.v1"
PREFLIGHT_SCHEMA_ID = "splitfusion.run4.modeled_smoke_preflight.v1"
COLLECTOR_CHECKPOINT_SCHEMA_ID = (
    "splitfusion.run4.modeled_transition_collector_checkpoint.v1"
)
FIT_PARTITION_LABEL = "FIT_ONLY_VALIDATION_EXCLUDED"
CHECKPOINT_UPDATES = smoke_preregistration.FROZEN_CONFIG.checkpoint_updates


class ModeledSmokeError(RuntimeError):
    """Base class for modeled-smoke orchestration refusal."""


class ModeledSmokeBindingError(ModeledSmokeError):
    """A collector, runner, schedule, or evidence binding differs."""


class ModeledSmokePreflightError(ModeledSmokeError):
    """The exact no-gradient 288-transition gate did not pass."""


class ModeledSmokeScheduleError(ModeledSmokeError):
    """Collection/update progress differs from the frozen schedule."""


class ModeledSmokeCheckpointError(ModeledSmokeError):
    """A checkpoint is incomplete, changed, or not exactly restorable."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ModeledSmokeError("value is not canonical JSON") from exc


def _digest(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ModeledSmokeBindingError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_nonnegative(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ModeledSmokeError(f"{name} must be a non-negative exact int")
    return value


def _finite_tuple(values: object, name: str) -> Tuple[float, ...]:
    if type(values) is not tuple:
        raise ModeledSmokeBindingError(f"{name} must be an exact tuple")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModeledSmokeBindingError(f"{name}[{index}] is not real")
        converted = float(value)
        if not math.isfinite(converted):
            raise ModeledSmokeBindingError(f"{name}[{index}] is not finite")
        result.append(converted)
    return tuple(result)


def _derive_seed(master_seed: int, label: str) -> int:
    material = {
        "domain": "RUN4_MODELED_SMOKE_RNG_STREAM_V1",
        "label": label,
        "master_seed": master_seed,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_bytes(material)).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise ModeledSmokeCheckpointError("checkpoint tensor is not Tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    header = _canonical_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    )
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _tree_document(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_sha256": _tensor_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _tree_document(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [_tree_document(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ModeledSmokeCheckpointError("non-finite checkpoint scalar")
        return value
    raise ModeledSmokeCheckpointError(
        f"unsupported checkpoint value {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _tree_sha256(value: Any) -> str:
    return _sha(_tree_document(value))


@dataclass(frozen=True, slots=True)
class RunnerSeedPlanV1:
    """Every RNG stream needed to reconstruct one registered Run-4 seed."""

    master_seed: int
    actor_seed: int
    critic_seed: int
    replay_seed: int
    target_seed: int
    trainer_actor_seed: int
    decision_q_seed: int
    decision_mode_seed: int

    @classmethod
    def for_registered_seed(cls, master: int) -> "RunnerSeedPlanV1":
        if type(master) is not int or (
            master not in smoke_preregistration.FROZEN_CONFIG.seed_order
        ):
            raise ModeledSmokeBindingError(
                "modeled campaign seed is not registered"
            )
        return cls(
            master_seed=master,
            actor_seed=_derive_seed(master, "model-actor"),
            critic_seed=_derive_seed(master, "model-critics"),
            replay_seed=_derive_seed(master, "replay-sampling"),
            target_seed=_derive_seed(master, "trainer-target-q"),
            trainer_actor_seed=_derive_seed(master, "trainer-actor-q"),
            decision_q_seed=_derive_seed(master, "decision-q"),
            decision_mode_seed=_derive_seed(master, "decision-mode"),
        )

    @classmethod
    def seed17(cls) -> "RunnerSeedPlanV1":
        """Compatibility constructor for the registered smoke seed."""
        return cls.for_registered_seed(
            smoke_preregistration.FROZEN_CONFIG.initial_smoke_seed
        )

    def __post_init__(self) -> None:
        if self.master_seed not in smoke_preregistration.FROZEN_CONFIG.seed_order:
            raise ModeledSmokeBindingError(
                "modeled campaign seed is not registered"
            )
        values = (
            self.actor_seed,
            self.critic_seed,
            self.replay_seed,
            self.target_seed,
            self.trainer_actor_seed,
            self.decision_q_seed,
            self.decision_mode_seed,
        )
        if any(type(item) is not int or item < 0 for item in values):
            raise ModeledSmokeBindingError("all stream seeds must be non-negative")
        if len(set(values)) != len(values):
            raise ModeledSmokeBindingError("every RNG stream seed must be distinct")

    def to_dict(self) -> dict[str, int]:
        return {
            "actor_seed": self.actor_seed,
            "critic_seed": self.critic_seed,
            "decision_mode_seed": self.decision_mode_seed,
            "decision_q_seed": self.decision_q_seed,
            "master_seed": self.master_seed,
            "replay_seed": self.replay_seed,
            "target_seed": self.target_seed,
            "trainer_actor_seed": self.trainer_actor_seed,
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha({"record": "run4_modeled_rng_plan_v1", **self.to_dict()})


@dataclass(frozen=True, slots=True)
class ModeledSmokeRunnerFactoryV1:
    """Exact constructor which delegates to ModeledCompositeOfflineFactoryV1."""

    modeled_binding: modeled.ModeledCompositeBindingV1
    gamma: float
    freshness_policy_sha256: str
    empirical_scaling_sha256: str
    trainer_config: trainer.TrainerConfigV1
    seed_plan: RunnerSeedPlanV1
    capacity: int = smoke_preregistration.FROZEN_CONFIG.replay_capacity

    def __post_init__(self) -> None:
        if type(self.modeled_binding) is not modeled.ModeledCompositeBindingV1:
            raise ModeledSmokeBindingError("modeled binding has a foreign type")
        self.modeled_binding.__post_init__()
        if type(self.trainer_config) is not trainer.TrainerConfigV1:
            raise ModeledSmokeBindingError("trainer config has a foreign type")
        if type(self.seed_plan) is not RunnerSeedPlanV1:
            raise ModeledSmokeBindingError("seed plan has a foreign type")
        self.seed_plan.__post_init__()
        expected = smoke_preregistration.FROZEN_CONFIG
        observed_config = self.trainer_config
        if (
            observed_config.alpha_d != expected.alpha_d
            or observed_config.alpha_c != expected.alpha_c
            or observed_config.tau != expected.polyak_tau
            or observed_config.actor_lr != expected.actor_learning_rate
            or observed_config.critic_lr != expected.critic_learning_rate
            or observed_config.nominal_batch_size != expected.batch_size
            or observed_config.float_dtype is not torch.float32
        ):
            raise ModeledSmokeBindingError(
                "trainer config differs from smoke preregistration"
            )
        if self.gamma != expected.gamma_per_tensor:
            raise ModeledSmokeBindingError("gamma differs from preregistration")
        if self.capacity != expected.replay_capacity:
            raise ModeledSmokeBindingError(
                "replay capacity differs from preregistration"
            )
        _digest(self.freshness_policy_sha256, "freshness_policy_sha256")
        _digest(self.empirical_scaling_sha256, "empirical_scaling_sha256")

    def build_runner(self) -> runtime.ModeledCompositeOfflineRunnerV1:
        plan = self.seed_plan
        candidate = runtime.ModeledCompositeOfflineFactoryV1.build(
            modeled_binding=self.modeled_binding,
            gamma=self.gamma,
            freshness_policy_sha256=self.freshness_policy_sha256,
            empirical_scaling_sha256=self.empirical_scaling_sha256,
            capacity=self.capacity,
            trainer_config=self.trainer_config,
            actor_seed=plan.actor_seed,
            critic_seed=plan.critic_seed,
            replay_seed=plan.replay_seed,
            target_seed=plan.target_seed,
            trainer_actor_seed=plan.trainer_actor_seed,
        )
        if type(candidate) is not runtime.ModeledCompositeOfflineRunnerV1:
            raise ModeledSmokeBindingError(
                "modeled factory returned a foreign runner"
            )
        self.require_runner(candidate)
        return candidate

    def require_runner(
        self, candidate: runtime.ModeledCompositeOfflineRunnerV1
    ) -> None:
        if type(candidate) is not runtime.ModeledCompositeOfflineRunnerV1:
            raise ModeledSmokeBindingError("runner must be exact modeled runner")
        candidate.binding.require_offline_modeled_training()
        if candidate.binding.modeled_composite_binding_sha256 != (
            self.modeled_binding.canonical_sha256
        ):
            raise ModeledSmokeBindingError("runner modeled binding differs")
        if candidate.binding.freshness_policy_sha256 != self.freshness_policy_sha256:
            raise ModeledSmokeBindingError("runner freshness binding differs")
        if candidate.binding.empirical_scaling_sha256 != (
            self.empirical_scaling_sha256
        ):
            raise ModeledSmokeBindingError("runner scaling binding differs")
        if candidate.replay_buffer.capacity != self.capacity:
            raise ModeledSmokeBindingError("runner replay capacity differs")
        if candidate.trainer.config != self.trainer_config:
            raise ModeledSmokeBindingError("runner trainer config differs")
        tensors = (
            *candidate.model_bundle.actor.parameters(),
            *candidate.model_bundle.actor.buffers(),
            *candidate.model_bundle.critics.parameters(),
            *candidate.model_bundle.critics.buffers(),
        )
        if any(value.device.type != "cpu" for value in tensors):
            raise ModeledSmokeBindingError("modeled smoke is CPU-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "empirical_scaling_sha256": self.empirical_scaling_sha256,
            "freshness_policy_sha256": self.freshness_policy_sha256,
            "gamma": self.gamma,
            "modeled_binding": self.modeled_binding.to_dict(),
            "modeled_binding_sha256": self.modeled_binding.canonical_sha256,
            "seed_plan": self.seed_plan.to_dict(),
            "trainer_config": {
                "actor_lr": self.trainer_config.actor_lr,
                "alpha_c": self.trainer_config.alpha_c,
                "alpha_d": self.trainer_config.alpha_d,
                "critic_lr": self.trainer_config.critic_lr,
                "float_dtype": str(self.trainer_config.float_dtype),
                "nominal_batch_size": self.trainer_config.nominal_batch_size,
                "tau": self.trainer_config.tau,
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha({"record": "run4_modeled_runner_factory_v1", **self.to_dict()})


@dataclass(frozen=True, slots=True)
class ModeledActionRequestV1:
    """Exact action offered to the modeled collector."""

    decision_ordinal: int
    mode_id: int
    q_e4: int
    source: str
    warmup_q_bin_index: Optional[int]

    def __post_init__(self) -> None:
        _exact_nonnegative(self.decision_ordinal, "decision_ordinal")
        if type(self.mode_id) is not int or not 0 <= self.mode_id < 12:
            raise ModeledSmokeBindingError("mode_id must lie in [0,11]")
        if type(self.q_e4) is not int or not 0 <= self.q_e4 <= 9800:
            raise ModeledSmokeBindingError("q_e4 must lie in [0,9800]")
        if self.source not in ("STRATIFIED_WARMUP", "STOCHASTIC_ACTOR"):
            raise ModeledSmokeBindingError("unknown action source")
        if self.source == "STRATIFIED_WARMUP":
            if (
                type(self.warmup_q_bin_index) is not int
                or not 0 <= self.warmup_q_bin_index < 6
            ):
                raise ModeledSmokeBindingError("warm-up action requires q bin")
        elif self.warmup_q_bin_index is not None:
            raise ModeledSmokeBindingError("actor action cannot claim warm-up bin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_ordinal": self.decision_ordinal,
            "mode_id": self.mode_id,
            "q_e4": self.q_e4,
            "source": self.source,
            "warmup_q_bin_index": self.warmup_q_bin_index,
        }


@dataclass(frozen=True, slots=True)
class CollectorCheckpointV1:
    """Canonical, collector-owned durable state and history index."""

    collector_schema_id: str
    collector_binding_sha256: str
    decision_count: int
    transition_sha256s: Tuple[str, ...]
    payload_json: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if type(self.collector_schema_id) is not str or not self.collector_schema_id:
            raise ModeledSmokeCheckpointError("collector schema id is empty")
        _digest(self.collector_binding_sha256, "collector_binding_sha256")
        _exact_nonnegative(self.decision_count, "decision_count")
        if (
            type(self.transition_sha256s) is not tuple
            or len(self.transition_sha256s) != self.decision_count
        ):
            raise ModeledSmokeCheckpointError(
                "collector transition ledger length differs"
            )
        for value in self.transition_sha256s:
            _digest(value, "collector transition digest")
        if len(set(self.transition_sha256s)) != len(self.transition_sha256s):
            raise ModeledSmokeCheckpointError("collector transition digest repeats")
        if type(self.payload_json) is not str:
            raise ModeledSmokeCheckpointError("collector payload must be JSON text")
        try:
            decoded = json.loads(self.payload_json)
        except (TypeError, ValueError) as exc:
            raise ModeledSmokeCheckpointError("collector payload is not JSON") from exc
        canonical = _canonical_bytes(decoded).decode("ascii")
        if canonical != self.payload_json:
            raise ModeledSmokeCheckpointError("collector payload is not canonical")
        _digest(self.payload_sha256, "payload_sha256")
        if hashlib.sha256(self.payload_json.encode("ascii")).hexdigest() != (
            self.payload_sha256
        ):
            raise ModeledSmokeCheckpointError("collector payload digest differs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "collector_binding_sha256": self.collector_binding_sha256,
            "collector_schema_id": self.collector_schema_id,
            "decision_count": self.decision_count,
            "payload_json": self.payload_json,
            "payload_sha256": self.payload_sha256,
            "transition_sha256s": list(self.transition_sha256s),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CollectorCheckpointV1":
        return cls(
            collector_schema_id=value["collector_schema_id"],
            collector_binding_sha256=value["collector_binding_sha256"],
            decision_count=value["decision_count"],
            transition_sha256s=tuple(value["transition_sha256s"]),
            payload_json=value["payload_json"],
            payload_sha256=value["payload_sha256"],
        )


@dataclass(frozen=True, slots=True)
class CollectedModeledTransitionV1:
    """One typed collector result plus causal, fit-only diagnostics."""

    request: ModeledActionRequestV1
    wrapper: modeled.ModeledCompositeOfflineTransitionV1
    state_features: Tuple[float, ...]
    state_features_sha256: str
    duration: int
    terminal: contract.RewardTerminal
    reward: float
    q_perc: Optional[float]
    latency_ms: Optional[float]
    modeled_binding_sha256: str
    mcs_acceptance_result_sha256: str
    mcs_model_binding_sha256: str
    source_partition: str
    validation_evidence_consumed: bool

    def __post_init__(self) -> None:
        if type(self.request) is not ModeledActionRequestV1:
            raise ModeledSmokeBindingError("collector request has a foreign type")
        if type(self.wrapper) is not modeled.ModeledCompositeOfflineTransitionV1:
            raise ModeledSmokeBindingError("collector wrapper has a foreign type")
        self.wrapper.require_attested()
        values = _finite_tuple(self.state_features, "state_features")
        if len(values) != contract.POLICY_FEATURE_COUNT:
            raise ModeledSmokeBindingError("collector state width differs")
        _digest(self.state_features_sha256, "state_features_sha256")
        if _sha(list(values)) != self.state_features_sha256:
            raise ModeledSmokeBindingError("collector state digest differs")
        if type(self.duration) is not int or self.duration != 2:
            raise ModeledSmokeBindingError("Run-4 modeled duration must be d=2")
        if type(self.terminal) is not contract.RewardTerminal:
            raise ModeledSmokeBindingError("terminal has a foreign type")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise ModeledSmokeBindingError("reward is not real")
        if not math.isfinite(float(self.reward)):
            raise ModeledSmokeBindingError("reward is not finite")
        if self.wrapper.terminal is not self.terminal:
            raise ModeledSmokeBindingError("diagnostic terminal differs from wrapper")
        if float(self.wrapper.reward) != float(self.reward):
            raise ModeledSmokeBindingError("diagnostic reward differs from wrapper")
        for value, name in (
            (self.modeled_binding_sha256, "modeled_binding_sha256"),
            (self.mcs_acceptance_result_sha256, "mcs_acceptance_result_sha256"),
            (self.mcs_model_binding_sha256, "mcs_model_binding_sha256"),
        ):
            _digest(value, name)
        if self.source_partition != FIT_PARTITION_LABEL:
            raise ModeledSmokeBindingError("collector consumed non-fit evidence")
        if type(self.validation_evidence_consumed) is not bool or (
            self.validation_evidence_consumed
        ):
            raise ModeledSmokeBindingError("validation evidence was consumed")

    def ledger_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "latency_ms": self.latency_ms,
            "mcs_acceptance_result_sha256": self.mcs_acceptance_result_sha256,
            "mcs_model_binding_sha256": self.mcs_model_binding_sha256,
            "modeled_binding_sha256": self.modeled_binding_sha256,
            "q_perc": self.q_perc,
            "request": self.request.to_dict(),
            "reward": float(self.reward),
            "source_partition": self.source_partition,
            "state_features_sha256": self.state_features_sha256,
            "terminal": self.terminal.value,
            "transition_sha256": self.wrapper.transition_sha256,
            "validation_evidence_consumed": self.validation_evidence_consumed,
        }


class ModeledTransitionCollectorV1(Protocol):
    """Public protocol required by the modeled smoke orchestrator."""

    @property
    def collector_binding_sha256(self) -> str: ...

    @property
    def decision_count(self) -> int: ...

    def current_state_features(self) -> Tuple[float, ...]: ...

    def collect(
        self, request: ModeledActionRequestV1
    ) -> CollectedModeledTransitionV1: ...

    def checkpoint(self) -> CollectorCheckpointV1: ...

    def restore(self, checkpoint: CollectorCheckpointV1) -> None: ...

    def history(self) -> Tuple[CollectedModeledTransitionV1, ...]: ...


CollectorFactory = Callable[[], ModeledTransitionCollectorV1]


@dataclass(frozen=True, slots=True)
class PreflightFeatureRequirementV1:
    """Evidence-bound lower bound for one warm-up state feature.

    No scientific threshold is supplied by this orchestration module. The
    caller must bind every lower bound to preregistration or calibration
    evidence used by its collector.
    """

    feature_name: str
    minimum_distinct_count: int
    minimum_span: float

    def __post_init__(self) -> None:
        if self.feature_name not in contract.POLICY_FEATURE_ORDER:
            raise ModeledSmokeBindingError("unknown preflight feature name")
        if (
            type(self.minimum_distinct_count) is not int
            or self.minimum_distinct_count < 2
        ):
            raise ModeledSmokeBindingError(
                "minimum_distinct_count must be an exact int >= 2"
            )
        if isinstance(self.minimum_span, bool) or not isinstance(
            self.minimum_span, (int, float)
        ):
            raise ModeledSmokeBindingError("minimum_span must be real")
        if not math.isfinite(float(self.minimum_span)) or self.minimum_span <= 0:
            raise ModeledSmokeBindingError("minimum_span must be finite and > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "minimum_distinct_count": self.minimum_distinct_count,
            "minimum_span": float(self.minimum_span),
        }


@dataclass(frozen=True, slots=True)
class PreflightVariationContractV1:
    """Explicit evidence binding for state-variation acceptance.

    The four causal scene/radio inputs are mandatory. Previous-action and
    previous-outcome fields are governed by exact structural reconciliation,
    not arbitrary numerical spans.
    """

    contract_id: str
    contract_version: int
    evidence_sha256: str
    feature_schema_sha256: str
    source_partition: str
    requirements: Tuple[PreflightFeatureRequirementV1, ...]

    def __post_init__(self) -> None:
        if type(self.contract_id) is not str or not self.contract_id:
            raise ModeledSmokeBindingError("preflight contract id is empty")
        if type(self.contract_version) is not int or self.contract_version < 1:
            raise ModeledSmokeBindingError("preflight contract version is invalid")
        _digest(self.evidence_sha256, "preflight evidence_sha256")
        _digest(self.feature_schema_sha256, "feature_schema_sha256")
        if self.feature_schema_sha256 != contract.FEATURE_SCHEMA_SHA256:
            raise ModeledSmokeBindingError("preflight feature schema differs")
        if self.source_partition != FIT_PARTITION_LABEL:
            raise ModeledSmokeBindingError("preflight contract is not fit-only")
        if type(self.requirements) is not tuple or any(
            type(item) is not PreflightFeatureRequirementV1
            for item in self.requirements
        ):
            raise ModeledSmokeBindingError("preflight requirements have foreign type")
        required_names = contract.POLICY_FEATURE_ORDER[:4]
        observed_names = tuple(item.feature_name for item in self.requirements)
        if observed_names != required_names:
            raise ModeledSmokeBindingError(
                "preflight must bind SI, P40, prior MCS and pre-action backlog "
                "in feature-schema order"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "evidence_sha256": self.evidence_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "requirements": [item.to_dict() for item in self.requirements],
            "source_partition": self.source_partition,
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha(
            {"record": "run4_preflight_variation_contract_v1", **self.to_dict()}
        )


@dataclass(frozen=True, slots=True)
class FeatureDiagnosticV1:
    """Exact finite-population diagnostic for one of all 21 actor features."""

    feature_name: str
    feature_index: int
    sample_count: int
    finite_count: int
    distinct_count: int
    minimum: float
    maximum: float
    span: float
    ordered_values_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.feature_index) is not int
            or not 0 <= self.feature_index < contract.POLICY_FEATURE_COUNT
            or self.feature_name
            != contract.POLICY_FEATURE_ORDER[self.feature_index]
        ):
            raise ModeledSmokePreflightError("feature diagnostic identity differs")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ModeledSmokePreflightError("feature diagnostic is empty")
        if self.finite_count != self.sample_count:
            raise ModeledSmokePreflightError(
                "feature diagnostic contains non-finite data"
            )
        if (
            type(self.distinct_count) is not int
            or not 1 <= self.distinct_count <= self.sample_count
        ):
            raise ModeledSmokePreflightError("feature distinct count is invalid")
        values = (float(self.minimum), float(self.maximum), float(self.span))
        if not all(math.isfinite(item) for item in values):
            raise ModeledSmokePreflightError("feature diagnostic is non-finite")
        if self.maximum < self.minimum or self.span != self.maximum - self.minimum:
            raise ModeledSmokePreflightError("feature diagnostic span differs")
        _digest(self.ordered_values_sha256, "ordered_values_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class PreviousFeatureReconciliationV1:
    """Population accounting for exact prior-action/outcome reconciliation."""

    genesis_without_previous_count: int
    reconciled_previous_count: int
    previous_success_count: int
    previous_failure_count: int

    def __post_init__(self) -> None:
        expected = smoke_preregistration.FROZEN_CONFIG.warmup_decision_count
        if self.genesis_without_previous_count != 1:
            raise ModeledSmokePreflightError("preflight must have one genesis row")
        if self.reconciled_previous_count != expected - 1:
            raise ModeledSmokePreflightError("previous-state chain is incomplete")
        if self.previous_success_count < 1 or self.previous_failure_count < 1:
            raise ModeledSmokePreflightError(
                "previous-state chain must expose both success and failure"
            )
        if (
            self.previous_success_count + self.previous_failure_count
            != self.reconciled_previous_count
        ):
            raise ModeledSmokePreflightError("previous outcome accounting differs")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class WarmupPreflightReportV1:
    """Immutable, auditable result of the exact no-gradient 288 gate."""

    decision_count: int
    coverage_counts: Tuple[Tuple[int, int, int], ...]
    success_count: int
    failure_count: int
    unique_state_count: int
    unique_reward_count: int
    variation_contract: PreflightVariationContractV1
    feature_diagnostics: Tuple[FeatureDiagnosticV1, ...]
    previous_reconciliation: PreviousFeatureReconciliationV1
    transition_ledger_sha256: str
    schedule_id: str
    runner_binding_sha256: str
    modeled_binding_sha256: str
    mcs_acceptance_result_sha256: str
    mcs_model_binding_sha256: str
    collector_binding_sha256: str
    collector_checkpoint_sha256: str
    model_before_sha256: str
    model_after_sha256: str
    optimizer_before_sha256: str
    optimizer_after_sha256: str
    trainer_update_count: int
    replay_accepted_count: int
    replay_seen_digest_count: int
    replay_seen_identity_count: int
    duration: int
    validation_evidence_consumed: bool
    gradient_free: bool
    passed: bool

    def __post_init__(self) -> None:
        expected = smoke_preregistration.FROZEN_CONFIG
        if self.decision_count != expected.warmup_decision_count:
            raise ModeledSmokePreflightError("preflight is not exactly 288 rows")
        expected_cells = tuple(
            (mode, q_bin, expected.warmup_samples_per_mode_q_bin)
            for mode in range(expected.warmup_mode_count)
            for q_bin in range(expected.warmup_q_bin_count)
        )
        if self.coverage_counts != expected_cells:
            raise ModeledSmokePreflightError("12x6x4 coverage is incomplete")
        if self.success_count < 1 or self.failure_count < 1:
            raise ModeledSmokePreflightError(
                "warm-up must contain both success and registered failure/timeout"
            )
        if self.unique_state_count < 2 or self.unique_reward_count < 2:
            raise ModeledSmokePreflightError(
                "warm-up state/outcome population has no finite variation"
            )
        if type(self.variation_contract) is not PreflightVariationContractV1:
            raise ModeledSmokePreflightError("variation contract has foreign type")
        if (
            type(self.feature_diagnostics) is not tuple
            or tuple(item.feature_name for item in self.feature_diagnostics)
            != contract.POLICY_FEATURE_ORDER
            or any(
                type(item) is not FeatureDiagnosticV1
                for item in self.feature_diagnostics
            )
        ):
            raise ModeledSmokePreflightError(
                "21-D feature diagnostics are incomplete"
            )
        diagnostics = {
            item.feature_name: item for item in self.feature_diagnostics
        }
        for requirement in self.variation_contract.requirements:
            observed = diagnostics[requirement.feature_name]
            if observed.distinct_count < requirement.minimum_distinct_count:
                raise ModeledSmokePreflightError(
                    f"{requirement.feature_name} distinct-count gate failed"
                )
            if observed.span < requirement.minimum_span:
                raise ModeledSmokePreflightError(
                    f"{requirement.feature_name} span gate failed"
                )
        if type(self.previous_reconciliation) is not PreviousFeatureReconciliationV1:
            raise ModeledSmokePreflightError(
                "previous reconciliation has foreign type"
            )
        if self.trainer_update_count != 0:
            raise ModeledSmokePreflightError("preflight took a gradient")
        if (
            self.replay_accepted_count != self.decision_count
            or self.replay_seen_digest_count != self.decision_count
            or self.replay_seen_identity_count != self.decision_count
        ):
            raise ModeledSmokePreflightError("replay accounting is not exact")
        if self.duration != 2:
            raise ModeledSmokePreflightError("preflight duration differs from d=2")
        if self.validation_evidence_consumed:
            raise ModeledSmokePreflightError("validation evidence entered preflight")
        if self.model_before_sha256 != self.model_after_sha256:
            raise ModeledSmokePreflightError("model changed during no-gradient gate")
        if self.optimizer_before_sha256 != self.optimizer_after_sha256:
            raise ModeledSmokePreflightError(
                "optimizer changed during no-gradient gate"
            )
        for value, name in (
            (self.transition_ledger_sha256, "transition_ledger_sha256"),
            (self.runner_binding_sha256, "runner_binding_sha256"),
            (self.modeled_binding_sha256, "modeled_binding_sha256"),
            (self.mcs_acceptance_result_sha256, "mcs_acceptance_result_sha256"),
            (self.mcs_model_binding_sha256, "mcs_model_binding_sha256"),
            (self.collector_binding_sha256, "collector_binding_sha256"),
            (self.collector_checkpoint_sha256, "collector_checkpoint_sha256"),
        ):
            _digest(value, name)
        if not self.gradient_free or not self.passed:
            raise ModeledSmokePreflightError("preflight report is not a pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "collector_binding_sha256": self.collector_binding_sha256,
            "collector_checkpoint_sha256": self.collector_checkpoint_sha256,
            "coverage_counts": [list(item) for item in self.coverage_counts],
            "decision_count": self.decision_count,
            "duration": self.duration,
            "failure_count": self.failure_count,
            "feature_diagnostics": [
                item.to_dict() for item in self.feature_diagnostics
            ],
            "gradient_free": self.gradient_free,
            "mcs_acceptance_result_sha256": self.mcs_acceptance_result_sha256,
            "mcs_model_binding_sha256": self.mcs_model_binding_sha256,
            "modeled_binding_sha256": self.modeled_binding_sha256,
            "model_after_sha256": self.model_after_sha256,
            "model_before_sha256": self.model_before_sha256,
            "optimizer_after_sha256": self.optimizer_after_sha256,
            "optimizer_before_sha256": self.optimizer_before_sha256,
            "passed": self.passed,
            "previous_reconciliation": self.previous_reconciliation.to_dict(),
            "replay_accepted_count": self.replay_accepted_count,
            "replay_seen_digest_count": self.replay_seen_digest_count,
            "replay_seen_identity_count": self.replay_seen_identity_count,
            "runner_binding_sha256": self.runner_binding_sha256,
            "schedule_id": self.schedule_id,
            "schema": PREFLIGHT_SCHEMA_ID,
            "success_count": self.success_count,
            "trainer_update_count": self.trainer_update_count,
            "transition_ledger_sha256": self.transition_ledger_sha256,
            "unique_reward_count": self.unique_reward_count,
            "unique_state_count": self.unique_state_count,
            "validation_evidence_consumed": self.validation_evidence_consumed,
            "variation_contract": self.variation_contract.to_dict(),
            "variation_contract_sha256": self.variation_contract.canonical_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha(self.to_dict())


@dataclass(frozen=True, slots=True)
class BoundaryFingerprintV1:
    """Hash-complete smoke boundary used by event-sourced restore."""

    update_count: int
    decision_count: int
    actor_sha256: str
    critics_sha256: str
    actor_optimizer_sha256: str
    critic_optimizer_sha256: str
    decision_q_rng_sha256: str
    decision_mode_rng_sha256: str
    runner_rng_reconstruction_sha256: str
    replay_resident_sha256: str
    replay_accepted_count: int
    replay_evicted_count: int
    replay_seen_digest_count: int
    replay_seen_identity_count: int
    collector_checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "actor_sha256",
            "critics_sha256",
            "actor_optimizer_sha256",
            "critic_optimizer_sha256",
            "decision_q_rng_sha256",
            "decision_mode_rng_sha256",
            "runner_rng_reconstruction_sha256",
            "replay_resident_sha256",
            "collector_checkpoint_sha256",
        ):
            _digest(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha({"record": "run4_modeled_boundary_v1", **self.to_dict()})


@dataclass(frozen=True, slots=True)
class ActionLedgerRowV1:
    ordinal: int
    action: ModeledActionRequestV1
    transition_sha256: str
    state_features_sha256: str
    terminal: str
    reward: float
    duration: int

    def __post_init__(self) -> None:
        if self.ordinal != self.action.decision_ordinal:
            raise ModeledSmokeCheckpointError("ledger ordinal differs")
        _digest(self.transition_sha256, "transition_sha256")
        _digest(self.state_features_sha256, "state_features_sha256")
        if self.terminal not in tuple(item.value for item in contract.RewardTerminal):
            raise ModeledSmokeCheckpointError("unknown ledger terminal")
        if not math.isfinite(float(self.reward)):
            raise ModeledSmokeCheckpointError("ledger reward is not finite")
        if self.duration != 2:
            raise ModeledSmokeCheckpointError("ledger duration differs from d=2")

    @classmethod
    def from_collected(
        cls, collected: CollectedModeledTransitionV1
    ) -> "ActionLedgerRowV1":
        return cls(
            ordinal=collected.request.decision_ordinal,
            action=collected.request,
            transition_sha256=collected.wrapper.transition_sha256,
            state_features_sha256=collected.state_features_sha256,
            terminal=collected.terminal.value,
            reward=float(collected.reward),
            duration=collected.duration,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "duration": self.duration,
            "ordinal": self.ordinal,
            "reward": self.reward,
            "state_features_sha256": self.state_features_sha256,
            "terminal": self.terminal,
            "transition_sha256": self.transition_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionLedgerRowV1":
        raw_action = value["action"]
        action = ModeledActionRequestV1(
            decision_ordinal=raw_action["decision_ordinal"],
            mode_id=raw_action["mode_id"],
            q_e4=raw_action["q_e4"],
            source=raw_action["source"],
            warmup_q_bin_index=raw_action["warmup_q_bin_index"],
        )
        return cls(
            ordinal=value["ordinal"],
            action=action,
            transition_sha256=value["transition_sha256"],
            state_features_sha256=value["state_features_sha256"],
            terminal=value["terminal"],
            reward=value["reward"],
            duration=value["duration"],
        )


@dataclass(frozen=True, slots=True)
class ModeledSmokeCheckpointV1:
    """Durable event-sourced checkpoint; no private replay snapshot needed."""

    update_count: int
    decision_count: int
    factory_sha256: str
    schedule_id: str
    preregistration_sha256: str
    seed_plan_sha256: str
    modeled_binding_document: Mapping[str, Any]
    replay_binding_document: Mapping[str, Any]
    collector_checkpoint: CollectorCheckpointV1
    ledger: Tuple[ActionLedgerRowV1, ...]
    preflight_report: WarmupPreflightReportV1
    boundary: BoundaryFingerprintV1

    def __post_init__(self) -> None:
        schedule = smoke_preregistration.FROZEN_CONFIG
        if self.update_count not in CHECKPOINT_UPDATES:
            raise ModeledSmokeCheckpointError("checkpoint update is not registered")
        expected_decisions = schedule.warmup_decision_count + (
            schedule.environment_transitions_per_update * self.update_count
        )
        if self.decision_count != expected_decisions:
            raise ModeledSmokeCheckpointError("checkpoint decision count differs")
        if len(self.ledger) != self.decision_count:
            raise ModeledSmokeCheckpointError("checkpoint ledger length differs")
        if tuple(item.ordinal for item in self.ledger) != tuple(
            range(self.decision_count)
        ):
            raise ModeledSmokeCheckpointError("checkpoint ledger is not contiguous")
        if self.collector_checkpoint.decision_count != self.decision_count:
            raise ModeledSmokeCheckpointError("collector checkpoint count differs")
        if tuple(item.transition_sha256 for item in self.ledger) != (
            self.collector_checkpoint.transition_sha256s
        ):
            raise ModeledSmokeCheckpointError("collector/ledger transitions differ")
        if self.boundary.update_count != self.update_count or (
            self.boundary.decision_count != self.decision_count
        ):
            raise ModeledSmokeCheckpointError("boundary position differs")
        for value, name in (
            (self.factory_sha256, "factory_sha256"),
            (self.preregistration_sha256, "preregistration_sha256"),
            (self.seed_plan_sha256, "seed_plan_sha256"),
        ):
            _digest(value, name)
        if self.preregistration_sha256 != (
            smoke_preregistration.PREREGISTRATION_SHA256
        ):
            raise ModeledSmokeCheckpointError("preregistration digest differs")
        if type(self.preflight_report) is not WarmupPreflightReportV1:
            raise ModeledSmokeCheckpointError("preflight report has foreign type")
        if (
            self.preflight_report.variation_contract.source_partition
            != FIT_PARTITION_LABEL
        ):
            raise ModeledSmokeCheckpointError("preflight binding is not fit-only")
        if type(self.modeled_binding_document) is not MappingProxyType:
            object.__setattr__(
                self,
                "modeled_binding_document",
                MappingProxyType(dict(self.modeled_binding_document)),
            )
        if type(self.replay_binding_document) is not MappingProxyType:
            object.__setattr__(
                self,
                "replay_binding_document",
                MappingProxyType(dict(self.replay_binding_document)),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "collector_checkpoint": self.collector_checkpoint.to_dict(),
            "decision_count": self.decision_count,
            "factory_sha256": self.factory_sha256,
            "ledger": [item.to_dict() for item in self.ledger],
            "modeled_binding_document": dict(self.modeled_binding_document),
            "preflight_report": self.preflight_report.to_dict(),
            "preregistration_sha256": self.preregistration_sha256,
            "replay_binding_document": dict(self.replay_binding_document),
            "schedule_id": self.schedule_id,
            "schema": CHECKPOINT_SCHEMA_ID,
            "seed_plan_sha256": self.seed_plan_sha256,
            "update_count": self.update_count,
        }

    @property
    def canonical_sha256(self) -> str:
        return _sha(self.to_dict())


@dataclass(frozen=True, slots=True)
class ModeledSmokeCheckpointEventV1:
    update: int
    checkpoint: ModeledSmokeCheckpointV1
    latest_metrics: Optional[trainer.UpdateMetricsV1]

    def __post_init__(self) -> None:
        if self.update != self.checkpoint.update_count:
            raise ModeledSmokeCheckpointError("event/checkpoint update differs")
        if self.update == 0 and self.latest_metrics is not None:
            raise ModeledSmokeCheckpointError("update zero has no gradient metrics")
        if self.update > 0:
            if type(self.latest_metrics) is not trainer.UpdateMetricsV1:
                raise ModeledSmokeCheckpointError("gradient checkpoint lacks metrics")
            self.latest_metrics.require_finite()
            if self.latest_metrics.update_index != self.update:
                raise ModeledSmokeCheckpointError("metric update differs")


@dataclass(frozen=True, slots=True)
class ModeledSmokeSummaryV1:
    starting_update: int
    final_update: int
    final_decision_count: int
    emitted_checkpoint_updates: Tuple[int, ...]
    preflight_report_sha256: str
    final_checkpoint_sha256: str


def build_frozen_warmup_schedule(
    master_seed: int = smoke_preregistration.FROZEN_CONFIG.initial_smoke_seed,
) -> exploration.StratifiedWarmupSchedule:
    if type(master_seed) is not int or (
        master_seed not in smoke_preregistration.FROZEN_CONFIG.seed_order
    ):
        raise ModeledSmokeScheduleError("warm-up seed is not registered")
    config = exploration.WarmupScheduleConfig(
        mode_q_e4_bounds=MODELED_SMOKE_SUPPORT.mode_q_e4_bounds,
        q_bin_count=smoke_preregistration.FROZEN_CONFIG.warmup_q_bin_count,
        samples_per_q_bin=(
            smoke_preregistration.FROZEN_CONFIG.warmup_samples_per_mode_q_bin
        ),
        master_seed=master_seed,
        support_contract_id=MODELED_SMOKE_SUPPORT_SHA256,
    )
    result = exploration.StratifiedWarmupSchedule(config)
    if len(result) != smoke_preregistration.FROZEN_CONFIG.warmup_decision_count:
        raise ModeledSmokeScheduleError("frozen warm-up is not 288 decisions")
    return result


class ModeledSmokeOrchestratorV1:
    """Exact bounded modeled smoke with preflight and event-sourced restore."""

    def __init__(
        self,
        *,
        runner_factory: ModeledSmokeRunnerFactoryV1,
        collector_factory: CollectorFactory,
        preflight_variation_contract: PreflightVariationContractV1,
    ) -> None:
        if type(runner_factory) is not ModeledSmokeRunnerFactoryV1:
            raise ModeledSmokeBindingError("runner_factory has a foreign type")
        if not callable(collector_factory):
            raise ModeledSmokeBindingError("collector_factory must be callable")
        if type(preflight_variation_contract) is not PreflightVariationContractV1:
            raise ModeledSmokeBindingError(
                "preflight_variation_contract has a foreign type"
            )
        if torch.get_num_threads() != (
            smoke_preregistration.FROZEN_CONFIG.torch_intraop_threads
        ):
            raise ModeledSmokeBindingError(
                "torch intra-op threads must be exactly 4"
            )
        self.runner_factory = runner_factory
        self.collector_factory = collector_factory
        self.preflight_variation_contract = preflight_variation_contract
        self.schedule = build_frozen_warmup_schedule(
            runner_factory.seed_plan.master_seed
        )
        self.runner = runner_factory.build_runner()
        self.collector = collector_factory()
        self._require_collector(self.collector)
        plan = runner_factory.seed_plan
        self._decision_q_generator = torch.Generator(device="cpu")
        self._decision_q_generator.manual_seed(plan.decision_q_seed)
        self._decision_mode_generator = torch.Generator(device="cpu")
        self._decision_mode_generator.manual_seed(plan.decision_mode_seed)
        self._ledger: list[ActionLedgerRowV1] = []
        self._history: list[CollectedModeledTransitionV1] = []
        self._preflight_report: Optional[WarmupPreflightReportV1] = None
        self._model_before = self._model_sha256()
        self._optimizer_before = self._optimizer_sha256()

    @property
    def decision_count(self) -> int:
        return len(self._ledger)

    @property
    def update_count(self) -> int:
        return self.runner.trainer.update_count

    def _require_collector(self, collector: object) -> None:
        required = (
            "collector_binding_sha256",
            "decision_count",
            "current_state_features",
            "collect",
            "checkpoint",
            "restore",
            "history",
        )
        if any(not hasattr(collector, name) for name in required):
            raise ModeledSmokeBindingError("collector does not implement protocol")
        _digest(collector.collector_binding_sha256, "collector binding")
        if collector.decision_count != 0:
            raise ModeledSmokeBindingError("fresh collector is not at genesis")
        values = _finite_tuple(
            collector.current_state_features(), "collector current state"
        )
        if len(values) != contract.POLICY_FEATURE_COUNT:
            raise ModeledSmokeBindingError("collector current state width differs")

    def _model_sha256(self) -> str:
        return _sha(
            {
                "actor": _tree_document(self.runner.model_bundle.actor.state_dict()),
                "critics": _tree_document(
                    self.runner.model_bundle.critics.state_dict()
                ),
            }
        )

    def _optimizer_sha256(self) -> str:
        return _sha(
            {
                "actor": _tree_document(
                    self.runner.trainer.actor_optimizer.state_dict()
                ),
                "critic": _tree_document(
                    self.runner.trainer.critic_optimizer.state_dict()
                ),
            }
        )

    def _execution(self, mode_id: int, q_e4: int) -> ExecutedActionIdentity:
        catalog = load_contract()
        executable = catalog.resolve(mode_id, q_e4 / float(Q_E4_SCALE))
        action = ExecutedActionIdentity.from_executable_action(
            executable, catalog
        )
        if (action.mode_id, action.q_e4) != (mode_id, q_e4):
            raise ModeledSmokeScheduleError("catalog changed selected action")
        return action

    def _warmup_request(self, ordinal: int) -> ModeledActionRequestV1:
        selected = self.schedule.action_at(ordinal)
        return ModeledActionRequestV1(
            decision_ordinal=ordinal,
            mode_id=selected.mode_id,
            q_e4=selected.q_e4,
            source="STRATIFIED_WARMUP",
            warmup_q_bin_index=selected.q_bin_index,
        )

    def _actor_request(
        self, ordinal: int, state_features: Tuple[float, ...]
    ) -> ModeledActionRequestV1:
        state = torch.tensor((state_features,), dtype=torch.float32)
        with torch.no_grad():
            sample = self.runner.model_bundle.actor.sample_all_modes(
                state, generator=self._decision_q_generator
            )
            mode_id = int(
                torch.multinomial(
                    sample.probs[0],
                    1,
                    generator=self._decision_mode_generator,
                )[0]
            )
            q_e4 = int(sample.q_e4[0, mode_id])
        self._execution(mode_id, q_e4)
        return ModeledActionRequestV1(
            decision_ordinal=ordinal,
            mode_id=mode_id,
            q_e4=q_e4,
            source="STOCHASTIC_ACTOR",
            warmup_q_bin_index=None,
        )

    def _next_request(self) -> ModeledActionRequestV1:
        ordinal = self.decision_count
        if ordinal < len(self.schedule):
            return self._warmup_request(ordinal)
        state_features = _finite_tuple(
            self.collector.current_state_features(), "collector current state"
        )
        return self._actor_request(ordinal, state_features)

    def _validate_collected(
        self,
        collected: CollectedModeledTransitionV1,
        expected: ModeledActionRequestV1,
    ) -> None:
        if type(collected) is not CollectedModeledTransitionV1:
            raise ModeledSmokeBindingError("collector returned a foreign record")
        if collected.request != expected:
            raise ModeledSmokeBindingError("collector substituted the request")
        binding = self.runner.binding
        binding.assert_wrapper(collected.wrapper)
        if collected.modeled_binding_sha256 != (
            binding.modeled_composite_binding_sha256
        ):
            raise ModeledSmokeBindingError("collector modeled binding differs")
        if collected.mcs_acceptance_result_sha256 != (
            binding.mcs_acceptance_result_sha256
        ):
            raise ModeledSmokeBindingError("collector MCS acceptance differs")
        if collected.mcs_model_binding_sha256 != binding.mcs_model_binding_sha256:
            raise ModeledSmokeBindingError("collector MCS model differs")
        # The wrapper's existing package-private seam is the sole supported
        # hand-off to dedicated modeled replay.  This is not a general export.
        transition = collected.wrapper._sealed_transition_for_modeled_replay(
            binding.modeled_composite_binding_sha256
        )
        transition.require_attested()
        if (transition.action.mode_id, transition.action.q_e4) != (
            expected.mode_id,
            expected.q_e4,
        ):
            raise ModeledSmokeBindingError("executed action differs from request")
        if transition.duration != 2 or collected.duration != 2:
            raise ModeledSmokeBindingError("transition duration differs from d=2")
        values = tuple(float(item) for item in transition.state_features.as_tuple())
        if values != collected.state_features:
            raise ModeledSmokeBindingError("collector state differs from transition")
        if _sha(list(values)) != collected.state_features_sha256:
            raise ModeledSmokeBindingError("collector state hash differs")

        named = dict(zip(contract.POLICY_FEATURE_ORDER, values))
        previous = transition.state.state.previous
        one_hot_names = contract.POLICY_FEATURE_ORDER[4:16]
        if expected.decision_ordinal == 0:
            if previous is not None:
                raise ModeledSmokeBindingError("genesis state carries previous")
            if any(named[name] != 0.0 for name in one_hot_names) or any(
                named[name] != 0.0
                for name in (
                    "prev_q_normalized",
                    "prev_quality_qperc",
                    "prev_latency_normalized",
                    "prev_present",
                    "prev_success",
                )
            ):
                raise ModeledSmokeBindingError(
                    "genesis previous features are not all zero"
                )
            return

        if len(self._history) != expected.decision_ordinal:
            raise ModeledSmokeBindingError("previous-history position differs")
        if previous is None:
            raise ModeledSmokeBindingError("non-genesis state omits previous")
        prior = self._history[-1]
        if (previous.action.mode_id, previous.action.q_e4) != (
            prior.request.mode_id,
            prior.request.q_e4,
        ):
            raise ModeledSmokeBindingError("previous action differs from prior row")
        if previous.terminal is not prior.terminal:
            raise ModeledSmokeBindingError(
                "previous terminal differs from prior row"
            )
        if previous.q_perc != prior.q_perc or previous.latency_ms != prior.latency_ms:
            raise ModeledSmokeBindingError(
                "previous quality/latency differs from prior row"
            )
        for mode_id, name in enumerate(one_hot_names):
            expected_value = 1.0 if mode_id == prior.request.mode_id else 0.0
            if named[name] != expected_value:
                raise ModeledSmokeBindingError(
                    "previous-action one-hot differs from prior row"
                )
        if named["prev_q_normalized"] != prior.request.q_e4 / 9800.0:
            raise ModeledSmokeBindingError("previous q differs from prior row")
        success = prior.terminal is contract.RewardTerminal.SUCCESS
        if named["prev_present"] != 1.0 or named["prev_success"] != (
            1.0 if success else 0.0
        ):
            raise ModeledSmokeBindingError("previous presence/success differs")
        expected_quality = float(prior.q_perc) if success else 0.0
        expected_latency = float(prior.latency_ms) / 170.0 if success else 0.0
        if named["prev_quality_qperc"] != expected_quality:
            raise ModeledSmokeBindingError("previous quality feature differs")
        if named["prev_latency_normalized"] != expected_latency:
            raise ModeledSmokeBindingError("previous latency feature differs")
        if not success and (prior.q_perc is not None or prior.latency_ms is not None):
            raise ModeledSmokeBindingError(
                "failed prior row fabricated quality or latency"
            )

    def collect_one(self) -> CollectedModeledTransitionV1:
        expected = self._next_request()
        collected = self.collector.collect(expected)
        self._validate_collected(collected, expected)
        self.runner.ingest(collected.wrapper)
        self._history.append(collected)
        self._ledger.append(ActionLedgerRowV1.from_collected(collected))
        if self.collector.decision_count != self.decision_count:
            raise ModeledSmokeScheduleError("collector/orchestrator count differs")
        return collected


    def _feature_diagnostics(self) -> Tuple[FeatureDiagnosticV1, ...]:
        columns = tuple(
            tuple(float(item.state_features[index]) for item in self._history)
            for index in range(contract.POLICY_FEATURE_COUNT)
        )
        result = []
        for index, values in enumerate(columns):
            finite_count = sum(math.isfinite(value) for value in values)
            minimum = min(values)
            maximum = max(values)
            result.append(
                FeatureDiagnosticV1(
                    feature_name=contract.POLICY_FEATURE_ORDER[index],
                    feature_index=index,
                    sample_count=len(values),
                    finite_count=finite_count,
                    distinct_count=len(set(values)),
                    minimum=minimum,
                    maximum=maximum,
                    span=maximum - minimum,
                    ordered_values_sha256=_sha(list(values)),
                )
            )
        return tuple(result)

    def _previous_reconciliation(self) -> PreviousFeatureReconciliationV1:
        genesis = 0
        reconciled = 0
        successes = 0
        failures = 0
        binding = self.runner.binding.modeled_composite_binding_sha256
        for ordinal, item in enumerate(self._history):
            transition = item.wrapper._sealed_transition_for_modeled_replay(binding)
            previous = transition.state.state.previous
            if ordinal == 0:
                genesis += int(previous is None)
                continue
            if previous is None:
                raise ModeledSmokePreflightError("previous chain is incomplete")
            reconciled += 1
            if previous.success:
                successes += 1
            else:
                failures += 1
        return PreviousFeatureReconciliationV1(
            genesis_without_previous_count=genesis,
            reconciled_previous_count=reconciled,
            previous_success_count=successes,
            previous_failure_count=failures,
        )

    def _build_preflight_report(self) -> WarmupPreflightReportV1:
        expected = smoke_preregistration.FROZEN_CONFIG.warmup_decision_count
        if self.decision_count != expected or self.update_count != 0:
            raise ModeledSmokePreflightError(
                "preflight requires exactly 288 rows and zero updates"
            )
        counts = {
            (mode, q_bin): 0
            for mode in range(12)
            for q_bin in range(6)
        }
        successes = 0
        failures = 0
        states = set()
        rewards = set()
        for item in self._history:
            q_bin = item.request.warmup_q_bin_index
            if q_bin is None:
                raise ModeledSmokePreflightError("actor row entered warm-up")
            counts[(item.request.mode_id, q_bin)] += 1
            states.add(item.state_features_sha256)
            rewards.add(float(item.reward))
            if item.terminal is contract.RewardTerminal.SUCCESS:
                successes += 1
            elif item.terminal in (
                contract.RewardTerminal.REGISTERED_DELIVERY_FAILURE,
                contract.RewardTerminal.REGISTERED_SERVICE_FAILURE,
                contract.RewardTerminal.TIMEOUT,
            ):
                failures += 1
            else:
                raise ModeledSmokePreflightError(
                    "infrastructure/evaluator fault entered learning warm-up"
                )
        collector_checkpoint = self.collector.checkpoint()
        self._validate_collector_checkpoint(collector_checkpoint)
        return WarmupPreflightReportV1(
            decision_count=expected,
            coverage_counts=tuple(
                (mode, q_bin, counts[(mode, q_bin)])
                for mode in range(12)
                for q_bin in range(6)
            ),
            success_count=successes,
            failure_count=failures,
            unique_state_count=len(states),
            unique_reward_count=len(rewards),
            variation_contract=self.preflight_variation_contract,
            feature_diagnostics=self._feature_diagnostics(),
            previous_reconciliation=self._previous_reconciliation(),
            transition_ledger_sha256=_sha(
                [item.to_dict() for item in self._ledger]
            ),
            schedule_id=self.schedule.config.schedule_id,
            runner_binding_sha256=self.runner.binding.canonical_sha256,
            modeled_binding_sha256=(
                self.runner.binding.modeled_composite_binding_sha256
            ),
            mcs_acceptance_result_sha256=(
                self.runner.binding.mcs_acceptance_result_sha256
            ),
            mcs_model_binding_sha256=self.runner.binding.mcs_model_binding_sha256,
            collector_binding_sha256=self.collector.collector_binding_sha256,
            collector_checkpoint_sha256=_sha(collector_checkpoint.to_dict()),
            model_before_sha256=self._model_before,
            model_after_sha256=self._model_sha256(),
            optimizer_before_sha256=self._optimizer_before,
            optimizer_after_sha256=self._optimizer_sha256(),
            trainer_update_count=self.update_count,
            replay_accepted_count=self.runner.replay_buffer.accepted_count,
            replay_seen_digest_count=self.runner.replay_buffer.seen_digest_count,
            replay_seen_identity_count=(
                self.runner.replay_buffer.seen_identity_count
            ),
            duration=2,
            validation_evidence_consumed=False,
            gradient_free=True,
            passed=True,
        )

    def run_no_gradient_preflight(self) -> WarmupPreflightReportV1:
        if self.update_count != 0 or self.decision_count > len(self.schedule):
            raise ModeledSmokePreflightError("runner is past the preflight boundary")
        while self.decision_count < len(self.schedule):
            self.collect_one()
        report = self._build_preflight_report()
        self._preflight_report = report
        return report

    def _validate_collector_checkpoint(
        self, checkpoint: CollectorCheckpointV1
    ) -> None:
        if type(checkpoint) is not CollectorCheckpointV1:
            raise ModeledSmokeCheckpointError("collector checkpoint has foreign type")
        if checkpoint.collector_binding_sha256 != (
            self.collector.collector_binding_sha256
        ):
            raise ModeledSmokeCheckpointError("collector binding changed")
        if checkpoint.decision_count != self.decision_count:
            raise ModeledSmokeCheckpointError("collector decision count differs")
        if checkpoint.transition_sha256s != tuple(
            item.transition_sha256 for item in self._ledger
        ):
            raise ModeledSmokeCheckpointError("collector history digest differs")

    def _boundary(self) -> BoundaryFingerprintV1:
        collector_checkpoint = self.collector.checkpoint()
        self._validate_collector_checkpoint(collector_checkpoint)
        buffer = self.runner.replay_buffer
        position_document = {
            "decision_count": self.decision_count,
            "runner_seed_plan": self.runner_factory.seed_plan.to_dict(),
            "schedule": {
                "batch_size": smoke_preregistration.FROZEN_CONFIG.batch_size,
                "transitions_per_update": (
                    smoke_preregistration.FROZEN_CONFIG
                    .environment_transitions_per_update
                ),
                "update_count": self.update_count,
            },
            "transition_ledger_sha256": _sha(
                [item.to_dict() for item in self._ledger]
            ),
        }
        return BoundaryFingerprintV1(
            update_count=self.update_count,
            decision_count=self.decision_count,
            actor_sha256=_tree_sha256(
                self.runner.model_bundle.actor.state_dict()
            ),
            critics_sha256=_tree_sha256(
                self.runner.model_bundle.critics.state_dict()
            ),
            actor_optimizer_sha256=_tree_sha256(
                self.runner.trainer.actor_optimizer.state_dict()
            ),
            critic_optimizer_sha256=_tree_sha256(
                self.runner.trainer.critic_optimizer.state_dict()
            ),
            decision_q_rng_sha256=_tensor_sha256(
                self._decision_q_generator.get_state()
            ),
            decision_mode_rng_sha256=_tensor_sha256(
                self._decision_mode_generator.get_state()
            ),
            runner_rng_reconstruction_sha256=_sha(position_document),
            replay_resident_sha256=_sha(
                list(buffer.resident_transition_digests())
            ),
            replay_accepted_count=buffer.accepted_count,
            replay_evicted_count=buffer.evicted_count,
            replay_seen_digest_count=buffer.seen_digest_count,
            replay_seen_identity_count=buffer.seen_identity_count,
            collector_checkpoint_sha256=_sha(collector_checkpoint.to_dict()),
        )

    def checkpoint(self) -> ModeledSmokeCheckpointV1:
        if self.update_count not in CHECKPOINT_UPDATES:
            raise ModeledSmokeCheckpointError("not at a registered checkpoint")
        if self._preflight_report is None:
            raise ModeledSmokeCheckpointError("preflight has not passed")
        expected = len(self.schedule) + 4 * self.update_count
        if self.decision_count != expected:
            raise ModeledSmokeCheckpointError("checkpoint ratio differs")
        collector_checkpoint = self.collector.checkpoint()
        self._validate_collector_checkpoint(collector_checkpoint)
        return ModeledSmokeCheckpointV1(
            update_count=self.update_count,
            decision_count=self.decision_count,
            factory_sha256=self.runner_factory.canonical_sha256,
            schedule_id=self.schedule.config.schedule_id,
            preregistration_sha256=(
                smoke_preregistration.PREREGISTRATION_SHA256
            ),
            seed_plan_sha256=self.runner_factory.seed_plan.canonical_sha256,
            modeled_binding_document=MappingProxyType(
                self.runner_factory.modeled_binding.to_dict()
            ),
            replay_binding_document=MappingProxyType(
                self.runner.binding.to_canonical_dict()
            ),
            collector_checkpoint=collector_checkpoint,
            ledger=tuple(self._ledger),
            preflight_report=self._preflight_report,
            boundary=self._boundary(),
        )

    def run_to_hard_stop(
        self,
        *,
        checkpoint_callback: Callable[[ModeledSmokeCheckpointEventV1], None],
        update_callback: Optional[Callable[[trainer.UpdateMetricsV1], None]] = None,
    ) -> ModeledSmokeSummaryV1:
        """Compatibility wrapper for the mandatory update-500 smoke stop."""
        return self.run_to_registered_update(
            smoke_preregistration.FROZEN_CONFIG.smoke_stop_update,
            checkpoint_callback=checkpoint_callback,
            update_callback=update_callback,
            emit_current_checkpoint=(self.update_count == 0),
        )

    def run_to_registered_update(
        self,
        target_update: int,
        *,
        checkpoint_callback: Callable[[ModeledSmokeCheckpointEventV1], None],
        update_callback: Optional[Callable[[trainer.UpdateMetricsV1], None]] = None,
        emit_current_checkpoint: bool = False,
    ) -> ModeledSmokeSummaryV1:
        """Continue exactly to one preregistered Run-4 checkpoint boundary."""
        if type(target_update) is not int or target_update not in CHECKPOINT_UPDATES:
            raise ModeledSmokeScheduleError(
                "target update is not a registered checkpoint"
            )
        if target_update < self.update_count:
            raise ModeledSmokeScheduleError("target update precedes current state")
        if not callable(checkpoint_callback):
            raise ModeledSmokeScheduleError("checkpoint callback is required")
        if update_callback is not None and not callable(update_callback):
            raise ModeledSmokeScheduleError("update callback is not callable")
        if type(emit_current_checkpoint) is not bool:
            raise ModeledSmokeScheduleError(
                "emit_current_checkpoint must be an exact bool"
            )
        starting_update = self.update_count
        emitted: list[int] = []
        if self._preflight_report is None:
            self.run_no_gradient_preflight()
        if emit_current_checkpoint and self.update_count in CHECKPOINT_UPDATES:
            event = ModeledSmokeCheckpointEventV1(
                self.update_count, self.checkpoint(), None
            )
            checkpoint_callback(event)
            emitted.append(self.update_count)
        config = smoke_preregistration.FROZEN_CONFIG
        latest: Optional[trainer.UpdateMetricsV1] = None
        while self.update_count < target_update:
            before = self.update_count
            for _ in range(config.environment_transitions_per_update):
                self.collect_one()
            latest = self.runner.train_once(config.batch_size)
            latest.require_finite()
            if self.update_count != before + 1:
                raise ModeledSmokeScheduleError("one loop did not make one update")
            if update_callback is not None:
                update_callback(latest)
            if self.update_count in CHECKPOINT_UPDATES:
                event = ModeledSmokeCheckpointEventV1(
                    self.update_count, self.checkpoint(), latest
                )
                checkpoint_callback(event)
                emitted.append(self.update_count)
        expected_decisions = config.warmup_decision_count + (
            config.environment_transitions_per_update * target_update
        )
        if (
            self.update_count != target_update
            or self.decision_count != expected_decisions
        ):
            raise ModeledSmokeScheduleError(
                "registered stop update/decision count differs"
            )
        final = self.checkpoint()
        return ModeledSmokeSummaryV1(
            starting_update=starting_update,
            final_update=self.update_count,
            final_decision_count=self.decision_count,
            emitted_checkpoint_updates=tuple(emitted),
            preflight_report_sha256=self._preflight_report.canonical_sha256,
            final_checkpoint_sha256=final.canonical_sha256,
        )

    @classmethod
    def restore(
        cls,
        checkpoint: ModeledSmokeCheckpointV1,
        *,
        runner_factory: ModeledSmokeRunnerFactoryV1,
        collector_factory: CollectorFactory,
        preflight_variation_contract: PreflightVariationContractV1,
    ) -> "ModeledSmokeOrchestratorV1":
        if type(checkpoint) is not ModeledSmokeCheckpointV1:
            raise ModeledSmokeCheckpointError("checkpoint has a foreign type")
        if checkpoint.factory_sha256 != runner_factory.canonical_sha256:
            raise ModeledSmokeCheckpointError("checkpoint factory differs")
        if checkpoint.seed_plan_sha256 != runner_factory.seed_plan.canonical_sha256:
            raise ModeledSmokeCheckpointError("checkpoint seed plan differs")
        if (
            checkpoint.preflight_report.variation_contract
            != preflight_variation_contract
        ):
            raise ModeledSmokeCheckpointError(
                "checkpoint preflight variation contract differs"
            )
        candidate = cls(
            runner_factory=runner_factory,
            collector_factory=collector_factory,
            preflight_variation_contract=preflight_variation_contract,
        )
        if checkpoint.schedule_id != candidate.schedule.config.schedule_id:
            raise ModeledSmokeCheckpointError("checkpoint schedule differs")
        candidate.collector.restore(checkpoint.collector_checkpoint)
        history = candidate.collector.history()
        if type(history) is not tuple or len(history) != checkpoint.decision_count:
            raise ModeledSmokeCheckpointError("collector history is incomplete")
        # Replay the exact logged actions and update boundaries from genesis.
        for ordinal, (collected, row) in enumerate(zip(history, checkpoint.ledger)):
            if type(collected) is not CollectedModeledTransitionV1:
                raise ModeledSmokeCheckpointError("collector history row is foreign")
            if ordinal < len(candidate.schedule):
                expected = candidate._warmup_request(ordinal)
            else:
                expected = candidate._actor_request(
                    ordinal, collected.state_features
                )
            if expected != row.action:
                raise ModeledSmokeCheckpointError(
                    "replayed policy action differs from checkpoint"
                )
            candidate._validate_collected(collected, expected)
            actual_row = ActionLedgerRowV1.from_collected(collected)
            if actual_row != row:
                raise ModeledSmokeCheckpointError("replayed ledger row differs")
            candidate.runner.ingest(collected.wrapper)
            candidate._history.append(collected)
            candidate._ledger.append(actual_row)
            completed = ordinal + 1 - len(candidate.schedule)
            transitions_per_update = (
                smoke_preregistration.FROZEN_CONFIG
                .environment_transitions_per_update
            )
            if completed > 0 and completed % transitions_per_update == 0:
                candidate.runner.train_once(
                    smoke_preregistration.FROZEN_CONFIG.batch_size
                )
        if candidate.update_count != checkpoint.update_count:
            raise ModeledSmokeCheckpointError("replayed update count differs")
        candidate._preflight_report = checkpoint.preflight_report
        candidate._validate_collector_checkpoint(checkpoint.collector_checkpoint)
        if candidate._boundary() != checkpoint.boundary:
            raise ModeledSmokeCheckpointError(
                "event-sourced restore is not bit-identical"
            )
        return candidate


def checkpoint_to_bytes(checkpoint: ModeledSmokeCheckpointV1) -> bytes:
    """Encode a checkpoint as canonical JSON bytes."""

    if type(checkpoint) is not ModeledSmokeCheckpointV1:
        raise ModeledSmokeCheckpointError("checkpoint has a foreign type")
    envelope = {
        "checkpoint": checkpoint.to_dict(),
        "checkpoint_sha256": checkpoint.canonical_sha256,
        "schema": CHECKPOINT_SCHEMA_ID,
    }
    return _canonical_bytes(envelope)


def _variation_contract_from_dict(
    value: Mapping[str, Any],
) -> PreflightVariationContractV1:
    return PreflightVariationContractV1(
        contract_id=value["contract_id"],
        contract_version=value["contract_version"],
        evidence_sha256=value["evidence_sha256"],
        feature_schema_sha256=value["feature_schema_sha256"],
        source_partition=value["source_partition"],
        requirements=tuple(
            PreflightFeatureRequirementV1(
                feature_name=item["feature_name"],
                minimum_distinct_count=item["minimum_distinct_count"],
                minimum_span=item["minimum_span"],
            )
            for item in value["requirements"]
        ),
    )


def _preflight_from_dict(value: Mapping[str, Any]) -> WarmupPreflightReportV1:
    if value.get("schema") != PREFLIGHT_SCHEMA_ID:
        raise ModeledSmokeCheckpointError("foreign preflight schema")
    variation = _variation_contract_from_dict(value["variation_contract"])
    if value["variation_contract_sha256"] != variation.canonical_sha256:
        raise ModeledSmokeCheckpointError("preflight variation digest differs")
    return WarmupPreflightReportV1(
        decision_count=value["decision_count"],
        coverage_counts=tuple(tuple(item) for item in value["coverage_counts"]),
        success_count=value["success_count"],
        failure_count=value["failure_count"],
        unique_state_count=value["unique_state_count"],
        unique_reward_count=value["unique_reward_count"],
        variation_contract=variation,
        feature_diagnostics=tuple(
            FeatureDiagnosticV1(**item) for item in value["feature_diagnostics"]
        ),
        previous_reconciliation=PreviousFeatureReconciliationV1(
            **value["previous_reconciliation"]
        ),
        transition_ledger_sha256=value["transition_ledger_sha256"],
        schedule_id=value["schedule_id"],
        runner_binding_sha256=value["runner_binding_sha256"],
        modeled_binding_sha256=value["modeled_binding_sha256"],
        mcs_acceptance_result_sha256=value["mcs_acceptance_result_sha256"],
        mcs_model_binding_sha256=value["mcs_model_binding_sha256"],
        collector_binding_sha256=value["collector_binding_sha256"],
        collector_checkpoint_sha256=value["collector_checkpoint_sha256"],
        model_before_sha256=value["model_before_sha256"],
        model_after_sha256=value["model_after_sha256"],
        optimizer_before_sha256=value["optimizer_before_sha256"],
        optimizer_after_sha256=value["optimizer_after_sha256"],
        trainer_update_count=value["trainer_update_count"],
        replay_accepted_count=value["replay_accepted_count"],
        replay_seen_digest_count=value["replay_seen_digest_count"],
        replay_seen_identity_count=value["replay_seen_identity_count"],
        duration=value["duration"],
        validation_evidence_consumed=value["validation_evidence_consumed"],
        gradient_free=value["gradient_free"],
        passed=value["passed"],
    )


def checkpoint_from_bytes(payload: bytes) -> ModeledSmokeCheckpointV1:
    """Decode and revalidate canonical JSON checkpoint bytes."""

    if type(payload) is not bytes:
        raise ModeledSmokeCheckpointError("checkpoint payload must be bytes")
    try:
        envelope = json.loads(payload.decode("ascii"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ModeledSmokeCheckpointError("checkpoint is not ASCII JSON") from exc
    if _canonical_bytes(envelope) != payload:
        raise ModeledSmokeCheckpointError("checkpoint bytes are not canonical")
    if set(envelope) != {"checkpoint", "checkpoint_sha256", "schema"}:
        raise ModeledSmokeCheckpointError("checkpoint envelope fields differ")
    if envelope["schema"] != CHECKPOINT_SCHEMA_ID:
        raise ModeledSmokeCheckpointError("foreign checkpoint schema")
    raw = envelope["checkpoint"]
    if raw.get("schema") != CHECKPOINT_SCHEMA_ID:
        raise ModeledSmokeCheckpointError("foreign checkpoint record")
    raw_boundary = raw["boundary"]
    boundary = BoundaryFingerprintV1(
        **{
            name: raw_boundary[name]
            for name in BoundaryFingerprintV1.__dataclass_fields__
        }
    )
    checkpoint = ModeledSmokeCheckpointV1(
        update_count=raw["update_count"],
        decision_count=raw["decision_count"],
        factory_sha256=raw["factory_sha256"],
        schedule_id=raw["schedule_id"],
        preregistration_sha256=raw["preregistration_sha256"],
        seed_plan_sha256=raw["seed_plan_sha256"],
        modeled_binding_document=MappingProxyType(
            dict(raw["modeled_binding_document"])
        ),
        replay_binding_document=MappingProxyType(
            dict(raw["replay_binding_document"])
        ),
        collector_checkpoint=CollectorCheckpointV1.from_dict(
            raw["collector_checkpoint"]
        ),
        ledger=tuple(ActionLedgerRowV1.from_dict(item) for item in raw["ledger"]),
        preflight_report=_preflight_from_dict(raw["preflight_report"]),
        boundary=boundary,
    )
    if checkpoint.canonical_sha256 != envelope["checkpoint_sha256"]:
        raise ModeledSmokeCheckpointError("checkpoint digest differs")
    return checkpoint



def write_checkpoint(path: Path, checkpoint: ModeledSmokeCheckpointV1) -> str:
    """Atomically publish one fsynced checkpoint without overwriting evidence."""

    target = Path(path)
    if target.exists():
        raise ModeledSmokeCheckpointError("checkpoint target already exists")
    if not target.parent.is_dir():
        raise ModeledSmokeCheckpointError("checkpoint parent does not exist")
    payload = checkpoint_to_bytes(checkpoint)
    staging = target.with_name(f".{target.name}.staging.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(staging, flags, 0o600)
    except FileExistsError as exc:
        raise ModeledSmokeCheckpointError(
            "checkpoint staging target already exists"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, target)
        except FileExistsError as exc:
            raise ModeledSmokeCheckpointError(
                "checkpoint target appeared"
            ) from exc
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(target.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(payload).hexdigest()


def read_checkpoint(path: Path) -> ModeledSmokeCheckpointV1:
    """Read one checkpoint without accepting symlinks or extra decoding."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ModeledSmokeCheckpointError(
            "checkpoint path must be a regular non-symlink file"
        )
    return checkpoint_from_bytes(target.read_bytes())


__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "CHECKPOINT_UPDATES",
    "COLLECTOR_CHECKPOINT_SCHEMA_ID",
    "CollectedModeledTransitionV1",
    "CollectorCheckpointV1",
    "FeatureDiagnosticV1",
    "ModeledActionRequestV1",
    "ModeledSmokeBindingError",
    "ModeledSmokeCheckpointError",
    "ModeledSmokeCheckpointEventV1",
    "ModeledSmokeCheckpointV1",
    "ModeledSmokeError",
    "ModeledSmokeOrchestratorV1",
    "ModeledSmokePreflightError",
    "ModeledSmokeRunnerFactoryV1",
    "ModeledSmokeScheduleError",
    "ModeledSmokeSummaryV1",
    "ModeledTransitionCollectorV1",
    "PreflightFeatureRequirementV1",
    "PreflightVariationContractV1",
    "PreviousFeatureReconciliationV1",
    "RunnerSeedPlanV1",
    "WarmupPreflightReportV1",
    "build_frozen_warmup_schedule",
    "checkpoint_from_bytes",
    "checkpoint_to_bytes",
    "read_checkpoint",
    "write_checkpoint",
]
