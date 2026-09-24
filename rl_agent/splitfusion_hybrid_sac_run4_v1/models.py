"""Run-4 bindings for the proven conditional Hybrid-SAC networks.

The neural implementation is reused unchanged.  This module supplies the
Run-4-specific 21-feature dimension and binds the resulting modules to the
v2 state contract and the registered continuous-q support.  Importing it
performs no file I/O, samples no RNG, and initializes no accelerator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Any, Dict, Mapping

import torch
from torch import nn

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import (
    CATALOG_SHA256,
    EXPECTED_MODE_COUNT,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor,
    HybridSacModelConfig,
    TwinHybridCritics,
    build_actor,
    build_twin_critics,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    canonical_sha256,
)

from .run4_contract import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    POLICY_FEATURE_COUNT,
    POLICY_FEATURE_ORDER,
    SCHEMA_ID,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
)

__all__ = [
    "RUN4_MODEL_BINDING",
    "RUN4_MODEL_BINDING_SHA256",
    "RUN4_MODEL_SCHEMA",
    "Run4ModelBundleV1",
    "Run4ModelError",
    "build_run4_models",
    "run4_model_config",
    "validate_run4_models",
]


RUN4_MODEL_SCHEMA = "splitfusion.run4.hybrid_sac_models.v1"


class Run4ModelError(RuntimeError):
    """A Run-4 model or its immutable binding differs from the contract."""


RUN4_MODEL_BINDING: Mapping[str, Any] = MappingProxyType(
    {
        "action_catalog_sha256": CATALOG_SHA256,
        "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "dtype": "torch.float32",
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "mode_count": EXPECTED_MODE_COUNT,
        "policy_feature_count": POLICY_FEATURE_COUNT,
        "policy_feature_order": tuple(POLICY_FEATURE_ORDER),
        "run4_contract_schema_id": SCHEMA_ID,
        "run4_contract_schema_sha256": SCHEMA_SHA256,
        "run4_contract_schema_version": SCHEMA_VERSION,
        "schema": RUN4_MODEL_SCHEMA,
    }
)
RUN4_MODEL_BINDING_SHA256 = canonical_sha256(RUN4_MODEL_BINDING)


def run4_model_config() -> HybridSacModelConfig:
    """Return the sole model configuration accepted by Run 4."""

    return HybridSacModelConfig(
        state_dim=POLICY_FEATURE_COUNT,
        mode_count=EXPECTED_MODE_COUNT,
        dtype=torch.float32,
        modeled_smoke_support=MODELED_SMOKE_SUPPORT,
    )


@dataclass(frozen=True, slots=True)
class Run4ModelBundleV1:
    actor: ConditionalHybridActor
    critics: TwinHybridCritics
    binding_sha256: str = RUN4_MODEL_BINDING_SHA256

    def __post_init__(self) -> None:
        if self.binding_sha256 != RUN4_MODEL_BINDING_SHA256:
            raise Run4ModelError("model binding digest differs")
        validate_run4_models(self.actor, self.critics)

    def binding_document(self) -> Dict[str, Any]:
        return dict(RUN4_MODEL_BINDING)


def _first_linear(module: nn.Module, name: str) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise Run4ModelError(f"{name} has no Linear layer")


def validate_run4_models(
    actor: ConditionalHybridActor,
    critics: TwinHybridCritics,
) -> None:
    """Fail closed unless modules and real tensors implement the binding."""

    if type(actor) is not ConditionalHybridActor:
        raise Run4ModelError("actor must be an exact ConditionalHybridActor")
    if type(critics) is not TwinHybridCritics:
        raise Run4ModelError("critics must be exact TwinHybridCritics")
    expected = run4_model_config()
    modules = (
        ("actor", actor),
        ("critic_1", critics.critic_1),
        ("critic_2", critics.critic_2),
        ("target_1", critics.target_1),
        ("target_2", critics.target_2),
    )
    for name, module in modules:
        if module.config != expected:
            raise Run4ModelError(f"{name} configuration differs from Run 4")
        for tensor_name, value in chain(
            module.named_parameters(), module.named_buffers()
        ):
            if value.device.type != "cpu":
                raise Run4ModelError(f"{name}.{tensor_name} escaped CPU")
            if value.is_floating_point() and value.dtype is not torch.float32:
                raise Run4ModelError(
                    f"{name}.{tensor_name} is not float32"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise Run4ModelError(f"{name}.{tensor_name} is non-finite")

    if not actor.uses_modeled_smoke_support:
        raise Run4ModelError("actor lacks the registered q support")
    if actor.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256:
        raise Run4ModelError("actor q-support digest differs")
    if _first_linear(actor.encoder, "actor.encoder").in_features != POLICY_FEATURE_COUNT:
        raise Run4ModelError("actor input width differs from Run-4 features")
    expected_critic_width = POLICY_FEATURE_COUNT + EXPECTED_MODE_COUNT + 1
    for name, critic in (
        ("critic_1", critics.critic_1),
        ("critic_2", critics.critic_2),
        ("target_1", critics.target_1),
        ("target_2", critics.target_2),
    ):
        if _first_linear(critic.trunk, name).in_features != expected_critic_width:
            raise Run4ModelError(f"{name} input width differs")
    if any(parameter.requires_grad for parameter in chain(
        critics.target_1.parameters(), critics.target_2.parameters()
    )):
        raise Run4ModelError("target critics must remain frozen")


def build_run4_models(*, actor_seed: int, critic_seed: int) -> Run4ModelBundleV1:
    """Build deterministic CPU models without advancing the global RNG."""

    for value, name in ((actor_seed, "actor_seed"), (critic_seed, "critic_seed")):
        if type(value) is not int or value < 0:
            raise Run4ModelError(f"{name} must be a non-negative exact int")
    config = run4_model_config()
    actor = build_actor(config, seed=actor_seed)
    critics = build_twin_critics(config, seed=critic_seed)
    return Run4ModelBundleV1(actor=actor, critics=critics)
