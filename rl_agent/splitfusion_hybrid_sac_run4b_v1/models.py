"""Run-4B bindings for the unchanged conditional Hybrid-SAC networks.

Actor input 20; critic input 33 (20 state + 12 mode one-hot + 1 q).
Importing performs no I/O, RNG or accelerator work.
"""

from __future__ import annotations

from itertools import chain

import torch
from torch import nn

from rl_agent.splitfusion_hybrid_sac_v1.action_contract import CATALOG_SHA256
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

from . import contract as C

MODEL_SCHEMA = "splitfusion.run4b.hybrid_sac_models.v1"
MODEL_BINDING = {
    "schema": MODEL_SCHEMA,
    "action_catalog_sha256": CATALOG_SHA256,
    "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
    "dtype": "torch.float32",
    "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],
    "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
    "feature_order": list(C.FEATURE_ORDER),
    "actor_input_width": C.FEATURE_COUNT,
    "critic_input_width": C.CRITIC_INPUT_WIDTH,
    "mode_count": C.MODE_COUNT,
}
MODEL_BINDING_SHA256 = C.canonical_sha256(MODEL_BINDING)


class ModelError(RuntimeError):
    """A Run-4B model differs from its binding."""


def model_config() -> HybridSacModelConfig:
    return HybridSacModelConfig(
        state_dim=C.FEATURE_COUNT, mode_count=C.MODE_COUNT,
        dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT)


def _first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise ModelError("module has no Linear layer")


def validate_models(actor: ConditionalHybridActor,
                    critics: TwinHybridCritics) -> None:
    if type(actor) is not ConditionalHybridActor:
        raise ModelError("actor must be an exact ConditionalHybridActor")
    if type(critics) is not TwinHybridCritics:
        raise ModelError("critics must be exact TwinHybridCritics")
    expected = model_config()
    members = (("actor", actor), ("critic_1", critics.critic_1),
               ("critic_2", critics.critic_2), ("target_1", critics.target_1),
               ("target_2", critics.target_2))
    for name, module in members:
        if module.config != expected:
            raise ModelError(f"{name} configuration differs from Run-4B")
        for tensor_name, value in chain(module.named_parameters(),
                                        module.named_buffers()):
            if value.device.type != "cpu":
                raise ModelError(f"{name}.{tensor_name} escaped CPU")
            if value.is_floating_point() and (
                    value.dtype is not torch.float32
                    or not bool(torch.isfinite(value).all())):
                raise ModelError(f"{name}.{tensor_name} invalid dtype/value")
    if not actor.uses_modeled_smoke_support or (
            actor.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256):
        raise ModelError("actor q support differs")
    if _first_linear(actor.encoder).in_features != C.FEATURE_COUNT:
        raise ModelError("actor input width is not 20")
    for name in ("critic_1", "critic_2", "target_1", "target_2"):
        width = _first_linear(getattr(critics, name).trunk).in_features
        if width != C.CRITIC_INPUT_WIDTH:
            raise ModelError(f"{name} input width is not 33")
    if any(p.requires_grad for p in chain(critics.target_1.parameters(),
                                          critics.target_2.parameters())):
        raise ModelError("target critics must remain frozen")


def build_models(*, actor_seed: int, critic_seed: int
                 ) -> tuple[ConditionalHybridActor, TwinHybridCritics]:
    config = model_config()
    actor = build_actor(config, seed=actor_seed)
    critics = build_twin_critics(config, seed=critic_seed)
    validate_models(actor, critics)
    return actor, critics
