"""Run-5B model binding (actor 21, critics 34) and exact checkpoint identity.

The networks are the unchanged conditional Hybrid-SAC modules with Run-4B's
configuration; only ``state_dim`` differs (20 -> 21).

Width alone is not identity (Run 4 is also 21-D).  :func:`classify_identity`
recognises the old Run-4 21-D, Run-5 22-D and Run-4B 20-D families and
:func:`require_run5b_identity` refuses them; a Run-5B checkpoint must match
the exact schema ID, feature order and order hash, feature schema hash, model
binding, registration and tensor tree.  Importing performs no I/O.
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Mapping, Optional

import torch
from torch import nn

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tensor_sha256, _tree_sha256,
)
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import models as M4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as R5M
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import CATALOG_SHA256
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor, HybridSacModelConfig, TwinHybridCritics, build_actor,
    build_twin_critics,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT, MODELED_SMOKE_SUPPORT_SHA256,
)

from . import run5b_state_contract as C

ACTOR_INPUT_KEY = "encoder.0.weight"
CRITIC_INPUT_KEYS = tuple(f"{n}.trunk.0.weight"
                          for n in ("critic_1", "critic_2", "target_1", "target_2"))
RUN4_FROZEN_ACTOR_TREE_SHA256 = "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3"
RUN4_FROZEN_ACTOR_ENCODER_TENSOR_SHA256 = (
    "805bcb6f3f317b08562710113ebc330a83a9f77963a954d87360902c876f1b36")
RUN4_FROZEN_ACTOR_WEIGHTS_SHA256 = (
    "d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29")
RUN5_PREREGISTRATION_SHA256 = "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf"

MODEL_SCHEMA = "splitfusion.run5b.hybrid_sac_models.v2"
MODEL_BINDING = {
    "schema": MODEL_SCHEMA,
    "action_catalog_sha256": CATALOG_SHA256,
    "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
    "dtype": "torch.float32",
    "feature_schema_id": C.FEATURE_SCHEMA["schema_id"],
    "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
    "feature_order": list(C.FEATURE_ORDER),
    "feature_order_sha256": C.FEATURE_ORDER_SHA256,
    "actor_input_width": C.FEATURE_COUNT,
    "critic_input_width": C.CRITIC_INPUT_WIDTH,
    "mode_count": C4.MODE_COUNT,
    "run4b_model_config": "identical except state_dim 20 -> 21",
}
MODEL_BINDING_SHA256 = C4.canonical_sha256(MODEL_BINDING)

RUN4_IDENTITY, RUN5_IDENTITY, RUN4B_IDENTITY = "RUN4_21D", "RUN5_22D", "RUN4B_20D"
RUN5B_IDENTITY, UNKNOWN_IDENTITY = "RUN5B_21D", "UNKNOWN"


class ModelError(RuntimeError):
    pass


class CheckpointRefused(ModelError):
    pass


def model_config() -> HybridSacModelConfig:
    return HybridSacModelConfig(state_dim=C.FEATURE_COUNT, mode_count=C4.MODE_COUNT,
                                dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT)


def _first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise ModelError("module has no Linear layer")


def validate_models(actor: ConditionalHybridActor, critics: TwinHybridCritics) -> None:
    """Run-4B ``validate_models`` checks at Run-5B widths."""
    if type(actor) is not ConditionalHybridActor:
        raise ModelError("actor must be an exact ConditionalHybridActor")
    if type(critics) is not TwinHybridCritics:
        raise ModelError("critics must be exact TwinHybridCritics")
    expected = model_config()
    members = (("actor", actor), ("critic_1", critics.critic_1), ("critic_2", critics.critic_2),
               ("target_1", critics.target_1), ("target_2", critics.target_2))
    for name, module in members:
        if module.config != expected:
            raise ModelError(f"{name} configuration differs from Run-5B")
        for tensor_name, value in chain(module.named_parameters(), module.named_buffers()):
            if value.device.type != "cpu":
                raise ModelError(f"{name}.{tensor_name} escaped CPU")
            if value.is_floating_point() and (value.dtype is not torch.float32
                                              or not bool(torch.isfinite(value).all())):
                raise ModelError(f"{name}.{tensor_name} invalid dtype/value")
    if not actor.uses_modeled_smoke_support or (
            actor.modeled_smoke_support_sha256 != MODELED_SMOKE_SUPPORT_SHA256):
        raise ModelError("actor q support differs")
    if _first_linear(actor.encoder).in_features != C.FEATURE_COUNT:
        raise ModelError("actor input width is not 21")
    for name in ("critic_1", "critic_2", "target_1", "target_2"):
        if _first_linear(getattr(critics, name).trunk).in_features != C.CRITIC_INPUT_WIDTH:
            raise ModelError(f"{name} input width is not 34")
    if any(p.requires_grad for p in chain(critics.target_1.parameters(),
                                          critics.target_2.parameters())):
        raise ModelError("target critics must remain frozen")


def build_models(*, actor_seed: int, critic_seed: int):
    config = model_config()
    actor = build_actor(config, seed=actor_seed)
    critics = build_twin_critics(config, seed=critic_seed)
    validate_models(actor, critics)
    return actor, critics


def _binding_sha(document: Mapping[str, Any]) -> Optional[str]:
    try:
        return C4.canonical_sha256(dict(document))
    except Exception:  # noqa: BLE001
        return None


_RUN4_BINDINGS = {R4M.RUN4_MODEL_BINDING_SHA256}
_RUN5_BINDINGS = {R5M.RUN5_MODEL_BINDING_SHA256, R5M.RUN5_TRAINING_MODEL_BINDING_SHA256}
_RUN4B_BINDINGS = {M4.MODEL_BINDING_SHA256}


def classify_identity(identity: Mapping[str, Any]) -> str:
    """Classify a bundle manifest, actor export or model binding by declared identity."""
    if not isinstance(identity, Mapping):
        return UNKNOWN_IDENTITY
    order = identity.get("feature_order", identity.get("policy_feature_order"))
    order = tuple(order) if isinstance(order, (list, tuple)) else None
    binding = identity.get("model_binding_sha256")
    if binding is None and "schema" in identity:
        document = dict(identity)
        for key in ("policy_feature_order", "log_std_bounds"):
            if key in document:
                document[key] = tuple(document[key])
        binding = _binding_sha(document) if "policy_feature_order" in document \
            else _binding_sha(identity)
    schema = identity.get("feature_schema_sha256")
    if (binding in _RUN5_BINDINGS
            or schema in (R5V1.FEATURE_SCHEMA_SHA256, R5SNR.FEATURE_SCHEMA_SHA256)
            or order == tuple(R5V1.RUN5_POLICY_FEATURE_ORDER)
            or identity.get("preregistration_sha256") == RUN5_PREREGISTRATION_SHA256
            or (order is not None and len(order) == 22)):
        return RUN5_IDENTITY
    if (binding in _RUN4_BINDINGS or schema == R4.FEATURE_SCHEMA_SHA256
            or order == tuple(R4.POLICY_FEATURE_ORDER)
            or (order is not None and C.REMOVED_RUN4_FEATURE in order)):
        return RUN4_IDENTITY
    if (binding in _RUN4B_BINDINGS or schema == C4.FEATURE_SCHEMA_SHA256
            or order == tuple(C4.FEATURE_ORDER) or (order is not None and len(order) == 20)):
        return RUN4B_IDENTITY
    if (binding == MODEL_BINDING_SHA256 and schema == C.FEATURE_SCHEMA_SHA256
            and order == tuple(C.FEATURE_ORDER)):
        return RUN5B_IDENTITY
    return UNKNOWN_IDENTITY


def require_run5b_identity(manifest: Mapping[str, Any], *, registration_sha256: str) -> None:
    kind = classify_identity(manifest)
    if kind == RUN4_IDENTITY:
        raise CheckpointRefused("old Run-4 21-D identity refused (equal width is not identity)")
    if kind == RUN5_IDENTITY:
        raise CheckpointRefused("old Run-5 22-D identity refused")
    if kind == RUN4B_IDENTITY:
        raise CheckpointRefused("Run-4B 20-D identity refused; it never becomes Run-5B")
    if kind != RUN5B_IDENTITY:
        raise CheckpointRefused("checkpoint does not declare the Run-5B identity")
    for key, expected in (("feature_schema_id", C.FEATURE_SCHEMA["schema_id"]),
                          ("feature_order_sha256", C.FEATURE_ORDER_SHA256),
                          ("model_binding_sha256", MODEL_BINDING_SHA256),
                          ("registration_sha256", registration_sha256)):
        if manifest.get(key) != expected:
            raise CheckpointRefused(f"checkpoint {key} differs from the Run-5B registration")


def require_run5b_actor_state(actor_state: Mapping[str, Any], *,
                              expected_tree_sha256: Optional[str]) -> str:
    if ACTOR_INPUT_KEY not in actor_state or not isinstance(
            actor_state[ACTOR_INPUT_KEY], torch.Tensor):
        raise CheckpointRefused("actor state lacks its input layer")
    width = int(actor_state[ACTOR_INPUT_KEY].shape[1])
    if width != C.FEATURE_COUNT:
        raise CheckpointRefused(f"actor input width {width} is not 21")
    tree = _tree_sha256(dict(actor_state))
    if (tree == RUN4_FROZEN_ACTOR_TREE_SHA256 or _tensor_sha256(
            actor_state[ACTOR_INPUT_KEY]) == RUN4_FROZEN_ACTOR_ENCODER_TENSOR_SHA256):
        raise CheckpointRefused("these are the frozen Run-4 seed-43 actor tensors")
    if expected_tree_sha256 is not None and tree != expected_tree_sha256:
        raise CheckpointRefused("actor tensor-tree identity differs from the registration")
    return tree


def load_run5b_actor(manifest: Mapping[str, Any], actor_state: Mapping[str, Any], *,
                     registration_sha256: str) -> ConditionalHybridActor:
    require_run5b_identity(manifest, registration_sha256=registration_sha256)
    require_run5b_actor_state(actor_state, expected_tree_sha256=manifest["actor_tree_sha256"])
    actor = build_actor(model_config(), seed=0)
    try:
        actor.load_state_dict(dict(actor_state), strict=True)
    except RuntimeError as exc:
        raise CheckpointRefused(f"strict actor load failed: {exc}") from exc
    return actor
