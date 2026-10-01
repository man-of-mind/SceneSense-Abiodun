"""Run-5B native 21-input model binding and exact checkpoint identity checks.

The Hybrid-SAC networks and every hyper-parameter are Run 4's/Run 5's; the
actor takes the 21-D Run-5B state and the critics 21 + 12 + 1 = 34 inputs.

Width alone cannot identify a Run-5B checkpoint: the old Run-4 actor is also
21-D.  :func:`classify_identity` and :func:`require_run5b_checkpoint` therefore
check, in order,

1. the old Run-4 21-D identity (model binding, feature schema, feature order,
   and the frozen seed-43 actor's tensor tree / first-layer tensor), refused;
2. the old Run-5 22-D identity (model bindings, feature schemas, feature
   order, preregistration, width), refused;
3. the exact Run-5B schema ID, feature order and order hash, feature schema
   hash, model binding, preregistration and tensor-tree identity, required.

Importing this module performs no file I/O and initializes no accelerator.
"""

from __future__ import annotations

from itertools import chain
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch
from torch import nn

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tensor_sha256, _tree_sha256,
)
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as R5M
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5SNR
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as R5V1
from rl_agent.splitfusion_hybrid_sac_v1.action_contract import CATALOG_SHA256, EXPECTED_MODE_COUNT
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor, HybridSacModelConfig, TwinHybridCritics, build_actor,
    build_twin_critics,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT, MODELED_SMOKE_SUPPORT_SHA256,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

from . import run5b_preregistration as PR
from . import run5b_state_contract as C

ACTOR_INPUT_KEY = "encoder.0.weight"
CRITIC_INPUT_KEYS = tuple(f"{n}.trunk.0.weight"
                          for n in ("critic_1", "critic_2", "target_1", "target_2"))
RUN5B_ACTOR_INPUT_WIDTH = C.RUN5B_POLICY_FEATURE_COUNT                      # 21
RUN5B_CRITIC_INPUT_WIDTH = C.RUN5B_POLICY_FEATURE_COUNT + EXPECTED_MODE_COUNT + 1  # 34
RUN5_ACTOR_INPUT_WIDTH = R5V1.RUN5_POLICY_FEATURE_COUNT                      # 22

# Old Run-4 frozen live actor (seed 43, update 10,000): ACTOR_BINDING_V2.json.
RUN4_FROZEN_ACTOR_TREE_SHA256 = "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3"
RUN4_FROZEN_ACTOR_ENCODER_TENSOR_SHA256 = (
    "805bcb6f3f317b08562710113ebc330a83a9f77963a954d87360902c876f1b36")
RUN4_FROZEN_ACTOR_WEIGHTS_SHA256 = (
    "d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29")
RUN5_PREREGISTRATION_SHA256 = PR.RUN5_PREREGISTRATION_SHA256

RUN5B_TRAINING_MODEL_SCHEMA = "splitfusion.run5b.native_hybrid_sac_models.v1"
RUN5B_TRAINING_MODEL_BINDING: Mapping[str, Any] = MappingProxyType({
    "schema": RUN5B_TRAINING_MODEL_SCHEMA,
    "action_catalog_sha256": CATALOG_SHA256,
    "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
    "dtype": "torch.float32",
    "mode_count": EXPECTED_MODE_COUNT,
    "hidden_width": PR.CONFIG.hidden_width,
    "hidden_depth": PR.CONFIG.hidden_depth,
    "log_std_bounds": (PR.CONFIG.log_std_min, PR.CONFIG.log_std_max),
    "actor_input_width": RUN5B_ACTOR_INPUT_WIDTH,
    "critic_input_width": RUN5B_CRITIC_INPUT_WIDTH,
    "policy_feature_count": C.RUN5B_POLICY_FEATURE_COUNT,
    "policy_feature_order": tuple(C.RUN5B_POLICY_FEATURE_ORDER),
    "feature_order_sha256": C.FEATURE_ORDER_SHA256,
    "feature_schema_id": C.FEATURE_SCHEMA_ID,
    "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
    "feature_schema_version": C.FEATURE_SCHEMA_VERSION,
    "removed_run4_feature": C.REMOVED_RUN4_FEATURE,
    "snr_scaling": "(snr_db - 5.5) / 19.0 on [5.5, 24.5]",
    "snr_provider_feature_schema_sha256": R5SNR.FEATURE_SCHEMA_SHA256,
})
RUN5B_TRAINING_MODEL_BINDING_SHA256 = canonical_sha256(RUN5B_TRAINING_MODEL_BINDING)

RUN4_IDENTITY = "RUN4_21D_ACTOR"
RUN5_IDENTITY = "RUN5_22D_ACTOR"
RUN5B_IDENTITY = "RUN5B_21D_ACTOR"
UNKNOWN_IDENTITY = "UNKNOWN"


class Run5BModelError(RuntimeError):
    """A Run-5B model or binding differs from the contract."""


class Run5BCheckpointRefused(Run5BModelError):
    """A checkpoint is not an exact Run-5B 21-D checkpoint and must not be loaded."""


def run5b_model_config() -> HybridSacModelConfig:
    return HybridSacModelConfig(
        state_dim=C.RUN5B_POLICY_FEATURE_COUNT, mode_count=EXPECTED_MODE_COUNT,
        hidden_width=PR.CONFIG.hidden_width, hidden_depth=PR.CONFIG.hidden_depth,
        log_std_min=PR.CONFIG.log_std_min, log_std_max=PR.CONFIG.log_std_max,
        dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT)


def _first_linear(module: nn.Module, name: str) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise Run5BModelError(f"{name} has no Linear layer")


def validate_run5b_models(actor: ConditionalHybridActor, critics: TwinHybridCritics) -> None:
    if type(actor) is not ConditionalHybridActor:
        raise Run5BModelError("actor must be an exact ConditionalHybridActor")
    if type(critics) is not TwinHybridCritics:
        raise Run5BModelError("critics must be exact TwinHybridCritics")
    expected = run5b_model_config()
    modules = (("actor", actor), ("critic_1", critics.critic_1), ("critic_2", critics.critic_2),
               ("target_1", critics.target_1), ("target_2", critics.target_2))
    for name, module in modules:
        if module.config != expected:
            raise Run5BModelError(f"{name} configuration differs from Run 5B")
        for tensor_name, value in chain(module.named_parameters(), module.named_buffers()):
            if value.device.type != "cpu":
                raise Run5BModelError(f"{name}.{tensor_name} escaped CPU")
    if _first_linear(actor.encoder, "actor.encoder").in_features != RUN5B_ACTOR_INPUT_WIDTH:
        raise Run5BModelError("actor input width differs from the Run-5B features")
    for name, critic in modules[1:]:
        if _first_linear(critic.trunk, name).in_features != RUN5B_CRITIC_INPUT_WIDTH:
            raise Run5BModelError(f"{name} input width differs")


def build_run5b_models(*, actor_seed: int, critic_seed: int):
    for value, name in ((actor_seed, "actor_seed"), (critic_seed, "critic_seed")):
        if type(value) is not int or value < 0:
            raise Run5BModelError(f"{name} must be a non-negative exact int")
    config = run5b_model_config()
    actor = build_actor(config, seed=actor_seed)
    critics = build_twin_critics(config, seed=critic_seed)
    validate_run5b_models(actor, critics)
    return actor, critics


# ---------------------------------------------------------------------------
# Identity classification and refusal
# ---------------------------------------------------------------------------

_RUN4_BINDINGS = frozenset({R4M.RUN4_MODEL_BINDING_SHA256})
_RUN4_FEATURE_SCHEMAS = frozenset({R4.FEATURE_SCHEMA_SHA256})
_RUN5_BINDINGS = frozenset({R5M.RUN5_MODEL_BINDING_SHA256, R5M.RUN5_TRAINING_MODEL_BINDING_SHA256})
_RUN5_FEATURE_SCHEMAS = frozenset({R5V1.FEATURE_SCHEMA_SHA256, R5SNR.FEATURE_SCHEMA_SHA256})


def _order(value: Any) -> Optional[tuple]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return None


def classify_identity(identity: Mapping[str, Any]) -> str:
    """Classify a checkpoint manifest / binding document by declared identity.

    Accepts either a bundle manifest (``model_binding_sha256``,
    ``feature_schema_sha256``, ``feature_order``, ``preregistration_sha256``)
    or a model-binding document (``policy_feature_order`` ...).
    """
    if not isinstance(identity, Mapping):
        return UNKNOWN_IDENTITY
    order = _order(identity.get("feature_order", identity.get("policy_feature_order")))
    binding = identity.get("model_binding_sha256")
    if binding is None and "schema" in identity and "policy_feature_order" in identity:
        document = dict(identity)
        document["policy_feature_order"] = tuple(document["policy_feature_order"])
        if "log_std_bounds" in document:
            document["log_std_bounds"] = tuple(document["log_std_bounds"])
        binding = canonical_sha256(document)
    schema = identity.get("feature_schema_sha256")
    count = identity.get("policy_feature_count", None if order is None else len(order))
    if (binding in _RUN4_BINDINGS or schema in _RUN4_FEATURE_SCHEMAS
            or order == tuple(R4.POLICY_FEATURE_ORDER)
            or (order is not None and C.REMOVED_RUN4_FEATURE in order and len(order) == 21)):
        return RUN4_IDENTITY
    if (binding in _RUN5_BINDINGS or schema in _RUN5_FEATURE_SCHEMAS
            or order == tuple(R5V1.RUN5_POLICY_FEATURE_ORDER)
            or identity.get("preregistration_sha256") == RUN5_PREREGISTRATION_SHA256
            or count == RUN5_ACTOR_INPUT_WIDTH):
        return RUN5_IDENTITY
    if (binding == RUN5B_TRAINING_MODEL_BINDING_SHA256
            and schema == C.FEATURE_SCHEMA_SHA256
            and order == tuple(C.RUN5B_POLICY_FEATURE_ORDER)):
        return RUN5B_IDENTITY
    return UNKNOWN_IDENTITY


def _input_width(state: Mapping[str, Any], key: str, owner: str) -> int:
    if key not in state:
        raise Run5BCheckpointRefused(f"{owner} state lacks {key}")
    tensor = state[key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise Run5BCheckpointRefused(f"{owner}.{key} is not a 2-D tensor")
    return int(tensor.shape[1])


def require_run5b_identity(manifest: Mapping[str, Any], *, preregistration_sha256: str) -> None:
    """Refuse every declared identity except the exact Run-5B one."""
    kind = classify_identity(manifest)
    if kind == RUN4_IDENTITY:
        raise Run5BCheckpointRefused("old Run-4 21-D actor identity refused; equal width is not "
                                     "identity and a Run-4 checkpoint never becomes Run 5B")
    if kind == RUN5_IDENTITY:
        raise Run5BCheckpointRefused("old Run-5 22-D actor identity refused; slicing or "
                                     "relabelling it into Run 5B is forbidden")
    if kind != RUN5B_IDENTITY:
        raise Run5BCheckpointRefused("checkpoint does not declare the Run-5B identity")
    checks = {
        "feature_schema_id": C.FEATURE_SCHEMA_ID,
        "feature_order_sha256": C.FEATURE_ORDER_SHA256,
        "model_binding_sha256": RUN5B_TRAINING_MODEL_BINDING_SHA256,
        "preregistration_sha256": preregistration_sha256,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise Run5BCheckpointRefused(f"checkpoint {key} differs from the Run-5B registration")


def require_run5b_actor_tensors(actor_state: Mapping[str, Any], *,
                                expected_tree_sha256: Optional[str]) -> str:
    """Refuse non-21-D, frozen-Run-4 or unregistered actor tensors; return the tree hash."""
    width = _input_width(actor_state, ACTOR_INPUT_KEY, "actor")
    if width == RUN5_ACTOR_INPUT_WIDTH:
        raise Run5BCheckpointRefused("actor is 22-D (Run 5); it cannot become a Run-5B actor")
    if width != RUN5B_ACTOR_INPUT_WIDTH:
        raise Run5BCheckpointRefused(f"actor input width {width} is not 21")
    tree = _tree_sha256(dict(actor_state))
    if (tree == RUN4_FROZEN_ACTOR_TREE_SHA256
            or _tensor_sha256(actor_state[ACTOR_INPUT_KEY]) == RUN4_FROZEN_ACTOR_ENCODER_TENSOR_SHA256):
        raise Run5BCheckpointRefused("these are the frozen Run-4 seed-43 actor tensors")
    if expected_tree_sha256 is not None and tree != expected_tree_sha256:
        raise Run5BCheckpointRefused("actor tensor-tree identity differs from the registration")
    return tree


def require_run5b_checkpoint(*, manifest: Mapping[str, Any], actor_state: Mapping[str, Any],
                             critic_state: Optional[Mapping[str, Any]],
                             preregistration_sha256: str,
                             expected_tree_sha256: Optional[str] = None):
    """Exact identity checks, then a strict load into freshly built modules."""
    require_run5b_identity(manifest, preregistration_sha256=preregistration_sha256)
    declared = manifest.get("actor_tree_sha256")
    tree = require_run5b_actor_tensors(actor_state, expected_tree_sha256=expected_tree_sha256)
    if declared and declared != tree:
        raise Run5BCheckpointRefused("actor tensor tree differs from the manifest declaration")
    actor, critics = build_run5b_models(actor_seed=0, critic_seed=0)
    if critic_state is not None:
        for key in CRITIC_INPUT_KEYS:
            critic_width = _input_width(critic_state, key, "critics")
            if critic_width != RUN5B_CRITIC_INPUT_WIDTH:
                raise Run5BCheckpointRefused(f"critic input width {critic_width} is not 34")
    try:
        actor.load_state_dict(dict(actor_state), strict=True)
        if critic_state is not None:
            critics.load_state_dict(dict(critic_state), strict=True)
    except RuntimeError as exc:
        raise Run5BCheckpointRefused(f"strict state-dict load failed: {exc}") from exc
    validate_run5b_models(actor, critics)
    return actor, (critics if critic_state is not None else None)
