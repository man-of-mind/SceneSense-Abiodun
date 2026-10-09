"""Run-5 model binding and the 21-D checkpoint refusal boundary.

The Hybrid-SAC networks and every hyper-parameter are Run 4's; only
``state_dim`` changes from 21 to 22.  A Run-4 checkpoint is never loaded,
padded, sliced or relabelled as a Run-5 checkpoint:

* :func:`refuse_run4_binding` rejects any binding document that is not the
  exact Run-5 binding, naming the Run-4 digest explicitly when it is one;
* :func:`load_run5_model_state` accepts only in-memory state dicts whose
  binding is Run 5's *and* whose first-layer widths are 22 (actor) and 35
  (critics), then loads them ``strict=True`` into freshly built modules;
* :func:`refuse_checkpoint_directory` inspects only ``manifest.json`` (no
  ``torch.load``) and refuses every existing durable checkpoint: Run 5 has no
  durable checkpoint format yet because no Run-5 training has happened.

Importing this module performs no file I/O and initializes no accelerator.
"""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping

import torch
from torch import nn

from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io as R4IO
from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as R4M
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
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

from . import run5_state_contract as C

RUN5_MODEL_SCHEMA = "splitfusion.run5.hybrid_sac_models.v1"
ACTOR_INPUT_KEY = "encoder.0.weight"
CRITIC_INPUT_KEYS = tuple(
    f"{name}.trunk.0.weight" for name in ("critic_1", "critic_2", "target_1", "target_2")
)
RUN5_ACTOR_INPUT_WIDTH = C.RUN5_POLICY_FEATURE_COUNT
RUN5_CRITIC_INPUT_WIDTH = C.RUN5_POLICY_FEATURE_COUNT + EXPECTED_MODE_COUNT + 1
RUN4_ACTOR_INPUT_WIDTH = C.RUN4_PREFIX_COUNT


class Run5ModelError(RuntimeError):
    """A Run-5 model or binding differs from the contract."""


class Run5CheckpointRefused(Run5ModelError):
    """A checkpoint is not a Run-5 22-D checkpoint and must not be loaded."""


RUN5_MODEL_BINDING: Mapping[str, Any] = MappingProxyType(
    {
        "action_catalog_sha256": CATALOG_SHA256,
        "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "dtype": "torch.float32",
        "feature_schema_id": C.FEATURE_SCHEMA_ID,
        "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
        "feature_schema_version": C.FEATURE_SCHEMA_VERSION,
        "mode_count": EXPECTED_MODE_COUNT,
        "policy_feature_count": C.RUN5_POLICY_FEATURE_COUNT,
        "policy_feature_order": tuple(C.RUN5_POLICY_FEATURE_ORDER),
        "run4_hyperparameter_source": R4M.RUN4_MODEL_SCHEMA,
        "run4_prefix_feature_schema_sha256": C.R4.FEATURE_SCHEMA_SHA256,
        "schema": RUN5_MODEL_SCHEMA,
    }
)
RUN5_MODEL_BINDING_SHA256 = canonical_sha256(RUN5_MODEL_BINDING)

# Native Run-5 training binding.  Networks are built for 22 inputs from the
# Run-5 preregistration's explicit hyper-parameters.  The only Run-4 identity
# referenced is the feature schema that defines positions 0-20.
from . import run5_preregistration as _PREREG  # noqa: E402
from . import run5_snr_v2 as _SNR2  # noqa: E402

RUN5_TRAINING_MODEL_SCHEMA = "splitfusion.run5.native_hybrid_sac_models.v1"
RUN5_TRAINING_MODEL_BINDING: Mapping[str, Any] = MappingProxyType(
    {
        "schema": RUN5_TRAINING_MODEL_SCHEMA,
        "action_catalog_sha256": CATALOG_SHA256,
        "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "dtype": "torch.float32",
        "mode_count": EXPECTED_MODE_COUNT,
        "hidden_width": _PREREG.CONFIG.hidden_width,
        "hidden_depth": _PREREG.CONFIG.hidden_depth,
        "log_std_bounds": (_PREREG.CONFIG.log_std_min, _PREREG.CONFIG.log_std_max),
        "actor_input_width": C.RUN5_POLICY_FEATURE_COUNT,
        "critic_input_width": C.RUN5_POLICY_FEATURE_COUNT + EXPECTED_MODE_COUNT + 1,
        "policy_feature_count": C.RUN5_POLICY_FEATURE_COUNT,
        "policy_feature_order": tuple(C.RUN5_POLICY_FEATURE_ORDER),
        "feature_schema_id": _SNR2.FEATURE_SCHEMA_ID,
        "feature_schema_sha256": _SNR2.FEATURE_SCHEMA_SHA256,
        "feature_schema_version": _SNR2.FEATURE_SCHEMA_VERSION,
        "snr_scaling": "(snr_db - 5.5) / 19.0 on [5.5, 24.5]",
        "positions_0_20_feature_schema_sha256": C.R4.FEATURE_SCHEMA_SHA256,
    }
)
RUN5_TRAINING_MODEL_BINDING_SHA256 = canonical_sha256(RUN5_TRAINING_MODEL_BINDING)


def run5_model_config() -> HybridSacModelConfig:
    """Native 22-input configuration from the Run-5 preregistration."""
    from . import run5_preregistration as prereg

    return HybridSacModelConfig(
        state_dim=C.RUN5_POLICY_FEATURE_COUNT,
        mode_count=EXPECTED_MODE_COUNT,
        hidden_width=prereg.CONFIG.hidden_width,
        hidden_depth=prereg.CONFIG.hidden_depth,
        log_std_min=prereg.CONFIG.log_std_min,
        log_std_max=prereg.CONFIG.log_std_max,
        dtype=torch.float32,
        modeled_smoke_support=MODELED_SMOKE_SUPPORT,
    )


def _first_linear(module: nn.Module, name: str) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise Run5ModelError(f"{name} has no Linear layer")


def validate_run5_models(actor: ConditionalHybridActor, critics: TwinHybridCritics) -> None:
    if type(actor) is not ConditionalHybridActor:
        raise Run5ModelError("actor must be an exact ConditionalHybridActor")
    if type(critics) is not TwinHybridCritics:
        raise Run5ModelError("critics must be exact TwinHybridCritics")
    expected = run5_model_config()
    for name, module in (
        ("actor", actor),
        ("critic_1", critics.critic_1),
        ("critic_2", critics.critic_2),
        ("target_1", critics.target_1),
        ("target_2", critics.target_2),
    ):
        if module.config != expected:
            raise Run5ModelError(f"{name} configuration differs from Run 5")
        for tensor_name, value in chain(module.named_parameters(), module.named_buffers()):
            if value.device.type != "cpu":
                raise Run5ModelError(f"{name}.{tensor_name} escaped CPU")
    if _first_linear(actor.encoder, "actor.encoder").in_features != RUN5_ACTOR_INPUT_WIDTH:
        raise Run5ModelError("actor input width differs from Run-5 features")
    for name, critic in (
        ("critic_1", critics.critic_1),
        ("critic_2", critics.critic_2),
        ("target_1", critics.target_1),
        ("target_2", critics.target_2),
    ):
        if _first_linear(critic.trunk, name).in_features != RUN5_CRITIC_INPUT_WIDTH:
            raise Run5ModelError(f"{name} input width differs")


def build_run5_models(*, actor_seed: int, critic_seed: int):
    for value, name in ((actor_seed, "actor_seed"), (critic_seed, "critic_seed")):
        if type(value) is not int or value < 0:
            raise Run5ModelError(f"{name} must be a non-negative exact int")
    config = run5_model_config()
    actor = build_actor(config, seed=actor_seed)
    critics = build_twin_critics(config, seed=critic_seed)
    validate_run5_models(actor, critics)
    return actor, critics


def refuse_run4_binding(
    binding: Mapping[str, Any], expected_sha256: str = RUN5_MODEL_BINDING_SHA256
) -> None:
    """Raise unless ``binding`` is exactly the expected Run-5 model binding."""
    if not isinstance(binding, Mapping):
        raise Run5CheckpointRefused("checkpoint binding must be a mapping")
    document = dict(binding)
    if "policy_feature_order" in document:
        document["policy_feature_order"] = tuple(document["policy_feature_order"])
    try:
        digest = canonical_sha256(document)
    except Exception as exc:  # noqa: BLE001 - any non-canonical binding refuses
        raise Run5CheckpointRefused("checkpoint binding is not canonical") from exc
    if digest == R4M.RUN4_MODEL_BINDING_SHA256 or (
        document.get("feature_schema_sha256") == C.R4.FEATURE_SCHEMA_SHA256
    ):
        raise Run5CheckpointRefused(
            "this is a 21-D Run-4 checkpoint binding; relabelling, padding or "
            "slicing it into Run 5 is forbidden"
        )
    if document.get("policy_feature_count") == RUN4_ACTOR_INPUT_WIDTH:
        raise Run5CheckpointRefused("21-feature checkpoint binding refused")
    if digest != expected_sha256:
        raise Run5CheckpointRefused("checkpoint binding is not the Run-5 binding")


def _input_width(state: Mapping[str, Any], key: str, owner: str) -> int:
    if key not in state:
        raise Run5CheckpointRefused(f"{owner} state lacks {key}")
    tensor = state[key]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise Run5CheckpointRefused(f"{owner}.{key} is not a 2-D tensor")
    return int(tensor.shape[1])


def load_run5_model_state(
    *,
    binding: Mapping[str, Any],
    actor_state: Mapping[str, Any],
    critic_state: Mapping[str, Any],
    expected_binding_sha256: str = RUN5_MODEL_BINDING_SHA256,
):
    """Load a Run-5 state dict into fresh modules; refuse anything 21-D."""
    width = _input_width(actor_state, ACTOR_INPUT_KEY, "actor")
    if width == RUN4_ACTOR_INPUT_WIDTH:
        raise Run5CheckpointRefused(
            "actor state is 21-D (Run 4); it cannot become a 22-D Run-5 actor"
        )
    if width != RUN5_ACTOR_INPUT_WIDTH:
        raise Run5CheckpointRefused(f"actor input width {width} is not 22")
    for key in CRITIC_INPUT_KEYS:
        critic_width = _input_width(critic_state, key, "critics")
        if critic_width != RUN5_CRITIC_INPUT_WIDTH:
            raise Run5CheckpointRefused(
                f"critic input width {critic_width} is not {RUN5_CRITIC_INPUT_WIDTH}"
            )
    refuse_run4_binding(binding, expected_binding_sha256)
    # A 21-D Run-4 matrix zero-padded to 22 columns is a relabel, even under a
    # forged Run-5 binding.  No initialized or trained Run-5 layer has an
    # identically zero SNR input column.
    padded = [ACTOR_INPUT_KEY] if not bool(
        actor_state[ACTOR_INPUT_KEY][:, C.SNR_FEATURE_INDEX].abs().sum() > 0) else []
    padded += [key for key in CRITIC_INPUT_KEYS if not bool(
        critic_state[key][:, C.SNR_FEATURE_INDEX].abs().sum() > 0)]
    if padded:
        raise Run5CheckpointRefused(
            f"SNR input column is identically zero in {padded}: padded 21-D "
            "Run-4 weights cannot become a Run-5 checkpoint")
    actor, critics = build_run5_models(actor_seed=0, critic_seed=0)
    try:
        actor.load_state_dict(dict(actor_state), strict=True)
        critics.load_state_dict(dict(critic_state), strict=True)
    except RuntimeError as exc:
        raise Run5CheckpointRefused(f"strict state-dict load failed: {exc}") from exc
    validate_run5_models(actor, critics)
    return actor, critics


def refuse_checkpoint_directory(directory: str | Path) -> Dict[str, Any]:
    """Refuse every existing durable checkpoint without deserializing it.

    Returns nothing on success because success is impossible: Run 5 has not
    defined a durable format.  The raised message names a Run-4 checkpoint
    explicitly when the manifest identifies one.
    """
    manifest_path = Path(directory) / R4IO.MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise Run5CheckpointRefused("checkpoint manifest is unreadable") from exc
    if isinstance(manifest, dict) and manifest.get("schema_id") == R4IO.SCHEMA_ID:
        raise Run5CheckpointRefused(
            "durable Run-4 (21-D) checkpoint refused; Run 5 never loads it"
        )
    raise Run5CheckpointRefused(
        "Run 5 has no durable checkpoint format before training is authorized"
    )
