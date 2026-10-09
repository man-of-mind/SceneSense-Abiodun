"""Original quality-aware Run-5 actor and state seam for live evaluation.

This module does not train or recover weights. It cold-loads the existing
seed-43/update-10000 export and verifies its exact manifest, file, tensor,
feature and native-model identities. Selecting seed 43 for this evaluation
is a new live choice, not a retroactive claim about the training registration.

The state seam calls the frozen Run-5 guard/builder and keeps the Run-4
quality-aware prefix, freshness, scaling, radio support and previous outcome.
It returns the Run-4 guarded prefix alongside the attested 22-D vector, so
the unchanged decision engine can continue reading ``guarded.boundary``.
Imports and all functions here are CPU-only and perform no network I/O.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as F4
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.ue_telemetry_provider_v2 import TRAINING_FRESHNESS
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_state_contract as C5
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S5
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as AC
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor, HybridSacModelConfig, build_actor,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT, MODELED_SMOKE_SUPPORT_SHA256,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256


SELECTED_SEED = 43
SELECTED_UPDATE = 10_000
WEIGHTS_SHA256 = "f17e812defd7d14204ed9c2a140d776727ecc5cf55e0b2f5a3db466e24d2defe"
EXPORT_MANIFEST_SHA256 = "20e9103ae952ee5b4d62dc9c23d029653e4c887c556e983fc3502087671f2d66"
ACTOR_TREE_SHA256 = "c51b4f1af3f5f8dedc5080036ff271c55c2542ef26b40da148ccf45f1f493aff"
FEATURE_SCHEMA_SHA256 = "487015205070d0a5dac6e61b9fe7aa8f39351bda5d1710de3bbaadc78358e596"
NATIVE_MODEL_BINDING_SHA256 = "a402a64d4305e99d5dca51fa2832657af9438d34dc9523c2decf2cdb01a9bf9a"
PREREGISTRATION_SHA256 = "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf"
SOURCE_CHECKPOINT_MANIFEST_SHA256 = "7a656d9f4ab181ecdca26ad54c99e646a55b2a003321a6d5854ccb3046b3e9e8"
DEPLOYMENT_RULE = F4.DEPLOYMENT_RULE
SELECTION_SCOPE = "NEW_LIVE_EVALUATION_CHOICE__ORIGINAL_RUN5_SEED43_UPDATE10000__NOT_RETRAINED"
LEASE_MAX_HEARTBEAT_AGE_NS = 200_000_000
_RUN4_BUILD_LIVE_STATE = LS.build_live_state

# These two original modules must be copied unchanged onto any runtime line
# that does not already contain them. Never discover a different worktree
# implicitly at import time.
REQUIRED_FROZEN_MODULES = {
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_state_contract.py":
        "718315b65a08b71c519b2328539d274800398794c46db363cf4a78b5a272e5ae",
    "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_snr_v2.py":
        "94d68d5a0e048c35b02a04932c3594185da4977eb2f1d18f62e371bd9a9c1846",
}


class Run5LiveBindingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Run5LiveBindingError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def verify_frozen_sources() -> dict[str, str]:
    result = {}
    for module, relative in ((C5, "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_state_contract.py"),
                             (S5, "rl_agent/splitfusion_hybrid_sac_run5_v1/run5_snr_v2.py")):
        actual = F4.sha256_file(Path(module.__file__))
        _require(actual == REQUIRED_FROZEN_MODULES[relative], f"frozen source drift: {relative}")
        result[relative] = actual
    _require(S5.FEATURE_SCHEMA_SHA256 == FEATURE_SCHEMA_SHA256,
             "Run-5 v2 feature schema differs from the trained export")
    _require(C5.RUN5_POLICY_FEATURE_COUNT == 22 and
             tuple(C5.RUN5_POLICY_FEATURE_ORDER[:21]) == R4.POLICY_FEATURE_ORDER,
             "Run-5 feature order differs from the quality-aware Run-4 prefix")
    _require(canonical_sha256(native_model_binding()) == NATIVE_MODEL_BINDING_SHA256,
             "native Run-5 model binding differs")
    return result


def native_model_binding() -> dict[str, Any]:
    """Exact native-training binding, not the obsolete scaffold binding."""
    return {
        "schema": "splitfusion.run5.native_hybrid_sac_models.v1",
        "action_catalog_sha256": AC.CATALOG_SHA256,
        "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "dtype": "torch.float32", "mode_count": AC.EXPECTED_MODE_COUNT,
        "hidden_width": 128, "hidden_depth": 2, "log_std_bounds": (-5.0, 2.0),
        "actor_input_width": 22, "critic_input_width": 35, "policy_feature_count": 22,
        "policy_feature_order": tuple(C5.RUN5_POLICY_FEATURE_ORDER),
        "feature_schema_id": S5.FEATURE_SCHEMA_ID,
        "feature_schema_sha256": S5.FEATURE_SCHEMA_SHA256,
        "feature_schema_version": S5.FEATURE_SCHEMA_VERSION,
        "snr_scaling": "(snr_db - 5.5) / 19.0 on [5.5, 24.5]",
        "positions_0_20_feature_schema_sha256": R4.FEATURE_SCHEMA_SHA256,
    }


def model_config() -> HybridSacModelConfig:
    return HybridSacModelConfig(state_dim=22, mode_count=12, hidden_width=128,
                                hidden_depth=2, log_std_min=-5.0, log_std_max=2.0,
                                dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT)


def training_lease_policy() -> S5.SnrLeasePolicyV1:
    return S5.SnrLeasePolicyV1(
        policy_id="run5-modeled-controller-lease", policy_version=1,
        evidence_sha256=S5.NETWORK_PROFILE_DESIGN_SHA256,
        max_heartbeat_age_ns=LEASE_MAX_HEARTBEAT_AGE_NS)


def build_run5_live_state(*, snr: S5.UlSnrLeaseObservationV1 | None, **state_kwargs):
    """Phase-6 keyword interface plus an already-causal SNR observation.

    Calling the original live builder first preserves its raw-radio support
    check (before clipping). The frozen Run-5 builder then reproduces those
    same 21 values and appends exactly one admitted SNR feature.
    """
    prefix_guarded, prefix = _RUN4_BUILD_LIVE_STATE(**state_kwargs)
    state = R4.PolicyStateV2(**{name: state_kwargs[name] for name in (
        "identity", "camera_si", "radar_p40", "prior_ul_mcs",
        "pre_action_rlc_backlog", "previous")})
    guarded = S5.guard_run5_state_v2(state, snr, state_kwargs["boundary"],
                                    TRAINING_FRESHNESS, training_lease_policy())
    features = S5.build_run5_features_v2(guarded, LS.TRAINING_SCALING)
    _require(features.as_tuple()[:21] == prefix.as_tuple(), "Run-5 prefix parity failed")
    return prefix_guarded, features


class LiveStateBuilder:
    """Adapter for a provider with explicit session adoption and observe()."""
    def __init__(self, snr_provider) -> None:
        self.provider = snr_provider
        self.last_observation = None

    def build(self, **state_kwargs):
        identity = state_kwargs["identity"]
        self.provider.adopt_session(identity.session_uuid)
        observation = self.provider.observe(state_kwargs["boundary"])
        self.last_observation = observation
        return build_run5_live_state(snr=observation, **state_kwargs)


@dataclass(frozen=True, slots=True)
class Run5ActorDecision:
    mode_id: int
    q_e4: int
    q_support_lower_e4: int
    q_support_upper_e4: int
    state_sha256: str
    heads_sha256: str
    actor_boundary_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "scenesense.run5_live_avo.actor_decision.v1",
                **{name: getattr(self, name) for name in self.__dataclass_fields__}}


class FrozenRun5Actor:
    """Read-only CPU float32 batch-one deployment, no critic or optimizer."""
    def __init__(self, actor: ConditionalHybridActor) -> None:
        _require(type(actor) is ConditionalHybridActor, "foreign actor class")
        _require(actor.config == model_config(), "actor is not the native Run-5 config")
        _require(actor.modeled_smoke_support_sha256 == MODELED_SMOKE_SUPPORT_SHA256,
                 "actor q-support differs")
        for name, tensor in (*actor.named_parameters(), *actor.named_buffers()):
            _require(tensor.device.type == "cpu", f"{name} must remain on CPU")
            _require(not tensor.is_floating_point() or tensor.dtype == torch.float32,
                     f"{name} must remain float32")
        _require(F4.actor_boundary_sha256(actor) == ACTOR_TREE_SHA256, "actor tensor identity differs")
        self._actor = actor.eval().requires_grad_(False)
        low, high = actor.active_q_e4_bounds()
        self._lower, self._upper = tuple(map(int, low)), tuple(map(int, high))

    @property
    def boundary_sha256(self) -> str:
        return ACTOR_TREE_SHA256

    @property
    def module(self) -> ConditionalHybridActor:
        return self._actor

    def act_on_vector(self, values: Sequence[float]) -> Run5ActorDecision:
        _require(type(values) is tuple and len(values) == 22, "expected exact 22-feature tuple")
        _require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and
                     math.isfinite(float(v)) for v in values), "non-finite/non-real feature")
        _require(0.0 <= values[21] <= 1.0, "SNR feature is outside registered support")
        _require(not self._actor.training, "actor left eval mode")
        state = torch.tensor((tuple(float(v) for v in values),), dtype=torch.float32)
        with torch.inference_mode():
            heads = self._actor(state)
            execution = self._actor.deterministic_execution(state)
        mode, q = int(execution.mode_index[0]), int(execution.q_e4[0])
        _require(0 <= mode < 12 and self._lower[mode] <= q <= self._upper[mode],
                 "execution escaped registered mode/q support")
        return Run5ActorDecision(mode, q, self._lower[mode], self._upper[mode],
                                 _sha([float(v) for v in values]), F4._heads_digest(heads),
                                 ACTOR_TREE_SHA256)

    def act(self, features: S5.Run5FeatureVectorV2) -> Run5ActorDecision:
        _require(type(features) is S5.Run5FeatureVectorV2,
                 "actor requires an attested Run5FeatureVectorV2, not a same-width foreign state")
        return self.act_on_vector(features.as_tuple())


def registered_fixture_states() -> tuple[tuple[float, ...], ...]:
    """New RNG-free batch-one live fixtures; not old batched export fixtures."""
    return tuple((*row, float((index % 5) / 4.0))
                 for index, row in enumerate(F4.registered_fixture_states()))


def binding_document(actor: FrozenRun5Actor) -> dict[str, Any]:
    body = {
        "schema": "scenesense.run5_live_avo.actor_binding.v1",
        "selection_scope": SELECTION_SCOPE, "seed": SELECTED_SEED, "update": SELECTED_UPDATE,
        "weights_sha256": WEIGHTS_SHA256, "actor_tree_sha256": ACTOR_TREE_SHA256,
        "source_export_manifest_sha256": EXPORT_MANIFEST_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "deployment_rule": DEPLOYMENT_RULE, "model": native_model_binding(),
        "required_frozen_modules": verify_frozen_sources(),
        "run4_scaling_sha256": LS.TRAINING_SCALING_SHA256,
        "run4_freshness_sha256": TRAINING_FRESHNESS.canonical_sha256(),
        "snr_lease_policy_sha256": training_lease_policy().canonical_sha256(),
        "batch1_fixtures": [actor.act_on_vector(row).to_dict() for row in registered_fixture_states()],
    }
    return {**body, "binding_sha256": _sha(body)}


def load_source_export(export_dir: Path) -> FrozenRun5Actor:
    """Load only the pinned existing original Run-5 seed-43 actor export."""
    verify_frozen_sources()
    export_dir = Path(export_dir)
    manifest_path, weights = export_dir / "manifest.json", export_dir / "actor_state_dict.pt"
    for path, expected in ((manifest_path, EXPORT_MANIFEST_SHA256), (weights, WEIGHTS_SHA256)):
        _require(path.is_file() and not path.is_symlink(), f"missing/nonregular source: {path}")
        _require(F4.sha256_file(path) == expected, f"source SHA-256 mismatch: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "seed": SELECTED_SEED, "update_count": SELECTED_UPDATE,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "model_binding_sha256": NATIVE_MODEL_BINDING_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "actor_tree_sha256": ACTOR_TREE_SHA256,
        "source_manifest_sha256": SOURCE_CHECKPOINT_MANIFEST_SHA256,
        "source_bundle": "checkpoint_010000", "bundle_schema": "splitfusion.run5.boundary_bundle.v1",
    }
    _require(all(manifest.get(k) == v for k, v in expected.items()), "export provenance differs")
    _require(manifest["files"]["actor_state_dict.pt"]["sha256"] == WEIGHTS_SHA256,
             "manifest weights binding differs")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping), "weights are not a state_dict")
    _require(F4.actor_boundary_sha256(state) == ACTOR_TREE_SHA256, "weights tensor-tree hash differs")
    actor = build_actor(model_config(), seed=0)
    actor.load_state_dict(state, strict=True)
    return FrozenRun5Actor(actor)
