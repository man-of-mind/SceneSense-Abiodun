"""Frozen live adapter for the full Run-5 actor (all 12 modes, original continuous-q support; seed 43, update 10,000).

Selection scope: seed 43 was preselected for live use before training; NOT a claim that it was the best seed.

Interface-compatible with ``splitfusion_run5_live_avo_v1.actor_binding.FrozenRun5Actor`` (``act``, ``act_on_vector``,
``boundary_sha256``, ``module``, ``_lower``/``_upper``) and returns the same ``Run5ActorDecision`` record, so the
unchanged Phase-6 engine, Run-5 state builder (22 features, causal SNR lease) and profiler replay consume it as is.
The actor is the exact original ``ConditionalHybridActor`` in the native Run-5 configuration (original
``MODELED_SMOKE_SUPPORT`` q bounds, no mask); only its weights are new. Verified on load and on every call:
  * weights file and tensor-tree hashes equal the bound full-run actor (so the original Run-5 export and any other
    weights are refused, and the original Run-5 wrapper refuses this actor by its own tree pin);
  * the deterministic deployment rule (argmax mode at its conditional mean q, CPU float32 batch-1, eval, inference
    mode); every decision must lie inside the original support of its mode.
No learning: parameters are frozen (eval, requires_grad False), no optimizer exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as F4
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S5
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import ConditionalHybridActor, _local_torch_seed
from rl_agent.splitfusion_run5_full_v1 import support as SUP
from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A

SELECTED = {"run": "FULL_RUN5", "seed": 43, "update": 10_000}
SELECTION_SCOPE = "PRESELECTED_LIVE_CHOICE__FULL_RUN5_SEED43_UPDATE10000__NOT_A_BEST_SEED_CLAIM"
WEIGHTS_SHA256 = "bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce"
ACTOR_TREE_SHA256 = "4e28b9ff6b2173f466bc80781e3395f2b5666075d2926e3aa6b7aa848ca0039b"
SUPPORT_SHA256 = SUP.SUPPORT_SHA256
DEPLOYMENT_RULE = A.DEPLOYMENT_RULE
EXPORT_DIR = Path(__file__).resolve().with_name("actor_export")
ROOT = Path(__file__).resolve().parents[2]
SOURCE_WEIGHTS = "experiments/splitfusion_run5_full_v1/runs_01/seed43/actor_u10000.pt"


class FullLiveActorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullLiveActorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("ascii")).hexdigest()


class FrozenFullRun5Actor:
    """Read-only CPU float32 batch-one deployment of the full Run-5 actor; no critic, optimizer or learning."""

    def __init__(self, actor: ConditionalHybridActor) -> None:
        _require(type(actor) is ConditionalHybridActor, "foreign actor class")
        _require(actor.config == A.model_config(), "actor is not the native Run-5 configuration")
        actor._validate_registered_support_buffers()
        low, high = actor.active_q_e4_bounds()
        self._lower, self._upper = tuple(map(int, low)), tuple(map(int, high))
        _require(tuple(zip(self._lower, self._upper)) == SUP.BOUNDS, "actor support differs from the original support")
        _require(not any("mask" in name for name, _ in actor.named_buffers()), "a mode mask is present")
        for name, tensor in (*actor.named_parameters(), *actor.named_buffers()):
            _require(tensor.device.type == "cpu", f"{name} must remain on CPU")
            _require(not tensor.is_floating_point() or tensor.dtype == torch.float32, f"{name} must remain float32")
        _require(F4.actor_boundary_sha256(actor) == ACTOR_TREE_SHA256, "actor tensor identity differs")
        self._actor = actor.eval().requires_grad_(False)

    @property
    def boundary_sha256(self) -> str:
        return ACTOR_TREE_SHA256

    @property
    def module(self) -> ConditionalHybridActor:
        return self._actor

    def act_on_vector(self, values: Sequence[float]) -> A.Run5ActorDecision:
        _require(type(values) is tuple and len(values) == 22, "expected exact 22-feature tuple")
        _require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))
                     for v in values), "non-finite/non-real feature")
        _require(0.0 <= values[21] <= 1.0, "SNR feature is outside registered support")
        _require(not self._actor.training, "actor left eval mode")
        state = torch.tensor((tuple(float(v) for v in values),), dtype=torch.float32)
        with torch.inference_mode():
            heads = self._actor(state)
            execution = self._actor.deterministic_execution(state)
        mode, q = int(execution.mode_index[0]), int(execution.q_e4[0])
        _require(SUP.in_support(mode, q), f"execution escaped the original support ({mode}, {q})")
        return A.Run5ActorDecision(mode, q, self._lower[mode], self._upper[mode], A._sha([float(v) for v in values]),
                                   F4._heads_digest(heads), ACTOR_TREE_SHA256)

    def act(self, features: S5.Run5FeatureVectorV2) -> A.Run5ActorDecision:
        _require(type(features) is S5.Run5FeatureVectorV2,
                 "actor requires an attested Run5FeatureVectorV2, not a same-width foreign state")
        return self.act_on_vector(features.as_tuple())


def _load_state(weights: Path) -> Mapping[str, torch.Tensor]:
    _require(weights.is_file() and not weights.is_symlink(), f"missing/nonregular weights: {weights}")
    _require(F4.sha256_file(weights) == WEIGHTS_SHA256, "weights SHA-256 differs from the bound full Run-5 actor")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping), "weights are not a state_dict")
    _require(F4.actor_boundary_sha256(state) == ACTOR_TREE_SHA256, "weights tensor-tree hash differs")
    return state


def _module(state) -> FrozenFullRun5Actor:
    with _local_torch_seed(0):                  # RNG-neutral construction (as the original loader)
        module = ConditionalHybridActor(A.model_config())
    module.load_state_dict(state, strict=True)
    return FrozenFullRun5Actor(module)


def load_export(export_dir: Path = EXPORT_DIR) -> FrozenFullRun5Actor:
    A.verify_frozen_sources()                       # same frozen Run-5 state/SNR sources as the original binding
    export_dir = Path(export_dir)
    binding = json.loads((export_dir / "ACTOR_BINDING.json").read_text())
    _require(binding["actor"]["weights_file_sha256"] == WEIGHTS_SHA256
             and binding["actor"]["boundary_sha256"] == ACTOR_TREE_SHA256, "binding names a different actor")
    actor = _module(_load_state(export_dir / "actor_state_dict.pt"))
    fixtures = [actor.act_on_vector(row).to_dict() for row in A.registered_fixture_states()]
    _require(fixtures == binding["batch1_fixtures"], "batch-1 fixtures differ from the sealed binding")
    return actor


def build_export(export_dir: Path = EXPORT_DIR) -> dict[str, Any]:
    """Create the package-local export from the full run's seed-43 actor (create-only)."""
    export_dir = Path(export_dir)
    _require(not export_dir.exists(), "export is create-only")
    source = ROOT / SOURCE_WEIGHTS
    actor = _module(_load_state(source))
    run_dir = source.parent
    result = json.loads((run_dir / "RUN_RESULT.json").read_text())
    _require(result["actor"]["sha256"] == WEIGHTS_SHA256 and result["updates"] == SELECTED["update"]
             and result["seed"] == SELECTED["seed"], "run result names a different actor")
    export_dir.mkdir(parents=True)
    shutil.copyfile(source, export_dir / "actor_state_dict.pt")
    _require(F4.sha256_file(export_dir / "actor_state_dict.pt") == WEIGHTS_SHA256, "copied weights differ")
    body = dict(
        schema="scenesense.run5_full_live.actor_binding.v1", selection_scope=SELECTION_SCOPE, selected=SELECTED,
        actor=dict(boundary_sha256=ACTOR_TREE_SHA256, weights_file_name="actor_state_dict.pt",
                   weights_file_sha256=WEIGHTS_SHA256, actor_class="ConditionalHybridActor"),
        binding=dict(feature_schema_id=S5.FEATURE_SCHEMA_ID, feature_schema_sha256=S5.FEATURE_SCHEMA_SHA256,
                     policy_feature_count=22, policy_feature_order=list(A.native_model_binding()["policy_feature_order"]),
                     deployment_rule=DEPLOYMENT_RULE, quality_in_state="prev_quality_qperc = original Q_perc"),
        support=dict(modes=12, bounds=[list(b) for b in SUP.BOUNDS], mask=None,
                     source="MODELED_SMOKE_SUPPORT (original Run-5 continuous-q support)"),
        support_sha256=SUPPORT_SHA256, frozen_dependencies=A.verify_frozen_sources(),
        provenance=dict(source_weights=SOURCE_WEIGHTS, run_result_sha256=F4.sha256_file(run_dir / "RUN_RESULT.json"),
                        run_binding_sha256=F4.sha256_file(run_dir / "RUN_BINDING.json"),
                        training_authorization="experiments/splitfusion_run5_full_v1/authorization_01/AUTHORIZATION.json",
                        evidence_label="FULL_RUN5_MODELED_TRAINING; live use is frozen inference only"),
        batch1_fixtures=[actor.act_on_vector(row).to_dict() for row in A.registered_fixture_states()])
    body["manifest_sha256"] = _sha(body)
    (export_dir / "ACTOR_BINDING.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    (export_dir / "COMMITTED").write_text(json.dumps({"weights": WEIGHTS_SHA256, "binding": F4.sha256_file(
        export_dir / "ACTOR_BINDING.json")}) + "\n")
    return body


def load_registered_actor() -> FrozenFullRun5Actor:
    """Zero-argument loader for the profiler's retained G8 actor replay (``semantics.actor_loader``)."""
    return load_export()


def audit(export_dir: Path = EXPORT_DIR) -> dict[str, Any]:
    rng_before = F4.rng_fingerprint()
    cuda_before = torch.cuda.is_initialized()
    actor = load_export(export_dir)
    binding = json.loads((Path(export_dir) / "ACTOR_BINDING.json").read_text())
    weights = Path(export_dir) / "actor_state_dict.pt"
    checks = {
        "full_run5_seed43_update10000": binding["selected"] == SELECTED,
        "original_quality_aware_run5_22_features": (binding["binding"]["policy_feature_count"] == 22
                                                    and binding["binding"]["policy_feature_order"][17] == "prev_quality_qperc"
                                                    and binding["binding"]["policy_feature_order"][21]
                                                    == "effective_external_ul_snr_proxy_scaled"),
        "weights_file_sha256": F4.sha256_file(weights) == WEIGHTS_SHA256,
        "loaded_actor_boundary_sha256": actor.boundary_sha256 == ACTOR_TREE_SHA256,
        "original_support_no_mask": (binding["support_sha256"] == SUPPORT_SHA256 == SUP.SUPPORT_SHA256
                                     and binding["support"]["mask"] is None
                                     and tuple(zip(actor._lower, actor._upper)) == SUP.BOUNDS),
        "fixtures_equal_sealed_binding": [actor.act_on_vector(r).to_dict() for r in A.registered_fixture_states()]
        == binding["batch1_fixtures"],
        "cpu_float32_readonly": all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
                                    for p in actor.module.parameters()),
        "eval_mode": not actor.module.training,
        "rng_neutral": rng_before == F4.rng_fingerprint(),
        "cuda_not_initialized": not cuda_before and not torch.cuda.is_initialized(),
    }
    return {"schema": "scenesense.run5_full_live.actor_audit.v1", "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "selected": SELECTED, "selection_scope": SELECTION_SCOPE, "weights_file": str(weights),
            "weights_file_sha256": F4.sha256_file(weights), "actor_boundary_sha256": actor.boundary_sha256,
            "binding_sha256": binding["manifest_sha256"], "support_sha256": SUPPORT_SHA256,
            "fixture_count": len(binding["batch1_fixtures"]), "cuda_initialized": torch.cuda.is_initialized(),
            "training_or_reconstruction_performed": False}
