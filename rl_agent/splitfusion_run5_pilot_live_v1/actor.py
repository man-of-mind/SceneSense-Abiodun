"""Frozen live adapter for the restricted Run-5 pilot actor (CANDIDATE arm, seed 43, update 10,000).

Selection scope: explicitly chosen as the live-test candidate; NOT a claim that it was the best-performing seed.

Interface-compatible with ``splitfusion_run5_live_avo_v1.actor_binding.FrozenRun5Actor`` (``act``, ``act_on_vector``,
``boundary_sha256``, ``module``, ``_lower``/``_upper``) and returns the same ``Run5ActorDecision`` record, so the
unchanged Phase-6 engine, Run-5 state builder (22 features, causal SNR lease) and profiler replay consume it as is.
Differences are only those of the pilot actor itself, each verified on load and on every call:
  * class ``PilotRestrictedActor`` (original Run-5 network) with the pilot's registered per-mode q bounds and mask;
  * the deterministic deployment rule (argmax mode at its conditional mean q, CPU float32 batch-1, eval, inference
    mode) applied to the MASKED logits; every decision must be one of modes 5/8/11 inside its bracket.
The original Run-5 loader and its checks are untouched (it refuses this checkpoint); this adapter refuses the
original checkpoint. No learning: parameters are frozen (eval, requires_grad False), no optimizer exists.
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
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import _local_torch_seed
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S5
from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A
from rl_agent.splitfusion_run5_restricted_pilot_v1 import support as SUP

SELECTED = {"arm": "CANDIDATE", "seed": 43, "update": 10_000}
SELECTION_SCOPE = "LIVE_TEST_CHOICE__PILOT_CANDIDATE_SEED43_UPDATE10000__NOT_A_BEST_SEED_CLAIM"
WEIGHTS_SHA256 = "ecd7ce4ae20de037fe41670a91406259fc2567f9228adfe1084a55e507c5897c"
ACTOR_TREE_SHA256 = "bf8b0e96c7023dfca0b3e5323b2829b952afe0d7c3a385b727c57268e5323a21"
SUPPORT_SHA256 = "40136534525c17625c26d06c98507bf342fdfb4d980ef500f72e2d67cfbe9e9e"
DEPLOYMENT_RULE = A.DEPLOYMENT_RULE
EXPORT_DIR = Path(__file__).resolve().with_name("actor_export")
ROOT = Path(__file__).resolve().parents[2]
SOURCE_WEIGHTS = "experiments/splitfusion_run5_restricted_pilot_v1/runs_01/CANDIDATE_seed43/actor_u10000.pt"


class PilotLiveActorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotLiveActorError(message)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("ascii")).hexdigest()


class FrozenPilotRun5Actor:
    """Read-only CPU float32 batch-one deployment of the pilot actor; no critic, optimizer or learning."""

    def __init__(self, actor: SUP.PilotRestrictedActor) -> None:
        _require(type(actor) is SUP.PilotRestrictedActor, "foreign actor class")
        _require(actor.config == A.model_config(), "actor is not the native Run-5 configuration")
        _require(SUP.SUPPORT_SHA256 == SUPPORT_SHA256, "pilot support contract differs from the bound one")
        actor._validate_registered_support_buffers()
        _require(torch.equal(actor._pilot_mask_bias, SUP.mask_bias()), "pilot mode mask differs")
        for name, tensor in (*actor.named_parameters(), *actor.named_buffers()):
            _require(tensor.device.type == "cpu", f"{name} must remain on CPU")
            _require(not tensor.is_floating_point() or tensor.dtype == torch.float32, f"{name} must remain float32")
        _require(F4.actor_boundary_sha256(actor) == ACTOR_TREE_SHA256, "actor tensor identity differs")
        self._actor = actor.eval().requires_grad_(False)
        low, high = actor.active_q_e4_bounds()
        self._lower, self._upper = tuple(map(int, low)), tuple(map(int, high))

    @property
    def boundary_sha256(self) -> str:
        return ACTOR_TREE_SHA256

    @property
    def module(self) -> SUP.PilotRestrictedActor:
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
        _require(SUP.in_support(mode, q) and self._lower[mode] <= q <= self._upper[mode],
                 f"execution escaped the pilot support ({mode}, {q})")
        return A.Run5ActorDecision(mode, q, self._lower[mode], self._upper[mode], A._sha([float(v) for v in values]),
                                   F4._heads_digest(heads), ACTOR_TREE_SHA256)

    def act(self, features: S5.Run5FeatureVectorV2) -> A.Run5ActorDecision:
        _require(type(features) is S5.Run5FeatureVectorV2,
                 "actor requires an attested Run5FeatureVectorV2, not a same-width foreign state")
        return self.act_on_vector(features.as_tuple())


def _load_state(weights: Path) -> Mapping[str, torch.Tensor]:
    _require(weights.is_file() and not weights.is_symlink(), f"missing/nonregular weights: {weights}")
    _require(F4.sha256_file(weights) == WEIGHTS_SHA256, "weights SHA-256 differs from the bound pilot actor")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping), "weights are not a state_dict")
    _require(F4.actor_boundary_sha256(state) == ACTOR_TREE_SHA256, "weights tensor-tree hash differs")
    return state


def load_export(export_dir: Path = EXPORT_DIR) -> FrozenPilotRun5Actor:
    A.verify_frozen_sources()                       # same frozen Run-5 state/SNR sources as the original binding
    export_dir = Path(export_dir)
    binding = json.loads((export_dir / "ACTOR_BINDING.json").read_text())
    _require(binding["actor"]["weights_file_sha256"] == WEIGHTS_SHA256
             and binding["actor"]["boundary_sha256"] == ACTOR_TREE_SHA256, "binding names a different actor")
    state = _load_state(export_dir / "actor_state_dict.pt")
    with _local_torch_seed(0):                  # RNG-neutral construction (as the original loader)
        module = SUP.PilotRestrictedActor(A.model_config())
    module.load_state_dict(state, strict=True)
    actor = FrozenPilotRun5Actor(module)
    fixtures = [actor.act_on_vector(row).to_dict() for row in A.registered_fixture_states()]
    _require(fixtures == binding["batch1_fixtures"], "batch-1 fixtures differ from the sealed binding")
    return actor


def build_export(export_dir: Path = EXPORT_DIR) -> dict[str, Any]:
    """Create the package-local export from the pilot run's actor (create-only)."""
    export_dir = Path(export_dir)
    _require(not export_dir.exists(), "export is create-only")
    source = ROOT / SOURCE_WEIGHTS
    state = _load_state(source)
    with _local_torch_seed(0):                  # RNG-neutral construction (as the original loader)
        module = SUP.PilotRestrictedActor(A.model_config())
    module.load_state_dict(state, strict=True)
    actor = FrozenPilotRun5Actor(module)
    run_dir = source.parent
    export_dir.mkdir(parents=True)
    shutil.copyfile(source, export_dir / "actor_state_dict.pt")
    _require(F4.sha256_file(export_dir / "actor_state_dict.pt") == WEIGHTS_SHA256, "copied weights differ")
    body = dict(
        schema="scenesense.run5_pilot_live.actor_binding.v1", selection_scope=SELECTION_SCOPE, selected=SELECTED,
        actor=dict(boundary_sha256=ACTOR_TREE_SHA256, weights_file_name="actor_state_dict.pt",
                   weights_file_sha256=WEIGHTS_SHA256, actor_class="PilotRestrictedActor"),
        binding=dict(feature_schema_id=S5.FEATURE_SCHEMA_ID, feature_schema_sha256=S5.FEATURE_SCHEMA_SHA256,
                     policy_feature_count=22, policy_feature_order=list(A.native_model_binding()["policy_feature_order"]),
                     deployment_rule=DEPLOYMENT_RULE, quality_in_state="prev_quality_qperc = original Q_perc"),
        support=dict(SUP.SUPPORT_DOCUMENT), support_sha256=SUPPORT_SHA256,
        frozen_dependencies=A.verify_frozen_sources(),
        provenance=dict(source_weights=SOURCE_WEIGHTS,
                        run_result_sha256=F4.sha256_file(run_dir / "RUN_RESULT.json"),
                        run_binding_sha256=F4.sha256_file(run_dir / "RUN_BINDING.json"),
                        training_authorization="experiments/splitfusion_run5_restricted_pilot_v1/authorization_02/PILOT_AUTHORIZATION.json",
                        evidence_label="PILOT_AUTHORIZED_MODELED training; live use is frozen inference only"),
        batch1_fixtures=[actor.act_on_vector(row).to_dict() for row in A.registered_fixture_states()])
    body["manifest_sha256"] = _sha(body)
    (export_dir / "ACTOR_BINDING.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    (export_dir / "COMMITTED").write_text(json.dumps({"weights": WEIGHTS_SHA256, "binding": F4.sha256_file(
        export_dir / "ACTOR_BINDING.json")}) + "\n")
    return body


def load_registered_actor() -> FrozenPilotRun5Actor:
    """Zero-argument loader for the profiler's retained G8 actor replay (``semantics.actor_loader``)."""
    return load_export()


def audit(export_dir: Path = EXPORT_DIR) -> dict[str, Any]:
    rng_before = F4.rng_fingerprint()
    cuda_before = torch.cuda.is_initialized()
    actor = load_export(export_dir)
    binding = json.loads((Path(export_dir) / "ACTOR_BINDING.json").read_text())
    weights = Path(export_dir) / "actor_state_dict.pt"
    checks = {
        "candidate_seed43_update10000": binding["selected"] == SELECTED,
        "original_quality_aware_run5_22_features": (binding["binding"]["policy_feature_count"] == 22
                                                    and binding["binding"]["policy_feature_order"][17] == "prev_quality_qperc"
                                                    and binding["binding"]["policy_feature_order"][21]
                                                    == "effective_external_ul_snr_proxy_scaled"),
        "weights_file_sha256": F4.sha256_file(weights) == WEIGHTS_SHA256,
        "loaded_actor_boundary_sha256": actor.boundary_sha256 == ACTOR_TREE_SHA256,
        "pilot_support_and_mask": binding["support_sha256"] == SUPPORT_SHA256 == SUP.SUPPORT_SHA256,
        "fixtures_equal_sealed_binding": [actor.act_on_vector(r).to_dict() for r in A.registered_fixture_states()]
        == binding["batch1_fixtures"],
        "cpu_float32_readonly": all(p.device.type == "cpu" and p.dtype == torch.float32 and not p.requires_grad
                                    for p in actor.module.parameters()),
        "eval_mode": not actor.module.training,
        "rng_neutral": rng_before == F4.rng_fingerprint(),
        "cuda_not_initialized": not cuda_before and not torch.cuda.is_initialized(),
    }
    return {"schema": "scenesense.run5_pilot_live.actor_audit.v1", "verdict": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "selected": SELECTED, "selection_scope": SELECTION_SCOPE, "weights_file": str(weights),
            "weights_file_sha256": F4.sha256_file(weights), "actor_boundary_sha256": actor.boundary_sha256,
            "binding_sha256": binding["manifest_sha256"], "support_sha256": SUPPORT_SHA256,
            "fixture_count": len(binding["batch1_fixtures"]), "cuda_initialized": torch.cuda.is_initialized(),
            "training_or_reconstruction_performed": False}
