"""Frozen, weights-only deployment actor for the selected Run-4 checkpoint.

The Run-4 JSON checkpoint is an event-sourced ledger, not a tensor snapshot.
The export CLI (:mod:`export_frozen_actor_v2`) restores it once through the
registered factories and writes a compact CPU ``state_dict``.  This module
owns everything else:

* the pinned identity of the post-hoc qualification candidate
  (seed 43, update 10,000);
* the boundary digest, computed exactly as
  ``ModeledSmokeOrchestratorV1._boundary`` computes ``actor_sha256``;
* a deterministic, RNG-free batch-1 fixture set;
* the manifest schema and its self-digest; and
* :func:`load_frozen_actor`, which re-verifies every one of those facts
  before returning a :class:`FrozenRun4ActorV2`.

The deployment rule is fixed: CPU float32, batch size 1, ``eval()`` and
inference mode, categorical argmax for the mode, the selected mode's
conditional mean mapped through its registered q support, and the registered
decimal half-up conversion to ``q_e4``.  Those semantics are
``ConditionalHybridActor.deterministic_execution`` unchanged.

Importing this module performs no file I/O and does not initialize CUDA.
Nothing here constructs an optimizer, a critic or a replay buffer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as run4_models
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tree_sha256 as boundary_tree_sha256,
)
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor,
    build_actor,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)

__all__ = [
    "FrozenActorError",
    "MANIFEST_SCHEMA",
    "REPOSITORY_ROOT",
    "SELECTED",
    "SelectedCandidateV1",
    "Run4ActorDecisionV2",
    "FrozenRun4ActorV2",
    "actor_boundary_sha256",
    "tensor_inventory",
    "registered_fixture_states",
    "fixture_outputs",
    "binding_document",
    "manifest_digest",
    "seal_manifest",
    "verify_manifest",
    "load_frozen_actor",
    "load_registered_actor",
    "ACTOR_EXPORT_RELPATH",
    "TRACKED_BINDING_PATH",
]


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "scenesense.run4_live_v2.frozen_actor_manifest.v1"
DECISION_SCHEMA = "scenesense.run4_live_v2.actor_decision.v1"
DEPLOYMENT_RULE = (
    "CPU_FLOAT32_BATCH1_EVAL_INFERENCE_MODE__ARGMAX_MODE__SELECTED_"
    "CONDITIONAL_MEAN__REGISTERED_PER_MODE_SUPPORT__DECIMAL_HALF_UP_Q_E4"
)
SELECTION_SCOPE = (
    "POSTHOC_DEVELOPMENT_CHOICE_FOR_BOUNDED_300_FRAME_QUALIFICATION__"
    "NOT_A_CONVERGENCE_OR_OPTIMALITY_CLAIM"
)


class FrozenActorError(RuntimeError):
    """The frozen actor, its manifest or one pinned source differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenActorError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pinned candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectedCandidateV1:
    seed: int
    update: int
    selection_record_relpath: str
    selection_record_sha256: str
    checkpoint_relpath: str
    checkpoint_file_sha256: str
    canonical_checkpoint_sha256: str
    actor_boundary_sha256: str
    seed_complete_relpath: str
    seed_complete_sha256: str
    checkpoint_manifest_relpath: str
    checkpoint_manifest_sha256: str
    transport_artifact_relpath: str
    transport_artifact_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


_CAMPAIGN = (
    "rl_agent/experiments/splitfusion_hybrid_sac_run4_v2_campaign/"
    "20260928_2a92201_three_seed_10000_v1/seed_43"
)

SELECTED = SelectedCandidateV1(
    seed=43,
    update=10_000,
    selection_record_relpath=(
        "experiments/splitfusion_hybrid_sac_run4_training_analysis_v1/"
        "20260929_advisor_plots/PROVISIONAL_LIVE_CANDIDATE.md"
    ),
    selection_record_sha256=(
        "d231653a9998f61eb58a6c713f5bb5af8a4d8e9764cfaa6e618ca98e52716413"
    ),
    checkpoint_relpath=f"{_CAMPAIGN}/checkpoints/update_010000.checkpoint.json",
    checkpoint_file_sha256=(
        "3db78b8433223be10386b8c4efacb58989577b0e057e6d13f55d19c718f4e775"
    ),
    canonical_checkpoint_sha256=(
        "aba44d2d51eb73aea87554c912e18062f6ee73e4d086d88714ad455507164672"
    ),
    actor_boundary_sha256=(
        "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3"
    ),
    seed_complete_relpath=f"{_CAMPAIGN}/SEED_COMPLETE.json",
    seed_complete_sha256=(
        "5f11e37e6de5a9c8443a4da8a731525a724b7d7983cb48df0f93e0021dbd698f"
    ),
    checkpoint_manifest_relpath=f"{_CAMPAIGN}/CHECKPOINT_MANIFEST.json",
    checkpoint_manifest_sha256=(
        "1ba8131db86f3d8e694ddec6e35ae0079bd766358f0ec64d8e290ff9b552a1d7"
    ),
    transport_artifact_relpath=(
        "rl_agent/experiments/ue_production_queue_capture_v1/"
        "20260929_model_v2b/transport_model_v2.json"
    ),
    transport_artifact_sha256=(
        "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c"
    ),
)


def verify_pinned_sources(repo_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """Re-hash every pinned source file; never trust a recorded digest."""
    observed: dict[str, str] = {}
    for relpath, expected in (
        (SELECTED.selection_record_relpath, SELECTED.selection_record_sha256),
        (SELECTED.checkpoint_relpath, SELECTED.checkpoint_file_sha256),
        (SELECTED.seed_complete_relpath, SELECTED.seed_complete_sha256),
        (SELECTED.checkpoint_manifest_relpath, SELECTED.checkpoint_manifest_sha256),
        (SELECTED.transport_artifact_relpath, SELECTED.transport_artifact_sha256),
    ):
        path = repo_root / relpath
        _require(path.is_file() and not path.is_symlink(),
                 f"pinned source missing or not a regular file: {relpath}")
        actual = sha256_file(path)
        _require(actual == expected,
                 f"pinned source drifted: {relpath}: {actual} != {expected}")
        observed[relpath] = actual
    return observed


# ---------------------------------------------------------------------------
# Tensor identity
# ---------------------------------------------------------------------------


def actor_boundary_sha256(actor_or_state: Any) -> str:
    """The exact ``BoundaryFingerprintV1.actor_sha256`` definition."""
    state = (
        actor_or_state.state_dict()
        if isinstance(actor_or_state, torch.nn.Module)
        else actor_or_state
    )
    return boundary_tree_sha256(state)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    header = _canonical_bytes({"dtype": str(value.dtype),
                               "shape": list(value.shape)})
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def tensor_inventory(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "dtype": str(state[name].dtype),
            "shape": list(state[name].shape),
            "tensor_sha256": _tensor_sha256(state[name]),
        }
        for name in sorted(state)
    ]


# ---------------------------------------------------------------------------
# Registered fixtures: a fixed, RNG-free grid over the 21-feature support
# ---------------------------------------------------------------------------


def registered_fixture_states() -> Tuple[Tuple[float, ...], ...]:
    """Deterministic batch-1 fixtures covering genesis, success and failure.

    Built from arithmetic rules only (no RNG), so a verifier can rebuild them.
    Every fixture is a legal output of ``build_policy_features``: previous
    fields follow the genesis/success/failure encoding exactly.
    """
    fixtures: list[Tuple[float, ...]] = []
    cameras = (-2.0, -0.5, 0.0, 0.75, 2.5)
    radars = (0.0, 0.2, 0.55, 0.9)
    mcs_indices = (0, 9, 17, 28)
    backlogs = (0, 40_000, 1_000_000)
    index = 0
    for camera in cameras:
        for radar in radars:
            mcs = mcs_indices[index % len(mcs_indices)]
            backlog = backlogs[index % len(backlogs)]
            previous = index % 3  # 0 genesis, 1 success, 2 failure/timeout
            mode = index % ac.EXPECTED_MODE_COUNT
            q_e4 = (index * 1237) % (ac.Q_E4_MAX + 1)
            one_hot = [0.0] * ac.EXPECTED_MODE_COUNT
            if previous == 0:
                tail = (0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                one_hot[mode] = 1.0
                if previous == 1:
                    q_perc = (index % 7) / 7.0
                    latency_ms = 20.0 + 7.5 * (index % 20)
                    tail = (q_e4 / float(ac.Q_E4_MAX), q_perc,
                            latency_ms / contract.REWARD_DEADLINE_MS, 1.0, 1.0)
                else:
                    tail = (q_e4 / float(ac.Q_E4_MAX), 0.0, 0.0, 1.0, 0.0)
            fixtures.append((
                float(camera), float(radar),
                (mcs - contract.UL_MCS_INDEX_MIN)
                / float(contract.UL_MCS_INDEX_MAX - contract.UL_MCS_INDEX_MIN),
                math.log1p(backlog) / math.log1p(50_000_000),
                *one_hot, *tail,
            ))
            index += 1
    for fixture in fixtures:
        _require(len(fixture) == contract.POLICY_FEATURE_COUNT,
                 "fixture width differs from the 21-feature contract")
    return tuple(fixtures)


def _heads_digest(heads: Any) -> str:
    return _sha({
        "logits": _tensor_sha256(heads.logits),
        "mean": _tensor_sha256(heads.mean),
        "log_std": _tensor_sha256(heads.log_std),
    })


def fixture_outputs(actor: ConditionalHybridActor) -> list[dict[str, Any]]:
    """Batch-1 deterministic outputs for every registered fixture."""
    outputs = []
    with torch.inference_mode():
        for ordinal, values in enumerate(registered_fixture_states()):
            state = torch.tensor((values,), dtype=torch.float32)
            heads = actor(state)
            execution = actor.deterministic_execution(state)
            outputs.append({
                "ordinal": ordinal,
                "state_sha256": _sha(list(values)),
                "heads_sha256": _heads_digest(heads),
                "mode_id": int(execution.mode_index[0]),
                "q_e4": int(execution.q_e4[0]),
            })
    return outputs


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def binding_document() -> dict[str, Any]:
    """Feature/action/model bindings the actor was trained against."""
    lower = [int(item[0]) for item in MODELED_SMOKE_SUPPORT.mode_q_e4_bounds]
    upper = [int(item[1]) for item in MODELED_SMOKE_SUPPORT.mode_q_e4_bounds]
    return {
        "action_catalog_sha256": ac.CATALOG_SHA256,
        "continuous_q_support_sha256": MODELED_SMOKE_SUPPORT_SHA256,
        "deployment_rule": DEPLOYMENT_RULE,
        "dtype": "torch.float32",
        "feature_schema_id": contract.FEATURE_SCHEMA_ID,
        "feature_schema_sha256": contract.FEATURE_SCHEMA_SHA256,
        "mode_count": ac.EXPECTED_MODE_COUNT,
        "policy_feature_count": contract.POLICY_FEATURE_COUNT,
        "policy_feature_order": list(contract.POLICY_FEATURE_ORDER),
        "q_e4_max": ac.Q_E4_MAX,
        "q_e4_scale": ac.Q_E4_SCALE,
        "q_support_lower_e4": lower,
        "q_support_upper_e4": upper,
        "run4_contract_schema_sha256": contract.SCHEMA_SHA256,
        "run4_model_binding_sha256": run4_models.RUN4_MODEL_BINDING_SHA256,
    }


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    return _sha(body)


def seal_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    _require("manifest_sha256" not in body, "manifest body is already sealed")
    sealed = dict(body)
    sealed["manifest_sha256"] = manifest_digest(body)
    return sealed


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on any drift between a manifest and the code's bindings."""
    _require(isinstance(manifest, Mapping), "manifest must be a JSON object")
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "foreign manifest schema")
    _require(manifest.get("manifest_sha256") == manifest_digest(manifest),
             "manifest self-digest differs (tampered or unsealed)")
    _require(manifest.get("selected") == SELECTED.to_dict(),
             "manifest selected candidate differs from the pinned seed/update")
    _require(manifest.get("binding") == binding_document(),
             "manifest feature/action/model binding differs from code")
    actor = manifest.get("actor")
    _require(isinstance(actor, Mapping), "manifest actor section is missing")
    _require(actor.get("boundary_sha256") == SELECTED.actor_boundary_sha256,
             "manifest actor digest differs from the checkpoint boundary")
    _require(actor.get("weights_format") == "torch.save_state_dict_weights_only",
             "manifest weights format differs")
    fixtures = manifest.get("fixtures")
    _require(isinstance(fixtures, list) and len(fixtures)
             == len(registered_fixture_states()),
             "manifest fixture set differs from the registered fixtures")
    for expected, recorded in zip(registered_fixture_states(), fixtures):
        _require(recorded.get("state_sha256") == _sha(list(expected)),
                 "manifest fixture state differs from the registered grid")


# ---------------------------------------------------------------------------
# Deployment actor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run4ActorDecisionV2:
    mode_id: int
    q_e4: int
    q_support_lower_e4: int
    q_support_upper_e4: int
    state_sha256: str
    heads_sha256: str
    actor_boundary_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": DECISION_SCHEMA,
                **{name: getattr(self, name) for name in self.__dataclass_fields__}}


class FrozenRun4ActorV2:
    """Read-only CPU actor with the registered deterministic deployment rule."""

    def __init__(self, actor: ConditionalHybridActor, *, boundary_sha256: str) -> None:
        _require(type(actor) is ConditionalHybridActor,
                 "actor must be an exact ConditionalHybridActor")
        _require(actor.config == run4_models.run4_model_config(),
                 "actor configuration differs from run4_model_config()")
        _require(actor.modeled_smoke_support_sha256 == MODELED_SMOKE_SUPPORT_SHA256,
                 "actor q-support digest differs")
        for name, value in list(actor.named_parameters()) + list(actor.named_buffers()):
            _require(value.device.type == "cpu", f"{name} is not on CPU")
            if value.is_floating_point():
                _require(value.dtype is torch.float32, f"{name} is not float32")
        actor.eval()
        actor.requires_grad_(False)
        _require(actor_boundary_sha256(actor) == boundary_sha256,
                 "actor tensors differ from the pinned boundary digest")
        self._actor = actor
        self._boundary_sha256 = boundary_sha256
        lower, upper = actor.active_q_e4_bounds()
        self._lower = tuple(int(item) for item in lower)
        self._upper = tuple(int(item) for item in upper)

    @property
    def boundary_sha256(self) -> str:
        return self._boundary_sha256

    @property
    def module(self) -> ConditionalHybridActor:
        """Exposed for verification only; parameters are frozen."""
        return self._actor

    def act_on_vector(self, values: Sequence[float]) -> Run4ActorDecisionV2:
        """Run the deployment rule on one exact 21-feature vector."""
        _require(type(values) is tuple, "features must be an exact tuple")
        _require(len(values) == contract.POLICY_FEATURE_COUNT,
                 "feature width differs from the 21-feature contract")
        for index, value in enumerate(values):
            _require(not isinstance(value, bool)
                     and isinstance(value, (int, float))
                     and math.isfinite(float(value)),
                     f"feature {index} is not a finite real")
        _require(not self._actor.training, "actor left eval mode")
        state = torch.tensor((tuple(float(v) for v in values),),
                             dtype=torch.float32)
        with torch.inference_mode():
            heads = self._actor(state)
            execution = self._actor.deterministic_execution(state)
        mode_id = int(execution.mode_index[0])
        q_e4 = int(execution.q_e4[0])
        _require(0 <= mode_id < ac.EXPECTED_MODE_COUNT, "mode out of range")
        _require(self._lower[mode_id] <= q_e4 <= self._upper[mode_id],
                 "q_e4 escaped the selected mode's registered support")
        return Run4ActorDecisionV2(
            mode_id=mode_id,
            q_e4=q_e4,
            q_support_lower_e4=self._lower[mode_id],
            q_support_upper_e4=self._upper[mode_id],
            state_sha256=_sha([float(v) for v in values]),
            heads_sha256=_heads_digest(heads),
            actor_boundary_sha256=self._boundary_sha256,
        )

    def act(self, features: contract.PolicyFeatureVectorV2) -> Run4ActorDecisionV2:
        """Act only on an attested ``build_policy_features`` vector."""
        _require(type(features) is contract.PolicyFeatureVectorV2,
                 "features must be an attested PolicyFeatureVectorV2")
        return self.act_on_vector(features.as_tuple())


def _empty_actor() -> ConditionalHybridActor:
    # build_actor seeds locally and restores the global RNG state.
    return build_actor(run4_models.run4_model_config(), seed=0)


def load_frozen_actor(
    weights_path: Path,
    manifest: Mapping[str, Any],
) -> FrozenRun4ActorV2:
    """Verify manifest, weights file, tensors and fixtures; return the actor."""
    verify_manifest(manifest)
    weights_path = Path(weights_path)
    _require(weights_path.is_file() and not weights_path.is_symlink(),
             "weights path must be a regular non-symlink file")
    _require(sha256_file(weights_path) == manifest["actor"]["weights_file_sha256"],
             "weights file SHA-256 differs from the manifest")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    _require(isinstance(state, Mapping), "weights file is not a state_dict")
    _require(tensor_inventory(state) == manifest["actor"]["tensor_inventory"],
             "weights tensors differ from the manifest inventory")
    actor = _empty_actor()
    actor.load_state_dict(state, strict=True)
    frozen = FrozenRun4ActorV2(actor, boundary_sha256=SELECTED.actor_boundary_sha256)
    _require(fixture_outputs(frozen.module) == manifest["fixtures"],
             "reloaded actor fixture outputs differ from the restored actor")
    return frozen


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON root must be an object")
    return value


# The weights file is gitignored evidence; the tracked binding below is a byte
# copy of the export manifest and pins the weights file by SHA-256.
ACTOR_EXPORT_RELPATH = (
    "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
    "20260929_seed43_update10000_actor_export"
)
TRACKED_BINDING_PATH = Path(__file__).resolve().with_name("ACTOR_BINDING_V2.json")


def load_registered_actor(
    repo_root: Path = REPOSITORY_ROOT,
) -> FrozenRun4ActorV2:
    """Load the pinned export through the tracked binding, fail-closed."""
    export_dir = repo_root / ACTOR_EXPORT_RELPATH
    exported = export_dir / "ACTOR_EXPORT_MANIFEST.json"
    _require(exported.is_file(), f"actor export manifest missing: {exported}")
    _require(TRACKED_BINDING_PATH.read_bytes() == exported.read_bytes(),
             "tracked ACTOR_BINDING_V2.json differs from the export manifest")
    manifest = load_json(TRACKED_BINDING_PATH)
    return load_frozen_actor(
        export_dir / manifest["actor"]["weights_file_name"], manifest)


def rng_fingerprint() -> str:
    """Digest of Python, NumPy and Torch CPU global RNG state (for tests)."""
    import numpy

    return _sha({
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(repr(numpy.random.get_state()).encode()).hexdigest(),
        "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    })
