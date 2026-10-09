"""Fail-closed CPU loader for selected Run-4B/Run-5B actor exports.

The tracked manifest binds identities only.  The ignored ``.pt`` artifact is
supplied explicitly and is never copied by this package.  Importing this
module performs no I/O and does not change any RNG state.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tree_sha256,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor,
    HybridSacModelConfig,
    build_actor,
)
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
    MODELED_SMOKE_SUPPORT,
    MODELED_SMOKE_SUPPORT_SHA256,
)


MANIFEST_SCHEMA = "scenesense.splitfusion.run4b5b.actor_artifact.v1"
DECISION_RULE = "BATCH1_CPU_DETERMINISTIC_ARGMAX__DECIMAL_HALF_UP_Q_E4"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FrozenActorLoadError(RuntimeError):
    """The artifact, manifest, model, or requested variant differs."""


class ActorVariant(str, enum.Enum):
    RUN4B = "RUN4B_MCS_BACKLOG_NO_QPERC"
    RUN5B = "RUN5B_MCS_BACKLOG_SNR_NO_QPERC"


RUN4B_FEATURE_ORDER = (
    "camera_si_scaled",
    "radar_p40",
    "prior_ul_mcs_normalized",
    "pre_action_rlc_backlog_log1p_scaled",
    *(f"prev_joint_mode_{index}_one_hot" for index in range(12)),
    "prev_q_normalized",
    "prev_operational_latency_normalized",
    "prev_present",
    "prev_operational_success",
)
RUN5B_FEATURE_ORDER = (
    *RUN4B_FEATURE_ORDER,
    "effective_external_ul_snr_proxy_scaled",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenActorLoadError(message)


def _digest(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SHA256.fullmatch(value)),
             f"{field} is not a lowercase SHA-256")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def tensor_inventory(state: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], ...]:
    _require(isinstance(state, Mapping), "weights payload is not a mapping")
    rows: list[dict[str, Any]] = []
    for name in sorted(state):
        value = state[name]
        _require(type(name) is str and type(value) is torch.Tensor,
                 "weights payload is not an exact tensor state_dict")
        rows.append({
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "tensor_sha256": _tensor_sha256(value),
        })
    return tuple(rows)


def expected_feature_order(variant: ActorVariant) -> tuple[str, ...]:
    _require(type(variant) is ActorVariant, "variant must be exact ActorVariant")
    return RUN4B_FEATURE_ORDER if variant is ActorVariant.RUN4B else RUN5B_FEATURE_ORDER


@dataclass(frozen=True, slots=True)
class FrozenActorArtifactManifestV1:
    variant: ActorVariant
    feature_schema_id: str
    feature_schema_sha256: str
    feature_order: tuple[str, ...]
    model_binding_sha256: str
    operational_latency_provider_sha256: str
    actor_boundary_sha256: str
    actor_tree_sha256: str
    weights_file_name: str
    weights_file_sha256: str
    fixture_outputs_sha256: str
    runner_binding_sha256: str
    q_support_sha256: str
    selected_seed: int
    selected_update: int
    source_export_schema: str
    source_export_sha256: str
    source_code_commit: str
    tensor_inventory: tuple[dict[str, Any], ...]
    decision_rule: str = DECISION_RULE

    def __post_init__(self) -> None:
        expected = expected_feature_order(self.variant)
        _require(self.feature_order == expected, "feature order differs from variant")
        _require(self.feature_schema_id in {
            "splitfusion_run4b_policy_features_v1",
            "splitfusion_run5b_policy_features_v1",
        }, "foreign feature schema id")
        required_run = "run4b" if self.variant is ActorVariant.RUN4B else "run5b"
        _require(required_run in self.feature_schema_id,
                 "feature schema id differs from variant")
        for field in (
            "feature_schema_sha256", "model_binding_sha256",
            "operational_latency_provider_sha256", "actor_boundary_sha256",
            "actor_tree_sha256", "weights_file_sha256",
            "fixture_outputs_sha256", "runner_binding_sha256",
            "q_support_sha256", "source_export_sha256",
        ):
            _digest(getattr(self, field), field)
        _require(self.actor_boundary_sha256 == self.actor_tree_sha256,
                 "boundary and canonical tensor tree differ")
        _require(self.q_support_sha256 == MODELED_SMOKE_SUPPORT_SHA256,
                 "q support differs from registered support")
        _require(self.weights_file_name == "actor_state_dict.pt",
                 "weights filename is not the registered export name")
        _require(type(self.selected_seed) is int and self.selected_seed >= 0,
                 "selected seed is invalid")
        _require(type(self.selected_update) is int and self.selected_update > 0,
                 "selected update is invalid")
        _require(type(self.source_code_commit) is str
                 and bool(re.fullmatch(r"[0-9a-f]{40}", self.source_code_commit)),
                 "source code commit is invalid")
        _require(self.decision_rule == DECISION_RULE, "decision rule drifted")
        _require(type(self.tensor_inventory) is tuple and self.tensor_inventory,
                 "tensor inventory is empty")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenActorArtifactManifestV1":
        _require(isinstance(raw, Mapping), "manifest root is not a mapping")
        expected = {field.name for field in cls.__dataclass_fields__.values()} | {"schema"}
        _require(set(raw) == expected, "manifest fields are incomplete or foreign")
        _require(raw["schema"] == MANIFEST_SCHEMA, "manifest schema differs")
        try:
            variant = ActorVariant(raw["variant"])
        except (TypeError, ValueError) as exc:
            raise FrozenActorLoadError("manifest variant is unknown") from exc
        values = dict(raw)
        values.pop("schema")
        values["variant"] = variant
        values["feature_order"] = tuple(values["feature_order"])
        values["tensor_inventory"] = tuple(values["tensor_inventory"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            **{
                name: (value.value if isinstance(value, enum.Enum)
                       else list(value) if isinstance(value, tuple)
                       else value)
                for name, value in (
                    (field, getattr(self, field))
                    for field in self.__dataclass_fields__
                )
            },
        }


def load_manifest(path: Path) -> FrozenActorArtifactManifestV1:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenActorLoadError("actor manifest is unreadable") from exc
    return FrozenActorArtifactManifestV1.from_mapping(raw)


def _model_config(width: int) -> HybridSacModelConfig:
    return HybridSacModelConfig(
        state_dim=width, mode_count=12, dtype=torch.float32,
        modeled_smoke_support=MODELED_SMOKE_SUPPORT,
    )


@dataclass(frozen=True, slots=True)
class FrozenActorDecisionV1:
    mode_id: int
    q_e4: int
    actor_boundary_sha256: str


class LoadedFrozenActorV1:
    """Frozen exact actor with the registered batch-1 deployment rule."""

    def __init__(self, actor: ConditionalHybridActor,
                 manifest: FrozenActorArtifactManifestV1) -> None:
        _require(type(actor) is ConditionalHybridActor,
                 "loaded actor has a foreign class")
        _require(actor.config == _model_config(len(manifest.feature_order)),
                 "loaded actor configuration differs")
        _require(actor.modeled_smoke_support_sha256 == MODELED_SMOKE_SUPPORT_SHA256,
                 "loaded actor q support differs")
        actor.eval()
        actor.requires_grad_(False)
        for name, value in list(actor.named_parameters()) + list(actor.named_buffers()):
            _require(value.device.type == "cpu", f"{name} is not on CPU")
            if value.is_floating_point():
                _require(value.dtype is torch.float32, f"{name} is not float32")
                _require(bool(torch.isfinite(value).all()), f"{name} is not finite")
        _require(_tree_sha256(actor.state_dict()) == manifest.actor_boundary_sha256,
                 "loaded actor boundary differs")
        lower, upper = actor.active_q_e4_bounds()
        self._actor = actor
        self._manifest = manifest
        self._lower = tuple(int(v) for v in lower)
        self._upper = tuple(int(v) for v in upper)

    @property
    def manifest(self) -> FrozenActorArtifactManifestV1:
        return self._manifest

    @property
    def module(self) -> ConditionalHybridActor:
        return self._actor

    def act_on_vector(self, values: Sequence[float]) -> FrozenActorDecisionV1:
        _require(type(values) is tuple, "actor features must be an exact tuple")
        _require(len(values) == len(self._manifest.feature_order),
                 "actor feature width differs")
        exact: list[float] = []
        for index, value in enumerate(values):
            _require(not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(float(value)),
                     f"feature {index} is not a finite real")
            exact.append(float(value))
        state = torch.tensor((tuple(exact),), dtype=torch.float32)
        with torch.inference_mode():
            execution = self._actor.deterministic_execution(state)
        mode_id = int(execution.mode_index[0])
        q_e4 = int(execution.q_e4[0])
        _require(0 <= mode_id < 12, "actor returned invalid mode")
        _require(self._lower[mode_id] <= q_e4 <= self._upper[mode_id],
                 "actor q escaped registered support")
        return FrozenActorDecisionV1(
            mode_id=mode_id, q_e4=q_e4,
            actor_boundary_sha256=self._manifest.actor_boundary_sha256,
        )


def load_frozen_actor(weights_path: Path,
                      manifest: FrozenActorArtifactManifestV1, *,
                      expected_variant: ActorVariant) -> LoadedFrozenActorV1:
    _require(type(manifest) is FrozenActorArtifactManifestV1,
             "manifest has foreign type")
    _require(type(expected_variant) is ActorVariant,
             "expected variant has foreign type")
    _require(manifest.variant is expected_variant, "actor variant differs")
    path = Path(weights_path)
    _require(path.is_file() and not path.is_symlink(),
             "weights must be a regular non-symlink file")
    _require(path.name == manifest.weights_file_name, "weights filename differs")
    _require(_sha_file(path) == manifest.weights_file_sha256,
             "weights file hash differs")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FrozenActorLoadError("weights-only CPU load failed") from exc
    _require(tensor_inventory(state) == manifest.tensor_inventory,
             "weights tensor inventory differs")
    _require(_tree_sha256(state) == manifest.actor_tree_sha256,
             "weights tensor tree differs")
    actor = build_actor(_model_config(len(manifest.feature_order)), seed=0)
    try:
        actor.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise FrozenActorLoadError("weights do not fit exact actor architecture") from exc
    return LoadedFrozenActorV1(actor, manifest)
