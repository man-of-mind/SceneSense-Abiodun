"""Materialized checkpoint sidecar for future (Run-5) Hybrid-SAC training.

The Run-4 modeled checkpoints (``modeled_smoke_orchestrator``) are
*event-sourced reconstruction checkpoints*: they retain the transition ledger,
the collector checkpoint and a ``BoundaryFingerprintV1`` of actor, critic,
optimizer and decision-RNG **hashes**. They contain no directly loadable
tensor state; restoring one replays training from genesis.

This module adds a sidecar that a Run-5 checkpoint callback writes next to each
event-sourced checkpoint. One sidecar directory holds, create-only and
atomically:

* ``actor.pt``            actor ``state_dict``
* ``online_critics.pt``   ``critic_1.*`` / ``critic_2.*`` of the twin critics
* ``target_critics.pt``   ``target_1.*`` / ``target_2.*`` of the twin critics
* ``actor_optimizer.pt``  actor Adam ``state_dict``
* ``critic_optimizer.pt`` critic Adam ``state_dict``
* ``generators.pt``       CPU ``torch.Generator`` states (decision q/mode,
                          replay sampling, trainer target/actor sampling)
* ``SIDECAR_MANIFEST.json`` canonical manifest: seed plan, update/decision
  counts, feature/action/model schema identity, event-checkpoint digest and
  boundary, per-artifact SHA-256/size/tensor inventory, actor fixture outputs.

Every artifact holds only tensors, containers and scalars and is read with
``torch.load(weights_only=True, map_location="cpu")``. Writing refuses unless
the tensors hash to the event checkpoint's boundary, so a sidecar can only be
produced at the boundary it names.

Scope: the replay buffer contents are *not* materialized. They remain
reconstructible from the event-sourced ledger; exact continuation of training
is sidecar tensors + generators + that replay reconstruction. The direct
actor cold-load needs nothing but the sidecar.

Importing this module performs no I/O and starts no runtime.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2
from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io
from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as run4_models
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tensor_sha256,
    _tree_document,
    _tree_sha256,
)
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
    ConditionalHybridActor,
    TwinHybridCritics,
    build_actor,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ACTION_IDENTITY_SCHEMA_SHA256,
)

SCHEMA_ID = "splitfusion.hybrid_sac.materialized_checkpoint_sidecar.v1"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "SIDECAR_MANIFEST.json"
ARTIFACT_NAMES = (
    "actor",
    "online_critics",
    "target_critics",
    "actor_optimizer",
    "critic_optimizer",
    "generators",
)
ARTIFACT_FILENAMES: Mapping[str, str] = MappingProxyType(
    {name: f"{name}.pt" for name in ARTIFACT_NAMES}
)
GENERATOR_NAMES = (
    "decision_q",
    "decision_mode",
    "replay",
    "trainer_target",
    "trainer_actor",
)
ONLINE_CRITIC_PREFIXES = ("critic_1.", "critic_2.")
TARGET_CRITIC_PREFIXES = ("target_1.", "target_2.")
LOADER = "torch.load(weights_only=True, map_location='cpu')"
_SIDECAR_NAME = re.compile(r"^update_(\d{6})\.sidecar$")
_CHECKPOINT_NAME = re.compile(r"^update_(\d+)\.checkpoint\.json$")


class SidecarError(checkpoint_io.CheckpointIoError):
    """Base class for sidecar failures."""


class SidecarWriteError(SidecarError, checkpoint_io.CheckpointWriteError):
    """A sidecar could not be published, or the state is not at its boundary."""


class SidecarReadError(SidecarError, checkpoint_io.CheckpointReadError):
    """A sidecar is malformed or cannot be applied to the target."""


class SidecarTamperError(SidecarReadError, checkpoint_io.CheckpointTamperError):
    """Bytes, hashes or tensor identities differ from the manifest/anchor."""


class SidecarIdentityError(SidecarReadError):
    """Foreign seed, update, schema or event checkpoint."""


class SidecarIncompleteError(SidecarReadError):
    """A registered checkpoint lacks a directly loadable materialized artifact."""


def _require(condition: bool, message: str, error: type = SidecarReadError) -> None:
    if not condition:
        raise error(message)


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def schema_identity() -> Dict[str, Any]:
    """Feature/action/model identity the sidecar tensors are bound to."""
    document = frozen_actor_v2.binding_document()
    document["action_identity_schema_sha256"] = ACTION_IDENTITY_SCHEMA_SHA256
    document["event_checkpoint_schema_id"] = orch.CHECKPOINT_SCHEMA_ID
    return document


@dataclass(frozen=True, slots=True)
class TrainingStateHandleV1:
    """Live objects whose state a sidecar captures or restores."""

    actor: ConditionalHybridActor
    critics: TwinHybridCritics
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    generators: Mapping[str, torch.Generator]

    def __post_init__(self) -> None:
        try:
            run4_models.validate_run4_models(self.actor, self.critics)
        except run4_models.Run4ModelError as exc:
            raise SidecarError(f"models are not Run-4 shaped: {exc}") from exc
        for name, value in (
            ("actor_optimizer", self.actor_optimizer),
            ("critic_optimizer", self.critic_optimizer),
        ):
            _require(isinstance(value, torch.optim.Optimizer),
                     f"{name} is not an optimizer", SidecarError)
        _require(self.actor_optimizer is not self.critic_optimizer,
                 "optimizers must be distinct", SidecarError)
        _require(tuple(sorted(self.generators)) == tuple(sorted(GENERATOR_NAMES)),
                 "generator set differs from the registered streams", SidecarError)
        seen = []
        for name in GENERATOR_NAMES:
            generator = self.generators[name]
            _require(isinstance(generator, torch.Generator)
                     and generator is not torch.default_generator
                     and generator.device.type == "cpu",
                     f"{name} must be a private CPU generator", SidecarError)
            _require(all(generator is not item for item in seen),
                     "generators must be distinct objects", SidecarError)
            seen.append(generator)


def training_state_from_orchestrator(
    orchestrator: orch.ModeledSmokeOrchestratorV1,
) -> TrainingStateHandleV1:
    """Read-only view of a modeled orchestrator's training state.

    The replay and trainer generators are private attributes of the frozen
    Run-4 runtime; they are read and, on restore, updated in place through
    ``get_state``/``set_state``. No Run-4 module is edited.
    """
    _require(type(orchestrator) is orch.ModeledSmokeOrchestratorV1,
             "orchestrator has a foreign type", SidecarError)
    runner = orchestrator.runner
    trainer = runner.trainer
    bundle = runner.model_bundle
    _require(trainer.actor is bundle.actor and trainer.critics is bundle.critics,
             "trainer is not wired to the runner's model bundle", SidecarError)
    return TrainingStateHandleV1(
        actor=bundle.actor,
        critics=bundle.critics,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        generators=MappingProxyType({
            "decision_q": orchestrator._decision_q_generator,
            "decision_mode": orchestrator._decision_mode_generator,
            "replay": runner._replay_generator,
            "trainer_target": trainer._target_generator,
            "trainer_actor": trainer._actor_generator,
        }),
    )


@dataclass(frozen=True, slots=True)
class EventBindingV1:
    """The event-sourced checkpoint a sidecar materializes."""

    seed: int
    seed_plan: Mapping[str, Any]
    seed_plan_sha256: str
    update_count: int
    decision_count: int
    event_checkpoint_sha256: str
    boundary: Mapping[str, Any]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: orch.ModeledSmokeCheckpointV1,
        seed_plan: orch.RunnerSeedPlanV1,
    ) -> "EventBindingV1":
        _require(type(checkpoint) is orch.ModeledSmokeCheckpointV1,
                 "event checkpoint has a foreign type", SidecarError)
        _require(type(seed_plan) is orch.RunnerSeedPlanV1,
                 "seed plan has a foreign type", SidecarError)
        _require(seed_plan.canonical_sha256 == checkpoint.seed_plan_sha256,
                 "seed plan differs from the event checkpoint", SidecarIdentityError)
        return cls(
            seed=seed_plan.master_seed,
            seed_plan=MappingProxyType(seed_plan.to_dict()),
            seed_plan_sha256=seed_plan.canonical_sha256,
            update_count=checkpoint.update_count,
            decision_count=checkpoint.decision_count,
            event_checkpoint_sha256=checkpoint.canonical_sha256,
            boundary=MappingProxyType(checkpoint.boundary.to_dict()),
        )

    def document(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "seed_plan": dict(self.seed_plan),
            "seed_plan_sha256": self.seed_plan_sha256,
            "update_count": self.update_count,
            "decision_count": self.decision_count,
            "event_checkpoint_sha256": self.event_checkpoint_sha256,
            "boundary": dict(self.boundary),
        }


# ---------------------------------------------------------------------------
# Material capture
# ---------------------------------------------------------------------------


def _cpu_tree(value: Any) -> Any:
    """Detached CPU copy containing only tensors, containers and scalars."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True).contiguous()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(value[key]) for key in value}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise checkpoint_io.UnsupportedCheckpointValue(
        f"unsupported sidecar value {type(value).__qualname__}"
    )


def _split_critics(state: Mapping[str, torch.Tensor]) -> Tuple[dict, dict]:
    online, target = {}, {}
    for name, value in state.items():
        if name.startswith(ONLINE_CRITIC_PREFIXES):
            online[name] = value
        elif name.startswith(TARGET_CRITIC_PREFIXES):
            target[name] = value
        else:
            raise SidecarError(f"critic tensor {name!r} has no registered prefix")
    _require(online and target, "critics lack an online or target twin", SidecarError)
    return online, target


def capture_material(handle: TrainingStateHandleV1) -> Dict[str, Any]:
    """Snapshot every artifact as detached CPU trees (no I/O)."""
    _require(type(handle) is TrainingStateHandleV1,
             "handle has a foreign type", SidecarError)
    online, target = _split_critics(handle.critics.state_dict())
    return {
        "actor": _cpu_tree(handle.actor.state_dict()),
        "online_critics": _cpu_tree(online),
        "target_critics": _cpu_tree(target),
        "actor_optimizer": _cpu_tree(handle.actor_optimizer.state_dict()),
        "critic_optimizer": _cpu_tree(handle.critic_optimizer.state_dict()),
        "generators": {
            name: handle.generators[name].get_state().clone()
            for name in GENERATOR_NAMES
        },
    }


def _tensor_leaves(value: Any, path: str = "") -> list:
    if isinstance(value, torch.Tensor):
        return [{
            "path": path,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "tensor_sha256": _tensor_sha256(value),
        }]
    leaves: list = []
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            leaves.extend(_tensor_leaves(value[key], f"{path}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            leaves.extend(_tensor_leaves(item, f"{path}/{index}"))
    return leaves


def _boundary_view(material: Mapping[str, Any]) -> Dict[str, str]:
    """Recompute the ``BoundaryFingerprintV1`` hash fields from material."""
    critics = {**material["online_critics"], **material["target_critics"]}
    return {
        "actor_sha256": _tree_sha256(material["actor"]),
        "critics_sha256": _tree_sha256(critics),
        "actor_optimizer_sha256": _tree_sha256(material["actor_optimizer"]),
        "critic_optimizer_sha256": _tree_sha256(material["critic_optimizer"]),
        "decision_q_rng_sha256": _tensor_sha256(material["generators"]["decision_q"]),
        "decision_mode_rng_sha256": _tensor_sha256(
            material["generators"]["decision_mode"]
        ),
    }


def _check_boundary(
    material: Mapping[str, Any], boundary: Mapping[str, Any], error: type
) -> None:
    for field, digest in _boundary_view(material).items():
        _require(boundary.get(field) == digest,
                 f"{field} differs from the event-checkpoint boundary", error)


def _actor_fixtures(actor_state: Mapping[str, torch.Tensor]) -> list:
    actor = _fresh_actor()
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    return frozen_actor_v2.fixture_outputs(actor)


def _fresh_actor() -> ConditionalHybridActor:
    # build_actor seeds locally and restores the global RNG.
    return build_actor(run4_models.run4_model_config(), seed=0)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SidecarArtifactV1:
    directory: str
    manifest_sha256: str
    event_checkpoint_sha256: str
    update_count: int
    seed: int


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only(path: Path, writer: Callable[[Any], None]) -> None:
    with path.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def write_sidecar(
    directory: str | os.PathLike[str],
    handle: TrainingStateHandleV1,
    binding: EventBindingV1,
) -> SidecarArtifactV1:
    """Publish one sidecar directory atomically; never overwrite."""
    target = Path(directory)
    _require(bool(target.name) and target.parent != target,
             "sidecar target must be a named directory", SidecarWriteError)
    parent = target.parent
    _require(parent.is_dir(), "sidecar parent does not exist", SidecarWriteError)
    _require(not (target.exists() or target.is_symlink()),
             "sidecar target already exists", SidecarWriteError)
    _require(type(binding) is EventBindingV1,
             "binding has a foreign type", SidecarWriteError)
    material = capture_material(handle)
    _check_boundary(material, binding.boundary, SidecarWriteError)
    fixtures = _actor_fixtures(material["actor"])

    staging = parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        staging.mkdir(mode=0o700)
        artifacts: Dict[str, Any] = {}
        for name in ARTIFACT_NAMES:
            path = staging / ARTIFACT_FILENAMES[name]
            _write_create_only(path, lambda stream, n=name: torch.save(material[n], stream))
            digest, size = checkpoint_io._sha256_file(path)
            artifacts[name] = {
                "filename": ARTIFACT_FILENAMES[name],
                "sha256": digest,
                "size_bytes": size,
                "tree_sha256": _tree_sha256(material[name]),
                "tensors": _tensor_leaves(material[name]),
            }
        manifest = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "loader": LOADER,
            "identity": binding.document(),
            "schema_identity": schema_identity(),
            "artifacts": artifacts,
            "generator_names": list(GENERATOR_NAMES),
            "actor_fixtures": fixtures,
            "scope": (
                "Materialized tensors, optimizer states and generator states at "
                "the named event-sourced boundary. Replay-buffer contents are "
                "not materialized; they are reconstructed from the event ledger."
            ),
        }
        manifest_bytes = checkpoint_io._canonical_json_bytes(manifest)
        _write_create_only(staging / MANIFEST_FILENAME,
                           lambda stream: stream.write(manifest_bytes))
        _fsync_directory(staging)
        _require(not (target.exists() or target.is_symlink()),
                 "sidecar target appeared during publication", SidecarWriteError)
        os.rename(staging, target)
        _fsync_directory(parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return SidecarArtifactV1(
        directory=str(target),
        manifest_sha256=checkpoint_io._sha256_bytes(manifest_bytes),
        event_checkpoint_sha256=binding.event_checkpoint_sha256,
        update_count=binding.update_count,
        seed=binding.seed,
    )


def sidecar_directory_name(update: int) -> str:
    return f"update_{update:06d}.sidecar"


class SidecarCheckpointCallbackV1:
    """Run-5 checkpoint callback: materialize every emitted boundary.

    Pass as ``checkpoint_callback`` to ``run_to_registered_update``. An
    optional ``chain`` callback (e.g. the existing event-checkpoint writer)
    runs first, so the event-sourced checkpoint is always written.
    """

    def __init__(
        self,
        orchestrator: orch.ModeledSmokeOrchestratorV1,
        root: str | os.PathLike[str],
        chain: Optional[Callable[[orch.ModeledSmokeCheckpointEventV1], None]] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._root = Path(root)
        self._chain = chain
        self.artifacts: Dict[int, SidecarArtifactV1] = {}

    def __call__(self, event: orch.ModeledSmokeCheckpointEventV1) -> None:
        _require(type(event) is orch.ModeledSmokeCheckpointEventV1,
                 "event has a foreign type", SidecarWriteError)
        _require(event.update == self._orchestrator.update_count,
                 "callback is not at the event's boundary", SidecarWriteError)
        if self._chain is not None:
            self._chain(event)
        binding = EventBindingV1.from_checkpoint(
            event.checkpoint, self._orchestrator.runner_factory.seed_plan
        )
        handle = training_state_from_orchestrator(self._orchestrator)
        self.artifacts[event.update] = write_sidecar(
            self._root / sidecar_directory_name(event.update), handle, binding
        )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SidecarMaterialV1:
    directory: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    material: Mapping[str, Any]


def _regular_file(path: Path, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        if name == ARTIFACT_FILENAMES["actor"]:
            raise SidecarIncompleteError(
                "sidecar lacks the directly loadable actor-weight artifact"
            ) from exc
        raise SidecarIncompleteError(f"sidecar lacks {name}") from exc
    _require(stat.S_ISREG(mode), f"{name} must be a regular non-symlink file")


def read_sidecar(
    directory: str | os.PathLike[str],
    *,
    expected_seed: int,
    expected_update_count: int,
    expected_manifest_sha256: Optional[str] = None,
    event_checkpoint: Optional[orch.ModeledSmokeCheckpointV1] = None,
) -> SidecarMaterialV1:
    """Verify and load one sidecar (weights-only); mutate nothing.

    At least one external anchor is required: a pinned manifest digest or the
    event-sourced checkpoint itself, whose digest and boundary must match.
    """
    _require(expected_manifest_sha256 is not None or event_checkpoint is not None,
             "an external anchor (manifest digest or event checkpoint) is required",
             SidecarIdentityError)
    root = Path(directory)
    if not root.exists():
        raise SidecarIncompleteError(
            f"no materialized sidecar at {root}: the checkpoint lacks a directly "
            "loadable actor-weight artifact"
        )
    _require(root.is_dir() and not root.is_symlink(),
             "sidecar must be a real directory")
    expected_files = {MANIFEST_FILENAME, *ARTIFACT_FILENAMES.values()}
    for name in sorted(expected_files):
        _regular_file(root / name, name)
    extra = sorted(item.name for item in root.iterdir())
    _require(set(extra) == expected_files, f"unexpected sidecar files: {extra}",
             SidecarTamperError)

    manifest_bytes = (root / MANIFEST_FILENAME).read_bytes()
    manifest_sha = checkpoint_io._sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 is not None:
        _require(manifest_sha == expected_manifest_sha256,
                 "manifest digest differs from the pinned anchor", SidecarTamperError)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SidecarTamperError("manifest is not JSON") from exc
    _require(isinstance(manifest, dict)
             and checkpoint_io._canonical_json_bytes(manifest) == manifest_bytes,
             "manifest is not canonical JSON", SidecarTamperError)
    _require(manifest.get("schema_id") == SCHEMA_ID
             and manifest.get("schema_version") == SCHEMA_VERSION,
             "foreign sidecar schema", SidecarIdentityError)
    _require(manifest.get("schema_identity") == schema_identity(),
             "feature/action/model schema identity differs", SidecarIdentityError)
    identity = manifest.get("identity")
    _require(isinstance(identity, dict), "manifest lacks identity", SidecarTamperError)
    _require(identity.get("seed") == expected_seed
             and type(identity.get("seed")) is int,
             "foreign seed", SidecarIdentityError)
    _require(identity.get("update_count") == expected_update_count
             and type(identity.get("update_count")) is int,
             "foreign update count", SidecarIdentityError)
    _require(set(manifest.get("artifacts", {})) == set(ARTIFACT_NAMES),
             "manifest artifact set differs", SidecarTamperError)
    _require(manifest.get("generator_names") == list(GENERATOR_NAMES),
             "manifest generator set differs", SidecarTamperError)
    if event_checkpoint is not None:
        _require(type(event_checkpoint) is orch.ModeledSmokeCheckpointV1,
                 "event checkpoint has a foreign type", SidecarIdentityError)
        _require(event_checkpoint.canonical_sha256
                 == identity.get("event_checkpoint_sha256"),
                 "sidecar names a different event checkpoint", SidecarIdentityError)
        _require(event_checkpoint.boundary.to_dict() == identity.get("boundary"),
                 "manifest boundary differs from the event checkpoint",
                 SidecarTamperError)
        _require(event_checkpoint.seed_plan_sha256 == identity.get("seed_plan_sha256")
                 and event_checkpoint.update_count == expected_update_count
                 and event_checkpoint.decision_count == identity.get("decision_count"),
                 "event checkpoint identity differs", SidecarIdentityError)

    material: Dict[str, Any] = {}
    for name in ARTIFACT_NAMES:
        entry = manifest["artifacts"][name]
        _require(entry.get("filename") == ARTIFACT_FILENAMES[name],
                 f"{name} filename differs", SidecarTamperError)
        path = root / ARTIFACT_FILENAMES[name]
        digest, size = checkpoint_io._sha256_file(path)
        _require(digest == entry.get("sha256") and size == entry.get("size_bytes"),
                 f"{name} bytes differ from the manifest", SidecarTamperError)
        try:
            with path.open("rb") as stream:
                value = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SidecarTamperError(f"{name} is not weights-only loadable") from exc
        _require(_tree_sha256(value) == entry.get("tree_sha256"),
                 f"{name} tree digest differs", SidecarTamperError)
        _require(_tensor_leaves(value) == entry.get("tensors"),
                 f"{name} tensor names/dtypes/shapes differ", SidecarTamperError)
        material[name] = value
    _require(isinstance(material["generators"], Mapping)
             and set(material["generators"]) == set(GENERATOR_NAMES),
             "generator artifact set differs", SidecarTamperError)
    _check_boundary(material, identity.get("boundary", {}), SidecarTamperError)
    return SidecarMaterialV1(
        directory=str(root),
        manifest=MappingProxyType(manifest),
        manifest_sha256=manifest_sha,
        material=MappingProxyType(material),
    )


def _same_tensor_names(
    loaded: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], name: str
) -> None:
    _require(sorted(loaded) == sorted(reference), f"{name} tensor names differ")
    for key in reference:
        _require(isinstance(loaded[key], torch.Tensor), f"{name}.{key} is not a tensor")
        _require(loaded[key].dtype == reference[key].dtype,
                 f"{name}.{key} dtype differs")
        _require(tuple(loaded[key].shape) == tuple(reference[key].shape),
                 f"{name}.{key} shape differs")


def load_actor_from_sidecar(
    directory: str | os.PathLike[str],
    *,
    expected_seed: int,
    expected_update_count: int,
    expected_manifest_sha256: Optional[str] = None,
    event_checkpoint: Optional[orch.ModeledSmokeCheckpointV1] = None,
) -> ConditionalHybridActor:
    """Cold-load the actor directly (no training replay); verify fixtures."""
    loaded = read_sidecar(
        directory,
        expected_seed=expected_seed,
        expected_update_count=expected_update_count,
        expected_manifest_sha256=expected_manifest_sha256,
        event_checkpoint=event_checkpoint,
    )
    actor = _fresh_actor()
    state = loaded.material["actor"]
    _same_tensor_names(state, actor.state_dict(), "actor")
    actor.load_state_dict(state, strict=True)
    actor.eval()
    _require(_tree_sha256(actor.state_dict())
             == loaded.manifest["identity"]["boundary"]["actor_sha256"],
             "loaded actor differs from the boundary", SidecarTamperError)
    _require(frozen_actor_v2.fixture_outputs(actor) == loaded.manifest["actor_fixtures"],
             "loaded actor fixture outputs differ", SidecarTamperError)
    return actor


def _validate_optimizer(
    loaded: Mapping[str, Any], optimizer: torch.optim.Optimizer, name: str
) -> None:
    reference = optimizer.state_dict()
    _require(isinstance(loaded, Mapping) and set(loaded) == {"state", "param_groups"},
             f"{name} is not an optimizer state_dict")
    _require(_tree_document(list(loaded["param_groups"]))
             == _tree_document(list(reference["param_groups"])),
             f"{name} parameter groups/hyperparameters differ")
    params = [p for group in optimizer.param_groups for p in group["params"]]
    indices = [i for group in reference["param_groups"] for i in group["params"]]
    by_index = dict(zip(indices, params))
    state = loaded["state"]
    _require(isinstance(state, Mapping) and set(state) <= set(by_index),
             f"{name} state names foreign parameters")
    for index, entry in state.items():
        param = by_index[index]
        _require(isinstance(entry, Mapping), f"{name} state {index} malformed")
        for key, value in entry.items():
            _require(isinstance(value, torch.Tensor),
                     f"{name} state {index}.{key} is not a tensor")
            if key == "step":
                _require(value.dim() == 0 and value.is_floating_point(),
                         f"{name} state {index}.step malformed")
            else:
                _require(value.dtype == param.dtype
                         and tuple(value.shape) == tuple(param.shape),
                         f"{name} state {index}.{key} dtype/shape differs")


def _handle_snapshot(handle: TrainingStateHandleV1) -> Dict[str, Any]:
    return {
        "actor": _cpu_tree(handle.actor.state_dict()),
        "critics": _cpu_tree(handle.critics.state_dict()),
        "actor_optimizer": copy.deepcopy(handle.actor_optimizer.state_dict()),
        "critic_optimizer": copy.deepcopy(handle.critic_optimizer.state_dict()),
        "generators": {n: handle.generators[n].get_state() for n in GENERATOR_NAMES},
    }


def _restore_snapshot(handle: TrainingStateHandleV1, snapshot: Mapping[str, Any]) -> None:
    handle.actor.load_state_dict(snapshot["actor"], strict=True)
    handle.critics.load_state_dict(snapshot["critics"], strict=True)
    handle.actor_optimizer.load_state_dict(snapshot["actor_optimizer"])
    handle.critic_optimizer.load_state_dict(snapshot["critic_optimizer"])
    for name in GENERATOR_NAMES:
        handle.generators[name].set_state(snapshot["generators"][name])


def apply_training_state(
    loaded: SidecarMaterialV1, handle: TrainingStateHandleV1
) -> Dict[str, Any]:
    """Load models, optimizers and generators into ``handle``.

    Everything is validated before the first mutation; if application or the
    post-load boundary check fails, the handle is restored exactly.
    Counters and the replay buffer are not touched.
    """
    _require(type(loaded) is SidecarMaterialV1, "material has a foreign type")
    _require(type(handle) is TrainingStateHandleV1, "handle has a foreign type")
    material = loaded.material
    critics = {**material["online_critics"], **material["target_critics"]}
    _same_tensor_names(material["actor"], handle.actor.state_dict(), "actor")
    _same_tensor_names(critics, handle.critics.state_dict(), "critics")
    _validate_optimizer(material["actor_optimizer"], handle.actor_optimizer,
                        "actor_optimizer")
    _validate_optimizer(material["critic_optimizer"], handle.critic_optimizer,
                        "critic_optimizer")
    for name in GENERATOR_NAMES:
        value = material["generators"][name]
        current = handle.generators[name].get_state()
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                 and tuple(value.shape) == tuple(current.shape),
                 f"generator {name} state dtype/shape differs")

    snapshot = _handle_snapshot(handle)
    try:
        handle.actor.load_state_dict(material["actor"], strict=True)
        handle.critics.load_state_dict(critics, strict=True)
        handle.actor_optimizer.load_state_dict(material["actor_optimizer"])
        handle.critic_optimizer.load_state_dict(material["critic_optimizer"])
        for name in GENERATOR_NAMES:
            handle.generators[name].set_state(material["generators"][name].clone())
        _check_boundary(capture_material(handle),
                        loaded.manifest["identity"]["boundary"], SidecarTamperError)
        for name in GENERATOR_NAMES:
            _require(torch.equal(handle.generators[name].get_state(),
                                 material["generators"][name]),
                     f"generator {name} state did not restore", SidecarTamperError)
    except Exception:
        _restore_snapshot(handle, snapshot)
        raise
    boundary = _boundary_view(capture_material(handle))
    return {
        "manifest_sha256": loaded.manifest_sha256,
        "event_checkpoint_sha256":
            loaded.manifest["identity"]["event_checkpoint_sha256"],
        "boundary": boundary,
        "generator_state_sha256": {
            name: _tensor_sha256(handle.generators[name].get_state())
            for name in GENERATOR_NAMES
        },
    }


# ---------------------------------------------------------------------------
# Registered-checkpoint acceptance
# ---------------------------------------------------------------------------


def require_materialized_checkpoints(
    checkpoint_dir: str | os.PathLike[str],
    sidecar_root: str | os.PathLike[str],
    *,
    expected_seed: int,
) -> Dict[int, str]:
    """Fail unless every registered checkpoint has a cold-loadable actor.

    Sidecar presence is checked for every checkpoint before any checkpoint is
    parsed, so an event-sourced-only run fails without replay or heavy I/O.
    Returns ``{update: manifest_sha256}``.
    """
    checkpoint_root = Path(checkpoint_dir)
    updates: Dict[int, Path] = {}
    for item in sorted(checkpoint_root.iterdir()):
        match = _CHECKPOINT_NAME.match(item.name)
        if match:
            updates[int(match.group(1))] = item
    _require(bool(updates), "no registered checkpoints found", SidecarIncompleteError)
    missing = [
        update for update in sorted(updates)
        if not (Path(sidecar_root) / sidecar_directory_name(update)
                / ARTIFACT_FILENAMES["actor"]).is_file()
    ]
    if missing:
        raise SidecarIncompleteError(
            "registered checkpoints lack a directly loadable actor-weight "
            f"artifact at updates {missing}"
        )
    verified: Dict[int, str] = {}
    for update in sorted(updates):
        checkpoint = orch.read_checkpoint(updates[update])
        directory = Path(sidecar_root) / sidecar_directory_name(update)
        load_actor_from_sidecar(
            directory,
            expected_seed=expected_seed,
            expected_update_count=update,
            event_checkpoint=checkpoint,
        )
        verified[update] = checkpoint_io._sha256_bytes(
            (directory / MANIFEST_FILENAME).read_bytes()
        )
    return verified


__all__ = [
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "ARTIFACT_NAMES",
    "ARTIFACT_FILENAMES",
    "GENERATOR_NAMES",
    "SidecarError",
    "SidecarWriteError",
    "SidecarReadError",
    "SidecarTamperError",
    "SidecarIdentityError",
    "SidecarIncompleteError",
    "TrainingStateHandleV1",
    "EventBindingV1",
    "SidecarArtifactV1",
    "SidecarMaterialV1",
    "SidecarCheckpointCallbackV1",
    "schema_identity",
    "training_state_from_orchestrator",
    "capture_material",
    "write_sidecar",
    "sidecar_directory_name",
    "read_sidecar",
    "load_actor_from_sidecar",
    "apply_training_state",
    "require_materialized_checkpoints",
]
