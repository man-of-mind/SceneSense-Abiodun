"""Crash-safe, code-execution-free persistence for Run-4 checkpoints.

The in-memory :class:`PersistentRunnerCheckpointV1` deliberately contains
module-private attestation tokens.  Those tokens prove that state, reward and
transition records passed the registered constructors in *this* interpreter;
they are not durable credentials and must never be pickled as though they
were.  This module therefore persists a restricted, weights-only material
record:

* model/optimizer/RNG tensors remain CPU tensors;
* all other values are primitives, tuples or string-keyed record envelopes;
* reconciled actions are re-issued against the frozen action catalogue;
* transition attestations are omitted and replaced by the exact causal replay
  inputs plus their expected digests.

Publication is an atomic directory rename.  ``manifest.json`` is the commit
record for ``checkpoint.pt`` and binds its byte length, SHA-256, the original
in-memory checkpoint digest, the runner binding and a content fingerprint.
Loading uses ``torch.load(..., weights_only=True)`` exclusively.

Restoration never trusts a persisted transition object. The runner's public
portable-journal seam replays ``(decision, prediction, successor_staged)``
through its normal causal execution path and re-issues every transition
attestation. The resulting in-memory checkpoint must reproduce the source
digest before the existing runner restore path is allowed to continue.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1 import empirical_quality_surface
from rl_agent.splitfusion_hybrid_sac_v1 import transaction_identity
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

from . import fit_scene_provider, held_payload, persistent_runner
from . import production_state_provider, quality_adapter, run4_contract
from . import sequential_kernel


SCHEMA_ID = "splitfusion.run4.durable_checkpoint_material.v1"
SCHEMA_VERSION = 1
PAYLOAD_FILENAME = "checkpoint.pt"
MANIFEST_FILENAME = "manifest.json"


class CheckpointIoError(RuntimeError):
    """Base class for durable-checkpoint refusal."""


class CheckpointWriteError(CheckpointIoError):
    pass


class CheckpointReadError(CheckpointIoError):
    pass


class CheckpointTamperError(CheckpointReadError):
    pass


class UnsupportedCheckpointValue(CheckpointIoError):
    pass


class PortableRestoreApiUnavailable(CheckpointIoError):
    pass


def _digest(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise CheckpointIoError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise CheckpointIoError(f"{name} must be a SHA-256 hex digest") from exc
    if value != value.lower():
        raise CheckpointIoError(f"{name} must use lowercase hexadecimal")
    return value


def _exact_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CheckpointIoError(f"{name} must be an exact int >= {minimum}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class PortableJournalEntryV1:
    """Attestation-free material sufficient for exact causal replay."""

    decision: sequential_kernel.KernelDecisionInputV1
    prediction: sequential_kernel.EmpiricalStepPredictionV1
    successor_staged: production_state_provider.StagedDecisionInputsV1
    transition_sha256: str
    environment_transition_sha256: str
    journal_sha256: str

    def __post_init__(self) -> None:
        if type(self.decision) is not sequential_kernel.KernelDecisionInputV1:
            raise CheckpointIoError("portable journal contains foreign decision")
        if type(self.prediction) is not sequential_kernel.EmpiricalStepPredictionV1:
            raise CheckpointIoError("portable journal contains foreign prediction")
        if type(self.successor_staged) is not (
            production_state_provider.StagedDecisionInputsV1
        ):
            raise CheckpointIoError("portable journal contains foreign successor")
        for name in (
            "transition_sha256",
            "environment_transition_sha256",
            "journal_sha256",
        ):
            _digest(getattr(self, name), name)
        if self.prediction.prediction_request_sha256 != (
            self.decision.prediction_request_sha256
        ):
            raise CheckpointIoError("portable decision/prediction join differs")
        expected_identity = run4_contract.DecisionIdentityV1(
            self.decision.identity.session_uuid,
            self.decision.identity.ue_id,
            self.decision.identity.decision_seq + 1,
        )
        if self.successor_staged.identity != expected_identity:
            raise CheckpointIoError("portable successor identity is not contiguous")
        if self.successor_staged.radio_state.canonical_sha256 != (
            self.prediction.next_state.canonical_sha256
        ):
            raise CheckpointIoError("portable successor radio state was substituted")
        if self.transition_sha256 != self.environment_transition_sha256:
            raise CheckpointIoError("environment/transition digest mismatch")
        observed_journal = canonical_sha256(
            {
                "decision": self.decision.canonical_sha256,
                "prediction": self.prediction.canonical_sha256,
                "staged": self.successor_staged.canonical_sha256,
                "transition": self.transition_sha256,
            }
        )
        if observed_journal != self.journal_sha256:
            raise CheckpointIoError("portable journal fingerprint mismatch")


@dataclass(frozen=True, slots=True)
class DurableCheckpointMaterialV1:
    """Safe, portable representation of one in-memory runner checkpoint."""

    runner_schema_id: str
    runner_schema_version: int
    runner_binding_sha256: str
    session_uuid: str
    ue_id: str
    genesis_staged: production_state_provider.StagedDecisionInputsV1
    journal: Tuple[PortableJournalEntryV1, ...]
    fit_scene_state: fit_scene_provider.FitSceneProviderStateV1
    initial_kernel_checkpoint: sequential_kernel.KernelCheckpointV1
    kernel_checkpoint: sequential_kernel.KernelCheckpointV1
    state_stager_state: Any
    prediction_provider_state: Any
    actor_state_dict: Mapping[str, Any]
    critics_state_dict: Mapping[str, Any]
    actor_optimizer_state_dict: Mapping[str, Any]
    critic_optimizer_state_dict: Mapping[str, Any]
    trainer_update_count: int
    decision_q_rng_state: torch.Tensor
    decision_mode_rng_state: torch.Tensor
    replay_rng_state: torch.Tensor
    trainer_target_rng_state: torch.Tensor
    trainer_actor_rng_state: torch.Tensor
    current_state_sha256: str
    exploration_decision_count: int
    replay_accepted_count: int
    replay_evicted_count: int
    replay_seen_digest_count: int
    replay_seen_identity_count: int
    replay_resident_transition_digests: Tuple[str, ...]
    source_checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.runner_schema_id != persistent_runner.SCHEMA_ID:
            raise CheckpointIoError("foreign persistent-runner schema id")
        if self.runner_schema_version != persistent_runner.SCHEMA_VERSION:
            raise CheckpointIoError("foreign persistent-runner schema version")
        _digest(self.runner_binding_sha256, "runner_binding_sha256")
        _digest(self.source_checkpoint_sha256, "source_checkpoint_sha256")
        _digest(self.current_state_sha256, "current_state_sha256")
        if type(self.genesis_staged) is not (
            production_state_provider.StagedDecisionInputsV1
        ):
            raise CheckpointIoError("foreign genesis staging")
        if type(self.journal) is not tuple or any(
            type(row) is not PortableJournalEntryV1 for row in self.journal
        ):
            raise CheckpointIoError("portable journal must be an exact tuple")
        if type(self.fit_scene_state) is not fit_scene_provider.FitSceneProviderStateV1:
            raise CheckpointIoError("foreign fit-scene state")
        for name in ("initial_kernel_checkpoint", "kernel_checkpoint"):
            if type(getattr(self, name)) is not sequential_kernel.KernelCheckpointV1:
                raise CheckpointIoError(f"foreign {name}")
        for name in (
            "trainer_update_count",
            "exploration_decision_count",
            "replay_accepted_count",
            "replay_evicted_count",
            "replay_seen_digest_count",
            "replay_seen_identity_count",
        ):
            _exact_int(getattr(self, name), name)
        for name in (
            "decision_q_rng_state",
            "decision_mode_rng_state",
            "replay_rng_state",
            "trainer_target_rng_state",
            "trainer_actor_rng_state",
        ):
            value = getattr(self, name)
            if type(value) is not torch.Tensor or value.device.type != "cpu":
                raise CheckpointIoError(f"{name} must be an exact CPU tensor")
        if type(self.replay_resident_transition_digests) is not tuple:
            raise CheckpointIoError("resident transition digests must be a tuple")
        for digest in self.replay_resident_transition_digests:
            _digest(digest, "resident transition digest")
        if self.initial_kernel_checkpoint.completed_steps != 0:
            raise CheckpointIoError("initial kernel checkpoint is not genesis")
        if self.kernel_checkpoint.completed_steps != len(self.journal):
            raise CheckpointIoError("kernel step count differs from journal length")
        if self.exploration_decision_count > len(self.journal):
            raise CheckpointIoError("exploration count exceeds journal length")
        if self.replay_accepted_count != len(self.journal):
            raise CheckpointIoError("accepted replay count differs from journal length")
        if self.replay_seen_digest_count != len(self.journal):
            raise CheckpointIoError("seen replay digest count differs from journal length")
        if self.replay_seen_identity_count != len(self.journal):
            raise CheckpointIoError("seen replay identity count differs from journal length")
        if self.replay_evicted_count + len(
            self.replay_resident_transition_digests
        ) != self.replay_accepted_count:
            raise CheckpointIoError("resident/evicted replay accounting differs")
        transition_digests = tuple(row.transition_sha256 for row in self.journal)
        resident_count = len(self.replay_resident_transition_digests)
        if self.replay_resident_transition_digests != (
            transition_digests[-resident_count:] if resident_count else ()
        ):
            raise CheckpointIoError("resident replay order differs from journal tail")
        for index, row in enumerate(self.journal):
            identity = row.decision.identity
            if (
                identity.session_uuid != self.session_uuid
                or identity.ue_id != self.ue_id
                or identity.decision_seq != index
            ):
                raise CheckpointIoError("portable journal sequence is not contiguous")
        if (
            self.genesis_staged.identity.session_uuid != self.session_uuid
            or self.genesis_staged.identity.ue_id != self.ue_id
            or self.genesis_staged.identity.decision_seq != 0
        ):
            raise CheckpointIoError("genesis staging differs from session identity")
        if self.journal and self.kernel_checkpoint.current_state.canonical_sha256 != (
            self.journal[-1].prediction.next_state.canonical_sha256
        ):
            raise CheckpointIoError("final kernel state differs from journal")

    @classmethod
    def from_checkpoint(
        cls, checkpoint: persistent_runner.PersistentRunnerCheckpointV1
    ) -> "DurableCheckpointMaterialV1":
        if type(checkpoint) is not persistent_runner.PersistentRunnerCheckpointV1:
            raise CheckpointIoError("checkpoint must be exact PersistentRunnerCheckpointV1")
        if checkpoint.compute_sha256() != checkpoint.checkpoint_sha256:
            raise CheckpointIoError("checkpoint changed after construction")
        journal = tuple(
            PortableJournalEntryV1(
                decision=row.decision,
                prediction=row.prediction,
                successor_staged=row.successor_staged,
                transition_sha256=row.transition.canonical_sha256(),
                environment_transition_sha256=row.environment_transition_sha256,
                journal_sha256=row.canonical_sha256,
            )
            for row in checkpoint.journal
        )
        return cls(
            runner_schema_id=persistent_runner.SCHEMA_ID,
            runner_schema_version=persistent_runner.SCHEMA_VERSION,
            runner_binding_sha256=checkpoint.runner_binding_sha256,
            session_uuid=checkpoint.session_uuid,
            ue_id=checkpoint.ue_id,
            genesis_staged=checkpoint.genesis_staged,
            journal=journal,
            fit_scene_state=checkpoint.fit_scene_state,
            initial_kernel_checkpoint=checkpoint.initial_kernel_checkpoint,
            kernel_checkpoint=checkpoint.kernel_checkpoint,
            state_stager_state=checkpoint.state_stager_state,
            prediction_provider_state=checkpoint.prediction_provider_state,
            actor_state_dict=checkpoint.actor_state_dict,
            critics_state_dict=checkpoint.critics_state_dict,
            actor_optimizer_state_dict=checkpoint.actor_optimizer_state_dict,
            critic_optimizer_state_dict=checkpoint.critic_optimizer_state_dict,
            trainer_update_count=checkpoint.trainer_update_count,
            decision_q_rng_state=checkpoint.decision_q_rng_state,
            decision_mode_rng_state=checkpoint.decision_mode_rng_state,
            replay_rng_state=checkpoint.replay_rng_state,
            trainer_target_rng_state=checkpoint.trainer_target_rng_state,
            trainer_actor_rng_state=checkpoint.trainer_actor_rng_state,
            current_state_sha256=checkpoint.current_state_sha256,
            exploration_decision_count=checkpoint.exploration_decision_count,
            replay_accepted_count=checkpoint.replay_accepted_count,
            replay_evicted_count=checkpoint.replay_evicted_count,
            replay_seen_digest_count=checkpoint.replay_seen_digest_count,
            replay_seen_identity_count=checkpoint.replay_seen_identity_count,
            replay_resident_transition_digests=(
                checkpoint.replay_resident_transition_digests
            ),
            source_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )


@dataclass(frozen=True, slots=True)
class CheckpointArtifactV1:
    directory: str
    manifest_sha256: str
    payload_sha256: str
    payload_size_bytes: int
    material_sha256: str
    source_checkpoint_sha256: str


# Every reconstructible class is explicitly admitted.  No payload-controlled
# import, getattr or Python global can run during weights-only loading.
_DATACLASS_TYPES = (
    run4_contract.DecisionIdentityV1,
    run4_contract.DecisionBoundaryV1,
    run4_contract.HoldTensorV1,
    run4_contract.ActionHoldV1,
    sequential_kernel.IntegerObservationV1,
    sequential_kernel.RadioQueueStateV1,
    sequential_kernel.KernelDecisionInputV1,
    sequential_kernel.FeedbackLatencyBreakdownV1,
    sequential_kernel.EmpiricalStepPredictionV1,
    sequential_kernel.KernelCheckpointV1,
    production_state_provider.ObservationTimingV1,
    production_state_provider.PriorGrantProvenanceV1,
    production_state_provider.BacklogProvenanceV1,
    production_state_provider.StagedDecisionInputsV1,
    fit_scene_provider.FitSceneDrawV1,
    fit_scene_provider.FitSceneProviderStateV1,
    quality_adapter.FitSceneSelectionV1,
    quality_adapter.RewardTensorEvidenceV1,
    quality_adapter.RewardTensorResultV1,
    quality_adapter.HeldTensorResultV1,
    held_payload.HeldSelectionCounterV1,
    held_payload.EndpointEvidenceV1,
    held_payload.HeldPayloadEstimateV1,
    empirical_quality_surface.EndpointEvidence,
    empirical_quality_surface.PolicySceneView,
    PortableJournalEntryV1,
    DurableCheckpointMaterialV1,
)
_DATACLASS_BY_TAG = {
    f"{item.__module__}.{item.__qualname__}": item for item in _DATACLASS_TYPES
}

_ENUM_TYPES = (
    run4_contract.PayloadEvidenceClass,
    sequential_kernel.KernelCalibrationPartition,
    sequential_kernel.KernelTerminalKind,
    sequential_kernel.KernelAuthorizationClass,
)
_ENUM_BY_TAG = {
    f"{item.__module__}.{item.__qualname__}": item for item in _ENUM_TYPES
}

_ACTION_TAG = (
    f"{transaction_identity.ExecutedActionIdentity.__module__}."
    f"{transaction_identity.ExecutedActionIdentity.__qualname__}"
)


def _encode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise UnsupportedCheckpointValue("non-finite float is not portable")
        return {"__float_hex__": value.hex()}
    if type(value) is torch.Tensor:
        if value.device.type != "cpu" or value.layout is not torch.strided:
            raise UnsupportedCheckpointValue("only dense CPU tensors are portable")
        return {"__tensor__": value.detach().clone().contiguous()}
    if isinstance(value, Enum):
        tag = f"{type(value).__module__}.{type(value).__qualname__}"
        if tag not in _ENUM_BY_TAG:
            raise UnsupportedCheckpointValue(f"unsupported enum {tag}")
        return {"__enum__": tag, "value": value.value}
    if type(value) is tuple:
        return {"__tuple__": [_encode(item) for item in value]}
    if type(value) is list:
        return {"__list__": [_encode(item) for item in value]}
    if isinstance(value, Mapping):
        return {
            "__mapping__": [
                [_encode(key), _encode(item)] for key, item in value.items()
            ]
        }
    if type(value) is transaction_identity.ExecutedActionIdentity:
        value.require_reconciled()
        return {"__dataclass__": _ACTION_TAG, "fields": _encode(value.to_canonical_dict())}
    if is_dataclass(value):
        tag = f"{type(value).__module__}.{type(value).__qualname__}"
        if tag not in _DATACLASS_BY_TAG:
            raise UnsupportedCheckpointValue(f"unsupported dataclass {tag}")
        return {
            "__dataclass__": tag,
            "fields": {
                item.name: _encode(getattr(value, item.name))
                for item in fields(value)
                if item.name not in ("_attestation", "_reconciliation")
            },
        }
    raise UnsupportedCheckpointValue(
        f"unsupported checkpoint value {type(value).__module__}.{type(value).__qualname__}"
    )


def _decode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is not dict:
        raise CheckpointReadError("weights-only payload contains an untagged value")
    keys = set(value)
    if keys == {"__float_hex__"}:
        try:
            result = float.fromhex(value["__float_hex__"])
        except (TypeError, ValueError) as exc:
            raise CheckpointReadError("invalid hexadecimal float") from exc
        if not math.isfinite(result):
            raise CheckpointReadError("non-finite decoded float")
        return result
    if keys == {"__tensor__"}:
        tensor = value["__tensor__"]
        if type(tensor) is not torch.Tensor or tensor.device.type != "cpu":
            raise CheckpointReadError("decoded tensor is not an exact CPU tensor")
        return tensor.detach().clone().contiguous()
    if keys == {"__tuple__"}:
        rows = value["__tuple__"]
        if type(rows) is not list:
            raise CheckpointReadError("tuple envelope is invalid")
        return tuple(_decode(item) for item in rows)
    if keys == {"__list__"}:
        rows = value["__list__"]
        if type(rows) is not list:
            raise CheckpointReadError("list envelope is invalid")
        return [_decode(item) for item in rows]
    if keys == {"__mapping__"}:
        rows = value["__mapping__"]
        if type(rows) is not list:
            raise CheckpointReadError("mapping envelope is invalid")
        result: Dict[Any, Any] = {}
        for row in rows:
            if type(row) is not list or len(row) != 2:
                raise CheckpointReadError("mapping item is invalid")
            key, item = _decode(row[0]), _decode(row[1])
            try:
                if key in result:
                    raise CheckpointReadError("mapping contains duplicate key")
                result[key] = item
            except TypeError as exc:
                raise CheckpointReadError("mapping key is not hashable") from exc
        return result
    if keys == {"__enum__", "value"}:
        kind = _ENUM_BY_TAG.get(value["__enum__"])
        if kind is None:
            raise CheckpointReadError("foreign enum type")
        try:
            return kind(value["value"])
        except (TypeError, ValueError) as exc:
            raise CheckpointReadError("invalid enum value") from exc
    if keys == {"__dataclass__", "fields"}:
        tag = value["__dataclass__"]
        raw_fields = value["fields"]
        if tag == _ACTION_TAG:
            decoded = _decode(raw_fields)
            if type(decoded) is not dict:
                raise CheckpointReadError("action fields are invalid")
            try:
                candidate = transaction_identity.ExecutedActionIdentity(**decoded)
                return candidate.reconciled_against(action_contract.load_contract())
            except Exception as exc:
                raise CheckpointReadError("executed action failed reconciliation") from exc
        kind = _DATACLASS_BY_TAG.get(tag)
        if kind is None:
            raise CheckpointReadError("foreign dataclass type")
        if type(raw_fields) is not dict:
            raise CheckpointReadError("dataclass fields are invalid")
        expected = {item.name for item in fields(kind)}
        if set(raw_fields) != expected:
            raise CheckpointReadError(f"field set differs for {kind.__name__}")
        decoded = {name: _decode(item) for name, item in raw_fields.items()}
        try:
            return kind(**decoded)
        except Exception as exc:
            raise CheckpointReadError(
                f"decoded {kind.__name__} failed validation"
            ) from exc
    raise CheckpointReadError("unknown weights-only value envelope")


def _fingerprint(value: Any) -> Any:
    """Canonical content projection independent of torch's ZIP bytes."""

    if type(value) is torch.Tensor:
        tensor = value.detach().cpu().contiguous()
        return {
            "tensor": {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": _sha256_bytes(tensor.numpy().tobytes(order="C")),
            }
        }
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointIoError("non-finite fingerprint float")
        return {"float_hex": value.hex()}
    if type(value) is list:
        return {"list": [_fingerprint(item) for item in value]}
    if type(value) is dict:
        rows = [(_fingerprint(key), _fingerprint(item)) for key, item in value.items()]
        rows.sort(key=lambda row: json.dumps(row[0], sort_keys=True))
        return {"mapping": rows}
    raise CheckpointIoError(f"encoded material contains {type(value).__name__}")


def _material_sha256(encoded: Any) -> str:
    return canonical_sha256(
        {"record": "splitfusion_run4_durable_material_content_v1", "value": _fingerprint(encoded)}
    )


def _manifest_document(
    *,
    payload_sha256: str,
    payload_size_bytes: int,
    material_sha256: str,
    material: DurableCheckpointMaterialV1,
) -> Dict[str, Any]:
    return {
        "material_sha256": material_sha256,
        "payload_filename": PAYLOAD_FILENAME,
        "payload_sha256": payload_sha256,
        "payload_size_bytes": payload_size_bytes,
        "runner_binding_sha256": material.runner_binding_sha256,
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint_sha256": material.source_checkpoint_sha256,
    }


def write_checkpoint(
    directory: str | os.PathLike[str],
    checkpoint: persistent_runner.PersistentRunnerCheckpointV1,
) -> CheckpointArtifactV1:
    """Publish one checkpoint directory atomically and refuse overwrite."""

    target = Path(directory)
    if not target.name or target.parent == target:
        raise CheckpointWriteError("checkpoint target must be a named directory")
    parent = target.parent
    if not parent.is_dir():
        raise CheckpointWriteError("checkpoint parent directory does not exist")
    if target.exists() or target.is_symlink():
        raise CheckpointWriteError("checkpoint target already exists")
    material = DurableCheckpointMaterialV1.from_checkpoint(checkpoint)
    encoded = _encode(material)
    material_sha = _material_sha256(encoded)
    staging = parent / f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        staging.mkdir(mode=0o700)
        payload = staging / PAYLOAD_FILENAME
        with payload.open("xb") as stream:
            torch.save(
                {
                    "schema_id": SCHEMA_ID,
                    "schema_version": SCHEMA_VERSION,
                    "material": encoded,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        payload_sha, payload_size = _sha256_file(payload)
        manifest = _manifest_document(
            payload_sha256=payload_sha,
            payload_size_bytes=payload_size,
            material_sha256=material_sha,
            material=material,
        )
        manifest_bytes = _canonical_json_bytes(manifest)
        manifest_path = staging / MANIFEST_FILENAME
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if target.exists() or target.is_symlink():
            raise CheckpointWriteError("checkpoint target appeared during publication")
        os.rename(staging, target)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return CheckpointArtifactV1(
            directory=str(target),
            manifest_sha256=_sha256_bytes(manifest_bytes),
            payload_sha256=payload_sha,
            payload_size_bytes=payload_size,
            material_sha256=material_sha,
            source_checkpoint_sha256=material.source_checkpoint_sha256,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _regular_file(path: Path, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise CheckpointReadError(f"missing {name}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise CheckpointReadError(f"{name} must be a regular non-symlink file")


def read_checkpoint_material(
    directory: str | os.PathLike[str],
    *,
    expected_runner_binding_sha256: str,
    expected_manifest_sha256: Optional[str] = None,
) -> DurableCheckpointMaterialV1:
    """Verify and decode one atomic checkpoint with weights-only loading."""

    expected_binding = _digest(
        expected_runner_binding_sha256, "expected_runner_binding_sha256"
    )
    if expected_manifest_sha256 is not None:
        expected_manifest_sha256 = _digest(
            expected_manifest_sha256, "expected_manifest_sha256"
        )
    root = Path(directory)
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise CheckpointReadError("checkpoint directory does not exist") from exc
    if not stat.S_ISDIR(mode) or root.is_symlink():
        raise CheckpointReadError("checkpoint root must be a non-symlink directory")
    names = {item.name for item in root.iterdir()}
    if names != {PAYLOAD_FILENAME, MANIFEST_FILENAME}:
        raise CheckpointReadError("checkpoint directory member set differs")
    payload_path = root / PAYLOAD_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    _regular_file(payload_path, PAYLOAD_FILENAME)
    _regular_file(manifest_path, MANIFEST_FILENAME)
    manifest_bytes = manifest_path.read_bytes()
    observed_manifest_sha = _sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 is not None and observed_manifest_sha != (
        expected_manifest_sha256
    ):
        raise CheckpointTamperError("manifest SHA-256 differs")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointReadError("manifest is not canonical JSON") from exc
    expected_manifest_keys = {
        "material_sha256",
        "payload_filename",
        "payload_sha256",
        "payload_size_bytes",
        "runner_binding_sha256",
        "schema_id",
        "schema_version",
        "source_checkpoint_sha256",
    }
    if type(manifest) is not dict or set(manifest) != expected_manifest_keys:
        raise CheckpointReadError("manifest field set differs")
    if _canonical_json_bytes(manifest) != manifest_bytes:
        raise CheckpointReadError("manifest JSON is not canonical")
    if manifest["schema_id"] != SCHEMA_ID or manifest["schema_version"] != SCHEMA_VERSION:
        raise CheckpointReadError("foreign durable checkpoint schema")
    if manifest["payload_filename"] != PAYLOAD_FILENAME:
        raise CheckpointReadError("manifest payload filename differs")
    for name in (
        "material_sha256",
        "payload_sha256",
        "runner_binding_sha256",
        "source_checkpoint_sha256",
    ):
        _digest(manifest[name], f"manifest.{name}")
    _exact_int(manifest["payload_size_bytes"], "manifest.payload_size_bytes", 1)
    if manifest["runner_binding_sha256"] != expected_binding:
        raise CheckpointReadError("checkpoint belongs to a foreign runner binding")
    payload_sha, payload_size = _sha256_file(payload_path)
    if payload_sha != manifest["payload_sha256"] or payload_size != (
        manifest["payload_size_bytes"]
    ):
        raise CheckpointTamperError("checkpoint payload bytes differ from manifest")
    try:
        envelope = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointReadError("weights-only checkpoint load failed") from exc
    if type(envelope) is not dict or set(envelope) != {
        "schema_id",
        "schema_version",
        "material",
    }:
        raise CheckpointReadError("payload envelope field set differs")
    if envelope["schema_id"] != SCHEMA_ID or envelope["schema_version"] != SCHEMA_VERSION:
        raise CheckpointReadError("foreign payload schema")
    encoded = envelope["material"]
    if _material_sha256(encoded) != manifest["material_sha256"]:
        raise CheckpointTamperError("decoded material fingerprint differs")
    material = _decode(encoded)
    if type(material) is not DurableCheckpointMaterialV1:
        raise CheckpointReadError("payload did not decode durable material")
    if material.runner_binding_sha256 != expected_binding:
        raise CheckpointReadError("decoded material belongs to foreign runner")
    if material.runner_binding_sha256 != manifest["runner_binding_sha256"]:
        raise CheckpointReadError("manifest/material runner binding differs")
    if material.source_checkpoint_sha256 != manifest["source_checkpoint_sha256"]:
        raise CheckpointReadError("manifest/material checkpoint digest differs")
    return material


def restore_runner(
    material: DurableCheckpointMaterialV1,
    *,
    fresh_factory: Callable[[], persistent_runner._RunnerCore],
) -> persistent_runner._RunnerCore:
    """Restore verified portable material without deserializing attestations.

    A disposable pristine runner first replays the portable causal inputs via
    the runner's public API. The newly attested rows are then combined with
    the already verified model, optimizer, provider and RNG material. Building
    :class:`PersistentRunnerCheckpointV1` proves that this reconstruction has
    the exact original checkpoint digest. Finally the established runner
    restore path performs an independent replay and all final equality gates.
    """

    if type(material) is not DurableCheckpointMaterialV1:
        raise PortableRestoreApiUnavailable("foreign durable checkpoint material")
    if not callable(fresh_factory):
        raise PortableRestoreApiUnavailable("fresh_factory must be callable")
    rows = tuple(
        persistent_runner.PortableJournalReplayRowV1(
            decision=row.decision,
            prediction=row.prediction,
            successor_staged=row.successor_staged,
            expected_transition_sha256=row.transition_sha256,
            expected_environment_transition_sha256=(
                row.environment_transition_sha256
            ),
            expected_journal_sha256=row.journal_sha256,
        )
        for row in material.journal
    )
    try:
        replay_receiver = fresh_factory()
    except Exception as exc:
        raise CheckpointReadError("fresh runner factory failed") from exc
    runner_type = type(replay_receiver)
    if runner_type not in (
        persistent_runner.Run4PersistentTrainingRunnerV1,
        persistent_runner._TestOnlyPersistentRunnerV1,
    ):
        raise CheckpointReadError("fresh factory returned an unsupported runner type")
    try:
        reissued = replay_receiver.reissue_portable_journal(
            runner_binding_sha256=material.runner_binding_sha256,
            session_uuid=material.session_uuid,
            ue_id=material.ue_id,
            genesis_staged=material.genesis_staged,
            initial_kernel_checkpoint_sha256=(
                material.initial_kernel_checkpoint.canonical_sha256
            ),
            rows=rows,
        )
        checkpoint = persistent_runner.PersistentRunnerCheckpointV1(
            runner_binding_sha256=material.runner_binding_sha256,
            session_uuid=material.session_uuid,
            ue_id=material.ue_id,
            genesis_staged=material.genesis_staged,
            journal=reissued,
            fit_scene_state=material.fit_scene_state,
            initial_kernel_checkpoint=material.initial_kernel_checkpoint,
            kernel_checkpoint=material.kernel_checkpoint,
            state_stager_state=material.state_stager_state,
            prediction_provider_state=material.prediction_provider_state,
            actor_state_dict=material.actor_state_dict,
            critics_state_dict=material.critics_state_dict,
            actor_optimizer_state_dict=material.actor_optimizer_state_dict,
            critic_optimizer_state_dict=material.critic_optimizer_state_dict,
            trainer_update_count=material.trainer_update_count,
            decision_q_rng_state=material.decision_q_rng_state,
            decision_mode_rng_state=material.decision_mode_rng_state,
            replay_rng_state=material.replay_rng_state,
            trainer_target_rng_state=material.trainer_target_rng_state,
            trainer_actor_rng_state=material.trainer_actor_rng_state,
            current_state_sha256=material.current_state_sha256,
            exploration_decision_count=material.exploration_decision_count,
            replay_accepted_count=material.replay_accepted_count,
            replay_evicted_count=material.replay_evicted_count,
            replay_seen_digest_count=material.replay_seen_digest_count,
            replay_seen_identity_count=material.replay_seen_identity_count,
            replay_resident_transition_digests=(
                material.replay_resident_transition_digests
            ),
            checkpoint_sha256=material.source_checkpoint_sha256,
        )
        restored = runner_type.restore(checkpoint, fresh_factory=fresh_factory)
    except Exception as exc:
        raise CheckpointReadError("portable runner restoration failed") from exc
    if restored.checkpoint().checkpoint_sha256 != material.source_checkpoint_sha256:
        raise CheckpointReadError("restored runner digest differs from durable source")
    return restored


__all__ = [
    "CheckpointArtifactV1",
    "CheckpointIoError",
    "CheckpointReadError",
    "CheckpointTamperError",
    "CheckpointWriteError",
    "DurableCheckpointMaterialV1",
    "MANIFEST_FILENAME",
    "PAYLOAD_FILENAME",
    "PortableJournalEntryV1",
    "PortableRestoreApiUnavailable",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "UnsupportedCheckpointValue",
    "read_checkpoint_material",
    "restore_runner",
    "write_checkpoint",
]
