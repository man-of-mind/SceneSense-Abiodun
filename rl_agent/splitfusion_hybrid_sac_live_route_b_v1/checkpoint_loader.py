"""Hash-verified loader for the pre-registered Run-3 seed-17 update-10,000 actor.

The loader walks the campaign's own provenance chain rather than trusting a
single pinned digest:

1. ``campaign_complete.json`` -- self-consistent, names seed 17 and carries the
   canonical digest of that seed's report;
2. ``RUN3_TRAINING_COMPLETE.json`` -- self-consistent, declares
   ``RUN3_FIXED_ENDPOINT_COMPLETE`` at update 10,000, and binds both the
   report's file digest and its canonical digest;
3. ``report.json`` -- its canonical digest is recomputed from its own content,
   it agrees with both markers above, and its ``artifact_hashes`` inventory is
   re-hashed file by file;
4. the model-only snapshot -- file digest matches the inventory *and* the
   module pin, and its internal ``snapshot_sha256`` is recomputed with the
   runner's own ``_hash_state`` implementation;
5. optionally, the 131 MB full checkpoint -- its actor tensors are compared
   bit-for-bit against the snapshot's.

The snapshot is loaded with ``weights_only=True``.  It is a flat mapping of
tensors and scalars, so no pickled object from the training process is ever
reconstructed in the live pilot.  The full checkpoint necessarily requires
``weights_only=False`` and is therefore opt-in and used only for
cross-verification, never as the live weight source.

Loading reads files and nothing else: no CARLA, OAI, Docker, network service or
CUDA context.  ``torch.cuda.is_initialized()`` is asserted False on entry and
on exit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_run3_runner import (
    RUN3_MODEL_SNAPSHOT_SCHEMA,
    _hash_state,
)

from . import pilot_contract as contract

__all__ = [
    "CheckpointLoadError",
    "LoadedPilotActorWeightsV1",
    "load_pilot_actor_weights",
    "project_root",
]

_TERMINAL_STATUS = "RUN3_FIXED_ENDPOINT_COMPLETE"
_TERMINAL_SCHEMA = "splitfusion.run3_training_terminal.v1"
_REPORT_RECORD = "run3_training_report_v1"
_CAMPAIGN_RECORD = "run3_three_seed_campaign_complete_v1"

#: Files the report's inventory deliberately excludes (they name the report).
_INVENTORY_EXCLUDED = ("report.json", "RUN3_TRAINING_COMPLETE.json")


class CheckpointLoadError(RuntimeError):
    """The pre-registered actor artifact chain is incomplete or altered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointLoadError(message)


def project_root() -> Path:
    """The ``abiodun/`` repository root, resolved from this module's location."""
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointLoadError(f"{label} is unreadable at {path}") from exc


def _assert_cuda_uninitialized(when: str) -> None:
    _require(
        not torch.cuda.is_initialized(),
        f"CUDA is initialized {when}; the pilot loader is CPU-only",
    )


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LoadedPilotActorWeightsV1:
    """Verified, read-only actor weights plus the proof that produced them.

    ``actor_state`` is the exact ``ConditionalHybridActor`` state dict recorded
    at seed 17, update 10,000.  Critic states are retained only so the
    provenance record is complete; the live pilot never builds a critic,
    because a critic is a learning artifact and this is deployment validation.
    """

    actor_state: Mapping[str, torch.Tensor]
    critic_1_state: Mapping[str, torch.Tensor]
    critic_2_state: Mapping[str, torch.Tensor]
    seed: int
    update: int
    snapshot_path: str
    snapshot_file_sha256: str
    snapshot_content_sha256: str
    runner_binding_sha256: str
    report_file_sha256: str
    report_canonical_sha256: str
    campaign_complete_sha256: str
    inventory_verified_file_count: int
    full_checkpoint_cross_verified: bool
    full_checkpoint_file_sha256: Optional[str]
    verification_chain: Tuple[str, ...]

    @property
    def actor_state_sha256(self) -> str:
        """Digest of the actor tensors alone, independent of the file format."""
        return _hash_state(dict(self.actor_state))

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "actor_state_sha256": self.actor_state_sha256,
            "campaign_complete_sha256": self.campaign_complete_sha256,
            "full_checkpoint_cross_verified": self.full_checkpoint_cross_verified,
            "full_checkpoint_file_sha256": self.full_checkpoint_file_sha256,
            "inventory_verified_file_count": self.inventory_verified_file_count,
            "record": "splitfusion.live_route_b_pilot_actor_weights.v1",
            "report_canonical_sha256": self.report_canonical_sha256,
            "report_file_sha256": self.report_file_sha256,
            "runner_binding_sha256": self.runner_binding_sha256,
            "seed": self.seed,
            "snapshot_content_sha256": self.snapshot_content_sha256,
            "snapshot_file_sha256": self.snapshot_file_sha256,
            "snapshot_path": self.snapshot_path,
            "update": self.update,
            "verification_chain": list(self.verification_chain),
        }

    def canonical_sha256(self) -> str:
        return contract.canonical_sha256(self.to_canonical_dict())


# --------------------------------------------------------------------------- #
# Chain verification
# --------------------------------------------------------------------------- #


def _verify_campaign_complete(root: Path, chain: list) -> Tuple[str, str]:
    path = root / contract.CAMPAIGN_COMPLETE_RELATIVE_PATH
    document = _read_json(path, label="campaign_complete.json")
    _require(
        document.get("record") == _CAMPAIGN_RECORD,
        f"campaign marker record drift: {document.get('record')!r}",
    )
    _require(
        contract.PREREGISTERED_SEED in list(document.get("seeds") or []),
        f"seed {contract.PREREGISTERED_SEED} is absent from the campaign marker",
    )
    reports = document.get("seed_reports")
    _require(isinstance(reports, dict), "campaign marker lacks seed_reports")
    expected_report_digest = reports.get(str(contract.PREREGISTERED_SEED))
    _require(
        isinstance(expected_report_digest, str) and len(expected_report_digest) == 64,
        "campaign marker does not name the seed-17 report digest",
    )
    chain.append("campaign_complete.json:seed_reports[17]")
    return _sha256_file(path), str(expected_report_digest)


def _verify_terminal_marker(seed_dir: Path, chain: list) -> Tuple[str, str]:
    path = seed_dir / "RUN3_TRAINING_COMPLETE.json"
    document = _read_json(path, label="RUN3_TRAINING_COMPLETE.json")
    supplied = dict(document)
    declared = supplied.pop("terminal_sha256", None)
    _require(
        declared == contract.canonical_sha256(supplied),
        "terminal marker canonical digest drift",
    )
    _require(
        document.get("schema") == _TERMINAL_SCHEMA,
        f"terminal marker schema drift: {document.get('schema')!r}",
    )
    _require(
        document.get("status") == _TERMINAL_STATUS,
        f"seed {contract.PREREGISTERED_SEED} did not complete: "
        f"{document.get('status')!r}",
    )
    _require(
        int(document.get("seed", -1)) == contract.PREREGISTERED_SEED,
        "terminal marker seed drift",
    )
    _require(
        int(document.get("update", -1)) == contract.PREREGISTERED_UPDATE,
        f"terminal marker is not the pre-registered update "
        f"{contract.PREREGISTERED_UPDATE}: {document.get('update')!r}",
    )
    chain.append("RUN3_TRAINING_COMPLETE.json:terminal_sha256")
    return str(document["report_file_sha256"]), str(document["report_sha256"])


def _verify_report(
    seed_dir: Path,
    *,
    expected_file_sha256: str,
    expected_canonical_sha256: Tuple[str, ...],
    chain: list,
) -> Tuple[Dict[str, Any], str, str, int]:
    path = seed_dir / "report.json"
    observed_file_sha = _sha256_file(path)
    _require(
        observed_file_sha == contract.REPORT_FILE_SHA256,
        f"report.json file digest differs from the module pin: "
        f"{observed_file_sha}",
    )
    _require(
        observed_file_sha == expected_file_sha256,
        "report.json file digest differs from the terminal marker's binding",
    )
    document = _read_json(path, label="report.json")
    _require(
        document.get("record") == _REPORT_RECORD,
        f"report record drift: {document.get('record')!r}",
    )
    supplied = dict(document)
    declared = supplied.pop("report_sha256", None)
    recomputed = contract.canonical_sha256(supplied)
    _require(declared == recomputed, "report canonical digest drift")
    for expected in expected_canonical_sha256:
        _require(
            declared == expected,
            "report canonical digest disagrees with an upstream marker",
        )
    summary = document.get("summary") or {}
    _require(
        int(summary.get("seed", -1)) == contract.PREREGISTERED_SEED,
        "report summary seed drift",
    )
    _require(
        int(summary.get("completed_updates", -1)) == contract.PREREGISTERED_UPDATE,
        "report summary did not complete the pre-registered update count",
    )
    _require(
        bool(summary.get("completed_training_hard_gates_passed")),
        "report summary did not pass the training hard gates",
    )
    _require(
        document.get("peak_checkpoint_selection") is False,
        "report declares peak-checkpoint selection; the pilot endpoint is fixed",
    )
    _require(
        int(document.get("fixed_endpoint_update", -1))
        == contract.PREREGISTERED_UPDATE,
        "report fixed endpoint is not the pre-registered update",
    )

    hashes = document.get("artifact_hashes")
    _require(isinstance(hashes, dict), "report lacks an artifact inventory")
    observed = {
        str(item.relative_to(seed_dir)): _sha256_file(item)
        for item in sorted(seed_dir.rglob("*"))
        if item.is_file() and item.name not in _INVENTORY_EXCLUDED
    }
    _require(
        observed == hashes,
        "seed artifact inventory drift: the recorded and observed file "
        "digests differ",
    )
    chain.append("report.json:report_sha256")
    chain.append(f"report.json:artifact_hashes[{len(observed)} files]")
    return document, observed_file_sha, str(declared), len(observed)


def _verify_snapshot_document(document: Mapping[str, Any], chain: list) -> str:
    _require(
        isinstance(document, dict),
        f"snapshot must be a mapping, got {type(document).__name__}",
    )
    _require(
        document.get("schema") == RUN3_MODEL_SNAPSHOT_SCHEMA,
        f"snapshot schema drift: {document.get('schema')!r}",
    )
    _require(
        int(document.get("seed", -1)) == contract.PREREGISTERED_SEED,
        f"snapshot seed is not the pre-registered {contract.PREREGISTERED_SEED}",
    )
    _require(
        int(document.get("update", -1)) == contract.PREREGISTERED_UPDATE,
        f"snapshot update is not the pre-registered "
        f"{contract.PREREGISTERED_UPDATE}",
    )
    _require(
        document.get("binding_sha256") == contract.REGISTERED_RUNNER_BINDING_SHA256,
        "snapshot runner-binding digest differs from the registered seed-17 "
        "binding",
    )
    expected = {
        "actor_state",
        "binding_sha256",
        "critic_1_state",
        "critic_2_state",
        "schema",
        "seed",
        "snapshot_sha256",
        "update",
    }
    _require(
        set(document) == expected,
        f"snapshot key inventory drift: {sorted(set(document) ^ expected)}",
    )
    content = {key: value for key, value in document.items() if key != "snapshot_sha256"}
    recomputed = _hash_state(content)
    _require(
        recomputed == document.get("snapshot_sha256"),
        "snapshot content digest drift under the runner's own _hash_state",
    )
    chain.append("model_snapshot:snapshot_sha256")
    return str(recomputed)


def _cross_verify_full_checkpoint(
    root: Path, actor_state: Mapping[str, torch.Tensor], chain: list
) -> str:
    """Prove the snapshot actor is bit-identical to the full checkpoint's.

    The full checkpoint is a pickled ``Run3CheckpointV1``; it is read only to
    corroborate the snapshot and never becomes the live weight source.
    """
    path = root / contract.FULL_CHECKPOINT_RELATIVE_PATH
    observed = _sha256_file(path)
    _require(
        observed == contract.FULL_CHECKPOINT_FILE_SHA256,
        f"full checkpoint file digest drift: {observed}",
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(
        type(checkpoint).__name__ == "Run3CheckpointV1",
        f"full checkpoint holds a foreign object: {type(checkpoint).__name__}",
    )
    _require(
        int(checkpoint.seed) == contract.PREREGISTERED_SEED
        and int(checkpoint.update_count) == contract.PREREGISTERED_UPDATE,
        "full checkpoint seed/update drift",
    )
    _require(
        checkpoint.runner_binding_sha256
        == contract.REGISTERED_RUNNER_BINDING_SHA256,
        "full checkpoint runner-binding drift",
    )
    _require(
        checkpoint.collection_session_uuid
        == contract.REGISTERED_SEED17_SESSION_UUID,
        "full checkpoint decision-lineage session drift",
    )
    reference = checkpoint.actor_state
    _require(
        sorted(reference) == sorted(actor_state),
        "full checkpoint and snapshot actor key sets differ",
    )
    for name, tensor in reference.items():
        candidate = actor_state[name]
        _require(
            tensor.dtype == candidate.dtype and tensor.shape == candidate.shape,
            f"actor tensor {name} shape/dtype differs between artifacts",
        )
        _require(
            torch.equal(tensor, candidate),
            f"actor tensor {name} differs between the snapshot and the "
            f"full checkpoint",
        )
    chain.append("full_checkpoint:bitwise_actor_equality")
    return observed


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def load_pilot_actor_weights(
    *,
    root: Optional[Path] = None,
    cross_verify_full_checkpoint: bool = True,
) -> LoadedPilotActorWeightsV1:
    """Load and fully verify the pre-registered seed-17 update-10,000 actor.

    Args:
        root: Repository root; defaults to the resolved ``abiodun/`` directory.
        cross_verify_full_checkpoint: When True (the default, and the required
            setting for any live preflight) the 131 MB full checkpoint is read
            and its actor tensors are compared bit-for-bit.  Tests that only
            need the weights may disable it.

    Raises:
        CheckpointLoadError: on any break in the provenance chain.  There is no
            partial success and no fallback artifact.
    """
    _assert_cuda_uninitialized("before loading the pre-registered actor")
    base = project_root() if root is None else Path(root).resolve(strict=True)
    seed_dir = base / contract.SEED_DIRECTORY_RELATIVE_PATH
    _require(seed_dir.is_dir(), f"seed directory is missing at {seed_dir}")

    chain: list = []
    campaign_sha, report_digest_from_campaign = _verify_campaign_complete(base, chain)
    report_file_from_terminal, report_digest_from_terminal = _verify_terminal_marker(
        seed_dir, chain
    )
    report, report_file_sha, report_canonical_sha, inventory_count = _verify_report(
        seed_dir,
        expected_file_sha256=report_file_from_terminal,
        expected_canonical_sha256=(
            report_digest_from_campaign,
            report_digest_from_terminal,
        ),
        chain=chain,
    )

    snapshot_path = base / contract.SNAPSHOT_RELATIVE_PATH
    _require(snapshot_path.is_file(), f"snapshot is missing at {snapshot_path}")
    snapshot_file_sha = _sha256_file(snapshot_path)
    _require(
        snapshot_file_sha == contract.SNAPSHOT_FILE_SHA256,
        f"snapshot file digest differs from the module pin: {snapshot_file_sha}",
    )
    relative = str(snapshot_path.relative_to(seed_dir))
    _require(
        report["artifact_hashes"].get(relative) == snapshot_file_sha,
        "snapshot file digest differs from the report inventory",
    )
    chain.append("module_pin:snapshot_file_sha256")

    try:
        document = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim
        raise CheckpointLoadError(
            f"snapshot at {snapshot_path} could not be loaded with "
            f"weights_only=True: {type(exc).__name__}: {exc}"
        ) from exc
    content_sha = _verify_snapshot_document(document, chain)

    actor_state = MappingProxyType(dict(document["actor_state"]))
    full_sha: Optional[str] = None
    if cross_verify_full_checkpoint:
        full_sha = _cross_verify_full_checkpoint(base, actor_state, chain)

    _assert_cuda_uninitialized("after loading the pre-registered actor")
    return LoadedPilotActorWeightsV1(
        actor_state=actor_state,
        critic_1_state=MappingProxyType(dict(document["critic_1_state"])),
        critic_2_state=MappingProxyType(dict(document["critic_2_state"])),
        seed=int(document["seed"]),
        update=int(document["update"]),
        snapshot_path=str(snapshot_path.relative_to(base)),
        snapshot_file_sha256=snapshot_file_sha,
        snapshot_content_sha256=content_sha,
        runner_binding_sha256=str(document["binding_sha256"]),
        report_file_sha256=report_file_sha,
        report_canonical_sha256=report_canonical_sha,
        campaign_complete_sha256=campaign_sha,
        inventory_verified_file_count=inventory_count,
        full_checkpoint_cross_verified=cross_verify_full_checkpoint,
        full_checkpoint_file_sha256=full_sha,
        verification_chain=tuple(chain),
    )
