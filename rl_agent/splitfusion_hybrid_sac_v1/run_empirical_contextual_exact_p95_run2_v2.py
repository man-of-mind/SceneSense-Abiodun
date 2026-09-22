"""Durable CLI for isolated exact-P95 Run-2-v2 training.

The writer is deliberately train-only.  It resumes solely from exact v2
runner checkpoints and never imports an evaluation surface.  Existing output
is treated as a closed artifact set and is completely validated before any
file is replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import struct
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .empirical_contextual_exact_p95_run2_runner_v2 import (
    EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2,
    PHASE_LABEL,
    REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
    RUNNER_SCHEMA,
    ExactP95Run2CheckpointV2,
    ExactP95Run2RunnerConfigV2,
    ExactP95Run2RunnerErrorV2,
    ExactP95Run2RunnerV2,
)
from .empirical_contextual_exact_p95_run2_terminal_trainer_v2 import (
    ExactP95Run2TerminalUpdateMetricsV2,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "OUTPUT_SCHEMA",
    "Run2V2ArtifactError",
    "load_exact_p95_run2_checkpoint_v2",
    "run_seed_to_directory",
]


OUTPUT_SCHEMA = "splitfusion.exact_p95_run2_training_output.v2"
CONFIG_SCHEMA = "splitfusion.exact_p95_run2_training_config.v2"
BINDINGS_SCHEMA = "splitfusion.exact_p95_run2_training_bindings.v2"
REPORT_SCHEMA = "splitfusion.exact_p95_run2_training_report.v2"
CAMPAIGN_SCHEMA = "splitfusion.exact_p95_run2_training_campaign.v2"
TARGET_RULE = (
    "AUTHENTICATED_EXACT_P95_RUN2_V2_EMITTED_CPU_FLOAT32_TERMINAL_REWARD"
)
CHECKPOINT_SELECTION = "NONE_UPDATE_5000_IS_THE_FIXED_PRIMARY_ENDPOINT"
FROZEN_COMPARATOR_COMMIT = "68f2a291783bd3546db6b49c1cbd10badbfa3fca"
FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256 = (
    "9711495c7a0c47edf5ac5362a192261a0ab96cc743df6b1429c51a9fc1001ab1"
)
FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256 = (
    "6166cfb4593949386802d06eaa2c4198b196d8a5734d17764d5a0b9649a58719"
)
FROZEN_COMPARATOR_MANIFEST_FILE_SHA256 = (
    "2c95fb808f0099bc25fefdb4f071ab27533991b68b62c27fcd7e031f2a8b86ba"
)
FROZEN_COMPARATOR_DECISION_CONTENT_SHA256 = (
    "d85a016366bced4fc85f06821389609981d89687175390c2518d11fe2fae8f51"
)
FROZEN_COMPARATOR_IMPLEMENTATION_SHA256 = (
    "cffaea677832196fcfb273a44a88a8282aef49c3a357ce47cb04b6ea3e38af67"
)
_COMPARATOR_RELATIVE_DIRECTORY = (
    "experiments/splitfusion_hybrid_sac_run2_fixed_comparator_v2/"
    "20260921_exact_p95_run2_fixed_comparator_v2"
)
_COMPARATOR_FILE_HASHES = {
    "REPORT.md": "8eafc6987ac627e416c332535a5ab7a073ef6bb4a28197df05e95f558e1e78de",
    "action_scores.csv": "a5f617cd5cdf57bffa85ee2dcd04d138247eed0e3e95d2a3a71473fb110203c7",
    "context_winner_revalidation.csv": "bd6af2f988acd89d21c44bb112596c488f4b6d6dcb1d20a7b21d69b98e28dda8",
    "decision.json": "01b2bc229791b55f498339c01467c93c647cd17fa3ca6aa4eaaa3a04586b3038",
    "summary.json": "6fcb4d68cc49f7a89ce0a552f0721fd209904d92bb9089d583aca56685986be5",
}
_CHECKPOINT_NAME = re.compile(r"^checkpoint_([0-9]{6})\.pt$")
_ROOT_INVENTORY = frozenset(
    {
        "bindings.json",
        "checkpoint_000000.pt",
        "checkpoint_latest.pt",
        "checkpoints",
        "config.json",
        "metrics.csv",
        "report.json",
        "transition_reward_audit.csv",
    }
)


class Run2V2ArtifactError(RuntimeError):
    """A v2 output/checkpoint failed closed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root(project_root: Optional[Path]) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[2]
    return Path(project_root).resolve(strict=True)


def _fixed_comparator_binding(project_root: Optional[Path]) -> Dict[str, Any]:
    """Revalidate the comparator frozen before any Run-2 outcome exists."""
    root = _project_root(project_root)
    directory = root / _COMPARATOR_RELATIVE_DIRECTORY
    implementation = (
        root
        / "rl_agent/splitfusion_hybrid_sac_v1/"
        "empirical_contextual_exact_p95_run2_fixed_comparator_v2.py"
    )
    for name, expected in _COMPARATOR_FILE_HASHES.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise Run2V2ArtifactError(
                f"frozen comparator artifact drift at {name}"
            )
    manifest_path = directory / "artifact_manifest.json"
    if (
        not manifest_path.is_file()
        or _sha256_file(manifest_path)
        != FROZEN_COMPARATOR_MANIFEST_FILE_SHA256
        or not implementation.is_file()
        or _sha256_file(implementation)
        != FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
    ):
        raise Run2V2ArtifactError("frozen comparator manifest/implementation drift")
    summary = _read_json(directory / "summary.json")
    manifest = _read_json(manifest_path)
    decision = _read_json(directory / "decision.json")
    _require_content_attestation(summary, field_name="summary_content_sha256")
    _require_content_attestation(manifest, field_name="manifest_content_sha256")
    _require_content_attestation(decision, field_name="decision_content_sha256")
    if (
        summary.get("summary_content_sha256")
        != FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256
        or manifest.get("manifest_content_sha256")
        != FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256
        or manifest.get("summary_content_sha256")
        != FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256
        or manifest.get("files") != _COMPARATOR_FILE_HASHES
        or decision.get("decision_content_sha256")
        != FROZEN_COMPARATOR_DECISION_CONTENT_SHA256
        or decision.get("decision")
        != "GO_FIXED_COMPARATOR_FROZEN_BEFORE_RUN2_OUTCOME_ACCESS"
        or summary.get("bindings", {}).get("comparator_implementation_sha256")
        != FROZEN_COMPARATOR_IMPLEMENTATION_SHA256
        or summary.get("winner") != decision.get("winner")
    ):
        raise Run2V2ArtifactError("frozen comparator semantic binding drift")
    try:
        resolved_commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{FROZEN_COMPARATOR_COMMIT}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                FROZEN_COMPARATOR_COMMIT,
                "HEAD",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Run2V2ArtifactError(
            "cannot authenticate frozen comparator commit"
        ) from exc
    if resolved_commit != FROZEN_COMPARATOR_COMMIT or ancestor.returncode != 0:
        raise Run2V2ArtifactError(
            "frozen comparator commit is absent or not ancestral"
        )
    return {
        "artifact_file_sha256": dict(_COMPARATOR_FILE_HASHES),
        "artifact_manifest_file_sha256": (
            FROZEN_COMPARATOR_MANIFEST_FILE_SHA256
        ),
        "decision_content_sha256": FROZEN_COMPARATOR_DECISION_CONTENT_SHA256,
        "fixed_action": dict(decision["winner"]),
        "frozen_commit": FROZEN_COMPARATOR_COMMIT,
        "frozen_before_run2_outcomes": True,
        "implementation_sha256": FROZEN_COMPARATOR_IMPLEMENTATION_SHA256,
        "manifest_content_sha256": FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256,
        "relative_directory": _COMPARATOR_RELATIVE_DIRECTORY,
        "summary_content_sha256": FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256,
    }


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _content_attested(
    document: Dict[str, Any], *, field_name: str
) -> Dict[str, Any]:
    if field_name in document:
        raise Run2V2ArtifactError("content digest field already present")
    result = dict(document)
    result[field_name] = canonical_sha256(result)
    return result


def _require_content_attestation(
    document: Mapping[str, Any], *, field_name: str
) -> None:
    supplied = document.get(field_name)
    payload = {key: value for key, value in document.items() if key != field_name}
    if supplied != canonical_sha256(payload):
        raise Run2V2ArtifactError(f"{field_name} mismatch")


def _atomic_durable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    _atomic_durable_bytes(path, _json_bytes(document))


def _dealias_checkpoint_value(value: Any) -> Any:
    """Rebuild equal values without construction-history object aliases."""
    if type(value) is str:
        return value.encode("utf-8").decode("utf-8")
    if type(value) is float:
        return struct.unpack(">d", struct.pack(">d", value))[0]
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if hasattr(value, "__dataclass_fields__"):
        return type(value)(
            **{
                field.name: _dealias_checkpoint_value(
                    getattr(value, field.name)
                )
                for field in fields(value)
            }
        )
    if isinstance(value, Mapping):
        return {
            _dealias_checkpoint_value(key): _dealias_checkpoint_value(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_dealias_checkpoint_value(child) for child in value)
    if isinstance(value, list):
        return [_dealias_checkpoint_value(child) for child in value]
    return value


def _checkpoint_bytes(checkpoint: ExactP95Run2CheckpointV2) -> bytes:
    checkpoint.require_valid()
    # Uninterrupted rows naturally share immutable D1 bindings across the
    # whole history.  After resume, restored rows and newly collected rows
    # form two equal-but-distinct alias groups.  Pickle records those aliases,
    # making equal scientific state produce different bytes.  Reconstructing
    # every value independently removes that incidental topology.
    normalized = _dealias_checkpoint_value(checkpoint)
    if type(normalized) is not ExactP95Run2CheckpointV2:
        raise Run2V2ArtifactError("checkpoint normalization changed exact type")
    normalized.require_valid()
    buffer = io.BytesIO()
    torch.save(normalized, buffer)
    return buffer.getvalue()


def _atomic_checkpoint(path: Path, checkpoint: ExactP95Run2CheckpointV2) -> None:
    _atomic_durable_bytes(path, _checkpoint_bytes(checkpoint))


def load_exact_p95_run2_checkpoint_v2(path: Path) -> ExactP95Run2CheckpointV2:
    """Load only the exact v2 checkpoint type and revalidate its digest."""
    path = Path(path)
    try:
        with path.open("rb") as stream:
            checkpoint = torch.load(
                stream, map_location="cpu", weights_only=False
            )
    except Exception as exc:
        raise Run2V2ArtifactError(
            f"could not load v2 checkpoint {path.name}"
        ) from exc
    if type(checkpoint) is not ExactP95Run2CheckpointV2:
        raise Run2V2ArtifactError("checkpoint file contains a foreign object")
    try:
        checkpoint.require_valid()
    except Exception as exc:
        raise Run2V2ArtifactError("checkpoint failed v2 revalidation") from exc
    return checkpoint


# Backwards-private spelling used by sibling writers and focused tests.
_load_checkpoint = load_exact_p95_run2_checkpoint_v2


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Run2V2ArtifactError(f"malformed JSON artifact {path.name}") from exc
    if type(document) is not dict:
        raise Run2V2ArtifactError(f"{path.name} is not a JSON object")
    return document


def _float64_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>Q', struct.pack('>d', float(value)))[0]:016x}"


def _float32_bits_hex(value: float) -> str:
    return f"0x{struct.unpack('>I', struct.pack('>f', float(value)))[0]:08x}"


def _config_document(
    config: ExactP95Run2RunnerConfigV2,
    seed: int,
    checkpoint_interval_updates: int,
    comparator_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    return _content_attested(
        {
            "checkpoint_interval_updates": checkpoint_interval_updates,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "config": config.to_canonical_dict(),
            "config_sha256": config.canonical_sha256(),
            "fixed_comparator": dict(comparator_binding),
            "output_schema": OUTPUT_SCHEMA,
            "phase_label": PHASE_LABEL,
            "record": CONFIG_SCHEMA,
            "seed": seed,
            "target_rule": TARGET_RULE,
            "training_population": "REGISTERED_TRAIN_IDS_ONLY",
            "validation_access_during_training": "FORBIDDEN",
        },
        field_name="config_content_sha256",
    )


def _bindings_document(
    runner: ExactP95Run2RunnerV2,
    comparator_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    replay_binding = runner.replay.binding
    document = {
        "collection_session_uuid": runner.collection_session_uuid,
        "config_sha256": runner.config.canonical_sha256(),
        "d1_binding": runner.environment.binding.to_canonical_dict(),
        "d1_binding_sha256": runner._d1_binding_sha256,
        "fit_partition_sha256": runner.environment.fit_partition_sha256,
        "fixed_comparator": dict(comparator_binding),
        "output_schema": OUTPUT_SCHEMA,
        "phase_label": PHASE_LABEL,
        "record": BINDINGS_SCHEMA,
        "replay_binding": (
            None if replay_binding is None else replay_binding.to_canonical_dict()
        ),
        "replay_binding_sha256": (
            None if replay_binding is None else replay_binding.canonical_sha256()
        ),
        "reward_binding": (
            None
            if replay_binding is None
            else replay_binding.reward_binding.to_canonical_dict()
        ),
        "reward_binding_sha256": (
            None
            if replay_binding is None
            else replay_binding.reward_binding.canonical_sha256()
        ),
        "runner_binding": runner.runner_binding_document,
        "runner_binding_sha256": runner.runner_binding_sha256,
        "runner_schema": RUNNER_SCHEMA,
        "sampling_contract_sha256": runner.environment.sampling_contract_sha256,
        "sampling_split": runner.environment.sampling_split,
        "seed": runner.seed,
        "trainer_config": runner.trainer_config.to_canonical_dict(),
        "trainer_config_sha256": runner.trainer_config.canonical_sha256(),
    }
    return _content_attested(document, field_name="bindings_content_sha256")


_METRIC_FIELDS = ("update_index",) + tuple(
    field.name for field in fields(ExactP95Run2TerminalUpdateMetricsV2)
)


def _metrics_csv(metrics: Iterable[ExactP95Run2TerminalUpdateMetricsV2]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(_METRIC_FIELDS))
    writer.writeheader()
    for index, metric in enumerate(metrics, 1):
        if type(metric) is not ExactP95Run2TerminalUpdateMetricsV2:
            raise Run2V2ArtifactError("metrics contain a foreign row")
        metric.assert_finite()
        writer.writerow({"update_index": index, **metric.as_dict()})
    return stream.getvalue().encode("utf-8")


_AUDIT_FIELDS = (
    "collection_seq",
    "collection_session_uuid",
    "source_d1_transition_sha256",
    "run2_v2_transition_sha256",
    "reward_binding_sha256",
    "sample_id",
    "episode_id",
    "frame_id",
    "hidden_network_profile",
    "hidden_radio_csv_row_number",
    "hidden_trace_id",
    "hidden_trace_step_index",
    "mode_id",
    "q_e4",
    "source_d1_reward64",
    "source_d1_reward64_bits_hex",
    "p95_base_reward64",
    "p95_base_reward64_bits_hex",
    "shaped_reward64",
    "shaped_reward64_bits_hex",
    "emitted_reward_float32",
    "emitted_reward_float32_bits_hex",
)


def _reward_audit_csv(runner: ExactP95Run2RunnerV2) -> bytes:
    d1_history = runner.d1_transition_history
    v2_history = runner.run2_v2_transition_history
    if len(d1_history) != len(v2_history):
        raise Run2V2ArtifactError("D1/v2 reward history cardinality drift")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(_AUDIT_FIELDS))
    writer.writeheader()
    for source, shaped in zip(d1_history, v2_history):
        source.revalidate()
        shaped.revalidate()
        if (
            source.canonical_sha256()
            != shaped.source_d1_transition.canonical_sha256()
            or source.reward != shaped.source_d1_reward64
        ):
            raise Run2V2ArtifactError("D1/v2 reward audit source mismatch")
        audit = source.result.audit
        writer.writerow(
            {
                "collection_seq": source.collection_seq,
                "collection_session_uuid": source.collection_session_uuid,
                "source_d1_transition_sha256": source.canonical_sha256(),
                "run2_v2_transition_sha256": shaped.canonical_sha256(),
                "reward_binding_sha256": shaped.reward_binding.canonical_sha256(),
                "sample_id": audit.sample_id,
                "episode_id": audit.episode_id,
                "frame_id": audit.frame_id,
                "hidden_network_profile": audit.hidden_network_profile,
                "hidden_radio_csv_row_number": (
                    audit.hidden_radio_csv_row_number
                ),
                "hidden_trace_id": audit.hidden_trace_id,
                "hidden_trace_step_index": audit.hidden_trace_step_index,
                "mode_id": source.action.mode_id,
                "q_e4": source.action.q_e4,
                "source_d1_reward64": shaped.source_d1_reward64,
                "source_d1_reward64_bits_hex": _float64_bits_hex(
                    shaped.source_d1_reward64
                ),
                "p95_base_reward64": shaped.p95_base_reward64,
                "p95_base_reward64_bits_hex": _float64_bits_hex(
                    shaped.p95_base_reward64
                ),
                "shaped_reward64": shaped.shaped_reward64,
                "shaped_reward64_bits_hex": _float64_bits_hex(
                    shaped.shaped_reward64
                ),
                "emitted_reward_float32": shaped.emitted_reward_float32,
                "emitted_reward_float32_bits_hex": _float32_bits_hex(
                    shaped.emitted_reward_float32
                ),
            }
        )
    return stream.getvalue().encode("utf-8")


def _expected_numbered_updates(
    *, latest_update: int, configured_updates: int, interval: int
) -> Tuple[int, ...]:
    updates = list(range(interval, latest_update + 1, interval))
    if latest_update == configured_updates and (
        not updates or updates[-1] != latest_update
    ):
        updates.append(latest_update)
    return tuple(updates)


def _numbered_checkpoint_paths(output_directory: Path) -> Dict[int, Path]:
    directory = output_directory / "checkpoints"
    if not directory.is_dir():
        raise Run2V2ArtifactError("checkpoints directory is missing")
    result: Dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            raise Run2V2ArtifactError("checkpoints contains a non-file entry")
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            raise Run2V2ArtifactError(
                f"unexpected checkpoint-series file {path.name}"
            )
        update = int(match.group(1))
        if update == 0 or update in result:
            raise Run2V2ArtifactError("invalid numbered checkpoint index")
        result[update] = path
    return result


def _report_document(
    *,
    output_directory: Path,
    runner: ExactP95Run2RunnerV2,
    checkpoint_interval_updates: int,
    checkpoint: ExactP95Run2CheckpointV2,
    comparator_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    update_zero_path = output_directory / "checkpoint_000000.pt"
    latest_path = output_directory / "checkpoint_latest.pt"
    config_path = output_directory / "config.json"
    bindings_path = output_directory / "bindings.json"
    metrics_path = output_directory / "metrics.csv"
    audit_path = output_directory / "transition_reward_audit.csv"
    numbered = _numbered_checkpoint_paths(output_directory)
    checkpoint_files = {
        f"{update:06d}": {
            "path": f"checkpoints/checkpoint_{update:06d}.pt",
            "file_sha256": _sha256_file(path),
        }
        for update, path in sorted(numbered.items())
    }
    complete = runner.completed_updates == runner.config.update_count
    document = {
        "artifact_file_sha256": {
            "bindings.json": _sha256_file(bindings_path),
            "checkpoint_000000.pt": _sha256_file(update_zero_path),
            "checkpoint_latest.pt": _sha256_file(latest_path),
            "config.json": _sha256_file(config_path),
            "metrics.csv": _sha256_file(metrics_path),
            "transition_reward_audit.csv": _sha256_file(audit_path),
        },
        "artifact_inventory": {
            "bindings": "bindings.json",
            "checkpoint_latest": "checkpoint_latest.pt",
            "checkpoint_series": "checkpoints/checkpoint_*.pt",
            "checkpoint_update_zero": "checkpoint_000000.pt",
            "config": "config.json",
            "metrics": "metrics.csv",
            "report": "report.json",
            "transition_reward_audit": "transition_reward_audit.csv",
        },
        "checkpoint_cadence_updates": checkpoint_interval_updates,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "checkpoint_series": checkpoint_files,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "completed_updates": runner.completed_updates,
        "configured_updates": runner.config.update_count,
        "fit_partition_sha256": runner.environment.fit_partition_sha256,
        "fixed_comparator": dict(comparator_binding),
        "latest_checkpoint_file_sha256": _sha256_file(latest_path),
        "output_schema": OUTPUT_SCHEMA,
        "phase_label": PHASE_LABEL,
        "record": REPORT_SCHEMA,
        "runner_binding_sha256": runner.runner_binding_sha256,
        "sampling_split": runner.environment.sampling_split,
        "seed": runner.seed,
        "status": "COMPLETE" if complete else "IN_PROGRESS",
        "summary": runner.summary().to_canonical_dict(),
        "target_rule": TARGET_RULE,
        "training_population": "REGISTERED_TRAIN_IDS_ONLY",
        "update_zero_checkpoint_sha256": (
            load_exact_p95_run2_checkpoint_v2(update_zero_path).checkpoint_sha256
        ),
        "validation_access_during_training": "FORBIDDEN",
    }
    return _content_attested(document, field_name="report_content_sha256")


def _require_checkpoint_identity(
    checkpoint: ExactP95Run2CheckpointV2,
    *,
    config: ExactP95Run2RunnerConfigV2,
    seed: int,
    expected_update: Optional[int] = None,
) -> None:
    if checkpoint.config != config or checkpoint.seed != seed:
        raise Run2V2ArtifactError("checkpoint schedule/seed mismatch")
    if expected_update is not None and checkpoint.update_count != expected_update:
        raise Run2V2ArtifactError("checkpoint filename/update mismatch")


def _require_update_zero(checkpoint: ExactP95Run2CheckpointV2) -> None:
    if (
        checkpoint.update_count != 0
        or checkpoint.collection_seq != 0
        or checkpoint.warmup_collected != 0
        or checkpoint.post_warmup_collected != 0
        or checkpoint.environment_state.d1_state.reset_count != 0
        or checkpoint.d1_transition_history
        or checkpoint.run2_v2_transition_history
        or checkpoint.metrics
        or checkpoint.replay_binding is not None
        or checkpoint.trainer_initialized
        or checkpoint.actor_optimizer_state.get("state") != {}
        or checkpoint.critic_optimizer_state.get("state") != {}
    ):
        raise Run2V2ArtifactError(
            "checkpoint_000000.pt is not pre-collection initialized state"
        )


def _validate_csv_shape(
    path: Path, *, expected_fields: Tuple[str, ...], expected_rows: int
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
    except Exception as exc:
        raise Run2V2ArtifactError(f"malformed CSV artifact {path.name}") from exc
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise Run2V2ArtifactError(f"{path.name} header drift")
    if len(rows) != expected_rows:
        raise Run2V2ArtifactError(f"{path.name} row-count drift")


def _preflight_existing_output(
    *,
    output_directory: Path,
    config: ExactP95Run2RunnerConfigV2,
    seed: int,
    checkpoint_interval_updates: int,
    comparator_binding: Mapping[str, Any],
) -> Tuple[ExactP95Run2CheckpointV2, ExactP95Run2CheckpointV2]:
    observed = {path.name for path in output_directory.iterdir()}
    if observed != _ROOT_INVENTORY:
        raise Run2V2ArtifactError("reusable output inventory is incomplete or foreign")
    expected_config = _config_document(
        config, seed, checkpoint_interval_updates, comparator_binding
    )
    supplied_config = _read_json(output_directory / "config.json")
    if supplied_config.get("record") != CONFIG_SCHEMA:
        raise Run2V2ArtifactError("output config is not exact Run-2 v2")
    _require_content_attestation(
        supplied_config, field_name="config_content_sha256"
    )
    if supplied_config != expected_config:
        raise Run2V2ArtifactError("output directory config identity mismatch")

    zero = load_exact_p95_run2_checkpoint_v2(
        output_directory / "checkpoint_000000.pt"
    )
    latest = load_exact_p95_run2_checkpoint_v2(
        output_directory / "checkpoint_latest.pt"
    )
    _require_checkpoint_identity(zero, config=config, seed=seed, expected_update=0)
    _require_checkpoint_identity(latest, config=config, seed=seed)
    _require_update_zero(zero)
    if (
        zero.runner_binding_sha256 != latest.runner_binding_sha256
        or zero.collection_session_uuid != latest.collection_session_uuid
    ):
        raise Run2V2ArtifactError("update-zero/latest identity mismatch")

    numbered = _numbered_checkpoint_paths(output_directory)
    expected_updates = _expected_numbered_updates(
        latest_update=latest.update_count,
        configured_updates=config.update_count,
        interval=checkpoint_interval_updates,
    )
    if tuple(sorted(numbered)) != expected_updates:
        raise Run2V2ArtifactError("numbered checkpoint schedule drift")
    for update, path in sorted(numbered.items()):
        candidate = load_exact_p95_run2_checkpoint_v2(path)
        _require_checkpoint_identity(
            candidate, config=config, seed=seed, expected_update=update
        )
        if (
            candidate.runner_binding_sha256 != latest.runner_binding_sha256
            or candidate.collection_session_uuid
            != latest.collection_session_uuid
        ):
            raise Run2V2ArtifactError("numbered checkpoint identity drift")
    if expected_updates:
        last = load_exact_p95_run2_checkpoint_v2(numbered[expected_updates[-1]])
        if last.checkpoint_sha256 != latest.checkpoint_sha256:
            raise Run2V2ArtifactError("latest checkpoint differs from series tail")
    elif latest.update_count != 0 or latest.checkpoint_sha256 != zero.checkpoint_sha256:
        raise Run2V2ArtifactError("zero-update latest checkpoint drift")

    bindings = _read_json(output_directory / "bindings.json")
    if bindings.get("record") != BINDINGS_SCHEMA:
        raise Run2V2ArtifactError("output bindings are not exact Run-2 v2")
    _require_content_attestation(
        bindings, field_name="bindings_content_sha256"
    )
    if (
        bindings.get("phase_label") != PHASE_LABEL
        or bindings.get("seed") != seed
        or bindings.get("runner_binding_sha256")
        != latest.runner_binding_sha256
        or bindings.get("collection_session_uuid")
        != latest.collection_session_uuid
        or bindings.get("sampling_split") != "train"
        or bindings.get("fixed_comparator") != comparator_binding
    ):
        raise Run2V2ArtifactError("output binding identity mismatch")
    report = _read_json(output_directory / "report.json")
    if report.get("record") != REPORT_SCHEMA:
        raise Run2V2ArtifactError("output report is not exact Run-2 v2")
    _require_content_attestation(report, field_name="report_content_sha256")
    if (
        report.get("checkpoint_sha256") != latest.checkpoint_sha256
        or report.get("completed_updates") != latest.update_count
        or report.get("seed") != seed
        or report.get("phase_label") != PHASE_LABEL
    ):
        raise Run2V2ArtifactError("output report/checkpoint identity mismatch")
    expected_artifact_paths = {
        "bindings.json": output_directory / "bindings.json",
        "checkpoint_000000.pt": output_directory / "checkpoint_000000.pt",
        "checkpoint_latest.pt": output_directory / "checkpoint_latest.pt",
        "config.json": output_directory / "config.json",
        "metrics.csv": output_directory / "metrics.csv",
        "transition_reward_audit.csv": (
            output_directory / "transition_reward_audit.csv"
        ),
    }
    reported_artifact_hashes = report.get("artifact_file_sha256")
    if type(reported_artifact_hashes) is not dict or set(
        reported_artifact_hashes
    ) != set(expected_artifact_paths):
        raise Run2V2ArtifactError("report physical artifact inventory drift")
    for name, path in expected_artifact_paths.items():
        if reported_artifact_hashes.get(name) != _sha256_file(path):
            raise Run2V2ArtifactError(
                f"physical artifact hash mismatch at {name}"
            )
    reported_series = report.get("checkpoint_series")
    if type(reported_series) is not dict or set(reported_series) != {
        f"{update:06d}" for update in numbered
    }:
        raise Run2V2ArtifactError("report checkpoint-series inventory drift")
    for update, path in numbered.items():
        expected_entry = {
            "file_sha256": _sha256_file(path),
            "path": f"checkpoints/checkpoint_{update:06d}.pt",
        }
        if reported_series.get(f"{update:06d}") != expected_entry:
            raise Run2V2ArtifactError(
                f"physical checkpoint-series hash mismatch at update {update}"
            )
    if report.get("latest_checkpoint_file_sha256") != _sha256_file(
        output_directory / "checkpoint_latest.pt"
    ):
        raise Run2V2ArtifactError("latest physical checkpoint hash mismatch")
    _validate_csv_shape(
        output_directory / "metrics.csv",
        expected_fields=_METRIC_FIELDS,
        expected_rows=latest.update_count,
    )
    _validate_csv_shape(
        output_directory / "transition_reward_audit.csv",
        expected_fields=_AUDIT_FIELDS,
        expected_rows=latest.collection_seq,
    )
    return zero, latest


def _validate_existing_bytes_against_runner(
    *,
    output_directory: Path,
    runner: ExactP95Run2RunnerV2,
    checkpoint_interval_updates: int,
    latest: ExactP95Run2CheckpointV2,
    comparator_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = {
        "config.json": _json_bytes(
            _config_document(
                runner.config,
                runner.seed,
                checkpoint_interval_updates,
                comparator_binding,
            )
        ),
        "bindings.json": _json_bytes(
            _bindings_document(runner, comparator_binding)
        ),
        "metrics.csv": _metrics_csv(runner.metrics),
        "transition_reward_audit.csv": _reward_audit_csv(runner),
    }
    for name, payload in expected.items():
        if (output_directory / name).read_bytes() != payload:
            raise Run2V2ArtifactError(
                f"existing {name} does not reconstruct from checkpoint"
            )
    expected_report = _report_document(
        output_directory=output_directory,
        runner=runner,
        checkpoint_interval_updates=checkpoint_interval_updates,
        checkpoint=latest,
        comparator_binding=comparator_binding,
    )
    if (output_directory / "report.json").read_bytes() != _json_bytes(
        expected_report
    ):
        raise Run2V2ArtifactError(
            "existing report.json does not reconstruct from checkpoint"
        )
    return expected_report


def _emit_snapshot(
    *,
    output_directory: Path,
    runner: ExactP95Run2RunnerV2,
    checkpoint_interval_updates: int,
    initial: bool,
    comparator_binding: Mapping[str, Any],
) -> Dict[str, Any]:
    checkpoint = runner.checkpoint()
    update = checkpoint.update_count
    if initial is not (update == 0):
        raise Run2V2ArtifactError("initial snapshot/update mismatch")
    checkpoint_payload = _checkpoint_bytes(checkpoint)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_path = output_directory / "config.json"
    if initial:
        if config_path.exists() or (output_directory / "checkpoint_000000.pt").exists():
            raise Run2V2ArtifactError("refusing to overwrite update-zero identity")
        _atomic_json(
            config_path,
            _config_document(
                runner.config,
                runner.seed,
                checkpoint_interval_updates,
                comparator_binding,
            ),
        )
        _atomic_durable_bytes(
            output_directory / "checkpoint_000000.pt", checkpoint_payload
        )
    else:
        numbered = (
            output_directory
            / "checkpoints"
            / f"checkpoint_{update:06d}.pt"
        )
        if numbered.exists():
            raise Run2V2ArtifactError(
                "refusing to overwrite an existing numbered checkpoint"
            )
        _atomic_durable_bytes(numbered, checkpoint_payload)
    _atomic_durable_bytes(
        output_directory / "checkpoint_latest.pt", checkpoint_payload
    )
    _atomic_json(
        output_directory / "bindings.json",
        _bindings_document(runner, comparator_binding),
    )
    _atomic_durable_bytes(
        output_directory / "metrics.csv", _metrics_csv(runner.metrics)
    )
    _atomic_durable_bytes(
        output_directory / "transition_reward_audit.csv",
        _reward_audit_csv(runner),
    )
    report = _report_document(
        output_directory=output_directory,
        runner=runner,
        checkpoint_interval_updates=checkpoint_interval_updates,
        checkpoint=checkpoint,
        comparator_binding=comparator_binding,
    )
    _atomic_json(output_directory / "report.json", report)
    return report


def run_seed_to_directory(
    *,
    output_directory: Path,
    seed: int,
    config: ExactP95Run2RunnerConfigV2 = REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
    checkpoint_interval_updates: int = (
        EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2
    ),
    stop_after_updates: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run or exactly resume one seed into a closed v2 artifact directory.

    ``stop_after_updates`` and non-registered checkpoint cadences are exposed
    solely for focused checkpoint/resume tests.  A test stop must itself be a
    checkpoint boundary so interrupted/resumed output equals uninterrupted
    output.
    """
    if type(config) is not ExactP95Run2RunnerConfigV2:
        raise Run2V2ArtifactError("config has a foreign type")
    config.__post_init__()
    if type(seed) is not int or seed not in config.seeds:
        raise Run2V2ArtifactError("seed is not configured")
    if (
        type(checkpoint_interval_updates) is not int
        or checkpoint_interval_updates < 1
    ):
        raise Run2V2ArtifactError("checkpoint interval must be positive")
    target = config.update_count if stop_after_updates is None else stop_after_updates
    if type(target) is not int or not 0 <= target <= config.update_count:
        raise Run2V2ArtifactError("stop_after_updates is outside the schedule")
    if (
        target not in (0, config.update_count)
        and target % checkpoint_interval_updates != 0
    ):
        raise Run2V2ArtifactError(
            "test stop must coincide with a checkpoint boundary"
        )
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and not output_directory.is_dir():
        raise Run2V2ArtifactError("output path is not a directory")
    # This gate precedes runner construction and every output mutation.
    comparator_binding = _fixed_comparator_binding(project_root)
    existing = output_directory.exists() and any(output_directory.iterdir())
    zero_checkpoint: Optional[ExactP95Run2CheckpointV2] = None
    latest_checkpoint: Optional[ExactP95Run2CheckpointV2] = None
    if existing:
        zero_checkpoint, latest_checkpoint = _preflight_existing_output(
            output_directory=output_directory,
            config=config,
            seed=seed,
            checkpoint_interval_updates=checkpoint_interval_updates,
            comparator_binding=comparator_binding,
        )

    try:
        with ExactP95Run2RunnerV2(
            seed=seed, config=config, project_root=project_root
        ) as runner:
            runner.require_registered_train_binding()
            pristine = runner.checkpoint()
            if existing:
                assert zero_checkpoint is not None
                assert latest_checkpoint is not None
                if pristine.checkpoint_sha256 != zero_checkpoint.checkpoint_sha256:
                    raise Run2V2ArtifactError(
                        "physical update-zero checkpoint differs from initialization"
                    )
                runner.load_checkpoint(latest_checkpoint)
                _validate_existing_bytes_against_runner(
                    output_directory=output_directory,
                    runner=runner,
                    checkpoint_interval_updates=checkpoint_interval_updates,
                    latest=latest_checkpoint,
                    comparator_binding=comparator_binding,
                )
            else:
                # This is the only path that creates output.  The exact
                # initialized checkpoint is durable before run_until_updates
                # is ever called, hence before the first environment reset.
                _emit_snapshot(
                    output_directory=output_directory,
                    runner=runner,
                    checkpoint_interval_updates=checkpoint_interval_updates,
                    initial=True,
                    comparator_binding=comparator_binding,
                )
            if runner.completed_updates > target:
                raise Run2V2ArtifactError(
                    "latest checkpoint is ahead of requested target"
                )
            while runner.completed_updates < target:
                next_boundary = min(
                    target,
                    config.update_count,
                    (
                        (runner.completed_updates // checkpoint_interval_updates)
                        + 1
                    )
                    * checkpoint_interval_updates,
                )
                runner.run_until_updates(next_boundary)
                _emit_snapshot(
                    output_directory=output_directory,
                    runner=runner,
                    checkpoint_interval_updates=checkpoint_interval_updates,
                    initial=False,
                    comparator_binding=comparator_binding,
                )
            return _read_json(output_directory / "report.json")
    except ExactP95Run2RunnerErrorV2 as exc:
        raise Run2V2ArtifactError("Run-2-v2 runner failed") from exc


def _campaign_document(
    *,
    reports: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    comparator_binding: Mapping[str, Any],
    config: ExactP95Run2RunnerConfigV2,
) -> Dict[str, Any]:
    complete = all(report.get("status") == "COMPLETE" for report in reports)
    document = {
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "config_sha256": config.canonical_sha256(),
        "fixed_comparator": dict(comparator_binding),
        "output_schema": OUTPUT_SCHEMA,
        "phase_label": PHASE_LABEL,
        "record": CAMPAIGN_SCHEMA,
        "reports": list(reports),
        "seeds": list(seeds),
        "status": "COMPLETE" if complete else "IN_PROGRESS",
        "target_rule": TARGET_RULE,
        "training_population": "REGISTERED_TRAIN_IDS_ONLY",
        "validation_access_during_training": "FORBIDDEN",
    }
    return _content_attested(
        document, field_name="campaign_report_content_sha256"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/resume exact-P95 Run-2-v2 train-only Hybrid-SAC."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        choices=REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds,
    )
    return parser.parse_args()


def _read_campaign_seed_report(
    *,
    output: Path,
    seed: int,
    comparator_binding: Mapping[str, Any],
    expected_config: ExactP95Run2RunnerConfigV2,
) -> Dict[str, Any]:
    directory = output / f"seed_{seed}"
    if not directory.is_dir():
        raise Run2V2ArtifactError(f"campaign seed_{seed} directory is missing")
    config = _read_json(directory / "config.json")
    _require_content_attestation(config, field_name="config_content_sha256")
    if (
        config.get("record") != CONFIG_SCHEMA
        or config.get("seed") != seed
        or config.get("config_sha256")
        != expected_config.canonical_sha256()
        or config.get("fixed_comparator") != comparator_binding
    ):
        raise Run2V2ArtifactError(f"campaign seed_{seed} config drift")
    report = _read_json(directory / "report.json")
    _require_content_attestation(report, field_name="report_content_sha256")
    if (
        report.get("record") != REPORT_SCHEMA
        or report.get("output_schema") != OUTPUT_SCHEMA
        or report.get("phase_label") != PHASE_LABEL
        or report.get("seed") != seed
        or report.get("configured_updates")
        != expected_config.update_count
        or report.get("fixed_comparator") != comparator_binding
    ):
        raise Run2V2ArtifactError(f"campaign seed_{seed} report drift")
    return report


def _campaign_report_can_advance(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    return (
        prior.get("status") == "IN_PROGRESS"
        and prior.get("seed") == current.get("seed")
        and prior.get("phase_label") == current.get("phase_label")
        and prior.get("runner_binding_sha256")
        == current.get("runner_binding_sha256")
        and type(prior.get("completed_updates")) is int
        and type(current.get("completed_updates")) is int
        and current["completed_updates"] > prior["completed_updates"]
    )


def _run_campaign_to_directory(
    *,
    output: Path,
    seeds: Sequence[int],
    comparator_binding: Optional[Mapping[str, Any]] = None,
    seed_runner=None,
    config: ExactP95Run2RunnerConfigV2 = REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
    checkpoint_interval_updates: int = (
        EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2
    ),
) -> Dict[str, Any]:
    """Run requested seeds while preserving every prior campaign seed."""
    requested = tuple(seeds)
    if type(config) is not ExactP95Run2RunnerConfigV2:
        raise Run2V2ArtifactError("campaign config has a foreign type")
    config.__post_init__()
    if (
        type(checkpoint_interval_updates) is not int
        or checkpoint_interval_updates < 1
    ):
        raise Run2V2ArtifactError("campaign checkpoint interval is invalid")
    registered = config.seeds
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(type(seed) is not int or seed not in registered for seed in requested)
    ):
        raise Run2V2ArtifactError("campaign seeds must be unique and registered")
    output = Path(output).resolve()
    if output.exists() and not output.is_dir():
        raise Run2V2ArtifactError("campaign output is not a directory")
    comparator = (
        _fixed_comparator_binding(None)
        if comparator_binding is None
        else dict(comparator_binding)
    )
    existing_reports: Dict[int, Dict[str, Any]] = {}
    if output.exists():
        allowed = {
            "campaign_report.json",
            *(f"seed_{seed}" for seed in registered),
        }
        unexpected = {path.name for path in output.iterdir()} - allowed
        if unexpected:
            raise Run2V2ArtifactError(
                f"foreign campaign output entries: {sorted(unexpected)}"
            )
        campaign_path = output / "campaign_report.json"
        prior_by_seed: Dict[int, Dict[str, Any]] = {}
        if campaign_path.exists():
            campaign = _read_json(campaign_path)
            if (
                campaign.get("record") != CAMPAIGN_SCHEMA
                or campaign.get("phase_label") != PHASE_LABEL
                or campaign.get("config_sha256")
                != config.canonical_sha256()
                or campaign.get("fixed_comparator") != comparator
            ):
                raise Run2V2ArtifactError("foreign campaign report")
            _require_content_attestation(
                campaign, field_name="campaign_report_content_sha256"
            )
            campaign_seeds = campaign.get("seeds")
            campaign_reports = campaign.get("reports")
            if (
                type(campaign_seeds) is not list
                or type(campaign_reports) is not list
                or len(campaign_seeds) != len(campaign_reports)
                or any(type(seed) is not int for seed in campaign_seeds)
                or len(set(campaign_seeds)) != len(campaign_seeds)
                or any(seed not in registered for seed in campaign_seeds)
                or any(
                    type(report) is not dict
                    or report.get("seed") != seed
                    for seed, report in zip(campaign_seeds, campaign_reports)
                )
            ):
                raise Run2V2ArtifactError("campaign seed/report inventory drift")
            prior_by_seed = dict(zip(campaign_seeds, campaign_reports))
        for seed in registered:
            directory = output / f"seed_{seed}"
            if directory.exists():
                current = _read_campaign_seed_report(
                    output=output,
                    seed=seed,
                    comparator_binding=comparator,
                    expected_config=config,
                )
                _preflight_existing_output(
                    output_directory=directory,
                    config=config,
                    seed=seed,
                    checkpoint_interval_updates=checkpoint_interval_updates,
                    comparator_binding=comparator,
                )
                prior = prior_by_seed.get(seed)
                if prior is not None and prior != current and not (
                    _campaign_report_can_advance(prior, current)
                ):
                    raise Run2V2ArtifactError(
                        f"campaign seed_{seed} report changed incompatibly"
                    )
                existing_reports[seed] = current
            elif seed in prior_by_seed:
                raise Run2V2ArtifactError(
                    f"campaign report references missing seed_{seed}"
                )
        if set(prior_by_seed) - set(existing_reports):
            raise Run2V2ArtifactError("campaign prior seed inventory is incomplete")

    invoke = run_seed_to_directory if seed_runner is None else seed_runner
    for seed in requested:
        reported = invoke(
            output_directory=output / f"seed_{seed}",
            seed=seed,
        )
        materialized = _read_campaign_seed_report(
            output=output,
            seed=seed,
            comparator_binding=comparator,
            expected_config=config,
        )
        if reported != materialized:
            raise Run2V2ArtifactError(
                f"seed_{seed} returned report differs from durable report"
            )
        existing_reports[seed] = materialized
    merged_seeds = tuple(seed for seed in registered if seed in existing_reports)
    reports = [existing_reports[seed] for seed in merged_seeds]
    campaign = _campaign_document(
        reports=reports,
        seeds=merged_seeds,
        comparator_binding=comparator,
        config=config,
    )
    _atomic_json(output / "campaign_report.json", campaign)
    return campaign


def main() -> int:
    args = _parse_args()
    seeds: Tuple[int, ...] = (
        tuple(args.seed)
        if args.seed
        else REGISTERED_EXACT_P95_RUN2_CONFIG_V2.seeds
    )
    campaign = _run_campaign_to_directory(output=args.output, seeds=seeds)
    output = args.output.resolve()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
