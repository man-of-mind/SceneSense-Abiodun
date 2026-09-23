"""Durable CLI for the bounded three-seed Run-3 training campaign.

Artifacts are create-only.  Full resumable checkpoints exist only at update
0, the matched Run-2 horizon (5,000), and the fixed primary endpoint (10,000).
The ``latest_checkpoint.json`` file is an atomic pointer to one of those files,
not a duplicate checkpoint.  Model-only snapshots are evaluation inputs and
are explicitly non-authoritative for resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from .empirical_contextual_run3_runner import (
    RUN3_REGISTERED_CONFIG,
    Run3CheckpointV1,
    Run3RunnerConfigV1,
    Run3RunnerError,
    Run3TrainingRunnerV1,
)
from .transaction_identity import canonical_json_bytes, canonical_sha256

__all__ = [
    "PROJECTED_BYTES_PER_SEED",
    "REQUIRED_FREE_RESERVE_BYTES",
    "load_run3_checkpoint",
    "run_campaign",
    "run_seed_to_directory",
]


PROJECTED_BYTES_PER_SEED = 3 * 1024**3
REQUIRED_FREE_RESERVE_BYTES = 10 * 1024**3
_MANIFEST_SCHEMA = "splitfusion.run3_training_artifact_manifest.v1"
_TERMINAL_SCHEMA = "splitfusion.run3_training_terminal.v1"


class Run3ArtifactError(RuntimeError):
    """Durable output, disk, or resume validation failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise Run3ArtifactError(f"refusing to overwrite {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_bytes(document: Dict[str, Any]) -> bytes:
    return canonical_json_bytes(document) + b"\n"


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _checkpoint_path(directory: Path, update: int) -> Path:
    return directory / "checkpoints" / f"full_{update:06d}.pt"


def _snapshot_path(directory: Path, update: int) -> Path:
    return directory / "model_snapshots" / f"model_{update:06d}.pt"


def load_run3_checkpoint(path: Path) -> Run3CheckpointV1:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Run3ArtifactError(f"could not load {path}") from exc
    if type(value) is not Run3CheckpointV1:
        raise Run3ArtifactError("checkpoint file contains a foreign object")
    value.require_valid()
    return value


def _write_full_checkpoint(directory: Path, runner: Run3TrainingRunnerV1) -> Path:
    update = runner.completed_updates
    if update not in runner.config.full_checkpoint_updates:
        raise Run3ArtifactError("full checkpoint requested off registered cadence")
    path = _checkpoint_path(directory, update)
    checkpoint = runner.checkpoint()
    _atomic_bytes(path, _torch_bytes(checkpoint), replace=False)
    pointer = {
        "checkpoint_content_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_file_sha256": _sha256_file(path),
        "path": str(path.relative_to(directory)),
        "record": "run3_latest_full_checkpoint_pointer_v1",
        "update": update,
    }
    pointer["pointer_sha256"] = canonical_sha256(pointer)
    _atomic_bytes(directory / "latest_checkpoint.json", _json_bytes(pointer))
    return path


def _quarantine_artifact(directory: Path, path: Path) -> None:
    """Move a non-authoritative artifact aside without discarding its bytes."""
    quarantine = directory / "orphaned_non_authoritative"
    quarantine.mkdir(parents=True, exist_ok=True)
    digest = _sha256_file(path)
    destination = quarantine / f"{path.parent.name}__{path.name}__{digest}"
    if destination.exists():
        if _sha256_file(destination) != digest:
            raise Run3ArtifactError("orphan quarantine hash collision")
        path.unlink()
    else:
        os.replace(path, destination)


def _bootstrap_checkpoint_pointer(
    directory: Path,
    runner: Run3TrainingRunnerV1,
) -> None:
    """Recover the create-seed crash window before the first pointer.

    The seed manifest is already authoritative at this point.  A matching
    update-zero checkpoint may have reached durable storage before the crash;
    it is verified and reused.  Any conflicting or later artifact is retained
    in quarantine, after which the deterministic update-zero checkpoint and
    pointer are completed.
    """
    if runner.completed_updates != 0:
        raise Run3ArtifactError("bootstrap recovery requires an update-zero runner")
    expected = runner.checkpoint()
    path = _checkpoint_path(directory, 0)
    if path.exists():
        matches = False
        try:
            found = load_run3_checkpoint(path)
            matches = found.checkpoint_sha256 == expected.checkpoint_sha256
        except Run3ArtifactError:
            matches = False
        if not matches:
            _quarantine_artifact(directory, path)
    _discard_non_authoritative_tail(directory, -1, preserve_update_zero=path.exists())
    if not path.exists():
        _atomic_bytes(path, _torch_bytes(expected), replace=False)
    pointer = {
        "checkpoint_content_sha256": expected.checkpoint_sha256,
        "checkpoint_file_sha256": _sha256_file(path),
        "path": str(path.relative_to(directory)),
        "record": "run3_latest_full_checkpoint_pointer_v1",
        "update": 0,
    }
    pointer["pointer_sha256"] = canonical_sha256(pointer)
    _atomic_bytes(directory / "latest_checkpoint.json", _json_bytes(pointer), replace=False)


def _write_model_snapshot(directory: Path, runner: Run3TrainingRunnerV1) -> None:
    update = runner.completed_updates
    path = _snapshot_path(directory, update)
    _atomic_bytes(path, _torch_bytes(runner.model_only_snapshot()), replace=False)


def _metrics_bytes(metrics: Iterable[Any]) -> bytes:
    rows = []
    for metric in metrics:
        row = metric.as_dict()
        for name, value in tuple(row.items()):
            if isinstance(value, tuple):
                row[name] = json.dumps(value, separators=(",", ":"))
        rows.append(row)
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_progress(directory: Path, runner: Run3TrainingRunnerV1) -> None:
    _atomic_bytes(directory / "metrics.csv", _metrics_bytes(runner.metrics))
    progress = {
        "completed_updates": runner.completed_updates,
        "configured_updates": runner.config.update_count,
        "latest_full_checkpoint_update": max(
            value for value in runner.config.full_checkpoint_updates
            if value <= runner.completed_updates
        ),
        "record": "run3_training_progress_v1",
        "seed": runner.seed,
        "transition_count": len(runner.transitions),
    }
    progress["progress_sha256"] = canonical_sha256(progress)
    _atomic_bytes(directory / "progress.json", _json_bytes(progress))


def _resume_checkpoint(directory: Path) -> Run3CheckpointV1:
    pointer_path = directory / "latest_checkpoint.json"
    document = json.loads(pointer_path.read_text(encoding="utf-8"))
    supplied = document.pop("pointer_sha256")
    if supplied != canonical_sha256(document):
        raise Run3ArtifactError("latest checkpoint pointer digest drift")
    path = directory / document["path"]
    if _sha256_file(path) != document["checkpoint_file_sha256"]:
        raise Run3ArtifactError("latest checkpoint file hash drift")
    checkpoint = load_run3_checkpoint(path)
    if checkpoint.checkpoint_sha256 != document["checkpoint_content_sha256"]:
        raise Run3ArtifactError("latest checkpoint content hash drift")
    return checkpoint


def _discard_non_authoritative_tail(
    directory: Path,
    update: int,
    *,
    preserve_update_zero: bool = False,
) -> None:
    """Quarantine artifacts newer than the authoritative pointer.

    They are not resume authorities, but retaining their bytes makes crash
    recovery auditable and avoids silently overwriting conflicting evidence.
    """
    snapshot_directory = directory / "model_snapshots"
    if snapshot_directory.exists():
        for path in snapshot_directory.glob("model_*.pt"):
            try:
                candidate = int(path.stem.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise Run3ArtifactError(f"unexpected snapshot name {path.name}") from exc
            if candidate > update:
                _quarantine_artifact(directory, path)
    checkpoint_directory = directory / "checkpoints"
    if checkpoint_directory.exists():
        for path in checkpoint_directory.glob("full_*.pt"):
            try:
                candidate = int(path.stem.split("_")[1])
            except (IndexError, ValueError) as exc:
                raise Run3ArtifactError(f"unexpected checkpoint name {path.name}") from exc
            if candidate > update and not (preserve_update_zero and candidate == 0):
                _quarantine_artifact(directory, path)


def _disk_gate(
    path: Path,
    seed_count: int = 0,
    *,
    projected_bytes: Optional[int] = None,
) -> Dict[str, int]:
    # A fresh campaign may name several not-yet-created directories.  Query the
    # nearest existing ancestor instead of assuming the immediate parent exists.
    usage_path = Path(path)
    while not usage_path.exists():
        parent = usage_path.parent
        if parent == usage_path:
            raise Run3ArtifactError(
                f"could not find an existing ancestor for disk preflight: {path}"
            )
        usage_path = parent
    usage = shutil.disk_usage(usage_path)
    projected = (
        PROJECTED_BYTES_PER_SEED * seed_count
        if projected_bytes is None
        else projected_bytes
    )
    if type(projected) is not int or projected < 0:
        raise Run3ArtifactError("projected byte count is invalid")
    if usage.free < REQUIRED_FREE_RESERVE_BYTES:
        raise Run3ArtifactError("less than 10 GiB is free before training")
    if projected > usage.free - REQUIRED_FREE_RESERVE_BYTES:
        raise Run3ArtifactError("projected artifacts would violate 10 GiB reserve")
    return {
        "free_bytes_at_preflight": usage.free,
        "projected_bytes": projected,
        "required_reserve_bytes": REQUIRED_FREE_RESERVE_BYTES,
    }


def _remaining_campaign_projection(output_root: Path, seeds: Sequence[int]) -> int:
    """Charge one full projection per unfinished seed.

    Existing incomplete bytes may be orphaned or may be replaced during
    deterministic recovery, so they are deliberately not treated as credit.
    Completed seeds are verified separately before this estimate is used.
    """
    projected = 0
    for seed in seeds:
        directory = output_root / f"seed_{seed}"
        if (directory / "RUN3_TRAINING_COMPLETE.json").is_file():
            continue
        projected += PROJECTED_BYTES_PER_SEED
    return projected


def _require_completed_report(
    report: Mapping[str, Any],
    *,
    seed: int,
    config: Run3RunnerConfigV1,
) -> None:
    summary = report.get("summary")
    if type(summary) is not dict:
        raise Run3ArtifactError("completed seed report lacks a summary")
    required = {
        "seed": seed,
        "configured_updates": config.update_count,
        "completed_updates": config.update_count,
        "transition_count": config.total_transitions,
        "replay_resident_count": config.total_transitions,
        "replay_eviction_count": 0,
        "excluded_fault_count": 0,
        "completed_training_hard_gates_passed": True,
    }
    if any(summary.get(name) != value for name, value in required.items()):
        raise Run3ArtifactError("completed seed hard-gate summary drift")
    if (
        report.get("fixed_endpoint_update") != config.update_count
        or report.get("matched_horizon_update") != config.matched_horizon_update
        or report.get("peak_checkpoint_selection") is not False
    ):
        raise Run3ArtifactError("completed seed endpoint semantics drift")


def _verified_completed_seed_report(
    seed_directory: Path,
    *,
    seed: int,
    config: Run3RunnerConfigV1,
) -> Optional[Dict[str, Any]]:
    terminal_path = seed_directory / "RUN3_TRAINING_COMPLETE.json"
    if not terminal_path.exists():
        return None
    try:
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Run3ArtifactError(f"seed {seed} terminal is unreadable") from exc
    terminal_digest = terminal.pop("terminal_sha256", None)
    if terminal_digest != canonical_sha256(terminal):
        raise Run3ArtifactError(f"seed {seed} terminal marker drift")
    if (
        terminal.get("schema") != _TERMINAL_SCHEMA
        or terminal.get("status") != "RUN3_FIXED_ENDPOINT_COMPLETE"
        or terminal.get("seed") != seed
        or terminal.get("update") != config.update_count
    ):
        raise Run3ArtifactError(f"seed {seed} terminal semantics drift")
    report_path = seed_directory / "report.json"
    if not report_path.is_file() or _sha256_file(report_path) != terminal.get(
        "report_file_sha256"
    ):
        raise Run3ArtifactError(f"seed {seed} report file drift")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Run3ArtifactError(f"seed {seed} report is unreadable") from exc
    if report.get("report_sha256") != terminal.get("report_sha256"):
        raise Run3ArtifactError(f"seed {seed} report identity drift")
    report_copy = dict(report)
    reported_digest = report_copy.pop("report_sha256", None)
    if reported_digest != canonical_sha256(report_copy):
        raise Run3ArtifactError(f"seed {seed} report canonical digest drift")
    _require_completed_report(report, seed=seed, config=config)
    hashes = report.get("artifact_hashes")
    if type(hashes) is not dict:
        raise Run3ArtifactError(f"seed {seed} artifact inventory missing")
    actual = {
        str(path.relative_to(seed_directory)): _sha256_file(path)
        for path in sorted(seed_directory.rglob("*"))
        if path.is_file()
        and path.name not in ("report.json", "RUN3_TRAINING_COMPLETE.json")
    }
    if hashes != actual:
        raise Run3ArtifactError(f"seed {seed} artifact inventory drift")
    return report


def _validate_seed_manifest(
    directory: Path,
    *,
    seed: int,
    config: Run3RunnerConfigV1,
) -> None:
    path = directory / "manifest.json"
    if not path.is_file():
        raise Run3ArtifactError("seed resume manifest is missing")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Run3ArtifactError("seed resume manifest is unreadable") from exc
    supplied = document.pop("manifest_sha256", None)
    if supplied != canonical_sha256(document):
        raise Run3ArtifactError("seed resume manifest digest drift")
    expected = {
        "config": config.to_canonical_dict(),
        "config_sha256": config.canonical_sha256(),
        "record": _MANIFEST_SCHEMA,
        "seed": seed,
    }
    if document != expected:
        raise Run3ArtifactError("seed resume manifest binding drift")


def run_seed_to_directory(
    directory: Path,
    *,
    seed: int,
    config: Run3RunnerConfigV1 = RUN3_REGISTERED_CONFIG,
    project_root: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    directory = Path(directory)
    if resume:
        if not directory.is_dir():
            raise Run3ArtifactError("resume directory does not exist")
    else:
        if directory.exists():
            raise Run3ArtifactError("output directory already exists")
        directory.mkdir(parents=True)
    runner = Run3TrainingRunnerV1(seed=seed, config=config, project_root=project_root)
    try:
        if resume:
            _validate_seed_manifest(directory, seed=seed, config=config)
            if not (directory / "latest_checkpoint.json").exists():
                _bootstrap_checkpoint_pointer(directory, runner)
            checkpoint = _resume_checkpoint(directory)
            runner.load_checkpoint(checkpoint)
            _discard_non_authoritative_tail(directory, checkpoint.update_count)
            _write_progress(directory, runner)
        else:
            manifest = {
                "config": config.to_canonical_dict(),
                "config_sha256": config.canonical_sha256(),
                "record": _MANIFEST_SCHEMA,
                "seed": seed,
            }
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            _atomic_bytes(directory / "manifest.json", _json_bytes(manifest), replace=False)
            _write_full_checkpoint(directory, runner)
            _write_progress(directory, runner)

        while runner.completed_updates < config.update_count:
            next_snapshot = min(
                config.update_count,
                ((runner.completed_updates // config.model_snapshot_interval) + 1)
                * config.model_snapshot_interval,
            )
            next_full = min(
                (value for value in config.full_checkpoint_updates if value > runner.completed_updates),
                default=config.update_count,
            )
            target = min(next_snapshot, next_full, config.update_count)
            runner.run_until_updates(target)
            if target % config.model_snapshot_interval == 0:
                _write_model_snapshot(directory, runner)
            if target in config.full_checkpoint_updates:
                _write_full_checkpoint(directory, runner)
            _write_progress(directory, runner)

        summary = asdict(runner.summary())
        report = {
            "artifact_hashes": {},
            "fixed_endpoint_update": config.update_count,
            "matched_horizon_update": config.matched_horizon_update,
            "peak_checkpoint_selection": False,
            "record": "run3_training_report_v1",
            "summary": summary,
        }
        _require_completed_report(report, seed=seed, config=config)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name not in ("report.json", "RUN3_TRAINING_COMPLETE.json"):
                report["artifact_hashes"][str(path.relative_to(directory))] = _sha256_file(path)
        report["report_sha256"] = canonical_sha256(report)
        report_payload = _json_bytes(report)
        report_path = directory / "report.json"
        if report_path.exists():
            if report_path.read_bytes() != report_payload:
                raise Run3ArtifactError("existing final report does not reconstruct")
        else:
            _atomic_bytes(report_path, report_payload, replace=False)
        terminal = {
            "report_file_sha256": _sha256_file(directory / "report.json"),
            "report_sha256": report["report_sha256"],
            "schema": _TERMINAL_SCHEMA,
            "seed": seed,
            "status": "RUN3_FIXED_ENDPOINT_COMPLETE",
            "update": config.update_count,
        }
        terminal["terminal_sha256"] = canonical_sha256(terminal)
        terminal_payload = _json_bytes(terminal)
        terminal_path = directory / "RUN3_TRAINING_COMPLETE.json"
        if terminal_path.exists():
            if terminal_path.read_bytes() != terminal_payload:
                raise Run3ArtifactError("existing terminal marker does not reconstruct")
        else:
            _atomic_bytes(terminal_path, terminal_payload, replace=False)
        return report
    finally:
        runner.close()


def run_campaign(
    output_root: Path,
    *,
    config: Run3RunnerConfigV1 = RUN3_REGISTERED_CONFIG,
    project_root: Optional[Path] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    if config != RUN3_REGISTERED_CONFIG:
        raise Run3ArtifactError(
            "the official Run-3 campaign entry point requires the exact registered schedule"
        )
    output_root = Path(output_root)
    manifest_path = output_root / "campaign_manifest.json"
    if resume:
        if not output_root.is_dir() or not manifest_path.is_file():
            raise Run3ArtifactError("campaign resume root/manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        supplied = manifest.pop("campaign_manifest_sha256", None)
        if supplied != canonical_sha256(manifest):
            raise Run3ArtifactError("campaign manifest digest drift")
        if manifest.get("config") != config.to_canonical_dict() or manifest.get("config_sha256") != config.canonical_sha256():
            raise Run3ArtifactError("campaign resume config drift")
        if manifest.get("seeds") != list(config.seeds):
            raise Run3ArtifactError("campaign resume seed drift")
        completed_reports = {
            seed: _verified_completed_seed_report(
                output_root / f"seed_{seed}", seed=seed, config=config
            )
            for seed in config.seeds
        }
        if (output_root / "campaign_complete.json").exists():
            complete = json.loads((output_root / "campaign_complete.json").read_text(encoding="utf-8"))
            supplied_complete = complete.pop("campaign_sha256", None)
            if supplied_complete != canonical_sha256(complete):
                raise Run3ArtifactError("completed campaign digest drift")
            expected_reports = {
                str(seed): completed_reports[seed]["report_sha256"]
                for seed in config.seeds
                if completed_reports[seed] is not None
            }
            if (
                len(expected_reports) != len(config.seeds)
                or complete.get("config_sha256") != config.canonical_sha256()
                or complete.get("manifest_file_sha256") != _sha256_file(manifest_path)
                or complete.get("seeds") != list(config.seeds)
                or complete.get("seed_reports") != expected_reports
            ):
                raise Run3ArtifactError("completed campaign binding drift")
            return {**complete, "campaign_sha256": supplied_complete}
        disk = _disk_gate(
            output_root,
            projected_bytes=_remaining_campaign_projection(output_root, config.seeds),
        )
    else:
        if output_root.exists():
            raise Run3ArtifactError("campaign root already exists")
        disk = _disk_gate(output_root, len(config.seeds))
        output_root.mkdir(parents=True)
        manifest = {
            "config": config.to_canonical_dict(),
            "config_sha256": config.canonical_sha256(),
            "disk_preflight": disk,
            "record": "run3_three_seed_campaign_manifest_v1",
            "seeds": list(config.seeds),
        }
        manifest["campaign_manifest_sha256"] = canonical_sha256(manifest)
        _atomic_bytes(manifest_path, _json_bytes(manifest), replace=False)
    campaign = {
        "config_sha256": config.canonical_sha256(),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "record": "run3_three_seed_campaign_complete_v1",
        "seed_reports": {},
        "seeds": list(config.seeds),
    }
    for seed in config.seeds:
        seed_directory = output_root / f"seed_{seed}"
        report = _verified_completed_seed_report(
            seed_directory, seed=seed, config=config
        )
        if report is None:
            report = run_seed_to_directory(
                seed_directory, seed=seed, config=config,
                project_root=project_root, resume=seed_directory.exists(),
            )
        campaign["seed_reports"][str(seed)] = report["report_sha256"]
    campaign["campaign_sha256"] = canonical_sha256(campaign)
    _atomic_bytes(output_root / "campaign_complete.json", _json_bytes(campaign), replace=False)
    return campaign


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_campaign(args.output_root, project_root=args.project_root, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
