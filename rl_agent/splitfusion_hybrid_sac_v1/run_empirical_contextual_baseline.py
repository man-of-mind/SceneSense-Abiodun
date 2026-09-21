"""Durable CLI for the preliminary train-split Hybrid-SAC baseline."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from .empirical_contextual_baseline_runner import (
    BASELINE_CHECKPOINT_INTERVAL_UPDATES,
    BASELINE_PROVENANCE_LABEL,
    BASELINE_SCOPE_DISCLOSURE,
    EmpiricalContextualBaselineRunnerV1,
    PHASE_LABEL,
    REGISTERED_BASELINE_CONFIG,
)
from .empirical_contextual_smoke_runner import (
    EmpiricalSmokeCheckpointV1,
    EmpiricalSmokeConfigV1,
)
from .transaction_identity import canonical_sha256

__all__ = ["run_seed_to_directory"]


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


def _atomic_json(path: Path, document: Dict[str, Any]) -> None:
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_durable_bytes(path, payload)


def _atomic_checkpoint(path: Path, checkpoint: EmpiricalSmokeCheckpointV1) -> None:
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    _atomic_durable_bytes(path, buffer.getvalue())


def _load_checkpoint(path: Path) -> EmpiricalSmokeCheckpointV1:
    with path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    if type(checkpoint) is not EmpiricalSmokeCheckpointV1:
        raise RuntimeError("checkpoint file contains a foreign object")
    checkpoint.require_valid()
    return checkpoint


def _metrics_csv(metrics: Iterable[Any]) -> bytes:
    rows = [dict(update_index=index, **metric.as_dict()) for index, metric in enumerate(metrics, 1)]
    if not rows:
        raise RuntimeError("metrics CSV requires at least one completed update")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _config_document(
    config: EmpiricalSmokeConfigV1, seed: int, checkpoint_interval_updates: int
) -> Dict[str, Any]:
    return {
        "checkpoint_interval_updates": checkpoint_interval_updates,
        "config": config.to_canonical_dict(),
        "config_sha256": config.canonical_sha256(),
        "provenance_label": BASELINE_PROVENANCE_LABEL,
        "record": "splitfusion.preliminary_baseline_config.v1",
        "seed": seed,
        "scope_disclosure": BASELINE_SCOPE_DISCLOSURE,
    }


def run_seed_to_directory(
    *,
    output_directory: Path,
    seed: int,
    config: EmpiricalSmokeConfigV1 = REGISTERED_BASELINE_CONFIG,
    checkpoint_interval_updates: int = BASELINE_CHECKPOINT_INTERVAL_UPDATES,
    stop_after_updates: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run or resume one seed and atomically materialize its diagnostics.

    ``stop_after_updates`` and a non-500 checkpoint interval exist only for
    focused schedule/resume tests.  The registered CLI never supplies them.
    """
    if type(checkpoint_interval_updates) is not int or checkpoint_interval_updates < 1:
        raise ValueError("checkpoint interval must be a positive integer")
    target = config.update_count if stop_after_updates is None else stop_after_updates
    if type(target) is not int or not 0 <= target <= config.update_count:
        raise ValueError("stop_after_updates is outside the configured schedule")
    output_directory = Path(output_directory).resolve()
    checkpoint_path = output_directory / "checkpoint_latest.pt"

    with EmpiricalContextualBaselineRunnerV1(
        seed=seed, config=config, project_root=project_root
    ) as runner:
        runner.require_registered_train_binding()
        config_document = _config_document(
            config, seed, checkpoint_interval_updates
        )
        config_path = output_directory / "config.json"
        if config_path.exists():
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            if existing_config != config_document:
                raise RuntimeError(
                    "output directory already belongs to another baseline config"
                )
        if checkpoint_path.exists():
            runner.load_checkpoint(_load_checkpoint(checkpoint_path))
        bindings_path = output_directory / "bindings.json"
        if bindings_path.exists():
            existing_bindings = json.loads(
                bindings_path.read_text(encoding="utf-8")
            )
            expected_identity = {
                "baseline_binding": runner.baseline_binding_document,
                "baseline_binding_sha256": runner._runner_binding_sha256,
                "d1_binding": runner.environment.binding.to_canonical_dict(),
                "d1_binding_sha256": runner._d1_binding_sha256,
                "fit_partition_sha256": runner.environment.fit_partition_sha256,
                "provenance_label": BASELINE_PROVENANCE_LABEL,
                "sampling_contract_sha256": (
                    runner.environment.sampling_contract_sha256
                ),
                "sampling_split": runner.environment.sampling_split,
                "trainer_config": runner.trainer_config.to_canonical_dict(),
                "trainer_config_sha256": runner.trainer_config.canonical_sha256(),
            }
            for name, expected in expected_identity.items():
                if existing_bindings.get(name) != expected:
                    raise RuntimeError(
                        f"output directory binding mismatch at {name}"
                    )
        if runner.completed_updates > target:
            raise RuntimeError("checkpoint is ahead of the requested target")
        # No durable artifact is touched until all reusable-directory identity
        # and checkpoint gates above have passed.
        _atomic_json(config_path, config_document)

        last_summary = None
        while runner.completed_updates < target:
            next_boundary = min(
                target,
                config.update_count,
                ((runner.completed_updates // checkpoint_interval_updates) + 1)
                * checkpoint_interval_updates,
            )
            last_summary = runner.run_until_updates(next_boundary)
            checkpoint = runner.checkpoint()
            numbered_checkpoint = (
                output_directory
                / "checkpoints"
                / f"checkpoint_{runner.completed_updates:06d}.pt"
            )
            _atomic_checkpoint(numbered_checkpoint, checkpoint)
            _atomic_checkpoint(checkpoint_path, checkpoint)
            _atomic_durable_bytes(
                output_directory / "metrics.csv", _metrics_csv(runner.metrics)
            )
            replay_binding = runner.replay.binding
            if replay_binding is None:
                raise RuntimeError("completed updates have no replay binding")
            _atomic_json(
                bindings_path,
                {
                    "baseline_binding": runner.baseline_binding_document,
                    "baseline_binding_sha256": runner._runner_binding_sha256,
                    "d1_binding": runner.environment.binding.to_canonical_dict(),
                    "d1_binding_sha256": runner._d1_binding_sha256,
                    "fit_partition_sha256": runner.environment.fit_partition_sha256,
                    "provenance_label": BASELINE_PROVENANCE_LABEL,
                    "replay_binding": replay_binding.to_canonical_dict(),
                    "replay_binding_sha256": replay_binding.canonical_sha256(),
                    "sampling_contract_sha256": runner.environment.sampling_contract_sha256,
                    "sampling_split": runner.environment.sampling_split,
                    "trainer_config": runner.trainer_config.to_canonical_dict(),
                    "trainer_config_sha256": runner.trainer_config.canonical_sha256(),
                },
            )

        if last_summary is None:
            if runner.completed_updates == 0:
                raise RuntimeError("zero-update artifact emission is unsupported")
            last_summary = runner.summary()
        complete = runner.completed_updates == config.update_count
        report = {
            "artifact_inventory": {
                "bindings": "bindings.json",
                "checkpoint_latest": "checkpoint_latest.pt",
                "checkpoint_series": "checkpoints/checkpoint_*.pt",
                "config": "config.json",
                "metrics": "metrics.csv",
                "report": "report.json",
            },
            "checkpoint_cadence_updates": checkpoint_interval_updates,
            "checkpoint_sha256": runner.checkpoint().checkpoint_sha256,
            "completed_updates": runner.completed_updates,
            "configured_updates": config.update_count,
            "phase_label": PHASE_LABEL,
            "provenance_label": BASELINE_PROVENANCE_LABEL,
            "scope_disclosure": BASELINE_SCOPE_DISCLOSURE,
            "seed": seed,
            "status": "COMPLETE" if complete else "IN_PROGRESS",
            "summary": last_summary.to_canonical_dict(),
            "target_rule": "y_equals_observed_terminal_reward",
        }
        report["report_content_sha256"] = canonical_sha256(report)
        _atomic_json(output_directory / "report.json", report)
        return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/resume the preliminary train-split Hybrid-SAC baseline."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, action="append", choices=REGISTERED_BASELINE_CONFIG.seeds
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seeds: Sequence[int] = (
        tuple(args.seed) if args.seed else REGISTERED_BASELINE_CONFIG.seeds
    )
    reports = []
    for seed in seeds:
        reports.append(
            run_seed_to_directory(
                output_directory=args.output / f"seed_{seed}", seed=seed
            )
        )
    campaign = {
        "config_sha256": REGISTERED_BASELINE_CONFIG.canonical_sha256(),
        "phase_label": PHASE_LABEL,
        "provenance_label": BASELINE_PROVENANCE_LABEL,
        "reports": reports,
        "scope_disclosure": BASELINE_SCOPE_DISCLOSURE,
        "seeds": list(seeds),
        "status": "COMPLETE",
    }
    campaign["campaign_report_content_sha256"] = canonical_sha256(campaign)
    _atomic_json(args.output / "campaign_report.json", campaign)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
