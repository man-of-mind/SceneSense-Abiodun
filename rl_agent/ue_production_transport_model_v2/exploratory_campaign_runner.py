#!/usr/bin/env python3
"""Run the bounded three-seed Run-4 exploratory Hybrid-SAC campaign.

The driver deliberately reuses the existing modeled collector, SAC trainer,
registered seeds and registered checkpoint boundaries.  It changes no
scientific component.  Seed 17 first reproduces the update-500 smoke, passes a
physical q/payload backlog-response gate, and proves update-250 -> update-500
resume identity in a fresh Python process.  Only then does training continue.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import smoke_preregistration
from rl_agent.ue_production_transport_model_v2 import smoke_runner


SCHEMA = "scenesense.run4_exploratory_campaign.v1"
EXPECTED_ARTIFACT_SHA256 = (
    "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c"
)
EXPECTED_SEED17_UPDATE500_SHA256 = (
    "18e678ddf4e762a0cfcf244babea3e84cd91059fa026d9a74330ffc5260c20b2"
)
PROTOCOL_RELPATH = Path(
    "rl_agent/ue_production_transport_model_v2/"
    "EXPLORATORY_CAMPAIGN_PROTOCOL_V1.md"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class _CheckpointRecorder:
    def __init__(self, seed_dir: Path) -> None:
        self.seed_dir = seed_dir
        self.checkpoint_dir = seed_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=False)
        self.records: list[dict[str, Any]] = []

    def __call__(self, event: orch.ModeledSmokeCheckpointEventV1) -> None:
        path = self.checkpoint_dir / (
            f"update_{event.update:06d}.checkpoint.json"
        )
        file_sha256 = orch.write_checkpoint(path, event.checkpoint)
        roundtrip = orch.read_checkpoint(path)
        if roundtrip.canonical_sha256 != event.checkpoint.canonical_sha256:
            raise RuntimeError("checkpoint read-back differs")
        record = {
            "update": event.update,
            "decision_count": event.checkpoint.decision_count,
            "checkpoint_sha256": event.checkpoint.canonical_sha256,
            "checkpoint_file_sha256": file_sha256,
            "checkpoint_relpath": str(path.relative_to(self.seed_dir)),
            "metrics": (
                dataclasses.asdict(event.latest_metrics)
                if event.latest_metrics is not None
                else None
            ),
        }
        self.records.append(record)
        _write_json(
            self.checkpoint_dir / f"update_{event.update:06d}.metrics.json",
            record,
        )

    def path_for(self, update: int) -> Path:
        return self.checkpoint_dir / f"update_{update:06d}.checkpoint.json"

    def write_manifest(self, seed: int) -> str:
        core = {
            "schema": "scenesense.run4_checkpoint_manifest.v1",
            "seed": seed,
            "checkpoints": self.records,
        }
        digest = hashlib.sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        _write_json(
            self.seed_dir / "CHECKPOINT_MANIFEST.json",
            {**core, "manifest_sha256": digest},
        )
        return digest


def _preflight_document(report, backlog: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": smoke_runner.PREFLIGHT_SCHEMA,
        "decision_count": report.decision_count,
        "success_count": report.success_count,
        "failure_count": report.failure_count,
        "preflight_report_sha256": report.canonical_sha256,
        "backlog_preflight": backlog,
        "passed": bool(report.passed and backlog["passed"]),
    }


def _write_decisions(path: Path, diagnostics: Sequence[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", buffering=1) as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")


def _run_cross_process_resume(
    *, artifact: Path, seed_dir: Path, recorder: _CheckpointRecorder
) -> dict[str, Any]:
    resume_dir = seed_dir / "resume_verification"
    resume_dir.mkdir(parents=True, exist_ok=False)
    resumed = resume_dir / "update_000500.checkpoint.json"
    command = [
        sys.executable,
        "-m",
        "rl_agent.ue_production_transport_model_v2.resume_registered_run",
        "--artifact",
        str(artifact),
        "--seed",
        "17",
        "--checkpoint",
        str(recorder.path_for(250)),
        "--target-update",
        "500",
        "--output-checkpoint",
        str(resumed),
        "--expected-checkpoint",
        str(recorder.path_for(500)),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"cross-process resume verifier failed with {completed.returncode}"
        )
    result_path = resumed.with_suffix(resumed.suffix + ".result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("byte_identical_to_expected") is not True:
        raise RuntimeError("cross-process resume is not byte-identical")
    return result


def _build_orchestrator(artifact: Path, seed: int):
    factory, collector_factory, variation = smoke_runner.build_harness(
        artifact, seed
    )
    return (
        orch.ModeledSmokeOrchestratorV1(
            runner_factory=factory,
            collector_factory=collector_factory,
            preflight_variation_contract=variation,
        ),
        factory,
    )


def _run_seed17_gate_and_training(
    artifact: Path, seed_dir: Path
) -> dict[str, Any]:
    orchestrator, factory = _build_orchestrator(artifact, 17)
    report = orchestrator.run_no_gradient_preflight()
    backlog = smoke_runner.backlog_report(orchestrator.collector.diagnostics())
    preflight = _preflight_document(report, backlog)
    _write_json(seed_dir / "PREFLIGHT_288.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("seed-17 preflight failed")

    recorder = _CheckpointRecorder(seed_dir)
    metrics_path = seed_dir / "UPDATE_METRICS.jsonl"
    with metrics_path.open("x", encoding="utf-8", buffering=1) as metrics_file:
        def on_update(metrics) -> None:
            metrics_file.write(
                json.dumps(dataclasses.asdict(metrics), sort_keys=True)
            )
            metrics_file.write("\n")

        smoke_summary = orchestrator.run_to_registered_update(
            500,
            checkpoint_callback=recorder,
            update_callback=on_update,
            emit_current_checkpoint=True,
        )
        if (
            smoke_summary.final_checkpoint_sha256
            != EXPECTED_SEED17_UPDATE500_SHA256
        ):
            raise RuntimeError(
                "seed-17 update-500 checkpoint did not reproduce retained smoke"
            )
        response = smoke_runner.backlog_response_probe(orchestrator)
        if response["desired_direction_passed"] is not True:
            raise RuntimeError("physical backlog-response gate failed")
        resume = _run_cross_process_resume(
            artifact=artifact, seed_dir=seed_dir, recorder=recorder
        )
        _write_json(
            seed_dir / "SMOKE_500_GATE.json",
            {
                "schema": "scenesense.run4_exploratory_smoke_gate.v1",
                "seed": 17,
                "historical_checkpoint_reproduced": True,
                "historical_checkpoint_sha256": (
                    EXPECTED_SEED17_UPDATE500_SHA256
                ),
                "backlog_response": response,
                "cross_process_resume": resume,
                "passed": True,
            },
        )
        orchestrator.run_to_registered_update(
            1500,
            checkpoint_callback=recorder,
            update_callback=on_update,
            emit_current_checkpoint=False,
        )
        final_summary = orchestrator.run_to_registered_update(
            10000,
            checkpoint_callback=recorder,
            update_callback=on_update,
            emit_current_checkpoint=False,
        )

    diagnostics = orchestrator.collector.diagnostics()
    _write_decisions(seed_dir / "DECISIONS.jsonl", diagnostics)
    manifest_sha256 = recorder.write_manifest(17)
    final = {
        "schema": SCHEMA,
        "seed": 17,
        "factory_sha256": factory.canonical_sha256,
        "final_update": final_summary.final_update,
        "final_decision_count": final_summary.final_decision_count,
        "final_checkpoint_sha256": final_summary.final_checkpoint_sha256,
        "checkpoint_manifest_sha256": manifest_sha256,
        "trajectory": smoke_runner.trajectory_report(diagnostics),
        "smoke_gate_passed": True,
    }
    _write_json(seed_dir / "SEED_COMPLETE.json", final)
    return final


def _run_additional_seed(
    artifact: Path, seed: int, seed_dir: Path
) -> dict[str, Any]:
    orchestrator, factory = _build_orchestrator(artifact, seed)
    report = orchestrator.run_no_gradient_preflight()
    backlog = smoke_runner.backlog_report(orchestrator.collector.diagnostics())
    preflight = _preflight_document(report, backlog)
    _write_json(seed_dir / "PREFLIGHT_288.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"seed-{seed} preflight failed")
    recorder = _CheckpointRecorder(seed_dir)
    metrics_path = seed_dir / "UPDATE_METRICS.jsonl"
    with metrics_path.open("x", encoding="utf-8", buffering=1) as metrics_file:
        def on_update(metrics) -> None:
            metrics_file.write(
                json.dumps(dataclasses.asdict(metrics), sort_keys=True)
            )
            metrics_file.write("\n")

        summary = orchestrator.run_to_registered_update(
            10000,
            checkpoint_callback=recorder,
            update_callback=on_update,
            emit_current_checkpoint=True,
        )
    diagnostics = orchestrator.collector.diagnostics()
    _write_decisions(seed_dir / "DECISIONS.jsonl", diagnostics)
    manifest_sha256 = recorder.write_manifest(seed)
    final = {
        "schema": SCHEMA,
        "seed": seed,
        "factory_sha256": factory.canonical_sha256,
        "final_update": summary.final_update,
        "final_decision_count": summary.final_decision_count,
        "final_checkpoint_sha256": summary.final_checkpoint_sha256,
        "checkpoint_manifest_sha256": manifest_sha256,
        "trajectory": smoke_runner.trajectory_report(diagnostics),
        "smoke_gate_passed": None,
    }
    _write_json(seed_dir / "SEED_COMPLETE.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    artifact = Path(args.artifact).resolve()
    if not artifact.is_file():
        raise SystemExit("transport artifact is missing")
    artifact_sha256 = _sha256_file(artifact)
    if artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise SystemExit("transport artifact differs from the accepted v2b model")
    protocol = Path(__file__).with_name(
        "EXPLORATORY_CAMPAIGN_PROTOCOL_V1.md"
    )
    if not protocol.is_file():
        raise SystemExit("exploratory campaign protocol is missing")

    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit("campaign output is create-only")
    output.mkdir(parents=True, exist_ok=False)
    start = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "protocol_relpath": str(PROTOCOL_RELPATH),
        "protocol_sha256": _sha256_file(protocol),
        "seeds": list(smoke_preregistration.FROZEN_CONFIG.seed_order),
        "checkpoint_updates": list(
            smoke_preregistration.FROZEN_CONFIG.checkpoint_updates
        ),
        "scientific_scope": (
            "OFFLINE_POSTHOC_EXPLORATORY_TRAINING; NOT CONFIRMATORY; "
            "NOT DEPLOYMENT AUTHORIZATION"
        ),
    }
    _write_json(output / "CAMPAIGN_START.json", start)

    results = []
    for seed in smoke_preregistration.FROZEN_CONFIG.seed_order:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        if seed == smoke_preregistration.FROZEN_CONFIG.initial_smoke_seed:
            result = _run_seed17_gate_and_training(artifact, seed_dir)
        else:
            result = _run_additional_seed(artifact, seed, seed_dir)
        results.append(result)

    terminal = {
        **start,
        "status": "COMPLETE",
        "seed_results": results,
        "all_seeds_complete": len(results) == 3,
    }
    _write_json(output / "CAMPAIGN_COMPLETE.json", terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
