#!/usr/bin/env python3
"""Restore one full Run-4 checkpoint and continue to a registered boundary.

This process-level verifier is intentionally narrow.  It does not change the
reward, policy, state, action space, transport model, or hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.ue_production_transport_model_v2 import smoke_runner


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-update", required=True, type=int)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--expected-checkpoint")
    args = parser.parse_args(argv)

    source = Path(args.checkpoint)
    target = Path(args.output_checkpoint)
    result_path = target.with_suffix(target.suffix + ".result.json")
    if target.exists() or result_path.exists():
        raise SystemExit("resume output is create-only")
    if not target.parent.is_dir():
        raise SystemExit("resume output parent does not exist")

    checkpoint = orch.read_checkpoint(source)
    factory, collector_factory, variation = smoke_runner.build_harness(
        Path(args.artifact), args.seed
    )
    restored = orch.ModeledSmokeOrchestratorV1.restore(
        checkpoint,
        runner_factory=factory,
        collector_factory=collector_factory,
        preflight_variation_contract=variation,
    )
    captured = []

    def on_checkpoint(event) -> None:
        if event.update != args.target_update:
            raise RuntimeError("resume emitted an unexpected checkpoint")
        captured.append(event.checkpoint)
        orch.write_checkpoint(target, event.checkpoint)

    summary = restored.run_to_registered_update(
        args.target_update,
        checkpoint_callback=on_checkpoint,
        emit_current_checkpoint=False,
    )
    if len(captured) != 1:
        raise RuntimeError("resume did not emit exactly one target checkpoint")
    exact_match = None
    expected_sha256 = None
    if args.expected_checkpoint is not None:
        expected = Path(args.expected_checkpoint)
        expected_sha256 = _sha256_file(expected)
        exact_match = target.read_bytes() == expected.read_bytes()

    document = {
        "schema": "scenesense.run4_cross_process_resume.v1",
        "seed": args.seed,
        "source_update": checkpoint.update_count,
        "target_update": args.target_update,
        "source_checkpoint_sha256": checkpoint.canonical_sha256,
        "resumed_checkpoint_sha256": captured[0].canonical_sha256,
        "resumed_checkpoint_file_sha256": _sha256_file(target),
        "expected_checkpoint_file_sha256": expected_sha256,
        "byte_identical_to_expected": exact_match,
        "summary_final_update": summary.final_update,
        "summary_final_decision_count": summary.final_decision_count,
    }
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if exact_match is not False else 2


if __name__ == "__main__":
    sys.exit(main())
