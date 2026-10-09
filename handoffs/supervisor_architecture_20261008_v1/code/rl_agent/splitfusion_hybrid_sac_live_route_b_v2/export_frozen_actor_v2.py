#!/usr/bin/env python3
"""Restore seed 43 / update 10,000 once and export the frozen actor.

Restoration uses only the registered event-sourced path:
``smoke_runner.build_harness(artifact, 43)`` ->
``modeled_smoke_orchestrator.read_checkpoint`` ->
``ModeledSmokeOrchestratorV1.restore``.  ``restore`` replays the exact ledger
from genesis and refuses anything that is not bit-identical to the stored
boundary.  No training step is taken after restoration.

The output directory is create-only and receives:

* ``actor_state_dict.pt``   CPU weights-only ``state_dict``;
* ``ACTOR_EXPORT_MANIFEST.json``  sealed manifest (see ``frozen_actor_v2``);
* ``RESTORE_RESULT.json``   timings, the restored boundary and reload proof.

CPU-only offline work.  No CARLA, OAI, Docker, network service or CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as FA
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.ue_production_transport_model_v2 import smoke_runner

EXPORT_ROOT_RELPATH = "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2"
WEIGHTS_NAME = "actor_state_dict.pt"
MANIFEST_NAME = "ACTOR_EXPORT_MANIFEST.json"
RESULT_NAME = "RESTORE_RESULT.json"


def _write_create_only(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    root = FA.REPOSITORY_ROOT
    output = Path(args.output_dir).resolve()
    export_root = (root / EXPORT_ROOT_RELPATH).resolve()
    if output.parent != export_root:
        raise SystemExit(f"output must be a direct child of {export_root}")
    if output.exists():
        raise SystemExit("export output directory is create-only")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("run with CUDA_VISIBLE_DEVICES= (empty) to keep CUDA dark")

    started = time.time()
    sources = FA.verify_pinned_sources(root)
    checkpoint = orch.read_checkpoint(root / FA.SELECTED.checkpoint_relpath)
    if checkpoint.canonical_sha256 != FA.SELECTED.canonical_checkpoint_sha256:
        raise SystemExit("checkpoint canonical digest differs")
    if (checkpoint.update_count, checkpoint.boundary.actor_sha256) != (
        FA.SELECTED.update, FA.SELECTED.actor_boundary_sha256
    ):
        raise SystemExit("checkpoint update/actor boundary differs")
    seed_complete = FA.load_json(root / FA.SELECTED.seed_complete_relpath)
    if (seed_complete["seed"], seed_complete["final_update"],
            seed_complete["final_checkpoint_sha256"],
            seed_complete["factory_sha256"]) != (
            FA.SELECTED.seed, FA.SELECTED.update,
            FA.SELECTED.canonical_checkpoint_sha256, checkpoint.factory_sha256):
        raise SystemExit("SEED_COMPLETE does not bind this seed/update/checkpoint")

    output.mkdir(parents=False, exist_ok=False)
    factory, collector_factory, variation = smoke_runner.build_harness(
        root / FA.SELECTED.transport_artifact_relpath, FA.SELECTED.seed)
    if factory.seed_plan.master_seed != FA.SELECTED.seed:
        raise SystemExit("harness seed plan is not seed 43")
    restore_started = time.time()
    restored = orch.ModeledSmokeOrchestratorV1.restore(
        checkpoint,
        runner_factory=factory,
        collector_factory=collector_factory,
        preflight_variation_contract=variation,
    )
    restore_seconds = time.time() - restore_started
    if restored.update_count != FA.SELECTED.update:
        raise SystemExit("restored update count differs")
    boundary = restored._boundary()
    if boundary != checkpoint.boundary:
        raise SystemExit("restored boundary differs from the checkpoint")

    actor = restored.runner.model_bundle.actor
    actor_digest = FA.actor_boundary_sha256(actor)
    if actor_digest != FA.SELECTED.actor_boundary_sha256:
        raise SystemExit("restored actor digest differs from the boundary")
    state = {name: tensor.detach().to("cpu").clone().contiguous()
             for name, tensor in actor.state_dict().items()}
    if FA.actor_boundary_sha256(state) != actor_digest:
        raise SystemExit("cloned state_dict digest differs")
    fixtures = FA.fixture_outputs(actor)

    weights_path = output / WEIGHTS_NAME
    with weights_path.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest = FA.seal_manifest({
        "schema": FA.MANIFEST_SCHEMA,
        "selection_scope": FA.SELECTION_SCOPE,
        "selected": FA.SELECTED.to_dict(),
        "binding": FA.binding_document(),
        "actor": {
            "boundary_sha256": actor_digest,
            "weights_file_name": WEIGHTS_NAME,
            "weights_file_sha256": FA.sha256_file(weights_path),
            "weights_format": "torch.save_state_dict_weights_only",
            "tensor_inventory": FA.tensor_inventory(state),
            "restored_factory_sha256": checkpoint.factory_sha256,
            "restored_decision_count": restored.decision_count,
            "restored_update_count": restored.update_count,
        },
        "fixtures": fixtures,
    })
    _write_create_only(output / MANIFEST_NAME, manifest)

    # Independent reload through the deployment loader (weights_only=True).
    frozen = FA.load_frozen_actor(weights_path, manifest)
    reloaded = {name: tensor for name, tensor in frozen.module.state_dict().items()}
    exact = set(reloaded) == set(state) and all(
        torch.equal(reloaded[name], state[name])
        and reloaded[name].dtype == state[name].dtype
        for name in state)
    if not exact:
        raise SystemExit("reloaded tensors are not exactly equal")

    result = {
        "schema": "scenesense.run4_live_v2.actor_restore_result.v1",
        "verdict": "RESTORED_BIT_IDENTICAL_AND_EXPORTED",
        "pinned_sources": sources,
        "restored_boundary": boundary.to_dict(),
        "restored_boundary_sha256": boundary.canonical_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "weights_file_sha256": manifest["actor"]["weights_file_sha256"],
        "reload_exact_tensor_equality": exact,
        "reload_fixture_equality": True,
        "fixture_count": len(fixtures),
        "training_steps_after_restore": 0,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "restore_seconds": round(restore_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
    }
    if result["cuda_initialized"]:
        raise SystemExit("CUDA was initialized; refusing to publish")
    _write_create_only(output / RESULT_NAME, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
