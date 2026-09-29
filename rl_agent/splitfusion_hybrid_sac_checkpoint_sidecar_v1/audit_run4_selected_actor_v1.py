"""Read-only audit of the selected Run-4 actor (seed 43, update 10,000).

No retraining and no checkpoint replay: the event-sourced checkpoint JSON is
only parsed to read its recorded actor boundary hash. The actor is loaded via
the deployment path (``frozen_actor_v2.load_registered_actor``, weights-only).

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m \
        rl_agent.splitfusion_hybrid_sac_checkpoint_sidecar_v1.audit_run4_selected_actor_v1 \
        [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as fa
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tree_sha256,
)

EXPECTED_WEIGHTS_SHA256 = "d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29"
EXPECTED_BOUNDARY_SHA256 = "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3"
AUDIT_PATH = Path(__file__).resolve().with_name("RUN4_SELECTED_ACTOR_AUDIT.json")


def audit(repo_root: Path = fa.REPOSITORY_ROOT) -> Dict[str, Any]:
    export_dir = repo_root / fa.ACTOR_EXPORT_RELPATH
    weights = export_dir / "actor_state_dict.pt"
    manifest = fa.load_json(export_dir / "ACTOR_EXPORT_MANIFEST.json")
    checks: Dict[str, bool] = {}
    checks["seed_43_update_10000"] = (
        manifest["selected"]["seed"] == 43 and manifest["selected"]["update"] == 10000
        and fa.SELECTED.seed == 43 and fa.SELECTED.update == 10000)
    checks["actor_state_dict_exists"] = weights.is_file() and not weights.is_symlink()
    weights_sha = fa.sha256_file(weights)
    checks["weights_file_sha256"] = (
        weights_sha == EXPECTED_WEIGHTS_SHA256
        == manifest["actor"]["weights_file_sha256"])
    fa.verify_manifest(manifest)
    checks["manifest_self_digest"] = True
    state = torch.load(weights, map_location="cpu", weights_only=True)
    checks["tensor_inventory"] = (
        fa.tensor_inventory(state) == manifest["actor"]["tensor_inventory"])
    checks["weights_boundary_sha256"] = (
        _tree_sha256(state) == EXPECTED_BOUNDARY_SHA256
        == manifest["actor"]["boundary_sha256"])
    frozen = fa.load_registered_actor(repo_root)
    checks["loaded_actor_boundary_sha256"] = (
        fa.actor_boundary_sha256(frozen.module) == EXPECTED_BOUNDARY_SHA256)
    checks["fixtures_equal_sealed_manifest"] = (
        fa.fixture_outputs(frozen.module) == manifest["fixtures"])

    checkpoint_path = repo_root / manifest["selected"]["checkpoint_relpath"]
    checks["event_checkpoint_file_sha256"] = (
        fa.sha256_file(checkpoint_path) == manifest["selected"]["checkpoint_file_sha256"])
    envelope = json.loads(checkpoint_path.read_bytes())
    recorded = envelope["checkpoint"]["boundary"]
    checks["event_checkpoint_sha256"] = (
        envelope["checkpoint_sha256"] == manifest["selected"]["canonical_checkpoint_sha256"])
    checks["event_boundary_actor_sha256"] = (
        recorded["actor_sha256"] == EXPECTED_BOUNDARY_SHA256
        and recorded["update_count"] == 10000)
    # Event-sourced only: exactly the dataclass fields (ledger, collector,
    # hash boundary) and a boundary of hex digests / counters, no tensors.
    checks["event_checkpoint_has_no_tensor_state"] = (
        set(envelope["checkpoint"])
        == {item.name for item in fields(orch.ModeledSmokeCheckpointV1)} | {"schema"}
        and all(type(value) is int or (type(value) is str and len(value) == 64)
                for value in recorded.values()))
    return {
        "schema": "scenesense.run4.selected_actor_audit.v1",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "seed": 43,
        "update": 10000,
        "weights_file": fa.ACTOR_EXPORT_RELPATH + "/actor_state_dict.pt",
        "weights_file_sha256": weights_sha,
        "actor_boundary_sha256": EXPECTED_BOUNDARY_SHA256,
        "export_manifest_sha256": manifest["manifest_sha256"],
        "tensor_count": len(manifest["actor"]["tensor_inventory"]),
        "fixture_count": len(manifest["fixtures"]),
        "event_checkpoint_sha256": envelope["checkpoint_sha256"],
        "event_checkpoint_top_level_fields": sorted(envelope["checkpoint"]),
        "provenance": (
            "The weights file was produced once by event-sourced replay of the "
            "update-10,000 reconstruction checkpoint (export_frozen_actor_v2); the "
            "checkpoint itself carries only hashes. This audit re-verified the export "
            "without retraining or replay."
        ),
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help=f"write {AUDIT_PATH.name} (create-only)")
    args = parser.parse_args(argv)
    result = audit()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write:
        with AUDIT_PATH.open("x", encoding="utf-8") as handle:
            handle.write(text)
    sys.stdout.write(text)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
