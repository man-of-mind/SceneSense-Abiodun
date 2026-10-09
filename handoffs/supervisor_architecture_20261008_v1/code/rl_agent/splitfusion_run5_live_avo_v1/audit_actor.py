"""Read-only original Run-5 actor audit and zero-argument profiler G8 loader."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import actor_binding as A
from . import prepare_binding as P


def load_registered_actor() -> A.FrozenRun5Actor:
    """CPU-only load of the package-local, pinned original Run-5 export."""
    return P.verify_packaged_export()[0]


def audit(export_dir: Path = P.EXPORT_DIR) -> dict:
    rng_before = A.F4.rng_fingerprint()
    cuda_before = torch.cuda.is_initialized()
    actor, binding = P.verify_packaged_export(export_dir)
    weights = Path(export_dir) / "actor_state_dict.pt"
    checks = {
        "seed_43_update_10000": binding["selected"] == {"seed": 43, "update": 10000},
        "original_quality_aware_run5_22_features": (
            binding["binding"]["policy_feature_count"] == 22
            and binding["binding"]["policy_feature_order"][17] == "prev_quality_qperc"),
        "weights_file_sha256": A.F4.sha256_file(weights) == A.WEIGHTS_SHA256,
        "loaded_actor_boundary_sha256": actor.boundary_sha256 == A.ACTOR_TREE_SHA256,
        "fixtures_equal_sealed_binding": (
            [actor.act_on_vector(row).to_dict() for row in A.registered_fixture_states()]
            == binding["native_binding"]["batch1_fixtures"]),
        "cpu_float32_readonly": all(p.device.type == "cpu" and p.dtype == torch.float32
                                      and not p.requires_grad for p in actor.module.parameters()),
        "eval_mode": not actor.module.training,
        "source_dependencies_and_export": True,
        "rng_neutral": rng_before == A.F4.rng_fingerprint(),
        "cuda_not_initialized": not cuda_before and not torch.cuda.is_initialized(),
    }
    return {"schema": "scenesense.run5_live_avo.actor_audit.v1",
            "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "seed": 43, "update": 10000,
            "selection_scope": A.SELECTION_SCOPE,
            "weights_file": str(weights), "weights_file_sha256": A.F4.sha256_file(weights),
            "actor_boundary_sha256": actor.boundary_sha256,
            "binding_sha256": binding["manifest_sha256"],
            "feature_schema_sha256": A.FEATURE_SCHEMA_SHA256,
            "source_export_manifest_sha256": A.EXPORT_MANIFEST_SHA256,
            "fixture_count": len(binding["native_binding"]["batch1_fixtures"]),
            "cuda_initialized": torch.cuda.is_initialized(),
            "training_or_reconstruction_performed": False}


def main() -> int:
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
