"""Create-only packaging of the existing original Run-5 actor; never train.

The output is relocatable with its enclosing repository. Dependencies are
verified, not fetched, rewritten, or searched in another worktree implicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import actor_binding as A

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = Path(__file__).resolve().parent / "actor_export"
COMMITTED_SHA256 = "18bf6986f0fdf1b9b45fa284cac4c6e4caf753a9bcbac7a07a0cdc466fbff3b9"
SOURCE_FILES = {"actor_state_dict.pt": A.WEIGHTS_SHA256,
                "manifest.json": A.EXPORT_MANIFEST_SHA256,
                "COMMITTED": COMMITTED_SHA256}
FROZEN_DEPENDENCIES = {
    **A.REQUIRED_FROZEN_MODULES,
    "rl_agent/splitfusion_hybrid_sac_run5_v1/__init__.py":
        "8468acc07597a41bccb957bf851ffef44a86ec93ef96e55a48cb5d9f224b83d4",
    "data_collection/route_b_publication_actor_volume_visibility_v1/__init__.py":
        "784836022f6c92292a8e41c55166c588a3db92420d1fe41c6290a943b80315f7",
    "data_collection/route_b_publication_actor_volume_visibility_v1/core.py":
        "02d39fd1d15c31ab8323f0ece09e951e8fc0e42aecf5de06a120918276f1d3e4",
    "data_collection/route_b_publication_actor_volume_visibility_v1/scoring.py":
        "3942b8c35c990c27676fc66c13f80ed8011e2a2ddc0768cac5c6e531f85b00dd",
    "data_collection/route_b_publication_actor_volume_visibility_v1/run_audit.py":
        "cc6a0afa9997d6d1ccc603ec990046e8c6b0fcbe9ba9b442abaad0589efa7701",
    "data_collection/route_b_publication_actor_volume_visibility_v1/agreement.py":
        "a5ebb150cbbdeebaafedb61b99c9130291ee5574d63a1aa9a418dcd559fcce28",
    "data_collection/route_b_publication_actor_volume_observability_model_comparison_v1/__init__.py":
        "975a8d8dd32b1056edecb344467c5bfcd258c4605de7dc0fdcc56db2e3ba7ece",
    "data_collection/route_b_publication_actor_volume_observability_model_comparison_v1/run_comparison.py":
        "fee680fd81319151fcfe0b42977afbd0f373712fa3fe2ce04d9b2bf85cc2cbc5",
}


def source_inventory(repo_root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Read-only dependency inventory, including missing dependencies."""
    root = Path(repo_root).resolve()
    result = {}
    for relative, expected in FROZEN_DEPENDENCIES.items():
        path = root / relative
        regular = path.is_file() and not path.is_symlink()
        contained = path.resolve().is_relative_to(root)
        observed = A.F4.sha256_file(path) if regular and contained else None
        result[relative] = {"expected_sha256": expected, "observed_sha256": observed,
                            "matches": observed == expected}
    return result


def verify_sources(repo_root: Path = ROOT) -> dict[str, str]:
    result = source_inventory(repo_root)
    bad = [path for path, row in result.items() if not row["matches"]]
    A._require(not bad, "frozen dependency absent or changed: " + ", ".join(bad))
    A.verify_frozen_sources()
    return dict(FROZEN_DEPENDENCIES)


def profiler_binding(actor: A.FrozenRun5Actor) -> dict[str, Any]:
    """Mirror exact identity under keys understood by the neutral profiler."""
    native = A.binding_document(actor)
    body = {
        "schema": "scenesense.run5_live_avo.profiler_actor_binding.v1",
        "selection_scope": A.SELECTION_SCOPE,
        "selected": {"seed": A.SELECTED_SEED, "update": A.SELECTED_UPDATE},
        "actor": {"weights_file_name": "actor_state_dict.pt",
                  "weights_file_sha256": A.WEIGHTS_SHA256,
                  "boundary_sha256": A.ACTOR_TREE_SHA256},
        "binding": {"feature_schema_id": A.S5.FEATURE_SCHEMA_ID,
                    "feature_schema_sha256": A.FEATURE_SCHEMA_SHA256,
                    "policy_feature_count": 22,
                    "policy_feature_order": list(A.C5.RUN5_POLICY_FEATURE_ORDER),
                    "deployment_rule": A.DEPLOYMENT_RULE},
        "model_binding_sha256": A.NATIVE_MODEL_BINDING_SHA256,
        "source_export_files": dict(SOURCE_FILES),
        "frozen_dependencies": dict(FROZEN_DEPENDENCIES),
        "native_binding": native,
    }
    # Normalize tuples exactly as JSON serialization will; verification compares
    # documents semantically as well as verifying the self-digest.
    body = json.loads(json.dumps(body, allow_nan=False))
    return {**body, "manifest_sha256": A._sha(body)}


def verify_packaged_export(export_dir: Path = EXPORT_DIR, repo_root: Path = ROOT):
    root = Path(export_dir)
    A._require(root.is_dir() and not root.is_symlink(), "packaged export must be a real directory")
    verify_sources(repo_root)
    for filename, expected in SOURCE_FILES.items():
        path = root / filename
        A._require(path.is_file() and not path.is_symlink(), f"missing/nonregular packaged {filename}")
        A._require(A.F4.sha256_file(path) == expected, f"packaged source changed: {filename}")
    binding_path = root / "ACTOR_BINDING.json"
    A._require(binding_path.is_file() and not binding_path.is_symlink(), "packaged binding missing")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    body = {k: v for k, v in binding.items() if k != "manifest_sha256"}
    A._require(binding.get("manifest_sha256") == A._sha(body), "binding self-digest differs")
    actor = A.load_source_export(root)
    A._require(binding == profiler_binding(actor), "binding differs from frozen actor/provenance/fixtures")
    return actor, binding


def prepare(source_export: Path, output_dir: Path = EXPORT_DIR) -> dict[str, Any]:
    source, output = Path(source_export), Path(output_dir)
    A._require(not output.exists() and not output.is_symlink(), "export output must be create-only")
    verify_sources()
    actor = A.load_source_export(source)
    source_bytes = {}
    for name, expected in SOURCE_FILES.items():
        path = source / name
        A._require(path.is_file() and not path.is_symlink(), f"missing source export {name}")
        data = path.read_bytes()
        A._require(hashlib.sha256(data).hexdigest() == expected, f"source changed: {name}")
        source_bytes[name] = data
    binding = profiler_binding(actor)
    output.mkdir(parents=False, exist_ok=False)
    for name, data in source_bytes.items():
        with (output / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    with (output / "ACTOR_BINDING.json").open("x", encoding="utf-8") as handle:
        json.dump(binding, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    verify_packaged_export(output)
    # No source mutation; detect concurrent changes to the original export.
    A._require(all(A.F4.sha256_file(source / n) == h for n, h in SOURCE_FILES.items()),
               "source export changed while packaging")
    return {"status": "PACKAGED_AND_VERIFIED", "manifest_sha256": binding["manifest_sha256"],
            "weights_sha256": A.WEIGHTS_SHA256, "files": len(SOURCE_FILES) + 1,
            "output_dir": str(output), "training_performed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-export", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        actor, binding = verify_packaged_export()
        result = {"status": "VERIFIED", "manifest_sha256": binding["manifest_sha256"],
                  "actor_boundary_sha256": actor.boundary_sha256}
    else:
        if args.source_export is None:
            parser.error("--source-export is required for create-only packaging")
        result = prepare(args.source_export)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
