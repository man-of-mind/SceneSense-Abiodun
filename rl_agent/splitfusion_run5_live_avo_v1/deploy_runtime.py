"""Additive, create-only runtime deployment; no Git or frozen-source writes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import prepare_binding as P


def deploy(worktree: Path) -> dict:
    source = Path(__file__).resolve().parent
    worktree = Path(worktree).resolve(strict=True)
    destination = worktree / "rl_agent" / source.name
    items = [(p, destination / p.relative_to(source)) for p in source.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    for filename in ("__init__.py", "run5_state_contract.py", "run5_snr_v2.py"):
        relative = Path("rl_agent/splitfusion_hybrid_sac_run5_v1") / filename
        items.append((P.ROOT / relative, worktree / relative))
    for path in sorted((source / "runtime_shims").glob("*.py")):
        items.append((path, worktree / "rl_agent/splitfusion_single_gpu_profiler_v1" / path.name))
    if any(path.is_symlink() for path, _ in items):
        raise ValueError("Deployment source symlinks refused")
    if any(target.exists() or target.is_symlink() for _, target in items):
        raise FileExistsError("Deployment is create-only; existing files require explicit review")
    manifest = {}
    for path, target in items:
        payload = path.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Deployment hash mismatch: {target}")
        manifest[str(target.relative_to(worktree))] = digest
    return {"status": "DEPLOYED_ADDITIVELY", "files": manifest, "frozen_sources_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worktree", type=Path)
    args = parser.parse_args()
    print(json.dumps(deploy(args.worktree), indent=2, sort_keys=True))
