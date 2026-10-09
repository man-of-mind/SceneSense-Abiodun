"""Add-only, hash-checked deployment into the retained profiler worktree."""
import argparse
import json
import os
from pathlib import Path
from . import live_contract as C

ROOT = Path(__file__).resolve().parents[2]


def write_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def deploy(worktree, report, *, execute=False):
    worktree = Path(worktree).resolve(strict=True)
    if worktree == ROOT or not (worktree / "rl_agent/splitfusion_single_gpu_profiler_v1/runner_v1.py").is_file():
        raise ValueError("existing separate profiler worktree required")
    files = {name: ROOT / name for name in C.worker_sources()}
    for module in (C.CHILD, C.EDGE):
        files[module.replace(".", "/") + ".py"] = (
            ROOT / C.PACKAGE / "runtime_shims" / (module.rsplit(".", 1)[-1] + ".py"))
    # Newly added live interfaces also use the retained complete-source wrapper.
    for package in ("rl_agent/splitfusion_live_parity_repair_v2",):
        for path in (ROOT / package).glob("*.py"):
            files[str(path.relative_to(ROOT))] = path
    hashes = {name: C.sha(path) for name, path in files.items()}
    added, identical = [], []
    for name, source in files.items():
        target = worktree / name
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(ROOT):
            raise ValueError("unsafe source: " + name)
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError("unsafe destination: " + name)
        if target.exists():
            if not target.is_file() or C.sha(target) != hashes[name]:
                raise ValueError("will not overwrite existing different source: " + name)
            identical.append(name)
        else:
            added.append(name)
    report = Path(report)
    if report.exists():
        raise FileExistsError(report)
    if execute:
        for name in added:
            write_bytes(worktree / name, files[name].read_bytes())
        if any(C.sha(worktree / name) != digest for name, digest in hashes.items()):
            raise ValueError("deployment readback differs")
    result = dict(schema="splitfusion.numeric_gt.live_deployment.v1", executed=execute,
        source_root=str(ROOT), worktree=str(worktree), sources=dict(sorted(hashes.items())),
        added=sorted(added), already_identical=sorted(identical), overwritten=[],
        launched_live_services=False)
    write_bytes(report, (json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = deploy(args.worktree, args.report, execute=args.execute)
    print(json.dumps(dict(executed=value["executed"], added=len(value["added"]),
                         identical=len(value["already_identical"]))))
