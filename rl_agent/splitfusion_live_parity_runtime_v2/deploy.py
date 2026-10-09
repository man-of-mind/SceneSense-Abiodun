"""Create-only v2 staging; the failed v1 source inventory is immutable."""
import argparse
import json
from pathlib import Path
from . import contract as C
from .startup import write_x

ROOT = Path(__file__).resolve().parents[2]


def deploy(worktree, previous, report, *, execute=False):
    worktree = Path(worktree).resolve(strict=True)
    previous = Path(previous).resolve(strict=True)
    record = json.loads(previous.read_text())
    if Path(record["worktree"]) != worktree:
        raise ValueError("foreign prior worktree")
    for name, digest in record["runtime_sources"].items():
        target = worktree / name
        if target.is_symlink() or not target.is_file() or C.sha(target) != digest:
            raise ValueError(f"prior source changed: {name}")
    directory = ROOT / C.PACKAGE
    files = {}
    for p in directory.rglob("*"):
        if "__pycache__" in p.parts:
            continue
        if p.is_symlink():
            raise ValueError("symlink in v2 source")
        if p.is_file():
            files[str(p.relative_to(ROOT))] = p
    for module in (C.CHILD, C.EDGE):
        leaf = module.rsplit(".", 1)[1] + ".py"
        files[module.replace(".", "/") + ".py"] = directory / "runtime_shims" / leaf
    inventory = {name: C.sha(p) for name, p in sorted(files.items())}
    for name, digest in inventory.items():
        target = worktree / name
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
            raise ValueError(f"symlink target: {name}")
        if target.exists() and (not target.is_file() or C.sha(target) != digest):
            raise ValueError(f"existing foreign target: {name}")
    if Path(report).exists():
        raise FileExistsError(report)
    if execute:
        for name, source in files.items():
            target = worktree / name
            if not target.exists():
                write_x(target, source.read_bytes())
        if any(C.sha(worktree / name) != digest for name, digest in inventory.items()):
            raise ValueError("staged v2 digest mismatch")
    result = {"schema": "splitfusion.parity_runtime.deployment.v2", "executed": execute,
        "worktree": str(worktree), "previous_prospective_sha256": C.sha(previous),
        "prior_source_files_unchanged": len(record["runtime_sources"]), "files": inventory,
        "count": len(inventory), "bytes": sum(p.stat().st_size for p in files.values())}
    write_x(report, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    print(json.dumps(deploy(a.worktree, a.previous, a.report, execute=a.execute), indent=2))


if __name__ == "__main__":
    main()
