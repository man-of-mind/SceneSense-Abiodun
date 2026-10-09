"""Create-only v2 staging; the failed v1 source inventory is immutable."""
import argparse
import json
from pathlib import Path
import shutil
from . import contract as C
from .startup import write_x

ROOT = Path(__file__).resolve().parents[2]


def deploy(worktree, previous, report, *, execute=False, candidate=None):
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
    for p in (p for folder in (directory, ROOT / C.REPAIR_PACKAGE, ROOT / C.OPTIMIZATION_PACKAGE) for p in folder.rglob("*")):
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
    prior_candidate = None if candidate is None else json.loads(Path(candidate).read_text())
    if prior_candidate is not None:
        if prior_candidate.get("worktree") != str(worktree) or prior_candidate.get("executed") is not True:
            raise ValueError("candidate revision requires the exact earlier executed staging record")
        for name, digest in prior_candidate["files"].items():
            if name not in files or C.sha(worktree / name) != digest:
                raise ValueError("candidate changed outside this bounded revision")
    for name, digest in inventory.items():
        target = worktree / name
        if target.exists() and C.sha(target) != digest:
            mutable_new = name.startswith(C.PACKAGE + "/") or name in {
                module.replace(".", "/") + ".py" for module in (C.CHILD, C.EDGE)}
            if not mutable_new:
                raise ValueError(f"sealed prior package may never be revised: {name}")
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
            raise ValueError(f"symlink target: {name}")
        if target.exists() and (not target.is_file() or C.sha(target) != digest) and (
                prior_candidate is None or prior_candidate["files"].get(name) != C.sha(target)):
            raise ValueError(f"existing foreign target: {name}")
    if Path(report).exists():
        raise FileExistsError(report)
    if execute:
        if prior_candidate is not None:
            backup = Path(report).parent / (Path(report).stem + "_previous_candidate")
            backup.mkdir(parents=True, exist_ok=False)
            for name, digest in prior_candidate["files"].items():
                write_x(backup / name, (worktree / name).read_bytes())
                if C.sha(backup / name) != digest:
                    raise ValueError("candidate source backup mismatch")
            write_x(backup / "MANIFEST.json", prior_candidate)
        for name, source in files.items():
            target = worktree / name
            if not target.exists():
                write_x(target, source.read_bytes())
            elif C.sha(target) != inventory[name]:
                # Pure byte synchronization of this NEW, never-live candidate.
                # Prior failed-run packages were verified above and are absent
                # from files. Every replaced candidate is backed up first.
                shutil.copyfile(source, target)
        if any(C.sha(worktree / name) != digest for name, digest in inventory.items()):
            raise ValueError("staged v2 digest mismatch")
    result = {"schema": "splitfusion.parity_runtime.deployment.v4", "executed": execute,
        "worktree": str(worktree), "previous_prospective_sha256": C.sha(previous),
        "prior_source_files_unchanged": len(record["runtime_sources"]), "files": inventory,
        "superseded_candidate": None if candidate is None else str(Path(candidate).resolve()),
        "count": len(inventory), "bytes": sum(p.stat().st_size for p in files.values())}
    write_x(report, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--candidate", type=Path)
    a = p.parse_args()
    print(json.dumps(deploy(a.worktree, a.previous, a.report, execute=a.execute, candidate=a.candidate), indent=2))


if __name__ == "__main__":
    main()
