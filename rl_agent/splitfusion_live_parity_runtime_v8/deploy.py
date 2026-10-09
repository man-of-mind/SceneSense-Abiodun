"""Add-only staging after verifying the complete preceding live source set."""
import argparse
import json
from pathlib import Path
from . import contract as C
from .startup import write_x

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT/"experiments/splitfusion_live_parity_runtime_v7/20261004_cpu_worker_v7/qualification_100/PROSPECTIVE.json"


def deploy(worktree, report, *, execute=False):
    worktree = Path(worktree).resolve(strict=True)
    prior = json.loads(PREVIOUS.read_text())
    if (prior["schema"] != "splitfusion.parity_runtime.prospective.v7"
            or prior["budget"] != C.BUDGET or Path(prior["worktree"]) != worktree):
        raise ValueError("exact previous worktree/budget required")
    for name, digest in prior["runtime_sources"].items():
        target = worktree/name
        C._regular(target, worktree)
        if C.sha(target) != digest:
            raise ValueError("sealed previous source changed: " + name)
    C.worker_inventory(ROOT)
    C.bootstrap_inventory(ROOT)
    files = {}
    for package in (C.PACKAGE, *C.BOOTSTRAP_PACKAGES):
        for path in (ROOT/package).rglob("*"):
            if path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
                C._regular(path, ROOT)
                files[str(path.relative_to(ROOT))] = path
    for module in (C.CHILD, C.EDGE):
        files[module.replace(".", "/")+".py"] = ROOT/C.PACKAGE/"runtime_shims"/(module.rsplit(".",1)[1]+".py")
    hashes = {name:C.sha(path) for name,path in files.items()}
    for name, digest in hashes.items():
        target = worktree/name
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError("unsafe staging path")
        if target.exists() and (not target.is_file() or C.sha(target) != digest):
            raise ValueError("foreign existing candidate: " + name)
    if Path(report).exists():
        raise FileExistsError(report)
    if execute:
        for name, path in files.items():
            target = worktree/name
            if not target.exists():
                write_x(target, path.read_bytes())
        if any(C.sha(worktree/name) != digest for name,digest in hashes.items()):
            raise ValueError("staged source mismatch")
    result = dict(schema="splitfusion.parity_runtime.deployment.v8", executed=execute,
        worktree=str(worktree), files=dict(sorted(hashes.items())),
        prior_source_files_unchanged=len(prior["runtime_sources"]),
        previous_prospective_sha256=C.sha(PREVIOUS), handoff_binding=C.handoff_binding(ROOT))
    write_x(report,result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(deploy(args.worktree,args.report,execute=args.execute)))
