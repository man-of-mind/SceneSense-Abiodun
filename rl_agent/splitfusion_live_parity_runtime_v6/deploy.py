"""Add-only staging of exact-input diagnostic after sealed-v5 verification."""
import argparse
import json
from pathlib import Path
from . import contract as C
from .startup import write_x

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT/"experiments/splitfusion_live_parity_runtime_v5/20261003_optimization_v5/qualification_100/PROSPECTIVE.json"
REPLAY_REPORT = ROOT/"experiments/splitfusion_paired_quality_replay_v1/20261003_offline_preparation_v1/REPORT.json"


def deploy(worktree, report, *, execute=False):
    worktree = Path(worktree).resolve(strict=True)
    prior = json.loads(PREVIOUS.read_text())
    if (prior["schema"] != "splitfusion.parity_runtime.prospective.v5"
            or prior["budget"] != 100 or Path(prior["worktree"]) != worktree):
        raise ValueError("exact prior v5 worktree/budget required")
    for name, digest in prior["runtime_sources"].items():
        target = worktree/name
        if target.is_symlink() or not target.is_file() or C.sha(target) != digest:
            raise ValueError(f"sealed prior source changed: {name}")
    proof = json.loads(REPLAY_REPORT.read_text())
    for name, digest in proof["sources"].items():
        if C.sha(ROOT/name) != digest:
            raise ValueError(f"offline-prepared input/replay code changed: {name}")
    files = {}
    for package in (C.PACKAGE, *C.REPLAY_PACKAGES):
        for path in (ROOT/package).rglob("*.py"):
            if path.is_symlink():
                raise ValueError("new source symlink")
            files[str(path.relative_to(ROOT))] = path
    for module in (C.CHILD,C.EDGE):
        files[module.replace(".","/")+".py"] = ROOT/C.PACKAGE/"runtime_shims"/(module.rsplit(".",1)[1]+".py")
    hashes = {name:C.sha(path) for name,path in files.items()}
    for name,digest in hashes.items():
        target = worktree/name
        if target.is_symlink() or any(p.is_symlink() for p in target.parents):
            raise ValueError("unsafe staging path")
        if target.exists() and (not target.is_file() or C.sha(target) != digest):
            raise ValueError(f"foreign existing candidate: {name}")
    if Path(report).exists():
        raise FileExistsError(report)
    if execute:
        for name,path in files.items():
            target = worktree/name
            if not target.exists():
                write_x(target,path.read_bytes())
        if any(C.sha(worktree/name) != digest for name,digest in hashes.items()):
            raise ValueError("staged source mismatch")
    result = {"schema":"splitfusion.parity_runtime.deployment.v6", "executed":execute,
        "worktree":str(worktree), "files":dict(sorted(hashes.items())),
        "prior_source_files_unchanged":len(prior["runtime_sources"]),
        "previous_prospective_sha256":C.sha(PREVIOUS), "replay_preparation_report_sha256":C.sha(REPLAY_REPORT)}
    write_x(report,result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(deploy(args.worktree,args.report,execute=args.execute)))
