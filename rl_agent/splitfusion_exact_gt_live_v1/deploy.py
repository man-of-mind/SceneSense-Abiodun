"""Deploy the integrated early-GT sources into the live worktree (new untracked files only).

Scope is exactly: every *.py of the three new packages and the two trampolines (into
rl_agent/splitfusion_single_gpu_profiler_v1/). No tracked or retained file is touched. An
existing differing file inside that scope (an earlier deployment of these packages) is backed
up into the evidence directory before an atomic replace; anything else is refused.
Usage (from the main checkout): python -m rl_agent.splitfusion_exact_gt_live_v1.deploy --name <dir> --target <worktree>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_exact_gt_live_v1"
SHIMS = {"rl_agent/splitfusion_exact_gt_live_v1/runtime_shims/child_v1_run5_exact_gt.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_run5_exact_gt.py",
         "rl_agent/splitfusion_exact_gt_live_v1/runtime_shims/edge_profiled_v1_run5_exact_gt.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_run5_exact_gt.py"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def plan_files():
    files = {}
    for package in C.NEW_PACKAGES:
        for path in sorted((ROOT / package).glob("*.py")):
            relative = str(path.relative_to(ROOT))
            files[relative] = relative
    files.update(SHIMS)
    return files


def run(name, target):
    target = Path(target).resolve(strict=True)
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    files = plan_files()
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"], capture_output=True, text=True,
                                 check=True).stdout.split())
    plan, changes = {}, []
    for source, destination in files.items():
        raw = (ROOT / source).read_bytes()
        if destination in tracked:
            raise ValueError("refusing to deploy over a tracked file: " + destination)
        dest = target / destination
        before = dest.read_bytes() if dest.exists() else None
        if dest.is_symlink():
            raise ValueError("symlink destination refused: " + destination)
        plan[destination] = dict(source=source, sha256=sha(raw), previous_sha256=None if before is None else sha(before))
        if before == raw:
            continue
        if before is not None:
            backup = out / "backup" / destination
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(before)
            if sha(backup.read_bytes()) != sha(before):
                raise ValueError("backup readback differs: " + destination)
        changes.append((dest, raw))
    (out / "PLAN.json").write_text(json.dumps(dict(target=str(target), files=plan), indent=1, sort_keys=True))
    for dest, raw in changes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=dest.parent, prefix=".deploy_")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, dest)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        directory = os.open(dest.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory); os.close(directory)
    verified = all(sha((target / d).read_bytes()) == v["sha256"] for d, v in plan.items())
    result = dict(status="DEPLOYED_VERIFIED" if verified else "FAILED", target=str(target), changed=len(changes),
                  files=len(plan), tracked_files_touched=0, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: sha(p.read_bytes()) for p in sorted(out.rglob("*"))
                                                         if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps(result))
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    run(args.name, args.target)
