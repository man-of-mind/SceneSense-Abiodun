"""Deploy the archive capacity repair into the live worktree (untracked files only, backups kept, verified).

Writes the two repaired handoff files (only if the worktree holds the registered ORIGINAL or already the REPAIRED
bytes) and this package. Usage (main checkout):
  python -m rl_agent.splitfusion_gt_archive_capacity_v1.deploy --name deploy_01 --target <worktree>
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_gt_archive_capacity_v1"


def run(name, target):
    target = Path(target).resolve(strict=True)
    out = ROOT / C.EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    C.repaired(ROOT)                                            # the source side is the registered repair
    files = {relative: relative for relative in C.ORIGINAL}
    files.update({str(p.relative_to(ROOT)): str(p.relative_to(ROOT)) for p in sorted((ROOT / PACKAGE).glob("*.py"))})
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"], capture_output=True, text=True,
                                 check=True).stdout.split())
    plan, changes = {}, []
    for source, destination in files.items():
        raw = (ROOT / source).read_bytes()
        dest = target / destination
        if destination in tracked or dest.is_symlink():
            raise ValueError("refusing tracked/symlink destination: " + destination)
        before = dest.read_bytes() if dest.exists() else None
        previous = None if before is None else C.sha(dest)
        if destination in C.ORIGINAL and previous not in (C.ORIGINAL[destination], C.REPAIRED[destination]):
            raise ValueError("worktree file is neither the registered original nor the repair: " + destination)
        plan[destination] = dict(source=source, sha256=C.sha(ROOT / source), previous_sha256=previous)
        if before == raw:
            continue
        if before is not None:
            backup = out / "backup" / destination
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(before)
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
    verified = all(C.sha(target / d) == v["sha256"] for d, v in plan.items()) and C.repaired(target) == C.REPAIRED
    result = dict(status="DEPLOYED_VERIFIED" if verified else "FAILED", target=str(target), changed=len(changes),
                  files=len(plan), tracked_files_touched=0, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({str(p.relative_to(out)): C.sha(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps(result))
    return 0 if verified else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.name, args.target))
