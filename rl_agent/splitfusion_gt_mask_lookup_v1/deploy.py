"""Deploy the mask-lookup candidate into the live worktree (untracked only, backup kept, verified).
Usage (main): python -m rl_agent.splitfusion_gt_mask_lookup_v1.deploy --name deploy_01 --target <worktree>
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from . import contract as MK

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_gt_mask_lookup_v1"


def run(name, target):
    target = Path(target).resolve(strict=True)
    out = ROOT / MK.EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    MK.repaired(ROOT)
    files = list(MK.ORIGINAL) + [str(p.relative_to(ROOT)) for p in sorted((ROOT / PACKAGE).glob("*.py"))]
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"], capture_output=True, text=True, check=True).stdout.split())
    plan = {}
    for rel in files:
        dest, raw = target / rel, (ROOT / rel).read_bytes()
        if rel in tracked or dest.is_symlink():
            raise ValueError("refusing tracked/symlink destination: " + rel)
        previous = MK.sha(dest) if dest.exists() else None
        if rel in MK.ORIGINAL and previous not in (MK.ORIGINAL[rel], MK.REPAIRED[rel]):
            raise ValueError("worktree file is neither the registered original nor the candidate: " + rel)
        plan[rel] = dict(sha256=MK.sha(ROOT / rel), previous_sha256=previous)
        if dest.exists() and dest.read_bytes() == raw:
            continue
        if dest.exists():
            backup = out / "backup" / rel; backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(dest.read_bytes())
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".deploy_")
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(tmp, 0o644); os.replace(tmp, dest)
    ok = all(MK.sha(target / r) == v["sha256"] for r, v in plan.items()) and MK.repaired(target) == MK.REPAIRED
    result = dict(status="DEPLOYED_VERIFIED" if ok else "FAILED", files=len(plan), tracked_files_touched=0,
                  live_services_started=False)
    (out / "PLAN.json").write_text(json.dumps(plan, indent=1, sort_keys=True))
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result))
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True); parser.add_argument("--target", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.name, args.target))
