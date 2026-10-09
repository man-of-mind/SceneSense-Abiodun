"""Deploy the untracked M1 fixed-v2 package without touching tracked files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_m1_fixed_live_v2"
EVIDENCE = ROOT / "experiments/splitfusion_m1_baselines_v1"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(name, target):
    target = Path(target).resolve(strict=True)
    base = "rl_agent/splitfusion_run5_full_live_retention_v1"
    here = {str(path.relative_to(ROOT)): sha(path.read_bytes())
            for path in sorted((ROOT / base).glob("*.py"))}
    there = {key: sha((target / key).read_bytes()) if (target / key).is_file() else None
             for key in here}
    if here != there:
        raise ValueError("retained live/retention base differs between checkout and worktree")
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    files = {str(path.relative_to(ROOT)): str(path.relative_to(ROOT))
             for path in sorted((ROOT / PACKAGE).glob("*.py"))}
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"],
        capture_output=True, text=True, check=True).stdout.split())
    plan, changes = {}, []
    for source, destination in files.items():
        raw = (ROOT / source).read_bytes()
        if destination in tracked:
            raise ValueError("refusing to deploy over tracked file: " + destination)
        dest = target / destination
        if dest.is_symlink():
            raise ValueError("symlink destination refused: " + destination)
        before = dest.read_bytes() if dest.exists() else None
        plan[destination] = dict(source=source, sha256=sha(raw),
                                 previous_sha256=None if before is None else sha(before))
        if before != raw:
            changes.append((dest, raw))
    (out / "PLAN.json").write_text(json.dumps(dict(target=str(target), files=plan),
                                                indent=1, sort_keys=True))
    for dest, raw in changes:
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=dest.parent, prefix=".deploy_")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, dest)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
    verified = all(sha((target / destination).read_bytes()) == value["sha256"]
                   for destination, value in plan.items())
    result = dict(status="DEPLOYED_VERIFIED" if verified else "FAILED",
        target=str(target), changed=len(changes), files=len(plan),
        tracked_files_touched=0, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({
        str(path.relative_to(out)): sha(path.read_bytes()) for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps(result))
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    run(args.name, args.target)

