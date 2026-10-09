"""Deploy only the prospective v5 host-launcher package; verify sealed v4 in place."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_m3_route_b_v5"
V4_PACKAGE = "rl_agent/splitfusion_m3_route_b_v4"
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v5"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(name, target):
    target = Path(target).resolve(strict=True)
    from rl_agent.splitfusion_m3_route_b_v4 import contract as V4
    if V4.own_sources(ROOT) != V4.own_sources(target):
        raise ValueError("sealed v4 package differs between checkout and deployment worktree")
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    files = {str(path.relative_to(ROOT)): str(path.relative_to(ROOT))
             for path in sorted((ROOT / PACKAGE).glob("*.py"))}
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"], capture_output=True,
                                 text=True, check=True).stdout.split())
    plan, changes = {}, []
    for source, destination in files.items():
        raw = (ROOT / source).read_bytes()
        if destination in tracked:
            raise ValueError("refusing to deploy over a tracked file: " + destination)
        dest = target / destination
        if dest.is_symlink():
            raise ValueError("symlink destination refused: " + destination)
        before = dest.read_bytes() if dest.exists() else None
        plan[destination] = dict(source=source, sha256=sha(raw),
                                 previous_sha256=None if before is None else sha(before))
        if before != raw:
            changes.append((dest, raw))
    stale = sorted(str(path.relative_to(target)) for path in (target / PACKAGE).glob("*.py")
                   if str(path.relative_to(target)) not in plan) if (target / PACKAGE).is_dir() else []
    if stale:
        raise ValueError("stale files in prospective package: " + ", ".join(stale))
    (out / "PLAN.json").write_text(json.dumps(dict(target=str(target), files=plan), indent=1, sort_keys=True) + "\n")
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
    result = dict(status="DEPLOYED_VERIFIED" if verified else "FAILED", target=str(target),
                  changed=len(changes), files=len(plan), tracked_files_touched=0,
                  sealed_v4_unchanged=True, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    manifest = {str(path.relative_to(out)): sha(path.read_bytes()) for path in sorted(out.rglob("*"))
                if path.is_file() and path.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not verified:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    run(args.name, args.target)
