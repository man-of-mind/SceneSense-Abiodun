"""Deploy this package and its two trampolines into the live worktree (new untracked files only).

Only the pilot live package (incl. its actor export), its two trampolines and the restricted-pilot support module
(``splitfusion_run5_restricted_pilot_v1/{__init__,support}.py``, which defines the actor class) are written. The integrated early-GT pipeline packages are NOT
redeployed: they must already be byte-identical between this checkout and the worktree (verified, else refused).
Usage (main checkout): python -m rl_agent.splitfusion_run5_pilot_live_v1.deploy --name <dir> --target <worktree>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_run5_pilot_live_v1"
PACKAGES = (PACKAGE,)
EXTRA = {f"{PACKAGE}/actor_export/{name}": f"{PACKAGE}/actor_export/{name}"
         for name in ("ACTOR_BINDING.json", "actor_state_dict.pt", "COMMITTED")}
EXTRA.update({f"rl_agent/splitfusion_run5_restricted_pilot_v1/{name}": f"rl_agent/splitfusion_run5_restricted_pilot_v1/{name}"
              for name in ("__init__.py", "support.py")})
EXTRA.update({f"rl_agent/splitfusion_ue_input_prep_v2/{name}": f"rl_agent/splitfusion_ue_input_prep_v2/{name}"
              for name in ("__init__.py", "candidate.py")})
PIPELINE = ("rl_agent/splitfusion_exact_gt_live_v2", "rl_agent/splitfusion_exact_gt_service_proto_v2",
            "rl_agent/splitfusion_fixed_action_screen_v2", "rl_agent/splitfusion_fixed_action_screen_v3",
            "rl_agent/splitfusion_run5_live_avo_v1")
EVIDENCE = ROOT / "experiments/splitfusion_run5_pilot_live_v1"
SHIMS = {f"{PACKAGE}/runtime_shims/child_v1_run5_pilot_live_v1.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/child_v1_run5_pilot_live_v1.py",
         f"{PACKAGE}/runtime_shims/edge_profiled_v1_run5_pilot_live_v1.py":
         "rl_agent/splitfusion_single_gpu_profiler_v1/edge_profiled_v1_run5_pilot_live_v1.py"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(name, target):
    target = Path(target).resolve(strict=True)
    for package in PIPELINE:                 # unchanged pipeline: identical, never redeployed
        here = {p.name: sha(p.read_bytes()) for p in sorted((ROOT / package).glob("*.py"))}
        there = {p.name: sha(p.read_bytes()) for p in sorted((target / package).glob("*.py"))}
        if here != there:
            raise ValueError("pipeline package differs between checkout and worktree: " + package)
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p.relative_to(ROOT)): str(p.relative_to(ROOT)) for package in PACKAGES
             for p in sorted((ROOT / package).glob("*.py"))}
    files.update(SHIMS)
    files.update(EXTRA)
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"], capture_output=True, text=True,
                                 check=True).stdout.split())
    plan, changes = {}, []
    for source, destination in files.items():
        raw = (ROOT / source).read_bytes()
        if destination in tracked:
            raise ValueError("refusing to deploy over a tracked file: " + destination)
        dest = target / destination
        if dest.is_symlink():
            raise ValueError("symlink destination refused: " + destination)
        before = dest.read_bytes() if dest.exists() else None
        plan[destination] = dict(source=source, sha256=sha(raw), previous_sha256=None if before is None else sha(before))
        if before == raw:
            continue
        if before is not None:
            backup = out / "backup" / destination
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(before)
        changes.append((dest, raw))
    stale = sorted(str(p.relative_to(target)) for package in PACKAGES if (target / package).is_dir()
                   for p in (target / package).glob("*.py") if str(p.relative_to(target)) not in plan)
    if stale:
        raise ValueError("stale files in the deployed package: " + ", ".join(stale))
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
    verified = all(sha((target / d).read_bytes()) == v["sha256"] for d, v in plan.items())
    result = dict(status="DEPLOYED_VERIFIED" if verified else "FAILED", target=str(target), changed=len(changes),
                  files=len(plan), tracked_files_touched=0, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({str(p.relative_to(out)): sha(p.read_bytes())
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps(result))
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    run(args.name, args.target)
