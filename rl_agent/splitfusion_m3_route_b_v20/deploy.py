"""Deploy the v20 package, its sampler-visible child/edge trampolines, the display-only live-view tap package and its
map trampoline, and the registered offline assessment package;
verify the sealed v3-v11 packages (incl. the v11 offline assessment), the deployed V1/V2/V3.v2 storage-accounting
packages, the prep FREEZE lifetime repair and the tilt-aware AVO candidate in place (byte-identical between the
checkout and the deployment worktree).  Nothing outside the v20 destinations is written.

Deployment refuses unless every registered runtime file matches its qualified digest.

Usage (main checkout): python3 -m rl_agent.splitfusion_m3_route_b_v20.deploy --name <dir> --target <worktree>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_m3_route_b_v20"
SEALED = ("rl_agent/splitfusion_m3_route_b_v19", "rl_agent/splitfusion_m3_route_b_v19_offline_v1",
          "rl_agent/splitfusion_m3_route_b_v17", "rl_agent/splitfusion_m3_route_b_v17_offline_v1",
          "rl_agent/splitfusion_gt_geometry_containment_v1", "rl_agent/splitfusion_m3_route_b_v3", "rl_agent/splitfusion_m3_route_b_v4", "rl_agent/splitfusion_m3_route_b_v5",
          "rl_agent/splitfusion_m3_route_b_v6", "rl_agent/splitfusion_m3_route_b_v7",
          "rl_agent/splitfusion_m3_route_b_v8", "rl_agent/splitfusion_m3_route_b_v9",
          "rl_agent/splitfusion_m3_route_b_v10", "rl_agent/splitfusion_m3_route_b_v11",
          "rl_agent/splitfusion_m3_route_b_v11_offline_v1",
          "rl_agent/splitfusion_m3_route_b_v16", "rl_agent/splitfusion_m3_route_b_v16_offline_v1",
          "rl_agent/splitfusion_gt_freeze_lifetime_v1", "rl_agent/splitfusion_avo_tilt_v1",
          "rl_agent/splitfusion_gt_storage_accounting_v1", "rl_agent/splitfusion_gt_storage_accounting_v2",
          "rl_agent/splitfusion_gt_storage_accounting_v3")
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v20"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _package(root, package):
    return {p.name: sha(p.read_bytes()) for p in sorted((root / package).glob("*.py"))}


def run(name, target):
    from . import contract as C
    target = Path(target).resolve(strict=True)
    for package in SEALED:
        if _package(ROOT, package) != _package(target, package):
            raise ValueError("sealed package differs between checkout and deployment worktree: " + package)
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p.relative_to(ROOT)): str(p.relative_to(ROOT)) for p in sorted((ROOT / PACKAGE).glob("*.py"))}
    files.update(C.SHIMS)
    files.update({str(p.relative_to(ROOT)): str(p.relative_to(ROOT))
                  for p in sorted((ROOT / C.OFFLINE_ASSESSMENT_PACKAGE).glob("*.py"))})
    # the reviewed GT containment package (every file, at its reviewed digest; refused otherwise)
    containment = {str(p.relative_to(ROOT)): str(p.relative_to(ROOT))
                   for p in sorted((ROOT / C.CONTAINMENT_PACKAGE).glob("*.py"))}
    if {k: sha((ROOT / k).read_bytes()) for k in containment} != C.CONTAINMENT_QUALIFIED:
        raise ValueError("checkout GT containment sources are not the reviewed ones")
    # Qualified containment is already deployed and was compared as SEALED; never overwrite it.
    # v20: the display-only live-view tap package and its map trampoline (deployed into the profiler directory)
    files.update({str(p.relative_to(ROOT)): str(p.relative_to(ROOT))
                  for p in sorted((ROOT / C.LIVE_VIEW_PACKAGE).glob("*.py"))})
    files[C.LIVE_VIEW_TRAMPOLINE_SOURCE + "#trampoline"] = C.MAP_SERVER_TRAMPOLINE
    for relative, digest in {**C.PREP_LIFETIME_QUALIFIED, **C.AVO_TILT_QUALIFIED, **C.ACCOUNTING_QUALIFIED,
                             **C.ACCOUNTING_V1_QUALIFIED, **C.ACCOUNTING_V2_QUALIFIED}.items():
        if sha((ROOT / relative).read_bytes()) != digest or sha((target / relative).read_bytes()) != digest:
            raise ValueError("checkout or deployed source differs from the qualified source: " + relative)
    tracked = set(subprocess.run(["git", "-C", str(target), "ls-files"], capture_output=True, text=True,
                                 check=True).stdout.split())
    plan, changes = {}, []
    for source, destination in files.items():
        raw = (ROOT / source.split("#")[0]).read_bytes()
        if destination in tracked:
            raise ValueError("refusing to deploy over a tracked file: " + destination)
        dest = target / destination
        if dest.is_symlink():
            raise ValueError("symlink destination refused: " + destination)
        before = dest.read_bytes() if dest.exists() else None
        plan[destination] = dict(source=source.split("#")[0], sha256=sha(raw),
                                 previous_sha256=None if before is None else sha(before))
        if before != raw:
            changes.append((dest, raw))
    stale = sorted(str(p.relative_to(target))
                   for package in (PACKAGE, "rl_agent/splitfusion_m3_route_b_v20_offline_v1", C.LIVE_VIEW_PACKAGE)
                   if (target / package).is_dir()
                   for p in (target / package).glob("*.py") if str(p.relative_to(target)) not in plan)
    if stale:
        raise ValueError("stale files in the deployed package: " + ", ".join(stale))
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
    verified = all(sha((target / destination).read_bytes()) == value["sha256"] for destination, value in plan.items())
    result = dict(status="DEPLOYED_VERIFIED" if verified else "FAILED", target=str(target), changed=len(changes),
                  files=len(plan), tracked_files_touched=0, sealed_packages_unchanged=list(SEALED),
                  live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({str(p.relative_to(out)): sha(p.read_bytes())
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1) + "\n")
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
