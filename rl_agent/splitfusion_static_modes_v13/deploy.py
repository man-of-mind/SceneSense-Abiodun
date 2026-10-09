"""Deploy the v13 package (+ FIXED_IDENTITIES.json), its two sampler-visible trampolines and its offline assessment
package into the deployment worktree; verify the sealed v3-v12 packages and qualified repair/AVO/accounting sources
are byte-identical between the checkout and the worktree.  Nothing outside the v13 destinations is written and no
tracked file is touched.  (Copy of the v12 deploy with the v13 destinations.)

  python3 -m rl_agent.splitfusion_static_modes_v13.deploy --name <dir> --target <worktree>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_static_modes_v13"
OFFLINE = "rl_agent/splitfusion_static_modes_v13_offline_v1"
SEALED = ("rl_agent/splitfusion_m3_route_b_v3", "rl_agent/splitfusion_m3_route_b_v4", "rl_agent/splitfusion_m3_route_b_v5",
          "rl_agent/splitfusion_m3_route_b_v6", "rl_agent/splitfusion_m3_route_b_v7",
          "rl_agent/splitfusion_m3_route_b_v8", "rl_agent/splitfusion_m3_route_b_v9",
          "rl_agent/splitfusion_m3_route_b_v10", "rl_agent/splitfusion_m3_route_b_v11",
          "rl_agent/splitfusion_m3_route_b_v11_offline_v1", "rl_agent/splitfusion_m3_route_b_v12",
          "rl_agent/splitfusion_m3_route_b_v12_offline_v1",
          "rl_agent/splitfusion_gt_freeze_lifetime_v1", "rl_agent/splitfusion_avo_tilt_v1",
          "rl_agent/splitfusion_gt_storage_accounting_v1", "rl_agent/splitfusion_gt_storage_accounting_v2",
          "rl_agent/splitfusion_gt_storage_accounting_v3", "rl_agent/splitfusion_live_dispatch_v1",
          "rl_agent/splitfusion_hybrid_sac_live_route_b_v2", "rl_agent/splitfusion_run5_full_live_v1")
EVIDENCE = ROOT / "experiments/splitfusion_static_modes_v13"


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
    for relative in ("rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
                     "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json", C.LIVE_CONFIG):
        if sha((ROOT / relative).read_bytes()) != sha((target / relative).read_bytes()):
            raise ValueError("pinned input differs between checkout and deployment worktree: " + relative)
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p.relative_to(ROOT)): str(p.relative_to(ROOT)) for p in sorted((ROOT / PACKAGE).glob("*.py"))}
    files[C.FIXED_BUNDLES_FILE] = C.FIXED_BUNDLES_FILE
    files.update(C.SHIMS)
    files.update({str(p.relative_to(ROOT)): str(p.relative_to(ROOT)) for p in sorted((ROOT / OFFLINE).glob("*.py"))})
    for relative, digest in {**C.PREP_LIFETIME_QUALIFIED, **C.AVO_TILT_QUALIFIED, **C.ACCOUNTING_QUALIFIED,
                             **C.ACCOUNTING_V1_QUALIFIED, **C.ACCOUNTING_V2_QUALIFIED}.items():
        if sha((ROOT / relative).read_bytes()) != digest or sha((target / relative).read_bytes()) != digest:
            raise ValueError("checkout or deployed source differs from the qualified source: " + relative)
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
        if before != raw:
            changes.append((dest, raw))
    stale = sorted(str(p.relative_to(target)) for package in (PACKAGE, OFFLINE) if (target / package).is_dir()
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
