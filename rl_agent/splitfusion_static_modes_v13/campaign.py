"""Static-mode v13 campaign driver: prepare 13 arm prospectives, verify their startup, plan storage, and (only with a
separate written authorization) launch one stage of arms in the frozen order.

  plan            free disk vs the registered launch gate + per-arm peak estimate -> STORAGE_PLAN.json (feasible stage)
  prepare         host.prepare for every arm (create-only prospective roots, frozen order)
  verify-startup  startup_verification for every prepared arm (fresh processes; edge image; GT service)
  launch          REFUSES unless --authorization names this exact campaign plan, the stage, its arms in frozen order and
                  ``explicit_live_launch_approved: true``.  Per arm: clean host, disk >= gate + peak, startup proof
                  re-verified; numeric host rc captured; outcome-aware assessment written; stop (no retry) on
                  INFRASTRUCTURE_FAILURE / INVALID_IDENTITY / unclean host / storage gate.  Never changes transport
                  limits, never retries, never tunes.
Run from the deployment worktree with ``env -u PYTHONPATH``.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
MAIN_CHECKOUT = ROOT.parent / "abiodun"          # offline assessment runs from the main checkout (as V12's did)
CAMPAIGN = "20261009_static_modes_v13"
CAPTURE_PARENT = ROOT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1" / CAMPAIGN
EVIDENCE = ROOT / "experiments/splitfusion_static_modes_v13"
REGISTERED_GATE_BYTES = 16_000_000_000          # registration profiler.minimum_free_disk_bytes (runner_v1 enforces)
PER_ARM_PEAK_BYTES = 2_500_000_000              # storage plan: measured V11/V12 rates scaled to 1,000 frames + margin
AUTH_SCHEMA = "splitfusion.static_modes_v13.launch_authorization.v1"
NEEDLES = ("CarlaUE4", "nr-softmodem", "nr-uesoftmodem", "splitfusion_exact_gt_live_v2.service", "child_v1_",
           "edge_profiled_v1_", "phase6_live_child", "runner_cpu_partition_v1")
EDGE_IMAGE = "sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def free_bytes(path=ROOT):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def host_clean():
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and str(os.getpid()) != path.parent.name and any(n in command for n in NEEDLES):
            found.append(dict(pid=int(path.parent.name), command=command[:300]))
    containers = subprocess.run(["sudo", "-n", "docker", "ps", "-q", "--filter", f"ancestor={EDGE_IMAGE}"],
                                capture_output=True, text=True).stdout.split()
    return dict(clean=not found and not containers, processes=found, edge_containers=containers)


def arm_root(arm):
    return CAPTURE_PARENT / arm


def plan(out):
    free = free_bytes()
    feasible, projected = [], free
    for arm in C.RUN_ORDER:
        if projected >= REGISTERED_GATE_BYTES + PER_ARM_PEAK_BYTES:
            feasible.append(arm)
            projected -= PER_ARM_PEAK_BYTES
        else:
            break
    remaining = [a for a in C.RUN_ORDER if a not in feasible]
    document = dict(
        schema="splitfusion.static_modes_v13.storage_plan.v1", measured_free_bytes=free, measured_at_ns=time.time_ns(),
        registered_gate_bytes=REGISTERED_GATE_BYTES, per_arm_peak_bytes=PER_ARM_PEAK_BYTES,
        per_arm_launch_condition="free >= registered gate + per-arm peak, re-measured before EVERY arm",
        first_stage=feasible, remaining=remaining,
        bytes_needed_to_run_remaining_without_offload=max(0, REGISTERED_GATE_BYTES + PER_ARM_PEAK_BYTES * len(remaining)
                                                          - projected),
        run_order=list(C.RUN_ORDER), run_order_rule=C.RUN_ORDER_RULE, deletion_authorized=False)
    out.mkdir(parents=True, exist_ok=False)
    (out / "STORAGE_PLAN.json").write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(json.dumps(document, indent=1, sort_keys=True))
    return document


def prepare(proofs):
    from . import host as H
    CAPTURE_PARENT.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in C.RUN_ORDER:
        results[arm] = H.prepare(arm_root(arm), arm=arm, **proofs)
        print(json.dumps(dict(arm=arm, status=results[arm]["status"], token=results[arm]["token"])), flush=True)
    return results


def verify_startup(name_prefix):
    results = {}
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                         "MKL_NUM_THREADS")}
    for arm in C.RUN_ORDER:
        name = f"{name_prefix}_{arm}"
        proc = subprocess.run(["/usr/bin/python3", "-B", "-m", C.MODULE + ".startup_verification", "run", "--name", name,
                               "--prospective", str(arm_root(arm))], cwd=ROOT, env=env, capture_output=True, text=True)
        (EVIDENCE / f"{name}.stdout").write_text(proc.stdout + proc.stderr)
        results[arm] = dict(returncode=proc.returncode, proof=str(EVIDENCE / name / "RESULT.json"))
        print(json.dumps(dict(arm=arm, returncode=proc.returncode)), flush=True)
    return results


def _authorization(path, stage_arms, storage_plan):
    value = json.loads(Path(path).read_bytes())
    if (value.get("schema") != AUTH_SCHEMA or value.get("explicit_live_launch_approved") is not True
            or not value.get("approved_by") or value.get("campaign") != CAMPAIGN
            or value.get("arms") != list(stage_arms) or value.get("storage_plan_sha256") != sha(storage_plan)):
        raise SystemExit("launch refused: no matching written authorization for this campaign stage")
    order = [a for a in C.RUN_ORDER if a in stage_arms]
    if order != list(stage_arms):
        raise SystemExit("launch refused: stage arms are not in the frozen run order")
    return value


def launch(authorization, storage_plan, stage_arms):
    from . import host as H
    _authorization(authorization, stage_arms, storage_plan)
    log = []
    for arm in stage_arms:
        root = arm_root(arm)
        proof = EVIDENCE / f"startup_{CAMPAIGN}_{arm}" / "RESULT.json"
        record = dict(arm=arm, started_wall_ns=time.time_ns())
        clean = host_clean()
        free = free_bytes()
        record.update(host_before=clean, free_before=free)
        if not clean["clean"]:
            record["stop"] = "STOP_HOST_NOT_CLEAN_BEFORE_ARM"
        elif free < REGISTERED_GATE_BYTES + PER_ARM_PEAK_BYTES:
            record["stop"] = "STOP_STORAGE_GATE"
        if "stop" in record:
            log.append(record)
            break
        H.verify(root, proof)
        token = C.token(arm)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        with (root / "campaign_launch.log").open("x") as stream:
            proc = subprocess.run(["/usr/bin/python3", "-B", "-m", C.HOST, "launch", "--output", str(root),
                                   "--execute", token, "--startup-proof", str(proof)], cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT)
        (root / "campaign_launch.rc").write_text(f"{proc.returncode}\n")
        record.update(launch_rc=proc.returncode, finished_wall_ns=time.time_ns())
        assessment = root / "ASSESSMENT_V13.json"
        sub = subprocess.run(["/usr/bin/python3", "-B", "-m", C.OFFLINE_ASSESSMENT_MODULE, "--root", str(root),
                              "--launch-rc", str(proc.returncode), "--output", str(assessment)],
                             cwd=MAIN_CHECKOUT, env=env, capture_output=True, text=True)
        verdict = json.loads(assessment.read_bytes())["verdict"] if assessment.exists() else "ASSESSMENT_ERROR"
        record.update(assessment=str(assessment), verdict=verdict, assessment_stderr=sub.stderr[-800:])
        after = host_clean()
        record["host_after"] = after
        log.append(record)
        if verdict != "VALID_MEASUREMENT":
            record["stop"] = "STOP_" + verdict
            break
        if not after["clean"]:
            record["stop"] = "STOP_HOST_NOT_CLEAN_AFTER_ARM"
            break
    out = EVIDENCE / f"campaign_stage_{int(time.time())}.json"
    out.write_text(json.dumps(dict(schema="splitfusion.static_modes_v13.stage_record.v1", arms=list(stage_arms),
                                   records=log, no_retry=True), indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(log, indent=1, default=str))
    return log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("prepare")
    from .host import PROOF_ARGS
    for name in PROOF_ARGS:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p = sub.add_parser("verify-startup")
    p.add_argument("--name-prefix", default=f"startup_{CAMPAIGN}")
    p = sub.add_parser("launch")
    p.add_argument("--authorization", type=Path, required=True)
    p.add_argument("--storage-plan", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True, choices=C.ARM_KEYS)
    args = parser.parse_args(argv)
    if args.op == "plan":
        plan(args.out)
    elif args.op == "prepare":
        prepare({name: getattr(args, name) for name in PROOF_ARGS})
    elif args.op == "verify-startup":
        verify_startup(args.name_prefix)
    else:
        launch(args.authorization, args.storage_plan, args.arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
