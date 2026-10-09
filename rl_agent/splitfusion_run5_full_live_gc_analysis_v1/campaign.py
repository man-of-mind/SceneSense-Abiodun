"""Run the ONE registered control/candidate pair of the GT GC live qualification, exactly once, and stop.

Before any frame: FREEZE.json binds both prepared registrations/prospective records, the complete runtime source
inventory, this driver and the acceptance file. Per arm, in order control -> candidate:
  root fresh; storage (registered plan) and host MemAvailable floor; exclusive live window (polled up to 10 min,
  else hard blocker, no launch); ONE launch through the retained host lifecycle; the retained actor-driven
  integration verdict (unchanged) and the GC-trace check in separate CPU-only processes; post-run window and leftover
  GT/trampoline processes. Between arms: settle >= 120 s and until the live window is exclusive with loadavg <= 2.0
  (up to 15 min). A control integration STOP, a crash, leftovers or a non-exclusive window stop the campaign
  (completed evidence kept, no retry). No implementation or acceptance change after frames are transmitted.
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

from rl_agent.splitfusion_fixed_action_screen_analysis_v3.campaign import live_window

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOTS = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261006_run5_full_live_gc_v1"
EVIDENCE = MAIN / "experiments/splitfusion_run5_full_live_gc_v1"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
ARMS = ("control", "candidate")
GT_PATTERN = "splitfusion_run5_full_live_gc_v1|splitfusion_exact_gt_live_v2.service|splitfusion_exact_gt_service_proto_v2.worker"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.write(json.dumps(value, sort_keys=True) + "\n")
    log.flush()
    print(json.dumps(value), flush=True)


def mem_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024


def leftovers():
    p = subprocess.run(["pgrep", "-af", GT_PATTERN], capture_output=True, text=True)
    return [l for l in p.stdout.splitlines() if "pgrep" not in l and "campaign" not in l]


def wait_window(log, *, timeout_s, max_load=None, min_wait_s=0):
    t0 = time.monotonic()
    while True:
        w = live_window()
        ok = w["exclusive"] and (max_load is None or w["loadavg_1min"] <= max_load)
        if ok and time.monotonic() - t0 >= min_wait_s:
            return w, True
        if time.monotonic() - t0 > timeout_s:
            return w, False
        time.sleep(30)


def freeze(out):
    from rl_agent.splitfusion_run5_full_live_gc_v1 import contract as C   # main copy == worktree copy (deploy verified)
    record = dict(schema="splitfusion.run5_full_live_gc.freeze.v1", worktree=str(WT),
                  runtime_sources=C.source_inventory(WT),
                  arms={arm: {name: sha(ROOTS / arm / name) for name in
                              ("registration.json", "PROSPECTIVE.json", "PREREQUISITE.json", "ACCEPTANCE.json",
                               "arm.json", "compose_partition.yaml")} for arm in ARMS},
                  campaign_sha256=sha(__file__), gc_check_sha256=sha(Path(__file__).with_name("gc_check.py")),
                  acceptance_source_sha256=sha(EVIDENCE / "ACCEPTANCE.json"),
                  storage_plan_sha256=sha(EVIDENCE / "STORAGE_PLAN.json"), frozen_before_control=True)
    (out / "FREEZE.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=False, exist_ok=False)
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    frozen = freeze(out)
    event(log, kind="frozen", sources=len(frozen["runtime_sources"]))
    plan = json.loads((EVIDENCE / "STORAGE_PLAN.json").read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    results = {}
    for index, arm in enumerate(ARMS):
        root = ROOTS / arm
        reg = json.loads((root / "registration.json").read_bytes())
        if (root / "capture").exists() or (root / "gt_gc_trace").exists() or reg["gt_gc"]["arm"] != arm:
            event(log, kind="stop", arm=arm, reason="ROOT_NOT_FRESH_OR_MISMATCHED")
            return 2
        free = shutil.disk_usage("/").free
        need = plan["required_before_control"] if index == 0 else plan["required_before_candidate"]
        mem = mem_available()
        event(log, kind="preflight", arm=arm, free_bytes=free, required_bytes=need, mem_available=mem,
              leftover_gt=leftovers())
        if free < need or mem < reg["gt_gc"]["safety"]["host_mem_available_min_bytes"] or leftovers():
            event(log, kind="stop", arm=arm, reason="STORAGE_MEMORY_OR_LEFTOVER_PRECONDITION")
            return 2
        if index == 0:
            window, ok = wait_window(log, timeout_s=600)
        else:
            window, ok = wait_window(log, timeout_s=900, max_load=2.0, min_wait_s=120)
        (out / f"window_{arm}.json").write_text(json.dumps(window, indent=1, sort_keys=True))
        event(log, kind="live_window", arm=arm, exclusive=window["exclusive"], loadavg=window["loadavg_1min"],
              gpu_mib=window["gpu_memory_used_mib"], containers=len(window["containers"]), ok=ok)
        if not ok:
            event(log, kind="stop", arm=arm, reason="LIVE_WINDOW_NOT_EXCLUSIVE_HARD_BLOCKER_NOT_LAUNCHED")
            return 2
        token = reg["arms"]["C2"]["execute_token"]
        event(log, kind="launch_start", arm=arm, root=str(root), budget=reg["transmitted_budget"], token=token)
        with open(out / f"launch_{arm}.log", "x") as stream:
            rc = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_run5_full_live_gc_v1.host", "launch",
                                 "--output", str(root), "--execute", token], cwd=WT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT).returncode
        event(log, kind="launch_end", arm=arm, returncode=rc)
        code = ("import json,sys\n"
                "from rl_agent.splitfusion_run5_full_live_analysis_v1.integration import evaluate\n"
                "from rl_agent.splitfusion_run5_full_live_gc_analysis_v1.gc_check import check\n"
                "v = evaluate(sys.argv[1], launch_returncode=int(sys.argv[2]), launch_log=sys.argv[3])\n"
                "v['gc_trace'] = check(sys.argv[1])\n"
                "print(json.dumps(v, default=str, indent=1))\n"
                "sys.exit(0 if v['verdict'] == 'PASS_ACTOR_DRIVEN_INTEGRATION' and v['gc_trace']['passed'] else 3)\n")
        with open(out / f"VERDICT_{arm}.json", "x") as stream, open(out / f"verdict_{arm}.stderr", "x") as err:
            crc = subprocess.run([sys.executable, "-B", "-c", code, str(root), str(rc), str(out / f"launch_{arm}.log")],
                                 cwd=MAIN, env=check_env, stdout=stream, stderr=err).returncode
        left = leftovers()
        after = live_window()
        (out / f"window_after_{arm}.json").write_text(json.dumps(after, indent=1, sort_keys=True))
        event(log, kind="verdict_end", arm=arm, verdict_returncode=crc, leftover_gt=left,
              post_exclusive=after["exclusive"], post_containers=len(after["containers"]))
        results[arm] = crc
        if crc != 0 or left:
            event(log, kind="stop", arm=arm, reason="INTEGRATION_OR_CLEANUP_FAILURE_COMPLETED_EVIDENCE_KEPT_NO_RETRY")
            return 3
    event(log, kind="pair_complete", results=results)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
