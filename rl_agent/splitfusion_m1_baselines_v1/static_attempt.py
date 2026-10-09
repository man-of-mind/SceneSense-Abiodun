"""Run the one separately sealed M1 static attempt; never retries or runs agent."""
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
from rl_agent.splitfusion_m1_baselines_v1.campaign import (
    leftovers,
    mem_available,
    retention_postflight,
    static_verdict,
)

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ATTEMPT = (WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
           "20261007_m1_baselines_v1/static_attempt_02")
EVIDENCE = MAIN / "experiments/splitfusion_m1_baselines_v1"
PREFLIGHT = EVIDENCE / "preflight_static_02/RESULT.json"
HOST = "rl_agent.splitfusion_m1_static_live_v1.host"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
REQUIRED_FREE = 6_442_450_944 + 2_147_483_648 + 4_294_967_296


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=str)
        stream.write("\n")


def event(stream, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = json.dumps(value, sort_keys=True, default=str)
    stream.write(line + "\n")
    stream.flush()
    print(line, flush=True)


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=False, exist_ok=False)
    log = open(out / "ATTEMPT.jsonl", "x", buffering=1)
    reg_path = ATTEMPT / "registration.json"
    reg = json.loads(reg_path.read_bytes())
    preflight = json.loads(PREFLIGHT.read_bytes())
    agent_root = (WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                  "20261007_m1_baselines_v1/agent")
    failed_root = ATTEMPT.parent / "static"
    freshness_names = ("capture", "gt_gc_trace", "paired_retention_v2", "RETENTION_RESULT.json",
                       "controller_events.jsonl")
    free = shutil.disk_usage("/").free
    memory = mem_available()
    left = leftovers()
    window = live_window()
    frozen = dict(
        schema="splitfusion.m1_baselines_v1.static_attempt_freeze.v2",
        attempt_id="M1_STATIC_ATTEMPT_02_IMPORT_SAFE_VERIFIER",
        attempt_root=str(ATTEMPT),
        registration_sha256=sha(reg_path),
        registration_identity_sha256=reg["registration_sha256"],
        preflight_sha256=sha(PREFLIGHT),
        preflight_status=preflight.get("status"),
        preflight_registration_matches=preflight.get("registration_sha256") == sha(reg_path),
        actor_weights_sha256=reg["policy"]["expected_weights_sha256"],
        frame_budget=reg["transmitted_budget"],
        route=reg["network_profile"],
        retention_plan_sha256=reg["paired_retention"]["plan_sha256"],
        fixed_action=reg["external_fixed_action"],
        selection_semantics=reg["selection_semantics"],
        zero_completion=reg["zero_completion_semantics"],
        valid_agent_preserved=dict(root=str(agent_root), registration_sha256=sha(agent_root / "registration.json")),
        failed_static_preserved=dict(root=str(failed_root), registration_sha256=sha(failed_root / "registration.json")),
        fresh=not any((ATTEMPT / item).exists() for item in freshness_names),
        disk_free_bytes=free,
        required_free_bytes=REQUIRED_FREE,
        mem_available_bytes=memory,
        leftovers=left,
        live_window=window,
        no_agent_rerun=True,
        no_retry=True,
    )
    write(out / "FREEZE.json", frozen)
    event(log, kind="prelaunch", fresh=frozen["fresh"], disk_free=free,
          required_free=REQUIRED_FREE, memory=memory, leftovers=left,
          exclusive=window.get("exclusive"), loadavg=window.get("loadavg_1min"),
          gpu_mib=window.get("gpu_memory_used_mib"), preflight=preflight.get("status"))
    if not (frozen["fresh"] and free >= REQUIRED_FREE and memory >= 4 * 2**30
            and not left and window.get("exclusive") and window.get("loadavg_1min", 99) <= 2.0
            and preflight.get("status") == "PASS" and frozen["preflight_registration_matches"]):
        event(log, kind="stop_before_launch", reason="PRELAUNCH_GUARD_FAILED")
        return 2

    env = {key: value for key, value in os.environ.items() if key not in STRIP}
    token = reg["arms"]["C2"]["execute_token"]
    event(log, kind="launch_start", token=token, budget=reg["transmitted_budget"])
    with open(out / "launch_static.log", "x") as stream:
        rc = subprocess.run([sys.executable, "-B", "-m", HOST, "launch",
            "--output", str(ATTEMPT), "--execute", token], cwd=WT, env=env,
            stdout=stream, stderr=subprocess.STDOUT).returncode
    event(log, kind="launch_end", returncode=rc)

    try:
        post = retention_postflight(ATTEMPT, out, "static")
        event(log, kind="retention_postflight", manifest_matches=post["manifest_matches_child"],
              gt=post.get("gt_references"))
    except Exception as exc:
        post = None
        event(log, kind="retention_postflight_error", error=f"{type(exc).__name__}: {exc}")
    try:
        verdict = static_verdict(ATTEMPT, rc)
    except Exception as exc:
        verdict = dict(schema="splitfusion.m1_static_live_v1.verdict.v1",
            verdict="STOP_INVALID_STATIC_MEASUREMENT", passed=False,
            launch_returncode=rc, evidence_collection_status="FAILED",
            error=f"{type(exc).__name__}: {exc}")
    write(out / "VERDICT_static.json", verdict)
    after = live_window()
    left_after = leftovers()
    write(out / "WINDOW_AFTER.json", after)
    event(log, kind="verdict", verdict=verdict.get("verdict"), passed=verdict.get("passed"),
          postflight=post is not None, leftovers=left_after, exclusive=after.get("exclusive"))
    files = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*"))
             if path.is_file() and path.name != "OUTPUT_SHA256.json"}
    write(out / "OUTPUT_SHA256.json", files)
    return 0 if verdict.get("passed") and post is not None and not left_after and after.get("exclusive") else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
