"""Launch one prepared full Run-5 live run exactly once (100-frame diagnostic or 1,000-frame validation) and judge it.

Per run: exclusive live-window check (recorded; a non-exclusive window is a hard blocker, no launch); one launch
through the retained host lifecycle; the actor-driven integration verdict (``integration.evaluate``) in a separate
CPU-only process. No retry. Create-only evidence under experiments/splitfusion_run5_full_live_v1/<name>.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rl_agent.splitfusion_fixed_action_screen_analysis_v3.campaign import live_window

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOTS = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261006_run5_full_live_v1"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.write(json.dumps(value, sort_keys=True) + "\n")
    log.flush()
    print(json.dumps(value), flush=True)


def run(name, root_name):
    out = MAIN / "experiments/splitfusion_run5_full_live_v1" / name
    out.mkdir(parents=True, exist_ok=False)
    arm = ROOTS / root_name
    reg = json.loads((arm / "registration.json").read_bytes())
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    if (arm / "capture").exists():
        event(log, kind="stop", reason="ROOT_NOT_FRESH")
        return 2
    window = live_window()
    (out / "window.json").write_text(json.dumps(window, indent=1, sort_keys=True))
    event(log, kind="live_window", exclusive=window["exclusive"], loadavg=window["loadavg_1min"],
          gpu_mib=window["gpu_memory_used_mib"], containers=len(window["containers"]), process_hits=len(window["process_hits"]))
    if not window["exclusive"]:
        event(log, kind="stop", reason="LIVE_WINDOW_NOT_EXCLUSIVE_HARD_BLOCKER_NOT_LAUNCHED")
        return 2
    token = reg["arms"]["C2"]["execute_token"]
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    event(log, kind="launch_start", root=str(arm), budget=reg["transmitted_budget"], token=token)
    with open(out / "launch.log", "x") as stream:
        rc = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_run5_full_live_v1.host", "launch", "--output",
                             str(arm), "--execute", token], cwd=WT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    event(log, kind="launch_end", returncode=rc)
    code = ("import json,sys\n"
            "from rl_agent.splitfusion_run5_full_live_analysis_v1.integration import evaluate\n"
            "v = evaluate(sys.argv[1], launch_returncode=int(sys.argv[2]), launch_log=sys.argv[3])\n"
            "print(json.dumps(v, default=str, indent=1))\n"
            "sys.exit(0 if v['verdict'] == 'PASS_ACTOR_DRIVEN_INTEGRATION' else 3)\n")
    check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with open(out / "VERDICT.json", "x") as stream, open(out / "verdict.stderr", "x") as err:
        crc = subprocess.run([sys.executable, "-B", "-c", code, str(arm), str(rc), str(out / "launch.log")], cwd=MAIN,
                             env=check_env, stdout=stream, stderr=err).returncode
    event(log, kind="verdict_end", verdict_returncode=crc)
    window_after = live_window()
    (out / "window_after.json").write_text(json.dumps(window_after, indent=1, sort_keys=True))
    event(log, kind="post_run_window", exclusive=window_after["exclusive"], containers=len(window_after["containers"]),
          process_hits=len(window_after["process_hits"]))
    return crc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.name, args.root))
