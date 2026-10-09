"""Three-arm mode-11 frontier campaign (D10): each prepared arm launched once, registered order, no retries.

Per arm:
  1. exclusive live-window check (recorded): no CARLA/OAI process or container from anyone, GPU memory and load
     below registered bounds. A failed check before an arm is a hard blocker -> stop (no launch).
  2. one launch through the retained host lifecycle (``splitfusion_fixed_action_screen_v3.host launch``).
  3. integration (v3: v2 I1-I11 + strict I11 + I12, exact re-scoring) and the v3 forced-screen verdict
     (amendment 02 carried forward) in a separate CPU-only process.
Anything but PASS_FORCED_SCREEN is a genuine integration/safety stop: no further arm, no retry. Deadline misses,
poor quality and low coverage are outcomes, never stop reasons. Create-only evidence under
experiments/splitfusion_fixed_action_screen_v3/<name>.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_m11_frontier_v3"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
WINDOW = dict(max_gpu_memory_mib=4096, max_loadavg_1min=6.0,
              forbidden_process_patterns=("CarlaUnreal", "CarlaUE", "nr-softmodem", "nr-uesoftmodem", "lte-softmodem"))


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.write(json.dumps(value, sort_keys=True) + "\n")
    log.flush()
    print(json.dumps(value), flush=True)


def live_window():
    """Exclusive-window evidence; never kills or touches anything."""
    def run(cmd):
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, p.stdout.strip()
    _, containers = run(["docker", "ps", "--format", "{{.Names}}"])
    _, procs = run(["ps", "-eo", "user,pid,comm,args"])
    hits = [l for l in procs.splitlines() if any(k in l for k in WINDOW["forbidden_process_patterns"]) and "grep" not in l]
    _, gpu = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    _, apps = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    load = float(Path("/proc/loadavg").read_text().split()[0])
    gpu_used = max((int(x) for x in gpu.split()), default=0) if gpu else None
    ok = (not containers and not hits and gpu_used is not None and gpu_used <= WINDOW["max_gpu_memory_mib"]
          and load <= WINDOW["max_loadavg_1min"])
    return dict(exclusive=ok, containers=containers.splitlines(), process_hits=hits, gpu_memory_used_mib=gpu_used,
                gpu_compute_apps=apps.splitlines(), loadavg_1min=load, bounds=WINDOW)


def run(name, arms):
    out = MAIN / "experiments/splitfusion_fixed_action_screen_v3" / name
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    event(log, kind="campaign_start", arms=arms, screen=str(SCREEN))
    stopped, completed = None, []
    for index, key in enumerate(arms):
        arm = SCREEN / key
        reg = json.loads((arm / "registration.json").read_bytes())
        if reg.get("arm_key") != key or reg.get("arm_index") != index or (arm / "capture").exists():
            stopped = dict(arm=key, reason="ARM_ROOT_NOT_FRESH_OR_MISMATCHED")
            break
        window = live_window()
        (out / f"window_{key}.json").write_text(json.dumps(window, indent=1, sort_keys=True))
        event(log, kind="live_window", arm=key, exclusive=window["exclusive"], loadavg=window["loadavg_1min"],
              gpu_mib=window["gpu_memory_used_mib"], containers=len(window["containers"]), process_hits=len(window["process_hits"]))
        if not window["exclusive"]:
            stopped = dict(arm=key, reason="LIVE_WINDOW_NOT_EXCLUSIVE_HARD_BLOCKER_NOT_LAUNCHED")
            break
        token = reg["arms"]["C2"]["execute_token"]
        event(log, kind="launch_start", arm=key, index=index, forced=reg["forced_action"], token=token)
        launch_log = out / f"launch_{key}.log"
        with open(launch_log, "x") as stream:
            rc = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_fixed_action_screen_v3.host", "launch",
                                 "--output", str(arm), "--execute", token], cwd=WT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT).returncode
        event(log, kind="launch_end", arm=key, returncode=rc)
        code = ("import json,sys\n"
                "from rl_agent.splitfusion_fixed_action_screen_analysis_v3.integration import Arm, integration\n"
                "from rl_agent.splitfusion_fixed_action_screen_analysis_v3.verdict import collect, evaluate\n"
                "arm, rc, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
                "try:\n"
                "    integ = integration(Arm(arm), None, do_rescore=True)\n"
                "except Exception as error:\n"
                "    integ = dict(error=f'{type(error).__name__}: {error}')\n"
                "verdict = evaluate(collect(arm, launch_returncode=rc, launch_log=log, integration_result=integ))\n"
                "print(json.dumps(dict(integration=integ, forced_screen_verdict=verdict), default=str, indent=1))\n"
                "sys.exit(0 if verdict['verdict'] == 'PASS_FORCED_SCREEN' else 3)\n")
        with open(out / f"verdict_{key}.json", "x") as stream, open(out / f"verdict_{key}.stderr", "x") as err:
            crc = subprocess.run([sys.executable, "-B", "-c", code, str(arm), str(rc), str(launch_log)], cwd=MAIN,
                                 env=check_env, stdout=stream, stderr=err).returncode
        event(log, kind="verdict_end", arm=key, verdict_returncode=crc)
        completed.append(key)
        if crc != 0:
            stopped = dict(arm=key, launch_returncode=rc, verdict_returncode=crc,
                           reason="FORCED_SCREEN_VERDICT_STOP" if crc == 3 else "VERDICT_EVALUATION_CRASHED")
            break
    result = dict(schema="splitfusion.fixed_action_screen_v3.campaign.v1", order=arms, stopped=stopped,
                  completed_arms=completed, no_retry=True)
    (out / "CAMPAIGN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    event(log, kind="campaign_end", stopped=stopped)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    run(args.name, args.arms)
