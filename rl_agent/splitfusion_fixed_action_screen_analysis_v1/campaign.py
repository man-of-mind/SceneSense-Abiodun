"""Sequential one-attempt-per-arm campaign in the registered order with the registered stop rule.

For each arm: launch once (retained lifecycle, cold host), then run the registered integration check.
A failed check, a crashed check or an incomplete launch stops the campaign for review. Deadline misses
never stop it. Nothing is retried. Create-only evidence in experiments/splitfusion_fixed_action_screen_v1/<name>.
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
SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_screen_v1"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.write(json.dumps(value, sort_keys=True) + "\n")
    log.flush()
    print(json.dumps(value), flush=True)


def run(name):
    out = MAIN / "experiments/splitfusion_fixed_action_screen_v1" / name
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads((MAIN / "experiments/splitfusion_fixed_action_screen_v1/selection_01/SELECTION.json").read_bytes())
    order = [int(a) for a in selection["arm_order"]]
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    stopped = None
    for index, action in enumerate(order):
        arm = SCREEN / f"arm{index}_a{action}"
        reg = json.loads((arm / "registration.json").read_bytes())
        if reg["forced_action"]["action_id"] != action or reg["arm_index"] != index or (arm / "capture").exists():
            stopped = dict(arm=arm.name, reason="ARM_ROOT_NOT_FRESH_OR_MISMATCHED")
            break
        token = f"RUN5_FIXED_ACTION_SCREEN_A{action:02d}_ONCE_EXECUTE_V1"
        event(log, kind="launch_start", arm=arm.name, action=action, index=index,
              loadavg=Path("/proc/loadavg").read_text().strip())
        with open(out / f"launch_{arm.name}.log", "x") as stream:
            rc = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_fixed_action_screen_v1.host", "launch",
                                 "--output", str(arm), "--execute", token], cwd=WT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT).returncode
        event(log, kind="launch_end", arm=arm.name, returncode=rc)
        check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with open(out / f"check_{arm.name}.json", "x") as stream, open(out / f"check_{arm.name}.stderr", "x") as err:
            crc = subprocess.run([sys.executable, "-B", "-m", "rl_agent.splitfusion_fixed_action_screen_analysis_v1.analyze",
                                  "--arm", str(arm), "--launch-returncode", str(rc)], cwd=MAIN, env=check_env,
                                 stdout=stream, stderr=err).returncode
        event(log, kind="check_end", arm=arm.name, check_returncode=crc)
        if rc != 0 or crc != 0:
            stopped = dict(arm=arm.name, launch_returncode=rc, check_returncode=crc,
                           reason="INTEGRATION_CHECK_FAILED" if crc == 3 else ("CHECK_CRASHED" if crc != 0 else "LAUNCH_FAILED"))
            break
    result = dict(schema="splitfusion.fixed_action_screen.campaign.v1", order=order, stopped=stopped,
                  completed_arms=[json.loads(l)["arm"] for l in open(out / "CAMPAIGN.jsonl") if json.loads(l)["kind"] == "check_end"],
                  no_retry=True)
    (out / "CAMPAIGN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    event(log, kind="campaign_end", stopped=stopped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
