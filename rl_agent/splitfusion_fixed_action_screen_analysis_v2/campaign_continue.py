"""Continuation of the replacement screen: arms 1-6 once each, original order/limits, amendment-01 stop rule.

Arm 0 is never re-run. Per arm: launch once through the retained lifecycle; integration check with the
launch clause evaluated separately; separate forced-screen verdict (forced_verdict.evaluate). Anything but
PASS_FORCED_SCREEN stops the campaign. No retries. Create-only evidence under
experiments/splitfusion_fixed_action_screen_v2/<name>.
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
SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_screen_v2"
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")


def event(log, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log.write(json.dumps(value, sort_keys=True) + "\n")
    log.flush()
    print(json.dumps(value), flush=True)


def run(name, amendment):
    out = MAIN / "experiments/splitfusion_fixed_action_screen_v2" / name
    out.mkdir(parents=True, exist_ok=False)
    amendment_doc = json.loads(Path(amendment).read_bytes())
    selection = json.loads((MAIN / "experiments/splitfusion_fixed_action_screen_v1/selection_01/SELECTION.json").read_bytes())
    order = [int(a) for a in selection["arm_order"]]
    env = {k: v for k, v in os.environ.items() if k not in STRIP}
    check_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = open(out / "CAMPAIGN.jsonl", "x", buffering=1)
    event(log, kind="continuation_start", amendment=str(amendment), amendment_id=amendment_doc.get("id"),
          arms=[f"arm{i}_a{a}" for i, a in enumerate(order)][1:])
    stopped = None
    for index, action in list(enumerate(order))[1:]:
        arm = SCREEN / f"arm{index}_a{action}"
        reg = json.loads((arm / "registration.json").read_bytes())
        if reg["forced_action"]["action_id"] != action or reg["arm_index"] != index or (arm / "capture").exists():
            stopped = dict(arm=arm.name, reason="ARM_ROOT_NOT_FRESH_OR_MISMATCHED")
            break
        token = f"RUN5_FIXED_ACTION_SCREEN_V2_A{action:02d}_ONCE_EXECUTE_V1"
        event(log, kind="launch_start", arm=arm.name, action=action, index=index,
              loadavg=Path("/proc/loadavg").read_text().strip())
        launch_log = out / f"launch_{arm.name}.log"
        with open(launch_log, "x") as stream:
            rc = subprocess.run([sys.executable, "-m", "rl_agent.splitfusion_fixed_action_screen_v2.host", "launch",
                                 "--output", str(arm), "--execute", token], cwd=WT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT).returncode
        event(log, kind="launch_end", arm=arm.name, returncode=rc)
        code = ("import json,sys\n"
                "from rl_agent.splitfusion_fixed_action_screen_analysis_v2.analyze import Arm, integration\n"
                "from rl_agent.splitfusion_fixed_action_screen_analysis_v2.forced_verdict import collect, evaluate\n"
                "arm, rc, log = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
                "try:\n"
                "    integ = integration(Arm(arm), None, do_rescore=True)\n"
                "except Exception as error:\n"
                "    integ = dict(error=f'{type(error).__name__}: {error}')\n"
                "verdict = evaluate(collect(arm, launch_returncode=rc, launch_log=log, integration_result=integ))\n"
                "print(json.dumps(dict(integration=integ, forced_screen_verdict=verdict), default=str, indent=1))\n"
                "sys.exit(0 if verdict['verdict'] == 'PASS_FORCED_SCREEN' else 3)\n")
        with open(out / f"verdict_{arm.name}.json", "x") as stream, open(out / f"verdict_{arm.name}.stderr", "x") as err:
            crc = subprocess.run([sys.executable, "-B", "-c", code, str(arm), str(rc), str(launch_log)], cwd=MAIN,
                                 env=check_env, stdout=stream, stderr=err).returncode
        event(log, kind="verdict_end", arm=arm.name, verdict_returncode=crc)
        if crc != 0:
            stopped = dict(arm=arm.name, launch_returncode=rc, verdict_returncode=crc,
                           reason="FORCED_SCREEN_VERDICT_STOP" if crc == 3 else "VERDICT_EVALUATION_CRASHED")
            break
    result = dict(schema="splitfusion.fixed_action_screen_v2.continuation.v1", order=order, amendment=str(amendment),
                  stopped=stopped, arm0_rerun=False, no_retry=True,
                  completed_arms=[json.loads(l)["arm"] for l in open(out / "CAMPAIGN.jsonl") if json.loads(l)["kind"] == "verdict_end"])
    (out / "CAMPAIGN_RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    event(log, kind="campaign_end", stopped=stopped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--amendment", required=True)
    args = parser.parse_args()
    run(args.name, args.amendment)
