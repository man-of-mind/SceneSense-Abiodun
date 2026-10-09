"""Launch the two preregistered M1 fixed arms, sequentially and exactly once."""
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
from rl_agent.splitfusion_m1_baselines_v1.campaign import retention_postflight


MAIN = Path(__file__).resolve().parents[2]
WT = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
ROOTS = (WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
         "20261007_m1_baselines_v3")
EVIDENCE = MAIN / "experiments/splitfusion_m1_baselines_v1"
PREFLIGHT = EVIDENCE / "preflight_fixed_v3/run_01/RESULT.json"
ARMS = (
    ("fixed_ae32_uint8_a54", "rl_agent.splitfusion_m1_fixed_live_v2.host_ae32"),
    ("fixed_mode11_q6000", "rl_agent.splitfusion_m1_fixed_live_v2.host_mode11"),
)
STRIP = ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
         "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD")
ATTEMPT_FILES = ("registration.json", "PROSPECTIVE.json", "PREREQUISITE.json",
                 "ACCEPTANCE.json", "arm.json", "compose_partition.yaml",
                 "RETENTION_PLAN.json")
FRESH_OUTPUTS = ("capture", "gt_gc_trace", "paired_retention_v2",
                 "RETENTION_RESULT.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mem_available():
    line = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))
    return int(line.split()[1]) * 1024


def event(stream, **value):
    value["wall"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    value["monotonic_ns"] = time.monotonic_ns()
    line = json.dumps(value, sort_keys=True)
    stream.write(line + "\n")
    stream.flush()
    print(line, flush=True)


def wait_exclusive(timeout_s=600):
    started = time.monotonic()
    while True:
        value = live_window()
        if value["exclusive"]:
            return value, True
        if time.monotonic() - started >= timeout_s:
            return value, False
        time.sleep(15)


def freeze(output):
    preflight = json.loads(PREFLIGHT.read_bytes())
    if preflight.get("status") != "PASS":
        raise ValueError("both-arm slim-image preflight did not pass")
    arm_records = {}
    actor_hashes = set()
    registrations = []
    for key, module in ARMS:
        root = ROOTS / key
        if not root.is_dir() or any((root / name).exists() for name in FRESH_OUTPUTS):
            raise ValueError(f"attempt root is missing or not fresh: {root}")
        files = {name: sha(root / name) for name in ATTEMPT_FILES}
        reg = json.loads((root / "registration.json").read_bytes())
        actor_hashes.add(reg["policy"]["expected_weights_sha256"])
        registrations.append(reg["registration_sha256"])
        arm_records[key] = dict(
            root=str(root), host_module=module, files=files,
            registration_sha256=reg["registration_sha256"],
            fixed_action=reg["external_fixed_action"],
            retention_resource_plan=reg["retention_resource_plan"],
            execute_token=reg["arms"]["C2"]["execute_token"],
            fresh_before_first_capture=True)
    if len(actor_hashes) != 1 or len(set(registrations)) != 2:
        raise ValueError("fixed attempts do not have one actor audit or separate registrations")
    record = dict(
        schema="splitfusion.m1_baselines_v1.fixed_pair_freeze.v3",
        order=[key for key, _ in ARMS], arms=arm_records,
        actor_boundary_sha256=next(iter(actor_hashes)),
        both_registered_before_first_capture=True,
        separate_registrations=True,
        no_retry=True, live_launch_count_per_arm=1,
        route="FAVORABLE_STABLE", budget_frames_per_arm=1000,
        separate_sessions_not_same_input=True,
        preflight_path=str(PREFLIGHT), preflight_sha256=sha(PREFLIGHT),
        preflight_status=preflight["status"],
        launcher_sources={str(path.relative_to(MAIN)): sha(path) for path in sorted(
            (MAIN / "rl_agent/splitfusion_m1_fixed_live_v2").glob("*.py"))},
        controller_sha256=sha(__file__))
    (output / "FREEZE.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def run(name):
    output = EVIDENCE / name
    output.mkdir(parents=False, exist_ok=False)
    log = open(output / "CAMPAIGN.jsonl", "x", buffering=1)
    try:
        frozen = freeze(output)
    except Exception as exc:
        event(log, kind="prelaunch_stop", error=f"{type(exc).__name__}: {exc}")
        return 2
    event(log, kind="pair_frozen", order=frozen["order"],
          actor_boundary_sha256=frozen["actor_boundary_sha256"],
          preflight_sha256=frozen["preflight_sha256"])
    env = {key: value for key, value in os.environ.items() if key not in STRIP}
    completed, stopped = [], None
    for index, (key, module) in enumerate(ARMS):
        root = ROOTS / key
        reg = json.loads((root / "registration.json").read_bytes())
        resource = reg["retention_resource_plan"]
        remaining_bounds = sum(
            json.loads((ROOTS / later / "registration.json").read_bytes())
                ["retention_resource_plan"]["retained_bytes_upper_bound"]
            for later, _ in ARMS[index:])
        required_disk = 8 * 2**30 + remaining_bounds
        free = shutil.disk_usage("/").free
        memory = mem_available()
        event(log, kind="resource_preflight", arm=key, free_bytes=free,
              required_free_bytes=required_disk, shortfall_bytes=max(0, required_disk - free),
              mem_available_bytes=memory, retention_bound_bytes=resource["retained_bytes_upper_bound"])
        if free < required_disk or memory < 4 * 2**30:
            stopped = dict(arm=key, reason="DISK_OR_MEMORY_PREFLIGHT_FAILED",
                           disk_shortfall_bytes=max(0, required_disk - free),
                           memory_shortfall_bytes=max(0, 4 * 2**30 - memory))
            event(log, kind="stop", **stopped)
            break
        window, ok = wait_exclusive()
        (output / f"window_before_{key}.json").write_text(
            json.dumps(window, indent=1, sort_keys=True) + "\n")
        event(log, kind="exclusive_window", arm=key, passed=ok,
              containers=window["containers"], process_hits=window["process_hits"],
              gpu_memory_used_mib=window["gpu_memory_used_mib"],
              loadavg_1min=window["loadavg_1min"])
        if not ok:
            stopped = dict(arm=key, reason="EXCLUSIVE_LIVE_WINDOW_UNAVAILABLE")
            event(log, kind="stop", **stopped)
            break
        token = reg["arms"]["C2"]["execute_token"]
        event(log, kind="launch_start", arm=key, index=index,
              registration_sha256=reg["registration_sha256"], token=token)
        started = time.monotonic_ns()
        with open(output / f"launch_{key}.log", "x") as stream:
            returncode = subprocess.run(
                [sys.executable, "-B", "-m", module, "launch",
                 "--output", str(root), "--execute", token],
                cwd=WT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        elapsed = time.monotonic_ns() - started
        event(log, kind="launch_end", arm=key, returncode=returncode,
              elapsed_monotonic_ns=elapsed)
        completed.append(dict(arm=key, returncode=returncode,
                              elapsed_monotonic_ns=elapsed))
        try:
            post = retention_postflight(root, output, key)
            event(log, kind="retention_postflight", arm=key,
                  manifest_matches_child=post.get("manifest_matches_child"),
                  status_counts=(post.get("retention_result") or {}).get("status_counts"),
                  gt_references=post.get("gt_references"))
        except Exception as exc:
            event(log, kind="retention_postflight_error", arm=key,
                  error=f"{type(exc).__name__}: {exc}")
        after = live_window()
        (output / f"window_after_{key}.json").write_text(
            json.dumps(after, indent=1, sort_keys=True) + "\n")
        event(log, kind="postlaunch_window", arm=key, exclusive=after["exclusive"],
              containers=after["containers"], process_hits=after["process_hits"])
        # A measured overload, timeout, zero completion, or nonzero generic
        # qualification return is an outcome, not a retry or reason to replace
        # this arm.  Only unsafe host cleanup prevents the separately registered
        # second arm from starting.
        if not after["exclusive"] and index + 1 < len(ARMS):
            stopped = dict(arm=key, reason="POSTLAUNCH_HOST_NOT_CLEAN_SECOND_ARM_NOT_STARTED")
            event(log, kind="stop", **stopped)
            break
    result = dict(
        schema="splitfusion.m1_baselines_v1.fixed_pair_capture.v3",
        authorized_order=[key for key, _ in ARMS], completed_attempts=completed,
        stopped=stopped, no_retry=True,
        exactly_one_launch_each=(len(completed) == 2 and
                                 [item["arm"] for item in completed] ==
                                 [key for key, _ in ARMS]))
    (output / "CAMPAIGN_RESULT.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n")
    event(log, kind="campaign_end", result=result)
    return 0 if result["exactly_one_launch_each"] else 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
