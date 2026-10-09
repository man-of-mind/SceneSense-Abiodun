"""Create-only CPU regression record and source seal. Never runs a live route."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = {
        "repair": ["discover", "-s", "rl_agent/splitfusion_live_parity_repair_v1", "-t", ".", "-p", "test_*.py", "-q"],
        "existing_run5": ["discover", "-s", "rl_agent/splitfusion_run5_live_avo_v1", "-t", ".", "-p", "test_*.py", "-q"],
        "existing_target_and_phase6": [
            "rl_agent.splitfusion_live_target_parity_v1.test_adapter",
            "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2",
            "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase4_reward_hold_v2",
            "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2",
            "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_prewarm_gt_priority_v2", "-q"],
    }
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    results = {}
    with tempfile.TemporaryDirectory(prefix="parity-validation-mpl-") as mpl:
        env["MPLCONFIGDIR"] = mpl
        for name, arguments in groups.items():
            start = time.monotonic()
            command = [sys.executable, "-m", "unittest", *arguments]
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=180)
            with (output / f"{name}.log").open("x") as stream:
                stream.write(done.stdout)
            count = re.search(r"Ran (\d+) tests?", done.stdout)
            skipped = re.search(r"skipped=(\d+)", done.stdout)
            results[name] = {"command": command, "returncode": done.returncode,
                "tests": None if count is None else int(count.group(1)),
                "skipped": 0 if skipped is None else int(skipped.group(1)),
                "elapsed_s": time.monotonic()-start, "log_sha256": sha(output/f"{name}.log")}
        audit = subprocess.run([sys.executable,"-m","rl_agent.splitfusion_run5_diagnosis_v1.seal_audit"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    if audit.returncode:
        raise ValueError("preserved diagnosis/source hashes changed: " + audit.stderr)
    source_files = {str(path.relative_to(ROOT)): sha(path)
        for path in sorted(PACKAGE.iterdir()) if path.suffix in {".py", ".json"}}
    result = {"schema": "splitfusion.live_parity_repair.validation.v1", "suites": results,
        "all_test_groups_passed": all(r["returncode"] == 0 for r in results.values()),
        "total_tests_run": sum(r["tests"] or 0 for r in results.values()),
        "preserved_diagnosis": json.loads(audit.stdout), "new_package_files": source_files,
        "live_run_started": False, "training_started": False,
        "status": "CPU_REPAIR_TESTED__NOT_DEPLOYED__LIVE_QUALIFICATION_REQUIRED",
        "notes": ["Local IPC regression tests require sandbox socket permission; no external launcher is enabled.",
                  "Frozen Phase6 teardown fixture also performs a read-only Docker image identity lookup.",
                  "CPU tests and retained-fixture timing do not establish improved live accuracy or deadline success."]}
    write(output/"VALIDATION.json", result)
    print(json.dumps({k: result[k] for k in ("all_test_groups_passed","total_tests_run","status")}))
    return 0 if result["all_test_groups_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
