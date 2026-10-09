"""Fresh-process CPU verification and complete source seal; no live launch."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from . import contract as C
from .entries import write_x
from .deploy import CPU_FIXTURE_FILES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = {name: ["discover", "-s", "rl_agent/"+package, "-t", ".", "-p", "test_*.py", "-q"]
        for name, package in (("integration", "splitfusion_live_parity_runtime_v1"),
                              ("repair", "splitfusion_live_parity_repair_v1"),
                              ("existing_run5", "splitfusion_run5_live_avo_v1"))}
    groups["frozen_semantics"] = [
        "rl_agent.splitfusion_live_target_parity_v1.test_adapter",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase4_reward_hold_v2",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_prewarm_gt_priority_v2", "-q"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    results, logs = {}, {}
    sources = C.source_inventory(root)
    fixtures = {name: digest for name, (_size, digest) in CPU_FIXTURE_FILES.items()}
    if any((root/name).is_symlink() or C.sha(root/name) != digest for name, digest in fixtures.items()):
        raise ValueError("staged finite regression fixtures differ")
    with tempfile.TemporaryDirectory(prefix="parity-runtime-mpl-") as temp:
        env["MPLCONFIGDIR"] = temp
        for name, arguments in groups.items():
            command = [sys.executable, "-m", "unittest", *arguments]
            done = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=300)
            path = output / f"{name}.log"
            with path.open("x") as stream:
                stream.write(done.stdout)
            count = re.search(r"Ran (\d+) tests?", done.stdout)
            skip = re.search(r"skipped=(\d+)", done.stdout)
            results[name] = dict(command=command, returncode=done.returncode,
                tests=int(count.group(1)) if count else 0, skipped=int(skip.group(1)) if skip else 0)
            logs[path.name] = C.sha(path)
    if C.source_inventory(root) != sources:
        raise ValueError("sources changed during qualification")
    if any(C.sha(root/name) != digest for name, digest in fixtures.items()):
        raise ValueError("regression fixtures changed during qualification")
    report = dict(schema="splitfusion.parity_runtime.offline.v1", sources=sources, suites=results,
        logs=logs, fixtures=fixtures, passed=all(r["returncode"] == 0 and r["tests"] > 0 for r in results.values()),
        live_started=False, training_started=False, total_tests=sum(r["tests"] for r in results.values()),
        status="OFFLINE_ONLY__LIVE_CORRECTNESS_AND_PERFORMANCE_UNPROVEN")
    write_x(output/"OFFLINE_VERIFICATION.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sources", "logs")}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
