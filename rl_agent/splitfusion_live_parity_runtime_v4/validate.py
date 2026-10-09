"""Fresh-process staged regression suites; exact skip identity is recorded."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from . import contract as C
from .startup import write_x
from rl_agent.splitfusion_live_parity_runtime_v1.deploy import CPU_FIXTURE_FILES

CHILD = r'''
import json, sys, unittest
spec = json.loads(sys.argv[1])
loader = unittest.TestLoader()
suite = loader.discover(spec["directory"], pattern="test_*.py", top_level_dir=".") if "directory" in spec else loader.loadTestsFromNames(spec["names"])
result = unittest.TextTestRunner(verbosity=1).run(suite)
print("RESULT_JSON=" + json.dumps({"tests": result.testsRun,
    "skipped": len(result.skipped), "skips": [{"id": t.id(), "reason": reason} for t, reason in result.skipped]}))
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = {name: {"directory": "rl_agent/"+package} for name, package in (
        ("integration_v4", "splitfusion_live_parity_runtime_v4"),
        ("optimization", "splitfusion_live_gt_optimization_v1"),
        ("integration_v3", "splitfusion_live_parity_runtime_v3"),
        ("integration_v2", "splitfusion_live_parity_runtime_v2"),
        ("integration_v1", "splitfusion_live_parity_runtime_v1"),
        ("repair_v2", "splitfusion_live_parity_repair_v2"),
        ("repair", "splitfusion_live_parity_repair_v1"),
        ("existing_run5", "splitfusion_run5_live_avo_v1"))}
    groups["frozen_semantics"] = {"names": [
        "rl_agent.splitfusion_live_target_parity_v1.test_adapter",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase4_reward_hold_v2",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_prewarm_gt_priority_v2"]}
    sources = C.source_inventory(root)
    fixtures = {name: digest for name, (_size, digest) in CPU_FIXTURE_FILES.items()}
    assert all(not (root / name).is_symlink() and C.sha(root / name) == digest for name, digest in fixtures.items())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    results, logs = {}, {}
    with tempfile.TemporaryDirectory(prefix="parity-v2-tests-") as scratch:
        env["MPLCONFIGDIR"] = scratch
        for name, spec in groups.items():
            cmd = [sys.executable, "-c", CHILD, json.dumps(spec)]
            done = subprocess.run(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=300)
            path = output / (name + ".log")
            write_x(path, done.stdout.encode())
            records = [line[len("RESULT_JSON="):] for line in done.stdout.splitlines() if line.startswith("RESULT_JSON=")]
            result = json.loads(records[-1]) if records else {"tests": 0, "skipped": 0, "skips": []}
            results[name] = {**result, "returncode": done.returncode, "command": cmd}
            logs[path.name] = C.sha(path)
            print(json.dumps({"suite": name, **result, "returncode": done.returncode}), flush=True)
    assert sources == C.source_inventory(root)
    assert all(C.sha(root / name) == digest for name, digest in fixtures.items())
    report = {"schema": "splitfusion.parity_runtime.offline.v4", "sources": sources,
        "fixtures": fixtures, "suites": results, "logs": logs,
        "passed": all(r["returncode"] == 0 and r["tests"] > 0 for r in results.values()),
        "live_started": False, "training_started": False,
        "total_tests": sum(r["tests"] for r in results.values())}
    write_x(output / "OFFLINE_VERIFICATION.json", report)
    # Apply the same exact skip gate that preparation/launch will apply.
    from .host_capture import verify_offline
    verify_offline(output / "OFFLINE_VERIFICATION.json", root)
    print(json.dumps({"status": "OFFLINE_PASS", "total_tests": report["total_tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
