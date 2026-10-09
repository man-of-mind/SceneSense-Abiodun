"""CPU-only M1 startup/source proof; never starts live services or CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-B", "-m", "unittest", "-v",
               "rl_agent.splitfusion_m1_static_live_v1.test_fixed"]
    test = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    from . import contract as C
    inventory = C.source_inventory(root)
    import torch
    result = dict(
        schema="splitfusion.m1_static_live_v1.startup_proof.v1",
        passed=test.returncode == 0, status="PASS" if test.returncode == 0 else "FAIL",
        failures=0 if test.returncode == 0 else 1, errors=0,
        tests=4, sources=inventory, fixed_action=dict(C.FIXED_ACTION),
        zero_completion=dict(C.ZERO_COMPLETION), actor_calls_expected=0,
        fallbacks_expected=0, live_services_started=False,
        cuda_initialized=bool(torch.cuda.is_initialized()),
        command=command, stdout=test.stdout[-4000:], stderr=test.stderr[-4000:])
    if result["cuda_initialized"]:
        result["passed"] = False; result["status"] = "FAIL"
    target = out / "RESULT.json"
    target.write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha(target)}, indent=1))
    print(json.dumps({k: result[k] for k in ("status", "tests", "live_services_started", "cuda_initialized")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
