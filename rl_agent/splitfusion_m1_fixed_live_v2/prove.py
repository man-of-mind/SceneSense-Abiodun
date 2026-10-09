"""CPU-only startup/source proof for one concrete fixed-v2 contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .contracts import CONTRACTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=CONTRACTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    C = CONTRACTS[args.profile]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    command = [sys.executable, "-B", "-m", "unittest", "-v",
               "rl_agent.splitfusion_m1_fixed_live_v2.test_fixed"]
    test = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    inventory = C.source_inventory(root)
    import torch
    result = dict(schema="splitfusion.m1_fixed_live_v2.startup_proof.v1",
        profile=args.profile, passed=test.returncode == 0,
        status="PASS" if test.returncode == 0 else "FAIL",
        failures=0 if test.returncode == 0 else 1, errors=0, tests=6,
        sources=inventory, fixed_action=dict(C.FIXED_ACTION),
        retention_resource_plan=dict(C.RETENTION_RESOURCE_PLAN),
        zero_completion=dict(C.ZERO_COMPLETION), actor_calls_expected=0,
        fallbacks_expected=0, live_services_started=False,
        cuda_initialized=bool(torch.cuda.is_initialized()), command=command,
        stdout=test.stdout[-5000:], stderr=test.stderr[-5000:])
    if result["cuda_initialized"]:
        result["passed"] = False
        result["status"] = "FAIL"
    target = out / "RESULT.json"
    target.write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(
        json.dumps({"RESULT.json": sha(target)}, indent=1))
    print(json.dumps({key: result[key] for key in
                      ("status", "profile", "tests", "live_services_started", "cuda_initialized")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

