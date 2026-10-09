"""Focused v4 deployment-path preflight (imports, source inventory, affected tests)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v4"


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    suites = ("rl_agent.splitfusion_m3_route_b_v4.test_abort_evidence",
              "rl_agent.splitfusion_m3_route_b_v3.test_m3_v3")
    proc = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", *suites], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    (out / "tests.log").write_text(proc.stdout + proc.stderr)
    imports = ("import torch\n"
        "import rl_agent.splitfusion_m3_route_b_v4.entries\n"
        "import rl_agent.splitfusion_m3_route_b_v4.host\n"
        "import rl_agent.splitfusion_m3_route_b_v4.runner\n"
        "import rl_agent.splitfusion_m3_route_b_v4.partition\n"
        "import rl_agent.splitfusion_m3_route_b_v4.service_entry\n"
        "import rl_agent.splitfusion_m3_route_b_v4.worker_entry\n"
        "import rl_agent.splitfusion_m3_route_b_v4.abort_evidence\n"
        "import rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_m3_route_b_v4\n"
        "import rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_m3_route_b_v4\n"
        "from rl_agent.splitfusion_m3_route_b_v4 import contract as C\n"
        "from pathlib import Path\n"
        "root=Path(C.__file__).resolve().parents[2]\n"
        "print(__import__('json').dumps(dict(sources=C.source_inventory(root),cuda=torch.cuda.is_initialized())))\n")
    startup = subprocess.run([sys.executable, "-B", "-c", imports], cwd=ROOT, env=env,
                             capture_output=True, text=True)
    (out / "startup.log").write_text(startup.stdout + startup.stderr)
    value = json.loads(startup.stdout.strip().splitlines()[-1]) if startup.returncode == 0 else {}
    text = proc.stdout + proc.stderr
    passed = (proc.returncode == 0 and "FAILED" not in text and startup.returncode == 0
              and value.get("cuda") is False and value.get("sources") == C.source_inventory(ROOT))
    result = dict(schema="splitfusion.m3_route_b_v4.deployment_preflight.v1",
        status="PASS" if passed else "FAIL", affected_suites=list(suites), test_returncode=proc.returncode,
        startup_returncode=startup.returncode, cuda_initialized=value.get("cuda"),
        source_files=len(value.get("sources") or {}), sources=value.get("sources"), live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.rglob("*")) if path.is_file() and path.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "test_returncode", "startup_returncode",
                                              "cuda_initialized", "source_files")}))
    if not passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
