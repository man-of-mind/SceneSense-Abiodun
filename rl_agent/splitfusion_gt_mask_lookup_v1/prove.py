"""Pre-launch proof of the mask-lookup candidate (create-only, CPU only, no CARLA/OAI/GPU).

Retained integration v1/v2 suites + exact equivalence suite; paired timing on retained live masks (one pinned
core); actual v2 service composition at the exact 1,000-frame limits with exact scores. Binds the current pipeline.
Run from MAIN: python -m rl_agent.splitfusion_gt_mask_lookup_v1.prove --name proof_01
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from . import contract as MK

ROOT = Path(__file__).resolve().parents[2]
ENV = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
SUITES = tuple(f"rl_agent.splitfusion_numeric_gt_integration_v1.{n}" for n in
               ("test_contract", "test_join", "test_transactions", "test_composition", "test_sources", "test_feasibility")) + \
         tuple(f"rl_agent.splitfusion_numeric_gt_integration_v2.{n}" for n in
               ("test_ready", "test_join", "test_scoring", "test_ownership", "test_edge_digest")) + \
         ("rl_agent.splitfusion_gt_mask_lookup_v1.test_equivalence",)


def run(name):
    from rl_agent.splitfusion_exact_gt_live_v2 import contract as E
    out = ROOT / MK.EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}; env.update(ENV)
    before = E.exact_sources(ROOT)
    test = subprocess.run([sys.executable, "-B", "-m", "unittest", *SUITES], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=1800)
    (out / "TEST_LOG.txt").write_text(test.stdout + test.stderr)
    ran = re.search(r"Ran (\d+) tests", test.stderr)
    bench = subprocess.run(["taskset", "-c", "5", sys.executable, "-B", "-m", "rl_agent.splitfusion_gt_mask_lookup_v1.benchmark",
                            str(out / "BENCHMARK.json")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    rehearsal = subprocess.run([sys.executable, "-B", "-m", "rl_agent.splitfusion_gt_archive_capacity_v1.service_rehearsal",
                                "--out", str(out / "service_rehearsal")], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=900)
    (out / "SERVICE_REHEARSAL_LOG.txt").write_text(rehearsal.stdout + rehearsal.stderr)
    service = json.loads((out / "service_rehearsal/RESULT.json").read_bytes())
    benchmark = json.loads((out / "BENCHMARK.json").read_bytes()) if bench.returncode == 0 else {}
    sources = E.exact_sources(ROOT)
    passed = (test.returncode == 0 and ran and int(ran.group(1)) >= 142 and bench.returncode == 0
              and rehearsal.returncode == 0 and service.get("status") == "PASS" and sources == before)
    result = dict(schema="splitfusion.gt_mask_lookup_v1.proof.v1", status="PASS" if passed else "FAIL", failures=0 if
        test.returncode == 0 else None, errors=0 if test.returncode == 0 else None, tests=int(ran.group(1)) if ran else 0,
        suites=list(SUITES), benchmark=benchmark, service_rehearsal=dict(status=service.get("status"),
        checks=service.get("checks"), limits=service.get("limits"), frames=service.get("frames")),
        original=MK.ORIGINAL, repaired=MK.repaired(ROOT), sources=sources, live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    manifest = {str(p.relative_to(out)): MK.sha(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.relative_to(out).parts}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(dict(status=result["status"], tests=result["tests"], saving=benchmark.get("paired_saving_per_admission_ms"))))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
