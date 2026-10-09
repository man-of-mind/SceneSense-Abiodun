"""Pre-launch proof of the archive capacity repair (create-only, CPU only, no CARLA/OAI/GPU).

1. Retained suites re-run against the repaired module: ``splitfusion_live_gt_handoff_v1.test_archive`` (incl. the
   updated upper-bound refusal) and ``splitfusion_numeric_gt_runtime_v1.test_live_receipt`` (modeled client).
2. Focused capacity suite: 4,000 real-fsync jobs complete durably, the 4,001st is refused (parent) and the child
   enforces its own bound; job/byte credits, duplicate identity/path, bookkeeping bound and abort/reap enforced;
   only the registered two-file substitution is accepted.
3. Offline start / four-frame use / clean stop of the actual v2 service composition at the exact 1,000-frame limits.
Binds ``exact_gt_live_v2.contract.exact_sources`` (the pipeline source set incl. the repaired files).
Run from the MAIN checkout: python -m rl_agent.splitfusion_gt_archive_capacity_v1.prove --name proof_01
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / C.EVIDENCE
ENV = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
SUITES = ("rl_agent.splitfusion_live_gt_handoff_v1.test_archive", "rl_agent.splitfusion_numeric_gt_runtime_v1.test_live_receipt",
          "rl_agent.splitfusion_gt_archive_capacity_v1.test_archive_capacity")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(name):
    from rl_agent.splitfusion_exact_gt_live_v2 import contract as E
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(ENV, ARCHIVE_CAPACITY_MEASURED=str(out / "CAPACITY_MEASURED.json"))
    sources_before = E.exact_sources(ROOT)
    repaired = C.repaired(ROOT)
    test = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", *SUITES], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    (out / "TEST_LOG.txt").write_text(test.stdout + test.stderr)
    ran = re.search(r"Ran (\d+) tests", test.stderr)
    failures = re.search(r"failures=(\d+)", test.stderr)
    errors = re.search(r"errors=(\d+)", test.stderr)
    tests = dict(returncode=test.returncode, tests=int(ran.group(1)) if ran else 0,
                 failures=int(failures.group(1)) if failures else 0, errors=int(errors.group(1)) if errors else 0)
    rehearsal = subprocess.run([sys.executable, "-B", "-m", "rl_agent.splitfusion_gt_archive_capacity_v1.service_rehearsal",
                                "--out", str(out / "service_rehearsal")], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=900)
    (out / "SERVICE_REHEARSAL_LOG.txt").write_text(rehearsal.stdout + rehearsal.stderr)
    service = json.loads((out / "service_rehearsal/RESULT.json").read_bytes())
    measured = json.loads((out / "CAPACITY_MEASURED.json").read_bytes()) if (out / "CAPACITY_MEASURED.json").exists() else {}
    sources = E.exact_sources(ROOT)
    passed = (tests["returncode"] == 0 and tests["tests"] >= 29 and not tests["failures"] and not tests["errors"]
              and rehearsal.returncode == 0 and service.get("status") == "PASS" and measured.get("jobs") == 4000
              and sources == sources_before)
    result = dict(schema="splitfusion.gt_archive_capacity_v1.proof.v1", status="PASS" if passed else "FAIL",
        failures=tests["failures"], errors=tests["errors"], tests=tests["tests"], suites=list(SUITES),
        capacity_measured=measured, service_rehearsal=dict(status=service.get("status"), checks=service.get("checks"),
        limits=service.get("limits"), archive_and_scorer_max_frames=service.get("archive_and_scorer_max_frames"),
        service_ready_s=service.get("service_ready_s")),
        original=C.ORIGINAL, repaired=repaired, max_cumulative_jobs=C.MAX_CUMULATIVE_JOBS, sources=sources,
        live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "OUTPUT_SHA256.json" and "archive" not in p.relative_to(out).parts}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(dict(status=result["status"], tests=tests, service=service.get("status"), measured=measured), indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
