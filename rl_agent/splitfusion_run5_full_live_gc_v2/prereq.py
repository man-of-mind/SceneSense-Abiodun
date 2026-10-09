"""Pre-launch evidence for the paired GT GC live qualification (create-only, CPU only, no CARLA/OAI/GPU).

  offline : the retained full Run-5 suite (test_full_live) AND the GC-qualification suite (test_gc_live, which also
            starts the real host GT service with the exact 1,000-frame limits through the trampolines), bound to the
            exact sources
  startup : import every screen and integrated entry module in the live tree; CUDA must stay
            uninitialised; binds the complete runtime source inventory
Run from the live worktree root. Evidence goes to <worktree>/experiments/splitfusion_run5_full_live_gc_v2/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import contract as C

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_run5_full_live_gc_v2"
ENV = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def _out(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    return out


def _seal(out):
    manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob("*"))
                if p.is_file() and p.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _env(**extra):
    env = dict(os.environ, **ENV, **extra)
    env.pop("PYTHONPATH", None)
    return env


def _suite(out, suite, scratch):
    log = out / (suite.rsplit(".", 1)[1] + ".log")
    code = subprocess.run([sys.executable, "-B", "-W", "ignore::ResourceWarning", "-m", "unittest", "-v", suite], cwd=ROOT,
                          env=_env(TMPDIR=str(scratch)), stdout=open(log, "x"), stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    ran = int(text.rsplit("Ran ", 1)[1].split()[0]) if "Ran " in text else 0
    failures = int(text.split("failures=")[1].split(")")[0].split(",")[0]) if "failures=" in text else 0
    errors = int(text.split("errors=")[1].split(")")[0].split(",")[0]) if "errors=" in text else 0
    skipped = int(text.split("skipped=")[1].split(")")[0].split(",")[0]) if "skipped=" in text else 0
    errors += int(code != 0 and failures == 0 and errors == 0)
    return dict(suite=suite, tests=ran, failures=failures, errors=errors, skipped=skipped, returncode=code)


def offline(name):
    out = _out(name)
    sources = C.exact_sources(ROOT)
    scratch = out / "tmp"
    scratch.mkdir()
    retained = _suite(out, "rl_agent.splitfusion_run5_full_live_v1.test_full_live", scratch)
    gc_suite = _suite(out, "rl_agent.splitfusion_run5_full_live_gc_v2.test_gc_live", scratch)
    shutil.rmtree(scratch)
    ok = (all(r["returncode"] == 0 and r["failures"] == 0 and r["errors"] == 0 and r["skipped"] == 0 and r["tests"] > 0
              for r in (retained, gc_suite)) and C.exact_sources(ROOT) == sources)
    result = dict(schema="splitfusion.run5_full_live_gc_v2.offline_proof.v1", status="PASS" if ok else "FAIL",
                  tests=retained["tests"], gc_tests=gc_suite["tests"], failures=retained["failures"] + gc_suite["failures"],
                  errors=retained["errors"] + gc_suite["errors"], suites=[retained, gc_suite], sources=sources,
                  pipeline_delta_from_v1=C.pipeline_delta(ROOT), live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps({k: result[k] for k in ("status", "tests", "gc_tests", "failures", "errors")}))


def startup(name):
    out = _out(name)
    code = ("import json,torch\n"
            "import rl_agent.splitfusion_run5_full_live_gc_v2.entries, rl_agent.splitfusion_run5_full_live_v1.actor\n"
            "import rl_agent.splitfusion_run5_full_live_gc_v2.host, rl_agent.splitfusion_run5_full_live_gc_v2.runner\n"
            "import rl_agent.splitfusion_run5_full_live_gc_v2.partition, rl_agent.splitfusion_run5_full_live_v1.child_entry\n"
            "import rl_agent.splitfusion_run5_full_live_v1.runtime_policy, rl_agent.splitfusion_run5_full_live_v1.audit_actor\n"
            "import rl_agent.splitfusion_run5_full_live_gc_v2.gc_seams, rl_agent.splitfusion_run5_full_live_gc_v2.service_entry\n"
            "import rl_agent.splitfusion_run5_full_live_gc_v2.worker_entry, rl_agent.splitfusion_run5_full_live_gc_v2.gc_policy_live\n"
            "import rl_agent.splitfusion_numeric_gt_producer_v2.worker_switch, rl_agent.splitfusion_numeric_gt_producer_v2.source_v2\n"
            "import rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_full_live_gc_v2\n"
            "import rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_full_live_gc_v2\n"
            "import rl_agent.splitfusion_exact_gt_live_v2.ue, rl_agent.splitfusion_exact_gt_live_v2.edge\n"
            "import rl_agent.splitfusion_exact_gt_live_v2.service, rl_agent.splitfusion_ue_input_prep_v1.candidate\n"
            "from rl_agent.splitfusion_run5_full_live_gc_v2 import contract as C\n"
            "from pathlib import Path\n"
            "root=Path(C.__file__).resolve().parents[2]\n"
            "print(json.dumps(dict(sources=C.source_inventory(root), cuda=torch.cuda.is_initialized(), root=str(root))))\n")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True)
    (out / "startup.log").write_text(proc.stdout[-4000:] + proc.stderr[-20000:])
    value = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    result = dict(schema="splitfusion.run5_full_live_gc_v2.startup_proof.v1",
                  passed=proc.returncode == 0 and value.get("cuda") is False and value.get("root") == str(ROOT),
                  sources=value.get("sources"), cuda_initialized=value.get("cuda"), root=str(ROOT), live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps(dict(passed=result["passed"], files=len(result["sources"] or {}))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("offline", "startup"))
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    dict(offline=offline, startup=startup)[args.kind](args.name)
