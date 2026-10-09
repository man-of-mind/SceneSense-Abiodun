"""Pre-launch evidence for the retention launcher (create-only, CPU only, no CARLA/OAI/GPU).

  retention : the four splitfusion_paired_retention_v2 suites + this package's test_retention_live, the seam pins,
              the registered 1,000-frame plan, and an exact rescore of the CURRENT numeric-GT archive layout (gc_v2
              validation: producer v2, mask lookup, capacity repair) through gt_adapter, bound to the exact sources

Original gc_v2 header: Pre-launch evidence for the paired GT GC live qualification (create-only, CPU only, no CARLA/OAI/GPU).

  offline : the retained full Run-5 suite (test_full_live) AND the GC-qualification suite (test_gc_live, which also
            starts the real host GT service with the exact 1,000-frame limits through the trampolines), bound to the
            exact sources
  startup : import every screen and integrated entry module in the live tree; CUDA must stay
            uninitialised; binds the complete runtime source inventory
Run from the live worktree root. Evidence goes to <worktree>/experiments/splitfusion_run5_channel_response_v1/.
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
EVIDENCE = ROOT / "experiments/splitfusion_run5_channel_response_v1"
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
    gc_suite = _suite(out, "rl_agent.splitfusion_run5_channel_response_v1.test_gc_live", scratch)
    channel = _suite(out, "rl_agent.splitfusion_run5_channel_response_v1.test_channel_selection", scratch)
    shutil.rmtree(scratch)
    ok = (all(r["returncode"] == 0 and r["failures"] == 0 and r["errors"] == 0 and r["skipped"] == 0 and r["tests"] > 0
              for r in (retained, gc_suite, channel)) and C.exact_sources(ROOT) == sources)
    result = dict(schema="splitfusion.run5_channel_response_v1.offline_proof.v1", status="PASS" if ok else "FAIL",
                  tests=retained["tests"], gc_tests=gc_suite["tests"], channel_tests=channel["tests"],
                  failures=retained["failures"] + gc_suite["failures"] + channel["failures"],
                  errors=retained["errors"] + gc_suite["errors"] + channel["errors"], suites=[retained, gc_suite, channel],
                  channel=C.channel_registration(ROOT), sources=sources,
                  pipeline_delta_from_v1=C.pipeline_delta(ROOT), live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps({k: result[k] for k in ("status", "tests", "gc_tests", "channel_tests", "failures", "errors")}))


def startup(name):
    out = _out(name)
    code = ("import json,torch\n"
            "import rl_agent.splitfusion_run5_channel_response_v1.entries, rl_agent.splitfusion_run5_full_live_v1.actor\n"
            "import rl_agent.splitfusion_run5_channel_response_v1.host, rl_agent.splitfusion_run5_channel_response_v1.runner\n"
            "import rl_agent.splitfusion_run5_channel_response_v1.partition, rl_agent.splitfusion_run5_full_live_v1.child_entry\n"
            "import rl_agent.splitfusion_run5_full_live_v1.runtime_policy, rl_agent.splitfusion_run5_full_live_v1.audit_actor\n"
            "import rl_agent.splitfusion_run5_channel_response_v1.gc_seams, rl_agent.splitfusion_run5_channel_response_v1.service_entry\n"
            "import rl_agent.splitfusion_run5_channel_response_v1.worker_entry, rl_agent.splitfusion_run5_channel_response_v1.gc_policy_live\n"
            "import rl_agent.splitfusion_numeric_gt_producer_v2.worker_switch, rl_agent.splitfusion_numeric_gt_producer_v2.source_v2\n"
            "import rl_agent.splitfusion_run5_channel_response_v1.retention, rl_agent.splitfusion_paired_retention_v2.observer\n"
            "import rl_agent.splitfusion_paired_retention_v2.plan, rl_agent.splitfusion_paired_retention_v2.writer\n"
            "import rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_channel_response_v1\n"
            "import rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_channel_response_v1\n"
            "import rl_agent.splitfusion_exact_gt_live_v2.ue, rl_agent.splitfusion_exact_gt_live_v2.edge\n"
            "import rl_agent.splitfusion_exact_gt_live_v2.service, rl_agent.splitfusion_ue_input_prep_v1.candidate\n"
            "from rl_agent.splitfusion_run5_channel_response_v1 import contract as C\n"
            "from pathlib import Path\n"
            "root=Path(C.__file__).resolve().parents[2]\n"
            "print(json.dumps(dict(sources=C.source_inventory(root), cuda=torch.cuda.is_initialized(), root=str(root))))\n")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True)
    (out / "startup.log").write_text(proc.stdout[-4000:] + proc.stderr[-20000:])
    value = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    result = dict(schema="splitfusion.run5_channel_response_v1.startup_proof.v1",
                  passed=proc.returncode == 0 and value.get("cuda") is False and value.get("root") == str(ROOT),
                  sources=value.get("sources"), cuda_initialized=value.get("cuda"), root=str(ROOT), live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps(dict(passed=result["passed"], files=len(result["sources"] or {}))))


GC_V2_ARCHIVE = (ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
                 "/20261006_run5_full_live_gc_v2/validation/capture/cells/a71__favorable_stable/numeric_gt/service/archive")
RETENTION_SUITES = tuple(f"rl_agent.splitfusion_paired_retention_v2.{m}"
                         for m in ("test_plan", "test_observer", "test_validate", "test_gt_adapter"))


def retention(name):
    from rl_agent.splitfusion_paired_retention_v2.observer import SEAM_PINS
    out = _out(name)
    sources = C.exact_sources(ROOT)
    scratch = out / "tmp"
    scratch.mkdir()
    suites = [_suite(out, suite, scratch) for suite in RETENTION_SUITES]
    launcher = _suite(out, "rl_agent.splitfusion_run5_channel_response_v1.test_retention_live", scratch)
    shutil.rmtree(scratch)
    log = out / "archive_rescore.json"
    code = subprocess.run([sys.executable, "-B", "-m", "rl_agent.splitfusion_paired_retention_v2.gt_adapter",
                           "--archive", str(GC_V2_ARCHIVE), "--output", str(log)], cwd=ROOT, env=_env(),
                          capture_output=True, text=True).returncode
    rescore = json.loads(log.read_bytes()) if log.exists() else {}
    archive = dict(path=str(GC_V2_ARCHIVE), returncode=code, frames=rescore.get("frames"), exact_frames=rescore.get("exact"),
                   field_failures=rescore.get("field_failures"),
                   exact=code == 0 and rescore.get("frames", 0) > 0 and rescore.get("exact") == rescore.get("frames"))
    every = [*suites, launcher]
    ok = (all(r["returncode"] == 0 and r["failures"] == 0 and r["errors"] == 0 and r["skipped"] == 0 and r["tests"] > 0
              for r in every) and archive["exact"] and C.exact_sources(ROOT) == sources
          and SEAM_PINS == {k: C.sha(ROOT / k) for k in SEAM_PINS})
    result = dict(schema="splitfusion.run5_channel_response_v1.retention_proof.v1", status="PASS" if ok else "FAIL",
                  retention_tests=sum(r["tests"] for r in suites), launcher_tests=launcher["tests"],
                  failures=sum(r["failures"] for r in every), errors=sum(r["errors"] for r in every), suites=every,
                  sources=sources, seam_pins=dict(SEAM_PINS), plan=C.retention_plan(1000), archive_rescore=archive,
                  live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps({k: result[k] for k in ("status", "retention_tests", "launcher_tests", "failures", "errors")}
                     | dict(archive_exact=archive["exact"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("offline", "startup", "retention"))
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    dict(offline=offline, startup=startup, retention=retention)[args.kind](args.name)
