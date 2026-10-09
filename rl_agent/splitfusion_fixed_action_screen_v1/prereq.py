"""Pre-launch evidence for the fixed-action screen (create-only, CPU only, no CARLA/OAI/GPU).

  offline : forced-seam suite against the real retained engine/controller/pipeline and the real
            frozen actor, bound to the exact screen sources
  startup : import every screen and integrated entry module in the live tree; CUDA must stay
            uninitialised; binds the complete runtime source inventory
Run from the live worktree root. Evidence goes to <worktree>/experiments/splitfusion_fixed_action_screen_v1/.
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
EVIDENCE = ROOT / "experiments/splitfusion_fixed_action_screen_v1"
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


def offline(name):
    out = _out(name)
    sources = C.exact_sources(ROOT)
    scratch = out / "tmp"
    scratch.mkdir()
    suite = "rl_agent.splitfusion_fixed_action_screen_v1.test_forced"
    log = out / "test_forced.log"
    code = subprocess.run([sys.executable, "-B", "-W", "ignore::ResourceWarning", "-m", "unittest", "-v", suite], cwd=ROOT,
                          env=_env(TMPDIR=str(scratch)), stdout=open(log, "x"), stderr=subprocess.STDOUT).returncode
    shutil.rmtree(scratch)
    text = log.read_text()
    ran = int(text.rsplit("Ran ", 1)[1].split()[0]) if "Ran " in text else 0
    failures = int(text.split("failures=")[1].split(")")[0].split(",")[0]) if "failures=" in text else 0
    errors = int(text.split("errors=")[1].split(")")[0].split(",")[0]) if "errors=" in text else 0
    errors += int(code != 0 and failures == 0 and errors == 0)
    ok = code == 0 and failures == 0 and errors == 0 and ran > 0 and C.exact_sources(ROOT) == sources
    result = dict(schema="splitfusion.fixed_action_screen.offline_proof.v1", status="PASS" if ok else "FAIL", tests=ran,
                  failures=failures, errors=errors, returncode=code, suite=suite, sources=sources,
                  early_gt_pipeline_unchanged=C.early_gt_pipeline_unchanged(ROOT), live_services_started=False)
    if not result["early_gt_pipeline_unchanged"]:
        result["status"] = "FAIL"
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    _seal(out)
    print(json.dumps({k: result[k] for k in ("status", "tests", "failures", "errors", "early_gt_pipeline_unchanged")}))


def startup(name):
    out = _out(name)
    code = ("import json,torch\n"
            "import rl_agent.splitfusion_fixed_action_screen_v1.entries, rl_agent.splitfusion_fixed_action_screen_v1.forced\n"
            "import rl_agent.splitfusion_fixed_action_screen_v1.host, rl_agent.splitfusion_fixed_action_screen_v1.runner\n"
            "import rl_agent.splitfusion_fixed_action_screen_v1.partition\n"
            "import rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_run5_fixed_action\n"
            "import rl_agent.splitfusion_single_gpu_profiler_v1.edge_profiled_v1_run5_fixed_action\n"
            "import rl_agent.splitfusion_exact_gt_live_v1.ue, rl_agent.splitfusion_exact_gt_live_v1.edge\n"
            "import rl_agent.splitfusion_exact_gt_live_v1.service, rl_agent.splitfusion_ue_input_prep_v1.candidate\n"
            "from rl_agent.splitfusion_fixed_action_screen_v1 import contract as C\n"
            "from pathlib import Path\n"
            "root=Path(C.__file__).resolve().parents[2]\n"
            "print(json.dumps(dict(sources=C.source_inventory(root), cuda=torch.cuda.is_initialized(), root=str(root))))\n")
    proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, env=_env(), capture_output=True, text=True)
    (out / "startup.log").write_text(proc.stdout[-4000:] + proc.stderr[-20000:])
    value = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
    result = dict(schema="splitfusion.fixed_action_screen.startup_proof.v1",
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
