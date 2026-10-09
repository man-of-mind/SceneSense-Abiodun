"""Pre-launch proof of the single-buffer input candidate (create-only; CPU suites + CUDA parity/timing on GPU 0).
Run from MAIN: env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m rl_agent.splitfusion_ue_input_prep_v2.prove --name proof_01
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_ue_input_prep_v2/20261006_v1"
SOURCES = ("rl_agent/splitfusion_ue_input_prep_v2/__init__.py", "rl_agent/splitfusion_ue_input_prep_v2/candidate.py",
           "rl_agent/splitfusion_ue_input_prep_v1/candidate.py", "rl_agent/splitfusion_live_parity_repair_v1/perception.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT / name) for name in SOURCES}


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    before = sources()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    test = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", "rl_agent.splitfusion_ue_input_prep_v2.test_candidate",
                           "rl_agent.splitfusion_ue_input_prep_v1.test_candidate"], cwd=ROOT, env=env, capture_output=True,
                          text=True, timeout=900)
    (out / "TEST_LOG.txt").write_text(test.stdout + test.stderr)
    ran = re.search(r"Ran (\d+) tests", test.stderr)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from . import breakdown as BD, candidate as V2
    from rl_agent.splitfusion_ue_input_prep_v1.candidate import prepare_live_input_bgr2rgb as V1
    dev = torch.device("cuda:0"); sync = torch.cuda.synchronize
    data = BD.frames()
    cuda_exact = all(V2.prepare_live_input_fused(b, r, dev).cpu().numpy().tobytes() == live.tobytes() for _, b, r, live in data)
    for _ in range(5):
        V1(data[0][1], data[0][2], dev); V2.prepare_live_input_fused(data[0][1], data[0][2], dev); sync()
    t1, t2 = [], []
    for rep in range(4):
        for _, b, r, _ in data:
            for fn in ((V1, V2.prepare_live_input_fused) if rep % 2 == 0 else (V2.prepare_live_input_fused, V1)):
                a = time.perf_counter_ns(); fn(b, r, dev); sync(); d = (time.perf_counter_ns() - a) / 1e6
                (t1 if fn is V1 else t2).append(d)
    t1, t2 = np.array(t1), np.array(t2)
    timing = dict(v1_installed=dict(p50=float(np.median(t1)), p95=float(np.percentile(t1, 95))),
                  v2_candidate=dict(p50=float(np.median(t2)), p95=float(np.percentile(t2, 95))),
                  paired_saving=dict(p50=float(np.median(t1 - t2)), p10=float(np.percentile(t1 - t2, 10)), min=float((t1 - t2).min())),
                  frames=len(data), repeats=4, device=str(dev), torch_threads=1, note="idle host; live contention differs")
    passed = test.returncode == 0 and ran is not None and int(ran.group(1)) >= 13 and cuda_exact and len(data) == 49 \
        and sources() == before
    result = dict(schema="splitfusion.ue_input_prep_v2.proof.v1", status="PASS" if passed else "FAIL",
                  failures=0 if test.returncode == 0 else None, errors=0 if test.returncode == 0 else None,
                  tests=int(ran.group(1)) if ran else 0, cuda_bit_exact_49=cuda_exact, timing=timing, sources=sources(),
                  live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                                        if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps(dict(status=result["status"], tests=result["tests"], cuda_exact=cuda_exact, saving=timing["paired_saving"])))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    raise SystemExit(run(parser.parse_args().name))
