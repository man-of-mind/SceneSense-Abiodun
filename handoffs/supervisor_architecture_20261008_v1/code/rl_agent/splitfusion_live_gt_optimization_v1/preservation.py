"""Read-only baseline and restoration proof for the retained pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1"
PROTECTED = (
    "splitfusion_live_target_parity_v1", "splitfusion_live_parity_repair_v1",
    "splitfusion_live_parity_repair_v2", "splitfusion_live_parity_runtime_v1",
    "splitfusion_live_parity_runtime_v2", "splitfusion_live_parity_runtime_v3",
    "splitfusion_run5_live_avo_v1", "splitfusion_cpu_partition_v1",
    "splitfusion_single_gpu_profiler_v1", "splitfusion_hybrid_sac_live_route_b_v2",
    "splitfusion_edge_optimization_v1", "splitfusion_live_dispatch_v1",
)
INVARIANTS = {
    "actor": "original Run-5 seed43/update10000, 22-D quality-aware",
    "weights_sha256": "f17e812defd7d14204ed9c2a140d776727ecc5cf55e0b2f5a3db466e24d2defe",
    "reward_clock": "action-open through Q_perc evaluation and UE feedback, inclusive 170ms",
    "hold": "k_min=2; retained ticket and fallback semantics",
    "cpu_partition": "retained disjoint radio/edge masks",
    "ue_thread_cap": "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, UE child only",
    "radar": "production CPU radar, no CUDA radar bundle",
    "edge_order": "retained serial map then evaluator submission",
    "no_change": ["CUDA/MPS", "codec/precision", "CARLA sensors", "OAI/channel", "GT/AVO rules"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    roots = {"main": ROOT, "runtime": RUNTIME}
    files = {}
    for label, root in roots.items():
        for package in PROTECTED:
            directory = root / "rl_agent" / package
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if path.suffix in (".py", ".json", ".pt") and "__pycache__" not in path.parts and path.is_file():
                    files[f"{label}/{path.relative_to(root)}"] = sha(path)
    dirty = subprocess.check_output(["git", "diff", "--name-only", "-z"], cwd=ROOT).split(b"\0")
    for name in dirty:
        if name:
            path = ROOT / name.decode()
            if path.is_file():
                files[f"main/{path.relative_to(ROOT)}"] = sha(path)
    return {"schema": "splitfusion.gt_optimization.preservation.v1", "roots": {k:str(v) for k,v in roots.items()},
            "files": files, "invariants": INVARIANTS,
            "cached_diff": subprocess.check_output(["git", "diff", "--cached", "--binary"], cwd=ROOT).decode(),
            "oai_diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "diff", "--binary"], cwd=ROOT/"OAI/openairinterface5g")).hexdigest()}


def verify(record):
    mismatches = []
    for key, digest in record["files"].items():
        label, relative = key.split("/", 1)
        path = Path(record["roots"][label]) / relative
        if not path.is_file() or sha(path) != digest:
            mismatches.append(key)
    now = snapshot()
    if now["cached_diff"] != record["cached_diff"]:
        mismatches.append("git staged changes")
    if now["oai_diff_sha256"] != record["oai_diff_sha256"]:
        mismatches.append("OAI user diff")
    if mismatches:
        raise ValueError(f"protected pipeline changed: {mismatches}")
    return {"status":"UNCHANGED", "protected_files":len(record["files"]),
            "staging_unchanged":True, "oai_diff_unchanged":True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify(json.loads(args.record.read_text()))))
    else:
        record = snapshot()
        args.record.parent.mkdir(parents=True, exist_ok=True)
        with args.record.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(json.dumps({"protected_files":len(record["files"]), "record":str(args.record)}))


if __name__ == "__main__":
    main()
