"""Alternating CPU-only target preparation benchmark on the retained 16 frames.

Rebuilds complete targets with baseline/candidate, compares all numerical arrays,
PNG bytes, canonical target JSON, prepared digests, and original evaluator reward
inputs with three declared synthetic predictions. No retained predictions exist:
this does NOT rescore the failed live attempt or qualify live quality/timeliness.
Only a fresh output evidence directory is written; source/capture bytes are sealed
before and after. Original temporary output remains temporary, not durable GT.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time
import unittest

import cv2
import numpy as np

from . import target_builder as B
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELL = ROOT/"experiments/splitfusion_live_parity_runtime_v2/20261003_integration_v2/qualification_100/capture/cells/a71__favorable_stable"
DEFAULT_OUTPUT = ROOT/"experiments/splitfusion_live_gt_optimization_v1/target_builder_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_x(path, value):
    with Path(path).open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b"\n")


def measure(function):
    wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    value = function()
    return value, {"wall_ms": (time.perf_counter_ns()-wall)/1e6,
                   "thread_cpu_ms": (time.thread_time_ns()-cpu)/1e6}


def summary(values):
    return {"min": float(np.min(values)), "p50": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values)), "mean": float(np.mean(values))}


def target_json(prepared):
    return T._canonical({key: value for key, value in prepared.target.items()
        if key not in ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png")})


def compare(candidate, reference):
    if candidate.content_sha256 != reference.content_sha256 or target_json(candidate) != target_json(reference):
        raise AssertionError("target digest/canonical JSON differs")
    for key in ("segmentation", "object_ignore"):
        if candidate.target[key+"_png"] != reference.target[key+"_png"] or not np.array_equal(candidate.target[key], reference.target[key]):
            raise AssertionError("target PNG/array differs")


def replay(*, cell, output, repetitions=3):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("bounded 1..10 repetitions required")
    cell, output = Path(cell).resolve(strict=True), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    targets = cell/"parity_complete_targets"
    captured = {str(p.relative_to(cell)): sha(p) for folder in (targets, cell/"run5_live_avo_inputs")
                for p in sorted(folder.rglob("*")) if p.is_file()}
    ue_path = cell/"run4_phase6/PHASE6_UE_EVIDENCE.json"
    captured[str(ue_path.relative_to(cell))] = sha(ue_path)
    own_sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    # Preserve the reference audit's direct baseline rather than an approximation.
    reference_script = ROOT/"experiments/splitfusion_live_parity_repair_v2/20261003_v2/gt_cost_audit/replay_cost.py"
    reference_sha = sha(reference_script)
    verified_sources = A.verify_sources()
    action_rows = json.loads(ue_path.read_bytes())["transmitted_identities"]
    actions = {row["frame_id"]: row["run4_identity"] for row in action_rows}
    manifest = json.loads((targets/"MANIFEST.json").read_bytes())
    bundles = [T.verify_full_inputs(targets, expected_identity=row["identity"], expected_manifest_sha256=row["manifest_sha256"])
               for row in sorted(manifest["frames"], key=lambda row: row["identity"]["frame_id"])]
    if len(bundles) != 16:
        raise ValueError("expected the retained 16-frame failed-attempt subset")
    warmup, warm_timing = measure(B.prewarm)
    # Warm each path symmetrically; immutable code cache persists, frame output does not.
    E.prepare_target(bundles[0].inputs)
    with B.install_target_builder(): E.prepare_target(bundles[0].inputs)
    rows = []; reward_proofs = []
    for repetition in range(repetitions):
        for index, bundle in enumerate(bundles):
            frame = bundle.identity["frame_id"]
            before = A.input_content_sha256(bundle.inputs)
            order = ("baseline", "candidate") if (index+repetition) % 2 == 0 else ("candidate", "baseline")
            prepared = {}; timing = {}
            for label in order:
                if label == "candidate":
                    with B.install_target_builder():
                        prepared[label], timing[label] = measure(lambda: E.prepare_target(bundle.inputs))
                else:
                    prepared[label], timing[label] = measure(lambda: E.prepare_target(bundle.inputs))
                compare(prepared[label], bundle.prepared)
                if target_json(prepared[label]) != (bundle.directory/"target.json").read_bytes():
                    raise AssertionError("rebuilt target JSON differs from retained exact bytes")
            compare(prepared["candidate"], prepared["baseline"])
            if before != A.input_content_sha256(bundle.inputs):
                raise AssertionError("target preparation mutated input")
            rows.append({"frame_id": frame, "repetition": repetition, "order": list(order),
                         "timing": timing, "prepared_content_sha256": bundle.prepared.content_sha256,
                         "target_json_sha256": hashlib.sha256(target_json(prepared["candidate"])).hexdigest(), "exact": True})
            if repetition == 0:
                matches = list((cell/"run5_live_avo_inputs").glob(f"*_{frame}.avo.json"))
                if len(matches) != 1: raise ValueError("ambiguous AVO input identity")
                avo = json.loads(matches[0].read_bytes())["avo"]
                shape = bundle.prepared.target["segmentation"].shape
                predictions = {"zero": np.zeros(shape, np.uint8),
                    "target_classes_ignore_to_zero": np.where(bundle.prepared.target["segmentation"] == 255, 0, bundle.prepared.target["segmentation"]).astype(np.uint8),
                    "three_class_stripes": (np.indices(shape).sum(axis=0) % 3).astype(np.uint8)}
                measurements = {}
                for name, labels in predictions.items():
                    a = E.evaluate_prepared(identity=actions[frame], prepared=prepared["baseline"],
                        predicted_labels=labels, predictions=[], avo=avo)
                    b = E.evaluate_prepared(identity=actions[frame], prepared=prepared["candidate"],
                        predicted_labels=labels, predictions=[], avo=avo)
                    if A.canonical(a.measurement) != A.canonical(b.measurement) or A.canonical(a.evidence) != A.canonical(b.evidence):
                        raise AssertionError("original evaluator reward inputs/evidence differ")
                    measurements[name] = hashlib.sha256(A.canonical(b.measurement)).hexdigest()
                reward_proofs.append({"frame_id": frame, "predictions_are_synthetic": True,
                    "detection_records": 0, "measurement_sha256": measurements, "exact": True})
    text = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("rl_agent.splitfusion_live_gt_optimization_v1.test_target_builder")
    tests = unittest.TextTestRunner(stream=text, verbosity=2).run(suite)
    write_x(output/"TESTS.txt", text.getvalue().encode())
    if not tests.wasSuccessful(): raise AssertionError("target optimization regression suite failed")
    if captured != {name: sha(cell/name) for name in captured} or own_sources != {name: sha(ROOT/name) for name in own_sources}:
        raise AssertionError("retained/candidate source changed during replay")
    if verified_sources != A.verify_sources() or reference_sha != sha(reference_script):
        raise AssertionError("original builder or sealed reference audit changed")
    distributions = {label: {metric: summary([row["timing"][label][metric] for row in rows])
        for metric in ("wall_ms", "thread_cpu_ms")} for label in ("baseline", "candidate")}
    paired = {metric: summary([row["timing"]["baseline"][metric]/row["timing"]["candidate"][metric] for row in rows])
              for metric in ("wall_ms", "thread_cpu_ms")}
    report = {"schema": B.SCHEMA, "status": "OFFLINE_TARGET_PARITY_VERIFIED", "passed": True,
        "live_qualified": False, "live_run_performed": False, "gpu_used": False,
        "retained_frames": 16, "repetitions": repetitions, "paired_samples": len(rows),
        "exact_png_json_tables_and_prepared_hashes": True, "exact_original_evaluator_reward_inputs": True,
        "inputs_sources_and_reference_unchanged": True, "reference_audit_sha256": reference_sha,
        "original_sources": verified_sources, "candidate_sources": own_sources, "input_sha256": captured,
        "prewarm": {"result": warmup, "timing": warm_timing}, "timing": distributions, "paired_speedup": paired,
        "rows": rows, "reward_input_proofs": reward_proofs,
        "tests": {"run": tests.testsRun, "failed": len(tests.failures), "errors": len(tests.errors), "sha256": sha(output/"TESTS.txt")},
        "environment": {"opencv_threads": cv2.getNumThreads(), "numpy": np.__version__, "opencv": cv2.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "thread_caps": {k: os.environ.get(k) for k in
                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
        "caveats": ["Only the retained successful 16 GT frames are available; missing 32 were not reconstructed.",
            "Reward-input checks use declared synthetic masks and empty detections, not missing original live predictions.",
            "Preparation timing excludes source geometry, native transport/archive fsync, live contention, and policy inference.",
            "Compiled code only is cached; all original source checks and fresh per-frame mutable namespaces remain.",
            "Only baseline temporary build output is replaced; original durable bundles/fsync/READY remain unchanged."]}
    write_x(output/"RESULT.json", report)
    write_x(output/"OUTPUT_SHA256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, default=DEFAULT_CELL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    result = replay(cell=args.cell, output=args.output, repetitions=args.repetitions)
    print(json.dumps({key: result[key] for key in ("status", "retained_frames", "paired_samples", "timing", "paired_speedup", "tests")}, indent=2))


if __name__ == "__main__":
    main()
