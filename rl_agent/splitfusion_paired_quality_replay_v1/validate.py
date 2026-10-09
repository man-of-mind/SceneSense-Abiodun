"""Create-only offline readiness evidence; never runs CARLA, OAI or inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .capture import INPUT_BYTES, sha, write_json
from .scope import PINS

TESTS = {
    "new_capture_and_replay": [
        "rl_agent.splitfusion_paired_quality_replay_v1.test_capture",
        "rl_agent.splitfusion_paired_quality_replay_v1.test_replay"],
    "retained_geometry_quality_native_labels_hold": [
        "rl_agent.splitfusion_live_parity_repair_v1.test_perception",
        "rl_agent.splitfusion_live_parity_repair_v1.test_evaluation",
        "rl_agent.splitfusion_live_parity_repair_v2.test_prediction_contract",
        "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase4_reward_hold_v2"],
}
CHILD = '''
import json, sys, torch, unittest
assert not torch.cuda.is_initialized()
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromNames(json.loads(sys.argv[1])))
assert not torch.cuda.is_initialized()
print("RESULT=" + json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "cuda_initialized": False}))
raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=False, exist_ok=False)
    package = Path(__file__).parent
    paths = list(package.glob("*.py")) + [root/name for name in PINS]
    sources = {str(path.relative_to(root)): sha(path) for path in paths}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    results = {}
    for name, modules in TESTS.items():
        done = subprocess.run([sys.executable, "-c", CHILD, json.dumps(modules)], cwd=root,
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        with (output/(name+".log")).open("x") as stream:
            stream.write(done.stdout)
        values = [json.loads(line[7:]) for line in done.stdout.splitlines() if line.startswith("RESULT=")]
        results[name] = {"returncode": done.returncode, **(values[-1] if values else {"tests": 0})}
        print(json.dumps({"suite": name, **results[name]}), flush=True)
    preservation = {}
    commands = {
        "prior_capture": ["experiments/splitfusion_live_parity_runtime_v5/20261003_optimization_v5/finalize.py", "--verify"],
        "prior_diagnosis": ["experiments/splitfusion_gt_runtime_diagnosis_v3/20261003_v1/finalize.py", "--verify"],
        "protected_sources": ["-m", "rl_agent.splitfusion_live_gt_optimization_v1.preservation", "--record",
            "experiments/splitfusion_live_gt_optimization_v1/20261003_v1/PROTECTED_BEFORE.json", "--verify"],
    }
    for name, arguments in commands.items():
        done = subprocess.run([sys.executable, *arguments], cwd=root, env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        with (output/(name+".log")).open("x") as stream:
            stream.write(done.stdout)
        preservation[name] = {"returncode": done.returncode}
        if done.returncode == 0:
            preservation[name]["evidence"] = json.loads(done.stdout.strip().splitlines()[-1])
    unchanged = all(sha(root/name) == digest for name, digest in sources.items())
    passed = (unchanged and all(v["returncode"] == 0 and v["tests"] > 0 and v["skipped"] == 0
                                for v in results.values())
              and all(v["returncode"] == 0 for v in preservation.values()))
    plan = {
        "purpose": "Separate perception-model/scene limitations from selected-compression loss on identical inputs",
        "observation": "Exact CPU float32 prepared seven-channel tensor before original H2D transfer",
        "planned_transmitted_budget": 100, "maximum_policy_inputs": 50,
        "input_payload_bytes_for_50": 50*INPUT_BYTES,
        "input_plus_wire_payload_cap_bytes": 512*1024*1024,
        "python_metadata_overhead_additional_and_bounded_by_frame_count": True,
        "hot_path": "CPU references only; no hashing, serialization, disk, tensor scan, cloning or extra device transfer",
        "persistence": "after verifying preparation worker AND GT worker are not alive",
        "uncompressed_reference": "Full frozen FP32 C2 to contextual tail; no ranker/drop/AE/quantization",
        "admission": ["externally pinned input archive manifest and model/source identities",
            "join every prepared policy identity against actual last-send/transmitted evidence",
            "join actual selected prediction plus complete target and AVO; count every missing join",
            "regenerated selected inner payload must equal captured inner bytes",
            "replayed selected native labels and complete records must equal live archive",
            "then compare dense and selected with the same corrected GT/AVO/scorer"],
        "report": ["vehicle/person recall, precision, matched XY error and sample counts separately",
            "native-label vehicle IoU and depth-supported person box-mask surrogate IoU",
            "Q_perc including explicit undefined-quality denominator; no fabricated reward for empty scenes",
            "paired per-frame differences, not cross-scene historical comparison"],
        "limitations": ["Exact prepared-input replay is not retained-live-C2 equality",
            "Retention can affect host allocation pressure; overhead is not yet measured",
            "Postrun dense inference is a quality reference, not an achievable radio deadline result",
            "This is not a GT runtime repair; quality evaluation still counts inside170ms"],
        "not_changed": ["actor", "state", "reward", "170ms inclusive feedback deadline", "k_min2 holds",
            "AVO and conservative vehicle conflicts", "UE thread cap", "CPU partition", "radar", "MPS", "precision"],
        "remaining_before_results": ["new prospective runtime integration and startup/teardown proof",
            "one bounded exact-input capture", "qualified real-model paired replay and scoring"],
        "live_launch_authorized_by_this_file": False,
    }
    write_json(output/"COMPARISON_PLAN.json", plan)
    report = {"status": "OFFLINE_PREPARATION_PASS__LIVE_CAPTURE_AND_REAL_MODEL_REPLAY_PENDING" if passed else "OFFLINE_PREPARATION_FAILED",
        "passed": passed, "suites": results, "preservation": preservation,
        "sources": sources, "sources_unchanged_during_tests": unchanged,
        "live_started": False, "model_inference_run": False, "training_started": False,
        "gt_runtime_fixed": False, "quality_comparison_results_available": False,
        "latest_measured_gt_worker_p50_ms": 259.34,
        "latest_measured_feedback_p50_ms": 463.97,
        "latest_policy_feedback_timely": "0/46",
        "latest_runtime_evidence": "experiments/splitfusion_live_parity_runtime_v5/20261003_optimization_v5/",
        "new_tests": sum(v.get("tests", 0) for k,v in results.items() if k == "new_capture_and_replay"),
        "total_tests": sum(v.get("tests", 0) for v in results.values())}
    write_json(output/"REPORT.json", report)
    write_json(output/"OUTPUT_SHA256.json", {"sha256": {p.name:sha(p) for p in sorted(output.iterdir())}})
    print(json.dumps({"status": report["status"], "tests": report["total_tests"], "output": str(output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
