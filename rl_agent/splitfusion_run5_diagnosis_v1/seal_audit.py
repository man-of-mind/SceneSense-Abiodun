"""Create/verify the offline diagnosis manifest; never changes source evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "experiments/splitfusion_run5_diagnosis_v1/20261003_v1"
CODE = ROOT / "rl_agent/splitfusion_run5_diagnosis_v1"
RESULT_NAMES = (
    "REPORT.md", "inference_audit.json", "inference_source_provenance.json",
    "environment_audit.json", "outcomes_audit.json", "outcomes_bins.json",
    "outcomes_policy_features.csv", "outcomes_sha256.json",
)
CODE_NAMES = (
    "__init__.py", "audit_inference.py", "test_audit_inference.py",
    "audit_environment.py", "audit_outcomes.py", "seal_audit.py",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true", help="Create-only; otherwise verify")
    args = parser.parse_args()
    inference = json.loads((RESULT / "inference_audit.json").read_text())
    environment = json.loads((RESULT / "environment_audit.json").read_text())
    outcomes = json.loads((RESULT / "outcomes_audit.json").read_text())
    inputs = {}
    groups = [environment["source_sha256"],
              outcomes["source_verification"]["source_hashes_before"],
              {v["path"]: v["sha256"] for v in inference["source_hashes"].values()}]
    for group in groups:
        for name, digest in group.items():
            if name in inputs and inputs[name] != digest:
                raise ValueError(f"audits disagree on source hash: {name}")
            inputs[name] = digest
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError(f"source changed: {name}")
    files = [RESULT / name for name in RESULT_NAMES] + [CODE / name for name in CODE_NAMES]
    payload = {
        "schema": "splitfusion.run5.offline_diagnosis_manifest.v1",
        "scope": "DIAGNOSIS_ONLY_NO_FIX_TRAINING_INFERENCE_OR_LIVE_RUN",
        "unique_reverified_inputs": len(inputs),
        "files_relative_to_main_root": {str(p.relative_to(ROOT)): sha(p) for p in files},
    }
    target = RESULT / "OUTPUT_SHA256.json"
    if args.seal:
        with target.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    elif json.loads(target.read_text()) != payload:
        raise ValueError("diagnosis output manifest differs")
    print(json.dumps({"status": "SEALED" if args.seal else "VERIFIED",
                      "outputs": len(files), "unique_inputs": len(inputs),
                      "manifest_sha256": sha(target)}))


if __name__ == "__main__":
    main()
