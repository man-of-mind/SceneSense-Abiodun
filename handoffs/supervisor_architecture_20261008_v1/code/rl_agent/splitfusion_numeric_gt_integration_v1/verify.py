"""CPU-only correctness verification and small create-only evidence inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "experiments/splitfusion_numeric_gt_integration_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    return {str(path.relative_to(ROOT)): sha(path)
            for path in sorted(PACKAGE.iterdir()) if path.suffix in (".py", ".json")}


def verify_retained_references():
    checked = {}
    for path in (RESULT_ROOT / "contract_replay/RESULT.json", RESULT_ROOT / "feasibility/RESULT.json"):
        result = json.loads(path.read_bytes())
        pins = result.get("input_sha256", result.get("input_hashes", {}))
        # Feasibility output calls the field input_sha256 too; keep the explicit
        # nonempty check so future schema changes cannot turn this into a no-op.
        if not pins:
            raise ValueError("verification input inventory missing: " + str(path))
        for name, expected in pins.items():
            target = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha(target) != expected:
                raise ValueError("retained input changed: " + str(target))
            checked[str(target)] = expected
    for directory in (RESULT_ROOT / "contract_replay", RESULT_ROOT / "feasibility"):
        manifest = json.loads((directory / "OUTPUT_SHA256.json").read_bytes())
        pins = manifest.get("files", manifest)
        for name, expected in pins.items():
            if sha(directory / name) != expected:
                raise ValueError("candidate evidence changed: " + str(directory / name))
    return len(checked)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("create-only verification output required")
    before = inventory()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE), pattern="test_*.py", top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("offline correctness verification failed")
    inputs_checked = verify_retained_references()
    if before != inventory():
        raise ValueError("candidate source changed during verification")
    record = dict(status="OFFLINE_FOUNDATION_VERIFIED_NOT_LIVE_INTEGRATED", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        retained_input_files_rechecked=inputs_checked, candidate_source_sha256=before,
        source_unchanged_during_tests=True, live_run=False, performance_benchmark=False,
        external_producer_integrated=False, external_scorer_integrated=False,
        actual_tracker_transaction_proved=False,
        reference_contract_replay="contract_replay/RESULT.json", feasibility="feasibility/RESULT.json")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "RESULT.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
    with (output / "OUTPUT_SHA256.json").open("x") as stream:
        json.dump({"RESULT.json": sha(output / "RESULT.json")}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: record[k] for k in ("status", "tests_run", "retained_input_files_rechecked")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
