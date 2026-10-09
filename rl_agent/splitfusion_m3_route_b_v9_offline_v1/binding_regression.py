"""Focused valid/negative regression for the V9 offline assessment binding."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from rl_agent.splitfusion_m3_route_b_v9 import contract as C
from rl_agent.splitfusion_m3_route_b_v9 import host as H

from . import binding as B

SCHEMA = "splitfusion.m3_route_b_v9_offline_v1.binding_regression.v1"


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(arm, mutate_registration=None, mutate_record=None):
    arm = Path(arm).resolve(strict=True)
    temporary = tempfile.TemporaryDirectory(prefix="m3_v9_offline_binding_")
    root = Path(temporary.name) / "prospective"
    root.mkdir()
    record = json.loads((arm / "PROSPECTIVE.json").read_bytes())
    for name in ("registration.json", "arm.json", "PREREQUISITE.json", "ACCEPTANCE.json", C.RETENTION_PLAN_FILE):
        shutil.copy2(arm / name, root / name)
    registration = json.loads((root / "registration.json").read_bytes())
    if mutate_registration is not None:
        mutate_registration(registration)
        registration.pop("registration_sha256", None)
        registration["registration_sha256"] = C.canonical(registration)
        _write(root / "registration.json", registration)
    (root / "compose_partition.yaml").write_text(H.compose_text(root.resolve(), record["retained_code"]))
    record["registration_file_sha256"] = C.sha(root / "registration.json")
    record["compose_file_sha256"] = C.sha(root / "compose_partition.yaml")
    if mutate_record is not None:
        mutate_record(record)
    _write(root / "PROSPECTIVE.json", record)
    return temporary, root


def run(arm, out):
    arm, out = Path(arm).resolve(strict=True), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    valid = B.resolve(arm)
    cases = {
        "wrong_target": (lambda r: r["channel"]["schedule"].__setitem__("target_snr_db", 21.5), None),
        "wrong_controller": (lambda r: r["channel"].__setitem__("controller_module", "wrong.controller"), None),
        "wrong_trace_identity": (lambda r: r["channel"]["schedule"].__setitem__("trace_id", "WRONG_TRACE"), None),
        "wrong_registration_binding": (lambda r: r.__setitem__("runtime_revision", "M3_1_ROUTE_B_GOOD_ALL_V8"), None),
        "wrong_exact_source_binding": (lambda r: r["exact_gt_sources"].__setitem__(
            sorted(r["exact_gt_sources"])[0], "0" * 64), None),
        "wrong_runtime_source_inventory": (None, lambda r: r["runtime_sources"].__setitem__(
            sorted(r["runtime_sources"])[0], "0" * 64)),
    }
    negatives = {}
    for name, (reg_mutation, record_mutation) in cases.items():
        temporary, fixture = _fixture(arm, reg_mutation, record_mutation)
        try:
            try:
                B.resolve(fixture)
                negatives[name] = dict(rejected=False, error=None)
            except Exception as error:
                negatives[name] = dict(rejected=True, error=f"{type(error).__name__}: {error}")
        finally:
            temporary.cleanup()
    checks = dict(valid_v9_binding=True, **{name: value["rejected"] for name, value in negatives.items()})
    result = dict(schema=SCHEMA, status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                  valid_binding=B.document(valid), negatives=negatives, isolated_temporary_fixtures=True,
                  retained_evidence_modified=False, live_services_started=False)
    _write(out / "RESULT.json", result)
    manifest = {"RESULT.json": hashlib.sha256((out / "RESULT.json").read_bytes()).hexdigest()}
    _write(out / "OUTPUT_SHA256.json", manifest)
    print(json.dumps(dict(status=result["status"], checks=checks), sort_keys=True))
    return result["status"] == "PASS"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.arm, args.out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
