"""Negative controls for the registration-bound V14 two-phase authority."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from rl_agent.splitfusion_m3_route_b_v14 import contract as C
from rl_agent.splitfusion_m3_route_b_v14 import host as H
from . import binding as B

SCHEMA = "splitfusion.m3_route_b_v14_offline_v1.binding_regression.v1"


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(arm, mutate_registration=None, mutate_record=None):
    arm = Path(arm).resolve(strict=True)
    temporary = tempfile.TemporaryDirectory(prefix="m3_v14_offline_binding_")
    root = Path(temporary.name) / "prospective"
    root.mkdir()
    record = json.loads((arm / "PROSPECTIVE.json").read_bytes())
    for name in ("registration.json", "arm.json", "PREREQUISITE.json", "ACCEPTANCE.json", C.RETENTION_PLAN_FILE):
        shutil.copy2(arm / name, root / name)
    registration = json.loads((root / "registration.json").read_bytes())
    if mutate_registration:
        mutate_registration(registration)
        registration.pop("registration_sha256", None)
        registration["registration_sha256"] = C.canonical(registration)
        _write(root / "registration.json", registration)
    (root / "compose_partition.yaml").write_text(H.compose_text(root.resolve(), record["retained_code"]))
    record["registration_file_sha256"] = C.sha(root / "registration.json")
    record["compose_file_sha256"] = C.sha(root / "compose_partition.yaml")
    if mutate_record:
        mutate_record(record)
    _write(root / "PROSPECTIVE.json", record)
    return temporary, root


def run(arm, out):
    arm, out = Path(arm).resolve(strict=True), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    valid = B.resolve(arm)

    def wrong_good_level(reg):
        reg["channel"]["schedule"]["phases"][0]["target_snr_db"] = 21.5
    def missing_switch(reg):
        reg["channel"]["schedule"]["phases"] = reg["channel"]["schedule"]["phases"][:1]
    def repeated_switch(reg):
        reg["channel"]["schedule"]["phases"].append(dict(reg["channel"]["schedule"]["phases"][0]))
    def wrong_fraction(reg):
        reg["channel"]["schedule"]["transition_fraction"] = 0.6
    def wrong_progress_binding(reg):
        reg["launcher_binding"]["child"]["route_progress"] = "wrong.progress"

    cases = {
        "wrong_phase_level": (wrong_good_level, None),
        "missing_switch": (missing_switch, None),
        "repeated_switch": (repeated_switch, None),
        "wrong_transition_fraction": (wrong_fraction, None),
        "wrong_registration_binding": (lambda r: r.__setitem__("runtime_revision", C.V12.RUNTIME_REVISION), None),
        "wrong_progress_binding": (wrong_progress_binding, None),
        "wrong_exact_source": (lambda r: r["exact_gt_sources"].__setitem__(
            f"{C.PACKAGE}/route_progress.py", "0" * 64), None),
        "wrong_runtime_inventory": (None, lambda r: r["runtime_sources"].__setitem__(
            f"{C.PACKAGE}/bad_to_good_controller.py", "0" * 64)),
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
    schedule = valid.SCHEDULE
    checks = dict(
        valid_v14_binding=True,
        authority_is_registration_bound=valid.CONTROLLER_MODULE == C.CONTROLLER_MODULE,
        two_phase_schedule=[p["phase"] for p in schedule["phases"]] == ["BAD", "GOOD"],
        exact_levels=[(p["target_snr_db"], p["expected_rfsim_command_db"]) for p in schedule["phases"]]
                     == [(8.5, -4.0), (23.5, -12.0)],
        exact_half_trigger=schedule["transition_fraction"] == 0.5,
        one_way_latch="permanent" in schedule["latch"],
        **{name: value["rejected"] for name, value in negatives.items()})
    result = dict(schema=SCHEMA, status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                  valid_binding=B.document(valid), negatives=negatives, isolated_temporary_fixtures=True,
                  retained_evidence_modified=False, live_services_started=False)
    _write(out / "RESULT.json", result)
    _write(out / "OUTPUT_SHA256.json", {"RESULT.json": hashlib.sha256((out / "RESULT.json").read_bytes()).hexdigest()})
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
