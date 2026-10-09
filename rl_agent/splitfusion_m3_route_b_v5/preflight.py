"""Run the exact production runner CLI and isolated negative controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import contract as C
from .host import verify_prerequisite, verify_prospective, verify_source_inventory

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v5"


def _process_snapshot():
    needles = ("CarlaUE4", "nr-softmodem", "splitfusion_exact_gt_live_v2.service",
               "splitfusion_m3_route_b_v4.worker_entry")
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(needle in command for needle in needles):
            found.append(dict(pid=int(path.parent.name), command=command[:1000]))
    return found


def _copy_fixture(prospective, destination):
    destination.mkdir()
    for path in Path(prospective).iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def _rejects(callable_):
    try:
        callable_()
    except Exception as exc:  # the evidence records the fail-closed reason
        return dict(rejected=True, exception=type(exc).__name__, reason=str(exc))
    return dict(rejected=False, exception=None, reason=None)


def run(name, prospective):
    prospective = Path(prospective).resolve(strict=True)
    record = verify_prospective(prospective)
    verify_prerequisite(prospective, record)
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SCENESENSE_PARTITION_OUTPUT=str(prospective),
               SCENESENSE_PARTITION_CODE=record["retained_code"], PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    capture = out / "unused_preflight_capture"
    command = [sys.executable, "-B", "-m", C.RUNNER,
               "--registration", str(prospective / "registration.json"), "--arm", "C2",
               "--output-root", str(capture), "--preflight"]
    before = _process_snapshot()
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    after = _process_snapshot()
    (out / "production_cli.log").write_text(proc.stdout + proc.stderr)
    binding_lines = [line.split("=", 1)[1] for line in proc.stdout.splitlines()
                     if line.startswith("M3_VERSION_BINDING=")]
    binding = json.loads(binding_lines[-1]) if len(binding_lines) == 1 else None

    controls = {}
    with tempfile.TemporaryDirectory(prefix="m3_v5_negative_") as temporary:
        temp = Path(temporary)
        source = temp / "source"
        source.mkdir()
        (source / "registered.py").write_text("sealed\n")
        source_manifest = {"registered.py": C.sha(source / "registered.py")}
        verify_source_inventory(source_manifest, source, label="isolated source")
        (source / "registered.py").write_text("drifted\n")
        controls["source_drift"] = _rejects(
            lambda: verify_source_inventory(source_manifest, source, label="isolated source"))

        inventory_fixture = temp / "inventory"
        _copy_fixture(prospective, inventory_fixture)
        inventory_record = json.loads((inventory_fixture / "PROSPECTIVE.json").read_bytes())
        relative = sorted(inventory_record["runtime_sources"])[0]
        inventory_record["runtime_sources"][relative] = "0" * 64
        (inventory_fixture / "PROSPECTIVE.json").write_text(json.dumps(inventory_record, indent=2, sort_keys=True) + "\n")
        controls["prospective_inventory_drift"] = _rejects(lambda: verify_prospective(inventory_fixture))

        registration_fixture = temp / "registration"
        _copy_fixture(prospective, registration_fixture)
        registration = json.loads((registration_fixture / "registration.json").read_bytes())
        registration["launcher_binding"]["verify_prerequisite_module"] = "wrong.version.host"
        registration.pop("registration_sha256", None)
        registration["registration_sha256"] = C.canonical(registration)
        (registration_fixture / "registration.json").write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n")
        prospective_record = json.loads((registration_fixture / "PROSPECTIVE.json").read_bytes())
        prospective_record["registration_file_sha256"] = C.sha(registration_fixture / "registration.json")
        (registration_fixture / "PROSPECTIVE.json").write_text(
            json.dumps(prospective_record, indent=2, sort_keys=True) + "\n")
        controls["registration_drift"] = _rejects(lambda: verify_prospective(registration_fixture))

        prerequisite_fixture = temp / "prerequisite"
        _copy_fixture(prospective, prerequisite_fixture)
        prerequisite = json.loads((prerequisite_fixture / "PREREQUISITE.json").read_bytes())
        prerequisite["schema"] = "wrong.prerequisite"
        (prerequisite_fixture / "PREREQUISITE.json").write_text(
            json.dumps(prerequisite, indent=2, sort_keys=True) + "\n")
        prerequisite_record = json.loads((prerequisite_fixture / "PROSPECTIVE.json").read_bytes())
        prerequisite_record["prerequisite_file_sha256"] = C.sha(prerequisite_fixture / "PREREQUISITE.json")
        (prerequisite_fixture / "PROSPECTIVE.json").write_text(
            json.dumps(prerequisite_record, indent=2, sort_keys=True) + "\n")
        controls["prerequisite_drift"] = _rejects(
            lambda: verify_prerequisite(prerequisite_fixture, verify_prospective(prerequisite_fixture)))

    all_rejected = all(value["rejected"] for value in controls.values())
    no_live_started = before == after and not capture.exists()
    passed = proc.returncode == 0 and binding is not None and all_rejected and no_live_started
    registration = json.loads((prospective / "registration.json").read_bytes())
    result = dict(
        schema="splitfusion.m3_route_b_v5.actual_cli_preflight.v1",
        status="PASS" if passed else "FAIL",
        command=command, cwd=str(ROOT), pythonpath_removed=True,
        returncode=proc.returncode, resolved_binding=binding,
        legitimate_inventory_verified=True, legitimate_prerequisite_verified=True,
        negative_controls=controls,
        process_snapshot_before=before, process_snapshot_after=after,
        carla_started=False if no_live_started else None,
        oai_started=False if no_live_started else None,
        live_inference_started=False if no_live_started else None,
        capture_directory_created=capture.exists(),
        registration_file_sha256=C.sha(prospective / "registration.json"),
        registration_content_sha256=registration["registration_sha256"],
        source_inventory_sha256=C.canonical(record["runtime_sources"]),
        source_files=len(record["runtime_sources"]),
        live_services_started=False if no_live_started else None)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    manifest = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(out.rglob("*")) if path.is_file() and path.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "returncode": proc.returncode,
                      "resolved_binding": binding, "negative_controls": controls,
                      "live_services_started": result["live_services_started"]}, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prospective", type=Path, required=True)
    args = parser.parse_args()
    run(args.name, args.prospective)
