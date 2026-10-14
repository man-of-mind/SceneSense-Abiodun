"""Focused V13 runner/verifier/transition binding regression (offline only)."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from . import contract as C
from . import entries, host, runner

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT_ROOT = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1"
EVIDENCE = ROOT / "experiments/splitfusion_m3_route_b_v13"
REGRESSION_SCHEMA = host.REGRESSION_SCHEMA


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(name):
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    channel = C.channel_registration(ROOT)
    transition = channel["schedule"]["route_transition"]
    v12_path = DEPLOYMENT_ROOT / host.V12_ROOT_RELATIVE / "registration.json"
    v12 = json.loads(v12_path.read_bytes())
    synthetic = json.loads(json.dumps(v12))
    synthetic["channel"] = channel
    checks = {
        "runner_native_v13_contract": runner.C is C and runner.C.__name__ == C.MODULE + ".contract",
        "entries_native_v13_contract": entries.C is C and entries.C.__name__ == C.MODULE + ".contract",
        "runner_uses_relative_v13_verifiers": "from .host import verify_prerequisite, verify_prospective" in inspect.getsource(runner.validate_parent),
        "child_uses_relative_v13_verifier": "from .host import verify_prospective" in inspect.getsource(entries.validate_child_startup),
        "role_constants_coherent": all(C.role_constant_audit().values()),
        "launcher_binds_route_progress": C.launcher_binding()["child"]["route_progress"] == C.ROUTE_PROGRESS_MODULE,
        "two_registered_anchor_levels": [(p["target_snr_db"], p["expected_rfsim_command_db"]) for p in channel["schedule"]["phases"]] == [(23.5, -12.0), (8.5, -4.0)],
        "actual_geometry_half_threshold": abs(transition["threshold_distance_m"] - transition["route_length_m"] * 0.5) < 1e-12,
        "one_way_latch_registered": "permanent" in channel["schedule"]["latch"],
        "progress_and_phase_not_actor_inputs": channel["actor_input"]["route_progress_to_actor"] is False and channel["actor_input"]["phase_identity_to_actor"] is False,
        "freshness_and_delay_fail_closed": channel["schedule"]["max_progress_sample_age_ms"] == 1000.0 and channel["schedule"]["max_progress_to_command_ms"] == 1100.0,
        "v12_preserved_settings_match": bool(host.v12_semantic_identity(synthetic, DEPLOYMENT_ROOT)),
        "v12_channel_diff_is_declared": host.v12_channel_identity(synthetic, DEPLOYMENT_ROOT)["v13_schedule"]["schedule_id"] == "GOOD_TO_BAD",
        "v12_registration_still_sealed": sha(v12_path) == host.V12_REGISTRATION_FILE_SHA256,
        "route_and_controller_sources_inventory_bound": all(name in C.exact_sources(ROOT) for name in (f"{C.PACKAGE}/route_progress.py", f"{C.PACKAGE}/good_to_bad_controller.py")),
    }
    result = dict(schema=REGRESSION_SCHEMA, status="PASS" if all(checks.values()) else "FAIL",
                  checks=checks, launcher_binding=C.launcher_binding(), sources=C.exact_sources(ROOT),
                  channel=channel, role_constant_audit=C.role_constant_audit(),
                  v12_registration_path=str(v12_path), v12_registration_sha256=sha(v12_path),
                  live_services_started=False, carla_started=False, oai_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha(out / "RESULT.json")}, indent=1) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks), sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
