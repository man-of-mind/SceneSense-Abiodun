#!/usr/bin/env python3
"""Run the six registered Route B v3 additional train-only episodes, sequentially, once.

This is a wrapper, not a second collector.  Every episode is launched by
``run_canonical_campaign_v1._run_episode`` verbatim, so the launch command, the fresh
Epic ``-RenderOffScreen`` CARLA per episode, the create-only outputs, the no-retry rule
and the gate evaluation are the canonical ones.  Only split, episode identity, scenario
seed, Traffic Manager seed and density differ, and those come from a registered matrix.

A fail-closed preflight runs once before episode 1.  Each episode is additionally held
to the registered acceptance list before the campaign is allowed to continue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_COLLECTION = HERE.parent
REPO_ROOT = DATA_COLLECTION.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_collection.run_route_b_collection_supervised_v1 as base  # noqa: E402
import data_collection.route_b_perception_v3.run_canonical_campaign_v1 as canonical  # noqa: E402
from data_collection.route_b_perception_v3.run_smoke_supervised_v1 import (  # noqa: E402
    _delete_failed_heavy_payload,
)

CONFIG_DEFAULT = HERE / "additional_train_campaign_v1.json"

# The only change to the frozen v3 collector is the additive registration of the six
# train-only episode tuples.  Reverting exactly these hunks must reproduce the collector
# SHA-256 recorded in the retained canonical v3 collection report; the preflight proves it.
COLLECTOR_REGISTRATION_EDITS = (
    (
        """    ("test", "traffic_50_50", 702, 1702),
}

# Six additional independent train-only episodes, registered for the Route B v3.1
# expanded training view.  Purely additive: the canonical eight above are untouched and
# every other bound - 25 km/h, fast rasterizer, 2.0 s replenish, 600 s budget, roadblock
# clearing, no hybrid physics - still applies unchanged to these tuples.
ADDITIONAL_TRAIN_EPISODE_KEYS = {
    ("train", "traffic_30_30", 801, 1801),
    ("train", "traffic_50_50", 802, 1802),
    ("train", "traffic_30_30", 803, 1803),
    ("train", "traffic_50_50", 804, 1804),
    ("train", "traffic_30_30", 805, 1805),
    ("train", "traffic_50_50", 806, 1806),
}
REGISTERED_EPISODE_KEYS = CANONICAL_EPISODE_KEYS | ADDITIONAL_TRAIN_EPISODE_KEYS
""",
        """    ("test", "traffic_50_50", 702, 1702),
}
""",
    ),
    (
        "    canonical_request = requested in REGISTERED_EPISODE_KEYS\n",
        "    canonical_request = requested in CANONICAL_EPISODE_KEYS\n",
    ),
    (
        "    v2.ALLOWED_SEED_BUNDLES.update((key[2], key[3]) for key in REGISTERED_EPISODE_KEYS)\n",
        "    v2.ALLOWED_SEED_BUNDLES.update((key[2], key[3]) for key in CANONICAL_EPISODE_KEYS)\n",
    ),
)
GIB = 1024 ** 3
PREFLIGHT_FAILED = "ROUTE_B_V3_1_COLLECTION_PREFLIGHT_FAILED"


# --------------------------------------------------------------------------- preflight

def _carla_processes() -> list[dict[str, Any]]:
    """Live CARLA processes read from /proc; never a name-matching killer."""
    found: list[dict[str, Any]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmdline = Path("/proc", entry, "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
        except OSError:
            continue
        if "CarlaUnreal" in cmdline or "CarlaUE4" in cmdline:
            found.append({"pid": int(entry), "cmdline": cmdline[:200]})
    return found


def _docker_containers() -> list[str]:
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}} {{.Image}}"],
                             text=True, capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001 - docker absence is not a preflight failure
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def _gpu_compute_apps() -> list[dict[str, str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], text=True, capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        return []
    apps = []
    for line in out.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 3:
            apps.append({"pid": parts[0], "process_name": parts[1], "used_memory": parts[2]})
    return apps


def _port_free(port: int, host: str) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) != 0


def preflight(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    spec = config["preflight"]
    checks: dict[str, Any] = {}
    detail: dict[str, Any] = {}

    report_path = REPO_ROOT / spec["canonical_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    detail["canonical_report"] = {
        "path": str(report_path), "sha256": canonical._sha256(report_path),
        "terminal": report.get("terminal"),
    }
    checks["canonical_report_passed"] = report.get("terminal") == "ROUTE_B_V3_CANONICAL_COLLECTION_PASSED"

    actual_hashes = {
        "collector_sha256": canonical._sha256(canonical.V3_RUNNER),
        "config_sha256": canonical._sha256(canonical.V3_CONFIG),
        "visibility_helper_sha256": canonical._sha256(canonical.VISIBILITY_HELPER),
    }
    reverted = canonical.V3_RUNNER.read_text(encoding="utf-8")
    for added, original in COLLECTOR_REGISTRATION_EDITS:
        if reverted.count(added) != 1:
            reverted = None
            break
        reverted = reverted.replace(added, original)
    reverted_hash = (hashlib.sha256(reverted.encode("utf-8")).hexdigest()
                     if reverted is not None else None)
    detail["collection_code_hashes"] = {
        "expected_from_canonical_report": report["hashes"], "actual": actual_hashes,
        "collector": str(canonical.V3_RUNNER), "config": str(canonical.V3_CONFIG),
        "visibility_helper": str(canonical.VISIBILITY_HELPER),
        "collector_registration_reverted_sha256": reverted_hash,
        "collector_change_scope": (
            "additive registration of the six train-only episode tuples only; reverting "
            "exactly those three hunks reproduces the canonical-report collector hash"),
    }
    checks["config_and_visibility_helper_hashes_match_canonical_report"] = (
        actual_hashes["config_sha256"] == report["hashes"]["config_sha256"]
        and actual_hashes["visibility_helper_sha256"] == report["hashes"]["visibility_helper_sha256"])
    checks["collector_differs_only_by_registered_additive_episode_keys"] = (
        reverted_hash == report["hashes"]["collector_sha256"])

    catalog_path = REPO_ROOT / spec["static_catalog"]
    import csv as _csv
    with catalog_path.open("r", encoding="utf-8", newline="") as stream:
        catalog = list(_csv.DictReader(stream))
    map_names = {row["map_name"] for row in catalog}
    map_hashes = {row["map_sha256"] for row in catalog}
    detail["static_environment_catalog"] = {
        "path": str(catalog_path), "sha256": canonical._sha256(catalog_path),
        "rows": len(catalog), "map_names": sorted(map_names), "map_sha256": sorted(map_hashes),
    }
    checks["static_catalog_present_and_nonempty"] = len(catalog) > 0
    checks["static_catalog_map_is_town10hd_opt"] = map_names == {spec["expected_map_name"]}
    checks["static_catalog_map_sha256_matches"] = map_hashes == {spec["expected_map_sha256"]}

    route_meta = json.loads((Path(report["episodes"][0]["output_dir"]) / "metadata.json").read_text(
        encoding="utf-8"))
    route_file = Path(route_meta["route_file"])
    detail["route"] = {
        "path": str(route_file), "sha256": canonical._sha256(route_file),
        "expected_sha256": spec["route_file_sha256"], "map": route_meta["world"],
    }
    checks["route_b_file_hash_matches_canonical"] = detail["route"]["sha256"] == spec["route_file_sha256"]

    carla_live = _carla_processes()
    containers = _docker_containers()
    detail["carla_processes"] = carla_live
    detail["docker_containers"] = containers
    checks["no_carla_process_running"] = not carla_live
    checks["no_carla_container_running"] = not any("carla" in line.lower() for line in containers)

    detail["ports"] = {str(args.port): _port_free(int(args.port), args.host),
                       str(args.tm_port): _port_free(int(args.tm_port), args.host)}
    checks["rpc_and_tm_ports_free"] = all(detail["ports"].values())

    system_python = Path(spec["system_python"])
    venv_python = Path(spec["carla_venv_python"])
    venv_carla = subprocess.run(
        [str(venv_python), "-c", "import carla; print(carla.__file__)"],
        text=True, capture_output=True, env=base.child_env())
    detail["python_roles"] = {
        "system_python": str(system_python), "system_python_present": system_python.is_file(),
        "carla_venv_python": str(venv_python), "carla_venv_python_present": venv_python.is_file(),
        "carla_venv_python_is_launcher_venv": str(venv_python) == str(base.VENV_PYTHON),
        "venv_carla_import_returncode": venv_carla.returncode,
        "venv_carla_module": venv_carla.stdout.strip(),
    }
    checks["python_roles_match_prior_successful_collection"] = (
        system_python.is_file() and venv_python.is_file()
        and str(venv_python) == str(base.VENV_PYTHON) and venv_carla.returncode == 0)

    gpu_apps = _gpu_compute_apps()
    allowed = tuple(spec["gpu_allowed_process_names"])
    unexpected = [app for app in gpu_apps
                  if not any(name in app["process_name"] for name in allowed)]
    detail["gpu_compute_apps"] = gpu_apps
    detail["gpu_unexpected_apps"] = unexpected
    checks["no_model_training_process_on_gpu"] = not unexpected

    sizes_by_density: dict[str, int] = {}
    for item in report["episodes"]:
        if not item.get("passed"):
            continue
        density = str(item["density"])
        sizes_by_density[density] = max(sizes_by_density.get(density, 0), int(item["output_bytes"]))
    projected = sum(sizes_by_density[str(row["density"])] for row in config["episodes"])
    margin = int(float(spec["safety_margin_gib"]) * GIB)
    free = shutil.disk_usage(
        args.output_root if args.output_root.exists() else args.output_root.parent).free
    detail["storage"] = {
        "retained_canonical_max_bytes_by_density": sizes_by_density,
        "projected_six_episode_bytes": projected,
        "safety_margin_bytes": margin,
        "required_bytes": projected + margin,
        "free_bytes": free,
        "projected_gib": round(projected / GIB, 2),
        "required_gib": round((projected + margin) / GIB, 2),
        "free_gib": round(free / GIB, 2),
    }
    checks["disk_free_covers_projection_plus_30gib"] = free >= projected + margin

    existing = [str(args.output_root / row["name"]) for row in config["episodes"]
                if (args.output_root / row["name"]).exists()]
    detail["existing_outputs"] = existing
    checks["create_only_episode_outputs_absent"] = not existing

    return {"passed": all(bool(value) for value in checks.values()),
            "checks": checks, "detail": detail}


# --------------------------------------------------------------- per-episode acceptance

def _acceptance(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """The registered per-episode acceptance list, read from the episode's own summary."""
    summary_path = output_dir / "route_summary.json"
    if not summary_path.is_file():
        return {"available": False, "checks": {"episode_summary_present": False}}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = summary.get("gates", {})
    v2_gates = summary.get("v2_gates", {})
    v3_gates = summary.get("v3_gates", {})
    route = summary.get("route_result") or {}
    policy = summary.get("intervention_policy", {})
    alignment = summary.get("sensor_alignment", {})
    visibility = summary.get("depth_visibility", {})
    cadence = summary.get("cadence", {})
    population = summary.get("population", {})
    controller = summary.get("controller_health", {})
    cleanup = summary.get("sensor_cleanup", {})

    permitted = set(policy.get("permitted_actions", []))
    events = policy.get("intervention_events", [])
    # The RGB/semantic/depth/radar alignment delta is the timestamp and the two transform
    # deltas.  camera_frame_parity is deliberately NOT part of it: it is the phase of the
    # first camera frame inside the two-tick prepare cycle (camera_frame % 2), so canonical
    # episodes legitimately show both 0 and 1.  It is recorded, not gated.
    alignment_delta = max(
        float(alignment.get("max_timestamp_delta_s", 1.0)),
        float(alignment.get("max_camera_transform_delta_m", 1.0)),
        float(alignment.get("max_radar_transform_delta_m", 1.0)),
    )
    checks = {
        "all_v2_gates_pass": bool(v2_gates) and all(bool(value) for value in v2_gates.values()),
        "all_v3_gates_pass": bool(v3_gates) and all(bool(value) for value in v3_gates.values()),
        "all_gates_pass": bool(gates) and all(bool(value) for value in gates.values()),
        "route_completed": bool(route.get("completed")) and bool(route.get("all_ordered_waypoints_reached")),
        "no_watchdog_abort": bool(gates.get("no_watchdog_abort")) and not route.get("abort_reason"),
        "frame_counts_reconcile_raw_prepared_saved": (
            int(cadence.get("prepared_inputs", {}).get("count", -1))
            == int(summary.get("prepared_inputs", -2))
            and int(cadence.get("saved_frames", {}).get("count", -1))
            == int(summary.get("saved_samples", -2))
            and int(cadence.get("raw_callbacks", -1))
            >= int(summary.get("prepared_inputs", 0)) >= int(summary.get("saved_samples", 0)) > 0
            and int(cadence.get("dropped_callback_frames", -1)) == 0
            and int(cadence.get("duplicate_callbacks", -1)) == 0
            and int(cadence.get("out_of_order_callbacks", -1)) == 0
            and bool(gates.get("zero_missing_or_corrupt_records"))),
        "prepared_and_saved_alignment_records_exact": bool(gates.get(
            "depth_callback_frame_timestamp_alignment_exact_prepared_and_saved"))
            and bool(gates.get("no_missing_or_invalid_depth_frame"))
            and bool(gates.get("depth_frame_tick_ownership_exact_no_hidden_ticks")),
        "modality_alignment_delta_zero": alignment_delta == 0.0
            and not alignment.get("frame_content_failures")
            and int(alignment.get("frame_content_checks", 0)) == int(summary.get("prepared_inputs", -1))
            and bool(gates.get("sensor_frames_exactly_aligned"))
            and bool(gates.get("rgb_depth_colocation_exact")),
        "visibility_rows_equal_object_rows": int(visibility.get("visibility_rows", -1))
            == int(visibility.get("object_rows_reconciled", -2)),
        "only_approved_intervention_actions": all(
            str(event.get("action", "")).upper() in permitted for event in events),
        "no_unexpected_intervention_events": not policy.get("unexpected_intervention_events"),
        "no_forced_overtaking": policy.get("forced_overtaking") is False
            and int(policy.get("maximum_overtakes", -1)) == 0
            and int(route.get("ego_overtakes", -1)) == 0,
        "population_deficit_within_bounds": bool(gates.get(
            "no_population_deficit_beyond_replenish_plus_2s")) and bool(gates.get(
            "population_alive_95pct_every_saved_frame")),
        "controller_deficit_within_bounds": bool(gates.get(
            "no_controller_deficit_beyond_replenish_plus_2s")),
        "sensor_cleanup_succeeded": bool(summary.get("sensor_cleanup_succeeded")),
        "carla_cleanup_verified": bool(record.get("carla_shutdown", {}).get("shutdown_verified")),
        "client_exit_clean": int(summary.get("client_returncode", 1)) == 0,
    }
    observed = {
        "gate_counts": {"v2": len(v2_gates), "v3": len(v3_gates), "total": len(gates)},
        "failed_gates": sorted(name for name, value in gates.items() if not value),
        "raw_callbacks": cadence.get("raw_callbacks"),
        "prepared_inputs": summary.get("prepared_inputs"),
        "saved_samples": summary.get("saved_samples"),
        "logical_sweeps": (cadence.get("logical_sweeps") or {}).get("count"),
        "alignment": alignment,
        "visibility_rows": visibility.get("visibility_rows"),
        "object_rows_reconciled": visibility.get("object_rows_reconciled"),
        "person_retained_v010_percent": visibility.get("geometry_qualified_person_retained_v010_percent"),
        "person_retained_v025_percent": visibility.get("geometry_qualified_person_retained_v025_percent"),
        "tier_counts_by_class_and_distance": visibility.get("tier_counts_by_class_and_distance"),
        "intervention_events": events,
        "permitted_actions": sorted(permitted),
        "driven_distance_m": route.get("driven_distance_m"),
        "collision_count": route.get("collision_count"),
        "collision_incident_count": route.get("collision_incident_count"),
        "npc_roadblocks_cleared": route.get("npc_roadblocks_cleared"),
        "population_deficit_spans": population.get("deficit_spans"),
        "population_max_deficit_span_s": population.get("max_deficit_span_s"),
        "controller_deficit_spans": controller.get("deficit_spans"),
        "controller_deficit_limit_s": controller.get("deficit_limit_s"),
        "sensor_cleanup": cleanup.get("sensors"),
        "storage": summary.get("storage"),
    }
    return {"available": True, "passed": all(checks.values()), "checks": checks, "observed": observed}


# ------------------------------------------------------------------------------- report

def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for item in report["episodes"]:
        observed = (item.get("acceptance") or {}).get("observed", {})
        rows.append(
            f"| {item['label']} | {item['episode']} | {item['split']} | {item['density']} | "
            f"{item['scenario_seed']}/{item['tm_seed']} | "
            f"{item.get('route_wall_duration_s', 0.0):.1f}/{item.get('simulation_duration_s', 0.0):.1f} | "
            f"{item.get('saved_frames', 0)}/{item.get('prepared_frames', 0)}/{item.get('raw_callbacks', 0)} | "
            f"{'PASS' if item.get('passed') else 'FAIL'} | "
            f"{observed.get('gate_counts', {}).get('total', 0)} | "
            f"{item.get('visibility_rows', 0)}/{item.get('object_rows', 0)} | "
            f"{item.get('person_retained_v010_percent', 0.0):.2f}/{item.get('person_retained_v025_percent', 0.0):.2f}% | "
            f"{item.get('marginal_person_within_40m', 0)}/{item.get('unobservable_person_within_40m', 0)} | "
            f"{item.get('permitted_intervention_count', 0)} | {item.get('output_bytes', 0)} |"
        )
    text = f"""# Route B v3 additional train-only collection report

Terminal: `{report['terminal']}`

| label | episode | split | density | scenario/TM | wall/sim s | saved/prepared/raw | accepted | gates | visibility/object rows | person retained v010/v025 | marginal/unobservable <=40m | interventions | bytes |
|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Total new corpus bytes: {report['total_corpus_bytes']}.
- Remaining disk bytes: {report['remaining_disk_bytes']}.
- All CARLA servers shut down: `{report['all_carla_shutdown_verified']}`.
- v3 collector SHA-256: `{report['hashes']['collector_sha256']}`.
- v3 config SHA-256: `{report['hashes']['config_sha256']}`.
- Visibility helper SHA-256: `{report['hashes']['visibility_helper_sha256']}`.
- Automatic retry: `{report['automatic_retry']}`.

No training, evaluation, model inference, checkpoint load, q/AE, OAI or 288-measurement work
occurred, and no locked-test episode was read, resolved or referenced.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--preflight-json", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--carla-ready-timeout-s", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_root = args.output_root.resolve()
    args.report_json = args.report_json.resolve()
    args.report_md = args.report_md.resolve()
    args.preflight_json = args.preflight_json.resolve()
    args.sentinel = args.sentinel.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    for path in (args.report_json, args.report_md, args.preflight_json, args.sentinel):
        if path.exists():
            print(f"create-only path already exists: {path}", file=sys.stderr)
            return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.sentinel.parent.mkdir(parents=True, exist_ok=True)

    check = preflight(args, config)
    check.update({"schema": "route_b_v3_additional_train.preflight.v1",
                  "config_sha256": canonical._sha256(args.config)})
    with args.preflight_json.open("x", encoding="utf-8") as stream:
        json.dump(check, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    if not check["passed"]:
        failed = sorted(name for name, value in check["checks"].items() if not value)
        args.sentinel.write_text(f"{PREFLIGHT_FAILED} {failed}\n", encoding="utf-8")
        print(json.dumps({"terminal": PREFLIGHT_FAILED, "failed_checks": failed}, indent=2))
        return 2

    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "route_b_perception_v3.additional_train_campaign.v1",
        "terminal": "RUNNING", "episodes": [], "automatic_retry": False,
        "config": config, "config_sha256": canonical._sha256(args.config),
        "preflight": check,
        "hashes": {
            "collector_sha256": canonical._sha256(canonical.V3_RUNNER),
            "config_sha256": canonical._sha256(canonical.V3_CONFIG),
            "visibility_helper_sha256": canonical._sha256(canonical.VISIBILITY_HELPER),
            "campaign_wrapper_sha256": canonical._sha256(Path(__file__).resolve()),
            "canonical_campaign_sha256": canonical._sha256(
                Path(canonical.__file__).resolve()),
        },
        "test_policy": "no locked-test episode is collected, read, resolved or referenced",
    }

    stopped_at: int | None = None
    for row in config["episodes"]:
        spec = (int(row["episode"]), row["split"], row["density"],
                int(row["scenario_seed"]), int(row["tm_seed"]), row["name"])
        result = canonical._run_episode(args, spec)
        result["label"] = row["label"]
        output_dir = Path(result["output_dir"])
        acceptance = _acceptance(result, output_dir)
        result["acceptance"] = acceptance
        accepted = bool(result.get("passed")) and bool(acceptance.get("passed"))
        if result.get("passed") and not acceptance.get("passed"):
            result["passed"] = False
            result["failure"] = "registered per-episode acceptance failure"
            if output_dir.exists():
                result["failed_payload_reclaim"] = _delete_failed_heavy_payload(output_dir)
        report["episodes"].append(result)
        print(json.dumps({"episode_boundary": {
            "label": row["label"], "episode": row["episode"], "accepted": accepted,
            "output_dir": str(output_dir), "bytes": result.get("output_bytes"),
        }}, sort_keys=True), flush=True)
        if not accepted:
            stopped_at = int(row["episode"])
            report["terminal"] = (
                f"ROUTE_B_V3_1_ADDITIONAL_TRAIN_COLLECTION_STOPPED_EPISODE_{stopped_at}")
            break

    if stopped_at is None:
        report["terminal"] = "ROUTE_B_V3_1_ADDITIONAL_TRAIN_COLLECTION_EPISODES_PASSED"
    report["all_carla_shutdown_verified"] = bool(report["episodes"]) and all(
        item.get("carla_shutdown", {}).get("shutdown_verified") for item in report["episodes"])
    report["total_corpus_bytes"] = sum(
        int(item.get("output_bytes", 0)) for item in report["episodes"] if item.get("passed"))
    report["remaining_disk_bytes"] = shutil.disk_usage(args.output_root).free
    report["wall_seconds"] = time.monotonic() - started
    with args.report_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    _write_markdown(args.report_md, report)
    args.sentinel.write_text(
        f"{report['terminal']} episodes={len(report['episodes'])}\n", encoding="utf-8")
    print(json.dumps({
        "terminal": report["terminal"], "episodes_completed": len(report["episodes"]),
        "report_json": str(args.report_json), "report_md": str(args.report_md),
    }, indent=2), flush=True)
    return 0 if stopped_at is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
