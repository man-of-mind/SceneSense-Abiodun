#!/usr/bin/env python3
"""Run the eight registered Route B v3 canonical episodes sequentially, once."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from data_collection.route_b_perception_v3.run_smoke_supervised_v1 import (  # noqa: E402
    _delete_failed_heavy_payload,
    _gpu_used_mib,
    _host_ram_used_kib,
    _patch_summary_after_shutdown,
    _proc_rss_kib,
    _tree_bytes,
)


V3_RUNNER = DATA_COLLECTION / "run_route_b_perception_collection_v3.py"
V3_CONFIG = DATA_COLLECTION / "configs" / "route_b_perception_v3.yaml"
VISIBILITY_HELPER = HERE / "visibility_v1.py"
PLAN = (
    (1, "train", "traffic_30_30", 501, 1501, "canonical_v3_01_train_30_30_s501_tm1501"),
    (2, "train", "traffic_50_50", 502, 1502, "canonical_v3_02_train_50_50_s502_tm1502"),
    (3, "train", "traffic_30_30", 503, 1503, "canonical_v3_03_train_30_30_s503_tm1503"),
    (4, "train", "traffic_50_50", 504, 1504, "canonical_v3_04_train_50_50_s504_tm1504"),
    (5, "val", "traffic_30_30", 601, 1601, "canonical_v3_05_val_30_30_s601_tm1601"),
    (6, "val", "traffic_50_50", 602, 1602, "canonical_v3_06_val_50_50_s602_tm1602"),
    (7, "test", "traffic_30_30", 701, 1701, "canonical_v3_07_test_30_30_s701_tm1701"),
    (8, "test", "traffic_50_50", 702, 1702, "canonical_v3_08_test_50_50_s702_tm1702"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within_40_count(summary: dict[str, Any], tier: str) -> int:
    counts = summary["depth_visibility"]["tier_counts_by_class_and_distance"]
    return sum(
        int(value) for key, value in counts.items()
        if key.startswith(f"person:{tier}:") and not key.endswith(":over_40m")
    )


def _run_episode(args: argparse.Namespace, spec: tuple[Any, ...]) -> dict[str, Any]:
    number, split, density, scenario_seed, tm_seed, name = spec
    output_dir = (args.output_root / name).resolve()
    server_log = output_dir.parent / f"{name}_carla_server.log"
    client_log = output_dir.parent / f"{name}_client.log"
    record: dict[str, Any] = {
        "episode": number, "split": split, "density": density,
        "scenario_seed": scenario_seed, "tm_seed": tm_seed,
        "output_dir": str(output_dir), "server_log": str(server_log),
        "client_log": str(client_log), "attempts": 1, "retry_permitted": False,
    }
    if output_dir.exists():
        record.update({"passed": False, "failure": "create-only output already exists"})
        return record
    if base.rpc_port_listening(args.port, args.host) or base.rpc_port_listening(args.tm_port, args.host):
        record.update({"passed": False, "failure": "CARLA or Traffic Manager port already in use"})
        return record

    server, pgid = base.start_carla(int(args.port), server_log)
    record.update({"carla_pid": server.pid, "carla_pgid": pgid})
    client_returncode = 2
    peak_client_rss = 0
    peak_host_used = _host_ram_used_kib()
    peak_gpu_used = _gpu_used_mib()
    started = time.monotonic()
    shutdown: dict[str, Any] = {"shutdown_verified": False}
    try:
        version = base.wait_for_rpc(int(args.port), float(args.carla_ready_timeout_s))
        record["carla_rpc_ready"] = version is not None
        record["carla_server_version"] = version
        if version is None:
            record["failure"] = "fresh Epic CARLA did not become RPC-ready"
        else:
            command = [
                str(base.VENV_PYTHON), str(V3_RUNNER),
                "--density", density, "--split", split,
                "--output-dir", str(output_dir), "--host", str(args.host),
                "--port", str(args.port), "--tm-port", str(args.tm_port),
                "--scenario-seed", str(scenario_seed), "--tm-seed", str(tm_seed),
                "--target-speed-kph", "25.0", "--rasterizer", "fast",
                "--replenish-interval-s", "2.0", "--maximum-loop-sim-s", "600.0",
                "--no-hybrid-physics", "--allow-roadblock-clearing",
            ]
            record["client_argv"] = command
            with client_log.open("wb") as stream:
                child = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, env=base.child_env())
                while child.poll() is None:
                    peak_client_rss = max(peak_client_rss, _proc_rss_kib(child.pid))
                    peak_host_used = max(peak_host_used, _host_ram_used_kib())
                    gpu = _gpu_used_mib()
                    if gpu is not None:
                        peak_gpu_used = gpu if peak_gpu_used is None else max(peak_gpu_used, gpu)
                    time.sleep(5.0)
                client_returncode = int(child.returncode)
            record["client_exit"] = base.describe_exit(client_returncode)
    finally:
        shutdown = base.stop_carla(server, pgid, int(args.port))
        record["carla_shutdown"] = shutdown

    resources = {
        "sampling_interval_s": 5.0,
        "peak_client_rss_kib": peak_client_rss,
        "peak_host_ram_used_kib": peak_host_used,
        "peak_gpu_memory_used_mib": peak_gpu_used,
        "gpu_scope": "sum of whole-device memory.used across visible GPUs",
    }
    summary = _patch_summary_after_shutdown(
        output_dir, shutdown, resources, client_returncode)
    passed = bool(
        summary is not None
        and summary.get("terminal") == "ROUTE_B_V3_CANONICAL_EPISODE_PASSED"
        and summary.get("status") == "COLLECTION_EPISODE_PASSED"
        and all(bool(value) for value in summary.get("gates", {}).values())
        and bool(shutdown.get("shutdown_verified"))
        and client_returncode == 0
    )
    record.update({
        "wall_seconds_supervised": time.monotonic() - started,
        "resources": resources, "passed": passed,
        "terminal": (summary or {}).get("terminal", "ROUTE_B_V3_COLLECTION_FAILED"),
    })
    if summary is not None:
        visibility = summary["depth_visibility"]
        record.update({
            "simulation_duration_s": float((summary.get("route_result") or {}).get(
                "simulation_duration_s", 0.0)),
            "route_wall_duration_s": float((summary.get("route_result") or {}).get(
                "wall_clock_duration_s", 0.0)),
            "saved_frames": int(summary["saved_samples"]),
            "prepared_frames": int(summary["prepared_inputs"]),
            "raw_callbacks": int(summary["cadence"]["raw_callbacks"]),
            "visibility_rows": int(visibility["visibility_rows"]),
            "object_rows": int(visibility["object_rows_reconciled"]),
            "person_retained_v010_percent": float(
                visibility["geometry_qualified_person_retained_v010_percent"]),
            "person_retained_v025_percent": float(
                visibility["geometry_qualified_person_retained_v025_percent"]),
            "marginal_person_within_40m": _within_40_count(summary, "marginal_or_heavily_occluded"),
            "unobservable_person_within_40m": _within_40_count(summary, "unobservable"),
            "permitted_intervention_count": int(
                summary.get("intervention_policy", {}).get("intervention_count", 0)),
            "failed_gates": sorted(
                key for key, value in summary.get("gates", {}).items() if not value),
        })
    record["output_bytes"] = _tree_bytes(output_dir)
    if not passed:
        record["failed_payload_reclaim"] = (
            _delete_failed_heavy_payload(output_dir) if output_dir.exists()
            else {"deleted_explicit_paths": [], "bytes_reclaimed": 0, "recoverable": False}
        )
    return record


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for item in report["episodes"]:
        rows.append(
            f"| {item['episode']} | {item['split']} | {item['density']} | "
            f"{item['scenario_seed']}/{item['tm_seed']} | "
            f"{item.get('route_wall_duration_s', 0.0):.1f}/{item.get('simulation_duration_s', 0.0):.1f} | "
            f"{item.get('saved_frames', 0)}/{item.get('prepared_frames', 0)}/{item.get('raw_callbacks', 0)} | "
            f"{'PASS' if item.get('passed') else 'FAIL'} | "
            f"{item.get('visibility_rows', 0)}/{item.get('object_rows', 0)} | "
            f"{item.get('person_retained_v010_percent', 0.0):.2f}/{item.get('person_retained_v025_percent', 0.0):.2f}% | "
            f"{item.get('marginal_person_within_40m', 0)}/{item.get('unobservable_person_within_40m', 0)} | "
            f"{item.get('permitted_intervention_count', 0)} | {item.get('output_bytes', 0)} |"
        )
    text = f"""# Route B v3 canonical collection report

Terminal: `{report['terminal']}`

| episode | split | density | scenario/TM | wall/sim s | saved/prepared/raw | gates | visibility/object rows | person retained v010/v025 | marginal/unobservable <=40m | interventions | bytes |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Total corpus bytes: {report['total_corpus_bytes']}.
- Remaining disk bytes: {report['remaining_disk_bytes']}.
- Evidence archive: `{report['archive']['path']}` ({report['archive']['bytes']} bytes), SHA-256 `{report['archive']['sha256']}`.
- Superseded raw space reclaimed: {report['raw_reclaimed_bytes']} bytes.
- All eight CARLA servers shut down: `{report['all_carla_shutdown_verified']}`.
- v3 collector SHA-256: `{report['hashes']['collector_sha256']}`.
- v3 config SHA-256: `{report['hashes']['config_sha256']}`.
- Visibility helper SHA-256: `{report['hashes']['visibility_helper_sha256']}`.

No training, evaluation, q/AE work, model inference, test inspection, agent measurement, commit, push, or OAI modification occurred. Test episodes were collected and structurally reconciled only.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--raw-reclaimed-bytes", type=int, required=True)
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
    archive = args.archive_path.resolve()
    if args.report_json.exists() or args.report_md.exists():
        print("create-only campaign report path already exists", file=sys.stderr)
        return 2
    if not archive.is_file():
        print(f"verified evidence archive is missing: {archive}", file=sys.stderr)
        return 2
    existing_outputs = [str(args.output_root / spec[5]) for spec in PLAN
                        if (args.output_root / spec[5]).exists()]
    if existing_outputs:
        print(f"create-only canonical outputs already exist: {existing_outputs}", file=sys.stderr)
        return 2
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "route_b_perception_v3.canonical_campaign.v1",
        "terminal": "RUNNING", "episodes": [],
        "archive": {"path": str(archive), "bytes": archive.stat().st_size,
                    "sha256": _sha256(archive)},
        "raw_reclaimed_bytes": int(args.raw_reclaimed_bytes),
        "hashes": {
            "collector_sha256": _sha256(V3_RUNNER),
            "config_sha256": _sha256(V3_CONFIG),
            "visibility_helper_sha256": _sha256(VISIBILITY_HELPER),
        },
        "automatic_retry": False,
        "test_policy": "collection and structural reconciliation only; no inspection/evaluation",
    }
    for spec in PLAN:
        result = _run_episode(args, spec)
        report["episodes"].append(result)
        print(json.dumps({"episode_boundary": result}, sort_keys=True), flush=True)
        if not result.get("passed"):
            report["terminal"] = (
                f"ROUTE_B_V3_CANONICAL_COLLECTION_STOPPED_EPISODE_{spec[0]}_FAILED")
            break
    if len(report["episodes"]) == len(PLAN) and all(
            item.get("passed") for item in report["episodes"]):
        report["terminal"] = "ROUTE_B_V3_CANONICAL_COLLECTION_PASSED"
    report["all_carla_shutdown_verified"] = bool(report["episodes"]) and all(
        item.get("carla_shutdown", {}).get("shutdown_verified")
        for item in report["episodes"])
    report["total_corpus_bytes"] = sum(
        int(item.get("output_bytes", 0)) for item in report["episodes"] if item.get("passed"))
    report["remaining_disk_bytes"] = shutil.disk_usage(args.output_root).free
    report["wall_seconds"] = time.monotonic() - started
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_markdown(args.report_md, report)
    print(json.dumps({
        "terminal": report["terminal"], "episodes_completed": len(report["episodes"]),
        "report_json": str(args.report_json), "report_md": str(args.report_md),
    }, indent=2), flush=True)
    return 0 if report["terminal"] == "ROUTE_B_V3_CANONICAL_COLLECTION_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
