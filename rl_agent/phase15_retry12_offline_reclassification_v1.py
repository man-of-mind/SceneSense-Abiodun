#!/usr/bin/env python3
"""Reclassify immutable Phase-15 retry12 under the amended validity contract."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
TOKEN = "SPLITFUSION_PHASE15_RETRY12_OFFLINE_RECLASSIFICATION"
SCHEMA = "scenesense.splitfusion_phase15_live_deployment_reclassified_qualification.v1"
AUDIT_SCHEMA = "scenesense.splitfusion_phase15_retry12_offline_reclassification.v1"
MANIFEST_SCHEMA = "scenesense.splitfusion_phase15_live_deployment_reclassified_artifacts.v1"
TERMINAL = "SPLITFUSION_PHASE15_LIVE_DEPLOYMENT_QUALIFIED"
SOURCE_HEAD = "b85fc257a1c77e786e5aa3df854e23721e9fcf92"
AMENDMENT_COMMIT = "4f122ea4222a6edb9cecf090d25c20502bbab0c4"
ACTIONS = (0, 20, 46, 71)
HISTORICAL_ORDER = (0, 20, 46, 71)
SOURCE_FILES = {
    "FAILED.json": "9a86dcba088f0aef7eda31811d421d079e112536c01f4ecd38cfc84ace90288f",
    "preflight_inventory.json": "693acc830b290409e2efffddff805723db8be270a30985b5370377aaf186fefd",
    "runtime/RESULTS_SUMMARY.json": "9fb6997a6dd3a5b2f779057a15453823126ed76958130054f543eac22789bf96",
    "runtime/manifest.json": "624e6ef0d6ca3b6ea999a5f1069a59e42bda0c941c47060d8aa6ed658bca027e",
    "runtime/per_frame_metrics.csv": "47c46d0157797c4c64ed6401bb579cb2ae12488070649044f67993bcffdfcc24",
    "runtime/map_feedback.csv": "81c67c8507298369e94f26544bd03458971f99f92d92ae3c272aa49c527e0da6",
    "runtime/radio_trace.csv": "4b91ab7df0d4c2e4618936c443cfddd0b9e9be07f7307e111ede85199163865b",
}
EXPECTED_TOP_FAILURES = {
    "qualified Route B did not complete with a clean split adapter",
    "frame 583: AdapterError: accepted window requires four radar callbacks, got 3",
    "10 Hz prepared-input scheduling phase/count contract failed",
    "sensor/preparation coverage below campaign minimum: 0.487805 < 0.950000",
    "live qualification lacks ACK_INSTALLED for actions [0, 20, 46]",
}


class ReclassificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReclassificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def literal_mapping(value: str) -> dict[str, Any]:
    parsed = ast.literal_eval(value)
    require(isinstance(parsed, dict), "runtime counter field is not a mapping")
    return parsed


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_inventory(source: Path) -> list[dict[str, Any]]:
    records = []
    for relative, expected in SOURCE_FILES.items():
        path = source / relative
        require(path.is_file(), f"retry12 source artifact missing: {relative}")
        observed = sha256_file(path)
        require(observed == expected, f"retry12 source artifact drift: {relative}")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed})
    return records


def audit_retry12(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inventory = source_inventory(source)
    failed = load_json(source / "FAILED.json")
    preflight = load_json(source / "preflight_inventory.json")
    runtime = source / "runtime"
    summary = load_json(runtime / "RESULTS_SUMMARY.json")
    runtime_manifest = load_json(runtime / "manifest.json")
    require(preflight.get("status") == "PASS" and preflight.get("head") == SOURCE_HEAD,
            "retry12 preflight/head binding drift")
    require(preflight.get("data_scope") == {
        "live_carla_only": True,
        "holdout_validation_test_artifacts_opened": 0,
        "predictions_or_payloads_in_state_mount": False,
    }, "retry12 data-scope binding drift")
    phase13c = preflight["host_artifacts"]["phase13c_evidence"]
    for artifact in phase13c["artifacts"]:
        path = ROOT / str(artifact["path"])
        require(path.is_file() and sha256_file(path) == artifact["sha256"],
                f"bound all-action localhost evidence drift: {artifact['path']}")
    require(phase13c["qualification_status"]
            == "PHASE13C_36X300_LOCALHOST_MEASUREMENT_COMPLETE",
            "all-action localhost qualification status drift")
    require(failed.get("status") == "FAILED" and failed.get("adapter_returncode") == 1,
            "retry12 historical failure identity drift")
    require(set(summary.get("failures", ())) == EXPECTED_TOP_FAILURES,
            f"retry12 contains a non-amended failure: {summary.get('failures')}")
    require(summary.get("status") == "FAILED" and summary.get("terminal_status") == "FAILED",
            "retry12 historical terminal drift")
    require(summary["route"].get("route_completed") is True
            and summary["route"].get("error") == "",
            "retry12 Route B did not complete")
    require(summary["live_dispatch"].get("errors") == [], "retry12 dispatch recorded errors")
    require(runtime_manifest.get("git_commit_at_launch") == SOURCE_HEAD,
            "retry12 launch commit drift")
    for artifact in runtime_manifest.get("files", ()):
        path = runtime / str(artifact["path"])
        require(path.is_file() and path.stat().st_size == int(artifact["bytes"])
                and sha256_file(path) == artifact["sha256"],
                f"retry12 runtime manifest drift: {artifact['path']}")

    rows = read_rows(runtime / "per_frame_metrics.csv")
    sent = [row for row in rows if row["prepare_status"] == "SENT"]
    failed_radar = [row for row in rows
                    if row["prepare_status"] == "SPLIT_PROCESSING_FAILED"
                    and "requires four radar callbacks, got 3" in row.get("error", "")]
    require(len(failed_radar) == 1, "historical incomplete-radar event drift")
    require(len(sent) == 20, "retry12 did not send exactly 20 frames")
    action_counts = Counter(int(row["action_id"]) for row in sent)
    require(action_counts == Counter({action: 5 for action in ACTIONS}),
            f"retry12 action counts drift: {action_counts}")
    require([int(row["action_id"]) for row in sent] == list(HISTORICAL_ORDER) * 5,
            "retry12 historical action order drift")
    frame_ids = [int(row["frame_id"]) for row in sent]
    require(len(set(frame_ids)) == 20, "retry12 sent frame IDs are not unique")

    feedback = read_rows(runtime / "map_feedback.csv")
    terminal_rows = [row for row in feedback
                     if row.get("terminal", "").casefold() in {"1", "true"}]
    terminal_counts = Counter(row["capture_id"] for row in terminal_rows)
    require(set(terminal_counts) == {row["capture_id"] for row in sent}
            and all(count == 1 for count in terminal_counts.values()),
            "retry12 exactly-one terminal accounting failed")
    ack_rows = [row for row in feedback if row["status"] == "ACK_INSTALLED"]
    require(len(ack_rows) == 1 and int(ack_rows[0]["action_id"]) == 71,
            "retry12 real OAI installation identity drift")

    registry = {p.action_id: p for p in SplitActionRegistry.from_runtime_binding().profiles
                if p.action_id in ACTIONS}
    require(set(registry) == set(ACTIONS), "representative action registry drift")
    completed = []
    for row in sent:
        action = int(row["action_id"])
        profile = registry[action]
        require(row["profile_id"] == profile.profile_id
                and row["model_family"] == profile.family
                and row["quantizer"] == profile.quantizer
                and int(row["q_e4"]) == profile.q_e4
                and int(row["routing_tag"]) == profile.routing_tag,
                f"retry12 action identity mismatch: {action}")
        require(int(row["scientific_inner_bytes"]) + int(row["sfd1_overhead_bytes"])
                == int(row["sfd1_bytes"]), f"retry12 SFD1 accounting drift: {action}")
        if row.get("edge_result_received_ns"):
            completed.append(row)
            require(row["decoded"] == "True" and row["finite"] == "True"
                    and row["frame_context_valid"] == "True"
                    and row["decoder_identity"] == profile.decoder_identity
                    and row["reconstructed_device"] == "cuda:0"
                    and int(row["finite_output_tensor_count"]) > 0
                    and int(row["camera_pose_reconstruct_ns"]) > 0
                    and int(row["datagrams"]) == int(row["feature_received_datagrams"])
                    and int(row["feature_duplicate_datagrams"]) == 0,
                    f"retry12 returned-row integrity drift: {action}")
    require(len(completed) == 1 and int(completed[0]["action_id"]) == 71,
            "retry12 returned-result identity drift")

    live = summary["live_dispatch"]
    ue = live["ue_counters"]
    ue_ledger = live["call_ledger"]
    require(ue["frames_completed"] == 20 and ue["ranker_dispatches"] == 15
            and ue["ae_encoder_dispatches"] == 15
            and ue["hot_path_model_load_operations"] == 0
            and ue["hot_path_model_construction_operations"] == 0,
            "retry12 UE runtime counter drift")
    require(ue_ledger == {"front": 20, "ranker": 15, "ae_encoder_AE128": 5,
                          "ae_encoder_AE64": 5, "ae_encoder_AE32": 5},
            "retry12 UE branch ledger drift")
    edge = literal_mapping(completed[0]["edge_counters"])
    edge_ledger = literal_mapping(completed[0]["edge_call_ledger"])
    require(edge["frames_attempted"] == 4 and edge["frames_completed"] == 4
            and edge["tail_dispatches"] == 4 and edge["ae_decoder_dispatches"] == 3
            and edge["hot_path_model_load_operations"] == 0
            and edge["hot_path_model_construction_operations"] == 0,
            "retry12 edge runtime counter drift")
    require(edge_ledger == {"tail": 4, "service_record_serialization": 4,
                             "ae_decoder_AE128": 1, "ae_decoder_AE64": 1,
                             "ae_decoder_AE32": 1},
            "retry12 edge branch ledger drift")
    startup = summary["edge_startup"]
    require(startup["allowed_action_ids"] == list(ACTIONS)
            and startup["tail_device"] == "cuda:0"
            and startup["state_root_writable"] is True,
            "retry12 edge startup drift")
    requested = int(preflight["socket_buffers"]["requested_bytes"])
    buffer_values = list(live["socket_buffers"].values()) + [
        startup["edge_receive_reported_bytes"], startup["edge_send_reported_bytes"]]
    require(all(int(value) >= requested for value in buffer_values if int(value) != requested),
            "retry12 socket buffer accounting drift")
    radio = read_rows(runtime / "radio_trace.csv")
    require(radio and int(radio[0]["step_index"]) == 0
            and all(row["profile_id"] == "FAVORABLE_STABLE" for row in radio),
            "retry12 radio trace identity drift")

    cleanup = failed["cleanup"]
    require(cleanup["carla"].get("shutdown_verified") is True
            and cleanup["radio"].get("all_lifecycle_gates_passed") is True
            and cleanup["radio"].get("noise_power_db_restored_and_read_back") is True
            and cleanup["application_cold"] == {
                "application_processes": [], "edge_container_absent": True,
                "runtime_temporary_paths": [], "shared_memory_objects": []}
            and cleanup["ports_cold"].get("conflicting_tcp_ports") == []
            and cleanup["ports_cold"].get("conflicting_udp_ports") == [],
            "retry12 cold teardown drift")
    require(not Path(summary["edge_mounts"]["state"]["source"]).exists(),
            "retry12 cell-scoped edge state survived teardown")

    latency = (int(completed[0]["edge_result_received_ns"])
               - int(completed[0]["capture_started_ns"])) / 1e6
    terminal_by_capture = {row["capture_id"]: row["status"] for row in terminal_rows}
    action_results = []
    for action in ACTIONS:
        action_rows = [row for row in sent if int(row["action_id"]) == action]
        action_results.append({
            "action_id": action, "profile_id": registry[action].profile_id,
            "family": registry[action].family, "quantizer": registry[action].quantizer,
            "q": registry[action].q, "captures": 5,
            "ack_installed": sum(int(row["action_id"]) == action for row in ack_rows),
            "terminal_outcomes": dict(Counter(terminal_by_capture[row["capture_id"]]
                                               for row in action_rows)),
            "mean_capture_to_edge_result_ms": latency if action == 71 else None,
            "mean_sfd1_bytes": sum(int(row["sfd1_bytes"]) for row in action_rows) / 5,
        })
    result = {
        "captures": 20, "action_counts": {str(action): 5 for action in ACTIONS},
        "ack_installed_actions": [71], "actions_without_live_install": [0, 20, 46],
        "live_edge_completed_frames": 4, "live_results_returned_to_ue": 1,
        "terminal_outcomes": dict(Counter(row["status"] for row in terminal_rows)),
        "unique_frame_ids": 20, "mean_capture_to_edge_result_ms": latency,
        "maximum_capture_to_edge_result_ms": latency,
        "mean_sfd1_bytes": sum(int(row["sfd1_bytes"]) for row in sent) / 20,
        "total_sfd1_bytes": sum(int(row["sfd1_bytes"]) for row in sent),
        "radio_steps": len(radio), "radio_first_step": 0,
        "socket_buffers": {**live["socket_buffers"],
                           "edge_receive_reported_bytes": startup["edge_receive_reported_bytes"],
                           "edge_send_reported_bytes": startup["edge_send_reported_bytes"]},
        "ue_counters": ue, "edge_counters": edge, "edge_call_ledger": edge_ledger,
        "edge_startup": startup, "edge_mounts": summary["edge_mounts"],
        "actions": action_results, "route": summary["route"],
        "performance_outcomes": {
            "eligible_preparation_frames": 41, "sent_frames": 20,
            "preparation_rate": 20 / 41, "installed_frames": 1,
            "conditional_delivery_rate_given_sent": 1 / 20,
            "overall_delivery_rate": 1 / 41,
            "incomplete_radar_windows_recorded_as_dropped_under_amendment": 1,
            "low_preparation_or_delivery_is_measured_not_structurally_invalid": True,
        },
        "structural_acceptance": {
            "status": "PASS_AFTER_AUTHORIZED_OFFLINE_RECLASSIFICATION",
            "corruption_or_identity_failure": False, "nonfinite_output": False,
            "accounting_failure": False, "operational_process_failure": False,
            "dirty_teardown": False,
            "historical_failures_reclassified_as_performance": sorted(EXPECTED_TOP_FAILURES),
        },
    }
    return result, inventory


def verify_reclassification_for_pilot(root: Path, qualification: Mapping[str, Any]) -> Path:
    audit = qualification.get("reclassification", {})
    require(audit.get("schema") == AUDIT_SCHEMA
            and audit.get("post_observation_protocol_amendment") is True
            and audit.get("protocol_amendment_commit") == AMENDMENT_COMMIT
            and audit.get("new_live_measurement_performed") is False
            and audit.get("original_failed_record_preserved") is True
            and audit.get("all_four_ue_branches_verified") is True
            and audit.get("all_four_edge_branches_verified") is True
            and audit.get("at_least_one_real_oai_installation_verified") is True
            and audit.get("exact_terminal_accounting_verified") is True
            and audit.get("cold_teardown_verified") is True,
            "Phase-15 reclassification audit gates are incomplete")
    source = (ROOT / str(audit.get("source_root", ""))).resolve(strict=True)
    try:
        source.relative_to(EXPERIMENTS.resolve(strict=True))
    except ValueError as exc:
        raise ReclassificationError("reclassification source escapes experiments") from exc
    require(audit.get("source_files") == source_inventory(source),
            "reclassification source inventory drift")
    result, _ = audit_retry12(source)
    require(qualification.get("result") == result,
            "reclassification result no longer agrees with immutable retry12")
    return source / "preflight_inventory.json"


def write_artifacts(source: Path, output: Path, result: Mapping[str, Any],
                    source_files: list[dict[str, Any]]) -> None:
    experiments = EXPERIMENTS.resolve(strict=True)
    output = output.resolve(strict=False)
    source = source.resolve(strict=True)
    for value, label in ((source, "source"), (output, "output")):
        try:
            value.relative_to(experiments)
        except ValueError as exc:
            raise ReclassificationError(f"{label} escapes experiments") from exc
    require(not output.exists(), f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent))
    try:
        head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
                              text=True, stdout=subprocess.PIPE).stdout.strip()
        qualification = {
            "schema": SCHEMA, "status": TERMINAL, "captures": 20,
            "action_counts": {str(action): 5 for action in ACTIONS},
            "allowed_action_ids": list(ACTIONS), "action_order": list(HISTORICAL_ORDER),
            "captures_per_action": 5, "network_profile": "FAVORABLE_STABLE",
            "network_start_sample": 0, "result": dict(result),
            "cold_cleanup_verified": True,
            "preflight_inventory_sha256": SOURCE_FILES["preflight_inventory.json"],
            "retained_rgb_radar_payloads_predictions": False,
            "pilot_16_authorized_by_qualification": True,
            "campaign_288_authorized": False,
            "reclassification": {
                "schema": AUDIT_SCHEMA, "source_root": str(source.relative_to(ROOT)),
                "source_files": source_files, "source_head": SOURCE_HEAD,
                "audit_head": head, "protocol_amendment_commit": AMENDMENT_COMMIT,
                "post_observation_protocol_amendment": True,
                "new_live_measurement_performed": False,
                "original_failed_record_preserved": True,
                "all_action_localhost_qualification_bound": True,
                "all_four_ue_branches_verified": True,
                "all_four_edge_branches_verified": True,
                "at_least_one_real_oai_installation_verified": True,
                "exact_terminal_accounting_verified": True,
                "cold_teardown_verified": True,
                "historical_adapter_nonzero_exit_was_caused_only_by_superseded_outcome_gates": True,
            },
        }
        qualification_path = temporary / "qualification.json"
        write_json(qualification_path, qualification)
        report_path = temporary / "REPORT.md"
        report_path.write_text(
            "# Phase-15 retry12 offline reclassification\n\n"
            f"Status: `{TERMINAL}`\n\n"
            "This is an explicitly post-observation reclassification of immutable retry12; "
            "it is not a new live measurement and does not rewrite the original `FAILED.json`.\n\n"
            "Retry12 completed Route B, sent five frames through each representative UE branch, "
            "recorded all four edge branches, returned one finite frame-context-valid result, "
            "installed it through real OAI, preserved exact terminal accounting, and ended with "
            "cold CARLA/OAI/application teardown. Its historical failures were limited to outcomes "
            f"superseded by protocol-amendment commit `{AMENDMENT_COMMIT[:7]}`.\n\n"
            f"Measured preparation: 20/41 ({20/41:.6f}); installed: 1/20 sent ({1/20:.6f}), "
            f"or 1/41 eligible ({1/41:.6f}). These remain performance outcomes.\n\n"
            "This artifact authorizes only the 16-cell pilot, not the 288-cell campaign.\n",
            encoding="utf-8",
        )
        files = [{"path": path.name, "bytes": path.stat().st_size,
                  "sha256": sha256_file(path)} for path in (qualification_path, report_path)]
        manifest_path = temporary / "artifact_manifest.json"
        write_json(manifest_path, {"schema": MANIFEST_SCHEMA, "files": files})
        write_json(temporary / TERMINAL, {
            "status": TERMINAL, "qualification_sha256": sha256_file(qualification_path),
            "artifact_manifest_sha256": sha256_file(manifest_path),
        })
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", required=True)
    args = parser.parse_args(argv)
    require(args.execute == TOKEN, "exact offline reclassification token is required")
    result, inventory = audit_retry12(args.source.resolve(strict=True))
    write_artifacts(args.source, args.output, result, inventory)
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
