#!/usr/bin/env python3
"""Freeze the offline UE-N3 boundary-calibration plan without running OAI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/ue_n3_oai_ul_boundary_calibration_v1.json"
SUCCESS_STATUS = "UE_N3_PLAN_FROZEN_REVIEW_REQUIRED"


class PlanError(RuntimeError):
    """Raised when the N3 offline plan violates its frozen claim boundary."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise PlanError(f"path escapes repository root: {relative}") from exc
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema") == "scenesense.ue_n3_oai_ul_boundary_calibration_config.v1",
        "unexpected N3 config schema",
    )
    contract = config["contract"]
    contract_path = resolve(str(contract["path"]))
    _require(contract_path.is_file(), "N3 contract is missing")
    _require(sha256(contract_path) == contract["sha256"], "N3 contract hash drift")
    authority = config["authority"]
    _require(authority.get("offline_plan_freeze_authorized") is True, "offline plan authority is absent")
    for forbidden in (
        "oai_run_authorized",
        "carla_run_authorized",
        "socket_execution_authorized",
        "rf_command_extrapolation_authorized",
        "numeric_bound_promotion_authorized",
        "policy_training_authorized",
    ):
        _require(authority.get(forbidden) is False, f"forbidden authority enabled: {forbidden}")

    predecessor = config["predecessor"]
    evidence = resolve(str(predecessor["evidence_dir"]))
    manifest = evidence / str(predecessor["manifest_json"])
    terminal_path = evidence / str(predecessor["terminal_json"])
    _require(manifest.is_file() and terminal_path.is_file(), "sealed UE-N2 predecessor is missing")
    _require(sha256(manifest) == predecessor["manifest_sha256"], "UE-N2 manifest hash drift")
    _require(sha256(terminal_path) == predecessor["terminal_sha256"], "UE-N2 terminal hash drift")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    _require(terminal.get("status") == predecessor["required_status"], "UE-N2 status mismatch")
    _require(terminal.get("next") == predecessor["required_next"], "UE-N2 next-item mismatch")

    screening = config["screening"]
    targets = [float(value) for value in screening["desired_achieved_pusch_snr_db"]]
    _require(targets == [6.0, 4.0, 3.0, 2.0], "screening targets must be exactly 6,4,3,2 dB")
    mappings = screening["commanded_noise_power_db_by_target"]
    _require(set(mappings) == {"6.0", "4.0", "3.0", "2.0"}, "target-command mapping keys mismatch")
    _require(all(value is None for value in mappings.values()), "v1 must not guess RFsim command mappings")
    _require(math.isclose(float(screening["target_tolerance_db"]), 0.5), "target tolerance must be 0.5 dB")
    _require(float(screening["settle_duration_s"]) >= 10.0, "settle duration is below 10 seconds")
    _require(float(screening["service_duration_s"]) >= 60.0, "service duration is below 60 seconds")

    traffic = config["traffic_probe"]
    _require(
        traffic["status"] == "OFFLINE_RECEIVER_IMPLEMENTED_LIVE_INTEGRATION_REQUIRED",
        "traffic status does not preserve the live-integration gate",
    )
    _require(math.isclose(float(traffic["offered_rate_mbps"]), 1.0), "probe rate must be 1 Mbit/s")
    sender_path = resolve(str(traffic["sender_path"]))
    _require(sender_path.is_file(), "structured sender implementation is missing")
    _require(sha256(sender_path) == traffic["sender_sha256"], "structured sender hash drift")
    _require(math.isclose(float(traffic["sender_fps"]), 10.0), "sender rate must be 10 Hz")
    _require(int(traffic["sender_frames"]) == 600, "sender must emit 600 frames")
    _require(int(traffic["sender_frame_bytes"]) == 12_500, "sender frame size must be 12500 bytes")
    _require(int(traffic["sender_chunk_bytes"]) == 12_500, "sender chunk size must be 12500 bytes")
    _require(int(traffic["expected_chunks_per_frame"]) == 1, "probe must use one matched chunk per frame")
    receiver_path = resolve(str(traffic["receiver_path"]))
    _require(receiver_path.is_file(), "matched receiver implementation is missing")
    _require(sha256(receiver_path) == traffic["receiver_sha256"], "matched receiver hash drift")
    _require(traffic["receiver_offline_test_status"] == "PASSED", "matched receiver is not offline-tested")
    _require(
        traffic["one_second_gap_gate_metric"]
        == "INTERARRIVAL_GAPS_GTE_1S_EQUALS_ZERO_DURING_OBSERVED_STREAM",
        "one-second gap semantics are not frozen",
    )
    _require(traffic["packet_loss_gate"] is None, "unreviewed packet-loss gate must remain unset")
    _require(traffic["goodput_gate_mbps"] is None, "unreviewed goodput gate must remain unset")

    refinement = config["refinement"]
    _require(int(refinement["boundary_repetitions"]) == 3, "boundary repetitions must be three")
    _require(refinement["stable_outcome_requirement"] == "3_OF_3", "stable outcome must require 3/3")
    _require(config["cold_attach"]["authorized_now"] is False, "cold attach cannot precede N3A selection")
    _require(config["upper_boundary"]["authorized_now"] is False, "upper-bound live execution is not authorized")


def trial_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    screening = config["screening"]
    mappings = screening["commanded_noise_power_db_by_target"]
    for index, target in enumerate(screening["desired_achieved_pusch_snr_db"]):
        rows.append(
            {
                "phase": "N3A_SUSTAIN_SCREEN",
                "trial_order": index,
                "desired_achieved_pusch_snr_db": target,
                "commanded_noise_power_db": mappings[f"{float(target):.1f}"],
                "repetitions": 1,
                "clean_attach_first": True,
                "settle_duration_s": screening["settle_duration_s"],
                "service_duration_s": screening["service_duration_s"],
                "status": "BLOCKED_PENDING_TARGET_TO_COMMAND_CALIBRATION",
            }
        )
    rows.extend(
        [
            {
                "phase": "N3A_REFINEMENT",
                "trial_order": "TBD_AFTER_SCREEN",
                "desired_achieved_pusch_snr_db": "TBD_BRACKET_MIDPOINT",
                "commanded_noise_power_db": None,
                "repetitions": "MAX_2_TARGETS",
                "clean_attach_first": True,
                "settle_duration_s": screening["settle_duration_s"],
                "service_duration_s": screening["service_duration_s"],
                "status": "BLOCKED_PENDING_SCREEN_BRACKET",
            },
            {
                "phase": "N3A_BOUNDARY_REPLICATION",
                "trial_order": "TBD_AFTER_REFINEMENT",
                "desired_achieved_pusch_snr_db": "LOWEST_PASS_AND_ADJACENT_FAIL",
                "commanded_noise_power_db": None,
                "repetitions": config["refinement"]["boundary_repetitions"],
                "clean_attach_first": True,
                "settle_duration_s": screening["settle_duration_s"],
                "service_duration_s": screening["service_duration_s"],
                "status": "BLOCKED_PENDING_REFINED_BRACKET",
            },
            {
                "phase": "N3B_COLD_ATTACH_CONFIRMATION",
                "trial_order": "TBD_AFTER_N3A",
                "desired_achieved_pusch_snr_db": "LOWEST_REPLICATED_N3A_PASS",
                "commanded_noise_power_db": None,
                "repetitions": config["cold_attach"]["repetitions"],
                "clean_attach_first": False,
                "settle_duration_s": screening["settle_duration_s"],
                "service_duration_s": screening["service_duration_s"],
                "status": "BLOCKED_PENDING_N3A_SELECTION",
            },
        ]
    )
    for index, target in enumerate(config["upper_boundary"]["desired_achieved_pusch_snr_db"]):
        rows.append(
            {
                "phase": "N3_UPPER_BOUNDARY_VERIFICATION",
                "trial_order": index,
                "desired_achieved_pusch_snr_db": target,
                "commanded_noise_power_db": -50.0 if math.isclose(float(target), 50.5) else None,
                "repetitions": 1,
                "clean_attach_first": True,
                "settle_duration_s": screening["settle_duration_s"],
                "service_duration_s": screening["service_duration_s"],
                "status": "REFERENCE_ONLY" if math.isclose(float(target), 50.5) else "BLOCKED_PENDING_TARGET_TO_COMMAND_CALIBRATION",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(config_path: Path, output_dir: Path) -> Path:
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise PlanError(f"create-only output already exists: {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_dir.mkdir(parents=True)
    outputs = config["outputs"]
    atomic_json(output_dir / outputs["resolved_config"], config)
    rows = trial_rows(config)
    write_csv(output_dir / outputs["trial_matrix"], rows)
    report = "\n".join(
        [
            "# UE-N3 OAI uplink boundary-calibration plan", "",
            f"**Status:** `{SUCCESS_STATUS}`", "",
            "The desired achieved-PUSCH-SNR targets are frozen, but every non-clean RFsim command remains unset.",
            "No OAI, CARLA, socket, traffic, or policy process was executed.", "",
            "## Screening targets", "",
            "| Desired achieved PUSCH SNR | RFsim noise command | State |",
            "|---:|---:|---|",
            *[
                f"| {target:.1f} dB | unset | blocked pending calibration |"
                for target in config["screening"]["desired_achieved_pusch_snr_db"]
            ],
            "", "## Required next work", "",
            "1. Exercise the offline-tested SSBURST-aware receiver once in the live OAI namespace.",
            "2. Review the packet-loss and goodput gates; the no-one-second-interarrival-gap rule is frozen.",
            "3. Calibrate RFsim commands without relabelling commands as achieved SNR.",
            "4. Authorize and execute N3A one rung at a time with clean restoration.",
            "5. Refine and replicate the bracket before N3B cold-attach confirmation.", "",
        ]
    )
    atomic_text(output_dir / outputs["report"], report)
    manifest_files = []
    for name in (outputs["resolved_config"], outputs["trial_matrix"], outputs["report"]):
        path = output_dir / name
        manifest_files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema": "scenesense.ue_n3_oai_ul_boundary_calibration_manifest.v1",
        "status": SUCCESS_STATUS,
        "created_at": utc_now(),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "runtime_executed": False,
        "socket_executed": False,
        "dependency_files": [
            {
                "path": config["contract"]["path"],
                "sha256": config["contract"]["sha256"],
            },
            {
                "path": config["traffic_probe"]["sender_path"],
                "sha256": config["traffic_probe"]["sender_sha256"],
            },
            {
                "path": config["traffic_probe"]["receiver_path"],
                "sha256": config["traffic_probe"]["receiver_sha256"],
            },
        ],
        "outputs": manifest_files,
    }
    atomic_json(output_dir / outputs["manifest"], manifest)
    terminal = {
        "status": SUCCESS_STATUS,
        "runtime_executed": False,
        "socket_executed": False,
        "numeric_bound_promoted": False,
        "target_to_command_mapping_status": "UNRESOLVED",
        "traffic_probe_status": config["traffic_probe"]["status"],
        "manifest_sha256": sha256(output_dir / outputs["manifest"]),
        "next": "LIVE_INTEGRATE_MATCHED_PROBE_AND_REVIEW_TARGET_TO_COMMAND_CALIBRATION",
    }
    atomic_json(output_dir / outputs["terminal"], terminal)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = prepare(Path(args.config), Path(args.output_dir))
    print(json.dumps({"output_dir": str(output), "status": SUCCESS_STATUS}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
