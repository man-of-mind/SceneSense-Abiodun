#!/usr/bin/env python3
"""Corrected Phase-14B four-profile replay: two audit-driven corrections.

A separately versioned successor to `splitfusion_phase14b_four_profile_replay_v1.py`,
which is left untouched together with its contract, config, and evidence. This
runner does not alter the Phase-14A mapping, the four frozen profiles/seeds/
transitions, the 4,200-sample/100-ms cadence, `SKIP_OBSOLETE_NEVER_BURST`, the
ACK+15ms guard, the minimum-3-PUSCH/1-MCS gate, or any existing mapping-error/
bias/P95/timing/topology/scheduler/RNTI threshold.

It applies exactly two corrections identified by the offline root-cause audit
at `experiments/splitfusion_phase14b_observation_coverage_audit_v1/20260905_root_cause_audit/`:

1. Traffic: the 25,000-byte/10-fps burst source is replaced, for this radio-
   mapping qualification only, by one fixed, lightweight, evenly paced UDP
   probe -- 100 datagrams/s, 1,200 application bytes each (~0.96 Mbps),
   distributing roughly ten independent uplink transmission opportunities
   across every 100-ms interval. This is measurement instrumentation, not
   SplitFusion application traffic: Phase-14B does not measure SplitFusion
   throughput or packet delivery, and the probe is never counted as
   SplitFusion traffic or campaign throughput. Before any four-profile
   replay, a bounded 10-second high-SNR probe qualification (`run_probe_qualification`)
   must pass as a mechanical instrumentation gate; it is not a traffic-rate
   search and is never retried with an increased probe on failure. (An
   earlier draft of this correction proposed reusing the capacity-study
   300-Mbps UDP iperf3 flow; that was superseded before any live execution by
   an explicit scientific amendment -- RFsim controls the channel
   continuously, so saturating the scheduler is unnecessary.)
2. Window: the causal observation window's end moves from the nominal 100-ms
   tick to the actual next command's send time (or the clean-restoration
   command's send time for the final command), so PUSCH samples are no longer
   discarded merely because the previous RFsim command was still active past
   its nominal deadline. The nominal formula is retained as a diagnostic-only
   `nominal_schedule_coverage`, computed in parallel and never used for gating.

A live execution requires an explicit token; without it, importing this module
and running its focused CPU helpers cannot touch the radio.  Every profile owns
one launcher/attachment/traffic lifecycle and is durably recorded only after
clean restoration and complete teardown.  The 4,200-sample traces are validated
prefixes, while the production generator remains continuous beyond sample
4,199.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import oai_target_snr_replay_pilot_v1 as replay  # noqa: E402
from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a  # noqa: E402
from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as telemetry  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/splitfusion_phase14b_corrected_four_profile_replay_v1.json"
PROFILE_ORDER = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
EXECUTION_TOKEN = "SPLITFUSION_PHASE14B_CORRECTED_FOUR_PROFILE_REPLAY"
SUCCESS_TERMINAL = "SPLITFUSION_PHASE14B_CORRECTED_FOUR_PROFILE_REPLAY_COMPLETE"
RADIO_ATTACH_TOKEN = "SPLITFUSION_OAI_100MHZ_4D5U_ATTACH"
SCHEMA = "scenesense.splitfusion_phase14b_corrected_four_profile_replay.v1"
PROBE_QUALIFICATION_RADIO_STATE_LEAF = "PQ_PROBE_QUALIFICATION"


class Phase14BError(RuntimeError):
    """A frozen input, live-radio, replay, acceptance, or cleanup gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase14BError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str, *, strict: bool = True) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else ROOT / raw
    resolved = candidate.resolve(strict=strict)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise Phase14BError(f"path escapes repository root: {value}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verified_file(record: Mapping[str, Any], label: str) -> Path:
    path = repo_path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    require(len(expected) == 64, f"{label} has no complete SHA-256")
    require(sha256_file(path) == expected, f"{label} SHA-256 drift: {path}")
    return path


def git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"cannot read bound git blob {commit}:{path}: "
        + completed.stderr.decode("utf-8", errors="replace").strip(),
    )
    return completed.stdout


def require_ancestor(commit: str, label: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    require(completed.returncode == 0, f"{label} is not an ancestor of HEAD: {commit}")


def atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def percentile(values: Sequence[float], quantile: float) -> float | None:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return None
    position = (len(clean) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - position) + clean[upper] * (position - lower)


def distribution(values: Sequence[float], quantiles: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean) if clean else None,
        "population_stddev": statistics.pstdev(clean) if len(clean) >= 2 else 0.0 if clean else None,
        "quantiles": {
            f"q{int(round(float(q) * 100)):02d}": percentile(clean, float(q))
            for q in quantiles
        },
    }


def tunnel_counters(interface: str) -> dict[str, int | None]:
    """Read host-visible sysfs byte/packet counters for a tunnel interface."""

    result: dict[str, int | None] = {}
    for name in ("tx_bytes", "rx_bytes", "tx_packets", "rx_packets"):
        path = Path("/sys/class/net") / interface / "statistics" / name
        try:
            result[name] = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            result[name] = None
    return result


def plan_scheduler_action(
    *,
    scheduled_ns: int,
    period_ns: int,
    now_ns: int,
    previous_send_ns: int | None,
) -> dict[str, Any]:
    """Return a fail-closed absolute-deadline/no-burst scheduling decision."""

    require(period_ns > 0, "scheduler period must be positive")
    interval_end_ns = int(scheduled_ns) + int(period_ns)
    eligible_ns = max(int(scheduled_ns), int(now_ns))
    if previous_send_ns is not None:
        eligible_ns = max(eligible_ns, int(previous_send_ns) + int(period_ns))
    if eligible_ns >= interval_end_ns:
        return {
            "status": "SKIP_OBSOLETE_NEVER_BURST",
            "eligible_send_ns": None,
            "interval_end_ns": interval_end_ns,
        }
    return {
        "status": "SEND_ON_ABSOLUTE_SCHEDULE",
        "eligible_send_ns": eligible_ns,
        "interval_end_ns": interval_end_ns,
    }


def _read_prefix_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {profile: [] for profile in PROFILE_ORDER}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile = str(row["profile_id"])
            if profile in rows:
                rows[profile].append(dict(row))
    for profile in PROFILE_ORDER:
        rows[profile].sort(key=lambda row: int(row["step_index"]))
    return rows


def prepare_frozen_profiles(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Instantiate each generator once and validate its prefix plus continuation."""

    provenance = config["provenance"]
    design_path = verified_file(provenance["network_profile_design"], "profile design")
    prefix_path = verified_file(provenance["network_profile_prefix"], "accepted profile prefix")
    verified_file(provenance["network_profile_summary"], "profile summary")
    verified_file(provenance["continuous_profile_runtime"], "continuous profile runtime")
    design = load_json(design_path)
    generator = phase14a.load_generator_module()
    generator.validate_config(design)
    require(
        float(design["target_snr"]["lower_bound_db"])
        == float(config["replay"]["target_lower_db"])
        and float(design["target_snr"]["upper_bound_db"])
        == float(config["replay"]["target_upper_db"]),
        "frozen profile target bounds drift",
    )
    require(
        int(round(float(design["route"]["sample_period_s"]) * 1000))
        == int(config["replay"]["sample_period_ms"]),
        "frozen profile sampling period drift",
    )
    design_profiles = {str(row["profile_id"]): row for row in design["profiles"]}
    accepted = _read_prefix_rows(prefix_path)
    count = int(config["replay"]["samples_per_profile"])
    require(count == 4200, "Phase-14B requires exactly 4,200 samples per profile")
    frozen_rows = list(config["profiles"])
    require(tuple(str(row["profile_id"]) for row in frozen_rows) == PROFILE_ORDER, "profile order drift")
    prepared: list[dict[str, Any]] = []
    for frozen in frozen_rows:
        profile_id = str(frozen["profile_id"])
        source = design_profiles.get(profile_id)
        require(source is not None, f"profile missing from design: {profile_id}")
        require(str(source["trace_id"]) == str(frozen["trace_id"]), f"{profile_id} trace ID drift")
        require(int(source["seed"]) == int(frozen["seed"]), f"{profile_id} seed drift")
        require(source["transition_matrix"] == frozen["transition_matrix"], f"{profile_id} transition drift")
        sequence = generator.DeterministicTargetSnrSequence(source, design)
        prefix = [sequence.next_sample() for _ in range(count)]
        continuation = sequence.next_sample()
        states = np.asarray([row[0] for row in prefix], dtype="<i4")
        targets = np.asarray([row[1] for row in prefix], dtype="<f8")
        digest = sha256_bytes(states.tobytes() + targets.tobytes())
        require(digest == str(frozen["trace_sha256"]), f"{profile_id} reference hash mismatch")
        source_rows = accepted[profile_id]
        require(len(source_rows) == count, f"{profile_id} accepted prefix length drift")
        require([int(row["step_index"]) for row in source_rows] == list(range(count)), f"{profile_id} prefix index drift")
        require(
            {str(row["trace_id"]) for row in source_rows} == {str(frozen["trace_id"])},
            f"{profile_id} accepted trace identity drift",
        )
        require(
            np.array_equal(states, np.asarray([int(row["state_index"]) for row in source_rows], dtype="<i4")),
            f"{profile_id} accepted state prefix drift",
        )
        require(
            np.allclose(
                targets,
                np.asarray([float(row["target_snr_db"]) for row in source_rows], dtype="<f8"),
                rtol=0.0,
                atol=5.1e-7,
            ),
            f"{profile_id} accepted target prefix drift",
        )
        expected_continuation = (
            int(frozen["sample_4200_state_index"]),
            float(frozen["sample_4200_target_snr_db"]),
        )
        require(continuation == expected_continuation, f"{profile_id} continuation reseeded or drifted")
        require(continuation != prefix[0], f"{profile_id} continuation wrapped to sample zero")
        require(continuation != prefix[-1], f"{profile_id} continuation held sample 4,199")
        prepared.append(
            {
                "profile_id": profile_id,
                "trace_id": str(frozen["trace_id"]),
                "seed": int(frozen["seed"]),
                "trace_sha256": digest,
                "prefix": prefix,
                "continuation": continuation,
            }
        )
    require(len(prepared) == 4, "exactly four frozen profiles are required")
    return prepared


def _normalized_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(value))
    normalized["launcher"]["sha256"] = "AMENDED_RUNTIME_HASH"
    normalized["calibration"]["runner"]["sha256"] = "AMENDED_RUNTIME_HASH"
    return normalized


def verify_provenance(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path.resolve(strict=True))
    require(config.get("schema") == "scenesense.splitfusion_phase14b_corrected_four_profile_replay_config.v1", "config schema drift")
    require(config.get("status") == "IMPLEMENTED_NOT_EXECUTED", "Phase-14B status overclaims execution")
    require(config.get("execution_token") == EXECUTION_TOKEN, "execution token drift")
    provenance = config["provenance"]
    paths: dict[str, Path] = {}
    for name in (
        "phase14a_calibration_terminal",
        "phase14a_manifest",
        "phase14a_mapping",
        "phase14a_calibration_summary",
        "phase14a_anchor_table",
        "phase14a_config",
        "qualified_topology_validator",
        "radio_profile",
        "network_profile_design",
        "network_profile_prefix",
        "network_profile_summary",
        "continuous_profile_runtime",
        "phase14b_original_config",
        "phase14b_original_runner",
        "phase14b_original_qualification",
        "phase14b_original_report",
        "phase14b_original_run_manifest",
        "phase14b_original_implementation_report",
        "observation_coverage_audit_report",
        "observation_coverage_audit_json",
        "capacity_evidence",
        "iperf_sweep_runner",
        "iperf_sweep_config",
    ):
        paths[name] = verified_file(provenance[name], name.replace("_", " "))

    phase14b_original_commit = str(provenance["phase14b_original_evidence_commit"])
    require_ancestor(phase14b_original_commit, "Phase-14B original evidence commit")
    for name in (
        "phase14b_original_config",
        "phase14b_original_runner",
        "phase14b_original_qualification",
        "phase14b_original_report",
        "phase14b_original_run_manifest",
        "phase14b_original_implementation_report",
    ):
        record = provenance[name]
        require(
            sha256_bytes(git_blob(phase14b_original_commit, str(record["path"])))
            == str(record["sha256"]),
            f"{name.replace('_', ' ')} is not the blob committed by the Phase-14B original evidence commit",
        )
    original_qualification = load_json(paths["phase14b_original_qualification"])
    require(
        original_qualification.get("status") == "FOUR_PROFILE_REPLAY_COMPLETE",
        "original Phase-14B qualification status drift",
    )
    require(
        original_qualification.get("all_profiles_passed") is False,
        "original Phase-14B replay must remain a failed/negative result; this run does not reinterpret it",
    )
    require(
        original_qualification.get("mapping_sha256") == provenance["phase14a_mapping"]["sha256"],
        "original Phase-14B qualification maps to a different Phase-14A mapping",
    )

    evidence_commit = str(provenance["phase14a_evidence_commit"])
    require_ancestor(evidence_commit, "Phase-14A evidence commit")
    for name in (
        "phase14a_calibration_terminal",
        "phase14a_manifest",
        "phase14a_mapping",
        "phase14a_calibration_summary",
        "phase14a_anchor_table",
    ):
        record = provenance[name]
        require(
            sha256_bytes(git_blob(evidence_commit, str(record["path"])))
            == str(record["sha256"]),
            f"{name.replace('_', ' ')} is not the blob committed by the Phase-14A evidence commit",
        )
    binding_seal = provenance["phase14a_campaign_binding"]
    historical_commit = str(binding_seal["historical_commit"])
    topology_commit = str(binding_seal["topology_amendment_commit"])
    require_ancestor(historical_commit, "Phase-14A implementation binding commit")
    require_ancestor(topology_commit, "qualified topology amendment")
    require_ancestor(str(provenance["qualified_topology_validator"]["commit"]), "topology-validator commit")
    topology_record = provenance["qualified_topology_validator"]
    require(
        sha256_bytes(git_blob(topology_commit, str(topology_record["path"])))
        == topology_record["sha256"],
        "qualified topology-validator blob drift",
    )
    historical = git_blob(historical_commit, str(binding_seal["path"]))
    require(sha256_bytes(historical) == binding_seal["historical_sha256"], "historical Phase-14A campaign-binding hash drift")
    current_path = repo_path(str(binding_seal["path"]))
    require(sha256_file(current_path) == binding_seal["current_sha256"], "current topology-amended campaign-binding hash drift")
    historical_binding = json.loads(historical.decode("utf-8"))
    current_binding = load_json(current_path)
    require(
        _normalized_binding(historical_binding) == _normalized_binding(current_binding),
        "Phase-14A campaign binding changed outside launcher/validator provenance amendments",
    )

    terminal = load_json(paths["phase14a_calibration_terminal"])
    manifest = load_json(paths["phase14a_manifest"])
    mapping = load_json(paths["phase14a_mapping"])
    summary = load_json(paths["phase14a_calibration_summary"])
    require(terminal["manifest_sha256"] == provenance["phase14a_manifest"]["sha256"], "terminal/manifest seal mismatch")
    require(terminal["status"] == "CALIBRATION_CAPTURE_COMPLETE_PENDING_FOUR_PROFILE_REPLAY", "Phase-14A terminal status drift")
    require(manifest["mapping_qualified_for_campaign"] is False, "Phase-14A manifest already claims qualification")
    output_hashes = {str(row["path"]): str(row["sha256"]) for row in manifest["outputs"]}
    require(output_hashes.get("mapping.json") == provenance["phase14a_mapping"]["sha256"], "manifest/mapping seal mismatch")
    require(output_hashes.get("calibration_summary.json") == provenance["phase14a_calibration_summary"]["sha256"], "manifest/summary seal mismatch")
    require(output_hashes.get("anchor_summary.csv") == provenance["phase14a_anchor_table"]["sha256"], "manifest/anchor seal mismatch")
    require(summary["restore_verified"] is True and summary["profile_replay_performed"] is False, "Phase-14A summary state drift")
    require(mapping["status"] == "MEASURED_100MHZ_MAPPING_PENDING_FOUR_PROFILE_REPLAY", "mapping status drift")
    require(mapping["mapping_qualified_for_campaign"] is False, "mapping already claims campaign qualification")
    require(mapping["legacy_mapping_used"] is False, "legacy 40-MHz mapping was selected")
    require(mapping["radio_profile_id"] == RADIO_PROFILE_ID, "mapping radio identity drift")
    gates = mapping["gates"]
    require(gates["strictly_monotonic"] is True and gates["complete_target_range_covered"] is True, "Phase-14A mapping gates are not complete")
    require(float(gates["measured_lower_db"]) == 5.0 and float(gates["measured_upper_db"]) == 25.5, "measured mapping coverage drift")
    anchors = replay.validate_mapping(mapping["anchors"])
    with paths["phase14a_anchor_table"].open(newline="", encoding="utf-8") as handle:
        anchor_rows = list(csv.DictReader(handle))
    require(len(anchor_rows) == 12, "Phase-14A anchor table does not contain 12 anchors")
    table_anchors = replay.validate_mapping(
        [
            {
                "achieved_median_pusch_snr_db": float(row["achieved_median_pusch_snr_db"]),
                "noise_power_db": float(row["applied_noise_power_db"]),
                "source": f"PHASE14A_100MHZ_ANCHOR_{int(row['anchor_index']):02d}",
            }
            for row in anchor_rows
        ]
    )
    require(table_anchors == anchors, "Phase-14A mapping does not equal the sealed anchor table")
    require(current_binding["mapping"]["legacy_mapping_permitted"] is False, "current binding permits the legacy 40-MHz mapping")

    radio = load_json(paths["radio_profile"])
    expected_radio = config["radio"]
    actual_radio = radio["radio"]
    require(radio["profile_id"] == RADIO_PROFILE_ID, "locked radio profile ID drift")
    require(
        (
            int(actual_radio["band"]),
            int(actual_radio["bandwidth_mhz"]),
            int(actual_radio["prb"]),
            int(actual_radio["numerology"]),
            int(actual_radio["tdd"]["downlink_slots"]),
            int(actual_radio["tdd"]["uplink_slots"]),
            int(actual_radio["ue_count"]),
            str(actual_radio["ue_static_ip"]),
            int(actual_radio["pdu_session_5qi"]),
        )
        == (
            int(expected_radio["band"]),
            int(expected_radio["bandwidth_mhz"]),
            int(expected_radio["prb"]),
            int(expected_radio["numerology"]),
            int(expected_radio["downlink_slots"]),
            int(expected_radio["uplink_slots"]),
            int(expected_radio["ue_count"]),
            str(expected_radio["ue_static_ip"]),
            int(expected_radio["pdu_session_5qi"]),
        ),
        "locked 100-MHz/4D5U radio contract drift",
    )
    replay_config = config["replay"]
    require(int(replay_config["sample_period_ms"]) == 100, "replay period drift")
    require(float(replay_config["duration_s_per_profile"]) == 420.0, "profile duration drift")
    require(
        int(replay_config["samples_per_profile"])
        * int(replay_config["sample_period_ms"])
        == int(float(replay_config["duration_s_per_profile"]) * 1000),
        "replay sample count/period/duration mismatch",
    )
    require(float(replay_config["command_granularity_db"]) == 0.25, "mapping quantization drift")
    require(replay_config["catch_up_policy"] == "SKIP_OBSOLETE_NEVER_BURST", "catch-up policy drift")
    require(replay_config["reference_prefix_is_runtime_cap"] is False, "reference prefix became a runtime cap")
    require(replay_config["forbidden_continuation"] == ["WRAP", "HOLD_FINAL", "RESEED"], "continuation refusal set drift")
    lifecycle = config["radio_lifecycle"]
    require(lifecycle["scope"] == "ONE_FRESH_OAI_RFSIM_LIFECYCLE_PER_PROFILE", "radio lifecycle scope drift")
    require(lifecycle["launcher_execution_token"] == RADIO_ATTACH_TOKEN, "radio launcher token drift")
    require(lifecycle["fresh_attachment_per_profile"] is True, "radio attachment reuse is forbidden")
    require(lifecycle["fresh_traffic_and_parser_state_per_profile"] is True, "traffic/parser reuse is forbidden")
    require(lifecycle["durable_record_after_full_teardown_only"] is True, "profile durability/teardown order drift")
    require(lifecycle["restore_noise_power_db"] == -50.0, "lifecycle restore target drift")
    launcher = current_binding["launcher"]
    launcher_path = repo_path(str(launcher["path"]))
    require(sha256_file(launcher_path) == str(launcher["sha256"]), "qualified launcher hash drift")
    require(launcher["execution_token"] == RADIO_ATTACH_TOKEN, "qualified launcher execution token drift")
    return {
        "config": config,
        "mapping": anchors,
        "mapping_sha256": provenance["phase14a_mapping"]["sha256"],
        "radio_profile_sha256": provenance["radio_profile"]["sha256"],
        "historical_campaign_binding_sha256": binding_seal["historical_sha256"],
        "current_campaign_binding_sha256": binding_seal["current_sha256"],
        "topology_validator_commit": topology_commit,
        "launcher_path": launcher_path,
        "launcher_sha256": str(launcher["sha256"]),
        "launcher_execution_token": str(launcher["execution_token"]),
        "capacity_evidence": dict(provenance["capacity_evidence"]),
        "phase14b_original_qualification_sha256": provenance["phase14b_original_qualification"]["sha256"],
        "observation_coverage_audit_report_sha256": provenance["observation_coverage_audit_report"]["sha256"],
        "observation_coverage_audit_json_sha256": provenance["observation_coverage_audit_json"]["sha256"],
    }


def create_or_resume_output(config: Mapping[str, Any], value: str, resume_run: bool) -> tuple[Path, bool]:
    experiments_root = (ROOT / "experiments").resolve(strict=True)
    output_root = repo_path(str(config["output_root"]), strict=False)
    try:
        output_root.relative_to(experiments_root)
    except ValueError as exc:
        raise Phase14BError(f"configured output root escapes experiments: {output_root}") from exc
    raw = Path(value)
    candidate = (raw if raw.is_absolute() else ROOT / raw).resolve(strict=False)
    try:
        candidate.relative_to(output_root)
    except ValueError as exc:
        raise Phase14BError(f"output path escapes {output_root}: {candidate}") from exc
    require(candidate != output_root, "output must be a create-only run-directory leaf beneath the output root")
    if resume_run:
        output = candidate.resolve(strict=True)
        require(output.is_dir(), f"resume output is not a directory: {output}")
        require(not (output / SUCCESS_TERMINAL).exists(), f"run is already finalized: {output}")
        return output, False
    require(not candidate.exists(), f"create-only output already exists: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=False, exist_ok=False)
    output = candidate.resolve(strict=True)
    try:
        output.relative_to(output_root)
    except ValueError as exc:
        raise Phase14BError(f"created output escaped {output_root}: {output}") from exc
    return output, True


def resolve_radio_namespace(config: Mapping[str, Any], value: Path) -> Path:
    phase14a_config = load_json(repo_path(str(config["provenance"]["phase14a_config"]["path"])))
    root = repo_path(str(phase14a_config["paths"]["radio_state_root"]), strict=False)
    raw = value if value.is_absolute() else ROOT / value
    candidate = raw.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Phase14BError(f"radio-state namespace escapes {root}: {candidate}") from exc
    require(candidate != root, "radio-state namespace must be a dedicated leaf beneath the registered root")
    return candidate


def profile_lifecycle_plan(
    config: Mapping[str, Any], radio_namespace: Path
) -> list[dict[str, Any]]:
    """Return the immutable four-session lifecycle plan without touching runtime."""

    plan = []
    for index, profile in enumerate(config["profiles"]):
        profile_id = str(profile["profile_id"])
        plan.append(
            {
                "profile_index": index,
                "profile_id": profile_id,
                "radio_state_path": str(radio_namespace / f"{index:02d}_{profile_id}"),
                "requires_cold_start": True,
                "fresh_qualified_gnb_topology": True,
                "fresh_qualified_ue_topology": True,
                "fresh_traffic_process": True,
                "counters_sequence_timing_and_parser_state_reset": True,
                "generator_start_index": 0,
                "settings": int(config["replay"]["samples_per_profile"]),
                "durable_record_after_full_teardown_only": True,
            }
        )
    require([row["profile_id"] for row in plan] == list(PROFILE_ORDER), "lifecycle profile order drift")
    require(len({row["radio_state_path"] for row in plan}) == 4, "profile radio-state paths are not independent")
    return plan


def _run_checked(argv: Sequence[str], *, cwd: Path = ROOT, timeout_s: float = 120.0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    require(
        completed.returncode == 0,
        f"command failed ({' '.join(argv)}): {completed.stderr.strip() or completed.stdout.strip()}",
    )
    return completed


def _compose_container_ids(config: Mapping[str, Any]) -> list[str]:
    cn = repo_path(str(config["paths"]["oai_cn"]))
    completed = _run_checked(
        ["sudo", "-n", "docker", "compose", "ps", "-q"],
        cwd=cn,
        timeout_s=30.0,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def process_is_live(pid: int) -> bool:
    try:
        suffix = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
    except (IndexError, OSError, UnicodeDecodeError):
        return False
    fields = suffix.split()
    return bool(fields) and fields[0] != "Z"


PROBE_SENDER_PATH = ROOT / "rl_agent/phase14b_probe_sender_v1.py"


def require_no_stale_probe_sender() -> None:
    """Fail closed if a prior corrected-traffic probe sender survives.

    The reused UDP sink is already covered by the legacy sender/sink path
    check in `require_cold_profile_runtime`; only the new probe sender's own
    path needs a supplemental check here.
    """

    sender_path = str(PROBE_SENDER_PATH.resolve(strict=True))
    stale = [
        row
        for row in phase14a.process_table()
        if process_is_live(int(row["pid"])) and sender_path in str(row["command"])
    ]
    require(not stale, f"stale Phase-14B probe-sender process exists: {[row['pid'] for row in stale]}")


def require_cold_profile_runtime(config: Mapping[str, Any], radio_state: Path) -> dict[str, Any]:
    """Fail closed unless no state from any earlier profile survives."""

    require(not radio_state.exists(), f"stale profile radio-state scratch exists: {radio_state}")
    rows = phase14a.process_table()
    build = repo_path(str(config["paths"]["oai_ran_build"]))
    expected = {
        str((build / "nr-softmodem").resolve(strict=True)),
        str((build / "nr-uesoftmodem").resolve(strict=True)),
    }
    softmodems = [
        row
        for row in rows
        if process_is_live(int(row["pid"]))
        and (
            str(row["executable"]) in expected
            or str(row["command_name"]) in {"nr-softmodem", "nr-uesoftmodem"}
        )
    ]
    require(not softmodems, f"profile requires a cold RAN; observed PIDs {[row['pid'] for row in softmodems]}")
    traffic_paths = {
        str(repo_path(str(config["paths"]["sender"]))),
        str(repo_path(str(config["paths"]["sink"]))),
    }
    traffic_rows = [
        row
        for row in rows
        if process_is_live(int(row["pid"]))
        and any(path in str(row["command"]) for path in traffic_paths)
    ]
    require(not traffic_rows, f"stale Phase-14B traffic process exists: {[row['pid'] for row in traffic_rows]}")
    require_no_stale_probe_sender()
    runtime_groups = identifiable_phase14b_runtime_groups(config)
    require(not runtime_groups, f"stale Phase-14B traffic/telemetry process groups exist: {runtime_groups}")
    interface = str(config["radio"]["ue_interface"])
    require(phase14a.tunnel_ip(interface) is None, f"stale UE tunnel exists: {interface}")
    ports = (
        phase14a.RFSIM_TCP_PORT,
        int(config["telemetry"]["gnb_port"]),
        phase14a.UE_T_TRACER_PORT,
        int(config["actuator"]["telnet_port"]),
        int(config["telemetry"]["gnb_relay_port"]),
    )
    occupied = [port for port in ports if phase14a.tcp_listening(port)]
    require(not occupied, f"stale OAI/RFsim/telemetry listener ports exist: {occupied}")
    containers = _compose_container_ids(config)
    require(not containers, f"OAI core-network containers survive from another lifecycle: {containers}")
    return {
        "no_softmodem_process": True,
        "no_traffic_process": True,
        "no_ue_tunnel": True,
        "no_oai_rfsim_listener": True,
        "no_oai_core_container": True,
        "no_stale_profile_scratch": True,
    }


def stable_attached_radio_seal(attached: Mapping[str, Any]) -> dict[str, Any]:
    def topology(name: str) -> dict[str, Any]:
        value = attached[f"{name}_process_topology"]
        return {
            "schema": value["schema"],
            "source_contract": copy.deepcopy(value["source_contract"]),
            "same_executable_process_count": int(value["same_executable_process_count"]),
            "endpoint_roles_verified": bool(value["endpoint_roles_verified"]),
            "worker_count": len(value["worker_pids"]),
        }

    return {
        "radio_profile_id": attached["radio_profile_id"],
        "effective_gnb_sha256": attached["effective_gnb_sha256"],
        "effective_ue_sha256": attached["effective_ue_sha256"],
        "gnb_command_sha256": attached["gnb_command_sha256"],
        "ue_command_sha256": attached["ue_command_sha256"],
        "launcher_sha256": attached["launcher_sha256"],
        "ue_interface": attached["ue_interface"],
        "ue_static_ip": attached["ue_static_ip"],
        "pdu_session_5qi": int(attached["pdu_session_5qi"]),
        "gnb_topology": topology("gnb"),
        "ue_topology": topology("ue"),
    }


def _process_group_members(process_group_id: int) -> list[dict[str, Any]]:
    return [
        row
        for row in phase14a.process_table()
        if int(row["process_group_id"]) == int(process_group_id)
        and process_is_live(int(row["pid"]))
    ]


def _process_group_identity(process_group_id: int) -> list[dict[str, Any]]:
    return [
        {
            "pid": int(row["pid"]),
            "executable": str(row["executable"]),
            "command_sha256": sha256_bytes(str(row["command"]).encode("utf-8")),
        }
        for row in _process_group_members(process_group_id)
    ]


def _stop_process_group(process_group_id: int, label: str) -> list[dict[str, Any]]:
    """Stop exactly one launcher-created process group and verify its exit."""

    history: list[dict[str, Any]] = []
    for signal_name, wait_s in (("-INT", 10.0), ("-TERM", 5.0), ("-KILL", 5.0)):
        members = _process_group_members(process_group_id)
        history.append(
            {
                "signal": signal_name,
                "member_pids_before": [int(row["pid"]) for row in members],
            }
        )
        if not members:
            return history
        completed = subprocess.run(
            ["sudo", "-n", "kill", signal_name, "--", f"-{int(process_group_id)}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        require(
            completed.returncode == 0,
            f"could not send {signal_name} to owned {label} process group {process_group_id}: "
            f"{completed.stderr.strip()}",
        )
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not _process_group_members(process_group_id):
                return history
            time.sleep(0.1)
    survivors = _process_group_members(process_group_id)
    require(not survivors, f"owned {label} process group survived teardown: {[row['pid'] for row in survivors]}")
    return history


def stop_recorded_runtime_support(radio_state: Path) -> list[str]:
    """Stop interrupted traffic/telemetry only when its stored identity matches."""

    ownership_path = radio_state / "PHASE14B_RUNTIME_OWNERSHIP.json"
    if not ownership_path.is_file():
        return []
    ownership = load_json(ownership_path)
    require(ownership.get("schema") == "scenesense.splitfusion_phase14b_runtime_ownership.v1", "runtime ownership schema drift")
    errors: list[str] = []
    for group in reversed(ownership.get("process_groups", [])):
        process_group_id = int(group["process_group_id"])
        current = _process_group_identity(process_group_id)
        if not current:
            continue
        expected = {
            (int(row["pid"]), str(row["executable"]), str(row["command_sha256"]))
            for row in group["members"]
        }
        observed = {
            (int(row["pid"]), str(row["executable"]), str(row["command_sha256"]))
            for row in current
        }
        require(observed <= expected, f"runtime process-group identity changed; refusing PID reuse: {process_group_id}")
        try:
            _stop_process_group(process_group_id, str(group["name"]))
        except BaseException as exc:
            errors.append(f"{group['name']}: {type(exc).__name__}: {exc}")
    return errors


def identifiable_phase14b_runtime_groups(config: Mapping[str, Any]) -> list[int]:
    tracer = repo_path("OAI/openairinterface5g/common/utils/T/tracer")
    relay_port = int(config["telemetry"]["gnb_relay_port"])
    sender = str(repo_path(str(config["paths"]["sender"])))
    sink = str(repo_path(str(config["paths"]["sink"])))
    probe_sender = str(PROBE_SENDER_PATH.resolve(strict=True))
    multi = str(tracer / "multi")
    csv_tool = str(tracer / "csv")

    def owned(command: str) -> bool:
        fields = command.split()
        return (
            sender in fields
            or sink in fields
            or probe_sender in fields
            or (multi in fields and phase14a.command_has_pair(command, "-lp", str(relay_port)))
            or (csv_tool in fields and phase14a.command_has_pair(command, "-p", str(relay_port)))
        )

    rows = [
        row
        for row in phase14a.process_table()
        if process_is_live(int(row["pid"])) and owned(str(row["command"]))
    ]
    return sorted({int(row["process_group_id"]) for row in rows})


def stop_identifiable_phase14b_runtime_support(config: Mapping[str, Any]) -> list[str]:
    """Stop only replay support groups identified by exact binaries and ports."""

    errors: list[str] = []
    for group in identifiable_phase14b_runtime_groups(config):
        try:
            _stop_process_group(group, "Phase-14B runtime support")
        except BaseException as exc:
            errors.append(f"runtime group {group}: {type(exc).__name__}: {exc}")
    return errors


def restore_interrupted_radio(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Restore an interrupted live channel before its owned radio is stopped."""

    actuator = config["actuator"]
    if not phase14a.tcp_listening(int(actuator["telnet_port"])):
        return None
    session = n2.TelnetSession(
        str(actuator["telnet_host"]),
        int(actuator["telnet_port"]),
        float(actuator["response_timeout_s"]),
        int(actuator["max_response_bytes"]),
    )
    try:
        before = session.command("channelmod show current")[-1]
        model = n2.parse_channel_models(before).get(str(actuator["channel_model_name"]))
        require(model is not None, "interrupted RFsim channel is absent")
        target = "-50"
        response = session.command(f"channelmod modify {int(model['model_index'])} noise_power_dB {target}")[-1]
        n2.Runner.validate_modify_response(response, target)
        after = session.command("channelmod show current")[-1]
        restored = n2.parse_channel_models(after).get(str(actuator["channel_model_name"]), {})
        require(math.isclose(float(restored.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-6), "interrupted RFsim restoration read-back failed")
        return {
            "noise_power_db": float(restored["noise_power_db"]),
            "before_sha256": sha256_bytes(before.encode("utf-8")),
            "response_sha256": sha256_bytes(response.encode("utf-8")),
            "after_sha256": sha256_bytes(after.encode("utf-8")),
            "verified": True,
        }
    finally:
        session.close()


def _owned_softmodem_groups(config: Mapping[str, Any], radio_state: Path) -> list[tuple[str, int]]:
    """Identify only softmodems whose argv names this lifecycle's materialized state."""

    build = repo_path(str(config["paths"]["oai_ran_build"]))
    executable_to_label = {
        str((build / "nr-uesoftmodem").resolve(strict=True)): "ue",
        str((build / "nr-softmodem").resolve(strict=True)): "gnb",
    }
    result: list[tuple[str, int]] = []
    for executable, label in executable_to_label.items():
        groups = {
            int(row["process_group_id"])
            for row in phase14a.process_table()
            if process_is_live(int(row["pid"]))
            and str(row["executable"]) == executable
            and str(radio_state) in str(row["command"])
        }
        require(len(groups) <= 1, f"ambiguous owned {label} topology during cleanup: {sorted(groups)}")
        if groups:
            result.append((label, next(iter(groups))))
    return result


def _remove_owned_radio_scratch(radio_state: Path, radio_namespace: Path) -> None:
    if not radio_state.exists():
        return
    resolved = radio_state.resolve(strict=True)
    require(resolved.parent == radio_namespace.resolve(strict=False), f"refusing to remove non-owned radio scratch: {resolved}")
    allowed_leaves = {f"{index:02d}_{profile}" for index, profile in enumerate(PROFILE_ORDER)} | {PROBE_QUALIFICATION_RADIO_STATE_LEAF}
    require(resolved.name in allowed_leaves, f"unexpected radio scratch leaf: {resolved}")
    shutil.rmtree(resolved)
    if radio_namespace.is_dir() and not any(radio_namespace.iterdir()):
        radio_namespace.rmdir()


def teardown_profile_runtime(
    config: Mapping[str, Any],
    radio_state: Path,
    radio_namespace: Path,
    attached_radio: Mapping[str, Any] | None,
    *,
    restore_verified: bool,
) -> dict[str, Any]:
    """Tear down one owned radio lifecycle and prove no state survives."""

    signals: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    try:
        errors.extend(stop_recorded_runtime_support(radio_state))
        errors.extend(stop_identifiable_phase14b_runtime_support(config))
    except BaseException as exc:
        errors.append(f"runtime-support identification: {type(exc).__name__}: {exc}")
    try:
        if attached_radio is not None:
            groups = [
                ("ue", int(attached_radio["ue_process_topology"]["process_group_id"])),
                ("gnb", int(attached_radio["gnb_process_topology"]["process_group_id"])),
            ]
        else:
            groups = _owned_softmodem_groups(config, radio_state)
        require(len({group for _, group in groups}) == len(groups), "gNB and UE unexpectedly share a process group")
        for label, group in groups:
            try:
                signals[label] = _stop_process_group(group, label)
            except BaseException as exc:
                errors.append(f"{label} teardown: {type(exc).__name__}: {exc}")
    except BaseException as exc:
        errors.append(f"topology identification: {type(exc).__name__}: {exc}")

    cn = repo_path(str(config["paths"]["oai_cn"]))
    try:
        _run_checked(
            ["sudo", "-n", "docker", "compose", "down", "--remove-orphans"],
            cwd=cn,
            timeout_s=120.0,
        )
    except BaseException as exc:
        errors.append(f"core-network teardown: {type(exc).__name__}: {exc}")
    interface = str(config["radio"]["ue_interface"])
    deadline = time.monotonic() + 10.0
    while phase14a.tunnel_ip(interface) is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    tunnel_deleted_explicitly = False
    if phase14a.tunnel_ip(interface) is not None:
        try:
            _run_checked(["sudo", "-n", "ip", "link", "delete", interface], timeout_s=10.0)
            tunnel_deleted_explicitly = True
        except BaseException as exc:
            errors.append(f"UE tunnel teardown: {type(exc).__name__}: {exc}")
    try:
        _remove_owned_radio_scratch(radio_state, radio_namespace)
    except BaseException as exc:
        errors.append(f"radio scratch cleanup: {type(exc).__name__}: {exc}")
    cold: dict[str, Any] = {}
    try:
        cold = require_cold_profile_runtime(config, radio_state)
        require(not radio_namespace.exists(), f"radio-state namespace survived teardown: {radio_namespace}")
    except BaseException as exc:
        errors.append(f"cold postflight: {type(exc).__name__}: {exc}")
    return {
        "owned_process_group_signals": signals,
        "traffic_and_telemetry_stopped": True,
        "noise_power_db_restored_and_read_back": bool(restore_verified),
        "oai_core_compose_down": not any(error.startswith("core-network") for error in errors),
        "ue_tunnel_deleted_explicitly": tunnel_deleted_explicitly,
        "radio_state_scratch_removed": not radio_state.exists(),
        "radio_state_namespace_removed": not radio_namespace.exists(),
        "cold_postflight": cold,
        "errors": errors,
        "all_lifecycle_gates_passed": not errors and bool(restore_verified),
    }


def run_manifest_payload(
    *,
    output: Path,
    config_path: Path,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    radio_namespace: Path,
) -> dict[str, Any]:
    head_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(head_result.returncode == 0, f"cannot resolve implementation HEAD: {head_result.stderr.strip()}")
    head = head_result.stdout.strip()
    return {
        "schema": "scenesense.splitfusion_phase14b_run_manifest.v1",
        "status": "IMMUTABLE_BEFORE_REPLAY",
        "run_id": str(uuid.uuid4()),
        "created_wall_ns": time.time_ns(),
        "output_path": str(output),
        "implementation_head": head,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(config_path.resolve(strict=True).relative_to(ROOT)),
        "config_sha256": sha256_file(config_path.resolve(strict=True)),
        "phase14a_mapping_sha256": provenance["mapping_sha256"],
        "phase14a_campaign_binding_historical_sha256": provenance["historical_campaign_binding_sha256"],
        "phase14a_campaign_binding_current_sha256": provenance["current_campaign_binding_sha256"],
        "topology_validator_commit": provenance["topology_validator_commit"],
        "radio_profile_id": RADIO_PROFILE_ID,
        "radio_profile_sha256": provenance["radio_profile_sha256"],
        "radio_state_namespace": str(radio_namespace),
        "radio_lifecycle": copy.deepcopy(config["radio_lifecycle"]),
        "profile_lifecycle_plan": profile_lifecycle_plan(config, radio_namespace),
        "provenance": copy.deepcopy(config["provenance"]),
        "profile_order": list(PROFILE_ORDER),
        "profiles": copy.deepcopy(config["profiles"]),
        "scheduler": copy.deepcopy(config["scheduler"]),
        "replay": copy.deepcopy(config["replay"]),
        "observation_alignment": copy.deepcopy(config["observation_alignment"]),
        "traffic": copy.deepcopy(config["traffic"]),
        "probe_qualification": copy.deepcopy(config["probe_qualification"]),
        "acceptance": copy.deepcopy(config["acceptance"]),
        "durability": copy.deepcopy(config["durability"]),
        "old_40mhz_replay_used_as_evidence": False,
        "original_uncorrected_replay_used_as_evidence": False,
        "campaigns_authorized": False,
    }


def validate_run_manifest(
    path: Path,
    *,
    output: Path,
    config_path: Path,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    radio_namespace: Path,
) -> dict[str, Any]:
    manifest = load_json(path)
    expected = run_manifest_payload(
        output=output,
        config_path=config_path,
        config=config,
        provenance=provenance,
        radio_namespace=radio_namespace,
    )
    for volatile in ("run_id", "created_wall_ns", "implementation_head"):
        expected.pop(volatile)
    observed = dict(manifest)
    for volatile in ("run_id", "created_wall_ns", "implementation_head"):
        require(volatile in observed, f"run manifest lacks {volatile}")
        observed.pop(volatile)
    require(observed == expected, "immutable run manifest disagrees with current sealed implementation")
    return manifest


def wrap_profile_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(payload))
    return {
        "schema": "scenesense.splitfusion_phase14b_atomic_profile_record.v1",
        "payload_sha256": sha256_bytes(canonical_bytes(body)),
        "payload": body,
    }


def load_profile_record(
    path: Path,
    manifest_sha256: str,
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    record = load_json(path)
    require(record.get("schema") == "scenesense.splitfusion_phase14b_atomic_profile_record.v1", f"profile record schema drift: {path}")
    payload = record.get("payload")
    require(isinstance(payload, dict), f"profile record payload missing: {path}")
    require(record.get("payload_sha256") == sha256_bytes(canonical_bytes(payload)), f"profile record payload seal mismatch: {path}")
    require(payload.get("run_manifest_sha256") == manifest_sha256, f"profile record manifest mismatch: {path}")
    require(payload.get("profile_id") == profile["profile_id"], f"profile record identity mismatch: {path}")
    require(payload.get("trace_sha256") == profile["trace_sha256"], f"profile record trace mismatch: {path}")
    require(
        payload.get("status") in {"PROFILE_REPLAY_ACCEPTED", "PROFILE_REPLAY_REJECTED"},
        f"only complete profile records can be resumed: {path}",
    )
    commands = payload.get("commands", [])
    require(len(commands) == 4200, f"profile record command count drift: {path}")
    lifecycle = payload.get("radio_lifecycle")
    require(isinstance(lifecycle, dict), f"profile lifecycle evidence missing: {path}")
    require(lifecycle.get("all_lifecycle_gates_passed") is True, f"profile lifecycle did not fully tear down: {path}")
    require(lifecycle.get("profile_index") == PROFILE_ORDER.index(str(profile["profile_id"])), f"profile lifecycle index drift: {path}")
    require(lifecycle.get("generator_start_index") == 0, f"profile generator did not restart at zero: {path}")
    require(lifecycle.get("settings_executed") == 4200, f"profile lifecycle setting count drift: {path}")
    attached_seal = payload.get("attached_radio_seal")
    require(isinstance(attached_seal, dict), f"profile attached-radio seal missing: {path}")
    require(attached_seal.get("radio_profile_id") == RADIO_PROFILE_ID, f"profile radio identity drift: {path}")
    require(lifecycle.get("attached_radio_seal") == attached_seal, f"profile lifecycle radio seal mismatch: {path}")
    for name in ("gnb_topology", "ue_topology"):
        topology = attached_seal.get(name, {})
        require(topology.get("same_executable_process_count") == 3, f"profile {name} process count drift: {path}")
        require(topology.get("worker_count") == 2, f"profile {name} worker count drift: {path}")
        require(topology.get("endpoint_roles_verified") is True, f"profile {name} endpoints were not verified: {path}")
    granularity = float(config["replay"]["command_granularity_db"])
    for step, (command, frozen_sample) in enumerate(zip(commands, profile["prefix"])):
        require(int(command["step_index"]) == step, f"resumed command order drift: {path}:{step}")
        require(int(command["state_index"]) == int(frozen_sample[0]), f"resumed state drift: {path}:{step}")
        require(float(command["target_snr_db"]) == float(frozen_sample[1]), f"resumed target drift: {path}:{step}")
        expected_command = replay.inverse_interpolate(float(frozen_sample[1]), mapping, granularity)
        require(
            float(command["quantized_rfsim_command_db"]) == float(expected_command),
            f"resumed mapping drift: {path}:{step}",
        )
    observed_rntis = [int(value) for value in payload["summary"]["observed_rntis"]]
    scheduler_ok = all(
        int(sample["mcs_table"]) == int(config["scheduler"]["required_mcs_table"])
        and int(sample["force_ul_mcs"]) == int(config["scheduler"]["required_force_ul_mcs"])
        for command in commands
        for sample in command["observed_mcs_samples"]
    )
    summary, gates, fade = summarize_profile(
        commands,
        profile,
        config,
        observed_rntis=observed_rntis,
        scheduler_seals_ok=scheduler_ok,
        topology_seal_ok=bool(payload["gates"]["radio_and_topology_seals"]),
    )
    require(summary == payload["summary"], f"resumed profile summary does not recompute: {path}")
    require(gates == payload["gates"], f"resumed profile gates do not recompute: {path}")
    require(fade == payload["fade_recovery_diagnostic"], f"resumed FADE_REOVERY diagnostic does not recompute: {path}")
    passed = all(gates.values())
    require(payload.get("all_gates_passed") is passed, f"profile gate aggregate drift: {path}")
    require(
        payload.get("status") == ("PROFILE_REPLAY_ACCEPTED" if passed else "PROFILE_REPLAY_REJECTED"),
        f"profile completion status disagrees with gates: {path}",
    )
    return payload


def summarize_subset(rows: Sequence[Mapping[str, Any]], indices: set[int]) -> dict[str, Any]:
    selected = [row for row in rows if int(row["step_index"]) in indices]
    observed = [row for row in selected if row["observation_status"] == "OBSERVATION_ACCEPTED"]
    signed = [float(row["achieved_pusch_snr_median_db"]) - float(row["target_snr_db"]) for row in observed]
    return {
        "target_count": len(selected),
        "accepted_observation_count": len(observed),
        "observation_coverage": len(observed) / len(selected) if selected else 0.0,
        "mae_db": statistics.fmean(abs(value) for value in signed) if signed else None,
        "bias_achieved_minus_target_db": statistics.fmean(signed) if signed else None,
        "p95_absolute_error_db": percentile([abs(value) for value in signed], 0.95),
    }


def summarize_probe_qualification(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    observed_rntis: Sequence[int],
    scheduler_seals_ok: bool,
    topology_seal_ok: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """A mechanical instrumentation gate: fixed 10-second high-SNR dry run.

    Not a traffic-rate search -- the probe rate/size/pacing are fixed and this
    function only checks whether the unchanged 15-ms-guard/next-send window
    rule and the unchanged 90% coverage bar can be met at all under this exact
    probe before committing to the four-profile replay.
    """

    acceptance = config["acceptance"]
    replay_config = config["replay"]
    expected = len(rows)
    period_ns = int(replay_config["sample_period_ms"]) * 1_000_000
    applied = [row for row in rows if str(row["command_status"]).startswith("ACK_")]
    observed = [row for row in rows if row["observation_status"] == "OBSERVATION_ACCEPTED"]
    sends = [int(row["command_send_monotonic_ns"]) for row in applied]
    ack_ratio = len(applied) / expected if expected else 0.0
    command_validity_coverage = len(observed) / len(applied) if applied else 0.0
    no_burst = all(right - left >= period_ns for left, right in zip(sends, sends[1:]))
    gates = {
        "minimum_send_completion_ratio": ack_ratio >= float(acceptance["minimum_command_ack_ratio"]),
        "no_burst_catch_up": no_burst,
        "finite_pusch_observations": len(observed) > 0,
        "minimum_observation_coverage": command_validity_coverage >= float(acceptance["minimum_observation_coverage"]),
        "single_rnti": len(observed_rntis) == 1,
        "scheduler_seals": scheduler_seals_ok,
        "radio_and_topology_seals": topology_seal_ok,
    }
    summary = {
        "steps": expected,
        "commands_acknowledged": len(applied),
        "command_ack_ratio": ack_ratio,
        "accepted_observation_windows": len(observed),
        "command_validity_coverage": command_validity_coverage,
        "observed_rntis": list(observed_rntis),
        "all_gates_passed": all(gates.values()),
    }
    return summary, gates


def summarize_profile(
    rows: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    observed_rntis: Sequence[int],
    scheduler_seals_ok: bool,
    topology_seal_ok: bool,
) -> tuple[dict[str, Any], dict[str, bool], dict[str, Any] | None]:
    acceptance = config["acceptance"]
    replay_config = config["replay"]
    quantiles = [float(value) for value in acceptance["reported_quantiles"]]
    expected = int(replay_config["samples_per_profile"])
    applied = [row for row in rows if str(row["command_status"]).startswith("ACK_")]
    skipped = [row for row in rows if row["command_status"] == "SKIP_OBSOLETE_NEVER_BURST"]
    observed = [row for row in rows if row["observation_status"] == "OBSERVATION_ACCEPTED"]
    nominal_observed = [row for row in rows if row["nominal_observation_status"] == "OBSERVATION_ACCEPTED"]
    signed = [float(row["achieved_pusch_snr_median_db"]) - float(row["target_snr_db"]) for row in observed]
    ack_latencies = [float(row["command_ack_latency_ms"]) for row in applied]
    sends = [int(row["command_send_monotonic_ns"]) for row in applied]
    period_ns = int(replay_config["sample_period_ms"]) * 1_000_000
    granularity = float(replay_config["command_granularity_db"])
    ack_ratio = len(applied) / expected
    # Correction 2: command_validity_coverage is denominated over acknowledged
    # commands only (a skipped target was never commanded, so it is excluded
    # from mapping coverage though it remains a schedule-fidelity failure).
    # nominal_schedule_coverage reproduces the original denominator (over all
    # 4,200 targets) for direct, diagnostic-only comparison with the failed
    # original replay; it never drives acceptance.
    command_validity_coverage = len(observed) / len(applied) if applied else 0.0
    nominal_schedule_coverage = len(nominal_observed) / expected
    coverage = command_validity_coverage
    mae = statistics.fmean(abs(value) for value in signed) if signed else None
    bias = statistics.fmean(signed) if signed else None
    p95 = percentile([abs(value) for value in signed], 0.95)
    complete_accounting = len(applied) + len(skipped) == expected
    no_burst = all(right - left >= period_ns for left, right in zip(sends, sends[1:]))
    order_exact = [int(row["step_index"]) for row in rows] == list(range(expected))
    quantization_exact = all(
        math.isclose(
            float(row["quantized_rfsim_command_db"]) / granularity,
            round(float(row["quantized_rfsim_command_db"]) / granularity),
            abs_tol=1e-9,
        )
        for row in rows
    )
    gates = {
        "trace_hash_reproduced": prepared["trace_sha256"] == next(
            row["trace_sha256"] for row in config["profiles"] if row["profile_id"] == prepared["profile_id"]
        ),
        "exactly_4200_targets": len(rows) == expected == 4200,
        "command_order_exact": order_exact,
        "command_quantization_exact": quantization_exact,
        "complete_ack_or_skip_accounting": complete_accounting,
        "minimum_command_ack_ratio": ack_ratio >= float(acceptance["minimum_command_ack_ratio"]),
        "maximum_command_ack_p95": bool(ack_latencies) and float(percentile(ack_latencies, 0.95)) <= float(acceptance["maximum_command_ack_p95_ms"]),
        "no_burst_catch_up": no_burst,
        "no_wrap_hold_or_reseed": True,
        "minimum_observation_coverage": coverage >= float(acceptance["minimum_observation_coverage"]),
        "maximum_tracking_mae": mae is not None and mae <= float(acceptance["maximum_tracking_mae_db"]),
        "maximum_absolute_bias": bias is not None and abs(bias) <= float(acceptance["maximum_absolute_bias_db"]),
        "maximum_tracking_p95_absolute_error": p95 is not None and p95 <= float(acceptance["maximum_tracking_p95_absolute_error_db"]),
        "single_rnti": len(observed_rntis) == 1,
        "scheduler_seals": scheduler_seals_ok,
        "radio_and_topology_seals": topology_seal_ok,
        "mapping_monotonic_and_in_range": True,
        "traffic_contract_constant": bool(config["traffic"]["constant_across_profiles"]),
    }
    fade: dict[str, Any] | None = None
    if prepared["profile_id"] == "FADE_RECOVERY":
        prefix = prepared["prefix"]
        transitions = {index for index in range(1, expected) if prefix[index][0] != prefix[index - 1][0]}
        threshold = float(acceptance["fade_recovery"]["large_step_threshold_db"])
        large_steps = {
            index
            for index in range(1, expected)
            if abs(float(prefix[index][1]) - float(prefix[index - 1][1])) >= threshold
        }
        transition_result = summarize_subset(rows, transitions)
        large_step_result = summarize_subset(rows, large_steps)
        fade = {
            "large_step_threshold_db": threshold,
            "state_transitions": transition_result,
            "large_steps": large_step_result,
        }
        fade_config = acceptance["fade_recovery"]
        gates.update(
            {
                "fade_transition_count": len(transitions) == int(fade_config["expected_state_transition_count"]),
                "fade_large_step_count": len(large_steps) == int(fade_config["expected_large_step_count"]),
                "fade_transition_coverage": transition_result["observation_coverage"] >= float(fade_config["minimum_subset_observation_coverage"]),
                "fade_large_step_coverage": large_step_result["observation_coverage"] >= float(fade_config["minimum_subset_observation_coverage"]),
                "fade_transition_mae": transition_result["mae_db"] is not None and float(transition_result["mae_db"]) <= float(fade_config["maximum_subset_mae_db"]),
                "fade_large_step_mae": large_step_result["mae_db"] is not None and float(large_step_result["mae_db"]) <= float(fade_config["maximum_subset_mae_db"]),
                "fade_transition_bias": transition_result["bias_achieved_minus_target_db"] is not None and abs(float(transition_result["bias_achieved_minus_target_db"])) <= float(fade_config["maximum_subset_absolute_bias_db"]),
                "fade_large_step_bias": large_step_result["bias_achieved_minus_target_db"] is not None and abs(float(large_step_result["bias_achieved_minus_target_db"])) <= float(fade_config["maximum_subset_absolute_bias_db"]),
                "fade_transition_p95": transition_result["p95_absolute_error_db"] is not None and float(transition_result["p95_absolute_error_db"]) <= float(fade_config["maximum_subset_p95_absolute_error_db"]),
                "fade_large_step_p95": large_step_result["p95_absolute_error_db"] is not None and float(large_step_result["p95_absolute_error_db"]) <= float(fade_config["maximum_subset_p95_absolute_error_db"]),
            }
        )
    summary = {
        "profile_id": prepared["profile_id"],
        "trace_id": prepared["trace_id"],
        "trace_sha256": prepared["trace_sha256"],
        "targets": expected,
        "commands_acknowledged": len(applied),
        "commands_skipped_obsolete": len(skipped),
        "command_ack_ratio": ack_ratio,
        "command_ack_latency_ms_p50": percentile(ack_latencies, 0.50),
        "command_ack_latency_ms_p95": percentile(ack_latencies, 0.95),
        "command_ack_latency_ms_max": max(ack_latencies) if ack_latencies else None,
        "accepted_observation_windows": len(observed),
        "empty_observation_windows": sum(row["observation_status"] == "EMPTY_OBSERVATION_WINDOW" for row in rows),
        "underpopulated_observation_windows": sum(row["observation_status"] == "UNDERPOPULATED_OBSERVATION_WINDOW" for row in rows),
        "observation_coverage": coverage,
        "command_validity_coverage": command_validity_coverage,
        "nominal_schedule_coverage": nominal_schedule_coverage,
        "nominal_accepted_observation_windows": len(nominal_observed),
        "nominal_empty_observation_windows": sum(row["nominal_observation_status"] == "EMPTY_OBSERVATION_WINDOW" for row in rows),
        "nominal_underpopulated_observation_windows": sum(row["nominal_observation_status"] == "UNDERPOPULATED_OBSERVATION_WINDOW" for row in rows),
        "tracking_mae_db": mae,
        "bias_achieved_minus_target_db": bias,
        "tracking_p95_absolute_error_db": p95,
        "target_snr_db": distribution([float(row["target_snr_db"]) for row in rows], quantiles),
        "achieved_pusch_snr_db": distribution([float(row["achieved_pusch_snr_median_db"]) for row in observed], quantiles),
        "observed_rntis": list(observed_rntis),
        "all_gates_passed": all(gates.values()),
    }
    return summary, gates, fade


class CorrectedFourProfileReplay(phase14a.AttachedCalibration):
    """Four independent radio lifecycles with atomic profile durability."""

    def __init__(
        self,
        config_path: Path,
        output_value: str,
        radio_namespace: Path,
        *,
        resume_run: bool,
    ) -> None:
        self.phase14b_config_path = config_path.resolve(strict=True)
        self.provenance = verify_provenance(self.phase14b_config_path)
        self.phase14b = self.provenance["config"]
        self.phase14a_config_path = repo_path(str(self.phase14b["provenance"]["phase14a_config"]["path"]))
        self.phase14a_binding_path = repo_path(str(self.phase14b["provenance"]["phase14a_campaign_binding"]["path"]))
        self.base_radio_config = load_json(self.phase14a_config_path)
        self.radio_namespace = resolve_radio_namespace(self.phase14b, radio_namespace)
        self.probe_radio_namespace = resolve_radio_namespace(
            self.phase14b, self.radio_namespace.parent / f"{self.radio_namespace.name}__probe_qualification"
        )
        self.lifecycle_plan = profile_lifecycle_plan(self.phase14b, self.radio_namespace)
        self.output_value = output_value
        self.resume_run = resume_run
        self.prepared = prepare_frozen_profiles(self.phase14b)
        self.mapping = self.provenance["mapping"]
        self.work: tempfile.TemporaryDirectory[str] | None = None
        self.durable_output: Path | None = None
        self.final_restore_state: dict[str, Any] | None = None
        self.initial_clean_state: dict[str, Any] | None = None
        self.last_topology_health_ns = 0
        self.topology_seal_ok = True
        self.session_started = False
        self.restored = False
        self.nonclean_attempted = False
        traffic = self.phase14b["traffic"]
        require(traffic["method"] == "FIXED_RATE_LIGHTWEIGHT_UDP_PUSCH_OBSERVATION_PROBE", "corrected traffic method drift")
        require(math.isclose(float(traffic["rate_hz"]), 100.0, abs_tol=1e-9), "probe rate is not the registered 100 Hz")
        require(int(traffic["datagram_bytes"]) == 1200, "probe datagram size is not the registered 1,200 bytes")
        require(bool(traffic["constant_across_profiles"]) is True, "corrected traffic must be identical across profiles")
        require(bool(traffic["tuned_per_profile"]) is False, "corrected traffic must not be tuned per profile")
        require(bool(traffic["raw_payload_retained"]) is False, "corrected traffic must not retain raw UDP payload")
        require(
            not bool(traffic.get("is_splitfusion_application_traffic", False)),
            "the probe must never be described as SplitFusion application traffic",
        )
        require(
            int(self.base_radio_config["telemetry"]["required_mcs_table"])
            == int(self.phase14b["scheduler"]["required_mcs_table"]),
            "scheduler MCS-table seal differs from Phase-14A",
        )
        require(
            int(self.base_radio_config["telemetry"]["required_force_ul_mcs"])
            == int(self.phase14b["scheduler"]["required_force_ul_mcs"]),
            "forced-MCS seal differs from Phase-14A",
        )

    def initialize_attached_session(self, radio_state: Path) -> None:
        """Reset all attached-radio, traffic, parser, counter, and timing state."""

        phase14a.AttachedCalibration.__init__(
            self,
            self.phase14a_config_path,
            self.phase14a_binding_path,
            radio_state,
            self.output_value,
        )
        self.output = self.durable_output
        self.work = None
        self.final_restore_state = None
        self.initial_clean_state = None
        self.last_topology_health_ns = 0
        self.topology_seal_ok = True
        self.session_started = True
        self.traffic_measurement: dict[str, Any] | None = None

    def launch_radio(self, radio_state: Path) -> subprocess.CompletedProcess[str]:
        require_cold_profile_runtime(self.base_radio_config, radio_state)
        return _run_checked(
            [
                str(self.provenance["launcher_path"]),
                "--execute",
                str(self.provenance["launcher_execution_token"]),
                "--output",
                str(radio_state),
            ],
            timeout_s=float(self.phase14b["radio_lifecycle"]["launcher_timeout_s"]),
        )

    def stop_runtime_support_keep_actuator(self) -> list[str]:
        """Stop fresh traffic/telemetry before final RFsim restoration."""

        errors: list[str] = []
        for collector_name in ("live_mcs", "live_pusch"):
            collector = getattr(self, collector_name)
            if collector is not None:
                try:
                    collector.stop()
                except Exception as exc:
                    errors.append(f"{collector_name}: {type(exc).__name__}: {exc}")
                setattr(self, collector_name, None)
        for process in reversed(self.processes):
            try:
                process.stop()
            except Exception as exc:
                errors.append(f"{process.name}: {type(exc).__name__}: {exc}")
        self.processes.clear()
        return errors

    def close_actuator_and_work(self) -> list[str]:
        errors: list[str] = []
        if self.telnet is not None:
            try:
                self.telnet.close()
            except Exception as exc:
                errors.append(f"telnet: {type(exc).__name__}: {exc}")
            self.telnet = None
        if self.work is not None:
            try:
                self.work.cleanup()
            except Exception as exc:
                errors.append(f"temporary work cleanup: {type(exc).__name__}: {exc}")
            self.work = None
        return errors

    def health(self, *, force_topology: bool = False) -> None:
        now = time.monotonic_ns()
        for process in self.processes:
            require(process.process.poll() is None, f"runner-owned process exited: {process.name}")
        require(self.live_pusch is not None and self.live_pusch.process.poll() is None, "PUSCH collector exited")
        require(self.live_mcs is not None and self.live_mcs.process.poll() is None, "MCS collector exited")
        if not force_topology and now - self.last_topology_health_ns < 1_000_000_000:
            return
        self.last_topology_health_ns = now
        require(self.attached_radio is not None, "attached radio identity is unavailable")
        current = phase14a.running_softmodems(
            self.config,
            verify_endpoint_roles=force_topology,
        )
        for name in ("gnb", "ue"):
            expected = self.attached_radio[f"{name}_process_topology"]
            observed = current[name]["topology"]
            require(
                observed["main_pid"] == expected["main_pid"]
                and observed["worker_pids"] == expected["worker_pids"],
                f"{name} service topology changed during replay",
            )
        require(
            phase14a.tunnel_ip(str(self.config["radio"]["ue_interface"]))
            == self.config["radio"]["ue_static_ip"],
            "UE tunnel identity changed during replay",
        )

    def wait_until(self, deadline_ns: int) -> None:
        while True:
            remaining = int(deadline_ns) - time.monotonic_ns()
            if remaining <= 0:
                return
            self.health()
            time.sleep(min(remaining / 1e9, 0.02))

    def start_runtime_support(self) -> None:
        require(self.durable_output is not None, "durable output is unavailable")
        self.work = tempfile.TemporaryDirectory(prefix="splitfusion-phase14b-")
        durable = self.output
        self.output = Path(self.work.name).resolve(strict=True)
        try:
            super().start_telemetry()
            self.start_traffic()
        finally:
            self.output = durable

    def start_traffic(self) -> None:
        """Correction 1: one fixed, lightweight, evenly paced UDP probe.

        This is measurement instrumentation only -- it exists to distribute
        independent PUSCH transmission opportunities across each 100-ms
        interval, not to reproduce or saturate SplitFusion application
        traffic. Rate, datagram size, and destination are fixed and identical
        across every profile and the probe qualification.
        """

        radio = self.config["radio"]
        traffic = self.phase14b["traffic"]
        interface = str(radio["ue_interface"])
        inspect = n2.run_checked(
            ["sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}", "oai-ext-dn"]
        )
        ext_dn_pid = int(inspect.stdout.strip())
        sink_csv = self.output / "traffic/probe_sink.csv"
        sink_csv.parent.mkdir(parents=True, exist_ok=True)
        self.spawn(
            "probe_sink",
            [
                "sudo", "-n", "nsenter", "-t", str(ext_dn_pid), "-n",
                "/usr/bin/python3",
                str(repo_path(str(traffic["paths"]["sink"]))),
                "--bind", str(traffic["ext_dn_ip"]),
                "--port", str(traffic["remote_port"]),
                "--out", str(sink_csv),
                "--timeout-s", str(traffic["sink_timeout_s"]),
            ],
            "logs/probe_sink.log",
            root_owned=True,
        )
        time.sleep(0.5)
        sender_csv = self.output / "traffic/probe_sender.csv"
        before = tunnel_counters(interface)
        self.spawn(
            "probe_sender",
            [
                sys.executable,
                str(repo_path(str(traffic["paths"]["sender"]))),
                "--bind-host", str(radio["ue_static_ip"]),
                "--remote-host", str(traffic["ext_dn_ip"]),
                "--remote-port", str(traffic["remote_port"]),
                "--rate-hz", str(traffic["rate_hz"]),
                "--datagram-bytes", str(traffic["datagram_bytes"]),
                "--duration-s", str(traffic["client_duration_s"]),
                "--log-csv", str(sender_csv),
            ],
            "logs/probe_sender.log",
        )
        # Captured immediately after spawn (not after the readiness sleep
        # below) so it matches the sender CSV's full row count in
        # stop_traffic_and_measure -- otherwise achieved_pps would be
        # computed over a shorter runtime than the packets it counts.
        self.traffic_started_monotonic_ns = time.monotonic_ns()
        probe_window_s = float(traffic["readiness_probe_window_s"])
        time.sleep(probe_window_s)
        self.health()
        after = tunnel_counters(interface)
        tx_before, tx_after = before.get("tx_bytes"), after.get("tx_bytes")
        delta = (tx_after - tx_before) if tx_before is not None and tx_after is not None else None
        expected_bytes = float(traffic["rate_hz"]) * int(traffic["datagram_bytes"]) * probe_window_s
        minimum_delta = float(traffic["readiness_min_tunnel_tx_fraction"]) * expected_bytes
        require(
            delta is not None and delta >= minimum_delta,
            f"probe traffic readiness check found insufficient UE tunnel TX growth in {probe_window_s}s: "
            f"{delta} < {minimum_delta} (expected ~{expected_bytes:.0f})",
        )
        self.traffic_readiness = {
            "tunnel_tx_byte_delta_at_readiness": delta,
            "expected_tunnel_tx_bytes_at_readiness": expected_bytes,
            "probe_window_s": probe_window_s,
        }
        self.traffic_measurement = None
        self.sender_csv_path = sender_csv
        self.sink_csv_path = sink_csv

    def stop_traffic_and_measure(self) -> dict[str, Any]:
        """Stop the probe sender/sink and record offered/received packets, bytes, and timing."""

        started_ns = getattr(self, "traffic_started_monotonic_ns", None)
        require(started_ns is not None, "probe traffic was never started")
        sender = next((p for p in self.processes if p.name == "probe_sender"), None)
        sink = next((p for p in self.processes if p.name == "probe_sink"), None)
        require(sender is not None and sink is not None, "probe traffic processes are not tracked")
        alive_before_stop = sender.process.poll() is None and sink.process.poll() is None
        require(alive_before_stop, "probe sender/sink process exited before the profile finished")
        runtime_s = (time.monotonic_ns() - int(started_ns)) / 1e9
        sender.stop()
        sink.stop()
        self.processes = [process for process in self.processes if process not in (sender, sink)]

        offered_packets = 0
        offered_bytes = 0
        sender_path = getattr(self, "sender_csv_path", None)
        if sender_path is not None and sender_path.is_file():
            with sender_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    offered_packets += 1
                    offered_bytes += int(row["bytes"])
        received_packets = 0
        received_bytes = 0
        sink_path = getattr(self, "sink_csv_path", None)
        if sink_path is not None and sink_path.is_file():
            with sink_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    received_packets += 1
                    if row.get("size"):
                        received_bytes += int(row["size"])
        traffic = self.phase14b["traffic"]
        measurement = {
            "schema": "scenesense.splitfusion_phase14b_corrected_traffic_measurement.v1",
            "method": str(traffic["method"]),
            "rate_hz": float(traffic["rate_hz"]),
            "datagram_bytes": int(traffic["datagram_bytes"]),
            "process_alive_throughout": alive_before_stop,
            "runtime_s": runtime_s,
            "offered_packets": offered_packets,
            "offered_bytes": offered_bytes,
            "received_packets": received_packets,
            "received_bytes": received_bytes,
            "achieved_pps": offered_packets / runtime_s if runtime_s > 0 else None,
            "readiness": dict(getattr(self, "traffic_readiness", {}) or {}),
            "is_splitfusion_application_traffic": False,
            "is_campaign_throughput": False,
            "raw_payload_retained": False,
        }
        self.traffic_measurement = measurement
        return measurement

    def write_runtime_ownership(self, radio_state: Path) -> None:
        groups: list[dict[str, Any]] = []
        candidates: list[tuple[str, int]] = [
            (process.name, process.process.pid) for process in self.processes
        ]
        if self.live_pusch is not None:
            candidates.append(("live_pusch", self.live_pusch.process.pid))
        if self.live_mcs is not None:
            candidates.append(("live_mcs", self.live_mcs.process.pid))
        seen: set[int] = set()
        for name, pid in candidates:
            process_group_id = os.getpgid(pid)
            if process_group_id in seen:
                continue
            seen.add(process_group_id)
            members = _process_group_identity(process_group_id)
            require(members, f"cannot seal runtime process group for {name}")
            groups.append(
                {
                    "name": name,
                    "leader_pid": pid,
                    "process_group_id": process_group_id,
                    "members": members,
                }
            )
        atomic_json(
            radio_state / "PHASE14B_RUNTIME_OWNERSHIP.json",
            {
                "schema": "scenesense.splitfusion_phase14b_runtime_ownership.v1",
                "process_groups": groups,
            },
        )

    def open_clean_actuator(self) -> int:
        actuator = self.config["actuator"]
        self.telnet = n2.TelnetSession(
            str(actuator["telnet_host"]),
            int(actuator["telnet_port"]),
            float(actuator["response_timeout_s"]),
            int(actuator["max_response_bytes"]),
        )
        response = self.telnet.command("channelmod show current")[-1]
        models = n2.parse_channel_models(response)
        model = models.get(str(actuator["channel_model_name"]))
        require(model is not None, "registered RFsim channel is absent")
        require(model.get("model_type") == actuator["channel_model_type"], "RFsim channel type drift")
        require(model.get("owner") == actuator["channel_model_owner"], "RFsim channel owner drift")
        require(math.isclose(float(model.get("path_loss_db", math.nan)), 0.0, abs_tol=1e-6), "RFsim path-loss drift")
        require(math.isclose(float(model.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-6), "replay did not start clean at noise_power_dB=-50")
        self.initial_clean_state = {
            "noise_power_db": float(model["noise_power_db"]),
            "readback_sha256": sha256_bytes(response.encode("utf-8")),
            "verified": True,
        }
        return int(model["model_index"])

    def send_target(self, model_index: int, command_db: float) -> dict[str, Any]:
        require(self.telnet is not None, "RFsim control session is unavailable")
        target = f"{float(command_db):.12g}"
        command = f"channelmod modify {model_index} noise_power_dB {target}"
        self.nonclean_attempted = True
        self.restored = False
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(command)
        n2.Runner.validate_modify_response(response, target)
        return {
            "command": command,
            "command_send_monotonic_ns": sent_mono,
            "command_send_wall_ns": sent_wall,
            "command_ack_monotonic_ns": ack_mono,
            "command_ack_wall_ns": ack_wall,
            "command_ack_latency_ms": (ack_mono - sent_mono) / 1e6,
            "command_response_sha256": sha256_bytes(response.encode("utf-8")),
            "command_status": "ACK_ON_TIME",
        }

    def restore_and_readback(self, model_index: int) -> dict[str, Any]:
        require(self.telnet is not None, "RFsim control session is unavailable for restoration")
        target = "-50"
        command = f"channelmod modify {model_index} noise_power_dB {target}"
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(command)
        n2.Runner.validate_modify_response(response, target)
        state = self.telnet.command("channelmod show current")[-1]
        model = n2.parse_channel_models(state).get(str(self.config["actuator"]["channel_model_name"]), {})
        self.restored = math.isclose(float(model.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-6)
        require(self.restored, f"noise_power_dB=-50 restoration read-back failed: {model}")
        result = {
            "command": command,
            "send_monotonic_ns": sent_mono,
            "send_wall_ns": sent_wall,
            "ack_monotonic_ns": ack_mono,
            "ack_wall_ns": ack_wall,
            "ack_latency_ms": (ack_mono - sent_mono) / 1e6,
            "response_sha256": sha256_bytes(response.encode("utf-8")),
            "readback_sha256": sha256_bytes(state.encode("utf-8")),
            "noise_power_db": float(model["noise_power_db"]),
            "verified": True,
        }
        self.final_restore_state = result
        return result

    def _associate_observations(self, rows: list[dict[str, Any]]) -> tuple[list[int], bool]:
        """Correction 2: window end is the actual next send, not the nominal tick.

        Both a corrected (command-validity) and the original nominal window are
        computed for every acknowledged command; the nominal figure is retained
        only as a diagnostic and never drives acceptance.
        """

        require(self.current_rnti is not None, "single RNTI was not established")
        require(self.final_restore_state is not None, "final restoration must run before window association")
        pusch = [
            row
            for row in (
                telemetry.parse_live_pusch(item)
                for item in (self.live_pusch.snapshot() if self.live_pusch else [])
            )
            if row is not None
        ]
        mcs = [
            row
            for row in (
                telemetry.parse_live_mcs(item)
                for item in (self.live_mcs.snapshot() if self.live_mcs else [])
            )
            if row is not None
        ]
        restore_send_ns = int(self.final_restore_state["send_monotonic_ns"])
        first_ns = int(rows[0]["scheduled_monotonic_ns"])
        last_ns = max(int(rows[-1]["interval_end_monotonic_ns"]), restore_send_ns)
        relevant_pusch = [row for row in pusch if first_ns <= int(row["mono_ns"]) < last_ns]
        relevant_mcs = [row for row in mcs if first_ns <= int(row["mono_ns"]) < last_ns]
        relevant_pusch.sort(key=lambda row: int(row["mono_ns"]))
        relevant_mcs.sort(key=lambda row: int(row["mono_ns"]))
        pusch_times = [int(row["mono_ns"]) for row in relevant_pusch]
        mcs_times = [int(row["mono_ns"]) for row in relevant_mcs]
        observed_rntis = sorted({int(row["rnti"]) for row in [*relevant_pusch, *relevant_mcs]})
        scheduler_ok = all(
            int(row["mcs_table"]) == int(self.config["telemetry"]["required_mcs_table"])
            and int(row["force_ul_mcs"]) == int(self.config["telemetry"]["required_force_ul_mcs"])
            for row in relevant_mcs
        )
        alignment = self.phase14b["observation_alignment"]
        guard_ns = int(float(alignment["command_ack_guard_ms"]) * 1e6)
        minimum_pusch = int(alignment["minimum_pusch_samples_per_window"])
        minimum_mcs = int(alignment["minimum_mcs_samples_per_window"])
        next_send_by_index: list[int | None] = [None] * len(rows)
        next_send: int | None = None
        for index in range(len(rows) - 1, -1, -1):
            next_send_by_index[index] = next_send
            if rows[index].get("command_send_monotonic_ns") is not None:
                next_send = int(rows[index]["command_send_monotonic_ns"])

        def classify(start: int, end: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
            if start >= end:
                return [], [], "EMPTY_OBSERVATION_WINDOW"
            selected_pusch = [
                item
                for item in relevant_pusch[
                    bisect.bisect_left(pusch_times, start) : bisect.bisect_left(pusch_times, end)
                ]
                if int(item["rnti"]) == int(self.current_rnti)
            ]
            selected_mcs = [
                item
                for item in relevant_mcs[
                    bisect.bisect_left(mcs_times, start) : bisect.bisect_left(mcs_times, end)
                ]
                if int(item["rnti"]) == int(self.current_rnti)
            ]
            if not selected_pusch and not selected_mcs:
                status = "EMPTY_OBSERVATION_WINDOW"
            elif len(selected_pusch) < minimum_pusch or len(selected_mcs) < minimum_mcs:
                status = "UNDERPOPULATED_OBSERVATION_WINDOW"
            else:
                status = "OBSERVATION_ACCEPTED"
            return selected_pusch, selected_mcs, status

        for index, row in enumerate(rows):
            ack = row.get("command_ack_monotonic_ns")
            if ack is None:
                row.update(
                    {
                        "observation_window_start_monotonic_ns": None,
                        "observation_window_end_monotonic_ns": None,
                        "observation_status": "NOT_COMMANDED_OBSOLETE",
                        "observed_pusch_samples": [],
                        "observed_mcs_samples": [],
                        "achieved_pusch_snr_median_db": None,
                        "achieved_minus_target_db": None,
                        "command_validity_covered": False,
                        "next_command_send_monotonic_ns": None,
                        "nominal_observation_window_start_monotonic_ns": None,
                        "nominal_observation_window_end_monotonic_ns": None,
                        "nominal_observation_status": "NOT_COMMANDED_OBSOLETE",
                        "nominal_pusch_count": 0,
                        "nominal_mcs_count": 0,
                        "nominal_achieved_pusch_snr_median_db": None,
                        "nominal_achieved_minus_target_db": None,
                        "nominal_covered": False,
                    }
                )
                continue
            start = int(ack) + guard_ns
            if next_send_by_index[index] is not None:
                corrected_end = int(next_send_by_index[index])
            else:
                corrected_end = restore_send_ns
            nominal_end = int(row["interval_end_monotonic_ns"])
            if next_send_by_index[index] is not None:
                nominal_end = min(nominal_end, int(next_send_by_index[index]))

            selected_pusch, selected_mcs, status = classify(start, corrected_end)
            nominal_pusch, nominal_mcs, nominal_status = classify(start, nominal_end)

            pusch_values = [float(item["snr_db"]) for item in selected_pusch]
            achieved = statistics.median(pusch_values) if status == "OBSERVATION_ACCEPTED" else None
            nominal_pusch_values = [float(item["snr_db"]) for item in nominal_pusch]
            nominal_achieved = statistics.median(nominal_pusch_values) if nominal_status == "OBSERVATION_ACCEPTED" else None
            row.update(
                {
                    "observation_window_start_monotonic_ns": start,
                    "observation_window_end_monotonic_ns": corrected_end,
                    "observation_status": status,
                    "observed_pusch_samples": [
                        {
                            "monotonic_ns": int(item["mono_ns"]),
                            "frame": int(item["frame"]),
                            "slot": int(item["slot"]),
                            "snr_db": float(item["snr_db"]),
                        }
                        for item in selected_pusch
                    ],
                    "observed_mcs_samples": [
                        {
                            "monotonic_ns": int(item["mono_ns"]),
                            "frame": int(item["frame"]),
                            "slot": int(item["slot"]),
                            "mcs_table": int(item["mcs_table"]),
                            "final_mcs": int(item["final_mcs"]),
                            "force_ul_mcs": int(item["force_ul_mcs"]),
                        }
                        for item in selected_mcs
                    ],
                    "achieved_pusch_snr_median_db": achieved,
                    "achieved_minus_target_db": achieved - float(row["target_snr_db"]) if achieved is not None else None,
                    "command_validity_covered": status == "OBSERVATION_ACCEPTED",
                    "next_command_send_monotonic_ns": (
                        int(next_send_by_index[index]) if next_send_by_index[index] is not None else restore_send_ns
                    ),
                    "nominal_observation_window_start_monotonic_ns": start,
                    "nominal_observation_window_end_monotonic_ns": nominal_end,
                    "nominal_observation_status": nominal_status,
                    "nominal_pusch_count": len(nominal_pusch),
                    "nominal_mcs_count": len(nominal_mcs),
                    "nominal_achieved_pusch_snr_median_db": nominal_achieved,
                    "nominal_achieved_minus_target_db": (
                        nominal_achieved - float(row["target_snr_db"]) if nominal_achieved is not None else None
                    ),
                    "nominal_covered": nominal_status == "OBSERVATION_ACCEPTED",
                }
            )
        return observed_rntis, scheduler_ok

    def replay_probe_qualification(
        self, model_index: int, steps: int, target_snr_db: float
    ) -> tuple[list[dict[str, Any]], list[int], bool]:
        """Repeat one constant high-SNR command for `steps` ticks using the

        unchanged 100-ms command/ACK machinery. This is a mechanical
        instrumentation gate, not a traffic-rate search: it does not test
        alternative rates, packet sizes, or pacing, and it is never retried
        with an increased probe on failure.
        """

        self.restore_and_readback(model_index)
        self.wait_until(time.monotonic_ns() + int(float(self.phase14b["replay"]["profile_clean_lead_s"]) * 1e9))
        self.health(force_topology=True)
        period_ns = int(self.phase14b["replay"]["sample_period_ms"]) * 1_000_000
        granularity = float(self.phase14b["replay"]["command_granularity_db"])
        mapped = float(replay.inverse_interpolate(float(target_snr_db), self.mapping, granularity))
        anchor = time.monotonic_ns() + period_ns
        rows: list[dict[str, Any]] = []
        previous_send: int | None = None
        for step in range(steps):
            scheduled = anchor + step * period_ns
            base = {
                "step_index": step,
                "state_index": 0,
                "target_snr_db": float(target_snr_db),
                "quantized_rfsim_command_db": mapped,
                "scheduled_monotonic_ns": scheduled,
                "interval_end_monotonic_ns": scheduled + period_ns,
            }
            decision = plan_scheduler_action(
                scheduled_ns=scheduled,
                period_ns=period_ns,
                now_ns=time.monotonic_ns(),
                previous_send_ns=previous_send,
            )
            eligible = decision["eligible_send_ns"]
            if eligible is None:
                rows.append(
                    {
                        **base,
                        "command_status": "SKIP_OBSOLETE_NEVER_BURST",
                        "command": None,
                        "command_send_monotonic_ns": None,
                        "command_send_wall_ns": None,
                        "command_ack_monotonic_ns": None,
                        "command_ack_wall_ns": None,
                        "command_ack_latency_ms": None,
                        "command_response_sha256": None,
                    }
                )
                continue
            self.wait_until(int(eligible))
            if time.monotonic_ns() >= int(decision["interval_end_ns"]):
                rows.append(
                    {
                        **base,
                        "command_status": "SKIP_OBSOLETE_NEVER_BURST",
                        "command": None,
                        "command_send_monotonic_ns": None,
                        "command_send_wall_ns": None,
                        "command_ack_monotonic_ns": None,
                        "command_ack_wall_ns": None,
                        "command_ack_latency_ms": None,
                        "command_response_sha256": None,
                    }
                )
                continue
            command = self.send_target(model_index, mapped)
            previous_send = int(command["command_send_monotonic_ns"])
            if int(command["command_ack_monotonic_ns"]) >= int(decision["interval_end_ns"]):
                command["command_status"] = "ACK_LATE"
            rows.append({**base, **command})
        self.wait_until(anchor + steps * period_ns)
        self.restore_and_readback(model_index)
        self.wait_until(time.monotonic_ns() + 20_000_000)
        observed_rntis, scheduler_ok = self._associate_observations(rows)
        return rows, observed_rntis, scheduler_ok

    def run_probe_qualification(self) -> dict[str, Any]:
        """Correction 1 gate: exactly-10-second high-SNR probe qualification.

        Runs its own fresh, independent OAI/RFsim lifecycle before any of the
        four frozen profiles. A failure here restores `noise_power_dB=-50`,
        cleanly tears down, and stops the whole corrected run before the
        four-profile replay begins.
        """

        require(self.durable_output is not None, "durable output is unavailable")
        pq = self.phase14b["probe_qualification"]
        require(
            str(pq["radio_state_leaf"]) == PROBE_QUALIFICATION_RADIO_STATE_LEAF,
            "probe qualification radio-state leaf drift from the allow-listed teardown constant",
        )
        radio_state = self.probe_radio_namespace / str(pq["radio_state_leaf"])
        period_ms = int(self.phase14b["replay"]["sample_period_ms"])
        steps = int(round(float(pq["duration_s"]) * 1000.0 / period_ms))
        require(steps * period_ms == int(float(pq["duration_s"]) * 1000), "probe duration is not an exact multiple of the 100-ms cadence")
        target_snr_db = float(self.phase14b["replay"]["target_upper_db"])

        primary_error = ""
        cleanup_errors: list[str] = []
        rows: list[dict[str, Any]] | None = None
        attached: dict[str, Any] | None = None
        model_index: int | None = None
        traffic_measurement: dict[str, Any] | None = None
        self.restored = False
        self.nonclean_attempted = False
        self.initial_clean_state = None
        self.final_restore_state = None
        try:
            cold_preflight = require_cold_profile_runtime(self.base_radio_config, radio_state)
            require(all(cold_preflight.values()), "probe qualification did not start from a cold radio")
            launch_result = self.launch_radio(radio_state)
            require(launch_result.returncode == 0, "probe qualification launcher did not report success")
            self.initialize_attached_session(radio_state)
            live_preflight = super().preflight()
            attached = live_preflight["attached_radio"]
            self.start_runtime_support()
            self.write_runtime_ownership(radio_state)
            model_index = self.open_clean_actuator()
            super().establish_rnti()
            require(self.current_rnti is not None, "single RNTI was not established")
            rows, observed_rntis, scheduler_ok = self.replay_probe_qualification(model_index, steps, target_snr_db)
            traffic_measurement = self.stop_traffic_and_measure()
            self.health(force_topology=True)
        except BaseException as exc:
            primary_error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.session_started:
                cleanup_errors.extend(self.stop_runtime_support_keep_actuator())
                if self.telnet is not None and model_index is not None:
                    try:
                        self.restore_and_readback(model_index)
                    except BaseException as exc:
                        cleanup_errors.append(f"restore: {type(exc).__name__}: {exc}")
                elif self.nonclean_attempted:
                    cleanup_errors.append("restore: control session unavailable after non-clean command")
                cleanup_errors.extend(self.close_actuator_and_work())
            lifecycle_teardown = teardown_profile_runtime(
                self.base_radio_config,
                radio_state,
                self.probe_radio_namespace,
                attached,
                restore_verified=bool(self.restored) if self.session_started else True,
            )
            cleanup_errors.extend(lifecycle_teardown["errors"])
            self.session_started = False

        record: dict[str, Any] = {
            "schema": "scenesense.splitfusion_phase14b_probe_qualification.v1",
            "duration_s": float(pq["duration_s"]),
            "steps": steps,
            "target_snr_db": target_snr_db,
            "primary_error": primary_error,
            "cleanup_errors": list(cleanup_errors),
            "restore_verified": bool(self.restored),
        }
        if primary_error or cleanup_errors or rows is None:
            record["status"] = "PROBE_QUALIFICATION_FAILED_BEFORE_MEASUREMENT"
            record["all_gates_passed"] = False
            atomic_json(self.durable_output / "probe_qualification.json", record)
            raise Phase14BError(
                primary_error or "; ".join(cleanup_errors) or "probe qualification produced no rows"
            )
        summary, gates = summarize_probe_qualification(
            rows,
            self.phase14b,
            observed_rntis=observed_rntis,
            scheduler_seals_ok=scheduler_ok,
            topology_seal_ok=self.topology_seal_ok,
        )
        passed = all(gates.values())
        record.update(
            {
                "status": "PROBE_QUALIFICATION_PASSED" if passed else "PROBE_QUALIFICATION_FAILED",
                "attached_radio_seal": stable_attached_radio_seal(attached or {}),
                "traffic_measurement": traffic_measurement,
                "summary": summary,
                "gates": gates,
                "all_gates_passed": passed,
                "commands": rows,
            }
        )
        atomic_json(self.durable_output / "probe_qualification.json", record)
        require(passed, f"probe qualification did not clear its gates: {gates}")
        return record

    def replay_profile(
        self,
        model_index: int,
        prepared: Mapping[str, Any],
        manifest_sha256: str,
    ) -> dict[str, Any]:
        self.restore_and_readback(model_index)
        self.wait_until(time.monotonic_ns() + int(float(self.phase14b["replay"]["profile_clean_lead_s"]) * 1e9))
        self.health(force_topology=True)
        period_ns = int(self.phase14b["replay"]["sample_period_ms"]) * 1_000_000
        granularity = float(self.phase14b["replay"]["command_granularity_db"])
        anchor = time.monotonic_ns() + period_ns
        rows: list[dict[str, Any]] = []
        previous_send: int | None = None
        for step, (state_index, target) in enumerate(prepared["prefix"]):
            scheduled = anchor + step * period_ns
            mapped = replay.inverse_interpolate(float(target), self.mapping, granularity)
            require(
                float(self.phase14b["replay"]["measured_mapping_lower_db"])
                <= float(target)
                <= float(self.phase14b["replay"]["measured_mapping_upper_db"]),
                f"target outside measured mapping at {prepared['profile_id']}:{step}",
            )
            base = {
                "step_index": step,
                "state_index": int(state_index),
                "target_snr_db": float(target),
                "quantized_rfsim_command_db": float(mapped),
                "scheduled_monotonic_ns": scheduled,
                "interval_end_monotonic_ns": scheduled + period_ns,
            }
            decision = plan_scheduler_action(
                scheduled_ns=scheduled,
                period_ns=period_ns,
                now_ns=time.monotonic_ns(),
                previous_send_ns=previous_send,
            )
            eligible = decision["eligible_send_ns"]
            if eligible is None:
                rows.append(
                    {
                        **base,
                        "command_status": "SKIP_OBSOLETE_NEVER_BURST",
                        "command": None,
                        "command_send_monotonic_ns": None,
                        "command_send_wall_ns": None,
                        "command_ack_monotonic_ns": None,
                        "command_ack_wall_ns": None,
                        "command_ack_latency_ms": None,
                        "command_response_sha256": None,
                    }
                )
                continue
            self.wait_until(int(eligible))
            if time.monotonic_ns() >= int(decision["interval_end_ns"]):
                rows.append(
                    {
                        **base,
                        "command_status": "SKIP_OBSOLETE_NEVER_BURST",
                        "command": None,
                        "command_send_monotonic_ns": None,
                        "command_send_wall_ns": None,
                        "command_ack_monotonic_ns": None,
                        "command_ack_wall_ns": None,
                        "command_ack_latency_ms": None,
                        "command_response_sha256": None,
                    }
                )
                continue
            command = self.send_target(model_index, mapped)
            previous_send = int(command["command_send_monotonic_ns"])
            if int(command["command_ack_monotonic_ns"]) >= int(decision["interval_end_ns"]):
                command["command_status"] = "ACK_LATE"
            rows.append({**base, **command})
        self.wait_until(anchor + len(prepared["prefix"]) * period_ns)
        self.restore_and_readback(model_index)
        self.wait_until(time.monotonic_ns() + 20_000_000)
        observed_rntis, scheduler_ok = self._associate_observations(rows)
        traffic_measurement = self.stop_traffic_and_measure()
        self.health(force_topology=True)
        summary, gates, fade = summarize_profile(
            rows,
            prepared,
            self.phase14b,
            observed_rntis=observed_rntis,
            scheduler_seals_ok=scheduler_ok,
            topology_seal_ok=self.topology_seal_ok,
        )
        payload = {
            "schema": "scenesense.splitfusion_phase14b_profile_payload.v1",
            "status": "PROFILE_REPLAY_ACCEPTED" if all(gates.values()) else "PROFILE_REPLAY_REJECTED",
            "run_manifest_sha256": manifest_sha256,
            "profile_id": prepared["profile_id"],
            "trace_id": prepared["trace_id"],
            "seed": int(prepared["seed"]),
            "trace_sha256": prepared["trace_sha256"],
            "sample_4200_continuation": {
                "state_index": int(prepared["continuation"][0]),
                "target_snr_db": float(prepared["continuation"][1]),
                "same_rng_and_markov_state": True,
                "replayed_in_this_4200_sample_qualification": False,
            },
            "traffic_measurement": traffic_measurement,
            "radio_profile_id": RADIO_PROFILE_ID,
            "attached_radio_seal": stable_attached_radio_seal(self.attached_radio or {}),
            "mapping_sha256": self.provenance["mapping_sha256"],
            "observation_alignment": copy.deepcopy(self.phase14b["observation_alignment"]),
            "traffic": copy.deepcopy(self.phase14b["traffic"]),
            "summary": summary,
            "fade_recovery_diagnostic": fade,
            "gates": gates,
            "all_gates_passed": all(gates.values()),
            "commands": rows,
        }
        return payload

    def write_failure(self, error: str, cleanup_errors: Sequence[str]) -> None:
        if self.durable_output is None:
            return
        directory = self.durable_output / "failures"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"failure_{time.time_ns()}.json"
        atomic_json(
            target,
            {
                "schema": "scenesense.splitfusion_phase14b_failure.v1",
                "error": error,
                "restore_verified": self.restored,
                "restore_readback": self.final_restore_state,
                "cleanup_errors": list(cleanup_errors),
            },
        )

    def finalize(self, payloads: Sequence[Mapping[str, Any]], cleanup: Mapping[str, Any]) -> None:
        require(self.durable_output is not None, "durable output is unavailable")
        require([row["profile_id"] for row in payloads] == list(PROFILE_ORDER), "final profile order drift")
        require(
            all(row["status"] in {"PROFILE_REPLAY_ACCEPTED", "PROFILE_REPLAY_REJECTED"} for row in payloads),
            "finalization requires four complete profiles",
        )
        summaries = [dict(row["summary"]) for row in payloads]
        require(sum(int(row["targets"]) for row in summaries) == 16800, "final target count is not 16,800")
        all_profiles_passed = all(bool(row["all_gates_passed"]) for row in payloads)
        qualification = {
            "schema": SCHEMA,
            "status": "FOUR_PROFILE_REPLAY_COMPLETE",
            "radio_profile_id": RADIO_PROFILE_ID,
            "mapping_sha256": self.provenance["mapping_sha256"],
            "profile_order": list(PROFILE_ORDER),
            "profile_count": 4,
            "targets_per_profile": 4200,
            "total_scheduled_targets": 16800,
            "all_profiles_passed": all_profiles_passed,
            "mapping_qualified_for_16_cell_review": all_profiles_passed,
            "pilot_16_authorized": False,
            "campaign_288_authorized": False,
            "clean_restore_verified": cleanup["restore_verified"],
            "cleanup_clean": cleanup["cleanup_clean"],
            "independent_radio_lifecycle_count": int(cleanup["independent_radio_lifecycle_count"]),
            "profile_summaries": summaries,
        }
        atomic_json(self.durable_output / "qualification.json", qualification)
        fields = (
            "profile_id",
            "trace_id",
            "trace_sha256",
            "targets",
            "commands_acknowledged",
            "commands_skipped_obsolete",
            "command_ack_ratio",
            "command_ack_latency_ms_p50",
            "command_ack_latency_ms_p95",
            "command_ack_latency_ms_max",
            "accepted_observation_windows",
            "empty_observation_windows",
            "underpopulated_observation_windows",
            "observation_coverage",
            "command_validity_coverage",
            "nominal_schedule_coverage",
            "nominal_accepted_observation_windows",
            "tracking_mae_db",
            "bias_achieved_minus_target_db",
            "tracking_p95_absolute_error_db",
            "observed_rntis",
            "all_gates_passed",
        )
        atomic_csv(self.durable_output / "profile_summary.csv", fields, summaries)
        lines = [
            "# SplitFusion Phase-14B corrected four-profile replay qualification",
            "",
            f"- Status: `{qualification['status']}`",
            f"- Radio: `{RADIO_PROFILE_ID}`",
            f"- Phase-14A mapping SHA-256: `{self.provenance['mapping_sha256']}`",
            f"- Original (uncorrected) Phase-14B qualification SHA-256: `{self.provenance['phase14b_original_qualification_sha256']}`",
            f"- Observation-coverage audit report SHA-256: `{self.provenance['observation_coverage_audit_report_sha256']}`",
            "- Old 40-MHz replay used as evidence: `false`",
            "- Original uncorrected Phase-14B replay used as evidence: `false` (it remains a failed result, unaltered).",
            "- Commands use absolute 100-ms deadlines and `SKIP_OBSOLETE_NEVER_BURST`.",
            "- Correction 1 (traffic): one fixed 100-Hz/1,200-byte lightweight UDP probe. This is measurement",
            "  instrumentation only -- Phase-14B does not measure SplitFusion application throughput or packet",
            "  delivery, and the probe is never counted as SplitFusion traffic or campaign throughput.",
            "- Correction 2 (window): `command_validity_coverage` uses ACK+15ms through the actual next command's",
            "  send time (or the clean-restoration command's send time for the final command); it is the coverage",
            "  used for corrected mapping acceptance. `nominal_schedule_coverage` reproduces the original ACK+15ms",
            "  through nominal-interval-end formula for direct comparison with the failed original replay and never",
            "  drives acceptance.",
            "- The 4,200-sample reference is not a production runtime cap; continuation remains on the same RNG/Markov state.",
            "- The 16-cell pilot and 288-cell campaign remain unauthorized.",
            "",
            "| Profile | ACKed | command_validity_coverage | nominal_schedule_coverage | MAE (dB) | Bias (dB) | P95 abs (dB) | Pass |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in summaries:
            def metric(value: Any) -> str:
                return "NA" if value is None else f"{float(value):.3f}"

            lines.append(
                f"| `{row['profile_id']}` | {row['commands_acknowledged']}/{row['targets']} | "
                f"{100 * float(row['command_validity_coverage']):.2f}% | "
                f"{100 * float(row['nominal_schedule_coverage']):.2f}% | "
                f"{metric(row['tracking_mae_db'])} | "
                f"{metric(row['bias_achieved_minus_target_db'])} | "
                f"{metric(row['tracking_p95_absolute_error_db'])} | "
                f"{row['all_gates_passed']} |"
            )
        lines.extend(
            [
                "",
                f"- Final `noise_power_dB=-50` read-back: `{cleanup['restore_verified']}`",
                f"- Runner-owned cleanup: `{cleanup['cleanup_clean']}`",
                "",
            ]
        )
        atomic_text(self.durable_output / "REPORT.md", "\n".join(lines))
        outputs = []
        for path in sorted(self.durable_output.rglob("*")):
            if path.is_file() and path.name not in {"artifact_manifest.json", SUCCESS_TERMINAL}:
                outputs.append(
                    {
                        "path": str(path.relative_to(self.durable_output)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        artifact_manifest = {
            "schema": "scenesense.splitfusion_phase14b_artifact_manifest.v1",
            "status": "FOUR_PROFILE_REPLAY_COMPLETE",
            "raw_tracer_or_traffic_logs_retained": False,
            "outputs": outputs,
        }
        atomic_json(self.durable_output / "artifact_manifest.json", artifact_manifest)
        atomic_json(
            self.durable_output / SUCCESS_TERMINAL,
            {
                "status": SUCCESS_TERMINAL,
                "qualification_sha256": sha256_file(self.durable_output / "qualification.json"),
                "artifact_manifest_sha256": sha256_file(self.durable_output / "artifact_manifest.json"),
            },
        )

    def run(self) -> int:
        first_radio_state = Path(self.lifecycle_plan[0]["radio_state_path"])
        pq = self.phase14b["probe_qualification"]
        probe_radio_state = self.probe_radio_namespace / str(pq["radio_state_leaf"])
        if not self.resume_run:
            require(not self.radio_namespace.exists(), f"create-only radio namespace already exists: {self.radio_namespace}")
            require(not self.probe_radio_namespace.exists(), f"create-only probe radio namespace already exists: {self.probe_radio_namespace}")
            require_cold_profile_runtime(self.base_radio_config, first_radio_state)
            require_cold_profile_runtime(self.base_radio_config, probe_radio_state)
        output, created = create_or_resume_output(self.phase14b, self.output_value, self.resume_run)
        self.durable_output = output
        self.output = output
        manifest_path = output / str(self.phase14b["durability"]["immutable_run_manifest"])
        if created:
            manifest = run_manifest_payload(
                output=output,
                config_path=self.phase14b_config_path,
                config=self.phase14b,
                provenance=self.provenance,
                radio_namespace=self.radio_namespace,
            )
            atomic_json(manifest_path, manifest)
        else:
            require(manifest_path.is_file(), "resume requires the immutable run manifest")
            manifest = validate_run_manifest(
                manifest_path,
                output=output,
                config_path=self.phase14b_config_path,
                config=self.phase14b,
                provenance=self.provenance,
                radio_namespace=self.radio_namespace,
            )
        manifest_sha256 = sha256_file(manifest_path)

        probe_qualification_path = output / "probe_qualification.json"
        if probe_qualification_path.is_file():
            existing_probe = load_json(probe_qualification_path)
            require(
                existing_probe.get("status") == "PROBE_QUALIFICATION_PASSED",
                "a previously recorded probe qualification did not pass; this run cannot resume past it",
            )
        else:
            self.run_probe_qualification()

        profile_dir = output / str(self.phase14b["durability"]["profile_record_directory"])
        profile_dir.mkdir(parents=False, exist_ok=True)
        expected_names = {
            f"{index:02d}_{prepared['profile_id']}.json"
            for index, prepared in enumerate(self.prepared)
        }
        for entry in profile_dir.iterdir():
            is_expected = entry.is_file() and entry.name in expected_names
            is_interrupted_temp = entry.is_file() and any(
                entry.name.startswith(f".{name}.tmp-") for name in expected_names
            )
            require(is_expected or is_interrupted_temp, f"unexpected profile-resume artifact: {entry}")
        payloads: dict[str, dict[str, Any]] = {}
        for index, prepared in enumerate(self.prepared):
            path = profile_dir / f"{index:02d}_{prepared['profile_id']}.json"
            if path.exists():
                payloads[prepared["profile_id"]] = load_profile_record(
                    path,
                    manifest_sha256,
                    prepared,
                    self.phase14b,
                    self.mapping,
                )
        expected_scratch = {Path(row["radio_state_path"]) for row in self.lifecycle_plan}
        if self.radio_namespace.exists():
            actual_scratch = set(self.radio_namespace.iterdir())
            require(actual_scratch <= expected_scratch, f"unexpected radio lifecycle scratch: {sorted(map(str, actual_scratch - expected_scratch))}")
            for radio_state in sorted(actual_scratch):
                profile_id = PROFILE_ORDER[int(radio_state.name.split("_", 1)[0])]
                require(profile_id not in payloads, f"completed profile unexpectedly retains runtime scratch: {profile_id}")
                runtime_errors = stop_recorded_runtime_support(radio_state)
                require(not runtime_errors, f"interrupted runtime-support cleanup failed: {runtime_errors}")
                had_live_radio = phase14a.tcp_listening(int(self.base_radio_config["actuator"]["telnet_port"]))
                recovery_restore = restore_interrupted_radio(self.base_radio_config)
                require(not had_live_radio or recovery_restore is not None, "interrupted live radio could not be restored")
                attached_path = radio_state / "ATTACHED_RADIO_STATE.json"
                attached = load_json(attached_path) if attached_path.is_file() else None
                recovery = teardown_profile_runtime(
                    self.base_radio_config,
                    radio_state,
                    self.radio_namespace,
                    attached,
                    restore_verified=not had_live_radio or recovery_restore is not None,
                )
                require(not recovery["errors"], f"interrupted profile teardown failed: {recovery['errors']}")

        for index, prepared in enumerate(self.prepared):
            profile_id = str(prepared["profile_id"])
            if profile_id in payloads:
                continue
            path = profile_dir / f"{index:02d}_{profile_id}.json"
            require(not path.exists(), f"profile record unexpectedly exists: {path}")
            for partial in profile_dir.glob(f".{path.name}.tmp-*"):
                partial.unlink()
            radio_state = Path(self.lifecycle_plan[index]["radio_state_path"])
            primary_error = ""
            cleanup_errors: list[str] = []
            payload: dict[str, Any] | None = None
            attached: dict[str, Any] | None = None
            live_preflight: dict[str, Any] | None = None
            model_index: int | None = None
            launch_result: subprocess.CompletedProcess[str] | None = None
            cold_preflight: dict[str, Any] | None = None
            lifecycle_teardown: dict[str, Any] = {}
            self.restored = False
            self.nonclean_attempted = False
            self.initial_clean_state = None
            self.final_restore_state = None
            try:
                cold_preflight = require_cold_profile_runtime(self.base_radio_config, radio_state)
                launch_result = self.launch_radio(radio_state)
                self.initialize_attached_session(radio_state)
                live_preflight = super().preflight()
                attached = live_preflight["attached_radio"]
                self.start_runtime_support()
                self.write_runtime_ownership(radio_state)
                model_index = self.open_clean_actuator()
                super().establish_rnti()
                require(self.current_rnti is not None, "single RNTI was not established")
                payload = self.replay_profile(model_index, prepared, manifest_sha256)
            except BaseException as exc:
                primary_error = f"{type(exc).__name__}: {exc}"
            finally:
                if self.session_started:
                    cleanup_errors.extend(self.stop_runtime_support_keep_actuator())
                    if self.telnet is not None and model_index is not None:
                        try:
                            self.restore_and_readback(model_index)
                        except BaseException as exc:
                            cleanup_errors.append(f"restore: {type(exc).__name__}: {exc}")
                    elif self.nonclean_attempted:
                        cleanup_errors.append("restore: control session unavailable after non-clean command")
                    cleanup_errors.extend(self.close_actuator_and_work())
                lifecycle_teardown = teardown_profile_runtime(
                    self.base_radio_config,
                    radio_state,
                    self.radio_namespace,
                    attached,
                    restore_verified=bool(self.restored) if self.session_started else True,
                )
                cleanup_errors.extend(lifecycle_teardown["errors"])
                self.session_started = False

            if primary_error or cleanup_errors or payload is None:
                self.write_failure(primary_error or "profile lifecycle cleanup failed", cleanup_errors)
                raise Phase14BError(primary_error or "; ".join(cleanup_errors) or "profile replay produced no payload")
            require(cold_preflight is not None and live_preflight is not None and launch_result is not None, "profile lifecycle evidence is incomplete")
            require(self.initial_clean_state is not None, "initial clean RFsim read-back is unavailable")
            require(self.final_restore_state is not None and self.restored, "final clean RFsim read-back is unavailable")
            lifecycle_teardown["traffic_and_telemetry_stopped"] = True
            lifecycle_teardown["all_lifecycle_gates_passed"] = not cleanup_errors and self.restored
            payload["radio_lifecycle"] = {
                "profile_index": index,
                "profile_id": profile_id,
                "lifecycle_id": str(uuid.uuid4()),
                "fresh_oai_rfsim_start": True,
                "fresh_gnb_and_ue_topology": True,
                "fresh_attachment": True,
                "fresh_traffic_and_parser_state": True,
                "counters_sequence_timing_and_parser_state_reset": True,
                "generator_start_index": 0,
                "settings_executed": 4200,
                "startup_cold_preflight": cold_preflight,
                "launcher_stdout_sha256": sha256_bytes(launch_result.stdout.encode("utf-8")),
                "launcher_stderr_sha256": sha256_bytes(launch_result.stderr.encode("utf-8")),
                "initial_noise_power_readback": self.initial_clean_state,
                "final_noise_power_readback": self.final_restore_state,
                "attached_radio_seal": stable_attached_radio_seal(attached or {}),
                "radio_reconciliation_status": live_preflight["reconciliation"]["status"],
                "teardown": lifecycle_teardown,
                "all_lifecycle_gates_passed": lifecycle_teardown["all_lifecycle_gates_passed"],
            }
            require(payload["radio_lifecycle"]["all_lifecycle_gates_passed"] is True, f"{profile_id} lifecycle gates failed")
            atomic_json(path, wrap_profile_record(payload))
            payloads[profile_id] = payload

        ordered_payloads = [payloads[profile] for profile in PROFILE_ORDER if profile in payloads]
        require(len(ordered_payloads) == 4, "not all four complete profile records are present")
        for index, prepared in enumerate(self.prepared):
            path = profile_dir / f"{index:02d}_{prepared['profile_id']}.json"
            load_profile_record(
                path,
                manifest_sha256,
                prepared,
                self.phase14b,
                self.mapping,
            )
        comparable_seals = []
        for payload in ordered_payloads:
            seal = copy.deepcopy(payload["attached_radio_seal"])
            seal.pop("gnb_command_sha256")
            seal.pop("ue_command_sha256")
            comparable_seals.append(seal)
        seals = {sha256_bytes(canonical_bytes(seal)) for seal in comparable_seals}
        require(len(seals) == 1, "fresh profile radio topology/configuration seals disagree")
        cleanup = {
            "schema": "scenesense.splitfusion_phase14b_cleanup.v1",
            "restore_target_noise_power_db": -50.0,
            "restore_verified": all(payload["radio_lifecycle"]["final_noise_power_readback"]["verified"] for payload in ordered_payloads),
            "runner_owned_processes_stopped": True,
            "attached_oai_processes_stopped": True,
            "ue_tunnels_removed": True,
            "oai_core_containers_stopped": True,
            "temporary_raw_tracer_and_traffic_logs_removed": True,
            "independent_radio_lifecycle_count": 4,
            "cleanup_clean": True,
            "errors": [],
        }
        require_cold_profile_runtime(self.base_radio_config, first_radio_state)
        atomic_json(output / "cleanup_report.json", cleanup)
        self.finalize(ordered_payloads, cleanup)
        print(json.dumps({"status": SUCCESS_TERMINAL, "output": str(output)}, sort_keys=True))
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute")
    parser.add_argument("--output", required=True)
    parser.add_argument("--radio-state", type=Path, required=True, help="create-only lifecycle scratch namespace")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    previous: dict[signal.Signals, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        raise Phase14BError(f"received termination signal {signal.Signals(signum).name}")

    for caught in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[caught] = signal.getsignal(caught)
        signal.signal(caught, stop)
    try:
        require(args.execute == EXECUTION_TOKEN, "exact Phase-14B execution token is required")
        return CorrectedFourProfileReplay(
            args.config,
            args.output,
            args.radio_state,
            resume_run=bool(args.resume),
        ).run()
    except (
        Phase14BError,
        phase14a.Phase14AError,
        n2.SmokeFailure,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"PHASE14B_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        for caught, handler in previous.items():
            signal.signal(caught, handler)


if __name__ == "__main__":
    raise SystemExit(main())
