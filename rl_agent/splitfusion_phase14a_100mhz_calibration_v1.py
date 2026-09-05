#!/usr/bin/env python3
"""Phase-14A 100-MHz radio binding and attached-state RFsim calibration.

Offline reconciliation and radio-config materialization never start a process.
Calibration requires a separately launched, hash-bound, single-UE OAI state
and changes only rfsimu_channel_ue0.noise_power_dB. It does not replay or alter
the four frozen network profiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent import oai_target_snr_replay_pilot_v1 as replay  # noqa: E402
from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402
from rl_agent import ue_n3_oai_ul_command_calibration_v1 as legacy_calibration  # noqa: E402
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry  # noqa: E402


DEFAULT_CONFIG = ROOT / "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json"
DEFAULT_BINDING = ROOT / "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json"
RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
LEGACY_MAPPING_ID = "OAI_N78_40MHZ_106PRB_7D2U_LEGACY"
CALIBRATION_TOKEN = "SPLITFUSION_PHASE14A_100MHZ_CALIBRATION"
RADIO_ATTACH_TERMINAL = "SPLITFUSION_OAI_100MHZ_4D5U_ATTACHED"
CALIBRATION_TERMINAL = "SPLITFUSION_PHASE14A_100MHZ_CALIBRATION_CAPTURE_COMPLETE_PENDING_REPLAY"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Phase14AError(RuntimeError):
    """A Phase-14A identity, measurement, or cleanup gate failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase14AError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def repo_path(value: str, *, strict: bool = True) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else ROOT / raw
    path = candidate.resolve(strict=strict)
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise Phase14AError(f"path escapes repository root: {value}") from exc
    return path


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def create_only_directory(value: str | Path, expected_root: Path) -> Path:
    experiments_root = (ROOT / "experiments").resolve(strict=True)
    root = expected_root.resolve(strict=False)
    try:
        root.relative_to(experiments_root)
    except ValueError as exc:
        raise Phase14AError(f"configured output root is outside {experiments_root}: {root}") from exc
    raw = Path(value)
    candidate = (raw if raw.is_absolute() else ROOT / raw).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Phase14AError(f"output path is outside {root}: {candidate}") from exc
    require(not candidate.exists(), f"create-only output already exists: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.mkdir(parents=False, exist_ok=False)
    created = candidate.resolve(strict=True)
    try:
        created.relative_to(root)
    except ValueError as exc:
        raise Phase14AError(f"created output resolves outside {root}: {created}") from exc
    return created


def verified_file(record: Mapping[str, Any], *, label: str) -> Path:
    path = repo_path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    require(SHA256_RE.fullmatch(expected) is not None, f"{label} has invalid SHA-256")
    require(sha256_file(path) == expected, f"{label} SHA-256 drift: {path}")
    return path


def config_scalar(text: str, name: str) -> int:
    matches = re.findall(rf"\b{re.escape(name)}\s*=\s*([0-9]+)\s*;", text)
    require(len(matches) == 1, f"expected exactly one {name}, found {len(matches)}")
    return int(matches[0])


def derive_selected_gnb(config: Mapping[str, Any]) -> tuple[Path, str]:
    paths = config["paths"]
    seals = config["source_sha256"]
    source = repo_path(str(paths["gnb_source"]))
    require(sha256_file(source) == seals["gnb_source"], "273-PRB gNB source hash drift")
    text = source.read_text(encoding="utf-8")
    text, downlink_count = re.subn(
        r"(nrofDownlinkSlots\s*=\s*)7(\s*;)", r"\g<1>4\2", text
    )
    text, uplink_count = re.subn(
        r"(nrofUplinkSlots\s*=\s*)2(\s*;)", r"\g<1>5\2", text
    )
    require(
        (downlink_count, uplink_count) == (1, 1),
        f"100-MHz 4D5U derivation count drift: {downlink_count}/{uplink_count}",
    )
    require(
        sha256_bytes(text.encode("utf-8")) == seals["selected_gnb_before_channelmod"],
        "derived 100-MHz 4D5U gNB snapshot disagrees with 144-cell evidence",
    )
    radio = config["radio"]
    expected = {
        "dl_frequencyBand": int(radio["band"]),
        "dl_carrierBandwidth": int(radio["prb"]),
        "ul_carrierBandwidth": int(radio["prb"]),
        "dl_subcarrierSpacing": int(radio["numerology"]),
        "ul_subcarrierSpacing": int(radio["numerology"]),
        "absoluteFrequencySSB": int(radio["absolute_frequency_ssb"]),
        "nrofDownlinkSlots": int(radio["downlink_slots"]),
        "nrofDownlinkSymbols": int(radio["downlink_symbols"]),
        "nrofUplinkSlots": int(radio["uplink_slots"]),
        "nrofUplinkSymbols": int(radio["uplink_symbols"]),
    }
    observed = {name: config_scalar(text, name) for name in expected}
    require(observed == expected, f"derived 100-MHz radio identity mismatch: {observed}")
    return source, text


def validate_subscriber(config: Mapping[str, Any]) -> dict[str, Any]:
    sql_path = repo_path(str(config["paths"]["subscriber_sql"]))
    require(
        sha256_file(sql_path) == config["source_sha256"]["subscriber_sql"],
        "subscriber SQL hash drift",
    )
    radio = config["radio"]
    matches = [
        line
        for line in sql_path.read_text(encoding="utf-8").splitlines()
        if str(radio["imsi"]) in line and str(radio["ue_static_ip"]) in line
    ]
    require(len(matches) == 1, "expected exactly one bound IMSI/static-IP subscriber row")
    five_qi = re.search(r'\\"5qi\\"\s*:\s*([0-9]+)', matches[0])
    require(five_qi is not None, "bound subscriber row has no 5QI")
    require(int(five_qi.group(1)) == int(radio["pdu_session_5qi"]), "subscriber 5QI drift")
    return {
        "imsi": str(radio["imsi"]),
        "ue_static_ip": str(radio["ue_static_ip"]),
        "pdu_session_5qi": int(five_qi.group(1)),
        "subscriber_sql_sha256": sha256_file(sql_path),
    }


def validate_ue_and_channel_sources(config: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("ue_source", "channel_source"):
        path = repo_path(str(config["paths"][key]))
        observed = sha256_file(path)
        require(observed == config["source_sha256"][key], f"{key} hash drift")
        result[key] = observed
    ue = repo_path(str(config["paths"]["ue_source"])).read_text(encoding="utf-8")
    imsis = re.findall(r'(?m)^\s*imsi\s*=\s*"([0-9]+)"\s*;', ue)
    uiccs = re.findall(r"(?m)^\s*uicc\d+\s*=\s*\{", ue)
    require(len(uiccs) == 1 and imsis == [str(config["radio"]["imsi"])], "UE source is not the bound single UE")
    return result


def load_generator_module() -> Any:
    path = ROOT / "rl_agent/generate_network_profile_meeting_figures.py"
    spec = importlib.util.spec_from_file_location("phase14a_profile_generator", path)
    require(spec is not None and spec.loader is not None, "profile generator cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_network_profiles(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    network = binding["network_profiles"]
    design_path = verified_file(network["design"], label="network-profile design")
    trace_path = verified_file(network["accepted_prefix"], label="accepted profile prefix")
    runtime_path = verified_file(network["runtime"], label="continuous target-SNR runtime")
    runtime_source = runtime_path.read_text(encoding="utf-8")
    for required_source in (
        "while not stop_event.is_set() and not (args.stop_file and args.stop_file.exists()):",
        "if step < len(prefix):",
        "_state, target_snr = sequence.next_sample()",
        '"command_timing_status": "SKIP_OBSOLETE_NEVER_BURST"',
    ):
        require(required_source in runtime_source, f"continuous runtime behavior missing: {required_source}")
    design = load_json(design_path)
    require(int(network["sample_period_ms"]) == 100, "network period is not 100 ms")
    require(int(network["reference_prefix_samples"]) == 4200, "reference prefix is not 4,200")
    require(network["reference_prefix_is_runtime_cap"] is False, "4,200 was made a runtime cap")
    require(
        network["continuation"] == "CONTINUE_SAME_RNG_AND_MARKOV_STATE_AFTER_SAMPLE_4199",
        "profile continuation contract drift",
    )
    require(network["campaign_cell_start"] == "SAMPLE_ZERO", "campaign cells do not reset to sample zero")
    require(network["stop_condition"] == "EPISODE_OR_CONTROLLER_STOP_ONLY", "profile stop condition drift")
    require(network["catch_up_policy"] == "SKIP_OBSOLETE_NEVER_BURST", "catch-up policy drift")
    require(
        network["forbidden_continuation"] == ["WRAP", "HOLD_FINAL", "RESEED"],
        "forbidden continuation set drift",
    )
    accepted: dict[str, list[dict[str, str]]] = {}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            accepted.setdefault(str(row["profile_id"]), []).append(dict(row))
    by_id = {str(row["profile_id"]): row for row in design["profiles"]}
    generator = load_generator_module()
    audits: list[dict[str, Any]] = []
    for frozen in network["profiles"]:
        profile_id = str(frozen["profile_id"])
        profile = by_id.get(profile_id)
        require(profile is not None, f"profile missing from design: {profile_id}")
        require(int(profile["seed"]) == int(frozen["seed"]), f"{profile_id} seed drift")
        require(profile["transition_matrix"] == frozen["transition_matrix"], f"{profile_id} transition drift")
        sequence = generator.DeterministicTargetSnrSequence(profile, design)
        prefix = [sequence.next_sample() for _ in range(4200)]
        continuation = sequence.next_sample()
        states = np.asarray([row[0] for row in prefix], dtype="<i4")
        targets = np.asarray([row[1] for row in prefix], dtype="<f8")
        digest = hashlib.sha256(states.tobytes() + targets.tobytes()).hexdigest()
        require(digest == frozen["trace_semantic_sha256"], f"{profile_id} frozen prefix hash mismatch")
        rows = sorted(accepted.get(profile_id, []), key=lambda row: int(row["step_index"]))
        require(len(rows) == 4200, f"{profile_id} accepted prefix length drift")
        require([int(row["step_index"]) for row in rows] == list(range(4200)), f"{profile_id} prefix index drift")
        require(
            np.array_equal(states, np.asarray([int(row["state_index"]) for row in rows], dtype="<i4")),
            f"{profile_id} accepted state prefix drift",
        )
        require(
            np.allclose(
                targets,
                np.asarray([float(row["target_snr_db"]) for row in rows], dtype="<f8"),
                rtol=0.0,
                atol=5.1e-7,
            ),
            f"{profile_id} accepted target prefix drift",
        )
        require(continuation != prefix[0], f"{profile_id} continuation wrapped/reseeded")
        require(continuation != prefix[-1], f"{profile_id} continuation held final value")
        second = generator.DeterministicTargetSnrSequence(profile, design)
        reference_4201 = [second.next_sample() for _ in range(4201)][-1]
        require(continuation == reference_4201, f"{profile_id} continuation changed RNG/Markov state")
        audits.append(
            {
                "profile_id": profile_id,
                "seed": int(profile["seed"]),
                "trace_semantic_sha256": digest,
                "sample_4200_state": int(continuation[0]),
                "sample_4200_target_snr_db": float(continuation[1]),
                "same_rng_and_markov_state_continued": True,
            }
        )
    require(len(audits) == 4, "network profile count is not four")
    return audits


def require_mapping_identity(profile_id: str) -> None:
    require(profile_id == RADIO_PROFILE_ID, f"mapping radio identity is not {RADIO_PROFILE_ID}: {profile_id}")


def reconcile_contract(config_path: Path, binding_path: Path) -> dict[str, Any]:
    config = load_json(config_path.resolve(strict=True))
    binding = load_json(binding_path.resolve(strict=True))
    require(
        config.get("schema") == "scenesense.splitfusion_phase14a_100mhz_calibration_config.v1",
        "Phase-14A calibration schema drift",
    )
    require(config.get("status") == "IMPLEMENTED_NOT_LIVE_AUTHORIZED", "calibration status drift")
    require(config.get("execution_token") == CALIBRATION_TOKEN, "calibration token drift")
    require(
        binding.get("schema") == "scenesense.splitfusion_phase14a_campaign_binding.v1",
        "Phase-14A binding schema drift",
    )
    require(
        binding.get("status") == "IMPLEMENTATION_READY_PENDING_LIVE_100MHZ_CALIBRATION_AND_REPLAY",
        "Phase-14A binding status overclaims readiness",
    )
    require(binding["campaign_authorization"]["pilot_16_authorized"] is False, "16-cell campaign was authorized")
    require(binding["campaign_authorization"]["campaign_288_authorized"] is False, "288-cell campaign was authorized")
    require(binding["mapping"]["qualified"] is False, "pending 100-MHz mapping was marked qualified")
    require(binding["mapping"]["radio_profile_id"] == RADIO_PROFILE_ID, "pending mapping radio identity drift")
    require(binding["mapping"]["legacy_mapping_id"] == LEGACY_MAPPING_ID, "legacy mapping identity drift")
    verified_file(binding["implementation_terminal"], label="Phase-14A implementation terminal")
    try:
        require_mapping_identity(LEGACY_MAPPING_ID)
    except Phase14AError:
        legacy_rejected = True
    else:
        legacy_rejected = False
    require(legacy_rejected, "legacy 40-MHz mapping was not refused")

    for name, record in binding["phase13c_evidence"]["artifacts"].items():
        verified_file(record, label=f"Phase-13C evidence {name}")
    evidence_commit = str(binding["phase13c_evidence"]["commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", evidence_commit, "HEAD"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    require(ancestor.returncode == 0, "Phase-13C evidence commit is not an ancestor of HEAD")

    for section in ("capacity_evidence", "dispatcher", "frame_context", "map_install"):
        for name, record in binding[section]["artifacts"].items():
            verified_file(record, label=f"{section} {name}")
    for name, record in binding["calibration"]["reuse"].items():
        verified_file(record, label=f"calibration reuse {name}")
    verified_file(binding["calibration"]["runner"], label="Phase-14A calibration runner")
    launcher_path = verified_file(binding["launcher"], label="100-MHz SplitFusion launcher")
    calibration_path = verified_file(binding["calibration"]["config"], label="calibration config")
    require(calibration_path == config_path.resolve(strict=True), "binding selected another calibration config")
    legacy = binding["legacy_rejection"]
    forbidden_paths = [repo_path(value) for value in legacy["forbidden_launcher_paths"]]
    forbidden_hashes = list(legacy["forbidden_launcher_sha256"])
    require(len(forbidden_paths) == len(forbidden_hashes) >= 2, "legacy launcher rejection set is incomplete")
    for path, expected_hash in zip(forbidden_paths, forbidden_hashes):
        require(sha256_file(path) == expected_hash, f"legacy launcher provenance drift: {path}")
    require(launcher_path not in forbidden_paths, "selected launcher is a legacy launcher")
    launcher_source = launcher_path.read_text(encoding="utf-8")
    require("default106" not in launcher_source and "106PRB" not in launcher_source, "new launcher invokes a legacy 106-PRB path")
    require("legacy radio override" in launcher_source, "new launcher lacks legacy override refusal")
    ladder = [float(value) for value in config["calibration"]["commanded_noise_power_db"]]
    require(ladder == [float(value) for value in range(-13, -1)], "calibration command ladder drift")
    require(len(set(ladder)) == len(ladder), "calibration command ladder is not unique")

    _source, selected = derive_selected_gnb(config)
    ue_channel = validate_ue_and_channel_sources(config)
    subscriber = validate_subscriber(config)
    radio = config["radio"]
    require(
        (
            radio["profile_id"],
            int(radio["bandwidth_mhz"]),
            int(radio["prb"]),
            int(radio["numerology"]),
            int(radio["downlink_slots"]),
            int(radio["uplink_slots"]),
            int(radio["ue_count"]),
            radio["ue_static_ip"],
            int(radio["pdu_session_5qi"]),
        )
        == (RADIO_PROFILE_ID, 100, 273, 1, 4, 5, 1, "10.0.0.2", 6),
        "100-MHz radio contract drift",
    )

    registry = SplitActionRegistry.from_runtime_binding(verify_runtime_artifacts=False)
    require(len(registry.profiles) == 72, "dispatcher registry is not the immutable 72-action catalog")
    frame_binding = load_json(verified_file(binding["frame_context"]["binding"], label="frame-context binding"))
    require(
        frame_binding["wire"]["magic"] == "SFD1"
        and int(frame_binding["wire"]["context_envelope_version"]) == 2,
        "SFD1-v2 frame-context binding drift",
    )
    require(
        binding["map_install"]["ack_status"] == "ACK_INSTALLED"
        and binding["map_install"]["ack_semantics"]
        == "DECODED_RESULT_ACCEPTED_AND_INSTALLED_UNDER_MAP_LOCK",
        "map-install ACK semantics drift",
    )
    profiles = audit_network_profiles(binding)
    return {
        "status": "PHASE14A_CPU_RECONCILIATION_PASSED",
        "radio_profile_id": RADIO_PROFILE_ID,
        "radio": {
            "band": 78,
            "bandwidth_mhz": 100,
            "prb": 273,
            "numerology": 1,
            "tdd": "4D5U",
            "ue_count": 1,
            "ue_static_ip": "10.0.0.2",
            "pdu_session_5qi": 6,
            "selected_gnb_snapshot_sha256": sha256_bytes(selected.encode("utf-8")),
            "ue_source_sha256": ue_channel["ue_source"],
            "subscriber": subscriber,
        },
        "launcher": {"path": str(launcher_path.relative_to(ROOT)), "sha256": sha256_file(launcher_path)},
        "phase13c_evidence_commit_is_ancestor": True,
        "dispatcher_action_count": len(registry.profiles),
        "sfd1_version": 2,
        "ack_status": "ACK_INSTALLED",
        "legacy_mapping_id": LEGACY_MAPPING_ID,
        "legacy_mapping_rejected": legacy_rejected,
        "mapping_status": binding["mapping"]["status"],
        "network_profiles": profiles,
        "reference_prefix_is_runtime_cap": False,
        "profile_replay_performed": False,
        "campaigns_authorized": False,
    }


def materialize_radio(config_path: Path, binding_path: Path, output_value: str) -> dict[str, Any]:
    audit = reconcile_contract(config_path, binding_path)
    config = load_json(config_path)
    output_root = repo_path(str(config["paths"]["radio_state_root"]), strict=False)
    output = create_only_directory(output_value, output_root)
    _source, selected = derive_selected_gnb(config)
    ue_source = repo_path(str(config["paths"]["ue_source"]))
    channel_source = repo_path(str(config["paths"]["channel_source"]))
    ue_text = ue_source.read_text(encoding="utf-8")
    channel = channel_source.read_text(encoding="utf-8")
    channel, replacements = re.subn(
        r"noise_power_dB\s*=\s*[-+0-9.eE]+;", "noise_power_dB = -50;", channel
    )
    require(replacements == 3, f"expected three clean channel values, found {replacements}")
    require("noise_power_dBFS" not in channel, "global noise_power_dBFS is forbidden")
    include = '@include "channelmod_rfsimu_LEO_satellite.conf"'
    require(ue_text.count(include) == 1, "UE source channel include drift")
    gnb_path = output / "effective_gnb_100mhz_4d5u_clean_minus50.conf"
    ue_path = output / "effective_ue_100mhz_clean_minus50.conf"
    atomic_text(gnb_path, selected + "\n\n" + channel + "\n")
    atomic_text(ue_path, ue_text.replace(include, channel))
    materialized = {
        "schema": "scenesense.splitfusion_phase14a_radio_materialization.v1",
        "status": "MATERIALIZED_NOT_LAUNCHED",
        "radio_profile_id": RADIO_PROFILE_ID,
        "source_gnb_sha256": config["source_sha256"]["gnb_source"],
        "selected_gnb_before_channelmod_sha256": config["source_sha256"]["selected_gnb_before_channelmod"],
        "source_ue_sha256": config["source_sha256"]["ue_source"],
        "source_channel_sha256": config["source_sha256"]["channel_source"],
        "effective_gnb_path": str(gnb_path),
        "effective_gnb_sha256": sha256_file(gnb_path),
        "effective_ue_path": str(ue_path),
        "effective_ue_sha256": sha256_file(ue_path),
        "clean_noise_power_db": -50.0,
        "cpu_reconciliation": audit["status"],
    }
    atomic_json(output / "radio_materialization.json", materialized)
    print(json.dumps(materialized, sort_keys=True))
    return materialized


def process_rows(name: str) -> list[tuple[int, str]]:
    completed = subprocess.run(
        ["pgrep", "-a", "-x", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise Phase14AError(f"process query failed for {name}: {completed.stderr.strip()}")
    rows: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        pid_text, separator, command = line.strip().partition(" ")
        require(separator != "" and pid_text.isdigit(), f"malformed process row: {line!r}")
        rows.append((int(pid_text), command))
    return rows


def tunnel_ip(interface: str) -> str | None:
    completed = subprocess.run(
        ["ip", "-j", "-4", "addr", "show", "dev", interface],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        values = [
            str(info["local"])
            for row in json.loads(completed.stdout)
            for info in row.get("addr_info", [])
            if info.get("family") == "inet" and info.get("local")
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return values[0] if len(values) == 1 else None


def command_has_pair(command: str, flag: str, value: str) -> bool:
    fields = command.split()
    return any(left == flag and right == value for left, right in zip(fields, fields[1:]))


def attached_radio_snapshot(
    config_path: Path,
    binding_path: Path,
    radio_state: Path,
) -> dict[str, Any]:
    audit = reconcile_contract(config_path, binding_path)
    config = load_json(config_path)
    state = radio_state.resolve(strict=True)
    expected_root = repo_path(str(config["paths"]["radio_state_root"]))
    try:
        state.relative_to(expected_root)
    except ValueError as exc:
        raise Phase14AError(f"radio state is outside {expected_root}: {state}") from exc
    materialization_path = state / "radio_materialization.json"
    materialization = load_json(materialization_path)
    require(materialization["radio_profile_id"] == RADIO_PROFILE_ID, "materialized radio identity drift")
    for prefix in ("gnb", "ue"):
        path = Path(str(materialization[f"effective_{prefix}_path"])).resolve(strict=True)
        require(path.parent == state, f"effective {prefix} config escaped radio-state directory")
        require(
            sha256_file(path) == materialization[f"effective_{prefix}_sha256"],
            f"effective {prefix} config hash drift",
        )
    gnb = process_rows("nr-softmodem")
    ue = process_rows("nr-uesoftmodem")
    require(len(gnb) == 1 and len(ue) == 1, f"expected one gNB/UE process, observed {gnb}/{ue}")
    radio = config["radio"]
    gnb_cmd = gnb[0][1]
    ue_cmd = ue[0][1]
    require(str(materialization["effective_gnb_path"]) in gnb_cmd, "gNB did not use materialized 100-MHz config")
    require("--rfsim" in gnb_cmd and "--rfsimulator.[0].options chanmod" in gnb_cmd, "gNB RFsim/chanmod binding missing")
    require("--telnetsrv" in gnb_cmd, "gNB calibration control server is absent")
    require(str(materialization["effective_ue_path"]) in ue_cmd, "UE did not use materialized config")
    for flag, value in (
        ("-r", str(radio["prb"])),
        ("--numerology", str(radio["numerology"])),
        ("--band", str(radio["band"])),
        ("-C", str(radio["ue_frequency_hz"])),
        ("--ssb", str(radio["ue_ssb"])),
    ):
        require(command_has_pair(ue_cmd, flag, value), f"UE command lacks {flag} {value}")
    observed_ip = tunnel_ip(str(radio["ue_interface"]))
    require(observed_ip == radio["ue_static_ip"], f"UE tunnel identity mismatch: {observed_ip}")
    return {
        "schema": "scenesense.splitfusion_phase14a_attached_radio_state.v1",
        "status": "ATTACHED_STABLE_100MHZ_4D5U_ONE_UE",
        "radio_profile_id": RADIO_PROFILE_ID,
        "radio_materialization_path": str(materialization_path),
        "radio_materialization_sha256": sha256_file(materialization_path),
        "effective_gnb_sha256": materialization["effective_gnb_sha256"],
        "effective_ue_sha256": materialization["effective_ue_sha256"],
        "gnb_pid": gnb[0][0],
        "gnb_command_sha256": sha256_bytes(gnb_cmd.encode()),
        "ue_pid": ue[0][0],
        "ue_command_sha256": sha256_bytes(ue_cmd.encode()),
        "ue_interface": radio["ue_interface"],
        "ue_static_ip": observed_ip,
        "pdu_session_5qi": int(radio["pdu_session_5qi"]),
        "launcher_sha256": audit["launcher"]["sha256"],
    }


def record_attached_radio(
    config_path: Path, binding_path: Path, radio_state: Path
) -> dict[str, Any]:
    snapshot = attached_radio_snapshot(config_path, binding_path, radio_state)
    target = radio_state.resolve(strict=True) / "ATTACHED_RADIO_STATE.json"
    require(not target.exists(), f"attached-state record already exists: {target}")
    atomic_json(target, snapshot)
    terminal = radio_state.resolve(strict=True) / RADIO_ATTACH_TERMINAL
    atomic_text(terminal, RADIO_ATTACH_TERMINAL + "\n")
    print(json.dumps(snapshot, sort_keys=True))
    return snapshot


def validate_recorded_radio(
    config_path: Path, binding_path: Path, radio_state: Path
) -> dict[str, Any]:
    current = attached_radio_snapshot(config_path, binding_path, radio_state)
    recorded_path = radio_state.resolve(strict=True) / "ATTACHED_RADIO_STATE.json"
    terminal = radio_state.resolve(strict=True) / RADIO_ATTACH_TERMINAL
    require(recorded_path.is_file(), "attached radio-state record is missing")
    require(terminal.read_text(encoding="utf-8").strip() == RADIO_ATTACH_TERMINAL, "radio attach terminal drift")
    recorded = load_json(recorded_path)
    require(recorded == current, "attached radio state changed after launcher verification")
    return recorded


def carla_absent() -> None:
    completed = subprocess.run(
        ["ps", "-eo", "comm=,args="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.returncode == 0, "CARLA process check failed closed")
    matches = [
        line.strip()
        for line in completed.stdout.splitlines()
        if any(marker in line.lower() for marker in ("carlaue4", "carlaunreal"))
    ]
    require(not matches, f"CARLA is active: {matches}")


def tcp_listening(port: int) -> bool:
    completed = subprocess.run(
        ["ss", "-H", "-ltn"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.returncode == 0, f"TCP listener query failed: {completed.stderr.strip()}")
    return any(re.search(rf":{int(port)}\b", line) for line in completed.stdout.splitlines())


class AttachedCalibration:
    """Measure a fixed RFsim ladder without owning or restarting OAI."""

    def __init__(
        self,
        config_path: Path,
        binding_path: Path,
        radio_state: Path,
        output_value: str,
    ) -> None:
        self.config_path = config_path.resolve(strict=True)
        self.binding_path = binding_path.resolve(strict=True)
        self.radio_state = radio_state.resolve(strict=True)
        self.config = load_json(self.config_path)
        self.output_value = output_value
        self.output: Path | None = None
        self.telnet: n2.TelnetSession | None = None
        self.live_pusch: n2.LiveCsv | None = None
        self.live_mcs: n2.LiveCsv | None = None
        self.processes: list[n2.ManagedProcess] = []
        self.command_rows: list[dict[str, Any]] = []
        self.anchor_rows: list[dict[str, Any]] = []
        self.current_rnti: int | None = None
        self.nonclean_attempted = False
        self.restored = False

    def spawn(
        self,
        name: str,
        argv: Sequence[str],
        log_name: str,
        *,
        root_owned: bool = False,
    ) -> n2.ManagedProcess:
        assert self.output is not None
        log_path = self.output / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("xb")
        process = subprocess.Popen(
            list(argv),
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        managed = n2.ManagedProcess(name, process, handle, root_owned=root_owned)
        self.processes.append(managed)
        return managed

    def preflight(self) -> dict[str, Any]:
        audit = reconcile_contract(self.config_path, self.binding_path)
        radio = validate_recorded_radio(self.config_path, self.binding_path, self.radio_state)
        carla_absent()
        require(tcp_listening(int(self.config["telemetry"]["gnb_port"])), "gNB T-tracer port is unavailable")
        require(tcp_listening(int(self.config["actuator"]["telnet_port"])), "gNB telnet actuator is unavailable")
        require(n2.port_is_free(int(self.config["telemetry"]["gnb_relay_port"])), "gNB relay port is occupied")
        require(subprocess.run(["sudo", "-n", "true"], check=False).returncode == 0, "noninteractive sudo unavailable")
        return {"reconciliation": audit, "attached_radio": radio}

    def start_telemetry(self) -> None:
        assert self.output is not None
        telemetry = self.config["telemetry"]
        tracer = repo_path("OAI/openairinterface5g/common/utils/T/tracer")
        messages = repo_path(str(self.config["paths"]["t_messages"]))
        relay = int(telemetry["gnb_relay_port"])
        self.spawn(
            "gnb_relay",
            [
                str(tracer / "multi"),
                "-d",
                str(messages),
                "-ip",
                "127.0.0.1",
                "-p",
                str(telemetry["gnb_port"]),
                "-lp",
                str(relay),
            ],
            "logs/gnb_relay.log",
        )
        n2.wait_tcp(relay, 10)
        pusch_fields = (
            "time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
            "txpower_calc", "rbSize", "mcs", "rssi",
        )
        mcs_fields = (
            "time", "rnti", "frame", "slot", "sched_frame", "sched_slot",
            "avg_snr_x10", "mcs_table", "ul_bler_mcs_before", "selected_mcs",
            "pre_phr_mcs", "post_phr_mcs", "final_mcs", "estimated_ul_buffer",
            "sched_ul_bytes", "B", "min_rb", "available_rb_before",
            "available_rb_after", "ph", "pcmax", "rb_size_final", "tbs_final",
            "force_ul_mcs",
        )
        common = [
            str(tracer / "csv"),
            "-d",
            str(messages),
            "-ip",
            "127.0.0.1",
            "-p",
            str(relay),
            "-f",
            "-s",
            ",",
            "-t",
            "time",
        ]
        self.live_pusch = n2.LiveCsv(
            [*common, "GNB_MAC_PUSCH_POWER_CONTROL", *pusch_fields],
            self.output / "ttracer/live_pusch_with_ingest.csv",
        )
        self.live_mcs = n2.LiveCsv(
            [*common, "GNB_MAC_UL_MCS_DECISION", *mcs_fields],
            self.output / "ttracer/live_mcs_with_ingest.csv",
        )

    def start_traffic(self) -> None:
        assert self.output is not None
        traffic = self.config["traffic"]
        radio = self.config["radio"]
        inspect = n2.run_checked(
            [
                "sudo",
                "-n",
                "docker",
                "inspect",
                "-f",
                "{{.State.Pid}}",
                "oai-ext-dn",
            ]
        )
        ext_dn_pid = int(inspect.stdout.strip())
        sink_csv = self.output / "traffic/sink_packets.csv"
        sink_csv.parent.mkdir(parents=True, exist_ok=True)
        self.spawn(
            "calibration_sink",
            [
                "sudo",
                "-n",
                "nsenter",
                "-t",
                str(ext_dn_pid),
                "-n",
                "/usr/bin/python3",
                str(repo_path(str(self.config["paths"]["sink"]))),
                "--bind",
                str(radio["ext_dn_ip"]),
                "--port",
                str(traffic["remote_port"]),
                "--out",
                str(sink_csv),
                "--timeout-s",
                str(traffic["sink_timeout_s"]),
            ],
            "logs/sink.log",
            root_owned=True,
        )
        time.sleep(0.5)
        self.spawn(
            "calibration_sender",
            [
                sys.executable,
                str(repo_path(str(self.config["paths"]["sender"]))),
                "--bind-host",
                str(radio["ue_static_ip"]),
                "--remote-host",
                str(radio["ext_dn_ip"]),
                "--remote-port",
                str(traffic["remote_port"]),
                "--fps",
                str(traffic["fps"]),
                "--frames",
                str(traffic["frames"]),
                "--frame-bytes",
                str(traffic["frame_bytes"]),
                "--chunk-bytes",
                str(traffic["chunk_bytes"]),
                "--idle-before-s",
                "0",
                "--cooldown-s",
                "2",
                "--log-csv",
                str(self.output / "traffic/sender.csv"),
            ],
            "logs/sender.log",
        )

    def health(self) -> None:
        radio = self.config["radio"]
        require(len(process_rows("nr-softmodem")) == 1, "gNB process count changed")
        require(len(process_rows("nr-uesoftmodem")) == 1, "UE process count changed")
        require(tunnel_ip(str(radio["ue_interface"])) == radio["ue_static_ip"], "UE tunnel identity changed")
        for process in self.processes:
            require(process.process.poll() is None, f"runner-owned process exited: {process.name}")
        require(self.live_pusch is not None and self.live_pusch.process.poll() is None, "PUSCH collector exited")
        require(self.live_mcs is not None and self.live_mcs.process.poll() is None, "MCS collector exited")

    def wait(self, duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            self.health()
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def open_actuator(self) -> int:
        assert self.output is not None
        actuator = self.config["actuator"]
        self.telnet = n2.TelnetSession(
            str(actuator["telnet_host"]),
            int(actuator["telnet_port"]),
            float(actuator["response_timeout_s"]),
            int(actuator["max_response_bytes"]),
        )
        response = self.telnet.command("channelmod show current")[-1]
        atomic_text(self.output / "channel_state_before.txt", response)
        models = n2.parse_channel_models(response)
        model = models.get(str(actuator["channel_model_name"]))
        require(model is not None, "registered RFsim UE channel object is absent")
        require(model.get("model_type") == actuator["channel_model_type"], "RFsim model type drift")
        require(model.get("owner") == actuator["channel_model_owner"], "RFsim model owner drift")
        require(math.isclose(float(model.get("path_loss_db", math.nan)), 0.0, abs_tol=1e-6), "RFsim path loss drift")
        require(
            math.isclose(
                float(model.get("noise_power_db", math.nan)),
                float(actuator["clean_restore_noise_power_db"]),
                abs_tol=1e-6,
            ),
            "calibration did not start from noise_power_dB=-50",
        )
        return int(model["model_index"])

    def send_and_verify(self, model_index: int, value: float, purpose: str) -> dict[str, Any]:
        assert self.telnet is not None
        target = f"{value:.12g}"
        self.nonclean_attempted = self.nonclean_attempted or not math.isclose(value, -50.0)
        command = f"channelmod modify {model_index} noise_power_dB {target}"
        sent_mono, sent_wall, ack_mono, ack_wall, response = self.telnet.command(command)
        n2.Runner.validate_modify_response(response, target)
        state = self.telnet.command("channelmod show current")[-1]
        model = n2.parse_channel_models(state).get(str(self.config["actuator"]["channel_model_name"]), {})
        require(math.isclose(float(model.get("noise_power_db", math.nan)), value, abs_tol=1e-6), "post-command RFsim state mismatch")
        row = {
            "purpose": purpose,
            "noise_power_db": value,
            "command": command,
            "send_monotonic_ns": sent_mono,
            "send_wall_ns": sent_wall,
            "ack_monotonic_ns": ack_mono,
            "ack_wall_ns": ack_wall,
            "ack_latency_ms": (ack_mono - sent_mono) / 1e6,
            "response_sha256": sha256_bytes(response.encode()),
            "post_state_sha256": sha256_bytes(state.encode()),
            "status": "ACK_AND_POST_STATE_VALIDATED",
        }
        self.command_rows.append(row)
        return row

    def establish_rnti(self) -> None:
        telemetry = self.config["telemetry"]
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self.health()
            pusch = [
                row
                for row in (
                    legacy_calibration.parse_live_pusch(item)
                    for item in self.live_pusch.snapshot() if self.live_pusch
                )
                if row is not None
            ]
            mcs = [
                row
                for row in (
                    legacy_calibration.parse_live_mcs(item)
                    for item in self.live_mcs.snapshot() if self.live_mcs
                )
                if row is not None
            ]
            rntis = {int(row["rnti"]) for row in [*pusch, *mcs]}
            if (
                len(rntis) == 1
                and len(pusch) >= int(telemetry["minimum_baseline_pusch_samples"])
                and len(mcs) >= int(telemetry["minimum_baseline_mcs_samples"])
            ):
                self.current_rnti = next(iter(rntis))
                return
            time.sleep(0.1)
        raise Phase14AError("stable single-RNTI baseline was not established")

    def measure_anchor(self, model_index: int, index: int, command_db: float) -> dict[str, Any]:
        telemetry = self.config["telemetry"]
        command = self.send_and_verify(model_index, command_db, "CALIBRATION_ANCHOR")
        self.wait(float(telemetry["settle_s"]))
        start_ns = time.monotonic_ns()
        self.wait(float(telemetry["measurement_s"]))
        end_ns = time.monotonic_ns()
        require(self.current_rnti is not None, "current RNTI unavailable")
        tail = legacy_calibration.summarize_tail(
            self.live_pusch.snapshot() if self.live_pusch else [],
            self.live_mcs.snapshot() if self.live_mcs else [],
            start_ns=start_ns,
            end_ns=end_ns,
            expected_rnti=self.current_rnti,
            minimum_pusch=int(telemetry["minimum_anchor_pusch_samples"]),
            minimum_mcs=int(telemetry["minimum_anchor_mcs_samples"]),
            required_mcs_table=int(telemetry["required_mcs_table"]),
            required_force_mcs=int(telemetry["required_force_ul_mcs"]),
        )
        require(tail["status"] == "TAIL_ACCEPTED", f"anchor {index} observation gate failed: {tail}")
        lower = float(tail["achieved_pusch_snr_db_p05"])
        upper = float(tail["achieved_pusch_snr_db_p95"])
        row = {
            "anchor_index": index,
            "applied_noise_power_db": command_db,
            "command_ack_latency_ms": command["ack_latency_ms"],
            "measurement_start_monotonic_ns": start_ns,
            "measurement_end_monotonic_ns": end_ns,
            "pusch_sample_count": int(tail["pusch_samples"]),
            "mcs_sample_count": int(tail["mcs_samples"]),
            "achieved_median_pusch_snr_db": float(tail["achieved_pusch_snr_db_median"]),
            "achieved_pusch_snr_p05_db": lower,
            "achieved_pusch_snr_p95_db": upper,
            "uncertainty_p05_p95_width_db": upper - lower,
            "uncertainty_half_width_db": (upper - lower) / 2.0,
            "selected_mcs_median": tail["selected_mcs_median"],
            "final_mcs_median": tail["final_mcs_median"],
            "selected_mcs_histogram": json.dumps(tail["selected_mcs_histogram"], sort_keys=True),
            "final_mcs_histogram": json.dumps(tail["final_mcs_histogram"], sort_keys=True),
            "status": "ANCHOR_ACCEPTED",
        }
        self.anchor_rows.append(row)
        return row

    def build_mapping(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        calibration = self.config["calibration"]
        require(len(self.anchor_rows) == len(calibration["commanded_noise_power_db"]), "not every ladder rung was accepted")
        for left, right in zip(self.anchor_rows[:-1], self.anchor_rows[1:]):
            require(
                float(left["applied_noise_power_db"]) < float(right["applied_noise_power_db"]),
                "RFsim command ladder order drift",
            )
            require(
                float(left["achieved_median_pusch_snr_db"])
                > float(right["achieved_median_pusch_snr_db"]),
                f"non-monotonic achieved SNR at anchors {left['anchor_index']}/{right['anchor_index']}",
            )
        anchors = [
            {
                "achieved_median_pusch_snr_db": float(row["achieved_median_pusch_snr_db"]),
                "noise_power_db": float(row["applied_noise_power_db"]),
                "source": f"PHASE14A_100MHZ_ANCHOR_{int(row['anchor_index']):02d}",
            }
            for row in self.anchor_rows
        ]
        ordered = replay.validate_mapping(anchors)
        lower = float(ordered[0]["achieved_median_pusch_snr_db"])
        upper = float(ordered[-1]["achieved_median_pusch_snr_db"])
        target_lower = float(calibration["target_lower_db"])
        target_upper = float(calibration["target_upper_db"])
        require(lower <= target_lower, f"mapping does not cover target lower bound {target_lower}: {lower}")
        require(upper >= target_upper, f"mapping does not cover target upper bound {target_upper}: {upper}")
        granularity = float(calibration["command_granularity_db"])
        replay.inverse_interpolate(target_lower, ordered, granularity)
        replay.inverse_interpolate(target_upper, ordered, granularity)
        return ordered, {
            "strictly_monotonic": True,
            "measured_lower_db": lower,
            "measured_upper_db": upper,
            "required_lower_db": target_lower,
            "required_upper_db": target_upper,
            "complete_target_range_covered": True,
            "interpolation": "MONOTONIC_PIECEWISE_LINEAR_INVERSE",
            "command_granularity_db": granularity,
        }

    def restore(self, model_index: int) -> None:
        assert self.output is not None and self.telnet is not None
        clean = float(self.config["actuator"]["clean_restore_noise_power_db"])
        self.send_and_verify(model_index, clean, "MANDATORY_CLEAN_RESTORE")
        state = self.telnet.command("channelmod show current")[-1]
        model = n2.parse_channel_models(state).get(str(self.config["actuator"]["channel_model_name"]), {})
        self.restored = math.isclose(float(model.get("noise_power_db", math.nan)), clean, abs_tol=1e-6)
        atomic_text(self.output / "channel_state_restored.txt", state)
        require(self.restored, f"noise_power_dB={clean:g} restoration was not verified")

    def cleanup(self) -> list[str]:
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
        if self.telnet is not None:
            try:
                self.telnet.close()
            except Exception as exc:
                errors.append(f"telnet: {type(exc).__name__}: {exc}")
            self.telnet = None
        return errors

    def write_report(self, mapping_gate: Mapping[str, Any]) -> None:
        assert self.output is not None
        lines = [
            "# SplitFusion Phase-14A 100-MHz RFsim calibration",
            "",
            f"- Radio: {RADIO_PROFILE_ID}",
            f"- Accepted anchors: {len(self.anchor_rows)}",
            f"- Required target range: {mapping_gate['required_lower_db']:.1f}-{mapping_gate['required_upper_db']:.1f} dB",
            f"- Measured anchor range: {mapping_gate['measured_lower_db']:.3f}-{mapping_gate['measured_upper_db']:.3f} dB",
            f"- Strict monotonicity: {mapping_gate['strictly_monotonic']}",
            f"- Clean noise_power_dB=-50 restore: {self.restored}",
            "- Four-profile replay: deferred to a separately authorized later live phase.",
            "- Network-profile values, seeds, transitions, and accepted prefixes were not changed.",
            "- This output is measured calibration evidence pending replay, not campaign authorization.",
            "",
        ]
        atomic_text(self.output / "REPORT.md", "\n".join(lines))

    def write_manifest(self, summary: Mapping[str, Any]) -> Path:
        assert self.output is not None
        files = []
        for path in sorted(self.output.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", CALIBRATION_TERMINAL, "FAILED.json"}:
                files.append(
                    {
                        "path": str(path.relative_to(self.output)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "schema": "scenesense.splitfusion_phase14a_100mhz_calibration_manifest.v1",
            "status": summary["status"],
            "radio_profile_id": RADIO_PROFILE_ID,
            "mapping_qualified_for_campaign": False,
            "profile_replay_performed": False,
            "payload_blobs_retained": False,
            "outputs": files,
        }
        path = self.output / "manifest.json"
        atomic_json(path, manifest)
        return path

    def run(self) -> int:
        preflight = self.preflight()
        output_root = repo_path(str(self.config["paths"]["output_root"]), strict=False)
        self.output = create_only_directory(self.output_value, output_root)
        atomic_json(self.output / "resolved_config.json", self.config)
        atomic_json(self.output / "contract_reconciliation.json", preflight["reconciliation"])
        atomic_json(self.output / "attached_radio_snapshot.json", preflight["attached_radio"])
        model_index: int | None = None
        primary_error = ""
        mapping: list[dict[str, Any]] = []
        mapping_gate: dict[str, Any] = {}
        cleanup_errors: list[str] = []
        try:
            self.start_telemetry()
            self.start_traffic()
            model_index = self.open_actuator()
            self.establish_rnti()
            for index, command_db in enumerate(self.config["calibration"]["commanded_noise_power_db"]):
                self.measure_anchor(model_index, index, float(command_db))
            mapping, mapping_gate = self.build_mapping()
        except BaseException as exc:
            primary_error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.telnet is not None and model_index is not None:
                try:
                    self.restore(model_index)
                except BaseException as exc:
                    restore_error = f"{type(exc).__name__}: {exc}"
                    primary_error = f"{primary_error}; restore={restore_error}" if primary_error else restore_error
            elif self.nonclean_attempted:
                primary_error = (
                    f"{primary_error}; restore=control session unavailable after non-clean attempt"
                    if primary_error
                    else "restore=control session unavailable after non-clean attempt"
                )
            cleanup_errors = self.cleanup()
        atomic_json(
            self.output / "cleanup_report.json",
            {
                "restore_command_db": -50.0,
                "restore_verified": self.restored,
                "runner_owned_processes_stopped": not cleanup_errors,
                "attached_oai_processes_stopped": False,
                "errors": cleanup_errors,
            },
        )
        write_csv(
            self.output / "command_log.csv",
            (
                "purpose", "noise_power_db", "command", "send_monotonic_ns",
                "send_wall_ns", "ack_monotonic_ns", "ack_wall_ns", "ack_latency_ms",
                "response_sha256", "post_state_sha256", "status",
            ),
            self.command_rows,
        )
        write_csv(
            self.output / "anchor_summary.csv",
            (
                "anchor_index", "applied_noise_power_db", "command_ack_latency_ms",
                "measurement_start_monotonic_ns", "measurement_end_monotonic_ns",
                "pusch_sample_count", "mcs_sample_count",
                "achieved_median_pusch_snr_db", "achieved_pusch_snr_p05_db",
                "achieved_pusch_snr_p95_db", "uncertainty_p05_p95_width_db",
                "uncertainty_half_width_db", "selected_mcs_median", "final_mcs_median",
                "selected_mcs_histogram", "final_mcs_histogram", "status",
            ),
            self.anchor_rows,
        )
        if not primary_error and not cleanup_errors and self.restored:
            write_csv(
                self.output / "target_to_rfsim_mapping.csv",
                ("achieved_median_pusch_snr_db", "noise_power_db", "source"),
                mapping,
            )
            mapping_artifact = {
                "schema": "scenesense.splitfusion_phase14a_target_snr_mapping.v1",
                "status": "MEASURED_100MHZ_MAPPING_PENDING_FOUR_PROFILE_REPLAY",
                "radio_profile_id": RADIO_PROFILE_ID,
                "legacy_mapping_used": False,
                "mapping_qualified_for_campaign": False,
                "profile_replay_required": True,
                "gates": mapping_gate,
                "anchors": mapping,
            }
            atomic_json(self.output / "mapping.json", mapping_artifact)
            summary = {
                "status": "CALIBRATION_CAPTURE_COMPLETE_PENDING_FOUR_PROFILE_REPLAY",
                "radio_profile_id": RADIO_PROFILE_ID,
                "anchor_count": len(self.anchor_rows),
                "all_anchor_observation_gates_passed": True,
                "mapping_gate": mapping_gate,
                "restore_verified": True,
                "profile_replay_performed": False,
                "campaign_mapping_qualified": False,
            }
            atomic_json(self.output / "calibration_summary.json", summary)
            self.write_report(mapping_gate)
            manifest = self.write_manifest(summary)
            atomic_json(
                self.output / CALIBRATION_TERMINAL,
                {**summary, "manifest_sha256": sha256_file(manifest)},
            )
            return 0
        failure = {
            "status": "FAILED",
            "error": primary_error,
            "cleanup_errors": cleanup_errors,
            "restore_verified": self.restored,
            "accepted_anchor_count": len(self.anchor_rows),
            "mapping_written": False,
            "profile_replay_performed": False,
        }
        atomic_json(self.output / "FAILED.json", failure)
        self.write_manifest(failure)
        raise Phase14AError(f"live calibration failed: {failure}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--reconcile-only", action="store_true")
    modes.add_argument("--materialize-radio-config", action="store_true")
    modes.add_argument("--record-attached-radio", action="store_true")
    modes.add_argument("--execute")
    parser.add_argument("--output")
    parser.add_argument("--radio-state", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.reconcile_only:
            require(args.output is None and args.radio_state is None, "reconciliation takes no output/radio state")
            print(json.dumps(reconcile_contract(args.config, args.binding), indent=2, sort_keys=True))
            return 0
        if args.materialize_radio_config:
            require(args.output is not None and args.radio_state is None, "materialization requires only --output")
            materialize_radio(args.config, args.binding, args.output)
            return 0
        if args.record_attached_radio:
            require(args.radio_state is not None and args.output is None, "recording requires only --radio-state")
            record_attached_radio(args.config, args.binding, args.radio_state)
            return 0
        require(args.execute == CALIBRATION_TOKEN, "exact calibration execution token is required")
        require(args.output is not None and args.radio_state is not None, "live calibration requires --output and --radio-state")
        return AttachedCalibration(
            args.config,
            args.binding,
            args.radio_state,
            args.output,
        ).run()
    except (Phase14AError, n2.SmokeFailure, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PHASE14A_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
