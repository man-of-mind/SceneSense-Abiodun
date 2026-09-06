#!/usr/bin/env python3
"""Apply one fixed Gaussian-Markov target-SNR profile for one campaign cell.

One ``DeterministicTargetSnrSequence`` is instantiated.  Its first 4,200
samples are generated once, verified against the accepted prefix/hash, replayed
from cached sample zero, and then the same live RNG/Markov state continues.
The scheduler never wraps, holds the final prefix value, reseeds, or bursts
obsolete commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = ROOT / "rl_agent/configs/ue_288_campaign_v1.yaml"
DEFAULT_OAI = ROOT / "rl_agent/configs/ue_n3e_fallback_snr_floor_v1.json"
FIELDS = (
    "profile_id",
    "trace_id",
    "seed",
    "step_index",
    "target_snr_db",
    "mapped_rfsim_command_db",
    "mapped_rfsim_command",
    "achieved_snr_db",
    "scheduled_monotonic_ns",
    "interval_end_monotonic_ns",
    "command_send_monotonic_ns",
    "command_ack_monotonic_ns",
    "command_send_wall_ns",
    "command_ack_wall_ns",
    "command_latency_ms",
    "command_timing_status",
)


class RuntimeContractError(RuntimeError):
    """The fixed trace, mapping, or RFsim actuator contract failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"YAML root is not a mapping: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_generator_module() -> Any:
    path = ROOT / "rl_agent/generate_network_profile_meeting_figures.py"
    spec = importlib.util.spec_from_file_location("ue_network_profile_design_v2", path)
    require(spec is not None and spec.loader is not None, "cannot import deterministic SNR generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_replay_module() -> Any:
    path = ROOT / "rl_agent/oai_target_snr_replay_pilot_v1.py"
    spec = importlib.util.spec_from_file_location("ue_oai_target_replay", path)
    require(spec is not None and spec.loader is not None, "cannot import accepted target/RFsim mapping helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_oai_module() -> Any:
    path = ROOT / "rl_agent/ue_n2_oai_ul_calibration_smoke.py"
    spec = importlib.util.spec_from_file_location("ue_n2_actuator", path)
    require(spec is not None and spec.loader is not None, "cannot import OAI actuator helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_mapping(path: Path, replay: Any) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        anchors = [dict(row) for row in csv.DictReader(handle)]
    return replay.validate_mapping(anchors)


def load_accepted_prefix(path: Path, profile_id: str, count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["profile_id"] == profile_id and int(row["step_index"]) < count:
                rows.append(row)
    rows.sort(key=lambda row: int(row["step_index"]))
    require(len(rows) == count, f"{profile_id}: expected {count} accepted prefix rows, found {len(rows)}")
    require([int(row["step_index"]) for row in rows] == list(range(count)), f"{profile_id}: accepted prefix index drift")
    return rows


def prepare_sequence(
    *,
    campaign: Mapping[str, Any],
    profile_id: str,
) -> tuple[Any, list[tuple[int, float]], Mapping[str, Any]]:
    network = campaign["network"]
    design = load_json(repo_path(str(network["design_config"])))
    profiles = {row["profile_id"]: row for row in design["profiles"]}
    profile = profiles.get(profile_id)
    require(profile is not None, f"unknown network profile: {profile_id}")
    frozen = next((row for row in network["profiles"] if row["profile_id"] == profile_id), None)
    require(frozen is not None, f"profile is not registered by campaign: {profile_id}")
    require(int(profile["seed"]) == int(frozen["seed"]), f"seed drift for {profile_id}")
    require(str(profile["trace_id"]) == str(frozen["trace_id"]), f"trace ID drift for {profile_id}")

    generator = load_generator_module()
    sequence = generator.DeterministicTargetSnrSequence(profile, design)
    count = int(network["prefix_samples"])
    prefix = [sequence.next_sample() for _ in range(count)]
    states = np.asarray([row[0] for row in prefix], dtype="<i4")
    targets = np.asarray([row[1] for row in prefix], dtype="<f8")
    digest = hashlib.sha256(states.tobytes() + targets.tobytes()).hexdigest()
    require(digest == str(frozen["trace_sha256"]), f"generated prefix hash mismatch for {profile_id}")

    accepted = load_accepted_prefix(repo_path(str(network["traces_csv"])), profile_id, count)
    require(np.array_equal(states, np.asarray([int(row["state_index"]) for row in accepted])), f"accepted state prefix mismatch for {profile_id}")
    require(
        np.allclose(targets, np.asarray([float(row["target_snr_db"]) for row in accepted]), rtol=0.0, atol=5.1e-7),
        f"accepted target prefix mismatch for {profile_id}",
    )
    return sequence, prefix, frozen


def open_actuator(oai_config: Mapping[str, Any], oai: Any) -> tuple[Any, int]:
    actuator = oai_config["actuator"]
    session = oai.TelnetSession(
        str(actuator["telnet_host"]),
        int(actuator["telnet_port"]),
        float(actuator["response_timeout_s"]),
        int(actuator["max_response_bytes"]),
    )
    try:
        _sm, _sw, _am, _aw, response = session.command("channelmod show current")
        models = oai.parse_channel_models(response)
        row = models.get(str(actuator["channel_model_name"]))
        require(row is not None, f"active RFsim model missing: {actuator['channel_model_name']}")
        require(row.get("model_type") == actuator["channel_model_type"], f"RFsim model type drift: {row}")
        require(row.get("owner") == actuator["channel_model_owner"], f"RFsim model owner drift: {row}")
        require(math.isclose(float(row.get("path_loss_db", math.nan)), float(actuator["path_loss_db"]), abs_tol=1e-6), f"RFsim path-loss drift: {row}")
        return session, int(row["model_index"])
    except Exception:
        session.close()
        raise


def validate_modify(response: str, target: str, oai: Any) -> None:
    oai.Runner.validate_modify_response(response, target)


def write_row(handle: Any, writer: csv.DictWriter, row: Mapping[str, Any]) -> None:
    writer.writerow({field: row.get(field, "") for field in FIELDS})
    handle.flush()


def run(args: argparse.Namespace) -> int:
    campaign = load_yaml(args.campaign.resolve())
    network = campaign["network"]
    require(int(network["sample_period_ms"]) == 100, "runtime requires a 100-ms period")
    require(network["catch_up_policy"] == "SKIP_OBSOLETE_NEVER_BURST", "catch-up policy drift")
    require(float(network["clean_restore_noise_power_db"]) == -50.0, "clean restore must be -50 dB")
    sequence, prefix, frozen = prepare_sequence(campaign=campaign, profile_id=args.profile_id)
    replay = load_replay_module()
    mapping = load_mapping(repo_path(str(network["mapping_csv"])), replay)
    granularity = float(args.command_granularity_db)
    oai = load_oai_module()
    oai_config = load_json(args.oai_config.resolve())
    session, model_index = open_actuator(oai_config, oai)
    stop_event = threading.Event()

    def stop_handler(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    period_ns = 100_000_000
    step = 0
    restored = False
    error = ""
    try:
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
            writer.writeheader()
            handle.flush()
            anchor = time.monotonic_ns() + period_ns
            while not stop_event.is_set() and not (args.stop_file and args.stop_file.exists()):
                scheduled = anchor + step * period_ns
                interval_end = scheduled + period_ns
                while not stop_event.is_set():
                    remaining = scheduled - time.monotonic_ns()
                    if remaining <= 0:
                        break
                    stop_event.wait(min(remaining / 1e9, 0.02))
                if step < len(prefix):
                    _state, target_snr = prefix[step]
                else:
                    _state, target_snr = sequence.next_sample()
                mapped = replay.inverse_interpolate(float(target_snr), mapping, granularity)
                command_value = f"{mapped:.12g}"
                command_text = f"channelmod modify {model_index} noise_power_dB {command_value}"
                base = {
                    "profile_id": args.profile_id,
                    "trace_id": frozen["trace_id"],
                    "seed": int(frozen["seed"]),
                    "step_index": step,
                    "target_snr_db": float(target_snr),
                    "mapped_rfsim_command_db": mapped,
                    "mapped_rfsim_command": command_text,
                    "achieved_snr_db": "",
                    "scheduled_monotonic_ns": scheduled,
                    "interval_end_monotonic_ns": interval_end,
                }
                before_send = time.monotonic_ns()
                if before_send >= interval_end:
                    write_row(handle, writer, {**base, "command_timing_status": "SKIP_OBSOLETE_NEVER_BURST"})
                    step += 1
                    continue
                sent_mono, sent_wall, ack_mono, ack_wall, response = session.command(command_text)
                validate_modify(response, command_value, oai)
                timing = "ACK_ON_TIME" if ack_mono < interval_end else "ACK_LATE"
                write_row(
                    handle,
                    writer,
                    {
                        **base,
                        "command_send_monotonic_ns": sent_mono,
                        "command_ack_monotonic_ns": ack_mono,
                        "command_send_wall_ns": sent_wall,
                        "command_ack_wall_ns": ack_wall,
                        "command_latency_ms": (ack_mono - sent_mono) / 1e6,
                        "command_timing_status": timing,
                    },
                )
                step += 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        restore_error = ""
        try:
            clean = "-50"
            _sm, _sw, _am, _aw, response = session.command(
                f"channelmod modify {model_index} noise_power_dB {clean}"
            )
            validate_modify(response, clean, oai)
            _sm, _sw, _am, _aw, state = session.command("channelmod show current")
            row = oai.parse_channel_models(state).get(oai_config["actuator"]["channel_model_name"], {})
            restored = math.isclose(float(row.get("noise_power_db", math.nan)), -50.0, abs_tol=1e-6)
            if not restored and not error:
                error = f"post-restore state mismatch: {row}"
        except Exception as exc:
            restore_error = f"{type(exc).__name__}: {exc}"
            if not error:
                error = "clean restore failed: " + restore_error
        finally:
            session.close()
        atomic_json(
            output.with_suffix(output.suffix + ".summary.json"),
            {
                "schema": "scenesense.ue_target_snr_cell_runtime.v1",
                "profile_id": args.profile_id,
                "trace_id": frozen["trace_id"],
                "seed": int(frozen["seed"]),
                "rows": step,
                "same_sequence_continued_after_prefix": step > len(prefix),
                "clean_restore_noise_power_db": -50.0,
                "clean_restore_verified": restored,
                "clean_restore_error": restore_error,
                "error": error,
            },
        )
    require(restored, "RFsim noise_power_dB=-50 restore was not verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--oai-config", type=Path, default=DEFAULT_OAI)
    parser.add_argument("--command-granularity-db", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except RuntimeContractError as exc:
        print(f"target-SNR runtime contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
