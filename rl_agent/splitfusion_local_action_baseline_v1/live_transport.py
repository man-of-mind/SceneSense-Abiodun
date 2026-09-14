#!/usr/bin/env python3
"""Measure compact full-local map updates over four frozen OAI profiles.

This is the transport half of the LOCAL action qualification.  A fresh
OAI/RFsim lifecycle is used per profile.  The payloads are real current-FCOS
p025 object outputs produced from the registered 300-frame fit sample; they
are held only in memory.  The timing model reuses the same per-frame measured
full-local compute duration, then performs a single real UDP upload and waits
for an authoritative edge-install ACK.  CARLA and dense segmentation are not
part of this measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import struct
import subprocess
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent import oai_target_snr_replay_pilot_v1 as replay
from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b
from rl_agent.splitfusion_local_action_baseline_v1 import compute_baseline
from rl_agent.splitfusion_local_action_baseline_v1.map_sink import ACK_SCHEMA, CHUNK_HEADER


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "rl_agent/configs/splitfusion_local_action_baseline_v1.json"
EXECUTION_TOKEN = "SPLITFUSION_LOCAL_COMPACT_RESULT_FOUR_PROFILE_MEASUREMENT"
SUCCESS_TERMINAL = "SPLITFUSION_LOCAL_ACTION_BASELINE_COMPLETE"
CLOCK_RAW = getattr(time, "CLOCK_MONOTONIC_RAW", time.CLOCK_MONOTONIC)
PROFILE_FIELDS = (
    "profile_id", "step_index", "state_index", "target_snr_db",
    "command_noise_power_db", "command_ack_latency_ms", "schedule_status",
    "frame_id", "compute_ms", "capture_raw_ns", "local_result_available_raw_ns",
    "payload_bytes", "chunks", "application_send_attempts", "send_start_raw_ns",
    "send_end_raw_ns", "ack_status", "edge_first_datagram_raw_ns",
    "edge_complete_raw_ns", "edge_install_raw_ns", "feedback_emit_raw_ns",
    "ack_receive_raw_ns", "local_to_first_datagram_ms",
    "local_to_complete_message_ms", "local_to_map_install_ms",
    "map_install_to_ack_ms", "local_to_ack_ms", "capture_to_map_install_ms",
    "capture_to_ack_ms", "fresh_150ms", "fresh_200ms", "fresh_250ms",
)


class LocalTransportError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalTransportError(message)


def raw_ns() -> int:
    return time.clock_gettime_ns(CLOCK_RAW)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def repo_path(value: str) -> Path:
    candidate = (ROOT / value).resolve(strict=True)
    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise LocalTransportError(f"path escapes repository: {value}") from exc
    return candidate


def verified_file(record: Mapping[str, Any], label: str) -> Path:
    path = repo_path(str(record["path"]))
    require(sha256_file(path) == str(record["sha256"]), f"{label} hash drift")
    return path


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    return ordered[max(0, math.ceil(len(ordered) * float(fraction)) - 1)]


def distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(clean),
        "minimum": min(clean) if clean else None,
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "maximum": max(clean) if clean else None,
        "mean": sum(clean) / len(clean) if clean else None,
    }


def rewrite_payload(
    payload: bytes,
    *,
    run_id: str,
    stream_id: str,
    frame_id: int,
    capture_ns: int,
    available_ns: int,
) -> bytes:
    document = compute_baseline.decode_local_payload(payload)
    records = []
    for source in document["objects"]:
        record = dict(source)
        record["stream_id"] = stream_id
        record["frame_id"] = int(frame_id)
        record["capture_timestamp_ns"] = int(capture_ns)
        record["sample_id"] = f"{stream_id}:{frame_id}"
        records.append(record)
    rewritten = compute_baseline.build_local_payload(
        run_id=run_id,
        frame_id=frame_id,
        capture_timestamp_ns=capture_ns,
        local_result_available_ns=available_ns,
        records=records,
        checkpoint_sha256=str(document["checkpoint_sha256"]),
        stream_id=stream_id,
    )
    verified = compute_baseline.decode_local_payload(rewritten)
    require(int(verified["frame_id"]) == frame_id, "rewritten frame identity drift")
    require(str(verified["stream_id"]) == stream_id, "rewritten stream identity drift")
    return rewritten


def chunk_payload(payload: bytes, message_id: int, chunk_bytes: int) -> list[bytes]:
    require(chunk_bytes > 0, "chunk size must be positive")
    count = max(1, math.ceil(len(payload) / chunk_bytes))
    require(count <= 0xFFFF, "payload requires too many chunks")
    return [
        CHUNK_HEADER.pack(int(message_id), index, count)
        + payload[index * chunk_bytes : (index + 1) * chunk_bytes]
        for index in range(count)
    ]


class AckCollector:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.rows: dict[int, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="local-ack-collector", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        require(not self.thread.is_alive(), "ACK collector did not stop")

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                packet, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stop_event.is_set():
                    self.errors.append(f"OSError: {exc}")
                return
            received = raw_ns()
            try:
                value = json.loads(packet.decode("utf-8"))
                if value.get("schema") != ACK_SCHEMA:
                    raise ValueError("ACK schema")
                frame_id = int(value["frame_id"])
                value["ack_receive_raw_ns"] = received
                with self.lock:
                    if frame_id in self.rows:
                        raise ValueError(f"duplicate ACK {frame_id}")
                    self.rows[frame_id] = value
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def snapshot(self) -> dict[int, dict[str, Any]]:
        with self.lock:
            return {key: dict(value) for key, value in self.rows.items()}


def merge_sender_sink_rows(
    commands: Sequence[Mapping[str, Any]],
    acknowledgements: Mapping[int, Mapping[str, Any]],
    sink_rows: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for command in commands:
        row = dict(command)
        frame_id = int(row["frame_id"])
        ack = dict(acknowledgements.get(frame_id, {}))
        sink = dict(sink_rows.get(frame_id, {}))
        row["ack_status"] = str(ack.get("status") or "NO_ACK")
        for field in (
            "first_datagram_raw_ns", "complete_raw_ns", "edge_install_raw_ns",
            "feedback_emit_raw_ns",
        ):
            row[f"edge_{field}" if field in ("first_datagram_raw_ns", "complete_raw_ns") else field] = int(sink.get(field) or ack.get(field) or 0)
        row["ack_receive_raw_ns"] = int(ack.get("ack_receive_raw_ns") or 0)
        available = int(row.get("local_result_available_raw_ns") or 0)
        capture = int(row.get("capture_raw_ns") or 0)
        first = int(row.get("edge_first_datagram_raw_ns") or 0)
        complete = int(row.get("edge_complete_raw_ns") or 0)
        install = int(row.get("edge_install_raw_ns") or 0)
        emitted = int(row.get("feedback_emit_raw_ns") or 0)
        received = int(row.get("ack_receive_raw_ns") or 0)
        pairs = {
            "local_to_first_datagram_ms": (first, available),
            "local_to_complete_message_ms": (complete, available),
            "local_to_map_install_ms": (install, available),
            "map_install_to_ack_ms": (received, install),
            "local_to_ack_ms": (received, available),
            "capture_to_map_install_ms": (install, capture),
            "capture_to_ack_ms": (received, capture),
        }
        for name, (end, start) in pairs.items():
            row[name] = (end - start) / 1e6 if end > 0 and start > 0 else ""
        for budget in (150, 200, 250):
            row[f"fresh_{budget}ms"] = bool(
                install > 0 and capture > 0 and install - capture <= budget * 1_000_000
            )
        result.append(row)
    return result


def summarize_profile(rows: Sequence[Mapping[str, Any]], sink_summary: Mapping[str, Any]) -> dict[str, Any]:
    sent = [row for row in rows if row["schedule_status"] == "SENT_ONCE"]
    installed = [row for row in sent if row["ack_status"] == "ACK_INSTALLED"]
    metrics = (
        "local_to_first_datagram_ms", "local_to_complete_message_ms",
        "local_to_map_install_ms", "map_install_to_ack_ms", "local_to_ack_ms",
        "capture_to_map_install_ms", "capture_to_ack_ms",
    )
    return {
        "intended_intervals": len(rows),
        "sent_once": len(sent),
        "skipped_obsolete": len(rows) - len(sent),
        "ack_installed": len(installed),
        "installation_rate_per_sent": len(installed) / len(sent) if sent else 0.0,
        "one_shot_no_retry_verified": all(int(row.get("application_send_attempts", 0)) == 1 for row in sent),
        "timing_ms": {
            name: distribution([float(row[name]) for row in installed if row[name] != ""])
            for name in metrics
        },
        "fresh_install": {
            str(budget): {
                "count": sum(bool(row[f"fresh_{budget}ms"]) for row in sent),
                "per_sent": sum(bool(row[f"fresh_{budget}ms"]) for row in sent) / len(sent) if sent else 0.0,
            }
            for budget in (150, 200, 250)
        },
        "sink": dict(sink_summary),
    }


def wait_raw(deadline_ns: int) -> None:
    while True:
        remaining = deadline_ns - raw_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.002))


def construct_radio_runner(
    replay_config_path: Path,
    current_binding_path: Path,
    launcher_path: Path,
    mapping_path: Path,
    output: Path,
    radio_namespace: Path,
) -> phase14b.CorrectedFourProfileReplay:
    """Construct the proven lifecycle against the current Phase-14A binding.

    The historical corrected-replay config predates the SFD1-v2 dispatcher
    binding transition and therefore intentionally names the former binding
    hash.  Its four traces and radio contract remain authoritative, while this
    LOCAL measurement independently seals the current binding and launcher.
    Calling the historical runner's provenance gate would conflate that
    dispatch-only transition with radio drift, so the lifecycle object is
    assembled explicitly from both sealed authorities.
    """

    replay_config = load_json(replay_config_path)
    binding = load_json(current_binding_path)
    require(
        binding["launcher"]["path"] == str(launcher_path.relative_to(ROOT)),
        "current Phase-14A binding names another launcher",
    )
    require(
        binding["launcher"]["sha256"] == sha256_file(launcher_path),
        "current Phase-14A launcher seal drift",
    )
    require(
        binding.get("dispatcher", {}).get("runtime_binding_transition", {}).get("scope")
        == "Phase-13C SFD1-v2 source reconciliation; no radio semantic change",
        "current binding transition is not proven radio-semantic-neutral",
    )
    runner = object.__new__(phase14b.CorrectedFourProfileReplay)
    runner.phase14b_config_path = replay_config_path
    runner.provenance = {
        "launcher_path": launcher_path,
        "launcher_execution_token": binding["launcher"]["execution_token"],
        "launcher_sha256": binding["launcher"]["sha256"],
    }
    runner.phase14b = replay_config
    runner.phase14a_config_path = repo_path(str(replay_config["provenance"]["phase14a_config"]["path"]))
    runner.phase14a_binding_path = current_binding_path
    runner.base_radio_config = load_json(runner.phase14a_config_path)
    runner.radio_namespace = phase14b.resolve_radio_namespace(replay_config, radio_namespace)
    runner.probe_radio_namespace = runner.radio_namespace.parent / f"{runner.radio_namespace.name}__unused_probe"
    runner.lifecycle_plan = phase14b.profile_lifecycle_plan(replay_config, runner.radio_namespace)
    runner.output_value = str(output)
    runner.resume_run = False
    runner.prepared = phase14b.prepare_frozen_profiles(replay_config)
    mapping = load_json(mapping_path)
    runner.mapping = replay.validate_mapping(mapping["anchors"])
    runner.work = None
    runner.durable_output = None
    runner.final_restore_state = None
    runner.initial_clean_state = None
    runner.last_topology_health_ns = 0
    runner.topology_seal_ok = True
    runner.session_started = False
    runner.restored = False
    runner.nonclean_attempted = False
    return runner


def run_profile(
    runner: phase14b.CorrectedFourProfileReplay,
    prepared: Mapping[str, Any],
    profile_index: int,
    output: Path,
    compute: compute_baseline.ComputeMeasurement,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    profile_id = str(prepared["profile_id"])
    profile_dir = output / "profiles" / f"{profile_index:02d}_{profile_id}"
    profile_dir.mkdir(parents=True, exist_ok=False)
    radio_state = Path(runner.lifecycle_plan[profile_index]["radio_state_path"])
    attached = None
    model_index = None
    primary_error = ""
    cleanup_errors: list[str] = []
    lifecycle: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    runner.restored = False
    runner.nonclean_attempted = False
    runner.initial_clean_state = None
    runner.final_restore_state = None
    runner.durable_output = profile_dir
    runner.session_started = False
    try:
        cold = phase14b.require_cold_profile_runtime(runner.base_radio_config, radio_state)
        launch = runner.launch_radio(radio_state)
        runner.initialize_attached_session(radio_state)
        live = super(phase14b.CorrectedFourProfileReplay, runner).preflight()
        attached = live["attached_radio"]
        runner.start_runtime_support()
        runner.write_runtime_ownership(radio_state)
        model_index = runner.open_clean_actuator()
        super(phase14b.CorrectedFourProfileReplay, runner).establish_rnti()
        require(runner.current_rnti is not None, "single RNTI was not established")
        probe = runner.stop_traffic_and_measure()
        runner.health(force_topology=True)

        inspect = subprocess.run(
            ["sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}", "oai-ext-dn"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        require(inspect.returncode == 0, f"cannot inspect ext-DN: {inspect.stderr.strip()}")
        ext_pid = int(inspect.stdout.strip())
        stop_file = profile_dir / "STOP_MAP_SINK"
        sink_output = profile_dir / "map_sink"
        sink_process = runner.spawn(
            "local_map_sink",
            [
                "sudo", "-n", "nsenter", "-t", str(ext_pid), "-n",
                "--setuid", str(os.getuid()), "--setgid", str(os.getgid()),
                "/usr/bin/python3", str(ROOT / "rl_agent/splitfusion_local_action_baseline_v1/map_sink.py"),
                "--bind-host", str(config["measurement"]["ext_dn_ip"]),
                "--port", str(config["measurement"]["remote_port"]),
                "--expected-source-ip", str(config["measurement"]["ue_ip"]),
                "--output", str(sink_output), "--stop-file", str(stop_file),
                "--socket-buffer-bytes", str(config["measurement"]["socket_buffer_bytes"]),
                "--reassembly-timeout-s", str(config["measurement"]["map_sink_reassembly_timeout_s"]),
            ],
            "map_sink.log", root_owned=True,
        )
        runner.write_runtime_ownership(radio_state)
        time.sleep(0.5)
        require(sink_process.process.poll() is None, "map sink exited during readiness")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(config["measurement"]["socket_buffer_bytes"]))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(config["measurement"]["socket_buffer_bytes"]))
        sock.bind((str(config["measurement"]["ue_ip"]), 0))
        sock.settimeout(0.05)
        collector = AckCollector(sock)
        collector.start()
        try:
            runner.restore_and_readback(model_index)
            runner.wait_until(time.monotonic_ns() + 1_000_000_000)
            period_ns = int(config["measurement"]["sample_period_ms"]) * 1_000_000
            count = int(config["measurement"]["samples_per_profile"])
            granularity = float(runner.phase14b["replay"]["command_granularity_db"])
            anchor = time.monotonic_ns() + period_ns
            previous_send: int | None = None
            for step, (state_index, target) in enumerate(prepared["prefix"][:count]):
                scheduled = anchor + step * period_ns
                decision = phase14b.plan_scheduler_action(
                    scheduled_ns=scheduled, period_ns=period_ns,
                    now_ns=time.monotonic_ns(), previous_send_ns=previous_send,
                )
                base = {
                    "profile_id": profile_id, "step_index": step,
                    "state_index": int(state_index), "target_snr_db": float(target),
                    "frame_id": step, "compute_ms": float(compute.rows[step]["local_result_available_ms"]),
                    "payload_bytes": 0, "chunks": 0, "send_start_raw_ns": 0,
                    "send_end_raw_ns": 0, "application_send_attempts": 0,
                    "capture_raw_ns": 0, "local_result_available_raw_ns": 0,
                }
                if decision["status"] != "SEND_ON_ABSOLUTE_SCHEDULE":
                    rows.append({**base, "schedule_status": "SKIP_OBSOLETE_NEVER_BURST", "command_noise_power_db": "", "command_ack_latency_ms": ""})
                    continue
                runner.wait_until(int(decision["eligible_send_ns"]))
                capture = raw_ns()
                command_db = replay.inverse_interpolate(float(target), runner.mapping, granularity)
                command = runner.send_target(model_index, command_db)
                available = capture + int(round(float(base["compute_ms"]) * 1e6))
                wait_raw(available)
                stream = f"{output.name}/{profile_id}/local"
                payload = rewrite_payload(
                    compute.payloads[step], run_id=output.name, stream_id=stream,
                    frame_id=step, capture_ns=capture, available_ns=available,
                )
                packets = chunk_payload(payload, step, int(config["measurement"]["udp_chunk_bytes"]))
                send_started = raw_ns()
                for packet in packets:
                    sock.sendto(packet, (str(config["measurement"]["ext_dn_ip"]), int(config["measurement"]["remote_port"])))
                send_ended = raw_ns()
                previous_send = time.monotonic_ns()
                rows.append({
                    **base, "schedule_status": "SENT_ONCE",
                    "command_noise_power_db": command_db,
                    "command_ack_latency_ms": command["command_ack_latency_ms"],
                    "capture_raw_ns": capture, "local_result_available_raw_ns": available,
                    "payload_bytes": len(payload), "chunks": len(packets),
                    "send_start_raw_ns": send_started, "send_end_raw_ns": send_ended,
                    "application_send_attempts": 1,
                })
                if (step + 1) % 50 == 0:
                    print(f"LOCAL {profile_id}: {step + 1}/{count}", flush=True)
            runner.wait_until(time.monotonic_ns() + int(float(config["measurement"]["ack_drain_s"]) * 1e9))
            runner.health(force_topology=True)
        finally:
            collector.stop()
            sock.close()
        require(not collector.errors, f"ACK collector errors: {collector.errors}")
        stop_file.touch(exist_ok=False)
        sink_process.process.wait(timeout=3.0)
        sink_process.stop()
        runner.processes = [process for process in runner.processes if process is not sink_process]
        require(sink_process.process.returncode == 0, "map sink exited nonzero")
        runner.restore_and_readback(model_index)
        runner.health(force_topology=True)
        sink_rows_list = list(csv.DictReader((sink_output / "map_sink_frames.csv").open(newline="", encoding="utf-8")))
        sink_rows = {int(row["frame_id"]): row for row in sink_rows_list}
        require(len(sink_rows) == len(sink_rows_list), "duplicate complete sink messages")
        merged = merge_sender_sink_rows(rows, collector.snapshot(), sink_rows)
        sink_summary = load_json(sink_output / "map_sink_summary.json")
        summary = summarize_profile(merged, sink_summary)
        summary.update({
            "schema": "scenesense.splitfusion.local_transport_profile.v1",
            "profile_id": profile_id, "trace_id": prepared["trace_id"],
            "trace_sha256": prepared["trace_sha256"], "rnti": runner.current_rnti,
            "probe_traffic_used_only_for_rnti_establishment": probe,
            "desktop_compute_replayed_before_one_shot_transport": True,
        })
        require(summary["one_shot_no_retry_verified"] is True, "one-shot rule failed")
        require(int(sink_summary["nack_rejected"]) == 0, "map sink rejected a result")
        require(int(sink_summary["duplicate_datagrams"]) == 0, "duplicate datagrams observed")
        atomic_csv(profile_dir / "local_transport_frames.csv", PROFILE_FIELDS, merged)
        atomic_json(profile_dir / "profile_summary.json", summary)
        result = {"cold_preflight": cold, "launcher_stdout_sha256": hashlib.sha256(launch.stdout.encode()).hexdigest(), "attached": phase14b.stable_attached_radio_seal(attached), "summary": summary}
    except BaseException as exc:
        primary_error = f"{type(exc).__name__}: {exc}"
        result = {}
    finally:
        if runner.session_started:
            cleanup_errors.extend(runner.stop_runtime_support_keep_actuator())
            if runner.telnet is not None and model_index is not None:
                try:
                    runner.restore_and_readback(model_index)
                except BaseException as exc:
                    cleanup_errors.append(f"restore: {type(exc).__name__}: {exc}")
            cleanup_errors.extend(runner.close_actuator_and_work())
        lifecycle = phase14b.teardown_profile_runtime(
            runner.base_radio_config, radio_state, runner.radio_namespace, attached,
            restore_verified=bool(runner.restored) if runner.session_started else True,
        )
        cleanup_errors.extend(lifecycle.get("errors", []))
        runner.session_started = False
    if primary_error or cleanup_errors:
        atomic_json(profile_dir / "FAILED.json", {"primary_error": primary_error, "cleanup_errors": cleanup_errors, "lifecycle": lifecycle})
        raise LocalTransportError(primary_error or "; ".join(cleanup_errors))
    require(lifecycle.get("all_lifecycle_gates_passed") is True, "radio teardown gates failed")
    result["lifecycle"] = lifecycle
    atomic_json(profile_dir / "COMPLETE.json", result)
    return result


def run(config_path: Path, output: Path, radio_namespace: Path) -> dict[str, Any]:
    config = load_json(config_path)
    require(config["execution_token"] == EXECUTION_TOKEN, "config execution token drift")
    for label, record in config["provenance"].items():
        verified_file(record, label)
    require(not output.exists(), f"create-only output exists: {output}")
    require(not radio_namespace.exists(), f"radio namespace exists: {radio_namespace}")
    compute = compute_baseline.measure(f"{output.name}/compute")
    require(len(compute.rows) == len(compute.payloads) == 300, "compute batch size drift")
    baseline = load_json(verified_file(config["provenance"]["compute_baseline_summary"], "compute baseline summary"))
    require(compute.sample["selected_sample_id_sha256"] == baseline["selected_sample_id_sha256"], "compute sample drift")
    require(compute.model_binding["checkpoint_sha256"] == baseline["model_binding"]["checkpoint_sha256"], "compute model drift")
    output.mkdir(parents=True, exist_ok=False)
    (output / "profiles").mkdir()
    replay_config_path = verified_file(config["provenance"]["radio_replay_config"], "radio replay config")
    runner = construct_radio_runner(
        replay_config_path,
        verified_file(config["provenance"]["current_phase14a_binding"], "current Phase-14A binding"),
        verified_file(config["provenance"]["qualified_radio_launcher"], "qualified radio launcher"),
        verified_file(config["provenance"]["phase14a_mapping"], "Phase-14A mapping"),
        output,
        radio_namespace,
    )
    require([row["profile_id"] for row in runner.prepared] == config["measurement"]["profile_order"], "profile order drift")
    require(int(config["measurement"]["samples_per_profile"]) == 300, "sample count drift")
    atomic_json(output / "run_manifest.json", {
        "schema": "scenesense.splitfusion.local_action_run_manifest.v1",
        "config_path": str(config_path.relative_to(ROOT)), "config_sha256": sha256_file(config_path),
        "implementation_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "compute_sample_manifest_sha256": compute.sample["sample_manifest_sha256"],
        "selected_sample_id_sha256": compute.sample["selected_sample_id_sha256"],
        "checkpoint_sha256": compute.model_binding["checkpoint_sha256"],
        "profile_order": config["measurement"]["profile_order"],
        "freshness_budgets_ms": config["measurement"]["freshness_budgets_ms"],
    })
    results = [
        run_profile(runner, prepared, index, output, compute, config)
        for index, prepared in enumerate(runner.prepared)
    ]
    summary = {
        "schema": "scenesense.splitfusion.local_action_baseline_summary.v1",
        "status": "COMPLETE", "profiles": [row["summary"] for row in results],
        "compute": compute.summary, "quality": compute.summary["quality"],
        "interpretation": config["interpretation"],
    }
    atomic_json(output / "qualification.json", summary)
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    atomic_json(output / "artifact_manifest.json", {"schema": "scenesense.splitfusion.local_action_artifacts.v1", "sha256": artifacts})
    atomic_text(output / SUCCESS_TERMINAL, SUCCESS_TERMINAL + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radio-namespace", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.execute == EXECUTION_TOKEN, "execution token mismatch")
    summary = run(args.config.resolve(strict=True), args.output.resolve(), args.radio_namespace.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(SUCCESS_TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
