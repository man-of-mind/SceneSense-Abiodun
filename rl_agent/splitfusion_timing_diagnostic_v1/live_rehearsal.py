#!/usr/bin/env python3
"""Offline rehearsal of the live parent flow, CARLA excluded.

The live cell's newest seam is the parent/child boundary: the parent owns the
radio, the CARLA server, the edge container, the actuator, teardown and
evidence, while a child process owns everything that touches the CARLA client
API. This harness exercises the real :func:`live_runner.run_live_action`
verbatim and substitutes only what needs the radio host or CARLA:

* the qualified OAI launcher, the radio teardown, the target-SNR actuator and
  the T-tracer collector are stubbed;
* the CARLA server lifecycle is stubbed;
* the edge is the real ``edge_service`` module over loopback rather than in
  the container;
* the child is stubbed by a ``subprocess`` shim that transmits real
  precomputed payloads to that edge, records the same wall-clock send
  boundaries the instrumented socket records, and writes the same three
  artifacts the real child writes.

Everything else runs exactly as it does live: the warm-up and equivalence
proof, reassembly, the queue, the tail decomposition, the graceful-shutdown
handshake, the artifact handoff, the record join, the accounting gate, the
cold proof, the summaries, the comparisons and every evidence writer.

It produces **no scientific evidence**: there is no CARLA scene, no radio and
no route, so its outputs are a plumbing proof written to a scratch directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from phase2_map_sharing.transport import chunk_payload
from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry

from . import diagnostic_common as common
from . import live_runner, runner
from .edge_preload import preload_ue


REHEARSAL_ACTIONS = (50, 71)
REHEARSAL_FRAMES = 8
LOOPBACK = "127.0.0.1"
EDGE_PORT = 51902
RESULT_PORT = 51904
EDGE_SOURCE_PORT = 51913
MAP_PORT = 51910
FIRST_FRAME_ID = 9000


class _StubTelemetry:
    def __init__(self, base: Any, scratch: Path) -> None:
        self.status = "REHEARSAL_NOT_COLLECTED"
        self.error = ""

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        empty = common.summarize([])
        return {
            "status": self.status, "error": self.error, "pusch_samples": 0,
            "mcs_samples": 0, "achieved_pusch_snr_db": empty,
            "achieved_pusch_mcs": empty, "scheduler_avg_snr_db": empty,
            "scheduler_selected_ul_mcs": empty, "scheduler_final_ul_mcs": empty,
            "raw_tracer_rows_retained": False,
        }


class _ChildShim:
    """Stands in for ``subprocess`` inside live_runner for the child launch."""

    DEVNULL = subprocess.DEVNULL

    def __init__(self, payloads: dict[int, bytes], state: dict[str, Any]) -> None:
        self._payloads = payloads
        self._state = state

    def run(self, argv: Sequence[str], **keywords: Any) -> Any:
        values = list(argv)
        artifacts = Path(values[values.index("--artifacts-dir") + 1])
        budget = int(values[values.index("--transmitted-budget") + 1])
        artifacts.mkdir(parents=True, exist_ok=True)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
        boundaries: dict[str, dict[str, int]] = {}
        rows: list[dict[str, Any]] = []
        try:
            for index, (frame_id, payload) in enumerate(sorted(self._payloads.items())):
                first = time.time_ns()
                chunks = chunk_payload(payload, message_id=frame_id, chunk_bytes=12_500)
                for chunk in chunks:
                    sender.sendto(chunk, (LOOPBACK, EDGE_PORT))
                final = time.time_ns()
                boundaries[str(frame_id)] = {
                    "ue_first_send_wall_ns": first,
                    "ue_final_send_wall_ns": final,
                    "ue_datagrams_observed": len(chunks),
                }
                rows.append(
                    {
                        "frame_id": frame_id, "route_tick": index * 2 + 1,
                        "capture_id": f"rehearsal:{frame_id}",
                        "action_id": self._state["action_id"],
                        "carla_timestamp": 100.0 + index * 0.1,
                        "capture_wall_s": first / 1e9,
                        "prepare_status": "SENT", "queue_wait_ms": 1.0,
                        "sensor_wait_ms": 2.0, "radar_window_ms": 0.5,
                        "radar_prepare_ms": 40.0, "rgb_convert_ms": 2.0,
                        "scene_snapshot_ms": 1.0, "pre_front_compute_ms": 48.0,
                        "window_callbacks": 4, "window_returns": 900,
                        "ego_speed_mps": 6.0, "front_ms": 45.0,
                        "payload_bytes": len(payload),
                        "payload_bytes_uncompressed": len(payload) - 176,
                        "payload_chunks": len(chunks),
                    }
                )
                time.sleep(0.12)
            rows.append({"frame_id": 1, "prepare_status": "STALE_BEFORE_SEND"})
            rows.append(
                {"frame_id": 2, "prepare_status": "DROPPED_INCOMPLETE_RADAR_WINDOW"}
            )
            time.sleep(1.5)
        finally:
            sender.close()
        (artifacts / "collector_rows.jsonl").write_text(
            "\n".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
            )
            + "\n",
            encoding="utf-8",
        )
        transmitted = sum(1 for row in rows if row["prepare_status"] == "SENT")
        common.atomic_create_json(
            artifacts / "diagnostic_summary.json",
            {
                "transmitted_budget": budget,
                "safety_timeout_s": live_runner.SAFETY_TIMEOUT_S,
                "transmitted_frames": transmitted,
                "reached_budget": transmitted >= budget,
                "stop_reason": "TRANSMITTED_BUDGET_REACHED",
                "preparation_opportunities_dropped": 2,
                "route_ticks_observed": len(rows) * 2,
                "route_started_wall_ns": 0, "budget_reached_wall_ns": 0,
                "route_stopped_wall_ns": 0, "route_wall_seconds": 4.2,
                "transport_counters": {}, "evaluation_scaffolding": {},
                "send_boundaries": boundaries, "failures": [], "cleanup_ok": True,
            },
        )
        common.atomic_create_json(
            artifacts / "child_result.json",
            {
                "cell_id": self._state["cell_id"], "action_id": self._state["action_id"],
                "route_accepted_by_campaign_gate": False,
                "route_detail": {"error": "RouteBudgetReached: TRANSMITTED_BUDGET_REACHED"},
                "stop_reason": "TRANSMITTED_BUDGET_REACHED",
                "transmitted_frames": transmitted, "failures": [], "cleanup_ok": True,
                "error": "", "map_process_started": True, "map_process_stopped": True,
                "rehearsal": True,
            },
        )
        return subprocess.CompletedProcess(values, 0)


def _install_stubs(state: dict[str, Any], payloads: dict[int, dict[int, bytes]]) -> None:
    def start_radio(campaign, *, cell_id, service_log_dir):
        namespace = Path(service_log_dir) / "rehearsal_radio"
        radio_state = namespace / "00_REHEARSAL"
        radio_state.mkdir(parents=True, exist_ok=False)
        return namespace, radio_state, {
            "status": "ATTACHED_STABLE_100MHZ_4D5U_ONE_UE",
            "clean_noise_preflight": {
                "verified": True, "noise_power_db": common.CLEAN_NOISE_POWER_DB,
            },
            "rehearsal": True,
        }, {"rehearsal": True}

    def stop_radio(base, namespace, radio_state, attached, *, actuator_restore_verified):
        return {
            "final_restore": {"noise_power_db": common.CLEAN_NOISE_POWER_DB},
            "final_restore_status": "REHEARSAL_NOT_ACTUATED",
            "teardown": {"all_lifecycle_gates_passed": True, "rehearsal": True},
        }

    class _StubLifecycle:
        @staticmethod
        def start_carla(port: int, log_path: Path):
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text("rehearsal: no CARLA\n", encoding="utf-8")
            return object(), 0

        @staticmethod
        def wait_for_rpc(port: int, timeout_s: float) -> str:
            return "rehearsal-no-carla"

        @staticmethod
        def stop_carla(server, pgid, port):
            return {"shutdown_verified": True, "rehearsal": True}

        @staticmethod
        def child_env() -> dict[str, str]:
            import os

            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            return env

    def start_actuator(campaign=None, *, campaign_path, profile_id, temporary_dir, start_file):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return process, Path(temporary_dir) / "radio_trace.csv", Path(temporary_dir) / "stop"

    def stop_actuator(process, output, stop_file):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return True

    def start_edge_container(*, campaign, campaign_path, profile, warmup_payloads,
                            temporary_dir, run_id, cell_id, warmup_stream_id=""):
        state_root = (Path(temporary_dir) / runner.EDGE_EVIDENCE_LEAF).resolve()
        state_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        blob = state_root / "warmup_payloads.bin"
        offsets: list[int] = []
        with blob.open("xb") as handle:
            for payload in warmup_payloads:
                offsets.append(handle.tell())
                handle.write(payload)
        common.atomic_create_json(
            state_root / "warmup_index.json",
            {
                "count": len(warmup_payloads), "offsets": offsets,
                "sizes": [len(payload) for payload in warmup_payloads],
            },
        )
        ready = state_root / "ready.json"
        stop_host = state_root / "stop_edge"
        log = (Path(temporary_dir) / "edge_service.log").open("wb")
        process = subprocess.Popen(
            [
                sys.executable, "-m", runner.EDGE_MODULE, "--diagnostic-edge",
                "--config", str(campaign_path),
                "--action-id", str(profile.action_id),
                "--allowed-action-ids", str(profile.action_id),
                "--ready-file", str(ready),
                "--records-file", str(state_root / "edge_records.jsonl"),
                "--summary-file", str(state_root / "edge_summary.json"),
                "--warmup-payload", str(blob), "--stop-file", str(stop_host),
                "--warmup-iterations", str(len(warmup_payloads)),
                "--warmup-stream-id", str(warmup_stream_id),
                "--edge-port", str(EDGE_PORT), "--result-host", LOOPBACK,
                "--result-port", str(RESULT_PORT), "--run-id", run_id,
                "--cell-id", cell_id,
            ],
            cwd=str(common.ROOT), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=_StubLifecycle.child_env(),
        )
        state["edge_process"] = process
        state["edge_log"] = log
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise live_runner.DiagnosticError(
                    "rehearsal edge exited before readiness; tail="
                    + runner._bounded_tail(Path(temporary_dir) / "edge_service.log")
                )
            if ready.is_file():
                document = common.load_json(ready)
                equivalence = document.get("parent_equivalence") or {}
                live_runner.require(
                    equivalence.get("perception_bitwise_identical") is True,
                    "rehearsal edge did not prove tail equivalence",
                )
                return state_root, {
                    "ready": document,
                    "records_host": str(state_root / "edge_records.jsonl"),
                    "summary_host": str(state_root / "edge_summary.json"),
                    "stop_host": str(stop_host),
                }
            time.sleep(0.25)
        raise live_runner.DiagnosticError("rehearsal edge did not become ready")

    def edge_running() -> bool:
        process = state.get("edge_process")
        return process is not None and process.poll() is None

    def stop_edge_container() -> bool:
        process = state.get("edge_process")
        if process is None:
            return True
        if process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        log = state.pop("edge_log", None)
        if log is not None:
            log.close()
        state["edge_process"] = None
        return True

    adapter = live_runner._adapter()
    adapter.start_target_snr = start_actuator
    adapter.stop_target_snr = stop_actuator
    live_runner.runner.start_radio = start_radio
    live_runner.runner.stop_radio = stop_radio
    live_runner.runner.RadioTelemetry = _StubTelemetry
    live_runner.runner.start_edge_container = start_edge_container
    live_runner.runner._edge_running = edge_running
    live_runner.runner._stop_edge_container = stop_edge_container
    live_runner.runner.inspect_edge_mounts = lambda state_root: {"rehearsal": True}
    live_runner._lifecycle = lambda campaign: _StubLifecycle
    live_runner.subprocess = _ChildShim({}, state)
    state["child_shim_payloads"] = payloads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=REHEARSAL_FRAMES)
    parser.add_argument(
        "--actions", default=",".join(str(value) for value in REHEARSAL_ACTIONS)
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = Path(args.output).resolve()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    (output / "per_frame").mkdir()

    campaign_path = common.repo_path(common.PILOT_CONFIG_RELPATH)
    campaign = json.loads(json.dumps(common.load_json(campaign_path)))
    campaign["runtime"]["ue_bind_host"] = LOOPBACK
    campaign["runtime"]["edge_remote_host"] = LOOPBACK
    campaign["runtime"]["edge_receive_port"] = EDGE_PORT
    campaign["runtime"]["camera_result_port"] = RESULT_PORT
    campaign["runtime"]["edge_source_port"] = EDGE_SOURCE_PORT
    campaign["runtime"]["map_ingest_port"] = MAP_PORT

    device = torch.device("cuda:0")
    ue, _ledger, _models, _base, _registry = preload_ue(device)
    registry = SplitActionRegistry.from_runtime_binding()

    state: dict[str, Any] = {}
    _install_stubs(state, {})

    frames = int(args.frames)
    reports = []
    for action_id in (int(value) for value in str(args.actions).split(",")):
        profile = registry.resolve(action_id)
        generator = torch.Generator(device="cpu").manual_seed(4242)
        base = torch.randn((1, 7, 448, 768), generator=generator, dtype=torch.float32)
        stream = f"ue288_live_a{action_id:02d}__favorable_stable"
        payloads: dict[int, bytes] = {}
        anchor = time.time_ns()
        for index in range(frames):
            frame_id = FIRST_FRAME_ID + index
            capture = anchor + index * 100_000_000
            with torch.inference_mode():
                prepared = ue.prepare(
                    profile.action_id, base.to(device), sequence_id=frame_id,
                    capture_timestamp_ns=capture,
                    frame_context=build_frame_context_v1(
                        stream_id=stream, frame_id=frame_id, sequence_id=frame_id,
                        capture_timestamp_ns=capture, ego_world_x=1.0,
                        ego_world_y=2.0, ego_world_z=0.5, ego_world_pitch=0.0,
                        ego_world_yaw=10.0, ego_world_roll=0.0,
                    ),
                )
            payloads[frame_id] = bytes(prepared.wire_bytes)
        cell_id = f"live_a{action_id:02d}__{common.NETWORK_PROFILE_ID.lower()}"
        live_runner.subprocess = _ChildShim(
            payloads, {"cell_id": cell_id, "action_id": action_id}
        )
        live_runner.TRANSMITTED_BUDGET = frames
        reports.append(
            live_runner.run_live_action(
                action_id=action_id, campaign=campaign, campaign_path=campaign_path,
                ue=ue, registry=registry, run_id=output.name,
            )
        )
        print(f"  rehearsed live action {action_id}", flush=True)

    summaries = [live_runner.summarize_live_action(report) for report in reports]
    comparisons = live_runner.build_live_comparisons(summaries)
    for report in reports:
        name = f"action_{report['action_id']:02d}_{report['profile_id']}.csv"
        live_runner.write_live_per_frame_csv(output / "per_frame" / name, report)
    live_runner.write_live_summary_csv(output / "action_summary.csv", summaries)
    live_runner.write_live_figure(output / "live_timing_breakdown", summaries, comparisons)
    live_runner.write_live_report(
        output / "REPORT.md",
        manifest={
            "run_id": output.name, "git": {"head": "rehearsal"},
            "environment": {"device_name": common.DEVICE_NAME},
            "route_b": {
                "route_json": "rehearsal", "scenario_seed": 31,
                "traffic_manager_seed": 31,
            },
        },
        summaries=summaries, comparisons=comparisons, runtime_seconds=0.0,
    )
    runner.write_artifact_manifest(output)
    print("LIVE REHEARSAL COMPLETE — parent flow and child handoff executed")
    for report in reports:
        counts = live_runner.summarize_live_action(report)["counts"]
        print(
            f"  a{report['action_id']:02d}: transmitted={counts['transmitted_frames']} "
            f"delivered={counts['delivered_and_measured']} "
            f"reassembled={counts['complete_reassemblies']} "
            f"tail={counts['tail_completions']} "
            f"drops={counts['prepare_status_counts']} "
            f"child_rc={report['child']['returncode']} "
            f"scratch_removed={report['cell_scratch_removed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
