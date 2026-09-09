#!/usr/bin/env python3
"""Offline rehearsal of one full diagnostic action, radio excluded.

Every live failure of this diagnostic so far sat in a *seam*: the preflight
ordering, the edge shutdown handshake, the cold proof versus the cell's own
scratch. Component tests could not see them because they never ran
``run_action`` end to end.

This harness runs the real :func:`runner.run_action` verbatim and substitutes
only the four things that require the radio host: the qualified OAI launcher,
the radio teardown, the target-SNR actuator and the T-tracer collector. The
edge is the real ``edge_service`` module, started as a local subprocess
against loopback instead of inside the container, so the real warm-up,
equivalence proof, reassembly, queue, tail decomposition, graceful-shutdown
handshake, record join, cold proof, microbenchmark, summary, comparison and
evidence writers all execute exactly as they do live.

It produces **no scientific evidence**. Frame counts are reduced, the
transport is loopback and no radio is actuated, so its outputs are a
plumbing proof only and are written to a scratch directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry

from . import diagnostic_common as common
from . import runner
from .edge_preload import preload_instrumented_edge, preload_ue


REHEARSAL_ACTIONS = (50, 71)
REHEARSAL_FRAMES = 12
LOOPBACK_HOST = "127.0.0.1"
EDGE_PORT = 51902
RESULT_PORT = 51904
EDGE_SOURCE_PORT = 51913
MAP_PORT = 51910


class _StubTelemetry:
    """Stands in for the T-tracer collector; collects nothing."""

    def __init__(self, base: Mapping[str, Any], scratch: Path) -> None:
        self.status = "REHEARSAL_NOT_COLLECTED"
        self.error = ""

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        empty = common.summarize([])
        return {
            "status": self.status, "error": self.error,
            "pusch_samples": 0, "mcs_samples": 0,
            "achieved_pusch_snr_db": empty, "achieved_pusch_mcs": empty,
            "scheduler_avg_snr_db": empty, "scheduler_selected_ul_mcs": empty,
            "scheduler_final_ul_mcs": empty, "raw_tracer_rows_retained": False,
        }


def _install_stubs(state: dict[str, Any]) -> None:
    """Replace only the radio-host boundaries; everything else stays real."""

    def start_radio(campaign, *, cell_id, service_log_dir):
        namespace = Path(service_log_dir) / "rehearsal_radio"
        radio_state = namespace / "00_REHEARSAL"
        radio_state.mkdir(parents=True, exist_ok=False)
        attached = {
            "status": "ATTACHED_STABLE_100MHZ_4D5U_ONE_UE",
            "clean_noise_preflight": {
                "verified": True, "noise_power_db": common.CLEAN_NOISE_POWER_DB,
            },
            "rehearsal": True,
        }
        return namespace, radio_state, attached, {"rehearsal": True}

    def stop_radio(base, namespace, radio_state, attached, *, actuator_restore_verified):
        return {
            "final_restore": {"noise_power_db": common.CLEAN_NOISE_POWER_DB, "verified": True},
            "final_restore_status": "REHEARSAL_NOT_ACTUATED",
            "final_restore_error": "",
            "actuator_restore_verified": bool(actuator_restore_verified),
            "teardown": {"all_lifecycle_gates_passed": True, "rehearsal": True},
        }

    def start_actuator(*, campaign_path, temporary_dir, start_file):
        # A trivial process that outlives the transmission, so run_action's
        # liveness assertion exercises its real branch.
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return process, Path(temporary_dir) / "radio_trace.csv", Path(temporary_dir) / "stop_target_snr"

    def stop_actuator(process, output, stop_file):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        empty = common.summarize([])
        return {
            "summary": {"rehearsal": True}, "rows": 0, "commands_applied": 0,
            "obsolete_command_skips": 0, "late_command_acks": 0,
            "profile_id": common.NETWORK_PROFILE_ID, "trace_id": "REHEARSAL",
            "seed": 0, "sample_period_ms": 100,
            "target_snr_db": empty, "first_300_target_snr_db": empty,
            "first_300_target_digest": "", "mapped_rfsim_command_db": empty,
            "command_latency_ms": empty, "clean_restore_verified": True,
            "returncode": 0,
        }

    def start_edge_container(*, campaign, campaign_path, profile, warmup_payloads,
                            temporary_dir, run_id, cell_id):
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
        records = state_root / "edge_records.jsonl"
        summary = state_root / "edge_summary.json"
        stop_host = state_root / "stop_edge"
        log = (Path(temporary_dir) / "edge_service.log").open("wb")
        process = subprocess.Popen(
            [
                sys.executable, "-m", runner.EDGE_MODULE, "--diagnostic-edge",
                "--config", str(campaign_path),
                "--action-id", str(profile.action_id),
                "--allowed-action-ids", str(profile.action_id),
                "--ready-file", str(ready), "--records-file", str(records),
                "--summary-file", str(summary), "--warmup-payload", str(blob),
                "--stop-file", str(stop_host),
                "--warmup-iterations", str(len(warmup_payloads)),
                "--first-measured-sequence-id", str(runner.FIRST_MEASURED_SEQUENCE_ID),
                "--edge-port", str(EDGE_PORT), "--result-host", LOOPBACK_HOST,
                "--result-port", str(RESULT_PORT), "--run-id", run_id,
                "--cell-id", cell_id,
            ],
            cwd=str(common.ROOT), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT,
        )
        state["edge_process"] = process
        state["edge_log"] = log
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise runner.DiagnosticError(
                    "rehearsal edge exited before readiness; tail="
                    + runner._bounded_tail(Path(temporary_dir) / "edge_service.log")
                )
            if ready.is_file():
                document = common.load_json(ready)
                equivalence = document.get("parent_equivalence") or {}
                runner.require(
                    equivalence.get("perception_bitwise_identical") is True
                    and equivalence.get("service_records_byte_identical") is True,
                    "rehearsal edge did not prove tail equivalence",
                )
                return state_root, {
                    "ready": document, "records_host": str(records),
                    "summary_host": str(summary), "stop_host": str(stop_host),
                }
            time.sleep(0.25)
        raise runner.DiagnosticError("rehearsal edge did not become ready")

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

    runner.start_radio = start_radio
    runner.stop_radio = stop_radio
    runner.RadioTelemetry = _StubTelemetry
    runner._start_actuator = start_actuator
    runner._stop_actuator = stop_actuator
    runner.start_edge_container = start_edge_container
    runner._edge_running = edge_running
    runner._stop_edge_container = stop_edge_container
    runner.inspect_edge_mounts = lambda state_root: {"rehearsal": True}


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

    campaign = common.load_json(common.repo_path(common.PILOT_CONFIG_RELPATH))
    campaign_path = common.repo_path(common.PILOT_CONFIG_RELPATH)
    # Loopback transport in place of the UE tunnel and the container network.
    campaign = json.loads(json.dumps(campaign))
    campaign["runtime"]["ue_bind_host"] = LOOPBACK_HOST
    campaign["runtime"]["edge_remote_host"] = LOOPBACK_HOST
    campaign["runtime"]["edge_receive_port"] = EDGE_PORT
    campaign["runtime"]["camera_result_port"] = RESULT_PORT
    campaign["runtime"]["edge_source_port"] = EDGE_SOURCE_PORT
    campaign["runtime"]["map_ingest_port"] = MAP_PORT

    sample, context = runner.construct_registered_sample()
    frames = int(args.frames)
    runner.require(1 <= frames <= common.FRAMES, "rehearsal frame count is out of range")
    sample = {**sample, "selected_rows": sample["selected_rows"][:frames]}
    common.FRAMES = frames

    device = torch.device("cuda:0")
    ue, _ledger, _models, base, _registry = preload_ue(device)
    host_edge = preload_instrumented_edge(device)
    inference = base.data.InferenceDataset(context["dataset_root"], "train")
    registry = SplitActionRegistry.from_runtime_binding()

    state: dict[str, Any] = {}
    _install_stubs(state)

    reports = []
    try:
        for action_id in (int(value) for value in str(args.actions).split(",")):
            reports.append(
                runner.run_action(
                    action_id=action_id, campaign=campaign, campaign_path=campaign_path,
                    sample=sample, ue=ue, inference=inference, host_edge=host_edge,
                    registry=registry, run_id=output.name,
                )
            )
            print(f"  rehearsed action {action_id}", flush=True)
    finally:
        runner._stop_edge_container()

    summaries = [runner.summarize_action(report) for report in reports]
    comparisons = runner.build_comparisons(summaries)
    for report in reports:
        name = f"action_{report['action_id']:02d}_{report['profile_id']}.csv"
        runner.write_per_frame_csv(output / "per_frame" / name, report)
    runner.write_action_summary_csv(output / "action_summary.csv", summaries)
    runner.write_figure(output / "timing_breakdown", summaries, comparisons)
    common.atomic_create_json(
        output / "DIAGNOSTIC_RESULTS.json",
        {
            "schema": common.SCHEMA, "rehearsal": True,
            "comparisons": comparisons, "action_summaries": summaries,
        },
    )
    runner.write_report(
        output / "REPORT.md",
        manifest={
            "run_id": output.name, "git": {"head": "rehearsal"},
            "environment": {"device_name": common.DEVICE_NAME},
            "sample": {"sample_manifest_sha256": sample["sample_manifest_sha256"]},
        },
        summaries=summaries, comparisons=comparisons, runtime_seconds=0.0,
    )
    runner.write_artifact_manifest(output)
    print("REHEARSAL COMPLETE — every seam outside the radio host executed")
    for report in reports:
        counts = runner.summarize_action(report)["counts"]
        print(
            f"  a{report['action_id']:02d}: sent={counts['messages_sent']} "
            f"reassembled={counts['complete_reassemblies']} "
            f"tail={counts['tail_completions']} "
            f"results={counts['compact_results_returned_to_ue']} "
            f"microbench={counts['microbenchmark_observations']} "
            f"scratch_removed={report['cell_scratch_removed']} "
            f"cold_after={report['cold_after']['label']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
