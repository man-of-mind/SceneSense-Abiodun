#!/usr/bin/env python3

"""Build or launch timer-gated SceneSense OD/SEG split-inference runs."""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = "/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3"
DEFAULT_CONTROL_FILE = "/tmp/scenesense_task_gate.json"
DEFAULT_METRICS_DIR = ROOT / "metrics_logs" / "dual_task_orchestrator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or launch a timer-gated OD/SEG split-inference run. "
            "Default mode is dry-run so the commands can be inspected first."
        )
    )
    parser.add_argument("--run", action="store_true", help="Launch the processes.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--control-file", default=DEFAULT_CONTROL_FILE)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument(
        "--orchestrator-extra-s",
        type=float,
        default=90.0,
        help=(
            "Keep the timer gate alive this many seconds beyond --duration-s. "
            "The launcher terminates it early once OD and SEG clients exit."
        ),
    )
    parser.add_argument("--od-seconds", type=float, default=10.0)
    parser.add_argument("--seg-seconds", type=float, default=5.0)
    parser.add_argument("--startup-task", choices=("od", "seg"), default="od")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--metrics-log-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--run-tag-prefix", default="dual_task_timer")
    parser.add_argument("--camera-resolution", default="720p")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--npc-vehicles", type=int, default=20)
    parser.add_argument("--npc-pedestrians", type=int, default=30)
    parser.add_argument("--town", default="")
    parser.add_argument("--weather-preset", default="unchanged")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--disable-live-plot", action="store_true", default=True)
    parser.add_argument("--od-port-base", type=int, default=36100)
    parser.add_argument("--seg-port-base", type=int, default=36200)
    parser.add_argument("--od-tm-port", type=int, default=8000)
    parser.add_argument("--seg-tm-port", type=int, default=8001)
    parser.add_argument(
        "--process-start-gap-s",
        type=float,
        default=1.0,
        help="Delay between launching orchestrator, OD, and SEG processes in --run mode.",
    )
    parser.add_argument(
        "--seg-mode",
        choices=("generic", "trained"),
        default="generic",
        help="Use generic torchvision SEG or pole-trained LR-ASPP SEG script.",
    )
    parser.add_argument(
        "--trained-experiment-dir",
        default="",
        help="Required when --seg-mode=trained unless --seg-extra-arg supplies weights.",
    )
    parser.add_argument(
        "--enable-semantic-gt",
        action="store_true",
        help="Enable CARLA semantic GT for SEG metrics.",
    )
    parser.add_argument(
        "--extra-od-arg",
        action="append",
        default=[],
        help="Additional OD argument. Repeat for multiple args.",
    )
    parser.add_argument(
        "--extra-seg-arg",
        action="append",
        default=[],
        help="Additional SEG argument. Repeat for multiple args.",
    )
    return parser.parse_args()


def common_client_args(args: argparse.Namespace, *, task: str, port_base: int) -> List[str]:
    tm_port = int(args.od_tm_port if task == "od" else args.seg_tm_port)
    cmd = [
        "--town",
        str(args.town),
        "--tm-port",
        str(tm_port),
        "--weather-preset",
        str(args.weather_preset),
        "--camera-resolution",
        str(args.camera_resolution),
        "--fps",
        str(float(args.fps)),
        "--npc-vehicles",
        str(int(args.npc_vehicles)),
        "--npc-pedestrians",
        str(int(args.npc_pedestrians)),
        "--run-duration-s",
        str(float(args.duration_s)),
        "--metrics-log-dir",
        str(Path(args.metrics_log_dir).expanduser()),
        "--camera-source-port",
        str(port_base),
        "--remote-port",
        str(port_base + 1),
        "--remote-source-port",
        str(port_base + 2),
        "--camera-result-port",
        str(port_base + 3),
        "--tx-gate-file",
        str(args.control_file),
        "--tx-task-name",
        task,
        "--tx-gate-default-inactive",
        "--tx-gate-stale-timeout-s",
        "2.0",
    ]
    if bool(args.headless):
        cmd.append("--headless")
    if bool(args.disable_live_plot):
        cmd.append("--disable-live-plot")
    return cmd


def build_commands(args: argparse.Namespace) -> List[List[str]]:
    py = str(args.python)
    metrics_dir = Path(args.metrics_log_dir).expanduser()
    orchestrator = [
        py,
        str(ROOT / "scenesense_time_orchestrator.py"),
        "--control-file",
        str(args.control_file),
        "--duration-s",
        str(float(args.duration_s) + max(0.0, float(args.orchestrator_extra_s))),
        "--od-seconds",
        str(float(args.od_seconds)),
        "--seg-seconds",
        str(float(args.seg_seconds)),
        "--startup-task",
        str(args.startup_task),
        "--profile",
        str(args.profile),
        "--log-csv",
        str(metrics_dir / f"{args.run_tag_prefix}_gate_events.csv"),
    ]

    od = [
        py,
        str(ROOT / "carla_split_inference_udp_data_collect.py"),
        *common_client_args(args, task="od", port_base=int(args.od_port_base)),
        "--metrics-log-prefix",
        f"{args.run_tag_prefix}_od",
        "--run-tag",
        f"{args.run_tag_prefix}_od",
        *args.extra_od_arg,
    ]

    seg_script = (
        ROOT / "carla_split_inference_udp_segmentation_trained_lraspp_demo.py"
        if args.seg_mode == "trained"
        else ROOT / "carla_split_inference_udp_segmentation_demo.py"
    )
    seg = [
        py,
        str(seg_script),
        *common_client_args(args, task="seg", port_base=int(args.seg_port_base)),
        "--metrics-log-prefix",
        f"{args.run_tag_prefix}_seg",
        "--run-tag",
        f"{args.run_tag_prefix}_seg",
        *args.extra_seg_arg,
    ]
    if args.seg_mode == "trained" and str(args.trained_experiment_dir or "").strip():
        seg.extend(["--trained-experiment-dir", str(args.trained_experiment_dir)])
    if bool(args.enable_semantic_gt):
        seg.append("--enable-semantic-gt")

    return [orchestrator, od, seg]


def print_commands(commands: Sequence[Sequence[str]]) -> None:
    labels = ("orchestrator", "od", "seg")
    for label, cmd in zip(labels, commands):
        print(f"\n# {label}")
        print(" ".join(shlex.quote(part) for part in cmd))


def run_processes(commands: Sequence[Sequence[str]], *, start_gap_s: float) -> int:
    processes: List[subprocess.Popen[bytes]] = []
    stopping = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        for cmd in commands:
            processes.append(subprocess.Popen(list(cmd)))
            time.sleep(max(0.0, float(start_gap_s)))
        while not stopping:
            running = [proc for proc in processes if proc.poll() is None]
            if not running:
                break
            clients_done = len(processes) >= 3 and all(proc.poll() is not None for proc in processes[1:])
            if clients_done:
                if processes[0].poll() is None:
                    processes[0].terminate()
                break
            time.sleep(0.5)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()

    exit_codes = [proc.returncode for proc in processes]
    client_codes = exit_codes[1:] if len(exit_codes) > 1 else exit_codes
    orchestrator_ok = exit_codes[0] in (0, -signal.SIGTERM) if exit_codes else True
    clients_ok = all(code == 0 for code in client_codes)
    return 0 if orchestrator_ok and clients_ok else 1


def main() -> int:
    args = parse_args()
    commands = build_commands(args)
    print_commands(commands)
    if not args.run:
        print("\nDry run only. Re-run with --run to launch these processes.")
        return 0
    return run_processes(commands, start_gap_s=float(args.process_start_gap_s))


if __name__ == "__main__":
    raise SystemExit(main())
