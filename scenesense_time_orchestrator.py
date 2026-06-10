#!/usr/bin/env python3

"""Timer-based SceneSense task gate controller.

This is intentionally simple: it writes a JSON control file that split-
inference clients can read before sending feature payloads. Later, an RL agent
can replace the timer policy while keeping the same control-file interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timer-based OD/SEG task gate.")
    parser.add_argument(
        "--control-file",
        default="/tmp/scenesense_task_gate.json",
        help="JSON file read by OD/SEG split-inference clients.",
    )
    parser.add_argument("--od-seconds", type=float, default=10.0)
    parser.add_argument("--seg-seconds", type=float, default=5.0)
    parser.add_argument(
        "--startup-task",
        choices=("od", "seg"),
        default="od",
        help="Task to activate first in the repeating schedule.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after this many seconds. Use 0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.25,
        help="How often to refresh the active gate file.",
    )
    parser.add_argument(
        "--profile",
        default="baseline",
        help="Profile label written into the control file for later extensions.",
    )
    parser.add_argument(
        "--log-csv",
        default="",
        help="Optional switch-event CSV. Defaults beside the control file.",
    )
    return parser.parse_args()


def schedule_from_args(args: argparse.Namespace) -> List[Tuple[str, float]]:
    first = (args.startup_task, float(args.od_seconds if args.startup_task == "od" else args.seg_seconds))
    second_task = "seg" if args.startup_task == "od" else "od"
    second = (second_task, float(args.seg_seconds if second_task == "seg" else args.od_seconds))
    return [first, second]


def active_for_elapsed(schedule: Sequence[Tuple[str, float]], elapsed_s: float) -> Tuple[str, float, float]:
    cycle = sum(max(0.001, duration) for _, duration in schedule)
    offset = elapsed_s % cycle
    cursor = 0.0
    for task, duration in schedule:
        duration = max(0.001, float(duration))
        if offset < cursor + duration:
            return task, offset - cursor, duration
        cursor += duration
    task, duration = schedule[-1]
    return task, duration, duration


def atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    control_file = Path(args.control_file).expanduser()
    log_csv = (
        Path(args.log_csv).expanduser()
        if args.log_csv
        else control_file.with_name(control_file.stem + "_events.csv")
    )
    schedule = schedule_from_args(args)
    stop = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    log_csv.parent.mkdir(parents=True, exist_ok=True)
    log_exists = log_csv.exists()
    log_fh = log_csv.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_fh,
        fieldnames=[
            "wall_time_iso",
            "elapsed_s",
            "active_task",
            "task_elapsed_s",
            "task_duration_s",
            "profile",
            "control_file",
        ],
    )
    if not log_exists:
        writer.writeheader()

    started = time.time()
    last_task = ""
    try:
        while not stop:
            elapsed = time.time() - started
            if args.duration_s > 0.0 and elapsed >= float(args.duration_s):
                break
            active_task, task_elapsed, task_duration = active_for_elapsed(schedule, elapsed)
            payload = {
                "active_task": active_task,
                "profile": str(args.profile),
                "updated_at": time.time(),
                "elapsed_s": elapsed,
                "task_elapsed_s": task_elapsed,
                "task_duration_s": task_duration,
                "policy": "timer",
                "schedule": [{"task": task, "duration_s": duration} for task, duration in schedule],
            }
            atomic_write_json(control_file, payload)
            if active_task != last_task:
                writer.writerow(
                    {
                        "wall_time_iso": datetime.now().isoformat(timespec="milliseconds"),
                        "elapsed_s": f"{elapsed:.3f}",
                        "active_task": active_task,
                        "task_elapsed_s": f"{task_elapsed:.3f}",
                        "task_duration_s": f"{task_duration:.3f}",
                        "profile": str(args.profile),
                        "control_file": str(control_file),
                    }
                )
                log_fh.flush()
                print(
                    f"[{elapsed:8.3f}s] active_task={active_task} "
                    f"for {task_duration:.1f}s profile={args.profile}",
                    flush=True,
                )
                last_task = active_task
            time.sleep(max(0.05, float(args.poll_interval_s)))
    finally:
        log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

