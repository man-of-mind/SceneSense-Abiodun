#!/usr/bin/env python3
"""Supervisor for the one licensed visible-anchor scientific experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from common_v1 import sha256, utc_now, write_json_x  # noqa: E402


def run_logged(command: list[str], log: Path, *, defer_log_creation: bool = False) -> int:
    if defer_log_creation:
        # Preflight owns creation of the create-only timestamp directory. Buffer its
        # sparse progress output until that directory exists.
        process = subprocess.Popen(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        lines: list[str] = []
        for line in process.stdout:
            lines.append(line); print(line, end="", flush=True)
        code = process.wait()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as stream:
            stream.write("COMMAND " + json.dumps(command) + "\n")
            stream.writelines(lines)
        return code
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n"); stream.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line); stream.flush()
            print(line, end="", flush=True)
        return process.wait()


def notify(experiment: Path, terminal: str) -> None:
    command = ["notify-send", "LR-ASPP visible-anchor experiment complete",
               f"{terminal}\n{experiment}"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    write_json_x(experiment / "NOTIFICATION.json", {
        "created_utc": utc_now(), "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment = ROOT / "experiments/route_b_v3_1_person_visible_anchor_v1" / timestamp
    config = PACKAGE / "configs/person_visible_anchor_v1.json"
    started = time.monotonic()
    print(f"[pipeline] experiment={experiment}", flush=True)
    code = run_logged([
        sys.executable, str(PACKAGE / "preflight_v1.py"), "--experiment", str(experiment),
        "--config", str(config),
    ], experiment / "logs/preflight.log", defer_log_creation=True)
    if code != 0:
        print(f"[pipeline] preflight failed rc={code}; no optimizer step launched", flush=True)
        return code
    code = run_logged([
        sys.executable, str(PACKAGE / "train_v1.py"), "--experiment", str(experiment),
        "--end-epoch", "12",
    ], experiment / "logs/train_epochs_001_012.log")
    if code != 0:
        print(f"[pipeline] training stage 1 failed rc={code}", flush=True)
        return code
    code = run_logged([
        sys.executable, str(PACKAGE / "evaluate_v1.py"), "--experiment", str(experiment),
        "--phase", "catastrophic",
    ], experiment / "logs/epoch12_catastrophic_gate.log")
    if code != 0:
        print("[pipeline] epoch-12 catastrophic gate stopped the schedule", flush=True)
        return code
    latest = json.loads((experiment / "LATEST_SAFE.json").read_text())
    if int(latest["epoch"]) != 12:
        raise RuntimeError("epoch-12 recovery boundary missing")
    code = run_logged([
        sys.executable, str(PACKAGE / "train_v1.py"), "--experiment", str(experiment),
        "--end-epoch", "24", "--resume-checkpoint", latest["path"],
        "--resume-sha256", latest["sha256"],
    ], experiment / "logs/train_epochs_013_024.log")
    if code != 0:
        print(f"[pipeline] training stage 2 failed rc={code}", flush=True)
        return code
    code = run_logged([
        sys.executable, str(PACKAGE / "evaluate_v1.py"), "--experiment", str(experiment),
        "--phase", "final",
    ], experiment / "logs/final_evaluation.log")
    if code != 0:
        print(f"[pipeline] final evaluation failed rc={code}", flush=True)
        return code
    code = run_logged([
        sys.executable, str(PACKAGE / "report_v1.py"), "--experiment", str(experiment),
        "--pipeline-wall-seconds", str(time.monotonic() - started),
    ], experiment / "logs/report.log")
    if code != 0:
        return code
    terminal = (experiment / "TERMINAL_VERDICT.txt").read_text().strip()
    notify(experiment, terminal)
    print(json.dumps({
        "terminal": terminal, "experiment": str(experiment),
        "completion_sentinel": str(experiment / "COMPLETION_SENTINEL"),
        "pipeline_wall_seconds": time.monotonic() - started,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
