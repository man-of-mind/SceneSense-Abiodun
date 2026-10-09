from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from common import atomic_json, atomic_text, utc_now

PACKAGE = Path(__file__).resolve().parent
EPOCHS = (3, 8, 16, 22, 26)


def run(script: str, experiment: Path, *extra: str) -> None:
    command = [sys.executable, str(PACKAGE / script), "--experiment", str(experiment), *extra]
    print(json.dumps({"launch": command, "created_utc": utc_now()}), flush=True)
    subprocess.run(command, cwd=PACKAGE, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve()
    try:
        if not (experiment / "PREREGISTRATION_COMPLETE").exists(): run("preregister.py", experiment)
        if not (experiment / "STRUCTURAL_QUALIFICATION_COMPLETE").exists(): run("qualify.py", experiment)
        if not (experiment / "P2_ASSIGNMENT_AUDIT_COMPLETE").exists(): run("assignment_audit.py", experiment)
        if not (experiment / "QUALIFICATION_COMPLETE").exists(): run("train.py", experiment, "--mode", "qualification")
        if not (experiment / "TRAINING_COMPLETE").exists():
            extra = ("--mode", "scientific", "--resume") if (experiment / "SCIENTIFIC_TRAINING_STARTED.json").exists() else ("--mode", "scientific")
            run("train.py", experiment, *extra)
        for epoch in EPOCHS:
            if not (experiment / f"predictions/epoch_{epoch:03d}/INFERENCE_COMPLETE").exists():
                run("infer.py", experiment, "--epoch", str(epoch))
        if not (experiment / "EVALUATION_COMPLETE").exists(): run("evaluate.py", experiment)
        if not (experiment / "COMPLETION_SENTINEL").exists(): run("report.py", experiment)
        return 0
    except Exception as error:
        experiment.mkdir(parents=True, exist_ok=True)
        atomic_json(experiment / "PIPELINE_FAILURE.json", {"created_utc": utc_now(), "error": repr(error),
                                                            "traceback": traceback.format_exc()}, overwrite=True)
        atomic_text(experiment / "STATUS_FAILURE", "PIPELINE_FAILURE_SEE_JSON\n", overwrite=True)
        raise


if __name__ == "__main__": raise SystemExit(main())

