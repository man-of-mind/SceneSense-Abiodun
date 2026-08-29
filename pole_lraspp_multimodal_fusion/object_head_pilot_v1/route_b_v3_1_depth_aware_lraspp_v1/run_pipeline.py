from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("[pipeline]", " ".join(command), flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(f"pipeline command failed ({completed.returncode}): {command}")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args(); experiment = args.experiment.resolve(strict=True)
    python = sys.executable
    if not (experiment / "depth_cache/train/CACHE_COMPLETE").is_file():
        run([python, str(PACKAGE / "build_depth_cache.py"), "--experiment", str(experiment), "--split", "train"])
    if not (experiment / "QUALIFICATION_COMPLETE").is_file():
        run([python, str(PACKAGE / "qualify.py"), "--experiment", str(experiment)])
    if not (experiment / "TRAINING_COMPLETE").is_file():
        run([python, str(PACKAGE / "train.py"), "--experiment", str(experiment)])
    for epoch in (10, 20, 30, 40):
        if not (experiment / f"predictions/epoch_{epoch:03d}/INFERENCE_COMPLETE").is_file():
            run([python, str(PACKAGE / "infer.py"), "--experiment", str(experiment), "--epoch", str(epoch)])
    if not (experiment / "depth_cache/val/CACHE_COMPLETE").is_file():
        run([python, str(PACKAGE / "build_depth_cache.py"), "--experiment", str(experiment), "--split", "val"])
    if not (experiment / "EVALUATION_COMPLETE").is_file():
        run([python, str(PACKAGE / "evaluate.py"), "--experiment", str(experiment)])
    if not (experiment / "PIPELINE_COMPLETE.json").is_file():
        run([python, str(PACKAGE / "finalize.py"), "--experiment", str(experiment)])
    return 0


if __name__ == "__main__": raise SystemExit(main())
