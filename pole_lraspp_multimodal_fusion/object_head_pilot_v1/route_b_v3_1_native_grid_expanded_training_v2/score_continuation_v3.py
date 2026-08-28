#!/usr/bin/env python3
"""Namespace-clean scoring subprocess for continuation checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from continuation_scoring_v3 import (
    baseline_primary_with_errors, primary_with_errors, sensitivity_with_errors,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("primary", "sensitivity", "baseline-primary"))
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--amended-baseline", type=Path)
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    experiment = args.experiment.resolve()
    prediction_root = args.prediction_root.resolve()
    if args.mode == "primary":
        if args.checkpoint is None or args.checkpoint_sha256 is None or args.epoch is None:
            raise ValueError("primary scoring requires checkpoint, SHA-256, and epoch")
        result = primary_with_errors(
            experiment, prediction_root, args.checkpoint.resolve(),
            args.checkpoint_sha256, args.epoch,
        )
    elif args.mode == "sensitivity":
        result = sensitivity_with_errors(experiment, prediction_root)
    else:
        if args.amended_baseline is None:
            raise ValueError("baseline-primary scoring requires amended baseline")
        result = baseline_primary_with_errors(
            experiment, args.amended_baseline.resolve(), prediction_root,
        )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "mode": args.mode,
        "output": str(args.output),
        "epoch": args.epoch,
        "all_metrics_finite": result.get("all_metrics_finite", True),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
