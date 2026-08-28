#!/usr/bin/env python3
"""Dedicated namespace-clean subprocess for registered validation scoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scoring_v2 import score_primary, score_sensitivity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("primary", "sensitivity"))
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    if sys.executable != "/usr/bin/python3":
        raise RuntimeError(f"required /usr/bin/python3, got {sys.executable}")
    if args.mode == "primary":
        if args.checkpoint is None or args.checkpoint_sha256 is None or args.epoch is None:
            raise ValueError("primary scoring requires checkpoint, hash, and epoch")
        result = score_primary(
            args.experiment.resolve(), args.prediction_root.resolve(),
            args.checkpoint.resolve(), args.checkpoint_sha256, args.epoch,
        )
    else:
        result = score_sensitivity(args.experiment.resolve(), args.prediction_root.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "mode": args.mode, "output": str(args.output),
        "epoch": args.epoch, "all_metrics_finite": result["all_metrics_finite"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
