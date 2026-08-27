#!/usr/bin/env python3
"""Create a resolved create-only trial JSON for one phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", required=True, type=int, choices=(16, 24))
    parser.add_argument("--run-until-epoch", required=True, type=int, choices=(4, 24))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    trial = json.loads(args.source.read_text(encoding="utf-8"))
    trial["batch_size"] = int(args.batch_size)
    trial["run_until_epoch"] = int(args.run_until_epoch)
    args.output.write_text(json.dumps(trial, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

