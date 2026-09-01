from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .contract import load_revised_selector, verify_revised_holdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify frozen hashes and reproduce the revised train-holdout frontier point",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only verification output already exists: {output}")
    device = torch.device("cpu")
    runtime = load_revised_selector(device, require_holdout_verification=False)
    report = verify_revised_holdout(runtime, device)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
