"""Command-line entry point for the registered D2b mechanics-only smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .empirical_contextual_smoke_runner import run_registered_three_seed_smoke


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered mechanics-only empirical SAC smoke."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; written atomically after success.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_registered_three_seed_smoke()
    rendered = json.dumps(report.to_canonical_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
