"""Command-line entry point for the registered D2b mechanics-only smoke."""

from __future__ import annotations

import json

from .empirical_contextual_smoke_runner import run_registered_three_seed_smoke


def main() -> int:
    report = run_registered_three_seed_smoke()
    print(json.dumps(report.to_canonical_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
