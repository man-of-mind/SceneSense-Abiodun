#!/usr/bin/env python3
"""Install the clean CenterNet builder and run the existing Route B trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent.parent, HERE.parent.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from centernet_model_v1 import install  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--training-budget-hours", type=float, default=0.0)
    args = parser.parse_args()

    install()
    from pole_lraspp_multimodal_fusion import train_fusion

    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    sys.argv = [
        "train_fusion",
        "--config", str(Path(args.config).resolve()),
        "--experiment-dir", str(args.experiment_dir.resolve()),
        "--trial-json", json.dumps(trial),
        "--training-budget-hours", str(float(args.training_budget_hours)),
    ]
    return int(train_fusion.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())

