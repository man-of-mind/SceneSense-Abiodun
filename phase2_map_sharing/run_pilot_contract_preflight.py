"""Cheap offline preflight for the paired-causal pilot contract.

This command validates configuration and disk-budget headroom only. It cannot
launch CARLA/OAI and never changes an authorization flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .pilot_contract import load_and_validate_pilot_config
from .retention import RawRetentionBudget, RetentionLimits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "phase2_map_sharing" / "configs" / "paired_causal_pilot_v1.yaml"


def preflight(config_path: Path, disk_root: Path) -> dict:
    contract = load_and_validate_pilot_config(config_path)
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    limits = RetentionLimits.from_mapping(config["raw_retention"])
    storage = RawRetentionBudget(Path(disk_root), limits).preflight(
        int(config["pilot"]["trajectory_count"])
    )
    return {
        "verdict": "PASS",
        "scope": "offline_contract_and_logical_disk_budget_only",
        "live_pilot_authorized": bool(contract["live_run_authorized"]),
        "config": str(Path(config_path).resolve()),
        "disk_root": str(Path(disk_root).resolve()),
        "contract": contract,
        "storage": storage,
        "note": (
            "The future pilot writer must still acquire this budget and call "
            "authorize_write before every heavy-artifact write."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--disk-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    print(
        json.dumps(
            preflight(arguments.config, arguments.disk_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
