#!/usr/bin/env python3
"""Materialize the Route B v3.1 GT contract over the expanded train view.

A wrapper, not a second contract builder.  ``route_b_v3_1_clean_base_v1/build_contract_v1``
is executed verbatim with three module-level values redirected: the aggregate source view,
the admitted episode tuples (10 train, 2 validation) and the registered manifest row counts.

Everything that defines the contract is therefore unchanged: dynamic-actor and
environment-static vehicle positives/ignores, person v0.10 positives/ignores, the v0.25
sensitivity contract, semantic-component neutral/ignore handling, the registered
reconciliation quarantines, the audited Town10HD static catalog, the 0.2 s post-intervention
exclusion inherited from the source view, and the source hashes and provenance.

The registered train row count is derived independently from the two collection reports and
the source view's post-intervention exclusion provenance; the validation row count must stay
exactly the retained 3345.  The raw v3 episodes are never mutated or rewritten.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CLEAN_BASE_BUILDER = ROOT / (
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_clean_base_v1/"
    "build_contract_v1.py")
CANONICAL_REPORT = ROOT / (
    "data_collection/experiments/route_b_perception_v3/ROUTE_B_V3_CANONICAL_COLLECTION_REPORT.json")
ADDITIONAL_REPORT = ROOT / (
    "data_collection/experiments/route_b_perception_v3/"
    "ROUTE_B_V3_ADDITIONAL_TRAIN_COLLECTION_REPORT.json")

TRAIN_EPISODES = (
    "canonical_v3_01_train_30_30_s501_tm1501",
    "canonical_v3_02_train_50_50_s502_tm1502",
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
    "extra_v3_09_train_30_30_s801_tm1801",
    "extra_v3_10_train_50_50_s802_tm1802",
    "extra_v3_11_train_30_30_s803_tm1803",
    "extra_v3_12_train_50_50_s804_tm1804",
    "extra_v3_13_train_30_30_s805_tm1805",
    "extra_v3_14_train_50_50_s806_tm1806",
)
VAL_EPISODES = (
    "canonical_v3_05_val_30_30_s601_tm1601",
    "canonical_v3_06_val_50_50_s602_tm1602",
)
RETAINED_VALIDATION_ROWS = 3345


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def expected_train_rows(views: Path) -> dict[str, Any]:
    """saved frames from the collection reports, minus the view's post-intervention drops."""
    saved: dict[str, int] = {}
    for report_path in (CANONICAL_REPORT, ADDITIONAL_REPORT):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for item in report["episodes"]:
            output_dir = Path(item["output_dir"])
            if output_dir.name in TRAIN_EPISODES:
                saved[output_dir.name] = int(item["saved_frames"])
    missing = [name for name in TRAIN_EPISODES if name not in saved]
    if missing:
        raise RuntimeError(f"no reported saved-frame count for {missing}")
    excluded = read_csv(views / "provenance/train_excluded_post_intervention_samples.csv")
    expected = sum(saved.values()) - len(excluded)
    return {"saved_frames_by_episode": saved, "saved_frames_total": sum(saved.values()),
            "post_intervention_excluded": len(excluded), "expected_train_rows": expected}


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "route_b_v3_1_clean_base_build_contract_for_expanded", CLEAN_BASE_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CLEAN_BASE_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True, type=Path,
                        help="expanded aggregate view experiment directory")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    views = args.views.resolve(strict=True)
    output_root = args.output_root.resolve()

    rows = expected_train_rows(views)
    validation_rows = len(read_csv(views / "views/val/manifest.csv"))
    if validation_rows != RETAINED_VALIDATION_ROWS:
        raise RuntimeError(
            f"validation row drift: {validation_rows} != {RETAINED_VALIDATION_ROWS}")

    module = load_builder()
    module.FROZEN = views
    module.EXPECTED_ROWS = {"train": int(rows["expected_train_rows"]), "val": RETAINED_VALIDATION_ROWS}
    module.EXPECTED_EPISODES = {"train": TRAIN_EPISODES, "val": VAL_EPISODES}
    if set(TRAIN_EPISODES) & set(VAL_EPISODES):
        raise RuntimeError("train/validation episode overlap")

    code = module.run(output_root)
    if code == 0:
        with (output_root / "EXPANDED_SOURCE_ROW_DERIVATION.json").open("x", encoding="utf-8") as fh:
            json.dump({
                "schema": "route_b_v3_1_expanded_train_row_derivation_v1",
                "views": str(views), "train": rows,
                "validation_rows_retained_unchanged": RETAINED_VALIDATION_ROWS,
                "train_episodes": list(TRAIN_EPISODES), "val_episodes": list(VAL_EPISODES),
            }, fh, indent=2, sort_keys=True)
            fh.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
