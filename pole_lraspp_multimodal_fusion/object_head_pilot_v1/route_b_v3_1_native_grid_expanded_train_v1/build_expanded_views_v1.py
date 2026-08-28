#!/usr/bin/env python3
"""Build the expanded Route B v3 train/validation aggregate views.

A wrapper, not a second view builder.  The retained frozen builder
``route_b_v3_frozen_model_comparison_v1/scripts/build_views_v1.py`` is executed verbatim
with three module-level values redirected: the output experiment directory, the admitted
train episode tuple (the four canonical train episodes plus the six additional train-only
episodes) and the admitted validation episode tuple (unchanged).

The episode-namespaced sample IDs, the 0.2 s post-intervention exclusion, the retained
collision-window provenance policy, the depth-visibility contracts and the symlink-only
payload representation are therefore exactly the canonical ones.  Nothing resolves,
enumerates or reads a locked-split payload.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FROZEN_BUILDER = ROOT / (
    "experiments/route_b_v3_frozen_model_comparison_v1/20260827_184455/scripts/build_views_v1.py")
CANONICAL_REPORT = ROOT / (
    "data_collection/experiments/route_b_perception_v3/ROUTE_B_V3_CANONICAL_COLLECTION_REPORT.json")
ADDITIONAL_REPORT = ROOT / (
    "data_collection/experiments/route_b_perception_v3/"
    "ROUTE_B_V3_ADDITIONAL_TRAIN_COLLECTION_REPORT.json")

CANONICAL_TRAIN = (
    "canonical_v3_01_train_30_30_s501_tm1501",
    "canonical_v3_02_train_50_50_s502_tm1502",
    "canonical_v3_03_train_30_30_s503_tm1503",
    "canonical_v3_04_train_50_50_s504_tm1504",
)
ADDITIONAL_TRAIN = (
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
EXPECTED_ADDITIONAL_ROWS = {
    9: ("train", "traffic_30_30", 801, 1801), 10: ("train", "traffic_50_50", 802, 1802),
    11: ("train", "traffic_30_30", 803, 1803), 12: ("train", "traffic_50_50", 804, 1804),
    13: ("train", "traffic_30_30", 805, 1805), 14: ("train", "traffic_50_50", 806, 1806),
}


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("route_b_v3_build_views_for_expanded",
                                                  FROZEN_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {FROZEN_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_check_report(module: Any, original: Any):
    """Wrap, do not replace: ``original`` is captured before the module attribute is bound."""
    def check_report() -> dict[str, Any]:
        proof = original()  # canonical eight, verbatim; report-only for locked rows
        extra = json.loads(ADDITIONAL_REPORT.read_text(encoding="utf-8"))
        if extra.get("terminal") != "ROUTE_B_V3_1_ADDITIONAL_TRAIN_COLLECTION_EPISODES_PASSED":
            raise RuntimeError(f"additional-train report terminal: {extra.get('terminal')}")
        episodes = extra.get("episodes", [])
        if len(episodes) != 6:
            raise RuntimeError(f"additional-train report has {len(episodes)} episodes, expected 6")
        for item in episodes:
            number = int(item["episode"])
            actual = (str(item["split"]), str(item["density"]),
                      int(item["scenario_seed"]), int(item["tm_seed"]))
            if actual != EXPECTED_ADDITIONAL_ROWS.get(number) or not bool(item.get("passed")):
                raise RuntimeError(f"additional-train report mismatch at episode {number}: {actual}")
            if not bool((item.get("acceptance") or {}).get("passed")):
                raise RuntimeError(f"additional-train episode {number} failed acceptance")
        seeds = {(int(item["scenario_seed"]), int(item["tm_seed"])) for item in episodes}
        if len(seeds) != 6:
            raise RuntimeError("additional-train seed pairs are not six independent pairs")
        proof = dict(proof)
        proof.update({
            "additional_train_report_sha256": module.sha256(ADDITIONAL_REPORT),
            "additional_train_report_terminal": extra["terminal"],
            "additional_train_episode_count": len(episodes),
            "additional_train_independent_seed_pairs": sorted(seeds),
        })
        return proof
    return check_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    experiment.mkdir(parents=True, exist_ok=True)

    module = load_builder()
    module.EXPERIMENT = experiment
    module.TRAIN_EPISODES = CANONICAL_TRAIN + ADDITIONAL_TRAIN
    module.VAL_EPISODES = VAL_EPISODES
    module.check_report = make_check_report(module, module.check_report)
    if module.CANONICAL_REPORT != CANONICAL_REPORT:
        raise RuntimeError("frozen builder canonical report path drifted")
    if set(module.TRAIN_EPISODES) & set(module.VAL_EPISODES):
        raise RuntimeError("train/validation episode overlap")
    if len(module.TRAIN_EPISODES) != 10 or len(module.VAL_EPISODES) != 2:
        raise RuntimeError("expanded view must admit exactly 10 train and 2 validation episodes")
    code = module.main()
    if code == 0:
        (experiment / "EXPANDED_VIEWS_COMPLETE").write_text(
            "ROUTE_B_V3_EXPANDED_VIEWS_READY\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
