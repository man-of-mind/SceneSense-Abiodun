#!/usr/bin/env python3
"""Print the frozen future evaluation matrix; never loads a model checkpoint."""

from __future__ import annotations

import json

from .metrics import iou_thresholds, visibility_range_views
from .protocol import load_registered_protocol


def main() -> int:
    registration = load_registered_protocol()
    protocol = registration["protocol"]
    print(json.dumps({
        "schema": "route_b_publication_future_evaluation_plan_v1",
        "ground_truth_only": True,
        "visibility_range_views": visibility_range_views(protocol),
        "service_matching": {"method": "class-aware greedy nearest world-XY", "radius_m": 3.0},
        "standard_detection": {"AP50": 0.50, "AP50_95_iou_thresholds": iou_thresholds()},
        "segmentation": ["vehicle pixel IoU", "person pixel IoU", "foreground mean IoU"],
        "score_views": protocol["evaluation"]["fixed_score_views"],
        "episodes": protocol["prospective_episodes"],
        "registration": {key: registration[key] for key in (
            "lock_path", "protocol_path", "lock_sha256", "protocol_sha256", "bound_files_verified"
        )},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
