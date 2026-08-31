from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .core import (
    GROUP_IOU_THRESHOLDS,
    SEMANTIC_SUPPORT_THRESHOLDS,
    evaluate_frames,
    partition_frames,
    select_fit_configuration,
)
from .runtime import FROZEN_CHECKPOINT_SHA256


def _load_frames(cache: Path, shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for shard in shards:
        payload = torch.load(cache / str(shard["path"]), map_location="cpu", weights_only=True)
        shard_frames = payload.get("frames")
        if not isinstance(shard_frames, list) or len(shard_frames) != int(shard["frames"]):
            raise RuntimeError(f"consolidation cache shard drift: {shard['path']}")
        frames.extend(shard_frames)
    return frames


def _configuration(report: dict[str, Any]) -> dict[str, Any]:
    return {name: report[name] for name in (
        "grid_index", "semantic_support_threshold", "group_box_iou_threshold",
    )}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the preregistered train-only person consolidation grid")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cache = args.cache.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    expected_grid = {
        "semantic_support_thresholds": list(SEMANTIC_SUPPORT_THRESHOLDS),
        "group_box_iou_thresholds": list(GROUP_IOU_THRESHOLDS),
    }
    if (manifest.get("schema") != "splitfusion_fcos_person_instance_consolidation_cache_v1"
            or manifest.get("split") != "train"
            or int(manifest.get("pass_count", -1)) != 1
            or manifest.get("grid") != expected_grid
            or manifest.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or manifest.get("original_candidate_labels_stored") is not False
            or manifest.get("validation_or_test_accessed") is not False):
        raise RuntimeError("person consolidation cache contract drift")
    frames = _load_frames(cache, list(manifest["shards"]))
    if len(frames) != int(manifest["frames"]):
        raise RuntimeError("person consolidation cache frame count drift")
    fit, holdout, fit_ids, holdout_ids = partition_frames(frames)
    if manifest["episode_split"] != {"fit": list(fit_ids), "holdout": list(holdout_ids)}:
        raise RuntimeError("cached episode split drift")

    grid_reports, selected_fit = select_fit_configuration(fit)
    if len(grid_reports) != 36:
        raise RuntimeError("preregistered consolidation grid must contain exactly 36 configurations")
    holdout_report = None
    holdout_evaluations = 0
    if selected_fit is None or selected_fit["precision"] < 0.80 or selected_fit["recall"] < 0.80:
        status = "fit_infeasible"
    else:
        holdout_report = evaluate_frames(holdout, _configuration(selected_fit))
        holdout_evaluations = 1
        status = ("holdout_feasible" if holdout_report["precision"] >= 0.80
                  and holdout_report["recall"] >= 0.80 else "holdout_infeasible")

    result = {
        "schema": "splitfusion_fcos_person_instance_consolidation_result_v1",
        "cache_manifest": str(cache / "cache_manifest.json"),
        "base_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "fit_episodes": list(fit_ids),
        "holdout_episodes": list(holdout_ids),
        "grid_configuration_count": len(grid_reports),
        "selection": {
            "eligible_constraint": "recall >= 0.80",
            "order": ["maximum precision", "higher recall", "fewer retained predictions", "fixed grid order"],
        },
        "fit_grid": grid_reports,
        "selected_fit": selected_fit,
        "holdout": holdout_report,
        "holdout_evaluations": holdout_evaluations,
        "status": status,
        "validation_or_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(output), "status": status, "selected_fit": selected_fit,
        "holdout": holdout_report, "holdout_evaluations": holdout_evaluations,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
