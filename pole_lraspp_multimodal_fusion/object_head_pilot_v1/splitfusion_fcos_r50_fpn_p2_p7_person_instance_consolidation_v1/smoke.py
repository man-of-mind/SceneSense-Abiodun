from __future__ import annotations

import argparse
import json

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1.labeling import (
    contract_world_targets,
)

from .core import build_frame_record, evaluate_frames, grid_configurations
from .runtime import load_frozen_runtime, require_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one train-frame person consolidation smoke")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = require_device(args.device)
    runtime = load_frozen_runtime(device)
    dataset = runtime.base.data.RouteBDataset(runtime.dataset_root, "train", seed=20260830, augment=False)
    selected = None
    for index, row in enumerate(dataset.rows):
        _world, classes = contract_world_targets(dataset.objects.get(row["sample_id"], ()))
        if bool((classes == 1).any()):
            selected = index
            break
    if selected is None:
        raise RuntimeError("no training frame contains eligible person GT")
    item = dataset[selected]
    target, row = item["target"], item["row"]
    calibration = {name: target[name].to(device) for name in ("intrinsic", "extrinsic")}
    with torch.inference_mode():
        outputs = runtime.model(item["input"].unsqueeze(0).to(device), dense=False)
        detections = runtime.model.postprocess(outputs, [calibration])[0]
    gt_world_xy, gt_classes = contract_world_targets(dataset.objects.get(target["sample_id"], ()))
    frame = build_frame_record(
        outputs=outputs, detections=detections, ignore_mask=target["ignore_mask"],
        gt_person_world_xy=gt_world_xy[gt_classes == 1], sample_id=target["sample_id"],
        experiment_id=row["experiment_id"],
    )
    if frame["scores"].numel() == 0:
        raise RuntimeError("smoke frame contains no post-NMS person candidates")
    report = evaluate_frames([frame], grid_configurations()[0])
    print(json.dumps({
        "sample_id": frame["sample_id"], "person_candidates": frame["scores"].numel(),
        "semantic_components": frame["semantic_component_count"], "off_off_report": report,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
