from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.runtime import (
    load_frozen_runtime,
    require_device,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_roi_verifier_v1.verifier import (
    PersonRoIDescriptor,
)

from .contract import load_revised_selector
from .runtime import apply_relational_p070_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly one real train-frame CUDA smoke")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"create-only smoke output already exists: {output}")
    device = require_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the controlled real-frame smoke requires CUDA")

    relational = load_revised_selector(device)
    frozen = load_frozen_runtime(device)
    if frozen.checkpoint_sha256 != relational.base_checkpoint_sha256:
        raise RuntimeError("frozen base runtime/checkpoint contract mismatch")
    extractor = PersonRoIDescriptor().to(device).eval()
    dataset = frozen.base.data.InferenceDataset(frozen.dataset_root, "train")
    fused, row, calibration = dataset[0]
    fused = fused.unsqueeze(0).to(device)
    calibration_device = {name: value.to(device) for name, value in calibration.items()}

    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with torch.inference_mode():
        outputs = frozen.model(fused, dense=False)  # The only model forward in this script.
        base = frozen.model.postprocess(outputs, [calibration_device])[0]
        semantic_before = outputs["semantic_logits"].clone()
        result, keep = apply_relational_p070_policy(outputs, base, relational, extractor)
    torch.cuda.synchronize(device)

    count = int(base["scores"].numel())
    retained = int(result["scores"].numel())
    if (keep.shape != (retained,) or bool((keep < 0).any()) or bool((keep >= count).any())
            or (retained > 1 and not bool((keep[1:] > keep[:-1]).all()))):
        raise RuntimeError("smoke candidate source-index alignment drift")
    if not torch.equal(outputs["semantic_logits"], semantic_before):
        raise RuntimeError("relational policy changed segmentation logits")
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
           for value in result.values()):
        raise FloatingPointError("smoke produced a non-finite retained field")
    for name, value in base.items():
        if name != "scores" and not torch.equal(result[name], value.index_select(0, keep)):
            raise RuntimeError(f"smoke non-score field changed: {name}")

    base_classes = base["labels_internal"].long()
    retained_classes = result["labels_internal"].long()
    base_vehicle = int((base_classes != 1).sum())
    retained_vehicle = int((retained_classes != 1).sum())
    if base_vehicle != retained_vehicle:
        raise RuntimeError("smoke vehicle candidate count changed")
    report = {
        "schema": "splitfusion_fcos_relational_p070_cuda_smoke_v1",
        "device": str(device),
        "source_split": "train",
        "source_index": 0,
        "sample_id": str(row["sample_id"]),
        "model_forward_count": 1,
        "validation_or_test_accessed": False,
        "base_checkpoint_sha256": relational.base_checkpoint_sha256,
        "selector_checkpoint_sha256": relational.selector_checkpoint_sha256,
        "base_candidates": count,
        "retained_candidates": retained,
        "base_person_candidates": int((base_classes == 1).sum()),
        "retained_person_candidates": int((retained_classes == 1).sum()),
        "base_vehicle_candidates": base_vehicle,
        "retained_vehicle_candidates": retained_vehicle,
        "finite_outputs": True,
        "candidate_alignment_verified": True,
        "non_score_fields_unchanged": True,
        "vehicle_candidates_unchanged": True,
        "segmentation_logits_unchanged": True,
        "wall_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
