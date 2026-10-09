"""Frozen-model detection records for a documented frame subset (no segmentation written).

Identical inference path to splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.infer_service_candidates
(frozen epoch-26 runtime, InferenceDataset, model(fused, dense=False), postprocess,
apply_combined_service_policy, combined_records, FP32).  The only differences: split/index
selection and no segmentation PNGs (segmentation is not touched by the range correction).

Subset rules (fixed before any result was seen):
  --rule val_first_N       : first N validation rows in manifest order (reproduction proof).
  --rule train_stride      : every STRIDE-th train frame within each train episode, episodes
                             in manifest order, frames in manifest order, offset 0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1.runtime import (
    load_frozen_runtime, require_device, sha256,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.provenance import (
    FROZEN_CHECKPOINT_SHA256,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    apply_combined_service_policy, combined_records,
)


def episode_of(sample_id: str) -> str:
    return sample_id.rsplit("_", 2)[0]


def select(rows: list[dict[str, str]], rule: str, n: int, stride: int) -> list[int]:
    if rule == "val_first_N":
        return list(range(min(n, len(rows))))
    if rule == "train_stride":
        by_ep: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_ep[episode_of(r["sample_id"])].append(i)
        out: list[int] = []
        for ep in by_ep:  # insertion order = manifest order
            out.extend(by_ep[ep][::stride])
        return sorted(out)
    raise ValueError(rule)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val"), required=True)
    ap.add_argument("--rule", choices=("val_first_N", "train_stride"), required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    device = require_device(a.device)
    runtime = load_frozen_runtime(device)
    if runtime.checkpoint_sha256 != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("not frozen epoch 26")
    dataset = runtime.base.data.InferenceDataset(runtime.dataset_root, a.split)
    idx = select(dataset.rows, a.rule, a.n, a.stride)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    det = out / "detections.csv"
    t0 = time.monotonic()
    count = 0
    with det.open("x", encoding="utf-8", newline="") as s:
        w = csv.DictWriter(s, fieldnames=runtime.base.infer.FIELDS)
        w.writeheader()
        with torch.inference_mode():
            for k, i in enumerate(idx):
                fused, row, cal = dataset[i]
                fused = fused.unsqueeze(0).to(device)
                cal_d = {name: v.to(device) for name, v in cal.items()}
                outputs = runtime.model(fused, dense=False)
                base = runtime.model.postprocess(outputs, [cal_d])[0]
                d, orig = apply_combined_service_policy(outputs, base)
                recs = combined_records(runtime.base, row, d, orig)
                w.writerows(recs)
                count += len(recs)
                if (k + 1) % 500 == 0:
                    print(json.dumps({"frames": k + 1, "records": count}), flush=True)
    manifest = {
        "schema": "splitfusion_range_correction_v1.infer_subset",
        "split": a.split, "rule": a.rule, "n": a.n, "stride": a.stride,
        "frames": len(idx), "sample_ids_sha256": hashlib.sha256(
            "\n".join(dataset.rows[i]["sample_id"] for i in idx).encode()).hexdigest(),
        "records": count, "detections_sha256": sha256(det),
        "checkpoint_sha256": runtime.checkpoint_sha256, "precision": "FP32, torch defaults unchanged (same as original inference script)",
        "segmentation_written": False, "gt_paths_opened": 0,
        "wall_seconds": time.monotonic() - t0,
    }
    (out / "inference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "sample_ids.txt").write_text("\n".join(dataset.rows[i]["sample_id"] for i in idx) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
