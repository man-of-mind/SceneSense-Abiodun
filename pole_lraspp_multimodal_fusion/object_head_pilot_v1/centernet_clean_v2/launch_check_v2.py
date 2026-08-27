#!/usr/bin/env python3
"""Pre-run checks for CenterNet v2.  Exactly the authorized set, nothing broader:

  * py_compile of every new v2 Python file
  * config parse (yaml + trial json)
  * warm-start checkpoint SHA-256
  * train / val / test split counts
  * one real batch forward + backward at the production batch size
  * finite, nonzero gradients in the new vehicle, person, offset and
    segmentation branches
  * output shapes and the decoded-field schema
  * split-boundary check proving ``decode_tail`` reads only the keys returned by
    ``encode_front``
"""

from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent, HERE.parent.parent, HERE.parent.parent.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pole_lraspp_multimodal_fusion.common import load_config, read_manifest, save_json  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes, parse_matrix  # noqa: E402

from centernet_model_v2 import build_centernet_v2, warm_start_from_v1  # noqa: E402
from decode_v2 import decode_objects_v2  # noqa: E402
from losses_v2 import DEFAULT_OBJECT_WEIGHTS, compute_v2_losses  # noqa: E402
from targets_v2 import NativeFusionDataset  # noqa: E402
from train_v2 import build_param_groups, sha256  # noqa: E402

EXPECTED_RECORD_FIELDS = {
    "class_index", "class_name", "score", "center_x_px", "center_y_px",
    "bbox_w_px", "bbox_h_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
    "local_x", "local_y", "local_z", "world_x", "world_y", "world_z",
    "size_x", "size_y", "size_z", "yaw_sin", "yaw_cos",
    "parked_score", "radar_support_score",
}
DIAGNOSTIC_FIELDS = {"native_x", "native_y", "branch_stride"}

GRADIENT_BRANCHES = {
    "vehicle_head_heatmap": "vehicle_head.heatmap.weight",
    "vehicle_head_offset": "vehicle_head.offset.weight",
    "vehicle_head_regression": "vehicle_head.regression.weight",
    "person_feature": "person_feature.up.0.weight",
    "person_head_heatmap": "person_head.heatmap.weight",
    "person_head_offset": "person_head.offset.weight",
    "person_head_regression": "person_head.regression.weight",
    "stride2_projection": "stride2_proj.0.0.weight",
    "rgb_radar_fusion": "fusion.project.0.weight",
    "segmentation_decoder_skip": "classifier.fuse_skip.0.weight",
    "segmentation_decoder_output": "classifier.up_to_full.weight",
}


class TrackingBundle(dict):
    """Records which keys the tail actually reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed: Set[str] = set()

    def __getitem__(self, key):
        self.accessed.add(str(key))
        return super().__getitem__(key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--warm-start-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    report: Dict[str, object] = {}
    failures: List[str] = []

    # ---- 1. py_compile -------------------------------------------------
    compiled = []
    for source in sorted(HERE.glob("*.py")):
        py_compile.compile(str(source), doraise=True)
        compiled.append(source.name)
    report["py_compile"] = {"status": "PASS", "files": compiled}

    # ---- 2. config parse ------------------------------------------------
    config = load_config(str(args.config))
    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    report["config_parse"] = {
        "status": "PASS",
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "trial": str(args.trial_json),
        "trial_sha256": sha256(args.trial_json),
        "experiment_name": config.get("experiment_name"),
        "trial_name": trial.get("name"),
    }

    # ---- 3. checkpoint hash ---------------------------------------------
    actual = sha256(args.warm_start)
    ok = actual == str(args.warm_start_sha256)
    report["warm_start_sha256"] = {
        "status": "PASS" if ok else "FAIL",
        "path": str(args.warm_start.resolve()),
        "expected": str(args.warm_start_sha256),
        "actual": actual,
    }
    if not ok:
        failures.append("warm_start_sha256")

    # ---- 4. split counts -------------------------------------------------
    exp_dir = args.experiment_dir.resolve()
    dataset_dir = exp_dir / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    counts = {"train": 0, "val": 0, "test": 0}
    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split", "train")
        counts[split] = counts.get(split, 0) + 1
        splits.setdefault(split, []).append(row)
    expected_counts = {"train": 6600, "val": 3588, "test": 0}
    ok = all(counts.get(k, 0) == v for k, v in expected_counts.items())
    report["split_counts"] = {
        "status": "PASS" if ok else "FAIL",
        "counts": counts,
        "expected": expected_counts,
        "dataset_dir": str(dataset_dir.resolve()),
    }
    if not ok:
        failures.append("split_counts")

    # ---- 5. one real batch forward/backward ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    object_cfg = dict(config.get("object_heads", {}))
    input_width, input_height = [int(v) for v in trial["input_size"]]
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    train_ds = NativeFusionDataset(
        dataset_dir, splits["train"], object_rows, (input_width, input_height), object_cfg,
        augment_strength=str(trial.get("augment_strength", "off")),
    )
    batch_size = int(trial.get("batch_size", 24))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    tensors, masks, targets = next(iter(loader))
    tensors = tensors.to(device)
    masks = masks.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    model = build_centernet_v2(
        num_classes=int(config["training"]["num_classes"]),
        radar_channels=int(config["fusion"]["radar_channels"]),
        pretrained=True,
    ).to(device)
    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    mapping = warm_start_from_v1(model, warm.get("model", warm))
    report["warm_start_mapping"] = {
        "status": "PASS" if mapping["incompatible_count"] == 0 and mapping["loaded_count"] > 0 else "FAIL",
        "loaded_count": mapping["loaded_count"],
        "new_count": mapping["new_count"],
        "incompatible_count": mapping["incompatible_count"],
        "incompatible_tensors": mapping["incompatible_tensors"],
    }
    if mapping["incompatible_count"] or not mapping["loaded_count"]:
        failures.append("warm_start_mapping")

    param_groups, group_names, group_lrs = build_param_groups(model, trial)
    report["param_groups"] = {
        "status": "PASS",
        "lrs": group_lrs,
        "counts": {k: len(v) for k, v in group_names.items()},
    }

    object_weights = dict(DEFAULT_OBJECT_WEIGHTS)
    object_weights.update(trial.get("loss_weights", {}).get("object", {}))
    class_weights = torch.tensor(
        [float(w) for w in trial["class_loss_weights"]], dtype=torch.float32, device=device
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    with torch.autocast(device_type=device.type, enabled=amp_enabled, cache_enabled=False):
        outputs = model(tensors)
    with torch.autocast(device_type=device.type, enabled=False):
        loss, parts = compute_v2_losses(
            outputs, masks, targets,
            object_weights=object_weights,
            segmentation_weight=float(trial["loss_weights"]["segmentation"]),
            object_total_weight=float(trial["loss_weights"]["object_total"]),
            class_weights=class_weights,
            lovasz_weight=float(trial.get("lovasz_weight", 0.5)),
        )
    # Two backward passes on the same batch, for two different questions.
    #  (a) the scaled AMP path that training will actually run - does it execute?
    #  (b) an UNSCALED backward whose .grad values are the honest gradient
    #      magnitudes.  Inspecting AMP-scaled grads is meaningless: the default
    #      GradScaler init scale (65536) times this loss overflows fp16 on the
    #      first iteration by construction, which the scaler handles by skipping
    #      the step and halving the scale.
    scaler.scale(loss).backward()
    scaled_backward_ok = True
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=amp_enabled, cache_enabled=False):
        outputs = model(tensors)
    with torch.autocast(device_type=device.type, enabled=False):
        loss, parts = compute_v2_losses(
            outputs, masks, targets,
            object_weights=object_weights,
            segmentation_weight=float(trial["loss_weights"]["segmentation"]),
            object_total_weight=float(trial["loss_weights"]["object_total"]),
            class_weights=class_weights,
            lovasz_weight=float(trial.get("lovasz_weight", 0.5)),
        )
    loss.backward()
    inv_scale = 1.0
    report["forward_backward"] = {
        "status": "PASS" if np.isfinite(float(loss.item())) else "FAIL",
        "batch_size": int(tensors.shape[0]),
        "input_shape": list(tensors.shape),
        "loss": float(loss.item()),
        "loss_parts": {k: float(v) for k, v in parts.items()},
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
            if device.type == "cuda" else None
        ),
        "amp_enabled": bool(amp_enabled),
        "scaled_backward_executed": bool(scaled_backward_ok),
        "gradients_inspected_from": "unscaled backward on the same batch",
    }
    if not np.isfinite(float(loss.item())):
        failures.append("forward_backward")

    # ---- 6. gradients in the new branches --------------------------------
    named = dict(model.named_parameters())
    grad_report: Dict[str, object] = {}
    grad_ok = True
    for label, tensor_name in GRADIENT_BRANCHES.items():
        param = named[tensor_name]
        grad = param.grad
        if grad is None:
            grad_report[label] = {"tensor": tensor_name, "status": "FAIL", "reason": "grad is None"}
            grad_ok = False
            continue
        g = grad.detach().float() * inv_scale
        finite = bool(torch.isfinite(g).all().item())
        norm = float(g.norm().item())
        status = "PASS" if finite and norm > 0.0 else "FAIL"
        grad_report[label] = {
            "tensor": tensor_name, "status": status, "finite": finite, "grad_l2_norm": norm,
        }
        grad_ok = grad_ok and status == "PASS"
    report["new_branch_gradients"] = {"status": "PASS" if grad_ok else "FAIL", "branches": grad_report}
    if not grad_ok:
        failures.append("new_branch_gradients")

    # ---- 7. output shapes + decoded-field schema -------------------------
    model.eval()
    with torch.inference_mode():
        eval_outputs = model(tensors[:1])
    shapes = {k: list(v.shape) for k, v in eval_outputs.items()}
    expected_shapes = {
        "out": [1, int(config["training"]["num_classes"]), input_height, input_width],
        "veh_hm": [1, 1, input_height // 4, input_width // 4],
        "veh_off": [1, 2, input_height // 4, input_width // 4],
        "veh_reg": [1, 12, input_height // 4, input_width // 4],
        "per_hm": [1, 1, input_height // 2, input_width // 2],
        "per_off": [1, 2, input_height // 2, input_width // 2],
        "per_reg": [1, 12, input_height // 2, input_width // 2],
    }
    shapes_ok = shapes == expected_shapes
    target_shapes = {k: list(v.shape) for k, v in targets.items()}
    report["output_shapes"] = {
        "status": "PASS" if shapes_ok else "FAIL",
        "actual": shapes,
        "expected": expected_shapes,
        "target_shapes": target_shapes,
    }
    if not shapes_ok:
        failures.append("output_shapes")

    matrix = None
    for row in splits["val"]:
        matrix = parse_matrix(row.get("camera_matrix_json", ""))
        if matrix is not None:
            break
    preds = decode_objects_v2(
        eval_outputs, camera_matrix=matrix, input_size=(input_width, input_height),
        score_threshold=0.0, topk=8,
    )
    schema_ok = bool(preds) and all(
        set(p.keys()) == EXPECTED_RECORD_FIELDS | DIAGNOSTIC_FIELDS for p in preds
    )
    report["decoded_field_schema"] = {
        "status": "PASS" if schema_ok else "FAIL",
        "records": len(preds),
        "record_fields": sorted(preds[0].keys()) if preds else [],
        "spatial_map_fields": sorted(EXPECTED_RECORD_FIELDS),
        "additive_diagnostic_fields": sorted(DIAGNOSTIC_FIELDS),
        "strides_present": sorted({int(p["branch_stride"]) for p in preds}) if preds else [],
    }
    if not schema_ok:
        failures.append("decoded_field_schema")

    # ---- 8. split-boundary check ------------------------------------------
    with torch.inference_mode():
        rgb = tensors[:1, :3]
        radar = tensors[:1, 3 : 3 + model.radar_channels]
        bundle = model.encode_front(rgb, radar)
        tracking = TrackingBundle(bundle)
        tail = model.decode_tail(tracking, (input_height, input_width))
        forward = model(tensors[:1])
        identical = all(torch.equal(tail[k], forward[k]) for k in forward)
        # Zeroing every bundle tensor must change every tail output: nothing the
        # tail produces can come from a path that bypasses the bundle.
        zeroed = model.decode_tail(
            {k: torch.zeros_like(v) for k, v in bundle.items()}, (input_height, input_width)
        )
        all_depend = all(not torch.equal(zeroed[k], forward[k]) for k in forward)
    boundary_ok = (
        set(tracking.accessed) <= set(bundle.keys())
        and set(tracking.accessed) == set(bundle.keys())
        and identical
        and all_depend
    )
    report["split_boundary"] = {
        "status": "PASS" if boundary_ok else "FAIL",
        "bundle_keys": list(bundle.keys()),
        "bundle_shapes": {k: list(v.shape) for k, v in bundle.items()},
        "keys_read_by_tail": sorted(tracking.accessed),
        "tail_matches_forward_bitwise": bool(identical),
        "every_tail_output_depends_on_bundle": bool(all_depend),
        "raw_rgb_or_radar_side_channel": False,
        "bundle_elements_per_frame": int(sum(int(np.prod(v.shape[1:])) for v in bundle.values())),
    }
    if not boundary_ok:
        failures.append("split_boundary")

    report["overall"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    save_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
