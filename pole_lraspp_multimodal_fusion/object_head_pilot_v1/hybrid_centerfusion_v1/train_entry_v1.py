#!/usr/bin/env python3
"""Training entry point for the LR-ASPP/CenterFusion hybrid.

Installs the ``centerfusion_hybrid_v1`` builder dispatch and then hands control
to the *unmodified* production ``train_fusion`` entry point. Production files are
not edited; the trial JSON selects the hybrid via
``object_heads.head_arch = "centerfusion_hybrid_v1"``.

The warm-start mapping report produced while building the model is written next
to the checkpoints so the exact mapped / new / incompatible tensor lists are
recorded before any gradient step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hybrid_model_v1 import install  # noqa: E402


def _install_target_cap(trial: dict) -> str:
    """Match the baseline's object targets exactly.

    The fixed baseline (curriculum_stage2_joint_v1 epoch 13) was trained with the
    vehicle-only adaptive-radius cap installed. Installing the same cap here keeps
    the target tensors identical, so any delta is attributable to the architecture
    and not to a different supervision signal.
    """
    from object_head_pilot_v1.target_variants_v1 import assert_control_parity, install as install_cap
    import numpy as np

    rng = np.random.default_rng(20260824)
    objects = []
    for _ in range(24):
        width = float(rng.uniform(8.0, 400.0))
        height = float(rng.uniform(8.0, 300.0))
        objects.append({
            "class_index": int(rng.integers(0, 2)),
            "center_x": float(rng.uniform(0.0, 1280.0)),
            "center_y": float(rng.uniform(0.0, 720.0)),
            "bbox_w": width, "bbox_h": height, "area": width * height,
            "local_x": 1.0, "local_y": 2.0, "local_z": 0.3,
            "size_x": 4.0, "size_y": 2.0, "size_z": 1.5,
            "yaw_sin": 0.1, "yaw_cos": 0.9, "parked": 0.0, "radar_support": 1.0,
            "world_x": 10.0, "world_y": 5.0, "world_z": 0.2,
        })
    assert_control_parity({
        "objects": objects, "original_size": (1280, 720), "input_size": (768, 432),
        "heatmap_radius_px": 4, "max_objects": 64, "predict_bbox2d": True,
        "adaptive_heatmap_radius": True,
    })
    return install_cap(trial.get("object_heads", {}).get("vehicle_heatmap_radius_cap_px"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--training-budget-hours", type=float, default=0.0)
    args = parser.parse_args(argv)

    install()
    trial = json.loads(Path(args.trial_json).read_text(encoding="utf-8"))
    arm = _install_target_cap(trial)
    print(f"object-target arm: {arm}", flush=True)

    import torch

    from pole_lraspp_multimodal_fusion import model as model_module, train_fusion

    # Capture the warm-start report the builder attaches to the model.
    report_path = Path(args.experiment_dir) / "warm_start_mapping.json"
    original_build = model_module.build_multitask_fusion_lraspp

    def recording_build(**kwargs):
        built = original_build(**kwargs)
        report = getattr(built, "warm_start_report", None)
        if report is not None and not report_path.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return built

    model_module.build_multitask_fusion_lraspp = recording_build
    train_fusion.build_multitask_fusion_lraspp = recording_build

    print(f"torch={torch.__version__} cuda={torch.version.cuda} device_ok={torch.cuda.is_available()}", flush=True)
    sys.argv = [
        "train_fusion",
        "--config", str(Path(args.config).resolve()),
        "--experiment-dir", str(Path(args.experiment_dir).resolve()),
        "--trial-json", json.dumps(trial),
        "--training-budget-hours", str(float(args.training_budget_hours)),
    ]
    train_fusion.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
