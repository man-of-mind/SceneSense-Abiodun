#!/usr/bin/env python3
"""Object-head architecture pilot / staged curriculum runner (preparation only).

Wraps the production ``train_fusion`` entry point without editing it:

* installs the pilot target builder for the candidate arm
  (``vehicle_heatmap_radius_cap_px``), leaving the control arm on the
  unmodified production function;
* proves the pilot copy is bit-identical to production when the cap is disabled
  before any training step runs;
* records the exact environment (Python, PyTorch, CUDA, cuDNN, GPU, TF32 and
  determinism settings) alongside the resolved trial.

It never overwrites an existing checkpoint directory: ``--experiment-dir`` is
create-or-extend, and the runner refuses to start if the trial's checkpoint
directory already contains ``best.pt``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
ABIODUN = PKG_ROOT.parent
for path in (str(PKG_ROOT), str(ABIODUN)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from object_head_pilot_v1.target_variants_v1 import (  # noqa: E402
    assert_control_parity,
    install,
)


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": f"{properties.major}.{properties.minor}",
            "multi_processor_count": int(properties.multi_processor_count),
        }
        try:
            record["nvidia_smi_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            record["nvidia_smi_driver"] = ""
    return record


def parity_sample() -> dict[str, Any]:
    """A deterministic synthetic frame used only for the pilot-copy parity guard."""
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
    return {
        "objects": objects,
        "original_size": (1280, 720),
        "input_size": (768, 432),
        "heatmap_radius_px": 4,
        "max_objects": 64,
        "predict_bbox2d": True,
        "adaptive_heatmap_radius": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="base fusion YAML config")
    parser.add_argument("--trial-json", required=True, type=Path,
                        help="pilot trial JSON (arm A control or arm B candidate)")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--training-budget-hours", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve, guard and record the environment, then stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trial = json.loads(Path(args.trial_json).read_text(encoding="utf-8"))
    trial_name = str(trial["name"])
    cap = trial.get("object_heads", {}).get("vehicle_heatmap_radius_cap_px")

    checkpoint_dir = Path(args.experiment_dir) / "checkpoints" / trial_name
    if (checkpoint_dir / "best.pt").is_file():
        print(
            f"refusing to overwrite an existing checkpoint: {checkpoint_dir / 'best.pt'}",
            file=sys.stderr,
        )
        return 2

    # Guard before anything trains: the duplicated pilot builder must be exactly
    # the production one when the cap is off, or the control/candidate comparison
    # is measuring a copy bug rather than the cap.
    assert_control_parity(parity_sample())
    arm = install(cap)

    record = {
        "pilot": "object_head_arch_pilot_v1",
        "trial": trial_name,
        "arm": arm,
        "vehicle_heatmap_radius_cap_px": cap,
        "target_parity_guard": "PASS",
        "environment": environment_record(),
        "resolved_trial": trial,
        "config": str(args.config),
        "experiment_dir": str(Path(args.experiment_dir).resolve()),
    }
    Path(args.experiment_dir).mkdir(parents=True, exist_ok=True)
    record_path = Path(args.experiment_dir) / f"pilot_environment_{trial_name}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print(f"dry run only; environment recorded at {record_path}", flush=True)
        return 0

    from pole_lraspp_multimodal_fusion import train_fusion

    sys.argv = [
        "train_fusion",
        "--config", str(args.config),
        "--experiment-dir", str(args.experiment_dir),
        "--trial-json", json.dumps(trial),
        "--training-budget-hours", str(float(args.training_budget_hours)),
    ]
    train_fusion.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
