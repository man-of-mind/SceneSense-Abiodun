#!/usr/bin/env python3
"""Option A training entry: discriminative learning rate on the hybrid's new modules.

Two things this adds over the pilot's training entry, and nothing else. The
model, the losses, the object targets, the evaluator and the decoder are used
exactly as they already exist.

1. **Exact warm start.** The whole model state is loaded ``strict=True`` from the
   parity-verified ``warm_start.pt``, so all 379 tensors are bit-identical to the
   checkpoint Phase C validated - including the 35 freshly initialised ones. (The
   trial's ``init_rgb_checkpoint`` is empty; re-deriving the warm start from the
   frozen baseline would reproduce the 344 mapped tensors but redraw the 35 new
   ones from a different RNG state, which is not the same checkpoint.)

2. **Freeze audit at the exact right moment.** ``train_fusion`` applies every
   freeze flag and then calls ``_count_parameters(model)`` immediately before
   building the optimizer. Wrapping that one call audits the model in precisely
   the state the optimizer will see. The audit records every trainable parameter
   name grouped by module and aborts with IMPLEMENTATION_BLOCKED if any inherited
   parameter is trainable, if any required new module is fully frozen, or if any
   trainable parameter falls outside the six permitted groups.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_PILOT = _HERE.parent
for _p in (str(_PILOT / "hybrid_centerfusion_v1"), str(_PILOT.parent), str(_PILOT.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hybrid_model_v1 import install  # noqa: E402

# Inherited: must be entirely frozen.
INHERITED_PREFIXES = ("backbone.", "classifier.", "object_head.")

# The six modules that must train. Every trainable parameter must fall in one.
TRAINABLE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "radar_encoder": ("radar_encoder.",),
    "hybrid_fpn": ("lat16.", "lat8.", "lat4.", "norm16.", "norm8.", "norm4.",
                   "reduce8.", "smooth4."),
    "refinement_trunk": ("refine_trunk.",),
    "vehicle_refinement_heatmap_head": ("refine_vehicle_heatmap_head.",),
    "person_refinement_heatmap_head": ("refine_person_heatmap_head.",),
    "xyz_dim_yaw_refinement_head": ("refine_regression_head.",),
}


def _group_of(name: str) -> str:
    for group, prefixes in TRAINABLE_GROUPS.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            return group
    if any(name.startswith(prefix) for prefix in INHERITED_PREFIXES):
        return "inherited"
    return "unassigned"


def audit(model, record_path: Path) -> Dict[str, object]:
    frozen: Dict[str, Dict[str, int]] = {}
    trainable: Dict[str, List[str]] = {group: [] for group in TRAINABLE_GROUPS}
    trainable["inherited"] = []
    trainable["unassigned"] = []
    counts: Dict[str, Dict[str, int]] = {}

    for name, param in model.named_parameters():
        group = _group_of(name)
        bucket = counts.setdefault(group, {"trainable_tensors": 0, "trainable_params": 0,
                                           "frozen_tensors": 0, "frozen_params": 0})
        if param.requires_grad:
            bucket["trainable_tensors"] += 1
            bucket["trainable_params"] += int(param.numel())
            trainable[group].append(name)
        else:
            bucket["frozen_tensors"] += 1
            bucket["frozen_params"] += int(param.numel())
    frozen = counts

    failures: List[str] = []
    if trainable["inherited"]:
        failures.append(
            f"{len(trainable['inherited'])} inherited parameters are trainable: "
            f"{trainable['inherited'][:6]}")
    if trainable["unassigned"]:
        failures.append(
            f"{len(trainable['unassigned'])} trainable parameters outside the six permitted "
            f"groups: {trainable['unassigned'][:6]}")
    for group in TRAINABLE_GROUPS:
        if not trainable[group]:
            failures.append(f"required module '{group}' has no trainable parameters")

    total_trainable = sum(b["trainable_params"] for b in counts.values())
    total_frozen = sum(b["frozen_params"] for b in counts.values())
    result = {
        "check": "disc_lr_v1_freeze_audit",
        "status": "PASS" if not failures else "IMPLEMENTATION_BLOCKED",
        "failures": failures,
        "totals": {"trainable_params": total_trainable, "frozen_params": total_frozen,
                   "total_params": total_trainable + total_frozen},
        "by_group": counts,
        "trainable_parameter_names": {g: names for g, names in trainable.items() if names},
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k != "trainable_parameter_names"}, indent=2, sort_keys=True), flush=True)
    print("\ntrainable parameter names by module:", flush=True)
    for group, names in result["trainable_parameter_names"].items():
        print(f"  [{group}] {len(names)} tensors", flush=True)
        for name in names:
            print(f"      {name}", flush=True)
    return result


def _install_target_cap(trial: dict) -> str:
    """Identical object targets to the baseline and to the failed hybrid run."""
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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--warm-start-state", required=True, type=Path)
    parser.add_argument("--warm-start-sha256", required=True)
    parser.add_argument("--training-budget-hours", type=float, default=0.0)
    args = parser.parse_args(argv)

    import hashlib

    import torch

    warm_path = args.warm_start_state.resolve(strict=True)
    digest = hashlib.sha256()
    with warm_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != args.warm_start_sha256:
        print(f"warm-start sha256 mismatch: expected {args.warm_start_sha256}, got {actual}",
              file=sys.stderr)
        return 3
    print(f"warm start: {warm_path}\nsha256: {actual}", flush=True)

    install()
    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    print(f"object-target arm: {_install_target_cap(trial)}", flush=True)

    from pole_lraspp_multimodal_fusion import model as model_module, train_fusion

    exp_dir = Path(args.experiment_dir).resolve()
    warm_state = torch.load(warm_path, map_location="cpu", weights_only=False)["model"]
    resume_exists = (exp_dir / "checkpoints" / str(trial["name"]) / "last.pt").is_file()

    original_build = model_module.build_multitask_fusion_lraspp

    def warm_started_build(**kwargs):
        built = original_build(**kwargs)
        # strict: every tensor of the parity-verified warm start, nothing else.
        built.load_state_dict({k: v.to(next(built.parameters()).device) for k, v in warm_state.items()})
        print(f"loaded {len(warm_state)} tensors strict from the parity-verified warm start",
              flush=True)
        return built

    original_count = train_fusion._count_parameters

    def audited_count(module):
        # train_fusion calls this immediately after every freeze flag is applied
        # and immediately before the optimizer is built.
        result = audit(module, exp_dir / "freeze_audit_v1.json")
        if result["status"] != "PASS":
            raise SystemExit(4)
        return original_count(module)

    if not resume_exists:
        model_module.build_multitask_fusion_lraspp = warm_started_build
        train_fusion.build_multitask_fusion_lraspp = warm_started_build
    train_fusion._count_parameters = audited_count

    sys.argv = [
        "train_fusion",
        "--config", str(Path(args.config).resolve()),
        "--experiment-dir", str(exp_dir),
        "--trial-json", json.dumps(trial),
        "--training-budget-hours", str(float(args.training_budget_hours)),
    ]
    train_fusion.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
