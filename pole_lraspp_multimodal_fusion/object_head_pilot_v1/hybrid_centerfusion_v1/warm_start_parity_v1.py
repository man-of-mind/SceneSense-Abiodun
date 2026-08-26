#!/usr/bin/env python3
"""Phase C part 1: numeric warm-start parity of the hybrid against the baseline.

Builds the frozen baseline and the warm-started hybrid, runs both in eval / fp32
on real Route B *validation* batches, and compares the parts the hybrid retains:

* fused ``low`` / ``high`` features vs the baseline backbone features
* segmentation logits
* the coarse 1/8 object logits vs the baseline object head

Tolerances are the ones registered in ``HYBRID_NOAE_PILOT_PLAN.md`` and are read
from the constants below - they are not arguments, so they cannot be relaxed
from a command line after seeing a number.

Also writes ``warm_start.pt``: the un-trained hybrid in the production checkpoint
schema, so part 2 of Phase C can decode it on the full validation split through
the untouched evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hybrid_model_v1 import HEAD_ARCH_NAME, build_hybrid_centerfusion  # noqa: E402

from pole_lraspp_multimodal_fusion.common import load_config, read_manifest  # noqa: E402
from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import load_object_boxes  # noqa: E402
from pole_lraspp_multimodal_fusion.train_fusion import (  # noqa: E402
    FusionPoleMultiTaskDataset,
    _deep_merge_dicts,
    split_rows,
)

# Registered relative tolerance: max|delta| / max|baseline| over the compared tensor.
RELATIVE_TOLERANCE = 1e-4

# The gate is evaluated in strict fp32. TF32 is a property of the Ampere+ conv
# kernels, not of the warm start: the baseline computes one 7-channel stem
# convolution where the hybrid computes a 3-channel and a 4-channel one and adds
# them, and at TF32's 10-bit mantissa those two groupings simply round
# differently. The TF32 numbers are still measured and reported next to a
# control - the *baseline's own* TF32-vs-fp32 self-difference - so the residual
# can be attributed rather than assumed.


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-json", required=True, type=Path)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    trial = json.loads(args.trial_json.read_text(encoding="utf-8"))
    train_cfg = config["training"]
    fusion_cfg = config.get("fusion", {})
    object_cfg = _deep_merge_dicts(dict(config.get("object_heads", {})), trial.get("object_heads"))

    input_width, input_height = [int(v) for v in trial["input_size"]]
    num_classes = int(train_cfg.get("num_classes", 3))
    radar_channels = int(fusion_cfg.get("radar_channels", 4))
    object_class_names = tuple(object_cfg.get("object_classes", ("vehicle", "person")))
    predict_bbox2d = bool(object_cfg.get("predict_bbox2d", True))
    object_channels = len(object_class_names) + (12 if predict_bbox2d else 10)
    head_depth = int(object_cfg.get("head_depth", 3))
    hidden_channels = int(object_cfg.get("hidden_channels", 128))
    baseline_ckpt = str(trial["init_rgb_checkpoint"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    baseline = build_multitask_fusion_lraspp(
        num_classes=num_classes, radar_channels=radar_channels, pretrained=False,
        object_channels=object_channels, object_hidden_channels=hidden_channels,
        fuse_low_into_object_head=True, head_arch="shared", use_coordconv=False,
        head_depth=head_depth, predict_bbox2d=predict_bbox2d,
        use_groundplane_prior=False, groundplane_params={}, device=device,
    ).to(device)
    source = torch.load(baseline_ckpt, map_location=device, weights_only=False)
    baseline.load_state_dict(source["model"])
    baseline.eval()

    hybrid = build_hybrid_centerfusion(
        num_classes=num_classes, radar_channels=radar_channels,
        object_channels=object_channels, object_hidden_channels=hidden_channels,
        head_depth=head_depth, predict_bbox2d=predict_bbox2d,
        init_checkpoint=baseline_ckpt, device=device,
    ).to(device)
    hybrid.eval()
    warm_start = hybrid.warm_start_report

    dataset_dir = Path(args.experiment_dir).resolve() / "dataset"
    rows = read_manifest(dataset_dir / "manifest.csv")
    splits = split_rows(rows)
    if splits["test"]:
        raise RuntimeError("locked Route B test split is present in this view; refusing to run")
    object_rows = load_object_boxes(dataset_dir / "object_boxes.csv")
    val_ds = FusionPoleMultiTaskDataset(
        dataset_dir, splits["val"], object_rows, (input_width, input_height),
        object_cfg, augment_strength="off",
    )
    loader = torch.utils.data.DataLoader(val_ds, batch_size=int(args.batch_size),
                                         shuffle=False, num_workers=4)

    worst: Dict[str, Dict[str, float]] = {}

    def record(name: str, hyb: torch.Tensor, ref: torch.Tensor) -> None:
        delta = float((hyb.float() - ref.float()).abs().max().item())
        scale = float(ref.float().abs().max().item())
        relative = delta / max(scale, 1e-12)
        current = worst.get(name)
        if current is None or relative > current["relative"]:
            worst[name] = {"max_abs_delta": delta, "reference_max_abs": scale, "relative": relative}

    def _forward_pair(tensors: torch.Tensor):
        base_features = baseline.backbone(tensors)
        base_out = baseline(tensors)
        base_coarse = baseline.object_head(baseline._object_input(base_features))
        fused = hybrid.encode_front(tensors[:, :3], tensors[:, 3: 3 + radar_channels])
        hybrid_out = hybrid.decode_tail(fused, (input_height, input_width))
        hybrid_coarse = hybrid.object_head(torch.cat(
            [fused["low"], hybrid._match(fused["high"], fused["low"])], dim=1))
        return (
            {"feature_low": base_features["low"], "feature_high": base_features["high"],
             "segmentation_logits": base_out["out"], "coarse_object_logits_1_8": base_coarse},
            {"feature_low": fused["low"], "feature_high": fused["high"],
             "segmentation_logits": hybrid_out["out"], "coarse_object_logits_1_8": hybrid_coarse},
        )

    def _set_tf32(enabled: bool) -> None:
        torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
        torch.backends.cudnn.allow_tf32 = bool(enabled)

    tf32_control: Dict[str, Dict[str, float]] = {}

    def record_into(store, name, hyb, ref):
        delta = float((hyb.float() - ref.float()).abs().max().item())
        scale = float(ref.float().abs().max().item())
        relative = delta / max(scale, 1e-12)
        current = store.get(name)
        if current is None or relative > current["relative"]:
            store[name] = {"max_abs_delta": delta, "reference_max_abs": scale, "relative": relative}

    frames = 0
    with torch.inference_mode():
        for index, (tensors, _masks, _targets) in enumerate(loader):
            if index >= int(args.batches):
                break
            tensors = tensors.to(device).float()
            frames += int(tensors.shape[0])

            # --- gate: strict fp32, TF32 off ---
            _set_tf32(False)
            base_fp32, hybrid_fp32 = _forward_pair(tensors)
            for name in base_fp32:
                record(name, hybrid_fp32[name], base_fp32[name])

            # --- control: TF32 on. Reported, never gated. ---
            _set_tf32(True)
            base_tf32, hybrid_tf32 = _forward_pair(tensors)
            for name in base_fp32:
                record_into(tf32_control, f"hybrid_tf32_vs_baseline_tf32.{name}",
                            hybrid_tf32[name], base_tf32[name])
                record_into(tf32_control, f"baseline_tf32_vs_baseline_fp32.{name}",
                            base_tf32[name], base_fp32[name])
            _set_tf32(False)

    failures = [
        f"{name}: relative {values['relative']:.3e} > {RELATIVE_TOLERANCE:.0e} "
        f"(max abs delta {values['max_abs_delta']:.3e} on scale {values['reference_max_abs']:.3e})"
        for name, values in sorted(worst.items())
        if values["relative"] > RELATIVE_TOLERANCE
    ]

    # The un-trained hybrid, in the production checkpoint schema, so the full-split
    # decode in part 2 goes through the untouched evaluator.
    snapshot: Dict[str, Any] = {
        "model": hybrid.state_dict(),
        "epoch": -1,
        "training_epoch_index": -1,
        "trial": trial,
        "config": config,
        "input_size": [input_width, input_height],
        "radar_channels": radar_channels,
        "object_channels": object_channels,
        "object_predict_bbox2d": predict_bbox2d,
        "object_class_names": list(object_class_names),
        "fuse_low_into_object_head": True,
        "object_head_arch": HEAD_ARCH_NAME,
        "object_use_coordconv": False,
        "object_head_depth": head_depth,
        "object_use_groundplane_prior": False,
        "object_groundplane_params": {},
        "init_rgb_checkpoint": baseline_ckpt,
        "init_object_checkpoint": "",
        "model_task": "segmentation_plus_learned_object_localization",
        "note": "warm start only; zero gradient steps taken",
    }
    warm_path = Path(args.experiment_dir).resolve() / "checkpoints" / trial["name"] / "warm_start.pt"
    warm_path.parent.mkdir(parents=True, exist_ok=True)
    if not warm_path.exists():
        torch.save(snapshot, warm_path)

    result = {
        "check": "hybrid_centerfusion_v1_warm_start_parity",
        "status": "PASS" if not failures else "WARM_START_PARITY_FAILED",
        "failures": failures,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "baseline_checkpoint": baseline_ckpt,
        "frames_compared": frames,
        "tensor_parity_fp32_strict": worst,
        "tf32_control": tf32_control,
        "tf32_control_note": (
            "not gated. hybrid_tf32_vs_baseline_tf32 is the delta a TF32 conv path "
            "produces; baseline_tf32_vs_baseline_fp32 is the same baseline's own TF32 "
            "rounding against itself, i.e. the arithmetic noise floor of this GPU for "
            "this model."
        ),
        "warm_start_checkpoint": str(warm_path),
        "warm_start_mapping": warm_start,
        "splits": {k: len(v) for k, v in splits.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "warm_start_mapping"},
                     indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
