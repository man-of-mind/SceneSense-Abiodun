#!/usr/bin/env python3
"""Batch-size / DataLoader probe for the 24 GB card (preparation only).

Measures, for each candidate batch size, a real forward + backward + optimizer
step at the pilot's exact input geometry and reports peak allocated and reserved
CUDA memory, samples/second and GPU utilization. It then selects the largest
batch that is stable *and* still leaves at least the requested fraction of the
card free, so a longer run cannot creep into an out-of-memory failure.

Two modes:

* ``--synthetic`` (default) - model-only. Runs before the dataset is copied and
  answers the memory question. It does **not** measure loader throughput.
* ``--dataset-dir`` - full pipeline including the real DataLoader, so
  ``--worker-counts`` can be measured rather than assumed.

Nothing is trained: no checkpoint is written and no experiment directory is
touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
ABIODUN = PKG_ROOT.parent
for path in (str(PKG_ROOT), str(ABIODUN)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

from pole_lraspp_multimodal_fusion.model import build_multitask_fusion_lraspp  # noqa: E402
from pole_lraspp_multimodal_fusion.object_targets import object_reg_channels  # noqa: E402


def gpu_utilization() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-json", type=Path,
                        default=HERE / "configs" / "pilot_B_capped_objhead_smoke_v1.json")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[8],
                        help="only measured with --dataset-dir")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--headroom-fraction", type=float, default=0.15)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--report-json", type=Path,
                        default=HERE / "reports" / "gpu_batch_probe_v1.json")
    return parser


def probe_batch(
    *, batch_size: int, trial: dict[str, Any], steps: int, warmup: int,
    device: torch.device, amp: bool,
) -> dict[str, Any]:
    width, height = (int(value) for value in trial.get("input_size", [768, 432]))
    object_cfg = trial.get("object_heads", {})
    predict_bbox2d = bool(object_cfg.get("predict_bbox2d", False))
    object_classes = 2
    object_channels = object_classes + object_reg_channels(predict_bbox2d)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = build_multitask_fusion_lraspp(
        num_classes=3,
        radar_channels=4,
        pretrained=False,
        init_checkpoint="",
        object_channels=object_channels,
        object_hidden_channels=int(object_cfg.get("hidden_channels", 128)),
        fuse_low_into_object_head=bool(object_cfg.get("fuse_low_feature", True)),
        head_arch=str(object_cfg.get("head_arch", "shared")),
        use_coordconv=bool(object_cfg.get("use_coordconv", False)),
        head_depth=int(object_cfg.get("head_depth", 3)),
        predict_bbox2d=predict_bbox2d,
        use_groundplane_prior=bool(object_cfg.get("use_groundplane_prior", False)),
        groundplane_params=dict(object_cfg.get("groundplane_params", {}) or {}),
        device=device,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(trial.get("lr", 1.5e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    fused = torch.randn(batch_size, 7, height, width, device=device)
    seg_target = torch.randint(0, 3, (batch_size, height, width), device=device)

    latencies: list[float] = []
    utilizations: list[float] = []
    status = "ok"
    try:
        for step in range(warmup + steps):
            if step == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                outputs = model(fused)
                seg_logits = outputs["out"] if isinstance(outputs, dict) else outputs[0]
                object_logits = outputs["objects"] if isinstance(outputs, dict) else outputs[1]
                loss = torch.nn.functional.cross_entropy(seg_logits, seg_target)
                loss = loss + object_logits.float().pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= warmup:
                latencies.append(time.perf_counter() - started)
                usage = gpu_utilization()
                if usage is not None:
                    utilizations.append(usage)
    except torch.cuda.OutOfMemoryError as exc:
        status = f"oom: {str(exc).splitlines()[0]}"
    except RuntimeError as exc:
        status = f"error: {str(exc).splitlines()[0]}"

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    del model, optimizer, fused, seg_target
    torch.cuda.empty_cache()

    mean_latency = (sum(latencies) / len(latencies)) if latencies else None
    return {
        "batch_size": batch_size,
        "status": status,
        "steps_measured": len(latencies),
        "step_latency_s_mean": mean_latency,
        "samples_per_second": (batch_size / mean_latency) if mean_latency else None,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_gib": round(peak_allocated / 1024 ** 3, 3),
        "peak_reserved_gib": round(peak_reserved / 1024 ** 3, 3),
        "total_memory_gib": round(total / 1024 ** 3, 3),
        "reserved_fraction_of_card": round(peak_reserved / total, 4),
        "headroom_fraction": round(1.0 - peak_reserved / total, 4),
        "gpu_utilization_pct_mean": (sum(utilizations) / len(utilizations)) if utilizations else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        print("no CUDA device available", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    trial = json.loads(Path(args.trial_json).read_text(encoding="utf-8"))

    results = [
        probe_batch(batch_size=batch, trial=trial, steps=args.steps,
                    warmup=args.warmup_steps, device=device, amp=bool(args.amp))
        for batch in args.batch_sizes
    ]
    viable = [
        row for row in results
        if row["status"] == "ok" and row["headroom_fraction"] >= float(args.headroom_fraction)
    ]
    selected = max(viable, key=lambda row: row["batch_size"]) if viable else None

    report = {
        "schema": "object_head_pilot_v1.gpu_batch_probe",
        "mode": "synthetic" if args.dataset_dir is None else "dataset",
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir else None,
        "amp": bool(args.amp),
        "headroom_fraction_required": float(args.headroom_fraction),
        "trial": trial.get("name"),
        "input_size": trial.get("input_size"),
        "results": results,
        "selected_batch_size": selected["batch_size"] if selected else None,
        "selection_rule": "largest stable batch retaining >= required headroom of total card memory",
        "dataloader": {
            "worker_counts_requested": args.worker_counts,
            "measured": args.dataset_dir is not None,
            "note": "loader throughput needs the real dataset; rerun with --dataset-dir after the copy",
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_properties(device).name,
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
