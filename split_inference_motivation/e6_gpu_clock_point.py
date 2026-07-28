"""E6 (GPU arm) - benchmark FULL-local vs SPLIT-front at whatever GPU clock is CURRENTLY locked.

Called once per clock point by run_e6_gpu.sh, which does the (root-only) clock locking.
Appends a row to results/E6_gpu_raw.csv so the sweep survives partial runs.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import torch

from common_setup import BackWrapper, FrontWrapper, build_full_model, get_real_input

OUT = Path(__file__).parent / "results"
CSV = OUT / "E6_gpu_raw.csv"
FIELDS = ["locked_mhz", "actual_mhz", "power_W", "full_fps", "full_p50_ms", "full_p95_ms",
          "front_fps", "front_p50_ms", "front_p95_ms", "back_fps", "back_p50_ms",
          "speedup_front_vs_full", "full_meets_10fps", "full_meets_20fps", "full_meets_30fps",
          "front_meets_10fps", "front_meets_20fps", "front_meets_30fps"]


def gpu_stat():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.current.graphics,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().split(",")
    try:
        return float(q[0]), float(q[1])
    except (IndexError, ValueError):
        return float("nan"), float("nan")


def sustained(module, inputs, seconds, warmup_s=2.0):
    with torch.inference_mode():
        t_w = time.perf_counter() + warmup_s
        while time.perf_counter() < t_w:
            module(*inputs)
        torch.cuda.synchronize()
        lat, n = [], 0
        t0 = time.perf_counter()
        stop = t0 + seconds
        while time.perf_counter() < stop:
            ti = time.perf_counter()
            module(*inputs)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - ti) * 1000.0)
            n += 1
        wall = time.perf_counter() - t0
    s = sorted(lat)
    return n / wall, s[len(s) // 2], s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-mhz", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    dev = torch.device("cuda")
    model, input_size, _ = build_full_model(dev)
    x, _, _ = get_real_input(dev, input_size)
    out_hw = (int(x.shape[-2]), int(x.shape[-1]))
    front = FrontWrapper(model).eval()
    back = BackWrapper(model, out_hw).eval()
    with torch.inference_mode():
        feats = front(x)

    f_fps, f50, f95 = sustained(model, (x,), args.seconds)
    r_fps, r50, r95 = sustained(front, (x,), args.seconds)
    b_fps, b50, _ = sustained(back, (feats,), args.seconds)
    actual, power = gpu_stat()

    row = {
        "locked_mhz": args.locked_mhz, "actual_mhz": actual, "power_W": power,
        "full_fps": round(f_fps, 2), "full_p50_ms": round(f50, 3), "full_p95_ms": round(f95, 3),
        "front_fps": round(r_fps, 2), "front_p50_ms": round(r50, 3), "front_p95_ms": round(r95, 3),
        "back_fps": round(b_fps, 2), "back_p50_ms": round(b50, 3),
        "speedup_front_vs_full": round(r_fps / f_fps, 3),
        "full_meets_10fps": f_fps >= 10, "full_meets_20fps": f_fps >= 20, "full_meets_30fps": f_fps >= 30,
        "front_meets_10fps": r_fps >= 10, "front_meets_20fps": r_fps >= 20, "front_meets_30fps": r_fps >= 30,
    }
    new = not CSV.exists()
    with open(CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  {args.locked_mhz:>5} MHz (actual {actual:.0f}, {power:.0f} W): "
          f"FULL {f_fps:8.2f} FPS | FRONT {r_fps:8.2f} FPS | {r_fps/f_fps:.2f}x")


if __name__ == "__main__":
    main()
