"""E6 - Compute-constrained crossover: full-local vs split-front sustained FPS.

Sweeps an on-vehicle compute budget (pinned CPU cores + intra-op threads) and measures
SUSTAINED throughput (not isolated single-shot latency, as E1 did) for:
    FULL-local  = backbone + heads on the car
    SPLIT-front = backbone only, heads offloaded to the edge
Reports the crossover budget where full-local misses a real-time deadline but split-front
still meets it, at 10 / 20 / 30 FPS.

Honest framing (see E6_compute_crossover.md): total system compute is UNCHANGED - it is
relocated to the edge. The claim is about the on-vehicle budget only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from common_setup import BackWrapper, FrontWrapper, build_full_model, get_real_input

OUT = Path(__file__).parent / "results"
DEADLINES = (10.0, 20.0, 30.0)


def pin(n_cores):
    """Restrict the process to the first n logical CPUs (a compute-budget proxy)."""
    try:
        os.sched_setaffinity(0, set(range(n_cores)))
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        print(f"  [warn] could not set affinity: {exc}")
        return None


def sustained_fps(module, inputs, seconds, warmup_s=2.0):
    """Run continuously; return (sustained_fps, p50_ms, p95_ms, n)."""
    with torch.inference_mode():
        t_w = time.perf_counter() + warmup_s
        while time.perf_counter() < t_w:
            module(*inputs)
        lat = []
        n = 0
        t0 = time.perf_counter()
        stop = t0 + seconds
        while time.perf_counter() < stop:
            ti = time.perf_counter()
            module(*inputs)
            lat.append((time.perf_counter() - ti) * 1000.0)
            n += 1
        wall = time.perf_counter() - t0
    s = sorted(lat)
    p50 = s[len(s) // 2]
    p95 = s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]
    return n / wall, p50, p95, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    dev = torch.device("cpu")
    print(f"E6 compute crossover — {platform.processor() or 'CPU'}, "
          f"{os.cpu_count()} logical CPUs, torch {torch.__version__}")
    print("NOTE host is Core Ultra 9 285K: logical CPUs 0-7 are P-cores (5.0-5.1 GHz), "
          "8-23 are E-cores (4.6 GHz).")
    print(f"sustained window {args.seconds}s per point (warmup {args.warmup}s)\n")

    model, input_size, _ = build_full_model(dev)
    x, row, _ = get_real_input(dev, input_size)
    out_hw = (int(x.shape[-2]), int(x.shape[-1]))
    front = FrontWrapper(model).eval()
    back = BackWrapper(model, out_hw).eval()
    with torch.inference_mode():
        ref = model(x)
        feats = front(x)
        chk = back(feats)
    maxdiff = max((ref[k] - chk[k]).abs().max().item() for k in ref)
    assert maxdiff < 1e-3, f"front+back != full ({maxdiff})"
    print(f"validation: front+back vs full maxdiff={maxdiff:.2e}; input {tuple(x.shape)}\n")

    rows = []
    print(f"{'threads':>8} {'cores':>7} | {'FULL fps':>9} {'p50 ms':>8} | "
          f"{'FRONT fps':>10} {'p50 ms':>8} | {'speedup':>8}")
    print("-" * 74)
    for nt in args.threads:
        if nt > os.cpu_count():
            continue
        cpus = pin(nt)
        torch.set_num_threads(nt)
        torch.set_num_interop_threads(1) if nt == 1 else None
        f_fps, f_p50, f_p95, f_n = sustained_fps(model, (x,), args.seconds, args.warmup)
        r_fps, r_p50, r_p95, r_n = sustained_fps(front, (x,), args.seconds, args.warmup)
        b_fps, b_p50, b_p95, b_n = sustained_fps(back, (feats,), args.seconds, args.warmup)
        rows.append({
            "threads": nt, "pinned_cpus": len(cpus) if cpus else nt,
            "full_fps": round(f_fps, 3), "full_p50_ms": round(f_p50, 3), "full_p95_ms": round(f_p95, 3),
            "front_fps": round(r_fps, 3), "front_p50_ms": round(r_p50, 3), "front_p95_ms": round(r_p95, 3),
            "back_fps": round(b_fps, 3), "back_p50_ms": round(b_p50, 3),
            "speedup_front_vs_full": round(r_fps / f_fps, 3),
            **{f"full_meets_{int(d)}fps": bool(f_fps >= d) for d in DEADLINES},
            **{f"front_meets_{int(d)}fps": bool(r_fps >= d) for d in DEADLINES},
        })
        print(f"{nt:>8} {len(cpus) if cpus else nt:>7} | {f_fps:>9.2f} {f_p50:>8.2f} | "
              f"{r_fps:>10.2f} {r_p50:>8.2f} | {r_fps/f_fps:>7.2f}x")

    # restore
    pin(os.cpu_count())

    print("\n== crossover analysis ==")
    cross = {}
    for d in DEADLINES:
        band = [r["threads"] for r in rows if not r[f"full_meets_{int(d)}fps"] and r[f"front_meets_{int(d)}fps"]]
        full_min = [r["threads"] for r in rows if r[f"full_meets_{int(d)}fps"]]
        front_min = [r["threads"] for r in rows if r[f"front_meets_{int(d)}fps"]]
        cross[str(int(d))] = {
            "full_needs_threads": min(full_min) if full_min else None,
            "front_needs_threads": min(front_min) if front_min else None,
            "crossover_band_threads": band,
        }
        fm = min(full_min) if full_min else "never"
        rm = min(front_min) if front_min else "never"
        print(f"  {int(d):>2} FPS: full-local needs >={fm} threads | split-front needs >={rm} "
              f"| SPLIT-ONLY BAND = {band if band else 'none'}")

    with open(OUT / "E6_raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    meta = {
        "host_cpu": "Intel Core Ultra 9 285K (8 P-cores 5.0-5.1GHz + 16 E-cores 4.6GHz)",
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "sustained_window_s": args.seconds, "warmup_s": args.warmup,
        "input_shape": list(x.shape), "sample_id": row["sample_id"],
        "front_back_maxdiff": maxdiff,
        "gmacs": {"full": 10.164, "front": 2.445, "back": 7.719},
        "deadlines_fps": list(DEADLINES),
        "rows": rows, "crossover": cross,
        "gpu_clock_sweep": {"attempted": True, "available": False,
                            "reason": "nvidia-smi -lgc requires root: 'The current user does not have "
                                      "permission to change clocks'. CPU sweep only."},
    }
    (OUT / "E6_raw.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT/'E6_raw.csv'} and {OUT/'E6_raw.json'}")


if __name__ == "__main__":
    main()
