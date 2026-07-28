"""E1 - Local resource profile: FULL / FRONT / BACK on GPU, plus FULL+FRONT on CPU.

Params, FLOPs (fvcore), latency (warmup+timed, p50/p95), FPS, and sustained
GPU util/mem/power (pynvml, idle-subtracted).

Usage:  python e1_local_resource_profile.py [--iters 100] [--sustain-s 30]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from common_setup import CKPT,  BackTensorWrapper, BackWrapper, FrontWrapper, build_full_model, get_real_input

OUT = Path(__file__).parent / "results"
TARGET_FPS = 10.0


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def flops_of(module, inputs, label):
    """GMACs via fvcore. fvcore counts MACs (it calls them 'flops')."""
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(module, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
        skipped = dict(fca.unsupported_ops())
        return total / 1e9, skipped
    except Exception as exc:  # pragma: no cover
        import traceback
        print(f"  [ERROR] fvcore failed for {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return float("nan"), {}


def time_module(module, inputs, device, warmup, iters):
    """Returns list of per-iter latencies in ms."""
    is_cuda = device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            module(*inputs)
        if is_cuda:
            torch.cuda.synchronize()
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            module(*inputs)
            if is_cuda:
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
    return lat


def pctl(vals, q):
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


class GpuSampler:
    """Samples GPU util/mem/power at ~100ms via pynvml."""

    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample(self):
        u = self.nvml.nvmlDeviceGetUtilizationRates(self.h)
        mem = self.nvml.nvmlDeviceGetMemoryInfo(self.h)
        try:
            pw = self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        except Exception:
            pw = float("nan")
        return u.gpu, mem.used / 1e6, pw

    def collect(self, seconds, fn=None):
        t_end = time.perf_counter() + seconds
        rows = []
        while time.perf_counter() < t_end:
            if fn is not None:
                fn()
            rows.append(self.sample())
            time.sleep(0.1)
        util = statistics.mean(r[0] for r in rows)
        memv = statistics.mean(r[1] for r in rows)
        pws = [r[2] for r in rows if r[2] == r[2]]
        pw = statistics.mean(pws) if pws else float("nan")
        return util, memv, pw, len(rows)


def sustained(module, inputs, sampler, seconds, device):
    """Run inference continuously for `seconds` while sampling GPU telemetry."""
    stop = time.perf_counter() + seconds
    rows = []
    n = 0
    with torch.inference_mode():
        while time.perf_counter() < stop:
            for _ in range(10):
                module(*inputs)
                n += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            rows.append(sampler.sample())
    util = statistics.mean(r[0] for r in rows)
    memv = statistics.mean(r[1] for r in rows)
    pws = [r[2] for r in rows if r[2] == r[2]]
    pw = statistics.mean(pws) if pws else float("nan")
    return util, memv, pw, n, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--cpu-iters", type=int, default=30)
    ap.add_argument("--cpu-warmup", type=int, default=5)
    ap.add_argument("--sustain-s", type=float, default=30.0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    print(f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}")

    model, input_size, _ = build_full_model(dev)
    x, row, orig_hw = get_real_input(dev, input_size)
    out_hw = (int(x.shape[-2]), int(x.shape[-1]))
    print(f"input {tuple(x.shape)} from sample {row['sample_id']}")

    front = FrontWrapper(model).eval()
    back = BackWrapper(model, out_hw).eval()

    # --- validation: front+back == full, non-NaN ---
    with torch.inference_mode():
        ref = model(x)
        feats = front(x)
        chk = back(feats)
    maxdiff = max((ref[k] - chk[k]).abs().max().item() for k in ref)
    assert not any(torch.isnan(v).any().item() for v in ref.values()), "NaN in full output"
    assert maxdiff < 1e-3, f"front+back != full (maxdiff={maxdiff})"
    feat_elems = sum(v.numel() for v in feats.values())
    feat_shapes = {k: tuple(v.shape) for k, v in feats.items()}
    print(f"validation OK: front+back vs full maxdiff={maxdiff:.2e}; features {feat_shapes}")

    p_full = count_params(model)
    p_front = count_params(front)
    p_back = count_params(model.classifier) + (
        count_params(model.object_head) if getattr(model, "object_head", None) is not None else 0
    ) + (count_params(model.heatmap_head) if getattr(model, "heatmap_head", None) is not None else 0) + (
        count_params(model.reg_head) if getattr(model, "reg_head", None) is not None else 0
    )

    print("\n== FLOPs ==")
    g_full, sk_full = flops_of(model, (x,), "FULL")
    g_front, sk_front = flops_of(front, (x,), "FRONT")
    # fvcore traces with torch.jit and cannot take the OrderedDict feature input;
    # use the positional-tensor wrapper (identical compute).
    back_t = BackTensorWrapper(model, out_hw, list(feats.keys())).eval()
    # feats were produced under inference_mode; fvcore's torch.jit trace rejects
    # inference-mode tensors. Clone outside that context to get normal tensors.
    feats_trace = tuple(feats[k].clone() for k in feats)
    g_back, sk_back = flops_of(back_t, feats_trace, "BACK")
    print(f"  FULL {g_full:.3f} GMACs | FRONT {g_front:.3f} | BACK {g_back:.3f} | front+back {g_front+g_back:.3f}")
    if sk_full:
        print(f"  [note] fvcore unsupported ops (FULL): {sk_full}")

    print("\n== GPU latency ==")
    lat = {}
    lat["FULL"] = time_module(model, (x,), dev, args.warmup, args.iters)
    lat["FRONT"] = time_module(front, (x,), dev, args.warmup, args.iters)
    lat["BACK"] = time_module(back, (feats,), dev, args.warmup, args.iters)
    for k, v in lat.items():
        print(f"  {k:6s} mean {statistics.mean(v):7.3f}  p50 {pctl(v,.5):7.3f}  p95 {pctl(v,.95):7.3f} ms")
    print(f"  sanity: front p50 + back p50 = {pctl(lat['FRONT'],.5)+pctl(lat['BACK'],.5):.3f} "
          f"vs full p50 {pctl(lat['FULL'],.5):.3f} ms")

    print("\n== per-config GPU memory (torch allocator, isolated) ==")
    gpu_mem = {}
    for name, mod, inp in (("FULL", model, (x,)), ("FRONT", front, (x,)), ("BACK", back, (feats,))):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.inference_mode():
            mod(*inp)
        torch.cuda.synchronize()
        gpu_mem[name] = dict(
            peak_alloc_MB=torch.cuda.max_memory_allocated() / 1e6,
            activation_MB=(torch.cuda.max_memory_allocated() - base) / 1e6,
        )
        print(f"  {name:6s} peak_alloc {gpu_mem[name]['peak_alloc_MB']:7.1f} MB  "
              f"activations {gpu_mem[name]['activation_MB']:7.1f} MB")

    print("\n== GPU telemetry (sustained, max-rate) ==")
    print("  NOTE: raw wattage across configs is NOT comparable - a faster config runs")
    print("        more forwards/sec and so draws more power. Energy/FRAME is the honest")
    print("        metric (E2); it is derived as active_power x latency.")
    sampler = GpuSampler()
    print("  idle baseline (letting GPU clock down, 20s settle + 10s sample)...")
    time.sleep(20.0)
    idle_u, idle_m, idle_p, idle_n = sampler.collect(10.0)
    print(f"  IDLE util {idle_u:.1f}%  mem {idle_m:.0f}MB  power {idle_p:.1f}W  (n={idle_n})")
    tele = {}
    for name, mod, inp in (("FULL", model, (x,)), ("FRONT", front, (x,)), ("BACK", back, (feats,))):
        u, m, p, nfwd, ns = sustained(mod, inp, sampler, args.sustain_s, dev)
        rate = nfwd / args.sustain_s
        tele[name] = dict(util=u, mem=m, power=p, forwards=nfwd, samples=ns, fwd_per_s=rate)
        print(f"  {name:6s} util {u:5.1f}%  mem {m:7.0f}MB  power {p:6.1f}W  "
              f"rate {rate:8.1f} fwd/s  ({nfwd} fwd, {ns} samples)")
        time.sleep(5.0)

    print("\n== GPU power at fixed 10 FPS duty cycle (deployment condition) ==")
    tele10 = {}
    for name, mod, inp in (("FULL", model, (x,)), ("FRONT", front, (x,))):
        period = 1.0 / TARGET_FPS
        t_end = time.perf_counter() + args.sustain_s
        srows, n = [], 0
        with torch.inference_mode():
            while time.perf_counter() < t_end:
                t0 = time.perf_counter()
                mod(*inp)
                torch.cuda.synchronize()
                n += 1
                srows.append(sampler.sample())
                rest = period - (time.perf_counter() - t0)
                if rest > 0:
                    time.sleep(rest)
        u = statistics.mean(r[0] for r in srows)
        pw = statistics.mean(r[2] for r in srows if r[2] == r[2])
        tele10[name] = dict(util=u, power=pw, frames=n)
        print(f"  {name:6s} @10FPS  util {u:5.1f}%  power {pw:6.1f}W  ({n} frames)")
        time.sleep(5.0)

    print("\n== CPU latency (SWaP-C proxy), thread sweep ==")
    print("  Host CPU is a 24-core desktop part; a vehicle SoC has far fewer/weaker cores.")
    print("  The low-thread rows are the embedded-relevant proxy, NOT the 24-thread row.")
    cdev = torch.device("cpu")
    max_threads = torch.get_num_threads()
    model_c, _, _ = build_full_model(cdev)
    x_c = x.detach().to(cdev)
    front_c = FrontWrapper(model_c).eval()
    with torch.inference_mode():
        feats_c = front_c(x_c)
    back_c = BackWrapper(model_c, out_hw).eval()

    thread_counts = [t for t in (max_threads, 8, 4, 2, 1) if t <= max_threads]
    thread_counts = sorted(set(thread_counts), reverse=True)
    cpu_sweep = {}
    for nt in thread_counts:
        torch.set_num_threads(nt)
        res = {}
        for name, mod, inp in (("FULL", model_c, (x_c,)), ("FRONT", front_c, (x_c,)), ("BACK", back_c, (feats_c,))):
            v = time_module(mod, inp, cdev, args.cpu_warmup, args.cpu_iters)
            res[name] = v
            print(f"  threads={nt:2d} {name:6s} mean {statistics.mean(v):8.2f}  p50 {pctl(v,.5):8.2f}  "
                  f"p95 {pctl(v,.95):8.2f} ms -> {1000.0/pctl(v,.5):6.2f} FPS")
        cpu_sweep[nt] = res
    torch.set_num_threads(max_threads)
    cpu_lat = cpu_sweep[max_threads]
    nthreads = max_threads

    # ---- write artifacts ----
    def fps(ms):
        return 1000.0 / ms if ms > 0 else float("nan")

    rows = []
    for name, params, gmacs in (("FULL", p_full, g_full), ("FRONT", p_front, g_front), ("BACK", p_back, g_back)):
        L, C = lat[name], cpu_lat[name]
        t = tele[name]
        rows.append({
            "config": name,
            "params_M": round(params / 1e6, 4),
            "size_fp32_MB": round(params * 4 / 1e6, 3),
            "size_int8_MB": round(params * 1 / 1e6, 3),
            "gmacs": round(gmacs, 4),
            "gpu_lat_mean_ms": round(statistics.mean(L), 4),
            "gpu_lat_p50_ms": round(pctl(L, .5), 4),
            "gpu_lat_p95_ms": round(pctl(L, .95), 4),
            "gpu_fps_p50": round(fps(pctl(L, .5)), 2),
            "gpu_util_pct_maxrate": round(t["util"], 2),
            "gpu_fwd_per_s_maxrate": round(t["fwd_per_s"], 1),
            "gpu_peak_alloc_MB": round(gpu_mem[name]["peak_alloc_MB"], 1),
            "gpu_activation_MB": round(gpu_mem[name]["activation_MB"], 1),
            "gpu_power_W_maxrate": round(t["power"], 2),
            "gpu_power_active_W_maxrate": round(t["power"] - idle_p, 2),
            "energy_per_frame_J": round((t["power"] - idle_p) * pctl(L, .5) / 1000.0, 5),
            "gpu_power_W_at10fps": round(tele10[name]["power"], 2) if name in tele10 else "",
            "gpu_power_active_W_at10fps": round(tele10[name]["power"] - idle_p, 2) if name in tele10 else "",
            "cpu_lat_mean_ms": round(statistics.mean(C), 3),
            "cpu_lat_p50_ms": round(pctl(C, .5), 3),
            "cpu_lat_p95_ms": round(pctl(C, .95), 3),
            "cpu_fps_p50": round(fps(pctl(C, .5)), 3),
            "meets_10fps_gpu": fps(pctl(L, .5)) >= TARGET_FPS,
            "meets_10fps_cpu": fps(pctl(C, .5)) >= TARGET_FPS,
        })

    with open(OUT / "E1_raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cpu_threads": nthreads,
        "input_shape": list(x.shape),
        "input_size_WH": list(input_size),
        "sample_id": row["sample_id"],
        "original_frame_hw": list(orig_hw),
        "radar_points": row.get("radar_points"),
        "checkpoint": str(CKPT),
        "feature_shapes": {k: list(v) for k, v in feat_shapes.items()},
        "feature_elems": feat_elems,
        "feature_fp32_MB": round(feat_elems * 4 / 1e6, 4),
        "feature_u8_MB": round(feat_elems / 1e6, 4),
        "front_back_maxdiff": maxdiff,
        "idle": {"util_pct": idle_u, "mem_MB": idle_m, "power_W": idle_p, "samples": idle_n},
        "iters": args.iters, "warmup": args.warmup,
        "cpu_iters": args.cpu_iters, "sustain_s": args.sustain_s,
        "fvcore_unsupported": {"FULL": {k: v for k, v in sk_full.items()},
                               "FRONT": {k: v for k, v in sk_front.items()},
                               "BACK": {k: v for k, v in sk_back.items()}},
        "gpu_lat_all_ms": {k: v for k, v in lat.items()},
        "cpu_lat_all_ms": {k: v for k, v in cpu_lat.items()},
        "cpu_thread_sweep_ms": {str(nt): {k: v for k, v in res.items()} for nt, res in cpu_sweep.items()},
        "cpu_thread_sweep_p50_fps": {
            str(nt): {k: round(1000.0 / pctl(v, .5), 3) for k, v in res.items()} for nt, res in cpu_sweep.items()
        },
        "gpu_mem_per_config": gpu_mem,
        "telemetry_maxrate": tele,
        "telemetry_at_10fps": tele10,
        "telemetry_note": (
            "Max-rate wattage is NOT comparable across configs (faster config = more forwards/s = "
            "more power). Use energy_per_frame_J (active power x p50 latency) or the fixed-10FPS rows."
        ),
    }
    with open(OUT / "E1_raw.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {OUT/'E1_raw.csv'} and {OUT/'E1_raw.json'}")


if __name__ == "__main__":
    main()
