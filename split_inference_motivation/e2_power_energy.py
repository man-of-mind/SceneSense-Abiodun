"""E2 - Power / energy per frame, derived reproducibly from E1_raw.json.

No new measurement: every number here is a deterministic function of E1's raw data,
so the two experiments cannot drift out of sync.

Emits results/E2_raw.json (consumed by E2_power_energy.md).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "results"
TARGET_FPS = 10.0

# NVIDIA Jetson Orin series module configurable power modes (whole-module TDP, W).
# SOURCE: NVIDIA Jetson Orin series module documentation / power-mode tables.
# ACTION BEFORE PUBLICATION: verify each against the current NVIDIA Jetson Orin Series
# Modules Data Sheet revision; these are quoted from product specs, not measured here.
JETSON_BUDGETS_W = {
    "Jetson Orin Nano 8GB": [7, 15],
    "Jetson Orin NX 8GB": [10, 15, 20],
    "Jetson Orin NX 16GB": [10, 15, 25],
    "Jetson AGX Orin 32GB": [15, 30, 40],
    "Jetson AGX Orin 64GB": [15, 30, 50, 60],
}


def p50(v):
    s = sorted(v)
    return s[len(s) // 2]


def main():
    e1 = json.loads((OUT / "E1_raw.json").read_text())
    idle = e1["idle"]["power_W"]
    tele = e1["telemetry_maxrate"]
    lat = e1["gpu_lat_all_ms"]
    sweep = e1["cpu_thread_sweep_ms"]

    # ---- GPU energy per frame, two independent derivations ----
    gpu = {}
    for k in ("FULL", "FRONT", "BACK"):
        active = tele[k]["power"] - idle
        e_lat = active * p50(lat[k]) / 1000.0          # W * s
        e_rate = active / tele[k]["fwd_per_s"]          # W / (frames/s)
        gpu[k] = {
            "active_power_W": round(active, 2),
            "p50_latency_ms": round(p50(lat[k]), 4),
            "throughput_fwd_s": round(tele[k]["fwd_per_s"], 1),
            "energy_per_frame_J_via_latency": round(e_lat, 5),
            "energy_per_frame_J_via_rate": round(e_rate, 5),
            "energy_per_frame_J_mean": round((e_lat + e_rate) / 2, 5),
            "sustained_W_at_10fps": round((e_lat + e_rate) / 2 * TARGET_FPS, 3),
        }
    add = gpu["FRONT"]["energy_per_frame_J_mean"] + gpu["BACK"]["energy_per_frame_J_mean"]
    gpu_consistency = {
        "front_plus_back_J": round(add, 5),
        "full_J": gpu["FULL"]["energy_per_frame_J_mean"],
        "pct_diff": round(100 * abs(add - gpu["FULL"]["energy_per_frame_J_mean"])
                          / gpu["FULL"]["energy_per_frame_J_mean"], 2),
    }
    gpu_saving_pct = round(
        100 * (gpu["FULL"]["energy_per_frame_J_mean"] - gpu["FRONT"]["energy_per_frame_J_mean"])
        / gpu["FULL"]["energy_per_frame_J_mean"], 1)

    # ---- CPU work per frame (core-ms/frame) : embedded energy proxy, no RAPL needed ----
    # At 1 thread there is no parallel overhead, so core-ms == total CPU work per frame.
    cpu = {}
    for nt, res in sweep.items():
        cpu[nt] = {k: round(int(nt) * p50(v), 2) for k, v in res.items()}
    one = cpu.get("1")
    cpu_saving_ratio = round(one["FULL"] / one["FRONT"], 3) if one else None
    cpu_saving_pct = round(100 * (one["FULL"] - one["FRONT"]) / one["FULL"], 1) if one else None

    # ---- budget fit: GPU compute power needed to sustain 10 FPS ----
    budget = {}
    for dev, modes in JETSON_BUDGETS_W.items():
        budget[dev] = {
            "modes_W": modes,
            "min_mode_W": min(modes),
            "full_local_gpu_compute_W_at_10fps": gpu["FULL"]["sustained_W_at_10fps"],
            "split_front_gpu_compute_W_at_10fps": gpu["FRONT"]["sustained_W_at_10fps"],
            "delta_W": round(gpu["FULL"]["sustained_W_at_10fps"] - gpu["FRONT"]["sustained_W_at_10fps"], 3),
            "delta_pct_of_min_mode": round(
                100 * (gpu["FULL"]["sustained_W_at_10fps"] - gpu["FRONT"]["sustained_W_at_10fps"]) / min(modes), 1),
        }

    out = {
        "derived_from": "E1_raw.json",
        "idle_gpu_W": idle,
        "target_fps": TARGET_FPS,
        "gpu_energy": gpu,
        "gpu_additivity_check": gpu_consistency,
        "gpu_energy_saving_pct_by_split": gpu_saving_pct,
        "cpu_core_ms_per_frame": cpu,
        "cpu_work_saving_ratio_1thread": cpu_saving_ratio,
        "cpu_work_saving_pct_1thread": cpu_saving_pct,
        "jetson_budgets": budget,
        "cpu_energy_measurement": {
            "attempted": True,
            "available": False,
            "reason": "RAPL /sys/class/powercap/intel-rapl:0/energy_uj not readable (root-only); "
                      "perf 'power' PMU unavailable (perf_event_paranoid=4). "
                      "core-ms/frame used as the CPU energy proxy instead.",
        },
        "caveats": [
            "Measured on RTX 5090 + Core Ultra 9 285K, NOT a vehicle SoC. Ratios transfer; absolute W does not.",
            "Jetson TDP figures are quoted product specs, not measured here - verify against the current "
            "NVIDIA Jetson Orin Series Modules Data Sheet before publication.",
            "sustained_W_at_10fps is GPU COMPUTE power only (idle-subtracted); it is not whole-module power "
            "and is therefore not directly comparable to a Jetson module TDP.",
        ],
    }
    (OUT / "E2_raw.json").write_text(json.dumps(out, indent=2))

    # ---- console report ----
    print("== GPU energy per frame (idle-subtracted, idle = %.1f W) ==" % idle)
    for k, v in gpu.items():
        print(f"  {k:6s} active {v['active_power_W']:6.1f}W  "
              f"E/frame {v['energy_per_frame_J_mean']:.4f} J  "
              f"(lat {v['energy_per_frame_J_via_latency']:.4f} / rate {v['energy_per_frame_J_via_rate']:.4f})  "
              f"-> {v['sustained_W_at_10fps']:.2f} W @10FPS")
    print(f"  additivity: front+back {gpu_consistency['front_plus_back_J']:.4f} J vs full "
          f"{gpu_consistency['full_J']:.4f} J ({gpu_consistency['pct_diff']}% diff)")
    print(f"  SPLIT SAVES {gpu_saving_pct}% of on-car GPU energy/frame")

    print("\n== CPU work per frame (core-ms/frame) - embedded energy proxy ==")
    for nt in sorted(cpu, key=lambda a: -int(a)):
        r = cpu[nt]
        print(f"  threads={nt:>2s}  FULL {r['FULL']:8.2f}  FRONT {r['FRONT']:8.2f}  BACK {r['BACK']:8.2f} core-ms")
    print(f"  at 1 thread (no parallel overhead): split cuts on-car CPU work "
          f"{cpu_saving_ratio}x ({cpu_saving_pct}%)")

    print("\n== Budget fit: GPU compute power to sustain 10 FPS ==")
    print(f"  full-local {gpu['FULL']['sustained_W_at_10fps']:.2f} W  |  "
          f"split-front {gpu['FRONT']['sustained_W_at_10fps']:.2f} W  |  "
          f"delta {budget['Jetson Orin NX 16GB']['delta_W']:.2f} W")
    for dev, b in budget.items():
        print(f"  {dev:24s} modes {str(b['modes_W']):18s} delta = {b['delta_pct_of_min_mode']:5.1f}% "
              f"of the {b['min_mode_W']}W mode")
    print(f"\nwrote {OUT/'E2_raw.json'}")


if __name__ == "__main__":
    main()
