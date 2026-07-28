# E2 — Power / energy per frame + SWaP-C budget statement

**Date:** 2026-07-27 · **Raw:** `E2_raw.json`, `E2_run.log` · **Script:** `../e2_power_energy.py`

Every number here is derived deterministically from `E1_raw.json` (no separate measurement run), so E1 and E2
cannot drift out of sync. Re-run `e2_power_energy.py` to regenerate.

## 1. GPU energy per frame (idle-subtracted, idle = 29.6 W)

| config | active power (W) | E/frame via P×lat (J) | E/frame via P÷rate (J) | **E/frame mean (J)** | GPU compute W @10 FPS |
|---|---|---|---|---|---|
| FULL (full-local) | 254.4 | 0.4710 | 0.4116 | **0.4413** | 4.41 |
| FRONT (split, on car) | 217.9 | 0.2838 | 0.2954 | **0.2896** | 2.90 |
| BACK (split, on edge) | 279.5 | 0.1448 | 0.1521 | **0.1484** | 1.48 |

**Additivity check:** front + back = 0.4380 J vs full 0.4413 J → **0.75 % difference**. Two independent
derivations of the same quantity, agreeing this closely, is the main reason to trust these numbers.

> **Split cuts on-car GPU energy per frame by 34.4 %** (39.8 % by the P×latency derivation, 28.2 % by P÷rate;
> 34.4 % is the consolidated mean-based figure).

## 2. CPU work per frame — the embedded-relevant energy proxy

CPU package energy could **not** be measured directly on this host: RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`)
is root-only, and perf's `power` PMU is unavailable (`perf_event_paranoid=4`). Rather than escalate privileges,
E2 uses **core-milliseconds per frame** (threads × p50 latency), which is proportional to CPU energy at a fixed
per-core operating point.

| threads | FULL (core-ms/frame) | FRONT | BACK |
|---|---|---|---|
| 24 | 571.74 | 309.14 | 262.81 |
| 8 | 262.69 | 117.18 | 145.07 |
| 4 | 217.71 | 87.25 | 129.25 |
| 2 | 195.45 | 75.17 | 119.70 |
| **1** | **178.69** | **64.80** | 113.79 |

The **1-thread row is the meaningful one**: with no parallel overhead, core-ms equals the total CPU work per
frame. (The rise going up the table is thread-scaling overhead, not extra useful work.)

> **Split cuts on-car CPU work per frame by 2.76× (63.7 %).**

This is the more transferable of the two energy results, because a vehicle SoC's CPU is far closer to this
host's CPU than to an RTX 5090.

## 3. Budget fit against a cited embedded spec

GPU compute power required to sustain the 10 FPS target: **full-local 4.41 W vs split-front 2.90 W, Δ = 1.52 W.**

| module | configurable power modes (W) | Δ as % of the lowest mode |
|---|---|---|
| Jetson Orin Nano 8GB | 7, 15 | 21.7 % |
| Jetson Orin NX 8GB | 10, 15, 20 | 15.2 % |
| Jetson Orin NX 16GB | 10, 15, 25 | 15.2 % |
| Jetson AGX Orin 32GB | 15, 30, 40 | 10.1 % |
| Jetson AGX Orin 64GB | 15, 30, 50, 60 | 10.1 % |

**Source caveat:** these TDP figures are quoted NVIDIA Jetson Orin series module power modes, not measured here.
**Verify against the current NVIDIA Jetson Orin Series Modules Data Sheet revision before publication.**

## 4. Honest conclusion — what this experiment does and does not support

The plan (guardrail 4, and E2's own method note) anticipated that the GPU power delta might be small. **It is.**
Stating the position plainly:

**What E2 supports:**
- Split removes a **real and consistently measured** fraction of on-car compute energy: **34 % of GPU energy/frame**
  and **64 % of CPU work/frame**. The additivity check (0.75 %) says this is a measurement, not an artifact.
- Freeing 1.5 W of compute headroom is **10–22 % of the entire power budget** of a low-mode Orin module — on a
  SWaP-C platform where the perception stack shares that budget with planning, control, logging and the rest of
  the sensor pipeline, that is not a rounding error.

**What E2 does NOT support — do not claim these:**
- **Absolute watts do not transfer.** 4.41 W is what an RTX 5090 needs for this model, and the 5090 has far better
  perf/W on dense convolutions than an Orin. Read naively, "full-local needs only 4.41 W, so it fits in a 10 W
  Orin NX" — that inference is **invalid**, because Orin would need substantially more than 4.41 W for the same
  work. E2 licenses the *ratio* (full : front ≈ 1.52 : 1), not the absolute.
- **Energy alone is not the SWaP-C argument.** A ~1.5 W saving is helpful but not decisive on its own.

**Where the SWaP-C argument actually lands:** on **throughput, not energy** — exactly as the plan predicted.
E1 finding 3 is the load-bearing result: at a 1-core budget the full model runs at **5.6 FPS and misses the
10 FPS target**, while the split front runs at **15.4 FPS and clears it**; at 2 cores full-local scrapes 10.24 FPS
with 2 % headroom versus split-front's 2.7×. Split is the difference between missing and meeting real time on a
constrained core budget, and the energy saving is a secondary benefit on top.

Per the plan's framing note: since the compute/power saving is real but modest, the motivation leans additionally
on **E4 (cooperative accuracy)** and **E5 (privacy)**. E3 quantifies the latency and bandwidth cost.
