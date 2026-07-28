# E6 — Compute-constrained crossover: full-local vs split-front

**Date:** 2026-07-27 · **Raw:** `E6_raw.csv`, `E6_raw.json`, `E6_run.log` · **Figure:** `E6_compute_crossover.png`
**Scripts:** `../e6_compute_crossover.py`, `../e6_plot.py`, `../run_e6.sh`

Per PLAN.md this **supersedes E4 as the primary split-vs-local motivation**.

## Method

Sweep an on-vehicle compute budget and measure **sustained** throughput (a 20 s continuous window per point —
not the isolated single-shot latency E1 measured) for:
- **FULL-local** = backbone + heads on the car (10.164 GMACs)
- **SPLIT-front** = backbone only, heads offloaded (2.445 GMACs, 4.15× less)

The budget is set by **pinning the process with `sched_setaffinity`** to the first N logical CPUs *and* setting
`torch.set_num_threads(N)`, so the core budget is genuinely restricted rather than only limiting torch's pool.
Validation before measuring: `front + back` reproduces `full` exactly (`max|diff| = 0.00e+00`).

## Results

| threads | FULL-local FPS | p50 ms | SPLIT-front FPS | p50 ms | speedup | FULL ≥10 | ≥20 | ≥30 | FRONT ≥10 | ≥20 | ≥30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **5.49** | 182.02 | **15.68** | 63.46 | 2.86× | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| 2 | **9.96** | 100.27 | **25.91** | 38.60 | 2.60× | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| 4 | 17.79 | 56.17 | **44.32** | 22.52 | 2.49× | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| 8 | 29.49 | 33.78 | **65.19** | 15.31 | 2.21× | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| 16 | 29.89 | 33.40 | **58.58** | 17.04 | 1.96× | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

Every row is internally consistent (FPS matches 1/p50 to <1 %).

## The crossover

| deadline | full-local needs | split-front needs | **split-only band** |
|---|---|---|---|
| 10 FPS | ≥4 threads | ≥1 thread | **1–2 threads** |
| 20 FPS | ≥8 threads | ≥2 threads | **2–4 threads** |
| 30 FPS | **never reached** (max 29.89) | ≥4 threads | **4, 8, 16 threads** |

**Three concrete crossovers, all measured:**
- At a **1-core** budget, full-local runs 5.49 FPS and misses 10 FPS; split-front runs 15.68 FPS and clears it.
- At a **2-core** budget, full-local reaches 9.96 FPS — it misses even the 10 FPS bar, by 0.4 %. Split-front
  delivers 25.91 FPS and clears 20 FPS.
- **At 30 FPS the gap is categorical: full-local never reaches it on any budget tested**, topping out at
  29.89 FPS with all 16 cores, while split-front clears 30 FPS from 4 cores upward.

Split-front is **1.96–2.86× faster**, and the multiplier **grows as the budget shrinks** — the direction that
matters for SWaP-C. Note it exceeds the 1.6× that raw FLOPs would predict (10.164/2.445 = 4.15× is the FLOP
ratio, but the backbone is memory-bound so the realised speedup is lower than 4.15× and higher than 1.6×).

## Honest framing (required — state this in the paper)

1. **Total system compute is unchanged; it is relocated to the edge.** The 76 % of FLOPs that leave the car do
   not disappear — the edge runs them. Every claim here is about the **on-vehicle** budget only.
2. **Do not claim "the car cannot run our model."** It can: full-local clears 10 FPS from 4 cores and 20 FPS
   from 8. The defensible claim is that **the crossover exists and is reachable** — at ≤2 cores, or at a 30 FPS
   deadline, full-local fails where split succeeds.
3. **Our model is light (10.16 GMACs).** Heavier backbones, higher input resolution, more sensors, or multiple
   concurrent perception tasks all shift the crossover toward more capable hardware — so split becomes necessary
   on hardware that comfortably runs *this* model. That extrapolation is an argument, not a measurement here.
4. **This host is a high-clock desktop CPU** (Core Ultra 9 285K, P-cores at 5.0–5.1 GHz). An automotive SoC core
   (e.g. Cortex-A78AE at ~2 GHz) is substantially weaker, so **the crossover thread counts here are optimistic** —
   a real vehicle hits these walls with more nominal cores than this table suggests.

## Two measurement artifacts worth recording

**The 24-thread point is excluded.** It reported 4.34 FPS against a p50 of 30.24 ms — a 7.6× contradiction. The
cause is visible in the tail: **p95 = 1945 ms vs p50 = 30 ms**. Pinning 24 torch threads to all 24 cores leaves
no core for the OS scheduler or the timing thread, producing multi-second stalls. This is a scheduling pathology,
not a compute result, so it is reported and dropped rather than plotted. (Incidental deployment note: never pin an
inference worker to 100 % of available cores.)

**8 → 16 threads gains nothing** (29.49 → 29.89 FPS for full-local; split-front actually *drops*, 65.19 → 58.58).
This is the P-core/E-core boundary, not a scaling limit of the model: logical CPUs 0–7 are P-cores at 5.0–5.1 GHz
and 8–23 are E-cores at 4.6 GHz, so points past 8 add slower, less capable cores. Treat the x-axis as "cores of
decreasing quality" past 8, not as a linear compute budget.

## Concurrency requirement (learned the hard way)

This benchmark is invalid if anything else is using the CPU. An early run was discarded because two E6 instances
ran concurrently — both pin to the *same* first-N cores, so they measured contention, and the overlap corrupted
exactly the 1- and 2-thread points where the crossover lives. `run_e6.sh` now refuses to run as root, refuses if
another E6 is already running, waits for the 1-min load average to fall to ≤2.0, and logs it. The accepted run
started at **load 0.89**.

## GPU arm — run 2026-07-27 with sudo · **Figure:** `E6_gpu_arm.png` · **Raw:** `E6_gpu_raw.csv`

RTX 5090 with graphics clocks locked via `nvidia-smi -lgc`, 15 s sustained per config per point. The achieved
clock is recorded per row (requesting 3090 yields 2872 under power/thermal limits).

| locked | actual MHz | power W | FULL FPS | SPLIT-front FPS | speedup | FULL ≥30 FPS? |
|---|---|---|---|---|---|---|
| 3090 | 2872 | 315.0 | 539.06 | 757.58 | 1.405× | ✅ |
| 2100 | 2085 | 180.7 | 498.32 | 760.15 | 1.525× | ✅ |
| 1400 | 1395 | 126.0 | 359.69 | 592.96 | 1.649× | ✅ |
| 900 | 892 | 93.4 | 236.38 | 372.46 | 1.576× | ✅ |
| 600 | 592 | 71.9 | 159.98 | 264.39 | 1.653× | ✅ |
| 405 | 397 | 64.4 | 113.40 | 185.59 | 1.637× | ✅ |
| 210 | 210 | 50.7 | **54.82** | 91.66 | 1.672× | ✅ |

**Cross-validation:** at maximum clock this reproduces E1's independently-measured GPU numbers — FULL
1.842 ms / 539.06 FPS here vs E1's 1.852 ms / 540 FPS, FRONT 1.31 vs 1.302 ms, BACK p50 identical at 0.518 ms.
Different script, different measurement method (sustained vs isolated), same answer.

### Finding A — no crossover is reachable by clock-limiting (honest negative result)

**Full-local clears 30 FPS at every clock tested, including the 210 MHz minimum (54.82 FPS).** A 13.7× clock
reduction never forces a deadline miss. Clock-limiting reduces one dimension of compute but leaves SM count and
memory bandwidth untouched, so a throttled RTX 5090 is still far more capable than an embedded GPU — the
emulation does not reach embedded territory.

Note the reduction is **sublinear**: 13.7× less clock gives only 9.8× less throughput, precisely because
`-lgc` caps the SM clock and not memory bandwidth. Power falls 315 W → 51 W (6.2×).

**Consequence: the crossover claim rests on the CPU arm alone.** The GPU arm cannot support it, and this
write-up does not pretend otherwise.

### Finding B — but the mechanism is confirmed, confound-free

Split speedup rises monotonically (with minor noise) as the budget shrinks: **1.405× at 2872 MHz → 1.672× at
210 MHz**. This matters because the GPU arm varies compute budget **without changing core type**, so unlike the
CPU sweep it carries no P-core/E-core confound. Two independent budget axes agree that the split's advantage
grows as the device weakens.

The mechanism is the one E1 identified: at high clock the depthwise-separable backbone is memory-bandwidth-bound
and the dense head is compute-bound, so the head is disproportionately cheap and the split saves little (1.41×).
Lowering the SM clock while leaving bandwidth intact penalises the compute-bound head more than the
memory-bound backbone, so offloading the head buys more (1.67×).

The GPU speedup plateaus near 1.67× — well below the CPU arm's 2.86× and the 4.15× FLOP ratio — because even
throttled, the 5090 executes the head's dense convolutions very efficiently. **The split's benefit is larger on
CPU-like hardware than on GPU-like hardware at equal budget fraction**, which is the useful design lesson for a
vehicle SoC whose NPU/GPU may be much weaker relative to its CPU than this desktop pairing.

### Operational note
Clocks were reset cleanly on exit by the script's trap — verified afterwards:
`applications_clocks_setting: Not Active`, max clock back to 3090 MHz, persistence mode disabled.
