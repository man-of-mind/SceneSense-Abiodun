# E1 — Local resource profile (FULL vs FRONT vs BACK)

**Date:** 2026-07-27 · **Raw:** `E1_raw.csv`, `E1_raw.json`, `E1_run.log` · **Script:** `../e1_local_resource_profile.py`

## Setup

| Item | Value |
|---|---|
| Model | `MultiTaskFusionLRASPP` (MobileNetV3-Large backbone, early-concat RGB+radar), no-AE baseline |
| Checkpoint | `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt` (18.9 MB on disk) |
| Input | **real** fused tensor `1×7×432×768` (3 RGB + 4 radar) from test-split sample `moving_ego_pps200000_low_8loops_stride2_000002_frame82048` (18 161 radar points, 19.8 % of radar raster non-zero) |
| GPU | RTX 5090, torch 2.12.0+cu130 |
| CPU | Intel Core Ultra 9 285K, 24 cores |
| Split | FRONT = `model.backbone` (car) · BACK = `classifier` + `object_head` incl. low/high concat + final interpolate (edge) |

> **Note on the plan's stated input shape.** The plan lists RGB `1×3×432×768`. The deployed model is
> `fusion.mode: early_concat`, so the true input is **7-channel** (RGB + occupancy / inverse-range /
> radial-velocity / stationary-age). All numbers below use the real 7-channel tensor.

### Validation (done before any profiling — guardrail 5)
- Full-model output non-NaN; `out` `1×3×54×96`, `object` `1×14×432×768`.
- **FRONT+BACK reproduces FULL exactly: `max|diff| = 0.00e+00`** on both output heads.
- FLOPs are exactly additive: `2.445 + 7.719 = 10.164 GMACs`.
- Latency additivity: front p50 + back p50 = 1.820 ms vs full p50 1.852 ms (1.7 % gap).

## Main table

| config | params (M) | fp32 (MB) | int8 (MB) | GMACs | GPU p50/p95 (ms) | GPU FPS | GPU act. mem (MB) | CPU-8T p50 (ms) | CPU-8T FPS |
|---|---|---|---|---|---|---|---|---|---|
| **FULL**  | 4.669 | 18.67 | 4.67 | **10.164** | 1.852 / 1.921 | 540 | 73.6 | 32.82 | 30.5 |
| **FRONT** (car) | 2.973 | 11.89 | 2.97 | **2.445** | 1.302 / 1.366 | 768 | 47.8 | 14.65 | 68.3 |
| **BACK** (edge) | 1.696 | 6.78 | 1.70 | **7.719** | 0.518 / 0.621 | 1931 | 67.8 | 18.11 | 55.2 |

Transmitted feature payload (FRONT output): `low 1×40×54×96` + `high 1×960×27×48` = 1 451 520 elements
= **5.81 MB fp32 / 1.45 MB uint8** (before entropy coding).

## Finding 1 — the FLOP split is the *opposite* of what the plan assumed

Guardrail 2 warned that "the backbone is usually the bulk of compute, so the local compute saved by split
may be modest." **That is not true for this model.**

| | GMACs | share |
|---|---|---|
| FRONT (backbone, stays on car) | 2.445 | **24.1 %** |
| BACK (heads, offloaded to edge) | 7.719 | **75.9 %** |

The reason is architectural: `object_heads.fuse_low_feature: true` feeds a **1000-channel** (40 low + 960 high)
dense head at 1/8 stride (54×96), and MobileNetV3's backbone is depthwise-separable (very FLOP-cheap).
So by arithmetic, **the split offloads about three quarters of the work** — a stronger result for the split
than the plan anticipated. No inflation needed here; if anything the plan was too pessimistic.

**Independent hand-check of this claim** (because it contradicts the plan's assumption, it should not rest on
fvcore alone). Counting MACs directly from the layer shapes:

| layer | shape | GMACs |
|---|---|---|
| `object_head.0` | Conv2d(1000→128, 3×3) @ 54×96 | **5.972** |
| `object_head.3` | Conv2d(128→128, 3×3) @ 54×96 | 0.764 |
| `object_head.6` | Conv2d(128→128, 3×3) @ 54×96 | 0.764 |
| `object_head.9` | Conv2d(128→14, 1×1) @ 54×96 | 0.010 |
| `classifier` (LRASPPHead, all branches) | — | 0.162 |
| **manual BACK total** | | **7.672** |
| fvcore BACK | | 7.719 |

Agreement to **0.6 %**. A single layer — the `1000→128` 3×3 convolution at 1/8 stride — is **5.97 GMACs, 59 %
of the entire model's compute**. That one layer is what the split offloads, and it exists because the object
head fuses the low-stride feature. The finding is architectural and specific to this model, not a generic
claim about split inference.

## Finding 2 — but wall-clock share depends heavily on the hardware

| basis | FRONT share | BACK share |
|---|---|---|
| FLOPs | 24.1 % | 75.9 % |
| **GPU** wall-clock | **70.3 %** | 28.0 % |
| **CPU** (8 threads) wall-clock | 44.6 % | 55.2 % |

On the RTX 5090 the ordering *reverses*: the backbone dominates wall-clock despite being 24 % of the FLOPs,
because depthwise-separable convolutions are memory-bandwidth-bound and use the GPU poorly, while the dense
head is exactly what a GPU is fast at. On a CPU — much closer to a SWaP-C vehicle SoC — the wall-clock split
tracks the FLOP split far better.

**Honest reading:** how much compute the split actually offloads is hardware-dependent, ranging from ~28 %
(big discrete GPU, wall-clock) to ~76 % (FLOPs / CPU-like device). Quote the basis whenever quoting the number.

## Finding 3 — the SWaP-C headline: CPU-only real-time (the cleanest proxy we have)

10 FPS deployment target. p50 latency, real input:

| threads | FULL p50 (ms) | FULL FPS | ≥10 FPS? | FRONT p50 (ms) | FRONT FPS | ≥10 FPS? | split speedup |
|---|---|---|---|---|---|---|---|
| 24 | 23.50 | 42.6 | ✅ | 12.71 | 78.7 | ✅ | 1.85× |
| 8 | 32.82 | 30.5 | ✅ | 14.65 | 68.3 | ✅ | 2.24× |
| 4 | 54.38 | 18.4 | ✅ | 21.81 | 45.9 | ✅ | 2.49× |
| 2 | 97.69 | 10.2 | ⚠️ *just barely* | 37.53 | 26.7 | ✅ | 2.60× |
| **1** | **178.64** | **5.6** | ❌ **fails** | **64.75** | **15.4** | ✅ | **2.76×** |

> **Superseded by E6.** These are *isolated single-shot* p50 latencies. E6 measures *sustained* throughput on a
> pinned core budget, which is the deployment-relevant figure and is consistently slightly lower (e.g. 2 cores
> full-local: 10.24 FPS isolated vs **9.96 FPS sustained** — the difference between nominally passing and
> actually missing the 10 FPS bar). **Quote E6's numbers**; this table stands as the latency decomposition.

**This is the E1 result that carries the motivation.** Running the whole model locally on a 1-core budget
misses real time (5.6 FPS vs 10 required); running only the split front clears it (15.4 FPS). At 2 cores
full-local hits 10.24 FPS here — nominally passing, but with only **2 % headroom**, and E6's sustained
measurement puts the same configuration at **9.96 FPS, i.e. actually missing the bar**. Either way there is no
margin for the rest of the autonomy stack, thermal throttling, or frame-rate jitter, while split-front has
2.7× headroom.

Split reduces on-car latency by **1.85–2.76×**, and the advantage *grows as the core budget shrinks* —
exactly the direction that matters for SWaP-C.

**Caveat (important, do not drop):** these are threads of a modern high-clock desktop CPU (Core Ultra 9 285K,
up to 5.1 GHz). One such thread is considerably stronger than one Cortex-A78AE core on a Jetson Orin, so the
1-thread row is an **optimistic** stand-in for an embedded core — the real embedded gap is wider, not narrower.
This is a *relative* comparison (guardrail 4); it is not a claim about absolute Jetson throughput.

## Finding 4 — GPU energy per frame

Idle baseline (CUDA context resident, GPU clocked down, 20 s settle): **29.6 W**. Idle-subtracted:

| config | active power (W) | E/frame via P×latency (J) | E/frame via P÷rate (J) |
|---|---|---|---|
| FULL | 254.4 | 0.4710 | 0.4116 |
| FRONT | 217.9 | 0.2838 | 0.2954 |
| BACK | 279.5 | 0.1448 | 0.1521 |

Two independent derivations agree within 9 %. Taking the mean of the two per config, front+back (0.4380 J)
≈ full (0.4413 J) to **0.75 %**. Split cuts on-car GPU energy per frame by **39.8 %** (P×latency) /
**28.2 %** (P÷rate) → **34.4 % consolidated**; E2 uses the consolidated figure.

> **Do not compare the raw wattage column across configs.** In a max-rate loop a *faster* config issues more
> forwards/second and therefore draws more power — BACK draws the most (309 W) purely because it runs at
> 1838 fwd/s vs FULL's 618 fwd/s. It is not the most expensive config; per frame it is the cheapest.
> Energy/frame is the only honest cross-config comparison.

## Also measured

Fixed **10 FPS duty cycle** (deployment condition): FULL 74.7 W, FRONT 71.9 W (idle 29.6 W). At 10 FPS the GPU
is ~2 % utilised, so this figure is dominated by the discrete GPU idling at elevated clocks rather than by
inference — it is not a meaningful basis for embedded extrapolation. Recorded in `E1_raw.json`
(`telemetry_at_10fps`) for completeness; E2 uses energy/frame instead.

## Notes
- Installed into the venv: `fvcore`, `nvidia-ml-py`, `psutil`.
- fvcore does not count these elementwise ops: `hardswish_`(20), `add_`(10), `hardsigmoid`(8), `mul`(9),
  `sigmoid`(1), `add`(1). They are cheap relative to the conv MACs but mean GMACs is a slight **under**-count,
  affecting FULL and FRONT (which contain the MobileNet activations) more than BACK.
- GPU memory is reported from the torch allocator per config (`max_memory_allocated`); `nvidia-smi`/NVML
  device-level memory is useless here because it reports the whole device (2506 MB) unchanged across configs.
