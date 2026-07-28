# Why split inference? — motivation study results

**Date:** 2026-07-27 · Model: `MultiTaskFusionLRASPP` (MobileNetV3-Large + early-concat RGB+radar), no-AE baseline
unless a profile says otherwise · Hardware: RTX 5090 + Intel Core Ultra 9 285K · Input: real `1×7×432×768`
(3 RGB + 4 radar) from the test split.

| | file |
|---|---|
| E1 local resource profile | [E1_local_resource_profile.md](E1_local_resource_profile.md) |
| E2 power / energy | [E2_power_energy.md](E2_power_energy.md) |
| E3 latency + data rate | [E3_latency_datarate.md](E3_latency_datarate.md) |
| E4 cooperative gain | [E4_cooperative_gain.md](E4_cooperative_gain.md) |
| E5 privacy / feature inversion | [E5_privacy_inversion.md](E5_privacy_inversion.md) |
| **E6 compute crossover (primary split motivation)** | [E6_compute_crossover.md](E6_compute_crossover.md) |

---

## The three-way table, with real numbers

| | **A. Full-local + share detections** | **B. Full-offload raw** | **C. Split + feature fusion (OURS)** |
|---|---|---|---|
| **Car compute** (CPU 8T / GPU p50) | 32.82 ms / 1.85 ms | ~0 (JPEG encode only) | **14.65 ms / 1.30 ms** |
| **Sustained FPS @ 1 core** (E6) | **5.49 — misses 10 FPS** | n/a | **15.68 — clears it** |
| **Sustained FPS @ 16 cores** (E6) | **29.89 — never reaches 30** | n/a | **58.58 — clears 30 from 4 cores** |
| **Compute offloaded** | 0 % | 100 % | 75.9 % of FLOPs / 28 % of GPU wall-clock |
| **On-car energy** | baseline | lowest | −34 % GPU J/frame, −64 % CPU work/frame |
| **Uplink payload** | **2.27 KB** | 383 KB (JPEG q92 + radar) | 1045.5 KB no-AE / **129.2 KB** AE-128 u4 |
| **Uplink rate @10 FPS** | **0.18 Mbps (best)** | 13.9–29.9 Mbps | **81.68 Mbps (worst)** / 11.9 AE |
| **E2E to a shared result** | **~42 ms (best)** | not measured | 188.0 ms no-AE / **86.5 ms** AE-128 |
| **Delivery over OAI** | — | — | 83.6 % no-AE / **99.8 %** AE-128 |
| **Accuracy** | late fusion only | full model, single view | intermediate fusion possible |
| **Privacy (attacker SSIM)** | n/a (no imagery sent) | **0.979 — the image itself** | **0.725–0.736 — substantially invertible** |

Bold = best in row. **Architecture A wins compute-free operation, bandwidth, and latency. Split wins on-car
compute and enables cooperation. Neither privacy nor bandwidth is a win for split.**

---

## What each experiment actually showed

### E1 — the split offloads more compute than the plan assumed, but the SWaP-C case is throughput
`front + back` reproduces `full` **exactly** (`max|diff| = 0.00e+00`) and FLOPs are exactly additive
(2.445 + 7.719 = 10.164 GMACs), so the decomposition is measured, not estimated.

The plan's guardrail 2 predicted the backbone would dominate and the saving would be modest. **The opposite is
true here:** the backbone is only **24.1 %** of FLOPs. One layer — `Conv2d(1000→128, 3×3)` at 1/8 stride inside
the low-fused object head — is **5.97 GMACs, 59 % of the whole model**, and the split offloads it. Hand-verified
independently of fvcore (7.672 vs 7.719 GMACs, 0.6 %).

But wall-clock share is hardware-dependent (GPU 70 % front / CPU-8T 45 % front), so **always quote the basis.**

**The load-bearing result** (superseded in detail by E6): on a 1-core budget the full model misses the 10 FPS
target while the split front clears it, and split's advantage *grows* as the core budget shrinks.

> **Reconciling E1 and E6 — they measure different things and E6 is the one to quote.** E1 reports *isolated
> single-shot p50 latency* (20 warmup + 30 timed iterations); E6 reports *sustained throughput* over a 20 s
> continuous window with the process pinned. Sustained is consistently the lower of the two, as expected:
>
> | budget | E1 isolated (implied FPS) | E6 sustained FPS | verdict at 10 FPS |
> |---|---|---|---|
> | 1 core, full-local | 5.60 | 5.49 | both miss |
> | **2 cores, full-local** | **10.24 (passes)** | **9.96 (misses)** | **E6 is authoritative** |
> | 8 cores, full-local | 30.47 | 29.49 | both pass |
>
> At 2 cores full-local sits exactly on the 10 FPS boundary — isolated latency puts it 2 % above, sustained
> throughput 0.4 % below. **Quote E6's sustained numbers**; a deployed system runs continuously, not in
> isolated single shots.

### E2 — the energy saving is real but modest; do not lead with it
−34.4 % GPU energy/frame, −64 % CPU work/frame, additivity check 0.75 %. But the absolute delta is **1.5 W**
(10–22 % of a low-mode Jetson Orin budget), and **absolute watts do not transfer** from an RTX 5090 to an Orin.
E2 licenses the *ratio*, not the absolute. As the plan predicted, the SWaP-C argument rests on E1's throughput
result, not on energy.

### E3 — split is the worst architecture on both network axes, and that must be said plainly
Payloads measured independently here land within 1 % of the deployed references (1045.5 vs 1050.3 KB) and
reproduce the fp16 baseline exactly (2835.00 KB).

**Guardrail 1 confirmed emphatically: architecture A needs 0.18 Mbps, split needs 81.68 Mbps — ~450× more.**
Split also has the highest latency (188 ms vs A's ~42 ms). Two plan estimates were corrected: A is 2.27 KB
(not 8–12 KB); B is 383 KB once the radar the edge needs is included (not 100–200 KB).

Practical consequence: **feature compression is not optional.** At no-AE the split cannot reach 10 FPS over this
link at all (1.33 FPS sustainable, 83.6 % delivery); AE-128 makes it viable (9.1 FPS, 99.8 %, 86.5 ms).

### E4 — cooperation is the genuine perception win, on coverage more than accuracy
On real CARLA GT layouts and real ego-A poses (300 frames, 1096 objects; ego B's pose synthesized):
**map coverage rises 73.9 % → 87.3 %**, roughly halving the blind spot. No amount of on-car compute recovers
those objects — they are occluded or out of FOV, and only a second viewpoint fixes that.

Two-view triangulation improves localization **2.6–7.6×** at adequate baseline, and independently reproduces both
the prior live measurement (**2.61× here vs 2.54× measured live**) and its failure mode (triangulation is *worse*
than single-view at a 4 m baseline — ill-conditioned). Neither was tuned to match.

**E4 justifies cooperation, not the feature-level split point.** No share-features vs share-detections
side-by-side was run, so that claim is not made.

### E5 — the privacy argument does not survive contact with the experiment
A trained inversion attack recovers clearly recognisable scene content — buildings, lane markings, signage,
asphalt cracks — from the transmitted features at **SSIM ≈ 0.70–0.74**, against a no-information floor of 0.326.

**Payload size and privacy are almost unrelated:** across four accuracy-preserving profiles the payload spans
**22× (129.2 → 2835.0 KB)** while attack SSIM moves **0.017**. The AE bottleneck is trained to preserve exactly
the scene layout the heads need, which is exactly what an attacker wants. **Quantization and AE width are payload
knobs, not privacy knobs.**

**The one exception is ROI drop:** `ae64__uint8__roi0.3` gives SSIM 0.571 / 17.01 dB — **65 % of the PSNR gap to
the floor closed** — while still passing accuracy. It *deletes* information (verified: exactly 1555/5184 cells
zeroed per sample) rather than re-encoding it. For the RL agent, ROI is the only knob in the current action space
that trades payload for privacy.

Verified against a **leakage-controlled temporal holdout** (the dataset's own split is randomly interleaved and
frames are ~0.2 s apart, so test frames had near-duplicate neighbours in training): the attack loses only
~0.02 SSIM, so both conclusions hold.

### E6 — the compute crossover (primary split motivation)
Sustained throughput vs pinned-core budget. **Three measured crossovers:** at **1 core** full-local runs 5.49 FPS
and misses 10 FPS while split-front runs 15.68 and clears it; at **2 cores** full-local reaches 9.96 FPS, missing
even 10 FPS, while split-front delivers 25.91; and **at 30 FPS the gap is categorical — full-local never reaches
it on any budget tested** (max 29.89 with 16 cores) while split-front clears it from 4 cores.

Split-front is **1.96–2.86× faster, and the multiplier grows as the budget shrinks.**

Required framing: total compute is **unchanged, only relocated** — the edge runs the 76 % that leaves the car.
Do **not** claim the car cannot run the model; it can, on ≥4 cores at 10 FPS. The claim is that the crossover
exists and is reachable.

**GPU arm (run with sudo, 2872 → 210 MHz):** no crossover is reachable by clock-limiting — full-local clears
30 FPS at *every* clock, still managing 54.82 FPS at 210 MHz. A throttled RTX 5090 never reaches embedded
territory, so **the crossover claim rests on the CPU arm alone.** What the GPU arm does establish, confound-free
(clock variation changes no core types), is the *mechanism*: split speedup rises monotonically 1.405× → 1.672×
as the budget shrinks. It also cross-validates E1's GPU numbers exactly (539.06 vs 540 FPS) from a different
script and method.

---

## The honest bottom line

**Split inference is not globally better than the alternatives, and the paper should not claim it is.**
Against each baseline it wins a different axis and loses others:

- **vs A (full-local):** split roughly halves on-car compute and is the difference between missing and meeting
  real time on a constrained core budget. It **loses** on bandwidth (450×), latency (4.5×), and privacy.
- **vs B (full-offload):** split keeps a semantic feature on the wire rather than a picture — but E5 shows that
  distinction is much weaker than assumed. Split does keep meaningful work distributed rather than centralizing
  every vehicle's raw video at the edge.
- **The cooperative case (E4) is architecture-agnostic** — it motivates sharing perception with an edge/peer, not
  specifically sharing *features*.

**The defensible motivation, in one paragraph.** Running the full multimodal model locally on a SWaP-C vehicle
misses real-time deadlines that the split front meets: at a 1-core budget full-local sustains 5.49 FPS against a
10 FPS target while split-front reaches 15.68, and at a 30 FPS deadline full-local never qualifies on any budget
we tested (peak 29.89 FPS on 16 cores) while split-front clears it from 4 cores — a 1.96–2.86× throughput
advantage that *widens* as the budget shrinks. The split point matters because a single low-fused head layer
(`Conv2d(1000→128, 3×3)` at 1/8 stride) accounts for 59 % of the model's FLOPs, so moving the heads offloads
76 % of the arithmetic and cuts on-car compute energy ~34 %. Offloading is therefore justified, and once
perception leaves the vehicle, cooperation becomes available: two viewpoints raise map coverage from 73.9 % to
87.3 % — roughly halving the blind spot — and cut localization error 2.6× at realistic sensor noise, neither of
which any single-vehicle compute budget can buy. The costs are real and we report them: split has the highest
uplink demand of any architecture (81.7 Mbps uncompressed, ~450× architecture A) and the highest end-to-end
latency (188 ms), and feature compression is mandatory rather than optional to make it deployable (AE-128:
129 KB, 99.8 % delivery, 86.5 ms, no accuracy loss). We explicitly do **not** claim a privacy benefit: a trained
inversion attack recovers scene structure from the transmitted features at SSIM ≈ 0.70–0.74 against a 0.326
floor, and compressing the payload 22× does not meaningfully degrade that — only ROI drop, which deletes rather
than re-encodes information, measurably reduces invertibility.

---

## Gaps and things that did not run

| Item | Status |
|---|---|
| Architecture B over OAI | **Not measured.** No OAI run ships JPEG frames; E3 leaves its uplink latency blank rather than interpolating. |
| Intermediate vs late fusion head-to-head | **Not run.** Deliberately not claimed (E4 honest-scope note). |
| Live 2-ego CARLA capture for E4 | **Not run** — CARLA was not running; E4 used real GT layouts with a synthesized ego-B pose, anchored to the prior live measurement. |
| CPU package energy (RAPL) | **Unavailable** — root-only on this host; core-ms/frame used as the proxy instead of escalating privileges. |
| Jetson TDP figures | **Quoted product specs, not measured.** Verify against the current NVIDIA datasheet revision before publication. |
| E6 GPU clock sweep | **Done** (2026-07-27, with sudo). Confirms the mechanism but yields no crossover — see E6. |
| E5 object-crop privacy metric | **Not run.** ROI drop preserves high-objectness cells by design, so it may protect background better than objects; confirming needs PSNR/SSIM over object crops only. |
| Real cross-ego data association | Open problem (`spatial_map_coop/README.md` stage 4); E4 assumes it solved, as the prior live work did. |

## Reproducing

```bash
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
cd $AB/split_inference_motivation
$PY e1_local_resource_profile.py          # E1  (~6 min)
$PY e2_power_energy.py                    # E2  (derives from E1_raw.json)
$PY e3_payloads.py --frames 25            # E3  payloads
$PY e3_latency_datarate.py                # E3  assembly
$PY e4_cooperative_gain.py && $PY e4_plot.py                  # E4
$PY e5_privacy_inversion.py --minutes 15 --variant u8         # E5 baseline
$PY e5_profile_variants.py --profile ae128__uint4__roi0.0     # E5 per-profile
$PY e5_profile_variants.py --profile ae128__uint4__roi0.0 --split-mode temporal   # leakage control
$PY e5_plot.py
./run_e6.sh && $PY e6_plot.py             # E6 CPU sweep (as your user; NOT sudo)
./run_e6_gpu.sh                           # E6 GPU arm (calls sudo internally for nvidia-smi)
```

**E6 must run on an idle machine.** `run_e6.sh` enforces this (refuses as root, refuses if another E6 is running,
waits for 1-min load <= 2.0). Two concurrent instances pin to the same cores and measure contention, which
silently corrupts the low-thread points where the crossover lives.

Installed into the venv for this study: `fvcore`, `nvidia-ml-py`, `psutil`.

**Path corrections vs PLAN.md** (the plan's paths were slightly off): the package is nested at
`pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/`, and `experiments/` sits at the `abiodun/` level,
not inside the package. The plan also lists the input as `1×3×432×768`; the deployed model is `early_concat`,
so the real input is **7-channel**.
