# Why split inference? — motivation study (pick-up plan for a fresh session)

**Goal.** Produce experimental evidence that justifies the split-inference + cooperative-feature-fusion design,
for the paper's motivation section. Requested by the advisor (2026-07-22): quantify the *local* (full-model-on-car)
processing requirement, uplink data rate, and per-hop end-to-end latency, and infer power consumption — to show that
running the whole model locally on a SWaP-C (Size/Weight/Power/Cost) constrained vehicle is more demanding than
splitting, at the cost of higher end-to-end latency.

Run experiments **E1–E5** below and write results into `results/`. A later review pass will check correctness, so
**log commands, keep raw outputs, and report the true numbers — do NOT tune the story.** See Guardrails.

---

## 0. Framing — it is a 3-way comparison, not "split vs local"

"Why split" only holds up against the alternatives. Build every result into this table:

| Architecture | Car compute/power | Uplink rate | E2E latency | Accuracy | Privacy |
|---|---|---|---|---|---|
| **A. Full-local + share detections** (whole model on car; "late fusion") | highest | lowest (results only) | lowest | lower (late fusion) | high |
| **B. Full-offload raw** (send RGB+radar to edge; whole model on edge) | lowest | medium (compressed image) | high | high | worst (raw pixels leave car) |
| **C. Split + feature fusion** (OURS: backbone on car → features → edge heads+fusion) | medium | highest (~1 MB features) | high | best (intermediate fusion) | medium (features, not pixels) |

**Split wins a *different* axis against each baseline** — that is the whole argument:
- vs A: the car cannot run the full model in real time within its power/thermal budget → offload (E1, E2).
- vs A (late fusion): intermediate feature fusion beats sharing detections on accuracy/occlusion (E4).
- vs B: raw pixels never leave the car (E5).
Latency (E3) is the acknowledged **cost** of split, not a benefit.

---

## 0b. ARCHITECTURE FINDING + MOTIVATION REASSESSMENT (2026-07-27 — read this)

**We do DETECTION-level (late) cooperative fusion, NOT feature fusion.** Verified in code: `cooperative_fusion/fusion.py`
fuses `ViewDetection` objects (world pos / bearing / dims / score) via `fuse_triangulate`/`fuse_mean`; the spatial map
(`fusion_object_spatial_map.v1`) aggregates each ego's **object-head detections**. The split (front/back) is a
**single-ego compute cut** — the wire features are one car's backbone output going to *that car's* heads on the edge —
orthogonal to how egos combine. **The "intermediate > late feature fusion" literature does NOT describe us; do not cite
it as our motivation.**

**Consequence:** for detection-level fusion, split is **dominated by full-local + share-detections** on uplink
(~KB « ~1 MB), latency (no round-trip), and privacy (detections leak less than features); it ties on coverage
(same map). Split's **only** advantage is **on-vehicle compute** (car runs 24% of FLOPs). So the motivation must be
built on the compute/SWaP-C axis (E1/E2/**E6**), honestly scoped — NOT on fusion accuracy.

**Why not pivot to feature fusion (which would motivate feature-sharing):** two of our own findings warn against it —
the detection head is an architectural dead-end (F1 ~0.35), and our one working cooperative result is **detection-level
triangulation (1.40 m)**. Feeding better features into a broken head is unlikely to help. Feature fusion is also a
spatial-map redesign (feature-BEV vs detection aggregator). Treat a pivot as a project-direction decision for the
advisor, not a quick experiment.

**Primary motivation going forward = E6 (compute-constrained crossover).** See E6.

---

## 🚦 Guardrails (read before running — these decide credibility)

1. **Uplink does NOT favor split.** Full-local ships only results (~KB); split ships ~1 MB features; full-offload
   ships a ~100–200 KB image. So do NOT claim "split reduces bandwidth" — it does not. Report the true rates; split's
   case is compute/power + fusion accuracy + privacy, not bandwidth.
2. **Our split keeps the backbone ON the car** (front = `model.backbone`; back = heads). The backbone is usually the
   bulk of compute, so the *local compute saved by split may be modest*. **Report the real front-vs-back split; if the
   saving is small, say so** — then the motivation leans on E4 (cooperative accuracy) and E5 (privacy). Do not inflate.
3. **Model is small (19 MB).** So the SWaP-C argument is *real-time throughput within a power/thermal budget*, NOT
   "the model doesn't fit." Frame it that way. The CPU-only real-time check (E1) is the cleanest SWaP-C proxy we have.
4. **We measure on a server GPU (RTX 5090), not the vehicle SoC.** Report **relative** numbers (full vs front, GPU vs
   CPU, energy/frame) and extrapolate to a *cited* embedded budget (e.g. Jetson Orin NX/AGX TDP). Never present server
   wattage as the vehicle number.
5. Validate inputs are real/representative (right resolution, real radar raster), and sanity-check one forward pass
   (non-NaN outputs) BEFORE trusting a profiling loop.

---

## Environment + entry points

```bash
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
```

- **Full model + config:** `pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml`;
  checkpoint (no-AE full model) `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`
  (19 MB). Load exactly as `pole_lraspp_multimodal_fusion/evaluate_fusion.py` does (reuse its model-build code).
- **Split point:** `pole_lraspp_multimodal_fusion/split_runtime.py` → `MultimodalLRASPPSplitModel`:
  `features = model.backbone(input)` is the **FRONT (car)**; the heads/decode on `features` are the **BACK (edge)**;
  `serialize/deserialize_backbone_features` is the transport payload. Full (monolithic) = backbone + heads in one pass.
- **Model input:** RGB `1×3×432×768` (model-input 768×432) + radar raster (200k PPS recipe: HFOV 120, radius 4,
  2-frame temporal). Get real inputs from the eval dataloader on
  `fusion_training_data/moving_ego_pps200000_merged_8loops_stride2` (`--split test`) rather than random tensors.
- **Profiling tools are NOT in the venv.** Install into the venv (preferred):
  `"$PY" -m pip install fvcore nvidia-ml-py psutil` (fvcore for FLOPs, nvidia-ml-py→`pynvml` for GPU power/util,
  psutil for CPU). If pip is unavailable, fall back to: FLOPs via a manual `torch` module-hook MAC counter;
  GPU power/util/mem via `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader -lms 100`;
  CPU via `/proc/stat` + `/proc/self/status`.
- **Existing data to reuse (don't re-measure):** per-hop OAI latency in
  `abiodun/oai_layer_latency/` and RTT/latency in `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`
  (front ~25–50 ms, back ~7–11 ms GPU; OAI RTT ~163 ms zstd); cooperative results in
  `abiodun/spatial_map_coop/` and memory `coop_fusion_findings` (two-view triangulation ~1.40 m).

---

## E1 — Local resource profile (advisor's core ask)

**Objective.** Quantify the cost of running the full model locally, and how much split offloads.

**Method.** With one real input batch, profile three configurations — **FULL** (backbone+heads), **FRONT** (backbone
only), **BACK** (heads on precomputed features) — on **GPU**, and **FULL on CPU** (SWaP-C proxy):
- **Params + size:** `sum(p.numel())` per config; ×4 B = fp32 size; ×1 B ≈ int8. (Full weights = 19 MB on disk.)
- **FLOPs/MACs:** `fvcore.nn.FlopCountAnalysis(module, inputs)` per config.
- **Latency:** 20-iter warmup, 100-iter timed, `torch.cuda.synchronize()` around each; report mean/p50/p95. Also
  FULL on CPU (`--device cpu`, fewer iters ok).
- **Throughput:** FPS = 1/latency; state whether each config clears the **10 FPS** deployment target (esp. CPU-only).
- **GPU util/mem/power:** run a sustained (~30 s) inference loop per config while sampling
  `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw -lms 100` (or `pynvml`); report avg + idle baseline.

**Artifacts.** `results/E1_local_resource_profile.md` + `results/E1_raw.csv`. Table columns:
config | params(M) | size(MB) | GMACs | GPU lat p50/p95 (ms) | GPU FPS | GPU util % | GPU mem (MB) | GPU power (W) |
CPU lat (ms) | CPU FPS.

**Validation.** Front+Back latency ≈ Full latency (sanity); the front/back FLOP split quantifies what offloads;
report the CPU-only FPS honestly (the SWaP-C headline if it is < 10 FPS).

---

## E2 — Power / energy per frame

**Objective.** Turn E1 power into an energy comparison and a SWaP-C budget statement.

**Method.**
- **Energy/frame (J)** = avg active power (W) × latency (s), where active = busy − idle GPU power, for FULL-local
  vs SPLIT-front. Report both and the delta.
- **Sustained power @ 10 FPS** = energy/frame × 10 for each; compare against a **cited embedded budget**
  (e.g. Jetson Orin NX 10–25 W configurable TDP, Orin AGX up to 60 W — cite exact source/spec). Show whether
  full-local sustained power fits the budget and whether split-front is meaningfully lower.
- If GPU power deltas are small (likely — backbone dominates and stays local), say so and note the CPU-only /
  throughput result is the stronger SWaP-C evidence.

**Artifacts.** `results/E2_power_energy.md` (table + the budget-fit statement + the exact embedded spec cited).

---

## E3 — Per-hop end-to-end latency + uplink data rate (all 3 architectures)

**Objective.** The advisor's per-hop latency ask, done for A/B/C so the trade is explicit.

**Method.** Build a per-hop breakdown for each architecture using measured compute (E1) + existing OAI per-hop data
(`oai_layer_latency/`, RTT ~163 ms zstd). Hops: car-compute → uplink (UE→gNB→UPF→edge) → edge-compute → downlink
(edge→UE) → map-insert/share. Then the **uplink data rate** per architecture:
- A full-local: result payload (~8–12 KB) × 10 FPS.
- B full-offload: compressed RGB(+radar) (~100–200 KB; measure a real JPEG/PNG of the 1280×720 frame) × 10 FPS.
- C split: feature payload (zstd ~1.05 MB no-AE, or ~130 KB with AE) × 10 FPS.

**Artifacts.** `results/E3_latency_datarate.md`: per-hop latency table (A/B/C) + uplink Mbps table. Present latency as
split's **cost**; be explicit that A has the lowest latency AND uplink.

---

## E4 — Cooperative LOCALIZATION gain (reframed 2026-07-22 — the primary perception motivation)

**⚠️ Reframed:** do NOT frame E4 around detection **recall**. The detection head is a known **architectural
dead-end** (F1 stuck ~0.35 — see memory `coop_fusion_findings`), so an "intermediate ≥ late > single" *recall* table
will not materialize. The productive cooperative result is **two-view triangulation localization (~1.40 m, beats
radar)**. Frame E4 around **localization error + map completeness/coverage**, not recall.

**Objective.** Show the cooperative gain that motivates offloading to a shared edge: fusing **2 egos'** views yields
localization/coverage a **single local ego cannot** achieve.

**Method.** Reuse `spatial_map_coop/` (two-ego loopback deployment; see its README) + the two-view triangulation path.
On a controlled 2-ego scene with occlusion:
- **Single ego** (full-local) — baseline loc error + map coverage (what one car sees).
- **Cooperative (2 egos)** — triangulation/fusion at the edge — loc error (~1.40 m target) + coverage.
Metric: **localization MAE** and **map completeness/coverage** (fraction of GT objects localized), single vs coop.

**Honest scope (important).** Triangulation is *geometric 2-view fusion* → it demonstrates the **cooperative gain
(2 views > 1)**, which justifies offloading to a shared edge. It does **NOT** by itself prove *intermediate feature
fusion > late detection fusion* — do not claim that unless a cheap side-by-side (share features vs share
detections/bearings, fuse at edge, same scene) actually shows it. The feature-split-point defense otherwise rests on
compute (E1/E2) + privacy (E5) + "edge runs heavier fusion than the car can," addressed in method, not forced here.
**Report only what the numbers show.**

**Artifacts.** `results/E4_cooperative_gain.md` + one qualitative overlay (object localized by coop that single-ego
misses/mislocalizes). Confirm E4 framing with the review pass after E1–E3 numbers are in.

---

## E5 — Privacy: feature-inversion resistance (most involved; can be scoped/deferred)

**Objective.** Show transmitted features do not expose the raw image (vs full-offload which ships the image itself).

**Method.** Train a small decoder to reconstruct the RGB from the transmitted (quantized u8 / AE-bottleneck) features;
report **PSNR/SSIM** of reconstruction. Low fidelity → privacy preserved. Contrast: full-offload's "reconstruction" is
the image itself (perfect → no privacy). Also note quantization/AE further degrade invertibility.

**Artifacts.** `results/E5_privacy_inversion.md` + sample (original vs reconstructed).

**Time-box + fallback (blessed 2026-07-22).** Training the inversion decoder from scratch is the most involved piece —
time-box it. If it does not converge cheaply, the **fallback is sufficient for a first pass**: report that transmitted
features are non-human-viewable, that per-channel-u8 quantization + the AE bottleneck destroy invertibility, and cite
the feature-inversion literature; contrast with full-offload (option B) whose "reconstruction" is the raw image itself
(perfect → zero privacy). A fully-trained inversion attack that then *fails* is the strong version, not a requirement.

---

## E6 — Compute-constrained crossover: full-local vs split-front (PRIMARY split motivation, added 2026-07-27)

**Objective.** Show the on-vehicle compute regime where **full-local cannot meet the real-time deadline but split-front
can** — the one axis where split genuinely beats full-local. Car FLOPs: full-local 10.16 GMACs vs split-front 2.45
(4.15× less; heads are 76% and move to the edge).

**Method.** Sweep an on-vehicle compute budget and measure **sustained FPS** for **FULL-local (front+back)** vs
**SPLIT-front (backbone only)**:
- CPU: intra-op threads 1, 2, 4, 8, 16 (`torch.set_num_threads`), pinned; report p50 latency → FPS each.
- (Optional) GPU: cap clocks with `nvidia-smi -lgc <mhz>` at a few points to emulate a weaker GPU.
Plot FPS vs compute budget for both configs with horizontal real-time lines at **10 / 20 / 30 FPS**. Report the
**crossover budget** where full-local falls below each deadline while split-front stays above.

**Honest framing (state explicitly).** Total system compute is unchanged — it is **relocated to the edge**; the claim
is about the *on-vehicle* budget only. Our model is light (10 GMACs), so full-local already clears 10 FPS on ≥8 CPU
threads — so present this as: (a) the **crossover exists** (e.g. at ≤N threads / ≥30 FPS full-local misses, split
meets), (b) cite where real automotive SoCs sit, (c) note heavier/multi-sensor/higher-res models shift the crossover
so split is needed even on capable hardware. Do NOT claim "the car can't run our model" — it can, on decent hardware.

**Artifacts.** `results/E6_compute_crossover.md` + `results/E6_raw.csv` + FPS-vs-budget plot. This supersedes E4 as the
primary split-vs-local motivation.

## Suggested order & output

1. **E1** (pure measurement, model already here) → 2. **E2** (derives from E1) → 3. **E3** (E1 + existing OAI data)
→ 4. **E4** (reuse coop pipeline) → 5. **E5** (most involved). E1–E3 are the advisor's direct request; do them first.

Write a top-level `results/RESULTS.md` summarizing the 3-way table with real numbers + the one-paragraph motivation
conclusion. Keep raw CSVs/logs under `results/`. Note any tool installs done, and flag anything that did not run.

## Review rubric (what the review pass will check)
- Inputs real & correct shape; one sane forward pass verified before profiling.
- Front/back/full split measured correctly (front+back ≈ full); numbers self-consistent.
- Power is idle-subtracted; embedded extrapolation cites a real spec, not invented.
- Uplink rates honest (split is highest); latency framed as split's cost.
- Conclusions match the numbers — no inflation of the compute-saving if it is modest.
