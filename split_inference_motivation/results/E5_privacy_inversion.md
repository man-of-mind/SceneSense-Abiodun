# E5 — Privacy: feature-inversion resistance across split profiles

**Date:** 2026-07-27 · **Raw:** `E5_raw_u8.json`, `E5_raw_fp32.json`, `E5_profile_*.json`, `E5_run_*.log`
**Figures:** `E5_payload_vs_privacy.png`, `E5_samples_*.png` · **Scripts:** `../e5_privacy_inversion.py`, `../e5_profile_variants.py`, `../e5_plot.py`

## Headline

**The privacy argument for split inference does not survive the experiment.** A trained inversion attack
recovers clearly recognisable scene content from the transmitted features, and **compressing the payload 22×
does not meaningfully reduce that** — with one exception (ROI drop, below).

This contradicts the outcome the plan anticipated. Reported as measured.

## Threat model (deliberately generous to the attacker)

A failed *strong* attack is the only meaningful privacy evidence, so the attacker gets:
- the exact on-wire features (post ROI-gate, post AE-encode, post per-channel uintN quantization),
- knowledge of the encoder architecture,
- a large in-distribution corpus of (feature, image) pairs from the same scenes,
- full supervised training of a dedicated 0.93 M-param decoder (identical 15-min budget per profile).

**Reference points:** the *floor* is a mean-image predictor (zero information from features); the *ceiling* is
architecture B, whose "reconstruction" is the transmitted JPEG itself.

## Results — manifest split

| profile | payload | mIoU | ped-recall | accept | attack PSNR | attack SSIM |
|---|---|---|---|---|---|---|
| *floor* — mean image | — | — | — | — | 12.69 dB | 0.326 |
| **`ae128__uint4__roi0.0`** (Pareto pick) | **129.2 KB** | 0.819 | 0.887 | ✅ | 24.86 dB | 0.725 |
| `ae32__uint6__roi0.0` | 174.7 KB | 0.822 | 0.865 | ✅ | 25.03 dB | 0.733 |
| **`ae64__uint8__roi0.3`** | 195.7 KB | 0.805 | 0.864 | ✅ | **17.01 dB** | **0.571** |
| `noae__uint8__roi0.0` (deployed) | 1050.3 KB | 0.840 | 0.855 | — | 25.20 dB | 0.736 |
| `noae` fp32 (no-quantization control) | 2835.0 KB | 0.840 | 0.855 | — | 25.46 dB | 0.742 |
| *ceiling* — architecture B ships the JPEG | 383 KB | — | — | — | 38.94 dB | 0.979 |

Accuracy figures are from `rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md`, not re-measured here.

## Finding 1 — payload size and privacy are almost unrelated

Across the four non-ROI profiles the payload spans **129.2 → 2835.0 KB (22×)** while attack SSIM moves
**0.725 → 0.742 (0.017)** and PSNR **24.86 → 25.46 dB (0.6 dB)**. Every one of them sits far above the
no-information floor (0.326).

> **Quantization bit-depth and AE bottleneck width are payload knobs, not privacy knobs.**

The mechanism is clear in hindsight: the AE bottleneck is *trained to preserve exactly what the detection and
segmentation heads need* — object and scene layout — and that is precisely what a privacy attacker wants. Squeezing
the representation harder makes it cheaper to send without making it less descriptive of the scene.

Qualitatively this is starker than the metrics. At the **Pareto pick (129.2 KB, 8× smaller than deployed, accuracy
fully intact)** the reconstruction recovers buildings, lane markings, road geometry, signage, a green tarp, and even
asphalt crack patterns (`E5_samples_ae128__uint4__roi0.0.png`).

## Finding 2 — ROI drop is the one knob that actually buys privacy

`ae64__uint8__roi0.3` breaks the pattern: **17.01 dB / SSIM 0.571**, versus ~25 dB / ~0.73 for every non-ROI
profile — while still passing the accuracy tolerance (mIoU 0.805, ped-recall 0.864).

Measured as distance closed toward the no-information floor:
- PSNR: (25.03 − 17.01) / (25.03 − 12.69) = **65 % of the gap closed**
- SSIM: (0.733 − 0.571) / (0.733 − 0.326) = **40 % of the gap closed**

The reason is structural: quantization and the AE *re-encode* information, whereas ROI drop **deletes** it —
it zeroes the lowest-objectness cells outright (verified: exactly 1555/5184 cells per sample at q=0.3). Information
that was never transmitted cannot be inverted.

> **Actionable for the RL agent:** of the three knobs in the current action space, **ROI drop is the only one that
> trades payload for privacy.** If privacy ever becomes a reward term, it is the only available lever.

**Important caveat — do not oversell this.** ROI drop is *designed* to preserve high-objectness cells, i.e. exactly
the regions containing vehicles and pedestrians. The sample images show the degradation concentrated in buildings
and sky while the road surface and lane markings stay legible. So ROI drop plausibly protects *background* better
than it protects *objects* — and objects are the privacy-sensitive content. The aggregate PSNR/SSIM gain may
therefore overstate the real privacy benefit. Testing that needs a metric computed over object crops only; not run.

## Finding 3 — the result survives a leakage-controlled split

The dataset's own train/test split is **randomly interleaved per sample**, and consecutive samples are ~0.2 s apart
on a moving ego — so test frames have near-duplicate neighbours in training, which inflates any inversion attack.
This was checked rather than assumed: only 281 consecutive test rows vs 1880 gaps, i.e. heavily interleaved.

A **temporal holdout** was therefore run: within each experiment, order by `frame_id`, train on the first 65 %, test
on the last 20 %, leaving a 15 % buffer (verified ≈1500-frame gap per experiment, zero frame overlap).

| profile | manifest split | temporal holdout | change |
|---|---|---|---|
| `noae__uint8__roi0.0` (1050.3 KB) | 0.7357 | **0.7163** | −0.019 |
| `ae128__uint4__roi0.0` (129.2 KB) | 0.7246 | **0.7030** | −0.022 |

Leakage inflated the attack by only ~1 dB / 0.02 SSIM. **Both conclusions hold under the stricter split:** features
remain highly invertible (0.703 vs a 0.326 floor), and the 8× payload reduction still costs the attacker only
0.013 SSIM.

**Residual limitation:** the dataset is 8 loops of the *same route* in Town10HD, so no split can make geography
disjoint. The temporal holdout removes adjacent-frame leakage only. A genuinely held-out route would likely lower
the absolute numbers further — but it would lower them for *all* profiles, so the relative conclusion (compression
does not buy privacy) is unaffected.

## What this means for the paper

**Do not claim a privacy benefit for split inference over full-offload on the strength of these numbers.**
Defensible statements:

1. **Split transmits less directly-usable imagery than architecture B** — B's payload *is* a viewable JPEG
   (SSIM 0.979); split requires an attacker to train an inversion decoder and yields a visibly degraded
   reconstruction (SSIM ≈ 0.70–0.74). That is a real but **modest** difference in attacker effort, not a
   privacy guarantee.
2. **Features are not "non-human-viewable."** The plan's time-boxed fallback wording should not be used —
   the reconstructions are plainly interpretable.
3. **ROI drop is a genuine privacy lever** (65 % of the PSNR gap to the floor) and is the only one measured here.
4. **Compression is a bandwidth tool, not a privacy tool.** Anyone proposing "we compress the features, so it is
   private" should be shown Finding 1.

Since E5 does not deliver the expected privacy pillar, the motivation rests on **E1/E6 (on-vehicle compute and the
real-time crossover)** and **E4 (cooperative coverage and localization)**.

## Reproducing

```bash
$PY e5_privacy_inversion.py --minutes 15 --variant u8     # deployed profile + floor/ceiling
$PY e5_privacy_inversion.py --minutes 15 --variant fp32   # no-quantization control
$PY e5_profile_variants.py --profile ae128__uint4__roi0.0 --minutes 15
$PY e5_profile_variants.py --profile ae128__uint4__roi0.0 --minutes 15 --split-mode temporal
$PY e5_plot.py
```

Note `e5_profile_variants.py` replicates the wire pipeline in `evaluate_fusion.py`'s exact order
(ROI-gate → AE-encode `high` → per-channel uintN); the ROI gate ranks **per sample**, since pooling objectness
across a batch would let one frame decide another frame's dropped cells.
