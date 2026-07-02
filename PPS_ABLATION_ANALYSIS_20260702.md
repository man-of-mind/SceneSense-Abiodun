# Radar-PPS Ablation — Analysis (2026-07-02)

Five RGB+radar fusion models, identical winning recipe (seg-only Lovász → detection head, bbox2d +
gated 40 m + center-4), evaluated identically (NMS-6, gated ≤40 m, score thr 0.10). 150k–300k were
collected with the **same seed + same fixed route** (only radar pps varies) → a controlled ablation.
100k is the prior-work reference (its own earlier collection — see caveat).

## Headline
**Radar pps barely affects vehicles or segmentation (already saturated), but meaningfully improves the
radar-limited class — pedestrians — up to ~200k pps, then plateaus.** So ~200k pps looks like the
accuracy sweet spot to weigh against payload/latency next.

## 5-way comparison
| radar pps | veh seg IoU | mIoU | det F1 | veh F1 | **person F1** | 0–20 m veh rec | **0–20 m person rec** | XY MAE |
|---|---|---|---|---|---|---|---|---|
| 100k (ref) | 0.910 | 0.827 | 0.818 | 0.875 | 0.718 | ~0.92 | ~0.86 (0–10 m) | 1.43 |
| 150k | 0.943 | 0.837 | 0.814 | 0.850 | 0.742 | 0.94 | **0.74** | 1.43 |
| 200k | 0.934 | 0.837 | 0.846 | 0.870 | **0.806** | 0.94 | **0.90** | 1.45 |
| 250k | 0.925 | 0.835 | 0.828 | 0.851 | 0.783 | 0.92 | 0.85 | 1.42 |
| 300k | 0.939 | 0.849 | 0.831 | 0.856 | 0.790 | 0.95 | 0.88 | 1.44 |

## Person detection recall by distance (v2 models, NMS-6, thr 0.10)
| radar pps | 0–10 m | 10–20 m | 20–30 m | 30–40 m |
|---|---|---|---|---|
| 150k | 0.76 | 0.74 | 0.82 | 0.75 |
| **200k** | **0.94** | **0.88** | 0.87 | 0.80 |
| 250k | 0.85 | 0.85 | 0.87 | 0.78 |
| 300k | 0.90 | 0.88 | 0.88 | 0.76 |

Higher pps lifts person recall at **every** range (150k ~0.74–0.82 → 200k+ ~0.80–0.94), strongest in the
near field (0–10 m: 0.76 → 0.94). It flattens/declines by 30–40 m (0.75–0.80) — far pedestrians stay hard
regardless of pps. (Vehicle recall is flat ~0.92–0.95 across all bins and all pps — saturated.)

## Findings
1. **Segmentation is flat across pps** (vehicle IoU 0.91–0.94, mIoU 0.83–0.85). Seg is RGB-dominated;
   radar density adds little. All five are strong.
2. **Vehicle detection is saturated** (F1 0.85–0.88; 0–20 m recall 0.92–0.95, flat). Vehicles are large
   and reliably produce radar returns, so more points don't help.
3. **Pedestrian detection improves with pps — the key result.** Person F1 0.72→0.81 and near-field
   (0–20 m) recall **0.74 (150k) → 0.90 (200k)**, then plateaus (~0.85–0.88 at 250–300k). This directly
   confirms the earlier diagnosis that person detection is **radar-limited** (pedestrians are small and
   radar-sparse; missed ones lacked radar returns). More points → more pedestrians get a radar hit →
   higher recall — until saturation ~200k.
4. **Localization (XY MAE) is flat** (~1.42–1.45 m) — radar pps doesn't change position accuracy of
   detected objects.
5. **Sweet spot ≈ 200k pps** for accuracy: best person F1/recall, with diminishing returns beyond. The
   150k→200k step is the real gain; 200k↔300k differences are within run-to-run noise.

## Implication for the split-inference study (next step)
Higher pps buys pedestrian recall only up to ~200k, but **payload/latency will keep growing with pps**.
So the deployment measurement (payload, front/back/RTT latency per pps) should reveal whether 200k is
the accuracy-per-cost optimum — the ablation predicts accuracy plateaus while cost rises past 200k.

## Methodology notes (for defensibility)
- **Heatmap-collapse fix:** the first detection pass had a focal-loss instability (center heatmap
  collapsed to 0 for 200k/250k off a weak object-head init → F1 0). Fixed by warm-starting all
  detection heads from a common live head (the 150k head) — a *more* controlled protocol, and all four
  150k–300k share it, so the detection comparison is internally consistent.
- All eval identical: NMS-6, gated ≤40 m, score thr 0.10, match ≤5 m, 768×432.
- Person GT = engine 3D-box → 2D box (CARLA 0.10 can't render walker masks); vehicle GT = semantic mask.
- **Caveat:** 100k is from an earlier, separate collection (different NPC seed), so 100k↔others has a
  mild data confound; the **150k–300k comparison is clean** (same seed/route). 100k still lands in-family.

## Artifacts
- Datasets: `fusion_training_data/moving_ego_pps{150000..300000}_merged_8loops_stride2`
- Seg: `experiments/seg_pps{X}/…`  ·  Detection (final): `experiments/autonomous_arch_runs_20260625/det_pps{X}_v2/…`
- Per-pps rows: `PPS_ABLATION_RESULTS_v2.md`  ·  100k reference: `det_stage2c_centerw4`
