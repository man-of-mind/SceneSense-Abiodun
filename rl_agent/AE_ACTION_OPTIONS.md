# Making the feature-AE a usable RL action — problem, attempts, options (2026-07-09)

**Goal:** the feature-AE must be a real compression *action* — shrink the 960-ch split-point feature
while preserving BOTH segmentation AND object detection (recall + localization). No workaround: either
it works uncompromised, or we bring options to the supervisor.

## Problem
The AE preserves segmentation but **collapses object detection**. Same-subset (250 frames), M′:
| config | mIoU | veh IoU | ped recall | loc |
|---|---|---|---|---|
| M′ clean (no AE) | 0.802 | 0.805 | **0.843** | 1.59 m |
| AE b128 (v1, seg-weighted) | 0.796 | 0.798 | 0.094 | 2.65 m |
| AE b128 (v1, object-weighted) | 0.784 | 0.788 | 0.181 | 2.83 m |

## Root cause (not a loss-weighting issue)
The v1 AE is a single 1×1 conv each way = a **linear, per-pixel, low-rank channel projection** (960→128).
It keeps the coarse, redundant signal segmentation needs, but discards the high-dimensional feature
structure the object head regresses from. The information is gone *at encode*, so no reweighting of the
distillation loss recovers it (confirmed: object-weighting barely moved recall). Plumbing is verified
correct (payload does drop to ~24%, seg does survive) — this is a modeling limit, not a bug.

## Attempts
1. **Loss reweighting** (object-weighted, heat×8/reg×5/seg×0.3) — FAILED (recall 0.09→0.18, still unusable).
2. **Milder bottleneck b256 (v1)** — IN PROGRESS. Data point only: concedes compression to buy detail;
   not the "aggressive compression preserved" win we want.
3. **Better codec, same rate: AE v2 (nonlinear + spatial)** — IN PROGRESS (the uncompromised attempt).
   Same 128/64 bottleneck → identical payload, but 3×3 spatial conv + hidden layer + GELU (5.3M vs 263k
   params). If detection recovers here, the AE becomes a usable action at aggressive compression.

## Options for the supervisor (if v2 also fails to preserve detection)
- **A. Ship quant×ROI as the action set; AE = future work.** quant-u4 already gives 25% payload at 100%
  delivery with near-full accuracy, and ROI adds more — a complete, usable, characterized action set
  *without* the AE. The AE is upside, not a blocker for the RL controller.
- **B. AE only at the bottleneck where it's lossless-enough** (e.g. b256 if v2-b128 falls short) — usable
  but at a moderate compression rate.
- **C. Deeper redesign:** task-coupled rate allocation (use channel-importance to keep object-critical
  channels, compress the rest harder); or a detection-specific feature-matching objective at the object
  head's input; or brief joint AE+head fine-tune.
- **D. Question the premise:** does a learned AE earn its place given quant-u4's 25%/100%-delivery? The AE
  must beat that *and* preserve detection to be worth the complexity — a fair cost/benefit for the paper.

## Recommendation
Pending the v2 result. If v2 recovers detection at aggressive bottleneck → AE is in, cleanly. If not →
Option A (ship quant×ROI now, AE as characterized future work) is the honest, non-workaround path.
