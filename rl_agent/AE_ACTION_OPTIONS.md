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
2. **Milder bottleneck b256 (v1)** — DONE, FAILED. Even at 256ch (49% payload, milder than b128's 24%),
   v1 still collapses detection: ped recall 0.079, loc 3.69m (250-frame subset) — no better than b128, and
   at higher payload. CONCLUSION: it is the linear architecture, not the bottleneck size. Backing off
   compression is not the fix.
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

4. **AE v2 (nonlinear+spatial), properly trained (NO drop-in-loop, 40ep, lr3e-4)** — SUCCESS. Same 250-frame
   subset: ped recall 0.00→**0.575** (b128), **0.567** (b64); seg fully preserved (mIoU 0.80, veh 0.80); loc
   2.33-2.35m (vs clean 1.59m). Payload 9-12%. So the earlier collapse was the drop-in-loop + undertraining,
   NOT the architecture. **The AE is now a usable action.**

## RESOLUTION (2026-07-09): AE is viable — folding it in
The AE v2-clean is the AGGRESSIVE-compression action: smallest payload (9% at b64, < u4's 13%) at a bounded
accuracy cost (recall ~0.57 vs clean 0.84, loc ~2.3m). It is NOT lossless like quant/ROI and does not need to
be — it is a distinct operating point the RL agent selects under heavy bandwidth pressure, gated by the
pedestrian-recall floor. Fold v2-clean (b64, b128) into the loopback + matrix as the AE action.
Remaining upside (optional, not blocking): more epochs and/or Option C (co-adapt the back-half heads to AE
reconstructions) to close the recall gap toward clean — but the action is usable as-is.
