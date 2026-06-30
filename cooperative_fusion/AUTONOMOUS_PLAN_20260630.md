# Autonomous plan — detection to target (2026-06-30)

GOAL: strong near-field detection for car + person, like dimw05 on old data:
~0.95 recall / 0.91 precision / 0.93 F1 at 0-10 m (and strong 10-20 m). Operating range <=40 m.
Seg already done: vehicle IoU 0.9145, mIoU 0.8335; person 2D box (seg, frozen) near 10-20 m ~0.92 rec / 0.73 IoU.

## When Stage 2 (det_stage2_newdata_bbox2d_gated40) finishes
1. Read driver eval from RESULTS.md (gated40, but NMS=2 -> SUBOPTIMAL; NMS was the precision lever).
2. RE-EVAL with the real lens: NMS-6, gated 40 m, thr sweep 0.10/0.20/0.30.
3. BY-DISTANCE eval (custom): detection P/R/F1 for vehicle AND person at 0-10/10-20/20-30/30-40 m
   (note sample counts — near-field is sparse in autopilot data). Plus person 2D-box (seg) by distance,
   plus bbox2d 2D-box IoU as cross-check.

## DECISION GATE (near-field, <=20 m)
- MET (recall>=0.9 & precision>=0.85 near, F1>=0.9): finalize, write results, mark model the deliverable.
- SHORT -> diagnose and fire the cheapest sufficient lever, re-eval, iterate:

### Improvement levers (cheap -> expensive)
A. DECODE-ONLY (minutes, no retrain) — try FIRST:
   - NMS radius sweep 6/8/10 (precision). Threshold sweep 0.08/0.10/0.15/0.20 (recall<->precision).
   - Radar-gated decoding (marginal, but free).
   Pick the operating point that hits the near-field target.
B. LOSS-WEIGHT RETRAIN (~45 min, frozen backbone) if recall-limited:
   - center weight 2->4, keep dim 0.6; re-train det head; re-eval.
C. PARTIAL BACKBONE UNFREEZE (~1 h) if still recall-limited:
   - unfreeze_backbone_last_n=2-3 for the det head (let features adapt to objects), freeze_bn=true.
D. TIGHTER RANGE GATE (focus capacity near): retrain gated 30 m; report near-field.
E. If bbox2d 2D-box is weak but seg-box is good: rely on seg-box for person 2D (already strong); note.

## Guardrails (unchanged)
- No detector swap (R-CNN/YOLO stay in the OD phase). No giant new collection without sign-off.
- Edit only /abiodun + /cooperative_fusion. Tear down CARLA when idle (not needed here; eval is offline).
- Each step writes results to disk; report honestly incl. sample sizes and any target miss.

## Report to leave for morning
Final table: vehicle seg IoU + person 2D-box IoU/recall by distance + detection P/R/F1 (car+person)
by distance at the chosen operating point; which levers were used; honest gaps + next options.
