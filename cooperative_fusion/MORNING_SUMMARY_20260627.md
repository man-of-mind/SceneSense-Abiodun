# Morning summary — overnight 2026-06-26 -> 27

TL;DR: detection was NOT an architectural problem — it was eval policy + NMS. Fixed it on-architecture
(no detector swap). Cooperative fusion now delivers position + pedestrian + dimensions vs CARLA GT.
Best SEG-fusion model: `det_rangegated40_dimw05`. Nothing running; GPU free; CARLA down.

## Phase A — detection FIXED for the operating range (no detector swap)
Root cause (pinned with data): 63% of GT was >50 m (unresolvable) + NMS radius 2 px left duplicate FPs.
Three fair fixes:
| step | F1 | recall | precision |
|---|---|---|---|
| ungated baseline | 0.357 | 0.329 | 0.391 |
| + range gate (eval only, 40 m) | 0.465 | 0.463 | 0.466 |
| + gated RETRAIN (40 m targets) | 0.587 | 0.645 (0.74 @thr0.1) | 0.539 |
| + NMS 6 px | 0.707 | 0.665 | 0.754 |
| + dim/yaw-weight retrain (BEST) | **0.777** | **0.790** | **0.765** |
Near-object (the target): **vehicle 0-10 m recall 0.90, 10-20 m 0.87; person 0-10 m 0.95.**
Vehicle precision 0.85. Residual gap: persons at 10-40 m. Operating point: 40 m gate, thr 0.10-0.20, NMS 6 px.
Radar-gated decoding was only marginal — NMS was the precision lever.

## Phase B — cooperative fusion (full world 3D box: centroid + dims + yaw)
Two static egos + placed car + pedestrian, 5-frame averaging, oracle data-association, vs CARLA GT.
POSITION (triangulation of camera bearings):
- car: 1.68 m @ 8 m baseline (beats radar ~2 m); 4 m baseline ill-conditioned (need >=8 m).
- PEDESTRIAN: 0.26-0.35 m (radar-cluster bearings) vs single-view 0.6 m — clean cooperative win.
DIMENSIONS (`fuse_dimensions`, geometry-weighted): with the dim-trained head, fused mean-abs err
**0.36 m @ 8 m baseline** (0.69 @4 m, 0.95 @14 m). Math also validated offline (6.3x).
- The fix was TRAINING the regression head (dim weight 0.05 -> 0.6); the naive seg-2D-box extent
  stayed poor (width ~5 m) and would need oriented-3D-box-from-silhouette fitting.

## Models & files
- BEST checkpoint: experiments/autonomous_arch_runs_20260625/det_rangegated40_dimw05/.../best.pt
- Detection plumbing: range gate in object_targets/train_fusion/evaluate_fusion + driver
  (MAX_GT_DISTANCE_M); radar gate + NMS args in evaluate_fusion.
- Fusion: cooperative_fusion/fusion.py (estimators + self-tests), phase2_two_view_fusion.py (car pos),
  phase2b_full_fusion.py (full: +pedestrian +dims +baseline sweep +multi-frame).
- Results: RESULTS_fusion_module.md, RESULTS_phase2_two_view.md, phase2b/phase2b_results.json.

## Suggested next steps (for discussion)
1. Persons at 10-40 m: the remaining detection gap. Lever: person-distance loss weighting, or
   seg-CC+radar detection for pedestrians (seg-person is weak; radar reliable).
2. Replace oracle association with the (now working) detector for a fully end-to-end demo.
3. Dimensions: oriented-3D-box-from-two-silhouettes fitting would beat both current sources; or just
   ship the regression-head + geometry fusion (0.36 m) as the v1.
4. Moving objects (next phase of the plan) once the static demo is locked.
5. OD-fusion model (separate phase): Faster R-CNN+FPN baseline vs YOLO — detector + 7-ch localization.
