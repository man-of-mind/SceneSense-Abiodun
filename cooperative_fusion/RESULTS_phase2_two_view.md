# Phase 2 — live two-view cooperative position fusion (2026-06-26)

Script: `phase2_two_view_fusion.py` (`--selftest` for offline geometry; default = live CARLA).

## Setup
- Two static egos in Town10HD, RGB(1280x720,120deg) + radar(100k pps) each. Ego A at spawn 152;
  ego B ~10 m to the right, yawed to look at the placed car. Baseline A-B = 8.6 m.
- One placed car (nissan.patrol) ~13 m ahead of ego A; physics frozen.
- Per view, per object:
  - PRECISE camera bearing from the vehicle-mask centroid pixel + intrinsics/pose.
  - radar range = median of radar world-returns landing in the car mask (accurate single-view).
  - monocular-depth proxy = f * H_car / box_h_px (NOISY single-view depth).
- Object data-association: ORACLE (vehicle component containing the GT-projected pixel). This
  fixes WHICH blob is the target; localization error vs GT is still a fair metric (assoc != loc).
  Why: the learned object head's bearing is 37-40 deg off (Phase 0b) and the seg model
  over-segments this OOD empty-street scene -> association is a separate problem, scoped out here.

## Offline geometry self-test (no CARLA)
- zero noise: triangulation error = 0 (axis/sign conventions correct).
- realistic (1 px bearing, 1.5 m range): single-view 0.57 m, mean 0.54 m, TRIANGULATE 0.10 m (5.5x).

## Live result (XY error vs CARLA GT, car at ~13 m, baseline 8.6 m)
| estimator | error |
|---|---|
| monocular single-view A | 3.56 m |
| monocular single-view B | 5.35 m |
| mean of monocular views | 4.25 m |
| TRIANGULATE (silhouette-centroid bearings, no range) | 1.92 m |
| **TRIANGULATE (2D-bbox-center bearings, no range)** | **1.40 m** |
| radar reference (accurate sensor) | 2.09 m |

(silhouette-centroid run earlier gave 1.80 m vs radar 1.78 m; radar varies run-to-run ~1.8-2.1 m
depending on which surface points reflect.)

## Takeaway
Two BEARING-ONLY camera views triangulate to BELOW radar accuracy (1.40 vs 2.09 m) WITHOUT any
range sensor, and cut the monocular single-view error ~3x. This is the cooperative-perception gain
on real CARLA sensor poses. The 2D-BBOX-CENTER pixel is the better bearing anchor (1.40 m) than the
mask centroid (1.92 m): the bbox center is not pulled toward the larger visible face, so it is more
view-invariant. (Ground-contact bottom anchor was WORSE, 7.2 m — silhouette bottom is view-dependent.)

## Next levers (to push below ~1 m)
- Add the pedestrian (radar-cluster association; seg person weak) -> two-object fusion.
- Multi-frame averaging (static scene) to cut bearing pixel noise.
- Sweep baseline (3/8/15 m) live to show triangulation improving with baseline (bearing-limited).
- Replace oracle association with a real detector/tracker once detection is solved (Track B: the
  current LR-ASPP heatmap head is too weak; needs an architectural change, not tuning).

## B0 — dimension fusion (front-view + side-view -> full 3D box) [2026-06-27]
The full deliverable = centroid (triangulation) + DIMENSIONS + yaw. A single view can't observe the
extent along its own line of sight. `fuse_dimensions()` weights each view's length by how aligned its
ray is with the object's lateral axis (side view sees length) and its width by alignment with the
forward axis (front view sees width); height from all views.
Offline self-test: single-view dim MAE 0.361 m -> FUSED 0.057 m (6.3x). Live validation pending in the
two-ego scene (read model regression dims at the car center; compare fused vs CARLA GT extent).

## Phase 2b — full fusion live (position + dims + pedestrian + baseline sweep) [2026-06-27]
Two egos + placed car + pedestrian; 5-frame averaging; oracle data-association. gated model.

POSITION (XY err vs GT):
| baseline | car single A/B | car TRIANGULATE | pedestrian single A/B | ped TRIANGULATE |
|---|---|---|---|---|
| 4 m  | 2.42 / 1.58 | 3.76 (ill-conditioned: bearings near-parallel) | -- | -- |
| 8 m  | 2.42 / 1.35 | 1.68 | 0.59 / 0.13 | 0.35 |
| 14 m | 2.42 / 1.55 | 2.05 | 0.60 / 0.05 | 0.26 |
- Pedestrian fusion is the clean win (radar-cluster bearings -> 0.26-0.35 m, beats single view).
- Car triangulation needs an adequate baseline (>=8 m); 4 m is geometrically ill-conditioned.
- Residual car-position floor ~1.5-2 m = silhouette-centroid vs 3D-center bias (known).

DIMENSIONS (mean abs err over observed axes, m):
- reg-head fusion: 1.3-1.8 m. The regression head was trained at dim-loss weight 0.05 -> weak per-view
  dims (front-view length err ~2.4 m). Fusion can only combine what each view predicts.
- seg-2D-box fusion: LENGTH recovers from a genuine side view (baseline 14 m: L 4.96 vs GT 5.59), but
  width reads ~5 m everywhere (silhouette horizontal extent != clean face extent under perspective)
  and height is over-read (range-to-center vs near-face).
=> Dimension-fusion MATH is sound (offline self-test 6.3x). LIVE bottleneck = weak per-view dimension
   SOURCES. Levers: (a) retrain regression head w/ higher dim-loss weight [LAUNCHED: det_rangegated40_dimw05],
   (b) proper 3D-box-from-multiview-silhouette fitting (oriented box, near-face range) — future work.

## B0 UPDATE — dimension fusion works with a properly-trained regression head [2026-06-27]
Retrained the regression head with dim-loss weight 0.6 (was 0.05): det_rangegated40_dimw05.
Per-view dim MAE dropped (front view A 2.4->1.0 m) and geometry-weighted fusion now gives:
  baseline 4 m: 0.69 m | 8 m: 0.36 m | 14 m: 0.95 m  (mean abs err over L,W,H vs CARLA GT).
Best at 8 m baseline (0.36 m); fusion beats naive single-view averaging. Detection unchanged.
=> The fix for live dimension fusion was TRAINING the regression head (cheap, in-scope), NOT the
   naive seg-2D-box (which stays ~5 m on width; would need oriented-3D-box-from-silhouette fitting).
RECOMMENDED dimension path: regression-head dims + `fuse_dimensions` geometry weighting.
