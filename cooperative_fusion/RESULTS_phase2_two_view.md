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
