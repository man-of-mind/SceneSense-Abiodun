# Cooperative fusion module — validated (2026-06-26)

fusion.py: per-view detections -> {mean, covariance-weighted, triangulation} world position.

## Monte-Carlo (XY position error vs GT; depth_std=1.2 m, bearing_std=0.3 deg)
| view separation | single-view | mean/cov | triangulate |
|---|---|---|---|
| 6 m  | 0.96 | 0.70 | 0.28 |
| 12 m | 0.97 | 0.72 | 0.16 |
| 20 m | 0.97 | 0.74 | 0.14 |

## Conclusion
- Averaging: ~1/sqrt(2) variance reduction (depth-limited, ~0.7 m).
- Triangulation: 3.5-7x better than single-view (0.14-0.28 m), improving with baseline
  (bearing-limited). This dissolves the single-view depth variance that loss-tuning /
  ground-plane prior / radar pps could not fix -- the core reason for cooperative perception.

## Next
- Wire to LIVE two-view perception: two static egos view the same car+human, run archK per
  view -> bearings + world pos -> fuse -> compare to CARLA GT (Phase 2 on real data).
- Collector fix (PERSON_TAGS + person rasterization) -> Phase 1 retrain for the human side.

---
## Phase 0b-2 live single-view inference (archK) — finding
archK loads + runs in-process fine (live RGB + radar rasterization + decode all work), BUT it
is badly OUT-OF-DOMAIN on the static parked-152 scene:
- vehicle: NOT detected; person: detected but 24 m off (depth wildly wrong), score 0.32.
archK was trained on MOVING-ego data (~0.35 recall) and does not transfer to static parked.

Implication: the live two-view fusion DEMO needs an IN-DOMAIN model that reliably detects the
car+human. The fusion math is already validated synthetically; the missing piece is a detecting
model in the target (static) domain.

=> Phase 1 is the keystone: collect static-ego data (correct labels via PERSON_TAGS fix +
person rasterization) + retrain archK recipe -> in-domain detecting model -> then live fusion.

## Pilot in-domain retrain + live re-test (2026-06-26) — DECISIVE
- Pilot = parked-ego in Town10 AMBIENT traffic (17.6 obj/frame, ~10 peds/frame), not the controlled 1-car/1-human scene.
- Corrected person labels train cleanly: seg vehicle IoU 0.948, val mIoU 0.754. Label bug is dead.
- Pilot test detection: F1 0.22, recall 0.18 (dense crowd). 2D box-size head collapsed to points again (IoU ~0).
- Live controlled scene (phase0b2), corrected model: heatmap peaks vehicle 0.148 / person 0.170 (under-confident -> 0 dets at 0.20).
  At thr 0.08: vehicle xy_err 9.3 m / bearing_err 36.8 deg; person 2.7 m / bearing_err 40.4 deg (scores 0.11-0.16).
- CONCLUSION: learned object head too weak + bearing far too imprecise (37-40 deg) for triangulation (needs <1 deg).
  SEG (esp vehicle 0.95) is excellent. -> Derive per-view bearings from RADAR (primary) + vehicle seg-centroid, NOT the object head.
  Do NOT commit to a large "fix-detection" collection: in-domain data did not fix under-firing.

## Track B — detection-head calibration (2026-06-26): NEGATIVE RESULT
- Hypothesis: head under-fires (peaks ~0.15) because adaptive-radius focal targets too soft.
- Tried: sharp fixed radius 2 px + 4x center-loss weight, stage-2 frozen backbone, 40 ep.
- Result: F1 0.357 / recall 0.329 / person F1 0.391 / veh F1 0.322 (vs baseline F1 0.345). No real gain.
- Detection F1 now plateaued ~0.35 across GIoU, adaptive radius, sharp radius, high center wt,
  partial unfreeze, distillation. => limit is ARCHITECTURAL (LR-ASPP/MobileNetV3 heatmap head is a
  weak detector), not tuning. If detection is needed: change backbone/head (discuss w/ supervisor);
  do NOT keep grinding calibration. Cooperative fusion (seg + radar bearings) is the productive path.
