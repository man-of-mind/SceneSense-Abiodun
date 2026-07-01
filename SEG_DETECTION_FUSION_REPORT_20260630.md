# SceneSense RGB+Radar Fusion Model — Final Results (2026-06-30)

## Headline
A single multi-task model (shared MobileNetV3/LR-ASPP backbone, RGB + radar fused at the first conv)
that delivers, on a freshly collected moving-ego dataset:
- **Vehicle segmentation IoU 0.914** (fusion adds +0.15 over RGB-only 0.66).
- **Vehicle detection at target**: 0–10 m F1 0.92, 10–20 m F1 0.96.
- **Person 2D-box localization** near-field: recall 0.86–0.92, precision ~1.0.
- **Detection XY localization error ~1.1 m.**

## The data
Re-collected with the corrected pipeline: moving ego on a fixed route, **8 loops × 3 densities
(low/medium/crowded), 15,196 frames**, Town10. Radar 100k pps, raster-4, temporal-window-2.
- **Vehicle GT**: semantic camera (renders correctly).
- **Person GT**: engine 3D bounding box → projected **2D box** (see GT-fix below).

## GT fix (important methodology note)
CARLA 0.10 (UE5) **does not render pedestrians into ANY ground-truth sensor** — semantic camera,
instance camera, AND depth camera all return the background behind a walker (verified: depth read
10.4 m at a pedestrian truly 6.6 m away; 0 person-tag pixels across 300 frames). Pedestrians ARE
visible in RGB (a VOC-pretrained net segments them), so it is purely a missing-label problem.
**Decision (with supervisor): person GT = the engine 3D box projected to a 2D box** — accurate
localization, no silhouette needed. Vehicles keep the (correct) semantic-camera pixel mask.

## Training recipe (2-stage)
1. **Segmentation (seg-only)**: Lovász loss 0.5 + BN-freeze + batch-24 + cosine LR + person-weighted
   CE [0.5,1,4] + `person_miou` checkpoint selection. → **vehicle IoU 0.9145, mIoU 0.8335.**
2. **Detection head** (backbone + seg head frozen): bbox2d (GIoU) + adaptive heatmap radius +
   **operating-range gate ≤40 m** + center-loss weight 4 + dim/yaw weights. Decode at NMS-6.

(Note: an earlier end-to-end pipeline trained a *generic joint* recipe — plain CE, batch-2 — which
dropped vehicle IoU to 0.82. Re-applying the seg-only Lovász recipe recovered it to 0.914. The
technique hadn't "stopped working"; it simply wasn't applied by the generic pipeline.)

## Segmentation results (test)
| Class | IoU |
|---|---|
| Vehicle | **0.914** (RGB-only baseline 0.663 → fusion +0.15) |
| Person (2D-box region) | 0.575 global; near-field much higher (below) |
| mIoU (bg/veh/person) | 0.834 |

## Detection by distance (operating point: NMS-6, gated ≤40 m, score thr 0.10)
**Vehicle:**
| Distance | Recall | Precision | F1 |
|---|---|---|---|
| 0–10 m | 0.93 | 0.90 | **0.92** |
| 10–20 m | 0.96 | 0.97 | **0.96** |
| 20–30 m | 0.87 | 0.91 | 0.89 |
| 30–40 m | 0.76 | 0.82 | 0.79 |

**Person:**
| Distance | Recall | Precision |
|---|---|---|
| 0–10 m | 0.86 | 0.98 |
| 10–20 m | 0.76 | 0.87 |
| 20–30 m | 0.75 | 0.74 |
| 30–40 m | 0.76 | 0.57 |

**Person 2D-box accuracy (seg-region box vs engine GT box), near-field:** recall 0.92 / mean box IoU
0.73 at 10–20 m. **XY localization MAE ~1.1 m.** Aggregate gated detection F1 ≈ 0.80.

## Person near-field recall: honest gap + diagnosis
Person near-field recall is 0.86 (0–10 m), below the 0.95 vehicle-level target. Diagnosed precisely:
the missed pedestrians are **not smaller** (median 1546 px vs 1467 px for detected) — they **lack
radar returns** (false-negatives are radar-supported only 72% of the time vs 99% for detected). So
the model leans on radar for persons, and radar-less pedestrians are missed. The center-weight-4
retrain improved person recall (0.81 → 0.86 at 0–10 m) by pushing camera-based detection; precision
stayed near-perfect (≥0.98 near).

## Known limitations / next options (not done; for discussion)
- **Person near recall → 0.95**: radar-dropout training (force RGB-only person detection), or a
  targeted near-pedestrian collection (autopilot rarely drives close to pedestrians — 0–10 m is
  sparse, ~57 instances). The controlled cooperative-fusion scenes already reach ~0.95 person near.
- Person *pixel* silhouette is out of scope in this CARLA build (engine 2D box used instead).

## Model artifacts
- **Deliverable (seg + detection, bbox2d):**
  `experiments/autonomous_arch_runs_20260625/det_stage2c_centerw4/checkpoints/det_stage2c_centerw4/best.pt`
- Seg-only backbone: `experiments/seg_lovasz_newdata_20260629/checkpoints/seg_lovasz_newdata_bnfreeze_bs24/best.pt`
- Dataset: `fusion_training_data/moving_ego_tl16_spawn80_fixedroute_speed60_merged_8loops_cap6000_stride2`
- Operating point: 768×432 input, gated ≤40 m, score thr 0.10–0.15, NMS radius 6.
