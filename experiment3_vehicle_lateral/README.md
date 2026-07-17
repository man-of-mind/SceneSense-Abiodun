# Experiment 3 — parked-ego vehicle lateral diagnostic

## Current stop/go state

**STOP: the centered condition did not reproduce the accepted localization floor. No lateral crossing was run.**

Retained centered runs:

- `centered_200k_15m_v1/` — original corrected centered gate at 15 m.
- `centered_200k_10m_v1/` — follow-up centered gate at 10 m, same recipe except target distance.

The run used the current integrated no-AE checkpoint through actual UDP loopback with per-channel uint8, zlib,
ROI 0, score analysis threshold 0.20, NMS radius 2, and top-k 120. The parked Lincoln ego was placed at training
route spawn 80. A tagged Lincoln NPC was physics-settled and then held exactly 15 m ahead at zero lateral offset.
There were no background actors. The RGB/radar recipe matched training: 1280×720 camera at 120° FOV, 768×432
model input, radar 200,000 points/s at 120°×30°, 120 m range, raster radius 4, and a two-frame temporal maximum.

An initial harness audit found that freezing the vehicles immediately left the ego at the elevated CARLA spawn
transform and raised the camera to about 2.30 m. Those invocations were deleted. The canonical run first applied
the training collector's 30 physics-settling ticks. It then measured ego origin `z=0.020 m`, target origin
`z=-0.003 m`, and camera world `z=1.565 m`, matching the original training capture at spawn 80 (`~1.57 m`).

## 15 m centered result

| check | result |
|---|---:|
| measured target opportunities | 60 |
| loopback results received | 60/60 |
| target visible | 60/60 |
| target placement | forward 15.000 m; lateral 0.000 m; pixel offset 0.00 px |
| raw radar support | mean 1,860.2 points; 60/60 supported |
| ≤2 m target matches | 59/60 (98.3%) |
| error among ≤2 m matches | mean 1.474 m; median 1.483 m; p90 1.569 m |
| ≤5 m target matches | 60/60 |
| error among ≤5 m matches | mean 1.506 m; median 1.483 m; p90 1.577 m |
| mean target score | 0.247 |
| learned radar-support score | ~1.000 |
| provisional acceptance bound | mean ≤1.30 m; median ≤1.20 m |
| centered accuracy gate | **FAIL** |

The miss is not a delivery, placement, visibility, camera-height, raw-radar, or learned-radar-support failure.
Saved overlays show multiple vehicle peaks around the same centered NPC. The nearest score-qualified peak is
stable but not accurate enough. The first ten frames favored a peak ahead of the target (mean forward error
`+1.49 m`); later frames generally favored a peak behind it (about `-1.3 m`) with a smaller lateral bias. This
peak ambiguity explains why the full-run mean signed error alone (`-0.654 m` forward, `-0.489 m` lateral) hides
the larger radial localization error.

For context, the current no-AE held-out evaluation reports overall localization MAE `0.948 m` and vehicle MAE
`0.876 m`. Within its matched 10–15 m vehicles whose centers were within 50 px of image center (`n=21`), mean /
median / p90 were `0.699 / 0.688 / 1.014 m`, with all 21 within 2 m. The live centered target therefore does not
reproduce either the overall or comparable near-center held-out distribution. The older historical knob matrix's
~1.2 m number also remains below this run's 1.47–1.51 m result.

## 10 m centered follow-up

The 10 m run used the same corrected harness, no-AE checkpoint, loopback/u8/zlib/ROI-0 transport, 200k radar
recipe, spawn 80, 30 settle ticks, and zero lateral offset. It passed all protocol checks: 60/60 loopback results,
60/60 visible target opportunities, exact 10.000/0.000 m placement, camera world `z=1.565 m`, and raw radar
support in every frame (mean 3,354 points).

At the frozen score >=0.20 analysis threshold, it still failed the stop/go gate:

| check | result |
|---|---:|
| measured target opportunities | 60 |
| score-qualified vehicle frames | 52/60 |
| <=2 m target matches | 20/60 (33.3%) |
| error among <=2 m matches | mean 1.239 m; median 1.251 m; p90 1.336 m |
| <=5 m target matches | 52/60 (86.7%) |
| error among <=5 m matches | mean 2.215 m; median 2.657 m; p90 3.016 m |
| raw radar support | mean 3,354.0 points; 60/60 supported |
| centered accuracy gate | **FAIL** |

The 10 m failure is mainly a score-calibration / duplicate-peak issue. Lower-score target peaks are often more
accurate than the score >=0.20 candidate. A read-only threshold diagnostic showed:

| analysis score threshold | <=2 m matches | mean / median / p90 error |
|---:|---:|---:|
| 0.05 | 60/60 | 0.984 / 1.026 / 1.079 m |
| 0.10 | 60/60 | 1.106 / 1.113 / 1.284 m |
| 0.15 | 57/60 | 1.216 / 1.194 / 1.590 m |
| 0.20 | 20/60 | 2.215 / 2.657 / 2.992 m |

So reducing distance to 10 m does not pass the frozen centered gate, but it reveals that the correct peak is often
present below the official score threshold. Before any 10 m lateral sweep, explicitly re-freeze the score threshold
and target-selection rule; do not silently switch thresholds.

## Files and boundary

- `centered_200k_15m_v1/CENTERED_RESULT.md` — concise stop/go result.
- `centered_200k_15m_v1/analysis/summary.json` — protocol checks and machine-readable metrics.
- `centered_200k_15m_v1/analysis/target_frame_analysis.csv` — one row per target opportunity.
- `centered_200k_15m_v1/frames/` — raw and annotated sanity frames.
- `centered_200k_10m_v1/CENTERED_RESULT.md` — 10 m stop/go result.
- `centered_200k_10m_v1/SCORE_THRESHOLD_DIAGNOSTIC.md` — 10 m threshold sensitivity note.
- `../staleness/analyze_experiment3_vehicle_lateral.py` — reproducible target-only analysis.

Do not create or interpret a lateral folder until a centered run satisfies a deliberately frozen gate. If general
live-model parity is investigated next, keep it separate: run 1–2 loops on the original `80,85,91,94,99,80`
training route with the original low/medium/crowded recipe and compare the resulting live GT/predictions to
offline metrics.
