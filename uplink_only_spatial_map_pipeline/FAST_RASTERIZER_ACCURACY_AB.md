# Fast radar rasterizer accuracy A/B

Date: 2026-07-29

Status/correction: use this file only as a live-run sanity check. Do not use
the ~3 m loose-matcher localization values below as the model accuracy floor,
and do not compare them directly against the offline knob-matrix localization
numbers. For the actual rasterizer-equivalence evidence, use
`RADAR_RASTERIZER_SHADOW_VALIDATION.md`, which compares legacy and fast on the
same live radar frames.

Run root:
`runs/accuracy_ab_fast_vs_legacy_20260729`

Purpose: check whether the opt-in fast radar rasterizer changes model
localization behavior compared with the legacy Python rasterizer.

## Setup

- Closed-loop/full-result loopback, not uplink-only/no-return.
- Same route/model/traffic recipe for both runs:
  - Town10HD_Opt
  - moving ego, route `80,85,91,94,99,80`
  - seed 31
  - 28 requested vehicles, 35 pedestrians
  - no-AE baseline checkpoint
  - 200k radar PPS
  - radius 4
  - temporal radar window 2
  - zstd feature transport
- 80 frames per condition.
- First 10 frames excluded from the comparison.

The only intentional change was:

| Condition | Flag |
|---|---|
| Legacy | `--radar-rasterizer legacy` |
| Fast | `--radar-rasterizer fast` |

## Evaluation method

For each post-warm-up frame:

1. Keep predicted `vehicle` objects within 40 m.
2. Keep visible GT vehicles within 40 m.
3. Compare predictions against GT actor origins (`origin_x`, `origin_y`),
   because the training/validation convention predicts actor origin rather than
   bounding-box center.
4. Greedily match by nearest XY distance with a 5 m maximum match radius.
5. Report matched localization error.

This is a short live A/B sanity check, not a full offline benchmark. The two
runs are independent CARLA runs with the same seed/route, so count/match
differences can come from run-to-run traffic timing and the loose nearest-GT
matcher. The stronger invariance evidence is the same-frame shadow validation,
where both rasterizers consume the exact same live radar measurement.

## Result

| Metric | Legacy | Fast | Delta |
|---|---:|---:|---:|
| Eval frames | 70 | 70 | +0 |
| Visible GT vehicles <=40 m | 228 | 228 | +0 |
| Predicted vehicles <=40 m | 174 | 175 | +1 |
| Matched vehicles | 60 | 56 | -4 |
| Precision @5 m | 34.5% | 32.0% | -2.5 pp |
| Recall @5 m | 26.3% | 24.6% | -1.8 pp |
| Loc error mean | 3.016 m | 3.035 m | +0.018 m |
| Loc error p50 | 3.113 m | 3.095 m | -0.018 m |
| Loc error p90 | 4.126 m | 3.999 m | -0.127 m |
| Loc error p95 | 4.456 m | 4.405 m | -0.051 m |
| Matched score mean | 0.273 | 0.264 | -0.010 |
| Matched score p50 | 0.188 | 0.181 | -0.007 |

Latency/result sanity from the same post-warm-up frames:

| Metric | Legacy p50 | Fast p50 | Delta |
|---|---:|---:|---:|
| Closed-loop RTT | 22.6 ms | 14.8 ms | -7.8 ms |
| Edge tail/back time | 10.3 ms | 6.0 ms | -4.2 ms |
| Feature payload | 1069.2 KB | 1068.6 KB | -0.6 KB |
| Result payload | 3.9 KB | 4.0 KB | +0.1 KB |
| Object count | 4.0/frame | 4.0/frame | +0.0 |

## Interpretation

This independent-run A/B did not show a fast-rasterizer-specific localization
shift. Matched localization error under this loose matcher is effectively
unchanged:

- mean changes by only +0.018 m;
- p50 changes by -0.018 m;
- p95 changes by -0.051 m.

The short run shows a small match-count/precision/recall dip for fast, but it is
not accompanied by worse localization error and is within what can plausibly
come from independent live CARLA run timing plus the loose matching rule.

The publishable equivalence claim should come from
`RADAR_RASTERIZER_SHADOW_VALIDATION.md`, not from this independent-run matcher:
same-frame shadow decoding produced identical object counts on every frame,
zero unmatched decoded objects, identical center pixels, and at most about
1 cm world-XY difference.

For project decisions now, `--radar-rasterizer fast` looks safe enough to use
for the next loopback/OAI Track-1 profiling runs, while keeping legacy as the
fallback.
