# Evaluation-contract decision — advisor-rich v5

**Decision: `FAIL_QUARANTINED`; freshness, controller baselines, and RL remain
stopped.** The native-10-Hz sensor fix is confirmed, but the accepted
trajectory-held-out near-field contract is not satisfied and one validation
trajectory contains a real ego/walker impact.

The authoritative corpus is
`experiments/policy_corpus_advisor_rich_v5/20260813_045142_full`. All 24 runs
completed their online collection and cleanup gates (8,480/8,480 frames, zero
logged NPC collision incidents, zero leaked actors). The authoritative offline
verification is `verification/20260813_055323`; the earlier `055017` directory
is an interrupted pre-manifest attempt preserved for provenance.

## What is fixed

- The corpus was collected with one native 10 Hz world tick per 10 Hz sensor
  frame at 200,000 radar points/s, 1280x720, camera/radar FOV 120 degrees,
  legacy raster radius 4, and temporal window 2.
- Corpus median projected radar support is **19,412/frame**, or **104.41%** of
  the retained on-contract reference (18,591.5), inside the registered +/-10%
  band. Per-run medians are 19,339--19,532.
- All eight exact-fast runs retain the target on the authored route for 5.9 s;
  their maximum route offset is 3.46 m and no fast-run walker-impact signature
  is present.

## Frozen operating points and held-out result

Thresholds were selected independently by maximum F1 on complete validation
trajectories, then frozen before test:

| Class | Threshold | Validation P / R / F1 | Test P / R / F1 |
|---|---:|---:|---:|
| Pedestrian | 0.180 | 47.97% / 43.23% / 45.48% | 44.31% / 50.00% / 46.98% |
| Vehicle | 0.270 | 20.59% / 58.33% / 30.43% | 18.38% / 54.40% / 27.47% |

At the registered <=12 m test gate, pedestrian recall is **251/389 = 64.52%**
(trajectory-bootstrap 95% CI **43.75--66.80%**), below 80%. The test split has
**zero <=12 m vehicle rows**, so the >=90% vehicle gate is not evaluable. Even
at the decoder floor 0.05, all-pedestrian near recall is only **296/389 =
76.09%**; lowering the threshold cannot rescue the registered denominator.

The controlled crossing target is a materially different denominator: across
the two pedestrian test trajectories it scores 76.49% at 0.18 and 84.70% at
0.05. This explains why the retained close-target diagnostic and the diverse
all-pedestrian gate should not be treated as interchangeable, but changing the
denominator or choosing a recall-oriented threshold requires advisor review.

Vehicle precision is additionally confounded by the known actor-only GT issue:
static Town10HD vehicle objects can be predicted without appearing in the actor
inventory. Therefore maximum actor-only F1 may not be the right production
threshold rule until predictions are masked to an exhaustive annotation scope.

## Independent scene-validity failure

`pcarv5_mixed_va01` contains ten derived pedestrian-speed samples above 3.5
m/s (maximum **4.317 m/s**). The same walker approaches to 0.35--1.10 m from
the ego and is displaced at 3.7--4.32 m/s: this is an ego/ambient-walker impact,
not normal controller motion. The existing collision monitor covered managed
NPC vehicles but not the collector ego.

The code-side repair is prepared: direct-route ego control now yields to
ambient as well as controlled walkers, and every scenario family receives a
fail-fast pedestrian-impact motion gate. This repair has unit coverage but has
not authorized or launched another CARLA collection.

## Required decision before more data

Do not average in easy top-ups merely to make a pooled gate pass. Review and
freeze these two points first:

1. Is the pedestrian safety gate defined over all in-frustum actor pedestrians,
   or over the registered controlled crossing target with all-object range
   coverage reported separately? If it remains all-object, v5 fails even at the
   decoder floor and needs a perception/occlusion investigation rather than a
   threshold-only change.
2. Add genuine <=12 m vehicle validation/test trajectories so the vehicle gate
   has a denominator. Resolve the static-object annotation/precision caveat
   before using maximum actor-only F1 to choose its threshold.

The 23 unaffected trajectories should remain immutable evidence. Any approved
replacement for `mixed_va01` or near-vehicle top-up must be versioned and
trajectory-grouped; it must not mutate this quarantined batch.

