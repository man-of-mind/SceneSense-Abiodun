# Evaluation-contract decision — advisor-rich v4

**Decision: RE-COLLECT REQUIRED — confirmed sensor-contract drift.** The
`20260813_014501_full` quarantine stays in force. Do not freshness-rescore it,
run the controller ladder on it, or use it for RL. Traffic realism remains a
valid 24/24 win; this decision concerns the perception inputs, not NPC traffic.

This was a desk-only audit of the saved corpus. No CARLA process was launched.
`pcarv4_fast_te01` was excluded before threshold selection and evaluation
because its lead/walker collision is independently invalid. The retained
23-run corpus was not modified.

## Method

- Match representation: class-aware actor-origin XY center distance, one-to-one
  greedy assignment, 5 m gate.
- GT denominator: actor origin in camera frustum and at 0–25 m. Predictions are
  restricted to decoded range 0–25 m.
- Threshold selection: maximize F1 independently per class on complete
  **validation trajectories** over a 0.005 score grid; break ties toward the
  higher threshold. Test trajectories are used only after thresholds are
  frozen.
- Range reporting: 0–5, 5–10, 10–12, 12–15, 15–20, and 20–25 m, plus
  cumulative ≤12 m and ≤25 m recall.
- Radar comparison: the identical logged metric, valid projected returns per
  detection frame, against the decisive retained-input diagnostic.

The reproducible analysis is in
[`20260813_035529_desk_v4`](experiments/policy_corpus_advisor_rich_v4/20260813_014501_full/evaluation_contract/20260813_035529_desk_v4/analysis_manifest.json).
It includes source hashes, CSV tables, and both PNG and PDF figures.

## Precision-recall result

![Per-class precision-recall curves](experiments/policy_corpus_advisor_rich_v4/20260813_014501_full/evaluation_contract/20260813_035529_desk_v4/precision_recall_by_class.png)

| Class | Validation-selected threshold | Validation P / R / F1 | Held-out test P / R / F1 |
|---|---:|---:|---:|
| Pedestrian | 0.195 | 49.07% / 41.60% / 45.03% | 52.67% / 38.74% / 44.64% |
| Vehicle | 0.115 | 11.53% / 49.52% / 18.71% | 5.84% / 49.51% / 10.44% |

The result rejects one flat score threshold for both classes: the validation
knees differ materially. These values are **diagnostic thresholds for this
drifted corpus**, not production thresholds to copy into the next verifier.
They must be re-estimated on corrected validation trajectories.

The vehicle precision value is also a conservative lower bound: actor-origin GT
enumerates CARLA vehicle actors, while Town10HD can contain static vehicle scene
objects detected by the model but absent from that actor inventory. A future
headline vehicle PR claim therefore needs exhaustive static-object annotation
or an explicitly actor-only prediction mask. Recall against known actors is not
affected by this precision-denominator caveat.

## Coverage versus range

![Per-class coverage versus range](experiments/policy_corpus_advisor_rich_v4/20260813_014501_full/evaluation_contract/20260813_035529_desk_v4/coverage_vs_range_by_class.png)

Near-field coverage does not recover to the expected on-contract level. At the
validation-selected thresholds, cumulative ≤12 m recall is only 53.33%
pedestrian / 22.64% vehicle on validation and 61.35% / 29.17% on test.

Even the lowest score retained by the live decoder (0.05) is insufficient:

| Split | Class | Recall ≤12 m @0.05 | Recall ≤25 m @0.05 |
|---|---|---:|---:|
| Validation | Pedestrian | 74.02% (322/435) | 57.25% (545/952) |
| Test | Pedestrian | 72.32% (290/401) | 51.96% (503/968) |
| Validation | Vehicle | 41.51% (22/53) | 64.90% (135/208) |
| Test | Vehicle | 58.33% (14/24) | 67.96% (70/103) |
| All 23 valid runs | Pedestrian | 72.57% (1225/1688) | 52.83% (2052/3884) |
| All 23 valid runs | Vehicle | 47.28% (87/184) | 62.11% (477/768) |

The pedestrian 0–5 m bin is strong at the 0.05 floor (85.50% validation,
98.10% test), but recall falls within the still-safety-relevant 5–12 m region.
Vehicle recall is also non-monotonic, peaking at 15–20 m rather than the near
field. This is not the “near field is sound; only far/occluded objects are low”
pattern required to lift quarantine.

## Decisive sensor-contract check

| Source | Frames | Radar returns/frame median | P05–P95 | Control / detection clock | Requested radar pps |
|---|---:|---:|---:|---:|---:|
| Advisor-rich v4, 23 valid runs | 9,120 | **9,721.0** | 9,044.95–9,999.0 | 20 / 10 Hz | 200,000 |
| Retained on-contract diagnostic | 140 | **18,591.5** | 18,310.75–18,692.3 | 10 / 10 Hz | 200,000 |

The corpus median is **52.29%** of the on-contract reference. The values are
directly comparable: both count valid projected returns emitted by
`PoleRadarPipeline.build_tensor`, and the retained-input `raw_radar_points`
column equals that diagnostic metric frame for frame.

The resolved configurations explain the split. The reference had a 10 Hz world
tick and radar on every tick. The corpus correctly requested a 10 Hz sensor
period but advanced physics/control at 20 Hz. In this CARLA build, the radar
measurement count is budgeted from `points_per_second × fixed_delta_seconds`;
the corpus ceiling of 10,000 is exactly 200,000 / 20. `sensor_tick=0.1` skips an
emission but does not integrate the skipped tick's point budget. Therefore each
10 Hz tensor received roughly half the training-contract radar evidence. The
two-frame temporal window is half-density too.

This is a genuine live input-contract drift that argument/config validation did
not catch, because it checked requested settings rather than observed tensor
density.

## Replacement acceptance contract

For the corrected corpus, acceptance should be evaluated in this order:

1. **Observed sensor-density gate (before coverage):** retain the requested
   10 Hz detection, 1280×720, FOV 120°, legacy radius 4, temporal window 2 and
   separate 20 Hz control clock, but require every run's projected-radar median
   to be within ±10% of the retained 18,591.5/frame reference
   (16,732–20,451). A short smoke must pass this empirical gate before scale-up.
2. **Per-class operating point:** choose one threshold per class by maximum F1
   on whole validation trajectories only, then freeze both before test. Report
   the full PR curves and the operating points. Do not inherit 0.20 and do not
   choose on pooled train/test frames.
3. **Safety-relevant coverage gate:** on held-out test trajectories, require
   direct actor-origin recall at ≤12 m of at least 80% for pedestrians and 90%
   for vehicles at their frozen thresholds. Report trajectory-grouped 95% CIs;
   do not substitute the CI or a pooled frame-random split for the point gate.
4. **Range profile:** always report the six range bins above. Recall over the
   full diverse 0–25 m population is descriptive, not a hard acceptance gate.
5. **Scenario validity:** exclude predeclared invalid runs such as
   `pcarv4_fast_te01`; do not weaken perception gates to absorb collisions or
   bad trajectories.

The dual-clock radar issue must be corrected and verified by a tiny smoke before
another full collection. Because the density defect affects every v4 run, a
replacement corpus is justified by global sensor drift—not by the single bad
`fast_te01` run and not by an attempt to chase the inherited flat 0.20 gate.
Freshness, controller baselines, and RL remain held until that corrected corpus
passes this contract.
