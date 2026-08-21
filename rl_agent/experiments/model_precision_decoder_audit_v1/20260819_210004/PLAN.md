# Raw Object-Detection Precision / Decoder Audit Plan

Status: frozen before audit-tool implementation  
Audit ID: `model_precision_decoder_audit_v1/20260819_210004`  
Scope: offline model-output diagnosis and bounded decoder/post-processing only

## 1. Questions and hypotheses

The primary question is whether the roughly 0.50 vehicle precision at ROI drop
`q=0` is dominated by redundant predictions around real objects or by genuine
hallucinated objects.

Preregistered hypotheses:

1. **H1 -- same-object duplicates dominate vehicle false positives.** A baseline
   false positive is a duplicate when it is within the frozen 5 m class-aware
   match radius of a same-class GT object that was already claimed by a higher-
   priority one-to-one match.
2. **H2 -- genuine hallucinations are a minority.** False positives with no GT
   of either class within 10 m, plus false positives in GT-empty frames, form a
   smaller share than H1 duplicates.
3. **H3 -- predicted-world suppression is the most causal bounded correction.**
   Greedy, score-ordered, class-aware suppression using only predicted world XY
   should remove duplicates without using GT at deployment time.
4. **H4 -- modest class thresholds may complement suppression.** Vehicle-only
   score increases may improve precision, but are expected to trade recall; a
   combination is eligible only when each component has independent validation
   evidence.
5. **H5 -- image-space suppression is a useful but limited check.** Additional
   suppression among the already-retained decoder peaks can test whether spatial
   peak crowding remains after the current radius-2 decoder. Proper heatmap local-
   maximum suppression cannot be reconstructed without raw heatmaps and will be
   listed as unresolved if those heatmaps cannot be recovered.

## 2. Frozen evidence envelope and provenance

The reproduction source is the four `rl_agent/density_knob/raw/perframe_*.csv`
files restricted to `roi == 0.0`, pooling three complete quantizer profiles per
family. The 6,486 rows per family are 2,162 unique frames repeated for three
quantizers and are never treated as 6,486 independent scenes.

The causal replay source is the 12 per-object files at
`experiments/ae_integrated_20260710/sweeps_permodel_zstd/<family>__<quant>__roi0.0/metrics/test_learned_object_metrics.csv`
for four families and `uint4`, `uint6`, and `uint8`. These files retain the
baseline decoder's prediction score, predicted image box/center proxy, predicted
world XY, and all matched/unmatched GT rows.

Frozen evaluator envelope:

- score threshold: 0.20
- decoder NMS radius: 2 pixels
- decoder top-k: 120 across class heatmaps
- prediction and GT range: at most 40 m from the camera
- matching: greedy one-to-one, class-aware, at most 5 m world-XY distance
- GT minimum 2D box area: 12 px

The input/hash manifest will record all source files, checkpoints, evaluator and
decoder sources, config, Stage-A evidence, sizes, mtimes, and SHA-256 hashes.
Checkpoint metadata and the metric JSONs will pin model and evaluator settings.

## 3. Validation/test separation

The original dataset path recorded in `eval_settings.json` is absent and no
original validation per-object outputs are available. Therefore this audit will
not claim to use the original model-development validation split. Instead it
freezes a disjoint **audit validation / audit test** partition of the 2,162
published test identifiers before inspecting candidate results.

Partition algorithm (frozen):

1. Parse the source prefix (`low`, `medium`, or `crowded`) and the monotonically
   increasing collection index embedded in each `sample_id`.
2. Within each prefix, assign consecutive groups of 25 collection indices to a
   block, keeping nearby frames together.
3. Hash `"decoder-audit-v1|<prefix>|<block>"` with SHA-256.
4. Assign a block to audit validation when the first 64-bit hash value modulo 5
   is 0 or 1 (approximately 40%); otherwise assign it to frozen audit test.
5. Use the same identifier assignment for all four model families and all three
   quantizers. Assert zero identifier and zero block overlap.

The full published test set is used only for independent reproduction and the
baseline FP taxonomy requested by the task. Candidate selection uses audit
validation only. After selection, the selected configuration is applied once to
the frozen audit test by the audit driver.

## 4. False-positive taxonomy

Baseline predictions are rematched with the frozen evaluator rule. Each unmatched
prediction receives exactly one primary category, evaluated in this order:

1. `duplicate_same_class_claimed_gt`: within 5 m of a same-class GT already
   claimed by a baseline match.
2. `cross_class_confusion`: not category 1 and within 5 m of an other-class GT.
3. `same_class_near_outside_match_radius`: nearest same-class GT is greater than
   5 m and at most 10 m away.
4. `no_plausible_nearby_gt`: no GT of either class is within 10 m.
5. `other_nearby_gt_geometry`: any remaining unmatched prediction with some GT
   within 10 m.

`empty_scene_fp` is an additional flag when the frame has no eligible GT.

For temporal behavior, a false positive is `persistent` when a same-class FP in
the same family/quantizer/source-prefix lies within 3 m in either the immediately
preceding or following retained collection index; otherwise it is `single_frame`.
Because the evaluator samples are sparse, this is a conservative persistence
proxy and not a deployed tracker claim.

## 5. Bounded validation-only candidates

All candidates consume predicted fields only. The preregistered single-change set
is intentionally small:

- `baseline`: current retained predictions, class thresholds 0.20/0.20.
- `world_nms_1m`, `world_nms_2m`, `world_nms_3m`: descending-score,
  class-aware predicted-world XY suppression.
- `image_nms_6px`, `image_nms_8px`: descending-score, class-aware Chebyshev
  suppression using the retained predicted 2D centers. These are incremental
  post-decoder checks, not claims of re-running raw-heatmap NMS.
- `veh_thr_0p225`, `veh_thr_0p25`: vehicle threshold 0.225 or 0.25 with person
  threshold fixed at 0.20.

One combination may be evaluated only if the best world-NMS candidate and the
best vehicle-threshold candidate each independently improve validation vehicle
F1 over baseline. That combination uses exactly those two selected components;
no further radii or thresholds will be searched.

Selection rule: maximize pooled validation vehicle F1 across complete profiles;
ties within `1e-12` are broken by higher pedestrian F1, then lower FP/frame, then
lower incremental latency. The complete validation Pareto table is retained so
precision/recall trade-offs remain visible; the rule is not a deployment-safety
threshold.

## 6. Metrics and paired comparison

For every candidate, family, quantizer, split, class, density group, and distance
stratum, compute detection counts, precision, recall, F1, FP/frame, XY MAE, and
XY RMSE. Also report all-class totals, prediction/GT counts, empty-scene behavior,
and dense-scene (`density_bin == 5+`) behavior. GT distance strata are `0-10`,
`10-20`, `20-30`, and `30-40` m using recorded camera XY; an `unknown` row is
retained if distance cannot be reconstructed.

Uncertainty is frame-grouped: paired deltas use a fixed-seed (`20260819`) bootstrap
over unique `sample_id` values, carrying all three quantizer rows for a sampled
frame together. Report percentile 95% intervals from 2,000 replicates. Quantized
evaluations are profiles, not independent scenes.

Segmentation IoU is reproduced from the existing per-frame confusion counts only
and is not attributed to the decoder change.

## 7. Runtime latency

Because persisted artifacts contain retained object lists rather than raw heatmap
tensors, measure the causal deployable list post-processing stage in isolation.
For every profile/frame pair, time baseline list filtering and the selected
predicted-only correction in paired alternating order after warm-up, using
`time.perf_counter_ns`. Repeat each side 30 times, subtract matched loop overhead,
and report p50/p90/p95/max plus paired delta. This is incremental post-processing
latency, not end-to-end GPU head/decode latency; that missing measurement will be
called out explicitly.

## 8. Installed-map implication

Inspect the production map server read-only. If its same-stream installation
function can be invoked safely with no network/CARLA side effects, replay baseline
and selected prediction lists through an isolated import/harness. Otherwise build
no behavioral surrogate: report raw-count and predicted duplicate-count deltas,
pin the exact server code path showing the same-stream behavior, and mark actual
installed-map precision/recall as unverified. The production server will not be
edited.

## 9. Decision rule and retraining boundary

- `POSTPROCESSING_SUFFICIENT`: the validation-selected configuration yields a
  frozen-test Pareto improvement material enough to explain/reduce the dominant
  FP mechanism, with localization and measured incremental latency reported and
  no evidence that an object-head change is needed.
- `RETRAINING_PILOT_JUSTIFIED`: bounded predicted-only candidates fail to provide
  a useful frozen-test precision/FP improvement without clearly damaging recall,
  XY localization, or runtime. The report may propose AE64 as a one-family pilot,
  but will not start training.
- `INSUFFICIENT_EVIDENCE`: missing raw outputs, original validation provenance, or
  another limitation prevents a defensible conclusion. This remains eligible even
  if descriptive taxonomy results are strong.

No checkpoint, existing experiment, production decoder, map server, UE controller,
CARLA/OAI path, Phase-2 plan, or policy catalog will be changed.
