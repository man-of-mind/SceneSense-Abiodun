# Continuous ROI-Drop Control Audit -- Frozen Plan

Status: frozen before any offline inference  
Audit ID: `continuous_roi_control_audit_v1/20260820_203411`  
Scope: isolated, offline analysis; no registry, runtime, launcher, controller,
map-server, CARLA, OAI, checkpoint, existing-experiment, or RL-agent mutation

## 1. Question and hypotheses

Question: within each fixed `{model family, quantizer}` branch, is rank-based ROI
drop `q` regular enough to expose as a bounded continuous action, or must the UE
controller retain measured discrete q anchors?

Hypotheses:

1. Payload should usually decrease with q, but entropy coding may introduce small
   local non-monotonicities.
2. Object quality need not be monotonic. A continuously controlled q is defensible
   only if local changes and interpolation errors are bounded and reproducible.
3. Branch behavior is not assumed separable: all 12 family/quantizer branches are
   evaluated and decided separately.
4. The float input is structurally piecewise. Production drops
   `round(q*N)` lowest-ranked cells, so neighboring q values are identical until
   one of the discrete cell counts changes.

## 2. Frozen q grid and structural resolution

The inference grid is:

`{0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.98}`.

The 17 points from 0 through 0.8 cover the checkpoint's training support at 0.05
spacing. That produces eight preregistered midpoint holdouts (`.05, .15, ...,
.75`) between 0.10-spaced fit points (`0, .10, ...,.80`), and it inserts several
measurements into each gap between the current anchors. The 0.90 and 0.98 points
are descriptive measured extrapolation references only; they cannot establish
continuous validity outside training support. `q=1` is forbidden.

The registered native feature shapes are low `[1,40,54,96]` and high
`[1,960,27,48]`, hence `N_low=5,184` and `N_high=1,296`. The gate is applied to
native low/high features before the integrated AE. The exact action is
`k_low=round(q*5184)` and `k_high=round(q*1296)` using Python's tie-to-even
rounding. The audit will enumerate counts, transition points, plateau widths,
and the actual dropped fractions. A 0.05 grid step changes about 259 low cells
and 65 high cells: dense enough to test controller-scale macro smoothness, but
not to claim cell-by-cell smoothness. Structural enumeration covers the finer
piecewise action independently of inference.

## 3. Branches and evidence

Branches are the Cartesian product of:

- families: `noae`, `ae32`, `ae64`, `ae128`
- quantizers: `uint8`, `uint6`, `uint4`

No family or quantizer is pooled for a branch decision. Repeated q rows for one
frame are paired outcomes, not independent scenes. The current 72 profiles at
`{0,.3,.5,.7,.9,.98}` are provenance/reproduction evidence and may be analyzed
without inference. New q points require the frozen checkpoints and original
aligned dataset (or an equivalently hashed immutable cache).

## 4. Frozen validation/test roles

The evaluator's recorded source is a 2,162-identifier model-test split; no
original model-development validation predictions are preserved. Before dense-q
inference, create a separate audit partition from those identifiers:

1. Parse source prefix and monotonically increasing collection index from
   `sample_id`.
2. Group consecutive collection indices into blocks of 25 within each prefix.
3. Hash `continuous-roi-v1|<prefix>|<block>` with SHA-256.
4. Assign hash modulo 5 values 0 or 1 to `audit_validation`; assign 2, 3, or 4
   to frozen `audit_test`.
5. Assert identifier and block disjointness. Every q, family, and quantizer for
   an identifier remains in the same split.

Validation is used to inspect the preregistered diagnostics and verify tooling;
it does not alter the grid or thresholds below. The test partition is read once
after validation outputs are complete. The report must call this an audit
holdout of prior test evidence, not the original training validation split.

Interpolation protocols:

- **Dense midpoint test:** linearly interpolate aggregate metrics from
  `{0,.10,.20,...,.80}` and score held-out `{.05,.15,...,.75}`.
- **Current-anchor test:** within the convex hull `[0,.70]`, linearly interpolate
  from current in-distribution anchors `{0,.30,.50,.70}` and score all new grid
  points not used as anchors. No extrapolation above `.70` counts as support.

## 5. Metrics and comparisons

For every branch, q, and split, report:

- zstd-3 feature payload bytes: mean, median, p90, p95, maximum;
- vehicle/person TP, FP, FN, precision, recall, F1, and FP/frame;
- matched world-XY MAE and RMSE;
- prediction and GT counts;
- secondary background/vehicle/person IoU and mIoU from summed confusion counts;
- empty- and dense-scene strata when complete frame evidence is available.

Tests:

1. Aggregate and frame-paired payload monotonicity across adjacent q values.
2. Adjacent-q paired changes for detection/localization and segmentation.
3. Local slopes and second finite differences on the 0.05 grid.
4. Dense-midpoint and current-anchor interpolation errors.
5. Crossings in family/quantizer payload-quality ordering, reported without
   assuming factor separability.
6. The `.90/.98` interval is labeled extrapolation and excluded from the primary
   continuous-control decision.

GT is used only for offline scoring. Same-frame GT, tail outputs, post-tail
confidence, and segmentation/object results are forbidden as deployable q-policy
inputs.

## 6. Paired uncertainty

Use 2,000 fixed-seed (`20260820`) paired block-bootstrap replicates. Resample the
trajectory blocks defined above, carry every frame in a selected block, and keep
all q values for that frame together. Compute percentile 95% intervals for
adjacent-q deltas and interpolation errors. Analyze each family/quantizer branch
separately. If an across-branch summary is shown, resample the same frame blocks
and keep all 12 branches together; never treat q or quantizer repeats as new
independent frames.

## 7. Latency

Measure with warm-up and paired order, reporting p50/p90/p95/max:

1. objectness-map plus rank construction;
2. marginal q mask/select/apply with ranking cached;
3. full production-equivalent q gate;
4. integrated-AE encode/decode where applicable;
5. quantization plus serialization plus zstd-3 compression;
6. combined gate-to-serialized-payload latency.

Use CUDA events with synchronization for GPU stages and `perf_counter_ns` for CPU
serialization. Run at least 30 repetitions on a fixed 200-frame validation subset
after 20 warm-up repetitions. Record the host, device, torch version, and input
prediction/feature shapes. Cached-rank marginal latency must not be mislabeled as
the production full-gate latency.

## 8. Frozen decision criteria

Apply these per branch on the frozen audit test over `[0,.8]`:

`CONTINUOUS_SUPPORTED` requires complete dense-q evidence plus all of:

- aggregate payload is non-increasing at every adjacent step; at least 99% of
  paired frames are non-increasing or tied at every step;
- dense-midpoint maximum payload relative error <=5% and median <=2%;
- dense-midpoint and current-anchor maximum absolute errors are <=0.025 for
  vehicle/person precision or recall, <=0.15 m for XY MAE/RMSE, and <=0.10 for
  FP/frame;
- no adjacent 0.05 step has an absolute aggregate jump exceeding 0.05 recall,
  0.075 precision, 0.25 m XY MAE/RMSE, or 0.25 FP/frame;
- the bootstrap interval does not demonstrate a violation of those limits.

Segmentation is secondary: report interpolation errors and jumps, with 0.015 IoU
as the diagnostic tolerance. It does not independently veto a branch unless an
adjacent step loses more than 0.05 IoU, which indicates a broad perception
failure rather than a reward preference.

`DISCRETE_ONLY` requires a complete dense-q run and at least one primary failure
whose paired 95% interval confirms the violation. Branch-level outcomes are
reported, but the existing controller remains globally discrete if any of the 12
branches is `DISCRETE_ONLY`; no unsupported branch may inherit another branch's
smoothness.

`INSUFFICIENT_EVIDENCE` applies when dense q outputs are missing, a branch is
incomplete, validation/test provenance cannot be enforced, timing is missing, or
uncertainty is inconclusive. Existing six-anchor curves alone can diagnose gross
behavior but cannot earn `CONTINUOUS_SUPPORTED` or `DISCRETE_ONLY` under this plan.

The overall terminal is `CONTINUOUS_SUPPORTED` only for 12/12 passing branches,
`DISCRETE_ONLY` when complete evidence confirms any branch failure, and otherwise
`INSUFFICIENT_EVIDENCE`.

## 9. Compute budget and stop rule

Full scope is 19 q points x 12 branches x 2,162 frames = 492,936 profile-frame
evaluations, while backbone encoding and objectness ranking should be reused per
family/frame. Budget: at most 6 GPU-hours, 8 wall-clock hours, 100 GB of new
artifacts, and no CPU full-run fallback.

Preflight stop before inference if any required dataset file/checkpoint/hash is
missing, CUDA is unavailable, or the immutable output directory cannot be
created. After preflight, run a 32-frame `ae64/uint6` validation smoke across all
19 q points; stop if count reconciliation fails, NaNs appear, q semantics differ,
or projected full runtime exceeds 6 GPU-hours. During full inference, stop on any
branch/profile incompleteness or projected storage above 100 GB. Preserve partial
outputs and terminate `INSUFFICIENT_EVIDENCE`; do not relax the grid or silently
fall back to CPU.

## 10. Controller interpretation

If supported, the eventual action remains hybrid: one categorical choice among
12 `{family, quantizer}` branches plus bounded continuous `q in [0,.8]`. Standard
SAC and TD3 assume continuous actions and do not directly solve that mixed action.
A future design would need an explicit hierarchical policy, parameterized-action
critic, or categorical branch policy plus conditional q actor. This audit will not
implement or train an agent and will not add a reward for choosing any particular
q.
