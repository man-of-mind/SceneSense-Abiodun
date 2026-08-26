# FOCUSED_NOAE_TRAINING_PLAN (as implemented)

Revision 2 — 2026-08-26. Supersedes revision 1. This document now describes what is
**implemented and running**, not a proposal.

Authority: HEAD `2353ee2`; decoder terminal `FOCUSED_NOAE_TRAINING_REQUIRED`.

Warm start (verified by SHA-256 before launch, read-only, never overwritten):
Stage-2 `curriculum_stage2_joint_v1` **epoch 13**,
`0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30`, at
`experiments/route_b_noae_precision_full_v1/20260825_195301/checkpoints/curriculum_stage2_joint_v1/epoch_013.pt`.

Baseline of record for that checkpoint:
- recall ceiling @ score 0.02 / 3 m: 0.5702 vehicle, 0.4852 person (permissive ceiling, not the
  operating point — the matched re-decode at the production 0.20 threshold is what selection uses);
- segmentation (clean): vehicle IoU 0.8117, person IoU 0.3274, mIoU 0.7078;
- failures concentrate on small objects and 20-40 m; true occlusion unresolved and out of scope.

Preserved unchanged: 768x432 input; LR-ASPP backbone and split feature shapes; the Route B
train/val partition (6600 train / 3588 val rows, reused via the *same* manifest as the stage-2 run);
72-profile payload compatibility; locked test closure — nothing in this work reads `split == "test"`.

Run of record: `experiments/focused_noae_v1/20260826_hybridq`.
Baseline decodes: `experiments/focused_noae_v1/baseline_epoch13`.

---

## 1. Diagnosis (established from the code and the stage-2 run)

**1.1 The old drop sampler did not cover the deployed action set.**
`train_fusion.py` sampled `q = rand() * feature_drop_max` with `feature_drop_max = 0.8`. The 72-action
registry uses six measured anchors {0.00, 0.30, 0.50, 0.70, 0.90, 0.98}. So exact `q = 0.00` had
probability **zero** — and `model.py` gates `_objectness_drop` on `> 0.0`, meaning the clean forward
path used by 12 of the 72 actions was never trained — while `q = 0.90` and `0.98` (another 24 actions)
were outside support entirely. `feature_drop_val` also defaulted to 0.40, which is not an anchor.

**1.2 The heatmap loss was not class-balanced.** `focal_heatmap_loss` summed positives over all class
channels and divided by the *total* positive count, so each class's gradient share equalled its share
of positive cells; on vehicle-dominated Route B that structurally starves person. There was no size or
range term either — every positive weighed exactly 1.0. The config's `class_loss_weights: [0.5,1.0,4.0]`
is the *segmentation* CE weight and never touched the object head. This is the high-leverage term:
center is ~73% of the object loss (3.83 x 4.0 = 15.3 of 20.5).

**1.3 Segmentation weight 0.3 was not the binding constraint on person IoU.** At 0.3, stage-2
segmentation improved monotonically across all 25 epochs (mIoU 0.6860 -> 0.7049, person 0.2887 ->
0.3297, seg_loss 0.3111 -> 0.2698) — converging, not starved. Stage-1's `segmentation: 1.0` is not
counter-evidence: both backbone and classifier are frozen there, so the weight is inert, and stage-1's
apparent seg drift is an artifact of the drop-side validation pass moving with the object head.
The change to 0.6 is therefore **defensive**, not expected to move person IoU — see S2.3.

**1.4 In-training selection could not see the failing metric.** `selection_score_mode: "loc_dim_loss"`
= `-(loc_loss + 0.25*dim_loss)` is a regression proxy on already-matched cells, blind to recall. It
ranked epoch 8 best; epoch 13 — the checkpoint the decoder calibration actually selected — scored near
the bottom of the run.

---

## 2. The recipe as implemented

Trial `focused_noae_v1`, config `object_head_pilot_v1/configs/focused_noae_v1.json`.

### 2.1 Hybrid q sampler (`train_fusion.py`)

Per batch:
- with probability **0.60**, an **exact registered anchor**, uniform over the six
  {0.00, 0.30, 0.50, 0.70, 0.90, 0.98} — 0.10 mass each;
- with probability **0.40**, a **stratified continuous** draw: one of the five open intervals between
  consecutive anchors chosen uniformly (0.08 mass each), then uniform inside it.

Equal mass per stratum rather than width-proportional, so the narrow 0.90-0.98 gap is covered as
densely as the wide 0.00-0.30 one. A continuous draw lands on an anchor with probability zero, so
**explicit exact-anchor exposure is preserved entirely by the 0.60 branch and is never diluted**.
Exact `q = 0.00` short-circuits `_objectness_drop`, so clean batches are real training batches with
true forward *and* backward passes — verified live (S3, C2).

Config keys: `feature_drop_values` (the six anchors) and `feature_drop_anchor_prob: 0.6`.
`feature_drop_max` is retained but inert; with `feature_drop_values` absent, every existing config
reproduces its old behaviour exactly. Per-epoch `drop_hist` (exact anchors) and `drop_cont_hist`
(per stratum) are logged.

`feature_drop_val: 0.9` — the in-training maximin now logs against a real anchor instead of 0.40.
Logging only; it does not promote (S4).

### 2.2 Class-balanced bounded positive weighting (`object_targets.py`)

`focal_heatmap_loss` gained `pos_weight`, `weight_cap`, `class_balanced`, `stats`. With
`pos_weight=None` and `class_balanced=False` it is bit-identical to the original.

Per-positive weight, built inside `multitask_object_loss` from targets already on device — no new
dataloader field, no new target tensor:

```
w = (1 + small_gain * 1[bbox_w*bbox_h < small_area_frac]) * (1 + range_gain * 1[20 m <= r < 40 m])
```

`local_x/local_y` are raw **metres**; `REG_BBOX_WH` are input-image **fractions**
(`small_area_frac = 0.003` ~ a 32x32 px box at 768x432). Registered gains: `small_gain 1.0`,
`range_gain 0.8`, `pos_weight_cap 4.0`.

Three guarantees, all asserted live at launch (S3):

1. **Per-element cap.** `w` is clamped to `[0, 4.0]` before use.
2. **Class balance is explicit.** Weights are renormalised to mean 1.0 **separately within each
   class**, and the positive term is the **macro-average** of the per-class per-positive means — one
   equally-weighted vote per class. Vehicle cell count therefore cannot dominate person learning,
   regardless of the vehicle:person cell ratio in a batch. A class with **zero** positives is skipped
   entirely: never a divide-by-zero, never a fabricated gradient for an absent class. If no class has
   positives the positive term is zero.
3. **Fixed positive-loss budget.** Both the old pooled form and the new macro-average are a "mean
   focal loss per positive cell", and per-class renormalisation makes `sum(w) == count` within each
   class. Reweighting can only *redistribute* gradient, never inflate it — a handful of far or small
   objects cannot take over training.

**The background loss is preserved exactly**, including its original `pos_count` denominator, so the
positive:negative balance is not silently moved.

No change to the regression losses (`loc`, `dim`, `yaw`, `parked`, `radar_support`, `bbox2d`).

### 2.3 Segmentation weight: 0.6, single fixed choice

`segmentation: 0.3 -> 0.6`. No sweep. Per S1.3 this is defensive: at 0.3 segmentation was ~0.4% of the
total loss, and this run adds two forces pushing against it (upweighted object positives, and hard
q = 0.90/0.98 batches where the classifier sees a 2-5% surviving feature grid). 0.6 doubles the share
to ~0.8% — still object-dominated, so it will not stall localization. **No claim is made that it moves
person IoU 0.3274.** Segmentation appears in acceptance only as a guard, never as a target.
`class_loss_weights` and `lovasz_weight` are unchanged.

### 2.4 Head: shared, unchanged

`head_arch: "shared"`, `fuse_low_feature: true`. The decoupled architecture is **not** introduced —
not as an arm, not as a fallback. Epoch 13's trained weights live in `object_head`; selecting
`decoupled` sets `object_head = None` and would silently drop the warm start. No new backbone, FPN, or
input resolution.

### 2.5 Schedule

24 epochs, `early_stop_patience: 24` so a loss-only criterion cannot truncate the run,
`checkpoint_every_epochs: 1`. All other hyperparameters identical to stage 2: `lr 3e-05`, AdamW,
`weight_decay 1e-4`, cosine with 1 warmup epoch, batch 16, `augment_strength: strong`,
`geometric_augment: false`, `freeze_bn: true`, nothing frozen, `ae_bottleneck: 0`.
`training_seed: 20260826`.

### 2.6 Split / payload compatibility

Nothing touches the model graph, feature dict keys, high/low tensor shapes, the AE, the quantiser, or
the zstd-3 envelope. Both code changes are confined to the training objective and the training-time
sampling of `feature_drop_fraction`, an argument that already existed on `forward()`. The result drops
into the 72-profile registry with the same wire contract as epoch 13 and, unlike epoch 13, has trained
support at all six anchors *and* between them.

---

## 3. Launch check (one training batch; ran before launch, PASS)

`focused_noae_launch_check_v1.py` — a launch gate, not a test suite. One real batch, two anchors
(q = 0.00 and the degraded q = 0.90):

| # | Check | Result |
|---|-------|--------|
| C1 | loss finite at both anchors | PASS |
| C2 | shared object head receives nonzero, finite gradient at both (the 59f031a AMP zero-grad regression) | PASS — grad norm 9.71 @ q=0.00, 10.13 @ q=0.90 |
| C3 | positive mass class-balanced: per-class mean weight 1.0, per-class means computed separately | PASS — `pos_mean_w_class0 = pos_mean_w_class1 = 1.000`, 2 classes present |
| C4 | no positive weight above the 4.0 cap | PASS — max 1.65 |

The sampler's 60/40 mass split was separately verified as a pure function (200k draws: exact mass
0.5986, continuous 0.4014, 0.0992-0.1001 per anchor, 0.0796-0.0809 per stratum, and every
exact-anchor landing traced to the anchor branch).

Gradients are measured **unscaled**: the check backwards without `GradScaler`, since the initial
scale of 65536 overflows fp16 intermediates to `inf` and would make "nonzero" meaningless. The
autocast path that carried the 59f031a bug (`cache_enabled=False`) is still exercised.

---

## 4. Checkpoint selection — decoded validation service metrics only

`loc_dim_loss` and every in-training `selection_score` are logged, never promoting. `best.pt` is not
authoritative for this run.

Baseline (instruction 9): epoch 13 is re-decoded through the **identical** contract used for every
candidate — `evaluate_route_b_checkpoint_v1.py`, fixed decoder score 0.20 / top-k 120 / NMS 2 px /
3.0 m match / 40 m GT eligibility, `split=val` — at all six anchors.

Candidate scoring (instruction 10 — no 24x6 full-inference sweep). The evaluator has **no** feature
cache and **no** offline rank-drop replay, and none was built. Instead:
1. shortlist on the existing decoded clean (q = 0.00) metrics;
2. score only the shortlist at all six anchors **and** the five unseen interval midpoints
   {0.15, 0.40, 0.60, 0.80, 0.94}, which the hybrid sampler reaches only through its continuous branch
   and never as an exact draw — this is what shows whether the model interpolates or only memorises.

A minimal `--feature-drop-fraction` passthrough was added to `evaluate_fusion` (one argument into the
existing `model.forward(feature_drop_fraction=...)`). At q = 0.0 it is a structural no-op, so the
clean decode path is unchanged.

### Registered selection rule (fixed before any candidate was scored)

**Eligibility** — clean q = 0.00 non-regression vs the matched epoch-13 baseline, all required:
`vehicle_f1` and `person_f1` >= baseline - 0.005; `overall_xy_mae_m` <= baseline + 0.10 m;
`overall_dimension_mae_m` <= baseline + 0.05 m; `mIoU`, `vehicle_iou`, `person_iou` >= baseline - 0.005.

**Improvement** — clean q = 0.00 `mean(vehicle_f1, person_f1)` >= baseline + 0.010.

**Ranking** among candidates passing both, in strict order:
1. maximize the **worst-anchor** `mean(vehicle_f1, person_f1)` over the six registered anchors;
2. minimize mean `overall_xy_mae_m` across those anchors;
3. minimize mean duplicate FP per frame across those anchors.

Midpoints are reported, never ranked. Every anchor is reported separately.

**Advisory, visible, never gating:** clean vehicle recall 0.60, clean person recall 0.50,
clean mean F1 0.60.

---

## 5. Stop rule

If no candidate is both eligible and clearly improving — that is, none beats the matched epoch-13
baseline without material localization, segmentation, or dimension regression — the run terminates as
**`FOCUSED_NOAE_TRAINING_FAILED`**. In that case **no AE variant is trained**. There is no decoupled-head
fallback and no hyperparameter sweep.

If it passes: report the selected checkpoint, its SHA-256, all six-anchor metrics (plus midpoints) and
runtime immediately; then prepare the same bounded correction for AE64 / AE32 / AE128, introducing no
new architecture choices.

---

## 6. Change inventory (complete)

| File | Change |
|------|--------|
| `pole_lraspp_multimodal_fusion/train_fusion.py` | hybrid anchor/stratified-continuous q sampler + `drop_hist` / `drop_cont_hist` logging; the `feature_drop_max` path is preserved |
| `pole_lraspp_multimodal_fusion/object_targets.py` | `pos_weight` / `weight_cap` / `class_balanced` / `stats` on `focal_heatmap_loss`; per-class renormalised weight map and macro-average in `multitask_object_loss` |
| `pole_lraspp_multimodal_fusion/evaluate_fusion.py` | `--feature-drop-fraction` passthrough (no-op at 0.0) |
| `object_head_pilot_v1/evaluate_route_b_checkpoint_v1.py` | `--feature-drop-fraction` passthrough; **bug fix**: `--config` is now resolved absolute (the subprocess runs with `cwd=PKG_ROOT`, so a relative path failed) |
| `object_head_pilot_v1/configs/focused_noae_v1.json` | new |
| `object_head_pilot_v1/focused_noae_launch_check_v1.py` | new — one-batch launch gate |
| `object_head_pilot_v1/focused_noae_decode_sweep_v1.py` | new — create-only anchor/midpoint decode sweep |
| `object_head_pilot_v1/focused_noae_select_v1.py` | new — decoded-metric selection |

Backward compatible: with `feature_drop_values` absent and `pos_weight_enable` false, every existing
config reproduces today's behaviour exactly. No changes to the model graph, the dataset view builder,
the decoder thresholds, the split runtime, the codec, or anything under `test`.
