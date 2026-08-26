# FOCUSED_NOAE_TRAINING_PLAN

Status: plan only. Nothing in this document has been implemented, run, or measured.
Authority for this plan: HEAD `2353ee2`; decoder terminal `FOCUSED_NOAE_TRAINING_REQUIRED`.

Warm start (fixed): Stage-2 `curriculum_stage2_joint_v1` **epoch 13**,
SHA-256 `0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30`, at
`experiments/route_b_noae_precision_full_v1/20260825_195301/checkpoints/curriculum_stage2_joint_v1/epoch_013.pt`.

Baseline of record for this checkpoint:
- decoded recall ceiling @ score 0.02 / 3 m match: **0.5702 vehicle**, **0.4852 person**;
- segmentation (clean): **vehicle IoU 0.8117**, **person IoU 0.3274**, **mIoU 0.7078**;
- failures concentrate on small objects and the 20-40 m band; true occlusion unresolved and out of scope here.

Preserved without change: 768x432 input; LR-ASPP backbone and split feature shapes; the current Route B
train/val/test partition (3293 train rows / 3600 val rows, `--expected-train-rows 3293 --expected-val-rows 3600`);
72-profile payload compatibility; locked test closure (no test episode is read by anything in this plan).

---

## 1. Diagnosis (from the code and the existing run, not from new measurement)

### 1.1 Feature-drop sampling is misaligned with the measured action set

`train_fusion.py:836`:

```python
q_drop = float(torch.rand(1).item()) * feature_drop_max if feature_drop_max > 0.0 else 0.0
```

with `feature_drop_max: 0.8` in `configs/curriculum_stage2_joint_v1.json`. So training support is
`q ~ Uniform(0, 0.8)`, one scalar per batch.

The registry this model must serve (`rl_agent/UE_SPLIT_ONLY_EXPERIMENT_PLAN_V2.md:162-168`) is
4 model families x 3 quantizers x **6 measured q anchors {0.00, 0.30, 0.50, 0.70, 0.90, 0.98}** = 72 actions.

Consequences:
- **q = 0.00 has probability zero.** `Uniform(0, 0.8)` never returns exact 0, and `model.py:237` gates on
  `float(feature_drop_fraction) > 0.0`, so the clean forward path — the path used by 12 of the 72 actions —
  is *never* exercised during training. Clean quality is currently an extrapolation, not a trained condition.
- **q = 0.90 and q = 0.98 are outside support entirely.** Those are 24 of the 72 actions. The model has
  never seen them, and no amount of the current schedule reaches them.
- The four in-support anchors get uneven, incidental mass (`P(q>=0.70) = 12.5%`), not by design.
- `feature_drop_val` defaults to `feature_drop_max * 0.5 = 0.40` (`train_fusion.py:713`), which is **not a
  measured anchor**, so the logged drop-side validation number describes a condition that is never deployed.

### 1.2 The heatmap loss is not class-balanced, and has no size or range term

`object_targets.py:277-284`:

```python
pos_count = pos.sum().clamp(min=1.0)
return (pos_loss.sum() + neg_loss.sum()) / pos_count
```

`pos_loss` is summed over **all** heatmap channels and divided by the **total** positive count across both
classes. There is no per-class weight, no per-object size weight, and no range weight. Each positive cell
carries weight exactly 1.0.

So each class's share of the center gradient equals its share of positive cells. On Route B that is
vehicle-dominated, and person — the class at IoU 0.3274 and recall 0.4852 — is structurally underweighted
in proportion to how rare it is. The same holds for small and far objects: a 20-40 m pedestrian contributes
exactly as much as a 5 m vehicle.

Note `class_loss_weights: [0.5, 1.0, 4.0]` in the config is the **segmentation** cross-entropy weight
(`train_fusion.py:358`). It does not touch the object head at all. There is currently no class balancing
anywhere in the object path.

This is the high-leverage term. At epoch 13 the object loss decomposes as
`center 3.83 x 4.0 = 15.3` out of `object_loss = 20.5` — the center/heatmap term is **~73%** of the object
objective. Reweighting it acts directly on the dominant gradient.

### 1.3 Segmentation weight 0.3: not the binding constraint, but it will become one

Stage-2 ran at `segmentation: 0.3` and segmentation still improved monotonically across all 25 epochs:
mIoU 0.6860 -> 0.7049, person IoU 0.2887 -> 0.3297, `seg_loss` 0.3111 -> 0.2698. That is not a starved
objective; that is an objective converging to its own plateau (person IoU peaks at epoch 15, 0.3318, then
flattens). **0.3 is not demonstrably too low for the current recipe**, and I do not claim raising it will
move person IoU 0.3274.

Stage-1's `segmentation: 1.0` is not counter-evidence: stage 1 has `freeze_backbone: true` and
`freeze_classifier: true`, so the segmentation loss has no gradient path and the weight is inert. Stage-1's
apparent seg drift (mIoU 0.5563 -> 0.5366) is an artifact of the drop-side validation pass — the drop mask
is ranked by the object head's own objectness (`model.py:194-204`), so as the object head trains, the mask
moves and seg@drop moves with it. There is no live seg-weight evidence at 1.0.

The reason to change it is **defensive**. At the current weights, segmentation is
`0.3 x 0.270 = 0.081` against `1.0 x 20.5` object — about **0.4% of total loss**. This run adds two forces
pushing against segmentation: upweighted object positives (S1.2) and hard q = 0.90 / 0.98 batches, where
the classifier sees a 2-5% surviving feature grid. Holding 0.7078 mIoU / 0.8117 vehicle IoU through that
needs more than 0.4%.

### 1.4 Checkpoint selection cannot see the failing metric

`selection_score_mode: "loc_dim_loss"` -> `-(loc_loss + 0.25 * dim_loss)` (`train_fusion.py:439-440`).
This is a regression proxy on *already-matched* cells. It is blind to detection recall, which is the entire
failure. It also drives the maximin `min{clean, drop}` reduction at `train_fusion.py:880`, so both arms of
the robustness selection are computed on a recall-blind scalar.

Confirmation that this proxy is actively wrong for the goal: it ranks epoch 8 best (-3.0494); epoch 13,
the checkpoint the decoder calibration actually selected, scores -3.1114 — near the *bottom* of the run.

---

## 2. The recipe (one primary; no alternatives)

New trial name: **`focused_noae_v1`**. New config `configs/focused_noae_v1.json`, cloned from
`configs/curriculum_stage2_joint_v1.json` with the deltas in S2.5. Warm start is epoch 13 for
`init_rgb_checkpoint` and `init_object_checkpoint`.

### 2.1 Code change A — anchor-exact drop curriculum

**File:** `pole_lraspp_multimodal_fusion/train_fusion.py`

Replace the sampler at line 836. Read the schedule near line 709 alongside `feature_drop_max`:

```python
# ~line 709, next to feature_drop_max
_dv = trial.get("feature_drop_values", train_cfg.get("feature_drop_values"))
_dp = trial.get("feature_drop_probs",  train_cfg.get("feature_drop_probs"))
feature_drop_values = [float(v) for v in _dv] if _dv else None
feature_drop_probs  = [float(p) for p in _dp] if _dp else None
if feature_drop_values is not None:
    if feature_drop_probs is None or len(feature_drop_probs) != len(feature_drop_values):
        raise ValueError("feature_drop_probs must be present and match feature_drop_values in length")
    _s = float(sum(feature_drop_probs))
    if abs(_s - 1.0) > 1e-6:
        raise ValueError(f"feature_drop_probs must sum to 1.0, got {_s}")
    _drop_cdf = torch.tensor(feature_drop_probs, dtype=torch.float64).cumsum(0)
    _drop_vals = torch.tensor(feature_drop_values, dtype=torch.float64)
    drop_hist = {v: 0 for v in feature_drop_values}     # first-epoch gate G1
```

```python
# replaces line 836
if feature_drop_values is not None:
    _u = torch.rand(1, dtype=torch.float64)
    _i = int(torch.searchsorted(_drop_cdf, _u).clamp(max=len(_drop_vals) - 1).item())
    q_drop = float(_drop_vals[_i])
    drop_hist[q_drop] += 1
elif feature_drop_max > 0.0:
    q_drop = float(torch.rand(1).item()) * feature_drop_max
else:
    q_drop = 0.0
```

Log `drop_hist` per epoch (supervisor line + a `drop_hist_json` column) and reset it at the top of each
epoch. `torch.rand` is retained so `training_seed` reproducibility is unchanged. The `feature_drop_max`
branch is kept intact so every existing config keeps its current behaviour byte-for-byte.

**Schedule (registered):**

| q | 0.00 | 0.30 | 0.50 | 0.70 | 0.90 | 0.98 |
|---|------|------|------|------|------|------|
| p | 0.30 | 0.14 | 0.14 | 0.14 | 0.14 | 0.14 |

Rationale. Support is now *exactly* the six measured anchors and nothing else — no mass is spent on q
values no action ever requests. Clean gets 0.30 rather than its 1/6 registry share (12 of 72 actions) so
clean quality is not sacrificed to buy robustness; it is the condition every acceptance guard in S5 is
measured under. The five drop anchors split the remaining 0.70 evenly because there is no prior ranking
them. At 206 batches/epoch (3293 rows, batch 16, `drop_last=False`) that is ~62 clean and ~29 batches per
drop anchor per epoch, ~700 batches per anchor over the run.

Exact q = 0.00 also short-circuits `model.py:237`, skipping the extra `torch.no_grad()` object-head forward
inside `_objectness_drop`, so 30% of batches are slightly cheaper than today.

At q = 0.98, `k = round(0.98 * n)`: only ~2% of feature cells survive. If `high` is stride-16 (48x27 = 1296
cells at 768x432), that is ~26 surviving cells. This is a real but very sparse regime — gate G4 exists to
confirm it produces finite losses rather than NaN.

Also set `feature_drop_val: 0.98` (see S2.5): the logged drop-side validation becomes the *hardest measured
anchor* instead of the non-anchor 0.40, so the in-training maximin brackets the deployed envelope at its
two endpoints. This is for logging and monitoring only — it does not promote (S2.4).

### 2.2 Code change B — class / small-object / range positive weighting

**File:** `pole_lraspp_multimodal_fusion/object_targets.py`

Add an optional positive-weight map to `focal_heatmap_loss` (line 277):

```python
def focal_heatmap_loss(logits, target, *, alpha=2.0, beta=4.0,
                       pos_weight=None, weight_cap=4.0):
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = target.ge(1.0 - 1e-3).to(logits.dtype)
    neg = (1.0 - pos).to(logits.dtype)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * torch.pow(1.0 - target, beta) * neg
    pos_count = pos.sum().clamp(min=1.0)
    if pos_weight is not None:
        w = pos_weight.to(logits.dtype).clamp(min=0.0, max=float(weight_cap))
        # Renormalise to mean 1 over positives: total positive mass is INVARIANT.
        # Reweighting can only redistribute the existing budget, never inflate it,
        # and the negative term is left completely untouched.
        scale = pos_count / (w * pos).sum().clamp(min=1e-6)
        pos_loss = pos_loss * w * scale
    return (pos_loss.sum() + neg_loss.sum()) / pos_count
```

Build the map inside `multitask_object_loss` (line 287) from targets that are already on device — no new
dataloader field, no new target tensor, no change to `build_object_targets`:

```python
# after reg_target / reg_mask are resolved, before center_loss
pos_w = None
if weights.get("pos_weight_enable", False):
    cw = weights.get("class_pos_weights", None)          # e.g. [1.0, 2.5]  (vehicle, person)
    cw_t = (torch.tensor(cw, dtype=reg_target.dtype, device=reg_target.device)
              .view(1, -1, 1, 1) if cw else
            torch.ones(1, heatmap_channels, 1, 1, dtype=reg_target.dtype, device=reg_target.device))
    if cw_t.shape[1] != heatmap_channels:
        raise ValueError(f"class_pos_weights has {cw_t.shape[1]} entries, heatmap_channels={heatmap_channels}")
    m = reg_mask                                          # (B,1,H,W), 1 at positive cells
    if has_bbox2d:
        gw = reg_target[:, REG_BBOX_WH.start:     REG_BBOX_WH.start + 1].clamp(min=0.0)
        gh = reg_target[:, REG_BBOX_WH.start + 1: REG_BBOX_WH.start + 2].clamp(min=0.0)
        small = ((gw * gh) < float(weights.get("small_area_frac", 0.003))).to(reg_target.dtype) * m
    else:
        small = torch.zeros_like(m)
    r = torch.linalg.vector_norm(reg_target[:, 0:2], dim=1, keepdim=True)   # local_x/y, metres
    lo = float(weights.get("range_band_lo_m", 20.0)); hi = float(weights.get("range_band_hi_m", 40.0))
    band = ((r >= lo) & (r < hi)).to(reg_target.dtype) * m
    pos_w = (cw_t
             * (1.0 + float(weights.get("small_gain", 1.0)) * small)
             * (1.0 + float(weights.get("range_gain", 0.8)) * band))

center_loss = focal_heatmap_loss(center_logits, heatmap, pos_weight=pos_w,
                                 weight_cap=float(weights.get("pos_weight_cap", 4.0)))
```

Then export the invariant for gate G3:
`parts["pos_weight_mean"] = float(((pos_w * reg_mask).sum() / reg_mask.sum().clamp(min=1.0)).item())`
when `pos_w is not None`.

Units are correct as written: `local_x/local_y/local_z` are written to the regression target in raw
**metres** (`object_targets.py:227-229`), and `bbox_w/bbox_h` are stored as **input-image fractions**
(`object_targets.py:240-241`). `small_area_frac = 0.003` corresponds to roughly a 32x32 px box at 768x432
(`(32/768) * (32/432) = 0.00309`).

**Registered weights** — vehicle 1.0, person 2.5, `small_gain` 1.0, `range_gain` 0.8, `pos_weight_cap` 4.0.
Uncapped worst case is `2.5 x 2.0 x 1.8 = 9.0`; the cap holds any single positive at <= 4x the base.

**Why this cannot let a few far objects dominate** (the explicit requirement): two independent caps.
1. *Per-element*: `weight_cap = 4.0` bounds any single positive's multiplier.
2. *Aggregate*: the mean-1 renormalisation makes `sum(w * pos) == pos.sum()` exactly, so the **total**
   positive loss mass is unchanged. Reweighting is a pure redistribution of a fixed budget. A batch with
   one far pedestrian and forty near vehicles cannot inflate the loss — the pedestrian only takes a larger
   share. The negative/background term keeps its original `pos_count` denominator and is untouched, so the
   positive:negative balance, and therefore the precision/recall operating point of the loss itself, is not
   silently shifted. With all weights at 1.0 the function is **bit-identical** to today's, which is what
   makes gate G3 a meaningful check.

No change to the regression losses (`loc`, `dim`, `yaw`, `parked`, `radar_support`, `bbox2d`). Their
`denom = reg_mask.sum()` normalisation stays as-is. One change-set on one term.

### 2.3 Segmentation weight

`segmentation: 0.3` -> **`0.6`**.

Justification, stated for what it is. This doubles segmentation's share of the total loss from ~0.4% to
~0.8% — still overwhelmingly object-dominated, so it will not stall localisation, and it is far from
stage-1's inert 1.0. The value is chosen to *hold* segmentation at its current level against the two new
sources of pressure (S1.3), not to improve it. Per S1.3, segmentation was already converging at 0.3, and I
am **not** claiming 0.6 moves person IoU 0.3274. Segmentation appears in S5 only as a **guard** — a
condition the run must not break — never as a target it must beat. `class_loss_weights: [0.5, 1.0, 4.0]`
and `lovasz_weight: 0.5` are unchanged.

### 2.4 Head design — unchanged for this run

Keep `head_arch: "shared"` and `fuse_low_feature: true`.

The mandate is a warm start from epoch 13. Epoch 13 was trained with `head_arch: "shared"`, so its trained
weights live in `self.object_head`. Selecting `decoupled` builds `heatmap_head` / `reg_head` and sets
`self.object_head = None` (`model.py:81-88`), so the warm start would silently fail to load the one module
this run most depends on. Changing it now would also confound the head change with the curriculum and loss
changes in the same run. Decoupled is the **single fallback** (S7), not a parallel arm. No new backbone,
no FPN, no input-resolution change is proposed anywhere in this plan.

### 2.5 Config `configs/focused_noae_v1.json`

Clone `curriculum_stage2_joint_v1.json`; change exactly these keys:

```json
{
  "name": "focused_noae_v1",
  "epochs": 24,
  "early_stop_patience": 24,
  "init_rgb_checkpoint":    ".../checkpoints/curriculum_stage2_joint_v1/epoch_013.pt",
  "init_object_checkpoint": ".../checkpoints/curriculum_stage2_joint_v1/epoch_013.pt",
  "feature_drop_values": [0.00, 0.30, 0.50, 0.70, 0.90, 0.98],
  "feature_drop_probs":  [0.30, 0.14, 0.14, 0.14, 0.14, 0.14],
  "feature_drop_val": 0.98,
  "selection_score_mode": "loc_dim_loss",
  "checkpoint_every_epochs": 1,
  "training_seed": 20260826,
  "loss_weights": {
    "object_total": 1.0,
    "segmentation": 0.6,
    "object": {
      "center": 4.0, "location": 1.5, "dimensions": 0.6, "yaw": 0.3,
      "parked": 0.2, "radar_support": 0.1, "bbox2d": 1.0,
      "pos_weight_enable": true,
      "class_pos_weights": [1.0, 2.5],
      "small_gain": 1.0, "small_area_frac": 0.003,
      "range_gain": 0.8, "range_band_lo_m": 20.0, "range_band_hi_m": 40.0,
      "pos_weight_cap": 4.0
    }
  }
}
```

Deliberately unchanged: `lr: 3e-05`, `weight_decay`, `augment_strength: "strong"`, `geometric_augment: false`,
`input_size: [768, 432]`, `batch_size: 16`, all freeze flags, `freeze_bn: true`, cosine schedule with
`lr_warmup_epochs: 1`, `object_heads` block in full (`heatmap_radius_px: 4`, `head_arch: "shared"`,
`head_depth: 3`, `predict_bbox2d: true`, `adaptive_heatmap_radius: true`, `max_gt_distance_m: 40`),
`ae_bottleneck: 0`, and `feature_drop_max: 0.8` (now inert, retained so the config diff is honest about
what superseded it).

`class_pos_weights` order **must** match `object_heads.object_classes: [vehicle, person]`. The
`cw_t.shape[1] != heatmap_channels` check in S2.2 fails loudly if it does not.

`selection_score_mode` stays `loc_dim_loss` **for logging only**. Per S1.4 it cannot promote a model; the
promotion mechanism is S4. `early_stop_patience` is set equal to `epochs` so nothing early-stops on the
proxy, and the `best.pt` that `train_fusion.py` writes is **not authoritative** for this run.

### 2.6 Split / payload compatibility

Nothing in S2.1-S2.5 touches the model graph, the backbone, the feature dictionary keys, the `high`/`low`
tensor shapes, the AE (`ae_bottleneck: 0`), the quantiser, or the zstd-3 envelope. Both code changes are
confined to the training objective and the training-time sampling of `feature_drop_fraction`, an argument
that already exists on `forward()`. The resulting checkpoint drops into the 72-profile registry with the
same wire contract as epoch 13 and, unlike epoch 13, has trained support at all six q anchors.

---

## 3. First-epoch sanity gate (minimal; all must pass, checked after epoch 0)

| # | Check | Pass condition | Catches |
|---|-------|----------------|---------|
| G1 | Anchor coverage: epoch-0 `drop_hist` | all six anchors sampled, each >= 15 times | schedule wired but not firing; a mis-normalised CDF |
| G2 | Warm start loaded: epoch-0 val `center_loss` at q=0 | within +/-10% of the epoch-13 reference 3.83 | silent head reinit / partial `state_dict` load |
| G3 | Weight invariant: `pos_weight_mean` | `1.000 +/- 1e-3` | renormalisation wrong; far objects able to inflate total loss |
| G4 | Gradient liveness: mean abs grad of the object head's final conv, on one q=0.00 batch and one q=0.98 batch, both non-zero and finite | both `> 0` and finite | the AMP cast-cache zero-gradient regression fixed in `59f031a`; NaN in the 2%-survival regime |
| G5 | Segmentation not destroyed: epoch-0 clean val mIoU | `>= 0.700` | seg weight / drop curriculum wrecking the warm start immediately |

Any failure -> **stop the run**, do not continue to epoch 1, do not fall back to S7. G1-G4 are cheap
assertions inside the epoch-0 path; G5 already exists as `clean_miou` in the supervisor line.

---

## 4. Checkpoint selection — decoded validation service metrics only

`loc_dim_loss` and every in-training `selection_score` are **logged, never promoting** (S1.4).

Promotion procedure:
1. `checkpoint_every_epochs: 1` writes `epoch_000.pt` ... `epoch_023.pt`.
2. Run `evaluate_route_b_checkpoint_v1.py` (fixed decoder: score 0.20, top-k 120, NMS 2 px, 3.0 m match,
   40 m GT eligibility, no world suppression) on the **epoch-13 warm start** first. This establishes the
   baseline at the *production* threshold — the published 0.5702 / 0.4852 figures are at score 0.02 and are
   a permissive **ceiling**, not the operating point. This is the selection baseline, not a new diagnostic.
3. Decode-evaluate epochs 6..23 at **q = 0.00**. Rank by the registered scalar

   `S = 0.45 * person_recall + 0.30 * vehicle_recall + 0.25 * overall_f1`

   all from the `primary` block of the evaluator's derived JSON. Person is weighted highest because it is
   the failing class; `overall_f1` is included so recall cannot be bought purely with false positives.
4. Take the top 4 by `S` and decode-evaluate each at **q = 0.90**. Final rank is
   `min(S_clean, S_q090)` — the same maximin logic the trainer uses, but on decoded service metrics
   instead of a regression proxy.
5. Promote the top-ranked checkpoint **only if** it clears every S5 target. Record its SHA-256, its epoch,
   its full derived JSON at both q values, and the baseline JSON from step 2.

The 0.02-threshold ceiling may be recomputed for the promoted checkpoint and reported alongside, for
continuity with the baseline figures. It does not select.

---

## 5. Validation acceptance targets

All measured on the Route B **validation** view with the fixed decoder. Test stays closed.
`B_*` = the epoch-13 baseline re-decoded at threshold 0.20 in step 4.2. Every threshold below is
**registered pre-run**; the guards are anchored to measured epoch-13 numbers, the deltas and the
robustness ratios are judgement calls with no prior measurement.

**Primary (must pass; anchored to a measured baseline):**
- `person_recall >= B_person_recall + 0.05` (absolute)
- `vehicle_recall >= B_vehicle_recall + 0.03` (absolute)
- `overall_precision >= B_overall_precision - 0.02` — recall may not be bought with false positives
- `overall_duplicate_fp_fraction <= B_overall_duplicate_fp_fraction + 0.02`

**Segmentation guards (must pass; clean q=0.00, anchored to measured epoch-13 values):**
- `mIoU >= 0.7000` (baseline 0.7078)
- `vehicle_iou >= 0.8000` (baseline 0.8117)
- `person_iou >= 0.3200` (baseline 0.3274)

**Robustness (registered, not calibrated — epoch 13 has no trained support at these anchors, so there is
no baseline to compare against; report all, only the first binds):**
- binding: `overall_f1 @ q=0.90 >= 0.70 * overall_f1 @ q=0.00`
- report-only: `overall_f1 @ q=0.98 >= 0.50 * overall_f1 @ q=0.00`
- report-only: `overall_f1` at each of q = 0.30 / 0.50 / 0.70, monotone-non-increasing in q expected

**Localisation guard (must not regress):**
- `overall_xy_mae_m <= B_overall_xy_mae_m + 0.10` m

Small-object and 20-40 m outcomes are **reported, not gated** — the evaluator emits no range-banded or
size-banded split today, and adding one would be a new diagnostic, which this plan excludes.

---

## 6. Epoch count and runtime

**24 epochs.** Stage 2 plateaued by epoch ~15-20 (person IoU peaked at epoch 15; epochs 20-24 are flat to
4 decimal places). This run splits the drop budget across 6 discrete anchors instead of a continuum, so
each anchor sees ~29 batches/epoch and warrants a little more budget; 24 epochs with the existing cosine
schedule, no early stop.

Measured epoch cost from `20260825_195301` at identical batch size / workers / input size and the same
two validation passes: **142-180 s**, median ~171 s.

| Item | Estimate |
|------|----------|
| 24 training epochs @ 180 s (conservative) | ~72 min |
| Baseline decode of epoch 13 @ 0.20 | 1 pass |
| Clean decode, epochs 6..23 | 18 passes |
| q=0.90 decode, top 4 | 4 passes |
| 23 decode passes @ ~3 min (assumed; measure at the first pass) | ~69 min |
| **Total** | **~2 h 20 m** |

This fits the 2-3 h window. The decode cost is the only unmeasured term — if the first pass exceeds 4 min,
narrow step 4.3 from epochs 6..23 to every second epoch plus the top 5 by the logged proxy, and **record
the reduction explicitly in the run log**. Do not silently truncate.

---

## 7. Fallback trigger — the existing decoupled head (single, conditional)

**Trigger.** Fire if and only if, at the top-ranked checkpoint from S4:

> the primary recall targets in S5 fail, **and** the failure has the signature of the two objectives
> competing inside the shared trunk — that is, either
> (a) `person_recall >= B_person_recall + 0.03` while `overall_xy_mae_m > B_overall_xy_mae_m + 0.10` m, or
> (b) `overall_xy_mae_m <= B_overall_xy_mae_m` while `person_recall < B_person_recall + 0.02`.

One objective moving only at the other's expense is the specific evidence that a shared 3-layer trunk is
the bottleneck. If **both** recall and localisation are flat, the head is not the constraint and the
trigger does **not** fire — that outcome escalates to Abiodun as a scope question, not to another run.

**Fallback run.** Identical config with `head_arch: "decoupled"`, 12 epochs, same warm start. Because
`_make_head` builds the decoupled branches with the same trunk topology as the shared head and the branches
differ only in the final 1x1 output width (`model.py:83-88`), the epoch-13 shared head can be split into
them **exactly**: copy the trunk layers verbatim into both `heatmap_head` and `reg_head`, and slice the
shared final 1x1 conv's weight/bias `[:heatmap_channels]` into `heatmap_head[-1]` and
`[heatmap_channels:]` into `reg_head[-1]`. At initialisation the decoupled model then computes exactly the
same function as epoch 13, so the fallback starts from the warm start rather than from scratch. That loader
is ~15 lines and is **not** written unless the trigger fires.

This is the only fallback. There is no second alternative recipe in this plan.

---

## 8. Change inventory (complete)

| File | Change |
|------|--------|
| `pole_lraspp_multimodal_fusion/train_fusion.py` | anchor-exact drop sampler + `drop_hist` logging (S2.1); `feature_drop_max` path preserved |
| `pole_lraspp_multimodal_fusion/object_targets.py` | `pos_weight` / `weight_cap` on `focal_heatmap_loss`; weight-map construction + `pos_weight_mean` in `multitask_object_loss` (S2.2) |
| `object_head_pilot_v1/configs/focused_noae_v1.json` | new (S2.5) |
| epoch-0 gate assertions | G1-G4 (S3) |
| decoupled warm-start splitter | **only if** S7 fires |

Both code changes are backward-compatible: with `feature_drop_values` absent and `pos_weight_enable` false,
every existing config reproduces today's behaviour exactly.

No changes to: the model graph, the dataset view builder, the evaluator, the decoder thresholds, the split
runtime, the codec, or anything under `test`.
