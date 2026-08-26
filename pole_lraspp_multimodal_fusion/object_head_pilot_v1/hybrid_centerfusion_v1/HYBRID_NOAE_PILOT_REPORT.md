# LR-ASPP / CenterFusion hybrid noAE pilot — result

**Terminal verdict: `HYBRID_NOAE_PILOT_NO_GAIN`**

Stopped at the registered six-epoch early continuation gate. No second
architecture was tried and no gate threshold was changed.

Experiment dir: `experiments/hybrid_centerfusion_v1/20260826_162833`
Registered plan (written before any result): `HYBRID_NOAE_PILOT_PLAN.md`

---

## 1. Architecture and exact split point

Input is unchanged: the dataset still emits one 7-channel tensor
`cat[rgb(3), radar(4)]` at 768x432, so no data format changed.

```
x (B,7,432,768)
 |
 +-- rgb   = x[:, :3]              +-- radar = x[:, 3:7]
 |                                  |
 |                            radar_encoder.stem   Conv2d(4->16, s2)      1/2
 |                              |          \                              (warm-started from the
 |                              |           \                              baseline stem's radar slice)
 |   backbone['0'][0]           |            block4  (16->32,  s2)        1/4  radar4
 |   Conv2d(3->16, s2)  <--(+)--+            block8  (32->64,  s2)        1/8  radar8
 |        |                                  block16 (64->96,  s2)        1/16 radar16
 |   backbone['0'][1:]  BN + Hardswish
 |        |
 |   backbone['1'..'16']   (stock MobileNetV3-Large, stock parameter names)
 |        |-- '3'  -> quarter  24 ch  1/4      <-- new tap
 |        |-- '4'  -> low      40 ch  1/8
 |        '-- '16' -> high    960 ch  1/16
 |
=========================== encode_front() returns here =====================
 fused = {low, high, quarter, radar4, radar8, radar16}      <-- q / AE attach point
=========================== decode_tail() starts here =======================
 |
 +-- classifier({low, high})  ------------------------------> "out"  (seg, 3x54x96)
 |
 +-- COARSE 1/8 CenterNet stage  (warm-started, baseline-identical)
 |     cat[low, high^]  -> 1000 ch
 |     SplitClassHeatmapHead(depth 3, hidden 128)
 |       shared_trunk -> vehicle_heatmap(1) | person_heatmap(1) | regression(12)
 |     coarse: 14 ch @ 1/8
 |
 +-- FEATURE-LEVEL RGB/RADAR FUSION, FPN to 1/4
 |     p16 = ReLU GN lat16(cat[high , radar16])        96 ch  1/16
 |     p8  = ReLU GN (lat8(cat[low , radar8]) + p16^)  96 ch  1/8
 |     p4  = ReLU GN (lat4(cat[quarter, radar4]) + reduce8(p8)^)  64 ch  1/4
 |     f4  = smooth4(p4)                               64 ch  1/4
 |
 +-- RADAR-CONDITIONED REFINEMENT (CenterFusion second-stage idea)
 |     refine_trunk(cat[f4, coarse^])   78 -> 64 ch @ 1/4
 |     -> refine_vehicle_heatmap(1) | refine_person_heatmap(1) | refine_regression(12)
 |     delta: 14 ch @ 1/4
 |
 '-- object = upsample(coarse, 432x768) + upsample(delta, 432x768)   -> "object" (14 ch)
```

**Exact split point.** `encode_front(rgb, radar) -> OrderedDict{low, high,
quarter, radar4, radar8, radar16}` and `decode_tail(fused, out_hw) -> {"out",
"object"}`, at
[hybrid_model_v1.py:encode_front / decode_tail](hybrid_model_v1.py). `forward()`
is exactly `decode_tail(maybe_drop(encode_front(x[:,:3], x[:,3:7])))`. `q`
(`_objectness_drop`) and a feature AE (`_apply_feature_ae`, acting on
`fused["high"]`) both attach at that boundary and are structural no-ops at q=0
with no AE, exactly as on the baseline. The live UDP/OAI runtime was not touched.

**14-channel output order preserved:** `[vehicle_heatmap, person_heatmap,
local_xyz(3), dims(3), yaw_sin, yaw_cos, parked, radar_support, bbox_w, bbox_h]`,
upsampled to the input grid, so `decode_objects` and the map contract are
unchanged. Confirmed by the Phase B schema check and by four evaluations running
through the untouched production evaluator.

**Radar representation decision (requirement 6).** The corpus does carry raw
projected radar points (`radar_points_path`, with `u`, `v`, `camera_depth_m`,
`radial_velocity_mps`, `stationary_age_s`). Frustum association was *not* used:
it would need a per-detection dynamic gather, a second-stage target format and a
two-stage training loop, i.e. new data plumbing that the task explicitly scopes
out. The retained calibrated four-channel raster is the pixel-aligned projection
of those same fields at full 768x432 input resolution, so the refinement is
radar-conditioned through the radar branch as the fallback the task authorises.

**Parameters:** 4,952,138 total (baseline 4,693,964) — +258,174, +5.5%.

## 2. Files changed

No production file was edited. `pole_lraspp_multimodal_fusion/model.py`,
`train_fusion.py`, `evaluate_fusion.py` and `object_targets.py` are untouched;
the hybrid is selected by `object_heads.head_arch = "centerfusion_hybrid_v1"` and
routed by a name-level builder dispatch installed at entry.

All new, under `object_head_pilot_v1/hybrid_centerfusion_v1/`:

| file | role |
| --- | --- |
| `hybrid_model_v1.py` | model, warm start, `install()` builder dispatch |
| `train_entry_v1.py` | installs dispatch + baseline's target cap, calls production `train_fusion` |
| `eval_entry_v1.py` | installs dispatch, calls production `evaluate_fusion` |
| `evaluate_hybrid_route_b_v1.py` | Route B decode wrapper; imports `FIXED_DECODER`/`summarize` from the existing pilot evaluator so metric code cannot drift |
| `launch_check_v1.py` | Phase B: one real q=0 AMP fwd/bwd, per-branch gradient gate |
| `warm_start_parity_v1.py` | Phase C part 1: fp32 tensor parity + TF32 control; writes `warm_start.pt` |
| `gate_and_select_v1.py` | parity gate, early continuation gate, final selection |
| `run_hybrid_chain_v1.sh` | the automatic phase chain |
| `configs/hybrid_centerfusion_v1.json` | trial |
| `HYBRID_NOAE_PILOT_PLAN.md` | gate registration (pre-result) |

## 3. Warm-start tensor mapping

339 source tensors -> 344 hybrid tensors. **0 incompatible.** 35 new.

| transform | count | tensors |
| --- | --- | --- |
| `identity` | 318 | all `backbone.*` (except the stem) and all `classifier.*` |
| `identity (shared trunk)` | 18 | `object_head.{0,1,3,4,6,7}.*` -> `object_head.shared_trunk.*` |
| `channel_slice[:, 0:3]` | 1 | `backbone.0.0.weight` -> RGB stem |
| `channel_slice[:, 3:7]` | 1 | `backbone.0.0.weight` -> `radar_encoder.stem.weight` |
| `row_slice[0:1]` | 2 | `object_head.9.{weight,bias}` -> `vehicle_heatmap_head` |
| `row_slice[1:2]` | 2 | `object_head.9.{weight,bias}` -> `person_heatmap_head` |
| `row_slice[2:]` | 2 | `object_head.9.{weight,bias}` -> `regression_head` |

The two structural splits are exact re-parameterisations, not approximations:
`conv7(cat[rgb,radar]) == conv3(rgb) + conv4(radar)`, and
`cat[W[0:1]x, W[1:2]x, W[2:]x] + split bias == Wx + b`.

**New (35):** `radar_encoder.block{4,8,16}.*`, `lat{4,8,16}.*`, `norm{4,8,16}.*`,
`reduce8.*`, `smooth4.*`, `refine_trunk.*`,
`refine_{vehicle,person}_heatmap_head.*`, `refine_regression_head.*`.

Every mapped tensor was verified bit-equal to its staged source
(`torch.equal`). The source checkpoint was opened read-only and never written.

## 4. Phase B — launch check: PASS

One real q=0 AMP forward/backward, batch 16, on a real Route B training batch.
`torch.amp.GradScaler` overflows by design on its first steps at the default
65536 scale; the check reproduces that settling (3 halvings, settled scale 8192)
so the finite-gradient assertion is about the model, not the scaler warm-up.

| branch | trainable tensors | grad norm | zero-grad | non-finite |
| --- | --- | --- | --- | --- |
| rgb_backbone | 78 | 38.18 | 0 | 0 |
| radar_encoder | 10 | 4.684 | 0 | 0 |
| vehicle_heatmap | 4 | 1.553 | 0 | 0 |
| person_heatmap | 4 | 18.91 | 0 | 0 |
| regression/refinement | 27 | 69.03 | 0 | 0 |
| segmentation | 6 | 0.0943 | 0 | 0 |

Loss 7.3469 (finite). Output `[16, 14, 432, 768]`. Refinement residual
8.85e-3 (non-zero, i.e. the branch is genuinely wired). Peak VRAM 7,587 MiB.
Zero trainable tensors fell outside the six branches.

## 5. Phase C — warm-start parity: PASS

**Part 1, strict fp32** (registered relative tolerance 1e-4), 64 validation frames:

| quantity | relative delta |
| --- | --- |
| fused `low` feature | 7.43e-7 |
| fused `high` feature | 1.62e-6 |
| segmentation logits | 1.05e-6 |
| coarse 1/8 object logits | 7.62e-7 |

**TF32 control (reported, not gated).** Under the GPU's default TF32 conv path
the hybrid-vs-baseline delta is 4.9e-4 … 1.1e-3, but the *baseline's own*
TF32-vs-fp32 self-difference is 1.9e-3 … 4.4e-3 — larger. The hybrid is closer to
the baseline than the baseline is to itself under TF32, so the TF32 residual is
GPU arithmetic (one 7-channel convolution vs a 3-channel plus a 4-channel one,
rounded at a 10-bit mantissa), not a warm-start error.

**Part 2, decoded on the full 3,588-frame validation split** through the
untouched evaluator, primary decoder contract:

| metric | baseline | warm start | delta | tol |
| --- | --- | --- | --- | --- |
| vehicle P / R / F1 @0.20 | 0.4624 / 0.4498 / 0.4560 | 0.4626 / 0.4500 / 0.4562 | +0.0002 | 0.005 |
| person P / R / F1 @0.20 | 0.3480 / 0.3752 / 0.3611 | 0.3479 / 0.3748 / 0.3608 | <=0.0004 | 0.005 |
| vehicle / person XY MAE | 1.1343 / 1.3195 m | 1.1345 / 1.3203 m | <=0.0007 m | 0.01 |
| vehicle IoU / person IoU / mIoU | 0.8117 / 0.3274 / 0.7078 | 0.8117 / 0.3273 / 0.7078 | <0.0001 | 0.002 |
| recall ceiling @0.02 veh / per | 0.5702 / 0.4852 | 0.5703 / 0.4850 | <=0.0002 | 0.005 |

The baseline's score-0.02 ceiling was measured here for the first time and lands
on the stated 0.5702 / 0.4852. All seven pre-training baseline numbers reproduce.

**Scope note.** Bit-identical *object* parity is not achievable for this task as
specified: the mandated higher-resolution detection branch changes that output by
construction, and an exactly-zero refinement initialisation is mutually exclusive
with Phase B's "finite non-zero gradient in the refinement branch" at step 1. The
gate was therefore placed on what is architecturally retained (features,
segmentation, coarse head — all exact to ~1e-6) plus the decoded-metric
reproduction above. The refinement output convolutions are initialised at
std 1e-4; their whole contribution is inside the table.

## 6. Phase D/E — six warm-started clean-q epochs

Training was healthy: train loss 5.6714 -> 5.2724 monotonically, no NaN, no
collapse, mIoU stable-to-up. Epoch time 83-96 s (mean 87 s), peak VRAM
6,866 MiB allocated.

| epoch | train loss | val loss | mIoU | veh IoU | per IoU | center | loc | dim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.6714 | 20.4231 | 0.7135 | 0.8171 | 0.3389 | 3.7342 | 2.9370 | 0.5225 |
| 2 | 5.5741 | 20.5441 | 0.7084 | 0.8114 | 0.3296 | 3.7565 | 2.9510 | 0.5390 |
| 3 | 5.4969 | 20.7294 | 0.7108 | 0.8113 | 0.3368 | 3.8068 | 2.9377 | 0.5332 |
| 4 | 5.4052 | 21.0510 | 0.7107 | 0.8079 | 0.3403 | 3.8722 | 2.9863 | 0.5266 |
| 5 | 5.3344 | 21.0440 | 0.7144 | 0.8122 | 0.3464 | 3.8818 | 2.9507 | 0.5393 |
| 6 | 5.2724 | 21.0876 | 0.7126 | 0.8083 | 0.3455 | 3.8895 | 2.9660 | 0.5252 |

### Epoch 6, decoded (the only evaluated epoch — the gate stopped the run)

| metric | baseline | epoch 6 | delta |
| --- | --- | --- | --- |
| vehicle precision @0.20 | 0.4624 | 0.4498 | **-0.0127** |
| vehicle recall @0.20 | 0.4498 | 0.4569 | +0.0071 |
| vehicle F1 @0.20 | 0.4560 | 0.4533 | -0.0027 |
| vehicle XY MAE | 1.1343 m | 1.1420 m | +0.0077 |
| person precision @0.20 | 0.3480 | 0.3459 | -0.0021 |
| person recall @0.20 | 0.3752 | 0.3765 | +0.0013 |
| person F1 @0.20 | 0.3611 | 0.3606 | -0.0006 |
| person XY MAE | 1.3195 m | 1.3193 m | -0.0002 |
| vehicle IoU | 0.8117 | 0.8084 | -0.0033 |
| person IoU | 0.3274 | 0.3453 | **+0.0180** |
| mIoU | 0.7078 | 0.7126 | **+0.0048** |
| **vehicle recall ceiling @0.02** | **0.5702** | **0.5790** | **+0.0089** |
| **person recall ceiling @0.02** | **0.4852** | **0.4777** | **-0.0075** |
| vehicle duplicate-FP fraction | 0.3298 | 0.3479 | +0.0181 |
| person duplicate-FP fraction | 0.2798 | 0.2698 | -0.0099 |

### Early continuation gate

| criterion | value | threshold | result |
| --- | --- | --- | --- |
| score-0.02 vehicle recall gain | +0.0089 | >= +0.05 | **FAIL** |
| score-0.02 person recall gain | -0.0075 | >= +0.05 | **FAIL** |
| score-0.20 vehicle precision drop | +0.0127 | <= 0.03 | ok |
| score-0.20 person precision drop | +0.0021 | <= 0.03 | ok |
| mIoU drop | -0.0048 (improved) | <= 0.02 | ok |
| no NaN / collapse / schema mismatch | — | — | ok |

Two of six criteria failed, both on the recall ceiling. Stopped.

## 7. Selected checkpoint

None. Selection over epochs 6/10/14/18/22/24 never ran, because the gate stopped
the run at epoch 6. For the record, the only decoded hybrid checkpoint is:

```
experiments/hybrid_centerfusion_v1/20260826_162833/checkpoints/hybrid_centerfusion_v1/epoch_006.pt
sha256 34253bceef2495b6dd5acc41e758c5d558ea19f2507e1a1c69dd9a4b73eb2027
```

Service targets are **not** reported against a selection, because there is no
selected model. Epoch 6 misses every one of them (vehicle recall 0.4569 vs 0.85,
person recall 0.3765 vs 0.80, vehicle precision 0.4498 vs 0.80, person precision
0.3459 vs 0.80, vehicle XY MAE 1.1420 vs 1.0 m, person XY MAE 1.3193 vs 1.2 m,
vehicle IoU 0.8084 vs 0.85, person IoU 0.3453 vs 0.50, mIoU 0.7126 vs 0.80).
Targets were not relaxed and matching stayed at 3.0 m.

## 8. Runtime and resources

| phase | wall clock |
| --- | --- |
| Phase B launch check | ~1 min |
| Phase C part 1 (fp32 + TF32 parity, 64 frames) | ~1 min |
| Phase C part 2 (3 full-split decodes, 3 lanes) | 4m 55s |
| Phase D (6 epochs) | 8m 42s (mean 87 s/epoch) |
| Phase E (2 full-split decodes, 2 lanes) | 3m 29s |
| **chain total** | **17m 12s** |

Peak VRAM: 7,587 MiB (launch check, batch 16) / 6,866 MiB (training).
GPU: RTX 5090, 32 GiB. Interpreter `/usr/bin/python3`, torch 2.10.0.dev+cu128.

## 9. Does this model qualify for subsequent q / AE work?

**Structurally yes, empirically not yet.** The split boundary is real and tested,
`q` and a feature AE attach at the fused bundle exactly as on the baseline, the
14-channel contract and the evaluator/map contract are intact, and the warm start
is provably exact. But nothing should be spent on AE32/64/128 or a q sweep on
top of a detector that did not beat its own baseline: the compression study would
inherit a weaker clean anchor than the one already measured. The frozen
`epoch_013` baseline remains the anchor.

## 10. What the numbers actually say (observation, not a new plan)

The changes that landed are the ones the added capacity directly controls:
person segmentation IoU +0.0180 and mIoU +0.0048 (radar now reaches segmentation
through its own encoder), and person duplicate-FP fraction -0.0099 (the 1/4-grid
term sharpens person peaks). Detection recall barely moved, and vehicle
precision fell 0.0127 with vehicle duplicate-FP up 0.0181.

So the 1/8 detection grid was **not** the binding constraint on recall at this
operating point. Six epochs at the baseline's own joint-refinement learning rate
(3e-5, chosen to keep the comparison architecture-only) is also a short budget
for 258 k freshly initialised parameters. Both are hypotheses this run does not
test and this task does not authorise testing.

## 11. Contract compliance

* Locked Route B test split never touched: `splits/test.txt` is empty in the
  view, and both the launch check and the parity script refuse to run if any
  `split == "test"` row appears. Every decode is `--split val`.
* Baseline checkpoint sha256 verified as
  `0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30`, opened
  read-only, never overwritten.
* Decoder fixed for every evaluation: q=0, top-k 120, NMS 2 px, 3.0 m
  class-aware matching, 40 m GT range, 12 px GT area floor. Score 0.02 used only
  to measure the recall ceiling; no threshold was tuned or selected.
* Single changed variable: the trial copies every optimizer, schedule,
  augmentation, loss-weight and object-target setting from
  `curriculum_stage2_joint_v1` verbatim, including the
  `vehicle_heatmap_radius_cap_px = 4` targets (installed through
  `target_variants_v1`, whose bit-identity guard runs before training).
* No CARLA/OAI run, no live UDP runtime, no AE training, no q sweep, no decoder
  grid, no 288-measurement campaign.
* Scratch lane directories and the five unevaluated intermediate checkpoints were
  deleted. Retained: `warm_start.pt`, `epoch_006.pt`, `last.pt` (so review can
  resume the identical run if it overrides the gate), all gate/check JSON, the
  training metrics CSV, the decoded `derived_metrics.json` /
  `evaluator_metrics.json` / `detections.csv` for all five evaluations, and the
  logs.
