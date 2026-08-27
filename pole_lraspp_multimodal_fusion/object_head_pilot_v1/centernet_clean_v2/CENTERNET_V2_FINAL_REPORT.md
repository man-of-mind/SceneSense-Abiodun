# CenterNet v2 — native dual-stride object heads, Route B

One implementation, one training run, one evaluation pass. The v1 model,
epoch-12 checkpoint, v1 experiment, legacy evaluator, corrected v1 evaluation
artifacts, dataset, locked test split and production split runtime are all
untouched.

## Artifact of record

| item | value |
|---|---|
| package | `pole_lraspp_multimodal_fusion/object_head_pilot_v1/centernet_clean_v2/` |
| experiment | `experiments/route_b_centernet_clean_v2/20260827_033102/` |
| warm start | `experiments/route_b_centernet_clean_v1/.../epoch_012.pt`, SHA-256 `59884bb0…1928e` — **verified, matches required** |
| dataset view | `experiments/route_b_noae_precision_full_v1/20260825_195301/dataset` (train 6,600 / val 3,588 / **test 0**) |
| python | `/usr/bin/python3` 3.10.12, torch 2.10.0.dev20251114+cu128, RTX 5090 |
| training | 24/24 epochs, batch 24, **42.7 min** wall clock, **peak 19,933 MiB allocated / 22,192 MiB reserved** |
| selected checkpoint | `checkpoints/resnet34_fpn_centernet_native_v2/epoch_012.pt` |
| selected SHA-256 | `bfc1b0979e84a3d24dbc41827bb8ebaa915d89709e83afa23dd436b28ca936af` |

## What v2 changed

Every item is structural, not a tuning knob:

1. **Native training.** Heatmaps, offsets and regression are built and supervised
   directly on each head's native grid. Nothing is bilinearly enlarged, for
   training or decoding. v1 placed targets at full 768×432 resolution and
   supervised regression on an interpolated tensor.
2. **Private centre offsets.** Each branch predicts its own 2-channel offset;
   the decoded centre is `(floor_cell + offset) × stride`, the exact inverse of
   target construction.
3. **Vehicles at stride 4** (108×192): heatmap + 2 offsets + 12 regression fields.
4. **Persons on a compact stride-2 path** (216×384): a learned 2× upsample of the
   fused stride-4 context concatenated with the stride-2 RGB/radar skip, then its
   own heatmap + 2 offsets + 12 regression fields.
5. **Local maxima before top-k.** Per-class 3×3 native peak suppression, then
   threshold, then top-k = 120 per branch, then merge.
6. **Segmentation** uses a task-specific lightweight decoder: stride-4 fused
   context → learned 2× upsample → stride-2 RGB/radar skip → learned 2× upsample
   to the final output. Two transposed convs and two 3×3 convs; no HRNet, no
   LR-ASPP. Only the 3-channel logits ever exist at full resolution.
7. **Class-specific regression maps.** The branches own separate regression
   tensors, so v1's class-agnostic `reg_mask` overwrite is structurally impossible.

Preserved: 7-channel RGB(3)+radar(4) input, ImageNet-pretrained ResNet34 RGB
encoder, independent 4-channel radar encoder, RGB/radar fusion, three-class
segmentation, the decoded object field schema, and a real UE/edge split.

**Split boundary (verified, not asserted).** `encode_front` returns
`{rgb_p2 (128×108×192), radar_p2 (128×108×192), s2 (16×216×384)}` = 6,635,520
elements/frame. `decode_tail` was instrumented and read **exactly those three
keys and nothing else**; its output is bitwise identical to `forward`, and
zeroing the bundle changes every tail output. There is no raw RGB or radar side
channel, and the stride-2 skip crosses the boundary *inside* the bundle rather
than bypassing it, so a future q/AE stage still has one accountable attachment
point covering every feature the tail consumes. q and AE are disabled here.

## Checkpoint mapping (warm start from v1 epoch 12)

| | count |
|---|---|
| loaded (shape-compatible ResNet34 / RGB-FPN / radar-encoder / radar-FPN) | **283** |
| newly initialized | **53** — `vehicle_head` 12, `person_head` 12, `classifier` 11, `fusion` 6, `person_feature` 6, `stride2_proj` 6 |
| incompatible | **0** |
| total v2 tensors | 336 |
| v1 tensors deliberately outside warm-start scope | 35 — `object_head` 12, `refinement_head` 12, `classifier` 8, `fusion_projection` 3 |

Full tensor-name lists: `artifacts/checkpoint_mapping_report.json`.

## Optimization (one fixed schedule, no sweep)

AdamW, weight decay 1e-4, 1-epoch linear warmup then cosine to `min_lr_ratio`
0.01 over 24 epochs, AMP on with object losses evaluated in fp32, backbone BN
frozen. Three LR tiers by whether a tensor was newly initialized:

| tier | LR | tensors |
|---|---|---|
| new heads / offsets / seg decoder | **3e-4** | 53 |
| warm-started RGB-FPN, radar-FPN, radar encoder | **1e-4** | 67 |
| pretrained ResNet34 backbone | **3e-5** | 36 trainable (BN frozen) |

Loss weights carried over unchanged from the v1 config (center 4.0, location 1.5,
dimensions 0.6, yaw 0.3, parked 0.2, radar_support 0.1, bbox2d 1.0,
segmentation 0.4); `offset` is the single new weight, fixed at 1.0. Each branch is
an independent objective normalised by its own positive count, which is what
balances the two classes.

## Pre-run checks — all PASS

`py_compile` (8 files) · config parse · checkpoint SHA-256 · split counts
6600/3588/0 · one real batch-24 forward+backward (loss finite, 19.7 GiB) ·
finite nonzero gradients in all 11 new vehicle/person/offset/segmentation
tensors · output shapes and decoded-field schema (both strides present) ·
split-boundary check. Report: `artifacts/launch_check_v2.json`.

One methodology correction during the checks: gradients were first read while
still AMP-scaled, which reports `inf` by construction on the first iteration
(default GradScaler init scale 65536 × a cold-start loss of ~392). The check now
inspects an unscaled backward on the same batch.

## Per-epoch validation table (val 3,588 frames, frozen native v2 decoder)

Score 0.20 is the operating point; 0.02 is the permissive recall diagnostic only.
No threshold or top-k tuning. GT denominator unchanged.

| epoch | veh P | veh R | veh F1 | per P | per R | per F1 | veh R@0.02 | per R@0.02 | veh XY MAE | per XY MAE | veh dim MAE | per dim MAE | veh IoU | person box-mask IoU | mIoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.7753 | 0.6056 | 0.6800 | 0.6432 | 0.4412 | 0.5234 | 0.7182 | 0.5115 | 0.9315 | 1.2478 | 0.1996 | 0.1525 | 0.7074 | 0.3181 | 0.6671 |
| **12** | **0.7826** | **0.6119** | **0.6868** | **0.5058** | **0.4891** | **0.4973** | **0.7186** | **0.5426** | **0.8997** | **1.0448** | **0.1807** | **0.1330** | **0.7180** | **0.3110** | **0.6687** |
| 16 | 0.7787 | 0.6124 | 0.6856 | 0.5349 | 0.4829 | 0.5075 | 0.7063 | 0.5312 | 0.9053 | 1.0473 | 0.1821 | 0.1284 | 0.7340 | 0.3181 | 0.6767 |
| 20 | 0.8037 | 0.6040 | 0.6897 | 0.6266 | 0.4658 | 0.5344 | 0.7020 | 0.5079 | 0.8957 | 1.0295 | 0.1847 | 0.1276 | 0.7409 | 0.3200 | 0.6799 |
| 24 | 0.8109 | 0.6006 | 0.6901 | 0.6493 | 0.4604 | 0.5388 | 0.6989 | 0.5016 | 0.8817 | 1.0255 | 0.1802 | 0.1283 | 0.7409 | 0.3275 | 0.6824 |

| epoch | veh TP | veh FP | veh FN | per TP | per FP | per FN |
|---|---|---|---|---|---|---|
| 8 | 5395 | 1564 | 3513 | 2046 | 1135 | 2591 |
| **12** | **5451** | **1514** | **3457** | **2268** | **2216** | **2369** |
| 16 | 5455 | 1550 | 3453 | 2239 | 1947 | 2398 |
| 20 | 5380 | 1314 | 3528 | 2160 | 1287 | 2477 |
| 24 | 5350 | 1248 | 3558 | 2135 | 1153 | 2502 |

Person IoU is reported as `person_box_mask_iou` throughout: the person GT is a
filled projected box, not a silhouette, so its ceiling is set by the box
rasterization rather than by the model.

**Selection** (registered rule, in order: highest min class recall @0.20 → highest
mean class F1 → lowest mean XY MAE → earlier epoch) → **epoch 12**
(min class recall 0.48911). Epoch 24 has the best mean class F1 (0.6144) but
lower min class recall (0.4604), so it does not win under this rule.

## Recall by band, selected epoch 12

| distance | vehicle @0.20 | @0.02 | person @0.20 | @0.02 |
|---|---|---|---|---|
| 0–10 m | 0.5671 | 0.8030 | 0.8384 | 0.8872 |
| 10–20 m | 0.7697 | 0.8547 | 0.7422 | 0.7769 |
| 20–30 m | 0.5305 | 0.5800 | 0.4777 | 0.5432 |
| 30–40 m | 0.5133 | 0.5766 | 0.2550 | 0.3123 |

| resized box max dim | vehicle @0.20 | @0.02 | person @0.20 | @0.02 |
|---|---|---|---|---|
| <8 px | — | — | 0.0000 (17 GT) | 0.0588 |
| 8–16 px | 0.1624 | 0.1966 | 0.2563 | 0.3136 |
| 16–32 px | 0.4848 | 0.5446 | 0.5057 | 0.5657 |
| 32–64 px | 0.7160 | 0.7875 | 0.7955 | 0.8221 |
| 64+ px | 0.6293 | 0.8208 | 0.6879 | 0.7861 |

## Service-target gate — NOT met

| target | required | epoch 12 | |
|---|---|---|---|
| vehicle precision | ≥ 0.80 | **0.7826** | FAIL |
| vehicle recall | ≥ 0.85 | **0.6119** | FAIL |
| person precision | ≥ 0.80 | **0.5058** | FAIL |
| person recall | ≥ 0.80 | **0.4891** | FAIL |
| vehicle XY MAE | ≤ 1.0 m | 0.8997 | PASS |
| person XY MAE | ≤ 1.2 m | 1.0448 | PASS |
| vehicle IoU | ≥ 0.85 | **0.7180** | FAIL |
| person box-mask IoU | ≥ 0.50 | **0.3110** | FAIL |
| mIoU | ≥ 0.80 | **0.6687** | FAIL |

Seven of nine targets bind. No target was relaxed and the GT denominator was not
changed.

## Comparison with v1 at the selected epoch

Baseline = v1 epoch 12 through the **corrected** decoder from
`CENTERNET_EVALUATION_CONTRACT_AUDIT.md` (the fairest v1 number; the published v1
decoder was defective).

| metric @0.20 | v1 corrected e12 | v2 e12 | Δ |
|---|---|---|---|
| vehicle precision | 0.81441 | 0.7826 | −3.18 pp |
| vehicle recall | 0.57488 | 0.6119 | **+3.70 pp** |
| vehicle F1 | 0.67399 | 0.6868 | **+1.28 pp** |
| person precision | 0.54039 | 0.5058 | −3.46 pp |
| person recall | 0.48911 | 0.48911 | **exact tie (2,268 TP both)** |
| person F1 | 0.51347 | 0.4973 | −1.62 pp |
| vehicle XY MAE | 0.9143 m | 0.8997 m | **−1.5 cm** |
| person XY MAE | 1.0949 m | 1.0448 m | **−5.0 cm** |
| vehicle recall @0.02 | 0.72968 | 0.7186 | −1.11 pp |
| person recall @0.02 | 0.62303 | 0.5426 | −8.04 pp |
| mIoU | 0.68883 | 0.6687 | −2.01 pp |
| person box-mask IoU | 0.3703 | 0.3110 | −5.93 pp |

**Improvement rule, registered in `select_and_report_v2.py` before any v2
evaluation was run:** improved iff (min class recall up **and** mean class F1 not
down by more than 0.01) **or** (mean class F1 up **and** min class recall not down
by more than 0.01). At the selected epoch min class recall is −0.000001 (a literal
tie: both models produce exactly 2,268 person true positives) and mean class F1 is
−0.00166. Neither branch of the rule is satisfied, so the registered verdict is
`NO_GAIN`. This is knife-edge and should be read as such: the honest picture is a
**real vehicle-side gain, a person-side wash on recall with a precision loss, and
a small segmentation regression** — not a uniformly worse model.

## What the geometry correction did and did not buy

* **The decoder defect is gone.** Top-k = 120 never saturates in either branch at
  either threshold (max 30 person / 14 vehicle predictions per frame at 0.02;
  0 frames at cap). The v1 failure mode — ~81 % of the budget spent on
  interpolated duplicates — cannot occur. The remaining recall shortfall is a
  model/target property, not a decoder budget artifact.
* **Vehicles improved** on recall, F1 and localization at the same operating
  point, which is what native supervision plus offsets was expected to deliver.
* **The stride-2 person branch did not deliver a recall gain.** Person recall at
  0.20 is tied and at the 0.02 diagnostic it is 8 pp *worse* than v1's corrected
  decoder, i.e. the v2 model's person ceiling is genuinely lower. Two untested
  candidate causes, stated as hypotheses, not findings: (i) the stride-2 grid has
  4× the background cells of stride 4 under the same per-positive focal
  normalisation, so the person heatmap carries proportionally more background
  gradient pressure; (ii) the person branch's semantic depth comes only through a
  single learned upsample of the stride-4 context plus a 16-channel skip.
* **Segmentation regressed slightly** (mIoU 0.6687 at e12, 0.6824 at e24, vs
  v1's 0.68883), so the lightweight stride-2 skip decoder did not achieve its
  design intent here. Most likely cause, also untested: the v2 object loss is the
  **sum of two branch objectives** (~2× v1's single-tensor object loss) while the
  segmentation weight was deliberately held at v1's 0.4 to avoid an unauthorized
  loss sweep, so segmentation lost relative weight.
* **Overfitting is visible and was not stopped.** Validation loss rises from
  epoch ~6 while training loss keeps falling (`artifacts/training_metrics.csv`);
  all 24 epochs were trained as instructed, with no early stopping.

## Unresolved limitation — pedestrian occlusion

**True pedestrian visibility remains unresolved in this corpus, and person recall
cannot be attributed to model quality alone.** From the v1 audit, which this run
does not supersede: CARLA 0.10 renders **zero walker semantic pixels** in any of
the 3,588 validation frames, so no automatic pedestrian visibility gate is
possible and none was invented; 46.9 % of score-0.02 person false negatives have
a nearer GT box covering ≥ 50 % of their own box and 40.3 % have zero radar
support. No recall denominator anywhere in this report was changed to disguise
this, and every person figure above is scored against the full eligible GT set.

## Deliverables

Code/config: `centernet_model_v2.py`, `targets_v2.py`, `losses_v2.py`,
`decode_v2.py`, `train_v2.py`, `evaluate_v2.py`, `launch_check_v2.py`,
`select_and_report_v2.py`, `configs/`, `PROVENANCE.md`.
Reports: this file, `artifacts/` (checkpoint mapping, per-epoch table, selection
+ gate, launch check, LR groups, training metrics, per-epoch evaluation JSON,
exact commands). Detection CSVs and per-GT band CSVs stay in the experiment
directory (`experiments/` is gitignored); checkpoints and the dataset are not
committed.

---

# CENTERNET_V2_NO_GAIN
