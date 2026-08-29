# Route B v3.1 COCO Person Distillation — Final Report

Terminal: `LRASPP_COCO_DISTILLATION_TEACHER_NOT_ADOPTED`

This is the clean, bounded noAE verdict for the registered Route B v3.1
COCO-to-LR-ASPP experiment. The teacher-adoption guard fired at epoch 12:
person recall rose above the epoch-40 baseline (`0.546746 > 0.518079`) while
person precision fell below the registered `0.45` floor (`0.428196`). Training
therefore stopped before epoch 13. Epoch 18 is intentionally not available; it
was prevented by the registered guard rather than lost to a runtime failure.

## Provenance and frozen inputs

- Branch: local `master`; starting HEAD
  `a5af817a567553d6e50f41bc98b7ab1ff7dca2db`.
- Scientific config: unchanged, SHA-256
  `a6064b4ec5409c7bff4e21f8f939e576ab68b4f22bfd21fa324d84bfc23ed644`.
- Epoch-40 student: SHA-256
  `5c6bb268b43f4dd84bd7a283ff483ec4e87366a50ea51dfacee44979df2bf6e8`.
- Cached official COCO Faster R-CNN teacher: SHA-256
  `dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`,
  175,221,657 bytes; it was not downloaded or substituted.
- Dataset manifest: SHA-256
  `5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2`.
- Dataset object boxes: SHA-256
  `e085fbe8545e9703110bc83c78c6f6666996ce58fe265b08570ba6a4ab38384f`.
- Dataset: 16,827 train / 3,345 validation / 0 test; ten training and
  two validation episodes were disjoint. The locked test remained absent and
  unopened.
- Required lineage terminal: `PERSON_LINEAGE_AUDIT_SUPPORTS_COCO_TRANSFER`.
- Dimension/yaw scorer: SHA-256
  `14629e69a1617d05ca6dec2bad6901b69f83df96f2cf5543509dbe970d18069d`.

No CARLA, OAI, live-split, q/AE, or 288-measurement action was run. The pre-existing
dirty `OAI/openairinterface5g` worktree entry was preserved.

## Inherited and completed implementation

Inherited from `a5af817` and retained:

- `__init__.py`;
- the unchanged registered `configs/person_coco_distillation_v1.json`;
- the complete `distill_v1.py`;
- `teacher_v1.py` and `roi_v1.py`, with the evidence-backed corrections below.

Completed for this run:

- `dataset_v1.py`: one jointly sampled affine for RGB, four radar channels,
  segmentation/ignore masks, boxes, projected centres, regenerated native targets,
  and intrinsics; preserved world/camera-local coordinates; registered off-canvas
  quarantine; deterministic augmentation; exact teacher RGB; variable-box collate.
- `student_v1.py`: exact epoch-40 loading, allowlisted trainability, frozen BN,
  person segmentation-row masking and bit-exact restoration, one-pass split forward,
  FP32 supervised/distillation reductions, and clean deployable state handling.
- `evaluation_v1.py`: fixed v0.10 gates, taxonomy, and person
  area/distance/radar-support slices.
- `run_pipeline_v1.py`: input audit, CPU/GPU preflight, frozen numerical selection,
  bounded training/recovery, one-pass inference, offline scoring, selection, status,
  progress, decision, sentinel, report, and notification.

Corrections to the inherited partial implementation were limited to proving the
registered geometry:

1. ROIAlign now pads each feature map by one cell and shifts the model-frame boxes
   by the corresponding stride. This removes torchvision edge clamping and made
   the synthetic and real-box coordinate proof pass with maximum error below
   `1.22e-4` pixel (registered tolerance `1e-3`).
2. The transported MobileNet `high` tensor is physically 27x48 (stride 16), while
   the frozen ROI contract registers its high level as stride 32. It is now
   deterministically average-pooled to 14x24 before the registered ROI operation.
   This derives the level solely from the same transported tensor and adds no side
   channel.
3. Reporting/supervisor namespace and field-mapping defects found after the first
   epoch-6 scoring boundary were corrected without changing data, model state,
   predictions, or scientific settings. The epoch-6 recovery checkpoint was
   resumed exactly; inference was not repeated and no transient retry was consumed.

## Preflight qualification

All required structural and numerical gates passed before epoch 1: hashes and
counts; episode/sample disjointness; audit terminal; teacher frozen/eval/no-grad;
exact pinned teacher transform; joint geometry and off-canvas policy; synthetic and
real-box ROI round trips; monolithic/split parity; exact transport; no raw-modality
side channel; exact trainable/frozen allowlist; frozen BN statistics; segmentation
row masking/restoration; exact baseline reconciliation; finite activations, losses,
gradients, parameters, and optimizer state; nonzero gradients for every registered
trainable component and adapter; zero teacher/frozen gradients; clean deployable
state; inclusion of the previous batch-134 identities; and inaccessible test.

- Selected before epoch 1: `student_bf16_teacher_fp32_losses_fp32`.
- Teacher: FP32, frozen, train-only, half-batch chunks of 8 as the registered
  preflight fallback; student batch size 16.
- Distillation and all loss reductions: FP32 with autocast disabled.
- FP16/GradScaler: never used / disabled.
- Preferred BF16 candidate peak allocated/reserved: 3,841.025 / 5,646 MiB.
- Full-FP32 qualification also passed, at 5,196.728 / 7,006 MiB.
- Teacher RGB reconstruction maximum absolute error: `5.960464477539063e-08`.
- Deployable state: 351 tensors, zero teacher/projector/adapter keys.

## Bundle and parameter contract

The student backbone ran once per batch. Its same `{low, high}` bundle fed both
native decoding and distillation, with monolithic-versus-split output parity proven.

| Key | Shape for batch 1 | Elements | FP32 bytes |
|---|---:|---:|---:|
| `low` | `[1, 40, 54, 96]` | 207,360 | 829,440 |
| `high` | `[1, 960, 27, 48]` | 1,244,160 | 4,976,640 |
| Total | — | 1,451,520 | 5,806,080 |

There were no raw RGB, radar, teacher, metadata, or other modality side channels in
the transported bundle. The deployable student has 4,931,198 parameters:
2,314,311 trainable and 2,616,887 frozen. The `StudentRoiAdapter` has 775,680
trainable parameters; it is train-time-only, stored only for recovery, discarded
for deployment, and absent from both retained deployable checkpoints.

## Fixed v0.10 evaluation

Exactly one native-grid inference pass at score floor 0.02 was run at each reached
evaluation epoch. Score 0.20 was derived offline from the same prediction set;
matching stayed class-aware greedy nearest within 3 m. There was no threshold,
NMS, decoder, or calibration sweep.

| Epoch | Class | P | R | F1 | TP | FP | FN | Ignored | R@0.02 | XY m | Dim m | Yaw deg | IoU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | vehicle | 0.794434 | 0.839439 | 0.816316 | — | — | — | — | 0.871943 | 0.830483 | — | — | 0.870544 |
| baseline | person | 0.537513 | 0.518079 | 0.527617 | — | — | — | — | 0.615186 | 1.341153 | — | — | 0.453867 |
| 6 | vehicle | 0.702020 | 0.857187 | 0.771883 | 8,307 | 3,526 | 1,384 | 2,881 | 0.884635 | 0.832490 | 0.181748 | 34.5531 | 0.870289 |
| 6 | person | 0.450333 | 0.542097 | 0.491972 | 2,099 | 2,562 | 1,773 | 1,619 | 0.665548 | 1.345150 | 0.091104 | 82.8955 | 0.455940 |
| 12 | vehicle | 0.703201 | 0.854711 | 0.771588 | 8,283 | 3,496 | 1,408 | 2,892 | 0.881849 | 0.833024 | 0.181702 | 34.4378 | 0.870211 |
| 12 | person | 0.428196 | 0.546746 | 0.480263 | 2,117 | 2,827 | 1,755 | 1,797 | 0.665548 | 1.334078 | 0.093026 | 83.0496 | 0.456118 |
| 18 | — | N/A | N/A | N/A | — | — | — | — | — | — | — | — | — |

Foreground mIoU was 0.662206 at baseline, 0.663114 at epoch 6, and
0.663165 at epoch 12. Vehicle duplicate FP was 644, 1,020, and 1,073,
respectively.

### Baseline deltas

| Epoch | V P | V R | V F1 | V R@.02 | V XY | V IoU | V dup FP | P P | P R | P F1 | P R@.02 | P XY | P IoU | FG mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | -0.092414 | +0.017748 | -0.044434 | +0.012692 | +0.002007 | -0.000255 | +376 | -0.087181 | +0.024019 | -0.035645 | +0.050362 | +0.003997 | +0.002073 | +0.000909 |
| 12 | -0.091233 | +0.015272 | -0.044728 | +0.009906 | +0.002541 | -0.000333 | +429 | -0.109318 | +0.028667 | -0.047354 | +0.050362 | -0.007074 | +0.002251 | +0.000959 |

Positive XY deltas are regressions; negative deltas are improvements.

### Person diagnosis at epoch 12

At score 0.02, the 1,295 person false negatives partitioned exactly into 787
`CENTER_PRESENT_WORLD_WRONG`, 344 `HEATMAP_CENTER_MISS`, and 164
`MATCHING_CONTENTION`. At score 0.20, person recall/XY by box area were:
0.3473/1.5796 m (`[0,400)` px), 0.6630/1.3184 m (`[400,1600)`),
0.8124/1.0540 m (`[1600,6400)`), and 0.7500/1.0861 m (`[6400,1e6)`).
By distance they were 0.5537/1.3525 m (`[0,10)` m), 0.5813/1.3128 m
(`[10,20)`), 0.5654/1.2490 m (`[20,30)`), and 0.4049/1.5073 m
(`[30,40)`). Radar-supported objects achieved 0.5600 recall and 1.3257 m XY;
unsupported objects achieved 0.4583 and 1.4024 m. Slice metadata misses were zero.

## Training and evidence trajectory

| Epoch | Train loss | Validation loss | L_feat | Cosine | L_obj | Object sites | Teacher GT/evidence/miss | Omitted teacher positive without GT |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 3.035352 | 8.017263 | 0.498247 | 0.501753 | 0.630259 | 11,038 | 17,462 / 11,038 / 6,424 | 100,435 |
| 6 | 2.398624 | 7.381009 | 0.245809 | 0.754191 | 0.541754 | 11,007 | 17,433 / 11,007 / 6,426 | 99,874 |
| 12 | 2.238058 | 7.282967 | 0.222256 | 0.777744 | 0.521290 | 11,030 | 17,447 / 11,030 / 6,417 | 99,963 |

Teacher detections were evidence only for IoU-matched GT people. Teacher positives
without GT were omitted, teacher negatives never suppressed GT, teacher detections
never became pseudo-GT, and teacher box/world regression was never consumed.

Training epochs consumed 2,678.516 s. Epoch-6 and epoch-12 inference consumed
111.128 s and 118.077 s. The elapsed interval from successful preflight artifact
to terminal decision was 3,145.840 s, including scoring and the non-scientific
reporting recovery. Peak training allocated/reserved VRAM was 3,866.833 / 5,688
MiB; the maximum across both registered preflight candidates was 5,196.728 / 7,006
MiB (6.842 GiB reserved), below the 12 GiB budget.

## Decision and retention

- Evaluated checkpoints: 6 and 12; epoch 18 N/A because the adoption guard stopped
  the run.
- Eligible epochs: none. Both evaluated epochs failed person and vehicle F1
  eligibility and vehicle detection-count non-regression.
- Material route A: false at epochs 6 and 12.
- Material route B: false at epochs 6 and 12.
- Service readiness: false. At epoch 12 only vehicle recall, vehicle XY, and
  vehicle IoU met their absolute service targets.
- Teacher adoption: false as a model-design verdict; the adoption guard fired true.
- Selected checkpoint: none. v0.25 was correctly not run because no primary
  candidate was eligible.
- Non-dominated retained checkpoints: epochs 6 and 12.

Retained deployable checkpoints:

- epoch 6: `experiments/route_b_v3_1_person_coco_distillation_v1/20260828_165107/checkpoints/route_b_v3_1_person_coco_distillation_v1/epoch_006.pt`,
  SHA-256 `e7c99422f30ff1435a2fe26daec71e76c3ed702bce95bf0fd0a8d49a9d2e0e79`;
- epoch 12: `experiments/route_b_v3_1_person_coco_distillation_v1/20260828_165107/checkpoints/route_b_v3_1_person_coco_distillation_v1/epoch_012.pt`,
  SHA-256 `8592131ff4b6ba2ddc12abecc97f9aaee11134463c12a1efdec06449162d2231`.

The final structured artifacts are under
`experiments/route_b_v3_1_person_coco_distillation_v1/20260828_165107`.
`DECISION.json`, `STATUS.json`, `PROGRESS.csv`, `COMPLETION_SENTINEL`, and the
desktop notification record are present. No experiment payload is committed.

q/AE eligibility is **NO** (`q=0`, `AE=false`). The registered recovery contract
explicitly forbids q/AE continuation after this clean noAE terminal.

Registered caveat: even a gain would not isolate pretraining from the extra
supervision signal; the deliberately excluded randomized-teacher control would be
needed for that causal claim.
