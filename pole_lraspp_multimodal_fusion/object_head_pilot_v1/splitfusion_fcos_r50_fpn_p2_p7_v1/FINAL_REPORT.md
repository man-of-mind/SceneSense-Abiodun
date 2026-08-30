# SplitFusion FCOS R50 FPN P2-P7 V1 final report

Terminal verdict: `SPLITFUSION_FCOS_CLEAN_BASE_NOT_SERVICE_READY`.

The fixed selection chose epoch **8**. The retained checkpoint is `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1/20260829_214123/checkpoints/epoch_008.pt` with SHA-256 `1b4cf1e2668b3abee7a772a2eb907c8c20f7af8b77c356da60e37a70e99f132a`. It passed 5 of nine clean service targets.

## Provenance and scope

Starting local master: `35384e0106d61021459c30df20c8560eb7f9e131`. Report-generation HEAD: `35384e0106d61021459c30df20c8560eb7f9e131`. The final local master commit is the commit containing this report; its resolved hash is written after commit to the Git-ignored experiment `FINAL_GIT_STATE.json` to avoid an impossible self-referential commit hash.

Documented runtime-only source amendments: `[{"amended_source_state_sha256": "1a93a3a710fa17dbd193be5169547b9cbf566432ba1d62b180bf41380fac8683", "changed_files": ["report.py", "train.py"], "reason": "End-of-epoch telemetry requested autograd gradients for the intentionally frozen epoch-1-to-3 RGB stem parameter. PyTorch rejects a differentiated tensor with requires_grad=false even when allow_unused=true. The repair temporarily enables that flag only inside diagnostic telemetry and restores it in finally; autograd.grad writes no parameter gradients and no optimizer is called.", "scope": "diagnostic_runtime_only"}, {"amended_source_state_sha256": "e8b3f4835d8d3d2df94107dbc1fda144f4ee69387e90b168c61f7d3e0b5c5958", "changed_files": ["report.py", "train.py"], "reason": "Scientific epoch 9 update 862 had finite losses, finite gradient tensors, finite parameters, and finite optimizer state, but legitimate exact-zero parameter gradients for RGB stem, radar stem, and P2 on that batch. The implementation-only guard incorrectly required every trainable group to be nonzero on every scientific update. Phase B and disposable qualification already prove required nonzero gradient reachability. The repair retains per-update finiteness checks and complete nonzero telemetry but enforces per-update nonzero as an abort only during disposable qualification, as required by the contract.", "scope": "training_runtime_guard_only"}, {"amended_source_state_sha256": "221731714a541acaf3098876151bf9eb4af48fe149e11c0661ee565072cc2800", "changed_files": ["evaluate.py"], "reason": "The fixed scorer completed epochs 3 and 8, then encountered a valid epoch-16 zero-detection prediction set. The frozen flattener attempted to add the vehicle and person localization MAEs even though both are undefined (None) when there are no matches. The adapter preserves the frozen numeric path exactly, represents only the undefined mean localization MAE as null, marks undefined metrics non-finite for service gating, and resumes from already-durable evaluation records after verifying their checkpoint and prediction provenance.", "scope": "evaluation_undefined_metric_runtime_only"}]`. They were explicitly qualified, preserved the latest durable scientific checkpoint where applicable, and changed no architecture, target, mathematical loss, multiplier, optimizer, LR schedule, sampler, augmentation, batch size, or inference setting.

Official weights: `FCOS_ResNet50_FPN_Weights.COCO_V1`, 129,612,099 bytes, SHA-256 `99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7`, URL `https://download.pytorch.org/models/fcos_resnet50_fpn_coco-99b0c9b7.pth`. Installed Torchvision revision `4efae90d072d0d11e244d6e213208b357f89efe7` uses BSD-3-Clause source licensing. The COCO weight/dataset disclaimer is recorded separately in `OFFICIAL_PROVENANCE.md`.

Internal class 0 is vehicle copied exactly from COCO car row 3; internal class 1 is person copied exactly from COCO person row 1; canonical output labels are restored to vehicle=1/person=2 and background remains implicit.

The locked test split remained absent and unopened. CARLA, OAI, q, quantization, AE/hybrid-q, live split deployment, and the 288-cell campaign were not run. The pre-existing dirty OAI tree retained the exact starting status hash.

## Architecture and transport

The model has **36,313,947 parameters** across 158 parameter tensors. Group counts are `{"new": {"parameters": 4254102, "tensors": 53}, "pretrained_backbone": {"parameters": 23454912, "tensors": 53}, "pretrained_fpn_heads": {"parameters": 8604933, "tensors": 52}}`. The complete tensor-by-tensor transferred/new/frozen inventory and hashes are in `SCIENTIFIC_REGISTRATION.json` and `STRUCTURAL_QUALIFICATION.json`.

One normalized `[B,7,448,768]` tensor enters a single concatenated seven-channel convolution. The front returns only raw `[B,256,112,192]` FP32 C2. Identity transport is exact and carries 22,020,096 bytes (21.0 MiB) per frame. The edge accepts C2 plus calibration/metadata and has no RGB, radar, GT depth, or semantic-GT argument.

Monolithic/identity-split parity: C2 exact=True, same storage=True; front/tail/monolithic latency evidence is `{"front": {"median_ms": 0.8163039982318878, "p95_ms": 0.8538767993450164}, "iterations": 30, "monolithic": {"median_ms": 15.27843189239502, "p95_ms": 16.141089344024657}, "payload_copy_serialization": {"median_ms": 2.762268763035536, "p95_ms": 10.710224369540809, "serialized_bytes": 22020096}, "tail": {"median_ms": 14.46782398223877, "p95_ms": 15.391617631912231}, "transport_identity_gpu": {"median_ms": 0.00508800009265542, "p95_ms": 0.005396800069138407}, "warmup": 10}`.

## Assignment and geometry audits

| Class | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---:|---:|---:|---:|---:|---:|
| vehicle | 218799 | 253659 | 69302 | 33496 | 1934 | 0 |
| person | 94577 | 54952 | 927 | 146 | 43 | 0 |

P2 introduced 313,376 foreground locations without changing P3-P7. Total foreground locations were 727,835; actors without a carrier: 677. Carrier visibility counts were `{"elsewhere": 169250, "occluder": 103138, "own_visible": 455447}`. P2 FCOS-loss fractions are `{"bbox_ctrness": 0.4561597246793099, "bbox_regression": 0.5676060940604657, "classification": 0.7442930484658064}`.

Every geometry gather retained `(image, level, flattened point, internal class)` through filtering, top-k, concatenation, classwise NMS, and truncation. Synthetic adversarial and real-train lineage evidence is in structural qualification. Depth uses 32 log1p-spaced 0-40 m bins plus overflow and bounded `0.5*tanh` residual; physical ray plus intrinsics/extrinsics analytically yields XYZ; dimensions train directly in log space; yaw is independently normalized.

## Qualification and training

Loss-gradient calibration medians were `{"A": 0.00032373462268602395, "D": 0.10454584799303046, "G": 0.012994133748441675, "S": 4.5583782205237065e-05}` and fixed multipliers were `{"A": 10.0, "D": 1.0, "G": 4.022809446823192, "S": 10.0}`. The qualified physical batch was 4 with accumulation 4 to effective batch 16 under a 12288 MiB cap.

Disposable qualification processed all 16,827 epoch-1 frames and then 32 joint-stage updates, checking losses, gradients, parameters, and optimizer state after every update. Its state is archived under `QUALIFICATION_ONLY_DO_NOT_USE`; the scientific model was reconstructed at hash `82b48c761a6805b39c54e435a189389a552d58b9533b7ebfb254fbcfc9d47098` with a fresh empty optimizer.

Exactly 26 scientific epochs completed. Per-epoch raw/weighted loss curves are in `TRAINING_CURVES.csv`; full update records and P2 loss fractions are under `training_metrics/`. Fixed two-batch C2 gradient norms/cosines and separate RGB/radar stem evidence are under `gradient_telemetry/`.

## Fixed v0.10 validation

| Epoch | V P | V R | V F1 | P P | P R | P F1 | V XY m | P XY m | V IoU | P IoU | FG mIoU | Gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.0593 | 0.9524 | 0.1116 | 0.0342 | 0.7503 | 0.0654 | 0.6944 | 1.0441 | 0.7856 | 0.2000 | 0.4928 | 3/9 |
| 8 | 0.1985 | 0.9398 | 0.3278 | 0.0441 | 0.8649 | 0.0839 | 0.6163 | 0.9351 | 0.8741 | 0.4569 | 0.6655 | 5/9 |
| 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 0.0000 | 0.0000 | 0.0000 | 0/9 |
| 22 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 0.0000 | 0.0000 | 0.0000 | 0/9 |
| 26 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 0.0000 | 0.0000 | 0.0000 | 0/9 |

| Epoch | V TP | V FP | V FN | V R@.02 | P TP | P FP | P FN | P R@.02 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 9230 | 146460 | 461 | 0.9539 | 2905 | 82111 | 967 | 0.7588 |
| 8 | 9108 | 36778 | 583 | 0.9665 | 3349 | 72635 | 523 | 0.9070 |
| 16 | 0 | 0 | 9691 | 0.0000 | 0 | 0 | 3872 | 0.0000 |
| 22 | 0 | 0 | 9691 | 0.0000 | 0 | 0 | 3872 | 0.0000 |
| 26 | 0 | 0 | 9691 | 0.0000 | 0 | 0 | 3872 | 0.0000 |

| Epoch | V dim MAE m | V yaw MAE deg | P dim MAE m | P yaw MAE deg |
|---:|---:|---:|---:|---:|
| 3 | 0.2768 | 76.6578 | 0.0213 | 91.6260 |
| 8 | 0.2377 | 58.2768 | 0.0269 | 87.5975 |
| 16 | NA | NA | NA | NA |
| 22 | NA | NA | NA | NA |
| 26 | NA | NA | NA | NA |

Every checkpoint used one score-floor-0.02 inference pass; score-0.20 metrics were derived from the retained predictions. Each epoch JSON includes TP/FP/FN, recall at 0.02, dimension/yaw errors, ignore reconciliation, FPN attribution, duplicate/cross-level/background FP taxonomy, 2D-correct/world-wrong cases, person point misses, and distance/radar/visibility slices.

Selected-checkpoint duplicate and error taxonomy: `{"background_fp": {"person": 41241, "vehicle": 15721}, "cross_level_duplicate_fp": {"person": 22446, "vehicle": 10455}, "duplicate_fp": {"person": 39068, "person_cross_level": 22446, "vehicle": 19451, "vehicle_cross_level": 10455}, "geometry_errors": {"person": {"dimension_mae_m": 0.02691869115182899, "dimension_x_mae_m": 0.007773741772578602, "dimension_y_mae_m": 0.007737194950774438, "dimension_z_mae_m": 0.06524513673213397, "matched": 3349, "yaw_mae_deg": 87.59749257110867}, "vehicle": {"dimension_mae_m": 0.23766953356651127, "dimension_x_mae_m": 0.39115989400478635, "dimension_y_mae_m": 0.13506842428566648, "dimension_z_mae_m": 0.18678028240907985, "matched": 9108, "yaw_mae_deg": 58.2767841510046}}, "person_centre_point_miss": {"centre_point_miss": 189, "centre_present_world_wrong_or_contention": 334}, "two_d_correct_world_wrong": {"person": 2316, "vehicle": 5386}}`.

## Selected service gates

| Target | Value | Requirement | Attainment | Pass |
|---|---:|---:|---:|:---:|
| foreground_miou | 0.6655 | higher 0.675 | 0.9859 | no |
| person_box_mask_iou | 0.4569 | higher 0.5 | 0.9138 | no |
| person_precision | 0.0441 | higher 0.8 | 0.0551 | no |
| person_recall | 0.8649 | higher 0.8 | 1.0812 | yes |
| person_xy_mae_m | 0.9351 | lower 1.2 | 1.2833 | yes |
| vehicle_iou | 0.8741 | higher 0.85 | 1.0283 | yes |
| vehicle_precision | 0.1985 | higher 0.8 | 0.2481 | no |
| vehicle_recall | 0.9398 | higher 0.85 | 1.1057 | yes |
| vehicle_xy_mae_m | 0.6163 | lower 1.0 | 1.6227 | yes |

Selected v0.25 sensitivity: `{"person_f1": 0.08018162393162394, "person_precision": 0.04198366524949653, "person_recall": 0.889218009478673, "person_recall_002": 0.9218009478672986, "person_xy_mae_m": 0.8927891249667278, "vehicle_f1": 0.3276282640860559, "vehicle_precision": 0.19680991616421245, "vehicle_recall": 0.9771019677996422, "vehicle_recall_002": 0.9835420393559928, "vehicle_xy_mae_m": 0.5582925745166023}`. This was run only for the selected checkpoint and did not affect selection.

## Supervisor architecture story

This run isolates a clean scientific question: whether an official FCOS ResNet-50 detector, split exactly at raw C2 and extended with a non-destructive overlapping P2 plus task-private dense/geometry heads, can meet Route B service targets while preserving one seven-channel input and one learned payload. The fixed five-checkpoint result and failure taxonomy are retained even when the outcome is not service-ready; no follow-on architecture was launched.
