# Numerical operation audit

This is the equation-to-code review map for the active original model/loss/evaluator and the recovery replacement. `operation_inventory.py` AST-enumerates numerical divisions, matrix projections, normalization/norm, log/exp, sqrt, softmax, and box-decode calls so review can be repeated against the exact committed source. Filesystem `/` operators are not scientific operations and are excluded from the interpretation below. The implementation audit found 134 source occurrences; the implementation packet records that count and the inventory command.

## Registered loss map

- D: original `losses.py:154–164`. Sigmoid focal sum is divided by the positive denominator; FCOS deltas are decoded by `model.box_coder.decode`; GIoU sum is divided by the same denominator; centerness is the square root of clamped left/right and top/bottom minimum-to-maximum ratios. Unchanged.
- G: recovery `recovery_losses.py:36–84`, corresponding to original `losses.py:220–264`. Depth-bin cross entropy is unchanged. Residual is `0.5*tanh(raw)` and Smooth-L1. Ray target divides center displacement by anchor size, now with a zero-denominator check. Depth uses `log1p` edges, `expm1` bin decode clipped to the registered physical `[0,40]` range, softmax expectation plus overflow depth 40. Local projection divides by checked `fx/fy`. Endpoint uses unchanged `/3` scaling. Dimension target uses checked-positive `log`. Predicted log-dimension loss is unchanged. Only yaw normalization calls the shared recovery function; Smooth-L1 target and weight `0.15` are unchanged.
- S: original `losses.py:20–56`. Lovasz uses intersection divided by union clamped at `1e-6`, and class probabilities use FP32 softmax; weighted CE and `0.5` Lovasz composition are unchanged.
- A: original `losses.py:276–295`. Dense target uses `log1p`; radar consistency sum divides by `max(1, radar_count)` and retains weight `0.5`. Unchanged.
- Group total: recovery `recovery_losses.py:105–125`, corresponding to original `losses.py:298–325`. D/G/S/A multipliers and pressure fractions are unchanged and never recalibrated.

## Assignment, box, score, and projection map

- Original `losses.py:67–105`: GT/anchor centers divide by two; scale and center-sampling assignment remain unchanged.
- Original `model.py:331`: score is exactly `sqrt(sigmoid(class_logit)*sigmoid(centerness_logit))`.
- Original `model.py:336–342`: candidate index uses integer `torch.div(..., rounding_mode="floor")`; class index uses remainder; box regression uses the unchanged FCOS box coder followed by image clipping.
- Recovery `recovery_model.py:22–48`, corresponding to original `model.py:285–316`: depth `log1p/expm1`, anchor-center `/2`, pixel ray, checked `fx/fy` divisions, and FP64 homogeneous extrinsic matrix projection are unchanged. The registered depth clamp is not a dimension clamp.
- Dimension decode calls `exp_dimensions_fp64` (`safe_math.py:67–83`): input is FP64, finite and within FP64 exponential range, output must be finite/positive, and no fallback or physical clamp exists.
- Yaw decode and yaw loss both call `normalize_yaw_fp32` (`recovery_model.py:44`, `recovery_losses.py:63`). The function (`safe_math.py:30–64`) converts raw yaw to FP32, forms a scaled overflow-safe two-component L2 norm, divides by `clamp_min(norm,tau)`, fails on invalid tau/raw/result, and reports raw-norm/affected diagnostics. There is no direction fallback or magnitude loss.

## Empty and extreme states

- No candidates, zero detections, and empty geometry tensors remain valid empty tensors; the checked FP64 exponential accepts shape `[0,3]` and tensor audits accept finite empty outputs.
- Nonfinite raw yaw, nonfinite log-dimensions, zero projection denominators, zero anchor ray denominators, and FP64-exp range violations fail closed.
- Near-zero yaw uses the preregistered tau equation; ordinary and large finite yaw are covered by synthetic autograd tests.
- Inputs, C2, P2–P7, ResNet features, all detection/geometry/semantic/dense heads, loss parts, decoded geometry, scores, and boxes are recursively audited for dtype, shape, finite status, min, max, and absmax. Forward hooks cover C2/FPN/head boundaries during failure replay.

## Evaluator map

The evaluator is not reimplemented. `evaluate_recovered.py` creates a provenance-labeled combined view and invokes unchanged original `evaluate.py` (SHA-256 `107908651aa7e43aae1c3b55b288fa03f62beb59e0f86d03503144dd8b5cbdd8`), `scoring_v2.py` (`afba5c175107ffeed28363a59afcddbd96c58cd7efc7a092072a13ebee9af9b5`), native `evaluate_v1.py` (`3791a309f45b822d6142939718354b3bdf2368dc0cb473df9d38027ba3268379`), `score_contract_v1.py` (`14629e69a1617d05ca6dec2bad6901b69f83df96f2cf5543509dbe970d18069d`), and `audit_v1.py` (`64d7d677480328d64176a96b2e7bf7d1eda2e9eb12fc5aa6a74762f381c7727b`). Therefore all evaluator divisions remain byte-for-byte registered:

- box IoU intersection/guarded union (`evaluate.py:30`);
- source-to-model coordinate scale (`:39`) and box centers (`:103,123–124`);
- recall and diagnostic MAE means (`:141,147–151`);
- class-F1 means (`:194,240,293`);
- service attainment value/target or target/guarded-value (`:229`);
- normalized localization and ranking mean (`:241`).

The frozen scorer retains its registered precision, recall, F1, localization, segmentation, threshold, and sensitivity equations. Evaluation epochs and ordering are exactly 3, 8, 16, 22, 26; only 3/8 may come from the healthy original experiment, and only recovered 16/22/26 are admitted. Original epochs 10–26 are explicitly excluded and labeled corrupted. Sensitivity remains `v025_selected_only` after primary selection.

## Exhaustive occurrence index

The following is the complete 134-occurrence AST inventory for the active model/loss/evaluator/recovery numerical scope. `sensitive_call` rows are resolved to their equations in the sections above; the inventory function returns the exact expression for each row. Re-run with `python3 -c "from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1.operation_inventory import inventory; import json; print(json.dumps(inventory(), indent=2))"`.

- `evaluate.py`: 30 division; 39 division (x2); 103 division (x2); 123 division; 124 division; 141 division; 147–151 division; 194 division; 229 division (x2); 240 division; 241 division (x3); 293 division.
- `guards.py`: 14 streaming FP64 scalar vector norm; 96–97 proposed-update/parameter relative division and square roots; 102–111 streaming square roots for group/global gradient, momentum, update, parameter, and optimizer-state norms.
- `losses.py`: 25 division; 32 softmax; 55 Lovasz-Softmax; 67–68 division; 85 division; 154 division; 156 box decode; 157 division; 161 division and square root; 162–163 division; 196 division; 233 division; 235 division; 238 log1p (x2); 239 expm1; 240 softmax; 245–247 division (x4); 250 log; 251 normalize; 280 log1p; 291 division; 321 division.
- `model.py`: 98 GroupNorm; 297 log1p (x2); 299 expm1; 302 division; 307–308 division; 311 projection matrix multiply; 312 exp; 313 normalize; 331 square root; 338 integer division; 341 box decode. FrozenBatchNorm modules inherited from the official backbone are preserved and accounted by the state guard.
- `recovery_losses.py`: 44 division; 47 division; 50 log1p (x2); 51 expm1; 52 softmax; 57–59 division (x4); 64 log; 65 shared yaw normalize; 71 checked dimension exp; 74 world projection; 150 division.
- `recovery_model.py`: 23 softmax; 31 log1p (x2); 33 expm1; 35 division; 40–41 division; 43 world projection; 44 checked dimension exp; 45 shared yaw normalize.
- `safe_math.py`: 43 scaled-norm division; 44 square root; 48 clamped yaw division; 57 diagnostic fraction; 75–76 representable-range logarithms; 79 FP64 exponential.
- `scoring_v2.py`: 159 division; 245 division. All other frozen scoring equations are delegated unchanged through its native evaluator dependency and are hash-bound by the audit.
- `evaluate_v1.py`: 74 and 76 mean divisions; 145 and 152 taxonomy percentage divisions; 255 duplicate-reduction division.
- `score_contract_v1.py`: 48 IoU division; 93–97 precision/recall/F1 divisions; 137 box-mask IoU division; 163–168 precision/recall/F1 and localization means; 202 and 205 segmentation IoU divisions.
- `audit_v1.py`: 78–79 coordinate-scale divisions; 118–119 box-center divisions; 231–238 center/scale normalizations; 427 and 436 percentage divisions.

## Precision placement

| Region | Registered/recovery precision | Status |
|---|---|---|
| Seven-channel front and C2 boundary | FP32 | unchanged |
| Tail convolutions | BF16 autocast if qualified | unchanged; compared with FP32 on eight registered train batches |
| Detection, geometry, semantic, auxiliary losses | FP32 | unchanged |
| Yaw normalization | FP32 raw, norm, clamp, divide | shared recovery path |
| Depth log/unprojection | FP32 | unchanged plus denominator audits |
| Dimension exponential/world transform | FP64 | checked range; no clamp |
| Transport | raw FP32 noAE | unchanged |
| Evaluator | frozen original behavior | unchanged |
