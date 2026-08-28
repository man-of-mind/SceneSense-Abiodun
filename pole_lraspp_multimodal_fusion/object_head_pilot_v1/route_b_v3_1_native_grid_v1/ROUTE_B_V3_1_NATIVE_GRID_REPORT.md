# Route B v3.1 — native stride-4 LR-ASPP object-grid correction

**Terminal: `LRASPP_NATIVE_GRID_MATERIAL_GAIN_NOT_SERVICE_READY`**

Experiment: `experiments/route_b_v3_1_native_grid_v1/20260828_042729`
Package: `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_v1/`

One bounded correction to the object target/output geometry. The 7-channel input, the
MobileNetV3 LR-ASPP backbone, the segmentation path, the transported low/high split
bundle and the external object record are all unchanged.

---

## 1. Terminal

`LRASPP_NATIVE_GRID_MATERIAL_GAIN_NOT_SERVICE_READY`

All five registered material-gain gates pass on the selected checkpoint. Seven of the
nine advisory service targets are still unmet, so the pilot is not service-ready.

## 2. Structural diagnosis (verified before implementation)

`PHASE_A_DIAGNOSIS.json` — **9/9 checks confirmed.**

Source evidence, in the unchanged shared code:

| Claim | Evidence |
|---|---|
| Object head predicts at stride 8 | measured `object_head` output `(1,14,54,96)` for a 768×432 input |
| Predictions bilinearly enlarged to full resolution | `pole_lraspp_multimodal_fusion/model.py:306` |
| Targets allocated at full input resolution | `object_targets.py:194`, `:195` |
| Target centre rounded to a full-resolution pixel | `object_targets.py:206` |
| Decoder top-k over the enlarged map | `object_targets.py:490` |
| Decoder uses a box-occupancy NMS, not a local maximum | `object_targets.py:503` |
| Regression read at an enlarged (interpolated) pixel | `object_targets.py:506` |

Each expected consequence was measured, not assumed:

1. **One native response becomes a broad interpolated patch.** A single median-sized
   vehicle response (54.1 × 31.0 px), drawn with the unchanged target code at stride 8
   and enlarged, covers **295 full-resolution pixels** above score 0.20.
2. **Top-k + 2 px NMS emits repeated detections from that patch.** That single object
   response decodes to **5 detections**.
3. **Interpolated regression mixes neighbouring native values.** With adjacent native
   cells carrying `local_x` = 20 and 40, the decoded detections return 21.25 and 33.75 —
   values that exist in no native cell.
4. **Confirmed on the real model, not only synthetically.** Across the retained epoch-20
   validation predictions there are **6,679** vehicle pairs within 2 m of a
   higher-scoring vehicle (the registered `PREDICTED_DUPLICATE` condition). Their image
   separation is **median exactly 5.0 px** — the floor the 2-px NMS box permits —
   with **75.0% pinned at that floor**, **86.8% inside one stride-8 cell** and 94.9%
   within two. Duplicates are emitted at the decoder's own minimum spacing, from inside
   one interpolated patch.
5. **Small pedestrians occupy about one stride-8 cell or less.** Median person footprint
   is **2.66 stride-8 cells** (median width 7.7 px, i.e. under one 8-px cell); **16.6%**
   occupy ≤ 1 cell and **65.8%** ≤ 4 cells. At stride 4 the median becomes 10.6 cells.

## 3. Architecture, and why it stays split-inference compatible

```
UE FRONT (unchanged)
  7ch input (RGB 3 + radar raster 4)
  MobileNetV3-Large LR-ASPP backbone
  -> {low: 40ch @ 54x96 (stride 8), high: 960ch @ 27x48 (stride 16)}   <-- TRANSPORTED

EDGE TAIL
  classifier(low, high) -> segmentation                                 <-- unchanged
  _object_input(low, high) -> concat = 1000ch @ 54x96                   <-- unchanged
    shared_trunk: 3 x [Conv3x3 -> BN -> ReLU], 1000->128                <-- warm-started
    [upsampler:  ConvTranspose2d(128,128,k4,s2,p1) -> BN -> ReLU]       <-- NEW -> 108x192
    vehicle_heatmap_head  Conv1x1(128, 1)                               <-- warm-started
    person_heatmap_head   Conv1x1(128, 1)                               <-- warm-started
    regression_head       Conv1x1(128, 12)                              <-- warm-started
    [offset_head          Conv1x1(128, 2)]                              <-- NEW, private
  -> 16 channels at NATIVE 192x108 (output stride 4). Never enlarged.
```

Compatibility, verified by the launch check rather than asserted:

- `encode_front` returns exactly `{low, high}` — the same bundle as today. Nothing was
  added to the transported payload, so q / quantization / AE / zstd operate unchanged.
- The tail is a pure function of that bundle: monolithic forward and explicit
  encode/decode agree to **max abs delta 0.0** on both `out` and `object` in eval mode.
- No raw RGB or radar reaches the tail.
- The two offset channels are internal and consumed by the decoder; the exported
  detection CSV schema is field-identical to v3.1, so the downstream spatial-map object
  contract is untouched.

Parameters: **4,668,540 → 4,931,198**, i.e. **+262,658 (+5.6%)**, all tail-side
(upsampler 262,400 + offset head 258). The front-side cost is exactly zero.

The upsampler is initialised with a per-channel bilinear kernel, so at step zero the
block reproduces a bilinear 2× upsample (max interior deviation **2.41e-4**) under the
frozen-BN recipe. The warm-started 1×1 heads therefore start by emitting the baseline's
behaviour expressed on the stride-4 grid, and training moves it toward sharp native
peaks. The offset head is initialised to a constant 0.5 — the cell centre.

Note, out of scope but worth recording: `pole_lraspp_multimodal_fusion/split_runtime.py`
(the live UDP path) still bilinearly enlarges object maps. It was not modified. It would
need this native decoder before this head is deployed on that path.

## 4. Warm-start tensor mapping

From the verified epoch-20 checkpoint (SHA-256
`88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d`, matched exactly).
339 source tensors → 351 target tensors.

| Class | Count | Tensors |
|---|---:|---|
| **Loaded** (name and shape identical) | 319 | `backbone.*` (308), `classifier.*` (11) |
| **Transformed — renamed** | 18 | `object_head.{0,1,3,4,6,7}.*` → `object_head.shared_trunk.{...}` (3 conv + 3 BN, shapes unchanged) |
| **Transformed — output-channel slice** | 6 | `object_head.9.{weight,bias}` (14ch) split into `vehicle_heatmap_head` `[0:1]`, `person_heatmap_head` `[1:2]`, `regression_head` `[2:14]` |
| **New** | 8 | `object_head.upsampler.0.weight`, `upsampler.1.{weight,bias,running_mean,running_var,num_batches_tracked}`, `offset_head.{weight,bias}` |
| **Incompatible** | 0 | — |

The baseline head is `head_arch="shared"` — a single Sequential whose final 1×1 conv
emits all 14 channels. Splitting it into separate vehicle and person branches is
therefore a pure output-channel slice, which preserves every learned output weight.

## 5. Parameter counts by stage

| Group | Total | Trainable, Stage H | Trainable, Stage J |
|---|---:|---:|---:|
| backbone | 2,972,528 | 0 | 2,948,128 |
| classifier | 246,526 | 0 | 246,270 |
| object_head trunk | 1,447,680 | 1,446,912 | 1,446,912 |
| object_head upsampler *(new)* | 262,400 | 262,144 | 262,144 |
| vehicle heatmap head | 129 | 129 | 129 |
| person heatmap head | 129 | 129 | 129 |
| regression head | 1,548 | 1,548 | 1,548 |
| offset head *(new)* | 258 | 258 | 258 |
| **Model total** | **4,931,198** | **1,711,120** (frozen 3,220,078) | **4,905,518** (frozen 25,680) |

Frozen counts under `freeze_bn: true` are BatchNorm affine parameters, per the frozen
v3.1 stage-2 recipe.

## 6. Epoch table — v0.10 primary, score 0.20 (recall also at 0.02)

| Epoch | Eligible | veh P | veh R | veh F1 | per P | per R | per F1 | mean F1 | veh MAE | per MAE | fg mIoU | veh IoU | per IoU | veh R@.02 | per R@.02 |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | no | 0.6837 | 0.8071 | 0.7403 | 0.4359 | 0.4734 | 0.4539 | 0.5971 | 1.0864 | 1.4209 | 0.6514 | 0.8640 | 0.4388 | 0.8573 | 0.5999 |
| 6 | yes | 0.7498 | 0.7984 | 0.7733 | 0.4003 | 0.4731 | 0.4337 | 0.6035 | 1.0152 | 1.3947 | 0.6512 | 0.8640 | 0.4384 | 0.8398 | 0.5785 |
| 9 | yes | 0.7526 | 0.7965 | 0.7739 | 0.4342 | 0.4695 | 0.4512 | 0.6125 | 1.0026 | 1.3941 | 0.6529 | 0.8642 | 0.4416 | 0.8420 | 0.5754 |
| 12 | yes | 0.7663 | 0.7971 | 0.7814 | 0.4187 | 0.4811 | 0.4477 | 0.6146 | 1.0085 | 1.4068 | 0.6535 | 0.8645 | 0.4424 | 0.8421 | 0.5744 |
| **15** | **yes** | **0.7112** | **0.8080** | **0.7565** | **0.4950** | **0.4641** | **0.4791** | **0.6178** | **0.9847** | **1.3961** | **0.6546** | **0.8655** | **0.4437** | **0.8460** | **0.5607** |
| baseline ep20 | — | 0.4810 | 0.7043 | 0.5716 | 0.4613 | 0.4029 | 0.4301 | 0.5009 | 0.9981 | 1.4088 | 0.6514 | 0.8640 | 0.4388 | 0.7461 | 0.4561 |

Epoch 3 is ineligible on one gate only: vehicle XY MAE 1.0864 > 1.02.
**Selected: epoch 15**, by highest mean class F1 (0.6178) among eligible checkpoints.

Training-loss note, recorded rather than smoothed over: train loss falls monotonically
(3.52 → 0.97) while the validation *loss* rises (8.62 → 11.27), driven by the focal
centre term on a sparse grid. The validation *detection metrics* improve monotonically
over the same span. Selection is by the registered detection contract, not by loss, and
no early stopping was used — both fixed before training started.

## 7. FP/FN taxonomy — baseline vs selected (registered v3.1 taxonomy, unchanged)

Vehicle false positives at score 0.20 — **total 7,390 → 3,191 (−56.8%)**:

| Label | Baseline | Selected | Δ |
|---|---:|---:|---:|
| `PREDICTED_DUPLICATE` | 3,580 (48.44%) | **991 (31.06%)** | **−2,589 (−72.3%)** |
| `TWO_D_CORRECT_WORLD_WRONG` | 2,549 (34.49%) | 1,705 (53.43%) | −844 (−33.1%) |
| `BACKGROUND_OR_OTHER` | 1,261 (17.06%) | 495 (15.51%) | −766 (−60.7%) |

Person false negatives at score 0.02 — **total 2,106 → 1,701 (−19.2%)**:

| Label | Baseline | Selected | Δ |
|---|---:|---:|---:|
| `HEATMAP_CENTER_MISS` | 1,467 (69.66%) | **685 (40.27%)** | **−782 (−53.3%)** |
| `CENTER_PRESENT_WORLD_WRONG` | 534 (25.36%) | 854 (50.21%) | +320 |
| `MATCHING_CONTENTION` | 105 (4.99%) | 162 (9.52%) | +57 |

Every label sums exactly to its scorer denominator in both arms. The two rising
categories are the informative result: pedestrians the stride-8 grid could not represent
at all are now detected in 2D, but a substantial share of them still carry a world
position outside the 3 m match radius, so they convert from "missed entirely" to
"found, mislocated" rather than to true positives.

The registered scorer was reconciled against the published epoch-20 counts before use
and reproduces them bit-exactly (vehicle 6849/7390/2876/2744, person 1560/1822/2312/925,
and the 48.44 / 34.49 / 69.66 taxonomy percentages).

## 8. Material-gain gates and v0.25 sensitivity

Registered material gain — **all five pass**:

| Gate | Required | Achieved |
|---|---|---|
| mean class F1 ≥ 0.5309 (+0.03 over 0.5009) | 0.5309 | **0.6178** (+0.1169) |
| person F1 ≥ 0.4501 **or** person R@0.02 ≥ 0.5061 | either | **both**: F1 0.4791, R@0.02 0.5607 |
| vehicle precision ≥ 0.5310 (+0.05) | 0.5310 | **0.7112** (+0.2302) |
| vehicle recall ≥ 0.6943 (loss ≤ 0.01) | 0.6943 | **0.8080** (recall *rose* +0.1037) |
| vehicle duplicate FP reduction ≥ 30% | 30% | **72.3%** |

v0.25 sensitivity, selected checkpoint only (report-only, not a gate):

| Score | Class | P | R | F1 | XY MAE | TP/FP/FN | eligible GT |
|---|---|---:|---:|---:|---:|---|---:|
| 0.20 | vehicle | 0.7220 | 0.8827 | 0.7943 | 0.9433 | 7402/2850/984 | 8,386 |
| 0.20 | person | 0.4975 | 0.5071 | 0.5023 | 1.3947 | 1712/1729/1664 | 3,376 |
| 0.02 | vehicle | 0.3605 | 0.9038 | 0.5154 | 0.9262 | 7579/13446/807 | 8,386 |
| 0.02 | person | 0.1558 | 0.5924 | 0.2468 | 1.4275 | 2000/10834/1376 | 3,376 |

The v0.25 view moves in the same direction as v0.10; no reversal.

## 9. Service targets (advisory)

| Target | Required | Achieved | |
|---|---:|---:|:--|
| vehicle precision | ≥ 0.80 | 0.7112 | FAIL |
| vehicle recall | ≥ 0.85 | 0.8080 | FAIL |
| person precision | ≥ 0.80 | 0.4950 | FAIL |
| person recall | ≥ 0.80 | 0.4641 | FAIL |
| vehicle XY MAE | ≤ 1.0 m | 0.9847 | **PASS** |
| person XY MAE | ≤ 1.2 m | 1.3961 | FAIL |
| vehicle IoU | ≥ 0.85 | 0.8655 | **PASS** |
| person box-mask IoU | ≥ 0.50 | 0.4437 | FAIL |
| foreground mIoU | ≥ 0.675 | 0.6546 | FAIL |

2 of 9 met.

## 10. Wall time and VRAM

| Phase | Wall time | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Training (15 epochs, H+J) | 943.0 s (0.26 h) | 4,385.6 MiB | 5,102.0 MiB |
| Evaluation (5 inference passes + scoring) | 646.6 s (0.18 h) | 95.9 MiB | 156.0 MiB |
| **Pipeline total** | **26 min 32 s** | — | — |

Per-epoch 48.9–78.9 s. Per inference pass 103.5–119.6 s over 3,345 frames. Batch size 16
was fixed once to the frozen v3.1 stage-2 value so the geometry correction is not
confounded with an optimisation change; the launch check only verified it fits (5,102 of
32,607 MiB). No batch-size sweep was run.

## 11. Selected checkpoint

```
experiments/route_b_v3_1_native_grid_v1/20260828_042729/checkpoints/
  route_b_v3_1_native_grid_v1/epoch_015.pt
SHA-256: 1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed
```

## 12. Scope confirmation

- **Test split: untouched.** `splits/test.txt` is empty (0 rows); only `contracts/v010`
  and `contracts/v025` `val/` exist and were read. The inference script hard-rejects any
  `canonical_v3_07`/`canonical_v3_08` sample id (`infer_native_v1.py:91`).
- **CARLA: not started.** No CARLA process ran; no simulator client was launched.
- **OAI: not started.** No containers running; the dirty `OAI/openairinterface5g`
  submodule pointer is preserved byte-for-byte and was never staged or modified.
- **q / AE: not trained.** `ae_bottleneck = 0`, `feature_drop_max = 0.0`,
  `feature_drop_val = 0.0` throughout. Clean q=0, no AE.
- **288 measurements: not run.**
- **Baseline preserved.** The epoch-20 checkpoint still hashes to
  `88b34a69eeec…d747aa1d`. No existing experiment, checkpoint or canonical v3/v3.1 data
  was modified; the new package and experiment directory are entirely additive, and all
  run outputs were written create-only.

## 13. Conclusion

The failure decomposition of record was caused by the output geometry, not by the class
balance and not by the capacity of the transported bundle. Moving the object target,
head and decoder onto the native stride-4 grid — adding one 262 k-parameter tail-side
upsampler and a two-channel centre-offset branch, with no change to the 7-channel input,
the segmentation path or the transported low/high features — eliminated **72.3%** of
vehicle duplicate false positives and **53.3%** of pedestrian heatmap centre misses, and
raised mean class F1 from 0.5009 to 0.6178 while vehicle recall *increased* and both XY
MAEs improved.

On the question the brief asks to answer if the correction still fell short: the
remaining evidence does **not** indicate that the transported low/high bundle lacks
sufficient small-object spatial information. The same bundle, decoded at stride 4,
recovered 782 pedestrians the stride-8 decoder missed entirely and cut vehicle FP by
more than half. The residual gap has moved elsewhere and is now specific: 50.2% of
remaining person false negatives are `CENTER_PRESENT_WORLD_WRONG` (up from 25.4%) and
53.4% of remaining vehicle false positives are `TWO_D_CORRECT_WORLD_WRONG` (up from
34.5%). In both classes the object is now found in the image and lost in the
image-to-world regression — person XY MAE is essentially unmoved (1.4088 → 1.3961 m,
against a 1.2 m target). The binding constraint is metric depth/world-position accuracy,
not spatial resolution in the transported features.

Stopping here for review. No follow-up experiment is proposed, no gate was relaxed, and
the locked test split remains unopened.
