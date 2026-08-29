# Route B v3.1 depth-aware LR-ASPP — terminal report

**Terminal verdict: `DEPTH_AWARE_RUNTIME_FAILURE`**

Exactly one clean-lineage scientific model was started. It became non-finite at epoch 1, batch 14, before the first epoch checkpoint. The registered recovery exception therefore cannot apply: there is no prior exact checkpoint and the failed in-memory optimizer state no longer exists. No retry or scientific change was made. This runtime-failed attempt is not scientific evidence against LR-ASPP.

## Lineage, model, and pretrained weight

- Frozen source lineage: local `master` at `fa9b1e1b4f3af8f64de458a29df09ae1d7093b39`; required-ancestor check passed.
- Official backbone: `MobileNet_V3_Large_Weights.IMAGENET1K_V2` from `https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth`; 22,132,113 bytes; SHA-256 `5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`.
- Software: Python `3.10.12`, PyTorch `2.10.0.dev20251114+cu128`, torchvision `0.25.0.dev20251117+cu128`.
- Architecture: one dilated MobileNetV3-Large fused trunk; a shared low/high depth-aware stride-4 neck; segmentation and training-only dense-depth readouts; private vehicle/person heatmap and factorized geometry branches. Actor XYZ is derived only from physical-centre ray plus the 32-bin log-depth distribution—there is no learned XYZ head.

| Module | Parameters |
|---|---:|
| model | 4,174,643 |
| backbone | 2,972,528 |
| rgb_stem | 432 |
| radar_stem | 576 |
| depth_neck | 738,272 |
| segmentation | 74,051 |
| dense_depth | 73,857 |
| vehicle_branch | 158,032 |
| person_branch | 157,903 |

| Optimizer group | Tensors | Parameters |
|---|---:|---:|
| backbone_decay | 62 | 2,942,472 |
| backbone_no_decay | 108 | 29,480 |
| new_decay | 33 | 1,200,704 |
| new_no_decay | 40 | 1,987 |

## Input, stem, and split proofs

- Deployable input is seven channels: RGB in RGB order, scaled to `[0,1]` and ImageNet-normalized, followed by the prepared identity-normalized radar occupancy, inverse-range, radial-velocity, and stationary-age channels. The real-sample PIL RGB versus OpenCV BGR-reversal check passed.
- The bias-free official RGB convolution and exact-zero bias-free radar convolution concatenate to `[16, 7, 3, 3]`. Direct and concatenated FP32 outputs were equal with maximum absolute delta `0.0`.
- Transport is identity/disabled: `low [1, 40, 54, 96]` and `high [1, 960, 27, 48]`, both FP32; raw and serialized batch-1 sizes are each 5,806,080 bytes.
- Tail input keys were exactly `['low', 'high']`; all raw outputs were `torch.equal`; 240 decoded records were byte-identical and externally schema-compatible.
- The inference dataset/model signatures have no depth-label input. A nonexistent in-memory depth-path sentinel caused zero open attempts and byte-identical input/prediction behavior.

## Data, cache, radar, and qualification

- Authoritative manifest SHA-256 `5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2`: 16,827 train frames/10 episodes and 3,345 validation frames/two disjoint episodes; zero test rows.
- v0.10 object hashes: train `cafa517f84416f1f58cfecac42d059279c1367769b37649ef2db03d2dcb40423`, validation `fccf42b17c7468a85bfe367a209fb205a992e6e39b3a0300c5eb9c8b47a6cb08`. Visible-anchor SHA-256 `fdbe71ded09f042b9a16e9c6ce6e1543b7ea3160ff2e7bad0a3be0f1a651b653`.
- Train cache: 16,827 `sample_id` entries, 268,779,338 finite valid depth cells, and 38,570,270 retained current-sweep radar consistency points. Depth F16 SHA-256 `ec75d0a776097f6fb8a582e98e1fe907a7d0032267c1fcae27eb5a8937bf00ed`; valid-mask SHA-256 `5ec480cb3d2eefa9cba3d35368484ae22d09e253dbc337d7ba10230b67304ee8`; radar-cache SHA-256 `89f630ddeb32fbf41c83eed42b7ae7dd78ea10c492984fe6ae2764483ebbbdcf`.
- Depth synchronization had exact frame IDs and zero timestamp delta. Retained radar `camera_depth_m` delta was `0.0` m; current-sweep transform max delta was `0.00002214` m.
- Qualification passed in 8.1 s. Physical batch 16 × accumulation 1 was accepted at 8536.3 MiB allocated/9902.0 MiB reserved.
- The disposable 80-step overfit gates fell: person heatmap `112905.314062` → `9.208499`, person actor depth `19.653853` → `1.371711`, dense depth `17.744527` → `5.390108`. All disposable state was discarded.
- Stage-A clone proof kept official state bit-identical and gave finite nonzero gradients to every required new group. Geometry round-trip maximum error was `1.421e-14` m.
- Same-class collisions: train person 30/17587 and vehicle 58/46745; validation person 17/3872 and vehicle 14/9691. Cross-class overwrites and silent truncations were zero.

## Scientific runtime failure

- Exact exception: `nonfinite scientific loss epoch=1 batch=14`.
- Completed epoch boundaries: **0**. Optimizer updates before the failed forward/loss check: **13** (control-flow inference). Atomic checkpoints: **0**. Exact resume possible: **no**.
- A read-only reconstruction of the failed batch found all inputs and targets finite: True; 16 samples, 257,623 valid dense cells, and 35,291 consistent radar points. It performed no forward, backward, or optimizer step.
- No per-epoch loss, denominator, LR, clipping, or gradient-telemetry record exists because the exception preceded the first epoch boundary. Training wall time was not durably instrumented before the exception; the terminalization upper bound from `TRAINING_STARTED` creation is recorded in `SCIENTIFIC_RUNTIME_FAILURE.json`.

## Validation, selection, and gates

| Epoch | Prediction | v0.10 evaluation | Reason |
|---:|---|---|---|
| 10 | not run | unavailable | checkpoint absent |
| 20 | not run | unavailable | checkpoint absent |
| 30 | not run | unavailable | checkpoint absent |
| 40 | not run | unavailable | checkpoint absent |

Baseline/reference deltas, actor-depth and derived-XYZ slices, auxiliary dense-depth slices, detection/world-error taxonomy, latency, and the nine service/material gates are unavailable because there is no completed checkpoint or prediction. Forty-epoch preservation eligibility is false by construction. Selected checkpoint: **none**. v0.25 sensitivity was not licensed or run. Validation depth was never opened and no validation cache was built.

## Scope and durable completion

- Current branch at terminalization: `master`. Protected pre-existing dirty path `OAI/openairinterface5g` remains the sole out-of-scope status entry and was not modified by this work.
- Test payloads, CARLA, OAI execution, q/AE, live split inference, and the 288 measurements were untouched. No branch, push, pull, merge, rebase, architecture/loss/sampler variant, scientific retry, or non-identity compression was used.
- Commit allowlist (and nothing else):

  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/__init__.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/build_depth_cache.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/common.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/config.json`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/data.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/decode.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/evaluate.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/finalize.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/finalize_runtime_failure.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/infer.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/losses.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/model.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/qualify.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/run_pipeline.py`
  - `pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_depth_aware_lraspp_v1/train.py`
  - `experiments/route_b_v3_1_depth_aware_lraspp_v1/20260829_042423/FINAL_REPORT.md`

The terminal report is included in the local master commit; its commit hash is reported in the external handoff to avoid a self-referential hash. Checkpoints, predictions, caches, and JSON payloads are intentionally uncommitted.

Desktop notification attempted: `True`; delivered: `True`; return code: `0`. Completion sentinel: present.
