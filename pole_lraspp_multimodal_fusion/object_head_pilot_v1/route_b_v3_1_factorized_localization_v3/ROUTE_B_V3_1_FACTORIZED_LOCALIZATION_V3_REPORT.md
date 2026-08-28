# Route B v3.1 factorized localization v3 FP32-repair report

Terminal: `LRASPP_FACTORIZED_LOCALIZATION_NO_GAIN`

Experiment: `experiments/route_b_v3_1_factorized_localization_v3/cuda_resume_fp32_20260828_073100`

## CUDA provenance and numerical qualification

- Interpreter: `/usr/bin/python3`; Python `3.10.12`; PyTorch `2.10.0.dev20251114+cu128`; CUDA build `12.8`; `CUDA_VISIBLE_DEVICES` unset.
- Device: `NVIDIA GeForce RTX 5090`; compute capability `(12, 0)`; compiled architecture list `sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120`.
- A real CUDA convolution allocation, forward, and backward passed before launch.
- V3 changes only the new localization precision boundary. The localization trunk, log-depth head, projected-centre-offset head, depth decode, camera unprojection, and localization losses execute with autocast disabled on FP32 tensors. The inherited detector remains under the registered outer AMP policy.
- Deterministic qualification reproduced the exact former epoch-2 batch-134 sample identity. Its FP32 first-convolution input was finite `[0, 15272]`; output was finite `[-75052.53125, 16662.16797]`, including values outside FP16's finite range. Batch 134 loss was `1.992764`; immediate batch 135 loss was `3.064358`; both losses and every new-component gradient were finite and nonzero, with zero frozen-component gradients. Qualification SHA-256: `df473e8fad98d31f8e390b0e34e14bee978fdc64e2c8da85380905da472ab212`.

## Camera-plane contract

The reusable rule moves any localization-positive object with physical-centre camera-forward depth `<= 0` to localization-ignore with reason `CAMERA_PLANE_STRADDLING_CENTER_NONPOSITIVE_DEPTH`. Segmentation remains unchanged and the region is neutral, never background.

- v0.10 train exclusions: 100.
- v0.10 validation exclusions: 34 (26 actor, 8 static-environment, 11 identities, zero person).
- v0.25 train/validation exclusions: 10 / 1.
- All nine hard contract gates passed; test rows are absent and raw corpus files copied = 0.
- Contract summary SHA-256: `460a7adcebf2fa2107a572b20f6a06ea69701f9c8f852ac4b74ab6c603e08385`.

## Amended native epoch-15 baseline (v0.10)

- Vehicle: P/R/F1 `0.712543/0.807760/0.757170`, XY MAE `0.984324 m`.
- Person: P/R/F1 `0.495587/0.464101/0.479328`, XY MAE `1.396104 m`.
- IoU: vehicle `0.865511`, person box-mask `0.443745`, foreground mIoU `0.654628`.
- Re-score used retained detections only (`265e68dc0bc6e1b5a851cf7254be45918a23d20ed60dbb040f60c607fd3ae1ba`); new inference passes = 0. v0.10 and v0.25 ignore caches were independently keyed.
- Amended baseline and embedded taxonomy SHA-256: `622d7f5e579384facaccbcdf43ef23ec2b9b68493534b9ed0dc3caac909aba04`.

## Architecture, isolation, and split proof

The new tail-side path reads the frozen native stride-4 128-channel fused feature, applies two `3x3 Conv(64)+GroupNorm+ReLU` blocks, and emits one `log_depth` channel plus two projected-physical-centre offset channels. The complete new path runs in FP32, unprojects positive `exp(log_depth)` using per-frame intrinsics, and replaces only decoded XYZ. Legacy XYZ remains checkpoint-compatible but untrained.

The transported bundle remains exactly `['high', 'low']`; tail raw-modality side channels are `[]` and monolithic/split outputs were bit-identical. Trainable parameters: 111,171; frozen parameters: 4,931,198; total: 5,042,369.

## Validation checkpoints (v0.10)

| Epoch | Eligible | Vehicle P/R/F1 | Vehicle XY m | Person P/R/F1 | Person XY m |
|---:|:---:|---|---:|---|---:|
| 4 | false | 0.670599 / 0.750800 / 0.708437 | 1.176847 | 0.468508 / 0.438017 / 0.452750 | 1.273116 |
| 8 | false | 0.681597 / 0.764730 / 0.720774 | 1.106225 | 0.474004 / 0.442665 / 0.457799 | 1.255432 |
| 12 | false | 0.696448 / 0.785058 / 0.738103 | 1.061371 | 0.500550 / 0.469783 / 0.484679 | 1.255408 |

Exactly epochs 4, 8, and 12 were evaluated. Each checkpoint had one inference pass at score floor 0.02 supplying both fixed thresholds.

## Selection and taxonomy

Selected checkpoint: none (no checkpoint passed every eligibility gate). The registered selection contract SHA-256 was `3bc97b3f4b8ea7099c9c2f310f8c8a40a3a11b1ba3d846b18797ba8e372a3a77` and was byte-identical to v2.

Best-ranked retained checkpoint regardless of promotion: epoch 12, `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/route_b_v3_1_factorized_localization_v3/cuda_resume_fp32_20260828_073100/checkpoints/route_b_v3_1_factorized_localization_v2/epoch_012.pt` (`9d556093fa9f7d644a6b1f37677df23e772c2636551f5ee63dbcd7a795cb0d06`).

Baseline vehicle taxonomy: `{'BACKGROUND_OR_OTHER': 485, 'PREDICTED_DUPLICATE': 979, 'TWO_D_CORRECT_WORLD_WRONG': 1694}`. Baseline person taxonomy: `{'CENTER_PRESENT_WORLD_WRONG': 854, 'HEATMAP_CENTER_MISS': 685, 'MATCHING_CONTENTION': 162}`.

Selected vehicle taxonomy: `None`. Selected person taxonomy: `None`.

Best-ranked retained epoch-12 taxonomy (diagnostic, not promoted): vehicle `{'BACKGROUND_OR_OTHER': 486, 'PREDICTED_DUPLICATE': 887, 'TWO_D_CORRECT_WORLD_WRONG': 1943}`; person `{'CENTER_PRESENT_WORLD_WRONG': 904, 'HEATMAP_CENTER_MISS': 705, 'MATCHING_CONTENTION': 156}`. Relative to baseline, the prioritized world-error counts increased rather than meeting the material reductions: vehicle `1694 -> 1943`, person `854 -> 904`.

## Visibility and radar stratification

Selected v0.25 flat metrics: `None`. Because no checkpoint passed eligibility, the registered selected-only v0.25 sensitivity scorer was not assigned a candidate; no extra inference or substitute best-ranked sensitivity score was run. The amended v0.25 baseline remained vehicle F1/XY `0.794269/0.943158 m` and person F1/XY `0.502274/1.394697 m`.

Radar-supported/unsupported localization at score 0.20 and 3 m matching: amended baseline `{'person': {'supported': {'eligible_gt': 3506, 'matched': 1736, 'recall': 0.49515116942384485, 'xy_mae_m': 1.384308580638149}, 'unsupported': {'eligible_gt': 366, 'matched': 61, 'recall': 0.16666666666666666, 'xy_mae_m': 1.7318027895598087}}, 'vehicle': {'supported': {'eligible_gt': 9173, 'matched': 7725, 'recall': 0.8421454267960319, 'xy_mae_m': 0.9693124580799889}, 'unsupported': {'eligible_gt': 518, 'matched': 103, 'recall': 0.19884169884169883, 'xy_mae_m': 2.110153719616456}}}`. Selected-candidate radar results are `None` because no candidate passed eligibility.

## Service targets

No checkpoint passed all eligibility gates; service gates were not assigned to a selected model.

Frozen segmentation cannot improve in this localization-only run, so full service readiness is not claimed.

## Resources, cleanup, and safety

Measured combined contract/re-score/pipeline wall time: `765.194 s` (pipeline `666.561 s`; training `278.797 s`; registered evaluation `381.324 s`). Peak CUDA allocated/reserved: `1134.8/1898.0 MiB`.

Hashes and metrics were recorded before removing all three inference payloads and the two non-retained new checkpoints. Canonical data, retained epoch-15 checkpoint/predictions, and prior experiments were not changed. Test remained unopened; CARLA, OAI, containers, q/AE, and 288 measurements were not run. No q/AE phase or follow-on experiment was started.
