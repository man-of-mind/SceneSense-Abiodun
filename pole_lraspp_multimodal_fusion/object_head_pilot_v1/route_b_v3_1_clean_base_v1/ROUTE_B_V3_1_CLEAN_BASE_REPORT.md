# Route B v3.1 clean noAE LR-ASPP report

Terminal: `LRASPP_V3_1_IMPROVED_NOT_SERVICE_READY`

The immutable v3.1 primary contract contains 24,573 train positives and 104,900
train ignore records over 6,361 frames; validation contains 13,597 positives and
57,601 ignore records over 3,345 frames. Six source-episode symlinks provide the
RGB/radar/depth/semantic payloads. No corpus payload was copied and no locked
test payload was admitted or read.

## Selected checkpoint

- Epoch: 20
- Path: `experiments/route_b_v3_1_clean_base_v1/20260828_012309/checkpoints/route_b_v3_1_clean_noae_stage2_v1/epoch_020.pt`
- SHA-256: `88b34a69eeec7bf2f6444e70a0e346c365b979e6936d277cb0c75e8cd747aa1d`
- Loss-best epoch: 5, reported separately and not auto-promoted

## Primary v0.10 results

| model | class | TP / FP / FN | precision | recall | F1 | recall@0.02 | XY MAE m | dim MAE m | yaw MAE deg | class IoU | foreground mIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| epoch-13 frozen | vehicle | 3385 / 3335 / 6340 | 0.5037 | 0.3481 | 0.4117 | 0.4735 | 1.119 | 0.217 | 64.62 | 0.8303 | 0.5373 |
| epoch-13 frozen | person | 1432 / 2362 / 2440 | 0.3774 | 0.3698 | 0.3736 | 0.4811 | 1.444 | 0.095 | 93.26 | 0.2444 | 0.5373 |
| selected epoch 20 | vehicle | 6849 / 7390 / 2876 | 0.4810 | 0.7043 | 0.5716 | 0.7461 | 0.998 | 0.240 | 60.80 | 0.8640 | 0.6514 |
| selected epoch 20 | person | 1560 / 1822 / 2312 | 0.4613 | 0.4029 | 0.4301 | 0.4561 | 1.409 | 0.098 | 89.05 | 0.4388 | 0.6514 |

Mean class F1 improved by +0.1082. Vehicle/person F1 changed by +0.1599 and
+0.0565, and foreground mIoU improved by +0.1140. All five evaluated epochs
(5, 10, 15, 20, 25) passed the preregistered feasibility and material-gain
gates. The fixed lexicographic order selected epoch 20.

Relative to retained context, selected mean F1 changed by +0.2141 versus
M-prime LR-ASPP and -0.0849 versus Faster R-CNN. Foreground mIoU changed by
+0.3874 and +0.1362, respectively.

## v0.25 sensitivity

| class | precision | recall | F1 | recall@0.02 | XY MAE m |
|---|---:|---:|---:|---:|---:|
| vehicle | 0.4913 | 0.7910 | 0.6061 | 0.8346 | 0.988 |
| person | 0.4597 | 0.4479 | 0.4537 | 0.5021 | 1.401 |

Sensitivity segmentation is 0.8648 vehicle IoU, 0.4470 person box-mask IoU,
and 0.6559 foreground mIoU. Background IoU is 0.9940 and diagnostic only.

## Advisory service targets

| target | value | requirement | result |
|---|---:|---:|---|
| vehicle precision | 0.4810 | >= 0.80 | FAIL |
| vehicle recall | 0.7043 | >= 0.85 | FAIL |
| person precision | 0.4613 | >= 0.80 | FAIL |
| person recall | 0.4029 | >= 0.80 | FAIL |
| vehicle XY MAE | 0.9981 m | <= 1.0 m | PASS |
| person XY MAE | 1.4088 m | <= 1.2 m | FAIL |
| vehicle IoU | 0.8640 | >= 0.85 | PASS |
| person box-mask IoU | 0.4388 | >= 0.50 | FAIL |
| foreground mIoU | 0.6514 | >= 0.675 | FAIL |

## Runtime and integrity

Training completed all 25 epochs in 2,840.4 seconds. Baseline plus authorized
evaluation/scoring took 988.1 seconds. Peak allocated VRAM was 5,777.6 MiB and
peak reserved VRAM was 7,636.0 MiB. The launch batch used q=0 AMP with the
autocast cache disabled and produced finite nonzero backbone, classifier, and
object-head gradients.

Canonical v3, locked test payloads, OAI, prior checkpoints, and prior experiment
payloads were not modified. No CARLA/OAI/container run, remote Git operation,
AE/q experiment, threshold sweep, or extra architecture evaluation was run.

# LRASPP_V3_1_IMPROVED_NOT_SERVICE_READY
