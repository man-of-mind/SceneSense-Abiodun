# LR-ASPP / CenterFusion hybrid noAE pilot — registered plan

Registered **before** any training step or full-split decode. Nothing below is
re-tuned after seeing a result.

## Fixed baseline

* `experiments/route_b_noae_precision_full_v1/20260825_195301/checkpoints/curriculum_stage2_joint_v1/epoch_013.pt`
* sha256 `0882ef922edbcb8da47fe6568d8ba125e00bab71365d0370fd77268eb747dc30` (verified, read-only, never overwritten)
* Route B view: train 6,600 / val 3,588, `splits/test.txt` empty — the locked test
  split is absent from the view and is never touched.

Reproduced from the existing artifact `eval/curriculum_stage2_joint_v1_epoch_013/`
at score 0.20: vehicle P/R/F1 0.4624/0.4498/0.4560, person 0.3480/0.3752/0.3611,
XY MAE 1.1343 / 1.3195 m, vehicle IoU 0.8117, person IoU 0.3274, mIoU 0.7078.
The score-0.02 recall ceiling does not exist as an artifact and is measured in
Phase C alongside the hybrid warm start, with the identical decoder.

## Controlled comparison

`configs/hybrid_centerfusion_v1.json` copies **every** non-architectural setting
from `curriculum_stage2_joint_v1` verbatim: adamw, lr 3e-5, weight decay 1e-4,
strong photometric augmentation, no geometric augmentation, batch 16 / workers 8,
cosine with 1 warm-up epoch and min-lr ratio 0.01, `freeze_bn: true`, class loss
weights [0.5, 1.0, 4.0], lovasz 0.5, segmentation 0.3, the same seven object loss
weights, and the same `vehicle_heatmap_radius_cap_px = 4` object targets
(installed through `target_variants_v1`, whose control-parity guard runs first).
The single changed variable is the architecture.

The 24-epoch cosine schedule is registered up front and epochs 1-6 run under it,
so the continuation to 24 is a resume of the same run, not a restart.

## Phase C — warm-start parity gate

The mandated higher-resolution detection branch changes the object grid by
construction, so bit-identical *object* output is not achievable and is not
claimed. What is architecturally retained is checked exactly, and the object path
is checked against the decoded baseline numbers:

| quantity | tolerance |
| --- | --- |
| every mapped tensor vs its baseline source | bit-exact (`torch.equal`) |
| fused `low` / `high` features vs baseline backbone features | max abs diff / max abs value <= 1e-4 |
| segmentation logits vs baseline | max abs diff / max abs value <= 1e-4 |
| coarse 1/8 object logits vs baseline object head | max abs diff / max abs value <= 1e-4 |
| decoded val vehicle/person precision, recall, F1 | \|delta\| <= 0.005 |
| decoded val vehicle/person XY MAE | \|delta\| <= 0.01 m |
| decoded val vehicle IoU, person IoU, mIoU | \|delta\| <= 0.002 |

Any violation => `WARM_START_PARITY_FAILED`, stop.

Residual note: the radar-conditioned refinement's output convolutions are
initialised at std 1e-4 rather than exactly zero. Exact-zero initialisation and
the Phase B "finite non-zero gradient in the refinement branch" requirement are
mutually exclusive at step 1, so the near-zero initialisation is what allows both
checks to be real. Its effect on the decoded warm start is bounded by the table
above.

## Phase D/E — six warm-started clean-q epochs, then the early continuation gate

Decoder is the primary evaluator contract for every decode, with no per-epoch,
per-class or per-threshold tuning: q=0, score 0.20, top-k 120, image NMS 2 px,
3.0 m class-aware matching, 40 m GT range, 12 px GT area floor. Score 0.02 is
decoded only to measure the permissive recall ceiling and is never used to
select anything.

Continue to epoch 24 only if, at epoch 6, **all** of:

1. score-0.02 vehicle recall >= baseline + 0.05
2. score-0.02 person recall >= baseline + 0.05
3. score-0.20 vehicle precision >= baseline - 0.03
4. score-0.20 person precision >= baseline - 0.03
5. mIoU >= baseline - 0.02
6. no NaN, no training collapse, no output-schema mismatch

Otherwise stop immediately with `HYBRID_NOAE_PILOT_NO_GAIN`. No second
architecture, no gate change.

If the gate passes: continue the same run to epoch 24, save every epoch, decode
only epochs 6, 10, 14, 18, 22, 24. Select by highest mean vehicle/person F1, then
higher minimum class recall, then lower mean XY MAE.

## Service targets (reported separately; never relaxed, matching stays at 3.0 m)

vehicle recall >= 0.85, person recall >= 0.80, vehicle and person precision >= 0.80,
vehicle XY MAE <= 1.0 m, person XY MAE <= 1.2 m, vehicle IoU >= 0.85,
person IoU >= 0.50, mIoU >= 0.80.

## Out of scope for this task

CARLA/OAI runs, the live UDP split runtime, AE32/64/128 training, q robustness
sweeps, decoder grids, the 288-measurement campaign, and the locked Route B test
split.
