# Route B v3.1 factorized localization v2 CUDA-resume report

Terminal: `LRASPP_FACTORIZED_LOCALIZATION_RUNTIME_FAILURE`

CUDA-resume experiment: `experiments/route_b_v3_1_factorized_localization_v2/cuda_resume_20260828_063600`

## Outcome

GPU access was successfully resolved without changing source, training configuration, the camera-plane contract, amended baseline, or registered selection gates. All eight existing launch checks passed. The unchanged training loop then stopped fail-closed when the loss became non-finite at epoch 2, batch 134. No retry, redesign, configuration change, checkpoint evaluation, or follow-up experiment was performed.

## CUDA provenance

| Field | Recorded value |
|---|---|
| Executable | `/usr/bin/python3` |
| Python | `3.10.12` |
| PyTorch | `2.10.0.dev20251114+cu128` |
| CUDA build | `12.8` |
| `torch.cuda.is_available()` | `True` |
| `CUDA_VISIBLE_DEVICES` | unset |
| Device | `NVIDIA GeForce RTX 5090` |
| Compute capability | `(12, 0)` |
| Architecture list | `sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` |
| Driver | `575.57.08` |
| Memory | `32607 MiB` |

A real CUDA `Conv2d(3,16,3,padding=1)` forward/backward completed under `/usr/bin/python3`; loss was `0.3190063536`, and input/weight gradients were finite. The initial sandboxed `nvidia-smi` failure was correctly treated as device isolation, not host unavailability; the approved execution confirmed a healthy GPU.

## Immutable provenance checks

- HEAD before execution: `99a699d4f67ba01b9823b8eb533a38ef920a30d2`.
- Warm-start SHA-256: `1245b2028372d486ed0b25b8a6b8a3e8b341257d542ec57cfdabf3b543d7c9ed`.
- Contract-summary SHA-256: `460a7adcebf2fa2107a572b20f6a06ea69701f9c8f852ac4b74ab6c603e08385`.
- Amended-baseline file SHA-256: `622d7f5e579384facaccbcdf43ef23ec2b9b68493534b9ed0dc3caac909aba04`.
- Amended metrics canonical SHA-256: `72d5fd31ed6da8f1e41aa715e4891727a50023cf5e008c74a3cdf89e8b0f3e5b`.
- Amended taxonomy canonical SHA-256: `c6318d0e2b5ccd5f74e478def761706fa9cf53854accbf599d81c0b5429d1450`.
- Retained detections SHA-256: `265e68dc0bc6e1b5a851cf7254be45918a23d20ed60dbb040f60c607fd3ae1ba`.
- Retained prediction-set SHA-256: `0d3f290b81addf2f8ba58411ad2a10ee6341147a749b69929d98fe6ced65cd08`.

The committed contract was not regenerated or reinterpreted. Exclusions remain v0.10 train/validation `100/34` and v0.25 train/validation `10/1`; the v0.10 validation composition remains 26 actor plus 8 static vehicles across 11 identities, with zero person transitions.

## Launch and split isolation

All eight committed launch checks passed. The real batch used batch 16, q=0, AMP with autocast cache disabled, and both classes. Loss was finite at `5.1016125679`. Gradient absolute sums were `4.2160419` for the localization trunk, `10.4239998` for the log-depth head, and `0.6968441` for the projected-centre-offset head; frozen-gradient sum was exactly zero.

The tail read only `{high, low}` plus camera calibration metadata. Monolithic-versus-split maximum absolute deltas were exactly zero for segmentation, native object output, and factorized localization. No raw-modality side channel exists.

Parameter counts remained 111,171 trainable localization parameters and 4,931,198 frozen inherited parameters, 5,042,369 total.

## Training record

| Epoch | Status | Mean loss | Log-depth | Projected offset | Local XY endpoint | LR | Batches |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | completed | 5.821915 | 0.104866 | 1.837001 | 3.880048 | 0.0003 | 398 |
| 2 | failed at batch 134 | non-finite | — | — | — | cosine schedule unchanged | 133 finite batches before failure |
| 4 | not reached | — | — | — | — | — | — |
| 8 | not reached | — | — | — | — | — | — |
| 12 | not reached | — | — | — | — | — | — |

The exact recorded exception was `RuntimeError: non-finite loss at epoch=2 batch=134`. No epoch-4/8/12 checkpoint was created.

## Baseline, selection, and evaluation

The amended retained baseline remains:

| Contract | Class | Precision | Recall | F1 | Recall @ 0.02 | XY MAE m |
|---|---|---:|---:|---:|---:|---:|
| v0.10 | vehicle | 0.712543 | 0.807760 | 0.757170 | 0.845527 | 0.984324 |
| v0.10 | person | 0.495587 | 0.464101 | 0.479328 | 0.560692 | 1.396104 |
| v0.25 | vehicle | 0.721978 | 0.882648 | 0.794269 | 0.903757 | 0.943158 |
| v0.25 | person | 0.497530 | 0.507109 | 0.502274 | 0.592417 | 1.394697 |

Baseline taxonomy remains vehicle duplicate `979`, vehicle `TWO_D_CORRECT_WORLD_WRONG=1694`, person `CENTER_PRESENT_WORLD_WRONG=854`, and person `HEATMAP_CENTER_MISS=685`.

Exactly zero candidate validation inference passes ran because training failed before the first authorized checkpoint. Consequently there is no selected checkpoint or SHA, no candidate world-error taxonomy, no radar-supported/unsupported candidate result, no selected v0.10 result, and no selected-checkpoint v0.25 sensitivity result. Registered selection and material-gain gates were never changed or applied to an incomplete model.

## Runtime, resources, and safety

The CUDA-resume pipeline ran for `85.782 s`; epoch 1 took `30.049 s`. Recorded epoch-1 CUDA peak allocated/reserved memory was `928.482/1350.0 MiB`.

The completion sentinel was written and desktop notification succeeded. The failed runtime directory was not overwritten, and this CUDA resume used a new create-only sibling. Test, CARLA, OAI, containers, q/AE, and 288 measurements remained untouched. No branch, remote, dependency, threshold, NMS, loss, optimizer, batch, or configuration change occurred. The dirty OAI submodule and pre-existing untracked refinement pointer were preserved.
