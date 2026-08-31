# SplitFusion FCOS locked service candidates v1

This package is a thin inference-only composition of the frozen recovered epoch-26 SplitFusion FCOS model, the reviewed feasible person instance-consolidation rule, and one fixed train-derived monotonic vehicle-score calibration. It has no trainable parameters and never loads the candidate-quality or ROI-verifier heads.

`locked_config.json` binds the package to epoch-26 checkpoint SHA-256 `da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f` and person feasibility-result SHA-256 `a1bb8b2b7062abc2d0ef4c5cbc715154c5a4e9f1da64e050547de14c56bdddde`. Loading fails closed unless the result remains `holdout_feasible` at grid 27 with semantic support `0.10` and group-box IoU `0.20`.

The combined runtime calls the existing reviewed consolidation implementation unchanged. Retained person records keep every original field and score. Every vehicle candidate is retained in original post-NMS order; FP32 vehicle scores alone receive the locked logit bias `-1.476162131187961`. No candidate is created, no second NMS runs, and segmentation, geometry, candidate identity, and original prediction index remain unchanged.

Future one-pass validation inference:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.infer_service_candidates \
  --output experiments/splitfusion_fcos_service_candidate_v1/predictions \
  --device cuda:0
```

Future frozen nine-gate evaluation:

```bash
python3 -m pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.evaluate_service_candidates \
  --prediction-dir experiments/splitfusion_fcos_service_candidate_v1/predictions
```

The three compact CPU checks do not open validation or test data:

```bash
python3 -m unittest \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.tests.test_synthetic -v
```
