# SplitFusion-FCOS relational person selector, revised p070 contract

This package is a minimal service wrapper around two unchanged frozen inputs:
FCOS epoch 26 and the existing person relational-selector checkpoint. It applies
the post-hoc train-holdout calibration for the revised precision/recall 0.70
objective without rewriting the selector checkpoint's historical
`train_infeasible` result under its original 0.80/0.80 gate.

At service time it reconstructs the selector's exact cached feature
representation, including the FP16-to-FP32 ROI descriptor round trip. The
historical consolidation decision is an input feature only. Person scores are
calibrated and thresholded at 0.20; vehicles use the accepted service-candidate
calibrator unchanged. No NMS, geometry, class, identity, or segmentation value
is changed.

The loader fails closed unless both checkpoint hashes, the frozen architecture
and state dictionary, the historical infeasible status, the revised locked
configuration, and the checked-in train-holdout verification report all match.

CPU contract checks:

```bash
python3 -m unittest -v \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_relational_p070_v1.tests.test_synthetic
```

Train-holdout reproduction (CPU, create-only output):

```bash
python3 -m \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_relational_p070_v1.verify_contract \
  --output /tmp/relational_p070_holdout_verification.json
```

Validation has intentionally not been run. The sole future validation command
is recorded in `REVIEW_PACKET.md`.

After that one inference pass completes, the frozen canonical v0.10 scorer can
be invoked exactly once with:

```bash
python3 -m \
  pole_lraspp_multimodal_fusion.object_head_pilot_v1.\
splitfusion_fcos_r50_fpn_p2_p7_relational_p070_v1.evaluate_relational_p070 \
  --prediction-dir experiments/relational_p070_v1/<completed-validation-run>
```

The evaluator accepts no output override and creates
`evaluation_v010.json` inside the completed prediction directory without
overwriting an existing result. It retains the canonical nine-gate result and
adds a separate supplemental p070 decision.
