# SplitFusion FCOS R50 FPN P2-P7 V1

This package implements exactly one preregistered clean noAE seven-channel Route B v3.1 model. The UE/front ends at raw FP32 ResNet C2; the edge/tail alone creates P2-P7 and all detection, segmentation, dense-depth, and factorized geometry outputs. It contains no AE, quantizer, distillation adapter, competing detector, live deployment, CARLA, OAI, or locked-test path.

The run is orchestrated by `run_pipeline.py`. All mutable datasets, qualification artifacts, checkpoints, predictions, metrics, and sentinels are written below the one experiment directory supplied to that script. Source configuration and provenance in this directory are intended for the final local `master` commit; experiment payloads are Git-ignored.

