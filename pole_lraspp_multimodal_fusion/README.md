# Pole LR-ASPP Multimodal Fusion

This additive workflow trains a traffic-light-pole LR-ASPP multi-task model with
early RGB+radar fusion. It preserves the existing RGB-only trained pole model as
the initialization and segmentation baseline, while adding learned object heads
for vehicle center confidence, sensor-relative location, dimensions, yaw,
parked/stopped state, and radar-support confidence.

Object localization is a neural output of the fusion model. The evaluation
matches decoded model predictions to CARLA actor ground truth and reports learned
global x/y error after transforming the predicted sensor-relative location back
through the saved camera calibration. Classical radar-point localization is not
used as the primary localization metric.

The long CARLA run should be started from a normal host shell. Codex should only
prepare and validate this launcher, then inspect logs after the user starts it.

Key files:

- `configs/fusion_full_run.yaml`: full unattended data collection, tuning, and evaluation config.
- `configs/fusion_smoke.yaml`: short CARLA smoke config.
- `launch_unattended_fusion_training.sh`: host-shell launcher with `direct`, `screen`, and `nohup` modes.
- `status_unattended_fusion_training.sh`: status/log helper.
- `stop_unattended_fusion_training.sh`: requests clean stop and signals the supervisor.

Main artifacts are written under:

`experiments/pole_lraspp_multimodal_fusion/<timestamp>_pole_lraspp_multimodal_fusion/`

Important outputs:

- `dataset/manifest.csv`: synchronized RGB/radar/instance paths plus camera,
  radar, radar-to-camera, and anchor calibration.
- `dataset/object_boxes.csv`: CARLA actor ground truth including world position,
  sensor-relative position, dimensions, yaw, velocity, stationary age, parked
  label, and radar support count.
- `metrics/*_metrics.csv`: training losses including segmentation, center,
  learned location, dimension, yaw, parked, and radar-support terms.
- `metrics/test_fusion_evaluation_metrics.json`: learned object precision,
  recall, F1, global x/y MAE/RMSE, dimension MAE, yaw error, parked accuracy,
  and RGB-only segmentation baseline comparison.
- `metrics/test_learned_object_metrics.csv`: per-object learned prediction/GT
  matches.
