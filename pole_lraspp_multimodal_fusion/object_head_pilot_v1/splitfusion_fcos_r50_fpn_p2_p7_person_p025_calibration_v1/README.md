# SplitFusion-FCOS person p025 candidate

This package qualifies one fixed person output threshold and, only after the
train-only gate passes, wraps the accepted p020 service policy. The wrapper
runs the existing person consolidation and vehicle calibration unchanged, then
removes only consolidated person rows whose FP32 score is below `0.25`.

The generated evidence is create-only under
`experiments/splitfusion_fcos_person_p025_calibration_v1/`. The approved p020
service is not modified or automatically replaced; this p025 wrapper remains a
proposed deployment candidate pending final acceptance and untouched-test
confirmation.
