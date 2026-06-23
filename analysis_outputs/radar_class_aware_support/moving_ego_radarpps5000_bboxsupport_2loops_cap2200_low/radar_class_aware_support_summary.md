# Radar Class-Aware Support Diagnostic

- Dataset: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/moving_ego_radarpps5000_bboxsupport_2loops_cap2200_low_stride2`
- Samples inspected: `1182`
- Object rows inspected: `6160`
- Min support points: `1`
- Vehicle box margin: `1.00 m`
- Person association: `bbox`, radius `2.00 m`, z-down `0.50 m`, z-up `2.00 m`

| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | bbox | 4107 | 0.067 | 0.067 | 0 | 0 | 0.12 | 0.12 |
| vehicle | bbox | 2053 | 0.545 | 0.545 | 0 | 0 | 5.05 | 5.05 |

## Interpretation

This diagnostic does not use semantic IDs or hidden inference-time ground truth. It recomputes support from saved radar points and the supervised-training actor labels. If the person support rate rises, it means the original actor-box association was too strict for sparse pedestrian radar returns.
