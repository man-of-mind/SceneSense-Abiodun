# Radar Class-Aware Support Diagnostic

- Dataset: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/moving_ego_radar_pps12000_medium_2loops_stride2`
- Samples inspected: `1459`
- Object rows inspected: `14918`
- Min support points: `1`
- Vehicle box margin: `1.00 m`
- Person association: `bbox`, radius `2.00 m`, z-down `0.50 m`, z-up `2.00 m`

| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | bbox | 9066 | 0.263 | 0.154 | 0 | 996 | 0.81 | 0.34 |
| vehicle | bbox | 5852 | 0.549 | 0.549 | 0 | 0 | 16.84 | 16.85 |

## Interpretation

This diagnostic does not use semantic IDs or hidden inference-time ground truth. It recomputes support from saved radar points and the supervised-training actor labels. If the person support rate rises, it means the original actor-box association was too strict for sparse pedestrian radar returns.
