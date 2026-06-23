# Radar Class-Aware Support Diagnostic

- Dataset: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2`
- Samples inspected: `1000`
- Object rows inspected: `4319`
- Min support points: `1`
- Vehicle box margin: `1.00 m`
- Person association: `radius`, radius `1.50 m`, z-down `0.50 m`, z-up `2.00 m`

| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | person_radius | 2513 | 0.039 | 0.053 | 37 | 0 | 0.04 | 0.06 |
| vehicle | bbox | 1806 | 0.404 | 0.404 | 0 | 0 | 3.79 | 3.79 |

## Interpretation

This diagnostic does not use semantic IDs or hidden inference-time ground truth. It recomputes support from saved radar points and the supervised-training actor labels. If the person support rate rises, it means the original actor-box association was too strict for sparse pedestrian radar returns.
