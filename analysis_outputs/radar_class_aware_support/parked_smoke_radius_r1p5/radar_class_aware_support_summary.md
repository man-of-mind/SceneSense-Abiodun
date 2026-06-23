# Radar Class-Aware Support Diagnostic

- Dataset: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_smoke_60_stride2`
- Samples inspected: `60`
- Object rows inspected: `1950`
- Min support points: `1`
- Vehicle box margin: `1.00 m`
- Person association: `radius`, radius `1.50 m`, z-down `0.50 m`, z-up `2.00 m`

| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | person_radius | 976 | 0.033 | 0.045 | 13 | 1 | 0.04 | 0.07 |
| vehicle | bbox | 974 | 0.359 | 0.359 | 0 | 0 | 1.20 | 1.20 |

## Interpretation

This diagnostic does not use semantic IDs or hidden inference-time ground truth. It recomputes support from saved radar points and the supervised-training actor labels. If the person support rate rises, it means the original actor-box association was too strict for sparse pedestrian radar returns.
