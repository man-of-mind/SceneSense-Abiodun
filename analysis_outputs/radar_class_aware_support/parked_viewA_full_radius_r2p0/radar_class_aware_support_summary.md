# Radar Class-Aware Support Diagnostic

- Dataset: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2`
- Samples inspected: `11738`
- Object rows inspected: `114582`
- Min support points: `1`
- Vehicle box margin: `1.00 m`
- Person association: `radius`, radius `2.00 m`, z-down `0.50 m`, z-up `2.00 m`

| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | person_radius | 56516 | 0.047 | 0.102 | 3068 | 0 | 0.06 | 0.16 |
| vehicle | bbox | 58066 | 0.448 | 0.448 | 10 | 1 | 11.30 | 11.30 |

## Interpretation

This diagnostic does not use semantic IDs or hidden inference-time ground truth. It recomputes support from saved radar points and the supervised-training actor labels. If the person support rate rises, it means the original actor-box association was too strict for sparse pedestrian radar returns.
