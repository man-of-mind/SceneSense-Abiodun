# Radar Class-Aware Support Diagnostic

- Dataset: `/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_tl16_spawn80_right8_fwd16_merged_12000_stride2`
- Samples inspected: `11943`
- Object rows inspected: `138599`
- Min support points: `1`
- Vehicle box margin: `1.00 m`
- Person association: `radius`, radius `2.00 m`, z-down `0.50 m`, z-up `2.00 m`

| Class | New geometry | Rows | Current support rate | Class-aware support rate | Gained rows | Lost rows | Current mean pts | Class-aware mean pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | person_radius | 61466 | 0.050 | 0.107 | 3501 | 3 | 0.07 | 0.18 |
| vehicle | bbox | 77133 | 0.363 | 0.363 | 11 | 11 | 4.68 | 4.68 |

## Interpretation

This diagnostic does not use semantic IDs or hidden inference-time ground truth. It recomputes support from saved radar points and the supervised-training actor labels. If the person support rate rises, it means the original actor-box association was too strict for sparse pedestrian radar returns.
