# Raw vs Semantic LiDAR Diagnostic Summary

## Runs
- `lidar_diagnostic_runs/curbside_raw_vs_semantic_lidar_clean_crossing_radius`

## What The Modes Mean
- `raw_bbox`: raw LiDAR geometry assigned to CARLA actor boxes for evaluation only.
- `semantic_tag_bbox`: semantic LiDAR points filtered by semantic tag, then assigned to actor boxes.
- `semantic_object_id`: semantic LiDAR grouped by CARLA object ID; this is oracle association.

## Actor Coverage

| Run | Mode | Class | Recall | XY error mean (m) | Points/actor mean | Observations |
|---|---|---|---:|---:|---:|---:|
| curbside_lidar_clean_crossing_radius | raw_bbox | vehicle | 0.000 | nan | 10.0 | 250 |
| curbside_lidar_clean_crossing_radius | raw_bbox | person | 0.000 | nan | 0.0 | 250 |
| curbside_lidar_clean_crossing_radius | semantic_tag_bbox | vehicle | 0.000 | nan | 10.0 | 250 |
| curbside_lidar_clean_crossing_radius | semantic_tag_bbox | person | 0.000 | nan | 0.0 | 250 |
| curbside_lidar_clean_crossing_radius | semantic_object_id | vehicle | 1.000 | 1.941 | 86.0 | 250 |
| curbside_lidar_clean_crossing_radius | semantic_object_id | person | 0.840 | 0.286 | 11.2 | 250 |

## Frame-Level Sensor Load

| Run | Frames | Raw pts/frame | Semantic pts/frame | Raw bytes/frame est. | Semantic bytes/frame est. |
|---|---:|---:|---:|---:|---:|
| curbside_lidar_clean_crossing_radius | 250 | 30391.1 | 55568.3 | 486257 | 1111366 |

## Interpretation
Semantic LiDAR is expected to look better mainly because it exposes simulator-provided tags and object IDs. The useful radar-transfer ideas are the geometry-side behaviors: point density, actor-box association for training/evaluation, short temporal accumulation, and BEV/voxel features.
