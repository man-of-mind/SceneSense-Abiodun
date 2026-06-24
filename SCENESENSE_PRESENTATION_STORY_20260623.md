# SceneSense Update Storyline - 2026-06-23

## One-Sentence Story

We moved from "can a parked/fixed-view model transfer?" to a more realistic moving-ego RGB+radar model, found that model quality is now limited by low/medium-density vehicle IoU and person sensing, and used LiDAR/radar diagnostics to identify the next best sensor-processing experiment: full moving-domain training with denser radar and bbox-based radar support.

## Recommended Slide Flow

### 1. Motivation: Why We Changed Direction

**Message:** Single-view parked models do not generalize reliably, so the model should be trained in the moving domain if the final system will use a moving ego.

**Bullets:**
- Parked A+B model works well on its trained parked viewpoints.
- But parked A+B model is a negative-control result on moving data: moving-domain test performance drops sharply.
- Conclusion: transferability itself is a key bottleneck, not only network/compression.

**Plot:**
- `analysis_outputs/moving_ego_fusion_model_eval/moving_fusion_domain_gap_segmentation.png`

**Talk track:**
> The parked model result tells us that viewpoint matters. For the agent/RL work, we should assume the perception model must be trained for the operating domain first; otherwise, network-aware decisions are optimizing around a weak model.

### 2. Moving-Ego RGB+Radar Model: Current Baseline

**Message:** We now have a moving-domain fusion model that is meaningfully better than parked-to-moving transfer, but it is not yet at the desired quality gate.

**Numbers:**
- 8-loop moving model: `mIoU=0.825`, `vehicle IoU=0.874`, `person IoU=0.630`.
- Crowded density already reaches the vehicle target: `vehicle IoU=0.903`.
- Low/medium are the bottlenecks: low `0.773`, medium `0.828`.

**Plot:**
- `analysis_outputs/moving_ego_fusion_tuning/moving_fusion_tuning_clean_baseline_vs_best.png`

**Talk track:**
> The important finding is not just the overall score. The per-density breakdown says the model is strongest in crowded scenes and weaker in low/medium scenes, which means simply adding more crowded data is not the cleanest fix.

### 3. More Repeated Loops Did Not Fix It

**Message:** More samples alone, when collected from repeated route loops, did not improve segmentation.

**Numbers:**
- 8-loop moving model: `mIoU=0.825`, `vehicle IoU=0.874`, `person IoU=0.630`.
- 12-loop/more-data run: `mIoU=0.813`, `vehicle IoU=0.846`, `person IoU=0.624`.
- Localization improved slightly: F1 `0.287 -> 0.307`, XY error `1.430m -> 1.373m`.

**Plots:**
- `analysis_outputs/moving_ego_fusion_model_eval/moving_fusion_segmentation_8_vs_12loops.png`
- Optional backup: `analysis_outputs/moving_ego_fusion_model_eval/moving_fusion_localization_8_vs_12loops.png`

**Talk track:**
> This suggests dataset diversity matters more than simply collecting more of the same route. The model may already have saturated what it can learn from repeated loops.

### 4. Training Objective / Weight Tuning Was Not the Fix

**Message:** Loss and checkpoint tuning did not improve the vehicle-IoU target.
Here, the "baseline" is the original moving-ego RGB+radar model trained with
the default objective. The "tuned" model starts from the same model family but
changes training priorities, such as class weights, object-head weight, and
checkpoint selection target.

**Numbers:**
- Best overall vehicle IoU remains the original 8-loop baseline: `0.874`.
- Best tuned overall mIoU is similar: `0.8254`, but vehicle IoU drops to `0.8547`.
- Tuning slightly helps person IoU in one trial (`0.649` vs baseline `0.630`) but hurts vehicle IoU.

**Plots:**
- `analysis_outputs/moving_ego_fusion_tuning/moving_fusion_tuning_clean_baseline_vs_best.png`
- Optional backup: `analysis_outputs/moving_ego_fusion_objective_audit/moving_fusion_training_objective_audit.png`

**Talk track:**
> This is a useful negative result. By tuning, we mean changing the training
> objective and selection criteria, not changing the model architecture. Since
> the original moving model still has better vehicle IoU, the main bottleneck is
> probably not a simple loss-weight issue.

### 5. LiDAR Diagnostic: What Is Transferable To Radar?

**Message:** Semantic LiDAR looks strong partly because it exposes simulator-only tags/object IDs; the useful transferable ideas are geometry and point density, not oracle IDs.

**Numbers:**
- Raw LiDAR person recall in controlled curbside crossing: `0.848`.
- Semantic tag/person recall: `0.840`.
- Semantic object-ID vehicle recall: `1.000`, but this is oracle simulator association.
- Semantic LiDAR has much higher estimated payload: ~`1.11 MB/frame` vs raw ~`0.49 MB/frame`.

**Plots:**
- `analysis_outputs/lidar_raw_vs_semantic_curbside_radius_z5/lidar_raw_vs_semantic_recall.png`
- `analysis_outputs/lidar_raw_vs_semantic_curbside_radius_z5/lidar_raw_vs_semantic_points_per_frame.png`

**Talk track:**
> The LiDAR diagnostic did not magically solve person detection. Instead, it
> clarified what is realistic to transfer to radar: denser points, better
> geometry association, and possibly short temporal accumulation. Semantic
> object IDs are useful for diagnosis, but they are simulator-side oracle
> information, not a deployable sensor output.

### 6. Radar Ablation: Sensor Processing Helps, But Not Enough Yet

**Message:** Increasing radar points and changing association geometry changes model behavior, but the 2-loop ablation still does not reach the target.

**Numbers:**
- Best ablation cell by segmentation: `12000 pps + bbox support`.
- It reached `mIoU=0.812`, `vehicle IoU=0.841`, `person IoU=0.626`.
- Radius association helped localization in some cases, but bbox support looked better for vehicle segmentation.

**Plot:**
- `analysis_outputs/radar_model_ablation/moving_radar_model_ablation.png`

**Talk track:**
> The model-level ablation did not solve person detection or reach the 0.90
> vehicle-IoU target. But it gave us the most defensible next experiment: run
> the full moving pipeline with 12k radar points and bbox radar support.

### 7. Spatial Map Component: Parallel System Contribution

**Message:** In parallel with model improvement, we started a visibility-aware spatial map module for fusing local maps from multiple viewpoints.

**Bullets:**
- Each ego/camera can publish a local FOV footprint plus detected objects in world coordinates.
- The server can compute FOV overlap and reason about potential occlusion disagreement.
- This will later connect to the RL map-sharing policy: what to share, when to share, and at what detail level.

**Plot:**
- `analysis_outputs/spatial_map_geometry/two_view_overlap_demo.png`

**Talk track:**
> This is still an early geometry prototype, not a final map server. The key point is that we are now building the map-sharing side while the perception model is being improved.

### 8. Current Overnight/Recovery Experiment

**Message:** The full 12k+bbox overnight run partially completed; crowded stopped after 6 loops but produced enough samples to salvage.

**Bullets:**
- Target crowded collection: 8 loops / up to 6000 samples.
- Actual crowded collection: 6 loops / 4610 samples.
- Salvage plan: train with low+medium full data and crowded 6-loop subset.
- If salvage improves: keep 12k+bbox direction.
- If salvage does not improve: next likely lever is higher input resolution or more route/view diversity, not more weight tuning.

**No plot yet:** result pending.

## Recommended Plot Set For Today

Use these in the main update:

1. `analysis_outputs/moving_ego_fusion_model_eval/moving_fusion_domain_gap_segmentation.png`
2. `analysis_outputs/moving_ego_fusion_tuning/moving_fusion_tuning_clean_baseline_vs_best.png`
3. `analysis_outputs/moving_ego_fusion_model_eval/moving_fusion_segmentation_8_vs_12loops.png`
4. `analysis_outputs/moving_ego_fusion_tuning/moving_fusion_tuning_delta_vs_baseline.png`
5. `analysis_outputs/radar_model_ablation/moving_radar_model_ablation.png`
6. `analysis_outputs/lidar_raw_vs_semantic_curbside_radius_z5/lidar_raw_vs_semantic_recall.png`
7. `analysis_outputs/spatial_map_geometry/two_view_overlap_demo.png`

Backup plots:

- `analysis_outputs/moving_ego_fusion_model_eval/moving_fusion_localization_8_vs_12loops.png`
- `analysis_outputs/lidar_raw_vs_semantic_curbside_radius_z5/lidar_raw_vs_semantic_points_per_frame.png`
- `analysis_outputs/moving_ego_fusion_objective_audit/moving_fusion_training_objective_audit.png`

## Conclusions To Say Out Loud

- We should not start compression/RL characterization until the model quality is stable enough; otherwise the RL agent is optimizing a weak perception pipeline.
- Moving-domain training is the right direction; parked-to-moving transfer is too weak.
- The current moving model is promising but not final: vehicle IoU is close but below `0.90`; person IoU remains weak.
- More repeated loops and simple objective tuning did not solve the gap.
- Sensor-processing changes are the most justified next experiment: denser radar and bbox association, followed by higher input resolution or route/view diversity if needed.
- The spatial-map work is now started and can proceed in parallel as the eventual consumer of the perception outputs.

## Suggested Next Actions

1. Finish salvage run for full `12000 pps + bbox` moving model.
2. Compare against baseline 8-loop moving model by density.
3. If vehicle IoU improves, use that checkpoint for live visualization and later compression characterization.
4. If vehicle IoU does not improve, test higher model input resolution or route/view diversity.
5. Keep person localization as a separate sensor-processing problem: radar accumulation/BEV features may be needed.
