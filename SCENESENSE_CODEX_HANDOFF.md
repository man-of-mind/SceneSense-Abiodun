# SceneSense Codex Handoff

Last updated: 2026-06-04

This file is for handing the project to a fresh Codex session. It summarizes
where the SceneSense work stands, what was just completed, what is left, and
the next concrete engineering task.

## Current Repo State

Committed baseline is `c52ffff`; current 2026-06-04 follow-up files may still
be local/uncommitted unless a later session commits them.

Current 2026-06-04 local follow-up:

- `carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py`
  now writes per-object prediction and CARLA vehicle ground-truth CSVs when
  `--run-logging` is enabled.
- Parked-ego GT logging excludes the spawned ego vehicle itself so object recall
  measures surrounding vehicles rather than the sensor platform.
- `scripts/analyze_fusion_object_transfer.py` evaluates those CSVs for
  `fusion_as_od` recall/localization/yaw/dimension metrics.
- `SceneSense_Fusion_Model_Transferability_OD_SEG.pptx` summarizes the
  pole-vs-parked-ego transferability result for both segmentation and object
  detection/localization.
- `carla_collect_parked_ego_fusion_training_data.py` starts the saved
  parked-ego fusion training-data path.
- `scripts/validate_fusion_training_dataset.py` validates saved sample folders
  before attempting training/fine-tuning.
- `scripts/dry_run_fusion_training_targets.py` verifies that saved rows can be
  converted into fusion model inputs, segmentation targets, and current vehicle
  object-head targets without launching training.
- `FUSION_TRAINING_DRIVER_GAP_ANALYSIS.md` records the local search result:
  target construction is proven, but no standalone SceneSense fusion training
  driver was found in this checkout.
- `carla_split_inference_udp_segmentation_oai.py` now supports
  `--enable-semantic-gt` on the front/loopback side and logs camera-only SEG
  quality columns: binary foreground IoU, 3-class macro mIoU, vehicle IoU,
  person IoU, and GT pixel counts. The semantic camera is evaluation-only;
  RGB remains the model input.
- `scripts/analyze_camera_seg_metrics.py` summarizes camera-only SEG CSVs and
  flags old transport-only CSVs with `quality_columns_present=false`.
- Camera-only SEG loopback quality was collected and analyzed:
  `month1_camera_seg_loopback_20260604_145934.csv`, 451 frames, 450 GT frames,
  foreground/binary IoU mean 0.195, 3-class macro mIoU mean 0.508, vehicle IoU
  mean 0.172. There were no visible person GT pixels, so person IoU is not
  measured for that trace.
- `carla_split_inference_udp_oai.py` now supports `--enable-od-gt` on the
  front/loopback side. It projects CARLA vehicle/person actor boxes into the
  RGB camera and logs first-pass camera-only OD precision/recall using
  class-aware 2D IoU matching against Faster R-CNN predictions.
- `scripts/analyze_camera_od_metrics.py` summarizes camera-only OD CSVs and
  flags old transport-only CSVs with `quality_columns_present=false`.
- Camera-only OD loopback/OAI quality was collected and analyzed:
  `month1_camera_od_loopback_20260604_153409.csv` and
  `month1_camera_od_oai_20260604_153845.csv`. Overall at IoU 0.5: 2380
  frames, 9983 GT objects, 2294 predicted objects, 1047 matches, global recall
  0.105, global precision 0.456, mean matched IoU 0.713. Loopback recall /
  precision: 0.112 / 0.526; OAI recall / precision: 0.092 / 0.358. Summary
  artifact:
  `metrics_logs/month1_camera_od_analysis/month1_camera_od_quality_20260604_154333.md`.
- Camera-only OD-vs-SEG latency comparison over loopback/OAI was collected
  and packaged into `SceneSense_Camera_OD_SEG_Latency_Comparison.pptx`.
  Evidence lives in
  `metrics_logs/scenesense_analysis/camera_od_seg_latency_20260604/`.
  Headline: OD median RTT loopback/OAI `8.2/74.9 ms`; SEG median RTT
  loopback/OAI `13.4/107.9 ms`; SEG median feature payload is about `4.6x`
  OD. The deck includes the current OAI config: RFsim, band n78, 30 kHz SCS,
  106 PRB approximately 40 MHz, 5 ms TDD pattern, DNN `oai`, SST 1, 5QI 9.
- `SCENESENSE_MONTH1_COMMANDS.md` has been expanded into the single Month 1
  command sheet for OAI bring-up, camera OD/SEG, fusion loopback/OAI/multi-UE,
  curbside accident scenario, UE tunnel sampler, app/network analyzer,
  T-tracer/RAN extraction, post-run OAI snapshot collection, and camera/fusion
  task-quality analyzers.
- Syntax check passed for the shared fusion runtime, object-transfer analyzer,
  parked-ego collector, saved-dataset validator, target dry-run script,
  camera-only OD/SEG OAI runtimes, and camera-only OD/SEG analyzers.
- Live pole-vs-parked-ego transfer runs produced the OD/SEG evidence deck.
- Remote parked-ego saved-data smoke collection passed validation:
  `parked_ego_fusion_training_smoke_20260604` has 30 manifest rows, 474
  actor-derived object rows, vehicle/person labels, masks with classes `0/1/2`,
  RGB shape `(480, 854, 3)`, mask shape `(480, 854)`, radar tensor shape
  `(4, 432, 768)`, and no validator errors/warnings.

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
git status --short
git log --oneline -5
```

Expected latest commits:

```text
c52ffff Add parked ego fusion transfer evaluation
eee34c5 Add SceneSense OAI comparison presentation
e3de3f9 Add SceneSense fusion comparison tooling
31d6212 Lock SceneSense Month 1 occlusion evidence
911aaf5 Checkpoint SceneSense scenario harness tuning
```

The working tree was clean immediately after commit `c52ffff`.

## Remote Machine Workflow

Experiments are usually run on a remote machine with the same project path:

```text
shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/
```

After local edits, ship only the changed files with `rsync -av` or `rsync -avh`.
Example pattern:

```bash
rsync -avh \
  abiodun/carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  abiodun/carla_collect_parked_ego_fusion_training_data.py \
  abiodun/carla_split_inference_udp_fusion_object_ego_client.py \
  abiodun/scripts/analyze_fusion_object_transfer.py \
  abiodun/scripts/validate_fusion_training_dataset.py \
  abiodun/seg_radar_pole_fusion_readme.md \
  abiodun/payload_fusion_handoff_readme.md \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/
```

There is also a replica folder on another remote machine. Commands that work
locally should work there after the same code changes are synced.

## Project North Star

SceneSense is building toward a network-aware split-inference controller that
reduces payload and latency while preserving task utility.

The future controller may choose:

- quantization level
- entropy/compression profile
- ROI/saliency threshold
- AE channel count, where supported
- frame send/skip
- redundancy add/drop

Guardrails matter: object recall/AP, segmentation mIoU/foreground IoU, and
vulnerable-object recall must not silently collapse.

## Key Files

Use these first when re-orienting:

- `SCENESENSE_MONTHLY_CHECKLIST.md`: month-by-month project checklist.
- `SCENESENSE_MONTH1_TRACE_MATRIX.md`: concrete Month 1 trace definitions and
  the latest pole-vs-parked-ego transfer results.
- `SCENESENSE_MONTH1_COMMANDS.md`: single Month 1 command sheet for OAI
  bring-up, camera-only OD/SEG, RGB+radar fusion, curbside accident evidence,
  network/app/RAN analyzers, and output checks.
- `SCENESENSE_RL_SCHEMA.md`: Month 1 RL state/action/reward/guardrail schema.
- `seg_radar_pole_fusion_readme.md`: two-pole fusion, parked-ego fusion, and
  spatial-map runbook.
- `payload_fusion_handoff_readme.md`: broader payload/fusion workflow notes.
- `SCENESENSE_FUSION_OAI_LOOPBACK_SLIDE_DECK.html`: OAI vs loopback
  presentation deck.
- `scenesense_scenarios/scenesense_scenario_harness.py`: CARLA scenario harness.
- `carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py`:
  current shared RGB+radar fusion runtime for pole and parked-ego platforms.
- `carla_split_inference_udp_fusion_object_ego_client.py`: parked-ego wrapper
  around the shared fusion runtime.
- `scripts/analyze_fusion_object_transfer.py`: offline `fusion_as_od`
  evaluator for object recall, localization error, yaw error, dimension error,
  precision, and false positives.

## What Is Working

### 1. Canonical Curbside Accident Scenario

The Month 1 hidden-pedestrian curbside scenario is validated.

Final animated evidence run:

```text
metrics_logs/scenesense_scenarios/20260602_125157_curbside_parked_vehicle_pedestrian_occlusion_seed7
```

Validation summary:

- `walker_control`
- target collisions: 9
- target crossing progress: 0.569
- target start to first collision: 4.35 s
- ego RGB frames: 88
- helper RGB frames: 89
- validation: PASS with `--require-collision`

Treat this as the Month 1 animated curbside evidence demo. Earlier
transform-forced collision runs exist, but sliding pedestrian motion makes them
less suitable for presentation.

### 2. OAI vs Loopback Fusion Transport

RGB+radar split fusion runs over:

- loopback/local transport
- OAI 5G transport
- two OAI UEs

Major finding:

- OAI 5G path has much higher RTT than loopback.
- Chunk-size sweeps did not remove the high RTT.
- OAI transport is working, but root-cause latency tracing is still pending.

Presentation assets:

```text
SCENESENSE_FUSION_OAI_LOOPBACK_SLIDE_DECK.html
metrics_logs/scenesense_analysis/fusion_oai_loopback_6000_180s_clean_20260603_presentation/
```

### 3. Parked-Ego Fusion Runtime

The shared fusion runtime now supports:

- `--sensor-platform pole`
- `--sensor-platform ego_vehicle`
- parked-ego vehicle spawn
- curb-side spawn offsets
- ego-mounted RGB camera
- ego-mounted radar
- co-located semantic-GT camera with IoU logging
- per-stream manifests and metrics CSVs

Wrapper:

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py ...
```

The wrapper defaults to:

- `--sensor-platform ego_vehicle`
- `--spatial-map-stream-id fusion_ego_front`, unless overridden
- `--enable-semantic-gt`
- `--npc-vehicles 0`, unless overridden
- `--npc-pedestrians 0`, unless overridden

### 4. Two Parked-Ego Streams

The current locked parked-ego pair uses nearby face-to-face parked vehicles so
both streams see common moving objects from different viewpoints.

Stream 1:

```text
stream_id: fusion_ego_front
blueprint: vehicle.lincoln.mkz
spawn index: 152
forward offset: 0.0 m
right offset: 3.0 m
z offset: 0.15 m
yaw offset: 0 deg
```

Stream 2:

```text
stream_id: fusion_ego_front_view_2
blueprint: vehicle.dodge.charger
spawn index: 152
forward offset: 8.0 m
right offset: 3.0 m
z offset: 0.15 m
yaw offset: 180 deg
```

If the ego streams are active in JSON but not visible on the spatial map, start
the spatial-map server without the tight traffic-light-14 focus crop. The TL14
crop is useful for pole views but can hide parked-ego detections.

## Latest Pole vs Parked-Ego Transfer Result

This result evaluates both:

- `fusion_as_seg`: semantic mask transfer.
- `fusion_as_od`: object/localization transfer.

Experiment:

- two pole streams
- two face-to-face parked-ego streams
- same fusion checkpoint
- same loopback transport
- CARLA semantic-segmentation camera used as mask ground truth

Per-stream segmentation results:

| Platform/view | Binary IoU | 3-class macro IoU | Vehicle IoU | Person IoU | Frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pole_tl14_view_1` | 0.420 | 0.565 | 0.805 | 0.000 | 604 |
| `pole_tl14_view_2` | 0.349 | 0.556 | 0.730 | 0.000 | 714 |
| `ego_front` | 0.370 | 0.468 | 0.459 | 0.000 | 602 |
| `ego_front_view_2` | 0.376 | 0.501 | 0.530 | 0.000 | 712 |

Platform averages:

| Platform | Binary IoU | 3-class macro IoU | Vehicle IoU |
| --- | ---: | ---: | ---: |
| Pole streams | 0.384 | 0.560 | 0.768 |
| Parked-ego streams | 0.373 | 0.485 | 0.495 |

Interpretation:

- foreground/binary segmentation transfers well
- vehicle IoU drops to about 64% of pole performance
- parked-ego fusion is usable enough to continue experiments
- parked-ego fine-tuning is justified if parked-ego viewpoint becomes central
- person IoU is zero in this run; inspect `gt_person_pixels` before treating it
  as a person-class failure

Presentation artifacts are committed despite `metrics_logs/` usually being
ignored:

```text
metrics_logs/scenesense_analysis/pole_vs_ego_transfer_presentation/pole_vs_ego_platform_average_iou.png
metrics_logs/scenesense_analysis/pole_vs_ego_transfer_presentation/pole_vs_ego_stream_iou_core_metrics.png
metrics_logs/scenesense_analysis/pole_vs_ego_transfer_presentation/pole_vs_ego_stream_iou_all_metrics.png
metrics_logs/scenesense_analysis/pole_vs_ego_transfer_presentation/pole_vs_ego_transfer_iou_summary.csv
```

### Fusion As Object Detection / Localization

Experiment:

- two pole streams collected first
- two face-to-face parked-ego streams collected second
- same fusion checkpoint: `checkpoints/fusion_object_best.pt`
- same loopback transport
- CARLA vehicle actor/bounding-box GT projected into the camera view
- parked ego vehicle itself excluded from parked-ego GT
- selected GT requires visible center, minimum projected area, and max distance
  in the strict transfer comparison

Strict 2 m XY match gate:

| Platform/view | GT | Predictions | Recall@2m | Mean XY error |
| --- | ---: | ---: | ---: | ---: |
| `fusion_tl_14` | 1158 | 9894 | 0.282 | 1.088 m |
| `fusion_tl_14_view_2` | 1586 | 9499 | 0.354 | 1.151 m |
| `fusion_ego_front` | 2049 | 5091 | 0.025 | 1.301 m |
| `fusion_ego_front_view_2` | 1322 | 3901 | 0.024 | 1.412 m |

Loose 5 m sensitivity gate:

| Platform/view | GT | Predictions | Recall@5m | Mean XY error |
| --- | ---: | ---: | ---: | ---: |
| `fusion_tl_14` | 1158 | 9894 | 0.523 | 2.103 m |
| `fusion_tl_14_view_2` | 1586 | 9499 | 0.637 | 2.019 m |
| `fusion_ego_front` | 2049 | 5091 | 0.136 | 3.361 m |
| `fusion_ego_front_view_2` | 1322 | 3901 | 0.108 | 3.054 m |

Interpretation:

- Pole-trained object localization is much stronger on pole views than
  parked-ego views.
- Parked-ego OD recall retention is about 8% at 2 m and about 21% at 5 m.
- Matched detections have plausible XY error, so this is not an obvious total
  coordinate-frame failure.
- The likely issue is object-center detection/confidence/viewpoint transfer.
- Parked-ego OD/localization needs fine-tuning or retraining before it can
  support spatial-map sharing.

Presentation deck:

```text
SceneSense_Fusion_Model_Transferability_OD_SEG.pptx
metrics_logs/scenesense_analysis/fusion_transferability_presentation/
```

## Important Commands

### Syntax Check

Run this after editing the shared fusion runtime or ego wrapper:

```bash
python3 -m py_compile \
  carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  carla_split_inference_udp_fusion_object_ego_client.py
```

### Parked-Ego Stream 1

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py \
  --sync-world \
  --ego-vehicle-blueprint vehicle.lincoln.mkz \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 0.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --fusion-checkpoint checkpoints/fusion_object_best.pt \
  --entropy-coder zlib \
  --npc-vehicles 20 \
  --npc-pedestrians 10 \
  --run-group exp_parked_ego_loopback_smoke \
  --transport-label loopback \
  --spatial-map-stream-id fusion_ego_front \
  --spatial-map-port 39201 \
  --camera-source-port 51201 \
  --remote-port 51202 \
  --remote-source-port 51203 \
  --camera-result-port 51204 \
  --result-timeout 1.5 \
  --headless
```

### Parked-Ego Stream 2

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py \
  --async-world \
  --ego-vehicle-blueprint vehicle.dodge.charger \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 8.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --ego-spawn-yaw-offset-deg 180.0 \
  --ego-camera-yaw 0.0 \
  --fusion-checkpoint checkpoints/fusion_object_best.pt \
  --entropy-coder zlib \
  --npc-vehicles 0 \
  --npc-pedestrians 0 \
  --run-group exp_parked_ego_loopback_smoke \
  --transport-label loopback \
  --spatial-map-stream-id fusion_ego_front_view_2 \
  --spatial-map-port 39201 \
  --camera-source-port 51301 \
  --remote-port 51302 \
  --remote-source-port 51303 \
  --camera-result-port 51304 \
  --result-timeout 1.5 \
  --headless
```

### Parked-Ego Training-Data Smoke Collection

This is different from live inference. It saves RGB images, 3-class masks,
semantic tags, radar tensors, radar point clouds, camera/radar calibration,
and object actor labels into the pole fusion training schema.

Smoke collection:

```bash
python3 carla_collect_parked_ego_fusion_training_data.py \
  --sync-world \
  --experiment-id parked_ego_fusion_training_smoke_20260604 \
  --max-samples 30 \
  --sample-stride 2 \
  --ego-vehicle-blueprint vehicle.lincoln.mkz \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 0.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --npc-vehicles 20 \
  --npc-pedestrians 10
```

Validate:

```bash
python3 scripts/validate_fusion_training_dataset.py \
  fusion_training_data/parked_ego_fusion_training_smoke_20260604 \
  --max-samples 30
```

Dry-run target construction:

```bash
python3 scripts/dry_run_fusion_training_targets.py \
  fusion_training_data/parked_ego_fusion_training_smoke_20260604 \
  --max-samples 30
```

Expected output folder structure:

```text
fusion_training_data/<experiment_id>/
  manifest.csv
  object_boxes.csv
  metadata.json
  validation_summary.json
  rgb/
  masks/
  semantic_tags/
  radar_tensors/
  radar_points/
```

### Pole Accuracy Runs With Semantic GT

For accuracy comparison, use
`carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py`
with `--role loopback` and `--enable-semantic-gt`. The older original pole
stream scripts do not include the new semantic-GT metric columns.

Use `seg_radar_pole_fusion_readme.md` for the exact pole geometry:

- pole stream 1: TL14, `camera-x=9`, `camera-y=2`, `camera-pitch=-30`,
  `camera-yaw-offset=50`
- pole stream 2: TL14, `camera-x=11`, `camera-y=2`, `camera-pitch=-30`,
  `camera-yaw-offset=120`

### Inspect Active Fusion Streams

```bash
curl http://127.0.0.1:35011/api/fusion_streams/latest | python3 -m json.tool
curl http://127.0.0.1:35011/api/spatial_map/latest | python3 -m json.tool
```

## What Is Left In Month 1

The project has made strong progress, but these checklist items remain open or
partially open:

1. `fusion_as_od` evaluation:
   - instrumentation, live pole-vs-ego run, offline evaluator, sensitivity
     analysis, plots, and transferability deck are complete
   - conclusion: parked-ego OD/localization transfer is poor enough to justify
     parked-ego fine-tuning/retraining

2. Camera-only task metrics:
   - camera-only OD GT projection + analyzer are implemented; run fresh
     loopback/OAI traces with `--enable-od-gt` and summarize them
   - camera-only SEG loopback mIoU/foreground/class IoU is collected; repeat
     over OAI 5G later only if the presentation needs OAI task-quality parity

3. Prior OD-vs-SEG payload root:
   - `AI_traffic_characterization_IDCC_template.pptx` is the slide-level
     OD-vs-SEG traffic-characterization artifact
   - it summarizes OD/SEG payload sizing, ROI/AE/quantization candidates, and
     5QI burst-volume gaps
   - the six-month proposal itself does not name a specific
     `od_seg_fair_latency_*` run root; its Month 1 exit criterion is broader:
     repeatable OD/SEG traces with bytes, latency, loss, AP/mIoU, foreground
     IoU, and class-specific misses
   - the raw root
     `metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_recovery_20260520_220356/`
     and fair-latency launcher/analyzer scripts are not present in the current
     local/remote mirror
   - reuse the structure described in `payload_fusion_handoff_readme.md` where
     helpful, but do not assume the raw root is available

4. Parked-ego training-data collection path:
   - live parked-ego RGB/radar inference is working
   - saved parked-ego training-schema export is smoke-validated
   - latest smoke dataset:
     `fusion_training_data/parked_ego_fusion_training_smoke_20260604`
   - target-builder dry run passed: 30 samples produced `(7, 432, 768)`
     fusion inputs, `(432, 768)` segmentation targets, positive vehicle object
     targets in 30/30 samples, 369 valid vehicle targets, and no errors/warnings
   - local training-driver search found fusion helpers but no standalone
     SceneSense fusion trainer; missing pieces are listed in
     `FUSION_TRAINING_DRIVER_GAP_ANALYSIS.md`
   - next step is to implement a minimal fine-tuning smoke trainer or locate a
     remote-only original trainer

5. OAI latency root cause:
   - symptom is documented and plotted
   - cause is not yet diagnosed
   - use network sampler, gNB/UE logs, T-tracer, and application metrics

## Next Recommended Task

Locate or recreate the fusion training/fine-tuning entry point.

Current state:

- Parked-ego RGB/radar live inference works.
- Parked-ego semantic-GT metrics work.
- Parked-ego object-GT logging works.
- Pole-vs-parked-ego SEG/OD transferability evidence is now available.
- OD/localization transfer is poor, so model adaptation is justified.
- Saved parked-ego training-data collector and validator are implemented.
- Remote smoke dataset `parked_ego_fusion_training_smoke_20260604` validates
  cleanly: 30 samples, 474 object rows, all samples have RGB/mask/radar, and
  object labels include vehicle/person.
- Target dry-run also passes cleanly: 30/30 samples build 7-channel fusion
  inputs, segmentation targets, heatmap/regression object targets, and 369 valid
  vehicle object positives.

Next concrete steps:

1. Check whether the remote/replica has an original trainer that is absent from
   the local checkout.
2. If a remote-only training driver is found, point it at
   `fusion_training_data/parked_ego_fusion_training_smoke_20260604` and run a
   tiny CPU/GPU fine-tuning smoke job.
3. If no original driver exists, recreate only a minimal smoke trainer first:
   dataset class, DataLoader, checkpoint loading/freezing policy, segmentation
   loss, object loss, metrics, and checkpoint output.
4. Reuse `scripts/dry_run_fusion_training_targets.py` as the target-building
   sanity check for any recreated dataset class.
5. After a tiny smoke job succeeds, update the checklist with the exact command,
   log path, and checkpoint/output artifact.

## Caveats And Gotchas

- Do not use the original pole stream scripts for semantic-GT comparison unless
  they have been updated; use the shared OAI loopback script.
- `metrics_logs/` is generally ignored by Git. Presentation plots were
  force-added only for the small pole-vs-ego transfer folder.
- If a map view looks blank but `/api/fusion_streams/latest` shows an active
  stream with objects, check focus/cropping before assuming transport failed.
- For parked ego, CARLA spawn points are lane spawn points. Use
  `--ego-spawn-right-offset-m` to nudge the vehicle toward curb-side parking.
- The person-IoU result from the latest transfer run is not meaningful until
  `gt_person_pixels` confirms visible person GT.
- OAI chunk-size changes did not fix the high-RTT issue. Do not spend another
  full day tuning chunk size unless new evidence points there.
- Keep OD and SEG metrics separate. The fusion model produces both outputs,
  but `fusion_as_seg` and `fusion_as_od` answer different research questions.

## Current Month 1 Story

The work now has a coherent Month 1 story:

1. Repeatable CARLA scenario harness exists.
2. Canonical hidden-pedestrian curbside accident is validated.
3. RGB+radar split fusion works locally and over OAI 5G.
4. OAI 5G transport is functional but higher-latency than loopback.
5. Two pole fusion streams feed the spatial map.
6. Two parked-ego fusion streams now work.
7. Pole-trained segmentation partially transfers to parked ego:
   foreground transfer is strong, vehicle IoU drops to about 64% of pole
   performance.
8. Object/localization logging, offline analysis, and pole-vs-ego OD transfer
   evidence are complete.
9. Saved parked-ego training samples are smoke-validated and target-builder
   validated. The next missing engineering proof is the actual training or
   fine-tuning entry point.

That is where the next session should start.
