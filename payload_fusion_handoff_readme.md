# Intern Handoff: OD-vs-SEG Payload Comparison, Trained LR-ASPP, and RGB+Radar Split Fusion

This handoff is for moving the collected CARLA payload-comparison data and the
relevant training/inference scripts to a remote Ubuntu machine that already has
CARLA 0.10 and a Python virtual environment.

The short version:

- The original fair comparison was automated by `./run_od_seg_fair_latency_comparison.sh`.
- The finished report-grade data root to copy first is
  `metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_recovery_20260520_220356/`.
- `carla_split_inference_udp_segmentation_trained_lraspp_demo.py` and
  `carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py` are
  trained LR-ASPP segmentation inference clients.
- The RGB+radar split-fusion training code is in `pole_lraspp_multimodal_fusion/`.
- Important naming correction: `carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py`
  is a traffic-light-pole RGB segmentation runtime, not a trainer, not an ego-vehicle
  client, and not the RGB+radar fusion trainer by itself.

## Recommended Remote Layout

Use the same absolute layout if possible. Several launcher/common files still
contain absolute defaults for the local testbed path.

```text
/home/shr_aisvcs/workarea/carla_0_10_env/
  carla_0_10_venv/
    bin/python3
  Carla-0.10.0-Linux-Shipping/
    CarlaUnreal.sh
    PythonAPI/
      neu_collab/
        ...
```

If the remote machine uses a different username or base directory, pass
`PYTHON_BIN=/remote/venv/bin/python3` to `run_od_seg_fair_latency_comparison.sh`.
For the training supervisors, also update the constants in:

- `pole_lraspp_training/pole_lraspp_training/common.py`
- `pole_lraspp_training/scripts/start_background.sh`
- `pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/common.py`
- `pole_lraspp_multimodal_fusion/launch_unattended_fusion_training.sh`

The relevant constants are `PROJECT_PYTHON`, `CARLA_BIN`, and the
`NEU_COLLAB_ROOT` / `WORKFLOW_ROOT` launcher paths.

## What To Copy

Run these commands from the source machine:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

REMOTE_USER_HOST=intern@REMOTE_HOSTNAME
REMOTE_NEU=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab
```

### 1. Core OD-vs-SEG fair-latency code

Copy these files to repeat the original object-detection vs segmentation
payload comparison:

```bash
rsync -avR \
  run_od_seg_fair_latency_comparison.sh \
  analyze_od_seg_fair_latency.py \
  carla_split_inference_udp_data_collect.py \
  carla_split_inference_udp_demo.py \
  carla_split_inference_udp_detection_quality_collect.py \
  evaluate_split_detection_coco_map.py \
  carla_split_inference_udp_segmentation_demo.py \
  evaluate_lraspp_segmentation_metrics.py \
  checkpoints/learned_gate.pt \
  checkpoints/rd_ae_b128.pt \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

Do not copy the whole `checkpoints/rd_features/` directory unless you know you
need it. The fair-latency launcher only requires `learned_gate.pt` and
`rd_ae_b128.pt`.

### 2. Existing completed comparison data

Copy the final consolidated run first. This is the main dataset for analysis:

```bash
rsync -avR \
  metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_recovery_20260520_220356 \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

Optional provenance data:

```bash
rsync -avR \
  metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_20260520_183513 \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

The `od_seg_fair_latency_5qi_smoke_20260520_214951/` root is useful only as a
smoke-test example. Do not mix it into final result claims unless the objective
is smoke-test debugging.

### 3. Trained RGB LR-ASPP segmentation code and data

Copy this if the intern needs to inspect or repeat the trained LR-ASPP
segmentation workflow:

```bash
rsync -avR \
  carla_split_inference_udp_segmentation_trained_lraspp_demo.py \
  carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py \
  pole_lraspp_training \
  experiments/pole_lraspp_training/20260505_173329_pole_lraspp_training \
  metrics_logs/trained_pole_segmentation_metrics \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

If bandwidth is limited, copy only the best checkpoint plus `manifest.json` and
the metrics/report files from the training experiment. Copy the full experiment
directory when the intern needs the dataset images, masks, object boxes, and
sample-level metrics.

### 4. RGB+radar fusion training and runtime code

Copy this if the intern will train, evaluate, or run the RGB+radar split-fusion
model:

```bash
rsync -avR \
  pole_lraspp_multimodal_fusion \
  carla_split_inference_udp_fusion_object_pole_client.py \
  carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py \
  carla_split_inference_udp_fusion_object_pole_client_spatial_stream_2.py \
  real_time_spatial_map_server_fusion_object_v1.py \
  real_time_spatial_map_server_fusion_object_v2.py \
  traffic_lights_data.json \
  extract_traffic_lights.py \
  seg_radar_pole_fusion_readme.md \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

Copy the preserved operational checkpoint:

```bash
rsync -avR \
  pole_lraspp_multimodal_fusion/preserved_checkpoints/run5_lowfuse_obj_sel_best.pt \
  pole_lraspp_multimodal_fusion/preserved_checkpoints/run5_test_metrics.json \
  pole_lraspp_multimodal_fusion/preserved_checkpoints/run5_val_metrics.json \
  pole_lraspp_multimodal_fusion/preserved_checkpoints/README.md \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

Copy the full source fusion experiment only if the intern needs the original
dataset, logs, and figures:

```bash
rsync -avR \
  experiments/pole_lraspp_multimodal_fusion/20260508_070718_pole_lraspp_multimodal_fusion_learned_localization \
  "${REMOTE_USER_HOST}:${REMOTE_NEU}/"
```

## Remote Environment Checks

On the remote machine:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3

"$PY" -c "import carla; print('carla import ok')"
"$PY" -c "import torch, torchvision, numpy, pandas, matplotlib, cv2; print('ml deps ok')"
"$PY" -m py_compile \
  analyze_od_seg_fair_latency.py \
  carla_split_inference_udp_segmentation_trained_lraspp_demo.py \
  carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py \
  carla_split_inference_udp_fusion_object_pole_client.py
```

Install missing Python packages in the remote venv. Typical requirements are
`numpy`, `pandas`, `matplotlib`, `opencv-python`, `torch`, `torchvision`,
`flask`, and `pyyaml`. The TC/netem experiments also require Linux `tc`
from `iproute2` and root/passwordless-sudo access for shaped profiles.

## How To Analyze The Existing OD-vs-SEG Data

The main completed run is:

```text
metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_recovery_20260520_220356/
```

Key files inside that root:

- `manifest.json`: top-level completion status.
- `resolved_config.json`: profiles, scenario settings, run matrix, UDP ports.
- `run.log`: launcher log for the whole matrix.
- `runs/<profile>/<pipeline>/<config>/per_frame_metrics.csv`: raw per-frame
  metrics from each CARLA run.
- `runs/<profile>/<pipeline>/<config>/run_manifest.json`: per-run manifest.
- `runs/<profile>/seg/<config>/seg_no_result_status.json`: present when SEG
  saturated and did not return valid masks.
- `eval_outputs/<profile>/od/coco_eval_summary.csv`: formal OD COCO metrics.
- `analysis/normalized_per_frame_latency.csv`: merged per-frame data with a
  common latency schema for OD and SEG.
- `analysis/latency_summary_by_profile.csv`: p50/p95 latency summaries.
- `analysis/payload_summary_by_profile.csv`: payload bytes, KiB, chunks, and
  Mbps-at-10-fps summaries.
- `analysis/quality_summary_by_profile.csv`: SEG dense metrics and OD summary
  fields kept separate by pipeline/config/profile.
- `analysis/od_seg_fair_latency_performance_analysis.md`: report-grade writeup.
- `figures/`: PNG/PDF figures for slides and discussion.

To regenerate post-processing:

```bash
EXP=metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_recovery_20260520_220356

"$PY" analyze_od_seg_fair_latency.py \
  --experiment-root "$EXP" \
  --min-frames-per-run 1200 \
  --also-pdf
```

Interpretation rules:

- Compare payloads using `payload_bytes`, `payload_kib`, `payload_chunks`, and
  `payload_mbit_per_s_at_10fps`.
- Compare latency using `total_app_latency_ms`, `network_round_trip_ms_est`,
  `one_way_delay_proxy_ms`, `front_ms`, `back_ms`, and `send_overlap_ms`.
- Keep OD and SEG quality metrics separate. OD uses COCO-style mAP summaries.
  SEG uses dense mask metrics such as mIoU, per-class IoU, and pixel accuracy.
- The 5QI profiles are 5QI-inspired TC/netem approximations, not standards
  conformant 3GPP QoS flows.
- Treat `cellular_congested / SEG` rows marked `transport_saturated_no_result`
  as saturation/no-return evidence. Do not include those rows in returned-result
  median latency or quality claims.

## How To Repeat The Original OD-vs-SEG Experiments

Start CARLA 0.10 first, ideally with Town10HD/Town10HD_Opt already loaded.
Then run:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab

PYTHON_BIN=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
DRY_RUN=1 \
./run_od_seg_fair_latency_comparison.sh
```

Run a short smoke test:

```bash
PYTHON_BIN=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
FRAMES_PER_RUN=120 \
WARMUP_FRAMES=20 \
PROFILE_SET=local_unlimited \
CONFIG_SET=baseline_only \
./run_od_seg_fair_latency_comparison.sh
```

Run the full comparison:

```bash
sudo -v

PYTHON_BIN=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
FRAMES_PER_RUN=1200 \
WARMUP_FRAMES=60 \
PROFILE_SET=core_plus_5qi \
CONFIG_SET=default \
SEG_NO_RESULT_CONTINUE=1 \
./run_od_seg_fair_latency_comparison.sh
```

Useful launcher knobs:

- `PROFILE_SET=local_unlimited|three_basics|5qi_only|core_plus_5qi`
- `CONFIG_SET=baseline_only|default`
- `RESUME_EXISTING=1 SKIP_COMPLETED=1 EXPERIMENT_ROOT=<existing-root>` to
  continue a partially completed matrix.
- `SKIP_ANALYSIS=1` if you only want raw collection first.

Outputs will be under:

```text
metrics_logs/od_seg_latency_comparison/od_seg_fair_latency_<timestamp>/
```

## Using Trained LR-ASPP In The Fair-Latency Matrix

The current `run_od_seg_fair_latency_comparison.sh` SEG path calls
`evaluate_lraspp_segmentation_metrics.py`. That evaluator imports
`carla_split_inference_udp_segmentation_demo.py` and writes the exact
`runs/<profile>/seg/<config>/per_frame_metrics.csv` layout expected by
`analyze_od_seg_fair_latency.py`.

The trained scripts are not drop-in replacements for that launcher yet:

- `carla_split_inference_udp_segmentation_trained_lraspp_demo.py` supports
  `--trained-experiment-dir`, but its default metrics filenames are timestamped
  CSVs, not `per_frame_metrics.csv`.
- `carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py` is
  pole-mounted and takes traffic-light/camera placement arguments that the
  original ego-vehicle SEG matrix does not pass.

Recommended additive path:

1. Create `evaluate_trained_lraspp_segmentation_metrics.py` by adapting
   `evaluate_lraspp_segmentation_metrics.py` to import
   `carla_split_inference_udp_segmentation_trained_lraspp_demo.py`.
2. Add `--trained-experiment-dir`, `--seg-num-classes 3`, and
   `--seg-class-scheme carla_3class`.
3. Keep the output contract identical to the current evaluator:
   `per_frame_metrics.csv`, `run_manifest.json`, `frame_index.jsonl`, and
   optional `masks/`.
4. Either make the evaluator read `TRAINED_EXPERIMENT_DIR` from the environment
   or add a narrow `SEG_EXTRA_ARGS` array to the launcher.
5. Then run:

```bash
SEG_COLLECT_SCRIPT="$PWD/evaluate_trained_lraspp_segmentation_metrics.py" \
TRAINED_EXPERIMENT_DIR="$PWD/experiments/pole_lraspp_training/<timestamp>_pole_lraspp_training" \
FRAMES_PER_RUN=1200 \
WARMUP_FRAMES=60 \
PROFILE_SET=core_plus_5qi \
CONFIG_SET=default \
./run_od_seg_fair_latency_comparison.sh
```

The pole-client version should be a separate additive launcher, for example
`run_trained_pole_seg_latency_comparison.sh`, because it needs pole-specific
arguments such as `--traffic-light-id`, `--camera-z`, `--camera-yaw-offset`,
`--camera-pitch`, `--camera-width`, and `--camera-height`.

## Training The RGB-Only LR-ASPP Segmentation Model

This is the trained model used by:

- `carla_split_inference_udp_segmentation_trained_lraspp_demo.py`
- `carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py`

The trainer is not inside the runtime client. It is:

```text
pole_lraspp_training/pole_lraspp_training/run_pipeline.py
```

Dry run:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_training

PYTHONPATH="$PWD:$(dirname "$PWD")" \
MPLBACKEND=Agg \
PYTHONUNBUFFERED=1 \
"$PY" -m pole_lraspp_training.run_pipeline \
  --config configs/default_config.json \
  --dry-run
```

Unattended training:

```bash
./scripts/start_background.sh configs/default_config.json
```

Expected training root:

```text
experiments/pole_lraspp_training/<timestamp>_pole_lraspp_training/
```

Important outputs:

- `dataset/manifest.csv`: RGB/mask paths and camera metadata.
- `dataset/object_boxes.csv`: CARLA object boxes used for object-size analysis.
- `checkpoints/<trial>/best.pt`: trained LR-ASPP checkpoint.
- `metrics/test_evaluation_metrics.json`: held-out mIoU and per-class IoU.
- `metrics/test_sample_metrics.csv`: per-sample dense segmentation metrics.
- `metrics/test_object_metrics.csv`: component-vs-GT object-size behavior.
- `final_report.txt`: summary and selected checkpoint.
- `manifest.json`: top-level status and `best_checkpoint`.

## Evaluating The Trained RGB LR-ASPP Split Runtime

Run the ego-mounted demo version when you want front-facing RGB from a vehicle:

```bash
EXP=experiments/pole_lraspp_training/<timestamp>_pole_lraspp_training
OUT="$EXP/split_inference_eval"
mkdir -p "$OUT"

"$PY" carla_split_inference_udp_segmentation_trained_lraspp_demo.py \
  --trained-experiment-dir "$EXP" \
  --town "" \
  --weather-preset unchanged \
  --headless \
  --disable-live-plot \
  --enable-data-collection \
  --enable-semantic-gt \
  --metrics-log-dir "$OUT" \
  --metrics-log-prefix trained_split_eval \
  --run-tag trained_lraspp_ego_rgb_baseline \
  --camera-resolution 720p \
  --fps 10 \
  --npc-vehicles 40 \
  --npc-pedestrians 20 \
  --max-frames 1200 \
  --metrics-warmup-frames 60 \
  --per-level-compress-probe
```

Run the pole-mounted version when you want a fixed traffic-light camera:

```bash
OUT="$EXP/split_inference_pole_eval"
mkdir -p "$OUT"

"$PY" carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py \
  --trained-experiment-dir "$EXP" \
  --traffic-light-id 14 \
  --camera-z 5.0 \
  --camera-yaw-offset 90 \
  --camera-pitch -35 \
  --camera-fov 100 \
  --camera-width 854 \
  --camera-height 480 \
  --fps 10 \
  --headless \
  --disable-live-plot \
  --enable-data-collection \
  --enable-semantic-gt \
  --metrics-log-dir "$OUT" \
  --metrics-log-prefix trained_pole_split_eval \
  --run-tag trained_lraspp_pole_baseline \
  --max-frames 1200 \
  --metrics-warmup-frames 60 \
  --per-level-compress-probe
```

Analyze the resulting CSVs:

```bash
"$PY" pole_lraspp_training/scripts/analyze_trained_split_eval.py \
  --glob "$OUT/*.csv" \
  --output-dir "$OUT/analysis"
```

The output analysis includes `summary.json`, `report.txt`, `grouped_summary.csv`,
and latency/payload/quality figures.

## Training The RGB+Radar Split-Fusion Model

Use this path for the new split-fusion model:

```text
pole_lraspp_multimodal_fusion/
```

The model consumes aligned RGB plus radar tensors:

- RGB image.
- Radar occupancy.
- Radar inverse range.
- Radar radial velocity.
- Radar stationary age.

The model outputs:

- 3-class segmentation mask: background, vehicle, person.
- Learned object-localization maps for object centers, sensor-relative XYZ,
  dimensions, yaw, parked/stopped state, and radar support confidence.

Dry run:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_multimodal_fusion

./launch_unattended_fusion_training.sh \
  --mode direct \
  --config configs/fusion_smoke.yaml \
  --dry-run
```

Short smoke training/evaluation:

```bash
./launch_unattended_fusion_training.sh \
  --mode screen \
  --session-name pole_lraspp_fusion_smoke \
  --config configs/fusion_smoke.yaml \
  --resume auto
```

Full run:

```bash
./launch_unattended_fusion_training.sh \
  --mode screen \
  --session-name pole_lraspp_fusion_full \
  --config configs/fusion_full_run.yaml \
  --resume auto
```

Monitor:

```bash
./status_unattended_fusion_training.sh
tail -f background_logs/pole_lraspp_fusion_full.screen.log
```

Expected fusion experiment root:

```text
experiments/pole_lraspp_multimodal_fusion/<timestamp>_pole_lraspp_multimodal_fusion_learned_localization/
```

Important outputs:

- `dataset/manifest.csv`: synchronized RGB/radar/mask/sample metadata.
- `dataset/object_boxes.csv`: CARLA actor/object ground truth.
- `dataset/radar_tensor/`: projected radar tensors.
- `dataset/radar_points/`: raw and projected radar point records.
- `checkpoints/<trial>/best.pt`: trained fusion checkpoint.
- `metrics/test_fusion_evaluation_metrics.json`: test mIoU, object precision,
  recall, F1, global XY MAE, dimension MAE, yaw MAE, parked accuracy.
- `metrics/test_learned_object_metrics.csv`: per-object prediction/GT matches.
- `figures/`: evaluation figures.
- `final_report.txt`: summary and chosen checkpoint.
- `manifest.json`: top-level status and `best_checkpoint`.

Operational baseline checkpoint:

```text
pole_lraspp_multimodal_fusion/preserved_checkpoints/run5_lowfuse_obj_sel_best.pt
```

Use it when you want a known-good inference starting point without retraining.

## Hosting The Fusion Model

Current ready-to-run fusion runtime is pole-mounted:

```bash
FUSION_CKPT=pole_lraspp_multimodal_fusion/preserved_checkpoints/run5_lowfuse_obj_sel_best.pt

"$PY" carla_split_inference_udp_fusion_object_pole_client.py \
  --traffic-light-id 14 \
  --camera-z 6.0 \
  --camera-pitch -35 \
  --camera-yaw-offset 90 \
  --camera-fov 100 \
  --camera-width 854 \
  --camera-height 480 \
  --fusion-checkpoint "$FUSION_CKPT" \
  --quantization-mode per_channel_uint8 \
  --entropy-coder zlib \
  --headless \
  --max-frames 1200
```

For spatial-map streaming, run `real_time_spatial_map_server_fusion_object_v2.py`
first, then use:

```bash
"$PY" real_time_spatial_map_server_fusion_object_v2.py \
  --object-yaw-map-offset-deg 10.0 \
  --focus-traffic-light-ids 14 \
  --focus-radius-m 20
```

In another terminal:

```bash
"$PY" carla_split_inference_udp_fusion_object_pole_client_spatial_stream.py \
  --sync-world \
  --traffic-light-id 14 \
  --camera-z 6.0 \
  --camera-pitch -35 \
  --camera-yaw-offset 90 \
  --camera-fov 100 \
  --camera-width 854 \
  --camera-height 480 \
  --fusion-checkpoint "$FUSION_CKPT" \
  --quantization-mode per_channel_uint8 \
  --entropy-coder zlib \
  --spatial-map-stream-id fusion_tl_14 \
  --spatial-map-port 39201 \
  --headless
```

The spatial-map server exposes:

```text
http://127.0.0.1:35011/api/spatial_map/viewer
http://127.0.0.1:35011/api/spatial_map/latest
http://127.0.0.1:35011/api/spatial_map/live.png
```

## Mobile Ego-Vehicle RGB+Radar Hosting

This is the requested target, but it is not currently a single existing command.
The current state is:

- `carla_split_inference_udp_segmentation_trained_lraspp_demo.py`: ego vehicle,
  front RGB camera, segmentation, no radar fusion.
- `carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py`: pole
  camera, segmentation, no radar fusion.
- `carla_split_inference_udp_fusion_object_pole_client.py`: pole RGB+radar
  split-fusion runtime.
- `pole_lraspp_multimodal_fusion/`: pole RGB+radar training pipeline.

To host the RGB+radar fusion model on a mobile ego vehicle, create an additive
runtime such as:

```text
carla_split_inference_udp_fusion_object_ego_client.py
```

Build it by combining:

- Hero vehicle spawning, autopilot/manual drive, and front camera mounting from
  `carla_split_inference_udp_demo.py`.
- Radar sensor spawning and RGB+radar tensor construction from
  `carla_split_inference_udp_fusion_object_pole_client.py`.
- Split runtime helpers from `pole_lraspp_multimodal_fusion/split_runtime.py`.
- UDP transport helpers from `carla_split_inference_udp_data_collect.py`.
- Semantic GT camera and metrics logging style from
  `carla_split_inference_udp_segmentation_trained_lraspp_demo.py`.

Minimum ego-client acceptance criteria:

- Spawns or attaches to a hero vehicle.
- Attaches a front RGB camera and front radar sensor to that same vehicle.
- Builds a 7-channel `[RGB, radar]` input exactly like the fusion training path.
- Logs per-frame `front_ms`, `back_ms`, `round_trip_ms`,
  `network_round_trip_ms_est`, `payload_bytes`, `payload_bytes_uncompressed`,
  `payload_chunks`, segmentation metrics, and object metrics.
- Tears down sensors, vehicle, UDP sockets, and worker threads on CTRL-C.
- Supports `--headless`, `--max-frames`, `--metrics-log-dir`, `--run-tag`,
  `--quantization-mode`, `--entropy-coder`, and four explicit UDP ports.

For a true ego-trained model rather than a pole-trained model tested on an ego
vehicle, also create an ego data-collection path. Reuse the fusion model and
training code, but collect from the ego vehicle's front camera/radar viewpoint
instead of traffic-light poles. Keep the same dataset schema where possible:
`dataset/manifest.csv`, `dataset/object_boxes.csv`, `dataset/radar_tensor/`,
`dataset/radar_points/`, `resolved_config.yaml`, and `manifest.json`.

## Payload Characterization For The New Fusion Model

Once the fusion ego client writes a per-frame CSV with the schema above, run the
same profile logic used in the OD-vs-SEG launcher:

1. `local_unlimited`: no TC shaping.
2. `lte_typical`: 20 Mbit, 25 ms delay, 5 ms jitter, 0.1 percent loss.
3. `cellular_congested`: 5 Mbit, 80 ms delay, 15 ms jitter, 0.2 percent loss.
4. `5qi85_dl_ai_ml`: 5QI-inspired low-latency DL profile.
5. `5qi88_ul_ai_ml`: 5QI-inspired UL split-AI profile.
6. `5qi89_split_render`: 5QI-inspired split-rendering profile.
7. `5qi90_visual`: 5QI-inspired visual-content profile.

Use config points such as:

- `baseline`: `--quantization-mode per_tensor_uint8 --entropy-coder zlib`.
- `quant_per_channel_uint8`: `--quantization-mode per_channel_uint8 --entropy-coder zlib`.
- `quant_per_channel_uint4`: if supported by the client transport.
- `zstd`: `--entropy-coder zstd --zstd-level 3` if `zstandard` is installed.

The output directory should look like:

```text
metrics_logs/fusion_payload_characterization/fusion_payload_<timestamp>/
  manifest.json
  resolved_config.json
  runs/<profile>/<config>/per_frame_metrics.csv
  analysis/normalized_per_frame_latency.csv
  analysis/latency_summary_by_profile.csv
  analysis/payload_summary_by_profile.csv
  analysis/quality_summary_by_profile.csv
  figures/
```

For analysis, either extend `analyze_od_seg_fair_latency.py` to accept a
`fusion` pipeline label or create an additive
`analyze_fusion_payload_characterization.py` that keeps the same normalized
columns. Do not overwrite the existing OD-vs-SEG analyzer unless that is a
deliberate versioned change.

## 3GPP And Testbed Alignment Notes

- Treat the TC/netem profiles as network-condition experiment knobs. The
  `5qi*` names are inspired by 3GPP QoS characteristics, but the script is not
  enforcing full 3GPP QoS flows, MDBV, reflective QoS, or OAI core policy.
- Keep spatial-map streaming aligned with the SS_SmManagement style already
  used by `real_time_spatial_map_server_v4.py` and the fusion spatial-map
  server. Runtime outputs should be retrievable by frame, stream/node id,
  object type, global position, and timestamp.
- Every new experiment root should include `manifest.json`,
  `resolved_config.json` or `resolved_config.yaml`, raw per-frame CSVs, analysis
  CSVs, figures, and a short report so the run is reproducible.
- Do not silently mutate the copied historical run directories. If reanalysis is
  needed, write a sibling `analysis_<timestamp>/` directory or a new timestamped
  experiment root.

## Intern Objectives And Acceptance Criteria

Objective 1: Understand prior OD-vs-SEG payload comparison.

- Inputs: copied `od_seg_fair_latency_recovery_20260520_220356/`.
- Outputs: a short summary using `analysis/*.csv`, `figures/`, and
  `analysis/od_seg_fair_latency_performance_analysis.md`.
- Acceptance: explains payload/chunk differences, returned-result latency,
  SEG saturation caveat, and OD mAP vs SEG mIoU separation.

Objective 2: Repeat the original OD-vs-SEG run on the remote machine.

- Inputs: fair-latency scripts, CARLA 0.10, venv, checkpoints.
- Outputs: new timestamped root under `metrics_logs/od_seg_latency_comparison/`.
- Acceptance: `manifest.json` complete, raw `runs/*/*/*/per_frame_metrics.csv`
  present, analysis CSVs and figures regenerated.

Objective 3: Train and evaluate RGB LR-ASPP segmentation.

- Inputs: `pole_lraspp_training/` and `configs/default_config.json`.
- Outputs: new `experiments/pole_lraspp_training/<timestamp>...` root and
  split-inference eval under that root.
- Acceptance: `manifest.json` complete, `best_checkpoint` exists,
  `metrics/test_evaluation_metrics.json` exists, split eval CSV and analysis
  report exist.

Objective 4: Train/evaluate RGB+radar fusion.

- Inputs: `pole_lraspp_multimodal_fusion/`, configs, RGB seed checkpoint.
- Outputs: new `experiments/pole_lraspp_multimodal_fusion/<timestamp>...` root.
- Acceptance: `manifest.json` complete, `best_checkpoint` exists,
  `metrics/test_fusion_evaluation_metrics.json` exists, object metrics are not
  all zero or NaN, and `final_report.txt` summarizes caveats honestly.

Objective 5: Prepare mobile ego fusion payload characterization.

- Inputs: new additive ego fusion client plus trained fusion checkpoint.
- Outputs: timestamped `metrics_logs/fusion_payload_characterization/` root.
- Acceptance: per-frame CSV includes the normalized latency/payload schema,
  analyzer writes summaries and figures, and saturated/no-result cases are
  marked instead of silently dropped.

## Common Failure Modes

- CARLA import fails: activate the correct venv or install the CARLA Python API
  wheel/egg into the venv.
- TC/netem profiles fail: run `sudo -v` first or configure passwordless sudo for
  `/sbin/tc`.
- Remote path differs: update hardcoded `PROJECT_PYTHON`, `CARLA_BIN`, and
  launcher root constants before training.
- No traffic-light ID `14`: load the same Town10HD/Town10HD_Opt map or regenerate
  `traffic_lights_data.json`.
- Manual driving with `--headless`: do not combine them. Use headless autopilot
  for unattended sweeps and graphical/manual mode only on a machine with display.
- SEG quality looks missing under congestion: check for `seg_no_result_status.json`
  and treat it as saturation evidence, not as a valid returned-result quality row.
