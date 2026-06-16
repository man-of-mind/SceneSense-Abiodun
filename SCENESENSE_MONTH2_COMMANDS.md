# SceneSense Month 2 Reproducible Commands

Last updated: 2026-06-12

Purpose: one command sheet for Month 2 work: static task/payload sweeps,
model-transferability checks, and the first controller-shaped OD/SEG scheduler
harness.

## Common Setup

Run from the editable project folder:

```bash
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
```

Assumptions:

- CARLA 0.10 is running and reachable at `127.0.0.1:2000`.
- Start with local loopback. OAI/5QI work is intentionally deferred until the
  model and route behavior are clear.
- True OD currently means the camera-only Faster R-CNN split route.
- RGB+radar fusion currently means the fusion segmentation/localization route;
  do not treat its localization output as true OD boxes/classes/AP.

## 1. Single-Ego OD/SEG Controller Harness

Script:

```text
scenesense_single_ego_task_coordinator.py
```

Architecture:

- One CARLA ego vehicle.
- One RGB camera.
- OD and SEG receive the same frame stream.
- OD and SEG use separate UDP feature/result port groups.
- A timer gate chooses which task is active.
- The inactive task logs `tx_active=0` and sends zero feature payload.

### Visual Run

Use this when you want to inspect the live task switching. The defaults match
the original OD/SEG demo traffic density: 20 NPC vehicles and 30 pedestrians.

```bash
python3 scenesense_single_ego_task_coordinator.py \
  --run-duration-s 60 \
  --od-seconds 10 \
  --seg-seconds 5 \
  --startup-task od \
  --camera-resolution 480p \
  --fps 5 \
  --run-tag-prefix month2_single_ego_visual
```

### Headless Smoke

Use this before changing compression/task knobs:

```bash
python3 scenesense_single_ego_task_coordinator.py \
  --run-duration-s 30 \
  --od-seconds 10 \
  --seg-seconds 5 \
  --startup-task od \
  --camera-resolution 480p \
  --fps 5 \
  --headless \
  --run-tag-prefix month2_single_ego_smoke
```

Expected outputs:

```text
metrics_logs/single_ego_task_coordinator/<run>_od_<timestamp>.csv
metrics_logs/single_ego_task_coordinator/<run>_seg_<timestamp>.csv
metrics_logs/single_ego_task_coordinator/<run>_gate_events_<timestamp>.csv
metrics_logs/single_ego_task_coordinator/<run>_manifest_<timestamp>.json
```

### Measurement Run

Use this for a cleaner 180 s task-scheduling trace:

```bash
python3 scenesense_single_ego_task_coordinator.py \
  --run-duration-s 180 \
  --od-seconds 10 \
  --seg-seconds 5 \
  --startup-task od \
  --camera-resolution 480p \
  --fps 5 \
  --metrics-warmup-frames 10 \
  --headless \
  --run-tag-prefix month2_single_ego_timer_baseline
```

```bash
python3 scenesense_single_ego_task_coordinator.py   --run-duration-s 180   --od-seconds 10   --seg-seconds 5   --startup-task od   --camera-resolution 480p   --fps 5   --npc-vehicles 20   --npc-pedestrians 30   --run-tag-prefix local_single_ego_visual
```

## 2. RGB-Only Ego Transfer Client

Script:

```text
carla_split_inference_udp_rgb_ego_transfer_client.py
```

Use this to verify RGB-only OD/SEG routes from moving/autopilot and static
parked ego viewpoints without modifying the clean single-ego controller
architecture script.

Run the commands in this section from:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
```

Expected outputs:

```text
metrics_logs/rgb_ego_transfer/<run>_od_<timestamp>.csv
metrics_logs/rgb_ego_transfer/<run>_seg_<timestamp>.csv
metrics_logs/rgb_ego_transfer/<run>_gate_events_<timestamp>.csv
metrics_logs/rgb_ego_transfer/<run>_manifest_<timestamp>.json
```

### Clean RGB-Only Transferability Runs

For this round, keep only one task active for the full run. That gives cleaner
task accuracy numbers than alternating OD/SEG. The timer windows below are
longer than the 180 s run duration, so the inactive task stays quiet.

Use the original RGB-only demo defaults unless we explicitly change one item:
Town10HD_Opt, 10 FPS, 20 background vehicles, 30 pedestrians, front RGB camera
at x=1.6 m and z=1.7 m. For the moving-model runs below, use 1080p because that
is the documented OD demo command. For the pole-trained RGB-only SEG checkpoint,
remember the training data was collected at 854x480/480p; a 1080p pole-trained
run is useful, but should be described as a runtime stress variant rather than a
training-matched source-domain test.

1. Moving RGB-only SEG, moving ego source-domain reference:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task seg \
  --seg-seconds 240 \
  --od-seconds 1 \
  --camera-resolution 1080p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --seg-route moving \
  --enable-semantic-gt \
  --headless \
  --run-tag-prefix month2_clean_moving_rgb_seg_moving_1080p
```

2. Moving RGB-only SEG, parked ego in the same moving-demo spawn area:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task seg \
  --seg-seconds 240 \
  --od-seconds 1 \
  --ego-mode parked \
  --camera-resolution 1080p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --seg-route moving \
  --enable-semantic-gt \
  --headless \
  --run-tag-prefix month2_clean_moving_rgb_seg_parked_same_area_1080p
```

3. Pole-trained RGB-only SEG, live pole source-domain reference at TL14:

```bash
python3 carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py \
  --run-duration-s 180 \
  --traffic-light-id 14 \
  --camera-x 9 \
  --camera-y 2 \
  --camera-z 6.0 \
  --camera-pitch -30 \
  --camera-yaw-offset 50 \
  --camera-fov 100 \
  --camera-resolution 480p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --npc-vehicles 20 \
  --npc-pedestrians 30 \
  --seg-weights-path experiments/pole_lraspp_training/20260505_173329_pole_lraspp_training/checkpoints/adamw_640x360_lr1e-4_wd1e-4_aug_medium_bs6/best.pt \
  --enable-semantic-gt \
  --headless \
  --metrics-log-dir metrics_logs/rgb_ego_transfer \
  --metrics-log-prefix month2_clean_pole_rgb_seg_tl14_source_480p \
  --run-tag month2_clean_pole_rgb_seg_tl14_source_480p
```

4. Pole-trained RGB-only SEG, parked ego near TL14:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task seg \
  --seg-seconds 240 \
  --od-seconds 1 \
  --ego-mode parked \
  --ego-spawn-anchor-x -31.93084144592285 \
  --ego-spawn-anchor-y 20.30195426940918 \
  --ego-spawn-anchor-label tl14 \
  --camera-resolution 480p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --headless \
  --run-tag-prefix month2_clean_pole_rgb_seg_parked_near_tl14_480p
```

The default pole-trained checkpoint is resolved from:

```text
experiments/pole_lraspp_training/20260505_173329_pole_lraspp_training
```

The canonical selected checkpoint is:

```text
checkpoints/adamw_640x360_lr1e-4_wd1e-4_aug_medium_bs6/best.pt
```

### Fusion-Matched Pole-Trained SEG Rerun

Use this mini-batch when comparing the pole-trained RGB-only SEG checkpoint
against the Month 1 RGB+radar fusion camera geometry. The earlier generic
parked-near-TL14 command is useful as a quick parked test, but these commands
match the actual fusion transferability setup more closely:

- TL14 pole view 1: `camera-x=9`, `camera-y=2`, pitch `-30`, yaw offset `50`,
  FoV `100`.
- TL14 pole view 2: `camera-x=11`, `camera-y=2`, pitch `-30`, yaw offset
  `120`, FoV `100`.
- Parked ego view 1: spawn index `152`, right offset `3 m`, Lincoln MKZ.
- Parked ego view 2: same spawn index, `8 m` forward offset, `180 deg` yaw
  offset, Dodge Charger.
- Parked RGB-only transfer runs are still one-client/self-contained runs; they
  reuse the fusion parked ego poses, but do not yet run as a simultaneous
  two-ego shared-world pair.
- The two TL14 source commands below are written as standalone source-domain
  metric runs. For an exact concurrent two-pole replay, start view 1 first, then
  run view 2 with `--async-world --npc-vehicles 0 --npc-pedestrians 0`, matching
  the old fusion stream-2 behavior.

1. Pole-trained RGB-only SEG, TL14 fusion pole view 1:

```bash
python3 carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py \
  --run-duration-s 180 \
  --sync-world \
  --traffic-light-id 14 \
  --camera-x 9 \
  --camera-y 2 \
  --camera-z 6.0 \
  --camera-pitch -30 \
  --camera-yaw-offset 50 \
  --camera-roll 0 \
  --camera-fov 100 \
  --camera-resolution 480p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --npc-vehicles 20 \
  --npc-pedestrians 30 \
  --camera-source-port 51001 \
  --remote-port 51002 \
  --remote-source-port 51003 \
  --camera-result-port 51004 \
  --result-timeout 1.5 \
  --seg-weights-path experiments/pole_lraspp_training/20260505_173329_pole_lraspp_training/checkpoints/adamw_640x360_lr1e-4_wd1e-4_aug_medium_bs6/best.pt \
  --enable-semantic-gt \
  --headless \
  --metrics-log-dir metrics_logs/rgb_ego_transfer \
  --metrics-log-prefix month2_fusionmatched_pole_rgb_seg_tl14_view1_480p \
  --run-tag month2_fusionmatched_pole_rgb_seg_tl14_view1_480p
```

2. Pole-trained RGB-only SEG, TL14 fusion pole view 2:

```bash
python3 carla_split_inference_udp_segmentation_trained_lraspp_pole_client.py \
  --run-duration-s 180 \
  --sync-world \
  --traffic-light-id 14 \
  --camera-x 11 \
  --camera-y 2 \
  --camera-z 6.0 \
  --camera-pitch -30 \
  --camera-yaw-offset 120 \
  --camera-roll 0 \
  --camera-fov 100 \
  --camera-resolution 480p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --npc-vehicles 20 \
  --npc-pedestrians 30 \
  --camera-source-port 51101 \
  --remote-port 51102 \
  --remote-source-port 51103 \
  --camera-result-port 51104 \
  --result-timeout 1.5 \
  --seg-weights-path experiments/pole_lraspp_training/20260505_173329_pole_lraspp_training/checkpoints/adamw_640x360_lr1e-4_wd1e-4_aug_medium_bs6/best.pt \
  --enable-semantic-gt \
  --headless \
  --metrics-log-dir metrics_logs/rgb_ego_transfer \
  --metrics-log-prefix month2_fusionmatched_pole_rgb_seg_tl14_view2_480p \
  --run-tag month2_fusionmatched_pole_rgb_seg_tl14_view2_480p
```

3. Pole-trained RGB-only SEG, fusion parked ego view 1:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task seg \
  --seg-seconds 240 \
  --od-seconds 1 \
  --ego-mode parked \
  --vehicle-blueprint vehicle.lincoln.mkz \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 0.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --camera-resolution 480p \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --camera-yaw 0.0 \
  --camera-roll 0.0 \
  --camera-fov 100 \
  --fps 10 \
  --npc-vehicles 20 \
  --npc-pedestrians 10 \
  --metrics-warmup-frames 10 \
  --result-timeout 1.5 \
  --seg-port-base 51201 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --headless \
  --run-tag-prefix month2_fusionmatched_pole_rgb_seg_parked_view1_480p
```

4. Pole-trained RGB-only SEG, parked ego near TL14 with matched traffic:

Use this as the preferred parked-ego comparison against TL14 pole view 1. It
places the ego at the valid road spawn nearest the TL14 pole location and uses
the same background traffic density as the pole run, so the parked camera is
more likely to see the same kind of passing-vehicle traffic.

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task seg \
  --seg-seconds 240 \
  --od-seconds 1 \
  --ego-mode parked \
  --vehicle-blueprint vehicle.lincoln.mkz \
  --ego-spawn-anchor-x -31.93084144592285 \
  --ego-spawn-anchor-y 20.30195426940918 \
  --ego-spawn-anchor-label tl14 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --camera-resolution 480p \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --camera-yaw 0.0 \
  --camera-roll 0.0 \
  --camera-fov 100 \
  --fps 10 \
  --npc-vehicles 20 \
  --npc-pedestrians 30 \
  --metrics-warmup-frames 10 \
  --result-timeout 1.5 \
  --seg-port-base 51401 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --headless \
  --run-tag-prefix month2_tl14_pole_rgb_seg_parked_trafficmatched_480p
```

To visually confirm the passing-vehicle view before collecting metrics, run the
same command without `--headless` and shorten `--run-duration-s` to `60`.

5. Pole-trained RGB-only SEG, fusion parked ego view 2:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task seg \
  --seg-seconds 240 \
  --od-seconds 1 \
  --ego-mode parked \
  --vehicle-blueprint vehicle.dodge.charger \
  --ego-spawn-index 152 \
  --ego-spawn-forward-offset-m 8.0 \
  --ego-spawn-right-offset-m 3.0 \
  --ego-spawn-z-offset-m 0.15 \
  --ego-spawn-yaw-offset-deg 180.0 \
  --camera-resolution 480p \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --camera-yaw 0.0 \
  --camera-roll 0.0 \
  --camera-fov 100 \
  --fps 10 \
  --npc-vehicles 20 \
  --npc-pedestrians 10 \
  --metrics-warmup-frames 10 \
  --result-timeout 1.5 \
  --seg-port-base 51301 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --headless \
  --run-tag-prefix month2_fusionmatched_pole_rgb_seg_parked_view2_480p
```

6. Moving RGB-only OD, moving ego source-domain reference:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task od \
  --od-seconds 240 \
  --seg-seconds 1 \
  --camera-resolution 1080p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --enable-od-gt \
  --od-gt-iou-threshold 0.3 \
  --od-gt-min-area-px 400 \
  --od-gt-max-distance-m 50 \
  --headless \
  --run-tag-prefix month2_clean_moving_rgb_od_moving_1080p
```

7. Moving RGB-only OD, parked ego in the same moving-demo spawn area:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 180 \
  --startup-task od \
  --od-seconds 240 \
  --seg-seconds 1 \
  --ego-mode parked \
  --camera-resolution 1080p \
  --fps 10 \
  --metrics-warmup-frames 10 \
  --enable-od-gt \
  --od-gt-iou-threshold 0.3 \
  --od-gt-min-area-px 400 \
  --od-gt-max-distance-m 50 \
  --headless \
  --run-tag-prefix month2_clean_moving_rgb_od_parked_same_area_1080p
```

### Controller Optional: Compute Muted Front Halves

Default behavior skips inactive-task front-half compute. Use this only when we
want to measure the "what would compute cost have been?" counterfactual while
still muting network transmission:

```bash
python3 scenesense_single_ego_task_coordinator.py \
  --run-duration-s 60 \
  --od-seconds 10 \
  --seg-seconds 5 \
  --startup-task od \
  --camera-resolution 480p \
  --fps 5 \
  --compute-muted-fronts \
  --headless \
  --run-tag-prefix month2_single_ego_compute_muted
```

### Quick CSV Summary

Replace `<folder>` with `single_ego_task_coordinator` for controller runs or
`rgb_ego_transfer` for transferability runs. Replace the CSV names with the
latest run:

```bash
python3 - <<'PY'
import csv, math, pathlib

paths = {
    "od": pathlib.Path("metrics_logs/<folder>/<od_csv>"),
    "seg": pathlib.Path("metrics_logs/<folder>/<seg_csv>"),
}

for name, path in paths.items():
    rows = list(csv.DictReader(path.open(newline="")))
    active = [r for r in rows if int(float(r.get("tx_active") or 1)) == 1]
    muted = [r for r in rows if int(float(r.get("tx_active") or 1)) == 0]
    rtt = [
        float(r["round_trip_ms"])
        for r in active
        if r["round_trip_ms"] and not math.isnan(float(r["round_trip_ms"]))
    ]
    payload = [int(float(r["payload_bytes"])) for r in active]
    print(
        name,
        "rows", len(rows),
        "active", len(active),
        "muted", len(muted),
        "mean_payload_B", round(sum(payload) / max(1, len(payload)), 1),
        "mean_rtt_ms", round(sum(rtt) / max(1, len(rtt)), 2),
    )
PY
```

Reference local visual smoke:

```text
local_single_ego_visual_20260610_142722
OD: 550 rows, 434 active, 116 muted, mean active payload 88.1 kB, mean RTT 11.6 ms
SEG: 550 rows, 116 active, 434 muted, mean active payload 414.6 kB, mean RTT 16.0 ms
Gate: OD/SEG switched on the 10 s / 5 s cadence.
```

## 3. Model Transferability Plan, Local Loopback First

Goal: separate the model-transferability question from the network question.

### Track A: RGB-Only SEG Transferability

Question:

> Does the moving RGB-only LR-ASPP SEG model transfer better to parked ego
> viewpoints than the pole-trained RGB-only SEG model?

Runs to collect:

- Moving/autopilot RGB-only SEG, baseline.
- Moving RGB-only SEG on parked ego.
- Pole-trained RGB-only SEG on pole camera, baseline.
- Pole-trained RGB-only SEG on parked ego.

Metrics:

- Foreground/binary IoU.
- 3-class macro mIoU.
- Vehicle IoU.
- Person IoU when visible.
- Payload/latency for the split route.

Ground truth:

- Co-located CARLA semantic-segmentation camera, evaluation-only.

### Track B: RGB-Only OD Transferability

Question:

> Does the moving RGB-only Faster R-CNN OD route work acceptably from parked
> ego viewpoints?

Runs to collect:

- Moving/autopilot RGB-only OD, baseline.
- Moving RGB-only OD on parked ego.

Metrics:

- Actor-projection recall/precision proxy.
- Vehicle/person recall.
- Mean matched IoU.
- Payload/latency for the split route.

Ground truth:

- CARLA actors/transforms/bounding boxes projected into the RGB camera.

### Track C: Fusion SEG/Localization Transferability

Question:

> How much of the pole-trained RGB+radar fusion SEG/localization route survives
> when moved to parked ego viewpoints?

Status:

- Pole-vs-parked-ego transfer evidence exists from Month 1.
- Treat this as fusion SEG/localization evidence, not true OD evidence.
- Re-run only if supervisor asks for fresh data after the terminology cleanup.

### Track D: Supervisor Decision, Model-First Training

Decision after the 2026-06-11 supervisor discussion:

- The moving RGB-only SEG route is mostly pretrained and not CARLA-trained, so
  its low IoU is expected.
- The pole-trained RGB-only SEG route performs better because it is trained on
  CARLA classes.
- Next action is to collect parked-ego CARLA RGB+radar data at a dense
  intersection and train parked-ego RGB+radar models before returning to OAI
  5QI/QoS questions.

Local training workflows found:

```text
pole_lraspp_training/
  pole_lraspp_training/collect_dataset.py
  pole_lraspp_training/train_lraspp.py
  pole_lraspp_training/evaluate_lraspp.py
  pole_lraspp_training/run_pipeline.py

pole_lraspp_multimodal_fusion/
  pole_lraspp_multimodal_fusion/collect_dataset.py
  pole_lraspp_multimodal_fusion/train_fusion.py
  pole_lraspp_multimodal_fusion/evaluate_fusion.py
  pole_lraspp_multimodal_fusion/run_pipeline.py
  pole_lraspp_multimodal_fusion/model.py
  pole_lraspp_multimodal_fusion/object_targets.py
  pole_lraspp_multimodal_fusion/radar_fusion.py
```

Current interpretation:

- `pole_lraspp_multimodal_fusion` is reusable for RGB+radar SEG/localization.
- A true RGB+radar OD trainer was not found locally yet. Ask the supervisor if
  one exists; otherwise we need to define a separate OD model family or extend
  the current object-head/evaluator into a real boxes/classes OD pipeline.
- Before any overnight training, normalize the copied launcher paths under
  `abiodun/pole_lraspp_multimodal_fusion/`; the shell scripts still assume a
  root-level `neu_collab/pole_lraspp_multimodal_fusion` workflow.

Immediate model-first sequence:

1. Scout intersections and choose a parked-ego viewpoint with enough moving
   vehicles and pedestrians.
2. Collect a small RGB+radar parked-ego pilot dataset.
3. Run dataset validation and target dry-run.
4. Run a tiny training smoke job.
5. Launch full training overnight only after the smoke job writes a usable
   checkpoint.

### Scout Parked-Ego RGB+Radar Training Views

Run this first while CARLA is open in Town10HD/Town10HD_Opt. It ranks real
map spawn points near traffic-light/intersection anchors and writes both CSV
and Markdown outputs under `metrics_logs/parked_ego_view_scout/`.

```bash
python3 scenesense_scenarios/scout_parked_ego_training_views.py \
  --top 15 \
  --min-distance-m 12 \
  --max-distance-m 45 \
  --target-distance-m 24 \
  --coverage-range-m 95 \
  --camera-fov 120 \
  --right-offsets-m 0,3,-3 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --spawn-radius 95 \
  --camera-width 1280 \
  --camera-height 720
```

Supervisor-preferred right-side-road scout:

```bash
python3 scenesense_scenarios/scout_parked_ego_training_views.py \
  --top 20 \
  --min-distance-m 12 \
  --max-distance-m 45 \
  --target-distance-m 24 \
  --coverage-range-m 100 \
  --camera-fov 120 \
  --forward-offsets-m=-8,-4,0,4,8 \
  --right-offsets-m=4,5,6,7,8 \
  --preferred-lateral-side right \
  --require-preferred-lateral-side \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --spawn-radius 95 \
  --camera-width 1280 \
  --camera-height 720
```

Use this right-side scout before full collection. The goal is a parked ego that
is out of the active travel lane, close to the parking/curb lane, and sees
multiple traffic profiles: vehicles crossing in front, vehicles coming toward
it, vehicles passing along its side, and pedestrians near crosswalks. The scout
output now labels `parking_side` and includes forward/right offsets in the
generated experiment IDs so right-offset variants do not overwrite each other.

For the first full parked-ego training set, prefer three density profiles over
one giant single-density run:

- low/clear: 4,000 saved samples
- medium: 4,000 saved samples
- crowded: 4,000 saved samples

Use `--sample-stride 2` for the full dataset to reduce near-duplicate adjacent
frames while keeping temporal motion smooth.

Selected parked-ego training view after visual inspection:

- TL16, spawn `80`
- forward offset `4.0 m`
- right offset `7.0 m`
- yaw offset `-28.414 deg`
- rationale: ego is shifted out of the active travel lane toward the
  parking/curb lane while preserving oncoming, crossing, side-passing, and
  pedestrian/crosswalk profiles

### V2 Multi-View Parked-Ego Inspection

Use these before collecting any V2 data. The goal is to keep the V1 view as an
anchor and add 1-2 nearby parked views that improve crosswalk/person visibility,
side-passing profiles, and medium/crowded traffic coverage. Run them one at a
time with the preview window enabled and pick the views that are visibly useful.

View A, V1 anchor:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 90 \
  --startup-task seg \
  --seg-seconds 120 \
  --od-seconds 1 \
  --ego-mode parked \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 4.0 \
  --ego-spawn-right-offset-m 7.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --fps 10 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --seg-port-base 52101 \
  --run-tag-prefix month2_v2_viewA_spawn80_right7_fwd4_visual
```

Candidate View B, TL16 near-intersection curb view. This keeps the same general
road approach as View A, but moves the parked ego closer to the intersection and
slightly farther right so it sees more crossing/side-passing traffic:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 90 \
  --startup-task seg \
  --seg-seconds 120 \
  --od-seconds 1 \
  --ego-mode parked \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 16.0 \
  --ego-spawn-right-offset-m 8.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --fps 10 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --seg-port-base 52201 \
  --run-tag-prefix month2_v2_viewB_spawn80_right8_fwd16_visual
```

Candidate View C, View-B side-looking camera probe. This is the same parked ego
pose as View B, but with the camera yawed left to get a wider across-intersection
perspective without moving the parked ego into a drive lane:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 90 \
  --startup-task seg \
  --seg-seconds 120 \
  --od-seconds 1 \
  --ego-mode parked \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 16.0 \
  --ego-spawn-right-offset-m 8.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --camera-yaw -35.0 \
  --fps 10 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --seg-port-base 52301 \
  --run-tag-prefix month2_v2_viewC_tl16_viewB_cam_yawm35_visual
```

Optional TL14 shifted-right diversity probe. Use this only if we want a
different-intersection training view after View A and View B. The first TL14
visual landed in a drive lane, so this version shifts the ego farther toward
local right; inspect before using it for collection:

```bash
python3 carla_split_inference_udp_rgb_ego_transfer_client.py \
  --run-duration-s 90 \
  --startup-task seg \
  --seg-seconds 120 \
  --od-seconds 1 \
  --ego-mode parked \
  --ego-spawn-index 52 \
  --ego-spawn-forward-offset-m -8.0 \
  --ego-spawn-right-offset-m -2.0 \
  --ego-spawn-yaw-offset-deg -5.225 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --camera-x 1.8 \
  --camera-y 0.0 \
  --camera-z 1.55 \
  --camera-pitch -4.0 \
  --fps 10 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --seg-route pole_trained \
  --enable-semantic-gt \
  --seg-port-base 52501 \
  --run-tag-prefix month2_v2_tl14_shifted_right_spawn52_rightm2_fwdm8_visual
```

Selection rule:

- Keep View A as the baseline anchor.
- Add View B only if it is a plausible parked position and gives a genuinely
  different angle from View A.
- Add View C only as camera-orientation diversity from the valid View-B parked
  pose. For fusion data collection, yaw the RGB, semantic, and radar sensors
  together so the RGB/radar tensors remain aligned.
- Use the TL14 diversity view only after a visual pass confirms it is not in a
  drive lane.
- Reject any view where the parked ego blocks traffic, stares mostly at stopped
  vehicles, or sees too few pedestrians/vehicles for long stretches.

Automated View B and View A+B training pipeline:

```bash
mkdir -p logs

nohup bash scripts/run_viewB_viewAB_fusion_training_pipeline.sh \
  > logs/viewB_viewAB_pipeline_20260612.log 2>&1 &

tail -f logs/viewB_viewAB_pipeline_20260612.log
```

If the pipeline already completed View B collection/merge and stops during an
offline evaluation/plotting step, resume training with evaluation disabled:

```bash
RUN_EVAL=0 \
nohup bash scripts/run_viewB_viewAB_fusion_training_pipeline.sh \
  > logs/viewB_viewAB_pipeline_resume_noeval_20260615.log 2>&1 &

tail -f logs/viewB_viewAB_pipeline_resume_noeval_20260615.log
```

By default this collects `4000` samples for each View B density profile
low/medium/crowded, giving `12000` View B samples and `24000` View A+B samples
after merging with the existing View A dataset. The script stops the CARLA
server after collection/validation and before GPU training/evaluation so the
model has more GPU headroom. To keep CARLA running, launch with
`STOP_CARLA_BEFORE_TRAINING=0`.

If you intentionally want
`6000` samples per density instead, launch with:

```bash
SAMPLES_PER_DENSITY=6000 \
nohup bash scripts/run_viewB_viewAB_fusion_training_pipeline.sh \
  > logs/viewB_viewAB_pipeline_6000perdensity_20260612.log 2>&1 &
```

Note: `6000` samples per density gives `18000` View B samples and `30000`
combined View A+B samples, not `24000`.

Evaluate View A / View B / combined A+B checkpoints after training:

```bash
mkdir -p logs

nohup bash scripts/run_viewB_viewAB_fusion_eval_only.sh \
  > logs/viewB_viewAB_eval_20260616.log 2>&1 &

tail -f logs/viewB_viewAB_eval_20260616.log
```

Outputs:

```text
analysis_outputs/parked_ego_fusion_viewB_viewAB_eval_summary_20260612.csv
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_segmentation_bars.png
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_localization_bars.png
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_localization_precision_recall_bars.png
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_miou_matrix.png
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_vehicle_iou_matrix.png
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_localization_f1_matrix.png
analysis_outputs/parked_ego_fusion_viewpoint_eval/fusion_viewpoint_xy_error_matrix.png
```

Live visual sanity check for the combined View A+B checkpoint on View A. Start
CARLA first, then run without `--headless`:

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py \
  --run-duration-s 120 \
  --fusion-checkpoint experiments/parked_ego_tl16_viewAB_fusion_train_20260612/checkpoints/parked_viewAB_24000_768x432_lr1e-4_bs2/best.pt \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 4.0 \
  --ego-spawn-right-offset-m 7.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --ego-radar-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --fps 10 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --object-score-threshold 0.03 \
  --result-timeout 1.5 \
  --camera-source-port 53101 \
  --remote-port 53102 \
  --remote-source-port 53103 \
  --camera-result-port 53104 \
  --spatial-map-stream-id fusion_ab_viewA_visual \
  --no-spatial-map-stream \
  --run-group month2_ab_live_visual \
  --transport-label loopback_ab_viewA_visual
```

Live visual sanity check for the combined View A+B checkpoint on View B:

```bash
python3 carla_split_inference_udp_fusion_object_ego_client.py \
  --run-duration-s 120 \
  --fusion-checkpoint experiments/parked_ego_tl16_viewAB_fusion_train_20260612/checkpoints/parked_viewAB_24000_768x432_lr1e-4_bs2/best.pt \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 16.0 \
  --ego-spawn-right-offset-m 8.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --camera-resolution custom \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --ego-radar-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --fps 10 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --object-score-threshold 0.03 \
  --result-timeout 1.5 \
  --camera-source-port 53201 \
  --remote-port 53202 \
  --remote-source-port 53203 \
  --camera-result-port 53204 \
  --spatial-map-stream-id fusion_ab_viewB_visual \
  --no-spatial-map-stream \
  --run-group month2_ab_live_visual \
  --transport-label loopback_ab_viewB_visual
```

Selected-view collector smoke:

```bash
python3 carla_collect_parked_ego_fusion_training_data.py \
  --experiment-id parked_ego_tl16_spawn80_right7_fwd4_smoke_60_stride2 \
  --max-samples 60 \
  --sample-stride 2 \
  --fps 10 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --model-input-width 768 \
  --model-input-height 432 \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 4.0 \
  --ego-spawn-right-offset-m 7.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --npc-vehicles 20 \
  --npc-pedestrians 25 \
  --spawn-radius 95 \
  --seed 21 \
  --include-pedestrians
```

Validate selected-view smoke:

```bash
python3 scripts/validate_fusion_training_dataset.py \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_smoke_60_stride2 \
  --max-samples 30

python3 scripts/dry_run_fusion_training_targets.py \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_smoke_60_stride2 \
  --object-classes vehicle,person \
  --max-samples 30
```

First full collection commands:

```bash
# Low / clear-ish traffic
python3 carla_collect_parked_ego_fusion_training_data.py \
  --experiment-id parked_ego_tl16_spawn80_right7_fwd4_low_4000_stride2 \
  --max-samples 4000 \
  --sample-stride 2 \
  --fps 10 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --model-input-width 768 \
  --model-input-height 432 \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 4.0 \
  --ego-spawn-right-offset-m 7.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --npc-vehicles 5 \
  --npc-pedestrians 10 \
  --spawn-radius 95 \
  --seed 31 \
  --include-pedestrians

# Medium traffic
python3 carla_collect_parked_ego_fusion_training_data.py \
  --experiment-id parked_ego_tl16_spawn80_right7_fwd4_medium_4000_stride2 \
  --max-samples 4000 \
  --sample-stride 2 \
  --fps 10 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --model-input-width 768 \
  --model-input-height 432 \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 4.0 \
  --ego-spawn-right-offset-m 7.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --npc-vehicles 20 \
  --npc-pedestrians 25 \
  --spawn-radius 95 \
  --seed 41 \
  --include-pedestrians

# Crowded traffic
python3 carla_collect_parked_ego_fusion_training_data.py \
  --experiment-id parked_ego_tl16_spawn80_right7_fwd4_crowded_4000_stride2 \
  --max-samples 4000 \
  --sample-stride 2 \
  --fps 10 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-fov 120 \
  --model-input-width 768 \
  --model-input-height 432 \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 4.0 \
  --ego-spawn-right-offset-m 7.0 \
  --ego-spawn-yaw-offset-deg -28.414 \
  --ego-camera-x 1.8 \
  --ego-camera-y 0.0 \
  --ego-camera-z 1.55 \
  --ego-camera-pitch -4.0 \
  --ego-camera-yaw 0.0 \
  --radar-hfov 120 \
  --radar-vfov 30 \
  --radar-range 120 \
  --npc-vehicles 35 \
  --npc-pedestrians 45 \
  --spawn-radius 95 \
  --seed 51 \
  --include-pedestrians
```

Merge the three density folders into one training dataset with symlinks:

```bash
python3 scripts/merge_fusion_training_datasets.py \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2 \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_low_4000_stride2 \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_medium_4000_stride2 \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_crowded_4000_stride2 \
  --link-mode symlink
```

Validate merged dataset before training:

```bash
python3 scripts/validate_fusion_training_dataset.py \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2 \
  --max-samples 50

python3 scripts/dry_run_fusion_training_targets.py \
  fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2 \
  --object-classes vehicle,person \
  --max-samples 50
```

Interpretation:

- Higher `quality_score` is better.
- The score rewards broad road-spawn coverage, crosswalk coverage, angular
  spread, and left/center/right coverage inside the candidate camera FoV.
- The reported `spawn_index` is the actual CARLA map spawn index, so it can be
  reused directly with `carla_collect_parked_ego_fusion_training_data.py`.
- Start with one 110-120 degree front camera. A single 180-degree image is
  tempting for intersection awareness, but it is usually distorted for model
  training; if we need true 180-degree awareness later, use two/three cameras
  or multi-yaw captures.

Each scout Markdown file includes ready-to-run smoke and pilot collection
commands for the top candidate. After choosing a candidate, run the smoke
command first, validate it, then scale to pilot/full collection.

Useful local references:

```bash
python3 scripts/validate_fusion_training_dataset.py \
  fusion_training_data/<parked_ego_dataset>

python3 scripts/dry_run_fusion_training_targets.py \
  fusion_training_data/<parked_ego_dataset> \
  --object-classes vehicle,person
```

The dry-run is now class-aware for localization. Expected object target shapes:

- `center_heatmap`: `(2, 432, 768)` for vehicle/person centers.
- `regression`: `(10, 432, 768)` shared by both classes.
- `gt_objects`: `(64, 9)`.

The trainer also accepts parked `.npy` radar tensors from
`carla_collect_parked_ego_fusion_training_data.py`; it no longer requires only
the original `.npz` tensor format.

Launcher dry-run, after path normalization:

```bash
cd pole_lraspp_multimodal_fusion

./launch_unattended_fusion_training.sh \
  --dry-run \
  --config configs/fusion_smoke.yaml \
  --resume auto
```

Class-aware training smoke on the 60-sample TL16 parked-ego dataset:

```bash
mkdir -p experiments/parked_ego_classaware_train_smoke_20260611

ln -s \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_training_tl16_spawn80_60samp \
  experiments/parked_ego_classaware_train_smoke_20260611/dataset

cd pole_lraspp_multimodal_fusion

PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.train_fusion \
  --config configs/fusion_smoke.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_classaware_train_smoke_20260611 \
  --trial-json '{"name":"classaware_smoke_256x144_60samp","optimizer":"adamw","lr":0.0002,"weight_decay":0.0001,"augment_strength":"off","input_size":[256,144],"batch_size":2}' \
  --training-budget-hours 0.12
```

Smoke result: PASS. It wrote
`experiments/parked_ego_classaware_train_smoke_20260611/checkpoints/classaware_smoke_256x144_60samp/best.pt`
and checkpoint metadata confirms `object_channels=12` with
`object_class_names=['vehicle', 'person']`.

Plot training curves from any fusion trainer metrics CSV:

```bash
python3 scripts/plot_fusion_training_curves.py \
  experiments/parked_ego_classaware_train_smoke_20260611/metrics/classaware_smoke_256x144_60samp_metrics.csv \
  --prefix classaware_smoke_256x144_60samp
```

This writes PNG/PDF loss, mIoU, class-IoU, localization-loss, and auxiliary
signal curves into the experiment `figures/` directory.

### Selected TL16 Right-Lane 12k Fusion Training Run

Validated merged dataset:

```text
fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2
```

Validation summary:

- `12000` samples from low/medium/crowded density profiles.
- `114582` object rows: `58066` vehicle and `56516` person.
- Split counts: train `8620`, val `1666`, test `1714`.
- Class-aware target dry-run: PASS.
- Expected target shapes: heatmap `(2, 432, 768)`, regression `(10, 432, 768)`.
- `262` no-object samples are present, mostly from low traffic; keep them as
  useful background/negative frames.

Create the experiment folder and dataset link:

```bash
mkdir -p experiments/parked_ego_tl16_right7_fusion_train_20260612

ln -s \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_merged_12000_stride2 \
  experiments/parked_ego_tl16_right7_fusion_train_20260612/dataset
```

Run this one-line CUDA check before launching overnight training:

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Optional 60-sample GPU training smoke:

```bash
mkdir -p experiments/parked_ego_tl16_right7_fusion_train_smoke_20260612

ln -s \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/fusion_training_data/parked_ego_tl16_spawn80_right7_fwd4_smoke_60_stride2 \
  experiments/parked_ego_tl16_right7_fusion_train_smoke_20260612/dataset

PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.train_fusion \
  --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_smoke_20260612 \
  --trial-json '{"name":"smoke_parked_right7_classaware_768x432_bs2","optimizer":"adamw","lr":0.0001,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2}' \
  --training-budget-hours 0.001
```

Smoke result on 2026-06-12: PASS. It trained on `cuda` and wrote `best.pt`,
`last.pt`, `trial_summary.json`, and a metrics CSV.

Full training command:

```bash
PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.train_fusion \
  --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --trial-json '{"name":"parked_right7_lowmedcrowd_768x432_lr1e-4_bs2","optimizer":"adamw","lr":0.0001,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2}' \
  --training-budget-hours 9.0
```

Unattended version:

```bash
nohup env PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.train_fusion \
  --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --trial-json '{"name":"parked_right7_lowmedcrowd_768x432_lr1e-4_bs2","optimizer":"adamw","lr":0.0001,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2}' \
  --training-budget-hours 9.0 \
  > experiments/parked_ego_tl16_right7_fusion_train_20260612/train.log 2>&1 &
```

Monitor:

```bash
tail -f experiments/parked_ego_tl16_right7_fusion_train_20260612/train.log

nvidia-smi
```

Plot curves after training:

```bash
python3 scripts/plot_fusion_training_curves.py \
  experiments/parked_ego_tl16_right7_fusion_train_20260612/metrics/parked_right7_lowmedcrowd_768x432_lr1e-4_bs2_metrics.csv \
  --prefix parked_right7_lowmedcrowd_768x432_lr1e-4_bs2
```

Evaluate the best checkpoint on the held-out test split:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
  --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612/checkpoints/parked_right7_lowmedcrowd_768x432_lr1e-4_bs2/best.pt \
  --split test
```

Optional sanity check on the validation split:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion \
  --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --checkpoint /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612/checkpoints/parked_right7_lowmedcrowd_768x432_lr1e-4_bs2/best.pt \
  --split val
```

Expected evaluation outputs:

```text
experiments/parked_ego_tl16_right7_fusion_train_20260612/metrics/test_fusion_evaluation_metrics.json
experiments/parked_ego_tl16_right7_fusion_train_20260612/metrics/test_learned_object_metrics.csv
experiments/parked_ego_tl16_right7_fusion_train_20260612/figures/test_fusion_confusion_matrix.png
experiments/parked_ego_tl16_right7_fusion_train_20260612/figures/test_rgb_baseline_confusion_matrix.png
```

Analyze localization failures by traffic density, class, distance, object size,
and radar support:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
python3 scripts/analyze_fusion_localization_failures.py \
  --experiment-dir experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --output-dir analysis_outputs/parked_ego_fusion_v1
```

Expected analysis outputs:

```text
analysis_outputs/parked_ego_fusion_v1/fusion_localization_failure_summary.json
analysis_outputs/parked_ego_fusion_v1/fusion_localization_gt_enriched.csv
analysis_outputs/parked_ego_fusion_v1/localization_f1_by_density.png
analysis_outputs/parked_ego_fusion_v1/localization_recall_by_distance.png
analysis_outputs/parked_ego_fusion_v1/localization_recall_by_bbox_area.png
analysis_outputs/parked_ego_fusion_v1/localization_recall_by_radar_support.png
```

Optional second-stage fine-tune, only if the best validation score is still at
the final epoch and localization/object-head metrics need improvement. The
trainer supports `resume_lr` in the trial JSON, which overrides the optimizer LR
after loading `last.pt`; it also supports a trial-level `epochs` override:

```bash
PYTHONPATH=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/pole_lraspp_multimodal_fusion:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun \
python3 -m pole_lraspp_multimodal_fusion.train_fusion \
  --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --experiment-dir /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/experiments/parked_ego_tl16_right7_fusion_train_20260612 \
  --trial-json '{"name":"parked_right7_lowmedcrowd_768x432_lr1e-4_bs2","optimizer":"adamw","lr":0.0001,"resume_lr":0.00005,"weight_decay":0.0002,"augment_strength":"strong","input_size":[768,432],"batch_size":2,"epochs":80}' \
  --training-budget-hours 4.0
```

Evaluation override examples for operating-point diagnosis:

```bash
# Stricter/lower object score threshold.
python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion ... \
  --split val \
  --object-score-threshold 0.10

# Looser match distance, useful only for diagnosis; do not report as the strict
# final metric unless the threshold is clearly stated.
python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion ... \
  --split val \
  --object-score-threshold 0.03 \
  --match-distance-m 5.0
```

## 4. Remote Sync, When Needed

When moving the new coordinator to the remote machine:

```bash
rsync -avh \
  scenesense_single_ego_task_coordinator.py \
  scenesense_dual_task_launcher.py \
  carla_split_inference_udp_rgb_ego_transfer_client.py \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/
```

Pull single-ego controller and RGB transfer metrics back:

```bash
rsync -avh \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/single_ego_task_coordinator/ \
  metrics_logs/single_ego_task_coordinator/

rsync -avh \
  shr_aisvcs@L10319.idcc.lab:/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/rgb_ego_transfer/ \
  metrics_logs/rgb_ego_transfer/
```
