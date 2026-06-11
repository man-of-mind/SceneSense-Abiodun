# SceneSense Month 2 Reproducible Commands

Last updated: 2026-06-10

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

5. Moving RGB-only OD, moving ego source-domain reference:

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

6. Moving RGB-only OD, parked ego in the same moving-demo spawn area:

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

### Track D: New Training Decision Gate

Do not train new RGB+radar models until the RGB-only transferability table is
clear.

- If moving RGB-only SEG transfers well to parked ego, use it as the first
  SEG baseline and treat fusion retraining as an accuracy/payload improvement
  question.
- If pole-trained RGB-only SEG collapses on parked ego while moving SEG holds
  up, the failure is viewpoint/domain-specific and supports training a parked
  ego fusion SEG model.
- If moving RGB-only OD is weak from parked ego, then a parked-ego OD model or
  fusion OD model becomes justified.
- A true RGB+radar OD model is a new model family, not the current fusion
  SEG/localization checkpoint renamed. It needs its own labels, architecture,
  and evaluation target.

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
