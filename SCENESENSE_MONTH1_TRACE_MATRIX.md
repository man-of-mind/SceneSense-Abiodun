# SceneSense Month 1 Trace Matrix

This file pins down the Month 1 trace definitions. The goal is not to finish
every sweep; it is to make the baseline route, compression, network, sensor,
and ground-truth choices explicit before collecting comparable runs.

## Scenario Battery

| Scenario role | Harness scenario | Purpose |
| --- | --- | --- |
| Simple clear scene | `clear_low_density` | Low-density baseline with clear line of sight. |
| Crowded scene | `crowded_intersection` | Higher overlap and more background actors. |
| Canonical occlusion | `curbside_parked_vehicle_pedestrian_occlusion` | Hidden pedestrian dart-out with ego/helper evidence pack. |

Canonical occlusion validation:

```bash
python3 scenesense_scenarios/validate_evidence_pack.py \
  remote_files/20260601_183145_curbside_parked_vehicle_pedestrian_occlusion_seed7
```

Current copied validation result: PASS. The target used `walker_control`, reached
0.698 crossing progress, produced a 2.62 m danger event, and wrote 80 ego plus
80 helper RGB evidence frames.

Collision-tuned validation:

```bash
python3 scenesense_scenarios/validate_evidence_pack.py \
  metrics_logs/scenesense_scenarios/20260602_101912_curbside_parked_vehicle_pedestrian_occlusion_seed7 \
  --require-collision
```

Current collision validation result: PASS. The target used
`scripted_transform`, recorded 19 target collisions, reached 0.495 crossing
progress, and wrote 130 ego plus 131 helper RGB evidence frames. This is the
collision/evidence run; the transform-forced pedestrian can visually slide, so
do not treat it as the final animation-polish demo.

Animated route-lead validation:

```bash
python3 scenesense_scenarios/validate_evidence_pack.py \
  metrics_logs/scenesense_scenarios/20260602_104540_curbside_parked_vehicle_pedestrian_occlusion_seed7 \
  --require-collision
```

Animated route-lead validation result: PASS. The target used
`walker_control`, recorded 11 target collisions, reached 0.488 crossing
progress, and started when the ego was about 26 m from the route trigger.
This proves the ego-route-location trigger can produce a non-sliding
pedestrian collision.

Final animated curbside evidence demo:

```bash
python3 scenesense_scenarios/validate_evidence_pack.py \
  metrics_logs/scenesense_scenarios/20260602_125157_curbside_parked_vehicle_pedestrian_occlusion_seed7 \
  --require-collision
```

Final demo validation result: PASS. The run used `run_curbside_far_sidewalk_demo.sh`
with `TARGET_START_LAT=5.2`, `TARGET_FORWARD=-6.5`, `TARGET_SPEED=21`,
`ROUTE_LEAD=24`, `EGO_THROTTLE=0.45`, and forced
`vehicle.sprinter.mercedes`. It recorded 9 target collisions, 0.569 target
crossing progress, 4.35 s from target start to first collision, and 88 ego plus
89 helper RGB evidence frames. Treat this as the Month 1 animated curbside
evidence demo.

## Split Routes

| Route ID | Task metric family | Primary script | Transport baseline | Output interpretation |
| --- | --- | --- | --- | --- |
| `camera_od` | Camera-only OD | `carla_split_inference_udp_data_collect.py` | Local loopback first, OAI with `carla_split_inference_udp_oai.py` | Boxes/detections, AP or first-pass object recall. |
| `camera_seg` | Camera-only SEG | `carla_split_inference_udp_segmentation_trained_lraspp_demo.py` | Local loopback first, OAI with `carla_split_inference_udp_segmentation_oai.py` | Semantic mask, mIoU/foreground IoU/class IoU. |
| `fusion_as_od` | RGB+radar fusion object head | `carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py` | Local `--role loopback`, single-UE OAI, then multi-UE OAI | Object center, XYZ, yaw, size, parked/radar support fields evaluated as OD/localization. |
| `fusion_as_seg` | RGB+radar fusion segmentation head | `carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py` | Same runs as `fusion_as_od` | Fusion mask output evaluated as SEG. |

Fusion produces both object/localization and segmentation outputs. Keep
`fusion_as_od` and `fusion_as_seg` as separate analysis labels even when they
come from the same run folder.

## Sensor Placement

| Placement ID | Mount | Applies to | Notes |
| --- | --- | --- | --- |
| `pole_tl14_view_1` | Traffic-light pole candidate | `fusion_as_od`, `fusion_as_seg` | Runbook default: traffic light 14, camera x=9, y=2, pitch=-30, yaw offset=50, FoV=100. |
| `pole_tl14_view_2` | Traffic-light pole candidate | `fusion_as_od`, `fusion_as_seg` | Second stream: camera x=11, y=2, pitch=-30, yaw offset=120, FoV=100. |
| `ego_front` | Ego vehicle | Scenario harness evidence, future ego OD/SEG/fusion traces | RGB camera at x=1.8, z=1.6 and radar at x=2.0, z=1.0 in the harness. |
| `helper_front` | Opposite-lane helper vehicle | Occlusion evidence only | Observer camera, not a controller or participant in the ego collision. |

Every official run should record one of these placement IDs, plus the exact
resolved transform already written by the script manifest.

## Static Compression Settings

| Profile ID | Quantization | Entropy coder | ROI/saliency | AE mode | Purpose |
| --- | --- | --- | --- | --- | --- |
| `baseline_tensor_zlib` | `per_tensor_uint8` | `zlib` | off / 0.0 | off | Original camera OD/SEG-style baseline. |
| `baseline_channel_zlib` | `per_channel_uint8` | `zlib` | off / 0.0 | off | Default RGB+radar fusion baseline and fairer dynamic-range baseline. |
| `channel_uint4_zlib` | `per_channel_uint4` | `zlib` | off / 0.0 | off | Lower-byte static point where supported. |
| `channel_zstd` | `per_channel_uint8` | `zstd --zstd-level 3` | off / 0.0 | off | Stronger general-purpose coder when `zstandard` is installed. |
| `roi_gate_probe` | `per_channel_uint8` | `zlib` | OD objectness or SEG saliency enabled | off | First ROI/saliency threshold trace; exact threshold is a sweep value. |
| `ae_random_probe` | `per_channel_uint8` | `zlib` | off / 0.0 | `random_projection` | AE-channel placeholder before trained AE checkpoints exist. |

Supported knobs by route:

| Route ID | Quantization | Entropy | ROI/saliency | AE |
| --- | --- | --- | --- | --- |
| `camera_od` | yes | yes | RPN objectness gate | yes |
| `camera_seg` | yes | yes | saliency drop fraction | yes |
| `fusion_as_od` | yes | yes | not first Month 1 priority | not first Month 1 priority |
| `fusion_as_seg` | yes | yes | not first Month 1 priority | not first Month 1 priority |

## Network Trace Placeholders

| Network profile | Month | Meaning | Required logging |
| --- | --- | --- | --- |
| `local_no_stress` | Month 1 | Loopback or local LAN baseline, no intentional impairment. | Application metrics CSV and manifest. |
| `oai_5g_no_impairment` | Month 1 | OAI transport baseline, no intentional low-SNR/bandwidth/loss stress. | Application metrics, UE tunnel sampler, T-tracer/gNB stdout where available. |
| `logged_latency_loss_replay` | Month 1/2 | Save observed RTT/loss/timeout traces for later controller replay. | Per-frame RTT/timeout plus network sampler CSV aligned by `run_group`. |
| `controlled_delay_loss_bandwidth` | Month 2/3 | TC/netem or equivalent stress profiles. | Reserved; do not mix with Month 1 no-impairment baselines. |

## Ground Truth And Evaluation Mode

Month 1 evaluation is offline from saved traces. Live overlays remain useful
for smoke tests, but AP/mIoU/object-localization decisions should come from
run folders and scripts.

| Task | Ground-truth source | First metric |
| --- | --- | --- |
| Camera-only OD | CARLA actors/transforms/bounding boxes projected into the camera view, plus saved actor tables where available. | First-pass object recall, then AP/mAP. |
| Camera-only SEG | CARLA semantic-segmentation camera aligned with the RGB stream. | Foreground IoU, then 3-class or class IoU/mIoU. |
| Fusion object head | CARLA actors/transforms/bounding boxes and sensor pose/calibration. | Object recall, XY/XYZ localization error, yaw/dimension error. |
| Fusion segmentation head | CARLA semantic-segmentation camera aligned with fusion RGB/radar samples. | Foreground IoU, vehicle/person IoU, mIoU. |
| Radar fusion input | Saved radar tensor plus raw/projected radar point records. | Tensor sanity checks and support-vs-actor association. |

## Month 1 Gates

- Each official run has a `run_group`, route ID, compression profile, network
  profile, scenario ID, and sensor placement ID.
- OD and SEG quality metrics remain separate even for the same fusion run.
- `oai_5g_no_impairment` remains the only Month 1 OAI network profile.
- Occlusion evidence packs pass `validate_evidence_pack.py` before being used
  as a canonical hidden-hazard example.
