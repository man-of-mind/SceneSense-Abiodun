# SceneSense RL Schema Draft

This is the Month 1 controller schema. It is intentionally conservative: the
first controller should be evaluated offline against logged traces before any
online policy touches CARLA or OAI runs.

## Control Objective

Choose a split-inference operating point that reduces bytes and latency while
preserving task utility and safety-critical recall.

The controller is not allowed to trade away vulnerable-object recall or
foreground segmentation quality silently. Guardrails clamp or reject unsafe
actions before the action reaches the runtime.

## State Candidates

| State group | Feature examples | Source |
| --- | --- | --- |
| Scene density | Actor/object count, detections per frame, crowded/clear scenario label | CARLA actor traces, model outputs, scenario manifest. |
| Foreground fraction | Predicted foreground mask fraction, semantic GT foreground fraction when offline | SEG/fusion masks, semantic GT camera. |
| Vulnerable-object presence | Pedestrian/cyclist/hidden-hazard flags, target danger event, object class counts | CARLA actor roles, OD/fusion outputs, evidence traces. |
| Confidence/uncertainty | Mean/max detection confidence, object-head support confidence, segmentation entropy/probability margin | Model output tensors and result payloads. |
| Payload pressure | Payload bytes, uncompressed bytes, chunk count, compression profile, send/skip history | Application metrics CSV. |
| Latency pressure | Front time, back time, RTT, timeout count, stale-result age | Application metrics CSV. |
| Network health | UE tunnel bitrate, packet counters, ping RTT/loss, gNB/UE MCS/RB/TBS/HARQ/BLER where available | Network sampler, T-tracer, gNB stdout parser. |

Minimum Month 1 offline state vector:

```text
[
  route_id,
  scenario_id,
  sensor_placement_id,
  compression_profile_id,
  object_count,
  foreground_fraction,
  vulnerable_object_present,
  mean_confidence,
  payload_bytes,
  payload_chunks,
  round_trip_ms,
  timeout_or_missing_result,
  ue_tx_mbps,
  ue_rx_mbps,
  grant_mcs_ul,
  grant_rb_ul,
]
```

## Action Candidates

| Action group | Initial discrete values | Notes |
| --- | --- | --- |
| Quantization | `per_tensor_uint8`, `per_channel_uint8`, `per_channel_uint4` | Only expose values supported by the active route. |
| Entropy coder | `zlib`, `zstd`, `none` | `zstd` requires `zstandard`; `none` is for diagnosis. |
| ROI/saliency threshold | off, low, medium, high | OD uses RPN objectness; SEG uses saliency drop fraction. |
| AE channel setting | off, random/checkpoint bottleneck profiles | Only where the route supports AE. |
| Frame send/skip | send every frame, skip 1, skip 2 | Guardrail must block skips during vulnerable-object events. |
| Redundancy | single send, duplicate critical result/feature packet | Reserved for later OAI stress phases. |

Action masking is required: unsupported route/action combinations must not be
sampled or scored.

## Reward Sketch

Offline reward for a frame or short window:

```text
reward =
  task_utility
  - payload_weight * normalized_payload_bytes
  - latency_weight * normalized_round_trip_ms
  - timeout_weight * timeout_or_missing_result
  - loss_weight * observed_loss_or_retransmission_proxy
  - guardrail_penalty
```

Suggested task utility terms:

- OD: object recall or AP proxy, with extra weight for pedestrians/cyclists.
- SEG: foreground IoU or mIoU proxy, with extra weight for person/vehicle IoU.
- Fusion object head: object recall, XY localization error, yaw/dimension error.
- Fusion segmentation head: foreground IoU, vehicle/person IoU.

For Month 1, keep the weights fixed in configuration. Do not learn the reward
weights yet.

## Guardrail Sketch

The guardrail runs before action execution and after offline scoring:

| Guardrail | Rule |
| --- | --- |
| Task floor | Reject actions whose AP, object recall, mIoU, foreground IoU, or class IoU falls below the configured route floor. |
| Vulnerable-object floor | Reject aggressive compression, frame skip, or ROI drop when pedestrian/cyclist/hidden-hazard presence is true. |
| Confidence floor | Fall back to safer settings when model confidence drops or uncertainty rises. |
| Network timeout floor | If timeout/missing-result rate rises, prefer smaller payload actions before frame skipping. |
| Route support | Clamp unsupported AE/ROI/quantization choices to that route's safest supported profile. |

Initial safe fallback profile:

```text
quantization=per_channel_uint8
entropy_coder=zlib
roi_or_saliency=off
ae_mode=off
frame_skip=0
redundancy=off
```

## Offline Evaluation Plan

1. Join application metrics, network metrics, scenario metadata, and task
   metrics by `run_group`, `route_id`, and frame/window.
2. Compute static-profile baselines from the trace matrix.
3. Score candidate actions offline using the reward function.
4. Apply guardrails and record rejection/fallback reasons.
5. Compare learned or heuristic action choices against static baselines:
   payload, latency, timeout rate, and task utility.

## Month 1 Boundary

Month 1 ends with this schema and logged trace compatibility. Training a policy,
adding controlled network stress, and comparing learned actions against static
baselines belong to Month 2/3.
