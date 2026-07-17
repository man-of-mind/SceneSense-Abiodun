# SceneSense RL Schema Draft

Originally drafted in Month 1; reconciled with the July action, OAI, and
staleness evidence on 2026-07-16. The first controller must still be evaluated
offline against logged traces before any online policy touches CARLA or OAI.

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
| Object/ego dynamics | Object speed bin, ego speed, acceleration, turning, road state | CARLA trajectories, vehicle telemetry, map waypoint geometry. |
| Confidence/uncertainty | Mean/max detection confidence, object-head support confidence, segmentation entropy/probability margin | Model output tensors and result payloads. |
| Payload pressure | Payload bytes, uncompressed bytes, chunk count, compression profile, send/skip history | Application metrics CSV. |
| Latency pressure | Front time, back time, RTT, timeout count, stale-result age | Application metrics CSV. |
| Network health | UE tunnel bitrate, packet counters, ping RTT/loss, gNB/UE MCS/RB/TBS/HARQ/BLER where available | Network sampler, T-tracer, gNB stdout parser. |
| Map freshness | Capture-to-map latency `Y`, update FPS, held age, `Y + 1/FPS`, track freshness | Application timestamps, spatial-map snapshot, staleness analysis. |

Minimum Month 1 offline state vector:

```text
[
  route_id,
  scenario_id,
  sensor_placement_id,
  compression_profile_id,
  object_count,
  object_speed_mps,
  ego_speed_mps,
  road_state,
  foreground_fraction,
  vulnerable_object_present,
  mean_confidence,
  payload_bytes,
  payload_chunks,
  round_trip_ms,
  timeout_or_missing_result,
  update_fps,
  map_age_ms,
  ue_tx_mbps,
  ue_rx_mbps,
  grant_mcs_ul,
  grant_rb_ul,
]
```

## Action Candidates

| Action group | Initial discrete values | Notes |
| --- | --- | --- |
| Resident model / AE | no-AE, AE-128, AE-64, AE-32 | Switch among validated resident checkpoints; do not reload per frame. |
| Quantization | `per_channel_uint8`, `per_channel_uint6`, `per_channel_uint4` | Current fusion-route action levels. Route-mask unsupported values. |
| Entropy coder | fixed validated coder initially; `zlib`/`zstd` ablation | Keep lossless coder choice out of the first learned action unless it changes cost materially. |
| ROI drop fraction | `0.0`, `0.3`, `0.5` | Objectness-ranked quantile drop; aggressive values require segmentation guardrails. |
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
  - staleness_weight * normalized_map_age_error
  - guardrail_penalty
```

Suggested task utility terms:

- OD: object recall or AP proxy, with extra weight for pedestrians/cyclists.
- SEG: foreground IoU or mIoU proxy, with extra weight for person/vehicle IoU.
- Fusion object head: object recall, XY localization error, yaw/dimension error.
- Fusion segmentation head: foreground IoU, vehicle/person IoU.
- Spatial-map freshness: speed-conditioned error at age `Y + 1/FPS`, stale
  update rate, and vulnerable-object warning timeliness.

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
| Dynamics/freshness floor | Tighten the allowed `Y + 1/FPS` budget as object speed rises; reject actions predicted to exceed the configured localization-error envelope. |
| Route support | Clamp unsupported AE/ROI/quantization choices to that route's safest supported profile. |

There is no single universally safe fallback: no-AE u8 maximizes per-frame
quality but overloaded the current OAI path, while AE-128 u4 restored
availability with a small segmentation cost. The guardrail should choose
between a task-quality fallback and a transport-safe fallback:

```text
task_quality_fallback = no-AE + uint8 + ROI0
transport_safe_fallback = AE-128 + uint4 + ROI0
frame_skip = 0 when a vulnerable object or stale-map hazard is active
```

## Offline Evaluation Plan

1. Join application metrics, network metrics, scenario metadata, and task
   metrics by `run_group`, `route_id`, and frame/window.
2. Compute static-profile baselines from the trace matrix.
3. Score candidate actions offline using the reward function.
4. Apply guardrails and record rejection/fallback reasons.
5. Compare learned or heuristic action choices against static baselines:
   payload, latency, timeout rate, and task utility.

## Update (2026-07-16) — measured action and requirement evidence
- **New state feature (optional, cheap, model-free): Sobel edge-density** as a *scene-complexity* proxy,
  computable on the raw image on the UE **without running the model** — lets the policy modulate how
  aggressively to compress ("empty road → compress hard; cluttered intersection → conservative").
  Complements object-count / foreground-fraction. NB: this is a STATE input, *not* a drop signal.
- **ROI action = objectness-guided *quantile* importance-drop** (drop the lowest-objectness fraction `q`),
  not an absolute objectness threshold (the fusion object-head heatmap is focal-biased, so absolute
  thresholds don't transfer). `q` in [0, ~0.8]. Importance = the object-head objectness map (task-aware).
- **Drop-aware model:** M-prime implements objectness-guided feature dropout and
  provides the robust no-AE baseline.
- **Integrated AE models:** AE-128/64/32 recovered the localization/object-head
  behavior that collapsed in earlier standalone AE experiments. They are valid
  resident actions, not merely unsafe diagnostics.
- **Action-cost model measured:** `rl_agent/PERMODEL_KNOB_MATRIX.md` contains 42
  model × quantization × ROI profiles. Quantization is inexpensive; aggressive
  ROI is primarily a segmentation risk.
- **OAI action evidence:** no-AE u8 (`1141 KB`, `209 ms`, `75%` delivery) versus
  AE-128 u4 (`142 KB`, `77 ms`, `99%` delivery) demonstrates that payload affects
  both availability and map freshness.
- **Network action evidence:** TDD/5QI barely change the single-UE,
  no-impairment RFsim result. Do not make them primary fast actions until
  contention or impairment shows a measurable effect.
- **Dynamics requirement:** live measurements show a roughly `1.1 m` model
  floor, sharply increasing latency error for fast vehicles, and held-map age
  up to `1/FPS`. Object speed and `Y + 1/FPS` are required state/reward inputs.

## Current Implementation Boundary

The schema, action measurements, OAI measurements, and staleness requirements
exist. The trace join, executable action catalog, reward scorer, simple-policy
replay, LinUCB/DQN policy, and controller-level guardrail are not implemented.
No online action execution should be enabled until those offline checks pass.
