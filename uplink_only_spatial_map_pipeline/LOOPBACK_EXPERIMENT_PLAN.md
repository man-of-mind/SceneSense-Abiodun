# Track 1 loopback experiment plan — uplink-only spatial-map pipeline

Created: 2026-07-28

## Question

Before bringing OAI back into the loop, test whether the realistic uplink-only
pipeline can keep up on loopback:

```text
CARLA/front -> split features -> edge tail model -> spatial-map update
```

The goal is to separate application/compute/map bottlenecks from radio
bottlenecks.

## Core freshness/staleness metric

Track 1 should directly measure the Stage-1 freshness term from
`../rl_agent/AGENT_CONSTRAINTS.md` and
`../rl_agent/REQUIREMENTS_AND_RL_DESIGN.md`:

```text
Y_up = t_map_update_done - t_capture
```

where:

- `t_capture`: when the CARLA frame/sensor sample was captured at the vehicle;
- `t_map_update_done`: when the edge spatial-map application has ingested the
  detections and completed the map update/reasoning for that frame.

This is the age of the information when it becomes usable in the edge map. The
Stage-1 staleness constraint is:

```text
v * (Y_up + 1/FPS) <= sqrt(epsilon^2 - 1.1^2)
```

The extra `1/FPS` term is map hold/freshness between updates: even if the last
map update was fresh, the map can be almost one frame period old just before
the next update arrives.

For later cooperative actuation, extend the lag to:

```text
L_total = Y_up + 1/FPS + Y_down + Y_map_share
```

but Track 1 loopback should first measure `Y_up` cleanly.

## Important timing distinction

For a streaming/pipelined system, **per-frame latency does not need to be less
than the frame period**. A frame can take 120 ms end-to-end while the system
still sustains 10 FPS, as long as the bottleneck stage can complete one frame
every 100 ms.

The key condition is:

```text
bottleneck service time <= frame period
```

or, equivalently:

```text
service rate >= arrival rate
```

Examples:

| Target FPS | Frame period | What must be true |
|---:|---:|---|
| 10 FPS | 100 ms | bottleneck stage should process >= 10 frames/s |
| 20 FPS | 50 ms | bottleneck stage should process >= 20 frames/s |
| 30 FPS | 33 ms | bottleneck stage should process >= 30 frames/s |

Transport latency contributes to **data age/staleness**, but it is not always
the throughput bottleneck if frames are pipelined. Transport becomes the
bottleneck when the link cannot sustain:

```text
payload_bytes_per_frame * FPS
```

or when buffering/serialization causes queue growth.

## Why the concern is valid

If loopback delivery to the edge takes about `45 ms`, tail inference takes
`~7-15 ms`, and future spatial-map reasoning takes `~40 ms`, the important
questions are:

1. Does the edge receive frames at the requested cadence?
2. Does tail inference plus spatial-map update finish before the next frame
   arrives?
3. If not, where does queueing occur?
   - front/CARLA capture loop;
   - UDP/socket send path;
   - edge receive queue;
   - tail model queue;
   - spatial-map update queue.

At 10 FPS, a `tail + map ~= 45-60 ms` service time is likely safe.
At 20 FPS, the period is only `50 ms`, so the same service time becomes
borderline. At 30 FPS, it likely cannot keep up unless map compute is parallel,
batched, dropped, or reduced.

## Metrics to record

For each frame:

| Metric | Why |
|---|---|
| `t_capture_perf` / `t_capture_wall` | source timestamp / freshness start |
| `front_send_ts` | when feature payload is handed to transport |
| `edge_recv_ts` | uplink one-way arrival at edge |
| `tail_start_ts` | edge queue wait before model tail |
| `tail_done_ts` | tail inference service time |
| `map_publish_ts` | detection payload handed to map app |
| `map_ingest_ts` | spatial-map server receives packet |
| `map_update_done_ts` | map app finishes update/reasoning |
| `frame_id` gaps | detect drops or intentional frame skipping |
| queue depths | locate bottleneck queue |

Derived metrics:

| Derived metric | Meaning |
|---|---|
| front period | actual sensor/send cadence |
| uplink latency | `edge_recv_ts - front_send_ts` |
| edge queue wait | `tail_start_ts - edge_recv_ts` |
| tail service | `tail_done_ts - tail_start_ts` |
| map queue wait | `map_update_start_ts - map_ingest_ts` |
| map service | `map_update_done_ts - map_update_start_ts` |
| capture-to-map age `Y_up` | `map_update_done_ts - t_capture` |
| sustained map update FPS | actual useful map update rate |
| backlog growth | whether queue delay increases over time |

## Current instrumentation in this folder

The copied Track-1 scripts now produce three complementary logs:

Use `carla_fusion_staleness_scenario_uplink_only.py` for reportable runs. It
is copied from the corrected `../staleness/carla_fusion_staleness_scenario.py`
deployment path, so it keeps the same route/NPC/radar controls as the closed
loop experiments.

| File | Written by | Best use |
|---|---|---|
| front metrics CSV under `--metrics-run-dir` | CARLA/front process | actual frame cadence, front compute, feature payload bytes/chunks, scene/object GT context |
| `--edge-metrics-csv` | edge/back worker | feature receive time, tail inference time, edge direct-publish time, edge-side queue/drops |
| `--ingest-metrics-csv` | spatial-map server | packet ingest time, map update service time, final `capture_to_map_update_done_ms` |

For the Track-1 loopback split-inference/map result, treat
`map_ingest_metrics.csv:backbone_input_to_map_update_done_ms` as the reportable
core model-path latency:

```text
Y_model = t_map_update_done - t_backbone_input
```

where `t_backbone_input` is the moment the fused RGB+radar tensor is ready and
enters the front backbone (`model.encode(...)`).

Keep `capture_to_map_update_done_ms` as an optional full sensor-to-map
freshness view:

```text
Y_up = t_map_update_done - t_capture
```

The distinction matters because `Y_up` includes CARLA sensor/radar wait and
front-side tensor preparation, while `Y_model` isolates the model/uplink/edge
tail/map path that we currently need for the split-inference architecture.

The front CSV remains the source of exact compressed feature payload size and
front-side stage timing (`model_preprocess_ms`, `front_backbone_ms`,
`feature_serialize_ms`, `send_call_ms`), because the UDP chunker measures
payload bytes/chunks on the sender side.

In uplink-only mode, the edge worker now uses a bounded receive queue between
UDP message assembly and tail inference. If that queue grows, the tail/map path
is slower than the arriving feature stream. If `edge_receive_queue_dropped`
increases, old frames were intentionally dropped so the spatial map stays fresh.

At minimum, every frame should carry a stable timing block:

```json
{
  "frame_id": 123,
  "carla_timestamp": 456.7,
  "timing": {
    "t_capture_perf": 1000.000,
    "t_front_send_perf": 1000.030,
    "t_edge_recv_perf": 1000.037,
    "t_tail_start_perf": 1000.038,
    "t_tail_done_perf": 1000.047,
    "t_map_publish_perf": 1000.048,
    "t_map_ingest_perf": 1000.049,
    "t_map_update_done_perf": 1000.055
  }
}
```

In one-process loopback, `perf_counter()` is directly comparable across stages.
For multi-process/container OAI, same-host `perf_counter()` may not be
comparable across machines/containers, so we should either:

- use wall-clock timestamps with clock sync caveats; or
- compute per-side stage durations locally and pair them by `frame_id`;
- optionally add ping/offset calibration later.

Loopback first avoids that clock-sync problem.

## Loopback test matrix

Current result note:
`TRACK1_IDEAL_LOOPBACK_RESULTS.md`

Status as of 2026-07-29:

- corrected-source live no-return loopback matrix completed;
- exact-FPS model-boundary/map offered-load replay completed;
- fine-grained live frontend preparation profiling completed.

The key finding is that the core split model/uplink/tail/map path is fast on
ideal loopback, but the live CARLA frontend is not currently offering true
10/20 FPS model-ready frames because radar tensor construction dominates the
pre-model path.

### Transport precondition

Before treating no-AE uplink-only loopback numbers as application results,
verify that the actual granted UDP receive buffer is large enough for at least
one compressed feature burst.

The corrected-source 20-frame smoke on 2026-07-29 found:

```text
requested SO_RCVBUF = 8 MB
actual granted SO_RCVBUF = 425,984 bytes
no-AE/zstd feature payload ~= 1.06 MB per frame
```

With that cap, the edge UDP assembler drops incomplete multipart messages
before they ever enter the edge receive queue. ACK/backpressure helps only
slightly because it cannot recover chunks already dropped during the burst.

After restoring `net.core.rmem_max/wmem_max=8388608`, a fresh socket was granted
`SO_RCVBUF=16,777,216` bytes and the corrected-source no-AE/zstd 20-frame smoke
delivered 20/20 frames with `udp_partial_messages_dropped=0`.

So reportable loopback tests should keep this ideal-buffer precondition active.
If the sysctl is unavailable after reboot or on another host, use one of:

- restore the ideal loopback UDP receive-buffer setting used in earlier
  latency work, then rerun no-AE;
- switch to a payload that fits under the current buffer cap, e.g. ROI/int4 or
  AE-128 + ROI 0.5 + uint6;
- replace the raw UDP multipart transport with a reliable/framed transport or
  explicit chunk-level retransmission/backpressure.

Start simple:

| Test | FPS target | Payload/model | Map compute |
|---|---:|---|---|
| A | 10 | no-AE baseline | no-op / current map update |
| B | 20 | no-AE baseline | no-op / current map update |
| C | 10 | no-AE baseline | artificial `40 ms` map delay |
| D | 20 | no-AE baseline | artificial `40 ms` map delay |

If A/B are clean but C/D queue, the future spatial-map algorithm budget is the
limiting factor. If A/B already queue, the uplink-only feature/tail path has a
bottleneck even before advanced map reasoning.

## What to look for

Good behavior:

- frame send cadence near target FPS;
- stable edge queue wait;
- stable map queue wait;
- no increasing capture-to-map age over time;
- map update FPS tracks target FPS or intentional frame-drop policy.

Bad behavior:

- queue wait grows monotonically;
- map update FPS falls below send FPS without controlled dropping;
- capture-to-map age drifts upward;
- old frames are processed after newer frames are available.

## Design choice if overloaded

If the map stage cannot keep up, it should prefer freshness over completeness:

- process latest frame per stream;
- drop stale queued frames;
- keep a bounded queue;
- report dropped/stale frames as a metric;
- later, use the RL/compression policy to reduce payload/FPS when freshness
  budget is threatened.
