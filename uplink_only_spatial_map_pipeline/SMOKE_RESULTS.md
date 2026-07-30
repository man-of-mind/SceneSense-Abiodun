# Track 1 uplink-only loopback smoke results

Created: 2026-07-28

Status: **early Track-1 transport validation. Do not use the old prototype
numbers as reportable Track-1 results.**

The first smoke runs used
`carla_split_inference_udp_fusion_object_uplink_only_spatial.py`, the older
spatial-stream prototype copy. The reportable Track-1 implementation is now
`carla_fusion_staleness_scenario_uplink_only.py`, copied from the corrected
`../staleness/carla_fusion_staleness_scenario.py` deployment path.

The prototype result is still useful as a warning: no-wait, multi-chunk UDP
feature transport can lose complete feature messages if chunks go missing. But
the corrected-source Track-1 script must be used before drawing experiment
conclusions.

## Corrected-source sanity check

Date: 2026-07-29

Script:
`carla_fusion_staleness_scenario_uplink_only.py`

Configuration:

- loopback transport, Town10HD_Opt
- corrected moving-ego route from the staleness/downlink-latency runs:
  spawn indices `80,85,91,94,99,80`
- no-AE baseline checkpoint
- 200k radar PPS
- zstd entropy coding, `per_channel_uint8`, `roi_threshold=0.0`
- 28 requested vehicles, 35 pedestrians
- 10 FPS, 20 front frames
- spatial-map server with `--map-update-delay-ms 0`

| Run | Edge result behavior | Front frames | Edge/map frames | UDP partial messages dropped | Edge queue drops | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `corrected_loopback_none_20f` | no downlink result | 20 | 1 | 19 | 0 | open-loop feature bursts are not completing UDP reassembly |
| `corrected_loopback_ackwait_20f` | tiny ACK + front waits for ACK | 20 | 6 | 14 | 0 | ACK wait helps only slightly because some chunks are already missing before ACK can arrive |

Important diagnostic:

```text
UDP socket requests SO_RCVBUF = 8 MB
actual granted SO_RCVBUF = 425,984 bytes
no-AE/zstd feature payload ≈ 1.06 MB per frame
```

So the corrected-source failure is not caused by the spatial-map server,
the edge receive queue, route/NPC mismatch, or full-result downlink removal.
The current no-AE feature frame is larger than the actual granted UDP receive
buffer, so a single burst can lose chunks before the edge assembler completes
the frame. Once one chunk is missing, the whole feature message expires as a
partial/stale UDP message.

This also explains why ACK/backpressure alone is insufficient: waiting after a
send cannot recover chunks already dropped during the burst.

## Corrected-source rerun after restoring ideal loopback buffers

Date: 2026-07-29

Restored runtime sysctls:

```bash
sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608
```

Verification:

```text
SO_RCVBUF after requesting 8 MiB = 16,777,216 bytes
```

Rerun folders:

- `runs/corrected_ideal_loopback_none_20f`
- `runs/corrected_ideal_loopback_ackwait_20f`

| Run | Edge result behavior | Front frames | Edge/map frames | UDP partial messages dropped | Edge queue drops | Result |
|---|---|---:|---:|---:|---:|---|
| `corrected_ideal_loopback_none_20f` | no downlink result | 20 | 20 | 0 | 0 | fixed-buffer open-loop delivery is clean |
| `corrected_ideal_loopback_ackwait_20f` | tiny ACK + front waits for ACK | 20 | 20 | 0 | 0 | ACK path is also clean |

Post-warm-up timing snapshot, excluding the first 5 delivered frames:

| Run | front→edge p50 | edge queue p50 | tail p50 | map service p50 | capture→map-update p50 |
|---|---:|---:|---:|---:|---:|
| no-return | 6.6 ms | 0.0 ms | 9.2 ms | 0.0 ms | 184.9 ms |
| ACK-wait | 7.7 ms | 0.0 ms | 8.2 ms | 0.0 ms | 217.5 ms |

Interpretation:

- Restoring the ideal loopback buffer removes the UDP multipart loss. This
  validates the earlier diagnosis: the 425,984-byte granted buffer was the
  reason no-AE uplink-only frames were disappearing.
- The current 20-frame smoke does **not** show edge queue or spatial-map service
  bottleneck; both are effectively zero in this no-op map-update setup.
- `capture→map-update` is larger than `front→edge + tail` because it includes
  front-side sensor/radar wait and preprocessing before the feature send. Do not
  report `capture→map-update` as transport latency.
- For the next reportable loopback run, use the restored ideal buffer and a
  longer run/warm-up so the capture-side timing distribution is stable.

## Model-boundary timing rerun

Date: 2026-07-29

Run folder:
`runs/corrected_ideal_loopback_none_model_boundary_50f`

This run keeps the Track-1 no-return architecture, but adds the more relevant
model-path timestamp:

```text
t_backbone_input_perf = fused RGB+radar tensor ready, immediately before model.encode(...)
```

This separates the reportable split-inference/map path from optional
CARLA/sensor/radar preparation time.

Result:

```text
front frames = 50
edge/map frames = 50
udp_partial_messages_dropped = 0
edge_receive_queue_dropped = 0
```

Post-warm-up timing, excluding first 10 delivered frames:

| Stage | p50 | p95 |
|---|---:|---:|
| optional capture → backbone input | 158.0 ms | 240.5 ms |
| model input prep tensor | 12.0 ms | 21.7 ms |
| front backbone/encode | 3.6 ms | 7.7 ms |
| feature serialize + zstd | 0.8 ms | 1.3 ms |
| backbone input → front send | 4.9 ms | 8.9 ms |
| UDP send call | 4.9 ms | 8.2 ms |
| front send → edge receive | 6.9 ms | 11.0 ms |
| edge tail | 9.7 ms | 24.9 ms |
| map UDP queue/ingest | 7.9 ms | 60.6 ms |
| map service | ~0.0 ms | ~0.0 ms |
| **core backbone input → map update** | **33.0 ms** | **78.8 ms** |
| optional capture → map update | 195.4 ms | 280.9 ms |

Interpretation:

- For the project-relevant split-inference/spatial-map path, use
  `backbone_input_to_map_update_done_ms`. This is the latency from fused model
  input entering the front backbone to the edge map becoming updated.
- The core p50 is about **33 ms**, which is consistent with the earlier
  ideal-loopback closed-loop floor.
- The larger optional `capture_to_map_update_done_ms` mostly comes from
  pre-model CARLA/radar/sensor preparation and should be kept separate unless
  the question is full sensor-to-map information age.

## What was validated

- The copied client runs in `--uplink-only-spatial-map` mode.
- The front does **not** wait for returned detections:
  `result_received=False` for all front frames in the smoke runs.
- The edge/back worker can publish decoded detections directly to the copied
  spatial-map server.
- The map server writes `capture_to_map_update_done_ms`, which is the Track-1
  `Y_up` staleness metric.
- The no-AE/zstd feature payload scale matches expectations:
  about **1.06 MB compressed per frame**, about **2.90 MB uncompressed**.

## Smoke runs

These are short 20-frame validation runs, not reportable throughput runs.

| Run | Feature chunk size | Edge receive queue | Front frames | Edge/map frames | Notes |
|---|---:|---:|---:|---:|---|
| `smoke_loopback_10fps_no_delay` | 60 KB | old single receive/process | 20 | 3 | direct publish worked, but many feature messages did not complete at edge |
| `smoke_loopback_10fps_queue_drain` | 60 KB | 32 frames + 5 s drain | 20 | 3 | queue did not fill/drop; missing frames are below the queue |
| `smoke_loopback_10fps_queue_chunk8k` | 8 KB | 32 frames + 5 s drain | 20 | 6 | smaller chunks helped only slightly; chunk size alone is not the fix |
| `smoke_loopback_10fps_udp_counters` | 60 KB | 32 frames + 5 s drain | 20 | 3 | confirmed **17 incomplete UDP feature messages** expired in the assembler |

## Timing snapshot

For frames that did reach the edge/map:

| Run | front→edge mean | tail p50 / max | map service | capture→map p50 / max |
|---|---:|---:|---:|---:|
| queued 60 KB | 9.2 ms | 15.4 / 219.6 ms | ~0 ms | 257.7 / 394.6 ms |
| queued 8 KB | 10.0 ms | 8.4 / 301.5 ms | ~0 ms | 254.2 / 582.5 ms |

The high max tail values are first-frame/CUDA/model warm-up effects. After
warm-up, tail inference is small. The spatial-map update itself is negligible
with `--map-update-delay-ms 0`.

## Main interpretation

The first bottleneck is **not** the copied spatial-map server and not the new
edge receive queue. The edge queue depth stayed near zero and
`edge_receive_queue_dropped` stayed zero.

Instead, complete feature messages are missing before they enter the edge queue.
The UDP-counter smoke confirmed this directly: after a 20-frame front run, only
3 feature messages completed and **17 partial messages were dropped** by the
assembler after missing chunks became stale.

This points to UDP chunk/message completion loss under the new no-wait traffic
pattern:

```text
~1.06 MB feature frame
  -> 18-19 chunks at 60 KB, or ~131 chunks at 8 KB
  -> any missing chunk drops the whole feature frame
  -> old closed-loop result wait used to pace these bursts
```

This is an important Track-1 finding: removing the result wait makes the
architecture more realistic, but it also exposes that the current UDP chunk
transport is not robust enough for no-AE uplink-only streaming.

## Recommended next step before the full matrix

Do **not** run the long 10/20 FPS no-AE matrix yet as if transport were healthy.
First do one of the following:

1. Add UDP assembler counters for partial/stale messages, chunks received, and
   completed messages. The first lightweight counter is already in the copied
   Track-1 worker as `udp_partial_messages_dropped`; if needed, add per-chunk
   counts next.
2. Run a reduced-payload Track-1 loopback smoke:
   - no-AE + ROI/int4 payload around 392 KB; or
   - AE-128 + ROI 0.5 + uint6 payload around 155 KB.
3. If reduced payload works, the story is clean:
   no-wait spatial-map architecture needs either payload reduction, reliable
   transport/backpressure, or a freshness-aware dropping policy.
