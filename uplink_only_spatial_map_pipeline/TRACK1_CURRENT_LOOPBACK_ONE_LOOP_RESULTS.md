# Track 1 current loopback one-loop anchor

Date: 2026-07-30

Run root:
`runs/track1_loopback_one_loop_20260730_current/ideal_none_fps10_map0_fast`

Presentation artifacts:

- `plots/track1_current_loopback_one_loop/track1_current_loopback_horizontal_latency_breakdown.pdf`
- `plots/track1_current_loopback_one_loop/track1_current_loopback_anchor_comparison.pdf`
- `plots/track1_current_loopback_one_loop/track1_current_loopback_horizontal_breakdown_summary.csv`
- `plots/track1_current_loopback_one_loop/track1_current_loopback_one_loop_run_summary.csv`

## Setup

- Architecture: true Track-1 uplink-only path,
  `CARLA/front -> split features -> edge tail -> spatial-map update`.
- No detections/results are returned to the car (`result_mode=none`).
- World: `Carla/Maps/Town10HD_Opt`.
- Corrected fixed drivable route:
  `--ego-fixed-path-spawn-indices 80,85,91,94,99,80`.
- Target FPS: `10`; max frames: `1300`; first 10 frames excluded from latency
  statistics as warm-up.
- Model/knob: no-AE baseline, ROI 0, per-channel uint8, zstd feature transport,
  200k radar PPS, `--radar-rasterizer fast`.
- Traffic: 28 requested vehicles / 35 pedestrians; CARLA spawned 27 vehicles and
  35 pedestrians.
- Ideal loopback UDP buffers were active:
  `net.core.rmem_max=wmem_max=8388608`, with granted `SO_RCVBUF=16777216`.

## Sanity check

| Item | Value |
|---|---:|
| Sent frames | 1300 |
| Edge processed frames | 1300 |
| Map updated frames | 1300 |
| Delivery | 100.0% |
| UDP partial-message drops | 0 |
| Edge receive queue drops | 0 |
| Spatial publisher drops | 0 |
| Simulated CARLA FPS | 10.00 |
| Actual front-send FPS | 9.03 |
| Actual map-update FPS | 9.03 |
| Feature payload p50 / p95 | 1044.5 / 1069.6 KiB |
| Feature chunks p50 / p95 | 18 / 19 |
| Uncompressed feature payload | 2835.0 KiB |
| Map packet payload p50 / p95 | 1955 / 2701 bytes |
| Object count p50 / p95 | 3 / 7 |

Note: in this current uplink-only implementation, `edge_uplink_metrics.csv`
does not receive the post-serialization compressed byte count in the feature
payload metadata, so `uplink_payload_bytes` is zero there. The actual compressed
uplink feature payload is taken from
`front_metrics/streams/*_queue_probe_send_events.csv`.

## Current latency breakdown

Canonical Track-1 presentation convention:

- `Sensor prep` stops at the front-model call, so model preprocessing is not
  hidden inside sensor prep.
- `Front compute` covers model preprocessing, front/backbone encode, and
  feature serialization up to payload-ready time.
- `Uplink handling` is payload-ready/front-send timestamp -> edge receive. This
  matches the earlier loopback convention, so the number remains comparable to
  the knob matrix and Step-1 loopback runs.
- The legacy `front_ms` value is still available for comparison:
  `front_ms = front compute + UDP send-call`.

| Stage | Boundary | p50 | p95 |
|---|---|---:|---:|
| Sensor prep | capture -> front model start | 25.0 ms | 38.0 ms |
| Front compute | front model start -> payload ready | 19.2 ms | 29.8 ms |
| Uplink handling | payload ready/front send -> edge receive | 6.2 ms | 10.3 ms |
| Tail latency | edge receive -> tail done | 6.3 ms | 11.2 ms |
| Result -> map app | tail done -> map ingest | 6.7 ms | 24.3 ms |
| Map apply only | map ingest -> map update done | 0.2 ms | 2.0 ms |
| **Core front -> map update** | front model start -> map update done | **42.5 ms** | **67.7 ms** |
| **Full staleness `L`** | capture -> map update done | **67.9 ms** | **103.6 ms** |

For comparison with the older control-knob tables, legacy `front_ms` is
`24.3 / 36.5 ms` p50/p95. That is `front compute` plus the UDP send-call
component (`4.7 / 8.6 ms` p50/p95).

The reportable freshness lag for the spatial map is the full
`capture -> map update done` age, not only the network/uplink term. In this
run, sensor prep is about 37% of the p50 freshness age.

The map packet is tiny, about 2 KB p50, while the uplink feature packet is about
1045 KiB p50. So the `result -> map app` term is not payload-transfer pressure.
It is local Python handoff overhead: publisher queue wake-up, zlib/JSON
packaging, local UDP socket scheduling, map-server receive, decompression,
parsing, and ingest. The current map apply step itself is only 0.2 ms p50 and
does not yet include future cooperative-map work such as association, occluder
reasoning, JPDA/Hungarian tracking, or advisory generation.

## Which loopback anchor to report?

| Anchor | Frames used | Full `L` p50 / p95 | Actual map-update FPS | Comment |
|---|---:|---:|---:|---|
| Old Track-1 fast 50f | 40 | 93.3 / 136.1 ms | 7.18 | Conservative provenance anchor, short run. |
| Fresh staleness speed sweep | 570 | 67.5 / 101.8 ms | 8.72 | Good staleness-analysis estimate, but it was a traffic-regime sweep rather than the fixed-route one-loop anchor. |
| **Current fixed-route one loop** | **1290** | **67.9 / 103.6 ms** | **9.03** | **Use as the current Track-1 ideal-loopback reporting anchor.** |

Recommendation: report the current fixed-route one-loop run as the primary
Track-1 loopback result. Keep the broader 67-93 ms p50 range only as
provenance/sensitivity, explaining that the older 93 ms number came from a
shorter 50-frame profile and should no longer be the main anchor.

## Interpretation

The clean story is now:

1. The true uplink-only Track-1 path is stable on ideal loopback:
   1300/1300 frames reached both edge and map with zero UDP/reassembly drops.
2. The front/uplink/tail/map core is modest: about 42.5 ms p50 from front-model
   start to completed map update.
3. Full map freshness is larger, about 67.9 ms p50, because the object has
   already aged during CARLA/sensor preparation before the model starts.
4. The current 10 FPS target is nearly met in wall-clock terms: 9.03 FPS actual
   front-send/map-update cadence while CARLA simulation time advances at
   10 FPS.
