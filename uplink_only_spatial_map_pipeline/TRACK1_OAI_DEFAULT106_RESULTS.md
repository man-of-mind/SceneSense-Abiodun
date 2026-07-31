# Track 1 OAI default-106PRB uplink-only results

Date: 2026-07-29/30

Run group: `track1_track1_oai_default106_ttracer_fps10_track1_default106_20260729_204536`

This run uses the Track-1 uplink-only pipeline: CARLA/front split features go to the edge tail and are published toward the spatial-map side. No detection result is returned to the car.

## Configuration

- OAI: default adaptive MCS, 106 PRB, default 7DL/2UL TDD setup from `gnb.sa.band78.fr1.106PRB.usrpb210.conf`
- UE launch: `-r 106 -C 3619200000`
- Model/knob: no-AE baseline, ROI 0, 200k radar PPS, zstd feature transport, fast radar rasterizer
- Target FPS: 10 FPS, duration budget 130 s
- Traffic: normal Town10 drivable route with the corrected traffic count
- Map compute: not measured in this run; for reporting we add an explicit `+30 ms assumed map compute` row

## Main comparison

| Condition | Sent | Processed | Delivery | Actual send FPS | Uplink payload p50 | Chunks p50 | Sensor prep p50/p95 | Front model p50/p95 | UDP send p50/p95 | Uplink transport p50/p95 | Tail p50/p95 | Capture→tail p50/p95 | Capture→map with +30 ms p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ideal loopback | 179 | 179 | 100.0% | 7.22 | 1048.0 KiB | 18 | 29.5/59.9 ms | 21.1/29.7 ms | 6.1/10.7 ms | 1.7/5.5 ms | 7.5/20.4 ms | 72.2/133.4 ms | 102.2/163.4 ms |
| OAI default 106PRB | 1299 | 1239 | 95.4% | 7.08 | 1048.9 KiB | 18 | 36.5/95.8 ms | 24.1/52.1 ms | 6.4/15.0 ms | 65.6/99.1 ms | 10.6/19.9 ms | 154.8/247.2 ms | 184.8/277.2 ms |

Notes:

- `Sensor prep` is measured from the frame timestamp available at the front process to backbone input, excluding model preprocessing. It includes the front-side sensor packaging/rasterization path that contributes to spatial-map staleness.
- `Uplink transport` is `front_to_edge_ms - send_call_ms`; the raw `front_to_edge_ms` includes the UDP send call because the timestamp is taken immediately before the send.
- CARLA producer cadence (`sync_world_tick_ms`, `camera_frame_wait_ms`) is tracked separately. It constrains actual FPS, but it is not folded into `capture→tail` because the current capture timestamp is placed after camera/radar data are available to the client.
- The OAI edge no-return CSV records `uplink_payload_bytes=0`; therefore payload is taken from the front send-events CSV.

## OAI radio / queue summary

- Scheduled UL throughput: 42.2 Mbps
- Average / p50 / p95 MCS: 15.3 / 16 / 21
- Average / p50 / p95 PRB allocation: 100.4 / 106 / 106
- Retransmission grant rate in this trace: 0.000

Cross-layer notes from `uplink_layer_latency.md`:

- LCID 4 (data bearer): occupancy p50=0.0 KB  p95=930.2 KB  max=4024.4 KB  (461019/1730583 samples nonzero)
- SDU drain: 1420.3 MB over 295.5 s = 38.5 Mbps
- **RLC mean queueing delay (Little's law = mean occupancy / throughput):** 34.0 ms
- grant PRB: p50=106  p95=106
- grant MCS: p50=16  p95=21
- BSR reports: 471342 (471342 sent); reported backlog p50=577.1 KB  p95=1034.5 KB  max=4024.1 KB
- SNR dB: p50=50.5  min=50.0  max=51.5
- **UE PDCP-ingress -> gNB PDCP-deliver (whole RAN UL transit):** mean=38.4 ms  p50=37.0  p95=75.1  p99=98.2  max=235.1 ms
- **=> RLC queue-wait is ~89% of the uplink transit**; remainder (air K2 + gNB PHY/MAC/reassembly/PDCP) ~4 ms

## Optimized closed-loop OAI comparator

Both rows below use default OAI 106PRB / 7DL-2UL, no-AE, ROI 0, per-channel uint8, zstd, 200k radar PPS, corrected drivable route, and the fast radar rasterizer.

| Path | Vehicle waits for result? | Sent / received | Delivery | Actual send FPS | Payload p50 | 20 s idle bins | Uplink/feature handling p50 | Result RTT p50/p95 | UL MCS avg / p50 / p95 | Scheduled UL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Track-1 uplink-only OAI | no | 1299 / 1239 | 95.4% | 7.08 | 1048.9 KiB | 38% | 65.6 ms | n/a | 15.3 / 16 / 21 | 42.2 Mbps |
| Optimized closed-loop OAI | yes | 1300 / 1194 | 91.8% | 2.28 | 1046.0 KiB | 78% | 144.8 ms | 156.8/174.7 ms | 7.2 / 7 / 13 | 20.6 Mbps |

## 100 ms traffic-shape check

- Median compressed feature frame: 8.59 Mbit. One full frame in a 100 ms bin is therefore 85.9 Mbps equivalent.
- Track-1 OAI uplink-only app offered data appears in 62.8% of active 100 ms bins. The optimized closed-loop **OAI** no-AE run appears in 25.0% of comparable 100 ms bins.
- In the displayed 20 s zoom window, Track-1 OAI is active in 61.7% of 100 ms bins, while closed-loop OAI is active in 22.4%.
- Nonzero Track-1 app bins are usually one frame (8.60 Mbit p50), with occasional two-frame bins (17.57 Mbit max in the active window).
- MAC scheduling is active in 100.0% of active bins and RLC dequeue in 94.9%. Median drain over all active-window bins is 6.76 Mbit/100 ms, or 67.6 Mbps equivalent.
- RLC occupancy-drain view: one clean observed burst drains from 1069 KiB to 117 KiB in 98 ms, with burst-slope about 80 Mbps.
- Interpretation: this is now an optimized OAI-vs-OAI traffic-shape comparison. Track-1 removes the result-return wait from the vehicle, while the optimized closed-loop OAI run still shows the return-wait/timeout cadence that makes feature bursts sparse.


## Interpretation

Track 1 behaves differently from the earlier OAI closed-loop return-to-car deployment. The OAI traffic-shape panel now compares Track-1 uplink-only OAI against closed-loop OAI, so it isolates the effect of removing result-return waiting from the vehicle-side pacing. The median Track-1 OAI front-to-edge transport-only time is now about 65.6 ms, not the ~200 ms closed-loop symptom.

However, the 1 MB no-AE feature stream is still close to or above the sustained uplink drain rate. The front offers roughly one 1 MB feature frame every ~140 ms in this run, while the measured RLC/air drain is about 38--42 Mbps. That creates BSR/RLC backlog bursts and explains why capture→tail rises from loopback's 72.2 ms p50 to OAI's 154.8 ms p50.

Reliability is the main caveat: edge processed 1239 of 1299 frames (95.4%). Edge queue drops were 0, but UDP partial-message drops reached 59; that points to incomplete multi-chunk UDP reassembly over OAI rather than tail/map compute saturation.

## Plots

- `plots/track1_oai_default106/track1_latency_breakdown_loopback_vs_oai.pdf`
- `plots/track1_oai_default106/track1_oai_traffic_rates_1s.pdf`
- `plots/track1_oai_default106/track1_oai_radio_scheduler_comparison.pdf`
- `plots/track1_oai_default106/track1_oai_delivery_reassembly.pdf`
- `plots/track1_oai_default106/track1_oai_100ms_volume_drain_backlog.pdf`
- `plots/track1_oai_default106/track1_oai_observed_rlc_drain.pdf`

## Next actions

1. Keep this default-OAI uplink-only result as the Track-1 baseline.
2. Repeat Track 1 with reduced payload knobs to test whether backlog and UDP partial-frame loss improve.
3. Add a real map-worker timing path later; until then report map compute as an explicit assumed add-on, not a measured latency.
