# OAI Transport Bottleneck Discussion Notes

Date: 2026-07-22

Purpose: concise discussion points for the team after cleaning the obsolete
60-vehicle live-deployment runs. Current stance: the corrected reruns still
show a strong uplink/payload bottleneck. UL-heavy 106PRB improves latency
slightly; 273PRB allocates more PRBs but does not improve latency because MCS
drops and scheduled throughput stays similar.

## 2026-07-22 cleanup note

The earlier loopback/default-OAI/FPS/UL-heavy/273PRB/t-tracer artifacts used the
wrong frontend command: `60` vehicles, `20` pedestrians, and obey-all-lights.
Those raw run folders, summaries, and plots were deleted on 2026-07-22. They
should not be reported.

The currently reportable live Step-1 results use the corrected deployment
scene:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS for the main
  baseline/config rows;
- AE-128, per-channel-u6, ROI 0.5, 200k radar PPS for the reduced-payload
  follow-up row;
- live CARLA frontend, 10 FPS target for OAI config comparisons.

## Latency and payload breakdown

Live CARLA frontend, no-AE, per-channel-u8, corrected drivable route. OAI rows
are 10 FPS target, 1300 frames. This is closed-loop deployment behavior: the
frontend waits for result/timeout before advancing.

| Condition | Uplink feature payload p50 | Downlink result payload p50 | Delivery | Front p50 | Feature/uplink handling p50 | Edge tail p50 | Downlink p50 | RTT p50 / p95 | Capture→result p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected ideal loopback, zstd | 1053.9 KB, 18 chunks | 2.4 KB, 1 chunk | 100.0% | 26.7 ms | 7.6 ms | 8.3 ms | 1.6 ms | 18.3 / 39.2 ms | 46.1 ms |
| Corrected drivable OAI, 106PRB, zlib | 1084.8 KB, 19 chunks | 7.1 KB, 1 chunk | 72.1% | 48.0 ms | 183.8 ms | 7.1 ms | 9.0 ms | 202.5 / 237.7 ms | 251.4 ms |
| Corrected drivable OAI, 106PRB, zstd | 1055.2 KB, 19 chunks | 2.2 KB, 1 chunk | 83.6% | 25.2 ms | 151.1 ms | 6.9 ms | 3.0 ms | 162.2 / 175.3 ms | 188.0 ms |
| Corrected UL-heavy OAI, 106PRB 4DL/5UL, zstd | 1054.0 KB, 18 chunks | 2.5 KB, 1 chunk | 84.8% | 25.0 ms | 140.8 ms | 7.1 ms | 2.9 ms | 152.1 / 165.2 ms | 177.3 ms |
| Corrected wider-BW OAI, 273PRB, zstd | 1054.6 KB, 19 chunks | 2.4 KB, 1 chunk | 85.4% | 26.0 ms | 174.2 ms | 7.4 ms | 3.0 ms | 186.1 / 202.6 ms | 212.7 ms |
| Corrected default OAI, 106PRB, AE-128 uint6 ROI0.5, zstd | 152.7 KB, 3 chunks | 2.3 KB, 1 chunk | 99.8% | 21.9 ms | 52.6 ms | 7.5 ms | 3.2 ms | 64.2 / 76.8 ms | 86.5 ms |

The asymmetry is the main point: the uplink carries a dense split-feature tensor
of about `1.05–1.08 MB/frame`, broken into about `19` UDP chunks. The downlink
carries only compact model results: object boxes, class/score metadata,
centroid/location fields, and a small result summary. That return payload is
only a few KB and fits in one UDP chunk, so it is naturally much cheaper and
less fragile than the uplink feature burst.

## Key possible insights / contributors

1. **The no-AE feature payload is large enough to stress the OAI uplink path.**
   - Corrected live CARLA no-AE payload is about `1.05–1.08 MB/frame`.
   - Each frame is about `19` UDP chunks.
   - Losing or delaying enough chunks makes the whole frame unusable at the application layer.

2. **The bottleneck is uplink-side, not edge inference or result downlink.**
   - Edge tail stays around `7 ms`.
   - Downlink result return is only `3–9 ms`.
   - Feature/uplink handling is `151–184 ms`, so the heavy direction is car/front → edge/back-half.

3. **zstd is a real improvement but not a full fix.**
   - Delivery improves from `72.1%` to `83.6%`.
   - RTT p50 improves from `202.5 ms` to `162.2 ms`.
   - Capture→result p50 improves from `251.4 ms` to `188.0 ms`.
   - Feature/uplink handling is still about `151 ms`, so payload reduction beyond lossless entropy coding is still likely needed.

4. **UL-heavy 106PRB helps, but only modestly.**
   - The 4DL/5UL TDD run reduces RTT p50 from `162.2 ms` to `152.1 ms` relative
     to the corrected default 106PRB zstd run.
   - Delivery moves from `83.6%` to `84.8%`.
   - This is useful evidence that uplink scheduling matters, but not enough to
     make the no-AE payload reliable at 10 FPS over this OAI path.

5. **273PRB does not solve the bottleneck by itself.**
   - T-tracer shows the run allocates close to the wider PRB ceiling
     (`260` PRB average in a 1s p50 window).
   - MCS is lower (`3.9` average MCS p50-window) than the UL-heavy 106PRB run
     (`4.5`), so scheduled throughput stays around `17–18 Mbps`.
   - Application latency therefore gets worse, not better.

6. **Stronger feature compression is the first run that removes steady-state loss.**
   - AE-128 + uint6 + ROI0.5 reduces the feature burst from about `1.05 MB`
     to `153 KB`, and chunks from `18–19` to `3`.
   - Delivery is `99.8%` overall, but the two misses are both startup frames;
     after warmup the run delivered `1298/1298` frames.
   - RLC p95 occupancy falls to `118.6 KB`, and measured RAN UL p50 falls to
     `33.6 ms`. This supports the interpretation that the scattered no-AE
     losses were payload/backlog driven rather than downlink or edge-tail driven.

7. **Codec choice does not explain model accuracy changes.**
   - zstd is lossless after decompression.
   - The corrected live accuracy sanity check shows no localization regression.
   - The offline knob matrix remains the exact proof that the same no-AE/per-channel-u8/ROI-0 profile has identical task metrics under zlib and zstd.

8. **The exact internal OAI layer is still not fully isolated.**
   - The cleaned result still points strongly to the uplink feature-transfer path.
   - T-tracer gives useful grant/PRB/MCS/TBS evidence, but not a clean BSR/RLC/PDCP queue/backlog story yet.
   - The honest wording is: "The bottleneck is strongly uplink/payload related; the exact OAI layer is still under investigation."

9. **SNR/CQI/RSRP are not reportable from the current RFsim extraction.**
   - The UE PHY values look like sentinels/placeholders in both corrected
     t-tracer runs.
   - Report PRB allocation, MCS, scheduled uplink rate, app offered rate, tunnel
     TX/RX rate, delivery, and latency instead.

## Recommended plots to show now

Use only the corrected drivable-scene plots for the current deck:

1. `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd.pdf`
   - Shows the corrected 106PRB codec A/B: delivery improves and the latency stack shrinks with zstd.

2. `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd_accuracy.pdf`
   - Shows the corrected live localization sanity check: zstd does not degrade model accuracy.

3. `plots/oai_bottleneck/corrected_transport_latency_breakdown.pdf`
   - Compares loopback/default 106PRB/UL-heavy 106PRB/273PRB latency components.

4. `plots/oai_bottleneck/corrected_transport_reliability_rtt.pdf`
   - Shows delivery versus RTT across the corrected transport/config choices.

5. `../oai_layer_latency/plots/complementary_latency_summary.pdf`
   - Adds the no-AE uint4, fixed-MCS, and AE-128 reduced-payload comparisons
     using the newer layer-latency instrumentation.

6. `../oai_layer_latency/plots/complementary_mcs_prb_summary.pdf`
   - Shows that adaptive OAI still schedules low MCS for both no-AE uint4 and
     AE-128, while fixed-MCS is only a diagnostic control.

7. `../oai_layer_latency/plots/complementary_rlc_buffer_timeseries.pdf`
   - Shows how payload reduction shrinks the RLC queue/backlog.

8. `plots/oai_ttracer/ttracer_ul_mcs_prb_timeseries_ulheavy106.pdf`
   - Shows 106PRB 4DL/5UL PRB allocation and MCS over the run.

9. `plots/oai_ttracer/ttracer_ul_mcs_prb_timeseries_bw273.pdf`
   - Shows 273PRB PRB allocation and MCS over the run.

10. `plots/oai_ttracer/ttracer_tunnel_tx_rx_timeseries_ulheavy106.pdf`
   - Shows app-offered rate and UE tunnel TX/RX pattern for 106PRB 4DL/5UL.

11. `plots/oai_ttracer/ttracer_tunnel_tx_rx_timeseries_bw273.pdf`
   - Shows app-offered rate and UE tunnel TX/RX pattern for 273PRB.

Do **not** show SNR/CQI/RSRP plots from the current t-tracer extraction; those
fields are not valid in this RFsim run.

## Remaining follow-up

- Corrected default OAI FPS sweep, if we still need full FPS sensitivity over OAI.
- Queue/backlog probe at 10 FPS, if we need more direct evidence of application
  pacing/backpressure.
- Deeper UE/gNB counters if available: BSR, RLC/PDCP queue or drops, GTP-U/TUN
  drops, and socket receive drops.

## Suggested team wording

> We found that some earlier live runs used the wrong crowded frontend command,
> so we removed those artifacts and reran the key conditions with the corrected
> drivable route. The corrected results still show the same core pattern: edge
> tail compute and downlink result return are small, while the uplink
> split-feature transfer is the dominant bottleneck. zstd improves latency and
> delivery without hurting accuracy; UL-heavy TDD helps modestly; 273PRB
> allocates more PRBs but does not improve latency because MCS drops and
> scheduled throughput stays similar. The reduced-payload AE-128/uint6/ROI0.5
> run is the cleanest evidence so far: it cuts the uplink feature burst to about
> 153 KB and reaches 100% steady-state delivery on default 106PRB, with RTT p50
> around 64 ms. The exact internal OAI adaptation behavior remains under
> investigation, but the deployment bottleneck is clearly payload/backlog
> dominated.
