# OAI Transport Bottleneck Discussion Notes

Date: 2026-07-20

Purpose: concise discussion points for the team. Current stance: we have strong evidence that the deployed no-AE split payload is stressing the OAI uplink path, but we do **not** yet have a single definitive layer-level proof of exactly where inside OAI the loss/latency is introduced.

## Latency and payload breakdown

Live CARLA frontend, no-AE, zlib/per-channel-u8, 10 FPS target, 1300 frames. The OAI rows use the closed-loop deployment harness: the frontend waits for result/timeout before advancing. The ideal-loopback row uses the same model/payload recipe and is included as the software/transport baseline.

| Condition | Uplink feature payload p50 | Downlink result payload p50 | Delivery | Front p50 | Feature/uplink handling p50 | Edge tail p50 | Downlink p50 | RTT p50 / p95 | Capture→result p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ideal loopback, raised 8 MB buffers | 1091.0 KB, 19 chunks | 11.9 KB, 1 chunk | 100.0% | 43.9 ms | 31.2 ms | 6.6 ms | 4.9 ms | 43.2 / 54.2 ms | 88.0 ms |
| Default OAI, 106 PRB, 7DL/2UL | 1092.6 KB, 19 chunks | 11.2 KB, 1 chunk | 72.6% | 49.4 ms | 187.4 ms | 7.1 ms | 11.1 ms | 209.0 / 279.5 ms | 264.0 ms |
| UL-heavy OAI, 106 PRB, 4DL/5UL | 1092.5 KB, 19 chunks | 11.4 KB, 1 chunk | 71.9% | 48.6 ms | 177.0 ms | 10.3 ms | 9.8 ms | 200.1 / 249.5 ms | 252.9 ms |
| Wider-BW OAI, 273 PRB, 7DL/2UL | 1091.4 KB, 19 chunks | 12.2 KB, 1 chunk | 74.9% | 48.5 ms | 216.1 ms | 7.4 ms | 9.6 ms | 235.2 / 263.7 ms | 285.7 ms |

Correction to the correction: the `Wider-BW OAI, 273 PRB` row is valid for the
manual `bw273_mu1` run. The matching RAN proof is in
`../metrics_logs/oai_config_sweep/bw273_mu1_carla_live/`: gNB `DLBW 273` /
`fp->N_RB_DL=273`; UE `-r 273 -C 3649260000 --ssb 516`; tunnel IP `10.0.0.5`,
matching the CARLA frontend bind host. The failed automated `prb_273` sweep is a
separate non-reportable case caused by mismatched 273PRB bring-up parameters.

The asymmetry is the main point: the uplink carries a dense split-feature tensor of about `1.09 MB/frame`, broken into about `19` UDP chunks. The downlink carries only the compact model result: object boxes, class/score metadata, centroid/location fields, and a small segmentation/result summary. That return payload is only about `11–12 KB` and fits in one UDP chunk, so it is naturally much cheaper and less fragile than the uplink feature burst.

This is why the downlink latency stays low: there is little data to serialize, transmit, reassemble, and parse on the car side. The heavy operation is completing the uplink feature transfer before the edge tail can run. The loopback row is useful because it shows the same payload can be delivered cleanly when the transport path is ideal; the OAI rows show the large inflation appears in feature/uplink handling and delivery, not in edge compute or downlink result return.

## Key possible insights / contributors

1. **The no-AE feature payload is large enough to stress the path.**
   - Live CARLA no-AE payload is about `1.09 MB/frame`.
   - Each frame is about `19` UDP chunks.
   - Losing or delaying enough chunks makes the whole frame unusable at the application layer.
   - This is still the strongest practical explanation for the bottleneck.

2. **The bottleneck is uplink-side, not model inference or result downlink.**
   - Default OAI 10 FPS live CARLA:
     - `front_p50 ≈ 49 ms`
     - `back_p50 ≈ 7 ms`
     - `downlink_p50 ≈ 11 ms`
     - `feature/uplink handling p50 ≈ 187 ms`
   - Returned result payload is tiny, about `~10–12 KB` and one UDP chunk.
   - So the heavy direction is UE/front → edge/back-half, not edge result → car.

3. **Ideal loopback proves the model/back-half/result path can behave cleanly.**
   - Ideal loopback at the same no-AE/zlib recipe delivered `100%`.
   - Default OAI at 10 FPS delivered only about `72.6%`.
   - That points away from the perception model itself and toward the network/tunnel/transport path.

4. **T-tracer and queue-probe metrics support the uplink-pressure hypothesis.**
   - In the long 10 FPS T-tracer run, the app offered about `17.7–18.1 Mbps` p50/p95 of feature traffic.
   - The UE tunnel averaged `10.84 Mbps` TX but only `0.093 Mbps` RX, matching the payload asymmetry: heavy split-feature uplink, tiny result downlink.
   - UL scheduler windows show PRB pressure: PRB p50/window was `106 RB`, i.e., pinned at the 106-PRB config ceiling.
   - Average MCS was low/moderate: avg MCS p50/window was about `6.5`, with p95 MCS around `13`.
   - Extracted retransmission proxy was `0.0`, so this trace does not yet prove HARQ/retransmission-driven delay. It mainly shows that the large CARLA payload is keeping the UL grant path busy.

5. **The validated OAI config tuning did not solve the live CARLA closed-loop deployment.**
   - Default OAI: `72.6%` delivery, `209 ms` RTT p50.
   - UL-heavy TDD: `71.9%` delivery, `200 ms` RTT p50.
   - Wider-BW 273PRB: `74.9%` delivery, `235 ms` RTT p50.
   - Interpretation: UL-heavy helps latency modestly but does not improve delivery; 273PRB improves delivery slightly but worsens p50 latency. Neither removes the bottleneck.

6. **Replay/open-loop diagnostics showed stronger config sensitivity, but that is not the deployment result.**
   - Replay forced a clean ~`92 Mbps` offered load and showed delivery improving under UL-heavy.
   - Live CARLA closed-loop waits for result/timeout before advancing, so it does not maintain the same offered-load pattern.
   - This difference is important to explain: replay is useful for stress diagnosis, while CARLA frontend is the reportable deployment behavior.

7. **The failure mode still looks like incomplete/dropped frame reconstruction more than slow edge inference.**
   - Back-half compute remains small when a complete frame arrives.
   - Replay diagnostics showed frames either return quickly or effectively disappear; there was not much late draining after the send burst.
   - Candidate location: UDP chunk completion / UE tunnel / GTP-U / PDCP-RLC-MAC buffering path.
   - This is still a hypothesis; we need better OAI-side instrumentation to prove the exact layer.

8. **TDD/UL scheduling may be involved, but it is not the whole story.**
   - UL-heavy TDD improved live RTT p95 from `279.5 ms` to `249.5 ms`, suggesting scheduling shape matters.
   - But delivery did not improve, so the bottleneck is not solved by simply adding more UL slots.
   - Possible remaining contributors: grant timing, BSR behavior, RLC/PDCP queueing, application UDP chunk burstiness, or tunnel/socket drops.

9. **273PRB has one valid result and several invalid failed bring-up attempts.**
   - Valid/reportable: the manual `bw273_mu1` replay and live CARLA runs, proven by matching RAN logs and UE tunnel IP.
   - Invalid/non-reportable: the automated `prb_273` sweep and later T-tracer reproduction attempts that used the wrong/default center-frequency path or never reached a usable tunnel.
   - Do not show a 273PRB PRB/MCS T-tracer plot yet; rerun T-tracer with the exact working recipe before making a radio-layer 273PRB claim.

10. **Codec/serialization time is a separate local compute contributor.**
   - On zlib, ideal loopback has about `31 ms` feature payload-handling and about `44–49 ms` front time.
   - That is channel-independent overhead and should not be confused with OAI RF latency.
   - The zlib/zstd matrix suggests codec choice can lower local overhead, but it will not alone explain OAI delivery loss.

11. **Current evidence does not yet isolate the exact OAI layer.**
    - We have not yet cleanly captured UE-side/gNB-side BSR, RLC occupancy, PDCP drops, GTP-U drops, or socket receive queue drops aligned with each CARLA frame.
    - The honest message to the team should be: "The bottleneck is strongly uplink/payload related, but the exact internal OAI layer is still under investigation."

## Recommended plots to show

Use these as the clean core deck figures:

1. `plots/oai_bottleneck/loopback_vs_oai_10fps.pdf`
   - Shows the main gap: ideal loopback gives clean delivery/low post-send latency; default OAI inflates the uplink-side path and loses returned results.

2. `plots/oai_bottleneck/oai_live_latency_breakdown.pdf`
   - Shows that feature/uplink handling dominates the live OAI budget, while back-half compute and downlink remain small.

3. `plots/oai_bottleneck/oai_live_config_delivery_rtt.pdf`
   - Shows the validated live CARLA config comparison: default 106PRB, UL-heavy 106PRB, and manual validated 273PRB.

4. `plots/oai_bottleneck/oai_default_fps_sweep.pdf`
   - Shows default OAI delivery and RTT across requested FPS. Include the caveat that the current harness is closed-loop and waits for result/timeout.

5. `plots/oai_ttracer/ttracer_ul_mcs_prb_timeseries.pdf`
   - Shows UL PRB allocation pinned near the 106-RB ceiling while avg MCS stays low/moderate. This is the cleanest radio-side picture of uplink pressure.

6. `plots/oai_ttracer/ttracer_tunnel_tx_rx_timeseries.pdf`
   - Shows the transport asymmetry directly: UE tunnel TX carries the heavy feature stream, while UE tunnel RX/downlink result traffic stays tiny.

7. `plots/oai_ttracer/queueprobe_app_vs_ran_rate_timeseries.pdf`
   - Diagnostic plot showing app offered feature load against OAI layer drain-rate traces. Use as supporting evidence, not as the main deployment headline.

Optional backup / appendix:

8. `plots/oai_bottleneck/replay_vs_carla_delivery.pdf`
   - Use only if someone asks why the replay diagnostic looked more optimistic than CARLA. It explains the open-loop vs closed-loop difference for validated/default and UL-heavy conditions.

9. `plots/oai_ttracer/ttracer_carla_vs_iperf_radio_summary.pdf`
   - Use if the team asks how CARLA compares to iperf. It suggests CARLA behaves more like large-block traffic than small-datagram traffic, but the units are mixed, so keep it as a qualitative appendix.

10. `plots/ideal_loopback_latency_breakdown.pdf`
   - Use as a baseline/floor plot showing local software path behavior with ideal loopback.

11. `plots/ideal_loopback_payloads.pdf`
   - Use if someone asks whether payload size changed across FPS. It did not; feature payload stayed near ~1.1 MB.

## Suggested team wording

> We have narrowed the bottleneck to the uplink feature-transfer path for the large no-AE split payload. The edge tail and downlink result-return path are small and stable. OAI config changes produce only modest shifts in live CARLA closed-loop deployment, so the exact OAI-layer cause is still under investigation. Current hypotheses are chunk/frame completion under UDP, UE tunnel/GTP-U buffering, PDCP/RLC/MAC queueing, and UL grant/BSR behavior. The next evidence needed is layer-aligned OAI counters/logs or a CARLA queue-probe run that preserves real frames while forcing open-loop FPS.

## Next evidence to collect

- Run CARLA `--queue-probe-mode` at 10 FPS on default OAI and one selected tuned config to bridge replay and live frontend.
- Add/collect UE/gNB-side counters aligned by wall-clock:
  - BSR / UL buffer status
  - UL grants / PRB allocations
  - RLC/PDCP drops or queue occupancy
  - GTP-U/TUN packet drops
  - socket receive drops / UDP buffer occupancy if available
- Compare no-AE zlib vs zstd and compressed AE payloads over the same OAI path to separate local codec overhead from radio/tunnel delivery limits.
