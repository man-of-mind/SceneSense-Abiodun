# OAI Transport Bottleneck Discussion Notes

Date: 2026-07-22

Purpose: concise discussion points for the team after cleaning the obsolete
60-vehicle live-deployment runs. Current stance: the corrected default 106PRB
OAI run still shows a strong uplink/payload bottleneck, but we should rerun the
other conditions before reporting them.

## 2026-07-22 cleanup note

The earlier loopback/default-OAI/FPS/UL-heavy/273PRB/t-tracer artifacts used the
wrong frontend command: `60` vehicles, `20` pedestrians, and obey-all-lights.
Those raw run folders, summaries, and plots were deleted on 2026-07-22. They
should not be reported.

The currently reportable live Step-1 result is the corrected default 106PRB
codec A/B:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- live CARLA frontend, 10 FPS target, 1300 frames.

## Latency and payload breakdown

Live CARLA frontend, no-AE, per-channel-u8, default OAI 106PRB, 10 FPS target,
1300 frames. This is closed-loop deployment behavior: the frontend waits for
result/timeout before advancing.

| Condition | Uplink feature payload p50 | Downlink result payload p50 | Delivery | Front p50 | Feature/uplink handling p50 | Edge tail p50 | Downlink p50 | RTT p50 / p95 | Capture→result p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected drivable OAI, 106PRB, zlib | 1084.8 KB, 19 chunks | 7.1 KB, 1 chunk | 72.1% | 48.0 ms | 183.8 ms | 7.1 ms | 9.0 ms | 202.5 / 237.7 ms | 251.4 ms |
| Corrected drivable OAI, 106PRB, zstd | 1055.2 KB, 19 chunks | 2.2 KB, 1 chunk | 83.6% | 25.2 ms | 151.1 ms | 6.9 ms | 3.0 ms | 162.2 / 175.3 ms | 188.0 ms |

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

4. **Codec choice does not explain model accuracy changes.**
   - zstd is lossless after decompression.
   - The corrected live accuracy sanity check shows no localization regression.
   - The offline knob matrix remains the exact proof that the same no-AE/per-channel-u8/ROI-0 profile has identical task metrics under zlib and zstd.

5. **The exact internal OAI layer is still not isolated.**
   - The cleaned result still points strongly to the uplink feature-transfer path.
   - We have not yet rerun corrected-scene BSR/RLC/PDCP/GTP-U/socket-drop instrumentation.
   - The honest wording is: "The bottleneck is strongly uplink/payload related; the exact OAI layer is still under investigation."

## Recommended plots to show now

Use only the corrected drivable-scene plots for the current deck:

1. `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd.pdf`
   - Shows the corrected 106PRB codec A/B: delivery improves and the latency stack shrinks with zstd.

2. `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd_accuracy.pdf`
   - Shows the corrected live localization sanity check: zstd does not degrade model accuracy.

Do **not** show the deleted old loopback/OAI/FPS/UL-heavy/273/t-tracer plots.
Those will be regenerated after rerunning with the corrected command.

## Conditions to rerun with corrected command

- Ideal loopback FPS sweep, to re-establish the local software/transport floor.
- Default OAI FPS sweep.
- OAI queue/backlog probe at 10 FPS.
- UL-heavy 106PRB live CARLA run.
- 273PRB wider-bandwidth live CARLA run.
- 273PRB t-tracer live CARLA run.
- Any layer-level UE/gNB counters: BSR, UL grants, PRB allocation, MCS, RLC/PDCP drops or queue occupancy, GTP-U/TUN drops, and socket receive drops if available.

## Suggested team wording

> We found that some earlier live runs used the wrong crowded frontend command,
> so we removed those artifacts and are rerunning the non-default conditions.
> The corrected default 106PRB result still shows the key pattern: the edge tail
> and downlink result path are small, while the uplink split-feature transfer is
> large and fragile. zstd improves delivery and latency without hurting
> accuracy, but it does not fully solve the OAI bottleneck. The exact internal
> OAI layer remains under investigation.
