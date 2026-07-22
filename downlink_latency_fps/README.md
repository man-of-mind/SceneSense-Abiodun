# Downlink/result-return latency × FPS study

Purpose: measure the deployed split-fusion result-return path from edge tail output back to the ego/front process, while keeping the perception model and CARLA scene fixed.

This study uses the validated moving-ego no-AE deployment recipe:

- model: `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`
- sensor platform: moving ego vehicle, not pole, not parked
- route: 200k training route replay from `fusion_training_data/moving_ego_pps200000_crowded_8loops_stride2/route_progress.csv`
- camera/radar geometry: car-height ego, camera z `1.55`, pitch `-4`, FoV `120`
- radar recipe: `200000` PPS, radar HFOV `120`, raster radius `4`, temporal window `2`
- payload profile: no-AE, per-channel uint8, zlib, ROI `0.0`
- spatial-map stream: disabled for this stage, so the measured path is model result return only

## Transport conditions

| condition | script | meaning |
|---|---|---|
| `ideal_loopback` | `run_ideal_loopback_fps.sh` | current raised-buffer local loopback; clean software/transport floor |
| `bounded_loopback` | `run_bounded_loopback_fps.sh` | temporary old/default UDP buffer cap; bounded-buffer stress / historical comparison |
| `oai_default` | `run_oai_default_fps.sh` | current default OAI path, no OAI config tuning |

Do not mix OAI tuning into this folder. TDD pattern, 5QI/QoS, PRB/bandwidth, and Sionna varying-channel work belong in later network-condition studies.

## Metrics to report

Primary Step-1 fields:

- `result_received`: whether a result returned before the front timeout.
- `round_trip_result_recv_ms`: front send to actual result socket receive.
- `tail_done_to_result_recv_ms`: edge tail completion to front receive.
- `result_send_to_recv_ms_wall`: edge send-start wall-clock to front receive; best one-way downlink estimate when clocks are comparable/synchronized.
- `result_wait_ms`: how long the front loop waited for a result.
- `result_queue_wait_ms`: receive-thread queueing before the main loop consumed the result.
- `feature_payload_bytes` / `feature_payload_chunks`: uplink feature payload.
- `result_payload_bytes_estimate` / `result_payload_chunks_estimate`: returned result payload estimate.
- `front_ms`, `back_ms`: front-side processing/compression and edge tail compute.
- `ego_speed_mps`: sanity check that the runs are moving-ego/in-domain.

For presentation, use the component definitions from `IDEAL_LOOPBACK_RESULTS.md`:

- `front_ms`;
- `uplink_payload_handling_ms = round_trip_result_recv_ms - back_ms - result_send_to_recv_ms_wall`;
- `back_ms`;
- `downlink_ms = result_send_to_recv_ms_wall`;
- `capture_to_result_est_ms = front_ms + round_trip_result_recv_ms` with the caveat that current logs can
  conservatively double-count the feature-send burst.

`result_recv_to_display_ms` and `tail_done_to_display_ms` are only populated when GUI display or overlay saving is enabled. Headless runs without overlay saving correctly leave them as NaN.

## Default sweep

Each run uses the same simulated-duration target. By default:

```bash
FPS_LIST="5 10 20 30"
DURATION_S=130
```

The 200k crowded training route completed 8 loops in about 1030 seconds, so `DURATION_S=130` is approximately one route loop. Use `DURATION_S=260` for roughly two loops.

## Suggested order

1. Ideal loopback first:

   ```bash
   bash downlink_latency_fps/run_ideal_loopback_fps.sh
   ```

2. Summarize:

   ```bash
   python3 downlink_latency_fps/analyze_downlink_fps.py downlink_latency_fps/runs
   ```

3. Bounded/default-buffer loopback only after reviewing the ideal result. This script changes host sysctl values temporarily and restores the previous values on exit. Start with a calibration, not the full sweep:

   ```bash
   CONFIRM_SYSCTL=1 FPS_LIST=10 DURATION_S=10 BATCH_ID=calib_YYYYMMDD_bounded \
     bash downlink_latency_fps/run_bounded_loopback_fps.sh
   ```

   The clean calibration delivered only 3/100 no-AE frames under the old 208 KB cap, so bounded loopback is currently
   a reliability/buffer-failure condition rather than a clean latency point. The earlier 1/100 calibration is
   superseded because it also exposed stale-port and cleanup-trap hygiene issues. See `BOUNDED_LOOPBACK_CALIBRATION.md`.

4. Default OAI after the OAI CN/RAN/back-half are confirmed healthy:

   ```bash
   bash downlink_latency_fps/run_oai_default_fps.sh
   ```

   Current OAI status from the first setup check: OAI core containers were healthy, but `oaitun_ue1` did not exist,
   so the default OAI sweep still needs UE/RAN/back-half bring-up before running.

## Safety checks added

- Local loopback scripts now fail fast if the UDP back-half port is already occupied, instead of accidentally using a stale listener.
- The bounded-loopback script restores sysctls through `run_common.sh`'s cleanup hook.

## Presentation plots

Generate/update plots with:

```bash
python3 downlink_latency_fps/plot_downlink_fps.py
```

Outputs:

- `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd.png` / `.pdf`
- `plots/oai_bottleneck/oai_106prb_drivable_zlib_vs_zstd_accuracy.png` / `.pdf`

The older ideal-loopback/bounded-loopback plots were generated from the
obsolete 60-vehicle frontend command and were removed on 2026-07-22. Regenerate
those plots only after rerunning the loopback sweeps with the corrected
drivable-scene command.
