# OAI 273PRB CARLA + T-tracer Rerun

Date: 2026-07-22

Status: **obsolete / pending rerun**.

The previous 273PRB live CARLA + T-tracer result used the old frontend command
with `60` requested vehicles, `20` pedestrians, and obey-all-lights behavior.
The raw run folders, summary CSVs, RAN-side companion logs, and PRB/MCS plots
were deleted on 2026-07-22.

Do not report the old 273PRB latency, PRB-allocation, MCS, or scheduled-rate
numbers. Rerun 273PRB with the corrected `run_common.sh` deployment scene:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- live CARLA frontend, 10 FPS target, 1300 frames;
- working 273PRB OAI recipe: matching gNB 273PRB config and UE launch using
  the validated center-frequency/SSB settings.

After the corrected rerun, regenerate the PRB/MCS and tunnel TX/RX plots from
the new artifacts only.
