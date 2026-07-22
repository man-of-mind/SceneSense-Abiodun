# Ideal Loopback Results

Date: 2026-07-22

Status: **obsolete / pending rerun**.

The previous ideal-loopback FPS sweep used the old frontend command with
`60` requested vehicles, `20` pedestrians, and obey-all-lights behavior. Those
raw run folders, summary CSVs, and presentation plots were deleted on
2026-07-22 so they are not accidentally reported.

Rerun this condition with the corrected `run_common.sh` deployment scene before
using ideal loopback as the Step-1 floor:

- 28 vehicles;
- 35 pedestrians;
- seed 31;
- ego ignore-lights 50%;
- fixed waypoint loop `80,85,91,94,99,80`;
- no-AE checkpoint, per-channel-u8, ROI 0, 200k radar PPS;
- live CARLA frontend, 10 FPS target, 1300 frames.

The previous qualitative lesson is still useful as a hypothesis only: ideal
loopback should establish the local software/serialization/downlink floor, while
OAI adds the real transport/tunnel/RAN behavior. Do not quote the deleted
numbers until the corrected sweep is rerun.
