# UE-N1 OAI uplink actuator interface v2

**Status:** `FROZEN_INTERFACE_ONLY` — final v2 authority; next UE-N2

This create-only v2 bundle supersedes immutable v1 as pre-final observation-
audit evidence. It freezes the single-UE gNB-side RFsim uplink actuator:
`rfsimu_channel_ue0`, dynamically resolved per control session, with OAI
parameter `noise_power_dB`, canonical experiment field
`commanded_noise_power_db`, `ploss=0`, global noise unset, and clean `-50`
attachment/restoration.

`desired_achieved_pusch_snr_db` and `commanded_noise_power_db` are experiment
control/evaluation only and are excluded from policy state. gNB radio and
scheduler values remain post-action collector evidence until a measured UE-
visible feedback path supplies a policy availability timestamp no later than
the decision cutoff. Collector ingest time is not policy availability.

UE-N2 must use persistent Telnet and a monotonic 100-ms/no-catch-up schedule.
Send and response receipt bracket handler completion; the response ACK is not
an application timestamp. First-effect lag is estimated from the measured step
response. Direct UL BLER remains `UNAVAILABLE_UNRESOLVED` until genuine UL
CRC/HARQ evidence is bound.

The raw-event envelope preserves RAN epoch/session, source index and full
timestamp, unwrapped slot, collector ingest times, raw hash, and explicit
missing reason. Missing PUSCH is unresolved and is not called DTX without DTX
evidence.

Current `nr-softmodem`, `nr-uesoftmodem`, `libtelnetsrv.so`, and
`librfsimulator.so` hashes are sealed and require UE-N2 preflight recheck. No
numeric calibration/bounds, runtime edit, OAI/CARLA run, or socket execution
occurred.
