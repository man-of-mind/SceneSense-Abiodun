# UE-N1 OAI uplink actuator interface v1

**Status:** `FROZEN_INTERFACE_ONLY`

UE-N1 freezes the single-UE, gNB-side RFsim uplink actuator as the exact
`rfsimu_channel_ue0` AWGN object and the exact mutable parameter
`noise_power_dB`. Its session-local integer index must be resolved dynamically
from `channelmod show current`; `ploss` stays zero and global
`noise_power_dBFS` stays unset.

The UE attaches at the clean `-50` command before any runtime change. UE-N2
must use one persistent Telnet connection, a monotonic 100-ms schedule, and no
catch-up burst. The returned response/prompt is an ACK upper bound for command
handler completion, not a physical application timestamp.

Desired achieved PUSCH SNR, RFsim command, MAC-normalized instantaneous PUSCH
SNR, scheduler EMA SNR, and fixed OAI `pusch_TargetSNRx10` remain distinct. No
numeric mapping, command range, achieved-SNR bound, latency bound, or attach-
safe lower limit is claimed here.

Current SINR-policy traces do not provide direct UL BLER. BLER remains
`UNAVAILABLE_UNRESOLVED`; UE grant rounds are only a retransmission proxy, and
missing radio/backlog observations are never zero-filled.

The pinned two-UE runtime switch is mechanism evidence only. Its enclosing
experiment ended `FAILED_HOLD`; it is not a passing single-UE calibration or
cadence result.

No launcher/runtime source was edited, and no OAI, CARLA, Telnet, or other
socket execution was performed. The next checklist item is **UE-N2**.
