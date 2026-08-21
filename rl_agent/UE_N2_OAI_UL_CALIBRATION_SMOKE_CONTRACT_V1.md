# UE-N2 bounded OAI uplink calibration-smoke contract v1

## Claim boundary

UE-N2 is one short, single-UE, uplink-only OAI RFsim experiment. It tests the
frozen UE-N1 control and evidence path at a 100-ms schedule. It does not run
CARLA, sweep the 72 split profiles, train a policy, promote a numeric channel
bound, or claim that a Telnet response is an RF-sample application timestamp.

The only non-clean commands are the historically attach-safe probe values
`-10`, `-8`, `-5`, and `-4` dB. Each is held for 3 seconds while the identical
command is still issued every 100 ms. This preserves the intended actuator
cadence and gives the scheduler EMA time to respond. The range is evidence
selection, not an accepted operating bound. Attachment and every cleanup use
`-50`.

## One-command lifecycle

The runner requires the OAI core to be healthy and a cold local RAN. It then:

1. validates the final UE-N1 v2 bundle and rechecks all frozen runtime seals;
2. creates and hashes effective single-UE gNB and UE configs with exactly
   `noise_power_dB=-50`, `ploss=0`, and no global `noise_power_dBFS`;
3. starts one gNB and one UE with SINR scheduling and no forced UL MCS;
4. starts a 10-Hz, 25-kB-per-frame, single-chunk UDP workload bound to
   the single IPv4 address discovered on `oaitun_ue1` and received in the
   ext-DN namespace on port `56120`;
5. requires the current-session UE tunnel, reachability, telemetry recording,
   exact active `rfsimu_channel_ue0`, and fresh PUSCH traffic;
6. opens one persistent Telnet session, dynamically resolves the model index,
   and reuses the session for the entire trace;
7. schedules commands from one monotonic anchor, skips obsolete commands, and
   never emits a catch-up burst;
8. restores `-50`, verifies it with `show current`, stops the RAN, and publishes
   one create-only evidence directory.

Connection loss permanently fails the trace. A separate cleanup connection
may only restore `-50`; successful cleanup cannot turn failure into success.

## Timing and telemetry

Every sent command records scheduled, send, and full-response/prompt receipt
times in monotonic and UTC wall-clock domains. Send and response receipt are
lower and upper brackets for command-handler completion. No field named
`command_applied_at` is produced.

The bounded meeting-time runner uses the pinned stock OAI `record`, `replay`,
and `csv` tools. It preserves the raw T files, but stock CSV exposes only
local `HH:MM:SS.us` and no per-event collector-ingest clock. Therefore its
terminal is explicitly `PARTIAL_EVIDENCE`: it may characterize command ACK
timing and descriptive per-plateau radio values, but it cannot claim the full
UE-N1 raw-event envelope or a causal first-effect lag. Missing data is never
zero- or forward-filled. A later direct raw decoder may strengthen this
evidence without changing the physical smoke.

Only the first command of each plateau is a value transition. The 29 repeated
100-ms no-op commands do not reset EMA settling or first-effect analysis. An
EMA value is labelled settled only after at least 120 accepted PUSCH
observations since that value transition.

`GNB_MAC_PUSCH_POWER_CONTROL.snrx10/10` is the MAC-normalized instantaneous
PUSCH observation. `GNB_MAC_UL_MCS_DECISION.avg_snr_x10/10` is the scheduler
EMA. Selected/final MCS, grants/TBS, UE retransmission-round proxy, and
BSR/RLC backlog are kept distinct. Direct UL BLER remains
`UNAVAILABLE_UNRESOLVED` and is never filled with zero.

All radio values are post-action collector evidence. The runner does not mark
them as causal UE policy observations and does not fabricate a UE-visible
availability timestamp.

## Terminal interpretation

`UE_N2_SMOKE_CAPTURED` means the bounded actuator and partial evidence path executed,
the 100-ms trace was captured, and clean restoration was verified. It does not
mean the desired-to-command mapping, attach-safe lower achieved-SNR bound, or
24.5-dB MCS boundary is accepted. Those remain UE-N3/UE-N4 work.
