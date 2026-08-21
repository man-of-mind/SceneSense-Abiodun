# UE-N1 OAI uplink actuator interface contract v1

## Purpose and authority

UE-N1 freezes only the interface between a saved desired-achieved-PUSCH-SNR
trace, the OAI RFsim uplink channel actuator, and the achieved-radio
observations that UE-N2 will measure. It does not calibrate desired achieved
PUSCH SNR to an RFsim command, select a numeric operating range, replay a
trace, launch OAI/CARLA, or open a control socket.

The accepted terminal state is `FROZEN_INTERFACE_ONLY`, with UE-N2 as the next
checklist item. UE-A4 remains the predecessor authority for the 72-action
technical registry; UE-N1 neither filters nor retargets that registry.

## Frozen causal chain

```text
saved desired_achieved_pusch_snr_db (experiment truth)
    -> future calibrated desired-achieved-SNR-to-command mapping
    -> channel_command_db
    -> gNB RFsim AWGN noise_power_dB actuator
    -> MAC-normalized instantaneous achieved PUSCH SNR
    -> scheduler EMA SNR
    -> selected/final UL MCS and grant outcome
```

`desired_achieved_pusch_snr_db`, `channel_command_db`, instantaneous MAC-
normalized achieved SNR, and scheduler EMA SNR are four different quantities.
The desired trace is never relabelled as achieved SNR and is not a causal
policy observation. It is also distinct from OAI's fixed
`pusch_TargetSNRx10=150`, which is a UE power-control target retained from the
gNB configuration rather than the experiment trace. UE-N1 defines no numeric
desired-to-command mapping and no attach-safe lower or upper achieved-SNR
bound.

## Physical actuator

- Topology: one UE; gNB is the RFsim server.
- Direction: uplink only.
- Channel object: `rfsimu_channel_ue0`.
- Owner and type required at preflight: `rfsimulator`, `AWGN`.
- Mutable parameter: exact channelmod spelling `noise_power_dB`.
- Fixed per-channel path loss: `ploss=0`.
- Global `noise_power_dBFS`: required to be unset so it cannot replace the
  per-channel noise source.
- UE-side/downlink objects such as `rfsimu_channel_enB0` are not actuated.

The channel-model integer index is session-local and must never be hardcoded.
At every gNB session, issue `channelmod show current`, resolve exactly one
active object named `rfsimu_channel_ue0`, validate its owner/type/path-loss,
and use only the returned index in:

```text
channelmod modify <resolved_model_index> noise_power_dB <channel_command_db>
```

`channel_command_db` is the scalar consumed by RFsim. It is not a promise of
dBm, dBFS, desired achieved SNR, or measured achieved SNR.
Because OAI parses this field with permissive `atof`, UE-N2 must first require
a finite canonical base-10 decimal string. Booleans, NaN/infinity, exponents,
leading/trailing whitespace, junk suffixes, and control characters are invalid
before any command is constructed.

## Attach and control lifecycle

The single UE attaches while `rfsimu_channel_ue0.noise_power_dB=-50` and
`ploss=0`. The pinned `channelmod_rfsimu.conf` is only a source template and
contains `-10`; it is not the effective UE-N2 runtime configuration. UE-N2
must create and hash an effective runtime configuration or exact override,
record the effective argv/config, and verify `show current` reads `-50` before
attachment. Runtime channel commands are prohibited until UE attachment,
`oaitun_ue1`, reachability, an active traffic source, a ready telemetry
recorder, the expected channel object, and a fresh current-session PUSCH
observation have all been verified. Old CSV evidence cannot open the gate.
Failure and normal shutdown restore the clean `-50` command.

UE-N2 must use one persistent Telnet connection to the gNB control endpoint
for the entire trace. Reconnecting for each sample is not compliant with this
interface. Every modification requires a returned prompt/response and an
echoed owner/path-loss/noise state consistent with the command. The modify
response does not echo the model name or index: those remain bound to the
current `control_session_id` and the preceding `show current` resolution. An
error, missing object, duplicate name, unexpected owner/type, nonzero path
loss, response mismatch, or lost connection fails closed. A connection loss
terminates the trace. A new connection may be opened only for best-effort
cleanup to restore `-50`; cleanup success never changes the failed trace to a
pass.

## Schedule and timing semantics

- Trace indices are unique, contiguous, and ordered.
- Scheduled times are derived from one `time.monotonic_ns()` anchor at exact
  100-ms increments. Wall-clock time is recorded only for cross-log joining.
- The player never changes the next schedule relative to actual completion
  time, and it never emits a catch-up burst of obsolete commands.
- Exact lateness/jitter acceptance bounds are deferred to UE-N2 evidence.

For each command, record the scheduled time, local send time, and time at which
the full Telnet response/prompt is received in both monotonic and wall-clock
domains. The send time is a lower timing bracket and response receipt is an
upper timing bracket for command-handler completion. Response receipt is not
an RF-sample application timestamp. A field or claim named
`command_applied_at` is therefore prohibited. The first causally later PUSCH
event is only a post-command candidate, not proof that its RF samples used the
new channel. UE-N2 estimates application/effect lag from the measured step
response rather than assigning an invented instant.

## Exact telemetry bindings for UE-N2

The authoritative event identifiers and field order come from the pinned OAI
`T_messages.txt`:

- `GNB_MAC_PUSCH_POWER_CONTROL`: per-received-PUSCH `snrx10`, plus `rnti`,
  radio frame/slot, TBS, RB size, scheduled MCS, power-control and RSSI fields.
  `snrx10 / 10` is the MAC power-control-normalized instantaneous PUSCH SNR,
  not the raw PHY estimate. Retain `txpower_calc`; the CQI-domain value is
  `(snrx10 + 10*txpower_calc)/10`, quantized in 0.5-dB steps and censored when
  the PHY UL-CQI saturates. The event is conditional on an accepted receive
  context; a missing sample is unavailable/DTX and must never become zero.
- `GNB_MAC_UL_MCS_DECISION`: `avg_snr_x10`, MCS table, selected/pre-PHR/
  post-PHR/final MCS, buffer, RB, and final-TBS fields. `avg_snr_x10 / 10` is
  the scheduler estimate used by the SINR policy.
- `GNB_MAC_UL`: final UL scheduler MCS and TBS cross-check.
- `NRUE_MAC_DCI_GRANT`: UE-observed grant, HARQ process, NDI, RV, round, MCS,
  RB allocation, and TBS. Grant rounds are a retransmission proxy, not a
  direct UL CRC or BLER measurement.
- `NRUE_MAC_RLC_BUFFER_STATUS` and `NRUE_MAC_BSR_STATUS`: UE causal backlog
  and BSR state. Missing records cannot be forward-filled as zero.
- `GNB_MAC_BLER_MCS_DECISION` does not provide direct UL BLER evidence in the
  current SINR-policy trace path, which can bypass the BLER selector. Direct
  UL BLER is `UNAVAILABLE_UNRESOLVED`, never zero-filled. UE-N2/UE-N3 lower-
  bound acceptance requires a genuine UL CRC/HARQ outcome source or an
  explicit missing-evidence gate.

Scheduled MCS/TBS and decoded DCI are grants, not confirmed delivery. EMA
settling is indexed by accepted PUSCH-observation count, with age since the
last command also retained; it is not assumed to advance once per 100-ms wall-
clock step.

OAI T events carry a producer `CLOCK_REALTIME` emission time, and the CSV
`time` column formats that source time. It is not recorder availability. UE-N2
must add live-ingest wall-clock and monotonic availability timestamps. Radio
frame/slot wraps every 10.24 seconds and cannot be a join key alone: joins
require `control_session_id`, RNTI, unwrapped radio cycle/source time, and the
separate ingest time. Decision `frame/slot` and scheduled
`sched_frame/sched_slot` remain distinct.

## Scheduler constraints frozen for the later calibration

- `SCENESENSE_MCS_POLICY=sinr`.
- `SCENESENSE_FORCE_UL_MCS` unset.
- fixed `pusch_TargetSNRx10=150`, labelled only as the OAI power-control target.
- MCS table 0 and one UL layer; any other table/layer contract fails preflight.
- Existing 106-PRB, numerology-1, band-78, TDD, and PUSCH target configuration
  remain fixed while the actuator is calibrated.

The scheduler EMA and later PHR/resource constraints mean selected and final
MCS can differ. UE-N2 must retain both rather than presenting the selected MCS
as the executed grant.

## Evidence boundary

The pinned historical runtime-switch artifact proves only that a two-UE run
resolved active RFsim UL objects and completed a read-modify-read transition
from clean channel commands to a degraded command. Its enclosing experiment
ended `FAILED_HOLD` because both UEs measured approximately 6-dB PUSCH SNR and
MCS 8, outside that experiment's then-registered expected rung. It is mechanism
evidence; it is not single-UE calibration, 100-ms cadence evidence, an
attach-safe bound, or a passing UE-N1 runtime result.

## Explicitly deferred to UE-N2 and later

- persistent-connection implementation and socket execution;
- command latency/jitter and missed-deadline measurements;
- effective runtime-config/argv and executable/library seals reverified at
  UE-N2 preflight;
- command-to-instantaneous-SNR and command-to-EMA response;
- numeric command range and attach-safe achieved-SNR bounds;
- desired-achieved-SNR-to-command interpolation or inverse calibration;
- scheduler settling, MCS occupancy, BLER/HARQ, queue, and recovery results;
- four trace-family parameters; and
- any CARLA, perception, map, or policy experiment.
