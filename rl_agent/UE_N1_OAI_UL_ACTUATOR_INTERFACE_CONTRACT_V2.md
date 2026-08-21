# UE-N1 OAI uplink actuator interface contract v2

## Authority and supersession

UE-N1 v2 is the final interface-only freeze for the single-UE OAI RFsim
uplink actuator. It supersedes the create-only v1 bundle as pre-final evidence
after an observation-causality audit. The v1 bundle remains immutable and is
not deleted, overwritten, or silently promoted.

The accepted v2 terminal state is `FROZEN_INTERFACE_ONLY`, and the next
checklist item is UE-N2. UE-N1 v2 performs no numeric calibration, chooses no
numeric operating bound, replays no trace, edits no launcher/runtime, opens no
socket, and runs neither OAI nor CARLA.

## Frozen physical actuator

- Topology: exactly one UE; the gNB is the RFsim server.
- Direction: uplink only.
- Channel object: exactly one active `rfsimu_channel_ue0` owned by
  `rfsimulator` and typed `AWGN`.
- Mutable OAI parameter: exact spelling `noise_power_dB`.
- Canonical experiment command field: `commanded_noise_power_db`.
- Fixed per-channel path loss: `ploss=0`.
- Global `noise_power_dBFS`: unset.
- Downlink objects and `rfsimu_channel_ue1` are prohibited targets.

The model index is session-local and is never configured or hardcoded. Each
gNB control session starts with `channelmod show current`, resolves exactly one
object by the exact name `rfsimu_channel_ue0`, and binds its returned index to
the `control_session_id`. Only then may UE-N2 form:

```text
channelmod modify <resolved_model_index> noise_power_dB <commanded_noise_power_db>
```

OAI's `noise_power_dB` name remains unchanged. The experiment field is renamed
to avoid implying that the command is a channel SNR. Because OAI parses the
value with permissive `atof`, UE-N2 first requires a finite canonical base-10
decimal string and rejects booleans, NaN/infinity, exponents, whitespace,
junk suffixes, and control characters. It must remain finite through OAI's
binary64 `atof` parse and assignment to the binary32 `noise_power_dB` storage.
This input-representability rule defines no empirical calibration or operating
bound.

## Clean attachment and persistent control

The pinned `channelmod_rfsimu.conf` is a source template whose initial command
is `-10`; it is not the effective UE-N2 runtime configuration. UE-N2 creates
and hashes an effective runtime config or exact override, records effective
argv/config, and verifies `show current` reads
`rfsimu_channel_ue0.noise_power_dB=-50` and `ploss=0` before attachment.
The clean-attach and restore command token is the canonical string `"-50"`;
the parsed `show current` observation remains the numeric value `-50`.

No runtime command is admitted until all current-session gates pass: UE
attached, `oaitun_ue1` present, reachability passed, uplink traffic active,
telemetry recorder ready, exact active channel object validated, and a fresh
current-session PUSCH observation present. Old CSV evidence cannot open the
gate, and tunnel presence alone is insufficient.

UE-N2 uses one persistent Telnet connection for the experimental trace.
Reconnect-per-command is prohibited. A modification response must contain no
`ERROR`, must reach the expected prompt, and must echo matching owner,
path-loss, and noise state. The modify response does not echo the model name or
index; those stay bound to the preceding `show current` result and control
session. Connection loss permanently fails the trace. A new connection is
allowed only for best-effort cleanup to restore `-50`; cleanup success never
turns the failed trace into a pass. Normal shutdown also restores `-50`.

## Control, evidence, and policy-state separation

The following quantities are distinct:

```text
desired_achieved_pusch_snr_db
    -> future calibrated mapping
    -> commanded_noise_power_db
    -> OAI RFsim noise_power_dB actuator
    -> gNB MAC-normalized instantaneous PUSCH SNR
    -> scheduler EMA SNR
    -> selected/final MCS and observed service outcome
```

`desired_achieved_pusch_snr_db` and `commanded_noise_power_db` are experiment
control/evaluation fields. Both are explicitly excluded from policy state.
They are also distinct from fixed OAI `pusch_TargetSNRx10=150`, which is a UE
power-control target.

The gNB PUSCH SNR, scheduler EMA, MCS, grants, queue state, and outcomes are
post-action collector evidence. None becomes a causal UE policy observation
until a measured UE-visible feedback path is implemented and its availability
at the UE is logged. Collector ingest time is not UE policy availability.

A future policy observation is admissible only when all of the following hold:

```text
policy_observation_available_monotonic_ns is measured and non-null
policy_observation_available_monotonic_ns <= decision_cutoff_monotonic_ns
feedback_path_id identifies a measured UE-visible path
source ran_epoch_id/control_session_id matches the current decision context
```

No future trace value, actuator command, realized outcome, collector-only
timestamp, or ground truth may bypass this gate.

## Monotonic 100-ms command schedule

- Trace indices are unique, contiguous, and zero-based.
- Scheduled time is `anchor_monotonic_ns + trace_index * 100000000`.
- Actual completion never shifts the next scheduled time.
- Obsolete commands are never emitted as a catch-up burst.
- Numeric lateness and jitter acceptance remain deferred to UE-N2 evidence.

For every command, UE-N2 logs scheduled, send, and full response/prompt receipt
times in monotonic and UTC wall-clock domains. Send time is the lower bracket
and response receipt is the upper bracket for channelmod handler completion.
The response ACK is not an RF-sample application timestamp. Fields named
`command_applied_at` or equivalent are prohibited.

The first causally later PUSCH event is only a post-command candidate, not
proof that its RF samples used the new channel. UE-N2 estimates and logs first-
effect lag from the observed step response with a method/status; it never
invents a command application instant.

## Raw radio-event envelope

Every retained radio event or explicit missing observation uses the following
envelope before derived joins:

- `ran_epoch_id` and `control_session_id`;
- exact `source_event_id` and monotonic `source_event_index` within the RAN
  epoch;
- full producer timestamp as `source_event_realtime_sec`,
  `source_event_realtime_nsec`, and derived `source_event_timestamp_ns`;
- `rnti`, raw radio `frame` and `slot`, plus `unwrapped_absolute_slot`;
- `collector_ingest_wall_time_ns` and `collector_ingest_monotonic_ns`;
- `raw_event_sha256`; and
- `missing_reason_code`.

OAI T events carry producer `CLOCK_REALTIME`; the tracer CSV `time` is a
formatted source emission time. Neither is collector ingest time, and
collector ingest time is not UE policy availability. Radio frame/slot wraps
every 10.24 seconds, so joins never use frame/slot alone. Decision
`frame/slot` and scheduled `sched_frame/sched_slot` remain distinct.

If an expected PUSCH observation is absent, its numeric value is null and an
explicit missing reason is required. Missing PUSCH is `MISSING_UNRESOLVED`; it
must not be labelled DTX unless a separate source proves DTX. Missing radio,
RLC, or BSR samples are never zero-filled or silently forward-filled.

## Exact OAI telemetry semantics

- `GNB_MAC_PUSCH_POWER_CONTROL.snrx10/10` is MAC power-control-normalized
  instantaneous PUSCH SNR, not raw PHY SNR. Retain `txpower_calc`; the
  CQI-domain value is `(snrx10 + 10*txpower_calc)/10`, quantized at 0.5 dB and
  censored at UL-CQI saturation.
- `GNB_MAC_UL_MCS_DECISION.avg_snr_x10/10` is the scheduler EMA. Preserve
  selected, pre-PHR, post-PHR, and final MCS as separate fields.
- `GNB_MAC_UL` and `NRUE_MAC_DCI_GRANT` are grant evidence, not confirmed
  delivery. UE grant rounds are a retransmission proxy only.
- `NRUE_MAC_RLC_BUFFER_STATUS` and `NRUE_MAC_BSR_STATUS` retain causal queue
  context without zero-fill.
- Current SINR-policy traces do not provide direct UL BLER.
  `GNB_MAC_BLER_MCS_DECISION` is not promoted as direct UL BLER evidence.
  Direct UL BLER is `UNAVAILABLE_UNRESOLVED`, never zero. UE-N2/UE-N3 lower-
  bound acceptance needs a genuine UL CRC/HARQ outcome source or an explicit
  missing-evidence gate.

EMA settling is indexed by accepted PUSCH-observation count and age since the
last command, not assumed to advance once per 100-ms wall-clock step.

## Frozen scheduler and runtime artifacts

- `SCENESENSE_MCS_POLICY=sinr` before gNB startup.
- `SCENESENSE_FORCE_UL_MCS` unset.
- MCS table 0, one UL layer, 106 PRB, numerology 1, band 78, current TDD, and
  fixed `pusch_TargetSNRx10=150`.
- Exact current seals for `nr-softmodem`, `nr-uesoftmodem`,
  `libtelnetsrv.so`, and `librfsimulator.so` are evidence inputs only and must
  be rechecked at UE-N2 preflight.

## Evidence boundary and deferrals

The pinned historical two-UE runtime switch proves only a successful
channelmod read-modify-read mechanism. Its enclosing experiment ended
`FAILED_HOLD` because its measured SNR/MCS did not match the then-registered
rung. It is not single-UE calibration, 100-ms cadence evidence, an attach-safe
bound, or a passing UE-N1 runtime result.

Deferred to UE-N2 and later are the persistent control implementation, actual
socket execution, command timing/jitter, first-effect lag, numeric desired-
SNR-to-command calibration, attach-safe bounds, genuine UL CRC/HARQ binding,
measured UE-visible feedback availability, and all OAI/CARLA/perception/map/
policy experiments.
