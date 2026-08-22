# UE-N3 OAI uplink boundary-calibration contract v1

## Objective and claim boundary

UE-N3 determines a defensible lower achieved-PUSCH-SNR boundary for the later
network-profile generator and verifies the current `24.5 dB` scheduler
saturation boundary. It is independent of CARLA, Route B, perception-model
collection, the 72 split actions, and policy training.

This v1 artifact freezes the experiment design only. It does not launch OAI,
open the RFsim control socket, infer unmeasured RFsim commands, promote a
numeric bound, or claim deployment approval.

Three lower-bound quantities remain distinct:

- `L_sustain`: lowest achieved PUSCH SNR at which an already-attached UE
  sustains the fixed uplink probe;
- `L_attach`: lowest achieved PUSCH SNR at which a cold UE attaches, creates its
  PDU session, and sustains the same probe; and
- `L_operational = max(L_sustain, L_attach)`.

The source-code scheduler thresholds `-1.0..24.5 dB` are not attachment limits.

## Target versus actuator command

The screening targets `6`, `4`, `3`, and `2 dB` are desired values of
`GNB_MAC_PUSCH_POWER_CONTROL.snrx10 / 10`. They are not values for the RFsim
`noise_power_dB` actuator.

Every live rung requires a separately reviewed mapping:

```text
desired_achieved_pusch_snr_db -> commanded_noise_power_db
```

The v1 mappings are deliberately `null`. The latest bounded working anchor is
RFsim command `-4`, which produced median achieved PUSCH SNR `8.5 dB` in UE-N2.
Extrapolation below that achieved value is not authorized by this plan.

If service disappears before the achieved target is measured, the outcome is
`DETACHED_BEFORE_TARGET_CONFIRMATION`. It brackets actuator-command space but
must not be reported as failure at an unobserved numerical SNR.

## Staged experiment

### N3A: sustain screen

For each reviewed target, in descending order `6 -> 4 -> 3 -> 2 dB`:

1. attach one UE under clean RFsim command `-50`;
2. verify UE identity, tunnel address, ext-DN reachability, current RNTI, and
   fresh PUSCH traffic;
3. apply the reviewed RFsim command once;
4. settle for at least `10 s`;
5. require median achieved PUSCH SNR within `+/-0.5 dB` of the target;
6. sustain a structured UDP uplink at `1 Mbit/s` for `60 s`;
7. record delivery, one-second gaps, tunnel/RAN liveness, instantaneous and EMA
   SNR, selected/final MCS, grants, and available retransmission evidence;
8. restore `-50`, verify recovery, and tear down to a cold local RAN.

A failed restoration or instrumentation failure is infrastructure `FAILED`.
A cleanly captured low-SNR service failure is valid boundary evidence. Lower
targets stop after the first hard service loss until that boundary is reviewed.

The host does not currently expose an `iperf3` client. The existing deterministic
Python UDP sender and the offline-tested
`rl_agent/ue_n3_structured_udp_receiver.py` substitute use 600 frames at 10 Hz,
12,500 bytes per frame, and one matched `SSBURST` datagram per frame. This is a
nominal 1-Mbit/s application load for 60 seconds. The receiver records sequence,
wall-clock, goodput, duplicate, reordering, loss, and one-second gap evidence.
The older generic UDP sink must not be used to claim matched delivery because it
interprets a different header. One live namespace integration check remains
required before the first radio-boundary rung.

The one-second outage gate is defined by no gap of at least one second between
accepted unique datagrams during the observed stream. Receiver-start-anchored
empty bins remain diagnostic because their first and final bins depend on
external sender alignment. Packet-loss and goodput thresholds are deliberately
unset until reviewed; raw values are always retained.

### Refinement and replication

After an initial pass/fail bracket, add at most two achieved-SNR targets rounded
to `0.5 dB`, stopping when the bracket is at most `1 dB` wide. Then run three
fresh-clean-attachment repetitions of the lowest passing target and its adjacent
failing target. A numerical result is not stable unless the boundary outcomes
are `3/3`; a `2/3` result remains `REVIEW_REQUIRED`.

### N3B: cold-attach confirmation

Only after N3A selects a reviewed RFsim command may N3B materialize a new
single-UE configuration at that command. Cold attachment, PDU-session creation,
and the same 60-second service probe must pass `3/3`. If cold attachment needs a
higher achieved SNR than sustain-after-clean-attach, the higher value defines
`L_operational`.

### Upper scheduler boundary

Separately measure one condition immediately below `24.5 dB`, one at or just
above it, and the clean approximately `50.5 dB` reference. Use scheduler
selected-MCS evidence to verify the MCS-28 lookup boundary, while reporting any
later BLER-driven final-MCS reduction separately. `24.5 dB` is a scheduler
saturation design bound, not a physical RF maximum.

## Decision terminals

- `UE_N3_PLAN_FROZEN_REVIEW_REQUIRED`
- `UE_N3_TARGET_CALIBRATION_UNRESOLVED`
- `UE_N3_BOUND_BRACKETED_REVIEW_REQUIRED`
- `UE_N3_UNSTABLE_BOUND_REVIEW_REQUIRED`
- `UE_N3_LOWER_BOUND_BELOW_PROBE_RANGE`
- `UE_N3_ATTACH_SAFE_SERVICE_BOUND_FROZEN`
- `UE_N3_FAILED_RESTORE`
- `FAILED`

The four network-profile means and variances are regenerated only after
`L_operational` is frozen. Until then, values calculated over `8.5..24.5 dB`
remain explicitly provisional.
