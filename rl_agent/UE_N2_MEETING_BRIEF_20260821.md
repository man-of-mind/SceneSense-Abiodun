# UE-N2 meeting brief — bounded OAI uplink calibration smoke

## Main message

The OAI RFsim control path worked reliably at the intended 100-ms cadence and
produced a clear monotone radio response. The value sent to OAI is **noise
power**, not SNR. A less-negative noise command adds more noise, so achieved
PUSCH SNR and MCS decrease.

This is a successful bounded smoke with `PARTIAL_EVIDENCE`, not the final
channel calibration or accepted SNR range.

## Measured response

Single UE, 106 PRB, SINR scheduler, no forced UL MCS, 10-Hz UDP load at about
2 Mbps, and one 3-second plateau per command:

| RFsim noise command (dB) | median instantaneous PUSCH SNR (dB) | median scheduler EMA SNR (dB) | median final MCS |
|---:|---:|---:|---:|
| -10 | 19.5 | 19.5 | 24 |
| -8 | 16.0 | 15.6 | 19 |
| -5 | 10.0 | 9.8 | 12 |
| -4 | 8.5 | 8.3 | 9 |

The four points are monotone. A descriptive line over only this measured range
is:

```text
achieved PUSCH SNR ~= 0.89 - 1.87 * RFsim noise command
```

Its fitted R-squared is 0.999, but four points from one run are not enough to
promote a universal equation. The safe implementation choice is bounded
lookup/interpolation after replication, with no extrapolation beyond measured
attach-safe evidence.

## Timing and transport checks

- 120/120 commands were issued; none was skipped or caught up in a burst.
- Every full Telnet response arrived before the next 100-ms boundary.
- Handler timing bracket: p50 0.123 ms, p95 0.409 ms, p99 0.795 ms, maximum
  1.048 ms.
- Only 1/120 handler brackets exceeded 1 ms. The p95 is therefore not the
  single maximum outlier; 95% of commands completed within 0.409 ms.
- Send-schedule lag: p50 0.074 ms, p95 0.239 ms, maximum 0.499 ms.
- UDP delivery during the retained interval was 141/141 frames.
- All retained scheduler rows used MCS table 0 and `force_ul_mcs=-1`.
- RFsim was restored to clean `-50`; all owned processes, OAI UE tunnels, and
  experiment ports were absent after teardown.

The Telnet bracket is only control overhead. It is not the total UE decision,
uplink, inference, map-install, or ACK latency.

## Why desired SNR and OAI command differ

`desired_achieved_pusch_snr_db` is the experiment's target outcome. OAI's
`commanded_noise_power_db` is an actuator input. RFsim converts that input into
I/Q noise, while receiver processing and uplink power control determine the
instantaneous PUSCH observation. The scheduler then applies a smoothed SNR
estimate before selecting MCS. These values should not be numerically equal.

The observed tail EMA was close to instantaneous SNR because each 3-second
plateau supplied more than the 120-observation settling gate. This does not
show that the EMA follows every 100-ms transition immediately.

## Claim boundary and next step

The stock trace exporter retains raw evidence but exposes local time-of-day,
not the complete source/collector timestamp envelope. Therefore this run does
not claim an exact RF-application time or causal first-effect lag. Direct UL
BLER also remains unresolved and was not filled with zero.

Meeting conclusion: the actuator mechanism and 100-ms control cadence are
validated sufficiently to proceed to UE-N3. UE-N3 should replicate/order-check
the mapping, verify the deployed 24.5-dB MCS-28 boundary, and search for the
attach-safe lower achieved-SNR bound. Final network-profile bounds remain
unfrozen until that evidence is complete.

## Evidence

Successful create-only run:

`rl_agent/experiments/ue_n2_oai_ul_calibration_smoke_v1/20260821_meeting_smoke_04/`

Terminal status: `UE_N2_SMOKE_CAPTURED_PARTIAL_EVIDENCE`

Manifest SHA-256:
`06fff3a60c14cce5ed3a1dc57d2b1bea5af73bb39fb894d674442f24830d68c6`
