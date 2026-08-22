# UE-N3 OAI uplink boundary-calibration plan

**Status:** `UE_N3_PLAN_FROZEN_REVIEW_REQUIRED`

The desired achieved-PUSCH-SNR targets are frozen, but every non-clean RFsim command remains unset.
No OAI, CARLA, socket, traffic, or policy process was executed.

## Screening targets

| Desired achieved PUSCH SNR | RFsim noise command | State |
|---:|---:|---|
| 6.0 dB | unset | blocked pending calibration |
| 4.0 dB | unset | blocked pending calibration |
| 3.0 dB | unset | blocked pending calibration |
| 2.0 dB | unset | blocked pending calibration |

## Required next work

1. Exercise the offline-tested SSBURST-aware receiver once in the live OAI namespace.
2. Review the packet-loss and goodput gates; the no-one-second-interarrival-gap rule is frozen.
3. Calibrate RFsim commands without relabelling commands as achieved SNR.
4. Authorize and execute N3A one rung at a time with clean restoration.
5. Refine and replicate the bracket before N3B cold-attach confirmation.
