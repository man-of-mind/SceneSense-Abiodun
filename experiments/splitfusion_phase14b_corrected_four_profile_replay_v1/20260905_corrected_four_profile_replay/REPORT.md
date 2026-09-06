# SplitFusion Phase-14B corrected four-profile replay qualification

- Status: `FOUR_PROFILE_REPLAY_COMPLETE`
- Radio: `OAI_N78_100MHZ_273PRB_4D5U_V1`
- Phase-14A mapping SHA-256: `841ee69e53d7325570652a0f4baa7ae7f554a2204c4e775ce963a064cabd2677`
- Original (uncorrected) Phase-14B qualification SHA-256: `2388dcf9b08692ee25b7513722e33a15991e852c4eedfe0115bab84b0478efa1`
- Observation-coverage audit report SHA-256: `f3b2402e707bba8853e5232e59cf61d900195cadb42f0faef08f23a76e70ddac`
- Old 40-MHz replay used as evidence: `false`
- Original uncorrected Phase-14B replay used as evidence: `false` (it remains a failed result, unaltered).
- Commands use absolute 100-ms deadlines and `SKIP_OBSOLETE_NEVER_BURST`.
- Correction 1 (traffic): one fixed 100-Hz/1,200-byte lightweight UDP probe. This is measurement
  instrumentation only -- Phase-14B does not measure SplitFusion application throughput or packet
  delivery, and the probe is never counted as SplitFusion traffic or campaign throughput.
- Correction 2 (window): `command_validity_coverage` uses ACK+15ms through the actual next command's
  send time (or the clean-restoration command's send time for the final command); it is the coverage
  used for corrected mapping acceptance. `nominal_schedule_coverage` reproduces the original ACK+15ms
  through nominal-interval-end formula for direct comparison with the failed original replay and never
  drives acceptance.
- The 4,200-sample reference is not a production runtime cap; continuation remains on the same RNG/Markov state.
- The 16-cell pilot and 288-cell campaign remain unauthorized.

| Profile | ACKed | command_validity_coverage | nominal_schedule_coverage | MAE (dB) | Bias (dB) | P95 abs (dB) | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| `FAVORABLE_STABLE` | 4182/4200 | 99.95% | 64.86% | 0.302 | 0.230 | 0.685 | True |
| `MID_VARIABLE` | 4186/4200 | 99.83% | 73.05% | 0.280 | 0.188 | 0.663 | True |
| `ADVERSE_STABLE` | 4181/4200 | 99.28% | 63.90% | 0.190 | 0.037 | 0.526 | True |
| `FADE_RECOVERY` | 4179/4200 | 99.90% | 64.31% | 0.276 | 0.208 | 0.659 | True |

- Final `noise_power_dB=-50` read-back: `True`
- Runner-owned cleanup: `True`
