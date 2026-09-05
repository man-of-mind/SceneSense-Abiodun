# SplitFusion Phase-14B four-profile replay qualification

- Status: `FOUR_PROFILE_REPLAY_COMPLETE`
- Radio: `OAI_N78_100MHZ_273PRB_4D5U_V1`
- Phase-14A mapping SHA-256: `841ee69e53d7325570652a0f4baa7ae7f554a2204c4e775ce963a064cabd2677`
- Old 40-MHz replay used as evidence: `false`
- Commands use absolute 100-ms deadlines and `SKIP_OBSOLETE_NEVER_BURST`.
- PUSCH windows use collector-ingest monotonic time, begin 15 ms after ACK, and end no later than the interval boundary/next send.
- The 4,200-sample reference is not a production runtime cap; continuation remains on the same RNG/Markov state.
- The 16-cell pilot and 288-cell campaign remain unauthorized.

| Profile | ACKed | Coverage | MAE (dB) | Bias (dB) | P95 abs (dB) | Pass |
|---|---:|---:|---:|---:|---:|---|
| `FAVORABLE_STABLE` | 4178/4200 | 18.48% | 0.322 | 0.101 | 0.720 | False |
| `MID_VARIABLE` | 4180/4200 | 44.62% | 0.253 | 0.028 | 0.631 | False |
| `ADVERSE_STABLE` | 4182/4200 | 68.95% | 0.214 | -0.057 | 0.541 | False |
| `FADE_RECOVERY` | 4173/4200 | 13.76% | 0.268 | -0.001 | 0.719 | False |

- Final `noise_power_dB=-50` read-back: `True`
- Runner-owned cleanup: `True`
