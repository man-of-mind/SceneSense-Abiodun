# Verification: why AWGN vanilla shows higher MCS than good-channel 106PRB vanilla

Question: the AWGN bad-channel vanilla run reports much higher MCS than the official good-channel 106PRB vanilla baseline. This looked suspicious because the good-channel run should not perform worse at the MCS-selection level.

## Short answer

The AWGN vanilla run is valid, but it is not directly comparable to the official good-channel 106PRB vanilla baseline.

The two runs use different RAN regimes:

- Good-channel official baseline: 106PRB config, no AWGN channel model.
- AWGN policy gate: 273PRB RFsim config with AWGN channel model.

The TDD pattern is the same in both configs: 7 DL slots, 2 UL slots, and 4 UL symbols. The difference is not TDD. The difference is grant sizing / PRB regime / RFsim channel behavior / BLER-update sample pattern.

## Evidence

| Metric | Good 106PRB vanilla | AWGN 273PRB vanilla |
|---|---:|---:|
| gNB config | `gnb.sa.band78.fr1.106PRB.usrpb210.conf` | `gnb.sa.band78.fr1.273PRB.scenesense_rfsim.awgn.conf` |
| UE PRB | 106 | 273 |
| gNB PUSCH SNR p50 | 50.5 dB | 20.0 dB |
| UL MCS avg / p50 / p95 | 7.37 / 7 / 13 | 23.16 / 25 / 27 |
| UL scheduled rate | 21.54 Mbps | 17.73 Mbps |
| Grant TBS p50 / p95 | 1,089 B / 3,521 B | 18,447 B / 25,101 B |
| UL grants | 1,035,066 | 25,745 |
| Branch-3 few-sample updates | 71.3% | 41.2% |
| Retx rate | 0.0% | 5.14% |
| RLC mean queue wait | 88.3 ms | 132.3 ms |
| UE PDCP→gNB PDCP p50 / p95 | 87.5 / 133.8 ms | 127.1 / 238.5 ms |

Active-window check:

- Good 106PRB vanilla stays near MCS ~7 even during active data windows.
- AWGN 273PRB vanilla stays near MCS ~23–25 during active data windows.
- Despite this, scheduled rate is similar because 106PRB uses many small low-MCS grants while 273PRB uses fewer large high-MCS grants.

## Interpretation

The high MCS in AWGN vanilla does not mean the AWGN channel is “better” than the good-channel run. The AWGN run has lower SNR and nonzero retransmissions. Instead, the 273PRB/RFsim/AWGN setup changes the scheduler’s grant/sample behavior:

- In good 106PRB vanilla, most BLER updates have too few scheduled samples, so legacy OAI frequently enters branch 3 and decrements/holds low.
- In AWGN 273PRB vanilla, the scheduler often sees enough scheduled samples and retransmission evidence, so MCS can climb high and only backs off slowly by one step on high BLER.

Therefore:

- Do not compare absolute MCS values between official 106PRB good-channel runs and AWGN 273PRB policy-gate runs.
- The AWGN four-way comparison is still useful internally because vanilla, hold-few, uncapped AIMD, and capped AIMD were all tested under the same AWGN 273PRB setup.
- For the official Track 2 decision, capped AIMD must still be gated on the official 106PRB good-channel path.

