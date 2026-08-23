# UE-N3E — provisional lower operational SNR for the degraded/local fallback route

**Workload:** one 2,048-byte application payload UE->DN every 100 ms with a small DN->UE ACK carrying the sequence number (~164 kbps). This is *not* the 1 Mbps workload used by UE-N3/UE-N3A.

**Test type:** runtime sustain. Every run brings the RAN up at the known-good clean condition (commanded -50 dB), attaches the UE, proves the PDU tunnel and ext-DN reachability by ping, and only then applies the candidate. The good condition is restored before the next candidate. Cold attachment was not tested.

**Pass criteria (all four):** >=594/600 delivered and acknowledged; ACK latency p95 <=100 ms; no UE/PDU-session disconnection; no continuous outage >=1 s.

## Tested candidates

| RFsim command | achieved PUSCH SNR p05/med/p95 (dB) | rep | delivered/600 | % | ACK p50/p95/max (ms) | misses >100 ms | longest outage (s) | disconnect | recovered | verdict |
|---|---|---:|---:|---:|---|---:|---:|---|---|---|
| `channelmod modify 2 noise_power_dB -2.0` | 5.0 / 5.0 / 5.5 | 1 | 317 | 52.83 | 48.29 / 82.75 / 94.07 | 0 | 0.856 | no | yes | **FAIL** |
| `channelmod modify 2 noise_power_dB -2.25` | 5.0 / 5.5 / 5.5 | 1 | 600 | 100.00 | 44.52 / 76.68 / 97.63 | 0 | 0.174 | no | yes | **PASS** |
| `channelmod modify 2 noise_power_dB -2.25` | 5.0 / 5.5 / 5.5 | 2 | 600 | 100.00 | 44.91 / 78.77 / 97.78 | 0 | 0.183 | no | yes | **PASS** |
| `channelmod modify 2 noise_power_dB -2.25` | 5.0 / 5.5 / 5.5 | 3 | 600 | 100.00 | 44.23 / 76.51 / 107.73 | 3 | 0.184 | no | yes | **PASS** |

## Fail/pass boundary

- **Pass:** commanded `-2.25` dB -> achieved PUSCH SNR median **5.5 dB**, 3/3 repetitions passed all four criteria.
- **Fail:** commanded `-2.0` dB -> achieved PUSCH SNR median **5.0 dB**.

The boundary is bracketed between achieved 5.0 dB (fail) and 5.5 dB (pass).

## Recommended provisional lower operational SNR

**Achieved PUSCH SNR median 5.5 dB** (commanded RFsim `noise_power_dB -2.25`) for the degraded/local fallback route.

## Limitations

- This is a **runtime sustain** bound for an already-connected UE, not a cold-attach limit and not a universal physical limit. Cold attachment at these conditions is a separate open question (UE-N3B/N3C/N3D).
- The bound is workload-specific: 2 KB per 100 ms with a small ACK. The existing 1 Mbps UE-N3/UE-N3A results remain valid as higher-load evidence and are not superseded.
- Achieved SNR is the gNB `GNB_MAC_PUSCH_POWER_CONTROL` measurement over the exact 60-second window; RFsim `noise_power_dB` is the commanded knob, not the achieved SNR.
- Single UE, 106 PRB, band 78, AWGN RFsim channel, SINR-driven UL MCS, no CARLA load. Multi-UE contention, fading, and mobility are out of scope.
- ACK latency is a UE-side single-clock monotonic round trip (UE app -> DN -> UE app).
- **Latency margin at the floor is thin.** All three passing runs sit at p50 ~44 ms and p95 ~77 ms, but maximum ACK latency is 97.6 / 97.8 / 107.7 ms. Repetition 3 put 3 of 600 messages over 100 ms. The p95 criterion passes comfortably; the per-message worst case does not have headroom, so a hard per-message 100 ms deadline is not guaranteed at 5.5 dB.
- The failing candidate degraded purely by **loss**, not by latency or by disconnection: at 5.0 dB the ACK p95 was still 82.7 ms with no outage >=1 s and no session drop, yet only 52.8% of messages were delivered. Link-quality collapse here shows up as uplink erasure, so a latency-only health check would not detect it.
- The step between the tested rungs is 0.25 dB commanded, which resolved to a 0.5 dB difference in achieved median SNR (5.0 vs 5.5). The true boundary lies somewhere in that interval; it was not resolved more finely.
- Achieved SNR is reported at the measurement resolution of the tracer (0.5 dB steps in `snrx10`-derived medians), so 5.0 and 5.5 dB are adjacent quantized levels.

## Reproduction

```bash
# 1) descent from the first weak candidate (stops at the first failure)
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.ue_n3e_fallback_snr_floor_v1 \
  --config rl_agent/configs/ue_n3e_fallback_snr_floor_v1.json \
  --output-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_01

# 2) 0.25 dB refinement at -2.25, auto-replicated to three runs on a pass
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.ue_n3e_fallback_snr_floor_v1 \
  --config rl_agent/configs/ue_n3e_fallback_snr_floor_v1.json \
  --output-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_02 \
  --start-db -2.25 --stop-db -2.25 --step-db 0.25

# 3) combine both campaigns into the CSV and this report
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  -m rl_agent.ue_n3e_fallback_snr_floor_report \
  --campaign-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_01 \
  --campaign-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1/20260821_live_02 \
  --out-dir rl_agent/experiments/ue_n3e_fallback_snr_floor_v1
```
