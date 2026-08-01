# Track 2 P2 hold-few-samples gate check

Run inspected:

- `downlink_oai_default106_ttracer_fps10_track2_holdfew_20260801_default106_noae`

Frozen comparison baseline:

- `downlink_oai_default106_ttracer_fps10_drivable_fast_timingfix_20260731_default106_noae`

## Verdict

The P2 run is a valid policy-effect signal, but not yet a fully clean final comparison.

What is clean:

- The intended OAI policy flag reached the gNB: `hold_mcs_few_samples=1`.
- This was not a fixed-MCS run: `force_ul_mcs=adaptive` and `force_ul_mcs=-1` in the gNB trace.
- Same CARLA route/GT sequence: 1300 frames, identical 129.9 s simulated time, identical GT positions after frame-index alignment.
- Same payload regime: ~1.05 MiB uplink feature payload, 18 UDP chunks.
- Same frontend build regime: ~56-57 ms p50 feature build.
- Radio behavior changed exactly in the expected direction: MCS rose to 28, grant TBS increased, RLC queueing fell, and successful-frame RTT dropped.

What is suspicious / needs caution:

- Delivery stayed ~92%; the policy improved latency but did not solve missing returned results.
- UE tunnel TX drops were still nonzero, though lower than baseline.
- Live localization sanity MAE was worse in P2 than P0, even on common returned frames. Since MCS policy should not alter model content, do not use this pair to claim an accuracy result.
- The P0 manifest was captured while the staleness scenario file was dirty. For a publication-clean comparison, rerun vanilla adaptive OAI with the current exact code state, then compare current-vanilla vs P2.
- The generated layer-latency report for P2 still contains an old phrase saying "slow QPSK drain"; the numeric data show MCS 28, so that sentence should not be used as-is.

## Key comparison

| Metric | P0 vanilla adaptive | P2 hold few samples |
|---|---:|---:|
| Frames / returned | 1300 / 1194 | 1300 / 1196 |
| Delivery | 91.85% | 92.00% |
| Uplink payload p50 | 1046.0 KiB | 1049.0 KiB |
| Front feature build p50 | 57.0 ms | 56.4 ms |
| Feature uplink p50 | 144.8 ms | 37.5 ms |
| Feature uplink p95 | 157.8 ms | 44.0 ms |
| Edge tail p50 | 7.4 ms | 7.5 ms |
| Downlink p50 | 3.2 ms | 2.3 ms |
| Capture→result p50 | 215.1 ms | 105.2 ms |
| Capture→result p95 | 256.2 ms | 148.7 ms |
| UL scheduled Mbps | 20.6 Mbps | 27.6 Mbps |
| UL MCS avg / p50 / p95 | 7.25 / 7 / 13 | 27.73 / 28 / 28 |
| UL PRB p50 / p95 | 106 / 106 | 106 / 106 |
| UL retx rate | 0.0 | 0.0 |
| RLC queue wait estimate | ~89 ms | ~14 ms |
| UE PDCP→gNB PDCP p50 | 89.0 ms | 16.3 ms |

## Accuracy sanity check

Using `abiodun/staleness/validate_accuracy.py`, score >= 0.2, 5 m gate:

| Scope | P0 loc MAE | P2 loc MAE |
|---|---:|---:|
| All frames | 1.554 m | 1.811 m |
| Common returned frames | 1.545 m | 1.782 m |
| Common returned frames, after startup | 1.505 m | 1.758 m |

Interpretation: do not frame P2 as preserving or improving model accuracy until a matched same-code vanilla baseline is rerun. The main valid conclusion from this run is radio/latency behavior.

## Recommended next action

Before trying P3/P4, rerun vanilla adaptive OAI using the current exact runner and current scenario code:

```bash
BATCH_ID=track2_vanilla_current_20260801_default106_noae \
FRONT_DURATION_S=130 \
RADAR_RASTERIZER=fast \
ENTROPY_CODER=zstd \
QUANTIZATION_MODE=per_channel_uint8 \
ROI_THRESHOLD=0.0 \
CONDITION=oai_default106_ttracer \
HOLD_MCS_FEW_SAMPLES=0 \
bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Then compare:

- `track2_vanilla_current_...`
- `track2_holdfew_20260801_default106_noae`

This gives the cleanest apples-to-apples Track 2 baseline.
