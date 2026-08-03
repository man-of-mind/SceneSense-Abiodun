# Track 2 runbook

## Baseline artifact paths

Run group:

`downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae`

Metrics CSV:

`abiodun/downlink_latency_fps/runs/oai_default106_ttracer/fps_10_track2_vanilla_current_20260801_default106_noae/streams/downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae_metrics.csv`

CARLA/OAI summary:

`abiodun/metrics_logs/carla_oai_ttracer/downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae/CARLA10_OAI_TTRACER_SUMMARY.csv`

UE grant summary:

`abiodun/metrics_logs/scenesense_ttracer/downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae/ue/analysis/nrue_grant_summary.csv`

## Rebuild after source policy edits

Any change to `gNB_scheduler_primitives.c`, `gNB_scheduler_ulsch.c`, or `T_messages.txt` requires rebuilding before the next live run:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/OAI/openairinterface5g/cmake_targets/ran_build/build
ninja nr-softmodem nr-uesoftmodem
```

If `ccache` fails with a read-only `/run/user/.../ccache-tmp` directory, rebuild with ccache disabled:

```bash
env CCACHE_DISABLE=1 ninja nr-softmodem nr-uesoftmodem
```

For P3/P4, the expected binary must include `SCENESENSE_MCS_POLICY=aimd` support. For capped AIMD, it must also include `SCENESENSE_AIMD_MAX_DROP`.

For the SINR policy, the expected binary must include `SCENESENSE_MCS_POLICY=sinr` support in `gNB_scheduler_ulsch.c`. This policy keeps HARQ enabled but selects the new-data UL MCS from `get_mcs_from_SINRx10(pusch_pc.avg_snr)`.

The first P3 rebuild completed successfully with:

- command: `env CCACHE_DISABLE=1 ninja nr-softmodem nr-uesoftmodem`
- rebuilt binary: `nr-softmodem`
- timestamp: `2026-07-31 21:12:06 -0400`
- binary string check: `SCENESENSE_MCS_POLICY`, `aimd`, `hold_few`, `vanilla`

The capped-AIMD rebuild completed successfully with:

- command: `env CCACHE_DISABLE=1 ninja nr-softmodem nr-uesoftmodem`
- rebuilt binary: `nr-softmodem`
- timestamp: `2026-07-31 21:59:17 -0400`
- binary string check: `SCENESENSE_AIMD_MAX_DROP`

## Common run template

Run from:

`/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab`

Base command:

```bash
BATCH_ID=<unique_batch_id> \
FRONT_DURATION_S=130 \
RADAR_RASTERIZER=fast \
ENTROPY_CODER=zstd \
QUANTIZATION_MODE=per_channel_uint8 \
ROI_THRESHOLD=0.0 \
CONDITION=oai_default106_ttracer \
bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

## P2: hold MCS on few scheduled samples

This is the first Track 2 policy run because the source hook already exists.

```bash
BATCH_ID=track2_holdfew_20260801_default106_noae \
FRONT_DURATION_S=130 \
RADAR_RASTERIZER=fast \
ENTROPY_CODER=zstd \
QUANTIZATION_MODE=per_channel_uint8 \
ROI_THRESHOLD=0.0 \
CONDITION=oai_default106_ttracer \
HOLD_MCS_FEW_SAMPLES=1 \
bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Expected `run.log` should include:

`hold_mcs_few_samples=1`

If that string is missing, do not report the run as P2.

## P1: fixed MCS28 upper-bound diagnostic

```bash
BATCH_ID=track2_forcemcs28_20260801_default106_noae \
FRONT_DURATION_S=130 \
RADAR_RASTERIZER=fast \
ENTROPY_CODER=zstd \
QUANTIZATION_MODE=per_channel_uint8 \
ROI_THRESHOLD=0.0 \
CONDITION=oai_default106_ttracer \
FORCE_UL_MCS=28 \
bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Expected `run.log` should include:

`force_ul_mcs=28`

## P3: BLER-aware AIMD / TCP-Reno-like policy

This policy is compiled into the gNB MCS selector and selected at runtime with `MCS_POLICY=aimd`.

Behavior:

- If the BLER update window has too few samples and no retransmission, hold MCS.
- If the channel is clean and there are enough samples, increase MCS by one.
- If real BLER/retransmission appears above the upper BLER threshold, multiplicatively back off MCS over the allowed MCS range.

Run only after rebuilding OAI.

```bash
BATCH_ID=track2_aimd_current_20260801_default106_noae \
FRONT_DURATION_S=130 \
RADAR_RASTERIZER=fast \
ENTROPY_CODER=zstd \
QUANTIZATION_MODE=per_channel_uint8 \
ROI_THRESHOLD=0.0 \
CONDITION=oai_default106_ttracer \
MCS_POLICY=aimd \
HOLD_MCS_FEW_SAMPLES=0 \
bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Expected `run.log` should include:

`mcs_policy=aimd`

Expected good-channel behavior:

- MCS should rise toward 28, similar to P2.
- BLER branch 3 may still appear, but clean few-sample windows should not reduce MCS.
- If retransmissions remain zero, feature uplink latency should be close to P2.

## P4: capped AIMD decrease

This keeps the P3 AIMD structure but caps each high-BLER MCS drop. The first diagnostic cap is 3 MCS steps per bad update window.

Run only after rebuilding OAI.

```bash
BATCH_ID=track2_aimd_cap3_20260801_default106_noae \
FRONT_DURATION_S=130 \
RADAR_RASTERIZER=fast \
ENTROPY_CODER=zstd \
QUANTIZATION_MODE=per_channel_uint8 \
ROI_THRESHOLD=0.0 \
CONDITION=oai_default106_ttracer \
MCS_POLICY=aimd \
AIMD_MAX_DROP=3 \
HOLD_MCS_FEW_SAMPLES=0 \
bash abiodun/downlink_latency_fps/run_oai_default106_ttracer_10fps.sh
```

Expected `run.log` should include:

- `mcs_policy=aimd`
- `aimd_max_drop=3`

## P7: SINR-driven MCS policy

This is the current candidate fix for the controlled channel-sweep study.
Unlike hold-few/AIMD, it does not infer channel quality from sparse BLER windows.
It directly maps the gNB's tracked PUSCH SNR to MCS through OAI's existing `get_mcs_from_SINRx10()` table, while leaving HARQ/retransmissions enabled.

Run only after rebuilding OAI.

Clear-channel closed-loop gate:

```bash
BASE_BATCH_ID=track2_sinr_clear_20260803 \
FRONT_DURATION_S=130 \
RUNS="clear_vanilla clear_sinr" \
bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
```

Expected `run.log` for the SINR run should include:

`mcs_policy=sinr`

Expected clear-channel behavior:

- `avg_snr_x10` should remain near the clean-channel value (`~505`, i.e. `~50.5 dB`).
- selected/final UL MCS should stay near 28 instead of falling to ~7.
- feature uplink handling latency should move toward the HOLD/AIMD/fixed-MCS diagnostic range, not the vanilla ~142 ms range.
- retransmission rate should remain near zero.

AWGN SNR-ladder gate:

```bash
BASE_BATCH_ID=track2_sinr_awgn_ladder_20260803 \
PROFILES="mild medium strong" \
POLICIES="vanilla sinr" \
FRONT_DURATION_S=30 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_ladder.sh
```

Summarize:

```bash
python3 abiodun/oai_mcs_policy_track2/summarize_awgn_ladder.py \
  --base-batch track2_sinr_awgn_ladder_20260803 \
  --profiles "mild medium strong" \
  --policies "vanilla sinr"
```

Expected ladder behavior:

- `avg_snr_x10` should move monotonically with AWGN profile.
- SINR policy MCS should follow that SNR movement: mild should be lower than clear, medium lower than mild, and strong lower than medium if attach remains stable.
- HARQ retransmission metrics should still be present because HARQ is not disabled.

## Post-run checks

After each run:

```bash
RUN_GROUP=downlink_oai_default106_ttracer_fps10_<BATCH_ID>
ls abiodun/metrics_logs/carla_oai_ttracer/${RUN_GROUP}/CARLA10_OAI_TTRACER_SUMMARY.csv
ls abiodun/metrics_logs/scenesense_ttracer/${RUN_GROUP}/ue/analysis/nrue_grant_summary.csv
```

Then inspect:

- frames and returned frames
- p50/p95 capture→result
- p50/p95 feature uplink handling
- MCS avg/p50/p95
- scheduled UL Mbps
- retx rate / BLER decision trace
- PRB p50/p95

## Official bad-channel gate: AWGN 106PRB policy comparison

Purpose: exercise the policies under a bad channel while keeping the OAI configuration fixed to the same default 106PRB setup used by the clear-channel baseline.

This is the official fair bad-channel comparison. It changes only the channel condition and MCS policy:

- gNB config family: `gnb.sa.band78.fr1.106PRB.scenesense_rfsim.awgn_<profile>.conf`
- UE config family: `ue.awgn_<profile>.conf`
- channel model family: `channelmod_rfsimu_awgn_<profile>.conf`
- payload: no-AE, ROI 0.0, zstd, per-channel uint8, ~1 MiB
- default short diagnostic: 300 requested frames per policy/profile

AWGN ladder profiles:

| Profile | noise_power_dB | Intended use |
|---|---:|---|
| `mild` | -10 | Current AWGN setting; previously observed around 19-20 dB gNB PUSCH SNR |
| `medium` | -5 | +5 dB noise; tests whether BLER becomes more persistent |
| `strong` | -4 | Highest default stress point after `-2`/`harsh` showed attach instability |
| `harsh` | 0 | Boundary probe; can fail attach through repeated RA/RAR/Msg3 failures |
| `edge` | 5 | Optional extreme boundary probe; not part of default fair ranking |

Run one profile:

```bash
AWGN_PROFILE=mild \
BASE_BATCH_ID=track2_awgn106_mild_20260801 \
FRONT_DURATION_S=30 \
POLICIES="vanilla hold aimd aimd_cap" \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_policies.sh
```

For the SINR candidate, use:

```bash
AWGN_PROFILE=mild \
BASE_BATCH_ID=track2_sinr_awgn_mild_20260803 \
FRONT_DURATION_S=30 \
POLICIES="vanilla sinr" \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_policies.sh
```

Run the default ladder:

```bash
BASE_BATCH_ID=track2_awgn_ladder_20260801 \
PROFILES="mild medium strong" \
POLICIES="vanilla sinr" \
FRONT_DURATION_S=30 \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_ladder.sh
```

Summarize the ladder:

```bash
python3 abiodun/oai_mcs_policy_track2/summarize_awgn_ladder.py \
  --base-batch track2_awgn_ladder_20260801 \
  --profiles "mild medium strong" \
  --policies "vanilla sinr"
```

Expected `run.log` checks:

- all runs: `UE_PRB=106`
- all runs: `rfsim_chanmod=1`
- all runs: `awgn_profile=<profile>`
- all runs: profile-specific gNB/UE configs, for example `awgn_mild`, `awgn_medium`, or `awgn_strong`
- P0 vanilla: `mcs_policy=vanilla`
- P2 hold run: `hold_mcs_few_samples=1`, `mcs_policy=legacy`
- P3 AIMD run: `hold_mcs_few_samples=0`, `mcs_policy=aimd`, `aimd_max_drop=uncapped`
- P4 capped AIMD run: `mcs_policy=aimd`, `aimd_max_drop=3`
- P7 SINR run: `mcs_policy=sinr`

Pass/fail interpretation:

- If mild AWGN barely crosses the 15% BLER threshold, call it moderate; do not claim it is a severe channel.
- If medium/strong AWGN creates sustained BLER but vanilla/hold still keep high MCS, that is evidence of weak/slow BLER backoff.
- If medium/strong AWGN causes vanilla/hold MCS to drop naturally, then the previous mild result was simply not harsh enough.
- Capped AIMD is better than hold/vanilla only if it lowers BLER/retransmission pressure without a large queueing/latency penalty.
- Uncapped AIMD is too aggressive if it lowers BLER but increases RLC queueing substantially.
- If `harsh`/`edge` breaks attach or collapses delivery, treat it as a boundary check, not a fair policy-ranking run.

## Historical diagnostic only: AWGN 273PRB policy checks

Earlier AWGN policy checks used 273PRB because that was the first working AWGN harness. Those runs are now diagnostic-only and have been moved under `results/diagnostic_273prb_awgn/`.

Do not use 273PRB AWGN results as the official bad-channel comparison against the 106PRB clear-channel baseline. They are only evidence that the policy code path can react to BLER/retransmissions.
