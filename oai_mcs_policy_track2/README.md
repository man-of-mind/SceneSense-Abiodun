# Track 2: OAI UL MCS/link-adaptation policy study

Purpose: test whether OAI's current uplink MCS/link-adaptation behavior is a poor fit for split-inference feature bursts, and compare safer policy variants without changing the CARLA/model workload.

## Frozen baseline

Official current-code baseline run:

`downlink_oai_default106_ttracer_fps10_track2_vanilla_current_20260801_default106_noae`

This is the patched closed-loop OAI 106PRB/default-TDD run with detailed timestamp boundaries and the same current scenario/runner code as P2. Treat it as the vanilla adaptive-MCS reference for Track 2.

Older reproducibility baseline:

`downlink_oai_default106_ttracer_fps10_drivable_fast_timingfix_20260731_default106_noae`

It reproduced the same vanilla behavior, but the current-code rerun above is the cleaner apples-to-apples baseline.

Baseline headline:

| Metric | Value |
|---|---:|
| Frames / returned | 1300 / 1207 |
| Delivery | 92.8% |
| Feature payload p50 | 1046.6 KiB |
| Capture→result p50 / p95 | 209.5 / 252.3 ms |
| Front feature build p50 | 55.7 ms |
| Feature uplink handling p50 | 142.5 ms |
| Edge tail p50 | 7.0 ms |
| Result downlink p50 | 3.1 ms |
| UL scheduled throughput | 21.5 Mbps |
| UL MCS avg / p50 / p95 | 7.37 / 7 / 13 |
| UL PRB p50 / p95 | 106 / 106 |
| UL retx rate | 0.0 |
| gNB PUSCH SNR p50 | 50.5 dB |

Interpretation: the front-side timing is now aligned with uplink-only measurements. The closed-loop penalty is not downlink; it is low uplink spectral efficiency/MCS under return-wait burst pacing.

## Current OAI hooks already present

The OAI source includes SceneSense instrumentation and policy hooks:

- `T_GNB_MAC_BLER_MCS_DECISION` in `gNB_scheduler_primitives.c`
- `T_GNB_MAC_UL_MCS_DECISION` in `gNB_scheduler_ulsch.c`
- `SCENESENSE_FORCE_UL_MCS=<0..28>` for fixed-MCS diagnostics
- `SCENESENSE_HOLD_MCS_FEW_SAMPLES=1` for holding MCS when the update window has too few scheduled samples
- `SCENESENSE_MCS_POLICY=aimd` for the BLER-aware AIMD/TCP-Reno-like diagnostic policy
- `SCENESENSE_AIMD_MAX_DROP=<N>` for capping each AIMD bad-window MCS decrease
- `SCENESENSE_MCS_POLICY=sinr` for the SINR-driven UL MCS policy. This uses `get_mcs_from_SINRx10(pusch_pc.avg_snr)` for new-data MCS selection while keeping HARQ/retransmissions enabled.

The baseline above used vanilla-compatible behavior:

- `SCENESENSE_FORCE_UL_MCS` unset
- `SCENESENSE_MCS_POLICY` unset / `legacy`
- `SCENESENSE_HOLD_MCS_FEW_SAMPLES=0`

## Policy variants to run

Run each policy on the same workload unless explicitly noted:

- no-AE baseline checkpoint
- ROI `0.0`
- `per_channel_uint8`
- `zstd`
- 200k radar PPS
- fast radar rasterizer
- OAI default 106PRB / default 7DL-2UL TDD
- closed-loop CARLA path
- 10 FPS target / 1300 frames
- t-tracer enabled

| ID | Policy | Mechanism | Why run it |
|---|---|---|---|
| P0 | Vanilla adaptive OAI | `MCS_POLICY` unset or `MCS_POLICY=vanilla`, `HOLD_MCS_FEW_SAMPLES=0` | Current-code baseline |
| P1 | Fixed MCS28 | `FORCE_UL_MCS=28` | Upper-bound diagnostic: what if spectral efficiency is high? |
| P2 | Hold MCS on few samples | `HOLD_MCS_FEW_SAMPLES=1` | Tests whether sparse scheduling windows are wrongly causing MCS decay |
| P3 | AIMD/TCP-Reno-like MCS | `MCS_POLICY=aimd` | Hold sparse clean windows, additively increase when clean, multiplicatively back off only on real BLER/retransmission |
| P4 | Capped AIMD | `MCS_POLICY=aimd`, `AIMD_MAX_DROP=3` | Same AIMD logic, but caps one bad-window decrease to avoid overreacting |
| P5 | 106PRB AWGN policy gate | same P0/P2/P3/P4 policies with AWGN channel enabled | Official bad-channel comparison; same 106PRB config as clear-channel runs |
| P6 | 106PRB AWGN ladder | mild/medium/harsh AWGN profiles with selected policies | Tests whether stronger sustained BLER makes OAI lower MCS, or whether backoff remains too weak/slow |
| P7 | SINR-driven MCS | `MCS_POLICY=sinr` | Candidate fix for controlled channel sweeps: drive MCS from measured/injected SNR instead of sparse BLER windows |

Suggested order:

1. P0 current vanilla adaptive, already completed and validated.
2. P2 hold-few-samples, already completed and validated as the high-MCS/good-channel reference.
3. P7 SINR-driven MCS on the clear-channel closed-loop CARLA run. This is the current best candidate because it should map clean 50 dB SNR directly to MCS 28 without being affected by CARLA's burst/sparse-window pacing.
4. P7 SINR-driven MCS on an AWGN ladder (`mild medium strong`) to confirm MCS follows the injected channel quality monotonically while HARQ/retransmission metrics remain available.
5. P3/P4 AIMD only as secondary diagnostics if the SINR policy does not behave as expected or if we need a purely BLER-reactive comparison.
6. P5/P6 legacy/hold/AIMD AWGN comparisons remain useful background, but should not be treated as the main proposed fix after the SINR path is validated.
7. P1 fixed MCS28 only if we still want a pure upper-bound comparator under the same timing.

## AWGN ladder guardrail

The old `noise_power_dB=-10` AWGN run is now the `mild` profile, not the entire bad-channel conclusion.

Default ladder:

```bash
BASE_BATCH_ID=track2_awgn_ladder_20260801 \
PROFILES="mild medium strong" \
POLICIES="vanilla sinr" \
FRONT_DURATION_S=30 \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_ladder.sh
```

Summarize:

```bash
python3 abiodun/oai_mcs_policy_track2/summarize_awgn_ladder.py \
  --base-batch track2_awgn_ladder_20260801 \
  --profiles "mild medium strong" \
  --policies "vanilla sinr"
```

SINR policy guardrail:

- The SINR policy is only valid if `GNB_MAC_UL_MCS_DECISION.avg_snr_x10` moves monotonically with the AWGN profile.
- Expected rough mapping from the current OAI table: clear `~50 dB -> MCS 28`, mild `~19.5 dB -> MCS 24`, medium `~9.8 dB -> MCS 13`.
- Keep HARQ enabled; do not set `ul_harq_round_max=1`, because that would remove retransmission behavior needed for reliability measurements.

Important cleanup note: earlier 273PRB AWGN runs are diagnostic-only and have been moved under `results/diagnostic_273prb_awgn/`. Do not use them as the official bad-channel comparison against the 106PRB clear-channel baseline.

## Guardrails

- Do not compare runs unless the `run.log` clearly records the policy env.
- Rebuild OAI after any source-code policy change.
- Do not rebuild merely for env-only variants if the current binary already contains the hook, but record that no source changed.
- Use a unique `BATCH_ID` for every policy run.
- Confirm CARLA route/settings from the resolved config before reporting.
- Confirm the metrics CSV has detailed timing columns before regenerating final plots.
- Keep official policy comparisons on one OAI configuration. For Track 2, that means 106PRB unless we explicitly start a separate bandwidth study.
