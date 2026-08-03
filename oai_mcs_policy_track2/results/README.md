# Track 2 results index

## Official 106PRB results

Use these for the historical diagnosis story:

- `AWGN106_POLICY_GATE_CHECK.md`
- `AWGN106_LADDER_SETUP.md`
- `awgn106_policy_summary.md`
- `CURRENT_VANILLA_VS_HOLDFEW_GATE_CHECK.md`
- `P2_HOLDFEW_GATE_CHECK.md`
- `P3_AIMD_GATE_CHECK.md`
- `track2_policy_summary_current_vanilla_vs_holdfew.md`
- `track2_policy_summary_p0_p2_p3_aimd.md`
- `fair_mcs_grant_rerun_track2_fair_grant_20260801.md`
- `noncarla_vanilla_awgn106_noncarla_awgn_vanilla_20260803.md`

The official policy comparison must keep the OAI configuration fixed at default 106PRB unless a separate bandwidth study is explicitly started.

## Current candidate fix: SINR-driven UL MCS

The current candidate is `SCENESENSE_MCS_POLICY=sinr`, implemented in `gNB_scheduler_ulsch.c`.

It selects new-data UL MCS from OAI's existing SINR lookup table using the tracked PUSCH SNR, while keeping HARQ enabled. This is intended to avoid the clean-channel sparse-window artifact where vanilla OAI decrements MCS on CARLA burst gaps despite 0 BLER and ~50 dB SNR.

Run the clear-channel gate first:

```bash
BASE_BATCH_ID=track2_sinr_clear_20260803 \
FRONT_DURATION_S=130 \
RUNS="clear_vanilla clear_sinr" \
bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
```

Then, if clear behaves correctly, run the AWGN ladder:

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

## Legacy/HOLD/AIMD bad-channel command

The official bad-channel gate was generated with:

```bash
BASE_BATCH_ID=track2_awgn106_<date> \
FRONT_DURATION_S=30 \
POLICIES="vanilla hold aimd aimd_cap" \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_policies.sh
```

This keeps the default 106PRB setup and only enables AWGN/channelmod plus the selected MCS policy.

For the legacy/HOLD/AIMD AWGN ladder, use the attach-gated profiles documented in `AWGN106_LADDER_SETUP.md`:

```bash
BASE_BATCH_ID=track2_awgn_ladder_<date> \
PROFILES="mild medium strong" \
POLICIES="vanilla hold aimd_cap" \
FRONT_DURATION_S=30 \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_ladder.sh
```

## Diagnostic-only results

`diagnostic_273prb_awgn/` contains earlier AWGN policy checks run on 273PRB. Keep them as implementation/debug evidence only. Do not compare their absolute MCS values against the official 106PRB clear-channel baseline.
