# Track 2 results index

## Official 106PRB results

Use these for the main Track 2 story:

- `AWGN106_POLICY_GATE_CHECK.md`
- `AWGN106_LADDER_SETUP.md`
- `awgn106_policy_summary.md`
- `CURRENT_VANILLA_VS_HOLDFEW_GATE_CHECK.md`
- `P2_HOLDFEW_GATE_CHECK.md`
- `P3_AIMD_GATE_CHECK.md`
- `track2_policy_summary_current_vanilla_vs_holdfew.md`
- `track2_policy_summary_p0_p2_p3_aimd.md`

The official policy comparison must keep the OAI configuration fixed at default 106PRB unless a separate bandwidth study is explicitly started.

## Official bad-channel command

The official bad-channel gate was generated with:

```bash
BASE_BATCH_ID=track2_awgn106_<date> \
FRONT_DURATION_S=30 \
POLICIES="vanilla hold aimd aimd_cap" \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_policies.sh
```

This keeps the default 106PRB setup and only enables AWGN/channelmod plus the selected MCS policy.

For the AWGN ladder, use the attach-gated profiles documented in `AWGN106_LADDER_SETUP.md`:

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
