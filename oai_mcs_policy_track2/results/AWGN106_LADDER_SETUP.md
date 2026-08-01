# 106PRB AWGN ladder setup and attach gate

Purpose: test whether stronger RFsim AWGN makes BLER persistent enough that OAI's BLER-history MCS controller naturally lowers MCS, or whether the controller remains weak/slow for split-inference bursts.

## Implemented profiles

| Profile | noise_power_dB | Attach-gate status | Interpretation |
|---|---:|---|---|
| `mild` | -10 | previously used successfully | Current AWGN setting; around 19-20 dB observed gNB PUSCH SNR in earlier 106PRB runs |
| `medium` | -5 | passed attach-only check | First stronger usable channel point |
| `strong` | -4 | passed attach-only check | Highest default stress point currently validated for live CARLA runs |
| `harsh` | 0 | failed attach-only check | Boundary/failure probe; repeated RA/RAR/Msg3 failures, not fair for policy ranking |
| `edge` | 5 | not run | Extreme boundary probe only |

`strong` was first tried at `noise_power_dB=-2`, but that also failed attach. It was revised to `-4`, which attached successfully.

## Attach-only checks

| Check | Result | Notes |
|---|---|---|
| `medium` / `-5` | pass | UE tunnel `oaitun_ue1=10.0.0.2` created |
| `strong` / `-2` | fail | No UE tunnel; same class of attach failure as harsh |
| `strong` / `-4` | pass | UE tunnel `oaitun_ue1=10.0.0.2` created |
| `harsh` / `0` | fail | UE initially syncs intermittently, then repeated RAR/Msg3 failures; no UE tunnel |

## Default live CARLA ladder

Use the attach-safe profiles:

```bash
BASE_BATCH_ID=track2_awgn_ladder_20260801 \
PROFILES="mild medium strong" \
POLICIES="vanilla hold aimd_cap" \
FRONT_DURATION_S=30 \
AIMD_CAP_DROP=3 \
bash abiodun/oai_mcs_policy_track2/run_awgn_106prb_ladder.sh
```

Summarize:

```bash
python3 abiodun/oai_mcs_policy_track2/summarize_awgn_ladder.py \
  --base-batch track2_awgn_ladder_20260801 \
  --profiles "mild medium strong" \
  --policies "vanilla hold aimd_cap"
```

## Hypothesis readout

- If `medium`/`strong` BLER remains only slightly above the 15% threshold, then the channel is still only moderately bad.
- If BLER becomes persistent but vanilla/hold keep high MCS, that supports weak/slow OAI BLER backoff.
- If vanilla/hold naturally reduce MCS under persistent BLER, then the earlier mild AWGN case was simply not strong enough.
- `harsh` and `edge` should be reported only as attach/delivery boundary checks unless their attach behavior is stabilized separately.
