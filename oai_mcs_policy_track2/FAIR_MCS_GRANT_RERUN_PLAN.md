# Fair MCS / Grant-Rate Rerun Plan

Purpose: resolve the apparent contradiction where one run showed high MCS but still high uplink latency, while the clear-channel vanilla run showed low MCS and high latency.

The key hypothesis is that MCS alone is not enough. The full relationship is:

```text
effective fresh uplink drain ≈ first-transmission TBS/grant × useful UL grants/s
                           minus retransmission/overhead effects
```

So the rerun must measure MCS, PRB, TBS, grant rate, first-transmission scheduled rate, retransmission airtime, RLC backlog, and frame latency under matched conditions.

## Fairness constraints

Keep fixed across all runs:

- OAI path: default 106PRB / 7DL-2UL TDD.
- CARLA path: same closed-loop Step-1 frontend.
- Model/payload: no-AE, ROI 0.0, per-channel uint8, zstd level 3.
- Radar preprocessing: fast rasterizer.
- Trace profiles: UE `all`, gNB `latency`.
- T-tracer enabled for both UE and gNB.
- Duration: same `FRONT_DURATION_S`.

Only vary:

- Channel condition: clear, mild AWGN, medium AWGN.
- MCS policy: vanilla vs AIMD-cap.

## Recommended first-pass matrix

Run these four first:

| Label | Channel | Policy | Why |
| --- | --- | --- | --- |
| `clear_vanilla` | clear RFsim | OAI legacy/vanilla | Re-establish low-MCS clear-channel baseline using same trace profile as AWGN. |
| `clear_aimd_cap` | clear RFsim | AIMD cap=3 | Confirm good-channel fix under same trace profile. |
| `mild_vanilla` | AWGN -10 dB noise | OAI legacy/vanilla | Re-test the earlier high-MCS/high-latency case fairly. |
| `mild_aimd_cap` | AWGN -10 dB noise | AIMD cap=3 | Check whether AIMD-cap improves mild AWGN when tracing is matched. |

Then add medium only if the first four are clean:

| Label | Channel | Policy | Why |
| --- | --- | --- | --- |
| `medium_vanilla` | AWGN -5 dB noise | OAI legacy/vanilla | Bad-channel reference where MCS should drop. |
| `medium_aimd_cap` | AWGN -5 dB noise | AIMD cap=3 | Check whether AIMD-cap reduces BLER/retransmission without unsafe MCS holding. |

## Prepared run command

Dry-run / preflight only:

```bash
DRY_RUN=1 \
BASE_BATCH_ID=track2_fair_grant_20260801 \
RUNS="clear_vanilla clear_aimd_cap mild_vanilla mild_aimd_cap" \
bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
```

Actual first-pass run:

```bash
BASE_BATCH_ID=track2_fair_grant_20260801 \
FRONT_DURATION_S=30 \
RUNS="clear_vanilla clear_aimd_cap mild_vanilla mild_aimd_cap" \
bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
```

Optional medium extension:

```bash
BASE_BATCH_ID=track2_fair_grant_20260801 \
FRONT_DURATION_S=30 \
RUNS="medium_vanilla medium_aimd_cap" \
bash abiodun/oai_mcs_policy_track2/run_fair_mcs_grant_rerun.sh
```

## Prepared summary command

For the first-pass four:

```bash
python3 abiodun/oai_mcs_policy_track2/summarize_fair_mcs_grant_rerun.py \
  --base-batch track2_fair_grant_20260801
```

For all six:

```bash
python3 abiodun/oai_mcs_policy_track2/summarize_fair_mcs_grant_rerun.py \
  --base-batch track2_fair_grant_20260801 \
  --runs "clear_vanilla clear_aimd_cap mild_vanilla mild_aimd_cap medium_vanilla medium_aimd_cap"
```

## Metrics to inspect before interpreting

From frontend metrics:

- `payload_p50_kib`
- `front_build_p50_ms`
- `uplink_p50_ms`, `uplink_p95_ms`
- `capture_result_p50_ms`, `capture_result_p95_ms`

From UE grant trace:

- `ul_grant_rate_hz`
- `ul_first_tx_grant_rate_hz`
- `ul_scheduled_mbps`
- `ul_first_tx_mbps`
- `ul_retx_mbps`
- `ul_avg_mcs`, `ul_p50_mcs`, `ul_p95_mcs`
- `ul_avg_tbs_bytes`, `ul_p95_tbs_bytes`
- `ul_p50_rb_size`, `ul_p95_rb_size`, `ul_full_prb_grant_pct`
- `ul_retx_rate_pct`

From RLC / BSR / layer analysis:

- RLC LCID4 occupancy over time.
- UE BSR LCG backlog over time.
- RLC SDU drain Mbps.
- RLC mean queueing delay by Little's law.
- PDCP-ingress → gNB-PDCP-deliver latency if the matched timestamp section is available.

## Interpretation guardrails

- Do not explain a 2× latency increase using <10% retransmission alone. That airtime penalty is too small.
- If MCS/TBS is high but latency is high, check useful grants/s and first-transmission Mbps.
- If PRB is already p50/p95 106, do not claim RB starvation.
- If first-transmission Mbps is low while MCS/TBS is high, the likely cause is fewer useful uplink grants/s, not MCS.
- If first-transmission Mbps is high but latency remains high, inspect RLC backlog, frame reassembly, and frontend closed-loop pacing.
- Do not compare clear-channel `UE_profile=latency` runs against AWGN `UE_profile=all` runs as final evidence; this rerun fixes that mismatch.
