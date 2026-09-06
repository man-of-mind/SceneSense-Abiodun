# UE 288 campaign runbook v1

This is a focused launch contract for a 72-action by four-network-profile
response surface, `traffic_50_50`, and qualified Route B. The Cartesian design
remains fixed at 288 cells. The final 72-action SplitFusion registry, qualified
100-MHz radio path, live dispatcher and prerequisite evidence are now bound.
Execution still requires a separate explicit authorization.

## Current terminal

`SPLITFUSION_288_LIVE_CAMPAIGN_READY_FOR_AUTHORIZATION`

## Locked OAI radio baseline

The nominal single-UE uplink configuration is now
`OAI_N78_100MHZ_273PRB_4D5U_V1`: band n78, 100 MHz, 273 PRBs, numerology 1,
and the 4-downlink-slot/5-uplink-slot TDD pattern, with 5QI 6 unchanged. The
decision is hash-bound in
`rl_agent/configs/oai_radio_baseline_100mhz_4d5u_v1.json`.

The completed 144-cell OAI RFsim sweep measured a median maximum TCP receiver
rate of 389.0 Mbit/s for this configuration, versus 281.8 Mbit/s for 40 MHz
4D/5U. At an offered UDP load of 300 Mbit/s, median loss was 0.235% versus
2.571%; derived delivered goodput was 299.3 versus 292.3 Mbit/s. These are
single-UE host/RFsim measurements, not over-the-air or multi-UE capacity
claims.

The four saved Markov/Gaussian **target-SNR traces, seeds, transition models,
and 100-ms schedule remain unchanged**. The RFsim command mapping has been
recalibrated and replay-qualified under the locked 100-MHz/4D5U profile. The
campaign uses `run_splitfusion_oai_100mhz_4d5u_v1.sh`; the former 40-MHz
mapping and `default106` launchers remain provenance only.

The qualified adapter accepts the supervisor's narrow command interface:

```text
python ADAPTER.py --resolved-config CELL/resolved_config.yaml \
  --attempt-dir CELL --carla-host 127.0.0.1 --carla-port 2000
```

It installs a `collecting_drive` hook around the unchanged qualified density
runner. The hook receives Route B's exact ego and gives the unchanged drive
function a `SamplingWorld`; Route B therefore remains the sole ego and clock
owner. Split processing is asynchronous and bounded. The supervisor—not the
adapter—owns the fresh Epic off-screen CARLA process and the cell terminal.
The primary measurement contract uses a 3.0 m object-match radius, a 40.0 m
GT range gate, and a 12.0 px minimum projected GT area. These values are
stamped into every resolved cell, results summary, and manifest.

## Offline validation

This command verifies the final campaign, live runtime and evidence bindings;
reproduces all four accepted 4,200-sample trace-prefix hashes; and proves the
72 x 4 Cartesian product. It starts no external process and writes no
experiment data.

```bash
/usr/bin/python3 -m rl_agent.splitfusion_288_live_campaign_v1 validate
```

The adapter-specific dry contract check is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  rl_agent/ue_route_b_split_cell_adapter_v1.py --contract-check \
  --campaign rl_agent/configs/ue_288_campaign_v1.yaml \
  --campaign rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml
```

## Completed prerequisites

- Final noAE/AE128/AE64/AE32 UINT8/UINT6/UINT4 validation and the 72-action
  registry are complete.
- The 100-MHz/273-PRB/4D5U OAI launcher, RFsim mapping and four-profile replay
  are qualified.
- The 16-cell live pilot completed 16/16 with fresh CARLA/OAI lifecycles and
  cold teardown.
- The result-path/freshness repair completed its four-cell qualification.
- The preparation path passed two prospective cells at 0.9808 and 0.9813
  coverage, above the 0.95 campaign threshold.

These qualifications do not claim that a split action meets the 100-ms service
target. The 100-ms service target and 500-ms ACK horizon remain separate, and
missed deadlines remain measured campaign outcomes.

## Explicit full-campaign command

Use only the final gate below; do not invoke the historical supervisor directly.
Choose a new create-only output leaf. On an interrupted campaign, use the same
leaf with `--resume`; only hash-verified PASSED cells are skipped.

```bash
/usr/bin/python3 -m rl_agent.splitfusion_288_live_campaign_v1 run \
  --output-root experiments/splitfusion_288_live_campaign_v1/<RUN_ID> \
  --qualification-root experiments/splitfusion_phase15_live_deployment_qualification_v1/20260905_live_qualification_retry12_reclassified_v1 \
  --pilot-ledger experiments/splitfusion_16_cell_live_carla_oai_pilot_v1/20260905_live_carla_oai_pilot_retry4/campaign_ledger.json \
  --execute SPLITFUSION_288_CELL_LIVE_CARLA_OAI_CAMPAIGN \
  --authorize-full-sweep
```

The repository binding does not self-authorize the full sweep. The command
requires the explicit token and `--authorize-full-sweep`, and refuses launch
unless the bound 16-cell ledger and all later qualification evidence revalidate.
The historical W10275 estimate is 54.3 hours for 288 cells before retry
overhead; the fresh-lifecycle pilot suggests the actual total may differ.

## Runtime contracts already implemented

- `ue_target_snr_cell_runtime_v1.py` constructs one deterministic sequence per
  cell, verifies/caches samples 0–4199, replays from zero, and then continues
  the same RNG/Markov state indefinitely. Its 100-ms monotonic scheduler uses
  the accepted interpolation mapping, records `SKIP_OBSOLETE_NEVER_BURST`, and
  verifies the `noise_power_dB=-50` restore in `finally`.
- `ue_map_install_feedback_v1.py` keeps capture production asynchronous,
  records `TIMEOUT_NO_ACK` without resend, retains late ACK diagnostics after
  timeout, marks exactly one terminal feedback record per capture, and has
  explicit `NACK_REJECTED` and identifiable `NACK_REASSEMBLY_TIMEOUT` records.
- `spatial_map_server_moving_ego_uplink_only_baseline.py` emits
  `ACK_INSTALLED` only after the decoded result has been accepted into
  `latest_streams` and a bounded `(stream_id, frame_id)` history under the map
  lock. The adapter reads that exact installed record after ACK and never
  substitutes a newer frame. Schema/decode/install rejection emits
  `NACK_REJECTED`; inference completion is not treated as installation.
- The unchanged certified tail runtime is wrapped only to enqueue its decoded
  segmentation mask after normal map publication into a bounded out-of-band
  evaluation sink. A same-frame semantic-GT camera feeds the existing
  `_segmentation_quality_columns` path; neither mask is placed in the measured
  feature payload or spatial-map packet.
- The campaign supervisor uses create-only attempt directories, skips only a
  cell with hash-verified PASSED evidence, and gives every failed/interrupted
  cell a new attempt directory. It writes exactly one terminal after verified
  CARLA process-group cleanup and stops at the first failed or interrupted
  cell.
