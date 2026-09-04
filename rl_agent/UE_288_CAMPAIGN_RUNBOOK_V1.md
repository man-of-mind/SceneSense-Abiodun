# UE 288 campaign runbook v1

This is a focused launch contract for a 72-action by four-network-profile
response surface, `traffic_50_50`, and qualified Route B. The Cartesian design
remains fixed at 288 cells. The former action registry is retained only until
the final SplitFusion noAE/Hybrid-q/AE validations are bound into its
replacement; it must not be used to launch the campaign.

## Current terminal

`RADIO_LOCKED_100MHZ_4D5U_PENDING_SPLITFUSION_BINDINGS_SNR_REQUALIFICATION_AND_16_CELL_PILOT`

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
and 100-ms schedule remain unchanged**. Their RFsim command mapping does not:
the retained `target_to_rfsim_mapping.csv` was measured under the former
40-MHz/106-PRB/7D2U radio. It remains provenance only and must be recalibrated
and replay-qualified under the locked 100-MHz/4D5U profile before any pilot
cell can run. Likewise, the current `default106` launcher is explicitly marked
superseded and the supervisor refuses to launch it.

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

This command parses both YAML files, verifies the locked radio artifact,
frozen input hashes and Route B locks, reproduces all four accepted
4,200-sample trace-prefix hashes, proves the 72 x 4 and 4 x 4 Cartesian
products, and simulates PASSED/FAILED/INTERRUPTED resume behavior. It starts no
external process and writes no experiment data. A pass here confirms contract
integrity while reporting the unresolved real-launch blockers.

```bash
python3 rl_agent/ue_288_campaign_supervisor.py validate \
  --campaign rl_agent/configs/ue_288_campaign_v1.yaml \
  --pilot rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml
```

The adapter-specific dry contract check is:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3 \
  rl_agent/ue_route_b_split_cell_adapter_v1.py --contract-check \
  --campaign rl_agent/configs/ue_288_campaign_v1.yaml \
  --campaign rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml
```

## Required order before the 16-cell pilot

1. Complete the final SplitFusion noAE, Hybrid-q and AE deployment-path
   validations.
2. Rebuild and hash-bind the 72-action registry to those final checkpoints and
   codecs; do not reuse the LR-ASPP/M-prime registry.
3. Bind and qualify a SplitFusion OAI launcher using the exact locked
   100-MHz/273-PRB/4D5U radio configuration.
4. Recalibrate the target-SNR-to-RFsim command mapping under that exact radio,
   then repeat the short four-profile replay qualification. The Gaussian/
   Markov target traces themselves are not regenerated.
5. Re-run offline validation, then run the 16-cell integration pilot. Only a
   16/16 PASSED ledger can unlock the explicit 288-cell launch.

## Future 16-cell pilot command

After all five prerequisites above are satisfied, the final model paths and
hashes must agree in the model config and the rebuilt operational and technical
registries. The launch guard also verifies the qualified 100-MHz launcher hash
and the selected-radio identity of the new RFsim mapping before creating the
campaign output root.

```bash
python3 rl_agent/ue_288_campaign_supervisor.py run \
  --config rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml \
  --output-root rl_agent/experiments/ue_16_cell_integration_pilot_v1/<PILOT_RUN_ID> \
  --model noae=<FINAL_NOAE_PATH>@<FINAL_NOAE_SHA256> \
  --model ae32=<FINAL_AE32_PATH>@<FINAL_AE32_SHA256> \
  --model ae64=<FINAL_AE64_PATH>@<FINAL_AE64_SHA256> \
  --model ae128=<FINAL_AE128_PATH>@<FINAL_AE128_SHA256>
```

The full sweep remains unauthorized until one ledger contains exactly 16
PASSED pilot cells. A later full launch additionally requires both
`--authorize-full-sweep` and `--pilot-ledger PATH`; the supervisor refuses a
full launch without those gates. The W10275 baseline is 678.75 seconds per
cell, or 54.3 hours for 288 cells before OAI attach, cleanup, and retry
overhead.

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
