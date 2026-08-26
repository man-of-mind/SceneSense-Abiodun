# UE 288 campaign runbook v1

This is a focused launch contract for the existing 72-action registry, four
fixed target-SNR profiles, `traffic_50_50`, and qualified Route B. It is not a
new experiment framework.

## Current terminal

`IMPLEMENTATION_BLOCKED`

The exact missing runtime component is a qualified Route B split-cell adapter.
The existing Route B collector owns the CARLA synchronous clock, qualified ego
controller, traffic lifecycle, no-hybrid-physics setting, and accepted
stationary-roadblock clearing. The certified split/OAI launcher independently
spawns and controls another ego on a different route. It also has no external
front lifecycle boundary through which a Route B-owned ego can be attached.
Substituting that route would invalidate the campaign; running both owners in
one world would invalidate clock and scene causality.

The adapter must accept the supervisor's narrow command interface:

```text
python ADAPTER.py --resolved-config CELL/resolved_config.yaml \
  --attempt-dir CELL --carla-host 127.0.0.1 --carla-port 2000
```

It must attach the registered split front to the already-created qualified
Route B ego without taking world-clock or traffic ownership, start/reuse the
existing OAI back half, publish decoded results to the existing map endpoint,
surface identifiable reassembly failures, and write the seven configured
outputs. The supervisor—not the adapter—owns the fresh Epic off-screen CARLA
process and the cell terminal.

## Offline validation

This command parses both YAML files, verifies frozen input hashes and Route B
locks, reproduces all four accepted 4,200-sample trace-prefix hashes, proves
the 72 x 4 and 4 x 4 Cartesian products, and simulates PASSED/FAILED/INTERRUPTED
resume behavior. It starts no external process and writes no experiment data.

```bash
python3 rl_agent/ue_288_campaign_supervisor.py validate \
  --campaign rl_agent/configs/ue_288_campaign_v1.yaml \
  --pilot rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml
```

## Future 16-cell pilot command

After the adapter exists and the four final paths/hashes have been entered into
`rl_agent/configs/ue_split_profile_registry_v1.json` under
`factor_contract.models.{noae,ae32,ae64,ae128}`, the 72-row operational and
technical registries must bind those same files and hashes. The launch guard
checks all three before creating the campaign output root.

```bash
python3 rl_agent/ue_288_campaign_supervisor.py run \
  --config rl_agent/configs/ue_16_cell_integration_pilot_v1.yaml \
  --output-root rl_agent/experiments/ue_16_cell_integration_pilot_v1/<PILOT_RUN_ID> \
  --route-b-split-cell-adapter <QUALIFIED_ROUTE_B_SPLIT_CELL_ADAPTER.py> \
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
  records `TIMEOUT_NO_ACK` without resend, retains late ACKs after timeout, and
  has explicit `NACK_REJECTED` and identifiable `NACK_REASSEMBLY_TIMEOUT`
  records.
- `spatial_map_server_moving_ego_uplink_only_baseline.py` emits
  `ACK_INSTALLED` only after the decoded result has been accepted into
  `latest_streams` under the map lock. Schema/decode/install rejection emits
  `NACK_REJECTED`; inference completion is not treated as installation.
- The campaign supervisor uses create-only attempt directories, skips only a
  cell with preserved PASSED evidence, and gives every failed/interrupted cell
  a new attempt directory. It writes exactly one terminal after verified CARLA
  process-group cleanup.
