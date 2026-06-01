# SceneSense OAI Radio Metrics Notes

This note keeps us honest about what the RL agent can safely consume as
network state.

## SCAN-AI Lesson

SCAN-AI uses a cross-layer state vector:

- Scene complexity: SI/TI-style visual complexity.
- Vehicle state: speed, acceleration, turning.
- Network state: available bandwidth or SNR.

The important lesson is not that the network metric must come directly from the
UE process. The lesson is that the policy needs a timely feasibility signal that
correlates with whether the uplink can carry the next payload without causing
loss/queueing. For SceneSense, that signal can be a local UE observation, a
server/gNB observation returned to the UE, or a fused controller state.

## Current Trust Level

Good enough to use now:

- Application metrics: payload bytes, result timeouts, round-trip latency,
  useful FPS, object/mask counts.
- UE tunnel metrics: TX/RX bitrate, drops/errors, ping RTT/loss.
- UE decoded grant metrics:
  - `NRUE_MAC_DCI_GRANT`
  - MCS, RB allocation, symbol allocation, TBS, HARQ process, NDI, RV, TPC
  - derived scheduled UL/DL Mbps and grant rate after windowing
- gNB T-tracer MAC events:
  - `GNB_MAC_UL`, `GNB_MAC_DL`
  - `GNB_MAC_PUSCH_POWER_CONTROL`
  - `GNB_MAC_PUCCH_POWER_CONTROL`
  - `GNB_MAC_LCID_UL`, `GNB_MAC_LCID_DL`
- gNB/RLC/PDCP T-tracer events:
  - `ENB_RLC_UL`, `ENB_RLC_DL`
  - `ENB_RLC_MAC_UL`, `ENB_RLC_MAC_DL`
  - `ENB_PDCP_UL`
  - `ENB_PDCP_DL`
- gNB stdout MAC summary:
  - BLER
  - HARQ rounds/errors
  - DTX
  - average RSRP/SINR when present
  - MAC and LCID byte totals

Treat as diagnostic only for now:

- `UE_PHY_MEAS.rsrp`
- `UE_PHY_MEAS.w_cqi`
- `UE_PHY_MEAS.snr`

In the current rfsim run, `UE_PHY_MEAS.rsrp` used the sentinel value
`-2147483648`, which means the field is not valid for reporting. The gNB-side
statistics are more coherent and match the two observed RNTIs.

The clean UE-side panel therefore does not report CQI/RSRP/SNR yet. MCS is a
usable link-adaptation proxy because it is the modulation/coding choice the UE
actually received in the decoded grant. If explicit CQI/RSRP/SNR is needed for
paper plots, use validated gNB-side metrics first or add a separate NR UE
measurement event later.

In `exp07_full_logging_validation`, UE decoded UL grants matched OAI's existing
`UE_PHY_UL_PAYLOAD_TX_BITS` trace exactly. UE-vs-gNB MAC/PHY TBS totals also
matched closely after normalizing UE DL grant `tb_size` from bits to bytes.

SDAP/QFI/5QI is not yet a clean structured metric in the current smoke profile.
The local OAI tree exposes SDAP mostly through legacy log groups, so bearer-level
QoS labels should be a later instrumentation task if we need them for policy
state or result tables.

## UE-Side RL Implication

If the RL agent lives on the UE, it does not have to rely only on UE PHY fields.
Reasonable designs:

- Local-only first pass:
  - application payload size trend
  - frame RTT/timeout trend
  - UE tunnel TX/RX/drops
  - decoded UL/DL grants from `NRUE_MAC_DCI_GRANT`
  - last selected compression/action
- Network-assisted pass:
  - gNB/server periodically returns compact per-RNTI state:
    - recent MCS
    - recent PRB allocation
    - recent TBS
    - BLER/HARQ/DTX summary
    - RLC/PDCP or LCID byte rate
- Hierarchical pass:
  - UE makes fast frame-level decisions.
  - server/gNB controller broadcasts slower network constraints or budgets.

This matches the SCAN-AI idea: network state gates application demand. The
difference is that SceneSense may need the gNB/server to provide the cleanest
network-state estimate to each UE.

For runtime UE-side control, `NRUE_MAC_DCI_GRANT` is preferred over a gNB
feedback loop because it avoids extra downlink control overhead and reflects the
grant the UE has already decoded. Use gNB metrics as validation and as the input
to server-side spatial-map orchestration.

## Spatial-Map Sharing Agent

The spatial-map sharing agent should be network-aware because it decides which
UE should receive map updates, how much to send, and when the update is still
fresh enough to matter. Unlike the UE compression agent, this agent lives near
the server/back-half/spatial-map side, so gNB-side state is appropriate:

- Per-target UE downlink feasibility: DL MCS, DL TBS, PRB allocation, BLER/HARQ.
- Per-target freshness pressure: age of the spatial-map object/update.
- Per-target safety value: risk/occlusion/criticality for that UE.
- Payload budget: expected bytes for the candidate update.

This avoids pushing every gNB metric down to every UE. The server-side agent can
subscribe to gNB summaries or parsed T-tracer/stdout metrics locally, then send
only the selected perception update to the target UE.

## Next Validation Step

Run one T-tracer capture with the enhanced profile and matching application
`run_group`, then check:

- RNTI mapping remains stable for UE1/UE2 during the run.
- `GNB_MAC_LCID_UL`, RLC, and `ENB_PDCP_UL` totals track application uplink
  payload trends.
- `NRUE_MAC_DCI_GRANT` produces populated UE-side grant rows after rebuilding
  the UE softmodem with the local trace event.
- gNB stdout BLER/HARQ/SNR agrees with `GNB_MAC_PUSCH_POWER_CONTROL.snrx10`
  trends.
- UE-side `UE_PHY_MEAS` remains diagnostic unless it becomes coherent under a
  non-rfsim or better channel-model setup.
