# OAI uplink per-layer latency investigation

Goal: identify which OAI layer (PHY→MAC→RLC→PDCP→GTP-U) holds the uplink
split-feature frame too long, by timestamping the tensor as it leaves the UE app
layer and breaking the uplink transport latency down per layer, plus per-layer
buffer fill/drain rate. Deployment profile is fixed: **model with zstd, no-AE,
per-channel-u8, ROI 0, 200k radar PPS**. Validate with iperf first, then switch
to the CARLA split-inference pipeline.

## Status (2026-07-23)

Phase 0 through Phase 6 are complete for the current OAI bottleneck pass. The
main result is now localized to the UE RLC queue-wait caused by low adaptive UL
MCS under sparse closed-loop burst cadence. The latest direct trace rules out
the PHR/RB helper as the reducer and points to the BLER/OLLA selector cadence
inside `get_mcs_from_bler()`.

## Toolchain (validated, both halves)

OAI tree: `abiodun/OAI/openairinterface5g` on branch `scenesense-nrue-grant-trace`.
Record/extract via `scripts/ttracer_record_smoke.sh` + `ttracer_extract_csv_smoke.sh`.

- UE profile `queue`: `NRUE_MAC_DCI_GRANT`, `NRUE_MAC_RLC_BUFFER_STATUS`,
  `NRUE_MAC_BSR_STATUS`, `UE_PHY_UL_PAYLOAD_TX_BITS`.
- gNB profile `full`: `GNB_MAC_UL`, `GNB_MAC_LCID_UL`, `ENB_RLC_UL`,
  `ENB_RLC_MAC_UL`, `ENB_PDCP_UL`, `GNB_PHY_UL_PAYLOAD_RX_BITS`,
  `GNB_MAC_PUSCH_POWER_CONTROL` (carries real SNR/PHR/RSSI).

Pre-existing UE MAC instrumentation (uncommitted, validated, builds clean):
`NRUE_MAC_RLC_BUFFER_STATUS` in `nr_ue_scheduler.c:1454`, `NRUE_MAC_BSR_STATUS`
in `nr_ue_scheduler.c:2168`, plus `sdu_length_total` field in `NR_UE_MAC_CE_INFO`.

### GOVERNING CONSTRAINT (learned the hard way)

The T-tracer byte-compares `T_messages.txt` against each softmodem's compiled-in
copy (`tracer/configuration.c:verify_config`) and `abort()`s on any diff. So:

**Any edit to `T_messages.txt` requires rebuilding BOTH `nr-softmodem` and
`nr-uesoftmodem` and tracing with that exact DB.** A stale gNB binary (May 28)
vs a Jul-20 DB is exactly why gNB recording SIGABRT'd until we rebuilt.
Rebuild: `ninja nr-softmodem nr-uesoftmodem` in `cmake_targets/ran_build/build`.

## Phase-1 findings (iperf UDP uplink 17.8 Mbps, 273PRB, RFsim ideal channel)

Analyzer: `analyze_uplink_layer_latency.py --run-group <group>`.
Run group `validate_20260722_175810_iperf_ul`.

1. **Fixed K2 = 6 slots = 3.0 ms** DCI→PUSCH grant-to-transmit delay on every grant.
2. **UE RLC data-bearer queue is small on smooth traffic** (LCID 4 p50 2.9 KB →
   ~1.6 ms Little's-law residency). SRBs (LCID 1/2) idle. The queue will spike
   under CARLA's ~1 MB bursts — this run validates the *method*, not deployment magnitude.
3. **Scheduler over-grants: ~45% of uplink airtime is padding.** gNB decoded
   83.2 MB at PHY/MAC but only 45.8 MB reached RLC/PDCP. 14% of grants were
   padding-only (issued with no data queued), contributing 61% of the wasted
   airtime. **Effective goodput ≈ 55% of the scheduled ~17–18 Mbps.** This is a
   candidate root-cause contributor: a 1.05 MB frame effectively transfers at
   ~half the headline rate. MUST be re-measured under CARLA's bursty load.
4. **gNB PUSCH SNR is real** (~50 dB) but RFsim-ideal/flat (3.5 dB spread).
   Channel-quality metrics recoverable from the gNB side only; UE-side
   RSRP/CQI are sentinels. Keep ideal channel for now.

Caveats: iperf is saturated CBR, not CARLA's 1 MB bursts; RFsim ideal channel;
state-snapshot method gives per-layer *residency*, not true per-packet transit.

## Phase-1b findings (CARLA split-inference, 273PRB, 10 FPS, run group `..._layerbaseline_20260722_183259`)

The CARLA baseline OVERTURNS two iperf conclusions — validating on both mattered.

| Metric | iperf (smooth CBR) | CARLA (1 MB bursts) |
|---|---|---|
| UE RLC data-bearer occupancy | max 28 KB | **max 1084.8 KB (~1 whole feature frame)** |
| MAC airtime overhead | 45% (over-grant) | **2% (98% goodput)** |
| grant MCS p50 | 28 | **4** |
| grant TBS p50 / PRB p50 | 4992 B / 95 | 1697 B / 273 |

1. **The bottleneck layer is the UE RLC buffer.** A full ~1 MB feature frame queues
   in RLC (peak occupancy 1084.8 KB == the feature payload). Mean Little's-law RLC
   residency = **~100 ms**, consistent with the observed ~150 ms uplink handling.
2. **Root driver = low MCS under bursty load.** The gNB pins UL MCS ~4 on full 273
   PRB under CARLA, vs MCS 28 on the SAME RFsim ideal channel under iperf. Low MCS
   => ~12 Mbps drain => a 1 MB frame takes ~100 ms to clear RLC. Effective UL
   throughput ~1.45 feature-frames/s (vs 10 FPS target), ~689 ms/frame end-to-end.
   This is NOT channel-driven (ideal channel) — it looks like gNB UL link
   adaptation / OLLA not ramping under the bursty pattern. **Top lead to chase.**
3. **The iperf "45% airtime waste" does NOT hold under CARLA** (98% goodput, grants
   100% filled). That over-grant/padding effect was a smooth-CBR artifact — do not
   report it as a deployment finding.
4. K2 = 3 ms fixed in both. Tail Little's-law delay (p95 644 ms) is NOT valid under
   bursts (avg drain includes idle gaps); use the mean (~100 ms) + peak occupancy.

Open mechanism question for phase 2 + gNB UL scheduler review: why MCS 4 (not ~28)
under bursty CARLA load on an ideal channel?

## Phase-2 IMPLEMENTED + iperf-validated (2026-07-22)

Four new T events added (each carries CLOCK_MONOTONIC sec+nsec + ids + sdu_bytes).
Build clean (`ninja nr-softmodem nr-uesoftmodem`, exit 0, no warnings on edited files).

| Event | Site | Role |
|---|---|---|
| `NR_PDCP_TX_SDU` | nr_pdcp_oai_api.c `nr_pdcp_data_req_drb` | UE UL PDCP ingress = t0 |
| `NR_RLC_TX_SDU` | nr_rlc_oai_api.c `nr_rlc_data_req` (!enb_flag) | PDCP->RLC handoff |
| `GNB_MAC_RX_SDU` | gNB_scheduler_ulsch.c `_nr_rx_sdu` | gNB MAC UL reassembly in |
| `GNB_PDCP_RX_DELIVER` | nr_pdcp_oai_api.c `deliver_sdu_drb` | gNB PDCP->SDAP egress = t_final |

Key enabler: RFsim runs UE+gNB on ONE host, so their CLOCK_MONOTONIC values are
directly comparable — no clock sync. DRB SDUs are FIFO with matched counts, so
FIFO index correlation gives true per-packet transit.

iperf validation (17 Mbps, all 4 events populate; 30,516 matched SDUs):
- **UE PDCP-ingress -> gNB PDCP-deliver (whole RAN UL transit): mean 4.6 ms, p50 3.1,
  p95 8.4, max 83.6 ms** (small, as expected for smooth small-packet traffic).
- UE PDCP->RLC handoff: 0.003 ms (negligible).

Record with T binaries directly via `validate_layer_events_iperf.sh` (UE port 2023:
NR_PDCP_TX_SDU/NR_RLC_TX_SDU; gNB port 2021: GNB_MAC_RX_SDU/GNB_PDCP_RX_DELIVER).
Analyzer Section G computes per-packet transit when these CSVs are present.

NEXT: run under CARLA (record queue+full+the 4 new events, TTRACER_DURATION_S>=1200
to cover the full closed-loop run) to get the deployment per-frame per-layer transit
and confirm the RLC-queue-wait dominance under 1 MB bursts.

## Phase-2b DEFINITIVE result: instrumented CARLA run (`..._layerinstr_20260722_191024`)

Full closed-loop CARLA run, latency profiles, 1200s window. 918,409 matched SDUs.
Two independent methods AGREE, which is the headline confidence result:

| Method | UE->gNB uplink latency |
|---|---|
| Per-packet timestamps (PDCP-ingress -> gNB-PDCP-deliver) | mean 105.2 ms, p50 112, p95 163, p99 173, max 225 ms |
| Little's law (mean RLC occupancy / drain) | 103.3 ms |

Per-layer localization (answers "which layer holds the frame too long"):
- **UE PDCP->RLC handoff = 0.097 ms** => PDCP is NOT the bottleneck.
- Therefore ~all of the ~105 ms is RLC-onward: **UE RLC queue-wait for UL grants.**
- gNB airtime 98% efficient, K2 fixed 3 ms, gNB reassembly small => not the bottleneck.
- RLC peak occupancy 1105.9 KB = a whole ~1 MB feature frame sits in RLC.
- Drain ~10.9 Mbps at MCS 4 => ~1 MB / 11 Mbps ~ 100 ms to clear. Matches.

CONCLUSION: the uplink split-feature bottleneck is the **UE RLC buffer queue-wait**,
because a full ~1 MB frame drains at only ~11 Mbps due to **low UL MCS (~4) under
bursty load** (vs MCS 28 achievable on the same ideal channel with iperf). The fix
lever is the gNB UL link-adaptation / MCS selection, NOT PDCP/airtime/gNB compute.

The per-packet method also gives trustworthy tail (p95 163 ms) where Little's law
overstated it (711 ms) — the reason the phase-2 timestamps were worth building.

## Phase-3 diagnostic fixed-MCS experiment (2026-07-22; mechanism superseded by Phase-5)

### Earlier mechanism hypothesis — superseded
- The fixed-MCS experiment below remains valid as a diagnostic: forcing MCS28
  collapses the RLC queue and proves UL MCS is the key latency lever.
- However, the initial explanation that `nr_ue_max_mcs_min_rb()` directly reduces
  a high selected MCS to QPSK is **not supported** by the later Phase-5 pre/post
  trace. In the observed-CARLA-cadence run, selected/pre/post/final MCS were all
  already low (`4/8` p50/p95), and the PHR helper reduced MCS in `0` rows.
- Current best lead: the low MCS is upstream in the BLER/OLLA MCS-selection path
  under sparse closed-loop burst cadence, not in the PHR helper itself.

### Fix experiment: force fixed UL MCS
Env-guarded override `SCENESENSE_FORCE_UL_MCS` added at `gNB_scheduler_ulsch.c`
(after the PHR reduction; default behavior unchanged when unset). Run via
`oai_layer_latency/run_forced_mcs_carla.sh <mcs>` (starts gNB with
`sudo env SCENESENSE_FORCE_UL_MCS=<mcs>`, latency profiles, plot-safe).

| Metric | default (QPSK) | forced MCS 28 (64QAM) | change |
|---|---|---|---|
| End-to-end RTT (front->result) p50 | 186 ms | 48 ms | **3.9x lower** |
| per-packet RAN transit p50 / p95 | 112 / 163 ms | 17 / 26 ms | **~6x lower** |
| RLC mean queueing delay | 103 ms | 13 ms | 7.8x lower |
| RLC occupancy mean / p95 | 137 / 943 KB | 19 / ~0 KB | queue no longer builds |
| grant MCS | ~4-8 (QPSK) | 28 (64QAM) | forced |
| delivery | 76.5% | 73.8% | ~same |

CONCLUSION: the QPSK/low-MCS cap is THE immediate uplink bottleneck. Raising UL MCS
to 64QAM cuts the deployed uplink round trip 186->48 ms. Plot:
`plots/uplink_mcs_bottleneck_summary.png`.

Caveats: RFsim ideal channel (MCS 28 always decodes; on a real fading channel a fixed
high MCS would cause BLER — the proper fix is correcting the adaptive UL MCS / OLLA
behavior, not a blanket fixed MCS). Effective frame *throughput* improved only
~10% (closed-loop frame period is dominated by non-transport stages); the win is
per-frame *latency* (RTT 3.9x).

### Follow-ups
- Instrument the BLER/OLLA selector (`get_mcs_from_bler`) — old/new MCS, BLER
  window, scheduled-count window, retransmission count, and no-activity branch.
- Try capping via config (`ul_bler.max_mcs`/min) vs the env override.
- Re-test with an RFsim channel model so MCS reflects a real channel.

## Phase-4 complementary experiments (2026-07-23)

See `COMPLEMENTARY_EXPERIMENTS_20260723.md`.

Key additions:

- no-AE `uint4` / ROI `0.0` / zstd live OAI 273PRB adaptive run:
  payload p50 `394.6 KB`, delivery `99.8%`, app RTT p50 `112.7 ms`, RAN UL
  p50 `61.7 ms`, RLC mean queue `54.4 ms`;
- MCS is still low under adaptive OAI (`2/5` p50/p95), so uint4 is payload
  relief, not a link-adaptation fix;
- AE-128 / `uint6` / ROI `0.5` / zstd live OAI default-106PRB adaptive run:
  payload p50 `152.7 KB`, delivery `99.8%`, app RTT p50 `64.2 ms`, RAN UL
  p50 `33.6 ms`, RLC mean queue `29.3 ms`;
- the only missed AE-128 frames were the first two startup frames; after warmup,
  delivery was `1298/1298`, so reduced payload removed the steady-state loss
  pattern seen with the ~1 MB no-AE payload;
- fixed MCS28 rerun on 273PRB gives app RTT p50 `47.0 ms`, RAN UL p50
  `16.8 ms`, RLC queue `12.7 ms`;
- fixed MCS28 on 106PRB 4DL/5UL gives app RTT p50 `47.2 ms`, RAN UL p50
  `14.7 ms`, RLC queue `5.9 ms`;
- gNB PUSCH SNR remains flat around `50.5 dB` in RFsim, so the low adaptive MCS
  is not channel-quality driven in this setup.

New presentation plots:

- `plots/complementary_latency_summary.pdf`
- `plots/complementary_mcs_prb_summary.pdf`
- `plots/complementary_rlc_buffer_timeseries.pdf`
- `plots/complementary_gnb_snr_timeseries.pdf`

## Phase-4: hard RLC-wait split + advisor-hypothesis check (2026-07-23)

### Advisor hypothesis (traffic-class -> low PER threshold -> low MCS): not supported by current evidence
- iperf, CARLA, and the synthetic split-feature burst carry data on the same
  default data bearer: QFI 1 / 5QI 9 / DRB 1 / LCID 4 / LCG 1 (checked in RAN
  logs, BSR, and RLC buffer-status traces). Current evidence does not show a
  special split-inference traffic class.
- OAI UL MCS uses a single GLOBAL `nr_mac->ul_bler` (no per-5QI/per-LC PER target);
  comment at gNB_scheduler_primitives.c:3077 "do not limit MCS for individual UEs".
- The SNR-PER-MCS table (`SINRx10_MCS_mapping`, gNB_scheduler_primitives.c:230,
  target BLER 1e-3): MCS28 needs >=24.5 dB, MCS16 ~14.6 dB. Measured SNR ~50 dB,
  PC target 15 dB => table yields MCS 16-28, never 8. Zero HARQ retx.
- => traffic classification is not supported as the explanation in this setup.
  The Phase-5 direct MCS trace also shows the PHR helper is not reducing MCS;
  the stronger current lead is the BLER/OLLA selector's behavior under sparse
  closed-loop bursts.

### Hard per-layer split (NR_RLC_TX_DEQUEUE added at nr_mac_rlc_data_req)
Robust measures (Little's law RLC residency + FIFO-matched Section-G transit):
- **RLC queue-wait ~100 ms = ~95% of the ~105 ms uplink transit.**
- Remainder (air K2 3 ms + gNB PHY/MAC/reassembly/PDCP) ~5 ms. PDCP handoff 0.1 ms.
- So it is the RLC *queue-wait* (holding), not RLC/MAC/PHY *processing*, that adds
  the latency — the 1 MB frame waits for grants that drain it at the QPSK rate.
- NOTE: the per-byte cumulative ingress->dequeue method is confounded (RLC-header
  byte inflation ~0.6% skews cumulative alignment by seconds at ~1.4 MB/s); use
  Little's law + FIFO-matched transit instead. Dequeue event still confirms
  continuous drain (1402 MB over 1041 s).

### Drain-rate visualization: plots/uplink_drain_rate.png
- RLC occupancy decay: 1 MB frame drains over ~200 ms at QPSK vs near-instant at 64QAM.
- Spectral-efficiency ceiling @273 PRB: QPSK ~65, 16QAM ~172, 64QAM ~371 Mbps.

### RLC buffer size (asked): RLC_TX_MAXSIZE = 10 MB/DRB (common/platform_constants.h:60)
Peak used ~1.1 MB (11%), zero buffer-full drops. Increasing it will NOT lower
latency (bufferbloat): the buffer already holds the whole frame; the drain rate
(MCS) is the constraint. A bigger buffer would only let frames get staler.

## Phase-5 advisor follow-up experiments (2026-07-23)

See `ADVISOR_FOLLOWUP_EXPERIMENTS_20260723.md`.

Three checks were completed sequentially:

1. **CARLA-shaped UDP burst:** the low-MCS/RLC-wait behavior is reproduced
   without CARLA/model compute when the synthetic stream uses the observed
   closed-loop cadence (~1.2 FPS, ~1.08 MB bursts). Open-loop 10 FPS with the
   same burst size ramps to high MCS (`23/25` p50/p95), so the issue is cadence
   sensitive, not just “large payload”.
2. **Pre/post scheduler MCS trace:** new `GNB_MAC_UL_MCS_DECISION` event shows
   selected/pre-PHR/post-PHR/final MCS are identical (`4/8` p50/p95); the PHR
   helper reduced MCS in `0` rows. The low MCS is selected upstream.
3. **QFI/5QI/DRB mapping:** runtime logs and traces show DNN `oai` uses QFI 1,
   5QI 9, DRB 1, LCID 4, LCG 1. SMF QER is disabled. Current evidence does not
   support a special split-inference traffic-class/PER-threshold explanation.

## Phase-6 direct BLER/OLLA trace (2026-07-23)

See `BLER_OLLA_TRACE_RESULTS_20260723.md`.

Final follow-up instrumented `get_mcs_from_bler()` directly with
`GNB_MAC_BLER_MCS_DECISION` and compared two synthetic CARLA-shaped streams on
the same 273PRB RFsim path:

| Run | MCS p50 / p95 | `num_sched` p50 | Increase branch | Few-samples branch | Interpretation |
|---|---:|---:|---:|---:|---|
| observed pace ~1.2 FPS | 4 / 8 | 1.0 | 21.4% | 78.6% | selector repeatedly backs off between sparse bursts |
| open-loop 10 FPS | 23 / 25 | 3.5 | 50.0% | 50.0% | enough dense scheduling samples to keep MCS high |

This closes the current mechanism story:

- low MCS is selected inside the BLER/OLLA MCS selector, upstream of the PHR/RB
  helper;
- no high-BLER decrease branch was observed in either run (`0%`), consistent
  with the ideal RFsim channel and high gNB PUSCH SNR;
- sparse closed-loop burst cadence causes too many `num_sched <= 3` windows, so
  the selector walks MCS down between bursts;
- open-loop 10 FPS keeps enough scheduled samples active for the low-BLER
  increase branch to ratchet MCS into the 20s.

Presentation-safe plots:

- `plots/bler_olla_branch_comparison.pdf`
- `plots/bler_olla_mcs_timeseries.pdf`
- `plots/bler_olla_num_sched_timeseries.pdf`

Net conclusion: the deployable mitigation remains payload reduction / feature
compression; fixed MCS28 is only a diagnostic proof of the spectral-efficiency
lever. A scheduler-code fix would need to adjust OAI's BLER/OLLA behavior for
large sparse low-latency bursts, then retest under realistic Sionna channel
variation.

## Phase-2 design (reference — insertion points)

Strategy: queueing/state fidelity first. Add a monotonic `clock_gettime(
CLOCK_MONOTONIC)` nanosecond field to boundary events (the tracer's own receive
time has IPC jitter), bracketing the RAN transit at PDCP ingress (t0) and GTP-U
egress (t_final), reusing existing per-layer occupancy for residency.

UE transmit half:
- **App→stack ingress (t0):** `nr_pdcp_data_req_drb()` — `nr_pdcp_oai_api.c:896`.
  New event `NRUE_PDCP_TX_SDU`: monotonic ts, drb/lcid, sdu_bytes, pdcp queue depth.
- **PDCP→RLC handoff:** `nr_rlc_data_req()` — `nr_rlc_oai_api.c:293`.
  New event `NRUE_RLC_TX_SDU`: monotonic ts, lcid, sdu_bytes.
- **MAC wait + grant:** already covered by `NRUE_MAC_BSR_STATUS` +
  `NRUE_MAC_DCI_GRANT` + `NRUE_MAC_RLC_BUFFER_STATUS`; add a monotonic ts field.
- **PHY TX:** `UE_PHY_UL_PAYLOAD_TX_BITS` (exists).

gNB receive half:
- **PHY RX:** `GNB_PHY_UL_PAYLOAD_RX_BITS` (exists, has ts).
- **MAC RX / reassembly:** `_nr_rx_sdu()` — `gNB_scheduler_ulsch.c:894`.
  New event `GNB_MAC_RX_SDU`: monotonic ts, rnti, lcid, sdu_bytes.
- **PDCP RX:** `ENB_PDCP_UL` (exists, has ts).
- **GTP-U egress (t_final):** PDCP UL deliver → GTP-U toward UPF (locate exact
  `gtpv1u_*` tx call). New event `GNB_GTPU_TX_UL`: monotonic ts, teid, bytes.

Correlation: no single packet ID spans all layers in OAI. Use (a) per-layer
residency via occupancy/drain (Little's law), and (b) PDCP-ingress↔GTP-egress
bracketing over matched byte volume for whole-RAN transit. Per-packet SN
threading (PDCP SN/RLC SN) is a later higher-fidelity option if needed.

Validation loop for every edit: edit T_messages.txt + source → `ninja
nr-softmodem nr-uesoftmodem` (must build clean) → restart stack → iperf record →
confirm new events extract → only then CARLA.
