# Agent and static split traffic and perception results

Prepared on 13 October 2026 for replotting the agent and two static split configurations. This results folder sits alongside the existing [architecture handoff](../supervisor_architecture_20261008_v1/README.md), which is unchanged.

Start with [the PNG figures](figures/), [the P50 P95 P99 table](data/QUANTILES_P50_P95_P99.csv), and [the per-decision data](data/DECISIONS.csv). The CSVs can be opened in a spreadsheet or loaded into Python, MATLAB or R. The plotting scripts run from this folder without CARLA, OAI, model weights or the original experiment directories.

## Configurations and observation windows

These are three separate, existing 1,000-transmitted-frame sessions from the same OAI campaign, using FAVORABLE_STABLE. They are not matched-scene trials or full-town captures. Each session starts its own elapsed-time axis at its first transmitted frame's capture timestamp. Equal elapsed time across sessions does not identify the same scene.

| CSV arm key | Configuration | Policy decisions | Completed predictions | Prediction ready within 170 ms | Feedback received within 170 ms |
|---|---|---:|---:|---:|---:|
| `agent` | Frozen agent, adaptive mode and q | 493 | 493 | 479 | 390 |
| `mode03` | AE128, UINT8, q = 0 | 500 | 4 | 0 | 0 |
| `mode09` | AE32, UINT8, q = 0 | 500 | 476 | 123 | 8 |

All three sent 1,000 frames, including decision and hold frames. The agent also has fallback frames. Static q = 0 means no spatial dropping; it does not guarantee an exactly constant encoded byte count. Larger q means more dropping.

## Files for making plots

| File | Rows represent | Main fields and use |
|---|---|---|
| [FRAMES.csv](data/FRAMES.csv) | Every sent frame, 3,000 total | `arm`, `frame_id`, `elapsed_s`, `capture_wall_s`, `payload_bytes`, `q`; payload timeline and offered application rate |
| [DECISIONS.csv](data/DECISIONS.csv) | Every policy decision, 1,493 total | Mode, q, payload, SI, P40, prior UE MCS, prediction/feedback latency, outcome and available per-decision quality |
| [QUANTILES_P50_P95_P99.csv](data/QUANTILES_P50_P95_P99.csv) | 75 metric/configuration combinations | P50/P95/P99 with units, population, sample count and weighting; includes payload, latency, AoI, radio, rates, queues and logged reward |
| [METRICS.json](data/METRICS.json) | Session summaries | Counts, quality bounds, pooled object recall, segmentation IoU, precision, payload and latency statistics |
| [PUSCH.csv](data/PUSCH.csv) | Per-transmission gNB observations | Measured `PUSCH_SNR_dB`, `PUSCH_MCS`, RNTI, telemetry-ingest elapsed time |
| [CHANNEL_COMMANDS.csv](data/CHANNEL_COMMANDS.csv) | Acknowledged channel commands | `target_snr_db`; commanded proxy is distinct from measured PUSCH SNR |
| [TRANSPORT_RATES_1S.csv](data/TRANSPORT_RATES_1S.csv) | Complete 1-second bins | Application offered, PDCP ingress, RLC ingress and RLC dequeue, in Mb/s |
| [BUFFER_OCCUPANCY_1S.csv](data/BUFFER_OCCUPANCY_1S.csv) | Complete 1-second bins | Within-bin P50/P99 UE RLC occupancy and gNB estimated uplink buffer, in bytes; separate sample counts |
| [TRANSPORT_METRICS.json](data/TRANSPORT_METRICS.json) | Transport summaries | Whole-window mean rates, 1-second-rate quantiles, raw-event occupancy quantiles and trace coverage |
| [MAP_INSTALLS.csv](data/MAP_INSTALLS.csv) | Successful map updates | Capture-relative installation time, source capture time, capture-to-install age in ms |
| [AOI_SEGMENTS.csv](data/AOI_SEGMENTS.csv) | Intervals between map updates | `start_s`, `end_s`, `duration_s`, `age_start_s`; exact time-weighted map AoI |
| [FEEDBACK_REWARD_DECISIONS.csv](data/FEEDBACK_REWARD_DECISIONS.csv) | Every policy decision, including missing feedback | Wire quality/status, actual receive latency, packet identity and the locally logged reward resolution |
| [DOWNLINK_PACKETS.csv](data/DOWNLINK_PACKETS.csv) | Every captured feedback-port UDP packet | Kernel packet timestamp, IP/UDP byte sizes, message type, quality/status, exact payload hash and other control-message JSON |
| [DOWNLINK_RATES_1S.csv](data/DOWNLINK_RATES_1S.csv) | Complete 1-second bins | Feedback-port traffic and R4FB-only traffic separately, in Mb/s; packet counts |
| [DOWNLINK_SUMMARY.json](data/DOWNLINK_SUMMARY.json) | Downlink and reward summaries | Packet-match checks, missing receipts, reward terminal counts, latency and rate quantiles |

CSV blanks and JSON `null` mean unavailable or undefined, never an assumed zero. The exceptions are genuine measured zeros and explicitly specified timeout rewards. Join by **arm plus frame ID**, not frame ID alone. For controller identities, use arm, session UUID and decision sequence together.

### Figure guide

All 13 figures are PNGs at 3840 × 2160. [CAPTIONS.json](data/CAPTIONS.json) and [TRANSPORT_CAPTIONS.json](data/TRANSPORT_CAPTIONS.json) provide metric-specific captions.

| Figure prefix | Content |
|---|---|
| 01 | Prediction completion and feedback receipt curves |
| 02 | Timely, late and missing outcomes; conditional latency P50/P99 |
| 03 | Application payload P50/P99 bars only |
| 04 | Measured PUSCH SNR and MCS timelines |
| 05 | Prediction and UE feedback latency timelines |
| 06 | Deadline-aware recall and segmentation IoU; timely-prediction precision |
| 07 | Agent only: SI, P40, q, payload and measured PUSCH SNR aligned in time |
| 08 | Capture-to-install age, time-weighted map AoI CDF and timeline |
| 09 | Available per-frame quality measurements over time |
| 10 | Map installation age versus AoI P50/P99 |
| 11 | Application offered load and PDCP/RLC flow rates |
| 12 | UE RLC occupancy and separately labelled gNB uplink-buffer estimate |
| 13 | Absolute payload timelines, all three configurations on one linear kB axis |

## Uplink traffic definitions

Application payload is the recorded SFD4 application-frame byte count, not the native camera/radar input and not radio air-interface overhead. Decimal units are used: 1 kB = 1,000 bytes; 1 MB = 1,000,000 bytes; 1 Mb/s = 1,000,000 bits per second.

Application offered rate is the sum of frame payload bytes whose **capture times** fall in a complete one-second bin, multiplied by 8/1,000,000. It is not a packet-send-timestamp or receiver-goodput measurement. The raw per-frame and controller records retain send-stage timestamps for more detailed analysis.

PDCP ingress, RLC ingress and RLC dequeue come from recorded UE T-tracer events. They count bytes at different protocol boundaries; headers and buffering mean their totals need not match the application bytes. RLC dequeue is not successful end-to-end delivery. Rate quantiles use complete one-second bins; the final partial second is excluded.

RLC occupancy is a **byte count**, not a rate. The decoder sums distinct logical channel IDs at each completed UE tick; the final unclosed tick is excluded. The gNB series is a scheduler estimate. **Direct BSR reports were not recorded**, so BSR values or report frequency cannot be recovered from this package. The gNB estimate must not be labelled as BSR.

Compressed full UE T-tracer recordings and their embedded event schemas are in `raw/<arm>/ue.raw.gz`. These allow alternative time bins without returning to the large original capture. The gNB telemetry files are also included in compressed form.

## Downlink feedback and the reward

The downlink message and the locally resolved reward are different records:

1. The edge sends **R4FB quality/status feedback** over OAI to the UE. Its binary payload contains identity, action, status and Q_perc when defined, **not a precomputed reward scalar**.
2. The UE receives the feedback and resolves its deadline-dependent reward. The received quality and the resolved reward are both exported in `FEEDBACK_REWARD_DECISIONS.csv`.

Every recorded R4FB receipt matches an exact SHA-256 UDP payload in the UE-tunnel PCAP. All R4FB messages here are **176 application bytes**. The IPv4 datagram is 204 bytes including UDP/IP headers. Neither is an over-the-air byte count.

| Configuration | Captured R4FB packets | Other control packets on the same port | Decisions without a recorded R4FB receipt |
|---|---:|---:|---:|
| Agent | 493 | 0 | 0 |
| AE128 / 8-bit | 4 | 84 | 496 |
| AE32 / 8-bit | 476 | 47 | 24 |

Other control packets are preserved and labelled separately, not treated as quality feedback or reward. All three tcpdump logs report zero packets dropped by the kernel. These captures cover the feedback port, **not all downlink traffic, map ACK traffic or total downlink capacity**. Packet counts outside complete one-second rate bins are reported in `DOWNLINK_SUMMARY.json`; those packets remain in `DOWNLINK_PACKETS.csv`.

### Reward fields

`wire_q_perc` is the received perception-quality value. `logged_reward` is copied from the controller's actual resolution; it is not recomputed as a new experimental result. The recorded `reward_schema_id` is **splitfusion_run4_reward_v1**:

- Timely success: `Q_perc - 0.25 × (feedback_latency_ms / 170)`.
- Timeout: `-1`, even if a successful quality message subsequently arrives late.
- Evaluator fault: no reward; the exported reward is blank and `learning_included_flag` is false.

This retained runtime reward-logging schema is **not the different timely-localisation objective used to train the frozen Run5 actor**. The actor was not updated online. `learning_included_flag` is the recorded contract flag, not evidence of live training. The relevant source definitions are included under `source_reference/` for inspection.

`feedback_kind` and `controller_terminal` may differ legitimately: a wire `DELIVERED_SUCCESS` can end as controller `TIMEOUT`. Similarly, a timely evaluator-fault reply counts as a timely feedback receipt, not as a successful scored perception result. For the agent, 390 timely replies comprise 363 controller successes and 27 timely evaluator faults.

## Timing and quality definitions

- `prediction_ms` and `feedback_ms` use **action-open to the corresponding milestone**, in CLOCK_MONOTONIC_RAW. The 170-ms feedback deadline begins at action-open, not at camera capture. Feedback latency includes processing and uplink as well as feedback delivery; it is **not pure downlink latency**.
- Packet timestamps use CLOCK_REALTIME at the UE tunnel. Radio `elapsed_s` uses telemetry-ingest wall time. Do not subtract a wall-clock timestamp from a raw-monotonic timestamp without a validated clock conversion. This package does not infer one-way downlink latency from incompatible clocks.
- Map age at installation is **installation time minus frame capture time**. Map AoI at any later time is current time minus the capture time of the latest installed frame. It does not reset to zero on installation. AoI quantiles are time-weighted after the first installation until the last transmitted capture; initial no-map time is reported separately.
- Latency percentiles condition on recorded events. Missing predictions/receipts have no imputed latency. AE128's latency percentiles describe only four observations and are not representative full-session latency.
- Figure 06 uses **prediction ready within 170 ms**, not feedback receipt, as the quality boundary. Recall counts eligible objects, with tolerances of 1.0 m for vehicles and 1.2 m for pedestrians. Timely precision is conditional on timely predictions under the existing 3 m matching gate.
- `METRICS.json` preserves missing-GT bounds and unavailable segmentation measurements. AE32 deadline-aware recall/IoU are upper bounds where GT is missing. AE128 has no timely predictions; undefined precision and unmeasured pedestrian segmentation union are not invented as zero.
- Per-frame recall quantiles in the new quantile CSV are explicitly labelled; they are **not pooled object recall**. Use the pooled numerators/denominators and IoU in `METRICS.json` for the campaign comparison.
- NumPy linear percentiles are used for sample-weighted statistics. MCS percentiles are descriptive numeric quantiles of discrete indices. Map AoI is calculated from its continuous linear-age intervals instead.

## Recreate or change the plots

From this folder on the Ubuntu machine:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/replot.py
```

This creates `plots_regenerated/` without changing `figures/` or `data/`. All 13 generated PNGs have been checked byte-for-byte against the shared figures using the recorded environment. Edit the local plotting functions in `scripts/prepare_plots.py` or `scripts/transport_plots.py`, then rerun **replot.py** to try another presentation. Do not execute those two historical extractor scripts directly: their original extraction entry points reference the experiment workspace.

Python, NumPy and Matplotlib versions are in [ENVIRONMENT.json](ENVIRONMENT.json). No torch, GPU, model execution or network service is needed. Different plotting-library/font versions can change raster bytes without changing data.

Optional data regeneration from the included records:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/export_feedback.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/quantile_table.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B scripts/rebin_transport.py
```

The outputs go to `feedback_regenerated/`, `quantiles_regenerated.csv` and `transport_regenerated/`, respectively. The transport script temporarily unpacks roughly 0.4 GB of local trace copies; it removes only its own temporary directory on completion. It reproduces the supplied one-second transport tables; its copied decoder can be adapted for a different binning policy.

The `raw/` folder contains the phase evidence, actual reward resolutions, exact feedback PCAPs, frame/map metrics, edge evidence and compressed GT-service records for all three arms. It excludes camera images, tensors, model checkpoints and unrelated experiments. The full 13-arm campaign report under `provenance/campaign/` supports the historical metrics; only the three requested arms are plotted here.

## Provenance and verification

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) maps every copied file to its Ubuntu source path and digest. Compressed copies retain the original uncompressed hash as well. Those absolute paths are for provenance only; the replotting commands above use this folder's copies.

[VALIDATION.json](VALIDATION.json) records reproduction, packet matching and preservation checks. [FILE_SHA256.json](FILE_SHA256.json) lists the packaged file hashes. To check the original package after copying it:

```bash
python3 -B scripts/verify.py
```

The check covers the listed files and allows new user-generated outputs. If you intentionally edit a listed plotting script, its hash will no longer match; that does not change the immutable measurement copies. No original capture, historical verdict, production source or architecture-handoff file was changed to prepare this folder.
